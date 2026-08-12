"""Isaac-Sim-side client for a standalone LeRobot Diffusion Policy process."""

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _resize_rgb(image: np.ndarray, height: int = 360, width: int = 640) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
    image = image[..., :3]
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    try:
        import cv2
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        ys = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        resized = image[np.ix_(ys, xs)]
    return np.ascontiguousarray(resized)


class TrainedDPClient:
    def __init__(
        self,
        *,
        python_executable: str | Path,
        checkpoint: str | Path,
        model_root: str | Path,
        device: str = "cuda",
        seed: int | None = None,
        replan_steps: int = 8,
    ) -> None:
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = int(replan_steps)
        self._chunk = np.empty((0, 8), dtype=np.float32)
        self._steps_since_plan = 0
        # Isaac Sim exports Python 3.10 variables into its process. Remove
        # them before starting the independent Python 3.12 policy service;
        # otherwise the child interpreter can import Isaac's stdlib and fail
        # with an ``SRE module mismatch`` assertion.
        child_env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONSTARTUP"):
            child_env.pop(key, None)
        child_env.pop("PYTHONNOUSERSITE", None)
        self._process = subprocess.Popen(
            [
                str(python_executable),
                "-u",
                str(Path(__file__).with_name("trained_dp_server.py")),
                "--checkpoint",
                str(Path(checkpoint).resolve()),
                "--model-root",
                str(Path(model_root).resolve()),
                "--device",
                str(device),
                *([] if seed is None else ["--seed", str(seed)]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=child_env,
            bufsize=0,
        )
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = str(device)
        self.seed = seed

    def _read_exact(self, size: int) -> bytes | None:
        if self._process.stdout is None:
            return None
        chunks = []
        remaining = int(size)
        while remaining:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(f"trained DP process exited with code {self._process.returncode}")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("trained DP process pipes are unavailable")
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self._process.stdin.write(struct.pack("!I", len(payload)))
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        header = self._read_exact(4)
        if header is None:
            raise RuntimeError("trained DP process closed before replying")
        size = struct.unpack("!I", header)[0]
        if size > _MAX_MESSAGE_BYTES:
            raise RuntimeError(
                "invalid trained DP response size "
                f"{size} bytes; policy stdout may contain non-protocol output"
            )
        response = self._read_exact(size)
        if response is None:
            raise RuntimeError("truncated response from trained DP process")
        result = pickle.loads(response)
        if not result.get("ok", False):
            raise RuntimeError(result.get("error", "trained DP inference failed"))
        return result

    def reset(self) -> None:
        self._request({"command": "reset"})
        self._chunk = np.empty((0, 8), dtype=np.float32)
        self._steps_since_plan = 0

    def step(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"expected Franka policy state shape (8,), got {state.shape}")
        if self._chunk.size == 0 or self._steps_since_plan >= self.replan_steps:
            response = self._request(
                {"command": "predict", "image": _resize_rgb(image), "state": state}
            )
            self._chunk = np.asarray(response["actions"], dtype=np.float32)
            if self._chunk.shape != (32, 8) or not np.isfinite(self._chunk).all():
                raise RuntimeError(f"invalid trained DP output shape={self._chunk.shape}")
            self._steps_since_plan = 0
        action = self._chunk[0].copy()
        self._chunk = self._chunk[1:]
        self._steps_since_plan += 1
        return action

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "diffusion_policy",
            "policy_name": "franka_dp_100k_delivery",
            "initialization": "checkpoint",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "seed": self.seed,
            "observation_schema": {
                "image_key": "observation.images.head",
                "image_shape": [360, 640, 3],
                "state_key": "observation.state",
                "state_dim": 8,
                "state_order": [
                    "joint_0", "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6", "gripper_width_m",
                ],
            },
            "action_schema": {
                "action_shape": [32, 8],
                "action_type": "absolute_joint_position",
                "gripper_action": "width_m",
                "replan_steps": self.replan_steps,
            },
        }

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=5)
