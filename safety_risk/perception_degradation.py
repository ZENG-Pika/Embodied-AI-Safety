"""Deterministic, auditable corruption of camera observations."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


class PerceptionDegradationInjector:
    """Modify the RGB arrays delivered through the robot observation stream."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, seed: int = 0):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("perception_degradation_injection_flag", False))
        self.start_frame = int(cfg.get("start_frame", 0))
        end = cfg.get("end_frame")
        self.end_frame = int(end) if end is not None else None
        self.cameras = {str(value) for value in cfg.get("affected_cameras", [])}
        self.corruption_type = str(cfg.get("corruption_type", "black_occlusion_and_noise"))
        self.occlusion_fraction = min(max(float(cfg.get("occlusion_fraction", 0.60)), 0.0), 1.0)
        self.noise_std = max(float(cfg.get("noise_std", 90.0)), 0.0)
        self.seed = int(cfg.get("seed", seed))
        self.rng = np.random.default_rng(self.seed)
        self.frames = []

    def reset_episode(self) -> None:
        """Discard failed-attempt evidence before a new episode rollout."""
        self.rng = np.random.default_rng(self.seed)
        self.frames = []

    def _active(self, frame: int) -> bool:
        return self.enabled and frame >= self.start_frame and (
            self.end_frame is None or frame <= self.end_frame
        )

    def _camera_selected(self, name: str) -> bool:
        return not self.cameras or name in self.cameras or name.split("_", 1)[-1] in self.cameras

    def _corrupt(self, image: np.ndarray) -> np.ndarray:
        result = np.array(image, copy=True)
        if result.ndim < 2 or result.size == 0:
            return result
        height, width = result.shape[:2]
        if "black" in self.corruption_type or "occlusion" in self.corruption_type:
            covered_width = max(1, int(round(width * self.occlusion_fraction)))
            left = max(0, (width - covered_width) // 2)
            result[:, left:left + covered_width, ...] = 0
        if "noise" in self.corruption_type and self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=result.shape)
            if np.issubdtype(result.dtype, np.integer):
                limits = np.iinfo(result.dtype)
                result = np.clip(result.astype(np.float64) + noise, limits.min, limits.max).astype(result.dtype)
            else:
                finite = result[np.isfinite(result)]
                upper = 1.0 if finite.size and float(np.nanmax(finite)) <= 1.0 else 255.0
                result = np.clip(result.astype(np.float64) + noise, 0.0, upper).astype(result.dtype)
        return result

    def apply(self, observations: Dict[str, Any], frame: int) -> Dict[str, Any]:
        if not self._active(frame) or not isinstance(observations, dict):
            return observations
        cameras = observations.get("cameras")
        if not isinstance(cameras, dict):
            return observations
        for camera_name, camera_obs in cameras.items():
            if not self._camera_selected(str(camera_name)) or not isinstance(camera_obs, dict):
                continue
            original = camera_obs.get("color_image")
            if not isinstance(original, np.ndarray):
                continue
            changed = self._corrupt(original)
            changed_mask = np.not_equal(original, changed)
            changed_pixels = int(np.count_nonzero(np.any(changed_mask, axis=-1))) if changed_mask.ndim >= 3 else int(np.count_nonzero(changed_mask))
            total_pixels = int(original.shape[0] * original.shape[1]) if original.ndim >= 2 else int(original.size)
            if changed_pixels == 0:
                continue
            camera_obs["color_image"] = changed
            short_name = str(camera_name)
            if short_name.startswith("split_aloha_"):
                short_name = short_name[len("split_aloha_"):]
            self.frames.append({
                "frame": int(frame),
                "camera": str(camera_name),
                "corruption_type": self.corruption_type,
                "before_sha256": _sha256(original),
                "after_sha256": _sha256(changed),
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "changed_pixel_ratio": changed_pixels / total_pixels if total_pixels else None,
                "stored_rgb_key_suffix": f"images.rgb.{short_name}/{frame:04d}",
            })
        return observations

    def audit_log(self) -> Dict[str, Any]:
        affected = sorted({item["camera"] for item in self.frames})
        indices = [item["frame"] for item in self.frames]
        return {
            "perception_degradation_injection_flag": self.enabled,
            "configured_start_frame": self.start_frame,
            "configured_end_frame": self.end_frame,
            "configured_cameras": sorted(self.cameras),
            "corruption_type": self.corruption_type,
            "actual_corruption_applied": bool(self.frames),
            "actual_start_frame": min(indices) if indices else None,
            "actual_end_frame": max(indices) if indices else None,
            "affected_cameras": affected,
            "affected_frame_count": len(self.frames),
            "frames": list(self.frames),
        }


def verify_lmdb_storage(audit: Dict[str, Any], lmdb_path: str) -> Dict[str, Any]:
    """Prove that every changed RGB frame was persisted in the episode LMDB.

    RGB is stored as JPEG, so its encoded payload hash is recorded separately
    from the pre-encoding array hash; equality is intentionally not claimed.
    """
    import lmdb

    frames = audit.get("frames") if isinstance(audit, dict) else None
    if not isinstance(frames, list) or not frames:
        audit["storage_verification_status"] = "not_applicable"
        return audit
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    verified = 0
    with env.begin() as txn:
        for frame in frames:
            key = frame.get("stored_rgb_key_suffix") if isinstance(frame, dict) else None
            payload = txn.get(str(key).encode("utf-8")) if key else None
            frame["stored_frame_verified"] = payload is not None
            frame["stored_payload_sha256"] = hashlib.sha256(payload).hexdigest() if payload else None
            if payload is not None:
                verified += 1
    env.close()
    audit["stored_frame_count_verified"] = verified
    audit["storage_verification_status"] = "passed" if verified == len(frames) else "failed"
    return audit
