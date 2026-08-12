"""Standalone LeRobot Diffusion Policy inference server.

The server runs in the model's Python environment. Isaac Sim communicates
through length-prefixed pickle messages so its Torch installation is untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import pickle
import struct
import sys
from pathlib import Path


_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _read_exact(stream, size: int):
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = stream.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(stream):
    header = _read_exact(stream, 4)
    if header is None:
        return None
    size = struct.unpack("!I", header)[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"policy request is too large: {size} bytes")
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("truncated policy request")
    return pickle.loads(payload)


def _write_message(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.buffer.write(struct.pack("!I", len(payload)))
    stream.buffer.write(payload)
    stream.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from inference import DPInference

    # stdout is the binary protocol channel. Some libraries print progress while
    # loading or running inference, so keep all ordinary output on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        policy = DPInference(args.checkpoint, device=args.device, seed=args.seed)
    print(
        "[trained_dp] loaded "
        f"checkpoint={args.checkpoint} device={args.device}",
        file=sys.stderr,
        flush=True,
    )
    while True:
        request = _read_message(sys.stdin)
        if request is None:
            return 0
        try:
            command = request.get("command", "predict")
            if command == "reset":
                policy.reset()
                _write_message(sys.stdout, {"ok": True})
                continue
            if command != "predict":
                raise ValueError(f"unsupported policy command: {command}")
            with contextlib.redirect_stdout(sys.stderr):
                actions = policy.predict(request["image"], request["state"])
            # Do not pickle a NumPy array across the Python environments. Newer
            # NumPy records ``numpy._core`` module paths that Isaac's older
            # NumPy cannot import. The action chunk is small, so a list is cheap
            # and version-independent; the client converts it back to ndarray.
            _write_message(sys.stdout, {"ok": True, "actions": actions.tolist()})
        except Exception as exc:
            _write_message(
                sys.stdout,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )


if __name__ == "__main__":
    raise SystemExit(main())
