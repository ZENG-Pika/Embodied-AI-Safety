"""Standalone LeRobot Diffusion Policy inference server.

The server runs in the model's Python environment. Isaac Sim communicates
through length-prefixed pickle messages so its Torch installation is untouched.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
from pathlib import Path


def _read_message(stream):
    header = stream.buffer.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise EOFError("truncated policy request header")
    size = struct.unpack("!I", header)[0]
    payload = stream.buffer.read(size)
    if len(payload) != size:
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
            actions = policy.predict(request["image"], request["state"])
            _write_message(sys.stdout, {"ok": True, "actions": actions})
        except Exception as exc:
            _write_message(
                sys.stdout,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )


if __name__ == "__main__":
    raise SystemExit(main())
