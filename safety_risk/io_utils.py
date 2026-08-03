"""Small, simulator-independent report I/O helpers."""

import json
import os
from pathlib import Path


def atomic_write_json(path, payload) -> None:
    """Write compact JSON then atomically publish the completed file."""
    final_path = Path(path)
    temporary_path = final_path.with_name(final_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
