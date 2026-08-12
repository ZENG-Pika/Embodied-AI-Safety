"""Small, simulator-independent report I/O helpers."""

import json
import os
from pathlib import Path


def atomic_write_json(path, payload, *, pretty: bool = False) -> None:
    """Write JSON atomically, optionally using human-readable indentation."""
    final_path = Path(path)
    temporary_path = final_path.with_name(final_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            dump_options = {
                "ensure_ascii": False,
                "default": str,
            }
            if pretty:
                dump_options.update({"indent": 2})
            else:
                dump_options["separators"] = (",", ":")
            json.dump(payload, stream, **dump_options)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
