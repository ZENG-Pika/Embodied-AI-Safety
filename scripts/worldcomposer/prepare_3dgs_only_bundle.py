#!/usr/bin/env python3
"""Create a portable, pure-3DGS NuRec calibration bundle.

The bundle contains exactly one visual asset, a WorldComposer NuRec ``.usdz``.
It deliberately does not extract, reference, or configure a MESH/proxy layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError("scene id must contain only letters, digits, '_' or '-'")
    return value


def _stage() -> str:
    return '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    # This is the only transform changed during manual calibration.
    def Xform "BackgroundCalibration"
    {
        double3 xformOp:translate = (0, 0, 0)
        double3 xformOp:rotateXYZ = (0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        def Xform "gauss" (
            prepend references = @./3DGS.usdz@
        )
        {
        }
    }
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", type=_scene_id, required=True)
    parser.add_argument("--nurec", type=Path, required=True, help="WorldComposer-exported 3DGS.usdz")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "assets/worldcomposer")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing scene-id bundle")
    args = parser.parse_args()

    nurec = args.nurec.expanduser().resolve()
    if not nurec.is_file() or nurec.suffix.lower() != ".usdz":
        parser.error(f"--nurec must be an existing .usdz file: {nurec}")
    bundle_dir = args.output_root.expanduser().resolve() / args.scene_id
    if bundle_dir.exists():
        if not args.overwrite:
            parser.error(f"bundle already exists: {bundle_dir}; pass --overwrite to replace it")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    copied_asset = bundle_dir / "3DGS.usdz"
    shutil.copy2(nurec, copied_asset)

    stage_path = bundle_dir / f"{args.scene_id}_3DGS.usda"
    stage_path.write_text(_stage(), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scene_id": args.scene_id,
        "bundle_dir": str(bundle_dir),
        "background_mode": "3dgs_only",
        "nurec_asset": "3DGS.usdz",
        "nurec_sha256": _sha256(copied_asset),
        "calibration_stage": stage_path.name,
        "calibration_parent_prim": "/World/BackgroundCalibration",
        "mesh_present": False,
        "proxy_present": False,
        "status": "ready_for_calibration",
        "calibration_transform": {
            "translation": [0.0, 0.0, 0.0],
            "euler_xyz_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"[worldcomposer] STATUS=SUCCESS step=prepare_3dgs_only_bundle bundle_dir={bundle_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"[worldcomposer] STATUS=FAILED step=prepare_3dgs_only_bundle exit_code={exc.code}", file=sys.stderr)
        raise
    except Exception:
        print("[worldcomposer] STATUS=FAILED step=prepare_3dgs_only_bundle", file=sys.stderr)
        raise
