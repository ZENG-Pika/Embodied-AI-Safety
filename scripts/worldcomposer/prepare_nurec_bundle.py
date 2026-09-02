#!/usr/bin/env python3
"""Create a portable NuRec 3DGS + MESH calibration bundle.

The script accepts an already exported WorldComposer NuRec ``.usdz`` and a
MESH archive.  It never changes either input.  The resulting ``Fused.usda``
uses only relative references, keeps the MESH hidden, and exposes one
``/World/BackgroundCalibration`` transform for manual calibration in Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "scene id must contain only letters, digits, '_' or '-'"
        )
    return value


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe archive member: {info.filename}")
        target = (destination / Path(*member.parts)).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"archive member escapes destination: {info.filename}")
    archive.extractall(destination)


def _usd_candidates(mesh_dir: Path) -> list[Path]:
    return sorted(
        (path for path in mesh_dir.rglob("*") if path.is_file() and path.suffix.lower() in USD_SUFFIXES),
        key=lambda path: (
            path.suffix.lower() != ".usd",
            -path.stat().st_size,
            path.as_posix(),
        ),
    )


def _fused_stage(mesh_reference: str) -> str:
    return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "BackgroundCalibration"
    {{
        double3 xformOp:translate = (0, 0, 0)
        double3 xformOp:rotateXYZ = (0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        def Xform "gauss" (
            prepend references = @./3DGS.usdz@
        )
        {{
        }}

        def Xform "mesh" (
            prepend references = @./{mesh_reference}@
        )
        {{
            token visibility = "invisible"
        }}
    }}
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", type=_scene_id, required=True)
    parser.add_argument("--nurec", type=Path, required=True, help="WorldComposer-exported 3DGS.usdz")
    parser.add_argument("--mesh-zip", type=Path, required=True, help="archive containing a converted MESH USD")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "assets/worldcomposer",
        help="directory that receives one <scene-id> bundle",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing scene-id bundle")
    args = parser.parse_args()

    nurec = args.nurec.expanduser().resolve()
    mesh_zip = args.mesh_zip.expanduser().resolve()
    if not nurec.is_file() or nurec.suffix.lower() != ".usdz":
        parser.error(f"--nurec must be an existing .usdz file: {nurec}")
    if not mesh_zip.is_file() or mesh_zip.suffix.lower() != ".zip":
        parser.error(f"--mesh-zip must be an existing .zip file: {mesh_zip}")

    output_root = args.output_root.expanduser().resolve()
    bundle_dir = output_root / args.scene_id
    if bundle_dir.exists():
        if not args.overwrite:
            parser.error(f"bundle already exists: {bundle_dir}; pass --overwrite to replace it")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    shutil.copy2(nurec, bundle_dir / "3DGS.usdz")
    mesh_dir = bundle_dir / "mesh"
    mesh_dir.mkdir()
    try:
        with zipfile.ZipFile(mesh_zip) as archive:
            _safe_extract(archive, mesh_dir)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        parser.error(f"could not extract MESH archive: {exc}")

    candidates = _usd_candidates(mesh_dir)
    manifest = {
        "scene_id": args.scene_id,
        "bundle_dir": str(bundle_dir),
        "nurec_asset": "3DGS.usdz",
        "nurec_sha256": _sha256(bundle_dir / "3DGS.usdz"),
        "mesh_archive": str(mesh_zip),
        "mesh_archive_sha256": _sha256(mesh_zip),
        "mesh_usd_candidates": [str(path.relative_to(bundle_dir)) for path in candidates],
        "calibration_stage": f"{args.scene_id}_Fused.usda",
        "calibration_parent_prim": "/World/BackgroundCalibration",
        "mesh_visibility": "invisible",
        "mesh_collision": "disabled",
        "status": "ready_for_calibration" if candidates else "mesh_conversion_required",
    }
    if not candidates:
        manifest["next_step"] = (
            "The archive contains no USD mesh. Convert the mesh PLY with the "
            "WorldComposer scene-assembler flow, then rerun this command with "
            "the archive containing the generated *_mesh.usd."
        )
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), file=sys.stderr)
        raise SystemExit(2)

    mesh_path = candidates[0]
    mesh_reference = mesh_path.relative_to(bundle_dir).as_posix()
    stage_path = bundle_dir / manifest["calibration_stage"]
    stage_path.write_text(_fused_stage(mesh_reference), encoding="utf-8")
    manifest["mesh_asset"] = mesh_reference
    manifest["mesh_sha256"] = _sha256(mesh_path)
    manifest["calibration_transform"] = {
        "translation": [0.0, 0.0, 0.0],
        "euler_xyz_deg": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"[worldcomposer] STATUS=SUCCESS step=prepare_bundle bundle_dir={bundle_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"[worldcomposer] STATUS=FAILED step=prepare_bundle exit_code={exc.code}", file=sys.stderr)
        raise
    except Exception:
        print("[worldcomposer] STATUS=FAILED step=prepare_bundle", file=sys.stderr)
        raise
