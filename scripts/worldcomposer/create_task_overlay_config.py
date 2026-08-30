#!/usr/bin/env python3
"""Create a separate native-NuRec task overlay configuration from calibration data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "configs/simbox/safety_scenarios_3dgs_usdz_native_manual_calibration_isaac50_visual.yaml"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # The runtime loader accepts absolute paths as well. Keeping this
        # fallback makes it possible to validate a new bundle in /tmp before
        # copying it into the repository asset pool.
        return str(path.resolve())


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--enable-nurec-proxy",
        action="store_true",
        help="bind the MESH as an RTX depth proxy only after proxy validation",
    )
    args = parser.parse_args()

    manifest_path = args.bundle_manifest.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    manifest = _load_json(manifest_path)
    calibration = _load_json(calibration_path)
    if manifest.get("status") != "ready_for_calibration":
        raise ValueError(f"bundle is not usable: {manifest.get('status')}")
    transform = calibration.get("calibration_transform")
    if not isinstance(transform, dict):
        raise ValueError("calibration JSON does not contain calibration_transform")
    translation = list(transform.get("translation", []))
    euler = list(transform.get("euler_xyz_deg", []))
    scale = list(transform.get("scale", []))
    if any(len(value) != 3 for value in (translation, euler, scale)):
        raise ValueError("calibration translation, euler_xyz_deg and scale must each have three values")
    task_root_translation = list(calibration.get("task_root_translation", [0.0, 0.0, 0.0]))
    if len(task_root_translation) != 3:
        raise ValueError("task_root_translation must have three values when present")
    if not template_path.is_file():
        raise FileNotFoundError(template_path)

    background_mode = str(manifest.get("background_mode", "3dgs_mesh"))
    if background_mode not in {"3dgs_only", "3dgs_mesh"}:
        raise ValueError(f"unsupported background_mode: {background_mode}")
    if background_mode == "3dgs_only" and args.enable_nurec_proxy:
        raise ValueError("a pure-3DGS bundle has no MESH and cannot enable a NuRec proxy")
    bundle_dir = manifest_path.parent
    nurec = bundle_dir / str(manifest["nurec_asset"])
    mesh = bundle_dir / str(manifest["mesh_asset"]) if background_mode == "3dgs_mesh" else None
    if not nurec.is_file() or (mesh is not None and not mesh.is_file()):
        raise FileNotFoundError("bundle manifest references a missing NuRec or MESH asset")
    with template_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"template must be a YAML mapping: {template_path}")
    config = copy.deepcopy(config)
    scene_id = str(manifest["scene_id"])
    config.setdefault("runtime", {})["generated_dir"] = f".generated/worldcomposer_{scene_id}_native"
    config["runtime"]["output_root"] = f"output/worldcomposer_{scene_id}_native"
    config["runtime"]["failure_output_root"] = f"failure_output/worldcomposer_{scene_id}_native"
    simulator = config["runtime"].setdefault("simulator_overrides", {})
    simulator["portable_root"] = f"/tmp/isaac50_worldcomposer_{scene_id}_native"
    # The static calibration stage keeps task_0 fixed and moves only the
    # reconstruction. Reapply its task-root translation at runtime rather
    # than retaining a scene-specific transform from the template.
    config["worldcomposer_task_alignment"] = {
        "translation": [float(value) for value in task_root_translation],
    }
    config["worldcomposer_background"] = {
        "enabled": True,
        "name": "native_nurec_background",
        "asset_path": _repo_relative(nurec),
        "translation": [float(value) for value in translation],
        "euler": [float(value) for value in euler],
        "scale": [float(value) for value in scale],
    }
    if mesh is None:
        # The template may belong to an older MESH-backed experiment. Remove
        # that layer entirely so the generated config cannot load it by mistake.
        config.pop("worldcomposer_mesh", None)
    else:
        config["worldcomposer_mesh"] = {
            "enabled": True,
            "name": "native_nurec_mesh",
            "asset_path": _repo_relative(mesh),
            "translation": [float(value) for value in translation],
            "euler": [float(value) for value in euler],
            "scale": [float(value) for value in scale],
            "visible": False,
            "disable_collision": True,
            "semantic_label": False,
            "nurec_proxy": bool(args.enable_nurec_proxy),
            "nurec_proxy_for": "native_nurec_background",
            "nurec_proxy_meshes": [],
        }
    config.setdefault("worldcomposer_bundle", {})
    config["worldcomposer_bundle"] = {
        "scene_id": scene_id,
        "manifest": _repo_relative(manifest_path),
        "calibration": _repo_relative(calibration_path),
        "template": _repo_relative(template_path),
        "background_mode": background_mode,
        "mesh_is_visual_only": mesh is not None and not args.enable_nurec_proxy,
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=False)
    print(output)
    print(f"[worldcomposer] STATUS=SUCCESS step=create_task_overlay_config output={output}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(
                f"[worldcomposer] STATUS=FAILED step=create_task_overlay_config exit_code={exc.code}",
                file=sys.stderr,
            )
        raise
    except Exception:
        print("[worldcomposer] STATUS=FAILED step=create_task_overlay_config", file=sys.stderr)
        raise
