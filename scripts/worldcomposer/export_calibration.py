#!/usr/bin/env python3
"""Export the manual BackgroundCalibration transform from a saved USD stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _value_list(value):
    return [float(component) for component in value]


def _read_xform(stage, prim_path: str, *, required: bool, missing_op_warnings: bool = True) -> dict:
    """Return the authored TRS values of an Xform prim.

    The task root is optional so this exporter remains usable with the
    background-only calibration stages created by older bundles.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        if required:
            raise RuntimeError(f"prim does not exist: {prim_path}")
        return {
            "translation": [0.0, 0.0, 0.0],
            "euler_xyz_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "warnings": [f"optional prim absent; exported defaults: {prim_path}"],
        }

    xformable = UsdGeom.Xformable(prim)
    values = {
        "translation": None,
        "euler_xyz_deg": None,
        "scale": None,
        "warnings": [],
    }
    for operation in xformable.GetOrderedXformOps():
        op_type = operation.GetOpType()
        op_name = operation.GetOpName()
        value = operation.Get()
        if op_type == UsdGeom.XformOp.TypeTranslate or op_name == "xformOp:translate":
            values["translation"] = _value_list(value)
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ or op_name == "xformOp:rotateXYZ":
            values["euler_xyz_deg"] = _value_list(value)
        elif op_type == UsdGeom.XformOp.TypeScale or op_name == "xformOp:scale":
            values["scale"] = _value_list(value)
        else:
            values["warnings"].append(f"unsupported xform op retained in stage: {operation.GetOpName()}")

    for key, fallback in (
        ("translation", [0.0, 0.0, 0.0]),
        ("euler_xyz_deg", [0.0, 0.0, 0.0]),
        ("scale", [1.0, 1.0, 1.0]),
    ):
        if values[key] is None:
            values[key] = fallback
            if missing_op_warnings:
                values["warnings"].append(f"{key} op was absent; exported default {fallback}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-prim", default="/World/BackgroundCalibration")
    parser.add_argument("--task-root-prim", default="/World/task_0")
    parser.add_argument(
        "--require-uniform-scale",
        action="store_true",
        help="reject non-uniform scale and export the reproducible seven background parameters",
    )
    args = parser.parse_args()
    stage_path = args.stage.expanduser().resolve()
    if not stage_path.is_file():
        parser.error(f"stage does not exist: {stage_path}")

    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not open stage: {stage_path}")
    values = _read_xform(stage, args.background_prim, required=True)
    task_root = _read_xform(stage, args.task_root_prim, required=False, missing_op_warnings=False)

    parameters_7d = None
    if args.require_uniform_scale:
        scale = values["scale"]
        if max(scale) - min(scale) > 1e-6:
            raise ValueError(
                "BackgroundCalibration has non-uniform scale; set Scale X/Y/Z to the same value "
                "before exporting a seven-parameter calibration"
            )
        translation = values["translation"]
        euler = values["euler_xyz_deg"]
        parameters_7d = {
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "roll_x_deg": euler[0],
            "pitch_y_deg": euler[1],
            "yaw_z_deg": euler[2],
            "scale": scale[0],
        }

    result = {
        "stage": str(stage_path),
        "background_prim": args.background_prim,
        "calibration_transform": {
            "translation": values["translation"],
            "euler_xyz_deg": values["euler_xyz_deg"],
            "scale": values["scale"],
        },
        # The generated runtime config uses this exact task-root offset. This
        # prevents an old scene-specific alignment from shifting the task
        # assets away from the reference frame used during manual calibration.
        "task_root_prim": args.task_root_prim,
        "task_root_translation": task_root["translation"],
        "background_parameters_7d": parameters_7d,
        "warnings": values["warnings"] + task_root["warnings"],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[worldcomposer] STATUS=SUCCESS step=export_calibration output={output}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"[worldcomposer] STATUS=FAILED step=export_calibration exit_code={exc.code}", file=sys.stderr)
        raise
    except Exception:
        print("[worldcomposer] STATUS=FAILED step=export_calibration", file=sys.stderr)
        raise
