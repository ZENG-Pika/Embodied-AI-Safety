#!/usr/bin/env python3
"""Create a static task-asset calibration stage around a NuRec bundle.

This is deliberately not a task rollout.  It references the same table,
robot, target object, and optional MANO hand used by the selected task, then
places the uncalibrated NuRec (and, for legacy bundles, optional MESH) under
``/World/BackgroundCalibration``.
Move only that background parent in Isaac Sim while using task assets as the
fixed calibration reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = REPO_ROOT / "workflows/simbox/core/configs/tasks/pick_and_place/franka/single_pick/omniobject3d-dish.yaml"
DEFAULT_ASSET_ROOT = REPO_ROOT.parents[1] / "InternDataAssets/assets"
DEFAULT_HAND = REPO_ROOT / "workflows/simbox/example_assets/task/hand_model/mano_hand_fixed.usda"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _reference_path(source: Path, stage_dir: Path) -> str:
    return Path(os.path.relpath(source, stage_dir)).as_posix()


def _quat_from_euler_xyz(degrees: list[float]) -> list[float]:
    if len(degrees) != 3:
        raise ValueError("Euler angle list must have three values")
    rx, ry, rz = (math.radians(float(value)) / 2.0 for value in degrees)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # USD quaternion ordering is real, i, j, k.
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


def _triple(value: list[float], name: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    return [float(component) for component in value]


def _vec(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:.9g}" for value in values) + ")"


def _quat(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:.16g}" for value in values) + ")"


def _xform_block(indent: str, translation: list[float], scale: list[float], orientation: list[float] | None = None) -> str:
    lines = [
        f"{indent}double3 xformOp:translate = {_vec(translation)}",
    ]
    order = ['"xformOp:translate"']
    if orientation is not None:
        lines.append(f"{indent}quatd xformOp:orient = {_quat(orientation)}")
        order.append('"xformOp:orient"')
    lines.append(f"{indent}double3 xformOp:scale = {_vec(scale)}")
    order.append('"xformOp:scale"')
    lines.append(f"{indent}uniform token[] xformOpOrder = [{', '.join(order)}]")
    return "\n".join(lines)


def _calibration_xform_block(indent: str) -> str:
    """Author explicit editable XYZ Euler rotation for the background parent."""
    return "\n".join([
        f"{indent}double3 xformOp:translate = (0, 0, 0)",
        f"{indent}double3 xformOp:rotateXYZ = (0, 0, 0)",
        f"{indent}double3 xformOp:scale = (1, 1, 1)",
        f'{indent}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]',
    ])


def _region_centers(task: dict, robot_names: set[str]) -> dict[str, list[float]]:
    """Return deterministic XY reference points from A-on-B region ranges."""
    centers: dict[str, list[float]] = {}
    for region in task.get("regions") or []:
        if not isinstance(region, dict):
            continue
        name = str(region.get("object", ""))
        if name == "${tasks.0.robots.0.name}" and robot_names:
            name = next(iter(robot_names))
        config = region.get("random_config") or {}
        points = config.get("pos_range") or []
        if len(points) != 2 or any(not isinstance(point, list) or len(point) != 3 for point in points):
            continue
        centers[name] = [
            (float(points[0][axis]) + float(points[1][axis])) / 2.0
            for axis in range(3)
        ]
    return centers


def _robot_child(robot_cfg: dict) -> str | None:
    """Infer the referenced robot child that receives the task transform."""
    for key in ("fl_base_path", "fr_base_path"):
        value = str(robot_cfg.get(key, ""))
        if value:
            return value.split("/", 1)[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robot-xy", nargs=2, type=float, default=[0.0, -0.47])
    parser.add_argument("--robot-z", type=float, default=0.75162232)
    parser.add_argument("--object-xy", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--object-z", type=float, default=0.82)
    parser.add_argument("--hand-xy", nargs=2, type=float, default=[0.38, 0.05])
    parser.add_argument("--hand-z", type=float, default=0.78932703)
    parser.add_argument("--hand-scale", type=float, default=1.25)
    parser.add_argument("--no-hand", action="store_true")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"bundle manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_calibration":
        parser.error(f"bundle is not ready: {manifest.get('status')}")
    nurec = bundle_dir / str(manifest["nurec_asset"])
    mesh_asset = manifest.get("mesh_asset")
    mesh = bundle_dir / str(mesh_asset) if mesh_asset else None
    task_path = args.task.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    if not nurec.is_file() or (mesh is not None and not mesh.is_file()) or not task_path.is_file() or not asset_root.is_dir():
        parser.error("NuRec, optional MESH, task YAML, or asset root is missing")

    task_document = _load_yaml(task_path)
    tasks = task_document.get("tasks") or []
    if not tasks or not isinstance(tasks[0], dict):
        raise ValueError(f"task YAML contains no task: {task_path}")
    task = tasks[0]
    robots = [robot for robot in task.get("robots") or [] if isinstance(robot, dict)]
    objects = [obj for obj in task.get("objects") or [] if isinstance(obj, dict)]
    if not robots or not objects:
        raise ValueError("task must contain at least one robot and one object")
    arena = _load_yaml(REPO_ROOT / str(task["arena_file"]))
    fixtures = [fixture for fixture in arena.get("fixtures") or [] if isinstance(fixture, dict)]
    if not any(fixture.get("name") == "table" for fixture in fixtures):
        raise ValueError("arena contains no fixture named table")
    hand_path = DEFAULT_HAND
    if not args.no_hand and not hand_path.is_file():
        raise FileNotFoundError(f"MANO hand asset is missing: {hand_path}")

    scene_id = str(manifest["scene_id"])
    output = args.output.expanduser().resolve() if args.output else bundle_dir / f"{scene_id}_TaskCalibration.usda"
    output.parent.mkdir(parents=True, exist_ok=True)
    task_root = [0.0, 0.0, 0.0]
    hand_translation = [float(args.hand_xy[0]), float(args.hand_xy[1]), float(args.hand_z)]
    robot_names = {str(robot.get("name", "")) for robot in robots}
    centers = _region_centers(task, robot_names)
    skipped_assets: list[dict[str, str]] = []

    fixture_specs = []
    for fixture in fixtures:
        relative_path = str(fixture.get("path", ""))
        if not relative_path:
            continue
        path = asset_root / relative_path
        if not path.is_file():
            skipped_assets.append({"kind": "fixture", "name": str(fixture.get("name", "")), "reason": f"asset missing: {path}"})
            continue
        fixture_specs.append({
            "name": str(fixture.get("name", "fixture")),
            "path": path,
            "translation": _triple(list(fixture.get("translation", [0.0, 0.0, 0.0])), "fixture translation"),
            "scale": _triple(list(fixture.get("scale", [1.0, 1.0, 1.0])), "fixture scale"),
            "orientation": _quat_from_euler_xyz(_triple(list(fixture.get("euler", [0.0, 0.0, 0.0])), "fixture euler")),
        })

    robot_specs = []
    for index, robot in enumerate(robots):
        cfg_path = REPO_ROOT / str(robot.get("robot_config_file", ""))
        if not cfg_path.is_file():
            skipped_assets.append({"kind": "robot", "name": str(robot.get("name", "")), "reason": f"robot config missing: {cfg_path}"})
            continue
        robot_cfg = _load_yaml(cfg_path)
        path = asset_root / str(robot_cfg.get("path", ""))
        if not path.is_file():
            skipped_assets.append({"kind": "robot", "name": str(robot.get("name", "")), "reason": f"asset missing: {path}"})
            continue
        name = str(robot.get("name", f"robot_{index}"))
        center = centers.get(name, [float(args.robot_xy[0]), float(args.robot_xy[1]), 0.0])
        robot_specs.append({
            "name": name,
            "path": path,
            "child": _robot_child(robot_cfg),
            "translation": [float(center[0]), float(center[1]), float(args.robot_z)],
            "orientation": _quat_from_euler_xyz(_triple(list(robot.get("euler", [0.0, 0.0, 0.0])), "robot euler")),
        })

    object_specs = []
    for index, obj in enumerate(objects):
        relative_path = str(obj.get("path", ""))
        name = str(obj.get("name", f"object_{index}"))
        if not relative_path:
            skipped_assets.append({"kind": "object", "name": name, "reason": "task object has no direct USD path"})
            continue
        path = asset_root / relative_path
        if not path.is_file():
            skipped_assets.append({"kind": "object", "name": name, "reason": f"asset missing: {path}"})
            continue
        center = centers.get(name, [float(args.object_xy[0]), float(args.object_xy[1]), 0.0])
        object_specs.append({
            "name": name,
            "path": path,
            "child": str(obj.get("prim_path_child", "Aligned")),
            "translation": [float(center[0]), float(center[1]), float(args.object_z)],
            "orientation": _quat_from_euler_xyz(_triple(list(obj.get("euler", [0.0, 0.0, 0.0])), "object euler")),
            "scale": _triple(list(obj.get("scale", [1.0, 1.0, 1.0])), "object scale"),
        })
    if not fixture_specs or not robot_specs or not object_specs:
        details = "; ".join(f"{item['kind']}:{item['name']} ({item['reason']})" for item in skipped_assets)
        raise RuntimeError(f"selected task has insufficient directly referenceable assets: {details}")

    mesh_stage_block = ""
    if mesh is not None:
        mesh_stage_block = f'''

        def Xform "mesh" (
            prepend references = @{_reference_path(mesh, output.parent)}@
        )
        {{
            token visibility = "invisible"
        }}
'''

    stage = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    # Move only this parent while matching the reconstruction to task assets.
    def Xform "BackgroundCalibration"
    {{
{_calibration_xform_block("        ")}

        def Xform "gauss" (
            prepend references = @{_reference_path(nurec, output.parent)}@
        )
        {{
        }}
{mesh_stage_block}
    }}

    # Static references for calibration only. They do not create controllers,
    # collision APIs, randomization, or a physics timeline.
    def Xform "task_0"
    {{
{_xform_block("        ", task_root, [1.0, 1.0, 1.0])}

'''
    for fixture in fixture_specs:
        stage += f'''
        def Xform "{fixture['name']}" (
            prepend references = @{_reference_path(fixture['path'], output.parent)}@
        )
        {{
{_xform_block("            ", fixture["translation"], fixture["scale"], fixture["orientation"])}
        }}
'''
    for robot in robot_specs:
        stage += f'''
        def Xform "{robot['name']}" (
            prepend references = @{_reference_path(robot['path'], output.parent)}@
        )
        {{
'''
        if robot["child"]:
            stage += f'''            over Xform "{robot['child']}"
            {{
{_xform_block("                ", robot["translation"], [1.0, 1.0, 1.0], robot["orientation"])}
            }}
'''
        else:
            stage += _xform_block("            ", robot["translation"], [1.0, 1.0, 1.0], robot["orientation"]) + "\n"
        stage += "        }\n"
    for obj in object_specs:
        stage += f'''
        def Xform "{obj['name']}" (
            prepend references = @{_reference_path(obj['path'], output.parent)}@
        )
        {{
            over Xform "{obj['child']}"
            {{
{_xform_block("                ", obj["translation"], obj["scale"], obj["orientation"])}
            }}
        }}
'''
    if not args.no_hand:
        stage += f'''
        def Xform "obstacle_1" (
            prepend references = @{_reference_path(hand_path, output.parent)}@
        )
        {{
            over Xform "mano"
            {{
{_xform_block("                ", hand_translation, [args.hand_scale] * 3, [1.0, 0.0, 0.0, 0.0])}
            }}
        }}
'''
    stage += "    }\n}\n"
    output.write_text(stage, encoding="utf-8")

    metadata = {
        "stage": str(output),
        "task_yaml": str(task_path),
        "asset_root": str(asset_root),
        "background_mode": str(manifest.get("background_mode", "3dgs_mesh")),
        "calibration_rule": "keep /World/task_0 fixed; move only /World/BackgroundCalibration",
        "task_root_translation": task_root,
        "fixtures": [
            {"name": fixture["name"], "asset": str(fixture["path"]), "translation": fixture["translation"]}
            for fixture in fixture_specs
        ],
        "robots": [
            {"name": robot["name"], "asset": str(robot["path"]), "translation": robot["translation"], "child": robot["child"]}
            for robot in robot_specs
        ],
        "objects": [
            {"name": obj["name"], "asset": str(obj["path"]), "translation": obj["translation"], "child": obj["child"]}
            for obj in object_specs
        ],
        "skipped_assets": skipped_assets,
        "hand": None if args.no_hand else {"asset": str(hand_path), "translation": hand_translation},
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(
        "[worldcomposer] STATUS=SUCCESS step=create_task_calibration_stage "
        f"stage={output} fixtures={len(fixture_specs)} robots={len(robot_specs)} "
        f"objects={len(object_specs)} skipped_assets={len(skipped_assets)}"
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(
                f"[worldcomposer] STATUS=FAILED step=create_task_calibration_stage exit_code={exc.code}",
                file=sys.stderr,
            )
        raise
    except Exception:
        print("[worldcomposer] STATUS=FAILED step=create_task_calibration_stage", file=sys.stderr)
        raise
