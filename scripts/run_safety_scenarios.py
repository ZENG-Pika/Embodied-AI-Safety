#!/usr/bin/env python3
"""Prepare or run existing SimBox tasks with a common MANO intrusion overlay."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "workflows/simbox/core/configs/tasks"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _write_yaml(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=False)


def _repo_path(value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_id(path: Path) -> str:
    try:
        relative = path.relative_to(TASK_ROOT).with_suffix("")
    except ValueError:
        relative = path.relative_to(REPO_ROOT).with_suffix("")
    return "/".join(relative.parts)


def _discover(config: Dict[str, Any]) -> List[Path]:
    selection = config.get("selection", {})
    filters = selection.get("filters", {}) or {}
    candidates = set()
    if selection.get("all_tasks"):
        candidates.update(TASK_ROOT.rglob("*.yaml"))
    for item in selection.get("include", []) or []:
        path = _repo_path(str(item))
        if path.is_dir():
            candidates.update(path.rglob("*.yaml"))
        elif path.is_file():
            candidates.add(path)
        else:
            raise FileNotFoundError(f"Selected task does not exist: {path}")
    for pattern in selection.get("globs", []) or []:
        candidates.update(REPO_ROOT.glob(str(pattern)))

    excludes = selection.get("exclude_globs", []) or []
    result = []
    for path in sorted(p.resolve() for p in candidates if p.suffix in (".yaml", ".yml")):
        relative = str(path.relative_to(REPO_ROOT))
        if any(fnmatch.fnmatch(relative, pattern) for pattern in excludes):
            continue
        document = _load_yaml(path)
        tasks = document.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            continue
        try:
            category = path.relative_to(TASK_ROOT.resolve()).parts[0]
        except ValueError:
            category = Path(relative).parts[0]
        include_categories = set(filters.get("include_categories", []) or [])
        exclude_categories = set(filters.get("exclude_categories", []) or [])
        if include_categories and category not in include_categories:
            continue
        if category in exclude_categories:
            continue
        arena_names = {
            Path(str(task.get("arena_file", ""))).name
            for task in tasks if isinstance(task, dict)
        }
        include_arenas = set(filters.get("include_arena_files", []) or [])
        exclude_arenas = set(filters.get("exclude_arena_files", []) or [])
        if include_arenas and not arena_names.intersection(include_arenas):
            continue
        if arena_names.intersection(exclude_arenas):
            continue
        result.append(path)
    if not result:
        raise ValueError("No task YAML files matched selection")
    return result


def _merge_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _infer_target_objects(task: Dict[str, Any]) -> List[str]:
    targets = []
    referenced_objects = []
    for skill_group in task.get("skills", []) or []:
        if not isinstance(skill_group, dict):
            continue
        for robot_sequences in skill_group.values():
            for arm_group in robot_sequences or []:
                if not isinstance(arm_group, dict):
                    continue
                for sequence in arm_group.values():
                    for skill in sequence or []:
                        if not isinstance(skill, dict):
                            continue
                        objects = skill.get("objects") or []
                        if objects and objects[0] not in referenced_objects:
                            referenced_objects.append(objects[0])
                        if (skill.get("name") in ("pick", "dexpick", "dynamicpick", "manualpick")
                                and objects and objects[0] not in targets):
                            targets.append(objects[0])
    if targets:
        return targets
    named_targets = [
        str(item.get("name")) for item in task.get("objects", []) or []
        if isinstance(item, dict) and str(item.get("name", "")).startswith("pick_object")
    ]
    return named_targets or referenced_objects


def _infer_placement_targets(task: Dict[str, Any], pick_targets: Iterable[str]) -> List[str]:
    pick_targets = set(pick_targets)
    result = []
    for skill_group in task.get("skills", []) or []:
        if not isinstance(skill_group, dict):
            continue
        for robot_sequences in skill_group.values():
            for arm_group in robot_sequences or []:
                for sequence in arm_group.values() if isinstance(arm_group, dict) else []:
                    for skill in sequence or []:
                        if not isinstance(skill, dict) or skill.get("name") not in ("place", "dexplace"):
                            continue
                        objects = skill.get("objects") or []
                        for name in objects[1:]:
                            if name not in pick_targets and name not in result:
                                result.append(name)
    return result


def _inject_task(task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(task)
    intrusion = config["human_intrusion"]
    evaluation = config["safety_evaluation"]
    object_name = str(intrusion.get("object_name", "obstacle_1"))

    objects = result.setdefault("objects", [])
    for obj in objects:
        if isinstance(obj, dict) and obj.get("target_class") == "ArticulatedObject":
            # Replacing an articulated USD while PhysX contact tensor views
            # still reference its colliders invalidates the simulation view.
            # Select the model during initial scene setup, then retain it for
            # all episodes in this safety-scenario process.
            obj["reload_each_episode"] = False
    if not any(isinstance(obj, dict) and obj.get("name") == object_name for obj in objects):
        asset_path = _repo_path(str(intrusion["asset_path"])).resolve()
        objects.append({
            "name": object_name,
            "path": str(asset_path),
            "target_class": "RigidObject",
            "prim_path_child": intrusion.get("prim_path_child", "mano"),
            "translation": [0.0, 0.0, 0.0],
            "euler": [0.0, 0.0, 0.0],
            "scale": intrusion.get("scale", [1.25, 1.25, 1.25]),
            "apply_randomization": False,
            "physical_params": intrusion.get("physical_params", {}),
        })

    spawn = intrusion.get("spawn", {})
    regions = result.setdefault("regions", [])
    if not any(isinstance(region, dict) and region.get("object") == object_name for region in regions):
        regions.append({
            "object": object_name,
            "target": spawn.get("support_object", "table"),
            "random_type": "A_on_B_region_sampler",
            "random_config": {
                "pos_range": [spawn.get("position_min_m", [0.30, 0.30, 0.0]),
                              spawn.get("position_max_m", [0.35, 0.35, 0.0])],
                "yaw_rotation": spawn.get("yaw_range_deg", [0.0, 0.0]),
            },
        })

    for robot in result.get("robots", []) or []:
        if not isinstance(robot, dict):
            continue
        ignored = list(robot.get("ignore_substring", []) or [])
        for token in ("obstacle", "mano"):
            if token not in ignored:
                ignored.append(token)
        robot["ignore_substring"] = ignored

    if evaluation.get("enable_depth_segmentation", True):
        camera_file = str(_repo_path(evaluation["depth_seg_camera_file"]).resolve())
        for camera in result.get("cameras", []) or []:
            if isinstance(camera, dict) and "head" in str(camera.get("name", "")).lower():
                camera["camera_file"] = camera_file

    targets = _infer_target_objects(result)
    placements = _infer_placement_targets(result, targets)
    motion = intrusion.get("motion", {})
    safety_overlay = {
        "enabled": bool(evaluation.get("enabled", True)),
        "risk_thresholds": evaluation.get("risk_thresholds"),
        "output_subdir": evaluation.get("output_subdir", "safety_reports"),
        "entities": {
            "target_objects": targets,
            "placement_targets": placements,
            "human_surrogates": [object_name],
        },
        "obstacle": {
            "enabled": bool(intrusion.get("enabled", True)),
            "name": object_name,
            "target": motion.get("target_m", [-0.10, -0.40, 0.80]),
            "speed": motion.get("speed_m_per_step", 0.0035),
            "fixed_z": motion.get("fixed_z_m", 0.80),
            "mode": motion.get("mode", "round_trip"),
        },
        "safety_gate": evaluation.get("safety_gate", {}),
    }
    result["safety_eval"] = _merge_dict(result.get("safety_eval", {}), safety_overlay)
    return result


def _prepare(source: Path, config: Dict[str, Any]) -> tuple[str, Path, Path]:
    runtime = config["runtime"]
    scenario_id = _relative_id(source)
    slug = scenario_id.replace("/", "__")
    generated_root = _repo_path(str(runtime.get("generated_dir", ".generated/safety_scenarios")))
    task_output = generated_root / "tasks" / f"{slug}.yaml"
    launcher_output = generated_root / "launchers" / f"{slug}.yaml"

    task_document = _load_yaml(source)
    task_document["tasks"] = [
        _inject_task(task, config) for task in task_document["tasks"]
    ]
    _write_yaml(task_output, task_document)

    launcher = _load_yaml(_repo_path(runtime["launcher_template"]))
    launcher["name"] = f"safety_{slug}"
    scene_args = launcher["load_stage"]["scene_loader"]["args"]
    scene_args["cfg_path"] = str(task_output.resolve())
    simulator = scene_args.setdefault("simulator", {})
    simulator["headless"] = bool(runtime.get("headless", True))
    randomizer = launcher["load_stage"]["layout_random_generator"]["args"]
    randomizer["random_num"] = int(runtime.get("random_num", 1))
    randomizer["strict_mode"] = bool(runtime.get("strict_mode", True))
    launcher["store_stage"]["writer"]["args"]["output_dir"] = (
        str(_repo_path(runtime.get("output_root", "output/safety_scenarios")).resolve())
        + f"/{slug}/"
    )
    _write_yaml(launcher_output, launcher)
    return scenario_id, task_output, launcher_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simbox/safety_scenarios.yaml")
    parser.add_argument(
        "--scenario", action="append",
        help="Select a scenario ID/glob from the catalog; repeatable",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-all", action="store_true", help="Required when selection.all_tasks=true and --run")
    args = parser.parse_args()

    config_path = _repo_path(args.config)
    config = _load_yaml(config_path)
    sources = _discover(config)
    if args.scenario:
        sources = [path for path in sources if any(
            fnmatch.fnmatch(_relative_id(path), pattern) for pattern in args.scenario
        )]
        if not sources:
            raise ValueError("--scenario filters matched no selected tasks")

    if args.list:
        for source in sources:
            print(_relative_id(source))
        print(f"Total: {len(sources)}")
        return 0

    if (config.get("selection", {}).get("all_tasks") and not args.scenario
            and not args.list and not args.confirm_all):
        raise SystemExit(
            "Select at least one task with --scenario, or explicitly use --confirm-all"
        )

    prepared = [_prepare(source, config) for source in sources]
    for scenario_id, task_path, launcher_path in prepared:
        print(f"Prepared {scenario_id}\n  task: {task_path}\n  launcher: {launcher_path}")
    if not args.run:
        return 0

    isaac_python = _repo_path(config["runtime"]["isaac_python"])
    for scenario_id, _, launcher_path in prepared:
        print(f"Running {scenario_id}", flush=True)
        subprocess.run(
            [str(isaac_python), str(REPO_ROOT / "launcher.py"), "--config", str(launcher_path)],
            cwd=REPO_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
