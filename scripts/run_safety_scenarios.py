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
_POLICY_INDEX_FIELDS = (
    "body_indices", "head_indices", "lift_indices",
    "left_joint_indices", "right_joint_indices",
    "left_gripper_indices", "right_gripper_indices",
)


class UnsupportedScenarioError(RuntimeError):
    pass


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
        if isinstance(obj, dict) and obj.get("target_class") in (
            "RigidObject",
            "ArticulatedObject",
        ):
            # Replacing a USD while PhysX tensor/contact views reference its
            # colliders invalidates the simulation view. Safety rollouts keep
            # the initially selected asset and randomize only its pose.
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
            "reload_each_episode": False,
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

    if evaluation.get("enable_depth_segmentation", True) and evaluation.get(
        "replace_head_camera_for_safety", False
    ):
        camera_file = str(_repo_path(evaluation["depth_seg_camera_file"]).resolve())
        for camera in result.get("cameras", []) or []:
            if isinstance(camera, dict) and "head" in str(camera.get("name", "")).lower():
                camera["camera_file"] = camera_file

    # Keep the policy's original head camera unchanged. Add a separate camera
    # for depth/segmentation so the trained visual policy sees its training view.
    if evaluation.get("enable_depth_segmentation", True) and not evaluation.get(
        "replace_head_camera_for_safety", False
    ):
        camera_file = str(_repo_path(evaluation["depth_seg_camera_file"]).resolve())
        cameras = result.setdefault("cameras", [])
        for robot in result.get("robots", []) or []:
            robot_name = str(robot.get("name", "")) if isinstance(robot, dict) else ""
            head = next(
                (
                    camera for camera in cameras
                    if isinstance(camera, dict)
                    and camera.get("name") == f"{robot_name}_head"
                ),
                None,
            )
            if head is None:
                continue
            safety_name = f"{robot_name}_safety"
            if any(isinstance(camera, dict) and camera.get("name") == safety_name for camera in cameras):
                continue
            safety_camera = copy.deepcopy(head)
            safety_camera["name"] = safety_name
            safety_camera["camera_file"] = camera_file
            cameras.append(safety_camera)

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
            "enabled": bool(intrusion.get("enabled", True)) and bool(motion.get("enabled", True)),
            "name": object_name,
            "motion_enabled": bool(motion.get("enabled", True)),
            "target": motion.get("target_m", [-0.10, -0.40, 0.80]),
            "speed": motion.get("speed_m_per_step", 0.0035),
            "fixed_z": motion.get("fixed_z_m", 0.80),
            "mode": motion.get("mode", "round_trip"),
        },
        "safety_gate": evaluation.get("safety_gate", {}),
    }
    result["safety_eval"] = _merge_dict(result.get("safety_eval", {}), safety_overlay)

    policy = dict(config.get("policy", {}) or {})
    if policy.get("enabled", False):
        result["policy"] = _merge_dict(result.get("policy", {}), policy)

    overview = config.get("overview_camera", {}) or {}
    if overview.get("enabled", False):
        camera_name = str(overview.get("name", "split_aloha_overview"))
        cameras = result.setdefault("cameras", [])
        cameras[:] = [camera for camera in cameras if camera.get("name") != camera_name]
        cameras.append({
            "name": camera_name,
            "translation": overview.get("translation", [2.8, -3.2, 2.6]),
            "orientation": overview.get("orientation", [0.780351, 0.51705, 0.194274, 0.293205]),
            "camera_axes": "usd",
            "camera_file": overview.get(
                "camera_file", "workflows/simbox/core/configs/cameras/astra.yaml"
            ),
            "parent": None,
            "apply_randomization": False,
        })
    return result


def _validate_random_policy_task(
    task_document: Dict[str, Any],
    source: Path,
    asset_root_override: str | None = None,
) -> None:
    for task in task_document.get("tasks", []) or []:
        asset_root = (
            _repo_path(str(asset_root_override)).resolve()
            if asset_root_override
            else _repo_path(str(task.get("asset_root", "")))
        )
        for robot in task.get("robots", []) or []:
            if not isinstance(robot, dict):
                continue
            merged = {}
            config_file = robot.get("robot_config_file")
            if config_file:
                config_path = _repo_path(str(config_file))
                if not config_path.is_file():
                    raise UnsupportedScenarioError(
                        f"{_relative_id(source)}: robot config not found: {config_path}"
                    )
                merged.update(_load_yaml(config_path))
            merged.update(robot)
            controllable = sum(
                len(merged.get(field, []) or []) for field in _POLICY_INDEX_FIELDS
            )
            if controllable == 0:
                raise UnsupportedScenarioError(
                    f"{_relative_id(source)}: robot {merged.get('name', '<unnamed>')} "
                    "has no supported joint-index fields"
                )
            robot_asset = merged.get("path")
            if robot_asset and not _asset_reference_exists(asset_root, str(robot_asset)):
                raise UnsupportedScenarioError(
                    f"{_relative_id(source)}: robot asset missing: {asset_root / str(robot_asset)}"
                )
        for obj in task.get("objects", []) or []:
            if not isinstance(obj, dict) or obj.get("art_cat"):
                continue
            object_asset = obj.get("path")
            if object_asset and not _asset_reference_exists(asset_root, str(object_asset)):
                raise UnsupportedScenarioError(
                    f"{_relative_id(source)}: object asset missing: {asset_root / str(object_asset)}"
                )


def _asset_reference_exists(asset_root: Path, value: str) -> bool:
    if "${" in value or value.startswith(("omniverse://", "http://", "https://")):
        return True
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = asset_root / path
    return path.exists()


def _prepare(source: Path, config: Dict[str, Any]) -> tuple[str, Path, Path]:
    runtime = config["runtime"]
    scenario_id = _relative_id(source)
    slug = scenario_id.replace("/", "__")
    generated_root = _repo_path(str(runtime.get("generated_dir", ".generated/safety_scenarios")))
    task_output = generated_root / "tasks" / f"{slug}.yaml"
    launcher_output = generated_root / "launchers" / f"{slug}.yaml"

    task_document = _load_yaml(source)
    asset_root_override = runtime.get("asset_root_override")
    if asset_root_override:
        for task in task_document.get("tasks", []) or []:
            if isinstance(task, dict):
                task["asset_root"] = str(_repo_path(str(asset_root_override)).resolve())
    if config.get("policy", {}).get("enabled", False):
        _validate_random_policy_task(task_document, source, asset_root_override)
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
    randomizer["max_attempts"] = int(runtime.get("max_attempts", 0)) or None
    writer_args = launcher["store_stage"]["writer"]["args"]
    writer_args["output_dir"] = (
        str(_repo_path(runtime.get("output_root", "output/safety_scenarios")).resolve())
        + f"/{slug}/"
    )
    writer_args["failure_output_dir"] = (
        str(_repo_path(runtime.get("failure_output_root", "failure_output/safety_scenarios")).resolve())
        + f"/{slug}/"
    )
    writer_args["max_attempts"] = int(runtime.get("max_attempts", 0)) or None
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
    mode.add_argument("--list-runnable", action="store_true")
    mode.add_argument("--list-skipped", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-all", action="store_true", help="Required when selection.all_tasks=true and --run")
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Run one scene attempt only; keep planner failures without re-randomizing the scene.",
    )
    parser.add_argument(
        "--policy", choices=("random-diffusion", "trained-diffusion"),
        help="Replace scripted skills with a policy rollout.",
    )
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--policy-control-grippers", action="store_true")
    parser.add_argument("--checkpoint", help="Trained DP checkpoint directory.")
    parser.add_argument("--model-root", help="Directory containing the model inference.py.")
    parser.add_argument("--policy-python", help="Python executable for the trained DP service.")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument(
        "--asset-root-override",
        help="Use a complete external SimBox asset root when a task asset_root is unavailable.",
    )
    parser.add_argument(
        "--max-attempts", type=int,
        help="Maximum scene attempts; random-policy runs default to one.",
    )
    parser.add_argument(
        "--evaluation-output-root", default="eval_output",
        help="Sibling directory for random-policy rollout records.",
    )
    parser.add_argument(
        "--evaluation-failure-root", default="failure_eval_output",
        help="Sibling directory for failed random-policy rollout records.",
    )
    args = parser.parse_args()

    config_path = _repo_path(args.config)
    config = _load_yaml(config_path)
    if args.policy == "random-diffusion":
        config["policy"] = {
            "enabled": True,
            "type": "random_diffusion",
            "seed": args.policy_seed,
            "action_horizon": 8,
            "diffusion_steps": 10,
            "hidden_dim": 128,
            "max_episode_steps": args.max_episode_steps,
            "control_grippers": args.policy_control_grippers,
        }
    elif args.policy == "trained-diffusion":
        missing = [
            name for name, value in (
                ("--checkpoint", args.checkpoint),
                ("--model-root", args.model_root),
                ("--policy-python", args.policy_python),
            ) if not value
        ]
        if missing:
            raise SystemExit(
                "--policy trained-diffusion requires " + ", ".join(missing)
            )
        config["policy"] = {
            "enabled": True,
            "type": "trained_diffusion",
            "checkpoint": str(_repo_path(args.checkpoint).resolve()),
            "model_root": str(_repo_path(args.model_root).resolve()),
            "python_executable": str(_repo_path(args.policy_python).resolve()),
            "device": args.policy_device,
            "seed": args.policy_seed,
            "max_episode_steps": args.max_episode_steps,
            "replan_steps": args.policy_replan_steps,
            "camera_name": "franka_head",
        }
    if args.asset_root_override:
        config.setdefault("runtime", {})["asset_root_override"] = args.asset_root_override
    if args.policy in ("random-diffusion", "trained-diffusion"):
        config.setdefault("runtime", {})["max_attempts"] = (
            args.max_attempts if args.max_attempts is not None else 1
        )
        config["runtime"]["output_root"] = args.evaluation_output_root
        config["runtime"]["failure_output_root"] = args.evaluation_failure_root
    elif args.max_attempts is not None:
        config.setdefault("runtime", {})["max_attempts"] = args.max_attempts
    if args.no_retry:
        runtime = config.setdefault("runtime", {})
        runtime["strict_mode"] = False
        runtime["max_attempts"] = 1
    if (args.list_runnable or args.list_skipped) and not config.get("policy", {}).get("enabled"):
        config["policy"] = {"enabled": True, "type": "random_diffusion"}
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

    if args.list_runnable or args.list_skipped:
        runnable = []
        skipped = []
        for source in sources:
            try:
                _validate_random_policy_task(
                    _load_yaml(source),
                    source,
                    config.get("runtime", {}).get("asset_root_override"),
                )
                runnable.append(_relative_id(source))
            except UnsupportedScenarioError as error:
                skipped.append(str(error))
        values = runnable if args.list_runnable else skipped
        for value in values:
            print(value)
        print(f"Total: {len(values)}", file=sys.stderr)
        return 0

    if (config.get("selection", {}).get("all_tasks") and not args.scenario
            and not args.list and not args.confirm_all):
        raise SystemExit(
            "Select at least one task with --scenario, or explicitly use --confirm-all"
        )

    prepared = []
    skipped = []
    for source in sources:
        try:
            prepared.append(_prepare(source, config))
        except UnsupportedScenarioError as error:
            skipped.append(str(error))
    for reason in skipped:
        print(f"Skipped unsupported scenario: {reason}", file=sys.stderr)
    if not prepared:
        print("No supported scenarios remained after policy preflight", file=sys.stderr)
        return 2
    for scenario_id, task_path, launcher_path in prepared:
        print(f"Prepared {scenario_id}\n  task: {task_path}\n  launcher: {launcher_path}")
    if not args.run:
        return 0

    isaac_python = _repo_path(config["runtime"]["isaac_python"])
    curobo_root = _repo_path(
        str(config["runtime"].get(
            "curobo_root",
            "/home/pika/Workspace/pika/InternDataEngine/InternDataAssets/curobo",
        ))
    )
    if not (curobo_root / "src/curobo").is_dir():
        raise FileNotFoundError(f"CuRobo source directory does not exist: {curobo_root / 'src/curobo'}")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(curobo_root / "src"), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child_env.pop("INTERNDATA_ISAAC5_COMPAT", None)
    for scenario_id, _, launcher_path in prepared:
        print(f"Running {scenario_id}", flush=True)
        subprocess.run(
            [str(isaac_python), str(REPO_ROOT / "launcher.py"), "--config", str(launcher_path)],
            cwd=REPO_ROOT,
            env=child_env,
            check=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
