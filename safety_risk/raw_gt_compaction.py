"""Loss-aware compaction for Sim_Raw_GT JSON payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable


def _dict_entries(series: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(series, list):
        return
    for frame in series:
        if isinstance(frame, dict):
            yield frame
        elif isinstance(frame, list):
            for entry in frame:
                if isinstance(entry, dict):
                    yield entry


def _compact_collision_gt(collision: Dict[str, Any]) -> None:
    # collision_pair_gt is the identity/timing channel.  Force and impulse
    # values have authoritative dedicated channels and must not be repeated.
    has_force_channel = isinstance(collision.get("contact_force_gt"), list)
    has_impulse_channel = isinstance(collision.get("contact_impulse_gt"), list)
    has_shared_provenance = isinstance(collision.get("_provenance"), dict)
    for entry in _dict_entries(collision.get("collision_pair_gt")):
        if has_force_channel:
            entry.pop("force_n", None)
            entry.pop("force_vector_n", None)
        if has_impulse_channel:
            entry.pop("impulse_ns", None)
            entry.pop("impulse_vector_ns", None)
        if has_shared_provenance:
            entry.pop("source", None)

    # Main risk rules use pair-level aggregates.  Thousands of per-point
    # records were repeated across location, force, and impulse, and dominated
    # the JSON size without changing any HS/PT/RS decision.
    aggregate_keys = {
        "collision_location_gt": ("location_m",),
        "contact_force_gt": ("force_n", "force_vector_n"),
        "contact_impulse_gt": ("impulse_ns", "impulse_vector_ns"),
    }
    for field, required_any in aggregate_keys.items():
        for entry in _dict_entries(collision.get(field)):
            if not any(entry.get(key) is not None for key in required_any):
                continue
            entry.pop("contacts", None)
            entry.pop("contact_points", None)
            entry.pop("contact_points_m", None)


def _compact_robot_state(robot: Dict[str, Any]) -> None:
    left = robot.get("ee_pose_gt")
    right = robot.get("ee_pose_right_gt")
    already_merged = (
        isinstance(left, list)
        and (not left or isinstance(left[0], dict))
        and not isinstance(right, list)
    )
    if not already_merged and (isinstance(left, list) or isinstance(right, list)):
        left_values = left if isinstance(left, list) else []
        right_values = right if isinstance(right, list) else []
        frame_count = max(len(left_values), len(right_values))
        robot["ee_pose_gt"] = [
            {
                "left": left_values[index] if index < len(left_values) else None,
                "right": right_values[index] if index < len(right_values) else None,
            }
            for index in range(frame_count)
        ]
    robot.pop("ee_pose_right_gt", None)
    robot.pop("joint_position_q_right_gt", None)

    # These legacy flags are not transforms.  Keep a field only when it holds
    # actual numeric data rather than the literal availability marker.
    for key in ("T_base_ee_fl", "T_base_ee_fr", "T_world_base"):
        if robot.get(key) == "available":
            robot.pop(key, None)


def _compact_planner_log(planner: Dict[str, Any]) -> None:
    trajectories = planner.get("planned_trajectory")
    if not isinstance(trajectories, list):
        return
    unique_plans: Dict[str, Any] = {}
    events = []
    for capture_index, plan in enumerate(trajectories):
        canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        plan_id = "plan_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        if plan_id not in unique_plans:
            stored = dict(plan) if isinstance(plan, dict) else {"trajectory": plan}
            stored["plan_id"] = plan_id
            unique_plans[plan_id] = stored
        events.append({"capture_index": capture_index, "plan_id": plan_id})
    planner["planned_trajectory"] = {
        "plans": list(unique_plans.values()),
        "events": events,
        "num_unique_plans": len(unique_plans),
        "num_capture_events": len(events),
    }


def compact_sim_raw_gt(raw_gt: Dict[str, Any]) -> None:
    """Compact a Sim_Raw_GT mapping in place without changing score inputs."""
    metadata = raw_gt.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("language_instruction", None)
        metadata.pop("detailed_language_instruction", None)

    robot = raw_gt.get("robot_state")
    if isinstance(robot, dict):
        _compact_robot_state(robot)

    collision = raw_gt.get("collision_gt")
    if isinstance(collision, dict):
        _compact_collision_gt(collision)

    distance = raw_gt.get("distance_gt")
    if isinstance(distance, dict):
        distance.pop("ee_obstacle_distance_approx_m", None)

    outcome = raw_gt.get("outcome_gt")
    if isinstance(outcome, dict):
        outcome.pop("drop_event_episode_gt", None)
        outcome.pop("drop_height_episode_max_m", None)

    planner = raw_gt.get("planner_log")
    if isinstance(planner, dict):
        _compact_planner_log(planner)
