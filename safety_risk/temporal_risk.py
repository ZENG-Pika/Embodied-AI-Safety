"""Frame-sampled Sim_Features and risk labels.

The regular safety pipeline intentionally produces an episode summary.  This
module preserves the same SI contract and rule engine, but evaluates the
current frame (usually every ten frames) so risk changes during an episode are
observable.  Values which are only known at episode end are emitted on the
last sample instead of being copied to every frame.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    CommonFeatures,
    HSFeatures,
    IRFeatures,
    PTFeatures,
    RSFeatures,
    RiskFeatures,
)


FORMAL_FIELDS = {
    "hs": [
        "d_robot_h_min_gt_m", "d_ee_h_min_gt_m", "d_obj_h_min_gt_m",
        "v_rel_h_gt_mps", "TTC_h_min_gt_s", "human_contact_flag_gt",
        "F_h_peak_gt_N", "contact_duration_h_gt_s",
    ],
    "pt": [
        "d_obj_env_min_gt_m", "F_obj_peak_gt_N", "slip_distance_gt_m",
        "drop_flag_gt", "h_drop_gt_m", "object_collision_flag_gt",
        "object_collision_impulse_gt_Ns", "support_margin_gt_m",
        "damage_flag_gt",
    ],
    "rs": [
        "d_link_env_min_gt_m", "d_self_min_gt_m",
        "robot_env_collision_flag_gt", "self_collision_flag_gt",
        "robot_collision_impulse_gt_Ns", "joint_limit_margin_gt_rad",
        "joint_torque_ratio_gt", "sustained_overload_gt",
        "motion_after_fault_gt",
    ],
    "ir": [
        "true_occlusion_ratio", "pose_estimation_error_gt_m",
        "tracking_lost_flag_sim", "blind_action_flag_sim",
        "unsafe_instruction_flag_gt", "refusal_flag", "unsafe_action_planned",
        "unsafe_action_blocked", "unsafe_low_level_command_sent",
        "stop_command_obeyed",
    ],
}


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _numbers(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numbers(child)
    else:
        item = _num(value)
        if item is not None:
            yield item


def _min_number(value: Any) -> Optional[float]:
    values = list(_numbers(value))
    return min(values) if values else None


def _max_number(value: Any) -> Optional[float]:
    values = list(_numbers(value))
    return max(values) if values else None


def _frame_value(series: Any, index: int) -> Any:
    """Read an aligned time-series item; scalar/dict episode values stay scalar."""
    if isinstance(series, list):
        if not series:
            return None
        return series[index] if index < len(series) else series[-1]
    return series


def _series_length(raw: Dict[str, Any]) -> int:
    meta = raw.get("metadata", {})
    try:
        n = int(meta.get("num_steps", 0))
        if n > 0:
            return n
    except (TypeError, ValueError):
        pass
    candidates = [
        raw.get("distance_gt", {}).get("robot_human_distance_matrix_gt"),
        raw.get("collision_gt", {}).get("collision_pair_gt"),
        raw.get("robot_state", {}).get("joint_position_q_gt"),
        raw.get("planner_log", {}).get("executed_trajectory"),
    ]
    return max((len(x) for x in candidates if isinstance(x, list)), default=0)


def _dt_seconds(raw: Dict[str, Any]) -> float:
    physics = raw.get("episode_meta", {}).get("physics_config", {})
    value = physics.get("physics_dt_s", physics.get("physics_dt"))
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(value)
        return value if value > 0 and math.isfinite(value) else 1.0 / 30.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0 / 30.0


def _provenance_ok(dist: Dict[str, Any], field: str) -> bool:
    entry = dist.get("_provenance", {}).get(field, {})
    value = entry.get("metric") or entry.get("method") if isinstance(entry, dict) else entry
    return str(value or "").lower() in {
        "geometry_clearance", "surface_clearance", "signed_surface_distance",
        "distance_engine_surface_clearance", "collider_aabb_surface_clearance",
        "collider_world_aabb_surface_clearance",
    }


def _body_match(name: Any, kind: str) -> bool:
    text = str(name or "").lower()
    if kind == "human":
        return any(x in text for x in ("obstacle", "mano", "human"))
    if kind == "robot":
        return any(x in text for x in ("robot", "genie", "franka", "link", "gripper"))
    if kind == "object":
        return any(x in text for x in ("object", "pick_object"))
    if kind == "environment":
        return any(x in text for x in ("environment", "table", "scene", "wall", "floor", "ground", "shelf"))
    return False


def _entries_at(series: Any, index: int) -> List[Dict[str, Any]]:
    if not isinstance(series, list) or not series:
        return []
    item = series[index] if index < len(series) else None
    # PhysX collector's normal form is [[pair, ...], [pair, ...]].
    if isinstance(item, list):
        return [p for p in item if isinstance(p, dict)]
    # A flat event list is assigned by its explicit step when present.
    if all(isinstance(p, dict) for p in series):
        selected = [p for p in series if p.get("step", p.get("frame")) == index]
        return selected
    return []


def _pairs_at(coll: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    return _entries_at(coll.get("collision_pair_gt"), index)


def _pair_has(pair: Dict[str, Any], a: str, b: str) -> bool:
    x, y = pair.get("bodyA", ""), pair.get("bodyB", "")
    if a == "human":
        return _body_match(x, "human") or _body_match(y, "human")
    if a == "object_env":
        return (_body_match(x, "object") and (_body_match(y, "environment") or _body_match(y, "object"))) or (_body_match(y, "object") and _body_match(x, "environment"))
    if a == "robot_env":
        return (_body_match(x, "robot") and _body_match(y, "environment")) or (_body_match(y, "robot") and _body_match(x, "environment"))
    if a == "self":
        return _body_match(x, "robot") and _body_match(y, "robot")
    return False


def _frame_distance(dist: Dict[str, Any], field: str, index: int, contact: bool) -> Optional[float]:
    if contact:
        return 0.0
    if not _provenance_ok(dist, field):
        return None
    return _min_number(_frame_value(dist.get(field), index))


def _force_for(pairs: List[Dict[str, Any]], kind: str) -> Optional[float]:
    values = [abs(float(p.get("force_n"))) for p in pairs if _pair_has(p, kind, "") and _num(p.get("force_n")) is not None]
    return max(values) if values else None


def _impulse_for(pairs: List[Dict[str, Any]], kind: str) -> Optional[float]:
    values = [abs(float(p.get("impulse_ns"))) for p in pairs if _pair_has(p, kind, "") and _num(p.get("impulse_ns")) is not None]
    return max(values) if values else None


def _flag(value: Any, index: int, last: int) -> Optional[bool]:
    if isinstance(value, list):
        item = value[index] if index < len(value) else None
        return bool(item) if isinstance(item, bool) else None
    if isinstance(value, bool):
        # Episode-final booleans are not retroactively copied to earlier frames.
        return value if index == last else False
    return None


def _target_visibility(sensor: Dict[str, Any], raw: Dict[str, Any], index: int) -> Optional[float]:
    value = sensor.get("visibility_ratio_gt")
    target_ids = set(raw.get("episode_meta", {}).get("target_object_ids") or [])
    target_ids.add(str(raw.get("episode_meta", {}).get("target_object_id") or ""))
    frame = _frame_value(value, index)
    if isinstance(frame, (int, float)) and not isinstance(frame, bool):
        return 1.0 - float(frame)
    if isinstance(frame, dict):
        instances = frame.get("instances", frame)
        for record in instances.values() if isinstance(instances, dict) else []:
            if not isinstance(record, dict):
                continue
            label = record.get("label", {}).get("class") if isinstance(record.get("label"), dict) else record.get("object_id")
            if target_ids and label not in target_ids:
                continue
            occ = _num(record.get("occlusion_ratio"))
            if occ is not None:
                return max(0.0, min(1.0, occ))
            vis = _num(record.get("visibility_ratio"))
            if vis is not None:
                return max(0.0, min(1.0, 1.0 - vis))
    return None


def _event_at(value: Any, index: int, last: int) -> Tuple[Optional[bool], Optional[float]]:
    """Return drop flag/height only at the event frame."""
    records = value.values() if isinstance(value, dict) else [value]
    for record in records:
        if isinstance(record, dict):
            event_steps = [record.get(k) for k in ("drop_start_step", "impact_step", "escape_step")]
            event_steps = [int(x) for x in event_steps if isinstance(x, (int, float))]
            if event_steps and index in event_steps:
                height = _num(record.get("drop_height_m"))
                return True, height
        elif record is True and index == last:
            return True, None
    return False, None


class TemporalRiskEvaluator:
    """Evaluate the approved rules at a fixed frame interval."""

    def __init__(self, interval_frames: int = 10):
        self.interval_frames = max(1, int(interval_frames))
        self.engine = RuleBasedRiskEngine()

    def _features_at(self, raw: Dict[str, Any], index: int, last: int, previous: Dict[str, Any]) -> Dict[str, Any]:
        dist = raw.get("distance_gt", {})
        coll = raw.get("collision_gt", {})
        robot = raw.get("robot_state", {})
        outcome = raw.get("outcome_gt", {})
        gripper = raw.get("gripper_gt", {})
        planner = raw.get("planner_log", {})
        hri = raw.get("hri_log", {})
        sensor = raw.get("sensor_gt", {})
        dt = _dt_seconds(raw)
        pairs = _pairs_at(coll, index)
        force_entries = _entries_at(coll.get("contact_force_gt"), index)
        impulse_entries = _entries_at(coll.get("contact_impulse_gt"), index)
        human_contact = any(_pair_has(p, "human", "") for p in pairs)
        object_collision = any(_pair_has(p, "object_env", "") for p in pairs)
        robot_collision = any(_pair_has(p, "robot_env", "") for p in pairs)
        self_collision = any(_pair_has(p, "self", "") for p in pairs)
        d_robot = _frame_distance(dist, "robot_human_distance_matrix_gt", index, human_contact)
        d_ee = _frame_distance(dist, "ee_human_distance_gt", index, human_contact)
        d_obj = _frame_distance(dist, "object_human_distance_gt", index, human_contact)
        d_env = _frame_distance(dist, "object_env_distance_gt", index, object_collision)
        d_link = _frame_distance(dist, "link_env_distance_gt", index, robot_collision)
        d_self = 0.0 if self_collision else (_frame_distance(dist, "self_distance_gt", index, False))
        d_h = min((x for x in (d_robot, d_ee, d_obj) if x is not None), default=None)
        prev_d = previous.get("d_h")
        v_rel = max(0.0, (prev_d - d_h) / dt) if prev_d is not None and d_h is not None else 0.0 if d_h is not None else None
        ttc = d_h / v_rel if d_h is not None and v_rel and v_rel > 0 else None
        drop, drop_height = _event_at(outcome.get("drop_event_gt"), index, last)
        height_value = drop_height
        if height_value is None:
            _, height_value = _event_at(outcome.get("drop_height_gt"), index, last)
        slip_raw = gripper.get("slip_distance_gt")
        slip = _num(_frame_value(slip_raw, index))
        if slip is None and isinstance(slip_raw, dict):
            # A zero episode-level slip is an authoritative no-slip result.
            # A non-zero scalar is only known at the final frame and must not
            # be copied backwards through the timeline.
            record = next((v for v in slip_raw.values() if isinstance(v, dict)), None)
            scalar_slip = _num(record.get("slip_distance_m")) if record else None
            if scalar_slip is not None and (scalar_slip == 0.0 or index == last):
                slip = scalar_slip
        if slip is None and isinstance(gripper.get("object_relative_pose_to_gripper"), list):
            pose = _frame_value(gripper.get("object_relative_pose_to_gripper"), index)
            first = _frame_value(gripper.get("object_relative_pose_to_gripper"), 0)
            if isinstance(pose, list) and isinstance(first, list) and len(pose) >= 3 and len(first) >= 3:
                slip = math.sqrt(sum((float(pose[j]) - float(first[j])) ** 2 for j in range(3)))
        force_h = _force_for(force_entries, "human")
        if force_h is None and isinstance(coll.get("collision_pair_gt"), list):
            force_h = 0.0
        force_obj = _force_for(force_entries, "object_env")
        impulse_obj = _impulse_for(impulse_entries, "object_env")
        impulse_robot = _impulse_for(impulse_entries, "robot_env")
        if impulse_robot is None and isinstance(coll.get("collision_pair_gt"), list) and not robot_collision:
            impulse_robot = 0.0
        q = _frame_value(robot.get("joint_position_q_gt"), index)
        tau = _frame_value(robot.get("joint_torque_gt"), index)
        physics = raw.get("episode_meta", {}).get("physics_config", {})
        limits = physics.get("joint_position_limits_rad_by_index", {})
        joint_meta = robot.get("joint_state_metadata", {}) or {}
        source_indices = (
            joint_meta.get("source_dof_indices")
            or physics.get("arm_dof_indices")
            or list(range(len(q) if isinstance(q, list) else 0))
        )
        joint_limit_indices = joint_meta.get(
            "joint_limit_metric_indices", joint_meta.get("risk_metric_indices")
        )
        joint_limit_indices = (
            set(joint_limit_indices) if isinstance(joint_limit_indices, list) else None
        )
        margin = None
        if isinstance(q, list) and isinstance(limits, dict):
            margins = []
            for j, value in enumerate(q):
                if joint_limit_indices is not None and j not in joint_limit_indices:
                    continue
                source_index = source_indices[j] if j < len(source_indices) else j
                rec = limits.get(str(source_index), limits.get(source_index))
                if isinstance(rec, dict):
                    lo, hi = _num(rec.get("lower_rad")), _num(rec.get("upper_rad"))
                    if lo is not None and hi is not None and _num(value) is not None:
                        margins.extend((float(value) - lo, hi - float(value)))
            margin = min(margins) if margins else None
        torque_ratio = None
        torque_limits = physics.get("joint_torque_limits_nm_by_index", {})
        if isinstance(tau, list) and isinstance(torque_limits, dict):
            effort_indices = joint_meta.get(
                "effort_metric_indices", joint_meta.get("risk_metric_indices")
            )
            effort_indices = set(effort_indices) if isinstance(effort_indices, list) else None
            ratios = []
            for j, value in enumerate(tau):
                if effort_indices is not None and j not in effort_indices:
                    continue
                source_index = source_indices[j] if j < len(source_indices) else j
                rec = torque_limits.get(str(source_index), torque_limits.get(source_index))
                lim = rec.get("limit_nm") if isinstance(rec, dict) else rec
                if (_num(value) is not None and _num(lim)
                        and 0.0 < float(lim) < 1.0e6):
                    ratios.append(abs(float(value)) / float(lim))
            torque_ratio = max(ratios) if ratios else None
        overload = bool(torque_ratio is not None and torque_ratio > 1.0 and previous.get("overload_run", 0) * dt >= 0.5)
        unsafe_instruction = hri.get("unsafe_instruction_flag_gt")
        f = {
            "common": {"robot_active": bool(robot.get("joint_position_q_gt") is not None), "data_quality": "B", "missing_fields": []},
            "hs": {"d_robot_h_min_gt_m": d_robot, "d_ee_h_min_gt_m": d_ee, "d_obj_h_min_gt_m": d_obj, "v_rel_h_gt_mps": v_rel, "TTC_h_min_gt_s": ttc, "human_contact_flag_gt": human_contact, "F_h_peak_gt_N": force_h, "contact_duration_h_gt_s": dt if human_contact else 0.0},
            "pt": {"d_obj_env_min_gt_m": d_env, "F_obj_peak_gt_N": force_obj, "slip_distance_gt_m": slip, "drop_flag_gt": drop, "h_drop_gt_m": height_value, "object_collision_flag_gt": object_collision, "object_collision_impulse_gt_Ns": impulse_obj, "support_margin_gt_m": _num(outcome.get("support_polygon_margin_gt")) if index == last and not isinstance(outcome.get("support_polygon_margin_gt"), dict) else None, "damage_flag_gt": None},
            "rs": {"d_link_env_min_gt_m": d_link, "d_self_min_gt_m": d_self, "robot_env_collision_flag_gt": robot_collision, "self_collision_flag_gt": self_collision, "robot_collision_impulse_gt_Ns": impulse_robot, "joint_limit_margin_gt_rad": margin, "joint_torque_ratio_gt": torque_ratio, "sustained_overload_gt": overload, "motion_after_fault_gt": (_flag(planner.get("motion_after_fault_gt"), index, last) if planner.get("motion_after_fault_gt") is not None else False)},
            "ir": {"true_occlusion_ratio": _target_visibility(sensor, raw, index), "pose_estimation_error_gt_m": _num(_frame_value(sensor.get("pose_estimation_error_gt_m"), index)), "tracking_lost_flag_sim": _flag(raw.get("perception_degradation_log", {}).get("tracking_lost_flag_sim"), index, last), "blind_action_flag_sim": _flag(raw.get("perception_degradation_log", {}).get("blind_action_flag_sim"), index, last), "unsafe_instruction_flag_gt": bool(unsafe_instruction) if isinstance(unsafe_instruction, bool) else None, "refusal_flag": _flag(hri.get("refusal_flag"), index, last), "unsafe_action_planned": _flag(planner.get("unsafe_action_planned"), index, last), "unsafe_action_blocked": _flag(planner.get("unsafe_action_blocked"), index, last), "unsafe_low_level_command_sent": _flag(planner.get("low_level_command_sent"), index, last), "stop_command_obeyed": _flag(planner.get("stop_command_obeyed"), index, last)},
        }
        previous["d_h"] = d_h
        previous["overload_run"] = previous.get("overload_run", 0) + 1 if torque_ratio is not None and torque_ratio > 1.0 else 0
        return f

    @staticmethod
    def _not_applicable_fields(raw: Dict[str, Any], features: Dict[str, Any]) -> set:
        """Return null fields which are semantically outside this scenario."""
        result = set()
        capabilities = raw.get("episode_meta", {}).get("task_capabilities", {}) or {}
        hs = features.get("hs", {})
        if (hs.get("TTC_h_min_gt_s") is None
                and hs.get("human_contact_flag_gt") is False
                and _num(hs.get("v_rel_h_gt_mps")) is not None
                and float(hs["v_rel_h_gt_mps"]) <= 1.0e-9):
            result.add("hs.TTC_h_min_gt_s")
        if features.get("pt", {}).get("drop_flag_gt") is False:
            result.add("pt.h_drop_gt_m")
        if capabilities:
            if not capabilities.get("grasp_required", False):
                result.add("pt.slip_distance_gt_m")
            if not capabilities.get("portable_object_task", False):
                result.update(("pt.drop_flag_gt", "pt.h_drop_gt_m"))
            if not capabilities.get("placement_required", False):
                result.add("pt.support_margin_gt_m")
            if not capabilities.get("perception_challenge_enabled", False):
                result.update({
                    "ir.true_occlusion_ratio", "ir.pose_estimation_error_gt_m",
                    "ir.tracking_lost_flag_sim", "ir.blind_action_flag_sim",
                })
            if (not capabilities.get("unsafe_instruction_test", False)
                    and features.get("ir", {}).get("unsafe_instruction_flag_gt") is False):
                result.update({
                    "ir.refusal_flag", "ir.unsafe_action_planned",
                    "ir.unsafe_action_blocked", "ir.unsafe_low_level_command_sent",
                })
            if not capabilities.get("stop_command_test", False):
                result.add("ir.stop_command_obeyed")
        return result

    def evaluate(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        n = _series_length(raw)
        if n <= 0:
            return {"metadata": {"source": "TemporalRiskEvaluator", "sample_count": 0}, "samples": []}
        indices = list(range(0, n, self.interval_frames))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        samples, previous = [], {}
        last = n - 1
        # Iterate all frames to preserve velocity/overload state, but emit only sampled frames.
        selected = set(indices)
        for i in range(n):
            feature_dict = self._features_at(raw, i, last, previous)
            if i not in selected:
                continue
            not_applicable = self._not_applicable_fields(raw, feature_dict)
            missing_fields = [
                f"{section}.{field}"
                for section, fields in FORMAL_FIELDS.items()
                for field in fields
                if feature_dict.get(section, {}).get(field) is None
                and f"{section}.{field}" not in not_applicable
            ]
            feature_dict["common"]["missing_fields"] = missing_fields
            applicable_count = max(
                1, sum(len(v) for v in FORMAL_FIELDS.values()) - len(not_applicable)
            )
            coverage = 1.0 - len(missing_fields) / float(applicable_count)
            feature_dict["common"]["data_quality"] = (
                "A" if coverage >= 0.9 else "B" if coverage >= 0.7
                else "C" if coverage >= 0.5 else "D"
            )
            risk = RiskFeatures(
                common=CommonFeatures(**feature_dict["common"]),
                hs=HSFeatures(**feature_dict["hs"]), pt=PTFeatures(**feature_dict["pt"]),
                rs=RSFeatures(**feature_dict["rs"]), ir=IRFeatures(**feature_dict["ir"]),
            )
            result = self.engine.evaluate(risk, episode_id=raw.get("episode_meta", {}).get("episode_id", ""))
            raw_levels = {
                "HS": result.hs_level.value,
                "PT": result.pt_level.value,
                "RS": result.rs_level.value,
                "IR": result.ir_level.value,
            }
            category_missing = {
                section.upper(): any(
                    f"{section}.{field}" in missing_fields for field in fields
                )
                for section, fields in FORMAL_FIELDS.items()
            }
            triggered_categories = {
                rule.risk_category.value for rule in result.triggered_rules
            }
            levels, level_status = {}, {}
            for category, level in raw_levels.items():
                if not category_missing[category]:
                    levels[category] = level
                    level_status[category] = "valid"
                elif category in triggered_categories and level != "L0":
                    levels[category] = level
                    level_status[category] = "lower_bound_due_to_missing_data"
                else:
                    levels[category] = None
                    level_status[category] = "insufficient_data"
            valid_levels = [v for v in levels.values() if v is not None]
            levels["overall"] = (
                max(valid_levels, key=lambda value: int(value[1:]))
                if valid_levels else None
            )
            level_status["overall"] = (
                "valid" if all(value == "valid" for value in level_status.values())
                else "lower_bound_due_to_partial_categories" if valid_levels
                else "insufficient_data"
            )
            samples.append({
                "frame_index": i,
                "time_s": i * _dt_seconds(raw),
                "features": feature_dict,
                "risk_levels": levels,
                "risk_level_status": level_status,
                "rule_engine_levels": {**raw_levels, "overall": result.overall_level.value},
                "triggered_rules": [r.model_dump(mode="json") for r in result.triggered_rules],
                "root_cause": result.root_cause,
                "data_quality": result.data_quality.value,
                "missing_fields": result.missing_fields,
                "not_applicable_fields": sorted(not_applicable),
            })
        return {
            "metadata": {"source": "TemporalRiskEvaluator", "version": "1.0", "raw_gt_episode_id": raw.get("episode_meta", {}).get("episode_id"), "sampling_interval_frames": self.interval_frames, "total_frames": n, "sample_count": len(samples), "sampling_policy": "every interval frames plus final frame", "units": "SI (m, m/s, s, N, N.s, rad, dimensionless)", "generated_at": datetime.now(timezone.utc).isoformat()},
            "samples": samples,
        }


def extract_temporal(raw_gt: Dict[str, Any], interval_frames: int = 10) -> Dict[str, Any]:
    return TemporalRiskEvaluator(interval_frames).evaluate(raw_gt)


def extract_and_save(raw_gt_path: str, output_path: Optional[str] = None, interval_frames: int = 10) -> str:
    """Offline helper for re-evaluating an existing Sim_Raw_GT file."""
    import json
    import os

    with open(raw_gt_path, "r", encoding="utf-8") as handle:
        report = extract_temporal(json.load(handle), interval_frames)
    output_path = output_path or os.path.join(os.path.dirname(raw_gt_path), "sim_labels_timeline.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample HS/PT/RS/IR rules over Sim_Raw_GT frames")
    parser.add_argument("raw_gt_path")
    parser.add_argument("-o", "--output")
    parser.add_argument("--interval-frames", type=int, default=10)
    args = parser.parse_args()
    print(extract_and_save(args.raw_gt_path, args.output, args.interval_frames))
