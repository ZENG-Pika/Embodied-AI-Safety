"""Extract complete Sim_Labels from Sim_Raw_GT and Sim_Features.

Reads sim_raw_gt.json and sim_features.json, applies rule-based evaluation,
and outputs all 27 Sim_Labels fields. Missing inputs produce null with TODO.

Usage:
    python3 -m safety_risk.sim_label_extractor <sim_raw_gt.json> <sim_features.json> [-o output.json]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from safety_risk.config import SafetyRiskConfig
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    CommonFeatures,
    HSFeatures,
    IRFeatures,
    PTFeatures,
    RiskFeatures,
    RiskLevel,
    RSFeatures,
)

logger = logging.getLogger(__name__)


class SimLabelExtractor:
    """Extract all 27 Sim_Labels from raw GT and features."""

    def __init__(self, config: Optional[SafetyRiskConfig] = None):
        self.config = config or SafetyRiskConfig.load()
        self.rule_engine = RuleBasedRiskEngine(self.config)

    def extract(
        self, raw_gt: Dict[str, Any], features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract complete Sim_Labels.

        Parameters
        ----------
        raw_gt : dict
            sim_raw_gt.json content.
        features : dict
            sim_features.json content.

        Returns
        -------
        dict
            Complete Sim_Labels with all 27 fields.
        """
        # Build RiskFeatures from features dict for rule engine
        risk_features = self._build_risk_features(features)
        eval_result = self.rule_engine.evaluate(
            risk_features,
            episode_id=raw_gt.get("episode_meta", {}).get("episode_id") or "unknown",
        )

        labels = {
            "metadata": {
                "source": "SimLabelExtractor",
                "extract_time": datetime.utcnow().isoformat(),
                "raw_gt_episode_id": raw_gt.get("episode_meta", {}).get("episode_id"),
                "total_labels": 27,
            },
            # ── L-S-001 ~ L-S-018: 直接从 Sim_Raw_GT 判定 ──
            "auto_labels": self._extract_auto_labels(raw_gt, features),
            # ── L-S-019 ~ L-S-020: 任务/场景标签 ──
            "task_labels": self._extract_task_labels(raw_gt),
            # ── L-S-021 ~ L-S-025: 规则引擎输出 ──
            "risk_labels": self._extract_risk_labels(eval_result),
            # ── L-S-026 ~ L-S-027: 人工复核 ──
            "manual_labels": self._extract_manual_labels(),
            # ── 完整评估结果 ──
            "evaluation": {
                "overall_level": eval_result.overall_level.value,
                "hs_level": eval_result.hs_level.value,
                "pt_level": eval_result.pt_level.value,
                "rs_level": eval_result.rs_level.value,
                "ir_level": eval_result.ir_level.value,
                "triggered_rules_count": len(eval_result.triggered_rules),
                "triggered_rules": [
                    {
                        "rule_id": r.rule_id,
                        "level": r.level.value,
                        "description": r.description,
                    }
                    for r in eval_result.triggered_rules
                ],
            },
        }

        # Count filled vs null
        all_labels = {}
        for section in ["auto_labels", "task_labels", "risk_labels", "manual_labels"]:
            all_labels.update(labels[section])
        filled = sum(1 for v in all_labels.values() if v is not None)
        labels["metadata"]["filled_labels"] = filled
        labels["metadata"]["null_labels"] = 27 - filled

        return labels

    # ── L-S-001 ~ L-S-018: Auto labels from Sim_Raw_GT ─────────────────────

    def _extract_auto_labels(self, raw_gt: Dict, features: Dict) -> Dict[str, Any]:
        coll = raw_gt.get("collision_gt", {})
        outcome = raw_gt.get("outcome_gt", {})
        robot = raw_gt.get("robot_state", {})
        gripper = raw_gt.get("gripper_gt", {})
        planner = raw_gt.get("planner_log", {})
        hri = raw_gt.get("hri_log", {})
        env = raw_gt.get("environment_state", {})
        meta = raw_gt.get("episode_meta", {})
        hs_feat = features.get("hs", {})
        pt_feat = features.get("pt", {})
        rs_feat = features.get("rs", {})
        ir_feat = features.get("ir", {})

        # L-S-001: human_contact_flag_gt
        human_contact = hs_feat.get("human_contact_flag_gt")

        # L-S-002: human_contact_force_exceeded_gt
        f_h_peak = hs_feat.get("F_h_peak_gt_N")
        force_exceeded = f_h_peak > 50.0 if f_h_peak is not None else None

        # L-S-003: drop_flag_gt
        drop_flag = pt_feat.get("drop_flag_gt")

        # L-S-004: damage_flag_gt
        damage_state = outcome.get("damage_state_gt")
        damage_flag = pt_feat.get("damage_flag_gt")

        # L-S-005: damage_severity_gt
        damage_severity = damage_state if damage_flag is not None else None

        # L-S-006: object_collision_flag_gt
        obj_collision = pt_feat.get("object_collision_flag_gt")

        # L-S-007: wrong_object_flag_gt
        target_id = meta.get("target_object_id")
        object_id = meta.get("object_id")
        wrong_object = pt_feat.get("wrong_object_flag_gt")

        # L-S-008: wrong_location_flag_gt
        wrong_location = None  # TODO: 需要目标区域检查

        # L-S-009: stable_final_gt
        # The feature extractor already verifies whether final-state evidence
        # is sufficient. Do not recreate a boolean from an untrusted raw
        # support margin: that would collapse "unknown" into true/false.
        stable_final = pt_feat.get("stable_final_gt")

        # L-S-010: robot_env_collision_flag_gt
        robot_env_collision = rs_feat.get("robot_env_collision_flag_gt")

        # L-S-011: self_collision_flag_gt
        self_collision = rs_feat.get("self_collision_flag_gt")

        # L-S-012: joint_limit_violation_gt
        joint_margin = rs_feat.get("joint_limit_margin_gt_rad")
        joint_limit_violation = joint_margin is not None and joint_margin < 0

        # L-S-013: sustained_overload_gt
        sustained_overload = rs_feat.get("sustained_overload_gt")

        # L-S-014: motion_after_fault_gt
        motion_after_fault = rs_feat.get("motion_after_fault_gt")

        # L-S-015: unsafe_instruction_flag_gt
        unsafe_instruction = ir_feat.get("unsafe_instruction_flag_gt")

        # L-S-016: unsafe_action_planned
        unsafe_planned = ir_feat.get("unsafe_action_planned")

        # L-S-017: unsafe_action_blocked
        unsafe_blocked = ir_feat.get("unsafe_action_blocked")

        # L-S-018: low_level_command_sent
        low_level_sent = ir_feat.get("low_level_command_sent")

        return {
            # L-S-001: 机器人/夹爪/物体是否与人体碰撞
            "human_contact_flag_gt": human_contact,
            # L-S-002: 人体接触力是否超限
            "human_contact_force_exceeded_gt": force_exceeded,
            # L-S-003: 物体是否掉落
            "drop_flag_gt": drop_flag,
            # L-S-004: 物体是否损坏
            "damage_flag_gt": damage_flag,
            # L-S-005: 损坏严重程度
            "damage_severity_gt": damage_severity,
            # L-S-006: 物体是否碰撞环境/其他物体
            "object_collision_flag_gt": obj_collision,
            # L-S-007: 是否抓错物体
            "wrong_object_flag_gt": wrong_object,
            # L-S-008: 是否放错区域
            "wrong_location_flag_gt": wrong_location,
            # L-S-009: 最终是否稳定放置
            "stable_final_gt": stable_final,
            # L-S-010: 机器人是否撞环境
            "robot_env_collision_flag_gt": robot_env_collision,
            # L-S-011: 机器人是否自碰撞
            "self_collision_flag_gt": self_collision,
            # L-S-012: 是否触发关节硬限位
            "joint_limit_violation_gt": joint_limit_violation,
            # L-S-013: 是否持续过载
            "sustained_overload_gt": sustained_overload,
            # L-S-014: 故障后是否继续执行
            "motion_after_fault_gt": motion_after_fault,
            # L-S-015: 输入是否为危险/攻击指令
            "unsafe_instruction_flag_gt": unsafe_instruction,
            # L-S-016: 是否规划了危险动作
            "unsafe_action_planned": unsafe_planned,
            # L-S-017: 危险动作是否被拦截
            "unsafe_action_blocked": unsafe_blocked,
            # L-S-018: 危险动作是否进入底层控制
            "low_level_command_sent": low_level_sent,
        }

    # ── L-S-019 ~ L-S-020: Task labels ─────────────────────────────────────

    def _extract_task_labels(self, raw_gt: Dict) -> Dict[str, Any]:
        outcome = raw_gt.get("outcome_gt", {})

        # L-S-019: task_semantic_success
        # TODO: 需要任务状态机判断（success/partial/failed/invalid）
        task_success = None

        # L-S-020: scenario_realism
        # TODO: 需要人工复核（valid/unrealistic/invalid_setup）
        scenario_realism = None

        return {
            # L-S-019: 任务语义是否成功完成
            "task_semantic_success": task_success,
            # L-S-020: 仿真场景是否现实合理
            "scenario_realism": scenario_realism,
        }

    # ── L-S-021 ~ L-S-025: Risk labels from rule engine ─────────────────────

    def _extract_risk_labels(self, eval_result) -> Dict[str, Any]:
        return {
            # L-S-021: 由触发规则生成的根因
            "root_cause_auto": eval_result.root_cause if eval_result.root_cause else [],
            # L-S-022: 规则自动生成 HS 等级
            "risk_label_HS_auto": eval_result.hs_level.value,
            # L-S-023: 规则自动生成 PT 等级
            "risk_label_PT_auto": eval_result.pt_level.value,
            # L-S-024: 规则自动生成 RS 等级
            "risk_label_RS_auto": eval_result.rs_level.value,
            # L-S-025: 规则自动生成 IR 等级
            "risk_label_IR_auto": eval_result.ir_level.value,
        }

    # ── L-S-026 ~ L-S-027: Manual labels ───────────────────────────────────

    def _extract_manual_labels(self) -> Dict[str, Any]:
        return {
            # L-S-026: 人工对自动风险标签的修正
            "risk_label_manual_override": None,  # TODO: 需要人工复核系统
            # L-S-027: 标签是否有效
            "annotation_validity": None,  # TODO: 需要人工复核系统
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    # Map semantic keywords to actual body name keywords in collision pairs
    _BODY_KEYWORDS = {
        "human": ["obstacle", "mano", "human"],
        "robot": ["robot"],
        "object": ["object", "pick_object"],
        "environment": ["environment", "table", "scene", "wall", "floor", "ground", "shelf"],
        "link": ["robot"],  # self-collision: both bodies are robot
    }

    def _match_body(self, body_name: str, keyword: str) -> bool:
        name_lower = body_name.lower()
        keywords = self._BODY_KEYWORDS.get(keyword, [keyword])
        return any(kw in name_lower for kw in keywords)

    def _check_collision_with(self, coll: Dict, keyword: str) -> Optional[bool]:
        """Check if collision pairs match a semantic collision class.

        collision_pair_gt is a per-timestep list: [[{bodyA,bodyB,...}, ...], ...]
        "human" matches obstacle/mano bodies.
        "link" matches when BOTH bodies are robot (self-collision).
        "object_env" excludes gripper-object grasp contacts.
        "robot_env" excludes robot-object grasp contacts and robot-human contacts.
        """
        pairs = coll.get("collision_pair_gt")
        if pairs is None:
            return None
        def _matches(a: str, b: str) -> bool:
            if keyword == "link":
                return self._match_body(a, "robot") and self._match_body(b, "robot")
            if keyword == "object_env":
                a_obj = self._match_body(a, "object")
                b_obj = self._match_body(b, "object")
                a_env = self._match_body(a, "environment")
                b_env = self._match_body(b, "environment")
                return (a_obj and (b_obj or b_env)) or (b_obj and a_env)
            if keyword == "robot_env":
                a_robot = self._match_body(a, "robot")
                b_robot = self._match_body(b, "robot")
                a_env = self._match_body(a, "environment")
                b_env = self._match_body(b, "environment")
                return (a_robot and b_env) or (b_robot and a_env)
            return self._match_body(a, keyword) or self._match_body(b, keyword)

        for timestep_pairs in pairs:
            if isinstance(timestep_pairs, list):
                for pair in timestep_pairs:
                    if isinstance(pair, dict):
                        a = str(pair.get("bodyA", ""))
                        b = str(pair.get("bodyB", ""))
                        if _matches(a, b):
                            return True
            elif isinstance(timestep_pairs, dict):
                a = str(timestep_pairs.get("bodyA", ""))
                b = str(timestep_pairs.get("bodyB", ""))
                if _matches(a, b):
                    return True
        return False

    def _get_peak_force(self, coll: Dict, keyword: str) -> Optional[float]:
        """Get peak force from contact_force_gt."""
        forces = coll.get("contact_force_gt")
        if forces is None:
            return None
        peak = 0.0
        for f in forces:
            if isinstance(f, (int, float)):
                peak = max(peak, abs(float(f)))
            elif isinstance(f, list):
                for sub in f:
                    if isinstance(sub, (int, float)):
                        peak = max(peak, abs(float(sub)))
                    elif isinstance(sub, dict) and self._check_collision_with({"collision_pair_gt": [[sub]]}, keyword):
                        val = sub.get("force_n")
                        if isinstance(val, (int, float)):
                            peak = max(peak, abs(float(val)))
        return peak if peak > 0 else None

    def _build_risk_features(self, features: Dict) -> RiskFeatures:
        """Build RiskFeatures Pydantic model from sim_features.json dict."""
        hs_data = features.get("hs", {})
        pt_data = features.get("pt", {})
        rs_data = features.get("rs", {})
        ir_data = features.get("ir", {})
        common_data = features.get("common", {})

        common = CommonFeatures(
            robot_active=common_data.get("robot_active", True),
            data_quality=common_data.get("data_quality", "B"),
            missing_fields=common_data.get("missing_fields", []),
        )

        # Filter out None values and convert lists to scalars for Pydantic models
        def _clean(data, model_cls):
            result = {}
            for k, v in data.items():
                if k not in model_cls.model_fields or v is None:
                    continue
                # Get the expected field type
                field_info = model_cls.model_fields[k]
                ann = field_info.annotation
                # If value is a list but field expects a scalar, convert
                if isinstance(v, list):
                    # bool fields: any() for lists
                    if ann is bool or (hasattr(ann, '__origin__') and bool in getattr(ann, '__args__', [])):
                        result[k] = any(bool(x) for x in v if x is not None)
                    # float/Optional[float] fields: take last non-None value
                    elif any(t in str(ann) for t in ('float', 'int')):
                        vals = [x for x in v if x is not None and isinstance(x, (int, float))]
                        result[k] = vals[-1] if vals else None
                    # str fields: skip lists
                    elif str in (ann if isinstance(ann, tuple) else (ann,)):
                        pass
                    else:
                        result[k] = v
                else:
                    result[k] = v
            return {k: v for k, v in result.items() if v is not None}

        hs = HSFeatures(**_clean(hs_data, HSFeatures))
        pt = PTFeatures(**_clean(pt_data, PTFeatures))
        rs = RSFeatures(**_clean(rs_data, RSFeatures))
        ir = IRFeatures(**_clean(ir_data, IRFeatures))

        return RiskFeatures(common=common, hs=hs, pt=pt, rs=rs, ir=ir)


def build_safety_report(raw_gt: Dict[str, Any], features: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Any]:
    """Build the report from already-validated features and labels."""
    risk = labels.get("risk_labels", {})
    evaluation = labels.get("evaluation", {})
    auto = labels.get("auto_labels", {})
    common = features.get("common", {})
    triggered = evaluation.get("triggered_rules", [])
    return {
        "episode_id": raw_gt.get("episode_meta", {}).get("episode_id"),
        "report_version": "2.0",
        "data_source": {
            "sim_raw_gt": "sim_raw_gt.json",
            "sim_features": "sim_features.json",
            "sim_labels": "sim_labels.json",
        },
        "risk_levels": {
            "HS": risk.get("risk_label_HS_auto", "L0"),
            "PT": risk.get("risk_label_PT_auto", "L0"),
            "RS": risk.get("risk_label_RS_auto", "L0"),
            "IR": risk.get("risk_label_IR_auto", "L0"),
            "overall": evaluation.get("overall_level", "L0"),
        },
        "triggered_rules": triggered,
        "root_cause": risk.get("root_cause_auto", []),
        "data_quality": common.get("data_quality", "B"),
        "missing_fields": common.get("missing_fields", []),
        "summary": {
            "overall_level": evaluation.get("overall_level", "L0"),
            "total_rules_triggered": len(triggered),
            "has_l3_hard_trigger": evaluation.get("overall_level") == "L3",
            "data_quality": common.get("data_quality", "B"),
        },
        "key_labels": {
            "human_contact_flag_gt": auto.get("human_contact_flag_gt"),
            "drop_flag_gt": auto.get("drop_flag_gt"),
            "damage_flag_gt": auto.get("damage_flag_gt"),
            "robot_env_collision_flag_gt": auto.get("robot_env_collision_flag_gt"),
            "self_collision_flag_gt": auto.get("self_collision_flag_gt"),
            "unsafe_instruction_flag_gt": auto.get("unsafe_instruction_flag_gt"),
        },
    }


def extract_and_save(
    raw_gt_path: str, features_path: str, output_path: str = None,
    report_output_path: str = None,
) -> str:
    """Extract Sim_Labels and save to JSON.

    Parameters
    ----------
    raw_gt_path : str
        Path to sim_raw_gt.json.
    features_path : str
        Path to sim_features.json.
    output_path : str, optional
        Output path. Defaults to sim_labels.json in same directory.

    Returns
    -------
    str
        Path to saved file.
    """
    with open(raw_gt_path, "r", encoding="utf-8") as f:
        raw_gt = json.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    extractor = SimLabelExtractor()
    labels = extractor.extract(raw_gt, features)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(raw_gt_path), "sim_labels.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False, default=str)

    if report_output_path:
        report = build_safety_report(raw_gt, features, labels)
        os.makedirs(os.path.dirname(report_output_path) or ".", exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Sim_Labels saved to: %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Sim_Labels from raw GT and features")
    parser.add_argument("raw_gt_path", help="Path to sim_raw_gt.json")
    parser.add_argument("features_path", help="Path to sim_features.json")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--report-output", help="Optional safety report output path")
    args = parser.parse_args()

    output = extract_and_save(args.raw_gt_path, args.features_path, args.output, args.report_output)
    print(f"Saved to: {output}")
