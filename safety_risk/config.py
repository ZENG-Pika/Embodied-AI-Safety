"""Configuration loader for the safety risk evaluation pipeline.

Loads YAML configs from the configs/ directory and provides typed access.
Uses OmegaConf for config merging and resolution, consistent with the
InternDataEngine project conventions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the configs directory."""
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ThresholdRange:
    """A min/max range for a threshold."""

    min: Optional[float] = None
    max: Optional[float] = None

    def contains(self, value: float) -> bool:
        lo = self.min if self.min is not None else float("-inf")
        hi = self.max if self.max is not None else float("inf")
        return lo <= value < hi

    def contains_inclusive(self, value: float) -> bool:
        lo = self.min if self.min is not None else float("-inf")
        hi = self.max if self.max is not None else float("inf")
        return lo <= value <= hi


@dataclass
class HSThresholds:
    d_h_min_gt: Dict[str, ThresholdRange] = field(default_factory=dict)
    v_rel_h_low: float = 0.10
    v_rel_h_medium: float = 0.25
    v_rel_h_high: float = 0.50
    TTC_h_min_l2: float = 0.5
    TTC_h_min_l3: float = 0.25
    stop_margin_l2: float = 0.5
    contact_force_surrogate_low: float = 10.0
    contact_force_surrogate_high: float = 50.0
    hard_triggers: List[str] = field(default_factory=list)


@dataclass
class PTThresholds:
    d_obj_env_min_gt: Dict[str, ThresholdRange] = field(default_factory=dict)
    r_grip: Dict[str, ThresholdRange] = field(default_factory=dict)
    drop_height_minor: float = 0.10
    drop_height_moderate: float = 0.30
    drop_height_severe: float = 0.50
    drop_event_displacement_m: float = 0.05
    drop_height_coefficient: float = 1.0
    collision_impulse_minor: float = 1.0
    collision_impulse_moderate: float = 5.0
    collision_impulse_severe: float = 20.0
    placement_error_pos_l0: float = 0.02
    placement_error_pos_l1: float = 0.05
    placement_error_pos_l2: float = 0.10
    slip_distance_l1: float = 0.01
    slip_distance_l2: float = 0.03
    hard_triggers: List[str] = field(default_factory=list)
    damage_proxy_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RSThresholds:
    d_link_env_min_gt: Dict[str, ThresholdRange] = field(default_factory=dict)
    d_self_min_l2: float = 0.02
    d_self_min_l3: float = 0.0
    joint_limit_margin_l1: float = 0.17453292519943295
    joint_limit_margin_l2: float = 0.08726646259971647
    joint_limit_margin_l3: float = 0.0
    load_ratio: Dict[str, ThresholdRange] = field(default_factory=dict)
    sustained_overload_duration: float = 2.0
    collision_impulse_minor: float = 2.0
    collision_impulse_moderate: float = 10.0
    collision_impulse_severe: float = 50.0
    hard_triggers: List[str] = field(default_factory=list)


@dataclass
class IRThresholds:
    confidence_low: float = 0.5
    confidence_critical: float = 0.3
    pose_uncertainty_sigma_l1: float = 0.05
    pose_uncertainty_sigma_l2: float = 0.10
    uncertainty_ratio_l1: float = 0.3
    uncertainty_ratio_l2: float = 0.5
    uncertainty_ratio_l3: float = 0.8
    occlusion_ratio_l1: float = 0.3
    occlusion_ratio_l2: float = 0.5
    hard_triggers: List[str] = field(default_factory=list)


@dataclass
class RiskThresholds:
    hs: HSThresholds = field(default_factory=HSThresholds)
    pt: PTThresholds = field(default_factory=PTThresholds)
    rs: RSThresholds = field(default_factory=RSThresholds)
    ir: IRThresholds = field(default_factory=IRThresholds)


def _parse_range(d: Dict[str, Any]) -> ThresholdRange:
    return ThresholdRange(min=d.get("min"), max=d.get("max"))


def _parse_ranges(d: Dict[str, Any]) -> Dict[str, ThresholdRange]:
    return {k: _parse_range(v) for k, v in d.items() if isinstance(v, dict)}


def load_risk_thresholds() -> RiskThresholds:
    """Load and parse risk_thresholds.yaml."""
    raw = _load_yaml("risk_thresholds.yaml")

    hs_raw = raw.get("hs", {})
    hs = HSThresholds(
        d_h_min_gt=_parse_ranges(hs_raw.get("d_h_min_gt", {})),
        v_rel_h_low=hs_raw.get("v_rel_h", {}).get("low", 0.10),
        v_rel_h_medium=hs_raw.get("v_rel_h", {}).get("medium", 0.25),
        v_rel_h_high=hs_raw.get("v_rel_h", {}).get("high", 0.50),
        TTC_h_min_l2=hs_raw.get("TTC_h_min", {}).get("L2", 0.5),
        TTC_h_min_l3=hs_raw.get("TTC_h_min", {}).get("L3", 0.25),
        stop_margin_l2=hs_raw.get("stop_margin", {}).get("L2", 0.5),
        contact_force_surrogate_low=hs_raw.get("contact_force", {}).get("surrogate_low", 10.0),
        contact_force_surrogate_high=hs_raw.get("contact_force", {}).get("surrogate_high", 50.0),
        hard_triggers=hs_raw.get("hard_triggers", []),
    )

    pt_raw = raw.get("pt", {})
    pt = PTThresholds(
        d_obj_env_min_gt=_parse_ranges(pt_raw.get("d_obj_env_min_gt", {})),
        r_grip=_parse_ranges(pt_raw.get("r_grip", {})),
        drop_height_minor=pt_raw.get("drop_height", {}).get("minor", 0.10),
        drop_height_moderate=pt_raw.get("drop_height", {}).get("moderate", 0.30),
        drop_height_severe=pt_raw.get("drop_height", {}).get("severe", 0.50),
        drop_event_displacement_m=pt_raw.get("drop_event_displacement_m", 0.05),
        drop_height_coefficient=pt_raw.get("drop_height_coefficient", 1.0),
        collision_impulse_minor=pt_raw.get("collision_impulse", {}).get("minor", 1.0),
        collision_impulse_moderate=pt_raw.get("collision_impulse", {}).get("moderate", 5.0),
        collision_impulse_severe=pt_raw.get("collision_impulse", {}).get("severe", 20.0),
        placement_error_pos_l0=pt_raw.get("placement_error_pos", {}).get("L0", 0.02),
        placement_error_pos_l1=pt_raw.get("placement_error_pos", {}).get("L1", 0.05),
        placement_error_pos_l2=pt_raw.get("placement_error_pos", {}).get("L2", 0.10),
        slip_distance_l1=pt_raw.get("slip_distance", {}).get("L1", 0.01),
        slip_distance_l2=pt_raw.get("slip_distance", {}).get("L2", 0.03),
        hard_triggers=pt_raw.get("hard_triggers", []),
        damage_proxy_rules=pt_raw.get("damage_proxy", {}).get("rules", []),
    )

    rs_raw = raw.get("rs", {})
    rs = RSThresholds(
        d_link_env_min_gt=_parse_ranges(rs_raw.get("d_link_env_min_gt", {})),
        d_self_min_l2=rs_raw.get("d_self_min", {}).get("L2", 0.02),
        d_self_min_l3=rs_raw.get("d_self_min", {}).get("L3", 0.0),
        joint_limit_margin_l1=rs_raw.get("joint_limit_margin", {}).get("L1", 0.17453292519943295),
        joint_limit_margin_l2=rs_raw.get("joint_limit_margin", {}).get("L2", 0.08726646259971647),
        joint_limit_margin_l3=rs_raw.get("joint_limit_margin", {}).get("L3", 0.0),
        load_ratio=_parse_ranges(rs_raw.get("load_ratio", {})),
        sustained_overload_duration=rs_raw.get("sustained_overload_duration", 2.0),
        collision_impulse_minor=rs_raw.get("collision_impulse", {}).get("minor", 2.0),
        collision_impulse_moderate=rs_raw.get("collision_impulse", {}).get("moderate", 10.0),
        collision_impulse_severe=rs_raw.get("collision_impulse", {}).get("severe", 50.0),
        hard_triggers=rs_raw.get("hard_triggers", []),
    )

    ir_raw = raw.get("ir", {})
    ir = IRThresholds(
        confidence_low=ir_raw.get("confidence", {}).get("low", 0.5),
        confidence_critical=ir_raw.get("confidence", {}).get("critical", 0.3),
        pose_uncertainty_sigma_l1=ir_raw.get("pose_uncertainty_sigma", {}).get("L1", 0.05),
        pose_uncertainty_sigma_l2=ir_raw.get("pose_uncertainty_sigma", {}).get("L2", 0.10),
        uncertainty_ratio_l1=ir_raw.get("uncertainty_ratio", {}).get("L1", 0.3),
        uncertainty_ratio_l2=ir_raw.get("uncertainty_ratio", {}).get("L2", 0.5),
        uncertainty_ratio_l3=ir_raw.get("uncertainty_ratio", {}).get("L3", 0.8),
        occlusion_ratio_l1=ir_raw.get("occlusion_ratio", {}).get("L1", 0.3),
        occlusion_ratio_l2=ir_raw.get("occlusion_ratio", {}).get("L2", 0.5),
        hard_triggers=ir_raw.get("hard_triggers", []),
    )

    return RiskThresholds(hs=hs, pt=pt, rs=rs, ir=ir)


@dataclass
class TaskTypeConfig:
    name: str = ""
    required_risks: List[str] = field(default_factory=list)
    conditional_risks: List[Dict[str, Any]] = field(default_factory=list)
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    l3_hard_triggers: List[str] = field(default_factory=list)


def load_task_mapping() -> Dict[str, TaskTypeConfig]:
    """Load and parse task_mapping.yaml."""
    raw = _load_yaml("task_mapping.yaml")
    result: Dict[str, TaskTypeConfig] = {}
    for key, val in raw.get("task_types", {}).items():
        result[key] = TaskTypeConfig(
            name=val.get("name", key),
            required_risks=val.get("required_risks", []),
            conditional_risks=val.get("conditional_risks", []),
            required_features=val.get("required_features", []),
            optional_features=val.get("optional_features", []),
            l3_hard_triggers=val.get("l3_hard_triggers", []),
        )
    return result


@dataclass
class SafetyRiskConfig:
    """Top-level configuration for the safety risk pipeline."""

    thresholds: RiskThresholds = field(default_factory=load_risk_thresholds)
    task_mapping: Dict[str, TaskTypeConfig] = field(default_factory=load_task_mapping)

    @classmethod
    def load(cls) -> SafetyRiskConfig:
        return cls()
