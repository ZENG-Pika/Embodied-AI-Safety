"""Pydantic schema models for the safety risk evaluation pipeline.

All field names, units, and enums are aligned with robot_safety_risk_data_contract.xlsx.
See configs/signal_schema.sim.yaml for the raw signal reference.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class RiskCategory(str, Enum):
    HS = "HS"  # Human Safety
    PT = "PT"  # Property / Object Damage
    RS = "RS"  # Robot Self-preservation
    IR = "IR"  # Input Robustness


class RequiredLevel(str, Enum):
    M0 = "M0"  # 必采/主评分
    M1 = "M1"  # 推荐采集/诊断
    M2 = "M2"  # 可选增强


class DataQuality(str, Enum):
    A = "A"  # 可信
    B = "B"  # 可评分
    C = "C"  # 需复核且不判 L0
    D = "D"  # 不进主评分


class GraspState(str, Enum):
    NOT_GRASPED = "not_grasped"
    GRASPED = "grasped"
    SLIPPING = "slipping"
    DROPPED = "dropped"


class DamageSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    FUNCTIONAL_DAMAGE = "functional_damage"
    BROKEN = "broken"
    LEAKAGE = "leakage"


class SafetyGateStatus(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    WARNING = "warning"
    DRY_RUN = "dry_run"
    SHADOW = "shadow"


class TaskSemanticSuccess(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"


class ScenarioRealism(str, Enum):
    VALID = "valid"
    UNREALISTIC = "unrealistic"
    INVALID_SETUP = "invalid_setup"


class AnnotationValidity(str, Enum):
    VALID = "valid"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


# ── Raw Episode Data Structures ──────────────────────────────────────────────


class EpisodeMeta(BaseModel):
    """Episode / scenario metadata (S-COM-*)."""

    episode_id: str
    scenario_id: str = ""
    random_seed: int = 0
    task_type: str = ""
    object_id: str = ""
    target_object_id: str = ""
    target_pose: Optional[List[float]] = None  # [x,y,z,qx,qy,qz,qw]
    object_hazard_class: str = "none"
    object_fragility_class: str = "none"
    physics_config: Optional[Dict[str, Any]] = None
    lighting_config: Optional[Dict[str, Any]] = None
    sensor_noise_config: Optional[Dict[str, Any]] = None


class RobotState(BaseModel):
    """Robot ground-truth state (S-ROBOT-*)."""

    # Time series: list of arrays, one per timestep
    joint_position_q: Optional[List[List[float]]] = None  # rad
    joint_velocity_dq: Optional[List[List[float]]] = None  # rad/s
    joint_acceleration: Optional[List[List[float]]] = None  # rad/s^2
    joint_torque: Optional[List[List[float]]] = None  # N·m
    joint_limit: Optional[Dict[str, Any]] = None  # {lower: [...], upper: [...]}
    link_pose: Optional[List[List[List[float]]]] = None  # [timestep][link][7]
    link_velocity: Optional[List[List[List[float]]]] = None  # [timestep][link][6]
    ee_pose: Optional[List[List[float]]] = None  # [timestep][7]
    ee_velocity: Optional[List[List[float]]] = None  # [timestep][6]
    robot_mode: Optional[str] = None
    controller_state: Optional[Dict[str, Any]] = None


class ObjectState(BaseModel):
    """Object ground-truth state (S-OBJ-*)."""

    # Keyed by object_id
    object_pose: Optional[Dict[str, List[List[float]]]] = None  # id -> [timestep][7]
    object_velocity: Optional[Dict[str, List[List[float]]]] = None  # id -> [timestep][3]
    object_angular_velocity: Optional[Dict[str, List[List[float]]]] = None  # id -> [timestep][3]
    object_physical_params: Optional[Dict[str, Any]] = None


class EnvironmentState(BaseModel):
    """Environment and human surrogate state (S-ENV-*, S-HUM-*)."""

    scene_mesh_ref: Optional[str] = None  # URI or path
    obstacle_pose: Optional[List[List[float]]] = None  # [obstacle][7]
    human_surrogate_pose: Optional[List[List[float]]] = None  # [timestep][7] or keypoints
    intrusion_trajectory: Optional[List[Dict[str, Any]]] = None  # [{t, pose}]
    table_boundary: Optional[Dict[str, Any]] = None
    support_surface: Optional[Dict[str, Any]] = None


class DistanceGT(BaseModel):
    """Precise distance ground truth (S-DIST-*)."""

    # Time series of distances (m)
    robot_human_distance_matrix: Optional[List[Any]] = None  # [timestep][link][bodypart]
    ee_human_distance: Optional[List[float]] = None  # [timestep]
    object_human_distance: Optional[List[float]] = None  # [timestep]
    object_env_distance: Optional[List[Any]] = None  # [timestep] or matrix
    link_env_distance: Optional[List[Any]] = None  # [timestep] or matrix
    self_distance: Optional[List[Any]] = None  # [timestep] or matrix


class CollisionGT(BaseModel):
    """Collision and contact ground truth (S-COLL-*)."""

    collision_pair: Optional[List[Dict[str, Any]]] = None  # [{bodyA, bodyB, time, ...}]
    collision_location: Optional[List[List[float]]] = None  # [[x,y,z], ...]
    penetration_depth: Optional[List[float]] = None  # m
    contact_force: Optional[List[Any]] = None  # [timestep] force vectors
    contact_impulse: Optional[List[float]] = None  # N·s per event
    contact_duration: Optional[List[float]] = None  # s per event
    contact_normal: Optional[List[List[float]]] = None  # [[nx,ny,nz], ...]


class GripperGT(BaseModel):
    """Gripper / grasp ground truth (S-GRASP-*)."""

    gripper_width: Optional[List[float]] = None  # m per timestep
    gripper_velocity: Optional[List[float]] = None  # m/s
    gripper_force: Optional[List[float]] = None  # N
    gripper_object_contact_force: Optional[List[float]] = None  # N per timestep
    grasp_state: Optional[List[str]] = None  # GraspState per timestep
    object_relative_pose_to_gripper: Optional[List[List[float]]] = None  # [timestep][7]
    slip_distance: Optional[List[float]] = None  # m per timestep


class OutcomeGT(BaseModel):
    """Drop / placement / damage ground truth (S-OUT-*)."""

    drop_event: bool = False
    drop_height: Optional[float] = None  # m
    final_object_pose: Optional[List[float]] = None  # [x,y,z,qx,qy,qz,qw]
    placement_error_pos: Optional[float] = None  # m
    placement_error_rot: Optional[float] = None  # rad
    stable_final: Optional[bool] = None
    damage_state: str = "none"  # DamageSeverity


class PlannerLog(BaseModel):
    """Planner / safety gate / controller logs (S-PLAN-*)."""

    planned_trajectory: Optional[List[Dict[str, Any]]] = None
    executed_trajectory: Optional[List[Dict[str, Any]]] = None
    replan_flag: bool = False
    t_replan_s: Optional[float] = None
    stop_command_sent: bool = False
    stop_success: Optional[bool] = None
    stop_margin_s: Optional[float] = None
    t_stop_s: Optional[float] = None
    safety_gate_status: str = "pass"  # SafetyGateStatus
    robot_motion_started: bool = False


class HRILog(BaseModel):
    """HRI / instruction logs (S-HRI-*)."""

    user_command_text: str = ""
    unsafe_instruction_flag: bool = False
    tool_call_trace: Optional[List[Dict[str, Any]]] = None
    model_response: str = ""
    refusal_flag: bool = False
    clarification_requested: bool = False
    stop_command_obeyed: Optional[bool] = None
    instruction_safety_assessment: Optional[Dict[str, Any]] = None


class SimRawEpisode(BaseModel):
    """Complete simulation raw episode with all GT signals.

    This is the input to the safety risk pipeline. Fields that the simulator
    cannot provide should be set to None; the feature extractor will record
    warnings for missing M0 fields.
    """

    episode_meta: EpisodeMeta
    robot_state: RobotState = Field(default_factory=RobotState)
    object_state: ObjectState = Field(default_factory=ObjectState)
    environment_state: EnvironmentState = Field(default_factory=EnvironmentState)
    distance_gt: DistanceGT = Field(default_factory=DistanceGT)
    collision_gt: CollisionGT = Field(default_factory=CollisionGT)
    gripper_gt: GripperGT = Field(default_factory=GripperGT)
    outcome_gt: OutcomeGT = Field(default_factory=OutcomeGT)
    planner_log: PlannerLog = Field(default_factory=PlannerLog)
    hri_log: HRILog = Field(default_factory=HRILog)
    perception_degradation_log: Optional[Dict[str, Any]] = None


# ── Risk Features ────────────────────────────────────────────────────────────


class CommonFeatures(BaseModel):
    """Common features shared across risk categories."""

    robot_active: bool = True
    data_quality: DataQuality = DataQuality.B
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class HSFeatures(BaseModel):
    """Human Safety risk features (SF-HS-*)."""

    d_robot_h_min_gt_m: Optional[float] = None
    d_ee_h_min_gt_m: Optional[float] = None
    d_obj_h_min_gt_m: Optional[float] = None
    v_rel_h_gt_mps: Optional[float] = None
    TTC_h_min_gt_s: Optional[float] = None
    # Internal diagnostics for the generic extractor; not emitted by the
    # minimal 31-field Sim_Features contract.
    time_d_h_below_0_15m_s: float = 0.0
    time_d_h_below_0_10m_s: float = 0.0
    time_d_h_below_0_05m_s: float = 0.0
    human_contact_flag_gt: bool = False
    F_h_peak_gt_N: float = 0.0
    contact_duration_h_gt_s: float = 0.0


class PTFeatures(BaseModel):
    """Property / Object Damage risk features (SF-PT-*)."""

    d_obj_env_min_gt_m: Optional[float] = None
    d_obj_env_eff_m: Optional[float] = None
    d_obj_edge_gt_m: Optional[float] = None
    d_obj_target_gt_m: Optional[float] = None
    object_collision_flag_gt: bool = False
    object_collision_impulse_gt_Ns: float = 0.0
    gripper_object_force_gt_N: float = 0.0
    F_obj_peak_gt_N: float = 0.0
    r_grip_gt: Optional[float] = None
    over_grip_flag: Optional[bool] = None
    grasp_success_flag: bool = False
    target_object_id: str = ""
    expected_object_id: str = ""
    wrong_object_flag_gt: Optional[bool] = None
    slip_flag_gt: bool = False
    slip_distance_gt_m: Optional[float] = None
    drop_flag_gt: bool = False
    h_drop_gt_m: Optional[float] = None
    placement_error_pos_gt_m: Optional[float] = None
    placement_error_rot_gt_rad: Optional[float] = None
    stable_final_gt: Optional[bool] = None
    damage_flag_gt: Optional[bool] = None
    damage_severity_gt: Optional[str] = None
    wrong_location_flag_gt: bool = False
    replan_flag: bool = False
    old_plan_continued_flag: bool = False
    manual_intervention_required: bool = False


class RSFeatures(BaseModel):
    """Robot Self-preservation risk features (SF-RS-*)."""

    d_link_env_min_gt_m: Optional[float] = None
    d_ee_env_min_gt_m: Optional[float] = None
    d_self_min_gt_m: Optional[float] = None
    d_link_env_eff_m: Optional[float] = None
    robot_env_collision_flag_gt: bool = False
    self_collision_flag_gt: bool = False
    robot_collision_impulse_gt_Ns: float = 0.0
    joint_limit_margin_gt_rad: Optional[float] = None
    joint_limit_violation: bool = False
    joint_torque_ratio_gt: Optional[float] = None
    joint_current_ratio_max: float = 0.0
    sustained_overload_gt: Optional[bool] = None
    protective_stop_flag: bool = False
    emergency_stop_flag: bool = False
    fault_code: Optional[str] = None
    manual_reset_required: bool = False
    unrecoverable_fault_flag: bool = False
    anomaly_detected_flag: bool = False
    safe_recovery_entered: bool = False
    motion_after_fault_gt: Optional[bool] = None
    recovery_retry_count: int = 0


class IRFeatures(BaseModel):
    """Input Robustness risk features (SF-IR-*)."""

    true_occlusion_ratio: Optional[float] = None
    reobserve_flag: bool = False
    degrade_mode_flag: bool = False
    blind_action_flag_sim: Optional[bool] = None
    ambiguous_command_flag: bool = False
    target_candidates_count: int = 0
    action_candidates_count: int = 0
    missing_constraint_flag: bool = False
    clarification_requested: bool = False
    confirmation_required: bool = False
    unsafe_execution_under_ambiguity: bool = False
    unsafe_instruction_flag_gt: bool = False
    unsafe_instruction_detected: bool = False
    refusal_flag: bool = False
    unsafe_action_planned: Optional[bool] = None
    robot_motion_started: bool = False
    multi_turn_attack_success: bool = False
    language_vision_conflict_flag: bool = False
    safety_state_conflict_flag: bool = False
    visual_prompt_attack_flag: bool = False
    unsafe_resolution_flag: bool = False
    stop_command_obeyed: Optional[bool] = None


class RiskFeatures(BaseModel):
    """Complete risk features extracted from a SimRawEpisode."""

    common: CommonFeatures = Field(default_factory=CommonFeatures)
    hs: HSFeatures = Field(default_factory=HSFeatures)
    pt: PTFeatures = Field(default_factory=PTFeatures)
    rs: RSFeatures = Field(default_factory=RSFeatures)
    ir: IRFeatures = Field(default_factory=IRFeatures)


# ── Risk Evaluation Result ───────────────────────────────────────────────────


class TriggeredRule(BaseModel):
    """A single triggered rule with evidence."""

    rule_id: str
    risk_category: RiskCategory
    level: RiskLevel
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RiskEvaluationResult(BaseModel):
    """Output of the rule-based risk engine."""

    episode_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    hs_level: RiskLevel = RiskLevel.L0
    pt_level: RiskLevel = RiskLevel.L0
    rs_level: RiskLevel = RiskLevel.L0
    ir_level: RiskLevel = RiskLevel.L0
    overall_level: RiskLevel = RiskLevel.L0

    triggered_rules: List[TriggeredRule] = Field(default_factory=list)
    root_cause: List[str] = Field(default_factory=list)

    data_quality: DataQuality = DataQuality.B
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    features: Optional[RiskFeatures] = None
