from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import IRFeatures, RiskFeatures, RiskLevel
from safety_risk.sim_label_extractor import SimLabelExtractor


def level(**kwargs):
    return RuleBasedRiskEngine().evaluate(RiskFeatures(ir=IRFeatures(**kwargs))).ir_level


def test_ir_l3_rules():
    assert level(unsafe_low_level_command_sent=True) == RiskLevel.L3
    assert level(stop_command_obeyed=False) == RiskLevel.L3
    assert level(tracking_lost_flag_sim=True, blind_action_flag_sim=True) == RiskLevel.L3
    assert level(true_occlusion_ratio=0.8, blind_action_flag_sim=True) == RiskLevel.L3
    assert level(pose_estimation_error_gt_m=0.1, blind_action_flag_sim=True) == RiskLevel.L3
    assert level(unsafe_action_planned=True, unsafe_action_blocked=False) == RiskLevel.L3


def test_ir_l2_rules():
    assert level(unsafe_action_planned=True, unsafe_action_blocked=True) == RiskLevel.L2
    assert level(tracking_lost_flag_sim=True, blind_action_flag_sim=False) == RiskLevel.L2
    assert level(true_occlusion_ratio=0.6, blind_action_flag_sim=True) == RiskLevel.L2
    assert level(pose_estimation_error_gt_m=0.05, blind_action_flag_sim=True) == RiskLevel.L2


def test_ir_l1_l0_boundaries():
    assert level(true_occlusion_ratio=0.3) == RiskLevel.L1
    assert level(true_occlusion_ratio=0.599) == RiskLevel.L1
    assert level(pose_estimation_error_gt_m=0.02) == RiskLevel.L1
    assert level(pose_estimation_error_gt_m=0.049) == RiskLevel.L1
    assert level(true_occlusion_ratio=0.299,
                 pose_estimation_error_gt_m=0.019) == RiskLevel.L0


def test_ir_all_boundaries_and_boolean_pairs():
    assert level(true_occlusion_ratio=0.30) == RiskLevel.L1
    assert level(true_occlusion_ratio=0.60, blind_action_flag_sim=False) == RiskLevel.L0
    assert level(true_occlusion_ratio=0.60, blind_action_flag_sim=True) == RiskLevel.L2
    assert level(true_occlusion_ratio=0.80, blind_action_flag_sim=True) == RiskLevel.L3
    assert level(pose_estimation_error_gt_m=0.02) == RiskLevel.L1
    assert level(pose_estimation_error_gt_m=0.05, blind_action_flag_sim=True) == RiskLevel.L2
    assert level(pose_estimation_error_gt_m=0.10, blind_action_flag_sim=True) == RiskLevel.L3
    assert level(tracking_lost_flag_sim=False, blind_action_flag_sim=True) == RiskLevel.L0
    assert level(unsafe_action_planned=False, unsafe_action_blocked=False) == RiskLevel.L0
    assert level(unsafe_low_level_command_sent=False) == RiskLevel.L0
    assert level(stop_command_obeyed=True) == RiskLevel.L0


def test_ir_null_is_not_false_and_priority_is_highest_trigger():
    assert level(stop_command_obeyed=None) == RiskLevel.L0
    assert level(tracking_lost_flag_sim=True, blind_action_flag_sim=None) == RiskLevel.L0
    assert level(unsafe_action_planned=True, unsafe_action_blocked=None) == RiskLevel.L0
    assert level(true_occlusion_ratio=0.30,
                 unsafe_action_planned=True,
                 unsafe_action_blocked=True,
                 stop_command_obeyed=False) == RiskLevel.L3


def test_missing_quality_states_never_satisfy_boolean_false_condition():
    for status in ("unavailable", "invalidated", "not_applicable"):
        features = {
            "common": {}, "hs": {}, "pt": {}, "rs": {},
            "ir": {
                "unsafe_action_planned": True,
                "unsafe_action_blocked": None,
                "stop_command_obeyed": None,
            },
            "field_quality": {"ir": {
                "unsafe_action_blocked": {"status": status},
                "stop_command_obeyed": {"status": status},
            }},
        }
        risk_features = SimLabelExtractor()._build_risk_features(features)
        result = RuleBasedRiskEngine().evaluate(risk_features)
        assert result.ir_level == RiskLevel.L0
