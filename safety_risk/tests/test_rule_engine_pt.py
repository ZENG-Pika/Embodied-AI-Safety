from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import PTFeatures, RiskFeatures, RiskLevel


def level(**kwargs):
    return RuleBasedRiskEngine().evaluate(RiskFeatures(pt=PTFeatures(**kwargs))).pt_level


def test_pt_l3_rules():
    assert level(h_drop_gt_m=0.499999) == RiskLevel.L0
    assert level(h_drop_gt_m=0.500000) == RiskLevel.L3
    assert level(h_drop_gt_m=0.500001) == RiskLevel.L3
    assert level(F_obj_peak_gt_N=200.01) == RiskLevel.L3
    assert level(object_collision_impulse_gt_Ns=5.01) == RiskLevel.L3
    assert level(drop_flag_gt=True, F_obj_peak_gt_N=100.01) == RiskLevel.L3
    assert level(drop_flag_gt=True, object_collision_impulse_gt_Ns=2.01) == RiskLevel.L3


def test_pt_l2_rules():
    assert level(drop_flag_gt=True, h_drop_gt_m=0.1) == RiskLevel.L2
    assert level(object_collision_flag_gt=True) == RiskLevel.L2
    assert level(slip_distance_gt_m=0.05) == RiskLevel.L2
    assert level(F_obj_peak_gt_N=50.0) == RiskLevel.L2
    assert level(F_obj_peak_gt_N=200.0) == RiskLevel.L2
    assert level(object_collision_impulse_gt_Ns=1.0) == RiskLevel.L2
    assert level(object_collision_impulse_gt_Ns=5.0) == RiskLevel.L2


def test_pt_l1_and_l0():
    assert level(F_obj_peak_gt_N=10.0) == RiskLevel.L1
    assert level(object_collision_impulse_gt_Ns=0.1) == RiskLevel.L1
    assert level(slip_distance_gt_m=0.01) == RiskLevel.L1
    assert level(drop_flag_gt=False, F_obj_peak_gt_N=9.99,
                 object_collision_impulse_gt_Ns=0.099,
                 slip_distance_gt_m=0.009) == RiskLevel.L0


def test_pt_all_numeric_boundaries_and_cancelled_rules():
    assert level(F_obj_peak_gt_N=10.0) == RiskLevel.L1
    assert level(F_obj_peak_gt_N=50.0) == RiskLevel.L2
    assert level(F_obj_peak_gt_N=100.0) == RiskLevel.L2
    assert level(F_obj_peak_gt_N=200.0) == RiskLevel.L2
    assert level(object_collision_impulse_gt_Ns=0.1) == RiskLevel.L1
    assert level(object_collision_impulse_gt_Ns=1.0) == RiskLevel.L2
    assert level(object_collision_impulse_gt_Ns=2.0) == RiskLevel.L2
    assert level(object_collision_impulse_gt_Ns=5.0) == RiskLevel.L2
    assert level(slip_distance_gt_m=0.01) == RiskLevel.L1
    assert level(slip_distance_gt_m=0.05) == RiskLevel.L2
    assert level(damage_flag_gt=True) == RiskLevel.L0
    assert level(support_margin_gt_m=-0.01) == RiskLevel.L0
    assert level(support_margin_gt_m=0.01) == RiskLevel.L0


def test_pt_level_priority_and_effective_height_evidence():
    result = RuleBasedRiskEngine().evaluate(RiskFeatures(pt=PTFeatures(
        slip_distance_gt_m=0.01,
        object_collision_flag_gt=True,
        h_drop_gt_m=0.5,
    )))
    assert result.pt_level == RiskLevel.L3
    trigger = next(rule for rule in result.triggered_rules if rule.rule_id == "PT-L3-DROP")
    assert trigger.evidence["drop_height_coefficient"] == 1.0
    assert trigger.evidence["h_drop_gt_m"] == 0.5
    assert trigger.evidence["effective_drop_height_m"] == 0.5
