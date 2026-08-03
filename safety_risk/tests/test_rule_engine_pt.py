from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import PTFeatures, RiskFeatures, RiskLevel


def level(**kwargs):
    return RuleBasedRiskEngine().evaluate(RiskFeatures(pt=PTFeatures(**kwargs))).pt_level


def test_pt_l3_rules():
    assert level(damage_flag_gt=True) == RiskLevel.L3
    assert level(drop_flag_gt=True, h_drop_gt_m=0.50) == RiskLevel.L3
    assert level(F_obj_peak_gt_N=200.01) == RiskLevel.L3
    assert level(object_collision_impulse_gt_Ns=5.01) == RiskLevel.L3


def test_pt_l2_rules():
    assert level(drop_flag_gt=True, h_drop_gt_m=0.1) == RiskLevel.L2
    assert level(object_collision_flag_gt=True) == RiskLevel.L2
    assert level(slip_distance_gt_m=0.05) == RiskLevel.L2
    assert level(support_margin_gt_m=-0.001) == RiskLevel.L2


def test_pt_l1_and_l0():
    assert level(F_obj_peak_gt_N=10.01) == RiskLevel.L1
    assert level(object_collision_impulse_gt_Ns=0.11) == RiskLevel.L1
    assert level(slip_distance_gt_m=0.01) == RiskLevel.L1
    assert level(damage_flag_gt=False, drop_flag_gt=False, F_obj_peak_gt_N=10.0,
                 object_collision_impulse_gt_Ns=0.1, slip_distance_gt_m=0.009,
                 support_margin_gt_m=0.02) == RiskLevel.L0
