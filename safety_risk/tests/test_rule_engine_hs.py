from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import HSFeatures, RiskFeatures, RiskLevel


def level(**kwargs):
    return RuleBasedRiskEngine().evaluate(RiskFeatures(hs=HSFeatures(**kwargs))).hs_level


def test_hs_l3_single_and_combination_rules():
    assert level(F_h_peak_gt_N=50.01) == RiskLevel.L3
    assert level(TTC_h_min_gt_s=0.249) == RiskLevel.L3
    assert level(d_robot_h_min_gt_m=0.049, v_rel_h_gt_mps=0.251) == RiskLevel.L3
    assert level(F_h_peak_gt_N=10.01, contact_duration_h_gt_s=0.5) == RiskLevel.L3
    assert level(human_contact_flag_gt=True) == RiskLevel.L3


def test_hs_contact_false_does_not_trigger_contact_rule():
    assert level(human_contact_flag_gt=False) == RiskLevel.L0


def test_hs_l2_l1_l0_boundaries():
    assert level(F_h_peak_gt_N=10.0) == RiskLevel.L0
    assert level(F_h_peak_gt_N=50.0) == RiskLevel.L0
    assert level(d_ee_h_min_gt_m=0.05) == RiskLevel.L2
    assert level(TTC_h_min_gt_s=0.25) == RiskLevel.L2
    assert level(TTC_h_min_gt_s=1.0, human_contact_flag_gt=False) == RiskLevel.L1
    assert level(TTC_h_min_gt_s=2.0, human_contact_flag_gt=False) == RiskLevel.L0
    assert level(d_obj_h_min_gt_m=0.10, human_contact_flag_gt=False) == RiskLevel.L1
    assert level(d_obj_h_min_gt_m=0.15, human_contact_flag_gt=False) == RiskLevel.L0
    assert level(d_obj_h_min_gt_m=0.049, v_rel_h_gt_mps=0.0) == RiskLevel.L0
    assert level(d_obj_h_min_gt_m=0.049, v_rel_h_gt_mps=0.10) == RiskLevel.L2
    assert level(d_obj_h_min_gt_m=0.049, v_rel_h_gt_mps=0.25) == RiskLevel.L2
    assert level(d_obj_h_min_gt_m=0.049, v_rel_h_gt_mps=0.250001) == RiskLevel.L3
    assert level(F_h_peak_gt_N=10.0, contact_duration_h_gt_s=0.5) == RiskLevel.L0
    assert level(d_robot_h_min_gt_m=0.15, v_rel_h_gt_mps=0.10, TTC_h_min_gt_s=2.0) == RiskLevel.L0


def test_hs_level_priority_and_same_level_or():
    assert level(d_robot_h_min_gt_m=0.10, human_contact_flag_gt=False,
                 TTC_h_min_gt_s=0.25) == RiskLevel.L2
    assert level(d_robot_h_min_gt_m=0.10, human_contact_flag_gt=True,
                 TTC_h_min_gt_s=0.25) == RiskLevel.L3
