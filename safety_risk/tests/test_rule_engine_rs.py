from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import RSFeatures, RiskFeatures, RiskLevel


def level(**kwargs):
    return RuleBasedRiskEngine().evaluate(RiskFeatures(rs=RSFeatures(**kwargs))).rs_level


def test_rs_l3_rules():
    assert level(motion_after_fault_gt=True) == RiskLevel.L3
    assert level(sustained_overload_gt=True) == RiskLevel.L3
    assert level(joint_limit_margin_gt_rad=0.0) == RiskLevel.L3
    assert level(joint_torque_ratio_gt=1.201) == RiskLevel.L3
    assert level(robot_collision_impulse_gt_Ns=5.01) == RiskLevel.L3


def test_rs_l2_rules():
    assert level(robot_env_collision_flag_gt=True) == RiskLevel.L2
    assert level(self_collision_flag_gt=True) == RiskLevel.L2
    assert level(d_link_env_min_gt_m=0.049) == RiskLevel.L2
    assert level(d_self_min_gt_m=0.019) == RiskLevel.L2
    assert level(joint_torque_ratio_gt=1.0) == RiskLevel.L2
    assert level(joint_torque_ratio_gt=1.20) == RiskLevel.L2
    assert level(joint_limit_margin_gt_rad=0.087) == RiskLevel.L2
    assert level(robot_collision_impulse_gt_Ns=1.0) == RiskLevel.L2
    assert level(robot_collision_impulse_gt_Ns=5.0) == RiskLevel.L2


def test_rs_l1_and_l0():
    assert level(d_link_env_min_gt_m=0.05) == RiskLevel.L1
    assert level(joint_torque_ratio_gt=0.8) == RiskLevel.L1
    assert level(joint_limit_margin_gt_rad=0.175) == RiskLevel.L1
    assert level(d_link_env_min_gt_m=0.10, d_self_min_gt_m=0.05,
                 robot_collision_impulse_gt_Ns=0.1, joint_limit_margin_gt_rad=0.176,
                 joint_torque_ratio_gt=0.79, sustained_overload_gt=False,
                 motion_after_fault_gt=False) == RiskLevel.L0


def test_rs_all_boundaries_and_cancelled_small_impulse_rule():
    assert level(joint_torque_ratio_gt=0.8) == RiskLevel.L1
    assert level(joint_torque_ratio_gt=1.0) == RiskLevel.L2
    assert level(joint_torque_ratio_gt=1.20) == RiskLevel.L2
    assert level(joint_limit_margin_gt_rad=0.0) == RiskLevel.L3
    assert level(joint_limit_margin_gt_rad=0.087) == RiskLevel.L2
    assert level(joint_limit_margin_gt_rad=0.175) == RiskLevel.L1
    assert level(robot_collision_impulse_gt_Ns=0.1) == RiskLevel.L0
    assert level(robot_collision_impulse_gt_Ns=0.999999) == RiskLevel.L0
    assert level(robot_collision_impulse_gt_Ns=1.0) == RiskLevel.L2
    assert level(robot_collision_impulse_gt_Ns=2.0) == RiskLevel.L2
    assert level(robot_collision_impulse_gt_Ns=5.0) == RiskLevel.L2
    assert level(d_link_env_min_gt_m=0.0) == RiskLevel.L2
    assert level(d_link_env_min_gt_m=0.05) == RiskLevel.L1
    assert level(d_link_env_min_gt_m=0.10) == RiskLevel.L0
    assert level(d_self_min_gt_m=0.0) == RiskLevel.L2
    assert level(d_self_min_gt_m=0.02) == RiskLevel.L1
    assert level(d_self_min_gt_m=0.05) == RiskLevel.L0


def test_rs_collision_mapping_and_priority():
    assert level(robot_env_collision_flag_gt=True,
                 robot_collision_impulse_gt_Ns=2.000001) == RiskLevel.L3
    assert level(self_collision_flag_gt=True,
                 robot_collision_impulse_gt_Ns=2.000001) == RiskLevel.L3
    assert level(d_link_env_min_gt_m=0.05,
                 robot_env_collision_flag_gt=True) == RiskLevel.L2
