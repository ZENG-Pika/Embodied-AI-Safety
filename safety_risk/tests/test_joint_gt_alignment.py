import pytest

from safety_risk.raw_gt_extractor import SimRawGTExtractor


ARM_INDICES = [12, 14, 16, 18, 20, 22, 13, 15, 17, 19, 21, 23]
ARM_NAMES = [
    "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6",
    "fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6",
]


def test_arm_joint_channels_share_one_axis_and_timeline():
    velocity_0 = [float(index) for index in range(28)]
    velocity_1 = [float(index + 1) for index in range(28)]
    effort = [float(index * 2) for index in range(28)]
    raw = {
        "episode_meta": {"physics_config": {}},
        "robot_state": {
            "joint_position_q_gt": [[0.1] * 6, [0.2] * 6],
            "joint_position_q_right_gt": [[0.3] * 6, [0.4] * 6],
            "joint_velocity_dq_gt": [velocity_0, velocity_1],
            "joint_torque_gt": [effort, effort],
            "ee_pose_gt": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2,
            "ee_pose_right_gt": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2,
        },
        "planner_log": {},
    }
    limits = {
        str(index): {"limit_nm": 100.0}
        for index in ARM_INDICES
    }

    extractor = SimRawGTExtractor()
    extractor.normalize_arm_joint_state(
        raw,
        ARM_INDICES,
        ARM_NAMES,
        left_dof_count=6,
        dt=0.1,
        effort_limits_by_index=limits,
    )
    extractor._compute_executed_trajectory(raw)

    robot = raw["robot_state"]
    assert robot["joint_position_q_gt"][0] == [0.1] * 6 + [0.3] * 6
    assert robot["joint_velocity_dq_gt"][0] == [velocity_0[i] for i in ARM_INDICES]
    assert robot["joint_torque_gt"][0] == [effort[i] for i in ARM_INDICES]
    assert robot["joint_acceleration_gt"][0] == [None] * 12
    assert robot["joint_acceleration_gt"][1] == pytest.approx([10.0] * 12)
    assert robot["joint_state_metadata"]["dof_names"] == ARM_NAMES
    assert raw["episode_meta"]["physics_config"]["joint_effort_limits_nm"] == [100.0] * 12

    executed = raw["planner_log"]["executed_trajectory"]
    assert executed[0]["dof_names"] == ARM_NAMES[:6]
    assert executed[1]["dof_names"] == ARM_NAMES[6:]
    assert executed[0]["ee_pose_frame"] == "world"
    assert len(executed[0]["trajectory"]) == 2
    assert len(executed[1]["trajectory"]) == 2
