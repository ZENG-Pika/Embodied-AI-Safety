import pytest

from safety_risk.sim_feature_extractor import SimFeatureExtractor


def _raw_gt():
    return {
        "episode_meta": {
            "episode_id": "feature-test",
            "physics_config": {"physics_dt": "1/30"},
            "target_object_id": "pick_object_left",
            "object_id": "pick_object_left",
        },
        "robot_state": {
            "joint_position_q_gt": [[0.0] * 6, [0.1] * 6],
            "joint_position_q_right_gt": [[0.0] * 6, [0.2] * 6],
            "joint_torque_gt": [[1.0] * 12],
            "ee_pose_gt": [[0.0] * 7, [0.0] * 7],
        },
        "distance_gt": {
            "robot_human_distance_matrix_gt": [
                {"hand": {"link1": 0.20, "link2": 0.30}},
                {"hand": {"link1": 0.15, "link2": 0.25}},
            ],
            "ee_human_distance_gt": [
                {"left_hand": 0.30, "right_hand": 0.25},
                {"left_hand": 0.20, "right_hand": 0.24},
            ],
            "object_human_distance_gt": [{"object_hand": 0.40}],
            "object_env_distance_gt": [{"object": {"table": 0.06}}],
            "link_env_distance_gt": [{"link1→table": 0.04}],
            "self_distance_gt": [{"robot": {"a→b": 0.0, "a→c": 0.12}}],
        },
        "collision_gt": {
            "collision_pair_gt": [[
                {"bodyA": "robot/left", "bodyB": "object/pick_object_left"},
                {"bodyA": "robot/link", "bodyB": "environment/table"},
                {"bodyA": "robot/link", "bodyB": "obstacle/mano"},
                {"bodyA": "object/pick_object_left", "bodyB": "environment/table"},
                {"bodyA": "robot/left/link5", "bodyB": "robot/right/link5"},
            ]],
            "contact_force_gt": [[
                {"bodyA": "robot/left", "bodyB": "object/pick_object_left", "force_n": 10.0},
                {"bodyA": "robot/link", "bodyB": "environment/table", "force_n": 5.0},
                {"bodyA": "robot/link", "bodyB": "obstacle/mano", "force_n": 7.0},
                {"bodyA": "object/pick_object_left", "bodyB": "environment/table", "force_n": 3.0},
                {"bodyA": "robot/left/link5", "bodyB": "robot/right/link5", "force_n": 2.0},
            ]],
            "contact_duration_gt": [
                {"bodyA": "robot/link", "bodyB": "obstacle/mano", "duration_s": 0.2},
            ],
        },
        "gripper_gt": {
            "gripper_width_left": [[0.04], [0.02]],
            "gripper_width_right": [[0.04], [0.04]],
            "gripper_object_contact_force_gt": [
                {"left": 10.0, "right": 2.0},
            ],
            "slip_distance_gt": 0.01,
            "grasp_state_gt": ["not_grasped", "grasped"],
        },
        "outcome_gt": {
            "drop_event_gt": False,
            "drop_height_gt": None,
            "support_polygon_margin_gt": 0.03,
            "damage_state_gt": "none",
        },
        "planner_log": {},
        "environment_state": {},
        "sensor_gt": {},
        "hri_log": {"unsafe_instruction_flag_gt": False},
    }


def test_distance_units_sources_and_exact_dt():
    features = SimFeatureExtractor().extract(_raw_gt())
    assert features["hs"]["d_robot_h_min_gt_m"] == pytest.approx(0.15)
    assert features["hs"]["d_ee_h_min_gt_m"] == pytest.approx(0.20)
    assert features["hs"]["d_obj_h_min_gt_m"] == pytest.approx(0.40)
    assert features["hs"]["v_rel_h_gt_mps"] == pytest.approx(1.5)
    assert features["hs"]["TTC_h_min_gt_s"] == pytest.approx(0.2 / 1.5)
    assert features["pt"]["d_obj_env_min_gt_m"] == pytest.approx(0.06)
    assert features["rs"]["d_link_env_min_gt_m"] == pytest.approx(0.04)
    assert features["rs"]["d_self_min_gt_m"] == pytest.approx(0.12)


def test_per_pair_collision_features():
    features = SimFeatureExtractor().extract(_raw_gt())
    assert features["hs"]["human_contact_flag_gt"] is True
    assert features["hs"]["F_h_peak_gt_N"] == pytest.approx(7.0)
    assert features["hs"]["contact_duration_h_gt_s"] == pytest.approx(0.2)
    assert features["pt"]["F_obj_peak_gt_N"] == pytest.approx(10.0)
    assert features["pt"]["object_collision_flag_gt"] is True
    assert features["pt"]["object_collision_impulse_gt"] == pytest.approx(3.0 / 30.0)
    assert features["rs"]["robot_env_collision_flag_gt"] is True
    assert features["rs"]["self_collision_flag_gt"] is True
    assert features["rs"]["robot_collision_impulse_gt"] == pytest.approx(14.0 / 30.0)


def test_requested_feature_counts_include_false_and_zero():
    features = SimFeatureExtractor().extract(_raw_gt())
    metadata = features["metadata"]
    assert metadata["total_features"] == 49
    assert metadata["filled_features"] + metadata["null_features"] == 49
    assert features["ir"]["unsafe_instruction_flag_gt"] is False
    assert features["rs"]["motion_after_fault_gt"] is None
    assert features["rs"]["joint_torque_ratio_gt"] is None
