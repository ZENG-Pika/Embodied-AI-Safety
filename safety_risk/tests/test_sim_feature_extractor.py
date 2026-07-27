import pytest

from safety_risk.sim_feature_extractor import REQUESTED_FEATURES, SimFeatureExtractor
from safety_risk.sim_label_extractor import SimLabelExtractor


def _raw_gt():
    return {
        "episode_meta": {
            "episode_id": "feature-test",
            "physics_config": {
                "physics_dt": "1/30",
                "gripper_max_width_m_by_arm": {"left": 0.10, "right": 0.10},
            },
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
            "_provenance": {
                "robot_human_distance_matrix_gt": {"metric": "surface_clearance"},
                "ee_human_distance_gt": {"metric": "surface_clearance"},
                "object_human_distance_gt": {"metric": "surface_clearance"},
                "object_env_distance_gt": {"metric": "surface_clearance"},
                "link_env_distance_gt": {"metric": "surface_clearance"},
                "self_distance_gt": {"metric": "surface_clearance"},
            },
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
            "_provenance": {
                "coverage": {
                    "human": "complete",
                    "robot_human": "complete",
                    "ee_human": "complete",
                    "object_human": "complete",
                    "object_env": "complete",
                    "robot_env": "complete",
                    "self": "complete",
                },
            },
            "collision_pair_gt": [[
                {"bodyA": "robot/left", "bodyB": "object/pick_object_left"},
                {"bodyA": "robot/link", "bodyB": "environment/table"},
                {"bodyA": "robot/left/link6", "bodyB": "obstacle/mano"},
                {"bodyA": "object/pick_object_left", "bodyB": "environment/table"},
                {"bodyA": "robot/left/link5", "bodyB": "robot/right/link5"},
            ]],
            "contact_force_gt": [[
                {"bodyA": "robot/left", "bodyB": "object/pick_object_left", "force_n": 10.0},
                {"bodyA": "robot/link", "bodyB": "environment/table", "force_n": 5.0},
                {"bodyA": "robot/left/link6", "bodyB": "obstacle/mano", "force_n": 7.0},
                {"bodyA": "object/pick_object_left", "bodyB": "environment/table", "force_n": 3.0},
                {"bodyA": "robot/left/link5", "bodyB": "robot/right/link5", "force_n": 2.0},
            ]],
            "contact_impulse_gt": [[
                {"bodyA": "robot/left", "bodyB": "object/pick_object_left", "impulse_ns": 0.1},
                {"bodyA": "robot/link", "bodyB": "environment/table", "impulse_ns": 0.2},
                {"bodyA": "robot/left/link6", "bodyB": "obstacle/mano", "impulse_ns": 0.3},
                {"bodyA": "object/pick_object_left", "bodyB": "environment/table", "impulse_ns": 0.4},
                {"bodyA": "robot/left/link5", "bodyB": "robot/right/link5", "impulse_ns": 0.5},
            ]],
            "contact_duration_gt": [
                {"bodyA": "robot/left/link6", "bodyB": "obstacle/mano", "duration_s": 0.2},
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
        "object_state": {},
        "planner_log": {},
        "environment_state": {"support_surface": "table"},
        "sensor_gt": {},
        "hri_log": {"unsafe_instruction_flag_gt": False},
    }


def test_distance_units_sources_and_exact_dt():
    raw = _raw_gt()
    raw["collision_gt"]["collision_pair_gt"] = [[]]
    features = SimFeatureExtractor().extract(raw)
    assert features["hs"]["d_robot_h_min_gt_m"] == pytest.approx(0.15)
    assert features["hs"]["d_ee_h_min_gt_m"] == pytest.approx(0.20)
    assert features["hs"]["d_obj_h_min_gt_m"] == pytest.approx(0.40)
    assert features["hs"]["v_rel_h_gt_mps"] == pytest.approx(1.5)
    assert features["hs"]["TTC_h_min_gt_s"] == pytest.approx(0.2 / 1.5)
    assert features["pt"]["d_obj_env_min_gt_m"] == pytest.approx(0.06)
    assert features["rs"]["d_link_env_min_gt_m"] == pytest.approx(0.04)
    assert features["rs"]["d_self_min_gt_m"] == pytest.approx(0.12)
    assert features["hs"]["d_robot_h_min_gt_cm"] == pytest.approx(15.0)
    assert features["hs"]["d_ee_h_min_gt_cm"] == pytest.approx(20.0)
    assert features["pt"]["d_obj_env_min_gt_cm"] == pytest.approx(6.0)
    assert features["rs"]["d_link_env_min_gt_cm"] == pytest.approx(4.0)
    assert features["rs"]["d_self_min_gt_cm"] == pytest.approx(12.0)


def test_relative_velocity_prefers_physx_velocity_projection():
    raw = _raw_gt()
    raw["collision_gt"]["collision_pair_gt"] = [[], []]
    link_name = "robot/fl/link6"
    raw["robot_state"]["link_pose_gt"] = [
        {"robot": {link_name: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}},
        {"robot": {link_name: [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}},
    ]
    raw["robot_state"]["link_velocity_gt"] = [
        {"robot": {link_name: [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}},
        {"robot": {link_name: [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]}},
    ]
    raw["environment_state"]["obstacle_pose_gt"] = {
        "hand": {"translation": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
    }
    features = SimFeatureExtractor().extract(raw)
    assert features["hs"]["v_rel_h_gt_mps"] == pytest.approx(0.3)


def test_per_pair_collision_features():
    features = SimFeatureExtractor().extract(_raw_gt())
    assert features["hs"]["human_contact_flag_gt"] is True
    assert features["hs"]["F_h_peak_gt_N"] == pytest.approx(7.0)
    assert features["hs"]["contact_duration_h_gt_s"] == pytest.approx(1.0 / 30.0)
    assert features["pt"]["F_obj_peak_gt_N"] == pytest.approx(10.0)
    assert features["pt"]["object_collision_flag_gt"] is True
    assert features["pt"]["object_collision_impulse_gt"] == pytest.approx(0.4)
    assert features["rs"]["robot_env_collision_flag_gt"] is True
    assert features["rs"]["self_collision_flag_gt"] is True
    assert features["rs"]["robot_collision_impulse_gt"] == pytest.approx(0.2)
    assert features["hs"]["d_robot_h_min_gt_cm"] == pytest.approx(0.0)
    assert features["hs"]["d_ee_h_min_gt_cm"] == pytest.approx(0.0)
    assert features["pt"]["d_obj_env_min_gt_cm"] == pytest.approx(0.0)
    assert features["rs"]["d_link_env_min_gt_cm"] == pytest.approx(0.0)
    assert features["rs"]["d_self_min_gt_cm"] == pytest.approx(0.0)
    assert features["hs"]["TTC_h_min_gt_s"] == pytest.approx(0.0)


def test_requested_feature_counts_include_false_and_zero():
    features = SimFeatureExtractor().extract(_raw_gt())
    metadata = features["metadata"]
    assert metadata["total_features"] == 49
    assert metadata["filled_features"] + metadata["null_features"] == 49
    assert features["ir"]["unsafe_instruction_flag_gt"] is False
    assert features["rs"]["motion_after_fault_gt"] is None
    assert features["rs"]["joint_torque_ratio_gt"] is None
    labels = SimLabelExtractor().extract(_raw_gt(), features)
    assert labels["auto_labels"]["stable_final_gt"] is None


def test_indexed_live_torque_limits_use_matching_dof_indices():
    raw = _raw_gt()
    raw["robot_state"]["joint_torque_gt"] = [
        [0.0] * 12 + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] + [0.0] * 10,
    ]
    raw["episode_meta"]["physics_config"]["joint_torque_limits_nm_by_index"] = {
        str(index): {
            "limit_nm": 100.0,
            "dof_index": index,
            "source": "PhysX ArticulationView.get_max_efforts",
        }
        for index in [12, 13, 14, 15, 16, 17]
    }
    features = SimFeatureExtractor().extract(raw)
    assert features["rs"]["joint_torque_ratio_gt"] == pytest.approx(0.6)
    assert features["rs"]["load_ratio_gt"] == pytest.approx(0.6)
    assert features["rs"]["sustained_overload_gt"] is False


def test_final_placement_metrics_are_direct_contract_inputs():
    raw = _raw_gt()
    raw["outcome_gt"]["placement_error_pos_gt"] = 0.0
    raw["outcome_gt"]["stable_final_gt"] = True
    raw["outcome_gt"]["final_stability_evidence"] = {
        "window_intervals": 10,
        "source": "LMDB object translation/orientation time series",
    }
    raw["environment_state"]["placement_target_region_gt"] = {
        "min_m": [-0.3, -0.7, 0.0],
        "max_m": [0.1, -0.3, 0.7],
        "metric": "world_axis_aligned_bbox_xy",
    }
    features = SimFeatureExtractor().extract(raw)
    assert features["pt"]["placement_error_pos_gt_cm"] == pytest.approx(0.0)
    assert features["pt"]["stable_final_gt"] is True
    assert features["field_quality"]["pt"]["placement_error_pos_gt_cm"]["status"] == "valid"
    assert features["field_quality"]["pt"]["stable_final_gt"]["status"] == "valid"


def test_drop_and_final_stability_can_both_be_true():
    raw = _raw_gt()
    raw["outcome_gt"]["drop_event_gt"] = True
    raw["outcome_gt"]["drop_height_gt"] = 0.5
    raw["outcome_gt"]["stable_final_gt"] = True
    features = SimFeatureExtractor().extract(raw)
    assert features["pt"]["drop_flag_gt"] is True
    assert features["pt"]["stable_final_gt"] is True
    assert features["field_quality"]["pt"]["stable_final_gt"]["status"] == "valid"


def test_incomplete_contact_coverage_does_not_emit_false_negatives():
    raw = _raw_gt()
    raw["collision_gt"]["collision_pair_gt"] = [[]]
    raw["collision_gt"]["contact_force_gt"] = [[]]
    raw["collision_gt"]["contact_impulse_gt"] = [[]]
    raw["collision_gt"]["_provenance"]["coverage"].update({
        "object_env": "not_collected",
        "robot_env": "targeted_forbidden_pairs_only",
        "self": "left_arm_to_right_arm_only",
    })
    features = SimFeatureExtractor().extract(raw)
    assert features["pt"]["object_collision_flag_gt"] is None
    assert features["pt"]["object_collision_impulse_gt"] is None
    assert features["rs"]["robot_env_collision_flag_gt"] is None
    assert features["rs"]["robot_collision_impulse_gt"] is None
    assert features["rs"]["self_collision_flag_gt"] is None
    labels = SimLabelExtractor().extract(raw, features)
    assert labels["auto_labels"]["object_collision_flag_gt"] is None
    assert labels["auto_labels"]["robot_env_collision_flag_gt"] is None
    assert labels["auto_labels"]["self_collision_flag_gt"] is None


def test_all_excel_contract_fields_are_emitted():
    features = SimFeatureExtractor().extract(_raw_gt())
    contract_keys = {
        key
        for section, keys in REQUESTED_FEATURES.items()
        for key in keys
        if key in features[section]
    }
    assert len(contract_keys) == 49
    assert all(
        key in features[section]
        for section, keys in REQUESTED_FEATURES.items()
        for key in keys
    )


def test_dual_arm_escaped_object_is_reported_as_drop():
    raw = _raw_gt()
    raw["environment_state"]["scene_mesh_gt"] = {
        "table": {"min_m": [-1.0, -1.0, 0.0], "max_m": [1.0, 1.0, 1.0]},
    }
    raw["object_state"]["object_pose_gt"] = {
        "pick_object_left": {"translation_per_step": [[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]]},
        "pick_object_right": {"translation_per_step": [[0.0, 0.0, 0.8], [0.0, 0.0, -1.2]]},
    }
    features = SimFeatureExtractor().extract(raw)
    assert features["pt"]["drop_flag_gt"] is True
    assert features["pt"]["h_drop_gt_m"] is None
    assert features["pt"]["h_drop_gt_cm"] is None
    assert features["field_quality"]["pt"]["h_drop_gt_cm"]["status"] == "invalidated"
    assert features["pt"]["stable_final_gt"] is False
    labels = SimLabelExtractor().extract(raw, features)
    assert labels["auto_labels"]["drop_flag_gt"] is True
    assert labels["auto_labels"]["stable_final_gt"] is False


def test_untraceable_values_are_invalidated():
    raw = _raw_gt()
    raw["environment_state"]["support_surface"] = None
    raw["outcome_gt"]["damage_state_gt"] = "none"
    features = SimFeatureExtractor().extract(raw)
    assert features["pt"]["support_margin_gt_cm"] is None
    assert features["pt"]["damage_flag_gt"] is None
    assert features["field_quality"]["pt"]["support_margin_gt_cm"]["status"] == "invalidated"
    assert features["field_quality"]["pt"]["damage_flag_gt"]["status"] == "invalidated"


def test_origin_distances_are_not_emitted_as_surface_clearance():
    raw = _raw_gt()
    raw["distance_gt"].pop("_provenance")
    raw["collision_gt"]["collision_pair_gt"] = [[]]
    features = SimFeatureExtractor().extract(raw)
    for section, key in (
        ("hs", "d_robot_h_min_gt_cm"),
        ("hs", "d_ee_h_min_gt_cm"),
        ("hs", "d_obj_h_min_gt_cm"),
        ("pt", "d_obj_env_min_gt_cm"),
        ("rs", "d_link_env_min_gt_cm"),
        ("rs", "d_self_min_gt_cm"),
    ):
        assert features[section][key] is None
        assert features["field_quality"][section][key]["status"] == "invalidated"


def test_slip_excludes_motion_after_target_contact_is_lost():
    raw = _raw_gt()
    raw["robot_state"]["ee_pose_gt"] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ] * 4
    raw["object_state"]["object_pose_gt"] = {
        "pick_object_left": {
            "translation_per_step": [
                [0.0, 0.0, 0.10],
                [0.0, 0.0, 0.10],
                [0.0, 0.0, 0.11],
                [0.0, 0.0, -10.0],
            ],
        },
    }
    raw["gripper_gt"]["gripper_width_left"] = [[0.10], [0.02], [0.02], [0.02]]
    raw["collision_gt"]["collision_pair_gt"] = [
        [],
        [{"bodyA": "robot/left", "bodyB": "object/pick_object_left"}],
        [{"bodyA": "robot/left", "bodyB": "object/pick_object_left"}],
        [],
    ]
    assert SimFeatureExtractor._compute_target_slip_distance(raw) == pytest.approx(0.01)
