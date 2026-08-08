import pytest

from safety_risk.sim_feature_extractor import REQUESTED_FEATURES, SimFeatureExtractor


def raw_gt():
    return {
        "episode_meta": {"episode_id": "v2", "physics_config": {"physics_dt": 0.1}},
        "robot_state": {},
        "distance_gt": {
            "_provenance": {name: {"metric": "surface_clearance"} for name in (
                "robot_human_distance_matrix_gt", "ee_human_distance_gt", "object_human_distance_gt",
                "object_env_distance_gt", "link_env_distance_gt", "self_distance_gt")},
            "robot_human_distance_matrix_gt": [{"link": {"palm": 0.15}}],
            "ee_human_distance_gt": [{"left": 0.20}],
            "object_human_distance_gt": [{"object": {"palm": 0.30}}],
            "object_env_distance_gt": [{"object": {"table": 0.04}}],
            "link_env_distance_gt": [{"link": {"table": 0.06}}],
            "self_distance_gt": [{"a": {"b": 0.05}}],
        },
        "collision_gt": {"_provenance": {"coverage": {
            "human": "complete", "robot_human": "complete", "ee_human": "complete",
            "object_human": "complete", "object_env": "complete", "robot_env": "complete", "self": "complete"}},
            "collision_pair_gt": [[]], "contact_force_gt": [[]], "contact_impulse_gt": [[]],
            "contact_duration_gt": []},
        "gripper_gt": {}, "outcome_gt": {"drop_event_gt": False, "drop_height_gt": None},
        "object_state": {}, "environment_state": {}, "sensor_gt": {"visibility_ratio_gt": [0.8]},
        "planner_log": {}, "hri_log": {"unsafe_instruction_flag_gt": False},
    }


def test_si_only_36_field_contract():
    result = SimFeatureExtractor().extract(raw_gt())
    expected = {section: set(keys) for section, keys in REQUESTED_FEATURES.items()}
    assert sum(map(len, expected.values())) == 36
    for section, keys in expected.items():
        assert set(result[section]) == keys
    assert not any(key.endswith("_cm") or key.endswith("_deg") for keys in expected.values() for key in keys)
    assert result["metadata"]["total_features"] == 36
    assert result["metadata"]["validation_status"] == "passed"


def test_si_distances_and_occlusion_are_not_rescaled():
    raw = raw_gt()
    for provenance in raw["distance_gt"]["_provenance"].values():
        provenance["metric"] = "collider_world_aabb_surface_clearance"
    result = SimFeatureExtractor().extract(raw)
    assert result["hs"]["d_robot_h_min_gt_m"] == pytest.approx(0.15)
    assert result["hs"]["d_ee_h_min_gt_m"] == pytest.approx(0.20)
    assert result["pt"]["d_obj_env_min_gt_m"] == pytest.approx(0.04)
    assert result["rs"]["d_link_env_min_gt_m"] == pytest.approx(0.06)
    assert result["ir"]["true_occlusion_ratio"] == pytest.approx(0.20)


def test_unsafe_low_level_name_is_explicit():
    raw = raw_gt()
    raw["planner_log"] = {"unsafe_action_planned": True, "low_level_command_sent": [False, True]}
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["unsafe_low_level_command_sent"] is True
    assert "low_level_command_sent" not in result["ir"]


def test_collision_impulse_is_peak_event_not_episode_sum():
    coll = {"contact_impulse_gt": [
        [{"bodyA": "object/a", "bodyB": "environment/table", "impulse_ns": 0.4}],
        [{"bodyA": "object/a", "bodyB": "environment/table", "impulse_ns": 0.7}],
    ]}
    assert SimFeatureExtractor()._compute_collision_impulse(coll, "object_env") == pytest.approx(0.7)


def test_unknown_damage_is_promoted_only_by_decisive_severe_evidence():
    raw = raw_gt()
    raw["outcome_gt"].update({
        "damage_state_gt": "unknown",
        "drop_event_gt": True,
        "drop_height_gt": 0.6,
    })
    result = SimFeatureExtractor().extract(raw)
    assert result["pt"]["damage_flag_gt"] is True
    assert result["field_quality"]["pt"]["damage_flag_gt"]["status"] == "valid"

    raw["outcome_gt"]["drop_event_gt"] = False
    raw["outcome_gt"]["drop_height_gt"] = None
    result = SimFeatureExtractor().extract(raw)
    assert result["pt"]["damage_flag_gt"] is None


def test_self_distance_supports_direct_pair_map_and_excludes_adjacent_links():
    series = [{
        "robot/fl/link1→robot/fl/link2": 0.0,
        "robot/fl/link1→robot/fl/link3": 0.02,
        "robot/fl/link1→robot/fr/link1": 0.10,
    }]
    assert SimFeatureExtractor._minimum_nonzero_self_distance(series) == pytest.approx(0.02)


def test_compact_joint_torques_use_source_dof_indices():
    raw = raw_gt()
    raw["robot_state"].update({
        "joint_torque_gt": [[10.0, 20.0]],
        "joint_velocity_dq_gt": [[0.0, 0.0]],
        "joint_state_metadata": {"source_dof_indices": [12, 14]},
    })
    raw["episode_meta"]["physics_config"]["joint_torque_limits_nm_by_index"] = {
        "12": {"limit_nm": 100.0}, "14": {"limit_nm": 40.0},
    }
    result = SimFeatureExtractor(dt=0.1).extract(raw)
    assert result["rs"]["joint_torque_ratio_gt"] == pytest.approx(0.5)


def test_joint_margin_uses_live_non_piper_limits():
    raw = raw_gt()
    raw["robot_state"].update({
        "joint_position_q_gt": [[0.9, -1.8]],
        "joint_velocity_dq_gt": [[0.0, 0.0]],
        "joint_state_metadata": {"source_dof_indices": [4, 7]},
    })
    raw["episode_meta"]["physics_config"]["joint_position_limits_rad_by_index"] = {
        "4": {"lower_rad": -1.0, "upper_rad": 1.0},
        "7": {"lower_rad": -2.0, "upper_rad": 2.0},
    }
    result = SimFeatureExtractor().extract(raw)
    assert result["rs"]["joint_limit_margin_gt_rad"] == pytest.approx(0.1)


def test_support_margin_falls_back_to_recorded_target_region():
    raw = raw_gt()
    raw["environment_state"]["placement_target_region_gt"] = {
        "min_m": [0.0, 0.0, 0.0], "max_m": [1.0, 1.0, 1.0],
    }
    raw["object_state"]["object_pose_gt"] = {
        "pick_object_left": {"translation_per_step": [[0.2, 0.3, 0.1]]},
        "pick_object_right": {"translation_per_step": [[0.7, 0.6, 0.1]]},
    }
    result = SimFeatureExtractor().extract(raw)
    assert result["pt"]["support_margin_gt_m"] == pytest.approx(0.2)


def test_visibility_loss_and_motion_produce_blind_action():
    raw = raw_gt()
    present = {
        "1": {"label": {"class": "pick_object_left"}, "visibility_ratio": 1.0},
        "2": {"label": {"class": "pick_object_right"}, "visibility_ratio": 1.0},
    }
    raw["sensor_gt"]["visibility_ratio_gt"] = [
        {"instances": present}, {"instances": {}}, {"instances": {}}, {"instances": {}}
    ]
    raw["robot_state"]["joint_velocity_dq_gt"] = [[0.0], [0.2], [0.2], [0.2]]
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["tracking_lost_flag_sim"] is True
    assert result["ir"]["blind_action_flag_sim"] is True


def test_motion_after_robot_environment_collision_uses_velocity_timeline():
    raw = raw_gt()
    raw["collision_gt"]["collision_pair_gt"] = [
        [{"bodyA": "robot/link", "bodyB": "environment/table", "step": 0}],
        [], [], [],
    ]
    raw["robot_state"]["joint_velocity_dq_gt"] = [[0.0], [0.2], [0.2], [0.2]]
    result = SimFeatureExtractor(dt=0.1).extract(raw)
    assert result["rs"]["motion_after_fault_gt"] is True


def test_motion_after_human_contact_is_a_fault():
    raw = raw_gt()
    raw["collision_gt"]["collision_pair_gt"] = [
        [{"bodyA": "object/target", "bodyB": "obstacle/obstacle_1", "step": 0}],
        [], [], [],
    ]
    raw["robot_state"]["joint_velocity_dq_gt"] = [[0.0], [0.2], [0.2], [0.2]]
    result = SimFeatureExtractor(dt=0.1).extract(raw)
    assert result["rs"]["motion_after_fault_gt"] is True
