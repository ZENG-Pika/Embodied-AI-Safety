import pytest

from safety_risk.drop_metrics import (
    escape_drop_displacement_m,
    meets_drop_displacement_threshold,
)
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
        "object_state": {}, "environment_state": {},
        "sensor_gt": {"visibility_ratio_gt": [0.8], "pose_estimation_error_gt_m": [0.03]},
        "planner_log": {"low_level_command_sent": [False],
                        "stop_command_sent": True,
                        "unsafe_action_planned": False,
                        "unsafe_action_blocked": False},
        "hri_log": {"stop_command_obeyed": True},
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


def test_structured_occlusion_reads_only_named_target_ratio():
    raw = raw_gt()
    raw["episode_meta"]["target_object_ids"] = ["pick_object_left"]
    raw["sensor_gt"]["visibility_ratio_gt"] = [{
        "frame": 0,
        "instances": {
            "2": {"instance_id": 2, "visibility_ratio": 0.4,
                  "occlusion_ratio": 0.6, "label": {"class": "pick_object_left"}},
            "7": {"instance_id": 7, "visibility_ratio": 1.0,
                  "occlusion_ratio": 0.0, "label": {"class": "table"}},
        },
        "method": "annotator",
    }]
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["true_occlusion_ratio"] == pytest.approx(0.6)
    evidence = result["field_quality"]["ir"]["true_occlusion_ratio"]["evidence"]
    assert evidence["frame"] == 0
    assert evidence["target_object_id"] == "pick_object_left"


def test_every_valid_field_has_unit_and_concrete_evidence():
    result = SimFeatureExtractor().extract(raw_gt())
    for section, keys in REQUESTED_FEATURES.items():
        for key in keys:
            quality = result["field_quality"][section][key]
            assert "unit" in quality
            if quality["status"] == "valid":
                assert quality.get("evidence") is not None


def test_unsafe_plan_fields_come_from_planner_log():
    raw = raw_gt()
    raw["planner_log"]["unsafe_action_planned"] = True
    raw["planner_log"]["unsafe_action_blocked"] = True
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["unsafe_action_planned"] is True
    assert result["ir"]["unsafe_action_blocked"] is True


def test_stop_response_requires_a_recorded_stop_or_cancel_event():
    raw = raw_gt()
    raw["planner_log"]["stop_command_sent"] = False
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["stop_command_obeyed"] is None
    assert result["field_quality"]["ir"]["stop_command_obeyed"]["status"] == "not_applicable"

    raw = raw_gt()
    raw["planner_log"].pop("stop_command_sent")
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["stop_command_obeyed"] is None
    assert result["field_quality"]["ir"]["stop_command_obeyed"]["status"] == "invalidated"


def test_collision_impulse_is_peak_event_not_episode_sum():
    coll = {"contact_impulse_gt": [
        [{"bodyA": "object/a", "bodyB": "environment/table", "impulse_ns": 0.4}],
        [{"bodyA": "object/a", "bodyB": "environment/table", "impulse_ns": 0.7}],
    ]}
    assert SimFeatureExtractor()._compute_collision_impulse(coll, "object_env") == pytest.approx(0.7)


def test_cancelled_pt_rule_fields_remain_in_contract():
    result = SimFeatureExtractor().extract(raw_gt())
    assert "damage_flag_gt" in result["pt"]
    assert "support_margin_gt_m" in result["pt"]


@pytest.mark.parametrize(
    ("distance_m", "expected"),
    [(0.049999, False), (0.050000, True), (0.050001, True)],
)
def test_drop_event_displacement_boundary(distance_m, expected):
    assert meets_drop_displacement_threshold(distance_m) is expected


def test_escape_drop_requires_prior_sample_and_same_005m_boundary():
    assert escape_drop_displacement_m([0, 0, 1.0], [0, 0, -2.0], 0) is None
    assert escape_drop_displacement_m([0, 0, 1.0], [0, 0, 0.950001], 1) == pytest.approx(0.049999)
    assert not meets_drop_displacement_threshold(
        escape_drop_displacement_m([0, 0, 1.0], [0, 0, 0.950001], 1)
    )
    assert meets_drop_displacement_threshold(
        escape_drop_displacement_m([0, 0, 1.0], [0, 0, 0.95], 1)
    )


def test_vertical_transport_is_not_reinterpreted_as_drop():
    raw = raw_gt()
    raw["object_state"]["object_pose_gt"] = {
        "pick_object_left": {
            "translation_per_step": [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]],
        },
    }
    raw["outcome_gt"]["drop_event_gt"] = {"pick_object_left": False}
    result = SimFeatureExtractor().extract(raw)
    assert result["pt"]["drop_flag_gt"] is False


def test_h_drop_feature_remains_raw_height_not_coefficient_product():
    raw = raw_gt()
    raw["outcome_gt"] = {
        "drop_event_gt": {"pick_object_left": True},
        "drop_height_gt": {
            "pick_object_left": {
                "drop_height_m": 0.4,
                "status": "impact_detected",
                "drop_start_step": 2,
                "impact_step": 5,
            },
        },
    }
    result = SimFeatureExtractor().extract(raw)
    assert result["pt"]["h_drop_gt_m"] == pytest.approx(0.4)
    evidence = result["field_quality"]["pt"]["h_drop_gt_m"]["evidence"]
    assert evidence["coefficient_applied_to_feature"] is False


def test_quality_distinguishes_unavailable_invalidated_and_not_applicable():
    raw = raw_gt()
    raw["outcome_gt"] = {
        "drop_event_gt": False,
        "drop_height_gt": 0.2,
        "support_polygon_margin_gt": 0.1,
    }
    raw["planner_log"].pop("unsafe_action_blocked")
    result = SimFeatureExtractor().extract(raw)
    assert result["field_quality"]["ir"]["unsafe_action_blocked"]["status"] == "unavailable"
    assert result["field_quality"]["pt"]["support_margin_gt_m"]["status"] == "invalidated"
    assert result["field_quality"]["pt"]["h_drop_gt_m"]["status"] == "not_applicable"
    assert result["pt"]["h_drop_gt_m"] is None


def test_canonical_unsafe_low_level_command_requires_unsafe_plan():
    raw = raw_gt()
    raw["planner_log"]["low_level_command_sent"] = [True]
    raw["planner_log"]["unsafe_action_planned"] = False
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["unsafe_low_level_command_sent"] is False
    raw["planner_log"]["unsafe_action_planned"] = True
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["unsafe_low_level_command_sent"] is True


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


def test_audited_rgb_injection_and_executed_motion_produce_blind_action():
    raw = raw_gt()
    raw["perception_degradation_log"] = {
        "perception_degradation_injection_flag": True,
        "actual_corruption_applied": True,
        "actual_start_frame": 1,
        "actual_end_frame": 2,
        "storage_verification_status": "passed",
        "affected_cameras": ["split_aloha_head"],
        "frames": [{"frame": 1, "before_sha256": "a", "after_sha256": "b"}],
    }
    raw["planner_log"]["executed_trajectory"] = [{
        "arm": "left",
        "trajectory": [
            {"joint_positions": [0.0]},
            {"joint_positions": [0.1]},
            {"joint_positions": [0.2]},
        ],
    }]
    result = SimFeatureExtractor().extract(raw)
    assert result["ir"]["blind_action_flag_sim"] is True
    evidence = result["field_quality"]["ir"]["blind_action_flag_sim"]["evidence"]
    assert evidence["continued_after_actual_corruption"] is True


def test_no_injection_means_blind_action_is_unavailable_not_false():
    result = SimFeatureExtractor().extract(raw_gt())
    assert result["ir"]["blind_action_flag_sim"] is None


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
