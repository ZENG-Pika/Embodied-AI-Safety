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
    result = SimFeatureExtractor().extract(raw_gt())
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
