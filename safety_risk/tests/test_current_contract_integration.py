import json

from safety_risk.raw_gt_extractor import SimRawGTExtractor
from safety_risk.schema import SimRawEpisode
from safety_risk.sim_feature_extractor import SimFeatureExtractor
from safety_risk.sim_label_extractor import SimLabelExtractor


REMOVED_IDENTIFIERS = {
    "support" + "_margin_gt_m",
    "pose" + "_estimation_error_gt_m",
    "tracking" + "_lost_flag_sim",
    "unsafe" + "_action_blocked",
    "unsafe" + "_low_level_command_sent",
}


def _raw_episode():
    return {
        "episode_meta": {
            "episode_id": "contract-integration",
            "physics_config": {"physics_dt": 0.1},
        },
        "robot_state": {"joint_position_q_gt": [[0.0], [0.1], [0.2]]},
        "distance_gt": {},
        "collision_gt": {},
        "gripper_gt": {},
        "outcome_gt": {},
        "object_state": {},
        "environment_state": {},
        "sensor_gt": {"visibility_ratio_gt": [0.8]},
        "planner_log": {
            "executed_trajectory": [{
                "arm": "left",
                "trajectory": [
                    {"joint_positions": [0.0]},
                    {"joint_positions": [0.1]},
                    {"joint_positions": [0.2]},
                ],
            }],
        },
        "hri_log": {
            "user_command_text": "Both arms pick up bottles while avoiding the moving hand.",
            "unsafe_instruction_flag_gt": False,
            "instruction_safety_assessment": {
                "api_call_attempted": True,
                "api_call_succeeded": True,
                "model": "integration-model",
                "raw_api_response": {"output_text": "{\"unsafe\": false}"},
                "parsed_label": False,
                "status": "valid",
            },
        },
        "perception_degradation_log": {
            "perception_degradation_injection_flag": True,
            "actual_corruption_applied": True,
            "actual_start_frame": 1,
            "actual_end_frame": 2,
            "storage_verification_status": "passed",
            "frames": [{"frame": 1, "before_sha256": "a", "after_sha256": "b"}],
        },
    }


def test_raw_feature_label_chain_has_only_current_contract():
    raw = _raw_episode()
    features = SimFeatureExtractor().extract(raw)
    labels = SimLabelExtractor().extract(raw, features)
    assert features["metadata"]["total_features"] == 31
    assert sum(len(features[name]) for name in ("hs", "pt", "rs", "ir")) == 31
    assert features["ir"]["blind_action_flag_sim"] is True
    assert features["ir"]["unsafe_action_planned"] is False
    assert labels["metadata"]["total_labels"] == 25
    serialized = json.dumps({"raw": raw, "features": features, "labels": labels})
    assert all(identifier not in serialized for identifier in REMOVED_IDENTIFIERS)


def test_raw_schema_and_default_extractors_have_no_removed_semantics():
    model_fields = json.dumps(list(SimRawEpisode.model_fields))
    extractor = SimRawGTExtractor()
    raw_sections = {
        "outcome": extractor._extract_outcome_gt({}),
        "planner": extractor._extract_planner_log({}),
    }
    serialized = model_fields + json.dumps(raw_sections)
    assert all(identifier not in serialized for identifier in REMOVED_IDENTIFIERS)
