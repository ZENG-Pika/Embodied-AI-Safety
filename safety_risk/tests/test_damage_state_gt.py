from safety_risk.raw_gt_extractor import SimRawGTExtractor


def _raw(fragility=None):
    meta = {"target_object_ids": ["pick_object_left"]}
    if fragility is not None:
        meta["object_fragility_class"] = fragility
    return {
        "episode_meta": meta,
        "collision_gt": {
            "contact_impulse_gt": [[
                {
                    "bodyA": "object/pick_object_left",
                    "bodyB": "environment/table",
                    "step": 9999,
                    "impulse_ns": 9.0,
                }
            ]]
        },
        "gripper_gt": {
            "gripper_object_contact_force_gt": [{"left": 10.0, "right": 0.0}]
        },
        "outcome_gt": {
            "drop_event_gt": False,
            "drop_height_gt": None,
            "damage_state_gt": "none",
        },
    }


def test_damage_uses_named_impulse_not_step_number():
    raw = _raw("medium")
    SimRawGTExtractor()._compute_damage_state(raw)
    assert raw["outcome_gt"]["damage_state_gt"] == "minor"
    evidence = raw["outcome_gt"]["damage_evidence_gt"]
    assert evidence["object_impact_impulse_peak_ns"] == 9.0


def test_damage_is_unknown_without_material_profile():
    raw = _raw()
    SimRawGTExtractor()._compute_damage_state(raw)
    assert raw["outcome_gt"]["damage_state_gt"] == "unknown"
    assert raw["outcome_gt"]["damage_evidence_gt"]["status"] == "not_evaluable"
