import pytest

from safety_risk.intervention_policy import RiskInterventionPolicy


def policy(**overrides):
    config = {
        "enabled": True,
        "l1_speed_scale": 0.5,
        "l2_brake_steps": 4,
        "l2_hold_steps": 2,
        "l3_hold_steps": 2,
        "stop_confirm_steps": 2,
    }
    config.update(overrides)
    return RiskInterventionPolicy(config)


def test_l0_nominal_and_l1_speed_limit():
    item = policy()
    assert item.observe(
        distance_m=0.20,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L0"
    assert item.command_decision(1)["command_scale"] == 1.0

    assert item.observe(
        distance_m=0.12,
        closing_speed_mps=0.05,
        ttc_s=1.5,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L1"
    decision = item.command_decision(2)
    assert decision["action"] == "speed_limit"
    assert decision["command_scale"] == pytest.approx(0.5)
    assert decision["control_training_eligible"] is True


def test_default_l2_braking_window_is_three_seconds_at_30_hz():
    item = RiskInterventionPolicy({"enabled": True})
    assert item.l2_brake_steps == 90
    assert item.l2_brake_steps / 30.0 == pytest.approx(3.0)


def test_l1_has_positive_minimum_scale_and_release_hysteresis():
    item = policy(l1_speed_scale=0.0, l1_hysteresis_m=0.02, l1_release_steps=2)
    assert item.l1_speed_scale == pytest.approx(0.05)
    assert item.observe(
        distance_m=0.14,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L1"
    assert item.observe(
        distance_m=0.18,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L1"
    assert item.observe(
        distance_m=0.18,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L0"


def test_l2_is_latched_and_linearly_brakes_to_hold():
    item = policy()
    assert item.observe(
        distance_m=0.08,
        closing_speed_mps=0.08,
        ttc_s=0.8,
        human_contact=False,
        contact_force_n=0.0,
    )["level"] == "L2"

    decisions = [item.command_decision(step) for step in range(1, 5)]
    assert [d["command_scale"] for d in decisions[:4]] == pytest.approx(
        [0.75, 0.50, 0.25, 0.0]
    )
    assert decisions[3]["action"] == "protective_hold"
    assert decisions[3]["terminate_after_step"] is False
    assert all(d["control_training_eligible"] for d in decisions)

    item.update_stop_confirmation(0.0)
    item.update_stop_confirmation(0.0)
    stopped = item.command_decision(5)
    assert stopped["stop_verified"] is True
    assert stopped["terminate_after_step"] is True

    # Returning to a large distance must not resume the interrupted trajectory.
    item.observe(
        distance_m=0.30,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=False,
        contact_force_n=0.0,
    )
    assert item.command_decision(7)["command_scale"] == 0.0


def test_l3_is_perception_only_and_emergency_holds():
    item = policy()
    observation = item.observe(
        distance_m=0.0,
        closing_speed_mps=0.4,
        ttc_s=0.0,
        human_contact=True,
        contact_force_n=80.0,
    )
    assert observation["level"] == "L3"

    first = item.command_decision(1)
    assert first["action"] == "perception_only_emergency_hold"
    assert first["command_scale"] == 0.0
    assert first["control_training_eligible"] is False
    assert first["perception_training_eligible"] is True
    assert first["terminate_after_step"] is False
    item.update_stop_confirmation(0.0)
    item.update_stop_confirmation(0.0)
    second = item.command_decision(2)
    assert second["terminate_after_step"] is True


def test_light_contact_is_l2_but_sustained_loaded_contact_is_l3():
    item = policy(physics_dt_s=0.25, sustained_contact_duration_s=0.5)
    first = item.observe(
        distance_m=0.0,
        closing_speed_mps=0.0,
        ttc_s=None,
        human_contact=True,
        contact_force_n=5.0,
    )
    assert first["level"] == "L2"

    item = policy(physics_dt_s=0.25, sustained_contact_duration_s=0.5)
    for force in (11.0, 11.0):
        result = item.observe(
            distance_m=0.0,
            closing_speed_mps=0.0,
            ttc_s=None,
            human_contact=True,
            contact_force_n=force,
        )
    assert result["level"] == "L3"
    assert "sustained_human_contact" in result["reasons"]


def test_l2_brake_window_is_shortened_by_available_clearance():
    item = RiskInterventionPolicy({"enabled": True, "physics_dt_s": 1.0 / 30.0})
    observation = item.observe(
        distance_m=0.08,
        closing_speed_mps=0.10,
        ttc_s=0.8,
        human_contact=False,
        contact_force_n=0.0,
    )
    assert observation["level"] == "L2"
    # 2 * (0.08 - 0.05) / 0.10 = 0.6 s = 18 frames.
    assert observation["active_l2_brake_steps"] == 18
