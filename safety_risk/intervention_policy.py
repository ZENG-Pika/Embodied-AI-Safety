"""Online risk intervention policy for safety-data collection.

The post-hoc HS/PT/RS/IR evaluator remains the source of benchmark labels.
This module has a narrower purpose: turn the *current human-safety state* into
an action constraint before the next low-level command is submitted.

L0: nominal command
L1: speed-limited command
L2: latched linear braking followed by a protective hold
L3: perception-only sample followed by an emergency hold
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class RiskInterventionPolicy:
    """Stateful L0-L3 intervention policy.

    L2 and L3 are intentionally latched for the rest of the episode.  A robot
    must not resume a nominal trajectory merely because a noisy distance
    estimate crosses a threshold for one frame.
    """

    DEFAULT_THRESHOLDS = {
        "l1_distance_m": 0.15,
        "l2_distance_m": 0.10,
        "l3_distance_m": 0.05,
        "l1_ttc_s": 2.0,
        "l2_ttc_s": 1.0,
        "l3_ttc_s": 0.25,
        "l1_closing_speed_mps": 0.10,
        "l3_closing_speed_mps": 0.25,
        "l3_contact_force_n": 50.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        thresholds = dict(self.DEFAULT_THRESHOLDS)
        thresholds.update(config.get("thresholds", {}) or {})

        self.enabled = bool(config.get("enabled", False))
        self.thresholds = thresholds
        self.l1_speed_scale = min(1.0, max(0.05, float(config.get("l1_speed_scale", 0.5))))
        self.physics_dt_s = max(1.0e-6, float(config.get("physics_dt_s", 1.0 / 30.0)))
        # SimBox safety scenarios run at 30 Hz. Ninety braking frames provide
        # a three-second L2 deceleration window by default. A later L3
        # observation still overrides this window with an immediate hold.
        self.l2_brake_steps = max(1, int(config.get("l2_brake_steps", 90)))
        self.l2_hold_steps = max(1, int(config.get("l2_hold_steps", 60)))
        self.l3_hold_steps = max(1, int(config.get("l3_hold_steps", 30)))
        self.stop_confirm_steps = max(1, int(config.get("stop_confirm_steps", 5)))
        self.stop_joint_velocity_threshold = max(
            0.0, float(config.get("stop_joint_velocity_threshold", 0.05))
        )
        self.sustained_contact_duration_s = max(
            0.0, float(config.get("sustained_contact_duration_s", 0.5))
        )
        self.sustained_contact_force_n = max(
            0.0, float(config.get("sustained_contact_force_n", 10.0))
        )
        self.l1_hysteresis_m = max(0.0, float(config.get("l1_hysteresis_m", 0.02)))
        self.l1_release_steps = max(1, int(config.get("l1_release_steps", 3)))

        self.observed_level = "L0"
        self.observed_reasons = []
        self._l2_latched = False
        self._l3_seen = False
        self._brake_step = 0
        self._hold_step = 0
        self._l3_action_step = 0
        self._brake_start_scale = 1.0
        self._last_command_scale = 1.0
        self._active_l2_brake_steps = self.l2_brake_steps
        self._contact_frames = 0
        self._stop_confirm_count = 0
        self._last_max_joint_speed = None
        self._l1_active = False
        self._l1_clear_frames = 0

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": "online_human_safety",
            "l1_speed_scale": self.l1_speed_scale,
            "l1_strategy": "trajectory_time_dilation",
            "l2_brake_steps": self.l2_brake_steps,
            "l2_max_brake_duration_s": self.l2_brake_steps * self.physics_dt_s,
            "l2_hold_steps": self.l2_hold_steps,
            "l3_hold_steps": self.l3_hold_steps,
            "physics_dt_s": self.physics_dt_s,
            "stop_confirm_steps": self.stop_confirm_steps,
            "stop_joint_velocity_threshold": self.stop_joint_velocity_threshold,
            "sustained_contact_duration_s": self.sustained_contact_duration_s,
            "sustained_contact_force_n": self.sustained_contact_force_n,
            "l1_hysteresis_m": self.l1_hysteresis_m,
            "l1_release_steps": self.l1_release_steps,
            "thresholds": dict(self.thresholds),
            "control_training_levels": ["L0", "L1", "L2"],
            "perception_only_levels": ["L3"],
            "l2_latched": True,
        }

    def observe(
        self,
        *,
        distance_m: Optional[float],
        closing_speed_mps: Optional[float],
        ttc_s: Optional[float],
        human_contact: bool,
        contact_force_n: Optional[float],
    ) -> Dict[str, Any]:
        """Classify the current frame using the benchmark HS thresholds."""
        if not self.enabled:
            self.observed_level = "L0"
            self.observed_reasons = ["intervention_disabled"]
            return self._observation(distance_m, closing_speed_mps, ttc_s, human_contact, contact_force_n)

        d = _finite(distance_m)
        speed = max(0.0, _finite(closing_speed_mps) or 0.0)
        ttc = _finite(ttc_s)
        force = max(0.0, _finite(contact_force_n) or 0.0)
        t = self.thresholds
        reasons = []

        if human_contact:
            self._contact_frames += 1
        else:
            self._contact_frames = 0
        contact_duration_s = self._contact_frames * self.physics_dt_s

        if force > float(t["l3_contact_force_n"]):
            reasons.append("human_contact_force_exceeded")
        if ttc is not None and ttc < float(t["l3_ttc_s"]):
            reasons.append("critical_ttc")
        if d is not None and d < float(t["l3_distance_m"]) and speed > float(t["l3_closing_speed_mps"]):
            reasons.append("critical_high_speed_approach")
        if (
            human_contact
            and contact_duration_s >= self.sustained_contact_duration_s
            and force > self.sustained_contact_force_n
        ):
            reasons.append("sustained_human_contact")

        if reasons:
            level = "L3"
        elif (
            human_contact
            or
            (d is not None and d < float(t["l2_distance_m"]))
            or (ttc is not None and float(t["l3_ttc_s"]) <= ttc < float(t["l2_ttc_s"]))
        ):
            level = "L2"
            reasons = ["protective_stop_zone"]
        elif (
            (d is not None and d < float(t["l1_distance_m"]))
            or (ttc is not None and float(t["l2_ttc_s"]) <= ttc < float(t["l1_ttc_s"]))
            or (d is not None and d < float(t["l1_distance_m"]) and speed > float(t["l1_closing_speed_mps"]))
        ):
            level = "L1"
            reasons = ["slowdown_zone"]
        else:
            level = "L0"
            reasons = ["safe_or_no_approach"] if d is not None else ["distance_unavailable"]

        if level == "L1":
            self._l1_active = True
            self._l1_clear_frames = 0
        elif level == "L0" and self._l1_active:
            release_distance = float(t["l1_distance_m"]) + self.l1_hysteresis_m
            safely_clear = (
                d is not None
                and d >= release_distance
                and (ttc is None or ttc >= float(t["l1_ttc_s"]))
            )
            self._l1_clear_frames = self._l1_clear_frames + 1 if safely_clear else 0
            if self._l1_clear_frames < self.l1_release_steps:
                level = "L1"
                reasons = ["slowdown_hysteresis"]
            else:
                self._l1_active = False
                self._l1_clear_frames = 0
        elif level in {"L2", "L3"}:
            self._l1_active = False
            self._l1_clear_frames = 0

        if level == "L3":
            self._l3_seen = True
            self._l2_latched = True
        elif level == "L2" and not self._l2_latched:
            self._l2_latched = True
            self._brake_start_scale = self._last_command_scale

        if level == "L2":
            safe_steps = self._safe_l2_brake_steps(
                distance_m=d,
                closing_speed_mps=speed,
                human_contact=human_contact,
            )
            if not self._l2_latched or self._brake_step == 0:
                self._active_l2_brake_steps = safe_steps
            else:
                # Risk may worsen while braking. Shorten the remaining
                # schedule, but never lengthen it after the stop has begun.
                self._active_l2_brake_steps = min(
                    self._active_l2_brake_steps, safe_steps
                )

        self.observed_level = level
        self.observed_reasons = reasons
        return self._observation(d, speed, ttc, human_contact, force)

    def _safe_l2_brake_steps(
        self,
        *,
        distance_m: Optional[float],
        closing_speed_mps: float,
        human_contact: bool,
    ) -> int:
        """Bound the requested 3 s stop by the available surface clearance."""
        if human_contact:
            return max(1, min(self.l2_brake_steps, round(0.1 / self.physics_dt_s)))
        if distance_m is None or closing_speed_mps <= 0.01:
            return self.l2_brake_steps
        clearance = max(0.0, distance_m - float(self.thresholds["l3_distance_m"]))
        safe_duration_s = 2.0 * clearance / closing_speed_mps
        # Stabilize exact frame boundaries against binary floating-point
        # rounding (for example 0.6 s at 30 Hz must remain 18 frames).
        safe_steps = max(
            1,
            int(math.floor(safe_duration_s / self.physics_dt_s + 1.0e-9)),
        )
        return min(self.l2_brake_steps, safe_steps)

    def update_stop_confirmation(self, max_joint_speed: Optional[float]) -> Dict[str, Any]:
        """Track actual robot rest; fixed hold windows are only timeouts."""
        speed = _finite(max_joint_speed)
        self._last_max_joint_speed = speed
        if self._last_command_scale > 0.0 or speed is None:
            self._stop_confirm_count = 0
        elif speed <= self.stop_joint_velocity_threshold:
            self._stop_confirm_count += 1
        else:
            self._stop_confirm_count = 0
        return {
            "max_joint_speed": speed,
            "stop_confirm_count": self._stop_confirm_count,
            "stop_confirmed": self._stop_confirm_count >= self.stop_confirm_steps,
        }

    def command_decision(self, step_id: int) -> Dict[str, Any]:
        """Return the constraint to apply to the next command."""
        if not self.enabled:
            return self._decision(step_id, "L0", "none", 1.0, False, True)

        if self._l3_seen:
            self._l3_action_step += 1
            stop_verified = self._stop_confirm_count >= self.stop_confirm_steps
            stop_timeout = self._l3_action_step >= self.l3_hold_steps
            terminate = stop_verified or stop_timeout
            self._last_command_scale = 0.0
            return self._decision(
                step_id,
                "L3",
                "perception_only_emergency_hold",
                0.0,
                terminate,
                False,
                stop_verified=stop_verified,
                stop_timeout=stop_timeout,
            )

        if self._l2_latched:
            if self._brake_step < self._active_l2_brake_steps:
                self._brake_step += 1
                remaining = 1.0 - self._brake_step / float(self._active_l2_brake_steps)
                scale = max(0.0, self._brake_start_scale * remaining)
                action = "linear_deceleration" if scale > 0.0 else "protective_hold"
            else:
                scale = 0.0
                action = "protective_hold"

            terminate = False
            stop_verified = False
            stop_timeout = False
            if scale <= 0.0:
                self._hold_step += 1
                stop_verified = self._stop_confirm_count >= self.stop_confirm_steps
                stop_timeout = self._hold_step >= self.l2_hold_steps
                terminate = stop_verified or stop_timeout
            self._last_command_scale = scale
            return self._decision(
                step_id,
                "L2",
                action,
                scale,
                terminate,
                True,
                brake_step=self._brake_step,
                brake_total_steps=self._active_l2_brake_steps,
                stop_verified=stop_verified,
                stop_timeout=stop_timeout,
            )

        if self.observed_level == "L1":
            self._last_command_scale = self.l1_speed_scale
            return self._decision(step_id, "L1", "speed_limit", self.l1_speed_scale, False, True)

        self._last_command_scale = 1.0
        return self._decision(step_id, "L0", "none", 1.0, False, True)

    def _observation(self, distance_m, closing_speed_mps, ttc_s, human_contact, contact_force_n):
        return {
            "level": self.observed_level,
            "reasons": list(self.observed_reasons),
            "distance_m": _finite(distance_m),
            "closing_speed_mps": _finite(closing_speed_mps),
            "ttc_s": _finite(ttc_s),
            "human_contact": bool(human_contact),
            "contact_force_n": _finite(contact_force_n),
            "contact_duration_s": self._contact_frames * self.physics_dt_s,
            "l2_latched": self._l2_latched,
            "l3_seen": self._l3_seen,
            "active_l2_brake_steps": self._active_l2_brake_steps,
            "max_joint_speed": self._last_max_joint_speed,
            "stop_confirm_count": self._stop_confirm_count,
        }

    def _decision(
        self,
        step_id,
        level,
        action,
        scale,
        terminate,
        control_eligible,
        **extra,
    ):
        result = {
            "step_id": int(step_id),
            "level_before_action": level,
            "action": action,
            "command_scale": float(scale),
            "terminate_after_step": bool(terminate),
            "control_training_eligible": bool(control_eligible),
            "perception_training_eligible": True,
            "reasons": list(self.observed_reasons),
        }
        result.update(extra)
        return result
