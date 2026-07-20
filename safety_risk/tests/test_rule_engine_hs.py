"""Tests for rule_engine.py - HS (Human Safety) rules."""

from __future__ import annotations

import pytest

from safety_risk.config import SafetyRiskConfig
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    CommonFeatures,
    DataQuality,
    HSFeatures,
    IRFeatures,
    PTFeatures,
    RiskCategory,
    RiskFeatures,
    RiskLevel,
    RSFeatures,
)


@pytest.fixture
def engine():
    return RuleBasedRiskEngine(SafetyRiskConfig.load())


def _make_features(hs: HSFeatures, robot_active: bool = True) -> RiskFeatures:
    return RiskFeatures(
        common=CommonFeatures(robot_active=robot_active, data_quality=DataQuality.A),
        hs=hs,
    )


class TestHSL3:
    """Test HS L3 hard triggers."""

    def test_human_contact(self, engine):
        features = _make_features(HSFeatures(human_contact_flag_gt=True))
        result = engine.evaluate(features, "test_contact")
        assert result.hs_level == RiskLevel.L3
        assert any(r.rule_id == "HS-L3-CONTACT" for r in result.triggered_rules)

    def test_force_exceeded(self, engine):
        features = _make_features(HSFeatures(human_contact_force_exceeded_gt=True))
        result = engine.evaluate(features, "test_force")
        assert result.hs_level == RiskLevel.L3

    def test_gripper_near_human(self, engine):
        features = _make_features(HSFeatures(gripper_close_near_human_gt=True))
        result = engine.evaluate(features, "test_gripper")
        assert result.hs_level == RiskLevel.L3

    def test_stop_failure(self, engine):
        features = _make_features(HSFeatures(
            intrusion_event_flag=True,
            stop_success_gt=False,
        ))
        result = engine.evaluate(features, "test_stop")
        assert result.hs_level == RiskLevel.L3

    def test_distance_below_0_05m_active(self, engine):
        features = _make_features(HSFeatures(d_h_min_gt_m=0.03), robot_active=True)
        result = engine.evaluate(features, "test_dist")
        assert result.hs_level == RiskLevel.L3

    def test_distance_below_0_05m_inactive(self, engine):
        """Inactive robot with < 0.05 m should NOT trigger L3."""
        features = _make_features(HSFeatures(d_h_min_gt_m=0.03), robot_active=False)
        result = engine.evaluate(features, "test_inactive")
        # Should be L2 or lower, not L3
        assert result.hs_level != RiskLevel.L3


class TestHSL2:
    """Test HS L2 rules."""

    def test_proximity_0_05_0_10m(self, engine):
        features = _make_features(HSFeatures(d_h_min_gt_m=0.07))
        result = engine.evaluate(features, "test_prox")
        assert result.hs_level == RiskLevel.L2

    def test_stop_margin_low(self, engine):
        features = _make_features(HSFeatures(stop_margin_gt_s=0.3))
        result = engine.evaluate(features, "test_margin")
        assert result.hs_level == RiskLevel.L2

    def test_high_speed_near_human(self, engine):
        features = _make_features(HSFeatures(
            d_h_min_gt_m=0.12,
            v_rel_h_gt_mps=0.3,
        ))
        result = engine.evaluate(features, "test_speed")
        assert result.hs_level == RiskLevel.L2


class TestHSL1:
    """Test HS L1 rules."""

    def test_approach_0_10_0_15m(self, engine):
        features = _make_features(HSFeatures(d_h_min_gt_m=0.12))
        result = engine.evaluate(features, "test_approach")
        assert result.hs_level == RiskLevel.L1


class TestHSL0:
    """Test HS L0 (safe)."""

    def test_safe_distance(self, engine):
        features = _make_features(HSFeatures(d_h_min_gt_m=0.20))
        result = engine.evaluate(features, "test_safe")
        assert result.hs_level == RiskLevel.L0
