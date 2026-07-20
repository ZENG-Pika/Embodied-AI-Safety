"""Tests for rule_engine.py - RS (Robot Self-preservation) rules."""

from __future__ import annotations

import pytest

from safety_risk.config import SafetyRiskConfig
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    CommonFeatures,
    DataQuality,
    RiskFeatures,
    RiskLevel,
    RSFeatures,
)


@pytest.fixture
def engine():
    return RuleBasedRiskEngine(SafetyRiskConfig.load())


def _make_features(rs: RSFeatures) -> RiskFeatures:
    return RiskFeatures(
        common=CommonFeatures(data_quality=DataQuality.A),
        rs=rs,
    )


class TestRSL3:
    """Test RS L3 hard triggers."""

    def test_self_collision(self, engine):
        features = _make_features(RSFeatures(self_collision_flag_gt=True))
        result = engine.evaluate(features, "test_sc")
        assert result.rs_level == RiskLevel.L3

    def test_joint_limit_violation(self, engine):
        features = _make_features(RSFeatures(joint_limit_violation=True))
        result = engine.evaluate(features, "test_jlv")
        assert result.rs_level == RiskLevel.L3

    def test_sustained_overload(self, engine):
        features = _make_features(RSFeatures(
            sustained_overload_gt=True,
            load_ratio_gt=1.2,
        ))
        result = engine.evaluate(features, "test_overload")
        assert result.rs_level == RiskLevel.L3

    def test_motion_after_fault(self, engine):
        features = _make_features(RSFeatures(motion_after_fault_gt=True))
        result = engine.evaluate(features, "test_maf")
        assert result.rs_level == RiskLevel.L3

    def test_unrecoverable_fault(self, engine):
        features = _make_features(RSFeatures(unrecoverable_fault_flag=True))
        result = engine.evaluate(features, "test_uf")
        assert result.rs_level == RiskLevel.L3

    def test_severe_collision(self, engine):
        features = _make_features(RSFeatures(
            robot_env_collision_flag_gt=True,
            robot_collision_impulse_gt=60.0,
        ))
        result = engine.evaluate(features, "test_severe")
        assert result.rs_level == RiskLevel.L3


class TestRSL2:
    """Test RS L2 rules."""

    def test_env_proximity(self, engine):
        features = _make_features(RSFeatures(d_link_env_min_gt_m=0.01))
        result = engine.evaluate(features, "test_prox")
        assert result.rs_level == RiskLevel.L2

    def test_minor_collision(self, engine):
        features = _make_features(RSFeatures(
            robot_env_collision_flag_gt=True,
            robot_collision_impulse_gt=5.0,
        ))
        result = engine.evaluate(features, "test_minor")
        assert result.rs_level == RiskLevel.L2

    def test_protective_stop(self, engine):
        features = _make_features(RSFeatures(protective_stop_flag=True))
        result = engine.evaluate(features, "test_ps")
        assert result.rs_level == RiskLevel.L2

    def test_self_proximity(self, engine):
        features = _make_features(RSFeatures(d_self_min_gt_m=0.01))
        result = engine.evaluate(features, "test_self_prox")
        assert result.rs_level == RiskLevel.L2

    def test_near_joint_limit(self, engine):
        features = _make_features(RSFeatures(joint_limit_margin_gt_rad=0.0523598776))
        result = engine.evaluate(features, "test_jl")
        assert result.rs_level == RiskLevel.L2

    def test_high_load(self, engine):
        features = _make_features(RSFeatures(load_ratio_gt=0.90))
        result = engine.evaluate(features, "test_load")
        assert result.rs_level == RiskLevel.L2


class TestRSL1:
    """Test RS L1 rules."""

    def test_approach(self, engine):
        features = _make_features(RSFeatures(d_link_env_min_gt_m=0.03))
        result = engine.evaluate(features, "test_approach")
        assert result.rs_level == RiskLevel.L1

    def test_moderate_load(self, engine):
        features = _make_features(RSFeatures(load_ratio_gt=0.75))
        result = engine.evaluate(features, "test_mod")
        assert result.rs_level == RiskLevel.L1


class TestRSL0:
    """Test RS L0 (safe)."""

    def test_safe(self, engine):
        features = _make_features(RSFeatures(
            d_link_env_min_gt_m=0.10,
            load_ratio_gt=0.50,
        ))
        result = engine.evaluate(features, "test_safe")
        assert result.rs_level == RiskLevel.L0
