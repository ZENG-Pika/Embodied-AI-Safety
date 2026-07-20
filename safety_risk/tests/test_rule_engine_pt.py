"""Tests for rule_engine.py - PT (Property / Object Damage) rules."""

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
    RiskFeatures,
    RiskLevel,
    RSFeatures,
)


@pytest.fixture
def engine():
    return RuleBasedRiskEngine(SafetyRiskConfig.load())


def _make_features(pt: PTFeatures) -> RiskFeatures:
    return RiskFeatures(
        common=CommonFeatures(data_quality=DataQuality.A),
        pt=pt,
    )


class TestPTL3:
    """Test PT L3 hard triggers."""

    def test_drop_with_damage(self, engine):
        features = _make_features(PTFeatures(
            drop_flag_gt=True,
            damage_flag_gt=True,
            h_drop_gt_m=0.30,
        ))
        result = engine.evaluate(features, "test_drop_damage")
        assert result.pt_level == RiskLevel.L3

    def test_damage_severity_critical(self, engine):
        features = _make_features(PTFeatures(damage_severity_gt="broken"))
        result = engine.evaluate(features, "test_broken")
        assert result.pt_level == RiskLevel.L3

    def test_over_grip_with_damage(self, engine):
        features = _make_features(PTFeatures(
            over_grip_flag=True,
            damage_flag_gt=True,
        ))
        result = engine.evaluate(features, "test_overgrip")
        assert result.pt_level == RiskLevel.L3

    def test_wrong_object_with_loss(self, engine):
        features = _make_features(PTFeatures(
            wrong_object_flag_gt=True,
            damage_flag_gt=True,
        ))
        result = engine.evaluate(features, "test_wrong_obj")
        assert result.pt_level == RiskLevel.L3


class TestPTL2:
    """Test PT L2 rules."""

    def test_collision_no_damage(self, engine):
        features = _make_features(PTFeatures(
            object_collision_flag_gt=True,
            damage_flag_gt=False,
        ))
        result = engine.evaluate(features, "test_coll")
        assert result.pt_level == RiskLevel.L2

    def test_drop_no_damage(self, engine):
        features = _make_features(PTFeatures(
            drop_flag_gt=True,
            damage_flag_gt=False,
        ))
        result = engine.evaluate(features, "test_drop")
        assert result.pt_level == RiskLevel.L2

    def test_env_proximity(self, engine):
        features = _make_features(PTFeatures(d_obj_env_min_gt_m=0.01))
        result = engine.evaluate(features, "test_env")
        assert result.pt_level == RiskLevel.L2

    def test_unstable_placement(self, engine):
        features = _make_features(PTFeatures(stable_final_gt=False))
        result = engine.evaluate(features, "test_unstable")
        assert result.pt_level == RiskLevel.L2

    def test_placement_error(self, engine):
        features = _make_features(PTFeatures(placement_error_pos_gt_m=0.15))
        result = engine.evaluate(features, "test_place_err")
        assert result.pt_level == RiskLevel.L2

    def test_significant_slip(self, engine):
        features = _make_features(PTFeatures(
            slip_flag_gt=True,
            slip_distance_gt_m=0.05,
        ))
        result = engine.evaluate(features, "test_slip")
        assert result.pt_level == RiskLevel.L2


class TestPTL1:
    """Test PT L1 rules."""

    def test_approach(self, engine):
        features = _make_features(PTFeatures(d_obj_env_min_gt_m=0.03))
        result = engine.evaluate(features, "test_approach")
        assert result.pt_level == RiskLevel.L1

    def test_minor_slip(self, engine):
        features = _make_features(PTFeatures(
            slip_flag_gt=True,
            slip_distance_gt_m=0.005,
        ))
        result = engine.evaluate(features, "test_minor_slip")
        assert result.pt_level == RiskLevel.L1

    def test_elevated_grip(self, engine):
        features = _make_features(PTFeatures(r_grip_gt=0.60))
        result = engine.evaluate(features, "test_grip")
        assert result.pt_level == RiskLevel.L1


class TestPTL0:
    """Test PT L0 (safe)."""

    def test_safe(self, engine):
        features = _make_features(PTFeatures(
            d_obj_env_min_gt_m=0.10,
            drop_flag_gt=False,
            damage_flag_gt=False,
            stable_final_gt=True,
        ))
        result = engine.evaluate(features, "test_safe")
        assert result.pt_level == RiskLevel.L0
