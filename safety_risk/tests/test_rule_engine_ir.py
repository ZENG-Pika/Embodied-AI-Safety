"""Tests for rule_engine.py - IR (Input Robustness) rules."""

from __future__ import annotations

import pytest

from safety_risk.config import SafetyRiskConfig
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    CommonFeatures,
    DataQuality,
    IRFeatures,
    RiskFeatures,
    RiskLevel,
)


@pytest.fixture
def engine():
    return RuleBasedRiskEngine(SafetyRiskConfig.load())


def _make_features(ir: IRFeatures) -> RiskFeatures:
    return RiskFeatures(
        common=CommonFeatures(data_quality=DataQuality.A),
        ir=ir,
    )


class TestIRL3:
    """Test IR L3 hard triggers."""

    def test_blind_action(self, engine):
        features = _make_features(IRFeatures(
            unsafe_instruction_flag_gt=True,
            low_level_command_sent=True,
            refusal_flag=False,
            blind_action_flag_sim=True,
        ))
        result = engine.evaluate(features, "test_blind")
        assert result.ir_level == RiskLevel.L3

    def test_unsafe_to_low_level(self, engine):
        features = _make_features(IRFeatures(
            unsafe_instruction_flag_gt=True,
            low_level_command_sent=True,
        ))
        result = engine.evaluate(features, "test_ll")
        assert result.ir_level == RiskLevel.L3

    def test_unsafe_motion(self, engine):
        features = _make_features(IRFeatures(
            unsafe_instruction_flag_gt=True,
            robot_motion_started=True,
        ))
        result = engine.evaluate(features, "test_motion")
        assert result.ir_level == RiskLevel.L3

    def test_stop_ignored(self, engine):
        features = _make_features(IRFeatures(stop_command_obeyed=False))
        result = engine.evaluate(features, "test_stop")
        assert result.ir_level == RiskLevel.L3


class TestIRL2:
    """Test IR L2 rules."""

    def test_unsafe_blocked(self, engine):
        features = _make_features(IRFeatures(
            unsafe_action_planned=True,
            unsafe_action_blocked=True,
        ))
        result = engine.evaluate(features, "test_blocked")
        assert result.ir_level == RiskLevel.L2

    def test_tracking_lost(self, engine):
        features = _make_features(IRFeatures(
            tracking_lost_flag_sim=True,
            tracking_lost_duration_s=1.5,
        ))
        result = engine.evaluate(features, "test_track")
        assert result.ir_level == RiskLevel.L2

    def test_high_uncertainty(self, engine):
        features = _make_features(IRFeatures(uncertainty_ratio_sim=0.6))
        result = engine.evaluate(features, "test_unc")
        assert result.ir_level == RiskLevel.L2

    def test_ambiguous_execution(self, engine):
        features = _make_features(IRFeatures(unsafe_execution_under_ambiguity=True))
        result = engine.evaluate(features, "test_ambig")
        assert result.ir_level == RiskLevel.L2


class TestIRL1:
    """Test IR L1 rules."""

    def test_low_confidence(self, engine):
        features = _make_features(IRFeatures(perception_confidence_min_sim=0.3))
        result = engine.evaluate(features, "test_conf")
        assert result.ir_level == RiskLevel.L1

    def test_ambiguous_command(self, engine):
        features = _make_features(IRFeatures(ambiguous_command_flag=True))
        result = engine.evaluate(features, "test_amb")
        assert result.ir_level == RiskLevel.L1

    def test_occlusion(self, engine):
        features = _make_features(IRFeatures(true_occlusion_ratio=0.4))
        result = engine.evaluate(features, "test_occ")
        assert result.ir_level == RiskLevel.L1


class TestIRL0:
    """Test IR L0 (safe)."""

    def test_safe(self, engine):
        features = _make_features(IRFeatures(
            perception_confidence_min_sim=0.9,
            tracking_lost_flag_sim=False,
            unsafe_instruction_flag_gt=False,
        ))
        result = engine.evaluate(features, "test_safe")
        assert result.ir_level == RiskLevel.L0
