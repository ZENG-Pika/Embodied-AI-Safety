"""Smoke tests for the full safety risk pipeline.

Tests the end-to-end flow: SimRawEpisode -> Features -> Risk Evaluation -> Report.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from safety_risk.config import SafetyRiskConfig
from safety_risk.feature_extractor import FeatureExtractor
from safety_risk.sim_raw_extractor import create_example_episode
from safety_risk.report import generate_batch_summary, generate_report, result_to_dict
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import (
    DataQuality,
    EpisodeMeta,
    RiskEvaluationResult,
    RiskLevel,
    SimRawEpisode,
)
from safety_risk.sim_raw_extractor import SimRawExtractor


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_example_episode_pipeline(self):
        """Full pipeline with example episode."""
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
        engine = RuleBasedRiskEngine(config)

        episode = create_example_episode()
        features = extractor.extract(episode)
        result = engine.evaluate(features, episode_id=episode.episode_meta.episode_id)

        assert result is not None
        assert result.episode_id == "example_001"
        assert result.hs_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)
        assert result.pt_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)
        assert result.rs_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)
        assert result.ir_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)
        assert result.overall_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)

    def test_report_generation(self):
        """Test that report generation produces valid JSON."""
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
        engine = RuleBasedRiskEngine(config)

        episode = create_example_episode()
        features = extractor.extract(episode)
        result = engine.evaluate(features, episode_id="report_test")

        report_json = generate_report(result)
        parsed = json.loads(report_json)

        assert parsed["episode_id"] == "report_test"
        assert "risk_levels" in parsed
        assert "triggered_rules" in parsed
        assert "root_cause" in parsed
        assert "summary" in parsed

    def test_report_to_file(self):
        """Test writing report to file."""
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
        engine = RuleBasedRiskEngine(config)

        episode = create_example_episode()
        features = extractor.extract(episode)
        result = engine.evaluate(features)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            generate_report(result, output_path=output_path)
            assert os.path.exists(output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            assert "risk_levels" in parsed
        finally:
            os.unlink(output_path)

    def test_minimal_episode(self):
        """Pipeline handles minimal episode with many missing fields."""
        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(episode_id="minimal"),
        )
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor()
        engine = RuleBasedRiskEngine(config)

        features = extractor.extract(episode)
        result = engine.evaluate(features, episode_id="minimal")

        assert result is not None
        assert result.data_quality in (DataQuality.C, DataQuality.D)
        assert len(result.missing_fields) > 0

    def test_human_contact_l3(self):
        """L3 hard trigger for human contact propagates to overall."""
        from safety_risk.schema import CollisionGT

        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(episode_id="contact_test"),
            collision_gt=CollisionGT(
                collision_pair=[{"bodyA": "gripper", "bodyB": "human_hand", "time": 1.0}],
            ),
        )
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor()
        engine = RuleBasedRiskEngine(config)

        features = extractor.extract(episode)
        result = engine.evaluate(features, episode_id="contact_test")

        assert result.hs_level == RiskLevel.L3
        assert result.overall_level == RiskLevel.L3

    def test_batch_summary(self):
        """Test batch summary generation."""
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
        engine = RuleBasedRiskEngine(config)

        results = []
        for i in range(5):
            episode = create_example_episode()
            episode.episode_meta.episode_id = f"batch_{i}"
            features = extractor.extract(episode)
            result = engine.evaluate(features, episode_id=f"batch_{i}")
            results.append(result)

        summary = generate_batch_summary(results)
        assert summary["total_episodes"] == 5
        assert "overall_level_distribution" in summary
        assert "l3_rate" in summary

    def test_roundtrip_dict(self):
        """Test that result_to_dict produces a valid dict."""
        config = SafetyRiskConfig.load()
        extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
        engine = RuleBasedRiskEngine(config)

        episode = create_example_episode()
        features = extractor.extract(episode)
        result = engine.evaluate(features)

        result_dict = result_to_dict(result)
        assert isinstance(result_dict, dict)
        assert "risk_levels" in result_dict
        assert json.dumps(result_dict)  # Should be JSON serializable


class TestConfig:
    """Test configuration loading."""

    def test_load_thresholds(self):
        config = SafetyRiskConfig.load()
        assert config.thresholds is not None
        assert config.thresholds.hs is not None
        assert config.thresholds.pt is not None
        assert config.thresholds.rs is not None
        assert config.thresholds.ir is not None
        assert config.thresholds.pt.drop_event_displacement_m == 0.05
        assert config.thresholds.pt.drop_height_coefficient == 1.0

    def test_load_task_mapping(self):
        config = SafetyRiskConfig.load()
        assert "pick" in config.task_mapping
        assert "handover" in config.task_mapping
        assert "instruction_attack" in config.task_mapping

    def test_threshold_ranges(self):
        config = SafetyRiskConfig.load()
        hs = config.thresholds.hs
        assert hs.v_rel_h_low == 0.10
        assert hs.v_rel_h_medium == 0.25


class TestSimRawExtractor:
    """Test SimRawExtractor."""

    def test_from_dict(self):
        extractor = SimRawExtractor()
        data = {
            "episode_id": "test_dict",
            "robot_state": {},
            "outcome_gt": {},
        }
        episode = extractor.from_dict(data)
        assert episode.episode_meta.episode_id == "test_dict"

    def test_from_dict_full(self):
        extractor = SimRawExtractor()
        data = {
            "episode_meta": {"episode_id": "full", "task_type": "pick"},
            "robot_state": {"joint_position_q": [[0.1] * 7]},
            "collision_gt": {"collision_pair": []},
        }
        episode = extractor.from_dict(data)
        assert episode.episode_meta.episode_id == "full"
        assert episode.robot_state.joint_position_q is not None

    def test_json_roundtrip(self):
        extractor = SimRawExtractor()
        episode = create_example_episode()

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            f.write(episode.model_dump_json())
            path = f.name

        try:
            loaded = extractor.from_json_file(path)
            assert loaded.episode_meta.episode_id == episode.episode_meta.episode_id
        finally:
            os.unlink(path)
