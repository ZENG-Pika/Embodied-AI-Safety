"""Tests for feature_extractor.py."""

from __future__ import annotations

import math

import pytest

from safety_risk.feature_extractor import (
    FeatureExtractor,
    _compute_ttc,
    _count_time_below,
    _infer_damage_from_proxy,
    _min_distance_from_matrix,
    _safe_min,
    _safe_max,
)
from safety_risk.sim_raw_extractor import create_example_episode
from safety_risk.schema import (
    CollisionGT,
    DistanceGT,
    EpisodeMeta,
    GripperGT,
    HSFeatures,
    OutcomeGT,
    PTFeatures,
    RiskFeatures,
    RSFeatures,
    SimRawEpisode,
)


class TestHelpers:
    """Test utility functions."""

    def test_safe_min_basic(self):
        assert _safe_min([1.0, 2.0, 3.0]) == 1.0

    def test_safe_min_with_none(self):
        assert _safe_min([1.0, None, 3.0]) == 1.0

    def test_safe_min_empty(self):
        assert _safe_min([]) is None

    def test_safe_min_none_input(self):
        assert _safe_min(None) is None

    def test_safe_min_default(self):
        assert _safe_min([], default=5.0) == 5.0

    def test_safe_max_basic(self):
        assert _safe_max([1.0, 2.0, 3.0]) == 3.0

    def test_safe_max_empty(self):
        assert _safe_max([]) == 0.0

    def test_count_time_below(self):
        values = [0.20, 0.10, 0.05, 0.15, 0.03]
        assert _count_time_below(values, 0.10, dt=0.02) == pytest.approx(0.04)

    def test_count_time_below_none(self):
        assert _count_time_below(None, 10.0) == 0.0

    def test_min_distance_from_matrix_flat(self):
        assert _min_distance_from_matrix([10.0, 5.0, 20.0]) == 5.0

    def test_min_distance_from_matrix_nested(self):
        assert _min_distance_from_matrix([[10.0, 5.0], [20.0, 3.0]]) == 3.0

    def test_min_distance_from_matrix_none(self):
        assert _min_distance_from_matrix(None) is None

    def test_compute_ttc(self):
        assert _compute_ttc(0.10, 0.5) == pytest.approx(0.2)

    def test_compute_ttc_zero_velocity(self):
        assert _compute_ttc(0.10, 0.0) is None

    def test_compute_ttc_negative_velocity(self):
        assert _compute_ttc(0.10, -0.5) is None

    def test_infer_damage_from_proxy_high_drop_fragile(self):
        flag, severity = _infer_damage_from_proxy(0.60, 0.0, 0.0, "extreme")
        assert flag is True
        assert severity == "broken"

    def test_infer_damage_from_proxy_low_drop(self):
        flag, severity = _infer_damage_from_proxy(0.05, 0.0, 0.0, "low")
        assert flag is False
        assert severity == "none"

    def test_infer_damage_from_proxy_high_impulse(self):
        flag, severity = _infer_damage_from_proxy(None, 25.0, 0.0, "high")
        assert flag is True
        assert severity == "broken"


class TestFeatureExtractor:
    """Test the FeatureExtractor class."""

    def test_extract_example_episode(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        assert features is not None
        assert features.common.robot_active is True
        assert features.hs is not None
        assert features.pt is not None
        assert features.rs is not None
        assert features.ir is not None

    def test_hs_distances(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        # Should have computed distances
        assert features.hs.d_ee_h_min_gt_m is not None
        assert features.hs.d_ee_h_min_gt_m > 0

    def test_hs_time_below_thresholds(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        # Distances decrease from 0.50 m to about 0.20 m.
        assert features.hs.time_d_h_below_0_15m_s >= 0

    def test_pt_features(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        assert features.pt.drop_flag_gt is False
        assert features.pt.damage_flag_gt is False
        assert features.pt.stable_final_gt is True

    def test_rs_features(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        # No collisions in example
        assert features.rs.robot_env_collision_flag_gt is False
        assert features.rs.self_collision_flag_gt is False

    def test_ir_features(self):
        episode = create_example_episode()
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        # No unsafe instructions in example
        assert features.ir.unsafe_instruction_flag_gt is False
        assert features.ir.blind_action_flag_sim is False

    def test_missing_fields_recorded(self):
        """Missing M0 fields should be recorded in common.missing_fields."""
        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(episode_id="minimal"),
            # Everything else defaults to empty/None
        )
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        assert len(features.common.missing_fields) > 0
        assert features.common.data_quality.value in ("C", "D")

    def test_collision_detection(self):
        """Test that collision events are properly detected."""
        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(episode_id="collision_test"),
            collision_gt=CollisionGT(
                collision_pair=[
                    {"bodyA": "robot_link_5", "bodyB": "table", "time": 1.0},
                    {"bodyA": "gripper", "bodyB": "human_hand", "time": 1.5},
                ],
                contact_impulse=[5.0, 20.0],
            ),
        )
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        assert features.hs.human_contact_flag_gt is True
        assert features.rs.robot_env_collision_flag_gt is True

    def test_drop_damage_proxy(self):
        """Test damage proxy inference when no damage model."""
        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(
                episode_id="drop_test",
                object_fragility_class="extreme",
            ),
            outcome_gt=OutcomeGT(
                drop_event=True,
                drop_height=60.0,
            ),
        )
        extractor = FeatureExtractor()
        features = extractor.extract(episode)

        assert features.pt.drop_flag_gt is True
        assert features.pt.damage_flag_gt is True
        assert features.pt.damage_severity_gt == "broken"
