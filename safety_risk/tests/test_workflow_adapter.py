"""Tests for workflow_adapter.py and workflow_integration.py."""

from __future__ import annotations

import pytest

from safety_risk.workflow_adapter import WorkflowSafetyAdapter
from safety_risk.workflow_integration import SafetyRiskEvaluator, evaluate_workflow_safety
from safety_risk.schema import RiskLevel


class MockLogger:
    """Mock logger that mimics BaseLogger / LmdbLogger in-memory data."""

    def __init__(self):
        self.proprio_data_logger = {}
        self.object_data_logger = {}
        self.scalar_data_logger = {}
        self.json_data_logger = {}
        self.language_instruction = ["Test instruction"]
        self.detailed_language_instruction = ["Detailed test"]
        self.log_num_steps = 10


class MockWorkflow:
    """Mock workflow that mimics SimBoxDualWorkFlow."""

    def __init__(self, logger=None):
        self.logger = logger

    def get_task_name(self):
        return "test_task"


class TestWorkflowAdapter:
    """Test WorkflowSafetyAdapter."""

    def test_no_logger(self):
        """Adapting a workflow without a logger should produce a minimal episode."""
        adapter = WorkflowSafetyAdapter()
        wf = MockWorkflow(logger=None)
        episode = adapter.from_workflow(wf, episode_id="no_logger")

        assert episode.episode_meta.episode_id == "no_logger"
        assert len(adapter.warnings) > 0

    def test_empty_logger(self):
        """Adapting an empty logger should produce a minimal episode."""
        adapter = WorkflowSafetyAdapter()
        mock_logger = MockLogger()
        wf = MockWorkflow(logger=mock_logger)

        episode = adapter.from_workflow(wf, episode_id="empty")

        assert episode.episode_meta.episode_id == "empty"
        assert episode.robot_state.joint_position_q is None

    def test_with_proprio_data(self):
        """Proprio data should be extracted correctly."""
        adapter = WorkflowSafetyAdapter()
        mock_logger = MockLogger()

        # Simulate logged proprio data
        mock_logger.proprio_data_logger = {
            "franka_0": {
                "states.joint.position": [[0.1] * 7, [0.2] * 7, [0.3] * 7],
                "states.gripper.pose": [
                    [0.5, 0.0, 0.3, 0.0, 0.707, 0.0, 0.707],
                    [0.5, 0.01, 0.3, 0.0, 0.707, 0.0, 0.707],
                    [0.5, 0.02, 0.3, 0.0, 0.707, 0.0, 0.707],
                ],
                "states.gripper.position": [0.04, 0.03, 0.02],
            },
        }

        wf = MockWorkflow(logger=mock_logger)
        episode = adapter.from_workflow(wf, episode_id="proprio_test")

        assert episode.robot_state.joint_position_q is not None
        assert len(episode.robot_state.joint_position_q) == 3
        assert episode.robot_state.ee_pose is not None
        assert len(episode.robot_state.ee_pose) == 3

    def test_with_object_data(self):
        """Object data should be extracted correctly."""
        adapter = WorkflowSafetyAdapter()
        mock_logger = MockLogger()

        mock_logger.object_data_logger = {
            "franka_0": {
                "cup_01/pose": [
                    [0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 1.0],
                    [0.5, -0.28, 0.8, 0.0, 0.0, 0.0, 1.0],
                ],
            },
        }

        wf = MockWorkflow(logger=mock_logger)
        episode = adapter.from_workflow(wf, episode_id="obj_test")

        assert episode.object_state.object_pose is not None
        assert "cup_01" in episode.object_state.object_pose

    def test_with_json_data(self):
        """JSON data (HRI/planner info) should be extracted."""
        adapter = WorkflowSafetyAdapter()
        mock_logger = MockLogger()

        mock_logger.json_data_logger = {
            "franka_0": {
                "user_command_text": "Pick up the cup",
                "unsafe_instruction_flag": False,
                "stop_command_obeyed": True,
            },
        }

        wf = MockWorkflow(logger=mock_logger)
        episode = adapter.from_workflow(wf, episode_id="json_test")

        assert episode.hri_log.user_command_text == "Pick up the cup"
        assert episode.hri_log.unsafe_instruction_flag is False
        assert episode.hri_log.stop_command_obeyed is True

    def test_warnings_recorded(self):
        """Missing M0 fields should generate warnings."""
        adapter = WorkflowSafetyAdapter()
        mock_logger = MockLogger()
        mock_logger.proprio_data_logger = {"franka_0": {}}

        wf = MockWorkflow(logger=mock_logger)
        episode = adapter.from_workflow(wf, episode_id="warn_test")

        # Should have warnings about missing M0 fields
        assert len(adapter.warnings) > 0
        warning_text = " ".join(adapter.warnings)
        assert "M0" in warning_text or "not available" in warning_text


class TestWorkflowIntegration:
    """Test SafetyRiskEvaluator with mock data."""

    def test_evaluate_from_workflow(self):
        """Full evaluation from mock workflow should produce a result."""
        mock_logger = MockLogger()
        mock_logger.proprio_data_logger = {
            "franka_0": {
                "states.joint.position": [[0.1] * 7 for _ in range(50)],
                "states.gripper.pose": [
                    [0.5, 0.0, 0.3, 0.0, 0.707, 0.0, 0.707] for _ in range(50)
                ],
            },
        }
        mock_logger.json_data_logger = {
            "franka_0": {
                "user_command_text": "Pick up the cup",
            },
        }

        wf = MockWorkflow(logger=mock_logger)
        evaluator = SafetyRiskEvaluator()

        result = evaluator.evaluate_from_workflow(
            wf, episode_id="integration_test", task_type="pick"
        )

        assert result is not None
        assert result.episode_id == "integration_test"
        assert result.hs_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)
        assert result.pt_level in (RiskLevel.L0, RiskLevel.L1, RiskLevel.L2, RiskLevel.L3)

    def test_evaluate_and_save(self, tmp_path):
        """Evaluation should save a JSON report to disk."""
        mock_logger = MockLogger()
        mock_logger.proprio_data_logger = {
            "franka_0": {
                "states.joint.position": [[0.1] * 7 for _ in range(10)],
            },
        }

        wf = MockWorkflow(logger=mock_logger)
        evaluator = SafetyRiskEvaluator()

        report_path = evaluator.evaluate_and_save(
            wf, save_dir=str(tmp_path), episode_id="save_test"
        )

        assert report_path is not None
        import os
        assert os.path.exists(report_path)

        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["episode_id"] == "save_test"
        assert "risk_levels" in report

    def test_convenience_function(self, tmp_path):
        """evaluate_workflow_safety convenience function should work."""
        mock_logger = MockLogger()
        wf = MockWorkflow(logger=mock_logger)

        result = evaluate_workflow_safety(
            wf, save_dir=str(tmp_path), episode_id="conv_test"
        )

        assert result is not None
        assert result.episode_id == "conv_test"
