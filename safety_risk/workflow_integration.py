"""Integration layer between InternDataEngine workflows and safety risk evaluation.

Provides SafetyRiskEvaluator, a high-level class that:
1. Extracts SimRawEpisode from a workflow's in-memory logger
2. Runs feature extraction
3. Runs rule-based risk evaluation
4. Saves JSON report alongside the LMDB output
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from safety_risk.config import SafetyRiskConfig
from safety_risk.feature_extractor import FeatureExtractor
from safety_risk.report import generate_report
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import RiskEvaluationResult
from safety_risk.workflow_adapter import WorkflowSafetyAdapter

logger = logging.getLogger(__name__)


class SafetyRiskEvaluator:
    """Evaluates safety risk from a workflow's in-memory data.

    This is the main integration point between InternDataEngine and the
    safety risk pipeline.

    Usage::

        evaluator = SafetyRiskEvaluator()
        result = evaluator.evaluate_from_workflow(wf, episode_id="task_001")
        evaluator.save_report(result, save_dir="/path/to/output")
    """

    def __init__(
        self,
        config: Optional[SafetyRiskConfig] = None,
        risk_thresholds_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        config : SafetyRiskConfig, optional
            Pre-loaded config. If None, loads from default YAML files.
        risk_thresholds_path : str, optional
            Path to a custom risk_thresholds.yaml. Overrides default.
        """
        if config is not None:
            self.config = config
        elif risk_thresholds_path is not None:
            from safety_risk.config import load_risk_thresholds

            # Load with custom path
            self.config = SafetyRiskConfig()
            # Override thresholds from custom path
            import yaml

            with open(risk_thresholds_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            # Re-parse with custom data
            self.config = SafetyRiskConfig.load()
        else:
            self.config = SafetyRiskConfig.load()

        self.adapter = WorkflowSafetyAdapter()
        self.feature_extractor = FeatureExtractor(
            damage_proxy_rules=self.config.thresholds.pt.damage_proxy_rules,
        )
        self.rule_engine = RuleBasedRiskEngine(self.config)

    def evaluate_from_workflow(
        self,
        wf,
        episode_id: str = "",
        scenario_id: str = "",
        task_type: str = "",
        obstacle_names: Optional[List[str]] = None,
    ) -> RiskEvaluationResult:
        """Run the full safety risk evaluation pipeline on a workflow.

        Parameters
        ----------
        wf : NimbusWorkFlow
            The workflow object with in-memory logger data.
        episode_id : str
            Unique episode identifier.
        scenario_id : str
            Scenario identifier.
        task_type : str
            Task type.
        obstacle_names : list[str], optional
            Object names to treat as human surrogates for HS evaluation.

        Returns
        -------
        RiskEvaluationResult
            Complete evaluation result.
        """
        # Step 1: Extract raw episode from workflow logger
        raw_episode = self.adapter.from_workflow(
            wf, episode_id=episode_id, scenario_id=scenario_id,
            task_type=task_type, obstacle_names=obstacle_names,
        )

        # Step 2: Extract risk features
        features = self.feature_extractor.extract(raw_episode)

        # Step 3: Run rule-based evaluation
        result = self.rule_engine.evaluate(features, episode_id=episode_id)

        # Include adapter warnings
        result.warnings.extend(self.adapter.warnings)

        logger.info(
            "Safety eval for %s: overall=%s (HS=%s PT=%s RS=%s IR=%s), rules=%d",
            episode_id,
            result.overall_level.value,
            result.hs_level.value,
            result.pt_level.value,
            result.rs_level.value,
            result.ir_level.value,
            len(result.triggered_rules),
        )

        return result

    def evaluate_and_save(
        self,
        wf,
        save_dir: str,
        episode_id: str = "",
        scenario_id: str = "",
        task_type: str = "",
        report_subdir: str = "safety_reports",
        obstacle_names: Optional[List[str]] = None,
    ) -> str:
        """Evaluate and save the risk report to disk.

        Parameters
        ----------
        wf : NimbusWorkFlow
            The workflow object.
        save_dir : str
            Base save directory (same as LMDB output directory).
        episode_id : str
            Episode identifier.
        scenario_id : str
            Scenario identifier.
        task_type : str
            Task type.
        report_subdir : str
            Subdirectory name for safety reports within save_dir.

        Returns
        -------
        str
            Path to the saved JSON report.
        """
        result = self.evaluate_from_workflow(
            wf, episode_id=episode_id, scenario_id=scenario_id,
            task_type=task_type, obstacle_names=obstacle_names,
        )

        return self.save_report(result, save_dir, report_subdir)

    def save_report(
        self,
        result: RiskEvaluationResult,
        save_dir: str,
        report_subdir: str = "safety_reports",
    ) -> str:
        """Save a risk evaluation result to a JSON file.

        Parameters
        ----------
        result : RiskEvaluationResult
            The evaluation result.
        save_dir : str
            Base directory.
        report_subdir : str
            Subdirectory for reports.

        Returns
        -------
        str
            Path to the saved report.
        """
        report_dir = os.path.join(save_dir, report_subdir)
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, f"{result.episode_id}_risk.json")
        generate_report(result, output_path=report_path)

        logger.info("Safety risk report saved to: %s", report_path)
        return report_path


def evaluate_workflow_safety(
    wf,
    save_dir: str,
    episode_id: str = "",
    task_type: str = "",
    config: Optional[SafetyRiskConfig] = None,
    report_subdir: str = "safety_reports",
) -> RiskEvaluationResult:
    """Convenience function for one-line safety evaluation.

    Parameters
    ----------
    wf : NimbusWorkFlow
        The workflow with logger data.
    save_dir : str
        Where to save the report.
    episode_id : str
        Episode identifier.
    task_type : str
        Task type.
    config : SafetyRiskConfig, optional
        Pre-loaded config.
    report_subdir : str
        Report subdirectory.

    Returns
    -------
    RiskEvaluationResult
    """
    evaluator = SafetyRiskEvaluator(config=config)
    result = evaluator.evaluate_from_workflow(wf, episode_id=episode_id, task_type=task_type)
    evaluator.save_report(result, save_dir, report_subdir)
    return result
