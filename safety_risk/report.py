"""Report generation for safety risk evaluation results.

Produces JSON reports with triggered rules, evidence, root causes,
and risk levels. Reports are self-contained and auditable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from safety_risk.schema import RiskEvaluationResult, RiskLevel


def result_to_dict(result: RiskEvaluationResult) -> Dict[str, Any]:
    """Convert a RiskEvaluationResult to a JSON-serializable dict."""
    report: Dict[str, Any] = {
        "episode_id": result.episode_id,
        "timestamp": result.timestamp,
        "risk_levels": {
            "HS": result.hs_level.value,
            "PT": result.pt_level.value,
            "RS": result.rs_level.value,
            "IR": result.ir_level.value,
            "overall": result.overall_level.value,
        },
        "triggered_rules": [
            {
                "rule_id": r.rule_id,
                "risk_category": r.risk_category.value,
                "level": r.level.value,
                "description": r.description,
                "evidence": r.evidence,
            }
            for r in result.triggered_rules
        ],
        "root_cause": result.root_cause,
        "data_quality": result.data_quality.value,
        "missing_fields": result.missing_fields,
        "warnings": result.warnings,
        "summary": _build_summary(result),
    }

    # Include features if available
    if result.features is not None:
        report["features"] = _features_to_dict(result.features)

    return report


def _build_summary(result: RiskEvaluationResult) -> Dict[str, Any]:
    """Build a human-readable summary of the evaluation."""
    level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    for r in result.triggered_rules:
        level_counts[r.level.value] += 1

    return {
        "overall_level": result.overall_level.value,
        "total_rules_triggered": len(result.triggered_rules),
        "level_distribution": level_counts,
        "unique_root_causes": list(set(result.root_cause)),
        "has_l3_hard_trigger": any(r.level == RiskLevel.L3 for r in result.triggered_rules),
        "data_quality": result.data_quality.value,
    }


def _features_to_dict(features) -> Dict[str, Any]:
    """Convert RiskFeatures to dict, excluding None values for cleanliness."""
    result = {}
    for section_name in ("common", "hs", "pt", "rs", "ir"):
        section = getattr(features, section_name)
        if section is None:
            continue
        section_dict = {}
        for field_name, field_value in section:
            if field_value is not None and field_value != [] and field_value != 0.0 and field_value != 0 and field_value is not False:
                section_dict[field_name] = _serialize_value(field_value)
            elif field_name in ("missing_fields", "warnings"):
                # Always include these even if empty
                section_dict[field_name] = field_value
        result[section_name] = section_dict
    return result


def _serialize_value(val: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if hasattr(val, "value"):  # Enum
        return val.value
    return str(val)


def generate_report(
    result: RiskEvaluationResult,
    output_path: Optional[str] = None,
    indent: int = 2,
) -> str:
    """Generate a JSON report and optionally write to file.

    Parameters
    ----------
    result : RiskEvaluationResult
        The evaluation result to report.
    output_path : str, optional
        Path to write the JSON file. If None, only returns the string.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        JSON string of the report.
    """
    report_dict = result_to_dict(result)
    json_str = json.dumps(report_dict, indent=indent, ensure_ascii=False)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str


def generate_batch_summary(results: List[RiskEvaluationResult]) -> Dict[str, Any]:
    """Generate a summary report for a batch of evaluation results.

    Parameters
    ----------
    results : list[RiskEvaluationResult]
        List of evaluation results.

    Returns
    -------
    dict
        Batch summary with statistics.
    """
    if not results:
        return {"total_episodes": 0}

    level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    category_counts = {"HS": {"L0": 0, "L1": 0, "L2": 0, "L3": 0},
                       "PT": {"L0": 0, "L1": 0, "L2": 0, "L3": 0},
                       "RS": {"L0": 0, "L1": 0, "L2": 0, "L3": 0},
                       "IR": {"L0": 0, "L1": 0, "L2": 0, "L3": 0}}
    all_root_causes: List[str] = []
    l3_episodes: List[str] = []

    for r in results:
        level_counts[r.overall_level.value] += 1
        category_counts["HS"][r.hs_level.value] += 1
        category_counts["PT"][r.pt_level.value] += 1
        category_counts["RS"][r.rs_level.value] += 1
        category_counts["IR"][r.ir_level.value] += 1
        all_root_causes.extend(r.root_cause)
        if r.overall_level == RiskLevel.L3:
            l3_episodes.append(r.episode_id)

    # Count root cause frequency
    cause_freq: Dict[str, int] = {}
    for c in all_root_causes:
        cause_freq[c] = cause_freq.get(c, 0) + 1

    return {
        "total_episodes": len(results),
        "overall_level_distribution": level_counts,
        "category_level_distribution": category_counts,
        "l3_episodes": l3_episodes,
        "root_cause_frequency": dict(sorted(cause_freq.items(), key=lambda x: -x[1])),
        "l3_rate": level_counts["L3"] / len(results) if results else 0.0,
    }
