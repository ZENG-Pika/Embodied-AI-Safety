import json
from pathlib import Path

from safety_risk.policy_evaluator import evaluate_saved_policy


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _risk_report(path: Path, episode_id: str, overall: str, hs="L0", pt="L0", rs="L0", ir="L0"):
    _write_json(path, {
        "episode_id": episode_id,
        "risk_levels": {"HS": hs, "PT": pt, "RS": rs, "IR": ir, "overall": overall},
    })


def test_evaluate_saved_dp_rollouts(tmp_path):
    success_root = tmp_path / "output"
    failure_root = tmp_path / "failure_output"
    _risk_report(success_root / "ep1/safety_reports/ep1_risk.json", "ep1", "L1", hs="L1")
    _risk_report(success_root / "ep2/safety_reports/ep2_risk.json", "ep2", "L2", pt="L2")
    _write_json(failure_root / "scene/failure_manifest.json", {
        "status": "failed", "attempt_count": 6, "recorded_frames": 120,
    })
    _risk_report(failure_root / "scene/episode/safety_reports/ep3_risk.json", "ep3", "L3", hs="L3")

    result = evaluate_saved_policy(
        [success_root], [failure_root], policy_name="dp-test", safe_max_level="L1"
    )

    assert result["policy"]["type"] == "DP"
    assert result["summary"]["total_episodes"] == 3
    assert result["summary"]["task_success_rate"] == 2 / 3
    assert result["summary"]["safe_success_rate"] == 1 / 3
    assert result["summary"]["l3_rate_among_evaluated"] == 1 / 3
    assert result["risk_level_distribution"]["overall"] == {
        "L0": 0, "L1": 1, "L2": 1, "L3": 1,
    }
    failure = next(item for item in result["episodes"] if not item["task_success"])
    assert failure["attempt_count"] == 6
    assert failure["recorded_frames"] == 120


def test_missing_failure_safety_report_is_not_safe(tmp_path):
    failure_root = tmp_path / "failure_output"
    _write_json(failure_root / "scene/failure_manifest.json", {
        "status": "failed", "attempt_count": 6,
    })

    result = evaluate_saved_policy([], [failure_root])

    assert result["summary"]["total_episodes"] == 1
    assert result["summary"]["safe_successes"] == 0
    assert result["summary"]["safety_not_evaluated_episodes"] == 1
    assert result["summary"]["l3_rate_among_evaluated"] is None


def test_success_without_risk_report_is_retained(tmp_path):
    success_root = tmp_path / "output"
    _write_json(success_root / "ep1/sim_raw_gt.json", {"scenario_id": "test"})

    result = evaluate_saved_policy([success_root], minimum_episode_count=1)

    assert result["summary"]["total_episodes"] == 1
    assert result["summary"]["safety_not_evaluated_episodes"] == 1
    assert result["summary"]["input_complete"] is False
    assert any(item.startswith("success_safety_report_missing:") for item in result["warnings"])


def test_missing_root_and_small_sample_are_reported(tmp_path):
    result = evaluate_saved_policy(
        [tmp_path / "missing-success"], [tmp_path / "missing-failure"]
    )

    assert result["summary"]["evaluation_valid"] is False
    assert result["summary"]["input_complete"] is False
    assert result["summary"]["sample_size_sufficient"] is False
    assert any(item.startswith("failure_root_missing:") for item in result["warnings"])
    assert "insufficient_sample_size:0<20" in result["warnings"]


def test_duplicate_conflict_prefers_failure(tmp_path):
    success_root = tmp_path / "output"
    failure_root = tmp_path / "failure"
    _risk_report(success_root / "ep1/safety_reports/ep1_risk.json", "same", "L0")
    _write_json(failure_root / "failed/failure_manifest.json", {
        "status": "failed", "episode_id": "same",
    })

    result = evaluate_saved_policy(
        [success_root], [failure_root], minimum_episode_count=1
    )

    assert result["summary"]["total_episodes"] == 1
    assert result["summary"]["task_failures"] == 1
    assert any(item.startswith("duplicate_episode_id:same:") for item in result["warnings"])


def test_failure_report_must_match_manifest_episode(tmp_path):
    failure_root = tmp_path / "failure"
    base = failure_root / "failed"
    _write_json(base / "failure_manifest.json", {
        "status": "failed", "episode_id": "wanted",
    })
    _risk_report(base / "safety_reports/other_risk.json", "other", "L3")

    result = evaluate_saved_policy([], [failure_root], minimum_episode_count=1)

    assert result["episodes"][0]["episode_id"] == "wanted"
    assert result["episodes"][0]["safety_evaluated"] is False
    assert any(item.startswith("failure_risk_report_ambiguous:") for item in result["warnings"])


def test_invalid_failure_manifest_status_is_ignored(tmp_path):
    failure_root = tmp_path / "failure"
    _write_json(failure_root / "scene/failure_manifest.json", {"status": "running"})

    result = evaluate_saved_policy([], [failure_root], minimum_episode_count=1)

    assert result["summary"]["total_episodes"] == 0
    assert any(item.startswith("failure_manifest_invalid_status:") for item in result["warnings"])


def test_partial_risk_report_keeps_overall_evaluation(tmp_path):
    success_root = tmp_path / "output"
    _write_json(success_root / "ep1/safety_reports/ep1_risk.json", {
        "episode_id": "ep1",
        "risk_levels": {"HS": "L3", "PT": None, "RS": "L1", "IR": "L0", "overall": "L3"},
    })

    result = evaluate_saved_policy([success_root], minimum_episode_count=1)

    assert result["summary"]["safety_evaluated_episodes"] == 1
    assert result["summary"]["risk_categories_complete"] is False
    assert result["summary"]["overall_evaluation_valid"] is True
    assert result["summary"]["evaluation_valid"] is False
    assert result["risk_category_evaluated_episodes"]["PT"] == 0
    assert result["risk_category_evaluated_episodes"]["overall"] == 1
    assert result["summary"]["l3_rate_among_evaluated"] == 1.0
