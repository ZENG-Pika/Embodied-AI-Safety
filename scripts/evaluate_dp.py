#!/usr/bin/env python3
"""Evaluate saved Diffusion Policy rollouts for task success and safety."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_risk.policy_evaluator import evaluate_saved_policy, save_policy_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate task completion and L0-L3 safety ratings for saved DP rollouts."
    )
    parser.add_argument(
        "--success-root", action="append", type=Path, required=True,
        help="Directory containing successful episodes; repeatable.",
    )
    parser.add_argument(
        "--failure-root", action="append", type=Path, default=[],
        help="Directory containing failure_manifest.json files; repeatable.",
    )
    parser.add_argument("--policy-name", default="diffusion_policy")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--safe-max-level", choices=("L0", "L1", "L2", "L3"), default="L1")
    parser.add_argument(
        "--minimum-episodes", type=int, default=20,
        help="Minimum sample count required before the aggregate is marked valid.",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation_output/dp_evaluation.json"))
    args = parser.parse_args()

    result = evaluate_saved_policy(
        args.success_root,
        args.failure_root,
        policy_name=args.policy_name,
        checkpoint=args.checkpoint,
        safe_max_level=args.safe_max_level,
        minimum_episode_count=args.minimum_episodes,
    )
    save_policy_evaluation(result, args.output)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    print(f"Full report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
