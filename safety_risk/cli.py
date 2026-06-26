"""CLI interface for the safety risk evaluation pipeline.

Usage:
    python -m safety_risk.cli evaluate <input_json> [--output <output_json>]
    python -m safety_risk.cli batch <input_dir> [--output <output_dir>]
    python -m safety_risk.cli example [--output <output_json>]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from safety_risk.config import SafetyRiskConfig
from safety_risk.feature_extractor import FeatureExtractor
from safety_risk.sim_raw_extractor import create_example_episode
from safety_risk.report import generate_batch_summary, generate_report
from safety_risk.rule_engine import RuleBasedRiskEngine
from safety_risk.schema import RiskEvaluationResult
from safety_risk.sim_raw_extractor import SimRawExtractor

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a single episode from a JSON file."""
    config = SafetyRiskConfig.load()
    raw_extractor = SimRawExtractor()
    feature_extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
    engine = RuleBasedRiskEngine(config)

    # Load episode
    episode = raw_extractor.from_json_file(args.input)

    # Extract features
    features = feature_extractor.extract(episode)

    # Evaluate
    result = engine.evaluate(features, episode_id=episode.episode_meta.episode_id)

    # Generate report
    output_path = args.output or args.input.replace(".json", "_risk_report.json")
    report_json = generate_report(result, output_path=output_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Risk Evaluation: {result.episode_id}")
    print(f"{'='*60}")
    print(f"  HS: {result.hs_level.value}")
    print(f"  PT: {result.pt_level.value}")
    print(f"  RS: {result.rs_level.value}")
    print(f"  IR: {result.ir_level.value}")
    print(f"  Overall: {result.overall_level.value}")
    print(f"  Rules triggered: {len(result.triggered_rules)}")
    print(f"  Data quality: {result.data_quality.value}")
    if result.root_cause:
        print(f"  Root causes: {', '.join(set(result.root_cause))}")
    print(f"  Report: {output_path}")
    print(f"{'='*60}\n")

    if args.verbose:
        print(report_json)


def cmd_batch(args: argparse.Namespace) -> None:
    """Evaluate all episodes in a directory."""
    config = SafetyRiskConfig.load()
    raw_extractor = SimRawExtractor()
    feature_extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
    engine = RuleBasedRiskEngine(config)

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, "risk_reports")
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON episode files
    json_files = sorted(Path(input_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    results: List[RiskEvaluationResult] = []
    for json_file in json_files:
        if "_risk_report" in json_file.name:
            continue  # Skip report files

        try:
            episode = raw_extractor.from_json_file(str(json_file))
            features = feature_extractor.extract(episode)
            result = engine.evaluate(features, episode_id=episode.episode_meta.episode_id)
            results.append(result)

            # Write individual report
            report_path = os.path.join(output_dir, f"{json_file.stem}_risk_report.json")
            generate_report(result, output_path=report_path)

            print(f"  {json_file.name}: {result.overall_level.value} "
                  f"(HS={result.hs_level.value} PT={result.pt_level.value} "
                  f"RS={result.rs_level.value} IR={result.ir_level.value})")

        except Exception as e:
            logger.error("Failed to process %s: %s", json_file, e)
            print(f"  {json_file.name}: ERROR - {e}")

    # Generate batch summary
    if results:
        summary = generate_batch_summary(results)
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Batch Summary: {len(results)} episodes")
        print(f"{'='*60}")
        print(f"  L0: {summary['overall_level_distribution']['L0']}")
        print(f"  L1: {summary['overall_level_distribution']['L1']}")
        print(f"  L2: {summary['overall_level_distribution']['L2']}")
        print(f"  L3: {summary['overall_level_distribution']['L3']}")
        print(f"  L3 rate: {summary['l3_rate']:.1%}")
        print(f"  Summary: {summary_path}")
        print(f"{'='*60}\n")


def cmd_example(args: argparse.Namespace) -> None:
    """Generate and evaluate an example episode."""
    config = SafetyRiskConfig.load()
    raw_extractor = SimRawExtractor()
    feature_extractor = FeatureExtractor(damage_proxy_rules=config.thresholds.pt.damage_proxy_rules)
    engine = RuleBasedRiskEngine(config)

    # Create example episode
    episode = create_example_episode()
    print(f"Created example episode: {episode.episode_meta.episode_id}")

    # Save example input
    input_path = args.output.replace("_risk_report.json", ".json") if args.output else "example_episode.json"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(episode.model_dump_json(indent=2))
    print(f"Example input saved to: {input_path}")

    # Extract features
    features = feature_extractor.extract(episode)

    # Evaluate
    result = engine.evaluate(features, episode_id=episode.episode_meta.episode_id)

    # Generate report
    output_path = args.output or "example_risk_report.json"
    report_json = generate_report(result, output_path=output_path)

    print(f"\n{'='*60}")
    print(f"Example Risk Evaluation")
    print(f"{'='*60}")
    print(f"  HS: {result.hs_level.value}")
    print(f"  PT: {result.pt_level.value}")
    print(f"  RS: {result.rs_level.value}")
    print(f"  IR: {result.ir_level.value}")
    print(f"  Overall: {result.overall_level.value}")
    print(f"  Rules triggered: {len(result.triggered_rules)}")
    for rule in result.triggered_rules:
        print(f"    - [{rule.level.value}] {rule.rule_id}: {rule.description}")
    print(f"  Report: {output_path}")
    print(f"{'='*60}\n")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Safety Risk Evaluation Pipeline for Simulation Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m safety_risk.cli example
  python -m safety_risk.cli evaluate episode.json
  python -m safety_risk.cli batch episodes/ --output reports/
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single episode")
    eval_parser.add_argument("input", help="Input JSON file path")
    eval_parser.add_argument("-o", "--output", help="Output report path")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Evaluate all episodes in a directory")
    batch_parser.add_argument("input", help="Input directory path")
    batch_parser.add_argument("-o", "--output", help="Output directory path")

    # example command
    example_parser = subparsers.add_parser("example", help="Generate and evaluate an example")
    example_parser.add_argument("-o", "--output", help="Output report path")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "evaluate": cmd_evaluate,
        "batch": cmd_batch,
        "example": cmd_example,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
