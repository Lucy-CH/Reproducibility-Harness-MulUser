"""CLI entry point for Harness-MU."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root or harness_mu directory
_ENV_PATHS = [
    Path(__file__).resolve().parents[1] / ".env",          # project root
    Path(__file__).resolve().parent / ".env",               # harness_mu dir
]
for _ENV_PATH in _ENV_PATHS:
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=str(_ENV_PATH), override=True)
        break

from harness_mu.config.loader import load_config
from harness_mu.core.pipeline import PipelineOrchestrator
from harness_mu.llm.factory import LLMFactory
from harness_mu.metrics.collector import MetricsCollector


def main():
    parser = argparse.ArgumentParser(
        description="Harness-MU: Multi-User LLM Agent Orchestration Framework",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["instruction_following", "access_control", "meeting_scheduling"],
        help="Benchmark scenario to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (merged with defaults).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override path to JSONL data file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output file path.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (first 3 scenarios, single worker).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Override the per-scenario transcript max_turns "
            "(applies to access_control transcript.mode=live|replay)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configuration
    config = load_config(args.config)
    if args.debug:
        config.execution.debug = True
    if args.max_turns is not None:
        scenario_cfg = config.scenarios.get(args.scenario)
        if scenario_cfg is not None and scenario_cfg.transcript is not None:
            scenario_cfg.transcript.max_turns = int(args.max_turns)

    # Initialize infrastructure
    metrics_collector = MetricsCollector()
    llm_factory = LLMFactory(config, metrics_collector)

    # Build pipeline
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_factory=llm_factory,
        metrics_collector=metrics_collector,
    )

    # Run batch
    try:
        orchestrator.run_batch(
            scenario_name=args.scenario,
            data_file=args.data,
            output_file=args.output,
            debug=args.debug,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
