#!/usr/bin/env python3
"""Run the full reproduction experiment suite for Harness-MU.

Executes four conditions on the access_control scenario:
  1. Harness-MU full (all modules on), none-attack
  2. Muses-Bench baseline (no harness), none-attack
  3. Harness-MU full, fake_authorized attack
  4. Muses-Bench baseline, fake_authorized attack
  5. Harness-MU ablation (ComplianceChecker off), fake_authorized attack

Results are written to results/ in the standard layout so that
aggregate_results.py can build the comparison table.

Usage:
    python repro/run_experiments.py --model deepseek-chat --trials 3
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DATA = REPO_ROOT / "harness_mu" / "data" / "access_control"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_TEMPLATE = REPO_ROOT / "harness_mu" / "config" / "my_experiment.yaml"
CONFIG_ABLATION = REPO_ROOT / "harness_mu" / "config" / "ablation_no_cc.yaml"

DATASETS = {
    "none": "template_colon_attack_none_2_to_10_each_2.jsonl",
    "fake_authorized": "template_colon_attack_fake_authorized_2_to_10_each_2.jsonl",
}


def run_cmd(cmd: list[str], cwd: Path = REPO_ROOT, label: str = "") -> int:
    """Run a command, streaming output to the terminal."""
    print(f"\n{'='*60}")
    print(f"[RUN] {label}")
    print(f"{'='*60}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[ERROR] {label} exited with code {result.returncode}")
    else:
        print(f"[OK] {label} completed")
    return result.returncode


def make_ablation_config():
    """Create a copy of my_experiment.yaml with compliance_checker disabled."""
    import yaml

    if not CONFIG_TEMPLATE.exists():
        print(f"[ERROR] Config template not found: {CONFIG_TEMPLATE}")
        sys.exit(1)

    with open(CONFIG_TEMPLATE) as f:
        cfg = yaml.safe_load(f)

    cfg["scenarios"]["access_control"]["enabled"] = True
    cfg["scenarios"]["access_control"]["modules"]["compliance_checker"] = False

    CONFIG_ABLATION.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_ABLATION, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Ablation config written to {CONFIG_ABLATION}")


def run_harness(dataset_key: str, config: Path, tag: str, trials: int):
    """Run Harness-MU on a dataset, optionally multiple trials."""
    data_file = DATASETS[dataset_key]
    data_path = HARNESS_DATA / data_file

    for trial in range(1, trials + 1):
        label = f"Harness-MU | {tag} | trial {trial}/{trials}"
        cmd = [
            sys.executable, "-m", "harness_mu",
            "--scenario", "access_control",
            "--config", str(config),
            "--data", str(data_path),
            "--verbose",
        ]
        run_cmd(cmd, label=label)


def run_baseline(dataset_key: str, model: str, tag: str, trials: int):
    """Run Muses-Bench baseline (no harness) on a dataset."""
    data_file = DATASETS[dataset_key]
    data_path = HARNESS_DATA / data_file
    baseline_dir = REPO_ROOT / "Multi-User-LLM-Agent"

    for trial in range(1, trials + 1):
        output_file = RESULTS_DIR / f"baseline_{dataset_key}_trial{trial}.jsonl"
        label = f"Baseline (no harness) | {tag} | trial {trial}/{trials}"
        cmd = [
            sys.executable, "run.py",
            "--scenario", "access_control",
            "--data", str(data_path),
            "--model", model,
            "--provider", "openai",
            "--output", str(output_file),
        ]
        run_cmd(cmd, cwd=baseline_dir, label=label)


def main():
    parser = argparse.ArgumentParser(
        description="Run full Harness-MU reproduction experiment suite"
    )
    parser.add_argument(
        "--model", default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials per condition for stochastic runs (default: 1)"
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip the ComplianceChecker ablation run"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write a run manifest so results are traceable
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "trials": args.trials,
        "datasets": DATASETS,
        "repo_root": str(REPO_ROOT),
    }
    manifest_path = RESULTS_DIR / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest written to {manifest_path}")

    # 1. Harness-MU full, none-attack
    run_harness("none", CONFIG_TEMPLATE, "none-attack, full harness", args.trials)

    # 2. Baseline, none-attack
    run_baseline("none", args.model, "none-attack", args.trials)

    # 3. Harness-MU full, fake_authorized attack
    run_harness(
        "fake_authorized", CONFIG_TEMPLATE,
        "fake_auth attack, full harness", args.trials
    )

    # 4. Baseline, fake_authorized attack
    run_baseline("fake_authorized", args.model, "fake_auth attack", args.trials)

    # 5. Ablation: ComplianceChecker off, fake_authorized attack
    if not args.skip_ablation:
        make_ablation_config()
        run_harness(
            "fake_authorized", CONFIG_ABLATION,
            "fake_auth attack, NO ComplianceChecker (ablation)", args.trials
        )

    print(f"\n{'='*60}")
    print("All experiments complete. Run aggregate_results.py to build the table.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
