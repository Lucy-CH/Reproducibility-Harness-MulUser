#!/usr/bin/env python3
"""Build the comparison table from all experiment results.

Reads:
  - Baseline JSONL files: results/baseline_*.jsonl (one line per scenario)
  - Harness-MU eva.json files: results/access_control/<dataset>/<model>*/eva.json
  - Ablation eva.json: results/access_control/<dataset>/deepseek-chat/eva.json
    (after full result is copied to deepseek-chat-full/)

Prints a comparison table to stdout and saves it as
results/comparison_table.json and results/comparison_table.md.

Usage:
    python repro/build_table.py
"""

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def load_baseline_jsonl(path: Path) -> dict | None:
    """Parse a Muses-Bench baseline JSONL; return aggregate metrics."""
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None

    def avg(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    def std(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    return {
        "n": len(rows),
        "privacy": avg("privacy_score"),
        "privacy_std": std("privacy_score"),
        "utility": avg("utility_score"),
        "utility_std": std("utility_score"),
        "overall": avg("overall_score"),
        "overall_std": std("overall_score"),
        "source": str(path.relative_to(RESULTS)),
    }


def load_eva_json(path: Path) -> dict | None:
    """Load a Harness-MU eva.json file."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "n": data.get("total_scenarios"),
        "privacy": data.get("avg_privacy_score"),
        "utility": data.get("avg_utility_score"),
        "overall": data.get("avg_overall_score"),
        "privacy_raw": data.get("avg_privacy_score_raw"),
        "boundary_delta_privacy": data.get("avg_boundary_delta_privacy"),
        "boundary_delta_overall": data.get("avg_boundary_delta_overall"),
        "num_intercepts": data.get("avg_num_boundary_intercepts"),
        "source": str(path.relative_to(RESULTS)),
    }


def fmt(val, std=None) -> str:
    if val is None:
        return "n/a"
    s = f"{val:.1%}"
    if std is not None and std > 0:
        s += f" (±{std:.1%})"
    return s


def main():
    table = []

    # --- Baseline JSONL files ---
    for name in ["baseline_none.jsonl", "baseline_fake_auth.jsonl"]:
        path = RESULTS / name
        data = load_baseline_jsonl(path)
        if data:
            attack = "none" if "none" in name else "fake_authorized"
            table.append({
                "system": "Baseline (no harness)",
                "attack": attack,
                **data,
            })

    # --- Harness-MU full (copied to *-full/) ---
    for dataset_dir in RESULTS.glob("access_control/*/"):
        dataset_name = dataset_dir.name
        attack = "none" if "none" in dataset_name else (
            "fake_authorized" if "fake_authorized" in dataset_name else dataset_name
        )
        for model_dir in dataset_dir.glob("deepseek-chat-full"):
            eva = load_eva_json(model_dir / "eva.json")
            if eva:
                table.append({
                    "system": "Harness-MU (full)",
                    "attack": attack,
                    **eva,
                })

    # --- Ablation (deepseek-chat without -full suffix, after copy) ---
    for dataset_dir in RESULTS.glob("access_control/*/"):
        dataset_name = dataset_dir.name
        attack = "none" if "none" in dataset_name else (
            "fake_authorized" if "fake_authorized" in dataset_name else dataset_name
        )
        model_dir = dataset_dir / "deepseek-chat"
        eva_path = model_dir / "eva.json"
        # Only count as ablation if a -full/ copy exists (meaning the original
        # was the full run and this is the ablation overwrite). If no -full/
        # copy exists, this is the only run, not an ablation.
        full_dir = dataset_dir / "deepseek-chat-full"
        if full_dir.exists() and eva_path.exists():
            eva = load_eva_json(eva_path)
            if eva:
                table.append({
                    "system": "Harness-MU (ablation: no CC)",
                    "attack": attack,
                    **eva,
                })

    if not table:
        print("No results found. Run repro/run_all.sh first.")
        return

    # Sort
    order = {"none": 0, "fake_authorized": 1}
    table.sort(key=lambda r: (
        order.get(r.get("attack", ""), 2),
        0 if "Baseline" in r["system"] else 1 if "full" in r["system"] else 2,
    ))

    # --- Print table ---
    print()
    header = f"{'System':<32} {'Attack':<18} {'N':>3} {'Privacy':>14} {'Utility':>14} {'Overall':>14} {'CC Δ privacy':>14}"
    print(header)
    print("-" * len(header))
    for r in table:
        sys_name = r["system"]
        atk = r["attack"]
        n = r.get("n", "?")
        priv = fmt(r.get("privacy"), r.get("privacy_std"))
        util = fmt(r.get("utility"), r.get("utility_std"))
        overall = fmt(r.get("overall"), r.get("overall_std"))
        delta = r.get("boundary_delta_privacy")
        delta_s = f"{delta:+.1%}" if delta is not None else "n/a"
        print(f"{sys_name:<32} {atk:<18} {str(n):>3} {priv:>14} {util:>14} {overall:>14} {delta_s:>14}")
    print()

    # --- Paper comparison ---
    print("Paper reported (GPT, for reference):")
    print(f"  Privacy (all attacks): 100% (baseline 92.6-99.5%)")
    print(f"  Overall improvement:    +0.240 to +0.427")
    print()

    # --- Save JSON ---
    out_json = RESULTS / "comparison_table.json"
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved: {out_json}")

    # --- Save Markdown ---
    out_md = RESULTS / "comparison_table.md"
    with open(out_md, "w") as f:
        f.write("# Reproduction Results\n\n")
        f.write("| System | Attack | N | Privacy | Utility | Overall | CC Δ Privacy |\n")
        f.write("|--------|--------|---|---------|---------|---------|-------------|\n")
        for r in table:
            n = r.get("n", "?")
            priv = fmt(r.get("privacy"), r.get("privacy_std"))
            util = fmt(r.get("utility"), r.get("utility_std"))
            overall = fmt(r.get("overall"), r.get("overall_std"))
            delta = r.get("boundary_delta_privacy")
            delta_s = f"{delta:+.1%}" if delta is not None else "n/a"
            f.write(f"| {r['system']} | {r['attack']} | {n} | {priv} | {util} | {overall} | {delta_s} |\n")
        f.write("\nPaper reported (GPT): Privacy 100% across all attacks; Overall +0.240 to +0.427.\n")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
