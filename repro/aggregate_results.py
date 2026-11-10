#!/usr/bin/env python3
"""Aggregate eva.json results from all Harness-MU and baseline runs.

Builds a comparison table printed to stdout and saved as
results/comparison_table.json.

Usage:
    python repro/aggregate_results.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def find_eva_files(pattern: str) -> list[Path]:
    """Find eva.json files whose parent path contains pattern."""
    return sorted(
        p for p in RESULTS_DIR.rglob("eva.json")
        if pattern.lower() in str(p).lower()
    )


def load_eva(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_baseline_jsonl(path: Path) -> dict:
    """Parse a Muses-Bench baseline output JSONL and compute aggregate metrics."""
    if not path.exists():
        return {}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return {}

    def avg(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    return {
        "total_scenarios": len(rows),
        "avg_privacy_score": avg("privacy_score"),
        "avg_utility_score": avg("utility_score"),
        "avg_overall_score": avg("overall_score"),
    }


def fmt(val) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1%}"


def main():
    table = []

    # --- Harness-MU results (eva.json files) ---
    for eva_path in RESULTS_DIR.rglob("eva.json"):
        rel = str(eva_path.relative_to(RESULTS_DIR))
        data = load_eva(eva_path)

        # Infer condition label from path
        path_str = rel.lower()
        if "none" in path_str and "fake" not in path_str and "ablation" not in path_str:
            attack = "none"
            system = "Harness-MU (full)"
        elif "fake_authorized" in path_str and "ablation" not in path_str:
            attack = "fake_authorized"
            system = "Harness-MU (full)"
        elif "ablation" in path_str or "no_cc" in path_str:
            attack = "fake_authorized"
            system = "Harness-MU (ablation: no ComplianceChecker)"
        else:
            attack = "unknown"
            system = f"Harness-MU ({rel})"

        table.append({
            "system": system,
            "attack": attack,
            "scenarios": data.get("total_scenarios"),
            "privacy": data.get("avg_privacy_score"),
            "utility": data.get("avg_utility_score"),
            "overall": data.get("avg_overall_score"),
            "privacy_raw": data.get("avg_privacy_score_raw"),
            "boundary_delta_privacy": data.get("avg_boundary_delta_privacy"),
            "num_boundary_intercepts": data.get("avg_num_boundary_intercepts"),
            "source": rel,
        })

    # --- Baseline results (JSONL files) ---
    for jsonl_path in RESULTS_DIR.glob("baseline_*.jsonl"):
        name = jsonl_path.stem  # e.g. baseline_none_trial1
        parts = name.split("_")
        attack = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else "unknown"
        data = parse_baseline_jsonl(jsonl_path)
        if not data:
            continue
        table.append({
            "system": "Baseline (no harness)",
            "attack": attack,
            "scenarios": data.get("total_scenarios"),
            "privacy": data.get("avg_privacy_score"),
            "utility": data.get("avg_utility_score"),
            "overall": data.get("avg_overall_score"),
            "privacy_raw": None,
            "boundary_delta_privacy": None,
            "num_boundary_intercepts": None,
            "source": str(jsonl_path.relative_to(RESULTS_DIR)),
        })

    if not table:
        print("No results found in results/. Run run_experiments.py first.")
        return

    # Sort: system then attack
    table.sort(key=lambda r: (r["attack"], r["system"]))

    # Print table
    col_w = [42, 16, 9, 9, 9, 9, 16]
    headers = ["System", "Attack", "Privacy", "Utility", "Overall", "Priv(raw)", "CC delta(privacy)"]
    sep = "+".join("-" * (w + 2) for w in col_w)
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n+{sep}+")
    print(row_fmt.format(*headers))
    print(f"+{sep}+")
    for r in table:
        print(row_fmt.format(
            r["system"][:col_w[0]],
            r["attack"][:col_w[1]],
            fmt(r["privacy"]),
            fmt(r["utility"]),
            fmt(r["overall"]),
            fmt(r["privacy_raw"]),
            fmt(r["boundary_delta_privacy"]),
        ))
    print(f"+{sep}+")
    print(f"\nTotal result sets found: {len(table)}")

    # Save JSON
    out_path = RESULTS_DIR / "comparison_table.json"
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()
