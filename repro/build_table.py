#!/usr/bin/env python3
"""Build the comparison table from the per-trial runs written by run_all.sh.

Reads   results/runs/trial_<N>/<condition>/metrics.jsonl (metrics only, no transcripts)
Writes  results/comparison_table.json and results/comparison_table.md

Every number is computed from the per-scenario rows in metrics.jsonl:
  * each trial's score is the mean over its scenarios;
  * the table reports the mean of the trial scores, and the standard deviation
    across trials when there are at least two trials;
  * "scenario sd" is the standard deviation over all scenarios of all trials.

A condition appears only if a run for it exists. Incomplete runs are flagged.

Usage:
    python repro/build_table.py
"""

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
RUNS = RESULTS / "runs"
DATA_DIR = REPO_ROOT / "harness_mu" / "data" / "access_control"

DATASETS = {
    "none": "template_colon_attack_none_2_to_10_each_2.jsonl",
    "fake_authorized": "template_colon_attack_fake_authorized_2_to_10_each_2.jsonl",
}

# (condition folder, attack, system label)
CONDITIONS = [
    ("baseline_none", "none", "Baseline (no harness)"),
    ("full_none", "none", "Harness-MU (full)"),
    ("ablation_none", "none", "Harness-MU (no ComplianceChecker)"),
    ("baseline_fake", "fake_authorized", "Baseline (no harness)"),
    ("full_fake", "fake_authorized", "Harness-MU (full)"),
    ("ablation_fake", "fake_authorized", "Harness-MU (no ComplianceChecker)"),
]

# Paper Table 1, DeepSeek, colon template, max-turns = 10: (privacy, utility, overall)
PAPER = {
    "baseline_none": (0.815, 0.607, 0.456),
    "full_none": (1.000, 0.989, 0.994),
    "baseline_fake": (0.926, 0.506, 0.475),
    "full_fake": (1.000, 0.972, 0.981),
}

METRICS = ["privacy_score", "utility_score", "overall_score"]
HARNESS_ONLY = ["privacy_score_raw", "boundary_delta_privacy", "num_boundary_intercepts"]


def load_rows(path: Path):
    """Return (rows, n_error). Harness rows keep metrics under 'metrics'.

    Duplicate scenario rows (two processes appending to one file) keep the last.
    """
    by_id, errors = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if r.get("error"):
                errors += 1
                continue
            m = dict(r["metrics"]) if isinstance(r.get("metrics"), dict) else dict(r)
            m["_unscored"] = "per_turn_metrics" in r and not r["per_turn_metrics"]
            by_id[r.get("scenario_id")] = m
    return list(by_id.values()), errors


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def summarise(cond: str, attack: str, label: str):
    expected = sum(1 for l in open(DATA_DIR / DATASETS[attack]) if l.strip())
    trial_dirs = sorted(
        (p for p in RUNS.glob("trial_*") if (p / cond / "metrics.jsonl").exists()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not trial_dirs:
        return None

    out = {
        "condition": cond, "attack": attack, "system": label,
        "expected_scenarios": expected, "trials": [], "warnings": [],
    }
    pooled = {k: [] for k in METRICS + HARNESS_ONLY}
    for td in trial_dirs:
        rows, errors = load_rows(td / cond / "metrics.jsonl")
        if not rows:
            continue
        t = {"trial": td.name, "n": len(rows), "errors": errors,
             "unscored": sum(1 for r in rows if r["_unscored"])}
        for k in METRICS + HARNESS_ONLY:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                t[k] = mean(vals)
                pooled[k].extend(vals)
        if len(rows) != expected or errors:
            out["warnings"].append(
                f"{td.name}: {len(rows)}/{expected} scenarios, {errors} failed row(s)")
        out["trials"].append(t)

    if not out["trials"]:
        return None
    out["n_trials"] = len(out["trials"])
    for k in METRICS + HARNESS_ONLY:
        per_trial = [t[k] for t in out["trials"] if k in t]
        if not per_trial:
            continue
        out[k] = mean(per_trial)
        out[k + "_trial_sd"] = statistics.stdev(per_trial) if len(per_trial) >= 2 else None
        out[k + "_scenario_sd"] = statistics.stdev(pooled[k]) if len(pooled[k]) >= 2 else None
    out["unscored_per_trial"] = mean([t["unscored"] for t in out["trials"]])
    if cond in PAPER:
        out["paper"] = dict(zip(METRICS, PAPER[cond]))
    return out


def cell(row, key, signed=False):
    if key not in row:
        return "n/a"
    s = f"{row[key]:+.3f}" if signed else f"{row[key]:.3f}"
    sd = row.get(key + "_trial_sd")
    return s + (f" ± {sd:.3f}" if sd is not None else "")


def main():
    if not RUNS.exists():
        print(f"No runs found under {RUNS}. Run: bash repro/run_all.sh")
        return
    table = [s for s in (summarise(*c) for c in CONDITIONS) if s]
    if not table:
        print(f"No runs found under {RUNS}. Run: bash repro/run_all.sh")
        return

    lines = [
        "# Reproduction results",
        "",
        "Values are the mean over trials of each trial's scenario mean; ± is the "
        "standard deviation across trials (shown when there are at least two trials). "
        "Model and settings: see harness_mu/config/my_experiment.yaml, max-turns = 10.",
        "",
        "| Attack | System | Trials | Privacy | Utility | Overall | Raw privacy | CC Δ privacy | CC intercepts / scenario | Paper P / U / O |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        paper = "n/a"
        if "paper" in r:
            p = r["paper"]
            paper = f"{p['privacy_score']:.3f} / {p['utility_score']:.3f} / {p['overall_score']:.3f}"
        lines.append(
            f"| {r['attack']} | {r['system']} | {r['n_trials']} | "
            f"{cell(r, 'privacy_score')} | {cell(r, 'utility_score')} | {cell(r, 'overall_score')} | "
            f"{cell(r, 'privacy_score_raw')} | {cell(r, 'boundary_delta_privacy', signed=True)} | "
            f"{cell(r, 'num_boundary_intercepts')} | {paper} |"
        )

    lines += ["", "## Per-trial scores", "",
              "| Attack | System | Trial | Scenarios | Privacy | Utility | Overall | Raw privacy | Never scored |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        for t in r["trials"]:
            raw = f"{t['privacy_score_raw']:.3f}" if "privacy_score_raw" in t else "n/a"
            lines.append(
                f"| {r['attack']} | {r['system']} | {t['trial']} | {t['n']}/{r['expected_scenarios']} | "
                f"{t['privacy_score']:.3f} | {t['utility_score']:.3f} | {t['overall_score']:.3f} | "
                f"{raw} | {t['unscored']} |")
    lines += ["",
              "\"Never scored\" counts baseline scenarios in which no agent reply was "
              "evaluated (zero entries in per_turn_metrics). The baseline records these "
              "with its default scores: privacy 1.0, utility 0.0.", ""]

    warnings = [f"{r['condition']}: {w}" for r in table for w in r["warnings"]]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    text = "\n".join(lines)
    print(text)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "comparison_table.json", "w") as f:
        json.dump(table, f, indent=2)
    with open(RESULTS / "comparison_table.md", "w") as f:
        f.write(text)
    print(f"Saved: {RESULTS / 'comparison_table.json'}")
    print(f"Saved: {RESULTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
