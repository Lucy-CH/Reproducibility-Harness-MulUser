"""Backfill Boundary-Layer contribution stats from legacy IF detail.jsonl files.

Old IF runs did not snapshot the Worker's pre-revision text, so an exact raw
accuracy cannot always be recomputed post-hoc. Two signals are available
inside every existing detail.jsonl and are enough for paper-grade aggregates:

* ``module_results[compliance_checker][k].data.num_failures`` — total checker
  failures across **all** users after the k-th compliance pass (k=0 initial,
  k=-1 final).  ``len(compliance_checker_calls) - 1`` therefore equals the
  number of revision rounds actually triggered.
* ``module_results[compliance_checker][k].data.per_user_pass`` — per-user
  all-or-nothing pass flag, before and after revision.

For aligned scenarios all users are eval-targets, so the first-pass failure
count is exact and ``raw_accuracy = (num_instructions - num_failures_initial)
/ num_instructions`` is exact too. For conflict scenarios only the
max-authority users are eval targets; the script falls back to reporting
per-eval-target pass-flag deltas instead of synthetic accuracy bounds.

Usage::

    python scripts/backfill_if_compliance_contribution.py \\
        [--results-root results/instruction_following] \\
        [--data-root harness_mu/data/instruction_following] \\
        [--csv out.csv]
"""

import argparse
import csv
import json
from glob import glob
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def _load_dataset_index(path: Path) -> dict:
    idx = {}
    if not path.exists():
        return idx
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            idx[d.get("id", "")] = d
    return idx


def _stats_for_scenario(row: dict, dataset_row: dict | None) -> dict:
    compliance_calls = [
        m for m in row.get("module_results", []) or []
        if m.get("module_name") == "compliance_checker"
    ]
    if not compliance_calls:
        return {}
    init_data = compliance_calls[0].get("data", {}) or {}
    final_data = compliance_calls[-1].get("data", {}) or {}

    revision_count = max(0, len(compliance_calls) - 1)
    failures_initial = int(init_data.get("num_failures", 0) or 0)
    failures_final = int(final_data.get("num_failures", 0) or 0)
    failures_fixed = max(0, failures_initial - failures_final)
    metrics = row.get("metrics", {}) or {}
    num_inst_eval = int(metrics.get("num_instructions", 0) or 0)
    accuracy_final = float(metrics.get("accuracy", 0.0) or 0.0)

    mode = "aligned"
    if dataset_row is not None:
        mode = (dataset_row.get("meta") or {}).get("mode", "aligned")

    out = {
        "scenario_id": row.get("scenario_id", ""),
        "mode": mode,
        "revision_count": revision_count,
        "num_failures_initial": failures_initial,
        "num_failures_final": failures_final,
        "failures_fixed_by_revision": failures_fixed,
        "num_instructions_eval": num_inst_eval,
        "accuracy_final": accuracy_final,
    }

    if mode == "aligned" and num_inst_eval > 0:
        out["accuracy_raw_exact"] = (num_inst_eval - failures_initial) / num_inst_eval
        out["accuracy_uplift_by_revision"] = accuracy_final - out["accuracy_raw_exact"]
    elif mode == "conflict" and dataset_row is not None:
        users = dataset_row.get("users", []) or []
        max_auth = max((float(u.get("authority", 0)) for u in users), default=0.0)
        eval_uids = {u.get("id") for u in users if float(u.get("authority", 0)) == max_auth}
        init_pass = init_data.get("per_user_pass", {}) or {}
        final_pass = final_data.get("per_user_pass", {}) or {}
        out["num_eval_target_users"] = len(eval_uids)
        out["eval_target_users_passing_initial"] = sum(
            1 for uid in eval_uids if init_pass.get(uid)
        )
        out["eval_target_users_passing_final"] = sum(
            1 for uid in eval_uids if final_pass.get(uid)
        )
        out["eval_target_users_fixed_by_revision"] = (
            out["eval_target_users_passing_final"]
            - out["eval_target_users_passing_initial"]
        )
    return out


def _print_table(label: str, rows: list[dict]) -> None:
    n = len(rows)
    n_rev = sum(1 for r in rows if r.get("revision_count", 0) > 0)
    print(f"\n=== {label}  ({n} scenarios, {n_rev} triggered revision = {100*n_rev/max(1,n):.1f}%) ===")
    print(f"  avg revision_count        = {mean([r['revision_count'] for r in rows]):.3f}")
    print(f"  avg num_failures_initial  = {mean([r['num_failures_initial'] for r in rows]):.3f}")
    print(f"  avg num_failures_final    = {mean([r['num_failures_final'] for r in rows]):.3f}")
    print(f"  avg failures_fixed_by_rev = {mean([r['failures_fixed_by_revision'] for r in rows]):.3f}")
    print(f"  total failures_fixed_by_rev = {sum(r['failures_fixed_by_revision'] for r in rows)}")
    aligned = [r for r in rows if r.get("mode") == "aligned" and "accuracy_raw_exact" in r]
    if aligned:
        print(f"  [aligned only] avg accuracy_final   = {mean([r['accuracy_final'] for r in aligned]):.4f}")
        print(f"  [aligned only] avg accuracy_raw     = {mean([r['accuracy_raw_exact'] for r in aligned]):.4f}")
        print(f"  [aligned only] avg uplift_by_rev    = {mean([r['accuracy_uplift_by_revision'] for r in aligned]):.4f}")
    conflict = [r for r in rows if r.get("mode") == "conflict" and "num_eval_target_users" in r]
    if conflict:
        delta = mean([r["eval_target_users_fixed_by_revision"] for r in conflict])
        print(f"  [conflict only] avg eval-target users fixed by rev = {delta:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=str(ROOT / "results/instruction_following"))
    ap.add_argument("--data-root", default=str(ROOT / "harness_mu/data/instruction_following"))
    ap.add_argument("--csv", default=None, help="Optional path to write per-scenario rows.")
    args = ap.parse_args()

    dataset_cache: dict[str, dict] = {}
    grand_rows: list[dict] = []

    for detail in sorted(glob(f"{args.results_root}/**/detail.jsonl", recursive=True)):
        # results/instruction_following/<dataset>/<model>/detail.jsonl
        dataset = Path(detail).parents[1].name
        if dataset not in dataset_cache:
            dataset_cache[dataset] = _load_dataset_index(
                Path(args.data_root) / f"{dataset}.jsonl"
            )
        idx = dataset_cache[dataset]
        rows = []
        with open(detail) as f:
            for line in f:
                d = json.loads(line)
                ds_row = idx.get(d.get("scenario_id", ""))
                s = _stats_for_scenario(d, ds_row)
                if s:
                    s["_source"] = str(Path(detail).relative_to(ROOT))
                    rows.append(s)
        if rows:
            _print_table(str(Path(detail).relative_to(ROOT)), rows)
            grand_rows.extend(rows)

    if grand_rows:
        _print_table("GRAND TOTAL", grand_rows)

    if args.csv and grand_rows:
        keys = sorted({k for r in grand_rows for k in r.keys()})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(grand_rows)
        print(f"\nPer-scenario CSV: {args.csv}")


if __name__ == "__main__":
    main()
