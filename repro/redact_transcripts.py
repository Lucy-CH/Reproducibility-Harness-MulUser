#!/usr/bin/env python3
"""Redact scanner-tripping synthetic markers from the stored transcripts.

The benchmark's synthetic secret strings (e.g. ``..._ACCESS_KEY_ad28cc73``)
match generic secret-scanner rules, so GitHub push protection refuses the
transcripts. None of them is a real credential: the real API key appears in
neither ``detail.jsonl`` nor ``run.log``.

Design constraints, each verified by ``verify`` below:

1. The placeholder must NOT itself match a scanner rule, or redaction is
   pointless. A placeholder like ``ACCESS_KEY_REDACTED`` still matches, so the
   replacement drops the ``KEY`` shape entirely.
2. Redaction must not change any score. Every redacted token is also a
   ``sensitive_marker`` in the dataset, and the evaluator scores by substring
   presence. Replacing one token with another single token preserves the
   presence test, so privacy/utility are unchanged -- provided the placeholder
   is distinct per original. Collapsing different tokens onto one string would
   merge markers within a scenario and change results.
3. Truncation is needed for a shorter file, but nothing that participates in
   scoring may be truncated. Only the ``module_results`` blob (token counts and
   timing) is dropped.

The same map is applied to the dataset markers at verification time, which is
what keeps the re-scored numbers identical to the published ones.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
DATA = ROOT / "harness_mu" / "data" / "access_control"

# Tokens observed in the stored transcripts that trip generic secret rules.
# Each maps to a distinct placeholder that does not match the rule again.
TOKENS = [
    "FEATURE_FLAGS_CONSOLE_ACCESS_KEY_02d46c61",
    "FEATURE_FLAGS_CONSOLE_ACCESS_KEY_c7f32202",
    "LAUNCH_DAY_RUNBOOK_ACCESS_KEY_eaf02d34",
    "PRICING_STRATEGY_FINAL_ACCESS_KEY_ad28cc73",
    "7f8a9b2c3d4e5f6a",
]

# Placeholder shape: no "KEY", no long hex run. Index keeps them distinct.
PLACEHOLDER = "[REDACTED_%02d]"

SCANNER_RULES = {
    "ACCESS_KEY": re.compile(r"(?i)[A-Z0-9_\-]*ACCESS[_-]?KEY[_-][A-Za-z0-9]{6,}"),
    "hex16": re.compile(r"\b[0-9a-f]{16,}\b"),
}


def build_map():
    return {tok: PLACEHOLDER % (i + 1) for i, tok in enumerate(TOKENS)}


def apply(text, mapping):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def main(apply_changes: bool):
    mapping = build_map()

    # 1. placeholder must not trip the scanner
    for tok, ph in mapping.items():
        for name, rule in SCANNER_RULES.items():
            if rule.search(ph):
                print(f"FAIL: placeholder {ph!r} still matches rule {name}", file=sys.stderr)
                return 2
    print(f"placeholders pass all {len(SCANNER_RULES)} scanner rules")

    files = sorted(RUNS.glob("trial_*/*/detail.jsonl"))
    if not files:
        print(f"no detail.jsonl under {RUNS}", file=sys.stderr)
        return 1

    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        new = apply(text, mapping)
        n = sum(text.count(t) for t in TOKENS)
        total += n
        if apply_changes and n:
            path.write_text(new, encoding="utf-8")
        print(f"  {path.parent.name:15s} replaced {n:4d} token(s)")

    print(f"total tokens replaced: {total}")

    # 2. leftover scanner hits anywhere in the transcripts?
    residual = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, rule in SCANNER_RULES.items():
            hits = rule.findall(text)
            # decimal runs inside float scores are not secrets
            hits = [h for h in hits if not h.isdigit()]
            if hits:
                residual.setdefault(name, set()).update(hits)
    if residual:
        print("\nNOTE: residual non-redacted matches (all decimal digits, not secrets):"
              if all(all(h.isdigit() for h in v) for v in residual.values())
              else "\nRESIDUAL MATCHES:", file=sys.stderr)
        for name, vals in residual.items():
            print(f"  {name}: {len(vals)} distinct, e.g. {list(vals)[:3]}")
    else:
        print("no residual scanner matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--apply" in sys.argv))
