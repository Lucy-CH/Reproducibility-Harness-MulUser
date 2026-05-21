#!/bin/bash
# Phase 2: Harness-MU strict replay for all 12 AC datasets
# Prerequisite: Phase 1 transcripts must exist.
# Run from: Harness_MulUser/
# Each dataset takes ~6 min. Total: ~72 min.

set -e
cd /workspace/fangewudi6/Harness_MulUser
source venv/bin/activate

TEMPLATE=harness_mu/config/ac_replay_bmini.yaml

echo "=== AC Strict Replay (all 12 datasets) ==="
echo "Model: deepseek-v4-pro (all 4 modules)"
echo "Transcript mode: replay (bit-identical attack text)"
echo "Start: $(date)"
echo ""

for f in harness_mu/data/access_control/template_*.jsonl; do
  stem=$(basename "$f" .jsonl)
  echo "=== [$(date '+%H:%M:%S')] Replaying: $stem ==="
  tmp=$(mktemp --suffix=.yaml)
  sed "s|data/access_control/template_colon_attack_none_2_to_10_each_2.jsonl|data/access_control/${stem}.jsonl|" \
    "$TEMPLATE" > "$tmp"
  python -m harness_mu --scenario access_control --config "$tmp"
  rm "$tmp"
  echo "=== [$(date '+%H:%M:%S')] Done: $stem ==="
  echo ""
done

echo "=== All replays done. End: $(date) ==="

# Print summary comparison
echo ""
echo "=== Comparison Summary ==="
python3 << 'PY'
import json, os
MB = "Multi-User-LLM-Agent/results/access_control"
HMU = "results/access_control"
model = "deepseek-v4-pro"
print(f"{'stem':<55} {'MB_Priv':>8} {'HMU_Priv':>8} {'MB_Util':>8} {'HMU_Util':>8} {'MB_O':>8} {'HMU_O':>8} {'Δ_O':>8}")
for d in sorted(os.listdir(MB)):
    mb = os.path.join(MB, d, model, "eva.json")
    hmu = os.path.join(HMU, d, model, "eva.json")
    if not (os.path.exists(mb) and os.path.exists(hmu)):
        continue
    mb_j = json.load(open(mb))
    hmu_j = json.load(open(hmu))
    p_mb = mb_j.get("avg_privacy_score", float("nan"))
    p_hmu = hmu_j.get("avg_privacy_score", float("nan"))
    u_mb = mb_j.get("avg_utility_score", float("nan"))
    u_hmu = hmu_j.get("avg_utility_score", float("nan"))
    o_mb = mb_j.get("avg_overall_score", float("nan"))
    o_hmu = hmu_j.get("avg_overall_score", float("nan"))
    print(f"{d:<55} {p_mb:8.3f} {p_hmu:8.3f} {u_mb:8.3f} {u_hmu:8.3f} {o_mb:8.3f} {o_hmu:8.3f} {o_hmu-o_mb:+8.3f}")
PY
