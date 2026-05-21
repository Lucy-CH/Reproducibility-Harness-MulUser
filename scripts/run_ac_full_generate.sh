#!/bin/bash
# Phase 1: Generate transcripts for all 12 AC datasets using Muses-Bench
# Run from: Multi-User-LLM-Agent/
# Each dataset takes ~25 min. Total: ~5h.
# Supports resume: already-cached scenario_ids are skipped.

set -e
cd /workspace/fangewudi6/Harness_MulUser/Multi-User-LLM-Agent
source ../venv/bin/activate

DATA_DIR=data/scenarios/access_control/test_datasets/controlled_exp_large

echo "=== AC Transcript Generation (all 12 datasets) ==="
echo "Model: deepseek-v4-pro (defender & simulated user)"
echo "Max turns: 10"
echo "Start: $(date)"
echo ""

for f in "$DATA_DIR"/template_*.jsonl; do
  stem=$(basename "$f" .jsonl)
  echo "=== [$(date '+%H:%M:%S')] Generating: $stem ==="
  python run.py --scenario access_control \
    --data "$f" \
    --model deepseek-v4-pro --provider openai \
    --user-model deepseek-v4-pro \
    --max-turns 10 \
    --transcript-mode generate
  echo "=== [$(date '+%H:%M:%S')] Done: $stem ==="
  echo ""
done

echo "=== All transcripts generated. End: $(date) ==="
