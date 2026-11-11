#!/bin/bash
# Run all 5 reproduction conditions for Harness-MU.
# Run this from the Harness-MulUser repo root on your Mac.
#
# Usage:
#   cd "Harness-MulUser"
#   bash repro/run_all.sh
#
# Prerequisites (already done if you followed the README):
#   - .venv activated
#   - .env filled with DeepSeek credentials
#   - IHEval cloned into Multi-User-LLM-Agent/IHEval/
#   - my_experiment.yaml exists with deepseek-chat

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

CONFIG="harness_mu/config/my_experiment.yaml"
ABLATION_CONFIG="harness_mu/config/ablation_no_cc.yaml"
# Data paths are relative to the harness_mu/ package root (pipeline.py prepends it),
# so do NOT include harness_mu/ prefix here.
DATA_AC="data/access_control"
RESULTS="results"

# ---- Enable access_control in my_experiment.yaml if not already ----
python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['scenarios']['access_control']['enabled'] = True
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('[OK] access_control enabled in my_experiment.yaml')
"

# ---- Create ablation config (ComplianceChecker off) ----
python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['scenarios']['access_control']['modules']['compliance_checker'] = False
with open('$ABLATION_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('[OK] ablation config written: $ABLATION_CONFIG')
"

echo ""
echo "============================================"
echo "  CONDITION 1/5: Baseline (no harness)"
echo "  Dataset: none-attack"
echo "============================================"
cd Multi-User-LLM-Agent
python run.py \
  --scenario access_control \
  --data ../harness_mu/$DATA_AC/template_colon_attack_none_2_to_10_each_2.jsonl \
  --model deepseek-v4-pro \
  --provider openai \
  --output ../$RESULTS/baseline_none.jsonl
cd ..
echo "[DONE] Condition 1"

echo ""
echo "============================================"
echo "  CONDITION 2/5: Harness-MU (full)"
echo "  Dataset: none-attack"
echo "============================================"
python -m harness_mu \
  --scenario access_control \
  --config $CONFIG \
  --data $DATA_AC/template_colon_attack_none_2_to_10_each_2.jsonl
# Copy aside so ablation doesn't overwrite
cp -r $RESULTS/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-v4-pro \
      $RESULTS/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-v4-pro-full
echo "[DONE] Condition 2"

echo ""
echo "============================================"
echo "  CONDITION 3/5: Baseline (no harness)"
echo "  Dataset: fake_authorized attack"
echo "============================================"
cd Multi-User-LLM-Agent
python run.py \
  --scenario access_control \
  --data ../harness_mu/$DATA_AC/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl \
  --model deepseek-v4-pro \
  --provider openai \
  --output ../$RESULTS/baseline_fake_auth.jsonl
cd ..
echo "[DONE] Condition 3"

echo ""
echo "============================================"
echo "  CONDITION 4/5: Harness-MU (full)"
echo "  Dataset: fake_authorized attack"
echo "============================================"
python -m harness_mu \
  --scenario access_control \
  --config $CONFIG \
  --data $DATA_AC/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl
cp -r $RESULTS/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro \
      $RESULTS/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro-full
echo "[DONE] Condition 4"

echo ""
echo "============================================"
echo "  CONDITION 5/5: Harness-MU (ablation: no CC)"
echo "  Dataset: fake_authorized attack"
echo "============================================"
python -m harness_mu \
  --scenario access_control \
  --config $ABLATION_CONFIG \
  --data $DATA_AC/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl
echo "[DONE] Condition 5"

echo ""
echo "============================================"
echo "  ALL CONDITIONS COMPLETE"
echo "============================================"
echo ""
echo "Result files:"
echo "  Baseline none:        $RESULTS/baseline_none.jsonl"
echo "  Baseline fake_auth:   $RESULTS/baseline_fake_auth.jsonl"
echo "  Harness none:         $RESULTS/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-v4-pro-full/eva.json"
echo "  Harness fake_auth:    $RESULTS/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro-full/eva.json"
echo "  Ablation fake_auth:   $RESULTS/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro/eva.json"
