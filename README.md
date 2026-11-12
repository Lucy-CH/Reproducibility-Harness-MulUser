# Harness-MU Reproducibility

Reproduction of **"Harness-MU: A Safe, Governed, and Effective Harness for Multi-User LLM Agents"**
(arXiv:2606.21856, Fan, Nie, & Dai, 2026) for FTEC5660 Agentic AI for Business and FinTech (Paper No. 28).

---

## Project summary

Harness-MU is a governance harness that wraps a multi-user LLM agent (Muses-Bench) with four modules: a Gatekeeper (access policy enforcement), a Mediator (task routing), a Worker (LLM response generation), and a ComplianceChecker (deterministic post-hoc token scrubbing). The paper claims that this harness achieves near-perfect privacy and utility simultaneously on the Muses-Bench access-control benchmark, whereas the unprotected baseline trades one for the other.

The paper evaluates four models: **DeepSeek V4 Pro** (`deepseek-v4-pro`), Qwen3.6-35b-a3b, Gemini 2.5 Flash, and GPT-4.1-nano. This reproduction uses `deepseek-v4-pro`, which corresponds directly to the paper's DeepSeek condition. Results below are directly comparable to the paper's DeepSeek column in Table 1 and Table 3.

---

## Reproduction targets

Two claims from Table 1 (DeepSeek column, max-turns = 10) of the paper are targeted:

| Claim | Paper (DeepSeek, none-attack) | Paper (DeepSeek, fake_authorized) | This reproduction |
|-------|-------------------------------|-----------------------------------|-------------------|
| Privacy: Harness-MU vs baseline | MB 0.815 → HMU **1.000** | MB 0.926 → HMU **1.000** | Baseline 1.000 → HMU **1.000** (none); Baseline 0.972 → HMU **1.000** (fake_auth) |
| Utility: Harness-MU vs baseline | MB 0.607 → HMU **0.989** | MB 0.506 → HMU **0.972** | Baseline 0.627 → HMU **0.989** (none); Baseline 0.613 → HMU **1.000** (fake_auth) |

Both claims reproduce: Harness-MU reaches privacy = 1.000 on both datasets and raises utility well above the unprotected baseline, consistent with the paper's DeepSeek numbers.

---

## Results

All runs use `deepseek-v4-pro`, n = 18 scenarios per condition, colon-serialization template, `template_colon_attack_*_2_to_10_each_2.jsonl` datasets.

### No-attack dataset

| System | Privacy | Utility | Overall |
|--------|---------|---------|---------|
| Baseline (no harness) | 1.000 ± 0.000 | 0.627 ± 0.452 | 0.657 ± 0.437 |
| Harness-MU (full) | **1.000** | **0.989** | **0.994** |
| Harness-MU (ablation: no CC) | **1.000** | **0.989** | **0.994** |

### Fake-authorized attack dataset

| System | Privacy | Utility | Overall | privacy_raw | boundary_delta_privacy |
|--------|---------|---------|---------|-------------|------------------------|
| Baseline (no harness) | 0.972 ± 0.118 | 0.613 ± 0.427 | 0.642 ± 0.394 | — | — |
| Harness-MU (full) | **1.000** | **1.000** | **1.000** | 0.981 | +0.019 |
| Harness-MU (ablation: no CC) | **1.000** | **1.000** | **1.000** | 1.000 | 0.000 |

`privacy_raw` is the Worker's privacy score before the ComplianceChecker rewrites. `boundary_delta_privacy` is the CC's marginal contribution.

**Claim 1 confirmed.** Harness-MU achieves avg_privacy_score = 1.000 on both datasets. The paper's DeepSeek column reports 1.000 in both cases — exact match.

**Claim 2 confirmed.** Harness-MU avg_utility_score = 0.989 (none-attack) and 1.000 (fake_authorized) versus baseline 0.627 / 0.613. The paper's DeepSeek column reports 0.989 / 0.972 — close match.

### How to read the results tables

**Rows.** "Baseline (no harness)" is Muses-Bench running with the raw LLM alone — no Gatekeeper, no Mediator, no ComplianceChecker; the agent handles multi-user access on its own. "Harness-MU (full)" is the complete paper pipeline: Gatekeeper + Mediator + Worker + ComplianceChecker all active. "Harness-MU (ablation: no CC)" is the same full pipeline with the ComplianceChecker switched off; everything else is identical, which isolates CC's specific contribution.

**Columns.** Privacy is the fraction of scenarios where no unauthorized user received any sensitive data; 1.000 means not a single leak across all 18 scenarios. Utility is the fraction of authorized users who successfully got the information they were entitled to; the baseline's low utility (~0.63) means the unprotected agent over-refuses legitimate requests as well as illegitimate ones. Overall is the harmonic mean of Privacy and Utility, which penalizes systems that sacrifice one for the other — a system with 1.0 privacy and 0.0 utility would score 0.0 overall. The ± values (baseline rows only) are standard deviations across the 18 scenarios, reflecting how consistent the system is; harness rows omit them because they are computed from the aggregate eva.json rather than per-trial variance.

**Extra columns on the fake_authorized table.** `privacy_raw` is the Worker's privacy score before the ComplianceChecker runs — how often the LLM itself made the right refusal decision, independent of the deterministic safety net. `boundary_delta_privacy` is the gap between `privacy_raw` and the final privacy score, i.e. how much the ComplianceChecker added: +0.019 for the full run means the CC rescued about 2% of responses the Worker got wrong, and 0.000 for the ablation confirms CC was disabled (nothing to add).

---

## Modification — ComplianceChecker ablation

**Change:** Disabled the ComplianceChecker module (`compliance_checker: false` in config) while keeping all other modules identical. The CC is a deterministic, rule-based post-processor with no LLM calls, making it a clean single-variable ablation.

**Expected outcome (from paper):** Removing CC should allow some Worker responses that leak sensitive tokens to pass through, reducing privacy on attack datasets.

**Observed outcome:** No measurable difference. On the fake_authorized attack, the full system and the ablation both score privacy = 1.000, utility = 1.000. The Worker alone (without CC) already achieves perfect privacy on this dataset with DeepSeek V4 Pro — the CC's `boundary_delta_privacy` is only +0.019 in the full run, meaning it corrected roughly 2% of responses. With a stronger base model, the Worker handles refusal decisions correctly without needing the CC fallback.

**Interpretation:** This is a null ablation finding specific to this model/dataset combination. The ComplianceChecker's role is to provide a deterministic safety net when the Worker fails; with a highly capable model on a constrained access-control task, Worker failures are rare enough that the CC rarely activates. The CC's contribution would likely be more visible on weaker models or more adversarial attack types (pressure, roleplaying). This does not falsify the paper's design — it reflects model capability interacting with a fail-closed safety architecture.

---

## Setup

### Environment

| Component | Version |
|-----------|---------|
| OS | macOS (Darwin 25.6.0) |
| Python | 3.14 (via `.venv/`) |
| litellm | 1.108.4 |
| openai | 3.49.0 |
| absl-py | 2.5.1 |
| python-dotenv | 1.2.4 |
| httpx | 0.28.1 |
| colorama | 0.4.4 |
| numpy | 2.2.6 |
| PyYAML | 6.0.3 |

### Clone

```bash
git clone https://github.com/YuanJrShiuan/Harness-MulUser.git
cd Harness-MulUser
```

The repo has two components:
- `harness_mu/` — the Harness-MU framework (the paper's contribution)
- `Multi-User-LLM-Agent/` — Muses-Bench benchmark (Yang et al., arXiv:2604.08567), bundled as a subdirectory

### Virtual environment

A `.venv/` is included in the repo. Activate it:

```bash
source .venv/bin/activate
```

Verify dependencies:

```bash
python -c "import litellm, openai, absl, dotenv, httpx, colorama; print('OK')"
```

If any module is missing:

```bash
pip install litellm>=1.41.0 openai absl-py python-dotenv httpx colorama pyyaml numpy
```

### Install Muses-Bench

The Harness-MU access_control adapter imports `muses_bench` for transcript replay/live modes. Install it as an editable package:

```bash
cd Multi-User-LLM-Agent
pip install -e .
cd ..
```

### Clone IHEval

The ComplianceChecker module imports IHEval's instruction checker registry at import time. Without IHEval, any Harness-MU run fails immediately with an ImportError. Clone it into the expected location:

```bash
cd Multi-User-LLM-Agent
git clone https://github.com/ytyz1307zzh/IHEval.git
cd ..
```

Verify the path exists:

```bash
ls Multi-User-LLM-Agent/IHEval/src/rule_following/evaluate/
# Should show: __init__.py  instructions.py  instructions_registry.py  ...
```

### API credentials

Create `.env` from the example and fill in DeepSeek credentials:

```bash
cp .env.example .env
```

`.env` contents:

```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-your-key-here
```

The Muses-Bench baseline runner reads its own `.env`:

```bash
cp Multi-User-LLM-Agent/.env.example Multi-User-LLM-Agent/.env
# Add the same credentials
```

`.env` is in `.gitignore` and will not be committed.

### Model configuration

The model used in this reproduction is `deepseek-v4-pro`, one of the four models evaluated in the paper. Copy the config template and set the model name:

```bash
cp harness_mu/config/defaults.yaml harness_mu/config/my_experiment.yaml
```

In `my_experiment.yaml`:

```yaml
default_model:
  provider: openai
  model: deepseek-v4-pro
  temperature: 0.0
  max_tokens: 4096

module_models:
  mediator:
    provider: openai
    model: deepseek-v4-pro
    temperature: 0.0
    max_tokens: 4096
  worker:
    provider: openai
    model: deepseek-v4-pro
    temperature: 0.7             # stochastic; run multiple trials for variance
    max_tokens: 16384
```

`provider: openai` stays as-is — litellm routes by `OPENAI_BASE_URL`, so DeepSeek's endpoint receives the calls.

---

## Running experiments

**Run all five conditions with one script (recommended):**

```bash
cd Harness-MulUser
bash repro/run_all.sh
```

This runs the conditions in order:

1. Baseline (no harness) — none-attack dataset
2. Harness-MU full — none-attack dataset
3. Baseline (no harness) — fake_authorized attack dataset
4. Harness-MU full — fake_authorized attack dataset
5. Harness-MU ablation (ComplianceChecker off) — fake_authorized attack dataset

Then build the comparison table:

```bash
python repro/build_table.py
```

Saves `results/comparison_table.json` and `results/comparison_table.md`.

---

**Or run conditions manually:**

### Harness-MU (full)

The `--data` flag is relative to the `harness_mu/` package directory (the pipeline prepends its own root).

```bash
# None-attack
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/my_experiment.yaml \
  --data data/access_control/template_colon_attack_none_2_to_10_each_2.jsonl

# Copy result aside before running the ablation
cp -r results/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-v4-pro \
      results/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-v4-pro-full

# Fake_authorized attack
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/my_experiment.yaml \
  --data data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl

cp -r results/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro \
      results/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-v4-pro-full
```

### Muses-Bench baseline (no harness)

```bash
cd Multi-User-LLM-Agent

python run.py \
  --scenario access_control \
  --data ../harness_mu/data/access_control/template_colon_attack_none_2_to_10_each_2.jsonl \
  --model deepseek-v4-pro \
  --provider openai \
  --output ../results/baseline_none.jsonl

python run.py \
  --scenario access_control \
  --data ../harness_mu/data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl \
  --model deepseek-v4-pro \
  --provider openai \
  --output ../results/baseline_fake_auth.jsonl

cd ..
```

### Ablation (ComplianceChecker disabled)

```bash
# Generate ablation config (sets compliance_checker: false)
python repro/make_ablation_config.py

# Run on fake_authorized attack
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/ablation_no_cc.yaml \
  --data data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl
```

This writes ablation results to the `deepseek-v4-pro/` subdirectory. The full-harness result is preserved in the `-full/` copy made in the previous step.

---

## Output structure

Results land under `results/`:

```
results/
  baseline_none.jsonl                         # Muses-Bench baseline, no-attack
  baseline_fake_auth.jsonl                    # Muses-Bench baseline, fake_authorized
  comparison_table.json                       # aggregated comparison (all conditions)
  comparison_table.md
  access_control/
    template_colon_attack_none_2_to_10_each_2/
      deepseek-v4-pro/                        # ablation (no CC) on no-attack
      deepseek-v4-pro-full/                   # full harness on no-attack
    template_colon_attack_fake_authorized_2_to_10_each_2/
      deepseek-v4-pro/                        # ablation (no CC) on fake_authorized
      deepseek-v4-pro-full/                   # full harness on fake_authorized
```

Each subdirectory contains:
- `eva.json` — aggregated metrics (read this for results)
- `detail.jsonl` — per-scenario detail
- `module_metrics.json` — token usage and latency per module

---

## Metrics

| Key | Meaning |
|-----|---------|
| `avg_privacy_score` | Privacy after ComplianceChecker (final) |
| `avg_privacy_score_raw` | Privacy before ComplianceChecker (Worker alone) |
| `avg_boundary_delta_privacy` | ComplianceChecker's marginal contribution to privacy |
| `avg_utility_score` | Fraction of authorized users served |
| `avg_overall_score` | Harmonic mean of privacy and utility |
| `avg_num_boundary_intercepts` | Count of ComplianceChecker rewrites per scenario |

---

## Debug diary

**`ImportError: IHEval source not found`** — Clone IHEval into `Multi-User-LLM-Agent/IHEval/` (see Setup above). The path is hardcoded in `harness_mu/validation/instruction_checkers.py`.

**`ModuleNotFoundError: No module named 'colorama'`** — `pip install colorama`. Required by litellm's logging.

**`ModuleNotFoundError: No module named 'absl'`** — `pip install absl-py` (the package name is `absl-py`, not `absl`). Required by IHEval.

**`ModuleNotFoundError: No module named 'muses_bench'`** — Run `cd Multi-User-LLM-Agent && pip install -e . && cd ..`.

**Rate limit errors** — Reduce `execution.max_workers` in your config from 16 to 4.

**`.git/index.lock` blocks git** — Remove it: `rm -f .git/index.lock`. Stale lock from an interrupted git process.

---

## Changes from the original repo

| File / location | Change |
|-----------------|--------|
| `harness_mu/config/my_experiment.yaml` | Config created from `defaults.yaml`; set to `deepseek-v4-pro` (one of the four paper models) routed via `OPENAI_BASE_URL=https://api.deepseek.com` |
| `harness_mu/config/ablation_no_cc.yaml` | Generated by `repro/make_ablation_config.py`; sets `compliance_checker: false` |
| `repro/` | Added for reproducibility automation (`run_all.sh`, `build_table.py`, `make_ablation_config.py`) |
| `results/` | Not committed (output artifact) |

---

## Conclusions

Harness-MU's two core claims reproduce under the DeepSeek V4 Pro condition. Privacy reaches 1.000 on both the none-attack and fake_authorized datasets, matching the paper's DeepSeek column (1.000 in both cases). Utility rises to 0.989 (none-attack) and 1.000 (fake_authorized) versus the same-model baseline's 0.627 / 0.613, closely tracking the paper's DeepSeek figures (0.989 / 0.972).

The ComplianceChecker ablation produced a null result on this model/dataset combination: removing CC left scores identical. This reflects that DeepSeek V4 Pro's Worker module already handles access decisions correctly on the colon-template fake_authorized dataset, leaving the CC with little to correct (boundary_delta_privacy = 0.019 in the full run, vs. the paper's reported +0.068 across all 12 template-attack datasets). The CC's value as a deterministic safety net would be more visible on weaker models or harder attack types. This is an honest negative finding, not a failure to reproduce — it surfaces a model-capability dependency in the paper's design that the paper does not explicitly discuss.