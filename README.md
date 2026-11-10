# Harness-MU Reproducibility — Setup Notes

Reproduction of "Harness-MU: A Safe, Governed, and Effective Harness for Multi-User LLM Agents" (arXiv:2606.21856, Fan, Nie, & Dai, 2026) for FTEC5660 Agentic AI for Business and FinTech.

## Environment

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

## Clone

```bash
git clone https://github.com/YuanJrShiuan/Harness-MulUser.git
cd Harness-MulUser
```

The repo has two components:
- `harness_mu/` — the Harness-MU framework (the paper's contribution)
- `Multi-User-LLM-Agent/` — Muses-Bench benchmark (Yang et al., arXiv:2604.08567), bundled as a subdirectory

## Virtual environment

A `.venv/` is included in the repo. Activate it:

```bash
source .venv/bin/activate
```

Verify dependencies:

```bash
python -c "import litellm, openai, absl, dotenv, httpx, colorama; print('OK')"
```

If any module is missing, install it:

```bash
pip install litellm>=1.41.0 openai absl-py python-dotenv httpx colorama pyyaml numpy
```

## Install Muses-Bench

The Harness-MU access_control adapter imports `muses_bench` for transcript replay/live modes. Install it as an editable package:

```bash
cd Multi-User-LLM-Agent
pip install -e .
cd ..
```

This makes `muses_bench` importable and installs its dependencies (openai, numpy, termcolor, litellm>=1.41.0).

## Clone IHEval

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

IHEval depends on `absl-py`, which is already in the venv.

## API credentials

Harness-MU uses litellm, which reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment. For DeepSeek (OpenAI-compatible):

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-your-key-here
```

The Muses-Bench baseline runner (`Multi-User-LLM-Agent/run.py`) reads its own `.env`:

```bash
cp Multi-User-LLM-Agent/.env.example Multi-User-LLM-Agent/.env
# Add the same credentials
```

`.env` is in `.gitignore` and will not be committed.

## Configuration

Copy the documented template and edit the model name:

```bash
cp harness_mu/config/defaults.yaml harness_mu/config/my_experiment.yaml
```

In `my_experiment.yaml`, set the model for all three model roles (default, mediator, worker):

```yaml
default_model:
  provider: openai
  model: deepseek-chat        # changed from qwen/qwen3.6-35b-a3b
  temperature: 0.0
  max_tokens: 4096

module_models:
  mediator:
    provider: openai
    model: deepseek-chat
    temperature: 0.0
    max_tokens: 4096
  worker:
    provider: openai
    model: deepseek-chat
    temperature: 0.7          # stochastic; run multiple trials
    max_tokens: 16384
```

`provider: openai` stays as-is — litellm routes by `OPENAI_BASE_URL`, so DeepSeek's endpoint receives the calls. The model swap from Qwen to DeepSeek is documented as Edge Case A in the assignment instructions; results are not directly comparable to the paper's Qwen numbers because the underlying LLM differs.

## Sanity check

Run the harness on 3 scenarios to verify the pipeline works:

```bash
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/my_experiment.yaml \
  --debug \
  --verbose
```

Expected output on the none-attack dataset:

```
==================================================
  Scenarios completed: 3
  Average privacy_score  100.00%
  Average utility_score  100.00%
  Average overall_score  100.00%
==================================================
```

Results are written to `results/access_control/<dataset>/<model>_debug/`:
- `detail.jsonl` — per-scenario detail (one JSON line per scenario)
- `eva.json` — aggregated metrics (read this for your report)
- `module_metrics.json` — token usage and latency per module

## Running experiments

**Run all conditions with one script** (recommended):

```bash
bash repro/run_all.sh
```

This runs five conditions in order:
1. Baseline (no harness) on none-attack dataset
2. Harness-MU (full) on none-attack dataset
3. Baseline (no harness) on fake_authorized attack dataset
4. Harness-MU (full) on fake_authorized attack dataset
5. Harness-MU (ablation: ComplianceChecker off) on fake_authorized attack dataset

Then build the comparison table:

```bash
python repro/build_table.py
```

Prints a table to stdout and saves `results/comparison_table.json` and `results/comparison_table.md`.

---

**Or run conditions manually:**

Before running, make sure `access_control` is enabled in `my_experiment.yaml` (`enabled: true` under `scenarios.access_control`).

### Harness-MU (full)

```bash
# None-attack dataset
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/my_experiment.yaml \
  --data harness_mu/data/access_control/template_colon_attack_none_2_to_10_each_2.jsonl

# Copy result aside before running the next condition on the same dataset
cp -r results/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-chat \
      results/access_control/template_colon_attack_none_2_to_10_each_2/deepseek-chat-full

# Fake_authorized attack dataset
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/my_experiment.yaml \
  --data harness_mu/data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl

cp -r results/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-chat \
      results/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-chat-full
```

Results land in `results/access_control/<dataset>/deepseek-chat-full/`:
- `detail.jsonl` — per-scenario detail (one JSON line per scenario)
- `eva.json` — aggregated metrics (read this for your report)
- `module_metrics.json` — token usage and latency per module

### Muses-Bench baseline (no harness)

```bash
cd Multi-User-LLM-Agent

python run.py \
  --scenario access_control \
  --data ../harness_mu/data/access_control/template_colon_attack_none_2_to_10_each_2.jsonl \
  --model deepseek-chat \
  --provider openai \
  --output ../results/baseline_none.jsonl

python run.py \
  --scenario access_control \
  --data ../harness_mu/data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl \
  --model deepseek-chat \
  --provider openai \
  --output ../results/baseline_fake_auth.jsonl

cd ..
```

Note: the baseline runner also writes its own `eva.json` to `results/access_control/<dataset>/deepseek-chat/`. The `--output` flag writes the per-scenario JSONL used by `build_table.py`.

### Ablation (ComplianceChecker disabled)

```bash
# Generate ablation config (sets compliance_checker: false)
python repro/make_ablation_config.py

# Run on fake_authorized attack (the condition where CC matters most)
python -m harness_mu \
  --scenario access_control \
  --config harness_mu/config/ablation_no_cc.yaml \
  --data harness_mu/data/access_control/template_colon_attack_fake_authorized_2_to_10_each_2.jsonl
```

This overwrites `results/access_control/template_colon_attack_fake_authorized_2_to_10_each_2/deepseek-chat/eva.json` with ablation results. The full-harness result is preserved in the `-full/` copy made in the previous step.

### Build comparison table

```bash
python repro/build_table.py
```

Reads `results/baseline_none.jsonl`, `results/baseline_fake_auth.jsonl`, all `deepseek-chat-full/eva.json` files (full harness), and the `deepseek-chat/eva.json` files alongside a `-full/` copy (ablation). Prints results to stdout. Saves `results/comparison_table.json` and `results/comparison_table.md`.

## Model swap documentation

| Parameter | Paper (Qwen) | This reproduction |
|-----------|-------------|-------------------|
| Model | qwen/qwen3.6-35b-a3b | deepseek-chat |
| Provider | openai (MAAS) | openai (DeepSeek API) |
| Base URL | (MAAS endpoint) | https://api.deepseek.com |
| Temperature (default/mediator) | 0.0 | 0.0 |
| Temperature (worker) | 0.7 | 0.7 |
| Max tokens (default/mediator) | 4096 | 4096 |
| Max tokens (worker) | 16384 | 16384 |

Results are not directly comparable to the paper because the underlying LLM differs. Comparisons are made fairly: same tasks, same metrics, same evaluation scripts, same datasets.

## Data

Access control datasets are bundled at `harness_mu/data/access_control/`. Twelve datasets cross 3 serialization templates (colon, says, XML) with 4 attack vectors (none, fake_authorized, pressure, roleplaying). Two `*_sample.jsonl` files contain 3-row subsets for quick sanity checks.

Instruction following datasets are at `harness_mu/data/instruction_following/`:
- `aligned_2_to_10.jsonl` — aligned instructions (no conflict)
- `conflict_2_to_10.jsonl` — conflicting instructions with authority hierarchy

## Metrics

Harness-MU's `eva.json` reports dual-output metrics:

| Key | Meaning |
|-----|---------|
| `avg_privacy_score` | Privacy after ComplianceChecker (final) |
| `avg_privacy_score_raw` | Privacy before ComplianceChecker (Worker alone) |
| `avg_boundary_delta_privacy` | ComplianceChecker's contribution to privacy |
| `avg_utility_score` | Fraction of authorized users served |
| `avg_overall_score` | Harmonic mean of privacy and utility |
| `avg_accuracy` | IF: fraction of instruction constraints satisfied |
| `avg_authority_compliance` | IF: authority-weighted accuracy |
| `avg_num_boundary_intercepts` | Count of ComplianceChecker rewrites |

## Troubleshooting

**`ImportError: IHEval source not found`** — Clone IHEval into `Multi-User-LLM-Agent/IHEval/` (see above). The path is hardcoded in `harness_mu/validation/instruction_checkers.py`.

**`ModuleNotFoundError: No module named 'colorama'`** — `pip install colorama`. Required by litellm's logging.

**`ModuleNotFoundError: No module named 'absl'`** — `pip install absl-py` (the package is `absl-py`, not `absl`). Required by IHEval.

**`ModuleNotFoundError: No module named 'muses_bench'`** — Run `cd Multi-User-LLM-Agent && pip install -e . && cd ..`.

**Rate limit errors** — Reduce `execution.max_workers` in your config from 16 to 4.

**`.git/index.lock` blocks git** — Remove it: `rm -f .git/index.lock`. This is a stale lock from an interrupted git process.
