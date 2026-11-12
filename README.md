# Harness-MU — reproduction code

Code for reproducing the access-control result of *Harness-MU: A Safe, Governed, and Effective Harness
for Multi-User LLM Agents* (arXiv:2606.21856) on DeepSeek V4 Pro, plus one modification (ComplianceChecker
disabled). The experiment scripts are in `repro/`.

---

## Requirements

- macOS or Linux, Python 3.14
- A DeepSeek API key (OpenAI-compatible endpoint)
- About 2.5 hours of wall time for all five conditions; no GPU needed

## Install

```bash
git clone https://github.com/Lucy-CH/Reproducibility-Harness-MulUser.git
cd Reproducibility-Harness-MulUser
python3 -m venv .venv
source .venv/bin/activate
pip install litellm openai pydantic absl-py python-dotenv httpx colorama numpy pyyaml
```

Install the bundled benchmark, which the harness imports as `muses_bench`:

```bash
cd Multi-User-LLM-Agent && pip install -e . && cd ..
```

Clone IHEval, which the ComplianceChecker imports at load time:

```bash
cd Multi-User-LLM-Agent
git clone https://github.com/ytyz1307zzh/IHEval.git
cd ..
ls IHEval/src/rule_following/evaluate/     # must not be empty
```

## Configure

Create both `.env` files (the harness and the baseline each read their own; neither is committed):

```bash
cp .env.example .env
cp Multi-User-LLM-Agent/.env.example Multi-User-LLM-Agent/.env
```

In each file:

```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-your-key-here
```

The model roles are read from `harness_mu/config/my_experiment.yaml`. If that file does not exist, the run
script creates it from `defaults.yaml` with `deepseek-v4-pro` for all three roles (Gatekeeper and Mediator
at temperature 0.0, Worker at 0.7, simulated users at 1.0).

## Run

```bash
bash repro/run_all.sh
```

This runs the five conditions (baseline and Harness-MU on both datasets, plus the ComplianceChecker
ablation) with `--max-turns 10`, then prints the comparison table.

Other invocations:

```bash
DRY_RUN=1 bash repro/run_all.sh       # print the plan, make no API calls
CONDITIONS="full_fake" bash repro/run_all.sh   # a single condition
python repro/build_table.py           # rebuild the table from existing results
```

The script skips a condition that is already complete, resumes an interrupted one at the scenario that
stopped, and retries an attempt that makes no progress for 12 minutes. Re-running the same command is
always safe.

## Outputs

```
results/runs/trial_1/<condition>/
    detail.jsonl          one JSON line per scenario, with its metrics
    eva.json              aggregate metrics for the condition
    module_metrics.json   token usage and latency per module
    run.log               console output of the run
results/comparison_table.{json,md}   summary built by build_table.py
```

Conditions are `baseline_none`, `full_none`, `baseline_fake`, `full_fake`, `ablation_fake`
(and optionally `ablation_none`).

## If a step fails

`ImportError: IHEval source not found` — IHEval is not at `Multi-User-LLM-Agent/IHEval/`; repeat the clone step
above.

`ModuleNotFoundError: No module named 'muses_bench'` — repeat `cd Multi-User-LLM-Agent && pip install -e .`.

`ModuleNotFoundError: No module named 'absl'` or `'colorama'` — `pip install absl-py colorama` (note the
package name `absl-py`).

`[ERROR] A previous run still appears to be active` — an earlier run is still writing to the results
directory. Stop it: `pkill -f "harness_mu"; pkill -f "Multi-User-LLM-Agent/run.py"`.

Rate limits — lower `execution.max_workers` in `harness_mu/config/my_experiment.yaml` from 16.
