# Harness-MU — reproduction code

Code for reproducing the access-control result of *Harness-MU: A Safe, Governed, and Effective Harness
for Multi-User LLM Agents* (arXiv:2606.21856) on DeepSeek V4 Pro, plus one modification (ComplianceChecker
disabled). The experiment scripts are in `repro/`.

---

## Requirements

- macOS or Linux, Python 3.14
- A DeepSeek API key (OpenAI-compatible endpoint)
- About 4 hours of wall time for all five conditions; no GPU needed

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
    metrics.jsonl         one JSON line per scenario, with its metrics
    eva.json              aggregate metrics for the condition
    module_metrics.json   token usage and latency per module
results/comparison_table.{json,md}   summary built by build_table.py
```

A run also writes `detail.jsonl` (full transcripts) and `run.log` next to those. The transcripts are
committed, because Sections 4.3 and 5.2 of the report argue from them. They contain synthetic secret
markers that the benchmark defines and that the model reproduces during role-play; five of those
strings match generic secret-scanner rules, so `repro/redact_transcripts.py` replaces them with
placeholders before the files are published. `run.log` is not committed.

Conditions are `baseline_none`, `full_none`, `baseline_fake`, `full_fake`, `ablation_fake`
(and optionally `ablation_none`).

## Known fault in the baseline parser

The evaluator in `Multi-User-LLM-Agent/muses_bench/evaluators/access_control.py` decodes each reply
with `json.JSONDecoder.raw_decode` and keeps only results that are a bare object with `target` and
`content`. When the model answers with an array of such objects instead —
`[{"target": ..., "content": ...}, ...]` — the reply is discarded. The scenario is then written out
with the evaluator's initialised defaults, privacy 1.0 and utility 0.0.

In the recorded trial this happens in 3 of 18 baseline scenarios in each dataset, and in 22 later-turn
replies inside four already-scored baseline scenarios. The affected rows are visible in
`results/runs/trial_1/<condition>/detail.jsonl`: their `per_turn_metrics` list is empty.

This is not fixed. The evaluator and the recorded results are left exactly as they ran, so the numbers
in `results/` are what the original code produced. The consequence for reading them: baseline utility
is understated. Re-parsing the same stored replies with an array-tolerant version of the same parser
would raise baseline utility from 0.650 to about 0.872 on `none` and from 0.590 to about 0.756 on
`fake_authorized`.

A separate fault: the "Never scored" column in `results/comparison_table.md` always reads 0. Its flag
is computed from `metrics.jsonl`, which has no `per_turn_metrics` field, so the test is false for every
row. The true count is 3. Only that column is affected; the scores are not.

## If a step fails

`ImportError: IHEval source not found` — IHEval is not at `Multi-User-LLM-Agent/IHEval/`; repeat the clone step
above.

`ModuleNotFoundError: No module named 'muses_bench'` — repeat `cd Multi-User-LLM-Agent && pip install -e .`.

`ModuleNotFoundError: No module named 'absl'` or `'colorama'` — `pip install absl-py colorama` (note the
package name `absl-py`).

`[ERROR] A previous run still appears to be active` — an earlier run is still writing to the results
directory. Stop it: `pkill -f "harness_mu"; pkill -f "Multi-User-LLM-Agent/run.py"`.

Rate limits — lower `execution.max_workers` in `harness_mu/config/my_experiment.yaml` from 16.

