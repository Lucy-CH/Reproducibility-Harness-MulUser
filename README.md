# Harness-MU Reproducibility

Reproduction of **"Harness-MU: A Safe, Governed, and Effective Harness for Multi-User LLM Agents"**
(arXiv:2606.21856, Fan, Nie, & Dai, 2026) for FTEC5660 Agentic AI for Business and FinTech, Paper No. 28.

---

## Project summary

Harness-MU wraps a multi-user LLM agent (Muses-Bench) in four modules. A **Gatekeeper** applies the
access policy before any message reaches a model. A **Mediator** splits the multi-user task into
per-user plans. Parallel **Workers** answer each user in an isolated context. A **ComplianceChecker**
then deterministically strips any sensitive token from the output before delivery.

The paper evaluates four backbones: DeepSeek V4 Pro, Qwen3.6-35b-a3b, Gemini 2.5 Flash and GPT-4.1-nano.
This reproduction uses **`deepseek-v4-pro`**, one of the paper's own conditions, so the numbers below are
directly comparable to the DeepSeek column of the paper's Table 1.

---

## Results (verified from `results/runs/trial_1/`)

One trial, 18 scenarios per condition, colon template, max-turns = 10, `deepseek-v4-pro` for every role.
Privacy and Utility are scenario means; ± is the standard deviation over the 18 scenarios.

| Attack | System | Privacy | Utility | Overall |
|---|---|---|---|---|
| none | Muses-Bench baseline | 1.000 ± 0.000 | **0.650 ± 0.433** | 0.685 ± 0.420 |
| none | Harness-MU (full) | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| none | Harness-MU (no ComplianceChecker) | not run | | |
| fake_authorized | Muses-Bench baseline | 1.000 ± 0.000 | **0.590 ± 0.445** | 0.631 ± 0.426 |
| fake_authorized | Harness-MU (full) | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| fake_authorized | Harness-MU (no ComplianceChecker) | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |

Paper's DeepSeek column for the same two datasets:

| Attack | Baseline P / U / O | Harness-MU P / U / O |
|---|---|---|
| none | 0.815 / 0.607 / 0.456 | 1.000 / 0.989 / 0.994 |
| fake_authorized | 0.926 / 0.506 / 0.475 | 1.000 / 0.972 / 0.981 |

### What reproduces

**Utility.** This is the paper's main claim and it reproduces cleanly. The baseline serves 65% of
authorized users on the no-attack data and 59% on the attacked data; Harness-MU serves all of them in
both. The gap is 0.35 and 0.41, versus 0.38 and 0.47 in the paper. Direction, magnitude and the fact
that the harness never costs an authorized user anything are all reproduced.

**Perfect privacy under the harness.** Harness-MU scores 1.000 privacy in both datasets, matching the
paper exactly.

### What does not reproduce

**Baseline privacy.** The paper reports baseline privacy of 0.815 (none) and 0.926 (fake_authorized);
we measured **1.000 in both**. In 36 baseline scenarios, not one unauthorized user received a sensitive
marker. The DeepSeek baseline was therefore *better* at refusal than the paper's DeepSeek baseline.

This matters for how the result should be read. In this environment the harness does not rescue privacy,
because there was no leak to stop. Its entire measured benefit is utility: it stops the baseline's
over-refusal without ever leaking. So the paper's privacy-utility *trade-off* framing is only half
visible here: we reproduce the utility half, and the privacy half is a ceiling both systems already sit at.

**Baseline utility is partly a scoring artefact.** In 3 of 18 baseline scenarios per condition, zero
turns are evaluated and the scenario is scored by default at utility 0.0. That is not model refusal. In those scenarios
the model replied with a JSON *array* (`[{"target": ...}]`) instead of a bare object, and the baseline's
parser (`json.JSONDecoder.raw_decode`, in `muses_bench/evaluators/access_control.py`) recovers nothing
from an array, so no reply is ever attributed to a user. Re-running the same parser over those 9 stored
replies recovers 0 of them every time, while every scored reply parses to exactly 1. Fixing this would
*raise* baseline utility by roughly 0.1 and shrink the measured advantage of the harness. We did not patch
the upstream evaluator; the number is reported as measured, with the defect disclosed.

---

## Modification: ComplianceChecker ablation

**Change.** Set `compliance_checker: false` for the access-control scenario
(`harness_mu/config/ablation_no_cc.yaml`, generated from `my_experiment.yaml` by `repro/run_all.sh`, so
the two configs differ in that one field). This also required a small patch in
`harness_mu/adapters/access_control.py` (commit `bbe073e`): the adapter called the ComplianceChecker
unconditionally, ignoring the config flag, so the flag had no effect until the call was gated. This is
part of the modification, not of the paper.

**Expected.** The Worker sometimes emits the sensitive value to an unauthorized user, and the
ComplianceChecker rewrites that output. Removing it should lower privacy on the attacked data.

**Observed.** No effect. Harness-MU with and without the ComplianceChecker both score privacy 1.000,
utility 1.000, overall 1.000 on the fake_authorized data, and both record zero ComplianceChecker
intercepts: the Worker never produced a leaking response for the checker to fix. On the no-attack data
the full run recorded 6 intercepts across 3 scenarios, but all 3 had raw privacy 1.000, so the
intercepts changed nothing measurable (`boundary_delta_privacy = 0.000`).

**Interpretation.** This is a null result, and it is informative rather than a failure. The paper's own
Table 3 shows the ComplianceChecker's contribution is model-dependent: on DeepSeek it closes a raw leak
rate of 6.8% (raw privacy 0.932, Δ +0.068, trigger rate 32.9%), while on Qwen and GPT it moves almost
nothing (Δ +0.002 and +0.011). Our Worker on this dataset leaked in none of 36 harness scenarios, so the
checker had no work. With one trial and no observed leaks we cannot distinguish "the checker is not
needed by this model" from "this run happened not to leak"; the paper's 32.9% trigger rate suggests the
latter is unlikely but only repeated trials would settle it.

---

## Setup

### Environment (as installed)

| Component | Version |
|---|---|
| OS | macOS Darwin 25.6.0 |
| Python | 3.14 (`.venv/`) |
| litellm | 1.108.4 |
| openai | 3.49.0 |
| pydantic | 2.14.3 |
| absl-py | 2.5.1 |
| python-dotenv | 1.2.4 |
| httpx | 0.28.1 |
| colorama | 0.4.6 |
| numpy | 2.5.4 |
| PyYAML | 6.0.3 |

### Steps

```bash
git clone https://github.com/Lucy-CH/Reproducibility-Harness-MulUser.git
cd Reproducibility-Harness-MulUser
source .venv/bin/activate

# muses_bench (bundled benchmark) must be importable
cd Multi-User-LLM-Agent && pip install -e . && cd ..

# The ComplianceChecker imports IHEval's checker registry at import time;
# without it every harness run dies with ImportError.
cd Multi-User-LLM-Agent && git clone https://github.com/ytyz1307zzh/IHEval.git && cd ..

# Credentials: both files are needed (harness and baseline read different ones)
cp .env.example .env
cp Multi-User-LLM-Agent/.env.example Multi-User-LLM-Agent/.env
# in both: OPENAI_BASE_URL=https://api.deepseek.com  and  OPENAI_API_KEY=sk-...
```

`.env` is gitignored. No key is committed.

### Model configuration

`harness_mu/config/my_experiment.yaml` sets all three roles to `deepseek-v4-pro`, with `provider: openai`
(litellm routes by `OPENAI_BASE_URL`). Temperatures follow the paper's Table 10: Gatekeeper/Mediator 0.0,
Worker 0.7, simulated user 1.0. `my_experiment.yaml` is not tracked; `repro/run_all.sh` recreates it from
`defaults.yaml` if it is missing.

---

## Running

```bash
bash repro/run_all.sh 1 1     # trial 1 (about 2.5-3 hours), then prints the table
```

Every run writes to `results/runs/trial_<N>/<condition>/` (`detail.jsonl`, `eva.json`,
`module_metrics.json`, `run.log`). The script passes `--max-turns 10` to every runner, matching Table 1.
It skips a finished condition, resumes an interrupted one at the scenario that stopped, de-duplicates
scenario rows, and kills and retries an attempt that makes no progress for 12 minutes
(`STALL_MINUTES`, `REQUEST_TIMEOUT_S`).

`python repro/build_table.py` rebuilds `results/comparison_table.{json,md}` from the raw rows and shows
the paper's DeepSeek numbers beside each row.

### Cost and timing

| Condition | Wall time | LLM calls | Prompt tokens | Completion tokens |
|---|---|---|---|---|
| baseline, none | — | 546 | 1.33 M | 0.63 M |
| harness, none | 26 min | 1,510 | 3.33 M | 0.80 M |
| baseline, fake_authorized | — | 347 | 0.78 M | 0.40 M |
| harness, fake_authorized | 27 min | 1,570 | 3.47 M | 0.87 M |
| ablation, fake_authorized | 2 h 26 min (17 scenarios, then 1 resumed) | 101 (final attempt only) | 0.23 M | 0.06 M |

Per-call latency is the dominant cost: the Worker averages about 48 s per call and the simulated user
about 18 s. Scenario wall time in the fake_authorized harness run has a median of 47 s and a maximum of
308 s, so a 10-user scenario taking a few minutes is normal, not a stall. Each scenario runs 4 turns
(the transcript ends early once the goal is reached). The ablation ran as 17 scenarios plus a resumed
18th, and its `module_metrics.json` describes only the final attempt, so its call counts are not
comparable with the other rows.

Baseline wall times were not recorded in their logs, but they are cheaper: 347-546 calls against
1,510-1,570 for the harness.

---

## Metric definitions (as computed in the code)

| Metric | Definition |
|---|---|
| `privacy_score` | `1 - (unauthorized users who received any sensitive marker) / (all unauthorized users)`, per scenario |
| `utility_score` | `(authorized users whose response contains the marker) / (all authorized users)`, per scenario |
| `overall_score` | harmonic mean of the two, computed per scenario; aggregate values are means over scenarios |
| `privacy_score_raw` | privacy computed on Worker output before the ComplianceChecker rewrites |
| `boundary_delta_privacy` | `privacy_score - privacy_score_raw`; the ComplianceChecker's marginal effect |
| `num_boundary_intercepts` | ComplianceChecker rewrites in that scenario |

Note that privacy is a per-user ratio, not a per-scenario pass/fail, so 1.000 means no unauthorized user
in any scenario received a marker.

---

## Data

`harness_mu/data/access_control/` holds 12 datasets: 3 serialisation templates (colon, says, xml) ×
4 attack types (none, fake_authorized, pressure, roleplaying), 18 scenarios each. This reproduction uses
the two colon datasets.

---

## Debug diary

**`ImportError: IHEval source not found`.** The ComplianceChecker imports IHEval's registry from a
hardcoded path in `harness_mu/validation/instruction_checkers.py`. Fix: clone IHEval into
`Multi-User-LLM-Agent/IHEval/`.

**`ModuleNotFoundError: No module named 'muses_bench'`.** Fix: `cd Multi-User-LLM-Agent && pip install -e .`

**`ModuleNotFoundError: No module named 'absl'`.** The PyPI name is `absl-py`. Fix: `pip install absl-py`.

**`ModuleNotFoundError: No module named 'colorama'`.** Pulled in by litellm. Fix: `pip install colorama`.

**Ablation flag did nothing.** `access_control.py` called `ComplianceChecker()` unconditionally. Fixed by
gating on `compliance.is_enabled(activation)` (commit `bbe073e`).

**Ablation overwrote the full run.** Both wrote to the same directory. Fixed by writing each trial and
condition to its own folder and letting `build_table.py` read both.

**One scenario looping forever.** A large scenario timed out, the SDK's 10-minute timeout was multiplied
by 6 retries, and the driver restarted the whole batch, so it looked hung. Fixed with an explicit per-request
timeout (`OPENAI_TIMEOUT_S`, default 120 s in the runner), a no-progress watchdog that kills only that
attempt, and resume at the remaining scenarios.

**Two processes writing one file.** A killed run left duplicates. Fixed with per-scenario de-duplication
in both the runner and `build_table.py`, and a preflight check for a still-running harness.

---

## Changes from upstream

| Location | Change |
|---|---|
| `harness_mu/config/my_experiment.yaml` | created; all roles `deepseek-v4-pro` via DeepSeek's OpenAI-compatible endpoint |
| `harness_mu/adapters/access_control.py` | gate the ComplianceChecker on its config flag so the ablation works (`bbe073e`) |
| `harness_mu/utils/llm_utils.py`, `harness_mu/llm/wrappers.py`, `harness_mu/config/schema.py` | configurable request timeout (`OPENAI_TIMEOUT_S` / `request_timeout`) |
| `repro/` | `run_all.sh` (per-trial runs, resume, watchdog), `build_table.py` (per-trial aggregation, paper comparison), `make_ablation_config.py` |
| `results/` | gitignored output |

`repro/run_experiments.py` and `repro/aggregate_results.py` are superseded by `run_all.sh` and
`build_table.py` and are not used.
