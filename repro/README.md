# repro/ — reproduction scripts

Scripts added for the FTEC5660 reproducibility coursework. They are not part of the authors' original
repository. Install and run steps are in the top-level README.

## Files

**`run_all.sh`** — runs the five conditions of the reproduction, then calls `build_table.py`.

```bash
bash repro/run_all.sh              # all five conditions
DRY_RUN=1 bash repro/run_all.sh    # print the plan, no API calls
```

Conditions: `baseline_none`, `full_none`, `baseline_fake`, `full_fake`, `ablation_fake`
(and optionally `ablation_none`). Set `CONDITIONS="full_fake"` to run a subset. Every run passes
`--max-turns 10`.

Each condition writes to `results/runs/trial_1/<condition>/`. Finished conditions are skipped, an
interrupted one resumes at the scenario that stopped, and an attempt that makes no progress for
12 minutes (`STALL_MINUTES`) is stopped and retried.

**`build_table.py`** — reads `results/runs/trial_*/<condition>/metrics.jsonl` and writes
`results/comparison_table.{json,md}`, with the paper's DeepSeek numbers beside each row.

`metrics.jsonl` holds `{scenario_id, metrics}` only. The runner also writes `detail.jsonl` (transcripts)
and `run.log`; both are gitignored and not published, because the model invents credential-shaped strings
during role-play.

```bash
python repro/build_table.py
```

**`make_ablation_config.py`** — writes `harness_mu/config/ablation_no_cc.yaml` from
`my_experiment.yaml` with `compliance_checker: false`. `run_all.sh` does the same inline, so this is only
needed to produce the file by hand.

## The modification

The ComplianceChecker is disabled for the access-control scenario, which is the one config field that
differs between `my_experiment.yaml` and `ablation_no_cc.yaml`.

Disabling it took one code change in `harness_mu/adapters/access_control.py`: the adapter called the
ComplianceChecker unconditionally, so the config flag had no effect. The call is now gated on
`is_enabled(activation)`, and the disabled branch fills the four `compliance.*` artifact keys from the
Worker's raw responses so the evaluator still runs. That is the only change to `harness_mu/`.

## What this fork adds to the authors' repository

Forked from `https://github.com/YuanJrShiuan/Harness-MulUser` at its final commit.

| Path | Status |
|---|---|
| `repro/` | added |
| `README.md` | rewritten for install and run |
| `harness_mu/adapters/access_control.py` | one change: ComplianceChecker gating |
| `harness_mu/config/my_experiment.yaml` | local only, created from `defaults.yaml`; not committed |
| `harness_mu/config/ablation_no_cc.yaml` | local only, generated; not committed |
| `results/` | local only, gitignored |

`harness_mu/` and `Multi-User-LLM-Agent/` are otherwise unchanged from upstream.

## Unused

`run_experiments.py` and `aggregate_results.py` are earlier versions of the runner and the table builder.
They are superseded by `run_all.sh` and `build_table.py` and are not used.
