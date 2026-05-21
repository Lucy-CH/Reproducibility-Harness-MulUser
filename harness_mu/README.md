# Harness-MU

A minimal three-layer harness for evaluating multi-user LLM scenarios. The
goal is to test whether *hard constraints* (privacy, instruction following)
hold under multi-user contention, by pushing as much enforcement as possible
out of the LLM and into deterministic Python code.

## Architecture

```
Skill Layer    (per-scenario adapter — orchestrates the pipeline)
   │
   ▼
Mediation Layer (LLM, single pass — plans utility under fixed constraints)
   │
   ▼
Worker         (LLM — produces the actual response per user, in parallel)
   │
   ▼
Boundary Layer (deterministic Python — last, non-bypassable filter)
```

* **Boundary Layer** (`modules/compliance_checker.py`) never calls an LLM.
  It runs IFEval-style white-box checkers (IF) and substring/marker scans
  (AC), and **fail-closes** by overwriting any leaking response with a
  refusal template.
* **Mediation Layer** (`modules/mediator.py`) is a single-pass LLM planner.
  Authority / authorization is taken **verbatim from ground-truth fields**;
  the LLM only paraphrases constraints and proposes per-user response
  postures. A second pass-2 LLM-as-judge has been deliberately removed.
* **Skill Layer** (`adapters/{instruction_following,access_control}.py`)
  glues parsing + module activation + per-scenario evaluation.
* **Gatekeeper** (`modules/gatekeeper.py`) is AC-only and deterministic by
  default: `is_authorized` ↔ membership in `scenario.authorized_users`;
  `malicious` ↔ non-authorized user under a known `attack_type`.
* **Worker** (`modules/worker.py`) supports `run_per_user` (default), which
  issues one isolated LLM call per user in parallel. This is the primary
  defense against cross-user prompt injection.
* **Transcript Replay** (AC-only, optional) — when
  `scenarios.access_control.transcript.mode == "replay"` the Skill Layer
  loads a per-scenario JSONL cache produced by Muses-Bench (multi-turn
  `SimulatedUser` dialogue) and overwrites each user's `instructions`
  with the flattened multi-turn attack content. The Worker then sees
  bit-identical attack text to what the Muses-Bench defender saw,
  enabling apples-to-apples comparison. Open-loop: inter-turn
  dynamics are lost on the Harness side. See
  `docs/run_experiments.md §9.6` for the generate-then-replay workflow.

## Directory layout

```
harness_mu/
├── __main__.py                 # python -m harness_mu entry
├── run.py                      # CLI parser
├── adapters/                   # Skill Layer
│   ├── base.py                 #   ScenarioAdapter ABC
│   ├── instruction_following.py
│   └── access_control.py
├── modules/                    # Boundary + Mediation modules
│   ├── base.py                 #   BaseModule ABC + activation flags
│   ├── compliance_checker.py   #   Boundary Layer (deterministic)
│   ├── mediator.py             #   Mediation Layer (LLM, single pass)
│   ├── gatekeeper.py           #   AC admission control (deterministic)
│   └── worker.py               #   per-user parallel LLM worker
├── core/
│   ├── pipeline.py             #   PipelineOrchestrator (batch driver)
│   └── types.py                #   PipelineContext / ModuleResult / ScenarioRecord
├── llm/
│   ├── factory.py              #   LLMFactory (per-module model assignment)
│   └── wrappers.py             #   InstrumentedLLMCaller (latency + tokens)
├── config/
│   ├── schema.py               #   pydantic config models
│   ├── defaults.yaml           #   default model + scenario activation
│   ├── ac_replay_bmini.yaml    #   AC strict replay (4 modules = deepseek)
│   └── loader.py               #   YAML + HMU_* env-var override
├── metrics/
│   ├── collector.py            #   thread-safe per-module aggregation
│   └── custom/
│       ├── instruction_following.py   # authority / per-user fairness
│       └── access_control.py          # privacy / utility / overall (F1)
├── prompts/prompts.py          # System prompts for all modules
├── validation/
│   └── instruction_checkers.py # Thin wrapper forwarding to IHEval
├── utils/                      # llm_utils, file_utils, timer, format_utils
└── data/                       # bundled sample datasets
    ├── instruction_following/
    │   ├── aligned_2_to_10.jsonl       (187 scenarios)
    │   └── conflict_2_to_10.jsonl      (368 scenarios)
    └── access_control/
        ├── template_{colon,says,xml}_attack_{none,fake_authorized,pressure,roleplaying}_*.jsonl
        │                               (12 files × 18 scenarios = 216)
        ├── none_sample.jsonl           (3 benign  scenarios, sanity check)
        └── fake_authorized_sample.jsonl(3 attack  scenarios, sanity check)
```

Transcript cache for AC replay lives outside `harness_mu/` at
`<repo_root>/transcripts/access_control/`; both `muses_bench`
(generator) and `harness_mu` (consumer) read/write the same files.

## Configuration

Defaults live in `config/defaults.yaml` (model `qwen/qwen3.6-35b-a3b`).
Override with a YAML file (`--config foo.yaml`) or env vars
(`HMU_DEFAULT_MODEL`, `HMU_WORKER_MODEL`, `HMU_MEDIATOR_MODEL`,
`HMU_DEBUG`, `HMU_MAX_WORKERS`, …).

API credentials are read from a `.env` file at the repo root or
`harness_mu/.env`. Required keys: `OPENAI_API_KEY`, `OPENAI_BASE_URL`.

Per-scenario module activation is in `defaults.yaml > scenarios.<name>.modules`:

| Scenario              | gatekeeper | mediator | worker | compliance_checker |
|-----------------------|------------|----------|--------|--------------------|
| instruction_following | false      | true*    | true   | true               |
| access_control        | true       | true     | true   | true               |

*Mediator runs only when `meta.mode == "conflict"`.

Per-scenario transcript replay is in
`defaults.yaml > scenarios.access_control.transcript`:

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"off"` | `"off"` skips replay (uses dataset `initial_messages`); `"replay"` loads cached multi-turn attack content. |
| `cache_dir` | `null` | `null` → `<repo_root>/transcripts`. Override to point elsewhere. |
| `simulator_model` | `deepseek-v4-pro` | Must match the `--user-model` used when generating the cache on the Muses-Bench side; participates in the cache filename. |
| `max_turns` | `10` | Must match the `--max-turns` used when generating; participates in the cache filename. |

The `ac_replay_bmini.yaml` config is a ready-made override that flips
all four AC modules to `deepseek-v4-pro` and sets `mode: replay` — used
for the strict apples-to-apples comparison against Muses-Bench. See
`docs/run_experiments.md §9.6` for the full generate-then-replay flow.

## Running experiments

From the repo root (`/workspace/fangewudi6/Harness_MulUser`), with the
bundled venv active:

```bash
source venv/bin/activate

# IF aligned, 3 sample scenarios
python -m harness_mu --scenario instruction_following --debug

# IF conflict mode (set data file explicitly)
python -m harness_mu --scenario instruction_following --debug \
    --data data/instruction_following/conflict_2_to_10.jsonl

# AC benign (none) attacks
python -m harness_mu --scenario access_control --debug

# AC fake_authorized attacks (Boundary Layer stress test)
python -m harness_mu --scenario access_control --debug \
    --data data/access_control/fake_authorized_sample.jsonl

# AC strict-replay (fair vs Muses-Bench: same defender, same attack text)
# Requires cache file under transcripts/access_control/ — generate first
# via Muses-Bench: see docs/run_experiments.md §9.6.
python -m harness_mu --scenario access_control \
    --config harness_mu/config/ac_replay_bmini.yaml
```

Useful flags: `--debug` (first 3 scenarios, single worker), `--verbose`
(DEBUG logging incl. HTTP), `--config path.yaml`, `--output path.jsonl`.

## Output

Every run writes to `results/<scenario>/<dataset_stem>/<model>[_debug]/`:

* `detail.jsonl` — one line per scenario with metrics, all module results
  (latency, tokens, data dict), and parsed worker outputs.
* `eva.json` — running aggregate (avg accuracy, privacy, utility, …).
* `module_metrics.json` — per-module total calls / tokens / latency.

The orchestrator resumes from `detail.jsonl`: re-running skips already-
processed scenario IDs. Delete the directory for a clean run.

## Key metrics

* **IF**: `accuracy`, `weighted_accuracy`, plus
  `_harness_mu.{authority_compliance_score, per_user_satisfaction}`.
* **AC**: `privacy_score = 1 - leak_rate(unauth)`,
  `utility_score = success_rate(auth)`,
  `overall_score = F1(privacy, utility)`,
  `num_boundary_intercepts` (how many times the Boundary Layer rewrote a
  leaking response — direct evidence of the layer's contribution).

## Instruction-following checkers (IHEval)

`validation/instruction_checkers.py` is a thin wrapper that re-exports
the official rule-based checker registry from
[IHEval](https://github.com/ytyz1307zzh/IHEval) (NAACL 2025). The
Muses-Bench IF evaluator uses the exact same registry, so harness_mu and
muses_bench grade IF outputs with **byte-identical** Python — any
accuracy delta is attributable to the pipeline, not the metric.

Setup (one-time, in the repo root):

```bash
git clone https://github.com/ytyz1307zzh/IHEval.git Multi-User-LLM-Agent/IHEval
source venv/bin/activate
pip install rouge-score colorama termcolor langdetect immutabledict nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The wrapper resolves IHEval at
`Multi-User-LLM-Agent/IHEval/src/rule_following/evaluate/`. Override
with the `IHEVAL_SRC` env var if checked out elsewhere. Import fails
loudly if the path is missing — there is no silent-fallback empty
registry.

## Future work

* **IHEval Safety Defense as a single-user baseline.** IHEval ships
  four `safety_defense` tasks (`hijack_*`, `extraction_*`) whose threat
  model is structurally identical to our AC scenarios but with a single
  user. Wiring them into harness_mu as an additional adapter would let
  us report cross-benchmark generalization for the Boundary Layer.
  Tracked but deliberately deferred.
