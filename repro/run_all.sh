#!/bin/bash
# Run the Harness-MU reproduction conditions, several independent trials each.
#
# Usage (from anywhere inside the repo):
#   bash repro/run_all.sh              # trials 1 to 3
#   bash repro/run_all.sh 2 3          # only trials 2 and 3
#   DRY_RUN=1 bash repro/run_all.sh    # print what would run, call no API
#   CONDITIONS="full_fake ablation_fake" bash repro/run_all.sh 4 5
#
# Conditions (default: the first five):
#   baseline_none   Muses-Bench baseline, no-attack dataset
#   full_none       Harness-MU, all modules on, no-attack dataset
#   baseline_fake   Muses-Bench baseline, fake_authorized dataset
#   full_fake       Harness-MU, all modules on, fake_authorized dataset
#   ablation_fake   Harness-MU with ComplianceChecker off, fake_authorized dataset
#   ablation_none   Harness-MU with ComplianceChecker off, no-attack dataset (optional)
#
# Every run writes to its own folder:
#   results/runs/trial_<N>/<condition>/{detail.jsonl, eva.json, module_metrics.json, run.log}
#
# Safe to re-run: a condition that already has every scenario is skipped, and an
# interrupted one resumes where it stopped. Scenarios that failed with an error
# are removed from the file so they are retried.
#
# All runs use max-turns = 10, the setting of the paper's Table 1.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FIRST_TRIAL="${1:-1}"
LAST_TRIAL="${2:-3}"
CONDITIONS="${CONDITIONS:-baseline_none full_none baseline_fake full_fake ablation_fake}"
DRY_RUN="${DRY_RUN:-0}"
MAX_TURNS=10
MAX_ATTEMPTS=2
# Kill a batch attempt that makes no progress for this many minutes.
STALL_MINUTES="${STALL_MINUTES:-12}"
POLL_S="${POLL_S:-30}"

CONFIG="$REPO_ROOT/harness_mu/config/my_experiment.yaml"
ABLATION_CONFIG="$REPO_ROOT/harness_mu/config/ablation_no_cc.yaml"
DEFAULTS="$REPO_ROOT/harness_mu/config/defaults.yaml"
DATA_DIR="$REPO_ROOT/harness_mu/data/access_control"
DS_NONE="template_colon_attack_none_2_to_10_each_2.jsonl"
DS_FAKE="template_colon_attack_fake_authorized_2_to_10_each_2.jsonl"
RUNS="$REPO_ROOT/results/runs"

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
echo "[INFO] Python: $PY"

# ---------------- preflight ----------------
fail=0
for f in "$REPO_ROOT/.env" "$REPO_ROOT/Multi-User-LLM-Agent/.env"; do
  if [ ! -f "$f" ]; then echo "[ERROR] Missing credentials file: $f"; fail=1; fi
done
if [ ! -d "$REPO_ROOT/Multi-User-LLM-Agent/IHEval/src" ]; then
  echo "[ERROR] IHEval not found at Multi-User-LLM-Agent/IHEval (see README, 'Clone IHEval')"; fail=1
fi
for f in "$DATA_DIR/$DS_NONE" "$DATA_DIR/$DS_FAKE"; do
  if [ ! -f "$f" ]; then echo "[ERROR] Missing dataset: $f"; fail=1; fi
done
if [ "$fail" -ne 0 ]; then exit 1; fi

# Two runs appending to the same detail.jsonl would duplicate scenarios.
stale="$("$PY" - <<'PYEOF'
import os, subprocess
me = {os.getpid(), os.getppid()}
out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
for line in out.splitlines():
    pid, _, args = line.strip().partition(" ")
    if not pid.isdigit() or int(pid) in me:
        continue
    exe = args.split()[0] if args.split() else ""
    if "python" not in os.path.basename(exe):
        continue
    if ("-m harness_mu" in args or "harness_mu" in args.split(" ")[:6]) or \
       ("run.py" in args and "Multi-User-LLM-Agent" in args):
        print(pid)
PYEOF
)"
if [ -n "$stale" ]; then
  echo "[ERROR] A previous run still appears to be active (pid: $(echo $stale | tr '\n' ' '))."
  echo "        Stop it first:  kill $(echo $stale | tr '\n' ' ')"
  exit 1
fi

# ---------------- configs ----------------
# my_experiment.yaml is created from defaults.yaml if it does not exist yet
# (MODEL defaults to deepseek-v4-pro). An existing file is left untouched.
# The ablation config is regenerated from it on every run, so the two configs
# differ in exactly one field.
MODEL="$(MODEL="${MODEL:-deepseek-v4-pro}" "$PY" - "$DEFAULTS" "$CONFIG" "$ABLATION_CONFIG" <<'EOF'
import copy, os, sys, yaml
defaults, config, ablation = sys.argv[1:4]
if not os.path.exists(config):
    cfg = yaml.safe_load(open(defaults))
    model = os.environ["MODEL"]
    cfg["default_model"]["model"] = model
    for m in cfg.get("module_models", {}).values():
        m["model"] = model
    cfg["scenarios"]["access_control"]["enabled"] = True
    with open(config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[OK] created {config} from defaults.yaml with model {model}", file=sys.stderr)
cfg = yaml.safe_load(open(config))
ac = cfg["scenarios"]["access_control"]
if not ac.get("enabled", False):
    sys.exit("[ERROR] scenarios.access_control.enabled is false in my_experiment.yaml")
abl = copy.deepcopy(cfg)
abl["scenarios"]["access_control"]["modules"]["compliance_checker"] = False
with open(ablation, "w") as f:
    yaml.dump(abl, f, default_flow_style=False, sort_keys=False)
print(cfg["default_model"]["model"])
EOF
)" || exit 1
echo "[INFO] Model (from my_experiment.yaml): $MODEL"
echo "[INFO] Trials $FIRST_TRIAL..$LAST_TRIAL | conditions: $CONDITIONS | max-turns $MAX_TURNS"

# ---------------- helpers ----------------
# Print every descendant pid of a process (macOS has no setsid; pgrep -P works
# on both macOS and Linux).
descendants() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do
    echo "$kid"
    descendants "$kid"
  done
}

# Print the number of usable scenario rows in a detail.jsonl. Rows that carry an
# "error" (or are not valid JSON) are removed so the runner retries them.
count_done() {
  DRY_RUN="$DRY_RUN" "$PY" - "$1" <<'EOF'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print(0); sys.exit()
keep, bad = [], 0
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        bad += 1; continue
    if row.get("error"):
        bad += 1; continue
    keep.append(line if line.endswith("\n") else line + "\n")
if bad and os.environ.get("DRY_RUN") != "1":
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(keep)
    print(f"[prune] removed {bad} failed row(s) from {path}", file=sys.stderr)
# Dedupe: a second process appending to the same file would repeat a scenario.
seen, uniq, dups = set(), [], 0
for line in keep:
    sid = json.loads(line).get("scenario_id")
    if sid in seen:
        dups += 1
        continue
    seen.add(sid); uniq.append(line)
if dups and os.environ.get("DRY_RUN") != "1":
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(uniq)
    print(f"[dedupe] removed {dups} duplicate row(s) from {path}", file=sys.stderr)
print(len(uniq))
EOF
}

INCOMPLETE=""

run_condition() {
  trial="$1"; cond="$2"
  case "$cond" in
    baseline_none) kind=baseline; ds="$DS_NONE"; cfg="" ;;
    baseline_fake) kind=baseline; ds="$DS_FAKE"; cfg="" ;;
    full_none)     kind=harness;  ds="$DS_NONE"; cfg="$CONFIG" ;;
    full_fake)     kind=harness;  ds="$DS_FAKE"; cfg="$CONFIG" ;;
    ablation_none) kind=harness;  ds="$DS_NONE"; cfg="$ABLATION_CONFIG" ;;
    ablation_fake) kind=harness;  ds="$DS_FAKE"; cfg="$ABLATION_CONFIG" ;;
    *) echo "[ERROR] Unknown condition: $cond"; exit 1 ;;
  esac

  out="$RUNS/trial_$trial/$cond"
  log="$out/detail.jsonl"
  expected="$(grep -c . "$DATA_DIR/$ds")"
  done_n="$(count_done "$log")"

  echo ""
  echo "============================================================"
  echo "  trial $trial | $cond | $done_n/$expected scenarios done"
  echo "============================================================"
  if [ "$done_n" -ge "$expected" ]; then
    echo "[SKIP] already complete"
    return
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] would run $kind on $ds -> $out"
    return
  fi
  mkdir -p "$out"

  attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    done_n="$(count_done "$log")"
    if [ "$done_n" -ge "$expected" ]; then break; fi
    echo "[RUN] attempt $attempt/$MAX_ATTEMPTS, $done_n/$expected to go  ($(date '+%H:%M:%S'))"

    if [ "$kind" = "baseline" ]; then
      cmd=( "$PY" run.py --scenario access_control --data "$DATA_DIR/$ds"
            --model "$MODEL" --provider openai --max-turns "$MAX_TURNS" --output "$log" )
      workdir="$REPO_ROOT/Multi-User-LLM-Agent"
    else
      cmd=( "$PY" -m harness_mu --scenario access_control --config "$cfg"
            --data "data/access_control/$ds" --max-turns "$MAX_TURNS" --output "$log" )
      workdir="$REPO_ROOT"
    fi

    # Run in the background, then stop it if no new scenario lands in
    # STALL_MINUTES. A killed attempt keeps the rows already written, so the
    # next attempt resumes at the scenario that stalled.
    (
      cd "$workdir" || exit 1
      exec "${cmd[@]}"
    ) >> "$out/run.log" 2>&1 &
    runner=$!

    last_n="$done_n"; last_progress="$(date +%s)"
    while kill -0 "$runner" 2>/dev/null; do
      sleep "$POLL_S"
      now_n="$(count_done "$log")"
      if [ "$now_n" -gt "$last_n" ]; then
        last_n="$now_n"; last_progress="$(date +%s)"
        echo "  [progress] $now_n/$expected scenarios  ($(date '+%H:%M:%S'))"
      elif [ $(( $(date +%s) - last_progress )) -gt $(( STALL_MINUTES * 60 )) ]; then
        echo "[WARN] no progress for ${STALL_MINUTES} min at $now_n/$expected, stopping this attempt"
        # TERM the run and everything it spawned, then KILL whatever is left.
        descendants "$runner" | xargs -r kill -TERM 2>/dev/null
        kill -TERM "$runner" 2>/dev/null
        for _ in 1 2 3 4 5 6; do
          kill -0 "$runner" 2>/dev/null || break
          sleep 5
        done
        descendants "$runner" | xargs -r kill -KILL 2>/dev/null
        kill -KILL "$runner" 2>/dev/null
        break
      fi
    done
    wait "$runner" 2>/dev/null

    done_n="$(count_done "$log")"
    if [ "$done_n" -ge "$expected" ]; then break; fi
    echo "[WARN] $done_n/$expected scenarios after attempt $attempt"
    attempt=$((attempt + 1))
  done

  if [ "$done_n" -ge "$expected" ]; then
    echo "[DONE] trial $trial | $cond  ($(date '+%H:%M:%S'))"
  else
    echo "[INCOMPLETE] trial $trial | $cond: $done_n/$expected"
    INCOMPLETE="$INCOMPLETE trial_$trial/$cond($done_n/$expected)"
  fi
}

# ---------------- main loop ----------------
# Trials are the outer loop, so each trial is finished before the next starts.
trial="$FIRST_TRIAL"
while [ "$trial" -le "$LAST_TRIAL" ]; do
  for cond in $CONDITIONS; do
    run_condition "$trial" "$cond"
  done
  trial=$((trial + 1))
done

echo ""
echo "============================================================"
if [ -n "$INCOMPLETE" ]; then
  echo "  FINISHED WITH INCOMPLETE RUNS:$INCOMPLETE"
  echo "  Re-run the same command to retry only the missing scenarios."
else
  echo "  ALL REQUESTED RUNS COMPLETE"
fi
echo "============================================================"

if [ "$DRY_RUN" != "1" ]; then
  "$PY" "$REPO_ROOT/repro/build_table.py"
fi
