"""Cross-evaluate AC: apply both evaluators to both pipelines' outputs.

Reconstructs per-user response dicts from raw outputs of each pipeline, then
runs muses-bench's evaluator (with heuristic) and harness_mu's strict evaluator
on the same inputs. Prints a 2x2 table per dataset and macro averages.
"""
import json
import sys
from pathlib import Path
from glob import glob

ROOT = Path('/workspace/fangewudi6/Harness_MulUser')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Multi-User-LLM-Agent'))

from harness_mu.metrics.custom.access_control import evaluate_access_control as eval_strict
from muses_bench.metrics.access_control_metrics import evaluate_access_control as eval_lenient


def load_scenario_meta(dataset_name):
    """Load ground-truth authorized_users / sensitive_markers / secret_key per scenario."""
    data_root = ROOT / 'Multi-User-LLM-Agent/data/scenarios/access_control/test_datasets/controlled_exp_large'
    f = data_root / f'{dataset_name}.jsonl'
    if not f.exists():
        return {}
    meta = {}
    with open(f) as fp:
        for line in fp:
            d = json.loads(line)
            er = d.get('expected_result', {}) or {}
            meta[d['id']] = {
                'authorized_users': d.get('authorized_users', []),
                'all_user_ids': [u['id'] for u in d.get('all_users', [])],
                'sensitive_markers': er.get('sensitive_markers') or d.get('sensitive_markers', []),
                'secret_key': er.get('sensitive_data') or d.get('secret_key', ''),
            }
    return meta


def muses_per_user_responses(raw_outputs, all_user_ids):
    """Reconstruct {user_id: cumulative_response} from muses raw_outputs (list of turns)."""
    out = {uid: '' for uid in all_user_ids}
    for item in raw_outputs or []:
        rr = item.get('raw_response', '')
        try:
            p = json.loads(rr)
            tgt = p.get('target', '')
            ct = p.get('content', '')
            if tgt in out:
                out[tgt] = (out[tgt] + ' ' + ct).strip()
        except Exception:
            continue
    return out


def hmu_per_user_responses(raw_outputs):
    """Extract {user_id: response} from harness_mu raw_outputs."""
    wp = raw_outputs.get('worker_parsed', '{}') if isinstance(raw_outputs, dict) else '{}'
    try:
        d = json.loads(wp) if isinstance(wp, str) else wp
    except Exception:
        return {}
    out = {}
    for uid, val in (d or {}).items():
        if isinstance(val, str):
            try:
                vp = json.loads(val)
                out[uid] = vp.get('content', val) if isinstance(vp, dict) else val
            except Exception:
                out[uid] = val
        elif isinstance(val, dict):
            out[uid] = val.get('content', json.dumps(val))
        else:
            out[uid] = str(val)
    return out


def aggregate(results_iter, evaluator_strict, evaluator_lenient):
    acc = {'strict': [0.0, 0.0, 0.0], 'lenient': [0.0, 0.0, 0.0]}
    n = 0
    for resp_by_user, meta in results_iter:
        s = evaluator_strict(resp_by_user, meta['authorized_users'], meta['all_user_ids'],
                             meta['sensitive_markers'], meta['secret_key'])
        l = evaluator_lenient(resp_by_user, meta['secret_key'], meta['authorized_users'],
                              meta['all_user_ids'], None, meta['sensitive_markers'])
        acc['strict'][0] += s['privacy_score']; acc['strict'][1] += s['utility_score']; acc['strict'][2] += s['overall_score']
        acc['lenient'][0] += l['privacy_score']; acc['lenient'][1] += l['utility_score']; acc['lenient'][2] += l['overall_score']
        n += 1
    if n == 0:
        return None
    return {k: [v / n for v in vals] for k, vals in acc.items()}, n


def iter_muses(dataset_name, meta):
    p = ROOT / f'Multi-User-LLM-Agent/results/access_control/{dataset_name}/deepseek-v4-pro/results.jsonl'
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            sid = d['scenario_id']
            if sid not in meta:
                continue
            yield muses_per_user_responses(d.get('raw_outputs'), meta[sid]['all_user_ids']), meta[sid]


def iter_hmu(dataset_name, meta):
    p = ROOT / f'results/access_control/{dataset_name}/deepseek-v4-pro/detail.jsonl'
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('scenario_id', '')
            if sid not in meta:
                continue
            yield hmu_per_user_responses(d.get('raw_outputs', {})), meta[sid]


datasets = sorted([Path(p).name for p in
                   glob(str(ROOT / 'results/access_control/template_*'))])
print(f'Found {len(datasets)} datasets')

macro = {('muses','strict'): [0,0,0], ('muses','lenient'): [0,0,0],
         ('hmu','strict'): [0,0,0], ('hmu','lenient'): [0,0,0]}
mc = 0

for ds in datasets:
    meta = load_scenario_meta(ds)
    if not meta:
        print(f'SKIP {ds}: no meta'); continue
    rm = aggregate(iter_muses(ds, meta), eval_strict, eval_lenient)
    rh = aggregate(iter_hmu(ds, meta),   eval_strict, eval_lenient)
    if rm is None or rh is None:
        print(f'SKIP {ds}: missing results'); continue
    (rm_a, n_m), (rh_a, n_h) = rm, rh
    print(f'\n{ds}  muses_n={n_m}  hmu_n={n_h}')
    print(f'  muses pipeline:  strict P={rm_a["strict"][0]:.3f} U={rm_a["strict"][1]:.3f} O={rm_a["strict"][2]:.3f}'
          f'   lenient P={rm_a["lenient"][0]:.3f} U={rm_a["lenient"][1]:.3f} O={rm_a["lenient"][2]:.3f}')
    print(f'  hmu pipeline:    strict P={rh_a["strict"][0]:.3f} U={rh_a["strict"][1]:.3f} O={rh_a["strict"][2]:.3f}'
          f'   lenient P={rh_a["lenient"][0]:.3f} U={rh_a["lenient"][1]:.3f} O={rh_a["lenient"][2]:.3f}')
    for i in range(3):
        macro[('muses','strict')][i]   += rm_a['strict'][i]
        macro[('muses','lenient')][i]  += rm_a['lenient'][i]
        macro[('hmu','strict')][i]     += rh_a['strict'][i]
        macro[('hmu','lenient')][i]    += rh_a['lenient'][i]
    mc += 1

print(f'\n=== MACRO AVERAGE over {mc} datasets ===')
print(f'{"":<22} {"Privacy":>9} {"Utility":>9} {"Overall":>9}')
for (pipe, ev), vals in macro.items():
    print(f'{pipe+"/"+ev:<22} {vals[0]/mc:>9.4f} {vals[1]/mc:>9.4f} {vals[2]/mc:>9.4f}')
