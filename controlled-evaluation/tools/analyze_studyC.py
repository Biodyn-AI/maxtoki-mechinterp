#!/usr/bin/env python3
"""Study C analysis: blind re-audit of the pre-audit MaxToki snapshot.

Inputs (all under studyC/results/): workflow_output.json (9 reviews with
grades and verifier verdicts), costs.csv (per-agent tokens/tool calls/wall
time), ../../protocol/STUDYC_KEY.csv (frozen answer key, 68 items).
Outputs: studyC/results/analysis.json, item_matrix.csv, per_run.csv,
per_arm.csv and source-data CSVs for the figures.

Detection is scored on the adjudicated grade ('final'). 'yes' = full
detection; 'partial' is reported separately. Intervals: percentile bootstrap
over key items (2,000 resamples, seed 7), because items are the unit the arms
are compared on; run-to-run spread is shown separately.
"""
import json, csv, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
R = os.path.join(WS, 'studyC', 'results')
res = json.load(open(os.path.join(R, 'workflow_output.json')))['result']
key = list(csv.DictReader(open(os.path.join(WS, 'protocol', 'STUDYC_KEY.csv'))))
ids = [k['id'] for k in key]
K = {k['id']: k for k in key}
ARMS = ['deployed', 'checklist', 'generic']
rng = np.random.default_rng(7)


def pattern_group(p):
    p = (p or '').strip()
    m = re.match(r'P(10|[1-9])\b', p)
    return 'in_checklist' if m else 'outside_checklist'


def kappa(a, b, cats):
    a = np.array(a); b = np.array(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- item matrix
runs = []
for r in res:
    fin = {i['ledger_id']: i['detected'] for i in r['final']['items']}
    ga = {i['ledger_id']: i['detected'] for i in r['a']['items']}
    gb = {i['ledger_id']: i['detected'] for i in r['b']['items']}
    ver = collections.Counter(v['verdict'] for v in (r.get('verify') or {}).get('verdicts', []))
    matched = sum(1 for f in r['final']['findings'] if f.get('matched_ledger_ids'))
    runs.append(dict(arm=r['arm'], rep=r['rep'], final=fin, a=ga, b=gb,
                     n_findings=len(r['review']['findings']),
                     n_findings_matched=matched,
                     unmatched_real=ver.get('real_error', 0), unmatched_trivial=ver.get('real_but_trivial', 0),
                     unmatched_false=ver.get('false_alarm', 0), unmatched_undecided=ver.get('cannot_decide', 0),
                     sev={f['id']: f['severity'] for f in r['review']['findings']}))

with open(os.path.join(R, 'item_matrix.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['id', 'severity', 'error_type', 'pattern_group', 'original_audit'] + [f"{x['arm']}_{x['rep']}" for x in runs])
    for i in ids:
        w.writerow([i, K[i]['severity'], K[i]['error_type'], pattern_group(K[i]['audit_pattern']), K[i]['detected_by_original_audit_sweep']] + [x['final'].get(i, 'no') for x in runs])

# ---------------------------------------------------------------- per run
per_run = []
for x in runs:
    yes = [i for i in ids if x['final'].get(i) == 'yes']
    part = [i for i in ids if x['final'].get(i) == 'partial']
    row = dict(arm=x['arm'], rep=x['rep'], n_findings=x['n_findings'], yes=len(yes), partial=len(part),
               yes_or_partial=len(yes) + len(part))
    for s in ['critical', 'major', 'minor']:
        sid = [i for i in ids if K[i]['severity'] == s]
        row[f'yes_{s}'] = sum(1 for i in sid if i in yes)
        row[f'n_{s}'] = len(sid)
    for g in ['in_checklist', 'outside_checklist']:
        gid = [i for i in ids if pattern_group(K[i]['audit_pattern']) == g]
        row[f'yes_{g}'] = sum(1 for i in gid if i in yes)
        row[f'n_{g}'] = len(gid)
    row.update(unmatched_real=x['unmatched_real'], unmatched_trivial=x['unmatched_trivial'],
               unmatched_false=x['unmatched_false'], unmatched_undecided=x['unmatched_undecided'])
    # grader agreement on the 3-level detected grade
    row['kappa_graders'] = round(kappa([x['a'].get(i, 'no') for i in ids], [x['b'].get(i, 'no') for i in ids], ['yes', 'partial', 'no']), 3)
    row['agree_graders'] = sum(x['a'].get(i) == x['b'].get(i) for i in ids)
    per_run.append(row)

# costs
cost = {}
cp = os.path.join(R, 'costs.csv')
if os.path.exists(cp):
    for c in csv.DictReader(open(cp)):
        lab = c['label']
        if lab.startswith('C:'):
            _, arm, rep = lab.split(':')
            cost[(arm, int(rep))] = c
for row in per_run:
    c = cost.get((row['arm'], row['rep']))
    if c:
        row['output_tokens'] = int(c['output_tokens'])
        row['input_tokens_incl_cache'] = int(c['input_tokens']) + int(c['cache_read_tokens']) + int(c['cache_creation_tokens'])
        row['tool_calls'] = int(c['tool_calls'])
        row['wall_minutes'] = round(float(c['wall_seconds']) / 60, 1)

with open(os.path.join(R, 'per_run.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_run[0].keys())); w.writeheader(); w.writerows(per_run)

# ---------------------------------------------------------------- per arm, item-level
def item_rate(arm, level=('yes',)):
    reps = [x for x in runs if x['arm'] == arm]
    return np.array([np.mean([x['final'].get(i, 'no') in level for x in reps]) for i in ids])

def boot_mean(v, n=2000):
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

arm_rows = []
item_rates = {a: item_rate(a) for a in ARMS}
item_rates_any = {a: item_rate(a, ('yes', 'partial')) for a in ARMS}
for a in ARMS:
    m, lo, hi = boot_mean(item_rates[a])
    m2, lo2, hi2 = boot_mean(item_rates_any[a])
    reps = [x for x in runs if x['arm'] == a]
    union = sum(1 for i in ids if any(x['final'].get(i) == 'yes' for x in reps))
    crit = [i for i in ids if K[i]['severity'] == 'critical']
    arm_rows.append(dict(arm=a, detection_rate=m, ci_low=lo, ci_high=hi,
                         detection_rate_incl_partial=m2, ci_low_incl_partial=lo2, ci_high_incl_partial=hi2,
                         union_yes_over_3_runs=union, n_items=len(ids),
                         critical_rate=float(np.mean([item_rates[a][ids.index(i)] for i in crit]))))
orig = np.array([1.0 if K[i]['detected_by_original_audit_sweep'] == 'yes' else 0.0 for i in ids])
orig_part = np.array([1.0 if K[i]['detected_by_original_audit_sweep'] in ('yes', 'partly') else 0.0 for i in ids])
m, lo, hi = boot_mean(orig)
arm_rows.append(dict(arm='original_audit', detection_rate=m, ci_low=lo, ci_high=hi,
                     detection_rate_incl_partial=float(orig_part.mean()), ci_low_incl_partial=None, ci_high_incl_partial=None,
                     union_yes_over_3_runs=int(orig.sum()), n_items=len(ids), critical_rate=float(np.mean([orig[ids.index(i)] for i in ids if K[i]['severity'] == 'critical']))))
with open(os.path.join(R, 'per_arm.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(arm_rows[0].keys())); w.writeheader(); w.writerows(arm_rows)

# paired comparisons by item (difference in item-level detection rates)
from scipy.stats import wilcoxon
pairs = {}
for a, b in [('checklist', 'generic'), ('deployed', 'generic'), ('deployed', 'checklist')]:
    d = item_rates[a] - item_rates[b]
    mm, lo, hi = boot_mean(d)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    pairs[f'{a}_minus_{b}'] = dict(mean_diff=mm, ci_low=lo, ci_high=hi, wilcoxon_p=p, n_items_differing=int(len(nz)))
# every fresh arm vs the original audit (fresh runs are a strict superset of chances)
for a in ARMS:
    d = item_rates[a] - orig
    mm, lo, hi = boot_mean(d)
    pairs[f'{a}_minus_original'] = dict(mean_diff=mm, ci_low=lo, ci_high=hi)

all_union = sum(1 for i in ids if any(x['final'].get(i) == 'yes' for x in runs))
never = [i for i in ids if all(x['final'].get(i, 'no') == 'no' for x in runs)]
never_by_sev = collections.Counter(K[i]['severity'] for i in never)

# by error type
by_type = {}
for t in sorted(set(K[i]['error_type'] for i in ids)):
    tid = [ids.index(i) for i in ids if K[i]['error_type'] == t]
    by_type[t] = {a: float(item_rates[a][tid].mean()) for a in ARMS} | {'original': float(orig[tid].mean()), 'n': len(tid)}
by_group = {}
for g in ['in_checklist', 'outside_checklist']:
    gid = [ids.index(i) for i in ids if pattern_group(K[i]['audit_pattern']) == g]
    by_group[g] = {a: float(item_rates[a][gid].mean()) for a in ARMS} | {'original': float(orig[gid].mean()), 'n': len(gid)}

out = dict(n_items=len(ids), per_arm=arm_rows, paired=pairs, union_all_9_runs=all_union,
           never_detected=len(never), never_detected_by_severity=dict(never_by_sev), never_detected_ids=never,
           by_error_type=by_type, by_pattern_group=by_group,
           grader_kappa_mean=float(np.mean([r['kappa_graders'] for r in per_run])),
           grader_kappa_range=[float(min(r['kappa_graders'] for r in per_run)), float(max(r['kappa_graders'] for r in per_run))],
           unmatched_real_total=sum(r['unmatched_real'] for r in per_run),
           unmatched_false_total=sum(r['unmatched_false'] for r in per_run))
json.dump(out, open(os.path.join(R, 'analysis.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('never_detected_ids',)}, indent=1))
