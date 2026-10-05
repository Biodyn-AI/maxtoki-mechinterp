#!/usr/bin/env python3
"""Study B analysis: generic checklist vs generic review on 22 held-out items.

Usage: analyze_studyB.py <workflow_output.json>
Writes studyB/results/{reviews.csv, items.csv, analysis.json}.

Rules (PREREGISTRATION.md section 4, AMENDMENT_1):
- detected (flawed items): graders 1 and 2 decide when they agree on
  error_detected; grader 3 decides otherwise;
- load-bearing false alarms: findings classified 'false_alarm' and
  load_bearing by the deciding grade (grader 1 when 1 and 2 agree on the
  count, else grader 3); findings marked 'disputed' are resolved by the
  verifier ('false_alarm' -> false alarm, 'real_problem' -> real);
- intervals: percentile bootstrap over items (2,000 resamples, seed 13).
"""
import json, sys, os, csv
import numpy as np
from scipy.stats import wilcoxon

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'studyB', 'results'); os.makedirs(OUT, exist_ok=True)
d = json.load(open(sys.argv[1])); res = [r for r in (d['result'] if 'result' in d else d) if r]
keys = {it['id']: json.load(open(os.path.join(WS, 'studyB', 'keys', it['id'] + '.json'))) for it in json.load(open(os.path.join(WS, 'protocol', 'STUDYB_ITEMS.json')))}
rng = np.random.default_rng(13)

def lb_false(g):
    return [f['id'] for f in (g or {}).get('findings', []) if f['classification'] == 'false_alarm' and f['load_bearing']]

rows = []
for r in res:
    k = keys[r['id']]; a, b, c = r.get('grade_a'), r.get('grade_b'), r.get('grade_c')
    flawed = k['status'] == 'flawed'
    if flawed:
        if a and b and a['error_detected'] == b['error_detected']: det = a['error_detected']
        elif c: det = c['error_detected']
        else: det = (a or b or {}).get('error_detected')
    else:
        det = None
    deciding = a if (a and b and len(lb_false(a)) == len(lb_false(b))) else (c or a or b)
    fa = set(lb_false(deciding))
    disputed = {f['id'] for g in (a, b, c) if g for f in g['findings'] if f['classification'] == 'disputed'}
    vmap = {v['id']: v['verdict'] for v in (r.get('verify') or {}).get('verdicts', [])}
    fa |= {i for i in disputed if vmap.get(i) == 'false_alarm'}
    other_real = {f['id'] for f in (deciding or {}).get('findings', []) if f['classification'] == 'other_real_problem'} | {i for i in disputed if vmap.get(i) == 'real_problem'}
    sev = {f['id']: f['severity'] for f in r['review']['findings']}
    rows.append(dict(item=r['id'], pair=k['pair_id'], status=k['status'], in_checklist=(k.get('error') or {}).get('in_checklist'),
                     arm=r['arm'], rep=r['rep'], n_findings=len(r['review']['findings']),
                     n_critical_or_major=sum(1 for s in sev.values() if s in ('critical', 'major')),
                     detected=det, g1=(a or {}).get('error_detected'), g2=(b or {}).get('error_detected'),
                     n_false_alarm_lb=len(fa), any_false_alarm_lb=len(fa) > 0, n_other_real=len(other_real),
                     n_disputed=len(disputed)))
with open(os.path.join(OUT, 'reviews.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def boot(v, n=2000):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), size=(n, len(v))); m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

items = sorted(set(r['item'] for r in rows))
item_rows = []
for it in items:
    k = keys[it]
    row = dict(item=it, pair=k['pair_id'], status=k['status'], in_checklist=(k.get('error') or {}).get('in_checklist'))
    for arm in ['checklist', 'generic']:
        xs = [r for r in rows if r['item'] == it and r['arm'] == arm]
        row[f'{arm}_n'] = len(xs)
        row[f'{arm}_detect'] = float(np.mean([bool(x['detected']) for x in xs])) if xs and k['status'] == 'flawed' else None
        row[f'{arm}_fa_any'] = float(np.mean([x['any_false_alarm_lb'] for x in xs])) if xs else None
        row[f'{arm}_fa_n'] = float(np.mean([x['n_false_alarm_lb'] for x in xs])) if xs else None
    item_rows.append(row)
with open(os.path.join(OUT, 'items.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(item_rows[0].keys())); w.writeheader(); w.writerows(item_rows)

out = {'n_reviews': len(rows), 'n_items': len(items)}
for arm in ['checklist', 'generic']:
    fl = [r for r in item_rows if r['status'] == 'flawed' and r[f'{arm}_detect'] is not None]
    cl = [r for r in item_rows if r['status'] == 'clean' and r[f'{arm}_fa_any'] is not None]
    o = {}
    if fl:
        o['sensitivity'] = boot([r[f'{arm}_detect'] for r in fl])
        for g in [True, False]:
            sub = [r[f'{arm}_detect'] for r in fl if r['in_checklist'] == g]
            if sub: o[f'sensitivity_in_checklist_{g}'] = boot(sub) + (len(sub),)
    if cl:
        o['clean_any_false_alarm'] = boot([r[f'{arm}_fa_any'] for r in cl])
        o['clean_false_alarms_per_review'] = boot([r[f'{arm}_fa_n'] for r in cl])
    o['flawed_any_false_alarm'] = boot([r[f'{arm}_fa_any'] for r in item_rows if r['status'] == 'flawed' and r[f'{arm}_fa_any'] is not None])
    xs = [r for r in rows if r['arm'] == arm]
    o['mean_findings'] = float(np.mean([x['n_findings'] for x in xs])); o['mean_other_real'] = float(np.mean([x['n_other_real'] for x in xs]))
    out[arm] = o
fl = [r for r in item_rows if r['status'] == 'flawed' and r['checklist_detect'] is not None and r['generic_detect'] is not None]
if fl:
    dif = np.array([r['checklist_detect'] - r['generic_detect'] for r in fl]); nz = dif[dif != 0]
    out['paired_sensitivity_checklist_minus_generic'] = dict(zip(['mean', 'ci_low', 'ci_high'], boot(dif))) | {'wilcoxon_p': float(wilcoxon(nz).pvalue) if len(nz) else 1.0, 'n_items': len(fl)}
cl = [r for r in item_rows if r['status'] == 'clean' and r['checklist_fa_any'] is not None and r['generic_fa_any'] is not None]
if cl:
    dif = np.array([r['checklist_fa_any'] - r['generic_fa_any'] for r in cl]); nz = dif[dif != 0]
    out['paired_false_alarm_checklist_minus_generic'] = dict(zip(['mean', 'ci_low', 'ci_high'], boot(dif))) | {'wilcoxon_p': float(wilcoxon(nz).pvalue) if len(nz) else 1.0, 'n_items': len(cl)}
g = [(r['g1'], r['g2']) for r in rows if r['status'] == 'flawed' and r['g1'] is not None and r['g2'] is not None]
if g:
    a = np.array([x[0] for x in g], float); b = np.array([x[1] for x in g], float)
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    out['grader_agreement_detected'] = dict(n=len(g), agreement=float(po), kappa=float((po - pe) / (1 - pe)) if pe < 1 else 1.0)
json.dump(out, open(os.path.join(OUT, 'analysis.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
