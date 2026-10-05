#!/usr/bin/env python3
"""Study A analysis: 2 x 3 factorial (executor arm x review condition).

Usage: analyze_studyA.py <workflow_output.json> [<workflow_output.json> ...]
Writes studyA/results/{deliverables.csv, analysis.json, reviews.csv}
(or to $STUDYA_OUT if set, used for the exploratory Haiku arm).

Scoring rules (pre-registered, PREREGISTRATION.md section 3):
- primary outcome = verdict correct; graders 1 and 2 decide when they agree,
  grader 3 decides when they disagree (or when one of them is missing);
- trap score = mean over the key's traps of yes=1, partly=0.5, no=0
  (not_applicable skipped), averaged over graders 1 and 2;
- false statements = mean count over graders 1 and 2;
- key numbers = share of matched key numbers within tolerance (graders 1-2 mean).
Contrasts: contract vs paper at review=none (Fisher exact, Newcombe CI,
Mantel-Haenszel stratified by task); paired exact McNemar for
checklist vs generic, checklist vs none, generic vs none; Holm across the four.
"""
import json, sys, os, csv, math
import numpy as np
from scipy.stats import fisher_exact, binomtest, norm

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('STUDYA_OUT', os.path.join(WS, 'studyA', 'results'))  # STUDYA_OUT: exploratory arm
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)

runs = []
for f in sys.argv[1:]:
    d = json.load(open(f))
    runs += [r for r in (d['result'] if isinstance(d, dict) and 'result' in d else d) if r]

H = {'yes': 1.0, 'partly': 0.5, 'no': 0.0}

def grade_summary(g):
    if not g:
        return None
    a, b, c = g.get('a'), g.get('b'), g.get('c')
    gs = [x for x in (a, b) if x]
    if not gs:
        return None
    if a and b and a['verdict_correct'] == b['verdict_correct']:
        correct = a['verdict_correct']
    elif c:
        correct = c['verdict_correct']
    else:
        correct = None
    def trap(x):
        v = [H[t['handled']] for t in x['traps'] if t['handled'] in H]
        return float(np.mean(v)) if v else None
    traps = [trap(x) for x in gs if trap(x) is not None]
    kn = []
    for x in gs:
        v = [k['within_tolerance'] for k in x['key_numbers'] if k['within_tolerance'] is not None]
        if v: kn.append(float(np.mean(v)))
    return dict(correct=correct, verdict_given=(a or b)['verdict_given'],
                g1=a['verdict_correct'] if a else None, g2=b['verdict_correct'] if b else None,
                trap=float(np.mean(traps)) if traps else None,
                false_statements=float(np.mean([len(x['false_statements']) for x in gs])),
                key_numbers=float(np.mean(kn)) if kn else None)

rows = []
reviews = []
for r in runs:
    for cond in ['none', 'generic', 'checklist']:
        s = grade_summary(r.get(f'grade_{cond}'))
        deliv = r['exec'] if cond == 'none' else r.get(f'repaired_{cond}')
        rows.append(dict(task=r['task'], arm=r['arm'], rep=r['rep'], cond=cond,
                         delivered=deliv is not None,
                         verdict=(deliv or {}).get('results', {}).get('verdict') if deliv else None,
                         **({k: v for k, v in s.items()} if s else dict(correct=None, verdict_given=None, g1=None, g2=None, trap=None, false_statements=None, key_numbers=None))))
    for cond in ['generic', 'checklist']:
        rv, rg = r.get(f'review_{cond}'), r.get(f'rgrade_{cond}')
        if rv:
            j = [p['judgement'] for p in (rg or {}).get('points', [])]
            reviews.append(dict(task=r['task'], arm=r['arm'], rep=r['rep'], review=cond,
                                n_points=len(rv['points']),
                                n_critical=sum(p['severity'] == 'critical' for p in rv['points']),
                                valid=j.count('valid'), invalid=j.count('invalid'), unclear=j.count('unclear'),
                                graded=rg is not None))

with open(os.path.join(OUT, 'deliverables.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, 'reviews.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(reviews[0].keys())); w.writeheader(); w.writerows(reviews)

def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def newcombe(k1, n1, k2, n2):
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

def mh_test(strata):
    # strata: list of (a, b, c, d) = (contract correct, contract wrong, paper correct, paper wrong)
    num = 0.0; var = 0.0; ORn = 0.0; ORd = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2: continue
        m1, n1, t1 = a + c, a + b, a + c
        e = (a + b) * (a + c) / n
        v = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        num += a - e; var += v
        ORn += a * d / n; ORd += b * c / n
    chi = (abs(num) - 0.5) ** 2 / var if var > 0 else 0.0
    from scipy.stats import chi2
    return dict(chi2=chi, p=float(chi2.sf(chi, 1)) if var > 0 else 1.0, mh_or=(ORn / ORd) if ORd > 0 else float('inf'))

def get(task=None, arm=None, cond=None):
    return [x for x in rows if (task is None or x['task'] == task) and (arm is None or x['arm'] == arm) and (cond is None or x['cond'] == cond)]

tasks = sorted(set(x['task'] for x in rows))
summary = {'n_executor_runs': len(runs), 'tasks': tasks}
# cell table
cells = []
for t in tasks + ['ALL']:
    for arm in ['paper', 'contract']:
        for cond in ['none', 'generic', 'checklist']:
            xs = [x for x in get(None if t == 'ALL' else t, arm, cond) if x['correct'] is not None]
            k = sum(bool(x['correct']) for x in xs); n = len(xs)
            lo, hi = wilson(k, n) if n else (None, None)
            tr = [x['trap'] for x in xs if x['trap'] is not None]
            fs = [x['false_statements'] for x in xs if x['false_statements'] is not None]
            cells.append(dict(task=t, arm=arm, cond=cond, n=n, correct=k, rate=(k / n if n else None), ci_low=lo, ci_high=hi,
                              trap_mean=(float(np.mean(tr)) if tr else None), false_statements_mean=(float(np.mean(fs)) if fs else None)))
summary['cells'] = cells

# contrast 1: contract vs paper at none
c_none = [x for x in get(cond='none') if x['correct'] is not None]
kc = sum(x['correct'] for x in c_none if x['arm'] == 'contract'); nc = sum(1 for x in c_none if x['arm'] == 'contract')
kp = sum(x['correct'] for x in c_none if x['arm'] == 'paper'); np_ = sum(1 for x in c_none if x['arm'] == 'paper')
contr = {}
if nc and np_:
    _, pf = fisher_exact([[kc, nc - kc], [kp, np_ - kp]])
    d, lo, hi = newcombe(kc, nc, kp, np_)
    strata = []
    for t in tasks:
        xs = [x for x in c_none if x['task'] == t]
        a = sum(x['correct'] for x in xs if x['arm'] == 'contract'); b = sum(1 for x in xs if x['arm'] == 'contract') - a
        c = sum(x['correct'] for x in xs if x['arm'] == 'paper'); dd = sum(1 for x in xs if x['arm'] == 'paper') - c
        strata.append((a, b, c, dd))
    contr['contract_vs_paper_none'] = dict(contract=f'{kc}/{nc}', paper=f'{kp}/{np_}', risk_diff=d, ci_low=lo, ci_high=hi, fisher_p=float(pf), cmh=mh_test(strata))

def paired(c1, c2):
    key = lambda x: (x['task'], x['arm'], x['rep'])
    A = {key(x): x['correct'] for x in get(cond=c1) if x['correct'] is not None}
    B = {key(x): x['correct'] for x in get(cond=c2) if x['correct'] is not None}
    common = sorted(set(A) & set(B))
    b = sum(1 for k in common if A[k] and not B[k]); c = sum(1 for k in common if B[k] and not A[k])
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    ra = np.mean([A[k] for k in common]) if common else None; rb = np.mean([B[k] for k in common]) if common else None
    return dict(n_pairs=len(common), rate_1=ra, rate_2=rb, c1_right_c2_wrong=b, c2_right_c1_wrong=c, mcnemar_exact_p=p)

contr['checklist_vs_generic'] = paired('checklist', 'generic')
contr['checklist_vs_none'] = paired('checklist', 'none')
contr['generic_vs_none'] = paired('generic', 'none')
# Holm
ps = [('contract_vs_paper_none', contr.get('contract_vs_paper_none', {}).get('fisher_p', 1.0)),
      ('checklist_vs_generic', contr['checklist_vs_generic']['mcnemar_exact_p']),
      ('checklist_vs_none', contr['checklist_vs_none']['mcnemar_exact_p']),
      ('generic_vs_none', contr['generic_vs_none']['mcnemar_exact_p'])]
order = sorted(ps, key=lambda x: x[1]); m = len(order); holm = {}; running = 0.0
for i, (nm, p) in enumerate(order):
    running = max(running, min(1.0, (m - i) * p)); holm[nm] = running
for nm in holm:
    if nm in contr: contr[nm]['holm_p'] = holm[nm]
summary['primary_contrasts'] = contr

# secondary: trap score differences (paired). Main interval: bootstrap over executor runs pooled over
# tasks (as run). The pre-registration asked for resampling stratified by task; that interval is added as
# ci_strat_low/high (runs resampled within each task, 4,000 draws, seed 12).
def paired_mean_diff(field, c1, c2):
    key = lambda x: (x['task'], x['arm'], x['rep'])
    A = {key(x): x[field] for x in get(cond=c1) if x[field] is not None}
    B = {key(x): x[field] for x in get(cond=c2) if x[field] is not None}
    common = sorted(set(A) & set(B))
    if not common: return None
    d = np.array([A[k] - B[k] for k in common])
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    tasks = np.array([k[0] for k in common]); r2 = np.random.default_rng(12)
    groups = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    bs = [np.concatenate([d[g[r2.integers(0, len(g), len(g))]] for g in groups]).mean() for _ in range(4000)]
    return dict(n=len(common), mean_diff=float(d.mean()), ci_low=float(np.percentile(bt, 2.5)), ci_high=float(np.percentile(bt, 97.5)),
                ci_strat_low=float(np.percentile(bs, 2.5)), ci_strat_high=float(np.percentile(bs, 97.5)))
sec = {}
for fld in ['trap', 'false_statements', 'key_numbers']:
    sec[fld] = {f'{a}_vs_{b}': paired_mean_diff(fld, a, b) for a, b in [('checklist', 'generic'), ('checklist', 'none'), ('generic', 'none')]}
    xs = [x for x in get(cond='none') if x[fld] is not None]
    pc = [x[fld] for x in xs if x['arm'] == 'contract']; pp = [x[fld] for x in xs if x['arm'] == 'paper']
    if pc and pp:
        bt = [np.mean(np.random.default_rng(i).choice(pc, len(pc))) - np.mean(np.random.default_rng(10000 + i).choice(pp, len(pp))) for i in range(4000)]
        tc = [x['task'] for x in xs if x['arm'] == 'contract']; tp = [x['task'] for x in xs if x['arm'] == 'paper']
        pc_, pp_, tc_, tp_ = map(np.array, (pc, pp, tc, tp)); r2 = np.random.default_rng(13); bs = []
        for _ in range(4000):
            a = np.concatenate([pc_[np.flatnonzero(tc_ == t)][r2.integers(0, (tc_ == t).sum(), (tc_ == t).sum())] for t in sorted(set(tc))])
            b = np.concatenate([pp_[np.flatnonzero(tp_ == t)][r2.integers(0, (tp_ == t).sum(), (tp_ == t).sum())] for t in sorted(set(tp))])
            bs.append(a.mean() - b.mean())
        sec[fld]['contract_vs_paper_none'] = dict(n=(len(pc), len(pp)), mean_diff=float(np.mean(pc) - np.mean(pp)), ci_low=float(np.percentile(bt, 2.5)), ci_high=float(np.percentile(bt, 97.5)),
                                                  ci_strat_low=float(np.percentile(bs, 2.5)), ci_strat_high=float(np.percentile(bs, 97.5)))
summary['secondary'] = sec

# repairs: transitions from none to each review condition
trans = {}
for cond in ['generic', 'checklist']:
    key = lambda x: (x['task'], x['arm'], x['rep'])
    A = {key(x): x['correct'] for x in get(cond='none') if x['correct'] is not None}
    B = {key(x): x['correct'] for x in get(cond=cond) if x['correct'] is not None}
    common = set(A) & set(B)
    trans[cond] = dict(wrong_to_right=sum(1 for k in common if not A[k] and B[k]), right_to_wrong=sum(1 for k in common if A[k] and not B[k]),
                       stayed_right=sum(1 for k in common if A[k] and B[k]), stayed_wrong=sum(1 for k in common if not A[k] and not B[k]))
summary['repair_transitions'] = trans
# review validity
rv = {}
for cond in ['generic', 'checklist']:
    xs = [x for x in reviews if x['review'] == cond and x['graded']]
    tot = sum(x['n_points'] for x in xs)
    rv[cond] = dict(n_reviews=len(xs), points=tot, mean_points=(tot / len(xs) if xs else None),
                    valid=sum(x['valid'] for x in xs), invalid=sum(x['invalid'] for x in xs), unclear=sum(x['unclear'] for x in xs),
                    share_invalid=(sum(x['invalid'] for x in xs) / tot if tot else None))
summary['review_points'] = rv
# grader agreement on primary
g = [(x['g1'], x['g2']) for x in rows if x['g1'] is not None and x['g2'] is not None]
if g:
    a = np.array([x[0] for x in g], dtype=float); b = np.array([x[1] for x in g], dtype=float)
    po = np.mean(a == b); pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    summary['grader_agreement'] = dict(n=len(g), agreement=float(po), kappa=float((po - pe) / (1 - pe)) if pe < 1 else 1.0)
summary['missing'] = dict(deliverables_missing=sum(1 for x in rows if not x['delivered']), grades_missing=sum(1 for x in rows if x['correct'] is None))
json.dump(summary, open(os.path.join(OUT, 'analysis.json'), 'w'), indent=1, default=float)
print(json.dumps({k: summary[k] for k in ['n_executor_runs', 'primary_contrasts', 'repair_transitions', 'review_points', 'missing'] if k in summary}, indent=1, default=float))
for c in cells:
    print(c['task'], c['arm'], c['cond'], f"{c['correct']}/{c['n']}", None if c['trap_mean'] is None else round(c['trap_mean'], 2))
