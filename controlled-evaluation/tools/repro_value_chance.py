#!/usr/bin/env python3
"""Chance rate for value matching in the Study A reproducibility check.

For each re-checked deliverable whose script output was saved (studyA/results/repro_stdout/),
compare the share of its own reported estimates found among the printed numbers (within
max(0.005, 1% of |value|)) with the share found for estimates drawn from deliverables of a
DIFFERENT task (same count, 50 draws). Writes studyA/results/repro_value_chance.json.
"""
import json, os, re, math, random, sys
sys.argv = [sys.argv[0]]
import importlib.util
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(WS, 'studyA', 'results')
spec = importlib.util.spec_from_file_location('rc', os.path.join(WS, 'tools', 'repro_check_studyA.py'))
src = open(spec.origin).read()
flatten_src = src[src.index('def flatten'):src.index('def norm')]
ns = {}; exec(flatten_src, ns); flatten = ns['flatten']

runs = []
for f in ['workflow_output_batch1_T1T3T4.json', 'workflow_output_batch2_T2.json']:
    d = json.load(open(os.path.join(R, f)))
    runs += [r for r in d['result'] if r]
est = {}
for r in runs:
    for cond, dl in [('none', r.get('exec')), ('generic', r.get('repaired_generic')), ('checklist', r.get('repaired_checklist'))]:
        if dl:
            est[f"{r['task']}:{r['arm']}:{r['rep']}:{cond}"] = [e['value'] for e in dl['results'].get('estimates', []) if isinstance(e.get('value'), (int, float)) and not isinstance(e.get('value'), bool)]

def numbers(txt):
    obj = None
    for cand in [txt, txt[txt.find('{'):] if '{' in txt else '']:
        try: obj = json.loads(cand); break
        except Exception: pass
    if obj is None:
        for c in re.findall(r'\{.*\}', txt, re.S)[::-1]:
            try: obj = json.loads(c); break
            except Exception: pass
    if obj is None: return []
    return [v for _, v in flatten(obj) if isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))]

def share(vals, nums):
    if not vals: return float('nan')
    return sum(any(abs(c - v) <= max(0.005, 0.01 * abs(v)) for c in nums) for v in vals) / len(vals)

rng = random.Random(7); out = {}
for f in sorted(x for x in os.listdir(os.path.join(R, 'repro_stdout')) if not x.startswith('._')):
    lab = f[:-4].replace('_', ':')
    nums = numbers(open(os.path.join(R, 'repro_stdout', f)).read())
    if not nums or lab not in est: continue
    own = share(est[lab], nums)
    pool = [v for k, vs in est.items() if k[:2] != lab[:2] for v in vs]
    ch = [share(rng.sample(pool, len(est[lab])), nums) for _ in range(50)]
    out[lab] = dict(n=len(est[lab]), n_printed=len(nums), own=own, chance_mean=sum(ch) / len(ch), chance_max=max(ch))
json.dump(out, open(os.path.join(R, 'repro_value_chance.json'), 'w'), indent=1)
import statistics as st
print(f'{len(out)} deliverables; own share median {st.median(v["own"] for v in out.values()):.3f}; chance share median {st.median(v["chance_mean"] for v in out.values()):.3f}, max over deliverables {max(v["chance_mean"] for v in out.values()):.3f}')
for k, v in out.items():
    if k.startswith('T3'): print(k, round(v['own'], 3), round(v['chance_mean'], 3), v['n_printed'])
