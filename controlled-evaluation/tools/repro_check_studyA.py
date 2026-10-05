#!/usr/bin/env python3
"""Reproducibility check for Study A deliverables (pre-registered secondary outcome 6).

For every final deliverable (executor output and each repaired output), the
returned analysis_script is run from a temporary folder whose contents are
symlinks to the package (so the package itself is never written to), with the
same interpreter the subjects used, a 20-minute timeout, and BLAS threads
capped at 2. The script must print JSON. Each numeric estimate the
deliverable reported is looked up by name in the printed JSON (any nesting);
it is 'reproduced' if |reported - rerun| <= max(0.005, 1% of |reported|).
Point values only; interval bounds are not compared (they depend on
resampling seeds the scripts may not fix).

Usage: repro_check_studyA.py <out.csv> <workflow_output.json> [...] [--jobs=N] [--timeout=S] [--redo=label,label]
(--redo re-checks the listed deliverables, e.g. into a separate csv after a matching fix)
"""
import json, sys, os, csv, tempfile, subprocess, re, concurrent.futures as cf, math

PY = '<CODE_ROOT>/analysis-tasks/bin/python'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
jobs = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--jobs=')), 3))
TIMEOUT = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--timeout=')), 1200))
SAVE = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--save-stdout=')), None)  # folder for each script's stdout
out_csv, files = args[0], args[1:]
runs = []
for f in files:
    d = json.load(open(f)); runs += [r for r in (d['result'] if 'result' in d else d) if r]

def flatten(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            # {name: {"value": x, ...}} -> (name, x)
            if isinstance(v, dict) and isinstance(v.get('value'), (int, float)) and not isinstance(v.get('value'), bool):
                yield (str(k), v['value'])
            yield from flatten(v, f'{pre}.{k}' if pre else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, dict) and 'name' in v and 'value' in v:
                yield (str(v['name']), v['value'])
            yield from flatten(v, f'{pre}[{i}]')
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield (pre, o)

def norm(s): return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def check(job):
    lab, pkg, deliv = job
    script = deliv['results'].get('analysis_script', '')
    est = [(e['name'], e['value']) for e in deliv['results'].get('estimates', []) if isinstance(e.get('value'), (int, float)) and not isinstance(e.get('value'), bool)]
    row = dict(label=lab, n_estimates=len(est), ran=False, json_ok=False, n_reproduced=0, n_found=0, n_value_found=0, seconds=None, error='')
    if not script.strip():
        row['error'] = 'no script'; return row
    with tempfile.TemporaryDirectory() as td:
        for name in os.listdir(pkg):
            if not name.startswith('._'): os.symlink(os.path.join(pkg, name), os.path.join(td, name))
        sp = os.path.join(td, '_analysis_script.py'); open(sp, 'w').write(script)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', PYTHONWARNINGS='ignore')
        import time; t0 = time.time()
        try:
            p = subprocess.run([PY, sp], cwd=td, capture_output=True, text=True, timeout=TIMEOUT, env=env)
            row['seconds'] = round(time.time() - t0, 1); row['ran'] = p.returncode == 0
            if p.returncode != 0: row['error'] = (p.stderr[-300:].replace('\n', ' ') or f'exit code {p.returncode}, no stderr')
            txt = p.stdout.strip()
            if SAVE:
                os.makedirs(SAVE, exist_ok=True)
                open(os.path.join(SAVE, lab.replace(':', '_') + '.txt'), 'w').write(p.stdout)
            obj = None
            for cand in [txt] + [txt[txt.find('{'):]] + [txt[txt.find('['):]]:
                try: obj = json.loads(cand); break
                except Exception: pass
            if obj is None:
                m = re.findall(r'\{.*\}', txt, re.S)
                for c in m[::-1]:
                    try: obj = json.loads(c); break
                    except Exception: pass
            if obj is not None:
                row['json_ok'] = True
                flat = {}
                for k, v in flatten(obj):
                    if isinstance(v, (int, float)) and not isinstance(v, bool): flat.setdefault(norm(k.split('.')[-1]), []).append(v); flat.setdefault(norm(k), []).append(v)
                allvals = [v for vs in flat.values() for v in vs if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
                for name, val in est:
                    cands = flat.get(norm(name), [])
                    if cands:
                        row['n_found'] += 1
                        if any(abs(c - val) <= max(0.005, 0.01 * abs(val)) for c in cands if isinstance(c, (int, float)) and not (isinstance(c, float) and math.isnan(c))):
                            row['n_reproduced'] += 1
                    elif any(abs(c - val) <= max(0.005, 0.01 * abs(val)) for c in allvals):
                        # name differs: count the value as found if any printed number matches it
                        row['n_value_found'] += 1
        except subprocess.TimeoutExpired:
            row['error'] = f'timeout {TIMEOUT} s'
    return row

jobs_list = []
for r in runs:
    base = f"{r['task']}:{r['arm']}:{r['rep']}"
    for cond, dl in [('none', r.get('exec')), ('generic', r.get('repaired_generic')), ('checklist', r.get('repaired_checklist'))]:
        if dl: jobs_list.append((f'{base}:{cond}', r['pkg'], dl))
done = set()
if os.path.exists(out_csv):
    done = {x['label'] for x in csv.DictReader(open(out_csv))}
redo = next((a.split('=', 1)[1].split(',') for a in sys.argv if a.startswith('--redo=')), None)
todo = [j for j in jobs_list if j[0] in redo] if redo else [j for j in jobs_list if j[0] not in done]
print(f'{len(jobs_list)} deliverables, {len(todo)} to check', flush=True)
fields = ['label', 'n_estimates', 'ran', 'json_ok', 'n_reproduced', 'n_found', 'n_value_found', 'seconds', 'error']
new = not os.path.exists(out_csv)
with open(out_csv, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    if new: w.writeheader()
    with cf.ThreadPoolExecutor(jobs) as ex:
        for row in ex.map(check, todo):
            w.writerow(row); fh.flush(); print(row['label'], row['ran'], row['n_reproduced'], '/', row['n_estimates'], row['error'][:80], flush=True)
