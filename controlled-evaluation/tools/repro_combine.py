#!/usr/bin/env python3
"""Combine the Study A reproducibility passes into studyA/results/repro_final.csv and a summary.

Pass 1: repro_check.csv (all 120, 3 parallel jobs, 20-min limit, name matching only).
Pass 2: repro_check_redo.csv (the 30 not fully reproduced in pass 1: 5 timeouts, 25 with names
        that did not match; 2 jobs, 60-min limit; name matching fixed for {name: {value}}
        output; values matched when names differ).
Pass 3: repro_check_redo2.csv (2 scripts that were killed without output in pass 2, run alone).
The last pass for each deliverable is used. Value matching is reported but not used as evidence
of reproduction, because its chance rate is high (repro_value_chance.json).
"""
import pandas as pd, json, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'studyA', 'results')
p1 = pd.read_csv(os.path.join(R, 'repro_check.csv')).assign(n_value_found=0, final_pass=1)
p2 = pd.read_csv(os.path.join(R, 'repro_check_redo.csv')).assign(final_pass=2)
p3 = pd.read_csv(os.path.join(R, 'repro_check_redo2.csv')).assign(final_pass=3)
d = pd.concat([p1, p2, p3]).drop_duplicates('label', keep='last').sort_values('label').reset_index(drop=True)
d['task'] = d.label.str[:2]
d['all_named_agree'] = d.ran & (d.n_reproduced == d.n_found)
d.to_csv(os.path.join(R, 'repro_final.csv'), index=False)
g = d.groupby('task').agg(deliverables=('label', 'size'), ran=('ran', 'sum'), estimates=('n_estimates', 'sum'),
                          matched_by_name=('n_found', 'sum'), reproduced=('n_reproduced', 'sum'),
                          value_only=('n_value_found', 'sum'),
                          fully_by_name=('n_found', lambda s: int((s == d.loc[s.index, 'n_estimates']).sum())))
print(g); print('total', g.sum(numeric_only=True).to_dict())
ch = json.load(open(os.path.join(R, 'repro_value_chance.json')))
t3 = [v['chance_mean'] for k, v in ch.items() if k.startswith('T3')]
summ = dict(per_task=g.reset_index().to_dict(orient='records'), total=g.sum(numeric_only=True).to_dict(),
            any_named_disagreement=int((d.n_found > d.n_reproduced).sum()),
            t3_value_match_chance_range=[min(t3), max(t3)], passes={'1': int((d.final_pass == 1).sum()), '2': int((d.final_pass == 2).sum()), '3': int((d.final_pass == 3).sum())})
json.dump(summ, open(os.path.join(R, 'repro_summary.json'), 'w'), indent=1, default=int)
print(json.dumps({k: v for k, v in summ.items() if k != 'per_task'}, default=int))
