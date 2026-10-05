#!/usr/bin/env python3
"""Study B figure: held-out benchmark.

(A) Clean items: share of the three reviews per arm that claimed at least one
load-bearing (critical or major) error that the item's key does not count as
an error. (B) Post-hoc classification of the 82 load-bearing claims that the
graders marked as false alarms (clean and flawed items), by arm; the 23 further
claims that the verifier ruled false alarms after a grader dispute were not
classified. Every review detected the documented error
on every flawed item (both arms, 33/33), so detection is stated in the caption.
Inputs: maxtoki-framework-eval/studyB/results/{items.csv,
false_alarms_posthoc_classified.json}. Writes source-data/fig_studyB_*.csv.
"""
import os, sys, json, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, ARM, ARM_LABEL, INK2, SRC, SERIES

R = '<EVAL_ROOT>/studyB/results'
items = pd.read_csv(os.path.join(R, 'items.csv'))
fa = pd.DataFrame(json.load(open(os.path.join(R, 'false_alarms_posthoc_classified.json'))))
items.to_csv(os.path.join(SRC, 'fig_studyB_items.csv'), index=False)
fa[['uid', 'item', 'status', 'arm', 'rep', 'posthoc']].to_csv(os.path.join(SRC, 'fig_studyB_false_alarms.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1.1, 1.0], wspace=0.55))
ax = axes[0]
cl = items[items.status == 'clean'].copy()
cl['mean'] = (cl.checklist_fa_any + cl.generic_fa_any) / 2
cl = cl.sort_values('mean')
y = np.arange(len(cl))
for arm, dy in [('generic', -0.13), ('checklist', 0.13)]:
    ax.scatter(cl[f'{arm}_fa_any'] * 3, y + dy, s=36, color=ARM[arm], edgecolor='white', linewidth=0.8, label=ARM_LABEL[arm], zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f'Item {i + 1}' for i in range(len(cl))], fontsize=8)
ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-0.3, 3.3)
ax.set_xlabel('Reviews (of 3) raising a load-bearing\nfalse alarm on a correct analysis')
ax.set_title('Correct (clean) analyses'); ax.tick_params(axis='y', length=0)
for v in [0, 1, 2, 3]: ax.axvline(v, color='#eeede8', lw=0.8, zorder=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.27), ncol=2, frameon=False, fontsize=8)
panel_label(ax, 'A')

ax = axes[1]
CATS = ['factually_wrong', 'real_problem_key_missed', 'right_but_acknowledged', 'right_but_no_effect_on_conclusions']
LAB = {'factually_wrong': 'Factually wrong', 'real_problem_key_missed': 'Real problem the key missed',
       'right_but_acknowledged': 'Correct, already stated\nin the summary', 'right_but_no_effect_on_conclusions': 'Correct, but no effect\non the conclusions'}
COL = {'factually_wrong': '#e34948', 'real_problem_key_missed': '#eda100', 'right_but_acknowledged': '#86b6ef', 'right_but_no_effect_on_conclusions': '#2a78d6'}
arms = ['generic', 'checklist']
left = np.zeros(2)
for c in CATS:
    v = np.array([((fa.arm == a) & (fa.posthoc == c)).sum() for a in arms], float)
    ax.barh([1, 0], v, left=left, height=0.55, color=COL[c], edgecolor='white', linewidth=1, label=LAB[c])
    left += v
for yy, t in zip([1, 0], left): ax.text(t + 0.8, yy, f'{int(t)}', va='center', fontsize=8, color=INK2)
ax.set_yticks([1, 0]); ax.set_yticklabels([ARM_LABEL[a] for a in arms])
ax.set_xlabel('False alarms marked by graders (all items)'); ax.set_xlim(0, max(left) * 1.15)
ax.set_title('What the false alarms were'); ax.tick_params(axis='y', length=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.28), ncol=2, frameon=False, fontsize=8)
panel_label(ax, 'B')
save(fig, 'Fig_studyB')
