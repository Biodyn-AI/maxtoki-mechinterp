#!/usr/bin/env python3
"""Error-ledger figure: errors in the deployment's run outputs, by type and
severity (A) and by who or what first found them (B).

Input: projects/maxtoki/framework-eval/error_ledger.csv (rows with
layer == 'run_outputs'). Writes source-data/fig_ledger.csv and the figure.
"""
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, SEV, INK, INK2, MUTED, SRC

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'framework-eval', 'error_ledger.csv')
L = pd.read_csv(LEDGER)
L = L[L.layer == 'run_outputs'].copy()

TYPES = ['code bug', 'wrong statistic', 'missing baseline or null', 'unsupported claim', 'wrong description', 'reproducibility']
TYPE_LABEL = {'code bug': 'Code bug', 'wrong statistic': 'Wrong statistic', 'missing baseline or null': 'Missing baseline or null',
              'unsupported claim': 'Unsupported claim', 'wrong description': 'Wrong description', 'reproducibility': 'Reproducibility'}
# one primary type per error (column primary_type; the same as the 'type' column of S1 Table)
assert L.primary_type.isin(TYPES).all(), 'every run_outputs row needs one of the six primary types'
L['type'] = L.primary_type

def norm_finder(s):
    s = str(s).lower()
    if 'audit sweep' in s: return 'In-session audit'
    if 'repair-round' in s or 'executor agent' in s: return 'Deployment agent, outside the audit sweep'
    if 'blind re-audit' in s: return 'Blind re-audit (Study C)'
    if 'human' in s: return 'Human reader'
    return 'Independent verification'
L['finder'] = L.first_found_by.map(norm_finder)
FINDERS = ['In-session audit', 'Deployment agent, outside the audit sweep', 'Independent verification', 'Blind re-audit (Study C)', 'Human reader']

L[['id', 'short_name', 'pipeline', 'type', 'severity', 'finder', 'audit_pattern', 'present_in_pre_audit_state']].to_csv(os.path.join(SRC, 'fig_ledger.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.95))
for ax, cats, col, labels, title in [
        (axes[0], TYPES, 'type', [TYPE_LABEL[t] for t in TYPES], 'Errors by type'),
        (axes[1], FINDERS, 'finder', FINDERS, 'Who or what found them first')]:
    y = np.arange(len(cats))[::-1]
    left = np.zeros(len(cats))
    for sev in ['critical', 'major', 'minor']:
        v = np.array([((L[col] == c) & (L.severity == sev)).sum() for c in cats], float)
        ax.barh(y, v, left=left, height=0.62, color=SEV[sev], edgecolor='white', linewidth=1.0, label=sev.capitalize())
        left += v
    for yi, tot in zip(y, left):
        ax.text(tot + 1, yi, f'{int(tot)}', va='center', ha='left', fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Number of errors'); ax.set_title(title)
    ax.set_xlim(0, max(left) * 1.18)
    ax.tick_params(axis='y', length=0)
panel_label(axes[0], 'A'); panel_label(axes[1], 'B')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), title=None)
save(fig, 'Fig_ledger')
print(L.severity.value_counts().to_dict(), L.finder.value_counts().to_dict())
