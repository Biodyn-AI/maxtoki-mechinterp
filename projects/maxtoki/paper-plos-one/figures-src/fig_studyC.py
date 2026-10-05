#!/usr/bin/env python3
"""Study C figure: blind re-audits scored against the 68-error reference set.

(A) item-level detection rate per arm (full detection; open marker = full or
partial), with 95% bootstrap intervals over items and the three runs shown
as small points; the in-session audit for comparison.
(B) which reference errors each run detected, rows grouped by severity.
Inputs: maxtoki-framework-eval/studyC/results/{per_arm.csv, item_matrix.csv,
per_run.csv}. Writes source-data/fig_studyC_*.csv.
"""
import os, sys, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, ARM, ARM_LABEL, INK, INK2, MUTED, SEV, SRC
from matplotlib.colors import ListedColormap

R = '<EVAL_ROOT>/studyC/results'
arm = pd.read_csv(os.path.join(R, 'per_arm.csv'))
mat = pd.read_csv(os.path.join(R, 'item_matrix.csv'))
runs = pd.read_csv(os.path.join(R, 'per_run.csv'))
for f in ['per_arm.csv', 'item_matrix.csv', 'per_run.csv']:
    shutil.copy(os.path.join(R, f), os.path.join(SRC, 'fig_studyC_' + f))

fig = plt.figure(figsize=(7.4, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.35)
ax = fig.add_subplot(gs[0])
order = ['original_audit', 'generic', 'checklist', 'deployed']
lab = {'original_audit': ARM_LABEL['original'], **ARM_LABEL}
col = {'original_audit': ARM['original'], **ARM}
for i, a in enumerate(order):
    r = arm[arm.arm == a].iloc[0]
    y = len(order) - 1 - i
    ax.plot([r.ci_low, r.ci_high], [y, y], color=col[a], lw=2, solid_capstyle='round')
    ax.plot(r.detection_rate, y, 'o', ms=8, color=col[a], mec='white', mew=1.5, zorder=3)
    if a != 'original_audit':
        ax.plot(r.detection_rate_incl_partial, y, 'o', ms=8, mfc='white', mec=col[a], mew=1.5, zorder=3)
        rr = runs[runs.arm == a]
        ax.plot(rr['yes'] / 68, [y + 0.22] * len(rr), '|', ms=7, color=col[a], mew=1.2)
    else:
        ax.plot(r.detection_rate_incl_partial, y, 'o', ms=8, mfc='white', mec=col[a], mew=1.5, zorder=3)
    ax.text(1.02, y, f'{r.detection_rate*100:.0f}%', va='center', ha='left', fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[a] for a in order[::-1]])
ax.set_xlim(0, 1); ax.set_xlabel('Share of reference errors detected')
ax.set_title('Detection by review arm'); ax.tick_params(axis='y', length=0)
ax.plot([], [], 'o', color=INK2, label='Full detection (95% CI)'); ax.plot([], [], 'o', mfc='white', mec=INK2, label='Full or partial'); ax.plot([], [], '|', color=INK2, ms=7, label='Single runs')
ax.legend(loc='upper center', bbox_to_anchor=(0.36, -0.17), ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=0.9)
panel_label(ax, 'A')

ax2 = fig.add_subplot(gs[1])
sev_order = {'critical': 0, 'major': 1, 'minor': 2}
mat['s'] = mat.severity.map(sev_order)
runcols = [c for c in mat.columns if c.split('_')[0] in ('deployed', 'checklist', 'generic')]
runcols = sorted(runcols, key=lambda c: (['generic', 'checklist', 'deployed'].index(c.split('_')[0]), c))
code = {'no': 0, 'partial': 1, 'yes': 2}
mat['rate'] = mat[runcols].apply(lambda r: np.mean([code[v] == 2 for v in r]), axis=1)
mat = mat.sort_values(['s', 'rate'], ascending=[True, False])
M = np.array([[code[v] for v in mat[c]] for c in runcols]).T
orig = np.array([2 if v == 'yes' else (1 if v == 'partly' else 0) for v in mat.original_audit])
M = np.column_stack([orig, M])
cmap = ListedColormap(['#f3f2ee', '#9ec5f4', '#1c5cab'])
ax2.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
xl = ['A'] + [f"{c.split('_')[0][0].upper()}{c.split('_')[1]}" for c in runcols]
ax2.set_xticks(range(M.shape[1])); ax2.set_xticklabels(xl, rotation=0, fontsize=8)
for x in [0.5, 3.5, 6.5]:
    ax2.axvline(x, color='white', lw=2)
yb = 0
for s in ['critical', 'major', 'minor']:
    n = int((mat.severity == s).sum())
    ax2.axhline(yb - 0.5, color='white', lw=2)
    ax2.text(-1.0, yb + n / 2 - 0.5, f'{s.capitalize()}\n(n = {n})', ha='right', va='center', fontsize=8, color=INK2)
    yb += n
ax2.set_yticks([])
ax2.set_title('Which reference errors each run detected')
ax2.text(0.5, -0.075, 'A = in-session audit; G = generic review, C = generic checklist,\nD = checklist as deployed; 1-3 = runs', transform=ax2.transAxes, ha='center', va='top', fontsize=8, color=INK2)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color='#1c5cab', label='Detected'), Patch(color='#9ec5f4', label='Partly'), Patch(color='#f3f2ee', label='Not detected')],
           loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8)
for sp in ax2.spines.values(): sp.set_visible(False)
panel_label(ax2, 'B')
save(fig, 'Fig_studyC')
