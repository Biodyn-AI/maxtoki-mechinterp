#!/usr/bin/env python3
"""Study A figure: controlled execution study (Claude Opus 5.5 subjects).

(A) Correct verdicts out of 5 executor runs, per task, executor arm and
review condition. (B) Share of the key's pitfalls handled (graders' mean),
per review condition and executor arm; mean with a 95% bootstrap interval
over executor runs. (C) Review points judged valid or invalid by a grader
with the key.
Inputs: maxtoki-framework-eval/studyA/results/{deliverables.csv, reviews.csv}.
"""
import os, sys, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, ARM, ARM_LABEL, INK, INK2, MUTED, SRC
EXC = {'paper': INK2, 'contract': '#a8a79f'}; MK = {'paper': 'o', 'contract': 's'}
from matplotlib.colors import ListedColormap, BoundaryNorm

R = '<EVAL_ROOT>/studyA/results'
dl = pd.read_csv(os.path.join(R, 'deliverables.csv')); rv = pd.read_csv(os.path.join(R, 'reviews.csv'))
for f in ['deliverables.csv', 'reviews.csv']: shutil.copy(os.path.join(R, f), os.path.join(SRC, 'fig_studyA_' + f))
rng = np.random.default_rng(5)
COND = ['none', 'generic', 'checklist']; CLAB = {'none': 'No review', 'generic': 'Generic\nreview', 'checklist': 'Generic\nchecklist'}
TASKS = ['T1', 'T2', 'T3', 'T4']

fig = plt.figure(figsize=(7.4, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.75, wspace=0.45)
ax = fig.add_subplot(gs[0, :])
cols = [(a, c) for a in ['paper', 'contract'] for c in COND]
M = np.array([[dl[(dl.task == t) & (dl.arm == a) & (dl.cond == c)].correct.sum() for (a, c) in cols] for t in TASKS], float)
cmap = ListedColormap(['#fbe3dc', '#f6c2b0', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95'])
ax.imshow(M, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5], 6), aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{int(M[i, j])}/5', ha='center', va='center', fontsize=8, color='white' if M[i, j] >= 4 else INK)
ax.set_yticks(range(4)); ax.set_yticklabels(['T1 attention vs TRRUST', 'T2 CRISPRi direction', 'T3 TF specificity', 'T4 cross-model alignment'])
ax.set_xticks(range(6)); ax.set_xticklabels([CLAB[c] for (_, c) in cols], fontsize=8)
ax.axvline(2.5, color='white', lw=3)
ax.text(1, -0.75, 'Paper arm (brief + source paper)', ha='center', fontsize=8, color=INK2)
ax.text(4, -0.75, 'Contract arm (+ specification)', ha='center', fontsize=8, color=INK2)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
ax.set_title('Correct verdicts (of 5 executor runs)', pad=30)
panel_label(ax, 'A')

ax = fig.add_subplot(gs[1, 0])
for k, a in enumerate(['paper', 'contract']):
    for j, c in enumerate(COND):
        v = dl[(dl.arm == a) & (dl.cond == c)].trap.dropna().values
        bt = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        x = j + (k - 0.5) * 0.25
        ax.plot([x, x], np.percentile(bt, [2.5, 97.5]), color=EXC[a], lw=2, solid_capstyle='round')
        ax.plot(x, v.mean(), MK[a], color=EXC[a], ms=7, mec='white', mew=1.2, label=('Paper arm' if a == 'paper' else 'Contract arm') if j == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([CLAB[c] for c in COND], fontsize=8)
ax.set_ylim(0.8, 1.01); ax.set_ylabel('Share of pitfalls handled')
ax.set_title('Pitfall handling'); ax.legend(frameon=False, fontsize=8, loc='lower right')
panel_label(ax, 'B')

ax = fig.add_subplot(gs[1, 1])
y = [1, 0]
for yy, c in zip(y, ['generic', 'checklist']):
    s = rv[rv.review == c]
    v, iv = s.valid.sum(), s.invalid.sum() + s.unclear.sum()
    ax.barh(yy, v, color=ARM[c], height=0.55, edgecolor='white')
    ax.barh(yy, iv, left=v, color='#e1e0d9', height=0.55, edgecolor='white')
    ax.text(v / 2, yy, f'{v}', ha='center', va='center', color='white', fontsize=8)
    ax.text(v + iv / 2, yy, f'{iv}', ha='center', va='center', color=INK, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([ARM_LABEL['generic'], ARM_LABEL['checklist']])
ax.set_xlabel('Review points (40 reviews each)'); ax.tick_params(axis='y', length=0)
ax.set_title('Review points: valid | invalid or unclear', fontsize=9)
panel_label(ax, 'C')
save(fig, 'Fig_studyA')
