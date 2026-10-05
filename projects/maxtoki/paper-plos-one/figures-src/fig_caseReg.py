#!/usr/bin/env python3
"""Case-study figure: regulatory probes of MaxToki-217M (corrected analyses).

(A) Knockdown response, RPE1 and K562: mean
    per-knockdown AUROC with 95% bootstrap interval over knockdowns.
(B) TRRUST pairs, same runs: pooled AUROC with 95% bootstrap interval over
    transcription factors.
(C) GATA1: power of the count-matched test to detect planted ChIP-seq targets
    among the top 20 genes of one responding feature (200 simulations each;
    Wilson 95% interval).
(D) Direction of CRISPRi knockdown responses on identical records: the
    circuit's accuracy and balanced accuracy against a constant rule and a
    no-model rule (95% interval over silenced genes where available).
"""
import os, sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, SERIES, INK, INK2, MUTED, GRID, SRC

RUNS = '<REPO_ROOT>/projects/maxtoki/runs'
ATT = pd.read_csv(f'{RUNS}/attention-grn-217M/outputs/v2b_attention/fig_attention_variants.csv')
ATT = ATT[ATT.run == '217M_RPE1'].assign(cells='RPE1')
K562_V3 = f'{RUNS}/attention-grn-217M/outputs/v3_attention_k562/fig_attention_variants_v3_k562.csv'
if os.path.exists(K562_V3):
    k = pd.read_csv(K562_V3)
    ATT = pd.concat([ATT, k.assign(cells='K562')], ignore_index=True)
ATT.to_csv(os.path.join(SRC, 'fig_caseReg_attention.csv'), index=False)

POW = pd.read_csv(f'{RUNS}/sae-atlas-217M/outputs/v3_tf_specificity/power.csv')
POW = POW[(POW.tf == 'GATA1') & (POW.db == 'ChIP') & (POW.cutoff == 0.5)].sort_values('k')
POW.to_csv(os.path.join(SRC, 'fig_caseReg_gata1_power.csv'), index=False)
CRI = pd.read_csv(f'{RUNS}/circuit-tracing-217M/outputs/v3_circuit/crispri/comparison_table.csv').iloc[0]

CELLS = list(dict.fromkeys(ATT.cells))
MEASURE = {'knockdown': 'mean per-perturbation AUROC', 'trrust': 'pooled AUROC'}
CCOL = {'RPE1': SERIES[0], 'K562': SERIES[1]}
fig = plt.figure(figsize=(7.4, 6.7))
fig.subplots_adjust(top=0.88)
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=1.05)


def dotplot(ax, endpoint, scores, labels, title, xlabel):
    y = np.arange(len(scores))[::-1]
    for j, c in enumerate(CELLS):
        sub = ATT[(ATT.endpoint == endpoint) & (ATT.cells == c) & (ATT.measure == MEASURE[endpoint])].set_index('score')
        dy = -(j - (len(CELLS) - 1) / 2) * 0.22   # first series on top, matching the legend order
        for yy, s in zip(y, scores):
            if s not in sub.index:
                continue
            r = sub.loc[s]
            ax.plot([r.ci_low, r.ci_high], [yy + dy] * 2, color=CCOL[c], lw=2, solid_capstyle='round')
            ax.plot(r.value, yy + dy, 'o', color=CCOL[c], ms=6, mec='white', mew=1, label=c if yy == y[0] else None)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls='--')
    ax.set_xlim(0.45, 0.82)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.tick_params(axis='y', length=0)


ax = fig.add_subplot(gs[0, 0])
dotplot(ax, 'knockdown', ['forward', 'sym_mean', 'order_share_F', 'coexpr_abs_spearman', 'gene_variance'],
        ['Attention, regulator to target', 'Attention, both directions', 'Order share (no model)',
         'Co-expression (no model)', 'Target variance (no model)'],
        'Knockdown response', 'Mean AUROC over knockdowns')
panel_label(ax, 'A')

ax = fig.add_subplot(gs[0, 1])
dotplot(ax, 'trrust', ['forward', 'sym_mean', 'coexpr_abs_spearman', 'gene_variance', 'gene_model_3feat'],
        ['Attention, regulator to target', 'Attention, both directions', 'Co-expression (no model)',
         'Target variance (no model)', 'Target-gene model (no model)'],
        'TRRUST pairs', 'Pooled AUROC')
panel_label(ax, 'B')
if len(CELLS) > 1:
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(CELLS), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0),
               title='Cells', title_fontsize=8)

ax = fig.add_subplot(gs[1, 0])
n = 200
def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
lo, hi = zip(*[wilson(p, n) for p in POW.power_cm_uncapped])
ax.fill_between(POW.k, lo, hi, color=INK2, alpha=0.15, lw=0)
ax.plot(POW.k, POW.power_cm_uncapped, '-o', color=INK, ms=5, mec='white', mew=1)
ax.set_xticks(range(1, 9)); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel('GATA1 ChIP-seq targets planted\namong a feature\'s top 20 genes')
ax.set_ylabel('Share of simulations detected')
ax.set_title('GATA1 (K562): power of the test')
panel_label(ax, 'C')

ax = fig.add_subplot(gs[1, 1])
ci_acc = json.loads(CRI['CI accuracy (genes)'])
bal_ci = json.loads(CRI['CI balanced acc - 0.5 (genes)'])
tb_ci = json.loads(CRI['CI acc - target baseline (genes)'])
rows = [('Circuit, accuracy', CRI['accuracy'], ci_acc, INK),
        ('Circuit, balanced accuracy', CRI['balanced accuracy'], [0.5 + bal_ci[0], 0.5 + bal_ci[1]], INK),
        ('Always "down" (constant)', CRI['always decrease'], None, MUTED),
        ('Usual direction of target\n(no model)', CRI['target-direction baseline (cross-fitted)'], None, MUTED)]
y = np.arange(len(rows))[::-1]
for yy, (lab, v, ci, col) in zip(y, rows):
    if ci is not None:
        ax.plot(ci, [yy, yy], color=col, lw=2, solid_capstyle='round')
    ax.plot(v, yy, 'o' if col != MUTED else 's', color=col, ms=6, mec='white', mew=1)
    ax.text((ci[1] if ci is not None else v) + 0.004, yy, f'{100 * v:.1f}%', fontsize=8, color=INK2, va='center')
ax.axvline(0.5, color=MUTED, lw=0.8, ls='--')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.tick_params(axis='y', length=0)
ax.set_xlim(0.48, 0.62); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel('Share of records predicted correctly')
ax.set_title('Direction of responses (K562)')
panel_label(ax, 'D')
pd.DataFrame([dict(label=r[0].replace('\n', ' '), value=r[1], ci=r[2]) for r in rows]).to_csv(os.path.join(SRC, 'fig_caseReg_crispri.csv'), index=False)
save(fig, 'Fig_caseReg')
