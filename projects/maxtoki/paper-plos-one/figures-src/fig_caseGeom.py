#!/usr/bin/env python3
"""Case-study figure: representation geometry of MaxToki-217M (corrected analyses).

(A) Input gene-embedding tables: Pearson correlation of MaxToki's gene-gene cosine matrix
    with Geneformer V2-316M's and with scGPT's, per gene panel (95% gene bootstrap interval);
    chance (gene labels shuffled) is about 0.
(B) Effective rank of per-gene mean vectors by layer, trained model vs three random-initialised
    models (same 2,000 Tabula Sapiens immune cells, 1,500 genes; 95% jackknife over 10 cell blocks).
(C) Cross-sample CKA (three disjoint 2,000-cell samples) from layer 1, trained vs random-initialised
    (mean over the three sample pairs; 95% jackknife interval over 30 groups of 50 genes).
(D) Cross-lineage ('global') developmental order of anchor centroids on the external and zero-shot
    panels: MaxToki vs model-free comparators (95% donor cluster bootstrap, 2,000 replicates).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, save, panel_label, SERIES, INK, INK2, MUTED, SRC

RUNS = '<REPO_ROOT>/projects/maxtoki/runs'
XM = pd.read_csv(f'{RUNS}/topology-141-217M/outputs/v2_crossmodel/fig_crossmodel.csv')
SP = f'{RUNS}/spectral-geometry-217M/outputs/v3_spectral/analysis'
_I = json.load(open(f'{SP}/intervals.json')); IV = _I['er_jackknife_cells']; CJ = _I['cka3_jackknife']
CKA = json.load(open(f'{SP}/cka.json'))
MD = f'{RUNS}/manifold-discovery-217M/outputs/v3_devorder'
BOOT = {p: json.load(open(f'{MD}/v2b_boot_{p}_global.json'))['results'] for p in ['external', 'zeroshot']}

fig = plt.figure(figsize=(7.4, 6.4))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.42)

# (A) cross-model
ax = fig.add_subplot(gs[0, 0])
sub = XM[(XM.metric == 'pairwise_cosine_pearson') & (XM.model_key.isin(['gf', 'scgpt_fixed'])) & (XM.panel.isin(['lung', 'immune', 'external_lung']))]
panels = ['lung', 'immune', 'external_lung']; plab = {'lung': 'Lung', 'immune': 'Immune', 'external_lung': 'External\nlung'}
for j, (mk, lab, col) in enumerate([('gf', 'Geneformer V2-316M', SERIES[2]), ('scgpt_fixed', 'scGPT', SERIES[3])]):
    for i, p in enumerate(panels):
        r = sub[(sub.model_key == mk) & (sub.panel == p)].iloc[0]
        x = i + (j - 0.5) * 0.25
        ax.plot([x, x], [r.ci_low, r.ci_high], color=col, lw=2, solid_capstyle='round')
        ax.plot(x, r.value, 'o', color=col, ms=6, mec='white', mew=1, label=lab if i == 0 else None)
ax.axhline(0, color=MUTED, lw=0.8, ls='--')
ax.set_xticks(range(3)); ax.set_xticklabels([plab[p] for p in panels], fontsize=8)
ax.set_ylim(-0.05, 0.5); ax.set_ylabel('Agreement of gene-gene\nsimilarity with MaxToki (r)')
ax.set_title('Input gene embeddings'); ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 0.17))  # empty band between 0 and the lowest interval
panel_label(ax, 'A')
sub.to_csv(os.path.join(SRC, 'fig_caseGeom_crossmodel.csv'), index=False)

# (B) effective rank
ax = fig.add_subplot(gs[0, 1])
rows = []
for key, lab, col, ls in [('trained_A', 'Trained', INK, '-'), ('rand0_A', 'Random init. (3 seeds)', MUTED, '--'),
                          ('rand1_A', None, MUTED, '--'), ('rand2_A', None, MUTED, '--')]:
    L = sorted(int(k) for k in IV[key] if k.isdigit())
    est = np.array([IV[key][str(l)]['est'] for l in L]); lo = np.array([IV[key][str(l)]['ci'][0] for l in L]); hi = np.array([IV[key][str(l)]['ci'][1] for l in L])
    ax.fill_between(L, lo, hi, color=col, alpha=0.2, lw=0)
    ax.plot(L, est, ls, color=col, lw=1.6, marker='o', ms=3, label=lab)
    rows += [dict(model=key, layer=l, er=e, ci_low=a, ci_high=b) for l, e, a, b in zip(L, est, lo, hi)]
ax.set_xlabel('Layer (0 = token embedding)'); ax.set_ylabel('Effective rank')
ax.set_xticks(range(0, 12, 2)); ax.set_ylim(0, 900)
ax.set_title('Effective rank by depth'); ax.legend(frameon=False, fontsize=8, loc='upper right')
panel_label(ax, 'B')
pd.DataFrame(rows).to_csv(os.path.join(SRC, 'fig_caseGeom_effective_rank.csv'), index=False)

# (C) cross-sample CKA from layer 1
ax = fig.add_subplot(gs[1, 0])
rows = []
for key, lab, col in [('v3_trained', 'Trained', INK), ('v3_rand0', 'Random init. (seed 0)', MUTED)]:
    pl = CJ[key]['per_layer']; L = [l for l in sorted(int(k) for k in pl if k.isdigit()) if l >= 1]
    m = np.array([pl[str(l)]['est_mean'] for l in L])
    ci = np.array([pl[str(l)]['gene_jk']['ci'] for l in L], float)   # 95% jackknife over 30 gene groups
    ax.fill_between(L, ci[:, 0], ci[:, 1], color=col, alpha=0.2, lw=0)
    ax.plot(L, m, '-o', color=col, lw=1.6, ms=3, label=lab)
    rows += [dict(model=key, layer=l, cka=v, ci_low=a, ci_high=b) for l, v, (a, b) in zip(L, m, ci)]
ax.set_xlabel('Layer'); ax.set_ylabel('CKA between cell samples')
ax.set_xticks(range(1, 12, 2)); ax.set_title('Stability across cell samples')
ax.legend(frameon=False, fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.6))  # empty area between the two curves
panel_label(ax, 'C')
pd.DataFrame(rows).to_csv(os.path.join(SRC, 'fig_caseGeom_cka.csv'), index=False)

# (D) developmental (global) order
ax = fig.add_subplot(gs[1, 1])
reps = [('maxtoki', 'MaxToki-217M', INK), ('tokenbag_pca64', 'Bag of input tokens', SERIES[0]),
        ('lookup_ct_mean5', 'Cell-type label lookup', SERIES[1]), ('hvg_pca64', 'Expression (HVG)', SERIES[4])]
rows = []
for i, p in enumerate(['external', 'zeroshot']):
    for j, (k, lab, col) in enumerate(reps):
        r = BOOT[p][f'global|{k}']; x = i + (j - 1.5) * 0.17
        ax.plot([x, x], r['ci95'], color=col, lw=2, solid_capstyle='round')
        ax.plot(x, r['observed'], 'o' if k == 'maxtoki' else 's', color=col, ms=5.5, mec='white', mew=1, label=lab if i == 0 else None)
        rows.append(dict(panel=p, representation=k, observed=r['observed'], ci_low=r['ci95'][0], ci_high=r['ci95'][1]))
ax.set_xticks([0, 1]); ax.set_xticklabels(['External panel', 'Zero-shot panel'], fontsize=8)
ax.set_ylabel('Cross-lineage order\n(Spearman)'); ax.set_ylim(0.45, 1.0)
ax.set_title('Cross-lineage developmental order')
ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, columnspacing=0.8, handletextpad=0.3)  # below the panel: too wide to fit inside at 8 pt
panel_label(ax, 'D')
pd.DataFrame(rows).to_csv(os.path.join(SRC, 'fig_caseGeom_devorder.csv'), index=False)
save(fig, 'Fig_caseGeom')
