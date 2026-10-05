"""H02: Per-gene local intrinsic dimensionality (LID) stratified by TF/target annotation.

Uses Levina-Bickel MLE on cosine k-NN distances.
Tests: (1) do TF vs target genes occupy different-dimensional local regions?
       (2) does LID heterogeneity change across MaxToki layers?
"""
import numpy as np
import pandas as pd
from scipy import stats
import json
import os
import time

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
BIOM = '<DATA_ROOT>/biodyn-nmi-paper'
OUT = f'{BASE}/autoloop/iterations/iter_0002'
os.makedirs(OUT, exist_ok=True)

emb = np.load(f'{BASE}/outputs/phase0/layer_gene_embeddings.npy')
genes = pd.read_csv(f'{BASE}/outputs/phase0/gene_features.csv')
n_layers, n_genes, d_model = emb.shape

# Gene annotations
trrust = pd.read_csv(
    f'{BIOM}/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv',
    sep='\t', header=None, names=['TF', 'target', 'mode', 'pmid']
)
trrust_tfs = set(trrust['TF'].str.upper())
trrust_targets = set(trrust['target'].str.upper())

with open(f'{BIOM}/results/biological_impact/reference_edge_sets/string_ppi_edges.json') as f:
    string_data = json.load(f)

string_genes = set()
for pair in string_data.get('pairs_700', []):
    string_genes.update(g.upper() for g in pair[:2])

gene_symbols = [s.upper() for s in genes['symbol']]

def label_gene(sym):
    is_tf = sym in trrust_tfs
    is_tgt = sym in trrust_targets
    if is_tf and is_tgt:
        return 'TF_and_target'
    elif is_tf:
        return 'TF_only'
    elif is_tgt:
        return 'target_only'
    elif sym in string_genes:
        return 'string_member'
    return 'other'

gene_labels = np.array([label_gene(s) for s in gene_symbols])
print('Label distribution:')
for lbl in ['TF_only', 'TF_and_target', 'target_only', 'string_member', 'other']:
    print(f'  {lbl}: {(gene_labels == lbl).sum()}')


def lid_mle(X, k=20):
    """Levina-Bickel LID MLE using cosine distances.

    Returns LID per gene (nan if degenerate).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n = X / (norms + 1e-10)
    cos_dist = 1.0 - (X_n @ X_n.T).astype(np.float64)
    np.fill_diagonal(cos_dist, np.inf)

    # k nearest neighbors
    sorted_d = np.sort(cos_dist, axis=1)[:, :k]  # (n_genes, k)
    r_k = sorted_d[:, k - 1]                       # kth-NN distance
    r_inner = sorted_d[:, :k - 1] + 1e-12          # 1..(k-1)-th distances

    log_ratios = np.log(r_k[:, None] / r_inner)    # (n, k-1)
    log_ratios = np.clip(log_ratios, 1e-12, None)

    mean_log = log_ratios.mean(axis=1)
    lid = np.where(mean_log > 1e-10, 1.0 / mean_log, np.nan)
    return lid


lid_matrix = np.zeros((n_layers, n_genes), dtype=np.float32)
for layer in range(n_layers):
    t0 = time.time()
    lid = lid_mle(emb[layer], k=20)
    lid_matrix[layer] = lid.astype(np.float32)
    t1 = time.time()
    valid = ~np.isnan(lid)
    print(f'L{layer:02d}: mean={np.nanmean(lid):.2f}, median={np.nanmedian(lid):.2f}, '
          f'IQR=[{np.nanpercentile(lid,25):.1f},{np.nanpercentile(lid,75):.1f}], t={t1-t0:.1f}s')

np.save(f'{OUT}/h02_lid_matrix.npy', lid_matrix)

# Per-layer group summary
summary_rows = []
for layer in range(n_layers):
    lid = lid_matrix[layer]
    for lbl in ['TF_only', 'TF_and_target', 'target_only', 'string_member', 'other']:
        mask = (gene_labels == lbl) & ~np.isnan(lid)
        if mask.sum() < 2:
            continue
        summary_rows.append({
            'layer': layer,
            'group': lbl,
            'n': int(mask.sum()),
            'median_lid': float(np.median(lid[mask])),
            'mean_lid': float(np.mean(lid[mask])),
            'std_lid': float(np.std(lid[mask])),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUT}/h02_lid_summary.csv', index=False)

# Statistical tests: Mann-Whitney between annotation groups
test_rows = []
pairs = [('TF_only', 'target_only'), ('TF_only', 'other'), ('target_only', 'other'),
         ('TF_and_target', 'other'), ('string_member', 'other')]
for layer in range(n_layers):
    lid = lid_matrix[layer].astype(np.float64)
    for g1, g2 in pairs:
        m1 = (gene_labels == g1) & ~np.isnan(lid)
        m2 = (gene_labels == g2) & ~np.isnan(lid)
        if m1.sum() < 3 or m2.sum() < 3:
            continue
        stat, pval = stats.mannwhitneyu(lid[m1], lid[m2], alternative='two-sided')
        test_rows.append({
            'layer': layer,
            'group1': g1, 'group2': g2,
            'n1': int(m1.sum()), 'n2': int(m2.sum()),
            'median1': float(np.median(lid[m1])),
            'median2': float(np.median(lid[m2])),
            'delta_median': float(np.median(lid[m1]) - np.median(lid[m2])),
            'mw_stat': float(stat),
            'pval': float(pval),
        })
tests_df = pd.DataFrame(test_rows)
from statsmodels.stats.multitest import multipletests
if len(tests_df) > 0:
    _, q, _, _ = multipletests(tests_df['pval'], method='fdr_bh')
    tests_df['q_bh'] = q
tests_df.to_csv(f'{OUT}/h02_lid_tests.csv', index=False)

# LID variance across layers per gene (heterogeneity measure)
lid_var = np.nanvar(lid_matrix, axis=0)  # (1500,) - variance across layers
lid_change = lid_matrix[-1] - lid_matrix[0]  # L0→L11 change per gene
gene_summary = pd.DataFrame({
    'symbol': genes['symbol'],
    'group': gene_labels,
    'lid_l0': lid_matrix[0],
    'lid_l11': lid_matrix[11],
    'lid_change_l0_l11': lid_change,
    'lid_variance_across_layers': lid_var,
})
gene_summary.to_csv(f'{OUT}/h02_lid_per_gene.csv', index=False)

print('\nSaved:', f'{OUT}/h02_lid_summary.csv')
print(summary_df.pivot(index='layer', columns='group', values='median_lid').to_string())
