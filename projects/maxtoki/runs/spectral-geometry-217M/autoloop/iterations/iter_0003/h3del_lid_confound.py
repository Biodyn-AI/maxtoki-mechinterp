"""H3-DEL: Hub degree vs LID + LID confound regression + Geneformer static LID.

Sub-D: STRING-700 hub degree vs per-gene LID (does the TF/target LID gap
       survive controlling for PPI degree?).
Sub-E: OLS confound regression: LID_L6 ~ TF_group + expression + dropout.
Sub-L: Geneformer V2-316M static embedding LID — does TF < target hold there too?
"""
import json
import time
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
BIOM = '<DATA_ROOT>'
OUT  = f'{BASE}/autoloop/iterations/iter_0003'
ITER2 = f'{BASE}/autoloop/iterations/iter_0002'

print("=" * 70)
print("H3-DEL: Hub degree vs LID + confound regression + Geneformer static LID")
print("=" * 70)
t_start = time.time()

# ── Load data ─────────────────────────────────────────────────────────────────
lid_matrix = np.load(f'{ITER2}/h02_lid_matrix.npy')  # (12, 1500)
lid_per_gene = pd.read_csv(f'{ITER2}/h02_lid_per_gene.csv')  # symbol, group, lid_l0, lid_l11, ...
gene_features = pd.read_csv(f'{BASE}/outputs/phase0/gene_features.csv')  # mean_expr, dropout_rate

# Align
n_layers, n_genes = lid_matrix.shape
gene_symbols = [s.upper() for s in gene_features['symbol']]
sym_to_idx   = {s: i for i, s in enumerate(gene_symbols)}

# Gene groups from iter_0002 (same order as gene_features)
groups = lid_per_gene['group'].values  # aligned with gene_features rows

# STRING degree per gene
with open(f'{BIOM}/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json') as f:
    string_data = json.load(f)

degree = np.zeros(n_genes, dtype=int)
for a, b in string_data.get('pairs_700', []):
    i, j = sym_to_idx.get(a.upper()), sym_to_idx.get(b.upper())
    if i is not None:
        degree[i] += 1
    if j is not None:
        degree[j] += 1

print(f"Loaded: lid_matrix={lid_matrix.shape}, n_genes={n_genes}")
print(f"STRING-700 degree: max={degree.max()}, mean={degree.mean():.1f}, zeros={(degree==0).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# Sub-D: Hub degree vs LID
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Sub-D: STRING degree vs LID ---")

lid_l0  = lid_matrix[0]
lid_l11 = lid_matrix[11]
lid_l6  = lid_matrix[6]

rho_d_l0,  p_d_l0  = stats.spearmanr(degree, lid_l0)
rho_d_l11, p_d_l11 = stats.spearmanr(degree, lid_l11)
rho_d_l6,  p_d_l6  = stats.spearmanr(degree, lid_l6)
print(f"Spearman(degree, LID_L0):  rho={rho_d_l0:.4f}, p={p_d_l0:.4f}")
print(f"Spearman(degree, LID_L6):  rho={rho_d_l6:.4f}, p={p_d_l6:.4f}")
print(f"Spearman(degree, LID_L11): rho={rho_d_l11:.4f}, p={p_d_l11:.4f}")

# Partial Spearman: TF/target LID gap controlling for degree
# Compute residuals of LID on degree, then test group difference
from scipy.stats import rankdata

def partial_spearman_residual(y, covariate, groups_mask_1, groups_mask_2):
    """Residualise y on covariate via rank-rank regression, then MW test on residuals."""
    ry = rankdata(y)
    rc = rankdata(covariate)
    slope, intercept, _, _, _ = stats.linregress(rc, ry)
    residuals = ry - (slope * rc + intercept)
    g1 = residuals[groups_mask_1]
    g2 = residuals[groups_mask_2]
    stat, pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return float(np.median(g1) - np.median(g2)), stat, pval, g1, g2

tf_only = groups == 'TF_only'
tgt_only = groups == 'target_only'

# Unconditional gap at L6
raw_stat, raw_p = stats.mannwhitneyu(lid_l6[tf_only], lid_l6[tgt_only], alternative='two-sided')
raw_delta = float(np.median(lid_l6[tf_only]) - np.median(lid_l6[tgt_only]))
print(f"\nRaw TF vs target LID_L6: delta={raw_delta:.3f}, MW_p={raw_p:.4f}")

# After controlling for degree
delta_deg, stat_deg, p_deg, _, _ = partial_spearman_residual(lid_l6, degree, tf_only, tgt_only)
print(f"Controlling for degree:   delta_resid={delta_deg:.3f}, MW_p={p_deg:.4f}")

# Save Sub-D
sub_d_df = pd.DataFrame({
    'symbol':   gene_features['symbol'],
    'group':    groups,
    'degree':   degree,
    'lid_l0':   lid_l0,
    'lid_l6':   lid_l6,
    'lid_l11':  lid_l11,
})
sub_d_df.to_csv(f'{OUT}/h3del_degree_lid.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Sub-E: LID confound regression (OLS)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Sub-E: Confound regression ---")

try:
    import statsmodels.api as sm
    _has_sm = True
except ImportError:
    _has_sm = False
    print("  statsmodels not available — using numpy OLS")

mean_expr   = gene_features['mean_expr'].values.astype(float)
dropout_rate = gene_features['dropout_rate'].values.astype(float)
log_expr    = np.log1p(mean_expr)

# Spearman of expression/dropout with LID at L0, L6, L11
for name, layer_i in [('L0', 0), ('L6', 6), ('L11', 11)]:
    lid_l = lid_matrix[layer_i]
    r_expr,    p_expr    = stats.spearmanr(log_expr, lid_l)
    r_drop,    p_drop    = stats.spearmanr(dropout_rate, lid_l)
    print(f"  LID_{name}: Spearman(log_expr)={r_expr:.4f}(p={p_expr:.4f}), "
          f"Spearman(dropout)={r_drop:.4f}(p={p_drop:.4f})")

# OLS: LID_L6 ~ TF_only + target_only + log_expr + dropout
is_tf_only  = (groups == 'TF_only').astype(float)
is_tgt_only = (groups == 'target_only').astype(float)

X_ols = np.column_stack([
    np.ones(n_genes),  # intercept
    is_tf_only,
    is_tgt_only,
    log_expr,
    dropout_rate,
])
y_ols = lid_l6.astype(float)

if _has_sm:
    model = sm.OLS(y_ols, X_ols).fit()
    coefs     = model.params
    ses       = model.bse
    tstats    = model.tvalues
    pvals     = model.pvalues
    pred_names = ['intercept', 'TF_only', 'target_only', 'log_expr', 'dropout_rate']
    print("\n  OLS LID_L6 ~ TF_only + target_only + log_expr + dropout:")
    for name, c, se, t, p in zip(pred_names, coefs, ses, tstats, pvals):
        print(f"    {name:15s}: coef={c:.4f}, SE={se:.4f}, t={t:.3f}, p={p:.4f}")
    reg_df = pd.DataFrame({
        'predictor': pred_names,
        'coef': coefs, 'se': ses, 't': tstats, 'pval': pvals,
    })
else:
    # Manual OLS via numpy
    coeffs, res, rank, sv = np.linalg.lstsq(X_ols, y_ols, rcond=None)
    n, k = X_ols.shape
    if len(res) > 0:
        sse = float(res[0])
    else:
        sse = float(np.sum((y_ols - X_ols @ coeffs)**2))
    sigma2 = sse / (n - k)
    XTX_inv = np.linalg.pinv(X_ols.T @ X_ols)
    ses     = np.sqrt(np.diag(XTX_inv) * sigma2)
    tstats  = coeffs / (ses + 1e-12)
    pvals   = 2 * (1 - stats.t.cdf(np.abs(tstats), df=n-k))
    pred_names = ['intercept', 'TF_only', 'target_only', 'log_expr', 'dropout_rate']
    print("\n  OLS LID_L6 ~ TF_only + target_only + log_expr + dropout:")
    for name, c, se, t, p in zip(pred_names, coeffs, ses, tstats, pvals):
        print(f"    {name:15s}: coef={c:.4f}, SE={se:.4f}, t={t:.3f}, p={p:.4f}")
    reg_df = pd.DataFrame({
        'predictor': pred_names,
        'coef': coeffs, 'se': ses, 't': tstats, 'pval': pvals,
    })

reg_df.to_csv(f'{OUT}/h3del_confound_regression.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Sub-L: Geneformer static embedding LID
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Sub-L: Geneformer static LID ---")

GENEFORMER_DIR = Path(
    "<HF_CACHE>/hub/models--ctheodoris--Geneformer"
    "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M"
)

try:
    from safetensors.torch import safe_open
    import torch

    gfm_emb_full = None
    with safe_open(str(GENEFORMER_DIR / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        print(f"  Geneformer safetensors keys: {keys[:5]}...")
        for k in keys:
            if "word_embeddings" in k:
                gfm_emb_full = f.get_tensor(k).cpu().float().numpy()
                print(f"  Geneformer word_embeddings shape: {gfm_emb_full.shape}")
                break

    if gfm_emb_full is None:
        raise ValueError("word_embeddings key not found")

    # Filter to the 1500 HVG token IDs (from gene_features.csv maxtoki_token_id)
    hvg_token_ids = gene_features['maxtoki_token_id'].values.astype(int)
    # Both models share vocab; maxtoki_token_id == Geneformer token_id for shared genes
    valid = hvg_token_ids < gfm_emb_full.shape[0]
    print(f"  HVG token IDs in Geneformer vocab: {valid.sum()}/{len(hvg_token_ids)}")

    gfm_hvg = gfm_emb_full[hvg_token_ids[valid]]  # (n_valid, gfm_hidden)
    valid_idx = np.where(valid)[0]

    # Compute LID on Geneformer static embeddings (Levina-Bickel, k=20, cosine)
    def lid_mle_cosine(X, k=20):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_n = X / (norms + 1e-10)
        cos_dist = 1.0 - (X_n @ X_n.T).astype(np.float64)
        np.fill_diagonal(cos_dist, np.inf)
        sorted_d = np.sort(cos_dist, axis=1)[:, :k]
        r_k = sorted_d[:, k - 1]
        r_inner = sorted_d[:, :k - 1] + 1e-12
        log_ratios = np.log(r_k[:, None] / r_inner)
        log_ratios = np.clip(log_ratios, 1e-12, None)
        mean_log = log_ratios.mean(axis=1)
        return np.where(mean_log > 1e-10, 1.0 / mean_log, np.nan)

    gfm_lid = lid_mle_cosine(gfm_hvg, k=20)
    print(f"  Geneformer LID: mean={np.nanmean(gfm_lid):.2f}, median={np.nanmedian(gfm_lid):.2f}")

    # Align back to gene_features
    gfm_lid_full = np.full(n_genes, np.nan)
    gfm_lid_full[valid_idx] = gfm_lid

    # Group comparison
    from statsmodels.stats.multitest import multipletests as _mt
    group_list = ['TF_only', 'TF_and_target', 'target_only', 'string_member']
    test_rows_l = []
    for g1, g2 in [('TF_only', 'target_only'), ('TF_only', 'string_member'),
                   ('target_only', 'string_member')]:
        m1 = (groups == g1) & ~np.isnan(gfm_lid_full)
        m2 = (groups == g2) & ~np.isnan(gfm_lid_full)
        if m1.sum() < 3 or m2.sum() < 3:
            continue
        stat, pval = stats.mannwhitneyu(gfm_lid_full[m1], gfm_lid_full[m2], alternative='two-sided')
        test_rows_l.append({
            'group1': g1, 'group2': g2,
            'n1': int(m1.sum()), 'n2': int(m2.sum()),
            'median1': float(np.median(gfm_lid_full[m1])),
            'median2': float(np.median(gfm_lid_full[m2])),
            'delta': float(np.median(gfm_lid_full[m1]) - np.median(gfm_lid_full[m2])),
            'mw_stat': float(stat), 'pval': float(pval),
        })
        print(f"  GFM LID {g1} vs {g2}: med1={np.median(gfm_lid_full[m1]):.2f}, "
              f"med2={np.median(gfm_lid_full[m2]):.2f}, "
              f"delta={np.median(gfm_lid_full[m1])-np.median(gfm_lid_full[m2]):.2f}, "
              f"p={pval:.4f}")

    gfm_lid_df = pd.DataFrame({
        'symbol':      gene_features['symbol'],
        'group':       groups,
        'lid_gfm':     gfm_lid_full,
        'lid_maxtoki_l0':  lid_l0,
        'lid_maxtoki_l11': lid_l11,
    })
    gfm_lid_df.to_csv(f'{OUT}/h3del_geneformer_lid.csv', index=False)

    tests_l_df = pd.DataFrame(test_rows_l)
    if len(tests_l_df) > 0:
        _, q, _, _ = _mt(tests_l_df['pval'], method='fdr_bh')
        tests_l_df['q_bh'] = q
    tests_l_df.to_csv(f'{OUT}/h3del_geneformer_lid_tests.csv', index=False)
    _sub_l_ok = True

except Exception as e:
    print(f"  Sub-L failed: {e}")
    _sub_l_ok = False

# ── Consolidated summary ──────────────────────────────────────────────────────
summary_rows = [
    # Sub-D
    {'sub': 'D', 'test': 'spearman_degree_lid_l0',  'value': rho_d_l0,  'pval': p_d_l0},
    {'sub': 'D', 'test': 'spearman_degree_lid_l6',  'value': rho_d_l6,  'pval': p_d_l6},
    {'sub': 'D', 'test': 'spearman_degree_lid_l11', 'value': rho_d_l11, 'pval': p_d_l11},
    {'sub': 'D', 'test': 'raw_tf_vs_target_delta_l6', 'value': raw_delta, 'pval': raw_p},
    {'sub': 'D', 'test': 'degree_controlled_delta_l6', 'value': delta_deg, 'pval': p_deg},
    # Sub-E
    {'sub': 'E', 'test': 'ols_TF_only_coef_l6',
     'value': float(reg_df.loc[reg_df['predictor']=='TF_only','coef'].values[0]),
     'pval':  float(reg_df.loc[reg_df['predictor']=='TF_only','pval'].values[0])},
]
if _sub_l_ok and len(tests_l_df) > 0:
    tf_row = tests_l_df[tests_l_df['group1']=='TF_only']
    if len(tf_row) > 0:
        summary_rows.append({
            'sub': 'L', 'test': 'gfm_tf_vs_target_delta',
            'value': float(tf_row['delta'].values[0]),
            'pval': float(tf_row['pval'].values[0]),
        })

pd.DataFrame(summary_rows).to_csv(f'{OUT}/h3del_summary.csv', index=False)

print(f"\nTotal time: {time.time()-t_start:.1f}s")
print("Saved: h3del_degree_lid.csv, h3del_confound_regression.csv, h3del_geneformer_lid.csv, h3del_summary.csv")
