"""Phase 3 — Residualization, degree-preserving null, propensity matching.

Three sub-analyses on the Phase-0 attention and Spearman edges:

  (3a) Cross-fitted residualization of edge scores on gene-level features.
       Fit edge ~ f(mean_expr, variance, dropout) with OLS and GBDT under
       5-fold cross-fit; recompute TRRUST AUROC on the residuals.

  (3b) Degree-preserving null via the curveball algorithm (n=50 here, vs
       n=200 in the source paper — scoped down for compute budget).

  (3c) Propensity matching. Fit is_de_target ~ gene features; nearest
       5 DE-negative matches per DE-positive target; recompute AUROC on
       the matched set; run GroupKFold logistic regression on matched
       pairs.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests as _mult
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
DATASET = os.environ.get("DATASET", "k562").lower()
_SUFFIX = os.environ.get("OUT_SUFFIX", "" if DATASET == "k562" else f"_{DATASET}")
PHASE0 = PROJ / f"runs/attention-grn-217M/outputs/phase0{_SUFFIX}"
PHASE1 = PROJ / f"runs/attention-grn-217M/outputs/phase1{_SUFFIX}"
OUT_DIR = PROJ / f"runs/attention-grn-217M/outputs/phase3{_SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds, SYM2ENS_PKL, load_hvg_matrix

TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
PRIMARY_LAYER = int(os.environ.get("PHASE3_PRIMARY_LAYER",
                                   os.environ.get("PHASE1_PRIMARY_LAYER", "8")))
CURVEBALL_N = int(os.environ.get("PHASE3_CURVEBALL_N", "50"))
LABEL_SHUFFLE_N = int(os.environ.get("PHASE3_LABEL_SHUFFLE_N", "1000"))
PROPENSITY_K = 5
SEED = 42

print("=" * 70)
print(f"PHASE 3 — Residualization + degree null + propensity matching "
      f"(MaxToki L{PRIMARY_LAYER})")
print("=" * 70)

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------------
# Load artefacts
# ------------------------------------------------------------------------
print("\n[1] Loading Phase 0 + Phase 1 artefacts...")
edges_layer_mean = np.load(PHASE0 / "attention_edges_layer_mean.npy")
spearman_edges = np.load(PHASE0 / "spearman_edges.npy").astype(np.float32)
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
n_hvg = len(gene_features)
hvg_symbols_upper = [s.upper() for s in gene_features["symbol"]]
symbol_to_hvg = {s: i for i, s in enumerate(hvg_symbols_upper)}

attn = edges_layer_mean[PRIMARY_LAYER].astype(np.float32).copy()
np.fill_diagonal(attn, 0.0)
np.fill_diagonal(spearman_edges, 0.0)

gmean = gene_features["mean_expr"].to_numpy(dtype=np.float32)
gvar = gene_features["variance"].to_numpy(dtype=np.float32)
gdrop = gene_features["dropout_rate"].to_numpy(dtype=np.float32)
tf_deg = gene_features["tf_out_degree"].to_numpy(dtype=np.float32)

# ------------------------------------------------------------------------
# Build TRRUST ground truth in HVG space
# ------------------------------------------------------------------------
print("\n[2] Building TRRUST ground-truth edge matrix...")
trrust = pd.read_csv(
    TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"]
)
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
tf_in_hvg = set(trrust["tf"]) & set(hvg_symbols_upper)
print(f"  TRRUST rows: {len(trrust):,}  TFs in HVG: {len(tf_in_hvg)}")

trrust_edges = np.zeros((n_hvg, n_hvg), dtype=np.int8)
for _, r in trrust.iterrows():
    i = symbol_to_hvg.get(r["tf"])
    j = symbol_to_hvg.get(r["target"])
    if i is not None and j is not None and i != j:
        trrust_edges[i, j] = 1
n_trrust_edges = int(trrust_edges.sum())
print(f"  TRRUST edges in HVG grid: {n_trrust_edges}")

# TF mask — rows we evaluate
tf_mask = np.zeros(n_hvg, dtype=bool)
for tf in tf_in_hvg:
    i = symbol_to_hvg[tf]
    if trrust_edges[i].sum() >= 3:  # need at least 3 positives per TF row
        tf_mask[i] = True
print(f"  TFs with >= 3 TRRUST targets in HVG: {int(tf_mask.sum())}")

def trrust_auroc(edge_matrix: np.ndarray) -> float:
    """Global TRRUST recovery AUROC across all (TF, candidate) pairs."""
    scores = []
    labels = []
    for tf_i in np.where(tf_mask)[0]:
        row = edge_matrix[tf_i].copy()
        lbl = trrust_edges[tf_i].copy()
        # Exclude self
        mask = np.ones(n_hvg, dtype=bool); mask[tf_i] = False
        scores.append(row[mask])
        labels.append(lbl[mask])
    s = np.concatenate(scores)
    l = np.concatenate(labels)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    return float(roc_auc_score(l, s))

# Directional (absolute) attention — attention is always positive, but for
# consistency use raw attn. Spearman uses absolute value since the sign is
# not meaningful for edge strength.
attn_score = attn
spear_score = np.abs(spearman_edges)

baseline_attn_auroc = trrust_auroc(attn_score)
baseline_spear_auroc = trrust_auroc(spear_score)
print(f"  TRRUST AUROC baseline — attention: {baseline_attn_auroc:.4f}")
print(f"  TRRUST AUROC baseline — spearman:  {baseline_spear_auroc:.4f}")

# ------------------------------------------------------------------------
# 3a. Cross-fitted residualization
# ------------------------------------------------------------------------
print("\n[3a] Cross-fitted residualization...")

def _flatten_pairwise(edges: np.ndarray):
    """Convert a (G, G) edge matrix to an (n_pairs, 5) feature matrix:
    columns = [src_mean, src_var, tgt_mean, tgt_var, tgt_dropout]."""
    G = edges.shape[0]
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    mask = ii != jj
    src = ii[mask]
    tgt = jj[mask]
    X = np.stack([
        gmean[src], gvar[src], gmean[tgt], gvar[tgt], gdrop[tgt]
    ], axis=1)
    y = edges[src, tgt]
    return X, y, src, tgt

# --- residualise attention and spearman ---
def _cross_fit_residualize(edges, model_fn, n_splits=5, seed=SEED):
    X, y, src, tgt = _flatten_pairwise(edges)
    resid = np.empty_like(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_train = []
    for tr, te in kf.split(X):
        m = model_fn(); m.fit(X[tr], y[tr])
        pred_te = m.predict(X[te])
        resid[te] = y[te] - pred_te
        pred_tr = m.predict(X[tr])
        ss_tot = ((y[tr] - y[tr].mean()) ** 2).sum() + 1e-12
        ss_res = ((y[tr] - pred_tr) ** 2).sum()
        r2_train.append(1.0 - ss_res / ss_tot)
    # rebuild (G, G) residual matrix
    R = np.zeros_like(edges, dtype=np.float32)
    R[src, tgt] = resid
    return R, float(np.mean(r2_train))

def _ols():
    return Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())])

def _gbdt():
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=SEED, subsample=0.8,
    )

resid_results = []
for name, edges, baseline in [
    ("attention", attn_score, baseline_attn_auroc),
    ("spearman",  spear_score, baseline_spear_auroc),
]:
    for fit_name, fit in [("ols", _ols), ("gbdt", _gbdt)]:
        t0 = time.time()
        R, r2 = _cross_fit_residualize(edges, fit, n_splits=5)
        # TRRUST AUROC on residual
        auc = trrust_auroc(R)
        resid_results.append({
            "edge": name, "model": fit_name,
            "r2_train_mean": r2,
            "baseline_trrust_auroc": baseline,
            "residualized_trrust_auroc": auc,
            "delta_auroc_lost": baseline - auc,
            "fraction_above_chance_retained": (
                (auc - 0.5) / max(baseline - 0.5, 1e-6)
            ),
            "seconds": float(time.time() - t0),
        })
        print(f"  {name:10s} | {fit_name:4s} | R²={r2:.3f}  "
              f"baseline AUROC={baseline:.4f}  residual AUROC={auc:.4f}")
resid_df = pd.DataFrame(resid_results)
resid_df.to_csv(OUT_DIR / "residualization_results.csv", index=False)

# ------------------------------------------------------------------------
# 3b. Degree-preserving null via curveball
# ------------------------------------------------------------------------
print(f"\n[3b] Degree-preserving null (curveball n={CURVEBALL_N})...")

def curveball_permute(mat: np.ndarray, n_iter: int, seed: int) -> np.ndarray:
    """Curveball algorithm (Strona et al. 2014). Permutes a binary matrix
    while preserving row and column sums."""
    rng = np.random.default_rng(seed)
    # Work on a row-list representation for speed
    rows = [set(np.where(r)[0]) for r in mat.astype(bool)]
    G = len(rows)
    for _ in range(n_iter):
        i, j = rng.choice(G, 2, replace=False)
        a, b = rows[i], rows[j]
        inter = a & b
        sym = (a | b) - inter
        if len(sym) < 2:
            continue
        sym_list = list(sym)
        rng.shuffle(sym_list)
        k = len(a) - len(inter)
        new_a = inter | set(sym_list[:k])
        new_b = inter | set(sym_list[k:])
        rows[i] = new_a
        rows[j] = new_b
    out = np.zeros_like(mat)
    for r_idx, cols in enumerate(rows):
        for c in cols:
            out[r_idx, c] = 1
    return out

# Binary "degree" representation of TRRUST in the HVG context
t0 = time.time()
cb_aurocs = []
for trial in range(CURVEBALL_N):
    perm_edges = curveball_permute(trrust_edges, n_iter=5 * n_trrust_edges,
                                    seed=SEED + trial)
    # Compute AUROC of attention against the permuted edges
    scores, labels = [], []
    for tf_i in np.where(tf_mask)[0]:
        m = np.ones(n_hvg, dtype=bool); m[tf_i] = False
        scores.append(attn_score[tf_i][m])
        labels.append(perm_edges[tf_i][m])
    s = np.concatenate(scores); l = np.concatenate(labels)
    if l.sum() == 0 or l.sum() == len(l):
        continue
    cb_aurocs.append(float(roc_auc_score(l, s)))
print(f"  curveball null complete ({time.time()-t0:.1f}s)")
cb_null_mean = float(np.mean(cb_aurocs)) if cb_aurocs else float("nan")
cb_null_std = float(np.std(cb_aurocs)) if cb_aurocs else float("nan")
cb_z = (baseline_attn_auroc - cb_null_mean) / max(cb_null_std, 1e-12)
print(f"  observed: {baseline_attn_auroc:.4f}  "
      f"null mean={cb_null_mean:.4f}  std={cb_null_std:.4f}  z={cb_z:.2f}")

# Label-shuffle null (cheap; permute the binary TRRUST labels within each TF row)
t0 = time.time()
ls_aurocs = []
for trial in range(LABEL_SHUFFLE_N):
    shuf_edges = trrust_edges.copy()
    rng_trial = np.random.default_rng(SEED + 1000 + trial)
    for tf_i in np.where(tf_mask)[0]:
        row = shuf_edges[tf_i]
        rng_trial.shuffle(row)
        shuf_edges[tf_i] = row
    scores, labels = [], []
    for tf_i in np.where(tf_mask)[0]:
        m = np.ones(n_hvg, dtype=bool); m[tf_i] = False
        scores.append(attn_score[tf_i][m])
        labels.append(shuf_edges[tf_i][m])
    s = np.concatenate(scores); l = np.concatenate(labels)
    if l.sum() == 0 or l.sum() == len(l):
        continue
    ls_aurocs.append(float(roc_auc_score(l, s)))
ls_null_mean = float(np.mean(ls_aurocs))
ls_null_std = float(np.std(ls_aurocs))
ls_z = (baseline_attn_auroc - ls_null_mean) / max(ls_null_std, 1e-12)
print(f"  label-shuffle null ({time.time()-t0:.1f}s): "
      f"mean={ls_null_mean:.4f}  std={ls_null_std:.4f}  z={ls_z:.2f}")

null_results = {
    "baseline_trrust_auroc_attention": baseline_attn_auroc,
    "curveball": {
        "n_iter": CURVEBALL_N,
        "null_mean": cb_null_mean,
        "null_std": cb_null_std,
        "z": cb_z,
    },
    "label_shuffle": {
        "n_iter": LABEL_SHUFFLE_N,
        "null_mean": ls_null_mean,
        "null_std": ls_null_std,
        "z": ls_z,
    },
}
with open(OUT_DIR / "null_results.json", "w") as f:
    json.dump(null_results, f, indent=2)

# ------------------------------------------------------------------------
# 3c. Propensity matching
# ------------------------------------------------------------------------
print("\n[3c] Propensity matching on Phase-1 DE targets...")
# Rebuild the DE dataset (shared with Phase 2).
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
LFC_THRESHOLD = 0.5
DE_PVAL_THRESHOLD = 0.05
MIN_CELLS_PER_PERT = 30
MIN_DE_POSITIVES = 3
t0 = time.time()
var_idx_col = gene_features["var_idx"].to_numpy(dtype=np.int64)
ds = load_ds(DATASET)
with h5py.File(ds.h5_path, "r") as f:
    k562 = ds.cell_of_interest_mask
    pg_cats = ds.perturbation_categories
    pg_codes = ds.perturbation_codes
    nt_codes_set = ds.control_category_codes
    nt_idx = np.where(k562 & np.isin(pg_codes, list(nt_codes_set)))[0]
    if len(nt_idx) > 5000:
        nt_idx = np.sort(rng.choice(nt_idx, 5000, replace=False))

    print(f"  loading HVG-only expression (sequential)...")
    _t_io = time.time()
    X_all_hvg = load_hvg_matrix(ds, var_idx_col)
    print(f"  loaded HVG matrix: {X_all_hvg.shape}  ({time.time()-_t_io:.1f}s)")

    X_nt = X_all_hvg[nt_idx]
    rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_nt = np.log1p(X_nt / rs * 1e4)
    mu_nt = X_nt.mean(axis=0); var_nt = X_nt.var(axis=0, ddof=1); N_nt = len(X_nt)
    k562_cells = np.where(k562)[0]; k562_codes = pg_codes[k562]

    pair_rows = []
    for code in np.unique(k562_codes):
        cat = pg_cats[code]
        if code in nt_codes_set:
            continue
        if cat.lower() == "control":
            continue
        if sym2ens.get(cat) is None:
            continue
        pert_hvg = symbol_to_hvg.get(cat.upper())
        if pert_hvg is None:
            continue
        sel = k562_cells[k562_codes == code]
        if len(sel) < MIN_CELLS_PER_PERT:
            continue
        rows_sorted = np.sort(sel)
        X_p = X_all_hvg[rows_sorted]
        rs_p = X_p.sum(axis=1, keepdims=True); rs_p[rs_p == 0] = 1.0
        X_p = np.log1p(X_p / rs_p * 1e4)
        mu_p = X_p.mean(axis=0); var_p = X_p.var(axis=0, ddof=1); N_p = len(X_p)
        se = np.sqrt(var_p / N_p + var_nt / N_nt); se[se == 0] = 1e-12
        t_stat = (mu_p - mu_nt) / se
        df = (var_p / N_p + var_nt / N_nt) ** 2 / (
            (var_p / N_p) ** 2 / max(N_p - 1, 1)
            + (var_nt / N_nt) ** 2 / max(N_nt - 1, 1))
        pval = 2.0 * sp_stats.t.sf(np.abs(t_stat), df)
        _, p_bh, _, _ = _mult(np.nan_to_num(pval, nan=1.0), method="fdr_bh")
        lfc = mu_p - mu_nt
        is_de = (np.abs(lfc) >= LFC_THRESHOLD) & (p_bh < DE_PVAL_THRESHOLD)
        valid = np.ones(n_hvg, dtype=bool); valid[pert_hvg] = False
        is_de_eff = is_de & valid
        if is_de_eff.sum() < MIN_DE_POSITIVES or (valid.sum() - is_de_eff.sum()) < MIN_DE_POSITIVES:
            continue
        for t in np.where(valid)[0]:
            pair_rows.append({
                "pert_symbol": cat, "pert_hvg": pert_hvg, "target_hvg": int(t),
                "is_de": int(is_de_eff[t]),
                "t_mean": gmean[t], "t_var": gvar[t], "t_drop": gdrop[t],
                "attn_edge": attn_score[pert_hvg, t],
                "spear_edge": spear_score[pert_hvg, t],
            })
df_full = pd.DataFrame(pair_rows)
print(f"  full pair set: {len(df_full):,}  positive rate {df_full['is_de'].mean():.4f}")
df_full.to_csv(OUT_DIR / "full_pair_dataset.csv", index=False)

# Fit propensity model on target features (shared across perturbations)
pm_feats = df_full[["t_mean", "t_var", "t_drop"]].to_numpy()
pm_y = df_full["is_de"].to_numpy()
pm = Pipeline([("sc", StandardScaler()),
               ("lr", LogisticRegression(max_iter=1000))])
pm.fit(pm_feats, pm_y)
df_full["propensity"] = pm.predict_proba(pm_feats)[:, 1]

# Within-perturbation nearest-neighbour matching
matched_rows = []
for pert, grp in df_full.groupby("pert_symbol"):
    pos = grp[grp.is_de == 1]
    neg = grp[grp.is_de == 0]
    if len(pos) == 0 or len(neg) < PROPENSITY_K:
        continue
    nn = NearestNeighbors(n_neighbors=PROPENSITY_K).fit(
        neg[["propensity"]].to_numpy())
    _, idxs = nn.kneighbors(pos[["propensity"]].to_numpy())
    matched_rows.append(pos)
    matched_rows.append(neg.iloc[idxs.flatten()])
df_matched = pd.concat(matched_rows, ignore_index=True)
print(f"  matched set: {len(df_matched):,}  "
      f"positive rate {df_matched['is_de'].mean():.4f}")
df_matched.to_csv(OUT_DIR / "matched_pair_dataset.csv", index=False)

# SMD check: target features before/after matching
def _smd(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))
smd = {}
for col in ["t_mean", "t_var", "t_drop"]:
    pre = _smd(df_full[df_full.is_de == 1][col], df_full[df_full.is_de == 0][col])
    post = _smd(df_matched[df_matched.is_de == 1][col],
                df_matched[df_matched.is_de == 0][col])
    smd[col] = {"pre": pre, "post": post}
print("  SMD pre→post:", smd)

# Raw AUROC on matched set
def _pert_auroc(df, score_col):
    aucs = []
    for _, g in df.groupby("pert_symbol"):
        if g["is_de"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g["is_de"], g[score_col]))
    return float(np.mean(aucs)) if aucs else float("nan"), int(len(aucs))

raw_attn_auc, n = _pert_auroc(df_matched, "attn_edge")
raw_spear_auc, _ = _pert_auroc(df_matched, "spear_edge")
print(f"  matched per-pert mean AUROC (raw): attn={raw_attn_auc:.4f}  "
      f"spear={raw_spear_auc:.4f}  (n_perts={n})")

# GroupKFold logistic regression on matched set (gene-only vs gene+edge)
gkf = GroupKFold(n_splits=5)
groups = df_matched["pert_symbol"].astype("category").cat.codes.to_numpy()
y = df_matched["is_de"].to_numpy()
def _run_gkf(X, y, groups):
    aurs = []
    for tr, te in gkf.split(X, y, groups=groups):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000,
                                                 class_weight="balanced",
                                                 solver="lbfgs"))])
        m.fit(X[tr], y[tr])
        aurs.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return float(np.mean(aurs)) if aurs else float("nan")

X_go = df_matched[["t_mean", "t_var", "t_drop"]].to_numpy(dtype=np.float32)
X_ga = np.column_stack([X_go, df_matched["attn_edge"].to_numpy(dtype=np.float32)])
X_gc = np.column_stack([X_go, df_matched["spear_edge"].to_numpy(dtype=np.float32)])
match_go = _run_gkf(X_go, y, groups)
match_ga = _run_gkf(X_ga, y, groups)
match_gc = _run_gkf(X_gc, y, groups)
print(f"  matched GroupKFold: gene_only={match_go:.4f}  "
      f"gene+attn={match_ga:.4f}  gene+corr={match_gc:.4f}")

match_results = {
    "n_matched_pairs": int(len(df_matched)),
    "positive_rate_matched": float(df_matched["is_de"].mean()),
    "smd": smd,
    "matched_mean_per_pert_auroc": {
        "attention": raw_attn_auc, "spearman_abs": raw_spear_auc,
    },
    "matched_gkf_auroc": {
        "gene_only": match_go,
        "gene_plus_attn": match_ga,
        "gene_plus_corr": match_gc,
        "delta_attn": match_ga - match_go,
        "delta_corr": match_gc - match_go,
    },
}
with open(OUT_DIR / "propensity_matching_results.json", "w") as f:
    json.dump(match_results, f, indent=2)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "primary_layer": PRIMARY_LAYER,
        "curveball_n": CURVEBALL_N,
        "label_shuffle_n": LABEL_SHUFFLE_N,
        "propensity_k": PROPENSITY_K,
        "n_trrust_edges_in_hvg": n_trrust_edges,
        "n_tfs_evaluated": int(tf_mask.sum()),
    }, f, indent=2)
print(f"\nPHASE 3 COMPLETE — outputs: {OUT_DIR}")
