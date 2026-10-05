"""Phase 2 — Conditional incremental-value testing.

For every (perturbation, target-gene) pair, build a feature vector with
target gene-level features and each of three candidate edge scores
(attention, Spearman, both-null). Fit logistic regression + GBDT under
5-fold GroupKFold cross-validation grouped by perturbation, and report
whether attention adds any predictive value beyond gene-level features.

Primary test is cross-perturbation GroupKFold (held-out perturbations).
Second harder split: cross-target-gene GroupKFold. Third: joint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    HAS_GBDT = True
except Exception:
    HAS_GBDT = False

import os
PROJ = Path(__file__).resolve().parents[3]
DATASET = os.environ.get("DATASET", "k562").lower()
_SUFFIX = os.environ.get("OUT_SUFFIX", "" if DATASET == "k562" else f"_{DATASET}")
PHASE0 = PROJ / f"runs/attention-grn-217M/outputs/phase0{_SUFFIX}"
PHASE1 = PROJ / f"runs/attention-grn-217M/outputs/phase1{_SUFFIX}"
OUT_DIR = PROJ / f"runs/attention-grn-217M/outputs/phase2{_SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds, SYM2ENS_PKL, load_hvg_matrix

PRIMARY_LAYER = int(os.environ.get("PHASE2_PRIMARY_LAYER",
                                   os.environ.get("PHASE1_PRIMARY_LAYER", "8")))
N_SPLITS = 5
SEED = 42

print("=" * 70)
print(f"PHASE 2 — Incremental value testing (MaxToki L{PRIMARY_LAYER})")
print("=" * 70)

# ------------------------------------------------------------------------
# Load artefacts
# ------------------------------------------------------------------------
print("\n[1] Loading artefacts...")
edges_layer_mean = np.load(PHASE0 / "attention_edges_layer_mean.npy")
spearman_edges = np.load(PHASE0 / "spearman_edges.npy")
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
auc_df = pd.read_csv(PHASE1 / "per_perturbation_auroc.csv")
n_hvg = len(gene_features)
attn = edges_layer_mean[PRIMARY_LAYER].copy()
np.fill_diagonal(attn, 0.0)
np.fill_diagonal(spearman_edges, 0.0)

# Recover DE labels by re-running the same DE-call procedure as Phase 1.
# Simpler: re-use the per-perturbation DE arrays stored in phase1 by
# regenerating them. To avoid reloading h5py, dump a compact NPZ from Phase 1.
# We stored per_perturbation_auroc.csv but NOT the DE masks; re-compute
# them here via a quick pass over the h5.
#
# That means Phase 2 has to re-do the DE-call. To keep this self-contained,
# factor the DE-call into a tiny helper and call it. But for efficiency,
# cache the Phase-1 DE masks to disk. Let's just rebuild them now.
#
# IMPLEMENTATION: call Phase 1 DE-building as a function.

import h5py
import pickle
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests as _mult
ds = load_ds(DATASET)

hvg_symbols = gene_features["symbol"].tolist()
symbol_to_hvg = {s.upper(): i for i, s in enumerate(hvg_symbols)}
var_idx = gene_features["var_idx"].to_numpy(dtype=np.int64)

with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)

LFC_THRESHOLD = 0.5
DE_PVAL_THRESHOLD = 0.05
MIN_CELLS_PER_PERT = 30
MIN_DE_POSITIVES = 3

print(f"\n[2] Rebuilding DE masks from {DATASET.upper()}...")
t0 = time.time()
with h5py.File(ds.h5_path, "r") as f:
    k562 = ds.cell_of_interest_mask
    pg_cats = ds.perturbation_categories
    pg_codes = ds.perturbation_codes

    nt_codes_set = ds.control_category_codes
    nt_idx = np.where(k562 & np.isin(pg_codes, list(nt_codes_set)))[0]
    rng = np.random.default_rng(SEED)
    if len(nt_idx) > 5000:
        nt_idx = np.sort(rng.choice(nt_idx, 5000, replace=False))

    print(f"  loading HVG-only expression (sequential)...")
    _t_io = time.time()
    X_all_hvg = load_hvg_matrix(ds, var_idx)
    print(f"  loaded HVG matrix: {X_all_hvg.shape}  ({time.time()-_t_io:.1f}s)")

    X_nt = X_all_hvg[nt_idx]
    rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_nt = np.log1p(X_nt / rs * 1e4)
    mu_nt = X_nt.mean(axis=0)
    var_nt = X_nt.var(axis=0, ddof=1)
    N_nt = len(X_nt)

    k562_cells = np.where(k562)[0]
    k562_codes = pg_codes[k562]

    pert_records = []
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
        mu_p = X_p.mean(axis=0)
        var_p = X_p.var(axis=0, ddof=1)
        N_p = len(X_p)
        se = np.sqrt(var_p / N_p + var_nt / N_nt); se[se == 0] = 1e-12
        t_stat = (mu_p - mu_nt) / se
        df = (var_p / N_p + var_nt / N_nt) ** 2 / (
            (var_p / N_p) ** 2 / max(N_p - 1, 1)
            + (var_nt / N_nt) ** 2 / max(N_nt - 1, 1)
        )
        pval = 2.0 * sp_stats.t.sf(np.abs(t_stat), df)
        _, p_bh, _, _ = _mult(np.nan_to_num(pval, nan=1.0), method="fdr_bh")
        lfc = mu_p - mu_nt
        is_de = (np.abs(lfc) >= LFC_THRESHOLD) & (p_bh < DE_PVAL_THRESHOLD)
        valid = np.ones(n_hvg, dtype=bool)
        valid[pert_hvg] = False
        is_de_eff = is_de & valid
        n_pos = int(is_de_eff.sum())
        n_neg = int(valid.sum() - n_pos)
        if n_pos < MIN_DE_POSITIVES or n_neg < MIN_DE_POSITIVES:
            continue
        pert_records.append({
            "pert_symbol": cat,
            "pert_hvg_idx": pert_hvg,
            "is_de": is_de_eff,
            "valid": valid,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })
print(f"  evaluable perturbations: {len(pert_records)}  ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------------------
# Build per-(pert, target) dataset
# ------------------------------------------------------------------------
print("\n[3] Building (pert, target) dataset...")
rows = []
pert_ids = []
target_ids = []
labels = []
for pid, rec in enumerate(pert_records):
    src = rec["pert_hvg_idx"]
    valid_idx = np.where(rec["valid"])[0]
    for t in valid_idx:
        label = int(rec["is_de"][t])
        rows.append({
            "pert_idx": pid,
            "pert_hvg_idx": src,
            "target_hvg_idx": int(t),
            "label": label,
            "target_mean": float(gene_features["mean_expr"].iloc[t]),
            "target_variance": float(gene_features["variance"].iloc[t]),
            "target_dropout": float(gene_features["dropout_rate"].iloc[t]),
            "target_tf_out_degree": float(gene_features["tf_out_degree"].iloc[t]),
            "attention_edge": float(attn[src, t]),
            "spearman_edge": float(spearman_edges[src, t]),
            "spearman_edge_abs": float(abs(spearman_edges[src, t])),
        })
ds = pd.DataFrame(rows)
print(f"  n pairs: {len(ds):,}  positive rate: {ds['label'].mean():.4f}")
print(f"  n perturbations: {ds['pert_idx'].nunique()}")
print(f"  n unique targets: {ds['target_hvg_idx'].nunique()}")

ds.to_csv(OUT_DIR / "pair_dataset.csv", index=False)

# ------------------------------------------------------------------------
# 5-fold GroupKFold under three split designs
# ------------------------------------------------------------------------
def _evaluate_model(X, y, groups, model_fn):
    gkf = GroupKFold(n_splits=N_SPLITS)
    auroc, auprc = [], []
    for tr, te in gkf.split(X, y, groups=groups):
        y_tr = y[tr]; y_te = y[te]
        if y_tr.sum() == 0 or y_te.sum() == 0:
            continue
        model = model_fn()
        model.fit(X[tr], y_tr)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X[te])[:, 1]
        else:
            prob = model.decision_function(X[te])
        auroc.append(float(roc_auc_score(y_te, prob)))
        auprc.append(float(average_precision_score(y_te, prob)))
    return {
        "auroc_mean": float(np.mean(auroc)) if auroc else float("nan"),
        "auroc_std": float(np.std(auroc)) if auroc else float("nan"),
        "auprc_mean": float(np.mean(auprc)) if auprc else float("nan"),
        "auprc_std": float(np.std(auprc)) if auprc else float("nan"),
        "n_valid_folds": int(len(auroc)),
    }

def _mk_lr():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])

def _mk_gbdt():
    return lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        min_child_samples=40, class_weight="balanced",
        random_state=SEED, verbosity=-1,
    )

GENE_COLS = ["target_mean", "target_variance", "target_dropout"]
RESULTS = []

print("\n[4] Running GroupKFold incremental-value tests...")
t0 = time.time()
for split_name, group_col in [
    ("cross_pert", "pert_idx"),
    ("cross_gene", "target_hvg_idx"),
]:
    groups = ds[group_col].to_numpy()
    y = ds["label"].to_numpy().astype(np.int64)
    FEATURE_SETS = {
        "gene_only":        GENE_COLS,
        "gene_plus_attn":   GENE_COLS + ["attention_edge"],
        "gene_plus_corr":   GENE_COLS + ["spearman_edge_abs"],
        "gene_plus_both":   GENE_COLS + ["attention_edge", "spearman_edge_abs"],
        "attn_only":        ["attention_edge"],
        "corr_only":        ["spearman_edge_abs"],
    }
    for model_name, model_fn in [("logreg", _mk_lr)] + (
        [("gbdt", _mk_gbdt)] if HAS_GBDT else []
    ):
        for feat_name, cols in FEATURE_SETS.items():
            X = ds[cols].to_numpy(dtype=np.float32)
            res = _evaluate_model(X, y, groups, model_fn)
            res.update({
                "split": split_name, "model": model_name,
                "feature_set": feat_name, "n_features": len(cols),
            })
            RESULTS.append(res)
            print(f"  [{split_name:10s} | {model_name:6s} | {feat_name:16s}]"
                  f" AUROC={res['auroc_mean']:.4f} ± {res['auroc_std']:.4f}  "
                  f"AUPRC={res['auprc_mean']:.4f}")

# Joint split: hold out by product of pert × target is impractical; approximate
# by grouping by (pert_idx * 10000 + target_hvg_idx // 100).
groups_joint = (ds["pert_idx"].to_numpy() * 100000 + ds["target_hvg_idx"].to_numpy() // 10)
for model_name, model_fn in [("logreg", _mk_lr)] + ([("gbdt", _mk_gbdt)] if HAS_GBDT else []):
    for feat_name, cols in {
        "gene_only":      GENE_COLS,
        "gene_plus_attn": GENE_COLS + ["attention_edge"],
        "gene_plus_corr": GENE_COLS + ["spearman_edge_abs"],
    }.items():
        X = ds[cols].to_numpy(dtype=np.float32)
        y = ds["label"].to_numpy().astype(np.int64)
        res = _evaluate_model(X, y, groups_joint, model_fn)
        res.update({"split": "joint", "model": model_name,
                    "feature_set": feat_name, "n_features": len(cols)})
        RESULTS.append(res)
        print(f"  [joint      | {model_name:6s} | {feat_name:16s}]"
              f" AUROC={res['auroc_mean']:.4f}  AUPRC={res['auprc_mean']:.4f}")

print(f"  split time: {time.time()-t0:.1f}s")

# ------------------------------------------------------------------------
# Delta-AUROC table
# ------------------------------------------------------------------------
print("\n[5] Delta-AUROC summary (vs gene_only)...")
res_df = pd.DataFrame(RESULTS)
res_df.to_csv(OUT_DIR / "incremental_value_results.csv", index=False)

delta_rows = []
for (split, model), g in res_df.groupby(["split", "model"]):
    base = g[g["feature_set"] == "gene_only"]
    if base.empty:
        continue
    base_auroc = float(base["auroc_mean"].iloc[0])
    for _, r in g.iterrows():
        if r["feature_set"] == "gene_only":
            continue
        delta = float(r["auroc_mean"] - base_auroc)
        delta_rows.append({
            "split": split, "model": model, "feature_set": r["feature_set"],
            "gene_only_auroc": base_auroc,
            "new_auroc": float(r["auroc_mean"]),
            "delta_auroc": delta,
        })
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUT_DIR / "delta_auroc_summary.csv", index=False)
print(delta_df.to_string(index=False))

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "primary_layer": PRIMARY_LAYER,
        "n_splits": N_SPLITS,
        "n_pairs": int(len(ds)),
        "n_perturbations": int(ds['pert_idx'].nunique()),
        "positive_rate": float(ds["label"].mean()),
        "has_gbdt": HAS_GBDT,
    }, f, indent=2)
print(f"\nPHASE 2 COMPLETE — outputs: {OUT_DIR}")
