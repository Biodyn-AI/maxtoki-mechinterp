"""Phase 1 — Trivial-baseline comparison.

For each perturbed gene in Replogle K562, compute per-perturbation AUROC
against DE-positive/DE-negative targets using:

  - attention edge (MaxToki primary layer)
  - Spearman correlation edge
  - gene-level variance (univariate predictor)
  - gene-level mean expression
  - gene-level 1 - dropout rate
  - gene-level TF out-degree (TRRUST, negative control)

Then paired Wilcoxon signed-rank between per-perturbation AUROCs for each
baseline vs attention and baseline vs correlation, BH-corrected.
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
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests as _mult

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")

import os
DATASET = os.environ.get("DATASET", "k562").lower()
_SUFFIX = os.environ.get("OUT_SUFFIX", "" if DATASET == "k562" else f"_{DATASET}")
PHASE0 = PROJ / f"runs/attention-grn-217M/outputs/phase0{_SUFFIX}"
OUT_DIR = PROJ / f"runs/attention-grn-217M/outputs/phase1{_SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds, SYM2ENS_PKL, load_hvg_matrix

# Primary layer for MaxToki 217M: same relative depth as scGPT layer_09 / Geneformer L13.
# scGPT uses layer 9 of 12 (75% depth). Geneformer V2-316M uses L13 of 18 (72% depth).
# For MaxToki-217M (11 layers), 72-75% depth is layer 8 (0-indexed).
PRIMARY_LAYER = int(__import__("os").environ.get("PHASE1_PRIMARY_LAYER", "8"))

LFC_THRESHOLD = 0.5
DE_PVAL_THRESHOLD = 0.05
MIN_CELLS_PER_PERT = 30
MIN_DE_POSITIVES = 3

print("=" * 70)
print(f"PHASE 1 — Trivial baselines (MaxToki primary layer L{PRIMARY_LAYER})")
print("=" * 70)

# ------------------------------------------------------------------------
# Load Phase 0 artefacts
# ------------------------------------------------------------------------
print("\n[1] Loading Phase 0 artefacts...")
t0 = time.time()
edges_per_head = np.load(PHASE0 / "attention_edges_per_head.npy")     # (L, H, G, G)
edges_layer_mean = np.load(PHASE0 / "attention_edges_layer_mean.npy") # (L, G, G)
spearman_edges = np.load(PHASE0 / "spearman_edges.npy")               # (G, G)
pair_counts = np.load(PHASE0 / "attention_pair_counts.npy")           # (G, G)
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
n_layers, n_heads, n_hvg, _ = edges_per_head.shape
print(f"  edges shape: {edges_per_head.shape}  primary layer: {PRIMARY_LAYER}")
print(f"  gene_features: {len(gene_features)}")
print(f"  load: {time.time()-t0:.1f}s")

hvg_symbols = gene_features["symbol"].tolist()
symbol_to_hvg = {s.upper(): i for i, s in enumerate(hvg_symbols)}

# Primary layer attention as (G, G) by mean over heads
attn_primary = edges_layer_mean[PRIMARY_LAYER]
# Normalize per row (source-gene) since raw attention is a probability over
# keys in the cell context, not over all HVGs. Without normalization the
# attention matrix has very different scales across cells and genes.
# Per-row softmax-like rescaling to unit max preserves ordering for AUROC.
row_mx = attn_primary.max(axis=1, keepdims=True)
row_mx[row_mx == 0] = 1.0
attn_primary_norm = attn_primary / row_mx

# Zero-out self-edges (diagonal) in all edge predictors — a gene is not a
# candidate target for its own perturbation.
np.fill_diagonal(attn_primary, 0.0)
np.fill_diagonal(attn_primary_norm, 0.0)
np.fill_diagonal(spearman_edges, 0.0)

# ------------------------------------------------------------------------
# DE-calling per perturbation vs non-targeting controls
# ------------------------------------------------------------------------
print(f"\n[2] DE-calling against {DATASET.upper()} control cells...")
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)

ds = load_ds(DATASET)

# Load metadata + raw expression restricted to HVG columns for speed
with h5py.File(ds.h5_path, "r") as f:
    k562 = ds.cell_of_interest_mask
    pg_cats = ds.perturbation_categories
    pg_codes = ds.perturbation_codes
    n_cells_total = ds.n_cells_total

    # hvg var indices
    var_idx = gene_features["var_idx"].to_numpy(dtype=np.int64)

    print(f"  cells in COI: {int(k562.sum()):,}")

    # control cells
    nt_codes = ds.control_category_codes
    nt_mask = k562 & np.isin(pg_codes, list(nt_codes))
    nt_idx = np.where(nt_mask)[0]
    print(f"  controls: {len(nt_idx):,}")

    # Subsample NT to keep DE-calling fast
    rng = np.random.default_rng(42)
    if len(nt_idx) > 5000:
        nt_idx = np.sort(rng.choice(nt_idx, 5000, replace=False))

    # Sequential HVG-only expression load (dense or CSR-backed, via helper)
    print(f"  loading HVG-only expression (sequential)...")
    _t_io = time.time()
    X_all_hvg = load_hvg_matrix(ds, var_idx)
    print(f"  loaded HVG matrix: {X_all_hvg.shape}  ({time.time()-_t_io:.1f}s)")

    # log-normalize NT controls
    X_nt = X_all_hvg[nt_idx]
    rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_nt = np.log1p(X_nt / rs * 1e4)
    print(f"  NT expression (HVG space): {X_nt.shape}")
    mu_nt = X_nt.mean(axis=0)
    var_nt = X_nt.var(axis=0, ddof=1)

    # Build index of cells by perturbation gene
    k562_idx_by_code: dict[int, np.ndarray] = {}
    k562_codes = pg_codes[k562]
    k562_cells = np.where(k562)[0]
    for code in np.unique(k562_codes):
        cat = pg_cats[code]
        if code in nt_codes:
            continue
        if cat.lower() == "control":
            continue
        sel = k562_cells[k562_codes == code]
        if len(sel) >= MIN_CELLS_PER_PERT:
            k562_idx_by_code[int(code)] = sel

    print(f"  perturbations with >= {MIN_CELLS_PER_PERT} cells: {len(k562_idx_by_code)}")

    # DE-call each perturbation vs NT
    t1 = time.time()
    pert_records = []
    for pi, (code, rows) in enumerate(k562_idx_by_code.items()):
        pert_gene = pg_cats[code]
        pert_ens = sym2ens.get(pert_gene)
        if pert_ens is None:
            continue
        pert_hvg = symbol_to_hvg.get(pert_gene.upper())
        # Fast path: slice from in-RAM HVG matrix (no disk I/O)
        X_p = X_all_hvg[rows]
        rs_p = X_p.sum(axis=1, keepdims=True); rs_p[rs_p == 0] = 1.0
        X_p = np.log1p(X_p / rs_p * 1e4)
        mu_p = X_p.mean(axis=0)
        var_p = X_p.var(axis=0, ddof=1)
        # Welch t (vectorized)
        se = np.sqrt(var_p / max(len(X_p), 1) + var_nt / len(X_nt))
        se[se == 0] = 1e-12
        t_stat = (mu_p - mu_nt) / se
        df = (var_p / max(len(X_p), 1) + var_nt / len(X_nt)) ** 2 / (
            (var_p / max(len(X_p), 1)) ** 2 / max(len(X_p) - 1, 1)
            + (var_nt / len(X_nt)) ** 2 / max(len(X_nt) - 1, 1)
        )
        p = 2.0 * stats.t.sf(np.abs(t_stat), df)
        # BH-correct raw p-values per-perturbation to match source paper's DE convention.
        p_masked = np.nan_to_num(p, nan=1.0)
        _, p_bh, _, _ = _mult(p_masked, method="fdr_bh")
        lfc = mu_p - mu_nt

        is_de = (np.abs(lfc) >= LFC_THRESHOLD) & (p_bh < DE_PVAL_THRESHOLD)
        # exclude the perturbation gene itself from the target pool
        valid = np.ones(n_hvg, dtype=bool)
        if pert_hvg is not None:
            valid[pert_hvg] = False
        is_de_eff = is_de & valid
        n_pos = int(is_de_eff.sum())
        n_neg = int(valid.sum() - n_pos)
        if n_pos < MIN_DE_POSITIVES or n_neg < MIN_DE_POSITIVES:
            continue
        pert_records.append({
            "pert_symbol": pert_gene,
            "pert_ensembl": pert_ens,
            "pert_hvg_idx": pert_hvg,
            "n_cells": int(len(rows)),
            "is_de_target_mask": is_de_eff,
            "valid_target_mask": valid,
            "n_de_positive": n_pos,
            "n_de_negative": n_neg,
            "lfc_mean_abs": float(np.abs(lfc[valid]).mean()),
        })

    print(f"  evaluable perturbations after DE filter: {len(pert_records)}  "
          f"({time.time()-t1:.1f}s)")

# ------------------------------------------------------------------------
# Per-perturbation AUROC for each predictor
# ------------------------------------------------------------------------
print("\n[3] Per-perturbation AUROCs...")
predictors = ["attention_primary", "attention_primary_norm", "spearman",
              "gene_variance", "gene_mean_expr", "gene_one_minus_dropout",
              "gene_tf_out_degree"]

auc_rows = []
for rec in pert_records:
    pert_hvg = rec["pert_hvg_idx"]
    if pert_hvg is None:
        # Perturbation not in HVG set — skip edge predictors (no source row).
        continue
    valid = rec["valid_target_mask"]
    y = rec["is_de_target_mask"][valid].astype(int)
    out = {
        "pert_symbol": rec["pert_symbol"],
        "n_cells": rec["n_cells"],
        "n_de_positive": rec["n_de_positive"],
        "n_de_negative": rec["n_de_negative"],
    }
    # Edge predictors: row of pert_hvg, scored at each valid target
    row_attn = attn_primary[pert_hvg, :][valid]
    row_attn_n = attn_primary_norm[pert_hvg, :][valid]
    row_sp = spearman_edges[pert_hvg, :][valid]
    # Gene-level univariate predictors: same value per pert, but AUROC only
    # depends on ordering of targets, so it's a shared predictor.
    gv = gene_features["variance"].to_numpy()[valid]
    gm = gene_features["mean_expr"].to_numpy()[valid]
    gd = (1.0 - gene_features["dropout_rate"].to_numpy())[valid]
    gtf = gene_features["tf_out_degree"].to_numpy()[valid]

    # Spearman may have negative signs; use |ρ| for ranking (edge strength).
    preds = {
        "attention_primary": row_attn,
        "attention_primary_norm": row_attn_n,
        "spearman": np.abs(row_sp),
        "gene_variance": gv,
        "gene_mean_expr": gm,
        "gene_one_minus_dropout": gd,
        "gene_tf_out_degree": gtf,
    }
    for name, pred in preds.items():
        try:
            auc = roc_auc_score(y, pred)
        except ValueError:
            auc = float("nan")
        out[f"auc_{name}"] = float(auc)
    auc_rows.append(out)

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(OUT_DIR / "per_perturbation_auroc.csv", index=False)
print(f"  saved per-perturbation AUROC table: {auc_df.shape}")

# --- Per-layer attention AUROC profile ---
print("\n[3b] Per-layer attention AUROC profile...")
layer_auc_rows = []
for li in range(n_layers):
    layer_attn = edges_layer_mean[li].copy()
    np.fill_diagonal(layer_attn, 0.0)
    aucs_layer = []
    for rec in pert_records:
        ph = rec["pert_hvg_idx"]
        if ph is None:
            continue
        valid = rec["valid_target_mask"]
        y = rec["is_de_target_mask"][valid].astype(int)
        pred = layer_attn[ph, :][valid]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            aucs_layer.append(roc_auc_score(y, pred))
        except ValueError:
            continue
    layer_auc_rows.append({
        "layer": li,
        "mean_auroc": float(np.mean(aucs_layer)) if aucs_layer else float("nan"),
        "median_auroc": float(np.median(aucs_layer)) if aucs_layer else float("nan"),
        "std_auroc": float(np.std(aucs_layer)) if aucs_layer else float("nan"),
        "n_perts": int(len(aucs_layer)),
    })
layer_auc_df = pd.DataFrame(layer_auc_rows)
layer_auc_df.to_csv(OUT_DIR / "attention_per_layer_auroc.csv", index=False)
print(layer_auc_df.to_string(index=False))

# ------------------------------------------------------------------------
# Paired Wilcoxon: baseline vs attention and baseline vs correlation
# ------------------------------------------------------------------------
print("\n[4] Paired Wilcoxon tests...")
tests = []
for baseline in ["gene_variance", "gene_mean_expr", "gene_one_minus_dropout",
                 "gene_tf_out_degree"]:
    for ref in ["attention_primary", "attention_primary_norm", "spearman"]:
        a = auc_df[f"auc_{baseline}"].to_numpy()
        b = auc_df[f"auc_{ref}"].to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 10:
            continue
        delta = a[mask] - b[mask]
        try:
            W, p = stats.wilcoxon(a[mask], b[mask], zero_method="zsplit")
        except ValueError:
            W, p = float("nan"), float("nan")
        tests.append({
            "baseline": baseline,
            "reference": ref,
            "n_valid": int(mask.sum()),
            "mean_auc_baseline": float(a[mask].mean()),
            "mean_auc_reference": float(b[mask].mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wilcoxon_W": float(W),
            "wilcoxon_p_raw": float(p),
        })

# Direct edge comparison: attention vs spearman (no gene-level)
a = auc_df["auc_attention_primary"].to_numpy()
b = auc_df["auc_spearman"].to_numpy()
mask = np.isfinite(a) & np.isfinite(b)
if mask.sum() >= 10:
    try:
        W, p = stats.wilcoxon(a[mask], b[mask], zero_method="zsplit")
    except ValueError:
        W, p = float("nan"), float("nan")
    tests.append({
        "baseline": "attention_primary",
        "reference": "spearman",
        "n_valid": int(mask.sum()),
        "mean_auc_baseline": float(a[mask].mean()),
        "mean_auc_reference": float(b[mask].mean()),
        "mean_delta": float((a[mask] - b[mask]).mean()),
        "median_delta": float(np.median(a[mask] - b[mask])),
        "wilcoxon_W": float(W),
        "wilcoxon_p_raw": float(p),
    })

test_df = pd.DataFrame(tests)
if len(test_df) > 0:
    _, p_bh, _, _ = _mult(test_df["wilcoxon_p_raw"].to_numpy(), method="fdr_bh")
    test_df["wilcoxon_p_bh"] = p_bh
test_df.to_csv(OUT_DIR / "wilcoxon_baseline_vs_edges.csv", index=False)
print(test_df.to_string(index=False))

# ------------------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------------------
mean_aucs = {c[4:]: float(auc_df[c].mean()) for c in auc_df.columns if c.startswith("auc_")}
print("\n[5] Mean AUROC across evaluable perturbations:")
for k, v in sorted(mean_aucs.items(), key=lambda x: -x[1]):
    print(f"  {k:30s} {v:.4f}")

summary = {
    "primary_layer": PRIMARY_LAYER,
    "n_perturbations_evaluated": int(len(auc_df)),
    "mean_auc": mean_aucs,
    "lfc_threshold": LFC_THRESHOLD,
    "de_pval_threshold": DE_PVAL_THRESHOLD,
    "min_cells_per_pert": MIN_CELLS_PER_PERT,
    "min_de_positives": MIN_DE_POSITIVES,
}
with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nPHASE 1 COMPLETE — outputs: {OUT_DIR}")
