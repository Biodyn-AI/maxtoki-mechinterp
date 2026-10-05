"""Phases 6 + 7 — cell-type marker clustering + B-cell attractor dynamics.

Uses the raw per-layer gene embeddings (not SVD coords) from Phase 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
PHASE0 = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
OUT6 = PROJ / "runs/spectral-geometry-217M/outputs/phase6"
OUT7 = PROJ / "runs/spectral-geometry-217M/outputs/phase7"
for d in (OUT6, OUT7):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

E = np.load(PHASE0 / "layer_gene_embeddings.npy")  # (L, G, H)
counts = np.load(PHASE0 / "gene_counts.npy")
nonzero_mask = counts > 0
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
gene_symbols_all = gene_features["symbol"].tolist()
sym_to_idx = {s.upper(): i for i, s in enumerate(gene_symbols_all)}
nonzero_idx = np.where(nonzero_mask)[0]
n_layers = E.shape[0]

print("=" * 70)
print(f"PHASES 6 + 7 — cell-type + B-cell attractor  E shape {E.shape}")
print("=" * 70)


# ===================================================================
# Phase 6 — cell-type marker clustering (AUROC pairs within vs across type)
# ===================================================================
MARKERS = {
    "B cell":       ["CD19", "CD79A", "MS4A1", "BLK", "VPREB3", "FCRL1", "PAX5"],
    "T cell":       ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "LCK", "ZAP70",
                     "TCF7", "LEF1", "TBX21", "GATA3"],
    "NK cell":      ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY"],
    "Macrophage":   ["CD68", "CD163", "MRC1", "MSR1", "MERTK"],
    "Monocyte":     ["CD14", "LYZ", "VCAN", "S100A8", "S100A9"],
    "Dendritic":    ["CLEC9A", "CLEC10A", "CD1C", "FCER1A", "IRF8"],
    "Myeloid":      ["ELANE", "MPO", "AZU1", "DEFA3", "CSF3R"],
}

# Map markers to nonzero HVG positions
marker_idx = {}
for ct, lst in MARKERS.items():
    idxs = [sym_to_idx[s] for s in lst if s.upper() in {x.upper() for x in gene_symbols_all}
            and sym_to_idx[s] in nonzero_idx if False]  # broken; fix below

marker_idx = {}
for ct, lst in MARKERS.items():
    idxs = []
    for s in lst:
        i = sym_to_idx.get(s.upper())
        if i is not None and nonzero_mask[i]:
            idxs.append(i)
    if len(idxs) >= 2:
        marker_idx[ct] = idxs

print("[Phase 6] marker counts per cell type:")
for ct, idxs in marker_idx.items():
    print(f"    {ct}: {len(idxs)}")

# For each layer, compute pairwise Euclidean distances among all markers
# concat across types, then AUROC for within-type vs across-type pairs
all_markers = []
type_of_marker = []
for ct, idxs in marker_idx.items():
    for i in idxs:
        all_markers.append(i)
        type_of_marker.append(ct)
all_markers = np.array(all_markers, dtype=np.int64)
type_of_marker = np.array(type_of_marker)

p6_rows = []
for li in range(n_layers):
    X = E[li][all_markers]
    D = squareform(pdist(X, metric="euclidean"))
    # pairs: upper triangle only
    n = len(all_markers)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dists = D[idx_i, idx_j]
    same = (type_of_marker[idx_i] == type_of_marker[idx_j]).astype(int)
    if same.sum() == 0 or same.sum() == len(same):
        continue
    # smaller distance ⇒ more likely same type ⇒ AUROC uses -dist
    auc = float(roc_auc_score(same, -dists))
    p6_rows.append({"layer": li, "n_pairs": len(dists), "n_same": int(same.sum()),
                    "auc_same_vs_diff_type": auc})
p6_df = pd.DataFrame(p6_rows)
p6_df.to_csv(OUT6 / "cell_type_clustering_auroc.csv", index=False)
print("\n[Phase 6] AUROC per layer (within vs across cell type):")
print(p6_df.to_string(index=False))

with open(OUT6 / "run_config.json", "w") as f:
    json.dump({
        "marker_sets": {ct: MARKERS[ct] for ct in marker_idx},
        "marker_sizes_in_vocab": {ct: len(v) for ct, v in marker_idx.items()},
        "mean_auc": float(p6_df["auc_same_vs_diff_type"].mean()),
        "max_auc": float(p6_df["auc_same_vs_diff_type"].max()),
        "best_layer": int(p6_df["layer"].iloc[p6_df["auc_same_vs_diff_type"].idxmax()]),
    }, f, indent=2)


# ===================================================================
# Phase 7 — B-cell attractor dynamics
# ===================================================================
print("\n[Phase 7] B-cell attractor dynamics...")

B_MARKERS = ["CD19", "CD79A", "MS4A1", "BLK", "VPREB3", "FCRL1", "PAX5"]
GC_TFS = ["BATF", "BACH2", "PAX5"]
PLASMA_TFS = ["IRF4", "IRF8"]
WATCH = ["PAX5", "BATF", "BACH2", "BCL6", "PRDM1", "IRF4", "IRF8"]
METABOLIC_REF = ["NAMPT", "GLUL", "PFKFB3", "HK1", "HK2", "PKM", "LDHA", "MTOR",
                 "STAT3", "GAPDH", "ENO1", "PGK1", "PGAM1", "TKT"]

def _resolve(symbol_list):
    out = {}
    for s in symbol_list:
        i = sym_to_idx.get(s.upper())
        if i is not None and nonzero_mask[i]:
            out[s] = i
    return out

b_marker_idx = _resolve(B_MARKERS)
gc_tf_idx = _resolve(GC_TFS)
plasma_tf_idx = _resolve(PLASMA_TFS)
watch_idx = _resolve(WATCH)
metabolic_idx = _resolve(METABOLIC_REF)
print(f"  B-markers in vocab: {len(b_marker_idx)}/{len(B_MARKERS)}: {list(b_marker_idx)}")
print(f"  watch genes: {list(watch_idx)}")
print(f"  metabolic reference: {list(metabolic_idx)}")

# 7a) Per-layer B-cell centroid; rank of each watched gene to centroid;
#     distance to PAX5
pax5_i = watch_idx.get("PAX5")
p7_rows = []
for li in range(n_layers):
    Ei = E[li]
    # centroid = mean of B markers
    if not b_marker_idx:
        break
    centroid = Ei[list(b_marker_idx.values())].mean(axis=0)
    # Rank each gene in non-zero HVG set by distance to centroid
    dists_to_c = np.linalg.norm(Ei[nonzero_idx] - centroid, axis=1)
    rank_order = np.argsort(dists_to_c)
    # invert: rank[hvg_idx] = position 1-indexed
    inv_rank = np.empty_like(rank_order)
    inv_rank[rank_order] = np.arange(1, len(rank_order) + 1)
    # rank for each watched gene (hvg_idx → nonzero_idx position)
    nz_to_pos = {int(h): p for p, h in enumerate(nonzero_idx)}
    row = {"layer": li}
    for name, gi in watch_idx.items():
        pos = nz_to_pos[gi]
        rank_val = int(inv_rank[pos])
        row[f"rank_{name}"] = rank_val
        if pax5_i is not None and name != "PAX5":
            d = float(np.linalg.norm(Ei[gi] - Ei[pax5_i]))
            row[f"dist_{name}_to_PAX5"] = d
    # Precision@10: among 10 nearest to centroid, how many are B-cell markers?
    top10_hvg = nonzero_idx[rank_order[:10]]
    b_set = set(b_marker_idx.values())
    row["prec_at_10_b"] = int(sum(int(i in b_set) for i in top10_hvg)) / max(len(b_marker_idx), 1)
    # GC-plasma angle at B-cell centroid
    if len(gc_tf_idx) >= 2 and len(plasma_tf_idx) >= 1:
        gc_vec = Ei[list(gc_tf_idx.values())].mean(axis=0) - centroid
        plasma_vec = Ei[list(plasma_tf_idx.values())].mean(axis=0) - centroid
        cos = float((gc_vec * plasma_vec).sum() / (
            (np.linalg.norm(gc_vec) + 1e-12) * (np.linalg.norm(plasma_vec) + 1e-12)
        ))
        row["gc_plasma_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    # BCL6 metabolic neighbourhood overlap
    bcl6 = sym_to_idx.get("BCL6")
    if bcl6 is not None and nonzero_mask[bcl6]:
        ds = np.linalg.norm(Ei[nonzero_idx] - Ei[bcl6], axis=1)
        top20_hvg = set(nonzero_idx[np.argsort(ds)[1:21]].tolist())  # skip self
        overlap_m = len(top20_hvg & set(metabolic_idx.values())) if metabolic_idx else 0
        overlap_b = len(top20_hvg & set(b_marker_idx.values())) if b_marker_idx else 0
        row["bcl6_top20_metabolic_overlap"] = overlap_m
        row["bcl6_top20_bcell_overlap"] = overlap_b
    p7_rows.append(row)
p7_df = pd.DataFrame(p7_rows)
p7_df.to_csv(OUT7 / "b_cell_attractor_trajectory.csv", index=False)
print("\n[Phase 7] B-cell attractor trajectory:")
print(p7_df.to_string(index=False))

# Depth correlations for the key trajectories
summary = {}
for name in watch_idx:
    col = f"rank_{name}"
    if col in p7_df:
        rho, p = spearmanr(p7_df["layer"], p7_df[col])
        summary[f"rank_{name}_vs_depth"] = {"rho": float(rho), "p": float(p)}
    if f"dist_{name}_to_PAX5" in p7_df:
        rho, p = spearmanr(p7_df["layer"], p7_df[f"dist_{name}_to_PAX5"])
        summary[f"dist_{name}_to_PAX5_vs_depth"] = {"rho": float(rho), "p": float(p)}
if "gc_plasma_angle_deg" in p7_df:
    rho, p = spearmanr(p7_df["layer"], p7_df["gc_plasma_angle_deg"])
    summary["gc_plasma_angle_vs_depth"] = {"rho": float(rho), "p": float(p)}

# 7b) Lineage-specific intrinsic dimensionality (B-cell, T-cell, myeloid)
def _twonn(X, seed=0):
    from scipy.spatial.distance import cdist
    n = X.shape[0]
    if n < 4: return float("nan")
    rng = np.random.default_rng(seed)
    D = cdist(X, X); D.sort(axis=1)
    r1, r2 = D[:, 1], D[:, 2]
    ok = (r1 > 1e-9) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    mu.sort()
    F = np.arange(1, len(mu) + 1) / len(mu)
    mask = F < 0.9
    if mask.sum() < 3: return float("nan")
    x = np.log(mu[mask]); y = -np.log(1 - F[mask] + 1e-12)
    d, _ = np.polyfit(x, y, 1)
    return float(d)

lineage_markers = {
    "B cell": list(b_marker_idx.values()),
    "T cell": [sym_to_idx.get(s.upper()) for s in MARKERS["T cell"]
               if sym_to_idx.get(s.upper()) is not None and nonzero_mask[sym_to_idx[s.upper()]]],
    "Myeloid": [sym_to_idx.get(s.upper()) for s in MARKERS["Myeloid"]
                if sym_to_idx.get(s.upper()) is not None and nonzero_mask[sym_to_idx[s.upper()]]],
}
twonn_rows = []
for ct, idxs in lineage_markers.items():
    for li in range(n_layers):
        d = _twonn(E[li][idxs], seed=SEED + li)
        twonn_rows.append({"cell_type": ct, "layer": li, "n_markers": len(idxs), "twonn": d})
twonn_df = pd.DataFrame(twonn_rows)
twonn_df.to_csv(OUT7 / "lineage_twonn_by_layer.csv", index=False)
print("\n[Phase 7b] Lineage-specific intrinsic dimensionality:")
print(twonn_df.to_string(index=False))
for ct in lineage_markers:
    sub = twonn_df[twonn_df["cell_type"] == ct]
    rho, p = spearmanr(sub["layer"], sub["twonn"])
    summary[f"twonn_{ct}_vs_depth"] = {"rho": float(rho), "p": float(p)}

with open(OUT7 / "depth_correlations.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nPHASES 6+7 COMPLETE — outputs: {OUT6}, {OUT7}")
