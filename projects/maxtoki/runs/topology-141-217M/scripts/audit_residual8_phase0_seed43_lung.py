"""Audit residual #8 (full) — re-extract Phase 0 for lung at SEED=43.

Monkey-patches phase0_extract's SEED + OUT_ROOT, runs the lung-only
extraction, then recomputes H123 motif AUROC at all 12 layers and compares
seed-42 vs seed-43 layer-by-layer. Closes the upstream cell-sampling-seed
stability check that the Louvain-seed-only test left open.

~15-20 min wall clock for lung extraction + ~2 min for H123 recomputation.

Output: outputs/seed_stability_phase0/{seed43_lung/, summary.json}
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
OUT = RUN / "outputs/seed_stability_phase0"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))

# Monkey-patch phase0_extract before main run
import phase0_extract
phase0_extract.SEED = 43
phase0_extract.OUT_ROOT = OUT / "seed43"
(OUT / "seed43").mkdir(parents=True, exist_ok=True)

print("[A-residual-8] Re-running Phase 0 lung extraction at SEED=43...")
t0 = time.time()
phase0_extract.main(domain_filter=["lung"])
print(f"  Phase 0 lung re-extraction took {time.time()-t0:.0f}s")

# Now recompute H123 motif AUROC on the new embeddings
from phase78_community_signed_motif import comembership_matrix
from phase6_manifold_distances import pairwise_distances
from phase123_splits_nulls import load_trrust_signed

trrust = load_trrust_signed()
KNN = 12

print("\n[A-residual-8] Recomputing H123 motif AUROC at SEED=43 vs SEED=42 (lung)...")

def compute_h123_auroc(emb_layer, sym_to_idx, trrust_subset, pi, pj, y, louvain_seed=42):
    D = pairwise_distances(emb_layer)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :KNN]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    communities = nx.community.louvain_communities(g, seed=louvain_seed)
    labels = np.zeros(n, dtype=np.int32)
    for ci, comm in enumerate(communities):
        for nd in comm:
            labels[int(nd)] = ci
    com = comembership_matrix(labels)
    feat = np.zeros((n, n), dtype=np.float32)
    for sign, row in zip(trrust_subset["sign"].to_numpy(), trrust_subset.itertuples()):
        if row.tf not in sym_to_idx or row.target not in sym_to_idx:
            continue
        tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
        same_com = bool(com[tf, tgt])
        consistency = 1.0 if (
            (sign > 0 and same_com) or (sign < 0 and not same_com)
        ) else (-1.0 if sign != 0 else 0.0)
        feat[tf, tgt] += consistency
        feat[tgt, tf] += consistency
    score = feat[pi, pj]
    if len(np.unique(y)) < 2: return float("nan")
    try: return float(roc_auc_score(y, score))
    except: return float("nan")


def load_and_score(phase0_dir: Path, p1_dir: Path):
    emb = np.load(phase0_dir / "lung/layer_gene_embeddings_pca20.npy")
    gf = pd.read_csv(phase0_dir / "lung/gene_features.csv")
    pairs = pd.read_csv(p1_dir / "lung/pair_table.csv")
    aucs = []
    for li in range(emb.shape[0]):
        E = emb[li]
        keep = np.linalg.norm(E, axis=1) > 0
        if keep.sum() < 50:
            aucs.append(float("nan")); continue
        E_kept = E[keep]
        mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
        new_idx = -np.ones(len(gf), dtype=np.int64)
        new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
        valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
        P_lay = pairs[valid].reset_index(drop=True).copy()
        P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
        P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
        kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(), mask_g.tolist()) if m]
        sym_to_idx = {s: i for i, s in enumerate(kept_syms)}
        pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
        y = P_lay["is_trrust"].to_numpy().astype(int)
        trrust_in = trrust[trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)].reset_index(drop=True)
        aucs.append(compute_h123_auroc(E_kept, sym_to_idx, trrust_in, pi, pj, y))
    return aucs

# Note: phase1 (pair tables) was built on seed-42 cell sample. For a clean comparison
# we use the SAME phase1 pair tables; the cell-sample seed only affects gene EMBEDDINGS,
# not the pair table (which is symbolic gene-pair).
P1 = RUN / "outputs/phase1"

aucs_seed42 = load_and_score(RUN / "outputs/phase0", P1)
aucs_seed43 = load_and_score(OUT / "seed43", P1)

print("\nLayer | seed42 | seed43 | |Δ|")
diffs = []
for li, (a42, a43) in enumerate(zip(aucs_seed42, aucs_seed43)):
    d = abs(a42 - a43) if not (np.isnan(a42) or np.isnan(a43)) else float("nan")
    diffs.append(d)
    print(f"  L{li:02d}  | {a42:.4f} | {a43:.4f} | {d:.4f}" if not np.isnan(d) else f"  L{li:02d}  | {a42} | {a43} | nan")

diffs_arr = np.array([d for d in diffs if not np.isnan(d)])
summary = {
    "scope": "Lung domain × 12 layers, H123 motif AUROC under SEED=42 vs SEED=43 cell-sample re-extraction",
    "aucs_seed42": [float(a) for a in aucs_seed42],
    "aucs_seed43": [float(a) for a in aucs_seed43],
    "abs_differences": [float(d) for d in diffs],
    "max_abs_difference": float(diffs_arr.max()) if len(diffs_arr) else float("nan"),
    "mean_abs_difference": float(diffs_arr.mean()) if len(diffs_arr) else float("nan"),
    "interpretation": (
        f"Mean |Δ AUROC| across 12 layers: {diffs_arr.mean():.4f} (max {diffs_arr.max():.4f}). "
        + ("**Phase-0 cell-sample-seed STABLE.** H123 conclusion is robust to upstream "
           "cell-sampling stochasticity." if diffs_arr.max() < 0.05 else
           "**Phase-0 cell-sample-seed SENSITIVE.** Original single-seed result needs revision.")
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[A-residual-8] Wrote {OUT}/summary.json")
print(f"[A-residual-8] max |Δ| = {diffs_arr.max():.4f}, mean |Δ| = {diffs_arr.mean():.4f}")
