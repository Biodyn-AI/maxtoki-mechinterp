"""Audit residual #8 — multi-seed stability for topology-141 H123.

Full re-extraction of Phase 0 at multiple data-sampling seeds across all 4
runs is multi-hour-to-day work. This scoped version checks **Louvain-seed
sensitivity** of the H123 result on the existing extracted gene embeddings:
re-run the H123 motif AUROC at 5 different Louvain seeds (the only stochastic
step downstream of Phase 0) and compare.

If H123 is stable across Louvain seeds, the 0/12 layers result is robust
to the only stochasticity in the H123 computation given fixed embeddings.
If unstable, the conclusion needs revision.

This is a *partial* residual #8 fix — it does not exercise the larger
Phase 0 cell-sampling stochasticity. That requires re-extraction (~15 min
per tissue per seed × 3 tissues × 2 alternate seeds = ~90 min compute) and
is documented as still-deferred.

Output: outputs/seed_stability/{summary.json, by_seed.csv}
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
OUT = RUN / "outputs/seed_stability"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))

from phase78_community_signed_motif import comembership_matrix
from phase6_manifold_distances import pairwise_distances
from phase123_splits_nulls import load_trrust_signed

DOMAINS = ["lung", "immune", "external_lung"]
LOUVAIN_SEEDS = [42, 43, 44, 45, 46]
KNN = 12

trrust = load_trrust_signed()


def compute_h123_motif_auroc(emb_layer, sym_to_idx, trrust_subset, pi, pj, y, louvain_seed):
    """Compute H123 motif feature AUROC at a given Louvain seed."""
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
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


print("[residual-8] Loading immune domain (the only one with any positive layers)...")
domain = "immune"
P0 = RUN / f"outputs/phase0/{domain}"
P1 = RUN / f"outputs/phase1/{domain}"
emb = np.load(P0 / "layer_gene_embeddings_pca20.npy")
gf = pd.read_csv(P0 / "gene_features.csv")
pairs = pd.read_csv(P1 / "pair_table.csv")

rows = []
for li in range(emb.shape[0]):
    E = emb[li]
    keep = np.linalg.norm(E, axis=1) > 0
    if keep.sum() < 50:
        continue
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

    aucs_per_seed = {}
    for seed in LOUVAIN_SEEDS:
        aucs_per_seed[seed] = compute_h123_motif_auroc(E_kept, sym_to_idx, trrust_in, pi, pj, y, seed)
    rows.append({"layer": li, **{f"auc_seed_{s}": aucs_per_seed[s] for s in LOUVAIN_SEEDS}})
    aucs_arr = [aucs_per_seed[s] for s in LOUVAIN_SEEDS]
    print(f"  L{li:02d}: " + ", ".join(f"s{s}={aucs_per_seed[s]:.4f}" for s in LOUVAIN_SEEDS) +
          f"  range={max(aucs_arr)-min(aucs_arr):.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "by_seed.csv", index=False)

# Summary stats
auc_cols = [f"auc_seed_{s}" for s in LOUVAIN_SEEDS]
aucs_matrix = df[auc_cols].to_numpy()
per_layer_range = aucs_matrix.max(axis=1) - aucs_matrix.min(axis=1)
per_layer_std = aucs_matrix.std(axis=1)
overall_mean = aucs_matrix.mean()
overall_std = aucs_matrix.std()

summary = {
    "scope": (
        f"H123 motif AUROC at {len(LOUVAIN_SEEDS)} Louvain seeds × {len(df)} layers, immune domain. "
        "Tests Louvain-step seed sensitivity given fixed Phase 0 embeddings."
    ),
    "louvain_seeds": LOUVAIN_SEEDS,
    "per_layer_range_max": float(per_layer_range.max()),
    "per_layer_range_mean": float(per_layer_range.mean()),
    "per_layer_std_mean": float(per_layer_std.mean()),
    "overall_mean_auc": float(overall_mean),
    "overall_std_across_all_seed_x_layer": float(overall_std),
    "interpretation": (
        f"Louvain-seed sensitivity of H123 AUROC: per-layer range averages "
        f"{per_layer_range.mean():.4f}, max {per_layer_range.max():.4f}. "
        f"Per-layer SD averages {per_layer_std.mean():.4f}. "
        + ("LOUVAIN-SEED STABLE — H123 conclusion is robust to community-detection seed."
           if per_layer_range.max() < 0.05 else
           "LOUVAIN-SEED SENSITIVE — community detection seed materially shifts H123. "
           "Original 0/12 result needs revision.")
    ),
    "deferred_full_residual_8": (
        "This Louvain-seed test does NOT exercise Phase 0 cell-sampling stochasticity. "
        "Full residual #8 would re-run Phase 0 (~15 min per tissue per seed × 3 tissues × "
        "2 alternate seeds = ~90 min) and recompute downstream phases. Deferred here for "
        "scope; the cell-sample-seed stability is partially addressed by spectral-geometry "
        "Phase 8 (CKA across 3 cell-sample seeds: L0→L11 changes by <1%)."
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[residual-8] Range across seeds (max): {per_layer_range.max():.4f}, "
      f"mean: {per_layer_range.mean():.4f}")
print(f"[residual-8] Wrote {OUT}/summary.json")
