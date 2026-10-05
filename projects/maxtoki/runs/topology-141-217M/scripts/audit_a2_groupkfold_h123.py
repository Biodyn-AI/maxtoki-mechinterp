"""Audit action A2 — GroupKFold-by-TF and -by-target re-evaluation of
topology-141 H123 motif AUROC.

The original H123 score (`outputs/phase78/h123_signed_motif_degree_strata.csv`)
computes AUROC over all 1308–2544 pairs at once. Pairs share TF (sym_i) and
target (sym_j) endpoints; per pattern P4 in the audit, this can inflate
effective N if shared-endpoint structure is informative.

This audit re-evaluates AUROC under three CV schemes:
  - **No grouping** (the original): all pairs in a single AUROC.
  - **GroupKFold by TF** (sym_i): each fold holds out all pairs from a TF
    group; the H123 feature is recomputed using only the training fold's
    TF→target signed edges (so feature construction respects fold structure).
  - **GroupKFold by target** (sym_j): same with target groups.
  - **Dual-axis disjoint**: TFs in test fold AND targets in test fold are
    BOTH held out from training. The strictest split.

If the no-grouping AUROC is materially higher than the GroupKFold AUROCs,
the original H123 finding is leakage-inflated. If they agree, no leakage.

Run on **immune domain** (the only domain with any positive layers) at
**all 12 layers**. Output: outputs/groupkfold_h123/{summary.json, by_layer.csv}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
OUT = RUN / "outputs/groupkfold_h123"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))

from phase123_splits_nulls import load_trrust_signed
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature,
)
from phase78_community_signed_motif import (
    louvain_communities, comembership_matrix,
)

DOMAIN = "immune"
KNN = 12
N_FOLDS = 5
SEED = 42

print(f"[A2] Loading {DOMAIN}...")
P0 = RUN / f"outputs/phase0/{DOMAIN}"
P1 = RUN / f"outputs/phase1/{DOMAIN}"
emb = np.load(P0 / "layer_gene_embeddings_pca20.npy")
gf = pd.read_csv(P0 / "gene_features.csv")
pairs = pd.read_csv(P1 / "pair_table.csv")
trrust = load_trrust_signed()
n_layers = emb.shape[0]
print(f"  emb {emb.shape}, pairs {len(pairs)}, n_layers {n_layers}")


def compute_motif_feature(emb_layer, pairs_lay, sym_to_idx, trrust_subset, k=KNN):
    """Compute H123-style signed-motif feature using ONLY trrust_subset edges.
    Returns per-pair feature values."""
    D = pairwise_distances(emb_layer)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    labels = louvain_communities(g, seed=42)
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
    pi = pairs_lay["i"].to_numpy(); pj = pairs_lay["j"].to_numpy()
    return feat[pi, pj]


def auc_safe(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    try: return float(roc_auc_score(y, score))
    except: return float("nan")


print(f"\n[A2] Running GroupKFold-by-TF, GroupKFold-by-target, dual-axis CV at all {n_layers} layers...")
rows = []
for li in range(n_layers):
    E = emb[li]
    keep = np.linalg.norm(E, axis=1) > 0
    if keep.sum() < 50:
        continue
    E_kept = E[keep]
    n_genes_kept = int(keep.sum())
    mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
    new_idx = -np.ones(len(gf), dtype=np.int64)
    new_idx[np.where(mask_g)[0]] = np.arange(n_genes_kept)
    valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
    P_lay = pairs[valid].reset_index(drop=True).copy()
    P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
    P_lay["j"] = new_idx[P_lay["j"].to_numpy()]

    kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(),
                                   mask_g.tolist()) if m]
    sym_to_idx = {s: i for i, s in enumerate(kept_syms)}

    trrust_in = trrust[
        trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
    ].reset_index(drop=True)

    pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
    y = P_lay["is_trrust"].to_numpy().astype(int)

    # 1. NO GROUPING (replicates original phase78 H123 motif AUROC)
    feat_full = compute_motif_feature(E_kept, P_lay, sym_to_idx, trrust_in)
    auc_no_group = auc_safe(y, feat_full)

    # Map pair indices back to TF / target symbols
    tf_syms = np.array([kept_syms[i] for i in pi])
    target_syms = np.array([kept_syms[j] for j in pj])
    pair_uses_known_tf = np.array([s in trrust_in["tf"].values for s in tf_syms])

    # 2. GroupKFold by TF (sym_i): partition unique TFs, hold out one fold's TFs
    unique_tfs = np.unique(tf_syms)
    rng = np.random.default_rng(SEED)
    perm_tfs = rng.permutation(unique_tfs)
    tf_folds = np.array_split(perm_tfs, N_FOLDS)
    aucs_groupkfold_tf = []
    for fold_i, held_out_tfs in enumerate(tf_folds):
        tr_trrust = trrust_in[~trrust_in["tf"].isin(held_out_tfs)].reset_index(drop=True)
        feat_tr = compute_motif_feature(E_kept, P_lay, sym_to_idx, tr_trrust)
        # Evaluate on test pairs (those with TF in held-out group)
        test_mask = np.isin(tf_syms, held_out_tfs)
        if test_mask.sum() < 20 or len(np.unique(y[test_mask])) < 2:
            continue
        auc_fold = auc_safe(y[test_mask], feat_tr[test_mask])
        if not np.isnan(auc_fold):
            aucs_groupkfold_tf.append(auc_fold)
    auc_groupkfold_tf_mean = float(np.mean(aucs_groupkfold_tf)) if aucs_groupkfold_tf else float("nan")

    # 3. GroupKFold by target (sym_j)
    unique_targets = np.unique(target_syms)
    perm_targets = rng.permutation(unique_targets)
    target_folds = np.array_split(perm_targets, N_FOLDS)
    aucs_groupkfold_tgt = []
    for fold_i, held_out_targets in enumerate(target_folds):
        # Restrict trrust: keep only edges whose TARGET is in training set
        tr_trrust = trrust_in[~trrust_in["target"].isin(held_out_targets)].reset_index(drop=True)
        feat_tr = compute_motif_feature(E_kept, P_lay, sym_to_idx, tr_trrust)
        test_mask = np.isin(target_syms, held_out_targets)
        if test_mask.sum() < 20 or len(np.unique(y[test_mask])) < 2:
            continue
        auc_fold = auc_safe(y[test_mask], feat_tr[test_mask])
        if not np.isnan(auc_fold):
            aucs_groupkfold_tgt.append(auc_fold)
    auc_groupkfold_tgt_mean = float(np.mean(aucs_groupkfold_tgt)) if aucs_groupkfold_tgt else float("nan")

    # 4. Dual-axis disjoint: hold out BOTH TFs and targets
    # Use the same TF folds + target folds; for each pair (tf_fold_i, tgt_fold_i),
    # hold out pairs where both endpoints are in held-out groups
    aucs_dual = []
    for fold_i in range(N_FOLDS):
        held_tfs = tf_folds[fold_i]
        held_targets = target_folds[fold_i]
        tr_trrust = trrust_in[
            (~trrust_in["tf"].isin(held_tfs)) & (~trrust_in["target"].isin(held_targets))
        ].reset_index(drop=True)
        feat_tr = compute_motif_feature(E_kept, P_lay, sym_to_idx, tr_trrust)
        test_mask = np.isin(tf_syms, held_tfs) & np.isin(target_syms, held_targets)
        if test_mask.sum() < 10 or len(np.unique(y[test_mask])) < 2:
            continue
        auc_fold = auc_safe(y[test_mask], feat_tr[test_mask])
        if not np.isnan(auc_fold):
            aucs_dual.append(auc_fold)
    auc_dual_mean = float(np.mean(aucs_dual)) if aucs_dual else float("nan")

    rows.append({
        "domain": DOMAIN,
        "layer": li,
        "auc_no_grouping": auc_no_group,
        "auc_groupkfold_tf": auc_groupkfold_tf_mean,
        "auc_groupkfold_target": auc_groupkfold_tgt_mean,
        "auc_dual_axis_disjoint": auc_dual_mean,
        "delta_no_group_minus_tf": auc_no_group - auc_groupkfold_tf_mean,
        "delta_no_group_minus_target": auc_no_group - auc_groupkfold_tgt_mean,
        "delta_no_group_minus_dual": auc_no_group - auc_dual_mean,
        "n_test_pairs_dual": int(test_mask.sum()) if 'test_mask' in dir() else 0,
    })
    print(f"  L{li:02d}  no-group={auc_no_group:.3f}  TF={auc_groupkfold_tf_mean:.3f}  "
          f"target={auc_groupkfold_tgt_mean:.3f}  dual={auc_dual_mean:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "by_layer.csv", index=False)

# Summary
summary = {
    "scope": f"H123 motif AUROC under 4 CV schemes (no-grouping, GroupKFold-by-TF, GroupKFold-by-target, dual-axis disjoint), {DOMAIN} domain × {n_layers} layers",
    "n_folds": N_FOLDS,
    "by_layer": df.to_dict(orient="records"),
    "aggregate": {
        "mean_auc_no_grouping": float(df["auc_no_grouping"].mean()),
        "mean_auc_groupkfold_tf": float(df["auc_groupkfold_tf"].mean()),
        "mean_auc_groupkfold_target": float(df["auc_groupkfold_target"].mean()),
        "mean_auc_dual_axis": float(df["auc_dual_axis_disjoint"].mean()),
        "mean_delta_no_group_minus_dual": float(df["delta_no_group_minus_dual"].mean()),
    },
    "interpretation": (
        f"On {DOMAIN}, mean H123 motif AUROC under no-grouping = "
        f"{df['auc_no_grouping'].mean():.3f}; under GroupKFold-by-TF = "
        f"{df['auc_groupkfold_tf'].mean():.3f}; under GroupKFold-by-target = "
        f"{df['auc_groupkfold_target'].mean():.3f}; under dual-axis disjoint = "
        f"{df['auc_dual_axis_disjoint'].mean():.3f}. "
        + ("Group-aware splits give materially LOWER AUROC — the original H123 number "
           "was inflated by shared-endpoint structure. The corrected numbers are the "
           "appropriate ones to report." if (df['auc_no_grouping'].mean() - df['auc_dual_axis_disjoint'].mean()) > 0.02
           else "Group-aware splits give similar AUROC to no-grouping — the original "
                "H123 number is NOT inflated by shared-endpoint structure. The "
                "no-grouping result stands as a valid AUROC estimate.")
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[A2] Aggregate: no-group={df['auc_no_grouping'].mean():.3f}, "
      f"TF={df['auc_groupkfold_tf'].mean():.3f}, "
      f"target={df['auc_groupkfold_target'].mean():.3f}, "
      f"dual={df['auc_dual_axis_disjoint'].mean():.3f}")
print(f"[A2] Wrote {OUT}/summary.json")
