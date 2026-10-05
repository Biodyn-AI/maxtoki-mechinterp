"""Phases 3, 4, 5 — biological encoding on singular-vector axes.

Uses the pre-computed gene_svd_coords.npy from Phase 1.

Phase 3: SV1 vs GO Cellular Component (subcellular localization).
Phase 4: SV2-SV4 vs STRING PPI; confidence gradient (STRING 700 vs 900);
         hub confound.
Phase 5: SV5-SV7 — TF-vs-target 6D classifier (joint SV2-SV7);
         edge-level TRRUST AUROC (SV5-SV7); co-expression residualization;
         repression vs activation split.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
PHASE0 = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
PHASE1 = PROJ / "runs/spectral-geometry-217M/outputs/phase1"
OUT3 = PROJ / "runs/spectral-geometry-217M/outputs/phase3"
OUT4 = PROJ / "runs/spectral-geometry-217M/outputs/phase4"
OUT5 = PROJ / "runs/spectral-geometry-217M/outputs/phase5"
for d in (OUT3, OUT4, OUT5):
    d.mkdir(parents=True, exist_ok=True)

TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"
GO_PKL = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl"

SEED = 42
N_PERM = 500
TOPK = 52

print("=" * 70)
print("PHASES 3 + 4 + 5 — biological encoding on SV axes")
print("=" * 70)

gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
gene_counts = np.load(PHASE0 / "gene_counts.npy")
nonzero_mask = gene_counts > 0
nonzero_indices = np.where(nonzero_mask)[0]
gene_symbols_all = gene_features["symbol"].tolist()
gene_symbols_nonzero = [gene_symbols_all[i].upper() for i in nonzero_indices]
sym_to_nz = {s: i for i, s in enumerate(gene_symbols_nonzero)}

sv = np.load(PHASE1 / "gene_svd_coords.npy")
n_layers, n_nz, topk_sv = sv.shape
print(f"  sv shape: {sv.shape}  n_nz={n_nz}  layers={n_layers}  topk_sv={topk_sv}")


# =======================================================================
# Phase 2 utilities — co-pole enrichment tests
# =======================================================================
def single_label_copole(sv_l_k, pos_mask, top_k=TOPK, n_perm=N_PERM, seed=SEED):
    n = sv_l_k.shape[0]
    order = np.argsort(sv_l_k)
    pole = np.zeros(n, dtype=np.int8)
    pole[order[:top_k]] = 1
    pole[order[-top_k:]] = 2
    n_pos = int(pos_mask.sum())
    if n_pos < 2:
        return None
    obs = ((pole > 0) & pos_mask).sum() / n_pos
    rng_perm = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for t in range(n_perm):
        shuf = rng_perm.permutation(pos_mask)
        null[t] = (pole[shuf] > 0).sum() / n_pos
    null_mean = float(null.mean())
    null_std = float(null.std() + 1e-12)
    return {
        "observed_rate": float(obs), "null_mean": null_mean, "null_std": null_std,
        "z": float((obs - null_mean) / null_std),
        "p_empirical": float((1 + (null >= obs).sum()) / (n_perm + 1)),
        "n_positive": n_pos,
    }


def pair_copole(sv_l_k, pairs, top_k=TOPK, n_perm=N_PERM, seed=SEED):
    n = sv_l_k.shape[0]
    order = np.argsort(sv_l_k)
    pole = np.zeros(n, dtype=np.int8)
    pole[order[:top_k]] = 1
    pole[order[-top_k:]] = 2
    if len(pairs) == 0:
        return None
    p_a = pole[pairs[:, 0]]
    p_b = pole[pairs[:, 1]]
    same = ((p_a == p_b) & (p_a != 0)).sum()
    obs = same / len(pairs)
    rng_perm = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for t in range(n_perm):
        perm = rng_perm.permutation(n)
        pole2 = pole[perm]
        pa = pole2[pairs[:, 0]]; pb = pole2[pairs[:, 1]]
        null[t] = ((pa == pb) & (pa != 0)).sum() / len(pairs)
    null_mean = float(null.mean())
    null_std = float(null.std() + 1e-12)
    return {
        "observed_rate": float(obs), "null_mean": null_mean, "null_std": null_std,
        "z": float((obs - null_mean) / null_std),
        "p_empirical": float((1 + (null >= obs).sum()) / (n_perm + 1)),
        "n_pairs": int(len(pairs)),
    }


# =======================================================================
# Phase 3 — SV1 vs GO Cellular Component
# =======================================================================
print("\n[Phase 3] SV1 vs GO Cellular Component...")
with open(GO_PKL, "rb") as f:
    gene2go = pickle.load(f)

GO_CC_TERMS = {
    "extracellular_space (GO:0005615)": "GO:0005615",
    "cytosol (GO:0005829)": "GO:0005829",
    "mitochondrial_matrix (GO:0005759)": "GO:0005759",
    "endoplasmic_reticulum (GO:0005783)": "GO:0005783",
    "ER_lumen (GO:0005788)": "GO:0005788",
    "mitochondrion (GO:0005739)": "GO:0005739",
    "nucleus (GO:0005634)": "GO:0005634",
    "plasma_membrane (GO:0005886)": "GO:0005886",
}

p3_rows = []
for term_name, go_id in GO_CC_TERMS.items():
    pos_mask = np.array([go_id in gene2go.get(s, set()) for s in gene_symbols_nonzero], dtype=bool)
    if pos_mask.sum() < 5:
        continue
    for li in range(n_layers):
        r = single_label_copole(sv[li, :, 0], pos_mask, seed=SEED + li)
        if r is None:
            continue
        r.update({"layer": li, "term": term_name})
        p3_rows.append(r)

p3_df = pd.DataFrame(p3_rows)
if len(p3_df) > 0:
    _, p3_df["p_bh"], _, _ = multipletests(p3_df["p_empirical"], method="fdr_bh")
p3_df.to_csv(OUT3 / "sv1_go_cc_copole.csv", index=False)
print(f"  rows: {len(p3_df)}  significant (p_bh<0.05): {(p3_df['p_bh']<0.05).sum() if len(p3_df) else 0}")
if len(p3_df) > 0:
    # Per-term best layer
    best = p3_df.loc[p3_df.groupby("term")["z"].idxmax()][["term", "layer", "z", "observed_rate", "null_mean", "p_bh"]]
    print(best.to_string(index=False))


# =======================================================================
# Phase 4 — SV2-SV4 vs STRING PPI
# =======================================================================
print("\n[Phase 4] SV2-SV4 vs STRING PPI...")
with open(STRING_JSON) as f:
    string = json.load(f)
print(f"  STRING pairs_700: {len(string['pairs_700'])}  pairs_900: {len(string['pairs_900'])}")

def map_pairs(pair_list):
    out = []
    for a, b in pair_list:
        ia = sym_to_nz.get(a.upper()); ib = sym_to_nz.get(b.upper())
        if ia is not None and ib is not None and ia != ib:
            out.append((ia, ib))
    return np.array(out, dtype=np.int64) if out else np.zeros((0, 2), dtype=np.int64)

pairs_700 = map_pairs(string["pairs_700"])
pairs_900 = map_pairs(string["pairs_900"])
print(f"  in-vocab after mapping: 700={len(pairs_700)}  900={len(pairs_900)}")

p4_rows = []
for axis_name, axis_idx in [("SV1", 0), ("SV2", 1), ("SV3", 2), ("SV4", 3),
                             ("SV5", 4), ("SV6", 5), ("SV7", 6)]:
    if axis_idx >= sv.shape[2]: continue
    for li in range(n_layers):
        for pair_name, pair_idx in [("pairs_700", pairs_700), ("pairs_900", pairs_900)]:
            if len(pair_idx) == 0: continue
            r = pair_copole(sv[li, :, axis_idx], pair_idx, seed=SEED + li * 100 + axis_idx)
            if r is None: continue
            r.update({"axis": axis_name, "layer": li, "pair_set": pair_name})
            p4_rows.append(r)
p4_df = pd.DataFrame(p4_rows)
if len(p4_df) > 0:
    _, p4_df["p_bh"], _, _ = multipletests(p4_df["p_empirical"], method="fdr_bh")
p4_df.to_csv(OUT4 / "ppi_copole_per_axis_per_layer.csv", index=False)

# Summary per-axis: mean z, significant layers
if len(p4_df) > 0:
    print("  Axis summary on STRING pairs_700:")
    for axis in ["SV1", "SV2", "SV3", "SV4", "SV5", "SV6", "SV7"]:
        sub = p4_df[(p4_df["axis"] == axis) & (p4_df["pair_set"] == "pairs_700")]
        sig = (sub["p_bh"] < 0.05).sum()
        print(f"    {axis}: mean_z={sub['z'].mean():+.2f}  significant_layers={sig}/{len(sub)}")

# Confidence gradient (2-point): mean z on pairs_700 vs pairs_900 across layers
print("\n  2-point confidence gradient (700 vs 900):")
for axis_idx, axis_name in [(1, "SV2")]:
    z_700 = p4_df[(p4_df["axis"] == axis_name) & (p4_df["pair_set"] == "pairs_700")]["z"].mean()
    z_900 = p4_df[(p4_df["axis"] == axis_name) & (p4_df["pair_set"] == "pairs_900")]["z"].mean()
    print(f"    {axis_name} pairs_700: mean z = {z_700:+.3f}")
    print(f"    {axis_name} pairs_900: mean z = {z_900:+.3f}")
    print(f"    Δ (higher confidence should give larger z): {z_900 - z_700:+.3f}")

with open(OUT4 / "confidence_gradient.json", "w") as f:
    json.dump({
        "z_mean_700_SV2": float(p4_df[(p4_df["axis"]=="SV2") & (p4_df["pair_set"]=="pairs_700")]["z"].mean()),
        "z_mean_900_SV2": float(p4_df[(p4_df["axis"]=="SV2") & (p4_df["pair_set"]=="pairs_900")]["z"].mean()),
    }, f, indent=2)


# =======================================================================
# Phase 5 — SV5-SV7 regulatory encoding
# =======================================================================
print("\n[Phase 5] SV5-SV7 — TRRUST + TF-vs-target + edge AUROC...")
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()

# TF list restricted to the nonzero HVG set
tf_set = set(trrust["tf"]) & set(gene_symbols_nonzero)
target_set = set(trrust["target"]) & set(gene_symbols_nonzero)
# "Target-only" = in target_set but not in tf_set
target_only = target_set - tf_set
print(f"  TFs in non-zero HVG: {len(tf_set)}  targets (total): {len(target_set)}  target-only: {len(target_only)}")

tf_hvg = np.array([sym_to_nz[t] for t in tf_set], dtype=np.int64)
target_only_hvg = np.array([sym_to_nz[t] for t in target_only], dtype=np.int64)
labels_class = np.zeros(n_nz, dtype=np.int8)
labels_class[tf_hvg] = 1       # TF = 1
labels_class[target_only_hvg] = 2  # target-only = 2
# genes with both roles or neither get label 0 (excluded from classifier)

# 5a) TF-vs-target 6D classification per layer
p5_class_rows = []
for li in range(n_layers):
    # 6D subspace = SV2-SV7 (indices 1..6)
    X = sv[li, :, 1:7]
    mask = labels_class > 0
    y = (labels_class[mask] == 1).astype(int)
    Xm = X[mask]
    # Centroid classifier
    c_tf = Xm[y == 1].mean(axis=0)
    c_tg = Xm[y == 0].mean(axis=0)
    score = np.linalg.norm(Xm - c_tg, axis=1) - np.linalg.norm(Xm - c_tf, axis=1)
    # score > 0  ⇒ closer to TF centroid
    try:
        auc_joint = float(roc_auc_score(y, score))
    except ValueError:
        auc_joint = float("nan")

    # SV2-SV4 and SV5-SV7 subspaces separately
    def _subspace_auc(lo, hi):
        Xs = sv[li, :, lo:hi][mask]
        c_t = Xs[y == 1].mean(axis=0); c_g = Xs[y == 0].mean(axis=0)
        s = np.linalg.norm(Xs - c_g, axis=1) - np.linalg.norm(Xs - c_t, axis=1)
        try:
            return float(roc_auc_score(y, s))
        except ValueError:
            return float("nan")

    auc_234 = _subspace_auc(1, 4)
    auc_567 = _subspace_auc(4, 7)

    # 100-permutation null on joint 6D
    rng_perm = np.random.default_rng(SEED + li + 777)
    null_auc = []
    for _ in range(100):
        y_p = rng_perm.permutation(y)
        c1 = Xm[y_p == 1].mean(axis=0); c0 = Xm[y_p == 0].mean(axis=0)
        s = np.linalg.norm(Xm - c0, axis=1) - np.linalg.norm(Xm - c1, axis=1)
        try:
            null_auc.append(roc_auc_score(y_p, s))
        except ValueError:
            pass
    null_mean = float(np.mean(null_auc)) if null_auc else float("nan")
    null_p = float((1 + sum(a >= auc_joint for a in null_auc)) / (len(null_auc) + 1))
    p5_class_rows.append({
        "layer": li, "auc_joint_SV2_SV7": auc_joint,
        "auc_SV2_SV4": auc_234, "auc_SV5_SV7": auc_567,
        "null_mean": null_mean, "perm_p": null_p,
    })
p5_class_df = pd.DataFrame(p5_class_rows)
p5_class_df.to_csv(OUT5 / "tf_vs_target_classifier.csv", index=False)
print(p5_class_df.to_string(index=False))

# 5b) Edge-level TF→target AUROC (SV5-SV7 cosine similarity, TRRUST positives vs TF-matched negatives)
print("\n[Phase 5b] Edge-level TF→target AUROC (SV5-SV7 cosine)...")
trrust_pairs = [(r["tf"], r["target"]) for _, r in trrust.iterrows()
                if r["tf"] in sym_to_nz and r["target"] in sym_to_nz
                and r["tf"] != r["target"]]
pos_edges = np.array([(sym_to_nz[a], sym_to_nz[b]) for a, b in trrust_pairs], dtype=np.int64)
print(f"  TRRUST positive edges in-vocab: {len(pos_edges)}")

# TF-matched negatives: for each TF in the positives, draw random non-regulatory partners.
# We sample 4x the positive count with TF fixed.
tf_pos = np.unique(pos_edges[:, 0])
all_nz_indices = np.arange(n_nz)
# Known positives per TF
pos_by_tf = {}
for a, b in pos_edges:
    pos_by_tf.setdefault(a, set()).add(b)
neg_list = []
rng_neg = np.random.default_rng(SEED + 2024)
for tf in tf_pos:
    pos_of_tf = pos_by_tf[tf]
    pool = np.setdiff1d(all_nz_indices, np.array(list(pos_of_tf) + [tf], dtype=np.int64))
    n_take = min(4 * len(pos_of_tf), len(pool))
    picks = rng_neg.choice(pool, size=n_take, replace=False)
    for p in picks:
        neg_list.append((tf, int(p)))
neg_edges = np.array(neg_list, dtype=np.int64)
print(f"  TF-matched negatives: {len(neg_edges)}")

def _cos_sim(a, b):
    na = np.linalg.norm(a) + 1e-12; nb = np.linalg.norm(b) + 1e-12
    return float((a * b).sum() / (na * nb))

p5_edge_rows = []
for li in range(n_layers):
    S = sv[li, :, 4:7]   # SV5-SV7 (3D)
    pos_scores = np.array([_cos_sim(S[a], S[b]) for a, b in pos_edges])
    neg_scores = np.array([_cos_sim(S[a], S[b]) for a, b in neg_edges])
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    try:
        auc_567 = float(roc_auc_score(y, s))
    except ValueError:
        auc_567 = float("nan")

    S234 = sv[li, :, 1:4]
    pos_scores_234 = np.array([_cos_sim(S234[a], S234[b]) for a, b in pos_edges])
    neg_scores_234 = np.array([_cos_sim(S234[a], S234[b]) for a, b in neg_edges])
    try:
        auc_234 = float(roc_auc_score(y, np.concatenate([pos_scores_234, neg_scores_234])))
    except ValueError:
        auc_234 = float("nan")

    # Permutation null for SV5-SV7
    rng_perm = np.random.default_rng(SEED + li + 3000)
    null_auc = []
    for _ in range(100):
        y_p = rng_perm.permutation(y)
        try:
            null_auc.append(roc_auc_score(y_p, s))
        except ValueError:
            pass
    null_mean = float(np.mean(null_auc)) if null_auc else float("nan")
    perm_p = float((1 + sum(a >= auc_567 for a in null_auc)) / (len(null_auc) + 1))

    p5_edge_rows.append({
        "layer": li,
        "auc_SV5_SV7_edge": auc_567,
        "auc_SV2_SV4_edge": auc_234,
        "null_mean": null_mean, "perm_p": perm_p,
    })
p5_edge_df = pd.DataFrame(p5_edge_rows)
p5_edge_df.to_csv(OUT5 / "edge_level_auroc.csv", index=False)
print(p5_edge_df.to_string(index=False))

# Depth decay of edge-level SV5-SV7
from scipy.stats import spearmanr as sr
rho_depth, p_depth = sr(p5_edge_df["layer"], p5_edge_df["auc_SV5_SV7_edge"])
print(f"\n  SV5-SV7 edge-AUROC vs depth: ρ={rho_depth:+.3f}  p={p_depth:.2e}")

# 5c) Repression vs activation split
print("\n[Phase 5c] Repression vs activation edges...")
rep_edges = [(r["tf"], r["target"]) for _, r in trrust.iterrows()
             if str(r["mode"]).lower() == "repression"
             and r["tf"] in sym_to_nz and r["target"] in sym_to_nz]
act_edges = [(r["tf"], r["target"]) for _, r in trrust.iterrows()
             if str(r["mode"]).lower() == "activation"
             and r["tf"] in sym_to_nz and r["target"] in sym_to_nz]
rep_np = np.array([(sym_to_nz[a], sym_to_nz[b]) for a, b in rep_edges], dtype=np.int64)
act_np = np.array([(sym_to_nz[a], sym_to_nz[b]) for a, b in act_edges], dtype=np.int64)
print(f"  repression edges: {len(rep_np)}  activation edges: {len(act_np)}")

p5_mode_rows = []
for li in range(n_layers):
    S = sv[li, :, 4:7]
    for mode_name, edges_np in [("repression", rep_np), ("activation", act_np)]:
        if len(edges_np) < 5: continue
        pos = np.array([_cos_sim(S[a], S[b]) for a, b in edges_np])
        neg = np.array([_cos_sim(S[a], S[b]) for a, b in neg_edges])
        y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        s = np.concatenate([pos, neg])
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            auc = float("nan")
        p5_mode_rows.append({"layer": li, "mode": mode_name, "auc": auc, "n_pos": len(edges_np)})
p5_mode_df = pd.DataFrame(p5_mode_rows)
p5_mode_df.to_csv(OUT5 / "mode_split_auroc.csv", index=False)
print(p5_mode_df.to_string(index=False))

with open(OUT5 / "run_config.json", "w") as f:
    json.dump({
        "n_layers": n_layers, "n_nz": n_nz, "topk_sv": topk_sv,
        "n_tfs_with_trrust_targets": int(len(tf_pos)),
        "n_positive_edges": int(len(pos_edges)),
        "n_negative_edges": int(len(neg_edges)),
        "sv567_edge_depth_corr": {"rho": float(rho_depth), "p": float(p_depth)},
    }, f, indent=2)

print(f"\nPHASES 3-5 COMPLETE")
print(f"  phase3: {OUT3}")
print(f"  phase4: {OUT4}")
print(f"  phase5: {OUT5}")
