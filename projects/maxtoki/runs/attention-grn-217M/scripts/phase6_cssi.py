"""Phase 6 — CSSI (Cell-State Stratified Interpretability) on K562.

Cluster the 2000 K562 NT control cells into K cell-state groups on HVG
expression, re-run MaxToki-217M forward passes with per-cluster accumulators
of the primary layer's per-head attention, then aggregate the per-cluster
edge matrices via CSSI-max, CSSI-mean, and SCENIC-style weighted mean.
Evaluate each against TRRUST recovery (global AUROC) and per-perturbation
DE AUROC.

This tests whether stratification by cell state rescues regulatory signal
from attention.  If it does, that would contradict the paper's scGPT /
Geneformer finding that CSSI doesn't rescue signal on homogeneous data.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

PHASE0 = PROJ / "runs/attention-grn-217M/outputs/phase0"
PHASE1 = PROJ / "runs/attention-grn-217M/outputs/phase1"
OUT_DIR = PROJ / "runs/attention-grn-217M/outputs/phase6_cssi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

PRIMARY_LAYER = int(os.environ.get("PHASE1_PRIMARY_LAYER", "8"))
N_CLUSTERS = int(os.environ.get("CSSI_K", "10"))
N_CTRL = int(os.environ.get("PHASE0_N_CTRL", "2000"))
N_HVG = 1500
MAX_LEN = 2048
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

print("=" * 70)
print(f"PHASE 6 — CSSI  K={N_CLUSTERS}  primary L{PRIMARY_LAYER}  device={DEVICE}")
print("=" * 70)

# ------------------------------------------------------------------------
# Load the exact 2000 control cells used by Phase 0
# ------------------------------------------------------------------------
print("\n[1] Loading Phase 0 control sample...")
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
n_hvg = len(gene_features)
hvg_symbols_upper = [s.upper() for s in gene_features["symbol"]]
symbol_to_hvg = {s: i for i, s in enumerate(hvg_symbols_upper)}
var_idx = gene_features["var_idx"].to_numpy(dtype=np.int64)
ctrl_cells = pd.read_csv(PHASE0 / "control_cells.csv")
ctrl_idx = ctrl_cells["cell_idx_global"].to_numpy(dtype=np.int64)

ds = load_ds("k562")
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((N_CTRL, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CTRL, 100):
        X_ctrl[i:i+100] = f["X"][ctrl_idx[i:i+100], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
var_ens = [sym2ens.get(s) for s in var_symbols]

# log-normalize HVG matrix for clustering
X_ctrl_hvg = X_ctrl[:, var_idx]
rs = X_ctrl_hvg.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log_hvg = np.log1p(X_ctrl_hvg / rs * 1e4).astype(np.float32)

# ------------------------------------------------------------------------
# KMeans clustering
# ------------------------------------------------------------------------
print(f"\n[2] Clustering 2000 controls into K={N_CLUSTERS} groups (KMeans on HVG)...")
t0 = time.time()
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(X_log_hvg)
cluster = km.labels_.astype(np.int32)
sizes = np.bincount(cluster, minlength=N_CLUSTERS)
print(f"  done in {time.time()-t0:.1f}s  cluster sizes: {sizes.tolist()}")
pd.DataFrame({
    "ctrl_idx_local": np.arange(N_CTRL),
    "cell_idx_global": ctrl_idx,
    "cluster": cluster,
}).to_csv(OUT_DIR / "clusters.csv", index=False)

# ------------------------------------------------------------------------
# Load MaxToki + tokenizer mapping
# ------------------------------------------------------------------------
print(f"\n[3] Loading MaxToki on {DEVICE}...")
tokenizer = MaxTokiTokenizer()
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)
var_to_hvg = -np.ones(ds.n_genes_total, dtype=np.int64)
for hi, vi in enumerate(var_idx):
    var_to_hvg[vi] = hi
mt_var_to_hvg = var_to_hvg[var_indices_full]

xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layers, n_heads = xt.n_layers, xt.n_heads
head_dim = xt.hidden_size // n_heads

# ------------------------------------------------------------------------
# Accumulate primary-layer attention per cluster
# ------------------------------------------------------------------------
print(f"\n[4] Per-cluster forward passes (primary L{PRIMARY_LAYER} only)...")
# (K, H, G, G) float32 = 10 * 8 * 1500² * 4 ≈ 720 MB
attn_sum = np.zeros((N_CLUSTERS, n_heads, n_hvg, n_hvg), dtype=np.float32)
pair_counts = np.zeros((N_CLUSTERS, n_hvg, n_hvg), dtype=np.int32)

per_cell_times = []
for ci in tqdm(range(N_CTRL), ncols=80):
    t0 = time.time()
    cell = tokenizer.tokenize_cell(
        X_ctrl[ci], var_indices_full, var_tokens_full, var_medians_full, max_len=MAX_LEN
    )
    if cell is None:
        continue
    pos = cell.gene_positions
    hvg_pp = np.full(len(pos), -1, dtype=np.int64)
    in_vocab = pos >= 0
    hvg_pp[in_vocab] = mt_var_to_hvg[pos[in_vocab]]
    is_hvg = hvg_pp >= 0
    pos_idx = np.where(is_hvg)[0]
    if len(pos_idx) < 2:
        continue
    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, attn = xt.forward_with_attention(input_ids)
    # Take only the primary layer
    al = attn[PRIMARY_LAYER][0].numpy()  # (H, seq, seq)
    sub = al[:, pos_idx[:, None], pos_idx[None, :]]  # (H, k, k)
    hvg_ids = hvg_pp[pos_idx]
    A, B = np.meshgrid(hvg_ids, hvg_ids, indexing="ij")
    k_cluster = int(cluster[ci])
    attn_sum[k_cluster, :, A.ravel(), B.ravel()] += sub.reshape(n_heads, -1).T
    pair_counts[k_cluster, A.ravel(), B.ravel()] += 1
    per_cell_times.append(time.time() - t0)
    if (ci + 1) % 200 == 0 or (ci + 1) == N_CTRL:
        tqdm.write(f"  [{ci+1}/{N_CTRL}] mean fwd {np.mean(per_cell_times[-200:]):.2f}s")

# finalize per-cluster, per-head edges, mean-over-heads → (K, G, G)
denom = pair_counts.astype(np.float32)
denom[denom == 0] = np.nan
per_cluster_per_head = attn_sum / denom[:, None, :, :]
per_cluster_per_head = np.nan_to_num(per_cluster_per_head, nan=0.0).astype(np.float32)
per_cluster_edges = per_cluster_per_head.mean(axis=1).astype(np.float32)  # (K, G, G)

np.save(OUT_DIR / "per_cluster_edges.npy", per_cluster_edges)
np.save(OUT_DIR / "per_cluster_pair_counts.npy", pair_counts)
print(f"\n  per_cluster_edges: {per_cluster_edges.shape}")

# ------------------------------------------------------------------------
# CSSI aggregations
# ------------------------------------------------------------------------
print("\n[5] Aggregating per-cluster edges (CSSI-max / CSSI-mean / SCENIC-style)...")
cssi_max = per_cluster_edges.max(axis=0)  # (G, G)
cssi_mean = per_cluster_edges.mean(axis=0)
# SCENIC-style: weight each cluster by its size (fraction of total cells)
weights = sizes / sizes.sum()
cssi_scenic = (per_cluster_edges * weights[:, None, None]).sum(axis=0).astype(np.float32)

np.fill_diagonal(cssi_max, 0.0)
np.fill_diagonal(cssi_mean, 0.0)
np.fill_diagonal(cssi_scenic, 0.0)
np.save(OUT_DIR / "cssi_max.npy", cssi_max)
np.save(OUT_DIR / "cssi_mean.npy", cssi_mean)
np.save(OUT_DIR / "cssi_scenic.npy", cssi_scenic)

# Reference: baseline attention at primary layer (Phase 0)
attn_baseline = np.load(PHASE0 / "attention_edges_layer_mean.npy")[PRIMARY_LAYER].copy()
np.fill_diagonal(attn_baseline, 0.0)

# ------------------------------------------------------------------------
# TRRUST recovery AUROC per aggregation
# ------------------------------------------------------------------------
print("\n[6] TRRUST recovery AUROC...")
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_edges = np.zeros((n_hvg, n_hvg), dtype=np.int8)
for _, r in trrust.iterrows():
    i = symbol_to_hvg.get(r["tf"]); j = symbol_to_hvg.get(r["target"])
    if i is not None and j is not None and i != j:
        trrust_edges[i, j] = 1
tf_mask = np.zeros(n_hvg, dtype=bool)
for i in range(n_hvg):
    if trrust_edges[i].sum() >= 3:
        tf_mask[i] = True

def _trrust_auroc(em):
    s, l = [], []
    for tf_i in np.where(tf_mask)[0]:
        m = np.ones(n_hvg, dtype=bool); m[tf_i] = False
        s.append(em[tf_i][m]); l.append(trrust_edges[tf_i][m])
    scores = np.concatenate(s); labels = np.concatenate(l)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))

trrust_results = {
    "attention_baseline_L{}".format(PRIMARY_LAYER): _trrust_auroc(attn_baseline),
    "cssi_max":    _trrust_auroc(cssi_max),
    "cssi_mean":   _trrust_auroc(cssi_mean),
    "cssi_scenic": _trrust_auroc(cssi_scenic),
}
# Per-cluster individual AUROCs (best-case "oracle cluster" upper bound)
per_cluster_aurocs = []
for k in range(N_CLUSTERS):
    em = per_cluster_edges[k].copy()
    np.fill_diagonal(em, 0.0)
    per_cluster_aurocs.append({
        "cluster": k, "size": int(sizes[k]),
        "trrust_auroc": _trrust_auroc(em),
    })
pcdf = pd.DataFrame(per_cluster_aurocs)
pcdf.to_csv(OUT_DIR / "per_cluster_trrust.csv", index=False)
trrust_results["oracle_cluster_max"] = float(pcdf["trrust_auroc"].max())
trrust_results["oracle_cluster_mean"] = float(pcdf["trrust_auroc"].mean())

for k, v in trrust_results.items():
    print(f"  {k:35s} {v:.4f}")

# ------------------------------------------------------------------------
# Per-perturbation DE AUROC (Phase-1-style) for each CSSI aggregation
# ------------------------------------------------------------------------
print("\n[7] Re-using Phase-1 DE masks for per-pert AUROC...")
auc_df = pd.read_csv(PHASE1 / "per_perturbation_auroc.csv")
# We need the DE masks; simplest path — re-derive them quickly using the cached
# sym2ens mapping and Phase-1's methodology.  Instead of re-calling DE, exploit
# the already-saved per_perturbation_auroc.csv which has one row per
# evaluable perturbation; use pert_symbol to look up the source gene and the
# positions of DE targets that already survived Phase-1 criteria.
#
# But the phase-1 csv doesn't include the full DE mask — only the AUROC values.
# We need to re-run Phase-1's DE call to get masks for this comparison.
# Faster: read the masks we stored in Phase 2's pair_dataset.csv (it has
# (pert, target, is_de) for every valid pair).
p2_csv = PROJ / "runs/attention-grn-217M/outputs/phase2/pair_dataset.csv"
if not p2_csv.exists():
    print("  Phase-2 pair_dataset.csv missing; skipping per-pert comparison.")
else:
    pairs = pd.read_csv(p2_csv)
    # Recover pert_hvg_idx per perturbation + build per-pert DE masks
    by_pert = pairs.groupby("pert_idx")
    pert_info = []
    for pid, g in by_pert:
        ph = int(g["pert_hvg_idx"].iloc[0])
        valid = np.zeros(n_hvg, dtype=bool)
        valid[g["target_hvg_idx"].to_numpy(dtype=np.int64)] = True
        is_de = np.zeros(n_hvg, dtype=bool)
        is_de[g.loc[g["label"] == 1, "target_hvg_idx"].to_numpy(dtype=np.int64)] = True
        pert_info.append({"pert_hvg": ph, "valid": valid, "is_de": is_de})
    print(f"  Phase-2 perturbations loaded: {len(pert_info)}")

    def _pert_auroc_per_method(edge_matrix):
        aucs = []
        em = edge_matrix
        for rec in pert_info:
            ph = rec["pert_hvg"]; valid = rec["valid"]; y = rec["is_de"][valid].astype(int)
            pred = em[ph, :][valid]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            try:
                aucs.append(roc_auc_score(y, pred))
            except ValueError:
                continue
        return aucs

    pert_aucs = {
        "attention_baseline": _pert_auroc_per_method(attn_baseline),
        "cssi_max":    _pert_auroc_per_method(cssi_max),
        "cssi_mean":   _pert_auroc_per_method(cssi_mean),
        "cssi_scenic": _pert_auroc_per_method(cssi_scenic),
    }
    per_pert_results = {
        k: {
            "mean": float(np.mean(v)) if v else float("nan"),
            "median": float(np.median(v)) if v else float("nan"),
            "std": float(np.std(v)) if v else float("nan"),
            "n": int(len(v)),
        } for k, v in pert_aucs.items()
    }
    print("  Per-perturbation mean AUROC:")
    for k, v in per_pert_results.items():
        print(f"    {k:25s} {v['mean']:.4f} (median {v['median']:.4f}, n={v['n']})")

    # Wilcoxon vs baseline
    from scipy import stats
    baseline = np.array(pert_aucs["attention_baseline"])
    wilcoxon_results = {}
    for name in ["cssi_max", "cssi_mean", "cssi_scenic"]:
        a = np.array(pert_aucs[name])
        if len(a) != len(baseline) or len(a) == 0:
            continue
        try:
            W, p = stats.wilcoxon(a, baseline, zero_method="zsplit")
        except ValueError:
            W, p = float("nan"), float("nan")
        wilcoxon_results[name] = {
            "mean_delta": float(a.mean() - baseline.mean()),
            "W": float(W), "p": float(p),
        }
    print("\n  Wilcoxon vs baseline attention:")
    for k, v in wilcoxon_results.items():
        print(f"    {k:25s} Δ={v['mean_delta']:+.4f}  p={v['p']:.3e}")

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "CSSI",
        "primary_layer": PRIMARY_LAYER,
        "n_clusters": N_CLUSTERS,
        "n_ctrl": N_CTRL,
        "cluster_sizes": sizes.tolist(),
        "trrust_auroc_results": trrust_results,
        "per_pert_results": per_pert_results if p2_csv.exists() else None,
        "wilcoxon_vs_baseline": wilcoxon_results if p2_csv.exists() else None,
    }, f, indent=2)
print(f"\nPHASE 6 CSSI COMPLETE — outputs: {OUT_DIR}")
