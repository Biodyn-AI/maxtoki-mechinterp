"""Phase 4 — 6-condition causal head ablation.

Rerun MaxToki-217M forward pass over the same 2000 Replogle K562 control
cells with specific heads masked. Compute the TRRUST recovery AUROC
for each condition and compare to baseline.

Conditions (matching source-paper Supp Note 14.3):
  1. baseline
  2. top-5 TRRUST-ranked heads
  3. random-5 heads, seed A
  4. random-5 heads, seed B
  5. random-5 heads, seed C
  6. entropy-matched-5 heads (picked to match baseline entropy of top-5)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

PHASE0 = PROJ / "runs/attention-grn-217M/outputs/phase0"
OUT_DIR = PROJ / "runs/attention-grn-217M/outputs/phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPLOGLE_H5 = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

N_CTRL = int(os.environ.get("PHASE4_N_CTRL", "2000"))
N_HVG = 1500
MAX_LEN = 2048
CELL_LINE = "k562"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
TOP_K_HEADS = 5

print("=" * 70)
print(f"PHASE 4 — 6-condition causal head ablation  N_ctrl={N_CTRL}")
print("=" * 70)

# ------------------------------------------------------------------------
# Load Phase 0 edges and build TRRUST ground truth
# ------------------------------------------------------------------------
print("\n[1] Loading Phase 0 artefacts...")
edges_per_head = np.load(PHASE0 / "attention_edges_per_head.npy")  # (L, H, G, G)
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
n_layers, n_heads, n_hvg, _ = edges_per_head.shape
hvg_symbols_upper = [s.upper() for s in gene_features["symbol"]]
symbol_to_hvg = {s: i for i, s in enumerate(hvg_symbols_upper)}

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
n_tfs = int(tf_mask.sum())
print(f"  TRRUST-evaluable TFs in HVG: {n_tfs}")

def _edge_auroc(edge_matrix: np.ndarray) -> float:
    scores, labels = [], []
    for tf_i in np.where(tf_mask)[0]:
        m = np.ones(n_hvg, dtype=bool); m[tf_i] = False
        scores.append(edge_matrix[tf_i][m])
        labels.append(trrust_edges[tf_i][m])
    s = np.concatenate(scores); l = np.concatenate(labels)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    return float(roc_auc_score(l, s))

# Per-head TRRUST AUROC from Phase 0
print("\n[2] Ranking heads by per-head TRRUST AUROC...")
per_head_auroc = np.zeros((n_layers, n_heads), dtype=np.float32)
for li in range(n_layers):
    for hi in range(n_heads):
        per_head_auroc[li, hi] = _edge_auroc(edges_per_head[li, hi])
flat_idx = np.argsort(-per_head_auroc.flatten())
ranked = [
    (int(idx // n_heads), int(idx % n_heads), float(per_head_auroc.flatten()[idx]))
    for idx in flat_idx
]
ranked_df = pd.DataFrame(ranked, columns=["layer", "head", "trrust_auroc"])
ranked_df.to_csv(OUT_DIR / "per_head_trrust_auroc_ranking.csv", index=False)
print("  top-10 heads by TRRUST AUROC:")
print(ranked_df.head(10).to_string(index=False))

# Select top-5 and entropy-matched-5
top5 = ranked[:TOP_K_HEADS]
# Entropy: compute per-head row-entropy of the attention edge matrix and pick
# 5 heads with closest entropy to the top-5 mean (but not in top-5).
def _row_entropy(m):
    """Mean row-entropy of a (G, G) edge matrix. Skips rows with zero sum
    (gene never co-occurred with others in the control set)."""
    row_sum = m.sum(axis=1)
    keep = row_sum > 0
    if keep.sum() == 0:
        return float("nan")
    p = m[keep] / row_sum[keep, None]
    return float(-(p * np.log(p + 1e-12)).sum(axis=1).mean())

entropies = np.zeros((n_layers, n_heads), dtype=np.float32)
for li in range(n_layers):
    for hi in range(n_heads):
        entropies[li, hi] = _row_entropy(edges_per_head[li, hi])
target_ent = float(np.mean([entropies[l, h] for l, h, _ in top5]))
top5_set = {(l, h) for l, h, _ in top5}
cand = []
for li in range(n_layers):
    for hi in range(n_heads):
        if (li, hi) not in top5_set:
            cand.append((abs(entropies[li, hi] - target_ent), li, hi))
cand.sort()
entropy_matched = [(l, h) for _, l, h in cand[:TOP_K_HEADS]]
print(f"  top-5 TRRUST heads: {[(l, h) for l, h, _ in top5]}")
print(f"  entropy-matched-5: {entropy_matched}  "
      f"(target entropy {target_ent:.3f})")

rng = np.random.default_rng(SEED)
def _random5(seed):
    r = np.random.default_rng(seed)
    picks = set()
    while len(picks) < TOP_K_HEADS:
        picks.add((int(r.integers(0, n_layers)), int(r.integers(0, n_heads))))
    return list(picks)
random_A = _random5(SEED + 1001)
random_B = _random5(SEED + 1002)
random_C = _random5(SEED + 1003)

conditions = {
    "baseline":       [],
    "top5_trrust":    [(l, h) for l, h, _ in top5],
    "random5_A":      random_A,
    "random5_B":      random_B,
    "random5_C":      random_C,
    "entropy_matched_5": entropy_matched,
}

def _mask_dict(pairs):
    d = {}
    for l, h in pairs:
        d.setdefault(l, []).append(h)
    return d

# ------------------------------------------------------------------------
# Load data + sample same cells as Phase 0
# ------------------------------------------------------------------------
print(f"\n[3] Loading {N_CTRL} K562 NT control cells (matching Phase-0 sample)...")
t0 = time.time()
with h5py.File(REPLOGLE_H5, "r") as f:
    cl_cats = [(s.decode() if isinstance(s, bytes) else s).lower()
               for s in f["obs"]["cell_line"]["categories"][:]]
    cl_codes = f["obs"]["cell_line"]["codes"][:]
    k562 = (cl_codes == cl_cats.index("k562"))
    pg_cats = [s.decode() if isinstance(s, bytes) else s
               for s in f["obs"]["gene"]["categories"][:]]
    pg_codes = f["obs"]["gene"]["codes"][:]
    nt_codes_set = {i for i, g in enumerate(pg_cats)
                    if "non-targeting" in g.lower() or "nontargeting" in g.lower()}
    ctrl_all = np.where(k562 & np.isin(pg_codes, list(nt_codes_set)))[0]
    # Use the SAME seed as Phase 0 so we hit the same cells.
    ctrl_idx = np.sort(np.random.default_rng(SEED).choice(
        ctrl_all, size=N_CTRL, replace=False))
    n_cells_total, n_genes_total = f["X"].shape
    X_ctrl = np.empty((N_CTRL, n_genes_total), dtype=np.float32)
    for i in range(0, N_CTRL, 100):
        X_ctrl[i:i+100] = f["X"][ctrl_idx[i:i+100], :]
print(f"  loaded {X_ctrl.shape}  ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------------------
# Build tokenizer mapping
# ------------------------------------------------------------------------
print("\n[4] Preparing tokenizer + var mapping...")
import pickle
SYM2ENS_PKL = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
with h5py.File(REPLOGLE_H5, "r") as f:
    var_symbols = [s.decode() if isinstance(s, bytes) else s
                   for s in f["var"]["gene_name_index"][:]]
var_ens = [sym2ens.get(s) for s in var_symbols]

tokenizer = MaxTokiTokenizer()
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)
var_to_hvg = -np.ones(n_genes_total, dtype=np.int64)
for hi, vi in enumerate(gene_features["var_idx"].to_numpy(dtype=np.int64)):
    var_to_hvg[vi] = hi
mt_var_to_hvg = var_to_hvg[var_indices_full]

# ------------------------------------------------------------------------
# Run each condition
# ------------------------------------------------------------------------
print(f"\n[5] Loading MaxToki on {DEVICE}...")
xt = MaxTokiAttentionExtractor(device=DEVICE)

results = []
for cond_name, mask_pairs in conditions.items():
    print(f"\n[Phase 4] === condition: {cond_name} ({len(mask_pairs)} heads masked) ===")
    xt.clear_head_mask()
    if mask_pairs:
        xt.set_head_mask(_mask_dict(mask_pairs))
    attn_sum = np.zeros((n_layers, n_heads, n_hvg, n_hvg), dtype=np.float32)
    pair_counts = np.zeros((n_hvg, n_hvg), dtype=np.int32)

    t0 = time.time()
    for ci in tqdm(range(N_CTRL), ncols=80):
        cell = tokenizer.tokenize_cell(
            X_ctrl[ci], var_indices_full, var_tokens_full, var_medians_full,
            max_len=MAX_LEN,
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
        for li in range(n_layers):
            al = attn[li][0].numpy()
            sub = al[:, pos_idx[:, None], pos_idx[None, :]]
            hvg_ids_this_cell = hvg_pp[pos_idx]
            A, B = np.meshgrid(hvg_ids_this_cell, hvg_ids_this_cell, indexing="ij")
            attn_sum[li, :, A.ravel(), B.ravel()] += sub.reshape(n_heads, -1).T
        A, B = np.meshgrid(hvg_pp[pos_idx], hvg_pp[pos_idx], indexing="ij")
        pair_counts[A.ravel(), B.ravel()] += 1

    denom = pair_counts.astype(np.float32); denom[denom == 0] = np.nan
    edges = attn_sum / denom[None, None]
    edges = np.nan_to_num(edges, nan=0.0).astype(np.float32)
    layer_mean = edges.mean(axis=1)
    # Pick best layer by TRRUST AUROC (same primary layer as Phase 1-3)
    pri_layer = int(os.environ.get("PHASE1_PRIMARY_LAYER", "8"))
    auc_primary = _edge_auroc(layer_mean[pri_layer])
    auc_per_layer = [_edge_auroc(layer_mean[l]) for l in range(n_layers)]
    best_layer = int(np.nanargmax(auc_per_layer))
    print(f"  TRRUST AUROC @primary L{pri_layer} = {auc_primary:.4f}  "
          f"best layer L{best_layer} = {auc_per_layer[best_layer]:.4f}  "
          f"({time.time()-t0:.1f}s)")
    results.append({
        "condition": cond_name,
        "n_heads_masked": len(mask_pairs),
        "masked_heads": str(mask_pairs),
        "trrust_auroc_primary": auc_primary,
        "trrust_auroc_best_layer": float(auc_per_layer[best_layer]),
        "best_layer": best_layer,
        "seconds": float(time.time() - t0),
    })
    np.save(OUT_DIR / f"edges_layer_mean_{cond_name}.npy", layer_mean)

xt.clear_head_mask()

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
print("\n[Phase 4 summary]")
print(res_df.to_string(index=False))

# Deltas vs baseline
baseline_row = res_df[res_df.condition == "baseline"].iloc[0]
res_df["delta_primary"] = res_df["trrust_auroc_primary"] - baseline_row["trrust_auroc_primary"]
res_df["delta_best_layer"] = res_df["trrust_auroc_best_layer"] - baseline_row["trrust_auroc_best_layer"]
res_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "n_ctrl": N_CTRL,
        "primary_layer": int(os.environ.get("PHASE1_PRIMARY_LAYER", "8")),
        "conditions": {k: v for k, v in conditions.items()},
        "n_heads_per_condition": TOP_K_HEADS,
    }, f, indent=2, default=str)
print(f"\nPHASE 4 COMPLETE — outputs: {OUT_DIR}")
