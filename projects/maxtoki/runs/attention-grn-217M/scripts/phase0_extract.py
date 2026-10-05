"""Phase 0 — MaxToki-217M attention extraction on Replogle K562 control cells.

Produces the Phase-0 artefact set required by the attention-GRN pipeline:

  outputs/phase0/
    attention_edges_per_head.npy    (n_layers, n_heads, n_hvg, n_hvg) float32
    attention_edges_layer_mean.npy  (n_layers, n_hvg, n_hvg) float32
    attention_pair_counts.npy       (n_hvg, n_hvg) int32
    spearman_edges.npy              (n_hvg, n_hvg) float32
    gene_features.csv               hvg-level mean_expr, variance, dropout, tf_out_degree
    hvg_gene_table.csv              symbol, ensembl_id, maxtoki_token_id, var_idx
    control_cells.csv               cell metadata for the 2000 cells used
    run_config.json                 parameters + timings
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
import torch
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

# Repo imports
PROJ = Path(__file__).resolve().parents[3]      # projects/maxtoki
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
REPLOGLE_H5 = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
SYM2ENS_PKL = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"
TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

OUT_DIR = PROJ / "runs/attention-grn-217M/outputs/phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import os
N_CTRL = int(os.environ.get("PHASE0_N_CTRL", "2000"))
N_HVG = int(os.environ.get("PHASE0_N_HVG", "1500"))
MAX_LEN = int(os.environ.get("PHASE0_MAX_LEN", "2048"))
CELL_LINE = "k562"     # filter the concatenated file to K562 only
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
_smoke = os.environ.get("PHASE0_SMOKE") == "1"
if _smoke:
    N_CTRL = 5
    print("*** SMOKE MODE: N_CTRL=5 ***")

print("=" * 70)
print("PHASE 0 — MaxToki-217M attention extraction on Replogle K562")
print(f"  N_ctrl={N_CTRL}  N_hvg={N_HVG}  max_len={MAX_LEN}  device={DEVICE}")
print("=" * 70)

rng = np.random.default_rng(SEED)
t_phase = time.time()

# ------------------------------------------------------------------------
# Step 1 — Load Replogle K562 control cells and metadata
# ------------------------------------------------------------------------
print("\n[1] Loading Replogle metadata...")
t0 = time.time()
with h5py.File(REPLOGLE_H5, "r") as f:
    n_cells_total, n_genes_total = f["X"].shape
    # var gene symbols
    var_symbols = [
        s.decode() if isinstance(s, bytes) else s
        for s in f["var"]["gene_name_index"][:]
    ]
    # cell_line categorical
    cl_cats = [
        (s.decode() if isinstance(s, bytes) else s).lower()
        for s in f["obs"]["cell_line"]["categories"][:]
    ]
    cl_codes = f["obs"]["cell_line"]["codes"][:]
    # perturbation gene categorical
    pg_cats = [
        s.decode() if isinstance(s, bytes) else s
        for s in f["obs"]["gene"]["categories"][:]
    ]
    pg_codes = f["obs"]["gene"]["codes"][:]

print(f"  total cells: {n_cells_total:,}  total genes: {n_genes_total:,}")
print(f"  cell lines: {cl_cats}")

# K562 mask
k562_code = cl_cats.index(CELL_LINE)
k562_mask = (cl_codes == k562_code)
print(f"  K562 cells: {int(k562_mask.sum()):,}")

# Non-targeting mask
nt_candidates = [
    i for i, g in enumerate(pg_cats)
    if "non-targeting" in g.lower() or "nontargeting" in g.lower()
]
nt_mask = np.isin(pg_codes, np.array(nt_candidates, dtype=np.int64))
print(f"  non-targeting cells: {int(nt_mask.sum()):,}")

control_mask = k562_mask & nt_mask
ctrl_all_idx = np.where(control_mask)[0]
print(f"  K562 non-targeting cells: {len(ctrl_all_idx):,}")
if len(ctrl_all_idx) < N_CTRL:
    raise RuntimeError(f"not enough K562 control cells: {len(ctrl_all_idx)} < {N_CTRL}")

ctrl_idx = np.sort(rng.choice(ctrl_all_idx, size=N_CTRL, replace=False))
print(f"  sampled: {len(ctrl_idx):,}")

print(f"  step time: {time.time()-t0:.1f}s")

# ------------------------------------------------------------------------
# Step 2 — Load control expression matrix
# ------------------------------------------------------------------------
print("\n[2] Loading control expression matrix...")
t0 = time.time()
with h5py.File(REPLOGLE_H5, "r") as f:
    X_ctrl = np.empty((N_CTRL, n_genes_total), dtype=np.float32)
    block = 100
    for i in range(0, N_CTRL, block):
        end = min(i + block, N_CTRL)
        rows = ctrl_idx[i:end]
        # h5py fancy indexing requires sorted unique rows; they are sorted.
        X_ctrl[i:end] = f["X"][rows, :]
print(f"  loaded shape: {X_ctrl.shape}  "
      f"mean nz/cell: {float((X_ctrl > 0).sum(axis=1).mean()):.0f}  "
      f"step time: {time.time()-t0:.1f}s")

# Normalize to log1p(CPM/1e4) for HVG selection + gene-level features
print("\n[2b] Normalizing for HVG selection...")
row_sums = X_ctrl.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
X_log = np.log1p(X_ctrl / row_sums * 1e4)

# ------------------------------------------------------------------------
# Step 3 — Select HVG restricted to MaxToki vocabulary
# ------------------------------------------------------------------------
print("\n[3] Selecting HVG ∩ MaxToki vocab...")
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)

tokenizer = MaxTokiTokenizer()
mt_vocab = set(tokenizer.gene_token_dict.keys())

# variance of log-normalized expression
gene_var_all = X_log.var(axis=0)
# map each var gene to ensembl, check membership
var_ens = [sym2ens.get(s) for s in var_symbols]
eligible = np.array(
    [e is not None and e in mt_vocab for e in var_ens],
    dtype=bool,
)
print(f"  var genes in MaxToki vocab: {int(eligible.sum())}/{len(var_symbols)}")

# Top-N_HVG by variance, restricted to eligible
var_rank = np.argsort(-gene_var_all)
hvg_indices: list[int] = []
for gi in var_rank:
    if eligible[gi]:
        hvg_indices.append(int(gi))
        if len(hvg_indices) >= N_HVG:
            break
if len(hvg_indices) < N_HVG:
    raise RuntimeError(f"Only {len(hvg_indices)} eligible HVG; need {N_HVG}")
hvg_indices_np = np.array(sorted(hvg_indices), dtype=np.int64)

hvg_symbols = [var_symbols[i] for i in hvg_indices_np]
hvg_ens = [var_ens[i] for i in hvg_indices_np]
hvg_tokens = np.array(
    [tokenizer.gene_token_dict[e] for e in hvg_ens], dtype=np.int64
)
# sanity
assert len(set(hvg_tokens.tolist())) == len(hvg_tokens)
print(f"  HVG list built: {len(hvg_symbols)} genes, all unique token ids")

X_ctrl_hvg = X_ctrl[:, hvg_indices_np]          # raw counts
X_log_hvg = X_log[:, hvg_indices_np]            # log-normalized

# Gene-level features in HVG space
gene_mean = X_log_hvg.mean(axis=0)
gene_var = X_log_hvg.var(axis=0)
gene_dropout = (X_ctrl_hvg == 0).mean(axis=0)

# TF out-degree from TRRUST
trrust = pd.read_csv(
    TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"]
)
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
tf_out = trrust.groupby("tf")["target"].nunique().to_dict()
tf_out_degree = np.array(
    [float(tf_out.get(s.upper(), 0)) for s in hvg_symbols], dtype=np.float32
)

gene_features = pd.DataFrame({
    "hvg_idx": np.arange(len(hvg_symbols)),
    "symbol": hvg_symbols,
    "ensembl_id": hvg_ens,
    "maxtoki_token_id": hvg_tokens,
    "var_idx": hvg_indices_np,
    "mean_expr": gene_mean,
    "variance": gene_var,
    "dropout_rate": gene_dropout,
    "tf_out_degree": tf_out_degree,
})
gene_features.to_csv(OUT_DIR / "gene_features.csv", index=False)
gene_features[["hvg_idx", "symbol", "ensembl_id", "maxtoki_token_id", "var_idx"]].to_csv(
    OUT_DIR / "hvg_gene_table.csv", index=False
)
print(f"  gene_features.csv, hvg_gene_table.csv written")

# Build per-var fast lookup: var_idx -> hvg_idx (or -1 if not HVG)
var_to_hvg = -np.ones(n_genes_total, dtype=np.int64)
for hi, vi in enumerate(hvg_indices_np):
    var_to_hvg[vi] = hi

# ------------------------------------------------------------------------
# Step 4 — Same-cells Spearman baseline on HVG expression
# ------------------------------------------------------------------------
print("\n[4] Spearman correlation baseline...")
t0 = time.time()
X_ranked = np.apply_along_axis(rankdata, 0, X_log_hvg).astype(np.float32)
X_ranked -= X_ranked.mean(axis=0, keepdims=True)
X_ranked /= (X_ranked.std(axis=0, keepdims=True) + 1e-12)
spearman_edges = (X_ranked.T @ X_ranked) / X_log_hvg.shape[0]
spearman_edges = spearman_edges.astype(np.float32)
np.save(OUT_DIR / "spearman_edges.npy", spearman_edges)
print(f"  shape: {spearman_edges.shape}  step time: {time.time()-t0:.1f}s")

# ------------------------------------------------------------------------
# Step 5 — Load MaxToki and set up tokenizer mapping for var genes
# ------------------------------------------------------------------------
print("\n[5] Loading MaxToki-217M...")
t0 = time.time()
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
print(f"  loaded in {time.time()-t0:.1f}s  "
      f"L={xt.n_layers} H={xt.n_heads} D={xt.hidden_size}")

# Tokenizer state for the full var set (only genes in MT vocab will be kept)
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)
print(f"  var mapped to MT: {len(var_indices_full)}/{n_genes_total}")

# For each var index that is in MT vocab, its hvg_idx (else -1)
mt_var_to_hvg = var_to_hvg[var_indices_full]   # int64
n_layers = xt.n_layers
n_heads = xt.n_heads
n_hvg = len(hvg_symbols)

# Accumulators (float32). Only (layer, head, hvg, hvg) triples are populated.
attn_sum = np.zeros((n_layers, n_heads, n_hvg, n_hvg), dtype=np.float32)
pair_counts = np.zeros((n_hvg, n_hvg), dtype=np.int32)
per_cell_seq_lens = []
per_cell_times = []

# ------------------------------------------------------------------------
# Step 6 — Cell-by-cell forward pass + projection
# ------------------------------------------------------------------------
print("\n[6] Running forward passes (this is the slow step)...")
for ci in tqdm(range(N_CTRL), ncols=80):
    t0 = time.time()
    cell = tokenizer.tokenize_cell(
        X_ctrl[ci], var_indices_full, var_tokens_full, var_medians_full,
        max_len=MAX_LEN,
    )
    if cell is None:
        continue

    # gene_positions is per-token but indexed into var_indices_full, not HVG.
    # Recover the HVG index per position via mt_var_to_hvg.
    # <bos>/<eos> positions are already -1.
    pos = cell.gene_positions.copy()
    seq_len = len(pos)
    hvg_per_pos = np.full(seq_len, -1, dtype=np.int64)
    in_vocab = pos >= 0
    hvg_per_pos[in_vocab] = mt_var_to_hvg[pos[in_vocab]]
    is_hvg = hvg_per_pos >= 0
    hvg_pos_indices = np.where(is_hvg)[0]
    if len(hvg_pos_indices) < 2:
        continue

    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, attn = xt.forward_with_attention(input_ids)

    # attn: tuple[Tensor(1, H, seq, seq)]  float32 on CPU
    # Slice to HVG positions only (square submatrix)
    for li in range(n_layers):
        al = attn[li][0].numpy()     # (H, seq, seq)
        sub = al[:, hvg_pos_indices[:, None], hvg_pos_indices[None, :]]  # (H, k, k)
        hvg_ids_this_cell = hvg_per_pos[hvg_pos_indices]  # (k,)
        # scatter-add into global accumulator
        # attn_sum[:, :, A, B] += sub[:, a, b]  where A = hvg_ids_this_cell[a]
        # Use advanced indexing with meshgrids.
        A, B = np.meshgrid(hvg_ids_this_cell, hvg_ids_this_cell, indexing="ij")
        attn_sum[li, :, A.ravel(), B.ravel()] += sub.reshape(n_heads, -1).T  # (k*k, H)

    # pair_counts: add once per cell (shared across layers)
    A, B = np.meshgrid(hvg_ids_this_cell, hvg_ids_this_cell, indexing="ij")
    pair_counts[A.ravel(), B.ravel()] += 1

    per_cell_seq_lens.append(seq_len)
    per_cell_times.append(time.time() - t0)

    # Periodic checkpoint to disk so we recover from interruptions
    if (ci + 1) % 200 == 0 or (ci + 1) == N_CTRL:
        mean_fwd = float(np.mean(per_cell_times[-200:]))
        tqdm.write(
            f"  [{ci+1}/{N_CTRL}] last-200 mean cell time: {mean_fwd:.2f}s  "
            f"mean seq_len: {float(np.mean(per_cell_seq_lens[-200:])):.0f}"
        )

print(f"\n  total forward time: {float(np.sum(per_cell_times)):.1f}s  "
      f"mean per cell: {float(np.mean(per_cell_times)):.2f}s")
print(f"  mean seq_len: {float(np.mean(per_cell_seq_lens)):.0f}  "
      f"max: {int(np.max(per_cell_seq_lens))}")

# ------------------------------------------------------------------------
# Step 7 — Finalize gene-space attention edges
# ------------------------------------------------------------------------
print("\n[7] Finalizing gene-space edges...")
denom = pair_counts.astype(np.float32)
denom[denom == 0] = np.nan
edges_per_head = attn_sum / denom[None, None, :, :]
edges_per_head = np.nan_to_num(edges_per_head, nan=0.0).astype(np.float32)
edges_layer_mean = edges_per_head.mean(axis=1)  # (L, HVG, HVG)

np.save(OUT_DIR / "attention_edges_per_head.npy", edges_per_head)
np.save(OUT_DIR / "attention_edges_layer_mean.npy", edges_layer_mean)
np.save(OUT_DIR / "attention_pair_counts.npy", pair_counts)
print(f"  attention_edges_per_head.npy:   {edges_per_head.shape}")
print(f"  attention_edges_layer_mean.npy: {edges_layer_mean.shape}")

# ------------------------------------------------------------------------
# Step 8 — Persist config + save control-cell metadata
# ------------------------------------------------------------------------
print("\n[8] Saving run config + cell metadata...")
# Cell metadata for the sampled controls
with h5py.File(REPLOGLE_H5, "r") as f:
    cell_barcodes = f["obs"]["cell_barcode"][ctrl_idx]
    cell_barcodes = [b.decode() if isinstance(b, bytes) else b for b in cell_barcodes]
    umi = f["obs"]["UMI_count"][ctrl_idx]
    mito = f["obs"]["mitopercent"][ctrl_idx]
pd.DataFrame({
    "ctrl_idx_local": np.arange(N_CTRL),
    "cell_idx_global": ctrl_idx,
    "cell_barcode": cell_barcodes,
    "UMI_count": umi,
    "mitopercent": mito,
}).to_csv(OUT_DIR / "control_cells.csv", index=False)

config = {
    "pipeline": "attention-grn-extraction-and-evaluation",
    "phase": "0a",
    "model": "MaxToki-217M-HF",
    "model_n_layers": int(xt.n_layers),
    "model_n_heads": int(xt.n_heads),
    "model_hidden_size": int(xt.hidden_size),
    "device": DEVICE,
    "n_ctrl": int(N_CTRL),
    "n_hvg": int(n_hvg),
    "max_len": int(MAX_LEN),
    "cell_line": CELL_LINE,
    "dataset": "replogle_concat.h5ad (K562 non-targeting)",
    "seed": SEED,
    "mean_seq_len": float(np.mean(per_cell_seq_lens)),
    "max_seq_len": int(np.max(per_cell_seq_lens)),
    "min_seq_len": int(np.min(per_cell_seq_lens)),
    "mean_fwd_seconds": float(np.mean(per_cell_times)),
    "total_fwd_seconds": float(np.sum(per_cell_times)),
    "total_phase_seconds": float(time.time() - t_phase),
    "pair_counts_median": int(np.median(pair_counts[pair_counts > 0])),
    "pair_counts_max": int(pair_counts.max()),
    "gene_features_file": "gene_features.csv",
    "hvg_gene_table_file": "hvg_gene_table.csv",
    "attention_file": "attention_edges_per_head.npy",
    "spearman_file": "spearman_edges.npy",
}
with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 70)
print(f"PHASE 0 COMPLETE in {time.time()-t_phase:.1f}s")
print("  outputs:", OUT_DIR)
print("=" * 70)
