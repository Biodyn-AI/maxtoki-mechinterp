"""Phase 0 extraction for any dataset (DATASET env var selects).

Handles both dense and CSR-sparse X backends. Uses anndata.read_h5ad to
uniformly obtain a dense numpy view of the control cells' expression
matrix regardless of the underlying storage format.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

DATASET = os.environ.get("DATASET", "k562").lower()
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "" if DATASET == "k562" else f"_{DATASET}")
MODEL_DIR_ENV = os.environ.get("MODEL_DIR", str(PROJ / "setup/MaxToki-217M-HF"))
OUT_DIR = PROJ / f"runs/attention-grn-217M/outputs/phase0{OUT_SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CTRL = int(os.environ.get("PHASE0_N_CTRL", "2000"))
N_HVG = int(os.environ.get("PHASE0_N_HVG", "1500"))
MAX_LEN = int(os.environ.get("PHASE0_MAX_LEN", "2048"))
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

print("=" * 70)
print(f"PHASE 0 (any) — MaxToki-217M  DATASET={DATASET}  N_ctrl={N_CTRL}  device={DEVICE}")
print("=" * 70)

rng = np.random.default_rng(SEED)
t_phase = time.time()

print("\n[1] Loading dataset...")
ds = load_ds(DATASET)
print(f"  n_cells: {ds.n_cells_total:,}  n_genes: {ds.n_genes_total:,}")
print(f"  control_codes: {sorted(ds.control_category_codes)}")
ctrl_all = np.where(
    ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes))
)[0]
print(f"  control cells: {len(ctrl_all):,}")
if len(ctrl_all) < 100:
    raise RuntimeError("insufficient controls")
n_ctrl_eff = min(N_CTRL, len(ctrl_all))
ctrl_idx = np.sort(rng.choice(ctrl_all, size=n_ctrl_eff, replace=False))
print(f"  sampled: {n_ctrl_eff}")

# --- Load expression (dense-via-anndata to handle CSR uniformly) ---
print("\n[2] Loading control expression matrix (anndata.read_h5ad backed)...")
t0 = time.time()
adata = ad.read_h5ad(str(ds.h5_path), backed="r")
# Slice controls via AnnData indexing which handles both dense and CSR
# backends (including anndata's SparseDataset wrapper).
X_sub = adata[ctrl_idx].X
if sp.issparse(X_sub) or hasattr(X_sub, "toarray"):
    X_ctrl = X_sub.toarray().astype(np.float32)
else:
    X_ctrl = np.asarray(X_sub).astype(np.float32)
adata.file.close()
print(f"  shape: {X_ctrl.shape}  mean nz/cell: {float((X_ctrl > 0).sum(axis=1).mean()):.0f}  ({time.time()-t0:.1f}s)")

rs = X_ctrl.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_ctrl / rs * 1e4)

print("\n[3] Selecting HVG ∩ MaxToki vocab...")
tokenizer = MaxTokiTokenizer()
mt_vocab = set(tokenizer.gene_token_dict.keys())
var_ens = ds.var_ensembl
eligible = np.array([e is not None and e in mt_vocab for e in var_ens], dtype=bool)
print(f"  var in MaxToki vocab: {int(eligible.sum())}/{ds.n_genes_total}")
n_hvg_eff = min(N_HVG, int(eligible.sum()))
gene_var_all = X_log.var(axis=0)
var_rank = np.argsort(-gene_var_all)
hvg_indices = []
for gi in var_rank:
    if eligible[gi]:
        hvg_indices.append(int(gi))
        if len(hvg_indices) >= n_hvg_eff:
            break
hvg_indices_np = np.array(sorted(hvg_indices), dtype=np.int64)
hvg_symbols = [ds.var_symbols[i] for i in hvg_indices_np]
hvg_ens = [ds.var_ensembl[i] for i in hvg_indices_np]
hvg_tokens = np.array([tokenizer.gene_token_dict[e] for e in hvg_ens], dtype=np.int64)

X_ctrl_hvg = X_ctrl[:, hvg_indices_np]
X_log_hvg = X_log[:, hvg_indices_np]
gene_mean = X_log_hvg.mean(axis=0)
gene_var = X_log_hvg.var(axis=0)
gene_dropout = (X_ctrl_hvg == 0).mean(axis=0)
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
tf_out = trrust.groupby("tf")["target"].nunique().to_dict()
tf_out_degree = np.array([float(tf_out.get(s.upper(), 0)) for s in hvg_symbols], dtype=np.float32)

gene_features = pd.DataFrame({
    "hvg_idx": np.arange(len(hvg_symbols)), "symbol": hvg_symbols,
    "ensembl_id": hvg_ens, "maxtoki_token_id": hvg_tokens,
    "var_idx": hvg_indices_np,
    "mean_expr": gene_mean, "variance": gene_var,
    "dropout_rate": gene_dropout, "tf_out_degree": tf_out_degree,
})
gene_features.to_csv(OUT_DIR / "gene_features.csv", index=False)
gene_features[["hvg_idx", "symbol", "ensembl_id", "maxtoki_token_id", "var_idx"]].to_csv(
    OUT_DIR / "hvg_gene_table.csv", index=False
)

var_to_hvg = -np.ones(ds.n_genes_total, dtype=np.int64)
for hi, vi in enumerate(hvg_indices_np):
    var_to_hvg[vi] = hi

# --- Spearman baseline ---
print("\n[4] Spearman baseline...")
t0 = time.time()
X_ranked = np.apply_along_axis(rankdata, 0, X_log_hvg).astype(np.float32)
X_ranked -= X_ranked.mean(axis=0, keepdims=True)
X_ranked /= (X_ranked.std(axis=0, keepdims=True) + 1e-12)
spearman_edges = (X_ranked.T @ X_ranked) / X_log_hvg.shape[0]
np.save(OUT_DIR / "spearman_edges.npy", spearman_edges.astype(np.float32))
print(f"  {spearman_edges.shape}  {time.time()-t0:.1f}s")

# --- MaxToki ---
print(f"\n[5] Loading MaxToki on {DEVICE} (model_dir={MODEL_DIR_ENV})...")
t0 = time.time()
xt = MaxTokiAttentionExtractor(model_dir=Path(MODEL_DIR_ENV), device=DEVICE, dtype=torch.float32)
print(f"  loaded in {time.time()-t0:.1f}s")
n_layers, n_heads = xt.n_layers, xt.n_heads
n_hvg = len(hvg_symbols)

var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)
mt_var_to_hvg = var_to_hvg[var_indices_full]

attn_sum = np.zeros((n_layers, n_heads, n_hvg, n_hvg), dtype=np.float32)
pair_counts = np.zeros((n_hvg, n_hvg), dtype=np.int32)
per_cell_seq_lens = []
per_cell_times = []

print("\n[6] Forward passes...")
for ci in tqdm(range(n_ctrl_eff), ncols=80):
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
    for li in range(n_layers):
        al = attn[li][0].numpy()
        sub = al[:, pos_idx[:, None], pos_idx[None, :]]
        hvg_ids_this_cell = hvg_pp[pos_idx]
        A, B = np.meshgrid(hvg_ids_this_cell, hvg_ids_this_cell, indexing="ij")
        attn_sum[li, :, A.ravel(), B.ravel()] += sub.reshape(n_heads, -1).T
    A, B = np.meshgrid(hvg_pp[pos_idx], hvg_pp[pos_idx], indexing="ij")
    pair_counts[A.ravel(), B.ravel()] += 1
    # Free MPS unified-memory cache to prevent progressive slowdown on large models.
    del attn
    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    per_cell_seq_lens.append(len(pos))
    per_cell_times.append(time.time() - t0)
    if (ci + 1) % 200 == 0 or (ci + 1) == n_ctrl_eff:
        tqdm.write(f"  [{ci+1}/{n_ctrl_eff}] mean {np.mean(per_cell_times[-200:]):.2f}s  "
                   f"seq_len {np.mean(per_cell_seq_lens[-200:]):.0f}")

denom = pair_counts.astype(np.float32); denom[denom == 0] = np.nan
edges_per_head = attn_sum / denom[None, None]
edges_per_head = np.nan_to_num(edges_per_head, nan=0.0).astype(np.float32)
edges_layer_mean = edges_per_head.mean(axis=1)
np.save(OUT_DIR / "attention_edges_per_head.npy", edges_per_head)
np.save(OUT_DIR / "attention_edges_layer_mean.npy", edges_layer_mean)
np.save(OUT_DIR / "attention_pair_counts.npy", pair_counts)

pd.DataFrame({
    "ctrl_idx_local": np.arange(n_ctrl_eff),
    "cell_idx_global": ctrl_idx,
}).to_csv(OUT_DIR / "control_cells.csv", index=False)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "attention-grn", "phase": "0a",
        "dataset": DATASET, "model": "MaxToki-217M-HF",
        "model_n_layers": n_layers, "model_n_heads": n_heads,
        "device": DEVICE, "n_ctrl": n_ctrl_eff, "n_hvg": n_hvg,
        "mean_seq_len": float(np.mean(per_cell_seq_lens)),
        "mean_fwd_seconds": float(np.mean(per_cell_times)),
        "total_fwd_seconds": float(sum(per_cell_times)),
        "total_phase_seconds": float(time.time() - t_phase),
        "seed": SEED,
    }, f, indent=2)

print(f"\nPHASE 0 ({DATASET}) COMPLETE — outputs: {OUT_DIR}")
