"""Phase 0 extraction on Replogle RPE1 (cross-context replication of K562).

Same methodology as phase0_extract.py but:
- uses ReplogleWeissman2022_rpe1.h5ad (no cell_line column; all cells are RPE1)
- uses var/ensembl_id directly (no sym2ens lookup)
- control cells are in the "control" perturbation category
- writes artefacts under outputs/phase0_rpe1/
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
from scipy.stats import rankdata
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

OUT_DIR = PROJ / "runs/attention-grn-217M/outputs/phase0_rpe1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CTRL = int(os.environ.get("PHASE0_N_CTRL", "2000"))
N_HVG = 1500
MAX_LEN = 2048
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

print("=" * 70)
print(f"PHASE 0 (RPE1) — MaxToki-217M attention extraction")
print(f"  N_ctrl={N_CTRL}  N_hvg={N_HVG}  max_len={MAX_LEN}  device={DEVICE}")
print("=" * 70)

rng = np.random.default_rng(SEED)
t_phase = time.time()

# --- load RPE1 ---
print("\n[1] Loading RPE1 dataset...")
ds = load_ds("rpe1")
print(f"  n cells: {ds.n_cells_total:,}  n genes: {ds.n_genes_total:,}")
print(f"  perturbation categories: {len(ds.perturbation_categories)}")
print(f"  control categories: {sorted(ds.control_category_codes)}")

ctrl_all_idx = np.where(
    ds.cell_of_interest_mask
    & np.isin(ds.perturbation_codes, list(ds.control_category_codes))
)[0]
print(f"  control cells: {len(ctrl_all_idx):,}")
if len(ctrl_all_idx) < N_CTRL:
    raise RuntimeError(f"not enough RPE1 controls: {len(ctrl_all_idx)} < {N_CTRL}")
ctrl_idx = np.sort(rng.choice(ctrl_all_idx, size=N_CTRL, replace=False))

# --- load expression matrix ---
print("\n[2] Loading control expression matrix...")
t0 = time.time()
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((N_CTRL, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CTRL, 100):
        end = min(i + 100, N_CTRL)
        X_ctrl[i:end] = f["X"][ctrl_idx[i:end], :]
print(f"  shape: {X_ctrl.shape}  mean nz/cell: {float((X_ctrl > 0).sum(axis=1).mean()):.0f}  ({time.time()-t0:.1f}s)")

# --- log-normalize for HVG + features ---
rs = X_ctrl.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_ctrl / rs * 1e4)

# --- HVG selection ∩ MaxToki vocab ---
print("\n[3] Selecting HVG ∩ MaxToki vocab...")
tokenizer = MaxTokiTokenizer()
mt_vocab = set(tokenizer.gene_token_dict.keys())
var_ens = ds.var_ensembl
eligible = np.array(
    [e is not None and e in mt_vocab for e in var_ens], dtype=bool
)
print(f"  var in MaxToki vocab: {int(eligible.sum())}/{ds.n_genes_total}")

gene_var_all = X_log.var(axis=0)
var_rank = np.argsort(-gene_var_all)
hvg_indices = []
for gi in var_rank:
    if eligible[gi]:
        hvg_indices.append(int(gi))
        if len(hvg_indices) >= N_HVG:
            break
if len(hvg_indices) < N_HVG:
    raise RuntimeError(f"Only {len(hvg_indices)} eligible HVG; need {N_HVG}")
hvg_indices_np = np.array(sorted(hvg_indices), dtype=np.int64)
hvg_symbols = [ds.var_symbols[i] for i in hvg_indices_np]
hvg_ens = [ds.var_ensembl[i] for i in hvg_indices_np]
hvg_tokens = np.array([tokenizer.gene_token_dict[e] for e in hvg_ens], dtype=np.int64)
assert len(set(hvg_tokens.tolist())) == len(hvg_tokens)
print(f"  HVG list built: {len(hvg_symbols)} genes")

X_ctrl_hvg = X_ctrl[:, hvg_indices_np]
X_log_hvg = X_log[:, hvg_indices_np]

gene_mean = X_log_hvg.mean(axis=0)
gene_var = X_log_hvg.var(axis=0)
gene_dropout = (X_ctrl_hvg == 0).mean(axis=0)

trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
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

# --- Spearman ---
print("\n[4] Spearman baseline...")
t0 = time.time()
X_ranked = np.apply_along_axis(rankdata, 0, X_log_hvg).astype(np.float32)
X_ranked -= X_ranked.mean(axis=0, keepdims=True)
X_ranked /= (X_ranked.std(axis=0, keepdims=True) + 1e-12)
spearman_edges = (X_ranked.T @ X_ranked) / X_log_hvg.shape[0]
np.save(OUT_DIR / "spearman_edges.npy", spearman_edges.astype(np.float32))
print(f"  {spearman_edges.shape}  {time.time()-t0:.1f}s")

# --- MaxToki + tokenizer mapping ---
print(f"\n[5] Loading MaxToki on {DEVICE}...")
t0 = time.time()
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
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
    for li in range(n_layers):
        al = attn[li][0].numpy()
        sub = al[:, pos_idx[:, None], pos_idx[None, :]]
        hvg_ids_this_cell = hvg_pp[pos_idx]
        A, B = np.meshgrid(hvg_ids_this_cell, hvg_ids_this_cell, indexing="ij")
        attn_sum[li, :, A.ravel(), B.ravel()] += sub.reshape(n_heads, -1).T
    A, B = np.meshgrid(hvg_pp[pos_idx], hvg_pp[pos_idx], indexing="ij")
    pair_counts[A.ravel(), B.ravel()] += 1
    per_cell_seq_lens.append(len(pos))
    per_cell_times.append(time.time() - t0)
    if (ci + 1) % 200 == 0 or (ci + 1) == N_CTRL:
        tqdm.write(
            f"  [{ci+1}/{N_CTRL}] last-200 mean {np.mean(per_cell_times[-200:]):.2f}s  "
            f"seq_len {np.mean(per_cell_seq_lens[-200:]):.0f}"
        )

# --- finalize ---
denom = pair_counts.astype(np.float32); denom[denom == 0] = np.nan
edges_per_head = attn_sum / denom[None, None]
edges_per_head = np.nan_to_num(edges_per_head, nan=0.0).astype(np.float32)
edges_layer_mean = edges_per_head.mean(axis=1)

np.save(OUT_DIR / "attention_edges_per_head.npy", edges_per_head)
np.save(OUT_DIR / "attention_edges_layer_mean.npy", edges_layer_mean)
np.save(OUT_DIR / "attention_pair_counts.npy", pair_counts)

pd.DataFrame({
    "ctrl_idx_local": np.arange(N_CTRL),
    "cell_idx_global": ctrl_idx,
}).to_csv(OUT_DIR / "control_cells.csv", index=False)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "attention-grn", "phase": "0a",
        "dataset": "rpe1",
        "model": "MaxToki-217M-HF",
        "model_n_layers": n_layers, "model_n_heads": n_heads,
        "device": DEVICE, "n_ctrl": N_CTRL, "n_hvg": n_hvg,
        "mean_seq_len": float(np.mean(per_cell_seq_lens)),
        "mean_fwd_seconds": float(np.mean(per_cell_times)),
        "total_fwd_seconds": float(sum(per_cell_times)),
        "total_phase_seconds": float(time.time() - t_phase),
        "seed": SEED,
    }, f, indent=2)

print(f"\nPHASE 0 RPE1 COMPLETE — outputs: {OUT_DIR}")
