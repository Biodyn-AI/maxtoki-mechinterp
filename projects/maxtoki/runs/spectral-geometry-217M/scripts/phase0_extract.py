"""Phase 0 — residual-stream per-gene embedding extraction on Tabula Sapiens immune.

Runs MaxToki-217M over 2000 immune cells, extracts per-layer hidden states,
and averages them per gene (at the gene's token position) across cells.

Output: [n_layers + 1, n_hvg, d_model] embedding tensor (layer 0 = post-embed,
layers 1..11 = post-transformer-block-i).
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
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

TABULA_SAPIENS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"

OUT_DIR = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CELLS = int(os.environ.get("PHASE0_N_CELLS", "2000"))
N_HVG = int(os.environ.get("PHASE0_N_HVG", "1500"))
MAX_LEN = int(os.environ.get("PHASE0_MAX_LEN", "2048"))
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 70)
print(f"PHASE 0 — spectral geometry  N_cells={N_CELLS}  N_hvg={N_HVG}  device={DEVICE}")
print("=" * 70)
t_phase = time.time()

# ------------------------------------------------------------------------
# Load Tabula Sapiens immune
# ------------------------------------------------------------------------
print("\n[1] Loading Tabula Sapiens immune...")
t0 = time.time()
adata = ad.read_h5ad(str(TABULA_SAPIENS_H5), backed="r")
print(f"  adata: {adata.shape}")
print(f"  var columns: {list(adata.var.columns)[:10]}")

# Clean Ensembl IDs: the 'ensembl_id' column has version suffixes like ENSG.15;
# strip them. The '_index' column is already clean.
var_ens = adata.var_names.to_numpy().astype(str)
var_ens_clean = np.array([e.split(".")[0] for e in var_ens])
var_feature_name = adata.var["feature_name"].to_numpy().astype(str)

# Sample N_CELLS from the full 592k Tabula Sapiens immune pool
n_total = adata.n_obs
rng_sample = np.random.default_rng(SEED)
sample_idx = np.sort(rng_sample.choice(n_total, size=N_CELLS, replace=False))
print(f"  sampled {N_CELLS}/{n_total} cells")

# Pull expression — Tabula Sapiens is CSR
X_ctrl = adata[sample_idx].X
if sp.issparse(X_ctrl) or hasattr(X_ctrl, "toarray"):
    X_ctrl = X_ctrl.toarray()
X_ctrl = np.asarray(X_ctrl, dtype=np.float32)
print(f"  X shape: {X_ctrl.shape}  mean nz/cell: {float((X_ctrl > 0).sum(axis=1).mean()):.0f}  ({time.time()-t0:.1f}s)")

# Pull cell-type labels for Phase 6/7
cell_types = adata.obs["cell_type"].iloc[sample_idx].astype(str).to_numpy()
adata.file.close()

# Normalize
rs = X_ctrl.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_ctrl / rs * 1e4)

# ------------------------------------------------------------------------
# HVG selection ∩ MaxToki vocab, biased toward annotated-reference genes
# ------------------------------------------------------------------------
# To make the downstream biology tests (PPI, regulatory, cell-type markers)
# well-powered, HVG selection prefers genes that are (a) in MaxToki vocab AND
# (b) have at least one of: TRRUST annotation, STRING PPI membership, or
# canonical-marker role. Genes meeting (b) are prioritised; the remaining
# budget is filled with top-variance genes among the MaxToki-vocab set.
print("\n[2] Selecting HVG ∩ MaxToki vocab (annotation-biased)...")
tokenizer = MaxTokiTokenizer()
mt_vocab = set(tokenizer.gene_token_dict.keys())
eligible = np.array([e in mt_vocab for e in var_ens_clean], dtype=bool)
print(f"  var in MaxToki vocab: {int(eligible.sum())}/{len(var_ens_clean)}")

# Build the "priority" set of reference-annotated genes.
import pickle as _pkl
SYM2ENS = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"
TRRUST_TSV_ = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON_ = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"
import json as _json
sym2ens_inv = {v: k for k, v in _pkl.load(open(SYM2ENS, "rb")).items()} if SYM2ENS.exists() else {}
# Priority symbols (uppercase)
priority_symbols = set()
trrust = pd.read_csv(TRRUST_TSV_, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
priority_symbols |= set(trrust["tf"].str.upper()) | set(trrust["target"].str.upper())
with open(STRING_JSON_) as f:
    string_data = _json.load(f)
for pair_list in string_data.values():
    for a, b in pair_list:
        priority_symbols.add(a.upper()); priority_symbols.add(b.upper())
# Canonical lineage markers (for Phase 6+7)
_markers = [
    "CD19","CD79A","MS4A1","BLK","VPREB3","FCRL1","PAX5","BATF","BACH2","BCL6",
    "PRDM1","IRF4","IRF8","CD3D","CD3E","CD3G","CD4","CD8A","CD8B","LCK","ZAP70",
    "TCF7","LEF1","TBX21","GATA3","NCAM1","KLRD1","KLRF1","NKG7","GNLY","CD68",
    "CD163","MRC1","MSR1","MERTK","CD14","LYZ","VCAN","S100A8","S100A9",
    "CLEC9A","CLEC10A","CD1C","FCER1A","ELANE","MPO","AZU1","DEFA3","CSF3R",
    "NAMPT","GLUL","PFKFB3","HK1","HK2","PKM","LDHA","MTOR","STAT3",
]
priority_symbols |= set(s.upper() for s in _markers)

priority_mask = np.array(
    [(var_feature_name[i].upper() in priority_symbols) for i in range(len(var_ens_clean))],
    dtype=bool,
)
print(f"  reference-annotated pool: {int(priority_mask.sum())} genes")

# Selection:
#  - first, all priority genes that are (a) in MaxToki vocab (b) non-zero variance
n_hvg_eff = int(N_HVG)
gene_var_all = X_log.var(axis=0)
gene_nz_pct = (X_log > 0).mean(axis=0)

# Priority ∩ vocab ∩ has some expression
pool_mask = priority_mask & eligible & (gene_var_all > 0) & (gene_nz_pct > 0.005)
pool_idx = np.where(pool_mask)[0]
# Sort by variance descending within the pool
pool_sorted = pool_idx[np.argsort(-gene_var_all[pool_idx])]
priority_taken = pool_sorted[: min(n_hvg_eff, len(pool_sorted))]
print(f"  priority HVG selected: {len(priority_taken)}")

# Fill remaining budget with top-variance non-priority vocab genes
remaining_need = n_hvg_eff - len(priority_taken)
if remaining_need > 0:
    rest_mask = ~pool_mask & eligible & (gene_var_all > 0)
    rest_idx = np.where(rest_mask)[0]
    rest_sorted = rest_idx[np.argsort(-gene_var_all[rest_idx])][:remaining_need]
    hvg_indices_np = np.sort(np.concatenate([priority_taken, rest_sorted])).astype(np.int64)
else:
    hvg_indices_np = np.sort(priority_taken).astype(np.int64)
n_hvg_eff = len(hvg_indices_np)

hvg_ens = [var_ens_clean[i] for i in hvg_indices_np]
hvg_symbols = [var_feature_name[i] for i in hvg_indices_np]
hvg_tokens = np.array([tokenizer.gene_token_dict[e] for e in hvg_ens], dtype=np.int64)

# Coverage diagnostics
hvg_priority = sum(1 for s in hvg_symbols if s.upper() in priority_symbols)
hvg_trrust_tfs = sum(1 for s in hvg_symbols if s.upper() in set(trrust["tf"].str.upper()))
hvg_trrust_targets = sum(1 for s in hvg_symbols if s.upper() in set(trrust["target"].str.upper()))
hvg_markers = sum(1 for s in hvg_symbols if s.upper() in set(x.upper() for x in _markers))
print(f"  HVG list built: {n_hvg_eff}  priority_in_HVG={hvg_priority}  "
      f"trrust_tfs={hvg_trrust_tfs}  trrust_targets={hvg_trrust_targets}  "
      f"canonical_markers={hvg_markers}")

X_log_hvg = X_log[:, hvg_indices_np]

gene_features = pd.DataFrame({
    "hvg_idx": np.arange(len(hvg_symbols)),
    "symbol": hvg_symbols,
    "ensembl_id": hvg_ens,
    "maxtoki_token_id": hvg_tokens,
    "var_idx": hvg_indices_np,
    "mean_expr": X_log_hvg.mean(axis=0),
    "variance": X_log_hvg.var(axis=0),
    "dropout_rate": (X_ctrl[:, hvg_indices_np] == 0).mean(axis=0),
})
gene_features.to_csv(OUT_DIR / "gene_features.csv", index=False)
pd.DataFrame({
    "sample_idx_local": np.arange(N_CELLS),
    "cell_type": cell_types,
}).to_csv(OUT_DIR / "cell_metadata.csv", index=False)

# Map to MaxToki var-indexed token id table
var_to_hvg = -np.ones(len(var_ens_clean), dtype=np.int64)
for hi, vi in enumerate(hvg_indices_np):
    var_to_hvg[vi] = hi

# Build var mapping for tokenizer (use the full Tabula Sapiens var list)
# tokenizer.make_var_mapping expects an iterable of ensembl ids; provide var_ens_clean
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens_clean.tolist())
mt_var_to_hvg = var_to_hvg[var_indices_full]

# ------------------------------------------------------------------------
# Load MaxToki + extract hidden states
# ------------------------------------------------------------------------
print(f"\n[3] Loading MaxToki-217M on {DEVICE}...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layers = xt.n_layers           # 11 transformer layers
hidden_size = xt.hidden_size     # 1232
# output_hidden_states returns n_layers + 1 tensors (embed + per-layer)
n_layer_states = n_layers + 1

# Accumulators: per-gene sum of hidden-state vector plus count
#  shape: (n_layer_states, n_hvg, hidden_size) float32 = 12 * 1500 * 1232 * 4 = 88 MB
hidden_sum = np.zeros((n_layer_states, n_hvg_eff, hidden_size), dtype=np.float32)
gene_counts = np.zeros(n_hvg_eff, dtype=np.int32)

per_cell_times = []
per_cell_seq_lens = []

print(f"\n[4] Forward passes (extracting per-layer hidden states)...")
for ci in tqdm(range(N_CELLS), ncols=80):
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
    if len(pos_idx) == 0:
        continue
    hvg_ids_this_cell = hvg_pp[pos_idx]   # shape (k,)

    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, hidden = xt.forward_with_hidden_states(input_ids)
    # hidden[l]: (1, seq, hidden_size) float32
    for li in range(n_layer_states):
        hl = hidden[li][0].numpy()  # (seq, H)
        sub = hl[pos_idx]            # (k, H)
        # Scatter-add sub[i] to hidden_sum[l, hvg_ids_this_cell[i]]
        np.add.at(hidden_sum[li], hvg_ids_this_cell, sub)

    gene_counts[hvg_ids_this_cell] += 1
    per_cell_times.append(time.time() - t0)
    per_cell_seq_lens.append(len(pos))
    if (ci + 1) % 200 == 0 or (ci + 1) == N_CELLS:
        tqdm.write(
            f"  [{ci+1}/{N_CELLS}] mean {np.mean(per_cell_times[-200:]):.2f}s  "
            f"seq_len {np.mean(per_cell_seq_lens[-200:]):.0f}  "
            f"genes with ≥1 obs: {(gene_counts > 0).sum()}/{n_hvg_eff}"
        )

print(f"\n  total forward: {sum(per_cell_times):.1f}s")

# Finalise per-gene means
denom = gene_counts.astype(np.float32)
nonzero = denom > 0
print(f"  non-zero HVG: {int(nonzero.sum())}/{n_hvg_eff}")
denom[denom == 0] = np.nan
layer_gene_embeddings = hidden_sum / denom[None, :, None]
layer_gene_embeddings = np.nan_to_num(layer_gene_embeddings, nan=0.0).astype(np.float32)
print(f"  layer_gene_embeddings shape: {layer_gene_embeddings.shape}")

np.save(OUT_DIR / "layer_gene_embeddings.npy", layer_gene_embeddings)
np.save(OUT_DIR / "gene_counts.npy", gene_counts)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "residual-stream-spectral-geometry",
        "phase": "0",
        "model": "MaxToki-217M-HF",
        "dataset": "tabula_sapiens_immune",
        "n_cells": N_CELLS,
        "n_hvg": n_hvg_eff,
        "max_len": MAX_LEN,
        "device": DEVICE,
        "n_layer_states": int(n_layer_states),
        "hidden_size": int(hidden_size),
        "mean_seq_len": float(np.mean(per_cell_seq_lens)),
        "mean_fwd_seconds": float(np.mean(per_cell_times)),
        "total_fwd_seconds": float(sum(per_cell_times)),
        "total_phase_seconds": float(time.time() - t_phase),
        "n_genes_with_observations": int(nonzero.sum()),
        "seed": SEED,
    }, f, indent=2)
print(f"\nPHASE 0 COMPLETE — outputs: {OUT_DIR}")
