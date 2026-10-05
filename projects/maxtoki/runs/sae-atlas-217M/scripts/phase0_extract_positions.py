"""Phase 0 — per-position activation extraction for SAE training.

Extracts residual-stream hidden states at EVERY (cell, token-position) for
selected layers. Unlike the spectral-geometry Phase 0 which averages per
gene, this emits the raw per-position activations for SAE training.

Output per layer: memory-mapped float32 npy of shape [n_positions, d_model],
plus parallel int arrays for gene_ids and cell_ids.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

REPLOGLE_H5 = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"

OUT_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CTRL = int(os.environ.get("PHASE0_N_CTRL", "500"))
MAX_LEN = int(os.environ.get("PHASE0_MAX_LEN", "2048"))
DEVICE = os.environ.get("PHASE0_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
# Layers to extract (0-indexed into the n_layers+1 hidden_states tuple)
# L0 = post-embedding, L5 = mid transformer, L11 = final transformer block
TARGET_LAYERS = [int(x) for x in os.environ.get("PHASE0_LAYERS", "0,5,11").split(",")]

print("=" * 70)
print(f"SAE Phase 0 — per-position extraction  N_ctrl={N_CTRL}  layers={TARGET_LAYERS}")
print("=" * 70)
t_phase = time.time()
rng = np.random.default_rng(SEED)

# Load K562 control cells
print("\n[1] Loading K562 controls...")
sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds, SYM2ENS_PKL
ds = load_ds("k562")
ctrl_all = np.where(
    ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes))
)[0]
ctrl_idx = np.sort(rng.choice(ctrl_all, size=N_CTRL, replace=False))
print(f"  sampled {N_CTRL} controls")

with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((N_CTRL, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CTRL, 100):
        X_ctrl[i:i+100] = f["X"][ctrl_idx[i:i+100], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]

with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
var_ens = [sym2ens.get(s) for s in var_symbols]

# Tokenizer setup
tokenizer = MaxTokiTokenizer()
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)

# Build reverse mapping: token_id → gene symbol
token_to_gene = {}
for vi, ti in zip(var_indices_full, var_tokens_full):
    token_to_gene[int(ti)] = var_symbols[vi]

print(f"\n[2] Loading MaxToki-217M on {DEVICE}...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layer_states = xt.n_layers + 1  # 12 (post-embed + 11 transformer blocks)
hidden_size = xt.hidden_size       # 1232
print(f"  n_layer_states={n_layer_states}  hidden_size={hidden_size}")
print(f"  target layers: {TARGET_LAYERS}")

# Pre-allocate accumulation lists (will convert to arrays at end)
layer_positions = {l: [] for l in TARGET_LAYERS}
layer_gene_ids = {l: [] for l in TARGET_LAYERS}
layer_cell_ids = {l: [] for l in TARGET_LAYERS}

print(f"\n[3] Forward passes ({N_CTRL} cells)...")
per_cell_times = []
total_positions = 0
for ci in tqdm(range(N_CTRL), ncols=80):
    t0 = time.time()
    cell = tokenizer.tokenize_cell(
        X_ctrl[ci], var_indices_full, var_tokens_full, var_medians_full, max_len=MAX_LEN
    )
    if cell is None:
        continue
    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, hidden = xt.forward_with_hidden_states(input_ids)
    # hidden[l]: (1, seq_len, hidden_size)
    seq_len = cell.token_ids.shape[0]
    for li in TARGET_LAYERS:
        hl = hidden[li][0].numpy()  # (seq_len, H)
        # Keep all positions (including <bos>/<eos> — SAE should see the full distribution)
        layer_positions[li].append(hl)
        # Gene IDs per position: map token_id → gene symbol
        gids = [token_to_gene.get(int(cell.token_ids[pos]), "<special>") for pos in range(seq_len)]
        layer_gene_ids[li].append(gids)
        layer_cell_ids[li].extend([ci] * seq_len)
    total_positions += seq_len
    per_cell_times.append(time.time() - t0)
    # Free MPS cache
    del hidden
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except: pass
    if (ci + 1) % 100 == 0 or (ci + 1) == N_CTRL:
        tqdm.write(f"  [{ci+1}/{N_CTRL}] mean {np.mean(per_cell_times[-100:]):.2f}s  "
                   f"total_pos={total_positions:,}")

print(f"\n  total positions: {total_positions:,}  mean/cell: {total_positions/N_CTRL:.0f}")

# Concatenate and save as memory-mapped npy
print(f"\n[4] Saving per-layer activation files...")
for li in TARGET_LAYERS:
    acts = np.concatenate(layer_positions[li], axis=0).astype(np.float32)
    n_pos = acts.shape[0]
    print(f"  L{li}: {n_pos:,} positions × {hidden_size} = {n_pos * hidden_size * 4 / 1e9:.2f} GB")
    np.save(OUT_DIR / f"layer_{li:02d}_activations.npy", acts)
    # Gene names as JSON list
    gnames = []
    for gl in layer_gene_ids[li]:
        gnames.extend(gl)
    with open(OUT_DIR / f"layer_{li:02d}_gene_names.json", "w") as f:
        json.dump(gnames, f)
    # Cell IDs
    np.save(OUT_DIR / f"layer_{li:02d}_cell_ids.npy",
            np.array(layer_cell_ids[li], dtype=np.int32))
    del acts  # free RAM

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "sae-atlas",
        "phase": "0",
        "model": "MaxToki-217M-HF",
        "dataset": "k562",
        "n_ctrl": N_CTRL,
        "max_len": MAX_LEN,
        "target_layers": TARGET_LAYERS,
        "n_layer_states": n_layer_states,
        "hidden_size": hidden_size,
        "total_positions": total_positions,
        "mean_positions_per_cell": total_positions / N_CTRL,
        "mean_fwd_seconds": float(np.mean(per_cell_times)),
        "total_phase_seconds": float(time.time() - t_phase),
    }, f, indent=2)
print(f"\nSAE PHASE 0 COMPLETE — outputs: {OUT_DIR}")
