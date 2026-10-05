"""H3-G: Cell-type centroid trajectory across MaxToki-217M layers.

Hypothesis: depth encodes a pseudotime-like coordinate — related cell-type
centroids converge (distance decreases) while unrelated types remain apart.

Samples 200 cells per type (or all available) from Tabula Sapiens immune.
Runs MaxToki inference with output_hidden_states=True.
Computes centroid distances, Spearman(distance, depth), and bootstrap variance.
"""
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats
from tqdm import tqdm

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
BIOM = '<DATA_ROOT>'
OUT  = Path(f'{BASE}/autoloop/iterations/iter_0003')
PROJ = Path(f'{BASE}').parents[1]

sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402

TABULA_SAPIENS_H5 = Path(f'{BIOM}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad')
PHASE0_DIR = Path(f'{BASE}/outputs/phase0')

DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LEN = 2048
N_PER_TYPE = 200
SEED    = 42

TARGET_TYPES = {
    "B cell":                               "B_cell",
    "CD4-positive, alpha-beta T cell":      "CD4_T",
    "CD8-positive, alpha-beta T cell":      "CD8_T",
    "natural killer cell":                  "NK",
    "monocyte":                             "monocyte",
    "macrophage":                           "macrophage",
}
LINEAGE_PAIRS = {
    # (type_a, type_b): related=True means same-lineage (expect convergence)
    ("B_cell",   "CD4_T"):     False,
    ("B_cell",   "CD8_T"):     False,
    ("B_cell",   "NK"):        False,
    ("B_cell",   "monocyte"):  False,
    ("B_cell",   "macrophage"):False,
    ("CD4_T",    "CD8_T"):     True,   # T cell lineage
    ("CD4_T",    "NK"):        False,
    ("CD4_T",    "monocyte"):  False,
    ("CD4_T",    "macrophage"):False,
    ("CD8_T",    "NK"):        False,
    ("CD8_T",    "monocyte"):  False,
    ("CD8_T",    "macrophage"):False,
    ("NK",       "monocyte"):  False,
    ("NK",       "macrophage"):False,
    ("monocyte", "macrophage"):True,   # myeloid lineage
}
type_names = list(TARGET_TYPES.values())  # canonical short names

print("=" * 70)
print("H3-G: Cell-type centroid trajectory across MaxToki-217M layers")
print(f"  Device: {DEVICE}  N_per_type: {N_PER_TYPE}")
print("=" * 70)
t_total = time.time()

# ── Load HVG list ──────────────────────────────────────────────────────────────
gene_features  = pd.read_csv(PHASE0_DIR / "gene_features.csv")
hvg_ens        = gene_features["ensembl_id"].astype(str).tolist()
hvg_var_idx    = gene_features["var_idx"].astype(int).to_numpy()
n_hvg          = len(hvg_ens)
print(f"HVG list: {n_hvg} genes")

# ── Load Tabula Sapiens & sample cells ────────────────────────────────────────
print("\n[1] Loading Tabula Sapiens immune...")
adata     = ad.read_h5ad(str(TABULA_SAPIENS_H5), backed="r")
n_total   = adata.n_obs
cell_types_all = adata.obs["cell_type"].astype(str).to_numpy()
print(f"  adata: {adata.shape}")

rng = np.random.default_rng(SEED)
selected_idx  = []   # global indices into adata
selected_labels = [] # short type name

for long_name, short_name in TARGET_TYPES.items():
    pool = np.where(cell_types_all == long_name)[0]
    n_avail = len(pool)
    n_take  = min(n_avail, N_PER_TYPE)
    if n_take == 0:
        print(f"  WARNING: no cells for '{long_name}' — skipping")
        continue
    chosen = rng.choice(pool, size=n_take, replace=False)
    selected_idx.extend(chosen.tolist())
    selected_labels.extend([short_name] * n_take)
    print(f"  {long_name}: {n_avail} avail → using {n_take}")

selected_idx    = np.array(selected_idx, dtype=np.int64)
selected_labels = np.array(selected_labels)
n_cells_total   = len(selected_idx)
print(f"\nTotal selected cells: {n_cells_total}")

# Load expression for selected cells
var_ens_clean = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
X_cells = adata[selected_idx].X
if sp.issparse(X_cells) or hasattr(X_cells, "toarray"):
    X_cells = X_cells.toarray()
X_cells = np.asarray(X_cells, dtype=np.float32)
adata.file.close()
print(f"Expression loaded: {X_cells.shape}")

# ── Tokenizer + extractor ─────────────────────────────────────────────────────
print("\n[2] Loading MaxToki-217M...")
tokenizer = MaxTokiTokenizer()
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens_clean.tolist())
var_to_hvg_map = -np.ones(len(var_ens_clean), dtype=np.int64)
for hi, vi in enumerate(hvg_var_idx):
    var_to_hvg_map[vi] = hi
mt_var_to_hvg = var_to_hvg_map[var_indices_full]

xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layer_states = xt.n_layers + 1  # 12
hidden_size    = xt.hidden_size   # 1232
print(f"  n_layer_states={n_layer_states}, hidden_size={hidden_size}")

# ── Run inference: accumulate per-cell hidden states ─────────────────────────
print("\n[3] Running inference (accumulate per-cell centroid sums)...")

# Accumulate: per cell type, per layer → sum of hidden states at HVG positions
# We also store per-cell hidden states for bootstrap: (n_cells, n_layers, hidden_size)
# at 71MB float32 — manageable.
unique_types = list(dict.fromkeys(selected_labels))  # ordered unique
n_types = len(unique_types)
type_to_idx = {t: i for i, t in enumerate(unique_types)}

cell_hidden = np.full((n_cells_total, n_layer_states, hidden_size), np.nan, dtype=np.float32)
cell_valid  = np.zeros(n_cells_total, dtype=bool)

t_inf = time.time()
for ci in tqdm(range(n_cells_total), ncols=80, desc="inference"):
    cell = tokenizer.tokenize_cell(
        X_cells[ci], var_indices_full, var_tokens_full, var_medians_full, max_len=MAX_LEN,
    )
    if cell is None:
        continue
    pos       = cell.gene_positions
    hvg_pp    = np.full(len(pos), -1, dtype=np.int64)
    in_vocab  = pos >= 0
    hvg_pp[in_vocab] = mt_var_to_hvg[pos[in_vocab]]
    is_hvg    = hvg_pp >= 0
    pos_idx   = np.where(is_hvg)[0]
    if len(pos_idx) == 0:
        continue
    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, hidden = xt.forward_with_hidden_states(input_ids)
    # Average HVG-position hidden states across tokens for this cell → (n_layers, hidden_size)
    for li in range(n_layer_states):
        hl = hidden[li][0].numpy()   # (seq_len, hidden_size)
        cell_hidden[ci, li] = hl[pos_idx].mean(axis=0)
    cell_valid[ci] = True
    if DEVICE == "mps" and ci % 100 == 0:
        try: torch.mps.empty_cache()
        except Exception: pass

print(f"  Inference done: {cell_valid.sum()}/{n_cells_total} valid cells  ({time.time()-t_inf:.1f}s)")

# ── Compute centroids ─────────────────────────────────────────────────────────
print("\n[4] Computing centroids...")
centroids = np.full((n_types, n_layer_states, hidden_size), np.nan, dtype=np.float32)
counts_per_type = {}
for ti, tname in enumerate(unique_types):
    mask = (selected_labels == tname) & cell_valid
    counts_per_type[tname] = int(mask.sum())
    if mask.sum() == 0:
        print(f"  WARNING: {tname} — 0 valid cells")
        continue
    centroids[ti] = cell_hidden[mask].mean(axis=0)
    print(f"  {tname}: {mask.sum()} cells, centroid computed")

np.save(OUT / "h3g_centroids.npy", centroids)
np.save(OUT / "h3g_cell_hidden.npy", cell_hidden)

# ── Pairwise centroid distances ───────────────────────────────────────────────
print("\n[5] Centroid pairwise distances...")

def cosine_dist(a, b):
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(1.0 - np.dot(a, b) / (na * nb))

dist_rows = []
for ti in range(n_types):
    for tj in range(ti+1, n_types):
        tname_a = unique_types[ti]
        tname_b = unique_types[tj]
        is_related = LINEAGE_PAIRS.get((tname_a, tname_b),
                     LINEAGE_PAIRS.get((tname_b, tname_a), False))
        for li in range(n_layer_states):
            ca = centroids[ti, li]
            cb = centroids[tj, li]
            if np.any(np.isnan(ca)) or np.any(np.isnan(cb)):
                d = np.nan
            else:
                d = cosine_dist(ca, cb)
            dist_rows.append({
                'type_a': tname_a, 'type_b': tname_b,
                'is_related': is_related, 'layer': li, 'cosine_dist': d,
            })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT / "h3g_centroid_distances.csv", index=False)

# ── Spearman(distance, layer depth) per pair ──────────────────────────────────
print("\n[6] Spearman(distance, layer depth) per pair...")
pair_labels = []
pairs_done = set()
spear_rows = []
for ti in range(n_types):
    for tj in range(ti+1, n_types):
        tname_a = unique_types[ti]
        tname_b = unique_types[tj]
        key = (tname_a, tname_b)
        if key in pairs_done:
            continue
        pairs_done.add(key)
        is_related = LINEAGE_PAIRS.get(key, LINEAGE_PAIRS.get((tname_b, tname_a), False))
        sub = dist_df[(dist_df['type_a']==tname_a) & (dist_df['type_b']==tname_b)]
        dists = sub['cosine_dist'].values.astype(float)
        layers = sub['layer'].values.astype(float)
        valid = ~np.isnan(dists)
        if valid.sum() < 4:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = stats.spearmanr(layers[valid], dists[valid])
        spear_rows.append({
            'type_a': tname_a, 'type_b': tname_b,
            'is_related': is_related,
            'spearman_rho': float(rho) if not np.isnan(rho) else np.nan,
            'spearman_p': float(pval) if not np.isnan(pval) else np.nan,
            'dist_l0': float(dists[0]) if not np.isnan(dists[0]) else np.nan,
            'dist_l11': float(dists[11]) if not np.isnan(dists[11]) else np.nan,
            'n_valid_layers': int(valid.sum()),
        })
        direction = "converge" if rho < 0 else "diverge" if rho > 0 else "flat"
        print(f"  {tname_a} vs {tname_b}: rho={rho:.3f}, p={pval:.4f}, {direction}"
              f" ({'related' if is_related else 'unrelated'})")

spear_df = pd.DataFrame(spear_rows)
spear_df.to_csv(OUT / "h3g_distance_depth_spearman.csv", index=False)

# ── Bootstrap centroid stability ───────────────────────────────────────────────
print("\n[7] Bootstrap centroid stability...")
N_BOOT = 3
N_SUBSAMP = 100
boot_rows = []
rng2 = np.random.default_rng(SEED + 1)

for ti, tname in enumerate(unique_types):
    mask = (selected_labels == tname) & cell_valid
    idx_pool = np.where(mask)[0]
    n_avail  = len(idx_pool)
    if n_avail < N_SUBSAMP:
        sub_size = max(n_avail // 2, 1)
    else:
        sub_size = N_SUBSAMP

    boot_centroids = []  # (N_BOOT, n_layer_states, hidden_size)
    for b in range(N_BOOT):
        chosen = rng2.choice(idx_pool, size=sub_size, replace=False)
        boot_centroids.append(cell_hidden[chosen].mean(axis=0))

    boot_arr = np.array(boot_centroids)  # (N_BOOT, n_layers, hidden_size)
    # Variance across bootstrap samples per layer (mean over hidden dims)
    boot_var = boot_arr.var(axis=0).mean(axis=1)  # (n_layers,)

    # Mean pairwise cosine distance between bootstrap centroids
    boot_dist_l11 = []
    for ba in range(N_BOOT):
        for bb in range(ba+1, N_BOOT):
            boot_dist_l11.append(cosine_dist(boot_arr[ba, -1], boot_arr[bb, -1]))

    boot_rows.append({
        'cell_type': tname,
        'n_cells': n_avail,
        'subsample_size': sub_size,
        'boot_var_l0': float(boot_var[0]),
        'boot_var_l11': float(boot_var[-1]),
        'boot_mean_pairwise_dist_l11': float(np.mean(boot_dist_l11)) if boot_dist_l11 else np.nan,
        'centroid_dist_l11': float(np.nanmean([
            cosine_dist(centroids[ti, -1], centroids[tj, -1])
            for tj in range(n_types) if tj != ti and not np.any(np.isnan(centroids[tj, -1]))
        ])),
    })
    print(f"  {tname}: boot_var_L11={boot_var[-1]:.6f}, "
          f"pairwise_dist_L11={np.mean(boot_dist_l11) if boot_dist_l11 else np.nan:.4f}")

pd.DataFrame(boot_rows).to_csv(OUT / "h3g_bootstrap_variance.csv", index=False)

# ── Summary JSON ──────────────────────────────────────────────────────────────
related_rhos   = [r['spearman_rho'] for r in spear_rows if r['is_related']  and not np.isnan(r['spearman_rho'])]
unrelated_rhos = [r['spearman_rho'] for r in spear_rows if not r['is_related'] and not np.isnan(r['spearman_rho'])]

converging_related   = [r for r in spear_rows if r['is_related']     and r.get('spearman_rho', 0) < 0]
converging_unrelated = [r for r in spear_rows if not r['is_related'] and r.get('spearman_rho', 0) < 0]

summary = {
    "n_cell_types": n_types,
    "n_cells_total": int(n_cells_total),
    "n_cells_valid": int(cell_valid.sum()),
    "cells_per_type": counts_per_type,
    "n_pairs_total": len(spear_rows),
    "n_related_pairs": len([r for r in spear_rows if r['is_related']]),
    "n_converging_related": len(converging_related),
    "n_converging_unrelated": len(converging_unrelated),
    "mean_rho_related":   float(np.mean(related_rhos))   if related_rhos   else None,
    "mean_rho_unrelated": float(np.mean(unrelated_rhos)) if unrelated_rhos else None,
    "pseudotime_verdict": (
        "lineage-specific convergence" if len(converging_related) > len(converging_unrelated)
        else "lineage-agnostic compression"
    ),
    "wall_clock_seconds": round(time.time() - t_total, 1),
}
with open(OUT / "h3g_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n--- H3-G Summary ---")
print(f"  Related pair mean rho:   {summary['mean_rho_related']}")
print(f"  Unrelated pair mean rho: {summary['mean_rho_unrelated']}")
print(f"  Verdict: {summary['pseudotime_verdict']}")
print(f"\nTotal time: {time.time()-t_total:.1f}s")
print("Saved: h3g_centroids.npy, h3g_centroid_distances.csv, h3g_distance_depth_spearman.csv, h3g_bootstrap_variance.csv, h3g_summary.json")
