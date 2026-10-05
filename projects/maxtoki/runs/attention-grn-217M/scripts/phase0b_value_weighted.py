"""Phase 0b — value-weighted edge extraction on MaxToki-217M.

For the same 2000 K562 NT control cells used in Phase 0, capture the
post-attention pre-o_proj context tensor A_vh = softmax(QK/sqrt(d)) @ V per
layer / head / position / head_dim, then build per-layer per-head pairwise
cosine-similarity edges and evaluate TRRUST recovery + Phase-1-style
per-perturbation AUROC.

This tests whether the regulatory signal is hidden in the value pathway
rather than the attention pattern itself (paper Supp Note 18).

Accumulation strategy: we store a per-layer per-gene *sum* of A_vh vectors
across cells (length-normalized to unit vectors first so the eventual cosine
similarity on the averaged vectors is well-defined) plus a per-gene count,
then at the end compute cos-sim between averaged per-gene vectors for each
layer / head. Also accumulate a "centroid" cosine: average each gene's
length-normalized vector across cells, then cosine between means.
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
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests as _mult
from scipy import stats
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

PHASE0 = PROJ / "runs/attention-grn-217M/outputs/phase0"
OUT_DIR = PROJ / "runs/attention-grn-217M/outputs/phase0b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPLOGLE_H5 = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
SYM2ENS_PKL = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"
TRRUST_TSV = BIOM_ROOT / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

N_CTRL = int(os.environ.get("PHASE0B_N_CTRL", "2000"))
N_HVG = 1500
MAX_LEN = 2048
CELL_LINE = "k562"
DEVICE = os.environ.get("PHASE0B_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
PRIMARY_LAYER = int(os.environ.get("PHASE1_PRIMARY_LAYER", "8"))

print("=" * 70)
print(f"PHASE 0b — value-weighted edges (MaxToki-217M)  N_ctrl={N_CTRL}  device={DEVICE}")
print("=" * 70)

# Load Phase-0 artefacts so we reuse the same HVG set
print("\n[1] Loading Phase 0 artefacts...")
gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
n_hvg = len(gene_features)
hvg_symbols_upper = [s.upper() for s in gene_features["symbol"]]
symbol_to_hvg = {s: i for i, s in enumerate(hvg_symbols_upper)}
var_idx = gene_features["var_idx"].to_numpy(dtype=np.int64)

print(f"  HVG: {n_hvg}")

# Reload control cell sample (same seed as Phase 0)
rng = np.random.default_rng(SEED)
with h5py.File(REPLOGLE_H5, "r") as f:
    cl_cats = [(s.decode() if isinstance(s, bytes) else s).lower()
               for s in f["obs"]["cell_line"]["categories"][:]]
    cl_codes = f["obs"]["cell_line"]["codes"][:]
    k562 = (cl_codes == cl_cats.index(CELL_LINE))
    pg_cats = [s.decode() if isinstance(s, bytes) else s
               for s in f["obs"]["gene"]["categories"][:]]
    pg_codes = f["obs"]["gene"]["codes"][:]
    nt_codes_set = {i for i, g in enumerate(pg_cats)
                    if "non-targeting" in g.lower() or "nontargeting" in g.lower()}
    ctrl_all = np.where(k562 & np.isin(pg_codes, list(nt_codes_set)))[0]
    ctrl_idx = np.sort(rng.choice(ctrl_all, size=N_CTRL, replace=False))
    n_cells_total, n_genes_total = f["X"].shape
    # Load expression block
    X_ctrl = np.empty((N_CTRL, n_genes_total), dtype=np.float32)
    for i in range(0, N_CTRL, 100):
        X_ctrl[i:i+100] = f["X"][ctrl_idx[i:i+100], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s
                   for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
var_ens = [sym2ens.get(s) for s in var_symbols]

# tokenizer + var mapping
tokenizer = MaxTokiTokenizer()
var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(var_ens)
var_to_hvg = -np.ones(n_genes_total, dtype=np.int64)
for hi, vi in enumerate(var_idx):
    var_to_hvg[vi] = hi
mt_var_to_hvg = var_to_hvg[var_indices_full]

print(f"\n[2] Loading MaxToki on {DEVICE}...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layers, n_heads, hidden = xt.n_layers, xt.n_heads, xt.hidden_size
head_dim = hidden // n_heads
print(f"  L={n_layers}  H={n_heads}  D={hidden}  head_dim={head_dim}")

# Accumulators: sum of length-normalized value-context vectors per (layer, head, gene).
# Shape: (L, H, n_hvg, head_dim) float32. At n_hvg=1500: 11 * 8 * 1500 * 154 * 4 = ~81 MB. Tiny.
ctx_sum = np.zeros((n_layers, n_heads, n_hvg, head_dim), dtype=np.float32)
gene_counts = np.zeros(n_hvg, dtype=np.int32)
per_cell_times = []
per_cell_seq_lens = []

print(f"\n[3] Forward passes + accumulation ({N_CTRL} cells)...")
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
    hvg_ids = hvg_pp[pos_idx]
    if len(pos_idx) < 2:
        continue

    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, ctx = xt.forward_with_value_context(input_ids)
    # ctx[li] shape: (1, n_heads, seq_len, head_dim)
    for li in range(n_layers):
        vh = ctx[li][0].numpy()  # (H, seq, d_h)
        sub = vh[:, pos_idx, :]   # (H, k, d_h)
        # L2-normalize each vector so the running average stays well-conditioned
        norm = np.linalg.norm(sub, axis=2, keepdims=True) + 1e-8
        subn = sub / norm
        # Scatter-add by HVG id
        for h in range(n_heads):
            np.add.at(ctx_sum[li, h], hvg_ids, subn[h])
    # pair counts share across layers/heads
    gene_counts[hvg_ids] += 1
    per_cell_times.append(time.time() - t0)
    per_cell_seq_lens.append(len(pos))
    if (ci + 1) % 200 == 0 or (ci + 1) == N_CTRL:
        tqdm.write(
            f"  [{ci+1}/{N_CTRL}] mean fwd {np.mean(per_cell_times[-200:]):.2f}s  "
            f"seq_len {np.mean(per_cell_seq_lens[-200:]):.0f}"
        )

print(f"\n  total forward time: {sum(per_cell_times):.1f}s  "
      f"mean cell: {np.mean(per_cell_times):.2f}s")

# Finalize per-gene averaged value-context vectors, then compute per-layer,
# per-head pairwise cosine similarity edges.
print("\n[4] Building per-layer per-head VW cosine-similarity edges...")
denom = gene_counts.astype(np.float32)
denom[denom == 0] = np.nan
# Mean unit-vectors per gene: shape (L, H, n_hvg, head_dim)
ctx_mean = ctx_sum / denom[None, None, :, None]
ctx_mean = np.nan_to_num(ctx_mean, nan=0.0)

# Per-gene cosine similarity: (L, H, G, G) = normalize then dot
vw_edges_per_head = np.zeros((n_layers, n_heads, n_hvg, n_hvg), dtype=np.float32)
for li in range(n_layers):
    for h in range(n_heads):
        v = ctx_mean[li, h]  # (G, d)
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vn = v / norm
        edges = (vn @ vn.T).astype(np.float32)
        # Mask rows/cols where the gene had no observations
        mask = (gene_counts == 0)
        edges[mask, :] = 0.0
        edges[:, mask] = 0.0
        vw_edges_per_head[li, h] = edges
vw_edges_layer_mean = vw_edges_per_head.mean(axis=1)

np.save(OUT_DIR / "vw_edges_per_head.npy", vw_edges_per_head)
np.save(OUT_DIR / "vw_edges_layer_mean.npy", vw_edges_layer_mean)
np.save(OUT_DIR / "vw_gene_counts.npy", gene_counts)
print(f"  vw_edges_per_head:   {vw_edges_per_head.shape}")
print(f"  vw_edges_layer_mean: {vw_edges_layer_mean.shape}")

# ------------------------------------------------------------------------
# Evaluation vs TRRUST (global AUROC) and vs Phase-1 DE perturbations
# (per-perturbation AUROC).
# ------------------------------------------------------------------------
print("\n[5] TRRUST recovery AUROC per layer (mean-over-heads)...")
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_edges_bool = np.zeros((n_hvg, n_hvg), dtype=np.int8)
for _, r in trrust.iterrows():
    i = symbol_to_hvg.get(r["tf"]); j = symbol_to_hvg.get(r["target"])
    if i is not None and j is not None and i != j:
        trrust_edges_bool[i, j] = 1
tf_mask = np.zeros(n_hvg, dtype=bool)
for i in range(n_hvg):
    if trrust_edges_bool[i].sum() >= 3:
        tf_mask[i] = True

def _trrust_auroc(em):
    scores, labels = [], []
    for tf_i in np.where(tf_mask)[0]:
        m = np.ones(n_hvg, dtype=bool); m[tf_i] = False
        scores.append(em[tf_i][m]); labels.append(trrust_edges_bool[tf_i][m])
    s = np.concatenate(scores); l = np.concatenate(labels)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    return float(roc_auc_score(l, s))

# Attention L8 (Phase 0 primary) for comparison
attn_le = np.load(PHASE0 / "attention_edges_layer_mean.npy")[PRIMARY_LAYER]
np.fill_diagonal(attn_le, 0.0)
auc_attn_primary = _trrust_auroc(attn_le)

# Per-layer VW AUROC (mean over heads)
vw_per_layer_auc = []
for li in range(n_layers):
    em = vw_edges_layer_mean[li].copy()
    np.fill_diagonal(em, 0.0)
    vw_per_layer_auc.append(_trrust_auroc(em))
    print(f"  L{li:2d}: VW-TRRUST AUROC = {vw_per_layer_auc[-1]:.4f}")

print(f"\n  attention L{PRIMARY_LAYER} TRRUST AUROC (reference): {auc_attn_primary:.4f}")

# Per-perturbation AUROC against Phase-1 DE targets (re-use Phase 1's DE masks
# by re-running the same DE call on 5000 NT + K562 perturbations).
print("\n[6] Per-perturbation AUROC (re-using Phase-1 DE procedure)...")
LFC_THRESHOLD = 0.5
DE_PVAL_THRESHOLD = 0.05
MIN_CELLS_PER_PERT = 30
MIN_DE_POSITIVES = 3

t0 = time.time()
with h5py.File(REPLOGLE_H5, "r") as f:
    # NT expression
    nt_idx_all = np.where(k562 & np.isin(pg_codes, list(nt_codes_set)))[0]
    if len(nt_idx_all) > 5000:
        nt_idx_all = np.sort(np.random.default_rng(SEED).choice(nt_idx_all, 5000, replace=False))
    X_nt = np.empty((len(nt_idx_all), n_hvg), dtype=np.float32)
    for i in range(0, len(nt_idx_all), 500):
        rows = nt_idx_all[i:i+500]
        X_nt[i:i+500] = f["X"][rows, :][:, var_idx]
    rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_nt = np.log1p(X_nt / rs * 1e4)
    mu_nt = X_nt.mean(axis=0); var_nt = X_nt.var(axis=0, ddof=1); N_nt = len(X_nt)

    k562_cells = np.where(k562)[0]; k562_codes = pg_codes[k562]
    pert_records = []
    for code in np.unique(k562_codes):
        cat = pg_cats[code]
        if "non-targeting" in cat.lower() or "nontargeting" in cat.lower():
            continue
        if sym2ens.get(cat) is None:
            continue
        pert_hvg = symbol_to_hvg.get(cat.upper())
        if pert_hvg is None:
            continue
        sel = k562_cells[k562_codes == code]
        if len(sel) < MIN_CELLS_PER_PERT:
            continue
        rows_sorted = np.sort(sel)
        X_p = np.empty((len(rows_sorted), n_hvg), dtype=np.float32)
        for i in range(0, len(rows_sorted), 300):
            r = rows_sorted[i:i+300]
            X_p[i:i+300] = f["X"][r, :][:, var_idx]
        rs_p = X_p.sum(axis=1, keepdims=True); rs_p[rs_p == 0] = 1.0
        X_p = np.log1p(X_p / rs_p * 1e4)
        mu_p = X_p.mean(axis=0); var_p = X_p.var(axis=0, ddof=1); N_p = len(X_p)
        se = np.sqrt(var_p / N_p + var_nt / N_nt); se[se == 0] = 1e-12
        t_stat = (mu_p - mu_nt) / se
        df = (var_p / N_p + var_nt / N_nt) ** 2 / (
            (var_p / N_p) ** 2 / max(N_p - 1, 1)
            + (var_nt / N_nt) ** 2 / max(N_nt - 1, 1))
        pval = 2.0 * stats.t.sf(np.abs(t_stat), df)
        _, p_bh, _, _ = _mult(np.nan_to_num(pval, nan=1.0), method="fdr_bh")
        lfc = mu_p - mu_nt
        is_de = (np.abs(lfc) >= LFC_THRESHOLD) & (p_bh < DE_PVAL_THRESHOLD)
        valid = np.ones(n_hvg, dtype=bool); valid[pert_hvg] = False
        is_de_eff = is_de & valid
        if is_de_eff.sum() < MIN_DE_POSITIVES or (valid.sum() - is_de_eff.sum()) < MIN_DE_POSITIVES:
            continue
        pert_records.append({"pert_symbol": cat, "pert_hvg": pert_hvg,
                             "is_de": is_de_eff, "valid": valid})

print(f"  evaluable perturbations: {len(pert_records)}  ({time.time()-t0:.1f}s)")

rows = []
for li in range(n_layers):
    vw = vw_edges_layer_mean[li].copy(); np.fill_diagonal(vw, 0.0)
    aucs = []
    for rec in pert_records:
        ph = rec["pert_hvg"]
        y = rec["is_de"][rec["valid"]].astype(int)
        pred = vw[ph, :][rec["valid"]]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            aucs.append(roc_auc_score(y, pred))
        except ValueError:
            continue
    rows.append({
        "layer": li,
        "mean_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "median_auroc": float(np.median(aucs)) if aucs else float("nan"),
        "n_perts": int(len(aucs)),
    })
vw_per_layer_pert = pd.DataFrame(rows)
vw_per_layer_pert.to_csv(OUT_DIR / "vw_per_layer_per_pert_auroc.csv", index=False)
print(vw_per_layer_pert.to_string(index=False))

# Wilcoxon: VW at primary layer vs attention primary layer (per-perturbation)
print("\n[7] Paired Wilcoxon VW-primary vs attention-primary (per-perturbation)...")
attn_primary = np.load(PHASE0 / "attention_edges_layer_mean.npy")[PRIMARY_LAYER].copy()
np.fill_diagonal(attn_primary, 0.0)
vw_primary = vw_edges_layer_mean[PRIMARY_LAYER].copy(); np.fill_diagonal(vw_primary, 0.0)

attn_aucs, vw_aucs = [], []
for rec in pert_records:
    ph = rec["pert_hvg"]
    y = rec["is_de"][rec["valid"]].astype(int)
    pa = attn_primary[ph, :][rec["valid"]]
    pv = vw_primary[ph, :][rec["valid"]]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    try:
        attn_aucs.append(roc_auc_score(y, pa))
        vw_aucs.append(roc_auc_score(y, pv))
    except ValueError:
        continue
w, p = stats.wilcoxon(vw_aucs, attn_aucs, zero_method="zsplit")
delta = float(np.mean(vw_aucs) - np.mean(attn_aucs))
print(f"  VW primary mean AUROC: {np.mean(vw_aucs):.4f}")
print(f"  attn primary mean AUROC: {np.mean(attn_aucs):.4f}")
print(f"  Δ = {delta:+.4f}  Wilcoxon W={w:.0f}  p={p:.3e}")

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump({
        "pipeline": "attention-grn (Phase 0b)",
        "model": "MaxToki-217M-HF",
        "n_ctrl": N_CTRL,
        "n_hvg": n_hvg,
        "primary_layer": PRIMARY_LAYER,
        "device": DEVICE,
        "mean_seq_len": float(np.mean(per_cell_seq_lens)),
        "mean_fwd_seconds": float(np.mean(per_cell_times)),
        "total_fwd_seconds": float(sum(per_cell_times)),
        "trrust_auroc_per_layer_vw": vw_per_layer_auc,
        "trrust_auroc_attention_primary": auc_attn_primary,
        "wilcoxon_vw_vs_attn_primary": {
            "mean_auc_vw": float(np.mean(vw_aucs)),
            "mean_auc_attn": float(np.mean(attn_aucs)),
            "delta": delta,
            "W": float(w),
            "p": float(p),
            "n_perts": int(len(vw_aucs)),
        },
    }, f, indent=2)
print(f"\nPHASE 0b COMPLETE — outputs: {OUT_DIR}")
