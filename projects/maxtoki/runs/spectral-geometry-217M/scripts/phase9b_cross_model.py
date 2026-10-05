"""Phase 9b — cross-model alignment between MaxToki-217M and Geneformer V2-316M.

Both models share the 20,275-token Ensembl-ID gene vocabulary (`<pad>`, `<mask>`,
`<cls>`, `<eos>` at ids 0..3, then ENSG ids 4..20274). This makes the cross-
model alignment direct: load each model's input-embedding layer, restrict to
the 1500 HVGs from Phase 0, and compare.

Tests (per the spec §9b):
  1. Cosine similarity between MaxToki layer-0 (post-embed) gene embeddings
     and Geneformer's `bert.embeddings.word_embeddings.weight` static layer.
     Permutation test against gene-label-shuffle null.
  2. STRING PPI co-pole test on Geneformer static embeddings (SV1..SV7).
     Compare to the analogous test on MaxToki (Phase 4 layer 0).
  3. B-cell marker precision@10 on Geneformer static embeddings (compare to
     MaxToki Phase 7).

Outputs:
  outputs/phase9b/cross_model_cosine.json
  outputs/phase9b/geneformer_static_ppi.csv
  outputs/phase9b/geneformer_static_bcell_prec10.json
  outputs/phase9b/summary.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402

PHASE0_DIR = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
OUT_DIR = PROJ / "runs/spectral-geometry-217M/outputs/phase9b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENEFORMER_DIR = Path(
    "<HF_CACHE>/hub/models--ctheodoris--Geneformer"
    "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M"
)
MAXTOKI_DIR = PROJ / "setup/MaxToki-217M-HF"

STRING_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"

print("=" * 70)
print("PHASE 9b — cross-model alignment (MaxToki vs Geneformer V2-316M)")
print("=" * 70)
t_start = time.time()

# ---------------------------------------------------------------
# 1. Load both static embedding layers
# ---------------------------------------------------------------
def load_safetensor_key(path: Path, key_substrs: list[str]) -> np.ndarray:
    """Find a key whose name contains all substrings, return tensor as np float32."""
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if all(s in k for s in key_substrs):
                return f.get_tensor(k).cpu().float().numpy()
    raise KeyError(f"no key with {key_substrs} in {path}; available: {keys[:5]}...")


print("\n[1] Loading Geneformer static embeddings...")
gfm_emb = load_safetensor_key(
    GENEFORMER_DIR / "model.safetensors",
    ["embeddings.word_embeddings"],
)
print(f"  Geneformer embed: {gfm_emb.shape} dtype={gfm_emb.dtype}")

print("\n[2] Loading MaxToki static embeddings...")
mt_emb = load_safetensor_key(
    MAXTOKI_DIR / "model.safetensors",
    ["embed_tokens"],
)
print(f"  MaxToki embed:    {mt_emb.shape} dtype={mt_emb.dtype}")

assert gfm_emb.shape[0] == mt_emb.shape[0] == 20275, "Vocab size mismatch"

# ---------------------------------------------------------------
# 2. Load HVG list + map to token ids (shared)
# ---------------------------------------------------------------
gene_features = pd.read_csv(PHASE0_DIR / "gene_features.csv")
hvg_ens = gene_features["ensembl_id"].astype(str).tolist()
hvg_symbols = gene_features["symbol"].astype(str).tolist()
mt_tok_ids = gene_features["maxtoki_token_id"].astype(int).to_numpy()

with open(GENEFORMER_DIR.parent / "geneformer/token_dictionary_gc104M.pkl", "rb") as f:
    gfm_tokenizer = pickle.load(f)
# Sanity: the tokenizers should be identical for ENSG entries
gfm_tok_ids = np.array([gfm_tokenizer.get(e, -1) for e in hvg_ens], dtype=np.int64)
n_match = int((gfm_tok_ids == mt_tok_ids).sum())
print(f"  HVG token-id agreement (MaxToki ↔ Geneformer): {n_match}/{len(hvg_ens)}")
assert n_match >= len(hvg_ens) - 5, "Tokenizers diverge on HVG set"

valid = (gfm_tok_ids >= 0) & (gfm_tok_ids < gfm_emb.shape[0])
gfm_static_hvg = gfm_emb[gfm_tok_ids[valid]]      # (n_hvg, 1152)
mt_static_hvg = mt_emb[mt_tok_ids[valid]]          # (n_hvg, 1232)
hvg_ens_v = [hvg_ens[i] for i, ok in enumerate(valid) if ok]
hvg_sym_v = [hvg_symbols[i] for i, ok in enumerate(valid) if ok]
n_hvg = len(hvg_ens_v)
print(f"  shared HVG static embeddings: MaxToki {mt_static_hvg.shape}  "
      f"Geneformer {gfm_static_hvg.shape}")

# ---------------------------------------------------------------
# 3. Cross-model cosine alignment + permutation test
# ---------------------------------------------------------------
def normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


print("\n[3] Cross-model cosine alignment...")
mt_n = normalise(mt_static_hvg)
gfm_n = normalise(gfm_static_hvg)

# Per-gene cosine similarity is undefined directly (different dims), so we
# compare gene-by-gene similarity STRUCTURE: build a (n_hvg × n_hvg) cosine
# matrix per model, then compare matrices.
mt_S = mt_n @ mt_n.T
gfm_S = gfm_n @ gfm_n.T

# Off-diagonal vector
iu = np.triu_indices(n_hvg, k=1)
mt_v = mt_S[iu]
gfm_v = gfm_S[iu]
pearson = float(np.corrcoef(mt_v, gfm_v)[0, 1])
spearman_rs = float(pd.Series(mt_v).corr(pd.Series(gfm_v), method="spearman"))
print(f"  Pearson  (similarity matrices, off-diag): {pearson:.4f}")
print(f"  Spearman (similarity matrices, off-diag): {spearman_rs:.4f}")

# Permutation null: shuffle Geneformer's gene labels
N_PERM = 500
rng = np.random.default_rng(42)
null_pearsons = np.zeros(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(n_hvg)
    gS = gfm_S[perm][:, perm]
    null_pearsons[k] = float(np.corrcoef(mt_v, gS[iu])[0, 1])
p_emp = float((null_pearsons >= pearson).mean())
print(f"  null mean: {null_pearsons.mean():.4f}  null SD: {null_pearsons.std():.4f}  "
      f"empirical p: {p_emp:.4f}")

cosine_out = {
    "n_hvg_shared": n_hvg,
    "pearson_off_diag": round(pearson, 4),
    "spearman_off_diag": round(spearman_rs, 4),
    "null_mean": round(float(null_pearsons.mean()), 4),
    "null_sd": round(float(null_pearsons.std()), 4),
    "p_empirical": round(p_emp, 4),
    "n_perm": N_PERM,
    "verdict": (
        "alignment significantly above shuffled null" if p_emp < 0.05
        else "alignment NOT distinguishable from shared-vocab null (paper-style)"
    ),
}
with open(OUT_DIR / "cross_model_cosine.json", "w") as f:
    json.dump(cosine_out, f, indent=2)

# ---------------------------------------------------------------
# 4. Geneformer static-embedding SVD → STRING PPI co-pole test
# ---------------------------------------------------------------
print("\n[4] Geneformer static SVD + STRING PPI co-pole test...")
Xc = gfm_static_hvg - gfm_static_hvg.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
gfm_proj = U[:, :7] * S[:7]  # (n_hvg, 7)
print(f"  SVD top-7 singular values: {S[:7]}")

# Load STRING PPI edges
with open(STRING_JSON) as f:
    string_data = json.load(f)
# Pick the highest-confidence pair set available
pair_sets = sorted(string_data.keys())
print(f"  available STRING pair sets: {pair_sets}")
pairs_700 = string_data.get("pairs_700") or string_data[pair_sets[0]]
print(f"  using pairs_700 with {len(pairs_700)} pairs")

# Build symbol→hvg index
sym_to_hvg = {s.upper(): i for i, s in enumerate(hvg_sym_v)}
ppi_pairs = []
for pair in pairs_700:
    if isinstance(pair, list) and len(pair) == 2:
        a, b = pair[0].upper(), pair[1].upper()
    elif isinstance(pair, dict):
        a, b = pair["gene_a"].upper(), pair["gene_b"].upper()
    else:
        continue
    if a in sym_to_hvg and b in sym_to_hvg and a != b:
        ppi_pairs.append((sym_to_hvg[a], sym_to_hvg[b]))
print(f"  PPI pairs intersected with HVG: {len(ppi_pairs)}")

K = 52
N_PERM_PPI = 500

def copole_test(proj_axis: np.ndarray, pairs: list[tuple[int, int]]) -> dict:
    sorted_idx = np.argsort(proj_axis)
    top_pole = set(sorted_idx[-K:].tolist())
    bot_pole = set(sorted_idx[:K].tolist())
    n_p = len(pairs)
    obs = sum(
        1 for a, b in pairs
        if (a in top_pole and b in top_pole) or (a in bot_pole and b in bot_pole)
    )
    obs_rate = obs / max(n_p, 1)
    null_rates = np.zeros(N_PERM_PPI)
    n_genes = len(proj_axis)
    rng_ = np.random.default_rng(42)
    for k in range(N_PERM_PPI):
        perm = rng_.permutation(n_genes)
        new_top = set(perm[-K:].tolist())
        new_bot = set(perm[:K].tolist())
        cnt = sum(1 for a, b in pairs if (a in new_top and b in new_top) or (a in new_bot and b in new_bot))
        null_rates[k] = cnt / max(n_p, 1)
    z = (obs_rate - null_rates.mean()) / max(null_rates.std(), 1e-12)
    p = float((null_rates >= obs_rate).mean())
    return {
        "observed_rate": round(obs_rate, 6),
        "null_mean": round(float(null_rates.mean()), 6),
        "null_std": round(float(null_rates.std()), 6),
        "z": round(float(z), 3),
        "p_empirical": round(p, 4),
        "n_pairs": n_p,
    }

ppi_rows = []
for axis_i in range(7):
    res = copole_test(gfm_proj[:, axis_i], ppi_pairs)
    res["axis"] = f"SV{axis_i + 1}"
    ppi_rows.append(res)
ppi_df = pd.DataFrame(ppi_rows)
ppi_df.to_csv(OUT_DIR / "geneformer_static_ppi.csv", index=False)
print(ppi_df.to_string(index=False))

# Best Geneformer static axis
best_idx = int(ppi_df["z"].idxmax())
gfm_best = ppi_df.iloc[best_idx].to_dict()
print(f"  Geneformer best static axis: {gfm_best['axis']}  z={gfm_best['z']}  p={gfm_best['p_empirical']}")

# ---------------------------------------------------------------
# 5. B-cell marker precision@10 on Geneformer static embeddings
# ---------------------------------------------------------------
B_CELL_MARKERS = {"CD19", "CD79A", "CD79B", "MS4A1", "BLK", "VPREB3", "FCRL1", "PAX5",
                  "BANK1", "BACH2", "BCL6", "PRDM1", "IRF4", "IRF8"}

print("\n[5] Geneformer static B-cell marker precision@10...")
b_idx = [i for i, s in enumerate(hvg_sym_v) if s.upper() in B_CELL_MARKERS]
print(f"  B-cell markers in HVG: {len(b_idx)} → {[hvg_sym_v[i] for i in b_idx]}")

if b_idx:
    gfm_n_local = normalise(gfm_static_hvg)
    sim = gfm_n_local @ gfm_n_local.T
    np.fill_diagonal(sim, -np.inf)
    precs = []
    for src in b_idx:
        top10 = np.argsort(-sim[src])[:10]
        prec = sum(1 for j in top10 if j in b_idx) / 10.0
        precs.append(prec)
    mean_prec = float(np.mean(precs))
    # Bootstrap null: random gene picks
    rng_ = np.random.default_rng(42)
    null_means = np.zeros(500)
    for k in range(500):
        rand_b = rng_.choice(n_hvg, size=len(b_idx), replace=False)
        rprecs = []
        for src in rand_b:
            top10 = np.argsort(-sim[src])[:10]
            rprecs.append(sum(1 for j in top10 if j in rand_b) / 10.0)
        null_means[k] = float(np.mean(rprecs))
    z_prec = (mean_prec - null_means.mean()) / max(null_means.std(), 1e-12)
    bcell_out = {
        "n_markers_in_hvg": len(b_idx),
        "mean_precision_at_10": round(mean_prec, 4),
        "null_mean": round(float(null_means.mean()), 4),
        "null_std": round(float(null_means.std()), 4),
        "z": round(float(z_prec), 3),
    }
else:
    bcell_out = {"error": "no B-cell markers in HVG list"}
with open(OUT_DIR / "geneformer_static_bcell_prec10.json", "w") as f:
    json.dump(bcell_out, f, indent=2)
print(f"  Geneformer B-cell prec@10: {bcell_out}")

# ---------------------------------------------------------------
# 6. Compare to MaxToki layer-0 (Phase 4 + Phase 7)
# ---------------------------------------------------------------
mt_phase4 = pd.read_csv(PROJ / "runs/spectral-geometry-217M/outputs/phase4/ppi_copole_per_axis_per_layer.csv")
mt_layer0 = mt_phase4[mt_phase4["layer"] == 0]
mt_max_z_l0 = float(mt_layer0["z"].max()) if len(mt_layer0) else float("nan")

summary = {
    "phase": "9b",
    "models": {
        "model_a": {"name": "MaxToki-217M-HF", "d": int(mt_emb.shape[1]), "n_layers": 11},
        "model_b": {"name": "Geneformer V2-316M", "d": int(gfm_emb.shape[1]), "n_layers": 18},
        "shared_vocab_size": int(gfm_emb.shape[0]),
        "n_hvg_compared": n_hvg,
    },
    "cross_model_cosine_alignment": cosine_out,
    "geneformer_static_ppi_best_axis": gfm_best,
    "maxtoki_layer0_ppi_max_z": round(mt_max_z_l0, 3),
    "ppi_convergent_across_architectures": gfm_best["p_empirical"] < 0.05 and mt_max_z_l0 > 1.5,
    "geneformer_static_bcell_prec10": bcell_out,
    "interpretation": (
        "Both models share the 20,275-token Ensembl-ID vocabulary (token "
        "dictionaries identical). Cross-model alignment is therefore a clean "
        "test of architecture-vs-shared-prior: does the Llama-style decoder "
        "(MaxToki) and BERT-style encoder (Geneformer) converge on the same "
        "biological geometry from a shared vocabulary?"
    ),
    "wall_clock_seconds": round(time.time() - t_start, 1),
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n" + "=" * 70)
print("SUMMARY:")
print(json.dumps(summary, indent=2))
print(f"\nPHASE 9b COMPLETE — {time.time()-t_start:.1f}s")
