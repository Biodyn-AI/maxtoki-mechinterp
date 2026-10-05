"""Experiment 3 rerun — Trajectory-guided feature steering (fixes L11 crash).

Differences from experiments_2_3.py:
  - Hook layer index clamped to min(li, n_layers-1): for MaxToki-217M (n_layers=11),
    li=11 → hook on layers[10], i.e. modify the output of the last block so the
    final residual stream carries the steering delta. Prior script crashed on
    layers[11] lookup.
  - Persists one JSON per feature with alpha_results + top-gene logit deltas.
  - Caches state_signatures.npz (pseudotime, clean logit signatures).
  - Resume-safe: skips features whose JSON already exists.

Runs ONLY Experiment 3. Experiments 1 & 2 are already persisted.
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
import pandas as pd
import scipy.sparse as sp
import torch

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402
from dataset_loader import SYM2ENS_PKL  # noqa: E402

SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
OUT3 = PROJ / "runs/exhaustive-mapping-217M/outputs/experiment3"
OUT3.mkdir(parents=True, exist_ok=True)

TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
D_SAE = 4928
STEERING_LAYERS = [0, 3, 6, 9, 11]
ALPHAS = [2, 5]
N_CELLS = 200  # same as the original run so pseudotime splits match

print("=" * 70)
print("EXPERIMENT 3 RERUN — steering with L11 fix + per-feature persistence")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Load SAEs for steering layers
# ---------------------------------------------------------------
saes: dict[int, TopKSAE] = {}
mus: dict[int, torch.Tensor] = {}
for li in STEERING_LAYERS:
    ckpt = torch.load(
        SAE_DIR / f"layer_{li:02d}/sae_final.pt",
        map_location="cpu", weights_only=False,
    )
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval()
    saes[li] = sae
    mus[li] = ckpt["mu"]
print(f"  SAEs loaded for layers {STEERING_LAYERS}")

# ---------------------------------------------------------------
# 2. Load cached switch features (from previous run)
# ---------------------------------------------------------------
switch_csv = OUT3 / "switch_features.csv"
assert switch_csv.exists(), f"switch_features.csv missing at {switch_csv}"
switch_df = pd.read_csv(switch_csv)
print(f"  {len(switch_df)} switch features loaded from cache")

steer_features = []
for li in STEERING_LAYERS:
    layer_switches = switch_df[switch_df["layer"] == li].head(3)
    for _, r in layer_switches.iterrows():
        steer_features.append({
            "layer": int(r["layer"]),
            "feature": int(r["feature"]),
            "switch_score": float(r["switch_score"]),
            "direction": str(r["direction"]),
        })
print(f"  selected {len(steer_features)} features to steer (top 3 per layer)")

# ---------------------------------------------------------------
# 3. Tokenizer + symbol↔ENSG + reverse token map
# ---------------------------------------------------------------
tokenizer = MaxTokiTokenizer()
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
ens2sym = {v: k for k, v in sym2ens.items()}
tok2ens = {v: k for k, v in tokenizer.gene_token_dict.items()}  # int → ENSG

def token_to_gene(tid: int) -> str:
    ens = tok2ens.get(int(tid))
    if ens is None:
        return f"<tok{tid}>"
    sym = ens2sym.get(ens)
    return sym if sym else ens

# ---------------------------------------------------------------
# 4. Load Tabula Sapiens immune cells + pseudotime (same seed as prior run)
# ---------------------------------------------------------------
print("\n  Loading Tabula Sapiens immune cells...")
adata = ad.read_h5ad(str(TS_H5), backed="r")
rng_ts = np.random.default_rng(SEED + 3333)
ts_idx = np.sort(rng_ts.choice(adata.n_obs, size=N_CELLS, replace=False))
X_ts = adata[ts_idx].X
if sp.issparse(X_ts) or hasattr(X_ts, "toarray"):
    X_ts = X_ts.toarray()
X_ts = np.asarray(X_ts, dtype=np.float32)
var_ens_ts = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
adata.file.close()

rs = X_ts.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_ts / rs * 1e4)

from sklearn.decomposition import PCA
pca1 = PCA(n_components=1, random_state=SEED).fit_transform(X_log).flatten()
pseudotime = (pca1 - pca1.min()) / (pca1.max() - pca1.min() + 1e-12)
print(f"  {N_CELLS} cells, pseudotime range [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")

early_mask = pseudotime < np.percentile(pseudotime, 25)
late_mask = pseudotime > np.percentile(pseudotime, 75)
print(f"  early: {int(early_mask.sum())} cells   late: {int(late_mask.sum())} cells")

var_idx_ts, var_tok_ts, var_med_ts = tokenizer.make_var_mapping(var_ens_ts.tolist())

# Save gene-space signatures (mean log-normalised expression per tertile)
g_early_expr = X_log[early_mask].mean(axis=0)
g_late_expr = X_log[late_mask].mean(axis=0)

# ---------------------------------------------------------------
# 5. Extractor + helpers
# ---------------------------------------------------------------
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
n_layers = xt.n_layers
print(f"  model: n_layers={n_layers}, hidden_size={xt.hidden_size}, vocab={xt.model.config.vocab_size}")

def hook_layer_for(li: int) -> int:
    """Which transformer block's output to override to inject an SAE-layer-li patch.

    For li < n_layers: hook model.layers[li] (same as original convention,
    replacing that block's output). For li == n_layers (the final-residual
    tap), hook the last block [n_layers - 1] so the modification reaches
    the final norm + lm_head unchanged.
    """
    return min(li, n_layers - 1)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------------------------------------------------------------
# 6. Compute clean-logit signatures g_early_logits / g_late_logits
#    (needed for the spec's Δs = cos(z', late) - cos(z', early) - baseline metric)
# ---------------------------------------------------------------
sig_cache = OUT3 / "state_signatures.npz"
if sig_cache.exists():
    sig = np.load(sig_cache)
    g_early_logits = sig["g_early_logits"]
    g_late_logits = sig["g_late_logits"]
    clean_logits_cache = sig["clean_logits_per_cell"]
    print(f"  state signatures loaded from cache")
else:
    print("\n  Computing clean-logit signatures (one forward per cell)...")
    vocab_size = int(xt.model.config.vocab_size)
    clean_logits_cache = np.zeros((N_CELLS, vocab_size), dtype=np.float32)
    t0 = time.time()
    for ci in range(N_CELLS):
        cell = tokenizer.tokenize_cell(
            X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048,
        )
        if cell is None:
            continue
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)
        with torch.no_grad():
            out = xt.model(input_ids, use_cache=False, return_dict=True)
            logits_mean = out.logits[0].mean(dim=0).cpu().numpy().astype(np.float32)
        clean_logits_cache[ci] = logits_mean
        del out
        if DEVICE == "mps" and ci % 20 == 0:
            try: torch.mps.empty_cache()
            except Exception: pass
    g_early_logits = clean_logits_cache[early_mask].mean(axis=0)
    g_late_logits = clean_logits_cache[late_mask].mean(axis=0)
    print(f"  signatures computed in {time.time()-t0:.1f}s")
    np.savez_compressed(
        sig_cache,
        pseudotime=pseudotime,
        early_mask=early_mask,
        late_mask=late_mask,
        g_early_expr=g_early_expr,
        g_late_expr=g_late_expr,
        g_early_logits=g_early_logits,
        g_late_logits=g_late_logits,
        clean_logits_per_cell=clean_logits_cache,
    )
    print(f"  cached to {sig_cache}")

# Sanity: signatures should differ
cos_sig = cos(g_early_logits, g_late_logits)
print(f"  cos(g_early, g_late) = {cos_sig:.4f} (should be < 1.0)")

# ---------------------------------------------------------------
# 7. Per-feature steering loop (resume-safe)
# ---------------------------------------------------------------
print(f"\n  Steering {len(steer_features)} features over α ∈ {ALPHAS}...")

steering_summary_rows = []

for sf in steer_features:
    li = sf["layer"]; fi = sf["feature"]
    out_json = OUT3 / f"steering_F{fi:04d}_L{li:02d}.json"
    if out_json.exists():
        print(f"  [skip] L{li}_F{fi} (JSON already exists)")
        with open(out_json) as f:
            steering_summary_rows.append(json.load(f))
        continue

    hl = hook_layer_for(li)
    results_per_alpha: dict[str, dict] = {}
    vocab_size = int(xt.model.config.vocab_size)

    for alpha in ALPHAS:
        n_toward_direction = 0
        n_toward_maturity = 0
        n_steered = 0
        mean_logit_deltas: list[float] = []
        delta_s_vals: list[float] = []
        logit_delta_accum = np.zeros(vocab_size, dtype=np.float64)

        for ci in np.where(early_mask)[0]:
            cell = tokenizer.tokenize_cell(
                X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048,
            )
            if cell is None:
                continue
            input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

            with torch.no_grad():
                out_clean = xt.model(
                    input_ids, output_hidden_states=True,
                    use_cache=False, return_dict=True,
                )
                clean_logits = out_clean.logits[0].cpu().numpy().astype(np.float32)
                h_li = out_clean.hidden_states[li][0].cpu()

            with torch.no_grad():
                h_np = h_li.numpy().astype(np.float32)
                z = saes[li].encode(torch.from_numpy(h_np), mus[li])
                z_steered = z.clone()
                z_steered[:, fi] = z[:, fi] * alpha
                h_recon = saes[li].decode(z, mus[li])
                h_steered_recon = saes[li].decode(z_steered, mus[li])
                delta = h_steered_recon - h_recon
                h_patched = (h_li + delta.cpu()).to(DEVICE).unsqueeze(0)

            def _hook(module, input, output, _p=h_patched):
                if isinstance(output, tuple):
                    return (_p.to(output[0].device),) + output[1:]
                return _p.to(output.device)

            hook = xt.model.model.layers[hl].register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    out_steered = xt.model(
                        input_ids, output_hidden_states=False,
                        use_cache=False, return_dict=True,
                    )
                    steered_logits = out_steered.logits[0].cpu().numpy().astype(np.float32)
            finally:
                hook.remove()

            logit_delta = (steered_logits - clean_logits).mean(axis=0)
            logit_delta_accum += logit_delta
            mean_logit_deltas.append(float(np.mean(logit_delta)))

            # direction-based metric (original convention)
            if sf["direction"] == "positive":
                toward_dir = float(np.mean(logit_delta)) > 0
            else:
                toward_dir = float(np.mean(logit_delta)) < 0
            if toward_dir:
                n_toward_direction += 1

            # spec-compliant Δs metric: cos to late minus cos to early, relative to baseline
            z_clean = clean_logits.mean(axis=0)
            z_steered_mean = steered_logits.mean(axis=0)
            delta_s = (
                (cos(z_steered_mean, g_late_logits) - cos(z_steered_mean, g_early_logits))
                - (cos(z_clean, g_late_logits) - cos(z_clean, g_early_logits))
            )
            delta_s_vals.append(delta_s)
            if delta_s > 0:
                n_toward_maturity += 1

            n_steered += 1
            del out_clean, out_steered
            if DEVICE == "mps" and n_steered % 5 == 0:
                try: torch.mps.empty_cache()
                except Exception: pass

        mean_logit_delta_vec = logit_delta_accum / max(n_steered, 1)
        top_up_idx = np.argsort(-mean_logit_delta_vec)[:15]
        top_dn_idx = np.argsort(mean_logit_delta_vec)[:15]
        results_per_alpha[str(alpha)] = {
            "n_steered": n_steered,
            "frac_positive_direction": round(n_toward_direction / max(n_steered, 1), 4),
            "frac_positive_maturity": round(n_toward_maturity / max(n_steered, 1), 4),
            "mean_logit_delta_scalar": round(float(np.mean(mean_logit_deltas)), 6)
                if mean_logit_deltas else 0.0,
            "mean_delta_s": round(float(np.mean(delta_s_vals)), 6)
                if delta_s_vals else 0.0,
            "top_upregulated": [
                {"token_id": int(t), "gene": token_to_gene(int(t)),
                 "mean_logit_delta": round(float(mean_logit_delta_vec[t]), 6)}
                for t in top_up_idx
            ],
            "top_downregulated": [
                {"token_id": int(t), "gene": token_to_gene(int(t)),
                 "mean_logit_delta": round(float(mean_logit_delta_vec[t]), 6)}
                for t in top_dn_idx
            ],
        }

    record = {
        "layer": li,
        "feature": fi,
        "switch_score": sf["switch_score"],
        "direction": sf["direction"],
        "hook_layer": hl,
        "alpha_results": results_per_alpha,
    }
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    steering_summary_rows.append(record)

    a5 = results_per_alpha["5"]
    print(f"  L{li}_F{fi:>4d} ({sf['direction']:>8s}): "
          f"α=5 frac_dir={a5['frac_positive_direction']:.2f} "
          f"frac_mat={a5['frac_positive_maturity']:.2f} "
          f"Δ_logit={a5['mean_logit_delta_scalar']:+.4f} "
          f"Δs={a5['mean_delta_s']:+.4f}  "
          f"↑{a5['top_upregulated'][0]['gene']}")

# ---------------------------------------------------------------
# 8. Aggregate summary
# ---------------------------------------------------------------
per_layer = {}
for li in STEERING_LAYERS:
    rows = [r for r in steering_summary_rows if r["layer"] == li]
    if not rows:
        continue
    a5s = [r["alpha_results"]["5"] for r in rows]
    per_layer[f"L{li}"] = {
        "n_features": len(rows),
        "mean_frac_positive_direction": round(
            float(np.mean([x["frac_positive_direction"] for x in a5s])), 4),
        "mean_frac_positive_maturity": round(
            float(np.mean([x["frac_positive_maturity"] for x in a5s])), 4),
        "mean_delta_s": round(float(np.mean([x["mean_delta_s"] for x in a5s])), 6),
    }

with open(OUT3 / "steering_summary.json", "w") as f:
    json.dump({
        "n_switch_features_total": int(len(switch_df)),
        "n_steered_features": len(steering_summary_rows),
        "steering_layers": STEERING_LAYERS,
        "alphas": ALPHAS,
        "n_cells": int(N_CELLS),
        "n_early_cells": int(early_mask.sum()),
        "per_layer_summary_alpha5": per_layer,
        "features": steering_summary_rows,
    }, f, indent=2)

print("\n  Per-layer summary (α=5):")
for k, v in per_layer.items():
    print(f"    {k}: {v}")

print("\nEXPERIMENT 3 RERUN COMPLETE")
