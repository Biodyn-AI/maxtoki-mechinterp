"""Experiments 2 + 3 — Combinatorial ablation + Trajectory steering.

Experiment 2: Three-way combinatorial ablation on 4 feature triplets
              across pathways, measuring redundancy deepening.
Experiment 3: Trajectory-guided feature steering on Tabula Sapiens
              immune cells using pseudotime-correlated switch features.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
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
from topk_sae import TopKSAE
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT2 = PROJ / "runs/exhaustive-mapping-217M/outputs/experiment2"
OUT3 = PROJ / "runs/exhaustive-mapping-217M/outputs/experiment3"
OUT2.mkdir(parents=True, exist_ok=True)
OUT3.mkdir(parents=True, exist_ok=True)

TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
D_SAE = 4928

print("=" * 70)
print("EXPERIMENTS 2 + 3")
print("=" * 70)

# Load SAEs at layers 0, 3, 5, 6, 8, 9, 11
NEEDED_LAYERS = [0, 3, 5, 6, 8, 9, 11]
saes = {}; mus = {}
for li in NEEDED_LAYERS:
    ckpt = torch.load(SAE_DIR / f"layer_{li:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval(); saes[li] = sae; mus[li] = ckpt["mu"]
print(f"  SAEs loaded: {list(saes.keys())}")

# Load feature annotations for triplet selection
feat_terms = {}
for li in NEEDED_LAYERS:
    try:
        enr = pd.read_csv(ANN_DIR / f"layer_{li:02d}/significant_enrichments.csv")
        for fid, grp in enr.groupby("feature_id"):
            feat_terms[(li, fid)] = set(grp["term"].tolist())
    except: pass


# ================================================================
# EXPERIMENT 2 — Higher-order combinatorial ablation
# ================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2 — Combinatorial ablation (4 triplets)")
print("=" * 60)

# Select 4 triplets: features from layers L0, L5, L9 with shared pathway
# Find features that share ontology terms across layers
def find_shared_features(layer_a, layer_b, layer_c, n=2):
    """Find feature triplets sharing ontology terms across 3 layers."""
    triplets = []
    for (la, fa), terms_a in feat_terms.items():
        if la != layer_a: continue
        for (lb, fb), terms_b in feat_terms.items():
            if lb != layer_b: continue
            shared_ab = terms_a & terms_b
            if not shared_ab: continue
            for (lc, fc), terms_c in feat_terms.items():
                if lc != layer_c: continue
                shared_abc = shared_ab & terms_c
                if shared_abc:
                    triplets.append((fa, fb, fc, len(shared_abc), list(shared_abc)[:3]))
                    if len(triplets) >= n:
                        return triplets
    return triplets

print("\n  Selecting triplets (L0, L5, L9 with shared ontology)...")
triplets = find_shared_features(0, 5, 9, n=4)
if len(triplets) < 4:
    # Fallback: relax to any features with annotations
    print(f"  only {len(triplets)} shared triplets; padding with top-annotated features")
    for li in [0, 5, 9]:
        top = sorted([(fid, len(t)) for (l, fid), t in feat_terms.items() if l == li],
                     key=lambda x: -x[1])[:2]
        if top:
            for _ in range(4 - len(triplets)):
                triplets.append((top[0][0], top[min(1,len(top)-1)][0], top[0][0], 0, []))
                if len(triplets) >= 4: break

print(f"  selected {len(triplets)} triplets")

# Load cells
ds = load_ds("k562")
rng = np.random.default_rng(SEED)
ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
ctrl_idx = np.sort(rng.choice(ctrl_all, size=100, replace=False))
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((100, ds.n_genes_total), dtype=np.float32)
    for i in range(0, 100, 50):
        X_ctrl[i:i+50] = f["X"][ctrl_idx[i:i+50], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f2:
    sym2ens = pickle.load(f2)
tokenizer = MaxTokiTokenizer()
var_idx, var_tok, var_med = tokenizer.make_var_mapping([sym2ens.get(s) for s in var_symbols])
cells = [tokenizer.tokenize_cell(X_ctrl[ci], var_idx, var_tok, var_med, max_len=2048)
         for ci in range(100)]
cells = [c for c in cells if c is not None]
print(f"  {len(cells)} cells")

xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

# For each triplet: ablate A, B, C, AB, AC, BC, ABC and measure at L11
TARGET_LAYER = 11
triplet_results = []

for ti, (fa, fb, fc, n_shared, shared_terms) in enumerate(triplets[:4]):
    print(f"\n  Triplet {ti}: L0_F{fa} × L5_F{fb} × L9_F{fc} (shared: {n_shared} terms)")
    conditions = {
        "A": [0], "B": [5], "C": [9],
        "AB": [0, 5], "AC": [0, 9], "BC": [5, 9],
        "ABC": [0, 5, 9],
    }
    feature_map = {0: fa, 5: fb, 9: fc}

    # Accumulate per-condition mean effect at target layer
    condition_effects = {cond: np.zeros(D_SAE, dtype=np.float64) for cond in conditions}
    n_cells_used = 0

    for ci, cell in enumerate(cells[:50]):  # 50 cells per triplet
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

        # Clean forward
        with torch.no_grad():
            out_clean = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            clean_hs = {l: out_clean.hidden_states[l][0].cpu() for l in NEEDED_LAYERS}
        z_clean_tgt = saes[TARGET_LAYER].encode(
            torch.from_numpy(clean_hs[TARGET_LAYER].numpy().astype(np.float32)),
            mus[TARGET_LAYER]).detach().numpy()

        for cond_name, ablate_layers in conditions.items():
            # Build hooks for all layers being ablated
            hooks = []
            for al in ablate_layers:
                fi = feature_map[al]
                with torch.no_grad():
                    h_src = clean_hs[al].numpy().astype(np.float32)
                    z_src = saes[al].encode(torch.from_numpy(h_src), mus[al])
                    z_abl = z_src.clone(); z_abl[:, fi] = 0.0
                    h_recon = saes[al].decode(z_src, mus[al])
                    h_abl_recon = saes[al].decode(z_abl, mus[al])
                    delta = h_abl_recon - h_recon
                    h_patched = (clean_hs[al] + delta.cpu()).to(DEVICE).unsqueeze(0)

                def _hook(module, input, output, _p=h_patched):
                    return _p.to(output.device)
                hooks.append(xt.model.model.layers[al].register_forward_hook(_hook))

            with torch.no_grad():
                out_abl = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            for h in hooks:
                h.remove()

            z_abl_tgt = saes[TARGET_LAYER].encode(
                torch.from_numpy(out_abl.hidden_states[TARGET_LAYER][0].cpu().numpy().astype(np.float32)),
                mus[TARGET_LAYER]).detach().numpy()
            delta_z = (z_abl_tgt - z_clean_tgt).mean(axis=0)
            condition_effects[cond_name] += delta_z
            del out_abl

        del out_clean, clean_hs
        n_cells_used += 1
        if DEVICE == "mps" and ci % 10 == 0:
            try: torch.mps.empty_cache()
            except: pass

    # Average effects
    for cond in condition_effects:
        condition_effects[cond] /= max(n_cells_used, 1)

    # Compute redundancy ratios
    # Single effects
    eff_A = np.abs(condition_effects["A"]).mean()
    eff_B = np.abs(condition_effects["B"]).mean()
    eff_C = np.abs(condition_effects["C"]).mean()
    # Pairwise
    eff_AB = np.abs(condition_effects["AB"]).mean()
    eff_AC = np.abs(condition_effects["AC"]).mean()
    eff_BC = np.abs(condition_effects["BC"]).mean()
    # Three-way
    eff_ABC = np.abs(condition_effects["ABC"]).mean()

    pairwise_ratio = eff_AB / max(eff_A + eff_B, 1e-12)
    threeway_ratio = eff_ABC / max(eff_A + eff_B + eff_C, 1e-12)
    marginal_C_given_AB = (eff_ABC - eff_AB) / max(eff_C, 1e-12) if eff_C > 1e-8 else 0

    # Synergy: count targets where ABC > A + B + C (superadditive)
    abc_effect = np.abs(condition_effects["ABC"])
    sum_singles = np.abs(condition_effects["A"]) + np.abs(condition_effects["B"]) + np.abs(condition_effects["C"])
    n_superadditive = int((abc_effect > sum_singles * 1.1).sum())  # 10% margin
    n_targets_with_effect = int((abc_effect > 0.01).sum())

    result = {
        "triplet": ti, "features": {"L0": fa, "L5": fb, "L9": fc},
        "shared_terms": n_shared,
        "pairwise_ratio": round(pairwise_ratio, 4),
        "threeway_ratio": round(threeway_ratio, 4),
        "marginal_C_given_AB": round(marginal_C_given_AB, 4),
        "n_superadditive": n_superadditive,
        "n_targets_with_effect": n_targets_with_effect,
        "pct_superadditive": round(n_superadditive / max(n_targets_with_effect, 1), 4),
    }
    triplet_results.append(result)
    print(f"    pairwise_ratio={pairwise_ratio:.3f}  threeway_ratio={threeway_ratio:.3f}  "
          f"marginal_C|AB={marginal_C_given_AB:.3f}  synergy={n_superadditive}/{n_targets_with_effect}")

# Summary
mean_pw = np.mean([r["pairwise_ratio"] for r in triplet_results])
mean_tw = np.mean([r["threeway_ratio"] for r in triplet_results])
mean_syn = np.mean([r["pct_superadditive"] for r in triplet_results])
print(f"\n  SUMMARY: pairwise={mean_pw:.3f}  threeway={mean_tw:.3f}  synergy={mean_syn:.2%}")
print(f"  Paper: pairwise=0.74  threeway=0.59  synergy=0.14%")

with open(OUT2 / "combinatorial_summary.json", "w") as f:
    json.dump({
        "n_triplets": len(triplet_results),
        "mean_pairwise_ratio": round(mean_pw, 4),
        "mean_threeway_ratio": round(mean_tw, 4),
        "mean_pct_superadditive": round(mean_syn, 4),
        "triplets": triplet_results,
    }, f, indent=2)
print("  EXPERIMENT 2 COMPLETE")

del xt
if DEVICE == "mps":
    try: torch.mps.empty_cache()
    except: pass


# ================================================================
# EXPERIMENT 3 — Trajectory-guided feature steering
# ================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3 — Trajectory steering")
print("=" * 60)

# Step 1: Load Tabula Sapiens immune cells + compute pseudotime
print("\n  Loading Tabula Sapiens immune cells...")
adata = ad.read_h5ad(str(TS_H5), backed="r")
rng_ts = np.random.default_rng(SEED + 3333)
ts_idx = np.sort(rng_ts.choice(adata.n_obs, size=200, replace=False))
cell_types = adata.obs["cell_type"].iloc[ts_idx].astype(str).to_numpy()
X_ts = adata[ts_idx].X
if sp.issparse(X_ts) or hasattr(X_ts, "toarray"):
    X_ts = X_ts.toarray()
X_ts = np.asarray(X_ts, dtype=np.float32)
var_ens_ts = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
adata.file.close()

# Simple pseudotime proxy: use PC1 of log-normalized expression
rs = X_ts.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_ts / rs * 1e4)
from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=SEED).fit_transform(X_log)
pseudotime = pca.flatten()
# Normalize to [0, 1]
pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-12)
print(f"  200 cells, pseudotime range [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")

# Step 2: Identify switch features — features whose activation correlates with pseudotime
print("\n  Identifying switch features...")
var_idx_ts, var_tok_ts, var_med_ts = tokenizer.make_var_mapping(var_ens_ts.tolist())

xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

# Extract hidden states and encode through SAEs
STEERING_LAYERS = [0, 3, 6, 9, 11]
switch_candidates = []

for li in STEERING_LAYERS:
    feat_activations = np.zeros((200, D_SAE), dtype=np.float32)
    for ci in range(200):
        cell = tokenizer.tokenize_cell(X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048)
        if cell is None:
            continue
        input_ids = torch.from_numpy(cell.token_ids[None, :])
        _, hidden = xt.forward_with_hidden_states(input_ids)
        h = hidden[li][0].numpy().astype(np.float32)
        with torch.no_grad():
            z = saes[li].encode(torch.from_numpy(h), mus[li]).detach().numpy()
        feat_activations[ci] = z.mean(axis=0)  # mean over positions
        del hidden
        if DEVICE == "mps" and ci % 20 == 0:
            try: torch.mps.empty_cache()
            except: pass

    # Correlate each feature with pseudotime
    from scipy.stats import spearmanr
    for fi in range(D_SAE):
        act = feat_activations[:, fi]
        if act.std() < 1e-8:
            continue
        rho, p = spearmanr(act, pseudotime)
        if abs(rho) > 0.3 and p < 0.01:
            switch_candidates.append({
                "layer": li, "feature": fi,
                "switch_score": round(abs(rho), 4),
                "direction": "positive" if rho > 0 else "negative",
                "rho": round(rho, 4), "p": round(p, 6),
            })
    n_switches = sum(1 for s in switch_candidates if s["layer"] == li)
    print(f"  L{li}: {n_switches} switch features (|ρ| > 0.3, p < 0.01)")

switch_df = pd.DataFrame(switch_candidates).sort_values("switch_score", ascending=False)
switch_df.to_csv(OUT3 / "switch_features.csv", index=False)
print(f"  total switch features: {len(switch_df)}")

# Step 3: Steer top switch features
# Select top 3 per layer (max 15 total)
steer_features = []
for li in STEERING_LAYERS:
    layer_switches = switch_df[switch_df["layer"] == li].head(3)
    for _, r in layer_switches.iterrows():
        steer_features.append({"layer": int(r["layer"]), "feature": int(r["feature"]),
                              "switch_score": r["switch_score"], "direction": r["direction"]})
print(f"\n  steering {len(steer_features)} features...")

# Compute early/late pseudotime signatures (mean logit difference)
early_mask = pseudotime < np.percentile(pseudotime, 25)
late_mask = pseudotime > np.percentile(pseudotime, 75)

# For each switch feature: amplify activation (α=2, α=5) in early-pseudotime cells
# and measure whether the output shifts toward late-pseudotime signature
ALPHAS = [2, 5]
steering_results = []

for sf in steer_features:
    li = sf["layer"]; fi = sf["feature"]
    results_per_alpha = {}

    for alpha in ALPHAS:
        n_toward_mature = 0
        n_steered = 0
        deltas = []

        for ci in np.where(early_mask)[0]:
            cell = tokenizer.tokenize_cell(X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048)
            if cell is None:
                continue
            input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

            # Clean forward
            with torch.no_grad():
                out_clean = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
                clean_logits = out_clean.logits[0].cpu().numpy()
                h_li = out_clean.hidden_states[li][0].cpu()

            # Amplify the switch feature
            with torch.no_grad():
                h_np = h_li.numpy().astype(np.float32)
                z = saes[li].encode(torch.from_numpy(h_np), mus[li])
                # Amplify: multiply the feature activation by alpha
                z_steered = z.clone()
                z_steered[:, fi] = z[:, fi] * alpha
                h_recon = saes[li].decode(z, mus[li])
                h_steered_recon = saes[li].decode(z_steered, mus[li])
                delta = h_steered_recon - h_recon
                h_patched = (h_li + delta.cpu()).to(DEVICE).unsqueeze(0)

            def _hook(module, input, output, _p=h_patched):
                return _p.to(output.device)
            hook = xt.model.model.layers[li].register_forward_hook(_hook)
            with torch.no_grad():
                out_steered = xt.model(input_ids, output_hidden_states=False, use_cache=False, return_dict=True)
                steered_logits = out_steered.logits[0].cpu().numpy()
            hook.remove()

            # Measure shift: does the logit delta point toward late-pseudotime cells?
            logit_delta = (steered_logits - clean_logits).mean(axis=0)  # mean over positions
            deltas.append(float(np.mean(logit_delta)))
            # Simple proxy: positive delta = toward maturity (higher pseudotime)
            if sf["direction"] == "positive":
                toward = float(np.mean(logit_delta)) > 0
            else:
                toward = float(np.mean(logit_delta)) < 0
            if toward:
                n_toward_mature += 1
            n_steered += 1

            del out_clean, out_steered
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass

        frac_pos = n_toward_mature / max(n_steered, 1)
        mean_delta = float(np.mean(deltas)) if deltas else 0
        results_per_alpha[alpha] = {
            "frac_positive": round(frac_pos, 4),
            "n_steered": n_steered,
            "mean_delta": round(mean_delta, 6),
        }

    steering_results.append({
        **sf, "alpha_results": results_per_alpha,
    })
    a5 = results_per_alpha.get(5, {})
    print(f"  L{li}_F{fi} ({sf['direction']}): α=5 frac_pos={a5.get('frac_positive', 0):.2f}  "
          f"Δ={a5.get('mean_delta', 0):.4f}")

# Summary by layer
print(f"\n  Steering summary by layer (α=5):")
for li in STEERING_LAYERS:
    layer_results = [r for r in steering_results if r["layer"] == li]
    if not layer_results:
        continue
    fracs = [r["alpha_results"][5]["frac_positive"] for r in layer_results if 5 in r["alpha_results"]]
    if fracs:
        print(f"    L{li}: mean frac_positive={np.mean(fracs):.2f} (n={len(fracs)} features)")

with open(OUT3 / "steering_summary.json", "w") as f:
    json.dump({
        "n_switch_features": len(switch_df),
        "n_steered_features": len(steering_results),
        "steering_layers": STEERING_LAYERS,
        "alphas": ALPHAS,
        "results": steering_results,
    }, f, indent=2)

del xt
if DEVICE == "mps":
    try: torch.mps.empty_cache()
    except: pass

print(f"\nEXPERIMENTS 2 + 3 COMPLETE")
