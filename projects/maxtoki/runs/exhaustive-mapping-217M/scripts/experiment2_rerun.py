"""Experiment 2 rerun — diversified triplet selection.

The original `experiments_2_3.py` used a greedy selector that locked onto the
first (L0_fa, L5_fb) pair that shared any ontology term, then iterated the L9
feature — producing 4 triplets that all shared the same (L0_F0, L5_F0) base.
Result: near-identical pairwise/threeway ratios across the 4 triplets (not 4
independent samples).

This script picks 4 triplets with:
  * pairwise-distinct L0 features
  * pairwise-distinct L5 features
  * pairwise-distinct L9 features
  * pairwise-distinct primary ontology term (one triplet per pathway)

Writes to runs/exhaustive-mapping-217M/outputs/experiment2_v2/.
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

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402
from dataset_loader import resolve as load_ds, SYM2ENS_PKL  # noqa: E402

SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT2V2 = PROJ / "runs/exhaustive-mapping-217M/outputs/experiment2_v2"
OUT2V2.mkdir(parents=True, exist_ok=True)

DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
D_SAE = 4928
TRIPLET_LAYERS = (0, 5, 9)
TARGET_LAYER = 11
NEEDED_LAYERS = list(TRIPLET_LAYERS) + [TARGET_LAYER]
N_TRIPLETS = 4
N_CELLS = 50
SEL_SEED = 137  # distinct from SEED so cell pool matches the v1 run

print("=" * 70)
print("EXPERIMENT 2 RERUN — diversified triplet selection")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Load SAEs
# ---------------------------------------------------------------
saes: dict[int, TopKSAE] = {}
mus: dict[int, torch.Tensor] = {}
for li in NEEDED_LAYERS:
    ckpt = torch.load(SAE_DIR / f"layer_{li:02d}/sae_final.pt",
                      map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval()
    saes[li] = sae
    mus[li] = ckpt["mu"]
print(f"  SAEs loaded for layers {NEEDED_LAYERS}")

# ---------------------------------------------------------------
# 2. Load feature → ontology term mapping
# ---------------------------------------------------------------
feat_terms: dict[tuple[int, int], set[str]] = {}
for li in TRIPLET_LAYERS:
    enr_path = ANN_DIR / f"layer_{li:02d}/significant_enrichments.csv"
    if not enr_path.exists():
        continue
    enr = pd.read_csv(enr_path)
    for fid, grp in enr.groupby("feature_id"):
        feat_terms[(li, int(fid))] = set(grp["term"].tolist())
print(f"  annotated features: L0={sum(1 for k in feat_terms if k[0]==0)}, "
      f"L5={sum(1 for k in feat_terms if k[0]==5)}, "
      f"L9={sum(1 for k in feat_terms if k[0]==9)}")

# ---------------------------------------------------------------
# 3. Diversified triplet selection
#    Strategy: enumerate all (fa, fb, fc) with ≥1 shared term across all 3,
#    group by primary shared term, then greedy-pick 4 triplets such that
#    no feature and no pathway is reused.
# ---------------------------------------------------------------
la, lb, lc = TRIPLET_LAYERS
a_feats = {fa: t for (lx, fa), t in feat_terms.items() if lx == la}
b_feats = {fb: t for (lx, fb), t in feat_terms.items() if lx == lb}
c_feats = {fc: t for (lx, fc), t in feat_terms.items() if lx == lc}

candidates: list[dict] = []
for fa, ta in a_feats.items():
    for fb, tb in b_feats.items():
        shared_ab = ta & tb
        if not shared_ab:
            continue
        for fc, tc in c_feats.items():
            shared_abc = shared_ab & tc
            if shared_abc:
                # primary term: pick deterministically (alphabetical)
                primary = sorted(shared_abc)[0]
                candidates.append({
                    "fa": fa, "fb": fb, "fc": fc,
                    "primary_term": primary,
                    "shared_terms": sorted(shared_abc),
                    "n_shared": len(shared_abc),
                })
print(f"  {len(candidates)} candidate triplets")

# Sort by pathway-richness (more shared terms first — suggests tighter biological coherence)
candidates.sort(key=lambda d: (-d["n_shared"], d["primary_term"]))

# Group by primary term
by_pathway: dict[str, list[dict]] = {}
for c in candidates:
    by_pathway.setdefault(c["primary_term"], []).append(c)
print(f"  {len(by_pathway)} distinct primary pathways available")

# Greedy pick: for each distinct pathway (in popularity order), take the top
# candidate whose fa / fb / fc are not yet used
selected: list[dict] = []
used_a: set[int] = set(); used_b: set[int] = set(); used_c: set[int] = set()
pathways_by_size = sorted(by_pathway.keys(),
                          key=lambda p: (-len(by_pathway[p]), p))
for pw in pathways_by_size:
    if len(selected) >= N_TRIPLETS:
        break
    for cand in by_pathway[pw]:
        if cand["fa"] in used_a or cand["fb"] in used_b or cand["fc"] in used_c:
            continue
        selected.append(cand)
        used_a.add(cand["fa"]); used_b.add(cand["fb"]); used_c.add(cand["fc"])
        break

print(f"\n  selected {len(selected)} triplets (distinct features + distinct pathways):")
for i, s in enumerate(selected):
    print(f"    T{i}: L0_F{s['fa']} × L5_F{s['fb']} × L9_F{s['fc']}  "
          f"pathway='{s['primary_term']}'  (#shared={s['n_shared']})")

if len(selected) < N_TRIPLETS:
    print(f"  WARNING: could only diversify to {len(selected)}/{N_TRIPLETS} triplets; "
          f"filling remainder with pathway-distinct but possibly feature-reusing triplets")
    for pw in pathways_by_size:
        if len(selected) >= N_TRIPLETS:
            break
        if pw in {s["primary_term"] for s in selected}:
            continue
        selected.append(by_pathway[pw][0])

# ---------------------------------------------------------------
# 4. Load cells + tokenizer (identical to v1 — same seed, same pool)
# ---------------------------------------------------------------
ds = load_ds("k562")
rng = np.random.default_rng(SEED)
ctrl_all = np.where(ds.cell_of_interest_mask &
                    np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
ctrl_idx = np.sort(rng.choice(ctrl_all, size=100, replace=False))
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((100, ds.n_genes_total), dtype=np.float32)
    for i in range(0, 100, 50):
        X_ctrl[i:i+50] = f["X"][ctrl_idx[i:i+50], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s
                   for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f2:
    sym2ens = pickle.load(f2)
tokenizer = MaxTokiTokenizer()
var_idx, var_tok, var_med = tokenizer.make_var_mapping([sym2ens.get(s) for s in var_symbols])
cells = [tokenizer.tokenize_cell(X_ctrl[ci], var_idx, var_tok, var_med, max_len=2048)
         for ci in range(100)]
cells = [c for c in cells if c is not None][:N_CELLS]
print(f"\n  tokenized {len(cells)} cells for ablation")

xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

# ---------------------------------------------------------------
# 5. Triplet ablation loop (same mechanics as v1)
# ---------------------------------------------------------------
triplet_results = []
t_overall = time.time()

for ti, tr in enumerate(selected[:N_TRIPLETS]):
    fa, fb, fc = tr["fa"], tr["fb"], tr["fc"]
    print(f"\n  Triplet {ti}: L0_F{fa} × L5_F{fb} × L9_F{fc}  pathway='{tr['primary_term']}'")
    conditions = {
        "A": [0], "B": [5], "C": [9],
        "AB": [0, 5], "AC": [0, 9], "BC": [5, 9],
        "ABC": [0, 5, 9],
    }
    feature_map = {0: fa, 5: fb, 9: fc}
    condition_effects = {cond: np.zeros(D_SAE, dtype=np.float64) for cond in conditions}
    n_cells_used = 0

    for ci, cell in enumerate(cells):
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

        with torch.no_grad():
            out_clean = xt.model(input_ids, output_hidden_states=True,
                                  use_cache=False, return_dict=True)
            clean_hs = {l: out_clean.hidden_states[l][0].cpu() for l in NEEDED_LAYERS}
        z_clean_tgt = saes[TARGET_LAYER].encode(
            torch.from_numpy(clean_hs[TARGET_LAYER].numpy().astype(np.float32)),
            mus[TARGET_LAYER]).detach().numpy()

        for cond_name, ablate_layers in conditions.items():
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
                    if isinstance(output, tuple):
                        return (_p.to(output[0].device),) + output[1:]
                    return _p.to(output.device)
                hooks.append(xt.model.model.layers[al].register_forward_hook(_hook))

            with torch.no_grad():
                out_abl = xt.model(input_ids, output_hidden_states=True,
                                    use_cache=False, return_dict=True)
            for h in hooks:
                h.remove()

            z_abl_tgt = saes[TARGET_LAYER].encode(
                torch.from_numpy(out_abl.hidden_states[TARGET_LAYER][0]
                                  .cpu().numpy().astype(np.float32)),
                mus[TARGET_LAYER]).detach().numpy()
            delta_z = (z_abl_tgt - z_clean_tgt).mean(axis=0)
            condition_effects[cond_name] += delta_z
            del out_abl

        del out_clean, clean_hs
        n_cells_used += 1
        if DEVICE == "mps" and ci % 10 == 0:
            try: torch.mps.empty_cache()
            except Exception: pass

    for cond in condition_effects:
        condition_effects[cond] /= max(n_cells_used, 1)

    eff_A = np.abs(condition_effects["A"]).mean()
    eff_B = np.abs(condition_effects["B"]).mean()
    eff_C = np.abs(condition_effects["C"]).mean()
    eff_AB = np.abs(condition_effects["AB"]).mean()
    eff_AC = np.abs(condition_effects["AC"]).mean()
    eff_BC = np.abs(condition_effects["BC"]).mean()
    eff_ABC = np.abs(condition_effects["ABC"]).mean()

    pairwise_ratios = [
        eff_AB / max(eff_A + eff_B, 1e-12),
        eff_AC / max(eff_A + eff_C, 1e-12),
        eff_BC / max(eff_B + eff_C, 1e-12),
    ]
    pairwise_ratio_mean = float(np.mean(pairwise_ratios))
    threeway_ratio = float(eff_ABC / max(eff_A + eff_B + eff_C, 1e-12))
    marginal_C_given_AB = float(
        (eff_ABC - eff_AB) / max(eff_C, 1e-12)) if eff_C > 1e-8 else 0.0

    abc_effect = np.abs(condition_effects["ABC"])
    sum_singles = (np.abs(condition_effects["A"])
                   + np.abs(condition_effects["B"])
                   + np.abs(condition_effects["C"]))
    n_superadditive = int((abc_effect > sum_singles * 1.1).sum())
    n_targets_with_effect = int((abc_effect > 0.01).sum())

    result = {
        "triplet": ti,
        "features": {"L0": int(fa), "L5": int(fb), "L9": int(fc)},
        "primary_pathway": tr["primary_term"],
        "shared_terms_count": int(tr["n_shared"]),
        "pairwise_ratio_AB": round(pairwise_ratios[0], 4),
        "pairwise_ratio_AC": round(pairwise_ratios[1], 4),
        "pairwise_ratio_BC": round(pairwise_ratios[2], 4),
        "pairwise_ratio_mean": round(pairwise_ratio_mean, 4),
        "threeway_ratio": round(threeway_ratio, 4),
        "marginal_C_given_AB": round(marginal_C_given_AB, 4),
        "n_superadditive": n_superadditive,
        "n_targets_with_effect": n_targets_with_effect,
        "pct_superadditive": round(
            n_superadditive / max(n_targets_with_effect, 1), 4),
    }
    triplet_results.append(result)
    print(f"    pairwise[AB/AC/BC]={pairwise_ratios[0]:.3f}/{pairwise_ratios[1]:.3f}/"
          f"{pairwise_ratios[2]:.3f}  threeway={threeway_ratio:.3f}  "
          f"marg_C|AB={marginal_C_given_AB:.3f}  "
          f"synergy={n_superadditive}/{n_targets_with_effect}")

# ---------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------
mean_pw = float(np.mean([r["pairwise_ratio_mean"] for r in triplet_results]))
mean_tw = float(np.mean([r["threeway_ratio"] for r in triplet_results]))
mean_syn = float(np.mean([r["pct_superadditive"] for r in triplet_results]))
pw_std = float(np.std([r["pairwise_ratio_mean"] for r in triplet_results]))
tw_std = float(np.std([r["threeway_ratio"] for r in triplet_results]))

summary = {
    "n_triplets": len(triplet_results),
    "selection_strategy": "pathway-distinct + feature-distinct (one triplet per primary ontology term, no L0/L5/L9 feature reused)",
    "triplet_layers": list(TRIPLET_LAYERS),
    "measurement_layer": TARGET_LAYER,
    "n_cells_per_condition": N_CELLS,
    "mean_pairwise_ratio": round(mean_pw, 4),
    "std_pairwise_ratio": round(pw_std, 4),
    "mean_threeway_ratio": round(mean_tw, 4),
    "std_threeway_ratio": round(tw_std, 4),
    "mean_pct_superadditive": round(mean_syn, 6),
    "v1_comparison": {
        "v1_mean_pairwise_ratio": 0.1654,
        "v1_mean_threeway_ratio": 0.1897,
        "v1_issue": "all 4 v1 triplets shared (L0_F0, L5_F0) base",
    },
    "triplets": triplet_results,
}
out_json = OUT2V2 / "combinatorial_summary.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  SUMMARY (diversified): pairwise={mean_pw:.3f} ±{pw_std:.3f}  "
      f"threeway={mean_tw:.3f} ±{tw_std:.3f}  synergy={mean_syn:.2%}")
print(f"  v1 reference:         pairwise=0.165 (no spread — identical triplets)  "
      f"threeway=0.190  synergy=0.00%")
print(f"  Paper (Geneformer):   pairwise=0.74   threeway=0.59   synergy=0.14%")
print(f"  EXPERIMENT 2 RERUN COMPLETE — wrote {out_json}")
print(f"  wall-clock: {(time.time() - t_overall)/60:.1f} min")
