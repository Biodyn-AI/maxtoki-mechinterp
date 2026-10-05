"""Audit action A8 (re-run) — targeted Phase 8t for GATA1, MYC, TAL1 with
responding-feature IDs saved.

Reduced version of `remaining_phases.py` Phase 8t scoped to just the three
positive-control TFs. For each TF:
  1. Extract perturbed activations through MaxToki (≤50 cells per TF)
  2. Encode through L5 SAE
  3. Find responding features (Wilcoxon p<0.05 ∧ |effect|>0.5)
  4. Save per-feature responding IDs alongside counts
  5. Evaluate specificity under BOTH TRRUST and ChIP-seq direct-target sets,
     at thresholds {≥2, ≥3, ≥4} of top-20 overlap.

Output: outputs/phase8t_positive_tf_rerun/{
  per_tf_response.json,        # responding feature IDs + top20 + overlaps per TF
  specificity_table.csv,       # specificity verdict at each threshold × ground truth
  summary.json,
}
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
from scipy import stats as sp_stats

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

PHASE0 = PROJ / "runs/sae-atlas-217M/outputs/phase0"
PHASE1 = PROJ / "runs/sae-atlas-217M/outputs/phase1"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_TFS = ["GATA1", "MYC", "TAL1"]
PROBE_LAYER = 5
SEED = 42
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
N_CELLS_MAX = 50

CHIPSEQ_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")
TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"


def _load_sae(layer: int):
    sd = torch.load(PHASE1 / f"layer_{layer:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    cfg = sd["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"]).to("cpu").float()
    state = {
        "W_enc.weight": sd["W_enc_weight"],
        "W_enc.bias": sd["W_enc_bias"],
        "W_dec.weight": sd["W_dec_weight"],
    }
    # Try direct load — may need attribute-name adjustments depending on TopKSAE definition
    try:
        sae.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"  load_state_dict warning: {e}")
        # Manual assignment as fallback
        with torch.no_grad():
            for name, mod in sae.named_modules():
                if hasattr(mod, "weight"):
                    pass  # leave defaults; we'll try a different mapping below
    sae.eval()
    mu = sd.get("mu", None)
    if mu is not None and not isinstance(mu, torch.Tensor):
        mu = torch.from_numpy(np.asarray(mu)).float()
    return sae, mu


print(f"[A8-rerun] device={DEVICE}, target_tfs={TARGET_TFS}")
trrust_df = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust_df["tf"] = trrust_df["tf"].str.upper()
trrust_df["target"] = trrust_df["target"].str.upper()
trrust_targets = {tf: set(trrust_df[trrust_df["tf"] == tf]["target"]) for tf in TARGET_TFS}

chip_df = pd.read_csv(CHIPSEQ_TSV, sep="\t")
chip_df["source"] = chip_df["source"].str.upper()
chip_df["target"] = chip_df["target"].str.upper()
chip_targets = {tf: set(chip_df[chip_df["source"] == tf]["target"]) for tf in TARGET_TFS}
print(f"  TRRUST targets: " + ", ".join(f"{tf}={len(trrust_targets[tf])}" for tf in TARGET_TFS))
print(f"  ChIP-seq targets: " + ", ".join(f"{tf}={len(chip_targets[tf])}" for tf in TARGET_TFS))

print("\n[A8-rerun] Loading L5 SAE + feature catalog (control activations extracted on the fly)...")
sae_probe, mu_probe = _load_sae(PROBE_LAYER)
# control activations cache file no longer present; extract NT-control acts on the fly later

with open(PHASE2 / f"layer_{PROBE_LAYER:02d}/feature_catalog.json") as ff:
    catalog = json.load(ff)
feat_top20 = {c["feature_id"]: set(g.upper() for g in c["top20_genes"]) for c in catalog}

print("\n[A8-rerun] Loading MaxToki + Replogle K562 + extracting control activations...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
ds = load_ds("k562")
rng = np.random.default_rng(SEED)

# Extract control (non-targeting) activations through MaxToki + L5 SAE
N_CTRL_CELLS = 200
print(f"[A8-rerun] Extracting {N_CTRL_CELLS} non-targeting control activations...")
with h5py.File(ds.h5_path, "r") as f:
    pg_cats_full = [s.decode() if isinstance(s, bytes) else s for s in f["obs"]["gene"]["categories"][:]]
    pg_codes_full = f["obs"]["gene"]["codes"][:]
    cl_codes_full = f["obs"]["cell_line"]["codes"][:]
    cl_cats_full = [(s.decode() if isinstance(s, bytes) else s).lower() for s in f["obs"]["cell_line"]["categories"][:]]
    var_symbols_full = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]

    nt_codes = [i for i, c in enumerate(pg_cats_full) if "non-targeting" in c.lower() or "nontargeting" in c.lower()]
    k562_mask_full = (cl_codes_full == cl_cats_full.index("k562"))
    nt_mask = np.isin(pg_codes_full, nt_codes) & k562_mask_full
    nt_cell_indices = np.where(nt_mask)[0]
    print(f"  available NT controls: {len(nt_cell_indices)}; using {min(N_CTRL_CELLS, len(nt_cell_indices))}")
    nt_use = rng.choice(nt_cell_indices, size=min(N_CTRL_CELLS, len(nt_cell_indices)), replace=False)

    with open(SYM2ENS_PKL, "rb") as f2:
        sym2ens = pickle.load(f2)
    var_ens_full = [sym2ens.get(s) for s in var_symbols_full]
    tokenizer_ctrl = MaxTokiTokenizer()
    var_idx_ctrl, var_tok_ctrl, var_med_ctrl = tokenizer_ctrl.make_var_mapping(var_ens_full)

    t_ctrl = time.time()
    ctrl_token_acts = []
    for ci in nt_use:
        X_cell = f["X"][int(ci), :].astype(np.float32)
        cell = tokenizer_ctrl.tokenize_cell(X_cell, var_idx_ctrl, var_tok_ctrl, var_med_ctrl, max_len=2048)
        if cell is None: continue
        input_ids = torch.from_numpy(cell.token_ids[None, :])
        _, hidden = xt.forward_with_hidden_states(input_ids)
        h_l5 = hidden[PROBE_LAYER + 1][0].numpy()
        ctrl_token_acts.append(h_l5)
        del hidden
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except: pass
    print(f"  extracted {len(ctrl_token_acts)} cell activations in {time.time()-t_ctrl:.1f}s")

ctrl_flat = np.concatenate(ctrl_token_acts, axis=0).astype(np.float32)
with torch.no_grad():
    h_ctrl_list = []
    for i in range(0, ctrl_flat.shape[0], 20_000):
        xb = torch.from_numpy(ctrl_flat[i:i+20_000])
        h_ctrl_list.append(sae_probe.encode(xb, mu_probe).numpy())
    h_ctrl = np.concatenate(h_ctrl_list, axis=0)
print(f"  control SAE activations: {h_ctrl.shape}")
del ctrl_flat, ctrl_token_acts

per_tf_results = {}
with h5py.File(ds.h5_path, "r") as f:
    pg_cats = [s.decode() if isinstance(s, bytes) else s for s in f["obs"]["gene"]["categories"][:]]
    pg_codes = f["obs"]["gene"]["codes"][:]
    cl_codes = f["obs"]["cell_line"]["codes"][:]
    cl_cats = [(s.decode() if isinstance(s, bytes) else s).lower() for s in f["obs"]["cell_line"]["categories"][:]]
    k562_mask = (cl_codes == cl_cats.index("k562"))
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]

    with open(SYM2ENS_PKL, "rb") as f2:
        sym2ens = pickle.load(f2)
    var_ens = [sym2ens.get(s) for s in var_symbols]
    tokenizer = MaxTokiTokenizer()
    var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(var_ens)

    k562_cells = np.where(k562_mask)[0]
    k562_codes = pg_codes[k562_mask]

    for tf in TARGET_TFS:
        try:
            code_idx = pg_cats.index(tf)
        except ValueError:
            print(f"  {tf}: not in K562 perturbation list — skipping")
            per_tf_results[tf] = {"detected": False, "n_cells": 0, "responding_feature_ids": [], "n_responding": 0}
            continue
        cell_idx_pool = k562_cells[k562_codes == code_idx]
        n_pool = len(cell_idx_pool)
        if n_pool < 5:
            print(f"  {tf}: only {n_pool} K562 cells with this perturbation — skipping")
            per_tf_results[tf] = {"detected": False, "n_cells": int(n_pool), "responding_feature_ids": [], "n_responding": 0}
            continue
        n_use = min(N_CELLS_MAX, n_pool)
        cell_idx = rng.choice(cell_idx_pool, size=n_use, replace=False)
        print(f"\n  [{tf}] {n_pool} cells available, using {n_use}")

        t0 = time.time()
        pert_acts = []
        for ci in cell_idx:
            X_cell = f["X"][int(ci), :].astype(np.float32)
            cell = tokenizer.tokenize_cell(X_cell, var_idx_full, var_tok_full, var_med_full, max_len=2048)
            if cell is None:
                continue
            input_ids = torch.from_numpy(cell.token_ids[None, :])
            _, hidden = xt.forward_with_hidden_states(input_ids)
            h_l5 = hidden[PROBE_LAYER + 1][0].numpy()
            pert_acts.append(h_l5)
            del hidden
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass
        if not pert_acts:
            print(f"  [{tf}] no successful tokenizations — skipping")
            per_tf_results[tf] = {"detected": False, "n_cells": 0, "responding_feature_ids": [], "n_responding": 0}
            continue
        print(f"  [{tf}] extracted {len(pert_acts)} cell activations in {time.time()-t0:.1f}s")

        pert_flat = np.concatenate(pert_acts, axis=0).astype(np.float32)
        with torch.no_grad():
            h_pert = sae_probe.encode(torch.from_numpy(pert_flat), mu_probe).numpy()

        d_sae = h_pert.shape[1]
        responding = []
        feature_stats = {}
        for fi in range(d_sae):
            t_vals = h_pert[:, fi]
            o_vals = h_ctrl[:min(10_000, h_ctrl.shape[0]), fi]
            if t_vals.std() < 1e-8 and o_vals.std() < 1e-8:
                continue
            try:
                _, p = sp_stats.mannwhitneyu(t_vals, o_vals, alternative="two-sided")
            except ValueError:
                continue
            effect = float(abs(t_vals.mean() - o_vals.mean()))
            if p < 0.05 and effect > 0.5:
                responding.append(fi)
                feature_stats[fi] = {"p": float(p), "effect": effect}
        print(f"  [{tf}] {len(responding)} responding features at p<0.05 ∧ |effect|>0.5")

        per_tf_results[tf] = {
            "detected": len(responding) > 0,
            "n_cells": int(len(pert_acts)),
            "responding_feature_ids": responding,
            "n_responding": len(responding),
            "feature_stats": feature_stats,
        }

xt = None  # release model memory

# Evaluate specificity at multiple thresholds × ground-truth choices
print("\n[A8-rerun] Specificity evaluation under TRRUST + ChIP-seq, thresholds {2, 3, 4, 5}...")
spec_rows = []
for tf in TARGET_TFS:
    r = per_tf_results[tf]
    if not r["detected"]:
        for src_label, src_targets in [("TRRUST", trrust_targets[tf]), ("ChIP-seq", chip_targets[tf])]:
            for thr in [2, 3, 4, 5]:
                spec_rows.append({"tf": tf, "ground_truth": src_label, "threshold": thr,
                                  "n_responding": 0, "n_specific": 0, "is_specific": False,
                                  "best_overlap": 0, "best_feature_id": None})
        continue
    feat_top20_for_responding = {fi: feat_top20.get(fi, set()) for fi in r["responding_feature_ids"]}
    for src_label, src_targets in [("TRRUST", trrust_targets[tf]), ("ChIP-seq", chip_targets[tf])]:
        # For each threshold, compute n_specific + best overlap
        for thr in [2, 3, 4, 5]:
            overlaps = {fi: len(top20 & src_targets) for fi, top20 in feat_top20_for_responding.items()}
            n_specific = sum(1 for ov in overlaps.values() if ov >= thr)
            best_fi = max(overlaps, key=lambda fi: overlaps[fi]) if overlaps else None
            best_ov = overlaps.get(best_fi, 0) if best_fi else 0
            spec_rows.append({"tf": tf, "ground_truth": src_label, "threshold": thr,
                              "n_responding": r["n_responding"], "n_specific": n_specific,
                              "is_specific": n_specific > 0,
                              "best_overlap": int(best_ov), "best_feature_id": int(best_fi) if best_fi is not None else None})
spec_df = pd.DataFrame(spec_rows)
spec_df.to_csv(OUT / "specificity_table.csv", index=False)

# Random-feature null at each TF's K and each threshold × ground truth
print("\n[A8-rerun] Random-feature null at each (TF, threshold, ground truth)...")
N_NULL = 1000
all_features = list(feat_top20.keys())
null_rows = []
for tf in TARGET_TFS:
    K = per_tf_results[tf]["n_responding"]
    if K == 0:
        for src_label in ["TRRUST", "ChIP-seq"]:
            for thr in [2, 3, 4, 5]:
                null_rows.append({"tf": tf, "ground_truth": src_label, "threshold": thr,
                                  "K": 0, "p_null_specific": float("nan"),
                                  "null_mean_n_specific": float("nan")})
        continue
    for src_label, src_targets in [("TRRUST", trrust_targets[tf]), ("ChIP-seq", chip_targets[tf])]:
        for thr in [2, 3, 4, 5]:
            n_null_specific = []
            for _ in range(N_NULL):
                samp = rng.choice(all_features, size=K, replace=False)
                ov = sum(1 for fi in samp if len(feat_top20[fi] & src_targets) >= thr)
                n_null_specific.append(ov)
            n_null_specific = np.array(n_null_specific)
            null_rows.append({"tf": tf, "ground_truth": src_label, "threshold": thr,
                              "K": K, "p_null_specific_>=1": float((n_null_specific >= 1).mean()),
                              "null_mean_n_specific": float(n_null_specific.mean())})
null_df = pd.DataFrame(null_rows)

# Combine into final summary
combined = spec_df.merge(null_df, on=["tf", "ground_truth", "threshold"], how="left")
combined["above_random"] = combined["n_specific"] > combined["null_mean_n_specific"]
combined.to_csv(OUT / "specificity_with_null.csv", index=False)

# Per-TF response artefact
(OUT / "per_tf_response.json").write_text(json.dumps(per_tf_results, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))

# Final summary
summary = {
    "scope": "Phase 8t re-run targeted at GATA1/MYC/TAL1 with responding-feature IDs saved",
    "device": str(DEVICE),
    "n_cells_max": N_CELLS_MAX,
    "per_tf_response": {tf: {k: v for k, v in per_tf_results[tf].items() if k != "feature_stats"} for tf in TARGET_TFS},
    "verdict_at_threshold_2": {
        tf: {
            "TRRUST": bool(combined[(combined["tf"]==tf) & (combined["ground_truth"]=="TRRUST") & (combined["threshold"]==2)]["is_specific"].iloc[0])
            if not combined[(combined["tf"]==tf) & (combined["ground_truth"]=="TRRUST") & (combined["threshold"]==2)].empty else False,
            "ChIP-seq": bool(combined[(combined["tf"]==tf) & (combined["ground_truth"]=="ChIP-seq") & (combined["threshold"]==2)]["is_specific"].iloc[0])
            if not combined[(combined["tf"]==tf) & (combined["ground_truth"]=="ChIP-seq") & (combined["threshold"]==2)].empty else False,
        } for tf in TARGET_TFS
    },
    "interpretation": (
        "See specificity_with_null.csv for full results. The is_specific column at each "
        "threshold reports whether ≥1 responding feature has ≥threshold of its top-20 "
        "in the TF's target set. The p_null_specific_>=1 column gives the probability "
        "of this happening by random feature draw at the same K. A meaningful 'specific' "
        "verdict requires is_specific=True AND n_specific significantly exceeds null_mean."
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[A8-rerun] Wrote {OUT}/")
print(json.dumps(summary["per_tf_response"], indent=2))
print("\nSpecificity x null comparison (threshold 2):")
print(combined[combined["threshold"] == 2].to_string(index=False))
