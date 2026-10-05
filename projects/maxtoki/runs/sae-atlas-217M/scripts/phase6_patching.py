"""Phase 6 — Causal feature patching at L5.

For each of 50 richly-annotated SAE features, zero the feature at L5 via
a forward hook, re-run the model, and measure target-gene vs off-target
logit changes. Reports specificity ratio per feature.

Key fix: Llama decoder layers return a plain Tensor (not a tuple), so
the hook returns the replacement tensor directly.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

PHASE1 = PROJ / "runs/sae-atlas-217M/outputs/phase1"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/sae-atlas-217M/outputs/phase6"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_LAYER = 5
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
N_CELLS = 50
N_FEATURES = 50

print("=" * 70)
print(f"Phase 6 — Causal feature patching at L{PROBE_LAYER}")
print("=" * 70)

# Load SAE
ckpt = torch.load(PHASE1 / f"layer_{PROBE_LAYER:02d}/sae_final.pt",
                  map_location="cpu", weights_only=False)
sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
sae.W_enc.weight.data = ckpt["W_enc_weight"]
sae.W_enc.bias.data = ckpt["W_enc_bias"]
sae.W_dec.weight.data = ckpt["W_dec_weight"]
sae.eval()
mu = ckpt["mu"]

# Select 50 richly annotated features
with open(PHASE2 / f"layer_{PROBE_LAYER:02d}/feature_catalog.json") as f:
    catalog = json.load(f)
try:
    enrich_df = pd.read_csv(PHASE2 / f"layer_{PROBE_LAYER:02d}/significant_enrichments.csv")
    feat_enrich_count = enrich_df.groupby("feature_id").size().to_dict()
except FileNotFoundError:
    feat_enrich_count = {}
rich_features = sorted(
    [fc for fc in catalog if feat_enrich_count.get(fc["feature_id"], 0) >= 3],
    key=lambda x: -feat_enrich_count.get(x["feature_id"], 0),
)[:N_FEATURES]
feat_top20 = {fc["feature_id"]: set(fc["top20_genes"]) for fc in catalog}
print(f"  richly annotated features: {len(rich_features)}")

# Load MaxToki
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

# Load K562 control cells
ds = load_ds("k562")
rng = np.random.default_rng(SEED)
ctrl_all = np.where(
    ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes))
)[0]
ctrl_sel = np.sort(rng.choice(ctrl_all, size=N_CELLS, replace=False))
with h5py.File(ds.h5_path, "r") as f:
    X_cells = np.empty((N_CELLS, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CELLS, 50):
        X_cells[i:i+50] = f["X"][ctrl_sel[i:i+50], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s
                   for s in f["var"]["gene_name_index"][:]]

with open(SYM2ENS_PKL, "rb") as f2:
    sym2ens = pickle.load(f2)
var_ens = [sym2ens.get(s) for s in var_symbols]
tokenizer = MaxTokiTokenizer()
var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(var_ens)

# Token ID → gene symbol reverse map
token_to_gene = {}
for vi, ti in zip(var_idx_full, var_tok_full):
    token_to_gene[int(ti)] = var_symbols[vi].upper()

print(f"  cells: {N_CELLS}  device: {DEVICE}")
print(f"\n  Patching {len(rich_features)} features × {N_CELLS} cells...")

results = []
for fi_idx, fc in enumerate(rich_features):
    fi = fc["feature_id"]
    target_genes = feat_top20.get(fi, set())
    if len(target_genes) < 3:
        continue

    delta_targets = []
    delta_others = []

    for ci in range(N_CELLS):
        cell = tokenizer.tokenize_cell(
            X_cells[ci], var_idx_full, var_tok_full, var_med_full, max_len=2048
        )
        if cell is None:
            continue
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)
        seq_len = input_ids.shape[1]

        # 1. Clean forward
        with torch.no_grad():
            out_clean = xt.model(input_ids, output_hidden_states=True,
                               use_cache=False, return_dict=True)
            clean_logits = out_clean.logits[0].cpu()
            # Hidden state AFTER layer PROBE_LAYER (= index PROBE_LAYER+1 in hidden_states)
            h_probe = out_clean.hidden_states[PROBE_LAYER + 1][0].cpu()

        # 2. SAE encode → zero feature → decode
        with torch.no_grad():
            h_np = h_probe.numpy().astype(np.float32)
            h_enc = sae.encode(torch.from_numpy(h_np), mu)
            h_enc[:, fi] = 0.0
            h_patched = sae.decode(h_enc, mu).to(DEVICE).unsqueeze(0)

        # 3. Patched forward via hook on layer PROBE_LAYER
        # The hook replaces the layer's output with our patched hidden state
        def _patch_hook(module, input, output, _p=h_patched):
            return _p.to(output.device)

        hook = xt.model.model.layers[PROBE_LAYER].register_forward_hook(_patch_hook)
        with torch.no_grad():
            out_patched = xt.model(input_ids, output_hidden_states=False,
                                  use_cache=False, return_dict=True)
            patched_logits = out_patched.logits[0].cpu()
        hook.remove()

        # 4. Per-position absolute logit change
        d_logits = (patched_logits - clean_logits).abs().mean(dim=1).numpy()  # (seq,)

        # Map positions to target/other based on gene symbol
        target_mask = np.zeros(seq_len, dtype=bool)
        for pos in range(seq_len):
            tok_id = int(cell.token_ids[pos])
            gene = token_to_gene.get(tok_id, "")
            if gene in target_genes:
                target_mask[pos] = True

        if target_mask.sum() > 0:
            delta_targets.append(float(d_logits[target_mask].mean()))
        if (~target_mask).sum() > 0:
            delta_others.append(float(d_logits[~target_mask].mean()))

    if delta_targets and delta_others:
        mt = float(np.mean(delta_targets))
        mo = float(np.mean(delta_others))
        spec = mt / max(mo, 1e-12)
        results.append({
            "feature_id": fi,
            "mean_target_delta": mt,
            "mean_other_delta": mo,
            "specificity_ratio": spec,
        })

    if (fi_idx + 1) % 10 == 0:
        print(f"    {fi_idx + 1}/{len(rich_features)} features done")

if results:
    df = pd.DataFrame(results)
    df.to_csv(OUT / "causal_patching.csv", index=False)
    med = float(df["specificity_ratio"].median())
    pct2 = float((df["specificity_ratio"] > 2).mean())
    pct10 = float((df["specificity_ratio"] > 10).mean())
    print(f"\n  median specificity: {med:.2f}×")
    print(f"  >2×: {pct2:.1%}  >10×: {pct10:.1%}")
    print(f"  mean target Δ: {df['mean_target_delta'].mean():.4f}")
    print(f"  mean other Δ: {df['mean_other_delta'].mean():.4f}")
    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "n_features": len(results),
            "median_specificity": med,
            "pct_above_2x": pct2,
            "pct_above_10x": pct10,
            "mean_target_delta": float(df["mean_target_delta"].mean()),
            "mean_other_delta": float(df["mean_other_delta"].mean()),
        }, f, indent=2)
else:
    print("  no results")

print(f"\nPhase 6 COMPLETE — outputs: {OUT}")
