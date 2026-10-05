"""Experiment 1 — Exhaustive circuit tracing at L5.

Trace 1000 active features (top-500 annotated + 500 random unannotated)
at L5, measuring effects at downstream layers L6, L8, L11 using 20 cells.
Resume-safe: skips features whose JSON already exists.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pickle
import h5py

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/exhaustive-mapping-217M/outputs/experiment1"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
SOURCE_LAYER = 5
DOWNSTREAM_LAYERS = [6, 8, 11]
N_CELLS = 20
N_FEATURES = int(os.environ.get("N_FEATURES", "1000"))
D_SAE = 4928
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7

print("=" * 70)
print(f"EXPERIMENT 1 — Exhaustive tracing at L{SOURCE_LAYER}")
print(f"  N_features={N_FEATURES}  N_cells={N_CELLS}  downstream={DOWNSTREAM_LAYERS}")
print("=" * 70)


class WelfordAccumulator:
    __slots__ = ("n", "mean", "M2", "pos_count")
    def __init__(self, size):
        self.n = 0; self.mean = np.zeros(size, np.float64)
        self.M2 = np.zeros(size, np.float64); self.pos_count = np.zeros(size, np.int64)
    def update(self, x):
        self.n += 1; d = x - self.mean; self.mean += d / self.n
        self.M2 += d * (x - self.mean); self.pos_count += (x > 0).astype(np.int64)
    def finalize(self):
        if self.n < 2: return self.mean, np.ones_like(self.mean), self.mean
        std = np.sqrt(self.M2 / (self.n - 1) + 1e-12)
        return self.mean / std, np.maximum(self.pos_count, self.n - self.pos_count) / self.n, self.mean


# Phase 0: Load SAEs
print("\n[Phase 0] Loading SAEs...")
saes = {}; mus = {}
for li in [SOURCE_LAYER] + DOWNSTREAM_LAYERS:
    ckpt = torch.load(SAE_DIR / f"layer_{li:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval(); saes[li] = sae; mus[li] = ckpt["mu"]

# Phase 1: Select active features
print("\n[Phase 1] Selecting features...")
with open(ANN_DIR / f"layer_{SOURCE_LAYER:02d}/feature_catalog.json") as f:
    catalog = json.load(f)
# All features with freq > 0.001
active = [fc for fc in catalog if fc["activation_frequency"] >= 0.001]
print(f"  active features (freq >= 0.001): {len(active)}")

# Top-500 by annotation count + 500 random unannotated
try:
    enrich_df = pd.read_csv(ANN_DIR / f"layer_{SOURCE_LAYER:02d}/significant_enrichments.csv")
    feat_n = enrich_df.groupby("feature_id").size().to_dict()
except: feat_n = {}

annotated = sorted([fc for fc in active if feat_n.get(fc["feature_id"], 0) > 0],
                   key=lambda x: -feat_n.get(x["feature_id"], 0))[:500]
unannotated = [fc for fc in active if feat_n.get(fc["feature_id"], 0) == 0]
rng = np.random.default_rng(SEED)
unannotated_sample = list(rng.choice(unannotated, size=min(N_FEATURES - len(annotated), len(unannotated)), replace=False))
selected = annotated + unannotated_sample
feature_ids = [fc["feature_id"] for fc in selected]
print(f"  selected: {len(annotated)} annotated + {len(unannotated_sample)} unannotated = {len(feature_ids)}")

# Load cells
ds = load_ds("k562")
rng2 = np.random.default_rng(SEED + 111)
ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
ctrl_idx = np.sort(rng2.choice(ctrl_all, size=N_CELLS, replace=False))
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((N_CELLS, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CELLS, 20):
        X_ctrl[i:i+20] = f["X"][ctrl_idx[i:i+20], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f2:
    sym2ens = pickle.load(f2)
tokenizer = MaxTokiTokenizer()
var_idx, var_tok, var_med = tokenizer.make_var_mapping([sym2ens.get(s) for s in var_symbols])
cells = [tokenizer.tokenize_cell(X_ctrl[ci], var_idx, var_tok, var_med, max_len=2048)
         for ci in range(N_CELLS)]
cells = [c for c in cells if c is not None]
print(f"  {len(cells)} cells")

# Load model
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

# Phase 2: Feature-outer, cell-inner tracing loop
t_total = time.time()
edge_counts = {}
n_total_edges = 0

for fi_idx, fi in enumerate(feature_ids):
    result_path = OUT / f"feature_F{fi:04d}.json"
    if result_path.exists():
        # Resume: load existing result
        r = json.loads(result_path.read_text())
        edge_counts[fi] = r.get("total_edges", 0)
        n_total_edges += r.get("total_edges", 0)
        if (fi_idx + 1) % 100 == 0:
            print(f"  [{fi_idx+1}/{len(feature_ids)}] (cached)")
        continue

    # Initialize accumulators for this feature
    accums = {dl: WelfordAccumulator(D_SAE) for dl in DOWNSTREAM_LAYERS}

    for ci, cell in enumerate(cells):
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

        # Clean forward
        with torch.no_grad():
            out = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            clean_hs = {l: out.hidden_states[l][0].cpu() for l in [SOURCE_LAYER] + DOWNSTREAM_LAYERS}
        clean_sae = {}
        for dl in DOWNSTREAM_LAYERS:
            with torch.no_grad():
                clean_sae[dl] = saes[dl].encode(
                    torch.from_numpy(clean_hs[dl].numpy().astype(np.float32)), mus[dl]).numpy()

        # Ablate
        with torch.no_grad():
            h_src = clean_hs[SOURCE_LAYER].numpy().astype(np.float32)
            z_src = saes[SOURCE_LAYER].encode(torch.from_numpy(h_src), mus[SOURCE_LAYER])
            z_abl = z_src.clone(); z_abl[:, fi] = 0.0
            h_recon = saes[SOURCE_LAYER].decode(z_src, mus[SOURCE_LAYER])
            h_abl_recon = saes[SOURCE_LAYER].decode(z_abl, mus[SOURCE_LAYER])
            delta = h_abl_recon - h_recon
            h_patched = (clean_hs[SOURCE_LAYER] + delta.cpu()).to(DEVICE).unsqueeze(0)

        def _hook(module, input, output, _p=h_patched):
            return _p.to(output.device)
        hook = xt.model.model.layers[SOURCE_LAYER].register_forward_hook(_hook)
        with torch.no_grad():
            out_abl = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        hook.remove()

        for dl in DOWNSTREAM_LAYERS:
            h_abl_dl = out_abl.hidden_states[dl][0].cpu().numpy().astype(np.float32)
            with torch.no_grad():
                z_abl_dl = saes[dl].encode(torch.from_numpy(h_abl_dl), mus[dl]).numpy()
            delta_z = (z_abl_dl - clean_sae[dl]).mean(axis=0)
            accums[dl].update(delta_z)

        del out, out_abl, clean_hs, clean_sae
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except: pass

    # Finalize: filter significant edges
    feature_edges = []
    for dl in DOWNSTREAM_LAYERS:
        d, cons, mean_d = accums[dl].finalize()
        sig = (np.abs(d) > D_THRESHOLD) & (cons > CONSISTENCY_THRESHOLD)
        for tgt in np.where(sig)[0]:
            feature_edges.append({
                "tgt_layer": dl, "tgt_feature": int(tgt),
                "cohens_d": round(float(d[tgt]), 4),
                "consistency": round(float(cons[tgt]), 4),
                "sign": "inhibitory" if mean_d[tgt] < 0 else "excitatory",
            })

    result = {
        "feature_id": fi,
        "source_layer": SOURCE_LAYER,
        "total_edges": len(feature_edges),
        "edges_by_layer": {str(dl): sum(1 for e in feature_edges if e["tgt_layer"] == dl)
                          for dl in DOWNSTREAM_LAYERS},
        "top_edges": sorted(feature_edges, key=lambda x: -abs(x["cohens_d"]))[:50],
    }
    result_path.write_text(json.dumps(result))
    edge_counts[fi] = len(feature_edges)
    n_total_edges += len(feature_edges)

    if (fi_idx + 1) % 20 == 0:
        elapsed = time.time() - t_total
        rate = (fi_idx + 1) / elapsed
        remaining = (len(feature_ids) - fi_idx - 1) / max(rate, 0.001)
        print(f"  [{fi_idx+1}/{len(feature_ids)}] edges={n_total_edges:,}  "
              f"this_feat={len(feature_edges)}  {elapsed/60:.1f}min elapsed  ETA={remaining/60:.0f}min")

# Phase 3: Hub analysis
print(f"\n[Phase 3] Hub analysis...")
ec_sorted = sorted(edge_counts.items(), key=lambda x: -x[1])
print(f"  total edges: {n_total_edges:,}")
print(f"  mean/feature: {n_total_edges/max(len(feature_ids),1):.1f}")
print(f"  features with >1000 edges: {sum(1 for _,c in ec_sorted if c > 1000)}")
print(f"  features with 0 edges: {sum(1 for _,c in ec_sorted if c == 0)}")
print(f"\n  Top 20 hubs:")
for fi, count in ec_sorted[:20]:
    ann = "annotated" if feat_n.get(fi, 0) > 0 else "unannotated"
    print(f"    F{fi}: {count} edges ({ann})")

# Phase 4: Annotation bias
n_top20_annotated = sum(1 for fi, _ in ec_sorted[:20] if feat_n.get(fi, 0) > 0)
n_top100_annotated = sum(1 for fi, _ in ec_sorted[:100] if feat_n.get(fi, 0) > 0)
overall_ann_rate = sum(1 for fc in selected if feat_n.get(fc["feature_id"], 0) > 0) / len(selected)
print(f"\n[Phase 4] Annotation bias:")
print(f"  overall annotation rate: {overall_ann_rate:.1%}")
print(f"  top-20 hubs annotated: {n_top20_annotated}/20 ({n_top20_annotated/20:.0%})")
print(f"  top-100 hubs annotated: {n_top100_annotated}/100 ({n_top100_annotated/100:.0%})")
print(f"  unannotated in top-20: {20 - n_top20_annotated} ({(20-n_top20_annotated)/20:.0%})")

# Save summary
with open(OUT / "exhaustive_summary.json", "w") as f:
    json.dump({
        "source_layer": SOURCE_LAYER,
        "downstream_layers": DOWNSTREAM_LAYERS,
        "n_features_traced": len(feature_ids),
        "n_cells": N_CELLS,
        "total_edges": n_total_edges,
        "mean_edges_per_feature": round(n_total_edges / max(len(feature_ids), 1), 1),
        "n_hubs_above_1000": sum(1 for _, c in ec_sorted if c > 1000),
        "n_zero_edge_features": sum(1 for _, c in ec_sorted if c == 0),
        "top20_annotation_rate": n_top20_annotated / 20,
        "overall_annotation_rate": round(overall_ann_rate, 4),
        "top20_hubs": [{"feature": fi, "edges": ct, "annotated": feat_n.get(fi, 0) > 0}
                       for fi, ct in ec_sorted[:20]],
    }, f, indent=2)

print(f"\nEXPERIMENT 1 COMPLETE — {n_total_edges:,} edges across {len(feature_ids)} features")
