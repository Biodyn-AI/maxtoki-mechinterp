"""Stage 2 — Causal Circuit Tracing over MaxToki-217M SAE features.

Phases 0-6: source selection, clean cache, single-feature ablation with
Welford accumulation, edge filtering, hub analysis, biological coherence.

For each of 30 source features at each of 4 source layers, zero the feature
in SAE space at the source layer, propagate the ablation via a forward hook,
and measure Cohen's d at every downstream layer's SAE features.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
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

SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/circuit-tracing-217M/outputs"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
N_CELLS = int(os.environ.get("N_CELLS", "200"))
N_FEATURES_PER_LAYER = 30
SOURCE_LAYERS = [int(x) for x in os.environ.get("SOURCE_LAYERS", "0,3,6,9").split(",")]
N_LAYERS = 12
D_SAE = 4928
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7

print("=" * 70)
print(f"STAGE 2 — Causal Circuit Tracing  N_cells={N_CELLS}")
print(f"  source_layers={SOURCE_LAYERS}  features/layer={N_FEATURES_PER_LAYER}")
print("=" * 70)


# ================================================================
# Welford online accumulator
# ================================================================
class WelfordAccumulator:
    """Online mean + variance for Cohen's d without storing per-cell values."""
    __slots__ = ("n", "mean", "M2", "pos_count")

    def __init__(self, size: int):
        self.n = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.M2 = np.zeros(size, dtype=np.float64)
        self.pos_count = np.zeros(size, dtype=np.int64)

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.pos_count += (x > 0).astype(np.int64)

    def finalize(self):
        if self.n < 2:
            return self.mean, np.ones_like(self.mean), np.zeros_like(self.mean)
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var + 1e-12)
        cohens_d = self.mean / std
        consistency = np.maximum(self.pos_count, self.n - self.pos_count) / self.n
        return cohens_d, consistency, self.mean


# ================================================================
# Phase 0: Load SAEs + model
# ================================================================
print("\n[Phase 0] Loading SAEs + model...")
saes = {}
mus = {}
for li in range(N_LAYERS):
    ckpt = torch.load(SAE_DIR / f"layer_{li:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval()
    saes[li] = sae
    mus[li] = ckpt["mu"]
print(f"  loaded {len(saes)} SAEs")

# Load cells
ds = load_ds("k562")
rng = np.random.default_rng(SEED)
ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
ctrl_idx = np.sort(rng.choice(ctrl_all, size=N_CELLS, replace=False))
with h5py.File(ds.h5_path, "r") as f:
    X_ctrl = np.empty((N_CELLS, ds.n_genes_total), dtype=np.float32)
    for i in range(0, N_CELLS, 100):
        X_ctrl[i:i + 100] = f["X"][ctrl_idx[i:i + 100], :]
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
with open(SYM2ENS_PKL, "rb") as f:
    sym2ens = pickle.load(f)
var_ens = [sym2ens.get(s) for s in var_symbols]
tokenizer = MaxTokiTokenizer()
var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(var_ens)

# Pre-tokenize
cells = []
for ci in range(N_CELLS):
    cell = tokenizer.tokenize_cell(X_ctrl[ci], var_idx_full, var_tok_full, var_med_full, max_len=2048)
    if cell is not None:
        cells.append(cell)
print(f"  {len(cells)} cells tokenized")

# Load model
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)


# ================================================================
# Phase 1: Source feature selection
# ================================================================
print("\n[Phase 1] Selecting source features...")
source_features = {}
for src_l in SOURCE_LAYERS:
    try:
        enrich_df = pd.read_csv(ANN_DIR / f"layer_{src_l:02d}/significant_enrichments.csv")
        # Annotation quality score = sum of -log10(p)
        scores = enrich_df.groupby("feature_id").apply(
            lambda g: (-np.log10(g["p_raw"].clip(1e-300))).sum()
        ).sort_values(ascending=False)
        top_features = scores.head(N_FEATURES_PER_LAYER).index.tolist()
    except FileNotFoundError:
        top_features = list(range(N_FEATURES_PER_LAYER))
    source_features[src_l] = top_features
    print(f"  L{src_l}: {len(top_features)} source features selected")


# ================================================================
# Phases 2-4: Per-source-layer circuit tracing
# ================================================================
all_edges = []
per_layer_stats = []

for src_l in SOURCE_LAYERS:
    downstream_layers = list(range(src_l + 1, N_LAYERS))
    if not downstream_layers:
        continue
    features = source_features[src_l]
    print(f"\n{'='*50}")
    print(f"  Tracing L{src_l} → {downstream_layers} ({len(features)} features × {len(cells)} cells)")
    print(f"{'='*50}")

    # Initialize Welford accumulators: one per (source_feature, downstream_layer)
    accumulators = {}
    for fi in features:
        for dl in downstream_layers:
            accumulators[(fi, dl)] = WelfordAccumulator(D_SAE)

    t0 = time.time()
    n_passes = 0

    for ci, cell in enumerate(cells):
        input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

        # Phase 2: Clean forward — capture all hidden states
        with torch.no_grad():
            out_clean = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            clean_hidden = {l: out_clean.hidden_states[l][0].cpu() for l in range(N_LAYERS)}
        n_passes += 1

        # Encode clean hidden states through SAEs at downstream layers
        clean_sae = {}
        for dl in downstream_layers:
            with torch.no_grad():
                h_clean = clean_hidden[dl].numpy().astype(np.float32)
                z_clean = saes[dl].encode(torch.from_numpy(h_clean), mus[dl]).numpy()
            clean_sae[dl] = z_clean  # (seq, d_sae)

        # Phase 3: Per-feature ablation
        for fi in features:
            # Encode source layer, zero feature, decode
            with torch.no_grad():
                h_src = clean_hidden[src_l].numpy().astype(np.float32)
                z_src = saes[src_l].encode(torch.from_numpy(h_src), mus[src_l])
                z_abl = z_src.clone()
                z_abl[:, fi] = 0.0
                h_src_recon = saes[src_l].decode(z_src, mus[src_l])
                h_src_abl_recon = saes[src_l].decode(z_abl, mus[src_l])
                # Delta in residual stream basis
                delta = h_src_abl_recon - h_src_recon
                # Apply to original hidden state
                h_patched = (clean_hidden[src_l] + delta.cpu()).to(DEVICE).unsqueeze(0)

            # Ablated forward via hook
            def _hook(module, input, output, _p=h_patched):
                return _p.to(output.device)

            hook = xt.model.model.layers[src_l].register_forward_hook(_hook)
            with torch.no_grad():
                out_abl = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            hook.remove()
            n_passes += 1

            # Measure downstream effects
            for dl in downstream_layers:
                h_abl_dl = out_abl.hidden_states[dl][0].cpu().numpy().astype(np.float32)
                with torch.no_grad():
                    z_abl_dl = saes[dl].encode(torch.from_numpy(h_abl_dl), mus[dl]).numpy()
                # Per-feature delta (mean across sequence positions)
                delta_z = (z_abl_dl - clean_sae[dl]).mean(axis=0)  # (d_sae,)
                accumulators[(fi, dl)].update(delta_z)

            del out_abl

        del out_clean, clean_hidden, clean_sae
        if DEVICE == "mps" and ci % 10 == 0:
            try:
                torch.mps.empty_cache()
            except:
                pass

        if (ci + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = n_passes / elapsed
            remaining = (len(cells) - ci - 1) * (1 + len(features)) / rate
            print(f"    cell {ci+1}/{len(cells)}  passes={n_passes}  "
                  f"rate={rate:.1f}/s  ETA={remaining/60:.0f}min")

    # Phase 4: Finalize and filter edges
    n_edges = 0
    layer_edges = defaultdict(list)
    for fi in features:
        for dl in downstream_layers:
            d, cons, mean_delta = accumulators[(fi, dl)].finalize()
            # Find significant targets
            sig_mask = (np.abs(d) > D_THRESHOLD) & (cons > CONSISTENCY_THRESHOLD)
            for tgt_idx in np.where(sig_mask)[0]:
                edge = {
                    "src_layer": src_l, "src_feature": fi,
                    "tgt_layer": dl, "tgt_feature": int(tgt_idx),
                    "cohens_d": round(float(d[tgt_idx]), 4),
                    "consistency": round(float(cons[tgt_idx]), 4),
                    "sign": "inhibitory" if mean_delta[tgt_idx] < 0 else "excitatory",
                }
                all_edges.append(edge)
                layer_edges[dl].append(edge)
                n_edges += 1

    elapsed = time.time() - t0
    per_layer_stats.append({
        "source_layer": src_l, "n_features": len(features),
        "n_downstream": len(downstream_layers), "n_passes": n_passes,
        "n_edges": n_edges, "compute_seconds": round(elapsed),
        "edges_per_feature": round(n_edges / max(len(features), 1), 1),
    })
    print(f"  L{src_l} complete: {n_edges} edges in {elapsed:.0f}s  "
          f"({n_edges/max(len(features),1):.0f} edges/feature)")

del xt
if DEVICE == "mps":
    try:
        torch.mps.empty_cache()
    except:
        pass


# ================================================================
# Phase 5: Circuit graph + hub analysis
# ================================================================
print(f"\n[Phase 5] Circuit graph + hub analysis...")
edges_df = pd.DataFrame(all_edges)
edges_df.to_csv(OUT / "circuit_edges.csv", index=False)
n_total = len(edges_df)
print(f"  total edges: {n_total}")

if n_total > 0:
    # Inhibitory/excitatory
    n_inh = int((edges_df["sign"] == "inhibitory").sum())
    inh_pct = n_inh / n_total
    print(f"  inhibitory: {n_inh} ({inh_pct:.1%})  excitatory: {n_total - n_inh} ({1 - inh_pct:.1%})")

    # Effect size distribution
    abs_d = edges_df["cohens_d"].abs()
    print(f"  mean |d|: {abs_d.mean():.3f}  median: {abs_d.median():.3f}")
    print(f"  |d| > 1.0: {(abs_d > 1).mean():.1%}  |d| > 2.0: {(abs_d > 2).mean():.1%}")

    # Hub analysis — out-degree
    out_deg = edges_df.groupby(["src_layer", "src_feature"]).size().sort_values(ascending=False)
    print(f"\n  Top 10 broadcast hubs (out-degree):")
    for (sl, sf), deg in out_deg.head(10).items():
        print(f"    L{sl}_F{sf}: {deg}")

    # In-degree
    in_deg = edges_df.groupby(["tgt_layer", "tgt_feature"]).size().sort_values(ascending=False)
    print(f"\n  Top 10 convergent targets (in-degree):")
    for (tl, tf), deg in in_deg.head(10).items():
        print(f"    L{tl}_F{tf}: {deg}")

    # Target coverage
    n_unique_targets = edges_df[["tgt_layer", "tgt_feature"]].drop_duplicates().shape[0]
    total_possible = D_SAE * (N_LAYERS - min(SOURCE_LAYERS) - 1)
    coverage = n_unique_targets / max(total_possible, 1)
    print(f"\n  unique targets: {n_unique_targets}  coverage: {coverage:.1%}")


# ================================================================
# Phase 6: Biological coherence
# ================================================================
print(f"\n[Phase 6] Biological coherence...")
# Load feature annotations
feat_terms = {}
for li in range(N_LAYERS):
    try:
        enr = pd.read_csv(ANN_DIR / f"layer_{li:02d}/significant_enrichments.csv")
        for fid, grp in enr.groupby("feature_id"):
            feat_terms[(li, fid)] = set(grp["term"].tolist())
    except FileNotFoundError:
        pass

n_annotated_edges = 0
n_shared = 0
for _, e in edges_df.iterrows():
    src_terms = feat_terms.get((e["src_layer"], e["src_feature"]), set())
    tgt_terms = feat_terms.get((e["tgt_layer"], e["tgt_feature"]), set())
    if src_terms and tgt_terms:
        n_annotated_edges += 1
        if src_terms & tgt_terms:
            n_shared += 1

coherence = n_shared / max(n_annotated_edges, 1)
print(f"  annotated edges: {n_annotated_edges}")
print(f"  shared ontology: {n_shared} ({coherence:.1%})")
print(f"  (paper: ~53% for K562/K562)")


# ================================================================
# Save results
# ================================================================
summary = {
    "model": "MaxToki-217M",
    "condition": "K562/K562",
    "n_cells": len(cells),
    "source_layers": SOURCE_LAYERS,
    "n_features_per_layer": N_FEATURES_PER_LAYER,
    "total_source_features": sum(len(v) for v in source_features.values()),
    "total_edges": n_total,
    "inhibitory_pct": round(inh_pct, 4) if n_total > 0 else 0,
    "mean_abs_d": round(float(abs_d.mean()), 4) if n_total > 0 else 0,
    "median_abs_d": round(float(abs_d.median()), 4) if n_total > 0 else 0,
    "pct_d_above_1": round(float((abs_d > 1).mean()), 4) if n_total > 0 else 0,
    "unique_targets": n_unique_targets if n_total > 0 else 0,
    "biological_coherence": round(coherence, 4),
    "per_layer_stats": per_layer_stats,
}
with open(OUT / "circuit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print(f"CIRCUIT TRACING COMPLETE")
print(f"  total edges: {n_total}")
print(f"  inhibitory: {inh_pct:.1%}" if n_total > 0 else "  no edges")
print(f"  coherence: {coherence:.1%}")
print(f"  outputs: {OUT}")
print(f"{'='*70}")
