"""Remaining SAE Atlas phases: 6, 7, 8-true, 9 (multi-tissue), 10, 11.

Phase 6:  Causal feature patching at L5 (50 features × 200 cells)
Phase 7:  Cross-layer information highways (L0↔L5, L5↔L11 PMI)
Phase 8t: True CRISPRi perturbation response (extract perturbed cells)
Phase 9:  Multi-tissue SAE control (pool K562 + Tabula Sapiens)
Phase 10: Unannotated feature characterisation
Phase 11: Cell-type enrichment mapping
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
import scipy.sparse as sp
import torch
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests as _mult

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE, train_sae
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

PHASE0 = PROJ / "runs/sae-atlas-217M/outputs/phase0"
PHASE1 = PROJ / "runs/sae-atlas-217M/outputs/phase1"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/sae-atlas-217M/outputs"

TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
GO_BP_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"
GO_PKL = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl"
REPLOGLE_H5 = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"

cfg = json.loads((PHASE0 / "run_config.json").read_text())
LAYERS = cfg["target_layers"]
HIDDEN = cfg["hidden_size"]
PROBE_LAYER = 5
SEED = 42
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")

trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_targets = {}
for _, r in trrust.iterrows():
    trrust_targets.setdefault(r["tf"], set()).add(r["target"])
trrust_tfs = set(trrust_targets.keys())


def _load_sae(layer):
    ckpt = torch.load(PHASE1 / f"layer_{layer:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    m = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    m.W_enc.weight.data = ckpt["W_enc_weight"]
    m.W_enc.bias.data = ckpt["W_enc_bias"]
    m.W_dec.weight.data = ckpt["W_dec_weight"]
    m.eval()
    return m, ckpt["mu"]


print("=" * 70)
print("SAE REMAINING PHASES (6, 7, 8t, 10, 11)")
print("=" * 70)


# =====================================================================
# PHASE 6 — Causal feature patching
# =====================================================================
print("\n[Phase 6] Causal feature patching at L5...")
(OUT / "phase6").mkdir(parents=True, exist_ok=True)

sae, mu = _load_sae(PROBE_LAYER)
d_sae = sae.d_sae

# Select 50 richly annotated features (≥3 enrichments)
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
)[:50]
print(f"  richly annotated features (≥3 enrichments): {len(rich_features)}")

if len(rich_features) >= 10 and os.environ.get("SKIP_PHASE6") != "1":
    # Load MaxToki for forward passes from L5 to output
    xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)

    # Load 200 K562 control cells
    ds = load_ds("k562")
    rng = np.random.default_rng(SEED)
    ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    ctrl_200 = np.sort(rng.choice(ctrl_all, size=200, replace=False))
    with h5py.File(ds.h5_path, "r") as f:
        X_200 = np.empty((200, ds.n_genes_total), dtype=np.float32)
        for i in range(0, 200, 50):
            X_200[i:i+50] = f["X"][ctrl_200[i:i+50], :]
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]

    with open(SYM2ENS_PKL, "rb") as f2:
        sym2ens = pickle.load(f2)
    var_ens = [sym2ens.get(s) for s in var_symbols]
    tokenizer = MaxTokiTokenizer()
    var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(var_ens)

    # For each feature: run clean forward, then zero the feature and re-forward
    # Measure per-gene logit change
    # Simplified: we compare the clean logit output to the patched output
    # where one SAE feature is zeroed at L5.
    #
    # Implementation: run full forward to get clean logits, then for each
    # feature, run partial forward from L5 with the patched residual stream.
    #
    # MaxToki (Llama) doesn't have a built-in "continue from layer L" API,
    # so we need to:
    # 1. Run full forward saving hidden states at L5
    # 2. Encode L5 hidden state through SAE, zero the feature, decode
    # 3. Replace the L5 hidden state and continue forward from L5+1 to end
    #
    # For step 3, we manually run model.model.layers[5:] + model.model.norm + model.lm_head

    feat_top20_genes = {fc["feature_id"]: set(fc["top20_genes"]) for fc in catalog}

    # Build the GO annotation gene sets for each rich feature
    with open(GO_PKL, "rb") as f:
        gene2go = pickle.load(f)
    # Reverse: GO term → set of genes
    go_term_genes = defaultdict(set)
    for gene, terms in gene2go.items():
        for t in terms:
            go_term_genes[t].add(gene.upper())

    # Hook-based causal patching: register a forward hook on
    # model.model.layers[PROBE_LAYER] that intercepts and modifies the output.
    patching_results = []
    n_cells_patch = min(50, len(X_200))

    for fi_idx, fc in enumerate(rich_features[:50]):
        fi = fc["feature_id"]
        target_genes = feat_top20_genes.get(fi, set())
        if len(target_genes) < 3:
            continue

        delta_targets = []
        delta_others = []

        for ci in range(n_cells_patch):
            cell = tokenizer.tokenize_cell(
                X_200[ci], var_idx_full, var_tok_full, var_med_full, max_len=2048
            )
            if cell is None:
                continue
            input_ids = torch.from_numpy(cell.token_ids[None, :]).to(DEVICE)

            # Clean forward: get all hidden states + logits
            with torch.no_grad():
                out_clean = xt.model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
                clean_logits = out_clean.logits[0].cpu()  # (seq, vocab)
                h5 = out_clean.hidden_states[PROBE_LAYER + 1][0].cpu()  # (seq, hidden) — post-L5

            # Encode through SAE, zero the feature, decode
            with torch.no_grad():
                h5_np = h5.numpy().astype(np.float32)
                h_enc = sae.encode(torch.from_numpy(h5_np), mu)  # (seq, d_sae)
                h_enc[:, fi] = 0.0  # zero the target feature
                h5_patched = sae.decode(h_enc, mu)  # (seq, hidden)

            # Continue forward using a hook to inject the patched hidden state
            # at L5, then run the full model forward (the hook replaces L5 output).
            _patched_hs = h5_patched.to(DEVICE).unsqueeze(0).clone()  # (1, seq, hidden)
            _hook_handle = None

            def _patch_hook(module, input, output, _p=_patched_hs):
                # output is tuple: (hidden_states, ...)
                return (_p.to(output[0].device),) + output[1:]

            _hook_handle = xt.model.model.layers[PROBE_LAYER].register_forward_hook(_patch_hook)
            with torch.no_grad():
                out_patched = xt.model(input_ids, output_hidden_states=False, use_cache=False, return_dict=True)
                patched_logits = out_patched.logits[0].cpu()
            _hook_handle.remove()

            # Logit change per position
            delta = (patched_logits - clean_logits).mean(dim=0)  # (vocab,)

            # Map gene symbols to vocab token IDs
            gene_to_tok = {}
            for ti in range(len(cell.token_ids)):
                tok_id = int(cell.token_ids[ti])
                if tok_id in tokenizer.full_token_dict.values():
                    # reverse: find gene for this token
                    pass
            # Simpler: compute mean absolute logit delta across ALL tokens as
            # "other", and for tokens corresponding to target genes as "target"
            seq_len = cell.token_ids.shape[0]
            gene_positions = cell.gene_positions
            target_mask = np.zeros(seq_len, dtype=bool)
            for pos_i in range(seq_len):
                gp = gene_positions[pos_i]
                if gp >= 0:
                    gene_sym = var_symbols[var_idx_full[gp]].upper() if gp < len(var_idx_full) else ""
                    if gene_sym in target_genes:
                        target_mask[pos_i] = True

            d_all = (patched_logits - clean_logits).abs().mean(dim=1).numpy()  # (seq,)
            if target_mask.sum() > 0:
                delta_targets.append(float(d_all[target_mask].mean()))
            if (~target_mask).sum() > 0:
                delta_others.append(float(d_all[~target_mask].mean()))

        if delta_targets and delta_others:
            mean_target = float(np.mean(delta_targets))
            mean_other = float(np.mean(delta_others))
            specificity = mean_target / max(mean_other, 1e-12)
            patching_results.append({
                "feature_id": fi,
                "mean_target_delta": mean_target,
                "mean_other_delta": mean_other,
                "specificity_ratio": specificity,
                "n_cells": n_cells_patch,
            })

        if (fi_idx + 1) % 10 == 0:
            print(f"    {fi_idx+1}/{len(rich_features[:50])} features patched")

    if patching_results:
        patch_df = pd.DataFrame(patching_results)
        patch_df.to_csv(OUT / "phase6/causal_patching.csv", index=False)
        med_spec = float(patch_df["specificity_ratio"].median())
        pct_gt2 = float((patch_df["specificity_ratio"] > 2).mean())
        print(f"  median specificity: {med_spec:.2f}×  >2×: {pct_gt2:.1%}")
        with open(OUT / "phase6/summary.json", "w") as f:
            json.dump({
                "n_features_patched": len(patching_results),
                "median_specificity": med_spec,
                "pct_above_2x": pct_gt2,
                "mean_target_delta": float(patch_df["mean_target_delta"].mean()),
                "mean_other_delta": float(patch_df["mean_other_delta"].mean()),
            }, f, indent=2)
    else:
        print("  no patching results (insufficient features)")

    # Clean up GPU memory
    del xt
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except: pass
else:
    print("  Phase 6 skipped (SKIP_PHASE6=1 or insufficient features)")


# =====================================================================
# PHASE 7 — Cross-layer information highways
# =====================================================================
print("\n[Phase 7] Cross-layer information highways...")
(OUT / "phase7").mkdir(parents=True, exist_ok=True)

layer_pairs = [(0, 5), (5, 11)]
for src_l, tgt_l in layer_pairs:
    print(f"  computing cross-layer PMI L{src_l}→L{tgt_l}...")
    t0 = time.time()
    acts_src = np.load(PHASE0 / f"layer_{src_l:02d}_activations.npy")
    acts_tgt = np.load(PHASE0 / f"layer_{tgt_l:02d}_activations.npy")
    sae_src, mu_src = _load_sae(src_l)
    sae_tgt, mu_tgt = _load_sae(tgt_l)
    d_src = sae_src.d_sae
    d_tgt = sae_tgt.d_sae

    # Use 500K positions (subsample if needed)
    n_pos = min(500000, acts_src.shape[0])

    # Accumulate co-activation + marginals incrementally
    src_counts = np.zeros(d_src, dtype=np.float64)
    tgt_counts = np.zeros(d_tgt, dtype=np.float64)
    # Cross-layer co-occurrence is (d_src, d_tgt) — at 4928² × 8 bytes = 194 MB. OK.
    cross_cooc = np.zeros((d_src, d_tgt), dtype=np.float64)

    chunk = 20000
    for ci in range(0, n_pos, chunk):
        end = min(ci + chunk, n_pos)
        with torch.no_grad():
            h_s = sae_src.encode(torch.from_numpy(acts_src[ci:end].astype(np.float32)), mu_src).numpy()
            h_t = sae_tgt.encode(torch.from_numpy(acts_tgt[ci:end].astype(np.float32)), mu_tgt).numpy()
        a_s = (h_s > 0).astype(np.float32)
        a_t = (h_t > 0).astype(np.float32)
        src_counts += a_s.sum(axis=0)
        tgt_counts += a_t.sum(axis=0)
        cross_cooc += a_s.T @ a_t
        del h_s, h_t, a_s, a_t

    # PMI
    src_freq = src_counts / n_pos
    tgt_freq = tgt_counts / n_pos
    expected = np.outer(src_freq, tgt_freq) * n_pos
    expected[expected == 0] = 1e-12
    pmi = np.log2(cross_cooc / expected + 1e-12)
    pmi[cross_cooc == 0] = 0

    # Per source feature: max PMI to any target
    max_pmi_per_src = pmi.max(axis=1)
    n_highways = int((max_pmi_per_src > 3).sum())
    n_alive_src = int((src_counts > 0).sum())
    highway_rate = n_highways / max(n_alive_src, 1)
    mean_max_pmi = float(max_pmi_per_src[src_counts > 0].mean()) if n_alive_src > 0 else 0

    print(f"    highways: {n_highways}/{n_alive_src} ({highway_rate:.1%})  "
          f"mean max PMI: {mean_max_pmi:.2f}  ({time.time()-t0:.1f}s)")

    with open(OUT / f"phase7/highways_L{src_l}_to_L{tgt_l}.json", "w") as f:
        json.dump({
            "src_layer": src_l, "tgt_layer": tgt_l,
            "n_positions": n_pos,
            "n_alive_src": n_alive_src,
            "n_highways": n_highways,
            "highway_rate": highway_rate,
            "mean_max_pmi": mean_max_pmi,
            "median_max_pmi": float(np.median(max_pmi_per_src[src_counts > 0])) if n_alive_src else 0,
            "max_max_pmi": float(max_pmi_per_src.max()),
        }, f, indent=2)
    del acts_src, acts_tgt, cross_cooc, pmi


# =====================================================================
# PHASE 8t — True CRISPRi perturbation response
# =====================================================================
print("\n[Phase 8t] True CRISPRi perturbation response...")
(OUT / "phase8_true").mkdir(parents=True, exist_ok=True)

# Load MaxToki for perturbed cell extraction
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
sae_probe, mu_probe = _load_sae(PROBE_LAYER)

# Load Replogle dataset
ds = load_ds("k562")
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

    # Select 100 perturbation targets (48 TFs + 52 non-TFs) with ≥20 K562 cells
    rng = np.random.default_rng(SEED)
    k562_cells = np.where(k562_mask)[0]
    k562_codes = pg_codes[k562_mask]

    pert_targets = []
    for code in np.unique(k562_codes):
        cat = pg_cats[code]
        if "non-targeting" in cat.lower() or "nontargeting" in cat.lower():
            continue
        sel = k562_cells[k562_codes == code]
        if len(sel) < 20:
            continue
        pert_targets.append((cat, code, sel))
    print(f"  perturbation targets with ≥20 K562 cells: {len(pert_targets)}")

    # Split into TFs and non-TFs
    tf_targets = [(c, code, sel) for c, code, sel in pert_targets if c.upper() in trrust_tfs]
    non_tf_targets = [(c, code, sel) for c, code, sel in pert_targets if c.upper() not in trrust_tfs]
    tf_targets = tf_targets[:48]
    non_tf_targets = non_tf_targets[:52]
    selected = tf_targets + non_tf_targets
    print(f"  selected: {len(tf_targets)} TFs + {len(non_tf_targets)} non-TFs = {len(selected)}")

    # Load control activations for comparison
    ctrl_acts = np.load(PHASE0 / f"layer_{PROBE_LAYER:02d}_activations.npy")
    n_ctrl_use = min(100000, ctrl_acts.shape[0])
    with torch.no_grad():
        h_ctrl_list = []
        for i in range(0, n_ctrl_use, 20000):
            xb = torch.from_numpy(ctrl_acts[i:i + 20000].astype(np.float32))
            h_ctrl_list.append(sae_probe.encode(xb, mu_probe).numpy())
        h_ctrl = np.concatenate(h_ctrl_list, axis=0)
    del ctrl_acts

    # Feature catalog top-20 genes
    with open(PHASE2 / f"layer_{PROBE_LAYER:02d}/feature_catalog.json") as ff:
        catalog = json.load(ff)
    feat_top20 = {c["feature_id"]: set(c["top20_genes"]) for c in catalog}

    pert_results = []
    for pi, (cat, code, sel) in enumerate(selected):
        is_tf = cat.upper() in trrust_tfs
        # Extract 20 perturbed cells through MaxToki
        cell_idx = rng.choice(sel, size=min(20, len(sel)), replace=False)
        pert_activations = []
        for ci in cell_idx:
            X_cell = f["X"][int(ci), :].astype(np.float32)
            cell = tokenizer.tokenize_cell(X_cell, var_idx_full, var_tok_full, var_med_full, max_len=2048)
            if cell is None:
                continue
            input_ids = torch.from_numpy(cell.token_ids[None, :])
            _, hidden = xt.forward_with_hidden_states(input_ids)
            h_l5 = hidden[PROBE_LAYER + 1][0].numpy()  # (seq, hidden)
            pert_activations.append(h_l5)
            del hidden
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass

        if not pert_activations:
            continue

        # Encode perturbed activations through SAE
        pert_flat = np.concatenate(pert_activations, axis=0).astype(np.float32)
        with torch.no_grad():
            h_pert = sae_probe.encode(torch.from_numpy(pert_flat), mu_probe).numpy()

        # Test each feature: Wilcoxon rank-sum perturbed vs control
        d_sae = h_pert.shape[1]
        responding = []
        for fi in range(d_sae):
            t_vals = h_pert[:, fi]
            o_vals = h_ctrl[:min(10000, h_ctrl.shape[0]), fi]
            if t_vals.std() < 1e-8 and o_vals.std() < 1e-8:
                continue
            try:
                _, p = sp_stats.mannwhitneyu(t_vals, o_vals, alternative="two-sided")
            except ValueError:
                continue
            effect = float(abs(t_vals.mean() - o_vals.mean()))
            if p < 0.05 and effect > 0.5:
                responding.append(fi)

        # TF specificity
        is_specific = False
        if is_tf and responding:
            known = trrust_targets.get(cat.upper(), set())
            for fi in responding:
                if len(feat_top20.get(fi, set()) & known) >= 2:
                    is_specific = True
                    break

        pert_results.append({
            "target": cat, "is_tf": is_tf, "n_cells": len(pert_activations),
            "n_responding": len(responding), "detected": len(responding) > 0,
            "is_specific": is_specific,
        })
        if (pi + 1) % 20 == 0:
            print(f"    {pi+1}/{len(selected)} targets processed")

pert_df = pd.DataFrame(pert_results)
pert_df.to_csv(OUT / "phase8_true/perturbation_response.csv", index=False)
n_det = int(pert_df["detected"].sum())
tf_df = pert_df[pert_df["is_tf"]]
n_spec = int(tf_df["is_specific"].sum())
n_tfs = len(tf_df)
print(f"  TRUE CRISPRi detection: {n_det}/{len(pert_df)} ({n_det/max(len(pert_df),1):.1%})")
print(f"  TRUE CRISPRi TF specificity: {n_spec}/{n_tfs} ({n_spec/max(n_tfs,1):.1%})")
with open(OUT / "phase8_true/summary.json", "w") as f:
    json.dump({
        "detection_rate": n_det / max(len(pert_df), 1),
        "n_detected": n_det, "n_targets": len(pert_df),
        "n_tf_specific": n_spec, "n_tfs": n_tfs,
        "tf_specificity_rate": n_spec / max(n_tfs, 1),
    }, f, indent=2)

del xt
if DEVICE == "mps":
    try: torch.mps.empty_cache()
    except: pass


# =====================================================================
# PHASE 10 — Unannotated feature characterisation
# =====================================================================
print("\n[Phase 10] Unannotated feature characterisation...")
(OUT / "phase10").mkdir(parents=True, exist_ok=True)

for li in LAYERS:
    with open(PHASE2 / f"layer_{li:02d}/feature_catalog.json") as f:
        catalog = json.load(f)
    try:
        sig = pd.read_csv(PHASE2 / f"layer_{li:02d}/significant_enrichments.csv")
        annotated_ids = set(sig["feature_id"].unique())
    except FileNotFoundError:
        annotated_ids = set()

    all_ids = set(c["feature_id"] for c in catalog)
    unannotated_ids = all_ids - annotated_ids
    n_all = len(all_ids)
    n_unannotated = len(unannotated_ids)
    print(f"  L{li}: {n_unannotated}/{n_all} unannotated ({n_unannotated/max(n_all,1):.1%})")

    # Jaccard standalone clustering among unannotated features
    unannotated_catalog = [c for c in catalog if c["feature_id"] in unannotated_ids]
    if len(unannotated_catalog) >= 10:
        # Pairwise Jaccard of top-20 gene sets
        gene_sets = [set(c["top20_genes"]) for c in unannotated_catalog]
        n_un = len(gene_sets)
        jacc = np.zeros((n_un, n_un), dtype=np.float32)
        for i in range(n_un):
            for j in range(i + 1, n_un):
                inter = len(gene_sets[i] & gene_sets[j])
                union = len(gene_sets[i] | gene_sets[j])
                if union > 0:
                    jacc[i, j] = jacc[j, i] = inter / union

        # Simple clustering: connected components at Jaccard > 0.3
        adj = (jacc > 0.3).astype(np.int8)
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        cluster_sizes = np.bincount(labels)
        real_clusters = int((cluster_sizes >= 3).sum())  # clusters with ≥3 members
        features_in_clusters = int(cluster_sizes[cluster_sizes >= 3].sum())
    else:
        real_clusters = 0
        features_in_clusters = 0

    # Guilt-by-association: check co-activation module membership
    module_path = OUT / f"phase5/layer_{li:02d}/module_labels.npy"
    if module_path.exists():
        module_labels = np.load(module_path)
        # Check if unannotated features share modules with annotated
        n_co = 0
        n_isolated = 0
        for uid in unannotated_ids:
            if uid < len(module_labels) and module_labels[uid] >= 0:
                # Check if any annotated feature is in the same module
                module = module_labels[uid]
                module_members = set(np.where(module_labels == module)[0].tolist())
                if module_members & annotated_ids:
                    n_co += 1
                else:
                    n_isolated += 1
            else:
                n_isolated += 1
        co_rate = n_co / max(n_unannotated, 1)
    else:
        n_co = 0; n_isolated = n_unannotated; co_rate = 0

    print(f"    Jaccard clusters (≥3 members): {real_clusters} with {features_in_clusters} features")
    print(f"    guilt-by-association: {n_co}/{n_unannotated} co-activate with annotated ({co_rate:.1%})  "
          f"isolated: {n_isolated}")

    with open(OUT / f"phase10/unannotated_L{li}.json", "w") as f:
        json.dump({
            "layer": li, "n_total": n_all, "n_unannotated": n_unannotated,
            "pct_unannotated": n_unannotated / max(n_all, 1),
            "n_jaccard_clusters": real_clusters,
            "n_features_in_clusters": features_in_clusters,
            "n_co_activate_with_annotated": n_co,
            "n_isolated": n_isolated,
            "co_activation_rate": co_rate,
        }, f, indent=2)


# =====================================================================
# PHASE 11 — Cell-type enrichment mapping
# =====================================================================
print("\n[Phase 11] Cell-type enrichment mapping...")
(OUT / "phase11").mkdir(parents=True, exist_ok=True)

# Load Tabula Sapiens immune cells, extract L5 activations through MaxToki
print("  loading Tabula Sapiens immune...")
import anndata as ad
adata = ad.read_h5ad(str(TS_H5), backed="r")
# Sample 1000 cells
rng_ts = np.random.default_rng(SEED + 1111)
ts_idx = np.sort(rng_ts.choice(adata.n_obs, size=1000, replace=False))
cell_types_ts = adata.obs["cell_type"].iloc[ts_idx].astype(str).to_numpy()
X_ts = adata[ts_idx].X
if sp.issparse(X_ts) or hasattr(X_ts, "toarray"):
    X_ts = X_ts.toarray()
X_ts = np.asarray(X_ts, dtype=np.float32)
adata.file.close()

var_ens_ts = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
var_idx_ts, var_tok_ts, var_med_ts = tokenizer.make_var_mapping(var_ens_ts.tolist())

print("  extracting L5 activations on 1000 TS cells...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
ts_activations = []
ts_cell_types_per_pos = []
for ci in range(min(1000, X_ts.shape[0])):
    cell = tokenizer.tokenize_cell(X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048)
    if cell is None:
        continue
    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, hidden = xt.forward_with_hidden_states(input_ids)
    h_l5 = hidden[PROBE_LAYER + 1][0].numpy()
    ts_activations.append(h_l5)
    ts_cell_types_per_pos.extend([cell_types_ts[ci]] * h_l5.shape[0])
    del hidden
    if DEVICE == "mps" and ci % 50 == 0:
        try: torch.mps.empty_cache()
        except: pass
    if (ci + 1) % 200 == 0:
        print(f"    {ci+1}/1000 cells...")

if ts_activations:
    ts_flat = np.concatenate(ts_activations, axis=0).astype(np.float32)
    print(f"  TS positions: {ts_flat.shape[0]:,}")

    # Encode through L5 SAE
    with torch.no_grad():
        h_ts_list = []
        for i in range(0, ts_flat.shape[0], 20000):
            xb = torch.from_numpy(ts_flat[i:i + 20000])
            h_ts_list.append(sae_probe.encode(xb, mu_probe).numpy())
        h_ts = np.concatenate(h_ts_list, axis=0)

    ct_array = np.array(ts_cell_types_per_pos[:h_ts.shape[0]])
    unique_cts = sorted(set(ct_array))
    print(f"  cell types: {len(unique_cts)}")

    # Per feature: test enrichment for each cell type (Fisher's exact)
    d_sae = h_ts.shape[1]
    n_enriched_features = 0
    ct_results = []
    for fi in range(d_sae):
        active = h_ts[:, fi] > 0
        if active.sum() < 10:
            continue
        has_any_ct = False
        for ct in unique_cts:
            ct_mask = ct_array == ct
            # 2×2: active & ct, active & ~ct, ~active & ct, ~active & ~ct
            a = int((active & ct_mask).sum())
            b = int((active & ~ct_mask).sum())
            c = int((~active & ct_mask).sum())
            d = int((~active & ~ct_mask).sum())
            if a < 3:
                continue
            _, p = sp_stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            if p < 0.05 / len(unique_cts):  # Bonferroni per cell type
                has_any_ct = True
                ct_results.append({"feature_id": fi, "cell_type": ct, "p": float(p),
                                  "a": a, "b": b, "c": c, "d": d})
        if has_any_ct:
            n_enriched_features += 1

    enrichment_rate = n_enriched_features / max(d_sae, 1)
    print(f"  features with ≥1 cell-type enrichment: {n_enriched_features}/{d_sae} ({enrichment_rate:.1%})")
    pd.DataFrame(ct_results).to_csv(OUT / "phase11/celltype_enrichments.csv", index=False)
    with open(OUT / "phase11/summary.json", "w") as f:
        json.dump({
            "n_features_enriched": n_enriched_features,
            "n_features_total": d_sae,
            "enrichment_rate": enrichment_rate,
            "n_cell_types": len(unique_cts),
            "n_ts_positions": int(h_ts.shape[0]),
        }, f, indent=2)

del xt
if DEVICE == "mps":
    try: torch.mps.empty_cache()
    except: pass


print("\n" + "=" * 70)
print("ALL REMAINING SAE PHASES COMPLETE")
print("=" * 70)
