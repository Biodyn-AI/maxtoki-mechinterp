"""Phase 9 — Multi-tissue SAE control.

Pool 500K K562 positions + 500K Tabula Sapiens positions at L5, retrain
SAE, re-annotate, re-run perturbation specificity test. Tests whether
training data breadth rescues regulatory specificity.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests as _mult

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE, train_sae
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
from dataset_loader import resolve as load_ds, SYM2ENS_PKL

PHASE0 = PROJ / "runs/sae-atlas-217M/outputs/phase0"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT = PROJ / "runs/sae-atlas-217M/outputs/phase9"
OUT.mkdir(parents=True, exist_ok=True)

TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
REPLOGLE_H5 = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"

PROBE_LAYER = 5
HIDDEN = 1232
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

print("=" * 70)
print("Phase 9 — Multi-tissue SAE control")
print("=" * 70)

# Step 1: Pool K562 + Tabula Sapiens L5 activations
print("\n[1] Pooling K562 + Tabula Sapiens activations at L5...")
k562_acts = np.load(PHASE0 / f"layer_{PROBE_LAYER:02d}_activations.npy")
n_k562 = k562_acts.shape[0]
print(f"  K562 positions: {n_k562:,}")

# Extract TS L5 activations (500 cells to get ~500K positions)
print("  extracting Tabula Sapiens L5 activations (500 cells)...")
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
tokenizer = MaxTokiTokenizer()
adata = ad.read_h5ad(str(TS_H5), backed="r")
rng = np.random.default_rng(SEED + 9999)
ts_idx = np.sort(rng.choice(adata.n_obs, size=500, replace=False))
X_ts = adata[ts_idx].X
if sp.issparse(X_ts) or hasattr(X_ts, "toarray"):
    X_ts = X_ts.toarray()
X_ts = np.asarray(X_ts, dtype=np.float32)
var_ens_ts = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
adata.file.close()
var_idx_ts, var_tok_ts, var_med_ts = tokenizer.make_var_mapping(var_ens_ts.tolist())

ts_activations = []
for ci in range(X_ts.shape[0]):
    cell = tokenizer.tokenize_cell(X_ts[ci], var_idx_ts, var_tok_ts, var_med_ts, max_len=2048)
    if cell is None:
        continue
    input_ids = torch.from_numpy(cell.token_ids[None, :])
    _, hidden = xt.forward_with_hidden_states(input_ids)
    ts_activations.append(hidden[PROBE_LAYER + 1][0].numpy())
    del hidden
    if DEVICE == "mps" and ci % 50 == 0:
        try: torch.mps.empty_cache()
        except: pass
    if (ci + 1) % 100 == 0:
        print(f"    {ci+1}/500 cells")
del xt
if DEVICE == "mps":
    try: torch.mps.empty_cache()
    except: pass

ts_flat = np.concatenate(ts_activations, axis=0).astype(np.float32)
n_ts = ts_flat.shape[0]
print(f"  TS positions: {n_ts:,}")

# Pool: 500K from each source
n_pool_each = 500000
rng_pool = np.random.default_rng(SEED + 8888)
k562_sample = k562_acts[rng_pool.choice(n_k562, size=min(n_pool_each, n_k562), replace=False)]
ts_sample = ts_flat[rng_pool.choice(n_ts, size=min(n_pool_each, n_ts), replace=False)]
pooled = np.concatenate([k562_sample, ts_sample], axis=0).astype(np.float32)
print(f"  pooled: {pooled.shape[0]:,} ({k562_sample.shape[0]:,} K562 + {ts_sample.shape[0]:,} TS)")
del k562_acts, ts_flat, k562_sample, ts_sample

# Save pooled activations temporarily
pooled_path = OUT / "pooled_l5_activations.npy"
np.save(pooled_path, pooled)
del pooled

# Step 2: Train multi-tissue SAE
print("\n[2] Training multi-tissue SAE at L5...")
results = train_sae(
    activations_path=pooled_path,
    output_dir=OUT / "sae_multitissue",
    d_model=HIDDEN,
    d_sae=4 * HIDDEN,
    k=32, lr=3e-4, batch_size=4096, n_epochs=4,
    n_train=1_000_000, n_eval=100_000,
    device=DEVICE,
)
print(f"  var_explained={results['variance_explained']:.4f}  "
      f"alive={results['n_alive']}  dead={results['n_dead']}")

# Step 3: Re-run perturbation specificity test with multi-tissue SAE
print("\n[3] Re-running perturbation specificity with multi-tissue SAE...")
ckpt = torch.load(OUT / "sae_multitissue/sae_final.pt", map_location="cpu", weights_only=False)
sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
sae.W_enc.weight.data = ckpt["W_enc_weight"]
sae.W_enc.bias.data = ckpt["W_enc_bias"]
sae.W_dec.weight.data = ckpt["W_dec_weight"]
sae.eval()
mu_mt = ckpt["mu"]
d_sae = sae.d_sae

# Annotate features (quick: top-20 genes only, skip full enrichment for speed)
k562_acts_phase0 = np.load(PHASE0 / f"layer_{PROBE_LAYER:02d}_activations.npy")
with open(PHASE0 / f"layer_{PROBE_LAYER:02d}_gene_names.json") as f:
    gene_names = [g.upper() for g in json.load(f)]
unique_genes = sorted(set(gene_names))
gene_to_code = {g: i for i, g in enumerate(unique_genes)}
pos_codes = np.array([gene_to_code[g] for g in gene_names], dtype=np.int32)
n_unique = len(unique_genes)

# Incremental feature top-20 computation
feat_gene_sum = np.zeros((d_sae, n_unique), dtype=np.float64)
feat_gene_cnt = np.zeros((d_sae, n_unique), dtype=np.int32)
feat_active_total = np.zeros(d_sae, dtype=np.int64)
n_pos = k562_acts_phase0.shape[0]
chunk = 20000
for ci in range(0, n_pos, chunk):
    end = min(ci + chunk, n_pos)
    with torch.no_grad():
        h = sae.encode(torch.from_numpy(k562_acts_phase0[ci:end].astype(np.float32)), mu_mt).numpy()
    codes_chunk = pos_codes[ci:end]
    for fi in range(d_sae):
        col = h[:, fi]
        active = col > 0
        n_act = int(active.sum())
        if n_act == 0: continue
        feat_active_total[fi] += n_act
        ac = codes_chunk[active]
        av = col[active].astype(np.float64)
        np.add.at(feat_gene_sum[fi], ac, av)
        np.add.at(feat_gene_cnt[fi], ac, 1)
    del h

mt_top20 = {}
for fi in range(d_sae):
    if feat_active_total[fi] == 0: continue
    cnt = feat_gene_cnt[fi]; s = feat_gene_sum[fi]
    nz = cnt > 0
    mean_pg = np.zeros(n_unique, dtype=np.float32)
    mean_pg[nz] = s[nz] / cnt[nz]
    top_idx = np.argsort(-mean_pg)[:20]
    mt_top20[fi] = set(unique_genes[i] for i in top_idx if mean_pg[i] > 0)
del feat_gene_sum, feat_gene_cnt

# Perturbation test (same approach as Phase 8t)
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_targets_map = {}
for _, r in trrust.iterrows():
    trrust_targets_map.setdefault(r["tf"], set()).add(r["target"])
trrust_tfs = set(trrust_targets_map.keys())

# Encode control positions for comparison
n_ctrl_use = min(100000, n_pos)
with torch.no_grad():
    h_ctrl_list = []
    for i in range(0, n_ctrl_use, 20000):
        xb = torch.from_numpy(k562_acts_phase0[i:i + 20000].astype(np.float32))
        h_ctrl_list.append(sae.encode(xb, mu_mt).numpy())
    h_ctrl = np.concatenate(h_ctrl_list, axis=0)
del k562_acts_phase0

# Load perturbed cells and test
xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
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
    var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(var_ens)

    rng_p = np.random.default_rng(SEED)
    k562_cells = np.where(k562_mask)[0]
    k562_codes = pg_codes[k562_mask]

    pert_targets = []
    for code in np.unique(k562_codes):
        cat = pg_cats[code]
        if "non-targeting" in cat.lower() or "nontargeting" in cat.lower(): continue
        sel = k562_cells[k562_codes == code]
        if len(sel) < 20: continue
        pert_targets.append((cat, code, sel))

    tf_targets = [(c, code, sel) for c, code, sel in pert_targets if c.upper() in trrust_tfs][:48]
    non_tf = [(c, code, sel) for c, code, sel in pert_targets if c.upper() not in trrust_tfs][:52]
    selected = tf_targets + non_tf
    print(f"  targets: {len(tf_targets)} TFs + {len(non_tf)} non-TFs")

    pert_results = []
    for pi, (cat, code, sel) in enumerate(selected):
        is_tf = cat.upper() in trrust_tfs
        cell_idx = rng_p.choice(sel, size=min(20, len(sel)), replace=False)
        pert_acts = []
        for ci in cell_idx:
            X_cell = f["X"][int(ci), :].astype(np.float32)
            cell = tokenizer.tokenize_cell(X_cell, var_idx_full, var_tok_full, var_med_full, max_len=2048)
            if cell is None: continue
            ids = torch.from_numpy(cell.token_ids[None, :])
            _, hidden = xt.forward_with_hidden_states(ids)
            pert_acts.append(hidden[PROBE_LAYER + 1][0].numpy())
            del hidden
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass
        if not pert_acts: continue
        pert_flat = np.concatenate(pert_acts, axis=0).astype(np.float32)
        with torch.no_grad():
            h_pert = sae.encode(torch.from_numpy(pert_flat), mu_mt).numpy()
        responding = []
        for fi in range(d_sae):
            t_v = h_pert[:, fi]; o_v = h_ctrl[:min(10000, h_ctrl.shape[0]), fi]
            if t_v.std() < 1e-8 and o_v.std() < 1e-8: continue
            try: _, p = sp_stats.mannwhitneyu(t_v, o_v, alternative="two-sided")
            except: continue
            if p < 0.05 and abs(float(t_v.mean() - o_v.mean())) > 0.5:
                responding.append(fi)
        is_specific = False
        if is_tf and responding:
            known = trrust_targets_map.get(cat.upper(), set())
            for fi in responding:
                if len(mt_top20.get(fi, set()) & known) >= 2:
                    is_specific = True; break
        pert_results.append({"target": cat, "is_tf": is_tf, "n_responding": len(responding),
                            "detected": len(responding) > 0, "is_specific": is_specific})
        if (pi + 1) % 20 == 0:
            print(f"    {pi+1}/{len(selected)} processed")

df = pd.DataFrame(pert_results)
df.to_csv(OUT / "perturbation_response_multitissue.csv", index=False)
tf_df = df[df["is_tf"]]
n_spec = int(tf_df["is_specific"].sum())
n_tfs = len(tf_df)
n_det = int(df["detected"].sum())
print(f"\n  Multi-tissue detection: {n_det}/{len(df)} ({n_det/max(len(df),1):.1%})")
print(f"  Multi-tissue TF specificity: {n_spec}/{n_tfs} ({n_spec/max(n_tfs,1):.1%})")

# Compare to K562-only result
k562_only = json.loads((PROJ / "runs/sae-atlas-217M/outputs/phase8_true/summary.json").read_text())
print(f"  K562-only TF specificity: {k562_only['n_tf_specific']}/{k562_only['n_tfs']} "
      f"({k562_only['tf_specificity_rate']:.1%})")
print(f"  Δ multi-tissue vs K562-only: {n_spec/max(n_tfs,1) - k562_only['tf_specificity_rate']:+.1%}")

with open(OUT / "summary.json", "w") as f:
    json.dump({
        "multitissue_detection_rate": n_det / max(len(df), 1),
        "multitissue_tf_specificity": n_spec / max(n_tfs, 1),
        "multitissue_n_specific": n_spec,
        "multitissue_n_tfs": n_tfs,
        "k562_only_tf_specificity": k562_only["tf_specificity_rate"],
        "delta_pp": n_spec / max(n_tfs, 1) - k562_only["tf_specificity_rate"],
        "sae_var_explained": results["variance_explained"],
        "sae_alive": results["n_alive"],
        "sae_dead": results["n_dead"],
    }, f, indent=2)

# Clean up pooled activations
pooled_path.unlink(missing_ok=True)
print(f"\nPhase 9 COMPLETE — outputs: {OUT}")
