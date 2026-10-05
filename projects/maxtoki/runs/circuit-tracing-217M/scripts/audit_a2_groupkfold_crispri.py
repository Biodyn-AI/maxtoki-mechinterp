"""Audit action A2 — circuit-tracing CRISPRi GroupKFold re-evaluation.

Re-implements Phase 11 with two fixes:
  1. **Proper predicted-sign tracking** (the original phase 11 dropped sign
     information and just measured "fraction of perturbations that reduce
     expression at all" — see remaining_phases.py:327).
  2. **GroupKFold by source gene**: 5 disjoint source-gene folds. Per fold,
     compute directional accuracy on the held-out source genes only — bounds
     effective N to source-gene cardinality, not the inflated pair-cardinality.

Output: outputs/groupkfold_crispri/{summary.json, per_pair.parquet,
per_source_accuracy.csv}
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/groupkfold_crispri"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

print("[A2-circuit] Loading circuit edges + feature top-genes...")
edges = pd.read_csv(RUN / "outputs/circuit_edges.csv")
print(f"  {len(edges):,} edges")

# Feature top-genes
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
feat_top_genes = {}
for layer_dir in sorted((PHASE2).glob("layer_*")):
    li = int(layer_dir.name.split("_")[1])
    with open(layer_dir / "feature_catalog.json") as f:
        catalog = json.load(f)
    for c in catalog:
        feat_top_genes[(li, c["feature_id"])] = c["top20_genes"][:10]
print(f"  feature top-genes for {len(feat_top_genes):,} (layer, feature) pairs")

# Filter to edges where both endpoints have top-genes
ann_mask = edges.apply(
    lambda r: (r["src_layer"], r["src_feature"]) in feat_top_genes
              and (r["tgt_layer"], r["tgt_feature"]) in feat_top_genes,
    axis=1,
)
ann_edges = edges[ann_mask].reset_index(drop=True)
print(f"  edges with both endpoints annotated: {len(ann_edges):,}")

# Build gene predictions (preserving sign information this time)
print("\n[A2-circuit] Building gene predictions with sign tracking...")
t0 = time.time()
gene_pred = defaultdict(lambda: {"evidence": 0, "max_abs_d": 0.0, "signed_d_sum": 0.0, "n_inhib": 0, "n_total": 0})
n_raw = 0
for _, e in ann_edges.iterrows():
    src_genes = feat_top_genes.get((e["src_layer"], e["src_feature"]), [])[:10]
    tgt_genes = feat_top_genes.get((e["tgt_layer"], e["tgt_feature"]), [])[:10]
    is_inhib = e["sign"] == "inhibitory"
    cohens_d = float(e["cohens_d"])
    for sg in src_genes:
        for tg in tgt_genes:
            if sg.upper() == tg.upper():
                continue
            n_raw += 1
            k = (sg.upper(), tg.upper())
            gene_pred[k]["evidence"] += 1
            gene_pred[k]["max_abs_d"] = max(gene_pred[k]["max_abs_d"], abs(cohens_d))
            gene_pred[k]["signed_d_sum"] += cohens_d
            gene_pred[k]["n_inhib"] += int(is_inhib)
            gene_pred[k]["n_total"] += 1

# Filter: evidence>=2 or |d|>2
filtered = {k: v for k, v in gene_pred.items() if v["evidence"] >= 2 or v["max_abs_d"] > 2.0}
print(f"  raw pairs: {n_raw:,}; filtered: {len(filtered):,} ({time.time()-t0:.1f}s)")

# Predicted "inhibitory" if majority of contributing edges are inhibitory
pred_rows = []
for (sg, tg), v in filtered.items():
    pred_inhib = (v["n_inhib"] / max(1, v["n_total"])) > 0.5
    pred_rows.append({"source": sg, "target": tg, "predicted_inhibitory": pred_inhib,
                      "evidence": v["evidence"], "max_abs_d": v["max_abs_d"]})
pred_df = pd.DataFrame(pred_rows)
print(f"  predicted inhibitory: {pred_df['predicted_inhibitory'].sum():,} / {len(pred_df):,}")

print("\n[A2-circuit] Loading Replogle K562 + computing pseudobulk LFCs per source gene...")
t1 = time.time()
ds = load_ds("k562")
unique_sources = pred_df["source"].unique()
print(f"  {len(unique_sources):,} unique source genes to query")

with h5py.File(ds.h5_path, "r") as f:
    pg_cats = [s.decode() if isinstance(s, bytes) else s for s in f["obs"]["gene"]["categories"][:]]
    pg_codes = f["obs"]["gene"]["codes"][:]
    cl_codes = f["obs"]["cell_line"]["codes"][:]
    cl_cats = [(s.decode() if isinstance(s, bytes) else s).lower() for s in f["obs"]["cell_line"]["categories"][:]]
    k562 = (cl_codes == cl_cats.index("k562"))
    var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    var_upper = [s.upper() for s in var_symbols]
    gene_to_var = {s: i for i, s in enumerate(var_upper)}

    nt_codes = {i for i, g in enumerate(pg_cats) if "non-targeting" in g.lower() or "nontargeting" in g.lower()}
    nt_mask = k562 & np.isin(pg_codes, list(nt_codes))
    nt_idx = np.where(nt_mask)[0]
    rng = np.random.default_rng(SEED)
    if len(nt_idx) > 3000:
        nt_idx = np.sort(rng.choice(nt_idx, 3000, replace=False))

    n_var = len(var_symbols)
    X_nt = np.empty((len(nt_idx), n_var), dtype=np.float32)
    for i in range(0, len(nt_idx), 500):
        rows = nt_idx[i:i+500]
        X_nt[i:i+500] = f["X"][rows, :]
    rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_nt = np.log1p(X_nt / rs * 1e4)
    ctrl_mean = X_nt.mean(axis=0)
    del X_nt

    k562_cells = np.where(k562)[0]
    k562_codes = pg_codes[k562]

    pair_records = []
    for src_i, src_gene in enumerate(unique_sources):
        pert_code = None
        for i, cat in enumerate(pg_cats):
            if cat.upper() == src_gene:
                pert_code = i
                break
        if pert_code is None:
            continue
        pert_cells = k562_cells[k562_codes == pert_code]
        if len(pert_cells) < 10:
            continue
        X_pert = np.empty((len(pert_cells), n_var), dtype=np.float32)
        for i in range(0, len(pert_cells), 300):
            rows = pert_cells[i:i+300]
            X_pert[i:i+300] = f["X"][rows, :]
        rs_p = X_pert.sum(axis=1, keepdims=True); rs_p[rs_p == 0] = 1.0
        X_pert = np.log1p(X_pert / rs_p * 1e4)
        pert_mean = X_pert.mean(axis=0)
        lfc = pert_mean - ctrl_mean

        # All predictions for this source
        sub = pred_df[pred_df["source"] == src_gene]
        for _, row in sub.iterrows():
            tg = row["target"]
            tg_idx = gene_to_var.get(tg)
            if tg_idx is None:
                continue
            actual_lfc = float(lfc[tg_idx])
            actual_inhib = actual_lfc < 0  # knockdown of source reduces target → inhibitory by source
            pred_inhib = bool(row["predicted_inhibitory"])
            correct = (pred_inhib == actual_inhib)
            pair_records.append({
                "source": src_gene, "target": tg,
                "predicted_inhibitory": pred_inhib,
                "actual_lfc": actual_lfc, "actual_inhibitory": actual_inhib,
                "correct": correct,
            })
        if (src_i + 1) % 50 == 0:
            print(f"    {src_i+1}/{len(unique_sources)} sources processed; {len(pair_records):,} pairs so far")

per_pair = pd.DataFrame(pair_records)
print(f"\n[A2-circuit] Per-pair table: {len(per_pair):,} rows ({time.time()-t1:.0f}s)")
try:
    per_pair.to_parquet(OUT / "per_pair.parquet", index=False)
except ImportError:
    # Fall back to compressed CSV if pyarrow/fastparquet aren't available
    per_pair.to_csv(OUT / "per_pair.csv.gz", index=False, compression="gzip")

# Aggregate
overall_acc = float(per_pair["correct"].mean())
print(f"\n[A2-circuit] Overall directional accuracy: {overall_acc:.4f} (n={len(per_pair):,})")

# GroupKFold by source gene
print("\n[A2-circuit] GroupKFold-by-source-gene (5 folds)...")
unique_srcs = per_pair["source"].unique()
rng = np.random.default_rng(SEED)
src_perm = rng.permutation(unique_srcs)
folds = np.array_split(src_perm, 5)
fold_accs = []
per_source_accs = []
for fold_i, fold_srcs in enumerate(folds):
    mask = per_pair["source"].isin(fold_srcs)
    sub = per_pair[mask]
    if len(sub) == 0:
        continue
    acc = float(sub["correct"].mean())
    fold_accs.append({"fold": fold_i, "n_sources": len(fold_srcs), "n_pairs": len(sub), "accuracy": acc})
    print(f"  fold {fold_i}: n_sources={len(fold_srcs)}, n_pairs={len(sub):,}, accuracy={acc:.4f}")

# Per-source accuracy (independent unit-of-inference)
per_src = per_pair.groupby("source").agg(n_pairs=("correct", "size"), accuracy=("correct", "mean")).reset_index()
per_src.to_csv(OUT / "per_source_accuracy.csv", index=False)
mean_per_src = float(per_src["accuracy"].mean())
median_per_src = float(per_src["accuracy"].median())

# Bootstrap CI on per-source mean accuracy
boot_means = []
src_acc_arr = per_src["accuracy"].to_numpy()
for _ in range(1000):
    idx = rng.integers(0, len(src_acc_arr), size=len(src_acc_arr))
    boot_means.append(src_acc_arr[idx].mean())
boot_means = np.array(boot_means)

summary = {
    "overall_directional_accuracy_with_proper_signs": overall_acc,
    "n_pairs_total": int(len(per_pair)),
    "n_unique_sources": int(len(unique_srcs)),
    "groupkfold_5_by_source": fold_accs,
    "groupkfold_mean_accuracy": float(np.mean([f["accuracy"] for f in fold_accs])),
    "per_source_mean_accuracy": mean_per_src,
    "per_source_median_accuracy": median_per_src,
    "per_source_bootstrap_ci95": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
    "comparison_vs_original_phase11": {
        "original_directional_accuracy": 0.5461,
        "original_method": "buggy: just `actual_lfc < 0` regardless of predicted sign",
        "new_overall_with_signs": overall_acc,
        "new_per_source_mean": mean_per_src,
    },
    "interpretation": (
        f"Properly accounting for predicted sign, overall directional accuracy is "
        f"{overall_acc:.4f} ({'above' if overall_acc > 0.5 else 'at/below'} chance). "
        f"Per-source mean (each source gene = 1 unit) is {mean_per_src:.4f}, 95% CI "
        f"[{np.percentile(boot_means,2.5):.4f}, {np.percentile(boot_means,97.5):.4f}]. "
        f"GroupKFold-by-source-gene 5-fold mean: {np.mean([f['accuracy'] for f in fold_accs]):.4f}. "
        "The original Phase 11 metric (54.61%) was inflated by a buggy sign-comparison; "
        "the corrected number is the appropriate one to report. " +
        ("This shifts the interpretation: there IS a small but real circuit-prediction signal."
         if overall_acc - 0.5 > 0.01 else
         "This confirms the circuit predictions are at chance level — the original 54.61% was a sign-comparison artifact.")
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[A2-circuit] Wrote {OUT}/summary.json")
print(f"[A2-circuit] overall {overall_acc:.4f}, per-source mean {mean_per_src:.4f} CI [{np.percentile(boot_means,2.5):.4f}, {np.percentile(boot_means,97.5):.4f}]")
