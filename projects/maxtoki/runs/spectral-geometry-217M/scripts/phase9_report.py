"""Phase 9 negative controls + final summary.

9d: 591-equivalent GO BP enrichment test on SV2 (expect 0/N significant
    after BH correction).
9e: ER-AUROC correlation and partial correlation controlling for layer depth.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
PHASE0 = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
PHASE1 = PROJ / "runs/spectral-geometry-217M/outputs/phase1"
PHASE5 = PROJ / "runs/spectral-geometry-217M/outputs/phase5"
OUT = PROJ / "runs/spectral-geometry-217M/outputs/phase9"
OUT.mkdir(parents=True, exist_ok=True)

GO_BP_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"

SEED = 42
N_PERM = 200
TOPK = 52

gene_features = pd.read_csv(PHASE0 / "gene_features.csv")
gene_counts = np.load(PHASE0 / "gene_counts.npy")
nonzero_mask = gene_counts > 0
nonzero_indices = np.where(nonzero_mask)[0]
gene_symbols_all = gene_features["symbol"].tolist()
gene_symbols_nonzero = [gene_symbols_all[i].upper() for i in nonzero_indices]
sym_to_nz = {s: i for i, s in enumerate(gene_symbols_nonzero)}

sv = np.load(PHASE1 / "gene_svd_coords.npy")   # (L, n_nz, TOPK_SV)
n_layers, n_nz, topk_sv = sv.shape
print(f"  sv shape: {sv.shape}")

# ========================================================================
# 9d — GO BP terms on SV2  (expect null)
# ========================================================================
print("\n[Phase 9d] GO BP terms on SV2 — expect 0 significant after BH")
with open(GO_BP_JSON) as f:
    go_bp = json.load(f)
print(f"  GO BP terms available: {len(go_bp)}")


def single_label_copole(sv_l_k, pos_mask, top_k=TOPK, n_perm=N_PERM, seed=SEED):
    n = sv_l_k.shape[0]
    order = np.argsort(sv_l_k)
    pole = np.zeros(n, dtype=np.int8)
    pole[order[:top_k]] = 1; pole[order[-top_k:]] = 2
    n_pos = int(pos_mask.sum())
    if n_pos < 5:
        return None
    obs = ((pole > 0) & pos_mask).sum() / n_pos
    rng_perm = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for t in range(n_perm):
        shuf = rng_perm.permutation(pos_mask)
        null[t] = (pole[shuf] > 0).sum() / n_pos
    null_mean = float(null.mean()); null_std = float(null.std() + 1e-12)
    return {
        "observed_rate": float(obs), "null_mean": null_mean, "null_std": null_std,
        "z": float((obs - null_mean) / null_std),
        "p_empirical": float((1 + (null >= obs).sum()) / (n_perm + 1)),
        "n_positive": n_pos,
    }


# Pick one layer (early) for the SV2 test, matching the paper convention
test_layer = min(3, n_layers - 1)   # L3 or last
rows9d = []
for term_name, gene_list in go_bp.items():
    pos_mask = np.array([s in {g.upper() for g in gene_list} for s in gene_symbols_nonzero], dtype=bool)
    if pos_mask.sum() < 5:
        continue
    r = single_label_copole(sv[test_layer, :, 1], pos_mask, seed=SEED + hash(term_name) % 100000)
    if r is None:
        continue
    r.update({"term": term_name, "layer": test_layer})
    rows9d.append(r)
df9d = pd.DataFrame(rows9d)
if len(df9d) > 0:
    _, df9d["p_bh"], _, _ = multipletests(df9d["p_empirical"], method="fdr_bh")
df9d.to_csv(OUT / "go_bp_sv2_copole.csv", index=False)
sig = int((df9d["p_bh"] < 0.05).sum()) if len(df9d) else 0
print(f"  GO BP terms tested on SV2@L{test_layer}: {len(df9d)}  BH-significant: {sig}/{len(df9d)}")

# ========================================================================
# 9e — ER-AUROC partial correlation (controlling for layer depth)
# ========================================================================
print("\n[Phase 9e] ER ↔ classifier AUROC partial correlation")
p1_df = pd.read_csv(PHASE1 / "per_layer_metrics.csv")
# Use Phase 5's TF-vs-target joint AUROC as the classifier AUROC per layer
try:
    p5_class = pd.read_csv(PHASE5 / "tf_vs_target_classifier.csv")
    merged = pd.merge(p1_df, p5_class, on="layer", how="inner")
    raw_rho, raw_p = spearmanr(merged["effective_rank"], merged["auc_joint_SV2_SV7"])
    # Partial correlation controlling for layer: residualize each variable on layer, then correlate
    lr_er = LinearRegression().fit(merged[["layer"]], merged["effective_rank"])
    lr_au = LinearRegression().fit(merged[["layer"]], merged["auc_joint_SV2_SV7"])
    res_er = merged["effective_rank"] - lr_er.predict(merged[["layer"]])
    res_au = merged["auc_joint_SV2_SV7"] - lr_au.predict(merged[["layer"]])
    part_rho, part_p = spearmanr(res_er, res_au)
    print(f"  raw     ER ~ AUROC: ρ={raw_rho:+.3f}  p={raw_p:.2e}")
    print(f"  partial ER ~ AUROC | layer: ρ={part_rho:+.3f}  p={part_p:.2e}")
    with open(OUT / "er_auroc_partial_correlation.json", "w") as f:
        json.dump({
            "raw_spearman_rho": float(raw_rho), "raw_spearman_p": float(raw_p),
            "partial_spearman_rho": float(part_rho), "partial_spearman_p": float(part_p),
        }, f, indent=2)
except FileNotFoundError:
    print("  phase5 tf_vs_target_classifier.csv not available — skipping 9e")

# ========================================================================
# Final summary report
# ========================================================================
print("\n[Final] Building run report...")
lines = ["# Residual-stream spectral geometry — MaxToki-217M run report\n"]

p0_cfg = json.loads((PHASE0 / "run_config.json").read_text())
p1_cfg = json.loads((PHASE1 / "run_config.json").read_text())
lines.append("## Phase 0 — extraction\n")
for k in ["model", "dataset", "n_cells", "n_hvg", "max_len", "device",
          "n_layer_states", "hidden_size", "mean_seq_len", "mean_fwd_seconds",
          "total_phase_seconds", "n_genes_with_observations"]:
    if k in p0_cfg:
        lines.append(f"- {k}: `{p0_cfg[k]}`")

lines.append("\n## Phase 1 — global spectral metrics\n")
p1_metrics = pd.read_csv(PHASE1 / "per_layer_metrics.csv")
lines.append("| Layer | Effective rank | Participation ratio | SV1 var frac | TwoNN |")
lines.append("|---|---|---|---|---|")
for _, r in p1_metrics.iterrows():
    lines.append(f"| {int(r['layer'])} | {r['effective_rank']:.2f} | {r['participation_ratio']:.2f} | "
                 f"{r['sv1_variance_fraction']:.3f} | {r['twonn_intrinsic_dim']:.2f} |")
p1_dc = p1_cfg["depth_correlations"]
lines.append(f"\n- depth vs effective rank: ρ={p1_dc['effective_rank']['rho']:+.3f} p={p1_dc['effective_rank']['p']:.2e}")
lines.append(f"- depth vs SV1 variance fraction: ρ={p1_dc['sv1_variance_fraction']['rho']:+.3f} p={p1_dc['sv1_variance_fraction']['p']:.2e}")

ns = json.loads((PHASE1 / "null_feature_shuffle.json").read_text())
lines.append(f"- feature-shuffle null on final layer: ER observed {ns['observed_effective_rank_final_layer']:.2f}, shuffled {ns['feature_shuffle_effective_rank_final_layer']:.2f} (ratio {ns['ratio_shuffle_vs_observed']:.2f}×)")

# Phase 3 headlines
p3 = PHASE0.parent / "phase3"
f3 = p3 / "sv1_go_cc_copole.csv"
if f3.exists():
    lines.append("\n## Phase 3 — SV1 vs GO Cellular Component\n")
    df3 = pd.read_csv(f3)
    best = df3.loc[df3.groupby("term")["z"].idxmax()]
    lines.append("| GO CC term | Best layer | z | observed | null mean | p_bh |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in best.iterrows():
        lines.append(f"| {r['term']} | L{int(r['layer'])} | {r['z']:+.2f} | "
                     f"{r['observed_rate']:.3f} | {r['null_mean']:.3f} | {r['p_bh']:.2e} |")

# Phase 4 headlines
p4 = PHASE0.parent / "phase4" / "ppi_copole_per_axis_per_layer.csv"
if p4.exists():
    lines.append("\n## Phase 4 — SV2–SV4 vs STRING PPI (pairs_700)\n")
    df4 = pd.read_csv(p4)
    sum4 = df4[df4["pair_set"] == "pairs_700"].groupby("axis").agg(
        mean_z=("z", "mean"), n_sig=("p_bh", lambda x: (x < 0.05).sum())
    ).reset_index().sort_values("mean_z", ascending=False)
    lines.append("| Axis | Mean z across layers | Significant layers |")
    lines.append("|---|---|---|")
    for _, r in sum4.iterrows():
        lines.append(f"| {r['axis']} | {r['mean_z']:+.2f} | {int(r['n_sig'])} / {n_layers} |")
    cg_path = PHASE0.parent / "phase4" / "confidence_gradient.json"
    if cg_path.exists():
        cg = json.loads(cg_path.read_text())
        lines.append(f"- SV2 2-point gradient: z_mean(STRING≥700)={cg['z_mean_700_SV2']:+.2f}, z_mean(STRING≥900)={cg['z_mean_900_SV2']:+.2f}")

# Phase 5 headlines
p5c = PHASE5 / "tf_vs_target_classifier.csv"
p5e = PHASE5 / "edge_level_auroc.csv"
if p5c.exists():
    lines.append("\n## Phase 5 — TF vs target classifier + edge-level AUROC\n")
    df5c = pd.read_csv(p5c)
    lines.append(f"- Joint SV2-SV7 AUROC mean across {n_layers} layers: {df5c['auc_joint_SV2_SV7'].mean():.4f}  max: {df5c['auc_joint_SV2_SV7'].max():.4f} at L{int(df5c['layer'].iloc[df5c['auc_joint_SV2_SV7'].idxmax()])}")
    lines.append(f"- Permutation-null p min: {df5c['perm_p'].min():.4f}  max: {df5c['perm_p'].max():.4f}")
if p5e.exists():
    df5e = pd.read_csv(p5e)
    lines.append(f"- SV5-SV7 edge-level AUROC per layer: {df5e['auc_SV5_SV7_edge'].tolist()}")
    rho_e, p_e = spearmanr(df5e["layer"], df5e["auc_SV5_SV7_edge"])
    lines.append(f"- Edge-AUROC vs depth: ρ={rho_e:+.3f} p={p_e:.2e}")

# Phase 6 headlines
p6 = PHASE0.parent / "phase6" / "cell_type_clustering_auroc.csv"
if p6.exists():
    lines.append("\n## Phase 6 — Cell-type marker clustering\n")
    df6 = pd.read_csv(p6)
    lines.append(f"- Mean AUROC (within-type vs across-type) across layers: {df6['auc_same_vs_diff_type'].mean():.4f}  max: {df6['auc_same_vs_diff_type'].max():.4f} at L{int(df6['layer'].iloc[df6['auc_same_vs_diff_type'].idxmax()])}")

# Phase 7 headlines
p7a = PHASE0.parent / "phase7" / "b_cell_attractor_trajectory.csv"
if p7a.exists():
    lines.append("\n## Phase 7 — B-cell attractor dynamics\n")
    df7 = pd.read_csv(p7a)
    dc_path = PHASE0.parent / "phase7" / "depth_correlations.json"
    if dc_path.exists():
        dc = json.loads(dc_path.read_text())
        for k, v in dc.items():
            lines.append(f"- {k}: ρ={v['rho']:+.3f} p={v['p']:.2e}")
    # Key trajectories
    if "rank_PAX5" in df7:
        lines.append(f"- PAX5 rank L0→L{n_layers-1}: {int(df7['rank_PAX5'].iloc[0])} → {int(df7['rank_PAX5'].iloc[-1])}")
    if "rank_BATF" in df7:
        lines.append(f"- BATF rank L0→L{n_layers-1}: {int(df7['rank_BATF'].iloc[0])} → {int(df7['rank_BATF'].iloc[-1])}")
    if "rank_BACH2" in df7:
        lines.append(f"- BACH2 rank L0→L{n_layers-1}: {int(df7['rank_BACH2'].iloc[0])} → {int(df7['rank_BACH2'].iloc[-1])}")
    if "gc_plasma_angle_deg" in df7:
        lines.append(f"- GC-plasma angle L0→L{n_layers-1}: {df7['gc_plasma_angle_deg'].iloc[0]:.1f}° → {df7['gc_plasma_angle_deg'].iloc[-1]:.1f}°")

# Phase 9 headlines
lines.append("\n## Phase 9 — Negative-control findings\n")
lines.append(f"- 9d: GO BP SV2 BH-significant: {sig}/{len(df9d)} (expect near 0)")
p9e = OUT / "er_auroc_partial_correlation.json"
if p9e.exists():
    r9e = json.loads(p9e.read_text())
    lines.append(f"- 9e: raw ER↔AUROC ρ={r9e['raw_spearman_rho']:+.3f} (p={r9e['raw_spearman_p']:.2e})")
    lines.append(f"- 9e: partial ER↔AUROC|layer ρ={r9e['partial_spearman_rho']:+.3f} (p={r9e['partial_spearman_p']:.2e})")

(PROJ / "runs/spectral-geometry-217M/outputs/run_report.md").write_text("\n".join(lines))
print(f"  wrote: runs/spectral-geometry-217M/outputs/run_report.md")
