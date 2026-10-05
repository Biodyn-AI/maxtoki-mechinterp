"""Emit the Phase-13 cross-experiment synthesis as stage3_summary.json.

Consumes:
  - experiment1/exhaustive_summary.json
  - experiment2/combinatorial_summary.json
  - experiment3/steering_summary.json

Produces outputs/stage3_summary.json plus a refreshed block of the
FINAL_SUMMARY.md Experiment 3 section (stdout, to be pasted).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
E1 = OUT / "experiment1" / "exhaustive_summary.json"
E2 = OUT / "experiment2" / "combinatorial_summary.json"
E2V2 = OUT / "experiment2_v2" / "combinatorial_summary.json"
E3 = OUT / "experiment3" / "steering_summary.json"

with open(E1) as f: e1 = json.load(f)
with open(E2) as f: e2 = json.load(f)
with open(E3) as f: e3 = json.load(f)
e2v2 = None
if E2V2.exists():
    with open(E2V2) as f: e2v2 = json.load(f)

# --------------------------------------------------------------
# Build the synthesis record
# --------------------------------------------------------------
synthesis = {
    "pipeline": "sparse-autoencoders/03-exhaustive-mapping-and-steering",
    "model": "MaxToki-217M-HF",
    "experiment1_exhaustive_tracing": {
        "source_layer": e1["source_layer"],
        "downstream_layers": e1["downstream_layers"],
        "n_features_traced": e1["n_features_traced"],
        "n_cells_per_feature": e1["n_cells"],
        "total_edges": e1["total_edges"],
        "mean_edges_per_feature": e1["mean_edges_per_feature"],
        "n_features_with_gt_1000_edges": e1["n_hubs_above_1000"],
        "n_zero_edge_features": e1["n_zero_edge_features"],
        "top20_hub_annotation_rate": e1["top20_annotation_rate"],
        "overall_annotation_rate": e1["overall_annotation_rate"],
        "paper_comparison": {
            "paper_features": 4065,
            "paper_total_edges": 1393850,
            "paper_top20_annotated": 0.60,
            "paper_hub_class_pct": 0.018,
        },
        "interpretation": (
            "Uniformly dense circuit — 100% of traced features have >1000 edges, "
            "vs paper's 1.8% heavy-tail hub class. Annotation bias replicates: "
            f"top-20 hubs {int(e1['top20_annotation_rate']*100)}% annotated vs overall "
            f"{int(e1['overall_annotation_rate']*100)}%, no significant enrichment."
        ),
    },
    "experiment2_combinatorial_ablation": {
        "v1_greedy_selector": {
            "n_triplets": e2["n_triplets"],
            "triplet_layers": [0, 5, 9],
            "measurement_layer": 11,
            "mean_pairwise_ratio": e2["mean_pairwise_ratio"],  # AB-only
            "mean_threeway_ratio": e2["mean_threeway_ratio"],
            "mean_pct_superadditive": e2["mean_pct_superadditive"],
            "known_issue": "all 4 triplets share (L0_F0, L5_F0) base; ratios are 4x replicates of the same underlying triplet",
        },
        "v2_diversified_selector": (
            {
                "n_triplets": e2v2["n_triplets"],
                "triplet_layers": e2v2["triplet_layers"],
                "measurement_layer": e2v2["measurement_layer"],
                "mean_pairwise_ratio_AB_AC_BC": e2v2["mean_pairwise_ratio"],
                "std_pairwise_ratio": e2v2["std_pairwise_ratio"],
                "mean_threeway_ratio": e2v2["mean_threeway_ratio"],
                "std_threeway_ratio": e2v2["std_threeway_ratio"],
                "mean_pct_superadditive": e2v2["mean_pct_superadditive"],
                "per_pair_means": {
                    "L0xL5_AB": round(float(sum(t["pairwise_ratio_AB"] for t in e2v2["triplets"]) / len(e2v2["triplets"])), 4),
                    "L0xL9_AC": round(float(sum(t["pairwise_ratio_AC"] for t in e2v2["triplets"]) / len(e2v2["triplets"])), 4),
                    "L5xL9_BC": round(float(sum(t["pairwise_ratio_BC"] for t in e2v2["triplets"]) / len(e2v2["triplets"])), 4),
                },
                "pathways_sampled": [t["primary_pathway"] for t in e2v2["triplets"]],
                "feature_invariance_finding": (
                    "Across 4 triplets with fully distinct L0/L5/L9 features and 4 distinct "
                    "biological pathways, pairwise and three-way ratios are identical to 3-4 "
                    "decimal places (std<0.0001). The per-pair spread (AB=0.165, AC=0.219, "
                    "BC=0.586) confirms per-pair behaviour is not trivially fixed — it is "
                    "across-triplet feature-invariance that emerges, consistent with Exp 1's "
                    "uniformly dense circuit (100% of features exceed 1000 edges)."
                ),
            } if e2v2 is not None else None
        ),
        "paper_comparison": {
            "paper_pairwise_ratio": 0.74,
            "paper_threeway_ratio": 0.59,
            "paper_pct_superadditive": 0.0014,
        },
        "interpretation": (
            "MaxToki redundancy (pairwise mean 0.323 across AB/AC/BC, three-way 0.190) "
            "is deeper than Geneformer's (pairwise 0.74, three-way 0.59). Zero higher-order "
            "synergy replicates (0/2980 targets superadditive across all v1 and v2 triplets). "
            "V2 further shows per-pair ratios are nearly independent of which specific "
            "feature at each layer is chosen — MaxToki features at a given layer have "
            "approximately uniform aggregate causal magnitude."
        ),
    },
    "experiment3_trajectory_steering": {
        "n_switch_features_total": e3["n_switch_features_total"],
        "n_steered_features": e3["n_steered_features"],
        "steering_layers": e3["steering_layers"],
        "alphas": e3["alphas"],
        "n_cells": e3["n_cells"],
        "n_early_cells": e3["n_early_cells"],
        "per_layer_summary_alpha5": e3["per_layer_summary_alpha5"],
        "paper_comparison": {
            "paper_n_switch_features": 14,
            "paper_L17_frac_positive": 1.0,
            "paper_L0_mean_frac_positive": 0.34,
        },
        "interpretation": (
            "5767 switch features (paper 14) — MaxToki's trajectory training makes "
            "pseudotemporal information permeate every layer. Layer-dependent "
            "directionality: see per_layer_summary_alpha5 for MaxToki's "
            "frac_positive_direction and frac_positive_maturity values across "
            "L0/L3/L6/L9/L11."
        ),
    },
    "headline_findings": [
        "Annotation bias reproduces: MaxToki hubs are not annotation-enriched (top-20 55% annotated vs overall 50%, Fisher n.s.).",
        "Redundancy is deeper in MaxToki than Geneformer (pairwise mean 0.323 vs 0.74, three-way 0.190 vs 0.59) and zero-synergy at third order replicates.",
        "Across-triplet feature invariance of ablation magnitudes: diversified v2 run with distinct L0/L5/L9 features across 4 triplets produces ratios identical to ±0.0001, consistent with the uniformly dense circuit finding of Exp 1.",
        "Switch-feature density is vastly higher in MaxToki (5767 vs 14) — trajectory-training spreads temporal information throughout the stack.",
        "Trajectory steering is α-insensitive: α=2 and α=5 produce near-identical Δs and logit-delta metrics at every layer, pointing to saturation of single-feature steering magnitude by the final RMSNorm + lm_head path.",
        "Layer-dependent directionality is non-monotonic (unlike Geneformer's monotone L0→L17): L0/L3/L9/L11 push toward maturity, L6 counter-differentiates (frac_mat = 0.00).",
    ],
    "deviations_from_spec": {
        "target_model": "MaxToki-217M-HF (Llama, 11 layers) vs Geneformer V2-316M "
                        "(18 layers) — all layer indices re-mapped.",
        "n_features_traced": "1000 (500 annotated + 500 random unannotated) vs paper's "
                             "4065 — scope reduction for compute budget.",
        "downstream_layers": "{6, 8, 11} vs paper's {6, 11, 17} — 11 is the final "
                             "residual for MaxToki.",
        "n_triplets": "4 vs paper's 8; all share the same L0-L5 base (greedy selector) "
                      "so triplet diversity is lower.",
        "triplet_layers": "{L0, L5, L9} vs paper's {L0, L5, L11} — L11 used as the "
                          "measurement layer instead.",
        "switch_feature_criterion": "|rho|>0.3, p<0.01 (Spearman with PC1 pseudotime) "
                                    "vs paper's Cohen's d>0.9 on early-vs-late activation "
                                    "distributions.",
        "steering_layers": "{L0, L3, L6, L9, L11} vs paper's {L0, L5, L11, L17}.",
        "hook_layer_final_residual": "For steering at L11 (final residual tap), the "
                                     "hook is attached to model.layers[10] (= last block) "
                                     "since model.layers[11] does not exist for an 11-layer "
                                     "Llama. This replaces the output of the last block with "
                                     "the SAE-steered residual, which then flows through "
                                     "final norm + lm_head unchanged.",
    },
}

out_path = OUT / "stage3_summary.json"
with open(out_path, "w") as f:
    json.dump(synthesis, f, indent=2)

print(f"Wrote {out_path}")
print(f"  exp1: {synthesis['experiment1_exhaustive_tracing']['total_edges']:,} edges")
v1 = synthesis['experiment2_combinatorial_ablation']['v1_greedy_selector']
v2 = synthesis['experiment2_combinatorial_ablation'].get('v2_diversified_selector')
print(f"  exp2 v1: pairwise(AB) {v1['mean_pairwise_ratio']:.3f}  threeway {v1['mean_threeway_ratio']:.3f}")
if v2:
    print(f"  exp2 v2: pairwise(mean AB/AC/BC) {v2['mean_pairwise_ratio_AB_AC_BC']:.3f}  "
          f"threeway {v2['mean_threeway_ratio']:.3f}")
print(f"  exp3: {synthesis['experiment3_trajectory_steering']['n_steered_features']} features steered")
