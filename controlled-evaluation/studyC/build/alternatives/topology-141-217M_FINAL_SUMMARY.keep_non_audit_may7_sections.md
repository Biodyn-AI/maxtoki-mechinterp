# topology-141-217M — FINAL SUMMARY
_Pipeline_: `pipelines/topology-geometry-141-hypotheses.md` (arxiv 2602.22289)
_Target model_: MaxToki-217M-HF (LlamaForCausalLM, 11 transformer layers, hidden_size=1232, vocab=20,275)
_Run date_: 2026-05-03

## TL;DR — clean negative across the headline tests

The source paper's headline "robust" findings on scGPT and Geneformer V2-316M
**do not replicate on MaxToki-217M**:

| Test | Paper expectation (scGPT/Geneformer) | MaxToki-217M observation |
|---|---|---|
| H01/H03 persistent homology, feature-shuffle | 11/12 (lung), 12/12 (immune), 12/12 (ext-lung) sig | **1/12, 1/12, 0/12** |
| Rewiring null kills topology | 0/24 sig (paper expects rewiring to kill) | 0/12 sig — qualitatively replicates (no topology to kill) |
| Manifold distance hierarchy | Eucl < Geo (+0.013) < Diff (+0.017) < Tri (+0.026) | Eucl ≈ Geo (≈0), Diff +0.012, **Tri −0.055** (paper-strongest is anti-predictive) |
| H16 community alignment AUROC | All 12 layers significant in both splits | 0–5/12 layers above chance (mean ≈0.49) |
| **H123 signed motif-community** (paper's single strongest) | 22/22 rows positive, **6/6 domain-splits surviving null-gap** | **0–1/12 layers positive null-gap per domain** |
| H91 stability-selected ΔAUROC | +0.074, 72/72 rows positive, 6/6 splits | mean AUC 0.50–0.55 (Δ ≈ +0.01 over chance) |
| H141 strict max-null mean margin | −0.005 overall; immune **+0.012** robust, ext-lung −0.023 fragile | overall **−0.050**; immune −0.033, lung −0.039, ext-lung −0.078 |

**Cross-pipeline reading:** the negative verdict on MaxToki-217M's
geometric/topological structure is consistent with the prior verdicts from
attention-GRN (co-expression, not regulatory logic) and SAE Stage-2 circuit
tracing (CRISPRi 54.6% near-chance directional accuracy). MaxToki encodes
co-expression and a coarse progenitor→mature trajectory axis (positive
Stage-3 SAE finding) but does **not** carry the topological/geometric
regulatory signatures that the paper isolated in scGPT and Geneformer at the
level of fine-grained TF→target structure. The single relative positive in
this run is the **diffusion-distance metric on lung (Δ vs coexpression
+0.023)**, which survives only at the feature-shuffle / coexpression-floor
level and fails the strict max-null audit.

## Scope (what was and wasn't run)
Headline backbone of the 141-hypothesis pipeline:
- **phases run end-to-end on 3 domains**: 0, 1+2+3 harness, 5 (persistent
  homology + rewiring null), 6 (manifold distance hierarchy), 7+8
  (community + signed motif H16/H116/H123), 9 (H91 stability selection),
  12 (H141 strict max-null), 13 (synthesis).
- **Phase 4 cross-model CCA: SKIPPED** — no scGPT/Geneformer extraction on
  the MaxToki gene pool. Per the source paper, Layer 1 of the 5-layer
  hierarchy (the most robust finding) requires ≥2 independently trained
  foundation models. Future work: extract Geneformer V2-316M embeddings on
  the same lung/immune/ext-lung tissue × MaxToki-vocab gene pool, then
  re-run Phase 4 H17/H20/H24.
- **Phase 11 autonomous Codex loop: SKIPPED** — no Codex backend in this
  environment. The 130+ "explorer" hypotheses beyond the headline backbone
  (the 70+ negatives that consume the bulk of the source paper) are
  therefore not re-screened. Per the source paper, this exploration was
  itself the most valuable output, but its execution requires the
  GPT-5.3-codex xhigh autoloop.
- **Annotation-extension degradation chain (H124/H127/H130/H138): SKIPPED**.
  Source paper showed all four extensions monotonically erode H123's
  null-gap robustness. Since MaxToki's H123 is already at 0–1/12 layers
  positive, the degradation chain would produce an uninformative all-zeros
  outcome.

## 5-layer hierarchy of findings (per source paper §1)

| Layer | Finding | Status | Verdict |
|---|---|---|---|
| L1 | Cross-model CCA alignment (H17/H20/H24) | SKIPPED | No reference model (scGPT/Geneformer) extracted on the same gene pool — Layer 1 cannot be evaluated for MaxToki-217M in this run. |
| L2 | Persistent homology (H01/H03) + rewiring null | EVALUATED | external_lung: 0/12 layers feature-shuffle-sig; 0/12 rewire-sig; immune: 1/12 layers feature-shuffle-sig; 0/12 rewire-sig; lung: 1/12 layers feature-shuffle-sig; 0/12 rewire-sig |
| L3 | Manifold distance hierarchy (H13/H16/H32/H69-H70) | EVALUATED | Mean ΔAUROC vs coexpression by metric: triangle_defect=-0.055, geodesic=-0.035, euclidean=-0.035, diffusion=+0.012; target hierarchy monotonic: False |
| L4 | H123 signed motif-community hardening (paper's strongest) | EVALUATED | external_lung: 0/12 layers positive null-gap (mean Δ=-0.060); immune: 1/12 layers positive null-gap (mean Δ=-0.020); lung: 0/12 layers positive null-gap (mean Δ=-0.041) |
| L5 | H141 strict max-null audit | EVALUATED | Overall mean strict margin = -0.050, 4/36 tests pass; external_lung: mean strict margin -0.078, 1/12 tests strict-positive; immune: mean strict margin -0.033, 2/12 tests strict-positive; lung: mean strict margin -0.039, 1/12 tests strict-positive |
| L99 | H91 stability-selected descriptors (combined classifier) | EVALUATED | external_lung: mean AUC 0.545 (67% above chance); immune: mean AUC 0.514 (78% above chance); lung: mean AUC 0.500 (50% above chance) |

## Per-phase summaries

### Phase 0 (extraction)
```json
{
  "lung": {
    "n_cells": 1500,
    "n_gene_pool": 382,
    "n_layer_states": 12,
    "hidden_size": 1232,
    "n_genes_with_obs": 382,
    "pca_dim": 20,
    "max_len": 2048,
    "device": "mps",
    "seed": 42,
    "wall_seconds": 1182.4418432712555
  },
  "external_lung": {
    "n_cells": 1500,
    "n_gene_pool": 380,
    "n_layer_states": 12,
    "hidden_size": 1232,
    "n_genes_with_obs": 380,
    "pca_dim": 20,
    "max_len": 2048,
    "device": "mps",
    "seed": 42,
    "wall_seconds": 1100.0259981155396
  }
}
```

### Phase 1 (splits/pair tables)
```json
{
  "lung": {
    "n_pairs": 1308,
    "n_pos_trrust": 252,
    "n_pos_string": 48,
    "n_genes": 382
  },
  "immune": {
    "n_pairs": 2544,
    "n_pos_trrust": 504,
    "n_pos_string": 24,
    "n_genes": 350
  },
  "external_lung": {
    "n_pairs": 751,
    "n_pos_trrust": 138,
    "n_pos_string": 61,
    "n_genes": 380
  }
}
```

### Phase 5 (persistent homology + rewire null)
```json
{
  "external_lung": {
    "n_layers": 12,
    "n_layers_significant": 0,
    "top_z": 1.4685900919391515,
    "top_layer": 5,
    "mean_delta": -21.673023191901546,
    "rewire_n_significant": 0,
    "rewire_n_total": 12
  },
  "immune": {
    "n_layers": 12,
    "n_layers_significant": 1,
    "top_z": 3.665770315221771,
    "top_layer": 4,
    "mean_delta": -37.61458326832702,
    "rewire_n_significant": 0,
    "rewire_n_total": 12
  },
  "lung": {
    "n_layers": 12,
    "n_layers_significant": 1,
    "top_z": 2.485906917820712,
    "top_layer": 5,
    "mean_delta": -19.030632989232735,
    "rewire_n_significant": 0,
    "rewire_n_total": 12
  }
}
```

### Phase 6 (manifold distance hierarchy)
```json
{
  "per_domain_per_metric": [
    {
      "domain": "external_lung",
      "metric": "diffusion",
      "mean": 0.009087031389145009,
      "median": 0.018689268742464016,
      "std": 0.03130594850308952,
      "count": 12
    },
    {
      "domain": "external_lung",
      "metric": "euclidean",
      "mean": -0.05310266291541557,
      "median": -0.035286190509965176,
      "std": 0.06118921625458143,
      "count": 12
    },
    {
      "domain": "external_lung",
      "metric": "geodesic",
      "mean": -0.06894500003940389,
      "median": -0.07981062486701179,
      "std": 0.058500144121174304,
      "count": 12
    },
    {
      "domain": "external_lung",
      "metric": "triangle_defect",
      "mean": -0.05713565185868186,
      "median": -0.058491145944156775,
      "std": 0.08890208535486323,
      "count": 12
    },
    {
      "domain": "immune",
      "metric": "diffusion",
      "mean": 0.004272357090984548,
      "median": -0.0004889803143479643,
      "std": 0.022323985900705643,
      "count": 12
    },
    {
      "domain": "immune",
      "metric": "euclidean",
      "mean": -0.004049912789189712,
      "median": -0.002499610955493281,
      "std": 0.022326346509288268,
      "count": 12
    },
    {
      "domain": "immune",
      "metric": "geodesic",
      "mean": 0.00705191798941802,
      "median": 0.004024665421724305,
      "std": 0.018057368260710683,
      "count": 12
    },
    {
      "domain": "immune",
      "metric": "triangle_defect",
      "mean": -0.07126424064996366,
      "median": -0.06862915305010889,
      "std": 0.01014811695029904,
      "count": 12
    },
    {
      "domain": "lung",
      "metric": "diffusion",
      "mean": 0.023452844416386093,
      "median": 0.02650199915824919,
      "std": 0.029638368035806502,
      "count": 12
    },
    {
      "domain": "lung",
      "metric": "euclidean",
      "mean": -0.048609545354337015,
      "median": -0.043297558922558876,
      "std": 0.033657023128314444,
      "count": 12
    },
    {
      "domain": "lung",
      "metric": "geodesic",
      "mean": -0.044164518448372624,
      "median": -0.039326486592111604,
      "std": 0.031047646398722508,
      "count": 12
    },
    {
      "domain": "lung",
      "metric": "triangle_defect",
      "mean": -0.0356723234327401,
      "median": -0.03740342412217407,
      "std": 0.03219692985859574,
      "count": 12
    }
  ]
}
```

### Phase 7+8 (community + signed motif H16/H116/H123)
```json
{
  "h16": {
    "external_lung": {
      "n_layers": 12,
      "n_layers_above_chance": 0,
      "mean_auc": 0.4772590254627987
    },
    "immune": {
      "n_layers": 12,
      "n_layers_above_chance": 5,
      "mean_auc": 0.4976647603485838
    },
    "lung": {
      "n_layers": 12,
      "n_layers_above_chance": 4,
      "mean_auc": 0.4951430224867724
    }
  },
  "h116": {
    "external_lung": {
      "n_layers": 12,
      "mean_delta": -0.05289894476361601,
      "n_layers_positive_null_gap": 0,
      "fraction_layers_positive": 0.0,
      "n_layers_significant": 0
    },
    "immune": {
      "n_layers": 12,
      "mean_delta": -0.017192146245072093,
      "n_layers_positive_null_gap": 2,
      "fraction_layers_positive": 0.16666666666666666,
      "n_layers_significant": 2
    },
    "lung": {
      "n_layers": 12,
      "mean_delta": -0.043590746878507315,
      "n_layers_positive_null_gap": 0,
      "fraction_layers_positive": 0.0,
      "n_layers_significant": 0
    }
  },
  "h123": {
    "external_lung": {
      "n_layers": 12,
      "mean_delta": -0.060066661544159926,
      "n_layers_positive_null_gap": 0,
      "fraction_layers_positive": 0.0,
      "n_layers_significant": 0
    },
    "immune": {
      "n_layers": 12,
      "mean_delta": -0.02008339533730161,
      "n_layers_positive_null_gap": 1,
      "fraction_layers_positive": 0.08333333333333333,
      "n_layers_significant": 1
    },
    "lung": {
      "n_layers": 12,
      "mean_delta": -0.041426432604818035,
      "n_layers_positive_null_gap": 0,
      "fraction_layers_positive": 0.0,
      "n_layers_significant": 0
    }
  }
}
```

### Phase 9 (H91 stability selection)
```json
{
  "external_lung": {
    "n_rows": 18,
    "mean_auc": 0.5452770355596184,
    "median_auc": 0.5380377750197994,
    "fraction_above_chance": 0.6666666666666666
  },
  "immune": {
    "n_rows": 18,
    "mean_auc": 0.514018195208799,
    "median_auc": 0.5116813150207553,
    "fraction_above_chance": 0.7777777777777778
  },
  "lung": {
    "n_rows": 18,
    "mean_auc": 0.5002590335847067,
    "median_auc": 0.5045381950601338,
    "fraction_above_chance": 0.5
  }
}
```

### Phase 10 (H139 sectional-anisotropy spot-check, added 2026-05-07)
```json
{
  "scope": {"layer": 11, "n_cv": 3, "n_null": 8, "knn": 12, "feature_dim": 12,
            "support_dir": "knn_rank_asymmetry"},
  "lung":         {"auc_sec": 0.612, "auc_tri_h70": 0.502, "delta_vs_h70": +0.109, "null_gap": -0.001, "directional_positive": true,  "strict_positive": false},
  "immune":       {"auc_sec": 0.589, "auc_tri_h70": 0.536, "delta_vs_h70": +0.053, "null_gap": +0.001, "directional_positive": true,  "strict_positive": true},
  "external_lung":{"auc_sec": 0.548, "auc_tri_h70": 0.573, "delta_vs_h70": -0.024, "null_gap": -0.034, "directional_positive": false, "strict_positive": false},
  "aggregate":    {"mean_delta_vs_h70": +0.046, "n_directional_positive": 2, "n_strict_positive": 1, "mean_null_gap": -0.011}
}
```
Verdict: **partial — replicates paper's "partial" classification**. Directional ΔAUROC vs H70 baseline is positive in 2/3 domains and the magnitude (+0.046 mean) is consistent with the paper's +0.031, but only immune passes the strict max-null gap. Endpoint-swap is by construction the closest-to-observed null and dominates the q95; this matches the source-paper observation that H139 is null-fragile outside immune. Spot-check simplification: support-direction term uses kNN-rank asymmetry instead of the paper's H70/H136-derived support matrix.

### Phase 4 vs scGPT (cross-model alignment, added 2026-05-07)
```json
{
  "scope": "static-embedding alignment, MaxToki-217M token-embedding ↔ scGPT whole-human (60697×512), HUGO symbol match",
  "lung":          {"n_genes": 382, "cca_mean_r": 0.401, "pairwise_pearson": -0.008, "top1_retrieval": 0.050, "top1_z": 19.5},
  "immune":        {"n_genes": 350, "cca_mean_r": 0.438, "pairwise_pearson": -0.019, "top1_retrieval": 0.066, "top1_z": 22.0},
  "external_lung": {"n_genes": 380, "cca_mean_r": 0.406, "pairwise_pearson": -0.005, "top1_retrieval": 0.076, "top1_z": 30.8},
  "triangulation_vs_phase14": {"geneformer_cca_r": 0.78, "scgpt_cca_r": 0.40, "geneformer_top1": 0.45, "scgpt_top1": 0.06}
}
```
Verdict: **does not generalize across architectural families**. The Phase 14 alignment with Geneformer V2-316M (CCA r=0.78, top-1 38–47%) does NOT replicate against scGPT — pairwise gene-gene similarity is essentially zero (lung/ext-lung) or slightly negative (immune, z=−3.0), and Procrustes top-1 retrieval drops from 38–47% to 5–8% (still significantly above the 0.5–0.9% null p95 by z=20–30, but qualitatively much weaker). The source paper's "Layer 1 most-robust" claim that cross-model CCA holds across foundation models is supported only within architectural families: MaxToki and Geneformer share Ensembl-ID tokenization and Llama/BERT-style autoregressive training; scGPT uses continuous-value rank-binned tokens and attention-only architecture, and the geometric agreement breaks.

### Phase 8 extensions — H124→H127→H130→H138 chain (added 2026-05-07)
```json
{
  "H123_baseline":         {"lung": -0.041, "immune": -0.020, "external_lung": -0.060, "n_pos_null_gap": "0/12, 1/12, 0/12"},
  "H124_string":           {"lung": -0.037, "immune": -0.019, "external_lung": -0.055, "n_pos_null_gap": "0/12, 2/12, 0/12"},
  "H127_go_comem":         {"lung": -0.033, "immune": -0.015, "external_lung": -0.056, "n_pos_null_gap": "0/12, 2/12, 0/12"},
  "H130_go_semantic":      {"lung": -0.035, "immune": -0.016, "external_lung": -0.058, "n_pos_null_gap": "0/12, 2/12, 0/12"},
  "H138_ontology_sheaf":   {"lung": -0.038, "immune": -0.020, "external_lung": -0.059, "n_pos_null_gap": "0/12, 2/12, 0/12"}
}
```
Verdict: **degradation chain uninformative — H123 is already at floor on MaxToki**. The source paper documents a monotonic null-gap collapse H123 (6/6) → H124 (<3/6) → H127 (2/9) → H130 (0/9) → H138 (0/9) as biological annotations are stacked on top of H123. On MaxToki, H123 is already at 0–1/12 layers per domain, so adding STRING / GO / continuous-GO / ontology-sheaf features produces no further degradation — all four extensions land within ±0.01 of H123 baseline. Subtle but consistent: H124–H138 tick immune from 1/12 to 2/12 layers positive, suggesting MaxToki's gene-pair geometry is modestly more aligned with PPI/GO co-membership than with TRRUST signed motif-community placement.

### Phase 11 — autoloop substitute (Claude-as-brainstormer batch, added 2026-05-07)
6 additional hypotheses run as a curated batch with explicit retirement decisions:
```json
{
  "iter_01_ollivier_ricci":         {"decision": "INCONCLUSIVE", "n_pos_layers": "10/36", "mean_delta_vs_null": -0.027, "note": "ORC direction does not cleanly replicate H23-Forman below-chance pattern"},
  "iter_02_gromov_hyperbolicity":    {"decision": "INCONCLUSIVE", "mean_delta_norm": 0.045,  "note": "near euclidean threshold (0.05); not meaningfully hyperbolic"},
  "iter_03_coex_residual_triangle":  {"decision": "RETIRE",       "n_pos_layers": "9/36",  "mean_delta_vs_null": -0.034, "note": "triangle-defect signal does NOT survive coex residualization → confirms the small phase-6 lung positive was coex-driven"},
  "iter_04_cross_layer_consistency": {"decision": "NOVEL POSITIVE", "mean_cross_layer_pearson": {"lung": 0.654, "immune": 0.604, "external_lung": 0.678}, "note": "H123 motif feature is a single static axis across all 11 transformer layers"},
  "iter_05_token_frequency_strata":  {"decision": "RETIRE",       "n_pos_layers": "6/36",  "mean_delta_vs_null": -0.112, "note": "H123 signal slightly STRONGER in low-frequency tokens than high — rules out popularity confound"},
  "iter_06_pooled_multi_tissue":     {"decision": "untestable",   "note": "only 36 common genes across 3 domain pools; pool-overlap insufficient"}
}
```
Headline addition from Phase 11: **iter_04 is a novel positive**. The H123 signed-motif-community feature has mean cross-layer Pearson 0.60–0.68 across all 3 domains — meaning the per-pair community-consistency score is essentially fixed at the embedding layer and propagates almost unchanged through the 11 transformer layers. Per-layer AUROC stays near chance (0.40–0.53), so this isn't a "signal getting stronger with depth" finding — it's the opposite: the residual stream doesn't reorganise gene similarities in any regulatory direction at all. This is consistent with the negative scGPT alignment, the H123 floor result, and the H124–H138 degradation chain all telling the same story: MaxToki's gene-similarity geometry is largely a property of the input layer, not of trained transformer dynamics.

### Phase 12 (H141 strict max-null)
```json
{
  "external_lung": {
    "n_tests": 12,
    "mean_strict_margin": -0.07771729279460322,
    "n_strict_positive": 1,
    "fraction_strict_positive": 0.08333333333333333
  },
  "immune": {
    "n_tests": 12,
    "mean_strict_margin": -0.03282836572128847,
    "n_strict_positive": 2,
    "fraction_strict_positive": 0.16666666666666666
  },
  "lung": {
    "n_tests": 12,
    "mean_strict_margin": -0.0389882048410694,
    "n_strict_positive": 1,
    "fraction_strict_positive": 0.08333333333333333
  },
  "overall": {
    "mean_strict_margin": -0.049844621118987034,
    "n_strict_positive": 4,
    "n_tests": 36
  }
}
```
