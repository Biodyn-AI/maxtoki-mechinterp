# Cross-context replication — MaxToki-217M, K562 vs RPE1

One-line summary of the attention-GRN pipeline verdict on both cell lines.

## Headline

- **K562:** FAIL — at least one mandatory criterion failed
- **RPE1:** FAIL — at least one mandatory criterion failed

## Four mandatory criteria

| Criterion | K562 | RPE1 |
|---|---|---|
| C1_attention_beats_trivial_baselines | FAIL | FAIL |
| C2_incremental_value_over_gene_features | FAIL | FAIL |
| C3_residualized_retains_signal | FAIL | FAIL |
| C4_top_heads_causal | FAIL | – |

## Headline metrics

| Metric | K562 | RPE1 |
|---|---|---|
| attention mean AUROC (primary layer) | 0.5119 | 0.6033 |
| incremental ΔAUROC (cross-pert gene+attn vs gene-only) | 0.0007 | -0.0003 |
| residualized TRRUST AUROC (OLS cross-fit) | 0.4866 | 0.4922 |
| baseline TRRUST AUROC | 0.4829 | 0.6045 |
| top-5 TRRUST-head ablation Δ AUROC (primary layer) | 0.0051 | None |

## Scope

- Dataset K562: Replogle concat h5ad, non-targeting NT controls, 2000 cells
- Dataset RPE1: ReplogleWeissman2022_rpe1.h5ad, `control` category, 2000 cells
- Reference network: TRRUST only
- N_hvg: 1500 per dataset (intersected with MaxToki Ensembl vocab)
- K562 primary layer: L8  (Llama decoder, 11 layers)
- K562 ran all phases (0, 0b, 1, 2, 3, 4, 12). RPE1 scope: 0, 1, 2, 3, 12 (no Phase 0b or Phase 4 to fit compute budget).