# Attention-GRN pipeline — MaxToki-217M run report

**Overall verdict:** FAIL — at least one mandatory criterion failed

## Phase 0 — extraction

- model: `MaxToki-217M-HF`
- n_ctrl: `200`
- n_hvg: `1500`
- model_n_layers: `20`
- model_n_heads: `16`
- mean_seq_len: `2040.41`
- mean_fwd_seconds: `28.386985354423523`
- total_phase_seconds: `5735.826719045639`
- device: `mps`

## Phase 1 — trivial baselines

- mean AUROC `gene_variance`: **0.5961**
- mean AUROC `gene_mean_expr`: **0.5724**
- mean AUROC `gene_one_minus_dropout`: **0.5604**
- mean AUROC `attention_primary`: **0.5398**
- mean AUROC `attention_primary_norm`: **0.5398**
- mean AUROC `gene_tf_out_degree`: **0.5014**
- mean AUROC `spearman`: **0.4924**

### Paired Wilcoxon vs attention/correlation (n=155)

| baseline | reference | mean Δ | Wilcoxon p (BH) |
|---|---|---|---|
| gene_variance | attention_primary | +0.0562 | 9.47e-05 |
| gene_variance | attention_primary_norm | +0.0562 | 9.47e-05 |
| gene_variance | spearman | +0.1037 | 6.26e-16 |
| gene_mean_expr | attention_primary | +0.0325 | 3.92e-04 |
| gene_mean_expr | attention_primary_norm | +0.0325 | 3.92e-04 |
| gene_mean_expr | spearman | +0.0800 | 9.69e-10 |
| gene_one_minus_dropout | attention_primary | +0.0205 | 7.09e-03 |
| gene_one_minus_dropout | attention_primary_norm | +0.0205 | 7.09e-03 |
| gene_one_minus_dropout | spearman | +0.0680 | 1.14e-07 |
| gene_tf_out_degree | attention_primary | -0.0384 | 1.04e-05 |
| gene_tf_out_degree | attention_primary_norm | -0.0384 | 1.04e-05 |
| gene_tf_out_degree | spearman | +0.0090 | 3.72e-01 |
| attention_primary | spearman | +0.0475 | 1.04e-05 |

## Phase 2 — incremental value (ΔAUROC vs gene-only)

| split | model | feature_set | gene_only | new | ΔAUROC |
|---|---|---|---|---|---|
| cross_gene | logreg | gene_plus_attn | 0.5972 | 0.5993 | +0.0021 |
| cross_gene | logreg | gene_plus_corr | 0.5972 | 0.5973 | +0.0001 |
| cross_gene | logreg | gene_plus_both | 0.5972 | 0.5995 | +0.0023 |
| cross_gene | logreg | attn_only | 0.5972 | 0.5632 | -0.0339 |
| cross_gene | logreg | corr_only | 0.5972 | 0.5030 | -0.0942 |
| cross_pert | logreg | gene_plus_attn | 0.5970 | 0.5991 | +0.0020 |
| cross_pert | logreg | gene_plus_corr | 0.5970 | 0.5969 | -0.0001 |
| cross_pert | logreg | gene_plus_both | 0.5970 | 0.5990 | +0.0020 |
| cross_pert | logreg | attn_only | 0.5970 | 0.5626 | -0.0344 |
| cross_pert | logreg | corr_only | 0.5970 | 0.5028 | -0.0942 |
| joint | logreg | gene_plus_attn | 0.5974 | 0.5995 | +0.0021 |
| joint | logreg | gene_plus_corr | 0.5974 | 0.5975 | +0.0001 |

## Phase 3a — cross-fitted residualization

| edge | model | R² | baseline AUROC | residual AUROC | Δ lost |
|---|---|---|---|---|---|
| attention | ols | 0.000 | 0.5554 | 0.5434 | +0.0120 |
| attention | gbdt | 0.023 | 0.5554 | 0.5111 | +0.0443 |
| spearman | ols | 0.001 | 0.5800 | 0.5760 | +0.0040 |
| spearman | gbdt | 0.006 | 0.5800 | 0.5711 | +0.0089 |

## Phase 3b — degree-preserving + label-shuffle nulls

- observed attention TRRUST AUROC: **0.5554**
- curveball null (n=50): mean=0.5554 ± 0.0007  z = -0.04
- label-shuffle null (n=1000): mean=0.5091 ± 0.0316  z = 1.47

## Phase 3c — propensity matching

- n matched pairs: `51522`
- matched positive rate: `0.1667`
- matched GroupKFold AUROC: gene_only=0.5028  gene+attn=0.5485  gene+corr=0.5047  (Δattn=+0.0457, Δcorr=+0.0019)

## Verdict on four mandatory criteria

- **C1_attention_beats_trivial_baselines** → `FAIL`
- **C2_incremental_value_over_gene_features** → `FAIL`
- **C3_residualized_retains_signal** → `PASS`

**Overall:** FAIL — at least one mandatory criterion failed

## Scope

- dataset: `Replogle K562 non-targeting controls only`
- reference network: `TRRUST only (no STRING/Reactome/KEGG/GO)`
- N_ctrl: `200`  N_hvg: `1500`
- curveball null iters: `50`

**Phases run:** 0a, 1, 2, 3, 4, 12
**Phases skipped:** 0b (value-weighted edges); 5 (cross-context replication — RPE1, Adamson, Dixit, Shifrut, Tian); 6 (CSSI); 7 (biological characterization — STRING, Reactome, KEGG, GO); 8 (detectability); 9 (ortholog transfer); 10 (pseudotime); 11 (batch leakage, calibration, TRRUST circularity, HVG protocol, mediation)