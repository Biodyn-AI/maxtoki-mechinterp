# Attention-GRN pipeline — MaxToki-217M run report

**Overall verdict:** FAIL — at least one mandatory criterion failed

## Phase 0 — extraction

- model: `MaxToki-217M-HF`
- n_ctrl: `2000`
- n_hvg: `1500`
- max_len: `2048`
- model_n_layers: `11`
- model_n_heads: `8`
- mean_seq_len: `2038.9945`
- mean_fwd_seconds: `1.6374004645347595`
- total_phase_seconds: `3286.275767803192`
- device: `mps`

## Phase 1 — trivial baselines

- mean AUROC `gene_variance`: **0.5957**
- mean AUROC `gene_mean_expr`: **0.5605**
- mean AUROC `gene_one_minus_dropout`: **0.5483**
- mean AUROC `attention_primary`: **0.5119**
- mean AUROC `attention_primary_norm`: **0.5119**
- mean AUROC `spearman`: **0.5103**
- mean AUROC `gene_tf_out_degree`: **0.5000**

### Paired Wilcoxon vs attention/correlation (n=174)

| baseline | reference | mean Δ | Wilcoxon p (BH) |
|---|---|---|---|
| gene_variance | attention_primary | +0.0837 | 4.03e-10 |
| gene_variance | attention_primary_norm | +0.0837 | 4.03e-10 |
| gene_variance | spearman | +0.0854 | 7.13e-12 |
| gene_mean_expr | attention_primary | +0.0485 | 2.31e-08 |
| gene_mean_expr | attention_primary_norm | +0.0485 | 2.31e-08 |
| gene_mean_expr | spearman | +0.0502 | 2.22e-07 |
| gene_one_minus_dropout | attention_primary | +0.0363 | 5.58e-06 |
| gene_one_minus_dropout | attention_primary_norm | +0.0363 | 5.58e-06 |
| gene_one_minus_dropout | spearman | +0.0380 | 1.46e-05 |
| gene_tf_out_degree | attention_primary | -0.0120 | 1.77e-02 |
| gene_tf_out_degree | attention_primary_norm | -0.0120 | 1.77e-02 |
| gene_tf_out_degree | spearman | -0.0103 | 1.78e-01 |
| attention_primary | spearman | +0.0017 | 6.98e-01 |

## Phase 2 — incremental value (ΔAUROC vs gene-only)

| split | model | feature_set | gene_only | new | ΔAUROC |
|---|---|---|---|---|---|
| cross_gene | logreg | gene_plus_attn | 0.5898 | 0.5903 | +0.0005 |
| cross_gene | logreg | gene_plus_corr | 0.5898 | 0.5901 | +0.0003 |
| cross_gene | logreg | gene_plus_both | 0.5898 | 0.5906 | +0.0008 |
| cross_gene | logreg | attn_only | 0.5898 | 0.5299 | -0.0599 |
| cross_gene | logreg | corr_only | 0.5898 | 0.4944 | -0.0955 |
| cross_pert | logreg | gene_plus_attn | 0.5905 | 0.5912 | +0.0007 |
| cross_pert | logreg | gene_plus_corr | 0.5905 | 0.5905 | +0.0000 |
| cross_pert | logreg | gene_plus_both | 0.5905 | 0.5912 | +0.0007 |
| cross_pert | logreg | attn_only | 0.5905 | 0.5357 | -0.0548 |
| cross_pert | logreg | corr_only | 0.5905 | 0.4901 | -0.1004 |
| joint | logreg | gene_plus_attn | 0.5905 | 0.5911 | +0.0006 |
| joint | logreg | gene_plus_corr | 0.5905 | 0.5910 | +0.0005 |

## Phase 3a — cross-fitted residualization

| edge | model | R² | baseline AUROC | residual AUROC | Δ lost |
|---|---|---|---|---|---|
| attention | ols | 0.000 | 0.4829 | 0.4866 | -0.0037 |
| attention | gbdt | 0.004 | 0.4829 | 0.4970 | -0.0141 |
| spearman | ols | 0.019 | 0.5432 | 0.5065 | +0.0367 |
| spearman | gbdt | 0.081 | 0.5432 | 0.4956 | +0.0477 |

## Phase 3b — degree-preserving + label-shuffle nulls

- observed attention TRRUST AUROC: **0.4829**
- curveball null (n=50): mean=0.4831 ± 0.0011  z = -0.14
- label-shuffle null (n=1000): mean=0.4956 ± 0.0429  z = -0.30

## Phase 3c — propensity matching

- n matched pairs: `58446`
- matched positive rate: `0.1667`
- matched GroupKFold AUROC: gene_only=0.5104  gene+attn=0.5151  gene+corr=0.5101  (Δattn=+0.0046, Δcorr=-0.0003)

## Phase 4 — 6-condition causal head ablation

| condition | n_masked | TRRUST AUROC (primary) | ΔAUROC | best layer |
|---|---|---|---|---|
| baseline | 0 | 0.4399 | +0.0000 | L1 |
| top5_trrust | 5 | 0.4347 | -0.0051 | L2 |
| random5_A | 5 | 0.4544 | +0.0146 | L1 |
| random5_B | 5 | 0.4344 | -0.0054 | L1 |
| random5_C | 5 | 0.4578 | +0.0180 | L1 |
| entropy_matched_5 | 5 | 0.4533 | +0.0135 | L1 |

## Verdict on four mandatory criteria

- **C1_attention_beats_trivial_baselines** → `FAIL`
- **C2_incremental_value_over_gene_features** → `FAIL`
- **C3_residualized_retains_signal** → `FAIL`
- **C4_top_heads_causal** → `FAIL`

**Overall:** FAIL — at least one mandatory criterion failed

## Scope

- dataset: `Replogle K562 non-targeting controls only`
- reference network: `TRRUST only (no STRING/Reactome/KEGG/GO)`
- N_ctrl: `2000`  N_hvg: `1500`
- curveball null iters: `50`

**Phases run:** 0a, 1, 2, 3, 4, 12
**Phases skipped:** 0b (value-weighted edges); 5 (cross-context replication — RPE1, Adamson, Dixit, Shifrut, Tian); 6 (CSSI); 7 (biological characterization — STRING, Reactome, KEGG, GO); 8 (detectability); 9 (ortholog transfer); 10 (pseudotime); 11 (batch leakage, calibration, TRRUST circularity, HVG protocol, mediation)