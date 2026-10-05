# Attention-GRN pipeline — MaxToki-217M run report

**Overall verdict:** FAIL — at least one mandatory criterion failed

## Phase 0 — extraction

- model: `MaxToki-217M-HF`
- n_ctrl: `2000`
- n_hvg: `1500`
- model_n_layers: `11`
- model_n_heads: `8`
- mean_seq_len: `991.08`
- mean_fwd_seconds: `0.818928124666214`
- total_phase_seconds: `1645.8092510700226`
- device: `mps`

## Phase 1 — trivial baselines

- mean AUROC `gene_mean_expr`: **0.7125**
- mean AUROC `gene_variance`: **0.7123**
- mean AUROC `gene_one_minus_dropout`: **0.6979**
- mean AUROC `spearman`: **0.6079**
- mean AUROC `attention_primary_norm`: **0.5081**
- mean AUROC `attention_primary`: **0.5081**
- mean AUROC `gene_tf_out_degree`: **0.4871**

### Paired Wilcoxon vs attention/correlation (n=57)

| baseline | reference | mean Δ | Wilcoxon p (BH) |
|---|---|---|---|
| gene_variance | attention_primary | +0.2042 | 1.38e-10 |
| gene_variance | attention_primary_norm | +0.2042 | 1.38e-10 |
| gene_variance | spearman | +0.1044 | 1.34e-06 |
| gene_mean_expr | attention_primary | +0.2044 | 1.38e-10 |
| gene_mean_expr | attention_primary_norm | +0.2044 | 1.38e-10 |
| gene_mean_expr | spearman | +0.1046 | 1.23e-06 |
| gene_one_minus_dropout | attention_primary | +0.1898 | 1.38e-10 |
| gene_one_minus_dropout | attention_primary_norm | +0.1898 | 1.38e-10 |
| gene_one_minus_dropout | spearman | +0.0900 | 1.32e-05 |
| gene_tf_out_degree | attention_primary | -0.0210 | 6.02e-02 |
| gene_tf_out_degree | attention_primary_norm | -0.0210 | 6.02e-02 |
| gene_tf_out_degree | spearman | -0.1208 | 2.43e-09 |
| attention_primary | spearman | -0.0998 | 2.85e-05 |

## Phase 2 — incremental value (ΔAUROC vs gene-only)

| split | model | feature_set | gene_only | new | ΔAUROC |
|---|---|---|---|---|---|
| cross_gene | logreg | gene_plus_attn | 0.7971 | 0.7970 | -0.0002 |
| cross_gene | logreg | gene_plus_corr | 0.7971 | 0.8053 | +0.0082 |
| cross_gene | logreg | gene_plus_both | 0.7971 | 0.8051 | +0.0080 |
| cross_gene | logreg | attn_only | 0.7971 | 0.5399 | -0.2573 |
| cross_gene | logreg | corr_only | 0.7971 | 0.5992 | -0.1980 |
| cross_pert | logreg | gene_plus_attn | 0.8022 | 0.8022 | -0.0001 |
| cross_pert | logreg | gene_plus_corr | 0.8022 | 0.8081 | +0.0059 |
| cross_pert | logreg | gene_plus_both | 0.8022 | 0.8080 | +0.0057 |
| cross_pert | logreg | attn_only | 0.8022 | 0.5388 | -0.2634 |
| cross_pert | logreg | corr_only | 0.8022 | 0.5957 | -0.2065 |
| joint | logreg | gene_plus_attn | 0.7998 | 0.7998 | -0.0001 |
| joint | logreg | gene_plus_corr | 0.7998 | 0.8078 | +0.0079 |

## Phase 3a — cross-fitted residualization

| edge | model | R² | baseline AUROC | residual AUROC | Δ lost |
|---|---|---|---|---|---|
| attention | ols | 0.001 | 0.5195 | 0.5202 | -0.0007 |
| attention | gbdt | 0.008 | 0.5195 | 0.5317 | -0.0122 |
| spearman | ols | 0.017 | 0.5389 | 0.5216 | +0.0174 |
| spearman | gbdt | 0.075 | 0.5389 | 0.5190 | +0.0200 |

## Phase 3b — degree-preserving + label-shuffle nulls

- observed attention TRRUST AUROC: **0.5195**
- curveball null (n=50): mean=0.5190 ± 0.0031  z = 0.16
- label-shuffle null (n=1000): mean=0.4996 ± 0.0185  z = 1.08

## Phase 3c — propensity matching

- n matched pairs: `26628`
- matched positive rate: `0.1667`
- matched GroupKFold AUROC: gene_only=0.5414  gene+attn=0.5416  gene+corr=0.5689  (Δattn=+0.0001, Δcorr=+0.0275)

## Verdict on four mandatory criteria

- **C1_attention_beats_trivial_baselines** → `FAIL`
- **C2_incremental_value_over_gene_features** → `FAIL`
- **C3_residualized_retains_signal** → `PASS`

**Overall:** FAIL — at least one mandatory criterion failed

## Scope

- dataset: `Replogle K562 non-targeting controls only`
- reference network: `TRRUST only (no STRING/Reactome/KEGG/GO)`
- N_ctrl: `2000`  N_hvg: `1500`
- curveball null iters: `50`

**Phases run:** 0a, 1, 2, 3, 4, 12
**Phases skipped:** 0b (value-weighted edges); 5 (cross-context replication — RPE1, Adamson, Dixit, Shifrut, Tian); 6 (CSSI); 7 (biological characterization — STRING, Reactome, KEGG, GO); 8 (detectability); 9 (ortholog transfer); 10 (pseudotime); 11 (batch leakage, calibration, TRRUST circularity, HVG protocol, mediation)