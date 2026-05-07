# Attention-GRN pipeline — MaxToki-217M run report

**Overall verdict:** FAIL — at least one mandatory criterion failed

## Phase 0 — extraction

- model: `MaxToki-217M-HF`
- n_ctrl: `2000`
- n_hvg: `1500`
- model_n_layers: `11`
- model_n_heads: `8`
- mean_seq_len: `2038.8755`
- mean_fwd_seconds: `2.1165070651769637`
- total_phase_seconds: `4259.69699716568`
- device: `mps`

## Phase 1 — trivial baselines

- mean AUROC `gene_variance`: **0.7664**
- mean AUROC `spearman`: **0.6219**
- mean AUROC `attention_primary`: **0.6033**
- mean AUROC `attention_primary_norm`: **0.6033**
- mean AUROC `gene_mean_expr`: **0.5822**
- mean AUROC `gene_tf_out_degree`: **0.5069**
- mean AUROC `gene_one_minus_dropout`: **0.5002**

### Paired Wilcoxon vs attention/correlation (n=325)

| baseline | reference | mean Δ | Wilcoxon p (BH) |
|---|---|---|---|
| gene_variance | attention_primary | +0.1632 | 1.51e-51 |
| gene_variance | attention_primary_norm | +0.1632 | 1.51e-51 |
| gene_variance | spearman | +0.1445 | 9.69e-43 |
| gene_mean_expr | attention_primary | -0.0210 | 7.25e-08 |
| gene_mean_expr | attention_primary_norm | -0.0210 | 7.25e-08 |
| gene_mean_expr | spearman | -0.0397 | 1.22e-05 |
| gene_one_minus_dropout | attention_primary | -0.1030 | 4.06e-36 |
| gene_one_minus_dropout | attention_primary_norm | -0.1030 | 4.06e-36 |
| gene_one_minus_dropout | spearman | -0.1217 | 1.86e-27 |
| gene_tf_out_degree | attention_primary | -0.0963 | 3.45e-41 |
| gene_tf_out_degree | attention_primary_norm | -0.0963 | 3.45e-41 |
| gene_tf_out_degree | spearman | -0.1150 | 5.99e-34 |
| attention_primary | spearman | -0.0187 | 3.26e-03 |

## Phase 2 — incremental value (ΔAUROC vs gene-only)

| split | model | feature_set | gene_only | new | ΔAUROC |
|---|---|---|---|---|---|
| cross_gene | logreg | gene_plus_attn | 0.7476 | 0.7470 | -0.0006 |
| cross_gene | logreg | gene_plus_corr | 0.7476 | 0.7455 | -0.0020 |
| cross_gene | logreg | gene_plus_both | 0.7476 | 0.7452 | -0.0024 |
| cross_gene | logreg | attn_only | 0.7476 | 0.5888 | -0.1588 |
| cross_gene | logreg | corr_only | 0.7476 | 0.6156 | -0.1320 |
| cross_pert | logreg | gene_plus_attn | 0.7475 | 0.7472 | -0.0003 |
| cross_pert | logreg | gene_plus_corr | 0.7475 | 0.7456 | -0.0019 |
| cross_pert | logreg | gene_plus_both | 0.7475 | 0.7455 | -0.0020 |
| cross_pert | logreg | attn_only | 0.7475 | 0.5886 | -0.1589 |
| cross_pert | logreg | corr_only | 0.7475 | 0.6159 | -0.1316 |
| joint | logreg | gene_plus_attn | 0.7477 | 0.7475 | -0.0002 |
| joint | logreg | gene_plus_corr | 0.7477 | 0.7458 | -0.0019 |

## Phase 3a — cross-fitted residualization

| edge | model | R² | baseline AUROC | residual AUROC | Δ lost |
|---|---|---|---|---|---|
| attention | ols | 0.061 | 0.6045 | 0.4922 | +0.1123 |
| attention | gbdt | 0.372 | 0.6045 | 0.5191 | +0.0854 |
| spearman | ols | 0.140 | 0.5963 | 0.4799 | +0.1164 |
| spearman | gbdt | 0.281 | 0.5963 | 0.4624 | +0.1339 |

## Phase 3b — degree-preserving + label-shuffle nulls

- observed attention TRRUST AUROC: **0.6045**
- curveball null (n=50): mean=0.6043 ± 0.0015  z = 0.17
- label-shuffle null (n=1000): mean=0.5019 ± 0.0245  z = 4.18

## Phase 3c — propensity matching

- n matched pairs: `289470`
- matched positive rate: `0.1667`
- matched GroupKFold AUROC: gene_only=0.5702  gene+attn=0.5717  gene+corr=0.5931  (Δattn=+0.0015, Δcorr=+0.0230)

## Verdict on four mandatory criteria

- **C1_attention_beats_trivial_baselines** → `FAIL`
- **C2_incremental_value_over_gene_features** → `FAIL`
- **C3_residualized_retains_signal** → `FAIL`

**Overall:** FAIL — at least one mandatory criterion failed

## Scope

- dataset: `Replogle K562 non-targeting controls only`
- reference network: `TRRUST only (no STRING/Reactome/KEGG/GO)`
- N_ctrl: `2000`  N_hvg: `1500`
- curveball null iters: `50`

**Phases run:** 0a, 1, 2, 3, 4, 12
**Phases skipped:** 0b (value-weighted edges); 5 (cross-context replication — RPE1, Adamson, Dixit, Shifrut, Tian); 6 (CSSI); 7 (biological characterization — STRING, Reactome, KEGG, GO); 8 (detectability); 9 (ortholog transfer); 10 (pseudotime); 11 (batch leakage, calibration, TRRUST circularity, HVG protocol, mediation)