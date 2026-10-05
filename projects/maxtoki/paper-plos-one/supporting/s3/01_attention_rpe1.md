## Attention edges as gene-regulation scores (MaxToki-217M, RPE1 cells)

This analysis asks whether MaxToki-217M attention between two genes ranks gene pairs by regulation. It uses two endpoints. **Endpoint A (knockdown response):** for one knocked-down gene, which other genes change? **Endpoint B (TRRUST):** for one transcription factor (TF), which genes does the curated TRRUST database list as its targets? This section reports the RPE1 run: its stored data are raw counts, so the deployed code encoded them correctly. The K562 run with correct encoding is described in section 08. The deployed K562, Adamson and MaxToki-1B attention runs used wrongly encoded inputs and are not reported.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors (11 decoder blocks × 8 heads, hidden size 1,232, 216,946,576 parameters). float32, Apple MPS, eager attention so that attention weights are returned.
- **Cells.** Replogle RPE1 Perturb-seq data (`ReplogleWeissman2022_rpe1.h5ad`, 8,749 genes). `X` holds integer counts.
- **Cells for attention and gene statistics.** 2,000 non-targeting control cells, drawn with numpy `default_rng(42)` from 11,485 and sorted.
- **Gene set.** The 1,500 genes with the highest variance of log1p(CP10k) over the 2,000 control cells, among genes in the MaxToki vocabulary.
- **Model input.** Each cell's integer counts, encoded as in section 00: counts / gene median, sorted high to low, first 2,046 genes, `<bos>` and `<eos>` (max_len 2,048). Mean 2,039 tokens per cell.
- **Knockdown labels.** Perturbed RPE1 cells from the same file, against 5,000 control cells drawn with `default_rng(42)` from the 11,485.
- **TRRUST.** TRRUST human TF–target table (version 2). 16 TFs have at least 3 targets in the gene set: JUN, PARP1, PTTG1, HDAC2, HIPK2, KLF6, FOXM1, TFDP1, HIF1A, YY1, BRCA1, NFIC, DNMT1, JUND, CEBPB, ATF4. This gives 138 positive and 23,846 negative (TF, gene) pairs.

### Method

The attention edge averages, over control cells, how much one gene's token attends to another gene's token. Because MaxToki is a causal decoder, this average is partly set by gene order alone, so the method also builds order-only and gene-level baselines.

1. **Attention extraction.** One forward pass per control cell. For every pair (P, T) of genes from the gene set that are both in the cell, take the attention from the position of P (query) to the position of T (key), per layer and head. Sum over cells and divide by n_pair, the number of cells that contain both genes. Average over the 8 heads. The analysis layer is index 8 (0-based, of 11). It was chosen in advance by a depth rule, not by score.
2. **Split of the edge into order and attention.** In a causal decoder, P can attend to T only if T comes before P. Genes enter in order of falling counts / median. Let C[P, T] be the number of cells that contain both genes and put T before P.
   - Order share F[P, T] = C[P, T] / n_pair. It uses no network weights (only the tokenizer's gene medians).
   - Rank-conditioned edge Ec[P, T] = E[P, T] × n_pair / C[P, T], the mean attention over the cells where the mask allows it. Ec = 0 where C = 0.
   - So E = F × Ec exactly.
3. **Scores compared.**

   | Score | Definition | Uses the model? |
   |---|---|---|
   | forward | E[P, T], P = regulator or knocked-down gene (the pipeline's readout) | yes |
   | transpose | E[T, P] | yes |
   | symmetric mean | (E[P, T] + E[T, P]) / 2 | yes |
   | symmetric max | max(E[P, T], E[T, P]) | yes |
   | rank-conditioned | Ec[P, T] | yes |
   | order share F | C[P, T] / n_pair | no |
   | \|Spearman\| co-expression | absolute Spearman correlation over the same 2,000 control cells (reference) | no |
   | target variance | variance of the candidate gene's log1p(CP10k) in the 2,000 control cells | no |
   | 3-feature gene model | logistic regression on the candidate's mean, variance and dropout (TRRUST only) | no |

4. **Endpoint A labels.** A perturbation is used if it has ≥ 30 cells, its gene symbol is in the Geneformer symbol-to-Ensembl dictionary, and its gene is in the gene set. Expression is restricted to the 1,500 genes; each cell is scaled to 10,000 over those genes and log1p-transformed (one log, because `X` holds counts). A Welch t-test compares perturbed cells with the 5,000 controls for each gene, with Benjamini–Hochberg (BH) adjustment over the 1,500 genes. A gene is a positive if |difference of mean log1p values| ≥ 0.5 (natural log, not log2) and q < 0.05. The perturbed gene itself is removed. A perturbation is kept if it has ≥ 3 positives and ≥ 3 negatives: 325 of 434 candidates are kept, with 48,245 positive pairs (median 103 per perturbation).
5. **Endpoint A scoring.** Score = score[perturbed gene, candidate]. AUROC per perturbation, then the mean over perturbations. Interval: percentile bootstrap over perturbations (n = 325), 2,000 resamples. Paired Wilcoxon signed-rank test on the per-perturbation AUROCs (two-sided, `zero_method="zsplit"`), not adjusted for multiple testing.
6. **Endpoint A, added value of attention.** Logistic regression (standardised features, balanced class weights) on one row per (perturbation, candidate) pair. Gene-only features (candidate mean, variance, dropout) against the same plus forward attention. 5-fold GroupKFold grouped by perturbation. Δ = pooled out-of-fold AUROC difference. Interval: bootstrap over perturbations (n = 325), with the fitted fold models not refitted.
7. **Endpoint B scoring.** For each TF, the candidates are the other 1,499 genes; positive = TRRUST target, all other pairs negative. All (TF, candidate) pairs are pooled into one AUROC. Interval: percentile bootstrap over TFs (n = 16), 2,000 resamples; a TF drawn k times counts k times. The 3-feature gene model uses 5-fold GroupKFold grouped by TF; out-of-fold predictions are pooled; models are not refitted inside the bootstrap.
8. **Degree-preserving null for Endpoint B.** The network is the 0/1 matrix of the 16 TFs × 1,500 genes. A null network keeps each TF's number of targets and each gene's number of listing TFs. Curveball sampler: pick two TF rows, pool the targets that only one of them has, and deal them out again at random so each row keeps its count. A TF's own gene never gets an edge. 1,000 independent networks, each made by 5,000 trades from the real network. z = (observed − null mean) / null SD. p = (1 + number of null AUROCs ≥ observed) / 1,001, one-sided. Gene-level scores cannot be tested this way, because the null leaves their AUROC unchanged.
9. **Residualised scores (Endpoint B).** Each score over all 1,500 × 1,499 ordered pairs is predicted from gene-level features with 5-fold cross-fitting over pairs (KFold, shuffled, seed 42). The residual is scored on TRRUST and against the same 1,000 null networks. Three models: (a) ordinary least squares (OLS, float64) on mean, variance and dropout of both genes; (b) the main model, stated before looking: (a) plus F and an indicator for n_pair = 0 (for F itself, model (a)); (c) histogram gradient boosting (HGB) on the features of (b). A fourth check used HGB on the gene features only.
10. **Other layers (not pre-specified).** The forward edge at every layer 0–10 on both endpoints. Layer-adjusted test for Endpoint B: the statistic is the largest per-layer z; its null is the same maximum computed on each of the 1,000 null networks.

Seeds: master seed 20261001 with offsets (knockdown +100, TRRUST +500, Curveball +600, residual bootstrap +800); KFold seed 42.

### Checks

| Check | Result |
|---|---|
| Input is counts | Every non-zero value in the 2,000 control cells is a whole number (largest 1,491). On 55 RPE1 control cells, tokens built this way pass `inputs_v3.check_encoding` (55 / 55; section 00). |
| Gene order rebuilt | Rebuilt pair counts equal the stored pair counts in every entry (0 mismatches). Off the diagonal, C + Cᵀ = n_pair. |
| Order matches the stored attention | Over 2,248,494 off-diagonal pairs with n_pair > 0, the stored layer-8 edge is 0 exactly where C = 0 and positive exactly where C > 0 (0 mismatches each way). 6 pairs have n_pair = 0. C = 0 in 0.18% of pairs with n_pair > 0. |
| One log only | Gene variance on a single log scale equals the variance used (Spearman 1.0, largest difference 9.3 × 10⁻⁸). |
| Knockdown labels, two routes | The pipeline's per-perturbation AUROC file and a separate re-implementation of the DE test from the h5ad file give the same perturbations, the same positive counts and AUROCs equal within 2 × 10⁻¹⁶. Labels match the stored pair file in all 487,175 pairs. |
| AUROC code | A constant score gives exactly 0.5; tied scores match scikit-learn. |
| Null networks | Every network keeps all row and column counts and fills no forbidden cell. None equals the real network (mean Jaccard distance 0.877, smallest 0.82). The variance AUROC is unchanged in every network (largest change 0.0). The mean Jaccard distance reaches its plateau after about 100–500 trades. |
| Second null sampler | Checkerboard swaps (1,000 networks, 30 × number of edges accepted swaps each): forward edge null mean 0.574, z +1.76. |

### Results

**Main findings.**
- On knockdown response, target variance beats every attention score by 0.163 to 0.279 AUROC. The forward edge (0.603) is below the order share F (0.648), which uses no network weights.
- On TRRUST, the symmetric edges beat the degree-preserving null (z +3.19 and +3.02). Plain co-expression from the same cells does the same (z +3.04). The 3-feature gene model still scores higher (0.749).

**How much of the forward edge is gene order** (layer 8; all pairs with n_pair > 0; `order/order_check_217M_RPE1.json`):

| Measure | Value |
|---|---|
| Spearman(E, F), all pairs | 0.61 |
| Median within-row Spearman(E[P, ·], F[P, ·]) | 0.77 |
| R² of log E on log F (pairs with C > 0) | 0.38 |
| Median within-row Spearman of F[P, ·] with a target-only order score (how often T sits above any partner) | 0.86 |
| Spearman(Ec, F), all pairs | 0.005 |
| Median within-row Spearman(Ec[P, ·], F[P, ·]) | 0.46 |

So the order share explains 38% of the variance of the log edge. Within a row, F behaves like a property of the target gene alone. The rank-conditioned edge removes the cells where the mask forces 0, but it still follows gene order within a row (0.46). It is not order-free.

**Endpoint A, knockdown response** (n = 325 perturbations; intervals: percentile bootstrap over perturbations, 2,000 resamples; Wilcoxon p two-sided, not adjusted):

| Score | Mean per-perturbation AUROC [95% CI] | Score − forward [95% CI] | Variance − score [95% CI] | Wilcoxon p, variance vs score |
|---|---|---|---|---|
| forward | 0.603 [0.594, 0.612] | – | +0.163 [+0.154, +0.174] | 2.3 × 10⁻⁵² |
| transpose | 0.487 [0.477, 0.498] | −0.116 [−0.129, −0.103] | +0.279 [+0.265, +0.294] | 6.8 × 10⁻⁵⁴ |
| symmetric mean | 0.583 [0.572, 0.593] | −0.021 [−0.027, −0.014] | +0.184 [+0.172, +0.196] | 2.9 × 10⁻⁵² |
| symmetric max | 0.580 [0.571, 0.590] | −0.023 [−0.030, −0.016] | +0.186 [+0.175, +0.197] | 4.3 × 10⁻⁵³ |
| rank-conditioned | 0.560 [0.551, 0.570] | −0.043 [−0.048, −0.038] | +0.206 [+0.194, +0.218] | 6.5 × 10⁻⁵³ |
| order share F | **0.648 [0.637, 0.658]** | **+0.045 [+0.037, +0.052]** | +0.118 [+0.109, +0.129] | 1.1 × 10⁻⁴⁹ |
| \|Spearman\| co-expression | 0.622 [0.606, 0.636] | +0.019 [+0.004, +0.033] | +0.144 [+0.130, +0.160] | 2.2 × 10⁻⁴³ |
| target variance | **0.766 [0.757, 0.776]** | +0.163 [+0.154, +0.174] | – | – |

- Variance beats the forward edge in 314 of 325 perturbations.
- Among the attention scores, the forward edge is the best. Each other variant is below it, and every interval excludes 0.
- The order share F beats the forward edge by 0.045. So the knockdown signal of the forward edge is mostly gene order.
- Added value of forward attention over the three gene features: Δ = −0.0003 [−0.0005, −0.0001] (pooled out-of-fold AUROC 0.7477 without attention, 0.7475 with it; bootstrap over perturbations, n = 325, fitted models not refitted).

**Endpoint B, TRRUST: pooled AUROC and the degree-preserving null** (n = 16 TFs, 138 positive pairs; intervals: percentile bootstrap over TFs, 2,000 resamples; null: 1,000 Curveball networks):

| Score | Pooled AUROC [95% CI] | Null mean (SD) | z | p (one-sided) |
|---|---|---|---|---|
| forward | 0.605 [0.549, 0.655] | 0.574 (0.017) | +1.82 | 0.046 |
| transpose | 0.546 [0.453, 0.656] | 0.511 (0.015) | +2.32 | 0.008 |
| symmetric mean | 0.638 [0.578, 0.701] | 0.577 (0.019) | **+3.19** | 0.003 |
| symmetric max | 0.647 [0.579, 0.711] | 0.594 (0.018) | **+3.02** | 0.003 |
| rank-conditioned | 0.603 [0.541, 0.662] | 0.563 (0.018) | +2.29 | 0.013 |
| order share F | 0.552 [0.484, 0.616] | 0.551 (0.009) | +0.04 | 0.48 |
| \|Spearman\| co-expression (reference) | 0.596 [0.542, 0.650] | 0.541 (0.018) | +3.04 | 0.002 |
| target variance | 0.686 [0.633, 0.731] | not testable | – | – |
| 3-feature gene model | 0.749 [0.691, 0.801] | not testable | – | – |

Multiple testing: BH adjustment over the 12 network-null tests of the two correctly encoded cell lines (six scores here and the same six in K562, section 08) gives q = 0.018 for both symmetric edges, 0.032 for the transpose, 0.039 for the rank-conditioned edge, 0.069 for the forward edge and 0.53 for the order share (empirical p from 1,000 rewired networks; the smallest possible p is 0.001, so the symmetric p of 0.003 carries Monte Carlo noise: three separate null samplers gave z between +3.0 and +3.3). A second statistic, the mean of per-TF AUROCs, gives z = +0.75 for the forward edge.

**Endpoint B, differences** (n = 16 TFs; percentile bootstrap over TFs, 2,000 resamples):

| Score | Score − forward [95% CI] | Score − co-expression [95% CI] | Gene model − score [95% CI] | Variance − score [95% CI] |
|---|---|---|---|---|
| forward | – | +0.008 [−0.086, +0.085] | +0.145 [+0.055, +0.238] | +0.082 [+0.028, +0.145] |
| transpose | −0.058 [−0.184, +0.061] | −0.050 [−0.151, +0.069] | +0.203 [+0.045, +0.317] | +0.140 [+0.026, +0.246] |
| symmetric mean | +0.034 [−0.015, +0.091] | +0.042 [−0.040, +0.127] | +0.111 [+0.002, +0.202] | +0.048 [−0.021, +0.110] |
| symmetric max | +0.043 [+0.007, +0.086] | +0.051 [−0.041, +0.139] | +0.102 [−0.009, +0.204] | +0.039 [−0.024, +0.105] |
| rank-conditioned | −0.001 [−0.038, +0.041] | +0.007 [−0.089, +0.092] | +0.146 [+0.040, +0.243] | +0.083 [+0.012, +0.155] |
| order share F | −0.053 [−0.123, +0.010] | −0.045 [−0.137, +0.041] | +0.198 [+0.134, +0.272] | +0.135 [+0.058, +0.212] |
| \|Spearman\| co-expression | −0.008 [−0.085, +0.086] | – | +0.153 [+0.080, +0.220] | +0.090 [+0.025, +0.156] |

- Direction matters on TRRUST. The symmetric edges score above the forward edge; the gap is clear only for symmetric max (+0.043 [+0.007, +0.086]).
- Symmetric attention is about as good as co-expression, not clearly better: the interval for symmetric mean minus co-expression reaches +0.127.
- The gene model is ahead of symmetric attention by about 0.10, with the lower bound near 0. Target variance alone ties the symmetric edges.
- For the forward edge, variance is higher in 13 of 16 TFs and the gene model in 15 of 16.
- F carries no TRRUST signal (0.552, null z +0.04).

**Endpoint B, residualised scores** (n = 16 TFs; percentile bootstrap over TFs, 2,000 resamples, fitted models not refitted; z against the same 1,000 null networks):

| Score | (a) OLS, gene features: AUROC [95% CI]; z | (b) OLS, gene features + F (main): AUROC [95% CI]; z; p | (c) HGB, gene features + F: AUROC [95% CI]; z; p |
|---|---|---|---|
| forward | 0.492 [0.404, 0.568]; +1.44 | 0.511 [0.425, 0.593]; +2.15; 0.010 | 0.593 [0.521, 0.651]; +2.08; 0.017 |
| transpose | 0.537 [0.456, 0.626]; +2.56 | 0.552 [0.487, 0.616]; +3.04; 0.001 | 0.566 [0.475, 0.662]; +2.19; 0.013 |
| symmetric mean | 0.538 [0.456, 0.614]; +2.75 | 0.538 [0.456, 0.614]; +2.75; 0.003 | 0.603 [0.540, 0.666]; +2.97; 0.002 |
| symmetric max | 0.536 [0.445, 0.615]; +3.10 | 0.536 [0.445, 0.615]; +3.10; 0.002 | 0.606 [0.524, 0.676]; +2.70; 0.003 |
| rank-conditioned | 0.497 [0.408, 0.575]; +1.82 | 0.501 [0.413, 0.582]; +1.97; 0.028 | 0.574 [0.507, 0.634]; +2.00; 0.029 |
| order share F | 0.495 [0.418, 0.566]; +0.21 | (model (a) is used for F) | HGB on gene features: 0.464 [0.407, 0.528]; +0.04; 0.47 |
| \|Spearman\| co-expression | 0.487 [0.436, 0.532]; +2.94 | 0.487 [0.436, 0.532]; +2.94; 0.002 | not run |

- F is antisymmetric (F[P, T] = 1 − F[T, P]), so adding it changes nothing for the symmetric edges.
- HGB on the gene features only, forward edge: 0.559 [0.488, 0.617].
- Against 0.5, no OLS residual is above chance: every interval includes 0.5.
- Against the degree null, the transpose and symmetric residuals stay above it (z +2.75 to +3.10). This happens because the null mean falls to 0.476–0.497 after residualising (co-expression: residual 0.487, null mean 0.425). These residual tests were not adjusted for multiple testing within this run.
- So the word "chance" must be defined. "No better than 0.5" holds for every residual here. "No better than the degree null" does not hold for the transpose and symmetric residuals.

**Other layers, forward edge only (not pre-specified).** Knockdown: mean per-perturbation AUROC (n = 325; point estimates). TRRUST: pooled AUROC, gap to the 3-feature gene model (percentile bootstrap over TFs, n = 16, 2,000 resamples), and z and p against the same 1,000 null networks (per-layer p not adjusted).

| Layer | Knockdown AUROC | TRRUST AUROC | Gene model − attention [95% CI] | Null z | p |
|---|---|---|---|---|---|
| 0 | 0.682 | 0.697 | +0.052 [−0.010, +0.121] | 3.35 | 0.002 |
| 1 | 0.707 | 0.758 | −0.009 [−0.072, +0.053] | 1.06 | 0.15 |
| 2 | 0.684 | 0.748 | +0.001 [−0.045, +0.054] | 2.09 | 0.017 |
| 3 | 0.605 | 0.630 | +0.120 [+0.025, +0.202] | 2.71 | 0.003 |
| 4 | 0.645 | 0.679 | +0.070 [+0.003, +0.145] | 2.66 | 0.005 |
| 5 | 0.602 | 0.637 | +0.113 [+0.026, +0.208] | 2.34 | 0.009 |
| 6 | 0.602 | 0.649 | +0.100 [+0.032, +0.179] | 2.74 | 0.002 |
| 7 | 0.557 | 0.609 | +0.140 [+0.051, +0.219] | 2.40 | 0.008 |
| 8 | 0.603 | 0.605 | +0.145 [+0.063, +0.241] | 1.82 | 0.046 |
| 9 | 0.531 | 0.563 | +0.186 [+0.111, +0.273] | 2.20 | 0.011 |
| 10 | 0.621 | 0.651 | +0.099 [+0.026, +0.179] | 2.16 | 0.020 |

- Target variance (0.766) beats the forward edge on knockdown at every layer.
- Layer-adjusted TRRUST test: largest z = 3.35 at layer 0, p = 0.003 (1,000 null networks). At layer 0 the gene model is level with attention (+0.052 [−0.010, +0.121]).

**Relation to co-expression (computed during verification; see below).**
- Over all ordered gene pairs, symmetric-mean attention and |Spearman| co-expression barely correlate (Spearman 0.10).
- Symmetric mean still beats the degree null after co-expression is removed: 0.611 with a linear fit (z +2.53) and 0.620 with 50 co-expression bins (z +2.94). Symmetric max: 0.623 (z +2.50) and 0.635 (z +3.04).
- Co-expression still beats it after symmetric mean is removed (50 bins): 0.563, z +2.19.
- A null test of the difference "symmetric − co-expression" gives z +0.22 (mean) and −0.04 (max).
- So symmetric attention holds a second, separate signal of about the same size as co-expression, not a larger one.
- A score with no model, −|mean(TF) − mean(gene)|, reaches z +1.81 (p 0.041), the same as the layer-8 forward edge. Co-expression reaches z +3.09 (p 0.003) under the same sampler. Passing the degree null therefore does not by itself show knowledge specific to the model.
- Leave-one-TF-out: symmetric mean stays above the null when any one of the 16 TFs is dropped (z 2.37 to 3.58, largest p 0.009); symmetric max z 2.12 to 3.46 (largest p 0.020). The forward edge drops to z 0.93 without HIF1A.

### Verification

Three other agents re-checked this analysis with their own scripts. None of the scripts imports the main analysis code.

- **Base evaluation** (`v2verify_01_endpointA.py` to `v2verify_10_run_config.py`; outputs in `outputs/v2_eval/verification/`). Per-perturbation AUROCs equal the main files within 2.2 × 10⁻¹⁶; perturbation and positive counts and the 314 / 325 count match. Own perturbation bootstrap (2,000 resamples, other seed): variance − forward +0.163 [+0.153, +0.173]. TRRUST counts and AUROCs match; own TF bootstrap CIs within 0.01. A leave-one-TF-out gene model gives a gap of +0.145 to the forward edge, with the interval above 0. Own boolean-matrix Curveball sampler (1,000 networks × 1,000 trades, seed 23): z +1.73, p 0.047. Layer-adjusted p 0.005 (main: 0.003). It confirmed that the RPE1 file holds integer counts and that float32 and float64 OLS agree for this run. It added the co-expression null (z +3.09) and the two-gene-means score (z +1.81).
- **Score variants, first check** (`v2b_attention_verify.py`, seed 777001; outputs in `outputs/v2b_attention/verification/`). Own rank code rebuilt C from the h5ad file with 0 mismatches; R² 0.381 and within-row Spearman 0.77 and 0.86 reproduce. Knockdown AUROCs equal within 2.2 × 10⁻¹⁶; F − forward +0.045 [+0.037, +0.052]. TRRUST AUROCs equal. Own Curveball (1,000 networks × 3,000 trades): z forward +1.82, transpose +2.27, symmetric mean +3.17, symmetric max +2.97, rank-conditioned +2.26, F +0.04, co-expression +2.96. Own OLS residuals match to 3 decimals (transpose 0.552, z +3.02; symmetric mean 0.538, z +2.80; symmetric max 0.536, z +3.06). HGB spot checks: symmetric mean 0.601 [0.542, 0.661], z +2.86; transpose 0.571, z +2.19. With its own random TF-to-fold map the gene model is 0.753 and its gap to symmetric max is +0.106 [+0.008, +0.207] instead of +0.102 [−0.009, +0.204]. So "ties" against "just beats" depends on the fold map. All 205 rows of `fig_attention_variants.csv` equal the JSON values.
- **Score variants, second check** (`v2b_attention_verify2.py`, seed 50505; outputs in `outputs/v2b_attention/verification2/`). It checked C against its invariants (0 ≤ C ≤ n_pair, C + Cᵀ = n_pair, edge = 0 exactly where C = 0). Knockdown AUROCs equal for all 9 scores; F − forward +0.045 [+0.037, +0.052]. TRRUST AUROCs equal exactly; TF bootstrap CIs within 0.007. A different null sampler (checkerboard swaps, 1,000 networks, 20 × number of edges accepted swaps): z forward +1.80, transpose +2.38, symmetric mean +3.26, symmetric max +3.08, rank-conditioned +2.30, F +0.05, co-expression +3.02. Residual AUROCs equal within 0.001 (residual z: transpose +3.11, symmetric mean +2.87, symmetric max +3.15). It also ran the co-expression and leave-one-TF-out analyses above, found that the rank-conditioned edge still follows gene order, and found that target mean expression is below F on knockdown (−0.066 [−0.077, −0.054]).

All three agreed with every number they recomputed. Their extra findings are reported in the Results above.

### Limits

- One cell line (RPE1), one model size (217M) and one pre-specified layer (8). Other layers were scored for the forward edge only. Residualisation was run at layer 8 only. The early-layer TRRUST result was not compared with co-expression.
- TRRUST is small: 16 TFs and 138 positive pairs. The TF bootstrap treats TFs as independent, although the same candidate genes recur across TFs, so intervals are somewhat too narrow. Intervals for the gene model and the residuals keep the fitted models as they are, which also makes them too narrow.
- Only TRRUST was used as a curated reference. DoRothEA, STRING and ChIP-seq references were not tested.
- Endpoint A was not residualised and has no null. The added-value test was run for the forward edge only, not with F as a covariate.
- Only |Spearman| co-expression from the same 2,000 control cells was used as a co-expression baseline.
- Attention was averaged over heads. Single heads were not tested.
- The rank-conditioned edge is set to 0 where T never comes before P (0.18% of pairs). No other fill rule was tried.

### Files

- Attention extraction: `projects/maxtoki/runs/attention-grn-217M/scripts/phase0_rpe1.py` → `projects/maxtoki/runs/attention-grn-217M/outputs/phase0_rpe1/` (`attention_edges_layer_mean.npy`, `attention_edges_per_head.npy`, `attention_pair_counts.npy`, `control_cells.csv`, `gene_features.csv`, `spearman_edges.npy`, `run_config.json`). Per-layer knockdown AUROCs: `.../outputs/phase1_rpe1/attention_per_layer_auroc.csv`.
- Evaluation of the forward edge: `.../scripts/v2_common.py` and `v2_01_facts.py` to `v2_11_run_config.py` → `.../outputs/v2_eval/` (`run_facts.json`, `knockdown/knockdown_summary.json`, `knockdown/per_perturbation_217M_RPE1.csv`, `knockdown/incremental_217M_RPE1.json`, `trrust/`, `curveball/curveball_summary.json`, `residualised/residualised_summary.json`, `layers/per_layer_trrust.csv`, `layers/layers_summary.json`, `fig_knockdown.csv`, `fig_trrust.csv`, `run_config.json`).
- Score variants: `.../scripts/v2b_attention.py` → `.../outputs/v2b_attention/` (`order/order_check_217M_RPE1.json`, `order/order_counts_217M_RPE1.npy`, `knockdown/knockdown_variants_summary.json`, `knockdown/per_perturbation_217M_RPE1.csv`, `trrust/trrust_variants_217M_RPE1.json`, `trrust/per_tf_217M_RPE1.csv`, `trrust/gene_model_gaps_217M_RPE1.json`, `trrust/null_positions_217M_RPE1.npy`, `resid/resid_ols_217M_RPE1.json`, `resid/resid_hgb_217M_RPE1_*.json`, `resid/resid_summary.json`, `table_attention_variants.csv`, `fig_attention_variants.csv`, `run_config.json` with sha256 of inputs, scripts and outputs).
- Verification: `.../scripts/v2verify_01_endpointA.py` to `v2verify_10_run_config.py`, `v2b_attention_verify.py`, `v2b_attention_verify2.py`; outputs in `.../outputs/v2_eval/verification/`, `.../outputs/v2b_attention/verification/`, `.../outputs/v2b_attention/verification2/`.
- The CSV and JSON summary files hold rows for four runs; the rows of this analysis have run = `217M_RPE1`.
