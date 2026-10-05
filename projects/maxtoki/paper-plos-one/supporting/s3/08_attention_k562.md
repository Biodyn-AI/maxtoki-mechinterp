## Attention edges as gene-regulation scores (MaxToki-217M, K562 cells)

This section applies the attention analysis of section 01 to K562 cells. The two endpoints, the scores, the degree-preserving null, the residual models and the seeds are the same as in section 01. The cells, the gene set and the knockdown labels are those of K562. All numbers are for layer 8 unless a table says otherwise. **Endpoint A (knockdown response):** for one knocked-down gene, which other genes change? **Endpoint B (TRRUST):** for one transcription factor (TF), which genes does TRRUST list as its targets? AUROC 0.5 is chance.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, as in section 01. float32, Apple MPS, eager attention so that attention weights are returned. torch 2.11.0, transformers 5.5.4. Analysis on CPU (numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 2.3.3).
- **Data file.** Replogle K562 Perturb-seq data (`replogle_concat.h5ad`, 6,546 genes). `X` holds log1p(CP10k).
- **Cells for attention and gene statistics.** 2,000 non-targeting control cells, drawn with numpy `default_rng(42)` from 10,691 and sorted. These are the same rows as in the deployed attention analysis.
- **Counts.** Recovered from `X` as in section 00: round(expm1(X) / u), with u = the cell's smallest non-zero expm1(X) (`inputs_v3.CountReader("k562")` with `counts_accept_unit_one`).
- **Gene set.** log1p(counts / row sum × 10,000), with the row sum over the file's 6,546 genes (one log). The 1,500 genes with the highest variance over the 2,000 cells, among the 6,332 file genes in the MaxToki vocabulary. Gene features (mean, variance, dropout rate) and |Spearman| co-expression use the same values and the same 2,000 cells.
- **Model input.** Each cell's counts, encoded as in section 00: counts / total × 10,000 / gene median, sorted high to low, first 2,046 genes, `<bos>` and `<eos>` (max_len 2,048). Mean 2,039 tokens per cell (smallest 1,016). 1,946 of 2,000 cells are cut at 2,046 genes. A pair of two different genes is held by a median of 648 cells.
- **Knockdown labels.** Perturbed K562 cells from the same file, against 5,000 control cells drawn with `default_rng(42)` from the 10,691.
- **TRRUST.** TRRUST human TF–target table (version 2, `trrust_human.tsv`). 16 TFs have at least 3 targets in the gene set: HDAC1, JUN, NFE2L2, PTTG1, EZH2, MYC, TFDP1, CHD8, YY1, CTCF, BRCA1, DNMT1, JUND, CEBPB, XBP1, ATF4. This gives 107 positive and 23,877 negative (TF, gene) pairs. MYC alone has 27 of the 107 positives.

### Method

1. **Attention extraction.** One forward pass per control cell (`LlamaForCausalLM`). A forward hook on each layer's `self_attn` module takes the attention weights (the same tensor as `output_attentions[l]`), averages the 8 heads, and adds the gene × gene block to a running sum. Cells run in blocks of 25. `inputs_v3.assert_encoding_batch` checks every block before its forward passes. Free memory was checked before every block (stop below 3 GB). The sum covers 2,373 genes: the 1,500 genes of the gene set plus the gene set of the deployed analysis, which is used only in the secondary analysis below. All 11 layers are kept. Layer 8 (0-based) is the pre-specified layer, as in section 01.
2. **Edges.** E[P, T] = summed attention from the position of P (query) to the position of T (key), divided by n_pair, the number of cells that hold both genes. Cells in which the causal mask forces the attention to 0 are included. C[P, T] = the number of these cells that put T before P, counted in the same loop. Order share F = C / n_pair. Rank-conditioned edge Ec = E × n_pair / C (0 where C = 0). So E = F × Ec.
3. **Scores.** The nine scores of section 01: forward E[P, T]; transpose E[T, P]; symmetric mean and symmetric max of the two; rank-conditioned Ec[P, T]; order share F (no network weights); |Spearman| co-expression (reference, no model); target variance (no model); 3-feature gene model (TRRUST only, no model).
4. **Endpoint A labels (pre-specified rule of section 01).** A perturbation is a candidate if it has ≥ 30 cells, its symbol is in the Geneformer symbol-to-Ensembl dictionary and its gene is in the gene set. expm1(X) is restricted to the 1,500 genes, each cell is scaled to 10,000 over those genes, then log1p (one log). Welch t-test of perturbed cells against the 5,000 controls for each gene, BH over the 1,500 genes. A gene is a positive if |difference of mean log1p values| ≥ 0.5 and q < 0.05. The perturbed gene itself is removed. A perturbation is kept if it has ≥ 3 positives and ≥ 3 negatives: 194 of 446 candidates, with 8,714 positive pairs (median 17 per perturbation). This rule was chosen before looking at results, so that both cell lines use the same rule.
5. **Endpoint A, sensitivity labels.** The same test on the stored log1p(CP10k) values (normalised over the whole file, no rescaling over the 1,500 genes). 67 perturbations are kept, with 720 positive pairs (median 7).
6. **Endpoint A scoring.** Score = score[perturbed gene, candidate]. AUROC per perturbation (average ranks for ties), then the mean over perturbations. Interval: percentile bootstrap over perturbations (n = 194), 2,000 resamples. Paired Wilcoxon signed-rank test of target variance against each of the six model and order scores (two-sided, `zero_method="zsplit"`), BH over these six tests.
7. **Endpoint B scoring.** For each TF, the candidates are the other 1,499 genes. All (TF, candidate) pairs are pooled into one AUROC. Interval: percentile bootstrap over TFs (n = 16), 2,000 resamples; a TF drawn k times counts k times. The 3-feature gene model is a logistic regression (standardised features, balanced class weights) on the candidate's mean, variance and dropout rate, with 5-fold GroupKFold grouped by TF and out-of-fold predictions pooled. Its fitted models are not refitted inside the bootstrap. Fold-map check: the same model with a random TF-to-fold map (seed 7).
8. **Degree-preserving null for Endpoint B.** The Curveball sampler of section 01 (`v2_06_curveball.py`): 1,000 networks, each made by 5,000 trades from the real 16 × 1,500 network. Each network keeps every TF's number of targets and every gene's number of listing TFs. z = (observed − null mean) / null SD. p = (1 + number of null AUROCs ≥ observed) / 1,001, one-sided. Multiple testing: BH over the six model and order scores in K562, and a joint BH over 12 tests (these six and the same six scores in RPE1, section 01). Gene-level scores cannot be tested this way, because the null leaves their AUROC unchanged.
9. **Residualised scores (Endpoint B).** Each score over all 1,500 × 1,499 ordered pairs is predicted from gene-level features with 5-fold cross-fitting over pairs (KFold, shuffled, seed 42). The residual is scored on TRRUST and against the same 1,000 null networks. Three models, as in section 01: (a) ordinary least squares (OLS, float64) on mean, variance and dropout rate of both genes; (b) the main model, stated in advance: (a) plus F (for F itself, model (a)); (c) histogram gradient boosting (HGB; 200 iterations, learning rate 0.1, 31 leaves) on the features of (b), and on the gene features only for F. No pair has n_pair = 0, so the n_pair = 0 indicator of section 01 would be a constant and is left out. BH over the six scores within each model. Interval: percentile bootstrap over TFs (n = 16), 2,000 resamples, fitted models not refitted.
10. **Relation to co-expression (Endpoint B).** Spearman correlation between each attention score and co-expression over all ordered gene pairs. Attention after removing co-expression (linear fit, or the mean within 50 co-expression quantile bins), and co-expression after removing attention, each scored against the same null networks. A null test of the difference "attention AUROC − co-expression AUROC" on the same networks.
11. **Leave one TF out.** Pooled AUROC and null z with each of the 16 TFs dropped in turn (same null networks).
12. **Other layers (not pre-specified).** Forward and symmetric-mean edges at layers 0–10 on both endpoints. Layer-adjusted TRRUST test: the statistic is the largest z over the 11 layers; its null is the same maximum on each of the 1,000 null networks.
13. **Secondary analysis (not pre-specified).** The same attention sums read on the gene set of the deployed analysis (see Results).

Seeds: master seed 20261001 with offsets (knockdown bootstrap +100, TRRUST bootstrap +500, Curveball +600, residual bootstrap +800); KFold 42; HGB 20261001; loss bootstrap 20261002.

### Checks

| Check | Result |
|---|---|
| Same cells as the deployed analysis | 2,000 of 2,000 rows equal `outputs/phase0/control_cells.csv` |
| Counts are counts | In 2,000 of 2,000 cells every non-zero expm1(X) is a whole-number multiple of the cell's smallest value (largest deviation 3.7 × 10⁻⁴; multiples up to 1,110). The library size implied by the recovered counts is 0.956–0.994 of the file's `UMI_count` (median 0.987). All 86,056 rows read for the knockdown labels pass the same test (largest deviation 4.7 × 10⁻⁴). |
| Gene set stable to arithmetic | A float32 route gives the same 1,500 genes. A separate route that uses expm1(X) without rounding gives the same set, with features within 4.2 × 10⁻⁹. Variance at rank 1,500 and rank 1,501 differs by 0.04%. |
| Encoding check at tokenisation | 2,000 of 2,000 cells pass `inputs_v3.check_encoding`; 0 order violations; 240 positions sit inside exact ties (either order allowed) |
| Encoding check before forward passes | 80 blocks, 2,000 of 2,000 cells pass |
| Negative control | The token order of the deployed analysis, ranked from the stored log values, fails `check_encoding` in 2,000 of 2,000 cells |
| Model works on these inputs | Next-gene loss 3.09 nats per gene token (95% CI 3.08–3.10; bootstrap over cells, n = 2,000, 10,000 resamples); top-1 next-gene accuracy 0.207. Five K562 control cells of the encoding tests (section 00) give 3.08. |
| Causal mask | Layer-8 attention above the diagonal is 0.0 in every cell |
| Zero pattern | E > 0 exactly where C > 0 at all 11 layers, 0 mismatches either way (5,628,746 off-diagonal pairs with n_pair > 0 in the 2,373-gene union; 10 union pairs have n_pair = 0). No pair of the 1,500 genes has n_pair = 0. C = 0 in 0.08% of the pairs of the 1,500 genes. |
| Counts consistent | Off the diagonal C + Cᵀ = n_pair; n_pair equals PᵀP computed from the saved tokens |
| Null networks | 1,000 networks. None breaks a row or column count or fills a forbidden cell. None equals the real network (mean Jaccard distance 0.86, smallest 0.79). The variance AUROC is unchanged in every network (largest change 0.0). |
| AUROC code | A constant score gives exactly 0.5; tied scores match scikit-learn |
| Deployed gene set rebuilt (for the secondary analysis) | The deployed variance rule applied to the stored `X` gives exactly the deployed 1,500 genes |

### Results

**Main findings.**
- On knockdown response with the main labels, every layer-8 attention score is at chance: 0.499 to 0.509, and every interval includes 0.5. Target variance scores 0.690 and beats the forward edge by 0.185 [+0.162, +0.208]. Co-expression (0.556) beats every attention score. With the sensitivity labels (67 perturbations) the forward edge scores 0.550 [0.517, 0.580]: above chance, but below F (0.611) and target variance (0.695). Under both label rules, attention is no better than F.
- On TRRUST, the forward edge (0.548, z +1.72) and the symmetric mean (0.571, z +1.76) do not pass the degree-preserving null after BH (q 0.082 over the six K562 tests; q 0.069 over the 12 tests of both cell lines). In the same 12-test family the two RPE1 symmetric scores have q 0.018. So K562 does not reproduce the RPE1 result. The forward z depends on one TF (MYC).
- The 3-feature gene model (0.723) beats every attention score. After gene-level features are removed, no score passes the null (lowest q 0.17).
- The order share explains 41% of the variance of the log edge, close to RPE1 (38%).

**How much of the forward edge is gene order** (layer 8; `order/order_check.json`; RPE1 from section 01):

| Measure | K562 | RPE1 |
|---|---|---|
| Spearman(E, F), all pairs with n_pair > 0 | 0.60 | 0.61 |
| Median within-row Spearman(E[P, ·], F[P, ·]) | 0.76 | 0.77 |
| R² of log E on log F (pairs with C > 0) | 0.41 | 0.38 |
| Median within-row Spearman of F[P, ·] with a target-only order score | 0.88 | 0.86 |
| Spearman(Ec, F), all pairs | −0.07 | 0.005 |
| Median within-row Spearman(Ec[P, ·], F[P, ·]) | 0.37 | 0.46 |
| Pairs with n_pair > 0 and C = 0 | 0.08% | 0.18% |

The rank-conditioned edge removes the cells where the mask forces 0, but within a row it still follows gene order (0.37). It is not order-free.

**Endpoint A, knockdown response** (n = 194 perturbations; intervals: percentile bootstrap over perturbations, 2,000 resamples; `knockdown/knockdown_summary.json`):

| Score | Mean per-perturbation AUROC [95% CI] | Score − forward [95% CI] | Variance − score [95% CI] | Co-expression − score [95% CI] | Perturbations with score > variance | BH q, variance vs score (Wilcoxon) |
|---|---|---|---|---|---|---|
| forward | 0.505 [0.491, 0.520] | – | +0.185 [+0.162, +0.208] | +0.051 [+0.028, +0.073] | 21 | 2.8 × 10⁻²⁸ |
| transpose | 0.509 [0.495, 0.524] | +0.005 [−0.017, +0.026] | +0.181 [+0.158, +0.203] | +0.046 [+0.026, +0.067] | 17 | 7.4 × 10⁻²⁹ |
| symmetric mean | 0.502 [0.488, 0.517] | −0.002 [−0.014, +0.010] | +0.188 [+0.164, +0.211] | +0.053 [+0.033, +0.075] | 19 | 1.7 × 10⁻²⁸ |
| symmetric max | 0.500 [0.486, 0.514] | −0.005 [−0.019, +0.010] | +0.190 [+0.166, +0.214] | +0.055 [+0.035, +0.076] | 16 | 5.2 × 10⁻²⁹ |
| rank-conditioned | 0.499 [0.484, 0.514] | −0.006 [−0.016, +0.004] | +0.191 [+0.168, +0.213] | +0.057 [+0.034, +0.078] | 16 | 3.4 × 10⁻²⁹ |
| order share F | 0.513 [0.496, 0.531] | +0.009 [−0.004, +0.020] | +0.177 [+0.150, +0.203] | +0.042 [+0.018, +0.066] | 34 | 9.3 × 10⁻²⁵ |
| \|Spearman\| co-expression | 0.556 [0.540, 0.573] | +0.051 [+0.028, +0.073] | +0.134 [+0.114, +0.153] | – | 23 | not in the BH family |
| target variance | **0.690 [0.674, 0.708]** | +0.185 [+0.162, +0.208] | – | – | – | – |

- Variance beats the forward edge in 173 of 194 perturbations.
- No attention variant differs from the forward edge: every "score − forward" interval includes 0. The order share F also ties the forward edge.
- Target variance computed on the stored log1p(CP10k) values gives 0.695 [0.679, 0.712].

**Endpoint A, sensitivity labels** (stored log1p(CP10k) values; n = 67 perturbations, 720 positive pairs; same interval method):

| Score | Mean per-perturbation AUROC [95% CI] |
|---|---|
| forward | 0.550 [0.517, 0.580] |
| symmetric mean | 0.532 [0.497, 0.569] |
| order share F | 0.611 [0.568, 0.651] |
| \|Spearman\| co-expression | 0.553 [0.515, 0.591] |
| target variance | 0.695 [0.659, 0.730] |

Variance − forward: +0.145 [+0.089, +0.200]. F − forward: +0.061 [+0.034, +0.087]. The verdict is the same as with the main labels: variance is far ahead, and attention is no better than F.

**Endpoint B, TRRUST: pooled AUROC and the degree-preserving null** (n = 16 TFs, 107 positive pairs; intervals: percentile bootstrap over TFs, 2,000 resamples; null: 1,000 Curveball networks; `trrust/trrust_summary.json`, `trrust/gene_model_gaps.json`):

| Score | Pooled AUROC [95% CI] | Null mean (SD) | z | p (one-sided) | BH q, 6 K562 tests | BH q, 12 tests (K562 + RPE1) |
|---|---|---|---|---|---|---|
| forward | 0.548 [0.499, 0.594] | 0.517 (0.018) | +1.72 | 0.038 | 0.082 | 0.069 |
| transpose | 0.526 [0.446, 0.593] | 0.528 (0.019) | −0.09 | 0.55 | 0.55 | 0.55 |
| symmetric mean | 0.571 [0.521, 0.617] | 0.534 (0.021) | +1.76 | 0.041 | 0.082 | 0.069 |
| symmetric max | 0.538 [0.505, 0.574] | 0.514 (0.022) | +1.11 | 0.14 | 0.17 | 0.17 |
| rank-conditioned | 0.532 [0.477, 0.594] | 0.499 (0.019) | +1.68 | 0.038 | 0.082 | 0.069 |
| order share F | 0.537 [0.455, 0.611] | 0.528 (0.009) | +1.08 | 0.14 | 0.17 | 0.17 |
| \|Spearman\| co-expression (reference) | 0.588 [0.533, 0.691] | 0.551 (0.024) | +1.55 | 0.064 | not in the family | not in the family |
| target variance | 0.642 [0.579, 0.714] | not testable | – | – | – | – |
| 3-feature gene model | **0.723 [0.644, 0.814]** | not testable | – | – | – | – |

**The 12-test family** (p from 1,000 rewired networks; the smallest possible p is 0.001; RPE1 p and z from section 01):

| Score | K562 z; p; q | RPE1 z; p; q |
|---|---|---|
| forward | +1.72; 0.038; 0.069 | +1.82; 0.046; 0.069 |
| transpose | −0.09; 0.55; 0.55 | +2.32; 0.008; 0.032 |
| symmetric mean | +1.76; 0.041; 0.069 | +3.19; 0.003; **0.018** |
| symmetric max | +1.11; 0.14; 0.17 | +3.02; 0.003; **0.018** |
| rank-conditioned | +1.68; 0.038; 0.069 | +2.29; 0.013; 0.039 |
| order share F | +1.08; 0.14; 0.17 | +0.04; 0.48; 0.53 |

No K562 test goes below q 0.069. Four RPE1 tests are below q 0.05.

**Endpoint B, differences** (n = 16 TFs; percentile bootstrap over TFs, 2,000 resamples; gene-model intervals keep the fitted models fixed):

| Score | Score − forward [95% CI] | Score − co-expression [95% CI] | Gene model − score [95% CI] | Variance − score [95% CI] | TFs with score > variance |
|---|---|---|---|---|---|
| forward | – | −0.040 [−0.161, +0.037] | +0.175 [+0.084, +0.287] | +0.094 [+0.016, +0.187] | 5 |
| transpose | −0.022 [−0.125, +0.071] | −0.062 [−0.186, +0.024] | +0.197 [+0.080, +0.341] | +0.116 [+0.005, +0.238] | 6 |
| symmetric mean | +0.023 [−0.034, +0.076] | −0.017 [−0.130, +0.060] | +0.152 [+0.050, +0.269] | +0.071 [−0.017, +0.170] | 7 |
| symmetric max | −0.010 [−0.058, +0.040] | −0.050 [−0.152, +0.015] | +0.186 [+0.107, +0.280] | +0.104 [+0.028, +0.184] | 5 |
| rank-conditioned | −0.016 [−0.061, +0.035] | −0.057 [−0.159, +0.022] | +0.192 [+0.108, +0.285] | +0.111 [+0.011, +0.198] | 4 |
| order share F | −0.011 [−0.097, +0.055] | −0.051 [−0.212, +0.058] | +0.186 [+0.054, +0.346] | +0.105 [+0.028, +0.210] | 5 |
| \|Spearman\| co-expression | +0.040 [−0.037, +0.161] | – | +0.135 [+0.071, +0.214] | +0.054 [−0.072, +0.141] | 8 |

- The gene model beats every score; every interval is above 0.
- Target variance alone beats every attention score except the symmetric mean, which it ties (+0.071 [−0.017, +0.170]).
- The symmetric edges do not score above the forward edge (mean +0.023 [−0.034, +0.076]; max −0.010 [−0.058, +0.040]).
- Attention and co-expression tie.
- With a random TF-to-fold map (seed 7) the gene model scores 0.691 instead of 0.723. The gene-model gaps would then be about 0.03 smaller.

**Leave one TF out** (`trrust/leave_one_tf_out.json`; same null networks):

| Score | z range when one TF is dropped | TF whose removal gives the lowest z |
|---|---|---|
| forward | +0.12 to +2.19 | MYC |
| symmetric mean | +1.11 to +2.25 | MYC |
| symmetric max | +0.65 to +1.42 | ATF4 |
| \|Spearman\| co-expression | +0.92 to +1.87 | ATF4 |

The forward result rests on MYC: without it, z falls from +1.72 to +0.12.

**Endpoint B, residualised scores** (n = 16 TFs; percentile bootstrap over TFs, 2,000 resamples, fitted models not refitted; z against the same 1,000 null networks; q = BH over the six scores within each model; `resid/resid_summary.json`):

| Score | (a) OLS, gene features: AUROC [95% CI]; z | (b) OLS, gene features + F (main): AUROC [95% CI]; null mean; z; q | (c) HGB, gene features + F: AUROC [95% CI]; z; q |
|---|---|---|---|
| forward | 0.529 [0.479, 0.577]; +1.48 | 0.487 [0.408, 0.579]; 0.467; +1.08; 0.18 | 0.509 [0.457, 0.578]; +1.67; 0.17 |
| transpose | 0.507 [0.445, 0.576]; −0.16 | 0.545 [0.457, 0.614]; 0.527; +0.91; 0.18 | 0.533 [0.439, 0.604]; +0.48; 0.31 |
| symmetric mean | 0.538 [0.497, 0.591]; +1.51 | 0.538 [0.497, 0.591]; 0.505; +1.51; 0.18 | 0.543 [0.501, 0.591]; +1.39; 0.18 |
| symmetric max | 0.526 [0.487, 0.575]; +1.07 | 0.526 [0.487, 0.575]; 0.502; +1.07; 0.18 | 0.532 [0.494, 0.577]; +1.12; 0.21 |
| rank-conditioned | 0.483 [0.410, 0.586]; +1.41 | 0.481 [0.402, 0.588]; 0.454; +1.42; 0.18 | 0.499 [0.444, 0.580]; +1.52; 0.17 |
| order share F | 0.538 [0.434, 0.630]; +1.05 | (model (a) is used for F); q 0.18 | HGB on gene features: 0.525 [0.455, 0.590]; +0.93; 0.21 |
| \|Spearman\| co-expression (reference) | 0.505 [0.432, 0.637]; +1.91 | 0.505 [0.432, 0.637]; 0.457; +1.91 (p 0.030); not in the family | not run |

- No score passes the degree null after BH under either model (lowest q 0.17).
- Against 0.5, every interval includes 0.5 except the HGB symmetric mean, whose lower end is 0.501.
- F is antisymmetric (F[P, T] = 1 − F[T, P]), so adding it changes nothing for the symmetric edges.
- In RPE1 the transpose and symmetric residuals stayed above the null (section 01). That does not hold in K562.

**Relation to co-expression** (`coexpr/coexpr_comparison.json`; same null networks):
- Over all ordered gene pairs, attention and |Spearman| co-expression barely correlate: Spearman 0.043 (forward and transpose), 0.050 (symmetric max), 0.058 (symmetric mean).
- Symmetric mean after removing co-expression: 0.540 (linear fit, z +1.05) and 0.539 (50 bins, z +1.08). Forward: 0.522 (z +0.94) and 0.529 (z +1.27).
- Co-expression after removing symmetric mean: 0.584 (linear, z +1.41) and 0.584 (50 bins, z +1.48).
- Null test of the difference "attention − co-expression": symmetric mean z +0.00, forward −0.18, symmetric max −0.38, transpose −1.29.
- So in K562 attention and co-expression are separate signals of similar, weak size on TRRUST. Neither passes the degree null. On knockdown, co-expression is ahead of every attention score (table above).

**Other layers (not pre-specified)** (`layers/per_layer.csv`, `layers/layers_summary.json`; point estimates; TRRUST z against the same 1,000 null networks; per-layer z not adjusted):

| Layer | Forward: TRRUST AUROC; z; knockdown AUROC | Symmetric mean: TRRUST AUROC; z; knockdown AUROC |
|---|---|---|
| 0 | 0.708; +2.49; 0.578 | 0.738; +2.90; 0.577 |
| 1 | 0.712; +1.32; 0.645 | 0.674; +1.13; 0.632 |
| 2 | 0.729; +1.76; 0.635 | 0.687; +1.21; 0.620 |
| 3 | 0.600; +2.04; 0.549 | 0.583; +1.83; 0.535 |
| 4 | 0.614; +1.13; 0.572 | 0.581; +0.03; 0.574 |
| 5 | 0.590; +1.79; 0.508 | 0.581; +1.92; 0.508 |
| 6 | 0.588; +1.10; 0.534 | 0.608; +1.57; 0.524 |
| 7 | 0.574; +1.28; 0.508 | 0.585; +1.52; 0.506 |
| 8 | 0.548; +1.72; 0.505 | 0.571; +1.76; 0.502 |
| 9 | 0.538; +1.34; 0.470 | 0.537; +1.97; 0.462 |
| 10 | 0.618; +1.71; 0.536 | 0.641; +1.95; 0.531 |

- Layer-adjusted TRRUST test: forward largest z +2.49 at layer 0, p 0.041; symmetric mean largest z +2.90 at layer 0, p 0.019.
- Knockdown is highest at layers 1–2 (forward 0.645 and 0.635), still below target variance (0.690).
- Co-expression, gene-model and residual controls were not run at these layers. Treat the early-layer result as a lead, not a finding.

**Secondary analysis: the gene set of the deployed analysis (not pre-specified).** The deployed analysis chose its 1,500 genes by the same variance rule, but on log1p values that were logged a second time; 627 of those genes are in the gene set used above (Jaccard 0.26). The attention sums of step 1 were read on that gene set, with its TRRUST positives (8 TFs, 45 positive pairs, 11,947 negative pairs) and the 1,000 null networks stored for it. This analysis was not planned before the main results. Its p-values are not adjusted for multiple testing. Sources: `oldgenes/deployed_genes_comparison.json`, `oldgenes/deployed_genes_coexpr_single_log.json`, `trrust/leave_one_tf_out.json`.

| Score (TRRUST, 8 TFs) | Pooled AUROC [95% CI, bootstrap over TFs] | z | p (one-sided) |
|---|---|---|---|
| forward | 0.575 [0.485, 0.691] | +2.55 | 0.008 |
| transpose | 0.487 [0.384, 0.561] | +0.26 | 0.40 |
| symmetric mean | 0.591 [0.549, 0.638] | +2.38 | 0.010 |
| symmetric max | 0.609 [0.558, 0.662] | +2.07 | 0.017 |
| order share F | 0.535 [0.445, 0.658] | +1.77 | 0.047 |
| \|Spearman\| co-expression (one log) | 0.542 [0.453, 0.637] | −0.66 | 0.73 |

- Leave one TF out: symmetric mean z +2.04 to +2.50 (largest p 0.020); forward z +1.59 to +3.24.
- Knockdown on this gene set (labels by the rule of step 4; 182 perturbations, 10,410 positive pairs): forward 0.523 [0.509, 0.537], symmetric mean 0.516 [0.505, 0.529], F 0.534 [0.519, 0.549].
- So on this gene set attention beats the degree null on TRRUST and co-expression does not. On the pre-specified gene set (16 TFs) it does not pass after BH. With 8 to 16 TFs, the K562 TRRUST answer depends on which genes and TFs enter the test. Knockdown AUROCs of the attention scores and F lie between 0.498 and 0.534 on both gene sets.

### Verification

No separate verification by a second agent is recorded for this analysis. The report describes these second-way checks (script `v3_attention_k562_verify.py`, own seeds 777–779; outputs in `outputs/v3_attention_k562/verification/`). All pass.

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Attention edges | Hooked `self_attn` weights, summed over cells on MPS | 6 random gene pairs (24 cells) run again through a separate extractor that reads `output_attentions=True` from a CPU copy of the model | Rebuilt E equals saved E at all 11 layers (largest relative difference 2.1 × 10⁻⁷). Attention rows sum to 1 (largest deviation 4.8 × 10⁻⁷). One cell, CPU vs MPS: largest difference 1.8 × 10⁻⁶. The 24 cells pass `check_encoding`. |
| n_pair and C | Counted in the extraction loop | Rebuilt from the saved tokens with a position matrix, both gene sets | Equal |
| Knockdown labels | Main code | Own block reader, own Welch t, statsmodels BH, all 446 candidates | Same 194 perturbations kept, 0 label disagreements, 8,714 positives |
| AUROCs | Tie-safe rank AUROC | scikit-learn | Equal within 1.1 × 10⁻¹⁶ for every score, both endpoints |
| Bootstrap intervals | Main seeds | Own seeds; TRRUST by explicit copying of each drawn TF's pairs | Variance − forward (knockdown) [+0.162, +0.209] against [+0.162, +0.208]. TRRUST intervals within 0.007 of the main ones. |
| Degree null | Curveball, 1,000 networks × 5,000 trades | Checkerboard swaps (own code, 1,000 networks × 2,140 swaps) | z forward +1.67 (p 0.051), symmetric mean +1.76 (p 0.042), co-expression +1.60; every z within 0.06 of Curveball |
| Residuals | numpy least squares, KFold seed 42 | scikit-learn `LinearRegression`, different random folds; forward, transpose, both symmetric edges, F and co-expression (the rank-conditioned edge was not rechecked) | Residual AUROCs within 0.0003, z within 0.012 |
| 12-test BH q | Main code | statsmodels `multipletests(method="fdr_bh")` on the 12 stored p-values | Identical q values |

Under the checkerboard sampler the forward p is 0.051, so even the unadjusted forward result sits at the 0.05 line.

### Limits

- One model size (217M), one cell line here (K562), one pre-specified layer (8). Other layers were scored only for the forward and symmetric-mean edges, without co-expression, gene-model or residual controls.
- TRRUST is small: 16 TFs and 107 positive pairs. z values near 2 move with single TFs (dropping MYC moves the forward z from +1.72 to +0.12). The K562 TRRUST answer also depends on the gene set (secondary analysis).
- The TF bootstrap treats TFs as independent, although the same candidate genes recur across TFs. Intervals for the gene model and the residuals keep the fitted models fixed. Both make intervals somewhat too narrow. The gene model also depends on the TF-to-fold map (0.723 against 0.691).
- The two knockdown label rules give very different numbers of perturbations (194 and 67). The verdict is the same under both.
- The Wilcoxon q-values for knockdown are BH-adjusted over the six scores within K562 only. Endpoint A has no null and was not residualised.
- The data file keeps 6,546 genes. Counts, the gene set and the token order cover only these genes.
- Only TRRUST was used as a curated reference. Only |Spearman| co-expression from the same 2,000 control cells was used as a co-expression baseline.
- Attention was averaged over the 8 heads; per-head attention was not saved, so single heads and head ablation cannot be tested from these files.
- Not run on these inputs: value-weighted edges, head ablation, the CSSI analysis, and the added-value logistic regression of section 01 (step 6). No MaxToki-1B or Adamson attention analysis with correctly encoded inputs is reported.
- Another project's job shared the machine during extraction, so the timings below are not clean benchmarks.

### Files

- Scripts (`projects/maxtoki/runs/attention-grn-217M/scripts/`):
  - `v3_attention_k562.py`: `prep` (cells, counts, gene set, features, tokens, checks), `extract [max_minutes]` (resumable forward passes), `finalize` (edges, n_pair, C, checks).
  - `v3_attention_k562_eval.py`: `de`, `order`, `knockdown`, `trrust`, `gene3`, `resid ols|hgb|combine`, `coexpr`, `layers`, `oldgenes`, `oldgenes_coexpr`, `loo`, `table`.
  - `v3_attention_k562_verify.py`: `pairs`, `counts`, `geneset`, `labels`, `scores`, `resid`, `config`.
  - Shared code: `v2_common.py`, `v2_06_curveball.py` (same folder); `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/maxtoki_adapter.py`, `projects/maxtoki/setup/dataset_loader.py`, `projects/maxtoki/setup/hooks_v2.py`.
- Outputs (`projects/maxtoki/runs/attention-grn-217M/outputs/v3_attention_k562/`):
  - `prep/`: `cells_tokens.npz` (tokens of the main encoding and of the deployed order), `counts_int32.npy`, `control_cells.csv`, `count_row_checks.csv`, `gene_features.csv`, `gene_sets.npz`, `hvg_gene_table.csv`, `union_gene_table.csv`, `spearman_edges.npy`, `order_agreement_per_cell.csv`, `prep_check.json`.
  - `extract/`: `state.npz` (attention sums over the 2,373-gene union, all layers), `state_meta.json` (per-cell sequence length, next-gene loss, encoding checks, chunk times).
  - `edges/`: `attention_edges_layer_mean.npy` (11 × 1,500 × 1,500), `attention_pair_counts.npy`, `order_counts.npy`, the same three with `_deployed_genes`, `union_*_layer8.npy`, `finalize_check.json`.
  - `de_labels/`: `v3_primary.npz` (main labels), `v3_storedX.npz` (sensitivity labels), `deployed_genes_single_log.npz` (secondary analysis), `de_meta.json`.
  - `order/order_check.json`; `knockdown/` (`knockdown_summary.json`, `per_perturbation_v3_primary.csv`, `per_perturbation_v3_storedX.csv`); `trrust/` (`trrust_summary.json` with both BH families, `gene_model_gaps.json`, `per_tf.csv`, `leave_one_tf_out.json`, `null_positions.npy`); `resid/` (`resid_ols.json`, `resid_hgb_*.json`, `resid_summary.json`); `coexpr/coexpr_comparison.json`; `layers/` (`per_layer.csv`, `layers_summary.json`); `oldgenes/` (secondary analysis); `verification/`.
  - `table_attention_variants_v3_k562.csv`, `fig_attention_variants_v3_k562.csv` (figure source data), `comparison_v2b_k562_v3_k562_v2b_rpe1.csv`.
  - `run_config.json`: device, package versions, seeds, the 2,000 h5ad rows, the 1,500 Ensembl ids, the input-encoding record and check result, sha256 of counts and tokens, wall time per chunk, sha256 of 20 inputs, 9 code files and 58 outputs.
- Inputs from other runs: `projects/maxtoki/runs/attention-grn-217M/outputs/phase0/` (`control_cells.csv`, `hvg_gene_table.csv`, `gene_features.csv` of the deployed analysis); `.../outputs/v2b_attention/trrust/null_positions_217M_K562.npy` (null networks of the deployed gene set); `.../outputs/v2b_attention/trrust/trrust_variants_217M_RPE1.json` (RPE1 p-values for the 12-test BH). Encoding-test cells: `projects/maxtoki/checks/v3_inputs/model/K562_ctrl_replogle_concat.json`.
- Order to run: `v3_attention_k562.py prep`; `extract 7.3` (repeat until done); `finalize`; the eval commands in the order listed; the verify commands, with `config` last.
- Run time: extraction 948 s in three chunks (105 s, 426 s, 417 s) on MPS; peak memory 5.4 GB.
