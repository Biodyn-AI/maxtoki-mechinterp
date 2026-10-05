# Contents of this appendix {-}

This appendix describes each corrected analysis of MaxToki-217M reported in the paper: the data, the method in enough detail to repeat it, the checks that were run, the full results, how the key numbers were derived a second time, and what was not tested. Section 00 describes the input encoding and the intervention code shared by several analyses. Paths are relative to the root of the code and data release.


# Shared code: input encoding and intervention hooks

Sections 02 to 05 use two shared modules. `projects/maxtoki/setup/inputs_v3.py` builds the model input from counts and checks it. `projects/maxtoki/setup/hooks_v2.py` edits the residual stream with sparse autoencoder (SAE) features. Section 01 (attention, RPE1 cells) has its own extraction path; its input check is described there.

## Input encoding

MaxToki uses the rank-value encoding of Geneformer. For one cell with counts c_g:

1. value(g) = c_g / (sum of counts in the cell) × 10,000 / median(g). Here median(g) is the Geneformer gc104M gene median that the MaxToki tokenizer uses.
2. Keep the genes in the MaxToki vocabulary with value > 0. Sort them from high to low value.
3. Keep the first max_len − 2 genes. Add `<bos>` (token 2) at the start and `<eos>` (token 3) at the end.

The per-cell factor 10,000 / (sum of counts) does not change the order. So any per-cell multiple of the counts gives the same tokens.

Where the counts come from:

| Data file | Stored `X` | Counts used |
|---|---|---|
| K562 (`replogle_concat.h5ad`), Adamson | log1p(CP10k) | round(expm1(X) / u), with u = the smallest non-zero expm1(X) in the cell |
| RPE1 (`ReplogleWeissman2022_rpe1.h5ad`) | integer counts | `X` |
| Tabula Sapiens, Krasnow lung | log1p values | `raw/X` (integer counts) |

For K562 and Adamson the loader first checks that every non-zero expm1(X) value is a whole-number multiple of u. The tolerance is max(10⁻³, 5 × 10⁻⁶ × multiple), and a multiple above 10⁵ fails. Dividing by u and rounding gives integer counts up to one factor per cell. Rounding also removes the float32 noise in `X` (relative error about 3 × 10⁻⁷). In 2 of 9,200 K562 cells, expm1(X) is already a whole number with u = 1. The function `counts_accept_unit_one` accepts these two cells and returns the same result as the main reader for every other cell.

## Per-cell encoding check

`check_encoding(counts, tokens, ...)` compares a token sequence with the counts of the same cell. It passes only if all of these hold:

1. `<bos>` is first and `<eos>` is last. There is no other special token and no repeated gene. Every token is a vocabulary gene with count > 0.
2. counts / median does not increase along the sequence. Values within 10⁻⁶ of each other (relative) count as tied and may come in either order.
3. The length is min(number of expressed vocabulary genes, max_len − 2), and no left-out gene has a larger value than a kept gene.
4. The counts look like counts: whole numbers, or whole-number multiples of the smallest value. Log values fail this test.
5. The sequence agrees with an independent stable sort, except at positions that hold tied values.

`assert_encoding_batch` runs this check on every cell of a batch and stops the run on any failure. Every analysis in sections 02 to 05 calls it before each forward pass.

Tests of the encoding (`projects/maxtoki/setup/test_inputs_v3.py`; outputs in `projects/maxtoki/checks/v3_inputs/`):

| Test | Cells | Result |
|---|---|---|
| Count test on the count source | 530 cells from 10 dataset × context-length groups | 530 / 530 pass |
| Tokens pass `check_encoding` | same 530 | 530 / 530 pass |
| Same tokens as the tokenizer run on integer counts | same 530 | 528 / 530 identical; the other 2 (RPE1) differ only inside exact ties |
| Negative control | 475 non-RPE1 cells | The stored log1p rows fail the count test in 475 / 475, and the tokens ranked from them (the input order of the deployed analysis) fail `check_encoding` in 475 / 475 |
| Model smoke test (MPS, float32) | 50 cells, context 1,024 to 4,096 | 50 / 50 forward passes give finite logits |

## Intervention hooks

**Where the edit goes.** MaxToki-217M has 11 decoder blocks. `output_hidden_states` returns 12 tensors. `hidden_states[0]` is the token embedding, which is the input of block 0. For l = 1 to 10, `hidden_states[l]` is the output of block l − 1, which is the input of block l. `hidden_states[11]` is the final RMSNorm of the output of block 10, which is the input of `lm_head`. The SAE of layer l is trained on `hidden_states[l]`. So the edit for layer l is a forward pre-hook on block l (l = 0 to 10) or on `lm_head` (l = 11). The edit changes exactly the tensor the SAE reads. At layer 11 the edit comes after the final norm, so the norm is applied once.

**What the edit is.** The SAE code z is computed inside the forward pass from the live input x at the hook. A delta is added to x:

- Ablate(l, F): delta = − Σ over f in F of z_f(x) · W_dec[:, f]. The SAE reconstruction error stays in place.
- Steer(l, f, α), scale mode: delta = (α − 1) · z_f(x) · W_dec[:, f]. α = 1 is an exact no-op.
- ZeroDelta(l): delta = 0. AddVector(l, v): delta = v. Both are for tests.

If several edits target one layer, z is computed once and their deltas are summed. Because z is computed from the live tensor, an edit at a later layer sees the effect of the edits at lower layers. So edits at several layers combine. By default every position is edited, including `<bos>` and `<eos>`. Read-outs use captures of the live input of each layer (before and after the edit at that layer), not `output_hidden_states`.

The analyses load the SAEs of section 02 by passing `sae_dir = projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae`. They load cells with `inputs_v3.load_k562_control_cells`, which picks the same K562 rows as `hooks_v2.load_k562_control_cells` but encodes them from counts.

**Hook unit tests** (`projects/maxtoki/setup/test_hooks_v2.py`; 3 K562 non-targeting control cells, rows 289300, 292833 and 293501; MPS, float32; results in `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/summary.json`). Every "difference" or "change" below is the largest absolute difference over all positions and dimensions.

| Test | What is done | Pass rule | Result (worst of 3 cells) |
|---|---|---|---|
| Layer convention | Compare the pre-hook input of each block with `hidden_states[l]`; compare `hidden_states[11]` with the final norm of the block-10 output | Difference ≤ 10⁻⁶ (≤ 10⁻⁵ for the norm) | 0.0 and 0.0 |
| 1. A zero edit changes nothing | ZeroDelta at each of the 12 sites; Steer with α = 1 at 5 layers | Logit change < 10⁻⁴ | 0.0 |
| 2. An edit changes only later layers | Ablate the most active feature at layer 0, 3, 5, 9, 10 or 11 | Inputs of lower layers and the pre-edit input of the same layer change ≤ 10⁻⁶; every later layer and the logits change > 10⁻⁴; the observed delta equals −z_f · W_dec[:, f] within 10⁻⁴ | Lower layers: 0.0. Smallest later-layer change: 0.27 |
| 3. Combined edits differ from single edits | Ablate one feature at L0 (A), L5 (B) and L9 (C); compare AB with B, ABC with C, BC with C, ABC with BC | Each difference > 10⁻⁴ at the input of `lm_head` and in the logits | Smallest AB vs B: 0.55. Smallest ABC vs C: 2.55 |
| 4. A silent feature changes nothing | Ablate a feature that is inactive at every position, at L0, L5 and L11 | Delta exactly 0; logit change < 10⁻⁵ | 0.0 |
| Layer-11 path | Add a random vector at layer 11; ablate a feature at layer 11 | Logit change equals `lm_head`(delta) within 10⁻³ | 5.6 × 10⁻⁵ |

These unit tests used the module's default SAE files (the SAEs of the deployed atlas) and cells from `hooks_v2.load_k562_control_cells`, which ranks the stored log1p values. They test the hook mechanics, which do not depend on the SAE or on the input order. The same four properties were checked again inside the analyses, with the SAEs of section 02 and correctly encoded inputs:

| Property | Analysis | Result |
|---|---|---|
| 1. A zero edit changes nothing | Encoding tests: ZeroDelta at all 12 sites, one correctly encoded K562 cell | Logit change 0.0 |
| | Section 04: ZeroDelta at 4 source layers, 200 cells | Largest SAE-code change 0.0; 0 edges |
| | Section 05: ZeroDelta at L0, L5, L9 and all three together, 2 cells | Logit change 0.0; SAE-code change 0.0 |
| 2. An edit changes only later layers | Section 04: a forward pass that starts at block s from the stored clean input, against a full forward pass (2 cells × 8 features) | Difference 0.0; edges appear at the very next layer from all 4 source layers |
| | Section 05: same comparison for edits at L5 and L9 | Difference 0.0 |
| The edit equals the SAE term | Section 02: ablate all features at layer 5 | The input changes by exactly the SAE reconstruction (largest error 9.5 × 10⁻⁷ on values up to 352) |
| 3. Combined edits differ from single edits | Section 05: AB vs B for all 64 (A, B) pairs; ABC vs C for all 512 triplets | 0 of 64 and 0 of 512 equal; smallest largest-differences 0.020 and 0.019 |
| 4. A silent feature changes nothing | Section 05: 24 triplets that contain an exactly silent L0 or L9 feature | Three-way interaction term 0.0 exactly in every cell and target; 2,144 of 2,144 paired conditions with and without the silent feature are bit-identical |

## Files

- Code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py` (SAE class), `projects/maxtoki/setup/maxtoki_adapter.py` (tokenizer and model loader).
- Tests: `projects/maxtoki/setup/test_inputs_v3.py`, `projects/maxtoki/setup/test_hooks_v2.py`.
- Test outputs: `projects/maxtoki/checks/v3_inputs/` (`summary.json`, `cpu_results.json`, `cells_cpu.csv`, `model/`), `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/` (`summary.json`, one JSON per cell).


# Attention edges as gene-regulation scores (MaxToki-217M, RPE1 cells)

This analysis asks whether MaxToki-217M attention between two genes ranks gene pairs by regulation. It uses two endpoints. **Endpoint A (knockdown response):** for one knocked-down gene, which other genes change? **Endpoint B (TRRUST):** for one transcription factor (TF), which genes does the curated TRRUST database list as its targets? This section reports the RPE1 run: its stored data are raw counts, so the deployed code encoded them correctly. The K562 run with correct encoding is described in section 08. The deployed K562, Adamson and MaxToki-1B attention runs used wrongly encoded inputs and are not reported.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors (11 decoder blocks × 8 heads, hidden size 1,232, 216,946,576 parameters). float32, Apple MPS, eager attention so that attention weights are returned.
- **Cells.** Replogle RPE1 Perturb-seq data (`ReplogleWeissman2022_rpe1.h5ad`, 8,749 genes). `X` holds integer counts.
- **Cells for attention and gene statistics.** 2,000 non-targeting control cells, drawn with numpy `default_rng(42)` from 11,485 and sorted.
- **Gene set.** The 1,500 genes with the highest variance of log1p(CP10k) over the 2,000 control cells, among genes in the MaxToki vocabulary.
- **Model input.** Each cell's integer counts, encoded as in section 00: counts / gene median, sorted high to low, first 2,046 genes, `<bos>` and `<eos>` (max_len 2,048). Mean 2,039 tokens per cell.
- **Knockdown labels.** Perturbed RPE1 cells from the same file, against 5,000 control cells drawn with `default_rng(42)` from the 11,485.
- **TRRUST.** TRRUST human TF–target table (version 2). 16 TFs have at least 3 targets in the gene set: JUN, PARP1, PTTG1, HDAC2, HIPK2, KLF6, FOXM1, TFDP1, HIF1A, YY1, BRCA1, NFIC, DNMT1, JUND, CEBPB, ATF4. This gives 138 positive and 23,846 negative (TF, gene) pairs.

## Method

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

## Checks

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

## Results

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

## Verification

Three other agents re-checked this analysis with their own scripts. None of the scripts imports the main analysis code.

- **Base evaluation** (`v2verify_01_endpointA.py` to `v2verify_10_run_config.py`; outputs in `outputs/v2_eval/verification/`). Per-perturbation AUROCs equal the main files within 2.2 × 10⁻¹⁶; perturbation and positive counts and the 314 / 325 count match. Own perturbation bootstrap (2,000 resamples, other seed): variance − forward +0.163 [+0.153, +0.173]. TRRUST counts and AUROCs match; own TF bootstrap CIs within 0.01. A leave-one-TF-out gene model gives a gap of +0.145 to the forward edge, with the interval above 0. Own boolean-matrix Curveball sampler (1,000 networks × 1,000 trades, seed 23): z +1.73, p 0.047. Layer-adjusted p 0.005 (main: 0.003). It confirmed that the RPE1 file holds integer counts and that float32 and float64 OLS agree for this run. It added the co-expression null (z +3.09) and the two-gene-means score (z +1.81).
- **Score variants, first check** (`v2b_attention_verify.py`, seed 777001; outputs in `outputs/v2b_attention/verification/`). Own rank code rebuilt C from the h5ad file with 0 mismatches; R² 0.381 and within-row Spearman 0.77 and 0.86 reproduce. Knockdown AUROCs equal within 2.2 × 10⁻¹⁶; F − forward +0.045 [+0.037, +0.052]. TRRUST AUROCs equal. Own Curveball (1,000 networks × 3,000 trades): z forward +1.82, transpose +2.27, symmetric mean +3.17, symmetric max +2.97, rank-conditioned +2.26, F +0.04, co-expression +2.96. Own OLS residuals match to 3 decimals (transpose 0.552, z +3.02; symmetric mean 0.538, z +2.80; symmetric max 0.536, z +3.06). HGB spot checks: symmetric mean 0.601 [0.542, 0.661], z +2.86; transpose 0.571, z +2.19. With its own random TF-to-fold map the gene model is 0.753 and its gap to symmetric max is +0.106 [+0.008, +0.207] instead of +0.102 [−0.009, +0.204]. So "ties" against "just beats" depends on the fold map. All 205 rows of `fig_attention_variants.csv` equal the JSON values.
- **Score variants, second check** (`v2b_attention_verify2.py`, seed 50505; outputs in `outputs/v2b_attention/verification2/`). It checked C against its invariants (0 ≤ C ≤ n_pair, C + Cᵀ = n_pair, edge = 0 exactly where C = 0). Knockdown AUROCs equal for all 9 scores; F − forward +0.045 [+0.037, +0.052]. TRRUST AUROCs equal exactly; TF bootstrap CIs within 0.007. A different null sampler (checkerboard swaps, 1,000 networks, 20 × number of edges accepted swaps): z forward +1.80, transpose +2.38, symmetric mean +3.26, symmetric max +3.08, rank-conditioned +2.30, F +0.05, co-expression +3.02. Residual AUROCs equal within 0.001 (residual z: transpose +3.11, symmetric mean +2.87, symmetric max +3.15). It also ran the co-expression and leave-one-TF-out analyses above, found that the rank-conditioned edge still follows gene order, and found that target mean expression is below F on knockdown (−0.066 [−0.077, −0.054]).

All three agreed with every number they recomputed. Their extra findings are reported in the Results above.

## Limits

- One cell line (RPE1), one model size (217M) and one pre-specified layer (8). Other layers were scored for the forward edge only. Residualisation was run at layer 8 only. The early-layer TRRUST result was not compared with co-expression.
- TRRUST is small: 16 TFs and 138 positive pairs. The TF bootstrap treats TFs as independent, although the same candidate genes recur across TFs, so intervals are somewhat too narrow. Intervals for the gene model and the residuals keep the fitted models as they are, which also makes them too narrow.
- Only TRRUST was used as a curated reference. DoRothEA, STRING and ChIP-seq references were not tested.
- Endpoint A was not residualised and has no null. The added-value test was run for the forward edge only, not with F as a covariate.
- Only |Spearman| co-expression from the same 2,000 control cells was used as a co-expression baseline.
- Attention was averaged over heads. Single heads were not tested.
- The rank-conditioned edge is set to 0 where T never comes before P (0.18% of pairs). No other fill rule was tried.

## Files

- Attention extraction: `projects/maxtoki/runs/attention-grn-217M/scripts/phase0_rpe1.py` → `projects/maxtoki/runs/attention-grn-217M/outputs/phase0_rpe1/` (`attention_edges_layer_mean.npy`, `attention_edges_per_head.npy`, `attention_pair_counts.npy`, `control_cells.csv`, `gene_features.csv`, `spearman_edges.npy`, `run_config.json`). Per-layer knockdown AUROCs: `.../outputs/phase1_rpe1/attention_per_layer_auroc.csv`.
- Evaluation of the forward edge: `.../scripts/v2_common.py` and `v2_01_facts.py` to `v2_11_run_config.py` → `.../outputs/v2_eval/` (`run_facts.json`, `knockdown/knockdown_summary.json`, `knockdown/per_perturbation_217M_RPE1.csv`, `knockdown/incremental_217M_RPE1.json`, `trrust/`, `curveball/curveball_summary.json`, `residualised/residualised_summary.json`, `layers/per_layer_trrust.csv`, `layers/layers_summary.json`, `fig_knockdown.csv`, `fig_trrust.csv`, `run_config.json`).
- Score variants: `.../scripts/v2b_attention.py` → `.../outputs/v2b_attention/` (`order/order_check_217M_RPE1.json`, `order/order_counts_217M_RPE1.npy`, `knockdown/knockdown_variants_summary.json`, `knockdown/per_perturbation_217M_RPE1.csv`, `trrust/trrust_variants_217M_RPE1.json`, `trrust/per_tf_217M_RPE1.csv`, `trrust/gene_model_gaps_217M_RPE1.json`, `trrust/null_positions_217M_RPE1.npy`, `resid/resid_ols_217M_RPE1.json`, `resid/resid_hgb_217M_RPE1_*.json`, `resid/resid_summary.json`, `table_attention_variants.csv`, `fig_attention_variants.csv`, `run_config.json` with sha256 of inputs, scripts and outputs).
- Verification: `.../scripts/v2verify_01_endpointA.py` to `v2verify_10_run_config.py`, `v2b_attention_verify.py`, `v2b_attention_verify2.py`; outputs in `.../outputs/v2_eval/verification/`, `.../outputs/v2b_attention/verification/`, `.../outputs/v2b_attention/verification2/`.
- The CSV and JSON summary files hold rows for four runs; the rows of this analysis have run = `217M_RPE1`.


# Sparse autoencoders on correctly encoded inputs (MaxToki-217M, 12 layers)

A sparse autoencoder (SAE) rewrites each hidden-state vector as a sum of a few learned directions ("features"). This analysis trains one TopK SAE at each of the 12 residual-stream sites of MaxToki-217M, on hidden states from correctly encoded K562 control cells. It reports how well each SAE reconstructs held-out hidden states. Sections 03, 04 and 05 use these SAEs.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4.
- **Cells.** 500 K562 non-targeting control cells from `replogle_concat.h5ad`, drawn with numpy `default_rng(42)` from a pool of 10,691 and sorted.
- **Input.** Counts rebuilt and encoded with `inputs_v3` (section 00): counts / gene median, high to low, `<bos>` + up to 2,046 genes + `<eos>` (at most 2,048 tokens). 489 of 500 cells are cut at 2,046 genes. In total 1,019,996 token positions. One cell per forward pass.
- **Sites.** `hidden_states[0]` to `hidden_states[11]` (section 00), at every position, including `<bos>` and `<eos>`. Hidden size 1,232.
- **Split.** The first 90% of positions train the SAE (917,996 positions). The last 102,000 positions are held out. These are cells 450–499; cell 450 is only partly held out (1,648 of its 2,048 positions).

## Method

Each SAE encodes a hidden-state vector into 4,928 non-negative feature values, of which at most 32 are non-zero, and decodes them back. It is trained to make the decoded vector close to the input.

1. **SAE** (`projects/maxtoki/setup/topk_sae.py`). z = TopK₃₂(W_enc (x − μ) + b_enc), with negative kept values set to 0. x̂ = W_dec z + μ. μ is the mean of the training rows (summed in float64, stored as float32). d_model = 1,232, d_sae = 4,928, k = 32. W_dec starts as the transpose of W_enc, and its columns are scaled to unit L2 norm after every step.
2. **Training.** Loss = mean squared error between x and x̂ (no L1 term). Adam, learning rate 3 × 10⁻⁴, batch 4,096, 4 epochs (900 steps), `torch.manual_seed(42)`, shuffled by a PyTorch DataLoader. Training saves a checkpoint each epoch so it can resume; layers 0, 3, 4, 5, 6, 8 and 11 were resumed at least once. Training rows are read from disk by index to keep memory under about 8 GB; the shuffle order still comes from the same torch random generator.
3. **Evaluation on the 102,000 held-out positions.**
   - Fraction of variance explained (FVE) = 1 − (sum of squared reconstruction errors) / (sum of squared distances to the held-out mean). Sums run over all 1,232 dimensions and all held-out positions.
   - Dead features = features that never fire on the held-out positions. Also counted on all 1,019,996 positions.
   - Mean L0 = average number of non-zero features per position.
   - Intervals: 95% percentile interval from a cluster bootstrap over the 50 held-out cells (2,000 resamples, seed 20261001).
4. **Loading in later analyses.** The SAEs are saved as `layer_XX/sae_final.pt` in the format that `hooks_v2.load_saes(..., sae_dir=...)` reads.

## Checks

| Check | Result |
|---|---|
| Encoding check before every forward pass | `inputs_v3.assert_encoding_batch` on every cell of all 20 batches of 25 cells: 500 / 500 pass, 0 order violations (68 positions are exact ties in another order, which the check allows). The count rows are whole-number multiples of their smallest value in all 500 cells. Tokens ranked from the stored log1p values fail the same check in 500 / 500 cells. |
| Same cell rows as the atlas design | Encoding these 500 rows from the stored log1p values rebuilds the stored gene list of the atlas (`outputs/phase0/layer_00/gene_names.json`) at all 1,019,996 positions. |
| Hooks see the training activations | `hooks_v2.load_saes(range(12), sae_dir=outputs/v3_sae)` loads all 12 SAEs. On 3 held-out cells (451, 475, 499), the live input captured at every hook site equals the stored training activation exactly (largest difference 0.0, all 12 sites, including the input of `lm_head`). Reconstruction error from the live codes matches the evaluation within 3.4 × 10⁻⁹ (relative). Ablating all features at layer 5 changes the input by exactly the SAE reconstruction (largest error 9.5 × 10⁻⁷ on values up to 352). |
| Training code | Given the layer-0 inputs of the deployed atlas (rebuilt from its tokens without a forward pass, because layer 0 is the embedding lookup), the same training code reproduced the deployed layer-0 SAE: held-out FVE within 0.0001, the same number of dead features, epoch losses within 0.011%, and decoder columns that moved during training matched at median \|cos\| 0.998 (min 0.970). Weights are not bit-identical (relative difference 3–5%), most likely from float rounding that grows over 900 steps. |
| Second computation | Separate code (CPU, float64 sums, two-pass variance) reproduces every FVE within 1.5 × 10⁻⁸ and every L0 and dead count exactly. The FVE stored at the end of training equals the summary value (difference < 10⁻¹⁵). |
| Overfitting | FVE on the training positions is 0.701–0.929, which is 0.001–0.010 above the held-out FVE at each layer. |

## Results

Held-out reconstruction (n = 102,000 positions from 50 cells; 95% percentile interval, cluster bootstrap over the 50 held-out cells, 2,000 resamples):

| Layer | FVE [95% CI] | Dead features, held-out / all positions (of 4,928) | Mean L0 |
|---|---|---|---|
| 0 | 0.700 [0.696, 0.704] | 2,431 / 2,431 | 32.00 |
| 1 | 0.910 [0.908, 0.913] | 0 / 0 | 32.00 |
| 2 | 0.928 [0.926, 0.930] | 0 / 0 | 32.00 |
| 3 | 0.912 [0.909, 0.915] | 0 / 0 | 32.00 |
| 4 | 0.900 [0.897, 0.903] | 0 / 0 | 32.00 |
| 5 | 0.862 [0.858, 0.866] | 0 / 0 | 32.00 |
| 6 | 0.847 [0.842, 0.851] | 0 / 0 | 32.00 |
| 7 | 0.847 [0.842, 0.851] | 5 / 0 | 32.00 |
| 8 | 0.842 [0.836, 0.847] | 18 / 0 | 32.00 |
| 9 | 0.859 [0.853, 0.863] | 12 / 0 | 32.00 |
| 10 | 0.869 [0.864, 0.874] | 4 / 0 | 32.00 |
| 11 | 0.871 [0.865, 0.876] | 4 / 0 | 32.00 |

- The SAEs explain 70% of the held-out variance at layer 0 and 84–93% at layers 1–11.
- No feature is dead on all positions at layers 1–11. On the held-out positions alone, 4–18 features are dead at layers 7–11.
- Layers 6 and 7 have the same FVE by chance (0.84715 and 0.84717). Their data and SAEs differ (hashes in `eval/meta.json`).
- Layer 0 is special. `hidden_states[0]` is the embedding row of each gene token, and the 500 cells contain 6,330 distinct tokens. So layer 0 sees only 6,330 distinct vectors, and 2,431 features never fire.
- Mean L0 is exactly 32.00 at every layer and in every bootstrap resample. A TopK SAE keeps 32 features per position unless some of the top 32 are ≤ 0, and that never happened. So L0 carries no information here.
- For context, the SAEs of the deployed atlas, run on the same held-out tokens, explain 0.667 at layer 0, 0.848 at layer 1, and less than zero at layers 8–10 (−0.041, −0.131, −0.012).

## Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these cross-checks (all in the Checks table): an independent recomputation of every FVE, L0 and dead count with separate CPU code (`scripts/v3_sae_verify.py`, `verify/recompute.json`); the hook check on 3 held-out cells (`hooks_check.json`); and the layer-0 reproduction of the training protocol (`verify/l0_protocol_compare.json`). All agreed.

## Limits

- One training seed. There is no seed-to-seed spread for FVE, dead counts or which features exist.
- One cell type. Training and held-out positions come from the same 500 K562 control cells. The intervals cover cell-to-cell variation among the 50 held-out cells only. They do not cover other cell types or perturbed cells.
- One SAE size (4,928 features, k = 32). Other sizes and sparsity levels were not tried.
- Only reconstruction was measured. Whether features are interpretable was not tested here.
- The training-code check covers layer 0 only. Other layers would need forward passes on the deployed inputs, which were not run.

## Files

- Code: `projects/maxtoki/runs/sae-atlas-217M/scripts/v3_sae.py` (sub-commands: prepare, extract, train, evaluate, hooks check, summary), `projects/maxtoki/runs/sae-atlas-217M/scripts/v3_sae_verify.py`; shared code `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py`.
- SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_XX/sae_final.pt` (with `results.json` and `training_log.json`), XX = 00 to 11.
- Evaluation: `.../outputs/v3_sae/eval/summary.json`, `eval/table.csv`, `eval/layer_XX.npz` (per-cell sums), `eval/meta.json` (data and SAE hashes).
- Checks: `.../outputs/v3_sae/hooks_check.json`, `verify/recompute.json`, `verify/l0_protocol_compare.json`, `verify/l0_protocol/layer_00/`.
- Inputs and activations: `.../outputs/v3_sae/cells.npz` (rows, offsets, tokens), `gene_names_v3.json`, `activations/layer_XX.npy` (training activations, 60.3 GB) and `activations_eval/layer_XX.npy` (held-out rows, 6.0 GB), which are not deposited because of their size (ZENODO_MANIFEST.csv says how to rebuild them), and `codes/layer_XX_topk.npz` (top-32 codes at all 1,019,996 positions, 2.35 GB).
- Provenance: `.../outputs/v3_sae/run_config.json` (device, versions, seeds, cell rows, encoding-check result, input and code sha256, wall time per chunk), `prepare.json`, `chunks.json`, `verify/run_config.json`, `logs/`.
- Run time (MPS): extraction 17.5 min, training about 65 min for 12 layers, evaluation 13 min, checks about 6 min.


# TF specificity of layer-5 SAE features (MaxToki-217M, K562 knockdowns)

This analysis asks whether knocking down a transcription factor (TF) changes SAE features whose top genes are that TF's known targets. For each TF it selects the layer-5 SAE features that respond to the knockdown. It then tests whether the top-20 genes of those features overlap the TF's target set more than matched chance.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS, eager attention, batch size 1, max_len 2,048. torch 2.11.0, transformers 5.5.4. The forward pass stops at the input of block 5 (`hidden_states[5]`).
- **SAE.** The layer-5 SAE of section 02 (`projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_05/sae_final.pt`).
- **Cells** (Replogle K562 Perturb-seq, `replogle_concat.h5ad`), 9,200 in total:
  - 400 reference controls and 800 pool controls (for fake TFs), drawn from the 10,691 K562 non-targeting cells with seed 20261001;
  - 7,500 knockdown cells for 87 TFs, at most 100 per TF (seed 20261002); 34 to 100 cells per TF; GATA1 has 95;
  - 500 catalog control cells (seed 42), the SAE training cells of section 02.
- **Input.** Counts rebuilt and encoded with `inputs_v3` (section 00). Two cells (rows 282850 and 401789) have expm1(X) values that are already whole numbers with smallest value 1; `inputs_v3.counts_accept_unit_one` reads them as counts.
- **TFs.** 87 knocked-down TFs. 20 primary TFs have DoRothEA ChIP-seq target sets: ATF4, BRF2, CEBPZ, E2F6, GATA1, GTF2B, HINFP, MAX, MYBL2, SETDB1, SP2, SRF, STAT5A, TBP, TERF1, TERF2, TFDP1, THAP1, THAP11, ZNF24. TRRUST targets are used as a second database.
- **Gene universe.** 6,328 genes seen in the 500 catalog cells. GATA1 targets in the universe: 221 ChIP-seq, 11 TRRUST.
- **Knockdown strength** (expm1(X) values as a label, not a model input). For the 80 TFs whose gene is in the 6,546-gene panel, the TF's mean expression in knockdown cells is at most 13% of the reference level (median 0%). GATA1 and 6 other TF genes are not in the panel.

## Method

1. **Per-cell feature values.** For each cell, the layer-5 SAE code is computed at every position. The unit of analysis is the mean of each feature over the gene positions of the cell (`<bos>` and `<eos>` excluded).
2. **Top-20 catalog.** Encode every position of the 500 catalog cells, including `<bos>`/`<eos>` (named `<SPECIAL>`). For each feature and gene, take the mean activation over positions where the feature is active. The top-20 genes are the 20 genes with the highest mean (no minimum count). All 4,928 features have 20 genes; 47 lists contain `<SPECIAL>`. 1.1% of top-20 slots hold a gene seen in 5 or fewer cells.
3. **Selection of responding features.** For each TF: two-sided Mann–Whitney U test of the knockdown cells against the 400 reference cells, per feature, with the full tie correction. BH over the 4,928 features. A feature responds if q < 0.05 and |mean difference| > cut-off. Cut-offs: 0.5 (the value of the deployed analysis), 0.25, 0.1, 0.05, 0.02, 0.01 and 0. K = number of responding features. The full tie correction matters here: 78% of the 9,200 cells start with the same gene (RANBP1), so a feature that fires only near the start of the sequence has the same per-cell mean in many cells.
4. **Statistic.** M = the best overlap between the top-20 of any responding feature and the TF's target set. T = min(M, 5) if M ≥ 2, else 0. T is the statistic of the deployed analysis (thresholds 2 to 5). p = P_null(T ≥ T_obs). The uncapped M is also tested.
5. **Nulls.**
   - rf: K random catalog features (exact). This is the null of the deployed analysis.
   - cm (count-matched): each top-20 gene swapped for a random gene from the same detection-count bin (exact, by hypergeometric convolution).
   - cml: bins of detection count × gene-length tertile (cuts at 21,392 and 56,262 bp).
   - fake: random subsets of the 800 pool cells of the knockdown size, with the whole selection run again (2,000 per size).
   - fakeK: as fake, but the fake TF keeps its K most-changed features.
   - okd: the responding features of every other knockdown with K ≥ 1.
6. **Rank.** The cm p-value of the same features against the target set of each of 291 ChIP-seq TFs (≥ 20 targets) or 109 TRRUST TFs (≥ 5 targets). The rank of the true TF is reported.
7. **Multiple testing.** BH across TFs, separately per database, cut-off and null (over the 20 primary TFs and over all 87). No correction across cut-offs or nulls.
8. **Power.** Plant k = 1 to 8 true targets into the top-20 of one random responding feature. 200 repeats per (TF, cut-off, database, k), for the 20 primary TFs.
9. **GATA1 bootstrap.** Knockdown and reference cells are resampled separately with replacement (1,000 resamples, seed 20263027). The selection is run again in each resample with scipy's Mann–Whitney test.

## Checks

| Check | Result | Pass |
|---|---|---|
| Encoding check before every forward pass | `inputs_v3.assert_encoding_batch` on every cell of all 368 batches of 25: 9,200 / 9,200 pass, 0 order violations (1,212 positions are exact ties in another order, which is allowed) | yes |
| Negative control for the check | Tokens ranked from the stored log1p values fail the check in 100 / 100 sample cells | yes |
| Catalog cells = SAE training cells | 1,019,996 / 1,019,996 positions identical | yes |
| Live codes vs stored SAE codes (catalog cells) | identical at all 1,019,996 positions (same 32 features, value difference 0.0); the catalogs built from both are identical for all 4,928 features | yes |
| `<bos>` code | the same in every cell (largest difference 0.0 over 9,200 cells) | yes |
| Second model path (12 cells: 6 GATA1, 6 reference) | `LlamaForCausalLM(output_hidden_states=True).hidden_states[5]` + `TopKSAE.encode`, no hooks: per-cell means equal the stored ones within 2.4 × 10⁻⁷ (values up to 2.6); tokens identical | yes |
| Mann–Whitney code vs scipy (float64) | largest difference in p 0.0 (GATA1 and a 95-cell fake group); same BH sets | yes |
| Exact swap null vs Monte Carlo (GATA1's 7 features, 10,000 draws) | P(best ≥ t), t = 1 to 8: largest gap 0.005 (cm) and 0.007 (cml), at most 1.8 Monte Carlo SE; expected overlaps within 0.010 | yes |
| GATA1 K and best overlap from the written tables (by gene name) | all 14 (cut-off × database) rows equal `tf_results.csv` | yes |
| Selection false-positive rate (1,000 fresh random splits of the 1,200 controls, 400 vs 95) | P(≥ 1 responding feature): 0.032 at cut-off 0 (95% Clopper–Pearson 0.022 to 0.045); 0.002 at 0.01; 0.001 at 0.02 and 0.05; 0 at 0.1 and above. Single-feature p-values uniform (KS p 0.03 to 0.45 for 5 pre-chosen features; 0.36 for a random feature per split; share p < 0.05: 0.031 to 0.062) | yes |
| Fake-TF p-values uniform | 1,000 new fake TFs per primary TF (20,000), scored against the stored nulls; 28 settings: KS p median 0.56, min 0.040 (2 of 28 below 0.05; 1.4 expected); share of p ≤ 0.05 at most 1.0%; two-sample chi-square p < 0.05 in 5 of 132 per-TF tests | yes |
| fakeK null with a new seed (7 TFs with K > 0 at cut-off 0.25) | chi-square p 0.12 to 0.98; overlaps counted by a second implementation (by gene name) | yes |
| Planted signal (k = 8 ChIP targets, GATA1, cut-off 0.5) | detected with probability 1.00 [0.98, 1.00] by the uncapped count-matched test | yes |

Shared-pool effect. Fake TFs all share the same 400 reference cells and 800 pool cells. So single-feature fake-TF p-values are not uniform across fake TFs (KS p down to 5 × 10⁻²³ for one feature; 5.0% of fake TFs have ≥ 1 BH-significant feature at cut-off 0). The pool itself does not differ from the reference (0 BH-significant features; 7 features with p < 0.001 against 4.9 expected; 331 with p < 0.05 against 246). With fresh random splits the test is calibrated (row above).

## Results

**Main findings.**
- No TF has SAE features that are specific to its targets. Over 87 TFs, 7 cut-offs and both databases, no TF passes BH (q < 0.05 across TFs) under the cm, cml, fakeK or okd null.
- GATA1 changes 7 features at cut-off 0.5. The best one (feature 3843) has 4 of its top-20 genes among the 221 GATA1 ChIP-seq targets. This is close to chance for these genes (cm p = 0.32; expected best overlap 3.1). GATA1 ranks 82nd of 291 TFs for these features.
- The deployed analysis reported one GATA1 feature with 8 of its top-20 genes among GATA1 ChIP-seq targets (p = 0.030 against random features). In this SAE, no feature of the 4,928 has more than 5 GATA1 ChIP-seq targets in its top 20 (10 features have 5), and none of those 10 responds to GATA1 at cut-off 0.5.

**Responding features per primary TF** (`task_data/tf_summary.tsv`):

| TF | Cells | BH q < 0.05 (no cut-off) | K at 0.5 | 0.25 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|
| GATA1 | 95 | 4,097 | **7** | 13 | 21 | 46 | 249 | 543 |
| MAX | 100 | 2,888 | 1 | 3 | 10 | 20 | 95 | 304 |
| THAP1 | 100 | 1,492 | 0 | 1 | 3 | 6 | 18 | 93 |
| HINFP | 100 | 1,435 | 0 | 1 | 3 | 6 | 21 | 87 |
| TFDP1 | 100 | 725 | 0 | 0 | 0 | 4 | 4 | 36 |
| CEBPZ | 100 | 680 | 0 | 0 | 2 | 2 | 7 | 24 |
| GTF2B | 34 | 510 | 0 | 1 | 2 | 7 | 24 | 90 |
| MYBL2 | 81 | 266 | 0 | 0 | 4 | 5 | 7 | 17 |
| TERF2 | 47 | 250 | 1 | 1 | 1 | 3 | 7 | 21 |
| TBP | 100 | 186 | 0 | 1 | 1 | 4 | 8 | 20 |
| 10 others | 49–100 | 0–22 | 0 | 0 | 0 | 0–1 | 0–2 | 0–5 |

- Per-cell feature means are small (median over features of the reference mean 0.012).
- At cut-off 0.5, 17 of 87 TFs have at least one responding feature: AATF, ATF5, BDP1, CDC5L, ECD, GATA1 (7), GTF2A1, MAX, MED1 (2), PDCD11 (2), PHB2 (2), RUVBL1, SNIP1, TAF1, TAF5, TERF2, TSG101 (2). With no effect cut-off, 74 of 87 do.

**GATA1 responding features at cut-off 0.5** (n = 95 knockdown and 400 reference cells; effect = mean difference of per-cell feature means, knockdown minus reference; interval: percentile cell bootstrap, knockdown and reference resampled separately, 1,000 resamples):

| Feature | Effect [95% CI] | Cohen's d | Selected in bootstraps | GATA1 ChIP targets in top-20 |
|---|---|---|---|---|
| 4905 | 1.695 [1.568, 1.804] | 3.50 | 100% | 1 (GNA12) |
| 2665 | 0.756 [0.588, 0.917] | 1.19 | 100% | 1 (SUCO) |
| 1725 | 0.738 [0.687, 0.781] | 3.43 | 100% | 2 (FBXW11, ACER3) |
| 4264 | 0.702 [0.634, 0.767] | 2.71 | 100% | 1 (RNF220) |
| 3763 | 0.597 [0.460, 0.741] | 1.13 | 92% | 1 (DIS3L2) |
| **3843** | 0.561 [0.496, 0.626] | 2.37 | 97% | **4** (MBD5, FRYL, CDKAL1, TBL1XR1) |
| 892 | 0.506 [0.434, 0.579] | 1.99 | 56% | 0 |

- Across bootstraps, K at 0.5 has median 7 (95% range 5 to 7). Only one other feature is ever selected. The best ChIP overlap is 4 in 96.5% of resamples and 2 in 3.5%. It never reaches 5.
- The four ChIP targets of feature 3843 are seen in 2, 16, 23 and 100 of the 500 catalog cells. Two are rare genes. This is why the count-matched null expects a best overlap of 3.1.

**GATA1 tests at every cut-off, ChIP-seq targets** (`summary_tables.md`, Table B):

| Cut-off | K | M | T | p rf | p cm | p cml | E[M] under cm | p fake | p fakeK | p okd (other knockdowns) | Rank / 291 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 7 | 4 | 4 | 0.055 | **0.318** | 0.431 | 3.11 | 0.0005 | 0.189 | 0.059 (16) | 82 |
| 0.25 | 13 | 4 | 4 | 0.101 | 0.471 | 0.546 | 3.54 | 0.0005 | 0.554 | 0.088 (33) | 176 |
| 0.1 | 21 | 4 | 4 | 0.158 | 0.554 | 0.628 | 3.71 | 0.0005 | 0.696 | 0.128 (46) | 178 |
| 0.05 | 46 | 4 | 4 | 0.314 | 0.656 | 0.720 | 3.89 | 0.0005 | 0.865 | 0.151 (52) | 190 |
| 0.02 | 249 | 5 | 5 | 0.405 | 0.573 | 0.786 | 4.86 | 0.0005 | 0.802 | 0.105 (56) | 122 |
| 0.01 | 543 | 5 | 5 | 0.689 | 0.667 | 0.852 | 5.01 | 0.0005 | 0.945 | 0.422 (63) | 138 |
| 0 | 4,097 | 5 | 5 | 1.000 | 0.943 | 0.979 | 5.42 | 0.0015 | 1.000 | 0.568 (73) | 177 |

- With TRRUST targets, GATA1's best overlap is 1 of 11 at every cut-off, so T = 0 and every p = 1.
- The uncapped cm p equals the capped one at every cut-off (for example 0.318 at 0.5).
- fakeK at 0.5: p = 0.189, Monte Carlo SE 0.009 (2,000 draws).
- The fake null is degenerate. Random control groups of these sizes never select a feature at cut-off 0.25 or above (0 of 5,000 fake TFs at 0.5). So every knockdown with T ≥ 2 gets the floor p of 0.0005 (BH q 0.005 across the 20 primary TFs). This shows that the knockdown changed something, not that the features know the TF's targets.
- The okd p of 0.059 at 0.5 is the smallest value possible with 16 other knockdowns, so it cannot reach 0.05.
- MAX at 0.5 (1 feature, best ChIP overlap 2): p cm 0.33, rank 73 of 291. TERF2 at 0.5 (1 feature): ChIP overlap 1, T = 0.

**All 20 primary TFs, by cut-off** (`summary_tables.md`, Table A; "BH" = number of TFs with q < 0.05 across TFs):

| Cut-off | Database | Testable TFs | Median K | T ≥ 2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | True TF in top 5% of ranks |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 3 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | ChIP | 7 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 0.1 | ChIP | 9 | 3 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 0.05 | ChIP | 11 | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 0.02 | ChIP | 12 | 8 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| 0.01 | ChIP | 13 | 24 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| 0 | ChIP | 16 | 258 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| 0.5 to 0.01 | TRRUST | 2 to 8 | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | TRRUST | 11 | 266 | 1 (TFDP1) | 1 | 0 | 0 | 0 | 0 | 1 |

Over all 87 TFs (Table A2 of `summary_tables.md`), 0 TFs pass BH under cm, cml, fakeK or okd at any cut-off.

**Nominal hits** (p < 0.05 under a matched null; none survives BH):

| TF | Database | Cut-off | K | Best overlap | p cm | p fakeK | p okd | Smallest q |
|---|---|---|---|---|---|---|---|---|
| BDP1 | TRRUST | 0 | 1,084 | 2 | 0.009 | 0.27 | 0.36 | q cm 0.52 (87 TFs) |
| SETDB1 | ChIP | 0 | 22 | 4 | 0.036 (cml 0.033) | 0.046 | 0.54 | q fakeK 0.44 |
| TBP | ChIP | 0.05 | 4 | 3 | 0.31 | 0.017 | 0.40 | q fakeK 0.19 |

Each hit appears at one cut-off and under one or two nulls only. With 7 cut-offs × 2 databases × 4 matched nulls × 87 TFs and no correction across cut-offs, a few such p-values are expected.

**Power** (planted targets; mean detection probability at alpha 0.05; one TF: 95% Wilson interval over 200 repeats; several TFs: 95% interval from a bootstrap over TFs; `summary_tables.md` Table D, `power_summary.json`):

| Cut-off | Database | k planted | TFs | rf | cm | cm uncapped | cml | cml uncapped | fakeK |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 4 | GATA1 | 0.88 [0.82, 0.91] | 0.00 [0.00, 0.02] | 0.29 [0.23, 0.36] | 0.00 | 0.29 | 0.88 |
| 0.5 | ChIP | 8 | GATA1 | 1.00 [0.98, 1.00] | 0.00 [0.00, 0.02] | 1.00 [0.98, 1.00] | 0.00 | 1.00 | 1.00 |
| 0.5 | TRRUST | 2 | GATA1 | 1.00 [0.98, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 | ChIP | 8 | 3 (GATA1, MAX, TERF2) | 1.00 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 0.66 | 1.00 | 1.00 |
| 0.1 | ChIP | 8 | 9 | 1.00 [1.00, 1.00] | 0.89 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.86 [0.63, 1.00] | 1.00 | 1.00 |
| 0 | ChIP | 8 | 16 | 0.50 [0.25, 0.75] | 0.56 [0.31, 0.81] | 1.00 [1.00, 1.00] | 0.56 | 0.93 [0.82, 1.00] | 0.50 |

- The cap at 5 removes the power of the capped tests for GATA1. Under the count-matched null, P(best of 7 GATA1 features ≥ 5) = 0.099 (exact; Monte Carlo 0.099). The capped T is at most 5, so its p can never go below 0.099. The uncapped test finds 8 planted targets every time (P(best ≥ 8) = 0.0006 under the null).
- The observed GATA1 result is not hidden by the cap: the observed best overlap is 4, and the uncapped p equals the capped p (0.32).
- TRRUST: 2 planted targets are found every time when the TF has at least 2 TRRUST targets in the universe. TFs with fewer can never pass (mean power 0.75 to 0.88 at lower cut-offs).
- The okd null has no power at 0.5 (16 other knockdowns; smallest possible p 0.059).

**The TRRUST rule of the deployed analysis** (≥ 2 TRRUST targets in the top-20 of a responding feature; `old_trrust_test.csv`):

| Cut-off | 48 TFs of the deployed analysis: with responding features / pass | 20 primary TFs | All 87 TFs |
|---|---|---|---|
| 0.5 | 10 / 0 | 3 / 0 | 17 / 0 |
| 0.25 | 17 / 0 | 7 / 0 | 34 / 0 |
| 0.1 | 25 / 0 | 9 / 0 | 47 / 0 |
| 0.02 | 30 / 0 | 12 / 0 | 57 / 0 |
| 0 | 41 / 2 (BDP1, ERCC2) | 16 / 1 (TFDP1) | 74 / 3 |

Every pass has p rf ≥ 0.05. GATA1 meets this rule at no cut-off (best TRRUST overlap 1).

## Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these cross-checks (Checks table): a second model path without the hook code for 12 cells (`scripts/v3_tf_specificity_verify.py`, `checks/verify.json`); the GATA1 numbers re-derived from the written tables by gene name; the exact nulls compared with Monte Carlo; the Mann–Whitney code compared with scipy; and the calibration checks (`scripts/v3_tf_specificity_calib_check.py`, `calibration.json`, `calibration_extra.json`), including a second, independent overlap count for the fakeK null. All agreed.

## Limits

- One cell line (K562), one layer (5) and one SAE training seed. Whether another seed gives the same GATA1 features was not tested.
- Two target databases only (DoRothEA ChIP-seq for 20 TFs, TRRUST).
- The fake-TF null cannot measure specificity, because control groups select no feature at cut-offs ≥ 0.25. It is reported next to the K-matched and other-knockdown nulls.
- Fake TFs share one reference group and one pool. Their p-values are calibrated given that pool, but they are not independent draws from the population.
- No correction across the 7 cut-offs or across the nulls. BH is applied only across TFs.
- The rank test is descriptive: target sets of different TFs differ in size and gene mix.
- Power was simulated for the 20 primary TFs only.
- GATA1 and six other TF genes are not in the 6,546-gene panel, so their knockdown strength cannot be checked.
- The top-20 definition (mean activation when active, no minimum count) is kept, so rare genes can enter top-20 lists. Only the matched nulls (cm, cml) correct for this.
- Gene length is the genomic span from a gene-position table that is not part of the release; the lengths used are in the `gene_length_bp` column of `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/task_data/gene_universe.tsv`; 54 of 6,328 genes have none and are put in the middle tertile.

## Files

- Scripts (`projects/maxtoki/runs/sae-atlas-217M/scripts/`): `v3_tf_specificity_extract.py` (cell manifest, encoding check, forward passes to block 5, per-cell SAE summaries), `v3_tf_specificity_taskdata.py` (gene universe, target sets, top-20 catalog), `v3_tf_specificity_stats.py` (selection, nulls, rank, power, BH, calibration), `v3_tf_specificity_calib_check.py`, `v3_tf_specificity_verify.py`, `v3_tf_specificity_summary.py` (GATA1 bootstrap and detail, tables, `run_config.json`). Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py`.
- Outputs (`projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/`): `tf_results.csv` (all TFs × cut-offs × databases, all p and q), `summary.json`, `summary_tables.md`, `summary_counts.csv`, `power.csv`, `power_summary.json`, `old_trrust_test.csv`, `gata1_bootstrap.json`, `gata1_detail.json`, `knockdown_efficiency.json`, `calibration.json`, `calibration_extra.json`, `checks/verify.json`, `prepare.json`, `cell_manifest.csv`, `cell_manifest_meta.json`, `cells/` (per-cell summaries and the token ids fed to the model), `fake_null/`, `stats_cache/`.
- Reusable data: `.../outputs/v3_tf_specificity/task_data/` with a `README.md` that defines every file (gene universe, top-20 catalog, target sets, responding features with signed effects, p and q for all 87 TFs, per-TF summaries, test results, cell manifest, per-cell feature means for all 9,200 cells in `cell_feature_means.npz`).
- Provenance: `run_config.json` (device, versions, seeds, cell IDs, feature IDs, input sha256, code sha256, wall time per chunk, encoding-check result), `run_config_extract.json`, `run_config_stats.json`.
- Run time: 9,200 forward passes at 0.21 s per cell, about 34 minutes in total (MPS).


# Feature circuits and the CRISPRi direction test (MaxToki-217M, K562)

This analysis builds a map of "edges" between SAE features. An edge means: removing a source feature at one layer consistently changes a target feature at a later layer. It then asks whether the sign of these edges predicts which way a gene moves when its source gene is silenced by CRISPRi.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4.
- **SAEs.** The 12 SAEs of section 02.
- **Cells for tracing.** 200 K562 non-targeting control cells from `replogle_concat.h5ad`, drawn with numpy `default_rng(42)` from the 10,691 control rows and sorted (`inputs_v3.load_k562_control_cells(200, pool=200, seed=42)`). Encoded from counts (section 00). 195 of 200 cells are cut at 2,046 genes; tokens per cell 1,434–2,048 (median 2,048).
- **Cells for knockdown labels.** All K562 knockdown cells of `replogle_concat.h5ad` (6,546 genes in the file) and 3,000 K562 non-targeting control cells drawn with `default_rng(42)` from the 10,691.
- **Gene-set databases for feature selection.** GO Biological Process, KEGG, Reactome and TRRUST TF target sets, each cut to the K562 gene panel and kept if they have ≥ 5 genes (2,377 terms).

## Method

**Part 1: choosing source features.** The rule of the deployed analysis picks the features whose top genes are most enriched in gene-set databases.

1. Top-20 catalog per feature and layer: the mean activation over the positions where the feature is active, over all 1,019,996 positions of the 500 SAE training cells (including `<bos>`/`<eos>` as `<SPECIAL>`); the 20 genes with the highest mean.
2. Enrichment: for each feature and term with ≥ 2 shared genes, a hypergeometric p (N = 6,329 distinct names, K = term size, n = list size). Rows with p < 0.1 are BH-adjusted together per layer; significant = q < 0.05.
3. Score = sum of −log10 p over the feature's significant rows. The 30 features with the largest score are taken at each of layers 0, 3, 6 and 9 (120 source features).

**Part 2: tracing edges.** Each source feature is removed in each cell, and the change of every later SAE feature is measured.

1. Edit: `hooks_v2.Ablate(s, [f])`, which adds −z_f(x) · W_dec[:, f] to the live input of block s at every position, including `<bos>`/`<eos>` (section 00).
2. Read-out: the captured live input of every later site t = s + 1 to 11 (t = 11 is the input of `lm_head`), encoded with that layer's SAE. For each cell and each of the 4,928 target features: Δz = mean over positions of (z with the edit − z without it).
3. Per (source, target) pair over the 200 cells: Cohen's d = mean(Δz) / sqrt(var(Δz, ddof = 1) + 10⁻¹²); consistency = max(number of cells with Δz > 0, n − that number) / n.
4. Edge if |d| > 0.5 and consistency > 0.7. "Inhibitory" if mean Δz < 0 (removing the source lowers the target). About 3.8 million (feature, target) pairs are tested, with no multiple-testing correction (the rule of the deployed analysis).
5. Exact speed-ups: the forward pass starts at block s from the stored clean input of block s; a source feature that is zero at every position gives Δz = 0 without a forward pass.

**Part 3: knockdown log-fold-changes (LFC).** One log only.

1. n_ig = round(expm1(X_ig) / u_i), with u_i = the smallest non-zero expm1(X) in cell i. Every cell was checked to be whole-number multiples of u_i (largest deviation 4.7 × 10⁻⁴; 3 of 50,729 knockdown cells already had u_i = 1). This gives integer counts up to one factor per cell.
2. v_ig = log1p(10,000 · n_ig / Σ_g' n_ig'), summing over the 6,546 genes in the file.
3. LFC(s, g) = mean of v_ig over the K562 cells whose guide targets s − mean of v_ig over the 3,000 control cells.
4. A silenced gene is used only with ≥ 10 cells. LFCs were computed for 1,543 candidate genes (a superset of the top-10 genes of the source features); 394 have ≥ 10 cells.
5. Observed decrease = LFC < 0. No LFC in the records is exactly 0.

**Part 4: CRISPRi direction test.** Each edge says "removing source feature A lowers (or raises) target feature B". The test turns this into gene-level predictions.

1. Records (rule in `projects/maxtoki/runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py:51-162`): pair each top-10 gene of an edge's source feature with each top-10 gene of its target feature (no self pairs). Per gene pair: evidence = number of such triples, n_inhibitory, and max |d|. Keep the pair if evidence ≥ 2 or max |d| > 2.
2. Predicted decrease = n_inhibitory / evidence > 0.5 (a tie counts as "increase").
3. Keep the record if the source gene has a K562 guide with ≥ 10 cells and the target gene is measured.
4. Metrics: accuracy, balanced accuracy, Matthews correlation coefficient (MCC), Cohen's kappa. Baselines: always "decrease" (the majority class), always "increase", the accuracy expected from the two class rates alone, and a target-direction baseline that uses no model. The target-direction baseline predicts each target gene's usual direction in the other 4 of 5 folds of silenced genes (seed 42).
5. Intervals: 95% percentile grouped bootstrap, 2,000 resamples (seed 42). Three ways to group records: (a) silenced genes (n = 248); (b) source-feature groups, i.e. silenced genes that sit in the same set of source features with ≥ 1 edge (n = 150); (c) connected components of silenced genes that share any source feature (n = 42; the largest has 140 genes). The best constant is chosen again in each resample. The target-direction baseline is fitted once on the 5 folds and not refitted inside resamples.
6. Exploratory follow-up: the same metrics in 6 bands of |LFC|.

## Checks

**Source features.**
- Fed the catalogs of the deployed analysis, the enrichment code rebuilds all 12 of its enrichment tables (same rows in the same order, p within 8.5 × 10⁻¹³ relative), and the selection code rebuilds its 120 source features exactly.
- The layer-5 catalog built here equals the catalog built independently in section 03 (4,928 of 4,928 ordered lists identical).
- No tie at the cut-off: score of the 30th vs 31st feature 138.41 vs 137.70 (L0), 113.06 vs 112.98 (L3), 90.58 vs 89.25 (L6), 87.69 vs 86.60 (L9).

**Input encoding.**
- At load time all 200 cells pass `inputs_v3.check_encoding` (counts are whole-number multiples of the cell's smallest value; largest deviation 3.7 × 10⁻⁴).
- Before every chunk, the next ≤ 12 cells are checked again with `assert_encoding_batch` before any forward pass: 27 of 27 chunks pass (317 cell checks, 0 order violations; a few exact ties in either order, which the check allows). The spot check (below) checked its cells the same way (5 of 5 chunks, 200 cells).

**Tracing.**

| Check | Result | Pass |
|---|---|---|
| Zero edit (`hooks_v2.ZeroDelta(s)`, same partial forward, 4 source layers × 200 cells) | largest \|Δz\| = 0.0 in every cell; 0 edges | yes |
| Edges at the very next layer (s → s + 1) | L0→L1 37,382; L3→L4 2,360; L6→L7 2,598; L9→L10 4,138 | yes |
| Partial forward vs full forward (2 cells × 8 features) | largest difference 0.0 in hidden states and in Δz | yes |
| 3 edges recomputed from scratch | see the spot-check table below | yes |
| Edges recomputed a second way (two-pass numpy, no running mean) | same 318,387 edges, all signs equal, largest \|Δd\| 4.7 × 10⁻⁷ | yes |

**Spot check** (`scripts/v3_circuit_spotcheck.py`). Separate code: cells re-drawn from the h5ad file, counts rebuilt with its own code, tokenised with `tokenize_cell`. The edit is a plain forward hook that adds decode(z with f set to 0) − decode(z) (CPU SAE) to the output of the module that produces `hidden_states[s]`. The read-out is `output_hidden_states[t]`. Same rows and tokens as the trace. Three edges drawn with seed 20261001:

| Edge | d (trace) | d (from scratch) | Consistency (trace / scratch) | Largest per-cell difference in Δz |
|---|---|---|---|---|
| L0 F199 → L1 F580, excitatory | 0.6294 | 0.62937 | 0.95 / 0.95 | 4.4 × 10⁻⁹ |
| L9 F2421 → L10 F2835, inhibitory | −0.6047 | −0.60471 | 0.985 / 0.98 | 1.3 × 10⁻⁸ |
| L6 F4315 → L11 F973, excitatory | 0.9334 | 0.93343 | 0.96 / 0.96 | 3.7 × 10⁻⁸ |

The d differences (< 5 × 10⁻⁵) come from 4-decimal rounding in the CSV. In the second edge, one cell has a Δz within 1.3 × 10⁻⁸ of zero, and its sign differs between MPS and CPU; this moves consistency from 0.985 to 0.98.

**Knockdown labels.**
- Without rounding (expm1(X) used directly), the LFC changes by at most 7.6 × 10⁻⁸.
- A second implementation (cells found with `dataset_loader`, controls drawn again, own count code) gives the same LFC for 5 random silenced genes (largest difference 1.1 × 10⁻¹⁴) and the same cell counts.
- On-target knockdown: all 248 silenced genes have their own LFC < 0 (median −0.67; 69% below −0.5).

## Results

**Main findings.**
- The edge map has 318,387 edges; 57.9% are inhibitory. Most come from layer-0 sources.
- The circuit has no detectable skill at predicting which way a gene moves after silencing. Balanced accuracy is 50.19% and MCC is +0.0037; both intervals include chance under all three groupings.
- The circuit is below always saying "decrease" (by 2.2 points) and 8.3 points below a target-direction rule that uses no model (58.54%).

**Source features** (rank order; `source_features.json`; scores, best terms and top-10 genes in `annotation/selection_detail.csv`):
- L0: 3635, 3083, 1495, 4594, 3421, 73, 199, 948, 1213, 1216, 1116, 1715, 3605, 4802, 734, 603, 1762, 968, 136, 2591, 1990, 3810, 3582, 4714, 4321, 1883, 661, 1087, 2785, 3765
- L3: 3050, 1370, 1335, 1952, 2982, 3262, 4036, 150, 3662, 4912, 3708, 2737, 1949, 3020, 4029, 693, 3043, 4361, 1863, 2050, 3295, 2256, 2217, 3150, 778, 1891, 4162, 1885, 3788, 4801
- L6: 3559, 271, 2942, 4456, 3396, 925, 4212, 4255, 1371, 2337, 4818, 3937, 3421, 1422, 4315, 1508, 3798, 2720, 3018, 2123, 840, 3397, 991, 4029, 262, 1645, 922, 3654, 376, 2700
- L9: 1468, 1107, 1873, 2007, 4645, 1826, 271, 630, 1294, 3199, 1879, 3683, 1652, 4255, 1089, 3493, 198, 3971, 2551, 2421, 1386, 4880, 1863, 2753, 1137, 2680, 4170, 813, 3664, 56
- Rank 1 per layer: L0 F3635 (score 481; best term Reactome G1/S transition; top genes include ST13, FKBP4, STIP1, TCP1), L3 F3050 (352; KEGG proteasome), L6 F3559 (372; KEGG proteasome), L9 F1468 (382; GO BP proton transmembrane transport).
- Features with at least one significant enrichment (of 4,928): L0 1,979 of 2,497 alive; L3 3,463; L6 3,202; L9 3,146.

**Edges by layer pair** (`edges_per_layer_pair.csv`):

| Source → target | Edges | Share inhibitory |
|---|---|---|
| L0 → L1 | 37,382 | 0.581 |
| L0 → L2 | 34,741 | 0.594 |
| L0 → L3 | 33,911 | 0.584 |
| L0 → L4 | 24,369 | 0.583 |
| L0 → L5 | 22,319 | 0.594 |
| L0 → L6 | 23,143 | 0.574 |
| L0 → L7 | 20,697 | 0.564 |
| L0 → L8 | 19,963 | 0.556 |
| L0 → L9 | 19,353 | 0.541 |
| L0 → L10 | 18,544 | 0.521 |
| L0 → L11 | 17,413 | 0.568 |
| L3 → L4 | 2,360 | 0.728 |
| L3 → L5 | 2,207 | 0.689 |
| L3 → L6 | 2,447 | 0.518 |
| L3 → L7 | 2,253 | 0.543 |
| L3 → L8 | 2,149 | 0.560 |
| L3 → L9 | 2,137 | 0.566 |
| L3 → L10 | 2,287 | 0.531 |
| L3 → L11 | 2,438 | 0.541 |
| L6 → L7 | 2,598 | 0.639 |
| L6 → L8 | 3,134 | 0.682 |
| L6 → L9 | 3,695 | 0.684 |
| L6 → L10 | 4,302 | 0.718 |
| L6 → L11 | 4,863 | 0.531 |
| L9 → L10 | 4,138 | 0.683 |
| L9 → L11 | 5,544 | 0.582 |
| **Total** | **318,387** | **0.579** |

- By source layer: L0 271,835 edges (57.3% inhibitory); L3 18,278 (58.4%); L6 18,592 (64.5%); L9 9,682 (62.5%). Edges at the next layer: 46,478.
- Edges per source feature: median 567, mean 2,653, max 48,035. Every source feature has at least one edge. Median per layer: L0 7,947; L3 560; L6 553; L9 281.
- |d| of edges at the 10th / 50th / 90th / 99th percentile: 0.52 / 0.66 / 1.53 / 3.50. 18.4% of edges have |d| > 1.
- The feature with the most edges is L0 F1883 (48,035 edges, 15% of all edges). In the 500 SAE training cells it fires at 100% of `<bos>`/`<eos>` positions and at 1.5% of gene positions. It is the only source feature that fires on most special-token positions.
- The source features are active in almost every cell: only 202 of 24,000 (cell, feature) cases have no active position.
- A symmetric consistency rule (max(#Δz > 0, #Δz < 0) / n > 0.7) gives 316,461 edges. Only 1,926 edges (0.6%) depend on how Δz = 0 is counted.

**CRISPRi records.** All 318,387 edges have top-gene lists at both ends. Result: 796,304 records, 248 silenced genes (31–570 cells each; median 110; 31,539 cells in total), 6,328 target genes. 77,976 records (9.8%) are ties and are predicted "increase".

**Main table** (`crispri/results.json`; interval: percentile grouped bootstrap by silenced gene, n = 248, 2,000 resamples):

| Quantity | Value |
|---|---|
| Observed decrease / increase | 0.5248 / 0.4752 |
| Predicted decrease / increase | 0.5211 / 0.4789 |
| Confusion (pred dec & obs dec, pred inc & obs dec, pred dec & obs inc, pred inc & obs inc) | 218,513 / 199,396 / 196,450 / 181,945 |
| **Accuracy** | **0.5029** [0.4987, 0.5070] |
| Always "decrease" (majority class) | 0.5248 |
| Always "increase" | 0.4752 |
| Recall on decreases / on increases | 0.523 / 0.481 |
| **Balanced accuracy** | **0.5019** |
| **MCC** | **+0.0037** |
| Cohen's kappa | +0.0037 |
| Accuracy expected from the two class rates | 0.5010 |
| Majority class learned on other folds of silenced genes | 0.5248 |
| **Target-direction baseline** (no model) | **0.5854** (balanced 0.582, MCC 0.165) |
| Majority within the same silenced gene, other target folds (uses that knockdown's own labels; reference only) | 0.5719 |

**Intervals under three groupings** (95% percentile grouped bootstrap, 2,000 resamples):

| Statistic | Point | By silenced gene (n = 248) | By feature group (n = 150) | By component (n = 42) |
|---|---|---|---|---|
| Accuracy − best constant | −0.0219 | [−0.0369, −0.0062] | [−0.0447, −0.0001] | [−0.0387, +0.0063] |
| Balanced accuracy − 0.5 | +0.0019 | [−0.0018, +0.0057] | [−0.0026, +0.0068] | [−0.0006, +0.0130] |
| MCC | +0.0037 | [−0.0037, +0.0115] | [−0.0052, +0.0138] | [−0.0013, +0.0261] |
| Accuracy − chance from class rates | +0.0018 | [−0.0018, +0.0057] | – | – |
| Accuracy − target-direction baseline | −0.0826 | [−0.0918, −0.0730] | – | – |

- Balanced accuracy and MCC are at chance under every grouping.
- Raw accuracy is above 50% only because both the circuit and the data lean a little towards "decrease".
- The circuit is below always-decrease (the interval is below 0 by silenced gene and by feature group; by component it includes 0).
- Fold by fold (5 folds of silenced genes, seed 42): circuit 0.495–0.511; learned majority 0.516–0.536; target-direction baseline 0.566–0.594. The circuit is below the baseline in every fold.

**Per silenced gene** (mean over the 248 genes; interval: percentile bootstrap over silenced genes, 2,000 resamples):

| Quantity | Value |
|---|---|
| Mean accuracy | 0.5029 [0.4983, 0.5070] |
| Mean accuracy − always decrease | −0.0167 [−0.0283, −0.0049] |
| Mean balanced accuracy − 0.5 | +0.0010 [−0.0009, +0.0029] |
| Mean MCC | +0.0020 [−0.0020, +0.0061] |
| Mean (median) predicted-decrease share | 0.535 (0.512) |
| Genes where the circuit beats always-decrease | 81 |
| Genes with balanced accuracy > 0.5 / MCC > 0 | 126 / 126 (2 genes get one sign for all targets) |
| Genes where the circuit beats the target-direction baseline | 19 |

**Sensitivity analyses** (same edges):

| Variant | Accuracy | Balanced accuracy | MCC |
|---|---|---|---|
| Ties predicted as "decrease" | 0.5058 | 0.4999 | −0.0001 |
| Only edges whose two ends carry an enrichment label (217,212 edges, 713,995 records) | 0.5029 | 0.5021 | +0.0042 |
| Sign of the summed d instead of the majority sign | 0.5043 | 0.5007 | +0.0013 |
| Labels from the file's own full-library total (single log; agree with the main labels in 99.3% of records, r = 0.9995) | 0.5035 | 0.5023 | +0.0046 [−0.0024, +0.0119] (by silenced gene) |

With the file-total labels, always-decrease gives 0.5292 and the target-direction baseline 0.5866.

**By size of the observed change (exploratory).** The bands were planned. The Bonferroni and component intervals were added after looking at these numbers. Intervals: percentile grouped bootstrap by silenced gene (n = 248, 2,000 resamples); Bonferroni = adjusted for 6 bands; component = grouped by the 42 components.

| \|LFC\| band | Records | Observed decrease | Accuracy | Always-decrease | Balanced − 0.5 [95% CI] | Bonferroni CI | Component CI | Target baseline (accuracy / balanced) |
|---|---|---|---|---|---|---|---|---|
| < 0.01 | 144,824 | 0.506 | 0.497 | 0.506 | −0.0031 [−0.0056, −0.0006] | [−0.0064, +0.0002] | [−0.0045, +0.0032] | 0.513 / 0.513 |
| 0.01–0.025 | 190,600 | 0.519 | 0.499 | 0.519 | −0.0013 [−0.0047, +0.0019] | [−0.0059, +0.0033] | [−0.0044, +0.0020] | 0.542 / 0.540 |
| 0.025–0.05 | 210,107 | 0.528 | 0.501 | 0.528 | −0.0003 [−0.0041, +0.0035] | [−0.0054, +0.0049] | [−0.0027, +0.0095] | 0.587 / 0.583 |
| 0.05–0.1 | 165,582 | 0.533 | 0.505 | 0.533 | +0.0034 [−0.0028, +0.0094] | [−0.0048, +0.0118] | [−0.0010, +0.0210] | 0.638 / 0.631 |
| 0.1–0.25 | 75,098 | 0.537 | 0.520 | 0.537 | **+0.0181** [+0.0063, +0.0297] | **[+0.0019, +0.0351]** | [+0.0044, +0.0456] | 0.696 / 0.684 |
| ≥ 0.25 | 10,093 (179 genes) | 0.613 | 0.534 | 0.613 | +0.0284 [+0.0032, +0.0537] | [−0.0053, +0.0655] | [−0.0106, +0.0547] | 0.727 / 0.700 |

- The 0.1–0.25 band (9.4% of records) shows a small positive result: balanced accuracy 51.8%, MCC +0.036. It survives the Bonferroni and component intervals. The ≥ 0.25 band survives neither.
- This is weak. In the same records the circuit is still below always-decrease (0.520 vs 0.537), and the target-direction baseline reaches 68.4% balanced accuracy, 17 points more.
- Treat it as a hint for a later planned test, not as a finding.

## Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these second-way checks (`outputs/v3_circuit/verify/verify.json`, script `scripts/v3_circuit_verify.py`; all pass):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Edge set | running mean and variance over cells | two-pass numpy in two halves | identical 318,387 edges, all signs; largest \|Δd\| 4.7 × 10⁻⁷ |
| Records | vectorised bincount | pandas merge and groupby | identical 796,304 records (evidence, n_inhibitory, prediction, LFC) |
| Accuracy, balanced accuracy, MCC | confusion-count formulas | scikit-learn | equal to machine precision |
| Target-direction baseline | pandas groupby | plain loop | 0.585446 both |
| Interval, balanced accuracy − 0.5 (by gene) | [−0.0018, +0.0057] (seed 42) | own loop, seed 7: [−0.0019, +0.0057] | within Monte Carlo error |
| Interval, MCC (by gene) | [−0.0037, +0.0115] | [−0.0039, +0.0114] | within Monte Carlo error |
| LFC, 5 silenced genes | main script | separate code | largest difference 1.1 × 10⁻¹⁴; same cell counts |

The spot check of 3 edges with fully separate code (Checks) covers all 200 cells.

## Limits

- **No null for edge counts.** Random source features or random directions of matched size were not run. So "318,387 edges" is a count under the edge rule, not evidence that the edges mean something.
- **The selection rule was not tested.** It picks features by enrichment strength, and annotation rates of this kind are close to chance in this project. Other rules would give other source features.
- **One SAE seed**, one cell type (K562) and four source layers.
- **Design choices of the deployed analysis were kept**: edits at every position including `<bos>`/`<eos>` (L0 F1883 gives 15% of all edges); no correction for about 3.8 million (feature, target) tests; a consistency rule that counts Δz = 0 against "positive" (only 0.6% of edges depend on it).
- **CRISPRi limits.** No significance filter on the LFC; K562 only; the target-direction baseline is not refitted inside resamples; records share genes, which the grouped bootstraps handle only in part. The band analysis was not planned.
- **Downstream uses of the edge list** (pointwise mutual information, knowledge graph, disease mapping, edge-density audit) were not run.
- Cells 0–18 and the source-feature selection were run in a separate session with the same scripts (hashes recorded per chunk). They are covered by the per-cell file checks, the spot check (all 200 cells) and the second-way edge computation, but those 19 cells were not traced a second time.
- Other jobs shared the machine, so per-cell timings (38–100 s, mean 57 s) are not clean benchmarks. The exact zero-edit result and the spot-check agreement show that results were not affected.

## Files

- Scripts (`projects/maxtoki/runs/circuit-tracing-217M/scripts/`): `v3_circuit_annotate.py` (catalogs, enrichments, source-feature selection, checks), `v3_circuit_trace.py` (trace; resumable; `--check-partial`), `v3_circuit_aggregate.py` (per-cell vectors to edges), `v3_circuit_spotcheck.py`, `v3_circuit_lfc.py`, `v3_circuit_crispri.py` (records, metrics, baselines, bootstraps; metric code from `v2_circuit_crispri.py`; record rule from `audit_a2_groupkfold_crispri.py`), `v3_circuit_bands.py`, `v3_circuit_verify.py`, `v3_circuit_finalize.py`. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`.
- Outputs (`projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/`):
  - edges: `circuit_edges_v3.csv` (src_layer, src_feature, tgt_layer, tgt_feature, cohens_d, consistency, sign, mean_delta, n_pos, n_neg, n_zero, consistency_sym, passes_symmetric_rule), `edge_stats_main.npz`, `edges_per_layer_pair.csv`, `edges_per_source_feature.csv`, `aggregate_summary.json`;
  - source features: `source_features.json`, `annotation/` (catalogs, enrichments, `selection_detail.csv`, `checks.json`, `summary.json`, `run_config.json`);
  - trace: `cells_tokens.npz`, `cells_encoding.json`, `trace_cells/` (200 files), `trace_progress.json` (wall time, memory and encoding check per chunk), `combos_main.csv`, `combos_ctrl.csv`, `checks/partial_forward_check.json`, `spotcheck/`;
  - labels: `lfc/` (`lfc_panel.npy` main labels, `lfc_file.npy` file-total labels, `sources.csv` with cell counts and on-target LFC, `control_rows.npy`, `checks.json`, `run_config.json`);
  - CRISPRi: `crispri/` (`results.json`, `per_source_metrics_v3_edges.csv`, `bands_followup.json`, `run_config.json`, `run_config_bands.json`), `per_pair.parquet` (796,304 records);
  - checks and provenance: `verify/verify.json`, `verify/run_config.json`, `run_config.json` (device, versions, seeds, the 200 dataset rows and token counts, the 120 source features, SAE and code sha256, encoding-check result per chunk, wall time per chunk).
- Run time: 27 chunks of ≤ 8 min, 3.23 h in total (MPS).


# Feature triplets: do three SAE features interact? (MaxToki-217M, K562)

This analysis removes three SAE features (one each at layers 0, 5 and 9) alone, in pairs and all together. It asks whether removing all three does more or less than the sum of the parts. The non-additive part is the three-way interaction term. It is measured on the layer-11 SAE codes and on the logits.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4. CPU threads capped at 4.
- **SAEs.** The SAEs of section 02 at L0, L5 and L9 (edits) and L11 (read-out).
- **Cells.** 20 K562 non-targeting control cells of `replogle_concat.h5ad`: rows 289300, 292833, 293501, 294821, 296499, 296999, 297671, 298043, 298575, 304014, 304434, 306407, 308960, 310091, 310869, 314183, 315807, 316745, 317964, 322713. They are the first 20 cells that tokenize from a pool of 100 drawn with numpy `default_rng(42)` and sorted (`inputs_v3.load_k562_control_cells(20, pool=100, seed=42)`). Barcodes are in `cells.json`.
- **Input.** Counts rebuilt and encoded as in section 00 (largest deviation from a whole-number multiple 1.3 × 10⁻⁴). All 20 cells are cut at 2,046 genes, so every cell has 2,048 tokens.

## Method

1. **Feature choice (made before any ablation).** At each of L0, L5 and L9:
   - candidates = SAE features active (z > 0) in ≥ 1% of all tokens of the 20 clean cells (all positions): L0 1,750, L5 460, L9 618;
   - the log10 activation frequency of the candidates is cut into 8 equal-count bins (octiles);
   - one feature is drawn at random from each bin with numpy `default_rng(20261001 + layer)`.
   The choice was written to `selection.json` at 13:40:16, before the first ablation (13:41:51). Chosen frequencies are 1.0% to 2.1%. Every chosen feature is active in all 20 cells, and none fires at `<bos>`. The "annotated" label below comes from the enrichment annotation of section 04. It was added after the choice and is a description only.

   | Layer | Feature | Octile | Frequency in 20 cells | Frequency in SAE training cells | Active positions per cell | Single effect eff_X (× 10⁻⁴) | Annotated (post hoc) | Top term |
   |---|---|---|---|---|---|---|---|---|
   | L0 | 763 | 0 | 0.0113 | 0.0109 | 23.1 | 1.78 | yes | GO negative regulation of programmed cell death |
   | L0 | 3035 | 1 | 0.0119 | 0.0121 | 24.4 | 3.00 | yes | Reactome transcriptional regulation by TP53 |
   | L0 | 2518 | 2 | 0.0137 | 0.0132 | 28.1 | 1.54 | yes | GO histone H3-K4 methylation |
   | L0 | 3778 | 3 | 0.0147 | 0.0137 | 30.1 | 2.64 | no | – |
   | L0 | 859 | 4 | 0.0158 | 0.0159 | 32.4 | 2.60 | yes | TRRUST E2F6 |
   | L0 | 293 | 5 | 0.0172 | 0.0173 | 35.2 | 2.76 | yes | Reactome synthesis of active ubiquitin |
   | L0 | 1910 | 6 | 0.0195 | 0.0191 | 40.0 | 2.92 | yes | GO tRNA modification |
   | L0 | 3275 | 7 | 0.0208 | 0.0207 | 42.6 | 6.36 | yes | KEGG Cushing syndrome |
   | L5 | 3812 | 0 | 0.0104 | 0.0098 | 21.2 | 0.85 | yes | Reactome metabolism of lipids |
   | L5 | 4369 | 1 | 0.0108 | 0.0094 | 22.1 | 0.87 | no | – |
   | L5 | 1812 | 2 | 0.0113 | 0.0126 | 23.1 | 1.50 | yes | GO regulation of endopeptidase activity |
   | L5 | 1550 | 3 | 0.0122 | 0.0127 | 24.9 | 1.10 | yes | GO intracellular protein transport |
   | L5 | 3142 | 4 | 0.0134 | 0.0149 | 27.4 | 2.07 | no | – |
   | L5 | 4903 | 5 | 0.0145 | 0.0163 | 29.8 | 2.40 | no | – |
   | L5 | 4821 | 6 | 0.0168 | 0.0199 | 34.4 | 1.49 | yes | Reactome transport of small molecules |
   | L5 | 1053 | 7 | 0.0213 | 0.0214 | 43.7 | 1.36 | no | – |
   | L9 | 1154 | 0 | 0.0102 | 0.0103 | 20.8 | 0.57 | no | – |
   | L9 | 4302 | 1 | 0.0106 | 0.0119 | 21.6 | 0.63 | yes | GO negative regulation of programmed cell death |
   | L9 | 1535 | 2 | 0.0114 | 0.0135 | 23.4 | 0.59 | no | – |
   | L9 | 235 | 3 | 0.0117 | 0.0114 | 24.1 | 0.67 | yes | GO hemopoiesis |
   | L9 | 494 | 4 | 0.0125 | 0.0131 | 25.6 | 0.72 | no | – |
   | L9 | 110 | 5 | 0.0138 | 0.0142 | 28.2 | 0.78 | yes | KEGG ribosome biogenesis in eukaryotes |
   | L9 | 2772 | 6 | 0.0154 | 0.0156 | 31.5 | 0.97 | no | – |
   | L9 | 1628 | 7 | 0.0166 | 0.0141 | 34.0 | 0.73 | yes | GO NADH dehydrogenase complex assembly |

   eff_X = mean over the 4,928 L11 targets of |cell-mean Δz| for that feature removed alone.

2. **Noise-floor features.** Two features per layer that should do nothing. L0: 1581 and 3049; L9: 1995 and 3537; all four are exactly silent in the 20 cells. No L5 feature is silent in these cells, so the 2 rarest L5 features were used: 3437 (active at 4 positions in 3 cells) and 4481 (40 positions, 2 per cell in all 20 cells). This rule was set before any ablation. Noise-floor triplets: 8 with all three null (2 × 2 × 2) and 24 with one null feature (8 each with null A, B or C; the null-B set uses 3437).
3. **Conditions.** 512 real triplets (8 × 8 × 8) and 32 noise-floor triplets. Each triplet needs the singles A, B, C, the pairs AB, AC, BC and the triple ABC. In total 826 distinct conditions × 20 cells = 16,520 ablation runs, plus 20 clean runs.
4. **Edits.** `hooks_v2.Ablate(l, [f])` removes z_f · W_dec[:, f] from the live input of block l at every position (section 00). Edits combine: the L5 code is computed after the L0 edit, and so on. Conditions without an L0 edit start at block 5 or 9 from the stored clean input.
5. **Read-outs per condition and cell.**
   - L11 SAE code change: Δz = mean over the 2,048 positions of (z with edits − z clean), 4,928 values.
   - Logit change: mean over positions, centred over the vocabulary, 20,275 values.
   - KL(clean ‖ edited) of the next-token distribution, mean over positions.
   Position means are computed as sums over the position axis.
6. **Effects.** e_X[t] = mean over cells of Δz[t] under condition X. Three-way term I_ABC = e_ABC − e_AB − e_AC − e_BC + e_A + e_B + e_C, computed per cell first. Pairwise term I_XY = d_XY − d_X − d_Y.
7. **Tests.**
   - Per target: one-sample t-test of the per-cell I_ABC over the 20 cells (df 19), and a bootstrap p as a second p-value. BH at q < 0.05 within each triplet (4,928 tests) and across all 512 × 4,928 = 2,523,136 tests.
   - Per triplet (aggregate): T = Σ_t |mean over cells of I_ABC[t]|. Null = 999 random per-cell sign flips (seed 20261002). BH across the 512 triplets.
   - Count null: the same 999 sign flips applied to all triplets at once (each flip ranked inside each triplet's own values). This gives a null for the number of triplets at p < 0.05 that keeps the dependence between triplets (they share cells and conditions).
   - Per feature: a pooled sign-flip test over each feature's 64 triplets, with BH over the 24 features.
8. **Size of the interaction.** Noise-adjusted energy: ‖true mean‖² is estimated as ‖mean‖² − Σ_t var_t / n. Energy share = noise-adjusted energy of I_ABC / noise-adjusted energy of the joint effect e_ABC. ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖², noise-adjusted. ρ2 is the same for pairs. Intervals: 95% basic bootstrap, resampling cells (n = 20), 1,000 multinomial cell-weight vectors (seed 20261001) shared by all triplets and targets. The basic interval pivots on the unadjusted value, because the bootstrap of the adjusted energy centres on the unadjusted one.
9. **Power.** A planted interaction is added to d_ABC of 64 seeded triplets (seed 20261003), at 5%, 10%, 20%, 50% and 100% of each cell's own joint effect, and as a constant at 2%, 5% and 10% of the cell-mean joint effect.
10. **Redundancy ratios** (formulas of the deployed analysis). With eff_X = mean over targets of |e_X[t]|: pairwise AB = eff_AB / (eff_A + eff_B) (likewise AC, BC); three-way = eff_ABC / (eff_A + eff_B + eff_C); marginal C given AB = (eff_ABC − eff_AB) / eff_C. Each is also computed for an exactly additive joint effect (e_ABC set to e_A + e_B + e_C).
11. **Superadditivity rules.** Rule of the deployed analysis: |e_ABC| > 1.1 × (|e_A| + |e_B| + |e_C|), over targets with |e_ABC| > 0.01. Rule of the pipeline specification: |d_ABC| > |d_A| + |d_B| + |d_C|, over targets with |d| > 0.5 in any condition, where d is Cohen's d_z over the 20 cells; also in bands of the ratio of |d|.
12. **Per-position read-out check.** 4 cells × 4 seeded triplets (seed 20261021) × 8 full forward passes. Per-position energy share of I_ABC in the joint change at four read-outs.

## Checks

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding | `inputs_v3.check_encoding` on every cell right before its clean forward pass, in every model stage | 20 / 20 cells pass in every stage; 0 order violations; 2 positions with exact ties ordered differently (allowed). Tokens ranked from the stored log1p values fail 20 / 20 | yes |
| Zero edit | `ZeroDelta` at L0, L5, L9 and all three vs clean, 2 cells (MPS); cell 0 again with CPU float64 sums | largest logit difference 0.0; largest Δz 0.0; KL 0.0 | yes |
| Clean repeat | two clean passes | 0.0 | yes |
| Shortened path = full path | 2 conditions per condition type, 2 cells | largest difference 0.0 (Δz and logits) | yes |
| float32 vs float64 position mean | 1 triple condition, 2 cells | largest difference 8.0 × 10⁻⁹ | yes |
| AB vs B | cell-mean Δz of AB vs B, all 64 (A, B) pairs | 0 of 64 equal; smallest largest-difference 0.020 (median 0.049) | yes |
| ABC vs C | same, all 512 triplets | 0 of 512 equal; smallest 0.019 (median 0.049) | yes |
| Silent features | triplets with an exactly silent L0 or L9 feature (8 all-null, 8 null-A, 8 null-C) | I_ABC = 0.0 exactly in every cell and target | yes |
| Near-silent L5 feature | 8 null-B triplets (L5 3437, 4 active positions in 3 cells) | I ≠ 0 only in those cells: largest cell-mean \|I\| 2.4 × 10⁻⁴; 0 of 8 with any interval excluding 0; 1 of 8 at aggregate p < 0.05 | yes |
| Paired conditions | a condition with a silent null feature must equal the same condition without it (separate passes, often separate chunks) | 2,144 of 2,144 bit-identical | yes |
| Results vary across triplets | SD of the three-way ratio across triplets | 0.042 | yes |
| Stored values | 2 random conditions in each of the 220 groups (440) recomputed from scratch, full model path, CPU float64 sums | largest difference Δz 1.1 × 10⁻⁷, logits 2.5 × 10⁻⁸; KL within 0.15%; 220 of 220 stored clean fingerprints correct | yes |
| Read-out vs stored | position-mean Δz of ABC, 16 cell-triplet pairs | largest difference 1.2 × 10⁻⁸ | yes |

## Results

**Main findings.**
- There is no three-way interaction of a size that matters. The noise-adjusted share of the joint effect's energy that is three-way interaction is 0.0000028 (median over 512 triplets; 95% interval of the median [−0.00092, +0.00059]). So it is at most about 0.06%.
- The tests had power: a planted interaction of 5% of each cell's joint effect was found in 64 of 64 triplets.
- Two small effects are visible. One L5 feature shows a borderline excess of nominal three-way hits (found after looking; at most 0.03% of the energy). Two L5 → L9 feature pairs are 3–4% sub-additive.
- The edits are small (2.2% of the L11 residual norm). So the result says the model is smooth for small edits, not that its features never combine.

**Is there a three-way interaction?**

| Test | Result |
|---|---|
| Triplets with ≥ 1 target whose 95% interval of I_ABC excludes 0 (no adjustment) | 512 of 512 (488 targets per triplet on average) |
| Same count when each cell's I gets a random sign (no true interaction) | 488 per triplet (ratio 1.004) |
| Triplets with ≥ 1 target at BH q < 0.05 within the triplet (t-test) | **4 of 512** (up to 25.6 expected by chance) |
| Triplets with ≥ 1 target at BH q < 0.05 across all 2,523,136 tests | **0 of 512** |
| Same two counts with bootstrap p-values | 0 and 0 |
| Smallest t-test p over all tests | 6.7 × 10⁻⁶ |
| Targets with p < 0.05 (no adjustment) per triplet, mean | 110 (5% of the moving targets would be 246) |
| Logits: triplets with any vocabulary logit at BH q < 0.05 (within) | 7 of 512 |
| Aggregate sign-flip test, p < 0.05 | **44 of 512** |
| Count null for that number | mean 25.1, SD 8.4, 95th percentile 40; **p = 0.029** |
| Aggregate sign-flip test, BH across 512 | 0 of 512 |
| T_obs / T_null median (median over triplets) | 1.001 (5–95%: 0.984–1.022) |
| All interaction orders (e_ABC − e_A − e_B − e_C): p < 0.05 / count null / BH | 28 / mean 25.1, SD 13.3, p = 0.38 / 0 |

The 4 per-target hits (one target in one triplet each):

| Triplet (L0, L5, L9) | Target | p | q within | q across all | Mean I | \|I\| / \|e_ABC\| at that target |
|---|---|---|---|---|---|---|
| 2518, 4369, 110 | 4731 | 9.9 × 10⁻⁶ | 0.049 | 0.64 | 1.1 × 10⁻⁸ | 0.00012 |
| 293, 4903, 1154 | 1404 | 7.6 × 10⁻⁶ | 0.037 | 0.64 | −3.4 × 10⁻⁸ | 0.00023 |
| 1910, 1812, 1154 | 1535 | 9.5 × 10⁻⁶ | 0.047 | 0.64 | −9.7 × 10⁻⁹ | 0.00007 |
| 1910, 4903, 235 | 4858 | 6.7 × 10⁻⁶ | 0.033 | 0.64 | −1.5 × 10⁻⁸ | 0.00094 |

- These are not evidence of a three-way interaction. BH within a triplet allows a 5% chance of one false hit per triplet, so up to 25.6 of 512 triplets would show one by chance; 4 do.
- Their size (about 10⁻⁸) is below the float32 rounding of the stored values (the recomputation differs by up to 1.1 × 10⁻⁷).
- The t-test finds them because the L11 read-out has little noise: tiny, consistent shifts in all 20 cells give a small p even at 10⁻⁸.

The excess of nominal aggregate hits (44 of 512) against the count null (p = 0.029):

| Pooled over a feature's 64 triplets | L5 4369 | L9 1535 | All 512 triplets |
|---|---|---|---|
| Triplets at p < 0.05 | 15 of 64 | 10 of 64 | 44 of 512 |
| Pooled sign-flip p (same flips for all its triplets) | **0.002** | 0.008 | 0.181 |
| BH q over the 24 features | **0.048** | 0.096 | – |
| Pooled noise-adjusted energy share of I_ABC [95% basic CI] | 0.000010 [−0.00054, +0.00028] | 0.000014 [−0.00074, +0.00040] | 0.0000021 [−0.00061, +0.00034] |
| Pooled ρ3 [95% basic CI] | 0.9998 [0.9960, 1.0035] | 1.0058 [0.9985, 1.0131] | – |

- The largest per-feature count (15, L5 4369) is above what the null gives for the most-hit of 24 features (null mean 7.9, 95th percentile 12; p = 0.018).
- One feature passes BH over 24, just (q = 0.048). The others have q ≥ 0.096.
- Even for 4369, the three-way term is at most 0.03% of the joint effect's energy (upper interval bound).
- This was found after looking at the results. It is borderline and tiny.

A limit of the aggregate test: with 999 sign flips the smallest possible p is 0.001. BH over 512 triplets can then reject only if at least 11 triplets sit at that floor (over 192 pairs: at least 4). So "0 of 512 after BH" in the aggregate test is partly built into the design. The count null and the pooled tests do not have this limit.

**How big is the interaction?** (medians over the 512 triplets; 95% basic bootstrap interval of the median, cells resampled, n = 20, 1,000 resamples):

| Quantity | L11 SAE codes | Logits |
|---|---|---|
| Noise-adjusted energy share of I_ABC in the joint effect | **0.0000028** [−0.00092, +0.00059] | about 10⁻⁸ or less (see note) |
| Same, all interaction orders | 0.0000078 [−0.0041, +0.0027] | – |
| Triplets whose own interval of the I_ABC share is above 0 | 0 of 512 | – |
| Noise-adjusted ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖² | **1.0037** [0.9977, 1.0112] | 1.00055 [1.00047, 1.00067] |
| ρ3 without the noise adjustment | 0.998 | 1.00057 |
| Triplets with a ρ3 interval above 1 / below 1 | 28 / 11 | – |
| Noise-adjusted pairwise ρ2 (mean of AB, AC, BC) | 1.0020 [0.9993, 1.0063] | – |
| Per-cell I_ABC energy share (median over cells, then triplets) | 0.0091 | 2.1 × 10⁻¹⁰ |

- On the logits, the unadjusted share is 1.6 × 10⁻⁸ and the noise-adjusted median is −9 × 10⁻¹³. At this size the values are float32 rounding, and the interval is not meaningful.
- On the logits, ρ3 = 1.00055 excludes 1 by 0.055%. That is the pairwise part (below); it is negligible.
- 28 triplets have a ρ3 interval above 1 and 11 below. If the intervals were exact, about 13 per side would be expected. With 20 cells the basic intervals are likely somewhat too narrow.
- KL(clean ‖ edited) per position, median over conditions: singles 2.0 × 10⁻⁴, pairs 5.2 × 10⁻⁴, triples 7.5 × 10⁻⁴ nats. It grows roughly with the number of edits (1 : 2.6 : 3.8).
- For the joint edit ABC, a median of 274 of 4,928 targets per triplet pass BH.

**Power** (planted interaction added to d_ABC of 64 seeded triplets):

| Planted interaction | Aggregate sign-flip test: p < 0.05 / BH within 64 | Per-target t-test: triplets with ≥ 1 BH target (within / across 64) |
|---|---|---|
| 5% of each cell's own joint effect | 64 / 64 | 64 / 64 |
| 10% | 64 / 64 | 64 / 64 |
| 20% | 64 / 64 | 64 / 64 |
| 50% | 64 / 64 | 64 / 64 |
| 100% | 64 / 64 | 64 / 64 |

- A planted interaction that is the same in every cell is found even more easily: 2% of the cell-mean joint effect (0.04% of its energy) gives 64 of 64 by the aggregate test. Its energy-share interval is above 0 in 5 of 64 at 2%, 57 of 64 at 5% and 64 of 64 at 10%.
- The L11 SAE read-out is dense: in a typical cell 4,768 of its 4,928 features are active somewhere. So a small edit moves most targets in every cell, and the change is consistent across cells. The median number of targets with a BH-significant single effect per triplet is 415 (A), 284 (B) and 1,502 (C).

**Pairwise interactions** (192 pairs: 64 each of AB, AC, BC):

| Quantity | Result |
|---|---|
| Pairs at aggregate sign-flip p < 0.05 | 17 of 192 (count null mean 9.4, SD 3.8; p = 0.048) |
| Pairs passing BH (aggregate) | 0 of 192 (BH needs ≥ 4 pairs at the p floor) |
| Pairs with ≥ 1 target at BH q < 0.05 (within pair) | **29 of 192** (up to 9.6 expected by chance) |
| Pairs whose interval of the I_XY energy share is above 0 | 0 of 192 |
| Pairs with a ρ2 interval below 1 / above 1 | 5 / 12 |
| Median noise-adjusted ρ2 | 1.0012 |
| \|I_XY\| share of \|e_XY\| (observed / sign-flip noise) | 0.0807 / 0.0811 |

Two pairs stand out. Both are an L5 feature followed by an L9 feature (ρ2 intervals: 95% basic bootstrap over cells, n = 20, 1,000 resamples):

| Pair (L5 → L9) | Targets at BH q < 0.05 | ρ2 [95% CI] | Aggregate p | cos(I_XY, e of the L9 feature) | L9 feature active positions per cell: alone → after the L5 edit |
|---|---|---|---|---|---|
| 3812 → 110 | 1,537 | 0.961 [0.939, 0.980] | 0.003 | −0.38 | 28.3 → 28.0 (changes in 7 of 20 cells) |
| 1812 → 1628 | 623 | 0.966 [0.944, 0.993] | 0.001 | −0.40 | 34.1 → 32.2 (changes in 15 of 20 cells) |

- Removing both features changes the L11 codes by 3.4–3.9% less energy than the two single removals added up.
- The interaction term itself is small (energy share 0.0013 and 0.0007; intervals include 0). The shortfall comes from its overlap with the additive part: it points partly against the L9 feature's own effect.
- Reading: the L5 feature helps drive the L9 feature. Once the L5 feature is removed, less of the L9 feature is left to remove. For 1812 → 1628 the L9 feature is active at fewer positions after the L5 edit. For 3812 → 110 the count barely changes, so the effect is probably on activation values, which were not recorded.
- The other 27 pairs with a per-target BH hit have 1–35 hits each. At those targets |I_XY| is 0.008–3.2% of the joint effect (median per pair). 24 of the 27 have a ρ2 interval that includes 1.
- 12 pairs have a ρ2 interval above 1 (the joint effect is 0.5–2.2% larger in energy than the sum); 6 of them involve L0 feature 763. 5 pairs have an interval below 1 (the two above, plus three with no per-target hit). If the intervals were exact, about 5 per side would be expected. These are weak signs, not findings.

**Where the per-cell non-additivity of the L11 codes comes from** (4 cells × 4 triplets; triplets (763, 4369, 110), (3035, 4903, 110), (1910, 3812, 110), (1910, 4369, 2772); per-position energy share of I_ABC in the joint change, median (range) over 16 cell-triplet pairs):

| Read-out | Per position | Position mean |
|---|---|---|
| L11 residual stream (input of `lm_head`, dense) | 5.0 × 10⁻⁹ (1.8 × 10⁻⁹ to 7.0 × 10⁻⁴) | 8.7 × 10⁻¹¹ |
| L11 SAE pre-activation (dense, linear in h) | 8.2 × 10⁻⁹ (2.8 × 10⁻⁹ to 6.3 × 10⁻⁴) | 1.4 × 10⁻¹⁰ |
| Logits (centred) | 7.1 × 10⁻⁹ (2.6 × 10⁻⁹ to 7.6 × 10⁻⁴) | 1.1 × 10⁻¹⁰ |
| **L11 SAE code (TopK 32)** | **0.046** (0.033 to 0.074) | 0.0068 |

- 99.99% of the code-level I_ABC energy sits on entries where the top-32 set differs between conditions. 90% of the joint code change itself sits on such entries.
- Under ABC, 24% of positions change their top-32 set; 0.31 features are swapped per position.
- The joint change of the L11 residual is 2.2% of its norm (median; 1.6–4.1%).
- So the model responds additively. The apparent non-additivity of the codes comes from features entering or leaving the top 32.

**Superadditivity:**

| Rule | Pool | Superadditive | Other classes |
|---|---|---|---|
| Deployed analysis: \|e_ABC\| > 1.1 × (\|e_A\| + \|e_B\| + \|e_C\|), targets with \|e_ABC\| > 0.01 | 6,738 targets pooled over 512 triplets | 3 (0.04%), in 3 of 512 triplets | – |
| Pipeline specification, strict: \|d_ABC\| > \|d_A\| + \|d_B\| + \|d_C\|, targets with \|d\| > 0.5 in any condition | 1,528,321 targets (60.6% of 2,523,136) | 5,407 (0.35%), in 512 of 512 triplets | – |
| Pipeline specification, banded (ratio of \|d\|) | same | > 1.1×: 3,344 (0.22%) | 0.9–1.1×: 7,168 (0.47%); < 0.9×: 1,517,809 (99.3%) |

- These rules have no valid null, so the counts are not evidence for or against synergy.
- The inclusion rule keeps 60.6% of targets because effects are consistent across cells (pure noise would keep 23.5%).
- "Sub-additive" is the usual outcome when |d_ABC| is compared with a sum of three |d|: with the raw-mean rule at 0.9×, an exactly additive joint effect is already called sub-additive for 44.7% of included targets (observed 56.7%).

**Redundancy ratios** (distribution across the 512 triplets; interval: 95% basic bootstrap of the median, cells resampled, n = 20, 1,000 resamples):

| Ratio | Median [95% CI of median] | 5%–95% across triplets | SD across triplets | If exactly additive (median) |
|---|---|---|---|---|
| Pairwise AB | 0.797 [0.790, 0.806] | 0.729–0.876 | 0.044 | – |
| Pairwise AC | 0.888 [0.873, 0.889] | 0.853–0.937 | 0.024 | – |
| Pairwise BC | 0.864 [0.838, 0.856] | 0.840–0.896 | 0.018 | – |
| Pairwise mean | **0.851** [0.836, 0.850] | 0.818–0.895 | 0.022 | 0.865 |
| Three-way | **0.743** [0.730, 0.746] | 0.686–0.838 | 0.042 | 0.767 |
| Marginal C given AB | 0.440 [0.415, 0.450] | 0.328–0.533 | 0.060 | – |
| Three-way, logits | 0.758 [0.751, 0.770] | 0.670–0.900 | 0.065 | – |

- For a few ratios (pairwise BC, pairwise mean) the noise bias is about as large as the spread, so the basic interval does not contain the plain estimate.
- All 512 triplets have a three-way interval below 1. That does not mean redundancy: the formula gives 0.767 for an exactly additive joint effect, and 0.758 on the logits, where the response is additive to about 10⁻⁸. The ratios fall below 1 because effects of opposite sign cancel across targets and noise inflates |mean|.
- Observed minus additive three-way ratio: −0.023 (interval of the median [−0.019, −0.010]). This gap is noise: e_ABC is one noisy condition, while e_A + e_B + e_C sums three, so its |mean| is inflated more. The noise-adjusted ρ3 removes it (1.0037).
- For context, the deployed analysis reported a three-way ratio of 0.1896 and a pairwise mean of 0.3235 from 4 triplets.
- Breakdowns: by frequency half (octiles 0–3 vs 4–7 at each layer), the median three-way ratio is 0.715–0.779 and the median I_ABC energy share is −0.00002 to +0.00001 in all 8 groups. By the post-hoc annotation label it is 0.734–0.761 and −0.00001 to +0.00002. No group has any target passing BH across all tests.

## Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes a cross-check with separately written code (`v3_triplets.py crosscheck`, CPU float64): conditions read straight from the cell files, `scipy.stats.ttest_1samp`, BH as a step-up rule, the noise-adjusted energy from a pairwise-product (U-statistic) formula, and a new sign-flip seed.

| Number | Main analysis | Cross-check |
|---|---|---|
| Triplets with any BH target within / across | 4 / 0 of 512 | 4 / 0 of 512 |
| Smallest p | 6.708 × 10⁻⁶ | 6.708 × 10⁻⁶ |
| Median ρ3 | 1.0037108 | 1.0037108 (largest relative difference over triplets 9 × 10⁻¹⁶) |
| Median I_ABC energy share | 2.7829 × 10⁻⁶ | 2.7829 × 10⁻⁶ (largest difference 2 × 10⁻¹⁸) |
| Aggregate sign-flip p < 0.05 / BH (new seed 20261099) | 44 / 0 | 44 / 0 |
| Interval of the median energy share | basic [−0.00092, +0.00059] | new seed, percentile [0.00062, 0.00223]; as a basic interval around the unadjusted median 0.00123: [−0.00100, +0.00061] |

The percentile interval sits around the unadjusted value, as expected (Method, step 8); turned into a basic interval it matches. The count-null stage recomputed all 512 triplet and 192 pair sign-flip p-values: identical to the main analysis (largest difference 0.0). The 440 conditions recomputed from scratch (Checks) also match.

## Limits

- **20 cells**, not the 200 of the pipeline specification. Cells were traded for triplets.
- **Small edits only.** Each edit removes one feature active in 1.0–2.1% of tokens. The joint change of the L11 residual is 2.2% of its norm. A smooth network is close to linear here. Larger edits were not tested.
- **One read-out layer** (L11 codes and logits), one cell type (K562), one SAE seed.
- **No annotation was used to choose features.** The labels are post hoc and descriptive. Annotation rates in this project are near chance, so they carry little biological meaning.
- **The L5 noise floor is near-null, not null.** The exact-zero checks hold for the L0 and L9 null features only.
- **The L5 4369 excess is post hoc and borderline** (q = 0.048 over 24 features). It was not tested on new cells.
- **The pairwise mechanism is inferred, not measured.** It rests on active-position counts and on the cosine with the L9 feature's single effect. L9 activation values under the L5 edit were not recorded.
- **The 4 per-target hits were not recomputed in float64.** Their size (10⁻⁸) is below the stored float32 rounding.
- **No valid null for the superadditivity rules.**
- **The basic bootstrap intervals with 20 cells are likely somewhat too narrow** (28 + 11 triplets have a ρ3 interval that excludes 1, against about 26 if the intervals were exact).
- **The main script file was edited once during the `run` stage.** Chunks 1–5 (74 of 220 groups: cells 0–5 and groups 0–7 of cell 6) ran with `v3_triplets.py` sha256 73ada252…; chunks 6–17 and all later stages ran with 94e31c03…. The text of the first version was not kept. The checks that cover both parts all pass: the recomputation covers all 220 groups (440 conditions, largest difference 1.1 × 10⁻⁷), 2,144 of 2,144 paired conditions are bit-identical, 220 of 220 clean fingerprints are correct, and the read-out of cells 0–3 (all from the first part) matches the stored values to 1.2 × 10⁻⁸.
- Another job ran on the machine from about 13:45 to 15:55. It slowed the grid from about 14 s to 38 s per group of conditions. It did not change any result (every determinism check gave 0.0).

## Files

- Scripts (`projects/maxtoki/runs/exhaustive-mapping-217M/scripts/`): `v3_triplets.py` (stages `select`, `verify`, `run` (resumable), `recheck`, `readout`, `analyze`, `power`, `crosscheck`, `compare`), `v3_triplets_followup.py` (stages `countnull`, `pairs`, `perfeature`; no model). Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`. SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_{00,05,09,11}/sae_final.pt`.
- Outputs (`projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_triplets/`): `run_config.json` (device, versions, seeds, cell rows and barcodes, counts and token sha256, input-encoding record and check results, features and null features, SAE, hook and input sha256, wall time, free memory and script sha256 per chunk), `selection.json` (rule, bins, choice, post-hoc labels), `cells.json`, `cells_tokens.npz`, `cells_counts.npz`, `clean_frequencies.npz`, `verify.json`, `recheck/cell*.json`, `readout/cell*.json`, `readout_summary.json`, `cells/cell{ci}_g{gi}.npz` (raw per-cell results), `summary.json`, `per_triplet.json`, `per_pair.json`, `interaction_terms_real.npz`, `power.json`, `crosscheck.json`, `bootstrap_cell_weights.npy`, `signflip_signs.npy`, `analyze_cache/`.
- Follow-up outputs (`.../outputs/v3_triplets_followup/`): `countnull.json`, `pairs.json`, `perfeature.json`, `run_config.json`.
- Run time: model stages on MPS: select 0.3 min, verify 0.7 min, run 113.2 min in 17 chunks of ≤ 7.3 min (16,520 ablation runs), recheck 14.7 min, readout 2.6 min. CPU stages: analyze 8.0 min, power 0.3 min, crosscheck 0.2 min; countnull 0.3, pairs 0.6, perfeature 0.9 min. Lowest free memory seen 3.54 GB (guard at 3 GB).


# Cross-model alignment of input gene-embedding tables (MaxToki-217M vs Geneformer and scGPT)

This analysis asks whether MaxToki-217M places genes in its input embedding table in a similar way to two other single-cell foundation models. It compares the static gene-embedding tables only: no forward pass is run, so no cell input and no input encoding is involved. Three measures are used: canonical correlation (CCA), agreement of gene–gene cosine matrices (gene-pair Pearson), and top-1 retrieval of the same gene after a rotation.

## Data

**Tables compared** (no contextual layer output was compared):

| Model | Tensor | What it is |
|---|---|---|
| MaxToki-217M | `model.embed_tokens.weight` (20,275 × 1,232) | Token embedding table. Equal to `hidden_states[0]` for a gene token (Llama adds no position vector at the input). |
| Geneformer V2-316M | `bert.embeddings.word_embeddings.weight` (20,275 × 1,152) | Raw word-embedding table, before position embeddings and before the embedding LayerNorm. |
| scGPT whole-human | `encoder.embedding.weight`, then the `encoder.enc_norm` LayerNorm (60,697 × 512) | Gene-identity encoder output, before the expression-value encoder is added. |

Input files (sha256 in `run_config.json`): MaxToki `model.safetensors` 9da1fbf8…; Geneformer V2-316M `model.safetensors` 965ceccea819… (matches its Hugging Face blob name); scGPT embeddings `.pt` b41289b8…; scGPT checkpoint 6cb5d451….

**Gene sets.**
- Three topology panels: lung (Tabula Sapiens lung, 382 genes), immune (Tabula Sapiens immune, 350 genes) and external lung (Krasnow lung Smart-seq2, 380 genes). Genes were picked by TRRUST, STRING and marker priority and by expression variance (`projects/maxtoki/runs/topology-141-217M/scripts/phase0_extract.py:112-163`). Static tables do not depend on cells, so a panel only sets the gene list. The panels are tissue or dataset gene lists, not cell types.
- The panels overlap: lung–immune 75 genes, lung–external 168, immune–external 77; 36 genes are in all three; 828 in the union. So they are three overlapping gene samples, not three independent replications.
- Every panel gene is found in all three tables.
- A fourth gene set for the gene-pair Pearson only: 1,500 highly variable genes (HVGs) from 2,000 Tabula Sapiens immune cells (1,124,250 gene pairs).

**Gene matching.** MaxToki and Geneformer share a vocabulary; rows are matched by Ensembl ID. scGPT rows are found by gene symbol (exact case) through the token index stored in the scGPT vocabulary (`symbol → token id`). Note that the position of a symbol in the vocabulary dictionary equals its token id for only 1 of 60,697 genes, so the index, not the position, must be used.

## Method

All metric settings follow the deployed analysis, so the numbers measure the same thing.

1. **Pre-processing.** Each table is centred and reduced to 30 dimensions by PCA (scikit-learn, `random_state=42`), fitted separately per model.
2. **In-sample CCA.** scikit-learn `CCA(n_components=10, max_iter=200)` on the two PCA-30 tables. Score = mean of the 10 in-sample canonical correlations.
3. **Gene-pair Pearson.** Pearson correlation of the two gene–gene cosine matrices (upper triangle), computed on the full static rows.
4. **Top-1 retrieval.** An orthogonal Procrustes rotation maps MaxToki PCA-30 onto the other model's PCA-30. A gene is retrieved correctly if its own row is the nearest of all panel genes.
5. **Chance level.** The gene correspondence (rows of the other table) is permuted 1,000 times with the same n, dimensions and settings. CCA is refitted each time. For top-1, the rotation is refitted on each permutation.
6. **Gene bootstrap.** Genes are resampled with replacement (1,000 draws; percentile 2.5 / 97.5). PCA, CCA and the rotation are refitted on each resample. For the Pearson measure, pairs made of two copies of one gene are dropped.
7. **Held-out versions.**
   - 5-fold split of genes: PCA, CCA and rotation are fitted on 80% of genes and scored on the other 20%; repeated over 10 random splits (the range over splits is given). Chance from 200 permutations of the same procedure.
   - Out-of-bag (OOB) gene bootstrap: fit on each with-replacement resample and score the about 37% of genes not drawn (1,000 draws). A copy of a gene can never sit on both sides.
8. **Paired differences** (Geneformer minus scGPT) use the same genes and the same resamples: 2,000 draws for the Pearson measure, 500 draws for the held-out OOB measures.
9. **The 1,500-HVG Pearson interval.** Gene bootstrap with replacement (2,000 draws, copy pairs dropped, weighted-pair formula, seed 20261101), plus a leave-one-gene-out jackknife (normal interval). Chance from 1,000 permutations of gene labels.

## Checks

- **scGPT rows.** (a) The rows used are identical (largest difference 0.0) to `scgpt_gene_embeddings.npz`, which the extraction script saved with an index lookup. (b) The table equals `encoder.embedding.weight` in the original scGPT checkpoint (difference 0.0), and the normed table equals its LayerNorm with `encoder.enc_norm` (difference 1.4 × 10⁻⁶). (c) Known related pairs have high cosine: RPL3–RPL5 0.47, CD3D–CD3E 0.62, HBA1–HBB 0.57; random pairs 0.09 ± 0.10.
- **Reproduction.** The metric code reproduces every deployed value to within 3 × 10⁻⁸.
- **Pearson bootstrap.** The bootstrap mean equals the observed value (for example 0.3980 vs 0.3979), so this interval is not biased by gene copies. The weighted-pair formula for the 1,500-HVG bootstrap matched direct indexing to 10⁻¹⁵ on the first 5 draws.
- **No valid interval for in-sample CCA.** In-sample CCA rises when a resample holds fewer distinct genes; a resample of 382 genes holds only about 240 distinct ones. For Geneformer lung, the bootstrap mean is 0.909 and the 95% range [0.888, 0.929], which does not contain the observed 0.783. Permuting whole genes inside the same resamples gives a chance level of 0.79–0.80 for the Geneformer table and 0.74–0.77 for the scGPT table. So no percentile interval is reported for in-sample CCA. In-sample top-1 has the same problem (whole-gene chance inside a resample 26–30%), so only held-out top-1 gets an interval.
- **PCA solver.** For these table shapes `PCA(random_state=42)` uses scikit-learn's randomized solver. Refitting PCA on permuted rows instead of permuting the PCA rows changes CCA by up to 0.006–0.011. This comes from the randomized PCA, not from CCA: scikit-learn's CCA matches exact CCA within 10⁻⁶ on the same tables. With the exact PCA solver, in-sample CCA moves by up to +0.011 (Geneformer lung 0.783 → 0.794) and in-sample top-1 by up to +4.9 points (Geneformer 48.7 / 51.4 / 40.3% instead of 45.0 / 46.6 / 38.2%). No conclusion changes. The held-out values are stable.

## Results

**Main findings.**
- MaxToki's input gene table agrees with both other tables well above chance. It agrees more with Geneformer (same vocabulary) than with scGPT (different vocabulary, architecture and training data). This design cannot say which difference matters.
- In-sample CCA has a high chance level (about 0.41). The Geneformer value of 0.78 is about 0.36 above it.
- Top-1 retrieval of 38–47% is in-sample. With the query gene held out, it is 13–18% for Geneformer and 6–9% for scGPT, against a chance level of about 0.3%.
- For context, the deployed analysis built the scGPT lookup from dictionary positions, so 0 of 382 / 350 / 380 panel genes received their own scGPT row, and its scGPT values (in-sample CCA 0.401 / 0.438 / 0.406; Pearson −0.008 / −0.019 / −0.005) are at the chance level.

**In-sample CCA** (mean of 10 canonical correlations; chance from 1,000 permutations of gene correspondence):

| Comparison | Lung (382 genes) | Immune (350) | External lung (380) |
|---|---|---|---|
| Geneformer | 0.783 | 0.776 | 0.766 |
| scGPT | 0.722 | 0.736 | 0.733 |
| Chance mean (SD), Geneformer table | 0.411 (0.009) | 0.429 (0.010) | 0.412 (0.009) |
| Chance 95th percentile, Geneformer table | 0.427 | 0.445 | 0.428 |

- Geneformer is 0.35–0.37 above chance (z 34–40; p ≤ 0.001, the smallest possible with 1,000 permutations).
- scGPT is 0.31–0.32 above chance (z 32–35).

**Held-out canonical correlation** (mean of 10; 5-fold: mean and range over 10 random splits; OOB: mean and 95% percentile interval over 1,000 gene-bootstrap draws, unit = gene; chance from 200 permutations):

| Comparison | Lung | Immune | External lung |
|---|---|---|---|
| Geneformer, 5-fold (range) | 0.604 (0.592–0.619) | 0.558 (0.542–0.584) | 0.578 (0.563–0.591) |
| Geneformer, OOB [95% CI] | 0.563 [0.517, 0.608] | 0.531 [0.485, 0.575] | 0.544 [0.495, 0.591] |
| scGPT, 5-fold (range) | 0.495 (0.475–0.509) | 0.481 (0.472–0.492) | 0.493 (0.479–0.505) |
| scGPT, OOB [95% CI] | 0.449 [0.396, 0.499] | 0.426 [0.371, 0.473] | 0.445 [0.395, 0.491] |
| Chance, Geneformer table: mean / 95th percentile | 0.00 / 0.027 | 0.00 / 0.032 | 0.00 / 0.031 |
| Chance, scGPT table: mean / 95th percentile | 0.00 / 0.028 | 0.00 / 0.032 | 0.00 / 0.037 |

- OOB values sit a little below the 5-fold values because OOB trains on about 63% distinct genes, not 80%.
- The first held-out canonical pair is strong for both models: 0.79–0.92 (Geneformer) and 0.77–0.90 (scGPT), 5-fold.

**Gene-pair Pearson** (agreement of gene–gene cosine matrices; 95% percentile interval from a gene bootstrap with replacement, unit = gene, copy pairs dropped: 1,000 draws for the panels, 2,000 for the 1,500 HVGs with Geneformer and 1,000 with scGPT; chance from 1,000 permutations for the panels and Geneformer HVGs, 200 for scGPT HVGs):

| Comparison | Lung | Immune | External lung | 1,500 HVGs |
|---|---|---|---|---|
| Geneformer [95% CI] | 0.398 [0.382, 0.414] | 0.400 [0.385, 0.415] | 0.387 [0.374, 0.401] | **0.382 [0.377, 0.387]** |
| scGPT [95% CI] | 0.265 [0.247, 0.285] | 0.263 [0.246, 0.282] | 0.283 [0.267, 0.301] | 0.239 [0.233, 0.245] |
| Chance mean (SD) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.001) |

**The 1,500-HVG Geneformer interval, two ways** (n = 1,500 genes, unit = gene):

| Method | 95% interval | SD / SE |
|---|---|---|
| Gene bootstrap with replacement, 2,000 draws, copy pairs dropped, percentile | [0.3772, 0.3867] | 0.0024 |
| Leave-one-gene-out jackknife, normal interval | [0.3775, 0.3864] | 0.0023 |

Chance: −0.00005 (SD 0.0010, 1,000 permutations).

**Top-1 retrieval** (in-sample chance: rotation refitted on each of 1,000 permutations; held-out 5-fold: mean and range over 10 splits; OOB: mean and 95% percentile interval over 1,000 gene-bootstrap draws, unit = gene; held-out chance from 200 permutations):

| Comparison | Lung | Immune | External lung |
|---|---|---|---|
| Geneformer, in-sample | 45.0% | 46.6% | 38.2% |
| scGPT, in-sample | 30.9% | 34.9% | 31.3% |
| In-sample chance, Geneformer table: mean / 95th percentile | 6.6% / 8.4% | 8.0% / 10.0% | 7.2% / 9.2% |
| In-sample chance, scGPT table: mean | 5.4% | 6.7% | 5.5% |
| Geneformer, held-out 5-fold (range) | 17.9% (16.0–21.2) | 15.8% (13.1–18.9) | 14.6% (13.4–16.1) |
| Geneformer, held-out OOB [95% CI] | 14.7% [9.1, 20.7] | 16.2% [10.2, 22.7] | 12.9% [7.7, 18.6] |
| scGPT, held-out 5-fold (range) | 5.9% (3.9–8.1) | 9.1% (6.3–10.6) | 6.8% (5.5–9.2) |
| scGPT, held-out OOB [95% CI] | 5.7% [2.2, 9.8] | 7.4% [3.3, 12.0] | 6.2% [2.3, 10.1] |
| Held-out chance: mean / 95th percentile | 0.2% / 0.5% | 0.3% / 0.9% | 0.2% / 0.5% |

In-sample top-1 fits the rotation on the same genes it then scores. That is why its chance level is already 6–8%.

**Geneformer minus scGPT, paired** (same genes and same resamples; 95% percentile interval; Pearson: 2,000 gene-bootstrap draws; held-out measures: 500 OOB draws):

| Gene set | Gene-pair Pearson [95% CI] | Held-out CCA (OOB) [95% CI] | Held-out top-1 (OOB) [95% CI] |
|---|---|---|---|
| Lung | 0.132 [0.119, 0.147] | 0.112 [0.052, 0.172] | 9.3 points [2.8, 15.8] |
| Immune | 0.137 [0.121, 0.153] | 0.106 [0.051, 0.154] | 8.9 points [1.5, 16.4] |
| External lung | 0.104 [0.088, 0.119] | 0.100 [0.044, 0.159] | 6.7 points [0.0, 13.5] |
| 1,500 HVGs | 0.143 [0.138, 0.148] | not run | not run |

Geneformer aligns with MaxToki better than scGPT does on every measure. The top-1 gap is the least certain: its interval touches 0 on external lung.

## Verification

A second agent re-checked this analysis with separate code (`scripts/v2_crossmodel_verify.py`; outputs `outputs/v2_crossmodel/verify/*.json`, inputs and sha256 in `verify/run_config.json`). The script does not import the main helper module. It rebuilds every table from the raw files and uses exact PCA (numpy), exact CCA (QR and SVD), scipy Procrustes and direct-index bootstraps. It used fewer draws than the main analysis because the machine was heavily loaded. Results:

1. **scGPT rows.** The scGPT `.pt` vocabulary equals the checkpoint's own `vocab.json` for all 60,697 genes. Dictionary position equals the token id for only 1 gene. This uses a source the main analysis did not use.
2. **Tables.** The rebuilt tables are identical to `embeddings_subset.npz` (largest difference 0.0, all panels, all tables).
3. **In-sample CCA chance, second method** (exact CCA, exact PCA, 300 permutations): 0.410 / 0.430 / 0.412 for Geneformer (main: 0.411 / 0.429 / 0.412). Observed exact-CCA values: Geneformer 0.794 / 0.778 / 0.769; scGPT 0.730 / 0.741 / 0.730.
4. **In-sample top-1 chance, rotation refitted:** 6.6% / 8.1% / 7.2% (main: 6.6 / 8.0 / 7.2%).
5. **Held-out values, second method** (exact PCA and CCA, own fold splits, 3 × 5-fold, 20 permutations, 100 OOB draws):
   - held-out CCA, Geneformer 0.607 / 0.573 / 0.579 (main 0.604 / 0.558 / 0.578); scGPT 0.499 / 0.496 / 0.504 (main 0.495 / 0.481 / 0.493);
   - OOB interval, Geneformer lung 0.566 [0.515, 0.608] (main 0.563 [0.517, 0.608]); unit = gene, with replacement, percentile;
   - held-out top-1, Geneformer 17.4% / 16.6% / 14.2% (main 17.9 / 15.8 / 14.6%); scGPT 3.7% / 8.9% / 6.5% (main 5.9 / 9.1 / 6.8%);
   - held-out chance: CCA about 0.00 (95th percentile 0.02–0.04); top-1 0.2–0.4%.
6. **The 1,500-HVG interval, second method.** Gene bootstrap with replacement by direct indexing (1,000 draws, copy pairs dropped, percentile): [0.3773, 0.3865], SD 0.0024. Jackknife: [0.3775, 0.3864]. Main: [0.3772, 0.3867].
7. **Panel Pearson intervals, second method** (jackknife, unit = gene): within 0.003 of the main bootstrap at each end. Example: Geneformer lung 0.398, jackknife [0.384, 0.412], main bootstrap [0.382, 0.414].
8. **scGPT on the 1,500 HVGs** (rows from `vocab.json`): 0.2390, the same as the main analysis.
9. **No valid with-replacement bootstrap for in-sample CCA, confirmed.** With exact CCA, Geneformer lung resamples give 0.910 [0.891, 0.928] against an observed 0.794; whole-gene chance inside the resamples is 0.805. Even pure Gaussian noise of the same size goes from 0.412 to 0.556 when rows are resampled with replacement.
10. **Raw scGPT table (before LayerNorm).** Pearson 0.266 / 0.256 / 0.284 vs 0.265 / 0.263 / 0.283 for the normed table; exact CCA 0.731 / 0.740 / 0.730 vs 0.730 / 0.741 / 0.730. The choice of scGPT table does not matter.
11. **Figure data.** `fig_crossmodel.csv` matches the job files (135 fields in 64 rows; 0 mismatches).
12. **PCA solver.** The verifier found that the small run-to-run differences in CCA come from the randomized PCA solver, not from CCA, and measured the solver effect on in-sample CCA and top-1 (reported in Checks).

The verifier agreed with every number within Monte Carlo noise.

## Limits

- **No contextual (layer) comparison.** Only static input tables were compared. This needs Geneformer and scGPT forward passes, which were not run. Any claim about the models' internal gene geometry stays untested.
- **No valid interval for in-sample CCA or in-sample top-1.** Held-out versions with intervals are reported instead.
- **Genes are not independent units.** They share pathways and regulators, so every gene-bootstrap interval here is probably somewhat too narrow (not tested).
- **scGPT rows are matched by gene symbol**, Geneformer rows by Ensembl ID. All symbols were found, but a symbol that names a different gene in scGPT's older vocabulary cannot be ruled out. This could only lower the scGPT numbers a little.
- One PCA size (30) and one number of CCA components (10). Other settings were not tried.
- Geneformer and scGPT were not compared with each other directly.
- The three panels overlap and come from tissue datasets, so they are not independent replications. The 1,500-HVG set and the panels are different gene sets; numbers from them should not be mixed in one figure without saying so.
- The in-sample numbers carry up to about 0.011 (CCA) and 5 points (top-1) of PCA-solver noise.

## Files

- Scripts: `projects/maxtoki/runs/topology-141-217M/scripts/v2_crossmodel_common.py` (metric code), `v2_crossmodel_prepare.py` (inputs, lookup checks, reproduction), `v2_crossmodel_stats.py` (chance, bootstrap, held-out, OOB; resumable), `v2_crossmodel_paired.py` (Geneformer minus scGPT), `v2_crossmodel_summarize.py` (summary and figure CSV), `v2_crossmodel_verify.py` (second agent); `projects/maxtoki/runs/spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py` (1,500-HVG interval).
- Outputs (`projects/maxtoki/runs/topology-141-217M/outputs/v2_crossmodel/`): `fig_crossmodel.csv` (figure source data, 64 rows; each row has value, interval, interval method, chance mean and 95th percentile; it also holds rows for the dictionary-position lookup), `summary.json`, `paired_differences.json`, `prepare_checks.json`, `run_config.json` (input paths, sha256, seeds, output hashes), `jobs/` (per-job JSON and raw draws), `embeddings_subset.npz`, `verify/` (`lookup.json`, `insample.json`, `heldout.json`, `pearson.json`, `hvgscgpt.json`, `rawscgpt.json`, `bootcheck.json`, `permorigin.json`, `pcasolver.json`, `csvcheck.json`, `run_config.json`).
- 1,500-HVG outputs: `projects/maxtoki/runs/spectral-geometry-217M/outputs/v2_crossmodel/pearson_1500hvg_ci.json`, `run_config.json`.
- CPU only; no model forward pass.


# Spectral geometry: effective rank and cross-sample CKA by layer (MaxToki-217M, Tabula Sapiens immune)

This analysis asks two questions about the per-gene vectors inside MaxToki-217M. First: how many directions do these vectors use at each layer (effective rank), and does the fall with depth come from training or from the architecture? A model with the same architecture and random weights answers the second part. Second: how much does the gene geometry of a layer change when each gene's average is taken over a different set of cells (cross-sample CKA)?

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, eager attention, Apple MPS. torch 2.11.0, transformers 5.5.4, numpy 1.26.4. The model has 11 decoder blocks and a hidden size of 1,232.
- **Random-initialised models.** Three models with the same configuration (`LlamaConfig` read from the checkpoint). For seed s = 0, 1, 2: `torch.manual_seed(s)`, then `LlamaForCausalLM(config)`. This is the transformers default initialisation: weights normal(0, 0.02), norm weights 1. The state-dict sha256 of each model is recorded and checked again in every job chunk.
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`, 592,317 cells). Counts come from `raw/X` (integer counts, before ambient-RNA removal). Three samples of 2,000 cells:
  - A: `numpy.random.default_rng(42).choice(592317, 2000, replace=False)`. These are the cells of the deployed analysis (cell types match in 2,000 of 2,000).
  - B and C: draws with `default_rng(43)` and `default_rng(44)`. Any cell that also occurs in an earlier sample was replaced by a new random cell that no other sample uses (`default_rng(20261002)` for B, `default_rng(20261003)` for C). 6 cells were replaced in B and 16 in C.
  - No cell is shared between A, B and C.
  - Smart-seq2 cells: 73 in A, 80 in B, 80 in C. The rest are droplet (UMI) cells. All cells were kept.
- **Genes.** The 1,500 genes of the deployed analysis (`projects/maxtoki/runs/spectral-geometry-217M/outputs/phase0/gene_features.csv`). Applying its rule to the same cells gives the same 1,500 genes. The rule: genes in the MaxToki vocabulary with a TRRUST, STRING or lineage-marker annotation, non-zero variance and expression in more than 0.5% of sample-A cells (3,073 genes), ranked by variance across sample-A cells; the top 1,500 are kept. The variance was computed on log1p(CP10k) of the atlas's stored `X`, which is already log-transformed.
- **Second gene panel (sensitivity only).** The same rule applied to log1p(CP10k) of the raw counts. It shares 1,426 of its 1,500 genes with the main panel.
- All 1,500 genes are seen in every sample. A gene's vector averages only the cells in which it is among the 2,046 kept genes: a median of 493.5 cells in A, 482 in B and 485 in C (range 38–1,854). In a 1,000-cell half of a sample the median is 240–247.

## Method

1. **Input encoding** (section 00). For each cell: counts / cell total × 10,000 / Geneformer gc104M gene median; rank from high to low; keep the first 2,046 genes; add `<bos>` and `<eos>` (at most 2,048 tokens). Every cell is checked before its forward pass (Checks).
2. **Forward pass.** One cell per sequence, batch 1. `output_hidden_states=True` gives 12 states. L0 is the token embedding. L1 to L10 are the outputs of blocks 0 to 9. L11 is the final RMSNorm applied to the output of block 10.
3. **Per-gene vectors.** At each layer, each panel gene's vector is the mean of its hidden state over all cells in which it is among the kept genes. Sums are stored per block of 200 cells, so the mean over any set of blocks can be formed exactly.
4. **Jobs.** Trained model on A, B and C. Random seeds 0, 1 and 2 on A. Random seed 0 also on B and C (for the cross-sample CKA control). This is 8 × 2,000 = 16,000 forward passes.
5. **Effective rank (ER).** Take the 1,500 × 1,232 matrix of per-gene vectors at one layer and centre its columns. With singular values σ_i, let p_i = σ_i² / Σ σ_j². ER = exp(−Σ p_i log p_i). ER is a number of dimensions; here it can be at most 1,232. The metric code (`_effective_rank`, `_svd_and_metrics`) is copied from `projects/maxtoki/runs/spectral-geometry-217M/scripts/phase1_svd.py`.
6. **Other measures from the same code.**
   - Participation ratio: (Σ σ_i²)² / Σ σ_i⁴.
   - Share of variance on the top direction: σ_1² / Σ σ_i².
   - TwoNN local intrinsic dimension (Facco et al., 2017): 500 genes drawn with a fixed seed; for each, μ = (distance to its 2nd-nearest gene) / (distance to its nearest gene); the slope of −log(1 − F(μ)) against log μ over the lowest 90% of μ. TwoNN describes the dimension of the structure near each gene, not the global spread.
7. **Feature-shuffle null.** Shuffle each of the 1,232 columns across genes, independently (one numpy stream), then compute ER. Computed at every layer.
8. **Compression summaries.** ER(L1) / ER(L11); share of ER lost from L1 to L11 = 1 − ER(L11) / ER(L1); the same from L0; Spearman ρ of ER with layer index.
9. **Cross-sample CKA.** Linear CKA (Kornblith et al., 2019) between the per-gene matrices X and Y of two samples at the same layer, rows matched by gene, columns centred: CKA = ‖XᵀY‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F). 1 means the same geometry up to rotation and overall scale. The value reported is the mean over the three pairs (A–B, A–C, B–C), for L1 to L11. L0 is left out: a gene's L0 vector is its embedding row, which is the same in every sample, so CKA is 1 by construction. Done for the trained model and random seed 0. Code: `linear_cka`, copied from `projects/maxtoki/runs/spectral-geometry-217M/scripts/phase8_stability.py`.
10. **Split-half CKA.** Inside each sample: per-gene means over blocks 0–4 against blocks 5–9 (1,000 against 1,000 cells). Done for all 8 jobs. It gives a CKA control for random seeds 1 and 2 without more forward passes, and it shows how 1 − CKA changes when each gene's average uses about half as many cells (in sample A, a median of 246–247 instead of 493.5).
11. **Intervals.** Delete-one-group jackknife. SE = sqrt((g − 1)/g × Σ (θ_i − θ̄)²); 95% CI = estimate ± t(0.975, g − 1) × SE.
    - Over cells: each sample is split into 10 random blocks of 200 cells (`default_rng(20261005 + k)`, k = sample index). For the three-sample CKA, block j is deleted from all three samples at once. g = 10.
    - Over genes (CKA only): 30 random groups of 50 genes (`default_rng(20261007)`). g = 30.
    - The ER intervals use the eigenvalue route (Verification, first row). The intervals are centred on the estimates and are not bias-corrected.

## Checks

| Check | Result |
|---|---|
| Encoding check before every forward pass (`tokenize_counts(check=True)` and `assert_encoding_batch` on every 50-cell batch; the tokens must also equal the tokens saved when the samples were prepared) | 16,000 of 16,000 cells in 320 batches pass |
| Counts are whole numbers (every input row of `raw/X`) | 6,000 of 6,000 rows |
| Metric code identical to the code it was copied from (`phase1_svd.py`, `phase8_stability.py`; abstract syntax tree comparison) | 5 of 5 functions identical |
| Samples disjoint; A equals the stated draw | pairwise overlap 0, 0, 0; A identical |
| L0 per-gene mean equals the model's embedding row (L0 does not depend on the input) | largest relative difference 1.6 × 10⁻⁶ |
| Model health: next-gene loss (nats per gene token) | trained, UMI cells: 3.28 in A, B and C; trained, Smart-seq2 cells: 7.5–7.7; random models: 10.16. A uniform guess over the 20,275-token vocabulary gives 9.92. |
| Memory | free memory never below 3.8 GB (guard at 3 GB) |

**Why jackknife intervals, not bootstrap.** Percentile bootstraps were also computed (`analysis/cka.json`). Both are shifted for this design:
- Drawing genes with replacement puts identical rows into both matrices, which pushes CKA up. For the trained model at L11 the estimate 0.9962 sits at the lower edge of the gene-bootstrap interval [0.9962, 0.9967].
- Drawing 200-cell blocks with replacement leaves about 63% distinct cells. The averages get noisier and CKA goes down: trained L11 [0.9903, 0.9943], random [0.918, 0.948], both below their estimates (0.9962 and 0.9647).

The jackknife never copies a gene or a cell. Its width agrees with two other estimates (Verification). For the same reason no cell bootstrap was used for ER.

## Results

**Main findings.**
- The trained model's ER is 561 at L0 and 328 at L1. It then falls, not quite steadily (it rises at L2, L5 and L8), to 227 at L11 [95% CI 224.8–229.5]. The two other samples give 225.6 and 226.3.
- Three random-weight models (seeds 0–2, on sample A) fall from 816–817 (L0) to 642–645 (L1) and 390–421 (L11).
- From L1 to L11 the random models lose 34.6–39.6% of their ER; the trained model loses 30.8–31.3%. Every random run loses more than every trained sample. The 95% cell-block jackknife intervals do not overlap (trained at most 31.8%, random at least 32.5%). These intervals cover cell sampling only, not variation between seeds. So the fall of ER from L1 to L11 is a property of the architecture with these inputs and the default initialisation. Training adds extra drops at the first block and at the final norm (see "Where the trained and random models differ").
- Training lowers ER at every layer, from the embedding table on (561 against 816–817). Its first block removes 41.5% of ER (random: 21.0–21.3%).
- Local intrinsic dimension (TwoNN) falls from 102 to 5.3 in the trained model. In the random models it falls from 204–207 to 107–120.
- The feature-shuffle null stays far above the real ER in every model and layer (at L11: 639–646 against 226–227 in the trained model; 809–812 against 390–421 in the random models). It cannot tell learned from architectural compression.
- Cross-sample CKA, trained: 0.9997 (L1) to 0.9962 (L11). Random seed 0: 0.9938 to 0.9647. With 1,000-cell halves instead of 2,000-cell samples (a median of about 246 instead of 493.5 cells behind each gene's average in sample A), 1 − CKA at L11 nearly doubles in the trained model (0.0038 to 0.0073) and rises ×1.7 in random seed 0 (0.035 to 0.059). So for the trained model most of the gap from 1 is sampling noise in the per-gene averages.

**ER per layer** (`analysis/per_layer_metrics.csv`, `analysis/per_layer_metrics_optional.csv`; 1,500 genes, all cells; rounded to whole dimensions):

| Model / sample | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trained, A | 561 | 328 | 355 | 343 | 302 | 304 | 275 | 261 | 273 | 267 | 266 | 227 |
| Trained, B | 561 | 328 | 355 | 343 | 302 | 303 | 275 | 261 | 273 | 267 | 266 | 226 |
| Trained, C | 561 | 328 | 355 | 343 | 302 | 304 | 276 | 262 | 273 | 267 | 266 | 226 |
| Random seed 0, A | 816 | 642 | 649 | 624 | 589 | 556 | 518 | 482 | 452 | 419 | 394 | 398 |
| Random seed 1, A | 817 | 645 | 650 | 625 | 583 | 546 | 506 | 464 | 431 | 403 | 383 | 390 |
| Random seed 2, A | 817 | 645 | 652 | 632 | 605 | 570 | 538 | 504 | 470 | 443 | 413 | 421 |
| Random seed 0, B | 816 | 643 | 650 | 626 | 593 | 562 | 526 | 490 | 461 | 428 | 404 | 411 |
| Random seed 0, C | 816 | 642 | 650 | 625 | 592 | 560 | 525 | 488 | 459 | 425 | 400 | 406 |

Jackknife SE over cells (dimensions): trained 0.1 at L1, 0.1–0.6 at L2–L10, 0.8–1.0 at L11; random 0.2–0.3 at L1, rising to 4.3–6.1 at L11. L0 does not depend on the cells (SE 0). The spread across the three disjoint samples is of the same size: trained L11 225.6–227.2 (SD 0.8); random seed 0 L11 397.6–410.6 (SD 6.6).

**Compression summary** (`analysis/intervals.json`, `analysis/summary.json`; 95% jackknife CI over 10 blocks of 200 cells):

| Model / sample | ER L0 | ER L1 | ER L11 [95% CI] | L1/L11 [95% CI] | Share lost L1 → L11 [95% CI] | L0/L11 [95% CI] | Spearman ρ, L0–L11 | Feature-shuffle ER at L11 |
|---|---|---|---|---|---|---|---|---|
| Trained, A | 560.8 | 328.1 | 227.2 [224.8, 229.5] | 1.44 [1.43, 1.46] | 30.8% [30.0, 31.5] | 2.47 [2.44, 2.49] | −0.930 | 645.6 |
| Trained, B | 560.8 | 328.2 | 225.6 [223.9, 227.3] | 1.45 [1.44, 1.47] | 31.3% [30.7, 31.8] | 2.49 [2.47, 2.50] | −0.930 | 639.0 |
| Trained, C | 560.8 | 328.1 | 226.3 [224.3, 228.2] | 1.45 [1.44, 1.46] | 31.0% [30.4, 31.6] | 2.48 [2.46, 2.50] | −0.930 | 641.2 |
| Random seed 0, A | 815.9 | 642.1 | 397.6 [386.7, 408.6] | 1.61 [1.57, 1.66] | 38.1% [36.4, 39.8] | 2.05 [2.00, 2.11] | −0.986 | 810.3 |
| Random seed 1, A | 816.8 | 644.9 | 389.8 [376.0, 403.5] | 1.65 [1.60, 1.71] | 39.6% [37.4, 41.7] | 2.10 [2.02, 2.17] | −0.986 | 809.3 |
| Random seed 2, A | 816.8 | 644.6 | 421.4 [408.2, 434.6] | 1.53 [1.48, 1.58] | 34.6% [32.5, 36.7] | 1.94 [1.88, 2.00] | −0.986 | 812.1 |
| Random seed 0, B | 815.9 | 642.9 | 410.6 [400.9, 420.3] | 1.57 [1.53, 1.60] | 36.1% [34.7, 37.6] | 1.99 [1.94, 2.03] | −0.986 | 810.9 |
| Random seed 0, C | 815.9 | 642.4 | 405.7 [394.3, 417.1] | 1.58 [1.54, 1.63] | 36.8% [35.1, 38.6] | 2.01 [1.95, 2.07] | −0.986 | 810.1 |

Spearman ρ is the rank correlation of ER with the layer index (12 layers). p = 1 × 10⁻⁵ for the trained model and 4 × 10⁻⁹ for the random models. From L1 to L11 only: ρ = −0.909 (trained) and −0.982 (random).

**Where the trained and random models differ** (ER, sample A for the random seeds; all from `analysis/per_layer_metrics.csv`):

| Step | Trained (A / B / C) | Random (seeds 0–2 on A; seed 0 on B, C) |
|---|---|---|
| L0 → L1 (first block) | −41.5% in all three | −21.0% to −21.3% |
| L1 → L11 | −30.8% to −31.3% | −34.6% to −39.6% |
| L10 → L11 (the step that applies the final norm) | −14.6% to −15.1% | +1.0% to +2.1% |
| L0 → L11 | −59.5% to −59.8% | −48.4% to −52.3% |

- Measured from L0, the trained model loses more (59.5–59.8% against 48.4–52.3%). The whole difference sits in the first block.
- After L1 the random models lose more.
- The learned norm weights are one possible reason for the trained model's last-step drop. This was not tested.

**Other measures** (same matrices and code; `analysis/per_layer_metrics.csv`, `analysis/per_layer_metrics_optional.csv`):

| Measure | Trained (A / B / C) | Random (seeds 0–2 on A; seed 0 on B, C) |
|---|---|---|
| TwoNN local dimension, L0 → L11 | 102 → 5.3 / 5.3 / 5.3 | 204–207 → 107–120 |
| TwoNN, L1 to L11 range | 82 at L1 falling steadily to 5.3 (sample A) | 48–121, mostly about 100 |
| Participation ratio, L1 → L11 | 243 → 119 / 116 / 117 | 487–489 → 70–93 |
| Share of variance on the top direction, L1 → L11 | 1.8% → 4.0% | 0.5% → 7.9–10.1% |

- Random-weight models put more and more variance on a few directions with depth. Their participation ratio falls more than the trained model's, and one direction grows to about 10% of the variance.
- But their genes stay locally high-dimensional (TwoNN about 100).
- The trained model is the opposite: its global spread falls less, but its genes sit on a locally low-dimensional structure (TwoNN about 5).
- These are observations. Why they differ was not tested.

**Feature-shuffle null** (ER after shuffling each column across genes):

| Model | L1 to L10 | L11 |
|---|---|---|
| Trained | 780–791 (sample A) | 639.0–645.6 (A, B, C) |
| Random | 809–817 (seeds 0–2, sample A) | 809.3–812.1 (all five random jobs) |

The null is far above the real ER in both the trained and the random models, at every layer. It is not near the maximum of 1,232 either. So it cannot separate compression learned in training from compression that the architecture produces with random weights. That needs the random-weight models.

**Sensitivity** (ER at L1 / L6 / L11; `analysis/report_tables.md`, Table 3):

| Setting | Trained, A | Random seed 0, A |
|---|---|---|
| Main (1,500 genes, all cells) | 328.1 / 275.2 / 227.2 | 642.1 / 518.4 / 397.6 |
| UMI cells only (73 Smart-seq2 cells dropped) | 328.0 / 274.2 / 227.4 | 641.6 / 514.9 / 392.5 |
| Second gene panel (picked on log1p(CP10k) of raw counts) | 332.4 / 278.3 / 228.5 | 641.6 / 504.8 / 379.2 |

Neither setting changes the picture. Over the three settings, L1/L11 is 1.44–1.45 for the trained model (samples A, B, C) and 1.53–1.73 for random seeds 0–2.

**Cross-sample CKA, three disjoint 2,000-cell samples** (mean of the three pairs; `analysis/intervals.json`, `cka3_jackknife`; 95% jackknife CI over 30 groups of 50 genes and over 10 cell blocks):

| Layer | Trained | [genes] | [cells] | Random seed 0 | [genes] | [cells] |
|---|---|---|---|---|---|---|
| L1 | 0.9997 | [0.9997, 0.9998] | [0.9997, 0.9998] | 0.9938 | [0.9936, 0.9941] | [0.9936, 0.9941] |
| L2 | 0.9997 | [0.9996, 0.9997] | [0.9996, 0.9997] | 0.9880 | [0.9871, 0.9889] | [0.9871, 0.9889] |
| L3 | 0.9993 | [0.9993, 0.9994] | [0.9992, 0.9994] | 0.9804 | [0.9785, 0.9823] | [0.9780, 0.9828] |
| L4 | 0.9991 | [0.9990, 0.9991] | [0.9989, 0.9992] | 0.9741 | [0.9714, 0.9769] | [0.9702, 0.9781] |
| L5 | 0.9988 | [0.9987, 0.9989] | [0.9987, 0.9990] | 0.9696 | [0.9662, 0.9731] | [0.9645, 0.9748] |
| L6 | 0.9981 | [0.9980, 0.9982] | [0.9979, 0.9982] | 0.9673 | [0.9631, 0.9715] | [0.9615, 0.9732] |
| L7 | 0.9972 | [0.9970, 0.9975] | [0.9969, 0.9976] | 0.9666 | [0.9619, 0.9714] | [0.9602, 0.9730] |
| L8 | 0.9974 | [0.9972, 0.9976] | [0.9972, 0.9977] | 0.9654 | [0.9603, 0.9706] | [0.9585, 0.9723] |
| L9 | 0.9978 | [0.9976, 0.9980] | [0.9976, 0.9980] | 0.9655 | [0.9601, 0.9709] | [0.9589, 0.9722] |
| L10 | 0.9976 | [0.9974, 0.9978] | [0.9973, 0.9978] | 0.9655 | [0.9600, 0.9710] | [0.9589, 0.9722] |
| L11 | **0.9962** | [0.9959, 0.9964] | [0.9958, 0.9966] | **0.9647** | [0.9598, 0.9697] | [0.9588, 0.9706] |

- From L1 to L11, CKA drops by 0.0036 in the trained model and by 0.029 in the random model.
- The intervals describe CKA of 2,000-cell samples, not a noise-free CKA. At L11 the cell-block jackknife bias is −0.0037 (trained) and −0.031 (random seed 0), 22 and 12 times the SE. A negative bias means that averages over more cells would give a CKA closer to 1.
- The three pairs agree: trained L11 0.99605, 0.99612, 0.99642; random 0.96263, 0.96530, 0.96631.
- The trained model's per-gene averages are clearly more stable across cell samples than the random model's, at every layer from L1.

**Split-half CKA inside one sample** (1,000 against 1,000 cells; `analysis/intervals.json`, `cka_half_jackknife`; 95% jackknife CI over genes and over cell blocks):

| Job | L1 | L6 | L11 [genes] [cells] |
|---|---|---|---|
| Trained, A | 0.9995 | 0.9963 | 0.9927 [0.9922, 0.9932] [0.9919, 0.9936] |
| Trained, B | 0.9995 | 0.9963 | 0.9930 [0.9925, 0.9935] [0.9917, 0.9943] |
| Trained, C | 0.9995 | 0.9961 | 0.9924 [0.9919, 0.9929] [0.9906, 0.9942] |
| Random seed 0, A | 0.9882 | 0.9462 | 0.9414 [0.9345, 0.9483] [0.9315, 0.9513] |
| Random seed 1, A | 0.9880 | 0.9475 | 0.9435 [0.9366, 0.9503] [0.9324, 0.9545] |
| Random seed 2, A | 0.9881 | 0.9386 | 0.9300 [0.9224, 0.9376] [0.9152, 0.9449] |
| Random seed 0, B | 0.9881 | 0.9467 | 0.9385 [0.9300, 0.9470] [0.9315, 0.9455] |
| Random seed 0, C | 0.9879 | 0.9425 | 0.9341 [0.9258, 0.9425] [0.9174, 0.9509] |

- All three random seeds are far less stable than the trained model at every layer. Seed 0 is typical of the three.

**What the CKA gap from 1 measures.** If the gap came only from sampling noise in the per-gene averages, halving the number of cells behind each average would double 1 − CKA.

| Model | 1 − CKA at L11, 2,000-cell samples (mean of the three pairs; median 482–494 cells per gene) | 1 − CKA at L11, 1,000-cell halves of sample A (median 246–247 cells per gene) | Ratio |
|---|---|---|---|
| Trained | 0.0038 | 0.0073 | 1.9 |
| Random seed 0 | 0.035 | 0.059 | 1.7 |

The halves of samples B and C give 0.0070 and 0.0076 (trained) and 0.061 and 0.066 (random seed 0). So for the trained model almost all of the gap from 1 is noise from which cells were averaged. For random seed 0 the ratio is lower (1.7), so noise is a smaller share of its gap. These CKA values measure how much a gene's average vector moves when it is computed from a different random set of cells of the same atlas. They are not a comparison between models or between tissues.

## Verification

No independent analysis of this run is recorded. The main numbers were computed a second way (`outputs/v3_spectral/verify/verify.json`; all pass):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| ER, all 12 layers (trained A, random seed 0 A, and the stored deployed vectors) | singular values (`_svd_and_metrics`, the copied metric code) | eigenvalues of the float64 covariance | equal to 3 decimals |
| Three-sample CKA, L1, L6, L11 (trained and random seed 0) | `linear_cka` (feature space) | HSIC on centred gram matrices, float64 | differences ≤ 4 × 10⁻⁶ |
| Tokens of 120 random cells | `inputs_v3` | own tokeniser, float64 | 115 identical; 5 differ only inside exact ties |
| Stored sums, all 8 jobs | sum of per-cell sums | sum of per-gene block sums | largest relative difference 3.9 × 10⁻⁹ |
| Hidden states, two cells per model | stored per-cell sums | recomputed from scratch on MPS and on CPU; `output_hidden_states` against hook captures (`hooks_v2.ResidualEditor`) at all 12 sites | captures identical (difference 0.0); stored sums reproduced within 7 × 10⁻⁷ relative; CPU against MPS within 2.6 × 10⁻⁶ relative |
| Width of the CKA interval, trained L11 | gene-jackknife SE 0.00012; cell-jackknife SE 0.00017 | gene-bootstrap SD 0.00013; SD of the three pairwise CKAs 0.0002 | same size |

Applied to the stored per-gene vectors of the deployed analysis, whose inputs were ranked from log values, the same metric code returns that analysis's published values exactly.

## Limits

- **The deployed input order was not run through the model here.** The gap between the deployed last-layer ER and the value here is put down to the input order because cells, genes and metric code are the same. It was not shown by a direct run. The deployed analysis did not record its torch and transformers versions.
- **One random-initialisation scheme.** Only the transformers default (normal(0, 0.02)) was used. Other schemes, or a model with shuffled trained weights, were not tried. The architecture conclusion holds for this scheme and these inputs.
- **Random-weight CKA across samples uses one seed** (seed 0 on A, B and C). Seeds 1 and 2 were run on sample A only; their split-half CKA agrees with seed 0.
- **One atlas, one gene panel.** Tabula Sapiens immune cells only. The panel is weighted towards annotated genes. A second panel picked on raw counts gives the same ER, but other tissues and other gene sets were not tested.
- **Per-gene averages.** ER and CKA describe genes' average vectors over cells, not single-cell states. CKA here mostly measures averaging noise. No other model's cross-sample CKA was computed, so these values are not compared with any other model or with CKA values from studies that compare fine-tuned models.
- **TwoNN has no interval** beyond the spread across samples and seeds (trained 5.3 in all three samples; random 107–120). It is a noisy estimator in high dimension. Read it as large against small, not as exact values. Why the trained and random models differ in TwoNN and participation ratio was not tested.
- **Intervals are not bias-corrected.** The jackknife bias estimate for ER at L11 is +0.78 dimensions for the trained model and +11.9 to +13.1 for the random models. A positive value means that averages over more cells would give a somewhat lower ER. Correcting for it would lower the random models' L11 ER and make their fall from L1 larger, so the comparison would not change.
- **Smart-seq2 cells** (73, 80 and 80) were kept. The model predicts them much worse than UMI cells (loss 7.5–7.7 against 3.28 nats; a uniform guess gives 9.92). Dropping them changes the trained ER at L11 by 0.3 in sample A (0.4 in B, 0.5 in C).
- **Counts** are `raw/X`, before ambient-RNA removal. The `decontXcounts` layer was not tried.
- **Other analyses built on these vectors were not run with these inputs:** gene-set enrichment of singular vectors (STRING, TF–target), cell-type-specific compression, the germinal-centre–plasma-cell angle, BATF/BCL6, and principal angles between samples. The per-gene vectors needed for them are saved for samples A, B and C.
- **One context per sequence.** Each sequence holds one cell (at most 2,046 genes). MaxToki's multi-cell trajectory input was not used. MaxToki-1B was not tested.

## Files

- Scripts (`projects/maxtoki/runs/spectral-geometry-217M/scripts/`): `v3_spectral.py` (stages `prepare`, `extract` (resumable), `analyze --part metrics|cka|cka_half|summary`, `verify --part code|samples|tokens|sums|l0|metrics2|forward`), `v3_spectral_intervals.py` (parts `er`, `cka`, `half` (jackknife), `tables`, `index`). Metric code copied from `phase1_svd.py` and `phase8_stability.py` in the same folder. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/maxtoki_adapter.py`.
- Outputs (`projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/`):
  - `run_config.json`: devices, versions, seeds, sample hashes, wall time per job, code sha256.
  - `prepare/`: `cells_A.npz`, `cells_B.npz`, `cells_C.npz` (rows, tokens, blocks, assay), `cells.csv`, `panel.json` (both gene panels), `run_config.json` (cell rows, gene IDs, counts and token sha256, encoding-check result).
  - `extract/<job>/` for jobs `trained_A`, `trained_B`, `trained_C`, `rand0_A`, `rand1_A`, `rand2_A`, `rand0_B`, `rand0_C`: `acc_sums.npy` (block sums), `acc_counts.npy`, `percell_sums.npy`, `cells.csv` (per-cell loss), `layer_gene_embeddings_panelD.npy` (12 × 1,500 × 1,232 per-gene vectors), `gene_counts_panelD.npy`, `state.json`, `run_config.json` (model, initialisation seed and hash, environment, encoding check, chunk times).
  - `analysis/`: `per_layer_metrics.csv` and `per_layer_metrics_optional.csv` (ER, participation ratio, top-direction share, TwoNN and shuffle-null ER per job and layer), `intervals.json` (jackknife intervals: `er_jackknife_cells`, `cka3_jackknife`, `cka_half_jackknife`, `contrast`), `cka.json` (CKA point estimates per pair and the bootstraps not used for intervals), `cka_half.json`, `summary.json`, `report_tables.md`, `er_jackknife_reps_<job>.npy`, `run_config.json`.
  - `verify/verify.json`: the second-way checks.
  - `code_versions/`: exact copies of `v3_spectral.py` for each script hash recorded by an extraction job.
- Run time: 10,134 s of MPS (2.8 h) for the 16,000 forward passes, in 30 chunks; about 1 h of CPU analysis.
- To reproduce, from `projects/maxtoki`: `OMP_NUM_THREADS=4 .venv/bin/python runs/spectral-geometry-217M/scripts/v3_spectral.py prepare`, then `extract --max-minutes 7` until it prints `ALL_DONE True` (add `--jobs rand0_B,rand0_C` for the seed-0 runs on B and C), then the `analyze`, `v3_spectral_intervals.py` and `verify` parts.


# Attention edges as gene-regulation scores (MaxToki-217M, K562 cells)

This section applies the attention analysis of section 01 to K562 cells. The two endpoints, the scores, the degree-preserving null, the residual models and the seeds are the same as in section 01. The cells, the gene set and the knockdown labels are those of K562. All numbers are for layer 8 unless a table says otherwise. **Endpoint A (knockdown response):** for one knocked-down gene, which other genes change? **Endpoint B (TRRUST):** for one transcription factor (TF), which genes does TRRUST list as its targets? AUROC 0.5 is chance.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, as in section 01. float32, Apple MPS, eager attention so that attention weights are returned. torch 2.11.0, transformers 5.5.4. Analysis on CPU (numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 2.3.3).
- **Data file.** Replogle K562 Perturb-seq data (`replogle_concat.h5ad`, 6,546 genes). `X` holds log1p(CP10k).
- **Cells for attention and gene statistics.** 2,000 non-targeting control cells, drawn with numpy `default_rng(42)` from 10,691 and sorted. These are the same rows as in the deployed attention analysis.
- **Counts.** Recovered from `X` as in section 00: round(expm1(X) / u), with u = the cell's smallest non-zero expm1(X) (`inputs_v3.CountReader("k562")` with `counts_accept_unit_one`).
- **Gene set.** log1p(counts / row sum × 10,000), with the row sum over the file's 6,546 genes (one log). The 1,500 genes with the highest variance over the 2,000 cells, among the 6,332 file genes in the MaxToki vocabulary. Gene features (mean, variance, dropout rate) and |Spearman| co-expression use the same values and the same 2,000 cells.
- **Model input.** Each cell's counts, encoded as in section 00: counts / total × 10,000 / gene median, sorted high to low, first 2,046 genes, `<bos>` and `<eos>` (max_len 2,048). Mean 2,039 tokens per cell (smallest 1,016). 1,946 of 2,000 cells are cut at 2,046 genes. A pair of two different genes is held by a median of 648 cells.
- **Knockdown labels.** Perturbed K562 cells from the same file, against 5,000 control cells drawn with `default_rng(42)` from the 10,691.
- **TRRUST.** TRRUST human TF–target table (version 2, `trrust_human.tsv`). 16 TFs have at least 3 targets in the gene set: HDAC1, JUN, NFE2L2, PTTG1, EZH2, MYC, TFDP1, CHD8, YY1, CTCF, BRCA1, DNMT1, JUND, CEBPB, XBP1, ATF4. This gives 107 positive and 23,877 negative (TF, gene) pairs. MYC alone has 27 of the 107 positives.

## Method

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

## Checks

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

## Results

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

## Verification

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

## Limits

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

## Files

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


# Developmental order in MaxToki-217M hidden states (Tabula Sapiens blood cells)

The deployed analysis fitted a small read-out head to MaxToki hidden states and reported that the model holds a blood developmental "manifold": an arrangement of cell groups that follows a curated tree of developmental stages and transfers to donors not used for fitting. This section asks whether MaxToki carries developmental order beyond cell-type identity. It uses the deployed cells, groups, layers, head and pass marks, with the model input encoded from counts (section 00). It compares MaxToki with three stand-ins that use no model, and with a null that keeps cell-type structure.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, eager attention, one cell per forward pass, Apple MPS. torch 2.11.0, transformers 5.5.4. The base `LlamaModel` was called (no output layer); its hidden states equal those of the full model (difference 0.0).
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`) and lung (`tabula_sapiens_lung.h5ad`). Counts come from `raw/X` (integer counts). The cell lists are those of the deployed analysis (`outputs/phase1/cells_<panel>_obs.csv`), with its cell subsample (numpy `default_rng(42)`, replayed from `phase1bc_hidden_states_and_centroids.py` and checked against the deployed number of cells per group). Every row's label matched the h5ad file.
- **Anchor.** One anchor is the group of cells that share donor × tissue × cell type × stage label. Its representation is the mean over those cells.
- **Stage tree (the "ruler").** The deployed curated tree of 34 blood stages (`planning/h65_stage_dag.json`). Each anchor gets a stage from its `free_annotation`, else its `cell_type`. The ruler distance between two anchors is the shortest path between their stages on the tree, treated as undirected. Each stage belongs to one branch (lineage).

| Panel | Role | Cells | Anchors | Donors | Cells per anchor | Smart-seq2 cells | Median tokens per cell |
|---|---|---:|---:|---:|---|---:|---:|
| internal | fit the head | 11,804 | 290 | 11 | 13–50 | 2,303 (19.5%) | 2,081 |
| external | frozen test; donors disjoint from internal | 12,000 | 600 | 13 | 20 | 673 (5.6%) | 2,217 |
| zero-shot | frozen test; donors disjoint from internal; its 12 donors are among the 13 external donors, but anchors differ | 5,120 | 160 | 12 | 32 | 470 (9.2%) | 2,148 |
| lung non-blood | negative control (16 non-blood lung cell types) | 1,500 | 50 | 4 | 30 | 186 (12.4%) | 3,093 |
| lung immune | blood cells in lung with real stage labels | 2,124 | 51 | 4 | 6–50 | 203 (9.6%) | 2,293 |

The two test panels share donors. All 12 zero-shot donors (`anchor_meta_zeroshot.csv`) are also external donors (`anchor_meta_external.csv`). No donor × tissue × cell type × stage group is in both panels, so the anchors and cells differ. The internal donors appear in neither test panel. So the external and zero-shot panels are not independent donor replications, and their donor bootstraps resample largely the same people.

**Which branches can test order.** Branches with one stage present (NK, macrophage, other lymphoid, dendritic) have a constant ruler and are not scored. Six branches are scored. Ruler values between anchors of different (cell type, stage) classes, from `v2b_tables.json` (`ruler_structure`), the same on all three blood panels except granulocyte on the zero-shot panel, where different classes are always at distance 2:

| Branch | Stages present (internal) | Ruler values between different classes |
|---|---|---|
| T lineage | DN_T, Naive/Memory CD4, Naive/Memory CD8, Treg (and gd_T on external) | 0, 1, 2, 3, 4 |
| B lineage | Naive_B, Memory_B, Plasma | 1 only |
| erythroid | MEP, Erythrocyte | 1 only |
| stem | HSC, MPP | 1 only |
| granulocyte | Basophil, Mast, Neutrophil | 0 or 2 (zero-shot: 2 only) |
| monocyte | Cl_Mono, NC_Mono | 0 or 1 |

So only the T lineage has a graded order. In the B, erythroid and stem branches the within-branch score can only measure whether anchors with the same label sit together.

## Method

**Part 1: hidden states and anchor features.**

1. Encode each cell from counts with the rank-value encoding of section 00 (`inputs_v3.tokenize_counts(check=True)`), context 4,096 tokens (at most 4,094 genes plus `<bos>` and `<eos>`).
2. Forward pass, batch 1. Keep all 12 hidden states (0 = embedding, 1–10 = outputs of blocks 0–9, 11 = after the final RMSNorm). Average each over gene positions 1 to L − 2 (no `<bos>`/`<eos>`).
3. Memory: from the fourth extraction call on, sequences were right-padded with `<pad>` (attention mask 0) to a multiple of 256 tokens. In a causal model padding does not change the real positions (checked below). The first 1,200 internal cells are unpadded.
4. Anchor centroid = mean of its cells' states (n × 12 × 1,232 per panel).
5. MaxToki feature (the deployed "pooled drift", 2,464 numbers): for blocks early (hidden states 1–4), mid (5–8) and late (9–11), average the states in the block and multiply by the block's operator A (the mean over the block's layers of the transposed attention output projection, 1,232 × 1,232). Feature = [y_early − y_mid, y_mid − y_late].
6. All features are standardised with the internal panel's mean and SD, so a head fitted on internal anchors applies unchanged to the other panels.

**Part 2: comparison features** (one row per anchor, the same cells as the MaxToki centroids). MaxToki PCA-64 is a reduced MaxToki feature and uses the model. The lookup, the token bag and HVG are the three stand-ins that use no model.

| Name | What it is |
|---|---|
| MaxToki PCA-64 (uses the model) | PCA to 64 numbers of the MaxToki feature (fitted on internal anchors), whitened; the same size as the stand-ins |
| Lookup (cell type), seeds 0–4 | one random 64-number Gaussian code per `cell_type` string, plus 0.5 × Gaussian noise per anchor. Knows the label and nothing else |
| Lookup (cell type + stage), seeds 0–2 | the same per (cell type, stage) class |
| Token bag | for each cell, each gene token at rank r of n gets weight 1 − r/n (the model's own input order); anchor mean; log1p; genes with SD > 0 on internal; standardise; PCA-64 fitted on internal anchors, whitened |
| HVG | `raw/X` counts per 10,000 + log1p per cell; anchor mean; 2,000 highly variable genes (Seurat v1 dispersion, 20 mean bins, internal cells only); standardise; PCA-64 fitted on internal anchors, whitened |

**Part 3: the read-out head and the gates** (the deployed code: `phase5_let_anchor.train_let`; frozen-panel code of `phase7_external_validation.py` and `phase8_zeroshot_transfer.py`).

1. Head: z = W(x − b), with W a 10 × d matrix. Predicted distance = β · arccos(cos(z_i, z_j)), β learnable. Loss = mean squared error to the ruler distances + 0.1 × reconstruction error. Adam, learning rate 5 × 10⁻³, 1,500 full-batch epochs, torch seed 42 (the deployed seed), 1 CPU thread.
2. Gates on the internal panel (head fitted on internal anchors):
   - trust = trustworthiness (scikit-learn, k = 15) of the 10-number output relative to the head's input feature. It checks that anchors among each other's 15 nearest neighbours in the output were also close in the input: it penalises output neighbours that were far apart in the input, weighted by their input rank. It never looks at the ruler.
   - random holdout = mean Spearman correlation between output distances and ruler distances on the held-out 20% of anchors, over 10 random 80/20 splits (head fitted on each 80%).
   - donor holdout = the same, leaving out one donor at a time (groups with ≥ 3 anchors).
   - branch holdout = the same, leaving out one branch at a time; plain mean over the 6 scored branches.
3. Gates on frozen panels (head fitted once on internal anchors): trust as above; random = mean Spearman over 20 random 20% subsets; donor = mean Spearman within each donor; within-branch = mean Spearman within each scored branch; global = Spearman over all anchor pairs of the panel.
4. Pass marks (fixed by the deployed analysis): trust ≥ 0.80; random, donor and branch Spearman ≥ 0.20.
5. "Different cell type" (diff): the same Spearman using only pairs of anchors whose `cell_type` differs. It averages fewer branches: 3 of 6 on internal and external, 2 on zero-shot, because in the other branches every such pair has the same ruler value.

**Part 4: nulls.**

1. Deployed null: stages shuffled among anchors within each branch (`default_rng(42)`). This also splits anchors of one cell type across stages.
2. Structured null (keeps cell-type structure). A class is a (cell type, stage) pair, over all blood panels. Each draw permutes the class → stage map among the classes of one branch. Every anchor keeps its branch, and all anchors of one class share one stage, so same-class pairs stay at distance 0, as in the real ruler. Only the developmental order between classes is broken.
   - Fitted null: the whole pipeline on the permuted ruler (internal head fitted again, then applied frozen to external and zero-shot under the same map). 40 draws (`default_rng([2026, d])`); internal branch holdout uses the first 20.
   - Evaluation-only null: the held-out branch heads fitted on the real ruler are kept; only the held-out ruler is drawn again. 2,000 draws (`default_rng([2027, d])`).
   - p = (1 + number of null values ≥ observed) / (1 + draws), one-sided.
3. Lung negative control, two designs, frozen head, 2,000 draws each: deployed design (one random stage per lung anchor, `default_rng([6161, d])`) and structured design (one random stage per lung cell type, `default_rng([6262, d])`).

**Part 5: intervals and seed spread.**

1. Donor cluster bootstrap: each replicate draws the panel's donors with replacement and keeps all their anchors; copies of one anchor are never paired; heads stay frozen (seed 42). 2,000 replicates (`default_rng([8080, panel, b])`), percentile 95% interval. Unit: donor (13 external, 12 zero-shot, 11 internal). Lookup = mean of the 5 cell-type seeds inside each replicate. Global numbers use pair weights, which equal expanding the resample.
2. Head seeds: the full internal head fitted with torch seeds 0–19 and scored with the same gate code. Seed-paired contrasts (same seed for both heads) use seeds 0–9.

**Part 6: the single head singled out by the deployment (L10H6).** Feature = hidden state 11 centroid × the 154 output-projection columns of attention head 6 in layer 10. Control: 30 Gaussian 154 × 1,232 projections of the same hidden state (`default_rng([6060, 11, seed])`). Scoring follows the deployed head screen (`phase9_head_attribution.py`): in-sample trust, 3 random splits, 6-branch holdout, composite = 0.5 trust + 0.25 random + 0.25 branch.

**Part 7: Smart-seq2 sensitivity.** Centroids rebuilt from 10x cells only. Anchors with no 10x cell are dropped (internal 290 → 247, external 600 → 587, zero-shot 160 → 152). The comparison is with all-cell centroids of the same kept anchors, over 10 head seeds (42 and 0–8).

## Checks

**Input encoding.**
- Every cell passed `inputs_v3.tokenize_counts(check=True)`, and every chunk of 50 cells passed `assert_encoding_batch`, before any forward pass: 653 chunks, 32,548 of 32,548 cells, 0 genes out of order. All count rows were whole numbers.
- A separate tokeniser (h5py `raw/X`, own Ensembl → token and gene-median maps, stable sort) gave the stored tokens for 108 of 108 cells, equal up to exact ties.
- Next-gene loss (mean cross-entropy of the model's prediction of each next gene token), 41 random 10x cells (`default_rng(20261003)`): 3.09 nats with the count-based input. With the deployed analysis's gene order (genes ranked by log1p(CP10k) divided by the gene median) it was 4.03 nats; difference −0.95 (95% percentile bootstrap interval over cells, 5,000 resamples: −1.18 to −0.73), lower in 41 of 41 cells. A uniform guess over the 20,275-token vocabulary scores 9.92. In 4 Smart-seq2 cells the loss was 7.82 nats.

**Forward passes and centroids.**

| Check | Result |
|---|---|
| Fresh full-model forward, no padding, against stored per-cell states (108 cells, padded and unpadded chunks) | largest difference 3.8 × 10⁻⁶ |
| Padding: 30 test cells, padded against unpadded | largest difference 3.8 × 10⁻⁶ (float32 rounding) |
| Fresh centroids of 3 whole anchors (13, 20, 30 cells) against stored | largest difference 4.8 × 10⁻⁷ |
| Deployed extraction code in this environment against a deployed centroid | difference 1.9 × 10⁻⁶ |
| Deployed tokens rebuilt from the stored `X` (5 cells per panel) | exact, which proves the row mapping |

**Analysis code.**
- The driver scripts import the shared analysis code: the task list (2,078 head fits), the task code, the seeds and the analysis arguments. The deployed head and internal gate code is imported read-only from `phase5_let_anchor.py`. The frozen-panel gate code of `phase7`/`phase8` sits inside their `main()`, so it is copied line for line.
- Rulers rebuilt from labels equal the deployed ruler files (and the deployed within-branch null ruler) on 4 panels.
- Own gate code, applied to the deployed centroids of the lung-immune panel, gives the deployed values exactly.

## Results

**Main findings.**
- MaxToki-217M shows no blood developmental order beyond cell-type identity that these tests detect. Only the T lineage has graded stages. Under the structured null, every branch and global number sits inside the null (one-sided p from 0.15 to 0.76). Two kinds of number are above it: zero-shot within-donor order, and internal and external trust at head seed 42 (structured-null table below).
- On pairs of anchors with different cell types, the branch numbers average near 0: 0.067 internal, 0.006 external, 0.016 zero-shot. These are means over 3, 3 and 2 branches; per branch they range from −0.28 to +0.30. In the T lineage, the only graded branch, held-out order is 0.038. On the frozen panels it is 0.205 (external) and 0.288 (zero-shot), below a cell-type lookup (0.347 and 0.484).
- A random code per cell-type label passes the three correlation gates on all three blood panels (5 of 5 seeds) and all four gates on external (4 of 5 seeds). Its trust is below 0.80 on the internal panel (0.754–0.790) and the zero-shot panel (0.710–0.770).
- On the zero-shot panel MaxToki fails the 0.20 within-branch gate: 0.180 at seed 42; 0.147 ± 0.048 over seeds 0–19; 3 of 20 seeds pass. This failure depends on the anchor set (Smart-seq2 paragraph below).
- Among the order numbers, MaxToki is clearly ahead of the stand-ins (donor-bootstrap interval above 0) only in cross-lineage (global) order. It beats a label lookup and HVG there by 0.06–0.11. Its lead over a bag of its own input tokens is small and depends on how the bag is built. On within-branch numbers the donor intervals include 0. MaxToki's trust is the highest of all representations on every blood panel, but trust does not use the ruler.

**H65 gates by representation** (head seed 42; lookup = cell-type seed 0; `table_gates_by_representation.csv`):

| Panel | Number | MaxToki | Lookup | Token bag | HVG | MaxToki PCA-64 |
|---|---|---:|---:|---:|---:|---:|
| internal (head fitted here) | trust | 0.845 | 0.785 | 0.705 | 0.717 | 0.686 |
| | random / donor | 0.849 / 0.783 | 0.739 / 0.661 | 0.674 / 0.556 | 0.726 / 0.676 | 0.796 / 0.575 |
| | branch holdout | 0.414 | 0.364 | 0.429 | 0.354 | 0.487 |
| | branch holdout, diff | 0.067 | 0.056 | −0.078 | 0.088 | 0.075 |
| | branch holdout, deployed null | −0.008 | −0.084 | −0.016 | 0.011 | – |
| external (frozen) | trust | 0.905 | 0.851 | 0.859 | 0.859 | 0.820 |
| | within-branch / diff | 0.417 / 0.006 | 0.312 / 0.070 | 0.386 / −0.039 | 0.367 / −0.011 | 0.381 / −0.014 |
| | global / global diff | 0.865 / 0.835 | 0.826 / 0.782 | 0.849 / 0.820 | 0.763 / 0.720 | 0.863 / 0.833 |
| zero-shot (frozen) | trust | 0.823 | 0.770 | 0.750 | 0.768 | 0.735 |
| | within-branch / diff | **0.180** / 0.016 | 0.319 / 0.145 | 0.133 / −0.085 | 0.030 / −0.060 | 0.457 / 0.160 |
| | global / global diff | 0.867 / 0.851 | 0.811 / 0.782 | 0.819 / 0.807 | 0.757 / 0.746 | 0.855 / 0.835 |
| lung non-blood (frozen) | trust / branch | 0.719 / −0.045 | 0.656 / −0.017 | 0.892 / −0.060 | 0.755 / −0.144 | 0.811 / −0.157 |

- Gate verdicts for MaxToki: internal and external pass all four; zero-shot fails the branch gate; lung non-blood fails all four.
- Lookup (cell type), 5 seeds: internally all pass random, donor and branch, and all fail trust (0.754–0.790). On external, 4 of 5 seeds pass all four gates (seed 4 fails trust, 0.781). On zero-shot, all 5 pass random, donor and branch, and all fail trust (0.710–0.770).
- Token bag and HVG pass all four gates on external and fail only trust internally.
- The deployed null fails for every representation, including the lookup (−0.084), because it splits cell types. It cannot tell order from identity.

**Per branch, internal held out** (MaxToki; `v2b_tables.json`, `summaries` → `h65|maxtoki|pos` → `per_group`):

| Branch | Anchors | MaxToki | MaxToki, diff |
|---|---:|---:|---:|
| T lineage | 101 | 0.038 | −0.027 |
| B lineage | 45 | 0.282 | – |
| monocyte | 34 | 0.020 | 0.018 |
| granulocyte | 25 | 0.530 | 0.211 |
| erythroid | 7 | 0.783 | – |
| stem | 4 | 0.828 | – |
| plain mean | | 0.414 | 0.067 |

The plain mean is carried by the small branches. Weighted by anchors it is 0.182, below the gate.

**T lineage, the only graded branch** (`table_h65_per_branch.csv`):

| Panel | MaxToki | Lookup (seed 0) | Token bag | HVG |
|---|---:|---:|---:|---:|
| internal, held out | 0.038 | 0.145 | 0.023 | 0.087 |
| external, frozen | 0.205 | 0.347 | 0.094 | 0.087 |
| zero-shot, frozen | 0.288 | 0.484 | −0.038 | 0.101 |

On frozen panels the head was fitted on internal T labels, and a lookup reproduces that learned order better than MaxToki.

**Structured null** (MaxToki; fitted null 40 draws, internal branch 20; evaluation-only null 2,000 draws; `v2b_tables.json` → `structured_null_refit`, `v2b_evalnull_internal_branch.json`):

| Number | Observed | Null mean (SD) | p |
|---|---:|---:|---:|
| internal branch holdout (fitted null) | 0.414 | 0.384 (0.073) | 0.38 |
| internal branch holdout (evaluation-only) | 0.414 | 0.402 (0.054) | 0.39 |
| internal branch, diff (evaluation-only) | 0.067 | 0.007 (0.099) | 0.27 |
| internal trust (fitted null) | 0.845 | 0.805 (0.013) | 0.024 |
| external within-branch | 0.417 | 0.332 (0.071) | 0.15 |
| external within-branch, diff | 0.006 | 0.099 (0.117) | 0.71 |
| external global | 0.865 | 0.843 (0.033) | 0.29 |
| external trust | 0.905 | 0.885 (0.011) | 0.024 |
| zero-shot within-branch | 0.180 | 0.124 (0.101) | 0.27 |
| zero-shot global | 0.867 | 0.831 (0.033) | 0.15 |
| zero-shot trust | 0.823 | 0.804 (0.012) | 0.098 |
| zero-shot within-donor | 0.880 | 0.764 (0.073) | 0.024 |

- Every branch and global number is inside its null.
- The internal and external trust rows reach the smallest p that 40 draws allow, but these use head seed 42. Using the seed mean over seeds 0–19, trust is 1.0 null SD above the null mean internally, 1.8 externally and 1.4 on zero-shot. Only 2 of 20 seeds (10%) are above the largest internal null value. Trust also never looks at the ruler.
- Zero-shot within-donor order is above its null (p = 0.024, one seed). Within-donor pairs include pairs from different branches, so this number mixes in cross-lineage order. It is also the one number that the Smart-seq2 cells move (below).
- The lookup, token bag and HVG are inside their own nulls on every branch number (p 0.12–0.85 across the three).

**Head-seed spread** (seeds 0–19; `seeds_summary.json`). Mean ± SD [min, max]:

| Number | 20 seeds | Seed 42 |
|---|---|---:|
| internal trust | 0.818 ± 0.009 [0.809, 0.844] | 0.845 |
| external trust | 0.904 ± 0.002 | 0.905 |
| external within-branch | 0.422 ± 0.011 | 0.417 |
| external within-branch, diff | 0.013 ± 0.016 | 0.006 |
| external global | 0.856 ± 0.004 | 0.865 |
| zero-shot trust | 0.822 ± 0.004 | 0.823 |
| zero-shot within-branch | 0.147 ± 0.048 [0.057, 0.248]; 3 of 20 ≥ 0.20 | 0.180 |
| zero-shot global | 0.871 ± 0.005 | 0.867 |
| lung non-blood trust | 0.725 ± 0.018 | 0.719 |

Seed 42 gives the highest internal trust of the 21 seeds. The zero-shot branch failure holds across seeds.

**Cross-lineage (global) order and contrasts** (donor bootstrap, frozen heads, 2,000 replicates, percentile 95% interval; lookup = mean of 5 seeds; `v2b_boot_<panel>.json`, `v2b_boot_<panel>_global.json`):

| Representation | External global [95% CI] | Zero-shot global [95% CI] |
|---|---|---|
| MaxToki | 0.865 [0.839, 0.896] | 0.867 [0.828, 0.888] |
| Lookup (mean of 5 seeds) | 0.801 [0.779, 0.821] | 0.776 [0.700, 0.822] |
| Token bag | 0.849 [0.809, 0.882] | 0.819 [0.691, 0.866] |
| HVG | 0.763 [0.714, 0.808] | 0.757 [0.704, 0.821] |

| Contrast | Panel | Within-branch | Within-branch, diff | Global | Global, diff |
|---|---|---|---|---|---|
| MaxToki − lookup | external | +0.075 [−0.038, 0.138] | −0.021 [−0.187, 0.075] | **+0.064 [0.050, 0.091]** | **+0.086 [0.068, 0.119]** |
| MaxToki − token bag | external | +0.031 [−0.067, 0.094] | +0.044 [−0.016, 0.184] | +0.016 [0.002, 0.046] | +0.015 [−0.000, 0.047] |
| MaxToki − HVG | external | +0.050 [−0.062, 0.111] | +0.017 [−0.087, 0.167] | **+0.102 [0.073, 0.140]** | **+0.115 [0.086, 0.156]** |
| MaxToki − lookup | zero-shot | −0.197 [−0.567, 0.005] | −0.210 [−0.366, 0.153] | **+0.091 [0.027, 0.163]** | **+0.109 [0.041, 0.186]** |
| MaxToki − token bag | zero-shot | +0.047 [−0.332, 0.363] | +0.100 [−0.059, 0.424] | +0.048 [0.009, 0.149] | +0.044 [0.008, 0.140] |
| MaxToki − HVG | zero-shot | +0.149 [−0.177, 0.298] | +0.076 [−0.125, 0.342] | **+0.110 [0.038, 0.151]** | **+0.105 [0.036, 0.154]** |
| MaxToki − lookup | internal (held out) | +0.021 [−0.393, 0.063] | +0.013 [−0.438, 0.087] | – | – |
| MaxToki − token bag | internal (held out) | −0.016 [−0.358, 0.063] | +0.146 [−0.025, 0.594] | – | – |
| MaxToki − HVG | internal (held out) | +0.060 [−0.347, 0.083] | −0.021 [−0.131, 0.082] | – | – |

MaxToki within-branch values: external 0.417 [0.297, 0.532]; zero-shot 0.180 [−0.103, 0.369]. Internal rows resample evaluation donors only (held-out heads fixed), so they are too narrow.

**Seed-paired contrasts, global order** (seeds 0–9, mean ± SD; `seeds_summary.json` → `seed_paired_contrasts`):

| Contrast | External | Zero-shot |
|---|---|---|
| MaxToki − token bag | +0.004 ± 0.005 (7 of 10 > 0) | +0.053 ± 0.005 (10 of 10) |
| MaxToki − HVG | +0.090 ± 0.007 | +0.108 ± 0.010 |
| MaxToki − lookup (seed 0) | +0.028 ± 0.003 | +0.059 ± 0.004 |

**Token bag built from the deployed gene order.** A second token bag, built the same way from the gene order the deployed analysis fed the model (genes ranked by log1p(CP10k) divided by the gene median), is also model-free. Its global order is 0.876 [0.837, 0.903] external and 0.879 [0.823, 0.902] zero-shot. MaxToki minus this bag: −0.010 [−0.023, 0.016] external and −0.012 [−0.031, 0.022] zero-shot (own bootstrap code, 1,000 replicates, unit donor, percentile; `verify/verify_gates.json`). So the zero-shot lead depends on which gene order the bag uses: zero-shot global order is 0.819 for the count-based bag, 0.879 for the deployed-order bag and 0.867 for MaxToki.

Reading:
- On every within-branch number, MaxToki is not separable from a lookup or from the expression baselines.
- On global order it beats the lookup and HVG by 0.06–0.11, with intervals clear of 0.
- Against a bag of its own input tokens it ties on external over head seeds (+0.004 ± 0.005) and leads on zero-shot by about 0.05. A bag of the deployed gene order matches it on both panels.
- Global order depends mostly on lineage and cell-type grouping. The structured null keeps both and breaks only the stage order between classes within a branch, and it keeps most of global order (null means 0.843 and 0.831 against 0.865 and 0.867; p = 0.29 and 0.15). How much of global order comes from pairs across lineages was not measured directly.

**Lung negative control** (`v2b_lung_control_6161.json`, `v2b_lung_control_6262.json`).
- Deployed design (one random stage per lung anchor): random, donor and branch gates pass together in 0 of 2,000 draws for all 12 representations, MaxToki and pure label codes alike. Branch alone passes in 9.5–12.0% of draws. This control cannot fail any representation.
- Structured design (one random stage per lung cell type), frozen head:

| Representation | Branch mean | Pass branch | Pass random | Pass donor | Pass random + donor + branch |
|---|---:|---:|---:|---:|---:|
| MaxToki | 0.470 | 93.0% | 15.5% | 5.8% | 5.2% |
| Lookup (seed 0) | 0.508 | 96.5% | 14.6% | 3.1% | 2.7% |
| Token bag | 0.563 | 97.2% | 19.0% | 5.3% | 5.2% |
| HVG | 0.430 | 89.9% | 12.1% | 4.8% | 4.2% |

Once random labels follow lung cell types, the branch gate passes for meaningless labels in about 93% of draws. MaxToki's lung trust is 0.719, so "all four" passes in 0%.
- Lung immune panel (blood cells in lung with real stage labels), frozen head, own code: trust 0.799, random 0.878, donor 0.899, branch 0.470, global 0.901. It passes the three correlation gates and is just below the trust gate.

**L10H6 against random projections of the same hidden state** (`v2b_n60_random_projection.json`):

| Number | L10H6 | 30 random projections, mean ± SD | Share of random ≥ L10H6 |
|---|---:|---|---:|
| trust | 0.875 | 0.809 ± 0.011 | 0 / 30 |
| random holdout | 0.789 | 0.776 ± 0.016 | 17% |
| branch holdout | 0.419 | 0.403 ± 0.071 | 50% (15 / 30) |
| branch holdout, diff | 0.066 | 0.049 ± 0.061 | 43% |
| composite | 0.739 | 0.699 ± 0.021 | 0 / 30 |

L10H6 leads only through trust, which does not use the ruler, and so through the composite, which is half trust. Its features are more concentrated (participation ratio 12.2 against 14.7, range 13.7–16.1, for the random projections). Whether this explains the trust lead was not tested. The scoring code is the deployed head screen's: trust and branch holdout are computed as for the H65 gates (trust of the head fitted on all internal anchors, relative to its own input feature), but random holdout uses 3 splits instead of 10.

**Smart-seq2 cells** (`verify/verify_ss2.json`). Smart-seq2 is a plate-based method. `raw/X` holds read counts for its cells, and the model predicts their genes poorly (next-gene loss 7.82 nats in 4 cells against 3.09 in 10x cells; Checks). Paired change, 10x-only minus all cells on the same kept anchors, mean over 10 head seeds: internal trust +0.010; external trust +0.002, global +0.003, within-branch −0.001; zero-shot global −0.005, within-branch −0.016, within-donor −0.078. Every other paired change was at most 0.016 in size. The paired check covers trust on all three panels and the frozen-panel numbers; it does not cover the internal random, donor or branch holdout. So with the anchors held fixed, the Smart-seq2 cells barely matter, except for zero-shot within-donor order. The anchor set matters more: on the kept anchors (247 internal, 152 zero-shot), zero-shot within-branch is 0.34 ± 0.01 (all cells, 10 seeds) instead of 0.147 on the full panels. Dropping the 8 zero-shot anchors made only of Smart-seq2 cells (and 43 such internal anchors) changes it by about 0.2.

**Other orderings tested by the deployed analysis** (head seed 42; structured null = class → category map permuted; 30 draws, except H95 within-category, which has 22 valid draws on external and 28 on zero-shot):

| Ordering | Result |
|---|---|
| H38 (7 signalling categories) | Internal MaxToki trust / random / donor / category holdout 0.801 / 0.758 / 0.599 / 0.229. External category: MaxToki 0.608, token bag 0.701, HVG 0.640, lookup (seed 0) 0.826. MaxToki 0.608 against null mean 0.331, p = 0.032; token bag and HVG reach the same p. Any expression-based representation follows these categories, and the lookups do better. |
| H95 (4 effector categories) | Internal trust 0.8018 with 1 CPU thread and 0.7998 with 4, so its pass of the 0.80 gate depends on the thread count. External within-category 0.206 against a null mean of 0.364, p = 0.78. |
| H103 (B-cell maturation) | Only two labels exist ("B cell", "plasma cell"), so the test only asks whether two labels separate. Internal random / donor 0.861 / 0.861 for MaxToki and for each of the 3 cell-type lookup seeds. |

## Verification

No separate verification by a second agent is recorded for this analysis. The run includes these second-way checks (script `scripts/v3_devorder_verify.py`, outputs `outputs/v3_devorder/verify/`; fresh model passes on MPS, the rest on CPU):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Model input | `inputs_v3` tokeniser | own tokeniser (h5py, own maps, stable sort) | 108 / 108 cells equal up to exact ties |
| Per-cell hidden states | stored chunks | fresh full-model forward, no padding | largest difference 3.8 × 10⁻⁶ (108 cells) |
| Anchor centroids | stored | fresh, 3 whole anchors | largest difference 4.8 × 10⁻⁷ |
| Pooled-drift features | feature files | own code | largest difference 7.5 × 10⁻⁶ |
| Rulers | stored ruler files | rebuilt from labels | identical, 4 panels |
| Internal branch, evaluation-only null | 0.414 vs null mean 0.402, p = 0.39 | own code, new stream (`default_rng([31338, d])`, 2,000 draws): null mean 0.398, p = 0.38; diff 0.067 vs 0.004, p = 0.27 | same reading |
| Donor bootstrap, global contrasts | pair weights, 2,000 replicates | explicit resampling, new stream, 1,000 replicates | external MaxToki − token bag +0.016 [0.002, 0.044] (main [0.002, 0.046]); zero-shot +0.048 [0.011, 0.136] (main [0.009, 0.149]); MaxToki − HVG and − lookup within 0.006 of the main interval ends |
| Structured lung control | analysis code | own code, new stream (`default_rng([62620, d])`) | MaxToki passes branch in 93.7% of draws (main 93.0%) |

One check failed in an informative way. Fitting the seed-42 head from the own features, which differ from the feature files by at most 7.5 × 10⁻⁶, gave internal trust 0.813 instead of 0.845. The frozen-panel numbers moved by up to 0.014 (for example, external within-branch 0.422 against 0.417). So single-seed numbers carry optimiser noise. The head-seed table measures it, and the main text quotes seed means where they matter.

## Limits

- **No random-initialised MaxToki** and no other cell model (scVI, Geneformer, Palantir, CellTypist) was compared.
- **The deployed 88-head screen was not run on these centroids.** So it is not known whether L10H6 would still be the top head. The L10H6 comparison only asks whether that head beats random projections.
- **Most numbers use one head seed (42).** Seeds 0–19 cover the full internal fit only (seeds 0–9 for the paired contrasts; seeds 42 and 0–8 for the Smart-seq2 check), not the held-out branch fits, the nulls or the bootstraps.
- **Fitted nulls are coarse:** 40 draws (20 for internal branch holdout), so the smallest p is 0.024 (0.048).
- **Donor intervals cover the within-branch and global numbers and the contrasts, not trust.** Internal bootstrap intervals are too narrow (evaluation donors only, heads fixed).
- **The two test panels share donors.** All 12 zero-shot donors are external donors; only the anchors differ. So the zero-shot panel is not an independent donor replication of the external panel.
- **Only the T lineage has graded stages.** In the other scored branches, the within-branch numbers can only show whether anchors with the same label sit together.
- **Only a curated stage tree was tested.** No real non-blood differentiation ruler and no new anchors.
- **The lookup noise level (0.5) and code size (64) were fixed, not tuned.** Lookup trust changes with them.
- **The thread-count effect** (H95) was checked at 1 and 4 threads only.
- **Smart-seq2 cells were kept** in the main analysis (`raw/X` holds read counts for them, and the model predicts their genes poorly). The sensitivity analysis above shows their effect.
- **Process memory** went above the 8 GB target (about 8.5 GB) in the first three extraction calls, which used unpadded sequences. Results are not affected (padding check above).
- The raw h5ad files (19.8 GB, 3.2 GB) were fingerprinted (size, time, sha256 of the first and last 16 MB), not hashed in full. The rows used are listed in `run_config_centroids.json`.

## Files

- Scripts (`projects/maxtoki/runs/manifold-discovery-217M/scripts/`):
  - drivers: `v3_devorder_centroids.py` (cell plans, encoding checks, forward passes, centroids), `v3_devorder_features.py` (MaxToki, lookups, token bag, HVG), `v3_devorder_pool.py` (all 2,078 head fits; thread check), `v3_devorder_analyze.py` (tables, nulls, lung control, bootstraps, random projections), `v3_devorder_seeds.py` (head seeds), `v3_devorder_verify.py` (second-way checks), `v3_devorder_common.py` (paths);
  - analysis code they import: `v2b_devorder_common.py`, `v2b_devorder_01_features.py` (feature recipes), `v2b_devorder_02_pool.py` (task list and task code), `v2b_devorder_03_analyze.py` (tables, nulls, bootstraps);
  - deployed code used read-only: `phase5_let_anchor.py` (head and internal gates), `phase7_external_validation.py` and `phase8_zeroshot_transfer.py` (frozen-panel gates), `phase1bc_hidden_states_and_centroids.py` (cell subsample, pooling), `phase9_head_attribution.py` (single-head scoring), `phase13_h38_lite.py`, `phase3a_manifold_sweep.py`, `phase3a_sweep2_ordinal.py` (H38, H95, H103);
  - shared input code: `projects/maxtoki/setup/inputs_v3.py`.
- Stage tree: `projects/maxtoki/runs/manifold-discovery-217M/planning/h65_stage_dag.json`. Deployed cell lists: `projects/maxtoki/runs/manifold-discovery-217M/outputs/phase1/cells_<panel>_obs.csv`.
- Outputs (`projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/`):
  - centroids and metadata: `artifacts/anchors/centroids_<panel>.npy` (n × 12 × 1,232), `anchor_meta_<panel>.csv`, `d_target_<panel>.npy` (rulers); operators: `artifacts/operators/`; per-cell states: `percell/<panel>/chunk_*.npz`; tokens: `cells/tokens_<panel>.npz`; cell plans: `cells/plan_<panel>.csv`, `cells/plan.json`;
  - features: `features/<representation>__<panel>.npy`, `features_manifest.json`;
  - head fits: `pool/results.jsonl`, `heads/`, `thread_check/results.jsonl`;
  - analysis: `v2b_tables.json` (summaries, ruler structure, fitted structured nulls), `table_gates_by_representation.csv`, `table_h65_per_branch.csv`, `v2b_evalnull_internal_branch.json`, `v2b_lung_control_6161.json`, `v2b_lung_control_6262.json`, `v2b_boot_external.json`, `v2b_boot_zeroshot.json`, `v2b_boot_internal_branch.json`, `v2b_boot_external_global.json`, `v2b_boot_zeroshot_global.json`, `v2b_n60_random_projection.json`, `v2b_n60_effective_dimension.json`;
  - head seeds: `seeds/results.jsonl`, `seeds_summary.json`;
  - checks: `verify/verify_gates.json`, `verify/verify_ss2.json`, `verify/ss2_per_seed.csv`, `verify/fresh_nll.json`, `verify/fresh_nll_cells.csv`, `verify/fresh_anchors.json`, `verify/fresh_anchor_cells.csv`;
  - provenance: `run_config_centroids.json` (device, versions, seeds, every cell row and label, encoding record and check result per panel, sha256 of inputs, code and outputs, wall time per chunk), `run_config.json` (analysis seeds and sha256), `run_config_features_*.json`, `verify/run_config.json`.
- Run time: 30,157 s of forward passes on MPS (0.84–1.27 s per cell, shared GPU), in 74 calls of at most about 7.8 minutes.


# Steering along an erythroid maturity score (MaxToki-217M, Tabula Sapiens bone marrow)

This analysis asks whether single SAE features act as maturity switches. Steering here means: while the model reads a cell, scale the part of the hidden state that one SAE feature writes, at every gene position, and measure how far the model's output moves along a maturity score. A feature is a switch only if it moves cells further, in its predicted direction, than random features of similar activity. The deployment reported steering effects and gene lists at five layers; its hook deleted a transformer block (at layer 11 it applied the final normalisation twice), and with a zero edit it gives the same table (main text). Everything below (differentiation series, cells, read-out, features, null, edits and tests) was written to `design.json` at 08:53:02, before the first forward pass at 08:53:38, and was not changed. The design fixed how each p value is computed, but it set no pass threshold. Analyses added after the first look at the results are marked "post hoc".

Here, moving a cell means moving the model's output score for that cell. No change in the cell's real state was measured.

## Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS, one cell per sequence. torch 2.11.0, transformers 5.5.4.
- **SAEs.** The SAEs of section 02 (trained on 500 correctly encoded K562 control cells; 4,928 features; TopK 32), at layers 0, 3, 6, 9 and 11. The site of layer l is the input of block l (`hidden_states[l]`); layer 11 is the input of `lm_head` (section 00).
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`), tissue = bone marrow, assay = 10x 3' v3 only. Counts from `raw/X`, encoded as in section 00 (at most 2,048 tokens).
- **Differentiation series.** The erythroid branch of blood development, with stage numbers from the `cell_type` annotation: 0 hematopoietic stem cell (HSC), 1 common myeloid progenitor (CMP), 2 erythroid progenitor cell (EryP), 3 erythrocyte. Stage numbers are ordinal labels, not measured time. Left out: "hematopoietic precursor cell" (its free annotations mix granulocytes, monocytes and other cells), Smart-seq2 cells (read counts, which MaxToki-217M handles badly) and 10x 5' v2 cells (so that one assay covers every stage).
- **Cell groups** (numpy `default_rng(20261003)`; groups drawn in a fixed order; 442 cells in total):

| Group | Donors | HSC | CMP | EryP | Erythrocyte | Used for |
|---|---|---:|---:|---:|---:|---|
| Selection | TSP14, TSP21, TSP25 (up to 30 per stage and donor) | 76 | 90 | 76 | 90 | maturity score, span, choice of candidates and null features |
| Steered | TSP2 (47), TSP27 (3) | 50 | – | – | – | steering |
| Axis check | TSP2, TSP27 | the 50 steered HSCs | 30 | 30 | 0 | held-out check of the score (clean passes only) |

- The steering donors are disjoint from the selection donors. TSP27 has only 9 HSCs, so 47 of the 50 steered HSCs come from TSP2. TSP2 and TSP27 have no erythrocytes, so the held-out check uses EryP as the late group.
- Median tokens per cell: 2,048 in every group except selection erythrocytes (1,796).

## Method

1. **Read-out.** For each cell, m = the next-gene logits averaged over all positions (20,275 tokens). g_early = mean m of the 76 selection HSCs; g_late = mean m of the 90 selection erythrocytes. Maturity score s(m) = cos(m, g_late) − cos(m, g_early). The effect of an edit is Δs = s(steered) − s(clean) for the same cell, both computed by the same code path. Δs has no unit.
2. **Span.** S = mean s of selection erythrocytes − mean s of selection HSCs = 0.198. Effects are also given as a percentage of S. A shift of 100% equals the full gap between the mean HSC and mean erythrocyte scores in the selection donors. (The steered HSCs come from other donors and start higher, at a mean s of −0.077 against −0.104.)
3. **Second read-out.** Gap share = (m_steered − m_clean) · (g_late − g_early) / |g_late − g_early|²: the share of the straight line from g_early to g_late that the edit moves the cell.
4. **Axis check (rule set in advance).** On axis-check cells: AUROC of s for EryP against HSC ≥ 0.80, and Spearman(s, stage) > 0 over HSC, CMP and EryP. Steering would not have run if this failed.
5. **Feature activity.** Per cell, a_f = mean over gene positions (not `<bos>`/`<eos>`) of the SAE code z_f. A feature is alive if z_f > 0 at ≥ 1 gene position in ≥ 50% of selection HSCs. Alive features: L0 2,481; L3 4,912; L6 4,370; L9 3,348; L11 4,190 (of 4,928).
6. **Candidates.** ρ_f = mean over the 3 selection donors of the within-donor Spearman correlation between a_f and stage (0–3). The candidates are the 3 alive features with the largest |ρ_f| at each layer (ties: lower feature ID). 15 candidates in total.
7. **Random-feature null.** 20 other alive features per layer, drawn at random without replacement (`default_rng(20261003 + layer)`): 7, 7 and 6 from the 50 alive features closest to each candidate in log10(mean a_f over selection HSCs). Mean activity sets the size of the edit, so null edits are about as large as candidate edits in selection HSCs.
8. **Edit.** `hooks_v2.Steer(layer, f, α, mode="scale")` at gene positions only: added vector = (α − 1) · z_f(x) · W_dec[:, f], with z computed from the live input at the hook. α = 0 removes the feature, α = 2 doubles it, α = 5 multiplies it by 5; α = 1 adds nothing. Predicted sign: for α = 2 and 5, sign(Δs) = sign(ρ_f); for α = 0, −sign(ρ_f). Edit size = mean L2 norm of the added vector per gene position.
9. **Positive controls.** `hooks_v2.AddVector` at gene positions, with v = mean gene-position hidden state at the site over selection erythrocytes minus that over selection HSCs. Two sizes: the full v ("full control"), and v scaled to the planned α = 5 edit size of the layer's candidates, 4 × the mean over the 3 candidates of mean a_f in selection HSCs ("size-matched control").
10. **Runs.** 50 cells × 5 layers × (23 features × 3 α + 2 controls) = 17,750 steered passes, plus 442 clean passes. Exact speed-up: an edit at site l changes nothing before block l, so each steered pass runs only blocks l to 10, starting from the stored clean input of block l, and computes the position-mean logits as `lm_head`(mean of the normed hidden states) (`lm_head` is linear with no bias). Every row is compared with a clean reference computed by the same path.
11. **Statistics.** 95% intervals: percentile bootstrap over the 50 steered cells, 10,000 resamples, seed 20261003 (the same resamples for every row), unless stated otherwise. Directional p per candidate and α = (1 + number of null features with an effect at least as large in the predicted direction) / (1 + 20); the smallest possible p is 1/21 = 0.048. Size p uses |Δs|. The design fixed these p values but no pass threshold. Below, a candidate is called a hit if its directional p is 0.048 (no null feature of its layer as large in the predicted direction) at any α; only L9 F2903 reaches this, and it does so at every α. Layer test: the 3 candidates against all C(23, 3) = 1,771 sets of 3 of the layer's 23 tested features (each feature signed by its own ρ). Pooled tests: (a) the candidates' mean within-layer directional rank, summed over the 5 layers, with labels permuted within layers (100,000 permutations); this is the permutation test that `design.json` names. (b) The mean over the 5 layers of the within-layer Spearman(ρ_f, Δs_f) over 23 features, also with labels permuted within layers (100,000 permutations), with one-sided and two-sided p. `design.json` also names a Spearman over all 115 features pooled; that number is not reported.
12. **Post-hoc analyses** (added after the first look at the results; marked "post hoc" below): the gap-share comparison with the null; the per-unit-of-edit comparison (mean Δs / mean edit size); a donor split; comparison with all 100 null features pooled over layers; the cosine between each mean logit change and g_late − g_early; the number of erythroid genes among the 10 most-raised genes (14-gene list: AHSP, ALAS2, HBM, GYPA, SLC4A1, HBG1, HBG2, HBZ, HBD, GYPB, KLF1, HBA1, HBA2, HBB); and gene descriptions of the candidates in 32 selection cells (8 per stage, all from donor TSP14). "Most-raised genes" = largest mean increase of the position-mean logit over the 50 cells at α = 5.

## Checks

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding, all cells | token order = counts / median order; counts are count-like (section 00) | 442 / 442 pass; 0 order violations (38 positions with exact ties, allowed) | yes |
| Encoding, negative control | the input order of the deployed analysis (stored values ranked as if they were counts) must fail | fails for 442 / 442 | yes |
| Encoding, before each forward pass | counts re-read from `raw/X`, sha256 equal to the prepared counts, check run again | all clean, steered and verify cells pass | yes |
| Axis check (rule set in advance) | held-out donors: AUROC(EryP vs HSC) ≥ 0.80 and Spearman(s, stage) > 0 | 0.991 and 0.844 | yes |
| Speed-up = full forward pass | 75 rows (cells 332–334; candidates, null features, controls) against `hooks_v2.run_with_edits` | largest Δs difference 2.2 × 10⁻⁹; largest logit difference 1.9 × 10⁻⁶ | yes |
| Zero edit | α = 1 and `ZeroDelta`, speed-up path and full path | logits and Δs change by 0.0 exactly (18 rows) | yes |
| Silent feature | a feature with z = 0 at every position | change 0.0 exactly (15 rows) | yes |
| CPU vs MPS | cell 332, L9 F2903, α = 5 | Δs difference 7.0 × 10⁻¹⁰ | yes |
| Silent rows in the main run | every (cell, feature, α) row where the feature is inactive | 807 rows; largest \|Δs\| 0.0 and largest logit change 0.0 | yes |
| Clean reference vs full pass | speed-up clean logits vs full clean logits, all cells and layers | largest difference 1.9 × 10⁻⁶ | yes |
| Edit applied when active | edit size > 0 in every row where the feature is active | yes | yes |
| Memory | free memory checked before each cell; stop below 3 GB | lowest free memory 5.8 GB; no chunk stopped | yes |

## Results

**The maturity score.**

| Quantity | Value |
|---|---|
| cos(g_early, g_late) | 0.894 (the two signatures are close, so s moves in small numbers) |
| Selection cells, mean s (SD 0.03–0.05 per stage) | HSC −0.104; CMP −0.009; EryP +0.063; erythrocyte +0.094 |
| Span S | 0.198 |
| Within-donor Spearman(s, stage), selection donors | TSP14 0.86; TSP21 0.73; TSP25 0.89 |
| Held-out donors, mean s | HSC −0.077; CMP +0.001; EryP +0.079 |
| Held-out AUROC (0.5 = chance) | EryP vs HSC 0.991; CMP vs HSC 0.917; EryP vs CMP 0.887 |
| Held-out Spearman(s, stage), HSC/CMP/EryP | 0.844 [0.782, 0.885] (percentile bootstrap over the 110 cells, 2,000 resamples) |
| Within-stage Spearman(s, number of tokens), selection cells | HSC −0.42; CMP −0.41; EryP +0.05; erythrocyte +0.27 |

AUROC is the probability that a random EryP cell scores above a random HSC.

**How well the K562-trained SAEs fit these cells.** Fraction of variance explained (FVE), all positions:

| Layer | Steered HSCs | Selection cells | Held-out K562 tokens (section 02) |
|---|---:|---:|---:|
| L0 | 0.46 | 0.50 | 0.700 |
| L3 | 0.59 | 0.62 | 0.912 |
| L6 | 0.37 | 0.39 | 0.847 |
| L9 | 0.34 | 0.37 | 0.859 |
| L11 | 0.43 | 0.42 | 0.871 |

A large part of what these bone-marrow cells carry is not in any single feature.

**Positive controls: the read-out can move.** Δs at α = 5 for the null; controls as described in Method step 9.

| Layer | Null Δs: mean (SD) | Null range | Null mean \|Δs\| (% of span) | Candidates mean \|Δs\| (% of span) | Size-matched control, % of span [95% CI] | Candidates / size-matched control | Edit size: candidates / null / size-matched control | Full control, % of span [95% CI] | Full control gap share |
|---|---|---|---|---|---|---|---|---|---|
| L0 | +1.0e-4 (3.0e-4) | [−4.9e-4, +6.9e-4] | 0.12% | 0.095% | 2.91% [2.59, 3.26] | 0.033 | 0.0045 / 0.0046 / 0.0046 | 23.7% [21.6, 25.8] | 0.26 |
| L3 | +2.8e-4 (7.2e-4) | [−2.2e-4, +2.4e-3] | 0.18% | 0.043% | 0.63% [0.57, 0.70] | 0.068 | 0.094 / 0.105 / 0.100 | 73.4% [70.3, 76.3] | 0.71 |
| L6 | +9.9e-6 (2.6e-5) | [−2.5e-5, +1.1e-4] | 0.008% | 0.004% | 0.09% [0.09, 0.10] | 0.040 | 0.031 / 0.031 / 0.021 | 73.1% [70.3, 75.9] | 0.71 |
| L9 | −7.2e-7 (1.1e-3) | [−2.9e-3, +1.9e-3] | 0.33% | 1.65% | 3.90% [3.77, 4.03] | 0.42 | 2.80 / 1.24 / 1.43 | 89.3% [87.4, 91.1] | 0.85 |
| L11 | −6.9e-5 (2.5e-4) | [−7.1e-4, +3.6e-4] | 0.075% | 0.039% | 1.36% [1.30, 1.42] | 0.029 | 0.45 / 0.74 / 0.63 | 109.5% [103.3, 115.6] | 1.00 |

- The full control moved 50 of 50 HSCs toward the erythrocyte signature at every layer, by 24% (L0) to 109% (L11) of the span on average (means over the 50 cells; single cells vary around these means). Its edits are far larger than the feature edits (24.9 against 3.44 per gene position for L9 F2903 at L9). At L11 the gap share is 1.000 and the cosine between the logit change and g_late − g_early is 0.9999996. This must hold, because `lm_head` is linear, so the L11 result checks the read-out code, not the model.
- At the planned size of the candidate edits, the same direction moved cells 0.09–3.9% of the span. Except at L9, the candidates reached 3–7% of what this direction reached at about the same edit size. At L9 the candidates' actual edits (mean 2.80 per gene position) were about twice the planned size (1.43).
- The full control raised AHSP, HBM, ALAS2, SLC4A1 and GYPA at every layer; from L3 on, all five are among its 10 most-raised genes (at L0, SLC4A1 and GYPA are not). The size-matched control has all five in its top 10 only at L9 and L11. At L0 it raised mostly other genes (LCN2, HBG2, RNASE1). Five of these genes (AHSP, ALAS2, HBM, SLC4A1, GYPA) are among the 15 genes highest in g_late − g_early itself (AHSP, ALAS2, HBM, LGALS3, SELENBP1, SLC4A1, MYL4, GYPA, KLF1, CTSE, HBZ, HBQ1, RNASE1, HEPACAM2, CD36).

**Candidates against the random-feature null.** Δs = mean over the 50 steered HSCs. ρ = maturity correlation in selection cells. p values are against the 20 random features of the same layer (smallest possible 0.048).

| Layer | Feature | ρ | Cells active | Δs α = 0 | Δs α = 2 | Δs α = 5 [95% CI] | α = 5 as % of span [95% CI] | Directional p (α = 0 / 2 / 5) | Size p (α = 5) |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1976 | +0.73 | 50/50 | −4.7e-5 | +5.1e-5 | +1.9e-4 [+1.1e-4, +2.7e-4] | +0.094 [+0.055, +0.134] | 0.38 / 0.38 / 0.43 | 0.57 |
| L0 | 1883 | +0.72 | 50/50 | +1.6e-5 | +1.6e-5 | +2.7e-4 [+1.3e-4, +4.3e-4] | +0.138 [+0.066, +0.216] | 0.81 / 0.62 / 0.29 | 0.38 |
| L0 | 2449 | −0.68 | 50/50 | −1.1e-6 | +2.8e-6 | +1.0e-4 [−1.0e-5, +2.2e-4] | +0.052 [−0.005, +0.110] | 0.38 / 0.33 / 0.62 | 0.62 |
| L3 | 1133 | −0.81 | 50/50 | +1.3e-5 | +6.1e-6 | +1.5e-4 [+8.5e-5, +2.2e-4] | +0.077 [+0.043, +0.112] | 0.38 / 0.52 / 0.81 | 0.29 |
| L3 | 980 | −0.72 | 50/50 | +1.0e-5 | −1.2e-5 | −8.2e-5 [−9.8e-5, −6.5e-5] | −0.041 [−0.050, −0.033] | 0.38 / 0.33 / 0.19 | 0.57 |
| L3 | 1029 | −0.72 | 50/50 | +1.4e-5 | −9.5e-6 | −1.9e-5 [−2.7e-5, −1.0e-5] | −0.010 [−0.014, −0.005] | 0.38 / 0.38 / 0.38 | 0.90 |
| L6 | 3965 | +0.77 | 50/50 | −3.4e-6 | +2.9e-6 | +1.4e-5 [+9.9e-6, +2.0e-5] | +0.007 [+0.005, +0.010] | 0.33 / 0.38 / 0.38 | 0.43 |
| L6 | 1976 | +0.70 | 41/50 | +2.0e-6 | −1.3e-6 | −8.3e-7 [−5.0e-6, +5.2e-6] | −0.0004 [−0.003, +0.003] | 0.81 / 0.81 / 0.76 | 1.00 |
| L6 | 646 | +0.69 | 36/50 | −1.2e-6 | +1.4e-6 | +7.0e-6 [+4.1e-6, +1.1e-5] | +0.004 [+0.002, +0.005] | 0.43 / 0.43 / 0.43 | 0.57 |
| **L9** | **2903** | **+0.84** | **42/50** | **−1.7e-3** | **+1.7e-3** | **+6.9e-3 [+3.8e-3, +1.0e-2]** | **+3.47 [+1.94, +5.18]** | **0.048 / 0.048 / 0.048** | **0.048** |
| L9 | 878 | −0.73 | 50/50 | −9.4e-5 | +4.0e-4 | +2.9e-3 [+1.9e-3, +3.9e-3] | +1.47 [+0.95, +1.95] | 0.81 / 0.95 / 1.00 | 0.048 |
| L9 | 3003 | +0.70 | 42/50 | +1.9e-6 | −1.6e-6 | −4.8e-6 [−6.6e-6, −3.2e-6] | −0.002 [−0.003, −0.002] | 0.57 / 0.57 / 0.57 | 0.86 |
| L11 | 2767 | −0.75 | 50/50 | −8.0e-6 | +8.1e-6 | +3.3e-5 [+2.6e-5, +4.1e-5] | +0.017 [+0.013, +0.021] | 0.76 / 0.76 / 0.76 | 0.57 |
| L11 | 4372 | +0.75 | 39/50 | −5.5e-6 | +5.5e-6 | +2.2e-5 [+1.7e-5, +2.9e-5] | +0.011 [+0.008, +0.014] | 0.33 / 0.33 / 0.33 | 0.67 |
| L11 | 2835 | −0.75 | 42/50 | +4.4e-5 | −4.4e-5 | −1.7e-4 [−2.2e-4, −1.3e-4] | −0.088 [−0.111, −0.067] | 0.24 / 0.24 / 0.24 | 0.29 |

- On the score set in advance, 14 of the 15 candidates do not move cells more than random features in the predicted direction (directional p 0.19–1.00 at every α). At α = 5 their effects are 0.0004% to 1.5% of the span in size.
- Most of their intervals exclude zero. That is not evidence of steering. The edit is deterministic, so it moves every cell a little, and random features move cells by similar amounts. The one exception in size is L9 F878 (size p 0.048; below).
- The response is close to linear in α − 1 for 5 of the 15 candidates (L9 F2903, the three L11 candidates and L0 F1976): Δs(α = 0) / Δs(α = 2) is −0.91 to −1.01 and Δs(α = 5) / Δs(α = 2) is 3.63 to 4.10 (L9 F2903: −1.00 and 4.02). For the other ten (at L0, L3, L6 and L9) the ratios are irregular (α = 5 to α = 2 from 0.62 to 36.4). The three α values are still close to one test repeated, because they use the same cells and features and, within each layer, rank the 23 features in almost the same order.
- The null features were matched on activity in selection HSCs. In the steered HSCs, the candidates' edits were 0.03 to 4.0 times the null mean (L9 F2903: 2.8 times; L9 F3003: 0.03 times). The per-unit-of-edit comparison below controls for this.

**The one hit: L9 feature 2903.**

- *What it is* (post hoc). Active at ≥ 1 gene position in 97% of selection HSCs, but at only 11% of gene positions in HSCs, 33% in CMPs, 99% in EryP and 97% in erythrocytes (32 selection cells, 8 per stage, all from donor TSP14; these shares are not a general figure for HSCs: in the steered HSCs it is active in 42 of 50 cells, at a mean of 382 gene positions per cell over all 50, of at most 2,046). It fires on 7,657 different genes, each a tiny share of its total activity (largest: ANK1, TFRC, BLVRB; 0.08–0.09% each). So it marks the cell's erythroid state, not a gene. ρ = +0.84 (+0.87, +0.80 and +0.84 in the three selection donors).
- *Effect.* α = 5: Δs = +0.0069 [+0.0038, +0.0103], 3.5% [1.9%, 5.2%] of the span. α = 2: +0.86% [+0.47%, +1.30%]. α = 0 (removed): −0.86% [−1.31%, −0.47%]. Gap share at α = 5: 2.6% [1.4%, 3.9%] of the line from g_early to g_late. Both read-outs agree in sign at every α.
- *Against the null.* At every α it beats all 20 random L9 features, in the predicted direction and in size (p = 0.048, the floor). z against the L9 null = +6.4 at α = 5 (this assumes a roughly normal null; the 20 null values span −0.0029 to +0.0019). Per unit of edit size (post hoc) it also beats all 20 (p = 0.048), so its larger edit does not explain the result.
- *Correction for 15 candidates.* p = 0.048 alone does not survive: about 0.7 of 15 candidates would reach the floor by chance (15 / 21). Post hoc, against all 100 null features pooled over the five layers, none is as large (p = 1/101 = 0.0099; × 15 candidates = 0.15), and its effect is 2.4 times the largest of them. So it is a clear outlier, but the formal p is limited by the size of the null.
- *Cells.* It moves 42 of 50 HSCs in the predicted direction. These are exactly the 42 cells where it is active; in the other 8 the edit is zero. Donor split (post hoc): TSP2 +0.0066 [+0.0035, +0.0102] (n = 47); TSP27 +0.0108 [+0.0039, +0.0175] (n = 3, too few to count as a replication).
- *Size in context.* The size-matched control moves cells 3.9% of the span with an edit of 1.43 per gene position, less than half of feature 2903's edit. Feature 2903 moves them 3.5% with an edit of 3.44. Per unit of edit (post hoc) it does 37% of what the control direction does (Δs 0.0020 against 0.0054 per unit). The full erythrocyte-minus-HSC difference at L9 has size 24.9 per gene position.
- *Genes.* Its most-raised genes at α = 5 are AHSP, ALAS2, HBM, GYPA, HBG1, SLC4A1, HBG2 and GYPB. 9 of its top 10 are on the 14-gene erythroid list, against 0.2 on average for the 20 random L9 features (post hoc). All six of AHSP, ALAS2, HBM, GYPA, HBG1 and SLC4A1 are among the 10 genes most raised by the size-matched control at L9. Five of them (not HBG1) are among the 15 genes highest in g_late − g_early. No small set of genes defines the score, which is a cosine over all 20,275 logits. But these are the genes that any push along this axis raises, so they restate the score rather than add biology.

**L9 feature 878: a large effect whose direction depends on the read-out.**

- An HSC-high feature (ρ = −0.73; genes with the largest share of its activity: RUNX1, ETV6, FKBP5, RNF220, CDK6). Active in 50 of 50 HSCs, at about 500 gene positions per cell.
- On the score set in advance, amplifying it moves HSCs *toward* the erythrocyte signature (α = 5: +0.0029 [+0.0019, +0.0039], +1.5% of the span). That is the wrong direction (directional p = 1.00). In size it beats all 20 random L9 features (size p = 0.048), is 4.5 times their mean |Δs| (1.47% against 0.33% of the span), and equals the largest of all 100 null effects (ratio 1.00).
- On the gap share (post hoc), it moves cells toward HSC, as predicted (α = 5: −3.3% [−3.8%, −2.9%]; beats all 20 null features at every α). The cosine between its logit change and g_late − g_early is −0.16.
- So the edit moves the logits back along the HSC-to-erythrocyte line, but it also changes the logit vector in other ways that raise the cosine score. Its direction depends on the read-out. It cannot be called a maturity switch either way.

**Floor hits on the post-hoc read-outs.** On the score set in advance, only L9 F2903 reaches the floor. On the post-hoc read-outs (candidates_table.csv), more candidates do:

- Gap share: L9 F2903 and L9 F878 at every α; L6 F3965 at α = 2 and 5 (its gap share is tiny, +0.0002 at α = 5).
- Gap share per unit of edit, α = 5: L9 F2903, L6 F646 and L11 F2835.
- Δs per unit of edit, α = 5: L9 F2903 only.
- With 15 candidates, 3 strengths and 2 read-outs, a few results at the floor are expected by chance. "14 of 15" refers to the score set in advance.

**Layer-level and pooled tests.**

| Layer | Candidates vs all 1,771 sets of 3: directional p (α = 0 / 2 / 5) | Size p (α = 0 / 2 / 5) | Spearman(ρ_f, Δs_f) over 23 features (α = 0 / 2 / 5) |
|---|---|---|---|
| L0 | 0.64 / 0.55 / 0.47 | 0.90 / 0.87 / 0.62 | −0.14 / +0.31 / +0.33 |
| L3 | 0.27 / 0.33 / 0.45 | 0.86 / 0.96 / 0.65 | −0.50 / +0.40 / +0.34 |
| L6 | 0.27 / 0.24 / 0.16 | 0.58 / 0.68 / 0.74 | +0.15 / −0.15 / −0.15 |
| L9 | 0.09 / 0.12 / 0.14 | 0.08 / 0.05 / 0.02 | −0.14 / +0.10 / +0.08 |
| L11 | 0.47 / 0.46 / 0.46 | 0.61 / 0.61 / 0.61 | −0.29 / +0.29 / +0.29 |

- At no layer does the candidate set push in the predicted direction more than random sets of 3. At L9 the candidates are larger than random sets (size p 0.02 at α = 5), but they point in different directions (2903 as predicted, 878 against).
- Pooled over layers, the test named in `design.json`: the candidates' summed directional rank is no higher than chance (p 0.29 / 0.36 / 0.24 at α = 0 / 2 / 5).
- Pooled over layers, mean within-layer Spearman(ρ_f, Δs_f) over 23 features per layer (predicted sign negative for α = 0, positive for α = 2 and 5): −0.18 / +0.19 / +0.18; one-sided permutation p 0.027 / 0.022 / 0.031; two-sided 0.055 / 0.045 / 0.063. The Spearman over all 115 features pooled is not reported.
- Most of the 115 features in this test are null features with |ρ| up to 0.66. So the trend says "features that track maturity push a little more along the axis". It does not say that the chosen candidates are switches. It is weak, and the three α values are close to one test repeated (same cells and features; within each layer they rank the features in almost the same order).

**Genes of the other candidates.** The candidates that do not beat the null raise mixed, unrelated genes (for example L3 F1133: RYR2, GRM5, PLCB1, RGS7, DOCK4; L11 F2767: CNBP, DARS1, MAGED1). L0 F1883 has 5 erythroid genes in its top 10 (HBA1, HBG2, AHSP, HBZ, HBA2), but its Δs does not beat the null (directional p 0.29 at α = 5). So erythroid genes in a list are not enough to call a feature a switch.

## Verification

No separate verification by a second agent is recorded for this analysis. The key numbers were computed a second way with separate code (`scripts/v3_steering_crosscheck.py`, output `outputs/v3_steering/crosscheck.json`; all pass), in addition to the forward-pass checks above (`scripts/v3_steering_verify.py`, `outputs/v3_steering/verify/verify.json`):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Δs of all 2,750 candidate and control rows | summary code | recomputed from the stored float32 logits | largest difference 4.4 × 10⁻¹⁶ |
| Δs of all 17,750 rows | summary code | recomputed from the stored dot products with the signatures | largest difference 1.4 × 10⁻¹⁴ |
| Bootstrap intervals (45 candidate × α rows) | own resampling loop | `scipy.stats.bootstrap` (percentile) | means identical; interval ends within 2.3% of the interval width |
| Candidate choice | selection stage | separate code from the stored clean files | same alive counts and the same 3 candidates at every layer; ρ equal within 10⁻¹⁵ |
| Held-out AUROC (EryP vs HSC) | own code | scikit-learn | 0.99133 both |
| 45 empirical p values | summary code | recomputed | identical (difference ≤ 1.1 × 10⁻¹⁶) |
| Δs, speed-up path | partial forward from block l | full forward with `hooks_v2.run_with_edits` (75 rows) | largest difference 2.2 × 10⁻⁹ |
| Δs, device | MPS | CPU (cell 332, L9 F2903, α = 5) | difference 7.0 × 10⁻¹⁰ |
| 42 cells moved by L9 F2903 | summary | per-cell files (`steer/cell_*.npz`): cells with Δs > 0 at α = 5 and cells with the feature active | 42 and 42, the same cells; mean 0.006875 |

## Limits

- **The SAEs fit these cells poorly** (FVE 0.34–0.59 on the steered HSCs, against 0.70–0.91 on held-out K562 tokens at the same layers). No SAE was trained on bone-marrow cells. So the negative result is "none of the 15 tested features of these K562-trained SAEs acts as a maturity switch beyond what a feature that marks erythroid cells would do", not "no feature is a switch" and not "the model has no such direction". Of the 2,481 to 4,912 alive features per layer, only the 3 candidates were tested as switches; 20 more per layer were steered only as the null. The positive control shows that a direction that moves cells does exist.
- **One donor in practice.** 47 of the 50 steered HSCs come from TSP2. The held-out axis check uses TSP2 and TSP27, which have no erythrocytes, so EryP stands in for the late stage there.
- **20 null features per layer** cap every per-feature p at 0.048. With 15 candidates, one feature at the floor is expected by chance. The case for L9 F2903 rests on its size against all 100 null features (2.4 times the largest), which is a post-hoc comparison.
- **Not tested:** other lineages (myeloid, lymphoid); steering erythroid cells back toward HSC; combinations of features; α above 5; layers other than 0, 3, 6, 9 and 11; features other than the 3 candidates per layer; read-outs other than the position-mean logits (for example hidden-state probes); MaxToki-1B. The deployed "pseudotime" axis was not used, because it separates two lineages rather than ordering maturity (main text).
- **The read-out is the model's position-mean next-gene logits.** It stands in for cell state; it is not a measured state. Stages are annotation labels, not time.
- **Length.** Selection erythrocytes have shorter sequences (median 1,796 tokens; 2,048 in the other groups), and within HSCs and CMPs the score correlates with length (−0.42, −0.41). So the late signature carries some length information. This cannot cause a steering effect, because edits do not change tokens, and the steered HSCs nearly all have 2,048 tokens.
- Other jobs shared the machine, so wall times are not clean timings (clean stage 530 s for 442 cells; steering 5,943 s for 50 cells in 16 chunks; verify stage 102 s). Free memory never fell below 5.8 GB.

## Files

- Scripts (`projects/maxtoki/runs/exhaustive-mapping-217M/scripts/`): `v3_steering.py` (stages design, prep, clean, select, steer), `v3_steering_verify.py`, `v3_steering_summarize.py`, `v3_steering_crosscheck.py`, `v3_steering_describe.py`, `v3_steering_extra.py`. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`. SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/`.
- Outputs (`projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_steering/`):
  - design and inputs: `design.json` (rules, written before any forward pass), `prep.json`, `cells.npz` (rows, donors, stages, tokens, counts sha256), `clean/cell_*.npz` (442 files);
  - selection: `selection.json` (axis, SAE fit, candidates, null features, norms of the control vectors), `signatures.npz` (g_early, g_late and the control vectors);
  - steering: `steer/cell_*.npz` (50 files; per row Δs, gap share, edit size, active positions, logit changes);
  - summaries: `summary.json` (all main numbers), `candidates_table.csv`, `null_table.csv`, `controls_table.csv`;
  - checks and post-hoc analyses: `verify/verify.json`, `crosscheck.json`, `describe.json`, `extra.json`;
  - provenance: `run_config.json` (device, versions, seeds, cell IDs, feature IDs, input sha256, code sha256, per-chunk wall time, free memory and encoding-check counts). `comparison.json` holds the deployed per-layer values for reference.

