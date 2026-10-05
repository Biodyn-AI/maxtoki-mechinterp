# V3-6: attention-GRN, MaxToki-217M, K562, with correct input encoding

Date: 2026-10-02. Model run: MPS, float32, eager attention. Everything else: CPU.
Scripts (new): `scripts/v3_attention_k562.py` (prep, extraction, edges),
`scripts/v3_attention_k562_eval.py` (the v2b evaluation), `scripts/v3_attention_k562_verify.py`
(second-way checks and `run_config.json`). Outputs: `outputs/v3_attention_k562/`.
No existing file was changed.

Words used below:
- **AUROC**: how well a score ranks true targets above non-targets. 0.5 = chance, 1 = perfect.
- **Endpoint A (knockdown)**: for one knocked-down gene, which genes change in the same CRISPR screen.
  Summary = mean of the per-perturbation AUROCs.
- **Endpoint B (TRRUST)**: for one transcription factor (TF), which genes the TRRUST database lists as
  its targets. Summary = one pooled AUROC over all (TF, gene) pairs.
- **Degree null (Curveball)**: random TRRUST networks that keep every TF's number of targets and every
  gene's number of listing TFs. z = (observed − null mean) / null SD. p is one-sided. q = BH-adjusted p.
- **Order share F[P,T]**: the share of cells (holding both genes) in which gene T is ranked above gene P.
  It uses no model weights. MaxToki is a causal decoder, so attn[P,T] is forced to 0 when T comes after P.
- **Edge variants**: forward attn[P,T] (the deployed one, P = query); transpose attn[T,P]; symmetric
  mean and max of the two; rank-conditioned (forward divided by F, i.e. mask zeros removed).

---

## 0. Results first

1. **Correct inputs, same cells.** The 2,000 K562 control cells are exactly the deployed ones. All
   2,000 token sequences are the counts / gene-median order (checked at tokenisation and again right
   before each forward pass; 0 order violations). The deployed tokens fail the same check in
   2,000 of 2,000 cells. The model predicts the next gene at 3.09 nats per gene token
   [95% CI 3.08, 3.10; cell bootstrap, n = 2,000], the same as the V3-0 baseline (3.08 [2.96, 3.23], 5 cells).
2. **The gene set changes a lot.** The same top-1,500-variance rule on a single log keeps only 627 of
   the deployed 1,500 genes (Jaccard 0.26). The deployed set was picked on doubly-logged values. The
   new genes are more widely expressed (mean dropout 0.19 vs 0.37). TRRUST now has 16 evaluated TFs and
   107 positive pairs (deployed: 8 and 45).
3. **The edge itself changes a lot.** On the deployed genes, the new and the old layer-8 edges agree at
   only Spearman 0.35 over all gene pairs. The order share F agrees at 0.89.
4. **Knockdown: the verdict holds and is stronger.** 194 perturbations. Every attention variant is at
   chance: forward 0.505 [0.491, 0.520], symmetric mean 0.502 [0.488, 0.517]. Gene variance scores 0.690
   [0.674, 0.708]. Gap variance − forward +0.185 [+0.162, +0.208] (perturbation bootstrap). Co-expression
   (0.556 [0.540, 0.573]) also beats every attention variant. The forward edge ties F
   (F − forward +0.009 [−0.004, +0.020]).
5. **TRRUST: attention rises but still does not pass the degree null after correction.** Forward 0.548
   [0.499, 0.594], z +1.72 (p 0.038, q 0.082 over 6 scores). Symmetric mean 0.571 [0.521, 0.617], z +1.76
   (p 0.041, q 0.082). Deployed K562 was 0.483 (z +0.47) and 0.513 (z +0.27). Co-expression: 0.588, z +1.55.
   The forward z hangs on one TF: without MYC it drops to +0.12.
6. **Gene-level baselines still lead on TRRUST.** The 3-feature gene model scores 0.723 [0.644, 0.814]
   and beats every attention variant (gaps +0.15 to +0.20, all CIs above 0). Variance alone (0.642) beats
   forward (+0.094 [+0.016, +0.187]) and ties symmetric mean (+0.071 [−0.017, +0.170]).
7. **After removing gene-level features, nothing passes.** Residualised AUROCs 0.48–0.55; null z
   +0.5 to +1.7; every BH q ≥ 0.17 (OLS and HGB).
8. **K562 does not reproduce the RPE1 positive.** In RPE1 (correct inputs, unchanged), symmetric
   attention beat the degree null (z +3.19 / +3.02). In correctly encoded K562 it does not
   (z +1.76 / +1.11). In a joint BH over both cell lines (12 tests), the RPE1 symmetric tests keep
   q 0.018; no K562 test goes below q 0.069.
9. **Secondary (not pre-specified): the gene set decides the TRRUST answer.** Read on the deployed
   genes (8 TFs, same null networks as v2b), the corrected attention beats the degree null: forward
   0.575, z +2.55 (p 0.008); symmetric mean 0.591, z +2.38 (p 0.010). Co-expression there is at z −0.66.
   On the new single-log gene set this does not hold. Knockdown does not move on either gene set.
10. **The edge now behaves like RPE1's.** The share of the log edge explained by F is R² 0.41 (deployed
    K562 0.26; RPE1 0.38). Within a regulator's row, edge and F agree at Spearman 0.76 (0.48; 0.77).

All named checks passed (section 2). Key numbers were re-derived a second way (section 2, rows V1–V7).

---

## 1. What was done

1. **Cells.** K562 non-targeting cells of `replogle_concat.h5ad`, `default_rng(42).choice(…, 2000)`,
   sorted. The draw equals `outputs/phase0/control_cells.csv` row for row.
2. **Counts.** `inputs_v3.CountReader("k562")` with `counts_accept_unit_one`: expm1(X) checked to be
   whole-number multiples of the cell's smallest value, then divided by it and rounded. All 2,000 rows
   pass (largest distance from a whole multiple 3.7 × 10⁻⁴; multiples up to 1,110).
3. **Gene set (single log).** log1p(counts / row sum × 10⁴), row sum over the file's 6,546 genes. This is
   the deployed rule and exactly what the RPE1 run did with its raw counts. Top 1,500 by variance among
   genes in the MaxToki vocabulary (6,332 eligible).
4. **Gene features and co-expression** on the same single-log values: mean, variance, dropout;
   |Spearman| over the 2,000 cells (deployed formula).
5. **Tokens.** `inputs_v3.tokenize_counts` (counts / total × 10⁴ / Geneformer median, ranked, first
   2,046 genes, `<bos>`/`<eos>`), with `check_encoding` on every cell.
6. **Forward passes.** `LlamaForCausalLM`, eager attention, MPS float32. A forward hook on each layer's
   `self_attn` took the attention weights (= `output_attentions[l]`), averaged the 8 heads, and added the
   gene × gene block to a running sum. Blocks of 25 cells; `assert_encoding_batch` on every block before
   its forward passes; memory checked before every block (stop below 3 GB). The sum ran over the
   **union** of the new and the deployed gene sets (2,373 genes), so the new edges can also be read on
   the deployed genes (section 7). All 11 layers were kept; layer 8 is the pre-specified one.
7. **Edges.** E[P,T] = summed attention / number of cells holding both genes (deployed definition,
   masked zeros included). Pair counts and order counts C (cells where T comes before P) were counted in
   the same loop.
8. **Knockdown labels** with a single log: the v2 rule (Welch t, BH over the 1,500 genes, |Δ| ≥ 0.5 and
   q < 0.05, ≥ 3 positives and negatives; same 5,000 controls, seed 42), on expm1(X) scaled to 10⁴ over
   the 1,500 genes, then log1p. This is the RPE1 path. A sensitivity version uses the stored
   log1p(CP10k) values directly (section 5b).
9. **Evaluation** exactly as v2b (`V2B_ATTENTION_REPORT.md`): six scores, both endpoints, perturbation
   and TF bootstraps (2,000 reps), Curveball null (v2_06 sampler unchanged, 1,000 draws × 5,000 trades),
   3-feature gene model, residualisation (OLS on gene features ± F; HGB check), co-expression
   comparison. Same seeds as v2b's K562 run (20261001 + 100 / 500 / 600 / 800; KFold 42).
10. Extra: leave-one-TF-out; per-layer supplement; the encoding-only comparison on the deployed genes.

Wall time: extraction 3 chunks (105 s, 426 s, 417 s; 0.44 s per cell). Peak memory footprint 5.4 GB
(first chunk, `/usr/bin/time -l`). Available memory never fell below 6.3 GB. Another project's job was
running on the machine at the same time.

## 2. Checks (all passed)

| Check | Result |
|---|---|
| Same cells as deployed | 2,000 / 2,000 rows equal `phase0/control_cells.csv` |
| Counts are counts | 2,000 / 2,000 rows whole-number multiples; implied cell total / `UMI_count` median 0.987 (0.956–0.994) |
| Deployed rule on stored X rebuilds the deployed gene set | yes, exactly |
| Gene set stable to arithmetic | float32 route gives the same 1,500 genes; independent float expm1 route (no rounding) gives the same set, features within 4 × 10⁻⁹ |
| Encoding check at tokenisation | 2,000 / 2,000 pass; 0 order violations; 240 positions reordered inside exact ties |
| Encoding check before forward passes | 80 blocks, 2,000 / 2,000 cells, all pass |
| Deployed tokens fail the check | 2,000 / 2,000 |
| Deployed tokens rebuild the deployed pair counts | yes, exactly (so the deployed comparison is like-for-like) |
| Causal mask | attention above the diagonal = 0.0 in every cell (layer 8) |
| Zero pattern | E > 0 exactly where C > 0, all 11 layers, 0 mismatches in either direction (5.63 million pairs) |
| Counts consistent | C + Cᵀ = pair count off the diagonal; pair counts = PᵀP from the saved tokens |
| Curveball null works | 1,000 draws; 0 break a row or column sum or fill a forbidden cell; 0 equal the real network; mean Jaccard distance 0.86 (min 0.79); variance AUROC unchanged in every draw (max change 0.0) |
| AUROC self-test | constant score 0.5; tied scores match sklearn |
| **V1** edges, second code path | 6 random gene pairs (24 cells) re-run through the deployed extractor (`output_attentions=True`, CPU copy): rebuilt E equals saved E at all 11 layers, max relative difference 2.1 × 10⁻⁷. Attention rows sum to 1 (max deviation 4.8 × 10⁻⁷). One cell CPU vs MPS: max difference 1.8 × 10⁻⁶ |
| **V2** counts, own code | pair counts and C rebuilt from the saved tokens with a position matrix: equal, both gene sets |
| **V3** labels, own code | own block reader, own Welch t, statsmodels BH, all 446 candidates: same 194 kept, 0 label disagreements, 8,714 positives |
| **V4** AUROCs | sklearn equals the main values to ≤ 1.1 × 10⁻¹⁶ for every score, both endpoints |
| **V5** bootstraps, own seed | variance − forward (knockdown) [+0.162, +0.209]; TRRUST CIs from an explicit pair-copy TF bootstrap within 0.007 of the main ones |
| **V6** second null sampler | checkerboard swaps (own code, 1,000 draws × 2,140 swaps): z forward +1.67, sym mean +1.76, co-expression +1.60; all within 0.06 of Curveball |
| **V7** residuals, own code | sklearn LinearRegression, different random folds: AUROCs within 0.0003, z within 0.01 |

## 3. What changed in the inputs

| | Deployed / v2b K562 | v3 K562 |
|---|---|---|
| Model input order | log1p(CP10k) / median (wrong) | counts / median (correct) |
| Agreement with the correct order (2,000 cells) | Spearman 0.840 (min 0.714); top-200 overlap 0.563 (min 0.47); kept-set overlap 0.917; same position 0.001 | 1.0 by construction |
| Gene set | top-1,500 variance of a double log | top-1,500 variance of a single log; 627 shared (Jaccard 0.26) |
| Mean dropout of the gene set | 0.37 | 0.19 |
| Median cells per gene pair | 326 | 649 |
| TRRUST TFs / positive pairs | 8 / 45 | 16 / 107 (4 TFs shared: NFE2L2, PTTG1, BRCA1, CEBPB) |
| Knockdown perturbations / positive pairs | 174 / 9,741 (double-log labels) | 194 / 8,714 (single-log labels) |

Why the gene set moves: the second log squeezes high values, so it ranked genes by a different
variance. Single-log and double-log variance agree at only Spearman 0.48 on the deployed genes. The
variance gap at the cut is small (rank 1,500 vs 1,501 differ by 0.04%), but the set is stable to
arithmetic (section 2).

## 4. How much of the edge is expression order (layer 8)

| | v3 K562 | v2b K562 (wrong input) | RPE1 (correct, v2b) |
|---|---|---|---|
| Spearman(E, F), all pairs | 0.60 | 0.41 | 0.61 |
| Median within-row Spearman(E, F) | 0.76 | 0.48 | 0.77 |
| R² of log E on log F | 0.41 | 0.26 | 0.38 |
| Median within-row Spearman of F with a target-only order score | 0.88 | 0.78 | 0.86 |
| Median within-row Spearman(rank-conditioned, F) | 0.37 | −0.14 | 0.46 |
| Pairs with n_pair > 0 but C = 0 | 0.08% | 1.4% | 0.18% |

With correct input, the K562 edge depends on expression order about as much as the RPE1 edge does.
About 40% of the variance of the log edge is the order share. The rank-conditioned edge is not
order-free: within a row it still follows F (0.37).

## 5. Endpoint A — knockdown

194 perturbations, 8,714 positive pairs (median 17 per perturbation). 95% CI = percentile bootstrap
over perturbations, 2,000 reps. Source: `knockdown/knockdown_summary.json`.

**Mean per-perturbation AUROC [95% CI]**

| Score | v3 K562 (correct) | v2b K562 (wrong input) | RPE1 (correct, v2b) |
|---|---|---|---|
| forward attn[P,T] | 0.505 [0.491, 0.520] | 0.512 [0.497, 0.527] | 0.603 [0.594, 0.612] |
| transpose | 0.509 [0.495, 0.524] | 0.484 [0.470, 0.498] | 0.487 [0.477, 0.498] |
| symmetric mean | 0.502 [0.488, 0.517] | 0.500 [0.484, 0.516] | 0.583 [0.572, 0.593] |
| symmetric max | 0.500 [0.486, 0.514] | 0.497 [0.481, 0.512] | 0.580 [0.571, 0.590] |
| rank-conditioned | 0.499 [0.484, 0.514] | 0.496 [0.482, 0.510] | 0.560 [0.551, 0.570] |
| order share F (no model) | 0.513 [0.496, 0.531] | 0.517 [0.501, 0.534] | 0.648 [0.637, 0.658] |
| \|Spearman\| co-expression | 0.556 [0.540, 0.573] | 0.510 [0.495, 0.525] | 0.622 [0.606, 0.636] |
| target variance (single log) | **0.690 [0.674, 0.708]** | 0.614 [0.597, 0.631] (stored X) | **0.766 [0.757, 0.776]** |

The v2b K562 column used doubly-logged labels and the deployed gene set, so its rows are not the same
test. They are shown for direction only.

Differences in v3 K562 [95% CI]:
- Variance − score: +0.177 [+0.150, +0.203] (F) to +0.191 [+0.168, +0.213] (rank-conditioned).
  Forward: +0.185 [+0.162, +0.208]. Variance beats forward in 173 of 194 perturbations. Paired Wilcoxon
  (zsplit, two-sided), BH over the 6 scores: every q ≤ 1 × 10⁻²⁴.
- Score − forward: transpose +0.005 [−0.017, +0.026]; symmetric mean −0.002 [−0.014, +0.010];
  symmetric max −0.005 [−0.019, +0.010]; rank-conditioned −0.006 [−0.016, +0.004]; F +0.009
  [−0.004, +0.020]. All ties.
- Co-expression − forward: +0.051 [+0.028, +0.073]. Co-expression beats every attention variant
  (+0.046 to +0.057) and F (+0.042 [+0.018, +0.066]).

What this means: with correct input, layer-8 attention in K562 carries no knockdown signal at all.
Gene variance is far ahead. This is the deployed verdict, now with a larger gap (part of the larger gap
is the better single-log variance baseline). It differs from RPE1, where attention is above chance
(0.603) but still below F, co-expression and variance.

### 5b. Sensitivity: labels from the stored log1p(CP10k) values

The |Δ| ≥ 0.5 rule depends on the scale. Using the stored values directly (normalised over the whole
transcriptome, no rescale over the 1,500 genes) keeps only 67 perturbations (720 positive pairs).
Forward 0.550 [0.517, 0.580], F 0.611 [0.568, 0.651], variance 0.695 [0.659, 0.730]. Variance − forward
+0.145 [+0.089, +0.200]; F − forward +0.061 [+0.034, +0.087]. Same verdict: variance wins; attention is
no better than F. I chose the RPE1 rule as primary before looking, so the two cell lines use the same
label rule.

## 6. Endpoint B — TRRUST

16 TFs (HDAC1, JUN, NFE2L2, PTTG1, EZH2, MYC, TFDP1, CHD8, YY1, CTCF, BRCA1, DNMT1, JUND, CEBPB, XBP1,
ATF4), 107 positive and 23,877 negative pairs. 95% CI = percentile bootstrap over TFs, 2,000 reps.
Source: `trrust/trrust_summary.json`, `trrust/gene_model_gaps.json`.

**Pooled AUROC [95% CI]; degree null z (p; q)**

| Score | v3 K562 (correct) | v2b K562 (wrong input) | RPE1 (correct, v2b) |
|---|---|---|---|
| forward | 0.548 [0.499, 0.594]; z +1.72 (p 0.038; q 0.082) | 0.483; z +0.47 | 0.605; z +1.82 |
| transpose | 0.526 [0.446, 0.593]; z −0.09 (p 0.55; q 0.55) | 0.548; z +1.21 | 0.546; z +2.32 |
| symmetric mean | 0.571 [0.521, 0.617]; z +1.76 (p 0.041; q 0.082) | 0.513; z +0.27 | **0.638; z +3.19** |
| symmetric max | 0.538 [0.505, 0.574]; z +1.11 (p 0.14; q 0.17) | 0.535; z +0.62 | **0.647; z +3.02** |
| rank-conditioned | 0.532 [0.477, 0.594]; z +1.68 (p 0.038; q 0.082) | 0.515; z −0.41 | 0.603; z +2.29 |
| order share F | 0.537 [0.455, 0.611]; z +1.08 (p 0.14; q 0.17) | 0.454; z +0.73 | 0.552; z +0.04 |
| \|Spearman\| co-expression (reference) | 0.588 [0.533, 0.691]; z +1.55 (p 0.064) | 0.543; z −0.48 | 0.596; z +3.04 |
| target variance (single log) | 0.642 [0.579, 0.714] | 0.638 | 0.686 |
| 3-feature gene model | **0.723 [0.644, 0.814]** | 0.745 | 0.749 |

q = BH over the 6 v3 K562 scores. Joint BH over 12 tests (these 6 + the 6 RPE1 tests): K562 forward,
symmetric mean and rank-conditioned q 0.069; RPE1 symmetric mean and max q 0.018.

Gaps in v3 K562 [95% CI, TF bootstrap]:
- Gene model − score: forward +0.175 [+0.084, +0.287]; symmetric mean +0.152 [+0.050, +0.269];
  symmetric max +0.186 [+0.107, +0.280]; transpose +0.197 [+0.080, +0.341]; rank-conditioned +0.192
  [+0.108, +0.285]; F +0.186 [+0.054, +0.346]; co-expression +0.135 [+0.071, +0.214]. With a random
  TF-to-fold map the gene model is 0.691 instead of 0.723, so read these gaps as "about 0.15, lower
  bound above 0".
- Variance − score: forward +0.094 [+0.016, +0.187]; symmetric mean +0.071 [−0.017, +0.170] (tie);
  symmetric max +0.104 [+0.028, +0.184]; transpose +0.116 [+0.005, +0.238]; rank-conditioned +0.111
  [+0.011, +0.198]; F +0.105 [+0.028, +0.210].
- Symmetric − forward: mean +0.023 [−0.034, +0.076]; max −0.010 [−0.058, +0.040]. Ties.
- Attention − co-expression: forward −0.040 [−0.161, +0.037]; symmetric mean −0.017 [−0.130, +0.060].

### 6a. Leave one TF out

Source: `trrust/leave_one_tf_out.json`. Raw scores, same null networks.
- Forward: z from +0.12 (MYC dropped) to +2.19. The forward signal depends on one TF.
- Symmetric mean: z from +1.11 (MYC dropped) to +2.25.
- Co-expression: z from +0.92 to +1.87.

### 6b. Residualised scores

Main model (pre-stated, as v2b): OLS on mean, variance and dropout of both genes (single-log features)
plus F, 5-fold cross-fitting over all 1,500 × 1,499 pairs. For F itself: gene features only. No pair
has n_pair = 0, so the n_pair = 0 indicator was dropped (it would be a constant). Check: HGB on the same
features. CI = TF bootstrap with fitted models fixed. Source: `resid/resid_summary.json`.

| Score | OLS residual [95% CI]; null mean; z; q | HGB residual [95% CI]; z; q |
|---|---|---|
| forward | 0.487 [0.408, 0.579]; 0.467; +1.08; 0.18 | 0.509 [0.457, 0.578]; +1.67; 0.17 |
| transpose | 0.545 [0.457, 0.614]; 0.527; +0.91; 0.18 | 0.533 [0.439, 0.604]; +0.48; 0.31 |
| symmetric mean | 0.538 [0.497, 0.591]; 0.505; +1.51; 0.18 | 0.543 [0.501, 0.591]; +1.39; 0.18 |
| symmetric max | 0.526 [0.487, 0.575]; 0.502; +1.07; 0.18 | 0.532 [0.494, 0.577]; +1.12; 0.21 |
| rank-conditioned | 0.481 [0.402, 0.588]; 0.454; +1.42; 0.18 | 0.499 [0.444, 0.580]; +1.52; 0.17 |
| order share F | 0.538 [0.434, 0.630]; 0.527; +1.05; 0.18 | 0.525 [0.455, 0.590]; +0.93; 0.21 |
| co-expression (reference) | 0.505 [0.432, 0.637]; 0.457; +1.91 (p 0.030) | — |

What this means: after removing gene-level features, no attention variant passes the degree null after
BH (lowest q 0.17), by either model. Every interval includes 0.5 except the HGB symmetric mean, whose
lower end is 0.501. In RPE1 the transpose and symmetric residuals did pass (z +2.75 to +3.10, q 0.024).
That does not replicate in K562.

### 6c. Co-expression comparison

Source: `coexpr/coexpr_comparison.json`.
- Over all ordered gene pairs, attention and |Spearman| co-expression barely relate: Spearman 0.04
  (forward) to 0.06 (symmetric mean). RPE1 had 0.10.
- Symmetric mean after removing co-expression: 0.540 (linear fit, z +1.05) and 0.539 (50 co-expression
  bins, z +1.08).
- Co-expression after removing symmetric mean: 0.584 (z +1.41) and 0.584 (z +1.48).
- Null test of "symmetric mean − co-expression": z 0.00. Forward: z −0.18.
- On knockdown, co-expression beats every attention variant (+0.046 to +0.057; all CIs above 0).

So in K562 the two are separate signals of similar, weak size on TRRUST. Neither clearly passes the
degree null. On knockdown, co-expression is ahead.

## 7. Encoding-only comparison on the deployed genes (secondary, not pre-specified)

Same 1,500 deployed genes, same v2 labels, same deployed gene features and co-expression, same 8
TRRUST TFs and the same 1,000 v2b null networks. Only the attention (and so F) changes. Paired
differences use the same bootstrap draws. Source: `oldgenes/deployed_genes_comparison.json`,
`oldgenes/deployed_genes_coexpr_single_log.json`. Checks: the old values equal v2b exactly
(AUROC and z differences 0.0).

Knockdown (174 perturbations, v2 labels): forward 0.512 → 0.517, change +0.005 [−0.011, +0.022];
symmetric mean 0.500 → 0.511, +0.011 [−0.009, +0.032]; F 0.517 → 0.529, +0.012 [+0.003, +0.020].
With single-log labels on these genes (182 perturbations): forward 0.517 → 0.523, +0.006
[−0.009, +0.021]. **The encoding fix does not change the knockdown result.**

TRRUST (8 TFs):

| Score | new AUROC [95% CI]; z (p) | old (v2b) AUROC; z | change [95% CI] |
|---|---|---|---|
| forward | 0.575 [0.485, 0.691]; +2.55 (0.008) | 0.483; +0.47 | +0.092 [−0.006, +0.253] |
| transpose | 0.487 [0.384, 0.561]; +0.26 (0.40) | 0.548; +1.21 | −0.062 [−0.157, +0.021] |
| symmetric mean | 0.591 [0.549, 0.638]; +2.38 (0.010) | 0.513; +0.27 | +0.078 [−0.023, +0.216] |
| symmetric max | 0.609 [0.558, 0.662]; +2.07 (0.017) | 0.535; +0.62 | +0.075 [−0.018, +0.181] |
| order share F | 0.535 [0.445, 0.658]; +1.77 (0.047) | 0.454; +0.73 | +0.081 [+0.044, +0.116] |
| co-expression, single log | 0.542 [0.453, 0.637]; −0.66 | (double log 0.543; −0.48) | — |

- Leave one TF out: symmetric mean z stays between +2.04 and +2.50; forward between +1.59 and +3.24.
- The new edge and the old edge agree at Spearman 0.35 over all pairs. F agrees at 0.89.

What this means: on the deployed genes, the corrected attention beats the degree null on TRRUST, and
co-expression does not. On the new single-log genes (the pre-stated v3 analysis, twice as many TFs),
it does not pass. Both use 8–16 TFs, so the TRRUST verdict for K562 depends on which genes and TFs
enter the test. I do not treat the deployed-gene result as the main one. It was not planned, and that
gene set was picked on doubly-logged values.

## 8. Other layers (supplement, not pre-specified)

Source: `layers/per_layer.csv`, `layers/layers_summary.json`. Point estimates; same null networks.

- TRRUST is higher at early layers: forward L0 0.708 (z +2.49), L1 0.712, L2 0.729; symmetric mean L0
  0.738 (z +2.90). Layer-corrected test (largest z over the 11 layers vs the same maximum on each null
  network): forward p 0.041, symmetric mean p 0.019, best layer L0 in both. Deployed K562 gave p 0.16.
- Knockdown is also higher early: forward L1 0.645, L2 0.635. Still below variance (0.690).
- Not done here: co-expression or gene-model controls at these layers, and residualisation. The v2
  report found that co-expression beats the degree null about as strongly as early-layer attention in
  RPE1. Treat the early-layer result as a lead, not a finding.

## 9. What changed compared with the deployed and v2 / v2b results

1. **Knockdown, "variance outranks attention":** holds, with a larger gap: +0.185 [+0.162, +0.208]
   (v2b K562 +0.084 with the deployed variance). Every attention variant is at chance (0.499–0.513).
2. **Knockdown, "the forward edge is no better than F":** holds (tie, +0.009 [−0.004, +0.020]).
3. **Knockdown, "removing the mask zeros lowers the edge":** now a tie (rank-conditioned − forward
   −0.006 [−0.016, +0.004]); v2b K562 had −0.016 [−0.027, −0.005].
4. **Knockdown, "transpose is worse than forward":** now a tie (+0.005 [−0.017, +0.026]); v2b K562 had
   −0.028 [−0.045, −0.011].
5. **TRRUST, K562 forward:** 0.483 (z +0.47) → 0.548 (z +1.72). Still not significant after BH.
   Variance alone still beats it (+0.094 [+0.016, +0.187]); the gene model still beats every variant.
6. **TRRUST, "symmetric edges score higher":** in v3 K562 symmetric mean − forward is +0.023
   [−0.034, +0.076], a tie. Symmetric max is not higher (−0.010).
7. **TRRUST, "symmetric attention beats the degree null (RPE1)":** does not replicate in correctly
   encoded K562 on the pre-stated gene set (z +1.76 / +1.11; q 0.082 / 0.17). It does on the deployed
   gene set (z +2.38 / +2.07, uncorrected p 0.010 / 0.017; secondary analysis, section 7).
8. **Residualised scores:** no K562 variant passes the null, as in v2b K562. The RPE1 residual pass
   (q 0.024) does not replicate.
9. **Order share:** with correct input, F explains 41% of the log-edge variance in K562 (v2b 26%), in
   line with RPE1 (38%). The deployed methods text should describe the edge as a causal-mask readout
   that mixes attention with expression order, in both cell lines.
10. **Gene set:** the K562 1,500-gene set was a product of the double log. Only 627 genes survive the
    correct rule.

Suggested plain wording for K562: "With correctly encoded inputs, layer-8 attention in K562 does not
predict knockdown responses (all variants at 0.50–0.51; gene variance 0.69) and does not beat a
degree-preserving null on TRRUST after correction (best z +1.8). Gene-level baselines remain ahead on
both endpoints."

## 10. What I did not do, and limits

- **Not re-run:** Phase 0b (value-weighted edges), Phase 4 (head ablation), Phase 6 (CSSI), and the
  Phase 2 incremental logistic regression ("+0.002 at most"). They were not part of this item. Their
  K562 numbers are still on wrong inputs. Per-head attention was not saved (head mean per layer only),
  so Phase 4 cannot be redone from these files.
- **Adamson and 1B K562** were not re-run (other items). Their v2 / v2b numbers stay on wrong inputs.
- **Small TRRUST test.** 16 TFs (8 on the deployed genes). z values near 2 move with single TFs
  (MYC alone moves forward z from +1.72 to +0.12).
- **Two "single log" label rules disagree in size:** 194 vs 67 perturbations. The verdict is the same
  under both.
- CIs for the gene model and for residuals hold the fitted models fixed. The TF bootstrap treats TFs as
  independent, while candidate genes recur across TFs. Both make intervals somewhat too narrow.
- The Wilcoxon p-values (knockdown) are BH-corrected over 6 scores within K562 only.
- The K562 file keeps 6,546 genes. Counts, gene sets and token orders cover only those genes.
- Only TRRUST was used as a curated reference.
- The early-layer result (section 8) has no co-expression, gene-model or residual control.
- No paper file was edited.

## 11. Files

- `scripts/v3_attention_k562.py`: `prep`, `extract [max_minutes]` (resumable), `finalize`.
- `scripts/v3_attention_k562_eval.py`: `de`, `order`, `knockdown`, `trrust`, `gene3`, `resid ols|hgb|combine`,
  `coexpr`, `layers`, `oldgenes`, `oldgenes_coexpr`, `loo`, `table`.
- `scripts/v3_attention_k562_verify.py`: `pairs`, `counts`, `geneset`, `labels`, `scores`, `resid`, `config`.
- `outputs/v3_attention_k562/`:
  - `prep/`: cells and tokens (`cells_tokens.npz`, new and deployed tokens), counts (`counts_int32.npy`),
    `gene_features.csv`, `hvg_gene_table.csv`, `union_gene_table.csv`, `spearman_edges.npy`,
    `order_agreement_per_cell.csv`, `prep_check.json`.
  - `extract/`: `state.npz` (sums over the 2,373-gene union, all layers), `state_meta.json` (per-cell
    sequence length, NLL, encoding checks, chunk times).
  - `edges/`: `attention_edges_layer_mean.npy` (11 × 1,500 × 1,500, new genes),
    `attention_pair_counts.npy`, `order_counts.npy`, the same three with `_deployed_genes`, union arrays,
    `finalize_check.json`.
  - `de_labels/`: `v3_primary.npz`, `v3_storedX.npz`, `deployed_genes_single_log.npz`, `de_meta.json`.
  - `order/`, `knockdown/`, `trrust/` (incl. `null_positions.npy`, `leave_one_tf_out.json`), `resid/`,
    `coexpr/`, `layers/`, `oldgenes/`, `verification/`.
  - `table_attention_variants_v3_k562.csv`, `fig_attention_variants_v3_k562.csv` (figure source data,
    v2b format), `comparison_v2b_k562_v3_k562_v2b_rpe1.csv`.
  - `run_config.json`: device, package versions, seeds, the 2,000 h5ad rows, the 1,500 Ensembl ids,
    input-encoding record and check result, counts and tokens sha256, wall time per chunk, sha256 of
    20 inputs, 9 code files and 58 outputs.
- Re-run order: `v3_attention_k562.py prep`; `extract 7.3` (repeat until done, about 16 min in total);
  `finalize`; then the eval commands in the order listed; then the verify commands; `config` last.

## Plain-words summary

The K562 attention run had fed MaxToki genes in the wrong order. I re-ran it on the same 2,000 cells
with the right order (raw counts divided by each gene's typical level). Every cell was checked before
the model saw it, and the model predicts genes as well as expected on these inputs.

Fixing the data also changed which 1,500 genes the pipeline picks: only 627 of the old genes stay.
The attention scores themselves changed a lot too.

The main results:
- **Knockdown responses.** Attention is at chance in every form (about 0.50). Simple gene variance
  scores 0.69. The old conclusion holds and is even clearer.
- **TRRUST targets.** Attention improved (0.48 → 0.55; symmetric 0.51 → 0.57). It still does not beat
  a fair random-network test after correcting for several tests, and the forward result depends on one
  TF (MYC). Simple gene-level baselines still score higher.
- **Compared with RPE1** (the run that was always correct): RPE1's symmetric attention beat the random
  networks. Correctly encoded K562 does not, on the planned gene set. On the old gene set it does,
  but that check was not planned and uses only 8 TFs.
- With correct inputs, about 40% of the attention score is just "which gene comes first in the cell",
  the same as in RPE1.

Not done: head ablation, value-weighted edges, CSSI, the incremental-value regression, and the Adamson
and 1B runs. Those numbers are still on the wrong inputs.
