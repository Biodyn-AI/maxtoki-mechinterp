# Attention edge: direction and rank baselines (revision item D1b)

Date: 2026-10-01. CPU only. No model forward pass and no model weights loaded. The
MaxToki tokeniser (a numpy rank encoder) was used only to rebuild the order of genes in
each control cell. Script: `scripts/v2b_attention.py` (one file, sub-commands).
Outputs: `outputs/v2b_attention/`. Inputs, scripts and outputs are listed with sha256 in
`outputs/v2b_attention/run_config.json`.

The four runs are 217M K562, 217M RPE1, 217M Adamson and 1B K562. Everything uses the
deployed attention tensor: layer index 8, mean over heads, averaged over the control
cells that contain both genes.

## 0. Why this matters, in one paragraph

MaxToki is a causal decoder. Genes enter in order of falling (median-normalised)
expression, and a token can only attend to tokens before it. The deployed edge is
`attn[P, T]`: P (the regulator, or the knocked-down gene) is the query and T (the
candidate target) is the key. In a cell where T comes after P, this attention is forced
to 0. Phase 0 still counted those cells in the average. So the saved edge splits
exactly into two parts:

- **E[P, T] = F[P, T] × Ec[P, T]**
- **F[P, T]** (the "order share") = the share of cells containing both genes in which T
  is ranked above P. It uses no model.
- **Ec[P, T]** (the "rank-conditioned edge") = the mean attention over only the cells
  where the mask allows it.

## 1. Scores compared

| Name | Definition | Uses the model? |
|---|---|---|
| forward | E[P, T] (deployed) | yes |
| transpose | E[T, P] | yes |
| symmetric mean | (E[P, T] + E[T, P]) / 2 | yes |
| symmetric max | max(E[P, T], E[T, P]) | yes |
| rank-conditioned | Ec[P, T] = E[P, T] × n_pair / C[P, T]; 0 where C = 0 | yes |
| order share F | C[P, T] / n_pair | **no** |
| \|Spearman\| co-expression | deployed `spearman_edges.npy`, same control cells (reference only, outside the test family) | no |
| target variance | deployed `gene_features.csv` | no |
| target variance, single log | variance of the stored h5ad values (only for the 3 runs below whose X is log1p) | no |
| 3-feature gene model | v2 logistic model on target mean, variance, dropout (TRRUST only) | no |

C[P, T] = number of control cells that contain both genes and put T before P.
n_pair = number of control cells that contain both genes (the deployed pair count).

## 2. Input scale of each run (V2_EVAL V3)

| Run | What the model was fed | Clean? |
|---|---|---|
| 217M K562 | log1p(CP10k) values, treated as counts | no |
| 217M RPE1 | raw integer counts | **yes, the clean reference** |
| 217M Adamson | log1p(CP10k) values, treated as counts | no |
| 1B K562 | log1p(CP10k) values, treated as counts (and only 200 cells) | no |

I rebuilt the token order exactly as the model saw it, wrong scale included. So F and
the split E = F × Ec are exact for the saved tensors. But in 3 runs neither the attention
nor F is what MaxToki would produce from correct counts. Re-extracting attention needs a
model run. Where the runs disagree, read RPE1 first.

## 3. Checks (all passed)

Source: `outputs/v2b_attention/order/order_check_<run>.json` and the `checks_vs_v2`
fields of the other JSONs.

- **Token order rebuilt exactly.** The pair counts I rebuilt equal the saved
  `attention_pair_counts.npy` in every cell of the matrix, in all 4 runs (0 mismatches).
  In every cell pair, exactly one direction is allowed: C + Cᵀ = n_pair off the diagonal.
- **The mask pattern matches the saved attention.** Over all pairs with n_pair > 0
  (2.22–2.25 million per run): the saved layer-8 edge is 0 exactly where C = 0 and
  positive exactly where C > 0. Mismatches: 0 and 0 in all 4 runs. This confirms the
  rebuilt order is the one the model saw.
- **v2 numbers reproduced exactly.** Forward and variance per-perturbation AUROCs equal
  v2 (max difference 1e-16). TRRUST forward and variance AUROCs, the forward TF-bootstrap
  CI, and all 1,000 Curveball null values for the forward edge equal v2 exactly (I
  regenerated the same null networks with the v2 sampler and seeds). The forward OLS
  residual with the deployed feature set equals v2 `ols5_f64` exactly. The 3-feature gene
  model equals v2 exactly.
- **AUROC self-test.** A constant score gives 0.5; tied scores match sklearn.

Pairs with n_pair > 0 but C = 0 (T never above P), where the rank-conditioned edge is set
to 0: 1.4% (K562), 0.18% (RPE1), 3.0% (Adamson), 3.6% (1B).

## 4. How much of the forward edge is just expression order

Source: `order/order_check_<run>.json`. All pairs with n_pair > 0, layer 8.

| | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| Spearman(E, F), all pairs | 0.41 | 0.61 | 0.53 | 0.67 |
| Median within-row Spearman(E[P,·], F[P,·]) | 0.48 | 0.77 | 0.62 | 0.76 |
| R² of log E on log F | 0.26 | 0.38 | 0.50 | 0.37 |
| Median within-row Spearman of F[P,·] with a target-only order score | 0.78 | 0.86 | 0.94 | 0.73 |

What this means:
- In the clean run (RPE1), the order share explains 38% of the variance of the log edge,
  and within one regulator's row the edge and F rank targets with Spearman 0.77.
- Within a row, F is mostly a property of the target alone (Spearman 0.73–0.94 with a
  score that ignores P: how often T sits above any partner). So F works like a
  gene-level baseline, not a pair-level one.

## 5. Endpoint A — knockdown response

Rules as in V2_EVAL §2 (v2 DE labels, mean of per-perturbation AUROCs). 95% CI =
percentile bootstrap over **perturbations**, 2,000 resamples (same draws as v2).
Source: `knockdown/knockdown_variants_summary.json`, `knockdown/per_perturbation_<run>.csv`.
Perturbations: 174 / 325 / 57 / 155.

**Mean per-perturbation AUROC [95% CI]**

| Score | 217M K562 (log input) | 217M RPE1 (raw counts) | 217M Adamson (log input) | 1B K562 (log input) |
|---|---|---|---|---|
| forward attn[P,T] (deployed) | 0.512 [0.497, 0.527] | 0.603 [0.594, 0.612] | 0.508 [0.486, 0.529] | 0.540 [0.523, 0.556] |
| transpose attn[T,P] | 0.484 [0.470, 0.498] | 0.487 [0.477, 0.498] | 0.501 [0.485, 0.517] | 0.488 [0.472, 0.504] |
| symmetric mean | 0.500 [0.484, 0.516] | 0.583 [0.572, 0.593] | 0.496 [0.475, 0.517] | 0.527 [0.510, 0.543] |
| symmetric max | 0.497 [0.481, 0.512] | 0.580 [0.571, 0.590] | 0.487 [0.468, 0.506] | 0.528 [0.512, 0.542] |
| rank-conditioned attn[P,T] | 0.496 [0.482, 0.510] | 0.560 [0.551, 0.570] | 0.449 [0.434, 0.465] | 0.535 [0.518, 0.552] |
| **order share F (no model)** | 0.517 [0.501, 0.534] | **0.648 [0.637, 0.658]** | 0.528 [0.504, 0.551] | 0.532 [0.514, 0.549] |
| \|Spearman\| co-expression (reference) | 0.510 [0.495, 0.525] | 0.622 [0.606, 0.636] | 0.608 [0.580, 0.637] | 0.492 [0.477, 0.506] |
| target variance (deployed) | 0.596 [0.581, 0.611] | 0.766 [0.757, 0.776] | 0.712 [0.697, 0.727] | 0.596 [0.580, 0.612] |
| target variance (single log) | 0.614 [0.597, 0.631] | same as deployed | 0.787 [0.777, 0.798] | 0.616 [0.598, 0.632] |

**Difference from the deployed forward edge (score − forward) [95% CI]**

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| transpose | −0.028 [−0.045, −0.011] | −0.116 [−0.129, −0.103] | −0.007 [−0.027, +0.014] | −0.052 [−0.075, −0.029] |
| symmetric mean | −0.012 [−0.021, −0.003] | −0.021 [−0.027, −0.014] | −0.012 [−0.023, −0.001] | −0.013 [−0.025, +0.000] |
| symmetric max | −0.015 [−0.026, −0.004] | −0.023 [−0.030, −0.016] | −0.021 [−0.034, −0.008] | −0.012 [−0.026, +0.001] |
| rank-conditioned | −0.016 [−0.027, −0.005] | −0.043 [−0.048, −0.038] | −0.059 [−0.078, −0.040] | −0.004 [−0.012, +0.004] |
| **order share F** | +0.005 [−0.008, +0.019] | **+0.045 [+0.037, +0.052]** | **+0.019 [+0.008, +0.033]** | −0.008 [−0.020, +0.004] |

**Gap, deployed target variance − score [95% CI]**

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| forward | +0.084 [+0.062, +0.107] | +0.163 [+0.154, +0.174] | +0.204 [+0.174, +0.236] | +0.056 [+0.033, +0.080] |
| transpose | +0.111 [+0.091, +0.133] | +0.279 [+0.265, +0.294] | +0.211 [+0.189, +0.233] | +0.108 [+0.084, +0.130] |
| symmetric mean | +0.096 [+0.073, +0.119] | +0.184 [+0.172, +0.196] | +0.216 [+0.187, +0.244] | +0.069 [+0.045, +0.093] |
| symmetric max | +0.099 [+0.076, +0.123] | +0.186 [+0.175, +0.197] | +0.225 [+0.197, +0.253] | +0.069 [+0.045, +0.093] |
| rank-conditioned | +0.100 [+0.080, +0.120] | +0.206 [+0.194, +0.218] | +0.263 [+0.240, +0.284] | +0.061 [+0.037, +0.085] |
| order share F | +0.078 [+0.054, +0.103] | +0.118 [+0.109, +0.129] | +0.185 [+0.151, +0.219] | +0.064 [+0.040, +0.088] |

Paired Wilcoxon tests (zsplit, two-sided, uncorrected) are in the JSON and in
`table_attention_variants.csv`. Variance vs every score: p ≤ 6e-5 in every run.

What Endpoint A shows:
1. **The variance verdict survives every variant.** Gene variance beats every attention
   variant and F in every run. The smallest gap is +0.056 [+0.033, +0.080] (1B, forward).
   Every CI is above 0.
2. **Among attention edges, the deployed forward edge is the best** in K562, RPE1 and
   Adamson. In 1B it ties the rank-conditioned edge (−0.004 [−0.012, +0.004]).
   *[Corrected in verification: it is clearly best only in K562 and RPE1. In Adamson it
   ties the transpose (−0.007 [−0.027, +0.014]). In 1B it also ties symmetric max
   (−0.012 [−0.026, +0.001]); symmetric mean is borderline (−0.013 [−0.025, +0.000]).]*
3. **The forward edge is no better than the model-free order share.** F ties forward in
   K562 and 1B. F beats forward in RPE1 (+0.045 [+0.037, +0.052]) and Adamson
   (+0.019 [+0.008, +0.033]). So the knockdown signal of the deployed edge is mostly
   expression order, which needs no model.
4. **When the order is removed, the edge gets worse.** The rank-conditioned edge is below
   forward in all 4 runs (by 0.004 to 0.059). *[Corrected in verification: below forward
   in 3 of 4 runs (by 0.016 to 0.059). In 1B the difference is −0.004 [−0.012, +0.004], a
   tie.]* It is at or below chance in K562 (0.496)
   and Adamson (0.449). Only RPE1 keeps some signal (0.560 [0.551, 0.570]), and that is
   below F (0.648) and co-expression (0.622).
   *[Second check: "the order is removed" is too strong. Dividing by F removes the cells
   where the mask forces 0. It does not make the edge independent of expression order.
   Within one regulator's row, the rank-conditioned edge still correlates with F: median
   Spearman 0.46 (RPE1), 0.41 (1B), −0.14 (K562), −0.20 (Adamson). Read point 4 as "with
   the mask zeros removed, the edge gets worse".]*

## 6. Endpoint B — TRRUST edges

Rules as in V2_EVAL §3 (evaluated TFs with ≥ 3 targets; pooled AUROC). 95% CI =
percentile bootstrap over **TFs**, 2,000 resamples (same draws as v2). TFs: 8 / 16 / 32 /
12; positive pairs: 45 / 138 / 236 / 73. Source: `trrust/trrust_variants_summary.json`,
`trrust/gene_model_gaps_<run>.json`, `trrust/per_tf_<run>.csv`.

**Pooled AUROC [95% CI]**

| Score | 217M K562 (log input) | 217M RPE1 (raw counts) | 217M Adamson (log input) | 1B K562 (log input) |
|---|---|---|---|---|
| forward attn[P,T] (deployed) | 0.483 [0.399, 0.560] | 0.605 [0.549, 0.655] | 0.520 [0.478, 0.561] | 0.555 [0.462, 0.650] |
| transpose attn[T,P] | 0.548 [0.446, 0.614] | 0.546 [0.453, 0.656] | 0.561 [0.503, 0.611] | 0.579 [0.513, 0.639] |
| symmetric mean | 0.513 [0.410, 0.583] | 0.638 [0.578, 0.701] | 0.551 [0.512, 0.589] | 0.636 [0.580, 0.687] |
| symmetric max | 0.535 [0.460, 0.588] | 0.647 [0.579, 0.711] | 0.544 [0.506, 0.583] | 0.629 [0.570, 0.686] |
| rank-conditioned attn[P,T] | 0.515 [0.386, 0.596] | 0.603 [0.541, 0.662] | 0.563 [0.520, 0.599] | 0.583 [0.500, 0.662] |
| order share F (no model) | 0.454 [0.376, 0.571] | 0.552 [0.484, 0.616] | 0.428 [0.370, 0.494] | 0.482 [0.376, 0.605] |
| \|Spearman\| co-expression (reference) | 0.543 [0.470, 0.608] | 0.596 [0.542, 0.650] | 0.539 [0.495, 0.585] | 0.580 [0.507, 0.640] |
| target variance (deployed) | 0.725 [0.619, 0.836] | 0.686 [0.633, 0.731] | 0.524 [0.476, 0.567] | 0.629 [0.554, 0.721] |
| target variance (single log) | 0.638 [0.511, 0.784] | same as deployed | 0.625 [0.588, 0.661] | 0.682 [0.596, 0.784] |
| 3-feature gene model | 0.745 [0.672, 0.841] | 0.749 [0.691, 0.801] | 0.652 [0.611, 0.699] | 0.679 [0.611, 0.768] |

**Degree-preserving (Curveball) null.** Same 1,000 networks as v2 (5,000 trades each).
z = (observed − null mean) / null SD. p = one-sided Phipson–Smyth. q24 = BH over the
24 tests (6 scores × 4 runs). Co-expression is a reference outside that family. Target-only
scores (variance, gene model) cannot be tested: this null keeps their AUROC fixed.

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| forward (deployed) | z +0.47 (p 0.32; q 0.43) | z +1.82 (p 0.046; q 0.14) | z +1.50 (p 0.070; q 0.17) | z +0.77 (p 0.22; q 0.37) |
| transpose | z +1.21 (p 0.11; q 0.22) | z +2.32 (p 0.008; q 0.064) | z +0.84 (p 0.20; q 0.37) | z +0.44 (p 0.32; q 0.43) |
| symmetric mean | z +0.27 (p 0.39; q 0.49) | **z +3.19 (p 0.003; q 0.036)** | z +1.50 (p 0.068; q 0.17) | z +2.15 (p 0.021; q 0.084) |
| symmetric max | z +0.62 (p 0.26; q 0.39) | **z +3.02 (p 0.003; q 0.036)** | z +1.37 (p 0.088; q 0.19) | z +2.09 (p 0.018; q 0.084) |
| rank-conditioned | z −0.41 (p 0.67; q 0.67) | z +2.29 (p 0.013; q 0.078) | z +0.25 (p 0.41; q 0.49) | z +1.70 (p 0.047; q 0.14) |
| order share F | z +0.73 (p 0.23; q 0.37) | z +0.04 (p 0.49; q 0.55) | z −0.07 (p 0.55; q 0.57) | z −0.09 (p 0.54; q 0.57) |
| \|Spearman\| co-expression | z −0.48 (p 0.67) | z +3.04 (p 0.002) | z +0.26 (p 0.39) | z +2.02 (p 0.017) |

**Gaps to the gene-level baselines (baseline − score) [95% CI, TF bootstrap]**

3-feature gene model:

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| forward | +0.262 [+0.166, +0.371] | +0.145 [+0.055, +0.238] | +0.133 [+0.067, +0.199] | +0.124 [+0.041, +0.213] |
| transpose | +0.196 [+0.071, +0.383] | +0.203 [+0.045, +0.317] | +0.091 [+0.016, +0.176] | +0.101 [−0.010, +0.230] |
| symmetric mean | +0.232 [+0.109, +0.398] | +0.111 [+0.002, +0.202] | +0.101 [+0.037, +0.169] | +0.044 [−0.035, +0.142] |
| symmetric max | +0.210 [+0.107, +0.336] | +0.102 [−0.009, +0.204] | +0.108 [+0.044, +0.173] | +0.051 [−0.036, +0.142] |
| rank-conditioned | +0.230 [+0.098, +0.414] | +0.146 [+0.040, +0.243] | +0.089 [+0.023, +0.161] | +0.097 [+0.037, +0.173] |
| order share F | +0.291 [+0.222, +0.357] | +0.198 [+0.134, +0.272] | +0.224 [+0.159, +0.286] | +0.197 [+0.069, +0.299] |

Target variance (deployed for RPE1; single log for the 3 log-input runs, which is the
fairer baseline there):

| Score | 217M K562 (single log) | 217M RPE1 | 217M Adamson (single log) | 1B K562 (single log) |
|---|---|---|---|---|
| forward | +0.155 [+0.012, +0.296] | +0.082 [+0.028, +0.145] | +0.105 [+0.054, +0.153] | +0.126 [+0.061, +0.201] |
| transpose | +0.090 [−0.094, +0.327] | +0.140 [+0.026, +0.246] | +0.064 [−0.002, +0.135] | +0.103 [−0.022, +0.251] |
| symmetric mean | +0.125 [−0.035, +0.328] | +0.048 [−0.021, +0.110] | +0.074 [+0.024, +0.123] | +0.046 [−0.043, +0.138] |
| symmetric max | +0.103 [−0.027, +0.257] | +0.039 [−0.024, +0.105] | +0.080 [+0.030, +0.127] | +0.053 [−0.051, +0.147] |
| rank-conditioned | +0.123 [−0.051, +0.347] | +0.083 [+0.012, +0.155] | +0.062 [+0.003, +0.127] | +0.099 [+0.032, +0.175] |
| order share F | +0.184 [+0.073, +0.269] | +0.135 [+0.058, +0.212] | +0.197 [+0.132, +0.252] | +0.200 [+0.099, +0.286] |

Gaps to the deployed (doubly logged) variance are in `table_attention_variants.csv`.
With that baseline, Adamson ties every attention variant, as v2 found for forward.

**Difference from the forward edge, and from co-expression [95% CI]**

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| symmetric mean − forward | +0.030 [−0.055, +0.089] | +0.034 [−0.015, +0.091] | +0.031 [−0.001, +0.061] | +0.080 [−0.005, +0.184] |
| symmetric max − forward | +0.052 [−0.012, +0.104] | +0.043 [+0.007, +0.086] | +0.025 [−0.007, +0.056] | +0.073 [−0.035, +0.199] |
| symmetric mean − co-expression | −0.030 [−0.151, +0.082] | +0.042 [−0.040, +0.127] | +0.012 [−0.048, +0.078] | +0.056 [−0.053, +0.161] |
| symmetric max − co-expression | −0.009 [−0.108, +0.102] | +0.051 [−0.041, +0.139] | +0.006 [−0.053, +0.069] | +0.049 [−0.065, +0.166] |

### 6a. Residualised scores

Every score was predicted from gene-level features over all 1,500 × 1,499 ordered pairs,
with 5-fold cross-fitting (KFold, shuffled, seed 42, as in v2). The residual was then
scored on TRRUST. Main model (stated before looking): OLS (float64) on mean, variance and
dropout of both genes, **plus F and an indicator for n_pair = 0**. For F itself the
features are the gene features only. Check: HistGradientBoosting (HGB) on the same
features. CI = TF bootstrap, 2,000 reps, fitted models held fixed. The null z uses the
same Curveball networks. q = BH over the 24 main tests. Source: `resid/resid_summary.json`.

Note: F is antisymmetric (F[P,T] = 1 − F[T,P]), so it cannot explain a symmetric score;
for the symmetric edges, "+F" changes nothing. *[Verification: identical in 3 runs. In
1B it changes the residual AUROC by up to 0.006, because 24,308 pairs never co-occur
there and F = 0 in both directions for them.]*

**Residualised pooled AUROC [95% CI]; null z; q**

| Score | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| forward (deployed) | 0.536 [0.435, 0.602]; z −0.20; q 0.61 | 0.511 [0.425, 0.593]; z +2.15; q 0.060 | 0.592 [0.560, 0.621]; z +1.61; q 0.15 | 0.588 [0.490, 0.679]; z +2.32; q 0.067 |
| transpose | 0.489 [0.430, 0.563]; z +0.31; q 0.45 | 0.552 [0.487, 0.616]; **z +3.04; q 0.024** | 0.513 [0.473, 0.558]; z +0.73; q 0.40 | 0.548 [0.456, 0.629]; z +0.83; q 0.39 |
| symmetric mean | 0.526 [0.431, 0.586]; z +0.24; q 0.46 | 0.538 [0.456, 0.614]; **z +2.75; q 0.024** | 0.562 [0.526, 0.601]; z +1.48; q 0.17 | 0.595 [0.540, 0.648]; z +1.83; q 0.12 |
| symmetric max | 0.553 [0.486, 0.607]; z +0.57; q 0.42 | 0.536 [0.445, 0.615]; **z +3.10; q 0.024** | 0.579 [0.544, 0.617]; z +1.39; q 0.18 | 0.607 [0.553, 0.661]; z +1.83; q 0.12 |
| rank-conditioned | 0.521 [0.417, 0.593]; z −0.49; q 0.69 | 0.501 [0.413, 0.582]; z +1.97; q 0.11 | 0.570 [0.530, 0.613]; z +0.42; q 0.45 | 0.518 [0.437, 0.622]; z +1.47; q 0.18 |
| order share F | 0.449 [0.369, 0.563]; z +0.59; q 0.42 | 0.495 [0.418, 0.566]; z +0.21; q 0.46 | 0.418 [0.363, 0.473]; z +0.42; q 0.45 | 0.464 [0.356, 0.584]; z +0.24; q 0.46 |
| \|Spearman\| co-expression (reference) | 0.505 [0.412, 0.577]; z −0.18 | 0.487 [0.436, 0.532]; z +2.94 | 0.503 [0.455, 0.562]; z +0.69 | 0.573 [0.495, 0.633]; z +2.09 |

HGB check (same features): RPE1 symmetric mean 0.603 [0.540, 0.666] (z +2.97, q 0.034),
symmetric max 0.606 [0.524, 0.676] (z +2.70, q 0.034), forward 0.593 [0.521, 0.651]
(z +2.08, q 0.098). 1B forward 0.604 [0.557, 0.668], symmetric mean 0.624
[0.578, 0.665] (z +1.55 and +1.68; q 0.12). K562 stays at chance. Full HGB table and the
gene-features-only OLS are in `table_attention_variants.csv`.

What Endpoint B shows:
1. **Direction matters on TRRUST.** Symmetric edges score higher than the forward edge
   in all 4 runs (+0.025 to +0.080). The difference is clear only for symmetric max in
   RPE1 (+0.043 [+0.007, +0.086]). The test is small (8–32 TFs).
2. **"No variant beats the degree null" is false.** In the clean run (RPE1), symmetric
   mean (z +3.19) and symmetric max (z +3.02) beat the working degree-preserving null
   after BH over all 24 tests (q = 0.036). The forward edge does not (z +1.82, q 0.14).
3. **But plain co-expression does the same.** |Spearman| co-expression from the same
   cells gets z +3.04 in RPE1 and +2.02 in 1B. Symmetric attention minus co-expression is
   +0.042 [−0.040, +0.127] in RPE1. So the pair-specific signal is about the size of
   co-expression, not more. *[Verification: "not clearly more" is the safer wording. The
   interval cannot rule out an advantage of up to about +0.12.]*
   *[Second check: "about the same size" must not be read as "the same signal". Over all
   ordered gene pairs, layer-8 attention and |Spearman| co-expression are almost
   unrelated (Spearman 0.10 for symmetric mean in RPE1; 0.005–0.025 in the other runs).
   In RPE1, symmetric attention still beats the degree null after co-expression is
   removed (z +2.5 with a linear fit, +2.9 to +3.0 with 50 co-expression bins), and
   co-expression still beats it after symmetric attention is removed (z +2.2 to +2.4).
   The degree null explains the small lead of symmetric attention over co-expression
   (null test of the difference: z +0.22 for symmetric mean, −0.04 for symmetric max).
   So: a separate signal of similar size, not a larger one.]*
4. **Residualising does not send every variant to chance.** In RPE1, the transpose and
   symmetric residuals still beat the degree null (z +2.75 to +3.10, q 0.024). The
   co-expression residual does too (z +2.94). In 1B and Adamson (log input) several
   residual intervals sit above 0.5 (e.g. 1B symmetric max 0.607 [0.553, 0.661]), but
   none passes the null after BH. Gene features there were computed on doubly logged
   data, so the residualisation may be incomplete.
   *[Added in verification: in RPE1 the residual AUROCs themselves are 0.536–0.552, and
   every TF-bootstrap interval includes 0.5. They beat the degree null only because the
   null mean falls to 0.48–0.50 after residualising (co-expression: residual 0.487, null
   mean 0.43). So "chance" must be defined. Against 0.5, no RPE1 residual is above
   chance, but some Adamson and 1B residuals are. Against the degree null, the RPE1
   transpose and symmetric residuals are above it, and no Adamson or 1B residual passes
   it after BH. Either way, "collapses to chance for every readout" is not supported.]*
5. **The gene-level baselines still lead, but not always clearly.** The 3-feature gene
   model beats every variant in K562 and Adamson. In RPE1 it beats symmetric mean only
   just (+0.111 [+0.002, +0.202]) and ties symmetric max (+0.102 [−0.009, +0.204]). In 1B
   it ties the symmetric and transpose edges. *[Verification: the RPE1 split between
   "just beats" and "ties" depends on which TFs share a fold. With a different random
   TF-to-fold map: +0.115 [+0.019, +0.206] and +0.106 [+0.008, +0.207]. Read both as "the
   gene model is ahead by about 0.10, with the lower bound near 0".]* Variance alone ties
   the symmetric edges in RPE1 (+0.048 [−0.021, +0.110]) and in 1B (single log).
6. **F alone carries no TRRUST signal.** It is at or below chance (0.43–0.55) and never
   beats the null (z −0.09 to +0.73). On TRRUST the order share works against the forward
   edge in some runs: after removing F, the forward residual rises in Adamson
   (0.534 → 0.592) and 1B (0.512 → 0.588).

## 7. What differs from the deployed claims

1. **Knockdown, "variance outranks attention by 6–20 AUROC points in every run":**
   holds for every attention variant (gaps +0.056 to +0.279, every CI above 0).
2. **New, knockdown:** the deployed directed edge is no better than F, a model-free
   statistic of expression order (ties in K562 and 1B; F wins in RPE1 by +0.045
   [+0.037, +0.052] and in Adamson by +0.019 [+0.008, +0.033]). Removing the order
   lowers the edge in all 4 runs. *[Corrected in verification: in 3 of 4 runs; a tie in
   1B (−0.004 [−0.012, +0.004]).]*
3. **TRRUST, "z ≈ 0 in all four runs":** v2 already showed this is false for the forward
   edge (z +0.47 to +1.82). It is also false that no attention edge beats the degree
   null: symmetric edges do in RPE1 (z +3.19 / +3.02, q 0.036 over 24 tests). This is
   matched by co-expression (z +3.04).
4. **TRRUST, "once the gene-level information is removed ... no better than chance" and
   the Fig 2 caption "residualizing attention collapses it to chance":** tested on the
   forward edge only. For transpose and symmetric edges in RPE1 the residual still beats
   the degree null (q 0.024), as does residualised co-expression. *[Added in verification:
   those RPE1 residual AUROCs are 0.536–0.552 and their intervals include 0.5; they beat
   the null because the null mean drops to 0.48–0.50. The paper should name the
   reference: "no better than 0.5" holds in RPE1 for every variant; "no better than the
   degree null" does not hold for the RPE1 transpose and symmetric residuals.]*
5. **Table 1 / methods:** the edge should be described as a directed causal-attention
   readout with the regulator as query, averaged over all co-occurring cells including
   cells where the mask forces 0. 26–50% of the variance of the log edge is the order
   share (R² 0.26 / 0.38 / 0.50 / 0.37).
6. **"Pipelines port to the autoregressive regime" / "replicates in a causal decoder":**
   the edge was ported with no change for the causal mask. On knockdown this does not
   change the verdict. On TRRUST the choice of direction moves the score by up to 0.08.
7. **README.md:85** says scGPT is "also a gene-token decoder" and that "the pipeline
   handles it". The paper (main.tex:844-845) calls scGPT encoder-only. Under this item's
   rules I did not edit README.md. Suggested replacement: "MaxToki is a causal decoder;
   attn[P,T] is zero in cells where T is ranked below P. The deployed edge averages over
   all co-occurring cells, so it mixes attention with the share of cells in which T
   outranks P (see V2B_ATTENTION_REPORT.md)."

Suggested wording for the paper (plain): "On the knockdown endpoint, every attention
readout loses to gene variance, and the deployed directed readout does no better than
a model-free statistic of expression order. On TRRUST, symmetrised attention in the
raw-count run carries pair-specific signal beyond node degree, about as large as
co-expression from the same cells, and still below a three-feature gene model."
*[Second check: if the paper uses this sentence, it should not suggest the attention
signal is co-expression. The two scores barely correlate across gene pairs, and in RPE1
each keeps its edge over the degree null after the other is removed (§6 point 3).]*

## 8. What I did not do, and limits

- **No model run.** I could not re-extract attention from correct count input for
  K562, Adamson and 1B, so only RPE1 is clean. I could not re-extract 1B with 2,000 cells.
- **Only layer 8.** The variants were not computed at other layers (possible on CPU,
  but not part of this item). v2 showed early layers score higher on TRRUST.
- **HGB check for F in 1B not finished.** Two attempts ran past the 9-minute limit while
  the machine was heavily loaded by other jobs (load average 60–105); I stopped both.
  The OLS result for F in 1B is given. The HGB BH family therefore has 23 tests.
  *[Done in verification, own folds: 1B F HGB residual 0.449 [0.367, 0.541], null z +0.21,
  p 0.42. F still has no TRRUST signal.]*
- **Phase 2 incremental value (knockdown logistic regression with F as a covariate) was
  not re-run.** Knockdown was not residualised and has no null.
- The Wilcoxon p-values are not corrected for the 8 scores × 4 runs. The null tests are
  BH-corrected over the 24 tests; co-expression is outside that family.
- CIs for residualised scores and the gene model hold the fitted models fixed. The TF
  bootstrap treats TFs as independent. Both make intervals somewhat too narrow.
- The rank-conditioned edge is set to 0 where T never ranks above P (0.2–3.6% of pairs).
  No other fill rule was tried.
- Gene features used in residualisation are the deployed ones (doubly logged in 3 runs).
- Only TRRUST was used as a curated reference.
- I did not edit the paper or README.

## 9. Files

- Script: `scripts/v2b_attention.py`. Order of steps: `order <run>` ×4, `knockdown`,
  `trrust <run>` ×4, `trrust combine`, `gene3 <run>` ×4, `resid <run> ols` ×4,
  `resid <run> hgb` (repeat until done), `resid combine`, `table`, `config`.
- Outputs in `outputs/v2b_attention/`:
  - `order/order_counts_<run>.npy` (C as int32, 1,500 × 1,500),
    `order/order_check_<run>.json`, `order/gene_variance_single_log_<run>.csv`
  - `knockdown/knockdown_variants_summary.json`, `knockdown/per_perturbation_<run>.csv`
  - `trrust/trrust_variants_summary.json`, `trrust/trrust_variants_<run>.json`,
    `trrust/per_tf_<run>.csv`, `trrust/gene_model_gaps_<run>.json`,
    `trrust/null_positions_<run>.npy` (the 1,000 null networks as pooled-pair indices)
  - `resid/resid_summary.json` (+ per-run files)
  - `table_attention_variants.csv`: one row per run × endpoint × score, all statistics.
  - **`fig_attention_variants.csv`**: figure source data, long format. Columns: run,
    input_scale, endpoint, score, score_label, score_kind, measure, value, ci_low,
    ci_high, ci_method, n_units. Measures: "mean per-perturbation AUROC" (knockdown),
    "pooled AUROC", "pooled AUROC after residualising (<model>)", "Curveball null z",
    "Curveball null mean" (TRRUST).
  - `run_config.json`: sha256 of 46 inputs, 3 scripts and 64 outputs; seeds; versions.
- Seeds: the v2 master seed 20261001 with the v2 offsets (knockdown +100, TRRUST +500,
  Curveball +600, residual bootstrap +800, per run index), KFold seed 42.
- Peak memory: 2.0 GB.

## Plain-words summary

- The attention score the pipeline used is read in one direction through a causal mask.
  So it is partly just "how often is the target ranked above the regulator", which needs
  no model. I rebuilt that order exactly (it matches the saved data in every pair).
- **Knockdown:** gene variance still beats every version of attention in every run, by
  at least 0.056 AUROC. The deployed attention score is no better than the order share
  alone; in the clean RPE1 run the order share is better (0.648 vs 0.603). Removing the
  order makes attention worse in 3 of 4 runs (a tie in 1B). So the knockdown verdict holds, and it is even stronger.
  *[Second check: what is removed is the masked zeros, not all of the order; see §5
  point 4.]*
- **TRRUST:** direction matters. Symmetric attention scores higher (0.638–0.647 in
  RPE1 vs 0.605). In RPE1 it beats the degree-preserving null after correction
  (z ≈ +3.1). But plain co-expression from the same cells does about as well. A simple
  3-feature gene model still scores higher, though in RPE1 and 1B the gap to symmetric
  attention is no longer clearly above zero.
- So the paper should say: the attention readout was directed (regulator as query),
  attention never beats gene-level baselines on knockdown, and on TRRUST a symmetric
  readout holds about as much pair-level signal as co-expression. It should not say
  that attention is at chance after removing gene-level information for every readout.
  (Verification: in RPE1 the residual scores are near 0.5, but they still beat the
  degree-preserving null. The paper should say which of the two it means.)
- Three of the four runs fed log values to the model as if they were counts. RPE1 is the
  only clean run. Fixing the others needs a model run.

---

## Verification notes (independent check, 2026-10-01)

A second agent re-checked this item with its own code. CPU only; no model run, no model
weights. Script: `scripts/v2b_attention_verify.py` (does not import `v2b_attention.py` or
`v2_common.py`). Outputs: `outputs/v2b_attention/verification/`, with
`run_config.json` (sha256 of 47 inputs, the script and 30 outputs; own seed 777001).

**Verdict: OK after fixes.** Every number I re-derived agrees. The fixes are wording: a
few "all 4 runs" or "best" statements were ties in one run, and the residualisation
claim needed to say what "chance" means. Bracketed notes marked *[verification]* were
added in §5, §6, §6a, §7, §8 and the plain-words summary. No CSV or JSON was changed.

### What was re-computed and agrees

- **Token order (own rank code, all 4 runs).** I read the control cells from the h5ad
  files with h5py, divided by the gene medians, sorted, cut at 2,046 genes, and built C
  by adding a lower-triangle block per cell (a different method from §3). My C equals
  the saved `order_counts_<run>.npy` in every entry (0 mismatches in all 4 runs). My pair
  counts equal the deployed ones. E = 0 exactly where C = 0 (0 and 0 mismatches). R² of
  log E on log F: 0.256 / 0.381 / 0.500 / 0.365. Within-row Spearman values in §4 also
  reproduce (0.48 / 0.77 / 0.62 / 0.76 and 0.78 / 0.86 / 0.94 / 0.73).
- **Endpoint A (knockdown).** sklearn per-perturbation AUROCs equal the D1b files to
  within 2.2e-16 for all 8 scores in all 4 runs. Own bootstrap over perturbations (2,000
  reps, own seed):
  - order share F − forward: RPE1 +0.045 [+0.037, +0.052]; Adamson +0.019
    [+0.007, +0.031]; K562 +0.005 [−0.008, +0.019]; 1B −0.008 [−0.021, +0.004].
  - smallest variance gap: 1B forward +0.056 [+0.033, +0.080]. Largest Wilcoxon p for
    variance vs any score: 5.8e-5.
- **Endpoint B (TRRUST).** Own TRRUST matrix, sklearn pooled AUROC. Same TFs (8 / 16 /
  32 / 12) and positives (45 / 138 / 236 / 73). All pooled AUROCs equal D1b. Own TF
  bootstrap (2,000 reps; exact pooled AUROC of each resample from a TF-by-TF count
  matrix) gives CIs within about 0.01 of D1b.
- **Degree null (own sampler).** Boolean-matrix Curveball, 1,000 draws × 3,000 trades,
  own seed, own-gene cell forbidden. Row and column sums kept in every draw; no draw
  equal to the real network; mean Jaccard distance 0.874 / 0.877 / 0.935 / 0.874.
  - RPE1 z: forward +1.82, transpose +2.27, symmetric mean +3.17, symmetric max +2.97,
    rank-conditioned +2.26, F +0.04, co-expression +2.96 (D1b: +1.82, +2.32, +3.19,
    +3.02, +2.29, +0.04, +3.04).
  - BH over the same 24 tests: RPE1 symmetric mean and max q = 0.024 (D1b 0.036; my
    p = 0.002 vs D1b 0.003). 1B symmetric z +2.19 / +2.17, q 0.064 (D1b 0.084).
  - D1b q-values recomputed from its own p-values: exact match (raw and residual).
- **Residualisation (own OLS, own random 5-fold split).** Matches D1b to 3 decimals.
  RPE1 residuals: transpose 0.552 (z +3.02), symmetric mean 0.538 (+2.80), symmetric
  max 0.536 (+3.06); own BH q 0.024 for all three. Forward with gene features only:
  0.492 (= v2).
- **HGB spot checks (own folds).** RPE1 symmetric mean 0.601 [0.542, 0.661], z +2.86
  (D1b 0.603, +2.97). RPE1 transpose 0.571, z +2.19 (D1b 0.566, +2.19). **New: 1B F**
  (the one D1b could not finish): 0.449 [0.367, 0.541], z +0.21.
- **Figure and table CSVs.** All 205 rows of `fig_attention_variants.csv` equal the
  JSON values exactly. The sha256 of D1b's 3 scripts and 64 outputs match the files.
- **README.md:85 and main.tex:844-845.** Checked; the conflict is as stated.

### Small problems found (fixed in the text above)

1. **"Forward is the best attention edge in K562, RPE1 and Adamson."** It is clearly
   best only in K562 and RPE1. It ties the transpose in Adamson. In 1B it ties symmetric
   max and the rank-conditioned edge.
2. **"The rank-conditioned edge is below forward in all 4 runs."** True in 3 runs. 1B is
   a tie (−0.004 [−0.012, +0.004]). This was in §5, §7 and the plain summary.
3. **Residualisation and "chance".** In RPE1 the transpose and symmetric residual AUROCs
   are 0.536–0.552, and every interval includes 0.5. They beat the degree null because
   the null mean falls to 0.476–0.497 after residualising. Co-expression is the extreme
   case: residual 0.487, null mean 0.425, z +2.94. So "no better than 0.5" still holds
   for every RPE1 residual. "No better than the degree null" does not. In Adamson and 1B
   the reverse happens: some residual intervals are above 0.5 (e.g. Adamson forward
   0.592 [0.560, 0.621]) but none passes the null after BH. The paper must say which
   reference it means. Either way, "collapses to chance for every readout" is not
   supported.
4. **Wording strength.** "Co-expression does exactly as well" became "about as well".
   "Not more than co-expression" should read "not clearly more": the interval for
   symmetric mean minus co-expression in RPE1 reaches +0.12.
5. **"+F changes nothing" for symmetric edges.** Exact in 3 runs. In 1B the change is up
   to 0.006, because 24,308 gene pairs never co-occur in its 200 cells.
6. **Gene-model gaps depend on fold assignment.** With my own random TF-to-fold map, the
   RPE1 gene model is 0.753 (D1b 0.749), and its gap to symmetric max is +0.106
   [+0.008, +0.207] instead of +0.102 [−0.009, +0.204]. In 1B (12 TFs) it is 0.645
   (D1b 0.679). Conclusions do not change, but "ties" vs "just beats" is not stable.

### Notes, not errors

- F uses no network weights. It does use the tokeniser's per-gene medians (the
  Geneformer gc104M median file; README deviation 2). "Model-free" means "no learned
  weights".
- The zero-pattern check is at the gene-pair level: E = 0 exactly for pairs in which T
  never comes before P. Zeros in single cells follow from the causal mask itself; they
  were not checked cell by cell (that would need the per-cell attention).
- On knockdown, co-expression also beats the forward edge in RPE1 (+0.019
  [+0.004, +0.033]) and Adamson (+0.100 [+0.063, +0.140]). This supports point 3 of §5.
- Not re-run by me: the HGB residuals other than the three above; the Phase-2
  incremental test with F (not run by D1b either); other layers.

### Plain-words summary of the verification

- I rebuilt the gene order with my own code and got the same counts in every gene pair
  in all 4 runs. All knockdown and TRRUST numbers reproduce. My own degree-preserving
  null gives the same z values to within about 0.1.
- The main conclusions hold. Gene variance beats every attention readout on knockdown.
  The deployed readout is no better than the order share. On TRRUST, symmetric attention
  in RPE1 beats the degree null, about as much as co-expression does.
- Six small wording fixes. The most important one: after removing gene-level features,
  the RPE1 attention scores sit near 0.5 but still beat the degree null. The paper has to
  say which of these two it calls "chance".

---

## Verification notes (second independent check, 2026-10-01)

A third agent re-checked this item with new code. CPU only; no model run, no model
weights, no tokeniser call. Script: `scripts/v2b_attention_verify2.py`. It does not
import `v2b_attention.py`, `v2b_attention_verify.py` or `v2_common.py`. Outputs:
`outputs/v2b_attention/verification2/`, with `run_config.json` (sha256 of 34 inputs, the
script and 32 outputs; own seed 50505). Steps: `kd`; per run `trvec <run> raw|resA|resB`,
`trnull <run> 0 1000`, `treval <run>`; then `extra`, `bh`, `fig`, `config`.

**Verdict: OK after fixes.** Every main number reproduces. Four wording fixes were added
in the text above, marked *[Second check: ...]*. No CSV or JSON of D1b was changed.

### What was re-computed and agrees

- **Order counts.** I did not rebuild the token order a third time (D1b used the
  tokeniser; the first check used its own rank code; both agree exactly). I checked the
  saved C against invariants in all 4 runs: C ≥ 0, C ≤ n_pair, C + Cᵀ = n_pair off the
  diagonal, and the layer-8 edge is 0 exactly where C = 0 (0 mismatches in each
  direction; no positive edge where n_pair = 0).
- **Endpoint A (knockdown).** sklearn per-perturbation AUROCs equal D1b to 2.2e-16 for
  all 9 scores in all 4 runs. Own bootstrap over perturbations (2,000 reps):
  - F − forward: RPE1 +0.045 [+0.037, +0.052]; Adamson +0.019 [+0.008, +0.032]; K562
    +0.005 [−0.009, +0.018]; 1B −0.008 [−0.021, +0.004].
  - Smallest variance gap: 1B forward +0.056 [+0.034, +0.081].
  - Wilcoxon, variance vs each of the 6 scores in the family (24 tests): largest p
    5.8e-5; after BH, largest q 5.8e-5. So "p ≤ 1e-4 after correction" holds for every
    variant, not just the forward edge.
- **Endpoint B (TRRUST).** Own TSV parser and own pooled AUROC (TF-by-TF count matrix;
  equals sklearn to 1e-16). Same TFs (8 / 16 / 32 / 12) and positives (45 / 138 / 236 /
  73). All pooled AUROCs equal D1b exactly. TF-bootstrap CIs within 0.007 of D1b.
- **Degree null with a different sampler.** Checkerboard swaps, not Curveball. 1,000
  draws; each starts from the real network and makes 20 × (number of edges) accepted
  swaps; own-gene cell forbidden. Every draw keeps all row and column counts; none uses
  a forbidden cell; none equals the real network (mean Jaccard distance 0.873 / 0.877 /
  0.935 / 0.873). Over 7 scores × 4 runs, |z − z(D1b)| ≤ 0.16 (raw) and ≤ 0.16
  (residualised). RPE1: forward +1.80, transpose +2.38, symmetric mean +3.26,
  symmetric max +3.08, rank-conditioned +2.30, F +0.05, co-expression +3.02.
- **Residualisation (own OLS, own folds).** Residual AUROCs equal D1b to 0.001. RPE1
  residual z: transpose +3.11, symmetric mean +2.87, symmetric max +3.15.
- **Figure CSV.** 205 rows; every variant and F appear for both endpoints in all 4 runs;
  every row with a CI names its method and unit. The 70 AUROC values I recomputed
  (knockdown means and TRRUST pooled AUROCs) match to 1e-16.
- **README.md:85.** Confirmed: it calls scGPT "also a gene-token decoder" and says the
  pipeline "handles it".

### Problems found (wording; fixed in the text above)

1. **The rank-conditioned edge is not order-free.** D1b says the order is "removed". It
   removes only the cells where the mask forces 0. Within a regulator's row, the
   rank-conditioned edge still correlates with F (median Spearman 0.46 RPE1, 0.41 1B,
   −0.14 K562, −0.20 Adamson). Over all pairs: 0.005, 0.17, −0.28, −0.35. So §5 point 4
   should say "with the mask zeros removed", not "with the order removed".
2. **"About the size of co-expression" is not "is co-expression".** Layer-8 attention
   and |Spearman| co-expression barely correlate over all ordered pairs (Spearman 0.10
   for symmetric mean in RPE1; 0.003–0.025 elsewhere). In RPE1:
   - symmetric mean after removing co-expression: 0.611 (linear fit, z +2.53) and
     0.620 (50 co-expression bins, z +2.94); symmetric max 0.623 (z +2.50) and 0.635
     (z +3.04).
   - co-expression after removing symmetric mean (50 bins): 0.563, z +2.19.
   - null test of the difference "symmetric − co-expression": z +0.22 (mean), −0.04
     (max). The degree structure explains the small lead.
   So the RPE1 signal is a second, separate signal of about the same size. This does
   not change the "not more than co-expression" reading. It does mean the paper should
   not describe this attention signal as co-expression. (Only |Spearman| from the same
   2,000 control cells was tested. Other co-expression measures were not.)
3. **The RPE1 symmetric q-value depends on the null draw.** With 1,000 null networks
   the smallest possible p is 0.001, and the q-value moves with Monte Carlo noise:
   0.024 (first check), 0.036 (D1b), 0.048 (this check). All three are below 0.05, and
   z is stable (+3.0 to +3.3). Quote z, and say q < 0.05 under three independent null
   samplers, rather than a single q.
4. **Leave-one-TF-out.** RPE1 symmetric mean stays above the null when any one of the
   16 TFs is dropped (z 2.37 to 3.58; largest p 0.009); symmetric max z 2.12 to 3.46
   (largest p 0.020). The 1B symmetric result is fragile: dropping STAT3 lowers z to
   1.30 (p 0.11). The RPE1 forward edge drops to z 0.93 without HIF1A.

### Notes, not errors

- On knockdown, target mean expression (a simpler gene-level score) is below F in RPE1
  (−0.066 [−0.077, −0.054]) but above F in K562, Adamson and 1B. Gene variance beats
  both everywhere. This does not change any conclusion.
- The RPE1 and K562 residual vectors of stage `resA` (and RPE1 `resB`) were made before
  I added a guard that drops constant feature columns (needed for Adamson, where no pair
  has n_pair = 0). The guard does nothing when no column is constant. Re-running K562
  `resB` with the final script gave the same vectors to 6e-17.
- The machine was heavily loaded (load average up to 190). One first attempt ran past
  9 minutes and was stopped; the work was then split into the stages above.
- Not re-checked by me: the 3-feature gene model and its gaps, the HGB residuals, the
  single-log variance values (read from D1b's CSVs), other layers, and the Phase-2
  incremental test (not run by D1b either).

### Plain-words summary of the second check

- All knockdown and TRRUST numbers reproduce. A different null sampler gives the same z
  values to within 0.16.
- The knockdown verdict holds for every attention variant, also after multiple-testing
  correction.
- Two wording fixes matter for the paper. First, the "rank-conditioned" edge still
  depends on expression order, so do not call it order-free. Second, in RPE1 symmetric
  attention matches co-expression in size, but it is a different signal: the two
  barely correlate, and each survives removal of the other.
- The RPE1 q-value is close to 0.05 and moves with the null draw (0.024–0.048). Quote
  z ≈ +3.1 and say it passed under three samplers.
