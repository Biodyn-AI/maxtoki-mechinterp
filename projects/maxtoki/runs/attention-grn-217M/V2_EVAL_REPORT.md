# Attention-GRN v2 re-evaluation (revision item D1)

Date: 2026-10-01. CPU only. No model was run. Every number below was computed by a
saved script in `scripts/v2_*.py` from saved files. Inputs, scripts and outputs are
listed with sha256 in `outputs/v2_eval/run_config.json`.

The four runs are: **217M K562**, **217M RPE1**, **217M Adamson**, **1B K562**.
All attention numbers use the deployed score: layer index 8 (0-based), mean over heads,
averaged over the control cells in which both genes appear (row = query gene,
column = key gene).

There are two different endpoints. They are kept apart everywhere below.

- **Endpoint A — knockdown response.** For one knocked-down gene, which other genes
  change? Positives are measured responders in the same CRISPR screen.
- **Endpoint B — TRRUST edges.** For one transcription factor (TF), which genes does the
  curated TRRUST database list as its targets?

---

## 1. Run facts (verified from files)

Source: `outputs/v2_eval/run_facts.json` (script `v2_01_facts.py`). Parameter counts
come from the safetensors headers (tensor shapes only).

| | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 |
|---|---|---|---|---|
| Model | MaxToki-217M | MaxToki-217M | MaxToki-217M | MaxToki-1B |
| Parameters (total) | 216,946,576 | same | same | 1,049,036,544 |
| Layers × heads | 11 × 8 | 11 × 8 | 11 × 8 | 20 × 16 |
| Control cells for attention, gene set and gene statistics | 2,000 | 2,000 | 2,000 | **200** |
| Control cells for knockdown DE calls | 5,000 (of 10,691) | 5,000 (of 11,485) | 5,000 (of 24,263) | 5,000 (of 10,691) |
| Layer used | 8 of 11 (index/layers = 0.73) | 8 of 11 | 8 of 11 | **8 of 20 (0.40)** |
| Mean tokens per cell | 2,039 | 2,039 | 991 | 2,040 |

- **Model size ratio.** 1B / 217M = **4.84×** in total parameters (5.72× without the
  embedding and output layers). It is not 10×.
- **The 1B run is the 1B model.** Its Phase-0 config says `"model": "MaxToki-217M-HF"`
  (a hard-coded name), but the stored attention has 20 layers × 16 heads, and the log
  loads `setup/MaxToki-1B-HF`.
- **The 1B run is not like-for-like with 217M K562.** It used 200 control cells, not
  2,000. Its 1,500-gene set was chosen from those 200 cells, so it shares only 1,259 of
  1,500 genes with the 217M K562 set. This is why it evaluates 155 (not 174)
  perturbations and 12 (not 8) TRRUST TFs. Its layer sits at 40% depth, not 73%.
- **Layer 8 was fixed in advance** (depth rule in `README.md`), not picked by score.

---

## 2. Endpoint A — knockdown response (per perturbation)

### The exact rule (verified by re-implementing it)

1. Controls: the cell line's non-targeting cells. If there are more than 5,000, a random
   5,000 (`numpy default_rng(42)`, without replacement).
2. A perturbation is used if: it has ≥ 30 cells; its gene symbol is in the Geneformer
   symbol→Ensembl dictionary; its gene is one of the run's 1,500 genes.
3. Expression: only the run's 1,500 genes. Each cell is scaled to 10,000 **over those
   1,500 genes** (not the whole transcriptome), then log1p.
   *[Added in verification, see V3 below: in 217M K562, 1B K562 and Adamson the stored
   h5ad values are already log1p(counts per 10k). So in those three runs this step logs
   the data a second time. Only RPE1 starts from raw counts.]*
4. Test: Welch t-test per gene, perturbed cells vs controls. Benjamini–Hochberg (BH)
   over the 1,500 genes of that perturbation.
5. **Positive ("responding gene")**: |difference of mean log1p values| ≥ 0.5 **and**
   BH q < 0.05. The perturbed gene itself is removed. All other genes are negatives.
   (The 0.5 is a difference of natural-log values, not a log2 fold change.)
6. A perturbation is kept only if it has ≥ 3 positives and ≥ 3 negatives. This drops
   202 of 376 candidate perturbations in 217M K562, 109 of 434 in RPE1, 0 of 57 in
   Adamson and 219 of 374 in 1B K562.

Scores: attention = `attn[perturbed gene, candidate]`. Gene variance = the candidate's
variance in the Phase-0 control cells (one number per gene; same ranking for every
perturbation; no model). The summary is the mean of the per-perturbation AUROCs.
AUROC = area under the ROC curve (0.5 = chance, 1 = perfect ranking).

### Results

95% CI = percentile bootstrap over **perturbations**, 2,000 resamples.
Source: `outputs/v2_eval/knockdown/knockdown_summary.json`, `fig_knockdown.csv`.

| Run | n perturbations | positive pairs (median per perturbation) | Attention AUROC [95% CI] | Gene-variance AUROC [95% CI] | Gap, variance − attention [95% CI] | Perturbations where variance > attention |
|---|---|---|---|---|---|---|
| 217M K562 | 174 | 9,741 (15) | 0.512 [0.497, 0.527] | 0.596 [0.581, 0.611] | +0.084 [+0.062, +0.107] | 118 / 174 |
| 217M RPE1 | 325 | 48,245 (103) | 0.603 [0.594, 0.612] | 0.766 [0.757, 0.776] | +0.163 [+0.154, +0.174] | 314 / 325 |
| 217M Adamson | 57 | 4,438 (47) | 0.508 [0.486, 0.529] | 0.712 [0.697, 0.727] | +0.204 [+0.174, +0.236] | 56 / 57 |
| 1B K562 | 155 | 8,587 (14) | 0.540 [0.523, 0.556] | 0.596 [0.580, 0.612] | +0.056 [+0.033, +0.080] | 91 / 155 |

**Test actually used by the deployed runs.** Paired Wilcoxon signed-rank test on the
per-perturbation AUROCs (variance vs attention; scipy, `zero_method="zsplit"`,
two-sided). BH over the **13 tests of that run's own table** (4 gene baselines × 3 edge
scores, plus attention vs Spearman). It is not corrected across runs.

| Run | raw p | BH p within run (13 tests) | BH p across the 4 runs (new) |
|---|---|---|---|
| 217M K562 | 9.3e-11 | 4.0e-10 | 1.2e-10 |
| 217M RPE1 | 2.3e-52 | 1.5e-51 | 9.3e-52 |
| 217M Adamson | 6.4e-11 | 1.4e-10 | 1.2e-10 |
| 1B K562 | 5.8e-05 | 9.5e-05 | 5.8e-05 |

**Verification (two routes agree).**
- Route 1: the deployed `phase1*/per_perturbation_auroc.csv`.
- Route 2: my own DE re-implementation from the raw h5ad files (`v2_02_rederive_de.py`)
  plus my own tie-safe rank AUROC. Same perturbations, same positive counts, and
  per-perturbation AUROCs equal to within 2e-16 in all 4 runs.
- The labels also match the deployed Phase-3 pair files exactly (0 disagreements over
  260,826 / 487,175 / 85,443 / 232,345 pairs).
- All 13 Wilcoxon p-values per run were recomputed from the deployed file and match.
- AUROC self-test: a constant score gives exactly 0.5; tied scores match sklearn.

**Other layers (from the deployed per-layer file, point estimates only).** Gene variance
beats attention at every layer. The best layers are: K562 L1 0.553 (vs 0.596), RPE1
L1 0.707 (vs 0.766), Adamson L0 0.554 (vs 0.712), 1B L5 0.581 (vs 0.596).

### What "incremental ΔAUROC" measured (verified by re-running it)

- Endpoint: **knockdown response** (Endpoint A), not TRRUST.
- Data: one row per (perturbation, candidate gene) pair.
- Model: logistic regression (standardised features, balanced class weights).
- Features: the candidate's mean, variance and dropout ("gene only"), versus the same
  plus the attention score.
- Split: 5-fold GroupKFold grouped by perturbation ("cross_pert").
- Metric: mean over the 5 held-out folds of the pooled AUROC. Δ = mean(gene + attention)
  − mean(gene only). No interval was computed. GBDT was never run (`has_gbdt: false`).

My re-run reproduces the deployed Δ exactly (+0.0007, −0.0003, −0.0001, +0.0020).
New 95% CIs (bootstrap over perturbations of the pooled out-of-fold AUROC difference;
the fitted fold models are held fixed, so refitting noise is not included):

| Run | pooled out-of-fold Δ [95% CI] |
|---|---|
| 217M K562 | +0.0006 [+0.00005, +0.0012] |
| 217M RPE1 | −0.0003 [−0.0005, −0.0001] |
| 217M Adamson | −0.0001 [−0.0002, −0.00002] |
| 1B K562 | +0.0021 [+0.0010, +0.0031] |

So attention adds at most about 0.002 AUROC. In the 1B run that small gain is above
zero. Source: `outputs/v2_eval/knockdown/incremental_value_check.json`.

---

## 3. Endpoint B — TRRUST curated edges

### The exact rule (deployed rule, reproduced exactly)

- Edges: TRRUST v2 human rows where both TF and target are among the run's 1,500 genes,
  TF ≠ target.
- Evaluated TFs: TFs in the gene set with ≥ 3 such targets.
- For each evaluated TF, candidates = the other 1,499 genes. Positive = TRRUST target.
  Unlisted pairs count as negatives.
- All (TF, candidate) pairs are pooled into **one** AUROC (not a per-TF mean).

Gene-level baselines on the same pairs:
- **Variance alone**: the candidate's variance in control cells.
- **3-feature gene model**: logistic regression on the candidate's mean, variance and
  dropout. 5-fold GroupKFold grouped by TF, so a TF's own pairs are never in training.
  Out-of-fold predictions are pooled into one AUROC.

95% CI = percentile bootstrap over **TFs**, 2,000 resamples (a TF drawn k times has all
its pairs counted k times). For the gene model the fitted fold models are held fixed.
Source: `outputs/v2_eval/trrust/trrust_summary.json`, `fig_trrust.csv`.

| Run | TFs | positive pairs | negative pairs | Attention AUROC [95% CI] | Variance AUROC [95% CI] | 3-feature gene model [95% CI] |
|---|---|---|---|---|---|---|
| 217M K562 | 8 | 45 | 11,947 | 0.483 [0.399, 0.560] | 0.725 [0.619, 0.836] | 0.745 [0.672, 0.841] |
| 217M RPE1 | 16 | 138 | 23,846 | 0.605 [0.549, 0.655] | 0.686 [0.634, 0.731] | 0.749 [0.691, 0.801] |
| 217M Adamson | 32 | 236 | 47,732 | 0.520 [0.478, 0.561] | 0.524 [0.476, 0.567] | 0.652 [0.611, 0.699] |
| 1B K562 | 12 | 73 | 17,915 | 0.555 [0.462, 0.650] | 0.629 [0.554, 0.721] | 0.679 [0.611, 0.768] |

Gaps (baseline − attention), 95% CI by TF bootstrap:

| Run | Variance − attention | 3-feature model − attention | TFs where variance > attention | TFs where model > attention |
|---|---|---|---|---|
| 217M K562 | +0.242 [+0.115, +0.350] | +0.262 [+0.166, +0.371] | 7 / 8 | 7 / 8 |
| 217M RPE1 | +0.082 [+0.028, +0.145] | +0.145 [+0.055, +0.238] | 13 / 16 | 15 / 16 |
| 217M Adamson | **+0.005 [−0.057, +0.060]** | +0.133 [+0.067, +0.199] | 13 / 32 | 26 / 32 |
| 1B K562 | **+0.074 [−0.036, +0.184]** | +0.124 [+0.041, +0.213] | 7 / 12 | 10 / 12 |

- Variance alone clearly beats attention in 2 of 4 runs. It ties in Adamson. In 1B the
  interval includes zero.
- The 3-feature gene model beats attention in all 4 runs, and every interval excludes 0.
- The per-TF mean view (mean of per-TF AUROCs) gives the same picture. The one change:
  in Adamson, attention is slightly ahead of variance on that view (−0.020
  [−0.090, +0.044]), still a tie.
- The raw attention AUROC interval includes 0.5 in 3 of 4 runs. Only RPE1 (0.605
  [0.549, 0.655]) is above chance.
- The pooled gene-model AUROCs above are pooled out-of-fold values. The mean of the five
  fold AUROCs (the investigation's number) is 0.763 / 0.753 / 0.667 / 0.683.
- The test is small: 8–32 TFs and 45–236 positive pairs.

### 3a. A degree-preserving null that actually rewires

**Set-up.** The network is the 0/1 matrix of evaluated TFs × the 1,500 genes. A null
network keeps every TF's number of targets and every gene's number of listing TFs. The
cell (TF, its own gene) is never scored, so it may never receive an edge.

- **Curveball** (primary): pick two TF rows at random. Pool the targets that only one of
  them has. Re-deal them at random; each row keeps its count. A target that is the other
  TF's own gene stays put. 1,000 independent draws, each 5,000 trades from the observed
  network.
- **Checkerboard swaps** (independent check): swap the targets of two edges when the new
  cells are empty and allowed. 1,000 draws, 30 × (number of edges) accepted swaps each.
- Statistic: the deployed pooled attention AUROC.
- z = (observed − null mean) / null SD.
- p = (1 + number of null draws ≥ observed) / (1 + 1,000). This is the Phipson–Smyth
  form. It is one-sided: "attention is better than the null".
- BH is applied across the 4 runs.

Source: `outputs/v2_eval/curveball/curveball_summary.json`.

| Run | Observed | Null mean (SD) | z | p | BH q (4 runs) | Draws equal to observed | Mean Jaccard distance from observed |
|---|---|---|---|---|---|---|---|
| 217M K562 | 0.483 | 0.466 (0.036) | +0.47 | 0.32 | 0.32 | 0 / 1000 | 0.876 |
| 217M RPE1 | 0.605 | 0.574 (0.017) | +1.82 | 0.046 | 0.14 | 0 / 1000 | 0.877 |
| 217M Adamson | 0.520 | 0.496 (0.016) | +1.50 | 0.070 | 0.14 | 0 / 1000 | 0.935 |
| 1B K562 | 0.555 | 0.539 (0.022) | +0.77 | 0.22 | 0.29 | 0 / 1000 | 0.872 |

(Jaccard distance = 1 − shared edges / all edges; 0 = identical network, 1 = no edge in
common. Values near 0.9 mean the null networks share only about 1 edge in 8 with the
real one.)

**Checks (all passed).**
- Every draw kept all row and column counts, and no draw filled a forbidden cell.
- No draw equals the observed network. The smallest Jaccard distance is 0.73 / 0.82 /
  0.89 / 0.78.
- The variance baseline's AUROC is exactly unchanged in every draw (maximum change 0.0).
  This must hold because it depends only on the column counts, so the column counts are
  really preserved.
- Mixing: the mean Jaccard distance reaches its plateau after about 100–500 trades
  (`mixing_curve.csv`). 5,000 trades is far past that.
- The checkerboard null gives the same answer: null means 0.470 / 0.574 / 0.497 / 0.540,
  and z = +0.36 / +1.76 / +1.42 / +0.68.
- The secondary statistic (per-TF mean AUROC) gives z = +0.72 / +0.75 / +1.32 / +0.69.
  All BH q = 0.24.

**The deployed null, re-run verbatim.** It used 50 draws, 5 × (edges) trade attempts,
and picked row pairs from all 1,500 rows. Draws identical to the real network on the
evaluated TF rows: **49/50, 37/50, 10/50, 46/50**. On the whole matrix: 49, 31, 10, 41.
It reproduces the deployed z exactly (−0.14 / +0.17 / +0.16 / −0.04). It could also put
an edge on a TF's own gene (2 cases in Adamson, 3 in 1B). So the deployed "z ≈ 0" was
produced by a null that almost never changed the network.

**Reading.** At layer 8, no run beats the degree-preserving null after BH correction.
But z is not ≈ 0: it is +1.8 in RPE1 and +1.5 in Adamson. The null mean is well above
0.5 in RPE1 (0.574) and 1B (0.539). So most of RPE1's 0.605 comes from which TFs and
genes have many TRRUST edges, not from the specific pairs.

### 3b. Residualised attention

The attention score of every ordered gene pair (all 1,500 × 1,499) is predicted from
gene-level features only, with 5-fold cross-fitting over pairs (KFold, shuffled, seed
42, as deployed). The residual (observed − predicted) is then scored like raw attention.

- **OLS.** Features: source mean, source variance, target mean, target variance, target
  dropout. *[Corrected in verification, see V1 below: the deployed OLS ran in float32
  and did not reach the least-squares fit in K562, Adamson and 1B. The OLS columns now
  show the float64 fit. The float32 replica of the deployed number is in the last
  column.]*
- **HGB (new non-linear check).** Histogram gradient boosting on mean, variance and
  dropout of both genes.

95% CI = TF bootstrap, 2,000 resamples (fitted models held fixed).
Source: `outputs/v2_eval/residualised/residualised_summary.json` (keys `ols5_f64`,
`hgb6`; `ols5` = float32 replica of the deployed value).

| Run | Raw attention [95% CI] | Residualised, OLS float64 [95% CI] | Drop from raw, OLS [95% CI] | Residualised, HGB [95% CI] | Drop, HGB [95% CI] | Deployed OLS (float32) / GBDT |
|---|---|---|---|---|---|---|
| 217M K562 | 0.483 [0.398, 0.561] | 0.492 [0.406, 0.575] | −0.009 [−0.016, −0.002] | 0.491 [0.414, 0.566] | −0.008 [−0.024, +0.003] | 0.487 / 0.497 |
| 217M RPE1 | 0.605 [0.546, 0.655] | 0.492 [0.404, 0.568] | +0.112 [+0.078, +0.153] | 0.559 [0.488, 0.617] | +0.046 [+0.023, +0.076] | 0.492 / 0.519 |
| 217M Adamson | 0.520 [0.480, 0.561] | 0.534 [0.498, 0.570] | −0.014 [−0.022, −0.007] | 0.529 [0.490, 0.568] | −0.009 [−0.014, −0.004] | 0.520 / 0.532 |
| 1B K562 | 0.555 [0.463, 0.641] | 0.510 [0.417, 0.592] | +0.045 [+0.027, +0.075] | 0.525 [0.433, 0.604] | +0.031 [+0.007, +0.078] | 0.543 / 0.511 |

- The float32 OLS reproduces the deployed values exactly. But the float32 fit is not
  the least-squares fit in 3 of 4 runs (V1). The float64 numbers above are the correct
  OLS residuals.
- Residualising lowers attention in RPE1 (by 0.112) and in 1B (by 0.045, to 0.510).
  In 1B the residual keeps about 18% of the above-chance signal, not the 78% the
  deployed float32 number implied.
- It raises attention slightly in K562 (+0.009) and Adamson (+0.014).
- K562 is below chance before residualising, so nothing can "collapse".
- In RPE1 the result depends on the model. OLS gives 0.492, HGB gives 0.559
  [0.488, 0.617], and the deployed GBDT gave 0.519.
- Every residualised interval includes 0.5 (Adamson only just: [0.498, 0.570]).

### 3c. Supplement: other layers (not pre-specified)

Source: `outputs/v2_eval/layers/per_layer_trrust.csv`, `layers_summary.json`. These
tests use the same 1,000 Curveball networks as §3a, and layer 8 reproduces §3a exactly.

- Raw TRRUST AUROC is higher at early layers. Examples: 217M K562 L1 0.709; RPE1 L1
  0.758; Adamson L1 0.591; 1B L4 0.751.
- **Layer-corrected null test.** The statistic is the largest per-layer z over all
  layers. Its null is the same maximum, computed on each null network.

  | Run | p | BH q across runs | Best layer (z) |
  |---|---|---|---|
  | 217M K562 | 0.16 | 0.16 | — |
  | 217M RPE1 | **0.003** | **0.012** | L0 (z = 3.35) |
  | 217M Adamson | 0.027 | 0.054 | L1 (z = 2.50) |
  | 1B K562 | 0.14 | 0.16 | — |

- In 1B at L4, attention beats the 3-feature gene model (gap −0.071 [−0.130, −0.012]).
  But it does not beat the degree null there (z = 0.56).
- At RPE1 L0, the gene model is still level with attention (gap +0.052 [−0.010, +0.121]).
- Per-layer gene-model gaps are not corrected across layers. Residualisation was not
  run at other layers.

What this means: at the pre-specified layer the TRRUST result is negative. It is not
negative at every layer in RPE1. The paper should not say attention carries no
TF-specific TRRUST information at any layer.
*[Added in verification, see V4 below: co-expression (|Spearman| in the same control
cells, no model) also beats the same degree null in RPE1 (z = +3.1, p = 0.003) and in
1B (z = +2.2, p = 0.021). A score built from two gene means with no model,
−|mean(TF) − mean(gene)|, gets z = +1.8 in RPE1, the same as layer-8 attention. So
beating the degree null does not show knowledge specific to the model. The early-layer
result was not compared with co-expression.]*

---

## 4. What differs from the deployed claims

1. **Figure 2 shows Endpoint A (knockdown response), not TRRUST.** Its values are right
   (0.512/0.603/0.508/0.540 vs 0.596/0.766/0.712/0.596). They now have CIs.
2. **"6–20 AUROC points; p ≤ 1e-4 after correction"** is true for Endpoint A only.
   The gaps are +5.6 to +20.4 points. The correction was BH within each run over 13
   tests; BH across runs gives the same conclusion.
3. **"+0.002 at most" incremental value** is Endpoint A, logistic regression only, a
   mean over 5 folds, with no CI. It reproduces exactly. With a CI, the 1B gain
   (+0.0021 [+0.0010, +0.0031]) is above zero but tiny.
4. **"Curveball z ≈ 0"** came from a null that left the network unchanged in 49, 37, 10
   and 46 of 50 draws (evaluated rows). With a working null: z = +0.47, +1.82, +1.50,
   +0.77; BH q = 0.32, 0.14, 0.14, 0.29. The negative verdict survives at layer 8, but
   "z ≈ 0" is false for RPE1 and Adamson. The quoted "1,000 iterations, z ≥ 3" set-up
   did not exist in the runs; 50 draws were used.
5. **On TRRUST, variance alone does not beat attention in every run.** It wins in
   K562 and RPE1, ties in Adamson, and its interval includes zero in 1B. The 3-feature
   gene model wins in all 4.
   *[Added in verification (V3): the Adamson tie and the 1B interval depend on the
   double log. With gene variance on a single log scale, variance beats attention in
   Adamson (0.625 vs 0.520, gap CI [+0.059, +0.156]) and in 1B (0.682 vs 0.555, gap CI
   [+0.060, +0.203]).]*
6. **"Residualising collapses attention to chance in every run" is not accurate.** There
   is a real drop in RPE1 (model-dependent) and in 1B (0.555 → 0.510 with a correct
   float64 OLS). K562 starts below chance, and Adamson goes up slightly (0.520 → 0.534).
   Also, raw attention is not above chance in 3 of 4 runs to begin with.
   *[Corrected in verification (V1). The earlier text said "small drop in 1B" and
   "Adamson does not move"; both came from the deployed float32 OLS, which is not the
   least-squares fit.]*
7. **The 1B model is 4.84× larger, not 10×.** The 1B run used 200 cells, a different
   gene set (1,259/1,500 shared) and layer 8 of 20. It is not a clean scale comparison.
8. Only TRRUST was used as a curated reference.

## 5. What I did not do, and limits

- No model was run. So I could not re-extract 1B attention with 2,000 cells or on the
  217M gene set.
- The CIs for the gene model, the residualised scores and the incremental Δ hold the
  fitted models fixed. They leave out refitting noise, so they are somewhat too narrow.
- I did not re-run the deployed sklearn GBDT residualisation (about 10 min per run).
  I used HGB instead and list the deployed GBDT values next to it.
- The Curveball chain with forbidden cells is not proven to reach every allowed network.
  It agrees with an independent checkerboard sampler, so I treat it as adequate.
- DoRothEA, STRING and ChIP-based references were not tested.
- I did not check whether Adamson et al. 2016 is CRISPRi or CRISPRa. The code's loader
  calls it CRISPRa. I believe the source study is a CRISPRi screen, but I did not
  verify this from a file here.
- The external drive creates macOS `._*` metadata files next to every new file. Leave
  them out when packaging `task_data_rpe1/`.

## 6. Task-data folder for the controlled experiment (217M RPE1 only)

`outputs/v2_eval/task_data_rpe1/` (18.1 MB; built by `v2_10_task_data_rpe1.py`) holds
inputs only, with no results:

- `attention_layer8_headmean.npy`: the deployed aggregate; 1500 × 1500 float32; diagonal
  kept as stored.
- `pair_cell_counts.npy`: the number of cells behind each average.
- `genes.tsv`
- `gene_stats_control_cells.tsv`: mean, variance and dropout over the same 2,000
  control cells.
- `trrust_edges_in_gene_set.tsv`: 192 rows; 175 unique pairs; 3 self-rows, kept as in
  the source.
- `README.md`: how everything was made.
- `MANIFEST.sha256`

Checks (`task_data_rpe1_build_check.json`):
- Per-gene statistics were recomputed from the raw h5ad. The maximum difference from the
  deployed values is 3e-7.
- The gene set equals the top-1,500-variance vocabulary genes.
- Attention is never non-zero where the pair count is 0.
- The README describes the data only. It gives no hints about the traps.

## 7. Files

- Scripts: `scripts/v2_common.py`, `v2_01_facts.py` … `v2_11_run_config.py`.
- Outputs: `outputs/v2_eval/`. This holds `knockdown/`, `trrust/`, `curveball/`,
  `residualised/`, `layers/`, `de_labels/`, `fig_knockdown.csv`, `fig_trrust.csv`,
  `run_facts.json` and `run_config.json`.
- Figure CSV columns: run, method, auroc, ci_low, ci_high, n.
  - In `fig_knockdown.csv`, n = number of perturbations.
  - In `fig_trrust.csv`, n = number of TFs. The row `degree_preserving_null_mean`
    gives the null mean, and its interval is the central 95% of the 1,000 null draws
    (not a CI).

## Plain-words summary

- **Knockdown endpoint.** Gene variance predicts which genes respond to a knockdown
  better than attention in all four runs. The gaps are 5.6 to 20.4 AUROC points, and
  every CI is well above zero. Adding attention to the gene features changes the score
  by 0.002 or less. I rebuilt these results from the raw data and they match to the
  last digit.
- **TRRUST endpoint.** The test is small (8–32 TFs). A simple 3-feature gene model beats
  attention in all four runs. Variance alone wins in two runs, ties in Adamson, and is
  uncertain in 1B. *[Verification, V3: that split comes from variance computed on
  doubly-logged data. On a single log scale, variance wins in all four runs.]*
- **The old null.** It almost never shuffled the network, so its "z ≈ 0" meant nothing.
  A working null gives z from +0.5 to +1.8. None survives correction across runs at
  layer 8.
- **Residualising.** With a correct (float64) OLS fit it lowers attention in RPE1 and
  1B, and raises it slightly in K562 and Adamson. In RPE1 the size of the drop depends
  on the model used. The deployed OLS numbers were computed in float32 and were not the
  true least-squares fit in 3 of 4 runs. *[Corrected in verification, V1.]*
- **Model size.** The 1B model is 4.8 times larger, not 10. Its run used 200 cells and a
  shallower layer.
- **Other layers.** At early layers in RPE1, attention does beat the degree null after
  correcting for the choice of layer. So "no signal at all" is too strong. The negative
  verdict holds at the pre-specified layer. *[Verification, V4: co-expression with no
  model beats the same null about as strongly (z = +3.1 in RPE1). So this is not
  evidence of model-specific knowledge.]*

---

## Verification notes (independent check, 2026-10-01)

A second agent re-checked this item. CPU only; no model was run. Scripts:
`scripts/v2verify_01_endpointA.py` … `v2verify_10_run_config.py`. Outputs:
`outputs/v2_eval/verification/` (one JSON per check, plus `run_config.json` with the
sha256 of all 58 inputs; the 43 inputs shared with the v2 run_config have the same hashes).

**Verdict: OK after fixes for the numbers in this report, with two problems left open
(V3, V4).**

### What was re-computed and agrees

- **Endpoint A (knockdown).** I used the deployed labels (`phase3*/full_pair_dataset.csv`)
  and the stored layer-8 attention, and scored them with sklearn. The per-perturbation
  AUROCs equal the v2 files to within 2.2e-16 in all 4 runs. The counts match too
  (174/325/57/155 perturbations; 9,741/48,245/4,438/8,587 positive pairs), as do the
  counts of "variance > attention" (118/314/56/91) and the raw Wilcoxon p-values.
  My gap CIs use a perturbation bootstrap with 2,000 reps and a different seed:
  +0.084 [+0.061, +0.107], +0.163 [+0.153, +0.173], +0.204 [+0.174, +0.234] and
  +0.056 [+0.034, +0.079]. A normal-approximation interval gives the same.
- **Incremental Δ, 1B.** +0.0021, with 95% CI [+0.0009, +0.0031] by explicit
  perturbation bootstrap. I also refitted on 20 random perturbation-grouped splits.
  The SD across those refits is 0.00005, so refitting noise does not change the CI.
- **TRRUST endpoint.** TF and pair counts, raw attention, variance and the 3-feature
  model AUROCs are the same. I redid the TF bootstrap by copying the drawn TFs' pairs
  (seed 11). Its CIs agree with v2 to within 0.01. A leave-one-TF-out gene model gives
  the same conclusion: gaps +0.274 / +0.145 / +0.139 / +0.111, and every CI is above 0.
- **Degree-preserving null.**
  - z and p re-derived from the saved draws match exactly.
  - My own sampler is a boolean-matrix Curveball (1,000 trades, 1,000 draws, seed 23).
    It gives z = +0.40 / +1.73 / +1.40 / +0.76 and p = 0.33 / 0.047 / 0.074 / 0.21.
    v2 gave +0.47 / +1.82 / +1.50 / +0.77 and 0.32 / 0.046 / 0.070 / 0.22.
  - No draw was identical to the real network. Mean Jaccard distance was 0.87–0.94.
    Every draw kept all row and column sums.
  - The max-over-layers p is 0.16 / 0.005 / 0.036 / 0.14. v2 gave 0.16 / 0.003 / 0.027 / 0.14.
  - I ran the deployed null from its own source code with the deployed seeds. It left
    the evaluated rows unchanged in 49 / 37 / 10 / 46 of 50 draws, and it created
    2 self-loops in Adamson and 3 in 1B. Both confirmed.
- **Model facts.**
  - Parameters: 216,946,576 vs 1,049,036,544. The ratio is 4.84.
  - The 1B run used 200 cells, has 20 stored layers and used primary layer 8.
  - It shares 1,259 of 1,500 genes with the 217M K562 run.
  - All confirmed.
- **Figure CSVs and task-data folder.**
  - The CSV values equal the JSONs.
  - `MANIFEST.sha256` verifies.
  - The attention and pair-count arrays are bit-identical to `phase0_rpe1`.
  - The gene statistics are equal, and the folder is 18.1 MB.
  - The README holds no results.
  - The README's account of how the data were made matches `phase0_rpe1.py`. RPE1 X is raw counts, so the README is right for this run.

### V1 — fixed: the deployed OLS residualisation ran in float32 and was not the OLS fit

- **Evidence** (`verification/ols_precision_check.json`). A least-squares fit has the
  highest training R² of any linear fit. The float32 fit has a lower training R² than the
  float64 fit in every fold of 3 runs:
  - K562: 0.00046 vs 0.00049
  - Adamson: 0.00087 vs 0.00139
  - 1B: 0.00011 vs 0.00036
  - RPE1: equal
- So the float32 result is not the OLS fit. Two float64 routes agree exactly (sklearn
  float64 and numpy `lstsq`). The residualised TRRUST AUROC is:
  - 0.492 in K562 (deployed 0.487)
  - 0.492 in RPE1 (deployed 0.492)
  - 0.534 in Adamson (deployed 0.520)
  - 0.510 in 1B (deployed 0.543)
- **Changes made.**
  - `v2_07_residualise.py`: added variant `ols5_f64`. It ran and its key was added to
    each `resid_*.json`. The `ols5` (float32 replica) and `hgb6` results are unchanged.
    `residualised_summary.json` was re-combined.
  - `v2_09_figdata.py` / `fig_trrust.csv`: the row `attention_L8_residualised_ols` now
    holds the float64 value. The float32 replica is kept as a separate row,
    `attention_L8_residualised_ols_float32_deployed_replica`. Do not plot that row.
  - This report: the §3b table and bullets, §4 item 6 and the plain-words summary.
  - `run_config.json` was refreshed with `v2_11_run_config.py`.
- **Consequence outside this item.**
  - The claim "1B keeps 78% of its signal; its verdict file passes C3"
    (attn_endpoints.md; REVISION_PLAN E7) comes from the float32 error.
  - With a correct OLS, 1B keeps about 18%: 0.510 [0.417, 0.592]. The deployed GBDT
    (0.511) agrees with this.
  - Adamson still does not "collapse". It rises to 0.534.
- The agent's own summary listed "Residualised attention (OLS) is 0.487 / 0.492 / 0.520
  / 0.543". That line is superseded.

### V2 — fixed: stale run_config

After the V1 changes I re-ran `v2_11_run_config.py`, so its hashes cover the new files.

### V3 — open: 3 of 4 runs started from log-normalised data, not counts

`verification/input_scale_check.json`, `single_log_trrust_check.json`.

- **What the files hold.** For 217M K562, 1B K562 and Adamson, the h5ad X holds
  log1p(counts per 10k): no whole numbers, and each cell's sum of expm1(X) is about
  9,400 (K562) or 5,200 (Adamson). RPE1 X holds raw integer counts.
- **What the code assumed.** The deployed code treats X as counts everywhere.
  `phase0_extract.py:182` even labels it "raw counts".
- **(a) Model input.** The tokeniser ranks genes by X divided by the gene median. On log
  values this order is not the count-based order MaxToki expects:
  - K562: the kept-gene sets overlap by 91%, and the rank Spearman is 0.85 (lowest 0.71).
  - Adamson: the rank Spearman is 0.76 (lowest 0.52).
  - So in 3 of 4 runs the model did not get its intended rank-value input. This cannot
    be fixed without re-extracting attention, which needs a model run.
- **(b) Gene features and gene choice.** These were computed after a second log.
  Variance on a single log vs the deployed variance: Spearman 0.47 (K562),
  0.62 (Adamson), 0.53 (1B).
- **(c) Knockdown labels.** The |Δ| ≥ 0.5 rule was applied to doubly-logged values.
  - In 217M K562, the same rule on a single log keeps only 19 perturbations
    (106 positive pairs) instead of 174 (9,741).
  - The gap still favours variance: +0.254 [+0.133, +0.367] (perturbation bootstrap,
    2,000 reps). Attention is 0.373 there.
  - So the direction holds, but the stated threshold does not mean what it says in 3 runs.
- **(d) TRRUST variance baseline on a single log.** Gap CIs are TF bootstrap, 2,000 reps.
  - K562: 0.638 (gap CI [+0.015, +0.297])
  - Adamson: 0.625 ([+0.059, +0.156])
  - 1B: 0.682 ([+0.060, +0.203])
  - So "variance ties attention in Adamson" and "the 1B interval includes zero" depend
    on the double log. On a single log, variance alone beats attention in all 4 runs.
- **What I changed.** I added a pointer to §2 (step 3) and §4 item 5. I did not redo the
  whole report on a single log, because the model input in these runs is affected too.

### V4 — open: passing the degree null is not evidence of knowledge specific to the model

`verification/null_check.json`. These use the same null networks (my sampler).

- **|Spearman| co-expression in the same control cells:** z = −0.52 / +3.09 / +0.31 /
  +2.20, with p = 0.71 / 0.003 / 0.39 / 0.021.
- **A score with no model, −|mean(TF) − mean(gene)|:** z = +0.80 / +1.81 / +0.67 /
  +1.48, with p = 0.21 / 0.041 / 0.26 / 0.068.
- **What this means.**
  - In RPE1, co-expression beats the degree null as strongly as the best layer of
    attention (L0, z = +3.3).
  - Two gene means already match layer-8 attention (z = +1.8).
  - The layer supplement (§3c) has no co-expression control.
- **Suggested wording:** "beyond degree, attention carries pair-specific signal of about
  the same size as co-expression from the same cells". Avoid wording that implies model
  knowledge.

### V5 — note: which null z values to quote

attn_endpoints.md and REVISION_PLAN E6 quote z = +0.55 / +1.87 / +2.00 / +1.13. That
quick fix rewired every TF row with edges and used 200 draws. This item restricted the
null to the evaluated TFs and used 1,000 draws, as the item asked. Quote the v2 values:
+0.47 / +1.82 / +1.50 / +0.77.

### V6 — minor, not changed

- §3 and §3b give slightly different CIs for raw attention (e.g. K562 [0.399, 0.560]
  vs [0.398, 0.561]). They use different bootstrap seeds. This is not an error.
- The TF bootstrap treats TFs as independent. The same candidate genes recur across TFs,
  so a two-way bootstrap (TFs and genes) would give wider intervals.
- Adamson CRISPRi vs CRISPRa is still not verified from a file. The loader's docstring
  says CRISPRa.

### Plain-words summary of the verification

- Most of this report checks out. I rebuilt the knockdown results, the TRRUST baselines,
  the fixed null and the model facts with my own code, and they match.
- One number type was wrong. The old "residualised" OLS score was computed in low
  precision and was not a real least-squares fit. Fixed:
  - 1B now drops to 0.510 (keeps about 18%, not 78%).
  - Adamson goes up to 0.534.
  - K562 goes to 0.492.
- New problem 1: three of the four runs fed the model log-transformed values as if they
  were counts. So the model input was not what MaxToki expects. The gene statistics and
  knockdown labels were also logged twice. The main direction still holds, since gene
  variance beats attention. But the label counts and the "tie in Adamson" do not hold up.
- New problem 2: plain co-expression beats the degree-preserving null as well as
  attention does. So passing that null does not show the model knows anything special.
