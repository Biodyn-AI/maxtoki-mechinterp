# T1 answer key: notes on the reference

Task: do MaxToki-217M's attention maps (layer 8, mean over heads, RPE1 control cells)
encode regulator-to-target relationships, with TRRUST as ground truth?

Key verdict: **not supported**. "inconclusive" counts as correct only under the
conditions in `key.json`. "supported" is never correct.

## Files

- `key.json`: verdict, conclusion, acceptable alternative, 29 key numbers with
  tolerances, 6 traps, 10 false-statement checks.
- `reference.py`: computes every key number from the package data
  (`../../tasks/T1-paper/data/`, byte-identical to the T1-contract data). It reads
  nothing else.
- `reference_output.json`: the output of `reference.py`, all five sections.
- `VERIFICATION.md`: independent re-check of the packages and this key, with every
  change made to the key.

Run it section by section (each well under 9 minutes on CPU):

```
bin/python reference.py main       # about 30 s
bin/python reference.py null       # about 60 s
bin/python reference.py null_alt   # about 15 s
bin/python reference.py resid      # about 20 s
bin/python reference.py incr_hgb   # about 1-3 min
```

## What the reference does

1. **Pair set.** Merge duplicate TRRUST rows (192 rows, 175 unique pairs), drop the
   2 self-pairs (173 edges left). Keep TFs with at least 3 targets in the gene set
   (16 TFs). For each TF, the candidates are the other 1,499 genes; positives are its
   TRRUST targets (138), the rest are negatives (23,846). The diagonal of the attention
   matrix is never scored. This is the deployed rule.
2. **Scores on the same pairs.** Attention with the TF as query (the deployed score),
   with the TF as key, symmetrised (mean and max of both directions); candidate variance,
   mean, 1 - dropout; a logistic model on candidate mean, variance and dropout with
   5-fold GroupKFold by TF; two model-free pair scores (expression proximity
   -|mean(TF) - mean(candidate)|, and co-detection lift
   C[TF, cand] x 2000 / (C[TF, TF] x C[cand, cand]), where C = pair_cell_counts).
3. **AUROC.** Mann-Whitney with average ranks. Pooled over all pairs, and as the mean
   of per-TF AUROCs.
4. **Uncertainty.** Percentile bootstrap over the 16 TFs, 2,000 resamples (a TF drawn
   k times counts k times). For contrast: a pair bootstrap, a two-way bootstrap over
   TFs and candidate genes, and the Hanley-McNeil standard error.
5. **Nulls.** (a) Degree-preserving: Curveball, rows = 16 TFs, columns = 1,500 genes,
   row and column sums kept, (TF, own gene) never an edge; 1,000 independent chains of
   5,000 trades from the observed network. Every draw is checked for margins; none
   equals the observed network (mean Jaccard distance 0.88). z = (observed - null
   mean) / null SD; p = (1 + #null >= observed) / 1,001, one-sided. (b) For contrast:
   uniform label permutation, and a row-only null (TF degree kept, targets uniform).
6. **Residualisation.** Attention of every ordered pair is predicted from gene features
   of both genes with 5-fold cross-fitting over pairs (OLS in float64 on 5 features;
   gradient-boosted trees on 6 features); the residual is scored like raw attention.
   Also OLS on symmetric features for the symmetrised score.
7. **Incremental value.** Out-of-fold AUROC of gene features + attention minus gene
   features alone; logistic or boosted-tree gene model; folds grouped by TF or by
   candidate gene. CI = TF bootstrap with the fitted models held fixed (so somewhat too
   narrow).

## Agreement with the source report

Source: the corrected re-evaluation of this run (V2 evaluation report, sections 3,
3a, 3b and verification notes V1 and V4). Same seeds were used, and every quantity
the report gives for 217M RPE1 is reproduced exactly from the package data:

| Quantity | Source report | Reference |
|---|---|---|
| TFs / positives / negatives | 16 / 138 / 23,846 | same |
| Attention AUROC [TF-bootstrap CI] | 0.605 [0.549, 0.655] | 0.6045 [0.5487, 0.6547] |
| Variance AUROC | 0.686 [0.634, 0.731] | 0.6864 [0.6335, 0.7311] |
| 3-feature gene model | 0.749 [0.691, 0.801] | 0.7494 [0.6909, 0.8011] |
| Variance - attention | +0.082 [+0.028, +0.145] | +0.0819 [+0.0281, +0.1451] |
| Gene model - attention | +0.145 [+0.055, +0.238] | +0.1449 [+0.0554, +0.2377] |
| Curveball null mean (SD), z, p | 0.574 (0.017), +1.82, 0.046 | 0.5739 (0.0168), +1.817, 0.0460 |
| Per-TF statistic z | +0.75 | +0.750 |
| Residualised OLS (float64) | 0.492 [0.404, 0.568] | 0.4922 [0.4040, 0.5685] |
| Residualised boosted trees | 0.559 [0.488, 0.617] | 0.5586 [0.4884, 0.6168] |
| Expression proximity vs degree null | z = +1.81 (other sampler) | z = +1.85 |

The package's gene statistics differ from the deployed ones by at most 3e-7, which does
not change any rounded number.

## What the reference adds, and why the verdict stays "not supported"

The source report only scored the deployed direction (TF as query). Three extra checks
were added, because analysts will try them:

1. **Symmetrised attention is stronger.** The mean of both directions gives 0.638
   [0.578, 0.701]. Against the same degree-preserving null it gives z = 3.3
   (p = 0.001; per-TF z = 2.3). After OLS residualisation it keeps 0.551
   [0.489, 0.611], 37% of its above-chance signal. So some pair-specific excess over
   target popularity exists for this variant. This does not contradict the source
   report, which did not test this variant. It is the reason "inconclusive" is
   accepted under conditions.
2. **A model-free pair score from the same cells does as well.** Co-detection lift,
   built only from `pair_cell_counts`, beats the same null with z = 3.1 (per-TF
   z = 3.0). Expression proximity matches the deployed score (z = 1.8). The source
   verification found |Spearman| co-expression gives z = 3.1 in this run. The package
   has no expression matrix, so Spearman co-expression cannot be computed by the
   subjects; the reference does not compute it either.
3. **Incremental value is not robust.** Adding attention to a logistic gene model gives
   +0.013 [+0.0005, +0.025] (folds by TF), +0.009 [-0.002, +0.019] (folds by gene),
   and -0.015 [-0.035, +0.003] with a boosted-tree gene model (whose gene-only AUROC is
   0.803 when folds are grouped by TF).
   *[Added in verification.]* The boosted-tree result moves with its settings and seed.
   With folds by TF it stays negative (-0.015 to -0.037). With folds by gene it ranges
   from -0.013 to +0.046. There the boosted gene-only model is weak (0.64-0.68), and
   gene + attention (0.66-0.71) is still below the logistic gene-only model (0.75). So
   a positive boosted-tree gain with folds by gene is not evidence that attention adds
   value. Judge such claims against the strongest gene-only model.

Why "not supported" is the key:

- No attention variant beats the gene-level model (0.749). The best variant only ties
  candidate variance (symmetrised: gap +0.048 [-0.021, +0.110]).
- For the provided score, about 70% of the excess over 0.5 is reproduced by a
  degree-preserving null. What remains is borderline (z = 1.8 pooled, 0.75 per TF)
  and vanishes after OLS residualisation.
- Where an attention variant does beat the degree null, a score with no model beats it
  about as strongly. So that excess is not evidence that attention encodes
  regulator-to-target relationships rather than co-occurrence or co-expression.
- The method specification's own pass rule (contract arm) also gives a fail: attention
  does not beat every gene-level baseline, and no residualised score keeps 80% of its
  signal.

## Traps and the original deployment

- The deployment's degree-preserving null left the evaluated network unchanged in 37 of
  50 draws (it picked row pairs from all 1,500 rows and used few trades). It reported
  z close to 0 for this run. A working null gives z = 1.8. The deployed verdict was
  negative, but partly for this wrong reason. This is trap `degree_preserving_null`.
- The deployment said residualising "collapses attention to chance in every run". In
  this run that holds for OLS (0.492) but not clearly for boosted trees
  (0.559 [0.488, 0.617]). This is part of trap `forking_paths`.
- The source verification (V4) found that co-expression beats the degree null as
  strongly as attention. This is trap `model_free_pair_signal`.
- Traps `gene_level_confound`, `resampling_unit` and `pair_set_definition` are general
  pitfalls of this design. The two-way TF x gene bootstrap is new here; it widens the
  attention CI to [0.509, 0.699] and the variance - attention gap to [-0.04, +0.20].
  Graders should accept TF-level or two-way resampling. Pairs treated as independent is
  the mishandling.
  *[Changed in verification.]* In these data a pair bootstrap gives almost the same
  attention CI as the TF bootstrap (0.547-0.655 vs 0.549-0.655). It is the candidate
  genes, not the TFs, that widen the interval. So a pair bootstrap is a mishandling only
  when the dependence is not acknowledged and pair-level precision carries a strong
  claim. `key.json` now says this.

## Guidance for graders

- Key numbers apply only when the subject computed the same quantity under the same
  definition. Different, defensible choices (for example a different TF threshold) are
  not errors; check them against the analyst-choice values in
  `reference_output.json` (`main.analyst_choices`).
- Monte Carlo quantities (null mean, SD, z, p) have wider tolerances because a
  different sampler or seed moves them (an independent sampler in the source gave
  z = 1.73 and p = 0.047).

## Limits of this key

- Only layer 8 (head mean) is in the package. The source report found that some
  earlier layers beat the degree null in RPE1 (best: layer 0, z = 3.35), but
  co-expression does too. The key's verdict is about the provided data.
- The bootstrap CIs for fitted models hold the models fixed, so they leave out
  refitting noise.
- The contract file `contract/SPEC.md` is the deployed specification copied verbatim.
  It contains paths from the original repository (`references/...`, `repos/...`,
  `src/...`) and results for scGPT and Geneformer, but no MaxToki results.
