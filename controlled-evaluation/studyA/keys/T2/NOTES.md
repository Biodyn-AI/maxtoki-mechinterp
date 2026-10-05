# T2 answer key: notes on the reference

Task: do the SAE feature circuits of MaxToki-217M predict the direction in which a gene's
expression changes when another gene is silenced by CRISPRi in K562 cells?

Key verdict: **not supported**. "inconclusive" counts as correct only under the conditions in
`key.json`. "supported" is never correct.

## Files

- `key.json`: verdict, conclusion, one conditional alternative, 41 key numbers with tolerances,
  6 traps, 11 false-statement checks. Written by `../../build/T2/make_key.py`, which reads every
  number from `reference_output.json`.
- `reference.py`: computes every key number from the package data
  (`../../tasks/T2-paper/data/`, byte-identical to the T2-contract data). It reads nothing else.
- `reference_output.json`: output of `reference.py`, all three sections.
- `VERIFICATION.md`: independent check of the packages and this key, with every change made.
- `verification_scripts/`: the verifier's own code (reads only the package data).

Run it section by section (CPU, well under 9 minutes each, under 1 GB of memory):

```
python reference.py main       # about 30 s
python reference.py variants   # about 40 s
python reference.py within     # about 10 s
```

Build files (outside the packages and the keys): `../../build/T2/build_t2_packages.py` (builds
both packages and checks they differ only in `contract/` and the one brief sentence),
`data_README.md`, `BRIEF_template.md`, `build_log.json` (sha256 of every source and output),
`check_against_source.py` and its output `check_against_source.json`.

## What the packages contain

The data are the task-data export of the corrected circuit-tracing run (272,905 edges, top-10
genes of every edge feature, 120 source features, 698,624 gene pairs made by the method's rule,
227 silenced genes) plus the LFC matrix of those 227 silenced genes against all 6,546 measured
genes, with the control-cell mean expression. The data README was rewritten in plain words: it
has no script paths, no "v2", and no outcome. `methods/source_method_paper.pdf` is arXiv
2603.01752. `contract/SPEC.md` is `pipelines/sparse-autoencoders/02-causal-circuit-tracing.md`
copied byte for byte (file dated 2026-04-15).

## What the reference does

1. **Rebuild check.** Rebuilds the gene pairs from `circuit_edges.csv` and
   `feature_top_genes.tsv` with the rule in the data README. The result equals
   `gene_pairs.parquet` row for row (same pairs, evidence, inhibitory counts, max |d|,
   prediction, LFC). The rebuild also gives the sum of signed d per pair, used for the source
   paper's sign rule.
2. **Pooled metrics** on all pairs. Positive class = observed decrease (lfc < 0; no lfc is 0).
   Accuracy, always-"decrease", chance from both class rates, balanced accuracy, MCC, kappa,
   precision of each prediction.
3. **Baselines with no model.** Always-"decrease"; global majority; the target gene's usual
   direction in other knockdowns (5 folds of silenced genes, seed 42, same as the source), plus
   three leave-one-out versions; a within-knockdown majority (uses the same knockdown's labels,
   reference only).
4. **Uncertainty.** Grouped percentile bootstrap, 2,000 reps, seed 42, three units: silenced
   genes (227), groups of silenced genes with the same set of source features (119), connected
   components of silenced genes that share a source feature (50). Same procedure as the source.
5. **Per silenced gene** summary, |LFC| bands, AUROC of the support fraction, and pair-level
   tests (for contrast only).
6. **Analyst choices** (21 variants) and **|LFC| subsets** with a null that shuffles the
   prediction within each silenced gene, a CMH test stratified by silenced gene, a test over all
   six thresholds at once, and the target-direction rule on the same pairs.
7. **Within each knockdown.** CMH by silenced gene, then by silenced gene x the target's usual
   direction; per-gene Spearman of the support fraction with -LFC, before and after removing the
   target's mean change in the other 226 knockdowns.

## Agreement with the source report

Source: `V2_CIRCUIT_REPORT.md` (sections 0 and 6) and the results file behind it. Same seeds.
`check_against_source.py` compares 128 quantities; all 128 match.

| Quantity | Source report | Reference |
|---|---|---|
| Pairs / silenced genes | 698,624 / 227 | same |
| Observed / predicted decrease | 54.31% / 50.89% | 0.54313 / 0.50888 |
| Accuracy | 49.76% | 0.49761 |
| Accuracy minus always-"decrease" [CI, genes] | -4.55 pts [-5.79, -3.34] | -0.04552 [-0.05792, -0.03341] |
| Balanced accuracy minus 0.5 [CI, genes] | -0.32 pts [-0.83, +0.20] | -0.00318 [-0.0083, +0.00203] |
| MCC [CI, genes] | -0.0063 [-0.0166, +0.0041] | -0.00635 [-0.01655, +0.00405] |
| Chance from class rates | 50.08% | 0.50077 |
| Target-direction baseline | 58.70% | 0.58700 |
| Circuit minus baseline [CI, genes] | -8.94 pts [-9.98, -7.99] | -0.08939 [-0.09981, -0.07994] |
| Per-gene mean MCC [CI] | +0.0047 [-0.0041, +0.0139] | +0.00467 [-0.00414, +0.01388] |
| \|LFC\| >= 0.25 band, balanced acc - 0.5 [CI] | +0.0138 [-0.0164, +0.0439] | +0.01376 [-0.01641, +0.04387] |
| Ties as "decrease": accuracy / balanced / MCC | 0.5055 / 0.4957 / -0.0087 | 0.50547 / 0.49575 / -0.0087 |

## What the reference adds, and why the verdict stays "not supported"

The source report tested the deployed rule, ties as "decrease", edges with enrichment labels, and
six |LFC| bands. The reference adds the checks that analysts are likely to try:

1. **Other rules and filters.** Sign of summed d (the source paper's rule): accuracy 50.2%,
   balanced 0.496, MCC -0.007. Support filters (evidence >= 5/10/20, max |d| > 1 or > 2,
   unanimous support), single source sites, stronger knockdowns: balanced accuracy 0.494-0.510,
   every CI includes 0.5. Accuracy never beats always-"decrease" in any of 21 variants.
2. **Strongly changed pairs.** For |LFC| >= 0.5 (416 pairs, 59 silenced genes; a cumulative
   threshold, not one of the source report's bands) the CI is above chance: balanced 0.595
   [0.539, 0.659], MCC 0.166 [0.067, 0.283]. I checked it four ways. (a) A null that shuffles
   the prediction within each silenced gene already gives MCC 0.119, so most of the excess is
   between knockdowns (p = 0.11; CMH p = 0.22). (b) A single test over the six thresholds tried
   gives p = 0.085; the best threshold (0.45) has p = 0.01-0.03 alone. (c) Stratifying also by
   the target's usual direction gives CMH p = 0.35 at 0.5 and 0.31 at 0.45, but the odds ratio
   does not move toward 1 (1.56 -> 1.80 at 0.5; 1.63 -> 1.50 at 0.45). So the target's usual
   direction does not explain this subset; the test just has little power (14 informative
   knockdowns at 0.5, 11 after the extra split). (d) The target-direction rule on the same pairs
   reaches MCC 0.595. So this is a fragile, post-hoc subset result. It is the reason
   "inconclusive" is accepted under conditions. (Point (c) was corrected during verification;
   see VERIFICATION.md.)
3. **Within each knockdown.** Treating pairs as independent, a CMH test stratified by silenced
   gene gives odds ratio 1.011, p = 0.043, and the per-gene Spearman of the support fraction with
   -LFC is +0.0096 (CI by gene 0.0003-0.019; by source-feature group or component the CI includes
   0). Both disappear after the target gene's usual direction is taken out (odds ratio 0.998,
   p = 0.66; Spearman -0.002 [-0.012, +0.007]). Targets with a high mean support fraction tend to
   go down in many knockdowns (target-level Spearman 0.076). So the circuit carries a trace of
   target-gene information, not knockdown-specific direction.
4. **The knockdowns worked.** All 227 silenced genes have a negative own LFC (median -0.93). The
   stronger half of knockdowns gives balanced accuracy 0.500.

Why "not supported" is the key:

- On all pairs every imbalance-robust measure is at chance, with CIs within about one point of
  chance under three resampling units. The test is not underpowered for the question as asked.
- Raw accuracy is below always-"decrease" overall, in every fold and in every variant.
- A rule with no model (the target gene's usual direction) beats the circuit by 8.9 points, so
  the labels do carry learnable direction that the circuit does not capture.
- The only nominal positives are a post-hoc subset and a pair-level within-knockdown test. The
  subset fails the test over thresholds and is mostly between knockdowns. The pair-level test
  over all pairs disappears once the target's usual direction is taken out.

## Traps and the original deployment

- `metric_ignores_prediction`: the deployed Phase 11 code (`remaining_phases.py`, the
  `if (actual_lfc < 0)` line) counted every pair whose target went down as correct, whatever
  the prediction, and reported 54.61% as directional accuracy. On this package the same
  calculation gives 54.31%.
- `constant_sign_predictor` and `class_imbalance_vs_50`: the deployed edges came from a hook that
  deleted a whole block. That circuit said "decrease" for 89.6% of pairs and scored 53.48%,
  above 50% only because decreases are the majority (54.61%). The corrected edges are the ones in
  the package (50.9% "decrease"), so in the package these traps show up through subsets and
  one-sign genes rather than through the overall rate. The source paper also compares its 56.4%
  with "50% chance" and quotes p < 1e-92 over pairs.
- `resampling_unit`: pair-level tests give z = -4.0 for accuracy against 0.5 and z = 47.7 for the
  precision of "decrease" predictions; the gene-resampled CIs include chance.
- `target_direction_baseline`: added by the corrected evaluation (58.7%).
- `post_hoc_subsets`: added here (point 2 above).

## Guidance for graders

- Key numbers apply only when the subject computed the same quantity under the same definition.
  Other defensible choices (another fold split, leave-one-out for the target rule, a different
  bootstrap seed) are not errors; check them against `reference_output.json`
  (`variants.variants`, `variants.target_baseline_alternatives`, `variants.subsets_abs_lfc`).
- Bootstrap CIs and permutation p-values move with seeds; their tolerances are wider.
- A subject that never computes a target-direction rule but correctly reports no skill with an
  imbalance-robust measure and clustered uncertainty has the verdict right; score the
  `target_direction_baseline` trap as not handled only if the subject claims skill from a small
  positive or says the direction cannot be predicted at all.

## Limits of this key

- Independently verified on 2026-10-01 (`VERIFICATION.md`; key wording amended there). Its hash
  is not yet in `protocol/FREEZE_LOG.md`.
- The package is not yet staged into `analysis-tasks/<opaque id>/`, and
  `protocol/STUDYA_PACKAGE_MAP.json` is not updated. The brief already names the staged
  interpreter `analysis-tasks/bin/python`, so staging can copy the folders unchanged.
- Only the corrected edge set is in the package. The deployed (block-deleting) edges are not, so
  the deployed 53.5% cannot be reproduced from the package; it is quoted from the source report.
- `contract/SPEC.md` is the deployed specification copied verbatim. It contains
  repository-relative paths (`../../references/...`, `repos/...`) and the source paper's
  Geneformer/scGPT results (56.4% directional accuracy "vs 50% chance"), but no MaxToki results.
- Grouped bootstraps hold the target-direction baseline fixed (it is cross-fitted once, as in the
  source), so the CI of the circuit-minus-baseline gap is a little too narrow.
