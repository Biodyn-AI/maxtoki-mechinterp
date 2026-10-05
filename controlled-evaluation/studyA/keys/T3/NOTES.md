# T3 answer key: notes on the reference

Task: are the layer-5 SAE features of MaxToki-217M that respond to a transcription factor's (TF's) CRISPRi
knockdown in K562 specific to that TF's target genes? Ground truth: TRRUST v2 and DoRothEA (ChIP-seq flag).

Key verdict: **not supported**. No alternative verdict counts as correct (reasons in `key.json`,
`acceptable_alternatives_note`).

## Files

- `key.json`: verdict, conclusion, 51 key numbers with tolerances, 6 traps, 13 false-statement checks.
  Written by `../../build/T3/make_key.py`, which reads every number from `reference_output.json`.
- `reference.py`: computes every key number from the package data only (`../../tasks/T3-paper/data/`;
  the T3-contract data are byte-identical). One run: about 3 minutes on CPU, peak memory about 2 GB.
  `bin/python studyA/keys/T3/reference.py` (optional argument: path to a package folder).
- `reference_output.json`: the output of `reference.py`. Fully deterministic (fixed seeds; two runs gave
  identical files apart from the wall time).

## What the reference does

1. **Selection (unit = cell).** For each of the 87 TFs and each of the 4,928 features: two-sided
   Mann-Whitney U of the per-cell value (`mean_gene`), knockdown cells vs the 400 `ref` cells; BH across
   features; a feature responds if q < 0.05 and |mean(kd) - mean(ref)| > cut-off. Cut-offs 0.5 (the
   source paper's value), 0.25, 0.1, 0.05, 0.02, 0.01, 0.
2. **Statistic.** M = largest overlap of a responding feature's top-20 genes with the TF's target set
   (ChIP = DoRothEA pairs with `chip_flag`; TRRUST). The source rule "specific if M >= t" for t = 2..5
   is one statistic, T = min(M, 5) if M >= 2 else 0. The uncapped M is also tested.
3. **Nulls (exact).** `rf`: K random catalog features (the source paper's null). `cm`: every top-20 gene
   swapped for a random universe gene from the same detection-count bin (edges 1, 2, 5, 10, 20, 50, 100,
   200, 500), computed by convolving hypergeometric distributions. `cml`: bins = count bin x gene-length
   tertile (cuts 21,375 and 56,220 bp; genes without a length go to the middle tertile).
4. **Specificity rank.** Same responding features, `cm` p-value for every TF set (ChIP: 291 sets with
   >= 20 targets in the universe; TRRUST: 109 sets with >= 5); mid-rank of the true TF.
5. **BH across TFs** per cut-off, database and null (TFs with no responding feature enter with p = 1).
6. **GATA1 details.** Cell bootstrap of effects (kd and ref resampled separately, 1,000 resamples);
   feature 2610; other-knockdown null; 300 random groups of 95 `pool` cells with the whole selection
   re-run (the degenerate random-group null) and a K-matched version; planted-target power (200 repeats
   per k, count-matched null).
7. **Naive analyses** that the traps produce: uniform hypergeometric tests per feature and on the pooled
   union of top genes; the GATA1 set scored on other knockdowns' features; a pooled count-matched z-test.
8. **Extra checks** written for the key: feature 2610's gene rarity and its overlap with all 291 ChIP sets;
   the count-matched null with coarse decile bins; a cut-off-free rank check (Spearman correlation across
   features between |effect| and target overlap, true TF's set ranked among 291 sets, 20 TFs, bootstrap
   over TFs); Wilson intervals for power; testable TFs per cut-off.
9. **The source paper's own test, read literally** (Phase 8 step 5 of `contract/SPEC.md`): one-sided Fisher
   exact test of responding vs all other catalog features x "the feature's top-20 overlaps the TF's targets"
   (overlap >= 1, and >= 2), BH across TFs. Then the same test on the same features with each of the 291
   ChIP sets, and the rank of the TF's own set, for every setting. Also how rare the responding features'
   genes are (median detection count of each feature's top genes).

## Agreement with the source report

The key agrees with the corrected analysis in
`biomi_automation/projects/maxtoki/runs/sae-atlas-217M/V2_TF_SPECIFICITY_REPORT.md`.

- **Exact parts match exactly.** `../../build/T3/check_against_source.py` runs steps 1 to 3 on the package
  data and compares with the source `task_data/tf_tests.tsv`: all 546 TF x cut-off x database rows have the
  same K, M and T, and the same p_rf, p_cm, p_cml (capped and uncapped) and E_M_cm (largest absolute
  difference 2.2e-16). GATA1 at 0.5: features 628, 1334, 2006, 2627, 3167; M = 3 (feature 3167);
  p_rf 0.257, p_cm 0.371, p_cml 0.564; rank 151/291. CEBPZ (ChIP, 0.01): K 7, M 4, p_cm 0.030,
  q 0.61, rank 15/291. 0 TFs pass BH under cm or cml at any cut-off. Feature 2610: expected overlap 4.44
  (count-matched), P(>= 8) 0.037 (0.072 with length), 69 of 291 ChIP sets overlap it by >= 5 (the
  investigation behind the report found the same 69).
- **Monte-Carlo parts differ only by random draws** (different seeds or fewer draws):

  | quantity | reference | source report |
  |---|---|---|
  | 2610 effect, 95% CI | 0.461 [0.408, 0.511] | 0.461 [0.412, 0.512] |
  | 2610 above 0.5 in bootstrap | 7.2% | 6% |
  | random groups with a responding feature, cut-off >= 0.05 | 0 of 300 | 0 of 2,000 (at >= 0.25) |
  | floor p of the random-group null | 1/301 = 0.0033 | 1/2,001 = 0.0005 |
  | K-matched random groups, GATA1 p | 0.860 | 0.864 |
  | planted power, k = 8, cm, capped | 0.95 [0.91, 0.97] | 0.93 [0.88, 0.95] |
  | planted power, k = 4 / k = 2 | 0.985 / 0.20 | 0.99 / 0.18 |
  | random groups with >= 1 BH feature, no cut-off | 7% (300 groups of 95) | 10.9% (fixed pool, all group sizes) |

  The last row comes from reusing one `pool` and one `ref` group; the report shows fresh random splits
  give 3.9%. It does not touch the verdict.
- **Not recomputed here** (in the report only): the fake and K-matched null for every TF, the
  other-knockdown null for TFs other than GATA1, the KS calibration checks, the count+length rank, power
  for TFs other than GATA1 and cut-offs other than 0.5, the old "0/48" table. The key uses none of them.
- **Additions not in the report:** the source paper's Fisher test read literally (step 9). With TRRUST
  it passes BH for 0 TFs at every cut-off. With ChIP sets it passes BH for GATA1, MAX, THAP1, TERF2, GTF2B
  (cut-offs 0.01 to 0.1) and CEBPZ (cut-off 0, overlap >= 2) - 16 TF x setting cases. None of these is
  specific: on the same features, 61 to 226 of the 290 other ChIP sets also give p < 0.05, and the TF's own
  set ranks 3 to 147 of 291. Over all 106 settings (14 testable ChIP TFs x cut-off x overlap rule) the
  own set's mean rank fraction is 0.570 (95% CI over TFs 0.48 to 0.67; 0.5 = chance), and only 2 settings
  land in the top 5% (5.3 expected by chance): CEBPZ at cut-off 0 (rank 3) and GATA1 at 0.5 with
  overlap >= 1 (rank 13). The cause is the same rarity effect: responding features at cut-offs 0.01 to 0.5
  are rare-gene features (GATA1 at 0.1: median top-gene detection count 47, against 119.5 for all features),
  and ChIP sets are rich in rare, long genes. This does not disagree with the report; it is the same
  verdict reached by a test the report did not run. Also new: the cut-off-free rank check (mean rank fraction 0.652, 95% CI over TFs
  0.51 to 0.78; 6 of 20 TFs in the top half, 0 in the top 5%), and the coarse-bin sensitivity (decile bins
  give GATA1 an uncapped p of 0.015 at cut-off 0.25 against 0.093 with fine bins). The second shows that a
  count-matched null must be fine at the rare end: the lowest decile spans detection counts 1 to 27, while
  14 of feature 2610's 20 genes are seen in fewer than 10 cells.

My computation does not disagree with the report on any point that bears on the verdict.

## Why "not supported" and not "inconclusive"

- Under every null that keeps gene rarity, no TF passes BH at any of the 7 cut-offs in either database.
- The TF's own set ranks in the middle or lower half of all TF sets (GATA1 151/291 at 0.5, 255 to 277 at
  lower cut-offs; mean over 20 TFs 0.65 in the cut-off-free check). This is no hint of a weak signal.
- Up to 14 ChIP TFs and 48 TRRUST TFs are testable at lower cut-offs, and planted-target power is high
  for moderate signal (4 targets 0.985, 8 targets 0.95). Only very weak specificity (1 or 2 extra targets
  in a top-20 list, power about 0.2) cannot be excluded. A report should name this as a limitation.

## How the packages were built

- `../../build/T3/build_package.py` writes `tasks/T3-paper/data/` from the deployment's
  `task_data/` (read only). It drops files that contain the answer (`responding_features.tsv`,
  `tf_summary.tsv`, `tf_tests.tsv`, `catalog_check.json`), drops the precomputed count and length bins,
  the `priority` and `is_primary_tf` columns, and renames the `*_rebuilt` columns. `tf_info.tsv` is new.
- `data/README.md` was written by hand (neutral, package-relative). On 2026-10-01 a paragraph on which
  TFs have targets was added (58 TRRUST, 30 DoRothEA, 20 ChIP-flagged, 22 none), because 22 of the 87
  TFs have no row in either target file and an analyst would otherwise have to work this out.
- `methods/source_method_paper.pdf` is a byte copy of `references/2603.02952_Kendiukhov_SAE_atlas.pdf`.
- `../../build/T3/make_spec.py` copies `pipelines/sparse-autoencoders/01-sae-atlas.md` (file date
  2026-04-15 20:05, unchanged since, so it is the deployed version) to `contract/SPEC.md`. Only local
  paths are rewritten (to package-relative or repository-relative form); no other text changes.
- `tasks/T3-contract/` = `data/` and `methods/` copied from `T3-paper`, plus `contract/SPEC.md`.
  `diff -r` shows only `contract/` and `BRIEF.md`; the two briefs differ by one sentence ("A method
  specification for this kind of analysis is provided in contract/SPEC.md; follow it.").
- The `._*` files next to every file are macOS AppleDouble files made by the exFAT drive. They hold only
  the `com.apple.provenance` attribute (no paths, no content).

## Notes for graders

- Trap 1 (rarity) and trap 2 (comparison with other TFs) are the core of the task. An analysis that handles
  either one well will usually reach the right verdict; one that handles neither usually reports GATA1
  (or MAX, GTF2B, TERF2, THAP1) as specific.
- A report may use a different selection test (Welch t-test gives the same 5 GATA1 features at 0.5 and 77
  instead of 70 TFs with any BH feature), different bins, or a Monte-Carlo null. Compare key numbers only
  when the report's definition matches the key's.
- The contract arm's specification describes the source paper's test (TRRUST, ">= 2 of top-20", random
  features, |effect| > 0.5). Following it literally gives "0 TFs specific" with almost no testable TFs. That
  reaches the key verdict but mishandles traps 1 and 5 unless the report adds a matched null, testability
  counts and a power check. If the analyst extends the same Fisher test to DoRothEA ChIP-seq sets and lower
  cut-offs, it passes BH for GATA1, MAX and others (step 9 above). Reporting those passes as specificity is
  trap 2 (no comparison with other TFs' sets) and trap 1 (rare-gene features), and leads to the wrong
  verdict `supported`.
- CEBPZ is the one TF that looks good in two separate places (count-matched p = 0.030 at cut-off 0.01;
  Fisher rank 3/291 at cut-off 0). These are different cut-offs and different tests, found among many settings (7 cut-offs, 2
  databases, several tests); at the neighbouring cut-offs CEBPZ ranks 88 to 246 of 291 in the Fisher test and its count-matched p is 0.19 at cut-off 0.
  A report that names CEBPZ as a candidate for follow-up is fine; a report that calls it a finding is not.
- With the uncapped overlap M (no cap at 5), the count-matched null gives two more nominal p < 0.05, each at one
  setting only: MAX (ChIP, cut-off 0.1, p = 0.041, q = 0.81) and TBP (ChIP, cut-off 0, p = 0.018, q = 0.35).
  They were found by the independent check (`VERIFICATION.md`); `reference_output.json` counts them in
  `bh_summary` (`n_p_lt_0.05` = 1 at those two settings) but does not name them. A report that quotes them is not
  making a false statement; a report that calls them findings falls into the forking-paths trap.
