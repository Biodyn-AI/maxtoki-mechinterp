# T2 verification (independent check of the packages and the key)

Date: 2026-10-01. CPU only. I used my own code, not `reference.py`. My scripts are in
`verification_scripts/` (`a_data.py`, `b_main.py`, `c_variants.py`, `d_within.py`,
`e_cmh_detail.py`, `f_small.py`) with their outputs (`*.json`). They read only the package data.
`a_data.py` writes a cache file (`pairs_with_sumd.parquet`) next to itself; run it first.
Peak memory 3.0 GB (`a_data.py`), every script under 30 s.

**Verdict: OK after fixes.** The packages needed no change. The key verdict and every key number
stand. One claim in the key was wrong and is fixed: the |LFC| >= 0.5 subset does **not** disappear
once the target gene's usual direction is accounted for. Four smaller wording fixes were made.

## 1. The two packages differ only where they should

`diff -r -x '._*' T2-paper T2-contract` shows two differences only:

- `BRIEF.md`: one added sentence, "A method specification for this kind of analysis is provided
  in contract/SPEC.md; follow it." Same sentence and same place as in T1.
- `contract/` exists only in T2-contract.

Other checks:
- `contract/SPEC.md` is byte-identical to `pipelines/sparse-autoencoders/02-causal-circuit-tracing.md`
  (file date 2026-04-15).
- `methods/source_method_paper.pdf` is byte-identical to the source PDF (2603.01752) in both packages.
- Five data files are byte-identical to the source `task_data/` files (`records.parquet` ->
  `gene_pairs.parquet`, `edge_feature_genes.tsv` -> `feature_top_genes.tsv`, the other three same name).
- `knockdown_lfc.npz` equals the source LFC output: `lfc` equals the 227 matching rows of
  `lfc/lfc_matrix.npy`; `measured_genes` equals `var_symbols.txt` in order; `control_mean` equals
  `control_mean.npy`.
- `MANIFEST.sha256` verifies (7 of 7 files).

## 2. Hints and outcome statements

I searched `BRIEF.md` (both) and `data/README.md` for: fix, bug, v2, retract, correct, wrong,
leak, confound, baseline, null, chance, artefact, expected, should, audit, review, error, verify,
answer, trap, imbalance, majority, balanced, MCC, accuracy, skill, deploy, hook, legacy, original,
investigation, experiment, study, tie, fold, bootstrap, resampling, independence, and paths
(`projects/`, `runs/`, `scripts/`, `biomi`, `studyA`, `framework`), and percent-like numbers. I also
read all three files in full, and the parquet and npz metadata.

- No hits except plain words in their normal sense ("block", "direction", "resampling" in the
  deliverable format copied from T1).
- No outcome statements and no result numbers. The README describes the method's rule neutrally.
- Parquet schema metadata holds only column names and types. The npz holds only plain arrays.
- The PDF does not mention MaxToki.

Kept as is, but worth knowing (none is a hint of the answer):

1. **The spec carries the source paper's expectations.** It quotes 56.4% "vs 50% chance", says a
   result above 60% "is suspicious", and calls above 60% with rho > 0.1 "a publishable positive". It
   also has repository paths. This is the deployed spec, given verbatim as the treatment
   (Amendment 1, item 4).
2. **Interpreter path.** The T2 brief already names `analysis-tasks/bin/python`. The frozen T1, T3
   and T4 briefs name `maxtoki-framework-eval/bin/python`, and staging rewrites that line. For T2 the
   rewrite is a no-op. Either way the subject sees the same text.
3. **macOS `._*` files.** Both packages contain AppleDouble files from this drive (12 and 14). They
   hold only the `com.apple.provenance` attribute, no text. T3 has them too. The staged copies of
   T1, T3 and T4 do not, so staging should skip them for T2 as well.

## 3. Data load and match their descriptions

- Edges: 272,905 rows, no duplicates; sign matches the sign of d in every row; min |d| = 0.5000 and
  min consistency 0.705 (rounded values); target site always after source site; source sites 0, 3,
  6, 9; 58.27% inhibitory.
- 120 source features (30 per site); 18 have no edge. Every edge feature (46,705) has a top-10 list,
  and no other feature does. All lists have 10 entries; 364 contain `<SPECIAL>`; all upper case.
  Source-feature lists equal their rows in `feature_top_genes.tsv`.
- npz: `lfc` float32 (227, 6546), finite; row order equals `silenced_genes.tsv`; cell counts 32-575
  and equal to the tsv; 6,546 unique measured genes; every silenced gene is measured.
- Own LFC: all 227 negative, median -0.927, 199 below -0.5.
- Gene pairs: 698,624 rows, 227 silenced genes (same set as the tsv), 6,324 targets (all measured),
  no self pairs, no duplicates, sorted as stated. `predicted_decrease` equals frac > 0.5 in every
  row; every row passes the keep rule; 72,275 ties; `lfc` equals the npz value exactly; no lfc is 0.
- **Rebuild.** From `circuit_edges.csv` + `feature_top_genes.tsv` with the README rule (27,277,642
  triples), my code gives the same 698,624 pairs with identical evidence, inhibitory counts and
  max |d|. So the README rule is exact.
- The process facts in the README match the source scripts: 3,000 control cells with
  `default_rng(42)`, log1p(count / total x 1e4), >= 10 cells per guide, candidate genes = top-10 genes
  of the source features, source features ranked by the sum of -log10 p over significant GO BP /
  KEGG / Reactome / TRRUST terms of the top-20 genes, SAEs trained on 500 control cells.

## 4. Key numbers re-derived with my own code

My own seeds (2026, 11, 12, 99, 31, 777, 4242), except the 5-fold split, which the key defines
as `default_rng(42)`. All 41 key numbers are within tolerance.

| Key number | Key | Mine |
|---|---|---|
| pairs / genes / targets / ties | 698,624 / 227 / 6,324 / 72,275 | same |
| observed / predicted decrease | 0.54313 / 0.50888 | 0.54313 / 0.50888 |
| accuracy [gene CI] | 0.49761 [0.49188, 0.5037] | 0.49761 [0.49189, 0.50399] |
| accuracy - always-decrease [CI] | -0.04552 [-0.05792, -0.03341] | -0.04552 [-0.05779, -0.03276] |
| balanced accuracy; minus 0.5 CI | 0.49682; [-0.0083, 0.00203] | 0.49682; [-0.00812, 0.00206] |
| MCC [CI] | -0.00635 [-0.01655, 0.00405] | -0.00635 [-0.01619, 0.00410] |
| MCC CI by source-feature group / component | [-0.0157, 0.0028] / [-0.0133, 0.0121] | [-0.0157, 0.0030] / [-0.0133, 0.0114] |
| groups / components / largest | 119 / 50 / 88 | 119 / 50 / 88 |
| kappa / chance from class rates / precision | -0.00633 / 0.50077 / 0.54002 | same |
| pair-level CI; z vs 0.5; precision z | [0.49643, 0.49878]; -4.003; 47.73 | same |
| target-direction baseline (5 folds, seed 42) | 0.587 (bal 0.5755, MCC 0.1567) | 0.58700 (0.57549, 0.15667) |
| same, other seeds / leave-one-gene-out | LOO 0.588-0.592 | 0.5864-0.5877 / 0.5881 |
| circuit - baseline [gene CI] | -0.08939 [-0.09981, -0.07994] | -0.08939 [-0.09925, -0.07947] |
| fold accuracies circuit / always / baseline | 0.494-0.506 / 0.537-0.561 / 0.575-0.592 | same |
| AUROC of support fraction [CI]; per-gene mean | 0.49786 [0.491, 0.505]; 0.50173 | 0.49786 [0.491, 0.505]; 0.50182 |
| per-gene mean MCC [CI] (213 genes) | 0.00467 [-0.0041, 0.0139] | 0.00467 [-0.0041, 0.0141] |
| per-gene mean acc - always [CI]; one-sign genes | -0.03446 [-0.0431, -0.0259]; 14 | -0.03446 [-0.0432, -0.0255]; 14 |
| ties as decrease: acc / ppred / bal / MCC | 0.50547 / 0.6123 / 0.49575 / -0.0087 | same |
| sign of summed d: acc / ppred / bal / MCC | 0.50179 / 0.5629 / 0.49634 / -0.00735 | same |
| unanimous: n / acc / ppred / bal | 79,269 / 0.51821 / 0.6642 / 0.50101 | same |
| max d > 2: n / genes / acc / ppred / bal | 29,373 / 158 / 0.55677 / 0.8549 / 0.51037 | same |
| site 9 only: n / ppred / acc - always | 1,078 / 0.79 / -0.00186 | 1,078 / 0.792 / -0.00186 |
| band >= 0.25: n / genes / bal / MCC | 10,029 / 145 / 0.51376 / 0.02426 | same |
| subset >= 0.5: n / genes / bal / MCC [CI] | 416 / 59 / 0.59519 / 0.16604 [0.0667, 0.2832] | same, CI [0.0666, 0.2845] |
| subset >= 0.5: within-gene shuffle null mean / p | 0.11921 / 0.109 | 0.11869 / 0.109 |
| threshold family p (max z, at) | 0.085 (1.99, 0.45; single p 0.030) | 0.078 (2.05, 0.45; single p 0.029) |
| CMH all pairs by gene: OR / p | 1.01063 / 0.0432 | 1.01063 / 0.0429 (no continuity correction) |
| CMH all pairs by gene x usual direction | 0.99765 / 0.66 | 0.99738 / 0.62 (3-level usual direction) |
| per-gene Spearman [CI]; after target mean | 0.00963 [0.0003, 0.0188]; -0.00214 | 0.00963 [0.0003, 0.0196]; -0.00214 [-0.0119, 0.0073] |
| on-target median; stronger half bal | -0.92738; 0.50004 | same |
| 21 variants: bal range; MCC range; never above always | 0.494-0.595; -0.011-0.166; yes | 19 recomputed, same values and same range |

`c_variants.json` shows NaN CMH p-values: statsmodels fails on strata with one pair. `d_within.py`
and `e_cmh_detail.py` compute CMH by hand and replace them.

## 5. The verdict follows from the evidence

"not supported" is right, and it agrees with the corrected source report (`V2_CIRCUIT_REPORT.md`:
"no detectable directional skill"; accuracy 49.76%, balanced 49.68%, MCC -0.0063, target baseline
58.70%; all reproduced above).

- On all pairs, every measure that is not fooled by the 54/46 class split is at chance. The CIs
  sit within about 1 point of chance under three resampling units.
- Raw accuracy is below always-"decrease" overall, in every fold and in all 21 variants.
- A rule with no model beats the circuit by 8.9 points, so the labels do hold learnable direction.

"inconclusive" under the key's three conditions is fair. A post-hoc subset of strongly changed
pairs has a CI above chance, and at |LFC| >= 0.45 the within-knockdown CMH is nominally significant
(p = 0.037). The family test over thresholds (p = 0.085) does not hold up, and the subset is 0.06%
of pairs. "supported" is not defensible.

## 6. Traps: real, and the descriptions are now fair

All six traps are real in these data (numbers in section 4). The deployed bug behind
`metric_ignores_prediction` is real: `remaining_phases.py` line 327 counts a pair as correct when
`actual_lfc < 0`, whatever the prediction. The source paper does compare 56.4% with "50% chance".

Changes (made in `build/T2/make_key.py`, then `key.json` regenerated from it; I first checked that
the unmodified script reproduced the old `key.json` byte for byte, so nothing else changed):

1. **Subset claim corrected (the one material error).** The key said the |LFC| >= 0.5 excess
   "disappears / is gone once the target gene's usual direction is accounted for (CMH p = 0.35)".
   The reference's own output shows the odds ratio does not shrink: by silenced gene 1.56, adding
   the target's usual direction 1.80 (at 0.45: 1.63 -> 1.50). The p-value rises because only 14
   knockdowns (273 pairs) carry information within a knockdown, and only 11 strata (137 pairs) after
   the extra split. So the target's usual direction does not explain the subset; the test is just
   weak. A subject who reported this correctly would have been marked as making a false statement.
   Fixed in `key_conclusion`, `traps[post_hoc_subsets].description`, `false_statement_checks[6]`
   (now says that reporting the unchanged odds ratio is not false) and the note of
   `subset_abs_lfc_ge_0p5_mcc` (adds the odds ratios, the stratum counts and the 0.45 result). The
   all-pairs claim (1.011 -> 0.998) is right and stays.
2. **`target_direction_baseline` description.** "The circuit's only within-knockdown association"
   changed to "within-knockdown association over all pairs", because the 0.45 subset also has a
   nominal one.
3. **`resampling_unit` description.** "The source paper's p < 1e-92" changed to say that this p is
   for the paper's magnitude correlation (the spec quotes 1e-32). It is not a p-value for direction.
4. **`constant_sign_predictor` correct handling.** It required both the predicted share for every
   set AND a robust measure. Now either is enough: a robust measure for every set, or raw accuracy
   only next to the predicted and observed shares and the always-"decrease" score.
5. `verification` field points here. `NOTES.md`: point 2(c) corrected, the "both fail" bullet
   reworded, file list and limits updated.

Hashes. Before: `key.json` 5c845d171b50b75f4edc4891ba7b2aa550cf585e79fd106cf1abce8b540a2836,
`NOTES.md` f1ee87a415cf4b8962beeb3ec5f831ae97cf212efa2253a8eecf16e85340fa68, `make_key.py`
687714b6de613e7e2ac3dac95c2b1fff93af82bcabc0952bdf1e96bed64b6ccd.
After: `key.json` 8f9e79aa6fb3e7e9c36590224deb12425d1227723787cbc6390ea887081dbd47,
`NOTES.md` 6c8896b82413296f393e0f92609def21fb337e295241b614b3fa731a4d97ca01, `make_key.py`
ab11221ebdd9f05978539f7c289b70040d7d0ab870c9ba7f8a237e93abf1f67f.

## Open points for the study runner

- `NOTES.md` has a grader rule ("score `target_direction_baseline` as not handled only if the
  subject claims skill or says direction cannot be predicted"). `key.json` does not have it, and the
  T1/T3/T4 keys have no such rule. Decide whether graders see `NOTES.md`; if not, the rule is unused.
- A second scratch folder (`scratchpad/t2v/`, 13:28-13:29) suggests another agent started a T2
  check. I did not read it. Nothing in `keys/T2/` had changed when I edited it.

## Not done

- I did not write the key hash into `protocol/FREEZE_LOG.md`, stage the packages into
  `analysis-tasks/`, or update `STUDYA_PACKAGE_MAP.json`.
- I did not run `reference.py`. I compared my numbers with `reference_output.json`.
- I did not recompute the deployed 53.5% (the deployed edges are not in the package).
- I recomputed 19 of the 21 analyst variants myself (including the four per-site rebuilds). The
  other two (drop ties; summed d with |LFC| >= 0.01) I only read from the reference output.

## Plain-words summary

The two packages are clean and differ only by the spec. The data load, match their README and
match the source files. My own code rebuilds the gene pairs exactly and reproduces all 41 key
numbers within tolerance. The verdict "not supported" follows and agrees with the corrected report.
One claim in the key was wrong: the small positive in strongly changed pairs does not go away when
you account for each target gene's usual direction. It is just too small to test. I fixed that
claim and four smaller wordings so graders do not mark correct statements as false.
