# P03 independent verification

- Date: 2026-10-01. Verifier: a separate agent, not the builder.
- Flawed package `packages/B3954` (key `keys/B3954.json`). Clean package `packages/B8827`
  (key `keys/B8827.json`).
- Python `bin/python`, CPU only, run with `-B` so no `__pycache__` was written.
- Verifier scripts are in `build/P03/verifier/`: `check_data.py` (data back to source),
  `indep_score.py` (my own scorer plus the original `setup/evaluate.py` on sample pairs),
  `indep_score_f32.py` (same with float32 thresholds), `dbg.py` (the 4 pairs that differ).
  `score_concepts_before_verification.py` is the flawed code as the builder left it.

## Verdict: OK after fixes

Both packages run, reproduce their outputs byte for byte, and give the documented wrong and
right results. They differ only at the error. I made three fixes. The main one: the code took
6 to 7 minutes on a busy machine, and now takes under 10 seconds with the same outputs.

## 1. Runs and reproduces outputs

| | B3954 (flawed) | B8827 (clean) |
|---|---|---|
| shipped code, fresh copy, run from package root | exit 0, 399 s wall, 0.41 GB | exit 0, 364 s wall, 0.41 GB |
| outputs vs shipped (sha256, all 3 files) | identical | identical |
| after fix 1, fresh copy of final package | exit 0, 5.8 s, 0.43 GB | exit 0, 10.2 s, 0.43 GB |
| outputs vs shipped after fix 1 | identical | identical |

The slow runs happened at load average 110 to 157 on a 10-core machine. Other agents were
running. The cause was OpenBLAS threads on thousands of tiny float32 matrix products (one per
feature). With BLAS limited to one thread, 300 features took 1.4 s instead of 36 s. Subjects
will run in parallel too, so the old code could pass 5 minutes for them. See fix 1.

## 2. Wrong vs right result, and the diff

My own scorer does not reuse the package code. It sorts each feature's split-A activations
and uses cumulative label counts. It reproduces every row of both packages exactly: winning
feature, split-B F1 and split-A F1, 21 of 21 concepts in each package.

| 21 concepts | flawed rule (rank on B, report B) | clean rule (rank on A, report B) | swapped clean (rank B, report A) | swapped flawed (rank A, report A) |
|---|---|---|---|---|
| mean best F1 | 0.360 | 0.162 | 0.247 | 0.376 |
| concepts passing | 6 | 2 | 3 | 5 |
| VARIANT (negative control) | 0.358 | 0.000 | 0.000 | 0.447 |
| REPEAT / PROPEP / TRANSIT | 0.626 / 0.267 / 0.257 | 0.003 / 0.000 / 0.230 | 0.014 / 0.482 / 0.756 | 0.569 / 0.482 / 0.782 |

So the flawed rule adds about 0.2 F1 per concept, in both split directions. This matches the
source study's finding (`runs/sweep/RESULTS.md` stage 4c: SAE winner's curse +0.167 at
layer 16, VARIANT falls to 0.000 under honest selection).

Two points from the pair description do not hold exactly. The builder already says so.
- VARIANT is 0.358 under the flawed rule here, not about 0.12. Different SAE file, proteins
  and labels.
- TRANSIT does not drop to about 0 (0.257 to 0.230). It did not drop in the source study
  either (0.591 both ways in `sweep_honest.json`, layer 16).

`diff -r` shows differences only in `code/score_concepts.py:140`, the three output files and
`SUMMARY.md`. README and all three data files are identical.

**Data back to source.** Checked by `check_data.py` against the protein-lm-sae files:
- The 600 proteins are exactly the seed-0 draw from `prots_test` + `prots_val`. I mapped
  them by accession, not by the builder's provenance file.
- None of the 600 proteins is in `prots_train`, and none of their residues is in the SAE's
  training mask. That mask covers exactly the 14,540 `prots_train` proteins.
- I re-encoded 4,000 random residues from the layer-16 activations with
  `sae_final_proteinsplit.pt`, using the same steps as `setup/topk_sae.py`. The active
  feature sets are the same for all 4,000 residues. Values agree to float16 precision (max
  relative difference 0.0005).
- Labels equal the matching rows of `residue_labels_bondfix.npz`. DISULFID has 268 positives
  and all are cysteine. Every residue has exactly 32 nonzero codes.

**Compared with the original scorer.** I imported `setup/evaluate.py` read-only and ran
`score_feature_vs_concept` on 102 pairs. These were all 42 winning pairs plus 60 random
pairs. 98 match exactly. The 4 that differ come from float rounding. The original compares
float32 activations with a float64 numpy scalar threshold. Numpy 1.26 then rounds the
threshold to float32. The package compares two arrays, so it stays in float64. When an
interpolated quantile lies within float32 rounding of a real activation value, one residue
can flip.

Over all 105,735 pairs, 167 split-B F1 values differ, by at most 0.0035. No winner changes
and no pass/fail result changes. The only reported number that moves is the clean ZN_FING
F1 (0.245 in the package, 0.242 with the original's rounding). This is not an error. I
corrected the key fact that said "exactly on 80 of 80".

## 3. Hints

- I searched README, SUMMARY, code and outputs for: fix, bug, leak, honest, curse, audit,
  review, error, correct, wrong, retract, double, dip, bias, winner, select, optimis, inflat,
  flaw, clean, trap, v1, v2, and repo paths. The only hits are the UniProt key COMPBIAS and
  "V2" inside UniProt entry names.
- The npz files hold only the documented arrays. Zip comments are empty and dates are the
  fixed 1980 default. No `._*`, `.DS_Store` or `__pycache__` in either package (after fixes).
- README step 7 ("the best feature is reported") does not say how the feature is chosen.
  It is the same in both packages, so it gives nothing away.
- SUMMARY wording. The flawed one says the threshold is tuned on split A and F1 is measured
  on split B. That is true and is what an analyst who missed the error would write. Its
  negative-control line ("does not pass the gate (F1 0.358)") is the natural reading of the
  numbers. The clean one says both choices are made on split A. That is an accurate method
  description. Neither says anything about the other version. I found no cue to remove.
- Every number in both SUMMARYs matches the outputs. I checked the pass lists, means,
  medians, margins, the 19/21 and 8/21 counts, the 10 concepts near 0, the "0.1% to 1.3%"
  range and the "66 to 1,275 positives" range.

## 4. Error location and detection rule

- `code/score_concepts.py:140` is `best = int(np.nanargmax(s["f1"][:, ci]))`. `s["f1"]` is
  set at line 109 from split-B counts. Thresholds are lines 99-104. All are correct in the
  key, and fix 1 kept these line numbers.
- The detection rule is fair. It needs both the cause (the feature is picked by the F1 that
  is reported) and the effect (inflated numbers). It does not count threshold "leaks",
  generic multiple-comparison remarks, or noting VARIANT 0.358 without the cause. I added
  one sentence. A finding counts even without naming split B or line 140, if it clearly says
  the reported F1 is itself the maximum of the values being reported. A reader would fix the
  code from that.
- The clean key's false-alarm list covers the likely wrong claims: threshold leak, residue
  split, SAE trained on test proteins, the noisy rare-concept values.

## 5. known_true_facts

I checked every fact by code: split sizes (300/300 proteins, 99,373/97,906 residues),
5,035 features scored, 21 concepts scored with the listed exclusions, ACT_SITE 0.00083,
protein counts for rare concepts (VARIANT 19, PROPEP 14, REPEAT 10, TRANSIT 8), SIGNAL
feature 2936 and DISULFID feature 1925 winning under both rules, all winner F1 and f1_a
values, the swapped-split values, SAE training set and VE 0.746 (`results_proteinsplit.json`).
Two facts were not exact (fix 3).

## Changes made

1. **Runtime (both packages, same edit).** In `code/score_concepts.py`, lines 82, 100 and
   101 now count with int32 instead of float32 (`Yi = Y.astype(np.int32)`,
   `.astype(np.int32)`, `pred_a.T @ Yi[ra]`). Numpy's integer matrix product does not use
   BLAS threads. The counts are whole numbers in both cases, so all outputs are byte-identical
   to the shipped ones. I checked this by sha256 in both packages. The line count is unchanged,
   so line 140 and the other key line numbers still hold. `build/P03/score_concepts_template.py`
   was updated the same way.
2. **README (both packages, same edit).** It said three keys have no positives (CA_BIND,
   METAL, NP_BIND). INTRAMEM also has 0 positives in these 600 proteins. It now says four
   keys. The run time line now says "under a minute" instead of "1 to 4 minutes". README is
   still byte-identical across the two packages.
3. **Keys (both).** I made three changes:
   - The reference-scorer fact now gives the float-rounding differences.
   - The exclusion fact now names INTRAMEM as having 0 positives.
   - The runtime fact now says under a minute.
   In B3954 I also added the one clarifying sentence to `detection_rule`.
   `dot_clean -m` removed the `._*` files that my edits created in the packages.

## Not done

- I did not re-run the source study's own 04c script or its neuron and untrained-SAE
  controls. Those numbers come from the source JSON files.
- Like the builder, I checked only one protein draw and one split seed, plus the swapped
  split.
- Nothing in biomi_automation was changed. `setup/evaluate.py` was imported with bytecode
  writing turned off.
