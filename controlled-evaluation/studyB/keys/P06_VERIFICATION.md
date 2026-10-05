# P06 verification (independent check of the build)

- Pair: P06, C2S-2B cell-cycle readout, K562 -> RPE1 zero-shot transfer.
- Flawed: `studyB/packages/B1144` (key `keys/B1144.json`). Clean: `studyB/packages/B2101` (key `keys/B2101.json`).
- Checked 2026-10-01 by a separate agent. CPU only, PYTHONDONTWRITEBYTECODE=1, outputs written to the scratchpad.
- Verdict: **OK after fixes.** Fixes were small: one README line (both packages) and four facts in the keys.
  No code, data, output or SUMMARY file was changed.

## 1. Runs from the package root

| package | wall time | peak memory | JSON vs `outputs/` | log vs `outputs/run_log.txt` |
|---|---|---|---|---|
| B1144 (flawed) | 96 s | 1.36 GB | byte-identical | identical except the `[done] ->` path line |
| B2101 (clean) | 144 s | 1.33 GB | byte-identical | identical except the `[done] ->` path line |

The machine load average was about 100 during these runs, so times are 2-4x the builder's 25-50 s.
Both are under 5 min. No `__pycache__` was written.

## 2. Wrong result, right result, and the diff

- Flawed run: circ_corr within 0.897 / 0.919, transfer -0.814 / -0.796, raw median error 24.0 / 16.0 deg,
  both gates fail, verdict "NEITHER ... target label is invalid". This matches the original V3 run
  (`results/transfer_rpe1.json`, `results/v3.log` "EXP B-prime"). Model readout differs by about 1e-15.
  Expression readout differs by at most 2.3e-7 on scores and 8e-6 deg on median errors. The log matches V3
  except that V3 ended the failure line with "NO WINNER DECLARED." (dropped, as the build notes say).
- Clean run: R_diff within 0.886 / 0.913, transfer 0.789 / 0.878, median error after rotation 23.3 / 16.0 deg,
  constant floor 0.043 / 86 deg, random 0.017 / 88 deg, both pass. Identical to the builder's rerun of the
  current source script (`build/P06/ref/ref_transfer_test_current.json`) on every shared field.
- Diff of the two packages: README, `cc_phase.py` and all data files are identical (sha256). `transfer_test.py`
  differs only in the scoring function (circ_corr vs R_diff), the median error (raw vs after best rotation),
  the chance level (assumed 90 deg vs measured constant and random floors), the gate, the failure text and
  verdict string, the docstring lines that describe these, and three extra output fields in the clean
  version (`pred_R`, `transfer_offset_deg`, source/target phase R). All of these are the metric change
  itself or come with it in the current source script. SUMMARY.md differs only in the numbers, the
  reading of them, and the follow-on Next/Limits sentences.

## 3. Hint scan

- Searched every text file in both packages for words like fixed, bug, retract, correct, wrong, leak, audit,
  review, legacy, oracle, diag, earlier version, plus absolute paths and repo names. Only hit: the word
  "invalid" in the flawed verdict string, which is part of the original wrong conclusion.
- h5ad files: only X, obs index and var index (no uns, obsm, layers or annotation columns). The `.npy` files
  and manifests carry no paths. The `._*` files hold only the `com.apple.provenance` attribute.
- The clean SUMMARY does not mention circ_corr and does not rank the model against expression. It scopes the
  claim to both readouts transferring.

## 4. Key location and detection rule

- `code/transfer_test.py` lines 39-43 (circ_corr), 67 (within), 74 (transfer), 36 (90 deg assumed), 131 (gate)
  all point at the right code in B1144. SUMMARY "Interpretation" 1-2 hold the wrong conclusion.
- The detection rule is fair. It needs the reader to blame the statistic and to back it with the mechanism,
  the small median errors, or a computed alternative. It does not credit "label is not cell cycle" or
  "the 90 deg floor is assumed" alone, which would not fix the result (measured floor is 86 deg).

## 5. Known true facts (recomputed with my own code on the package data)

All numbers in both keys were recomputed and hold: marker list 94 (42 S + 52 G2/M), present 87 / 93;
orientation (S 45 deg both, G2/M 153.8 / 142.1, K562 flipped); true R 0.079 / 0.043, true mean 11.8 deg;
prediction R 0.341 / 0.215, prediction mean 226.7 / 235.2 deg; 81.5% / 90.7% within 45 deg; rotation
-3.8 / -2.5 deg; shared-reference circ_corr +0.848 / +0.899; RPE1 refit (5-fold CV) circ_corr 0.928 / 0.958,
R_diff 0.909 / 0.952, median error 10.5 / 7.9 deg; row_cell_ids identity; 6,544 shared genes; activations
byte-identical to the source; h5ad X, gene names and cell names equal to the source.

## Changes made

1. `packages/B1144/README.md`, `packages/B2101/README.md`, `build/P06/common/README.md`: "runs on CPU in about
   one minute" -> "runs on CPU in one to three minutes". Reason: measured 96-144 s under load; a subject
   with a 2-minute command timeout could cut the run short. READMEs are still identical (new sha256
   e4e9d68cfb2c8a01...; the build record's older hash is now stale).
2. `keys/B1144.json` known fact 1: "agreement with the original JSON to about 1e-8" was too strong. Now:
   model to about 1e-15, expression within 3e-7 on scores and 1e-5 deg on median errors.
3. `keys/B1144.json` error.nature: "reference angles sit about 215 deg apart" was right only for the model.
   Now gives 215 / 223 deg (145 / 137 deg the short way) and notes the coefficient is roughly the cosine of
   that gap.
4. Both keys, data fact: "package data are bit-identical to the source project's inputs" was not literally
   true for the h5ad files (rewritten, annotation columns dropped). Now says activations are byte-identical
   and the h5ad X, gene names and cell names are equal.
5. Both keys, run-time fact: "about 30-60 s" -> "25-150 s depending on machine load, about 1.4-1.5 GB".

Backups of the keys before these edits are in this session's scratchpad only.

## Not changed, for the harness owner

- Running `python code/transfer_test.py` without `--out` writes into the package's `outputs/` folder, and
  importing `cc_phase` writes `code/__pycache__` if the folder is writable. Subjects should run with
  PYTHONDONTWRITEBYTECODE=1, or the packages should be made read-only, so that no run leaves files behind.
- The build record still says "215 deg apart" and gives the old README hash. Left as a build-time note.

## Summary

Both packages run in under 3 minutes and reproduce their stored outputs byte for byte. The flawed one gives
the documented wrong-signed circ_corr result; the clean one gives the R_diff result. They differ only in the
scoring statistic and what follows from it. No hints found. The key location is right and the rule is fair.
Four facts in the keys were worded too strongly or only half right; they are now corrected.
