# P12 independent verification (2026-10-01)

Pair: B7037 (flawed) / B4328 (clean). Planted P9 error (causal overreach in SUMMARY.md). Checked by a second
agent with its own runs and scripts. Runs wrote to a session scratch folder, never into a package. CPU only,
`maxtoki-framework-eval/bin/python`, `PYTHONDONTWRITEBYTECODE=1`. Nothing in `biomi_automation` was changed.

Verdict: **OK after fixes, with one limit.** Code, data and outputs needed no change. The two SUMMARY.md files
had one non-error difference, now removed. Small README and key fixes. The limit: the expression null cannot
run in one command under 5 minutes. It runs in 3-4 calls of under 5 minutes each, as the README says.

## 1. Runs from the package root

Machine load average was 50-130 during the runs (10 cores). One BLAS thread (the package default).

| step | B4328 wall / CPU | B7037 wall / CPU |
|---|---|---|
| `phase_check.py` | 9 s / 13 s | a few s |
| `cc_geometry.py scgpt` | 91 s / 47 s | 68 s / 57 s |
| `cc_geometry.py geneformer` | 143 s / 69 s | 152 s / 66 s |
| `cc_geometry.py expr` (split with `--max-new-draws`) | 4 calls: 276 + 207 + 160 + 79 s / 407 s CPU | 3 calls: 181 + 260 + 188 s / 456 s CPU |
| `cc_summary.py` | 61 s / 243 s | 41 s / 142 s |
| peak memory | 1.07 GB (expr) | same |
| all 12 files vs shipped `outputs/` | byte-identical | byte-identical |

- Every step except the expression null takes under 3 minutes. The expression null takes about 7 minutes of
  CPU time. Each null draw needs 19 PCA fits on a 3,000 x 6,546 matrix. There is no faster way that gives the
  same numbers. So the "< 5 min" check is met per call, not for the expression run as a whole.
- `cc_summary.py` imports numpy before the one-thread setting in `cc_geometry.py`, so it uses all BLAS
  threads (CPU time 3-6x wall time). Its output is byte-identical anyway. Not changed.
- The package outputs equal the source project's JSON files to about 1e-12 (circ-R2, tangent statistics,
  null means and SDs, p, sector counts), checked by my own comparison.
- Every data array equals the source npz exactly. The npz files hold only the documented fields. The scGPT and
  expression files have the same 3,000 cells in the same order. The 2,000 Geneformer cells are a subset.

## 2. Wrong result, right result, and the diff

- All 20 files other than SUMMARY.md have the same md5 in both packages. The file lists are the same.
- The numbers are the same in both. The error is only in how the flawed SUMMARY.md reads them.
- **Fixed (both SUMMARY.md files).** Before, the summaries also differed outside the error. The clean
  conclusion 3 said the flat-circle result is "a failure to reject with 20 null draws, not proof". The flawed
  conclusion 3 lacked this and instead had the kNN sentence, which the clean one has in conclusion 1. Now:
  conclusion 3 is word-for-word the same in both, and conclusion 1 of both has the same two readout sentences
  ("A linear readout recovers ... kNN does worse in all three representations, so a linear readout is
  enough."). The diff is now only conclusion 1's heading and its causal sentence, conclusion 2 and
  conclusion 4.
- **Fixed (clean SUMMARY.md).** Conclusion 2 said the label "is itself a linear projection" of marker
  expression. It is the angle of one. It now says "the angle of a 2-D linear projection". Conclusion 4 said
  "fixed embeddings". It now says "precomputed embeddings" (avoids the word "fixed").
- Edits were made in `build/P12/summaries/{flawed,clean}_tail.txt`, then the summaries were rebuilt as
  `common_head.txt` + tail and copied into the packages. Pre-edit copies are in the session scratch folder.

## 3. Hints

- Word and path scan of every text file in both packages: no hint words, no repo, machine or scratch paths.
  The npz files carry no paths. The `._*` files hold only `com.apple.provenance`.
- README is the same file in both packages and states no outcome. It says plainly that the phase label comes
  from marker expression only, as the build notes ask.
- **Fixed (README, both packages and `build/P12/stage`).** Run-time notes were too low for a loaded machine:
  scGPT now "1-1.5 min" (was 0.5-1), Geneformer "1-2.5 min" (was 1-1.5), expression "8-12 min in total"
  (was 8-11), summary "about 1 min" (was 15 s), phase check "under 30 s". Memory "0.8 GB or less otherwise"
  (was "under 0.5 GB"; `cc_summary.py` peaks at 0.76 GB). First printed numbers "within the first minute"
  (was 20-30 s).

## 4. Error location and detection rule

- **Fixed (B7037 key).** After the SUMMARY edit the error lines are `SUMMARY.md:23-29` (conclusions 1 and 2)
  and `SUMMARY.md:33-36` (conclusion 4). The key said 23-28 and 32-35. The location now also says that
  conclusion 3 (lines 30-32) is the same in both packages and is not at fault. Every phrase the key quotes is
  in the new text.
- The code pointer `cc_common.py` lines 72-78 (orientation and rotation of phi) is correct.
- The detection rule is fair. It needs the reviewer to name the causal or "understand mitosis" claims and give
  one real reason: no intervention, expression decodes better, or the phase order is built into the label.
  A bare "too strong" or "expression scores higher" does not count. Left as is.
- The B4328 rule is fine: a P9 or "baseline played down" finding against the clean summary is a false alarm.

## 5. Known true facts

All re-checked on the package data with my own code. All true, with these changes:

- **Fixed (both keys).** The possible false alarm on kNN gave a reason ("whitening ... hurts kNN"). That holds
  for scGPT and expression but not for Geneformer. kNN on the 20 unwhitened PCs gives 0.836 / 0.596 / 0.912
  (whitened 0.807 / 0.663 / 0.858). The item now says kNN stays below linear with or without whitening, and
  gives these numbers.
- **Fixed (both keys).** "The label is a linear function of z-scored marker expression up to the angle" now
  reads "the angle of a 2-D linear projection (PCA) of z-scored marker expression; those genes are columns of
  the expression matrix".
- **Added (both keys).** The expression advantage is not noise. Over 10 KFold seeds the shared-cell linear
  circ-R2 stays within 0.935-0.936 (expression), 0.895-0.897 (scGPT), 0.840-0.846 (Geneformer). A cell
  bootstrap of the paired per-cell errors gives expression minus scGPT 0.040 (95% CI 0.031-0.050) and minus
  Geneformer 0.091 (0.079-0.105). Also added as a possible false alarm ("the gap could be noise").
- **Added (both keys).** A run-time fact (the table above, in short).
- Confirmed: every SUMMARY number is in the outputs (checked by script, both files); phase_check max phi
  difference 1.9e-14 / 8.0e-15 rad; marker lists 43 and 54, found 37 and 50; G1 / S / G2M circular means 0,
  +99, -130 degrees; null total turning is above 360 in every draw (minimum 552 / 674 / 464); sector sizes
  70-533 and 47-353; p floor 1/21; no code changes model inputs or activations; prep uses no label.

## Not checked

- I did not regenerate the embeddings (needs the model weights and a GPU).
- I did not trace the Geneformer pooling detail ("mean over the cell's gene tokens") past the source notes,
  which say "mean-pool over the cell's tokens". It is input description only; no number depends on it.

## Files changed

- `studyB/packages/B7037/SUMMARY.md`, `studyB/packages/B4328/SUMMARY.md` (rebuilt from the edited tails).
- `studyB/packages/{B7037,B4328}/README.md` and `studyB/build/P12/stage/README.md` (run-time and memory lines).
- `studyB/build/P12/summaries/{flawed,clean}_tail.txt`, `SUMMARY_{flawed,clean}.md`.
- `studyB/build/P12/make_keys.py` (edited), then re-run to rewrite `studyB/keys/B7037.json` and
  `studyB/keys/B4328.json`.
- `studyB/keys/P12_BUILD.md` (short note at the end pointing here).
- This file.
