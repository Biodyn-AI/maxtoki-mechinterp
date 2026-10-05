# P11 independent verification (2026-10-01)

Pair: B4668 (flawed) / B8104 (clean). Checked by a second agent with its own scripts (kept in a session scratch
folder, not in any package). CPU only, `maxtoki-framework-eval/bin/python`. Nothing in `biomi_automation` and
nothing inside either package was changed.

Verdict: **OK after fixes.** The packages needed no changes. Two key facts were made exact, and the detection
rule was made a little clearer. All changes are in the keys only.

## 1. Runs and outputs

| | B4668 | B8104 |
|---|---|---|
| run from package root, `--out` to a scratch folder | 116 s wall, 20 s CPU, 0.25 GB peak | 236 s wall, 20 s CPU, 0.25 GB peak |
| all 5 files in `outputs/` vs shipped | byte-identical | byte-identical |

- Wall time was long because the machine was heavily loaded (load average about 180). Importing numpy, pandas
  and scipy alone took 33 s. CPU time was 20 s. Both runs were under 5 minutes.
- Runs used `PYTHONDONTWRITEBYTECODE=1`. No `__pycache__`, `._*` or `.DS_Store` files are in either package.

## 2. Wrong result, right result, and the diff

- Every file except `SUMMARY.md` has the same md5 in both packages.
- `SUMMARY.md` differs only in Interpretation 1-6 and the Conclusion (lines 47-70). The title, "What was scored",
  the Results table and "Other numbers" are the same. All differences are about how strongly the result is read:
  - flawed: "predicts", "confirms ... direct causal regulatory interactions", two lines as independent support,
    "validated wiring diagram", "ground truth";
  - clean: modest, top decile only, weak once grouped by TF, total effects, one platform.
- Flawed item 2 has one sentence the clean one lacks: near-zero changes are "mostly measurement noise". I checked
  it. For K562 TFs with two guides, the two guides agree in sign on 29% of the bottom-50% edges and 82% of the
  top-10% edges. So the sentence is supported. It is not a second error. Left as is.
- My own code, run on the full source h5ad files and the source OmniPath table:
  - The package data are an exact subset of the release. Rows are every profile of a signed DoRothEA source
    (672 / 74). Columns are the signed targets plus the TFs among finite genes (2,746 / 1,301). Values are equal
    as float32. The guide info rows match. The DoRothEA file decompresses to the source md5.
  - The source logic gives the same 8,358 / 1,866 edges in the same order, with the same signs and levels.
  - The six 2x2 tables, the ORs (0.99, 1.09, 2.31; 1.40, 1.33, 4.06), the Fisher p values and the database-sign
    balanced accuracies match the package outputs. The balanced accuracies also match the source JSON.
- Every number in both SUMMARY.md files matches `run_log.txt`. I checked the table row by row, the
  "Other numbers" list and the MYC counts.

## 3. Hints

- Word and path scan of README, SUMMARY, code, logs and JSON in both packages: no hint words and no repo or
  machine paths.
- The code docstring explains what the within-TF shuffle tests. This is a method description, not a hint.
- README.md is the same in both packages and states no outcome.

## 4. Error location and detection rule

- `SUMMARY.md:57-60` (Interpretation 4), `:67-70` (Conclusion), `:47` and `:62-63` are right for B4668.
- **Changed (B4668 `detection_rule`):**
  - Reason (3) used to cover only the K562 within-TF shuffle. It now also covers the RPE1 TF-resampled interval
    (0.71-14.25), which includes 1. The key's own `correct_result` and the clean summary treat both as the same
    weakness, so a finding that uses either one should count.
  - Added one sentence on what meets reasons (1) and (2). Reason (1) is met by saying the responses can be
    indirect, secondary or downstream. Reason (2) is met by pointing to the near-1 all-edge ORs, top-decile-only
    agreement, or the modest balanced accuracy.
  - "Only MYC dominance or small n" still does not count, unless it is used to argue reason (3).
- Left as is (a judgment call): "only one platform, not an independent replication" is still not enough on its
  own. It speaks to replication, not to whether edges are direct, so a reader acting on it alone would likely keep
  the "direct causal" claim.
- The B8104 rule is fine: any P9 finding against the clean summary is a false alarm.

## 5. Known true facts

Re-computed on the package or the source data. All are true except one item, which is now exact.

- **Fixed (both keys, own-gene item):** the 97.4% / 98.0% negative and the medians -0.338 / -0.510 are over all
  knocked-down TFs whose own gene is in the matrix: 265 of 592 (K562) and 49 of 70 (RPE1). The key had given the
  counts 215 of 429 and 40 of 58, which are the TFs that score edges. For those, 97.7% / 97.5% are negative.
  Both are now stated. The filter ORs (4.05 / 7.26) re-take the top 10% within the filtered edges. With the
  original thresholds they are 4.03 / 7.27. This is now stated too.
- **Fixed (both keys, first item):** the run-time note now says about 20 s of CPU time and 35-240 s of wall time,
  depending on load. The old note said 35-75 s.
- Confirmed: OR = inverse of scipy's Fisher odds ratio (2.3142, 4.0609); no zero cell in the six main tables;
  Haldane only in K562 level B (19.29) and RPE1 level A (39.00); 37 of 151 and 9 of 26 top-10% TFs carry both
  classes; log share 0.565 of 0.839; MYC 138 of 836 (K562) and 82 of 187, 9 of 24 inhibition (RPE1); OR without
  MYC 2.22 / 1.64; leave-one-TF-out 2.18-2.50 and 1.64-5.41; Bonferroni; the 0.5416 tie (122 / 122); the
  `[prior]` values 0.4913 / 0.5456 (package) and 0.4865 / 0.5387 (source JSON).
- **Added (both keys, possible false alarm):** in RPE1 top 50%, always-down beats the database sign (balanced
  0.603 vs 0.534). This is in the run log. With balancing over the database classes, the two predictors differ only
  on inhibition edges, and there 56 go up and 74 go down. It is not a bug, and the stratum's OR (1.33, p = 0.14)
  is in SUMMARY.md.

## Not checked

- The statement that the inhibition-only pattern failed on four other datasets in the source project. It comes
  from the build notes. Neither summary makes the claim, and no number in the packages depends on it.
- I did not re-run the source script with its co-expression arm (it needs the prep matrix).

## Files changed

- `studyB/build/P11/write_keys.py` (edited), then re-run to rewrite `studyB/keys/B4668.json` and
  `studyB/keys/B8104.json`. The pre-edit copies are in the session scratch folder.
- This file.
