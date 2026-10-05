# P02 independent verification

- Date: 2026-10-01. Verifier: a separate agent, not the builder.
- Flawed package `packages/B9296` (key `keys/B9296.json`). Clean package `packages/B3100`
  (key `keys/B3100.json`).
- Python `bin/python`, CPU only. Outputs went to the session scratchpad. No `__pycache__` was
  written into either package.
- Verifier scripts are in the session scratchpad, not in the repo: `raw_check.py` (one pass over
  the raw h5ad), `extract4000.py` + `full_run.py` (both package scripts on all cells, 4,000 genes),
  `indep.py` (own float64 re-implementation of the ceiling, with four split variants).

## Verdict: OK after fixes

Both packages run and reproduce their outputs. The flawed one gives the documented wrong result
and the clean one the correct result. They differ only in the ceiling block of the code and in the
numbers that follow from it. I found no hints. The keys needed three small changes and the build
record one (see "Changes").

## 1. Runs and reproduces outputs

| | B9296 | B3100 |
|---|---|---|
| `python code/learn_interaction.py` from package root | exit 0, 3.5 s, 0.89 GB | exit 0, 6.2 s, 0.90 GB |
| `results.json` vs shipped | byte-identical | byte-identical |
| `run_log.txt` vs shipped | identical except the `wrote` line (output path) | same |

## 2. Wrong vs correct result, and the diff

| | split-half | reliability | ceiling | best model share |
|---|---|---|---|---|
| B9296 (package) | 0.6286 | 0.772 | 0.879 | 46% |
| B3100 (package) | 0.3946 | 0.566 | 0.752 | 54% |
| B9296 code, all 111,445 cells, 4,000 genes | 0.6132 | 0.7602 | 0.872 | 44.5% |
| B3100 code, all cells, 4,000 genes | 0.3823 | 0.5532 | 0.744 | 52.2% |
| original `norman_learn.json` | 0.6132 | 0.7602 | 0.872 | - |
| documented correction (BENCHMARK.md) | 0.3835 | 0.5544 | 0.745 | 54% (other model) |

- On the full data the flawed code matches the original run to 4 decimals, including all model
  scores (0.3742 / 0.3788 / 0.3881). So the numpy port is faithful.
- My own float64 code, 5 split seeds, on the package data: double-only split 0.878; every group
  split 0.743-0.750. Package code, seeds 1-6: 0.877-0.880 vs 0.740-0.752. The gap is about 0.13
  every time.
- Code diff: only lines 72-81 added and line 78 replaced (ceiling block). README identical. Data
  identical (sha256 1084b3bc...1892). SUMMARY differs only in ceiling numbers, shares and the
  "more than half" / "just under half" wording. `results.json` differs only in `_ceiling` and
  `frac_of_ceiling`.

## 3. Hints

- No hint words in code, README, SUMMARY or logs ("corrected" only in "Spearman-Brown corrected").
- No paths outside the package. File names are neutral. The README describes the ceiling the same
  way in both and does not say which groups are split.
- `._*` files hold only `com.apple.provenance`.

## 4. Key location and detection rule

- Location is right: flawed lines 72-83, line 78 subtracts the same `add` in both halves. Clean
  ceiling block is lines 72-97.
- The rule accepted a finding about the shared control mean alone ("and/or"). That is too loose.
  Splitting only the control moves the ceiling from 0.878 to 0.868; splitting the singles too gives
  0.745. So almost all of the error is in the singles. I changed the rule to require the shared
  single terms (or a prescription to split the singles).

## 5. Known true facts

All checked and true:
- Raw file: row sums equal `ncounts` exactly. The 1,000 package genes are exactly the top 1,000 by
  mean log1p(CP10k) over all 111,445 cells, in the same order. Max raw count in these genes is 3,718,
  so uint16 is safe.
- Every one of the 63,360 package cells matches a distinct raw cell (same label, total and counts).
  Every group has min(cap, raw size) cells. 105 singles (113-400), 131 doubles, 115 kept, 2,000
  controls.
- Model scores, intervals, variance explained, shares: match the run.
- Slope of I on dA+dB, all 115 pairs: -0.221.
- Feature-target coupling check: 0.425 (shared cells) vs 0.227 (other half). Confirmed.

## Changes

1. Both keys: seed-stability numbers widened from seeds 1-3 (0.750-0.752) to seeds 1-6
   (0.740-0.752; flawed 0.877-0.880).
2. B9296 detection rule: shared single terms are required; control-only is not enough. Direction
   word relaxed to "not valid (inflated, or biased)".
3. Both keys: added a minor real issue. Spearman-Brown is applied to the mean split-half r, not per
   pair. Mean of per-pair ceilings is 0.872 (flawed) and 0.710 (clean), not 0.879 / 0.752. The B3100
   detection rule and its Spearman-Brown false-alarm entry now say this point is valid, not a
   false alarm.
4. P02_BUILD.md: wrong hash suffix "...12c6" corrected to "...1892".

No package file was changed.
