# P06 build record (C2S-2B cell-cycle readout, K562 -> RPE1 transfer)

- Flawed package: `studyB/packages/B1144` (key `keys/B1144.json`)
- Clean package: `studyB/packages/B2101` (key `keys/B2101.json`)
- Build folder: `studyB/build/P06/` (extract_data.py, common/, flawed/, clean/, ref/, verify_packages.py + .log)
- Built 2026-10-01. Python: `maxtoki-framework-eval/bin/python` (numpy 1.26.4, scikit-learn 1.8.0, anndata 0.12.10). CPU only.
- Ids drawn with `secrets`; which id got which status was a coin flip (B1144 = flawed, B2101 = clean).

## Sources (read only; nothing in biomi_automation was changed)

- Code: `projects/c2s-scale/route_genemanifold_c2s/transfer_test.py` (current, R_diff version), `transfer_diag.py`, `cc_phase.py`.
  All copied to `build/P06/` first. Python ran with PYTHONDONTWRITEBYTECODE=1, so no `__pycache__` was written anywhere.
- Data: K562 `projects/biotensor/data/cellcycle/k562_substrate_for_state.h5ad` (3000 x 6546) and RPE1
  `route_genemanifold_c2s/data/rpe1_substrate.h5ad` (3000 x 8749); activations `data/act_k562`, `data/act_rpe1`
  (layer 21, 3000 x 2304 float32, last-token residual stream, top-512-gene cell sentences with prompt).
- Reference outputs: `results/transfer_rpe1.json` + `results/v3.log` "EXP B-prime" (the flawed V3 run),
  `results/transfer_diag.json`, and `RESULTS_transfer_and_names.md` (R_diff table).

## Data (identical in both packages)

`extract_data.py` rewrote each h5ad with only X (CSR float32), obs names and var names (gzip), and copied
the activation `.npy`, `row_cell_ids.npy` and `manifest.json`. Round trip checked: X, var names and obs names
identical to the source files. `row_cell_ids` is the identity 0..2999 in both lines. Dropped: K562 obs
(`clusters`, `palantir_pseudotime`), all RPE1 obs/var annotation columns, `row_gene_names.json` (all `<CELL>`).
96 MB per package.

## Code changes

- `code/cc_phase.py` (identical in both): kept `score_genes`, `phase_angle`, `circ_mean`, `phase_angle_oriented`.
  Removed `validate`, the `__main__` block (absolute path) and the docstring text that described a phase
  readout "scoring strongly NEGATIVE while nothing is wrong" and "Goodfire days-of-week". Logic unchanged.
- `code/transfer_test.py`, both versions: package-relative defaults (`--out` option kept); neutral docstring
  (dropped the history of rejected targets, the "first run of this script did exactly that" comment, and the
  "does the model generalise better than expression" framing). The target-recentered arm was dropped (build
  note). The winner/margin logic was replaced by a per-readout pass/fail list, because the clean run would
  otherwise print "WINNER: EXPRESSION", which is not an information-matched comparison (P04).
- Flawed version = V3 reconstruction. The V3 source is not on disk (no git; the cached .pyc files are from
  the later version). Rebuilt from the V3 JSON keys and log format: `circ_corr` (line 39) as within and transfer
  score (lines 67, 74), raw median error, retention, gate `circ_corr > 0 and median error < 90` (line 131,
  assumed 90 deg at line 36), and the V3 failure message and verdict string. circ_corr has a one-line neutral
  docstring; the retraction docstring is gone.
- Clean version = current script metric: `circ_R_diff`, `med_err_rot` (with offset), `circ_R`, measured
  constant and random floors, gate R_diff > max(0.15, 2 x constant) and error < constant error. No `legacy_circ_corr`,
  no `_RETRACTED` key, no `verdict_note`. The unexecuted failure branch uses the current script's neutral text.
- README.md identical in both (sha256 3dabfcc1...). It names no statistic; it points to the code.
- SUMMARY.md: flawed = V3 reading (no valid transfer; RPE1 label is not cell cycle), 410 words. Clean =
  "transfer works for both readouts", does not rank model vs expression, 425 words. Same sections.

## Verification (`build/P06/verify_packages.log`)

| check | flawed B1144 | clean B2101 |
|---|---|---|
| within (K562 CV) model / expr | circ_corr 0.897 / 0.919 | R_diff 0.886 / 0.913 |
| transfer model / expr | circ_corr -0.814 / -0.796 | R_diff 0.789 / 0.878 |
| median error model / expr | 24.0 / 16.0 deg (raw) | 23.3 / 16.0 deg (after rotation of -3.8 / -2.5 deg) |
| retention | -0.91 / -0.87 | 0.89 / 0.96 |
| chance floors | not measured (90 deg assumed) | constant 0.043 / 86 deg, random 0.017 / 88 deg |
| verdict | NEITHER, label invalid | VALID: MODEL, EXPRESSION |
| matches source | V3 transfer_rpe1.json, max diff 8e-6 | current script rerun (ref/), identical |
| rerun determinism | identical JSON and log | identical JSON and log |

Run time 25-50 s, peak memory 1.5 GB. Extra diagnostics on package data: prediction mean 227 / 235 deg vs
true mean 12 deg (215 deg apart, which flips sin()); 81.5% / 90.7% of cells within 45 deg; circ_corr with one
shared reference angle +0.848 / +0.899; refit on RPE1 (CV) circ_corr +0.928 / +0.958, R_diff 0.909 / 0.952.
All match transfer_diag.json and the RESULTS file.

## Limits

- The flawed script is a reconstruction. Its outputs match the V3 JSON and log line for line, but the exact
  V3 gate expression is inferred.
- The P04 pair uses the same data. This build made its own copy of the data and did not coordinate ids.
- AppleDouble `._*` files appear on this exFAT drive in every folder; they carry no content.

## Re-check after restart (2026-10-01)

The session was interrupted after the build. On resume, both packages were re-run from their package roots
(output to the scratchpad, PYTHONDONTWRITEBYTECODE=1). The JSON was identical to `outputs/transfer_rpe1.json`
and the log identical to `outputs/run_log.txt` (except the `[done]` path line) in both. A word scan of both
packages found no hint words, absolute paths or cache files. Keys and line numbers (39-43, 67, 74, 36, 131)
match the flawed package code.
