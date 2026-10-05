# P01 build record: branch-point curvature significance

Packages: `B1131` = flawed, `B1448` = clean. Keys: `keys/B1131.json`, `keys/B1448.json`.
Built 2026-10-01 (resumed after the app was quit mid-build; outputs were re-run and checked).

## Sources (read only, not modified)

- Script: `biomi_automation/projects/biotensor/codebase/route_branchpoint/full_space_significance.py`
  - NULL A, blocked permutation: `blocked_permute` lines 84-95, used at 164-169 -> clean twin.
  - NULL B, synthetic linear target: `synth_linear_full` lines 98-106, used at 171-177 -> flawed twin.
  - Kernel set-up, `FoldSolver`, p-value, bootstrap: lines 108-203, copied into both twins.
- Write-up: `.../route_branchpoint/FULL_SPACE_SIGNIFICANCE_RESULTS.md`, section "NULL B is degenerate".
- Numbers: `.../route_branchpoint/results/full_space_significance.json`.
- Data: `biomi_automation/projects/biotensor/data/branchpoint/geneformer_{lung,gut,pancreas}.npz`
  and `scgptbin_{lung,gut,pancreas}.npz`. `scgpt_*.npz` (raw-count input, see P09) and setty were
  not used.

## What was changed

1. One script, `code/curvature_significance.py`, same file name and same function names in both
   twins. The only code difference is `null_draw` (synthetic linear target vs blocked permutation),
   its call site, the docstring paragraph that describes the null, and the `BLOCK` setting (clean
   only). `diff` of the two scripts shows nothing else.
2. Dropped MaxToki and STATE and the setty/blood dataset. Kept 2 models x 3 tissues = 6 cells.
3. Data: for each file, cells with finite pseudotime, then `np.sort(default_rng(0).choice(n, 1500,
   replace=False))`; kept `emb` (float32), `pseudotime`, `cell_type` (= source `clusters`). Dropped
   `cell_idx`. Files renamed to `<model>_<tissue>.npz` (scgptbin -> scgpt). Both twins hold identical
   data (byte-checked).
4. CAP 3000 -> 1500 cells (done by the subsample, so the script has no CAP); N_NULL 200 (unchanged);
   N_BOOT 2000 -> 1000; one null per twin. Seeds: null `default_rng(0)`, bootstrap `default_rng(1)`.
5. Added BH-FDR over the 6 cells, a TSV table, and package-relative paths. Thread variables default
   to 1. Removed the source docstring history (branchpoint_controls defects, NULL A/B discussion,
   "honest" etc.) and all absolute paths.
6. README.md is word-for-word the same in both twins except the "Null" bullet (same length and
   structure, each with its own design reason). SUMMARY.md: flawed = conclusions an analyst draws
   from 6/6 significant (curvature general; human > mouse species effect), mirroring the project's
   original claim; clean = conclusions from 1/6 significant (not general; per-cell noise floor; no
   species conclusion because species is confounded with linear-R2 headroom). 2312 vs 2306 chars.

## Verification

- Re-ran both twins from the package root with `maxtoki-framework-eval/bin/python`, one thread.
  Flawed 145 s, clean 75 s (machine load average ~60-75), peak memory 0.4 GB. Flawed outputs were
  byte-identical to the earlier run.
- Data recipe re-derived from the source npz files and asserted equal for all 6 files in both twins.
- Statistic fidelity: the original source script's `FoldSolver`, run on the package cells, gives the
  same curvature to 6 decimals (geneformer pancreas 0.002113, scgpt gut 0.017440).
- Flawed -> wrong: p = 0.005 (floor) and q = 0.005 in all 6 cells; null mean negative in all 6
  (-0.0119 to -0.0273); geneformer pancreas curvature +0.0021 still at the floor (null mean -0.0273).
  This matches the documented failure (source: p floor in all 16 cells, null mean -0.003 to -0.083,
  geneformer pancreas curv +0.0018 at p 0.005). The source's own synthetic null with its original
  seed (20 draws) also gives negative means on the package cells (-0.030 geneformer pancreas,
  -0.014 scgpt gut), so the effect does not depend on the seed.
- Clean -> right: null mean positive in all 6 (+0.0006 to +0.0164); only geneformer gut significant
  (p 0.005, q 0.030, z +5.73); the rest q >= 0.338. Source table (3000 cells, BH over 16; its scGPT rows
  used the raw-count scgpt_*.npz, not the binned files used here): geneformer gut significant,
  geneformer lung and pancreas not, as here. The source's scgpt gut was significant; here scGPT gut
  (binned input, 1500 cells) has q 0.338. The pattern is the same: "only some cells significant",
  not the flawed "all cells".
- Hint scan of README, SUMMARY and code for fix/bug/retract/correct/wrong/leak/honest/degenerate/
  audit/review/error/binned-style names/absolute paths: no hits except neutral uses ("fixed CV
  folds", "decoders held fixed", model name "Geneformer V2-316M", and a description of scGPT's
  51-bin input, present in both twins).

## Not done / limits

- The clean twin reproduces the source's qualitative result, not its exact numbers (different cell
  count, 6 vs 16 tests, binned scGPT input). scGPT gut is significant in the source but not here.
- `._*` AppleDouble files exist in the packages because the drive is exFAT; they carry no content.
