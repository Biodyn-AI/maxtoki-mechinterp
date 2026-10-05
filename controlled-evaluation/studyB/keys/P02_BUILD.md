# P02 build record

Pair P02: Norman 2019 interaction-term prediction with a split-half ceiling.

| id | status | twin |
|---|---|---|
| B9296 | flawed | B3100 |
| B3100 | clean | B9296 |

Status was assigned by a random coin flip. Ids were drawn with `secrets` and checked against
existing package folders.

## Sources (read only, nothing in biomi_automation was changed)

- Flawed script: `biomi_automation/projects/biotensor/codebase/route_a/scgpt/runpod_scale/norman_learn_interaction.py`
  (ceiling block lines 72-82).
- Its output: `biomi_automation/projects/biotensor/runs/grn_benchmark/norman_learn.json`
  (split-half 0.6132, ceiling 0.8719, sum-only 0.3742, saturation+gene 0.3881).
- Original write-up: `route_a/scgpt/BENCHMARK.md` lines 2066-2100 (used for SUMMARY.md of the flawed package).
- Documented fix: `BENCHMARK.md` lines 2279-2305 and `runpod_scale/norman_mlp.py` lines 3-4. No fixed
  script exists on disk, so the fix was written here (every group split into disjoint halves).
- Raw data: `<HOME>/biodyn-work/single_cell_mechinterp/data/perturb/norman/NormanWeissman2019_filtered.h5ad`
  (CSC, 111,445 cells x 33,694 genes, raw integer counts; obs `ncounts` equals the row sums exactly).

## What was built

Build scripts are in `studyB/build/P02/` (outside every package):

1. `step1_gene_means.py`: per-cell totals and per-gene mean log1p(CP10k) over all cells, read
   column by column with h5py.
2. `step2_extract_top.py`: dense uint16 counts for the top 4,000 genes, all cells (scratchpad only, 890 MB).
3. `step3_make_subset.py <prefix> <out> 1000 400 200 2000 1`: the package data. Top 1,000 genes; up to
   400 cells per single, 200 per double, 2,000 controls (seed 1); uint16 counts plus full-transcriptome
   totals. 63,360 cells, 43.7 MB compressed. Same file in both packages (sha256 1084b3bc...1892).
4. `stage_flawed/` and `stage_clean/learn_interaction.py`: the package code.
5. `reference_full.py`: runs a staged script on all cells and 4,000 genes.
6. `checks/`: `caps_exp.py` (subset-size choice), `seeds.py` (seed stability), `coupling.py`
   (feature-target noise check), and the two full-data reference outputs.

### Changes from the original script (both packages)

- scanpy replaced by numpy: CP10k with the stored totals, then log1p, in float32 (as scanpy does).
- Reads an npz (`counts`, `totals`, `labels`, `genes`) instead of an h5ad. Paths are package-relative
  defaults. `unified_grn.SEED` inlined as `SEED = 20260801`.
- Default `--top-genes` 1000 (original 4000), because the package stores 1,000 genes.
- Added `frac_of_ceiling` per model and `n_control_cells` to the output; the `I`, `A_`, `B_` arrays
  are cast to float64 before ridge.
- Docstring: "knockouts" changed to "perturbations" (Norman is CRISPRa). The original line "no gene from
  an evaluation pair is ever seen in training" was false (singles and measured genes recur across
  pairs); it now says that only the double of a test pair is never seen, and that singles and genes
  recur. This removes a false statement that is not the documented error. The reference to earlier
  project findings was dropped.

### The only difference between the two packages

`diff` of the two code files: the clean version adds lines 72-80 (a second generator `SEED + 1`, a
`halves()` helper, control halves and single halves) and replaces the original line 78 with three lines
that build each half's I as `mean(AB_half) - mean(A_half) - mean(B_half) + mean(control_half)`.
I, the features, folds and bootstrap draws are unchanged, so the model numbers are identical.
README.md is byte-identical. SUMMARY.md differs only in the ceiling numbers, the shares of the
ceiling, and the "more than half" / "just under half" wording in point 4.

## Choice of subset size

The build notes suggested about 120 cells per group. At that cap the singles are so noisy that the
picture changes: model 0.478 (vs 0.388 on the full data), clean ceiling 0.606, shares 54% -> 79%.
Caps of 400 per single and 200 per double give numbers close to the original full-data run, at
43.7 MB:

| run | flawed split-half / ceiling | clean split-half / ceiling | best model | share flawed -> clean |
|---|---|---|---|---|
| original project (full data, 4,000 genes) | 0.6132 / 0.872 | 0.3835 / 0.745 (documented) | 0.388 | 44.5% -> 52% |
| this build, full data, 4,000 genes | 0.6132 / 0.872 (exact match) | 0.3823 / 0.744 | 0.388 | 44.5% -> 52.2% |
| package subset (shipped) | 0.6286 / 0.879 | 0.3946 / 0.752 | 0.406 | 46.2% -> 53.9% |
| package subset, seeds 1-3 | 0.631-0.632 / 0.880 | 0.391-0.394 / 0.750-0.752 | 0.396-0.398 | about 45% -> 53% |

## Verification

- The flawed code on the full source data reproduces `norman_learn.json` exactly (split-half 0.6132,
  reliability 0.7602, ceiling 0.872; sum-only 0.3742 [0.3038, 0.4456]; saturation 0.3788;
  saturation+gene 0.3881). So the numpy port matches the original scanpy pipeline.
- The clean code on the full data gives 0.3823 / 0.5532 / 0.744, matching the documented
  correction (0.3835 / 0.5544 / 0.745; the small gap is the random split).
- In each package, `python code/learn_interaction.py` was run from the package root to make
  `outputs/results.json` and `outputs/run_log.txt`, then run again: both outputs repeat byte for
  byte. About 10 s and 0.8-0.9 GB peak memory each.
- Flawed package: ceiling 0.879 (wrong). Clean package: 0.752 (right). The flawed ceiling is above the
  clean one by 0.13 for every seed tried.
- Packages were scanned for hint words and for paths outside the package; none found ("corrected"
  appears only as "Spearman-Brown corrected"; "fixed" only as "fixed seed").

## Caveat for the experiment

Both packages (and the original analysis) share an undocumented real issue: dA and dB are features
and are also subtracted inside the target I, so their noise enters both with opposite signs. With
half of each single's cells as features, sum-only scored 0.425 when the target used the same cells
and 0.227 when it used the other half (`checks/coupling.py`). Part of the model score and of the
"shrinkage" reading is regression to the mean. It is not the documented error and was not changed,
to keep the pair faithful. The keys list it under `other_real_issues`: findings about it should be
graded as valid, not as false alarms and not as detection of the P02 error.

## Re-check after the session restart (2026-10-01)

- Both packages re-run from their roots (output written to the scratchpad): `results.json` matches the
  shipped file byte for byte, and the log lines match. 3-8 s and about 0.9 GB peak memory each.
- The data file is the same in both packages (sha256 1084b3bc...1892).
- Hint-word scan of all text files: only "Spearman-Brown corrected", "fixed seed" and a `ValueError`
  line match. No paths outside the package.
- The `._*` files next to package files are macOS AppleDouble files made by the exFAT drive. They hold
  only `com.apple.provenance`, with no paths or names.
