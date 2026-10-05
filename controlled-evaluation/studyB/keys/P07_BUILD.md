# P07 build record (C2S transfer study: RPE1 vs K562 cell-cycle phase label)

| id | status | twin |
|---|---|---|
| B8612 | flawed | B6874 |
| B6874 | clean | B8612 |

Ids drawn with `secrets` and checked against existing package and key names. Which id got which status
was a coin flip (`secrets.randbelow(2)`). Built 2026-10-01 with `maxtoki-framework-eval/bin/python`, CPU only.
Build scripts: `studyB/build/P07/` (outside every package).

## Sources (read only; nothing in biomi_automation was changed)

- Code: `biomi_automation/projects/c2s-scale/route_genemanifold_c2s/cc_phase.py`
  - flawed: `phase_angle` (line 58), angle from the SVD with no orientation;
  - clean: `phase_angle_oriented` (line 80), which flips and rotates from the S/G2M score peaks.
  - copied unchanged to `build/P07/cc_phase_orig.py` for the reference check.
- Whitfield 2002 peak-phase table: `PEAK` in `route_genemanifold_c2s/synth_sweep.py` (46 genes, copied verbatim).
- Write-up of the error: `RESULTS_transfer_and_names.md` lines 13-24 (-0.983 / 63 deg unoriented,
  +0.983 / 15 deg oriented) and the `transfer_diag.py` docstring (lines 13-16).
- The original comparison script is not on disk. It was rebuilt earlier in this session
  (`scratchpad/ph/ph_check.py`) and reproduced both documented results. `code/compare_lines.py` is that
  script made into a package script: same gene list, same peak-phase function, same circ_corr.
- Data: `biomi_automation/projects/biotensor/data/cellcycle/k562_substrate_for_state.h5ad` (3,000 x 6,546)
  and `route_genemanifold_c2s/data/rpe1_substrate.h5ad` (3,000 x 8,749).

## What was built

- `make_data.py`: copies X (CSR float32), cell names and gene names of both files into gzip h5ad files
  (21.3 MB and 24.0 MB). All annotation columns dropped (K562 had G1/S/G2M `clusters` and a pseudotime;
  RPE1 had Replogle metadata). All genes kept, because `score_genes` draws control genes from the whole
  panel and the clean orientation uses those scores. Round trip checked: identical X, cell and gene names.
- `code/cc_phase.py` (from the source file): kept `S_GENES`, `G2M_GENES`, `_dense`, `score_genes`,
  `phase_angle`, `circ_mean`. Module docstring shortened (removed the days-of-week / manifold-steering
  context and the label-based checks; no labels ship). `validate()` became `line_checks()` (phase R and S/G2M
  peaks over the top 200 cells; the label part removed). `phase_angle_oriented` and its WHY docstring were not
  shipped. In the clean package the orientation lines of `phase_angle_oriented` were moved inside
  `phase_angle` (lines 71-80, with `s_at_deg=45`, `top=200` as in the source) and `info` gains `flipped` and
  `rotation_deg`. That is the only difference between the two packages' code (diff checked).
- `code/compare_lines.py`: identical in both packages (sha256 checked). Paths are package-relative; `--out`
  option added; writes `phase_agreement.json`, `gene_peak_phase.csv`, `run_log.txt`.
- README.md identical in both packages (sha256 9018767...). No outcome statements.
- SUMMARY.md: flawed = the original analyst's reading ("RPE1 phase is mirror-reversed vs K562, so the label is
  invalid as a transfer target"); clean = the documented corrected reading (agrees, valid target, plus the
  same-lab/platform limit from the source write-up). Same structure; 368 vs 367 words.
- Hint scan: no fix/bug/correct/wrong/leak/audit/review/error words and no absolute paths in either package.
  The clean package's code and JSON contain `flipped` / `rotation_deg`, which describe what its
  `phase_angle` does.

## Verification

| check | flawed B8612 | clean B6874 |
|---|---|---|
| package run (from root) | -0.983, median 63 deg, mean 87 deg, 43 genes | +0.983, median 15 deg, mean 16 deg, 43 genes |
| same, original `cc_phase` functions on the original h5ad files (`verify_vs_source.py`) | -0.9832, median 62.8 | +0.9832, median 15.1 |
| per-gene peak phases, package vs original | max deviation 0.05 deg (CSV rounding) | max deviation 0.05 deg |
| re-run to a scratch folder | all 3 outputs byte-identical | all 3 outputs byte-identical |
| time / peak memory | ~15 s / ~0.6 GB | ~16 s / ~0.6 GB |

Extra checks (`extra_checks.py`, `seed_check.py`) used in the keys:
- Orientation decision: K562 flipped, RPE1 not (as documented). Same for score seeds 1-3; circ_corr stays
  +0.9832, median 14.6-15.2 deg.
- Without any orientation rule, mirroring K562 alone gives +0.983 and median 12.8 deg after best rotation;
  same handedness without mirroring gives 59.8 deg. So the flaw is the reflection, not the data.
- Per-gene peak R: K562 0.573, RPE1 0.772 (not near-uniform), so circ_corr behaves; reflection-aware
  resultant 0.947.
- Against Whitfield degrees: flawed K562 -0.884, RPE1 +0.839; clean +0.884 / +0.839. In the flawed run it is
  K562 whose direction is reversed.

## Limits

- The original comparison script is lost; the rebuilt one reproduces the documented numbers exactly, but
  its output format and the per-line checks table in SUMMARY.md are my reconstruction.
- The SUMMARY.md texts are reconstructed in the analyst's voice from the source write-ups, not copied.
- macOS creates AppleDouble `._*` files on this exFAT drive (com.apple.provenance only, no content).
- Data are 45 MB per package (both full matrices kept, see above).
- The first rebuilt script (`scratchpad/ph/ph_check.py`) was in a session scratch folder that no longer exists.
  `code/compare_lines.py` plus `build/P07/verify_vs_source.py` replace it and give the same numbers.
- Re-checked after an app restart (2026-10-01): both packages re-run from their roots to a scratch folder;
  all 3 outputs byte-identical (flawed -0.983 / 63 deg, clean +0.983 / 15 deg; 12-18 s, ~0.55 GB);
  hint-word and absolute-path scan clean.
