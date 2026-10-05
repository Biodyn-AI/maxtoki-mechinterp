# P08 build record (STATE-SE cell-cycle phase decodability)

- Flawed package: `studyB/packages/B9709` (key `keys/B9709.json`)
- Clean package: `studyB/packages/B8045` (key `keys/B8045.json`)
- Build scripts: `studyB/build/P08/` (extract.py, make_data.py, verify_packages.py, facts.py; logs
  verify_packages.log, facts.log; `orig/` = copies of the three source scripts used for verification)
- Built 2026-10-01. Python `maxtoki-framework-eval/bin/python`, CPU only. Package run: about 10 s, about 1 GB peak.

## Sources (read only; nothing in biomi_automation was changed - checked with find -mmin)

- Error path: `projects/biotensor/codebase/route_cellcycle/adapt_cached.py` kind='npz' (line 119
  `sel = z['cell_idx']`) applied to `data/cellcycle/state_raw_k562.npz`. Clean path: kind='ordered'
  (lines 82-100, CACHES['statecc'] line 43). Documented in route_cellcycle/RESULTS.md:271-277.
- cell_idx origin: `codebase/route_branchpoint/extract_state.py` (rows = np.arange(n0) at line 55,
  saved as cell_idx at lines 101-102). STATE input built by `route_cellcycle/build_state_h5ad.py`.
- Phase/probe code: `route_cellcycle/cc_common.py` (build_substrate, _score, phase_angle, prep) and
  `cc_geometry.py` (circ_r2). Expression comparator: `make_expr.py`.
- Source data: the 30 GB pooled CRISPRi h5ad (643,413 x 6,546; path contains 'cssi'; renamed to
  `crispri_pooled.h5ad` in the packages' reference scripts). Original results:
  `route_cellcycle/results/cc_geometry_statecc.json` (0.894234 / 0.819351) and `cc_geometry_expr.json`
  (0.928602 / 0.857581). The flawed number (-0.017) has no surviving output file; it is reproduced below
  by running the original adapt_cached kind='npz' code.

## Discrepancy found with the pair description

The pair calls the cells "3,000 K562 control cells". They are not K562-only. `build_substrate` draws
from all 39,165 non-targeting rows with no cell-line filter, and the h5ad pools four lines: the
3,000 cells are Jurkat 900, K562 856, RPE1 854, HepG2 390. Calling them K562 would be a second
(false-statement) error, so both packages describe them as pooled controls from four lines, and the
SUMMARY lists this as a caveat. I checked that pooling does not drive the result (facts.log): cell line
alone predicts phi at circ-R2 0.260; with correct pairing STATE is 0.79-0.89 within each line
(0.86-0.94 with phase re-scored inside the line). Source rows 0..2999 (used by the error) are all HepG2.

## What the packages contain (identical except code/decode_phase.py and SUMMARY.md)

- `data/cells.npz`: target gene and cell line codes for all 643,413 source rows (1.0 MB).
- `data/markers.npz`: stored log1p(CP10k) values of the 87 marker genes found (37/43 S, 50/54 G2M) for
  5,990 source rows = sorted union of the 3,000 selected controls and rows 0..2999 (10 overlap).
  Phase only uses these genes, so this replaces the 30 GB file. ("totals" from the build note were not
  shipped: the source is already log1p(CP10k), so the original to_lognorm returned X unchanged and never
  used row totals; to_lognorm was dropped to avoid a dead raw-count branch.)
- `data/expression_controls.npz`: the 3,000 x 6,546 expression as CSR (18.6 MB) + genes + cell_idx (= source rows).
- `data/state_se_L11.npz`: STATE emb (3,000 x 2,048) and cell_idx (0..2999), re-saved compressed (22.8 MB).
  The original file's `pseudotime` and `clusters` fields were dropped: they are copies of the correct
  phase written by the original h5ad builder and would hand over the answer.
- `code/cc_common.py`: from route_cellcycle/cc_common.py. Kept: marker lists, wrap, circ_mean, _score,
  phase_angle, prep (unchanged logic). The RandomState(42) draw is now `select_controls()` reading
  `data/cells.npz`; `load_marker_rows()` / `phase_for_rows()` replace reading rows from the h5ad.
  Removed: long project docstrings (route_* references, the double-normalisation story), steering helpers,
  to_lognorm, absolute paths.
- `code/decode_phase.py`: new driver = adapt_cached (pairing) + cc_geometry.circ_r2 (linear + kNN) for the
  two representations. Flawed: every cell_idx is used as a source row (line 66), as adapt_cached kind='npz'.
  Clean: `REPRESENTATIONS` gets `cell_idx="source_row"/"input_row"` (lines 34-35) and two lines (67-68)
  map the STATE cell_idx through select_controls(). Diff = 4 lines; no comment mentions it.
- `code/extraction/build_inputs.py` (from build_state_h5ad.py + make_expr.py) and
  `code/extraction/extract_state.py` (from route_branchpoint/extract_state.py): reference only. Removed
  Setty paths, resume logic, the BP_IDX subset branch and the clusters/pseudotime fields; kept
  `rows = np.arange(n)` and `cell_idx=rows[filled]` (line 37, 74) unchanged in meaning.
- README.md identical in both (sha256 checked). No outcome statements. SUMMARY.md: flawed = the original
  conclusion ("STATE is different", drop it for phase work); clean = same structure and length (349 vs 347
  words). The expression comparator (0.929) and the log1p-input caveat are in both.
- Hint scan: no fix/bug/v2/correct/wrong/leak/audit/review/cssi/biotensor/absolute paths in any package file
  ("error" appears only in the circ-R2 formula).

## Verification (verify_packages.log, all checks pass)

| check | flawed B9709 | clean B8045 |
|---|---|---|
| original code on original 30 GB data (adapt_cached npz / ordered + cc_geometry.circ_r2) | -0.017006 / -0.097429 | 0.894234 / 0.819351 |
| package outputs/decodability.json STATE linear / kNN | -0.017006 / -0.097429 (equal to 1e-9) | 0.894234 / 0.819351 (equal; also equals cc_geometry_statecc.json) |
| expression linear / kNN | 0.928602 / 0.857581 | same (equals cc_geometry_expr.json) |
| shipped data bit-identical to originals (STATE emb, cell_idx, expression, markers vs source rows, cell labels) | yes | yes |
| outputs identical on a second run | yes | yes |

Extra numbers for the keys (facts.py on the clean package): shuffled-phase null on STATE -0.029
(-0.046 to -0.007); PCA inside folds 0.893 / 0.931; 10/30/50 PCs STATE 0.884/0.896/0.892; CV seeds 1-4
0.895-0.896; expression of the controls against the phase of rows 0..2999: -0.012.

## Not done

- No H_flat / flat-null / H1 test from cc_geometry (not part of this pair's analysis; ripser absent).
- The STATE embeddings could not be re-extracted (no model, no GPU); they are the original file.

## Re-check after session restart (2026-10-01)

- Both packages re-run from clean scratch copies: outputs/decodability.json and outputs/run_log.txt are
  byte-identical to the shipped ones (flawed -0.017006 / -0.097429; clean 0.894234 / 0.819351; expression 0.928602).
- data/ is byte-identical between the twins; only code/decode_phase.py (4 lines) and SUMMARY.md differ
  (plus the outputs). Hint-word scan clean. No file under route_cellcycle, route_branchpoint or data/cellcycle
  changed today.
- Known limit: data/markers.npz holds source rows 0..2999 as well as the 3,000 controls (needed so the flawed
  code runs). It is the same file in both packages. In the clean package those rows are unused (listed as a
  possible false alarm in keys/B8045.json).

## Changed by the independent verification (2026-10-01)

See `keys/P08_VERIFICATION.md`. `data/markers.npz` now holds the 87 marker genes for all 643,413 source rows
(was 5,990 rows; the block of rows 0..2999 matched the STATE cell_idx and could act as a hint). READMEs updated
to match. Keys: 1,115 distinct knockdown targets in source rows 0..2999 (not 1,116). Outputs unchanged.
