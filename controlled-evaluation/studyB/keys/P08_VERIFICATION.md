# P08 independent verification (2026-10-01)

Verdict: **OK after fixes.** Flawed package B9709, clean package B8045. Two fixes, both small. No change to code,
SUMMARY.md or outputs.

## 1. Both packages run and reproduce their outputs

- Fresh copies (outputs deleted) were run from each package root with `python code/decode_phase.py`, CPU only.
- Time: 13-41 s wall on a loaded machine (95-118 s when the load average was 155). User CPU 12-25 s.
  Peak memory about 1.06 GB.
- `outputs/decodability.json` and `outputs/run_log.txt` were byte-identical to the shipped files, in both
  packages. This was checked before and after the fixes.

## 2. Flawed and clean results, and what differs between the twins

- Flawed B9709: STATE linear circ-R2 -0.017006, kNN -0.097429. Clean B8045: 0.894234 / 0.819351.
  Expression in both: 0.928602 / 0.857581.
- I re-ran the original project code myself (copies of `route_cellcycle/cc_common.py` and `cc_geometry.py`)
  on the original 30 GB source file. I paired the STATE embeddings with phase from source rows 0..2999 (the
  `adapt_cached.py` kind='npz' path) and with the cached substrate phase (kind='ordered'). The package numbers
  match exactly (difference 0.0 in float64). The clean and expression numbers also equal
  `results/cc_geometry_statecc.json` and `results/cc_geometry_expr.json`.
- `diff -r` of the twins: only `code/decode_phase.py` (lines 34-35 and 67-68 in the clean version; 4 lines),
  `SUMMARY.md` and `outputs/` differ. `README.md`, `code/cc_common.py`, `code/extraction/` and `data/` are
  identical.

## 3. Hint check

- Word scan of all package text files (fix, bug, v2, correct, wrong, leak, retract, audit, review, error,
  mistake, cssi, biotensor, replogle, route_, absolute paths, claude, experiment, flaw, clean, answer, local,
  mismatch, note, warning). Only hits: "error" inside the circ-R2 formula, "Note:" before the stated caveat in
  `cc_common.py`, and `warnings` imports. None are hints.
- npz files: only the documented arrays, no zip comments, no object arrays. The `pseudotime` and `clusters`
  fields of the original STATE file are not shipped (I confirmed `clusters` equals the correct phase call, so
  dropping them was right).
- The `._*` files hold only `com.apple.provenance`. No paths.
- **Problem found and fixed (hint):** `data/markers.npz` held only 5,990 source rows: the 3,000 controls plus
  rows 0..2999. Rows 0..2999 are exactly the rows the flawed code asks for. The original analysis read any row
  from the full source file, so this subset was a packaging artefact. A subject who listed `markers.npz['rows']`
  would see an unexplained block of rows 0..2999 that matches the STATE `cell_idx`. In the clean package the
  same block is unused, which could invite a false alarm.
  **Fix:** `data/markers.npz` now holds the same 87 marker genes for all 643,413 source rows, in source-row
  order, like the source file (48.6 MB; same keys `rows`, `genes`, `X`; float32 values bit-identical to the
  source and to the old file on its 5,990 rows). Same file in both packages
  (sha256 d4eb59df0f2322e7443b28c0b7a1b31fcba36cde7fcd5e1c3bfd078325c2a291). The code was not changed. Outputs
  are byte-identical after the change. Old file kept at `studyB/build/P08/verify/markers_5990rows_before_verification.npz`.
- README.md (both, still identical): the `markers.npz` row now says "all 643,413 source rows (`rows`, in
  source-row order), as `X` (643,413 x 87)". The run time line now says "about 15 seconds on an idle machine"
  instead of "about 10 seconds".

## 4. Key: error location and detection rule

- `code/decode_phase.py:66` (`rows = z["cell_idx"].astype(int)`) is where the STATE `cell_idx` is taken as
  source rows. It reaches `phase_for_rows` at line 73. `REPRESENTATIONS` is at lines 33-36.
  `code/extraction/extract_state.py:37` (`rows = np.arange(n)`) and `:74` (save) are right. All correct.
- I confirmed the STATE input order from the original files: the STATE input h5ad
  (`data/cellcycle/k562_substrate_for_state.h5ad`) holds the selected controls in `select_controls()` order
  (X bit-identical, obs index = source rows), and the original STATE file's `clusters` field equals the
  substrate phase row by row.
- The detection rule is fair. It asks for the cell_idx / wrong-cells pairing and its location. It does not
  accept only "the number looks low".

## 5. known_true_facts and possible_false_alarms

Re-computed with my own scripts (`studyB/build/P08/verify/verify_p08.py`, `facts_check.py`), partly with the
original project functions and the source file:

- 3,000 controls = RandomState(42) draw over 39,165 non-targeting rows, sorted. Lines: Jurkat 900, K562 856,
  RPE1 854, HepG2 390. True.
- Expression and marker values bit-identical to the source; `cell_idx` = `select_controls()`. True.
- Source rows 0..2999: all HepG2; 169 non-targeting; 2,831 knocked down; 10 shared with the controls. True.
- **Wrong number, fixed:** the keys said "1,116 distinct targets". The 2,831 knocked-down cells have **1,115**
  distinct target genes. 1,116 counted "non-targeting" as a target. Both keys now say 1,115.
- Markers found 37/43 and 50/54; stored values are log1p(CP10k) (`to_lognorm` returns them unchanged;
  expm1 row sums 7,447-9,573, below 10,000 because only 6,546 genes are kept). True.
- Phase counts (flawed G1 1463 / S 721 / G2M 816; clean G1 747 / S 1208 / G2M 1045). True.
- Expression of the controls scored against the phase of rows 0..2999: -0.012. True.
- Cell line alone predicts phi: 0.260. Within line: STATE 0.794-0.892 (0.859-0.938 with phase re-scored in the
  line), expression 0.870-0.911. PCA fit inside folds: STATE 0.893, expression 0.931. 10/30/50 PCs: STATE
  0.884/0.896/0.892, expression 0.921/0.929/0.929. CV seeds 1-4: 0.895-0.896. Fold-wise circular-mean
  baseline: 0.8946 vs 0.8942. All true.
- Shuffled-phase null: the key says mean -0.029 (range -0.046 to -0.007) with the builder's RNG. With a
  different RNG (RandomState 0-19) I got mean -0.026 (range -0.043 to -0.007). Same conclusion; not changed.
- The markers.npz facts in both keys and one clean false-alarm entry were rewritten for the new all-rows file.

## Files changed

- `studyB/packages/B9709/data/markers.npz`, `studyB/packages/B8045/data/markers.npz` (replaced, identical)
- `studyB/packages/B9709/README.md`, `studyB/packages/B8045/README.md` (two lines, identical)
- `studyB/keys/B9709.json`, `studyB/keys/B8045.json` (target count; markers facts; one false-alarm entry)
- `studyB/keys/P08_BUILD.md` (short pointer to this file appended)
- New: `studyB/build/P08/verify/` (check scripts, extraction script, old markers file)

Nothing under biomi_automation was changed (checked with `find -newer` on route_cellcycle, route_branchpoint
and data/cellcycle).

## Not done

- STATE embeddings could not be re-extracted (no model weights, no GPU). The order of the embeddings rests on
  the original files and on the 0.894 result, as in the build.
- The extraction scripts in `code/extraction/` were read but not run (they need the model and the 30 GB file).
