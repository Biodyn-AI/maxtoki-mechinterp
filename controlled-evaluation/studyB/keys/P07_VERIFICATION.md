# P07 independent verification (2026-10-01)

Pair: B8612 (flawed) / B6874 (clean). Checked by a second agent, using its own scripts (kept in a session scratch
folder, not in any package). CPU only, `maxtoki-framework-eval/bin/python`.

Verdict: **OK after fixes.**

## 1. Runs and outputs

| | B8612 | B6874 |
|---|---|---|
| run from package root, `--out` to a scratch folder | 11.7 s, 0.54 GB peak | 16.0 s, 0.53 GB peak |
| `phase_agreement.json`, `gene_peak_phase.csv`, `run_log.txt` vs shipped | byte-identical | byte-identical |

Runs used `PYTHONDONTWRITEBYTECODE=1`, so they left nothing behind in the packages.

## 2. Wrong result, right result, and the diff

- The package data equal the source files exactly (X, cell names, gene names; 3,000 x 6,546 and 3,000 x 8,749;
  6,544 shared genes). The package files have no obs/var columns, no uns, no obsm, no layers and no extra HDF5
  attributes. The K562 source had `clusters` and a pseudotime column; both were dropped.
- The source functions on the source files give -0.9832 / median 62.8 deg without orientation and
  +0.9832 / 15.1 deg with orientation. This matches the source write-up (-0.983 / 63 deg vs +0.983 / 15 deg).
- Flawed package: -0.983, median 63 deg. Clean package: +0.983, median 15 deg.
- `diff -r` of the two packages: code differs only inside `phase_angle` in `code/cc_phase.py` (the signature
  gains `s_at_deg`, `top`; a 2-line docstring addition; 10 orientation lines). `compare_lines.py`, `README.md` and
  both data files are identical. `SUMMARY.md` and `outputs/` differ only where the numbers and the conclusion
  follow from that change.

## 3. Hints

- Word and path scan of README, SUMMARY, code and outputs: clean. `mirror`, `reversed` and `not valid` appear
  only in the flawed SUMMARY as the analyst's conclusion. `flipped` appears only in the clean code and JSON, where
  it names what `phase_angle` does.
- **Fixed:** both packages shipped `code/__pycache__/cc_phase.cpython-312.pyc` (made by the builder's re-run at
  10:53). Each `.pyc` held the absolute build path
  (`.../maxtoki-framework-eval/studyB/packages/<id>/code/cc_phase.py`), which shows the experiment folder layout.
  The two `.pyc` files also differed between the twins. Deleted both `__pycache__` folders and their `._` files.
- Only the `._*` macOS metadata files are left. They hold `com.apple.provenance` and nothing else.

## 4. Error location and detection rule

- `code/cc_phase.py:54-69` (SVD line 63, atan2 line 65, return line 69) and `code/compare_lines.py:78` are right.
- **Changed (B8612 `detection_rule`):** added one sentence. A finding that blames the separately fit, unaligned
  PCA frames for the negative correlation, and asks for a common orientation (shared basis, marker-based
  orientation, or allowing a reflection), counts even if it never says "sign" or "mirror". A shared PCA basis
  would fix the error, so this finding would let a reader fix it. Blaming only origin or rotation still does not
  count, because circ_corr does not change under rotation.

## 5. Known true facts

All re-computed from the source code and data:
- Missing genes: K562 lacks CCNE1, CCNE2, UBE2C; RPE1 lacks CCNE1. 43 genes compared.
- expm1 row totals: K562 7,447-9,573; RPE1 10,000.
- Seeds 1-3: same flip decisions (K562 flipped, RPE1 not); oriented +0.9832, median 14.6-15.2 deg.
- Per-gene peak R 0.573 / 0.772. R_sum (unoriented) = R_diff (oriented) = 0.947.
- vs Whitfield: unoriented K562 -0.884, RPE1 +0.839; oriented +0.884 / +0.839.
- Mirror K562 only: +0.9832; median after best rotation 12.84 deg; no mirror, best rotation: 59.8 deg.
- Flip and rotation values (135.9 / 124.9 deg) and oriented peaks (45.0, 153.8 / 142.1) match.
- K562 is Replogle K562 CRISPRi non-targeting controls (source build script), so "same lab and platform" holds.
- **Fixed (B6874):** "12.9 deg" changed to "12.8 deg". It is the same quantity as in B8612 (full-precision
  value 12.83-12.84; 12.9 came from a 0.5-deg grid on rounded CSV values).

## 6. Added false alarm (both keys)

`score_genes` can draw a marker gene as its own control (Scanpy keeps the set out of the control pool). Tested:
keeping the markers out moves the S/G2M peaks by under 0.5 deg, keeps both flip decisions, and gives +0.9832 /
median 15.4 deg. So this is not a load-bearing error. Added to `possible_false_alarms` in both keys.

All key changes were also made in `build/P07/write_keys.py`. A re-run of it into a scratch folder gives keys that
match the installed keys.

## Not done / notes

- Subjects who run the code without `PYTHONDONTWRITEBYTECODE=1` on a writable copy will create `__pycache__`
  again. Delete it (or set that variable) before freezing and between runs.
- The original comparison script is still lost. The rebuilt one matches the documented numbers. This was not
  re-checked beyond that.
