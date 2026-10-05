# P12 build record (cell-cycle phase decodability and flat-circle test; scGPT, Geneformer, expression; planted P9)

| id | status | twin |
|---|---|---|
| B7037 | flawed | B4328 |
| B4328 | clean | B7037 |

The two ids were drawn in the first session (folders created at 08:25:58, the same second as `studyB/build/P12/`)
and checked against existing names. The session was cut off before anything was put in them. After the restart
the same two ids were reused, and which id got which status was a coin flip (`secrets.randbelow(2)`; recorded in
`build/P12/id_assignment.txt`). Built 2026-10-01 with `maxtoki-framework-eval/bin/python`, CPU only. Build files:
`studyB/build/P12/` (outside every package). The two packages differ only in `SUMMARY.md` (md5 of the other 20
files checked equal).

## Sources (read only; nothing in biomi_automation was changed)

- Code: `biomi_automation/projects/biotensor/codebase/route_cellcycle/cc_geometry.py` (`circ_r2`, `loop_tangents`,
  `turning_stats`, `flat_null`, `run`) and `cc_common.py` (`wrap`, `circ_mean`, `unit`, `ang`, `_score`,
  `phase_angle`, `prep`, Tirosh gene lists). Label construction checked against `build_substrate` (cc_common),
  `adapt_cached.py` (Geneformer: own 2,000-cell label) and `make_expr.py`.
- Data: `projects/biotensor/data/cellcycle/{scgptcc_k562,geneformer_k562,expr_k562}.npz`, gene names from
  `k562_cc_substrate.npz`.
- Results: `route_cellcycle/results/cc_geometry_{scgptcc,geneformer,expr}.json`, `cc_summary.json`; write-up
  `route_cellcycle/RESULTS.md` sections 1 and 10 (linear circ-R2 0.892 / 0.844 / 0.929; flat circle not rejected).
- Model facts for the README: scGPT whole-human, layer 11 of 12, mean over gene tokens, 51-bin input, top 1,200
  genes (`extract_scgpt_cc.py`); Geneformer V2-316M layer 11 of 18, per-cell mean (`adapt_cached.py`,
  `route_manifold/MODEL_NOTES.md`).
- Both SUMMARY.md texts are written by me in the analyst's voice. The flawed one carries the planted overreach.
  The clean one follows the source's own reading (RESULTS.md section 10: the data carry the loop; the models do
  not add to it).

## What was built

- `data/`: the three npz files with fields `emb` (float32), `phi`, `s_score`, `g2m_score`, `cc_phase`,
  `cell_idx`; `expr_k562.npz` also has `genes`. Saved compressed (expression 78 MB -> 11 MB; 25 MB per package).
  Every array equals the source exactly (checked after the build).
- `code/cc_common.py`: `wrap`, `circ_mean`, `unit`, `ang`, `prep`, `emb_path` and the phase-label functions,
  standalone. The docstring says plainly that phi is built from marker-gene expression only. Dropped: substrate
  loader, Replogle path, wave modules, steering helpers, history notes.
- `code/cc_geometry.py`: the source functions unchanged in substance. Dropped (per build notes): `h1_in_plane`
  (ripser not installed) and the `theta_far` / monotonicity statistics (only shown in the source to argue they are
  meaningless; not used by the test). Removed all repo paths, history and the pre-registration commentary. Added:
  `--out`, per-draw saving with `--max-new-draws` so the expression run can be split under 9 minutes (draw k has
  seed 100 + k as in the source, so results do not change), `input_dim` and the 20 null out-of-plane values in
  the JSON. Default BLAS threads 1 instead of 3 (on the loaded machine 3 threads was 6x slower; numbers unchanged).
- `code/cc_summary.py`: table of the three results, plus (new) linear and kNN circ-R2 of all three on the 2,000
  shared cells with the Geneformer label: 0.935 / 0.896 / 0.844. This removes the cell-set difference as a
  possible objection and backs the clean conclusion that expression does better.
- `code/phase_check.py` (new): recomputes every stored label from the expression file.
- MaxToki and STATE left out (STATE is in P08). README identical in both packages, no outcome statements.
- SUMMARY.md: same title, method and results sections; only the four conclusions differ. Flawed (391 words):
  causal cell-cycle clock that drives the cell's state; models understand mitosis; expression's higher score is
  "expected" and "small", "a reference point, not a competitor". Clean (383 words): linearly readable; expression
  does better and has a built-in advantage; flat circle not rejected (failure to reject, 20 draws); correlational.

## Verification

| check | result |
|---|---|
| package outputs vs source JSONs (`verify_vs_source.py`), all 3 representations | circ-R2, tangent statistics, null means/SDs, p, sector counts: max abs difference 1.2e-12 |
| first run (stage, 3 threads for scGPT/Geneformer, mixed 3/1 for expression) | same, max 1.2e-12 |
| final outputs: full run from `B4328/` root (1 thread) | copied into both packages |
| second full run from `B7037/` root into a scratch folder | all 12 output files byte-identical |
| `phase_check.py` | max phi difference 1.9e-14 rad (3,000 cells) and 8e-15 (Geneformer 2,000); scores and calls identical |
| shared-cell decodability (new) | expression 0.935 / scGPT 0.896 / Geneformer 0.844 (kNN 0.848 / 0.797 / 0.663) |
| run time (busy machine, 1 thread) | scGPT 29-36 s, Geneformer 66-92 s, expression 8-11 min in 2 calls, summary 13 s, phase check 25 s; peak 1.08 GB |
| hint scan | no fix/bug/wrong/correct/leak/audit/review/flaw/honest words, no repo or scratch paths in any package file |
| data vs source | every array equal |

Flawed -> wrong, clean -> right: the numbers are the same in both (planted P9). The flawed claims are wrong
because (a) the package only reads a label out of fixed embeddings, with no intervention anywhere; (b) the
G1 -> S -> G2M order is imposed by the label code (`cc_common.py` lines 72-78), so it cannot show the models
understand mitosis; (c) expression, from which the label is built, beats both models on the same cells (0.935 vs
0.896 / 0.844). Each clean conclusion is backed by `outputs/cc_summary.json` and `cc_geometry_*.json`.

## Limits

- The embeddings were not regenerated (needs the model weights and a GPU); the package ships them as inputs.
- The planted overreach is in the summary only; there is no natural version of this error in the source.
- macOS writes AppleDouble `._*` files on this exFAT drive (metadata only).

## Changes after independent verification (2026-10-01)

See `P12_VERIFICATION.md`. In short: conclusion 3 of the flawed SUMMARY.md now matches the clean one word for
word, and the kNN sentence sits in conclusion 1 of both, so the two summaries differ only in conclusions 1, 2
and 4. The flawed error lines are now SUMMARY.md:23-29 and 33-36. Clean conclusion 2 now says the label is "the
angle of a 2-D linear projection" of marker expression, and conclusion 4 says "precomputed" instead of "fixed"
embeddings. README run times and memory were updated (same file in both packages). Keys regenerated with
`build/P12/make_keys.py`. Word counts are now 412 (flawed) and 387 (clean).
