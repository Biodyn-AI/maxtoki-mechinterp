# P04 build record

Pair P04: frozen K562 -> RPE1 transfer of a cell-cycle phase readout, C2S-Scale-Gemma-2-2B layer 21
vs expression encoded like the model's input (top-512 genes, rank only).

| id | status | twin |
|---|---|---|
| B7148 | flawed | B7392 |
| B7392 | clean | B7148 |

Ids are 'B' + 4 random digits, checked against existing package folders. Neither id says which twin is which.
The build was started in an earlier session that was cut off. That session's scratch scripts were lost.
This session checked every package file again, re-ran both packages, ran extra checks, and wrote the keys.

## Sources (read only; nothing in biomi_automation was changed)

- Flawed script: `biomi_automation/projects/c2s-scale/route_genemanifold_c2s/encoding_matched_transfer.py`
  (`topk_mag` / `topk_rank` applied to the shared-gene matrix, lines 46-60 and 91-104).
  Its output: `results/encoding_matched_transfer.json`.
- Fix: `selection_matched_arm.py` (`rank_shared_only`, lines 30-40). Write-up: `RESULTS_transfer_and_names.md`
  lines 97-128 (section 1c). The original flawed write-up is the "first pass" described there.
- Helpers: `route_genemanifold_c2s/cc_phase.py`, `route_genemanifold_c2s/cell_sentences.py`,
  `c2s-scale/code/extract_activations.py`.
- Data: `biotensor/data/cellcycle/k562_substrate_for_state.h5ad` (3,000 x 6,546),
  `c2s-scale/route_genemanifold_c2s/data/rpe1_substrate.h5ad` (3,000 x 8,749),
  `route_genemanifold_c2s/data/act_{k562,rpe1}/layer_21_activations.npy`, `row_cell_ids.npy`, `manifest.json`.

## What is in each package (91 MB each)

- `data/`: `{k562,rpe1}_expression.npz` (CSR arrays `data`, `indices`, `indptr`, `shape`) and `{k562,rpe1}_genes.txt`,
  made from the h5ad files. Checked: X and gene names equal the h5ad exactly. `act_*/` copied byte for byte
  (`row_gene_names.json` not copied; it is all "<CELL>" for cell-summary runs). `data/` is byte-identical in both packages.
- `code/transfer.py` (the analysis), `cc_phase.py`, `data_io.py` (npz loader, replaces anndata),
  `cell_sentences.py` and `extract_activations.py` (not run; they show how the model's 512 genes were chosen).
- `outputs/transfer_results.json`, `outputs/run_log.txt`; `README.md` (byte-identical in both); `SUMMARY.md`.

### Changes from the original scripts (both packages)

- `transfer.py` docstring: removed the "why this exists" paragraph (it named an earlier comparison and its
  numbers), the "<- what was compared before" marker, the sentence quoting a 0.015 gap, and wording about wrong
  signs. Inputs are read from `data/` (package-relative) through `data_io.py` instead of h5ad paths.
- `topk_mag` and `topk_rank` were merged into one helper, `top_k_encode(F, cols, k, values)`, used by both
  512-gene arms. It takes the full expression matrix and the shared-gene column indices.
- Encodings are stored as float64 (original float32). Effect on R_diff: at most 1e-6. All printed numbers match
  the original JSON.
- `cc_phase.py`: removed the analogy paragraph, the "why this is needed / the fix" wording, `validate()` and the
  `__main__` block (it held an absolute repository path). Function bodies unchanged.
- `cell_sentences.py`, `extract_activations.py`: removed references to project phases, a test file and SAEs.
  The usage line now shows the settings used for these activations. Code unchanged.
- Added by the verifier (see `P04_VERIFICATION.md`): the words "(all its genes)" were removed from the
  `top_k_encode` docstring (line 48) in both packages. Outputs unchanged.

### The only difference between the two packages

`code/transfer.py` lines 52-57 (flawed) vs 52-59 (clean), inside `top_k_encode`:
- flawed: `X = F[:, cols]`, then the top k are picked within the shared genes (same as the original script);
- clean: the top k are picked from the full row `F[i]`, then only shared columns are kept, with the ranks
  (or values) from the full-panel order (same as `rank_shared_only`).
Both 512-gene arms change (rank and magnitude). `SUMMARY.md` differs only in the numbers and the conclusions
that follow from them, in the same style and length. No comment in either package describes the selection choice.

## Verification

Both packages were run from their root with `bin/python code/transfer.py` (PYTHONDONTWRITEBYTECODE=1).
Both reproduced their stored `outputs/` byte for byte (flawed 75 s, 1.9 GB; clean 35 s, 2.2 GB).
`__pycache__` was removed from both.

| | flawed B7148 | clean B7392 | source project |
|---|---|---|---|
| expr_512_rank R_diff | 0.774 (21 deg) | 0.792 (20 deg) | 0.774 / FULLSEL 0.792 |
| model - expr_512_rank | +0.0153 [+0.0022, +0.0283] | -0.0022 [-0.0144, +0.0105] | same two numbers |
| verdict | MODEL ADDS | PARITY | first pass ADDS, corrected PARITY |
| tokenisation loss | +0.1042 | +0.0868 | 0.104 (kept from the mismatched arms) |

Extra checks (`studyB/build/P04/checks.py`, `checks_alpha.py`; outputs `checks_out.json`, `checks_alpha_out.json`):
- No cell has fewer than 512 nonzero genes (min 991 K562, 1,127 RPE1; 926 within shared genes).
- RPE1 sentences hold 38-133 (mean 67.8) non-shared genes; K562 sentences 0-1.
- Using the exact sentence genes and order (stable ties) gives -0.0011 [-0.0132, +0.0111]: the clean verdict
  does not depend on tie-breaking.
- **Alpha is load-bearing (present in both packages and in the source; never checked there).** Matched contrast:
  alpha 100 -0.0081, alpha 1000 -0.0022, alpha 10000 +0.0175 [+0.0074, +0.0274]. 5-fold CV on K562 picks alpha
  10000 for all arms. So the clean "parity" conclusion is not robust. The shared-gene selection inflates the
  model's lead at every alpha (+0.035, +0.018, +0.007). Recorded in both keys under `other_real_issues`.

## Plain summary

The flawed package gives the documented wrong result (+0.0153, "model adds"). The clean one gives the documented
correct result (-0.0022, "parity"). They differ only in where the top 512 genes are picked. One caveat: the clean
"parity" depends on the fixed ridge alpha of 1000. With the alpha that K562 cross-validation prefers, the model
comes out ahead even with matched selection.
