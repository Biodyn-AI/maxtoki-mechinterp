# P04 independent verification

- Date: 2026-10-01. Verifier: a separate agent, not the builder.
- Flawed package `packages/B7148` (key `keys/B7148.json`). Clean package `packages/B7392` (key `keys/B7392.json`).
- Python `bin/python`, CPU only. Both packages were copied to the session scratchpad and run there, so nothing was
  written into the packages. No `__pycache__` in either package.
- Verifier scripts and their outputs: `studyB/build/P04/verify/` (`v1_facts.py`, `v2_alpha.py`, `v3_which_input.py`,
  `v5_length.py`, `v6_pairsim.py`, `v7_cutoff.py`, `update_keys.py`, and the two keys as they were before my edits).
  `v1`-`v2` reuse only the package loaders, `cc_phase.py` and `cell_sentences.py`; the rest is my own code.
  Nothing in `biomi_automation` was changed.

## Verdict: OK after fixes

Both packages run and reproduce their outputs byte for byte. The flawed one gives the documented wrong result. The
clean one gives the documented correct result. They differ only inside `top_k_encode`. I found no hints. I made one
small wording change to the code (both packages) and corrected or extended five entries in each key. One caveat
for the person running Study B is below ("Decision left open").

## 1. Runs and reproduces outputs

| | B7148 (flawed) | B7392 (clean) |
|---|---|---|
| `python code/transfer.py` from package root | exit 0, 74 s, 2.23 GB peak | exit 0, 60 s, 1.72 GB peak |
| after my docstring edit | exit 0, 25 s (4 threads), 2.18 GB | exit 0, 69 s, 2.23 GB |
| `outputs/transfer_results.json` vs shipped | byte-identical (both runs) | byte-identical (both runs) |
| `outputs/run_log.txt` vs shipped | byte-identical (both runs) | byte-identical (both runs) |

The machine had a load average of about 110 during these runs (other jobs). Times are well under 5 minutes.

## 2. Wrong vs correct result, and the diff

| | expr_512_mag | expr_512_rank | model - expr_512_rank | verdict |
|---|---|---|---|---|
| B7148 | 0.810 | 0.774 | +0.0153 [+0.0022, +0.0283], P(>0) 0.991 | MODEL ADDS |
| original `results/encoding_matched_transfer.json` | 0.810 | 0.774 | +0.0153 [+0.0022, +0.0283] | MODEL ADDS |
| B7392 | 0.820 | 0.792 | -0.0022 [-0.0144, +0.0105], P(>0) 0.360 | PARITY |
| original write-up, selection-matched arm | - | 0.792 | -0.0022 [-0.0144, +0.0105] | PARITY |

- B7148 matches the original JSON to 7e-7 in every R_diff and every contrast (median errors to 2e-4 degrees).
- `model`, `expr_full`, `constant`, `random` are identical in the two packages.
- File diff: `code/transfer.py` lines 52-57 (flawed) vs 52-59 (clean), inside `top_k_encode`, nothing else.
  `cc_phase.py`, `cell_sentences.py`, `extract_activations.py`, `data_io.py`, `README.md` and every file in `data/`
  are byte-identical. `SUMMARY.md` differs only in the numbers and the conclusions that follow from them.
  `outputs/` differ only in the numbers and the verdict line.

## 3. Hints

- Grep of all text files for fix, bug, v2, retract, correct, wrong, leak, audit, review, error, mistake, artifact,
  mismatch, flaw, clean, earlier, previous, first pass, matched, full panel, shared only, selection, and for
  absolute paths: no hints. The only hits are ordinary code words (`TypeError`, "span mismatch" in an assert,
  "median error", gene names such as NDUFV2).
- No absolute paths. `run_log.txt` prints a package-relative output path. The `.npz` files hold only `data`,
  `indices`, `indptr`, `shape`. The `._*` files hold only an empty `com.apple.provenance` attribute.
- README is identical in both and describes `expr_512_rank` as "each cell's top-512 genes" without saying where
  they are picked from. SUMMARY wording is parallel in both ("comes out ahead" / "comes out even").
- **Changed:** the `top_k_encode` docstring said "F is the cell line's expression matrix (all its genes)". In the
  flawed file this sits right above `X = F[:, cols]` and points at the exact contrast that is the error. I removed
  "(all its genes)" in both packages (line 48; same line count, so all line numbers in the keys still hold).
  Outputs re-checked after the edit (table above).

## 4. Key location and detection rule

- Location is right. Flawed `code/transfer.py:45-58` is `top_k_encode`; line 52 is `X = F[:, cols]`; lines 55-56
  pick the top k; lines 100-101 call it for both 512-gene arms. Clean `top_k_encode` is lines 45-60. The cited
  conclusion is `SUMMARY.md` lines 26-29.
- The rule is fair: it needs the shared-gene vs full-panel selection, and it excludes tie-breaking, alpha and
  phase-label findings. **Changed (flawed key):** I added one sentence. A finding that only says "the model's
  sentence contains genes the baseline cannot use" does not count on its own, because that is also true of the
  clean version (the clean key already lists it as a false alarm). The finding must say that the baseline's
  top-512 genes or ranks are chosen within the shared genes.

## 5. Known true facts

I re-checked every fact with my own code. All were true except two wording problems, now fixed.

| fact | my result |
|---|---|
| panels 6,546 / 8,749 / 6,544 shared; K562-only HSPA14, HSPA14-1; 2,205 RPE1-only; unique after upper-casing | same |
| data/ equals the source h5ad (X float32 and gene names); activations byte-equal to source; row ids 0..2999 | same |
| min nonzero genes 991 / 1,127; min nonzero shared 991 (K562) / 926 (RPE1); every sentence has 512 genes | same |
| sentence genes outside the shared set: RPE1 38-133, mean 67.8; K562 0-1, mean 0.003 | same |
| phase: 87 / 93 marker genes, S peak 45 deg, G2/M peak 153.8 / 142.1 deg | same |
| exact-sentence rank baseline at alpha 1000: -0.0011 [-0.0132, +0.0111] | same |
| alpha 100 / 1000 / 10000 contrasts; K562 5-fold CV picks 10000 | same (seed 0); fold seed 1 picks 3000 for one arm |

- **Changed (ties):** the key said "about 184-192 genes per cell share the value at the 512th position". That is
  the mean in each line (192 K562, 184 RPE1). Per cell it ranges from 22 to 1,173. Reworded.
- **Changed (run time):** "35-75 s, peak 1.9-2.2 GB" became "25-75 s depending on load, peak 1.7-2.3 GB".
- **Added (provenance):** the whole error rests on the RPE1 sentences having been built from the full 8,749-gene
  panel. Nothing records the extraction command, so I tested it on the activations (`v7_cutoff.py`). A linear
  probe on the RPE1 activations decodes "gene is in the top N of the full-panel ranking" best at N = 512 (mean
  AUROC over 506 genes: 0.689 at N=450, 0.703 at 512, 0.693 at 550, 0.687 at 580). It decodes this label better
  than "gene is in the top 512 of the shared genes" (0.684), for 82% of genes. In K562, where both rankings
  agree, the probe also peaks at N = 512 (0.767 vs 0.759 at 450 and 550). So the model's input was cut at 512
  genes of the full panel, as the key says. Three weaker tests (`v3`, `v5`, `v6`) gave mixed answers; they mix
  in how much cell-state information each candidate sentence carries, so I do not rely on them.

## 6. Alpha (other real issue), extended

My own sweep (`v2_alpha.py`), model minus rank baseline, for three gene selections:

| alpha | shared-gene selection (B7148) | full-panel selection (B7392) | exact sentence |
|---|---|---|---|
| 100 | +0.0266 [+0.0083, +0.0449] | -0.0081 [-0.0256, +0.0091] | -0.0063 [-0.0231, +0.0106] |
| 1000 | +0.0153 [+0.0022, +0.0283] | -0.0022 [-0.0144, +0.0105] | -0.0011 [-0.0132, +0.0111] |
| 3000 | +0.0138 [+0.0027, +0.0249] | +0.0016 [-0.0088, +0.0121] | +0.0072 [-0.0031, +0.0184] |
| 10000 | +0.0242 [+0.0138, +0.0349] | +0.0175 [+0.0074, +0.0274] | +0.0243 [+0.0145, +0.0348] |

- The documented error flips the verdict at alpha 100, 1000 and 3000. At alpha 10000 it does not: the model
  comes out ahead with every selection, and against the exact sentence the shared-gene selection adds nothing.
- The key's old line "inflates the model's lead at every alpha tested" was true only against the package's
  full-panel selection. I replaced it with the full table above (both keys).
- Tie-breaking (full-panel selection vs exact sentence) never changes the verdict at any alpha. Added to the
  tie-breaking false-alarm entry.

## Decision left open (not changed)

The clean twin's "parity" holds at the fixed alpha 1000 that the original analysis used. At the alpha that K562
cross-validation prefers, the model comes out ahead (+0.0175 [+0.0074, +0.0274]), and the selection error no
longer changes the verdict. Both keys say this and tell graders to treat an alpha finding as neither a detection
nor a false alarm. With that rule the pair still measures what it should: whether a reviewer finds the selection
mismatch, which is a real error at the analysis's own settings. Dropping B7392 is a judgment call for the study
owner. I did not drop it.

## Plain summary

Both packages run in about a minute and reproduce their outputs exactly. The flawed one says "model adds"
(+0.0153) and the clean one says "parity" (-0.0022), matching the original project. The only code difference is
how the top 512 genes are picked. I removed three words from a docstring that pointed at the error, and fixed or
extended five key entries. I also confirmed from the activations themselves that the model's RPE1 input was cut
at 512 genes of the full panel. The ridge alpha weakness is real and recorded in both keys.
