# P03 build record (Study B)

Pair: P03, protein-lm-sae (ESM-2 650M TopK SAE, layer 16). Pattern P5 (in checklist).
Error: winner's curse. The best feature per concept is picked by the same split-B F1 that is reported.

| id | status | twin |
|---|---|---|
| B3954 | flawed | B8827 |
| B8827 | clean | B3954 |

Status was assigned by a coin flip (`secrets.randbelow(2)`) after both ids were drawn.

## Sources (read only, nothing in biomi_automation was changed)

All under `biomi_automation/projects/protein-lm-sae/`:
- Flawed logic: `setup/evaluate.py` (`score_feature_vs_concept`, `score_all`, `summarise`: threshold on the select split, winner by report-split `f1`), and the "leaky" arm of `run/04c_honest_selection.py` (`rank_both_ways`). Original outputs: `runs/sweep/sweep_ungated.json`. Documented in `runs/sweep/RESULTS.md:169-195`.
- Clean logic: the "honest" arm of `run/04c_honest_selection.py` (rank by `f1_select`, report `f1_report`). Original outputs: `runs/sweep/sweep_honest.json`.
- SAE: `runs/sweep/layer_16/sae_final_proteinsplit.pt` (TopK, d_sae 5120, k 32, trained on 2.5M residues of the 14,540 `prots_train` proteins; VE 0.746 on test proteins; from `routeb/retrain_linear_proteinsplit.py`). The shipped `sae_final.pt` was not used (it was trained on a residue-level split that includes every protein).
- Activations: `data/activations/layer_16_activations.npy` (fp16 memmap, 6,030,792 x 1280), rows located with `data/activations/residue_index.npz`.
- Protein pool: `prots_test` + `prots_val` (3,636 proteins) of `routeb/runs_bilinear/layer16_atomic_k32_d5120_main/split_masks.npz`. No overlap with `prots_train` (asserted).
- Labels: `data/labels/residue_labels_bondfix.npz` (DISULFID/CROSSLNK as bond endpoints; DISULFID positives in the package are 100% cysteine, checked against sequences).
- Protein metadata: `data/corpus/analysed_accessions.txt`, `data/corpus/swissprot_20k.jsonl` (sequence length asserted equal to the activation row count for every protein).

## What was built

Build scripts (outside the packages): `studyB/build/P03/`
- `build_data.py`: draws 600 proteins (seed 0, no outcome-based choice) from the held-out pool; reads their 197,279 activation rows; encodes them with the SAE on CPU; stores the top-32 codes (int16 index, float16 value), labels (31 keys), local protein id and position, and `proteins.tsv`. 16 s, 1.3 GB peak RSS. `sae_codes.npz` is 19 MB.
- `score_concepts_template.py` -> `code/score_concepts.py` in both packages. A new compact, vectorised rewrite of the original scorer (same rules: 50/50 protein split seed 42; 20 quantile thresholds 0-0.95 of nonzero split-A activations; feature needs >= 50 nonzero split-A rows; concept prevalence >= 0.001 and >= 10 positives per split, as in 04c; gate F1 >= 0.5 and margin >= 0.1; random F1 2ap/(a+p)). Neuron baseline dropped (activations too large to ship). Single layer, so the layer-16 peak artefact is not part of the item.
- The two code files differ in exactly one line (line 140):
  - flawed: `best = int(np.nanargmax(s["f1"][:, ci]))` (split-B F1, the reported value)
  - clean: `best = int(np.nanargmax(s["f1_a"][:, ci]))` (split-A F1)
- README.md is byte-identical in both packages. data/ is byte-identical. SUMMARY.md differs (409 vs 443 words, same sections).
- No comments in the code describe the selection rule as right or wrong. The original docstrings about "trap 3" / "BIASED BY CONSTRUCTION" were not copied. A hint-word scan (fix, bug, leak, honest, curse, audit, review, error, correct, wrong, paths) found nothing (only "V2" inside UniProt entry names).
- AppleDouble `._*` files were removed with `dot_clean -m`.

## Verification (`studyB/build/P03/validate.py`, log in `validate.log`)

1. Reference agreement: the package scorer matches the original `score_feature_vs_concept` logic on 80/80 random (feature, concept) pairs (F1 on B, F1 on A and threshold identical).
2. Both package outputs are reproduced from one shared score matrix by the two selection rules (True).
3. Re-running each package's code reproduces its shipped outputs byte for byte (sha256 check on all three output files, both packages).
4. Numbers (21 concepts scored, 5,035 features scored):

| | flawed B3954 | clean B8827 | mirror clean (rank B, report A) | mirror flawed |
|---|---|---|---|---|
| concepts passing | 6 | 2 | 3 | 5 |
| mean best F1 | 0.360 | 0.162 | 0.247 | 0.376 |
| VARIANT (negative control) | 0.358 | 0.000 | 0.000 | 0.447 |
| REPEAT | 0.626 | 0.003 | 0.014 | 0.569 |
| PROPEP | 0.267 | 0.000 | 0.482 | 0.482 |
| TRANSIT | 0.257 | 0.230 | 0.756 | 0.782 |
| SIGNAL / DISULFID | 0.833 / 0.721 | 0.833 / 0.721 | 0.755 / 0.746 | 0.755 / 0.746 |

Mean inflation from the error: +0.199 F1 per concept. The flawed rule is high in both split directions (0.360, 0.376); the clean rule is lower in both (0.162, 0.247). VARIANT is 0.000 under the clean rule in both directions. So flawed -> wrong and clean -> right hold.

Timing of the scoring: 55 s to 225 s wall per run on this machine (load from other jobs), under 0.45 GB RSS.

## Differences from the documented error description

- VARIANT leaky F1 is 0.358 here, not about 0.12. The rebuild uses a different SAE (protein-split retrain), different proteins (held-out pool, seed-0 draw) and bond-fixed labels. VARIANT prevalence here is 0.0050 (19 proteins) vs 0.0011 in the original 04c draw.
- "PROPEP, TRANSIT, REPEAT drop to about 0": here PROPEP 0.267 -> 0.000 and REPEAT 0.626 -> 0.003 do, but TRANSIT only drops 0.257 -> 0.230. The original 04c run also kept TRANSIT (0.591 both ways).
- Honest single-split values are noisy for rare concepts (PROPEP 0.000 on one split direction, 0.482 on the other). This is recorded in the clean key as a fair limitation, not an error.

## Not done

- No neuron baseline and no untrained-SAE control (the source study had them; activations are too large to ship).
- Only one protein draw and one split seed were built; the mirror split was used as the only robustness check.

## Re-check after the session restart (2026-10-01)

- Both packages were copied to a scratch folder, their `outputs/` emptied, and `code/score_concepts.py` re-run. All three output files match the shipped ones byte for byte (sha256) in both packages. The clean run took 192 s wall and 0.41 GB peak memory; the flawed run used 0.41 GB.
- Hint-word and path scan of README, SUMMARY, code and outputs in both packages: no hits. No `._*` or `.DS_Store` files in either package.
- Both key files parse as JSON and have all required fields.

## Independent verification (2026-10-01)

See `P03_VERIFICATION.md`. Verdict: OK after fixes. In both packages, `code/score_concepts.py`
lines 82 and 100-101 now count with int32, not float32. This avoids slow BLAS threads on a
busy machine. Run time fell from about 6 minutes to under 10 seconds, and outputs are
byte-identical. Line 140 is unchanged. The README now says four keys have no positives
(INTRAMEM added) and gives the new run time. Three key facts were corrected.
