# P10 build record (ESM-2 650M attention vs disulfide-bonded cysteine pairs; planted P9)

| id | status | twin |
|---|---|---|
| B1249 | flawed | B4174 |
| B4174 | clean | B1249 |

Ids drawn with `secrets` and checked against existing package and key names. Which id got which status was
a coin flip (`secrets.randbelow(2)`). Built 2026-10-01 with `maxtoki-framework-eval/bin/python`, CPU only.
Build scripts: `studyB/build/P10/` (outside every package). The two packages differ only in `SUMMARY.md`
(md5 of every other file checked equal).

## Sources (read only; nothing in biomi_automation was changed)

- Code: `biomi_automation/projects/protein-lm-sae/routeb/disulfide_test.py`, part 2, attention arm:
  `AttentionPairScorer` (line 861; APC/symmetrise at lines 966-977), pair builders
  `build_pair_ground_truth` (354) and `build_candidate_pairs` (437), `part2` (1075), `per_protein_metrics`
  (1008); metric helpers from `routeb/concept_test.py` (`average_precision` 172, `honest_pick` 496,
  `expected_best_of_n` 517, `cluster_bootstrap_ap` 554).
- Data: `data/labels/esm2_attn_cyspairs_L16_bonded.npz` (= `routeb/results/attn_cache.npz`, same md5),
  `data/labels/disulfide_pairs.npz`, corpus `data/corpus/swissprot_20k.jsonl`, UniProt flat file, split
  `routeb/runs_bilinear/layer16_atomic_k32_d5120_main/split_masks.npz` (val = select, test = test).
- Results: `routeb/results/disulfide_test_L16_run_20260831T163513Z.json` (attention L32H13 test AP 0.71794,
  separation 0.11328, base rate 0.04904).
- The source has no write-up of this result to copy. Both SUMMARY.md texts are written by me in the
  analyst's voice; the flawed one carries the planted overreach.

## What was built

- `rebuild_labels.py` (imports the source module with bytecode writing off; the pair file was rebuilt into
  the build folder, never the repo): re-parsed `FT DISULFID` from `uniprot_sprot.dat.gz` with
  `build_pair_ground_truth(force=True)` -> 3,548 intrachain pairs, 150 interchain singles excluded, identical
  to the cached `disulfide_pairs.npz`. `build_candidate_pairs` (bonded scope, <=1,022 residues, <=40
  cysteines) -> select 4,419 pairs / 313 bonded / 113 proteins; test 7,463 / 366 / 123 (2 test proteins
  dropped by the cysteine cap). md5 pair-set signature `d9521e1c...` equals the attention cache's signature.
  Both ends of every pair are cysteines; every annotated bond of a kept protein is in its candidate set.
- Package data: `pairs_{select,test}.npz` (accession, length, pos_i, pos_j, bonded), `attention_scores.npz`
  (S_select 661 x 4,419, S_test 661 x 7,463, bit-identical to the cache), `proteins.tsv` (accession, split,
  length, cysteines, bonds, sequence). 28 MB per package.
- `code/disulfide_attention.py`: the attention arm and the sequence-separation arm of `part2`, rewritten as a
  standalone script with package-relative paths. Kept: AP with tie groups, select->test choice, test maximum
  and its within-protein-permutation best-of-N null, protein bootstrap (seed 2042), per-protein AP / P@1 /
  P@n_bonds, count of candidates >= 2x base rate. Added: mean-of-all-heads row reported on its own, per-layer
  summary, top-10 heads, `head_ap.csv`. Dropped: SAE, bilinear and pairwise-dictionary arms (per build notes),
  matched-N draws (only for dictionary arms), top-L/2 long-range block, diSBPred reference text, all repo
  paths and history docstrings. Variable names `honest`/`leaky` renamed (`chosen_ap`/`test_max`); the word
  "fixed" removed.
- `code/extract_attention.py`: the `_compute` loop of `AttentionPairScorer`, standalone (reads
  `proteins.tsv`), documented as not needed to reproduce. Only change: `dtype=` instead of `torch_dtype=`
  for transformers 5.
- README.md identical in both packages, no outcome statements. SUMMARY.md: same title, data and results
  sections; only the four conclusions differ. Flawed (419 words): dedicated circuit, head computes
  connectivity, other heads are earlier stages, model understands folding chemistry. Clean (439 words): ranks
  far above chance, signal spread over several heads, correlational readout with no intervention and no
  contact-specificity test, next step is ablation.

## Verification

| check | result |
|---|---|
| package run (both ids) vs source JSON (`verify_vs_source.py`) | chosen head, test AP 0.7179437, test max 0.7447274, 192 candidates >= 2x, bootstrap mean/lo/hi, per-protein AP / P@1 / P@n_bonds, separation arm incl. its null: all equal to 1e-12 |
| attention null best-of-N | 0.0869 in package (fresh RNG per arm) vs 0.0929 source; replaying the source's shared RNG (draws for the 7 earlier arms) gives 0.09287168 exactly |
| sklearn AP for L32H13 on test | 0.7179437 (equal) |
| `--selftest` | pass (AP vs sklearn 7e-18) |
| outputs re-run to a scratch folder | all 3 files byte-identical; identical across the two packages |
| `extract_attention.py` on 6 short proteins, CPU bf16 (`spotcheck_extract.py`) | corr 0.9995-0.9997 with shipped scores, L32H13 corr >= 0.997; peak footprint 1.7 GB |
| time / memory of the analysis | ~5 s / 64 MB |
| hint scan | no fix/bug/wrong/leak/audit/review/error/flaw/clean words, no repo paths; "corrected"/"correction" appear only as the name of APC |

Flawed -> wrong, clean -> right: the numbers are the same in both (planted P9). The flawed claims are wrong
because the package contains no intervention and no contact comparison (checked: no ablation, patching or
structure data anywhere), and the data show the signal is spread (5 heads > 0.5 test AP; 191 heads >= 2x base
rate, at least 2 in every layer). The clean conclusions are each backed by `results.json`.

## Limits

- ESM-2 extraction was not re-run end to end (236 proteins would take more than 9 minutes on CPU). Only the
  spot check above was done.
- The "SAE/bilinear code crashed" note in the build notes was not checked; those arms were simply dropped.
- macOS writes AppleDouble `._*` files on this exFAT drive (metadata only).

## Re-check after the app restart (2026-10-01)

The build was already complete when the session restarted. Re-checked: both packages re-run from their
roots into a scratch folder give byte-identical `results.json`, `head_ap.csv` and `run_log.txt` (2 s,
78 MB peak); `--selftest` passes; every file except `SUMMARY.md` has the same md5 in both packages; the
hint scan is still clean; the README count of 18,176 corpus entries matches the protein split file. The
only `._*` files in the packages hold `com.apple.provenance` metadata and nothing else.
