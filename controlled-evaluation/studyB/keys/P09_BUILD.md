# P09 build record (scGPT vs Geneformer vs STATE cell-embedding geometry, fetal gut)

| id | status | twin |
|---|---|---|
| B8833 | flawed | B2418 |
| B2418 | clean | B8833 |

Built 2026-10-01 with `maxtoki-framework-eval/bin/python`, CPU only. Build scripts are in `studyB/build/P09/`
(outside every package). The build was split over two sessions (the app was quit part way). The second session
re-checked everything below from scratch.

Two empty folders, `packages/B4328` and `packages/B7037`, were created at 08:25, the same minute this build and
the P11/P12 builds started. Nothing in the P09 scripts refers to them. I did not delete them because I cannot
tell whose they are.

## Sources (read only; nothing in biomi_automation was changed)

- Flawed extractor: `projects/biotensor/codebase/route_branchpoint/extract_scgpt.py` (raw counts into the value
  encoder). Its cache for gut: `data/branchpoint/scgpt_gut.npz`.
- Clean extractor: `projects/biotensor/codebase/route_steering/extract_scgpt_binned.py` (the same code except
  for `bin_row`, 51-bin per-cell quantile binning, seed-0 random tie-break). Its cache for gut:
  `data/branchpoint/scgptbin_gut.npz` (log: `route_steering/extract_binned_all.log`, kept 4269/4269).
- Comparators, the same in both packages: `data/branchpoint/geneformer_gut.npz` and `state_gut.npz`, with their
  extractors `route_branchpoint/extract_geneformer.py`, `extract_state.py` and `state_loader.py`.
- Count matrix: `data/pancreas/gut_setty_schema.h5ad` (4,269 cells x 25,770 genes, made by
  `route_branchpoint/preprocess_gut.py`).
- Checkpoint files: scGPT whole-human `args.json` and `vocab.json` (byte-identical copies), Geneformer V2-316M
  `config.json`, STATE SE-600M `config.yaml`.
- Error documentation: `codebase/route_celltoken/RESULTS.md:103-108`, which records Setty PR 1.5 / PC1 81% raw
  vs 9.2 / 22% binned, and the reading "scGPT has a beautifully compact manifold". The flawed SUMMARY carries
  that reading for gut.

## What was built

- `code/geometry.py`: a new standalone script, the same in both packages. It computes the full-spectrum PR,
  PC1 share, d90 and norm CV, plus ranges over 10 random halves (seed 0), and ranks the models. It writes
  `outputs/geometry.json`, `geometry_table.tsv` and `run_log.txt`. In the second session I added a two-thread
  BLAS cap at the top (`os.environ.setdefault`) in both packages. Without it the run took minutes and thrashed
  on the shared, loaded machine. The outputs did not change (byte-identical before and after).
- `code/extraction/extract_scgpt.py`: the source extractors with package-relative paths (`SCGPT_REPO`,
  `SCGPT_MODEL_DIR`, `BP_H5AD`, `BP_OUT`, `SCGPT_DEVICE` env overrides). The source docstrings were replaced
  by the same neutral docstring in both versions. Removed: the input-convention warning, the "WHY" block, the
  "ONLY change" comment, the args.json comment on `N_BINS`, and all repo paths. Setty names became general
  names. After this, the two files differ only in `N_BINS`, `_digitize`, `bin_row`, the `rng` argument and
  its seed.
- `code/extraction/extract_geneformer.py`, `extract_state.py`, `state_loader.py`: the source code with
  package-relative paths. These are the same in both packages.
- `data/embeddings/*.npz`: the source arrays re-saved with the four keys `emb`, `pseudotime`, `clusters`,
  `cell_idx`. They are bit-identical to the source (`copy_embeddings.py`). The flawed package gets the raw
  cache and the clean package gets the binned cache, both under the name `scgpt_gut.npz`.
- `data/raw/gut_epithelium_first200.h5ad`: rows 0-199 of the count matrix in the same layout
  (`make_sample.py`).
- README.md is the same in both packages and states no outcomes. SUMMARY.md has the same structure in both
  (387 / 396 words). The flawed one says scGPT is most compact on all four measures and the order is
  uniform. The clean one says Geneformer is most compact by PR and PC1, scGPT ties on d90, the ranking depends
  on the measure, and STATE is last.
- Files that differ between the twins (sha256): `SUMMARY.md`, `code/extraction/extract_scgpt.py`,
  `data/embeddings/scgpt_gut.npz`, and the three outputs. The other 12 files are identical.

## Verification (all re-run in the second session)

| check | result |
|---|---|
| `verify_packages.py`: file sets, which files differ, embeddings = source, re-run reproduces outputs | passed; all three outputs byte-identical for both packages |
| flawed package scGPT row | PR 1.82 (halves 1.81-1.83), PC1 0.724, d90 3, normCV 0.0031; order scgpt > geneformer > state on all four measures |
| clean package scGPT row | PR 8.11 (halves 7.91-8.26), PC1 0.292, d90 19, normCV 0.0029; order by PR and PC1 geneformer > scgpt > state |
| Geneformer / STATE rows | identical in both: 2.73 / 0.576 / 20 / 0.0186 and 56.48 / 0.113 / 775 / 0.0284 |
| shipped extractors reproduce the stored scGPT data (`verify_extract.py`; 30 cells, CPU, real weights, conda env `bio_mech_interp`) | flawed code vs raw cache max abs diff 1.9e-6; clean code vs binned cache 1.9e-6; cross pairs cosine down to 0.80 |
| shipped Geneformer extractor, 8 cells | max abs diff 0 vs the stored rows |
| geometry run cost | about 46 s and 0.49 GB with two threads on the loaded machine |
| hint scan (fix, bug, correct, wrong, leak, review, error, repo paths, Setty, MaxToki) | nothing in package code or docs; hits only in shipped third-party configs (`input_style: binned`, STATE `experiment:` / `dataset_correction`) |
| what PC1 tracks (`extra_facts.py`) | PC1 vs total counts, abs Spearman: raw scGPT 0.98, binned 0.69, Geneformer 0.97, STATE 0.98. Cell type explains 9% of raw-scGPT PC1 and 31% of binned |

## A second, smaller issue found during the build (present in BOTH packages)

> Updated by verification: the side check below kept the 2,048-token cap, so it is not Geneformer's own
> tokenization. With the real convention (no log1p, 4,096 tokens) nothing changes. See `P09_VERIFICATION.md`.

The Geneformer extractor ranks genes by `log1p(CP10k) / median`. Geneformer's own tokenizer ranks
`CP10k / median` with no log1p (checked in the cached HF `geneformer/tokenizer.py:576-577`). This came from
the source project. It changes the token order: the median Spearman between the two orders is 0.82, and only
30% of the first 100 genes are shared (`geneformer_rank_check.py`, 200 cells).

I re-ran Geneformer on CPU without log1p for 120 random cells (`gf_nolog_check.py`,
`geneformer_nolog_geometry.py`). On those cells Geneformer moves from PR 2.93 / PC1 0.555 / d90 17 to
PR 4.65 / PC1 0.427 / d90 28. Binned scGPT on the same cells is PR 7.17 / PC1 0.315 / d90 15.

What this means:
- The documented scGPT error is unaffected. The collapse comes from scGPT's input alone.
- The flawed conclusion stays wrong.
- The clean SUMMARY's conclusions 1, 3 and 4 still hold.
- Its d90 tie (conclusion 2) may not hold.

A full 4,269-cell Geneformer re-run would take about 15 h on CPU, so it was not done. Both keys list this
under `other_real_issues` and in `known_true_facts`. A reviewer who reports it is right, so it is not a
false alarm, but it does not count as finding the documented error. So neither package has exactly one
error. The orchestrator may want to keep this pair, mark it, or drop it.
