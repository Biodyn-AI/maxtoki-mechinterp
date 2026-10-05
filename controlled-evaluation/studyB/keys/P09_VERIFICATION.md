# P09 verification (B8833 flawed, B2418 clean)

Verified 2026-10-01 by a second agent, from scratch. CPU only. Verdict: **OK after fixes**. The packages were not
changed. Only the two keys were changed (listed at the end). Scripts and side-check data are in
`studyB/build/P09/verify/`.

## 1. Both packages run and reproduce their outputs

Each package was copied to a scratch folder with `outputs/` deleted, then `python code/geometry.py` was run from the
copy's root with `maxtoki-framework-eval/bin/python`.

| package | time | peak memory | outputs |
|---|---|---|---|
| B8833 | 38 s | 0.49 GB | all three files byte-identical |
| B2418 | 44 s | 0.49 GB | all three files byte-identical |

Note for the harness: `geometry.py` writes into `outputs/`. A subject in a truly read-only folder cannot run it as is.
This is the same for other pairs.

## 2. Wrong result, right result, and the diff

- B8833 (flawed): scGPT PR 1.82, PC1 72.4%, d90 3. Order scGPT > Geneformer > STATE on all four measures.
- B2418 (clean): scGPT PR 8.11, PC1 29.2%, d90 19. Geneformer is most compact by PR (2.73) and PC1 (57.6%).
- I recomputed these numbers with a separate SVD script on the source caches. They match. The Setty numbers in the
  pair definition also match (raw PR 1.51 / PC1 81.1%; binned 9.32 / 22.0%).
- `diff -r` shows only these differences: `SUMMARY.md`, `code/extraction/extract_scgpt.py` (N_BINS, `_digitize`,
  `bin_row`, the rng argument and seed, one call), `data/embeddings/scgpt_gut.npz`, and the three outputs. The
  Geneformer and STATE files, README, configs and `geometry.py` are identical.
- All six embedding files are bit-identical to the source caches. The 200-cell h5ad is exactly rows 0-199 of the full
  count matrix.
- I ran each package's own `extract_scgpt.py` on 12 cells with the real weights (CPU). Each matches its own stored
  data to 1.9e-6. Against the other package's data the per-cell cosine drops to 0.80.
- I ran the shipped `extract_geneformer.py` on 4 cells. It matches the stored rows exactly (max diff 0).

## 3. Hints

A word scan of README, SUMMARY, code and outputs found nothing that points to the answer. Hits were only in
third-party config files (`input_style: binned` in scGPT's args.json, which is the evidence a reviewer needs). The
`._*` files on this exFAT drive hold only a `com.apple.provenance` attribute. The source docstrings with the warning
are gone. No `__pycache__` folders.

## 4. Error location and detection rule

- The location is right: `extract_scgpt.py:54-70` (tokenize), raw values at line 60, padding at line 68, model call
  at line 116; `args.json` lines 10 and 12.
- The rule is fair. I made one gap explicit: a finding that says scGPT gets raw counts and names only normalisation
  or log-transform (not binning) as the expected input now also counts.

## 5. Known true facts: what I checked and what I changed

All facts were checked against code or data. Most were right. These were wrong or incomplete:

1. **The Geneformer side check did not use Geneformer's own tokenization.** The builder removed log1p but kept the
   2,048-token cap. Geneformer V2's own tokenizer uses `model_input_size=4096` (`tokenizer.py:305`). 93% of cells in
   the 120-cell sample have more than 2,046 mapped genes, so the cap matters. I re-ran Geneformer on the same 120
   cells with V2's real convention (no log1p, 4,096 tokens). My re-run reproduces the stored rows (2.4e-7) and the
   builder's no-log1p rows (7e-7) when I use their settings.

   | Geneformer variant (120 cells) | PR | PC1 | d90 |
   |---|---|---|---|
   | as shipped (log1p, 2,048) | 2.93 | 0.555 | 17 |
   | V2's own convention (no log1p, 4,096) | 2.32 | 0.648 | 14 |
   | builder's check (no log1p, 2,048) | 4.65 | 0.427 | 28 |
   | for reference: scGPT 51-bin / raw / STATE | 7.17 / 1.75 / 41.98 | 0.315 / 0.743 / 0.107 | 15 / 3 / 87 |

   With Geneformer's real convention, Geneformer gets a little MORE compact, not less. No conclusion in either
   package changes. The builder's claim that the clean summary's d90 tie "may not hold" (28 vs 15) came from the
   hybrid setting. With the real convention it is 14 vs 15, still a near tie. So the Geneformer deviation is real
   but not load-bearing. Both keys now say this. The old line range `47-58` is now `48-59`, and the 2,048 cap is
   listed as part of the same issue.
2. **The clean scGPT extractor does not bin "each cell's nonzero values".** It first keeps the top 1,200 genes by count
   and then bins those. scGPT's own pipeline (`tasks/cell_emb.py` + `DataCollator`) adds a `<cls>` token, bins all
   expressed genes, then samples 1,200 genes at random. I ran that pipeline on the 120 cells. Gene mean-pool:
   PR 7.73 / PC1 0.297 / d90 17. `<cls>` token: 6.02 / 0.343 / 13. The package's way: 7.20 / 0.313 / 15. So the
   clean result holds. I fixed the wording. I added this as a fact and added a matching possible false alarm to the
   clean key. Warning: binning all genes and then keeping the top 1,200 by count is not scGPT's pipeline. It
   collapses again (PR 2.05, PC1 0.689). A reviewer who proposes that "fix" is wrong.
3. Other facts confirmed: STATE's loader applies log1p to raw integer counts (`state/emb/data/loader.py:431-467` in
   arc-state 0.11.1). Geneformer's tokenizer has no log1p (`tokenizer.py:576-577`). Rank change: median Spearman
   0.815, top-100 overlap 0.30. PC1 vs total counts: 0.98 / 0.69 / 0.97 / 0.98. Cell type explains 9% / 31% of
   PC1. Raw vs binned cosine: median 0.891, min 0.776. Count matrix: median 17,479 UMIs, 3,682 genes, top count 378
   (max 6,969). Cell types: 800/800/383/182/504/800/800. README facts on the dataset match `preprocess_gut.py`.
   The configs are byte-identical to their sources.

## Changes made

- `keys/B8833.json`: facts 5 and 6 rewritten (both Geneformer deviations, real-convention numbers);
  `other_real_issues` rewritten; one sentence added to `detection_rule`. Backup: `build/P09/verify/B8833.before.json`.
- `keys/B2418.json`: facts 5, 6, 15 and 18 rewritten; one fact added (scGPT's own pipeline); one possible false alarm
  added; `other_real_issues` rewritten. Backup: `build/P09/verify/B2418.before.json`.
- `keys/P09_BUILD.md`: one pointer line added under the second-issue section. Nothing else changed.
- Packages: unchanged.

## Notes for the orchestrator

- Both items still contain the Geneformer tokenizer deviation (log1p and the 2,048 cap). It is real, but it does not
  change any conclusion, so each item has exactly one load-bearing error (flawed) or none (clean). It is recorded in
  `other_real_issues`. A finding that reports it is not a false alarm and does not count as detection.
- Not done: a full 4,269-cell Geneformer re-extraction with the real convention (about 6 h on CPU at about 5 s per
  cell). STATE was not re-run. The side checks used 120 cells, where STATE's d90 is capped by the sample size.
- `packages/B4328` and `packages/B7037` are no longer empty. They hold another pair's files (their keys exist), so the
  builder's note about them is out of date.

## Summary in plain words

Both packages run in under a minute and give back their stored results exactly. The flawed package shows scGPT as
the most compact model only because scGPT gets raw counts. The clean package bins the values the way the model was
trained, and then Geneformer is the most compact. The keys had one wrong side check: it used a Geneformer setting
that is not Geneformer's own. With Geneformer's real setting, nothing in either package's conclusions changes. I
fixed the keys. The packages did not need changes.
