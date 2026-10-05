# V2b report: SAE feature annotation rates, with the right BH family and a random-feature null (item N01)

Date: 2026-10-01. Model: MaxToki-217M SAE atlas, 12 layers. CPU only. No model forward pass.
Everything here is rebuilt from saved files: the top-20 gene lists in `outputs/phase2/layer_XX/feature_catalog.json`,
the gene universe in `outputs/phase0/layer_XX/gene_names.json`, the same GO BP / KEGG / Reactome / TRRUST files,
and the Replogle K562 var symbols. Script: `scripts/v2b_annotation_fdr.py`. Outputs: `outputs/v2b_annotation_fdr/`.

"Annotated" means a feature has at least one ontology term that passes BH at q < 0.05.
"Rate" means annotated features / alive features. All rates are a fraction of alive features.

## 0. Results first

1. **The deployed annotation rates (61.7% to 80.7%) are at chance level.** Random 20-gene lists with the same
   gene popularity as the real lists get 62.6% to 82.9% under the same deployed rule. The real rate is *below* all
   20 random catalogs at 10 of 12 layers. It sits inside the random range at L1, and below 19 of 20 at L11.
   Even lists drawn uniformly at random from the 6,325-gene universe get 65.4% to 68.4%.
2. **The layer shape is also chance.** Across the 12 layers, the deployed real rates and the random-list rates
   correlate at Pearson r = 0.95 (frequency-matched null) and 0.95 (exact-margin null). So the "U-shape" comes
   from which genes are popular in each layer's top-20 lists, not from the way features group genes together.
3. **With the BH family the spec asks for** (BH per feature over all 2,377 terms; `01-sae-atlas.md`, Phase 2 step 3),
   55 to 104 features per layer pass. That is **4.2% at L0, 1.6% to 1.9% at L1 to L9, and 1.1% at L10 and L11.**
   Random lists with the same gene popularity give 31 to 52 features per layer (0.6% to 1.7%).
   So the real lists beat the random ones at every layer (z = 3.3 to 11.6 in null-SD units; the real count is above
   all 20 random catalogs at every layer). But **about 40% to 59% of the annotated features are expected by chance**
   (random-list mean / real count). The per-feature family does not control errors across 2,500 to 4,900 features.
4. **With BH across all feature x term tests in a layer** (5.9 to 11.7 million tests), **55 features pass in total
   across the 12 layers**: 29 at L0, 5 at L1, 8 at L2, 4 at L3, 3 at L4, 5 at L5, 1 at L9, and none at L6, L7, L8,
   L10, L11. Random lists give 0.15 features in total across the 12 layers (at most 1 in any one catalog).
   These 55 are the labels that survive a layer-wide error control (not checked against co-expression; see
   section 6). 42 of them fall in four programs: mitochondrial translation or
   respiration (14; MRPL/MRPS genes), mitosis and cell cycle (13; PLK1, CCNB1, BUB1, E2F1 targets), DNA replication
   and repair (8; MCM genes), and sterol and cholesterol synthesis (7; HMGCR, DHCR7, FDFT1, SREBF1 targets).
   The other 13 mostly pass on a single broad term (for example RNA metabolism, protein transport,
   ER-to-Golgi transport). The four programs are well-known K562 co-expression programs. I did not test whether a
   co-expression baseline finds the same groups.
5. **The deployed numbers reproduce exactly.** Rebuilding every test and applying the deployed rule gives the same
   annotated-feature counts and the same significant rows on all 12 layers (0 rows missing, 0 extra; p and q agree
   to 1e-16).
6. **Phase 6 picks.** The 50 "richly annotated" layer-5 features used for causal patching are exactly the top 50 of
   the deployed file (45 to 152 deployed hits each). Under the spec's BH, 23 of the 50 are annotated and 19 have
   3 or more terms. Under global BH, 1 of the 50 is annotated. At L5 only 43 features have 3 or more terms under
   the spec's BH (random lists: 13 on average).

## 1. What was wrong in the deployed code

`scripts/full_12layer_pipeline.py:241-251` (this script wrote the current phase2 files; `phases2_to_8.py:176-197`
has the same logic) does this for each feature x term pair:

```python
x = len(top20 & in_vocab)
if x < 2: continue
p = float(hypergeom.sf(x - 1, N_total, K, n))
if p < 0.1:
    annotations.append(...)
...
_, ann_df["q_bh"], _, _ = _mult(ann_df["p_raw"], method="fdr_bh")   # one BH over the kept rows only
```

So BH sees only 26,361 to 39,725 kept tests per layer, out of 5.9 to 11.7 million. Every test it sees already has
p < 0.1, and most have an overlap of 2 genes with a small term. The largest raw p among the deployed
"significant" rows is 0.039 to 0.042. So the step is close to no correction. 79% to 85% of the deployed
significant rows rest on 2 genes out of 20.

Most of the inflation comes from leaving out tests with overlap 0 or 1, not from the p < 0.1 filter. With the
overlap >= 2 filter only (no p filter), the rates are 59.8% to 80.3%, 0.4 to 2 points below the deployed ones.
This "overlap >= 2, one pooled BH" rule is also what the reference code does
(`repos/bio-sae/src/04_annotate_features.py:140-141` returns no test when overlap < 2; pooled BH at :314-316).
So the Geneformer 52.4% used as a comparison has the same problem.

## 2. What I computed

All rules use alpha = 0.05 and the deployed hypergeometric test (`hypergeom.sf(x-1, N, K, n)`), with the deployed
N (genes seen at token positions; 6,325 at L5) and the deployed K (term genes in the Replogle var list).

| Rule | Family for BH |
|---|---|
| A, deployed | overlap >= 2 and p < 0.1, one BH over the kept rows (reproduction) |
| A2, bio-sae rule | overlap >= 2, one BH over the kept rows |
| B, spec | BH within each feature over all 2,377 terms. Tests with overlap 0 have p = 1 and still count |
| C, global | BH over all feature x term tests of the layer |
| Bu, Cu | B and C with K counted inside the 6,325-gene universe instead of the 6,546-gene var list |

Bu and Cu give the same annotated-feature counts as B and C on every layer. The K mismatch in the deployed code
does not matter.

I checked my vectorised per-feature BH against `statsmodels` `fdr_bh` on 153 to 197 features per layer:
0 mismatches.

### The random-feature nulls

Each null catalog has the same number of lists as the layer has alive features, with the same list sizes
(n is below 20 for a few features). Then I run the same rules A, B and C on it. 20 catalogs per kind per layer.
Seed for rep r of kind k at layer L: `20261001 + 1000*L + offset[k] + r` (offsets: freq 100, curveball 200, uniform 300).

- **freq** (the requested null): each list draws n distinct genes without replacement, with probability
  proportional to how often the gene appears across the layer's real top-20 lists. Sampling without replacement
  flattens the most popular genes a little: the per-gene counts correlate with the real ones at r = 0.80 to 0.94.
- **curveball**: the real lists are shuffled by curveball trades between pairs of lists (30 rounds). This keeps every
  list size and every gene's total count exactly (r = 1.000). After shuffling, 0.4% to 0.5% of gene slots are still
  in their original list, which is the chance level. So the gene groupings are fully broken.
- **uniform**: n distinct genes drawn uniformly from the universe. Context only.

The freq and curveball nulls agree closely everywhere, so the result does not hinge on how popularity is matched.

## 3. Per-layer results

Intervals: "[2.5, 97.5%]" is the 2.5th to 97.5th percentile over the 20 null catalogs (unit: rate or feature count).
"Wilson 95%" is a Wilson score interval for the real rate, treating features as the unit. It covers only the
sampling of features within this one trained SAE, not SAE retraining. "z" is (real - null mean) / null SD over
20 catalogs. With 20 catalogs, the smallest possible empirical p is 1/21 = 0.048.

### 3.1 Deployed rule vs random lists

| Layer | Alive | Deployed rule: real | Deployed rule: freq null, mean [2.5, 97.5%] | Deployed rule: curveball null, mean [2.5, 97.5%] | Overlap >= 2 only (bio-sae rule), real |
|---|---|---|---|---|---|
| L0 | 2498 | 80.7% (2016) | 82.9% [81.8, 84.1] | 82.7% [81.9, 83.7] | 80.3% |
| L1 | 4928 | 74.8% (3685) | 74.7% [73.8, 75.6] | 74.5% [73.7, 75.1] | 73.3% |
| L2 | 4928 | 71.1% (3503) | 73.3% [72.3, 74.5] | 73.1% [72.3, 74.3] | 69.8% |
| L3 | 4928 | 68.3% (3364) | 70.8% [70.1, 72.2] | 70.9% [70.0, 71.8] | 67.2% |
| L4 | 4928 | 66.1% (3255) | 69.1% [67.7, 70.2] | 68.8% [67.7, 69.8] | 64.2% |
| L5 | 4928 | 67.8% (3340) | 70.6% [69.2, 71.7] | 70.9% [69.6, 71.8] | 65.9% |
| L6 | 4928 | 68.4% (3371) | 72.8% [71.6, 74.3] | 72.8% [72.1, 73.8] | 67.5% |
| L7 | 4928 | 69.6% (3430) | 75.3% [74.3, 76.0] | 75.2% [74.3, 75.9] | 68.7% |
| L8 | 4928 | 71.5% (3523) | 75.9% [74.8, 76.8] | 75.9% [74.9, 76.9] | 70.5% |
| L9 | 4928 | 69.8% (3442) | 73.6% [72.7, 74.6] | 73.5% [72.8, 74.1] | 68.8% |
| L10 | 4928 | 64.1% (3159) | 66.5% [65.3, 68.0] | 66.3% [65.5, 67.3] | 62.3% |
| L11 | 4924 | 61.7% (3040) | 62.6% [61.6, 63.9] | 62.5% [61.7, 63.2] | 59.8% |

I did not run the nulls under the A2 (bio-sae) rule. On real data A2 is only 0.4 to 2 points below A, so the
same conclusion is very likely, but it is not measured here. (The verification notes at the end measure it: the
bio-sae rule is also at chance level.)

### 3.2 Correct BH families vs random lists

| Layer | Per-feature BH (spec): real features (rate) | Wilson 95% (rate) | Freq null: features, mean [2.5, 97.5%] | Curveball null: features, mean [2.5, 97.5%] | Share expected by chance (freq / curveball) | z (freq / curveball) | Global BH: real features | Global BH: null features, mean (max), freq / curveball |
|---|---|---|---|---|---|---|---|---|
| L0 | 104 (4.16%) | 3.45-5.02% | 41.5 [30, 50] | 40.8 [31, 50] | 40% / 39% | 11.6 / 10.8 | 29 | 0.00 (0) / 0.05 (1) |
| L1 | 86 (1.75%) | 1.42-2.15% | 44.5 [36, 57] | 41.1 [26, 49] | 52% / 48% | 7.2 / 6.9 | 5 | 0.00 (0) / 0.00 (0) |
| L2 | 85 (1.72%) | 1.40-2.13% | 39.4 [32, 53] | 42.1 [26, 58] | 46% / 50% | 8.2 / 4.7 | 8 | 0.00 (0) / 0.00 (0) |
| L3 | 83 (1.68%) | 1.36-2.08% | 38.0 [27, 47] | 34.4 [27, 42] | 46% / 41% | 7.4 / 10.9 | 4 | 0.00 (0) / 0.00 (0) |
| L4 | 81 (1.64%) | 1.32-2.04% | 34.1 [25, 44] | 36.0 [24, 48] | 42% / 44% | 8.4 / 6.7 | 3 | 0.00 (0) / 0.00 (0) |
| L5 | 84 (1.70%) | 1.38-2.11% | 38.4 [26, 50] | 36.5 [27, 47] | 46% / 43% | 6.6 / 8.1 | 5 | 0.00 (0) / 0.00 (0) |
| L6 | 86 (1.75%) | 1.42-2.15% | 41.0 [30, 54] | 44.4 [38, 56] | 48% / 52% | 7.2 / 7.4 | 0 | 0.00 (0) / 0.00 (0) |
| L7 | 91 (1.85%) | 1.51-2.26% | 47.5 [35, 58] | 48.6 [39, 62] | 52% / 53% | 6.6 / 6.1 | 0 | 0.00 (0) / 0.00 (0) |
| L8 | 88 (1.79%) | 1.45-2.19% | 51.6 [40, 66] | 51.8 [35, 60] | 59% / 59% | 5.1 / 5.0 | 0 | 0.05 (1) / 0.00 (0) |
| L9 | 80 (1.62%) | 1.31-2.02% | 45.3 [36, 58] | 47.5 [33, 64] | 57% / 59% | 5.4 / 3.7 | 1 | 0.05 (1) / 0.00 (0) |
| L10 | 55 (1.12%) | 0.86-1.45% | 32.6 [21, 44] | 31.4 [24, 42] | 59% / 57% | 3.4 / 4.6 | 0 | 0.05 (1) / 0.05 (1) |
| L11 | 55 (1.12%) | 0.86-1.45% | 31.2 [21, 42] | 32.5 [24, 44] | 57% / 59% | 3.9 / 3.3 | 0 | 0.00 (0) / 0.00 (0) |

Uniform random lists give 0.62% to 0.69% under per-feature BH and 0 to 0.05 features under global BH.

Notes on rule B:
- Under per-feature BH, 39% to 59% of the surviving feature x term rows still have an overlap of only 2 genes.
  A 2-gene overlap cannot pass at rank 1 for a full 20-gene list (smallest p = 9.4e-5, rank-1 threshold
  0.05/2,377 = 2.1e-5). But it can pass when the list is short (n = 7 to 18) or when the feature has many
  moderately small p-values, because BH is step-up. The step-up route is the main one: most surviving 2-gene rows
  come from full 20-gene lists (see the verification notes). One L0 feature (2591) passes 21 overlapping terms
  (19 KEGG, 2 Reactome)
  (insulin secretion, salivary secretion, gastric acid secretion, ...) that share the same few signalling genes.
  This is the term-redundancy problem; per-feature BH does not fix it.
- The per-feature-BH layer profile also follows the null (Pearson r = 0.95 freq, 0.93 curveball, across 12 layers).
  The one clear difference is L0 (post-embedding layer, 2,430 of 4,928 features dead), where both real and null
  rates are highest. This r comes mostly from L0: without L0 it is 0.74 (freq null) and 0.64 (PPS null of the
  second check). For the deployed rule the r stays high without L0 (0.92 freq, 0.91 PPS).

## 4. Downstream use of the labels (checked here without forward passes)

- **Phase 6** (`scripts/phase6_patching.py:56-66`, used for main.tex "median specificity 1.01x"): the 50 picks
  equal the top 50 of the current deployed L5 file (50 of 50). Under per-feature BH, 23 of 50 are annotated and
  19 have 3 or more terms. Under global BH, 1 of 50 (feature 2560, mitotic nuclear division). So "richly annotated"
  does not describe this sample. The picks are better described as "the 50 features with the most uncorrected
  enrichment hits". The specificity result itself is about patching, and I did not re-run it.
- Corrected per-feature labels for re-selection are in `outputs/v2b_annotation_fdr/real/layer_XX_feature_flags.csv`
  (columns: deployed count, per-feature-BH count, global-BH count, smallest raw p, best term).

## 5. What the paper can and cannot say

Can say (all numbers from this run):
- "Under BH over all feature x term tests in a layer, 55 of the 56,702 alive features across 12 layers have
  a significant ontology label (29 at layer 0; none at layers 6-8, 10 and 11). Random 20-gene lists with the same
  gene frequencies give 0.15 such features in total. Most labelled features (42 of 55) are mitochondrial
  translation, mitosis and cell cycle, DNA replication, and sterol synthesis programs."
- "Under BH within each feature (the protocol of the reference pipeline), 1.1% to 4.2% of features per layer are
  annotated, against 0.6% to 1.7% for frequency-matched random lists; about half of the annotated features are
  expected by chance."

Cannot say:
- Any annotation rate in the 60-80% range, or "excellent biological annotation".
- A U-shaped or any other layer profile of interpretability.
- "More interpretable than Geneformer (52.4%)". That number uses the same uncorrected family, and I could not
  recompute it (no Geneformer catalogs here).
- That the Phase 6 features or the Stage-3 triplets were "richly annotated" or "pathway-sharing", unless they
  are re-picked with corrected labels.

Note: per the verification notes, the current `main.tex` does not quote the annotation rates. They appear in
`summaries/sae-atlas-217M-FINAL_SUMMARY.md` and `FINAL_SUMMARY_12layer.md`. I did not edit those files.

## 6. What I did not do

- **No untrained or norm-matched SAE baseline.** It needs SAE encodings of the layer activations. The raw
  activations were deleted after training, so this needs a model forward pass (not allowed here).
- **No null under rule A2** (the bio-sae rule) in the first run. Real A2 rates are reported. The verification
  notes below add the A2 null.
- **No Geneformer recompute** (no Geneformer feature catalogs in this project).
- **STRING not used.** The deployed code loads 4 databases (GO BP, KEGG, Reactome, TRRUST TF sets); the spec lists
  STRING too. I kept the deployed 4 so the numbers are comparable.
- **No co-expression check** of the 55 global-BH features.
- **No re-run of downstream analyses** (Phase 6 patching, Phase 10 annotated/unannotated split, Stage-2 sources,
  Stage-3 triplets). They need forward passes. The triplet selection lives in other pipeline folders and was not
  checked.
- Null intervals rest on 20 catalogs per kind per layer. That is enough for the conclusions above (the gaps are
  many null SDs wide), but the empirical p cannot go below 0.048.

## 7. Files and commands

- Script: `runs/sae-atlas-217M/scripts/v2b_annotation_fdr.py`
- Config with input sha256, seeds, interval methods, command order: `outputs/v2b_annotation_fdr/run_config.json`
  (the 30 GB Replogle h5ad was not hashed whole; the sha256 of its extracted var symbol list is recorded).
- Terms used (2,377, same count as `outputs/full_12layer.log`): `outputs/v2b_annotation_fdr/terms.json`
- Per layer, real data: `outputs/v2b_annotation_fdr/real/layer_XX.json` (all rules + reproduction check + BH self-test),
  `layer_XX_feature_flags.csv`, `layer_XX_perfeature_bh_enrichments.csv` (rows passing per-feature BH, with
  per-feature q and a global-BH flag)
- Nulls: `outputs/v2b_annotation_fdr/null/layer_XX_{freq,curveball,uniform}.json` (20 catalogs each)
- Summary: `outputs/v2b_annotation_fdr/summary.csv`, `summary.json`, `summary_tables.md`, `summary_extra.json`
  (layer-profile correlations, global-BH totals, Phase 6 check)

Commands (Python: `projects/maxtoki/.venv/bin/python`, CPU, peak memory about 1.4 GB, each under 9 minutes):
`prep`; `real --layers 0-11`; `null --reps 20` in layer chunks (0-2, 3-4, 5, 6-7, 8, 9, 10, 11); `summary`;
`prep` again to record the final script hash.

## Plain-words summary

The old code checked only the tests that already looked good, then corrected for those few. That made 62-81% of
features look "annotated". Random gene lists get the same 62-83% with the same code, so those numbers mean
nothing, and neither does the layer shape. With the correction the spec asks for, 1-4% of features per layer are
annotated, about twice the random-list level, so about half of those are still chance. With a correction across
the whole layer, 55 features in all 12 layers keep a label, against almost none for random lists. Those 55 are
real but few, and most of them are well-known K562 gene programs.

## Verification notes

Date: 2026-10-01. Independent re-check of this item. CPU only, no model forward pass, peak memory about 1.1 GB,
every command under 1 minute per layer chunk. Verdict: **OK after small fixes.** All key numbers reproduce.
The conclusions follow from the numbers.

**How it was checked.** New code, written without reusing `v2b_annotation_fdr.py`:
`scripts/v2b_annotation_fdr_verify.py`, outputs in `outputs/v2b_annotation_fdr_verify/` (with `run_config.json`
holding input sha256, seeds and command order). Differences from the first run:
- term sets rebuilt from the raw GO BP / KEGG / Reactome / TRRUST files and the Replogle var list (2,377 terms;
  same names and same gene sets as `terms.json`);
- hypergeometric tail p-values from my own log-gamma sum (max relative difference from `scipy` 2.7e-11);
- my own BH code (q-values, deployed convention q < 0.05; the "q <= 0.05" form gives the same counts);
- two new random-list nulls with other samplers and other seeds, 20 catalogs each per layer.
  **es**: frequency-weighted lists by Efraimidis-Spirakis keys (each next gene drawn with probability
  proportional to its count among genes not yet drawn). Per-gene counts correlate with the real ones at
  r = 0.80 to 0.94. **slot**: exact margins. All gene slots of the real lists are pooled, shuffled, cut into
  lists of the real sizes, and within-list duplicates are removed by swaps. List sizes and every gene's count
  are kept exactly (max count difference 0);
- the nulls were also scored under the bio-sae rule (overlap >= 2, one pooled BH), which the first run did not do.

**What reproduced exactly.** On all 12 layers:
- the deployed rule gives the deployed files (annotated features, 20,759 to 32,339 significant rows, 0 rows
  missing, 0 extra);
- bio-sae rule real rates (59.8% to 80.3%), per-feature BH counts (104, 86, 85, 83, 81, 84, 86, 91, 88, 80, 55, 55),
  features with 3 or more terms, and global BH counts (29, 5, 8, 4, 3, 5, 0, 0, 0, 1, 0, 0; 55 in total);
- counting term size inside the universe changes no annotated-feature count (per-feature or global);
- Wilson 95% intervals (for example L5 1.38% to 2.11%, features as units);
- Phase 6: the 50 picks are the top 50 of the deployed L5 file; 23 of 50 annotated under per-feature BH,
  19 with 3 or more terms; 1 of 50 under global BH (feature 2560).

**Per-layer numbers from the independent nulls.** "es / slot" = the two nulls. Null values are means over
20 catalogs; [min-max] over the 20 catalogs, unit = annotated features. z = (real count - null mean) / null SD
(ddof = 1) over 20 catalogs, unit = null SD. "Chance share" = null mean / real count.

| Layer | Deployed rule, real | Deployed rule, null es / slot | Bio-sae rule, real | Bio-sae rule, null es / slot | Per-feature BH, real features | Per-feature BH, null es / slot | z es / slot | Chance share es / slot | Global BH, real | Global BH, null es / slot |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 80.7% | 82.5% / 82.7% | 80.3% | 81.6% / 81.8% | 104 | 41.5 [33-51] / 42.4 [31-65] | 11.0 / 7.7 | 40% / 41% | 29 | 0.10 / 0.05 |
| L1 | 74.8% | 74.4% / 74.4% | 73.3% | 72.3% / 72.4% | 86 | 41.9 [34-52] / 43.8 [34-54] | 8.6 / 8.7 | 49% / 51% | 5 | 0.05 / 0.00 |
| L2 | 71.1% | 73.0% / 73.0% | 69.8% | 70.8% / 70.8% | 85 | 37.5 [26-51] / 40.5 [30-59] | 7.7 / 6.0 | 44% / 48% | 8 | 0.00 / 0.00 |
| L3 | 68.3% | 71.0% / 71.0% | 67.2% | 68.7% / 68.7% | 83 | 35.2 [23-43] / 37.0 [24-47] | 8.6 / 8.7 | 42% / 45% | 4 | 0.00 / 0.00 |
| L4 | 66.1% | 69.2% / 69.1% | 64.2% | 66.8% / 66.6% | 81 | 32.4 [22-40] / 33.6 [29-42] | 11.1 / 14.6 | 40% / 42% | 3 | 0.00 / 0.00 |
| L5 | 67.8% | 70.7% / 70.8% | 65.9% | 68.5% / 68.3% | 84 | 38.5 [26-54] / 38.4 [31-51] | 6.9 / 8.6 | 46% / 46% | 5 | 0.00 / 0.00 |
| L6 | 68.4% | 72.9% / 72.6% | 67.5% | 70.9% / 70.5% | 86 | 45.8 [35-63] / 41.4 [29-52] | 5.3 / 7.4 | 53% / 48% | 0 | 0.00 / 0.05 |
| L7 | 69.6% | 75.1% / 75.3% | 68.7% | 73.1% / 73.3% | 91 | 45.8 [33-61] / 49.5 [34-65] | 5.7 / 4.4 | 50% / 54% | 0 | 0.00 / 0.00 |
| L8 | 71.5% | 76.1% / 76.0% | 70.5% | 74.2% / 74.1% | 88 | 49.8 [41-60] / 51.0 [35-64] | 7.8 / 5.5 | 57% / 58% | 0 | 0.00 / 0.00 |
| L9 | 69.8% | 73.1% / 73.4% | 68.8% | 71.0% / 71.3% | 80 | 45.3 [30-59] / 45.2 [37-56] | 4.5 / 6.0 | 57% / 57% | 1 | 0.00 / 0.05 |
| L10 | 64.1% | 66.7% / 66.6% | 62.3% | 64.2% / 64.0% | 55 | 32.2 [23-43] / 34.6 [23-50] | 4.4 / 3.1 | 59% / 63% | 0 | 0.00 / 0.00 |
| L11 | 61.7% | 62.2% / 62.4% | 59.8% | 59.3% / 59.7% | 55 | 30.6 [22-37] / 31.6 [22-44] | 5.9 / 3.9 | 56% / 57% | 0 | 0.00 / 0.00 |

What this confirms:
- Deployed rule: the real rate is below all 20 catalogs of both nulls at 10 of 12 layers (L0, L2 to L10).
  At L1 and L11 it sits inside the null range. Layer-profile Pearson r = 0.94 (es) and 0.95 (slot).
- Per-feature BH: the real count is above every null catalog at every layer. z = 3.1 to 14.6 null SDs.
  Null means 31 to 51 features per layer (0.6% to 1.7%).
- Global BH: the null gives 0.15 features in total across 12 layers for both nulls (one catalog had 2 features).
- Uniform lists (3 catalogs each at L0, L5, L11, `extra_checks.json`): deployed rule 63.4% to 68.2%,
  per-feature BH 0.4% to 0.8%. This agrees with the first run's uniform null.

**New result: the bio-sae rule is also at chance.** Under "overlap >= 2, one pooled BH" (the rule of
`repos/bio-sae/src/04_annotate_features.py`, which skips tests with overlap < 2 at :140-141 and :254, then runs
BH on the rest at :315-316), real minus null mean is -4.4 to -0.9 points at 10 layers, +1.0 point at L1 and
+0.4 / +0.1 points at L11 (es / slot). So the Geneformer 52.4% made with that rule is not a usable reference
either. I could not compute the Geneformer null (no Geneformer catalogs here).

**Small errors found and fixed in this report** (numbers above unchanged):
1. Section 0 item 4 and section 5 said all 55 global-BH features are mitochondrial translation, sterol
   synthesis, DNA replication or mitosis programs. It is 42 of 55: mitochondrial translation or respiration 14,
   mitosis and cell cycle 13, DNA replication and repair 8, sterol and cholesterol synthesis 7. The other 13 mostly
   pass on one broad term (RNA metabolism, protein transport, ER-to-Golgi transport, and others). List:
   `outputs/v2b_annotation_fdr_verify/real/global_bh_features.csv`.
2. Section 3.2: feature 2591 at L0 passes 21 terms, but they are 19 KEGG + 2 Reactome, not 21 KEGG.
3. Section 3.2: short lists whose 2-gene rows survive have n = 7 to 18, not 7 to 11. And the main route is BH
   step-up, not short lists: per layer, 153 to 438 surviving 2-gene rows come from full 20-gene lists, and only
   0 to 81 from shorter lists.
4. Section 0 item 2: "not from biology" was stronger than the test supports. The null keeps each layer's gene
   popularity, which is itself a property of the model. The test shows the layer shape does not come from how
   features group genes. Reworded.
5. "About 40% to 59% expected by chance": with the independent nulls it is 40% to 63%. "About half" still holds.

**Notes that do not change the conclusions.**
- Phase 6 pick boundary: the 50th and 51st features both have 45 deployed hits. So the top-50 set depends on the
  tie-break (catalog order, which is what `phase6_patching.py` uses through a stable sort).
- `outputs/phase6/causal_patching.csv` (2026-04-18) is older than the current `phase2/layer_05` files
  (2026-04-19). The picks still equal the top 50 of the current file, so the check stands.
- The per-feature BH family counts all 2,377 terms, including tests with overlap 0 (p = 1). That is the right
  family: dropping tests because of their own overlap is the same outcome-based selection that broke the
  deployed rule.

**Still not done** (same as section 6): no untrained-SAE baseline (needs a forward pass), no Geneformer null,
no co-expression check of the 55 features, no downstream re-runs. The empirical p from 20 catalogs cannot go
below 0.048. z values carry noise from using only 20 catalogs.

**Plain-words summary of the check.** I rebuilt every test with my own code and two new kinds of random gene
lists. Every number in this report came out the same or very close. Random lists get the same 62-83% "annotation
rate" as the real features under the old rule, and also under the bio-sae rule used for the Geneformer number.
With the right correction, real features beat random lists about two to one (1-4% of features), and 55 features
pass a layer-wide correction against almost none for random lists. I fixed five small wording and detail errors.
Most of the 55 are four well-known K562 programs, but 13 are not.

## Verification notes (second independent check)

Date: 2026-10-01. A second, separate re-check of this item. CPU only, no model forward pass, peak memory
under 1 GB, every command under 7 minutes. New code that does not reuse either earlier script:
`scripts/v2b_annotation_fdr_verify2.py`, outputs in `outputs/v2b_annotation_fdr_verify2/` (`run_config.json` has
input sha256, seeds, null rep counts and the command order). Verdict: **OK after small fixes.** Every key number
reproduces. The conclusions follow, with the caveats below.

**How it differs from the first run and the first check.**
- Overlaps from one sparse product (feature x gene) @ (gene x term). P-values from a lookup table of the deployed
  test, `hypergeom.sf(x-1, N, K, n)`. BH by counting ranks (largest j with p_(j) * m / j < 0.05).
- A new frequency-matched null ("PPS"): for each list of size n, each gene g is drawn with probability exactly
  n * c_g / S (c_g = the gene's count over the real lists, S = all slots), by systematic sampling in a fresh random
  gene order. So every gene's expected count equals its real count (checked at L5: slope 1.0007, r = 0.986 over
  10 catalogs). 20 catalogs at L5, 10 at every other layer.
- Two context nulls at L0, L5, L11: a random relabelling of the gene universe applied to the real lists (keeps
  which genes go together and the shape of the popularity curve, but not which genes are popular; 5 catalogs),
  and uniform lists (3 catalogs).
- One extra BH family: BH within each feature, but only over tests with overlap >= 2. This is a loose reading of
  the spec's "all terms tested for that feature", because the code only "tests" overlap >= 2.

**What reproduced exactly (all 12 layers).** Term sets (2,377, same names and genes as `terms.json`; same var
list hash). Deployed rule: same annotated counts, same significant rows (0 missing, 0 extra), p and q within
1e-16; overlap, n, K and N columns all match. Kept tests 26,361 to 39,725; largest deployed significant raw p
0.039 to 0.042; 79% to 85% of deployed rows on 2 genes. Bio-sae rule 59.8% to 80.3%. Per-feature BH counts
(104, 86, 85, 83, 81, 84, 86, 91, 88, 80, 55, 55) and per-feature term counts for every single feature (0
differences from `real/layer_XX_feature_flags.csv`). Global BH 55 features (29, 5, 8, 4, 3, 5, 0, 0, 0, 1, 0, 0).
K counted in the universe changes nothing. Wilson 95% intervals (L5 1.38% to 2.11%, features as units). Rows
surviving per-feature BH with overlap 2: 39% to 59%; 153 to 438 per layer from full 20-gene lists, 0 to 81 from
short lists. Phase 6: 50 of 50 picks are the top 50 of the deployed L5 file (45 to 152 hits); 23 annotated under
per-feature BH, 19 with 3 or more terms, 1 under global BH (feature 2560). The 42-of-55 program split also
checks out from the best terms (14 mitochondrial, 13 mitosis/cell cycle, 8 DNA replication/repair, 7 sterol).

**Per-layer numbers with the PPS null.** Null values are means over catalogs; [min-max] over catalogs.
Unit: % of alive features, or annotated features where a count is shown. z = (real - null mean) / null SD
(ddof = 1), unit = null SD. Chance share = null mean / real count.

| Layer | PPS reps | Deployed rule: real / PPS null mean [min-max] | Bio-sae rule: real / PPS null mean | Per-feature BH: real features / PPS null mean [min-max] | z | Chance share | Per-feature BH over overlap >= 2 only: real / PPS null mean | Global BH: real / PPS null mean |
|---|---|---|---|---|---|---|---|---|
| L0 | 10 | 80.7% / 82.6% [82.1%-83.4%] | 80.3% / 81.9% | 104 / 43.5 [26-51] | 7.3 | 42% | 75.1% / 74.2% | 29 / 0.00 |
| L1 | 10 | 74.8% / 74.3% [73.0%-75.8%] | 73.3% / 72.3% | 86 / 42.3 [32-53] | 6.0 | 49% | 65.8% / 64.2% | 5 / 0.00 |
| L2 | 10 | 71.1% / 72.7% [71.5%-73.7%] | 69.8% / 70.5% | 85 / 35.7 [30-44] | 11.5 | 42% | 63.6% / 62.6% | 8 / 0.00 |
| L3 | 10 | 68.3% / 71.0% [70.2%-72.1%] | 67.2% / 68.7% | 83 / 37.7 [29-45] | 7.8 | 45% | 60.4% / 61.0% | 4 / 0.00 |
| L4 | 10 | 66.1% / 69.3% [68.4%-70.2%] | 64.2% / 66.6% | 81 / 31.5 [21-38] | 8.1 | 39% | 58.6% / 59.4% | 3 / 0.00 |
| L5 | 20 | 67.8% / 70.7% [69.6%-71.3%] | 65.9% / 68.4% | 84 / 37.7 [28-45] | 9.1 | 45% | 59.6% / 61.1% | 5 / 0.00 |
| L6 | 10 | 68.4% / 72.3% [71.6%-73.2%] | 67.5% / 70.4% | 86 / 47.3 [29-65] | 3.9 | 55% | 61.1% / 62.9% | 0 / 0.00 |
| L7 | 10 | 69.6% / 75.1% [73.8%-75.9%] | 68.7% / 73.1% | 91 / 45.8 [36-55] | 7.6 | 50% | 61.4% / 65.3% | 0 / 0.00 |
| L8 | 10 | 71.5% / 75.9% [74.7%-77.1%] | 70.5% / 74.0% | 88 / 51.7 [44-62] | 6.2 | 59% | 63.8% / 66.0% | 0 / 0.00 |
| L9 | 10 | 69.8% / 73.4% [72.9%-74.6%] | 68.8% / 71.4% | 80 / 47.2 [43-52] | 9.7 | 59% | 62.0% / 63.6% | 1 / 0.10 |
| L10 | 10 | 64.1% / 66.1% [63.7%-66.9%] | 62.3% / 63.4% | 55 / 33.2 [22-45] | 2.8 | 60% | 56.3% / 56.8% | 0 / 0.00 |
| L11 | 10 | 61.7% / 62.4% [61.3%-63.5%] | 59.8% / 59.6% | 55 / 30.9 [24-43] | 4.1 | 56% | 53.9% / 53.9% | 0 / 0.00 |

What this adds:
1. **The deployed rule is at or below chance at every layer** (real minus null mean: -5.5 to +0.5 points).
   "Below every null catalog" holds at 9 of 12 layers with this null (L0, L2 to L9). At L10 the real rate is
   below 9 of 10 catalogs. So "below all catalogs at 10 of 12 layers" depends a little on the null. The safe
   wording is "at or below the random-list level at every layer".
2. **Per-feature BH beats every null catalog at every layer** (z = 2.8 to 11.5). Chance share 39% to 60%. Note
   that null mean / real count is the share expected if every real feature were noise. If some features are
   real, the true chance share is lower. So "up to about half" is the exact reading.
3. **The layer shape follows which genes are popular, not list grouping.** The relabelling null keeps each list's
   grouping but scrambles gene identity. It gives a flat deployed rate: 66.2% (L0), 67.6% (L5), 66.3% (L11),
   while the real rate goes 80.7%, 67.8%, 61.7%. The PPS null, which keeps gene identity, tracks the real shape
   (Pearson r = 0.95 over 12 layers, 0.91 without L0). Uniform lists: 64.7%, 67.5%, 66.7%.
4. **Restricting the family to overlap >= 2 tests is what breaks it.** Per-feature BH over only overlap >= 2
   tests gives 53.9% to 75.1%, and random lists give the same (real minus null -3.9 to +1.6 points). So the
   problem is dropping overlap 0 and 1 tests before BH, not pooling across features. This backs the choice of
   all 2,377 terms as the per-feature family.
5. **Global BH null:** 0.10 features in total across 12 layers (one catalog at L9 had 1).

**Small fixes made in this report** (numbers unchanged):
1. Section 3.2: the per-feature-BH layer correlation (r = 0.95) comes mostly from L0. Without L0 it is 0.74
   (freq null) and 0.64 (PPS null). Added. The deployed-rule correlation stays high without L0 (0.92 / 0.91).
2. Section 0 item 4: "the labels the paper can trust" changed to "the labels that survive a layer-wide error
   control", with a pointer to the missing co-expression check. Passing BH says the top-20 list overlaps the
   term more than chance. It does not say the model learned anything beyond K562 co-expression programs.

**Other notes.**
- Phase 6 tie: 7 features at L5 have exactly 45 deployed hits. 6 are inside the top 50 and 1 is outside. The
  pick depends on catalog order (stable sort).
- One more downstream use of the deployed labels: the local interactive atlas (`scripts/build_atlas_data.py:72`
  and the atlas block of `full_12layer_pipeline.py`) shows each feature's best deployed term and hit count. Those
  labels are at chance too. I did not check whether this atlas is public.
- `main.tex` does not quote the annotation rates (checked by text search). Its Phase 6 "1.01x" sentence does not
  call the features annotated. The triplet sentence does not mention pathways.

**Not done in this check:** no untrained-SAE baseline (needs a forward pass), no Geneformer recompute, no
co-expression check of the 55 features, no downstream re-runs, STRING not added. PPS null has only 10 catalogs
at 11 of 12 layers, so its smallest empirical p is 1/11 = 0.09 there; the conclusions rest on z and on all
catalogs falling on one side.

**Plain-words summary of the second check.** I rebuilt everything with new code and a new kind of random gene
list that matches each gene's popularity exactly. All numbers came out the same. The old 62-81% rates are what
random lists get. With the right correction, 1-4% of features are annotated, about twice the random level. 55
features pass a layer-wide correction, against almost none for random lists. I made two small wording fixes.
