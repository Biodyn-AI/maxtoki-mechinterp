# V2 cross-model alignment report (revision item D10)

Date: 2026-10-01. CPU only. No model forward pass was run. Only the three embedding tables were read.

## 1. The main result first

**The deployed scGPT comparison read the wrong genes.** `scripts/phase4_scgpt_cross_model.py:119-121` builds the
scGPT gene lookup from the *position* of each symbol in the vocab dict, not from its *index value*. In the saved
scGPT vocab, position equals index for only 1 of 60,697 genes. Under the deployed lookup, 0 of 382 / 350 / 380
panel genes got their own scGPT row. Example: the lookup for KRT18 read the row of NNMT, and for C3 the row of TMEM273.

With the correct lookup, scGPT **does** align with MaxToki:

- Gene-pair Pearson: 0.27 / 0.26 / 0.28 (lung / immune / external lung). Chance is 0.00 (SD 0.004).
- In-sample CCA: 0.72 / 0.74 / 0.73. Chance is 0.41 / 0.43 / 0.41.

All deployed scGPT numbers (CCA 0.40, Pearson about 0, top-1 5-8%) are exactly what an unrelated table gives:

- CCA: 0.401 / 0.438 / 0.406, against chance 0.411 / 0.429 / 0.411. The permutation p values are 0.87 / 0.17 / 0.73.
- Top-1: 5.0 / 6.6 / 7.6%, against a chance level of 5.9 / 7.0 / 5.9% once the rotation is refitted.

So the paper's line "Alignment with scGPT breaks ... the failure isolates cleanly to architectural-family scope"
has no support. Geneformer still aligns better than scGPT, by about 0.10-0.14 in gene-pair Pearson (paired CIs below).

I checked the corrected lookup in three ways:

- (a) The rows are identical (max diff 0.0) to `scgpt_gene_embeddings.npz`. The extraction script saved that file with the right lookup.
- (b) The table equals `encoder.embedding.weight` in the original scGPT checkpoint (diff 0.0). The "normed" table equals LayerNorm of it with `encoder.enc_norm` (diff 1.4e-6).
- (c) Known related pairs have high cosine with the fixed lookup and low cosine with the deployed one. RPL3-RPL5: 0.47 vs 0.10. CD3D-CD3E: 0.62 vs 0.27. HBA1-HBB: 0.57 vs -0.05. Random pairs: 0.09 ± 0.10.

## 2. Exactly what was compared

Every cross-model number in the paper compares **static input embedding tables**. No contextual layer output was compared.

| Model | Tensor | What it is |
|---|---|---|
| MaxToki-217M | `model.embed_tokens.weight` (20,275 x 1,232) | Token embedding table. Same as hidden_states[0] for a gene token (Llama adds no position vector at input). |
| Geneformer V2-316M | `bert.embeddings.word_embeddings.weight` (20,275 x 1,152) | Raw word-embedding table. It is before position embeddings and before the embedding LayerNorm. |
| scGPT whole-human | `encoder.embedding.weight` then `encoder.enc_norm` LayerNorm (60,697 x 512) | Gene-identity encoder output. It is before the expression-value encoder is added. |

- The Phase 14 docstring (`scripts/phase14_cross_model_cca.py:18-20`) says each MaxToki layer 1-11 was aligned to Geneformer per-layer embeddings. The code only runs the static tables (CSV column `tap = static`).
- MaxToki contextual gene states exist on disk (`outputs/phase0/<panel>/layer_gene_embeddings_full.npy`). No matching Geneformer or scGPT states exist.
- So "internal gene geometry" (paper figure caption) and "input embeddings carry most of the gene-pair geometric load" were never tested.

**Gene sets.**
- Topology panels (CCA, Pearson, top-1): 382 genes (lung), 350 (immune), 380 (external lung). Every panel gene is found in all three tables.
- Spectral Phase 9b (the 0.382): 1,500 HVGs from 2,000 Tabula Sapiens immune cells.
- So the paper's 0.382 and its CCA / top-1 numbers come from different gene sets and different pipelines.

**Panels are tissues / datasets, not cell types.**
- `lung` = Tabula Sapiens lung. `immune` = Tabula Sapiens immune. `external_lung` = Krasnow lung Smart-seq2.
- The paper's figure caption says "cell-type panels". That is wrong.
- Static tables do not depend on cells, so a panel only sets the gene list. Genes were picked by TRRUST/STRING/marker priority and expression variance (`scripts/phase0_extract.py:112-163`).
- The lists overlap: lung-immune 75 genes, lung-external 168, immune-external 77. 36 genes are in all three; 828 in the union.
- So the three panels are three overlapping gene samples, not three independent replications.

**Source summaries' own verdict wording.**
- `outputs/phase14_cross_model/phase14_summary.json`: "Layer-1 PARTIALLY REPLICATES (pearson z>3)" (all three panels).
- `EXTENDED_FINDINGS.md:17` (H24-ext): "POSITIVE", with "Top-1 retrieval 38-47% (paper 72%, partial)".
- `EXTENDED_FINDINGS.md:71`: "PARTIALLY REPLICATES vs Geneformer (r=0.78, top-1 38-47%); FAILS vs scGPT".
- `EXTENDED_FINDINGS.md:24` and `FINAL_SUMMARY.md:453`: "does not generalize across architectural families".
- `FINAL_SUMMARY.md:39,61` still say Phase 4 cross-model CCA was "SKIPPED". This text is stale.
- `summaries/spectral-geometry-217M-FINAL_SUMMARY.md:251` calls 0.382 "weaker" than the source paper's 0.825.

The paper's "replicates strongly" is stronger than any of these.

## 3. Methods (what each number is)

All settings copy the deployed code:
- Each table is centred and reduced with PCA to 30 dims (sklearn, random_state 42). PCA is fitted separately per model.
- CCA: sklearn `CCA(n_components=10, max_iter=200)`. The score is the mean of 10 in-sample canonical correlations.
- Gene-pair Pearson: correlation of the two gene-gene cosine matrices (upper triangle, full rows).
- Top-1: an orthogonal Procrustes rotation from MaxToki PCA-30 to the other model's PCA-30. Candidates are all panel genes.

My code reproduces every deployed value to within 3e-8.

- **Chance level.** I permuted the gene correspondence (rows of the other table) 1,000 times, with the same n, dims and settings. CCA was refitted each time. For top-1, the rotation was **refitted** on each permutation. The deployed null did not refit, so it was far too low (0.25-0.30%).
- **Gene bootstrap (unit = gene, with replacement, 1,000 draws, percentile 2.5/97.5).** PCA, CCA and rotation were refitted on each resample. For Pearson, pairs made of two copies of one gene were dropped.
- **Held-out versions.**
  - 5-fold split of genes. PCA, CCA and rotation were fitted on 80% of genes and scored on the other 20%. This was repeated over 10 random splits.
  - Its chance level comes from 200 permutations of the same procedure.
  - The **out-of-bag (OOB) gene bootstrap** fits on each with-replacement resample and scores the about 37% of genes not drawn (1,000 draws). A copy of a gene can never sit on both sides.
- **Paired differences** (Geneformer minus scGPT) use the same genes and the same resamples. Pearson used 2,000 draws; held-out OOB used 500 draws.

## 4. Results per panel

### 4.1 In-sample CCA (the deployed "CCA r")

| Comparison | lung | immune | external lung |
|---|---|---|---|
| Geneformer | 0.783 | 0.776 | 0.766 |
| scGPT, corrected lookup | 0.722 | 0.736 | 0.733 |
| scGPT, deployed lookup | 0.401 | 0.438 | 0.406 |
| Chance mean (SD), 1,000 perms | 0.411 (0.009) | 0.429 (0.010) | 0.412 (0.009) |
| Chance 95th percentile | 0.427 | 0.445 | 0.428 |

- Geneformer is 0.35-0.37 above chance (z 34-40, p ≤ 0.001, the smallest possible with 1,000 permutations).
- Corrected scGPT is 0.31-0.32 above chance (z 32-35).
- So the paper's "0.78" means about 0.36 above a chance level of about 0.41. It does not mean 0.78 above zero.
- The investigation's simulation (0.41-0.43) is confirmed with the real tables and the real sklearn settings.

**A with-replacement gene bootstrap does not give a valid interval for this in-sample statistic.**
- In-sample CCA rises when a resample holds fewer distinct genes. A resample of 382 genes holds only about 240 distinct ones.
- For Geneformer lung, the bootstrap mean is 0.909 and the 95% range is [0.888, 0.929]. That range does not even contain the observed 0.783.
- Permuting whole genes inside the same resamples (copies stay paired with copies) gives a **chance** level of 0.79-0.80 for the Geneformer table (0.74-0.77 for the corrected scGPT table).
- So the percentile interval is not reported. The held-out canonical r below is the number to use with an interval.

A note on precision (corrected during verification, see "Verification notes"): refitting PCA on permuted rows instead of permuting the PCA rows changes the CCA by up to 0.006-0.011. This comes from sklearn's **randomized PCA solver** (which `PCA(random_state=42)` picks for these table shapes), not from the CCA step: sklearn's CCA matches exact CCA to within 1e-6 on the same tables. With the exact PCA solver the deployed in-sample CCA moves by up to +0.011 (Geneformer lung 0.783 -> 0.794) and the in-sample top-1 by up to +4.9 points (Geneformer immune 46.6% -> 51.4%). No conclusion changes.

### 4.2 Held-out canonical correlation (mean of 10)

| Comparison | lung | immune | external lung |
|---|---|---|---|
| Geneformer, 5-fold (range over 10 splits) | 0.604 (0.592-0.619) | 0.558 (0.542-0.584) | 0.578 (0.563-0.591) |
| Geneformer, OOB bootstrap mean [95% CI] | 0.563 [0.517, 0.608] | 0.531 [0.485, 0.575] | 0.544 [0.495, 0.591] |
| scGPT corrected, 5-fold | 0.495 (0.475-0.509) | 0.481 (0.472-0.492) | 0.493 (0.479-0.505) |
| scGPT corrected, OOB mean [95% CI] | 0.449 [0.396, 0.499] | 0.426 [0.371, 0.473] | 0.445 [0.395, 0.491] |
| Chance (5-fold, 200 perms, Geneformer table): mean / 95th pct | 0.00 / 0.027 | 0.00 / 0.032 | 0.00 / 0.031 |
| Chance (same, scGPT corrected table): mean / 95th pct | 0.00 / 0.028 | 0.00 / 0.032 | 0.00 / 0.037 |

- OOB values sit a little below the 5-fold values because OOB trains on about 63% distinct genes, not 80%.
- The first held-out canonical pair is strong for both models: 0.79-0.92 (Geneformer) and 0.77-0.90 (scGPT), 5-fold.

### 4.3 Gene-pair Pearson (cosine-matrix agreement)

| Comparison | lung | immune | external lung | 1,500 HVGs (spectral) |
|---|---|---|---|---|
| Geneformer [95% CI] | 0.398 [0.382, 0.414] | 0.400 [0.385, 0.415] | 0.387 [0.374, 0.401] | **0.382 [0.377, 0.387]** |
| scGPT corrected [95% CI] | 0.265 [0.247, 0.285] | 0.263 [0.246, 0.282] | 0.283 [0.267, 0.301] | 0.239 [0.233, 0.245] |
| scGPT deployed | -0.008 | -0.019 | -0.005 | -0.001 [-0.005, 0.004] |
| Chance mean (SD) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.001) |

The bootstrap mean equals the observed value (for example 0.3980 vs 0.3979). So this interval is not biased by gene copies.

- The deployed immune value (-0.019, z = -3.1 against 1,000 permutations) comes from a fixed wrong-gene table. It says nothing about scGPT.
- The paper's "z = -3.0" should be dropped.

### 4.4 The 0.382 interval, three ways (1,500 HVGs, 1,124,250 gene pairs)

| Method (unit = gene) | 95% interval | SD / SE |
|---|---|---|
| Published: 80% subsample without replacement, 1,000 draws (reproduced to 3e-10) | [0.3797, 0.3843] | 0.0012 |
| Gene bootstrap with replacement, 2,000 draws, copy pairs dropped | **[0.3772, 0.3867]** | 0.0024 |
| Leave-one-gene-out jackknife, normal interval | [0.3775, 0.3864] | 0.0023 |
| Published SD rescaled by sqrt(m/(n-m)) = 2 | [0.3775, 0.3865] | 0.0023 |

- The three proper methods agree. The published interval is about half as wide as it should be.
- The conclusion does not change. Chance is -0.00005 (SD 0.0010, 1,000 permutations).
- The bootstrap uses a weighted-pair formula. It matched direct indexing to 1e-15 on the first 5 draws.

### 4.5 Top-1 retrieval

| Comparison | lung | immune | external lung |
|---|---|---|---|
| Geneformer, in-sample (deployed metric) | 45.0% | 46.6% | 38.2% |
| scGPT corrected, in-sample | 30.9% | 34.9% | 31.3% |
| scGPT deployed, in-sample | 5.0% | 6.6% | 7.6% |
| In-sample chance, rotation refitted (Geneformer table): mean / 95th pct | 6.6% / 8.4% | 8.0% / 10.0% | 7.2% / 9.2% |
| Deployed null (rotation not refitted): mean | 0.27% | 0.30% | 0.25% |
| Geneformer, held-out 5-fold (range) | 17.9% (16.0-21.2) | 15.8% (13.1-18.9) | 14.6% (13.4-16.1) |
| Geneformer, held-out OOB [95% CI] | 14.7% [9.1, 20.7] | 16.2% [10.2, 22.7] | 12.9% [7.7, 18.6] |
| scGPT corrected, held-out 5-fold (range) | 5.9% (3.9-8.1) | 9.1% (6.3-10.6) | 6.8% (5.5-9.2) |
| scGPT corrected, held-out OOB [95% CI] | 5.7% [2.2, 9.8] | 7.4% [3.3, 12.0] | 6.2% [2.3, 10.1] |
| Held-out chance (200 perms): mean / 95th pct | 0.2% / 0.5% | 0.3% / 0.9% | 0.2% / 0.5% |

- The in-sample top-1 fits the rotation on the very genes it then scores. That is why chance is already 6-8%.
- When the query gene is held out, Geneformer retrieval falls from 38-47% to 13-18%. That is still far above the 0.3% chance.
- The scGPT chance level for the corrected table is 5.4% / 6.7% / 5.5%.
- In-sample top-1 has the same copy problem as in-sample CCA. Whole-gene permutation inside a resample gives a chance level of 26-30%. So only held-out top-1 gets a CI.

### 4.6 Geneformer minus scGPT (corrected), paired

| Panel | Gene-pair Pearson [95% CI] | Held-out CCA (OOB) | Held-out top-1 (OOB) |
|---|---|---|---|
| lung | 0.132 [0.119, 0.147] | 0.112 [0.052, 0.172] | 9.3 pts [2.8, 15.8] |
| immune | 0.137 [0.121, 0.153] | 0.106 [0.051, 0.154] | 8.9 pts [1.5, 16.4] |
| external lung | 0.104 [0.088, 0.119] | 0.100 [0.044, 0.159] | 6.7 pts [0.0, 13.5] |
| 1,500 HVGs | 0.143 [0.138, 0.148] | not run | not run |

Geneformer aligns with MaxToki better than scGPT does on every measure. The top-1 gap is the least certain: its interval touches 0 on external lung.

## 5. What differs from the deployed claims and the paper

1. scGPT CCA 0.40, Pearson about 0 and top-1 5-8% are all chance-level values from a gene-lookup bug. With the right genes, scGPT aligns well above chance: CCA 0.72-0.74, Pearson 0.26-0.28.
2. "Alignment with scGPT breaks" and "isolates cleanly to architectural-family scope" are not supported. A fair restatement:
   - MaxToki's input gene table agrees with both other tables.
   - It agrees more with Geneformer (same vocabulary) than with scGPT (different vocabulary, architecture and data).
   - This design cannot say which difference matters.
3. "CCA r = 0.78" needs its chance level, about 0.41 under the same settings. "0.40" for scGPT is the chance level itself.
4. The 0.382 interval should be [0.377, 0.387] (gene bootstrap with replacement), not [0.380, 0.384].
5. "Top-1 38-47%" is in-sample. With the query gene held out it is 13-18% (chance about 0.3%). The deployed null (0.3%) was wrong for the in-sample metric, whose real chance is 6-8%.
6. Panels are tissue/dataset gene lists, not cell types. They overlap, so they are not independent.
7. "Internal gene geometry" should read "input gene-embedding tables". "Same family" is wrong: Geneformer is a BERT encoder and MaxToki a Llama decoder (`runs/spectral-geometry-217M/outputs/phase9b/summary.json`).
8. "Three independent measures agree": CCA, Pearson and top-1 come from the same two tables on the same genes.
9. The paper figure mixes the 1,500-HVG Pearson (0.382) with topology-panel CCA and top-1. `fig_crossmodel.csv` gives both gene sets so a figure can use one.
10. `runs/spectral-geometry-217M/outputs/phase9b/bootstrap_pearson.json` says "scGPT yields r=0.40 on the same comparison". That mixes a CCA value with a Pearson value, and the 0.40 was itself the bug.

## 6. What I did not do

- **No contextual (layer) comparison.** That needs Geneformer and scGPT forward passes, which were not allowed here. The claim about "internal" geometry stays untested.
- **No valid interval for in-sample CCA or in-sample top-1.** I show why above. I report held-out versions with intervals instead.
- **Genes are not independent units.** They share pathways and regulators. So every gene-bootstrap interval here is probably somewhat too narrow (inference, not tested).
- I did not compare Geneformer with scGPT directly. I did not test the raw (pre-LayerNorm) scGPT table, or any other PCA size or number of CCA components.
- I did not re-run the source paper's own scGPT-vs-Geneformer V1 number (0.825).

## 7. Files

Scripts (new):
- `runs/topology-141-217M/scripts/v2_crossmodel_common.py` (deployed metric code, shared)
- `runs/topology-141-217M/scripts/v2_crossmodel_prepare.py` (inputs, lookup checks, reproduction)
- `runs/topology-141-217M/scripts/v2_crossmodel_stats.py` (chance, bootstrap, held-out, OOB; resumable)
- `runs/topology-141-217M/scripts/v2_crossmodel_paired.py` (Geneformer minus scGPT)
- `runs/topology-141-217M/scripts/v2_crossmodel_summarize.py` (summary + figure CSV)
- `runs/spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py` (the 0.382 interval, 1,500 HVGs)

Outputs (new):
- `runs/topology-141-217M/outputs/v2_crossmodel/fig_crossmodel.csv`: figure source data, 64 rows. Each row has value, CI, CI method, chance mean, chance 95th percentile and deployed value.
- `runs/topology-141-217M/outputs/v2_crossmodel/summary.json`: everything above, plus the source verdict wording.
- `runs/topology-141-217M/outputs/v2_crossmodel/paired_differences.json`, `prepare_checks.json`, `run_config.json` (input paths, sha256, seeds, output hashes), `jobs/` (per-job JSON and raw draws), `embeddings_subset.npz`.
- `runs/spectral-geometry-217M/outputs/v2_crossmodel/pearson_1500hvg_ci.json` and `run_config.json`.

Input hashes (sha256, in `run_config.json`):
- MaxToki model.safetensors `9da1fbf8...`
- Geneformer V2-316M model.safetensors `965ceccea819...`. This matches its Hugging Face blob name.
- scGPT embeddings .pt `b41289b8...`
- scGPT checkpoint `6cb5d451...`

## Summary in plain words

- The paper compares only the gene embedding tables at the input of each model, not the model's inner layers.
- MaxToki's table agrees with Geneformer's well above chance. The "CCA 0.78" sounds large, but random tables already score about 0.41 on this test.
- The scGPT comparison was broken. The code looked up the wrong scGPT row for every gene. That is why it looked like "no alignment".
- With the right rows, scGPT also agrees with MaxToki clearly above chance, though less than Geneformer does (Pearson 0.26-0.28 vs 0.39-0.40).
- So the claim that alignment "breaks across architecture families" should be removed.
- The 0.382 interval was about half as wide as it should be. The proper interval is [0.377, 0.387].
- Top-1 retrieval of 38-47% is inflated because the rotation was fitted on the same genes it scored. With the gene held out, it is 13-18% for Geneformer and 6-9% for scGPT, against a chance level of about 0.3%.
- The three "panels" are overlapping gene lists from three tissue datasets, not cell types.

## Verification notes

Added 2026-10-01 by an independent verification pass. CPU only. No model forward pass.
Script: `scripts/v2_crossmodel_verify.py`. Outputs: `outputs/v2_crossmodel/verify/*.json`.
Inputs and their sha256 are in `outputs/v2_crossmodel/verify/run_config.json`.
The script does not import the D10 helper module. It rebuilds every table from the raw files and uses
different code: exact PCA (numpy), exact CCA (QR + SVD), scipy Procrustes, and direct-index bootstraps.

### Verdict

**OK after fixes.** The main finding holds. Two sentences in this report were corrected. No numbers in
`fig_crossmodel.csv`, `summary.json` or the job files were changed.

### What was re-checked, and the result

1. **The scGPT lookup bug is real.** The scGPT `.pt` vocab is a dict that equals the checkpoint's own
   `vocab.json` for all 60,697 genes (symbol -> token id). Dict position equals the id for only 1 gene.
   Under the deployed rule, 0 of 382 / 350 / 380 panel genes get their own row. This check uses a
   source the D10 pass did not use (`vocab.json`), so it does not depend on the npz cross-check.
2. **Rebuilt tables are identical** to D10's `embeddings_subset.npz` (max difference 0.0, all panels, all tables).
3. **Chance level of the in-sample CCA, second method** (exact CCA, exact PCA, 300 permutations):
   0.410 / 0.430 / 0.412 for Geneformer (D10: 0.411 / 0.429 / 0.412). Observed exact-CCA values:
   - Geneformer 0.794 / 0.778 / 0.769
   - scGPT corrected 0.730 / 0.741 / 0.730
   - scGPT deployed 0.411 / 0.437 / 0.399, with p = 0.50 / 0.22 / 0.93. These are at chance.
4. **In-sample top-1 chance with the rotation refitted:** 6.6% / 8.1% / 7.2% (Geneformer table). D10: 6.6 / 8.0 / 7.2%.
5. **Held-out values, second method** (exact PCA and CCA, own fold splits, 3 x 5-fold, 20 permutations, 100 OOB draws):
   - Held-out CCA, Geneformer: 0.607 / 0.573 / 0.579 (D10: 0.604 / 0.558 / 0.578).
   - Held-out CCA, scGPT corrected: 0.499 / 0.496 / 0.504 (D10: 0.495 / 0.481 / 0.493).
   - OOB CI, Geneformer lung: 0.566 [0.515, 0.608] (D10: 0.563 [0.517, 0.608]). Unit = gene, with replacement, percentile.
   - Held-out top-1, Geneformer: 17.4% / 16.6% / 14.2% (D10: 17.9 / 15.8 / 14.6%). scGPT: 3.7% / 8.9% / 6.5% (D10: 5.9 / 9.1 / 6.8%).
   - Held-out chance: CCA about 0.00 (95th percentile 0.02-0.04); top-1 0.2-0.4%.
6. **The 0.382 interval, second method.** Gene bootstrap with replacement by direct indexing (1,000 draws,
   copy pairs dropped, percentile): [0.3773, 0.3865], SD 0.0024. Leave-one-gene-out jackknife: [0.3775, 0.3864].
   D10: [0.3772, 0.3867]. The published [0.380, 0.384] is about half as wide as it should be. Confirmed.
7. **Panel Pearson intervals, second method** (jackknife, unit = gene). They match D10's bootstrap to within 0.003 at each end.
   Example: Geneformer lung 0.398, jackknife [0.384, 0.412], D10 bootstrap [0.382, 0.414].
8. **scGPT on the 1,500 HVGs** (rows from `vocab.json`): 0.2390. Deployed rule: -0.0007. Same as D10.
9. **The in-sample CCA really has no valid with-replacement bootstrap.** With exact CCA, Geneformer lung
   resamples give 0.910 [0.891, 0.928] against an observed 0.794. Whole-gene chance inside the resamples is 0.805.
   Even pure Gaussian noise of the same size goes from 0.412 to 0.556 when rows are resampled with replacement.
10. **scGPT raw table (before LayerNorm), corrected lookup.** D10 listed this as not done. Pearson 0.266 / 0.256 / 0.284
    vs 0.265 / 0.263 / 0.283 for the normed table. Exact CCA 0.731 / 0.740 / 0.730 vs 0.730 / 0.741 / 0.730.
    So the choice of scGPT table does not matter.
11. **`fig_crossmodel.csv` matches the job files.** 135 fields in 64 rows checked; 0 mismatches.
12. **Report text vs files.** All other numbers in sections 1-5 match the job JSONs. The source verdict wording
    (`phase14_summary.json`, `EXTENDED_FINDINGS.md:17,24,71`, `FINAL_SUMMARY.md:39,61,453`,
    `summaries/spectral-geometry-217M-FINAL_SUMMARY.md:251`) was read and is quoted correctly.
    The panel sources (`scripts/phase0_extract.py:47-57`) and the gene selection rule (`:112-163`) are as stated.

### What was fixed in this report

1. **Section 4.1, precision note.** The old text said the 0.006-0.011 wobble comes from "sklearn's iterative CCA".
   That is wrong. sklearn's CCA equals exact CCA to within 1e-6 on the same tables (`verify/permorigin.json`).
   The wobble comes from sklearn's randomized PCA solver: with the exact solver, permuting rows before or after
   PCA gives the same CCA to 4e-16. The note now says this.
2. **New caveat added to the same note.** The deployed in-sample numbers depend on the PCA solver
   (`verify/pcasolver.json`). With the exact solver:
   - in-sample CCA moves by up to +0.011 (Geneformer lung 0.783 -> 0.794);
   - in-sample top-1 moves by up to +4.9 points. Geneformer becomes 48.7 / 51.4 / 40.3% instead of 45.0 / 46.6 / 38.2%.
   - So "38-47%" carries about 5 points of solver noise. It is the wrong metric anyway; the held-out values are stable.
3. **Section 4.1 heading.** "No valid bootstrap interval exists" was too broad. It now says a with-replacement gene
   bootstrap does not give a valid interval. The whole-gene chance range for the scGPT table (0.74-0.77) was added.

### Remaining problems (not fixed)

- scGPT rows are matched by gene **symbol**; Geneformer rows by **Ensembl ID**. All symbols were found (exact case),
  but a symbol that names a different gene in scGPT's older vocabulary cannot be ruled out. This was not tested.
  It could only lower the scGPT numbers a little.
- Genes are not independent units (they share pathways). Every gene-level interval here, D10's and mine, is likely somewhat too narrow.
- The verification used fewer draws than D10 (300 permutations; 3 x 5-fold; 20 held-out permutations; 100 OOB draws).
  The machine was heavily loaded (load average above 170). The agreement with D10 is within the expected Monte Carlo noise.
- Still untested, as D10 says: any layer-wise (contextual) comparison, which needs Geneformer and scGPT forward passes.

### Verification in plain words

- I rebuilt everything from the raw files with different code. The numbers come out the same.
- The scGPT bug is confirmed with scGPT's own vocabulary file. The deployed code read the wrong gene for every row.
- With the right rows, scGPT agrees with MaxToki well above chance. Geneformer agrees more.
- The 0.382 interval should be [0.377, 0.387]. Confirmed two ways.
- One sentence about where the small numerical noise comes from was wrong. It comes from the PCA step, not the CCA step. Fixed.
- The in-sample top-1 numbers move by up to 5 points if PCA is computed exactly. The held-out numbers do not have this problem.
