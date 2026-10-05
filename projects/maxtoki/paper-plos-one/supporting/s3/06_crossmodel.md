## Cross-model alignment of input gene-embedding tables (MaxToki-217M vs Geneformer and scGPT)

This analysis asks whether MaxToki-217M places genes in its input embedding table in a similar way to two other single-cell foundation models. It compares the static gene-embedding tables only: no forward pass is run, so no cell input and no input encoding is involved. Three measures are used: canonical correlation (CCA), agreement of gene–gene cosine matrices (gene-pair Pearson), and top-1 retrieval of the same gene after a rotation.

### Data

**Tables compared** (no contextual layer output was compared):

| Model | Tensor | What it is |
|---|---|---|
| MaxToki-217M | `model.embed_tokens.weight` (20,275 × 1,232) | Token embedding table. Equal to `hidden_states[0]` for a gene token (Llama adds no position vector at the input). |
| Geneformer V2-316M | `bert.embeddings.word_embeddings.weight` (20,275 × 1,152) | Raw word-embedding table, before position embeddings and before the embedding LayerNorm. |
| scGPT whole-human | `encoder.embedding.weight`, then the `encoder.enc_norm` LayerNorm (60,697 × 512) | Gene-identity encoder output, before the expression-value encoder is added. |

Input files (sha256 in `run_config.json`): MaxToki `model.safetensors` 9da1fbf8…; Geneformer V2-316M `model.safetensors` 965ceccea819… (matches its Hugging Face blob name); scGPT embeddings `.pt` b41289b8…; scGPT checkpoint 6cb5d451….

**Gene sets.**
- Three topology panels: lung (Tabula Sapiens lung, 382 genes), immune (Tabula Sapiens immune, 350 genes) and external lung (Krasnow lung Smart-seq2, 380 genes). Genes were picked by TRRUST, STRING and marker priority and by expression variance (`projects/maxtoki/runs/topology-141-217M/scripts/phase0_extract.py:112-163`). Static tables do not depend on cells, so a panel only sets the gene list. The panels are tissue or dataset gene lists, not cell types.
- The panels overlap: lung–immune 75 genes, lung–external 168, immune–external 77; 36 genes are in all three; 828 in the union. So they are three overlapping gene samples, not three independent replications.
- Every panel gene is found in all three tables.
- A fourth gene set for the gene-pair Pearson only: 1,500 highly variable genes (HVGs) from 2,000 Tabula Sapiens immune cells (1,124,250 gene pairs).

**Gene matching.** MaxToki and Geneformer share a vocabulary; rows are matched by Ensembl ID. scGPT rows are found by gene symbol (exact case) through the token index stored in the scGPT vocabulary (`symbol → token id`). Note that the position of a symbol in the vocabulary dictionary equals its token id for only 1 of 60,697 genes, so the index, not the position, must be used.

### Method

All metric settings follow the deployed analysis, so the numbers measure the same thing.

1. **Pre-processing.** Each table is centred and reduced to 30 dimensions by PCA (scikit-learn, `random_state=42`), fitted separately per model.
2. **In-sample CCA.** scikit-learn `CCA(n_components=10, max_iter=200)` on the two PCA-30 tables. Score = mean of the 10 in-sample canonical correlations.
3. **Gene-pair Pearson.** Pearson correlation of the two gene–gene cosine matrices (upper triangle), computed on the full static rows.
4. **Top-1 retrieval.** An orthogonal Procrustes rotation maps MaxToki PCA-30 onto the other model's PCA-30. A gene is retrieved correctly if its own row is the nearest of all panel genes.
5. **Chance level.** The gene correspondence (rows of the other table) is permuted 1,000 times with the same n, dimensions and settings. CCA is refitted each time. For top-1, the rotation is refitted on each permutation.
6. **Gene bootstrap.** Genes are resampled with replacement (1,000 draws; percentile 2.5 / 97.5). PCA, CCA and the rotation are refitted on each resample. For the Pearson measure, pairs made of two copies of one gene are dropped.
7. **Held-out versions.**
   - 5-fold split of genes: PCA, CCA and rotation are fitted on 80% of genes and scored on the other 20%; repeated over 10 random splits (the range over splits is given). Chance from 200 permutations of the same procedure.
   - Out-of-bag (OOB) gene bootstrap: fit on each with-replacement resample and score the about 37% of genes not drawn (1,000 draws). A copy of a gene can never sit on both sides.
8. **Paired differences** (Geneformer minus scGPT) use the same genes and the same resamples: 2,000 draws for the Pearson measure, 500 draws for the held-out OOB measures.
9. **The 1,500-HVG Pearson interval.** Gene bootstrap with replacement (2,000 draws, copy pairs dropped, weighted-pair formula, seed 20261101), plus a leave-one-gene-out jackknife (normal interval). Chance from 1,000 permutations of gene labels.

### Checks

- **scGPT rows.** (a) The rows used are identical (largest difference 0.0) to `scgpt_gene_embeddings.npz`, which the extraction script saved with an index lookup. (b) The table equals `encoder.embedding.weight` in the original scGPT checkpoint (difference 0.0), and the normed table equals its LayerNorm with `encoder.enc_norm` (difference 1.4 × 10⁻⁶). (c) Known related pairs have high cosine: RPL3–RPL5 0.47, CD3D–CD3E 0.62, HBA1–HBB 0.57; random pairs 0.09 ± 0.10.
- **Reproduction.** The metric code reproduces every deployed value to within 3 × 10⁻⁸.
- **Pearson bootstrap.** The bootstrap mean equals the observed value (for example 0.3980 vs 0.3979), so this interval is not biased by gene copies. The weighted-pair formula for the 1,500-HVG bootstrap matched direct indexing to 10⁻¹⁵ on the first 5 draws.
- **No valid interval for in-sample CCA.** In-sample CCA rises when a resample holds fewer distinct genes; a resample of 382 genes holds only about 240 distinct ones. For Geneformer lung, the bootstrap mean is 0.909 and the 95% range [0.888, 0.929], which does not contain the observed 0.783. Permuting whole genes inside the same resamples gives a chance level of 0.79–0.80 for the Geneformer table and 0.74–0.77 for the scGPT table. So no percentile interval is reported for in-sample CCA. In-sample top-1 has the same problem (whole-gene chance inside a resample 26–30%), so only held-out top-1 gets an interval.
- **PCA solver.** For these table shapes `PCA(random_state=42)` uses scikit-learn's randomized solver. Refitting PCA on permuted rows instead of permuting the PCA rows changes CCA by up to 0.006–0.011. This comes from the randomized PCA, not from CCA: scikit-learn's CCA matches exact CCA within 10⁻⁶ on the same tables. With the exact PCA solver, in-sample CCA moves by up to +0.011 (Geneformer lung 0.783 → 0.794) and in-sample top-1 by up to +4.9 points (Geneformer 48.7 / 51.4 / 40.3% instead of 45.0 / 46.6 / 38.2%). No conclusion changes. The held-out values are stable.

### Results

**Main findings.**
- MaxToki's input gene table agrees with both other tables well above chance. It agrees more with Geneformer (same vocabulary) than with scGPT (different vocabulary, architecture and training data). This design cannot say which difference matters.
- In-sample CCA has a high chance level (about 0.41). The Geneformer value of 0.78 is about 0.36 above it.
- Top-1 retrieval of 38–47% is in-sample. With the query gene held out, it is 13–18% for Geneformer and 6–9% for scGPT, against a chance level of about 0.3%.
- For context, the deployed analysis built the scGPT lookup from dictionary positions, so 0 of 382 / 350 / 380 panel genes received their own scGPT row, and its scGPT values (in-sample CCA 0.401 / 0.438 / 0.406; Pearson −0.008 / −0.019 / −0.005) are at the chance level.

**In-sample CCA** (mean of 10 canonical correlations; chance from 1,000 permutations of gene correspondence):

| Comparison | Lung (382 genes) | Immune (350) | External lung (380) |
|---|---|---|---|
| Geneformer | 0.783 | 0.776 | 0.766 |
| scGPT | 0.722 | 0.736 | 0.733 |
| Chance mean (SD), Geneformer table | 0.411 (0.009) | 0.429 (0.010) | 0.412 (0.009) |
| Chance 95th percentile, Geneformer table | 0.427 | 0.445 | 0.428 |

- Geneformer is 0.35–0.37 above chance (z 34–40; p ≤ 0.001, the smallest possible with 1,000 permutations).
- scGPT is 0.31–0.32 above chance (z 32–35).

**Held-out canonical correlation** (mean of 10; 5-fold: mean and range over 10 random splits; OOB: mean and 95% percentile interval over 1,000 gene-bootstrap draws, unit = gene; chance from 200 permutations):

| Comparison | Lung | Immune | External lung |
|---|---|---|---|
| Geneformer, 5-fold (range) | 0.604 (0.592–0.619) | 0.558 (0.542–0.584) | 0.578 (0.563–0.591) |
| Geneformer, OOB [95% CI] | 0.563 [0.517, 0.608] | 0.531 [0.485, 0.575] | 0.544 [0.495, 0.591] |
| scGPT, 5-fold (range) | 0.495 (0.475–0.509) | 0.481 (0.472–0.492) | 0.493 (0.479–0.505) |
| scGPT, OOB [95% CI] | 0.449 [0.396, 0.499] | 0.426 [0.371, 0.473] | 0.445 [0.395, 0.491] |
| Chance, Geneformer table: mean / 95th percentile | 0.00 / 0.027 | 0.00 / 0.032 | 0.00 / 0.031 |
| Chance, scGPT table: mean / 95th percentile | 0.00 / 0.028 | 0.00 / 0.032 | 0.00 / 0.037 |

- OOB values sit a little below the 5-fold values because OOB trains on about 63% distinct genes, not 80%.
- The first held-out canonical pair is strong for both models: 0.79–0.92 (Geneformer) and 0.77–0.90 (scGPT), 5-fold.

**Gene-pair Pearson** (agreement of gene–gene cosine matrices; 95% percentile interval from a gene bootstrap with replacement, unit = gene, copy pairs dropped: 1,000 draws for the panels, 2,000 for the 1,500 HVGs with Geneformer and 1,000 with scGPT; chance from 1,000 permutations for the panels and Geneformer HVGs, 200 for scGPT HVGs):

| Comparison | Lung | Immune | External lung | 1,500 HVGs |
|---|---|---|---|---|
| Geneformer [95% CI] | 0.398 [0.382, 0.414] | 0.400 [0.385, 0.415] | 0.387 [0.374, 0.401] | **0.382 [0.377, 0.387]** |
| scGPT [95% CI] | 0.265 [0.247, 0.285] | 0.263 [0.246, 0.282] | 0.283 [0.267, 0.301] | 0.239 [0.233, 0.245] |
| Chance mean (SD) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.004) | 0.000 (0.001) |

**The 1,500-HVG Geneformer interval, two ways** (n = 1,500 genes, unit = gene):

| Method | 95% interval | SD / SE |
|---|---|---|
| Gene bootstrap with replacement, 2,000 draws, copy pairs dropped, percentile | [0.3772, 0.3867] | 0.0024 |
| Leave-one-gene-out jackknife, normal interval | [0.3775, 0.3864] | 0.0023 |

Chance: −0.00005 (SD 0.0010, 1,000 permutations).

**Top-1 retrieval** (in-sample chance: rotation refitted on each of 1,000 permutations; held-out 5-fold: mean and range over 10 splits; OOB: mean and 95% percentile interval over 1,000 gene-bootstrap draws, unit = gene; held-out chance from 200 permutations):

| Comparison | Lung | Immune | External lung |
|---|---|---|---|
| Geneformer, in-sample | 45.0% | 46.6% | 38.2% |
| scGPT, in-sample | 30.9% | 34.9% | 31.3% |
| In-sample chance, Geneformer table: mean / 95th percentile | 6.6% / 8.4% | 8.0% / 10.0% | 7.2% / 9.2% |
| In-sample chance, scGPT table: mean | 5.4% | 6.7% | 5.5% |
| Geneformer, held-out 5-fold (range) | 17.9% (16.0–21.2) | 15.8% (13.1–18.9) | 14.6% (13.4–16.1) |
| Geneformer, held-out OOB [95% CI] | 14.7% [9.1, 20.7] | 16.2% [10.2, 22.7] | 12.9% [7.7, 18.6] |
| scGPT, held-out 5-fold (range) | 5.9% (3.9–8.1) | 9.1% (6.3–10.6) | 6.8% (5.5–9.2) |
| scGPT, held-out OOB [95% CI] | 5.7% [2.2, 9.8] | 7.4% [3.3, 12.0] | 6.2% [2.3, 10.1] |
| Held-out chance: mean / 95th percentile | 0.2% / 0.5% | 0.3% / 0.9% | 0.2% / 0.5% |

In-sample top-1 fits the rotation on the same genes it then scores. That is why its chance level is already 6–8%.

**Geneformer minus scGPT, paired** (same genes and same resamples; 95% percentile interval; Pearson: 2,000 gene-bootstrap draws; held-out measures: 500 OOB draws):

| Gene set | Gene-pair Pearson [95% CI] | Held-out CCA (OOB) [95% CI] | Held-out top-1 (OOB) [95% CI] |
|---|---|---|---|
| Lung | 0.132 [0.119, 0.147] | 0.112 [0.052, 0.172] | 9.3 points [2.8, 15.8] |
| Immune | 0.137 [0.121, 0.153] | 0.106 [0.051, 0.154] | 8.9 points [1.5, 16.4] |
| External lung | 0.104 [0.088, 0.119] | 0.100 [0.044, 0.159] | 6.7 points [0.0, 13.5] |
| 1,500 HVGs | 0.143 [0.138, 0.148] | not run | not run |

Geneformer aligns with MaxToki better than scGPT does on every measure. The top-1 gap is the least certain: its interval touches 0 on external lung.

### Verification

A second agent re-checked this analysis with separate code (`scripts/v2_crossmodel_verify.py`; outputs `outputs/v2_crossmodel/verify/*.json`, inputs and sha256 in `verify/run_config.json`). The script does not import the main helper module. It rebuilds every table from the raw files and uses exact PCA (numpy), exact CCA (QR and SVD), scipy Procrustes and direct-index bootstraps. It used fewer draws than the main analysis because the machine was heavily loaded. Results:

1. **scGPT rows.** The scGPT `.pt` vocabulary equals the checkpoint's own `vocab.json` for all 60,697 genes. Dictionary position equals the token id for only 1 gene. This uses a source the main analysis did not use.
2. **Tables.** The rebuilt tables are identical to `embeddings_subset.npz` (largest difference 0.0, all panels, all tables).
3. **In-sample CCA chance, second method** (exact CCA, exact PCA, 300 permutations): 0.410 / 0.430 / 0.412 for Geneformer (main: 0.411 / 0.429 / 0.412). Observed exact-CCA values: Geneformer 0.794 / 0.778 / 0.769; scGPT 0.730 / 0.741 / 0.730.
4. **In-sample top-1 chance, rotation refitted:** 6.6% / 8.1% / 7.2% (main: 6.6 / 8.0 / 7.2%).
5. **Held-out values, second method** (exact PCA and CCA, own fold splits, 3 × 5-fold, 20 permutations, 100 OOB draws):
   - held-out CCA, Geneformer 0.607 / 0.573 / 0.579 (main 0.604 / 0.558 / 0.578); scGPT 0.499 / 0.496 / 0.504 (main 0.495 / 0.481 / 0.493);
   - OOB interval, Geneformer lung 0.566 [0.515, 0.608] (main 0.563 [0.517, 0.608]); unit = gene, with replacement, percentile;
   - held-out top-1, Geneformer 17.4% / 16.6% / 14.2% (main 17.9 / 15.8 / 14.6%); scGPT 3.7% / 8.9% / 6.5% (main 5.9 / 9.1 / 6.8%);
   - held-out chance: CCA about 0.00 (95th percentile 0.02–0.04); top-1 0.2–0.4%.
6. **The 1,500-HVG interval, second method.** Gene bootstrap with replacement by direct indexing (1,000 draws, copy pairs dropped, percentile): [0.3773, 0.3865], SD 0.0024. Jackknife: [0.3775, 0.3864]. Main: [0.3772, 0.3867].
7. **Panel Pearson intervals, second method** (jackknife, unit = gene): within 0.003 of the main bootstrap at each end. Example: Geneformer lung 0.398, jackknife [0.384, 0.412], main bootstrap [0.382, 0.414].
8. **scGPT on the 1,500 HVGs** (rows from `vocab.json`): 0.2390, the same as the main analysis.
9. **No valid with-replacement bootstrap for in-sample CCA, confirmed.** With exact CCA, Geneformer lung resamples give 0.910 [0.891, 0.928] against an observed 0.794; whole-gene chance inside the resamples is 0.805. Even pure Gaussian noise of the same size goes from 0.412 to 0.556 when rows are resampled with replacement.
10. **Raw scGPT table (before LayerNorm).** Pearson 0.266 / 0.256 / 0.284 vs 0.265 / 0.263 / 0.283 for the normed table; exact CCA 0.731 / 0.740 / 0.730 vs 0.730 / 0.741 / 0.730. The choice of scGPT table does not matter.
11. **Figure data.** `fig_crossmodel.csv` matches the job files (135 fields in 64 rows; 0 mismatches).
12. **PCA solver.** The verifier found that the small run-to-run differences in CCA come from the randomized PCA solver, not from CCA, and measured the solver effect on in-sample CCA and top-1 (reported in Checks).

The verifier agreed with every number within Monte Carlo noise.

### Limits

- **No contextual (layer) comparison.** Only static input tables were compared. This needs Geneformer and scGPT forward passes, which were not run. Any claim about the models' internal gene geometry stays untested.
- **No valid interval for in-sample CCA or in-sample top-1.** Held-out versions with intervals are reported instead.
- **Genes are not independent units.** They share pathways and regulators, so every gene-bootstrap interval here is probably somewhat too narrow (not tested).
- **scGPT rows are matched by gene symbol**, Geneformer rows by Ensembl ID. All symbols were found, but a symbol that names a different gene in scGPT's older vocabulary cannot be ruled out. This could only lower the scGPT numbers a little.
- One PCA size (30) and one number of CCA components (10). Other settings were not tried.
- Geneformer and scGPT were not compared with each other directly.
- The three panels overlap and come from tissue datasets, so they are not independent replications. The 1,500-HVG set and the panels are different gene sets; numbers from them should not be mixed in one figure without saying so.
- The in-sample numbers carry up to about 0.011 (CCA) and 5 points (top-1) of PCA-solver noise.

### Files

- Scripts: `projects/maxtoki/runs/topology-141-217M/scripts/v2_crossmodel_common.py` (metric code), `v2_crossmodel_prepare.py` (inputs, lookup checks, reproduction), `v2_crossmodel_stats.py` (chance, bootstrap, held-out, OOB; resumable), `v2_crossmodel_paired.py` (Geneformer minus scGPT), `v2_crossmodel_summarize.py` (summary and figure CSV), `v2_crossmodel_verify.py` (second agent); `projects/maxtoki/runs/spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py` (1,500-HVG interval).
- Outputs (`projects/maxtoki/runs/topology-141-217M/outputs/v2_crossmodel/`): `fig_crossmodel.csv` (figure source data, 64 rows; each row has value, interval, interval method, chance mean and 95th percentile; it also holds rows for the dictionary-position lookup), `summary.json`, `paired_differences.json`, `prepare_checks.json`, `run_config.json` (input paths, sha256, seeds, output hashes), `jobs/` (per-job JSON and raw draws), `embeddings_subset.npz`, `verify/` (`lookup.json`, `insample.json`, `heldout.json`, `pearson.json`, `hvgscgpt.json`, `rawscgpt.json`, `bootcheck.json`, `permorigin.json`, `pcasolver.json`, `csvcheck.json`, `run_config.json`).
- 1,500-HVG outputs: `projects/maxtoki/runs/spectral-geometry-217M/outputs/v2_crossmodel/pearson_1500hvg_ci.json`, `run_config.json`.
- CPU only; no model forward pass.
