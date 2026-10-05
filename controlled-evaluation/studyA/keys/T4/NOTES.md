# T4 answer key: notes on the reference

Task: cross-model gene-embedding alignment. MaxToki-217M vs scGPT whole-human, with
Geneformer V2-316M (same gene vocabulary as MaxToki) as the reference comparison.

Key verdict: **aligned with scGPT but clearly less than with Geneformer.** No alternative verdict is accepted.

## 1. What the reference does

`reference.py` reads only the package `data/` folder (default `studyA/tasks/T4-paper/`).

Gene mapping:
- MaxToki row = `maxtoki_217m/token_dictionary.json[ensembl_id]`.
- Geneformer row = `geneformer_v2_316m/token_dictionary.json[ensembl_id]`. These ids are identical to MaxToki's for all 20,271 genes.
- scGPT row = `scgpt_whole_human/vocab.json[symbol]`, the dict **value**. The symbol comes from `gene_ids/ensembl_symbol.csv`.
  All 1,112 panel rows (828 distinct genes) are found with exact case.
- scGPT vector = LayerNorm of the row with `encoder_enc_norm.npz` (eps 1e-5). This equals the table the deployment used to within 1.4e-6.
- It also computes the "positional" lookup (row = position of the symbol among the `vocab.json` keys). That is the deployed rule.
  It is computed only to show the numbers that rule produces. Under it, 0 of 382 / 350 / 380 genes get their own row.

Metrics, all on each panel separately:
- **Gene-pair Pearson**: Pearson r between the upper triangles of the two gene x gene cosine matrices, using full vectors.
- **In-sample CCA**: mean of 10 canonical correlations on PCA-30 of each table.
  Computed two ways: the deployed setting (sklearn randomized PCA, random_state 42, sklearn CCA) and exact (SVD PCA, closed-form CCA).
- **In-sample top-1**: Procrustes rotation fitted and scored on the same genes.
- **Held-out CCA and held-out top-1**: 5-fold split of genes, 10 random splits. PCA, CCA and rotation are fitted on 80% of genes and scored on the other 20%.
- **Chance levels**: the gene correspondence is permuted, and CCA and the rotation are refitted each time.
  1,000 permutations for in-sample metrics and Pearson; 200 for held-out metrics.
- **Intervals** (unit = gene):
  - Pearson: gene bootstrap with replacement, 1,000 draws, with pairs of two copies of one gene dropped.
  - Held-out metrics: out-of-bag (OOB) bootstrap, 500 draws (fit on the resample, score on the genes not drawn).
  - Geneformer minus scGPT: paired, same resamples (2,000 draws for Pearson, 500 OOB draws).
- **Extras** (not in the source report): Pearson on the union of the panels (828 genes) and on 2,000 random genes found in all three vocabularies.

Speed. The reference uses an exact shortcut. It replaces each n x p table with its n x n SVD coordinates. These keep all inner products, so every metric is unchanged.
For PCA it uses the top-30 eigenvectors (`scipy.linalg.eigh`), which give the same subspace as the SVD.
The machine was heavily loaded (load average 80-400), so the run was split into parts, each under 9 minutes:
`reference.py --part=lung`, `--part=immune`, `--part=external_lung`, `--part=extras`, then `reference.py --combine > reference_output.json`.
Per-part JSON files are in `parts/`. With no flag it runs every part in turn.

## 2. Results (reference_output.json) and the source report

| Quantity | lung | immune | external lung | Source report |
|---|---|---|---|---|
| Pearson, Geneformer | 0.398 [0.383, 0.414] | 0.400 [0.385, 0.416] | 0.387 [0.374, 0.402] | 0.398 / 0.400 / 0.387 |
| Pearson, scGPT (token-id lookup) | 0.265 [0.246, 0.287] | 0.263 [0.245, 0.282] | 0.283 [0.267, 0.300] | 0.265 / 0.263 / 0.283 |
| Pearson, chance mean (SD) | -0.000 (0.004) | -0.000 (0.004) | 0.000 (0.004) | 0.000 (0.004) |
| Pearson difference, Geneformer minus scGPT | 0.132 [0.118, 0.147] | 0.137 [0.121, 0.153] | 0.104 [0.089, 0.119] | 0.132 / 0.137 / 0.104, same CIs to 0.002 |
| In-sample CCA, Geneformer (deployed / exact) | 0.783 / 0.794 | 0.776 / 0.779 | 0.766 / 0.769 | 0.783 / 0.776 / 0.766 (exact 0.794 / 0.778 / 0.769) |
| In-sample CCA, scGPT (deployed / exact) | 0.722 / 0.730 | 0.736 / 0.741 | 0.733 / 0.730 | 0.722 / 0.736 / 0.733 |
| In-sample CCA, chance mean | 0.411 | 0.430 | 0.412 | 0.411 / 0.429 / 0.412 |
| Held-out CCA, Geneformer (5-fold) | 0.612 | 0.573 | 0.573 | 0.604 / 0.558 / 0.578 (verification pass 0.607 / 0.573 / 0.579) |
| Held-out CCA, scGPT (5-fold) | 0.498 | 0.491 | 0.498 | 0.495 / 0.481 / 0.493 (verification pass 0.499 / 0.496 / 0.504) |
| Held-out CCA difference (OOB) | 0.108 [0.056, 0.168] | 0.104 [0.049, 0.159] | 0.100 [0.042, 0.165] | 0.112 / 0.106 / 0.100 |
| Held-out top-1, Geneformer | 17.1% | 17.1% | 14.8% | 17.9 / 15.8 / 14.6% |
| Held-out top-1, scGPT | 4.8% | 9.6% | 6.6% | 5.9 / 9.1 / 6.8% |
| Held-out top-1 difference (OOB) | 9.2 pts [2.8, 15.6] | 8.8 pts [1.6, 16.7] | 6.8 pts [0.7, 13.3] | 9.3 / 8.9 / 6.7 pts; CI on external lung [0.0, 13.5] |
| In-sample top-1, Geneformer (deployed) | 45.0% | 46.6% | 38.2% | same |
| In-sample top-1 chance, rotation refitted | 6.6% | 7.9% | 7.2% | 6.6 / 8.0 / 7.2% |
| Positional lookup: Pearson / CCA / top-1 | -0.008 / 0.401 / 5.0% | -0.019 / 0.438 / 6.6% | -0.005 / 0.406 / 7.6% | same |
| scGPT raw table (no LayerNorm), Pearson | 0.266 | 0.256 | 0.284 | verification: 0.266 / 0.256 / 0.284 |

Extras: union of the panels (828 genes): Pearson 0.373 (Geneformer) vs 0.238 (scGPT); difference CI [0.128, 0.142].
2,000 random shared genes: 0.427 vs 0.218.
So the gap is not limited to the panel genes.

Sanity pairs, scGPT cosine with token-id lookup vs positional lookup:
- CD3D-CD3E 0.62 vs 0.27
- HBA1-HBB 0.57 vs -0.05
- RPL3-RPL5 0.47 vs 0.10
- random pairs 0.09 ± 0.10

## 3. Agreement with the source report

The reference agrees with the corrected analysis in `V2_CROSSMODEL_REPORT.md` and its verification notes. Every key number is within its tolerance.
Small differences, and why:
- **Held-out values** differ from the report by up to 0.015 (CCA) and 1.3 points (top-1). The reference uses exact PCA and its own fold splits; the report used sklearn's randomized PCA. The report's own verification pass shows the same size of difference.
- **The OOB top-1 difference interval on external lung** is [0.7, 13.3] here and [0.0, 13.5] in the report. The difference comes from Monte Carlo noise. Both say this is the least certain gap.

No disagreement needed investigation.

## 4. Why this verdict

- **scGPT is aligned beyond chance.** It is far above chance on every metric and panel:
  - Pearson 0.26-0.28 vs chance 0.00 (SD 0.004);
  - held-out CCA about 0.49 vs chance 0.00 (95th percentile about 0.03);
  - held-out top-1 5-10% vs chance about 0.2%.
- **It is clearly less aligned than Geneformer.**
  - Paired Pearson differences of 0.10-0.14 are about a third of Geneformer's value. Their CIs exclude 0 on every panel.
  - Held-out CCA differences are about 0.10, with CIs excluding 0.
  - The top-1 gap is the least certain: its CI comes near 0 on external lung.
  - Raw in-sample CCA (0.72-0.74 vs 0.77-0.78) makes the two look closer than they are. Above chance, it is about 0.32 vs 0.35-0.38.
- **"not aligned beyond chance"** comes only from the positional lookup.

## 5. Choices made when building the package

- **Which spec governs.** `pipelines/topology-geometry-141-hypotheses.md` governs this analysis. Its Phase 4 (H24 CCA, H20 Procrustes, H17) is the method that the deployed `phase14_cross_model_cca.py` and `phase4_scgpt_cross_model.py` implement.
  The spectral-geometry spec only has a raw-cosine scGPT x Geneformer check (Phase 9b).
  So the contract is the topology spec (file dated 2026-04-15 20:58, sha256 `43731ef3...`), copied verbatim.
  The source paper is `2602.22289_Kendiukhov_topology_141_hypotheses.pdf`.
- **What the verbatim spec contains.**
  - It has the source method's own word "audit" (the "strict max-null audit", H141).
  - It has relative `repos/...` code paths.
  - It has a pitfall saying gene-level top-1 above 10% suggests a bug, and that 19 correspondence methods fail.
  These were left unchanged because the spec is the treatment.
- **The scGPT table.** It is shipped as the checkpoint tensor `encoder.embedding.weight` plus the `enc_norm` LayerNorm parameters, rows in checkpoint order.
  This is "as distributed". `vocab.json` and `args.json` are byte-for-byte copies.
  The deployment used the LayerNorm output. Raw and normed tables give the same conclusion; tolerances cover both.
- **The Ensembl-symbol table.** It is Geneformer's `gene_name_id_dict_gc104M.pkl`, which is one-to-one.
  For all 1,112 panel rows it gives the same symbol the deployment used (from the CELLxGENE `feature_name`).
- **The panels.** They contain only Ensembl IDs, in the deployed row order. The MaxToki token id, `var_idx` and annotation flags were removed. No table is pre-joined.
- **No subsetting.** The tables were not subset, because the package is 308 MB.

## 6. Not done

- No layer-wise (contextual) comparison. The package has only input tables, and the question as built is about them.
- No sensitivity analysis over PCA size or the number of CCA components. The deployed 30 and 10 were used.
- **Mismatch with the pre-registration.** `protocol/PREREGISTRATION.md` section 3 describes T4 as a developmental-ordering transfer task, not cross-model alignment.
  This package follows the task definition given by the orchestrating workflow. The pre-registration text should be reconciled before Study A runs.
