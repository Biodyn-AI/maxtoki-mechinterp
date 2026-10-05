# Task data: TF specificity of MaxToki-217M layer-5 SAE features (revision item D3)

Made by `runs/sae-atlas-217M/scripts/v2_tf_specificity_*.py` (see `../run_config.json`).
Everything here is for MaxToki-217M, layer-5 SAE (`outputs/phase1/layer_05/sae_final.pt`, 4,928
TopK features, k = 32), applied to the INPUT of decoder block 5 (= `hidden_states[5]`), which is
the tensor the SAE was trained on. Gene symbols are upper case.

## Definitions

* **Gene universe**: the 6,324 genes that appear as tokens in the 500 K562 non-targeting control
  cells used to build the deployed catalog (full_12layer_pipeline.py, seed 42).
* **Detection count** of a gene: number of those 500 cells in which the gene is one of the cell's
  tokens (rank-value encoding, max_len 2,048; a gene appears at most once per cell). Maximum 500.
* **Top-20 genes of a feature** (deployed definition, kept unchanged): encode every token position
  of the 500 catalog cells; for each (feature, gene) take the mean activation over the positions
  where the feature is active (> 0) and the token is that gene; the 20 genes with the highest mean.
  No minimum count, so a gene seen once can enter. `<SPECIAL>` (= `<bos>`/`<eos>`) can enter the
  list (59 features); it is never a target. We rebuilt this catalog from our own forward passes:
  4,927 of 4,928 lists are identical (same genes, same order); one differs by one gene.
* **Per-cell feature value**: mean activation of the feature over the cell's gene tokens (all
  positions except `<bos>` and `<eos>`).
* **Responding feature** of a TF: two-sided Mann-Whitney U test, knockdown cells vs 400
  reference control cells, per feature; BH across the 4,928 features; q < 0.05 and
  |mean(knockdown) - mean(reference)| > cut-off. Deployed cut-off = 0.5; also 0.25, 0.1, 0.05, 0.02,
  0.01, 0.
* **Count bins** (for matched nulls): detection count digitised at 1, 2, 5, 10, 20, 50, 100, 200,
  500 (bins 1..9). **Length tertile**: genomic span (end - start, from
  `biotensor/data/genemanifold/gene_pos.json`) cut at 21,375 and 56,220 bp (0, 1, 2); genes without
  a length (54) are put in tertile 1. `count_x_length_bin` = 10 x count bin + tertile.

## Files

| file | rows | columns |
|---|---|---|
| `gene_universe.tsv` | 6,324 genes | gene, detection_count, gene_length_bp (NaN if unknown), chr, count_bin, length_tertile, count_x_length_bin |
| `feature_top20.tsv` | 97,492 (feature, rank) rows (4,807 features have 20 genes, 121 have fewer) | feature_id, rank (1..20), gene (deployed list), mean_act_when_active_rebuilt (our rebuild), n_active_positions_rebuilt (positions where the feature is active on this gene), gene_detection_count, in_rebuilt_top20 |
| `feature_info.tsv` | 4,928 features | n_top20, n_special_in_top20, median detection count and median length (kb) of the top-20 genes, deployed activation frequency, active positions in the rebuild, rebuilt list = deployed list |
| `targets_trrust.tsv` | TRRUST edges with target in universe | tf, target, mode (Activation/Repression/Unknown, ';'-joined), n_pmid |
| `targets_dorothea.tsv` | DoRothEA edges with target in universe | tf, target, confidence (A..D; a few 'B;D' etc.), chip_flag (edge is in the DoRothEA ChIP-seq subset = the "ChIP" sets) |
| `cell_manifest.csv` | 9,200 cells | row (0-based row of `X` in replogle_concat.h5ad), group (ref / pool / kd / catalog), tf (for kd), priority, barcode, gem_group, UMI_count, is_primary_tf |
| `cell_feature_means.npz` | 9,200 cells | rows (dataset row), mean_gene (cells x 4,928 per-cell feature values, float32), n_tokens |
| `responding_features.tsv` | every (TF, feature) with BH q < 0.05 | tf, feature_id, delta_mean (knockdown - reference, signed), cohen_d, mwu_p, bh_q, n_kd_cells, resp_cut_<c> (passes the effect cut-off c) |
| `tf_summary.tsv` | 87 TFs | n_kd_cells, n_bh_sig, K_cut_<c> (number of responding features), target-set sizes in universe, kd_remaining_frac_linear (mean expm1 expression of the TF gene in knockdown / reference cells; empty if the TF gene is not in the 6,546-gene panel) |
| `tf_tests.tsv` | TF x cut-off x database | all statistics and p-values (same as `../tf_results.csv`; column guide below) |
| `catalog_check.json` | – | token check and catalog rebuild check |

Cell groups: `ref` = 400 reference controls; `pool` = 800 held-out controls used for fake TFs;
`kd` = 7500 knockdown cells (up to 100 per TF, 87 TFs); `catalog` = the 500 catalog cells.
`ref`, `pool` and `catalog` are disjoint random K562 non-targeting cells.

### Column guide for `tf_tests.tsv`
* K = number of responding features; M_obs = best overlap of a responding feature's top-20 with the
  target set; T_obs = min(M, 5) if M >= 2 else 0 (the deployed thresholds 2..5 as one statistic).
* p_rf: deployed random-feature null (K random catalog features), exact. p_cm / p_cml: gene-swap
  null matched on detection count / on count and length, exact (hypergeometric), with threshold
  search over 2..5; *_uncapped: same with the uncapped M. E_M_cm: expected M under p_cm's null.
* p_fake: selection-aware null, 2,000 fake TFs (random pool subsets of the knockdown group's size,
  whole selection re-run). p_fakeK: fake TFs keep their K most-changed features. p_okd: responding
  features of the other knockdowns (n_okd of them with K > 0); okd_min_attainable_p = 1/(1+n_okd).
* rank_cm: rank of the true TF among all TFs with a set (ChIP: >= 20 targets in universe, 291 sets;
  TRRUST: >= 5, ~109 sets) by p_cm with the same features; n_rank_cand; only meaningful when T_obs >= 2.
* q_<test>_primary / q_<test>_all: BH across the 20 primary TFs / all 87 TFs that have a target
  set in that database (TFs with K = 0 enter with p = 1).
