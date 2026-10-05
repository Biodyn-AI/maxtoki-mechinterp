# Task data: TF specificity of MaxToki-217M v3 layer-5 SAE features (item V3-2)

Made by `runs/sae-atlas-217M/scripts/v3_tf_specificity_*.py` (see `../run_config.json`). Same layout as
`../../v2_tf_specificity/task_data/`, but everything comes from the v3 SAE
(`outputs/v3_sae/layer_05/sae_final.pt`, 4,928 TopK features, k = 32, retrained on correctly encoded
inputs) applied to the INPUT of decoder block 5 (= `hidden_states[5]`), and from correctly encoded
cells (counts / Geneformer gene median, `setup/inputs_v3.py`). Feature numbers are v3 numbers; they
do not match deployed or v2 feature numbers. Gene symbols are upper case.

## Definitions

* **Gene universe**: the 6328 genes that appear as tokens in the 500 K562 non-targeting control
  cells used to train the v3 SAE (and to build its catalog), with the correct encoding.
* **Detection count**: number of those 500 cells in which the gene is a token (max_len 2,048). Max 500.
* **Top-20 genes of a feature** (deployed definition): over every token position of the 500 catalog
  cells, mean activation per (feature, gene) over positions where the feature is active; the 20 genes
  with the highest mean. No minimum count. `<SPECIAL>` (= `<bos>`/`<eos>`) can enter the list; it is
  never a target. Built from `outputs/v3_sae/codes/layer_05_topk.npz`; rebuilt from our own forward
  passes it is identical for all 4,928 features.
* **Per-cell feature value**: mean activation of the feature over the cell's gene tokens.
* **Responding feature** of a TF: two-sided Mann-Whitney U (full tie correction), knockdown cells vs
  400 reference control cells, per feature; BH across 4,928 features; q < 0.05 and
  |mean(knockdown) - mean(reference)| > cut-off (0.5 deployed; also 0.25, 0.1, 0.05, 0.02, 0.01, 0).
* **Count bins**: detection count digitised at 1, 2, 5, 10, 20, 50, 100, 200, 500. **Length tertile**:
  genomic span from `biotensor/data/genemanifold/gene_pos.json`, cut at the 33.3 / 66.7 percentiles of
  the universe (genes without a length go to the middle tertile). `count_x_length_bin` = 10 x bin + tertile.

## Files

| file | content |
|---|---|
| `gene_universe.tsv` | gene, detection_count, gene_length_bp, chr, count_bin, length_tertile, count_x_length_bin |
| `feature_top20.tsv` | feature_id, rank, gene, mean_act_when_active, n_active_positions, gene_detection_count, in_forward_rebuild_top20 |
| `feature_info.tsv` | per feature: n_top20, n_special_in_top20, median detection count / length of top-20, activation_frequency, n_active_positions, forward_rebuild_same_list |
| `targets_trrust.tsv`, `targets_dorothea.tsv` | targets inside the universe (DoRothEA with confidence and chip_flag = ChIP-seq subset) |
| `cell_manifest.csv` | 9,200 cells: row (0-based X row of replogle_concat.h5ad), group, tf, priority, barcode, gem_group, UMI_count, is_primary_tf (same cells as v2) |
| `cell_feature_means.npz` | rows, mean_gene (9,200 x 4,928 float32), n_tokens |
| `responding_features.tsv` | every (TF, feature) with BH q < 0.05: delta_mean, cohen_d, mwu_p, bh_q, n_kd_cells, resp_cut_<c> |
| `tf_summary.tsv` | 87 TFs: n_kd_cells, n_bh_sig, K_cut_<c>, target-set sizes, kd_remaining_frac_linear |
| `tf_tests.tsv` | all statistics and p-values, TF x cut-off x database (same as `../tf_results.csv`) |
| `catalog_check.json` | token check, code check (stored vs live codes), catalog rebuild check, universe vs v2 |

Column guide for `tf_tests.tsv`: as in `../../v2_tf_specificity/task_data/README.md` (K, M_obs, T_obs,
p_rf, p_cm, p_cml, *_uncapped, E_M_cm, p_fake, p_fakeK, p_okd, n_okd, rank_cm, n_rank_cand, q_<test>_primary,
q_<test>_all).
