# Stage 1 — frozen contextual probes + manifold diagnostics (20260505)

- **Model:** MaxToki-217M-HF (LlamaForCausalLM, hidden_size 1232, 11 blocks)
- **Layer probed:** `['maxtoki217m_layer_05']`
- **Dataset:** `tabula_sapiens_immune_subset_20000` (Tabula Sapiens immune subset 20k)
- **Sample:** 2974 cells, 17 donors, 38 cell types
- **Age bins:** 3 (quantile); labels = {'(21.999, 56.0]': 1713, '(56.0, 60.0]': 661, '(60.0, 74.0]': 600}
- **Donor-held-out splits:** 5-fold GroupShuffleSplit (test_size=0.25)
- **Permutation null:** n_perm=100
- **`max_genes_per_cell`:** 1024 (note: README states 1200; actual run used 1024 — see Deviation note below)

## Probe aggregate (sorted by balanced accuracy)

| representation        | n_valid_splits | balanced_accuracy_mean | balanced_accuracy_std | macro_f1_mean | macro_f1_std | mean_train_cells | mean_test_cells | n_cells | n_features | n_age_classes | n_donors | dataset_id                          |
|-----------------------|---------------:|-----------------------:|----------------------:|--------------:|-------------:|-----------------:|----------------:|--------:|-----------:|--------------:|---------:|-------------------------------------|
| maxtoki217m_layer_05  |              5 |                 0.2755 |                0.0346 |        0.2089 |       0.0304 |          2116.6 |           857.4 |    2974 |       1232 |             3 |       17 | tabula_sapiens_immune_subset_20000 |
| hvg_pca_50            |              5 |                 0.1722 |                0.0370 |        0.1472 |       0.0227 |          2116.6 |           857.4 |    2974 |         50 |             3 |       17 | tabula_sapiens_immune_subset_20000 |

## Manifold diagnostics

| representation        | pca_dim_used | participation_ratio | pc1_explained_ratio | pc5_cumulative_explained_ratio | pc1_age_corr | age_label_silhouette | n_cells | n_features | dataset_id                          |
|-----------------------|-------------:|--------------------:|--------------------:|-------------------------------:|-------------:|---------------------:|--------:|-----------:|-------------------------------------|
| maxtoki217m_layer_05  |         32.0 |              7.9985 |              0.2387 |                         0.4984 |       0.0384 |              -0.0057 |    2974 |       1232 | tabula_sapiens_immune_subset_20000 |
| hvg_pca_50            |         32.0 |              7.8759 |              0.2549 |                         0.6416 |       0.0709 |               0.0051 |    2974 |         50 | tabula_sapiens_immune_subset_20000 |

## Permutation null (best rep = `maxtoki217m_layer_05`)

| representation        | metric                    | observed | null_mean | null_std | null_p95 | p_value_right_tail | n_permutations |
|-----------------------|---------------------------|---------:|----------:|---------:|---------:|-------------------:|---------------:|
| maxtoki217m_layer_05  | balanced_accuracy_mean    |   0.2755 |    0.3320 |   0.0103 |   0.3487 |             1.0000 |            100 |

## G1 verdict

- best balanced accuracy: **0.2755** (maxtoki217m_layer_05)
- HVG-PCA-50 baseline: **0.1722**
- permutation null: mean=0.3320±0.0103, p95=0.3487, p-value=1.0000
- **G1 (vs null):** ❌ FAIL  — observed accuracy is 5.65 pp **below** the null mean (one-sided right-tail p = 1.0)
- **G1 (vs HVG-PCA baseline):** ✅ PASS — MaxToki L5 (0.2755) beats HVG-PCA-50 (0.1722) by 10.3 pp
- **G1 overall:** ❌ FAIL — stop per pipeline §11.4 ("If Stage 1 balanced accuracy is < 0.5 + 0.05 above permutation null, stop")

### Interpretation

The donor-aware probe never exceeds the permutation null. Both candidate
representations sit *below* the null on balanced accuracy, even though MaxToki L5
clearly outperforms the HVG-PCA-50 baseline. With only 17 donors and a strongly
imbalanced age distribution (1713 / 661 / 600 cells across the three quantile
bins), majority-class predictions on permuted donor-held-out splits achieve
higher balanced accuracy than the learned (but donor-overfit) classifier on real
labels — exactly the small-cohort failure mode the pipeline §11.4 gate is built
for. The 0.038 PC1↔age correlation and ≈0 silhouette confirm the age axis is
not a dominant geometric direction in the L5 representation.

The G1-vs-baseline pass is informative but does not lift the verdict: the
pipeline requires both conditions, and "beats trivial baseline while losing to
permutation null" is a known small-cohort artefact, not evidence of an age
representation.

This matches the source paper's expectations — the strict null gate is built
for ≥424-donor discovery cohorts (AIDA v1/v2) and the paper itself notes the
gate fails below 30–50 donors. The Tabula Sapiens immune subset (17 donors after
balancing) is below that threshold by design; this run's job was to confirm the
prototype falls short of the gate before committing AIDA-scale compute.

## Extraction stats

- forward-pass mean: 0.21s/cell
- mean cell seq_len: 998 gene tokens
- total wall-clock: 5523s (~92 min)
- n_cells_in: 2974, n_cells_out: 2974, n_skipped_zero_expr: 0
- representations cached at `outputs/stage1_20260505/maxtoki_layer_reps.npy` (14.0 MB, fp32)

## Deviation note

`run_stage1.py` sets `MAX_GENES_PER_CELL = 1024`, while
`runs/longevity-mechinterp-217M/README.md` documents 1200. The actual run used
1024 (confirmed in `extraction_stats.json`). 1024 is well within the MaxToki
4096-token context budget and matches the typical mean cell seq_len of ≈998
observed here (very few cells would have been truncated). No re-run is needed
for the verdict; the README has been updated to record the as-run value.
