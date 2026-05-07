# Longevity mechanistic interpretability: donor-aware detection and robustness testing

## 1. Overview

A model-agnostic, eleven-stage pipeline that asks a strictly harder question than "can we decode age from a frozen single-cell foundation model": *does the decodable signal reflect a donor-aware, composition-aware, cross-model-coherent biological aging program, and does a single positive intervention branch survive a progressive battery of confound stress tests?*

The pipeline takes any frozen transformer-based single-cell foundation model (the source paper uses scGPT at `scgpt_layer_09` and Geneformer's contextual layer, plus an exploratory MethylGPT extension for the DNA methylome), extracts per-cell contextual representations, and pushes them through:

1. donor-held-out linear probes + compact manifold diagnostics,
2. top-k / within-cell-type manifold-geometry robustness gates,
3. sparse-autoencoder training with donor-aware feature scoring (permutation-calibrated),
4. cross-model pathway matching against an 8-pathway aging geneset panel (inflammation/NF-κB, senescence/SASP, mTOR/IGF, autophagy, proteostasis/UPR, mitochondria/OXPHOS, DNA damage, interferon),
5. targeted SAE-latent interventions (`old_push`, `young_push`, `ablate`, `random_push`) with donor-bootstrap confidence intervals,
6. a multi-criterion **strict gate** that only promotes branches where expected-age direction, marginal delta-balanced-accuracy, and old-class probability shift *all* agree,
7. a donor-threshold sweep (tightens `min_donors` per comparison from 20 → 500),
8. donor-composition reweighting (positive control: ridge-regression reweighting at penalties 0.1, 1, 10, 100),
9. **fully composition-matched forward-pass reruns** (stronger control — rebuilds the sampled cell set to match cell-type proportions across age bins and re-runs the frozen model pass),
10. a multi-seed composition-matched panel (four seeds — 42, 101, 202, 303 — each a full re-run).

The design principle is explicit: *the goal is not to maximize nominal detection rate, but to suppress exactly the kinds of confounded positives common in atlas-scale analyses.* The pipeline treats every candidate aging claim as provisional until all promotion gates fire together, and records failure modes verbatim so downstream analyses can tell *model-specific interpretable signal* from *cross-model robust mechanism*.

The source-paper instantiation produced one positive cell-type-local result (AIDA phase 1 v1 CD14 monocytes; inflammation/NF-κB; scGPT old-minus-random contrast 0.0419 [0.0368, 0.0474], Geneformer 0.0357 [0.0314, 0.0398] at intervention scale 2.0), one globally strong Geneformer inflammation branch that *survived* donor-threshold tightening up to `min_donors=400` but did *not* replicate under any of four fully composition-matched reruns, and an exploratory MethylGPT branch whose age signal localized sharply to recurrent 512-CpG windows mapping onto Horvath, PhenoAge, Skin & Blood, DNAmTL, and DunedinPACE clock probe families.

## 2. Source

- **Paper:** , "Inflammation-Linked Aging Signals in Frozen Single-Cell Foundation Models: Donor-Aware Detection and Robustness Testing", 2026 — [`references/biogerontology_manuscript.pdf`](../references/biogerontology_manuscript.pdf)
- **Code repo:** [`repos/longevity-mechinterp/`](../repos/longevity-mechinterp/) @ commit `5a61464632a3c3e8bebd396eb1ab17bce1dc2493` (2026-04-01; "Polish manuscript for journal submission"). Public companion at [`Biodyn-AI/longevity-mechinterp`](https://github.com/Biodyn-AI/longevity-mechinterp).
- **Author project plan:** `repos/longevity-mechinterp/planning/research_plan.md` (working principles, decision gates G1–G4, fine-tuning policy — read first).
- **Repo README:** `repos/longevity-mechinterp/README.md`.
- **Stage output README:** `repos/longevity-mechinterp/implementation/outputs/README.md`.

## 3. Inputs

### 3.1 Target models

Any frozen single-cell / multi-omic foundation model that:
- exposes per-cell contextual embeddings (one vector per cell, or a pooled layer output),
- has a gene-symbol or probe-level tokenizer that can be mapped to HGNC for pathway annotation (Stage 4),
- allows identifying a specific interior layer (the source paper uses scGPT `scgpt_layer_09` and Geneformer's contextual layer; MaxToki would use e.g. `layers.5.output` for 217M or `layers.10.output` for 1B).

The pipeline is *representation-level*: it does not require attention, gradients, or weight access beyond the forward pass. Any model that can produce a `(n_cells, hidden_dim)` array on demand is eligible.

### 3.2 Datasets

Age-labeled human single-cell datasets with:
- numeric age per cell (or derivable via `_parse_numeric_age`, `_derive_age_labels` in `run_stage1_longevity_mechinterp.py:164,184`),
- **real donor IDs** (pseudoreplication is the dominant confound),
- cell-type annotations compatible with an atlas ontology (needed for composition matching),
- enough donors to support donor-held-out splits (the source paper's deep-dive ran at ≈424 donors; the strict gate fails below 30–50 donors).

Author's canonical dataset ranking script: `implementation/scripts/rank_age_labeled_datasets.py` (ranks candidates by age span, age-level count, donor diversity, cell-type diversity, total cells).

Source-paper datasets used:
| Dataset ID | Role | Notes |
|---|---|---|
| `aida_phase1_v1` | primary discovery (AIDA pilot) | strongest monocyte inflammation signal |
| `aida_phase1_v2` | deep-dive (strict gate branch) | ≈424 donors baseline; ≈333–340 after composition matching |
| `allen_aging_plasma_cells` | stress-test external | |
| `allen_immune_atlas` | stress-test external | |
| `yazar_donor_cohort` | external single-cell eQTL cohort | |
| `tabula_sapiens_immune` | reference prior | broad donor coverage; prototype set |
| `tabula_sapiens_immune_subset_20000` | fast prototype | |

Methylation extension (exploratory):
| Dataset | Role |
|---|---|
| AltumAge pan-tissue methylation compendium (de Lima Camillo et al. 2022) | age-labeled testbed for MethylGPT |

### 3.3 Pathway genesets (fixed panel, shipped in repo)

`run_stage4_cross_model_convergence.py:26` — `PATHWAY_GENESETS` dict with 8 aging-relevant pathways:

1. `inflammation_nfkb` — IL1B, IL6, TNF, NFKB1, NFKBIA, RELA, STAT1, STAT3, CXCL8, CCL2, LST1, S100A8, S100A9, PTGS2
2. `senescence_sasp` — CDKN1A, CDKN2A, TP53, GDF15, SERPINE1, MMP1/3/9, IL1A, IL1B, IL6, CXCL8, GLB1
3. `mtor_igf_akt` — MTOR, RPTOR, RICTOR, AKT1/2, PIK3CA/CD, TSC1/2, RHEB, RPS6KB1, EIF4EBP1, IGF1R
4. `autophagy_lysosome` — BECN1, ATG5/7, MAP1LC3B, SQSTM1, LAMP1/2, CTSB/D, TFEB, ULK1
5. `proteostasis_upr` — HSPA1A/B, HSP90AA1/AB1, HSPH1, DNAJB1, HSPD1/E1, ATF4, DDIT3, XBP1, UBB/UBC
6. `mitochondria_oxphos` — NDUFA1, NDUFS1, UQCRC1/2, COX4I1, COX5A, ATP5F1A/B, TFAM, SOD2, PPARGC1A
7. `dna_damage_repair` — BRCA1/2, RAD51, ATM, ATR, CHEK1/2, TP53BP1, XRCC5/6, PARP1, MRE11
8. `interferon_antiviral` — IFIT1/2/3, IFI6/27, ISG15, MX1, OAS1/2, STAT1, IRF7

## 4. Outputs

Per-run tree follows the existing repo layout under `implementation/outputs/` — each stage writes a timestamped directory plus a Markdown report and CSV/JSON artefacts. Mandatory artefacts by stage:

- **Stage 1** (`stage1_full_contextual_<date>/`): per-representation probe aggregate CSV with columns `dataset_id, representation, balanced_accuracy_mean/std, macro_f1_mean/std, n_valid_splits`, manifold diagnostics CSV (`pca_dim_used, participation_ratio, pc1_explained_ratio, pc1_age_corr, age_label_silhouette`), permutation-null CSV for the best representation per dataset, and a Markdown checkpoint report (`_write_checkpoint_report` at `run_stage1_longevity_mechinterp.py:1205`).
- **Stage 2** (`stage2_manifold_robustness_<date>/`): top-k age-aligned geometry table, within-cell-type control table, geometry-gate pass/fail per dataset.
- **Stage 3** (`stage3_sae_pilot_<model>_<date>_<tag>/<dataset_id>/`): `sae_artifacts.npz` (encoder W/b, decoder W/b, input mean/std, latent dim), `feature_scores.csv` with donor-aware columns `mean_activation, frac_active, cell_age_spearman, donor_age_spearman, donor_age_perm_p, age_eta2, celltype_eta2, donor_eta2, is_robust_feature, top_abs_genes, top_abs_corr_values`, `annotations.csv` (pathway hypergeom per robust feature), `sampled_obs.csv` (cell-level metadata used for SAE training).
- **Stage 4** (`stage4_cross_model_convergence_<date>/<dataset_id>/`): `annot_df.csv` (robust-feature pathway annotations for both models), `pair_df.csv` (cross-model pairs with `convergence_score`, `same_direction`, `gene_jaccard_top50`), `summary_df.csv` (per-pathway cross-model pair counts).
- **Stage 5** (`stage5_intervention_<scope>_<date>/`): `split_level_results.csv` (one row per (dataset, model, pathway, cell_type, seed, split, intervention)), `donor_level_results.csv` (donor-bootstrap source; same columns keyed by `donor_id`), bootstrap CI summary CSV (`summarize_stage5_donor_bootstrap.py`).
- **Stage 6** (`stage6_inflammation_followup_<date>/`): model-specific claim strict-gate audits, dose-response curves, cell-type anchor analyses, specificity checks.
- **Stage 7** (`stage7_transfer_*`): cross-dataset transfer check of the strongest branch.
- **Stage 8** (`stage8_contrast_ci_gate_audit_<date>/`, `stage8_monocyte_aida_{v1,v2}_scale_summary_<date>/`): contrast + marginal CI gate audits, dose-scale curves for the monocyte focus.
- **Stage 9** (`stage9_aida_v2_global_scale2_seedexpansion_<date>/`, `stage9_contrast_gate_aida_v2_seedexpansion_min{20,30,40,50,100,200,300,400,500}_<date>/`, `stage9_composition_reweighting_sensitivity_<date>.md`): donor-threshold seed-expansion results, donor-composition reweighting sensitivity at ridge penalties 0.1/1/10/100.
- **Stage 10** (`stage10_aida_v2_global_scale2_compmatched_forwardpass_<date>/` with `donor_bootstrap_ci/`): full composition-matched forward-pass rerun using `build_composition_matched_sampled_obs.py`, donor-bootstrap CI on the matched sample.
- **Stage 11** (`stage11_compmatched_seedpanel_summary_<date>/`): four-seed composition-matched panel (seeds 42, 101, 202, 303), strict-row count per seed.

Final publication-ready artefacts produced by `paper/build_paper_assets.py` consume only these summary files.

## 5. Dependencies

### 5.1 Python packages

From repo inspection:
- `torch` (SAE + model runtimes)
- `numpy`, `pandas`, `scipy` (`scipy.stats.hypergeom`, `scipy.stats.spearmanr`)
- `scikit-learn` (`GroupShuffleSplit`, `Pipeline`, `StandardScaler`, `LogisticRegression`, `PCA`, `silhouette_score`, `balanced_accuracy_score`, `f1_score`)
- `anndata`, `h5py` (h5ad I/O with backed reads for composition matching)
- `matplotlib` (only for figure generation in `paper/`)

### 5.2 Model runtimes

Source paper uses custom runtime classes implemented directly in Stage 1:
- `ScGPTRuntime` at `run_stage1_longevity_mechinterp.py:449` — wraps bowang-lab/scGPT, extracts per-layer outputs via `extract_representations(adata, batch_size, max_genes, layer_indices)` returning a dict `{scgpt_layer_{i}: np.ndarray}`.
- `GeneformerRuntime` at `run_stage1_longevity_mechinterp.py:597` — wraps Theodoris-lab/Geneformer, `extract_representation(adata, max_genes_per_cell, batch_size)` returning `(name, array)` for the contextual layer.

To port the pipeline to a new model (e.g. MaxToki), add a third runtime class with the same interface (`extract_representation(s)` returning a `{name: ndarray}` dict) and register it in `_build_representations` at `run_stage3_sae_pilot.py:92` and analogously in Stage 1's representation dispatch at `run_stage1_longevity_mechinterp.py:965` (`_evaluate_representations`).

### 5.3 Hardware

- GPU with ≥16 GB VRAM recommended for scGPT/Geneformer contextual passes on AIDA-scale datasets (hundreds of donors × thousands of cells per donor).
- CPU is fine for Stages 2–11 downstream analyses; Stage 3 SAE training is a single-layer ReLU autoencoder and runs in minutes.
- **Runtime constraint noted in paper §4.2:** incomplete higher-budget composition-matched reruns were bottlenecked by forward-pass runtime for the frozen models — budget Stage 10/11 with this in mind (seed panel is 4× Stage 1 forward-pass cost).

## 6. Methodology

### Stage 0 — Dataset ranking and lock-in

Run `implementation/scripts/rank_age_labeled_datasets.py` over available h5ad files. The ranker scores each dataset on: age span, number of distinct age bins, donor count, cell-type diversity, total cells. Lock the top-k for discovery + at least one independent external set.

Write the ranking to `implementation/outputs/age_dataset_ranking.{csv,md}` and commit. Every subsequent stage references the ranked list.

### Stage 1 — Frozen contextual probes + manifold diagnostics

Goal: answer gate **G1 (signal existence)** — does donor-held-out probe beat null?

1. For each dataset, load h5ad with `_load_subset_anndata` (`run_stage1_longevity_mechinterp.py:336`). Build a balanced donor subsample with `_balanced_subsample_by_donor` (`:240`), target ≈20k cells for prototype, full donor panel for deep-dive.
2. Derive age labels: `_parse_numeric_age` → `_derive_age_labels` (`:164, :184`) with `n_bins` = 3–5 and `min_per_label` ≥ 50. The default bins are quantile-balanced in age_numeric space.
3. Instantiate `ScGPTRuntime` + `GeneformerRuntime` (+ new-model runtime). Call `extract_representations` at requested `scgpt_layers` (default: layer 9 — `scgpt_layer_09`, matching the source paper) and Geneformer contextual layer.
4. Compute a baseline HVG-PCA representation via `_baseline_expression_representation(adata, n_components, seed)` at `:350` — 50-component PCA over log-normalized counts. This is the **trivial-baseline** that any model representation must beat.
5. For each representation, call `_evaluate_representations` (`:965`) which runs:
   - `_group_split_scores` (`:856`) — 5-fold `GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=seed)` grouped by `donor_id`, fitting `Pipeline(StandardScaler, LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs'))`, reporting `balanced_accuracy_mean/std` and `macro_f1_mean/std`.
   - `_manifold_metrics` (`:919`) — PCA to `min(32, n-1, d-1)`, computes `participation_ratio = (Σev)² / Σ(ev²)`, `pc1_explained_ratio`, `pc5_cumulative_explained_ratio`, `pc1_age_corr` (Pearson of PC1 vs age_numeric), `age_label_silhouette` (sklearn `silhouette_score` on first ≤10 PCs).
6. For the best representation per dataset, run a permutation null via `_permutation_null_for_best` (`:1019`) — shuffle age labels within donor groups and re-fit the probe `n_perm ≥ 100` times.
7. Emit Stage-1 aggregate table and checkpoint report. **Gate G1 passes** if donor-held-out balanced accuracy > permutation-null 95th percentile *and* > HVG-PCA baseline.

Source-paper Stage-1 outcome: best BA = 0.384 on AIDA phase 1 v2; range 0.299–0.384 across 5 datasets. Geometry gate (Stage 2) passed 0/5 — see §3.2 of the paper.

### Stage 2 — Manifold robustness gate

Goal: verify that a decodable signal also corresponds to a **stable latent geometry**, not merely a linear-probe decision boundary.

1. Load Stage-1 representations. Call `_topk_representation_geometry` (`run_stage2_manifold_robustness.py:37`) — compute top-k geometry metrics (`k` typically 10–32 PCs) for each representation across datasets.
2. Call `_within_celltype_expr_controls` (`:66`) — repeat the geometry analysis restricted to within-cell-type, testing whether age geometry survives cell-composition conditioning.
3. `_write_stage2_report` (`:147`) produces a per-dataset pass/fail on the geometry gate. **Passing criterion:** PC1–age correlation and age-label silhouette must both be non-trivial after within-cell-type restriction. In the source paper this gate was 0/5 — strong signal that apparent age geometry was driven by cell-composition shifts across age bins, which motivated the compositional stress tests in Stages 9–11.

### Stage 3 — SAE pilot with donor-aware feature scoring

Goal: find sparse latent features whose age association survives donor-aware controls.

1. Select datasets via `_select_stage3_datasets` (`run_stage3_sae_pilot.py:72`) — defaults to top-k from Stage 1's aggregate ranked by `best_balanced_accuracy_mean`.
2. `_build_representations` (`:92`) pulls the chosen contextual representations for each dataset (scGPT layer 9, Geneformer contextual).
3. Train a **minimal ReLU SAE** per (dataset × representation) via `_train_sae` (`:130`):
   - Architecture: `SparseAutoencoder(input_dim, latent_dim)` with `encoder = Linear`, `decoder = Linear`, forward `z = relu(encoder(x)); x_hat = decoder(z)` (`:43–54`).
   - Standardize: `Xn = (X - mean) / std` (column-wise, floor `std < 1e-6` to 1.0).
   - Optimizer: Adam, `lr` default 1e-3, `epochs` default 30, `batch_size` 1024.
   - Loss: `L = MSE(x_hat, xn) + l1_coef * mean(|z|)` with `l1_coef` default 1e-3 (L1 on *activations*, not weights — this is the sparsity penalty). Reported metrics: `loss_mean, mse_mean, l1_mean` per epoch.
   - Latent dim: source paper uses `latent_dim = 4 × input_dim` (standard "overcomplete" SAE ratio). Record `recon_mse` on the training set as a sanity check.
4. For each SAE feature, run `_donor_aware_feature_scores` (`:240`):
   - `frac_active = mean(z > 1e-6)`, `mean_activation`, `std_activation`.
   - `cell_age_spearman = spearman(z, age_numeric)` (cell-level; prone to donor leakage).
   - **Donor-level**: compute `donor_means[d] = mean(z[cells_in_donor_d])`, then `donor_age_spearman = spearman(donor_means, donor_age)`. This is the donor-aware score.
   - **Permutation-calibrated p**: if `n_donors ≥ 20` and `perm_iters > 0`, permute `donor_age` `perm_iters` (default 500–1000) times, recompute `spearman(donor_means, perm_age)`, report `donor_age_perm_p = (1 + #{|null| ≥ |obs|}) / (perm_iters + 1)`.
   - `age_eta2`, `celltype_eta2`, `donor_eta2` via `_eta_squared` (`:208`) — group-sum-of-squares / total-sum-of-squares for the feature's activation against each categorical factor. A feature that's high `donor_eta2` *and* `donor_age_spearman` is a donor-identity confound, not an aging feature.
5. **Robust-feature filter** (paper's "donor-aware sparse features"): keep features that simultaneously pass
   - `|donor_age_spearman| ≥ donor_corr_min` (default 0.3),
   - `donor_age_perm_p ≤ donor_p_max` (default 0.05),
   - `celltype_eta2 ≤ max_celltype_eta2` (default 0.5 — reject cell-type identity features),
   - `frac_active` in a reasonable band (e.g. 0.05–0.95 — reject always-on and always-off features).
   
   Mark `is_robust_feature = True` in `feature_scores.csv`.
6. Decoder-dimension annotation via `_annotate_decoder_dimensions` (`:310`) — pull top decoder dimensions per feature for quick inspection.
7. Write `sae_artifacts.npz` (all encoder/decoder weights + normalization), `feature_scores.csv`, `sampled_obs.csv`, and a Markdown Stage-3 report via `_write_stage3_report` (`:327`).

Source-paper Stage-3 outcome: **132 donor-aware robust features** retained after filtering — 91 in scGPT core runs, 41 in Geneformer core runs; 54 robust features in `aida_phase1_v1_scGPT`, 40 in `aida_phase1_v2_Geneformer`, 37 in `aida_phase1_v2_scGPT`, 1 in `aida_phase1_v1_Geneformer`. Most robust signal concentrates in the two AIDA cohorts (Figure 2).

### Stage 4 — Cross-model convergence via pathway matching

Goal: test whether scGPT and Geneformer independently discover the same biology.

1. `_annotate_robust_features` (`run_stage4_cross_model_convergence.py:399`):
   - For each robust feature (per model, per dataset), compute its correlation with every gene in `adata.X` (log1p-normalized), pick the **top-50 |corr| genes** (`top_genes_per_feature`), and call `_pathway_annotation_for_feature` (`:343`).
   - Pathway test: hypergeometric (`scipy.stats.hypergeom.sf`) over the 8-pathway `PATHWAY_GENESETS` (`:26`), universe = all genes in `adata.var` (or a common universe if crossing datasets). Assigns the pathway with smallest `p` and `overlap ≥ min_pathway_overlap` (default 2), below `pathway_p_threshold` (default 0.05). Otherwise `primary_pathway = "unassigned"`.
2. `_build_convergence_pairs` (`:487`):
   - Within each dataset, pair up (scGPT-feature, Geneformer-feature) **with the same primary pathway**.
   - Compute `gene_jaccard_top50 = |A ∩ B| / |A ∪ B|` on the top-50 gene lists.
   - `same_direction = sign(scgpt_donor_age_spearman) == sign(geneformer_donor_age_spearman)`.
   - `convergence_score = same_direction_bonus × gene_jaccard_top50 × min(|ρ_scgpt|, |ρ_geneformer|) × pathway_confidence`.
3. `_build_consensus_summary` (`:538`) groups pairs by `(dataset_id, pathway)` and reports per-pathway counts + best convergence score.
4. Write `annot_df.csv`, `pair_df.csv`, `summary_df.csv`, and a Markdown report (`_write_report` at `:566`).

Source-paper Stage-4 outcome: **193 cross-model pathway-matched pairs total** (Figure 3). Inflammation/NF-κB dominates (175 pairs in `aida_phase1_v2`, 16 in `aida_phase1_v1`; best `convergence_score = 0.0957`). Senescence/SASP: 2 pairs in `aida_phase1_v2`. Proteostasis/UPR and interferon/antiviral: 0 pairs. This convergence pattern is the *reason* the subsequent stages drill specifically into the inflammation branch.

### Stage 5 — Targeted interventions with donor-bootstrap CIs

Goal: move from association to *causal-adjacent* evidence by perturbing SAE features in latent space and observing probe-side effects.

1. For each (dataset × model × pathway × cell_type × intervention_scale), load `sae_artifacts.npz` and `feature_scores.csv`. Select the top robust features for the target pathway via `_select_targets` (`run_stage5_intervention_validation.py:637`), typically top-N by `|donor_age_spearman|` with `is_robust_feature=True`.
2. Compute SAE latents on the full representation: `_compute_latent` (`:230`) = `relu(Xn @ W_enc.T + b_enc)`.
3. For each of `split_seeds` (default 5 seeds) × `n_splits` `GroupShuffleSplit` folds grouped by `donor_id`:
   - Fit a fresh age-classification probe (`StandardScaler → LogisticRegression`, same as Stage 1) on the **representation** (not the latent).
   - For the held-out set, compute baseline: `expected_age_mean`, `old_class_prob_mean`, `balanced_accuracy`, `pathway_activation_mean`.
4. Apply four interventions, each by modifying the latent `z` and decoding back via `_decode_from_latent` (`:243`) = `z @ W_dec.T + b_dec`, then un-standardizing to representation space:
   - **`old_push`**: `z[:, feature_ids] += intervention_scale × feature_signs × max(latent_std[feature_ids], 1e-4)`, where `feature_signs = sign(donor_age_spearman)` per feature. Intervention scale: 1.0 and 2.0 in the source paper.
   - **`young_push`**: same magnitude, negated signs.
   - **`ablate`**: `z[:, feature_ids] = 0`.
   - **`random_push`**: pick `n_features` random latent dimensions (not in `feature_ids`) with random ±1 signs, apply the same mean step magnitude. This is the **control** condition that the strict gate compares against.
5. Decode each modified latent, predict on the held-out set with the frozen probe, record `delta_balanced_accuracy`, `delta_expected_age_mean`, `delta_old_prob_mean`, `delta_pathway_activation_mean`.
6. **Donor-bootstrap CI** (`summarize_stage5_donor_bootstrap.py`): resample donors with replacement `B ≥ 1000` times, recompute `delta_expected_age_mean` per bootstrap, report 95% percentile CI. **Never bootstrap cells** — that understates the variance because of pseudoreplication (see Zimmerman et al. 2021 and Hicks et al. 2018, cited in the paper).
7. Cross-model directionality table via `_cross_model_directionality` (`:537`): for each (dataset, pathway, cell_type), does the scGPT and Geneformer `old_push` contrast point in the same direction?

Source-paper Stage-5 outcome, AIDA phase 1 v1 CD14 monocytes at intervention scale 2.0, inflammation/NF-κB:
- Geneformer old-minus-random expected-age contrast: 0.0357 [0.0314, 0.0398], positive.
- scGPT old-minus-random expected-age contrast: 0.0419 [0.0368, 0.0474], positive.

### Stage 6 — Model-specific claim promotion + strict gate

`summarize_stage6_model_specific_claims.py`, `summarize_stage6_dose_response.py`, `summarize_stage6_celltype_anchor.py`, `summarize_stage6_specificity.py`. Each candidate claim is audited against a **strict gate** with three independent criteria that must **all** fire:

1. **Expected-age contrast strict:** `old_push - random_push` 95% donor-bootstrap CI excludes zero in the expected (positive) direction; `young_push - random_push` CI excludes zero in the negative direction; `old_push - young_push` CI excludes zero in the positive direction.
2. **Marginal strict:** `delta_balanced_accuracy` under `old_push` vs `random_push` is positive with CI excluding zero.
3. **Old-probability-shift strict:** `delta_old_prob_mean` under `old_push` is positive; under `young_push` is negative.

All three must agree for a claim to be promoted. Failure of any one drops the claim to "exploratory".

**Author's promotion policy (paper §2.6, verbatim):** "Claims were promoted only when multiple directional checks agreed: expected-age contrasts had to point in the correct direction, marginal effects had to remain supportive, and old-probability shifts had to be consistent. This policy was intentionally stricter than contrast-only screening because the project goal was mechanism-grade evidence rather than loose signal detection."

Source-paper Stage-6 outcome: **0/4 strict claim passes** across model-specific promotions (Table 1, Block 6).

### Stage 7 — Transfer check

`summarize_stage7_transfer_check.py`. Does the surviving branch (from Stage 6's strict gate) transfer cross-dataset to Allen aging, Yazar, etc.? This is cheap and sharp — a claim that holds in one cohort but not any other is reported but never promoted.

### Stage 8 — Contrast + marginal strict gate audit + monocyte scale focus

- `summarize_stage8_contrast_ci_gates.py` — audits contrast + marginal CI gates across the full (dataset, model, pathway, cell_type, scale) grid. Source paper reports **1/19 runs with at least one full strict row**.
- `summarize_stage8_monocyte_scale_focus.py` — dose-response focus for AIDA v1 / AIDA v2 CD14 monocyte inflammation: scales {1.0, 2.0}, both models, donor-bootstrap CIs, saved as `stage8_monocyte_aida_{v1,v2}_scale_summary_<date>/`. Shows the signal is clean in AIDA v1 and weak/discordant in AIDA v2 (Figure 4).

### Stage 9 — Donor-threshold tightening + composition reweighting

Two sub-stages, both applied to the **single strongest surviving branch** (AIDA v2 Geneformer global inflammation at scale 2.0 in the source paper):

#### 9a. Donor-threshold seed expansion

Run `run_stage34_multiseed_hardening.py` (expands to 60 splits per model via multi-seed replication; `:62,105` build per-model Stage-3 commands, `:156` builds the Stage-4 cmd) at a grid of `min_donors ∈ {20, 30, 40, 50, 100, 200, 300, 400, 500}`. Emit one directory per threshold: `stage9_contrast_gate_aida_v2_seedexpansion_min{T}_<date>/`.

For each threshold, rerun the full strict gate (contrast + marginal + old-prob). Count `n_full_strict` rows. Source-paper outcome: the Geneformer AIDA v2 branch stays **full-strict positive up to `min_donors=400`** and fails at 500 because donor coverage becomes too sparse (Figure 5).

Deep-dive contrasts for the surviving branch:
- old-minus-random: 0.1494 [0.1423, 0.1564], positive.
- young-minus-random: −0.1324 [−0.1379, −0.1267], negative.
- old-minus-young: 0.2817 [0.2698, 0.2940], positive.

#### 9b. Donor-composition reweighting (positive control)

`summarize_stage9_composition_reweighting.py` — reweights donors by a ridge-regression propensity matching the age distribution across composition classes. Sweep `ridge_penalty ∈ {0.1, 1, 10, 100}`, rerun Stage 5 with donor weights. **Source-paper outcome:** the reweighting produces near-zero shifts in effect estimates and preserves the strict pass at every penalty (Figure 6 left). This is a positive control — it confirms that simple donor-level reweighting does *not* break the signal.

### Stage 10 — Fully composition-matched forward-pass rerun (strongest control)

This is the decisive stage. Instead of reweighting donors *post hoc*, this stage **rebuilds the sampled cell set** so that cell-type proportions are matched across age bins, and **re-runs the frozen model forward pass** on the matched sample.

1. `build_composition_matched_sampled_obs.py:14` defines `_allocate_with_caps(target_n, proportions, availability, rng)` — greedy cell-count allocator that respects per-(age_bin × cell_type) availability:
   - Compute target cell count per bin: `n_per_bin = [target_n // n_bins] × n_bins`, distribute remainder.
   - Target composition: `p_target = mean over age bins of (counts_per_bin[:, celltype] / row_sum)`. This is a pooled proportion vector that no single bin can trivially satisfy without drawing from all bins.
   - For each bin, allocate cells to cell types via `_allocate_with_caps` — floor `p_target × target_n`, distribute remaining budget by largest fractional residual where capacity remains, fallback to random picks where spare capacity exists.
   - **Donor-debiased sampling within each (bin, cell_type) stratum:** weight each candidate cell by `1 / donor_count`, sample without replacement using those weights. This suppresses donor-level duplication.
2. Write `sampled_obs.csv`, `<stem>_target_counts.csv`, `<stem>_achieved_counts.csv`, `<stem>_diagnostics.csv` (per-bin L1 proportion distance to target), `<stem>_run_config.json`.
3. Re-run Stage 3 (`run_stage3_sae_pilot.py`) with the matched `sampled_obs.csv`, then Stage 4 (cross-model convergence), Stage 5 (intervention), and the strict gate. Donor-bootstrap CI is recomputed on the matched sample. Outputs: `stage10_aida_v2_global_scale2_compmatched_forwardpass_<date>/donor_bootstrap_ci/`.

**Source-paper Stage-10 outcome:** the baseline strict claim **disappears** (`n_full_strict = 0`) even though one expected-age-contrast-strict row remains (Figure 6 right; Figure 7 compares donor-bootstrap intervention intervals before vs after full composition matching — aging-program push remains positive but the overall directional structure becomes less clean).

### Stage 11 — Composition-matched multi-seed panel

`stage11_compmatched_seedpanel_summary_<date>/`. Rerun Stage 10 end-to-end with **four independent seeds** (42, 101, 202, 303; ≈333–340 donors each). For each seed: rebuild the matched sampled_obs.csv, rerun the frozen forward pass, retrain the SAE, rerun cross-model + interventions, retest the strict gate.

Record for each seed: `n_full_strict`, directional pattern per condition (`aging_push`, `random_control`, `young_push`), Geneformer directional-pattern-retained y/n.

**Source-paper Stage-11 outcome (Table 2):**

| Run | `n_full_strict` | Aging push | Random control | Young push |
|---|---|---|---|---|
| Baseline expanded-split | 1 | positive | negative | negative |
| Comp-matched seed 42 | 0 | positive | positive | uncertain |
| Comp-matched seed 101 | 0 | uncertain | positive | positive |
| Comp-matched seed 202 | 0 | positive | positive | positive |
| Comp-matched seed 303 | 0 | positive | uncertain | uncertain |

- **0/4 full strict replications.**
- Geneformer directional-pattern criterion retained in **1/4** runs.
- Random/young intervention directional structure is **unstable across seeds** — the strongest possible signature that the baseline pattern was a sampling-realization artefact, not a conserved biological mechanism.

### Stage 12 (optional, exploratory) — MethylGPT CpG-window extension

A cross-modality triangulation; explicitly *not* a replication of the single-cell claim. The paper runs this branch because the aging substrate is stronger in DNA methylation (established clock literature) and uses it to show the same interpretability logic transfers.

**Key constraint:** MethylGPT's checkpoint expects full-length inputs of 49,156 CpG tokens. Faithful full-length inference is not feasible on commodity hardware because of fast-attention runtime. **Therefore the branch uses an explicitly approximate local-window procedure** — clearly labeled in every report.

Procedure:

1. **Local 512-CpG windows**, four layouts for robustness:
   - Evenly spaced (no overlap).
   - Half-shifted (offset by 256).
   - Random layout, seed 101.
   - Random layout, seed 202.
2. Embed each window independently → mean-pool across windows → one sample-level embedding per donor. Train a linear age probe on sample-level embeddings with donor-held-out splits. Report test Pearson per layout.
3. **Layout robustness test** (Figure 8A): source paper found test Pearson r = 0.843 / 0.826 / 0.850 / 0.845 across the four layouts — *materially stronger* than the base checkpoint and *stable* under layout perturbation, which rules out a trivial positional artefact.
4. **Recurrent high-signal region identification** (Figure 8B): rank 512-CpG windows by absolute age regression coefficient, keep windows that recur in the top-k across all four layouts. Source paper recovered four recurrent regions in canonical probe order: **45888–46400, 38510–39022, 11534–12046, 8844–9561**.
5. **Clock-family overlap annotation**: cross-reference recurrent regions against public clock probe sets — Horvath pan-tissue 2013, Horvath Skin & Blood 2018, Levine PhenoAge 2018, McEwen PedBE 2020, Lu DNAmTL 2019b, Belsky DunedinPACE 2022. Use Illumina EPIC annotation for genomic context.
   - Source-paper finding: region 45888–46400 contains 8 Horvath pan-tissue, 4 PhenoAge, 3 Skin & Blood probes. Region 8844–9561 overlaps all eight clock families considered.
6. **Fine 128-CpG probing within recurrent regions** (Figure 9A): slice each recurrent window into 128-CpG subwindows, retrain the linear probe. Source paper: subwindows 45888–46016 (test r = 0.819), 38510–38638 (0.802), and 11918–12046 (0.792) — sharpened localization onto Horvath Skin & Blood, Horvath pan-tissue, and DNAmTL respectively; the third primarily overlaps PhenoAge.
7. **Sparse top-k selection curve** (Figure 9B): rank 128-CpG fine subwindows by validation Pearson, compute the test Pearson of the cumulative top-k aggregate. Source paper: top-1 fine subwindow alone reaches r = 0.767 (97.5% of the full 32-subwindow aggregate); top-2 windows (46144–46272 + 11918–12046) reach 0.799, *slightly exceeding* the full aggregate (0.787). **Interpretation:** a very small sparse subset of subwindows carries almost all of the recoverable signal.
8. Annotate the top selected windows' genomic context — source paper finds mappings to promoter / first-exon contexts near **TLX3, CELSR1, HTR7, TENC1, LAD1**, frequently within CpG islands or shores.

**Report this branch as support, not replication.** The approximate local-window inference is not a faithful full-length forward pass, and the paper explicitly frames the result that way.

## 7. Code references

All paths relative to `repos/longevity-mechinterp/`.

| Stage | Primary script | Key functions |
|---|---|---|
| 1 | `implementation/scripts/run_stage1_longevity_mechinterp.py` | `ScGPTRuntime` (`:449`), `GeneformerRuntime` (`:597`), `_group_split_scores` (`:856`), `_manifold_metrics` (`:919`), `_evaluate_representations` (`:965`), `_permutation_null_for_best` (`:1019`), `_balanced_subsample_by_donor` (`:240`), `_derive_age_labels` (`:184`) |
| 2 | `implementation/scripts/run_stage2_manifold_robustness.py` | `_topk_representation_geometry` (`:37`), `_within_celltype_expr_controls` (`:66`), `_write_stage2_report` (`:147`) |
| 3 | `implementation/scripts/run_stage3_sae_pilot.py` | `SparseAutoencoder` (`:43`), `_train_sae` (`:130`), `_donor_aware_feature_scores` (`:240`), `_eta_squared` (`:208`), `_safe_spearman` (`:228`), `_annotate_decoder_dimensions` (`:310`) |
| 4 | `implementation/scripts/run_stage4_cross_model_convergence.py` | `PATHWAY_GENESETS` (`:26`), `_pathway_annotation_for_feature` (`:343`), `_annotate_robust_features` (`:399`), `_build_convergence_pairs` (`:487`), `_build_consensus_summary` (`:538`), `_compute_latent` (`:309`) |
| 5 | `implementation/scripts/run_stage5_intervention_validation.py` | `_run_probe_interventions` (`:282`), `_expected_age_and_old_prob` (`:255`), `_decode_from_latent` (`:243`), `_cross_model_directionality` (`:537`), `_select_targets` (`:637`), `_iter_celltype_strata` (`:661`) |
| 5 post | `implementation/scripts/summarize_stage5_donor_bootstrap.py`, `summarize_stage5_multiseed.py` | donor-level bootstrap CI summaries |
| 6 | `implementation/scripts/summarize_stage6_{model_specific_claims,dose_response,celltype_anchor,specificity}.py` | strict-gate audit variants |
| 7 | `implementation/scripts/summarize_stage7_transfer_check.py` | cross-dataset transfer |
| 8 | `implementation/scripts/summarize_stage8_{contrast_ci_gates,monocyte_scale_focus}.py` | contrast+marginal gate audit + monocyte scale focus |
| 9 | `implementation/scripts/run_stage34_multiseed_hardening.py`, `summarize_stage9_composition_reweighting.py` | multi-seed donor threshold sweep, ridge reweighting |
| 10 | `implementation/scripts/build_composition_matched_sampled_obs.py` | `_allocate_with_caps` (`:14`) |
| 11 | (orchestration via Stage-10 re-run with `--seed {42,101,202,303}`) | seed panel |

**Critical snippet — donor-aware SAE feature score** (`run_stage3_sae_pilot.py:240`):

```python
# donor_means[d] = mean activation of feature z across cells from donor d
donor_means = np.zeros(donor_ids.shape[0])
counts = np.zeros(donor_ids.shape[0], dtype=np.int64)
for i, d_idx in enumerate(donor_index):
    donor_means[d_idx] += z[i]; counts[d_idx] += 1
donor_means[counts > 0] /= counts[counts > 0]
donor_age_spear = _safe_spearman(donor_means, donor_age_arr)

# Permutation-calibrated p-value by shuffling donor-level age
if math.isfinite(donor_age_spear) and donor_ids.size >= 20 and perm_iters > 0:
    null_scores = np.zeros(perm_iters)
    for i in range(perm_iters):
        perm_age = rng.permutation(donor_age_arr)
        null_scores[i] = _safe_spearman(donor_means, perm_age)
    donor_age_perm_p = (1 + np.sum(np.abs(null_scores) >= abs(donor_age_spear))) / (perm_iters + 1)
```

**Critical snippet — intervention in SAE latent space** (`run_stage5_intervention_validation.py:380-408`):

```python
feature_steps = intervention_scale * np.maximum(latent_std[feature_ids], 1e-4)

interventions = [
    ("old_push",    feature_ids,  feature_signs, False),
    ("young_push",  feature_ids, -feature_signs, False),
    ("ablate",      feature_ids,  feature_signs, True),
]
if random_ids.size > 0:
    random_signs = rng.choice([-1.0, 1.0], size=random_ids.size)
    interventions.append(("random_push", random_ids, random_signs, False))

for name, target_ids, target_signs, do_ablate in interventions:
    latent_mod = latent_test.copy()
    if do_ablate:
        latent_mod[:, target_ids] = 0.0
    else:
        latent_mod[:, target_ids] += feature_steps.reshape(1, -1) * target_signs.reshape(1, -1)
    X_mod = _decode_from_latent(latent_mod, artifacts)  # z → x via W_dec
    mod_pred = probe.predict(X_mod)
```

## 8. Parameters

| Name | Stage | Default | Valid range / notes |
|---|---|---|---|
| `scgpt_layers` | 1 | `[9]` | any subset of 0–11; paper uses layer 9 |
| `scgpt_max_genes` | 1 | 1200 | ≤ model context window |
| `geneformer_max_genes` | 1 | 2048 | Geneformer hard limit |
| `n_splits` | 1, 5 | 5 | 5–10 donor-held-out folds |
| `test_size` | 1, 5 | 0.25 | 0.2–0.3 |
| `n_bins` (age) | 1 | 3 | 3–5; paper uses 3 |
| `min_per_label` | 1 | 50 | ≥ donor count / n_bins |
| `n_perm` (stage 1 null) | 1 | 100 | ≥ 100 |
| `top_k_pcs` | 2 | 10–32 | `min(32, n-1, d-1)` |
| `latent_dim` (SAE) | 3 | `4 × input_dim` | 2× to 8× |
| `epochs` (SAE) | 3 | 30 | 20–50 |
| `batch_size` (SAE) | 3 | 1024 | 256–4096 |
| `lr` (SAE) | 3 | 1e-3 | 1e-4 to 3e-3 |
| `l1_coef` (SAE sparsity) | 3 | 1e-3 | 1e-4 to 1e-2; watch recon MSE vs `frac_active` tradeoff |
| `donor_corr_min` | 3 | 0.3 | 0.2–0.5 |
| `donor_p_max` | 3 | 0.05 | 0.01–0.1 |
| `max_celltype_eta2` | 3 | 0.5 | 0.3–0.7 |
| `perm_iters` (donor permutation) | 3 | 500 | ≥ 500 for `p < 0.01` resolution |
| `top_genes_per_feature` | 4 | 50 | 30–100 |
| `pathway_p_threshold` | 4 | 0.05 | 0.01–0.1 |
| `min_pathway_overlap` | 4 | 2 | 2–4 |
| `intervention_scale` | 5 | 2.0 | 1.0 and 2.0 in paper; 0.5–3.0 for dose-response |
| `split_seeds` | 5, 9 | `[42, 7, 13, 21, 99]` | ≥ 5 seeds; 60 seeds in stage 9 deep-dive |
| `donor_bootstrap_B` | 5 | 1000 | ≥ 1000 |
| `min_donors` | 9 | grid | `{20, 30, 40, 50, 100, 200, 300, 400, 500}` |
| `ridge_penalty` | 9 | `{0.1, 1, 10, 100}` | positive-control sweep |
| `target_n_cells` | 10 | 700 | should be ≥ donor count × 2 |
| `top_celltypes` | 10 | 16 | 8–32 |
| `min_count_per_bin` | 10 | 1 | 1–3 |
| `seed` | 10, 11 | 42 | + panel {101, 202, 303} |

## 9. Validation

**Intra-stage sanity:**

- Stage 1: balanced accuracy > 0.5 on a 3-bin age task is above chance; source paper found 0.30–0.38. Permutation null CI must exclude the observed value.
- Stage 2: geometry gate explicitly allowed to fail — a 0/5 result is still informative (motivates Stages 9–11).
- Stage 3: `recon_mse` should stabilize by epoch 20; `frac_active` should be between 0.05–0.5 per feature (too low → dead features, too high → dense). `donor_age_perm_p` must be conservative — if all features have `p ≈ 0.001`, the permutation isn't doing work; inspect `donor_age_arr` for degeneracies.
- Stage 4: report the distribution of `primary_pathway` across all features — if >80% land in one pathway, the pathway genesets are not diverse enough for this representation.
- Stage 5: donor-bootstrap CIs should be asymmetric when signal is marginal. Symmetric CIs with narrow width on a tiny effect often indicate cell-level bootstrap was used by mistake.
- Stage 9a: the donor-threshold curve should be **monotonic in sparsity** (stricter threshold → fewer passing rows, or at worst constant). Non-monotonicity indicates a seed-specific accident.
- Stage 10: check `<stem>_diagnostics.csv` — `l1_prop_distance_to_target` should be < 0.05 per bin. Larger L1 means the matching wasn't achieved; report the achieved proportion vector alongside the result.
- Stage 11: explicitly expect some disagreement across seeds. Full agreement (4/4 positive) under composition matching is suspicious — check whether the seeds are actually independent (different donor subsamples, different SAE initializations).

**Cross-stage coherence checks:**

- A claim that passes Stage 8 strict gate but fails Stage 9b reweighting is **donor-imbalance-driven**.
- A claim that passes Stage 9b but fails Stage 10 is **cell-composition-driven** (the reweighting was too coarse to capture the composition shift).
- A claim that passes Stage 10 in one seed but fails in 3/4 across Stage 11 is **sampling-realization-driven**.
- **A claim that survives all of 8 → 9a → 9b → 10 → 11 is a mechanism-grade candidate.** In the source paper, no claim reached this bar.

**Negative-control expectations:**

- `random_push` should produce `delta_expected_age_mean` CIs that include zero. If `random_push` is systematically non-zero, the probe has memorized donor identity → reduce `test_size` and re-check `GroupShuffleSplit` grouping.
- The `ablate` intervention should weakly mimic `old_push` for features with positive `donor_age_spearman` (setting a positive-direction feature to zero moves the cell toward younger). Disagreement means the SAE decoder is recovering something other than the direct feature signal.

## 10. Known pitfalls

1. **Cell-level vs donor-level bootstrap.** Bootstrapping cells inflates sample size and shrinks CIs by factor √(n_cells/n_donors). Always resample at donor level. This is the single most common error in single-cell aging analyses (Zimmerman et al. 2021, Hicks et al. 2018).
2. **Cell-composition shifts masquerading as age signal.** Older donors often have different cell-type distributions (more memory T cells, fewer naive B cells, etc.). A linear probe will happily decode this as "age" unless you force composition matching via Stage 10 or within-cell-type restriction in Stage 2.
3. **Donor leakage via batch effects.** If `donor_id` correlates with sequencing batch, `GroupShuffleSplit` by donor *does not* remove batch leakage. Check `batch_eta2` alongside `donor_eta2` in Stage 3; if they track each other, stratify splits on batch too.
4. **SAE hyperparameter hypersensitivity.** `l1_coef` and `latent_dim` dominate the feature landscape. Always report `frac_active` and `recon_mse` alongside `n_robust_features`. Multi-seed SAEs (Stage 9 hardening expands to 60 split seeds per model) are mandatory for any headline claim.
5. **Pathway geneset curator bias.** The 8 pathways are hand-curated; they encode a strong prior that aging biology lives in inflammation + senescence + mTOR + autophagy + proteostasis + OXPHOS + DNA damage + IFN. Any robust feature that doesn't fit these will be `unassigned`. For new biology you will need to add genesets — do so *before* running Stage 4, never after observing the robust-feature top-gene lists (that's post-hoc fitting).
6. **Permutation calibration fails at low donor counts.** `_donor_aware_feature_scores` skips permutation when `n_donors < 20`. Features in such strata will have `donor_age_perm_p = NaN` and silently fail the robust-feature filter. Check `n_donors` per stratum before trusting `is_robust_feature=False`.
7. **Composition matching is slow.** Stage 10/11 rebuild the sample and rerun the frozen forward pass. Budget accordingly; the source paper flags incomplete higher-budget composition-matched reruns due to runtime. Cache representations at the cell-id level so matched samples can be reconstructed without re-running the forward pass.
8. **MethylGPT branch is not a replication.** The paper is explicit: local-window inference is approximate, this branch is triangulation only. Never pool MethylGPT and scGPT/Geneformer evidence into a single composite claim.
9. **Zero mechanism-grade claims is a valid result.** The source paper's headline finding is that the strongest candidate *does not* survive full composition matching — reported as "promising mechanistic signal, but not yet a robust mechanism-level claim". The pipeline's success criterion is honesty under stress, not positive detection. Treat a clean negative result as a win.

## 11. Porting to a new target model

To apply this pipeline to a new single-cell foundation model (e.g. MaxToki, under `projects/maxtoki/`):

1. Add a new runtime class alongside `ScGPTRuntime` / `GeneformerRuntime` in `run_stage1_longevity_mechinterp.py`. Interface: `extract_representation(adata, max_genes_per_cell, batch_size, layer_indices=None) → dict[name → ndarray(n_cells, hidden_dim)]`. For MaxToki specifically, load the HF safetensors variant, enable `output_hidden_states=True`, and expose a layer index CLI flag matching the convention `<model>_layer_<NN>`.
2. Register the new representation prefix in `_build_representations` at `run_stage3_sae_pilot.py:92,104,118` so `rep_keep` can select it.
3. Ensure the model's tokenizer can be mapped back to HGNC gene symbols for Stage 4 pathway annotation. If the vocabulary is gene symbols directly (as in MaxToki's 20,275-token vocab), use `adata.var_names` intersected with the vocab. Otherwise provide a `token_id → HGNC` mapping.
4. Run the pipeline at small scale first: `tabula_sapiens_immune_subset_20000.h5ad` → Stages 1–5 only. If Stage 1 balanced accuracy is < 0.5 + 0.05 above permutation null, stop — the model may not encode age at all, and the rest of the pipeline is moot.
5. Only after Stages 1–5 show a cross-model agreement with scGPT/Geneformer at the pathway level should you run the full deep-dive (Stages 6–11) on AIDA v1/v2.
6. The strict gates in Stages 6, 8, 9 use **the same thresholds** for every model. Don't tune them to the new model — that's circular.
