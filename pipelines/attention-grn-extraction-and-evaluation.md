# Pipeline: Attention-Derived Gene Regulatory Network Extraction and Evaluation for Single-Cell Foundation Models

## 1. Overview

This pipeline extracts a putative gene regulatory network (GRN) from the attention patterns of a single-cell transcriptomic foundation model, then subjects that GRN to a systematic evaluation battery that determines whether the attention-derived edges actually carry regulatory signal beyond trivial gene-level confounds, technical artefacts, and co-expression. It is **model-agnostic**: the attention-extraction step is an adapter that must be implemented per model; the rest of the pipeline consumes standard edge matrices and gene-level features.

The battery comprises **37 interlocking analyses yielding 153 statistical tests** (95 confirmatory, BH-corrected at α=0.05 framework-wide; 63/95 = 66% remain significant after correction). Tests are grouped into five primary families — (1) trivial-baseline comparison, (2) conditional incremental-value testing, (3) expression residualisation + propensity matching, (4) causal ablation with fidelity diagnostics, (5) cross-context replication — plus **CSSI** (Cell-State Stratified Interpretability, the constructive remedy for an attention-specific scaling failure), **six-database biological characterisation** (the refined claim: attention captures biologically structured signal with layer-specific hierarchy, but it is redundant with gene-level features for perturbation prediction), **value-weighted edge extraction** (ruling out that regulatory signal hides in the value/FFN pathway), **per-TF exploratory characterisation**, and a nine-part boundary-condition audit (detectability theory, cross-tissue consistency, ortholog transfer, pseudotime, batch/donor leakage, calibration, TRRUST circularity, HVG protocol, condition-specific mediation).

**The core refined finding (and the principle this pipeline is designed around).** For scGPT and Geneformer, attention patterns *do* encode structured biological information with clear layer-specific organisation: protein–protein interaction signal peaks at L0 (STRING≥700 AUROC 0.640; ρ = −0.608 with depth, *q* = 0.011), transcriptional regulatory signal increases with depth peaking at L15 (TRRUST AUROC 0.750; ρ = +0.511, *q* = 0.030), and functional co-annotation (KEGG, GO BP) tracks regulation. This hierarchy survives pairwise expression residualisation (97% of attention–TRRUST correlation retained after partial correlation on expression similarity). **But** the attention-derived GRN provides *zero incremental value* over trivial gene-level features (variance, mean expression, dropout rate) for perturbation-outcome prediction: gene-level baselines achieve AUROC 0.81–0.88 versus 0.70 for attention edges, and adding pairwise edge scores to a gene-level model yields ΔAUROC = −0.0004 [−0.001, 0.000]. Causal ablation of the top-ranked "regulatory" heads produces no degradation; random-head ablation causes a larger drop; orthogonal interventions (uniform attention replacement, MLP ablation) are null at the regulatory layers. The distinction matters: pairwise co-expression is a null attention covariate (controlled by partial correlation), but **gene-level features, not pairwise co-expression, are the dominant confound**. A responsible "GRN extraction" pipeline must therefore run the full battery and refuse to advertise the extracted edges as a regulatory network unless they pass every family. This pipeline enforces that by design — extraction and validation are inseparable, and Phase 12 emits an explicit pass/fail verdict.

## 2. Source

- **Paper:** Anonymous, (2026). *Systematic Evaluation of Single-Cell Foundation Model Interpretability Reveals Attention Captures Co-Expression Rather Than Unique Regulatory Signal.* arXiv:2602.17532 [q-bio.GN], 19 Feb 2026. 74 pages (main text + 19 Supplementary Notes + Supplementary Methods + 14 supplementary tables + 39 supplementary figures).
- **Local PDF:** `references/2602.17532_anon_SCFM_interpretability.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/biomechinterp-framework
- **Pinned commit:** `40e6da0ae21a768dbbcb2bfa62539845f0ef7c88` (2026-04-14, "Add revision analyses: positive controls, scGPT ablation, colorblind palette")
- **Local clone:** `repos/biomechinterp-framework/` (see `repos/biomechinterp-framework/PINNED_COMMIT.txt`)
- **Zenodo archive (data + code snapshot):** https://doi.org/10.5281/zenodo.18701417
- **Repository layout (16 topic modules):** `src/01_scaling_failure/`, `02_cssi_method/`, `03_perturbation_validation/`, `04_confound_decomposition/`, `05_causal_ablation/`, `06_cross_context/`, `07_mediation_bias/`, `08_detectability/`, `09_ortholog_transfer/`, `10_pseudotime/`, `11_batch_leakage/`, `12_calibration/`, `13_synthetic_validation/`, `14_multi_model/`, `15_biological/`, `16_statistical_framework/`, plus `shared/` and `plotting/`. See `src/README.md` and `src/FIGURE_SCRIPT_MAPPING.md`.
- **Key foundational references** (for context and theory; not re-implementations): Donoho & Jin 2004 (higher criticism / detectability), Peters et al. 2016 (invariant causal discovery), Haghverdi et al. 2016 (diffusion pseudotime), Strona et al. 2014 (curveball algorithm for degree-preserving null), Platt 1999 / Niculescu-Mizil & Caruana 2005 (calibration), Vovk et al. 2005 (conformal prediction), Benjamini & Hochberg 1995 (FDR), Pearl 2001 / Imai et al. 2010 (causal mediation), Shapley 1953 / Lundberg & Lee 2017 (Shapley values), Han et al. 2018 (TRRUST v2), Garcia-Alonso et al. 2019 (DoRothEA), Szklarczyk et al. 2021 (STRING), Replogle et al. 2022 (Perturb-seq K562/RPE1), Adamson et al. 2016 (K562 CRISPRa), Shifrut et al. 2018 (T-cell CRISPRi), Tabula Sapiens Consortium 2022, Maynard et al. 2021 (DLPFC), Wolf et al. 2018 (Scanpy), Cui et al. 2024 (scGPT), Theodoris et al. 2023 (Geneformer), Lopez et al. 2018 (scVI), Huynh-Thu et al. 2010 (GENIE3), Moerman et al. 2019 (GRNBoost2), Jain & Wallace 2019 / Serrano & Smith 2019 / Wiegreffe & Pinter 2019 / Bibal et al. 2022 (attention-as-explanation debate in NLP).

## 3. Inputs

### 3.1 Target model (to be evaluated)
A trained single-cell transcriptomic model that exposes attention weights. The model must provide:
1. A tokenizer or embedding function that maps a cell's expression vector to a sequence of gene-indexed tokens. scGPT: gene-token transformer. Geneformer: BERT-style rank-based tokenization.
2. A forward pass that returns per-layer, per-head attention matrices of shape `(n_layers, n_heads, seq_len, seq_len)`. For HuggingFace models, set `output_attentions=True` **and** `attn_implementation="eager"` (non-negotiable — see Phase 4 and Known Pitfalls #3).
3. A `head_mask` hook or equivalent mechanism to zero out individual heads during forward passes (required for causal ablation — Phase 4). For HF-BERT models the `head_mask` parameter has shape `(n_layers, n_heads)`.
4. Forward hooks on each self-attention module that expose the context layer `softmax(QK^T/√d)·V` (required for value-weighted edge extraction — Phase 0b, Supp Note 18).
5. A cell-level embedding interface (for CSSI stratification — Phase 6).

**Worked examples in source paper:**
- **scGPT** (Cui et al. 2024) — 3 tiers (small/medium/large)
- **Geneformer V1-10M** (Theodoris et al. 2023) — 6 layers × 4 heads (multi-model comparison; near-random TRRUST AUROC 0.444–0.549)
- **Geneformer V2-316M** — 18 layers × 18 heads (primary detailed analysis; L13 pre-specified, L15 nested-CV-selected)
- **scVI** (Lopez et al. 2018) — latent-distance edges, near-random AUROC 0.48–0.53
- **C2S-Pythia** (405M-parameter causal LM) — near-random AUROC 0.48–0.53

### 3.2 Single-cell transcriptomic datasets

| Dataset | Source | Role | Key parameters |
|---|---|---|---|
| Tabula Sapiens (immune, kidney, lung) | Tabula Sapiens Consortium 2022 | Scaling analysis, cross-tissue consistency, biological characterisation, controlled composition | ~200,000 cells; 20,000 cells for attention–correlation mapping (immune) |
| DLPFC brain scRNA-seq | Maynard et al. 2021 | Layer pre-specification (Geneformer L13), CSSI synthetic + real-attention validation | 497 cells |
| Replogle K562 CRISPRi | Replogle et al. 2022 | **Primary** perturbation-first evaluation, full confound decomposition, causal ablation | >640k cells, 2,024 perturbed genes |
| Replogle RPE1 CRISPRi | Replogle et al. 2022 | **Primary secondary** cross-context replication; full RPE1 confound battery | hTERT-RPE1, 1,251 evaluable perturbations, 1,167 primary |
| Adamson K562 CRISPRa | Adamson et al. 2016 | Cross-modality (CRISPRa) replication | 77 evaluable perturbations |
| Shifrut T-cell CRISPRi | Shifrut et al. 2018 | Cross-cell-type replication (primary T cells) | 7 evaluable perturbations (underpowered) |
| Tian iPSC neurons CRISPRi | Tian et al. 2021 | Cross-cell-type replication (iPSC-derived glutamatergic neurons) | 7 evaluable perturbations (underpowered) |
| Dixit 13-day / 7-day Perturb-seq | Dixit et al. 2016 | Condition-specific perturbation validation (scGPT mediation) | Perturbation pairs |
| Mouse lung (Krasnow Smart-seq2) | Travaglini et al. 2020 | Cross-species ortholog transfer (vs Tabula Sapiens human lung) | 9,409 mouse cells, 53,482 one-to-one orthologs |

Perturbation datasets are accessed via the `scperturb` package. All primary datasets are publicly available from original sources (paper §Data Availability).

### 3.3 Reference networks and biological databases

| Network | Type | Edges (in HVG context) | Usage |
|---|---|---|---|
| **TRRUST v2** (Han et al. 2018) | Curated TF→target | ~9,000 global; 175 pairs in 2000-HVG K562; 8,427 unique; 4,859 (58%) with known direction | Primary regulatory reference; GRN recovery AUROC/F1 |
| **DoRothEA** (Garcia-Alonso et al. 2019) | Curated TF→target | ~37,000 | Multi-model validation secondary reference |
| **STRING ≥700** (Szklarczyk et al. 2021) | PPI | 2,747 pairs | Biological characterisation — PPI signal at early layers |
| **STRING ≥900** | High-confidence PPI | 1,675 pairs | Biological characterisation — PPI signal at early layers |
| **Reactome** | Pathway co-membership | 238,640 pairs | Biological characterisation — functional co-annotation |
| **KEGG** | Pathway co-membership | 24,812 pairs | Biological characterisation — functional co-annotation (peaks at L16) |
| **GO Biological Process** | Functional co-annotation | 135,596 pairs | Biological characterisation (peaks at L16) |

Download: TRRUST v2 at https://www.grnpedia.org/trrust/; STRING at https://string-db.org/; Reactome/KEGG/GO via standard bioinformatics channels.

## 4. Outputs

Per dataset × model combination, the pipeline emits (all under `results/` in the repo layout):

1. **Per-layer per-head attention tensors** (NPY): shape `(n_layers, n_heads, n_genes, n_genes)`, cell-averaged over `n_ctrl=2000` control cells, with persistent gene ↔ token-index mapping.
2. **Value-weighted context tensors** (NPY, optional but recommended): `A_vh = softmax(QK^T/√d)·V` per layer/head from forward hooks, shape `(n_layers, n_heads, n_genes, d_head)`, with derived pairwise cosine-similarity edge matrices. Source: Supp Note 18.
3. **Pairwise edge-score matrices** (NPY/CSV): one matrix per layer plus per-head where analyses demand it. Three edge types: (a) raw attention, (b) Spearman correlation computed on the *same* `n_ctrl` cells (essential — see Pitfall #6), (c) value-weighted cosine similarity.
4. **Gene-level feature table** (CSV): mean expression, variance, dropout rate, and TF out-degree (TRRUST) per gene.
5. **Perturbation-first AUROC table** (JSON) per perturbation, per layer, per edge type (attention / correlation / gene-level baselines / value-weighted), with Mann-Whitney *U* or Welch *t* statistics, BH-corrected *p*-values, Cohen's *d*, and 200-iteration perturbation-level bootstrap CIs (10,000 resamples for headline CIs).
6. **Reference network recovery metrics** (JSON): top-*K* F1 (K ∈ {20, 50, 100, 200, 500}), continuous AUROC, AUPRC, Precision@10k against each of the seven reference networks in §3.3, per layer/head and after CSSI stratification.
7. **Confound-decomposition report** (JSON): gene-only / edge-only / gene+edge AUROC under 5-fold GroupKFold CV with three split designs (cross-perturbation, cross-gene, joint); cross-fitted residualisation (5-fold OLS and GBDT) with seed stability; propensity-matched evaluation (*k*=5 DE-negative matches per DE-positive target); degree-preserving null via curveball algorithm (*n*=200) and label-shuffling null (*n*=1,000); metric-robust extension to AUPRC and top-*k* recall (*k* ∈ {10, 50, 100}) under GBDT.
8. **Causal-ablation report** (JSON): 13-condition head-mask table (top-*k* TRRUST-ranked × {5,10,20,50}, composite-ranked × {5,10}, full-layer L14, bottom-*k* ∈ {5,10}, random × 3 seeds, entropy-matched) plus orthogonal interventions (uniform attention replacement at top-5/10/random-10; MLP zero at L15, L13–L15, random L8) and intervention-fidelity diagnostics (max hidden-state cosine, logit cosine, cells per intervention = 2000).
9. **Cross-context report** (JSON): per-dataset attention-vs-correlation per-layer effect sizes and *p*-values, HVG protocol confound test (restricted HVG = 2000 vs forced perturbation-gene HVG), per-perturbation ΔAUROC distributions with 10,000-sample bootstrap CIs.
10. **CSSI output** (JSON + NPY): per-stratum Spearman edges for each TF–target pair, aggregated via CSSI-max / CSSI-mean / SCENIC-style, null tests under shuffled/random/gene-permuted labels, K-sensitivity scan K ∈ {2,…,20}, synthetic oracle-label ground truth, real-attention per-layer ΔAUROC (see Table S7 in paper).
11. **Biological characterisation report** (JSON + CSV): per-layer AUROC of attention edges against all seven reference networks, partial correlation controlling for Spearman expression similarity, top-1,000 edge enrichment via Fisher's exact test with BH across 30 tests (5 layers × 3 databases × 2 tails).
12. **Boundary-condition report** (JSON): nine sub-reports — detectability phase diagrams, invariant causal discovery cross-tissue consistency, cross-species ortholog transfer, pseudotime directionality, batch/donor leakage audit with Artifact Sensitivity Index (ASI) blacklist, uncertainty calibration with ECE + conformal prediction sets, TRRUST circularity sensitivity, HVG protocol confound test, Dixit/Adamson/Shifrut condition-specific mediation validation.
13. **Multi-model comparison report** (JSON): cross-model AUROC table (scGPT / Geneformer V1-10M / Geneformer V2-316M / scVI / C2S-Pythia), attention–correlation mapping *R²* per model, degree-preserving null decomposition.
14. **Synthetic ground-truth validation report** (JSON): synthetic GRN recovery vs theoretical predictions, Shapley-vs-single-component contrast, empirical-vs-theoretical detection correlation.
15. **Per-TF characterisation report** (JSON): per-TF AUROC table with TF class annotation (master regulator / signal-dependent / lineage-specific / housekeeping), target-count diagnostics.
16. **Statistical test registry** (JSON + CSV): all 153 tests with section, hypothesis, test type, raw *p*, BH *p*, effect size, *N*, significance flag. One row per test (see Supp Table S10 for canonical format).
17. **Pass/fail verdict** (JSON): boolean + explanation for each of the four mandatory criteria in Phase 12 step 3.
18. **Claim-to-evidence mapping** (JSON): 20 headline claims → supporting analyses → key tests (see Supp Table S11).

## 5. Dependencies

- **Python 3.8–3.10**, PyTorch ≥ 1.12 (CUDA recommended), Transformers ≥ 4.20, scanpy ≥ 1.9, anndata ≥ 0.8, scikit-learn ≥ 1.0, statsmodels ≥ 0.13 (for BH FDR and multiple testing), pandas ≥ 1.4, numpy ≥ 1.21, scipy ≥ 1.9.
- **scperturb** — perturbation-dataset loader.
- **geneformer** — HuggingFace `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M` (primary) and `Geneformer-V1-10M` (multi-model comparison).
- **scGPT** reference implementation — for the scaling-failure module (`src/01_scaling_failure/`); accepts pretrained tiered checkpoints.
- **scVI** (Lopez et al. 2018) — for multi-model comparison.
- **leidenalg** — Leiden community detection for CSSI strata.
- **statsmodels** — for BH FDR correction at α=0.05 (framework-wide across 95 confirmatory tests, plus sensitivity analysis across 3 alternative family definitions).
- **mlxtend** or equivalent — for the curveball algorithm (Strona et al. 2014), degree-preserving null model.
- **Reference files:** TRRUST v2, DoRothEA, STRING ≥700 and ≥900, Reactome, KEGG, GO BP (cached locally).
- Full environment: `repos/biomechinterp-framework/src/environment.yml` or `src/requirements.txt`; lockfile at `repos/biomechinterp-framework/requirements.lock`.

## 6. Methodology

The pipeline runs in twelve phases. Phase 0 (incl. 0a, 0b) is the model-specific adapter; Phases 1–11 are the model-agnostic evaluation battery; Phase 12 emits the verdict.

---

### Phase 0a — Model adapter: attention-pattern extraction

**Purpose:** produce a standard set of artefacts — a per-layer-per-head attention tensor, the gene↔token-index mapping, a same-cells Spearman correlation baseline, and a gene-level feature table — from an arbitrary single-cell transformer.

**Steps:**

1. **Load and preprocess the dataset** with Scanpy standard QC (Wolf et al. 2018): filter cells/genes by minimum counts, normalise total counts to 1e4, log1p transform. Select `N_ctrl = 2000` control cells (primary K562 config) and HVG count per §7. Paper §4.1 and Supp Note 6.3.
2. **Tokenise** cells using the model's native tokenizer. For Geneformer V2-316M: rank-based tokenization via HuggingFace token dictionary (`Geneformer/token_dictionary_gc95M.pkl`). For scGPT: gene-token mapping via the model's gene vocabulary. **Record the bidirectional `gene_name ↔ token_id` map** and persist to disk — this map is the ground on which every downstream analysis sits. For Geneformer V2-316M: 1,941 of 2,000 HVG genes are in the token vocabulary (see Supp Note 18); only those genes are evaluable.
3. **Forward pass with `output_attentions=True` and `attn_implementation="eager"`.** Batched, eval mode. Accumulate per-head attention matrices across all `n_ctrl` control cells; **mean-reduce** over cells to obtain a tensor of shape `(n_layers, n_heads, seq_len, seq_len)`. Map sequence positions back to gene indices via the tokenizer map → gene-space attention tensor.
4. **Construct edge scores.** Raw edge score: `A[l,h,i,j]` = cell-averaged attention from gene *i* to gene *j* in layer *l*, head *h*. The pipeline also stores: per-layer mean over heads, per-layer max over heads, and a TRRUST-informed composite rank across layers/heads (used for Phase 4 ablation targeting).
5. **Compute Spearman correlation baseline from the same `n_ctrl` cells.** Store a shape-identical `(n_genes, n_genes)` matrix. **This is essential**: sample-size differences between attention extraction and correlation baseline would confound every downstream comparison.
6. **Compute gene-level features:** mean expression, variance, dropout rate (1 − fraction of cells with non-zero counts), and TF out-degree (from TRRUST). Store as a flat per-gene table. Source: §4.4 and §4.1.

**Layer pre-specification.** **Mandatory to avoid selection bias.** The paper pre-specifies the Geneformer V2-316M primary layer as **L13** using an *independent* dataset (DLPFC brain, 497 cells, Maynard et al. 2021) before any perturbation evaluation. Under strict 5-fold nested CV, **L15** is subsequently selected in all 5 folds with pooled held-out Δ = +0.040 [0.018, 0.062], *p*_Bonf = 0.017 (Supp Note 14.1). Early layers AUROC 0.47–0.64, mid layers 0.60–0.71, late layers 0.69–0.74 in the 18-layer profile. Do not loop over layers and report the best — cross-validated nested selection or pre-specification is required.

### Phase 0b — Model adapter: value-weighted edge extraction (Supp Note 18)

**Purpose:** test whether perturbation-predictive computation resides in the value/FFN pathway rather than the attention pattern itself. The attention-ablation null (Phase 4) hints that the causal circuit is downstream of the QK softmax; Phase 0b provides the affirmative test.

**Steps:**

1. **Register forward hooks** on each `BertSelfAttention` module (Geneformer) or equivalent attention block. Capture the context layer `A_vh = softmax(QK^T/√d) · V`, shape `(n_genes, d_head)` per layer, per head, for each of `n_ctrl=2000` control cells.
2. **Pairwise cosine-similarity edges.** For every gene pair *(i, j)* in each layer/head: `edge(i,j) = cos(A_vh[i,:], A_vh[j,:])`. Average across heads within each layer. Also compute: (a) **centroid cosine similarity** on the mean context vector across cells, (b) **dot-product** variant. Source: Supp Note 18.
3. **Evaluate VW edges** under the full pipeline (Phases 1–4) alongside raw attention and correlation. Expected result in Geneformer V2-316M (Table S13):
   - Best VW cosine: Pert-first AUROC 0.587 (L12), TRRUST AUROC 0.606 (L12)
   - Best raw attention: Pert-first 0.787 (L14), TRRUST 0.718 (L15)
   - Correlation: 0.706 / 0.638
   - Paired Wilcoxon VW vs raw attention: Δ = −0.200, *p* = 4.9×10⁻³⁹, *d* = −1.68
   - Paired Wilcoxon VW vs correlation: Δ = −0.120, *p* = 2.4×10⁻¹⁴, *d* = −1.00
4. **Conclusion from VW test.** Value-weighted cosine *underperforms* both raw attention and correlation, ruling out the "regulatory information is hidden in the value pathway" hypothesis. The context layer mixes information from all attended genes, collapsing pairwise structure. Combined with the Phase 4 null (random-head ablation > regulatory-head ablation), this indicates perturbation-predictive computation is **distributed across the network in a form not recoverable from any simple attention-derived edge score** — neither attention pattern nor value-weighted context.

**Code reference:** `src/15_biological/01_value_weighted.py` → `perturbation_first_auroc()`, `trrust_auroc()`.

---

### Phase 1 — Trivial-baseline comparison (Test family 1; Supp Table S10 §21)

**Purpose:** determine whether pairwise edge scores (attention or correlation) outperform univariate gene-level features that require no model at all.

**Steps:**

1. **Call differentially-expressed (DE) targets** for each perturbation `g`. Primary: Welch's *t*-test with LFC > 0.5 threshold (Replogle K562: *n* = 151 perturbations; *n* = 280 under attention tokenization; *n* = 44 at Mann-Whitney *U* / LFC > 0.1 baseline). BH correction at α=0.05. **Four datasets with distinct parametrisations** (Supp Table S2): Replogle K562 primary LFC>0.5, HVG=2000, N_ctrl=2000 → *n* = 151. Replogle K562 attention LFC>0.5, HVG=2000 → *n* = 280 (more genes evaluable via token matching). Adamson K562 CRISPRa LFC>0.5 → *n* = 77. RPE1 LFC>0.5 with perturbation genes forced into HVG=3309 → *n* = 1,251 (1,167 primary). Shifrut T-cell and Tian iPSC neuron at LFC>0.1 → *n* = 7 each (underpowered).
2. **Edge scores as predictors.** For each perturbation *g*: rank all target candidates by attention edge `A[g, target]` (primary layer) and by Spearman edge `C[g, target]`; compute AUROC against DE-positive/DE-negative labels.
3. **Gene-level univariate predictors.** For each target gene: rank by mean expression, by variance, by 1−dropout rate, and by TF out-degree (TRRUST, as negative control). Compute AUROC identically.
4. **Statistical test:** paired *t*-test or Wilcoxon signed-rank between per-perturbation AUROCs (gene-level vs attention; gene-level vs correlation), BH-corrected. Paper result on Replogle K562 at primary parameterisation (Supp Table S10 §21, all at *n* = 151):
   - **Variance vs correlation:** Δ = +0.186, *p* < 10⁻²⁴ (gene-level wins)
   - **Mean expression vs correlation:** Δ = +0.146, *p* < 10⁻²⁰
   - **1 − dropout vs correlation:** Δ = +0.113, *p* < 10⁻¹²
   - **TF out-degree vs correlation:** Δ = −0.196, *p* < 10⁻²⁹ (TF degree underperforms correlation — sanity-check counter-example)
   - Attention L13 AUROC ≈ 0.704 ± 0.147; Correlation ≈ 0.703 ± 0.164 (Wilcoxon *p* = 0.73)
   - Variance AUROC 0.881; Mean 0.851; 1−dropout 0.866 (all trivial baselines significantly > attention at *p* < 10⁻¹²)
5. **Red flag:** if trivial baselines outperform attention, downstream pipeline steps still run (they are the diagnostic tests) but the final verdict flag for "genuine regulatory content" is already tripped.

**Code reference:** `src/03_perturbation_validation/06_trivial_baselines.py` → `run_trivial_baselines()`. Shared helpers from `src/shared/01_unified_extraction.py`: `call_de_vectorized()`, `tokenize_cell()`, `load_categorical_column()`.

---

### Phase 2 — Conditional incremental-value testing (Test family 2; Supp Notes 15, 14.1; Supp Table S10 §26, §31)

**Purpose:** determine whether pairwise edge scores add *any* predictive value to a model that already has access to gene-level features — the core criterion for "does attention encode something gene-level features miss?"

**Steps:**

1. **Build a per-(perturbation, target-gene) dataset.** Columns: target gene-level features (mean expression, variance, dropout rate), attention edge `A[g, target]`, correlation edge `C[g, target]`, and binary label (DE target / non-DE target). For Replogle K562 at primary parametrisation: **559,720 pairs, 2.8% positive rate**.
2. **Fit three logistic-regression models** under **5-fold GroupKFold** cross-validation **grouped by perturbation** (so no perturbation leaks between train and test):
   - Gene-only: features = {mean expr, variance, dropout rate}
   - Gene+attention
   - Gene+correlation
3. **Metric:** cross-validated AUROC. Report ΔAUROC = (gene+edge) − (gene-only). Paper result:
   - Gene-only AUROC: 0.895
   - ΔAUROC_attention = **−0.0004** [95% CI −0.001, 0.000]
   - ΔAUROC_correlation = **−0.002** [−0.005, 0.000]
   - Both centred at zero; LR incremental value is **zero** (Supp Table S10 §26; Extended Data Fig. 6).
4. **Hard-generalisation protocols (Supp Table S10 §31).** Repeat step 2 under two progressively harder splits to guard against gene-level feature leakage:
   - **(a) Cross-gene GroupKFold by target gene** — preventing any target from appearing in both train and test
   - **(b) Joint cross-gene × cross-perturbation splits** — hardest
   Run both logistic regression (linear) and GBDT (nonlinear).
5. **Metric-robust extension (Supp Note 15; Supp Fig. S39).** Extend step 2 to AUPRC and top-*k* recall for *k* ∈ {10, 50, 100}, under GBDT across all three split designs. **Null persists universally.** Largest ΔAUPRC observed: +0.009 under joint splits with GBDT (<4% relative improvement). ΔAUROC bootstrap bounds under hard splits: cross-pert gene+attn vs gene Δ=−0.0004 [−0.001, 0.000]; cross-gene Δ=−0.0003 [−0.001, 0.000]; joint Δ=−0.0003 [−0.001, 0.000]. All null.
6. **Bootstrap ΔAUROC** with 100–200 perturbation-level resamples; compare to 0 via one-sample *t* / Wilcoxon.
7. **Power check.** 559,720 observations provide > 99% power to detect ΔAUROC = 0.005 at α = 0.05.
8. **TF stratification.** Subset analysis by TF vs non-TF perturbation targets. Paper result (Fig. 3C): identical pattern — gene-only TF = 0.913, non-TF = 0.895; adding attention or correlation provides no gain in either stratum.

**Code reference:** `src/04_confound_decomposition/01_incremental_value.py` → `build_dataset()`, `evaluate_model_cv()`, `run_incremental_value()`. Hard-gen and metric-robust variants in the same module.

---

### Phase 3 — Expression residualisation, degree-preserving null, propensity matching (Test family 3; Supp Notes 13.3, 13.4, 14.2, 14.8; Supp Table S10 §15, §16, §25, §35)

**Purpose:** isolate edge-specific signal by explicitly removing the gene-level confound from the edge scores themselves, rather than adding gene features alongside.

**Steps:**

1. **Raw covariate explanation.** Fit `A[g, target] ~ f(mean_expr, variance, dropout rate)` using 5-fold **OLS** and 5-fold **GBDT**. Paper results (Supp Note 13.3 / Fig. S26):
   - OLS *R²*: **0.27** (attention), **0.16** (correlation — less expression-confounded)
   - GBDT *R²*: **0.51** (attention), **0.31** (correlation)
   - Signed-vs-absolute covariate robust: *R²* stable across variants
2. **Cross-fitted residualisation.** The residual `A − f̂` is the component orthogonal to gene-level covariates. Compute AUROC of the residual for TRRUST edge prediction.
   - Attention: baseline 0.66 → OLS-residualised **0.54** (loses ~76% of above-chance signal)
   - Correlation: baseline 0.63 → OLS-residualised **0.62** (loses only ~9%)
   - Attention GBDT residual: 0.538 (Fig. S32B)
   - Cross-fitted residual AUROC with 10 seeds: σ = 0.0009 (highly stable)
   - **Asymmetry interpretation:** attention edges are substantially more expression-confounded than correlation edges; on residualisation, correlation has a suppressor effect while attention collapses to near-chance.
3. **Degree-preserving null model.** Generate 200 permuted GRNs using the **curveball algorithm** (Strona et al. 2014), which randomises edges while exactly preserving each TF's out-degree. Generate 1,000 additional permutations under the label-shuffling null. Paper results (Supp Note 13.4 / Fig. S27):
   - Observed raw attention AUROC: **0.757**
   - Decomposition: 0.50 (chance) + **0.19 (degree confound)** + **0.07 (residual signal above degree)**
   - Label-shuffling null: *z* = 6.9, *p* < 0.001
   - Degree-preserving null: *z* = 3.63, *p* < 0.005 (modest, residual above degree is small)
   - Per-TF evaluation: only **7/18 TFs (39%)** have 95% CI entirely above 0.5
   - Per-TF mean AUROC: 0.692 ± 0.18
4. **TRRUST circularity sensitivity analysis** (Supp Note 13.5). TRRUST entries curated partly via co-expression studies — a potential source of inflated AUROC. Restrict TRRUST to **direction-known entries only** (Activation or Repression mode; 4,859 of 8,427 unique pairs, 58%) which require more direct experimental evidence (perturbation, reporter assays, ChIP-seq). Expected result:
   - Global AUROC: 0.764 → 0.736 (Δ = −0.028 — small)
   - Per-TF mean AUROC: virtually unchanged (0.682 vs 0.692)
   - Per-TF median AUROC: improves (0.695 vs 0.660)
   - TFs above chance: 10/12 (83%) vs 14/18 (78%)
   - Conclusion: TRRUST-based evaluation conclusions are **not driven** by circularly validated entries; robust to restriction.
5. **Propensity-score matching (Supp Note 14.8; Fig. S38).** Fit logistic regression `is_DE_target ~ gene-level covariates`. For each DE-positive target, nearest-neighbour-match *k* = 5 DE-negative targets with similar propensity. Verify post-matching SMD < 0.1 for all four covariates (`tgt_mean_expr`, `tgt_variance`, `tgt_dropout`, `tgt_n_affected`). Paper pre-matching SMDs: 0.19–1.24; post-matching: 0.12–0.71 (tgt_n_affected remains highest at 0.71 — target-degree is harder to match).
   - Matched dataset: **59,153 pairs**, positive rate 26.5% vs 2.8% raw
   - Raw edge AUROC on matched set: attention **0.609** ± 0.121, correlation **0.574** ± 0.169
   - **Gene-only GroupKFold AUROC: 0.7492**; gene+attn: **0.7491**; gene+corr: **0.7489**
   - ΔAUROC gene+attn vs gene-only (matched, bootstrap): **−0.000** [−0.000, +0.000]
   - ΔAUPRC gene+attn vs gene-only: +0.001 [0.000, +0.002] (marginal, 4% relative)
   - TRRUST direct-target AUROC (*n* = 6 TFs): attention 0.695 ± 0.105, correlation 0.578
   - **Conclusion:** attention edges retain modest raw discriminability (0.609) under matching but zero incremental value over gene-level features. Matching rules out the "confound balance drives the null" explanation.
6. **Composite edge ranking** for Phase 4 ablation targeting. Compute TRRUST recovery AUROC × perturbation-first AUROC per head to produce the "composite-ranked" head order.

**Code references:**
- Residualisation: `src/04_confound_decomposition/04_residualization.py` → `run_residualization()`
- Propensity matching: `src/04_confound_decomposition/10_propensity_matched.py` → `run_propensity_matching()`, `compute_propensity_scores()`
- Degree-preserving null (curveball): `src/14_multi_model/` (Supp Note 13.4)
- TRRUST circularity: `src/14_multi_model/` (Supp Note 13.5)

---

### Phase 4 — Causal ablation with fidelity diagnostics (Test family 4; Supp Note 14.3, 14.4, 14.7; Supp Table S10 §29, §32, §34, §36)

**Purpose:** test whether "regulatory" heads — those whose attention edges best predict known TF–target pairs — make a *causal* contribution to perturbation prediction, via head-masking intervention.

**Steps:**

1. **Rank heads.** For every attention head `(l, h)`, compute TRRUST recovery AUROC and perturbation-first AUROC. Build per-head rankings by (a) TRRUST-AUROC, (b) perturbation-first AUROC, and (c) composite score (product or rank-sum). Paper result for Geneformer V2-316M (18 × 18 = 324 heads): per-head TRRUST AUROC range **0.34–0.75**; per-head perturbation-first mean = 0.58 ± 0.07.

2. **Original 6-condition head-mask ablation (Supp Note 14.3; Fig. S33).** Using the HF BERT `head_mask` parameter (shape `n_layers × n_heads`), zero out specific head outputs during the forward pass. The six original conditions:
   - Baseline
   - Ablate top-5 TRRUST-ranked regulatory heads
   - Ablate 5 random heads × 3 replicates (Random 1, 2, 3)
   - Ablate 5 entropy-matched heads
   Results (perturbation-first AUROC, *n* = 280):
   - Baseline 0.7035
   - Top-5 regulatory: 0.7037 (Δ = +0.0002, *p* = 0.244, not sig.)
   - Random 1: 0.7028 (Δ = −0.0007, *p* = 0.065)
   - Random 2: 0.7031 (Δ = −0.0004, *p* = 0.069)
   - Random 3: 0.7007 (Δ = −0.0028, *p* < 10⁻⁶, Wilcoxon sig.)
   - Entropy-matched: 0.6994 (Δ = −0.0042, *p* < 10⁻⁶, sig.)

3. **Expanded 13-condition head-mask ablation (Supp Table S10 §32; Fig. 4 main).** Test a dose-response grid:
   - Top-*k* TRRUST-ranked: *k* ∈ {5, 10, 20, 50}. Paper result: 0.704, 0.704, 0.7031 (*p*=0.596, *d*=0.02), 0.7021 (*p*=0.100, *d*=0.08) — all null
   - Bottom-*k*: *k* ∈ {5, 10}. Paper result: 0.7060 (Δ = −0.0025, *p* = 0.005, *d* = −0.16 — ablating bottom-5 heads *improves* performance marginally); 0.7005 (Δ = −0.0030, *p* = 0.050)
   - Composite-ranked top-*k*: *k* ∈ {5, 10}. Paper result: 0.7035 (*d* = 0.00), 0.7035 (*d* = 0.00)
   - Full-layer L14: 0.7035 (*d* = 0.00)
   - Random-*k*: *k* ∈ {10, 20, 50}. Paper result: Random-10 0.7064 (*p* = 0.009, *d* = 0.18, sig.); Random-20 0.6964 (*p* < 10⁻⁸, *d* = 0.33, sig.); Random-50 0.7037 (*p* = 0.351)
   - **Dose-response is inverted:** TRRUST-ranked heads show no dose-response; random heads at *k* = 20 cause the *largest* significant drop. Entropy-matched and random-head ablations produce 23× larger logit perturbation than the regulatory-head ablation at matched dose (Supp Note 14.7), yet the regulatory-head ablation is null. This rules out "the intervention was too small to be detected."

4. **Orthogonal interventions (Supp Note 14.4; Fig. S34; Supp Table S10 §34).** Two families beyond standard head masking:
   - **(a) Uniform attention replacement.** At the target layer/heads, set attention weights to `1/n` (uniform over keys) while preserving value projections. Conditions: uniform-reg top-5, uniform-reg top-10, uniform-random-10. Paper results:
     - Uniform reg top-5: 0.7040 (Δ = −0.0004, *d* = −0.17, *p* = 0.004, sig.)
     - Uniform reg top-10: 0.7033 (Δ = +0.0002, *d* = +0.07, *p* = 0.052)
     - Uniform random-10: 0.7067 (Δ = −0.0032, *d* = −0.20, *p* < 10⁻³, sig.)
     Regulatory-head uniform replacement is marginally significant at top-5 but indistinguishable from the random baseline at top-10 — inconsistent with a genuine causal role.
   - **(b) MLP ablation.** Zero the FFN output at specific layers. Conditions: MLP zero L15, MLP zero L13–L15, MLP zero L8 (random-layer control). Paper results:
     - MLP zero L15: 0.7035 (*d* = 0.00, null)
     - MLP zero L13–L15: 0.7035 (*d* = 0.00, null)
     - MLP zero random layer L8: 0.6982 (Δ = −0.0053, *d* = −0.27, *p* < 10⁻⁴, **significant**)
   - **Striking inversion:** MLP ablation at the "regulatory" layers (L13–L15) produces exactly baseline AUROC, while MLP ablation at a random layer (L8) produces the largest significant drop in the entire intervention battery. This confirms (a) the intervention infrastructure works (L8 ablation causes a real drop) and (b) the "regulatory" layers are *not* where perturbation-predictive computation lives.

5. **Intervention-fidelity diagnostics (Supp Note 14.7; Fig. S37; Supp Table S10 §36).** For each intervention, confirm it materially perturbs internal representations. Measure:
   - **Max hidden-state cosine distance** between intervened and baseline forward passes across *n* = 2000 cells, per layer
   - **Logit cosine distance** (overall output perturbation)
   Paper results (Supp Table S10 §36):
   - Uniform reg top-5: max hidden cos = **0.057**, logit cos = **0.005**
   - Uniform reg top-10: max hidden cos = **0.085**, logit cos = **0.021**
   - Uniform random-10: max hidden cos = **0.042**, logit cos = **0.001**
   - MLP zero L15: max hidden cos = **0.043**, logit cos = **0.003**
   - MLP zero L13–L15: max hidden cos = **0.190**, logit cos = **0.012**
   - MLP zero random L8: max hidden cos = **0.023**, logit cos = **0.001**
   - **TRRUST-ranked heads produce 23× larger logit perturbation than random heads at matched dose** — yet the behavioural AUROC drop is nil. Hidden-state cosine distance > 0.02 for all six conditions confirms all interventions are material.
   - **The convergence of null AUROC across six qualitatively different intervention channels, each with strong hidden-state perturbation, constitutes affirmative evidence for distributed functional redundancy** — not a failure of the intervention infrastructure.

**Code references:**
- Head-mask ablation (6-condition): `src/05_causal_ablation/01_head_ablation.py` → `extract_attention_with_ablation()`, `evaluate_perturbation_first()`
- Expanded 13-condition ablation: `src/05_causal_ablation/` (expanded variant)
- Orthogonal interventions: `src/05_causal_ablation/04_orthogonal_interventions.py`
- Intervention fidelity: `src/05_causal_ablation/05_intervention_fidelity.py`

---

### Phase 5 — Cross-context replication (Test family 5; Supp Notes 14.5, 14.6, 14.9; Supp Table S10 §28, §33, §37)

**Purpose:** determine whether any finding generalises across cell types and perturbation modalities.

**Steps:** repeat Phases 1–4 on each of the five non-K562 cross-context datasets. Enforce identical preprocessing, `n_ctrl`, HVG count, and LFC thresholds as the primary Replogle K562 run, except where noted in §7.

**Results summary:**

1. **Replogle RPE1 CRISPRi (Supp Table S10 §37; Fig. 6 / S38).** Primary non-K562 replication. *n* = 1,167 perturbations. Paper result: attention significantly outperforms correlation across all layers.
   - L6: diff = +0.118, *d* = 0.74, *p* < 10⁻¹⁰
   - L13: diff = +0.036, *d* = 0.20, *p* < 10⁻¹⁰
   - L15: diff = +0.090, *d* = 0.47, *p* < 10⁻¹⁰
   - L18: diff = +0.086, *d* = 0.48, *p* < 10⁻¹⁰
   **However**, the RPE1 confound battery (Fig. 6) shows:
   - Gene-only AUROC: **0.942** (L = 2,000 pairs); mean expression alone 0.851; 1-dropout 0.866; variance 0.797
   - Gene+attention = 0.942 (zero incremental value)
   - Gene+correlation = 0.941
   - Cross-fitted residualisation: attention loses ~88% of signal (baseline 0.657 → 0.538); correlation increases slightly (suppressor; 0.634 → 0.646)
   - TRRUST direct-target (*n* = 54 TFs): attention 0.559, correlation 0.540 — both near chance
   - **Conclusion:** RPE1 attention advantage over correlation does **not** reflect attention capturing regulatory structure that correlation misses — attention is a better proxy for gene-level features (which dominate RPE1 as well). The RPE1 gene universe (3,309 genes with forced perturbation genes) provides richer gene-level features than K562 (2,000 HVGs).

2. **Adamson K562 CRISPRa (Supp Note 14.5; Fig. S35; Supp Table S10 §28).** *n* = 77. Attention **significantly underperforms** correlation.
   - L6: diff = −0.098, *p* < 10⁻⁶
   - L13: diff = −0.106, *p* < 10⁻⁶
   - L18: diff = −0.105, *p* < 10⁻⁶
   - AUROC attention = 0.55, correlation = 0.65
   - **Direction reverses vs RPE1** — context-dependence of the attention–correlation gap.

3. **Shifrut T-cell CRISPRi (Supp Note 14.6; Fig. S36; Supp Table S10 §33).** Primary human T cells, *n* = 7 (underpowered). Attention ≈ correlation.
   - L6: diff = +0.016, *d* = 0.08, *p* = 1.000
   - L13: diff = −0.003, *d* = −0.01, *p* = 0.938
   - L15: diff = +0.029, *d* = 0.16, *p* = 0.813
   - L18: diff = +0.030, *d* = 0.18, *p* = 0.938
   - All null; underpowered.

4. **Tian iPSC neurons CRISPRi** (Supp Table S10 §37, last row). *n* = 7. L15 attention vs correlation: diff = +0.058, *d* = 0.80, *p* = 0.078 (trend, not significant after BH).

5. **HVG protocol confound test (Supp Note 14.9; Fig. S38C).** A methodological asymmetry — RPE1 forces perturbation genes into a 3,309-gene HVG while K562 uses 2,000 pure HVGs — could confound the cross-context comparison. Re-evaluate RPE1 using top-2,000 HVG by variance only, no forced perturbation gene inclusion. Paper result:
   - Of 1,251 RPE1 perturbation genes, **418 are naturally in top-2,000 HVG**; 833 excluded
   - Mean per-perturbation Δ (attention − correlation) shifts from **−0.024 to +0.168** (paired Wilcoxon *p* < 10⁻⁴⁶)
   - This shift is driven by: correlation AUROC drops substantially (0.723 → 0.593, Δ = −0.129) while attention modestly increases (0.699 → 0.762, Δ = +0.063)
   - **Bootstrap 95% CIs (10,000 samples)** on per-perturbation attention advantage exclude zero for all three conditions: K562 CRISPRi (*n* = 280) Δ = +0.060 [+0.040, +0.080], *p* = 8.9×10⁻⁸; RPE1 original (*n* = 1,167) Δ = +0.090 [+0.079, +0.101], *p* = 2.2×10⁻⁵⁴; RPE1 restricted (*n* = 418) Δ = +0.168 [+0.155, +0.182], *p* = 4.9×10⁻⁶⁰
   - **Interpretation:** the HVG protocol difference does *not* drive the attention advantage — the advantage *increases* under the restricted protocol. Correlation benefits from having more co-expressed genes in the scoring universe; attention is more robust to gene-universe size.
   - **Caveats:** the restricted comparison evaluates a biased subset (only naturally high-variance perturbation genes), and attention scores were precomputed on the 3,309-gene token context. A fully controlled test would re-extract attention on only 2,000 tokens.

6. **Condition-specific perturbation validation (Supp Note 6.1).** Counterfactual validation against four Perturb-seq datasets (Dixit 13-day, Dixit 7-day, Adamson, Shifrut) for scGPT mediation. Paper result: strongest positive signal is Dixit 13-day (ρ = 0.269, *p* = 0.032) remaining positive after confound adjustment (ρ = 0.199, *p* = 0.020); **only Dixit 13-day survives framework-level BH correction (adj. *p* = 0.042)**. All other dataset × condition combinations null.

**Code reference:** `src/06_cross_context/09_rpe1_confound_battery.py` → `run()` — canonical end-to-end confound battery on a non-K562 dataset. Adapt to new datasets by swapping `load_cached_data()` and re-running `run_trivial_baselines` → `run_incremental_value` → `run_propensity_matching` → `run_residualization` → `run_trrust_evaluation`.

---

### Phase 6 — Cell-State Stratified Interpretability (CSSI) (Supp Note 11; Supp Note 1; main text §2.2)

**Purpose:** attention-derived GRN recovery against TRRUST **degrades** as cell count increases in the input batch (top-*K* F1 drops in 9/9 scGPT kidney tier×seed runs at 200→1000 cells, sign test *p* = 0.002, 100% degradation). The paper formalises this as a dilution model `ρ_pool ≈ (n_1 / N) ρ_1 → 0` as heterogeneity grows. CSSI is the constructive remedy.

**Steps:**

1. **Stratify cells.** Compute a *k*-NN graph (default *k* = 15) from the model's cell-level embeddings — **not from raw expression**. Run Leiden community detection on the *k*-NN graph to partition cells into *K* strata.
2. **Per-stratum edge scoring.** Within each stratum independently, recompute Spearman correlation (and/or re-extract attention) for every TF→target pair. This yields *K* edge-score matrices.
3. **Aggregators.** Three variants evaluated:
   - **CSSI-max:** `edge(i,j) = max_k |score_k(i,j)|`
   - **CSSI-mean:** `edge(i,j) = mean_k score_k(i,j)`
   - **CSSI-range / CSSI-deviation:** variants tested per layer (see Supp Table S7)
4. **Null stress tests (Supp Note 11.2; Fig. S22, S23).** Repeat steps 2–3 using:
   - Shuffled cluster labels (preserve stratum sizes, break biology)
   - Random cluster labels (uniform random)
   - Gene-permuted labels
   Verify CSSI gains *vanish* under all three — rules out false-positive inflation from the stratification procedure. Paper result: pooled AUROC ≈ 0.5–0.8, CSSI-max oracle ≈ 0.95–1.0, shuffled / random CSSI ≈ 0.4–0.6 (below pooled). **No false-positive inflation across K ∈ {2,…,20}.**
5. **K sensitivity scan.** Test K ∈ {2,…,20}. Paper optimum: **K = 5–7** on DLPFC brain data, yielding **up to 1.85× improvement** in top-*K* TRRUST F1 over unstratified pooled baselines (Fig. 1).
6. **Synthetic validation (Supp Note 11.1; Table S5).** Construct ground-truth GRNs with known cross-state heterogeneity. Six configurations (Small *N* = 200, 2 states … Massive *N* = 1500, 12 states). Expected F1 ratio:
   | Config | *N* | States | Pooled F1 | CSSI-max F1 | Ratio |
   |---|---|---|---|---|---|
   | Small | 200 | 2 | 0.850 ± 0.053 | 0.957 ± 0.050 | 1.13× |
   | Medium | 400 | 4 | 0.657 ± 0.100 | 0.921 ± 0.071 | 1.40× |
   | Large | 600 | 6 | 0.486 ± 0.100 | 0.900 ± 0.069 | **1.85×** |
   | XLarge | 1000 | 8 | 0.550 ± 0.089 | 0.967 ± 0.029 | 1.76× |
   | XXLarge | 1000 | 10 | 0.514 ± 0.083 | 0.932 ± 0.049 | 1.81× |
   | Massive | 1500 | 12 | 0.527 ± 0.041 | 0.942 ± 0.027 | 1.79× |
7. **Real-data-structured validation (Supp Note 11.3; Table S6).** Realistic single-cell data with actual cell-type labels from 8 PBMC cell types (3,000 cells, 245 genes) and **22 known immune TF–target edges**:
   - CSSI-max recovers **22/22 (100%)** vs pooled **19/22 (86%)**
   - Advantage driven by cell-type-specific edges: **BCL6→PRDM1** (B-cell-specific), **IRF8→IL12B** (DC-specific), **RORC→IL17A** (Th17-specific)
   - Tabula Sapiens immune with 15 cell types: CSSI-max significantly outperforms pooled (Wilcoxon *p* = 2.4×10⁻⁸)
   - Scaling with *N* (from Table S6): AUROC CSSI vs pooled: 200: 0.935 vs 0.860; 500: 0.998 vs 0.932; 1000: 1.000 vs 0.972; 2000: 1.000 vs 0.989; 5000: 1.000 vs 1.000
8. **Real attention-matrix validation (Supp Note 11.4; Table S7).** Apply CSSI to Geneformer V2-316M attention on 497 DLPFC brain cells. Pooled all-layer baseline AUROC = 0.543; best pooled layer L13 = 0.694; early layers near chance. Per-layer ΔAUROC (CSSI best − pooled, from Table S7):
   - L8: +0.060 (maximum)
   - L10: +0.033
   - L5: +0.019
   - L6: +0.025
   - L9: +0.016
   - L11: +0.021
   - L12: +0.011
   - L15: +0.009
   - L13–L14: ≈ 0.000 (already maximal)
   - CSSI on real attention localises layer-specific signal; maximum ΔAUROC ≈ +0.060 at L8.
9. **SCENIC-style comparison (Fig. S23B).** CSSI-max, CSSI-mean, and SCENIC-style aggregation perform equivalently (CSSI ≠ standard SCENIC — it is a complementary scoring rule).
10. **Per-edge FDR control (Fig. S23C).** CSSI maintains FDR ≤ 0.11 at K = 5–15 with 95%+ power, confirming the test family controls false-discovery at the edge level.

**Scope note.** CSSI improves *reference-edge recovery* (TRRUST / DoRothEA), **not** perturbation-outcome predictive validity. The gene-level dominance finding of Phases 1–5 is unchanged by CSSI; the two findings are complementary. CSSI addresses an attention-specific scaling failure (population-level degradation as cells are pooled across heterogeneous states), whereas gene-level dominance addresses a different kind of failure (pairwise information redundancy with univariate features).

**Code references:** `src/02_cssi_method/02_full_pipeline.py` → `compute_edge_scores()`, `compute_contrast_scores()`, `classify_edge()`; `src/02_cssi_method/04_revised_statistics.py` → `permutation_test_contrast()`, `bootstrap_stability()`; `src/02_cssi_method/08_realdata_validation.py` → real PBMC/Tabula Sapiens validation; `src/01_scaling_failure/` → scaling characterisation.

---

### Phase 7 — Biological characterisation of attention patterns (Supp Note 17; Table S12)

**Purpose:** characterise *what biological signal* attention captures. This is the phase that **refines the headline claim**: attention does encode structured biological information with clear layer-specific hierarchy, but this hierarchy is redundant with gene-level features for perturbation prediction.

**Steps:**

1. **Evaluate Geneformer V2-316M attention edges against six reference databases.** Using 2000 K562 control cells, 2000 HVGs. Databases (with pair counts in HVG context):
   - TRRUST (transcriptional regulation): 175 pairs
   - STRING ≥ 700 (PPI): 2,747 pairs
   - STRING ≥ 900 (high-confidence PPI): 1,675 pairs
   - Reactome (pathway co-membership): 238,640 pairs
   - KEGG (pathway co-membership): 24,812 pairs
   - GO Biological Process (functional co-annotation): 135,596 pairs
2. **Compute per-layer AUROC** of attention edges at each of the 18 layers against each of the 6 reference edge sets. Compare to Spearman correlation edges as baseline. Paper result (Table S12 — key rows):

| Layer | STRING≥700 | STRING≥900 | TRRUST | Reactome | KEGG | GO BP |
|---|---|---|---|---|---|---|
| **L0** | **0.640** | **0.644** | 0.558 | 0.505 | 0.530 | 0.526 |
| L1 | 0.517 | 0.501 | 0.691 | 0.501 | 0.484 | 0.501 |
| L4 | 0.546 | 0.530 | 0.686 | 0.516 | 0.525 | 0.515 |
| L8 | 0.530 | 0.519 | 0.631 | 0.509 | 0.531 | 0.534 |
| L13 | 0.574 | 0.559 | 0.708 | 0.523 | 0.548 | 0.530 |
| **L15** | 0.525 | 0.500 | **0.750** | 0.521 | 0.555 | 0.541 |
| **L16** | 0.475 | 0.457 | 0.733 | 0.516 | **0.564** | **0.544** |
| L17 | 0.485 | 0.460 | 0.664 | 0.516 | 0.539 | 0.541 |
| Correlation | 0.562 | 0.559 | 0.649 | 0.514 | 0.541 | 0.513 |

3. **Layer-specific specialisation (Supp Note 17).** Test depth correlation per database via Spearman (all six survive BH at α = 0.05):
   - **PPI signal peaks at L0 and decreases with depth**: STRING ≥ 700 ρ = **−0.608**, *p*_raw = 0.0075, *q*_BH = 0.011; STRING ≥ 900 ρ = −0.581, *q*_BH = 0.014.
   - **Transcriptional regulatory signal increases with depth, peaks at L15**: TRRUST ρ = **+0.511**, *q*_BH = 0.030; AUROC 0.750 at L15 (best).
   - **Functional co-annotation increases with depth, peaks at L16**: GO BP ρ = **+0.846**, *q*_BH < 10⁻⁴; KEGG ρ = +0.564 at L16; Reactome ρ = **+0.731**, *q*_BH = 0.001.
   - **PPI and regulation are anti-correlated across layers**: STRING ≥ 900 vs TRRUST ρ = −0.546 (*p* = 0.019); STRING ≥ 700 vs TRRUST ρ = −0.445 (*p* = 0.064, marginal). KEGG, GO BP, Reactome are strongly positively correlated with each other (ρ = 0.75–0.89) and with TRRUST (ρ = 0.35–0.45), forming a coherent "functional/regulatory" cluster distinct from the PPI signal.
4. **Partial correlation controlling for expression similarity.** Compute partial correlation between attention edge membership and reference-database membership, controlling for Spearman expression correlation. Paper result:
   - **TRRUST signal is robust**: partial *r* = **0.353**, *p* = 1.1×10⁻¹¹ (baseline *r* = 0.363). **97% of the attention–TRRUST correlation is retained** after controlling for pairwise expression similarity.
   - GO BP: partial *r* = 0.846, *q*_BH < 10⁻⁴ (89% retained)
   - KEGG: partial *r* = 0.831, *q*_BH < 10⁻⁴ (72% retained)
   - **Reactome: partial *r* = 0.015, *p* = 0.12 — non-significant after expression control. Confirmed as a null.**
5. **Top-edge enrichment analysis.** Fisher's exact test for overlap between the top-1,000 highest-attention edges (at each of 5 representative layers) and each reference database. TRRUST excluded due to low base rate (0.009% yielding 0 overlap at all layers). BH correction across 30 tests (5 layers × 3 databases × 2 tails):
   - **L0 (early, PPI-related):** significant KEGG OR = 3.38 *q* < 10⁻¹⁰; GO BP OR = 1.47 *q* = 0.001; Reactome OR = 1.25 *q* = 0.028.
   - **L17 (late, regulation-related):** strongest enrichment across all databases — Reactome OR = **1.99** *q* < 10⁻¹⁶; KEGG OR = **3.71** *q* < 10⁻¹²; GO BP OR = **2.12** *q* < 10⁻¹³.
   - **L10 (mid-depth):** no significant enrichment (all OR ≈ 1.0).
   - Enrichment at periphery (L0 and L17) with a dead zone at mid-depth supports the layer-specialisation interpretation.
6. **Refined interpretation (paper, final paragraph of Supp Note 17; CRITICAL FOR CORRECT USAGE):**
   - Attention patterns capture a **hierarchy of biological signals with layer-specific organisation**: L0 preferentially encodes physical PPI; deeper layers progressively encode transcriptional regulation and functional co-annotation.
   - This hierarchy is **real** — it survives pairwise expression control (97% of TRRUST signal retained) and is statistically robust across all six databases after BH correction.
   - **However**, acknowledging this hierarchy does not contradict the main finding that attention provides **no incremental value over gene-level features for perturbation prediction** (Phase 2).
   - The key distinction is between the confound controls: the **partial correlations here remove pairwise expression similarity**, whereas the **incremental-value analysis removes gene-level features (variance, mean expression, dropout rate)**.
   - **Gene-level features, not pairwise co-expression, are the dominant confound.**
   - **Thus the title claim "attention captures co-expression rather than causal regulation" is more precisely stated as:** attention captures biologically structured signals, including regulatory ones, but these signals are entirely redundant with gene-level features and provide no unique information for predicting the functional consequences of genetic perturbations.

**Code reference:** `src/15_biological/02_biological_characterization.py` → pathway enrichment, TF motif analysis, per-layer AUROC vs reference databases.

---

### Phase 8 — Boundary-condition audits

These are correlation-based or semi-independent analyses that characterise the broader evaluation landscape and rule out alternative explanations for the null findings. They are not primary tests — they are the quality-control scaffold that makes the null findings interpretable.

#### 8a — Detectability phase diagrams (Supp Note 4; Figs. S8–S9)

**Purpose:** characterise the theoretical sample complexity needed to detect attention-like vs intervention-like signals, and calibrate to real data. Source: Donoho & Jin 2004 (higher criticism).

**Method:**
- Closed-form detectability formula for signal with effect size |μ|, noise scale σ, tail inflation factor τ:
  ```
  n* = ((z_{1−α/(2m)} + z_power) · τ · σ / |μ|)²
  ```
- Compare two signal classes: **attention-like** (derived from raw attention weight aggregation) vs **intervention-like** (obtained through activation patching).
- Sub-Gaussian baseline: intervention-like signals require **only 44.4%** as many cells as attention-like signals for equivalent detectability.
- Under tail inflation: advantage progressively collapses, with relative cell ratio approaching unity when **τ > 3**.
- Robust estimation (Huber M-estimators, Huber 1964): feasible detection region expands by 37% under 10% contamination.
- Real-data calibration: bootstrap distribution of projected relative cell requirements mostly below unity, but with wide CIs (see Fig. S9).

**Verdict:** intervention-based methods should outperform attention-based methods under sub-Gaussian conditions, but the advantage depends heavily on tail behaviour of the noise distribution. Report empirical kurtosis of expression data alongside detectability projections.

**Code reference:** `src/08_detectability/`

#### 8b — Cross-tissue consistency (Supp Note 5; Fig. S10; Supp Methods)

**Purpose:** test whether TF–target effects are invariant across tissues, via invariant causal discovery (Peters et al. 2016).

**Method:**
- Apply invariant causal prediction principles to **matched TF–target panels** across immune, kidney, lung tissues (Tabula Sapiens).
- Spearman correlations across tissue pairs.
- **Bootstrap uncertainty:** 10,000 resamples. Permutation-based significance: 5,000 permutations.
- Paper result: Spearman correlations **range −0.44 to 0.71** across 6 pair-granularity comparisons. **Only 2/6 survive BH at α = 0.05.**
- Limited transferability consistent with known tissue-specificity of gene regulation. Negative correlations (e.g., kidney–lung ρ = −0.44) suggest either genuine context-dependent regulation or tissue-specific technical confounds. Distinguishing these requires matched protocols across tissues (currently absent).
- **Important caveat:** batch-leakage analysis (Phase 8e) confirms technical covariates are recoverable from edge features — this may partly explain the low cross-tissue consistency.

**Code reference:** `src/_debug/` and `src/08_detectability/` (supporting material for Supp Note 5).

#### 8c — Cross-species ortholog transfer (Supp Note 7; Figs. S14–S15; Table S4)

**Purpose:** stress-test whether mechanistic signals generalise across species via ortholog-based edge transfer.

**Method:**
- Compare **correlation-based edge scores** computed independently in human lung (Tabula Sapiens, 65,847 cells) vs mouse lung (Krasnow Smart-seq2, 9,409 cells, Travaglini et al. 2020).
- 53,482 one-to-one orthologs; 61 shared transcription factors.
- Human data subsampled to 10,000 cells (matching computational constraints).
- Filter: edges with |ρ| < 0.05 discarded → **25,876 matched TF–target edges**.
- Compute Spearman ρ between human and mouse edge score distributions, sign agreement, and top-*k* overlap.

**Results:**
- Spearman ρ (human vs mouse edge scores) = **0.743**, *p* < 10⁻³⁰⁰
- Sign agreement: 88.6% overall, rising to **100% for edges with |ρ| > 0.4 in both species**
- Top-*k* overlap enrichment (Table S4):
  - Top-100: 26 observed vs 0.1 expected = **484× fold enrichment**
  - Top-500: 153 vs 1.3 = 114×
  - Top-1000: 289 vs 5.4 = 54×
  - Top-5000: 1,094 vs 134.2 = 8.2×
- **Per-TF conservation is highly non-uniform** (Fig. S15, Supp Note 7):
  - High conservation (ρ > 0.85): **lineage-specifying TFs** — XBP1 (ρ = 0.90), EPAS1 (0.89), ERG (0.88), NKX2-1 (0.81)
  - Low / negative conservation (ρ < 0.15): **signaling-responsive TFs** — CTNNB1 (0.01), HIF1A (0.10), STAT1 (0.06), CEBPB (0.13)
  - 599 "fragile" edges (sign-discordant between species) enriched for immune-cell-specific RUNX3 targets with species-divergent expression

**Verdict and recommendation:** ortholog-based edge transfer should be **stratified by TF class**. Lineage-specifying transcriptional programs can be transferred between species with high confidence; signaling-responsive and composition-dependent edges require species-specific validation.

**Code reference:** `src/09_ortholog_transfer/`

#### 8d — Pseudotime directionality audit (Supp Note 8; Figs. S16–S17; Supp Table S10 §7)

**Purpose:** test whether attention-derived edges encode directionally consistent lag-based temporal signal along pseudotime, which would support a causal interpretation.

**Method:**
- **Diffusion pseudotime** (Haghverdi et al. 2016) in 3 Tabula Sapiens immune lineages (T cell, B cell, myeloid; 20,000 immune cells).
- Per-lineage parameters: 2,000 HVGs, 30 PCA components, *k* = 15 nearest neighbours.
- **56 curated TF–target regulatory pairs** tested for lag-based directional consistency (TF-first precedes target-follows).

**Results:**
- Only **12 of 56 pairs (21.4%)** directionally consistent.
- Lineage breakdown: myeloid 6/17 (35.3%) — highest; T cell 4/24 (16.7%); B cell 2/15 (13.3%).
- Mean directionality score marginally exceeds shuffled-pseudotime null (*p* = 0.068) but NOT random-gene-pair null (*p* = 0.37).
- **After framework-level BH correction: *q* = 0.124 — not significant.**
- Failure mode distribution: most failures are "simultaneous" (26/44), followed by "target leads" (14/44), "weak signal" (3/44), "strong sign" (1/44).

**Verdict:** pseudotime should be treated as a **qualitative sanity check, not a pass/fail validator for mechanistic edges**. Perturbation-based validation and time-resolved modalities (e.g., RNA velocity) provide more direct temporal or causal evidence.

**Code reference:** `src/10_pseudotime/`

#### 8e — Batch and donor leakage audit (Supp Note 9; Figs. S18–S19; Supp Table S10 §8)

**Purpose:** quantify technical leakage (donor, batch, assay) in edge scores and define a practical blacklist mechanism.

**Method:**
- Compute **Pearson correlations** for ~8,000 TF–target pairs per tissue as edge scores.
- Train leakage classifiers (logistic regression, random forest) to predict donor / batch / assay-method labels from the edge-product feature vectors across 3 Tabula Sapiens tissue compartments.
- Define **Artifact Sensitivity Index (ASI)**: `ASI = |r_full − r_balanced| / max(|r_full|, 0.01)`, where `r_full` is the edge score computed on the full dataset and `r_balanced` is computed on a donor/batch-balanced resample.
- Flag / blacklist edges with **ASI > 0.5**.

**Results:**
- **Donor leakage is substantial.** Immune dataset AUC 0.85–0.87 (21 donors); lung 0.94–0.96 (4 donors — smaller donor pool = easier classification).
- **Assay method (10X vs Smart-seq2) is the dominant confound**, recoverable at AUC 0.96–0.99 across all tissues.
- Practical impact is dataset-dependent:
  - **Lung dataset (well-balanced, 4 donors):** aggregate edge scores stable under donor-balanced resampling (*r* = 0.997, 10.1% blacklisted).
  - **Immune dataset (imbalanced, 21 donors):** substantial instability — *r* = 0.929, **54.6% of edges blacklisted**, 17.1% sign-flipped.

**Recommendations:**
- Use **donor-stratified cross-validation**, never random CV when donor metadata is available.
- Report the generalisation gap between donor-stratified and random CV (6.6 percentage points in lung) as a built-in quality check.
- Publish the ASI distribution and blacklist fraction alongside any GRN inference result.

**Code reference:** `src/11_batch_leakage/`

#### 8f — Uncertainty calibration (Supp Note 10; Figs. S20–S21; Supp Table S10 §9)

**Purpose:** calibrate edge-score probability estimates and provide formal finite-sample coverage guarantees.

**Method:**
- Evaluate **six edge-scoring methods** — Pearson, Spearman, mutual information, partial correlation, LASSO, ensemble — against Perturb-seq ground truth from CRISPRi experiments.
- Compute Expected Calibration Error (ECE).
- Apply post-hoc calibration: **Platt scaling** (Platt 1999) and **isotonic regression** (Niculescu-Mizil & Caruana 2005).
- Construct **split conformal prediction sets** (Vovk et al. 2005) with finite-sample marginal coverage guarantees at α ∈ {0.05, 0.10, 0.20}.

**Results:**
- **Raw ECE**: 0.269 (ensemble) to 0.469 (LASSO) — all six methods severely miscalibrated.
- **Isotonic regression reduces ECE to 0.062–0.079** (4–7× reduction) without changing discrimination.
- Platt scaling produces similar improvement (slightly worse tails).
- **Split conformal prediction:** mutual information and ensemble methods achieve valid marginal coverage (≥ 95% at α = 0.05); mutual information yields 13.4% singleton prediction sets (highest precision).
- **Critical: calibrators do NOT transfer across datasets.** K562-trained calibrators applied to Shifrut T-cell dataset yield ECE 0.320–0.424, compared to 0.002–0.031 for locally trained calibrators.

**Recommendation:** GRN methods should report **calibrated scores alongside traditional rankings**. Conformal prediction sets transform the question from "which edges to call" into "which edges can be confidently called." **Calibrators must be retrained per dataset.**

**Code reference:** `src/12_calibration/`

#### 8g — TRRUST circularity sensitivity (Supp Note 13.5)

*Already detailed in Phase 3 step 4.* Brief summary: restrict TRRUST to direction-known (Activation / Repression) entries only — 4,859 of 8,427 unique pairs (58%). Global AUROC drops modestly from 0.764 to 0.736 (Δ = −0.028); per-TF median *improves* (0.695 vs 0.660). TRRUST-based conclusions are robust to restriction; the small decrease is attributable to loss of high-degree TFs (e.g., GATA1 drops from 18 to 5 positive targets), which reduces degree-driven signal.

#### 8h — HVG protocol confound test (Supp Note 14.9)

*Already detailed in Phase 5 step 5.* Critical methodological audit ruling out "forced perturbation-gene HVG inclusion drives the RPE1 attention advantage."

#### 8i — Condition-specific perturbation validation (scGPT mediation, Supp Note 6.1)

*Already detailed in Phase 5 step 6.* Counterfactual validation against four Perturb-seq datasets; only Dixit 13-day survives framework-level BH (adj. *p* = 0.042).

---

### Phase 9 — Multi-model validation (Supp Note 13; Figs. S25–S29; Supp Table S10 §12)

**Purpose:** confirm the failure to capture regulatory information is architecture-independent by testing multiple foundation-model families.

**Steps:**

1. **Geneformer V1-10M GRN recovery (Supp Note 13.1; Tables S8, S9).** Smaller Geneformer (6 layers × 4 heads) tested on DLPFC brain at 200/500/1000 cells:
   - TRRUST AUROC: 0.444 / 0.549 / 0.522 — near-random
   - DoRothEA AUROC: 0.473 / 0.486 / 0.486 — near-random
   - Bootstrap 95% CIs all include 0.5
   - Cross-model comparison: scGPT AUROC ≈ 0.46–0.51, Geneformer ≈ 0.44–0.55 — **both fail equivalently** against TRRUST and DoRothEA at matched cell counts.
2. **Attention–correlation mapping (Supp Note 13.2; Fig. S25).** For both scGPT and Geneformer V2-316M, compute Spearman and OLS *R²* between attention edge scores and (a) expression co-occurrence, (b) regulatory ground truth:
   - Attention vs expression co-occurrence: ρ = **0.31–0.42**, *p* < 10⁻⁵⁰; *R²* = 0.10–0.18
   - Attention vs regulatory ground truth: ρ = **−0.01 to −0.02**, *p* > 0.3 (null)
   - Cross-tissue analysis: *R²* < 0.02 across all tissue pairs
   - **Attention tracks co-expression, not regulation, in both model families.**
3. **Cross-fitted residualisation on expression covariates (Supp Note 13.3).** Already detailed in Phase 3.
4. **Degree-preserving null models (Supp Note 13.4; Figs. S27–S28).** Already detailed in Phase 3 (decomposition 0.50 + 0.19 + 0.07; per-TF 7/18 above chance).
5. **TRRUST circularity sensitivity (Supp Note 13.5).** Already detailed in Phase 3 / Phase 8g.
6. **Additional models tested in Supplementary Methods:**
   - **scVI** (Lopez et al. 2018) — latent-distance edges: AUROC 0.48–0.53 (near-random).
   - **C2S-Pythia** (405M parameter causal LM): AUROC 0.48–0.53 (near-random).
   - Qualitative convergence: **four foundation-model architectures (scGPT gene-token, Geneformer V1/V2 rank-tokenized BERT, scVI latent-distance, C2S-Pythia causal LM) all fail equivalently**, establishing the failure as architecture-independent.

**Code reference:** `src/14_multi_model/03_v1_attention_extraction.py` → `load_geneformer_model()`, `extract_attention_weights()`, `compute_grn_edges_from_attention()`. For adding a new model: follow the `AutoModel.from_pretrained()` pattern at `src/14_multi_model/03_v1_attention_extraction.py:40–56`.

---

### Phase 10 — Synthetic ground-truth validation (Supp Note 12; Fig. S24)

**Purpose:** confirm that the evaluation framework is internally consistent with its own theoretical predictions, using synthetic data where ground truth is known.

**Steps:**

1. **Generate synthetic single-cell expression data** using steady-state GRN dynamics with realistic noise sources: dropout (*p* = 0.1), technical noise, batch effects, heavy-tailed expression.
2. **Ground-truth network structure:** hierarchical TF–regulator–target, sparse connectivity (ρ = 0.15).
3. **Synthetic attention matrix:** `A_attention = tanh(A_true + ε_structured + ε_expression-bias)` — i.e., a noisy function of the true network plus an expression-bias component.
4. **Three confirmed predictions:**
   - (a) **Attention-based GRN recovery degrades monotonically with cell count** (*r* = 0.847 at 200 cells → *r* = 0.623 at 2000 cells), strongly correlated with expression heterogeneity (*r* = −0.94, *p* < 0.01).
   - (b) **Shapley value estimates substantially beat single-component estimates**: ρ_Shapley = 0.789 vs ρ_single = 0.412 — a **91% improvement** for recovering true interaction rankings.
   - (c) **Empirical detection performance matches theoretical predictions** from Phase 8a: *r* = 0.887, *p* < 10⁻⁶.
5. **Limitation.** Because the synthetic generator encodes the framework's theoretical assumptions by design, these experiments confirm internal consistency of the framework — real-data validation (the rest of the pipeline) provides a complementary check.

**Code reference:** `src/13_synthetic_validation/`

---

### Phase 11 — Per-TF exploratory characterisation (Supp Note 19; Table S14)

**Purpose:** hypothesis-generating analysis of whether attention-derived edge score quality varies systematically with TF biology (master regulators vs signal-dependent vs lineage-specific vs housekeeping).

**Steps:**

1. **Annotate TFs by biological class** in the Tabula Sapiens immune dataset (18 evaluable TFs total, categories: master regulator, signal-dependent, lineage-specific, housekeeping).
2. **Compute per-TF AUROC** as in Phase 1 but at the single-TF level. Paper result (Table S14, selection):
   - LEF1 (master regulator, 1 target): 0.997
   - KLF1 (master regulator, 1 target): 0.995
   - PPARG (master regulator, 10 targets): 0.900
   - FOXA1 (master regulator, 2 targets): 0.874
   - GATA1 (master regulator, 18 targets): 0.842
   - MAL (signal-dependent, 1 target): 0.860
   - EGR1 (signal-dependent, 9 targets): 0.852
   - WT1 (master regulator, 5 targets): 0.675
   - NCOA4 (signal-dependent, 1 target): 0.349
3. **Class comparison.** 9 master regulators (GATA1, PPARG, FOXA1, NKX2-5, KLF1, PAX3, TAL1, LEF1, WT1) vs 9 other TFs. Paper result: master regulators mean 0.80 ± 0.18 vs other mean 0.58 ± 0.18; permutation test *p* = 0.011 (10,000 shuffles). **Does not survive BH correction across all 11 tests (*q* = 0.12).**
4. **Severe power limitation.** **13 of 18 TFs have only a single evaluable TRRUST target in the HVG set**, meaning AUROC reflects the rank of *one gene* among ~2,000 rather than a regulon-level assessment. Restricted to the 5 TFs with ≥ 3 evaluable targets (GATA1, PPARG, EGR1, WT1, FOXF2), the master-regulator advantage is not significant (*p* = 0.30).
5. **No TF property significantly predicts AUROC** after BH correction: mean expression (ρ = −0.14, *q* = 0.66); variance (ρ = −0.20, *q* = 0.58); nonzero fraction (ρ = −0.01, *q* = 0.97); regulon size (ρ = 0.26, *q* = 0.53). Evaluable regulon size shows strongest raw trend (ρ = −0.46, *p* = 0.06, *q* = 0.33), suggesting TFs with more evaluable targets have AUROC closer to 0.5 — **regulon-level AUROC is a harder test than single-gene rank**.
6. **Verdict.** Hypothesis-generating only. The pattern is consistent with attention more reliably capturing regulatory relationships for master regulators with large, well-characterised regulons, but the small sample size (*n* = 18) and severe single-target problem preclude definitive conclusions. **Larger TF databases with greater regulon coverage — DoRothEA, ChIP-Atlas — would be needed to test this rigorously.**

**Code reference:** `src/15_biological/06_per_tf.py`

---

### Phase 12 — Statistical framework and verdict (Supp Note 16; Tables S10, S11)

**Steps:**

1. **Register every test** as a row in a machine-readable test registry with: section (Supp Note 16 organises into 37 numbered sections), hypothesis, test name, raw *p*, BH *p*, effect size (Cohen's *d*, correlation coefficient, fold-change, AUROC, or percentage improvement), sample size, significance flag. The paper's canonical registry is **Supp Table S10**, containing 153 rows — the pipeline should emit a CSV / JSON with one row per test in the same schema.
2. **Framework-level BH correction.** Apply Benjamini-Hochberg FDR correction at **α = 0.05 across all 95 confirmatory *p*-values framework-wide** (the remaining 58 tests are descriptive — effect sizes, bootstrap CIs — and are not subject to BH). After correction, **63 of 95 (66%) remain significant**.
3. **Sensitivity to family definition (Supp Note 16.1).** Run BH under three alternative family definitions:
   - (A) Primary family of 95 confirmatory tests
   - (B) Maximal family including all 153 tests
   - (C) Analysis-level family retaining one primary test per analysis (27 tests)
   Verify **12 of 17 (71%) headline inferences remain stable across all three family definitions.** All primary conclusions — attention–correlation equivalence on K562 CRISPRi, no incremental pairwise value, ablation null, L15 nested-CV result, CRISPRa underperformance, RPE1 attention advantage — remain significant under every family definition.
4. **Claim-to-evidence mapping (Supp Note 16.3, Table S11).** Build a table mapping each headline claim to its supporting analysis and key test, with the BH significance flag. The 20 headline claims to track:
   - Top-*K* scaling degradation (scaling sign test; BH sig. Yes)
   - Continuous AUROC improves (K-sensitivity; descriptive)
   - Mediation non-additivity (mediation bias lower bound; BH sig. Yes)
   - Detectability theory validated (detectability correlation *r* = 0.887; BH sig. Yes)
   - Perturbation-first AUROC > 0.5 (*t*-test all 27 conditions; BH sig. Yes)
   - Attention ≈ correlation on K562 CRISPRi (Wilcoxon *p* = 0.73; null)
   - L15 best layer, Δ = +0.040 (18-layer nested CV Wilcoxon, Bonferroni *p* = 0.017; BH sig. Yes)
   - L15 effect is small *d* = 0.22 (Cohen's *d* + bootstrap CI; BH sig. Yes)
   - No incremental pairwise value (bootstrap ΔAUROC ≤ 0; descriptive)
   - Trivial baselines outperform (paired *t*, all *p* < 10⁻¹²; BH sig. Yes)
   - CRISPRa: attention < correlation (Wilcoxon *p* < 10⁻⁶; BH sig. Yes)
   - T-cell: attention ≈ correlation (Wilcoxon *p* > 0.8; null)
   - Ablation null on regulatory heads (Wilcoxon *p* > 0.05; null)
   - Orthogonal interventions null (all |Δ| < 0.005; null)
   - Propensity-matched null (ΔAUROC ∈ [−0.000, +0.000]; null)
   - Interventions perturb representations (all max cos > 0.02; descriptive)
   - RPE1: attention > correlation, *d* = 0.47 (Wilcoxon *p* < 10⁻¹⁰; BH sig. Yes)
   - Random ablation causes drop (expanded ablation Wilcoxon *p* < 10⁻⁸; BH sig. Yes)
   - Heterogeneity improves correlation (controlled-composition Spearman ρ = +0.63; BH sig. Yes)
   - Edge–expression correlation ρ = 0.84 (BH sig. Yes)
   - Cross-species conservation ρ = 0.743 (BH sig. Yes)
5. **Emit summary statistics.**
   - Total statistical tests: 153 (95 confirmatory, 58 descriptive)
   - Significant after BH-FDR correction: 63/95 (66%)
   - Framework-level α: 0.05 with Benjamini–Hochberg correction
   - **Most robust findings** (smallest *p* after BH): scaling degradation (*p* = 10⁻³⁰⁰), CSSI synthetic validation (*p* = 2.4×10⁻⁸), heterogeneity–AUROC correlation (ρ = +0.63, *p* = 10⁻⁴), edge–expression correlation (ρ = 0.842, *p* < 10⁻⁵⁰), degree-preserving null (*z* = 3.63, *p* < 0.005), perturbation sensitivity (all 27 conditions *p* < 0.005), RPE1 attention advantage (*d* = 0.47, adj *p* < 10⁻⁸).
   - **Key null findings** (confirmed absent): pseudotime directionality (adj *p* = 0.124), Replogle CRISPRi baseline (AUROC 0.511, *p* = 0.32), primary AUROC 0.696 (*p* < 10⁻⁴), all 27 sensitivity conditions (AUROC 0.62–0.76, all *p* < 0.005), real-data CSSI improvement (adj *p* = 0.053), attention ≈ correlation in K562 CRISPRi (*p* = 0.73), **no incremental pairwise value** (ΔAUROC ≤ 0.002).

6. **Emit the pass/fail verdict.** A GRN "passes" this pipeline — and is cleared for downstream regulatory interpretation — **only if all four of the following hold:**
   - **(a)** attention edges significantly outperform **every** gene-level baseline (variance, mean expression, 1−dropout) at *p* < 0.05 BH-corrected (Phase 1 pass). None of scGPT / Geneformer / scVI / C2S-Pythia pass this on any of the five tested cell type × modality combinations.
   - **(b)** ΔAUROC for (gene+edge) − (gene-only) is significantly > 0 with lower 95% CI bound > 0 under **both** linear and nonlinear models and **all three** split protocols (cross-pert, cross-gene, joint); and the metric-robust extension (AUPRC, top-*k* recall) confirms non-null (Phase 2 pass). Largest observed ΔAUPRC in scGPT / Geneformer across all tested combinations: +0.009 (joint / GBDT) — less than 4% relative, fails this criterion.
   - **(c)** Cross-fitted residualised AUROC retains ≥ 80% of raw edge AUROC above chance; and degree-preserving null *z*-score > 3 (Phase 3 pass). scGPT / Geneformer attention loses 76% of signal on residualisation (retains only 24%), fails this criterion.
   - **(d)** Top-*k* TRRUST-ranked head ablation produces monotone AUROC degradation with effect size ≥ random-head control at matched *k*; AND orthogonal interventions (uniform attention replacement, MLP ablation at regulatory layers) produce significant AUROC drops; AND intervention-fidelity diagnostics confirm all interventions materially perturb representations (hidden-state cos > 0.02). Phase 4 pass. scGPT / Geneformer fail: random-head ablation causes *larger* drops than TRRUST-ranked; MLP ablation at regulatory layers is null while MLP ablation at a random layer is significant; fidelity diagnostics confirm interventions are material (hidden-state cos 0.023–0.190) — the null is real, not a fidelity failure.

**No model tested in the source paper passes all four criteria on any of the five tested cell type × modality combinations.** A new model that passes would be the first and should be treated with publication-level scrutiny (the paper's test registry, claim-to-evidence mapping, and sensitivity analyses should all be reproduced verbatim).

**Code reference:** `src/16_statistical_framework/01_statistical_audit.py` → `build_registry()`, `run()`; `src/16_statistical_framework/02_multiplicity_sensitivity.py`.

---

## 7. Parameters

| Parameter | Default (primary K562) | Alternate / scan range | Used in |
|---|---|---|---|
| `n_ctrl` (control cell count) | **2000** | {500, 2000, 10000} (Supp Note 6.3); baseline uses 500 | Phase 0, all edge extraction |
| `hvg` (highly-variable gene count) | **2000** (K562), **3309** (RPE1 with forced pert. genes) | {1000, 2000, 5000}; RPE1 restricted = 2000 | Phase 0, Phases 1–5 |
| `lfc_thresh` (DE log-fold-change) | **0.5** (primary) | {0.1 (baseline), 0.25, 0.5, 1.0} | Phase 1, §3.2 |
| `analysis_layer` (Geneformer V2-316M) | **L13** (pre-specified DLPFC); **L15** (nested-CV selected) | 18-layer profile as secondary | Phases 1–4 |
| `de_test` | **Welch *t*** (primary); Mann-Whitney *U* (baseline config) | — | Phase 1 |
| `n_splits` (CV folds) | **5** | 3–10 | Phase 2 |
| `split_protocol` | cross-perturbation (primary) | cross-gene, joint cross-gene × cross-pert | Phase 2 |
| `bootstrap_n` | **200** (per-test); **10,000** (headline CIs) | 100–10,000 | All CIs |
| `ablation_k_trrust_ranked` | {5, 10, 20, 50} | 1–`n_heads` | Phase 4 |
| `ablation_k_composite` | {5, 10} | — | Phase 4 |
| `ablation_k_bottom` | {5, 10} | — | Phase 4 |
| `ablation_k_random` | {10, 20, 50} | — | Phase 4 |
| `intervention_cells_fidelity` | **2000** | — | Phase 4 step 5 |
| `cssi_K` (strata count) | **5–7** (optimal) | scan {2,…,20} | Phase 6 |
| `cssi_knn_k` (k-NN neighbours) | **15** | 5–50 | Phase 6 |
| `cssi_aggregator` | CSSI-max (primary) | CSSI-mean, CSSI-range, CSSI-deviation | Phase 6 |
| `cssi_null_n` (null permutations) | 100+ | — | Phase 6 step 4 |
| `leiden_resolution` | **1.0** | 0.1–3.0 | Phase 6 |
| `propensity_k` (matches per DE+) | **5** | 1–10 | Phase 3 |
| `propensity_smd_threshold` | **0.1** | — | Phase 3 step 5 |
| `residualizer` | OLS (primary); GBDT (secondary) | — | Phase 3 steps 1–2 |
| `degree_null_n_curveball` | **200** | — | Phase 3 step 3 |
| `degree_null_n_labelshuffle` | **1000** | — | Phase 3 step 3 |
| `fdr_alpha` (framework-level) | **0.05** | 0.01–0.1 | Phase 12 |
| `fdr_family_primary` | **95** confirmatory | 153 maximal, 27 analysis-level | Phase 12 sensitivity |
| `attn_implementation` (HF models) | **"eager"** | — | Phase 0 (non-negotiable) |
| `output_attentions` | **True** | — | Phase 0 |
| `detectability_alpha` | 0.05 | 0.01–0.1 | Phase 8a |
| `detectability_power` | 0.80 | — | Phase 8a |
| `detectability_tau` (tail inflation) | 1.0 (sub-Gaussian) | scan 0.5–5 | Phase 8a |
| `ortholog_filter_abs_rho` | **0.05** | — | Phase 8c |
| `pseudotime_knn_k` | **15** | — | Phase 8d |
| `pseudotime_n_pca` | **30** | — | Phase 8d |
| `asi_blacklist_threshold` | **0.5** | 0.1–1.0 | Phase 8e |
| `calibration_methods` | Platt, isotonic | — | Phase 8f |
| `conformal_alpha` | {0.05, 0.10, 0.20} | — | Phase 8f |

**Full sensitivity scan:** **27 parameter combinations** at 3 LFC × 3 control-cell counts × 3 HVG counts (Supp Note 6.3, Fig. S12). All 27 configurations yield AUROC significantly above chance (*p* < 0.005). AUROC ranges 0.619–0.756. **All 27 configurations reproduce the gene-level dominance finding** — variance, mean expression, dropout rate outperform attention under every parametrisation.

## 8. Validation

A successful pipeline run must reproduce the paper's **sanity-check signatures** on the reference datasets before any new-model claim is trusted. If any signature deviates beyond the tolerance, the adapter or the environment is wrong.

1. **Replogle K562 CRISPRi primary config** (Phase 1): attention AUROC **0.704**, correlation AUROC **0.703**, variance **0.881**, mean expression **0.851**, 1−dropout **0.866**, TF out-degree **≈ 0.50** (counter-example). Tolerance: ±0.01.
2. **Incremental value** (Phase 2): ΔAUROC gene+attention vs gene-only ∈ [−0.005, +0.001]; ΔAUROC gene+correlation vs gene-only ∈ [−0.005, +0.001]. Hard-gen protocols all within same bounds. ΔAUROC > 0.01 indicates a data leak between train and test folds (most likely a target-gene leak that GroupKFold-by-target should catch).
3. **Residualisation** (Phase 3): attention cross-fitted residual AUROC **0.54** (from baseline 0.66, loses ~76%); correlation residual **0.62** (from 0.63, loses ~9%). Tolerance: ±0.02.
4. **Degree-preserving null** (Phase 3): decomposition 0.50 + 0.19 + 0.07 = 0.757; *z* = 3.63, *p* < 0.005. Per-TF CIs-above-0.5 count: **7/18 (39%)**.
5. **CSSI on DLPFC brain** (Phase 6): top-*K* TRRUST F1 improvement up to **1.85×** at K = 5–7. Null tests (shuffled/random/gene-permuted labels) produce AUROC *below* pooled baseline — no false-positive inflation. Maximum ΔAUROC ≈ +0.060 at L8 on Geneformer V2-316M real attention.
6. **Scaling signature** (Phase 6 diagnostic / main text §2.2): top-*K* F1 degrades monotonically in **9/9** scGPT kidney tier × seed runs from 200 → 1000 cells (sign test *p* = 0.002). If the continuous-AUROC metric is used instead, the signature reverses (AUROC 0.858 → 0.925 → 0.934; 0/9 runs degrade). **The two metrics disagree — this is the key finding of §2.2 and Supp Note 1; a pipeline that reports only one metric is broken.**
7. **Causal-ablation null** (Phase 4): top-5 TRRUST-head ablation AUROC within **±0.003** of baseline 0.7035. Random-head ablation at *k* = 20 causes a **larger** drop (0.6964, *d* = 0.33, *p* < 10⁻⁸). Drop > 0.01 in the TRRUST-ranked condition suggests the ablation implementation is not actually masking heads — **always verify with the intervention-fidelity diagnostic**. Expected max hidden-state cos distance for uniform reg top-5 ≈ **0.057**; for MLP zero L13–L15 ≈ **0.190**; for MLP zero L15 ≈ **0.043**. Logit cos for TRRUST-ranked heads should be 23× larger than random heads at matched dose.
8. **Orthogonal intervention inversion** (Phase 4): MLP zero at L15 / L13–L15 (regulatory layers) produces **exactly baseline** AUROC; MLP zero at random layer L8 produces the *largest* significant drop (0.6982, Δ = −0.0053, *d* = −0.27, *p* < 10⁻⁴). **If both L15 and L8 MLP ablations are null, your MLP ablation is not actually zeroing the FFN output.**
9. **HVG protocol confound test** (Phase 5 / 8h): RPE1 restricted to 2000 HVGs (no forced pert. genes) produces **larger** mean per-perturbation attention advantage (from −0.024 to +0.168; Wilcoxon *p* < 10⁻⁴⁶). Correlation AUROC should drop 0.723 → 0.593; attention should rise 0.699 → 0.762. If attention drops under restriction, the adapter is mis-tokenizing.
10. **Cross-species conservation** (Phase 8c): human–mouse lung Spearman ρ = **0.743**; sign agreement **88.6%**; top-100 fold enrichment **484×**; lineage-specifying TFs ρ > 0.85; signaling-responsive TFs ρ < 0.15. Deviation indicates ortholog mapping error.
11. **Framework-level** (Phase 12): 153 tests run, 95 confirmatory, BH-corrected at α = 0.05 yields **63 significant (66%)**. Deviation > 5 tests indicates inconsistent BH application (most likely wrong family definition).
12. **Calibration** (Phase 8f): raw ECE 0.269–0.469; isotonic-calibrated ECE 0.062–0.079 (4–7× reduction). Conformal α = 0.05 achieves ≥ 95% marginal coverage; mutual information achieves 13.4% singleton sets.
13. **Biological characterisation (Phase 7 / Supp Note 17):** PPI signal peak at L0 (STRING ≥ 700 AUROC ≈ 0.640); regulatory signal peak at L15 (TRRUST AUROC ≈ 0.750); partial correlation retains 97% of TRRUST signal after expression control. PPI vs TRRUST cross-layer ρ ≈ −0.55 (anti-correlated).

## 9. Known pitfalls

1. **Layer cherry-picking.** The most common way to fake a positive result is to profile all layers, pick the best, and report it. The paper pre-specifies **L13** for Geneformer V2-316M on DLPFC *before* touching Replogle; under strict 5-fold nested CV, **L15** is then selected in all 5 folds with Bonferroni *p* = 0.017. Do not deviate from this protocol. If you must profile all layers, report the full profile and use nested cross-validation or Bonferroni correction across layers.
2. **HVG leakage and the HVG protocol confound.** If perturbation genes are force-included in the HVG set (as required for RPE1 at 3,309 genes), they dominate the covariate distribution. Always run the HVG protocol confound test (Phase 5 / Phase 8h) to verify this does not drive the attention advantage. The paper's result — that attention advantage *increases* under the restricted protocol — is a positive rebuttal of HVG leakage as the driver; any new analysis should report this explicitly.
3. **`attn_implementation` misconfiguration.** HuggingFace defaults to a fused SDPA kernel that **silently ignores** `head_mask`. Phase 4 results will be identical to baseline and you will wrongly conclude "no effect." **Always set `attn_implementation="eager"` for Phase 4, and always run Phase 4 step 5 (intervention-fidelity diagnostics) to catch this.** Expected hidden-state cos distance > 0.02 across six intervention conditions — if all are zero, the mask is not being applied.
4. **TRRUST circularity.** TRRUST edges are curated partly from co-expression studies. A naïve AUROC against TRRUST can be inflated by co-expression signal. Run the direction-restricted sensitivity (Phase 3 step 4 / Phase 8g) to bound the circularity contribution. Expected: global AUROC drop Δ ≈ −0.028; per-TF median improves; conclusions stable.
5. **Top-*K* vs continuous AUROC metric dependence.** Top-*K* F1 is extremely sensitive to reference sparsity (TRRUST has ~51 edges among ~3.7M candidate pairs in a typical HVG set — positive rate < 0.002%). Continuous AUROC is much more stable. **Always report both** — they can and do disagree on the *sign* of scaling effects (see Validation signature #6; main text §2.2).
6. **Correlation baseline from same cells.** The correlation baseline **must** be computed from the same `n_ctrl` cells used for attention extraction. Sample-size differences confound every attention-vs-correlation comparison. The paper's `03_perturbation_validation/` scripts enforce this via the shared extraction interface in `src/shared/01_unified_extraction.py`.
7. **Propensity-score covariate balance.** After matching, standardised mean differences (SMD) on all four covariates — mean expression, variance, dropout, `tgt_n_affected` — must drop below 0.1. If any SMD remains > 0.1 after matching (the paper's tgt_n_affected remains at 0.71 — target-degree is hard to match), the matched-set AUROC is confounded in that dimension and should be interpreted with a caveat. See Fig. 6C for expected post-matching balance.
8. **Geneformer tokenisation length.** Rank-based tokenisation truncates to a fixed context (typically 2048). Highly-expressed housekeeping genes dominate the first tokens; low-expression regulators may be truncated. When porting to a new BERT-style single-cell model, verify perturbation genes are *not* truncated — store the post-truncation gene set and restrict Phases 1–5 to that set. For Geneformer V2-316M at 2000 HVGs, 1,941 / 2,000 genes are in the token vocabulary (Supp Note 18); only those are evaluable. **Always report the post-tokenisation evaluable set alongside the HVG set.**
9. **Single-component mediation bias** (Supp Note 3; main text §2.6). Standard activation-patching protocols implicitly assume mediator additivity. Real attention circuits violate additivity in **62.5% of run-pairs** in the paper's cross-tissue analysis (10 of 16 run-pairs have positive lower bound on aggregate non-additivity, median A_lb / |TE| = 0.725). If you use mediation analysis as part of Phase 4, report the residual non-additivity ratio A_lb / |TE| alongside the primary effect; otherwise the ranking certificate is fragile (Fig. S7 shows mean certified pair fraction dropping from 0.0669 at λ=1 to 0.0032 by λ≥3). **Shapley-value decomposition achieves 91% improvement over single-component estimates** (Phase 10).
10. **Scaling-run seed reconstruction.** TRRUST F1 is strongly anti-correlated with the number of *observed* cell types in the sampled cells (Spearman ρ = −0.76, *p* = 4.3×10⁻⁵). When sampling cells for extraction, fix and record the random seed — otherwise scaling results are non-reproducible.
11. **Donor / batch leakage** (Phase 8e). Donor identity is recoverable from edge-product features at AUC 0.85–0.96; assay method (10X vs Smart-seq2) at 0.96–0.99. **Use donor-stratified cross-validation**, never random CV, when donor metadata is available. Report the generalisation gap. Blacklist edges with ASI > 0.5 (54.6% of immune edges exceed this in the paper's audit).
12. **Calibration does not transfer** (Phase 8f). K562-trained calibrators applied to Shifrut T cell data produce ECE 0.32–0.42 vs 0.002–0.031 locally. **Retrain calibrators per dataset.**
13. **Pseudotime is not a causal validator** (Phase 8d). Only 21.4% of curated TF–target pairs are directionally consistent under diffusion pseudotime, failing BH correction (adj *p* = 0.124). Do not use pseudotime as a pass/fail criterion for regulatory edges — it is at best a qualitative sanity check.
14. **Pairwise expression similarity vs gene-level features are different confounds** (Phase 7, Supp Note 17). Partial correlation controlling for Spearman expression similarity retains 97% of attention–TRRUST signal — attention is NOT merely a linear function of pairwise co-expression. **But** the incremental-value analysis controlling for gene-level features (variance, mean, dropout) produces ΔAUROC = 0. **Gene-level features, not pairwise co-expression, are the dominant confound.** A pipeline that only runs partial correlation will mis-state the finding.
15. **Cross-species ortholog transfer requires TF-class stratification** (Phase 8c). Lineage-specifying TFs (XBP1, EPAS1, ERG, NKX2-1) transfer at ρ > 0.85; signaling-responsive TFs (CTNNB1, HIF1A, STAT1, CEBPB) at ρ < 0.15. A naïve global conservation estimate (ρ = 0.743) hides this per-TF heterogeneity and can mislead downstream use.
16. **Controlled-composition scaling vs heterogeneity.** The "attention-specific scaling failure" is *not* a sample-size effect; it is a cell-state heterogeneity effect. Controlled-composition experiments (Fig. S5) show AUROC stable under fixed composition with varying *N* (ρ = −0.05, *p* = 0.82) and AUROC *improving* with increasing heterogeneity at fixed *N* = 500 (ρ = +0.63, *p* = 10⁻⁴). CSSI (Phase 6) is the targeted remedy.

## 10. Quick-start for new-model evaluation

To apply this pipeline to a new single-cell model `NEWMODEL`:

1. **Implement the Phase 0a adapter** following the contract in §6 Phase 0a. Export `extract_attention(adata, model, n_ctrl=2000) → (attention_tensor, gene_token_map, gene_level_features, correlation_edges)`.
2. **Implement the Phase 0b value-weighted adapter** if your model has a well-defined context layer. Export `extract_value_weighted(model, adata, n_ctrl=2000) → (context_tensors, vw_edges)`. If the model does not expose Q, K, V separately, skip Phase 0b and note the omission.
3. **Validate the adapter** on Replogle K562 and reproduce the Phase 1 sanity signature (attention AUROC ≈ 0.70, variance AUROC ≈ 0.88, gene+edge incremental ΔAUROC ≈ 0) using `NEWMODEL`'s own gene vocabulary. If AUROC is implausible (e.g., > 0.95 on perturbation-first) before any confound test, there is a data leak.
4. **Pre-specify the analysis layer.** Run `NEWMODEL` + Phase 6 CSSI recovery metric on **DLPFC brain data** and select the layer with the highest top-*K* TRRUST F1. Freeze the choice. Do not re-select on Replogle.
5. **Run Phases 1–5 on Replogle K562** at the default parameterisation (§7). Save the test registry and the pass/fail flags per criterion.
6. **Run Phase 6 (CSSI)** on DLPFC brain and verify the 1.85× top-*K* F1 improvement and the null-stress signature (shuffled/random labels produce AUROC below pooled baseline).
7. **Run Phase 7 (biological characterisation)** on the 6 reference databases. Report the per-layer AUROC profile. Identify which layer peaks for PPI, regulation, and co-annotation. Compute partial correlation controlling for expression similarity. This is the only phase that can produce a *positive* signal for a genuinely novel model — "attention encodes layer-specific biological hierarchy" — while still failing Phases 1–5.
8. **Run Phase 5 (cross-context replication)** on at least Replogle RPE1 CRISPRi and Adamson K562 CRISPRa. Include the HVG protocol confound test.
9. **Run Phase 8 (boundary-condition audits)** — all 9 sub-phases (8a–8i). The donor leakage audit (8e) and the batch blacklist are mandatory for any dataset with donor metadata.
10. **Run Phase 9 (multi-model comparison)** placing `NEWMODEL` alongside scGPT, Geneformer V1/V2, scVI, and C2S-Pythia on matched DLPFC data.
11. **Run Phase 10 (synthetic ground-truth)** to confirm the framework's theoretical predictions hold for `NEWMODEL` and compute the Shapley-vs-single-component contrast.
12. **Run Phase 11 (per-TF exploratory)** if the tested cell type has ≥ 50 TFs with ≥ 3 evaluable regulon members in the HVG set. Report per-TF AUROC by TF class annotation. Otherwise mark this phase as underpowered and do not report conclusions.
13. **Run Phase 12** and emit the pass/fail verdict with the complete claim-to-evidence mapping.
14. **If the verdict is "pass" on all four criteria,** you have the first single-cell foundation model whose attention patterns carry genuine regulatory signal beyond gene-level confounds. This would be a publishable positive result; reproduce the 153-test registry verbatim and submit alongside.
15. **If the verdict is "fail"** — the expected outcome based on scGPT, Geneformer V1/V2, scVI, and C2S-Pythia — **do not claim the model's attention encodes regulatory structure**. You may claim the model's attention encodes layer-specific biological hierarchy (Phase 7) if the 6-database characterisation succeeds, but you must explicitly note this is redundant with gene-level features for perturbation prediction. The pipeline's diagnostic value is in its null findings; a clear, well-documented null is more valuable than a confounded positive.
