# Pipeline: Residual-Stream Spectral Geometry of Single-Cell Foundation Models

## 1. Overview

This pipeline decodes the **geometric structure** of a single-cell transformer's residual-stream representations and maps it to biological organisation. Given a pretrained single-cell model, it extracts per-layer gene embedding matrices, performs layer-wise singular value decomposition (SVD), and tests whether the resulting spectral axes encode specific biological concepts: **subcellular localisation** (the secretory pathway), **protein–protein interaction networks** (with quantitatively graded fidelity to STRING confidence), and **transcriptional regulatory relationships** (early layers preserve specific TF→target edges; deep layers compress to categorical TF-vs-target distinction). It also traces **cell-type attractor dynamics** — concretely, the convergence of germinal-centre master regulators toward the B-cell identity anchor PAX5 across transformer depth.

The methodology is **complementary** to attention-based interpretability. In the source paper, scGPT's residual-stream geometry reveals co-expression-*independent* regulatory signal in SV5–SV7 at early layers that the companion attention analysis (arxiv 2602.17532; pipeline `attention-grn-extraction-and-evaluation.md`) could not recover — while the attention analysis found that attention edges track co-expression rather than causal regulation. Both pipelines are needed: attention is the right lens for pairwise regulatory inference at late layers, residual-stream SVD is the right lens for early-layer spectral structure.

The pipeline is **model-agnostic**: a Phase 0 adapter must expose per-layer residual-stream gene embeddings and a stable gene-index map. All downstream phases consume a standard `[n_layers, n_genes, d_model]` tensor. The core evaluation is driven by an **autonomous two-agent hypothesis-screening loop** (executor + brainstormer) that over 63 iterations tested 183 hypotheses across 13 families, retiring any branch after two consecutive negative results — the pipeline therefore includes the loop infrastructure as Phase 10, so coding agents can extend or replicate the methodology rather than just re-run the fixed test battery.

**Key headline findings from the source paper** (scGPT, 12 layers × 4,803 genes × 512 dim, Tabula Sapiens immune):
- **SV1** (dominant axis) encodes the cellular secretory pathway with layer-specific ordering mitochondria → ER lumen → extracellular (extracellular-pole GO:0005615 OR = 6.37, *p* = 2.6×10⁻⁴; gene-label-shuffle empirical *p* = 0.004 at *N* = 500).
- **14.4-fold effective-rank collapse** across 12 layers (ER 23.6 → 1.6; SV1 variance fraction 19% → 77% on the 195-gene submatrix and 53.7% → 93.4% on the full vocabulary).
- **SV2–SV4** encode PPI networks with **monotonically graded** fidelity to STRING confidence quintiles (Spearman ρ = **1.000**, *n* = 5, *p* = 0.017; quintile-averaged co-pole *z*-scores increase from 1.00 at Q1 to 5.02 at Q5). PPI signal is driven by physical binding (low-GO-similarity but high-STRING pairs retain *z* ≈ 2.66 at 12/12 layers) — not by shared function.
- **SV5–SV7** encode transcriptional regulatory relationships with **co-expression-independent** signal at early layers. Joint SV2–SV7 TF-vs-target AUROC = **0.744** (mean across 12 layers; max 0.789 at layer 3; all 12 layers *p* < 0.01). Edge-level AUROC peaks at **0.602** at layer 0 and decays monotonically with depth (Spearman ρ = −0.958, *p* = 9.5×10⁻⁷); layers 0–8 significant, layers 9–11 not.
- **Repression > activation** in geometric separability (SV5–SV7 Δ = +0.021; SV2–SV4 Δ = +0.084).
- **B-cell attractor dynamics:** BATF (rank 1,510 → 189 across layers 0 → 11; distance to PAX5 18.3 → 5.1; ρ = −0.972), BACH2 (rank 611 → 146; 17.4 → 5.2; ρ = −0.844), PRDM1 converges alongside (ρ = −1.000). GC-TF–plasma-TF angle at B-cell anchor increases 77° (L0) → 94° (L11) — programs diverge toward orthogonality. BCL6 is geometrically isolated in a **metabolic compartment** (10/20 nearest neighbours are NAMPT, GLUL, PFKFB3, etc.; zero B-cell markers across all 12 layers). Intrinsic dimensionality of B-cell marker embeddings drops monotonically 8.2 → 5.1 (ρ = −0.951, *p* < 0.0001); T-cell and myeloid markers show no compression — compression is lineage-specific.
- **Cell-type marker clustering**: AUROC **0.851** in the full 512-dim embedding, surviving L2-norm regression.
- **Critical negative findings** (essential for interpreting positives): persistent homology signals vanish under degree-preserving graph-rewiring nulls (0/24 tests significant vs 11/12 under feature-shuffle), cross-model alignment fails a permutation test (*p* > 0.3) while Geneformer static embeddings independently encode STRING PPI (*p* = 7.8×10⁻¹²⁷), feed-forward loop geometry is null after permutation-corrected baselines, GO Biological Process terms do not load on SV2 (0/591 significant).

## 2. Source

- **Paper:** Kendiukhov, I. (2026). *Multi-Dimensional Spectral Geometry of Biological Knowledge in Single-Cell Transformer Representations.* arXiv:2602.22247 [q-bio.GN; cs.AI; cs.LG], 24 Feb 2026. 14 pages (12 pages main + 2 pages supplementary tables; 6 main figures; 6 supplementary tables; 13 hypothesis families; 63 iterations of an autonomous screening loop).
- **Local PDF:** `references/2602.22247_Kendiukhov_spectral_geometry.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/topology-biomechinterp1
- **Pinned commit:** `42ac179697c4dd938c57a40142356e8763b9fcc1` (2026-02-24, "Update README.md")
- **Local clone:** `repos/topology-biomechinterp1/` (see `repos/topology-biomechinterp1/PINNED_COMMIT.txt`)
- **Direct predecessor:** Kendiukhov 2026, arXiv:2602.17532 — the attention-based interpretability evaluation. See `pipelines/attention-grn-extraction-and-evaluation.md`. This pipeline is the residual-stream counterpart; both are needed for a complete geometric audit of a single-cell transformer.
- **Theoretical / methodological references:** Park et al. 2023 (linear representation hypothesis), Elhage et al. 2022 (superposition), Nanda et al. 2023 (depth-dependent processing stages in transformers), Alain & Bengio 2017 (linear classifier probes), Kornblith et al. 2019 (centered kernel alignment), Roy & Vetterli 2007 (effective rank), Facco et al. 2017 (TwoNN intrinsic dimensionality), Sharma et al. 2024 (layer-selective rank reduction), Cui et al. 2024 (scGPT), Theodoris et al. 2023 (Geneformer), Jones et al. 2022 (Tabula Sapiens), Han et al. 2018 (TRRUST v2), Szklarczyk et al. 2023 (STRING v12), Ashburner et al. 2000 (Gene Ontology).

## 3. Inputs

### 3.1 Target model (to be evaluated)
A trained single-cell transcriptomic model exposing **per-layer residual-stream activations**. The model must provide:
1. A tokenizer or embedding function mapping a cell's expression vector to a sequence of gene-indexed tokens. (scGPT: gene-token; Geneformer: BERT-style rank-based.)
2. A forward pass that returns **hidden states at every layer** — shape `(n_layers, seq_len, d_model)` per cell. For HuggingFace models: `output_hidden_states=True`. For scGPT, patch the forward to return the residual stream after each transformer block.
3. Access to **pre-contextual** (input) embeddings is sufficient for static analyses; for the full pipeline, contextualised hidden states across all layers are required.
4. Optional but recommended: access to attention weights for the cross-method dissociation check (Phase 9 step 2), and access to a companion model (e.g., Geneformer) for cross-model alignment tests.

**Worked example in source paper:** scGPT with **12 transformer layers**, **512-dim hidden states**, 8 attention heads (Cui et al. 2024). Source commit and weights from https://github.com/bowang-lab/scGPT.

### 3.2 Single-cell transcriptomic dataset

| Dataset | Source | Role | Key parameters |
|---|---|---|---|
| **Tabula Sapiens** (immune-lineage cells) | Jones et al. 2022 (Tabula Sapiens Consortium) | Primary input for embedding extraction | 2,000 cells; pre-contextual embedding matrix averaged across cells per layer |

Primary data access: https://tabula-sapiens-portal.ds.czbiohub.org/. Subset to **immune lineage** for the source paper's analysis. Format: h5ad.

### 3.3 Biological annotation databases

| Database | Version | Usage | Pair counts |
|---|---|---|---|
| **TRRUST v2** (Han et al. 2018) | — | TF → target regulatory edges; source of TF list (51 TFs) and positive regulatory pairs (589 pairs in-vocab), non-regulatory TF-target pairs as negatives (2,351 pairs) | https://www.grnpedia.org/trrust/ |
| **STRING v12.0** (Szklarczyk et al. 2023) | v12 | PPI edges at score ≥ 0.7 (1,022 pairs in-vocab, primary) and ≥ 0.4 (3,092 pairs) for confidence gradient; queried via API | https://string-db.org/ |
| **Gene Ontology** (Ashburner et al. 2000) | current | Cellular Component (CC) terms for subcellular localisation (GO:0005615 extracellular space, GO:0005829 cytosol, mitochondria, ER lumen, etc.); Biological Process (BP) terms as negative control (0/591 significant) | http://geneontology.org/ |

### 3.4 Curated gene sets (constructed per-run)

- **Full vocabulary:** all genes present in the model's input vocabulary. For scGPT on Tabula Sapiens immune: **4,803 genes** (of which only **2,039 have non-zero embedding norm** at every layer; the remaining positions are unused input slots — verify this to rule out sparsity artefacts).
- **195-gene submatrix:** intersection of the full vocabulary with annotated genes in TRRUST, STRING, and GO. The paper's primary statistical tests run on this curated set. Persist as `gene195_list.tsv`.
- **TF list:** 51 TFs, from TRRUST.
- **Target-only gene list:** 144 genes, non-TF but with TRRUST target annotations.
- **B-cell markers (7):** CD19, CD79A, MS4A1 (CD20), BLK, VPREB3, FCRL1, PAX5.
- **Cell-type marker panel (17 total across 7 types):** B-cell, T-cell, fibroblast, macrophage, dendritic cell, myeloid, and an additional lineage. Canonical markers from literature.
- **GC master-regulator set:** PAX5 (anchor), BATF, BACH2 (recruited GC-TFs), BCL6 (metabolically isolated repressor), PRDM1 (Blimp-1, transcriptional repressor).
- **Plasma-TF set:** IRF4, IRF8.
- **BCL6 metabolic neighbourhood reference set:** NAMPT, GLUL, PFKFB3, and genes identified from the BCL6 neighbourhood at layer 0 (used for metabolic-cluster overlap test).

## 4. Outputs

Per model × dataset × seed combination:

1. **Residual-stream embedding tensor** (NPY): shape `[n_layers, n_genes, d_model]` — for scGPT on immune Tabula Sapiens: `[12, 4803, 512]`. Stored as `layer_gene_embeddings.npy`. Plus the `[12, 195, 512]` submatrix.
2. **Per-layer SVD output** (NPY + JSON): left singular vectors `U`, singular values `σ`, right singular vectors `V^T`, mean-centering vector. Plus the scalar metrics: **effective rank** (Roy & Vetterli 2007), **TwoNN intrinsic dimensionality** (Facco et al. 2017), **participation ratio**, **SV1 variance fraction**.
3. **Per-gene spectral coordinates** (CSV): for each gene *g*, its projection onto SV1–SV7 at each layer. Shape `(n_genes, n_layers, 7)`.
4. **Co-pole enrichment report** (JSON + CSV): per (layer × axis × biological annotation) — observed co-pole rate, null mean, *z*-score, empirical *p*-value. For SV1: 8 GO CC terms × 12 layers. For SV2: STRING PPI at 12 layers, plus 5 STRING confidence quintiles.
5. **PPI confidence gradient report** (JSON): 5 quintiles × 12 layers × *z*-scores. Spearman ρ across quintile means, with *p*-value.
6. **Physical-vs-functional PPI dissociation report** (JSON): GO-only, STRING-only, and both-sets PPI signal at each layer; hub-vs-low-degree confound check.
7. **TF-vs-target classification report** (JSON): per-layer AUROC for joint SV2–SV7 (6D), SV5–SV7 (3D), SV2–SV4 (3D), plus 100-permutation null mean. Expected Table S1 format.
8. **Edge-level TF→target AUROC report** (JSON): per-layer AUROC for SV5–SV7 and SV2–SV4 using 589 TRRUST positives + 2,351 TF-matched non-regulatory negatives. Permutation null and *p*-values.
9. **Co-expression residualisation report** (JSON): residualised rank-biserial correlation *r_rb* per layer for SV5–SV7 and SV2–SV4, with 100-bootstrap CIs. Confirms which subspace retains signal after removing expression similarity.
10. **Signed regulation split** (JSON): AUROC for activation vs repression edges in each subspace. Expected Δ ≈ +0.021 (SV5–SV7) and +0.084 (SV2–SV4).
11. **Cell-type clustering report** (JSON): AUROC for each of 7 cell types based on within-type vs cross-type Euclidean distance in the full embedding; precision@10 per cell type; residual after L2-norm regression.
12. **B-cell attractor trajectory** (CSV): per-layer rank to B-cell centroid, distance to PAX5, GC-plasma angle, BCL6 metabolic overlap, per-lineage intrinsic dimensionality. Output tables sufficient to reproduce Figs. 4–6 of the source paper.
13. **Cross-seed robustness report** (JSON): mean / min / max AUROC across 3 fine-tuning seeds (main, seed43, seed44). Centered kernel alignment (CKA) across layers between seeds. Expected: Table S3 format.
14. **Negative-finding report** (JSON): results from persistent homology (feature-shuffle vs degree-preserving rewiring null contrast), cross-model alignment permutation test, feed-forward loop geometry permutation-corrected, GO BP SV2 enrichment (0/591 significant expected), ER-AUROC partial correlation confound check.
15. **Hypothesis registry** (JSON, one row per tested hypothesis): iteration number, family, hypothesis statement, method, result, null-model outcome, decision (promote / retire / revise). Paper registry: 183 hypotheses across 13 families over 63 iterations.
16. **Autonomous loop artefacts** (directory tree, if Phase 10 is run): per-iteration JSON schema (`executor_hypothesis_screen.json`, `executor_iteration_report.md`, `brainstormer_hypothesis_roadmap.md`, `brainstormer_next_iteration_brief.md`, etc.) under `iterations/iter_XXXX/`.

## 5. Dependencies

- **Python ≥ 3.10**, numpy ≥ 1.26 (< 2.0), scipy ≥ 1.12, scikit-learn ≥ 1.4, matplotlib ≥ 3.8, umap-learn ≥ 0.5.
- **scGPT** — clone from https://github.com/bowang-lab/scGPT; load pretrained weights per the project's instructions. The pipeline extracts residual-stream activations from forward passes.
- **Geneformer** (optional) — `ctheodoris/Geneformer` on HuggingFace; used for the cross-model alignment negative control (Phase 9 step 2).
- **External data:** Tabula Sapiens immune lineage (h5ad), TRRUST v2 (`trrust_human.tsv`), STRING v12.0 (queried via API or downloaded as `string_pairs.tsv`), Gene Ontology (`gene2go_all.pkl` or equivalent).
- Full environment: `repos/topology-biomechinterp1/requirements.txt`.
- **Claude CLI** (optional, Phase 10 only) — required if running the autonomous hypothesis-screening loop end-to-end. See `repos/topology-biomechinterp1/loop/config_claude.json`.

## 6. Methodology

The pipeline runs in ten phases. Phases 0–1 are the model-specific adapter and global spectral decomposition; Phases 2–8 are the biology-specific test battery; Phase 9 is the negative-control scaffold; Phase 10 is the autonomous hypothesis-screening loop that the paper used to generate the test battery in the first place.

---

### Phase 0 — Model adapter: residual-stream embedding extraction

**Purpose:** produce a standard `[n_layers, n_genes, d_model]` tensor of per-gene residual-stream embeddings, averaged across cells, for the full vocabulary and the 195-gene curated submatrix.

**Steps:**

1. **Load the dataset.** Tabula Sapiens immune-lineage cells (~20k cells), standard Scanpy QC (log1p, total-count normalisation). Sample 2,000 cells (primary; replicate with three fine-tuning seeds `main`, `seed43`, `seed44`).
2. **Tokenise** each cell per the model's native tokenizer. For scGPT: gene-token ordering with expression-rank-ordered input. For Geneformer (cross-model control): rank-based BERT tokens.
3. **Forward pass with hidden states enabled.** For HuggingFace models: `output_hidden_states=True`. For scGPT: register forward hooks on the output of each transformer block, or patch the forward to return the list of residual-stream activations.
4. **Per-cell per-layer per-gene extraction.** For each of the 2,000 cells, extract the `d_model`-dimensional activation at the gene's token position at every layer. Stack across cells and average: `embed[l, g] = mean_c activation[l, gene_token_position_c, :]`. Shape: `[12, n_genes, 512]` for scGPT. The paper uses "**pre-contextual**" embeddings — meaning the representation at the gene's token position averaged across cells, NOT the cell-level CLS embedding.
5. **Construct the gene vocabulary.**
   - Full vocab = 4,803 gene positions (scGPT immune run).
   - **195-gene submatrix** = intersection of full vocab with genes annotated in TRRUST, STRING, and GO. Persist `gene195_list.tsv`.
   - **Verify the non-zero embedding count is constant across layers** (expected: **2,039** non-zero norms; the remaining 2,764 positions are unused input slots). This rules out sparsity artefacts — critical for Phase 5 edge-level AUROC depth decay.
6. **Save artefacts.** `data/embeddings/<seed>/layer_gene_embeddings.npy` (full) and `layer_gene_embeddings_195.npy` (curated).

**Code reference:** `repos/topology-biomechinterp1/METHODS.md` for the extraction procedure. See iteration scripts for downstream consumption patterns, e.g. `iterations/iter_0031/run_iter0031_screen.py` uses the 195-gene submatrix for the clean spectral-gap test.

---

### Phase 1 — Global spectral decomposition (per layer)

**Purpose:** decompose each layer's gene-embedding matrix via SVD and compute global dimensionality metrics.

**Steps:**

1. **Mean-centre columns** of the per-layer gene matrix `M ∈ ℝ^(n_genes × d_model)`: subtract per-column mean (so each of the 512 embedding dimensions has zero mean over genes).
2. **SVD**: `M = U · diag(σ) · V^T`. Store `U`, `σ`, `V^T` per layer. The columns of `V` are the right singular vectors (directions in the 512-dim space); the left singular vectors `U` project each gene onto those directions. The *i*-th spectral coordinate of gene *g* is `U[g, i] · σ[i]` (we call this "projection onto SVᵢ"). Record both the full vocabulary SVD and the 195-gene SVD separately — they yield different spectra.
3. **Effective rank (Roy & Vetterli 2007):** `ER = exp(-∑ p_i log p_i)` where `p_i = σ_i² / ∑_j σ_j²`. Expected scGPT result: 23.6 at layer 0 → **1.6 at layer 11** (14.4-fold collapse; Spearman ρ = **−1.000** across layers).
4. **TwoNN intrinsic dimensionality (Facco et al. 2017):** compute on 2,000 randomly sampled genes from the full vocabulary (3 random-seed replicates). Expected: **32.6 → 18.1** across layers (44.6% reduction).
5. **Participation ratio:** `PR = (∑ σ_i²)² / ∑ σ_i⁴`, computed on the 195-gene submatrix. Expected: **58 → 9.5** (6.1-fold drop).
6. **SV1 variance fraction:** `σ_1² / ∑ σ_i²`. Expected on the full vocabulary: 53.7% (L0) → **93.4% (L11)**. On the 195-gene submatrix: 19% → 77%.
7. **Feature-shuffle null** (sparsity-artefact control). Shuffle each of the 512 embedding dimensions independently (preserving per-dimension distributions but destroying gene-level coordinate structure). Recompute SVD. Expected: effective rank jumps to ~28.9 (17.6× the observed layer-11 value). **The compression is learned, not architectural.**

**Code reference:** `repos/topology-biomechinterp1/iterations/iter_0031/run_iter0031_screen.py` — clean 195-gene effective-rank and participation-ratio computation.

---

### Phase 2 — Co-pole enrichment test (general-purpose biological axis hypothesis test)

**Purpose:** test whether a given biological annotation (e.g., GO term membership, STRING PPI edge) co-localises genes along a single singular-vector axis. This is the workhorse statistical test re-used in Phases 3–5.

**Steps:**

1. **Define poles.** For axis SVₖ at layer *l*, sort genes by their projection onto SVₖ. The **top pole** is the top-*K* genes; the **bottom pole** is the bottom-*K*. Default **K = 52** (paper's primary setting for the 195-gene submatrix). The "co-pole" region = top-pole ∪ bottom-pole.
2. **Define the annotation.** A set of gene pairs linked by the annotation (e.g., "both in GO:0005615 extracellular space", or "both are STRING PPI partners"). Equivalently, a set of "positive" genes for single-label annotations.
3. **Observed co-pole rate** = fraction of annotated gene pairs (or annotated genes, for single-label axes) for which both members are in the top pole, or both in the bottom pole, or both on the same pole. For single-label enrichment: use Fisher's exact test comparing pole membership to annotation membership.
4. **Gene-label-shuffle null.** Randomly permute the gene-to-embedding-row assignments *N* = 500 times. For each permutation, recompute pole membership (the embeddings are unchanged, but "which gene is which" is randomised). Compute the null co-pole rate distribution.
5. **Test statistic:** *z*-score = (observed − null mean) / null SD. Empirical *p* = fraction of permutations with co-pole rate ≥ observed.
6. **Run the test across all 12 layers and all candidate axes** (typically SV1 through SV7). Count significant layers (Bonferroni or BH at α = 0.05).
7. **Report Table S4 format:** per layer × axis × annotation: observed, null mean, *z*, empirical *p*, significance flag.

**Code reference:** `repos/topology-biomechinterp1/iterations/iter_0009/run_iter0009_screen.py` (SV1 GO compartment scan) and `iter_0010/run_iter0010_screen.py` (SV2 TRRUST co-pole and annotation-density confound).

---

### Phase 3 — SV1: subcellular localisation and the secretory pathway

**Purpose:** test whether the dominant spectral axis SV1 encodes subcellular localisation along the cellular secretory pathway (mitochondria → ER lumen → extracellular), and quantify the layer-wise ordering.

**Steps:**

1. **Test 8 GO Cellular Component terms across all 12 layers** using the co-pole enrichment test (Phase 2):
   - GO:0005615 (extracellular space), GO:0005829 (cytosol), GO:0005759 (mitochondrial matrix), GO:0005783 (endoplasmic reticulum), GO:0005788 (ER lumen), GO:0005739 (mitochondrion), plus two additional compartments the paper cites (check Methods §4.7 of the paper for the canonical 8-term list).
2. **Expected per-layer pattern** (scGPT immune):
   - **Layer 11 extracellular-pole enrichment:** GO:0005615 OR = **6.37**, *p* = 2.6×10⁻⁴.
   - **Layer 11 cytosolic-pole enrichment:** GO:0005829 OR = 2.96, *p* = 0.010.
   - **Layers 2–4 mitochondrial enrichment (transient):** peak OR = 23.3 at layer 3; the signal disappears at deeper layers.
   - **Layers 1–11 ER lumen enrichment strengthens progressively:** OR from 4.7 to 18.5.
   - **All 12 layers: extracellular space consistently enriched.**
3. **Interpret the layer-wise ordering as a geometric trace of the cellular secretory pathway:** mitochondria (early layers) → ER lumen (mid layers) → extracellular (late layers).
4. **Gene-label-shuffle validation** (at layer 11 for GO:0005615): *N* = 500 permutations, empirical *p* = **0.004** (only 0.4% of random assignments yield an enrichment this strong).
5. **Secondary finding:** SV1 is **depleted of transcription factors** — the high-SV1 pole has lower TF membership than baseline (OR ≈ 0.108); the low-SV1 pole is 2.3× enriched for TFs. This informs the TF-vs-target analysis in Phase 5: SV1 separates "secreted proteins" from "regulatory machinery" as the dominant distinction.

**Code reference:** `iter_0009/run_iter0009_screen.py`, `iter_0010/run_iter0010_screen.py`.

---

### Phase 4 — SV2–SV4: protein–protein interaction network encoding

**Purpose:** test whether orthogonal axes SV2, SV3, SV4 encode protein–protein interaction networks with quantitatively graded fidelity to experimental interaction strength.

**Steps:**

1. **Primary co-pole enrichment (SV2, STRING ≥ 0.7).** Using **1,022 high-confidence STRING PPI pairs in-vocab**, run the co-pole test on SV2 across all 12 layers. Expected (Table S4):
   - Layer 0: obs 0.252, null 0.121, *z* = **5.83**, *p* < 0.001
   - Layer 1: obs 0.266, null 0.121, *z* = **6.52**, *p* < 0.001
   - Layer 11: obs 0.239, null 0.122, *z* = **5.45**, *p* < 0.001
   - **All 12 layers significant at *p* < 0.001** (none drop below *z* = 3.26).
2. **Extend the test across axes SV1 to SV5** (Table 1). Expected mean observed / null / significant layers:
   - SV1: 0.119 / 0.124 / 2/12 (≈ null — SV1 is the localisation axis, not PPI)
   - SV2: **0.226 / 0.124 / 12/12 (strongest PPI signal)**
   - SV3: 0.198 / 0.124 / 12/12
   - SV4: 0.158 / 0.122 / 7/12
   - SV5: 0.149 / 0.122 / 6/12
   - **Three PPI-encoding axes (SV2–SV4) are nearly orthogonal** (pairwise Pearson *r* < 0.25) — each captures a distinct subset of the interaction network.
3. **STRING confidence gradient test (the most striking quantitative result).** Split STRING PPI pairs (score ≥ 0.4, 3,092 pairs) into 5 quintiles by confidence score:
   - Q1: [0.40, 0.46), 617 pairs
   - Q2: [0.46, 0.53), 619 pairs
   - Q3: [0.53, 0.64), 618 pairs
   - Q4: [0.64, 0.80), 619 pairs
   - Q5: [0.80, 1.00], 619 pairs
   Compute mean co-pole *z*-score per quintile on SV2, averaged across 12 layers. Expected (Table S5): **1.00, 1.48, 2.09, 3.18, 5.02** — a perfectly monotonic gradient. **Spearman ρ = 1.000, *n* = 5, *p* = 0.017.** The encoding is quantitatively graded — scGPT does not merely encode a binary "interacts / doesn't" — it encodes **confidence**.
4. **Physical-binding vs shared-function dissociation.** Stratify STRING pairs by **GO Jaccard similarity** (fraction of shared GO terms). Test three sets:
   - High STRING + high GO: *z* ≈ 2.66 (12/12 layers)
   - High STRING + **low GO** (Jaccard < 0.05): *z* ≈ **2.66** (12/12 layers, matching the high-GO set)
   - Low STRING + high GO: *z* ≈ **1.89** (only 7/12 layers)
   **Conclusion:** the geometry is driven by **physical molecular interactions**, not shared functional annotation.
5. **Hub confound exclusion.** Stratify PPI pairs by node degree. Expected: **low-degree proteins show *stronger* co-localisation** than hub proteins (low-degree *z* = 3.1 vs hub *z* = 2.6). This rules out "the signal is just STRING's high-degree hubs clustering in embedding space."
6. **PPI beyond STRING.** TRRUST TF→target pairs that are **absent from STRING** (*N* = 141 pairs) still show significant co-pole enrichment: AUROC = **0.573**, 12/12 layers at *p* < 0.007. The geometry encodes multiple types of biological interaction, not just physical binding.

**Code reference:** `iter_0015/run_iter0015_screen.py` (SV2–SV5 PPI, STRING confidence gradient, GO-vs-PPI dissociation), `iter_0010/run_iter0010_screen.py` (annotation-density confound check).

---

### Phase 5 — SV5–SV7: transcriptional regulatory encoding

**Purpose:** test whether SV5–SV7 encodes gene regulatory relationships, and dissect the regulatory signal into (a) categorical "gene class identity" (TF vs target), (b) specific TF→target edges, (c) activation vs repression asymmetry, with rigorous co-expression-confound controls.

**Steps:**

1. **TF-vs-target 6D classification (joint SV2–SV7).** Project gene embeddings onto the 6-dimensional spectral subspace spanning SV2, SV3, SV4, SV5, SV6, SV7 at each layer. Classify genes as **TF (*n* = 51)** vs **target-only (*n* = 144)** using multinomial cosine similarity to class centroids (or an equivalent linear classifier). Report per-layer AUROC. Expected (Table S1, iteration 0056):
   - Layer 0: joint 0.771, SV5–SV7 only 0.715, SV2–SV4 only 0.634 (null mean 0.506)
   - Layer 3: joint **0.789** (peak), SV5–SV7 0.716, SV2–SV4 0.703
   - Layer 11: joint 0.688, SV5–SV7 0.686, SV2–SV4 0.655
   - **Mean across 12 layers: 0.744.** Joint 6D outperforms either 3D subspace alone at 11/12 layers. All 12 layers significant at *p* < 0.01 (100-permutation null).
2. **Cross-seed replication (Table S3, iteration 0057).** Repeat for three fine-tuning seeds:
   - main: mean 0.744, min 0.687, max 0.789
   - seed43: 0.753, 0.697, 0.813
   - seed44: 0.757, 0.714, 0.802
   Expected cross-seed SD ≈ **0.016**. All seeds > 0.65 at every layer.
3. **Subspace dissociation via co-expression residualisation.** For each gene pair in the 195-gene set, compute pairwise co-expression similarity = Pearson correlation of expression across cells. Regress spectral proximity (cosine similarity in the relevant SV subspace) on co-expression similarity via OLS. Test whether the **residual** spectral proximity still separates TRRUST TF→target pairs from non-regulatory pairs. Quantify via **rank-biserial correlation** *r_rb*; significance by permutation of regulatory labels; robustness by 100 bootstrap resamples. Expected results:
   - **SV2–SV4: residualised signal is null at 0/12 layers.** SV2–SV4 regulatory signal is **entirely explained by co-expression** — this subspace encodes "is this gene a TF?", not "does this TF regulate that target".
   - **SV5–SV7: residualised signal retains at layer 0 with *r_rb* = 0.148, permutation *p* < 0.001, 100/100 bootstrap resamples positive.** SV5–SV7 encodes **co-expression-independent** regulatory proximity.
4. **Dual-regime interpretation:**
   - **Early layers (L0–L3)** — SV5–SV7 dominates (AUROC ≈ 0.78). The model maintains specific relational detail ("STAT3 regulates BCL2") in a form independent of whether STAT3 and BCL2 are co-expressed.
   - **Mid layers (L4–L8)** — SV2–SV4 takes over (AUROC ≈ 0.72). As the residual stream compresses (Phase 1's 14.4× collapse), this fine-grained structure is distilled into coarser categorical distinctions (TF vs target).
   - **Late layers (L9–L11)** — both subspaces degrade toward joint 0.688.
5. **Edge-level TF→target AUROC (Table S2, iteration 0062).** Score each TF→gene pair by cosine similarity in the SV5–SV7 subspace, using **589 known TRRUST regulatory pairs** as positives and **2,351 TF-matched non-regulatory pairs** (same TFs, non-target genes) as negatives. Expected per-layer AUROC:
   - Layer 0: SV5–SV7 = **0.602**, SV2–SV4 = 0.449, permutation *p* = 0.000 — significant
   - Layer 4: SV5–SV7 = 0.591 — significant
   - Layer 8: SV5–SV7 = 0.524, permutation *p* = 0.045 — marginal
   - Layer 9: SV5–SV7 = 0.494 — **not significant**
   - Layer 11: SV5–SV7 = 0.498 — not significant
   **Spearman ρ = −0.958, *p* = 9.5×10⁻⁷** — monotonic depth decay. **Edge-level signal peaks at L0 and is extinguished by L9.** The practical implication: **if you want specific regulatory edges from the model, use layer 0 — not the final layer.**
6. **Sparsity-artefact control.** Verify the depth trend is not driven by an increasing fraction of zero-norm embeddings. Expected: **non-zero gene count is constant at 2,039 across all layers** (the remaining 2,764 positions are unused input slots). If the non-zero count varies across layers, sparsity is confounding the result.
7. **Signed regulation asymmetry (iteration 0063).** Stratify TRRUST edges by activation vs repression. Compute AUROC separately. Expected:
   - **SV5–SV7: repression AUROC 0.620, activation AUROC 0.599, Δ = +0.021**
   - **SV2–SV4: repression AUROC 0.550, activation AUROC 0.466, Δ = +0.084**
   - **Repression edges are more geometrically separable than activation edges in both subspaces.**
   **Interpretation:** transcriptional repression often involves more stereotyped molecular mechanisms (chromatin remodelling, co-repressor recruitment) than activation (diverse enhancer/co-activator combinations); the model learns geometric regularities for the more structurally constrained repressive relationships. Alternatively, repressive TFs may be more functionally distinct from their targets than activating TFs, creating larger separations.
8. **Mechanistic probe: TF-TF vs TF-target proximity (iteration 0063).** In SV2–SV4, test TF-TF pair AUROC (0.539 — TFs cluster) and TF-target pair AUROC (0.485 — TFs and targets are **repelled**, below chance). In SV5–SV7, both TF-TF (0.528) and TF-target (0.599) pairs are **above chance**. This probe confirms that SV2–SV4 encodes "gene class identity" (TFs together, targets together) while SV5–SV7 encodes regulatory **co-embedding** (TFs near their targets).

**Code references:**
- Joint SV2–SV7 classifier: `iterations/iter_0056/run_*.py`
- Cross-seed robustness: `iterations/iter_0057/run_*.py`
- Edge-level AUROC: `iterations/iter_0062/run_*.py`
- Co-expression residualisation + dual-regime: `iterations/iter_0050/run_*.py` and `iter_0051/run_*.py`
- Signed regulation split + mechanistic probe: `iterations/iter_0063/run_*.py`

---

### Phase 6 — Cell-type marker clustering

**Purpose:** test whether the model's full 512-dim embedding organises cell-type identity geometrically, by measuring within-type vs cross-type Euclidean distance among canonical cell-type markers.

**Steps:**

1. **Curate 17 canonical markers across 7 cell types** (B-cell, T-cell, fibroblast, macrophage, dendritic cell, myeloid, and one additional lineage) from literature sources (Methods §4.6).
2. **Compute pairwise Euclidean distances** in the full 512-dim embedding at each layer.
3. **AUROC for same-type vs different-type pairs.** Pairs within a cell type are positives; pairs across cell types are negatives. Expected: **AUROC = 0.851** (primary run; cross-seed mean 0.789 ± 0.016).
4. **Cell-type contamination control.** Randomly partition non-marker genes into sets matched in size to each marker set. Verify the random partitions give AUROC ≈ 0.5 (chance). Expected: contamination-control AUROC is at chance (paper's Table S6 entry: "contamination ctrl chance").
5. **HLA-I perfect score.** Paper reports HLA-I markers achieve **AUROC = 1.000** within their class — a cell-type subfamily where the model's geometric clustering is perfect.

**Code reference:** cell-type clustering is scattered across several iterations; see `reports/autoloop_master_log.md` for the canonical iteration index. Paper references precision@10 and Euclidean-distance AUROC throughout Section 2.2.

---

### Phase 7 — B-cell attractor dynamics (germinal centre reaction trace)

**Purpose:** track individual master-regulator trajectories across transformer depth to reveal whether the model has internalised the temporal logic of cell-lineage differentiation — specifically the germinal centre (GC) reaction in B-cell differentiation.

**Steps:**

1. **Define the B-cell manifold centroid at each layer** as the mean embedding of 7 canonical B-cell markers (CD19, CD79A, MS4A1/CD20, BLK, VPREB3, FCRL1, PAX5).
2. **Gene-to-centroid rank:** for each of the 195 genes, rank by Euclidean distance to the centroid (rank 1 = closest). Report per layer for the key genes: PAX5, BATF, BACH2, BCL6, PRDM1, IRF4, IRF8.
3. **Precision@10:** among the 10 nearest neighbours of the B-cell centroid at each layer, fraction that are B-cell markers. Compare to bootstrap null (500 random draws of 7 genes from the 195-gene set). Expected:
   - B-cell precision@10: *z* = **7.55**, *p* < 0.001 (highly significant)
   - T-cell: *z* = −1.37 (not significant)
   - Dendritic cell: *z* = −0.87 (not significant)
   - Myeloid and other lineages: not significant
   **Only B cells show strong geometric clustering.**
4. **L2-norm regression** (structural-vs-magnitude control). Regress PC1 scores on L2 norms and check whether B-cell markers remain outliers in the residuals. Expected residual *p* = **0.002** — signal is structural, not magnitude-driven. Also empirical bootstrap *p* < 0.001.
5. **GC master-regulator convergence trajectory.** Track PAX5, BATF, BACH2, BCL6, PRDM1 across all 12 layers:
   - **PAX5** — stable position near B-cell centroid from layer 0 (rank ≈ **66**). Geometric anchor.
   - **BATF** — rank drops **1,510 (L0) → 189 (L11)**; distance to PAX5 **18.3 → 5.1**; Spearman ρ = **−0.972**, *p* < 0.0001.
   - **BACH2** — rank drops **611 (L0) → 146 (L11)**; distance to PAX5 **17.4 → 5.2**; Spearman ρ = **−0.844**, *p* < 0.001.
   - **PRDM1** (Blimp-1, transcriptional repressor, direct target of BCL6) — converges toward B-cell centroid at **the same rate** as the activating GC-TFs (Spearman ρ = **−1.000**, *p* < 0.001); BACH2–PRDM1 becomes the **tightest pair by layer 11** (distance = 3.94). The model encodes the **combinatorial GC regulatory circuit — activators and repressors alike — as a coherent geometric attractor**, not separating regulatory modes spatially.
   - **Convergence onset at layer 3** — independently coincides with the B-cell manifold compression breakpoint (step 7).
6. **GC-plasma orthogonality trajectory.** Compute the **angle subtended at the B-cell centroid between GC-TF centroid (mean of BATF, BACH2, PAX5) and plasma-TF centroid (mean of IRF4, IRF8)** at each layer. Expected: **77° (L0) → 94° (L11)**. The two differentiation programs diverge toward **near-orthogonality** by the final layer, mirroring the biological reality that GC and plasma cell fates are mutually exclusive outcomes from a common B-cell progenitor.
7. **B-cell manifold compression is lineage-specific.** Compute TwoNN intrinsic dimensionality separately for B-cell (*n* = 7), T-cell (*n* = 12), and myeloid (*n* = 7) marker gene sets across the 12 layers. Expected:
   - **B-cell: intrinsic dim decreases monotonically 8.2 → 5.1** (ρ = **−0.951**, *p* < 0.0001)
   - **T-cell: no compression** (ρ = +0.287, *p* = 0.37)
   - **Myeloid: slight reversed trend** (ρ = +0.699, *p* = 0.011)
8. **BCL6 metabolic isolation.** At each layer, compute the 20 nearest neighbours of BCL6 in the 195-gene embedding. Check overlap with a metabolic reference gene set (NAMPT, GLUL, PFKFB3, and others identified from the BCL6 neighbourhood at layer 0; plus STAT3). Expected: **10/20 neighbours are metabolic genes (or STAT3)**, **0/20 are B-cell markers or GC-TFs**, across **all 12 layers**. BCL6 is **never** in the B-cell cluster despite being essential for GC reactions — consistent with its biological role as a **pleiotropic repressor operating at the intersection of metabolic reprogramming and immune regulation**.
9. **GC attractor is uniquely delayed.** Compare across lineages: T-cell and myeloid TFs start at low ranks (170 and 86) near their respective centroids **from layer 0**, indicating pre-wired proximity. BATF (1,510) and BACH2 (611) converge **only from layer 3 onward** (Spearman ρ = −0.993, *p* < 10⁻⁶). This **delayed convergence pattern is unique to the B-cell/GC program**, suggesting the transformer encodes B-cell differentiation as a **computed geometric trajectory** rather than a static embedding property.

**Interpretation:** the GC master-regulator triangle (PAX5 anchor, BATF/BACH2 recruited, BCL6 metabolically isolated repressor) is a geometric reflection of well-established immunology. The model appears to have learned the **temporal logic** of B-cell differentiation — PAX5 establishes identity early, then GC factors are recruited. This trajectory information **cannot be learned from static co-expression data alone**: it suggests the model has internalised aspects of regulatory **dynamics**.

**Code reference:** B-cell attractor analyses are threaded through iterations 46–55 (see `reports/autoloop_master_log.md` and the "Key Iteration Map" in the repo README).

---

### Phase 8 — Cross-seed robustness and representational stability

**Purpose:** establish that spectral findings generalise across fine-tuning seeds and characterise layer-wise representational stability.

**Steps:**

1. **Run Phases 0–7 for 3 fine-tuning seeds** (`main`, `seed43`, `seed44`).
2. **Joint SV2–SV7 TF-vs-target AUROC cross-seed summary.** Expected (Table S3): main 0.744 [0.687, 0.789], seed43 0.753 [0.697, 0.813], seed44 0.757 [0.714, 0.802]. Cross-seed SD ≈ 0.016.
3. **Subspace basis alignment (principal angles).** Compute the principal angles between SV5–SV7 bases at each layer across seed pairs. Expected: 13°–41° drift (inconclusive — the subspaces are not literally identical across seeds, but the per-gene classifier outputs are stable).
4. **Centered Kernel Alignment (CKA, Kornblith et al. 2019)** between per-layer embedding matrices from different random cell samples (same pre-trained model, different input cells). Expected: CKA declines from **0.979 at layer 0 to 0.779 at layer 11**, with the steepest drop at layers 10–11. **Early layers are highly reproducible across cell samples; deep layers diverge** — consistent with early layers encoding conserved biological structure and late layers encoding input-specific features.

**Code reference:** `iterations/iter_0057/run_*.py` (cross-seed robustness).

---

### Phase 9 — Negative-finding control scaffold

**Purpose:** rigorously interpret the positive findings by establishing what does *not* survive controls. These five tests are mandatory — a positive finding that does not sit alongside these negatives cannot be trusted.

#### 9a — Persistent homology under rigorous nulls

**Why:** topological features are extremely sensitive to null-model choice, and weak nulls (feature-shuffle) produce spurious positives.

**Method:** compute persistent homology (persistence diagrams, Betti numbers) on kNN graphs from the full 512-dim embedding. Run under **two null models**:
- **Feature-shuffle null** (weak): independently permute values within each of the 512 embedding dimensions.
- **Degree-preserving graph rewiring null** (strict): rewire the kNN graph using the curveball algorithm, preserving each node's degree.

**Expected:** under feature-shuffle, **11/12 tests *p* < 0.05** (looks significant). Under degree-preserving rewiring, **0/24 tests significant.** The signal **vanishes under the stricter null.**

**Verdict: negative.** The initial "topological signal" was an artefact of weak null models. **This is the canonical cautionary tale of the pipeline** — always require the strictest null model to pass before promoting a finding.

#### 9b — Cross-model alignment

**Why:** if biological knowledge is "truly there", two different architectures should align.

**Method:**
- Compute cosine similarity between scGPT and Geneformer embedding matrices, restricted to shared vocabulary genes.
- Run a permutation test of representation alignment.
- Separately test whether **Geneformer static (pre-contextual) embeddings** independently encode STRING PPI.

**Expected:**
- Raw cosine similarity between scGPT and Geneformer: **0.825** (high).
- **Permutation test of representation alignment: *p* > 0.3** (fails significance — the 0.825 is not surprising under a shared-vocabulary null).
- **Geneformer static embeddings DO independently encode STRING PPI** at *p* = **7.8×10⁻¹²⁷** — confirming PPI encoding is **convergent across architectures**.
- **But the B-cell attractor dynamics are ABSENT from Geneformer's pre-contextual embeddings** (precision@10 = 0.000; top neighbours are biologically incoherent). This indicates **cell-type geometric structure requires contextual processing** and does not trivially generalise across models.

**Verdict: partial.** PPI is convergent; cell-type attractor dynamics are not.

#### 9c — Feed-forward loop geometry

**Why:** if the model encodes regulatory circuits, feed-forward loops (FFLs — where TF A regulates both TF B and a shared target) should have intermediate genes occupying geometrically intermediate positions.

**Method:** test 264 TRRUST-derived transcriptional FFLs. For each FFL, compute whether the intermediate gene's position lies between the source TF and the target in the relevant spectral subspace.

**Critical pitfall:** the naive baseline **t = 0** is **incorrect**. The correct null is **t ≈ 0.5 by symmetry** — because a gene randomly positioned between two endpoints is on average halfway between them in Euclidean space. **Use permutation-corrected baselines.**

**Expected:** after correct permutation correction, **no significant result.** Feed-forward loop geometry is null.

**Verdict: negative.** Methodological lesson: always use permutation-corrected baselines for betweenness statistics in high-dimensional spaces.

#### 9d — GO Biological Process terms on SV2

**Why:** bound the biological content of SV2 to compartment/network identity vs functional programs.

**Method:** test **591 GO BP terms** for SV2 pole enrichment using the Phase 2 co-pole test with BH correction.

**Expected:** **0/591 significant.** SV2 does **not** encode Biological Process programs — it encodes PPI network membership.

**Verdict: negative.** Informatively rules out a plausible alternative interpretation.

#### 9e — Effective rank does not independently predict classifier performance

**Why:** apparent correlations between dimensionality metrics and downstream performance must be disentangled from layer depth.

**Method:** test whether effective rank correlates with per-layer classifier AUROC, then compute the **partial correlation controlling for layer depth**.

**Expected:**
- Raw ER-AUROC correlation: ρ = **0.855** (looks strong).
- **Partial correlation controlling for layer: ρ = −0.045** (fully confounded — ER and AUROC both vary with layer, but ER does not independently predict AUROC).

**Verdict: negative.** The "dimensionality predicts classifier performance" claim does not survive a partial correlation control.

**Code references:**
- Persistent homology controls: `iter_0001/run_graph_topology_screen.py` and subsequent iterations that upgrade from feature-shuffle to degree-preserving nulls.
- Cross-model alignment: iterations in the 20–30 range (per the README iteration-map).
- FFL geometry + GO BP + ER-AUROC confound: late iterations (50–60 range); see `reports/autoloop_master_log.md` for the canonical placement.

---

### Phase 10 — Autonomous two-agent hypothesis-screening loop

**Purpose:** the source paper's methodology *is* an autonomous loop, not a fixed battery. Phase 10 documents how to run the loop so that coding agents can **extend** the test battery for a new model rather than just re-run the 63 iterations.

**Architecture:**

1. **Two agents.**
   - **Executor agent** — designs and runs computational experiments (SVD, permutation tests, enrichment analyses, classifier training), reviews results, proposes next-step hypotheses. Writes per-iteration structured JSON (`executor_hypothesis_screen.json`) and a narrative markdown report (`executor_iteration_report.md`).
   - **Brainstormer agent** — reads the executor's output, identifies stale directions, generates an ambitious portfolio of new testable hypotheses, selects the top 3 for the next iteration (high-probability, high-risk/high-reward, cheap-screen), and writes `brainstormer_hypothesis_roadmap.md`, `brainstormer_structured_feedback.md`, `brainstormer_next_iteration_brief.md`.

2. **Branch retirement policy.** A hypothesis branch is **retired after two consecutive negative results** (non-significant after appropriate null correction). This is mandatory — without retirement, the loop diverges.

3. **Positive-finding promotion rule.** Every positive finding must pass **at least one permutation-based null model** before being promoted. The Phase 9a persistent-homology cautionary tale is the reason — weak nulls will produce false positives at scale.

4. **Iteration artefacts.** Every iteration produces a **machine-readable JSON** artefact (`executor_hypothesis_screen.json`, 18-field schema including family, method, decision, novelty_type) plus narrative markdown plus executable analysis scripts (`run_*.py`). The full iteration log, intermediate results, and agent transcripts are preserved in `iterations/iter_XXXX/`.

5. **Hypothesis families.** The paper's loop worked across **13 families**: persistent homology, graph topology, geodesic distance, intrinsic dimensionality, cross-model alignment, module structure, null sensitivity, split robustness, dynamical stability, SVD biological axes, PPI network encoding, cell-type/family clustering, attention-SVD dissociation, regulatory geometry (both SV2–SV4 and SV5–SV7 branches), edge-level geometry, signed regulation, B-cell attractor dynamics. Each family has a target outcome in Supp Table S6.

6. **Executor prompt template** (`repos/topology-biomechinterp1/prompts/executor_prompt_topology_hypothesis_screening.md`, 113 lines). Instructs the executor to rapidly screen hypothesis space, target 2–3 tested hypotheses per iteration with at least 1 materially novel (new family, method, or biological anchor), use the designated conda environment, retire hypotheses after 2 failures with adequate controls, and produce the required JSON schema.

7. **Brainstormer prompt template** (`repos/topology-biomechinterp1/prompts/brainstormer_prompt_template.md`, 56 lines). Instructs the brainstormer to generate at least 10 (target 12–16) fresh testable hypotheses with broad coverage, provide per-idea one-sentence hypothesis + concrete test + expected signal + null/control + value/cost, select top 3 for the next iteration, and produce the required output files.

8. **Runtime.** `loop/run_claude_topology_autoloop.py` is the main runner. `loop/config_claude.json` specifies the agent backend (`--model`, `--reasoning-effort`, timeout, max iterations). `loop/start_claude_autoloop.sh` and `stop_claude_autoloop.sh` are bash wrappers. Requires the Claude CLI tool (or an equivalent agent runtime — the loop architecture is agent-backend-agnostic if the JSON schema contract is preserved).

9. **Target iteration count.** The source paper ran **63 iterations** and tested **183 hypotheses**. For a new-model screen, plan for **40–80 iterations** depending on how many hypothesis families transfer unchanged from the scGPT run. Monitor the master log; retire aggressively.

**Quickstart for extending the loop to a new model:**

1. Fork `repos/topology-biomechinterp1`.
2. Point `data/embeddings/<seed>/layer_gene_embeddings.npy` at the new model's Phase 0 output.
3. Update path variables (`SUBP38_OUTPUTS`, `TRRUST_PATH`, `EDGE_TSV`, `GENE2GO_PATH`) for your local environment.
4. Run iteration 0001 manually to validate the data-loading contract (`iter_0001/run_graph_topology_screen.py` is a minimal smoke test).
5. Run the Phase 3, 4, 5 positive-finding iterations (0009, 0010, 0015, 0031, 0056, 0057, 0062, 0063) to establish sanity signatures on the new model.
6. Start the autonomous loop with `loop/start_claude_autoloop.sh` for novel hypothesis generation. Seed the brainstormer with the 13 hypothesis families from Supp Table S6 and with the iteration 0001–0063 summary from `reports/autoloop_master_log.md`.
7. Require every positive finding to pass at least one permutation-based null. Require every negative finding to retire its branch after 2 consecutive non-significant iterations.
8. Emit a final `reports/autoloop_master_log.md` and a machine-readable hypothesis registry for the new model.

**Code references:**
- Loop runner: `loop/run_claude_topology_autoloop.py`
- Config: `loop/config_claude.json`
- Executor prompt: `prompts/executor_prompt_topology_hypothesis_screening.md` (113 lines)
- Brainstormer prompt: `prompts/brainstormer_prompt_template.md` (56 lines)
- Master narrative: `reports/autoloop_master_log.md` (2,518 lines, all 63 iterations chronologically)
- Repository entry point: `README.md` (Key Iteration Map)

---

## 7. Parameters

| Parameter | Default | Alternate / scan range | Used in |
|---|---|---|---|
| `n_cells` (cells for embedding extraction) | **2000** | 500–5000 | Phase 0 |
| `n_seeds` (fine-tuning seeds for robustness) | **3** (main, seed43, seed44) | 1–10 | Phase 8 |
| `n_layers` | **12** (scGPT) | model-dependent (18 for Geneformer V2) | Phases 0–9 |
| `d_model` | **512** (scGPT) | model-dependent | Phases 0–9 |
| `vocab_full` | **4803** (scGPT immune) | — | Phase 0 |
| `vocab_curated` | **195** (TRRUST ∩ STRING ∩ GO in-vocab) | — | Phases 2–9 |
| `nonzero_count_expected` | **2039** (constant across layers) | — | Phase 0 step 6 |
| `copole_K` (top-*K* / bottom-*K*) | **52** | 20–100 | Phase 2 |
| `copole_null_n` | **500** | 200–2000 | Phase 2 |
| `intrinsic_dim_method` | **TwoNN** (Facco 2017) | correlation dim, participation ratio | Phase 1 |
| `intrinsic_dim_n_samples` | **2000 random genes, 3 seeds** | — | Phase 1 |
| `effective_rank_method` | **Roy-Vetterli 2007** (entropy-exp) | — | Phase 1 |
| `tf_vs_target_subspace` | **Joint SV2–SV7** (6D, primary) | SV5–SV7 (3D), SV2–SV4 (3D), full SVD | Phase 5 step 1 |
| `tf_count` | **51** (from TRRUST ∩ vocab) | — | Phase 5 |
| `target_count` | **144** (non-TF, TRRUST target) | — | Phase 5 |
| `classifier` | **multinomial cosine-similarity to class centroid** | logistic regression | Phase 5 |
| `classifier_null_n` (permutations) | **100** | 100–1000 | Phase 5 |
| `edge_positives` | **589** (TRRUST) | — | Phase 5 step 5 |
| `edge_negatives` | **2351** (TF-matched non-target) | — | Phase 5 step 5 |
| `coexpression_residualizer` | **OLS** | GBDT | Phase 5 step 3 |
| `bootstrap_n` | **100** | 50–500 | Phase 5 step 3 |
| `string_confidence_primary` | **≥ 0.7** (1022 pairs) | — | Phase 4 step 1 |
| `string_confidence_gradient` | **≥ 0.4** (3092 pairs, 5 quintiles) | — | Phase 4 step 3 |
| `string_quintile_breakpoints` | **[0.40, 0.46, 0.53, 0.64, 0.80, 1.00]** | — | Phase 4 step 3 |
| `go_compartment_terms` | **8** (secretory pathway) | up to 591 (BP negative control) | Phase 3, 9d |
| `cell_type_markers` | **17 markers across 7 types** | — | Phase 6 |
| `bcell_markers` | **7** (CD19, CD79A, MS4A1, BLK, VPREB3, FCRL1, PAX5) | — | Phase 7 |
| `bcell_precision_at_k` | **10** | 5–20 | Phase 7 step 3 |
| `bcell_bootstrap_n` | **500** | 200–2000 | Phase 7 step 3 |
| `bcl6_neighbours_k` | **20** | 10–50 | Phase 7 step 8 |
| `angle_centroids` | **GC = mean(BATF, BACH2, PAX5); Plasma = mean(IRF4, IRF8); anchor = B-cell centroid** | — | Phase 7 step 6 |
| `ffl_permutation_baseline` | **t ≈ 0.5** (symmetry-corrected) | NOT t = 0 | Phase 9c |
| `ph_null_primary` | **degree-preserving curveball rewiring** | feature-shuffle as weak control | Phase 9a |
| `go_bp_terms` | **591** | — | Phase 9d |
| `autoloop_iterations` | **63** (paper); **40–80** for new models | — | Phase 10 |
| `autoloop_max_hypotheses_per_iter` | **2–3** (with ≥ 1 novel) | — | Phase 10 |
| `autoloop_retirement_policy` | **2 consecutive negative results** | — | Phase 10 |
| `autoloop_min_portfolio_size` | **10** (target 12–16) | — | Phase 10 |

## 8. Validation

A successful pipeline run must reproduce the paper's **sanity-check signatures** on scGPT + Tabula Sapiens immune before any new-model claim is trusted:

1. **Effective-rank collapse** (Phase 1): 23.6 at layer 0 → **1.6 at layer 11**, monotonic (Spearman ρ = −1.000). Tolerance ±0.2.
2. **SV1 variance fraction on 195-gene submatrix** (Phase 1): **19% → 77%** across layers 0 → 11. Full-vocabulary: 53.7% → 93.4%. Tolerance ±2%.
3. **TwoNN intrinsic dim** (Phase 1): **32.6 → 18.1** (44.6% reduction) across 3 seeds. Tolerance ±1.
4. **Feature-shuffle null effective rank at layer 11** (Phase 1): **28.9** (17.6× observed). **If feature-shuffle gives ER near 1.6, the shuffle is not breaking the structure — check the shuffling code.**
5. **SV1 layer-11 extracellular enrichment** (Phase 3): GO:0005615 OR = **6.37**, *p* = 2.6×10⁻⁴. Gene-label-shuffle empirical *p* = **0.004**. Tolerance on OR: ±0.5.
6. **Mitochondrial transient enrichment** (Phase 3): peak OR **23.3 at layer 3**. If no transient mitochondrial peak appears at early layers, SV1 has lost the secretory-pathway ordering.
7. **SV2 PPI co-pole *z*-scores** (Phase 4 / Table S4): all 12 layers significant; layer 1 has the strongest *z* = **6.52**; layer 11 *z* = **5.45**. Tolerance ±0.5.
8. **STRING confidence gradient** (Phase 4 step 3 / Table S5): quintile-averaged *z*-scores **1.00 → 1.48 → 2.09 → 3.18 → 5.02**; Spearman ρ = **1.000**, *p* = 0.017. **If ρ < 0.9, the confidence-graded encoding is broken.**
9. **SV2–SV4 orthogonality** (Phase 4): pairwise Pearson *r* < 0.25 between the three axes. If any pair is strongly correlated, the "distinct PPI subsets" claim fails.
10. **Joint SV2–SV7 TF-vs-target AUROC** (Phase 5 / Table S1): mean across 12 layers **0.744**; peak **0.789 at layer 3**; all layers *p* < 0.01 (null mean ≈ 0.50). Tolerance on mean: ±0.02.
11. **Cross-seed robustness** (Phase 8 / Table S3): 3 seeds give mean AUROC 0.744, 0.753, 0.757; cross-seed SD ≈ **0.016**. Min AUROC across any seed × layer ≥ **0.687**.
12. **SV5–SV7 edge-level AUROC peaks at layer 0 = 0.602** (Phase 5 / Table S2). Monotonic depth decay Spearman ρ = **−0.958**, *p* = 9.5×10⁻⁷. Layer 9 drops below significance. **If layer 11 is still significant, the depth trend is broken.**
13. **Co-expression residualisation** (Phase 5 step 3):
    - **SV2–SV4 residualised signal null at 0/12 layers** — if this is non-null, the "SV2–SV4 encodes gene class, not regulation" claim fails.
    - **SV5–SV7 residualised rank-biserial r_rb = 0.148 at layer 0**, permutation *p* < 0.001, 100/100 bootstrap resamples positive. If the residual signal vanishes, SV5–SV7 is not co-expression-independent.
14. **Cell-type clustering AUROC** (Phase 6): **0.851** (primary); cross-seed mean **0.789 ± 0.016**. Tolerance ±0.02.
15. **B-cell precision@10 *z*-score** (Phase 7): **7.55**; T-cell and dendritic cell at / below chance.
16. **GC-TF convergence** (Phase 7): BATF rank 1,510 → 189; BACH2 rank 611 → 146; Spearman ρ < −0.84 for both; PRDM1 Spearman ρ = −1.000. **If BATF starts at rank < 500, the "delayed convergence" signature is wrong.**
17. **GC-plasma angle** (Phase 7 step 6): 77° (L0) → 94° (L11). **Monotonic increase** is required for the "orthogonal differentiation programs" claim.
18. **BCL6 metabolic isolation** (Phase 7 step 8): 10/20 BCL6 neighbours are metabolic genes at every layer; zero B-cell markers. **If BCL6 converges into the B-cell cluster, the "metabolically isolated repressor" claim fails and Phase 7 step 8 must be rewritten.**
19. **B-cell intrinsic dim lineage-specific compression** (Phase 7 step 7): B-cell 8.2 → 5.1 (ρ = −0.951); T-cell ρ = +0.287 (no compression); myeloid ρ = +0.699 (reversed).
20. **Persistent homology cautionary tale** (Phase 9a): 11/12 tests *p* < 0.05 under feature-shuffle null; **0/24 tests significant** under degree-preserving rewiring null. **Any pipeline that reports a "topological signal" without running the degree-preserving null is broken.**
21. **Cross-model alignment** (Phase 9b): scGPT × Geneformer raw cosine 0.825; permutation test *p* > 0.3; **Geneformer static embeddings independently encode STRING PPI at *p* = 7.8×10⁻¹²⁷**; B-cell attractor precision@10 = 0.000 for Geneformer.
22. **ER-AUROC confound** (Phase 9e): raw ρ = 0.855; partial ρ = **−0.045** after controlling for layer depth. **If the partial correlation is still > 0.3, the layer-control is incomplete.**

## 9. Known pitfalls

1. **Pre-contextual vs contextualised embeddings.** The paper uses the **pre-contextual** gene embedding, which is the gene's hidden-state activation **at its token position, averaged across 2,000 input cells**. It is NOT the input lookup embedding (though those are static) and NOT the cell-level CLS embedding. The distinction matters for layer 0 — at layer 0 the pre-contextual embedding is after any input-token projection but before attention. Verify your adapter extracts the correct quantity by comparing layer-0 embeddings across two different input cell batches — if they're identical, you're extracting the static input embedding (wrong); if they're close but not identical, you're extracting the pre-contextual post-attention-layer-0 activation (right for the paper's convention).
2. **Sparsity artefact on edge-level AUROC.** scGPT's 4,803-position vocabulary has only **2,039 non-zero-norm positions** across all 12 layers (the rest are unused input slots). If your vocabulary-construction code accidentally drops or includes different gene sets at different layers, the edge-level AUROC depth trend will appear to have a sparsity-driven artefact. **Verify the non-zero count is exactly constant across layers** before trusting any depth decay claim (Phase 5 step 6).
3. **Feed-forward loop null must be 0.5, not 0.** See Phase 9c. The naïve baseline for betweenness statistics in high-dimensional spaces is **not zero** — a randomly positioned intermediate gene is on average halfway between two endpoints by symmetry. Use permutation-corrected baselines.
4. **Persistent homology needs the degree-preserving null.** See Phase 9a. Feature-shuffle nulls are too weak and will produce spurious topological significance. **The source paper's Phase 9a is the canonical example of a finding that looked significant under a weak null but vanished under a stricter one.** Always require the stricter null to pass before promoting.
5. **SV5–SV7 is a specific 3D subspace**, not any 3D subspace containing regulatory information. The paper did a **systematic window scan** over SV(k, k+2) for all k and found SV5–SV7 as the only window that produces a significant co-expression-independent regulatory signal at early layers after Bonferroni correction. If your new model has a different architecture / different number of layers, **run the window scan** rather than hard-coding SV5–SV7.
6. **Joint SV2–SV7 classifier beats either 3D subspace alone.** The paper's Table S1 shows joint 6D is the right primary for TF-vs-target classification. Reporting only SV5–SV7 or only SV2–SV4 will understate the TF-identity signal (joint peak 0.789 vs SV5–SV7 peak 0.723, SV2–SV4 peak 0.732). Always report the joint first.
7. **Co-expression residualisation is the gold-standard regulatory control.** Raw TRRUST AUROC is inflated by the fact that co-regulated genes are also co-expressed. The paper's rank-biserial-*after*-OLS-residualisation test is how "genuine regulatory signal" is defined. **Any claim of "regulatory encoding" in a new model must pass this test, or be labelled as "gene-class encoding" instead.**
8. **Co-expression residualisation must use the SAME cells as the embedding extraction.** If the embeddings come from one cell set and the co-expression from another, the residualisation is under-powered and can spuriously retain signal. Use the same 2,000 cells.
9. **Cell-type marker panels are canonical but not exhaustive.** The paper uses 17 markers across 7 types (with 7 B-cell markers). If your new data is not human immune, you need a different marker panel. Re-derive from canonical literature for the target tissue; do NOT reuse the immune panel blindly.
10. **BCL6 metabolic isolation is a strong prediction for human immune models.** For non-immune datasets or non-human species, BCL6's neighbourhood may look different. Do not hard-code NAMPT, GLUL, PFKFB3 as the metabolic reference set — derive from BCL6's layer-0 neighbourhood in the target dataset.
11. **Cross-model alignment via raw cosine is misleading.** scGPT × Geneformer raw cosine of 0.825 looks impressive but fails a permutation test. **Always run the permutation null** for any "models agree" claim.
12. **Partial correlation is the minimum for controlling layer depth.** Any correlation between a dimensionality metric and a performance metric is confounded by layer depth if both vary monotonically with depth. The paper's ER-AUROC partial-correlation check (Phase 9e) drops the apparent ρ = 0.855 to ρ = −0.045. Always report the partial.
13. **Autonomous loop retirement is mandatory.** Running the brainstormer without retirement bloats the hypothesis space. **2 consecutive negatives = retire.** This is not optional.
14. **Every positive finding needs ≥ 1 permutation-based null.** The brainstormer will generate creative ideas; without the mandatory null-based promotion rule, you will accumulate false positives at scale.
15. **Geneformer static embeddings encode STRING PPI but NOT cell-type attractor dynamics.** The cross-model check establishes that PPI is convergent across architectures (good — suggests the finding is about biology, not about scGPT's specific inductive bias) but cell-type attractor dynamics require contextual processing. **Do not assume a positive cross-model PPI result means all findings generalise.**
16. **GO Biological Process is a negative control for SV2, not a positive finding.** Phase 9d tests 591 GO BP terms on SV2 and expects **0/591** significant. If you find BP enrichment on SV2, check for multiple-testing errors before claiming a new finding.

## 10. Quick-start for new-model evaluation

To apply this pipeline to a new single-cell model `NEWMODEL`:

1. **Implement the Phase 0 adapter.** Export residual-stream hidden states at every layer as `layer_gene_embeddings.npy` with shape `[n_layers, n_genes, d_model]`. Verify the **non-zero count is constant across layers** (critical sparsity control). Run for 3 fine-tuning seeds.
2. **Run Phase 1** and reproduce the effective-rank collapse and intrinsic-dim reduction signatures. A trained biological model should show monotone spectral collapse; a randomly-initialised model will not.
3. **Run Phase 2 + Phase 3** on SV1 and verify the secretory-pathway ordering (mitochondria in early layers, ER lumen progressively, extracellular consistently across layers). This is the simplest positive-signal check — if SV1 does not encode subcellular localisation, the model has not learned cellular compartmentalisation.
4. **Run Phase 4 (PPI).** Verify STRING co-pole significance on SV2 across all 12 layers and compute the confidence-gradient Spearman ρ. Report whether ρ = 1.000 replicates. This is the **most quantitatively striking finding**; a new model that replicates it is structurally encoding PPI confidence.
5. **Run Phase 5 (regulatory).** Report joint SV2–SV7 TF-vs-target AUROC and the co-expression-residualised SV5–SV7 edge-level AUROC. The subspace window scan is mandatory here (see Pitfall #5).
6. **Run Phase 6 (cell-type).** Verify AUROC and per-type precision@10. **Caveat:** cell-type attractor dynamics require contextual processing (Phase 9b negative finding for Geneformer static). If your new model is purely static-embedding, expect this to be near chance.
7. **Run Phase 7 (B-cell attractor).** Only for human immune data. Track BATF, BACH2, BCL6, PRDM1 across layers; compute GC-plasma angle and B-cell intrinsic-dim compression. Report delayed convergence at layer 3 (or whatever the equivalent is in `NEWMODEL`'s architecture).
8. **Run Phase 8 (cross-seed)** with 3 fine-tuning seeds. Compute CKA layer-by-layer. Report cross-seed SD on joint SV2–SV7.
9. **Run Phase 9 (negative controls).** All 5 sub-phases are mandatory. **Especially Phase 9a:** re-run any persistent-homology positive findings under the degree-preserving null before promoting.
10. **Optional: Phase 10 autonomous loop.** For open-ended hypothesis discovery on the new model, fork the loop infrastructure, seed the brainstormer with the 13 hypothesis families from the source paper, and run 40–80 iterations. Retire branches aggressively.
11. **Cross-pipeline integration.** The residual-stream geometry is complementary to attention-based interpretability. Run this pipeline alongside `pipelines/attention-grn-extraction-and-evaluation.md` on the same `NEWMODEL` and report where the two agree and where they dissociate. Expected agreement: PPI at early layers, TF-vs-target identity at mid-depth. Expected dissociation: attention attributes TF-target regulation to late layers (Geneformer L15), while residual-stream SVD attributes it to early layers (L0–L3 in SV5–SV7) — because attention operates on contextualised stream while residual-stream SVD operates on the pre-contextual decomposition. Reporting both gives a complete picture.
12. **Emit a machine-readable hypothesis registry** with the same schema as the source paper (family, iteration, hypothesis, method, result, null outcome, decision) and a final summary table in Supp Table S6 format.
