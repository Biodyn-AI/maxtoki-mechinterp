# Pipeline: Topological and Geometric Structure Screening of Biological Foundation Models (141-Hypothesis Autonomous Audit)

## 1. Overview

This pipeline maps the boundary between **real topological/geometric structure** and **statistical artefact** in the internal representations of single-cell foundation models (scGPT and Geneformer V2-316M), using an autonomous executor-brainstormer hypothesis-screening loop that systematically generates, tests, and retires geometric claims under a hierarchical null-model framework. The source paper documents **141 hypotheses across 52 productive iterations** (53 total, 1 init failure) organized into **9 hypothesis families** — persistent homology, manifold distances, cross-model alignment, community structure, directed topology, intrinsic dimensionality, sparse descriptors, motif-community interactions, and other geometric properties (curvature, hyperbolicity, anisotropy) — plus 30 cross-cutting methodological variants (topology stability tests, null-framework development, split-design validation). Of 141 tested hypotheses, approximately **27 show robust positive effects, 21 are inconclusive or partial, and 63 are decisively negative** under feature-shuffle baselines; under the strictest max-null audit, **fewer than 15 survive**.

The pipeline is **model-agnostic**: a Phase 0 adapter pre-extracts per-layer gene embeddings from any transformer-based single-cell foundation model, and all downstream hypothesis families consume a standard `[n_genes, d_model]` matrix per (model, layer, tissue-domain) combination. The worked examples in the source paper are scGPT (12 transformer layers) and Geneformer V2-316M (18 layers) evaluated on three Tabula Sapiens tissue domains (lung, immune, external-lung = held-out lung dataset) with PCA reduction to 20 components.

The pipeline is **complementary** to the earlier residual-stream spectral-geometry pipeline (`residual-stream-spectral-geometry.md`, arxiv 2602.22247, which focused on 13 hypothesis families ending at SVD biological axes, B-cell attractor dynamics, and the 6D TF-vs-target classifier) and to the SAE-based mechanistic interpretability mega-pipeline (`sparse-autoencoders/README.md`, arxiv 2603.02952/01752/11940 — SAE feature atlases, causal circuit tracing, exhaustive mapping and steering). All three lenses (spectral, SAE-feature, and topological/geometric) are needed for a complete mechanistic audit of a biological foundation model. The topological/geometric pipeline is the **most methodologically conservative** of the three — it places the strongest emphasis on null-model hierarchy, cross-model consistency, disjoint gene-pool splits, and the strict max-null audit.

**Five-layer hierarchy of findings** (source paper §4.1, from most robust to most fragile):

1. **Layer 1 — Coarse geometric consistency across models (most robust).** scGPT and Geneformer, trained independently with different architectures, agree on the global geometric organisation of gene space. **H24: cross-model CCA alignment** yields mean canonical correlation *r* = **0.80**, pairwise-distance Spearman 0.75, gene-level top-1 retrieval 72%, combined Fisher *p* = **3.17 × 10⁻⁵** across all three tissue domains. **Even the ranking of which geometric features are most informative is conserved across models** (H17: centred cosine is top-ranked in all three domains for both models, *p* = 0.037). **But gene-level correspondence fails**: 19 distinct methods (Gromov-Wasserstein, optimal transport, topology-signature distillation, cycle-consistent mapping, codebook transport, chart/sheaf alignment, etc.) all yield top-1 retrieval **below 1%**. Models agree on "shape" but not on individual gene placement.
2. **Layer 2 — Non-trivial topology and community structure (robust under feature-shuffle controls).** **H01/H03: persistent homology** via Ripser on PCA-projected embeddings (350 genes, 20 dims) finds significantly more H1 structure than shuffled embeddings in **11/12 scGPT layers in lung (*p* < 0.01 in 9/12, top layer L0 *z* = 3.21, *p* = 0.006), 12/12 in immune (mean Δ = +12.1 lifetime units), 12/12 in external-lung (mean Δ = +12.5)**. Topology stable under bootstrap, PCA dimension scan {10, 20}, and k ∈ {8, 12, 16}. **H47/H49: bifiltration-like cycle-rank** independently confirms (ΔAUROC +0.006, Fisher-significant in 6 domain-splits, 13/24 rows at *p* < 0.05). **But topology vanishes under degree-preserving kNN rewiring nulls** (0/24 layer-tests significant across 6 iterations of refined rewiring controls) — the topology depends on fine-grained neighbourhood structure rather than a deep geometric invariant.
3. **Layer 3 — Manifold distance hierarchy and directed geometry (moderate).** Clear hierarchy emerges: **Euclidean < Geodesic (H13, ΔAUROC +0.013, 7/12 layers significant in both gene-pool splits) < Diffusion (H16, ΔAUROC +0.017, Fisher *p* 10⁻⁵ to 10⁻¹¹) < Triangle-defect spectrum (H69/H70, ΔAUROC +0.026, positive null-gap in 6/6 domain-splits under hard-null calibration)**. The triangle-defect spectrum becomes the baseline for all subsequent work in the campaign. **But diffusion distance uplift partially erodes under coexpression-matched null** (+0.017 → +0.008, loses significance in 2/3 domains), whereas the triangle-defect spectrum survives matched-random-third-order nulls.
4. **Layer 4 — Signed motif-community interactions (strongest single signal, but requires external annotations).** **H123: signed motif-community hardening** combines Louvain communities detected on the kNN graph of gene embeddings with signed regulatory motif annotations from TRRUST, with TF-identity-preserving sign shuffles + TF/target degree strata as null controls. **ΔAUROC +0.094, positive in every single test row (22/22), and surviving the null-gap criterion in every domain-split group tested — the only hypothesis in the entire 141-screen campaign to achieve complete null-gap coverage under the most stringent controls.** Requires knowledge of TF identity and regulatory sign (activation/repression) — information the model provides indirectly, not explicitly.
5. **Layer 5 — Localised robustness under strict auditing (fragile beyond immune tissue).** **H141: strict max-null audit** compares the observed signal against the **maximum** of the 95th percentile across ALL null families simultaneously. Overall mean strict margin = **−0.005** — on average, the strongest null family can explain the observed signal. Strict-positive support in only **3/9 domain-split groups**. By domain: **Immune: +0.012 (robust)**; **Lung: −0.008 (marginal)**; **External-lung: −0.023 (fragile)**. **Robust signal concentrates in immune tissue.** This pattern appeared throughout the campaign (H31 diffusion-distance uplift immune-only at *p* = 6.8 × 10⁻⁸; H48 cross-model motif overlap immune-only at *p* = 3.1 × 10⁻⁴; H62 biologically anchored community alignment immune-only at *p* = 0.001) — the strict audit simply made the pattern impossible to ignore.

**The most valuable output of this pipeline is not any single positive finding but the systematic documentation of what fails.** 70+ hypotheses that appeared promising (ΔAUROC > 0.03 under feature-shuffle) vanished under strict controls; 19 cross-model gene-level correspondence methods failed uniformly; paradoxically, adding biological annotations (GO, STRING, Reactome, ontology sheaves) consistently **inflated raw effect sizes while systematically destroying null-gap robustness** (H123 → H124 → H127 → H130 → H138: full null-gap → fewer than half → 2/9 → 0/9 → 0/9). This counterintuitive pattern is a cautionary tale: **in interpretability research, more biological priors are not always better — they can introduce confounds that make it harder to isolate model-internal structure**.

## 2. Source

- **Paper:** Kendiukhov, I. (2026). *What Topological and Geometric Structure Do Biological Foundation Models Learn? Evidence from 141 Hypotheses.* arXiv:2602.22289 [q-bio.QM], 5 Mar 2026. 13 pages (main text + references).
- **Local PDF:** `references/2602.22289_Kendiukhov_topology_141_hypotheses.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/topology-biomechinterp2 — pinned commit `32ec76d816b7b4ded3e9aa16a87624e1cd404f3c` (2026-02-25, "Initial release: autonomous topology hypothesis screening (141 hypotheses, 53 iterations)"). Local clone at `repos/topology-biomechinterp2/`. **Note:** paper's "Data and Code Availability" section lists https://github.com/ihorkendiukhov/biomechinterp which does not currently resolve — the actual release is under the `Biodyn-AI` organisation following the same naming pattern as the earlier spectral-geometry paper's `topology-biomechinterp1`.
- **Companion pipelines in this project:**
  - `residual-stream-spectral-geometry.md` (arxiv 2602.22247) — a direct predecessor that tested 183 hypotheses in 13 families over 63 iterations using a Claude-based executor-brainstormer loop, focused on SVD biological axes, 6D TF-vs-target classification, and B-cell germinal-centre attractor dynamics. This pipeline (arxiv 2602.22289) uses **OpenAI Codex 5.3 at extra-high reasoning**, a refined 9-family taxonomy, and explicit hierarchical null models + disjoint gene-pool splits.
  - `attention-grn-extraction-and-evaluation.md` (arxiv 2602.17532) — attention-based interpretability, which established that attention captures co-expression rather than causal regulation. This pipeline is the residual-stream-topological counterpart.
  - `sparse-autoencoders/README.md` (arxiv 2603.02952/01752/11940) — SAE-based mechanistic interpretability mega-pipeline (three stages: atlas, circuit tracing, exhaustive mapping + steering). Complementary lens to topological analysis.
- **Key foundational references:**
  - **Topology / persistent homology:** Edelsbrunner & Harer 2008 (survey); Zomorodian & Carlsson 2005 (computational persistent homology); Bauer 2021 (Ripser); Carlsson 2009 (topology and data); Rabadán & Blumberg 2019 (topological data analysis for genomics); Sreejith et al. 2016 (Forman curvature).
  - **Manifold geometry:** Coifman & Lafon 2006 (diffusion maps); Peyré & Cuturi 2019 (computational optimal transport — including Gromov-Wasserstein used in H108); Hardoon et al. 2004 (canonical correlation analysis).
  - **Community detection:** Blondel et al. 2008 (Louvain).
  - **Statistical methodology:** Meinshausen & Bühlmann 2010 (stability selection); Rapid scattered-seed lasso selection.
  - **Single-cell foundation models:** Cui et al. 2024 (scGPT); Theodoris et al. 2023 (Geneformer); Yang et al. 2022 (scBERT); Hao et al. 2024 (scFoundation); Lopez et al. 2018 (scVI); Luecken & Theis 2019 (scRNA-seq analysis).
  - **Mechanistic interpretability background:** Park et al. 2023 (linear representation); Elhage et al. 2022 (superposition); Conmy et al. 2023 (automated circuit discovery); Nanda et al. 2023 (progress measures for grokking); Olah et al. 2020 (circuits); Lucken & Theis 2019.
  - **Biological annotation databases:** Ashburner et al. 2000 (Gene Ontology); Garcia-Alonso et al. 2019 (DoRothEA); Han et al. 2018 (TRRUST v2); Szklarczyk et al. 2019 (STRING v11); Aibar et al. 2017 (SCENIC); Schaefer et al. 2012 (TFCat); Dibaeinia & Sinha 2020 (SERGIO simulator).
  - **GRN inference benchmarks:** Pratapa et al. 2020 (BEELINE); Saelens et al. 2019 (trajectory inference comparison).
  - **Companion Kendiukhov papers:** 2025 (`attention-grn-extraction-and-evaluation.md`); 2026 (`residual-stream-spectral-geometry.md`).

## 3. Inputs

### 3.1 Target models

The pipeline expects **two or more independently trained single-cell foundation models** for the cross-model consistency analyses in Phase 4. Source paper uses:

- **scGPT** (Cui et al. 2024) — **12 transformer layers**, continuous-value gene tokens, pre-trained on ~33M cells. HuggingFace or direct from the scGPT authors.
- **Geneformer V2-316M** (Theodoris et al. 2023) — **18 transformer layers**, **18 attention heads per layer**, rank-value tokens, pre-trained on ~100M cells. HuggingFace `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M`.

For a new-model audit, run this pipeline with the target model plus at least one reference model (scGPT or Geneformer) to enable the cross-model CCA alignment in Phase 4.

### 3.2 Single-cell datasets (three tissue domains)

| Domain | Source | Role in pipeline |
|---|---|---|
| **Lung** | Tabula Sapiens (Jones et al. 2022) | Primary domain for in-sample tuning and hypothesis development |
| **Immune** | Tabula Sapiens | Secondary domain — turns out to be the most robust across all null families (see §1 Layer 5) |
| **External-lung** | A held-out lung dataset not used for parameter tuning, with three tissue-independent cell-type splits reflecting different predictability sources | Strictest held-out validation domain |

Each domain provides a standard per-gene per-layer embedding matrix after Phase 0.

### 3.3 Regulatory ground truth

All hypotheses that involve "regulatory proximity" or "regulatory edges" need ground-truth regulatory annotations. The source paper uses a union of four databases:

| Database | Type | Usage |
|---|---|---|
| **DoRothEA** (Garcia-Alonso et al. 2019) | Curated TF→target, confidence-tiered | Primary regulatory ground truth for edge classification |
| **TRRUST v2** (Han et al. 2018) | **Signed** TF→target edges (activation / repression) | Signed regulatory motif annotations — critical for H116/H123 (the strongest finding in the campaign) |
| **STRING v11** (Szklarczyk et al. 2019) | Protein-protein interaction scores | Null-stratification and PPI co-membership tests |
| **Gene Ontology** (Ashburner et al. 2000) | Functional co-membership | Additional biological annotation (used in H94, H127–H130, H135, H138 — with the paradoxical result that more annotations degrade null-gap robustness) |

For each (model, domain, layer, split) tuple, the pipeline restricts to genes shared between the regulatory ground truth and the model's vocabulary — typically ~320 genes per domain.

### 3.4 Single-cell annotation databases for GO / ontology extensions

GO BP 2024-01, KEGG Release 109, Reactome v87 — used in a subset of Phase 10 hypotheses (the ontology-extension family, which paradoxically destroys null-gap robustness despite increasing raw effect sizes).

## 4. Outputs

Per model × domain × iteration, the pipeline emits (repo layout mirrors the source):

1. **Pre-extracted gene embedding matrices** (NPY or pickle): `data/embeddings/{model}_{domain}_layer{L}.npy`. Shape `[n_genes, d_model]`. For scGPT d_model=512; for Geneformer d_model=1,152. PCA-reduced to 20 components for downstream geometric analyses. ~320 genes per domain (after intersection with regulatory ground truth vocabulary).
2. **Per-iteration experiment directory**: `iterations/iter_{XXXX}/` containing:
   - `run_iter{XXXX}_screen.py` — auto-generated experiment script
   - `executor_hypothesis_screen.json` — structured outcome record per hypothesis (id, name, family, split_regime, method, status, result_direction, decision)
   - `executor_iteration_report.md` — narrative summary
   - `*.csv` — per-hypothesis numerical results
   - `executor_prompt.md`, `executor_stdout.log`, `executor_*.log` — execution artefacts
   - `brainstormer_structured_feedback.md`, `brainstormer_hypothesis_roadmap.md`, `brainstormer_next_iteration_brief.md` — brainstormer outputs for the next iteration
3. **Canonical hypothesis registry:** `reports/autoloop_master_log.md` (~1,500 lines for the source paper's 53 iterations) — every hypothesis's: ID, family, test, raw ΔAUROC, null-gap result, split-regime pass rate, promotion/retirement decision.
4. **Per-family summary reports:** `reports/family_{name}.md` — aggregated statistics per hypothesis family (e.g., `persistent_homology.md`, `manifold_distance.md`, `cross_model.md`).
5. **Cross-model alignment results:** `data/h24_cross_model_cca/` — canonical correlations per domain, pairwise-distance Spearman, gene-level top-1 retrieval accuracies, combined Fisher *p*-values.
6. **Persistent homology artefacts:** per-layer per-domain H1 persistence diagrams, lifetime-delta distributions vs feature-shuffle null, rewiring-null outcomes. Ripser (Bauer 2021) output stored as JSON.
7. **Manifold distance matrices:** per (model, domain, layer): Euclidean, geodesic (kNN shortest path), diffusion (Coifman-Lafon 2006 with *t* ∈ {1, 2, 4, 8}), triangle-defect spectrum at k ∈ {8, 12, 16}, convexity-deficit, detour-based graph-geometry metrics.
8. **Community detection graphs:** Louvain communities per (model, domain, layer) on kNN graph at k=12.
9. **Stability-selected descriptor set:** `data/h91_stability_selection/` — Jaccard similarity of feature sets across bootstrap seeds (target: 0.65, sign agreement 1.0), final feature vector per gene pair, cross-validated logistic classifier weights.
10. **H123 strongest-finding artefacts:** `data/h123_signed_motif_community/` — per-domain-split ΔAUROC (target: +0.094), per-row pass/fail (target: 22/22 positive), null-gap per domain-split group (target: positive in 6/6).
11. **Null distributions per family:** `data/null_distributions/{family}/` — per-hypothesis null distributions from feature-shuffle (20–24 replicates), label-permutation (100–200), degree-preserving rewiring (*k*=12 adaptive, 24 replicates), coexpression-matched null (gene-expression Pearson rank-binned).
12. **Strict max-null audit output (H141):** `data/h141_strict_max_null/` — per-domain strict margin (target: immune +0.012, lung −0.008, external-lung −0.023), strict-positive support count (target: 3/9 domain-split groups, 15/25 individual test rows).
13. **Autonomous loop state:** `runtime/loop_status.json`, `runtime/loop_events.jsonl`, final brainstormer output, `STOP` sentinel. Per-iteration timing and LLM-call metadata.
14. **Figures** (Figure 1 through Figure 8 of the paper): `figures/fig_persistent_homology.png`, `fig_cross_model_cca.png`, `fig_distance_hierarchy.png`, `fig_motif_community.png`, `fig_hypothesis_outcomes.png`, `codex_h123_vs_h139_null_gap.png`, `codex_h140_scaling_gain.png`, `codex_h141_strict_margin_audit.png`.
15. **Final 5-layer hierarchy report** (`reports/layered_findings.md`): synthesis of all 141 hypotheses into the 5-layer hierarchy (§1), with robust-positive verdict per layer.

## 5. Dependencies

- **Python ≥ 3.10** with: numpy ≥ 1.24 (< 2.0), pandas ≥ 2.0, scipy ≥ 1.10, scikit-learn ≥ 1.3, matplotlib ≥ 3.7.
- **Topology:** `ripser ≥ 0.6`, `persim ≥ 0.3` (persistent homology).
- **Network analysis:** `networkx ≥ 3.1`, `python-igraph` (for Louvain via `leidenalg` or native Louvain).
- **Foundation models:** `torch ≥ 2.0`, `transformers ≥ 4.30`, `scgpt ≥ 0.2`, `geneformer ≥ 0.1`, `scanpy ≥ 1.9`, `anndata ≥ 0.10`.
- **LLM orchestration (Phase 11):** **OpenAI Codex CLI (`codex` binary)** — the source paper uses **gpt-5.3-codex** at **`reasoning_effort="xhigh"`** ("extra high") as the backend for both executor and brainstormer agents. For a Claude-based alternative, use the `run_claude_topology_autoloop.py` pattern from `repos/topology-biomechinterp1/` (the spectral-geometry predecessor paper) and adapt the prompt templates.
- **Reference databases:** DoRothEA (confidence-tiered), TRRUST v2 (signed), STRING v11 (PPI), Gene Ontology 2024-01.
- **Compute:** the source paper does not document specific hardware; the loop is CPU-bound for most geometric analyses. Embedding extraction (Phase 0) requires GPU/MPS for transformer forward passes. The 53-iteration loop total wall-clock is not reported but is dominated by executor script execution time plus LLM call latency (Codex at xhigh reasoning ≈ minutes per call).
- **Environment file:** `../repos/topology-biomechinterp2/requirements.txt`.

## 6. Methodology

The pipeline runs in thirteen phases. Phases 0–3 are setup; Phases 4–10 are the nine hypothesis families plus stability-selection; Phase 11 is the autonomous loop orchestrator; Phase 12 is the strict max-null audit; Phase 13 is the synthesis.

---

### Phase 0 — Model adapter: pre-extract per-layer gene embeddings per tissue

**Purpose:** for each (model, tissue-domain) combination, produce a standard `[n_genes, d_model]` matrix per layer by averaging hidden-state vectors across cells.

**Steps:**

1. **Load each target foundation model** (scGPT whole-human; Geneformer V2-316M). For a new model, add it to the same loop — it must expose per-layer residual-stream hidden states.
2. **For each of three tissue domains (lung, immune, external-lung):**
   - Load a held-out cell batch from Tabula Sapiens (Jones et al. 2022) not used for parameter tuning. The external-lung domain is a genuinely held-out lung dataset with three tissue-independent cell-type splits.
   - Run the pretrained model on each cell. Capture per-layer hidden-state vectors.
3. **Per-gene averaging.** For each gene *g*, average its hidden-state vector across cells in the domain to produce a single embedding vector per gene per layer. For scGPT this yields 12 per-layer matrices; for Geneformer 18 per-layer matrices.
4. **Vocabulary restriction.** Restrict to genes shared between the model's vocabulary and the regulatory ground-truth databases (DoRothEA, TRRUST, STRING). Typical result: **~320 genes per domain** after intersection.
5. **PCA dimensionality reduction.** Apply PCA to **20 components** (for computational tractability and to mitigate the curse of dimensionality in downstream topological analyses). Retain enough variance for downstream analyses — the paper does not report the per-layer variance-explained, but all subsequent geometric analyses run on the 20-dim PCA output.
6. **Persist artefacts.** `data/embeddings/{model}_{domain}_layer{L}.npy` (PCA-reduced) plus `data/embeddings/{model}_{domain}_layer{L}_full.npy` (pre-PCA, retained for sensitivity analyses).

**Sanity signature:** for Geneformer V2-316M L0 on the lung domain, expect a `[~320, 20]` matrix with non-degenerate row norms (all rows non-zero) and no duplicate rows (no two genes should have identical embeddings at any layer).

**Code reference:** the source paper's data extraction is documented in §2.2 of the paper; the embedding artefacts are in `data/` of the repo (not versioned as the loop iterations were run on pre-extracted files).

---

### Phase 1 — Gene-pool construction and regulatory ground truth

**Purpose:** build disjoint source- and target-gene-pool splits to prevent information leakage during evaluation, and to enable the Phase 3 dual-split design.

**Steps:**

1. **Load the unified regulatory edge set.** For each tissue domain, construct an edge list `(TF, target, sign, source_database)` from DoRothEA (confidence tiers A, B, C), TRRUST v2 (signed), STRING v11 (≥ 700 confidence), and GO BP functional co-membership. The primary evaluation uses the union; sensitivity analyses use individual databases.
2. **Source-disjoint split.** Partition transcription factors into disjoint train/test sets. In the test set, the TFs (regulatory sources) are **entirely absent** from the training set — the model must predict regulatory edges using unseen regulators.
3. **Target-disjoint split.** Partition target genes similarly. The regulated target genes in the test set do **not appear** on either side of the train/test boundary for the target role.
4. **Dual-split evaluation.** A hypothesis is considered robust only if it holds in **both split regimes** across seeds. This is the disjoint gene-pool split design that underlies all ΔAUROC evaluations in subsequent phases.
5. **Persist splits:** `data/gene_pool_splits/{domain}_source_disjoint_seed{S}.json`, `data/gene_pool_splits/{domain}_target_disjoint_seed{S}.json`. At least 3 seeds per split regime.

**Critical design note:** this disjoint-gene-pool design is **more rigorous** than the standard random-split used in most interpretability papers. It prevents a common failure mode where a model has "memorised" a TF's name and can predict its targets via shortcut features rather than genuine geometric structure.

**Code reference:** `repos/topology-biomechinterp2/iterations/iter_0005/run_iter0005_screen.py` through `iter_0008` (the paper's initial split-design validation; see `reports/autoloop_master_log.md` for the exact iterations).

---

### Phase 2 — Hierarchical null-model infrastructure

**Purpose:** construct four increasingly stringent null models for every geometric test.

**The four null families** (paper §2.3):

1. **Feature-shuffle null** — randomly permute embedding features across genes (20–24 replicates per condition). Preserves geometric structure while destroying gene-to-embedding correspondence. **Weakest null.** Most hypotheses pass this; the 70+ that vanish under stricter nulls all looked promising here.
2. **Label-permutation null** — randomly permute regulatory labels (100–200 replicates). Preserves the base rate of positive edges in the evaluation set.
3. **Degree-preserving rewiring null** — rewire the k-nearest-neighbour graph (typically *k* = 12, with adaptive selection from 10–35 for graph-connectivity-based tests) while preserving each gene's degree. **Specifically destroys the persistent-homology signal** (see Phase 5): 0/24 layer-tests significant across 6 iterations of progressively refined rewiring controls.
4. **Coexpression-matched null** — compute pairwise absolute Pearson correlations of gene expression across cells, bin gene pairs by rank into 5 bins, and sample negative edges matched in coexpression to the positive regulatory edges. **Strongest baseline null** for regulatory-edge discrimination. Specifically strips the "coexpressed genes are geometrically proximal" confound.

**Persistence:** per-hypothesis null distributions stored as `data/null_distributions/{family}/{hypothesis_id}_{domain}_{split}_{null_type}.npy`.

**Strict max-null (introduced in Phase 12):** the maximum of the 95th percentile across ALL four null families simultaneously. A hypothesis passes strict-max-null if the observed signal exceeds this maximum in the domain-split group.

**Code reference:** null models implemented across multiple iterations starting from `iter_0002` (topology stability tests); see `reports/autoloop_master_log.md`.

---

### Phase 3 — Evaluation design: ΔAUROC against coexpression-adjusted baseline

**Purpose:** compute the primary evaluation metric for every hypothesis.

**Metric:** ΔAUROC = observed AUROC − 95th-percentile of the coexpression-matched null distribution. Positive ΔAUROC means the geometric feature discriminates regulatory edges beyond what coexpression alone explains. Two reporting modes:
- **Per-row ΔAUROC:** mean across folds, seeds, layers for a specific (hypothesis, domain, split-regime) row.
- **Domain-split pass rate:** fraction of (rank-split × domain) combinations with positive null-gap. The paper's critical threshold for "robust" is **6/6 domain-splits positive**.

**Steps:**

1. For each hypothesis, compute the raw AUROC of its geometric feature on the regulatory-edge classification task.
2. Compute the same AUROC on each null-generated dataset (for each of the four null families).
3. Compute ΔAUROC and check whether the observed value exceeds the 95th percentile of the null distribution.
4. Report per-domain-split pass rate and combined Fisher *p*-value across domains.

**Code reference:** `iterations/iter_0009` through `iter_0012` contain the initial evaluation-design validation. The metric is used in every subsequent iteration.

---

### Phase 4 — Cross-model alignment (Hypothesis family: Cross-model)

**Purpose:** test whether independently trained models learn the same geometric organisation of gene space. **This is the most robust finding in the campaign (Layer 1 of the hierarchy).**

**Steps:**

1. **Canonical Correlation Analysis (H24)** (Hardoon et al. 2004). Given per-gene embeddings from scGPT and Geneformer at matched genes (~320 per domain):
   - Project both models' embeddings into a common maximally-correlated space via CCA.
   - Compute mean canonical correlation across the top-k canonical components.
   - **Expected: *r* = 0.80**, pairwise-distance Spearman **0.75**, gene-level top-1 retrieval accuracy **72%**, combined Fisher *p* = **3.17 × 10⁻⁵** across all three tissue domains.
2. **Simpler Procrustes alignment (H20)** — orthogonal rotation to align two point clouds. **Expected: 40% top-1 retrieval, significant in all 3 domains, *p* = 1.6 × 10⁻⁵**.
3. **Feature-importance ranking comparison (H17)** — are the same geometric features ranked highest in both models? **Expected: centred cosine is the top-ranked feature in all 3 domains for both models, *p* = 0.037**.
4. **Gene-level correspondence methods (19 distinct approaches — all fail).** Test whether individual genes can be matched between models:
   - **Gromov-Wasserstein transport** (Peyré & Cuturi 2019)
   - **Optimal transport** (seeded and unseeded)
   - **Topology-signature distillation**
   - **Cycle-consistent mapping**
   - **Codebook transport**
   - **Chart/sheaf alignment**
   - **Perturbation-response rank alignment (H108)** — the only partial exception, achieving Spearman 0.73 and passing null-gap in 2/3 tissue domains, but **failing in the immune domain (−0.21 null-gap) and failing to replicate across seeds (H109: 2/9 rows passing)**.
   - **All others: mean top-1 retrieval < 1%.**
5. **Interpretation.** Models agree on the **shape** of gene space (distances, neighbourhoods, clusters) but assign **different internal coordinates** to individual genes within that shared shape. One cannot "translate" between models at the gene level, but geometric properties (distances, neighbourhoods, clusters) are transferable.

**Code reference:**
- H24 CCA: `iterations/iter_0013/run_iter0013_screen.py`
- H20 Procrustes, H17 feature ranking: `iterations/iter_0013–iter_0018` region
- H108/H109 perturbation-response: later iterations (see master log)
- 19-method gene-correspondence sweep: scattered across iterations; see `reports/autoloop_master_log.md` Table 3 for the list.

---

### Phase 5 — Persistent homology (Hypothesis family: Persistent homology)

**Purpose:** test whether the gene-embedding manifold contains non-trivial topological structure — specifically, whether embedding neighbourhoods form "loops" (1-dimensional topological features, H1 classes) beyond what random geometric arrangements would produce.

**Steps:**

1. **Build the per-layer per-domain point cloud.** Input: the PCA-reduced embedding matrix (350 genes, 20 dimensions; increased from Phase 0's ~320 by including additional regulatory-annotated genes).
2. **Compute H1 persistent homology via Ripser** (Bauer 2021). Build a Vietoris-Rips filtration over increasing distance thresholds on the point cloud; identify 1-dimensional topological features ("H1 classes") and track their birth and death filtration values.
3. **Total persistence:** sum of (death − birth) lifetimes across all H1 classes. Higher total persistence = more loop-like structure.
4. **Feature-shuffle null** (the weakest null; 20 replicates per condition):
   - Shuffle the embedding features (columns) across genes.
   - Recompute H1 persistence on the shuffled point cloud.
   - Compute `ΔH1_persistence = observed − null_mean`.
   - Significance: compare observed to the 95th percentile of the null distribution.
5. **Expected results (lung domain, H01/H03):**
   - **11/12 scGPT layers significantly more topological** than shuffled (*p* < 0.01 in 9/12 layers)
   - **Top layer L0: *z* = 3.21, *p* = 0.006**
6. **Replication across tissue domains:**
   - **Immune: 12/12 layers significant, mean H1 persistence delta = +12.1 lifetime units**
   - **External-lung: 12/12 layers significant, mean delta = +12.5**
7. **Bootstrap stability:** test topology across sample sizes *n* ∈ {350, 500, ...}, PCA dimensions *d* ∈ {10, 20}, and kNN neighbour count *k* ∈ {8, 12, 16}. **Expected: positive signal in 12/12 layers with complete parameter-setting stability**.
8. **Zigzag persistence** — tracks topological features that persist when alternating between two **disjoint gene pools**, testing whether loops are a property of the shared geometry rather than any particular gene subset. **Expected: exceeded null expectations in all tested configurations**.
9. **Independent confirmation: H47/H49 bifiltration-like cycle-rank.** Counts the number of independent cycles in the kNN graph at each filtration threshold, uses this count as a per-edge feature. **Expected: ΔAUROC +0.006, Fisher-significant in 6 domain-splits, 13/24 individual rows with *p* < 0.05**.
10. **CRITICAL: Rewiring-null test (degree-preserving).** Shuffle the neighbourhood graph while preserving each gene's degree. **Expected: topological signal vanishes completely (0/24 layer-tests significant across six iterations of progressively refined rewiring controls)**.

**Interpretation:** the topology is **real** in the sense that the model creates it (absent in feature-shuffled embeddings), but **fragile** in the sense that it depends on fine-grained neighbourhood structure rather than a deep geometric invariant. The observed topology is not a deeper intrinsic property of the manifold; it arises from the specific pattern of which genes are each other's nearest neighbours.

**Code reference:**
- H01/H03: `iterations/iter_0003/run_iter0003_screen.py`, `iter_0004/run_iter0004_screen.py`
- H47/H49 bifiltration: later iterations (see master log)
- Rewiring nulls: `iter_0007–iter_0012` region

---

### Phase 6 — Manifold distance hierarchy (Hypothesis family: Manifold distances)

**Purpose:** systematically test whether different distance metrics on the gene-embedding manifold capture different aspects of biological regulation.

**Hierarchy discovered** (Layer 3 of the hierarchy, moderate robustness):

**Euclidean (baseline) < Geodesic (H13) < Diffusion (H16) < Triangle-defect spectrum (H69/H70)**

**Steps:**

1. **Euclidean distance (baseline).** Straight-line distance in the 20-dim PCA-reduced embedding space. ΔAUROC close to zero; this is the reference.
2. **Geodesic distance (H13).** Shortest path along the kNN manifold graph. *k* adaptively chosen from 10–35 to ensure graph connectivity (isolated nodes can make geodesic distance undefined). **Expected: ΔAUROC +0.013, 7/12 layers significant in both gene-pool splits**.
3. **Diffusion distance (H16)** (Coifman & Lafon 2006). The L₂ distance between rows of a diffusion-map embedding derived from a random walk on the kNN graph. Diffusion time *t* swept over {1, 2, 4, 8} and selected by best performance. **Expected: ΔAUROC +0.017, significant in all three domains, Fisher *p*-values 10⁻⁵ to 10⁻¹¹**.
4. **Convexity-deficit and detour-based graph-geometry metrics (H32).** **Expected: ΔAUROC +0.017, Fisher-significant in 4/6 domain-splits**.
5. **Triangle-defect spectrum (H69/H70) — the strongest single geometric metric.** For each gene triplet in a kNN neighbourhood, the **triangle defect** measures how much the pairwise distances deviate from the triangle inequality — quantifying local departure from flat Euclidean geometry. Aggregating these defects at multiple neighbourhood scales *k* ∈ {8, 12, 16} produces a per-edge feature vector that captures multiscale curvature around each gene pair. **Expected: ΔAUROC +0.026 with positive null-gap in all 6/6 domain-split groups under hard-null calibration**.
6. **CRITICAL: triangle-defect spectrum becomes the baseline for all subsequent work in the campaign.** Nearly every hypothesis from H70 onward is evaluated as an **incremental improvement over this triangle-defect baseline** rather than over Euclidean distance.
7. **Coexpression-matched null sensitivity** (this is where Layer 3's "moderate" robustness comes from):
   - **Diffusion distance uplift shrinks from +0.017 to +0.008** under coexpression matching, losing statistical significance in 2/3 domains. A substantial fraction of the diffusion distance advantage reflects the coexpression confound, not independent geometric information.
   - **Triangle-defect spectrum survives matched-random-third-order nulls**, indicating genuinely independent geometric structure.

**Interpretation:** regulatory gene pairs are not simply "close" in embedding space — they are connected by **curved geodesic paths** that reflect the nonlinear way the model organises biological relationships. Local manifold curvature around regulatory pairs systematically differs from curvature around non-regulatory pairs.

**Code reference:**
- H13 geodesic: `iterations/iter_0010/run_iter0010_screen.py`
- H16 diffusion distance: mid-loop iterations (see master log)
- H69/H70 triangle-defect: `iter_0039` through `iter_0041` region

---

### Phase 7 — Community structure (Hypothesis family: Community structure)

**Purpose:** test whether embedding neighbourhoods form clusters that correspond to regulatory modules.

**Steps:**

1. **Build the kNN graph** on the PCA-reduced gene embeddings at *k* = 12 (the default; stability tests run at *k* ∈ {8, 12, 16}).
2. **Louvain community detection** (Blondel et al. 2008) at default resolution. Alternatives tested: Leiden (Traag et al. 2019), spectral clustering.
3. **Baseline hypothesis (H16):** test whether regulated gene pairs (from TRRUST) land in the same Louvain community more often than chance. **Expected: AUROC 0.54, all 12 layers significant in both gene-pool splits**.
4. **Confidence-tier scaling (H19, H37) — a clear negative.** Test whether larger-confidence DoRothEA tiers (A > B > C) produce progressively stronger community-alignment signal. **Expected: FAILS.** The community structure tracks binary regulatory identity ("is this gene regulated by anything?") but **not** regulatory strength.
5. **Biologically-anchored community alignment (H62) — immune-only.** **Expected: robust in immune (*p* = 0.001), not robust in lung/external-lung**. Consistent with the Layer-5 immune-concentration pattern.

**Interpretation:** community structure is a real geometric signal that tracks binary regulatory identity, but does not encode regulatory strength or confidence. The strongest community-based finding comes from combining community membership with signed motifs in Phase 8.

**Code reference:**
- H16 community alignment: mid-loop (see master log)
- H19/H37 confidence tiers: `iterations/iter_0013` region

---

### Phase 8 — Signed motif-community hardening (H116 → H123: the strongest finding)

**Purpose:** combine **geometric community structure** with **signed regulatory motif annotations** from TRRUST to produce the single most robust finding in the entire campaign. **This is Layer 4 of the hierarchy.**

**Steps:**

1. **Louvain community assignment** from Phase 7 at each (model, domain, layer).
2. **Load signed regulatory motifs from TRRUST v2.** Each motif is a (TF, target, sign) triple where sign ∈ {activation, repression}.
3. **H116 — TRRUST sign-motif interaction (first full null-gap coverage).** For each gene pair, compute:
   - Do they share a common TF with a known activation or repression sign?
   - Is the geometric community placement of the pair consistent with the annotated sign?
   - Does this combined feature predict regulatory edges **above the triangle-defect baseline** (H70)?
   **Expected: ΔAUROC +0.078, null-gap positive in 6/6 domain-splits — the first hypothesis in the campaign to achieve full null-gap coverage**.
4. **H123 — signed motif-community hardening (THE STRONGEST FINDING).** Refine H116 with stricter null controls:
   - **TF-identity-preserving sign shuffles** — preserve the TF → target structure, permute only the signs. Rules out "the model knows TF identity but not sign direction."
   - **TF/target degree strata** — stratify the null by TF degree (how many targets each TF has) and target degree. Rules out "high-degree TFs are easier to predict."
   - **Matched on TF/target degree pairs** for apples-to-apples comparison.
   - **Expected: ΔAUROC +0.094, positive in every single test row (22/22), surviving the null-gap criterion in every domain-split group tested (6/6).**
   - **The only hypothesis in the entire 141-screen campaign to achieve COMPLETE null-gap coverage under the most stringent controls.**
5. **Interpretation:** the model doesn't just place regulated genes near their regulators; it organises them so that **activation targets and repression targets occupy geometrically distinguishable positions relative to the TF within the community**. This is consistent with the model having learned something about **the direction and sign** of regulatory relationships, not just their existence.
6. **CRITICAL CAUTIONARY TAIL: biological annotation extensions systematically degrade robustness.** Building on H123 with additional biological features:
   - **H123 alone** — all 6/6 domain-splits passing
   - **H124 (with STRING conditioning)** — fewer than half passing
   - **H127 (with GO co-membership)** — 2/9 passing
   - **H130 (with continuous semantic GO similarity)** — 0/9 passing
   - **H138 (with ontology sheaf features)** — 0/9 passing
   Each additional biological feature **increases raw ΔAUROC (from +0.078 to +0.134)** while **progressively destroying null-gap robustness**. The additional features are partially correlated with the null control structure, making it easier for null models to explain the signal. **Lesson: more biological priors are not always better.**

**Code reference:**
- H116: `iterations/iter_0044/run_iter0044_screen.py` (approximate)
- H123: `iterations/iter_0046/run_iter0046_screen.py`
- H124-H138 degradation sequence: later iterations (see master log)

---

### Phase 9 — Stability-selected descriptors (H91)

**Purpose:** combine multiple geometric features into a multivariate classifier via stability selection (Meinshausen & Bühlmann 2010) — no single geometric measure captures all regulatory information.

**Steps:**

1. **For each gene pair, compute a vector of geometric edge features:**
   - Geodesic distance (Phase 6 step 2)
   - Triangle-defect spectrum at *k* ∈ {8, 12, 16} (Phase 6 step 5)
   - Community co-membership (Phase 7)
   - Bifiltration cycle-rank (Phase 5 step 9)
   - Directed topology features (from H52 — see Phase 10 below)
2. **Stability selection.** Fit randomised LASSO on many bootstrap subsamples; retain only features selected in a high fraction of runs.
3. **Cross-validated logistic classifier.** Train on stability-selected features; evaluate in the dual-split disjoint-gene-pool design.
4. **H91 expected result: ΔAUROC +0.074, positive in 72/72 test rows (three seeds × three domains × two splits × four layers), with positive null-gap in all 6/6 domain-split groups.**
5. **Biologically anchored variant (H93):** incorporates annotation and regulatory-sign weighting. **Expected: ΔAUROC +0.084 with similarly complete null-gap coverage.**
6. **Feature-selection reproducibility check:** Jaccard similarity of selected features across bootstrap seeds target = **0.65**, sign agreement = **1.0**. Non-trivial stability — the same geometric features are selected across independent runs.

**Interpretation:** biological regulatory information is **distributed** across multiple geometric dimensions of the embedding space (distance, topology, community, directionality). No single dimension captures everything. The consistency of feature selection indicates this is a stable property of the representations, not an artefact of random initialisation.

**Code reference:**
- H91: `iterations/iter_0042` through `iter_0044` region
- H93: adjacent iterations

---

### Phase 10 — Remaining hypothesis families (directed topology, intrinsic dimensionality, other geometric, 70+ negatives)

**Purpose:** test the remaining five of the nine families documented in the source paper. **Most of these produce negative or fragile results, but the negatives are themselves important.**

**Steps per family:**

1. **Directed topology (family 5, n=7 hypotheses).** Tests whether the geometry encodes directional information (TF → target). **H52 (directed/signed topology): ΔAUROC +0.015, 6/6 splits — moderate**. Directional features add modest signal beyond undirected geometry.
2. **Intrinsic dimensionality (family 6, n=8).** Tests whether the local complexity of the manifold correlates with regulatory architecture.
   - **H42/H45: in-sample manifold complexity correlates with regulation out-of-sample (ΔR² = −10.7) — likely overfitting. FAILS.**
   - **H30: hyperbolicity test. FAILS — the embedding manifold is NOT hyperbolic; tree-like hierarchical structure is not the right geometric metaphor.**
3. **Sparse descriptors (family 7, n=8)** — already covered by H91 in Phase 9.
4. **Motif-community (family 8)** — already covered by H116/H123 in Phase 8.
5. **Other geometric (family 9, n=26)** — the largest family, testing curvature, hyperbolicity, anisotropy, etc.:
   - **H23: Forman curvature (Sreejith et al. 2016) enrichment.** Hypothesis: high-curvature edges are regulatory. **FAILS in the opposite direction**: high-curvature edges are **less** likely to be regulatory (AUROC 0.34-0.39, **below chance**).
   - **H95/H97: bridge-curvature lineage.** Raw effect ΔAUROC +0.079 in 24/24 rows. **FAILS: 0/6 domain-splits survive null-gap auditing.**
   - **H111: biologically anchored finite-state grammar.** Raw effect ΔAUROC +0.112 in 6/6 domain-splits. **FAILS: only 1/6 surviving null-gap — despite being one of the largest observed effect sizes in the campaign.**
   - **H139: sectional anisotropy.** ΔAUROC +0.031 in 6/9 splits — **partial**.

**Cross-cutting methodological variants (30 additional hypotheses):** topology stability tests, null-framework development, split-design validation. These span multiple families and are documented in `reports/autoloop_master_log.md` rather than grouped into a single family.

**The 70+ negatives are the most important output of this phase.** Their systematic documentation prevents the selective-reporting failure mode in interpretability research.

**Code reference:** `iterations/iter_0040` through `iter_0052` for the later hypothesis families; see `reports/autoloop_master_log.md` for complete mapping.

---

### Phase 11 — Autonomous executor-brainstormer loop

**Purpose:** drive the 141-hypothesis generation process via an LLM-powered autonomous loop that alternates between **executor** (designs and runs experiments) and **brainstormer** (reviews results, retires stale directions, generates new hypotheses) agents.

**Architecture** (repo `loop/run_codex_topology_autoloop.py`, 853 lines):

1. **LLM backend: OpenAI Codex 5.3 at `reasoning_effort="xhigh"`**. This is a critical parameter — "extra high" reasoning enables deeper hypothesis generation and self-review per iteration.
2. **Two-agent alternation.**
   - **Executor agent** — receives a hypothesis brief from the previous brainstormer iteration. Designs and runs a self-contained Python experiment on pre-extracted embeddings, writes structured JSON outcomes, produces a narrative report. Outputs:
     - `executor_hypothesis_screen.json` — structured schema: `{id, name, family, split_regime, method, status, result_direction, decision}`
     - `executor_iteration_report.md` — narrative summary
     - `run_iter{XXXX}_screen.py` — the auto-generated Python script
     - `*.csv` — quantitative results
     - `executor_prompt.md`, `executor_stdout.log`, `executor_*.log` — execution artefacts
   - **Brainstormer agent** — reviews cumulative results. Outputs:
     - `brainstormer_structured_feedback.md` — synthesis of prior iterations
     - `brainstormer_hypothesis_roadmap.md` — explicit sections: **Retire/Deprioritize**, **New Hypothesis Portfolio**, **Top 3 for Immediate Execution**
     - `brainstormer_next_iteration_brief.md` — the brief the next executor iteration consumes
3. **Executor prompt template** (`prompts/executor_prompt_topology_hypothesis_screening.md`, 5.2 KB): instructs the executor to screen broad hypothesis portfolio (2-3 per iteration, at least 1 novel), rotate among the 9 hypothesis families, follow the dual-split design, and apply at least one null family. Evidence standards: reproducible, survives ≥ 1 null family, consistent direction, biological anchor.
4. **Brainstormer prompt template** (`prompts/brainstormer_prompt_template.md`, 2.3 KB): generates ambitious, testable hypotheses to maximise discovery odds. Reviews iteration artefacts, retires stale directions, generates 12–16 new hypothesis ideas, and selects **Top 3** for the next iteration — 1 high-probability, 1 high-risk/high-reward, 1 cheap broad-screen.
5. **Hypothesis retirement policy** (critical for loop convergence): hypotheses marked retired after **2 consecutive negative or inconclusive results** with adequate null controls. Rescue only with **explicit major method change**. Without this policy, the loop diverges and tests the same ideas repeatedly.
6. **Promotion rule** (critical for positive findings): every positive finding must pass at least one permutation-based null before being promoted to the running paper draft. Without this rule, weak feature-shuffle passes inflate the positive count.
7. **Runtime infrastructure:**
   - `loop/start_codex_autoloop.sh` — daemonised launch (double-fork)
   - `loop/stop_codex_autoloop.sh` — graceful shutdown via `STOP` sentinel
   - `loop/status_codex_autoloop.sh` — heartbeat monitoring
   - `loop/config.json` — model, reasoning effort, prompt paths, project root
   - `runtime/loop_events.jsonl` — per-event log (one JSON per line)
   - `runtime/loop_status.json` — final loop state

**The source paper ran 53 iterations total** (1 init failure, 52 productive), testing 141 hypotheses. Typical iteration produces 2–3 hypothesis outcomes plus methodological variants.

**Code references:**
- Main runner: `repos/topology-biomechinterp2/loop/run_codex_topology_autoloop.py` (853 lines)
- Executor prompt: `repos/topology-biomechinterp2/prompts/executor_prompt_topology_hypothesis_screening.md`
- Brainstormer prompt: `repos/topology-biomechinterp2/prompts/brainstormer_prompt_template.md`
- Config: `repos/topology-biomechinterp2/loop/config.json`

---

### Phase 12 — Strict max-null audit (H141)

**Purpose:** test every previously-promoted positive finding against the most conservative possible null — the **maximum** of the 95th percentile across ALL four null families (feature-shuffle, label-permutation, degree-preserving rewiring, coexpression-matched) simultaneously.

**Steps:**

1. **For each hypothesis that achieved positive null-gap in any single family** (the subset that "passed" in the running paper draft):
   - Compute the observed ΔAUROC.
   - Compute the 95th percentile of EACH null family's ΔAUROC distribution.
   - Compute the **strict max-null margin** = observed ΔAUROC − max(95th percentile across all four null families).
2. **Aggregate to the domain-split level.** A domain-split group "passes strict max-null" if its strict margin is positive.
3. **Expected results (source paper §3.6 and Figure 7):**
   - **Overall mean strict margin: −0.005** (on average, the strongest null family CAN explain the observed signal).
   - **Strict-positive support: only 3/9 domain-split groups (15/25 individual test rows).**
   - **By domain:**
     - **Immune: positive strict margin (+0.012)** — the signal is real and survives all controls
     - **Lung: slightly negative (−0.008)** — marginal
     - **External-lung: negative (−0.023)** — fragile under strict auditing
4. **Robust positive rate under max-null: roughly 10%** (fewer than 15 of 141 hypotheses survive).

**Interpretation:** the 141-hypothesis screen demonstrated that the same geometric feature can appear highly significant under feature-shuffle controls but non-significant under rewiring or strict max-null controls. **Any interpretability claim about biological model geometry should specify which null model it was tested against and acknowledge what would happen under stricter controls.**

**Concentration in immune tissue is consistent with two non-exclusive explanations**:
- The immune system has an unusually modular regulatory architecture (distinct T-cell, B-cell, myeloid programs) that creates stronger geometric signatures than the less discretely organised regulatory programs in lung tissue.
- Immune-system regulators are better annotated in the databases used as ground truth, so the "failure" in lung and external-lung may partly reflect annotation incompleteness rather than absence of geometric structure.

**Code reference:**
- H141: `iterations/iter_0052/run_iter0052_screen.py`, `iter_0053/run_iter0053_screen.py`
- Figure 7: `figures/codex_h141_strict_margin_audit.png`

---

### Phase 13 — Layered findings synthesis and final reporting

**Purpose:** compile the 141 hypothesis outcomes into the final 5-layer hierarchy of robustness and produce the paper-ready synthesis.

**Steps:**

1. **Classify every hypothesis outcome** into one of three categories:
   - **Robust positive** (~27): passes feature-shuffle AND at least one additional null. ~19% of total.
   - **Inconclusive/partial** (~21): directional signal but fails at least one null or one domain-split.
   - **Decisively negative** (~63): fails under all tested nulls or shows the wrong direction.
   - Plus ~30 methodological variants: topology stability tests, null-framework development, split-design validation.
2. **Assign each robust-positive hypothesis to a layer** of the 5-layer hierarchy (§1):
   - **Layer 1 (most robust):** H17, H20, H24 (cross-model CCA, Procrustes, feature ranking)
   - **Layer 2 (robust under feature-shuffle):** H01/H03 (persistent homology), H47/H49 (bifiltration cycle-rank), H16 (community alignment)
   - **Layer 3 (moderate, partially coexpression-confounded):** H13 (geodesic), H16 (diffusion), H32 (convexity-deficit), H69/H70 (triangle-defect — the baseline)
   - **Layer 4 (strongest single signal):** H116 (TRRUST sign-motif), H123 (signed motif-community hardening — the strongest)
   - **Layer 5 (localised to immune under strict audit):** H141 strict-positive subset, ~10% of total
3. **Compile the negative-findings table** (Table 3 in source paper) documenting which hypotheses failed and why (key examples: topology robust to graph rewiring H07-H12, Forman curvature H23, cross-model correspondence 19 methods, confidence-tier scaling H19/H37, intrinsic dimension transfer H42/H45, hyperbolicity H30, biological annotation extensions H94/H127-H130/H135/H138).
4. **Write the final paper draft** automatically via the brainstormer agent — the paper in `repos/topology-biomechinterp2/` has a cumulative draft maintained throughout the loop.
5. **Emit final figures** (via `generate_figures.py`):
   - Figure 1: outcome distribution across 9 families
   - Figure 2: cross-model CCA alignment (H24)
   - Figure 3: persistent homology per layer per domain (H01/H03)
   - Figure 4: geodesic > Euclidean advantage (H13)
   - Figure 5: H123 strongest finding (effect size + null-gap per domain-split)
   - Figure 6: H123 vs H139 null-gap comparison (the "hardening boundary")
   - Figure 7: strict max-null margin audit (H141)
   - Figure 8: neighbourhood-scaling gain diagnostic (H140)

**Code reference:**
- Figure generation: `repos/topology-biomechinterp2/generate_figures.py`
- Master log: `repos/topology-biomechinterp2/reports/autoloop_master_log.md` (~1,500 lines, per-iteration evidence trails)

## 7. Parameters

| Parameter | Default | Alternate / sensitivity | Used in |
|---|---|---|---|
| `n_genes_per_domain` | **~320** (intersection of model vocab and regulatory DB) | 200–500 | Phase 0 |
| `pca_components` | **20** | {10, 20} sensitivity | Phase 0 |
| `tissue_domains` | **{lung, immune, external-lung}** | any Tabula Sapiens domain | Phase 0 |
| `n_seeds_per_split` | **3** (at minimum) | 3–10 | Phase 1 |
| `null_feature_shuffle_replicates` | **20–24** | 10–100 | Phase 2 |
| `null_label_permutation_replicates` | **100–200** | 100–1,000 | Phase 2 |
| `null_rewiring_replicates` | **24** (across 6 refinement iterations) | 24+ | Phase 2 |
| `null_coexpression_bins` | **5** (quintiles of pairwise Pearson abs correlation) | — | Phase 2 |
| `null_rewiring_k` | **12** (adaptive 10–35 for connectivity) | — | Phase 2 |
| `delta_auroc_threshold` | positive null-gap in **6/6 domain-splits** for "robust" | — | Phase 3 |
| `cca_n_components` | top-k canonical components | all (min of d_model dimensions) | Phase 4 |
| `cca_target_retrieval_accuracy` | **72% top-1** (paper's observed) | — | Phase 4 |
| `persistent_homology_n_genes` | **350** | 200–500 | Phase 5 |
| `persistent_homology_k_range` | **k ∈ {8, 12, 16}** | 5–25 | Phase 5 |
| `ripser_max_dim` | **1** (H1 only; higher dims computationally prohibitive) | 0–2 | Phase 5 |
| `diffusion_time_t` | swept over **{1, 2, 4, 8}**, best selected | 1–16 | Phase 6 |
| `geodesic_knn_k` | **10–35 adaptive** (connectivity-based) | 5–50 | Phase 6 |
| `triangle_defect_scales` | **k ∈ {8, 12, 16}** | 5–25 | Phase 6 |
| `louvain_resolution` | **default (1.0)** | 0.5–2.0 | Phase 7 |
| `louvain_k` | **12** (with stability at {8, 12, 16}) | 5–25 | Phase 7 |
| `stability_selection_n_bootstrap` | **100** randomised LASSO fits | 50–500 | Phase 9 |
| `stability_selection_jaccard_target` | **0.65** (reproducibility threshold) | — | Phase 9 |
| `stability_selection_sign_agreement_target` | **1.0** (unanimous sign) | — | Phase 9 |
| `autoloop_model` | **gpt-5.3-codex** | any LLM with reasoning-effort param | Phase 11 |
| `autoloop_reasoning_effort` | **`xhigh` (extra high)** | high / xhigh | Phase 11 |
| `autoloop_hypotheses_per_iteration` | **2–3** (with ≥1 novel) | 1–5 | Phase 11 |
| `autoloop_retirement_policy` | **2 consecutive negatives** | — | Phase 11 |
| `autoloop_brainstormer_portfolio_size` | **12–16 ideas, top 3 selected** | — | Phase 11 |
| `autoloop_iteration_count_target` | **40–80** (source paper ran 53) | 30–100 | Phase 11 |
| `strict_max_null_threshold` | **95th percentile across all null families** | — | Phase 12 |

## 8. Validation

A successful pipeline run must reproduce the following sanity signatures before any new-model claim is trusted:

1. **Cross-model CCA alignment** (Phase 4, H24): mean canonical correlation *r* = **0.80**, pairwise-distance Spearman **0.75**, gene-level top-1 retrieval **72%**, combined Fisher *p* = **3.17 × 10⁻⁵** across 3 domains. Tolerance ±0.05 on *r*. If *r* < 0.7, the two models are substantially less aligned than Geneformer and scGPT are in the paper — either one of them is poorly trained or the domain has fewer shared genes than expected.
2. **Persistent homology significance** (Phase 5, H01/H03): 11/12 scGPT lung layers significant at *p* < 0.01 under feature-shuffle; 12/12 immune, 12/12 external-lung. Top lung layer L0 *z* = **3.21**, *p* = 0.006.
3. **Rewiring-null topology vanishing** (Phase 5 step 10): **0/24 layer-tests significant** under degree-preserving kNN rewiring. **If any layer-test is significant under rewiring, the rewiring null is not actually preserving degree distributions** — check the curveball or equivalent algorithm.
4. **Manifold distance hierarchy** (Phase 6): Euclidean < Geodesic (+0.013) < Diffusion (+0.017) < Triangle-defect (+0.026). The hierarchy must be monotonic. If diffusion > triangle-defect, the triangle-defect implementation is broken or the multi-scale aggregation k ∈ {8, 12, 16} is not being applied.
5. **H91 stability-selected descriptors** (Phase 9): ΔAUROC +0.074, **positive in 72/72 test rows**, 6/6 domain-split null-gap. Jaccard similarity of selected features across bootstrap seeds = **0.65**, sign agreement = **1.0**. If Jaccard < 0.5, the stability selection procedure is not actually stable — increase the number of bootstrap iterations.
6. **H123 strongest-finding signature** (Phase 8): **ΔAUROC +0.094, 22/22 test rows positive, 6/6 domain-splits surviving null-gap**. **The only hypothesis in the campaign with complete null-gap coverage under the strictest controls.** This is the mandatory diagnostic for a correctly-running pipeline — if H123 yields < 18/22 rows passing, the TF-identity-preserving sign shuffle or TF/target degree stratification is implemented incorrectly.
7. **Annotation-extension degradation pattern** (Phase 8 step 6): H123 (6/6) → H124 (< 3/6) → H127 (2/9) → H130 (0/9) → H138 (0/9). **The monotonic degradation is a required signature.** If H130 or H138 still has complete null-gap coverage, either the additional GO/STRING/ontology features are not being added correctly, or the null structure is being inadvertently relaxed.
8. **Strict max-null audit** (Phase 12, H141): overall mean strict margin **−0.005**. **Only 3/9 domain-split groups strict-positive**. By domain: immune **+0.012**, lung **−0.008**, external-lung **−0.023**. The domain ordering **immune > lung > external-lung is required** — if it reverses, the strict margin computation is comparing to the wrong null-family maximum.
9. **Hypothesis outcome distribution** (Phase 13): approximately **27 robust positive, 21 inconclusive/partial, 63 decisively negative**, plus ~30 methodological variants. Tolerance ±5 on each category. If the robust-positive count is > 40, the null-family enforcement is too weak; if < 15, it may be too strict.
10. **Cross-model gene-correspondence failure** (Phase 4 step 4): **all 19 methods below 1% top-1 retrieval**. **This is a strong negative diagnostic.** If any method achieves > 10%, check whether the gene sets are actually identical between the two models (most common bug: gene name capitalisation mismatch between scGPT and Geneformer tokenisers).
11. **Forman curvature directional signature** (Phase 10, H23): AUROC **0.34–0.39** (below chance). The direction is critical — high-curvature edges should be **less** likely to be regulatory, not more. If H23 produces AUROC > 0.5, the curvature sign is flipped in implementation.
12. **Bifiltration cycle-rank confirmation** (Phase 5 step 9): ΔAUROC **+0.006**, Fisher-significant in 6 domain-splits, 13/24 individual rows with *p* < 0.05.
13. **Autoloop iteration count** (Phase 11): source paper ran **53 iterations** (1 init failure, 52 productive). For a new model, expect 40–80 iterations before natural convergence (no new hypothesis branches to explore).

## 9. Known pitfalls

1. **Feature-shuffle null is too weak to be a publication standard.** The central lesson of the paper: 70+ hypotheses passed feature-shuffle but failed under rewiring or max-null. **Always run at least two of the four null families — ideally rewiring and coexpression-matched — before promoting any finding**. Report the strict max-null margin even if it is negative; negative strict margins are informative, not embarrassing.
2. **Rewiring null must preserve degree, not just edge count.** A common bug is to shuffle edges at random (preserving only the total edge count), which is equivalent to the feature-shuffle null in topological sensitivity. **Use the curveball algorithm or equivalent** (Strona et al. 2014) to preserve each node's degree.
3. **Coexpression-matched null is the highest-leverage null for biological foundation models.** Coexpressed genes are geometrically proximal in essentially every trained foundation model; if you skip this null, you will inflate your positive count by the coexpression confound. **Always include it.**
4. **PCA reduction to 20 components may discard information** that matters for intrinsic dimensionality tests (H42, H45). Ran the same hypotheses at PCA=10 and PCA=50 for sensitivity; report the full range.
5. **kNN graph connectivity is a silent failure mode.** If *k* is too small, the graph has disconnected components and geodesic distance is undefined for some pairs. The paper uses **adaptive k ∈ [10, 35]** to ensure full connectivity. Always verify the graph is connected before running geodesic/diffusion distance.
6. **Persistent homology is computationally prohibitive above H1.** Ripser can compute H2 and higher, but the runtime scales exponentially. For 350 genes in 20 dims, **H1-only** takes seconds; H2 takes minutes-to-hours; H3+ is infeasible. The paper only reports H1.
7. **CCA requires matched gene sets between models.** scGPT and Geneformer have different vocabularies; the intersection is ~320 genes. If the intersection is too small (< 100 genes), CCA becomes unstable. Use a larger tissue corpus if needed to inflate the shared gene set.
8. **Gene-level correspondence impossibility is robust, not a bug.** All 19 methods failing to recover gene-level matches is **the finding**, not a failure of the code. Do not try to "fix" gene-level correspondence by tuning methods — the paper's claim is that gene-level correspondence is genuinely impossible between these two models at the individual-gene level, while the shape of gene space is preserved.
9. **Louvain communities are stochastic.** Run with fixed random seed for reproducibility. The source paper uses Louvain directly (not Leiden) — Leiden can also be used but changes community IDs and requires seed coordination.
10. **Autoloop retirement policy is non-negotiable.** Running the brainstormer without retirement causes the loop to diverge and test the same ideas repeatedly. **2 consecutive negatives with adequate controls → retire unless explicitly rescued with a materially changed method.**
11. **The H123-to-H138 degradation is a cautionary tale specific to this paper.** Adding biological annotations (GO, STRING, ontology sheaves) to a strong hypothesis (H123) systematically eroded null-gap robustness, even though raw ΔAUROC increased. The failure mode: additional features are partially correlated with the null control structure, inflating both observed signal AND null signal but eroding the gap. **Lesson: more biological priors are not always better.**
12. **Codex 5.3 at xhigh reasoning is required for hypothesis novelty.** Lower reasoning efforts produce less diverse hypotheses and the loop converges to a narrower family coverage. If using Claude or another LLM, use the highest available reasoning mode.
13. **scGPT tokenisation (continuous values) vs Geneformer (rank values)** produces different vocabulary structures. Verify that per-gene embeddings from both models are defined for the same gene set before running CCA.
14. **The external-lung domain is genuinely held out** — no tuning on this dataset. Do not use external-lung results to fit any hyperparameters (threshold choice, k-value selection, diffusion time). That would invalidate the strict max-null audit.
15. **Biological anchoring inflates raw effects, degrades null-gap robustness.** Examples of the pattern: H24 cross-model CCA is robust without any biological annotation; H116 + TRRUST signed motifs is the first full null-gap pass; H123 with TF-identity-preserving shuffles + degree strata is the strongest; H124 + STRING conditioning destroys null-gap. **Biological priors are helpful for hypothesis motivation but dangerous for null control construction.**
16. **The 5-layer hierarchy is the right unit of reporting**, not individual hypotheses. A finding at Layer 1 (cross-model) is orders of magnitude more robust than a finding at Layer 4 (signed motif-community). Never report a single hypothesis's ΔAUROC without specifying which layer of the hierarchy it belongs to.

## 10. Quick-start for new-model evaluation

To apply this pipeline to a new single-cell foundation model `NEWMODEL`:

1. **Add `NEWMODEL` to the embedding extraction step (Phase 0).** Export per-layer gene embeddings for at least one matching cell population, reduced to 20 PCA components.
2. **Keep scGPT AND/OR Geneformer as reference models.** The Phase 4 cross-model CCA requires at least two independently-trained models to test consistency. A new model in isolation cannot produce the Layer-1 finding.
3. **Run Phase 0–3 smoke test.** Compute the ΔAUROC pipeline on 10 simple Euclidean-distance hypotheses. Verify the dual-split design, null model infrastructure, and gene-pool construction all work correctly before spinning up the autonomous loop.
4. **Run Phase 4 first (cross-model CCA).** This is the fastest and most informative single test. If `NEWMODEL` does not achieve CCA *r* > 0.6 with an established foundation model, it may have learned qualitatively different geometry; the subsequent hypothesis families may not transfer.
5. **Run Phase 5 (persistent homology) with feature-shuffle null**, then **Phase 5 step 10 (rewiring null)**. Report both. If topology passes feature-shuffle but fails rewiring (as in the paper), note that this is the expected signature, not a failure.
6. **Run Phase 6 (manifold distance hierarchy)** with the triangle-defect spectrum at the end. Establish the baseline for all subsequent hypotheses.
7. **Run Phase 7 + Phase 8 (community structure → signed motif-community hardening H123).** This is the strongest single-hypothesis test. If H123 passes with complete null-gap coverage on `NEWMODEL`, you have reproduced the most robust finding.
8. **Run Phase 9 (stability-selected descriptors H91).** This combines all previous geometric features into a classifier.
9. **Run Phase 10 for the remaining families.** Most will be negative; document the negatives.
10. **Optional: Phase 11 autonomous loop** for novel hypothesis generation. Fork the `topology-biomechinterp2` repo, update the prompt templates with any model-specific context, and run 40–80 iterations. Seed the brainstormer with the 9 hypothesis families from the source paper and with the `reports/autoloop_master_log.md` summary.
11. **Run Phase 12 strict max-null audit.** Emit the per-domain strict margins. If the immune domain shows positive strict margin and other domains do not, you have reproduced the Layer-5 localisation pattern.
12. **Run Phase 13 synthesis.** Place every robust-positive hypothesis in one of the 5 layers; compute the final outcome-distribution table; emit the layered findings report.
13. **Cross-reference all pipelines in this project:**
    - **Attention** (`attention-grn-extraction-and-evaluation.md`): does attention encode regulatory logic or co-expression? (Geneformer/scGPT verdict: co-expression.)
    - **Spectral geometry** (`residual-stream-spectral-geometry.md`): does SVD of the residual stream reveal biological axes? (Verdict: yes — SV1 = subcellular localisation, SV2–SV4 = PPI, SV5–SV7 = regulation.)
    - **SAE mega-pipeline** (`sparse-autoencoders/README.md`): do interpretable sparse features form biologically coherent causal circuits? (Verdict: yes for circuit structure, no for gene-level regulatory prediction.)
    - **This pipeline** (`topology-geometry-141-hypotheses.md`): does the topological/geometric structure of the embedding manifold carry regulatory information under strict null controls? (Verdict: yes for cross-model consistency and signed motif-community alignment; no for gene-level correspondence; yes only in immune tissue under strict audit.)
14. **A complete mechanistic audit of `NEWMODEL`** produces four independent verdicts across these four pipelines. A robust biological foundation model would ideally pass all four. **No model tested to date (scGPT, Geneformer V2-316M) passes all four** — the recurring boundary is that foundation models encode co-expression and pathway structure but not directed causal regulatory logic.
15. **Report the strict max-null margins per domain alongside the 141-hypothesis outcome table.** That table is the single most valuable artefact of this pipeline. Future interpretability claims about `NEWMODEL` should be benchmarked against the boundary this table defines.
