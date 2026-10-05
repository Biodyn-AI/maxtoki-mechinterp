# Pipeline: Biological Manifold Discovery, Extraction, and Compactification from Single-Cell Foundation Model Internals

## 1. Overview

This pipeline is a **generalizable, model-agnostic workflow** for discovering biological manifolds inside single-cell foundation models, extracting them as standalone performant algorithms, evaluating those algorithms against established baselines, and then aggressively compactifying the resulting operators down to a level where their mechanistic substrate can be directly interpreted at the gene-program level. It is built around the same autonomous executor–reviewer loop architecture used in the spectral-geometry (`residual-stream-spectral-geometry.md`, arxiv 2602.22247) and 141-hypotheses (`topology-geometry-141-hypotheses.md`, arxiv 2602.22289) pipelines, but the loop's output is not a list of rejected hypotheses — it is a **deployable algorithm** consisting of a frozen attention-operator export, a small learned adaptor, and a task-specific readout.

**The critical framing: this pipeline generalizes the methodology, not the finding.** The source paper discovered a **hematopoietic developmental manifold** inside scGPT as its primary case study and replicated the protocol on a second manifold family (**intercellular communication**, H38), demonstrating that the same Phase-1 loop, quality gates, extraction stages, and compactification chain work when only the *biological ruler* is swapped (developmental stage distances → OmniPath interaction-path distances). **When applying this pipeline to a new biological foundation model and/or a new biological system, you do not search for the hematopoietic manifold** — you define a biological ruler for your system of interest (developmental ordering, spatial organisation, cell–cell signalling, metabolic state progression, cell-cycle phase, epigenetic landscape, etc.) and let the autonomous loop screen candidate operator × adaptor × fitting-method combinations against your ruler until one branch passes all quality gates with a matched null branch failing.

**The three-stage extraction pipeline** (§6 Phases 5–7 below) is the core innovation:

```
  x  ──Stage 1: frozen operator──▶  f(x)  ──Stage 2: learned adaptor g_θ──▶  z  ──Stage 3: task readout h_φ──▶  ŷ
```

- **Stage 1 (direct operator export from frozen attention weights):** no parameters are learned. A fixed feature map is constructed directly from native attention-value-projection matrices `A_{ℓ,h} ∈ ℝ^{d×d}` read out of the pretrained foundation model. For scGPT whole-human these are the 12×8=**96 heads** of a `d=1,200` matrix each; for Geneformer V2-316M these would be 18×18=324 heads of `d=1,152`. The main benchmark operator in the source paper is a **pooled drift map** `f_drift(x) = [(xA_early − xA_mid); (xA_mid − xA_late)]`, concatenating the pairwise differences in how each gene's representation changes across early/middle/late layer blocks — capturing developmental trajectory information. This is a key conceptual advance: transformer layers progressively refine gene representations, and the **drift between adjacent layer groups encodes directional biological information** that is invisible in static per-layer embeddings.
- **Stage 2 (lightweight learned adaptor):** a small head `g_θ` (typically a linear `W_enc (x − b)` projection to `d=10` or smaller) trained on *internal* data only via the **Latent Embedding Transfer (LET)** objective, which combines a distance-fitting term (match latent pairwise arc-cosine distances to biological-ruler distances) with a reconstruction regulariser. This is the only learned component of the shared representation.
- **Stage 3 (task-specific readout):** small probes `h_φ` trained per downstream task. Deliberately kept tiny (5–170 trainable parameters per task in the source paper) and not part of the extracted representation itself.

**The source paper's headline results** (on the hematopoietic case study):

- **Manifold discovery:** internal trustworthiness **0.979**, geodesic-biological correlation **0.835**, blocked-permutation *p* = **0.0005**. Matched null branch (H66, same pipeline with a different biological target) failed, confirming branch-specificity.
- **External validation:** strict non-overlap Tabula Sapiens panel (616 anchors, 564,253 cells, 27 tissues, 21 donors, 34 fine stages, 7 branches) passed all quality gates (trustworthiness 0.985, random/donor/branch holdout correlations 0.594/0.596/0.453). A lung panel was run as an expected-to-fail control and **did fail** robust transfer (trustworthiness 0.617, branch correlation 0.058) — demonstrating that high global correlation alone is insufficient and that clade/branch holdout controls + trustworthiness are required. A separate multi-donor immune panel (7 cohorts, 165 anchors, 10,819 cells) confirmed frozen-head zero-shot transfer without any retraining.
- **Benchmark performance:** on 88 grouped donor-holdout splits (Robust-V2), the extracted cell head **beats 8 established baselines** (scVI, Palantir, DPT, CellTypist, PCA-10, SVD-10, scGPT avg-pool + 3-layer MLP, raw 1200-gene log1p) on pseudotime-depth Spearman correlation (BH-*q* ≤ **2.7 × 10⁻⁷** on all paired comparisons) and wins on 6 of 8 classification endpoints. Key subtype AUROCs: **CD4/CD8 0.867**, **monocyte/macrophage 0.951**.
- **Cost:** full 12-split evaluation campaign **34.5× faster** (~205 s vs ~7,074 s) than the frozen-scGPT-embedding + 3-layer MLP alternative, with **~1,000× fewer trainable parameters** (5–170 vs 172,610–174,690 per task).
- **Multi-stage compaction:** the exported operator compresses from 3 pooled attention heads (17.5 MB) to a **single head L2H5** (5.9 MB) without statistically significant loss, and further to a **rank-64 truncated-SVD surrogate (0.73 MB)**. A hard-sparse pruning to 16 factors × 60 read/write genes per factor (0.12 MB) degrades significantly but remains interpretable at baseline-level accuracy.
- **Mechanistic interpretability:** leave-one-factor-out ablation on the rank-64 compact operator reveals a **four-factor core** (f00, f01, f02, f03) that explains **66.2% of total pooled ablation impact**. Factors resolve into explicit hematopoietic gene programs: **f01 branch routing** (mono/macro + naive CD8⁺ T + granulopoiesis, drop 0.430), **f00 stage ordering** (classical monocyte + naive CD8⁺ T maturation, drop 0.280), **f02 lymphoid contrast** (lymphoid B vs T/NK, drop 0.254), **f03 mono/macro structure** (macrophage/phagocyte vs granulocytic, drop 0.241). The same four-factor core persists in the hard-sparse surrogate, explaining 68.9% of sparse-specific ablation impact.

**Second-manifold generalisation (H38 intercellular communication):** the same Phase-1 loop was run with **OmniPath ligand–receptor interaction paths** as the biological ruler instead of developmental stages. Internal panel: 104 anchors built from 30 curated ligand–receptor interaction-path pairs. Final rescued variant after an 11-iteration external-rescue process (v4→v11, expanding internal coverage to 100,000 cells, 392 internal anchors, 23 donors, 58 cell types, 3 tissues): external correlation **0.983**, internal trustworthiness **0.908**, manifold-minus-PCA2 baseline margin **+0.006**, blocked-permutation *p* = **0.0005**. Axis 1 carries broad-class structure with **η² = 0.735**, with cytokine/chemokine and lipid/metabolic programs on opposite loadings (ρ = −0.549 and +0.566 respectively). Classification benchmark is mixed (2D H38 manifold beats 2D PCA on grouped macro-F1 0.482 vs 0.464 but loses to 6D PCA 0.531) — H38 is framed as a "successful generalisation case" with lower overall maturity than H65.

**The generalisation lesson:** the same pipeline architecture (Phase-1 sweep × gate stack × extraction stages × compactification chain) works across distinct biological-manifold families. What changes between applications is (a) the biological ruler (anchor-distance matrix), (b) the anchor construction (cell-type × tissue × donor × stage schema), and (c) the external validation cohort. Everything else — the LET objective, the operator export, the head/layer attribution sweep, the multi-stage compaction, the factor-necessity ablation — is invariant.

## 2. Source

- **Paper:** Kendiukhov, I. (2026). *Discovery of a Hematopoietic Manifold in scGPT Yields a Method for Extracting Performant Algorithms from Biological Foundation Model Internals.* arXiv:2603.10261 [cs.LG], 10 Mar 2026. **45 pages** (15 pages main text + 5 pages references + 25 pages supplementary notes S1–S37).
- **Local PDF:** `references/2603.10261_Kendiukhov_manifold_extraction.pdf`
- **GitHub repo:** ⚠️ **no public repository is referenced in the paper**. Unlike all five other Kendiukhov 2025–2026 papers in this project — which consistently link to `github.com/Biodyn-AI/{name}` — this paper does not cite a GitHub URL, Zenodo DOI, or any other code/data repository anywhere in the 45-page document. Supplementary Note S1 references `iterations/` directories "in the project repo" but the project repo itself is not disclosed. Multiple likely URL patterns (`Biodyn-AI/manifold-extraction`, `bio-manifold`, `hematopoietic-manifold`, `scgpt-manifold`, `topology-biomechinterp3`, `bio-manifold-extract`, `algorithm-extraction`) return 404. **This pipeline therefore reconstructs the methodology from the paper text alone and does not have a reference implementation in `repos/`.**
- **Companion pipelines in this project (all use overlapping methodology and should be cross-referenced):**
  - `attention-grn-extraction-and-evaluation.md` (arxiv 2602.17532) — attention-based interpretability. Same scGPT attention tensor, different readout.
  - `residual-stream-spectral-geometry.md` (arxiv 2602.22247) — SVD spectral-axis analysis. Predecessor autonomous loop architecture (Claude-based rather than Codex/author-led).
  - `topology-geometry-141-hypotheses.md` (arxiv 2602.22289) — 141-hypothesis topological screen. Same executor-reviewer discipline and hierarchical null-model framework.
  - `sparse-autoencoders/README.md` (arxiv 2603.02952/01752/11940) — SAE-based mechanistic interpretability. The four-factor-core mechanistic audit here uses a conceptually similar leave-one-factor-out ablation as the SAE circuit-tracing pipeline.
- **Key foundational references:**
  - **Foundation model internals and mechanistic interpretability:** Elhage et al. 2021 (transformer circuits), Elhage et al. 2022 (superposition), Olah et al. 2020 (circuits program), Bills et al. 2023 (neuron explanation), Bricken et al. 2023 / Templeton et al. 2024 (SAE dictionary learning), Conmy et al. 2023 (automated circuit discovery), Marks et al. 2024 (sparse feature circuits), Geiger et al. 2021 (causal abstractions), Geva et al. 2021 (MLP as key-value memories), Meng et al. 2022 (ROME), Park et al. 2023 (linear representation hypothesis), Clark et al. 2019 (BERT attention analysis), Vig et al. 2021 (attention in protein language models).
  - **Single-cell foundation models:** Cui et al. 2024 (scGPT), Theodoris et al. 2023 (Geneformer), Yang et al. 2022 (scBERT), Hao et al. 2024 (scFoundation), Rosen et al. 2023 (UCE), Lopez et al. 2018 (scVI).
  - **Manifold learning and trajectory inference:** Tenenbaum et al. 2000 (Isomap), Roweis & Saul 2000 (LLE), Moon et al. 2019 (PHATE), Coifman & Lafon 2006 (diffusion maps), McInnes et al. 2018 (UMAP), Trapnell et al. 2014 (Monocle), Setty et al. 2019 (Palantir), Wolf et al. 2019 (PAGA), Haghverdi et al. 2016 (diffusion pseudotime), Qiu et al. 2017 (reversed graph embedding), Street et al. 2018 (Slingshot), Saelens et al. 2019 (trajectory-inference comparison), La Manno et al. 2018 (RNA velocity), Bergen et al. 2020 (scVelo).
  - **Single-cell data integration and benchmarking:** Lotfollahi et al. 2022 (scArches), Luecken et al. 2022 (benchmarking integration), Korsunsky et al. 2019 (Harmony), Stuart et al. 2019 (Seurat integration), Gayoso et al. 2021 (totalVI).
  - **Evaluation and statistics:** Benjamini & Hochberg 1995 (BH-FDR), Venna & Kaski 2006 (local multidimensional scaling with trustworthiness).
  - **Protein foundation model precedent for manifold extraction:** Rives et al. 2021 (ESM), Lin et al. 2023 (ESMFold), Jumper et al. 2021 (AlphaFold), Nguyen et al. 2024 (Evo2), Pearce et al. 2025 (**Goodfire phylogeny manifold — the key precedent: Evo2 was shown to internally encode a compact manifold corresponding to the evolutionary tree of life, recoverable via rigorous geometric auditing**). This is the result that motivates the current paper's question: *can the same thing be done for single-cell biology?*
  - **Biological ground truth:** Han et al. 2018 (TRRUST v2), Garcia-Alonso et al. 2019 (DoRothEA), Szklarczyk et al. 2019 (STRING), Ashburner et al. 2000 (GO), Türei et al. 2021 (OmniPath — used as the ruler for the H38 generalisation case), Tabula Sapiens Consortium 2022, Regev et al. 2017 (Human Cell Atlas), Dominguez Conde et al. 2022 (cross-tissue immune cell atlas), Laurenti & Göttgens 2018 (hematopoietic landscape), Orkin & Zon 2008 (hematopoiesis paradigm), Weinreb et al. 2020 (clonal barcoding → fate lineage tracing), Velten et al. 2017 (hematopoietic continuum), Paul et al. 2015 (myeloid lineage heterogeneity).
  - **Knowledge distillation and sparse subnetworks:** Hinton et al. 2015 (distillation), Frankle & Carbin 2019 (lottery ticket hypothesis).
  - **Autonomous scientific discovery:** Lu et al. 2024 (AI Scientist).
  - **Predecessor Kendiukhov papers:** Kendiukhov 2025 (arxiv 2602.17532 — attention-based interpretability establishing that attention captures co-expression not unique regulatory signal), Kendiukhov 2026a (arxiv 2603.01752 — causal circuit tracing), Kendiukhov 2026b (arxiv 2603.02952 — SAE atlas revealing organised biological knowledge but minimal causal regulatory logic).

## 3. Inputs

### 3.1 Target foundation model

Any transformer-based single-cell foundation model that exposes native attention-value-projection weight matrices `A_{ℓ,h}` at every (layer, head) combination. The pipeline requires:

1. **Frozen checkpoint access** — the model is used read-only; no fine-tuning is performed at any stage.
2. **Per-head value projection matrices** — for each of `n_layers × n_heads` attention units, the learned matrix `A_{ℓ,h} ∈ ℝ^{d×d}` where `d` is the model's input gene-vocabulary size (not the model's hidden dimension — the operator acts in gene space). For scGPT whole-human: `d = 1,200`, so each head is a 1200×1200 matrix. For Geneformer V2-316M: `d = 1,152`.
3. **The ability to aggregate attention-value-projection matrices into layer blocks** — the main benchmark operator computes `A_early`, `A_mid`, `A_late` as the average of native heads across early, middle, and late layer blocks. Block boundaries are configurable; source paper uses layers 1–4 / 5–8 / 9–12 for scGPT's 12-layer stack.
4. **No forward pass through the full model is required for Stage-1 operator export** — the matrices are read directly from the checkpoint, not computed online. This is what makes the extraction "direct": the operator is a fixed feature map `x ↦ x A_{ℓ,h}`, where `x` is a 1200-dimensional gene-expression vector for one cell.

**Worked example in source paper:** scGPT whole-human (Cui et al. 2024) — 12 transformer layers, 8 heads per layer, `d=1,200`. Total operator library: **96 heads**, each a 1200×1200 matrix (~11 MB each as float32).

### 3.2 Biological ruler (the target distance matrix)

**Choose at design time**, before touching the model. The biological ruler is the ground-truth distance matrix `d_target[i, j]` for pairs of anchors `(i, j)`, derived from a curated biological ontology or interaction database. This is what the LET objective fits the latent geometry to.

Examples from the source paper and its generalisations:

| Biological system | Ruler | Source |
|---|---|---|
| **Hematopoiesis (H65)** | developmental-stage-ontology distances: number of developmental steps separating two cell types on a curated hematopoietic stage graph | Laurenti & Göttgens 2018; Paul et al. 2015; hand-curated stage DAG |
| **Intercellular communication (H38)** | OmniPath interaction-path distances: weighted graph distances on the ligand-receptor interaction network, square-root weighted | Türei et al. 2021 (OmniPath) |
| Cell cycle | cell-cycle phase ordering (G0 → G1 → S → G2 → M) | curated cell-cycle gene signatures |
| Spatial organisation | physical distances between tissue compartments / niches | spatial transcriptomics coordinates |
| Metabolic state progression | metabolic pathway flux distances | KEGG metabolic network + flux balance |
| Differentiation (non-hematopoietic) | lineage-specific developmental orderings | scVelo / CellRank / Palantir trajectory inference on ground-truth panels |

The ruler must satisfy three requirements:
1. **Biological coherence** — distances must reflect real biological relationships (not statistical co-expression, which would be circular since the foundation model is trained on co-expression).
2. **Coverage of anchor space** — every pair of anchors must have a defined distance (use graph shortest-path or hop count if full metric is unavailable).
3. **Independence from the foundation model** — the ruler must be constructed from external prior knowledge, never from the model's outputs (using the model's own distances would be self-consistent but trivially verifiable).

### 3.3 Cell populations and anchor construction

Per §2.2 of the source paper:

**Anchors** are centroid representations — averages of scGPT embeddings over all cells sharing a **(donor × tissue × cell-type)** label. This aggregation suppresses per-cell noise while preserving biological structure.

**Internal panel** (training for Stage 2 adaptor): the source paper uses **298 anchors, 564,253 cells, 7 hematopoietic branch groups, 5 tissues** drawn from Tabula Sapiens immune data processed through frozen scGPT.

**Strict non-overlap external panel** (critical for external validation, Phase 7): built by excluding all internal-panel observation IDs, then mapping remaining cells to the same stage ontology and aggregating fresh anchors. Source paper result: **616 anchors, 27 tissues, 21 donors, 34 fine stages, 7 branches**. Not a single cell from the training panel appears.

**Multi-donor zero-shot transfer panel** (Phase 8): completely separate cohort. Source paper: **7 cohorts, 165 anchors, 10,819 sampled cells**. No external data used during any training step.

### 3.4 Hardware and baseline methods (for Phase 12 evaluation)

- **Hardware:** CPU-bound for Stage-1 operator export and most Stage-2/3 training. GPU only needed for one-time scGPT forward passes if comparing against frozen-embedding baselines.
- **Baseline methods:** raw-expression log1p (1200 genes), PCA-10, SVD-10, CellTypist, scVI-10D, Scanpy-DPT, Palantir, and frozen scGPT avg-pool embeddings + 3-layer MLP. All used in the source paper's 88-split Robust-V2 benchmark.

## 4. Outputs

Per (foundation-model, biological-system) combination:

1. **Internal anchor table** (CSV): `(anchor_id, donor, tissue, cell_type, stage_label, n_cells_averaged, embedding_centroid)` for the internal training panel. Plus the biological-ruler distance matrix `d_target` computed from the anchor set.
2. **External strict non-overlap anchor table** (CSV): same schema, with explicit verification that zero internal observation IDs appear.
3. **Phase-1 autonomous-loop iteration log** (`iterations/iter_XXXX/`): per-iteration executor report, hypothesis branch JSON, gate-pass/gate-fail verdict, artifact-closure log, auto-generated Python script. Source paper's equivalent directory reference is `iterations/` but not published.
4. **Hypothesis-branch registry** (`reports/hypothesis_registry.json`): every branch tested during Phase 1, with its biological target, ruler choice, featurisation strategy, geometric fitting method (Isomap, geodesic-MDS, LET, etc.), gate results, and closure/pivot decision.
5. **Stage-1 frozen operator artifacts:**
   - `operator_drift_pooled.npy` — pooled drift map matrix of shape `[d, 3d]` for scGPT (i.e., 1200×3600, representing the concatenation of three pooled heads). Size ~17.5 MB.
   - `operator_per_head/layer{L}_head{H}.npy` — all `n_layers × n_heads` individual head matrices.
6. **Stage-2 learned adaptor weights:** `let_head_anchor.pt`, `let_head_cell.pt`, `let_head_hybrid.pt` — three head variants from the source paper (anchor-trained, cell-trained, hybrid-topology-preserving). Each is a small linear layer `W_enc ∈ ℝ^{d_feature × d_latent}` plus bias. For `d_latent = 10` and `d_feature = 2400` (pooled drift), this is ~24k parameters.
7. **Stage-3 task probes:** `probe_branch.pt`, `probe_stage.pt`, `probe_cd4cd8.pt`, `probe_mono_macro.pt`, `probe_pseudotime.pt`. Each is a tiny model (linear or 2-layer MLP), typically 5–170 trainable parameters per task.
8. **Quality gate report** (`reports/quality_gates.json`): per panel × variant, the four gate statistics (trustworthiness, random holdout correlation, donor holdout correlation, clade/branch holdout correlation), with pass/fail verdict. Gate thresholds: trustworthiness ≥ **0.80**, all holdout correlations ≥ **0.20**.
9. **Blocked-permutation null report** (`reports/blocked_permutation.json`): per panel × variant, blocked-permutation *p*-value under donor/tissue blocks, with a paired null-branch entry demonstrating that a biologically-nonsensical target ruler fails the same gates.
10. **External validation report** (`reports/external_validation.json`): strict non-overlap panel gate results, zero-shot transfer multi-donor results, expected-to-fail negative control panel results.
11. **Head/layer attribution table** (`reports/head_layer_screen.csv`): per (layer, head) unit, the single-head transfer score, CD4/CD8 AUROC, mono/macro AUROC, residualized correlation against the biological ruler.
12. **Multi-stage compaction chain** (`artifacts/compaction_chain/`): full drift (17.5 MB) → compact-top1 single-head (5.9 MB) → rank-64 SVD surrogate (0.73 MB) → hard-sparse surrogate (0.12 MB). Each variant retrains a fresh Stage-2 adaptor and Stage-3 probes.
13. **Factor-necessity ablation** (`reports/factor_ablation.json`): leave-one-factor-out ablation results per rank-1 SVD factor of the rank-64 surrogate, total ablation impact per factor, cumulative impact concentration curve, per-task drop magnitudes.
14. **Gene-program summary** (`reports/gene_programs.csv`): top read/write genes for the core factors (4 factors in the source paper explain 66.2% of total ablation impact), GO BP enrichment of those gene sets.
15. **Benchmark results** (`reports/benchmark_88splits.json`): 88-split Robust-V2 donor-holdout benchmark vs 8 baselines, per-endpoint paired Wilcoxon BH-*q*, paired split-level deltas.
16. **Runtime comparison** (`reports/runtime.json`): one-time preparation cost, per-split training cost, full-campaign cost vs frozen-scGPT-MLP baseline. Source paper: 34.5× speedup, ~1000× fewer trainable params.
17. **Deployable artifact** (`deploy/{model}_{system}_{variant}.tar.gz`): the compact operator + small adaptor + task probes, packaged for standalone use. Source paper's best: 5.9 MB for single-head compact-top1.
18. **Second-manifold validation report** (optional but recommended): run the entire pipeline again on a distinct biological system using a different ruler. Source paper validates on H38 intercellular communication with OmniPath ruler. Confirms the pipeline itself generalises beyond any single case.

## 5. Dependencies

- **Python ≥ 3.10** with: `torch ≥ 2.0`, `numpy ≥ 1.24`, `scipy ≥ 1.10`, `scikit-learn ≥ 1.3`, `pandas ≥ 2.0`, `anndata ≥ 0.10`, `scanpy ≥ 1.9`, `matplotlib ≥ 3.7`.
- **Foundation model:** for the scGPT worked example, `scgpt ≥ 0.2` from Cui et al. 2024. Checkpoint: scGPT whole-human (pre-training on ~33M cells).
- **Manifold learning and trajectory inference baselines:** `umap-learn`, `palantir`, `celltypist`, `scvi-tools ≥ 1.0` (for scVI baseline), `scanpy` (for DPT baseline).
- **Biological ontology access:** depends on the ruler. For hematopoietic: curated stage DAG (hand-maintained). For intercellular communication: `omnipath` Python package (Türei et al. 2021).
- **Autonomous-loop LLM backend:** the source paper uses the autonomous executor–reviewer loop from the earlier Kendiukhov projects. The specific LLM (Claude, Codex 5.3, or author-led Phase-2) is not explicitly stated in this paper but the loop architecture is consistent with the `topology-biomechinterp1` (Claude) and `topology-biomechinterp2` (Codex 5.3 xhigh) codebases. **Either works** — the prompts for executor and reviewer are the critical components, not the specific model.
- **Data:** Tabula Sapiens (Jones et al. 2022), plus any application-specific cohort (the source paper pulls multi-donor immune data from 7 cohorts for zero-shot validation). For OmniPath-based H38 generalisation: additional immune + lung + kidney Tabula Sapiens slices.
- **No GPU strictly required** — the Stage-1 operator export is a read-only file operation and the Stage-2 adaptor training is on cached features, not through the foundation model. GPU only needed for one-time scGPT forward passes to materialise cell-level features if you are using cell-trained head variants.
- **Compute budget (source paper):** full 12-split benchmark campaign completes in ~205 seconds on commodity hardware; the one-time frozen scGPT forward pass to extract cell features is the dominant cost at ~5,164 seconds. For comparison, the baseline frozen-scGPT + 3-layer MLP path takes ~7,074 seconds. The autonomous Phase-1 hypothesis sweep is dominated by executor script execution time plus LLM call latency.
- **Environment file:** not published with this paper. Use the `requirements.txt` from the companion Kendiukhov projects (`repos/topology-biomechinterp1/requirements.txt` or `repos/topology-biomechinterp2/requirements.txt`) as a starting point and add `scgpt`, `palantir`, `celltypist`, `omnipath`.

## 6. Methodology

The pipeline runs in fourteen phases. Phases 0–2 set up the target system; Phase 3 is the autonomous discovery loop; Phases 4–6 are the three-stage extraction; Phase 7–8 are external validation; Phases 9–11 are compaction and mechanistic interpretation; Phase 12 is benchmark evaluation; Phase 13 is second-manifold generalisation.

---

### Phase 0 — Target selection: model, system, and biological ruler

**Purpose:** commit to the three design choices that define the application before any training happens.

**Steps:**

1. **Choose the target foundation model.** Any transformer-based single-cell foundation model whose attention-value-projection weights can be read directly from the checkpoint. Examples: scGPT whole-human (worked example), Geneformer V2-316M, scBERT, scFoundation, UCE.
2. **Choose the target biological system.** This is the biology you hypothesise the model may internally encode. Examples: hematopoietic development, intercellular signalling, cell-cycle progression, metabolic state, spatial tissue organisation, circadian rhythm, stress response, immune activation, etc. **The choice should be motivated by prior evidence that the system has recoverable geometric structure in curated ontologies — do not pick systems where the ground truth is itself derived from co-expression (that would be circular).**
3. **Commit to the biological ruler.** Build the pairwise anchor-distance matrix `d_target` from an external ontology. Verify it is:
   - Coherent (distances reflect curated biology)
   - Complete (every pair of anchors has a defined distance)
   - Independent of the target foundation model
4. **Define the null branch.** Before running Phase 1, commit to a paired **biologically-nonsensical target** that uses the same data but a shuffled or swapped ruler. For the source paper's H65 hematopoietic branch, the paired null H66 used a different biological target — same cell set, same operator library, same loop, but a target ruler that cannot succeed. This is the "sanity gate": if H66 also passes, Phase 1 is detecting pipeline artefacts rather than real biology.

**Code reference:** no external scripts for this phase — the choice is a design decision that feeds into Phases 1–3. Record it in `planning/research_plan.md`.

---

### Phase 1 — Data preparation and anchor construction

**Purpose:** build the internal and external anchor panels that feed the extraction pipeline.

**Steps:**

1. **Process cells through the frozen foundation model.** For scGPT: extract the hidden states or rank-based representations at the gene level. For Geneformer: extract per-gene per-layer hidden states similarly.
2. **Construct the internal anchor panel.** Group cells by **(donor × tissue × cell-type × stage)** quadruples. Compute the centroid (mean embedding) for each quadruple. Retain anchors with sufficient cell count (e.g., ≥ 5 cells per anchor). Source paper: 298 anchors, 564,253 cells, 7 branch groups, 5 tissues.
3. **Construct the strict non-overlap external panel.** Start from a fresh dataset (independent donors, tissues, or cohorts). **Explicitly exclude every internal observation ID** from the external panel before aggregating. Compute external anchors using the same (donor, tissue, cell-type) grouping. Source paper: 616 anchors, 27 tissues, 21 donors, 34 fine stages. **Verify zero overlap** with a deterministic ID intersection check.
4. **Construct the multi-donor zero-shot transfer panel.** A separate cohort collection for Phase 8 frozen-head zero-shot testing. Source paper: 7 cohorts, 165 anchors, 10,819 cells.
5. **Compute the biological ruler matrix** `d_target[i, j]` for all anchor pairs. For hematopoietic: number of developmental-stage edges on the curated DAG. For OmniPath: square-root-weighted graph distance on the ligand-receptor interaction path network.
6. **Persist everything** as tables and matrices: `anchors_internal.csv`, `anchors_external.csv`, `anchors_multidonor.csv`, `d_target_internal.npy`, `d_target_external.npy`.

**Critical rule:** no external panel data is used during any training step of Stages 2 or 3. This is enforced in Phase 7 by a deterministic artefact audit (Supplementary S4 of the source paper).

---

### Phase 2 — Hierarchical quality gates (the shared evaluation standard)

**Purpose:** establish the fixed quality-gate thresholds that every Phase-1 hypothesis candidate, every head variant, every compaction stage, and every external panel must pass. These are applied *uniformly* — internal, external, zero-shot — so that positive claims are only accepted when all gates fire simultaneously.

**The four shared gates (Supplementary S3):**

1. **Trustworthiness ≥ 0.80** — Venna & Kaski 2006 local neighbourhood preservation. The fraction of each point's nearest neighbours in the high-dimensional input that remain neighbours in the low-dimensional embedding.
2. **Random holdout correlation ≥ 0.20** — Spearman correlation between predicted latent distances and target ruler distances on a random 20% holdout.
3. **Donor holdout correlation ≥ 0.20** — same correlation computed when donors are held out (tests cross-donor generalisation).
4. **Clade / branch holdout correlation ≥ 0.20** — same correlation when entire biological branches (lineages) are held out (tests structural generalisation rather than statistical co-occurrence).

**Fifth gate (adversarial reviewer):** the autonomous reviewer role rejects candidates that either (a) fail any of the four numeric gates, (b) show confound sensitivity in bootstrap audits, or (c) would pass under a matched null branch. This catches pipeline artefacts.

**Persistence:** `reports/quality_gates.json` contains the gate threshold definitions (fixed once at project start) and the per-variant per-panel gate results.

---

### Phase 3 — Autonomous research loop (Phase-1 broad hypothesis search + Phase-2 focused investigation)

**Purpose:** systematically explore the combinatorial space of (biological target × operator type × featurisation × fitting method) and identify the branch that passes all quality gates with its matched null branch failing.

**Architecture:** two-role executor–reviewer loop, same as `topology-biomechinterp1` (Claude) / `topology-biomechinterp2` (Codex 5.3 xhigh).

**Phase 3a — Broad hypothesis search:**

1. **Executor agent** generates and runs hypothesis branches. Each branch varies:
   - **Biological target:** developmental ordering, regulatory structure, spatial organisation, communication geometry, etc. (the specific ruler is usually fixed; the biological *interpretation* is what varies here).
   - **Featurisation strategy:** attention drift (layer-pairwise differences), raw embeddings, mixed operators, value-weighted aggregations.
   - **Geometric fitting method:** Isomap, geodesic-MDS, LET (Latent Embedding Transfer), locally linear embedding, diffusion maps.
2. **Each candidate branch is evaluated against the four quality gates** (Phase 2), with blocked-permutation *p* ≤ **0.001** as an additional requirement.
3. **Reviewer agent** rejects candidates that fail any gate or show confound sensitivity, and proposes pivots for the next executor iteration.
4. **The loop runs at scale** — dozens of branches instantiated, tested, and rejected before a positive branch emerges. Source paper: H65 (hematopoietic developmental manifold) was the first strong positive; H66 (paired null with a different biological target) failed, confirming branch-specificity.
5. **Key output of Phase 3a:** a single validated hypothesis branch (here called "H65") with fully-logged artefacts, plus a paired null branch that cleanly fails.

**Phase 3b — Focused investigation (transitioning to author-led):**

Once the positive branch is identified, the workflow transitions to author-led investigation. Phase 3b runs:

1. **Methodological closure tests:** objective ablations, confidence intervals, structured holdouts, equation-level LET replication on the expanded panel.
2. **Extraction and benchmarking** of the manifold as a standalone algorithm (Phases 5–6, 12 below).
3. **Multi-stage operator compaction** (Phases 9–10).
4. **Mechanistic interpretability via factor ablation and sparse factorisation** (Phase 11).

Phase 3b uses the **same quantitative gate logic** as Phase 3a — the difference is that the experimental design, analysis, and interpretation are conducted manually rather than autonomously generated.

**Artefact-closure protocol (Supplementary S2):** before interpreting any result, the loop enforces a 4-step pipeline:
- **A: Local path reconciliation** (canonical-path verification)
- **B: Local materialisation** (rebuild missing derived artefacts)
- **C: Network retrieval** (autonomous download when local recovery fails)
- **D: Schema enrichment** (metadata audit and harmonisation)

For a validated positive branch to be promoted, at minimum A and D must succeed; if any unresolved artefacts remain, the branch is held at "pending."

**Code reference:** reuse `repos/topology-biomechinterp2/loop/run_codex_topology_autoloop.py` (Codex 5.3 xhigh) or `repos/topology-biomechinterp1/loop/run_claude_topology_autoloop.py` (Claude). The prompts in `prompts/executor_prompt_topology_hypothesis_screening.md` and `prompts/brainstormer_prompt_template.md` require domain adaptation for manifold search, but the overall architecture transfers directly.

---

### Phase 4 — Operator library enumeration (Stage 1 preparation)

**Purpose:** enumerate the full library of candidate operators that can be extracted from the foundation model's attention tensor, from which Phase 3a selects the best-performing one.

**Steps:**

1. **Per-head operators.** For each of `n_layers × n_heads` attention units, read the value-projection matrix `A_{ℓ,h} ∈ ℝ^{d×d}`. For scGPT whole-human this yields 96 candidate operators (12 × 8), each ~11 MB.
2. **Pooled drift operators.** Partition layers into early, middle, late blocks. Compute pooled heads `A_early`, `A_mid`, `A_late` as the block-wise average of native heads. Construct the pooled drift map:
   `f_drift(x) = [(x A_early − x A_mid); (x A_mid − x A_late)] ∈ ℝ^{2d}`
   (concatenation of two `d`-dimensional drift vectors). For scGPT `d=1200`, this gives a `2400`-dimensional feature vector per cell.
3. **Top-k weighted head combinations.** Compact operators selected by Phase-1 screen scores:
   `Ā_k = Σ_{i=1}^{k} α_i A_{ℓ_i, h_i}^⊤`
   where weights `α_i` are Phase-1-selected per-head screen scores. Source paper uses `k ∈ {1, 3}`.
4. **Value-weighted and mixed operators.** Variants that combine pooled heads with other transformations (e.g., `softmax(QK^T/√d) · V` context-layer outputs). Tested in Phase 3a but the source paper's winner is the pure pooled drift map.
5. **Persist the operator library** as `operators/{name}.npy`. Each is metadata-tagged with the heads/layers it was built from.

**Critical note on direct operator extraction:** this stage does not require any forward passes through the foundation model. The matrices `A_{ℓ,h}` are read from the checkpoint directly; the feature map `x ↦ x A_{ℓ,h}` is a single matrix multiplication per cell. **No target labels are used and no parameters are optimised.** This is what makes the extraction "direct" — it is a pure checkpoint export.

---

### Phase 5 — Stage 2: Lightweight learned adaptor via LET

**Purpose:** train a small task-agnostic head `g_θ` that maps fixed operator features to a compact latent `z` whose pairwise distances match the biological ruler.

**The Latent Embedding Transfer (LET) objective** (Section 2.3 of source paper):

Given fixed feature vector `x = f_drift(cell) ∈ ℝ^{2d}` and latent dimension `d_latent = 10`:

```
z = W_enc (x − b)                                     [linear projection]
d̂_{ij} = β arccos(cos(z_i, z_j))                    [arc-cosine latent distance]
L = ||d̂ − d_target||² + α ||W_enc^⊤ z + b − x||²    [fit + reconstruct]
```

The first term encourages the latent to match the biological ruler (arc-cosine distance on the normalised latent matches the target distance). The second term is a reconstruction regulariser preventing trivial collapse.

**Training variants (all tested in the source paper):**

1. **Anchor-trained head.** Train directly on the 298 internal anchors. Preserves compact branch-like structure. Source paper: trustworthiness 0.972, CD4/CD8 AUC 0.744 (global 3D), mono/macro 0.759.
2. **Cell-trained head.** Train on 6,925 individual cells (stage-balanced, max 500 cells/stage), not on anchors. Uses a weighted 4-term objective:
   `L_cell = w_stage · L_stage + w_local · L_local + w_recon · L_recon + w_cls · L_cls`
   with weights `w_stage = 1.0, w_local = 0.1, w_recon = 0.08, w_cls = 0.4`. 120 epochs, batch size 896, `d_latent = 10`. Source paper: trustworthiness 0.969, CD4/CD8 AUC **0.886**, mono/macro **0.937**. **This is the variant that wins the Robust-V2 benchmark.**
3. **Hybrid topology-preserving head.** Augments the cell-level objective with an anchor-topology prior:
   `L_hybrid = L_cell + λ_topo ||D(μ(z)) − D_ref||² + λ_compact Σ_s E_{i∈s} ||z_i − μ_s(z)||²`
   Two sweeps tested:
   - Strong topology: `λ_topo ∈ {0.04, 0.08, 0.12, 0.18}`, `λ_compact = 0.015`
   - Conservative: `λ_topo ∈ {0.005, 0.01, 0.02, 0.03}`, `λ_compact = 0`
   Selected via a fixed external objective: `score = ρ_resid + 0.25 AUC_CD4CD8 + 0.25 AUC_Mono/Macro + 0.40 silhouette_branch − 0.20 spread_norm`.

**Quality gate enforcement:** after training, each head variant must pass all four quality gates on internal data (trustworthiness ≥ 0.80, random/donor/clade holdout correlations ≥ 0.20) before promotion. Source paper's H65 cell-trained head: trustworthiness 0.979, random 0.495, donor 0.496, branch 0.382 — comfortably above thresholds.

**Effective dimensionality:** the LET transfer objective increases consistently with latent dimensionality:
- `d=3`: objective 0.507, trustworthiness 0.946
- `d=6`: objective 0.604, trustworthiness 0.976
- `d=10`: objective 0.632, trustworthiness 0.985

All gates pass at each tested `d`. Source paper selects `d=10` as the effective size of the hematopoietic manifold (8–10 in a flat subspace).

**Code reference:** not published externally. Prototype implementation requires ~200 lines of PyTorch (linear layer + LET loss + gate checker).

---

### Phase 6 — Stage 3: Task-specific readout

**Purpose:** train small downstream probes on the task-agnostic latent `z` for classification and pseudotime prediction. These probes are **not** part of the shared representation.

**Steps:**

1. **Per-task probe architecture:** typically linear (`z → C` classes) for classification or linear regression for pseudotime. For endpoints where nonlinearity helps (CD4/CD8 subtype), use 2-layer MLP (`10 → 64 → C`).
2. **Training:** standard classification/regression on internal training cells using the latent `z` as input. Probe parameters: **5–170 trainable parameters** per task (vs 172,000+ for the frozen-scGPT-MLP baseline).
3. **Probe depth sensitivity** (Supplementary S21): linear probes are the robust default. MLP2 (`10 → 64 → C`) gives BH-significant gains on CD4/CD8 AUROC (+0.0045) but worsens mono/macro (−0.0076) and stage endpoints. Keep linear as the default; use MLP2 as a targeted nonlinear follow-up for subtype-sensitive tasks only.
4. **Direct learnability baseline** (Supplementary S22): a tiny MLP trained directly on raw expression (1200 → 64) learns stage structure but does **not** recover the extracted head's subtype-sensitive geometry. CD4/CD8 AUROC 0.828 (direct MLP) vs **0.856** (extracted); mono/macro 0.920–0.926 vs **0.956**. **The extraction gain is not reducible to "train a tiny MLP on raw input."**

**Quality signature:** on Replogle / Tabula / your dataset, the 10D-head + MLP2 readout should **exceed 7/8 pooled metrics** of a frozen-scGPT avg-pool + 3-layer MLP baseline, with 6/8 BH-significant after paired splitwise correction.

---

### Phase 7 — External validation: strict non-overlap panel

**Purpose:** confirm that the manifold is not an internal-only artefact by applying the same quality gates on a held-out cohort with **no observation-ID overlap** with the internal training panel.

**Steps:**

1. **Freeze the Stage-2 adaptor and Stage-3 probes** as trained on the internal panel.
2. **Compute anchor centroids on the strict non-overlap external panel** using the same (donor × tissue × cell-type) grouping.
3. **Apply the frozen head to the external anchors** to produce external latents.
4. **Evaluate the four quality gates** on the external panel. For H65 on the Tabula Sapiens strict non-overlap panel:
   - Trustworthiness **0.985**
   - Random holdout **0.594**
   - Donor holdout **0.596**
   - Branch holdout **0.453**
   All pass.
5. **Expected-to-fail negative control:** run the same evaluation on a *biologically-mismatched* external panel. The source paper uses a **lung panel** (not relevant to hematopoietic developmental structure) as a negative control:
   - Trustworthiness **0.617** (fails ≥ 0.80)
   - Random holdout **0.236**
   - Donor holdout **0.687**
   - **Branch holdout 0.058 (fails)**
   The lung panel **fails robust transfer** despite strong global correlation — demonstrating that **global correlation alone is insufficient** and that trustworthiness + clade holdout controls are required.

**This is the critical validation step.** A manifold that passes only the internal gates may be overfitting to internal data structure. A manifold that also passes the strict non-overlap external panel while the negative control fails is genuinely capturing the biology.

---

### Phase 8 — Frozen-head zero-shot transfer (Supplementary S4)

**Purpose:** demonstrate that the Stage-2 adaptor trained on internal data transfers to a completely separate cohort without any retraining — the strongest test of whether the extraction generalises.

**Train-once, transfer-many protocol:**

1. **Train the Stage-2 head only on the internal anchor panel.** Select the best internal dimension (source paper: `d = 10`).
2. **Reconstruct the exact head deterministically** (source paper uses seed `15052`). Freeze all parameters.
3. **Build external anchors from a separate multi-donor panel** (source paper: 7 cohorts, 165 anchors, 10,819 cells).
4. **Apply the frozen head directly** — zero-shot, no external optimisation or retraining.
5. **Evaluate with the same blocked-permutation and trustworthiness controls.**

**Expected result** (source paper, Table 2): 165 anchors × 7 donors × 7 stages, correlation **0.500**, residualized correlation **0.476**, trustworthiness **0.993**, blocked *p* = **0.0005**. All gates pass on a completely separate cohort without any fitting.

**This protocol separates representation transfer from dataset-specific fitting** — the critical distinction for claims that the extracted algorithm captures model-intrinsic biology rather than training-panel-specific patterns.

---

### Phase 9 — Head/layer attribution scan

**Purpose:** localise the transferable biological signal to specific (layer, head) units within the 96-unit attention tensor. This is a prerequisite for meaningful compaction.

**Steps:**

1. **Single-head evaluation.** For each of the 96 units `(ℓ, h)`, train a single-head feature map `x ↦ x A_{ℓ,h}` with LET-10D on internal anchors.
2. **Zero-shot evaluation on the strict non-overlap external panel.** Each single-head variant is scored on the same gate stack.
3. **Rank heads by residualised correlation** (correlation after regressing out donor and tissue confounds).
4. **Expected top 5 for scGPT H65** (Supplementary Table 21):
   - **L2H5: residualized correlation 0.634, trustworthiness 0.985, CD4/CD8 3D AUROC 0.711, mono/macro 3D AUROC 0.873** — the winner
   - L0H3: 0.627, 0.981, 0.743, 0.769
   - L6H6: 0.621, 0.984, 0.765, 0.867
   - L6H4: 0.610, 0.986, 0.765, 0.810
   - L0H7: 0.636, 0.977, 0.746, 0.765
5. **Two-view interpretation:** single heads can carry strong global geometry (L2H5 residualised corr 0.634 matches pooled-head variants: anchor 0.605, cell 0.612, hybrid 0.632), while pooled objectives expose fine subtype margins (CD4/CD8 AUC 0.911 hybrid 3D vs 0.711 L2H5 single head).
6. **The head attribution table is the foundation for Phase 10 compaction.**

**Critical finding:** transferable biological geometry is distributed across multiple layers, but the signal is **localised enough** that a single attention head can carry most of it.

---

### Phase 10 — Multi-stage operator compaction

**Purpose:** systematically compress the extracted operator through a hierarchy of compression stages, retraining lightweight downstream components at each step to measure performance loss.

**The compaction chain:**

| Variant | Size | Description | Expected BH losses vs reference (of 8) |
|---|---|---|---|
| **Full drift (3 pooled heads)** | 17.5 MB | Reference: `f_drift` from early/mid/late pooled heads | — |
| **Compact top1 (L2H5 single head)** | **5.9 MB** | Replace pooled drift with `x ↦ α · xA_{L2H5}` | **0/8** — best single head |
| **Top1 + rank64** | 0.73 MB | Truncated-SVD surrogate of L2H5: `Ā ≈ U_r V_r^⊤` for `r=64` | 5/8 |
| **Top1 + rank32** | 0.37 MB | Same at `r=32` | 6/8 |
| **Hard sparse (16 factors, 60 genes)** | 0.12 MB | Top-16 SVD factors, top-60 read/write genes per factor | 7/8 |

**Steps:**

1. **Compact-V1 single-head operator.** Identify the top-ranked head from Phase 9 (L2H5 for scGPT H65). Replace the pooled drift map with `Ā_1 = α · A_{L2H5}`. **Retrain only the small Stage-2 adaptor and Stage-3 probes** on the same internal panel using the same LET objective and training protocol. **Do not retrain the foundation model or any attention weights.**
2. **Evaluate the retrained compact-V1 head** on the same benchmark as the full drift reference. Source paper result: **0/8 BH-significant losses**. The single-head compact operator preserves classification utility without significant loss across 8 repeated classification endpoints.
3. **Low-rank truncated-SVD surrogate.** Take the compact top1 operator and fit truncated-SVD surrogates `Ā_k ≈ U_r V_r^⊤` for `r ∈ {8, 16, 32, 64}`. Retrain adaptor + probes at each rank. Source paper: **rank-64 (0.73 MB) is the viable low-rank boundary** with 5/8 BH-significant losses; rank-32 (0.37 MB) yields 6/8 losses.
4. **Hard sparse pruning.** Apply top-*k* pruning to the rank-64 surrogate: retain only a subset of rank-1 factors and only the top-*k* read genes (highest input loadings) and top-*k* write genes (highest output loadings) per retained factor. Source paper's best sparse point: 16 factors, 60 genes/factor, 1,920 active operator weights, 124 KB deployable package. **7/8 BH-significant losses** vs compact top1 — informative for interpretability but not frontier accuracy.
5. **Each compression stage retrains the same lightweight adaptor and task probes; the compressed operator `Ā_k` is always treated as a *compressed surrogate*, not a direct checkpoint export of new native parameters.**

**Persistence:** `artifacts/compaction_chain/` contains each variant as a tarball plus a comparison report (`compaction_chain.csv`).

**Critical insight:** compaction localises the biology. When a single attention head carries the signal, it becomes possible to ask *what specific computation* that head is doing — which is what Phase 11 addresses.

---

### Phase 11 — Factor necessity and mechanistic interpretability

**Purpose:** decompose the rank-64 compact operator into 64 rank-1 factors and identify which factors are causally necessary for downstream task performance.

**Steps:**

1. **Decompose the rank-64 compact operator via truncated SVD.** Yielding 64 rank-1 components, each defined by an input gene-loading vector (read genes) and an output gene-loading vector (write genes).
2. **Fixed-probe leave-one-factor-out ablation.** Take the rank-64 compact operator and its trained Stage-2/Stage-3 components. **Keep the probes fixed.** For each factor, zero it out at inference time and measure the drop in each of 8 classification and regression endpoints. This measures *necessity* of the existing factorisation, not recoverability after retraining.
3. **Rank factors by total pooled ablation impact.**
4. **Expected for scGPT H65 rank-64** (Table 7 / Figure 4):
   - **f01 (branch routing factor):** drop **0.430** — dominant on branch classification. Read genes: CSF3R, SLC6A1, IL7R. Write genes: CSF3R, SLC25A37, FCER1G. Enriched programs: **monocyte/macrophage differentiation, naive CD8⁺ T, granulopoiesis**.
   - **f00 (stage ordering factor):** drop **0.280** — strongest stage signal, granulocytic/T-NK developmental axis. Read: EPB41, GMR1, VPEL3. Write: ISHC, LGALS14, EPB41. Programs: **classical monocyte, naive CD8⁺ T maturation**.
   - **f02 (lymphoid contrast factor):** drop **0.254** — B-cell vs T/NK separation. Read: IL7R, GSTPL, IGF2R. Write: HLA-DRA, MARCH6, IGKV5. Programs: **lymphoid B vs T/NK, CD4⁺ T program**.
   - **f03 (mono/macro structure factor):** drop **0.241** — monocyte/macrophage vs granulocytic. Read: LGALS3, GT58, LYZ. Write: VPEL5, ALPL, SLC28A3. Programs: **macrophage, granulocyte, B-cell markers**.
5. **Cumulative impact concentration:** the top 4 factors explain **66.2% of total pooled ablation impact**; the top 10 explain 75.8%; the top 16 explain 82.4%.
6. **Core-sufficiency test.** Run a core-only variant with just `{f00, f01, f02, f03}` (all other 60 factors zeroed). Measure: branch balanced accuracy drops 0.820 → **0.572**, CD4/CD8 AUROC drops 0.846 → **0.699**. All 8 endpoints BH-significantly worse than intact. **The four-factor core is necessary but not sufficient** — the remaining 60 factors contribute incremental refinement.
7. **Exhaustive 15-subset interaction sweep** over the 4 core factors. Task-specialised circuitry is visible: **mono/macro is best recovered by the pair `{f01, f03}` alone (0.95× intact)**, while branch and stage endpoints require all 4 factors. Best subsets per endpoint indicate task-specific circuitry rather than a monotone additive ladder.
8. **Persistence in sparse surrogate.** The same four-factor core persists after hard-sparse pruning (16 factors, 60 read/write genes each), explaining **68.9% of sparse-specific ablation impact**. Read/write gene overlaps between the rank-64 and sparse factorisations confirm **the same mechanistic substrate survives aggressive compression**.
9. **Gene Ontology enrichment.** For each core factor, run GO BP enrichment on the top read + write genes (e.g., via cached local marker enrichment or an external pathway tool). The source paper reports enrichment in hematopoietic-specific programs that match the biological-ruler orientation.

**This phase represents the deepest mechanistic decomposition of a foundation-model-derived algorithm that the paper's author claims exists** — the four-factor core is traceable from the rank-64 SVD all the way to explicit biological gene programs, and those programs survive aggressive compression.

---

### Phase 12 — Benchmark evaluation vs established baselines

**Purpose:** test the extracted algorithm as a standalone method against 8 established single-cell analysis baselines using a rigorous multi-split donor-holdout protocol.

**Steps:**

1. **Define comparators.** Source paper uses 8 baselines:
   - Raw expression (1200-gene log1p)
   - PCA-10
   - SVD-10
   - CellTypist (Dominguez Conde et al. 2022)
   - scVI-10D (Lopez et al. 2018)
   - Scanpy-DPT (diffusion pseudotime, Haghverdi et al. 2016)
   - Palantir (Setty et al. 2019)
   - Frozen scGPT avg-pool + 3-layer MLP
2. **Use the finalised Robust-V2 benchmark campaign.** 88 grouped donor-holdout splits, 100,000 training cells per split, donor-local test-only pseudotime. Comprises one 40-split all-method run plus two 24-split core-only runs.
3. **Endpoints** (8 total):
   - Branch balanced accuracy
   - Branch macro-F1
   - Stage balanced accuracy
   - Stage macro-F1
   - CD4/CD8 AUROC
   - CD4/CD8 balanced accuracy
   - Mono/Macro AUROC
   - Pseudotime-depth Spearman correlation
4. **Paired statistical comparisons:** split-level donor-mean differences, two-sided Wilcoxon signed-rank test, BH-FDR correction across all endpoints.
5. **Expected result for H65 cell head** (Table 11):
   - **Pseudotime Spearman: 0.249 (cell head), best comparator 0.132 (H65 hybrid), Δ = −0.0561, BH-q = 2.7 × 10⁻⁷** — dominant.
   - **CD4/CD8 AUROC: 0.867 (cell head), vs 0.858 (hybrid conservative), Δ = −0.0088, BH-q = 1.1 × 10⁻¹⁵**.
   - **Mono/Macro AUROC: 0.951 (cell head), vs 0.949 (hybrid conservative), BH-q = 7.1 × 10⁻¹²**.
   - Branch macro-F1, stage macro-F1, and mono/macro balanced accuracy are significantly led by specific comparators — the cell head is **mixed by endpoint** but dominates on pseudotime-depth alignment.
6. **Raw-expression MLP direct-learnability test.** Train a 1200-gene MLP (1200 → 64 or 1200 → 128 → 64) directly on raw expression. Result: 0.828 CD4/CD8 AUROC vs 0.856 for extracted head; 0.920–0.926 mono/macro vs 0.956. All BH-significant. **The extraction cannot be reduced to "train a tiny MLP on raw input."**
7. **Runtime comparison** (Supplementary S23 / Table 19):
   - Reference 1200-gene matrix extraction: 164.7 s (one-time shared)
   - H65 cell head cache build: **99.5 s** (drift + head training)
   - Frozen scGPT avg-pool cache build: **5,163.8 s** (~52× gap)
   - H65 per-split evaluation: 8.8 s
   - Frozen scGPT MLP per-split: 159.2 s (~18× gap)
   - Full 12-split campaign: **~205 s (extracted) vs ~7,074 s (scGPT-MLP)** — **34.5× speedup**
   - Trainable parameters per task: **5–170 (extracted) vs 172,610–174,690 (scGPT-MLP)** — ~1000× reduction
8. **Unsupervised branch recovery caveat** (Supplementary S18). When the extracted head's output is fed through an unsupervised wrapper (Scanpy Leiden+PAGA or Palantir fate assignment) and compared on Adjusted Rand Index (ARI), raw expression performs **comparably or better**. The extracted head's advantage is concentrated in **supervised classification and pseudotime ordering**, not in unsupervised topology recovery. **Do not claim the extracted head recovers unsupervised cluster structure.**

---

### Phase 13 — Second-manifold generalisation (the pipeline-level validation)

**Purpose:** demonstrate that the pipeline itself (not just its hematopoietic instantiation) generalises to distinct biological systems.

**Steps:**

1. **Choose a second biological system** that is qualitatively different from the first. The source paper generalises from developmental ordering (H65 hematopoiesis) to **intercellular communication (H38)**.
2. **Build a new biological ruler** from a different ontology. For H38: **OmniPath ligand-receptor interaction paths** with square-root-weighted graph distances.
3. **Re-run the autonomous Phase-1 loop** with the new ruler. The source paper's H38 Phase-1 sweep covered 3 seeds × 2 methods (geodesic MDS, Isomap) × k ∈ {10, 15, 20} on 104 anchors built from 30 selected ligand–receptor interaction-path pairs. Best branch: **seed 43, geodesic MDS, k = 10**.
4. **Run the same quality gates.** For H38 canonical internal panel:
   - Trustworthiness median **0.908** (range 0.891–0.928)
   - Geodesic-anchor correlation **0.828**
   - Blocked-permutation *p* = **0.0005**
   - Manifold-minus-PCA2 baseline margin **+0.054**
   - Subsample/cell-type-holdout sign consistency = 1.0
   All internal gates pass.
5. **Run external validation.** For H38, the initial external panel failed strict gates (trustworthiness 0.937, correlation 0.909, but margin vs PCA-2 was −0.074 — negative). Negative controls also did not cleanly separate.
6. **Iterative rescue with no target-dataset leakage** (Supplementary S36). Expand internal training coverage — more donors, more cell types, more tissues — but **never touch the external panel**. Source paper rescue progression (v4 → v11): external correlation rose 0.885 → 0.983, with final version using 100,000 cells, 392 internal anchors, 23 donors, 58 cell types, 3 tissues (immune + lung + kidney). Final result: **external correlation 0.983, internal trustworthiness 0.908, baseline margin +0.006, blocked-permutation *p* = 0.0005**. **All rescue was via expanded internal coverage, never via gate relaxation.**
7. **Axis interpretation.** For H38, Axis 1 carried broad-class structure with **η² = 0.735**, with cytokine/chemokine and lipid/metabolic programs on opposite loadings (ρ = −0.549 and +0.566). The axes resolve biologically meaningful signalling program contrasts.
8. **Report the generalisation as a case study with its own maturity level.** H38 is framed as "successful generalisation case" with lower maturity than H65 — classification benchmark is mixed (2D H38 beats 2D PCA on macro-F1 0.482 vs 0.464 but loses to 6D PCA 0.531). **The claim is that the pipeline generalises, not that every extracted manifold is equally strong.**

**This phase validates the pipeline itself.** Without a second-manifold case study, the H65 result is a single point and the "generalisable methodology" claim cannot be made.

---

## 7. Parameters

| Parameter | Default | Alternate / sensitivity | Used in |
|---|---|---|---|
| `target_foundation_model` | scGPT whole-human | Geneformer V2-316M, scBERT, scFoundation, UCE | Phase 0 |
| `target_biological_system` | Hematopoietic development | Intercellular communication (H38), cell cycle, metabolic state, spatial, immune activation | Phase 0 |
| `biological_ruler` | Curated stage DAG (H65) | OmniPath (H38), KEGG, custom | Phase 1 step 5 |
| `n_internal_anchors` | **298** (H65), **104 → 392 after rescue** (H38) | system-dependent | Phase 1 |
| `n_external_anchors_strict` | **616** (H65 Tabula Sapiens non-overlap) | — | Phase 1, Phase 7 |
| `n_multidonor_transfer` | **165 anchors, 7 cohorts** | — | Phase 1, Phase 8 |
| `trustworthiness_gate` | **≥ 0.80** | non-negotiable | Phase 2 |
| `random_holdout_gate` | **≥ 0.20** | non-negotiable | Phase 2 |
| `donor_holdout_gate` | **≥ 0.20** | non-negotiable | Phase 2 |
| `clade_holdout_gate` | **≥ 0.20** | non-negotiable | Phase 2 |
| `blocked_permutation_p` | **≤ 0.001** | non-negotiable | Phase 2 |
| `n_permutations` | at least 2000 (source paper does not specify exact n) | — | Phase 2 |
| `phase1_null_branch_required` | **true** — paired biologically-nonsensical branch must fail the same gates | — | Phase 3a |
| `operator_feature_dim` | `2400` (pooled drift for scGPT) | `d` (single head), variable | Phase 4 |
| `n_layers_block_partition` | 3 blocks (early/mid/late) | 2 or 4 blocks | Phase 4 step 2 |
| `top_k_heads_pooled` | **3** (pooled drift) or **1** (compact top1) | 1–5 | Phase 4, Phase 10 |
| `d_latent` | **10** | 3, 6, 10 (gate pass at all) | Phase 5 |
| `let_alpha_reconstruction_weight` | source paper does not specify exact value | sensitivity via S3 | Phase 5 |
| `let_beta_distance_scale` | source paper does not specify exact value | — | Phase 5 |
| `cell_trained_n_cells` | **6,925** | stage-balanced, max 500 cells/stage | Phase 5 |
| `cell_trained_epochs` | **120** | — | Phase 5 |
| `cell_trained_batch_size` | **896** | — | Phase 5 |
| `cell_trained_w_stage` | **1.0** | — | Phase 5 |
| `cell_trained_w_local` | **0.1** | — | Phase 5 |
| `cell_trained_w_recon` | **0.08** | — | Phase 5 |
| `cell_trained_w_cls` | **0.4** | — | Phase 5 |
| `hybrid_lambda_topo_strong` | **{0.04, 0.08, 0.12, 0.18}** | — | Phase 5 |
| `hybrid_lambda_topo_conservative` | **{0.005, 0.01, 0.02, 0.03}** | — | Phase 5 |
| `hybrid_lambda_compact_strong` | **0.015** | — | Phase 5 |
| `probe_architecture_default` | **linear** | 2-layer MLP for CD4/CD8 | Phase 6 |
| `probe_trainable_params_per_task` | **5–170** | vs 172k for frozen-scGPT-MLP | Phase 6 |
| `n_robustv2_splits` | **88** grouped donor-holdout | — | Phase 12 |
| `robustv2_training_cells_per_split` | **100,000** | — | Phase 12 |
| `zero_shot_seed` | **15052** (deterministic reconstruction) | — | Phase 8 |
| `compaction_ranks_svd` | **{8, 16, 32, 64}** | — | Phase 10 step 3 |
| `hard_sparse_n_factors` | **16** | — | Phase 10 step 4 |
| `hard_sparse_n_genes_per_factor` | **60** | — | Phase 10 step 4 |
| `factor_ablation_protocol` | **fixed-probe leave-one-out** (no retraining) | — | Phase 11 |
| `n_core_factors` | **4** (top-4 by impact) | 1–16 | Phase 11 |

**Critical design rules:**
- **Gates are set before Phase 1 starts** and are applied uniformly across internal, external, and zero-shot panels. They are **never loosened**.
- **Blocked permutation p is committed up front** (≤ 0.001 for primary gates, ≤ 0.05 for secondary). Adjusting p to make a failing branch pass is a forbidden pipeline violation.
- **Null branch pairing is mandatory.** Every positive branch needs a matched null that fails.
- **All rescue operations expand internal coverage, never touch the external panel.**

## 8. Validation

A successful pipeline run must reproduce the following sanity signatures on scGPT + Tabula Sapiens immune before any new-model or new-system claim is trusted:

1. **H65 Phase-1 discovery:** trustworthiness **0.979**, geodesic-biological correlation **0.835**, blocked-permutation *p* = **0.0005**. Paired null branch (H66, same pipeline, different biological target) fails all gates.
2. **H65 internal holdout gates** (Table 1, internal panel d=10): Trust 0.979, Random 0.495, Donor 0.496, Branch 0.382. All pass the 0.20 / 0.80 thresholds.
3. **Strict non-overlap external validation** (Tabula Sapiens, d=10): Trust **0.985**, Random **0.594**, Donor **0.596**, Branch **0.453**. All pass.
4. **Expected-to-fail negative control** (lung panel, d=6): Trust **0.617** (fails), Random 0.236, Donor 0.687, Branch **0.058** (fails). If the lung panel passes, the pipeline's expected-to-fail control is broken.
5. **Multi-donor zero-shot transfer:** 165 anchors, 7 cohorts, Corr **0.500**, Corr_resid **0.476**, Trust **0.993**, Blocked p **0.0005**.
6. **Robust-V2 88-split benchmark (pseudotime Spearman):** H65 cell head **0.249**, best comparator (H65 hybrid) **0.132**. All 8 comparators (raw, PCA-10, SVD-10, CellTypist, scVI-10D, Scanpy-DPT, Palantir, scGPT-avg-pool+MLP) yield BH-*q* ≤ 2.7 × 10⁻⁷ in paired comparisons. The H65 cell head **beats all 8 comparators** on pseudotime.
7. **Subtype AUROCs:** CD4/CD8 **0.867**, mono/macro **0.951** (cell head, pooled donor means across 88 splits).
8. **Runtime:** full 12-split campaign ~**205 s** (extracted) vs ~**7,074 s** (scGPT avg-pool + 3-layer MLP). 34.5× speedup.
9. **Trainable parameters:** 5–170 per task (extracted) vs 172,000+ (scGPT-MLP baseline). ~1000× reduction.
10. **Head/layer screen:** L2H5 top-ranked with residualised correlation **0.634**, trustworthiness **0.985**. If L2H5 is not in the top 5, the per-head screen is mis-scoring.
11. **Compaction chain:** full drift 17.5 MB → compact top1 (L2H5) 5.9 MB with **0/8 BH losses**. Rank-64 surrogate 0.73 MB with 5/8 BH losses. If the single-head compact has significant losses, the compact operator is not correctly aggregating the single head.
12. **Factor-necessity core:** top 4 factors explain **66.2%** of total pooled ablation impact. If the concentration is < 50% or > 85%, the compact operator has qualitatively different factorisation than the source paper.
13. **Core sufficiency:** {f00, f01, f02, f03} alone gives branch balanced accuracy **0.572** (vs 0.820 intact), CD4/CD8 AUROC **0.699** (vs 0.846 intact). All 8 endpoints BH-significantly worse than intact model.
14. **Subset interactions:** best mono/macro recovery from `{f01, f03}` alone at **0.95× intact**. Best CD4/CD8 recovery from triples centred on f00. All 15 subsets BH-significantly worse than intact (no subset closes the gap).
15. **Sparse persistence:** same four-factor core (f00, f01, f02, f03) persists after hard-sparse pruning to 16 factors × 60 genes, explaining **68.9%** of sparse ablation impact.
16. **Second-manifold validation (H38, recommended):** internal trustworthiness **0.908**, geodesic correlation **0.828**, blocked *p* **0.0005**. Rescued external correlation **0.983** (v11). Cytokine/chemokine vs lipid signalling axis η² = **0.735**. If H38 cannot be reproduced even after internal-coverage rescue, the pipeline is over-fit to the hematopoietic case.

## 9. Known pitfalls

1. **No public reference implementation.** Unlike the other five Kendiukhov papers in this project, this paper does not cite a GitHub or Zenodo repository. Reconstructing the pipeline requires careful re-implementation from the paper text plus the supplementary methods (S1–S37). Expect subtle differences in LET objective weights, cell-trained head hyperparameters, and hybrid-variant sweep structure unless you have access to the author's private implementation.
2. **Null branch pairing is not optional.** Phase 3a requires a matched biologically-nonsensical target branch that fails the same gates. Without the null branch, Phase 1 cannot distinguish real biology from pipeline artefacts. Source paper: H65 succeeded, H66 (same pipeline, different target) failed. A Phase 1 run without a failing null branch should be treated as inconclusive.
3. **Strict non-overlap external panel requires ID-level exclusion.** It is **not sufficient** to use a different dataset — you must explicitly verify that no internal observation IDs appear in the external panel. Source paper checks this with a deterministic ID intersection. Failing this check invalidates the external validation.
4. **Expected-to-fail negative control is mandatory.** The lung-panel failure in the source paper (trust 0.617, branch 0.058) is a required diagnostic signature. If your negative control passes, the gates are too loose or your biological ruler is confounded with generic tissue structure.
5. **LET objective constants are under-documented.** The source paper states the LET equations but does not provide specific values for `α` (reconstruction weight) or `β` (distance scale). These likely need hyperparameter search for new systems. Start with `α = 0.1` and `β = 1/π` and tune via the strict non-overlap external validation score.
6. **Cell-trained head outperforms anchor-trained but at a cost.** Cell-level training sharpens subtype margins (CD4/CD8 AUC 0.744 → 0.886; mono/macro 0.759 → 0.937) but relaxes the compact branch-like arrangement (normalised within-stage spread 0.217 → 0.314). The hybrid topology-preserving variant offers a Pareto tradeoff. **No single head variant dominates all endpoints** — choose based on downstream application priorities.
7. **Latent axes are correlated, not independent.** Supplementary S31: mean absolute off-diagonal correlation 0.493 (anchor-trained) and 0.400 (cell-trained). **Do not interpret individual LET-10D axes as one-factor-per-axis** — treat the latent as a distributed code. The source paper uses a regularised temporal-axis search combining 6 standardised components, yielding ρ = 0.623 with target and ρ = 0.627 with HSC depth (permutation p = 5 × 10⁻⁴).
8. **T-cell subtype info is hidden in default 3D views.** CD4/CD8 AUROC is only 0.646 in displayed 3D but 0.822 in full 10D latent, with strongest single axis at dim 4 (0.845). A regularised T-cell separation lens (λ = 8) improves 3D CD4/CD8 from 0.835 to 0.901. **Always evaluate in the full latent space, not in visualisation projections.**
9. **The manifold is flat but ripples are real.** Best-fit 2D plane explains 99.25% of 3D variance, but residuals show statistically significant periodic ripple-like structure (external R² = 0.171, p = 0.0033). Persists in full 10D (5/8 residual axes significant externally after multiple-testing correction). **Biological interpretation of ripples remains open** — associations with curated erythroid gene-program set did not survive multiple-testing correction.
10. **Unsupervised branch recovery is a weakness.** When the extracted head's output is fed through Scanpy Leiden+PAGA or Palantir fate assignment, raw expression performs comparably or better on Adjusted Rand Index (ARI). The extracted head's advantage is concentrated in supervised classification and pseudotime ordering. **Do not claim unsupervised cluster recovery.**
11. **Low-rank compression below rank-64 degrades substantially.** Rank-64 (0.73 MB) is the viable low-rank boundary; rank-32 (0.37 MB) shows 6/8 BH losses; hard-sparse (0.12 MB) shows 7/8 losses. **The trade-off is not linear** — there is a qualitative drop between rank-64 and rank-32.
12. **Factor ablation uses fixed-probe protocol.** It measures what the current model *uses*, not whether task performance could be recovered after retraining. A factor with low leave-one-out impact may still be necessary for training a new model. **Do not interpret factor ablation as a statement about minimal sufficient circuits.**
13. **Donor reuse across Robust-V2 splits.** The 88-split benchmark has substantial donor reuse — 15 unique test donors across all splits (14 unique in the 40-split full-method subset). This limits the independence of split-level inference. Source paper audits the 40-split full-method subset to confirm the cell-head advantage persists, but **the 88-split numbers should not be treated as 88 independent trials**.
14. **Direct operator export requires vocabulary alignment.** The operator `A_{ℓ,h} ∈ ℝ^{d×d}` acts in the model's input gene vocabulary. For cross-model transfer (e.g., scGPT → Geneformer), you must map between the two models' gene vocabularies. This is non-trivial and is not addressed in the source paper — all experiments use a single model.
15. **Second-manifold rescue requires internal expansion, not gate relaxation.** H38 initial external failed (margin −0.074 vs PCA-2). The rescue path (v4 → v11, external correlation 0.885 → 0.983) worked by **expanding internal training coverage** (more donors, more cell types, more tissues, 100k cells), not by touching the external panel or loosening gates. If your rescue path involves touching the external panel, you have leaked and the external validation is invalid.
16. **The paper does not specify executor and reviewer prompt templates.** Phase 3 autonomous loop architecture is described but the specific prompts are not released. Use the prompt templates from `repos/topology-biomechinterp1/prompts/` (Claude) or `repos/topology-biomechinterp2/prompts/` (Codex 5.3 xhigh) as starting points and adapt them for manifold discovery (the topology screen prompts focus on topological hypotheses; manifold discovery needs prompts that emphasise operator-geometry-ruler combinations).
17. **"Extraction is not reducible to a small MLP on raw expression"** is a testable claim. For your new model and system, run Supplementary S22's direct-learnability baseline (1200 → 64 MLP on raw expression). If your extracted head does **not** BH-significantly beat this baseline on subtype-sensitive endpoints, the extraction is not providing value beyond what a tiny MLP could learn directly.
18. **Benchmark leadership is mixed across endpoints.** H65 cell head wins pseudotime-depth Spearman decisively (BH-q 2.7 × 10⁻⁷ on all 88 paired splits) but is BH-beaten on branch macro-F1, stage macro-F1, and mono/macro balanced accuracy. **Do not claim uniform superiority** — report the per-endpoint Pareto structure.

## 10. Quick-start for new-model and new-system application

To apply this pipeline to a new foundation model `NEWMODEL` and/or new biological system `NEWSYSTEM`:

1. **Pick the target foundation model.** Verify it exposes attention-value-projection matrices `A_{ℓ,h}` in an accessible format. Extract the full library of `n_layers × n_heads` matrices as NPY files (Phase 4 step 1).
2. **Commit to the biological system.** Choose a system that has a curated external ontology usable as the biological ruler. Hematopoiesis, intercellular communication, cell cycle, metabolic state, spatial organisation, immune activation, differentiation toward specific fates, etc. **Pick systems where the ground truth is independent of co-expression.**
3. **Build the biological ruler.** Assemble `d_target[i, j]` from the chosen ontology. Verify the three requirements (coherent, complete, independent of the foundation model). Persist as `d_target_internal.npy`.
4. **Build the internal anchor panel.** Apply (donor × tissue × cell-type) aggregation on your source dataset. Verify each anchor has sufficient cell count. Compute internal centroids.
5. **Build the strict non-overlap external panel.** From a *separate* cohort, apply the same aggregation. **Explicitly verify zero internal-ID overlap**. Compute external centroids and `d_target_external.npy`.
6. **Build the zero-shot multi-donor panel.** From a third cohort collection (7+ donors if possible). Compute anchors.
7. **Commit to the four quality gate thresholds** (trustworthiness ≥ 0.80, three holdout correlations ≥ 0.20, blocked permutation p ≤ 0.001). Write them into `reports/quality_gates_spec.json` before any experiments start.
8. **Run Phase 3a autonomous hypothesis sweep.** Adapt the executor + reviewer prompts from `repos/topology-biomechinterp2/prompts/` to emphasise operator-geometry-ruler combinations. **Require a paired null branch** — include a biologically-nonsensical target variant. Run 30–80 iterations or until a branch + matched failing null emerge.
9. **Transition to Phase 3b focused investigation.** Run equation-level LET replication, confidence intervals, bootstrap audits. Commit to a single positive branch for extraction.
10. **Run Phase 4 operator enumeration.** Materialise the pooled drift operator and (if computational resources permit) the full 96-head (or `n_layers × n_heads`) library.
11. **Run Phase 5 LET training for each head variant** (anchor, cell, hybrid). Evaluate against the four quality gates. Select the variant that passes all gates on both internal and external panels.
12. **Run Phase 6 probe training** with linear probes as the default and MLP2 as a targeted nonlinear follow-up for subtype-sensitive endpoints.
13. **Run Phase 7 external validation.** Apply the frozen head to the strict non-overlap panel. **All four gates must pass.** Report negative-control expected-to-fail panel results alongside.
14. **Run Phase 8 zero-shot multi-donor transfer.** Apply the frozen head to the separate cohort. Verify gates pass.
15. **Run Phase 9 head/layer attribution scan.** Score all `n_layers × n_heads` units. Identify the top-5 and the single best unit.
16. **Run Phase 10 compaction chain.** Single-head compact operator, rank-64 SVD surrogate, hard-sparse surrogate. **Expect 0/8 BH losses for the single-head compact** and 5/8 losses for rank-64.
17. **Run Phase 11 factor necessity ablation.** Identify the core factor set (typically 3–5 factors) and measure cumulative impact concentration. Run gene-program enrichment on top read/write genes.
18. **Run Phase 12 benchmark evaluation.** Compare against PCA/SVD, CellTypist, scVI, DPT, Palantir, raw expression MLP, and frozen foundation-model + MLP. Report paired Wilcoxon BH-*q* per endpoint. **Expect mixed leadership** — dominance on some endpoints, tie or loss on others.
19. **Run Phase 13 second-manifold validation if possible.** Apply the same pipeline to a distinct biological system in the same foundation model. This is the strongest evidence that the extraction methodology generalises.
20. **Cross-reference all pipelines in this project.** For any scientific claim about what `NEWMODEL` encodes:
    - **Attention pipeline** (`attention-grn-extraction-and-evaluation.md`): does attention encode regulatory logic or co-expression?
    - **Spectral geometry** (`residual-stream-spectral-geometry.md`): does SVD of the residual stream reveal biological axes?
    - **Topology screen** (`topology-geometry-141-hypotheses.md`): does topological/geometric structure survive strict null controls?
    - **SAE mega-pipeline** (`sparse-autoencoders/README.md`): do interpretable sparse features form biologically coherent causal circuits?
    - **This pipeline** (`manifold-discovery-extraction-compactification.md`): can a biological manifold be discovered, extracted as a standalone algorithm, and compactified to mechanistic interpretability?
21. **The pipelines are complementary, not redundant.** Each captures a different facet of model internals. A complete mechanistic audit of `NEWMODEL` reports all five verdicts. **This pipeline is unique in that it produces a deployable compact algorithm** — the other four pipelines produce diagnostic reports. Use this pipeline when the research goal is to surface a competitive biological algorithm from the foundation model's internals, not merely to characterise what the model has learned.
