# SAE Mechanistic Interpretability of Single-Cell Foundation Models — Mega-Pipeline Guide

This subfolder hosts the **sparse-autoencoder (SAE) mechanistic-interpretability mega-pipeline** — a three-stage workflow for dissecting single-cell foundation models (Geneformer, scGPT, and any other transformer-based biological foundation model) through their residual-stream activations, decomposed into interpretable features via sparse autoencoders.

The mega-pipeline is divided into three tightly-coupled subpipelines, each corresponding to one paper in Kendiukhov's SAE trilogy. The subpipelines are **sequentially composed**: stage 2 consumes artefacts produced by stage 1, and stage 3 consumes artefacts from stages 1 and 2. They can also be run independently once their inputs are available.

## Scope and central premise

Single-cell foundation models encode biology in a high-dimensional residual stream via **superposition** (Elhage et al. 2022) — more concepts than dimensions, packed into nearly-orthogonal directions. Sparse autoencoders decompose those dense activations into overcomplete dictionaries of monosemantic features, yielding interpretable "atoms" of biological knowledge that linear tools like SVD cannot recover (99.8% of SAE features are invisible to top-50 SVD in Geneformer).

Across the three stages, the mega-pipeline answers four questions:

1. **What does the model represent?** (stage 1, atlas construction) — train SAEs, annotate features against ontology databases, discover co-activation modules, locate cross-layer information highways, test causal specificity of individual features.
2. **How do features interact causally?** (stage 2, circuit tracing) — ablate source features, measure downstream effects on all other SAE features, build the directed computational graph, compare across models and training conditions.
3. **What does a complete map look like, and how deep is the redundancy?** (stage 3, exhaustive mapping) — trace every active feature (not just the well-annotated ones), test higher-order combinatorial interactions, establish causal control over cell-state trajectories through targeted feature amplification.
4. **Do these interpretable features encode genuine regulatory logic, or only co-expression structure?** (all three stages, tested via CRISPR perturbation data) — this is the unifying negative finding that connects the SAE trilogy to the author's companion attention-based interpretability work (see `../attention-grn-extraction-and-evaluation.md`). The answer across all three stages: **co-expression, not causal regulation** — only 6.2% of TRRUST TFs produce target-specific feature responses; circuit-predicted gene pairs achieve only 56.4% directional accuracy under CRISPRi; trajectory steering moves cell states in directionally-consistent but absolutely tiny amounts.

## Structure of this subfolder

```
pipelines/sparse-autoencoders/
├── README.md                                      ← this guide
├── 01-sae-atlas.md                                ← stage 1 spec
├── 02-causal-circuit-tracing.md                   ← stage 2 spec
├── 03-exhaustive-mapping-and-steering.md          ← stage 3 spec
└── 04-atlas-deployment.md                         ← guide to deploying the interactive web atlas
```

Each stage file (01–03) follows the standard template from the top-level `../../README.md` (overview, source, inputs, outputs, dependencies, methodology, parameters, validation, known pitfalls, quick-start). The code references sit inside the methodology steps, not in a separate section. `04-atlas-deployment.md` is a deployment guide for the web atlas, not an analysis stage, and has its own sections.

## The three subpipelines at a glance

### Stage 1 — [SAE Atlas](01-sae-atlas.md)

**Paper:** Kendiukhov 2026, arXiv:2603.02952 — *Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT.* 28 pages. [arXiv page](https://arxiv.org/abs/2603.02952).

**Code:** https://github.com/Biodyn-AI/bio-sae (pinned `9277f5d`) — see `../../repos/bio-sae/`. Interactive atlases at https://biodyn-ai.github.io/geneformer-atlas/ and https://biodyn-ai.github.io/scgpt-atlas/.

**What it produces:**
- Per-layer TopK SAEs for every transformer layer of the target model (Geneformer V2-316M: 18 layers, `d=1152 → d_SAE=4608`, `k=32`, trained on ~4M K562 token positions; scGPT whole-human: 12 layers, `d=512 → d_SAE=2048`, trained on ~3.56M Tabula Sapiens positions; 4× overcomplete in both cases).
- **Feature atlases:** 82,525 alive features (Geneformer) and 24,527 (scGPT) = **~107,000 features across 30 layers**.
- **Feature annotations** against GO BP, KEGG, Reactome, STRING (≥700), and TRRUST via Fisher's exact (BH-FDR < 0.05) on top-20 genes per feature — 29–59% annotation rate with U-shaped layer profiles.
- **Co-activation modules** via pointwise mutual information (PMI) on feature pairs at each layer, clustered by Leiden community detection at resolution 1.0 — 141 (Geneformer) / 76 (scGPT) biologically coherent modules.
- **Single-feature causal patching** results for a curated 50-feature set at layer 11 (median specificity 2.36×, 60% > 2×, 12% > 10×).
- **Cross-layer information highways** via cross-layer PMI at three layer pairs (L0→L5, L5→L11, L11→L17).
- **Perturbation response mapping** against the Replogle K562 CRISPRi screen (100 targets, 48 TRRUST TFs) — establishing the 6.2% regulatory-specificity null finding.
- **Multi-tissue SAE control** (K562 + Tabula Sapiens pooled) to disentangle SAE-training-data confounds from model-representation limits.
- **Interactive web atlases** (React/Vite/Plotly) deployed to GitHub Pages.

**Key role:** this is the **foundation**. Every downstream subpipeline requires the trained SAE weights, the feature atlas, and the annotation tables.

### Stage 2 — [Causal Circuit Tracing](02-causal-circuit-tracing.md)

**Paper:** Kendiukhov 2026, arXiv:2603.01752 — *Causal Circuit Tracing Reveals Distinct Computational Architectures in Single-Cell Foundation Models: Inhibitory Dominance, Biological Coherence, and Cross-Model Convergence.* 33 pages. [arXiv page](https://arxiv.org/abs/2603.01752).

**Code:** https://github.com/Biodyn-AI/bio-sae-circuits (pinned `4f6bfb4`) — see `../../repos/bio-sae-circuits/`. Companion atlases at https://github.com/Biodyn-AI/bio-sae.

**What it consumes from stage 1:** trained SAEs per layer, feature atlas with annotations, cached residual-stream activations.

**What it produces:**
- **Directed causal edges** between SAE features across layers via single-feature ablation. For each well-annotated source feature at a chosen layer: zero its SAE activation, replace in the residual stream, continue the forward pass, measure Cohen's *d* between clean and ablated activations at all downstream features. Retain edges with |*d*| > 0.5 and consistency > 0.7.
- **96,892 total causal edges** across four experimental conditions: K562 cells + K562-only SAEs (all 18 Geneformer layers), K562 cells + multi-tissue SAEs (4 Geneformer layers), Tabula Sapiens cells + multi-tissue SAEs through Geneformer (4 layers), Tabula Sapiens cells + scGPT SAEs (all 12 layers).
- **Circuit statistics:** mean |*d*|, median |*d*|, fraction of edges |*d*| > 1.0 and > 2.0, inhibitory fraction (65–89% across conditions — inhibitory dominance is a signature finding), shared-ontology fraction (53–69% — biological coherence).
- **Hub analysis:** features with highest out-degree (broadcast nodes, concentrated at early layers) vs highest in-degree (convergent integration, concentrated at late layers). Top Geneformer hub L0_F2905 (Golgi Organization) has out-degree 8,028.
- **Cross-model consensus domain pairs:** 1,142 biological process pairs present in at least one Geneformer condition AND in scGPT, representing 10.6× enrichment over random label permutation (1,000 permutations, *p* < 0.001). 303 high-confidence (|*d*| > 1.0 in both models).
- **Gene-level predictions:** 975,369 source-gene → target-gene predictions extracted from 47,418 annotated edges (top 10 rank-weighted genes per feature, filtered by evidence ≥2).
- **CRISPRi validation:** directional accuracy 56.4% (marginally above 50% chance); magnitude correlation Spearman ρ = 0.038.
- **Disease circuit mapping:** 11 disease-relevant gene-set categories; disease-associated domains 3.59× more likely to appear as cross-model consensus pairs.
- **PMI validation:** 91–95% target-feature overlap between causally-derived edges and the stage-1 PMI co-activation graph — confirming the two methods capture the same underlying structure.

**Key computational optimization:** a **clean forward cache** pre-computed once stores source-layer hidden states, SAE encodings, and downstream clean SAE activations for all cells — reducing circuit-tracing compute from ~20 days to 17.7 hours on Apple Silicon.

**Key role:** **the relational layer** — stage 2 turns the static feature atlas into a directed computational graph. The architecture-level comparison (Geneformer's cooperative 80/20 inhibitory dynamics vs scGPT's more balanced 65/35) only emerges at this stage.

### Stage 3 — [Exhaustive Mapping and Steering](03-exhaustive-mapping-and-steering.md)

**Paper:** Kendiukhov 2026, arXiv:2603.11940 — *Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control.* 18 pages. [arXiv page](https://arxiv.org/abs/2603.11940).

**Code:** https://github.com/Biodyn-AI/sae-biological-map (pinned `6406f07`) — see `../../repos/sae-biological-map/`.

**What it consumes from stages 1 and 2:** trained SAEs (stage 1), feature annotations (stage 1), the selective-tracing circuit graph (stage 2), the 14 "switch features" previously identified in stage 1 as tracking immune differentiation along Tabula Sapiens pseudotime.

**What it produces:**

1. **Exhaustive circuit graph at Geneformer layer 5.** Every active SAE feature (activation frequency ≥ 0.001; 4,065 features) is ablated in turn; downstream effects measured at L6, L11, L17. **Result: 1,393,850 significant edges — a 27× expansion over the selective-tracing graph from stage 2** (which covered only 30 annotated features and 52,116 edges).
   - Heavy-tailed hub distribution: 72 features (1.8%) have > 1,000 downstream edges each; 759 features (18.7%) > 500 edges.
   - **Annotation bias exposed:** 8 of the top-20 hubs (40%) are **unannotated** — selective tracing in prior work systematically missed the most computationally central features.
   - Signal attenuation: L6 receives 694K edges, L11 443K, L17 256K — 2.7-fold attenuation across 12 layers.

2. **Three-way combinatorial ablation** on 8 feature triplets across 4 pathways (vesicle transport, mitosis, metabolism, cross-pathway DDR × mitosis). 7 ablation conditions per triplet (A, B, C, AB, AC, BC, ABC) plus clean baseline, 200 K562 cells per condition. Metrics: three-way redundancy ratio `R_ABC = |d_ABC| / (|d_A|+|d_B|+|d_C|)` and higher-order interaction `I_ABC = d_ABC − d_AB − d_AC − d_BC + d_A + d_B + d_C`.
   - Single-feature redundancy ratio: **1.00** (trivially); pairwise: **0.74** (from stage 2 / selective tracing); three-way: **0.59**. Redundancy deepens monotonically with interaction order.
   - **Zero synergy at three orders:** only 0.14% of targets are superadditive (7 / 5,000), 94.1% subadditive, 5.8% additive. The model implements no higher-order logical gates.

3. **Trajectory-guided feature steering** on 14 switch features across layers 0, 5, 11, 17. In early-pseudotime cells (bottom 30% of diffusion pseudotime, Haghverdi et al. 2016), amplify SAE feature activation by α ∈ {2, 5} via:
   `h'_ℓ = h_ℓ + (α − 1) · a_f · d_f`
   where `a_f` is the feature's activation coefficient and `d_f` is its decoder direction. Propagate through the rest of the model. Measure **state shift**:
   `Δs = cos(z', g_late) − cos(z', g_early) − [cos(z, g_late) − cos(z, g_early)]`
   where `g_late` and `g_early` are gene signatures from the top and bottom pseudotime deciles.
   - **Layer position determines directionality:**
     - **L17 features (3/3 tested):** 100% of cells shift toward maturity (fraction positive = 1.0).
     - L11 features (4/4 tested): predominantly push *away* from maturity (mean fraction positive = 0.26).
     - L5 features (2/2 tested): mixed (0.43–0.91).
     - **L0 features (5/5 tested):** negative or mixed (0.08–0.58).
   - Absolute state-shift magnitudes small (mean |Δs| ≈ 0.001–0.003 at α = 5) but directionally **remarkably consistent**.

**Key role:** **the advanced deepening** — stage 3 breaks three specific limitations of stage 2 (annotation bias, pairwise-only interactions, observational trajectory correlation) and establishes causal feature control over cell differentiation.

## Data flow between stages

```
                                 ┌────────────────────────────────┐
                                 │   Model + unlabelled dataset   │
                                 │   (Geneformer V2-316M / scGPT  │
                                 │   + K562 Replogle / Tabula     │
                                 │   Sapiens)                     │
                                 └────────────────┬───────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────┐
                       │  Stage 1: SAE Atlas                            │
                       │  ──────────────────                            │
                       │  • Forward-hook activation extraction          │
                       │  • TopK SAE training (k=32, 4× overcomplete)   │
                       │  • Feature annotation vs GO/KEGG/Reactome/     │
                       │    STRING/TRRUST                               │
                       │  • PMI co-activation modules                   │
                       │  • Cross-layer information highways            │
                       │  • Single-feature causal patching              │
                       │  • Perturbation response mapping (CRISPRi)     │
                       │  • Interactive web atlas                       │
                       └────────────────┬───────────────────────────────┘
                                        │
                        produces:       │
                        - trained SAE   │
                          weights/layer │
                        - feature atlas │
                          (top genes,   │
                          annotations,  │
                          activations)  │
                        - PMI graph     │
                        - switch feats  │
                        - baseline 50-  │
                          feature causal│
                          patching set  │
                                        ▼
                       ┌────────────────────────────────────────────────┐
                       │  Stage 2: Causal Circuit Tracing               │
                       │  ─────────────────────────────                 │
                       │  • Select source features from stage-1 atlas  │
                       │    (30–120 well-annotated per source layer)   │
                       │  • Clean forward cache                         │
                       │  • Single-feature ablation → Cohen's d         │
                       │  • Four experimental conditions                │
                       │    (K562/K562, K562/Multi, TS/Multi, scGPT)    │
                       │  • Directed circuit graph construction         │
                       │  • Cross-model consensus (Geneformer ∩ scGPT)  │
                       │  • Gene-level predictions                      │
                       │  • CRISPRi directional validation              │
                       │  • Disease circuit mapping                     │
                       └────────────────┬───────────────────────────────┘
                                        │
                        produces:       │
                        - directed      │
                          circuit       │
                          graph         │
                          (96,892       │
                          edges)        │
                        - selective     │
                          tracing base- │
                          line for S3   │
                        - 975k gene-    │
                          pair          │
                          predictions   │
                                        ▼
                       ┌────────────────────────────────────────────────┐
                       │  Stage 3: Exhaustive Map + Combinatorial +     │
                       │  Trajectory Steering                           │
                       │  ─────────────────────────────────             │
                       │  • Exhaustive tracing: ALL active features     │
                       │    at one layer (no annotation filtering)     │
                       │  • Heavy-tailed hub distribution analysis      │
                       │  • Annotation-bias quantification              │
                       │  • Three-way combinatorial ablation            │
                       │    (8 feature triplets, 4 pathways)            │
                       │  • Trajectory-guided feature steering          │
                       │    (14 switch features from stage 1, α∈{2,5})  │
                       │  • Layer-dependent directionality              │
                       └────────────────────────────────────────────────┘
```

## Shared infrastructure

All three stages share the same foundational components. A general-purpose implementation should expose these as reusable modules so that stages 2 and 3 do not re-implement stage 1's extraction code.

- **Model adapter.** Forward-hook registration on each transformer layer's residual stream output, producing per-cell per-layer per-position activation tensors stored as memory-mapped NumPy arrays. Apple Silicon MPS acceleration is the reference backend; CUDA is straightforward.
- **TopK SAE trainer.** `k=32` sparsity, 4× overcomplete dictionary, Adam (lr 3×10⁻⁴), batch 4,096, 4 epochs, MSE loss after per-column mean subtraction. Stage 1 trains the atlas; stages 2 and 3 reuse those trained weights.
- **Feature annotator.** Top-20 genes per feature by mean activation magnitude; Fisher's exact (one-sided, BH FDR < 0.05) against GO BP, KEGG, Reactome, STRING (≥700), TRRUST.
- **CRISPR validation harness.** Replogle K562 genome-scale CRISPRi screen ([Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013)). Primary endpoint for stage 1 is "6.2% TF regulatory specificity"; for stage 2 is "56.4% directional accuracy on 599 overlap genes"; for stage 3 is the trajectory-steering maturation shift.
- **Reference databases.** GO BP, KEGG, Reactome, STRING v12.0 (≥700), TRRUST v2, Tabula Sapiens cell-type atlas, Replogle K562/RPE1 CRISPRi.
- **Interactive atlases.** Live deployments at https://biodyn-ai.github.io/geneformer-atlas/ and https://biodyn-ai.github.io/scgpt-atlas/ (six integrated views: Layer Explorer, Feature Detail Pages, Module Explorer, Cross-Layer Flow, Gene Search, Ontology Search).

## When to use each subpipeline

- **Stage 1 alone** is sufficient if the question is "what does my trained single-cell foundation model represent?" — it produces the feature atlas, module graph, cell-type enrichment, and interactive explorer. Stage 1 also produces the 6.2% regulatory-specificity verdict on its own.
- **Stage 1 + Stage 2** are needed if the question extends to "how do features interact causally, and do circuit predictions generalise across architectures?" — this is where cross-model consensus (1,142 pairs, 10.6× enrichment) and inhibitory-dominance architectural comparisons live.
- **Full trilogy** is needed if the question involves (a) whether selective tracing has a systematic annotation bias (answer: yes — 40% of top hubs are unannotated), (b) whether higher-order logical gates exist in the model (answer: no — zero synergy at three orders), or (c) causal control of cell-state trajectories (answer: L17 features universally drive maturation, L0 features push away, layer position determines directionality).

## Cross-references

- **The companion attention-based interpretability pipeline** is at `../attention-grn-extraction-and-evaluation.md` (paper arxiv 2602.17532). The attention pipeline and the SAE mega-pipeline are **complementary**: attention captures one view of model computation, the residual stream via SAE captures another. **Crucially, both arrive at the same negative verdict on causal regulatory logic** — attention edges add zero incremental value over gene-level baselines, and SAE features respond to perturbations without target specificity. A full mechanistic audit of any new single-cell foundation model should run *both* pipelines.
- **The residual-stream spectral-geometry pipeline** at `../residual-stream-spectral-geometry.md` (paper arxiv 2602.22247) is a **parallel dimensional-reduction view** of the same residual stream that SAEs decompose. SVD-based spectral geometry finds 7 interpretable axes; SAEs find ~5,000 features per layer (99.8% of which are invisible to SVD). The two are complementary at opposite ends of the decomposition spectrum.

## Current status

All three stage specifications in this subfolder are written in full: overview, source, inputs, outputs, dependencies, step-by-step methodology with exact parameters and code references, parameters, validation, known pitfalls and quick-start. The papers have been read in full, and the GitHub repositories have been cloned and pinned (see `../../repos/bio-sae/`, `../../repos/bio-sae-circuits/`, `../../repos/sae-biological-map/`). This guide sets out how the stages fit together. `04-atlas-deployment.md` describes how to build and deploy the interactive web atlas from the stage outputs.
