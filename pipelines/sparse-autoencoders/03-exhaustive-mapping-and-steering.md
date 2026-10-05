# Stage 3 — Exhaustive Circuit Mapping, Higher-Order Combinatorial Ablation, and Trajectory-Guided Feature Steering

## 1. Overview

This pipeline runs **three experiments** that break three specific methodological limitations of [Stage 2 causal circuit tracing](02-causal-circuit-tracing.md):

- **Experiment 1 — Exhaustive circuit tracing.** Stage 2 traced only 30 well-annotated source features per layer, selected by annotation quality — a sampling strategy that systematically excludes unannotated but computationally central features. This pipeline traces **every active SAE feature** at Geneformer layer 5 (**4,065 features** with activation frequency ≥ 0.001), yielding **1,393,850 significant causal edges** at downstream layers {L6, L11, L17} — a **27× expansion** over Stage 2's selective trace (52,116 edges on 30 features). The exhaustive graph reveals a **heavy-tailed hub distribution** in which **1.8% of features account for disproportionate connectivity** and, crucially, **40% of the top-20 hubs are completely unannotated** — selective tracing in prior work systematically missed the most computationally central features.
- **Experiment 2 — Higher-order combinatorial ablation.** Stage 2 tested only pairwise interactions (single-feature ablation) and established a pairwise redundancy ratio of 0.74. This pipeline extends combinatorial ablation to **three-way triplets** across 8 feature triplets in 4 biological pathways, testing whether higher-order logical gates (synergy) emerge. **Redundancy deepens monotonically** from single-feature (1.00) through pairwise (0.74) to three-way (0.59), and **zero synergy** is observed at third order (only 0.14% of triplet-target pairs are superadditive across 5,000 targets), confirming the model's circuit architecture is fundamentally subadditive with **no higher-order logical gates**.
- **Experiment 3 — Trajectory-guided feature steering.** Stage 1 (`01-sae-atlas.md`) identified 14 "switch features" across layers 0, 5, 11, 17 whose activation patterns track immune cell differentiation along Tabula Sapiens pseudotime — but that discovery was purely **observational**. This pipeline amplifies each switch feature's SAE activation by α ∈ {2, 5} in early-pseudotime cells via the **activation addition framework** (Turner et al. 2023, Li et al. 2024, Zou et al. 2023), propagates the modified residual stream through the remaining layers, and measures whether cell state shifts toward or away from maturity. **Layer position is a near-perfect predictor of steering directionality**: **L17 features (3/3 tested) push cells toward maturity in 100% of steered cells** (fraction positive = 1.0), while **L0 features (5/5) predominantly push away** (0.08–0.58, mean 0.34), transitioning from correlation to causal evidence for layer-dependent cell-state control.

All three experiments are run on **Geneformer V2-316M** using the trained TopK SAEs and feature annotations produced by Stage 1, and use the same **clean forward cache + Welford online effect-size accumulation** infrastructure introduced in Stage 2 (with the cache optimisation reducing exhaustive tracing compute from ~20 days to **17.7 hours** on a single Apple Silicon M2 Max).

**This is Stage 3 of the SAE mega-pipeline.** It consumes trained SAE weights and feature annotations from [Stage 1](01-sae-atlas.md) and the selective-tracing baseline from [Stage 2](02-causal-circuit-tracing.md). See `README.md` in this directory for the mega-pipeline guide. The three experiments establish:

1. **Annotation bias is a systematic confound** — the "map" drawn by annotation-biased sampling (Stage 2) misrepresents the territory. 40% of top-20 hubs are unannotated; computational importance and biological annotation are not correlated (top-20 hub annotation rate 60% vs overall 53.8%, not significantly enriched).
2. **Redundancy is a fundamental design principle** of Geneformer's circuit architecture. Monotonic deepening from 1.00 → 0.74 → 0.59 with zero synergy at third order resonates with neural network pruning findings (Frankle & Carbin 2019; Hinton 1986) — suggesting distributed representations (Hinton 1986; Olsson et al. 2022) where pathway information is copied redundantly across many features, each carrying a sufficient representation of its pathway's contribution to downstream computation.
3. **Layer position encodes differentiation directionality** as a learned, causal property of the network — not just a correlational pattern. This is consistent with the **progressive refinement hypothesis**: early and middle layers encode raw gene co-expression patterns (which when amplified maintain or disrupt progenitor-like states), while final layers encode processed cell-identity representations (whose amplification drives cells toward terminal differentiation). The gradient mirrors the biological differentiation hierarchy (Orkin & Zon 2008; Graf & Enver 2009).

Stage 3 is Geneformer-only in the source paper; replication in scGPT, scBERT, or other foundation models (and in protein foundation models like ESM, Rives et al. 2021) remains an open direction.

## 2. Source

- **Paper:** Kendiukhov, I. (2026). *Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control.* arXiv:2603.11940 [cs.LG], 12 Mar 2026. 18 pages.
- **Local PDF:** `../../references/2603.11940_Kendiukhov_exhaustive_circuit_mapping.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/sae-biological-map — pinned commit `6406f07cd4e13ffede2bc9981cda5579e615f2b9` (2026-03-12, "Fix figure readability issues"). Local clone at `../../repos/sae-biological-map/`.
- **Companion Stage 1 repo:** https://github.com/Biodyn-AI/bio-sae — provides trained SAE atlases, feature annotations, and the 14 switch features. See `01-sae-atlas.md`.
- **Companion Stage 2 repo:** https://github.com/Biodyn-AI/bio-sae-circuits — provides the selective-tracing baseline (30 features, pairwise redundancy ratio 0.74) that Experiments 1 and 2 extend. See `02-causal-circuit-tracing.md`.
- **Mega-pipeline guide:** `README.md` in this directory.
- **Key foundational references:**
  - **Network biology:** Barabási & Albert 1999, Albert & Barabási 2002, Barabási & Oltvai 2004 (scale-free networks); Jeong et al. 2001 (hub essentiality in biological protein networks); Gillis et al. 2014 (study bias in interaction databases).
  - **Mechanistic interpretability:** Olah et al. 2020 (circuits program); Wang et al. 2023 (IOI circuit); Conmy et al. 2023 (automated circuit discovery); Hanna et al. 2023 (greater-than in GPT-2); Marks et al. 2024 (sparse feature circuits); Bereska & Gavves 2024 (MechInterp review); Elhage et al. 2022 (superposition); Cunningham et al. 2023 / Bricken et al. 2023 / Templeton et al. 2024 (SAEs); Gao et al. 2024 (TopK SAEs); Makhzani & Frey 2013 (k-sparse autoencoders).
  - **Causal intervention:** Pearl 2009 (do-calculus); Vig et al. 2020 (causal mediation); Geiger et al. 2021 (causal abstractions); Meng et al. 2022 (locating/editing factual associations).
  - **Activation steering:** Turner et al. 2023 (activation addition); Li et al. 2024 (inference-time intervention); Zou et al. 2023 (representation engineering — top-down approach to AI transparency).
  - **Biological cell-state control:** Haghverdi et al. 2016 (diffusion pseudotime); Orkin & Zon 2008 (hematopoiesis); Graf & Enver 2009 (lineage reprogramming); Wolf et al. 2018 (scanpy).
  - **Effect size & statistics:** Cohen 1988; Welford 1962 (online variance).
  - **Redundancy and neural network design:** Frankle & Carbin 2019 (lottery ticket hypothesis); Hinton 1986 (distributed representations); He et al. 2016 (ResNet skip connections); Sharma et al. 2024 (layer-selective rank reduction).
  - **Biological pathway references:** Ciccia & Elledge 2010 (DDR); Jackson & Bartek 2009; Malumbres & Barbacid 2009 (CDKs / cell cycle); Musacchio & Salmon 2007 (spindle assembly); Chandel 2021 (mitochondria); Simons & Ikonen 2000 (cholesterol).
  - **Related Kendiukhov 2025 papers:** `01-sae-atlas.md` (bio-sae, SAE atlas + switch feature identification); `02-causal-circuit-tracing.md` (bio-sae-circuits, selective tracing + pairwise ablation); `../attention-grn-extraction-and-evaluation.md` (attention-based interpretability, companion study).

## 3. Inputs

### 3.1 Target model

The pipeline is written for **Geneformer V2-316M** (Theodoris et al. 2023) — 18 transformer layers, 18 attention heads per layer, d_model = 1,152, pre-trained on ~100 million single-cell transcriptomes. HuggingFace: `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M`. The pipeline is in principle model-agnostic — any transformer with a residual-stream API supporting mid-layer hidden-state replacement (Stage 2 Phase 0's adapter contract) can be substituted — but the source paper ran all three experiments only on Geneformer.

### 3.2 Upstream dependencies (mandatory)

**Stage 1 artefacts** (from `01-sae-atlas.md`):

1. **Trained TopK SAEs** at every measured layer. For this pipeline: layers 0, 5, 6, 11, 17 at minimum (L5 as source; L6, L11, L17 as downstream; L0/L5/L11/L17 for steering).
2. **Feature catalog** per layer (`feature_catalog.json`) — includes per-feature activation frequency on held-out positions, top-gene lists, activation magnitude. **Used to filter to the 4,065 active features at L5** (activation frequency ≥ 0.001).
3. **Feature annotations** per layer (`feature_annotations.json`) — biological pathway labels via GO BP 2024-01, KEGG Release 109, Reactome v87, STRING v12.0 (score ≥ 700), TRRUST v2, computed via Fisher's exact test (BH FDR < 0.05). **Used to label top-20 hubs in Phase 4 and to select feature triplets for combinatorial ablation in Phase 6.**
4. **The 14 "switch features"** (Phase 10) identified in paper 1 (bio-sae) as tracking immune cell differentiation along Tabula Sapiens pseudotime. Source paper hard-codes their IDs in `trajectory_steering.py` lines 79–94; they are:
   - **L0 (5 features):** F2061 Chromatin Organization (switch score *d* = 1.33), F25 Mitotic Spindle (1.13), F3567 unannotated (1.07), F1483 L-amino Acid Transport (1.07), F1295 Neurodegeneration (1.01)
   - **L5 (2 features):** F4349 Pre-mRNA Processing (0.95), F2016 unannotated (0.90)
   - **L11 (4 features):** F854 RNA Catabolic (1.15), F4350 unannotated (1.09), F2174 unannotated (1.01), F3750 unannotated (1.00)
   - **L17 (3 features):** F3730 Protein-RNA Complex Assembly (1.25), F1607 unannotated (1.09), F3567 unannotated (1.09)

**Stage 2 artefacts** (from `02-causal-circuit-tracing.md`):

1. **Selective-tracing baseline**: 52,116 edges on 30 features at L5 (K562/K562 condition), yielding the **pairwise redundancy ratio of 0.74** that Experiment 2 extends to three-way. Stored at `../../repos/bio-sae-circuits/results/circuit_analysis.json` or equivalent.
2. **Pairwise ablation summary** — used by Experiment 2 (`higher_order_ablation.py`'s `select_triplets()` function) to identify feature triplets that are structurally consistent (features at L0, L5, L11 with same pathway annotation).

### 3.3 Cell populations

| Population | Role in pipeline | Source |
|---|---|---|
| **K562 control cells** (Replogle 2022) | Experiment 1 exhaustive tracing (**20 cells per feature**, subset of the Stage 1 training set); Experiment 2 triplet ablation (**200 cells per condition**, same cells as Stage 2). | Figshare DOI 10.6084/m9.figshare.21452470 |
| **Tabula Sapiens immune cells** | Experiment 3 trajectory steering (**481 cells**, from the immune tissue subset of Tabula Sapiens). Pseudotime computed via `scanpy.tl.diffmap` → `scanpy.tl.dpt` (diffusion pseudotime, Haghverdi et al. 2016). | Tabula Sapiens Consortium 2022 / CZ CELLxGENE |

### 3.4 Biological annotation databases

Same five databases as Stages 1 and 2: GO BP 2024-01, KEGG R109, Reactome v87, STRING v12.0 (≥700), TRRUST v2. Used exclusively for feature annotation lookups in Experiments 1 (hub labelling) and 2 (triplet pathway grouping).

## 4. Outputs

### 4.1 Experiment 1 — Exhaustive circuit graph

1. **Per-feature circuit files** (resume-safe, one JSON per feature): `results/exhaustive_tracing/feature_F{id:04d}.json` containing, for each downstream layer ℓ ∈ {L6, L11, L17}, the **top-50 significant downstream edges** with Cohen's *d*, consistency, sign (inhibitory/excitatory).
2. **Aggregated summary:** `results/exhaustive_tracing/exhaustive_summary.json`:
   - **Total significant edges: 1,393,850** (|d| > 0.5 AND consistency > 0.7)
   - **Features traced: 4,065** (activation frequency ≥ 0.001 on L5)
   - **Edge count per feature:** mean **343**, median **284**, range **0–2,138**; 8 features with 0 significant edges
   - **Selective-vs-exhaustive comparison** (Table 1): selective 30 features / 52,116 edges / 1,737 avg/feature / 200 cells/feature / 12 downstream layers / 7.5 hr; exhaustive **4,065 features / 1,393,850 edges / 343 avg / 20 cells / 3 downstream / 17.7 hr**.
   - **Signal attenuation by downstream layer** (Figure 1C): **L6: 694,646 edges (49.8%); L11: 443,381 (31.8%); L17: 255,823 (18.4%)**. 2.7-fold attenuation from L6 to L17.
3. **Hub statistics:** top 20 features by total edge count (Table 2). **Top 5:**
   - **F898 — Negative Regulation of Gene Expression (GO:0010629) — 2,138 edges**
   - **F1762 — G2/M Transition of Mitotic Cell Cycle (GO:0000086) — 2,098 edges**
   - **F4518 — unannotated — 1,951 edges**
   - **F3792 — Transcription factors — 1,943 edges**
   - **F1057 — Herpes simplex infection — 1,930 edges**
   - Positions 6–20 continue: F3604 (unannotated 1,743), F2270 (unannotated 1,624), F4473 (Glycerophospholipid Biosynthesis 1,532), F678 (Organelle Organization 1,516), F3420 (unannotated 1,508), F3797 (rRNA Processing 1,500), F2520 (ncRNA Processing 1,499), F2797 (unannotated 1,460), F3370 (Golgi Organization 1,404), F1303 (Reg. mRNA Metabolism 1,387), F972 (unannotated 1,386), F2674 (Protein Modification 1,386), F3169 (unannotated 1,378), F1560 (DNA Damage Response 1,334), F4434 (unannotated 1,327).
4. **Annotation-bias report:** `results/exhaustive_tracing/annotation_bias.json`:
   - **Overall annotation rate: 53.8%** (of 4,065 features)
   - **Top-100 hub annotation rate: 49.0%**
   - **Top-20 hub annotation rate: 60.0%** — 8 of 20 (**40%**) are unannotated
   - **No significant enrichment of annotation in hubs over baseline** — computational importance is not correlated with having a biological pathway label.
5. **Heavy-tailed distribution statistics:**
   - **Features with > 1,000 edges: 72 (1.8%)** — the heavy-tail hub class.
   - **Features with > 500 edges: 759 (18.7%)**
   - Distribution is right-skewed, reminiscent of scale-free network architectures (Barabási & Albert 1999; Barabási & Oltvai 2004; Jeong et al. 2001).

### 4.2 Experiment 2 — Higher-order combinatorial ablation

6. **Per-triplet ablation results:** `results/higher_order_ablation/triplet_L0_F{iA}_x_L5_F{iB}_x_L11_F{iC}.json` — 8 files, one per triplet. Each contains per-condition (A, B, C, AB, AC, BC, ABC, clean baseline) Cohen's *d* vectors at L17, pairwise and three-way ratios, marginal contribution `C|AB`.
7. **Summary:** `results/higher_order_ablation/summary.json`:
   - **Pairwise redundancy ratio:** 0.74 (from Stage 2 baseline; mean across 20 feature pairs)
   - **Three-way redundancy ratio:** **0.59 for same-pathway triplets, 0.56 for cross-pathway** (Table 3)
   - **Mean marginal contribution `C|AB`:** near-zero for most triplets (median range **0.006–0.183**)
   - **Synergy classification across 8 triplets × ~5,000 target features with ≥1 significant effect:**
     - **Superadditive (synergistic): 7 instances (0.14%)**
     - Additive: 290 (5.8%)
     - Subadditive: 4,703 (94.1%)
8. **Table 3 per-triplet results** (same-pathway mean / cross-pathway mean):

| Pathway | Type | Pairwise ratio | Three-way ratio | Super. count | Marginal C\|AB |
|---|---|---|---|---|---|
| Vesicle transport #1 | Same | 0.73 | 0.607 | 0 | 0.072 |
| Vesicle transport #2 | Same | 0.72 | 0.593 | 0 | 0.045 |
| Vesicle transport #3 | Same | 0.74 | 0.619 | 0 | 0.013 |
| Mitosis #1 | Same | 0.78 | 0.585 | 0 | 0.184 |
| Mitosis #2 | Same | 0.78 | 0.594 | 0 | 0.162 |
| Metabolism #1 | Same | 0.73 | 0.548 | 2 | 0.006 |
| Metabolism #2 | Same | 0.77 | 0.669 | 1 | 0.012 |
| DDR × Mitosis | Cross | 0.74 | 0.556 | 4 | 0.172 |
| **Mean (same)** | | **0.75** | **0.594** | **0.4** | **0.071** |
| **Cross-pathway** | | **0.74** | **0.556** | **4** | **0.172** |

### 4.3 Experiment 3 — Trajectory-guided feature steering

9. **Per-feature steering results:** `results/trajectory_steering/steering_F{idx}_L{layer}.json` — 14 files, one per switch feature. Each contains: layer, feature_idx, label, per-α (α=2 and α=5) results including state_shift_score, n_active_cells, mean logit delta vector, top upregulated/downregulated genes.
10. **Cached state signatures:** `results/trajectory_steering/state_signatures.npz` — early and late pseudotime logit signatures (`early_sig`, `late_sig` numpy arrays) computed once from all Tabula Sapiens immune cells.
11. **Summary:** `results/trajectory_steering/summary.json`:
    - **L17 features (3/3 tested):** fraction positive = **1.0** (100% of steered cells pushed toward maturity)
    - **L11 features (4/4 tested):** fraction positive = **0.00–0.44, mean 0.26** (predominantly push away from maturity)
    - **L5 features (2/2 tested):** fraction positive = **0.43–0.91** (mixed)
    - **L0 features (5/5 tested):** fraction positive = **0.08–0.58, mean 0.34** (predominantly push away)
    - Absolute |Δs| magnitudes: mean ≈ **0.001–0.003 at α=5** (small) but directionally consistent.
12. **Table 4 per-feature detail** (state shift at α=5, fraction positive, top upregulated gene):

| Layer | Feature | Switch score | Label | Δs (×10⁻³) | Frac pos | Top gene ↑ |
|---|---|---|---|---|---|---|
| L0 | F2061 | 1.33 | Chromatin Org. | +0.31 | 0.54 | CD24 |
| L0 | F25 | 1.13 | Mitotic Spindle | −1.03 | 0.29 | CDCA2 |
| L0 | F3567 | 1.07 | unannotated | −1.11 | 0.31 | CLEC4E |
| **L0** | **F1483** | **1.07** | **L-amino Acid** | **−2.82** | **0.08** | **CCL3** |
| L0 | F1295 | 1.01 | Neurodegen. | −1.67 | 0.47 | MIIP |
| **L5** | **F4349** | **0.95** | **Pre-mRNA Proc.** | **+2.79** | **0.91** | **ADAMTS2** |
| L5 | F2016 | 0.90 | unannotated | −0.80 | 0.43 | MCM4 |
| L11 | F854 | 1.15 | RNA Catabolic | −0.17 | 0.44 | UPK1B |
| L11 | F4350 | 1.09 | unannotated | −1.45 | **0.00** | PPP1R14A |
| L11 | F2174 | 1.01 | unannotated | −0.89 | 0.28 | SEMA4F |
| L11 | F3750 | 1.00 | unannotated | −1.39 | 0.32 | C15orf61 |
| **L17** | **F3730** | **1.25** | **Protein-RNA Complex** | **+0.96** | **1.00** | **ZYG11B** |
| **L17** | **F1607** | **1.09** | **unannotated** | **+1.25** | **1.00** | **POLR1E** |
| **L17** | **F3567** | **1.09** | **unannotated** | **+2.46** | **1.00** | **KLF9** |

### 4.4 Final report

13. **Cross-experiment synthesis** (`results/stage3_summary.json`): the three headline findings and their implications:
    - **Annotation bias confound:** 40% of hubs unannotated, selective sampling systematically misrepresents circuit structure.
    - **Redundancy deepens monotonically without synergy:** 1.00 → 0.74 → 0.59, zero synergy at third order.
    - **Layer position determines differentiation directionality:** L17 → maturity (1.0), L0 → away (0.34), mid-layers (L5, L11) mixed.

## 5. Dependencies

- **Python ≥ 3.10**, PyTorch **2.1** (Apple Silicon MPS backend), NumPy 1.26.4, scanpy (Wolf et al. 2018), h5py, scikit-learn, transformers (for Geneformer loader).
- **Stage 1 outputs:** trained SAEs + annotations (see §3.2). Required for all three experiments.
- **Stage 2 outputs:** selective-tracing baseline (see §3.2). Required for Experiments 1 (comparison baseline) and 2 (triplet selection from pairwise results).
- **Replogle K562 CRISPRi** h5ad (Figshare DOI 10.6084/m9.figshare.21452470). Required for K562 cell loading in Experiments 1 and 2.
- **Tabula Sapiens immune** h5ad (CZ CELLxGENE). Required for Experiment 3 pseudotime computation and steering cell pool.
- **Compute:** source paper ran on Apple Silicon M2 Max (38-core GPU, 96 GB unified memory). Total wall-clock: **~26.3 hours** (17.7 exhaustive + 6.5 combinatorial + 0.035 steering). CUDA users can expect ~2× speedup.
- **Environment files:** `../../repos/sae-biological-map/requirements.txt`.
- **Scripts:**
  - `../../repos/sae-biological-map/src/exhaustive_feature_tracing.py` (756 lines) — Experiment 1
  - `../../repos/sae-biological-map/src/higher_order_ablation.py` (1,057 lines) — Experiment 2
  - `../../repos/sae-biological-map/src/trajectory_steering.py` (732 lines) — Experiment 3
  - `../../repos/sae-biological-map/src/sae_model.py` (246 lines) — shared `TopKSAE` class

## 6. Methodology

The pipeline runs in thirteen phases across three experiments. Phases 0–1 are shared setup; Phases 2–5 are Experiment 1; Phases 6–8 are Experiment 2; Phases 9–12 are Experiment 3; Phase 13 is cross-experiment synthesis.

---

### Phase 0 — Input preparation: load Stage 1 SAEs and Stage 2 baseline

**Purpose:** verify upstream dependencies and load shared artefacts.

**Steps:**

1. **Verify Stage 1 completion** for Geneformer V2-316M at every layer required by Experiments 1–3:
   - **L5** (Experiment 1 source layer; Experiment 2 triplet mid-layer)
   - **L6, L11, L17** (Experiment 1 downstream layers; also Experiment 2 target layer L17)
   - **L0** (Experiment 2 triplet source; Experiment 3 steering layer)
   - Confirm each has `sae_final.pt`, `feature_catalog.json`, `feature_annotations.json`.
2. **Verify Stage 2 completion.** Confirm the selective-tracing baseline at L5 exists (`results/circuit_analysis.json`). Store the pairwise redundancy ratio (0.74) and edge count (52,116) as comparison baselines for Experiments 1 and 2.
3. **Load the `TopKSAE` class** (`src/sae_model.py`) and instantiate an `SAECache` (lazy-loading cache keyed by layer) to avoid reloading SAEs across iterations.
4. **Load the Geneformer V2-316M model** in eval mode on the target device (MPS primary, CUDA alternative). Register the adapter for mid-layer hidden-state replacement (see Stage 2 Phase 0 adapter contract).
5. **Fix random seeds** (PyTorch, NumPy, Python random) for reproducibility.

**Code reference:** `src/sae_model.py` → `TopKSAE.load()`; shared `SAECache` patterns identical to Stage 2.

---

### Phase 1 — Activation frequency filter (for Experiment 1)

**Purpose:** reduce the full 4,608-feature L5 dictionary to the 4,065 features with non-zero activation frequency, eliminating dead features that would produce zero significant edges and waste compute.

**Steps:**

1. **Load `feature_catalog.json`** for L5 from Stage 1 output. For each feature, read its activation frequency on the 100K held-out position sample.
2. **Apply threshold:** retain features with `activation_frequency ≥ 0.001` (≈ 1 activation per 1,000 cells). For Geneformer V2-316M L5 this yields **exactly 4,065 active features** (of 4,608 total = 543 dead or near-dead).
3. **Persist the active feature list** as `active_features_L5.json` for Phase 2 iteration.

**Rationale:** 0.001 is a conservative threshold that eliminates provably dead features while retaining even rare features. A stricter threshold (e.g. 0.01) would drop ~1,000 more features; the source paper uses 0.001 to avoid prematurely excluding features that are rare but functionally important.

**Code reference:** `src/exhaustive_feature_tracing.py` → `load_feature_catalog(source_layer=5)`.

---

### Phase 2 — Exhaustive circuit tracing (Experiment 1)

**Purpose:** for each of the 4,065 active features at L5, ablate the feature and measure the Cohen's *d* effect size on every downstream SAE feature at layers 6, 11, and 17 — yielding the complete causal circuit graph for L5.

**Steps:**

1. **Load 20 K562 cells** from the Replogle control population. Tokenise via Geneformer rank-value encoding.
2. **Build the clean forward cache** (same optimisation as Stage 2 Phase 2, adapted for fewer cells and fewer downstream layers):
   - Run one clean forward pass per cell, capturing `h^(L5)_clean`, `h^(L6)_clean`, `h^(L11)_clean`, `h^(L17)_clean`.
   - Encode each clean hidden state through its layer's SAE: `z^(ℓ)_clean`, `h̃^(ℓ)_clean = SAE_dec(z^(ℓ)_clean)`.
   - Store in CPU memory, keyed by `(cell_idx, layer)`.
3. **For each of 4,065 active source features at L5:**
   - For each of 20 cells:
     - Retrieve clean `h^(L5)_clean` from cache.
     - Encode through L5 SAE: `z = TopK(W_enc · (h^(L5)_clean − μ))`.
     - Zero the source feature: `z[source_feat_idx] = 0`.
     - Reconstruct: `h̃_ablated = W_dec · z + μ`.
     - Compute ablation delta: `δ = h̃_ablated − h̃^(L5)_clean` (the reconstruction of z *with* the feature).
     - **Apply the delta to the original hidden state:** `h^(L5)_abl = h^(L5)_clean + δ` (preserves the ~20% of variance not captured by the SAE; see Stage 2 Pitfall #3).
     - Register a hook at L5 that overwrites the forward output with `h^(L5)_abl`, then run the forward pass from the input. The hook intercepts at L5 and all subsequent layers receive the ablated state.
     - Capture `h^(ℓ)_abl` for ℓ ∈ {L6, L11, L17}.
     - For each downstream layer ℓ, encode through its SAE: `z^(ℓ)_abl`. Compute per-feature delta `Δ^(ℓ)_j = z^(ℓ)_abl[j] − z^(ℓ)_clean[j]` for every downstream feature *j*.
   - Accumulate the per-cell `Δ^(ℓ)_j` vectors across all 20 cells via the **Welford online algorithm** (Welford 1962), maintaining running mean and variance without storing per-cell data.
4. **After processing all 20 cells per source feature**, compute:
   - **Cohen's *d*** per downstream feature: `d^(ℓ)_j = Δ̄^(ℓ)_j / s^(ℓ)_j` where `s` is the pooled standard deviation.
   - **Consistency**: fraction of cells where the delta has the same sign as the mean.
5. **Retain significant edges** with **|d| > 0.5 AND consistency > 0.7** (same thresholds as Stage 2 for comparability). Store the top 50 strongest effects per downstream layer per source feature.
6. **Resume-safe checkpointing:** persist per-feature JSON (`feature_F{id:04d}.json`) after each feature completes. A crash-and-resume will skip features whose JSON already exists.

**Key parameters vs Stage 2 selective tracing:**
- **Cells per feature:** 20 (vs 200 in Stage 2) — compute tradeoff. With 4,065 features, using 200 cells/feature would cost 200/20 × 17.7 hr = 177 hr ≈ 7.4 days. The reduction to 20 cells reduces statistical power (see Limitation 1) but enables exhaustive coverage.
- **Downstream layers measured:** 3 (L6, L11, L17) vs 12 in Stage 2 — intermediate layers L7–L10 and L12–L16 are not captured. This is a deliberate tradeoff: Stage 2 cares about the full attenuation curve across all layers; Stage 3 cares about complete feature-space coverage at representative depths (adjacent L6, middle L11, final L17).
- **Computational wall-clock:** **17.7 hours** on Apple Silicon M2 Max (Stage 2 K562/K562 was 7.5 hr; the 2.4× wall-clock increase reflects the ~135× increase in source features partially offset by 10× fewer cells and 4× fewer downstream layers).

**Code reference:** `src/exhaustive_feature_tracing.py`:
- `load_feature_catalog(source_layer=5)` — loads active features
- `WelfordAccumulator(n_dims)` — `update()`, `finalize() → (cohens_d, consistency)`
- `precompute_clean_cache(all_tokens, model, device, source_layer=5, downstream_layers=[6,11,17], n_cells=20)` — builds the cache
- `make_hook(modified_hidden, device)` — hook factory for ablated forward pass
- `trace_feature(feature_idx, source_layer, clean_cache, ...)` — per-feature tracing loop
- `main()` — orchestrates iteration with resume-safe checkpointing

**Critical sanity check:** the first feature traced should produce **at least 50 significant edges** at L6 (the adjacent downstream layer). If it produces zero, either the ablation delta is not being applied correctly (see Stage 2 Pitfall #3) or the hook is registered at the wrong layer.

---

### Phase 3 — Heavy-tailed hub analysis

**Purpose:** characterise the hub distribution of the exhaustive circuit graph and identify the top 20 features by total edge count.

**Steps:**

1. **Aggregate per-feature edge counts** across the three downstream layers. For each source feature, sum the number of significant edges at L6, L11, L17 to get its total out-degree.
2. **Rank features by total edge count**. Compute the distribution statistics:
   - **Mean: 343 edges/feature**
   - **Median: 284**
   - **Range: 0–2,138**
   - **8 features with 0 significant edges** (features that are alive in Phase 1 activation-frequency sense but exert no measurable downstream influence).
   - **Right-skewed distribution** — mean > median indicates heavy tail.
3. **Heavy-tail quantification:**
   - **72 features (1.8%) have > 1,000 edges** — the hub class.
   - **759 features (18.7%) exceed 500 edges.**
   - Distribution is reminiscent of scale-free network architecture (Barabási & Albert 1999; Barabási & Oltvai 2004; Jeong et al. 2001), where a small fraction of nodes (hubs) carry most of the connectivity.
4. **Report the top-20 hub features** (Table 2). The top 5:
   - **F898 — Negative Regulation of Gene Expression (GO:0010629) — 2,138 edges**
   - **F1762 — G2/M Transition of Mitotic Cell Cycle (GO:0000086) — 2,098 edges** (Malumbres & Barbacid 2009)
   - **F4518 — unannotated — 1,951 edges**
   - **F3792 — Transcription factors — 1,943 edges**
   - **F1057 — Herpes simplex infection — 1,930 edges**
5. **Compare to Stage 2 selective trace.** Stage 2's 30 well-annotated features at L5 yielded 1,737 avg edges/feature (over 12 downstream layers). This pipeline's 4,065 features yield 343 avg/feature (over 3 downstream layers). When normalising by downstream layer count (3 vs 12), the per-feature-per-downstream-layer rate is comparable (~114 vs ~145). **The 27× expansion in edge count is driven by feature coverage, not by per-feature density.**

**Interpretation:** the heavy-tailed hub architecture has important robustness implications. A model with a few thousand features but whose connectivity is concentrated in ~70 hubs may be **unexpectedly fragile to targeted perturbation of those hubs, while robust to ablation of the long tail** — parallel to biological protein interaction networks where hub proteins are disproportionately essential (Jeong et al. 2001).

**Code reference:** `src/exhaustive_feature_tracing.py` → aggregation in `main()` after per-feature tracing loop.

---

### Phase 4 — Annotation-bias quantification

**Purpose:** test whether the top computational hubs have the same annotation rate as the full feature set, or whether annotation is systematically enriched (or depleted) among hubs. **This is the phase that reveals the selective-tracing bias.**

**Steps:**

1. **Compute the overall annotation rate** across all 4,065 features using Phase 0's `feature_annotations.json`. A feature is "annotated" if it has ≥ 1 significant ontology enrichment (GO BP, KEGG, Reactome, STRING ≥ 700, or TRRUST, at BH FDR < 0.05).
   - **Overall annotation rate: 53.8%** (2,187 / 4,065 features annotated).
2. **Compute the top-100 hub annotation rate:** the fraction of the top-100 features by edge count that are annotated.
   - **Top-100 annotation rate: 49.0%** (49 of 100 — *lower* than the overall rate).
3. **Compute the top-20 hub annotation rate:**
   - **Top-20 annotation rate: 60.0%** (12 of 20 annotated, 8 unannotated).
4. **Significance test:** enrichment of annotation in the top-20 / top-100 hub sets over the baseline 53.8% via Fisher's exact test.
   - **No significant enrichment** — the fraction of annotated features among the top-100 and top-20 hubs does not significantly exceed the baseline rate. **Annotation status is not predictive of computational centrality.**
5. **Identify the unannotated top-20 hubs**: F4518, F3604, F2270, F3420, F2797, F972, F3169, F4434. These 8 features would have been **entirely excluded** from Stage 2's selective trace (which required ontology annotation for the annotation-quality scoring in Stage 2 Phase 1), yet they occupy 4 of the top-10 positions in the exhaustive hub ranking.

**Interpretation (Section 3.1 of paper):** this result parallels the "study bias" phenomenon in network biology (Gillis et al. 2014), where the tendency to study well-known genes distorts interaction databases. Interpretability studies that select features based on annotation status (Cunningham et al. 2023; Templeton et al. 2024; Stage 1 and Stage 2 of this mega-pipeline) **necessarily over-represent well-characterised biology and under-represent novel or poorly characterised computational roles**. Exhaustive approaches that do not filter by annotation are essential for unbiased circuit discovery, as demonstrated for language model circuits (Wang et al. 2023; Conmy et al. 2023; Hanna et al. 2023).

**Code reference:** `src/exhaustive_feature_tracing.py` → annotation analysis in the aggregation step.

---

### Phase 5 — Signal attenuation across layers

**Purpose:** characterise how causal effects from a single source-layer ablation attenuate as they propagate through subsequent layers.

**Steps:**

1. **For each downstream layer** ℓ ∈ {L6, L11, L17}, count the total number of significant edges across all 4,065 source features.
2. **Expected attenuation** (Figure 1C):
   - **L6: 694,646 edges (49.8% of total)**
   - **L11: 443,381 edges (31.8%)**
   - **L17: 255,823 edges (18.4%)**
   - **2.7-fold attenuation from L6 to L17** (from 694,646 to 255,823).
3. **Interpretation (Section 3.4):** this attenuation is consistent with the progressive dilution of single-feature perturbations across intervening computational layers. At the same time, **effects persist across the full 12-layer span** — even 12 layers downstream, there are still a quarter of a million significant edges, indicating early-layer features contain foundational biological information that propagates to the output. This is the same phenomenon Stage 2 observed with L0 effects reaching L17 at ~58 edges/feature.
4. **Scale-of-the-computational-graph implication (Section 3.4):** 1,393,850 edges from a **single source layer** at just 3 downstream layers represents a **lower bound** on the model's circuit complexity. With 18 layers and ~4,000 active features per layer, the **complete circuit graph** (all 18 source layers × all 18 downstream layers × 4,000 features) would contain on the order of **tens of millions of edges**. Full circuit analysis of foundation models will require substantial computational investment and novel algorithmic approaches.

**Code reference:** `src/exhaustive_feature_tracing.py` → per-downstream-layer edge counting.

---

### Phase 6 — Three-way combinatorial ablation (Experiment 2)

**Purpose:** test whether higher-order logical gates (synergy at third order) emerge in Geneformer's circuit architecture, or whether the subadditive (redundant) pattern observed at pairwise order (Stage 2) deepens further.

**Steps:**

1. **Select 8 feature triplets** via `select_triplets(n_triplets=8)` in `higher_order_ablation.py`. Each triplet contains one feature from each of **layers 0, 5, 11** with same or related GO/pathway annotations:
   - **Vesicle transport** (3 triplets) — features in secretory pathway / membrane trafficking
   - **Mitosis** (2 triplets) — cell division, chromosome segregation
   - **Metabolism** (2 triplets) — energy / biosynthesis
   - **Cross-pathway DDR × Mitosis** (1 triplet) — DNA damage response features from one pathway × mitotic features from another
2. **For each triplet (A at L0, B at L5, C at L11)**, run **8 ablation conditions** measured at **L17 SAE features**:
   - **Clean baseline** (no ablation)
   - **Single-feature ablations:** A only, B only, C only
   - **Pairwise ablations:** A+B, A+C, B+C
   - **Three-way ablation:** A+B+C
3. **200 K562 cells per condition** (same cell pool as Stage 2). Measure Cohen's *d* at every L17 SAE feature via Welford online accumulation.
4. **Compute per-target-feature effects** for each condition. The effect vectors `d_A`, `d_B`, `d_C`, `d_AB`, `d_AC`, `d_BC`, `d_ABC` ∈ ℝ^4608 (L17 has 4,608 SAE features).
5. **Resume capability:** partial results per triplet checkpointed at the condition level.

**Compute budget:** 8 triplets × 8 conditions × 200 cells × ablated forward pass time ≈ **6.5 hours** on Apple Silicon M2 Max.

**Code reference:** `src/higher_order_ablation.py`:
- `select_triplets(n_triplets=8)` — triplet selection from Stage 2 pairwise summary
- `ablate_single(source_layer, feature_idx, ...)` — A, B, C conditions
- `ablate_two(layer_A, fi_A, layer_B, fi_B, ...)` — AB, AC, BC conditions
- `ablate_three(layer_A, fi_A, layer_B, fi_B, layer_C, fi_C, ...)` — ABC condition
- `run_experiment(triplets, ...)` — orchestration

---

### Phase 7 — Redundancy ratio and interaction term analysis

**Purpose:** quantify how much the three-feature ablation effect differs from the sum of its parts, and whether it is subadditive (redundant) or superadditive (synergistic).

**Steps:**

1. **Three-way redundancy ratio** per target feature:
   `R_ABC = |d_ABC| / (|d_A| + |d_B| + |d_C|)`
   - R_ABC = 1.0 → additive
   - R_ABC < 1.0 → subadditive (redundant; simultaneously ablating 3 features doesn't add up to the sum of individual effects)
   - R_ABC > 1.0 → superadditive (synergistic; features together produce larger-than-additive effect)
2. **Higher-order interaction term** via inclusion-exclusion (Möbius decomposition):
   `I_ABC = d_ABC − d_AB − d_AC − d_BC + d_A + d_B + d_C`
   This isolates the portion of the three-way effect not explained by single and pairwise effects alone.
3. **Marginal contribution of third feature given first two are ablated:**
   `C|AB = d_ABC − d_AB` (change in effect when adding the third ablation on top of the pair)
4. **Per-triplet results (Table 3):**

| Triplet | Type | Pairwise ratio | Three-way ratio | Super count | Marginal C\|AB |
|---|---|---|---|---|---|
| Vesicle Same #1 | Same | 0.73 | **0.607** | 0 | 0.072 |
| Vesicle Same #2 | Same | 0.72 | 0.593 | 0 | 0.045 |
| Vesicle Same #3 | Same | 0.74 | 0.619 | 0 | 0.013 |
| Mitosis Same #1 | Same | 0.78 | 0.585 | 0 | 0.184 |
| Mitosis Same #2 | Same | 0.78 | 0.594 | 0 | 0.162 |
| Metabolism Same #1 | Same | 0.73 | 0.548 | 2 | 0.006 |
| Metabolism Same #2 | Same | 0.77 | 0.669 | 1 | 0.012 |
| DDR × Mitosis | Cross | 0.74 | 0.556 | 4 | 0.172 |
| **Mean (same)** | | **0.75** | **0.594** | **0.4** | **0.071** |
| **Mean (cross)** | | **0.74** | **0.556** | **4** | **0.172** |

5. **Monotonic deepening of redundancy** (Section 2.2.1):
   - **Single-feature redundancy ratio: 1.00** (trivially — one feature ablated, effect is the feature's own effect)
   - **Pairwise redundancy ratio: 0.74** (from Stage 2 pairwise ablation baseline)
   - **Three-way redundancy ratio (same-pathway): 0.59** (from this pipeline)
   - **Three-way redundancy ratio (cross-pathway): 0.56**
   - **Redundancy deepens monotonically with interaction order** (Figure 3A).
6. **Marginal contribution near zero:** `C|AB` has median values 0.006–0.183 across triplets, indicating that **two features from a pathway already capture most of the unique pathway information, and a third feature adds negligible additional disruption**.

**Interpretation (Section 3.2):** monotonic deepening establishes Geneformer's circuit architecture is **fundamentally subadditive**. Resonates with findings in neural network pruning, where large fractions of parameters can be removed without performance loss (Frankle & Carbin 2019), suggesting redundancy is a **general property of overparameterized models**. Consistent with distributed representations (Hinton 1986; Olsson et al. 2022) where information is spread across many features but each carries a redundant copy.

**Code reference:** `src/higher_order_ablation.py` → `run_analysis(results, out_dir)` for ratio computation.

---

### Phase 8 — Zero-synergy verification

**Purpose:** classify the individual target-feature responses across all 8 triplets by whether they are subadditive, additive, or superadditive — confirming that Geneformer has no higher-order logical gates.

**Steps:**

1. **For each triplet** and each **target feature at L17** with at least one significant effect (|d| > 0.5 in any of the 7 ablation conditions):
   - Compute `|d_ABC|` (three-way effect magnitude)
   - Compute `|d_A| + |d_B| + |d_C|` (sum of individual effect magnitudes)
   - **Classify:**
     - **Superadditive (synergistic):** `|d_ABC| > |d_A| + |d_B| + |d_C|`
     - **Additive:** `|d_ABC| ≈ |d_A| + |d_B| + |d_C|` (within ±10%)
     - **Subadditive (redundant):** `|d_ABC| < |d_A| + |d_B| + |d_C|`
2. **Aggregate across 8 triplets × ~5,000 target features**:
   - **Superadditive: 7 instances (0.14%)**
   - **Additive: 290 (5.8%)**
   - **Subadditive: 4,703 (94.1%)**
3. **Interpretation (Section 2.2.2):** the near-complete absence of superadditive targets at third order **extends the zero-synergy finding from pairwise (Stage 2) to three-way interactions**. The model's circuit architecture **contains no higher-order logical gates** that require the simultaneous presence of multiple pathway features to activate. In biological systems, conjunctive regulation — where multiple signals must converge to activate a response — is a common motif in signalling cascades (Alon 2007); the lack of such synergy in Geneformer suggests the model does NOT implement higher-order logical operations across same-pathway features. Instead, each feature independently captures a sufficient representation of its pathway's contribution to downstream computation.
4. **Cross-pathway triplet note:** the DDR × Mitosis cross-pathway triplet yielded **4 superadditive instances** (the highest of any triplet). While still a tiny fraction of targets, this hints that genuine synergy may emerge at pathway boundaries — an open direction for future work with more cross-pathway triplets.

**Code reference:** `src/higher_order_ablation.py` → superadditivity classification in `run_analysis()`.

---

### Phase 9 — Diffusion pseudotime computation (Experiment 3 setup)

**Purpose:** order the 481 Tabula Sapiens immune cells along a pseudotemporal axis (from progenitor-like to mature) so that subsequent steering experiments can identify "early" and "late" cell populations.

**Steps:**

1. **Load 481 Tabula Sapiens immune cells** via h5py (hematopoietic tissues — spleen, thymus, bone marrow, lymph node, PBMC — filtered to myeloid and lymphoid cell types).
2. **Tokenise** each cell via the Geneformer token dictionary (rank-value encoding).
3. **Compute diffusion pseudotime** via scanpy (Wolf et al. 2018):
   - `sc.pp.neighbors(adata)` — k-NN graph in gene expression space
   - `sc.tl.diffmap(adata)` — diffusion map embedding
   - `sc.tl.dpt(adata)` — diffusion pseudotime (Haghverdi et al. 2016)
4. **Stratify cells into pseudotime tertiles:**
   - **Early (bottom 30%):** 144 cells — progenitor-like
   - **Middle (30–66.7%):** ~160 cells
   - **Late (top 33.3%):** 160 cells — mature / terminally differentiated
5. **Compute gene expression signatures** for early and late populations:
   - `g_early` = mean expression vector over the bottom pseudotime decile (10% of cells)
   - `g_late` = mean expression vector over the top pseudotime decile (10% of cells)
   - These serve as anchors for the state-shift metric in Phase 11.
6. **Cache pseudotime and signatures:** `results/trajectory_steering/state_signatures.npz` (numpy compressed format) with keys `early_sig`, `late_sig`, `pseudotime`, `early_cell_indices`.

**Code reference:** `src/trajectory_steering.py`:
- `load_cells_and_pseudotime(n_cells=481)` — loads h5ad, tokenises, computes or caches pseudotime
- `compute_state_signatures(all_tokens, pseudotime, model, device)` — computes `early_sig`, `late_sig`

---

### Phase 10 — Switch feature selection (from Stage 1)

**Purpose:** select the 14 SAE features whose activation patterns correlate with immune differentiation pseudotime, identified in Stage 1 (`01-sae-atlas.md`).

**Steps:**

1. **Hard-code the 14 switch feature IDs** from the Stage 1 output (`trajectory_steering.py` lines 79–94):
   - **L0 (5 features):** F2061 (Chromatin Organization, *d*=1.33), F25 (Mitotic Spindle, 1.13), F3567 (unannotated, 1.07), F1483 (L-amino Acid Transport, 1.07), F1295 (Neurodegeneration, 1.01)
   - **L5 (2 features):** F4349 (Pre-mRNA Processing, 0.95), F2016 (unannotated, 0.90)
   - **L11 (4 features):** F854 (RNA Catabolic Process, 1.15), F4350 (unannotated, 1.09), F2174 (unannotated, 1.01), F3750 (unannotated, 1.00)
   - **L17 (3 features):** F3730 (Protein-RNA Complex Assembly, 1.25), F1607 (unannotated, 1.09), F3567 (unannotated, 1.09)
2. **The "switch score" *d*** was computed in Stage 1 as the Cohen's *d* between early-pseudotime and late-pseudotime cell populations for that feature's activation distribution — measuring how much the feature "switches" on/off across pseudotime. Only features with switch *d* > 0.9 were retained, yielding 14 total across the 4 measured layers.
3. **Note the selection caveat:** the switch features are hard-coded by ID in the source paper's script for reproducibility. For a new model, you would rerun Stage 1's switch-feature identification on your model's SAE atlas + pseudotime.

**Code reference:** `src/trajectory_steering.py:79–94` → `SWITCH_FEATURES` list.

---

### Phase 11 — Trajectory-guided feature steering (Experiment 3)

**Purpose:** causally test whether amplifying a switch feature's activation in early-pseudotime cells pushes the cell state toward or away from maturity.

**Steering formula (activation addition framework, Turner et al. 2023; Li et al. 2024; Zou et al. 2023):**

For a feature *f* at layer *ℓ* with activation coefficient `a_f` and decoder direction `d_f`, and a steering multiplier α:

```
h'_ℓ = h_ℓ + (α − 1) · a_f · d_f
```

At α = 1, the modification is zero (baseline). At α = 2, the feature's contribution to the residual stream is doubled. At α = 5, it is 5× the original — an aggressive intervention.

**State-shift metric:**
```
Δs = cos(z', g_late) − cos(z', g_early) − [cos(z, g_late) − cos(z, g_early)]
```
where `z` is the clean logit vector, `z'` is the steered logit vector, and `g_late`, `g_early` are the pseudotime-decile gene signatures from Phase 9.

**Δs > 0 ⟺ the steered state shifts toward the late-pseudotime (mature) signature relative to the early signature**.

**Steps:**

1. **For each of 14 switch features:**
   a. Load the SAE at the feature's layer.
   b. Identify **early-pseudotime cells where the feature is active** (within the bottom 30% of pseudotime AND the feature appears in the cell's TopK=32 activations at some position).
   c. For each active cell and each α ∈ {2, 5}:
      - Run a clean forward pass through the model, capturing `h_ℓ^(clean)` and the final logits `z^(clean)`.
      - Encode `h_ℓ^(clean)` through the SAE to get the sparse activation vector and identify the feature's activation coefficient `a_f` and decoder column `d_f = W_dec[:, f]`.
      - Compute the steered hidden state: `h_ℓ^(steered) = h_ℓ^(clean) + (α − 1) · a_f · d_f`.
      - Register a hook at layer ℓ that replaces the forward output with `h_ℓ^(steered)`, then re-run the forward pass from the input. Capture the steered logits `z^(steered)`.
      - Compute `Δs` for this cell.
   d. **Aggregate across cells:**
      - **fraction_positive** = fraction of cells with Δs > 0 (toward maturity)
      - **mean_delta_s** = mean Δs
      - **top_upregulated_gene** = gene with the largest positive logit delta across cells
2. **Persist per-feature results:** `results/trajectory_steering/steering_F{id}_L{layer}.json` containing `(layer, feature_idx, label, alpha_results)` for α ∈ {2, 5}.
3. **Compute budget:** **2.1 minutes** total across all 14 features and both α values on Apple Silicon M2 Max (steering is extremely fast — no ablation loop, one forward pass per cell per α).

**Code reference:** `src/trajectory_steering.py` → `steer_feature(feature, alphas, early_cells, ...)` lines 400–500 (approx).

---

### Phase 12 — Layer-dependent directionality analysis

**Purpose:** test whether layer position is a predictor of steering directionality.

**Steps:**

1. **Group the 14 features by layer** and compute **per-layer mean fraction-positive**:
   - **L17 (n=3):** **1.00 (100% of cells push toward maturity)** — unanimous
   - **L11 (n=4):** **0.26** (0.00 to 0.44) — predominantly push **away** from maturity
   - **L5 (n=2):** **0.67** (0.43 to 0.91) — mixed, trending toward maturity
   - **L0 (n=5):** **0.34** (0.08 to 0.58) — predominantly push **away** from maturity
2. **Key observation:** **layer position is a near-perfect predictor of directionality at the extremes** (L0 away, L17 toward) but is **not strictly monotonic** — L11 shows *lower* mean fraction positive than L0. The L17 endpoint is unambiguous (3/3 features unanimously toward maturity across ALL cells).
3. **Effect size magnitudes (Section 2.3.2):** absolute |Δs| values are small — mean ≈ **0.001–0.003 at α=5**, reflecting the modest impact of amplifying a single feature among ~4,600 in an overcomplete representation. **However, the directionality is remarkably consistent**: L17 features achieve perfect positive fraction (1.0) despite the tiny absolute shifts — suggesting the directional signal is **robust even when the magnitude is small**.
4. **Gene-level coherence** (Section 2.3.3; Figure 5). Top upregulated genes under steering are biologically interpretable:
   - **L5 F4349 (Pre-mRNA Processing, +2.79 ×10⁻³, frac pos = 0.91):** top upregulated **ADAMTS2** (extracellular matrix metalloprotease) and developmental regulators — pushing toward mature tissue-remodeling programs.
   - **L0 F1483 (L-amino Acid Transport, −2.82 ×10⁻³, frac pos = 0.08):** top upregulated **CCL3** (inflammatory chemokine), downregulates structural proteins — pushing toward inflammatory progenitor programs.
   - **L17 F3730 (Protein-RNA Complex Assembly, +0.96, frac pos = 1.00):** upregulates **ZYG11B**, **POLR1E** — transcriptional regulators associated with terminal differentiation.
   - **L17 F3567 (unannotated, +2.46, frac pos = 1.00):** upregulates **KLF9** — a well-known terminal differentiation transcription factor.
5. **Interpretation (Section 3.3):** causal demonstration that **L17 features universally push cells toward maturity** while **earlier-layer features predominantly push them away** establishes a **functional distinction between late- and early/mid-layer representations**. Consistent with the **progressive refinement hypothesis**: early and middle layers encode raw gene co-expression patterns (which when amplified maintain or disrupt progenitor-like states), while final layers encode processed cell-identity representations (whose amplification drives cells toward terminal differentiation). Gradient mirrors the biological differentiation hierarchy (Orkin & Zon 2008; Graf & Enver 2009), where early transcriptional programs maintain multipotency while late programs commit cells to specific fates. **The transformer model spontaneously learns this hierarchical organisation without explicit supervision on differentiation stage**, suggesting the layer-wise processing naturally decomposes cell state into a progression from raw features to commitment signals.

**Code reference:** `src/trajectory_steering.py` → `run_analysis(results, out_dir)` for layer-grouped statistics.

---

### Phase 13 — Cross-experiment synthesis and final reporting

**Purpose:** consolidate the three headline findings into a single Stage 3 summary report and position them within the broader SAE mega-pipeline narrative.

**Steps:**

1. **Emit `results/stage3_summary.json`** containing:
   - Experiment 1: 4,065 features traced / 1,393,850 edges / 27× expansion over selective / 1.8% hubs / 40% of top-20 hubs unannotated / 17.7 hr compute.
   - Experiment 2: 8 triplets / 4 pathways / pairwise 0.74 → three-way 0.59 / zero synergy (0.14%) / 6.5 hr compute.
   - Experiment 3: 14 switch features / L17 100% toward maturity / L0 34% toward / 2.1 min compute.
2. **Write the cross-experiment narrative** (Section 3.6 of paper):
   - **Exhaustive circuit graph is dominated by unannotated hub features**, organised in a heavy-tailed distribution, characterised by deep redundancy without synergy. Annotation-based filtering systematically misrepresents this territory.
   - **Causal evidence establishes that late-layer (L17) features uniquely and universally drive cell state toward maturity**, while earlier layers do not. Layer position is a functional, not merely architectural, property.
   - **Geneformer's internal representations encode biologically meaningful functional specialisation implemented through massively redundant, hub-dominated circuits** — a substantially different picture from what selective, annotation-biased tracing suggested.
3. **Cross-reference to the companion pipelines:**
   - **Stage 1 (`01-sae-atlas.md`)** provided the SAEs, annotations, and switch features. Stage 3 revisits Stage 1's 40–55% unannotated feature pool and demonstrates that **unannotated features are not biologically irrelevant** — they occupy the most computationally central positions in the circuit graph.
   - **Stage 2 (`02-causal-circuit-tracing.md`)** provided the selective-tracing baseline and the pairwise redundancy ratio. Stage 3 extends selective tracing by 27×, extends pairwise ablation to three-way, and converts observational switch-feature analysis into causal steering.
   - **Companion attention pipeline (`../attention-grn-extraction-and-evaluation.md`)** established at the attention-head level that Geneformer captures co-expression rather than causal regulation. Stage 3's layer-dependent differentiation control provides the **first affirmative causal evidence** for a specific computational structure in Geneformer — not about TF→target regulation, but about **cell-state trajectory control** — and demonstrates that mechanistic interpretability of biological foundation models can produce positive findings, not only null results.

**Code reference:** no single script; this is cross-experiment aggregation driven by `paper/generate_*.py` figure generation scripts.

---

## 7. Parameters

| Parameter | Default | Range | Used in |
|---|---|---|---|
| **Experiment 1** (exhaustive tracing) | | | |
| `source_layer` | **5** (Geneformer) | any layer with trained SAE | Phase 2 |
| `downstream_layers` | **{6, 11, 17}** | subset of layers with trained SAEs | Phase 2 |
| `n_cells_per_feature` | **20** | 10–200 (↑ = more power, ↑ compute) | Phase 2 |
| `activation_frequency_threshold` | **0.001** | 0.0001–0.01 | Phase 1 |
| `cohens_d_threshold` | **\|d\| > 0.5** (Cohen "medium") | 0.3–1.0 | Phase 2 |
| `consistency_threshold` | **> 0.7** | 0.6–0.8 | Phase 2 |
| `top_edges_per_feature` | **50** (retained per downstream layer) | 10–200 | Phase 2 |
| `clean_forward_cache` | **on** (non-negotiable) | — | Phase 2 |
| `welford_online` | **on** (non-negotiable) | — | Phase 2 |
| **Experiment 2** (combinatorial ablation) | | | |
| `n_triplets` | **8** | 4–32 | Phase 6 |
| `triplet_layers` | **{L0, L5, L11}** | configurable | Phase 6 |
| `measurement_layer` | **L17** | any downstream layer | Phase 6 |
| `ablation_conditions` | **{A, B, C, AB, AC, BC, ABC, clean}** (7+1) | — | Phase 6 |
| `n_cells_per_condition` | **200** | 100–500 | Phase 6 |
| `triplet_pathway_grouping` | **same-pathway + cross-pathway** | — | Phase 6 |
| `redundancy_ratio_formula` | `\|d_ABC\| / (\|d_A\|+\|d_B\|+\|d_C\|)` | — | Phase 7 |
| `superadditivity_threshold` | `\|d_ABC\| > \|d_A\|+\|d_B\|+\|d_C\|` | — | Phase 8 |
| **Experiment 3** (trajectory steering) | | | |
| `n_cells_tabula_sapiens` | **481** | 100–2,000 | Phase 9 |
| `pseudotime_method` | **scanpy diffusion pseudotime** (Haghverdi 2016) | — | Phase 9 |
| `early_cells_tertile` | **bottom 30%** | 10–50% | Phase 9 |
| `gene_signature_decile` | **top/bottom 10% of pseudotime** | — | Phase 9 |
| `n_switch_features` | **14** (hard-coded from Stage 1) | configurable | Phase 10 |
| `switch_feature_layers` | **{L0: 5, L5: 2, L11: 4, L17: 3}** | configurable | Phase 10 |
| `switch_d_threshold` | **> 0.9** (switch score from Stage 1) | 0.5–2.0 | Phase 10 |
| `steering_alpha` | **α ∈ {2, 5}** | 1.5–10 | Phase 11 |
| `steering_formula` | `h'_ℓ = h_ℓ + (α−1) · a_f · d_f` | — | Phase 11 |
| `state_shift_metric` | cosine sim to `g_late` vs `g_early` | — | Phase 11 |

## 8. Validation

A successful Stage 3 run must reproduce the following sanity signatures on Geneformer V2-316M before any new-model claim is trusted:

1. **Active feature count at L5** (Phase 1): exactly **4,065** features with activation frequency ≥ 0.001 (out of 4,608 total). Tolerance ±5. If < 3,500 or > 4,500, the Stage 1 SAE training may be degenerate or the activation frequency computation is inconsistent with Stage 1.
2. **Total significant edges, exhaustive trace** (Phase 2): **1,393,850** edges across L6 + L11 + L17. Tolerance ±2% (~28K edges). If < 1M, threshold enforcement is too strict or the clean cache is not being used correctly (check wall-clock — should be ~17.7 hr not > 2 days).
3. **Per-feature edge distribution** (Phase 3): mean **343**, median **284**, range 0–2,138. **8 features** with 0 significant edges. Right-skewed (mean > median).
4. **Signal attenuation** (Phase 5): **L6: 694,646 (49.8%); L11: 443,381 (31.8%); L17: 255,823 (18.4%)**. 2.7× attenuation from L6 to L17. Tolerance ±3%.
5. **Heavy-tailed hub distribution** (Phase 3):
   - **72 features (1.8%) with > 1,000 edges**
   - **759 features (18.7%) with > 500 edges**
   - Top hub **F898 Negative Regulation of Gene Expression at 2,138 edges**; second **F1762 G2/M Transition at 2,098 edges**.
6. **Annotation bias** (Phase 4):
   - **Overall annotation rate 53.8%**
   - **Top-100 hub rate 49.0%** (non-enriched)
   - **Top-20 hub rate 60.0% — 8 of 20 (40%) unannotated**
   - Fisher's exact enrichment test: **not significant** for either top-20 or top-100 vs baseline. This is the required diagnostic signature.
7. **Three-way redundancy ratio** (Phase 7): **0.59 for same-pathway triplets, 0.56 for cross-pathway**, vs pairwise 0.74 baseline. Monotonic deepening 1.00 → 0.74 → 0.59. Tolerance ±0.05. If three-way > pairwise, the monotonic-deepening claim fails and either (a) the ablation is not applying all three interventions simultaneously, or (b) you have selected triplets with different redundancy properties than the paper's.
8. **Zero synergy verification** (Phase 8): across 8 triplets × ~5,000 target features, **superadditive 7 instances (0.14%), additive 290 (5.8%), subadditive 4,703 (94.1%)**. Tolerance ±2 percentage points on subadditive fraction. If superadditive > 5%, either the synergy classification threshold is too loose or the triplets have been accidentally selected as cross-pathway.
9. **Cross-pathway superadditive count** (Phase 8): DDR × Mitosis triplet yields **4 superadditive instances**, the highest of any triplet. If this signature is absent, cross-pathway vs same-pathway distinction is not being computed correctly.
10. **Pseudotime ordering** (Phase 9): on 481 Tabula Sapiens immune cells, early-tertile (bottom 30%) should contain ~144 cells; late-tertile (top 33.3%) ~160 cells. Gene expression signatures from top/bottom deciles should have non-trivial cosine similarity < 1 (otherwise pseudotime ordering is degenerate).
11. **Trajectory steering directionality by layer** (Phase 12):
    - **L17: 3/3 features with fraction positive = 1.0**. Non-negotiable signature. If any L17 feature has frac pos < 0.9, either the steering coefficient formula is wrong (off by α vs α−1 factor), or the state-shift metric is computed against swapped early/late signatures.
    - **L0: 5/5 features with fraction positive in 0.08–0.58, mean 0.34**. Predominantly push away.
    - **L11: 4/4 features with fraction positive in 0.00–0.44, mean 0.26**. L11 should NOT be monotonic with L0 — L11 mean (0.26) is lower than L0 mean (0.34).
    - **L5: 2/2 features with fraction positive in 0.43–0.91**. Mixed.
12. **Gene-level coherence signatures** (Phase 12 step 4):
    - **L5 F4349** steering upregulates **ADAMTS2**.
    - **L0 F1483** steering upregulates **CCL3**.
    - **L17 F3730** steering upregulates **ZYG11B**.
    - **L17 F3567** steering upregulates **KLF9**.
    - If any of these top-gene signatures is completely wrong, the decoder direction is being applied incorrectly or the logit-to-gene mapping is swapped.
13. **Absolute steering magnitude** (Phase 12 step 3): mean **|Δs| ≈ 0.001–0.003 at α=5**. If > 0.1, α is being applied as an absolute multiplier to `h_ℓ` instead of the corrective `(α−1) · a_f · d_f` form — re-check Phase 11 formula.
14. **Compute budget** (Phase 13 totals): **17.7 hours Experiment 1 + 6.5 hours Experiment 2 + 2.1 minutes Experiment 3 = ~24.2 hours + 2 min** on Apple Silicon M2 Max. Source paper reports "total compute ~26.3 hours" including setup, caching, and intermediate checkpointing. Tolerance ±30% for different hardware.

## 9. Known pitfalls

1. **Activation frequency filter is critical for compute budget.** Without the `freq ≥ 0.001` filter, you trace all 4,608 features, ~15% of which are dead. Tracing dead features wastes compute and can introduce numerical instabilities in Welford accumulation (division by near-zero variance). Always verify the active count matches ~4,065.
2. **The clean forward cache optimisation is non-negotiable** (same as Stage 2 Pitfall #1). A naive implementation (one forward pass per feature × cell) takes ~20 days for exhaustive tracing. Verify the outer loop is over cells and the cache is built before the feature loop.
3. **20 cells per feature reduces statistical power** (Limitation 1). Stage 2 uses 200 cells per feature and obtains mean |d| = 1.05; this pipeline uses 20 cells and obtains mean edges per feature of 343 (lower than Stage 2's 1,737 normalised per downstream layer). The 10× cell reduction results in wider variance in per-pair Cohen's *d*, potentially missing weak effects. If statistical sensitivity is critical, increase `n_cells_per_feature` to 50–100 at the cost of 2.5–5× wall-clock.
4. **Only 3 downstream layers measured** (Limitation 2). The pipeline measures L6, L11, L17 — skipping L7, L8, L9, L10, L12, L13, L14, L15, L16. Intermediate-layer effects are not captured. For full coverage, use Stage 2's 12-downstream-layer approach and accept the smaller feature subset.
5. **The ablation-delta-to-original-hidden-state bug** (same as Stage 2 Pitfall #3). Phase 2 step 3 must apply `δ = h̃_ablated − h̃_original` to `h^(L5)_clean`, not overwrite with `h̃_ablated` directly. If you overwrite, you lose the ~20% of variance the SAE does not explain, and the causal effect measurements become dominated by reconstruction error rather than feature ablation.
6. **Three-way ablation triplet selection bias.** The 8 triplets in the source paper are selected from pairwise ablation results in `higher_order_ablation.py`'s `select_triplets()` function, which groups features by pathway and requires one from each of L0/L5/L11 with matching GO annotation. **This selection procedure inherently finds redundant triplets** — same-pathway features are by construction likely to encode redundant information. Cross-pathway triplets (like the DDR × Mitosis example) are more likely to show synergy but are harder to identify systematically. The 8-triplet sample is a limited proof of concept; broader pathway coverage (e.g., 32–64 triplets across ≥10 pathways) would strengthen the zero-synergy claim.
7. **Steering α = 5 is aggressive** and can produce out-of-distribution hidden states that the downstream layers have not encountered during training, potentially yielding artefactual effects. The paper reports both α = 2 and α = 5; the α = 5 numbers are used in Tables/Figures because they produce larger state shifts. For a new model, start with α = 2 and scale up only if the frac-positive signal at α = 2 is too weak to distinguish from noise.
8. **Diffusion pseudotime is sensitive to preprocessing.** scanpy's `tl.dpt` depends on the k-NN graph built by `pp.neighbors`, which in turn depends on PCA dimensionality and metric choice. The source paper uses scanpy defaults, but for new tissues or datasets, the pseudotime ordering can flip (late ↔ early) depending on graph parameters. **Always validate the pseudotime orientation** by checking that known mature/differentiated cells (e.g., memory T cells, plasma cells) have larger pseudotime values than progenitor cells (e.g., hematopoietic stem cells).
9. **Switch feature IDs are specific to the Stage 1 training run.** The 14 switch features in `trajectory_steering.py` lines 79–94 are **hard-coded** for the paper's specific Stage 1 SAE training seed. If you retrain Stage 1 SAEs (even with the same hyperparameters, due to TopK non-determinism at ties), the feature ID ordering will change and the hard-coded IDs will no longer correspond to the same features. **For a new model or re-training, rerun the switch-feature identification procedure** (compute per-feature Cohen's *d* between early and late pseudotime populations, retain features with *d* > 0.9 across layers 0, 5, 11, 17).
10. **Small absolute state-shift magnitudes** (~0.001–0.003) mean the effect is real but biologically subtle. It is **NOT** evidence that steering actively moves cells through a differentiation program in a clinically meaningful sense — it is evidence that the model encodes a latent direction of "mature vs early" that can be probed via amplification. Do not overstate the biological significance. The claim is about the model's internal representation, not about effective cell-fate engineering.
11. **Geneformer-only scope** (Limitation 5). All three experiments are run only on Geneformer V2-316M. Replication in scGPT, scBERT, scFoundation, or protein models (ESM) is an open direction. The clean forward cache infrastructure (Stage 2) supports both Geneformer and scGPT APIs; the exhaustive tracing and combinatorial ablation can in principle run on scGPT with minor adaptations to the ablation hook (use `model.transformer_encoder.layers[L]` hook registration per Stage 2's scGPT path).
12. **The three-layer triplet structure (L0, L5, L11)** is specific to Geneformer's 18-layer architecture. For a model with a different depth (e.g., scGPT's 12 layers, or scFoundation's deeper stack), you must choose source layers that span early/mid/late while leaving enough downstream layers for measurement. A reasonable heuristic: pick layers at ~0%, ~25%, ~60% of depth, with measurement at the last layer before output.
13. **Annotation rate in hubs is not significantly enriched** (Phase 4). This is a **positive finding** — it demonstrates annotation-selection bias in selective tracing. But it means the top-20 hubs CANNOT be re-annotated via Fisher's exact test on the same ontology databases; those databases simply do not cover the biology encoded in those features. Further characterisation of unannotated hub features requires more flexible approaches (Stage 1 Phase 10's Jaccard guilt-by-association, or new ontology sources like MSigDB Hallmarks, or direct perturbation validation).

## 10. Quick-start for new-model evaluation

To apply Stage 3 to a new single-cell foundation model `NEWMODEL` with Stage 1 and Stage 2 completed:

1. **Verify Stage 1 and Stage 2 completion.** For `NEWMODEL`:
   - Stage 1 SAEs trained at all layers planned for Stage 3 measurement.
   - Stage 1 feature catalog with activation frequencies computed on ≥ 100K held-out positions.
   - Stage 1 feature annotations against GO BP / KEGG / Reactome / STRING / TRRUST.
   - Stage 1 switch features identified via per-feature Cohen's *d* between pseudotime extremes (only required for Experiment 3).
   - Stage 2 selective-tracing baseline at the layer you plan to use as source for Experiment 1.
   - Stage 2 pairwise ablation summary for triplet selection in Experiment 2.
2. **Choose the source layer for Experiment 1.** The paper uses Geneformer L5 because it sits at approximately one-third of network depth, has rich mid-layer biological representations, and allows 3 downstream layer measurements (L6, L11, L17) spanning adjacent-mid-final. For `NEWMODEL` with `n_layers` layers, analogue is layer `round(n_layers / 3)`.
3. **Run Phase 1 activation-frequency filter.** Count active features at the chosen source layer. If you have far fewer than ~4,000 active features, your Stage 1 SAE may be over-sparse (too many dead features) — investigate before proceeding.
4. **Smoke-test Phase 2 on 10 features × 5 cells.** Verify the clean forward cache is built, Welford accumulation produces reasonable Cohen's *d* magnitudes (|d| in 0.1–3.0 range for actual effects), and significant edges emerge at adjacent-layer downstream. Extrapolate to full-run compute budget; expect ~5 sec per feature on MPS.
5. **Run Phase 2 full pipeline** (all 4,065 features). This is the bulk of the compute (~17.7 hours for Geneformer). Use resume-safe checkpointing.
6. **Run Phases 3–5 aggregation.** Report hub distribution, annotation bias statistics, signal attenuation across downstream layers.
7. **Run Phases 6–8 (three-way ablation).** Select 8 triplets using the paper's `select_triplets()` heuristic. Report pairwise/three-way redundancy ratios and synergy classification. **If the redundancy does not deepen monotonically (three-way > pairwise), the triplet selection is wrong or the ablation is broken.**
8. **Run Phases 9–12 (trajectory steering).** Requires a pseudotime-ordered input cell population. For human immune cells, use Tabula Sapiens. For other tissues, use an appropriate reference dataset. Select switch features via Stage 1 Cohen's *d* at extreme pseudotime deciles.
9. **Run Phase 12 layer-dependent directionality analysis.** The critical claim is that the **last layer features produce unanimous directional shift** (fraction positive close to 1.0 or 0.0). If they do not, `NEWMODEL` may not encode differentiation in the same progressive refinement manner as Geneformer — this is itself a scientific finding worth reporting.
10. **Run Phase 13 synthesis.** Report the three headline findings in a Stage 3 summary JSON and cross-reference against Stages 1 and 2.
11. **Cross-reference all four pipelines** (attention, Stage 1, Stage 2, Stage 3). A complete mechanistic audit of `NEWMODEL` produces four verdicts:
    - **Attention pipeline** (`../attention-grn-extraction-and-evaluation.md`): does attention encode regulatory logic or co-expression? (Geneformer/scGPT verdict: co-expression.)
    - **Stage 1** (`01-sae-atlas.md`): does the SAE atlas encode pathway-level biology at the feature level? (Verdict: yes, with U-shaped layer profile.)
    - **Stage 2** (`02-causal-circuit-tracing.md`): do SAE features form dense, biologically coherent causal circuits? (Verdict: yes, but gene-level predictions fail CRISPRi validation at 56.4% directional accuracy.)
    - **Stage 3 (this pipeline)**: is the selective-trace picture complete, are there higher-order logical gates, and does layer position encode causal cell-state control? (Verdict: no (40% of hubs unannotated), no (zero synergy at third order), and yes (L17 features unanimously push toward maturity).)
12. **A new model that passes all four pipelines** with positive findings would establish the first single-cell foundation model whose internal computations encode genuine causal regulatory logic. Stage 3 is the stage where the first positive finding in the mega-pipeline appears (layer-dependent differentiation control), so it is the right place to start looking for future positive results in other models.
