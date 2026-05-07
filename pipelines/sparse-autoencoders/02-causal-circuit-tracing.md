# Stage 2 — Causal Circuit Tracing over SAE Features

## 1. Overview

This pipeline converts the static SAE feature atlas produced by [Stage 1](01-sae-atlas.md) into a **directed computational graph** of feature-to-feature causal influence. For each well-annotated source feature at a chosen transformer layer, it performs a **single-feature ablation** — zeroing that feature's SAE activation in the residual stream and propagating the perturbation forward through the rest of the network — and measures, at every subsequent layer, how every downstream SAE feature's activation changes via **Cohen's *d* effect size** computed online with Welford's algorithm (Welford 1962). Edges passing a two-threshold significance filter (`|d| > 0.5` AND consistency > 0.7) are retained as the directed causal circuit graph. Applied to four experimental conditions across two foundation models (Geneformer V2-316M and scGPT whole-human) with K562 and Tabula Sapiens cell populations, the pipeline yields **96,892 total causal edges** and reveals **two distinct computational architectures**: Geneformer distributes computation across 4,608 features with cooperative 80/20-inhibitory dynamics and mean |d| = 1.05; scGPT concentrates computation in 2,048 features with competitive 65/35-inhibitory dynamics and mean |d| = 1.40.

The pipeline is **model-agnostic**: it consumes trained per-layer TopK SAE weights, feature annotations, and cached residual-stream activations from Stage 1, and emits a directed causal graph with cross-model consensus, gene-level predictions, CRISPRi validation, and disease-relevant circuit mapping. The only model-specific component is how the ablation intervention is propagated through the transformer block API — the pipeline includes both a Geneformer-style HuggingFace `BertForMaskedLM` path (full forward pass with `output_hidden_states=True` + hook-based ablation injection) and a scGPT-style path (`TransformerModel._encode` with manual layer iteration via `model.transformer_encoder.layers[dl](x, src_key_padding_mask=mask)`).

**Critical computational optimisation: the clean forward cache.** A naive implementation — ablating each source feature, re-running the entire forward pass per feature, across 120 source features × 200 cells × 6–17 downstream layers — requires ~20 days of compute. The pipeline instead **pre-computes one clean forward pass per cell** (capturing all layer hidden states), and inside that per-cell loop iterates over source features (zeroing each, computing the ablation delta, and propagating only the downstream half). This reduces Geneformer K562/K562 compute from ~20 days to **7.5 hours** (24,776 forward passes); the four-condition battery totals **14.8 hours and 80,191 forward passes** on a single Apple Silicon M2 Max with MPS acceleration.

**The pipeline's central scientific findings:**
1. **Dense causal graphs, predominantly inhibitory.** 52,116 edges in the primary Geneformer K562/K562 condition (120 source features → 26,338 target features; 31.9% target coverage); 80.1% inhibitory (ablation reduces downstream activation, implying feature encoded necessary information). The 20% excitatory fraction represents disinhibition — ablating a feature releases downstream features from inhibition.
2. **Hub features concentrate at early layers.** Top Geneformer hub L0_F2905 (Golgi Organization) has out-degree **8,028** across 17 downstream layers; L16 features achieve in-degrees **88–93** (out of 120 possible source features), marking L16 as a convergent integration layer.
3. **Circuits encode interpretable biological cascades.** 52.9% of annotated edges share ≥1 ontology term (16,507/31,176). Specific cascades include DNA-damage-response → mitotic-checkpoint → cell-cycle-arrest (L0 DNA Repair F3717 → L6 Kinetochore, *d* = −3.47); neurodevelopment → proteostasis (L0 Nervous System Development F146 → L2 Proteasomal Catabolism, *d* = −0.96, 140 shared ontology terms); scGPT protein quality control → stress response → biosynthetic recovery (L0 Protein Catabolism F507 → L1 Chromatin Organization, *d* = −8.19, 161 shared terms — the strongest individual edge in any condition).
4. **Multi-tissue SAEs yield 68.8% biological coherence (vs 52.9% K562-only) — and coherence is SAE-dependent, not cell-type-dependent.** K562 cells and Tabula Sapiens cells processed through the same multi-tissue SAEs yield 68.8% vs 68.5% coherence (Δ = 0.3%, non-significant), while circuit **density** is 3–5× cell-type-dependent.
5. **Cross-model consensus: 1,142 conserved domain pairs at 10.6× enrichment over chance** (*p* < 0.001, 1,000 permutations), with 303 high-confidence pairs (|d| > 1.0 in both models). Consensus circuits reveal ~53% shared ontology as a **universal constant** of biological knowledge organisation across architecture, feature space, and training data.
6. **Gene-level perturbation validation bounds predictive utility.** 975,369 gene-pair predictions extracted from circuit edges. Against Replogle K562 CRISPRi genome-scale screen (282,250 validatable pairs): **directional accuracy 56.4%** (marginally above 50% chance), magnitude correlation Spearman ρ = 0.038 (near-zero). Only 6.0% of perturbed genes show enrichment of circuit-predicted targets. **This is consistent with the companion Stage 1 finding** (`01-sae-atlas.md`): single-cell foundation models encode **co-expression structure, not causal regulation**.
7. **Disease-relevant domains are more central AND more cross-model conserved.** Mann-Whitney *p* = 1.2 × 10⁻¹¹ for centrality; 3.59× more likely to appear in cross-model consensus (*p* < 0.001).

**This is Stage 2 of the SAE mega-pipeline.** It consumes trained SAE weights and feature annotations from Stage 1 (`01-sae-atlas.md`) and feeds its selective-tracing graph to Stage 3 (`03-exhaustive-mapping-and-steering.md`), which extends this selective 30-features-per-layer approach to **exhaustive tracing of all 4,065 active features at one layer** (27× expansion) plus three-way combinatorial ablation and trajectory-guided feature steering. See `README.md` in this directory for the mega-pipeline guide.

## 2. Source

- **Paper:** Anonymous, (2026). *Causal Circuit Tracing Reveals Distinct Computational Architectures in Single-Cell Foundation Models: Inhibitory Dominance, Biological Coherence, and Cross-Model Convergence.* arXiv:2603.01752 [cs.LG], 4 Mar 2026 (v2). 33 pages (20 main + 5 references + 8 supplementary notes).
- **Local PDF:** `../../references/2603.01752_anon_causal_circuit_tracing.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/bio-sae-circuits — pinned commit `4f6bfb4a2193c1104611db89f8c7263e1c96797d` (2026-03-02, "Remove manuscript files from repository"). Local clone at `../../repos/bio-sae-circuits/`.
- **Companion Stage 1 repo:** https://github.com/Biodyn-AI/bio-sae — provides the trained SAE atlases this pipeline consumes. Local clone at `../../repos/bio-sae/`. See `01-sae-atlas.md`.
- **Live interactive atlases** (for feature annotation lookup): https://biodyn-ai.github.io/geneformer-atlas/, https://biodyn-ai.github.io/scgpt-atlas/.
- **Mega-pipeline guide:** `README.md` in this directory.
- **Key foundational references:** Cunningham et al. 2023 / Bricken et al. 2023 / Templeton et al. 2024 (SAE interpretability), Gao et al. 2024 (TopK SAEs), Makhzani & Frey 2013 (k-sparse autoencoders), Elhage et al. 2022 (superposition), Vig et al. 2020 / Geiger et al. 2021 (causal abstraction / activation patching), Marks et al. 2024 (sparse feature circuits in language models), Wang et al. 2023 (IOI circuit), Welford 1962 (online variance algorithm), Cohen 1988 (effect size conventions), Manning & Schütze 1999 (PMI), Theodoris et al. 2023 (Geneformer), Cui et al. 2024 (scGPT), Replogle et al. 2022 (genome-scale Perturb-seq), Tabula Sapiens Consortium 2022, Ashburner et al. 2000 (GO), Kanehisa & Goto 2000 (KEGG), Jassal et al. 2020 (Reactome), Szklarczyk et al. 2023 (STRING), Han et al. 2018 (TRRUST v2), Ciccia & Elledge 2010 (DDR), Malumbres & Barbacid 2009 (CDKs), Walter & Ron 2011 (unfolded protein response), Hetz 2012 (integrated stress response), Chandel 2021 (mitochondria review), Ross & Poirier 2004 / Labbadia & Morimoto 2015 (protein homeostasis and neurodegeneration).

## 3. Inputs

### 3.1 Target model(s) to trace

The pipeline expects **one or more trained single-cell transformer models** with companion SAE atlases from Stage 1. Each target requires:

1. **The trained foundation model** accessible via a forward-pass API that supports mid-layer hidden-state replacement. Two supported forms:
   - **HuggingFace `BertForMaskedLM` form** (Geneformer): `model(input_ids, output_hidden_states=True)` for the clean pass; ablation via hook replacement at the source layer during a second forward pass. The HuggingFace `forward` returns `hidden_states` as a tuple of `(n_layers + 1)` tensors — index `source_layer + 1` corresponds to the output of transformer layer `source_layer`.
   - **scGPT form**: `model._encode(src=gene_ids, values=gene_values, src_key_padding_mask=mask)` for the clean pass, with hooks on `model.transformer_encoder.layers[L]` to capture per-layer hidden states. Ablated pass manually iterates remaining layers: `for dl in range(source_layer + 1, n_layers): x = model.transformer_encoder.layers[dl](x, src_key_padding_mask=mask)`.
   - For a new model family, the adapter contract is: given a source layer `L_src`, a replacement hidden state `h^(L_src)_abl`, and an input cell, return per-layer hidden states for all `ℓ > L_src`.
2. **FlashMHA → standard MultiheadAttention conversion** if the model uses FlashMHA: convert `W_qkv` → `in_proj_weight`. Source paper notes this conversion is required for scGPT inference on MPS because FlashMHA is CUDA-only.
3. **Trained SAEs at every layer to be measured** (from Stage 1). For the primary condition, this is all 18 Geneformer layers and all 12 scGPT layers. For multi-tissue experimental conditions, Stage-1 multi-tissue SAEs are required at layers {0, 5, 11, 17} for Geneformer.
4. **Feature annotations from Stage 1**: `feature_annotations.json` per layer (used for Phase 1 source feature selection and Phase 6 biological coherence), plus `feature_catalog.json` per layer (for Phase 10 gene-level prediction extraction).

**Worked examples in the source paper:**
- **Geneformer V2-316M** (Theodoris et al. 2023): 18 layers, d=1,152, 18 attention heads. SAEs: TopK k=32, 4× overcomplete (4,608 features per layer). Two SAE variants: **K562-only** (trained on 1M K562 positions/layer, available at all 18 layers) and **multi-tissue** (trained on 500K K562 + 500K Tabula Sapiens positions, available at layers 0, 5, 11, 17).
- **scGPT whole-human** (Cui et al. 2024): 12 layers, d=512, 8 attention heads. SAEs: TopK k=32, 4× (2,048 features). Natively multi-tissue — trained on all 3,561,832 Tabula Sapiens positions/layer.

### 3.2 Cell populations (inputs for Phase 2 clean forward cache)

| Population | Role in pipeline | Source |
|---|---|---|
| **K562 control cells** | Primary condition for Geneformer (K562/K562, K562/Multi). **200 control cells** from Replogle et al. 2022 genome-scale CRISPRi dataset — **same cells used for Stage 1 SAE training**. Tokenised via Geneformer's rank-value encoding. | Figshare DOI 10.6084/m9.figshare.21452470 |
| **Tabula Sapiens** | Multi-tissue condition for Geneformer (TS/Multi) and all scGPT conditions. **200 cells stratified by tissue:** 67 immune (41 cell types) + 67 kidney (13 cell types) + 66 lung (34 cell types) — totalling **88 cell types**. For Geneformer: rank-value tokenised. For scGPT: continuous expression values, genes sorted by expression (descending), padded to 1,200 positions. | Tabula Sapiens Consortium 2022 |

### 3.3 Biological annotation databases

Same five databases used in Stage 1, required here for Phase 6 biological coherence analysis and Phase 1 source feature selection:

| Database | Version | Usage |
|---|---|---|
| **Gene Ontology Biological Process** | GO BP 2024-01 (Ashburner et al. 2000) | Domain labelling, shared-ontology edge tests, disease keyword matching |
| **KEGG** | Release 109 (Kanehisa & Goto 2000) | Shared-ontology edge tests |
| **Reactome** | v87 (Jassal et al. 2020) | Shared-ontology edge tests |
| **STRING** | v12.0 ≥ 700 (Szklarczyk et al. 2023) | Shared-ontology edge tests |
| **TRRUST** | v2 (Han et al. 2018) | Shared-ontology edge tests |

### 3.4 Ground-truth perturbation data (for Phase 11 validation)

**Replogle K562 genome-scale CRISPRi screen** (Replogle et al. 2022): 643,413 cells across 2,023 knockdown targets. Used to compute pseudobulk log-fold-change responses per target vs control cells, then compared to circuit-derived gene-pair predictions. 599 of 2,023 perturbation targets overlap with circuit source genes, yielding **282,250 validatable gene pairs**.

## 4. Outputs

Per experimental condition × model, the pipeline emits:

1. **Source feature ranking** per layer: `source_features_L{L}.json` — list of 30 (or 120) source features ordered by annotation quality score, with their top-20 genes and ontology enrichments.
2. **Clean forward cache** (transient, in-memory): per cell × per layer hidden states, per-layer SAE encodings of both clean and ablated activations. Typically ~2-5 GB per condition.
3. **Per-source-layer circuit file:** `circuit_L{L:02d}_features.json` — for each source feature, the top-50 downstream effects per downstream layer: `{target_feature_id, target_layer, cohens_d, consistency, sign}`. Significant edges: `|d| > 0.5` AND consistency > 0.7.
4. **Unified circuit graph** (aggregated across all source layers): `circuit_graph.json` — all significant edges with `(source_layer, source_id, target_layer, target_id, d, consistency, sign)` plus per-feature in-degree and out-degree tables.
5. **Per-condition statistics summary** (per Table 1 of the paper): `circuit_stats_{condition}.json`:
   - K562/K562 Geneformer: **52,116 edges**, 120 source → 26,338 targets, 31.9% coverage, mean |d| = **1.05**, median 0.92, 41.4% |d| > 1.0, 4.3% |d| > 2.0, **80.1% inhibitory**, 52.9% shared ontology, 7.5 hr compute (24,776 passes)
   - K562/Multi Geneformer: **8,298 edges**, 90 → 4,171, mean |d| = 0.98, median 0.87, 34.4% |d| > 1.0, **79.9% inhibitory**, 68.8% shared ontology, 3.5 hr (18,465 passes)
   - TS/Multi Geneformer: **5,098 edges**, 90 → 2,962, mean |d| = **0.72**, median 0.63, 10.4% |d| > 1.0, **89.4% inhibitory**, 68.5% shared ontology, 3.2 hr (18,455 passes)
   - TS/Multi scGPT: **31,380 edges**, 90 → 1,960, 95.7% coverage, mean |d| = **1.40**, median 1.19, **65.2% |d| > 1.0**, **65.5% inhibitory**, 53.0% shared ontology, 37.2 min (18,495 passes)
6. **Hub feature tables:** `hub_features_{condition}.json` — top features ranked by out-degree (early-layer broadcast hubs) and in-degree (late-layer convergent integration). Geneformer K562/K562 top hubs: L0_F2905 Golgi Organization **8,028**; L0_F2982 RNA Methylation **6,921**; L0_F1568 Growth Factor Response **6,006**; L0_F3402 Cholesterol Biosynthesis **5,096**; L0_F4201 RNA Splicing **4,782**. Top in-degrees: L16_F2818 **93**, L16_F1691 **89**, L16_F4354 **89**, L16_F1375 **88**, L16_F1057 **88**. scGPT top hubs: L4_F446 Endonucleolytic Cleavage (rRNA) **6,494** (single highest-connectivity feature across both models); L4_F1643 Aerobic Electron Transport Chain **6,050**; L0_F552 NADH Dehydrogenase **4,785**; L0_F590 NADH Dehydrogenase **3,849**; L0_F233 Aerobic ETC **3,420**.
7. **Per-source-layer statistics:** `layer_stats_{condition}.json` (per Tables 2, 6). Geneformer K562/K562 per-source-layer (each with 30 features): L0 17 downstream layers / 6,176 passes / 150 min / 73,769 edges / 2,459 avg/feature (range 382–8,028); L5 12 / 6,200 / 140 min / 41,684 / 1,389 (533–4,703); L11 6 / 6,200 / 86 min / 30,981 / 1,033 (321–3,536); L15 2 / 6,200 / 71 min / 18,438 / 615 (79–3,257). scGPT: L0 11 downstream / 6,194 passes / 17.0 min / 49,777 / 1,659; L4 7 / 6,187 / 14.2 min / 41,409 / 1,380; L8 3 / 6,114 / 5.9 min / 11,361 / 379.
8. **Biological coherence report** per condition: `biological_coherence_{condition}.json` — fraction of annotated edges with shared ontology terms, per-database breakdown. Universal constant ≈ **53%** for K562/K562 architectures; **68.8%** for multi-tissue SAEs (SAE-dependent). TS/Multi vs K562/Multi coherence: **68.5% vs 68.8% (Δ = 0.3%, non-significant)** — confirms coherence is SAE-dependent, not cell-type-dependent.
9. **PMI validation report:** `pmi_validation.json` — target-feature overlap between causal circuits and Stage-1 PMI graphs. Per Table 5 for Geneformer K562/K562: L0→L5 **90.6%**, L5→L11 **94.8%**, L11→L17 **91.7%**. Zero source feature overlap (different selection criteria).
10. **Causal effect attenuation curves:** `attenuation_{condition}.json` — per source layer, number of significant edges per feature at each downstream layer. Geneformer K562/K562: L0 effects persist across all 17 downstream layers (~200–215 edges/feature for L1–L5, declining to 58 at L17); L5 effects 191 → 62 over 12 layers; L11 effects 223 → 145 over 6 layers; L15 effects **increase** 279 → 336 (late-layer consolidation).
11. **Cross-model consensus graph:** `consensus_graph.json` (per Table 9) — 1,142 conserved domain pairs at **10.6× enrichment** over chance (permutation test, 1,000 iterations, *p* < 0.001, expected = 107.3); **303 high-confidence** consensus pairs (|d| > 1.0 in both models); top consensus includes Golgi Organization → Protein Insertion Into Membrane (GF |d| = 4.75, scGPT |d| = 5.23), Cholesterol Biosynthesis → Sterol Biosynthesis (GF 4.44, scGPT 1.54), Maturation of SSU-rRNA → RNA Methylation (GF 1.67, scGPT 5.00).
12. **Gene-level predictions:** `gene_predictions.json` — **975,369 gene-pair predictions** extracted from 47,418 annotated edges (top-10 rank-weighted genes per feature, filtered by evidence ≥ 2 OR |d| > 2). 0.15% match STRING/TRRUST; 1.14% share ≥ 2 GO BP terms; **98.7% connect genes without established direct relationships**; 32.8% derive from cross-model consensus.
13. **CRISPRi validation report:** `crispri_validation.json` — 282,250 validatable pairs. **Directional accuracy 56.4%** (vs 50% chance, statistically significant due to sample size, *p* < 10⁻³²). Magnitude correlation Spearman **ρ = 0.038** (near-zero). Only **6.0% of 599 perturbation targets** show nominally significant enrichment of circuit-predicted targets (Fisher's exact uncorrected *p* < 0.05). Cross-model consensus predictions marginally better: **57.3%** vs 56.4%.
14. **Disease circuit mapping:** `disease_mapping.json` (per Table 10) — 11 disease-relevant categories via GO BP keyword matching. Disease-associated domains are **3.59× more likely** to appear in cross-model consensus pairs (*p* < 0.001, Fisher's) and significantly more central in the circuit graph (median 14 vs 3 edges, Mann-Whitney *p* = **1.2 × 10⁻¹¹**).
15. **Process hierarchy meta-graph:** `process_hierarchy.json` — directed domain graph with **1,126 nodes and 16,002 edges**, 499 feedback loops (reciprocal A↔B), **300 DDR → cell cycle edges all with positive layer deltas (ΔL > 0)**. Early-layer processes (mean layer < 1): MAPK Cascade L0.1, Ras Signalling L0.3, Histone Modification L0.4, COPII Vesicle Budding L0.7. Late-layer processes (mean layer > 16): gene expression regulation, RNA surveillance, protein localisation to chromatin.
16. **Tissue-specific circuits:** `tissue_specific_circuits.json` — 3,541 domain pairs unique to multi-tissue conditions vs 1,334 shared with K562. **Immune circuit pairs 3.18× more frequent** in multi-tissue-specific set (*p* < 0.001, Fisher's). **201 immune-specific circuit pairs**, 89% absent from K562 circuits.

## 5. Dependencies

- **Python ≥ 3.10**, PyTorch ≥ 2.0 (CUDA or Apple Silicon MPS, with `PYTORCH_ENABLE_MPS_FALLBACK=1` for missing MPS kernels), numpy ≥ 1.26, scipy ≥ 1.11, networkx ≥ 3.1, h5py ≥ 3.9, transformers ≥ 4.30, anndata ≥ 0.10, matplotlib ≥ 3.7.
- **Stage 1 outputs** (from `01-sae-atlas.md`):
  - `sae_final.pt` per layer per SAE variant (K562-only + multi-tissue for Geneformer; multi-tissue for scGPT)
  - `feature_annotations.json` per layer
  - `feature_catalog.json` per layer
  - Cached residual-stream activation memmaps (`phase1_k562/layer_{L:02d}_activations.npy`)
- **Foundation model checkpoints:**
  - Geneformer V2-316M: HuggingFace `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M`
  - scGPT whole-human: from Cui et al. 2024
- **Ground-truth data:**
  - Replogle K562 CRISPRi h5ad (Figshare DOI 10.6084/m9.figshare.21452470)
  - Tabula Sapiens h5ad (immune, kidney, lung)
- **Reference databases:** GO BP 2024-01, KEGG R109, Reactome v87, STRING v12.0 ≥ 700, TRRUST v2.
- **Compute budget:** entire four-condition circuit-tracing battery runs in **14.8 hours** wall-clock on a single Apple Silicon M2 Max (80,191 forward passes). Add ~3 hours for downstream knowledge extraction (Phases 9–13) and ~1 hour for figure generation.
- **Environment files:** `../../repos/bio-sae-circuits/requirements.txt`. Scripts: Geneformer `src/13_causal_circuit_tracing.py`, scGPT `scgpt_src/13_causal_circuit_tracing.py`.

## 6. Methodology

The pipeline runs in thirteen phases. Phases 0–4 are the core causal tracing algorithm; Phases 5–8 are the per-condition battery; Phases 9–13 are systematic knowledge extraction, gene-level prediction, validation, and disease mapping.

---

### Phase 0 — Input preparation: load Stage 1 artefacts

**Purpose:** consume the trained SAE atlas from Stage 1 and prepare the ablation infrastructure.

**Steps:**

1. **Verify Stage 1 completion.** For the target model at every layer that will be measured:
   - `sae_final.pt` must exist and load cleanly as a `TopKSAE` instance (`encoder`, `decoder`, `mu` centering vector, `k=32` sparsity).
   - `feature_annotations.json` must exist and have ≥ 30 features per source layer with ≥ 1 significant ontology enrichment (otherwise Phase 1 source selection will fail).
2. **Load SAEs into an `SAECache`** (lazy-loading cache, `repos/bio-sae-circuits/src/13_causal_circuit_tracing.py:209`). Call `cache.preload([L_src, L_src+1, ..., L_last])` up-front to avoid redundant loading during the per-cell inner loop.
3. **Load activation means** (the `mu` centering vector computed during Stage 1 training) per layer. These are applied before SAE encoding: `z = TopK(W_enc · (h − μ))`.
4. **Load the foundation model** in eval mode on the target device (MPS or CUDA). Confirm the forward API:
   - Geneformer: `model(input_ids, output_hidden_states=True)` returns `hidden_states = tuple(n_layers + 1)`.
   - scGPT: `model._encode(src, values, src_key_padding_mask)`; register hooks on `model.transformer_encoder.layers[L]`.
5. **Load the 200-cell population** (K562 or Tabula Sapiens, per the condition; §3.2). Pre-tokenise all cells to avoid per-iteration tokenisation cost. Store `(input_ids, attention_mask, gene_names_per_position)` per cell.

**Code references:**
- SAE class: `repos/bio-sae-circuits/src/sae_model.py` → `TopKSAE`, reused from Stage 1.
- SAE cache: `repos/bio-sae-circuits/src/13_causal_circuit_tracing.py:209` → `SAECache`.
- K562 loader: `src/13_causal_circuit_tracing.py:275` → `_load_k562_cells()`.
- Tabula Sapiens loader: `src/13_causal_circuit_tracing.py:342` → `_load_tabula_sapiens_cells()`.

---

### Phase 1 — Source feature selection

**Purpose:** select the 30 most richly annotated features per source layer for circuit tracing. The 30-feature-per-layer budget is deliberately small — tracing is expensive, and well-annotated features enable interpretable circuit analysis.

**Steps:**

1. **For each candidate source feature at a chosen source layer:** compute the **annotation quality score** as the sum `∑ −log₁₀(p_i)` over all significant ontology enrichments (GO BP, KEGG, Reactome, STRING, TRRUST; BH-corrected *q* < 0.05). Features with more enrichments at smaller *p*-values receive higher scores.
2. **Rank features by score** and select the **top 30**. The primary paper uses:
   - **Geneformer K562/K562:** source layers {L0, L5, L11, L15}, 30 features each → **120 total source features**.
   - **Geneformer K562/Multi and TS/Multi:** source layers {L0, L5, L11, L17} (multi-tissue SAEs only available at these layers), 30 features each → **90 total source features** (note: paper reports 90, not 120, because the L17 multi-tissue SAE has fewer well-annotated features than L15 K562-only).
   - **scGPT TS/Multi:** source layers {L0, L4, L8}, 30 features each → **90 total source features**.
3. **Persist `source_features_L{L}.json`** with the ranked feature IDs, annotation scores, and top enrichment terms.

**Rationale for 30 features per source layer:** balance between interpretability (all selected features are well-annotated) and coverage. The paper's Limitation 1 notes that results on well-annotated features may not generalise to unannotated features — this is the exact limitation that **Stage 3's exhaustive tracing** addresses by tracing all 4,065 active features at one layer without annotation filtering (27× expansion; see `03-exhaustive-mapping-and-steering.md`).

**Code reference:** `src/13_causal_circuit_tracing.py:114` → `select_features()`.

---

### Phase 2 — Clean forward cache construction

**Purpose:** pre-compute the "clean" (unperturbed) state of the network for every cell, so the inner ablation loop can skip redundant forward passes. **This is the critical compute optimisation.**

**Rationale:** a naive ablation loop iterates source_features × cells, running a full forward pass per iteration. For 120 features × 200 cells × 17 downstream layers with model inference ~5 sec/pass on MPS, this is ~20 days. The clean forward cache runs **one forward pass per cell** (capturing all hidden states), then iterates over features **inside** the per-cell loop (zeroing each feature, propagating only the downstream half). This yields ~7.5 hours for Geneformer K562/K562.

**Steps (per cell):**

1. **Clean forward pass.** Register PyTorch hooks at every transformer layer's output (post-residual-stream). Run the full forward pass. Store `h^(ℓ)_clean` for every layer `ℓ ∈ [0, n_layers)`.
   - **Geneformer:** `outputs_clean = model(input_ids, output_hidden_states=True)`. Access `h^(ℓ)_clean = outputs_clean.hidden_states[ℓ + 1][0]` (the `+1` is because `hidden_states[0]` is the input embedding; `hidden_states[ℓ + 1]` is the output of layer `ℓ`).
   - **scGPT:** hooks on `model.transformer_encoder.layers[ℓ]`, then `_ = model._encode(src, values, src_key_padding_mask)`. Captured via `capture_dict[layer_idx] = output`.
2. **Encode all clean hidden states through the corresponding layer SAEs.** For each layer `ℓ`: `z^(ℓ)_clean = TopK(W_enc^(ℓ) · (h^(ℓ)_clean − μ^(ℓ)))`, then `h̃^(ℓ)_clean = W_dec^(ℓ) · z^(ℓ)_clean + μ^(ℓ)` (the reconstructed clean state, used as the ablation baseline).
3. **Store the cache** in CPU memory (not GPU) as `{layer → tensor}` dictionaries. Typical footprint: ~1–2 GB per 200-cell batch.

**Note on reconstruction vs true hidden state:** the ablation is applied to the **reconstructed** clean state `h̃^(ℓ)_clean`, not the original `h^(ℓ)_clean`. This is necessary because the ablation operation is performed in the SAE feature space (zeroing `z_f`), and the decoded result is in the reconstructed subspace. Residual information present in `h^(ℓ)_clean` but not captured by the SAE (the ~20% of variance the SAE does not explain) is preserved by computing the ablation delta and adding it to the original hidden state — see Phase 3 step 3.

**Code reference:** `src/13_causal_circuit_tracing.py:445` → `trace_source_layer()`, cache construction at lines 537–553.

---

### Phase 3 — Single-feature ablation algorithm

**Purpose:** for each source feature, zero its SAE activation at the source layer, propagate the perturbation through the remaining transformer layers, and measure the per-feature effect at every downstream layer.

**Core algorithm** (from paper §4.1, per source feature `f` at source layer `L_src`, for each cell):

1. **Source ablation in SAE feature space.**
   `z = SAE_enc(h^(L_src)_clean)`
   `z_f ← 0` (zero feature `f`'s activation)
   `h̃ = SAE_dec(z)` (reconstruct)
2. **Compute the ablation delta** in the residual-stream basis:
   `δ = h̃ − SAE_dec(z_original)` (= `h̃ − h̃^(L_src)_clean`)
   This `δ` captures *only* the direction that feature `f` contributed to the clean reconstruction.
3. **Apply the ablation to the original hidden state:**
   `h^(L_src)_abl = h^(L_src)_clean + δ`
   This preserves the ~20% of variance not captured by the SAE while subtracting feature `f`'s contribution in the feature-basis direction.
4. **Ablated downstream pass.** Starting from `h^(L_src)_abl`, manually propagate through all subsequent transformer layers:
   - **Geneformer:** register a hook at `L_src` that replaces the forward output with `h^(L_src)_abl`, then re-run the forward pass from the input (the hook intercepts at `L_src` and overwrites the value). The subsequent layers receive the ablated state and produce ablated hidden states `h^(ℓ)_abl` for all `ℓ > L_src`.
   - **scGPT:** manually iterate remaining layers:
     ```
     x = h^(L_src)_abl
     for dl in range(L_src + 1, n_layers):
         x = model.transformer_encoder.layers[dl](x, src_key_padding_mask=mask)
         h^(dl)_abl = x
     ```
5. **Downstream measurement at every downstream layer `ℓ > L_src`:**
   `Δ^(ℓ)_j = SAE_enc(h^(ℓ)_abl)_j − SAE_enc(h^(ℓ)_clean)_j`
   for every downstream feature `j`. This is a per-feature delta vector of length `d_SAE`.
6. **Online accumulation via Welford's algorithm** (Welford 1962). Maintain a `WelfordAccumulator` keyed by `(source_layer, source_feature, target_layer, target_feature)`. For each new cell's `Δ^(ℓ)_j`, call `accumulator.update(Δ^(ℓ)_j)`. Welford's algorithm maintains:
   - Running mean `Δ̄_j`
   - Running variance `s²_j`
   - Count of positive vs negative deltas (for consistency computation)
   Without storing per-cell results (critical for memory — otherwise 200 cells × 120 source × 18 target layers × 4,608 target features ≈ 40 GB per condition).
7. **After all cells processed**, compute for each `(source, target)` pair:
   - **Cohen's *d*** = `Δ̄_j / s_j` (Cohen 1988; *d* > 0.5 is "medium effect").
   - **Consistency** = `max(pos_count, n − pos_count) / n` (directional agreement: fraction of cells where delta has the same sign as the mean).
   - **Sign**: inhibitory if mean delta is negative (ablation reduces downstream activation), excitatory if positive (ablation releases from inhibition).

**Resume capability:** partial results are checkpointed every 50 cells. If the pipeline is interrupted, rerun from the checkpoint; the Welford accumulator state is serialised and can be resumed.

**Code references:**
- Core tracing routine: `src/13_causal_circuit_tracing.py:445` → `trace_source_layer()`
- Welford accumulator: `src/13_causal_circuit_tracing.py:180` → `WelfordAccumulator` (`update()`, `finalize()`)
- Geneformer hook-based ablation: `src/13_causal_circuit_tracing.py:432` → `make_hook()`
- scGPT manual layer iteration: `scgpt_src/13_causal_circuit_tracing.py:468` → `make_capture_hook()` and the `_encode` + manual layer loop at lines 570+

**Critical pitfall:** the ablation must apply `δ` to `h^(L_src)_clean` (the original hidden state), not to `h̃^(L_src)_clean` (the reconstructed state). If you overwrite with the reconstruction instead of adding the delta, the model loses ~20% of its input information unrelated to feature `f`, producing spurious effects that have nothing to do with the feature being ablated. See Pitfall #3.

---

### Phase 4 — Circuit edge significance filtering and persistence

**Purpose:** apply the two-threshold significance filter to the raw effect sizes from Phase 3 and persist the directed circuit graph.

**Steps:**

1. **For each `(source_layer, source_feature, target_layer, target_feature)` tuple:** retain it as a significant edge if:
   - **|Cohen's *d*| > 0.5** (Cohen's convention for "medium effect size")
   - **AND consistency > 0.7** (the feature responds in the same direction in > 70% of cells)
2. **Persist** per-source-layer `circuit_L{L:02d}_features.json`. For each source feature, store the top-50 strongest downstream effects per downstream layer (sorted by |d|).
3. **Threshold sensitivity note** (paper Limitation 2): the thresholds are somewhat arbitrary. Doubling |d| to > 1.0 yields ~41% of the edges in Geneformer K562/K562 (vs 100% at |d| > 0.5); tripling to > 2.0 yields 4.3%. Qualitative conclusions (inhibitory dominance, hub features, cross-model consensus) are stable across these thresholds.

**Code reference:** significance filtering at `src/13_causal_circuit_tracing.py:657`.

---

### Phase 5 — Circuit graph construction and hub analysis

**Purpose:** aggregate per-source-layer results into a single directed circuit graph and characterise its hub architecture.

**Steps:**

1. **Aggregate.** Take the union of all significant edges across source layers. For each edge, record: source feature ID, target feature ID, Cohen's *d*, consistency, sign (inhibitory/excitatory), shared-ontology flag (Phase 6).
2. **Compute hub statistics.** For each feature, compute **out-degree** (number of downstream features it causally influences) and **in-degree** (number of upstream features that causally influence it). Rank separately.
3. **Expected hub distributions** (Geneformer K562/K562, Table 3):
   - **Top out-degree (broadcast hubs, early layers):**
     - L0_F2905 Golgi Organization: **8,028** (the most-connected feature across all 17 downstream layers; Supp Note 5 details)
     - L0_F2982 RNA Methylation: **6,921**
     - L0_F1568 Growth Factor Response: **6,006**
     - L0_F3402 Cholesterol Biosynthesis: **5,096** (Supp Note 4)
     - L0_F4201 RNA Splicing: **4,782**
   - **Top in-degree (convergent integration, L16):**
     - L16_F2818: **93**
     - L16_F1691: **89**
     - L16_F4354: **89**
     - L16_F1375: **88**
     - L16_F1057: **88**
   - **Interpretation:** hub features at early layers act as broadcast nodes, causally influencing thousands of downstream features. Late-layer features at L16 achieve in-degrees approaching the maximum possible 120 source features — **L16 serves as a convergent integration layer** where diverse upstream biological computations are consolidated before the final output layer.
4. **Expected hub distributions** (scGPT TS/Multi, Table 7):
   - **L4_F446 Endonucleolytic Cleavage (rRNA): 6,494** — the single highest-connectivity feature across either model. Rate-limiting step for protein synthesis.
   - **L4_F1643 Aerobic Electron Transport Chain: 6,050**
   - **L0_F552 NADH Dehydrogenase Complex Assembly: 4,785**
   - **L0_F590 NADH Dehydrogenase Complex Assembly: 3,849**
   - **L0_F233 Aerobic Electron Transport Chain: 3,420**
   - **L0_F880 Golgi Organization: 3,133**
   - **L4_F725 Heart Development: 3,046**
   - **Architectural contrast:** Geneformer hubs centre on chromatin/RNA processing; scGPT hubs centre on mitochondrial electron transport. **This divergence reflects training-objective differences** — Geneformer's rank-value + next-token prediction rewards identifying highly-expressed genes (chromatin/RNA hubs); scGPT's continuous-value + masked-gene prediction uses energy metabolism as a proxy for expression level variance across cell types.
5. **Causal effect attenuation curves** (Section 2.4). For each source layer, report the number of significant edges per feature at each downstream layer:
   - **Geneformer L0 effects persist remarkably:** ~200–215 edges/feature for L1–L5 (with a slight *increase* at L3), then gradually decaying; at L17 (17 layers downstream) still **58 edges/feature**. This persistence has important implications: early-layer features contain foundational biological information that propagates through the entire network.
   - **L5 effects decay more linearly:** 191 → 62 edges/feature over 12 downstream layers.
   - **L11 effects:** 223 → 145 over 6 downstream layers.
   - **L15 → L17 effects actually increase:** 279 → 336 — late-layer consolidation for output prediction.
   - **scGPT** shows a flat plateau (~155–176 edges/feature for L0 through L6) before gradual decline to 114 at L11. L4 features are the most broadly connected at 197.9 edges/downstream-layer — **scGPT is a "funnel-then-broadcast" architecture** where mid-network features serve as integration points.

**Code reference:** `src/13_causal_circuit_tracing.py:780` → `build_circuit_graph()`; attenuation at `analyze_attenuation()` lines 1000–1046.

---

### Phase 6 — Biological coherence analysis

**Purpose:** quantify what fraction of causal edges connect biologically related features — the "interpretability fraction" of the circuit graph.

**Steps:**

1. **For each significant edge** `(source_feature, target_feature)`, look up the annotation sets (from Phase 0 / Stage 1 `feature_annotations.json`) for both endpoints. An annotation set is a list of (database, term_id) pairs across GO BP 2024-01, KEGG R109, Reactome v87, STRING v12.0, TRRUST v2.
2. **Test for shared ontology:** edge has "shared ontology" if the intersection of source and target annotation sets is non-empty (at least one common `(database, term_id)` pair).
3. **Compute biological coherence fraction** = (number of edges with shared ontology) / (total edges with annotations on both endpoints).
4. **Expected values** (the "universal constant" finding):
   - **Geneformer K562/K562: 52.9%** (16,507 / 31,176 annotated edges)
   - **Geneformer K562/Multi: 68.8%** — multi-tissue SAEs yield a **+16 percentage point jump**
   - **Geneformer TS/Multi: 68.5%** — virtually identical to K562/Multi, despite processing completely different cells through the same SAEs
   - **scGPT TS/Multi: 53.0%** — virtually identical to Geneformer K562/K562 despite using different architecture, feature space, and training data
5. **Critical interpretation: biological coherence is SAE-dependent, NOT cell-type-dependent.** The K562/Multi vs TS/Multi comparison (68.8% vs 68.5%, Δ = 0.3%) disentangles the SAE lens from the input data: changing the input cell type (K562 → TS) while keeping the SAE fixed produces essentially identical coherence. The +16pp improvement from multi-tissue SAE training is a **SAE-methodology improvement**, not a model-capability improvement.
6. **~53% as a structural constant** (Section 3.2). Geneformer K562/K562 and scGPT TS/Multi both hit ~53% coherence despite fundamentally different architectures. This convergence suggests ~53% reflects a **structural property of biological knowledge organisation**: given the hierarchical structure of GO/KEGG/Reactome/STRING, approximately half of any set of causally interacting feature pairs will share at least one annotation term.
7. **Universal inhibitory dominance** (Section 3.3). The predominance of inhibitory edges (65–89%) across all four conditions implies features encode **necessary information** — removing a feature reduces its dependents' activation. The 20–35% excitatory fraction represents **disinhibition**: removing a feature releases downstream features from inhibition. TS/Multi achieves 89.4% inhibitory dominance — the highest of any condition — because processing unfamiliar cell types reduces per-feature redundancy, so ablation more consistently impairs downstream computation. scGPT's 65.5% inhibitory reflects more competitive dynamics with greater feature suppression.

**Code reference:** `src/13_causal_circuit_tracing.py` → `analyze_biology()` lines 917–997.

---

### Phase 7 — PMI validation against Stage 1 co-activation graph

**Purpose:** cross-validate the causal circuit graph against the statistical co-activation graph (PMI) computed independently in Stage 1, demonstrating that causal tracing captures the same structure PMI captures plus directionality and sign.

**Steps:**

1. **Load Stage 1 PMI graphs** at the three layer pairs L0→L5, L5→L11, L11→L17 (the pairs used in Stage 1 Phase 7 cross-layer information highways).
2. **For each source feature** at a layer in the causal graph: identify its top-50 causal target features at the downstream layer, and its top-50 PMI target features.
3. **Compute target overlap:** `|causal_targets ∩ pmi_targets| / |causal_targets|`.
4. **Expected results** (Table 5, Geneformer K562/K562):
   - **L0 → L5: 90.6%** target overlap (4,101 PMI targets, 1,113 causal targets)
   - **L5 → L11: 94.8%** (4,369 PMI, 996 causal)
   - **L11 → L17: 91.7%** (4,205 PMI, 881 causal)
5. **Zero source feature overlap.** Source features in Stage 2 are selected by annotation quality (Phase 1); in Stage 1 they are all alive features. These are different selection criteria and produce completely disjoint source sets. But the **downstream** feature sets identified as causally influenced are **91–95% the same** as those identified as statistically co-activated by PMI — confirming PMI and causal tracing capture genuine information flow.
6. **Interpretation (Section 3.2 of paper):** causal tracing adds **directionality, magnitude, and sign** (inhibitory vs excitatory) that PMI cannot provide, but PMI is computationally **much faster** (~10× faster for the same feature space) and captures the full feature space without annotation filtering. The two methods are **complementary** — use PMI for fast discovery and coverage; use causal tracing for mechanism. Stage 3's exhaustive tracing (`03-exhaustive-mapping-and-steering.md`) bridges this gap by scaling causal tracing to cover the full feature space at one layer.

**Code reference:** `src/13_causal_circuit_tracing.py:846` → `compare_with_pmi()`.

---

### Phase 8 — Four experimental conditions battery

**Purpose:** run the full pipeline (Phases 0–7) across four experimental conditions to disentangle model architecture, SAE training data, and input cell type effects.

**The four conditions and their diagnostic roles:**

1. **K562/K562 (Geneformer, all 18 layers, K562-only SAEs).** The primary condition. Source layers {L0, L5, L11, L15}, 120 source features, 200 K562 control cells, 24,776 forward passes, **7.5 hours** compute. Produces the largest circuit graph (52,116 edges) by tracing all 17 downstream layers from L0. Provides the baseline hub architecture (L0 Golgi Organization out-degree 8,028) and the DNA-damage-response cascade (L0 DNA Repair F3717 → L6 Kinetochore, *d* = −3.47).
2. **K562/Multi (Geneformer, 4 layers, multi-tissue SAEs).** Source layers {L0, L5, L11, L17}, 90 source features, 200 K562 cells, 18,465 passes, **3.5 hours**. Same cells as condition 1 but processed through the multi-tissue SAE — isolates the **SAE training data effect**. Yields 18–21% more edges per feature than K562/K562 at most layer pairs (exception: L11→L17 where K562-only is slightly denser at 0.92 ratio). Biological coherence jumps to 68.8%. **This is the condition that reveals the SAE-lens matters more than the model.**
3. **TS/Multi (Geneformer, 4 layers, multi-tissue SAEs).** Source layers {L0, L5, L11, L17}, 90 source features, 200 Tabula Sapiens cells (67 immune + 67 kidney + 66 lung, 88 cell types total), 18,455 passes, **3.2 hours**. Same SAEs as condition 2 but completely different input cells — isolates the **input cell type effect**. Yields 3–5× sparser circuits than K562/Multi (0.32× edges/feature at L0→L5: 75 edges for K562 cells vs 236 for TS cells), weaker effect sizes (mean |d| 0.72 vs 0.98, only 10.4% |d| > 1.0 vs 34.4%) — **but identical biological coherence 68.5% vs 68.8%** (Δ = 0.3%, non-significant). Highest inhibitory dominance at **89.4%** (vs 80% K562/Multi) — processing unfamiliar cell types increases per-feature necessity. Key TS-specific circuit: L0 NADH Dehydrogenase → L5 NADH Dehydrogenase (*d* = −2.17), reflecting mitochondrial energy metabolism variance across diverse tissues.
4. **TS/Multi (scGPT, all 12 layers, natively multi-tissue SAEs).** Source layers {L0, L4, L8}, 90 source features, 200 Tabula Sapiens cells, 18,495 passes, **37.2 minutes** compute (smallest due to smaller `d_model` and fewer layers). Produces 31,380 significant edges connecting 90 → 1,960 target features (95.7% coverage — nearly complete). Enables the critical cross-model comparison with Geneformer.

**Comparison table (Table 1):**

| Metric | K562/K562 | K562/Multi | TS/Multi (GF) | TS/Multi (scGPT) |
|---|---|---|---|---|
| Model | Geneformer | Geneformer | Geneformer | scGPT |
| Layers with SAEs | 18 (all) | 4 (subset) | 4 (subset) | 12 (all) |
| Features/layer | 4,608 | 4,608 | 4,608 | 2,048 |
| Source features | 120 | 90 | 90 | 90 |
| Total edges | **52,116** | 8,298 | 5,098 | **31,380** |
| Target features | 26,338 | 4,171 | 2,962 | 1,960 |
| Target coverage | 31.9% | — | — | **95.7%** |
| Mean \|d\| | 1.05 | 0.98 | **0.72** | **1.40** |
| Median \|d\| | 0.92 | 0.87 | 0.63 | **1.19** |
| %  \|d\| > 1.0 | 41.4% | 34.4% | 10.4% | **65.2%** |
| Inhibitory % | 80.1% | 79.9% | **89.4%** | **65.5%** |
| Shared ontology | 52.9% | **68.8%** | 68.5% | 53.0% |
| Compute time | 7.5 hr | 3.5 hr | 3.2 hr | 37 min |

**Key architectural findings (Section 3.1) that emerge from the four-condition comparison:**

- **Two distinct computational architectures.** Geneformer distributes computation across 4,608 features with a **cooperative, dependency-based architecture** (80/20 inhibitory, mean |d| = 1.05 — many features each carrying moderate effects). scGPT concentrates computation in 2,048 features with **competitive dynamics** (65/35 inhibitory, mean |d| = 1.40 — fewer features carrying stronger per-feature effects). Likely driven by dimensionality (1,152 vs 512) and tokenisation (continuous vs rank-value).
- **Different hub identities reveal different organising principles** (Section 3.4). Geneformer's chromatin/RNA processing hubs are consistent with next-token prediction in rank-ordered gene lists (rewards identifying highly expressed genes). scGPT's mitochondrial electron transport hubs are consistent with masked-gene prediction using continuous values (energy metabolism genes show large expression variation across cell types, making them informative for predicting masked expression levels; Chandel 2021).

**Code references:**
- Geneformer pipeline (all three Geneformer conditions): `repos/bio-sae-circuits/src/13_causal_circuit_tracing.py` — 1,190 lines, configurable via CLI (`--source-layers`, `--n-features`, `--n-cells`, `--available-layers`)
- scGPT pipeline: `repos/bio-sae-circuits/scgpt_src/13_causal_circuit_tracing.py` — 1,172 lines

---

### Phase 9 — Systematic biological knowledge extraction (Section 2.10)

**Purpose:** annotate all 96,892 causal edges across all four conditions with biological domain labels and extract quantitative cross-model patterns.

**Steps:**

1. **Full annotation.** For every edge, look up source and target feature annotations from Stage 1 and extract: `source_label` (best GO BP term), `source_genes` (top-10 rank-weighted), `target_label`, `target_genes`. Result: **37,088 edges (38.3%) with both endpoints annotated.**
2. **Build the domain-pair meta-graph.** For each edge, create a `(source_domain, target_domain)` pair collapsing across source and target layer indices. Result: **16,067 unique domain pairs** across 1,126 unique domains.
3. **Cross-model consensus.** Compute:
   - **Geneformer unique domain pairs:** 13,698
   - **scGPT unique domain pairs:** 3,511
   - **Consensus pairs (Geneformer ∩ scGPT):** **1,142**
4. **Permutation test** (1,000 iterations): shuffle domain labels within the scGPT set and recompute overlap with the Geneformer union. Report:
   - Observed consensus: **1,142**
   - Expected from null: **107.3**
   - **Enrichment: 10.6×**
   - *p* < 0.001
5. **High-confidence filter:** 303 consensus pairs where both models show mean |d| > 1.0.
6. **Top cross-model consensus circuits:**
   - **Golgi Organization → Protein Insertion Into Membrane** (Geneformer |d| = 4.75, scGPT |d| = 5.23). The highest-ranked consensus pair overall. Reflects fundamental dependence of membrane protein biogenesis on Golgi function.
   - **Cholesterol Biosynthesis → Sterol Biosynthesis** (GF 4.44, scGPT 1.54). Universal metabolic cascade.
   - **Maturation of SSU-rRNA → RNA Methylation** (GF 1.67, scGPT 5.00). Both models learned the regulatory cascade from epitranscriptomic modification (m⁶A and related marks) to transcriptional output.
   - **NADH Dehydrogenase Assembly → Positive Regulation of TOR Signalling** (GF 1.17, scGPT 3.72). Encodes the known dependence of mTOR signalling on cellular energy status.
   - **DDR → Negative Regulation of Cell Differentiation** (GF 1.02, scGPT 3.55) and **DDR → RNA Methylation** (GF 1.02, scGPT 2.61). Extends the DNA-damage cascade to differentiation arrest and epitranscriptomic regulation.
7. **Biological process hierarchy.** Aggregate the meta-graph into a directed domain graph. Compute each domain's mean source-layer position across all edges where it appears. Expected:
   - **Early-layer processes (mean layer < 1):** MAPK Cascade (L0.1), Ras Signalling (L0.3), Epithelial Cell Differentiation (L0.3), Organonitrogen Compound Catabolism (L0.3), Histone Modification (L0.4), Cellular Response to Growth Factor (L0.6), Establishment of Protein Localization (L0.6), COPII-coated Vesicle Budding (L0.7)
   - **Late-layer processes (mean layer > 16):** Positive Regulation of Macromolecule Biosynthesis (L16.8), Regulation of Glucose Import (L17.0), Nuclear RNA Surveillance (L17.0), Regulation of Endothelial Cell Proliferation (L17.0), Protein Localization to Chromatin (L17.0), Negative Regulation of G1/S Transition (L17.0), Blood Circulation (L17.0), Regulation of Calcium Ion Transport (L17.0)
8. **Directed meta-graph summary:** **1,126 nodes and 16,002 edges**; 499 feedback loops (reciprocal A↔B); **300 DDR → cell cycle edges all with positive layer deltas (ΔL > 0)** — validates the DNA damage → mitotic checkpoint → cell cycle arrest temporal ordering. Example: DNA Repair → Mitotic Chromatid Segregation: |d| = 1.27, ΔL = +7.5; DNA Repair → Mitotic Cell Cycle Regulation: |d| = 1.02, ΔL = +5.9.
9. **Tissue-specific circuits.** Compare domain pairs unique to multi-tissue conditions (3,541 pairs) vs those shared with K562 (1,334). Compute Fisher's exact enrichment for each of several tissue categories:
   - **Immune: OR = 3.18, *p* < 0.001** (highly significant)
   - **201 immune-specific circuit pairs, 89% absent from K562**
   - Representative immune-specific circuits (Supp Note 8):
     - Cellular Response to Cytokine Stimulus → G1/S Transition of Mitotic Cell Cycle (immune activation → proliferative entry)
     - Cytokine-Mediated Signalling → Regulation of Apoptotic Process (TNF/Fas in immune homeostasis)
     - Cellular Response to TGF-β → Cellular Response to Cytokine Stimulus (TGF-β/cytokine crosstalk)
     - Homotypic Cell-Cell Adhesion → Cytokine-Mediated Signalling (immunological synapse)
     - Cytokine-Mediated Signalling → Cellular Response to ROS (respiratory burst)
   - **Blood (OR = 2.45, *p* = 0.18)** and **Kidney (OR = 1.32, *p* = 0.43)** non-significant trends (likely due to limited GO keyword coverage for these tissues).

**Code reference:** `repos/bio-sae-circuits/src/14_biological_knowledge_extraction.py` — 1,204 lines. Sub-functions:
- `step1_full_annotation()` lines 140–338 — produces `step1_annotated_edges.json` (71 MB)
- `step2_consensus_graph()` lines 340–478 — produces `step2_consensus_graph.json` with 1,142 consensus pairs
- `step3_novel_discovery()` lines 480–661 — produces `step3_novel_candidates.json` with 29,864 novel edges
- `step4_hierarchy()` lines 663–913 — produces `step4_hierarchy.json` (directed meta-graph, DDR→cell cycle temporal validation)
- `step5_celltype_circuits()` lines 915+ — produces immune-specific circuit enrichment

---

### Phase 10 — Gene-level prediction extraction

**Purpose:** convert the domain-level circuit graph into gene-pair predictions that can be tested against perturbation data.

**Steps:**

1. **For each annotated edge** (47,418 edges with gene annotations on both endpoints): extract **top 10 genes per feature** from the rank-weighted top-20 in `feature_catalog.json` (Stage 1). This yields a Cartesian product of source_genes × target_genes per edge = up to 100 gene pairs per edge.
2. **Raw gene-pair count:** ~**2.58 million** raw pairs across all 47,418 edges.
3. **Filter by evidence strength:**
   - **Evidence ≥ 2:** the same gene-pair prediction must appear in ≥ 2 independent edges (i.e., in two different causal circuit connections). This filters out one-off noise.
   - **OR |Cohen's *d*| > 2:** strong single-edge evidence (very large effect sizes) is retained even without evidence ≥ 2.
4. **Post-filter result: 975,369 gene-pair predictions** (from 2.58M raw). Distribution:
   - 32.8% derive from **cross-model consensus** edges (present in both Geneformer and scGPT).
5. **Compare to known biology:**
   - **0.15%** of predicted pairs match established PPI (STRING ≥ 700) or regulatory (TRRUST v2) relationships.
   - **1.14%** share ≥ 2 GO BP terms (the "reference graph" of "known biology").
   - **98.7%** connect genes **without established direct relationships**. This "novel" rate is both (a) the interesting hypothesis-generation space and (b) a reminder that SAE feature gene lists are derived from co-expression patterns, so the novel rate reflects co-expression-derived pairs that rarely appear in curated interaction databases — **novel ≠ causal**.
6. **Reference graph construction:** connect GO BP terms that share ≥ 3 genes → **14,021 known domain links** used as the "known biology" baseline. Of the 37,088 annotated causal edges, **29,864 (80.5%) connect domain pairs NOT in the reference graph** — representing candidate novel relationships.
7. **Top novel relationships (all 4 conditions):**
   - **NADH Dehydrogenase Assembly → Protein Transport** (|d| = 7.21, 20 shared genes) — mitochondrial energy status causally influences protein trafficking and translation.
   - **Golgi Organization → ER Stress Response** (|d| = 6.29, 43 shared genes including CALR, HSPA5, P4HB, PDIA3, PDIA6) — secretory pathway disruption directly triggers ER stress, beyond the traditional unfolded protein response framework (Hetz 2012).
   - **Aerobic ETC → Mitochondrial Translation** (|d| = 4.67, 22 shared genes).

**Code reference:** `repos/bio-sae-circuits/src/15_phase6_predictions_and_validation.py:74` → `step1_gene_predictions()`. Output: `step1_gene_predictions.json` (~134 MB).

---

### Phase 11 — CRISPRi directional validation

**Purpose:** test whether circuit-derived gene-pair predictions match the direction and magnitude of Replogle K562 genome-scale CRISPRi responses — the critical ground-truth test.

**Steps:**

1. **Load Replogle K562 CRISPRi screen** (Replogle et al. 2022): 643,413 cells, 2,023 knockdown targets. For each target, compute **pseudobulk log-fold-change** of every other gene (perturbed cells vs non-targeting controls).
2. **Find the overlap** between circuit source genes and Replogle knockdown targets. Result: **599 of 2,023 targets** (≈ 30%) overlap circuit source genes.
3. **Build the validation set:** for each of the 599 overlap source genes × each of their circuit-predicted target genes → **282,250 gene pairs** to validate.
4. **Directional accuracy.** For each predicted `(source, target)` pair with |d| > 0.5:
   - The pipeline's prediction: the ablation direction (inhibitory vs excitatory) at the feature level.
   - The CRISPRi ground truth: sign of the target gene's log-fold change under source knockdown.
   - **Sign agreement:** does ablation-induced reduction match knockdown-induced reduction?
   - **Result: 56.4%** (vs 50% chance).
   - **Statistically significant due to sample size** (282,250 pairs) but biologically near-null.
5. **Magnitude correlation.** Spearman ρ between predicted |d| and actual |LFC|:
   - **ρ = 0.038**, *p* < 10⁻³² (statistically significant but biologically negligible)
6. **Per-target enrichment test.** For each of the 599 source genes, test whether circuit-predicted targets are enriched for CRISPRi-responsive targets via Fisher's exact test. Only **36 targets (6.0%)** show nominally significant enrichment (uncorrected *p* < 0.05).
7. **Cross-model consensus comparison.** Predictions from cross-model consensus circuits achieve marginally better directional accuracy: **57.3% vs 56.4%** — a +0.9pp improvement that does not rescue the overall null.
8. **Critical interpretation (Section 3.5 of paper, the "gene-level validation bounds predictive utility" block):**
   - **The contrast between domain-level and gene-level results crystallises what these models have and have not learned.**
   - At the **domain level**, the circuit graph is a reliable **map of biology**: it correctly identifies which biological processes relate to which, recovers temporal ordering (Phase 9 step 8), and converges across two independent models (Phase 9 cross-model consensus, 10.6× enrichment).
   - At the **gene level**, this map does not translate into reliable mechanistic predictions. Gene pairs drawn from different features are NOT independent regulatory hypotheses.
   - **These models know WHICH processes connect to WHICH, but NOT which specific genes causally drive which — consistent with the companion Stage 1 study (6.2% TF specificity, see `01-sae-atlas.md` Phase 8) demonstrating co-expression, not causal regulation, is the primary information encoded.**

**Caveats for gene-level validation** (paper §3.7 Limitations):
- **CRISPRi pseudobulk log-fold changes measure steady-state expression changes**, not immediate transcriptional responses. This may dilute true causal signals that are transient (recovered within the CRISPRi assay window).
- **Gene pairs inherit a co-expression basis**: SAE feature gene lists are derived from co-activation, so the Cartesian product of source × target genes is NOT a set of independent regulatory hypotheses — many pairs within a single edge reflect the same underlying co-expression pattern.
- **Pseudobulk vs single-cell.** Future work should test predictions against immediate transcriptional responses (nascent-RNA timepoint assays) rather than steady-state pseudobulk.

**Code reference:** `repos/bio-sae-circuits/src/15_phase6_predictions_and_validation.py:312` → `step2_perturbation_validation()`. Directional accuracy at lines 430–491; magnitude correlation at lines 494–524.

---

### Phase 12 — Disease-relevant circuit mapping

**Purpose:** test whether disease-relevant biological processes occupy hub positions in the circuit graph and whether they are preferentially conserved across models.

**Steps:**

1. **Define 11 disease categories** via keyword matching on GO BP terms (Table 10):
   - Transcription regulation: 739 domains, 28,155 circuit edges, 7,046 consensus
   - Immune response: 660 / 27,071 / 7,305
   - Apoptosis: 631 / 25,622 / 6,614
   - Cell cycle (cancer): 577 / 25,180 / 6,610
   - Protein quality control: 576 / 21,896 / 6,256
   - DNA damage/repair: 532 / 22,233 / 6,010
   - Oncogenic signalling: 513 / 15,921 / 3,826
   - Metastasis/migration: 461 / 12,850 / 3,601
   - Metabolism (cancer): 231 / 8,775 / 3,724
   - Angiogenesis: 243 / 4,066 / 1,302
   - TRRUST TFs: 152 / 2,510 / 958
2. **Fisher's exact enrichment** per circuit domain against each disease category.
3. **Centrality test.** Compute per-domain circuit edge count (in + out degree). **Disease-associated domains have median 14 circuit edges vs 3 for non-disease domains (Mann-Whitney *p* = 1.2 × 10⁻¹¹)** — disease-relevant processes are significantly more central.
4. **Cross-model conservation test.** **Disease-relevant circuit pairs are 3.59× more likely to be cross-model consensus pairs** (*p* < 0.001, Fisher's). The processes most relevant to human disease are also the processes most robustly encoded across model architectures.
5. **Critical caveat** (Section 3.5 caveats): keyword matching is broad — 1,073 of 1,126 domains (95%) match at least one disease category. The absolute counts overstate disease specificity. The **relative centrality** and **cross-model conservation** are the informative findings.

**Code reference:** `repos/bio-sae-circuits/src/15_phase6_predictions_and_validation.py:619` → `step3_disease_mapping()`. Output: `step3_disease_mapping.json`.

---

### Phase 13 — Cross-model architecture comparison and final reporting

**Purpose:** synthesise the four-condition results into the architecture-level comparison that is the paper's central scientific contribution.

**Steps:**

1. **Compile the four-condition summary table** (Table 1 of paper, reproduced in §6 Phase 8).
2. **Identify model-specific vs universal properties:**
   - **Universal (invariant across architectures)** — suggesting fundamental aspects of how transformers process biological data:
     - ~53% shared ontology coherence (Geneformer K562/K562 52.9%, scGPT 53.0%)
     - Inhibitory dominance (65–89% across all conditions)
     - Early-layer feature persistence (L0 effects reach all downstream layers)
     - Cross-model consensus at 10.6× enrichment
   - **Model-specific (divergent properties)** — revealing architecture-specific computation:
     - Effect magnitude (mean |d|: GF 1.05 vs scGPT 1.40)
     - Inhibitory ratio (GF 80/20 vs scGPT 65/35)
     - Hub identity (GF chromatin/RNA vs scGPT mitochondrial energy)
     - Integration layer location (Geneformer L16 convergence vs scGPT L4 mid-layer funnel)
3. **Document the four implications for mechanistic interpretability of biological models** (Section 3.6):
   1. **Feature-level analysis reveals structure invisible to component-level analysis.** Companion study (`../attention-grn-extraction-and-evaluation.md`) found ablating attention heads or MLP layers produces null behavioural effects; SAE circuit tracing reveals the same models have dense, biologically coherent computational graphs **at the feature level**. Features, not components, are the natural unit of biological computation in these models.
   2. **Causal tracing and statistical co-activation are complementary.** The 91–95% target overlap between PMI and causal edges validates both methods. PMI is faster and covers the full feature space; causal tracing adds directionality, magnitude, and sign.
   3. **The SAE lens matters as much as the model.** Multi-tissue SAEs improve biological coherence by 16 percentage points while keeping the model unchanged (Phase 6 finding). The choice of SAE training data substantially affects interpretability conclusions, independent of the model being interpreted.
   4. **Cross-model comparison is essential.** Properties invariant across models (53% coherence, inhibitory dominance, early-layer persistence) likely reflect universal aspects of how transformers process biological data. Properties that differ reveal architecture-specific computation invisible in single-model studies.
4. **Emit the final report** (`circuit_tracing_report.json`) combining:
   - Four-condition statistics table (Table 1)
   - Hub feature rankings per condition (Tables 3, 7)
   - Biological coherence fractions
   - PMI target overlap (Table 5)
   - Cross-model consensus summary (Table 9 rows 1–3)
   - Gene-level prediction count (975,369)
   - CRISPRi validation: 56.4% directional accuracy, ρ = 0.038
   - Disease centrality (Mann-Whitney *p* = 1.2 × 10⁻¹¹, 3.59× consensus enrichment)
5. **Write-up of per-condition biological cascades** (from Tables 4, 8 and Supp Notes 1–8). Representative examples to include for human verification:
   - **Geneformer DDR cascade** (Figure 2 of paper): L0 DNA Repair (F3717) → L1 DNA Damage Response (*d* = −1.87, 113 shared terms) → L5 DNA Damage Response (*d* = −3.84, 72 shared) → L11 G2/M Transition (*d* = −2.66, 75 shared) → L17 G2/M Transition. Plus "skip connection" L0 → L17 (*d* = −2.30, 65 shared). Biological progression: DNA damage detection → chromatin remodelling → cell cycle arrest (Ciccia & Elledge 2010; Jackson & Bartek 2009).
   - **Geneformer neurodevelopment-proteostasis cascade** (Supp Note 2, Figure 4): L0 Nervous System Development (F146) is the most biologically connected hub, appearing in 7 of top 10 circuits. Drives L1 Endosome Organization (*d* = −1.32, 142 shared terms), L2 Proteasomal Catabolic Process (*d* = −0.96, 140), L6 Protein Catabolism (*d* = −1.27, 139), L13 Golgi Vesicle Transport (*d* = −0.81, 141). Recapitulates neuronal vulnerability to disrupted protein homeostasis (Ross & Poirier 2004; Labbadia & Morimoto 2015).
   - **scGPT protein quality control cascade** (Supp Note 3, Figure 6, Table 8): L0 Protein Catabolism (F507) → L1 Chromatin Organization (*d* = −8.19, 161 shared terms — the strongest individual edge in any condition) — mediated by stress response and beta-catenin degradation pathways (Nusse & Clevers 2017). Plus L0 Protein Catabolism → L2 Chemical Stress Response (*d* = −3.12), L4 DNA Metabolism (*d* = −6.10, ER-phagosome crossover), L10 Protein Catabolism (*d* = −1.95, long-range maintenance). Complete cascade: protein quality control → stress detection → DNA repair → biosynthetic recovery (Walter & Ron 2011; Hetz 2012).
   - **scGPT ubiquitin-proteasome persistence:** L0 Proteasome (F379) → L9 Proteasome (*d* = −2.50, 141 shared terms). This cross-layer persistence demonstrates protein degradation status is tracked throughout scGPT's computation.

**Code reference:** `repos/bio-sae-circuits/paper/generate_phase4_figures.py` (figure generation), `paper/generate_phase5_figures.py` (consensus graph and disease enrichment heatmaps).

---

## 7. Parameters

| Parameter | Default | Alternate / sensitivity | Used in |
|---|---|---|---|
| `n_cells` (per condition) | **200** | 100–1,000 (sensitivity limited by compute) | Phase 0 |
| `n_source_features_per_layer` | **30** | 10–120 | Phase 1 |
| `annotation_quality_score` | sum of `−log₁₀(p)` over significant enrichments | — | Phase 1 |
| `cohens_d_threshold` | **|d| > 0.5** (Cohen's "medium") | {0.3, 0.5, 1.0} for sensitivity | Phase 4 |
| `consistency_threshold` | **> 0.7** | {0.6, 0.7, 0.8} for sensitivity | Phase 4 |
| `welford_algorithm` | online mean/variance (Welford 1962) | — | Phase 3 |
| `source_layers_geneformer_k562_k562` | **{L0, L5, L11, L15}** | — | Phase 8 condition 1 |
| `source_layers_geneformer_multi` | **{L0, L5, L11, L17}** | multi-tissue SAEs available only here | Phase 8 conditions 2, 3 |
| `source_layers_scgpt` | **{L0, L4, L8}** | — | Phase 8 condition 4 |
| `clean_forward_cache` | **per-cell** | non-negotiable for compute | Phase 2 |
| `checkpoint_interval_cells` | **50** | — | Phase 3 (resume) |
| `pmi_comparison_layer_pairs` | **L0→L5, L5→L11, L11→L17** (Geneformer); L0→L4, L4→L8, L8→L11 (scGPT) | — | Phase 7 |
| `permutation_iterations_consensus` | **1,000** | — | Phase 9 cross-model consensus |
| `high_confidence_threshold` | **|d| > 1.0 in both models** | — | Phase 9 |
| `gene_prediction_top_n` | **top 10 genes per feature** (rank-weighted) | — | Phase 10 |
| `gene_prediction_evidence_filter` | **≥ 2 edges OR |d| > 2** | — | Phase 10 |
| `crispri_perturbation_threshold` | **|d| > 0.5** for predicted edges | — | Phase 11 |
| `disease_categories` | **11** (Table 10, keyword-matched) | — | Phase 12 |
| `reference_graph_shared_genes` | **≥ 3** shared genes in GO BP terms | — | Phase 10 (novel detection) |
| `flash_mha_conversion` | `W_qkv → in_proj_weight` | required for scGPT on MPS | Phase 0 |
| `device` | **MPS** (Apple Silicon) or CUDA | — | all phases |
| `mps_fallback_env` | **`PYTORCH_ENABLE_MPS_FALLBACK=1`** | required for some ops on MPS | Phase 0 |

**Significance threshold sensitivity** (Limitation 2): doubling the |d| threshold from 0.5 to 1.0 retains 41.4% of edges in Geneformer K562/K562 and 65.2% in scGPT TS/Multi; tripling to 2.0 retains 4.3% and likely higher for scGPT. Qualitative findings (inhibitory dominance, hub features, cross-model consensus, biological coherence rate) are stable across these thresholds.

## 8. Validation

A successful pipeline run must reproduce the following sanity signatures before any new-model claim is trusted:

1. **K562/K562 Geneformer primary condition** (Phase 8 condition 1): exactly **52,116 significant edges**, 120 source → 26,338 targets, **31.9% coverage**, mean |d| = **1.05**, median 0.92, **41.4% |d| > 1.0**, **80.1% inhibitory**, **52.9% shared ontology**, **7.5 hours** compute. Tolerance ±2% on counts; deviations > 5% indicate threshold or algorithm differences.
2. **scGPT TS/Multi** (Phase 8 condition 4): **31,380 edges**, 90 → 1,960, **95.7% coverage**, mean |d| = **1.40**, median 1.19, **65.2% |d| > 1.0**, **65.5% inhibitory**, 53.0% shared ontology, **37.2 minutes**.
3. **Top hub feature, Geneformer K562/K562** (Phase 5): **L0_F2905 Golgi Organization, out-degree 8,028**. If top hub has out-degree > 9,000 or < 7,000, the graph is under- or over-retained; verify significance thresholds.
4. **Top hub feature, scGPT** (Phase 5): **L4_F446 Endonucleolytic Cleavage (rRNA), out-degree 6,494** — **the single highest-connectivity feature across either model**. This is a required signature for correct scGPT SAE loading.
5. **Causal effect attenuation, Geneformer** (Phase 5): L0 features maintain ~200–215 significant edges/feature for the first 5 layers (L1–L5), with a slight increase at L3, then decay. At L17 (17 layers downstream), still 58 edges/feature. L15 → L17 effects should *increase* (279 → 336). If L17 effects from L0 drop to < 20 edges/feature, the ablation is losing signal too quickly — check that the ablation delta is being applied to the original hidden state (not the reconstruction).
6. **PMI target overlap, Geneformer K562/K562** (Phase 7): L0 → L5 **90.6%**, L5 → L11 **94.8%**, L11 → L17 **91.7%**. Zero source feature overlap (Phase 7 step 5). If target overlap < 80%, either the PMI graph or the causal graph is wrong; rerun Phase 3 and Stage 1 Phase 5.
7. **Biological coherence universal constant** (Phase 6): Geneformer K562/K562 ≈ **52.9%**, scGPT TS/Multi ≈ **53.0%** — these must be within ±2% of each other. Multi-tissue conditions: **68.8% (K562/Multi) ≈ 68.5% (TS/Multi)** within ±1%. **The K562/Multi vs TS/Multi equivalence is the key diagnostic** — if it deviates by > 5%, the pipeline is measuring a cell-type effect instead of an SAE-lens effect, which means the multi-tissue SAE is not actually trained on pooled data.
8. **Cross-model consensus** (Phase 9): exactly **1,142 consensus domain pairs**, **10.6× enrichment** over null (expected 107.3), permutation *p* < 0.001 at 1,000 iterations. **303 high-confidence pairs** with |d| > 1.0 in both models. Tolerance ±5% on consensus count.
9. **Inhibitory dominance pattern**: K562/K562 **80.1%**, K562/Multi 79.9%, **TS/Multi (Geneformer) 89.4%** (highest), scGPT 65.5%. The TS/Multi > K562/Multi ordering is required (processing unfamiliar cell types increases per-feature necessity).
10. **Gene-level prediction counts** (Phase 10): **2.58M raw → 975,369 post-filter**. **0.15% match STRING/TRRUST**. **98.7% novel**. **32.8% from cross-model consensus**.
11. **CRISPRi directional accuracy** (Phase 11): **56.4%** on 282,250 gene pairs across 599 overlap targets. **ρ = 0.038** magnitude correlation. **6.0% of targets show nominally significant enrichment**. Cross-model consensus 57.3% (marginally better, +0.9pp). Any result > 60% directional accuracy is suspicious — check TRRUST target list construction and Replogle pseudobulk computation.
12. **Disease centrality** (Phase 12): disease-associated domains median **14 edges vs 3 non-disease** (Mann-Whitney *p* = **1.2 × 10⁻¹¹**). **3.59× more likely** to be cross-model consensus pairs (*p* < 0.001). If the Mann-Whitney *p* is not significant, either the disease keyword lists are too broad (> 95% of domains matching) or the circuit graph is too sparse.
13. **Compute budget** (Phase 8 totals): **14.8 hours total wall-clock** across all four conditions on a single Apple Silicon M2 Max, **80,191 total forward passes**. ±20% tolerance for hardware differences. A naive implementation without the clean forward cache takes ~20 days; if your runtime exceeds 24 hours per condition, the cache is not engaged.
14. **DNA damage cascade signature** (Phase 13 step 5): in Geneformer K562/Multi conditions, the L0 DNA Repair → L5 Damage Response → L11 G2/M → L17 G2/M cascade must appear with |d| values in the −1.8 to −3.8 range. The **300 DDR → cell cycle edges with positive ΔL** (Phase 9 step 8) is a required signature of correct temporal ordering.
15. **scGPT strongest individual edge** (Phase 13 step 5): L0 Protein Catabolism (F507) → L1 Chromatin Organization, **|d| = 8.19, 161 shared ontology terms**. If this edge is not the top-ranked scGPT individual edge, the scGPT SAE loading or ablation path is wrong.

## 9. Known pitfalls

1. **Clean forward cache is non-negotiable.** A naive implementation (forward pass per feature × cell) takes ~20 days. If your runtime exceeds 24 hours for a single condition, you have not built the cache correctly. Verify that the outer loop is over cells (not features) and that one clean forward pass is computed per cell before the feature ablation loop begins.
2. **Welford's algorithm is non-negotiable.** Storing per-cell deltas for 120 source × 18 target layers × 4,608 target features × 200 cells consumes ~40 GB per condition. Welford maintains a running mean and variance without storing the per-cell data. Use the reference implementation at `src/13_causal_circuit_tracing.py:180`.
3. **Ablation must add the delta to the original hidden state, not overwrite with the reconstruction.** Phase 3 step 3: `h^(L_src)_abl = h^(L_src)_clean + δ` where `δ = h̃_ablated − h̃_original`. If you overwrite with `h^(L_src)_abl = h̃_ablated` directly, you lose the ~20% of variance the SAE does not explain, producing spurious effects unrelated to the target feature. This is the single most common bug in SAE ablation implementations.
4. **Feature IDs are ephemeral across SAE training runs.** The Stage 1 SAE training is non-deterministic at TopK ties, so feature indices can shift between runs of the same seed. **Always match features across pipelines (Stage 1 → Stage 2 → Stage 3) by decoder cosine similarity, not by feature ID.** Use a threshold of 0.9 for matching.
5. **FlashMHA kernels break scGPT ablation on MPS.** Convert `W_qkv → in_proj_weight` per Phase 0 step 2 of the target-model adapter. Source paper notes this conversion explicitly (§4.2 Methods).
6. **HuggingFace `output_hidden_states` indexing is off-by-one.** For Geneformer BertForMaskedLM, `outputs.hidden_states[ℓ + 1]` is the output of layer `ℓ`, NOT `hidden_states[ℓ]` (which is the input embedding). This off-by-one is a common bug that produces results that look plausible but are actually measuring effects one layer earlier than intended.
7. **Source feature selection by annotation quality creates annotation bias.** The 30 features per layer are selected by having many significant ontology enrichments, which biases toward well-studied biology. The paper's Limitation 1 notes results may not generalise to unannotated features. **Stage 3's exhaustive tracing directly addresses this** by tracing all 4,065 active features at one layer without annotation filtering; 40% of the top-20 hubs in that exhaustive trace turn out to be unannotated features that selective tracing would have missed entirely.
8. **Single-feature ablation misses higher-order interactions.** Phase 3 tests only `feature_f → zero`. If ablating feature A and feature B jointly has an effect substantially different from the sum of individual effects (synergy or redundancy), this pipeline cannot detect it. **Stage 3 three-way combinatorial ablation directly addresses this** (and finds zero synergy at third order, 94.1% subadditive).
9. **Cell count (200) limits statistical power for weak effects.** The 200-cell default is a compute/power tradeoff. Phase 4 thresholds (|d| > 0.5, consistency > 0.7) are calibrated for this sample size. If you scale to 500–1,000 cells for higher sensitivity, increase the consistency threshold to > 0.75 to avoid false positives.
10. **Multi-tissue SAE layer availability.** Stage 1's multi-tissue SAEs are available only at {L0, L5, L11, L17} for Geneformer, restricting Phase 8 conditions 2 and 3 to those layers. You cannot compare K562/K562 (all 18 layers) to K562/Multi (4 layers) on an edge-count basis directly — compare per-source-layer statistics normalised by downstream-layer count.
11. **scGPT K562/K562 was NOT run.** The source paper performed scGPT circuit tracing only in TS/Multi. A K562/K562 scGPT condition would enable cleaner isolation of cell-type effects. If you add it for a new experiment, be aware that scGPT's continuous-value tokens require preserving original expression values through the forward cache (see Stage 1 Phase 0 critical pitfall on scGPT expression values).
12. **Biological coherence depends on annotation database completeness.** 53% shared ontology is partly an artefact of the GO/KEGG/Reactome/STRING/TRRUST structure. Novel or poorly annotated biology is invisible to this metric.
13. **Gene-level predictions inherit a co-expression basis.** Phase 10 extracts gene pairs from SAE feature top-gene lists, which are derived from co-activation. The Cartesian product across a feature pair is NOT a set of independent regulatory hypotheses. The 98.7% "novel" rate reflects this co-expression derivation, NOT genuine novel regulatory discovery.
14. **CRISPRi pseudobulk validation dilutes causal signals.** Replogle log-fold-changes measure steady-state expression changes, not immediate transcriptional responses. A causal signal that is transient or compensated for within the CRISPRi assay window will appear as null in the validation. Future work should test against nascent-RNA timepoint assays.
15. **Disease gene set definitions use broad GO keyword matching.** "Transcription regulation" captures 739 of 1,126 domains. The **absolute counts overstate specificity**; relative centrality and cross-model enrichment are the informative findings.
16. **Permutation null for cross-model consensus.** The 1,000-permutation test shuffles domain labels within the scGPT set and recomputes overlap with the Geneformer union. For very small consensus sets (< 50 pairs), this null may under-estimate variance; in that case increase to 5,000 permutations.

## 10. Quick-start for new-model evaluation

To apply Stage 2 to a new single-cell foundation model `NEWMODEL` with a trained Stage 1 atlas:

1. **Verify Stage 1 completion.** For `NEWMODEL`:
   - All-layer SAEs trained and saved (`sae_final.pt` per layer).
   - Feature annotations complete (`feature_annotations.json` with ≥ 30 features per candidate source layer having ≥ 1 significant enrichment).
   - Activation memmaps cached (Stage 1 Phase 0 output).
2. **Implement the Phase 0 ablation adapter** for `NEWMODEL`. The contract: given source layer `L_src`, replacement hidden state `h^(L_src)_abl`, and an input cell, return per-layer hidden states for all `ℓ > L_src`. Choose between the Geneformer-style (full forward + hook replacement) or scGPT-style (manual layer iteration) template.
3. **Run Phase 1 source feature selection** at 2–4 candidate source layers. Target: 30 well-annotated features per layer. Verify annotation score distribution is sufficiently wide to select top 30.
4. **Run a single-cell smoke test.** Build the Phase 2 clean forward cache for 1 cell. Ablate 1 source feature. Verify downstream deltas are non-zero at immediately adjacent layers and decay with distance. If deltas are zero, the ablation is not applied correctly (Pitfall #3).
5. **Run Phase 3 on 10 cells × 5 features as a sensitivity check.** Compute Welford statistics. Verify at least some edges exceed |d| > 0.5 AND consistency > 0.7. If zero significant edges emerge, either the thresholds are too strict for the new model (lower to |d| > 0.3 and consistency > 0.6 for diagnostic) or the SAE reconstruction is too coarse.
6. **Run Phase 3 full pipeline for all 200 cells × source features at one source layer.** Estimate compute time and extrapolate to the full condition.
7. **Run Phase 4 and Phase 5** to build the circuit graph and verify hub structure. Top hub out-degree should be substantial fraction (≥ 50%) of total target features. If the top hub out-degree is < 500, the SAE may be under-expressive — consider retraining with larger dictionary (8× instead of 4×).
8. **Run Phase 6 biological coherence analysis.** Report the coherence fraction. If < 40%, either the SAE training data is too narrow (consider multi-tissue) or the model's internal computation is not well-structured by ontology terms.
9. **Run Phase 7 PMI validation against Stage 1 PMI graphs.** Target overlap must be > 80%. If < 60%, the causal graph is measuring something different from co-activation, which warrants investigation.
10. **Run Phase 8 across as many experimental conditions as Stage-1 multi-tissue SAEs allow.** For a new model without multi-tissue SAEs, you can run only the equivalent of K562/K562 — in that case, report the single-condition result and flag the lack of SAE-lens vs cell-type disentanglement.
11. **Run Phase 9 cross-model consensus against an existing pipeline output** (Geneformer K562/K562 or scGPT TS/Multi). For a new model to be "conserved" with Geneformer/scGPT, the consensus enrichment should be > 5× with *p* < 0.001. Lower enrichment indicates architecture-specific computation.
12. **Run Phase 10 gene-level prediction extraction** and **Phase 11 CRISPRi validation**. The Replogle K562 screen is the ground-truth benchmark. Report directional accuracy and magnitude Spearman ρ. **A directional accuracy > 60% with ρ > 0.1 would be the first single-cell foundation model whose SAE circuits encode meaningful gene-level regulatory predictions** — this would be a publishable positive result.
13. **Run Phase 12 disease mapping** only if Phase 11 validates to > 55%. Below that, disease-relevant circuit "findings" cannot be trusted as regulatory hypotheses.
14. **Run Phase 13 comparison and final reporting.** Include the four-condition summary table and the per-condition cascade examples.
15. **Proceed to Stage 3** (`03-exhaustive-mapping-and-steering.md`) to break the three specific limitations of this pipeline: (a) annotation bias in source feature selection (→ exhaustive tracing), (b) single-feature ablation missing higher-order interactions (→ three-way combinatorial ablation), and (c) observational trajectory dynamics (→ trajectory-guided feature steering with causal control over cell-state direction).
16. **Cross-reference with the companion attention pipeline** (`../attention-grn-extraction-and-evaluation.md`) and Stage 1 (`01-sae-atlas.md`). For any scientific claim about whether `NEWMODEL` encodes regulatory logic, the three pipelines (attention, SAE atlas, SAE circuits) must be reconciled. The central finding across all three so far — for Geneformer and scGPT — is consistent: **single-cell foundation models encode co-expression structure and pathway membership, not directed TF→target causal regulatory wiring**.
