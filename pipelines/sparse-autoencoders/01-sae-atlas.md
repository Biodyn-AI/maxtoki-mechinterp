# Stage 1 — SAE Atlas Construction for Single-Cell Foundation Models

## 1. Overview

This pipeline decomposes a single-cell transcriptomic foundation model's **residual-stream activations** into an overcomplete dictionary of interpretable features via per-layer **TopK sparse autoencoders (SAEs)**, then systematically characterises those features along twelve complementary axes — ontology annotation, SVD superposition quantification, cross-layer tracking, co-activation modularity, causal specificity, cross-layer computational graphs, perturbation response, multi-tissue robustness, unannotated-feature guilt-by-association, cell-type enrichment, feature-space geometry, and interactive web deployment.

The pipeline is **model-agnostic**: a Phase 0 adapter must expose per-layer residual-stream activations from any transformer-based single-cell foundation model. All downstream phases consume a standard `[n_positions, d_model]` memory-mapped activation tensor per layer plus the trained TopK SAE. Worked examples in the source paper: **Geneformer V2-316M** (18 layers, `d=1,152`; 82,525 alive features across 18 SAEs) and **scGPT whole-human** (12 layers, `d=512`; 24,527 alive features across 12 SAEs) — together **~107,000 features across 30 layers**.

The central scientific question this pipeline answers — and the question Stage 1 of the SAE mega-pipeline was designed to answer — is: **does the trained foundation model encode causal gene regulatory logic, or only statistical co-expression structure?** The answer in the source paper is a clear negative: attention-based analyses in the companion study (arXiv:2602.17532, see `../attention-grn-extraction-and-evaluation.md`) established that attention edges carry no incremental predictive value beyond gene-level features; Stage 1 establishes the **same boundary at the SAE feature level** — the pipeline detects 92% of CRISPRi perturbations at the feature level (features respond when genes are knocked down) but only **6.2% of 48 TRRUST transcription factors produce feature responses specific to their known regulatory targets**. This is the unmet success criterion that defines what the model knows and does not know. A Stage 1 atlas that achieves high perturbation specificity would be the first single-cell foundation model whose internal representations encode directed TF→target wiring rather than co-expression.

Stage 1 produces the **foundation** for Stages 2 and 3 of the SAE mega-pipeline. The trained SAE weights, feature annotation tables, co-activation graphs, and activation caches emitted here are direct inputs to the causal circuit tracing of Stage 2 (`02-causal-circuit-tracing.md`) and the exhaustive mapping, combinatorial ablation, and trajectory-guided steering of Stage 3 (`03-exhaustive-mapping-and-steering.md`). See `README.md` in this directory for the mega-pipeline guide.

## 2. Source

- **Paper:** Anonymous, (2026). *Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT.* arXiv:2603.02952 [q-bio.GN], 3 Mar 2026. 28 pages (15 pages main + 5 pages references + 8 pages supplementary material with 10 tables and 4 figures).
- **Local PDF:** `../../references/2603.02952_anon_SAE_atlas.pdf`
- **GitHub repo:** https://github.com/Biodyn-AI/bio-sae — pinned commit `9277f5daec682cd34027172b04993833dfeb17e1` (2026-04-14, "Add BMC Genomics revision: reviewer-driven experiments and manuscript update"). Local clone at `../../repos/bio-sae/`.
- **Interactive web atlases (live):** https://biodyn-ai.github.io/geneformer-atlas/ (82,525 features × 18 layers), https://biodyn-ai.github.io/scgpt-atlas/ (24,527 features × 12 layers). Source: https://github.com/Biodyn-AI/geneformer-atlas, https://github.com/Biodyn-AI/scgpt-atlas.
- **Companion papers** (direct extensions, already in this repo):
  - `../attention-grn-extraction-and-evaluation.md` — attention-based interpretability audit (the prior work, arXiv:2602.17532). The feature-level perturbation specificity null (6.2%) in this Stage 1 pipeline and the component-level null in the companion paper are two ends of the same boundary.
  - `02-causal-circuit-tracing.md` — Stage 2 of the SAE mega-pipeline, directly consuming the atlas this pipeline produces.
  - `03-exhaustive-mapping-and-steering.md` — Stage 3.
- **Key foundational references:** Makhzani & Frey 2013 (k-sparse autoencoders), Sharkey et al. 2022 / Cunningham et al. 2023 / Bricken et al. 2023 / Templeton et al. 2024 (SAEs for language model interpretability), Gao et al. 2024 (TopK SAE architecture used here), Elhage et al. 2022 (superposition hypothesis), Olshausen & Field 1997 (sparse coding in V1), Theodoris et al. 2023 (Geneformer), Cui et al. 2024 (scGPT), Replogle et al. 2022 (genome-scale Perturb-seq used as ground truth), Tabula Sapiens Consortium 2022, Han et al. 2018 (TRRUST v2), Szklarczyk et al. 2023 (STRING), Ashburner et al. 2000 (Gene Ontology), Kanehisa & Goto 2000 (KEGG), Jassal et al. 2020 (Reactome), Traag et al. 2019 (Leiden), Manning & Schütze 1999 (PMI), Subramanian et al. 2005 (GSEA pattern).

## 3. Inputs

### 3.1 Target model (to be evaluated)

A trained single-cell transcriptomic transformer that exposes **per-layer residual-stream activations**. Contract:
1. A tokenizer mapping a cell's expression vector to a gene-indexed token sequence. Examples: Geneformer rank-value tokens (expression rank, descending); scGPT continuous-value gene tokens (expression floats alongside gene IDs).
2. A forward pass returning hidden states at every transformer block output. For HuggingFace models: `output_hidden_states=True`. For scGPT: `TransformerEncoderLayer` forward hooks on the post-residual output of each block.
3. A stable `gene ↔ token_id` mapping persisted alongside the model.
4. A fixed maximum sequence length. Geneformer V2-316M: 2,048 tokens (rank-value). scGPT whole-human: 1,200 tokens with padding/truncation.

**Worked examples in source paper:**
- **Geneformer V2-316M** (Theodoris et al. 2023) — 18 layers, `d_model = 1,152`, 18 attention heads per layer, pre-trained on ~100M single-cell transcriptomes. HuggingFace: `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M`.
- **scGPT whole-human** (Cui et al. 2024) — 12 layers, `d_model = 512`, 8 attention heads per layer, pre-trained on ~33M cells. Vocabulary ~60,700 gene tokens. Weights from the scGPT authors.

### 3.2 Single-cell datasets

| Dataset | Source | Role in pipeline |
|---|---|---|
| **Replogle K562 CRISPRi** (Replogle et al. 2022) | scPerturb / Figshare DOI 10.6084/m9.figshare.21452470 | Primary SAE training corpus (2,000 control cells for Geneformer extraction → ~4M token positions/layer); ground truth for Phase 8 perturbation response mapping (100 CRISPRi targets × 20 perturbed cells each). |
| **Tabula Sapiens** (Jones et al. 2022 / Tabula Sapiens Consortium) | https://tabula-sapiens-portal.ds.czbiohub.org/ and CZ CELLxGENE | Primary scGPT training corpus (3,000 cells: 1,000 immune across 43 cell types + 1,000 kidney + 1,000 lung); Phase 9 multi-tissue control (500K positions pooled with 500K K562); Phase 11 cell-type enrichment (56 cell types across three tissues). |

### 3.3 Biological annotation databases

| Database | Version / source | Phase in pipeline |
|---|---|---|
| **Gene Ontology Biological Process** (Ashburner et al. 2000) | http://geneontology.org/ | Phase 2 feature annotation (145,106 total enrichments across 18 Geneformer layers — the dominant ontology source in GO BP peaks at layers 0–5 with ~10k/layer) |
| **KEGG** (Kanehisa & Goto 2000) | Nucleic Acids Research D1 | Phase 2 (34,486 total enrichments; peaks at layers 0–3, 2,050–2,650/layer) |
| **Reactome** (Jassal et al. 2020) | Reactome pathway knowledgebase | Phase 2 (154,253 total enrichments; peaks at layers 0–3, 9,200–11,000/layer) |
| **STRING v12.0** (Szklarczyk et al. 2023) | https://string-db.org/, score ≥ 700 | Phase 2 (3,924 total enrichments; 181–302/layer; PPI gradient peaks at L0 266 pairs) |
| **TRRUST v2** (Han et al. 2018) | https://www.grnpedia.org/trrust/ | Phase 2 feature annotation (2,222 TF enrichments + 562 TF→target edge enrichments); Phase 8 perturbation-response regulatory specificity test (48 TFs among 100 CRISPRi targets) |

### 3.4 Reference cell-type marker sets

56 cell types from Tabula Sapiens (spanning immune, kidney, lung compartments). Used for Phase 11 cell-type enrichment mapping.

## 4. Outputs

Per model × seed × experimental condition, the pipeline emits:

1. **Per-layer activation tensors** (NPY memory-mapped): shape `[n_positions, d_model]`, float32. For Geneformer d=1,152: ~336.4 GB total across 18 layers (~18.7 GB/layer). For scGPT d=512: ~103.6 GB across 12 layers (~6.8 GB/layer, variable due to per-cell sequence length differences). Filename convention: `layer_{L:02d}_activations.npy`. Alongside: `layer_{L:02d}_gene_ids.npy`, `layer_{L:02d}_cell_ids.npy`, `layer_{L:02d}_gene_names.json`.
2. **Trained TopK SAE weights** per layer (PyTorch state dict): `sae_final.pt`. Contains `W_enc`, `b_enc`, `W_dec`, `μ` (pre-encoder mean-centre vector). Decoder columns are unit-normalised after every gradient step.
3. **Training log** per layer: `training_log.json` — per-step loss curve, dead-feature count trajectory, variance explained on held-out split.
4. **Training results summary** per layer: `results.json` — final variance explained, alive-feature count, dead-feature count, mean absolute pairwise decoder cosine.
5. **Feature catalog** per layer: `feature_catalog.json` — for each alive feature, its top-20 genes (by mean activation magnitude), max activation, activation frequency on 100K held-out positions, decoder norm, SVD alignment flag.
6. **Feature annotations** per layer: `feature_annotations.json` — for each feature, the list of significant enrichment terms (term id, database source, OR, raw *p*, BH-corrected *q* < 0.05, overlap size).
7. **SVD comparison report** per layer: `svd_comparison.json` — for each feature, max cosine with top-50 SV axes, SVD-aligned flag (cosine > 0.7), annotation rate split (SVD-aligned vs novel).
8. **Cross-layer tracking graph:** `cross_layer_tracking.json` — for each feature at each layer, decoder-cosine matches to features at subsequent layers (threshold 0.7), feature lifecycles (first/last layer), persistence distribution (per Supp Table S2: L0→L1 2.5%, L0→L4 1.5%, L0→L8 0.2%, L0→L12+ 0%).
9. **Co-activation module graph** per layer: `coactivation_modules_L{L}.json` — PMI-weighted feature co-activation edges, Leiden community assignment at resolution 1.0 (per Supp Table S3: 141 modules across 18 Geneformer layers, 96.0–99.5% feature coverage).
10. **Causal patching report** at the selected probe layer: `causal_patching_L{L}.json` — for each of 50 richly annotated features, mean specificity ratio `|Δlogit_target| / |Δlogit_other|`, bootstrap CIs, top target and off-target genes (per Table 5: median 2.36×, 60% >2×, 12% >10×; top feature F2035 Cell Differentiation neg. reg. 114.5× specificity).
11. **Cross-layer information highway graph:** `deps_L{A}_to_L{B}.json` — per source feature, top-50 target features by cross-layer PMI, "information highway" flag (≥1 target at PMI > 3). For Geneformer three primary layer pairs (L0→L5, L5→L11, L11→L17): per Supp Table S5, 97.4–99.8% of features are information highways; mean max PMI 6.61–6.79.
12. **Perturbation response map:** `perturbation_response_L{L}.json` — for each of 100 CRISPRi targets (48 TRRUST TFs + 52 other genes), list of responding features (Wilcoxon BH *q* < 0.05, |effect| > 0.5), Fisher's exact test for TF regulatory target enrichment. Summary: 92% detection rate, **6.2% TF-specific rate** (3/48 TFs).
13. **Multi-tissue comparison:** `phase3_multitissue/perturbation_response_L{L}_multitissue.json` + `comparison_summary.json` — K562-only vs multi-tissue at layers {0, 5, 11, 17}: L0 0.0%, L5 8.3%, L11 **10.4%** (+4.2pp over K562-only 6.2%), L17 2.1%. Per-TF head-to-head (Supp Table S8): 5 gained (ATF5, BRCA1, GATA1, RBMX, NFRKB), 3 lost (MAX, PHB2, SRF), 40 unchanged.
14. **Unannotated feature characterisation:** `novel_features_L{L}.json` — Jaccard-clustered standalone gene-set clusters (per Supp Table S10: 11–19 clusters/layer, 47–69 features/layer), guilt-by-association co-activation module membership (95–98.5% of unannotated features co-activate with annotated; 1.4–4.7% truly isolated).
15. **Cell-type enrichment map:** `celltype_enrichments_L{L}.json` — for each feature, list of significant cell-type associations (Fisher's exact, BH *q* < 0.05). scGPT L7: 2,028/2,048 features (99.0%) enriched for ≥1 cell type.
16. **Feature-space geometry** (visualisation artefacts): force-directed graph layouts of intra-module co-activation edges (Fruchterman–Reingold via NetworkX); TF-IDF weighted ontology vector UMAP (n_neighbors=15, min_dist=0.1, cosine) and t-SNE (perplexity 20, cosine) for independent validation.
17. **Interactive web atlases** (deployed): six integrated views (Layer Explorer, Feature Detail Pages, Module Explorer, Cross-Layer Flow, Gene Search, Ontology Search), built with React 18 + TypeScript + Vite 6 + Tailwind CSS + Plotly.js, served via GitHub Pages.
18. **Success-criteria scorecard** (JSON): five criteria, per Table 8 of the paper — modules match pathways (≥20 target, result 141, Exceeded); causal specificity (Majority >2× target, 60% Exceeded); **perturbation specificity (≥30% target, 6.2% Not Met)**; unannotated structure (≥10% in clusters, 95–98.5% in modules Partial); cross-layer tracking (≥50%, 97–99.8% highways Exceeded).

## 5. Dependencies

- **Python ≥ 3.10**, PyTorch ≥ 2.0 (CUDA or Apple Silicon MPS), transformers ≥ 4.20, scanpy ≥ 1.9, anndata ≥ 0.8, h5py, numpy, scipy, statsmodels (for BH FDR), scikit-learn (for Fisher's exact), networkx, leidenalg, python-igraph (Leiden community detection), pandas, matplotlib, plotly (for atlas dashboards), UMAP-learn, tqdm.
- **Reference implementations:**
  - **TopK SAE:** Makhzani & Frey 2013; Gao et al. 2024 variant (the version used by this pipeline, with decoder unit-normalisation after every gradient step).
  - **Leiden:** `leidenalg.find_partition(..., resolution_parameter=1.0, seed=42)`.
  - **PMI:** explicit log2(P(i,j) / [P(i)P(j)]) over batched activation sparse-matrix accumulation.
- **Single-cell transformer dependencies:**
  - **geneformer** package via HuggingFace `ctheodoris/Geneformer` (`Geneformer-V2-316M`), **required** for Phase 0a.
  - **scGPT** reference checkpoints (see `../../repos/bio-sae/scgpt_src/` for loader and vocab handling), **required** for the scGPT side of Phase 0b.
- **Reference data:**
  - GO BP, KEGG, Reactome, STRING ≥700, TRRUST v2 — local cached files.
  - Replogle K562 CRISPRi h5ad (~640k cells, 2,024 perturbations) — Figshare DOI 10.6084/m9.figshare.21452470.
  - Tabula Sapiens immune/kidney/lung h5ad — via tabula-sapiens-portal or CZ CELLxGENE.
- **Compute:** source paper ran on Apple Silicon (M2 Max, 96 GB unified memory, MPS backend). Total runtime ~18 hours Geneformer 18 layers (extraction) + 1 hour × 18 = 18 hours SAE training + ~10 hours downstream analyses + 4 hours scGPT + 8 hours multi-tissue control ≈ **60 hours wall-clock** for full Geneformer + scGPT pipeline on a single M2 Max. CUDA equivalent ~2–3× faster for extraction.
- **Environment files:** `../../repos/bio-sae/requirements.txt`. Run entry point: `../../repos/bio-sae/run_all.sh` (parameters hardcoded for author's data paths; adapt per environment).

## 6. Methodology

The pipeline runs in fourteen phases. Phase 0 is the model-specific adapter; Phases 1–13 are the model-agnostic evaluation battery.

---

### Phase 0 — Model adapter: residual-stream activation extraction

**Purpose:** produce per-layer per-position residual-stream activation tensors from the target model, stored as memory-mapped float32 NumPy arrays, for input to TopK SAE training and all downstream analyses.

**Steps:**

1. **Load the target model** in eval mode. For HuggingFace models: `AutoModel.from_pretrained(...)`. For scGPT: load checkpoint and vocabulary per the scGPT reference loader (see `repos/bio-sae/scgpt_src/`).
2. **Load the training cell population.** For Geneformer primary run: 2,000 K562 control cells from Replogle genome-scale CRISPRi (Replogle et al. 2022). Mean genes/cell ≈ 2,028. For scGPT primary run: 3,000 Tabula Sapiens cells balanced across 1,000 immune (43 cell types) + 1,000 kidney + 1,000 lung.
3. **Tokenise** each cell per the model's native protocol:
   - **Geneformer:** rank-value tokens (gene IDs sorted by expression, descending). `MAX_SEQ_LEN = 2048`.
   - **scGPT:** continuous-value gene tokens. `MAX_SEQ_LEN = 1200`. Padded/truncated to this length.
4. **Register PyTorch forward hooks** on every transformer block's post-residual output. The hook captures the hidden state after that block's residual-stream update (i.e., after `block(x) + x`), shape `(batch, seq_len, d_model)`.
5. **Forward pass, batch size 1.** Small batch size is critical on Apple Silicon MPS for memory constraints; CUDA users can increase but source paper used batch=1 throughout.
6. **Unfold to per-position activation rows.** For each cell of sequence length *L*, emit *L* rows of shape `[d_model]`, one per (cell, token-position). Total Geneformer: ~4,056,351 positions/layer (2,000 cells × ~2,028 mean genes). Total scGPT: ~3,561,832 positions/layer.
7. **Store as memory-mapped float32 NumPy arrays.** File naming: `phase1_k562/layer_{L:02d}_activations.npy` (Geneformer), `scgpt_atlas/layer_{L:02d}_activations.npy` (scGPT). Alongside: `layer_{L:02d}_gene_ids.npy` (token → gene-name mapping per position), `layer_{L:02d}_cell_ids.npy`, `layer_{L:02d}_gene_names.json`.
8. **Verify extraction integrity.** Randomly sample 100 positions per layer, recompute the activation by re-running the model on the single cell, and check numeric equality to the stored value within float32 precision. Failure to match indicates a forward-hook bug (most commonly: capturing pre-residual instead of post-residual output).

**Key artefact sizes (Geneformer worked example):** 336.4 GB total (18 layers × 18.7 GB). Plan disk accordingly.

**Code references:**
- Geneformer extraction: `repos/bio-sae/src/01_extract_activations.py` (+ `01b_extract_remaining_layers.py` for resume)
- scGPT extraction: `repos/bio-sae/scgpt_src/01_extract_activations.py`

**Critical pitfall:** scGPT's original **continuous expression values** are NOT preserved during activation extraction if you pass the gene IDs only. This breaks Phase 7 causal patching for scGPT (see Phase 7 scGPT-specific failure mode). For scGPT, you must store and re-inject the original expression floats during the patching forward pass if you want Phase 7 causal patching to work.

---

### Phase 1 — TopK SAE training (per layer)

**Purpose:** train an overcomplete dictionary of interpretable features on each layer's residual-stream activations using TopK sparse autoencoders.

**Architecture (Gao et al. 2024 / Makhzani & Frey 2013 TopK variant):**
- **Encoder:** `h = TopK_k(W_enc · (x − μ) + b_enc)`, where `x ∈ ℝ^d_model`, `μ` is the training-set mean (pre-computed), `W_enc ∈ ℝ^(d_SAE × d_model)`, and `TopK_k` is a hard sparsification retaining only the `k = 32` largest pre-activations (setting the rest to zero).
- **Decoder:** `x̂ = W_dec · h + μ`, where `W_dec ∈ ℝ^(d_model × d_SAE)`. **Decoder columns are unit-normalised after every gradient step** (this is critical for SAE quality — without it, features can have arbitrary magnitudes and feature-feature comparisons break).
- **Dictionary size:** `d_SAE = 4 × d_model` (4× overcomplete). Geneformer: **4,608** features per layer. scGPT: **2,048** features per layer.
- **Loss:** `L = ||x − x̂||²` (MSE reconstruction only — **no L1 penalty**; TopK enforces exact sparsity).
- **Optimiser:** Adam, learning rate **3×10⁻⁴**, **batch size 4,096**, **4 epochs**.
- **Training subsample:** 1,000,000 positions per layer (uniformly subsampled from the ~4M available) for training efficiency. The subsample is regenerated per layer with a fixed seed.

**Steps:**

1. **Pre-compute the mean.** Load the full per-layer activation memmap, compute per-column mean `μ`. Persist to disk alongside the SAE weights for inference-time centring.
2. **Initialise weights.** `W_enc` from a scaled Kaiming normal; `W_dec` is initialised as `W_enc.T` then unit-normalised.
3. **Train for 4 epochs.** Loop over 1M subsampled positions, batch 4,096, Adam lr=3e-4. After every backward + optimizer step, **unit-normalise `W_dec` columns**.
4. **Checkpoint every 10,000 steps.** Persist `sae_final.pt` at the end (encoder+decoder weights, μ).
5. **Evaluate on a 100K held-out split.** Compute:
   - **Variance explained** = 1 − (residual variance / input variance). Source paper results (Supp Table 1/2): Geneformer per-layer range **76.8%–85.3%**, mean **81.7%**. scGPT per-layer range 85.7%–93.5%, mean 90.2%. (scGPT's higher VarExpl reflects smaller `d_model`, so 4× expansion is less extreme.)
   - **Dead features** = features with zero activation across the entire 100K held-out set. Geneformer: 419/82,944 total across 18 layers (**0.5%**); range 0 (L0) to 70 (L16). scGPT: 49/24,576 total (**0.2%**). Highest dead-feature counts at Geneformer L15–L17 (60–70 dead each) — "mid-layer revival" of structured representations at L10–L11 visible.
   - **Mean absolute pairwise decoder cosine similarity**: Geneformer **0.033–0.040**; scGPT **0.038–0.049**. Both small, indicating well-separated dictionary directions.
6. **Orchestration.** For a full pipeline, loop over all layers via `repos/bio-sae/src/02b_train_all_layers.py` (skips layers already trained if `sae_final.pt` exists). Total training wall-clock: ~1 hour/layer Geneformer on M2 Max.

**Code references:**
- TopK SAE class: `repos/bio-sae/src/sae_model.py` → `TopKSAE` (encoder + topk sparsification + decoder + unit-norm columns), `SAETrainer` (Adam, lr 3e-4, batch 4096, 4 epochs, checkpoint every 10k steps)
- Single-layer training: `repos/bio-sae/src/02_train_sae.py`
- Orchestration across all layers: `repos/bio-sae/src/02b_train_all_layers.py`

**Critical sanity signatures:**
- **Mean absolute pairwise decoder cosine < 0.05.** If > 0.1, decoder directions are collapsing into a narrower cone than they should — check unit-normalisation.
- **Dead feature rate < 2%** per layer. If > 5%, either the subsample is too small or the layer activation distribution is highly non-stationary (e.g., you accidentally mixed layers).
- **Variance explained > 75%.** If < 60%, the dictionary is under-expressive — try `d_SAE = 8 × d_model` (but see §9 pitfall on hyperparameter ablation — the paper's revision tested expansion ∈ {2, 4, 8} and `k ∈ {16, 32, 64}`, confirming that 4×/32 is a near-optimum).

---

### Phase 2 — Feature analysis and ontology annotation

**Purpose:** identify each SAE feature's biological identity by its top-20 genes and test for enrichment against five ontology databases.

**Steps:**

1. **Extract top-20 genes per alive feature.** A feature is "alive" if its activation frequency > 0 on a 100K held-out position sample. For each alive feature, compute **mean activation magnitude** at each token position by grouping positions by gene name (using `layer_{L:02d}_gene_names.json` from Phase 0). The top-20 genes are those with the highest mean activation magnitude for that feature. Persist `feature_catalog.json` per layer.
2. **Map each gene to its annotation terms** in five databases:
   - **GO Biological Process** (Ashburner et al. 2000; gene → GO BP terms)
   - **KEGG** (Kanehisa & Goto 2000)
   - **Reactome** (Jassal et al. 2020)
   - **STRING v12.0** at confidence ≥ 700 (Szklarczyk et al. 2023) — used here as gene sets per STRING term (not pairwise PPI).
   - **TRRUST v2** (Han et al. 2018) — both TF-target sets (gene sets of each TF's targets) and individual TF→target edges.
3. **Test each feature × each annotation term** via **Fisher's exact test (one-sided)**. The 2×2 contingency table: rows = (gene in top-20, gene not in top-20), columns = (gene has term, gene does not). Apply **Benjamini–Hochberg FDR correction at α = 0.05** across all terms tested for that feature.
4. **Persist significant enrichments.** `feature_annotations.json` — for each feature, the list `[(db, term_id, term_name, OR, raw_p, q_value, overlap_size), ...]`.
5. **Compute global annotation statistics** per layer:
   - **Annotation rate** = fraction of alive features with ≥1 significant enrichment. Geneformer range: **45.4% (L8) – 58.6% (L0)**. scGPT range: **28.7% (L2) – 33.9% (L9)**. Mean: Geneformer 52.4%, scGPT 31.0%.
   - **Enrichments per layer** (Supp Table S1): Geneformer GO BP 6,628 (L7) to 10,153 (L0); KEGG 1,520–2,650; Reactome 6,869–11,001; STRING 150–302; TRRUST TF 87–164; TRRUST Edges 24–48. **Totals across 18 layers: 145,106 GO BP + 34,486 KEGG + 154,253 Reactome + 3,924 STRING + 2,222 TRRUST TF + 562 TRRUST Edges = 340,553 enrichments.**

**Expected U-shaped annotation profile across layers (Geneformer):**

- **Early layers (L0–L4) — "Molecular machinery":** highest GO BP (8,500–10,150), KEGG (2,050–2,650), Reactome (9,200–11,000); STRING 248–302; TRRUST TF 133–164. Features encode textbook biological programs (cell cycle G2/M, DNA replication/repair, B-cell activation, cytoskeleton, MAPK/TGFβ signalling, mitosis) — see representative Table 4 in the paper.
- **Middle layers (L5–L9) — "Abstract computation":** annotation rates drop below 50% at L6–L8 (minimum **45.4% at L8**); intermediate computational representations harder to map to ontology terms. GO BP 6,600–7,700, STRING 181–216.
- **Mid-late layers (L10–L12) — "Re-specialisation":** annotation rates recover to **53–56%** as features shift to integrative cellular programs (Cell Differentiation, Intracellular Signalling, Mitochondrial Organisation, Vesicular Transport, Chromatin/Transcription, Protein Modification, Cell Motility, Stress Response).
- **Terminal layers (L15–L17) — "Prediction-focused":** second annotation decline (47–55%), most dead features (28–70/layer), most distributed representations. Features at these layers respond broadly to perturbations but lack mid-layer specificity.

**Code references:**
- Top-gene extraction: `repos/bio-sae/src/03_analyze_features.py`
- Ontology annotation: `repos/bio-sae/src/04_annotate_features.py`
- scGPT equivalents: `repos/bio-sae/scgpt_src/03_analyze_features.py`, `scgpt_src/04_annotate_features.py`

**Representative feature table (Geneformer, source paper Table 4):**

| Layer | Feature | Top genes | Biological identity | Databases enriched |
|---|---|---|---|---|
| L0 | F3717 | CDK1, CDC20, DLGAP5, PBK | Cell cycle G2/M | GO, KEGG, Reactome, STRING |
| L0 | F3607 | RRM1, E2F1, MCM4, MCM6, RAD51 | DNA replication / repair | GO, KEGG, Reactome, STRING, TRRUST |
| L0 | F1116 | ARL6IP4, SQSTM1, JAK1 | B-cell activation | GO, KEGG, Reactome |
| L0 | F4536 | DYNC1H1, TLN1, MYH9, SPTAN1 | Cytoskeleton / focal adhesion | GO, KEGG, Reactome |
| L0 | F2829 | TGFB1, PKN1, GADD45A, ZBTB7A | MAPK / TGFβ signalling | GO, KEGG, Reactome, STRING, TRRUST |
| L0 | F1573 | MKI67, CENPF, TOP2A, CDK1, AURKB | Mitosis / chromosome segregation | GO, KEGG, Reactome, STRING |
| L11 | F2035 | (cell diff. genes) | Cell diff. (neg. reg.) | GO, Reactome |
| L11 | F3692 | (ERAD pathway genes) | ER-associated degradation | GO, KEGG, Reactome |
| L11 | F3933 | (signalling genes) | Intracell. signalling (neg. reg.) | GO, Reactome |
| L11 | F2936 | (mitochondrial genes) | Mitochondrion organisation | GO, KEGG, Reactome |

These features reappear in Phase 6 as causally specific (Phase 6 Table 5).

---

### Phase 3 — Superposition quantification via SVD comparison

**Purpose:** quantify how much of the biological knowledge encoded in SAE features is **invisible** to standard linear dimensionality reduction, confirming that the foundation model uses superposition (more concepts than dimensions, packed into nearly-orthogonal directions).

**Steps:**

1. **Compute top-50 SVD axes per layer.** Load the same 100K held-out activation sample, run `numpy.linalg.svd` (or `scipy.sparse.linalg.svds` if sparse), retain the top-50 right singular vectors `V^T[:50]`.
2. **For each SAE feature, compute max cosine with top-50 SV axes.** `max_cos(feat_i) = max_{k=1..50} |cos(W_dec[:, i], V^T[k, :])|`.
3. **Classify features as "SVD-aligned"** if `max_cos > 0.7` with any SV axis. Everything else is "novel."
4. **Per-layer SVD-aligned feature counts** (Geneformer):
   - L0: **41**
   - L1–L2: 25–29
   - L3–L14: **2–13** (most layers)
   - L17: 12
   - **Total across 18 Geneformer layers: 189 of 82,944 (0.2%) — 99.8% novel.** scGPT: 49 of 24,576 (0.2%) — 99.8% novel.
5. **Split-annotation-rate analysis.** Compute annotation rate separately for SVD-aligned and novel features. Paper result:
   - **SVD-aligned: 14.3% annotation rate** (27 of 189 aligned features annotated)
   - **Novel: 52.5% annotation rate** (43,214 of 82,336 novel features annotated)
   - **Exclusivity: 98.7% of ontology enrichment terms are found EXCLUSIVELY in novel features.** Only 1.3% of biological knowledge loadings on Geneformer are recoverable via top-50 SVD.
6. **Variance-explained comparison.** Compute `variance explained by SAE` vs `variance explained by top-50 SVD`:
   - SAE variance explained: 77–85%
   - Top-50 SVD variance explained: 31–38%
   - **Ratio: 2.4×** — the SAE dictionary represents structure that SVD simply cannot reach under any linear-rotation of the 50 dominant axes.

**Interpretation:** the model uses **massive superposition** (Elhage et al. 2022): 4,608 features packed into 1,152 dimensions via quasi-orthogonal directions. **Compression ratio exceeds 70×** for Geneformer. This implies that claims about what a single-cell foundation model "knows" based on SVD / PCA / linear probing are substantially underestimating the richness of encoded biological structure.

**Code references:**
- SVD comparison: `repos/bio-sae/src/05_compare_svd.py`
- Revision sensitivity sweep (thresholds 0.3, 0.5, 0.7, 0.9): `repos/bio-sae/src_revision/b1_svd_threshold_sweep.py`

---

### Phase 4 — Cross-layer feature tracking

**Purpose:** test whether SAE features persist across transformer depth (a "scaffold" hypothesis) or whether the biological content is rebuilt at every layer.

**Steps:**

1. **Compute pairwise decoder weight vector cosine similarity between feature *i* at layer *l* and feature *j* at layer *l'***. A "match" is declared if `cos(W_dec^{(l)}[:, i], W_dec^{(l')}[:, j]) > 0.7`.
2. **Persistence rate from L0 to layer *l'***: fraction of L0 features with ≥1 match at layer *l'*. Supp Table S2 expected (Geneformer):
   - L0 → L1: 114 / 4,608 matches = **2.5%**
   - L0 → L2: 93 = 2.0%
   - L0 → L4: 67 = 1.5%
   - L0 → L6: 25 = 0.5%
   - L0 → L8: 10 = 0.2%
   - L0 → L10: 1 ≈ 0%
   - **L0 → L12+: 0 = 0%**
3. **Per-feature layer persistence.** For every feature, compute `(first_layer, last_layer)` from the decoder-cosine graph. Report the distribution. Paper result: **98.2% of L0 features are transient** (present in ≤3 layers, mean 5.4 enrichment terms); **1.8% moderate-persistence** (present in 4–10 layers) have only 3.7% annotation rate and 0.1 mean enrichments. **Moderate-persistence features are LESS biologically meaningful than transient features** — a striking negative correlation between persistence and biological content.
4. **Interpretation:** features are overwhelmingly layer-specific. **Only 2–3% of features at any layer match a feature at the adjacent layer**, and no feature survives past L11. **The biological content is carried by features rebuilt at every layer, not by any persistent scaffold.** This aligns with the "radical representational transformation" narrative of transformer depth.
5. **Statistical controls** (from `src_revision/b3_crosslayer_null_and_consecutive.py`):
   - **Null test:** shuffle feature labels across layers and recompute match rates. Expected: shuffled match rate < observed rate.
   - **Consecutive-only test:** measure immediate L → L+1 persistence only, verifying the decay pattern is monotone (it is — 2.5% at L0→L1 declining to 0% by L0→L12+).

**Code references:**
- Cross-layer tracking: `repos/bio-sae/src/06_cross_layer.py`
- Null and consecutive tests: `repos/bio-sae/src_revision/b3_crosslayer_null_and_consecutive.py`

---

### Phase 5 — Co-activation graph and module discovery

**Purpose:** discover the relational structure among features at each layer by computing pointwise mutual information between features and applying Leiden community detection.

**Steps:**

1. **Define feature activity.** Feature *i* is "active" at position *j* if its TopK activation at that position is non-zero (i.e., feature *i* is among the 32 top features at position *j*).
2. **Accumulate co-activation counts across all ~4 million positions per layer.** For every pair of alive features `(i, j)`: `count(i, j) = #{positions where both i and j are active}`. Marginal counts: `count(i) = #{positions where i is active}`.
3. **Compute pointwise mutual information:** `PMI(i, j) = log2(P(i, j) / [P(i) · P(j)])`, where `P(i, j) = count(i, j) / N`, `P(i) = count(i) / N`, `N` is total positions.
4. **Retain edges exceeding a significance threshold.** Source paper uses `p < 0.001` under a permutation null (shuffle position labels, recompute PMI, check fraction of permutations with PMI ≥ observed). In practice the pipeline uses a hard PMI threshold of **2.0** as a fast proxy.
5. **Build the sparse feature-feature co-activation graph** as a `scipy.sparse.csr_matrix`.
6. **Run Leiden community detection** at resolution 1.0 via `leidenalg.find_partition(...)` on the graph. Per Supp Table S3 (Geneformer):
   - L0: 6 modules, 4,577 features in modules, 446,324 PMI edges, **99.3% coverage**
   - L5 (peak modularity): **12 modules**, 4,472 features, 390,845 edges, 97.7% coverage
   - L11: 8 modules, 4,565 features, 388,103 edges, 99.3% coverage
   - L17: 7 modules, 4,474 features, 343,059 edges, 97.7% coverage
   - **Total Geneformer: 141 distinct modules across 18 layers.** Coverage range 96.0%–99.5%.
   - Mean 7.8 modules/layer.
7. **scGPT equivalent** (Table 2 in paper): 5–7 modules/layer, 76 total, mean 6.3 modules/layer, 96.3% mean coverage.
8. **Biological identity of modules** (layer 0 reference, Geneformer): 6 modules corresponding to (1) Cell Cycle / DNA Replication, (2) Immune Signalling, (3) Metabolism, (4) Translation, (5) Protein Quality Control, (6) Cytoskeleton / Adhesion. At layer 11, 8 modules shift toward integrative cellular programs (Cell Differentiation, Intracellular Signalling, Mitochondrial Organisation, Vesicular Transport, Chromatin/Transcription, Protein Modification, Cell Motility, Stress Response).
9. **PMI edge density declines with depth:** 446K at L0 → 328K at L16, paralleling the reconstruction-quality decline.
10. **Note on triviality.** With `k = 32` sparsity out of 4,608 features, a random co-activation graph would produce a **connected but undifferentiated** graph. Leiden finding 6–12 well-separated communities per layer with 96–99.5% coverage is non-trivial — it indicates genuine biological compartmentalisation.

**Code references:**
- Primary co-activation: `repos/bio-sae/src/07_feature_coactivation.py` (PMI accumulation, Leiden clustering, module export)
- scGPT equivalent: `repos/bio-sae/scgpt_src/07_coactivation.py`
- Revision PMI memory + Leiden resolution sweep (0.5–2.0, ARI vs resolution 1.0 baseline): `repos/bio-sae/src_revision/b2_pmi_and_leiden_sweep.py`

**Critical sanity signature:** Leiden at default resolution 1.0 should find **6–12 modules per layer** with >96% feature coverage. If you see 1–2 giant modules covering everything, the PMI threshold is too lax; if you see hundreds of tiny modules, the threshold is too strict or resolution is too high.

---

### Phase 6 — Causal feature patching

**Purpose:** test whether individual SAE features are causally necessary for the model's downstream computation, by ablating each feature at a probe layer and measuring the effect on output logits.

**Steps:**

1. **Select the probe layer.** Source paper uses **Geneformer L11** (approximately two-thirds through the 18-layer stack) and **scGPT L7** (approximately two-thirds through the 12-layer stack).
2. **Select 50 richly annotated features** at the probe layer. "Richly annotated" = features with ≥3 significant ontology enrichments covering a coherent biological theme (see Phase 2 Table 4 for examples).
3. **For each of 50 features:**
   a. Load 200 K562 control cells.
   b. Run the forward pass up to the probe layer. At the probe layer, encode the hidden state through the SAE: `h = TopK(W_enc · (x − μ))`.
   c. **Zero the target feature's SAE activation:** set `h[target_feature_idx] = 0`.
   d. Decode back: `x̂ = W_dec · h + μ`. Replace the original residual-stream hidden state with `x̂` at the probe layer.
   e. **Continue the forward pass** from the probe layer through the remaining layers (L12–L17 for Geneformer, L8–L11 for scGPT) to output logits.
   f. Compute the change in output logits vs the clean forward pass at every gene position:
      - **Target genes** = genes matching the feature's ontology annotation (e.g., feature F2035 "Cell Differentiation (neg. reg.)" → genes annotated with GO:0045596).
      - **Other genes** = all other genes in the vocabulary.
      - **Δlogit_target** = mean logit change over target genes.
      - **Δlogit_other** = mean logit change over other genes.
   g. **Specificity ratio:** `specificity = |Δlogit_target| / |Δlogit_other|`.
4. **Aggregate across the 50 features** and compute the distribution of specificity ratios. Expected Geneformer L11 result (Table 5 in paper):
   - **Features tested:** 50
   - **Mean specificity ratio:** 8.98
   - **Median specificity ratio:** **2.36×**
   - **% specific (>2×):** 30/50 = **60%**
   - **% highly specific (>10×):** 6/50 = **12%**
   - **Mean target logit change:** −0.116
   - **Mean other logit change:** −0.005
   - Ratio of means: **23×** — target-gene disruption is 23 times larger than off-target disruption.
5. **Top 10 most causally specific features** (Supp Table S4):
   - F2035 Cell Differentiation (neg. reg.): **114.5×** (Δtarget −0.208, Δother +0.002)
   - F3692 ERAD Pathway: **108.1×** (Δtarget −0.129, Δother −0.001)
   - F3933 Intracellular Signalling (neg. reg.): **55.7×**
   - F157 Golgi Vesicle Transport: **25.4×**
   - F3532 Protein Metabolic Process (pos. reg.): **11.2×**
   - F4516 Mitotic Spindle Microtubules: **10.6×**
   - F1337 Cell Cycle Phase Transition: **9.4×**
   - F1023 Mitotic Spindle Microtubules: **7.6×**
   - F2936 Mitochondrion Organisation: **7.1×**
   - F3962 Endocytosis: **6.9×**
6. **Contrast with component-level null.** Paper notes (from companion study arXiv:2602.17532) that ablating entire attention heads or MLP layers produces **null** behavioural effects on the same Geneformer model. Feature-level interventions reveal structure that component-level interventions miss — the computations are encoded at the feature level (specific directions within the residual stream) rather than being localised to individual attention heads or MLP layers.

**scGPT probe at L7 — known failure mode:**
- Median specificity **0.98× (mean 1.02×)**, **0/50 features exceed 2×.**
- **This substantially weaker causal signal reflects the use of proxy expression values (uniform 1.0 for all genes) rather than actual expression levels**, because the original continuous expression values were not preserved during activation extraction (see Phase 0 critical pitfall). The Phase 7 cross-layer computational graph analysis confirms scGPT features ARE genuinely interconnected, but the causal patching specifically requires true per-cell expression values to work — fix the Phase 0 adapter for scGPT if you want Phase 6 to work on it.

**Code references:**
- Geneformer causal patching: `repos/bio-sae/src/08_causal_patching.py`
- scGPT causal patching: `repos/bio-sae/scgpt_src/08_causal_patching.py`

**Critical sanity signature:** median specificity ratio **≈ 2.36×** on Geneformer L11. If < 1.5×, the intervention is not actually zeroing the feature's contribution — check that the decoded `x̂` is correctly replacing the hidden state in the residual stream before the next block.

---

### Phase 7 — Cross-layer computational graph and information highways

**Purpose:** map how biological information flows through the network by computing cross-layer PMI between SAE feature activations at source and target layers.

**Steps:**

1. **Select 3 source-layer → target-layer pairs.** Source paper uses Geneformer pairs: **L0 → L5, L5 → L11, L11 → L17**. For scGPT: L0 → L4, L4 → L8, L8 → L11.
2. **Encode 500,000 positions through both layers' SAEs.** For each position, record which features activate at the source layer and which at the target layer.
3. **Compute cross-layer PMI** for every (source_feature, target_feature) pair:
   `PMI(f_src, f_tgt) = log2(P(f_src ∧ f_tgt) / [P(f_src) · P(f_tgt)])`
4. **For each source feature, retain the top-50 target features by PMI.** Persist `deps_L{A}_to_L{B}.json`.
5. **Flag "information highways":** a source feature is an information highway if it has **≥1 target-layer feature at PMI > 3**.
6. **Per-layer-pair statistics** (Supp Table S5, Geneformer):
   - **L0 → L5:** 4,604 features with dependencies; mean max PMI **6.61**, median 6.72, max 11.10; **4,530 / 4,608 highways (98.4%)**
   - **L5 → L11:** 4,518 with deps; mean max PMI 6.63, median 6.71, max 10.87; **4,401 highways (97.4%)**
   - **L11 → L17:** 4,555 with deps; mean max PMI **6.79** (strongest), median 6.86, max 10.66; **4,544 highways (99.8%)** — the densest connectivity, increasing toward the output.
7. **Biologically meaningful cross-layer cascades** (Supp Table S7):
   - **L0 → L5:** Protein Processing in ER → (ER stress cascade, PMI 11.10); mTORC1 Regulation → Autophagy (PMI **9.55**); Wnt Signalling → (Wnt pathway processing, PMI 9.48).
   - **L5 → L11:** Protein Polyubiquitination → (Protein quality control, PMI 10.87); Translation → (Translational regulation, PMI 10.35); RNA Splicing → (Post-transcriptional, PMI 10.21).
   - **L11 → L17:** Protein Modification → Angiogenesis regulation (PMI **10.62** — a "PTM → phenotype" cascade); COPII Vesicle Budding → Thermogenesis (PMI 10.29, "Secretory → metabolic"); Actomyosin Organisation → Cell Locomotion (PMI 10.14, "Structure → motility").
8. **scGPT cross-layer pattern — progressive concentration** (Supp Table S6, Fig. S1). **Upstream connectivity is consistently high (86.6–95.5%)** but **downstream connectivity drops from 95.7% at L0→L4 to 62.9% at L8→L11.** PMI edge density: 75K → 61K → 45K across the three scGPT layer pairs.
   - **Interpretation:** scGPT progressively bottlenecks information toward later layers — a pattern NOT observed in Geneformer (97.4–99.8% downstream throughout). This is a fundamental architectural difference. scGPT is continuous-value + masked-gene-prediction; Geneformer is rank-value + next-token prediction. The downstream bottleneck in scGPT may reflect its different training objective (masked prediction encourages representational concentration) vs Geneformer's next-token objective (encourages distributed representations).

**Code references:**
- Geneformer cross-layer graph: `repos/bio-sae/src/11_computational_graph.py`
- scGPT cross-layer graph: `repos/bio-sae/scgpt_src/11_computational_graph.py`

---

### Phase 8 — Perturbation response mapping (central scientific test)

**Purpose:** test whether the SAE feature atlas encodes **regulatory logic** — specifically, whether ablating a transcription factor causes SAE features to respond in a pattern that matches that TF's known regulatory targets. **This is the phase that determines whether the foundation model has internalised causal regulation or only statistical co-expression.**

**Steps:**

1. **Select 100 CRISPRi perturbation targets** from the Replogle K562 genome-scale screen:
   - **48 TRRUST transcription factors** (these are the regulatory specificity test — does knocking down TF X cause feature responses matching TF X's known targets?)
   - **52 other genes** (these are the control set — does knocking down non-TFs still cause feature changes?)
2. **For each target, extract 20 perturbed cells** from the Replogle dataset. Extract Geneformer activations at the probe layer (L11 primary) through the same forward-hook extraction as Phase 0.
3. **Encode through the layer-11 SAE** to obtain per-cell feature activations.
4. **For each feature:** compare perturbed-cell activations vs 100,000 control-position activations via **Wilcoxon rank-sum test**. Features "respond" if:
   - BH-corrected *q* < 0.05
   - |effect size| > 0.5
5. **For each of 48 TRRUST TFs:** test whether the responding features are **enriched for features whose top-20 genes overlap the TF's known regulatory targets** via Fisher's exact test. This is the regulatory-specificity test.
6. **Global detection statistics** (Table 6 in paper):
   - **Perturbation targets tested:** 100
   - **Targets causing feature changes:** **92/100 (92%)** — the pipeline has high detection sensitivity.
   - **TRRUST TFs tested:** 48
   - **TFs with specific response:** **3/48 (6.2%)** — only 6.2% of perturbations produce feature responses specific to their known regulatory targets.
   - **Mean responding features per target:** 2.54
   - **Mean specific features per target:** 0.03 — virtually none.
7. **The 6.2% null is the central finding.** The model **detects** that a perturbation has occurred (it registers the shift in cell state) but does **not encode which specific regulatory targets should be affected**. This confirms, at the SAE feature level, what the companion attention-based analysis (arXiv:2602.17532) found at the pairwise edge level: Geneformer's internal representations encode **co-expression structure and pathway membership, but not the directed TF → target regulatory wiring**.
8. **Breakdown (Fig. 8 of paper):**
   - Perturbation detection: 92% detected, 8% no response.
   - Regulatory specificity: 45/48 (93.8%) non-specific TFs, 3/48 (6.2%) specific.
   - Mean responding features per target ≈ 2.5 (distribution shown in Fig. 8C).
9. **Sanity / success scorecard.** Table 8 reports the target was ≥30% TF specificity. The observed 6.2% marks this criterion as **NOT MET** — the central unmet criterion and the scientifically most important one.

**Code references:**
- Perturbation response mapping: `repos/bio-sae/src/09_perturbation_response.py`
- Revision multi-database sensitivity (TRRUST + DoRothEA all + high-confidence): `repos/bio-sae/src_revision/c2_multidb_perturbation.py`
- Revision batch/assay confound analysis: `repos/bio-sae/src_revision/c6_batch_assay_analysis.py`

**Critical sanity signature for Geneformer:** exactly 3/48 TFs specific (6.2%) at primary parameterisation. If > 15%, either (a) your TF target lists are wider than TRRUST primary targets (check for accidental merging of TRRUST + DoRothEA + ChIP-Atlas), or (b) you are not applying BH correction properly, or (c) the effect threshold |effect| > 0.5 is not being enforced.

---

### Phase 9 — Multi-tissue SAE control experiment

**Purpose:** disentangle two possible causes of the 6.2% perturbation-specificity null: (a) SAE training data is too narrow (K562-only) to capture regulatory diversity, or (b) the foundation model's representations are themselves the bottleneck.

**Steps:**

1. **Pool training data.** Take **500,000 K562 positions** (random subsample from Phase 0 K562 activations) + **500,000 Tabula Sapiens positions** (3,000 cells: 1,000 immune across 43 cell types + 1,000 kidney + 1,000 lung, extracted through Geneformer). Total **1M balanced positions per layer**.
2. **Train new SAEs** with identical architecture, hyperparameters, and training schedule as Phase 1 — at **only four layers: 0, 5, 11, 17.**
3. **Re-run Phase 2 (annotation)**, **Phase 6 (causal patching)**, **Phase 8 (perturbation response)** on the multi-tissue SAEs.
4. **Per-layer TF specificity comparison** (Supp Table S7 in paper):
   - **K562-only SAE (L11):** 3/48 TFs specific = **6.2%**
   - **Multi-tissue SAE, L0:** 0/48 = **0.0%**
   - **Multi-tissue SAE, L5:** 4/48 = **8.3%**
   - **Multi-tissue SAE, L11:** 5/48 = **10.4%** (+4.2 percentage points over K562-only)
   - **Multi-tissue SAE, L17:** 1/48 = **2.1%**
5. **Per-TF head-to-head comparison at L11** (Supp Table S8):
   - **Gained specificity:** 5 TFs — ATF5, BRCA1, GATA1, RBMX, NFRKB
   - **Lost specificity:** 3 TFs — MAX, PHB2, SRF
   - **Unchanged (no specificity in either):** 40 TFs
   - **Disjoint sets of gained and lost** → stochastic variation, not systematic improvement.
6. **TF feature representation diagnostics** (Supp Table S9):
   - **K562-only L11:** 2,967 / 4,598 features (64.5%) have ≥1 TF in top-20 genes; 424 are TF-dominant (≥3 TFs in top-20).
   - **Multi-tissue L0:** 60.7%, 452 TF-dominant.
   - **Multi-tissue L5:** 60.8%, 337 TF-dominant.
   - **Multi-tissue L11:** 60.5%, 343 TF-dominant.
   - **Multi-tissue L17:** 58.2%, 346 TF-dominant.
   - **K562-only SAE has MORE TF-associated features (64.5%) than the multi-tissue SAE (60.5%).** Training on more diverse tissue data did not increase TF feature density.
7. **Layer pattern is informative:** L0 0%, L5 8.3%, **L11 10.4% (peak)**, L17 2.1%. Late-layer features are too abstract for regulatory mapping — this recapitulates the Phase 2 U-shaped annotation profile from a causal perspective.
8. **Conclusion:** the +4.2 percentage-point improvement is **non-systematic** (disjoint gained/lost sets) and **decreases TF feature representation**, meaning the SAE methodology and training data are **NOT** the bottleneck. **The limitation is in Geneformer's representations themselves.** The foundation model has not internalised regulatory logic.

**Code references:**
- Multi-tissue Tabula Sapiens extraction: `repos/bio-sae/src/12a_extract_tabula_sapiens.py`
- Pooled training: `repos/bio-sae/src/12b_pool_and_train.py`
- Multi-tissue annotation: `repos/bio-sae/src/12c_analyze_and_annotate.py`
- Multi-tissue perturbation test: `repos/bio-sae/src/12d_perturbation_test.py`
- Comparison summary: `repos/bio-sae/src/12e_compare_results.py`

---

### Phase 10 — Unannotated feature characterisation

**Purpose:** investigate whether the 41–55% of features lacking ontology annotations are noise or encode biology not captured by existing databases.

**Steps:**

1. **Identify unannotated features per layer** — features with zero significant enrichments from Phase 2. Counts (Supp Table S10, Geneformer):
   - L0: 1,906 unannotated / 4,608 (41.4%)
   - L5: 2,193 (47.6%)
   - L11: 2,015 (43.8%)
   - L17: 2,426 (53.0%)
2. **Test 1: Jaccard standalone clustering.** Compute pairwise Jaccard similarity of top-20 gene sets among unannotated features. Run Leiden at resolution 0.5 to form standalone gene-set clusters. Expected (Table S10):
   - L0: **15 clusters** containing **48 features** (2.5% of unannotated)
   - L5: 19 clusters, 69 features (3.1%)
   - L11: 11 clusters, 47 features (2.3%)
   - L17: 12 clusters, 58 features (2.4%)
   - **Only 2–3% of unannotated features form standalone gene-set clusters.** Biologically coherent standalone clusters include ribosomal protein programs and mitochondrial complex assembly.
3. **Test 2: Guilt-by-association via co-activation modules.** Test whether unannotated features belong to co-activation modules (from Phase 5) that also contain annotated features. Expected:
   - L0: **1,876 unannotated features co-activate with annotated (98.4%)**; 30 isolated (1.6%)
   - L5: 2,090 (95.3%); 103 isolated (4.7%)
   - L11: 1,984 (98.5%); 28 isolated (1.4%)
   - L17: 2,334 (96.2%); 75 isolated (3.1%)
   - **95–98.5% of unannotated features co-activate with annotated features in biological modules.** Only **1.4–4.7% are truly isolated.**
4. **Interpretation:** the vast majority of unannotated features are **not random noise** — they participate in the same biological modules as annotated features, but their specific function is not captured by the five tested databases. The gap between "not noise" and "biologically meaningful" remains a research direction.
5. **Caveats:**
   - With only 6–12 modules covering 96–99.5% of all features, co-membership may be nearly unavoidable.
   - Features may co-activate due to shared statistical properties rather than shared biology.
   - No perturbation validation performed for isolated features.

**Extensions in `src_revision/b4_unannotated_characterize.py`:**
- Test against MSigDB Hallmarks (as a broader database).
- Compute overlap of unannotated clusters with annotated clusters.

**Code references:**
- Primary: `repos/bio-sae/src/10_novel_features.py`
- Revision expansion: `repos/bio-sae/src_revision/b4_unannotated_characterize.py`

---

### Phase 11 — Cell-type enrichment mapping

**Purpose:** connect SAE features to cellular identity by computing mean activation per feature per cell type.

**Steps:**

1. **For each cell type in Tabula Sapiens** (56 cell types across immune, kidney, lung — scGPT primary; also applicable to Geneformer via cross-domain Tabula Sapiens extraction encoded through K562-trained SAEs), compute the **mean activation** of each SAE feature in cells of that type.
2. **Test enrichment** via Fisher's exact test — does the feature activate preferentially in this cell type vs all others? BH correction at α = 0.05.
3. **Persist** `celltype_enrichments_L{L}.json` per layer.
4. **Expected coverage:**
   - **scGPT L7: 2,028/2,048 features (99.0%) enriched for ≥1 cell type.** Similar coverage across all scGPT layers.
   - For Geneformer: encoding Tabula Sapiens through K562-trained SAEs yields comprehensive coverage despite being a cross-domain test. **Both models' features tile cell-identity space comprehensively**, consistent with training on diverse cell populations.
5. **Tissue-coherence check.** Enrichments are tissue-coherent: **immune features activate preferentially in T cells, B cells, macrophages, and dendritic cells; kidney features in proximal tubule cells and podocytes; lung features in alveolar epithelial cells and pulmonary endothelial cells.**
6. **Revision matched-data cross-model test** (`c5_matched_data_crossmodel.py`): pass the same 3,000 Tabula Sapiens cells through Geneformer (not scGPT), encode with K562-trained Geneformer SAEs at L0/L5/L11, measure variance explained vs K562 baseline. Variance drop is moderate, confirming K562-trained SAEs generalise to diverse tissue contexts.

**Code references:**
- Primary: `repos/bio-sae/scgpt_src/compute_celltype_enrichments.py` (scGPT; parallels available in `repos/bio-sae/src/` for Geneformer)
- Matched-data cross-model: `repos/bio-sae/src_revision/c5_matched_data_crossmodel.py`

---

### Phase 12 — Feature-space geometry and visualisation

**Purpose:** visualise the relational structure among SAE features despite the superposition-imposed quasi-orthogonality.

**Critical observation:** direct UMAP on decoder weight vectors produces **structureless point clouds**. Quantitative explanation: SAE decoder vectors are **quasi-orthogonal by design** (mean pairwise cosine = **0.0007**; within-module vs between-module Cohen's d = **0.075**). This is a geometric signature of superposition — 4,608 features pack into 1,152 dimensions by spreading nearly uniformly across the hypersphere. No linear projection can separate them meaningfully.

**Solution: visualise the co-activation network directly.**

**Steps:**

1. **Force-directed graph layout** of intra-module co-activation edges from Phase 5 using **Fruchterman–Reingold via NetworkX**. Each module forms a spatially coherent community (Supp Fig. S2, S3). Unassigned features cluster centrally. Per-layer module counts: 6 (L0), 12 (L5), 8 (L11), 7 (L17).
2. **TF-IDF weighted ontology vectors as independent validation.** For each annotated feature, build a sparse TF-IDF vector over the enriched ontology terms (term document frequency weighted by global rarity). Project using:
   - **UMAP:** n_neighbors=15, min_dist=0.1, cosine metric.
   - **t-SNE:** perplexity 20, cosine metric.
3. **Validation:** annotated features (blue) separate cleanly from unannotated features (grey). Fine-grained TF-IDF subclusters partially correspond to Phase 5 co-activation modules, **confirming that module structure reflects genuine biological similarity** rather than statistical artefacts of the PMI computation.
4. **Layer 11 six-panel visualisation** (Supp Fig. S3): (a) Leiden modules, (b) annotation status, (c) top ontology source, (d) annotation richness gradient, (e) activation frequency, (f) SVD alignment. SVD-aligned features (n=4) scatter across different modules rather than clustering — confirming they are not a coherent subset.

**Code references:**
- `repos/bio-sae/src/` visualisation scripts (dimensionality reduction, force-directed layouts).

---

### Phase 13 — Interactive web atlas deployment

**Purpose:** enable community access to the complete feature atlases via six integrated interactive views deployed via GitHub Pages.

**Built artefacts** (React 18 + TypeScript + Vite 6 + Tailwind CSS + Plotly.js):

1. **Geneformer Feature Atlas:** https://biodyn-ai.github.io/geneformer-atlas/ — 82,525 features across 18 Geneformer V2-316M layers. Source: https://github.com/Biodyn-AI/geneformer-atlas.
2. **scGPT Feature Atlas:** https://biodyn-ai.github.io/scgpt-atlas/ — 24,527 features across 12 scGPT whole-human layers. Source: https://github.com/Biodyn-AI/scgpt-atlas.

**Six integrated views per atlas:**
1. **Layer Explorer** — per-layer overview (variance explained, dead features, annotation rates, module counts).
2. **Feature Detail Pages** — for each feature: top-20 genes, ontology enrichments (with *q*-values), activation frequency, decoder norm, SVD alignment flag, cell-type enrichments, co-activation module membership, causal patching specificity (where applicable).
3. **Module Explorer** — Leiden communities per layer with force-directed layouts and biological identity labels.
4. **Cross-Layer Flow** — information highways: source feature → top-50 target features with PMI edge weights.
5. **Gene Search** — given a gene name, find all features where it is a top-20 gene.
6. **Ontology Search** — given an ontology term, find all features enriched for it.

**Deployment:**
- All feature data pre-processed into compact JSON files served statically via GitHub Pages.
- No server backend; no installation required for end users.
- Source code lives in `repos/bio-sae/atlas/` (Geneformer) and `repos/bio-sae/scgpt_atlas/` (scGPT).

**For a new model:** fork the atlas React app, replace the data preprocessing scripts in `atlas/scripts/` to consume your Phase 2/5/7/8 output JSONs, rebuild with `npm run build`, deploy to GitHub Pages.

---

## 7. Parameters

| Parameter | Default | Alternate / sensitivity range | Used in |
|---|---|---|---|
| `n_cells_training` (K562) | **2,000** (Geneformer) | — | Phase 0 |
| `n_cells_training` (TS) | **3,000** (1K immune + 1K kidney + 1K lung) | — | Phase 0 (scGPT) |
| `max_seq_len` | **2,048** (Geneformer), **1,200** (scGPT) | model-dependent | Phase 0 |
| `activations_batch_size` | **1** (MPS-friendly) | 8–64 (CUDA) | Phase 0 |
| `positions_per_layer_training` | **1,000,000** | 500K–2M | Phase 1 |
| `d_SAE_expansion` | **4×** | {2, 4, 8} per `c1_hyperparam_ablation.py` | Phase 1 |
| `topk_k` | **32** | {16, 32, 64} per `c1_hyperparam_ablation.py` | Phase 1 |
| `optimizer` | **Adam** | — | Phase 1 |
| `learning_rate` | **3×10⁻⁴** | — | Phase 1 |
| `batch_size` | **4,096** | — | Phase 1 |
| `epochs` | **4** | 3–6 | Phase 1 |
| `decoder_norm` | **unit-normalise after every step** | non-negotiable | Phase 1 |
| `top_n_genes_per_feature` | **20** | 10–50 | Phase 2 |
| `annotation_fdr` | **BH α=0.05** | 0.01–0.1 | Phase 2 |
| `annotation_databases` | **GO BP, KEGG, Reactome, STRING≥700, TRRUST v2** | +DoRothEA, +MSigDB | Phase 2 (+ b4) |
| `svd_n_axes` | **50** (top singular vectors) | — | Phase 3 |
| `svd_alignment_threshold` | **cosine > 0.7** | {0.3, 0.5, 0.7, 0.9} per `b1_svd_threshold_sweep.py` | Phase 3 |
| `cross_layer_match_threshold` | **cosine > 0.7** | — | Phase 4 |
| `pmi_threshold_coactivation` | **PMI > 2.0** (fast proxy) | permutation null *p* < 0.001 | Phase 5 |
| `leiden_resolution` | **1.0** | {0.5, 0.75, 1.0, 1.5, 2.0} per `b2_pmi_and_leiden_sweep.py`; stability via ARI vs 1.0 baseline | Phase 5 |
| `leiden_seed` | **42** | — | Phase 5 |
| `causal_patching_probe_layer` | **L11 (Geneformer), L7 (scGPT)** | — | Phase 6 |
| `causal_patching_features` | **50 richly annotated** | — | Phase 6 |
| `causal_patching_cells` | **200 K562** | — | Phase 6 |
| `cross_layer_pmi_positions` | **500,000** | — | Phase 7 |
| `information_highway_threshold` | **PMI > 3** (at least 1 target) | — | Phase 7 |
| `perturbation_targets` | **100** (48 TFs + 52 other) | — | Phase 8 |
| `perturbation_cells_per_target` | **20** | — | Phase 8 |
| `perturbation_control_positions` | **100,000** | — | Phase 8 |
| `perturbation_effect_threshold` | **|effect| > 0.5** | — | Phase 8 |
| `perturbation_fdr` | **Wilcoxon BH q < 0.05** | — | Phase 8 |
| `multi_tissue_pool_size` | **500K K562 + 500K TS** | — | Phase 9 |
| `multi_tissue_layers_trained` | **{0, 5, 11, 17}** | — | Phase 9 |
| `unannotated_jaccard_leiden_resolution` | **0.5** | — | Phase 10 |
| `umap_n_neighbors` | **15**, min_dist **0.1**, cosine | — | Phase 12 |
| `tsne_perplexity` | **20**, cosine | — | Phase 12 |

**Hyperparameter ablation** (from `src_revision/c1_hyperparam_ablation.py`): grid train **9 L11 SAEs** over `expansion ∈ {2, 4, 8}` × `k ∈ {16, 32, 64}`; measure variance explained, dead features, module count, annotation rate. The **4×/32 default used in the primary analyses sits at or near the optimum** on all four metrics.

## 8. Validation

A successful pipeline run must reproduce the following sanity signatures on Geneformer V2-316M + K562 (primary parametrisation) before any new-model claim is trusted:

1. **Training quality** (Phase 1): Geneformer per-layer variance explained **76.8%–85.3%** (mean **81.7%**). Dead features **419/82,944 (0.5%)** total across 18 layers. Mean pairwise decoder cosine **0.033–0.040**. If any metric is > 2× out of range, the SAE training is broken.
2. **Annotation rate U-profile** (Phase 2): highest at L0 (**58.6%**), minimum at **L8 (45.4%)**, recovery to **55–56% at L10–L11**, second decline to **47%–55%** at L15–L17. Deviations > 5 percentage points at any layer indicate annotation database mismatch (most commonly: old STRING version or missing TRRUST v2 → v1 downgrade).
3. **SVD superposition** (Phase 3): exactly **189 of 82,944 (0.2%) features SVD-aligned** at cos > 0.7 threshold. Novel annotation rate **52.5%** vs SVD-aligned **14.3%**. **98.7% of enrichment terms in novel features only**. If SVD-aligned fraction > 1%, the SVD basis is being computed wrong (probably no mean-centring before SVD).
4. **Cross-layer persistence decay** (Phase 4): from L0, matches at L1 = **114 (2.5%)**; L4 = 67 (1.5%); L8 = 10 (0.2%); **L12+ = 0 (0%)**. Monotone decay required. If any layer shows > 5% match rate, check that the decoder-cosine threshold is 0.7 and decoder columns are unit-normalised.
5. **Co-activation modules** (Phase 5): **141 modules across 18 Geneformer layers**; range **6–12 modules/layer**; peak **12 modules at L5**; **96.0–99.5% feature coverage per layer**; **~4,477 mean features/layer in modules**. Leiden resolution 1.0, seed 42. If modules count is 1–2 per layer, PMI threshold is too lax; if > 20, it's too strict or resolution is too high.
6. **Causal specificity** (Phase 6, Geneformer L11): median **2.36×** specificity, **60% of features > 2×**, **12% > 10×**. Top feature F2035 **114.5×**. Mean target Δlogit **−0.116**, off-target **−0.005** (23× ratio). If median < 1.5×, the ablation is not reaching the downstream layers — check that the hidden state is replaced before the next block.
7. **Cross-layer highways** (Phase 7): **97.4–99.8%** of Geneformer features form information highways at each of the 3 primary layer pairs. L11 → L17 is densest (99.8%). Mean max PMI **6.61–6.79**. scGPT shows progressive downstream bottlenecking (95.7% → 62.9%) — a **required** architectural difference signature.
8. **Perturbation specificity (central test)** (Phase 8): **92% detection, 6.2% (3/48) TRRUST TF specificity.** The 6.2% result is the central negative finding; if your pipeline yields > 15%, recheck the BH correction, the |effect| > 0.5 threshold, and the TRRUST target lists (must be TRRUST v2 primary targets, not DoRothEA merged).
9. **Multi-tissue control** (Phase 9): best multi-tissue layer **L11 = 10.4%** (+4.2 percentage points over K562-only), with disjoint gained (ATF5, BRCA1, GATA1, RBMX, NFRKB) and lost (MAX, PHB2, SRF) TF sets. **Non-systematic improvement** is the required signature. Systematic improvement > +10pp would indicate the SAE methodology is bottlenecked, not the model.
10. **Unannotated co-membership** (Phase 10): **95.3–98.5% of unannotated features co-activate with annotated features in biological modules**; **only 2.5–3.1% form standalone clusters**. If > 10% form standalone clusters, your Leiden resolution on the Jaccard graph is too high (should be 0.5).
11. **Cell-type coverage** (Phase 11, scGPT): **≥99% of features at any layer enriched for ≥1 cell type**. Immune features → T/B/macrophage/DC; kidney → proximal tubule/podocyte; lung → alveolar epithelial/pulmonary endothelial. Tissue coherence is required.
12. **Feature-space quasi-orthogonality** (Phase 12): mean pairwise decoder cosine **0.0007**; within-module vs between-module Cohen's *d* = **0.075**. If within-module *d* > 0.5, modules are leaking into each other and Phase 5 is oversaturated.

**Revision sensitivity analyses** (all in `src_revision/`) test the robustness of these signatures to threshold choices:
- SVD cos threshold 0.3–0.9 → SVD-aligned count scales predictably; novel features remain majority under all thresholds.
- Leiden resolution 0.5–2.0 → module count scales predictably; biological coherence (within-module annotation homogeneity) maximised at 1.0.
- Hyperparameter grid 2×/4×/8× × k ∈ {16, 32, 64} → 4×/32 near-optimal.

## 9. Known pitfalls

1. **Forward-hook placement: pre-residual vs post-residual.** The SAE atlas depends on **post-residual** activations (i.e., the hidden state after `block(x) + x`). If you hook the pre-residual output, you capture the block's delta instead of the residual stream. The result is plausibly-looking features with wrong variance explained (often ~30–40% instead of ~80%). Validate with Phase 0 step 8 (compare to a re-run of the model on a single cell).
2. **Decoder unit-normalisation is non-negotiable.** TopK SAEs without per-step decoder column normalisation produce features with arbitrary magnitudes, making pairwise cosine comparisons meaningless. The Phase 5 co-activation graph, Phase 4 cross-layer tracking, and Phase 6 causal patching all require unit-normalised decoder columns.
3. **scGPT causal patching requires true expression values.** scGPT uses continuous-value tokens; if you extract activations using gene IDs alone (with a dummy expression value of 1.0 everywhere), the forward pass at patch time uses those dummies and the causal signal collapses. Source paper Phase 6 on scGPT L7 reports median specificity 0.98× (null) for this exact reason. If you want scGPT causal patching to work, modify the Phase 0 adapter to preserve the original `(gene_id, expression_value)` pairs and re-inject them during the patch forward pass.
4. **Dead features at late Geneformer layers.** Layers 15–17 have 28–70 dead features each (out of 4,608). These are **not** an SAE training bug — they reflect the "re-specialisation" pattern where late layers encode prediction-focused distributed representations and a subset of features effectively carry no information. Report them in `results.json` but do not try to reduce them by increasing training epochs.
5. **Mid-layer revival.** Dead features drop from 70 (L16) to ~6 at L10–L11. This is a real structural signature (the transformer re-organises into compact integrative programs at middle depth). If you see monotone dead-feature growth with depth, your early layers are over-expressive (increase `d_SAE` there) or mid-layers are under-trained (increase epochs).
6. **SAE training on too few positions.** Training on < 500K positions per layer produces unstable feature assignments across random seeds. The source paper's 1M-position subsample is near the minimum for stable Geneformer features. scGPT needs ~500K minimum due to its smaller `d_model`.
7. **PMI memory blow-up.** Naive PMI computation materialises a dense `d_SAE × d_SAE` = 4,608² ≈ 21M cell matrix per layer. Use **sparse accumulation** (`scipy.sparse.csr_matrix` with incremental updates) and only retain edges exceeding the threshold. Source paper Phase 5 uses ~60 GB peak RSS per layer; the `src_revision/b2_pmi_and_leiden_sweep.py` logs peak memory for reference.
8. **Leiden resolution sensitivity.** Default resolution 1.0 yields 6–12 modules per Geneformer layer. Resolution 0.5 yields 3–6; resolution 2.0 yields 15–25. Report ARI against the 1.0 baseline for any resolution choice.
9. **TRRUST target list boundary.** Phase 8 regulatory specificity test is highly sensitive to which target lists you use. **TRRUST v2 primary direct targets only** (not DoRothEA, not ChIP-Atlas, not expanded to indirect). Merging with wider databases inflates specificity to 15–25% in ways that are not reproducible.
10. **BH vs raw p-value confusion in Phase 8.** The 92% detection / 6.2% specificity numbers require BH correction at α = 0.05 on both the feature-response Wilcoxon tests and the Fisher's exact TF-target enrichment tests. Skipping BH on the Fisher's test inflates "specific" TFs to ~15–25%; skipping on both inflates to ~30–40%. Always apply BH.
11. **Multi-tissue control interpretation.** The +4.2 percentage point improvement in Phase 9 is **not** evidence that the SAE methodology scales — the gained and lost TF sets are disjoint, and TF feature representation actually **decreased** (64.5% → 60.5%). Report both the gained/lost sets and the TF feature representation drop together; reporting only the +4.2pp without context is a misrepresentation.
12. **Unannotated guilt-by-association is a weak test.** With 6–12 modules covering 96–99.5% of features, co-membership in a module is nearly unavoidable — any feature will co-activate with *something*. The 95–98.5% co-membership rate is consistent with the null of random assignment, so Phase 10 is a **necessary but not sufficient** test. Truly validating unannotated features requires perturbation experiments (which the source paper did not perform).
13. **Cross-layer tracking is NOT a persistence test for biological content.** 98.2% of L0 features are transient (≤3 layers); 1.8% moderate-persistence features have **lower** annotation rates than transient features. **Persistent ≠ meaningful.** Do not use feature persistence as a proxy for feature importance.
14. **Apple Silicon MPS batch-size constraint.** Batch size > 4 on MPS for Geneformer V2-316M forward passes with `output_hidden_states=True` triggers out-of-memory on 96 GB M2 Max. CUDA users can safely use batch 16–32. Source paper uses batch 1 throughout for portability.
15. **Feature IDs are ephemeral per training run.** Re-training the SAE at a given layer will produce a different feature index ordering, even with the same seed (due to TopK non-determinism at ties). If you need stable feature IDs across runs, match by decoder cosine similarity (threshold 0.9) and use the best-match mapping. This is especially important for Stages 2 and 3 of the mega-pipeline, which reference features by ID.
16. **CUDA vs MPS numeric precision.** Minor variance-explained differences (~0.5%) between CUDA and MPS backends are expected due to floating-point reduction order. Do not treat this as a sanity-check failure.

## 10. Quick-start for new-model evaluation

To apply this pipeline to a new single-cell foundation model `NEWMODEL`:

1. **Implement the Phase 0 adapter** per §6 Phase 0. Export per-layer `layer_{L:02d}_activations.npy` memmaps with the correct dtype, gene-index mapping, and post-residual-stream semantics. Verify with the Phase 0 step 8 integrity check. For models with continuous expression values (like scGPT), preserve the original expression floats for Phase 6.
2. **Run Phase 1 SAE training** at all layers in parallel (if compute permits) or sequentially via `02b_train_all_layers.py` orchestration. Validate variance explained > 75%, dead features < 2%, decoder cosine 0.03–0.05.
3. **Run Phase 2 annotation** against GO BP, KEGG, Reactome, STRING ≥700, TRRUST v2. Check the U-shaped profile signature; early-layer annotation rates should be ≥ 55% and mid-layer minimum ≥ 40%. Deviations indicate either poor SAE training (features not corresponding to coherent gene programs) or model representations that do not encode biology in a recoverable form.
4. **Run Phase 3 SVD comparison.** The 99.8% novelty rate is a strong structural prior — any new model that shows < 95% novelty has a much more linear internal structure than Geneformer/scGPT and Phase 1 may need only a 2× dictionary.
5. **Run Phase 4 cross-layer tracking.** Check that persistence decays monotonically from L0. A model that shows high persistence across many layers has a different internal architecture (more like ResNet skip connections than progressive refinement).
6. **Run Phase 5 co-activation modules at resolution 1.0.** Verify 4–15 modules/layer with >96% coverage; biologically label each at L0 (should map cleanly to cell cycle, metabolism, translation, PPI, signaling, cytoskeleton) and at mid-depth (should shift to integrative programs).
7. **Run Phase 6 causal patching at a mid-to-late probe layer** (~2/3 through the stack). Verify median specificity > 2×. If your model shows median specificity < 1.5×, the causal signal is too weak — investigate whether you've preserved the correct input representation at patch time (most common cause of failure).
8. **Run Phase 7 cross-layer computational graphs** at three layer pairs (early, middle, late). Report information-highway rates and the cross-layer biological cascades table.
9. **Run Phase 8 perturbation response mapping against Replogle K562 CRISPRi** on 100 targets (48 TRRUST TFs + 52 controls). **Report both detection rate and TF specificity rate.** The published baseline for Geneformer is 92% detection, 6.2% TF specificity. If your new model shows TF specificity > 30%, you have the first single-cell foundation model that encodes regulatory logic at the feature level and should publish the result immediately with the full methodology and the multi-tissue control.
10. **Run Phase 9 multi-tissue control** at layers {L0, Lmid, Ltwo-thirds, Llast}. If Phase 8 passed, verify that multi-tissue training does not **increase** TF specificity further (that would indicate the SAE training data was the bottleneck, not the model). If Phase 8 failed, run multi-tissue to confirm the bottleneck is in the model (as for Geneformer: +4.2pp, non-systematic, disjoint gained/lost).
11. **Run Phase 10 unannotated characterisation.** Do not over-claim on "biologically meaningful unannotated features" — the guilt-by-association test is necessary but not sufficient.
12. **Run Phase 11 cell-type enrichment** on the new model's training cell population. Coverage should be ≥ 95% for any well-trained foundation model.
13. **Run Phase 12 visualisation.** Use force-directed graph layout of co-activation modules; UMAP on decoder cosine will fail.
14. **Run Phase 13 atlas deployment.** Fork the `atlas/` React app, replace data-preprocessing scripts, rebuild and deploy.
15. **Compute the success-criteria scorecard** per Table 8. Report all five criteria including the unmet ones. A new model's atlas is **not complete** without the scorecard.
16. **Cross-reference with the companion attention-based pipeline** (`../attention-grn-extraction-and-evaluation.md`). For any scientific claim about what `NEWMODEL` "knows," both pipelines must be run and their verdicts reconciled. The SAE atlas pipeline is necessary but not sufficient; the two pipelines together establish what single-cell foundation models have internalised about regulatory biology.
17. **Proceed to Stage 2** of the SAE mega-pipeline (`02-causal-circuit-tracing.md`) to build the directed computational graph among SAE features. Stage 2 consumes the trained SAE weights and feature annotations from this pipeline as direct inputs.
