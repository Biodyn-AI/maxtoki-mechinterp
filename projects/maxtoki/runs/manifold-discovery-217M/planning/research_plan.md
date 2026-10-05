# Research plan — manifold discovery / extraction / compactification on MaxToki-217M

This file commits to the **Phase 0** design choices required by
`pipelines/manifold-discovery-extraction-compactification.md`. Once committed,
these choices are frozen for the remainder of the run. Any deviation must be
recorded as an explicit decision in this file with date + reason.

## 0.1 Target foundation model

- **Model:** MaxToki-217M-HF (Llama-architecture; 11 layers × 8 heads = **88 attention units**)
- **Checkpoint:** `projects/maxtoki/setup/MaxToki-217M-HF/`
- **Adapter:** `projects/maxtoki/setup/maxtoki_adapter.py` (already validated in spectral-geometry-217M, attention-grn-217M, sae-atlas-217M runs)
- **Hidden / vocab dims:** `hidden_size=1232`, `vocab_size=20275`. The pipeline's "operator in gene space" is the per-head **value-projection matrix** `A_{ℓ,h} ∈ ℝ^{d_v × d_v}` where `d_v = head_dim = 154` per head, OR the per-head `o_proj` chunk projected back into hidden space. We follow the source-paper convention of treating the per-head value/output projection as the operator and apply it in the model's residual hidden space (not the gene vocabulary), since for Llama the "input gene-vocabulary" `x A_{ℓ,h}` interpretation does not transfer cleanly. **Per-head operator definition for MaxToki:** `A_{ℓ,h} ≡ (o_proj_ℓ block-h)^⊤ ∈ ℝ^{154 × 1232}`. Pooled drift then operates on hidden states sampled at gene-token positions.

  > **Deviation from source paper, to be tracked:** the source paper's scGPT operator acts in gene-vocab space (`d=1200`). For MaxToki (Llama causal LM with subword/gene-symbol tokenizer) we must apply the operator at the per-token hidden state, then project back to gene scores via the gene-token positions. Effective feature dim per head: **154** (head_dim). Pooled drift dim across early/mid/late blocks: **2 × hidden = 2,464**.

## 0.2 Target biological system (primary, H65)

- **System:** Hematopoietic developmental ordering.
- **Why this:** matches the source paper's primary case study. Tabula Sapiens immune is already preprocessed (`<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad`) and SAE-atlas-217M's existing cell-type / branch annotations can be reused as the anchor backbone (saves a full re-annotation pass).
- **Biological ruler:** developmental-stage-ontology distances on a **curated stage DAG**. Source: Laurenti & Göttgens 2018 + Paul et al. 2015 + the hand-curated stage assignments already used in `runs/sae-atlas-217M`.
- **Anchor schema:** `(donor × tissue × cell-type × stage)` quadruples with cell-count threshold ≥ 5.
- **Branch groups (7):** HSC/progenitor, granulocyte, monocyte/macrophage, B lineage, T lineage, NK, plasma. Same schema as source paper.

## 0.3 Null branch (mandatory pairing)

- **Null name:** `H65_null_shuffled`
- **Construction:** identical anchor set to H65, but the stage labels are randomly permuted *within branch group* (preserving branch composition but breaking developmental ordering). This isolates the developmental signal from any branch-level co-expression structure.
- **Required behaviour:** must FAIL all four quality gates. If the null passes, Phase 1 is detecting pipeline artefacts.

## 0.4 Second-manifold target (Phase 13, recommended)

- **System:** Intercellular communication (H38).
- **Ruler:** OmniPath ligand–receptor interaction-path distances, square-root-weighted graph distances (Türei et al. 2021).
- **Data:** Tabula Sapiens immune + lung + kidney (all three already on disk). Mirrors source paper's expansion (immune+lung+kidney) for the v11 H38 rescue.

## 0.5 External validation panels

| Panel role | Source dataset | Expected size |
|---|---|---|
| Internal (Phase 1 train) | Tabula Sapiens immune | ~290 anchors |
| Strict non-overlap external (Phase 7) | Tabula Sapiens immune held-out donors | ~600 anchors |
| Multi-donor zero-shot transfer (Phase 8) | Krasnow lung SmartSeq2 + remaining TS donors | ~150 anchors, 7+ cohorts |
| Expected-to-fail negative control (Phase 7) | Tabula Sapiens lung (non-hematopoietic) | ~400 anchors |

## 0.6 Quality gates (frozen — see `reports/quality_gates_spec.json`)

- Trustworthiness ≥ **0.80**
- Random holdout Spearman ≥ **0.20**
- Donor holdout Spearman ≥ **0.20**
- Clade / branch holdout Spearman ≥ **0.20**
- Blocked-permutation *p* ≤ **0.001** (≥ 2,000 permutations)

These are committed before Phase 1. Any positive branch must pass all five; any null branch must fail at least one.

## 0.6.1 User-confirmed scoping decisions (2026-05-03)

- **Variant scope:** **217M only**. 1B is out of scope for this run. If the 217M result is positive and the user later wants a 1B replication, that becomes a separate run under `runs/manifold-discovery-1B/`.
- **Phase 12 benchmark budget:** **start at 24 Robust-V2 splits.** Escalate to the full 88-split campaign only if the 24-split result already shows a clear cell-head advantage on at least one BH-significant endpoint. (Source paper does the same — its 88-split number is one 40-split run plus two 24-split core-only runs.)

## 0.6.2 Phase 1 stage-label derivation (caveat from data inspection)

Tabula Sapiens immune `obs.development_stage` is **donor age** ("61-year-old stage"), not hematopoietic differentiation stage. Phase 1 must derive hematopoietic stage labels from a curated mapping over `cell_type` (45 unique) + `free_annotation` (113 unique) per Laurenti & Göttgens 2018 / Paul et al. 2015. Plan: build the mapping in `scripts/phase1_anchor_construction.py` as an inline dict, then validate that every kept anchor has a defined stage. Cells without a confident mapping are excluded from anchors but kept available for the Phase 1 multi-donor zero-shot panel.

Bone-marrow representation in TS is limited (mostly mature lineage cells), so HSC / CMP / CLP anchors may be sparse. If sparse, the H65 internal panel will be **mature-leaf-dominated** and the developmental-stage ruler will compress to short paths. This is not a fatal issue — the source paper's Tabula Sapiens panel is also mature-leaf-dominated — but it should be reported in Phase 1 outputs.

## 0.7 Run order (cheap → expensive)

1. **Phase 1 — anchor construction.** Build internal + external + multi-donor + lung-control anchors. Dump centroid arrays + `d_target` matrices.
2. **Phase 4 — operator library export.** Read all 88 per-head matrices from the MaxToki-217M checkpoint. No forward pass needed; this is `~1 min` of file I/O.
3. **Phase 5 — LET training (anchor variant first).** Cheapest LET head. Gates checked.
4. **Phase 3a — autonomous hypothesis sweep.** With Phase 5 baseline running, iterate over featurisation × fitting-method × layer-block-partition. Each branch is one Phase 5 run with a different operator. Driven by the `/loop` skill.
5. **Phase 5 (cell-trained, hybrid variants)** on the winning operator.
6. **Phases 6–8** probes, external, zero-shot.
7. **Phase 9 — head/layer attribution.** 88 single-head LET runs.
8. **Phase 10 — compaction chain.**
9. **Phase 11 — factor ablation.**
10. **Phase 12 — Robust-V2 benchmark.** Tractability check first; may de-scope to ~24 splits if commodity hardware can't do 88.
11. **Phase 13 — H38 generalization** (only if Phase 7+8 pass on H65).

## 0.8 Status tracker

See `runs/manifold-discovery-217M/STATUS.md` (updated by every loop iteration).
