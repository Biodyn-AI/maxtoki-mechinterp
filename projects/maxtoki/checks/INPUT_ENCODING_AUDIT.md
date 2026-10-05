# Input encoding audit — every MaxToki run in `projects/maxtoki/runs/`

Date: 2026-10-01. CPU only. No model was loaded or run. Only the numpy tokeniser
(`setup/maxtoki_adapter.py`, `MaxTokiTokenizer.tokenize_cell`) was run.
Data files were read row by row with h5py (never whole).

Files:
- `checks/INPUT_ENCODING_AUDIT.csv` — the main table (one row per run / phase / script, 37 rows).
- `checks/input_encoding_audit/` — per-cell numbers (`order_metrics_cells.csv`), per-dataset
  summary (`order_metrics_summary.csv`), data-file probe (`data_file_probe.json`), check that the
  saved run tokens are reproduced (`saved_token_reproduction.json`), and the three scripts used
  (`probe_files.py`, `order_audit.py`, `build_csv.py`).

No existing file was changed.

---

## 0. Results first

1. **Every MaxToki model run except RPE1 fed the model the wrong gene order.**
   All scripts pass `X` straight to `tokenize_cell`, which ranks genes by value / gene median.
   In every file except RPE1, `X` holds **log1p(CP10k)** (Krasnow: log1p(CPM)), not counts.
   So the model ranked genes by log1p(CP10k) / median instead of counts / median.
   - This is a **single** log, not the double log found in the other project (there, `X` was
     normalised and logged a second time before ranking). No MaxToki script here does a second
     log on the model input. The double log appears only in features, labels and baselines
     (section 5).
2. **How wrong the order is** (5 real cells per dataset; 50-cell check in brackets):
   - Spearman correlation between the fed order and the correct order: **0.77–0.89**
     (0.76–0.87). So the order is related, but clearly not the same.
   - Overlap of the first 200 genes: **0.50–0.70** (0.49–0.70). About 30–50% of the 200
     highest-ranked genes are different genes.
   - Overlap of the first 2,046 genes (what a 2,048-token run keeps): **0.91–1.00**.
     In K562, 8% of the kept genes are the wrong genes (most K562 cells have more than
     2,046 genes, so the wrong order also decides which genes are cut).
   - Share of token positions holding the right gene: **0.000–0.027**. Almost no gene sits at
     its correct position.
   - RPE1 (raw counts in `X`): 1.00 on every measure. RPE1 is correct.
3. **Why the order shifts.** The log squeezes expression into a range of about 0.5 to 6. Gene
   medians span orders of magnitude. So after dividing by the median, the median dominates.
   The fed order agrees with a plain "1 / median" ranking at 0.49–0.72 (correct order:
   0.09–0.41). It agrees with the in-cell expression ranking at only 0.19–0.50 (correct order:
   0.56–0.93).
4. **The deployed tokens are exactly `tokenize_cell(X)`.** I rebuilt the tokens of 255 cells
   from 5 saved token files and all 255 match bit for bit (section 4). So this is what the
   model saw, not a guess about it.
5. **Almost every reported MaxToki result depends on it.** Affected: 3 of 4 attention-GRN
   runs (217M K562, Adamson, 1B K562), the SAE training activations (so every SAE result:
   annotation, patching, TF specificity, circuits and CRISPRi, triplets, exhaustive mapping,
   steering), manifold hidden-state centroids, spectral geometry (layers 1–11), topology
   (layers 1–11), and longevity. The v2 / v2b re-runs of the last two days reused the same
   loaders and are affected too.
   **Not affected:** attention-GRN RPE1; anything read from static embedding tables
   (spectral Phase 9b, topology v2 cross-model, and — added in verification — topology
   Phase 14 and Phase 4 cross-model); layer-0 per-gene results (in a Llama
   model, layer 0 is the token embedding, so it does not depend on order); CPU re-analyses
   that test hook arithmetic on the same inputs (v2 zero-edit control).
6. **Cost to re-run with the right encoding: about 29 h of MPS time** run one job at a time,
   plus CPU re-analysis. Add 5.5 h if the deployed Experiment 1 edge census is redone with
   fixed hooks. Section 7 gives the breakdown and a suggested order.

---

## 1. What each data file holds

Checked by reading 5 rows spread across each file (integer test, row sums, max values,
and the sum of expm1(X)). Raw = `raw/X`.

| File (short name) | Shape | `X` holds | Evidence | `raw/X` | `layers` |
|---|---|---|---|---|---|
| `replogle_concat.h5ad` (K562; also HEPG2/Jurkat/RPE1 rows) | 643,413 × 6,546, dense | **log1p(CP10k)** | 0% whole numbers; max 4.2–5.7; sum of expm1(X) 8,272–9,375 per cell (below 10k because the file keeps 6,546 genes) | none | none |
| `ReplogleWeissman2022_rpe1.h5ad` | 247,914 × 8,749, dense | **raw counts** | 100% whole numbers; row sums 13,317–37,804; max 244–436 | none | none |
| `adamson/perturb_processed_symbols.h5ad` | 68,603 × 4,888, CSR | **log1p(CP10k)** (normalised before gene subsetting) | 0% whole numbers; max 5.3–5.5; sum of expm1(X) 5,096–5,470 | none | none |
| `tabula_sapiens_immune.h5ad` | 592,317 × 60,606, CSR | **log1p(CP10k) of decontX counts** | 0% whole numbers; max 5.1–6.3; sum of expm1(X) 9,977–9,998 | raw integer counts (sums 4,687–636,292; the large ones are Smart-seq2 cells) | `decontXcounts` (int32), `scale_data` |
| `tabula_sapiens_immune_subset_20000.h5ad` | 20,000 × 60,606, CSR | same as above | sum of expm1(X) 9,978–10,000 | raw integer counts | same |
| `tabula_sapiens_lung.h5ad` | 65,847 × 60,606, CSR | same as above | sum of expm1(X) 9,979–9,998 | raw integer counts | same |
| `tabula_sapiens_kidney.h5ad` | 11,376 × 60,606, CSR | same as above | sum of expm1(X) 9,993–10,000 | raw integer counts | same |
| `krasnow_lung_smartsq2.h5ad` | 9,409 × 53,514, CSR | **log1p(CPM)** | 0% whole numbers; max 10.1–11.8; sum of expm1(X) 950k–992k | raw integer read counts (sums 58k–1.24M) | none |

Notes:
- In Tabula Sapiens, `raw/X` is the raw count before decontX (ambient-RNA removal). `X` is
  built from `decontXcounts`. For the same cell, `raw/X` sums to 4,687 and `decontXcounts` to 4,548.
  `raw/var` has the same genes in the same order as `var` (checked in all four TS files and Krasnow).
- K562 and Adamson files have **no raw counts**. The right input there is expm1(X) (= CP10k).
  Within a cell this gives the same order as counts / median, because a per-cell scale does not
  change the order.

## 2. How the scripts tokenise (one shared path)

Every MaxToki tokenisation in `runs/` calls `tokenize_cell(X_row, ...)` on the stored `X` row with
no transform. I listed the first argument of every `tokenize_cell` call (36 call sites in
`runs/`, plus `setup/hooks_v2.py`).
None of them applies expm1, reads `raw/X`, or reads `layers/decontXcounts`.

- `setup/maxtoki_adapter.py: tokenize_cell` computes expr / median, sorts high to low and keeps the
  first `max_len − 2` genes. This is correct **if** `expr` is counts or CP10k. It has no check
  that the input is counts.
- `setup/dataset_loader.py` returns `X` unchanged for k562, adamson and rpe1, and its callers treat
  it as counts (`attention-grn-217M/scripts/phase0_extract.py:182` even labels it "raw counts").
- `setup/hooks_v2.py: load_k562_control_cells` (used by v2_circuit_trace, v2_triplets,
  v2_sae_site_check) reads `f["X"]` rows and passes them to `tokenize_cell`. Same problem.
- `runs/longevity-mechinterp-217M/scripts/maxtoki_runtime.py` passes `adata.X` (log1p) with a
  1,024-token cap.

Context lengths used: 2,048 tokens (attention, SAE, circuit, exhaustive, spectral, topology),
4,096 (manifold), 1,024 (longevity).

## 3. Order agreement, measured

Reference (the correct order) = `tokenize_cell` on raw counts (`raw/X` for TS and Krasnow;
`X` for RPE1) or on CP10k (= expm1(X); K562 and Adamson, which have no counts).
For TS I also report agreement against expm1(X) (decontX CP10k) in the CSV; the mean Spearman
changes by at most 0.04, so the choice of reference does not drive the result.

Cells: 5 real cells per dataset, drawn from the cells the runs actually used (the saved token
files, the run's own sampling rule, or the longevity `sampled_obs.csv`). A 50-cell check
(35 for K562 perturbed cells) is in the `all` rows.

Measures:
- **Spearman (full)**: rank correlation of gene positions over all expressed genes, no cut.
- **Spearman (as fed)**: the same over the genes inside both cut sequences.
- **Top-200 / top-2,046**: share of the first 200 / 2,046 genes that are the same genes.
- **Kept set**: share of the genes in the correct cut sequence that the run also kept.
- **Same position**: share of token positions holding the same gene.
- **Double-log variant**: what the other project's bug (normalise and log `X` again) would give.
  For context only; no MaxToki script here does this to the model input. (For RPE1 the "double
  log" is a single log1p(CP10k), which is the error the other runs made.)

| Dataset (runs) | Cells | n | Ref cut | Spearman full: mean (min) | Spearman as fed | Top-200: mean (min) | Top-2,046 | Kept set | Same position | Double-log variant: Spearman / top-200 |
|---|---|---|---|---|---|---|---|---|---|---|
| K562 controls (attention 217M + 1B, SAE training, circuit, triplets, exhaustive) | 5 | 5 | 5 | 0.84 (0.78) | 0.85 | 0.55 (0.48) | 0.92 | 0.92 | 0.001 | 0.60 / 0.31 |
| | 50 | 55 | 53 | 0.83 (0.72) | 0.85 | 0.56 (0.48) | 0.92 | 0.92 | 0.001 | 0.59 / 0.29 |
| K562 perturbed (SAE Phase 8t, v2 TF specificity) | 5 | 5 | 5 | 0.84 (0.83) | 0.85 | 0.53 (0.47) | 0.92 | 0.92 | 0.000 | 0.60 / 0.26 |
| | 35 | 35 | 34 | 0.84 (0.79) | 0.85 | 0.56 (0.47) | 0.92 | 0.92 | 0.001 | 0.60 / 0.29 |
| Adamson controls (attention Adamson) | 5 | 5 | 0 | 0.79 (0.70) | 0.79 | 0.70 (0.66) | 1.00 | 1.00 | 0.004 | 0.48 / 0.45 |
| | 50 | 55 | 0 | 0.81 (0.68) | 0.81 | 0.70 (0.65) | 1.00 | 1.00 | 0.005 | 0.50 / 0.44 |
| RPE1 controls (attention RPE1) | 5 | 5 | 5 | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 | **1.000** | (0.82 / 0.53) |
| | 50 | 55 | 52 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | (0.83 / 0.54) |
| TS immune, 2,048 (spectral, topology immune, steering, SAE Phase 9/11) | 5 | 5 | 2 | 0.89 (0.85) | 0.87 | 0.65 (0.58) | 0.96 | 0.96 | 0.008 | 0.67 / 0.35 |
| | 50 | 55 | 29 | 0.86 (0.68) | 0.84 | 0.62 (0.43) | 0.95 | 0.95 | 0.002 | 0.68 / 0.37 |
| TS immune, 4,096 (manifold internal / external / zero-shot) | 5 | 5 | 0 | 0.87 (0.81) | 0.87 | 0.66 (0.57) | 0.98 | 1.00 | 0.015 | 0.71 / 0.39 |
| | 50 | 55 | 3 | 0.87 (0.75) | 0.87 | 0.63 (0.46) | 0.96 | 0.99 | 0.012 | 0.70 / 0.37 |
| TS lung, 4,096 (manifold lung_nonhema) | 5 | 5 | 0 | 0.87 (0.81) | 0.87 | 0.59 (0.42) | 0.93 | 0.98 | 0.027 | 0.71 / 0.35 |
| | 50 | 55 | 8 | 0.86 (0.59) | 0.86 | 0.59 (0.24) | 0.91 | 0.95 | 0.011 | 0.71 / 0.34 |
| TS lung, 2,048 (topology lung) | 5 | 5 | 3 | 0.88 (0.87) | 0.81 | 0.59 (0.54) | 0.91 | 0.91 | 0.002 | 0.67 / 0.32 |
| | 50 | 55 | 42 | 0.87 (0.56) | 0.80 | 0.59 (0.37) | 0.91 | 0.91 | 0.002 | 0.69 / 0.32 |
| TS immune 20k subset, 1,024 (longevity) | 5 | 5 | 4 | 0.88 (0.83) | 0.69 | 0.65 (0.56) | 0.91 | 0.80 | 0.004 | 0.69 / 0.41 |
| | 50 | 55 | 53 | 0.85 (0.29*) | 0.70 | 0.61 (0.39) | 0.93 | 0.84 | 0.002 | 0.68 / 0.35 |
| Krasnow lung Smart-seq2, 2,048 (topology external lung) | 5 | 5 | 2 | 0.77 (0.71) | 0.69 | 0.50 (0.37) | 0.95 | 0.95 | 0.002 | 0.58 / 0.39 |
| | 50 | 55 | 27 | 0.76 (0.65) | 0.68 | 0.49 (0.30) | 0.95 | 0.95 | 0.002 | 0.55 / 0.36 |

"Ref cut" = cells whose correct sequence is longer than the context and so gets cut.
\* The 0.29 cell has many genes that decontX set to zero, so the raw/X reference contains genes
the run never saw. Against expm1(X) the minimum is 0.70.

What these numbers mean:
- Spearman 0.77–0.89 is not a random order. But the top of the sequence is where a rank-value
  model puts the most weight, and there only 50–70% of the first 200 genes are right.
- The longevity run (1,024-token cap) is hit hardest at the sequence level: 16% of the genes it
  kept are the wrong genes, and the order inside the kept part agrees at only 0.70.
- In TS, 10x cells (most cells) agree at 0.85–0.87. The few Smart-seq2 cells agree better
  (0.94–0.97; 1–7 cells per set, so this is a weak estimate).
- The single-log error made here is milder than the double-log error in the other project
  (K562: 0.84 vs 0.60 Spearman; 0.55 vs 0.31 top-200). It is still large.

## 4. Proof that this is what the model saw

I rebuilt `tokenize_cell(X_row)` for the saved rows and compared with the saved token ids:

| Saved token file | Cells checked | Identical |
|---|---|---|
| `circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz` (K562, 2,048) | 55 | 55 |
| `sae-atlas-217M/outputs/v2_tf_specificity/cells/catalog_*.npz` (K562 perturbed, 2,048) | 35 | 35 |
| `exhaustive-mapping-217M/outputs/v2_steering/cells.npz` (TS immune, 2,048) | 55 | 55 |
| `manifold-discovery-217M/outputs/phase1/cells_internal.npz` (TS immune, 4,096) | 55 | 55 |
| `manifold-discovery-217M/outputs/phase1/cells_lung_nonhema.npz` (TS lung, 4,096) | 55 | 55 |

The other runs did not save tokens. Their code path is the same call on the same `X` (section 2).

## 5. Main table

Order numbers are the 5-cell means from section 3 (Spearman full / top-200 / top-2,046 /
same position). The CSV has the minimums, the 50-cell numbers and the expm1(X) reference.
"Transform" = what is applied to the h5ad values before / median and ranking.

| Run | Phase · script | Data file · `X` | Transform | Correct? | Order agreement | Results that depend on it | MPS re-run |
|---|---|---|---|---|---|---|---|
| attention-grn | Phase 0 217M K562 · `phase0_extract.py` | replogle_concat · log1p(CP10k) | none (log values used as counts) | **No** | 0.84 / 0.55 / 0.92 / 0.001 | all 217M K562 attention numbers (AUROC 0.512, TRRUST 0.483, incremental, curveball, residualised, layer sweep, verdict C1–C4) | 55 min |
| attention-grn | Phase 0b · `phase0b_value_weighted.py` | same | none | **No** | same | VW vs raw attention, VW layer peak | 18 min |
| attention-grn | Phase 4 · `phase4_causal_ablation.py` | same | none | **No** | same | head-ablation results | 45 min |
| attention-grn | Phase 6 CSSI · `phase6_cssi.py` | same | none | **No** | same | CSSI AUROCs | 37 min |
| attention-grn | Phase 0 Adamson · `phase0_any.py` | Adamson · log1p(CP10k) | none | **No** | 0.79 / 0.70 / 1.00 / 0.004 | all Adamson numbers (incl. the C3 "pass") | 27 min |
| attention-grn | Phase 0 1B K562 · `phase0_any.py` (MODEL_DIR=1B) | replogle_concat · log1p(CP10k) | none | **No** | 0.84 / 0.55 / 0.92 / 0.001 | all 1B numbers (AUROC 0.540, incremental +0.0021, residualised 0.510) | 95 min |
| attention-grn | Phase 0 RPE1 · `phase0_rpe1.py` | RPE1 · raw counts | none needed | **Yes** | 1.00 / 1.00 / 1.00 / 1.00 | all RPE1 numbers | 0 |
| attention-grn | v2 / v2verify · `v2_01..v2_11`, `v2verify_*` | stored attention | no model | inherits (3 of 4 runs) | — | every v2 number for K562, Adamson, 1B | CPU |
| attention-grn | v2b · `v2b_attention.py` | same files | rebuilds the order with `tokenize_cell(X)` | **No** (3 of 4) | as Phase 0 | order share F, rank-conditioned and symmetric scores | CPU |
| sae-atlas | **SAE training activations** · `full_12layer_pipeline.py` (and `phase0_extract_positions.py`) | replogle_concat · log1p(CP10k), 500 K562 controls | none | **No** | 0.84 / 0.55 / 0.92 / 0.001 | **every SAE (layers 0–11) and so every SAE-based result** in sae-atlas, circuit-tracing and exhaustive-mapping | 1.9 h |
| sae-atlas | Phases 6, 8t, 9, 11 · `remaining_phases.py`, `phase6_patching.py`, `phase9_multitissue.py`, `audit_a8_*` | replogle_concat + TS immune | none | **No** | 0.84 / 0.53 (K562 perturbed); 0.89 / 0.65 (TS) | patching 1.01×, Phase 8t TF specificity 0/48, GATA1 case study, multi-tissue, cell-type map | 1.5–2.5 h (estimate) |
| sae-atlas | v2 TF specificity · `v2_tf_specificity_extract.py` (+ stats) | replogle_concat | none | **No** | 0.84 / 0.53 / 0.92 / 0.000 | V2_TF_SPECIFICITY_REPORT (GATA1 fails, 0 TFs pass BH) | 52 min |
| sae-atlas | v2 site check · `v2_sae_site_check.py` | replogle_concat | none | **No** | as K562 | SAE site convention (should not depend on encoding) | < 5 min (optional) |
| sae-atlas | v2b annotation FDR · `v2b_annotation_fdr.py` | feature catalogs | no model | inherits | — | annotation rates at chance, per-feature BH counts | CPU |
| circuit-tracing | deployed · `circuit_trace.py` | replogle_concat | none | **No** | as K562 | 2.14M edges (already superseded: hook bug) | — |
| circuit-tracing | v2 trace · `v2_circuit_trace.py` (hooks_v2 loader) | replogle_concat | none | **No** | as K562 | 272,905 v2 edges, 58.3% inhibitory, v2 CRISPRi direction test | 4.3 h |
| circuit-tracing | v2 spot check · `v2_circuit_spotcheck.py` | replogle_concat | none | **No** | as K562 | 3-edge check | < 15 min |
| exhaustive-mapping | Exp 1 · `experiment1_exhaustive.py` | replogle_concat | none | **No** | as K562 | 4,970,096 edges (also hook bug) | 5.5 h if redone with hooks_v2 |
| exhaustive-mapping | Exp 2 · `experiments_2_3.py`, `experiment2_rerun.py` | replogle_concat | none | **No** | as K562 | superseded by v2 triplets | — |
| exhaustive-mapping | v2 triplets · `v2_triplets.py` (hooks_v2 loader) | replogle_concat | none | **No** | as K562 | V2_TRIPLETS_REPORT (0 of 512 triplets), follow-up | 1.7 h |
| exhaustive-mapping | Exp 3 steering · `experiment3_rerun.py` | TS immune · log1p(CP10k) | none | **No** | 0.89 / 0.65 / 0.96 / 0.008 | published steering table (already a hook artefact) | — |
| exhaustive-mapping | v2 steering · `v2_steering.py` | TS immune | none | **No** | 0.89 / 0.65 / 0.96 / 0.008 | v2 steering and random-feature null | 35 min |
| exhaustive-mapping | v2 zero-edit · `v2_zero_edit_control.py` | TS immune | none | **No** | same | "steering table is a hook artefact" — compares hooks on the same inputs, so this conclusion stands | 17 min (optional) |
| manifold-discovery | Phase 1A · `phase1a_subsample_and_tokenize.py` | TS immune + lung · log1p(CP10k) | none | **No** | 0.87 / 0.66 / 0.98 / 0.015 | tokens → Phase 1B/C centroids → every H65 number (trust 0.811 / 0.896 / 0.827, branch holdouts, L10H6, compaction, factor ablation), all v2_intervals, v2b_devorder MaxToki and token-bag features | 8.3 h |
| manifold-discovery | lung_nonhema · `phase1a_lung_nonhema_panel.py` | TS lung | none | **No** | 0.87 / 0.59 / 0.93 / 0.027 | negative-control panel | 22 min |
| spectral-geometry | Phase 0 · `phase0_extract.py` | TS immune | none | **No** | 0.89 / 0.65 / 0.96 / 0.008 | layers 1–11: rank collapse 561→94, TwoNN, SV enrichments, STRING, TF-vs-target, B/T compression, GC-plasma angle, BATF/BCL6; reused as topology immune domain | 13 min |
| spectral-geometry | Phase 8 · `phase8_stability.py` | TS immune | none | **No** | same | seed stability | 73 min |
| spectral-geometry | autoloop H3-G · `h3g_centroid_trajectory.py` | TS immune | none | **No** | same | centroid trajectory | 10–20 min (estimate) |
| spectral-geometry | Phase 9b, v2 cross-model | static `embed_tokens` | no tokenisation | **not affected** | — | MaxToki–Geneformer Pearson 0.382 | 0 |
| topology-141 | Phase 0 · `phase0_extract.py` (lung, ext-lung; immune from spectral) | TS lung · log1p(CP10k); Krasnow · log1p(CPM) | none | **No** | lung 0.88 / 0.59 / 0.91 / 0.002; Krasnow 0.77 / 0.50 / 0.95 / 0.002 | layers 1–11 of all topology tests (PH, distances, H16, H123, H91, H141, phases 15–16, seed stability). *Corrected in verification: was "phases 14–16"; Phase 14 reads static tables only.* | 38 min + 20 min (seed 43) |
| topology-141 | v2 cross-model · `v2_crossmodel_*.py`; Phase 14 · `phase14_cross_model_cca.py`; Phase 4 · `phase4_scgpt_cross_model.py` (last two added in verification) | static tables | no tokenisation | **not affected** | — | V2_CROSSMODEL_REPORT; Phase 14 H24/H17/H20 (only `tap=static` rows exist); Phase 4 scGPT alignment | 0 |
| longevity | Stage 1 · `run_stage1.py` → `maxtoki_runtime.py` | TS immune 20k · log1p(CP10k) | none | **No** | 0.88 / 0.65 / 0.91 / 0.004 (kept set 0.80) | G1 gate fail (0.2755 vs null 0.332) | 60–90 min |
| sae-atlas (web app) | `atlas/scripts/extract_and_enrich_missing_layers.py` | TS immune / kidney / lung | log1p(X/sum·1e4) on log1p X (double log) | **No**, but this is a **Geneformer V2-316M** script, not MaxToki | not measured | Geneformer layers in the atlas web app, if used | out of scope |

## 6. Other places that use the wrong scale (not the model input)

These do not change what the model saw, but they change numbers built next to it. Each one
computes `log1p(X / rowsum · 1e4)` on an `X` that is already log1p, so the values are logged twice:

- attention-grn Phase 0 / 0_any / 0b / 6 (K562, Adamson, 1B): HVG choice, gene mean and variance
  features, the v2 knockdown labels (|Δ| ≥ 0.5) and CSSI clusters. Already reported as V3 in
  `V2_EVAL_REPORT.md`; on a single log, variance alone beats attention in all 4 runs.
- circuit-tracing CRISPRi LFC labels (`v2_circuit_lfc.py`, `audit_a2_groupkfold_crispri.py`,
  SAE `remaining_phases.py` Phase 8t/11): pseudobulk means of double-logged values. The sign of the
  LFC is mostly kept, but the size is not.
- exhaustive-mapping steering (Exp 3, v2_steering, v2_zero_edit): "pseudotime" = PC1 of double-log
  expression, which picks the early and late cells.
- spectral Phase 0 and topology Phase 0: HVG / gene-pool choice and the co-expression null matrix
  (`log_expression_pool.npy`). *Corrected in verification:* the topology lung and external-lung
  gene pools are picked on the variance of the stored `X` (a single log), before the double log is
  computed. Only their co-expression null is double-logged. The topology immune pool is a subset of
  the spectral HVG panel, which was picked on double-logged values.
- Correct already: manifold v2b HVG baseline (`v2b_devorder_01_features.py` reads `raw/X`), RPE1
  features, and the v2 TF-specificity knockdown check (uses expm1(X)).

## 7. Re-run cost with the right encoding

MPS times come from the run logs and run_config files (same machine). They assume one job at a
time (this machine crashes when heavy jobs run together). CPU re-analysis is extra, about 1–2 h
per run family.

| Block | What to re-run | MPS time |
|---|---|---|
| 1. SAE retrain (do first; everything SAE-based depends on it) | `full_12layer_pipeline.py` | 1.9 h |
| 2. SAE consumers | Phases 6, 8t, 9, 11 (1.5–2.5 h, estimated from cell counts); v2 TF specificity 0.9 h; site check 0.1 h | 2.5–3.5 h |
| 3. Circuits | v2 trace 4.3 h + spot check 0.25 h | 4.5 h |
| 4. Triplets and steering | v2 triplets 1.7 h, v2 steering 0.6 h (zero-edit 0.3 h optional) | 2.3 h |
| 5. Attention-GRN | 217M K562 (Phase 0, 0b, 4, 6) 2.6 h; Adamson 0.45 h; 1B K562 1.6 h | 4.6 h |
| 6. Manifold | four TS panels + lung_nonhema | 8.7 h |
| 7. Spectral | Phase 0 0.2 h, Phase 8 1.2 h, H3-G 0.3 h | 1.7 h |
| 8. Topology | lung + ext-lung 0.65 h, seed-43 lung 0.33 h (immune comes from block 7) | 1.0 h |
| 9. Longevity | Stage 1 | 1–1.5 h |
| **Total** | | **about 28–30 h** (+5.5 h for a fixed-hook Experiment 1) |

Things to know before re-running:
- K562 and Adamson have no counts in the file. Feed expm1(X). TS and Krasnow: feed `raw/X`
  (or `layers/decontXcounts`; the two references give agreement numbers within 0.01–0.04).
- After an SAE retrain, feature numbers change. Scripts that assert they match the deployed
  run (for example `v2_circuit_trace.py` checks its 120 source features against the deployed edge
  file) will stop. They need their selection re-run, not the old feature IDs.
- The existing fix in the other project (`codebase/route_genemanifold/ctx_tokenise.py`, which reads
  `raw/X` and has a `check_counts` guard) can be copied. A guard in `tokenize_cell` (refuse input
  with no whole numbers unless told it is CP10k) would stop this for good.

## 8. What was not done

- **No model was run.** So this audit says how much the input changed, not how much any result
  changes. Spearman 0.77–0.89 means the model saw a related order. Some results may move little;
  results that depend on the top of the sequence or on which genes are kept may move more.
- 5 cells per dataset, as asked, plus a 50-cell check. The two agree closely.
- MPS times for SAE Phases 6/8t/9/11 and spectral H3-G are estimates (no timing in their logs).
- `tabula_sapiens_kidney.h5ad` is used only by the Geneformer atlas script, so I did not measure
  an order there.
- I did not check projects outside `projects/maxtoki/runs/` and `setup/`, and I did not check
  whether the papers in `paper*/` quote each number listed above.

## Plain-words summary

All MaxToki runs except RPE1 gave the model genes ranked from log-scaled values instead of raw
counts. The log flattens expression, so the gene's median took over the ranking. The fed order
still correlates 0.77–0.89 with the right order, but 30–50% of the top 200 genes are different
and almost no gene is in its right slot. I proved this is what the model saw by rebuilding 255
saved token sequences exactly. Nearly every MaxToki result depends on it, including all SAE work,
because the SAEs were trained on these inputs. RPE1, static-embedding comparisons and layer-0
results are safe. Re-running everything with the right input costs about 29 hours of MPS time,
one job at a time, starting with the SAE retrain.

---

## Verification notes

Date: 2026-10-01. Independent re-check with my own code. CPU only. Data files read one row at
a time with h5py. No model was run. Scripts and outputs are in
`checks/input_encoding_audit/verification/` (`verify_probe_files.py/.json`,
`verify_order_2cells.py/.json`).

**Verdict: the main finding holds.** Every MaxToki run except RPE1 fed the model genes ranked
from log1p values, not counts. Two parts of the affected-results list were wrong and are now
fixed (V4a, V4b). One line needed a clarification (V4c).

### V1. What each data file holds — confirmed for all 8 files

I read 3 new random rows per file (seed 20261001).

- **K562 (`replogle_concat`) and Adamson:** `X` = log1p(CP10k). New evidence: in every row,
  expm1(X) values are exact whole-number multiples of the smallest value in that cell. So
  expm1(X) is exactly proportional to integer counts, and feeding expm1(X) gives the correct
  order. Sum of expm1(X) is below 10,000 (K562 9,244–9,407; Adamson 5,079–5,326), because
  the files keep only some genes. No `raw`, no `layers`.
- **RPE1:** `X` = integer counts. One small correction to the table: row sums vary more than
  stated. My 3 rows summed to 6,350–11,623 with max 124–142 (the table gives 13,317–37,804 and
  244–436 from other rows). This changes nothing.
- **Tabula Sapiens immune, 20k subset, lung, kidney:** `X` = log1p(decontXcounts / row sum ×
  10,000). It matches to within 0.0025 (float32 rounding) in all 12 rows. `raw/X` is integer and
  a few counts higher than `decontXcounts` (for example 7,024 vs 7,017). `raw/var` has the same
  genes in the same order as `var`.
- **Krasnow:** `X` = log1p(CPM). Detail: the CPM total is 1–3% different from the `raw/X` row
  sum (for example 976k vs 958k). So the CPM was computed on a slightly different total. Within a
  cell this is one constant factor, so the order from `raw/X` and from expm1(X) is the same.

### V2. How the scripts tokenise — confirmed

- `grep "tokenize_cell("` over `runs/` finds 37 calls. 36 use the MaxToki adapter. The 37th is
  the Geneformer atlas script's own function (out of scope, as the audit says).
- I read the first argument of the calls. Every one is a stored `X` row (or `adata.X`). None
  applies expm1 or reads `raw/X` or `layers/decontXcounts`. Where a script also builds
  `X_log`, it uses `X_log` only for features, not for the model input.
- `setup/hooks_v2.load_k562_control_cells` reads `f["X"]` rows. Its users are
  `v2_circuit_trace.py`, `v2_triplets.py` and `v2_sae_site_check.py`.
- Longevity: the upstream `_load_subset_anndata`
  (`repos/longevity-mechinterp/implementation/scripts/run_stage1_longevity_mechinterp.py:336`)
  returns `X` unchanged, and `maxtoki_runtime.py` passes `adata.X` with a 1,024 cap.
- No other code in `projects/maxtoki/` (outside `runs/` and `setup/`) tokenises for MaxToki.

### V3. Order agreement — 2 cells per dataset, own code

Deployed = `tokenize_cell(X row)`. Correct = `tokenize_cell` on expm1(X) (K562, Adamson),
on `X` (RPE1), or on `raw/X` (TS, Krasnow). Cells come from the saved token files where they
exist, otherwise from the run's cell type (random controls, the longevity `sampled_obs.csv`,
or random rows).

| Dataset (context) | Rows | Spearman full | Top-200 | Kept set | Same position | Saved tokens rebuilt |
|---|---|---|---|---|---|---|
| K562 controls (2,048) | 287624, 399621 | 0.878, 0.860 | 0.545, 0.520 | 0.910, 0.909 | 0.000, 0.001 | 2 / 2 |
| K562 perturbed (2,048) | 453432, 456760 | 0.815, 0.856 | 0.600, 0.615 | 0.918, 0.901 | 0.001, 0.002 | 2 / 2 |
| Adamson controls (2,048) | 17094, 59654 | 0.831, 0.784 | 0.695, 0.685 | 1.000, 1.000 | 0.002, 0.008 | — |
| RPE1 controls (2,048) | 141680, 222370 | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 | — |
| TS immune (2,048) | 26453, 456397 | 0.776, 0.793 | 0.545, 0.595 | 0.998, 0.992 | 0.004, 0.001 | 2 / 2 |
| TS immune (4,096) | 407721, 114017 | 0.853, 0.873 | 0.615, 0.560 | 0.996, 0.993 | 0.004, 0.003 | 2 / 2 |
| TS lung (4,096) | 50224, 54703 | 0.964, 0.870 | 0.735, 0.580 | 0.960, 0.995 | 0.002, 0.002 | 2 / 2 |
| TS lung (2,048) | 14829, 54892 | 0.874, 0.870 | 0.625, 0.635 | 0.865, 0.971 | 0.002, 0.000 | — |
| TS immune 20k (1,024; 10x, SS2) | 6781, 217 | 0.840, 0.972 | 0.670, 0.730 | 0.879, 0.954 | 0.002, 0.000 | — |
| Krasnow (2,048) | 8218, 8586 | 0.797, 0.675 | 0.440, 0.430 | 0.951, 1.000 | 0.000, 0.002 | — |

What this shows:
- All numbers fall inside the audit's ranges (its 50-cell minimums included). K562 control row
  287624 is also in the audit's own cell table, and all six measures match to 3 decimals.
- All 10 cells with saved tokens rebuild exactly from `tokenize_cell(X)`.
- RPE1 is 1.000 on every measure, so the RPE1 run is correct.
- For TS, using `decontXcounts` instead of `raw/X` as the correct input changes Spearman by at
  most 0.02.
- The 1/median explanation holds in all 18 non-RPE1 cells. The fed order is closer to a pure
  1/median order than the correct order is, by 0.10 to 0.60 (Spearman). One small point: the
  ranges in section 0 ("0.49–0.72" fed, "0.09–0.41" correct) are the 50-cell means. The 5-cell
  means are 0.46–0.78 and 0.10–0.51. The CSV stores these with a minus sign because of how it
  ranks 1/median.

### V4. Which results are affected — corrections

**V4a. Topology Phase 14 is not affected (fixed).** The audit listed "phases 14–16" as
affected. `runs/topology-141-217M/scripts/phase14_cross_model_cca.py` (H24/H17/H20: CCA,
Procrustes and distance Spearman against Geneformer V2-316M) reads only the static
`embed_tokens` tables. Its output `outputs/phase14_cross_model/cross_model_alignment.csv` has
only `tap=static` rows. Its docstring mentions layers 1–11, but the code never runs them. The
same is true of topology `phase4_scgpt_cross_model.py` (scGPT, `tap="static"`), which the audit
did not list. Phases 15 and 16 read `layer_gene_embeddings_*.npy` and are affected. I fixed
section 0, the section 5 table, the CSV (2 rows) and `build_csv.py` to match.

**V4b. Topology gene-pool choice is a single log, not a double log (fixed in section 6).**
In `topology-141-217M/scripts/phase0_extract.py`, the lung and external-lung pools are picked on
the variance of the stored `X` (line 202), before the double log is computed (line 213). Only
the co-expression null (`log_expression_pool.npy`) is double-logged. The immune pool is a subset
of the spectral HVG panel, and that panel was picked on double-logged values.

**V4c. Layer 0: safe for per-gene vectors, not for the layer-0 SAE (clarification).** Layer 0
is `hidden_states[0]` in the spectral, topology and SAE scripts. In a Llama model that is just
the embedding row, so per-gene vectors at layer 0 (spectral, topology) do not depend on the
input. The layer-0 SAE is different. It was trained on the tokens kept from 500 K562 cells, and
about 9% of the kept genes differ. So its training data changed. The CSV already counts SAE
layer 0 as affected, and that is right. The same goes for any per-cell average at layer 0,
because it depends on which genes were kept.

**V4d. Caveat for the static-table comparisons.** Spectral Phase 9b, the spectral and topology
v2 cross-model reports, and topology Phases 14 and 4 do not use the model input. But their gene
panels come from Phase 0. The spectral 1,500-gene panel (and the topology immune panel taken
from it) was picked on double-logged variance. The topology immune panel also keeps only genes
that appeared in at least one tokenised cell. If Phase 0 is re-run with a corrected gene
selection, these panels change. The static numbers then have to be recomputed on the new panel
to stay comparable. The results themselves are still valid on the panel they used.

**V4e. Covered by the audit but not named.** These inherit the wrong input:
- `manifold-discovery-217M/scripts/audit_residual6_cell_level_benchmark_lite.py` runs the
  model on the saved `cells_internal.npz` tokens.
- `topology-141-217M/scripts/audit_residual9_cross_layer_replication.py` reads the layer
  embeddings.
- Spectral autoloop steps that read Phase 0 outputs.

Longevity Stages 2–5 were never run (G1 gate failed; `FINAL_SUMMARY.md`), so only Stage 1 is
affected. `manifold-discovery-217M/scripts/phase4_export_operators.py` reads weights only. It is
not affected itself, but every use of it with the hidden-state centroids is.

**V4f. Other claims checked.**
- `v2_circuit_trace.py:101` asserts that its source features match the deployed run. So it will
  stop after an SAE retrain, as the audit says.
- The SAE log (`sae-atlas-217M/outputs/full_12layer.log`) shows 551–639 s per layer. That is
  about 2 h for 12 layers, which matches the 1.9 h estimate. The same log shows 1,019,996 token
  positions from 500 cells (2,040 per cell). So almost every K562 cell was cut at the
  2,046-gene limit. This fits the 8–9% of kept genes that change.
- I did not re-check the other MPS times, or which numbers the paper folders quote.

### Corrected affected-results list

- **Affected** (the model saw the wrong gene order):
  - Attention-GRN: 217M K562 (Phases 0, 0b, 4, 6), Adamson and 1B K562, plus their v2,
    v2verify and v2b re-analyses.
  - All SAE work: SAE training (layers 0–11), annotation, patching, Phases 8t/9/11, v2 TF
    specificity, v2b annotation FDR, and the site-check numbers.
  - Circuit tracing: the deployed trace, the v2 trace and the spot check.
  - Exhaustive mapping: Experiments 1 and 2, v2 triplets, Experiment 3 steering, v2 steering,
    and the v2 zero-edit numbers.
  - Manifold: Phase 1A (all panels), lung_nonhema, and every H65 number. That includes
    v2_intervals, the v2b_devorder MaxToki features and audit_residual6.
  - Spectral: Phase 0 layers 1–11, Phase 8 and H3-G.
  - Topology: Phase 0 layers 1–11, and so persistent homology, distances, H16, H123, H91, H141,
    **Phases 15–16**, audit_residual9 and seed stability.
  - Longevity: Stage 1.
- **Not affected:**
  - The attention-GRN RPE1 run.
  - Static-table comparisons: spectral Phase 9b and its v2 cross-model check, topology v2
    cross-model, and **topology Phases 14 and 4**. They carry the panel caveat in V4d.
  - Layer-0 per-gene vectors in spectral and topology. This does not cover the layer-0 SAE.
  - The v2 zero-edit conclusion that the published steering table is a hook artefact. The
    conclusion stands, but its numbers will change.

**Edits made in this verification:** section 0 (not-affected list), section 5 (two topology
rows), section 6 (topology pool scale), `INPUT_ENCODING_AUDIT.csv` (2 topology rows; still 37
rows) and `input_encoding_audit/build_csv.py` (the same two strings). No other file was changed.

**Plain-words summary.** I re-read the data files and re-ran the order check on 20 cells with my
own code. The audit is right: all MaxToki runs except RPE1 ranked genes from log values, and the
saved tokens prove the model saw exactly that. One mistake is fixed. Topology Phase 14, and also
topology Phase 4, compare only the fixed embedding tables, so the input bug does not touch them.
One detail is fixed too: the topology gene pools for lung and external lung were picked on a
single log, not a double log. Everything else in the affected list stands.
