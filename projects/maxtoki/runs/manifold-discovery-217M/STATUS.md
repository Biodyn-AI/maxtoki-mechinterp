# Run status — manifold-discovery-217M

This file is the loop iteration tracker. Each `/loop` wake-up reads this, picks the next unfinished sub-task, executes it, then updates this file.

**Run RE-EXTENDED:** 2026-05-07 — Phase 14 Sweep #3 (6 fresh candidates: H107-H112) + backfill validation for H38 LITE (external + zero-shot) and H95 (zero-shot). **All 4 confirmed manifolds now externally validated; 3 of 4 also zero-shot transferable:**
- H65 developmental: internal ✅, external ✅, zero-shot ✅
- **H38 LITE signalling**: internal ✅, **external ✅ (trust 0.883, donor 0.759, cat 0.594)** [new], zero-shot 3/4 gates (trust 0.787 just under) [new]
- H95 effector modality: internal ✅, external ✅, **zero-shot ✅ 3-gate (trust 0.819, donor 0.794)** [new]
- H103 B-cell maturation: internal ✅, external ✅, zero-shot ✅

Sweep #3 produced 0 new POSITIVE manifolds, 4 new DIRECTIONAL_TRUST_SUBTHRESHOLD (H107, H109, H111, H112), 1 INCONCLUSIVE (H108 borderline), 1 decisive negative (H110 tissue-class — donor holdout 0.036, useful negative finding showing tissue-of-origin is NOT a separable axis in the residual). 11 cumulative directional candidates suggest a continuum of biological geometries rather than a small finite set; trust gate selects only ordinal-graph-distance rulers cleanly.

See FINAL_SUMMARY §14-16 for full registry + interpretation.

**Run EXTENDED + STOPPED:** 2026-05-05 — Sweeps #1 (Hamming, 6) + #2 (ordinal, 6) + Phase 7 external validation for H95 and H103 + Phase 8 zero-shot for H103. **4 confirmed manifolds for MaxToki-217M, 3 of which transfer externally:**
- H65 developmental: internal ✅, external ✅, zero-shot ✅
- H38 LITE signalling: internal ✅
- **H95 effector modality**: internal ✅, external ✅ (trust 0.888, donor 0.812)
- **H103 B-cell maturation**: internal ✅, external ✅ (trust 0.877), zero-shot ✅ (trust 0.857, ρ 0.865)

See FINAL_SUMMARY §14-15 for full registry + interpretation. Loop ending.

**Run COMPLETE:** 2026-05-04 — see `FINAL_SUMMARY.md`. **All 13 phases addressed at LITE+ scope.** H65 case full quality (Phases 0-11 + external + zeroshot + lung_nonhema). Phase 12 LITE anchor-level done (full cell-level Robust-V2 with scVI/Palantir still deferred). Phase 13 LITE H38 done with curated signalling-category ruler — POSITIVE, demonstrates pipeline-level generalisability (full OmniPath H38 still deferred). The two LITE phases are clearly scoped in the script and FINAL_SUMMARY §11 — use as relative-comparison signal, not source-paper-equivalent measurements.

**Run started:** 2026-05-03  
**Target model:** MaxToki-217M-HF  
**Primary branch:** H65 hematopoietic developmental ordering  
**Null branch:** H65_null_shuffled  
**Secondary branch:** H38 intercellular communication (OmniPath)  
**Tracker tasks:** see `TaskList` (#1–#12)

## Phase status

| Phase | State | Notes / artefact path |
|---|---|---|
| 0 — Design choices | ✅ done | `planning/research_plan.md`, `reports/quality_gates_spec.json` |
| 1 — Anchor construction | 🟡 internal+zeroshot+lung done; external running bg `bj0d7e7ph` | external 12k cells subsampled, ETA ~3hr MPS. **lung_nonhema panel still TODO** (current lung_control was actually lung-resident immune, see iteration log). |
| 7 — External validation (lung) | 🟢 lung_control passed all gates — methodology issue (panel was lung-immune cells, not non-hema) | Need to rebuild lung_nonhema panel post-external |
| 8 — Zero-shot transfer | ✅ POSITIVE | trust 0.827, rand 0.893, donor 0.870, branch 0.317. Global ρ 0.899 on full zeroshot panel. |
| 4 — Operator library | ✅ done (1.7s) | 88 per-head ops + pooled drift saved |
| 5 — LET anchor head | ✅ POSITIVE | trust 0.811, rand 0.834, donor 0.716, branch 0.370. Null fails trust (0.799) + branch (-0.001). Report at reports/quality_gates_let_anchor.json |
| 9 — Head/layer scan | ✅ done in 4672s | Top-1 L10H6 (trust 0.882, branch 0.467). 13/88 pass trust gate. Late-layer dominance — diverges from scGPT's L2H5. Report: reports/head_layer_screen.csv |
| 10 — Compaction chain | 🟡 running bg `b7e56l1zj` | full-drift → L10H6 single → rank-{8,16,32,64} SVD → hard_sparse_16f_60g. ETA ~30-45 min. |
| 2 — Quality gate spec | ✅ done | `reports/quality_gates_spec.json` |
| 3a — Autonomous hypothesis sweep | ⬜ blocked on Phase 1 + 4 + 5-init | iterations under `iterations/iter_XXXX/` |
| 3b — Author-led closure | ⬜ blocked on 3a winner | — |
| 4 — Operator library | (see above row) | done |
| 5 — Stage-2 LET adaptor | (see above row) | anchor-variant script ready, blocked on Phase 1B-internal centroids |
| 6 — Stage-3 probes | ⬜ blocked on Phase 5 winner | — |
| 7 — Strict non-overlap external | (see above) | — |
| 8 — Zero-shot transfer | (see above) | done |
| 9 — Head/layer attribution | (see above row) | running |
| 10 — Compaction chain | (see above row) | running |
| 11 — Factor ablation | ⬜ blocked on Phase 10 rank-64 | — |
| 12 — Robust-V2 benchmark | 🟢 LITE done | extracted beats raw (BH-q 0.049 on branch+stage); not signif over PCA/SVD/frozen-MaxToki at anchor scope. Full cell-level deferred. |
| 13 — H38 generalization | 🟢 LITE done | trust 0.814, branch 0.293; null fails all 4 gates. Pipeline-level generalisability validated at lite scope. Full OmniPath H38 deferred. |

## Loop protocol (read on every wake-up)

1. Read this file. Identify the lowest-numbered phase still in `⬜` or `🟡` state.
2. Read `planning/research_plan.md` for committed design choices. Never deviate without recording the decision here.
3. Read `reports/quality_gates_spec.json`. Never relax gates.
4. Execute the next sub-task (one per iteration is fine — keep iterations tractable).
5. Append a one-line entry to the iteration log below with timestamp + what was done + verdict.
6. Update the phase table above.
7. If a positive-branch promotion is at stake, also update the registry entry under `reports/hypothesis_registry.json`.
8. Decide self-paced cadence for next wake-up (minutes for fast runs; ~20–30 min for long compute).

## User-confirmed scoping (2026-05-03)

- 217M only (no 1B for this run)
- Phase 12 starts at 24 splits; escalate to 88 only if 24 already shows clear advantage

## Iteration log

(prepend new entries; format: `YYYY-MM-DDTHH:MM phase_X — short_summary [verdict]`)

- 2026-05-07 phase 17 — ✅ Cross-axis H65 × H95: PARTIALLY_ORTHOGONAL. Cross-head latent ρ=0.642 vs ground-truth ruler ρ=0.814 (Δ=−0.173); null check ρ=0.022. First quantitative support for multi-axis-hub claim.
- 2026-05-07 phase 16 — graph-DAG rescue for H107 (4 buckets × subclass tree, 12 distinct distances): trust 0.769 ≈ flat Hamming's 0.771. Trust ceiling is biology-bounded, NOT Hamming-resolution-bounded. Rand/donor improved (0.772→0.891, 0.690→0.842).
- 2026-05-07 phase 15 — H101 directional: external trust 0.717, zero-shot 0.706 (both BELOW internal 0.764). Trust ceiling not a panel-size artifact for H101 — flat ordinal projection of branching tree caps trust by construction.
- 2026-05-07 phase 15 — ✅ H95 zero-shot POSITIVE_3GATE_FALLBACK (trust 0.819, rand 0.806, donor 0.794, ρ 0.800). Closes H95 row of validation table.
- 2026-05-07 phase 15 — ✅ H38 LITE external POSITIVE all 4 gates (trust 0.883, rand 0.803, donor 0.759, cat 0.594, ρ 0.804). Zero-shot trust 0.787 just under 0.80 (3/4 gates pass) — directional positive on the smaller cohort.
- 2026-05-07 phase 14 sweep#3 — 6 candidates (H107-H112). 0 new POSITIVE. H107/H109/H111/H112 DIRECTIONAL_TRUST_SUBTHRESHOLD. H108 INCONCLUSIVE (rand 0.698 borderline). **H110 tissue class: useful negative** — donor holdout 0.036 ≈ null, tissue-of-origin not a separable residual-stream axis.
- 2026-05-07 phase 14 — verdict-classification post-hoc fix on hypothesis_registry_sweep3.json (script's chained `is False and (... or ...)` clause didn't fire; replaced with intermediate booleans + reclassified saved values).
- 2026-05-04 phase 6 — ✅ probes on anchor head: branch 0.906 / stage 0.884 / cd4-cd8 AUROC 1.0 / mono-macro 1.0 / pseudotime 0.567. Anchor-level (small test sets); cell-level Phase-12 differs.
- 2026-05-04 phase 10 — ✅ all 7 compaction variants pass all gates. compact-top1 (L10H6) BEATS full-drift (trust 0.882 vs 0.811). Hard-sparse 7.7 KB still passes (trust 0.909). 2,400× compression.
- 2026-05-04 phase 11 — ✅ MAJOR DIVERGENCE FROM SOURCE PAPER. Top-4 SVD factors of L10H6 explain only 18.3% of pooled impact (paper: 66.2%). Core (top-4) sufficiency collapses branch from 0.973 to 0.123. MaxToki H65 manifold geometrically real but mechanistically distributed.
- 2026-05-04 phase 1B-external — ✅ done in 12,789s (3.5 hr). 600 anchors / 12k cells / 20 per anchor.
- 2026-05-04 phase 7-external — ✅ POSITIVE. trust 0.896, rand 0.886, donor 0.887, branch 0.346. Global ρ 0.882. All four gates pass on strict non-overlap external panel.
- 2026-05-04 phase 1B-lung_nonhema — first attempt assertion-failed (panel name not in allowed set). Patched and re-launched (bg b32rky21j). 1500 cells, ~25 min.
- 2026-05-04 phase 1A-lung_nonhema — ✅ done in 55s. 50 anchors / 1,500 cells from 19 non-hema TS lung cell types (alveolar type 2, basal, endothelial, etc.). Random H65 stage labels assigned. Awaiting external completion to free MPS for forward pass.
- 2026-05-04 phase 1B-external — slowed to ~1.12 s/cell at 22%; ~140 min more.
- 2026-05-04 phase 1B-lung_control v3 — ✅ done in 1574s. 51 anchors / 2,124 cells.
- 2026-05-04 phase 1B-external v3 — kicked off as 2nd in chain (bj0d7e7ph). 12,000 cells, ETA ~3 hrs on MPS.
- 2026-05-04 phase 7-lung_control — METHODOLOGY ISSUE. Lung "negative control" PASSED all gates (trust 0.801, branch 0.522, global ρ 0.921). Reason: my Phase 1A built lung_control from cells where cell_type maps to hematopoietic stage, i.e. lung-resident immune cells — not non-hematopoietic lung cells. These belong to H65 manifold, so passing is correct biology but invalid as negative control. Need to rebuild a `lung_nonhema` panel using cells whose cell_type does NOT map to a hematopoietic stage (epithelial/stromal/endothelial). TODO after external completes.
- 2026-05-04 phase 1B-zeroshot v3 — ✅ done in 4445s. centroids (160,12,1232), d_target (160,160) range [0,9] median 6.
- 2026-05-04 phase 8 — ✅ POSITIVE zero-shot transfer. Frozen anchor head: trust 0.827, rand 0.893, donor 0.870, branch 0.317. Global ρ on full zeroshot 0.899. All four gates pass on completely separate cohort without retraining.
- 2026-05-04 phase 1B-remaining v3 — chained lung_control + external (bg bj0d7e7ph) under caffeinate. Lung at 42/2124 cells (2%); external 12k cells next.
- 2026-05-04 phase 1B-remaining — v2 also died (memory pressure: only 56MB free, system load 5+). v3 subsamples panels via stratified per-anchor cap (zeroshot 7796→5120, external 30k→12k, lung unchanged). Launched zeroshot v3 (bg be0qzru9k) under caffeinate.
- 2026-05-04 phase 1B-remaining — first attempt died silently at zeroshot 6665/7796 (likely macOS App Nap / sleep). Re-launched under `caffeinate -i` (bg bgtz7th0s).
- 2026-05-04 phase 9 — ✅ done in 4672s. **Top-1 L10H6** trust 0.882, branch 0.467. 13/88 pass trust gate; 82/88 pass branch. Late-layer dominance — divergent from scGPT's L2H5 source-paper finding.
- 2026-05-04 phase 10 — kicked off (bg b7e56l1zj) using L10H6 as compact-V1. SVD ranks {8,16,32,64} + hard sparse 16f×60g. ~30-45 min on CPU.
- 2026-05-04 phase 9 — at L4H0 (~37%); ETA ~50 min more. Best so far: L2H1 trust 0.833 (passes), L1H5 branch 0.488. Phase 1B-zeroshot at 2945/7796 (38%) on MPS in parallel.
- 2026-05-04 phase 12 LITE — ✅ done in 100s. Anchor-level 9-split LOO over 5 features × 5 endpoints. Extracted head ≫ raw_log1p (BH-q 0.049 on branch+stage); not significantly superior to PCA/SVD/frozen-MaxToki at this scope.
- 2026-05-04 phase 13 LITE H38 — ✅ POSITIVE in 385s. Trust 0.814, rand 0.760, donor 0.673, category-holdout 0.293; null fails all 4 gates (rand/donor/branch ≤ 0.026). Same MaxToki residual carries both H65 dev and H38 signalling geometries.
- 2026-05-03 phase 5-anchor — ✅ POSITIVE on H65 with paired null failing decisively on branch holdout. Report saved.
- 2026-05-03 phase 9 — kicked off (bg b5sppgzlh) — 88 single-head LET-10D, CPU. Will produce reports/head_layer_screen.csv ranked by composite score for use in Phase 10 compaction.
- 2026-05-03 phase 1B-internal — ✅ done in 10985s. centroids (290,12,1232), d_target (290,290) range [0,9] median 7, null-shuffled saved.
- 2026-05-03 phase 1B-remaining — chained zeroshot → lung_control → external (bg bpmrs1exd) sequential on MPS. Estimated total ~6 hours (zeroshot 1.6h + lung 0.5h + external 6h... actually external alone may be 5-6h).
- 2026-05-03 phase 5-anchor — first run died on operator_index.json key mismatch (block_partition uses "early"/"mid"/"late", not *_layers). Fixed. Second run died because β=1/π fixed couldn't fit a [0,9]-range ruler — fit term stuck at ~34. Made β learnable (init scaled to ruler max), routed Phase 5 to CPU to avoid MPS contention with Phase 1B-zeroshot. Resubmitted as bg bhd30azur.
- 2026-05-03 phase 1B-internal — at 11,092/11,804 cells (94%); ETA ~11 min more.
- 2026-05-03 phase 1B-internal — at 8,732/11,804 cells (74%); ETA ~45 min more. Steady.
- 2026-05-03 phase 1B-internal — at 6,608/11,804 cells (56%); ETA ~74 min more. Steady at ~0.85 s/cell.
- 2026-05-03 phase 1B-internal — at 4,720/11,804 cells (40%); ETA ~98 min more. No action this iteration.
- 2026-05-03 phase 1B-internal — at 2,596/11,804 cells (22%); rate 0.73 s/cell on MPS; ETA ~110 min remaining. Authored `phase1bc_run_remaining_panels.sh` to chain external+zeroshot+lung after internal completes (sequential, single MPS).
- 2026-05-03 phase 1A — third run completed in 900s. Panels persisted (internal 11,804 cells/290 anchors; external 30k/600; zeroshot 7,796/160; lung 2,124/51). seq_len median 2043, mean 2190, max 4096.
- 2026-05-03 phase 4 — done in 1.7s. 88 per-head operators (154,1232) + pooled-drift components (A_early/mid/late at (1232,1232)) saved to artifacts/operators/.
- 2026-05-03 phase 1B-internal — kicked off (bg bhqc498pm). Forward pass on MaxToki-217M on MPS, mean-pooling hidden states across gene-token positions. ETA ~100 min.
- 2026-05-03 phase 5-anchor — script authored (scripts/phase5_let_anchor.py). LET head, four-gate eval, paired null-shuffled comparison. Awaiting Phase 1B-internal output.
- 2026-05-03 phase 1A — second run reached tokenization on the internal panel (gathered 11,804 cells) but crashed: my hand-rolled tokenizer was indexing a backed-adata row that resolved to size-1 instead of (60606,). Refactored to use `MaxTokiTokenizer.tokenize_cell` (the adapter's official path, used by spectral-geometry phase 0). Loads cells in chunks of 1,000 from the backed h5ad. Resubmitted as bg bpcsctzux.
- 2026-05-03 phase 1A — first run got through panel selection (internal 290 / external 600 / zeroshot 160 / lung 51 anchors over 11+13+12 disjoint donors) but crashed in gather_cells on a pandas-index bug. Patched: now tracks integer position-in-adata from the start. Resubmitted as bg b2194ifdx.
- 2026-05-03 phase 1A — script authored, h65 stage DAG committed, kicked off via anaconda python (bg baug3vvtr). Note: must use `<CONDA_ROOT>/bin/python` for all subsequent compute (system python3 lacks torch).
- 2026-05-03 phase 0 — design choices committed; primary H65, null H65_null_shuffled, secondary H38; quality gates frozen; user confirmed 217M-only + 24-split start [ok]
