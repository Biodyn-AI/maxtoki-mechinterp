# V2 circuit report — Stage-2 circuit tracing with fixed hooks, and the CRISPRi direction test (item D2)

Date: 2026-10-01. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
Hook library: `setup/hooks_v2.py` (sha256 `1ba6fda5…`, unchanged during the run).
Every number below comes from a file listed in section 11.

## 0. Results first

**Edges (same 200 cells, same 120 source features, same edge rule as the deployed run)**

| | Deployed run (block-deleting hook) | v2 (fixed hooks) |
|---|---:|---:|
| Total edges | 2,144,011 | **272,905** (12.7% of deployed) |
| Share "inhibitory" (removing the source lowers the target) | 88.2% | **58.3%** |
| Edges at the next layer (s → s+1), sum over 4 source layers | 0 | **37,952** |
| Median \|d\| of an edge | 1.30 | 0.65 |

- The deployed edges were almost all made by the hook bug. The deployed hook with an edit of exactly zero
  gives 71,474 edges. That set contains at least 99.1% (median 99.76–99.96% per source layer) of the deployed
  edges of every one of the 120 features. 30 features × 71,474 = 2,144,220, against 2,144,011 deployed edges.
  So the deployed list is close to 30 copies (one per feature) of the same zero-edit set.
- With fixed hooks, the edges look different. Where both runs have an edge, the sign agrees only 57.8% of
  the time and the Cohen's d values barely correlate (Pearson r = 0.07).

**CRISPRi direction test (positive class = target gene goes down after silencing)**

| | Deployed edges | v2 edges |
|---|---:|---:|
| Records (silenced gene × target gene) | 1,503,408 | **698,624** |
| Silenced genes | 248 | **227** |
| Observed decrease | 54.6% | 54.3% |
| Predicted decrease | 89.6% | **50.9%** |
| Accuracy | 53.48% | **49.76%** |
| Always "decrease" (= majority class, = best constant) | 54.61% | 54.31% |
| Accuracy minus best constant [95% CI, silenced genes] | −1.13 pts [−1.61, −0.72] | **−4.55 pts [−5.79, −3.34]** |
| Balanced accuracy | 49.83% | **49.68%** |
| Balanced accuracy minus 0.5 [95% CI, silenced genes] | −0.17 pts [−0.51, +0.14] | **−0.32 pts [−0.83, +0.20]** |
| MCC [95% CI, silenced genes] | −0.0055 [−0.0160, +0.0047] | **−0.0063 [−0.0166, +0.0041]** |
| Accuracy expected from the two class rates alone | 53.64% | 50.08% |
| Target-direction baseline (cross-fitted) | 59.15% | **58.70%** |

- With the fixed edges the circuit still has **no detectable directional skill**. It now predicts "decrease"
  for about half of the pairs instead of 90%, so its accuracy falls to 49.8%: below 50% and 4.6 points below
  the always-decrease rule. Balanced accuracy (49.7%) and MCC (−0.006) are at chance, and their CIs include 0.
- The old 53.5% was above 50% only because the old circuit said "decrease" 90% of the time and decreases
  are the majority. Fixing the hooks removes that bias, and the apparent above-50% result goes with it.
- A simple rule that uses no model at all ("does this target gene usually go down when other genes are
  silenced?", learned on other silenced genes) gets 58.7%. That is 8.9 points above the circuit
  [CI −9.98, −7.99 for circuit minus baseline].

## 1. What I re-ran, and what stayed the same as the deployed run

Deployed script: `scripts/circuit_trace.py` (not modified). I checked every setting in the code.

| Setting | Deployed (`circuit_trace.py`) | v2 |
|---|---|---|
| Cells | 200 K562 non-targeting cells, `np.random.default_rng(42).choice`, sorted, ≤ 2,048 tokens (lines 100–123) | same 200 dataset rows (checked: the spot check re-draws and re-tokenises them, identical rows and tokens) |
| Source layers | 0, 3, 6, 9 | same |
| Source features | 30 per layer, largest summed −log10 p of enrichment (lines 133–145) | same 120 features, in the same order (asserted against `outputs/circuit_edges.csv`) |
| Edit | delta = decode(z with f zeroed) − decode(z), so −z_f · W_dec[:, f]; written as the OUTPUT of block s (deletes block s) | same delta (`hooks_v2.Ablate(s, [f])`), ADDED to the INPUT of block s, where SAE s was trained, computed from the live input |
| Positions edited | all, including `<bos>`/`<eos>` | same |
| Read-out | `output_hidden_states[t]`, t = s+1..11, SAE t; mean over positions of z_ablated − z_clean | hooks_v2 captures of the input of site t (never `output_hidden_states`); same formula |
| Cohen's d | mean / sqrt(var(ddof=1) + 1e-12) over the 200 cells (lines 56–80) | same |
| Consistency | max(#cells with Δz > 0, n − #cells with Δz > 0) / n | same (a cell with Δz = 0 counts against "positive"). A symmetric version is reported as a check only |
| Edge | \|d\| > 0.5 and consistency > 0.7 (line 246) | same |
| Sign | "inhibitory" if mean Δz < 0 (line 253) | same |
| CSV | d and consistency rounded to 4 decimals | same rounding (the rule uses unrounded values in both) |

Two exact speed-ups (checked):
- **Partial forward.** For a source at layer s, the ablated pass starts at block s from the clean input of
  block s. Blocks before s cannot change. Check (`checks/partial_forward_check.json`, 2 cells × 8 features):
  max difference to a full forward 0.0 in the hidden states and 0.0 in Δz.
- **Inactive source features.** If a feature is zero at every position in a cell, the edit is zero and every
  Δz is exactly 0. These cases are filled with exact zeros without a forward pass. Cells 0–4 were run without
  this skip; all 79 inactive cases there gave exact zeros (the script docstring says 70; the files have 79). Over all 200 cells, 2,682 (cell, feature) cases were
  inactive (out of 24,000); every one has exact zeros.

## 2. The interruption and how I resumed

- An earlier session ran cells 0–170 (31 chunks, about 3.8 h) and was stopped when the app quit. No process was left running.
- **Resume check** (`scripts/v2_circuit_resume_check.py` → `checks/resume_check.json`):
  - All 171 cell files: correct shapes, finite values, dataset rows and token counts match, files 0..170 with no gaps,
    no half-written temp files, zero-edit control exactly 0.
  - I recomputed two finished cells from scratch with the same code: cell 170 (the last one before the stop) and
    cell 29 (drawn with seed 20261001). The main array (780 × 4,928 values per cell), the v2 zero-edit control and the
    deployed-hook reproduction all matched **exactly (max difference 0.0)**.
  - One control did not match: the deployed-hook zero-edit control ("legacy_zero") in cell 170, row L0 → L1 only
    (max difference 1.86). The file records this check as failed. I found the cause, fixed it, and repaired the affected values:
- **Cause.** transformers 5.5.4 installs its `output_hidden_states` capture hooks only on the first forward that asks
  for hidden states (`transformers/utils/output_capturing.py:182–203`). PyTorch runs forward hooks in the order they
  were registered. In each chunk, the first such forward was the legacy zero-edit pass at L0 of the first cell, and its
  patch hook was registered before the capture hooks. So in that one pass, `hidden_states[1]` recorded the patched
  tensor. In every other pass (and in the deployed run, which always ran a clean pass first) it records the unpatched
  output of block 0. Affected: legacy_zero row L0 → L1 of the first cell of each chunk (31 cells; I confirmed the set
  of cells with a non-zero value there is exactly the set of first cells). Not affected: the v2 edges, the v2 zero-edit
  control and the deployed-hook reproduction (they never ran first, or never read `output_hidden_states`).
- **Fix.** The trace script now runs one short forward with `output_hidden_states=True` right after loading the model
  (`install_capture_hooks_first`). Cells 171–199 ran with it; the first cell of each new chunk has a legacy L0 → L1 row of exactly 0.
  `scripts/v2_circuit_legacy_fix.py` recomputed legacy_zero for the 31 affected cells into new files
  (`trace_cells_legacy_fix/`). Check (`checks/legacy_fix.json`): rows L0→L2 … L9→L11 equal the saved values exactly
  (max difference 0.0); the repaired L0 → L1 row is exactly 0 in all 31 cells. The trace files were not changed.
- I then ran cells 171–199 with the same settings (4 chunks, 50–64 s per cell). Total trace: 200 cells, 35 chunks,
  4.25 h wall time, 73 s per cell on average. MPS memory at the end of a chunk ≤ 3.6 GB; free memory at chunk start ≥ 5.4 GB.

## 3. Sanity checks (all pass)

| Check | Result | Pass |
|---|---|---|
| Edges now appear at s → s+1 (deployed: none) | L0→L1 34,539; L3→L4 2,349; L6→L7 774; L9→L10 290 | yes |
| Zero-edit control (`hooks_v2.ZeroDelta(s)`, same partial forward, 4 source layers × 200 cells) | max \|Δz\| = 0.0 in every cell; 0 edges | yes |
| Spot check: 3 edges recomputed from scratch (`scripts/v2_circuit_spotcheck.py`) | see below | yes |
| Deployed-hook reproduction: deployed hook + deployed delta, first feature of each source layer | Jaccard 1.0 with the deployed edges of that feature at all 4 layers (39,648 / 20,680 / 8,511 / 2,630 edges); sign agreement 1.0; max \|Δd\| 0.011 | yes |
| Deployed hook with an edit of exactly zero | 71,474 edges; contains a median 99.91% / 99.76% / 99.84% / 99.96% (min 99.69% / 99.09% / 99.45% / 99.81%) of each deployed feature's edges at L0 / L3 / L6 / L9 | (diagnostic) |
| Deployed CRISPRi records rebuilt from the deployed edges | 1,503,408 of 1,503,408 records; predicted sign identical for all; LFC difference 0.0 | yes |
| Deployed CRISPRi table vs the investigation (`projects/maxtoki/verification/crispri/results.json`) | accuracy, balanced accuracy, MCC and both bootstrap CIs identical to all printed digits; target baseline 0.59154 vs 0.5915 | yes |
| LFCs recomputed from the 30 GB h5ad vs the deployed records | 1,503,408 of 1,503,408 exactly equal | yes |

**Spot check details.** Edges drawn with seed 20261001, one from each of L0→L1, L9→L10 (layer pairs with no deployed
edges) and L6→L11. The script does not use `hooks_v2` or the trace code. It re-draws and re-tokenises the cells from the
h5ad, edits with a plain forward hook that ADDS the delta to the output of the module that produces `hidden_states[s]`
(embedding layer for s = 0, block s−1 otherwise), uses the CPU SAEs, and reads `output_hidden_states[t]`.

| Edge | d (trace) | d (from scratch) | consistency (both) | max per-cell \|ΔΔz\| |
|---|---:|---:|---:|---:|
| L0 F3421 → L1 F1053, excitatory | 0.5108 | 0.51075 | 0.885 | 5.2e-9 |
| L9 F4490 → L10 F1671, excitatory | 0.7219 | 0.72186 | 0.835 | 5.2e-9 |
| L6 F1524 → L11 F4558, inhibitory | −0.6667 | −0.66668 | 0.895 | 6.4e-8 |

The d differences (< 5e-5) are the 4-decimal rounding of the CSV.

## 4. Edge counts per (source layer, target layer)

| Source → target | v2 edges | v2 share inhibitory | Deployed edges | Deployed share inhibitory |
|---|---:|---:|---:|---:|
| L0 → L1 | 34,539 | 0.579 | 0 | – |
| L0 → L2 | 31,578 | 0.592 | 146,332 | 0.980 |
| L0 → L3 | 26,811 | 0.596 | 142,019 | 0.984 |
| L0 → L4 | 15,439 | 0.642 | 139,162 | 0.985 |
| L0 → L5 | 16,858 | 0.611 | 132,216 | 0.990 |
| L0 → L6 | 21,746 | 0.575 | 118,418 | 0.986 |
| L0 → L7 | 23,693 | 0.560 | 113,720 | 0.987 |
| L0 → L8 | 25,480 | 0.550 | 93,845 | 0.987 |
| L0 → L9 | 23,418 | 0.555 | 98,722 | 0.989 |
| L0 → L10 | 14,162 | 0.580 | 108,583 | 0.986 |
| L0 → L11 | 10,301 | 0.591 | 96,577 | 0.990 |
| L3 → L4 | 2,349 | 0.816 | 0 | – |
| L3 → L5 | 3,469 | 0.628 | 103,673 | 0.633 |
| L3 → L6 | 3,694 | 0.576 | 95,596 | 0.802 |
| L3 → L7 | 4,962 | 0.541 | 90,816 | 0.827 |
| L3 → L8 | 5,373 | 0.524 | 75,283 | 0.790 |
| L3 → L9 | 4,023 | 0.533 | 82,210 | 0.825 |
| L3 → L10 | 1,686 | 0.536 | 91,162 | 0.892 |
| L3 → L11 | 801 | 0.519 | 81,305 | 0.909 |
| L6 → L7 | 774 | 0.689 | 0 | – |
| L6 → L8 | 594 | 0.704 | 71,274 | 0.548 |
| L6 → L9 | 377 | 0.761 | 71,116 | 0.521 |
| L6 → L10 | 211 | 0.777 | 62,448 | 0.636 |
| L6 → L11 | 114 | 0.798 | 50,591 | 0.689 |
| L9 → L10 | 290 | 0.845 | 0 | – |
| L9 → L11 | 163 | 0.859 | 78,943 | 0.849 |
| **Total** | **272,905** | **0.583** | **2,144,011** | **0.882** |

By source layer (v2 / deployed): L0 244,025 / 1,189,594 (inhibitory 58.2% / 98.6%); L3 26,357 / 620,045 (57.6% / 80.6%);
L6 2,070 / 255,429 (72.1% / 59.0%); L9 453 / 78,943 (85.0% / 84.9%). Source CSV: `outputs/v2_circuit/edges_per_layer_pair.csv`.

Other facts about the v2 edges:
- Edges per source feature: median 195, mean 2,274, max 43,356; 18 of 120 features have no edge.
  Median per feature: L0 6,593; L3 479; L6 45; L9 6. Early-layer edits reach far more downstream features.
- The feature with the most edges, L0 F1883 (43,356), is the only L0 source feature that is active on the `<bos>` and
  `<eos>` tokens. Edits at special tokens can have large effects (V2_HOOKS_REPORT section 3). The deployed run also edited
  every position, so I kept that.
- \|d\| of v2 edges: 10th / 50th / 90th / 99th percentile 0.52 / 0.65 / 1.43 / 3.52 (deployed 0.61 / 1.30 / 3.78 / 5.38).
  16.4% of v2 edges have \|d\| > 1 (deployed 60.9%).
- Symmetric consistency rule (max(#Δz > 0, #Δz < 0) / n > 0.7): 269,634 edges, share inhibitory 57.8%. Only 3,271
  deployed-rule edges (1.2%) fail it. So the zero-counting asymmetry in the deployed rule does not drive the sign balance.

**Comparison with the old 2.14 M edges** (layer pairs t ≥ s+2 only, because the old run has nothing at s+1):
- 3,252,480 possible (source feature, target feature) pairs. The deployed run marks 65.9% of all of them as edges.
- 90.6% of the v2 edges there are also deployed edges (212,820), against 65.9% expected if the two sets were unrelated.
- On these shared edges the sign agrees in 57.8% of cases. Pearson r of d = 0.07, Spearman 0.02.
- So the old edge list is mostly a dense block-deletion pattern that happens to cover most v2 edges, with signs that
  do not track the feature effect.

## 5. How the CRISPRi records are built (same as the corrected deployed metric)

Re-implemented from `scripts/audit_a2_groupkfold_crispri.py:51–162` in `scripts/v2_circuit_crispri.py`
(vectorised; checked by rebuilding the deployed records exactly, section 3).

1. Use every edge whose two ends have a top-gene list (`runs/sae-atlas-217M/outputs/phase2/layer_XX/feature_catalog.json`, first 10 of `top20_genes`, upper case).
2. For each edge, pair each top gene of the source feature with each top gene of the target feature (no self pairs).
3. Per gene pair: evidence = number of such triples; max \|d\|; number from "inhibitory" edges.
4. Keep a pair if evidence ≥ 2 or max \|d\| > 2.0.
5. Predicted decrease = (inhibitory count / evidence) > 0.5. A tie counts as "increase" (72,275 v2 pairs, 10.3%, are ties).
6. Keep pairs whose source gene has a CRISPRi guide with ≥ 10 K562 cells and whose target gene is measured.
7. Observed decrease = LFC < 0. LFC = mean log1p(CP10k) of the guide's K562 cells minus the mean of 3,000 K562
   non-targeting cells (seed 42). Recomputed from the h5ad by `scripts/v2_circuit_lfc.py` (966 candidate source genes;
   248 have ≥ 10 cells; identical to the deployed values).

With the v2 edges: 272,905 edges used, 698,624 records, 227 silenced genes, 6,324 target genes. 698,586 of these
pairs are also in the deployed records (the deployed set is a near-full grid). No v2 LFC is exactly 0.

## 6. CRISPRi table on the identical records

Positive class = observed decrease. Pooled metrics come from the summed confusion counts.

**Uncertainty.** Grouped percentile bootstrap, 2,000 reps, seed 42. Each rep resamples groups with replacement and
recomputes the pooled metrics from the summed counts (the best constant is re-chosen in each rep). Three grouping units:
(a) silenced genes; (b) source-feature groups: silenced genes with the same set of source features (with ≥ 1 edge) whose
top-10 list holds them — 119 groups (v2), 139 (deployed); (c) connected components of silenced genes that share any
source feature — 50 (v2; largest 88 genes), 54 (deployed; largest 95). The target-direction baseline is cross-fitted
once on 5 fixed folds of silenced genes; it is not refit inside each rep.

| Quantity | Deployed edges | v2 edges |
|---|---:|---:|
| Pairs / silenced genes | 1,503,408 / 248 | 698,624 / 227 |
| Observed decrease / increase | 0.5461 / 0.4539 | 0.5431 / 0.4569 |
| Predicted decrease / increase | 0.8956 / 0.1044 | 0.5089 / 0.4911 |
| Confusion (pred dec & obs dec, pred inc & obs dec, pred dec & obs inc, pred inc & obs inc) | 733,979 / 86,971 / 612,473 / 69,985 | 191,985 / 187,457 / 163,528 / 155,654 |
| **Accuracy** | **0.5348** | **0.4976** |
| Always decrease | 0.5461 | 0.5431 |
| Always increase | 0.4539 | 0.4569 |
| Majority class (= always decrease in both) | 0.5461 | 0.5431 |
| Recall on decreases / on increases | 0.894 / 0.103 | 0.506 / 0.488 |
| **Balanced accuracy** | **0.4983** | **0.4968** |
| **MCC** | **−0.0055** | **−0.0063** |
| Cohen's kappa | −0.0036 | −0.0063 |
| Accuracy expected from the two class rates | 0.5364 | 0.5008 |
| Global majority, learned on other source folds | 0.5461 | 0.5431 |
| **Target-direction baseline** (target's usual direction in the other 4 folds of silenced genes) | **0.5915** (bal. 0.577, MCC 0.162) | **0.5870** (bal. 0.575, MCC 0.157) |
| Within-source majority, other target folds (uses the same knockdown's labels; reference only) | 0.5643 | 0.5671 |

Grouped bootstrap 95% CIs (point estimate, then CI by silenced gene / by source-feature group / by component):

| Statistic | Deployed edges | v2 edges |
|---|---|---|
| Accuracy − best constant | −0.0113; [−0.0161, −0.0072] / [−0.0173, −0.0058] / [−0.0217, −0.0062] | **−0.0455; [−0.0579, −0.0334] / [−0.0584, −0.0315] / [−0.0568, −0.0326]** |
| Balanced accuracy − 0.5 | −0.0017; [−0.0051, +0.0014] / [−0.0054, +0.0016] / [−0.0065, +0.0014] | **−0.0032; [−0.0083, +0.0020] / [−0.0079, +0.0014] / [−0.0067, +0.0060]** |
| MCC | −0.0055; [−0.0160, +0.0047] / [−0.0165, +0.0058] / [−0.0179, +0.0050] | **−0.0063; [−0.0166, +0.0041] / [−0.0157, +0.0028] / [−0.0133, +0.0121]** |
| Accuracy − chance from class rates | −0.0017; [−0.0050, +0.0014] (genes) | −0.0032; [−0.0083, +0.0020] (genes) |
| Accuracy − target-direction baseline | −0.0568; [−0.0639, −0.0504] (genes) | −0.0894; [−0.0998, −0.0799] (genes) |
| Pooled accuracy | 0.5348; [0.5270, 0.5423] (genes) | 0.4976; [0.4919, 0.5037] (genes) |

Fold by fold (5 folds of silenced genes, seed 42), v2: circuit 0.494–0.506; always-decrease 0.537–0.561;
target-direction baseline 0.575–0.592. The circuit is below both in every fold. (Deployed: circuit 0.519–0.546.)

**Per silenced gene** (mean over genes; CI by silenced gene):

| Quantity | Deployed edges (248 genes) | v2 edges (227 genes) |
|---|---:|---:|
| Mean accuracy | 0.5342 [0.5265, 0.5417] | 0.5127 [0.5016, 0.5247] |
| Mean always-decrease accuracy | 0.5457 | 0.5472 |
| Mean accuracy − always decrease | −0.0115 [−0.0163, −0.0074] | −0.0345 [−0.0431, −0.0259] |
| Mean balanced accuracy − 0.5 | −0.0014 [−0.0032, +0.0002] | +0.0011 [−0.0019, +0.0042] |
| Mean MCC | +0.0134 [+0.0068, +0.0197] | +0.0047 [−0.0041, +0.0139] |
| Mean predicted-decrease share (median) | 0.895 (0.967) | 0.626 (0.613) |
| Genes where the circuit beats always-decrease | 139 | 63 |
| Genes with balanced accuracy > 0.5 / MCC > 0 | 146 / 146 | 110 / 110 (14 genes get one sign for every target, so MCC is undefined) |
| Genes where the circuit beats the target-direction baseline | 22 | 29 |

The one small positive in the investigation (mean per-gene MCC +0.013, CI above 0 with the deployed edges) does not
survive with the fixed edges: +0.005, CI [−0.004, +0.014].

**By size of the observed change** (v2 edges; CI by silenced gene):

| \|LFC\| band | Pairs | Obs. decrease | Accuracy | Always-dec | Balanced acc − 0.5 [CI] | MCC [CI] |
|---|---:|---:|---:|---:|---|---|
| < 0.01 | 91,200 | 0.505 | 0.498 | 0.505 | −0.0023 [−0.0055, +0.0011] | −0.0045 [−0.0110, +0.0023] |
| 0.01–0.025 | 128,906 | 0.511 | 0.499 | 0.511 | −0.0013 [−0.0048, +0.0023] | −0.0026 [−0.0096, +0.0046] |
| 0.025–0.05 | 172,980 | 0.531 | 0.495 | 0.531 | −0.0056 [−0.0097, −0.0013] | −0.0112 [−0.0195, −0.0026] |
| 0.05–0.1 | 189,197 | 0.551 | 0.495 | 0.551 | −0.0059 [−0.0122, +0.0006] | −0.0118 [−0.0244, +0.0011] |
| 0.1–0.25 | 106,312 | 0.602 | 0.502 | 0.602 | −0.0002 [−0.0118, +0.0117] | −0.0003 [−0.0234, +0.0229] |
| ≥ 0.25 | 10,029 (145 genes) | 0.739 | 0.531 | 0.739 | +0.0138 [−0.0164, +0.0439] | +0.0243 [−0.0302, +0.0769] |

No band shows skill. The largest-change band is slightly above 0.5 in balanced accuracy, but its CI includes 0 and it
holds 1.4% of pairs. These band CIs were not corrected for looking at 6 bands.

**Sensitivity (v2 edges).**
- Ties predicted as "decrease": accuracy 0.5055, balanced accuracy 0.4957, MCC −0.0087.
- Only edges whose two ends carry an enrichment label (the original Phase 11 rule): 181,210 edges, 632,977 pairs;
  accuracy 0.4968, balanced accuracy 0.4973, MCC −0.0053.
- On the 698,586 pairs shared by both record sets: v2 accuracy 0.4976 (balanced 0.4968); deployed 0.5430
  (balanced 0.5006; it predicts "decrease" for 99.2% of these pairs). The two predictions agree on 50.8% of the shared pairs.

## 7. What changed versus the deployed result

- **Edge map.** 2,144,011 → 272,905 edges. Inhibitory share 88.2% → 58.3%. Next-layer edges 0 → 37,952.
  The deployed edge list was almost entirely the effect of deleting block s (zero-edit control, section 3).
  Any claim built on the old edge list (2.1 M edges, 88% inhibitory, hub and density analyses, PMI, knowledge graph,
  disease mapping, audit A4) is built on the bug.
- **CRISPRi direction.** The verdict does not change: no directional skill. What changes is how it looks. The old circuit
  said "decrease" for 90% of pairs, which pushed raw accuracy to 53.5% (above 50%, below the 54.6% majority rule).
  The fixed circuit says "decrease" for 51% of pairs, and raw accuracy is 49.8%. Balanced accuracy and MCC are at chance
  in both. Both lose clearly to the target-direction baseline (58.7–59.2%).
- The old 54.61% / 53.48% / 53.42% numbers measure class imbalance plus the hook bug. None of them is evidence that
  the circuit predicts regulation direction.

## 8. `outputs/v2_circuit/per_pair.parquet` (v2 edges; 698,624 rows)

| Column | Meaning |
|---|---|
| source | silenced gene (CRISPRi target), upper-case symbol |
| target | gene whose expression change is measured, upper-case symbol |
| evidence | number of (edge, source gene, target gene) triples for this pair |
| n_inhibitory | how many of those came from "inhibitory" edges |
| frac_inhibitory | n_inhibitory / evidence |
| max_abs_d | largest \|Cohen's d\| among them (d as written in the edge CSV, 4 decimals) |
| sum_d | sum of signed d over them |
| predicted_inhibitory | frac_inhibitory > 0.5: the circuit predicts LFC < 0 (tie → False) |
| actual_lfc | observed LFC, log1p-CP10k units (section 5) |
| actual_inhibitory | actual_lfc < 0 |
| correct | predicted_inhibitory == actual_inhibitory |
| src_fold | fold (0–4) of the silenced gene; numpy default_rng(42) permutation of the sorted gene list, split in 5 |
| target_baseline_pred_inhibitory | cross-fitted target-direction prediction (majority observed sign of this target across the silenced genes of the other 4 folds; tie or unseen → training-fold global majority) |
| source_group | source features (with ≥ 1 v2 edge) whose top-10 list holds the silenced gene, as `L{layer}F{feature}` joined by "\|" |
| source_component | connected component of silenced genes that share a source feature (named by one member gene) |

## 9. `outputs/v2_circuit/task_data/` (for a later controlled experiment)

Inputs only: no metrics, baselines, folds or groups. Built by `scripts/v2_circuit_taskdata.py`; file hashes in `MANIFEST.sha256`.
- `records.parquet` — 698,624 rows: silenced_gene, target_gene, evidence, n_inhibitory_evidence, max_abs_d, predicted_decrease, lfc.
- `circuit_edges.csv` — the 272,905 v2 edges (layers, features, d, consistency, sign).
- `source_features.tsv`, `edge_feature_genes.tsv` — feature top-10 gene lists.
- `silenced_genes.tsv` — the 227 silenced genes and their CRISPRi cell counts.
- `README.md` — how the edges, records and LFCs were made, in plain words; `run_config.json`.

## 10. What I could NOT do, and limits

- I did not re-run cells 0–170. An earlier session ran them. I checked all 171 files and recomputed two cells exactly.
- The resume check file (`checks/resume_check.json`) says FAIL. The failure is the legacy zero-edit L0 → L1 row explained
  in section 2, which I repaired and re-checked (`checks/legacy_fix.json`, pass). All other recomputed arrays matched exactly.
- The trace script was edited twice during the run (exact skip of inactive features after cell 4; capture-hook
  warm-up after cell 170). Neither changes the v2 edges; both are documented in the script and in `run_config.json`.
- The earlier session's other v2 scripts had not produced final outputs yet. Before running them I changed:
  `v2_circuit_aggregate.py` (use the repaired legacy control; round d and consistency in the CSV as the deployed run did),
  `v2_circuit_crispri.py` (source-feature groups count only features that have edges; shared-record comparison; CIs per
  \|LFC\| band), `v2_circuit_taskdata.py` (README describes the files without pointing at the analysis traps; the manifest
  skips the macOS `._` metadata files this drive creates). No deployed file was changed.
- I kept the deployed design choices, including ones I would not choose: edits at every position, including
  `<bos>`/`<eos>` (L0 F1883 fires there and has the most edges); no correction for testing about 3.8 M
  (feature, target) pairs; the consistency rule that counts Δz = 0 against "positive" (only 1.2% of edges depend on it).
- No null for edge counts (for example, random source features or random directions with matched norm). So "272,905
  edges" is a count under the deployed rule, not evidence that these edges are meaningful.
- I did not re-run the downstream analyses that read the edge list (PMI, knowledge graph, disease mapping, audit A4).
- CRISPRi: no other sign rule (for example, sign of summed d), no significance filter on LFC, no on-target knockdown
  check, no RPE1 or other datasets. The target-direction baseline is not refit inside bootstrap reps.
- Other non-MPS jobs from other projects ran on the machine at the same time. The exact recomputations (0.0 differences)
  show this did not change results; timings are not clean benchmarks.

## 11. Files

Scripts (`runs/circuit-tracing-217M/scripts/`):
- `v2_circuit_trace.py` — the trace (resumable, `--max-minutes`); `--check-partial` runs the partial-forward check.
- `v2_circuit_resume_check.py` — checks finished cells before resuming.
- `v2_circuit_legacy_fix.py` — repairs the legacy zero-edit control in 31 cells (new files only).
- `v2_circuit_aggregate.py` — per-cell vectors → edges, per-layer-pair counts, controls.
- `v2_circuit_spotcheck.py` — recomputes 3 edges from scratch.
- `v2_circuit_lfc.py` — CRISPRi LFCs from the h5ad.
- `v2_circuit_crispri.py` — records, metrics, baselines, grouped bootstraps (both edge sets).
- `v2_circuit_taskdata.py` — the task_data package.
- `v2_circuit_finalize.py` — `run_config.json` and `compare_summary.json`.

Outputs (`runs/circuit-tracing-217M/outputs/v2_circuit/`):
- `circuit_edges_v2.csv` — v2 edges (columns: src_layer, src_feature, tgt_layer, tgt_feature, cohens_d, consistency,
  sign, mean_delta, n_pos, n_neg, n_zero, consistency_sym, passes_symmetric_rule).
- `edges_per_layer_pair.csv`, `edges_per_source_feature.csv`, `edge_stats_main.npz`, `aggregate_summary.json`, `compare_summary.json`.
- `control_legacy_zero_edges.csv`, `control_legacy_real_edges.csv` — deployed-hook controls.
- `trace_cells/` (200 per-cell files), `trace_cells_legacy_fix/` (31 files), `cells_tokens.npz`, `source_features.json`,
  `combos_main.csv`, `combos_ctrl.csv`, `trace_progress.json`.
- `checks/` — `partial_forward_check.json`, `resume_check.json`, `legacy_fix.json`.
- `spotcheck/spotcheck_result.json`.
- `lfc/` — LFC matrix, sources, control rows, `check_vs_deployed_records.json`, `run_config.json`.
- `crispri/` — `results.json`, `comparison_table.csv`, `per_source_metrics_v2_edges.csv`,
  `per_source_metrics_deployed_edges.csv`, `run_config.json`.
- `per_pair.parquet`, `task_data/`, `run_config.json` (device, versions, seeds, the 200 dataset rows and token counts,
  the 120 source features, wall time per chunk, script hashes).

## Plain-words summary

The old circuit-tracing code put each edit one step too late in the model, so every "ablation" also deleted a whole
layer. I re-ran the tracing with the fixed tools on the same 200 cells, the same 120 features and the same edge rule.
The new run finds 272,905 edges instead of 2.14 million, and 58% of them are "inhibitory" instead of 88%. The old edge
list was almost entirely the bug: the old code with an edit of exactly zero reproduces more than 99% of it. The new
edges pass every check: edges now appear at the very next layer, an empty edit gives no edges at all, and three edges
recomputed from scratch with separate code match to within 0.00005 in d.

I then rebuilt the CRISPRi test from the new edges, exactly as before. The circuit now says "goes down" for about half
of the gene pairs. It is right 49.8% of the time. Always saying "goes down" is right 54.3% of the time, and a simple rule
that only looks at each target gene's usual direction is right 58.7% of the time. Balanced accuracy is 49.7% and MCC is
−0.006, both at chance. So the circuit has no detectable skill at predicting which way a gene moves after silencing.
The old 53.5% only looked above 50% because the old circuit almost always said "goes down". The conclusion is the same
as before the fix, but now it rests on correct edges.
