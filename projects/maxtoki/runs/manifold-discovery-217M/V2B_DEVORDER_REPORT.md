# Developmental ordering beyond cell-type identity (item D8b)

Date: 2026-10-01. CPU only. No model forward pass. All numbers come from the new `v2b_devorder_*` scripts run on
saved run files. Short names:

- RUN = `projects/maxtoki/runs/manifold-discovery-217M`, OUT = `RUN/outputs/v2b_devorder`
- "anchor" = the mean of the cells that share donor × tissue × cell type × stage label.
- "gates" = the run's pass marks: trustworthiness ≥ 0.80 and three Spearman correlations ≥ 0.20
  (random holdout, donor holdout, branch holdout). "Trust" below means trustworthiness (k = 15 anchors).
- "class" = a (cell_type, stage) pair. The stage comes from `free_annotation`, then `cell_type`, so one cell type
  can carry several stages (for example "CD8-positive T cell" carries 4).

## The question and the short answer

Question: does MaxToki-217M's internal state carry a blood developmental ordering (the H65 stage tree) beyond what
any representation that separates cell types would show?

Short answer: **no, not that these tests can detect.**

- A random code per cell-type label with no biology in it (the "lookup") scores the same as MaxToki on the key
  branch number: internal branch holdout 0.392 (mean of 5 seeds) vs MaxToki 0.370.
- Model-free expression features score the same too: token bag 0.417, HVG PCA 0.354.
- Under a null that keeps cell-type structure (stages re-assigned to cell-type classes within each branch),
  MaxToki's real H65 scores sit inside the null on every panel. Internal branch holdout: 0.370 vs null mean 0.381,
  p = 0.52.
- Counting only pairs of anchors with different cell types, the branch numbers fall to about 0.05:
  0.047 internal, 0.050 external, 0.094 zero-shot.
- The one place MaxToki beats the lookup and HVG is the cross-branch (global) order. There it ties the
  token bag built from its own input tokens on the external panel (+0.007, interval −0.011 to +0.033). On the
  zero-shot panel it is ahead of the token bag by a small margin (+0.020, interval +0.001 to +0.042).

## 1. What was done

| Step | Script (`RUN/scripts/`) | Output (`OUT/`) |
|---|---|---|
| Build representations | `v2b_devorder_01_features.py` | `features/`, `features_manifest.json` |
| Every head fit as a task pool (1,958 + 120 fits) | `v2b_devorder_02_pool.py` | `pool/results.jsonl`, `heads/` |
| Thread-count check (H38, H95) | `v2b_devorder_02b_threads.py` | `thread_check/results.jsonl` |
| Tables, nulls, lung control, bootstraps, random projections | `v2b_devorder_03_analyze.py` | `v2b_*.json`, `table_*.csv` |
| Provenance (sha256 of 48 inputs, 178 outputs, 5 scripts; all seeds) | `v2b_devorder_03_analyze.py config` | `run_config.json` (+ `run_config_v2b_devorder_01_features_*.json`) |

Gate code: `train_let`, `random_holdout_corr` and `grouped_holdout_corr` are imported read-only from
`phase5_let_anchor.py`. The frozen-panel gate code of `phase7_external_validation.py`, `phase8_zeroshot_transfer.py`
and `phase15_validate_candidate.py` sits inside their `main()`, so it is copied line for line. H38 uses
`phase13_h38_lite` functions, H95 uses `phase3a_manifold_sweep`, and H103 uses `phase3a_sweep2_ordinal`.
Each task is one head fit with torch at 1 thread. Held-out groups whose ruler is constant are skipped before fitting.
The run's code drops them anyway (NaN), so the numbers are the same.

Representations. Each has one row per anchor and is standardised with the internal panel's mean and SD, so a head
fitted on internal can be applied unchanged to the other panels.

| Name | What it is |
|---|---|
| `maxtoki` | the deployed 2,464-number pooled-drift feature from MaxToki hidden-state centroids |
| `maxtoki_pca64` | PCA-64 of `maxtoki` (97% of its variance) — same model, same size as the baselines |
| `lookup_ct_s0..s4` | one random 64-number Gaussian code per cell_type string, + 0.5 × noise per anchor; 5 seeds |
| `lookup_cls_s0..s2` | the same per (cell_type, stage) class; 3 seeds |
| `tokenbag_pca64` | model-free. In each cell, every gene token at rank r of n gets weight 1 − r/n (the tokenizer's own order). Anchor mean, log1p, then PCA-64 fitted on internal anchors. |
| `hvg_pca64` | model-free. Raw counts (`raw/X` of the Tabula Sapiens h5ad), counts per 10k + log1p, anchor mean, 2,000 HVGs (Seurat-v1 dispersion, internal cells only), then PCA-64. |

The token bag and HVG features use exactly the cells that went into the MaxToki centroids. External and zero-shot
re-draw the `phase1bc` subsample with `default_rng(42)`, and the per-anchor cell counts were checked against
`n_cells_centroided`.

## 2. Reproduction of the run's numbers (all at 1 torch thread)

- **H65, exact.** Internal trust 0.810936, random 0.834108, donor 0.715888, branch 0.370404.
  External frozen: 0.895688 / 0.885775 / 0.886953 / 0.346443, global 0.882120.
  Zero-shot: 0.826724 / branch 0.316830 / donor 0.870309. Lung: 0.799556 / −0.016420 / 0.013397 / −0.146871.
  The run's within-branch null gives branch −0.003 at 1 thread (−0.001 deployed at 4 threads, as already known).
- **H103, exact:** 0.8601 / 0.8610 / 0.8607.
- **L10H6 head screen:** trust 0.88179 and random 0.79326, both exact. Branch is 0.4665 here vs 0.4674 deployed.
- **H38 and H95** match exactly only at 4 or 6 threads. Checked on the H95 full fit and two H38 held-out groups
  (`thread_check/`). At 1 thread:
  - H38 category holdout is 0.2877 vs 0.2932 deployed.
  - **H95 trust is 0.7994 vs 0.8005 deployed.** So H95 passes the 0.80 trust gate or not depending on the thread count.
- All comparisons below use 1 thread for every representation, so they are like for like.

## 3. H65 on every panel, by representation

Gate values. "diff" = the same number using only pairs of anchors whose cell_type differs. Lookup = mean of 5 seeds
(range in brackets). The within-branch "diff" number is a mean over 3 branches (T, monocyte, granulocyte; 2 on
zero-shot), not 6. In the B, erythroid and stem branches every different-cell-type pair has ruler value 1, so the
Spearman is undefined there and those branches drop out.

| Panel | Number | MaxToki | Lookup (5 seeds) | Token bag | HVG PCA | MaxToki PCA-64 |
|---|---|---:|---:|---:|---:|---:|
| internal (head re-fitted) | trust | 0.811 | 0.776 (0.754–0.790) | 0.708 | 0.717 | 0.696 |
| | random / donor | 0.834 / 0.716 | 0.743 / 0.709 | 0.652 / 0.563 | 0.726 / 0.676 | 0.782 / 0.670 |
| | branch holdout | 0.370 | 0.392 (0.319–0.459) | 0.417 | 0.354 | 0.490 |
| | branch holdout, diff | 0.047 | 0.054 (−0.033–0.146) | 0.006 | 0.088 | 0.065 |
| external (frozen) | trust | 0.896 | 0.833 (0.781–0.866) | 0.875 | 0.859 | 0.843 |
| | within-branch | 0.346 | 0.342 (0.303–0.445) | 0.422 | 0.367 | 0.400 |
| | within-branch, diff | 0.050 | 0.026 (−0.075–0.213) | 0.009 | −0.011 | −0.037 |
| | global / global diff | 0.882 / 0.856 | 0.801 / 0.749 | 0.876 / 0.850 | 0.763 / 0.720 | 0.879 / 0.853 |
| zero-shot (frozen) | trust | 0.827 | 0.735 (0.710–0.770) | 0.759 | 0.768 | 0.749 |
| | within-branch / diff | 0.317 / 0.094 | 0.377 / 0.226 | 0.270 / 0.028 | 0.030 / −0.060 | 0.475 / 0.172 |
| | global / global diff | 0.899 / 0.886 | 0.776 / 0.742 | 0.879 / 0.866 | 0.757 / 0.746 | 0.876 / 0.861 |

Which gates each one passes:

- **Lookup, internal:** passes the three correlation gates in all 5 seeds. It fails only trust (0.754–0.790).
- **Lookup, external:** passes all four gates in 4 of 5 seeds.
- **Token bag and HVG:** pass all four gates on external. Internally they fail only trust (0.708, 0.717).
- **Trust depends on how the input is prepared, not on what the model knows.** The same MaxToki features cut to
  64 PCs drop to trust 0.696 internal. Most of that drop comes from whitening (each PC scaled to SD 1), not from
  the smaller size: without whitening, MaxToki PCA-64 gives 0.786 and the token bag PCA-64 gives 0.796
  (independent check, see Verification notes). All 64-number baselines here are whitened, so their lower trust is
  partly a preprocessing effect. (Trust compares the head's output with the head's own input. It never looks at
  the ruler.)

Full table: `OUT/table_gates_by_representation.csv`, with 112 rows (H65, H38, H95, H103; internal, external,
zero-shot, lung).

## 4. Why the branch number cannot tell order from identity

Inside a branch, the H65 ruler gives graded distances only in the T lineage. From `v2b_tables.json`
(`ruler_structure`):

| Branch | Stages present | Ruler values between different classes |
|---|---|---|
| B lineage | Naive_B, Memory_B, Plasma | 1 only (all three stages are joined to each other in the graph) |
| erythroid | MEP, Erythrocyte | 1 only |
| stem | HSC, MPP | 1 only |
| granulocyte | Basophil, Mast, Neutrophil | 0 or 2 (same stage or not) |
| monocyte | Cl_Mono, NC_Mono | 0 or 1 |
| T lineage | 6–7 stages | 0, 1, 2, 3, 4 |

So in the B, erythroid and stem branches, the branch Spearman can only measure whether anchors with the same label
sit together. These small branches give the large internal values: erythroid 0.783, granulocyte 0.708 and stem 0.621.
The three large branches give T 0.031, B 0.154 and monocyte −0.075.

The T lineage is the only real test of order. There:

- **Held out internally:** MaxToki 0.031 (diff pairs −0.068). The lookup gets 0.156 (diff 0.010), the token bag
  0.034, and HVG 0.087.
- **Frozen external:** MaxToki 0.289 (diff 0.319) vs lookup 0.354 (diff 0.329). The frozen head learned the T
  labels on internal anchors, and a lookup reproduces that learned order as well as MaxToki does.

Per-branch rows: `OUT/table_h65_per_branch.csv`.

## 5. Structured null: re-assign stages to cell-type classes within each branch

Method. A class is a (cell_type, stage) pair, taken over all blood panels. Each draw permutes the class → stage map
among the classes of one branch. Every anchor keeps its branch, and anchors of one class keep one shared stage.
Same-class pairs therefore stay at distance 0, exactly as in the real ruler.

There are two versions:

- **Refit:** the whole pipeline is re-run on the permuted ruler. The internal head is re-fitted, then applied frozen
  to external and zero-shot under the same map. 40 draws (`default_rng([2026, d])`); internal branch holdout
  re-fitted for the first 20.
- **Eval-only:** the held-out heads from the real run are kept, and only the held-out ruler is re-drawn.
  2,000 draws (`default_rng([2027, d])`).

p = (1 + number of null values ≥ observed) / (1 + number of draws). One-sided.

| MaxToki number | Observed | Null mean (SD) | Null max | p | Null passes 0.20 gate |
|---|---:|---:|---:|---:|---:|
| internal branch holdout (refit, 20 draws) | 0.370 | 0.381 (0.060) | 0.547 | 0.52 | 100% |
| internal branch holdout (eval-only, 2,000) | 0.370 | 0.370 | — | 0.55 | — |
| internal branch, diff (eval-only) | 0.047 | 0.010 | — | 0.38 | — |
| internal trust (refit) | 0.811 | 0.808 (0.010) | 0.825 | 0.37 | 80% pass 0.80 |
| external within-branch | 0.346 | 0.256 (0.085) | 0.409 | 0.20 | 72% |
| external within-branch, diff | 0.050 | 0.121 (0.105) | 0.359 | 0.76 | 20% |
| external global | 0.882 | 0.863 (0.026) | 0.910 | 0.22 | 100% |
| zero-shot within-branch | 0.317 | 0.193 (0.145) | 0.524 | 0.24 | 47% |
| zero-shot global | 0.899 | 0.860 (0.025) | 0.901 | 0.049 | 100% |

The lookup, token bag and HVG behave the same way. None is above its own null on any branch number (p 0.15–0.85).
The smallest p for MaxToki is 0.049 (zero-shot global, 1 of 40 null draws above). That is not small enough to stand
alone. Its external counterpart gives p = 0.22.

## 6. Lung negative control, redone with structure

The deployed design gives each of the 50 non-blood lung anchors its own random stage. Redrawn 2,000 times, no
representation ever passes random, donor and branch together. That holds for all 12 representations, MaxToki and
pure label codes alike. So this check fails for any input.

Structured version: one random stage per lung **cell type** (16 cell types), 2,000 draws (`default_rng([6262, d])`),
frozen H65 head.

| Representation | Branch mean | Pass branch | Pass random | Pass donor | Pass random + donor + branch | Branch ≥ 0.346 (external H65) |
|---|---:|---:|---:|---:|---:|---:|
| MaxToki | 0.527 | 96.5% | 15.0% | 3.2% | 3.1% | 86% |
| Lookup (seed 0) | 0.508 | 96.5% | 14.6% | 3.1% | 2.7% | 85% |
| Token bag | 0.570 | 97.4% | 20.3% | 5.0% | 4.9% | 91% |
| HVG PCA | 0.430 | 89.9% | 12.1% | 4.8% | 4.2% | 71% |

MaxToki's lung trust is fixed at 0.7996 (it does not depend on labels), so "all four" passes 0%. For the token bag it
passes 4.9%, because its lung trust is 0.928. **The branch gate passes for meaningless labels in 96% of draws** once
the labels follow cell types. The donor gate is the one that still fails most of the time.

## 7. Donor-level intervals for the key contrasts

Method: donor cluster bootstrap. Each replicate draws the panel's donors with replacement and keeps all their
anchors. Copies of one anchor are never paired. Heads stay frozen. 2,000 replicates, percentile 2.5–97.5%.
The lookup is the mean of 5 seeds within each replicate. Global numbers use pair weights, which give exactly the
same result as expanding the resample (checked on zero-shot: difference 2e-16). Unit: donor (13 external,
12 zero-shot, 11 internal).

| Contrast | Panel | Within-branch | Within-branch, diff | Global | Global, diff |
|---|---|---|---|---|---|
| MaxToki − lookup | external | +0.005 [−0.203, 0.158] | +0.023 [−0.086, 0.118] | **+0.081 [0.062, 0.104]** | **+0.106 [0.082, 0.137]** |
| MaxToki − token bag | external | −0.075 [−0.293, 0.062] | +0.040 [−0.018, 0.330] | +0.007 [−0.011, 0.033] | +0.006 [−0.015, 0.035] |
| MaxToki − HVG | external | −0.021 [−0.240, 0.115] | +0.061 [−0.045, 0.255] | **+0.119 [0.084, 0.160]** | **+0.136 [0.101, 0.177]** |
| MaxToki − lookup | zero-shot | −0.060 [−0.365, 0.088] | −0.132 [−0.236, 0.123] | **+0.123 [0.050, 0.188]** | **+0.144 [0.068, 0.210]** |
| MaxToki − token bag | zero-shot | +0.047 [−0.346, 0.373] | +0.066 [−0.146, 0.259] | +0.020 [0.001, 0.042] | +0.020 [−0.001, 0.046] |
| MaxToki − HVG | zero-shot | +0.287 [−0.044, 0.510] | +0.155 [−0.073, 0.360] | **+0.142 [0.065, 0.184]** | **+0.140 [0.065, 0.187]** |
| MaxToki − lookup | internal branch holdout | −0.022 [−0.213, 0.045] | −0.007 [−0.446, 0.053] | — | — |
| MaxToki − token bag | internal | −0.046 [−0.279, 0.020] | +0.041 [−0.145, 0.108] | — | — |
| MaxToki − HVG | internal | +0.017 [−0.236, 0.028] | −0.041 [−0.121, 0.032] | — | — |

Internal rows: the held-out heads are fixed and only the evaluation donors are resampled. These intervals leave out
re-fitting noise, so they are too narrow. Some internal replicates lose the TSP2-only branches (stem, erythroid), so
their plain means average fewer branches (the same issue as V3 in the D8 report).

What it means:

- On every within-branch number, MaxToki is not separable from the lookup or from either expression baseline.
- On the global order, MaxToki beats the lookup and HVG by about 0.08–0.14, with intervals clear of 0.
- But it ties the token bag, which is built from the same rank-ordered tokens the model reads.
- Global order is mostly cross-branch: lymphoid vs myeloid, which branch is near which. The structured null does
  not move it much, because branches are kept.

## 8. H38, H95, H103

| Ordering | Finding |
|---|---|
| H38 LITE (7 signalling categories) | Every lookup seed passes all four internal gates, with higher values than MaxToki: category holdout 0.42 / 0.53 / 0.38 vs 0.29. External category: lookup 0.75–0.83, token bag 0.738, HVG 0.640, MaxToki 0.594. Category holdout on diff pairs: MaxToki −0.020. Structured null (cell_type → category set permuted among cell types, 30 draws): MaxToki's real map beats all 30 draws on most external numbers (p = 0.032). Token bag and HVG beat theirs just as clearly (p = 0.032). So any expression-based representation follows these categories, and MaxToki is not special. |
| H95 (4 effector categories) | The category gate is undefined internally, so the 3-gate fallback applies. Lookups pass it (trust 0.816–0.861). MaxToki's trust is 0.7994 at 1 thread and 0.8005 at 4–6 threads, so its pass sits inside optimiser noise. Frozen within-category: MaxToki 0.155 external, which is below its own structured null (mean 0.389, p = 0.91). |
| H103 (B-cell maturation) | Only two labels exist ("B cell" at depth 1, "plasma cell" at depth 5), so the ruler is 0 for the same label and 4 otherwise. Swapping the two labels gives the same ruler, so the test can only ask "do the two labels separate?". Every representation that separates them gets the same random and donor numbers: internal 0.861 / 0.861 for MaxToki and all lookups; external 0.858 / 0.859 for every representation. Trust: lookups 0.885–0.908 internal vs MaxToki 0.860. |

## 9. The single-head claim (L10H6) vs random projections of the same residual

The L10H6 feature is the post-final-norm centroid (index 11) times the 154 W_O columns of head 6. As controls,
Gaussian 154 × 1,232 projections were applied to the same centroids: 30 draws at index 11 and 10 at index 1. Scoring
uses phase9's own rule: in-sample trust, 3 random splits, 6-branch holdout, and
composite = 0.5 trust + 0.25 random + 0.25 branch.

| Number | L10H6 | Random, index 11 (30): mean ± SD (range) | Share of random ≥ L10H6 | P(best of 88 random ≥ L10H6) |
|---|---:|---|---:|---|
| trust | 0.882 | 0.816 ± 0.011 (0.795–0.839) | 0/30 | 4e-7 (normal approximation; 0/30 seen) |
| random holdout | 0.793 | 0.790 ± 0.019 | 40% | ≈ 1 |
| branch holdout | 0.467 | 0.418 ± 0.063 (0.287–0.507) | 23% | ≈ 1 (normal: 0.99999) |
| branch holdout, diff pairs | −0.014 | 0.047 ± 0.054 | 83% | ≈ 1 |
| composite | 0.756 | 0.710 ± 0.018 (0.672–0.737) | 0/30 | 0.43 (normal approximation) |

- Random projections of the layer-0 output (index 1) give composite 0.707 and branch 0.387.
- Random projections beat the average real head (composite 0.654) and 6 of the 8 real layer-10 heads (the mean
  random composite 0.710 is below L10H6 0.756 and L10H7 0.719).
- **L10H6's only edge is trust.** Its 154 features are a little more concentrated: participation ratio 10.9 vs
  12.1 (range 11.0–13.1), and the top 10 directions hold 80% of the variance vs 74%. A more concentrated input is
  easier for a 10-number head to keep neighbours in. That says nothing about biology.
- On the branch gate, the one the summary calls key, a random projection of the same residual does as well.
  Allowing for best-of-88 selection, L10H6 is not special on that gate. Best-of-88 is a generous allowance for the
  composite, because the other 80 heads read other layers and score lower (mean 0.654). Against only the 8 heads of
  the same layer, a random projection would reach L10H6's composite with probability about 0.04 (normal
  approximation). That lead still comes from trust, which does not use the ruler.
- Not tested: the probability for trust was not checked empirically. With 30 draws, best-of-88 cannot be bounded
  from the data alone, so the 4e-7 rests on a normal approximation.

## 10. What changes relative to the deployed claims

1. "H65 passes all four gates and the null fails" is still true for the run's within-branch null. But it does not
   show developmental order:
   - a label lookup passes 3 of 4 gates internally and all 4 externally (4 of 5 seeds);
   - token-bag and HVG expression features pass all 4 externally;
   - a null that keeps cell-type structure scores like H65 (internal branch p = 0.52; external p = 0.20; zero-shot
     p = 0.24).
2. "Branch holdout separates a real developmental ordering from a null decisively" is not supported:
   - on pairs with different cell types the branch number is 0.047 / 0.050 / 0.094;
   - 3 of the 6 scored branches have no order in the ruler beyond "same label or not";
   - in the T lineage, the only graded branch, held-out order is 0.031.
3. The lung negative control fails for any input under the deployed design (0 of 2,000 redraws pass, for every
   representation). With one random stage per lung cell type, MaxToki passes the branch gate in 96.5% of draws.
4. "Multi-axis biological hub" (H38, H95, H103):
   - lookups match or beat MaxToki on all three;
   - H103 tests only whether two labels separate;
   - H95's trust pass depends on the torch thread count;
   - H38's structure is followed equally by model-free expression features on the external panel. Internally,
     MaxToki is ahead of them on random and donor holdout (0.761 / 0.673 vs token bag 0.563 / 0.423 and HVG
     0.509 / 0.394), but not ahead of the lookups (0.752–0.848 / 0.718–0.768).
5. L10H6: random projections of the same residual match its branch holdout. Its lead in trust and composite comes
   from trust, which does not use the ruler.
6. What MaxToki does show: its anchors are arranged by lineage across branches (global Spearman 0.882 external,
   0.899 zero-shot). That is better than a label lookup (+0.08 to +0.12) and HVG expression (+0.12 to +0.14).
   It ties a model-free bag of its own input tokens on external (+0.007 [−0.011, 0.033]) and is slightly ahead
   on zero-shot (+0.020 [0.001, 0.042]).

**What is specific to MaxToki:** nothing about developmental order that these tests can detect. The only advantage
over a label lookup and over standard HVG expression is cross-lineage arrangement. The rank-ordered gene-token input
already carries almost all of that, with no model (MaxToki's lead over it is 0.007 to 0.02).

## 11. What I could not do, and limits

- **No random-initialised MaxToki.** It needs forward passes, which were not allowed.
- **No scVI, Palantir or CellTypist comparison.**
- **No new anchors** and no real non-blood lung differentiation ruler.
- **Few refit-null draws.** Refit nulls use 40 draws (branch holdout 20) for H65 and 30 for H38/H95. The smallest
  reachable p is 0.024 / 0.048 / 0.032. These are coarse p-values. They cannot show strong evidence for MaxToki,
  but they do show its scores sit inside the null.
- **Internal intervals are too narrow.** They resample evaluation donors only (held-out heads fixed), with no
  re-fitting.
- **H38 and H95 values move with thread count.** The comparisons use 1 thread for all representations. Deployed
  MaxToki values for H38 and H95 match only at 4 or 6 threads, and differ by up to 0.016 in one held-out group.
- **The lookup noise level (0.5) and code size (64) were fixed, not tuned.** Lookup trust changes with these choices.
  Trust is not a biology measure in any case.
- **The class-level null re-draws which stage each class gets.** When a stage moves to a class that is absent from a
  panel, that panel's branch can become constant and is then not scored. The number of scored branches therefore
  varies a little between draws.
- **Raw h5ad files were not hashed** (19.8 GB and 3.2 GB). Their sizes are recorded in `run_config.json`.

## Summary in plain words

I ran the run's own gate code on MaxToki and on stand-ins that know nothing about development: a random code per
cell-type label, and two expression features built without the model. All of them pass the run's gates about as well
as MaxToki. A shuffled ruler that still keeps cell types together scores about as well as the real stage ruler. When
only pairs of different cell types are counted, the branch number drops to about 0.05.

The reason is that the "within-branch" ruler mostly asks whether cells with the same label sit together. Only the
T-cell branch has a real order inside it, and there MaxToki scores 0.03 on held-out T cells. The lung negative
control could not fail anything useful in its deployed form. When its random labels follow lung cell types, it
passes the branch gate 96% of the time.

The single head L10H6 is no better than random projections of the same residual on the branch gate.

What MaxToki does show is that its anchors are arranged by lineage across branches. A plain bag of its own input
tokens does this just as well.

## Verification notes (independent check, 2026-10-01)

Verdict: **OK after small fixes.** The main answer holds: these tests find no developmental ordering in MaxToki
beyond what a label lookup or model-free expression features show. A few numbers and sentences were corrected.

How it was checked. A separate script, `RUN/scripts/v2b_devorder_verify.py`, that does not import any
`v2b_devorder_*` code. Outputs are in `OUT/verify/` (`verify_results.json`, `fits.jsonl`, `boot_*.jsonl`,
`run_config.json` with sha256 of 50 inputs, the script and 30 outputs, plus all seeds). It rebuilds the MaxToki
feature with the run's `build_pooled_drift`, makes its own lookups with 3 new seeds (501–503), its own token bag, and
its own HVG features (scanpy `seurat` flavour). It fits heads with the run's `train_let` at 1 torch thread, and
computes every gate with its own code. CPU only, no forward pass.

What matched:

1. **Reproduction.** MaxToki H65: internal trust 0.8109, branch 0.3704, branch diff 0.0472; external
   0.8957 / 0.8858 / 0.8870 / 0.3464, global 0.8821; zero-shot trust 0.8267, branch 0.3168, global 0.8990; lung
   0.7996 / −0.0164 / 0.0134 / −0.1469. L10H6: trust 0.88179, random 0.79326, branch 0.4665. All exact.
2. **Features.** Own token bag vs the repair's: anchor-distance correlation 0.995–0.999 per panel. Own HVG: the same
   2,000 genes, distance correlation 0.997–0.999.
3. **Baselines.** Internal branch holdout: token bag 0.432 (repair 0.417), HVG 0.381 (0.354). External within-branch:
   0.415 (0.422) and 0.370 (0.367).
4. **Ruler structure (§4).** Confirmed on all three blood panels.
5. **Structured null, eval-only** (2,000 new draws, `default_rng([31337, d])`). MaxToki internal branch 0.370 vs
   null mean 0.369, p = 0.55 (repair 0.55). Diff pairs: p = 0.35 (repair 0.38). T lineage alone: 0.031 vs null mean
   0.105, p = 0.91. So the real T order fits the held-out geometry worse than a random re-assignment does.
   Lookups, token bag and HVG: p 0.08–0.99.
6. **Lung, deployed design** (one random stage per anchor, 2,000 new draws). Random + donor + branch together pass
   in 0 of 2,000 draws for MaxToki, the 3 new lookups and the token bag, and 1 of 2,000 for HVG. Branch alone passes
   in 10–11% of draws. The observed −0.147 is reached or beaten in 17–21% of draws, so it is ordinary noise.
7. **Lung, structured** (one random stage per lung cell type, 2,000 new draws). MaxToki passes branch in 95.7% of
   draws (repair 96.5%), mean 0.518; random 16.6%, donor 3.1%, all three 3.0%. Lookups 94–97%, token bag 96.5%,
   HVG 89.6%.
8. **Donor bootstrap.** Own code expands each resample explicitly and drops pairs made of two copies of one anchor.
   Unit = donor (13 external, 12 zero-shot). 1,000 replicates, percentile 95% interval. Lookup = mean of the 3 new
   seeds.

   | Contrast (global Spearman) | External | Zero-shot |
   |---|---|---|
   | MaxToki − lookup | +0.085 [0.070, 0.109] | +0.133 [0.074, 0.181] |
   | MaxToki − token bag | +0.012 [−0.007, 0.038] | +0.026 [0.003, 0.051] |
   | MaxToki − HVG | +0.110 [0.068, 0.157] | +0.132 [0.057, 0.175] |

   All six within-branch contrasts include 0, as in §7. The widest call is zero-shot MaxToki − HVG,
   +0.324 [−0.018, 0.529].
9. **N60.** 6 new random projections of the index-11 residual (composite 0.702–0.732). Pooled with the repair's 30
   (36 draws): composite mean 0.710 (SD 0.018), and none reaches L10H6's 0.756. Trust: none reaches 0.882 (max 0.839).
   Branch: 25% of draws reach 0.467. Random holdout: 42% reach 0.793. Normal approximation for the composite:
   P(best of 88 ≥ L10H6) = 0.37 (repair 0.43 with 30 draws); P(best of 8) = 0.04.

What differed, and was fixed in this report:

- **Lookup seeds vary a lot.** The 3 new seeds give internal branch holdout 0.298, 0.338 and 0.197 (repair's 5 seeds:
  0.319–0.459). One new seed fails the 0.20 gate by 0.003. So the lookup is about as good as MaxToki (0.370), not
  better. Across all 8 seeds the range is 0.197–0.459. External: all 3 new seeds pass all four gates.
- **Trust is mostly a whitening effect, not input size** (§3 fixed). MaxToki PCA-64 whitened 0.699 (repair 0.696);
  not whitened 0.786; PCA-256 not whitened 0.768. Token bag PCA-64 not whitened 0.796 (whitened 0.701).
- **"diff" averages fewer branches** (§3 now says so): 3 of 6 internally and externally, 2 on zero-shot.
- **Token bag tie holds only on external.** On zero-shot MaxToki is ahead by about 0.02, and the interval just
  clears 0 in both codes (Short answer, §10.6 and the closing statement fixed).
- **§9:** random projections beat 6 of the 8 layer-10 heads, not 7. Best-of-88 is a generous allowance for the
  composite. Against the 8 heads of the same layer it is about 0.04. The lead is still a trust lead.
- **§10.4:** on H38, MaxToki beats the two expression baselines internally on random and donor holdout. It does not
  beat the lookups. The sentence now says the tie with expression features is external only.

Also noted:

- The repair's closing summary says every representation gets 0.861 / 0.861 on H103. That is true only for
  representations that fully separate the two labels. Internally the token bag gets 0.576 / 0.072 and HVG
  0.731 / 0.814 (from `OUT/table_gates_by_representation.csv`). §8 of this report states it correctly.

Not re-derived:

- The refit nulls (H65 internal p = 0.52, external p = 0.20, zero-shot p = 0.24; the H38 and H95 30-draw nulls).
- The H38, H95 and H103 fits, and the thread-count effect. For these I relied on the repair's table. The earlier
  independent N58 check found the same pattern with its own lookups (H38 passes all four, H95 and H103 pass the
  3-gate fallback).
- The internal donor bootstrap.
- An all-gene (19,601-number) token-bag fit for trust. It did not finish inside the 9-minute limit.

Plain summary: the repair's numbers reproduce. Its conclusion holds. MaxToki shows no sign of blood developmental
order beyond cell-type identity in these tests. Six statements were too strong or slightly wrong and are now fixed.
The biggest fix: the trust gap between MaxToki and the 64-number baselines comes mostly from whitening, not from the
model. The one thing MaxToki does better than a lookup and HVG is cross-lineage order. A bag of its own input tokens
matches that on external and trails by about 0.02 on zero-shot.
