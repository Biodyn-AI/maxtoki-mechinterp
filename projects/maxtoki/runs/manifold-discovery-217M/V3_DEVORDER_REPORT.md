# Developmental ordering with the correct MaxToki input (item V3-7)

Date: 2026-10-03. Short names:

- RUN = `projects/maxtoki/runs/manifold-discovery-217M`; OUT = `RUN/outputs/v3_devorder`.
- "deployed" = the original run. "v2b" = `RUN/V2B_DEVORDER_REPORT.md`, which reproduced the deployed MaxToki numbers
  exactly. Both fed MaxToki the wrong gene order (genes ranked by log1p(CP10k) / gene median; see
  `checks/INPUT_ENCODING_AUDIT.md`). "v3" = this re-run with the correct order.
- "anchor" = the mean hidden state of the cells that share donor × tissue × cell type × stage label.
- "gates" = the run's pass marks: trustworthiness ("trust", k = 15 anchors) ≥ 0.80, and random, donor and branch
  Spearman correlations ≥ 0.20. "diff" = the same number using only pairs of anchors whose cell types differ.
- "seed" = the torch seed of the small read-out head (the LET head). The run and v2b always used seed 42.

## 0. Results first

1. **The model now sees the right input.** I re-ran MaxToki-217M on all 32,548 cells of the five panels (internal,
   external, zero-shot, non-blood lung control, first lung control), from Tabula Sapiens `raw/X` counts with
   `setup/inputs_v3.py`. Every cell passed the encoding check before its forward pass (32,548 of 32,548; 0 genes out
   of order). Same cells, anchors, layers and pooling as deployed.
2. **The correct input fits the model better.** On 41 random 10x cells, next-gene loss is 3.09 nats with the correct
   order and 4.03 with the deployed order (difference −0.95, 95% interval over cells [−1.18, −0.73]; lower in 41 of
   41 cells). A uniform guess scores 9.92.
3. **The main answer does not change.** MaxToki shows no blood developmental order beyond cell-type identity.
   - A null that keeps cell-type structure still scores like the real stage ruler: internal branch holdout 0.414 vs
     null mean 0.402 (p = 0.39, 2,000 draws); external within-branch p = 0.15; zero-shot p = 0.27.
   - On pairs of anchors with different cell types, the branch numbers stay near 0: 0.067 internal, 0.006 external,
     0.016 zero-shot.
   - In the T lineage, the only branch with a graded ruler, held-out order is 0.038 (v2b 0.031).
4. **Two numbers got worse.**
   - Zero-shot within-branch is now **0.180**, below the 0.20 gate (v2b 0.317). Over 20 head seeds it is 0.147 ± 0.048
     (mean ± SD); only 3 of 20 seeds pass. With v2b inputs all 10 seeds passed (0.355 ± 0.057). So with the correct
     input **H65 no longer passes all four gates on the zero-shot panel**.
   - Cross-branch ("global") order fell by about 0.03: external 0.882 → 0.865 and zero-shot 0.899 → 0.867 (seed means
     0.883 → 0.856 and 0.898 → 0.871).
5. **One number went up only by luck.** Internal trust rose from 0.811 to 0.845 at seed 42. But seeds 0–19 give
   0.818 ± 0.009 (v2b inputs: 0.825 ± 0.005), and seed 42 is the highest of the 21. So trust did not really change.
   At seed 42, trust also beats all 40 structured-null draws (p = 0.024). At the seed mean the gap is only 1.0 null SD.
   This is not a finding.
6. **MaxToki's lead over model-free features is small and depends on which token bag you use.**
   - Global order, MaxToki minus a label lookup: +0.064 [0.050, 0.091] external, +0.091 [0.027, 0.163] zero-shot.
     Minus HVG expression: +0.102 [0.073, 0.140] and +0.110 [0.038, 0.151]. (Donor bootstrap, 2,000 replicates,
     percentile 95% interval.)
   - Minus the token bag built from the correct tokens: +0.016 [0.002, 0.046] external and +0.048 [0.009, 0.149]
     zero-shot. Over 10 head seeds the external lead is +0.004 ± 0.005 (a tie). The zero-shot lead is +0.053 ± 0.005.
   - But that token bag itself got weaker with the correct tokens (zero-shot 0.879 → 0.819). The token bag built from
     the old log-ranked tokens is also model-free. MaxToki does not beat it: −0.010 [−0.023, 0.017] external,
     −0.012 [−0.031, 0.022] zero-shot (1,000 replicates, own code).
7. **Controls behave as before.** The deployed lung control (one random stage per anchor) cannot fail anything useful:
   no representation passes random + donor + branch in any of 2,000 draws. With one random stage per lung cell type,
   MaxToki passes the branch gate in 93.0% of draws (v2b 96.5%).
8. **L10H6 is still special only in trust.** Trust 0.875 vs 0.809 ± 0.011 for 30 random projections of the same
   layer (0 of 30 reach it). Branch holdout 0.419 vs 0.403 ± 0.071; 15 of 30 random projections reach it.
9. **H38, H95, H103: same reading as v2b.** H38 internal category holdout fell from 0.288 to 0.229. The H95 trust pass
   still flips with the CPU thread count (0.8018 at 1 thread, 0.7998 at 4). H103 still only asks whether two labels
   separate.
10. **Smart-seq2 cells do not drive the result.** 19.5% of internal cells are Smart-seq2 (read counts). Removing them
    changes the gate numbers by at most 0.016 on average over 10 seeds, except zero-shot within-donor (−0.078).

## 1. What was done

| Step | Script (`RUN/scripts/`) | Output (`OUT/`) | Device |
|---|---|---|---|
| Cell lists, checks, MaxToki forward passes, centroids | `v3_devorder_centroids.py` (plan / extract / finalize) | `cells/`, `percell/`, `artifacts/anchors/`, `v3_centroids_summary.json`, `run_config_centroids.json` | MPS (extract), CPU |
| Features (MaxToki, lookups, token bag, HVG) | `v3_devorder_features.py` | `features/`, `features_manifest.json`, `run_config_features_*.json` | CPU |
| All 2,078 head fits of v2b | `v3_devorder_pool.py` | `pool/results.jsonl`, `heads/`, `thread_check/` | CPU, 4 workers × 1 thread |
| v2b tables, nulls, lung control, bootstraps, N60; v2b vs v3 table | `v3_devorder_analyze.py` | `v2b_*.json` (same names as v2b), `table_*.csv`, `v3_compare.json`, `run_config.json` | CPU |
| Head-seed spread (extra) | `v3_devorder_seeds.py` | `seeds/results.jsonl`, `seeds_summary.json` | CPU |
| Independent checks | `v3_devorder_verify.py` | `verify/` | MPS (fresh), CPU |
| Shared paths | `v3_devorder_common.py` | — | — |

How the v2b analysis was re-run "exactly". The v2b code is imported, not copied. `v3_devorder_common.patch_v2b()`
points the path globals of `v2b_devorder_common.py` at OUT before any other v2b module is imported. The task list
(2,078 tasks), the task code, the run's own gate code (phase5 `train_let`, the phase7/8 frozen-panel code), the seeds
and the analysis arguments are the v2b ones. Proof that the wiring is right: **all 959 tasks whose features did not
change (label lookups, HVG) gave bit-identical results to v2b** (largest difference 0). No file outside OUT and the
new `v3_devorder_*.py` scripts was created or changed (checked: 178 v2b outputs and 48 v2b inputs have the same
sha256; file modification times).

What changed between v2b and v3: the MaxToki centroids (so the `maxtoki` and `maxtoki_pca64` features and the N60
single-head features) and the token bag (built from the new tokens). The lookups are rebuilt and identical to v2b.
The HVG feature was copied from v2b (sha256 checked) because it never used the model input (it reads `raw/X`).

## 2. The input and the forward passes

Cells: the deployed cell lists (`outputs/phase1/cells_<panel>_obs.csv`) and the deployed subsample (default_rng(42),
replayed from `phase1bc_hidden_states_and_centroids.py` and checked against the deployed `n_cells_centroided`).
Every row's obs label matched the h5ad. For 5 cells per panel the deployed tokens were rebuilt exactly from the
stored `X`, which proves the row mapping.

Forward pass: batch of 1 cell, float32, eager attention, attention mask of ones, all 12 hidden states (0 = embedding,
1–10 = block outputs, 11 = after the final RMSNorm), mean over gene positions 1..L−2. The base `LlamaModel` was called
(no output layer); its hidden states equal the full model's (difference 0.0). In this environment the deployed
pipeline reproduces a deployed centroid to 1.9 × 10⁻⁶.

| Panel | Cells | Anchors | Cells per anchor | Smart-seq2 cells | Median tokens (v3 / deployed) | Order vs deployed: Spearman / top-200 / kept set |
|---|---:|---:|---|---:|---|---|
| internal | 11,804 | 290 | 13–50 | 2,303 (19.5%) | 2,081 / 2,043 | 0.874 / 0.643 / 0.983 |
| external | 12,000 | 600 | 20 | 673 (5.6%) | 2,217 / 2,171 | 0.861 / 0.622 / 0.975 |
| zero-shot | 5,120 | 160 | 32 | 470 (9.2%) | 2,148 / 2,102 | 0.867 / 0.624 / 0.976 |
| lung non-blood | 1,500 | 50 | 30 | 186 (12.4%) | 3,093 / 2,938 | 0.865 / 0.605 / 0.959 |
| first lung control | 2,124 | 51 | 6–50 | 203 (9.6%) | 2,293 / 2,219 | 0.870 / 0.621 / 0.977 |

Spearman = rank agreement of gene positions over shared genes; top-200 = share of the first 200 genes that are the
same; kept set = share of the correct 4,094 kept genes the deployed run also kept. These match the audit (TS immune
4,096: 0.87 / 0.63). Sequences are a little longer now because `raw/X` keeps genes that decontX had set to zero.

Encoding check: `inputs_v3.tokenize_counts(check=True)` on every cell and `assert_encoding_batch` on all 50 cells of
every chunk, before any forward pass. 653 chunks, 32,548 cells, all passed; all count rows were whole numbers.

Wall time and memory: 30,157 s of forward passes (0.84–1.27 s per cell; other jobs shared the GPU), in 74 calls of
at most about 7.8 minutes. Free memory never fell below 3.1 GB. Process memory: in the first three calls the process grew by
about 6 MB per cell, because MPS keeps compiled kernels for every new sequence length. Its peak was about 8.5 GB
(3.6 GB CPU + 4.9 GB GPU), above the 8 GB target. From the fourth call on, sequences were right-padded with `<pad>`
(attention mask 0) to a multiple of 256 tokens. That cut the CPU part to ≤ 1.9 GB (peak about 6.8 GB). Padding does
not change the real positions in a causal model: 30 test cells moved by at most 3.8 × 10⁻⁶ (float32 rounding). The
first 1,200 internal cells are unpadded; the rest are padded. Both kinds were re-derived independently (section 4).

## 3. How much the centroids moved

Mean cosine between the v3 and deployed centroid of the same anchor, per hidden state (internal; other panels within
0.06): 1.00 (embedding), 0.97, 0.95, 0.91, 0.89, 0.87, 0.88, 0.85, 0.83, 0.75, 0.83, 0.82 (last). The rank order of
anchor-to-anchor distances is better kept: Spearman 0.88–0.99 per state (internal), 0.80–0.99 (other panels).
Layers 8–10 also got much larger (mean centroid size at layer 9: 24 → 53). Most of that is one hidden dimension (289),
whose mean size at layer 9 goes from 14 to 47.

## 4. Checks that the numbers are right

| Check | Result |
|---|---|
| Encoding check on every model input | 32,548 / 32,548 pass |
| Own tokeniser (h5py `raw/X`, own Ensembl → token and median maps, stable sort) vs stored tokens | 108 / 108 equal up to exact ties |
| Fresh full-model forward, no padding, vs stored per-cell states | largest difference 3.8 × 10⁻⁶ (108 cells, padded and unpadded chunks) |
| Fresh centroids of 3 whole anchors (13, 20, 30 cells) vs stored | largest difference 4.8 × 10⁻⁷ |
| Rulers rebuilt from labels vs deployed files (and the run's within-branch null ruler) | identical, 4 panels |
| Pooled-drift features with own code vs the feature files | largest difference 7.5 × 10⁻⁶ |
| Token-bag recipe on the deployed tokens vs the v2b file | identical (difference 0) |
| Lookups rebuilt vs v2b | identical sha256 (32 files) |
| v3 pool, unchanged representations vs v2b pool | 959 / 959 tasks identical |
| Own gate code on the first lung control with deployed centroids and the v2b head | 0.801 / 0.894 / 0.925 / 0.522 / 0.921, the v2 values exactly |
| Own eval-only structured null (new random stream, 2,000 draws) | internal branch 0.414 vs null mean 0.398, p = 0.38 (analysis code: 0.402, p = 0.39) |
| Own donor bootstrap (explicit resampling, new stream, 1,000 replicates) | external MaxToki − token bag +0.016 [0.002, 0.045]; zero-shot +0.048 [0.011, 0.136] (analysis code: [0.002, 0.046], [0.009, 0.149]) |
| Own structured lung control (new stream) | MaxToki branch passes 93.7% (analysis code 93.0%) |

One check failed in an informative way. Re-fitting the head from features that differ by only 7.5 × 10⁻⁶ gave
internal trust 0.813 instead of 0.845. The frozen-panel numbers moved by up to 0.014. So the single-seed numbers
carry optimiser noise. Section 6 measures it.

## 5. H65 on every panel, by representation

v3 values; v2b values in brackets where they changed. Lookup = `lookup_ct_s0` (the other seeds are unchanged from
v2b). Global = Spearman over all anchor pairs.

| Panel | Number | MaxToki | Lookup | Token bag | HVG | MaxToki PCA-64 |
|---|---|---:|---:|---:|---:|---:|
| internal (head re-fitted) | trust | 0.845 (0.811) | 0.785 | 0.705 (0.708) | 0.717 | 0.686 (0.696) |
| | random / donor | 0.849 / 0.783 (0.834 / 0.716) | 0.739 / 0.661 | 0.674 / 0.556 | 0.726 / 0.676 | 0.796 / 0.575 |
| | branch holdout | 0.414 (0.370) | 0.364 | 0.429 (0.417) | 0.354 | 0.487 (0.490) |
| | branch holdout, diff | 0.067 (0.047) | 0.056 | −0.078 (0.006) | 0.088 | 0.075 |
| external (frozen) | trust | 0.905 (0.896) | 0.851 | 0.859 (0.875) | 0.859 | 0.820 |
| | within-branch / diff | 0.417 / 0.006 (0.346 / 0.050) | 0.312 / 0.070 | 0.386 / −0.039 | 0.367 / −0.011 | 0.381 / −0.014 |
| | global / global diff | 0.865 / 0.835 (0.882 / 0.856) | 0.826 / 0.782 | 0.849 / 0.820 (0.876 / 0.850) | 0.763 / 0.720 | 0.863 / 0.833 |
| zero-shot (frozen) | trust | 0.823 (0.827) | 0.770 | 0.750 (0.759) | 0.768 | 0.735 |
| | within-branch / diff | **0.180** / 0.016 (0.317 / 0.094) | 0.319 / 0.145 | 0.133 / −0.085 (0.270 / 0.028) | 0.030 / −0.060 | 0.457 / 0.160 |
| | global / global diff | 0.867 / 0.851 (0.899 / 0.886) | 0.811 / 0.782 | 0.819 / 0.807 (0.879 / 0.866) | 0.757 / 0.746 | 0.855 / 0.835 |
| lung non-blood (frozen) | trust / branch | 0.719 / −0.045 (0.7996 / −0.147) | 0.656 / −0.017 | 0.892 / −0.060 | 0.755 / −0.144 | 0.811 / −0.157 |

Gate verdicts for MaxToki (seed 42): internal passes all four (v2b: all four); external passes all four (v2b: all
four); **zero-shot fails the branch gate** (v2b: passed); lung non-blood fails all four (v2b: all four). The lookup,
token bag and HVG verdicts are as in v2b: all pass the four gates on external and fail trust internally.

Per branch, internal held out (v3 / v2b): T lineage 0.038 / 0.031; B lineage 0.282 / 0.154; monocyte 0.020 / −0.075;
granulocyte 0.530 / 0.708; erythroid 0.783 / 0.783; stem 0.828 / 0.621. The plain mean (0.414) is still carried by
the small branches. Weighted by anchors it is 0.182 (v2b 0.153), below the gate. External frozen T lineage fell from
0.289 to 0.205. Full table: `OUT/table_gates_by_representation.csv`, per branch: `OUT/table_h65_per_branch.csv`.

## 6. Optimiser noise: head seeds 0–19 (extra check)

The run and v2b fit each head once (seed 42). I re-fitted the full H65 head with 20 seeds on the v3 features and 10
seeds on the v2b features, and scored it with the v2b gate code. Mean ± SD [min, max].

| Number | v3 (20 seeds) | v2b inputs (10 seeds) | Seed 42, v3 / v2b |
|---|---|---|---|
| internal trust | 0.818 ± 0.009 [0.809, 0.844] | 0.825 ± 0.005 [0.817, 0.835] | 0.845 / 0.811 |
| external trust | 0.904 ± 0.002 | 0.893 ± 0.002 | 0.905 / 0.896 |
| external within-branch | 0.422 ± 0.011 | 0.396 ± 0.029 | 0.417 / 0.346 |
| external within-branch, diff | 0.013 ± 0.016 | 0.140 ± 0.037 | 0.006 / 0.050 |
| external global | 0.856 ± 0.004 | 0.883 ± 0.004 | 0.865 / 0.882 |
| zero-shot trust | 0.822 ± 0.004 | 0.831 ± 0.005 | 0.823 / 0.827 |
| zero-shot within-branch | 0.147 ± 0.048 [0.057, 0.248]; 3 of 20 ≥ 0.20 | 0.355 ± 0.057 [0.219, 0.405]; 10 of 10 ≥ 0.20 | 0.180 / 0.317 |
| zero-shot global | 0.871 ± 0.005 | 0.898 ± 0.003 | 0.867 / 0.899 |
| lung non-blood trust | 0.725 ± 0.018 | 0.771 ± 0.016 | 0.719 / 0.800 |

What it means: the internal trust rise is noise. The zero-shot branch failure and the drop in global order are not.
External within-branch rises a little (0.396 → 0.422), but its different-cell-type part falls to about 0.

Seed-paired contrasts (same seed for both heads, seeds 0–9), global order:

| Contrast | External | Zero-shot |
|---|---|---|
| MaxToki − token bag (v3 tokens) | +0.004 ± 0.005 (7 of 10 > 0) | +0.053 ± 0.005 (10 of 10) |
| MaxToki − token bag, v2b inputs | +0.006 ± 0.004 | +0.018 ± 0.003 |
| MaxToki − HVG | +0.090 ± 0.007 | +0.108 ± 0.010 |
| MaxToki − lookup (seed s0) | +0.028 ± 0.003 | +0.059 ± 0.004 |

## 7. Structured null (stages re-assigned to cell-type classes within each branch)

Same method, seeds and draws as v2b. Refit null: 40 draws (internal branch holdout: 20). Eval-only null: 2,000 draws.
p = (1 + number of null values ≥ observed) / (1 + draws), one-sided.

| MaxToki number | v3 observed | Null mean (SD) | p (v3) | p (v2b) |
|---|---:|---:|---:|---:|
| internal branch holdout (refit) | 0.414 | 0.384 (0.073) | 0.38 | 0.52 |
| internal branch holdout (eval-only) | 0.414 | 0.402 | 0.39 | 0.55 |
| internal branch, diff (eval-only) | 0.067 | 0.007 | 0.27 | 0.38 |
| internal trust (refit) | 0.845 | 0.805 (0.013) | 0.024 | 0.37 |
| external within-branch | 0.417 | 0.332 (0.071) | 0.15 | 0.20 |
| external within-branch, diff | 0.006 | 0.099 (0.117) | 0.71 | 0.76 |
| external global | 0.865 | 0.843 (0.033) | 0.29 | 0.22 |
| external trust | 0.905 | 0.885 (0.011) | 0.024 | 0.10 |
| zero-shot within-branch | 0.180 | 0.124 (0.101) | 0.27 | 0.24 |
| zero-shot global | 0.867 | 0.831 (0.033) | 0.15 | 0.049 |

The two trust rows reach the smallest p that 40 draws allow. That is the seed-42 value. Using the seed mean
(section 6), internal trust is 1.0 null SD above the null mean, external 1.8 and zero-shot 1.4. 2 of 20 seeds (10%)
are above the largest internal null value. So trust does not separate the real ruler from the structured null
either. Every branch and global number stays inside its null, as in v2b. Lookup and HVG null results are identical to
v2b; the token bag's moved (p 0.12–0.78).

## 8. Donor-level intervals for the key contrasts

Donor cluster bootstrap, frozen heads (seed 42), 2,000 replicates, percentile 95% interval. Unit: donor (13 external,
12 zero-shot). Lookup = mean of 5 seeds.

| Contrast | Panel | Within-branch | Within-branch, diff | Global | Global, diff |
|---|---|---|---|---|---|
| MaxToki − lookup | external | +0.075 [−0.038, 0.138] | −0.021 [−0.187, 0.075] | **+0.064 [0.050, 0.091]** | **+0.086 [0.068, 0.119]** |
| MaxToki − token bag | external | +0.031 [−0.067, 0.094] | +0.044 [−0.016, 0.184] | +0.016 [0.002, 0.046] | +0.015 [−0.000, 0.047] |
| MaxToki − HVG | external | +0.050 [−0.062, 0.111] | +0.017 [−0.087, 0.167] | **+0.102 [0.073, 0.140]** | **+0.115 [0.086, 0.156]** |
| MaxToki − lookup | zero-shot | −0.197 [−0.567, 0.005] | −0.210 [−0.366, 0.153] | **+0.091 [0.027, 0.163]** | **+0.109 [0.041, 0.186]** |
| MaxToki − token bag | zero-shot | +0.047 [−0.332, 0.363] | +0.100 [−0.059, 0.424] | +0.048 [0.009, 0.149] | +0.044 [0.008, 0.140] |
| MaxToki − HVG | zero-shot | +0.149 [−0.177, 0.298] | +0.076 [−0.125, 0.342] | **+0.110 [0.038, 0.151]** | **+0.105 [0.036, 0.154]** |
| MaxToki − lookup | internal (held out) | +0.021 [−0.393, 0.063] | +0.013 [−0.438, 0.087] | — | — |
| MaxToki − token bag | internal (held out) | −0.016 [−0.358, 0.063] | +0.146 [−0.025, 0.594] | — | — |
| MaxToki − HVG | internal (held out) | +0.060 [−0.347, 0.083] | −0.021 [−0.131, 0.082] | — | — |

MaxToki values with intervals: external within-branch 0.417 [0.297, 0.532]; zero-shot within-branch 0.180
[−0.103, 0.369]; external global 0.865 [0.839, 0.896]; zero-shot global 0.867 [0.828, 0.888]. Internal intervals
resample evaluation donors only (heads fixed), so they are too narrow, as in v2b.

Extra comparison (own code, 1,000 replicates): MaxToki (v3) minus the token bag from the old log-ranked tokens:
−0.010 [−0.023, 0.017] external, −0.012 [−0.031, 0.022] zero-shot. MaxToki v3 minus MaxToki v2b: −0.017
[−0.032, 0.002] external, −0.032 [−0.043, −0.011] zero-shot.

Reading: on every within-branch number MaxToki is still not separable from a lookup or from the expression baselines.
On global order it beats the lookup and HVG by 0.06–0.11. Against a bag of its own input tokens it ties on external
and leads on zero-shot by about 0.05. That zero-shot lead exists only because the correct-token bag is weaker than the
old one; a model-free bag of the old tokens matches MaxToki on both panels.

## 9. Lung control

Deployed design (one random stage per non-blood lung anchor, 2,000 draws): random + donor + branch pass together in
0 of 2,000 draws for all 12 representations (v2b: the same). Branch alone passes in 10–12% of draws.

Structured design (one random stage per lung cell type, 2,000 draws), frozen H65 head:

| Representation | Branch mean | Pass branch | Pass random | Pass donor | Pass random + donor + branch |
|---|---:|---:|---:|---:|---:|
| MaxToki | 0.470 (0.527) | 93.0% (96.5%) | 15.5% | 5.8% | 5.2% (3.1%) |
| Lookup (seed 0) | 0.508 | 96.5% | 14.6% | 3.1% | 2.7% |
| Token bag | 0.563 (0.570) | 97.2% | 19.0% | 5.3% | 5.2% |
| HVG | 0.430 | 89.9% | 12.1% | 4.8% | 4.2% |

MaxToki's lung trust is 0.719, so "all four" is 0%. The conclusion is unchanged: once the random labels follow lung
cell types, the branch gate passes for meaningless labels about 93% of the time.

First lung control (lung immune cells with real stage labels, replaced in the deployed run), frozen head, own code
(seed-42 head re-fitted from own features): trust 0.799, random 0.878, donor 0.899, branch 0.470, global 0.901
(deployed: 0.801 / 0.894 / 0.925 / 0.522 / 0.921). It still passes the three correlation gates and sits at the trust
gate.

## 10. H38, H95, H103

| Ordering | v3 | v2b |
|---|---|---|
| H38 internal (MaxToki) trust / random / donor / category holdout | 0.801 / 0.758 / 0.599 / 0.229 | 0.814 / 0.761 / 0.673 / 0.288 |
| H38 external category: MaxToki / token bag / HVG / lookup s0 | 0.608 / 0.701 / 0.640 / 0.826 | 0.594 / 0.738 / 0.640 / 0.826 |
| H38 structured null (30 draws), MaxToki external category | 0.608 vs null 0.331, p = 0.032 | p = 0.032 |
| H95 internal trust (1 thread / 4 threads) | 0.8018 / 0.7998 | 0.7994 / 0.8005 |
| H95 external within-category vs structured null | 0.206 vs 0.364, p = 0.78 | 0.155 vs 0.389, p = 0.91 |
| H103 | still only "B cell" vs "plasma cell"; random and donor 0.861 / 0.861 for every representation that separates them | same |

H38: MaxToki still follows the categories, but so do the token bag and HVG, and the lookups do better. H95's pass
still depends on CPU thread count, now in the other direction. (6 threads, used in v2b, was not run; the cap here
was 4.)

## 11. L10H6 vs random projections of the same residual

Same scoring as phase9 and v2b (in-sample trust, 3 random splits, 6-branch holdout,
composite = 0.5 trust + 0.25 random + 0.25 branch). 30 Gaussian 154 × 1,232 projections of hidden state 11.

| Number | L10H6 v3 (v2b) | Random, mean ± SD | Share of random ≥ L10H6 |
|---|---:|---|---:|
| trust | 0.875 (0.882) | 0.809 ± 0.011 | 0 / 30 |
| random holdout | 0.789 (0.793) | 0.776 ± 0.016 | 17% |
| branch holdout | 0.419 (0.467) | 0.403 ± 0.071 | 50% |
| branch holdout, diff | 0.066 (−0.014) | 0.049 ± 0.061 | 43% |
| composite | 0.739 (0.756) | 0.699 ± 0.021 | 0 / 30 |

Same reading as v2b: L10H6 leads only through trust. Participation ratio 12.2 vs 14.7 (13.7–16.1) for the random
projections, so its features are again more concentrated. Not done: the 88-head screen that picked L10H6 was not
re-run on the v3 centroids, so I cannot say whether L10H6 is still the top head.

## 12. Smart-seq2 sensitivity

`raw/X` holds read counts for Smart-seq2 cells, and the model predicts their genes badly (next-gene loss 7.8 nats in
4 cells vs 3.1 in 10x cells; it was 8.3 with the old order). I rebuilt the centroids from 10x cells only. Anchors with
no 10x cell were dropped: internal 290 → 247, external 600 → 587, zero-shot 160 → 152. To isolate the cells, I
compared with all-cell centroids of the same kept anchors, over 10 head seeds.

- Paired change (10x only minus all cells), mean over seeds: internal trust +0.010, external trust +0.002, external
  global +0.003, external within-branch −0.001, zero-shot global −0.005, zero-shot within-branch −0.016,
  zero-shot within-donor −0.078. So the Smart-seq2 cells barely matter for the H65 numbers.
- But the anchor set matters. On the 247 / 587 / 152 kept anchors, zero-shot within-branch is 0.34 ± 0.01 (all cells)
  instead of 0.147 on the full panels. Dropping the 43 Smart-seq2-only internal anchors (and 8 zero-shot anchors)
  changes it by about 0.2. This is the same fragility v2 found (dropping donor TSP25 took it to 0.012).

## 13. What changed relative to deployed and v2b

1. The input is now correct for every MaxToki number in this report (v2b and deployed used log-ranked tokens).
2. **Zero-shot within-branch 0.317 → 0.180: H65 now fails one of the four gates on the zero-shot panel.** It is
   below 0.20 in 17 of 20 head seeds. It is also fragile to which anchors are in the panels (section 12).
3. Global (cross-branch) order fell about 0.03 on both frozen panels. It is still well above a label lookup and HVG.
4. Internal trust 0.811 → 0.845 is seed noise (seed mean 0.818 vs 0.825 before).
5. External within-branch 0.346 → 0.417, but the different-cell-type part fell to about 0 (seed mean 0.140 → 0.013).
6. MaxToki vs its own token bag: external still a tie (+0.004 ± 0.005 over seeds); zero-shot lead grew from +0.02 to
   +0.05, because the correct-token bag is weaker. A bag of the old tokens matches MaxToki.
7. Unchanged: no developmental order beyond cell type (structured nulls p 0.15–0.39; diff-cell-type branch 0.006–0.067;
   T lineage 0.038); the lung control cannot fail in its deployed form and passes the branch gate in 93% of structured
   draws; L10H6 leads only in trust; H38/H95/H103 readings.

**What is specific to MaxToki:** still nothing about developmental order that these tests can detect. Its only edge
over a label lookup and HVG expression is cross-lineage arrangement (+0.06 to +0.11). A model-free bag of rank-ordered
gene tokens reaches the same level.

## 14. What I could not do, and limits

- **No random-initialised MaxToki** and no other cell model (scVI, Geneformer) in this item.
- **The 88-head screen (phase9) was not re-run** on the v3 centroids. The N60 "best of 88" numbers still use the
  deployed screen.
- **The v2 interval analyses** (`V2_INTERVALS_AND_FACTS_REPORT.md`: frozen-head donor bootstrap of trust,
  training-donor bootstrap, fifth gate, compaction chain, factor ablation) were not re-run. Only the v2b items were.
  The donor intervals here cover the within-branch and global numbers and the contrasts, not trust.
- **Refit nulls are coarse:** 40 draws (20 for internal branch), so the smallest p is 0.024 (0.048).
- **All main-table numbers use one head seed (42), as v2b did.** Section 6 adds seeds only for the full fits, not
  for the held-out branch fits or the nulls.
- **Internal bootstrap intervals are too narrow** (evaluation donors only, heads fixed).
- **Thread check:** 1 and 4 threads only (v2b also used 6; the cap here was 4).
- **Smart-seq2 cells were kept** in the main analysis (they are what `raw/X` gives). Section 12 shows the effect.
- **Process memory went over the 8 GB target** (about 8.5 GB) in the first three extraction calls, before the
  length padding fix.
- Raw h5ad files (19.8 GB, 3.2 GB) were fingerprinted (size, time, sha256 of the first and last 16 MB), not hashed in
  full. The rows used are listed in `run_config_centroids.json`.

## 15. Files

- Report: `RUN/V3_DEVORDER_REPORT.md` (this file).
- Centroids and metadata: `OUT/artifacts/anchors/centroids_<panel>.npy` (n × 12 × 1,232), `anchor_meta_<panel>.csv`,
  rulers; per-cell states and tokens: `OUT/percell/<panel>/chunk_*.npz`; tokens: `OUT/cells/tokens_<panel>.npz`;
  cell plans: `OUT/cells/plan_<panel>.csv`.
- Analysis: `OUT/v2b_tables.json`, `OUT/table_gates_by_representation.csv`, `OUT/table_h65_per_branch.csv`,
  `OUT/v2b_evalnull_internal_branch.json`, `OUT/v2b_lung_control_6161.json`, `OUT/v2b_lung_control_6262.json`,
  `OUT/v2b_boot_*.json`, `OUT/v2b_n60_*.json`, `OUT/v3_compare.json` and `OUT/table_v3_vs_v2b.csv` (338 numbers,
  v2b vs v3), `OUT/seeds_summary.json`, `OUT/verify/*.json|csv`.
- Provenance: `OUT/run_config_centroids.json` (device, versions, seeds, every cell row and obs label, encoding record
  and check summary per panel, sha256 of inputs, code and outputs, wall time per chunk), `OUT/run_config.json`
  (analysis), `OUT/run_config_features_*.json`, `OUT/verify/run_config.json`.

## Plain-words summary

The manifold run gave MaxToki genes in the wrong order. I gave it the right order for all 32,548 cells and checked
every cell before the model saw it. The model fits the right order better: it predicts the next gene with about one
nat less loss.

Then I re-ran the whole v2b comparison with the same code. The answer is the same. MaxToki shows no blood
developmental order that a random code per cell-type label, or plain expression, does not also show. A shuffled stage
ruler that keeps cell types together still scores like the real one.

Two things got worse. The zero-shot branch number is now 0.18, below the 0.20 pass mark, and it stays below in most
head seeds. So H65 no longer passes all four checks on the zero-shot panel. The order across lineages also dropped by
about 0.03. One thing looked better, internal trust, but that was the luck of one random seed.

MaxToki still arranges lineages better than a label lookup or standard expression features. A model-free bag of its
input tokens does about as well. The lung control, the single head L10H6 and the H38/H95/H103 orderings read the same
as before.
