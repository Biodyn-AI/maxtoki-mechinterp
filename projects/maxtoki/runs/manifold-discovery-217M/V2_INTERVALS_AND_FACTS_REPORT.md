# Manifold discovery on MaxToki-217M — donor-level intervals, the fifth gate, and exact definitions (item D8)

Date: 2026-10-01. CPU only. No model forward pass. All numbers below come from saved v2 scripts run on saved
run files. Short names:

- RUN = `projects/maxtoki/runs/manifold-discovery-217M`
- OUT = `RUN/outputs/v2_intervals`
- "anchor" = one group of cells sharing donor × tissue × cell type × stage label (its mean hidden state is one data point).
- "gate" = a pass mark fixed before the run in `RUN/reports/quality_gates_spec.json`: trustworthiness ≥ 0.80,
  three Spearman correlations ≥ 0.20, and a blocked-permutation p ≤ 0.001 with ≥ 2,000 permutations.
- "frozen head" = the 10-number read-out fitted once on the internal panel and then applied unchanged to other panels.

Resampling unit everywhere: the **donor**, unless a line says otherwise. Every interval is a percentile bootstrap
interval (2.5% and 97.5% of the replicates) unless a line says otherwise.

---

## 1. Main results in one table (H65, the hematopoietic ordering)

| Panel and number | Value | Donor-level 95% interval | Share of resamples below gate | Leave-one-donor-out range | Deployed interval |
|---|---:|---|---:|---|---|
| Internal trust (head re-fitted) | 0.811 | [0.732, 0.872] (300 reps, 11 training donors) | 34% below 0.80 | 0.763 (drop TSP2) – 0.857 | none |
| Internal branch-holdout, plain mean of 6 branches | 0.370 | [0.241, 0.473] (300 reps) | 1.7% below 0.20 | 0.341 – 0.436 | none |
| Internal branch-holdout, anchor-weighted mean | **0.153** | [0.064, 0.245] (300 reps) | 72% below 0.20 | 0.124 – 0.189 (all 11 below 0.20) | not reported |
| External trust (frozen head) | 0.896 | [0.864, 0.918] (2,000 reps, 13 donors) | 0.05% below 0.80 | 0.884 (drop TSP25) – 0.899 | [0.884, 0.899] |
| External within-branch correlation (frozen head) | 0.346 | [0.146, 0.535] (2,000 reps) | 4.8% below 0.20 | 0.305 (drop TSP11) – 0.408 | [0.270, 0.480] |
| External trust, both donor levels | 0.896 | [0.866, 0.936] (300 reps) | 0% | — | — |
| External within-branch, both donor levels | 0.346 | [0.267, 0.567]; basic interval [0.126, 0.426] | 1.0% | — | — |
| Zero-shot trust (frozen head) | 0.827 | [0.749, 0.873] (2,000 reps, 12 donors) | **18% below 0.80** | 0.7995 (drop TSP21) – 0.838 | none |
| Zero-shot within-branch correlation | 0.317 | [−0.029, 0.471] (2,000 reps) | **20% below 0.20** | **0.012 (drop TSP25)** – 0.428 | none |
| Zero-shot trust, both donor levels | 0.827 | [0.743, 0.878] (300 reps) | 22% | — | — |
| Zero-shot within-branch, both donor levels | 0.317 | [−0.007, 0.468] (300 reps) | 20% | — | — |
| Lung control trust (frozen head) | 0.7996 | [0.749, 0.817] (2,000 reps, 4 donors) | 63% below 0.80 | 0.736 – 0.767 | none |
| Lung control within-branch correlation | −0.147 | [−0.426, 0.170] | 99% below 0.20 | −0.383 – −0.163 | none |

"Both donor levels" means: resample the 11 training donors and re-fit the head, then resample the evaluation donors.
"Basic interval" is a second bootstrap method, [2 × observed − upper percentile, 2 × observed − lower percentile].
It is shown where it disagrees clearly with the percentile interval (section 3.4).

What the table means in plain words:

- **External trust holds up.** Donor resampling widens the interval about 3.5 times. Only 1 of 2,000 resamples fell below 0.80 (to 0.799).
- **External within-branch correlation is weaker than reported.** The lower end moves from 0.270 to 0.146, below the 0.20 gate.
- **Zero-shot is fragile.** About one resample in five fails each gate. Dropping one donor (TSP25) takes the branch value from 0.317 to 0.012.
- **The internal trust gate is not stable across training donors.** One in three resamples of the 11 training donors gives trust below 0.80. Dropping donor TSP2 alone gives 0.763.
  *Added in verification:* most of this is panel size, not donor TSP2. Trustworthiness falls when a panel has fewer anchors.
  Random 103-anchor subsets of the internal panel give 0.774 on average with the head re-fitted (12 draws), about the same as dropping TSP2.
  See "Verification notes", V2.
- **The internal branch-holdout 0.370 depends on how branches are weighted.** Weighted by anchors it is 0.153, below the gate.

---

## 2. What was done

| Step | Script (`RUN/scripts/`) | Output (`OUT/`) |
|---|---|---|
| Reproduce every deployed H65 number; per-branch rows | `v2_00_reproduce.py` | `v2_00_reproduce.json`, `v2_00_per_branch.csv` |
| Donor bootstrap and leave-one-donor-out, frozen head (external, zero-shot, lung) | `v2_01_frozen_donor_bootstrap.py` | `v2_01_frozen_donor_bootstrap.json`, `frozen_boot/*.jsonl` |
| All head re-fits as a resumable task pool (reproduction, 300 training-donor bootstrap reps, 2,000 permutations, 10 seeds) | `v2_02_refit_pool.py` | `refit_pool/results.jsonl` (3,833 tasks) |
| Summary of the re-fits | `v2_03_aggregate_refit.py` | `v2_03_refit_summary.json`, `v2_03_internal_per_branch.csv`, `v2_03_boot_internal_reps.csv` |
| Fifth gate (blocked permutation) | `v2_04_permutation_gate.py` | `v2_04_permutation_frozen.json`, `v2_04_permutation_internal.json`, `perm_null_*.npy` |
| H38, H95, H103 re-fits and donor bootstraps | `v2_05_other_orderings.py` | `v2_05_other_orderings.json`, `orderings/` |
| Facts (definitions, cohorts, orderings, compaction, factor ablation) | `v2_06_facts.py` | `v2_06_facts.json`, `facts_cohorts.csv`, `facts_orderings.csv` |
| Leave-one-training-donor-out with re-fit; first lung control | `v2_10_internal_lotdo.py` | `v2_10_internal_lotdo.json`, `lotdo/results.jsonl` |
| Second-method checks (basic interval, bias, end-point precision, cross-checks) | `v2_11_interval_checks.py` | `v2_11_interval_checks.json` |
| Source data for the table and figure | `v2_07_tables.py` | `table_gates.csv`, `fig_gates.csv` |
| Task-data folder for a later controlled experiment | `v2_08_task_data.py` | `task_data/` (108 MB, README, manifest) |
| One combined provenance file | `v2_09_run_config.py` | `run_config.json` |

Provenance. Scripts v2_00 to v2_09 were written and run earlier on 2026-10-01 (01:07–06:01) as part of this item.
This pass checked them line by line, added v2_10 and v2_11, extended v2_07, v2_08 and v2_09, and rebuilt the
tables, the task-data folder and `run_config.json`. Each script writes `run_config_<script>.json` with the sha256
of every input. `run_config.json` lists 81 inputs, 14 scripts and 56 outputs with sha256, plus all seeds.
No file outside `OUT/`, the new `v2_*.py` scripts and this report was created or changed (checked by modification time).

Seeds: LET head torch seed 42 (as deployed); frozen-panel donor bootstrap `default_rng([4242, panel, rep])`;
training-donor bootstrap `default_rng([20261001, rep])`, evaluation level `default_rng([20261001, rep, 7])`;
permutations `default_rng([777, b])` (internal) and `default_rng([777, panel, b])` (frozen panels);
other orderings `default_rng([5353, key, rep])`; end-point precision `default_rng(99)`.

Copies of one anchor (which appear when a donor is drawn twice) are never paired and never count as each
other's neighbours. With no copies, this trustworthiness code gives exactly sklearn's value (difference 2e-7 on
the external panel, 0 on the others).

---

## 3. Results in detail

### 3.1 Reproduction (all exact)

- Re-fitting the Phase-5 head from saved centroids gives the saved weights exactly (max difference 0, at 1 and at 6 torch threads).
- Frozen-head values match the deployed reports to all digits: external trust 0.895688, within-branch 0.346443,
  within-donor 0.886953, random 0.885775, global 0.882120; zero-shot 0.826724 / 0.316830 / 0.870309;
  lung 0.799556 / −0.146871 / 0.013397 / −0.016420.
- Internal re-fits give trust 0.810936, branch-holdout 0.370404, donor-holdout 0.715888 and null branch-holdout
  −0.001012, all as deployed. The null value needs 4 torch threads; at 1 thread it is −0.003455, because one null
  branch (monocyte) moves from 0.195 to 0.180. This is floating-point noise in the optimiser, not a code change.

### 3.2 Item 1 — donor bootstrap of the frozen-head numbers

Method: draw the panel's donors with replacement (13 external, 12 zero-shot, 4 lung), keep all anchors of each
drawn donor, score the frozen head. 2,000 replicates per panel. Leave-one-donor-out: drop each donor once.

| Panel | Metric | Value | 95% interval | Below gate | Leave-one-out range |
|---|---|---:|---|---:|---|
| external | trust | 0.896 | [0.864, 0.918] | 0.05% | 0.884 – 0.899 |
| external | within-branch, plain mean | 0.346 | [0.146, 0.535] | 4.8% | 0.305 – 0.408 |
| external | within-branch, anchor-weighted | 0.333 | [0.297, 0.398] | 0% | 0.324 – 0.364 |
| external | within-donor | 0.887 | [0.845, 0.917] | 0% | 0.882 – 0.903 |
| external | global Spearman | 0.882 | [0.860, 0.900] | — | 0.878 – 0.890 |
| zero-shot | trust | 0.827 | [0.749, 0.873] | 18% | 0.7995 – 0.838 |
| zero-shot | within-branch, plain mean | 0.317 | [−0.029, 0.471] | 20% | 0.012 – 0.428 |
| zero-shot | within-branch, anchor-weighted | 0.311 | [0.142, 0.430] | 8.6% | 0.271 – 0.339 |
| zero-shot | within-donor | 0.870 | [0.821, 0.918] | 0% | 0.857 – 0.888 |
| lung control | trust | 0.7996 | [0.749, 0.817] | 63% | 0.736 – 0.767 |
| lung control | within-branch | −0.147 | [−0.426, 0.170] | 99% | −0.383 – −0.163 |
| lung control | within-donor | 0.013 | [−0.055, 0.069] | 100% | −0.009 – 0.049 |

Checks: an independent script from the investigation (different code, different random stream) gives external
trust [0.865, 0.918] and branch [0.142, 0.540]; zero-shot [0.751, 0.871] and [−0.030, 0.471]. These agree with the
table to within 0.006. The Monte-Carlo error of each end point (SD from re-drawing the 2,000 replicates) is at most
0.008 for external and zero-shot, and 0.012 for the lung branch value.

The deployed interval [0.884, 0.899] came from 200 random 80% subsets of the 600 anchors, drawn without replacement
(`RUN/scripts/audit_a3_bootstrap_h65_gates.py`). That is a stability check of anchors, not a donor-level interval.

### 3.3 Item 2 — internal panel, bootstrap over the 11 training donors with the head re-fitted

Method: draw the 11 internal donors with replacement, keep all their anchors, re-standardise the features on the
drawn rows, re-fit the head (exact Phase-5 trainer, seed 42), compute in-sample trust, and re-fit once more per
held-out branch for branch-holdout. 300 complete replicates (1,767 head fits). Training sets ranged from 49 to
1,043 rows. 36% of replicates contain no copy of donor TSP2.

| Metric | Value | 95% interval | Basic interval | Below gate | Leave-one-training-donor-out |
|---|---:|---|---|---:|---|
| trust (in-sample, k = 15) | 0.811 | [0.732, 0.872] | [0.750, 0.890] | 34% | 0.763 (drop TSP2) – 0.857 (drop TSP3) |
| branch-holdout, plain mean | 0.370 | [0.241, 0.473] | [0.268, 0.500] | 1.7% | 0.341 (drop TSP4) – 0.436 (drop TSP17) |
| branch-holdout, anchor-weighted | 0.153 | [0.064, 0.245] | [0.061, 0.243] | 72% | 0.124 – 0.189 |

*Added in verification:* the plain-mean branch value is a mean over a different set of branches in different
replicates. 193 replicates score all 6 branches; the 107 without TSP2 score only 2 or 3. Using only the 193
six-branch replicates gives plain mean [0.281, 0.466] (0% below 0.20) and anchor-weighted [0.122, 0.244] (68% below).
The leave-one-out value without TSP2 (0.363) is also a mean over 3 branches, not 6.

Optimiser noise, for scale: the same full fit with torch seeds 1–10 gives trust 0.817–0.835 for H65 and
0.788–0.803 for the null ruler. Seed 42 (0.811) is below all ten other seeds. The H65-minus-null trust gap is
0.012 at seed 42 and 0.029 on average over seeds 1–10.

The same re-fitted heads, scored on the external and zero-shot panels (only the training donors resampled):
external trust [0.894, 0.926], external branch [0.327, 0.489]; zero-shot trust [0.811, 0.845], zero-shot branch
[0.203, 0.450]. Leave-one-training-donor-out: external trust 0.895–0.910, external branch 0.323–0.442 (highest
when TSP2 is dropped); zero-shot trust 0.826–0.836, branch 0.240–0.351. These agree with the investigation's
separate leave-one-out run to within 0.009 (it used 4 torch threads, this run 1).

### 3.4 The two-level external interval is off-centre

When training donors are resampled, the external within-branch value goes **up** on average: bootstrap mean 0.409
(training donors only) and 0.434 (both levels), against 0.346 observed. The head fitted on all 11 donors sits on the
low side of what other training sets give. Dropping TSP2 gives 0.442. So the percentile interval [0.267, 0.567] and
the basic interval [0.126, 0.426] disagree. Both are reported. Either way, the lower end is between 0.13 and 0.27,
and 1% of two-level replicates fell below 0.20. For the zero-shot branch value the two methods disagree the other
way (percentile [−0.007, 0.468], basic [0.165, 0.640]); the share below the gate (20%) is the safer summary.

### 3.5 Item 3 — branch-holdout per branch

Internal panel (head re-fitted without each branch; interval = training-donor bootstrap of that branch):

| Held-out branch | Anchors | Stages | Donors | H65 rho | 95% interval | Null rho |
|---|---:|---:|---:|---:|---|---:|
| T_lineage | 101 | 6 | 9 | 0.031 | [−0.230, 0.128] | −0.072 |
| B_lineage | 45 | 3 | 7 | 0.154 | [0.112, 0.840] | 0.015 |
| monocyte | 34 | 2 | 5 | −0.075 | [−0.098, 0.119]* | 0.195 |
| granulocyte | 25 | 3 | 5 | 0.708 | [0.400, 0.864] | −0.026 |
| erythroid | 7 | 2 | 3 | 0.783 | [0.743, 0.853]* | 0.296 |
| stem | 4 | 2 | 2 | 0.621 | [0.092, 0.866]* | −0.414 |
| **plain mean** | | | | **0.370** | t-interval over branches [−0.025, 0.765] | −0.001 |
| **anchor-weighted mean** | | | | **0.153** | | −0.001 |
| pair-weighted mean | | | | 0.072 | | −0.035 |

\* These three branches have more than one stage only because of donor TSP2 (HSC, MEP and NC_Mono anchors come
only from TSP2). In the 107 bootstrap replicates without TSP2 they cannot be scored, so their intervals use 193 replicates.

The three largest branches (180 of 216 scored anchors) give 0.031, 0.154 and −0.075. The plain mean is carried by
three small branches with 36 anchors in total. The four other branches (macrophage, NK, lymphoid, dendritic, and
two single-anchor branches) have one stage each and give no value.

Frozen head, within-branch correlation (nothing held out):

| Branch | External anchors / rho | Zero-shot anchors / rho | Lung control anchors / rho |
|---|---|---|---|
| T_lineage | 219 / 0.289 | 56 / 0.392 | 14 / 0.187 |
| B_lineage | 106 / 0.375 | 18 / 0.160 | 4 / −0.441 |
| granulocyte | 57 / 0.601 | 23 / 0.587 | 10 / 0.135 |
| erythroid | 21 / 0.579 | 5 / 0.853 | — |
| monocyte | 54 / 0.075 | 20 / −0.188 | 5 / −0.402 |
| stem | 7 / 0.159 | 4 / 0.098 | — |
| dendritic | — | — | 5 / −0.213 |
| plain mean | 0.346 | 0.317 | −0.147 |
| anchor-weighted mean | 0.333 | 0.311 | −0.023 |

### 3.6 Item 4 — the fifth gate (blocked permutation p)

What the frozen spec says: p ≤ 0.001, at least 2,000 permutations, blocked by donor and tissue, two-sided. It does
not name the test statistic. I used the Spearman correlation between the head's pairwise arc-cos distances and the
H65 ruler over all anchor pairs (the run's "global correlation"; the source paper attaches its blocked p to the
matching "geodesic-biological correlation"). Null: the stage labels are shuffled among anchors of the same
donor × tissue block, the ruler is rebuilt on the 34-stage graph, and the statistic is recomputed. Two-sided
p = 2 × min(one-sided p's), with the +1 correction. No script in the deployed run computes this gate.

| Panel | Head | Permutations | Observed | Null mean (max) | p (two-sided) | Gate p ≤ 0.001 |
|---|---|---:|---:|---|---:|---|
| internal | re-fitted for every permutation | 2,000 | 0.98683 | 0.97841 (0.98659) | 0.0010 | passes only at the limit |
| external | frozen | 10,000 | 0.882 | 0.043 (0.110) | 0.0002 | passes (z = 64) |
| zero-shot | frozen | 10,000 | 0.899 | 0.319 (0.584) | 0.0002 | passes (z = 9.5) |
| lung control | frozen | 10,000 | −0.017 | 0.000 (0.226) | 0.74 | fails |

Reading the internal row:

- On the internal panel the head is fitted to the ruler, so a fair null must re-fit the head on each shuffled ruler.
  When it does, the head fits shuffled rulers almost as well as the real one (in-sample 0.978 on average vs 0.987).
- None of the 2,000 null values reached the observed one, so p = 1/2001 doubled = 0.0010. This is the smallest p that
  2,000 permutations can give. The margin over the largest null value is 0.0002.
- That margin is smaller than optimiser noise. With torch seeds 1, 7 and 9 for the observed fit, 124, 10 and 17
  null values exceed it (p = 0.125, 0.011, 0.018). The other 7 seeds keep p = 0.0010.
- "Two-sided" has two common forms. Doubling the one-sided p gives 0.0010. Counting null values as far from the
  null mean as the observed one gives 0.106 (the null has a long lower tail, down to 0.842). The spec does not say which.
- With trustworthiness as the statistic on the same permutations: observed 0.811, null mean 0.586, null max 0.703,
  p = 0.0010.
- Blocks: 44 donor × tissue blocks (9 with one anchor); a median of 227 of 290 labels change per permutation.

On the frozen panels the gate is clear. The external and zero-shot correlations are far above their nulls.
*Added in verification:* the run's own null ruler (stage labels shuffled within each branch) passes this gate too
(external 0.852, zero-shot 0.874, no permutation value reaches either; Verification notes, V5). So passing it shows
that the head's distances follow branch-level structure. It does not show that they follow the H65 order inside a branch.
Zero-shot has 46 one-anchor blocks out of 92; their labels cannot move, so the null keeps part of the real structure
(null mean 0.319). That makes the test harder to pass, not easier.

Check: the permutation statistic was recomputed with `scipy.stats.spearmanr` for 300 external and all 10,000 lung
permutations. It matches the saved values to 6e-17.

### 3.7 Item 5 — H38, H95 and H103 cells that were missing or measured differently

| Ordering | Panel | Trust | 95% donor interval | Below 0.80 | Fourth gate |
|---|---|---:|---|---:|---|
| H38 LITE (7 hand-made signalling categories) | internal | 0.814 | — | — | category holdout 0.293 (passes; 4 gates) |
| H38 LITE | external, 580 anchors, 13 donors | **0.883** | [0.850, 0.907] | 0% | within-category 0.594 [0.494, 0.673] |
| H38 LITE | zero-shot, 154 anchors, 12 donors | **0.787 — fails** | [0.712, 0.826] | 78% | within-category 0.585 [0.429, 0.683] |
| H95 (effector modality, 4 categories) | internal | 0.800 | — | — | category holdout undefined → 3-gate fallback |
| H95 | external, 441 anchors | 0.888 | [0.867, 0.903] | 0% | within-category 0.155 [0.090, 0.211] (fails; 1 scorable category) |
| H95 | zero-shot, 104 anchors | **0.819** | [0.739, 0.868] | 25% | within-category −0.004 (fails; 1 scorable category) |
| H103 (B-cell maturation) | internal, 45 B-lineage anchors | 0.860 | — | — | depth holdout undefined |
| H103 | external, 106 B-lineage anchors, k = 15 | 0.877 | — | — | — |
| H103 | external, all 600 anchors, k = 15 | 0.884 | [0.853, 0.906] | 0% | — |
| H103 | zero-shot, 18 B-lineage anchors, 9 donors, k = 5 (deployed) | 0.857 | — | — | — |
| H103 | zero-shot, 18 B-lineage anchors, k = 8 (largest allowed) | 0.831 | — | — | — |
| H103 | **zero-shot, all 160 anchors, 12 donors, k = 15** | **0.777 — fails** | [0.745, 0.832] | 73% | — |

Notes:
- H38 external and zero-shot and H95 zero-shot were all tested in the deployed run (`RUN/reports/external_validation_H38_lite.json`,
  `zeroshot_H38_lite.json`, `zeroshot_H95.json`); the paper's table shows them as "not tested". My re-fits reproduce
  the deployed values exactly.
- H103's ruler has only two values in practice: every kept anchor is "b cell" (depth 1) or "plasma cell" (depth 5).
  So its depth holdout is undefined on every panel. Trust on all anchors uses the H103 head on cells it was not
  trained on; it is the only way to use k = 15 on the zero-shot panel.

### 3.8 Item 6 — lung control, every gate

The lung control is a separate panel: 50 anchors of non-blood lung cells from 4 donors (TSP1, TSP14, TSP2, TSP25),
with stage labels drawn at random from the 34 stages. TSP2 is also a training donor. It is scored with the frozen head.

| Gate | Value | Pass mark | Result |
|---|---:|---|---|
| trustworthiness | 0.79956 | ≥ 0.80 | fails (by 0.0004); 63% of donor resamples below |
| random holdout (frozen) | −0.016 | ≥ 0.20 | fails |
| donor (within-donor) | 0.013 | ≥ 0.20 | fails |
| branch (within-branch, 5 branches) | −0.147 | ≥ 0.20 | fails |
| blocked permutation p | 0.74 | ≤ 0.001 | fails |
| global Spearman | −0.017 | — | — |

So the lung control fails all five gates, not "three of the four". A first lung control (51 anchors of lung immune
cells with their real stage labels) passed all four numeric gates: trust 0.801, random 0.894, donor 0.925,
branch 0.522, global 0.921 (recomputed here; identical to `RUN/reports/external_validation_lung_control.json`).
The run replaced it with the non-blood panel.

---

## 4. FACTS — what each number actually is (item 7)

### 4.1 Trustworthiness
- Code: `sklearn.manifold.trustworthiness(features, z, n_neighbors=15)` (`phase5_let_anchor.py:236`,
  `phase7_external_validation.py:94`, `phase8_zeroshot_transfer.py:76`).
- The two spaces: the 2,464-number pooled-drift features of each anchor (standardised with the internal mean and SD)
  and the 10-number head output for the same anchors. Euclidean distance in both.
- k = 15 neighbouring **anchors**, not cells.
- The biological ruler is **not** an input. The number asks whether the small read-out keeps the model's own
  neighbourhoods. It does not ask whether model neighbours are biological neighbours.
- Internal: in-sample (the head was fitted on the same 290 anchors). External, zero-shot, lung: frozen head.
- It depends on the number of anchors; panels of different size are not directly comparable.

### 4.2 Internal branch-holdout vs external "branch-holdout"
- Internal (0.370): for each branch with ≥ 3 anchors, re-fit the head without that branch, project the held-out
  anchors, and take Spearman(arc-cos distance in the head output, ruler distance) over held-out pairs. Plain mean
  over the 6 branches whose ruler is not constant (`phase5_let_anchor.py:167-199`).
- External (0.346), zero-shot (0.317), lung (−0.147): the head is frozen and was fitted on **all** internal
  branches. For each branch, Spearman over within-branch pairs; plain mean. **Nothing is held out**
  (`phase7_external_validation.py:132-150`, `phase8_zeroshot_transfer.py:120-139`). It is a within-branch correlation.
- Donor numbers differ in the same way: internal = re-fit leaving one donor out (9 of 11 donors have ≥ 3 anchors);
  frozen panels = mean within-donor correlation.
- Random holdout: internal = 10 random 80/20 splits with re-fit (the code comment says 20); frozen = 20 random 20%
  subsets, no re-fit.

### 4.3 The compaction ("7.7 KB", "2,400×", "60")
- The 7,744-byte object is a sparse form of one attention head's output map, layer 10 head 6 (L10H6). Start from
  `A = o_proj(layer 10)[:, 6·154:7·154]ᵀ` (154 × 1,232). Keep its top 16 singular triplets. In each, keep the 60
  largest entries of the 154-long left vector and the 60 largest of the 1,232-long right vector
  (`phase10_compaction_chain.py:148-165`). Stored values: 16 × (60 + 60) + 16 = 1,936 float32 = 7,744 bytes.
- **The "60" are coordinates, not genes**: 60 of 154 head-dimension coordinates and 60 of 1,232 residual-stream
  coordinates. The run's own comment says no gene mapping exists for either (`phase11_factor_ablation.py:274`).
- **2,352×** = 18,213,888 bytes (the three 1,232 × 1,232 pooled-drift matrices behind the 2,464-number features) ÷ 7,744.
  From the 154-dimensional single-head form (758,912 bytes) the factor is **98×**. The paper attributes ~2,400× to
  the 154-dimensional form.
- Not counted in 7.7 KB: the positions of the kept entries (1,920 indices; 3,840 bytes as int16), the 10-number read-out
  head on top (1,695 floats = 6,780 bytes; saved file 8,973 bytes), the 154 means and SDs used to standardise
  (computed at run time, not saved), and a full MaxToki-217M forward pass for every cell, then averaging cells into anchors.
- Quality claims for the compact forms are internal-panel only, in-sample, with the head re-fitted per variant, and
  with no interval and no external or zero-shot test. L10H6 was chosen on the same panel by
  0.5 × trust + 0.25 × random + 0.25 × branch (`phase9_head_attribution.py:98-105`). Trust values of different variants
  compare different input spaces (2,464 vs 154 numbers), so a higher trust is not better biology.
- The single-head feature reads hidden-state entry 11, which is the last layer's output after the final RMSNorm.

### 4.4 "0.973 → 0.123"
- It is the **balanced accuracy of a linear branch classifier**, not branch-holdout. Setup: rank-64 SVD of the L10H6
  operator; a head fitted on the 290 internal anchors; five linear probes fitted on the head output of the same
  anchors (no holdout); then ablations scored on the same anchors with head and probes frozen
  (`phase11_factor_ablation.py:150-193`).
- 0.973 = all 64 factors kept. 0.123 = only the 4 "core" factors kept. There are 12 branch classes, so chance is 0.083.

### 4.5 "Four strongest axes" and 18.3% vs 66.2%
- The four axes are factors 1, 4, 11 and 9. They were chosen as the top 4 by **single-factor ablation impact**, not by
  size. By singular value they rank 2nd, 5th, 12th and 10th.
- Impact of a factor = sum over 7 endpoint scores of max(0, intact − ablated) when that one factor is removed
  (`phase11_factor_ablation.py:229`; the comment says "sum of squared drops", the code sums plain drops).
- 18.3% = impact of those 4 factors ÷ impact summed over all 64 (0.1825; recomputed from the CSV, same value).
  Using the 4 largest singular values instead gives 8.9%.
- 66.2% is the source paper's figure for scGPT, copied from the pipeline spec. It was not recomputed here, and the
  same definition is assumed, not verified.

### 4.6 Cohorts

| Panel | Anchors | Donors | Tissues | Cells averaged | Cells per anchor | Largest donor share | Progenitor-stage anchors |
|---|---:|---:|---:|---:|---|---:|---:|
| internal (training) | 290 | 11 | 39 | 11,804 | 13–50 | TSP2 64% (187) | 6 (all TSP2 except one MPP) |
| external | 600 | 13 | 46 | 12,000 (of 30,000 gathered) | 20 | 22% | 11 |
| zero-shot | 160 | 12 | 45 | 5,120 (of 7,796) | 32 | 28% | 7 |
| lung control (non-blood, random labels) | 50 | 4 | 1 | 1,500 | 30 | 30% | (random labels) |
| first lung control (immune, replaced) | 51 | 4 | 1 | 2,124 | 6–50 | 29% | 0 |

Overlap: internal and external share no donor and no cell. **All 12 zero-shot donors are external donors** (only TSP19 is
missing). Zero-shot and external share no anchor and no cell. 41 of the 45 zero-shot tissues also appear in external,
and 13 zero-shot anchors share donor, tissue and cell type with an external anchor (they differ only in stage label).
So the zero-shot panel is new anchors from the same people, not a new cohort. The lung control shares TSP2 with
training and TSP1, TSP14, TSP25 with external.

### 4.7 Which orderings ran where, under which gates

| Ordering(s) | When | Ruler | Gates applied | Result |
|---|---|---|---|---|
| H65 | Phase 5, 2026-05-03 | shortest path on the 34-stage graph | 4 numeric gates + within-branch-shuffle null; permutation gate not computed | passes 4; null fails trust and branch, passes random and donor |
| H38 LITE | Phase 13, 2026-05-04 | Hamming on 7 hand-made signalling categories (not OmniPath) | 4 gates (category holdout 0.293) + shuffle null | passes 4 internally |
| H92–H97 (sweep 1) | 2026-05-05 17:37 | Hamming on cell-type categories | 4 gates | all 6 INCONCLUSIVE (category holdout undefined) |
| 3-gate fallback | script written 18:19; sweep 1 re-scored 18:27 | — | trust + random + donor, null must fail | H95 INCONCLUSIVE → POSITIVE_3GATE_FALLBACK (trust 0.8005, 0.0005 over the gate) |
| H101–H106 (sweep 2) | script 18:34, results 18:52 | ordinal depth | 4 gates, fallback built in; this fallback does **not** check the null | H103 POSITIVE_3GATE_FALLBACK; others INCONCLUSIVE |
| H107–H112 (sweep 3) | 2026-05-07 12:50 | Hamming + ordinal | 4 gates / fallback / "directional" tag; verdicts re-classified by hand after a code bug (`STATUS.md:84`) | 0 positive |
| H107 graph rescue | Phase 16 | tree shortest path | 4 gates + null | trust 0.769, not positive |

So: yes, a three-gate fallback was added **during** the sweeps, 42 minutes after sweep 1 produced no positive, and
it turned H95 from inconclusive to positive. 18 candidates ran in sweeps (the paper says 12). H65 was fixed in
advance from the source paper; it was not a sweep winner. No sweep candidate passed all four gates. There is no
141-item catalogue in the run (141 is the topology pipeline's number), and no record of H115 or H118. The research
plan listed 7 branch groups; the stage file used has 12.

---

## 5. What differs from the deployed claims

1. External trust interval: [0.884, 0.899] → **[0.864, 0.918]** by donor, [0.866, 0.936] with both donor levels.
2. External branch interval: [0.270, 0.480] → **[0.146, 0.535]**; 4.8% of donor resamples fall below the 0.20 gate.
3. "None of the intervals overturned a qualitative conclusion": not supported. Zero-shot fails each gate in about
   20% of donor resamples; internal trust is below 0.80 in 34% of training-donor resamples; dropping TSP2 gives
   internal trust 0.763; dropping TSP25 gives zero-shot branch 0.012. (The trust results here are mostly a panel-size
   effect; see Verification notes, V2.)
4. Internal branch-holdout 0.370 is a plain mean over 6 branches. Weighted by anchors it is 0.153 (below the gate).
   The three large branches give 0.031, 0.154 and −0.075.
5. The fifth pre-registered gate was never computed. Computed now, it passes clearly on external and zero-shot,
   fails on the lung control, and passes on the internal panel only at the smallest possible p, with a margin
   smaller than optimiser noise.
6. The lung control fails all five gates (trust 0.7996 < 0.80). The paper says it fails three of four and that trust
   "clears the gate". It is not an internal-panel construct; it is a frozen-head test of a separate 4-donor panel.
   A first lung control passed all four gates and was replaced; this is not reported.
7. Table cells shown as "n.t." were tested: H38 external 0.883, H38 zero-shot **0.787 (fails)**, H95 zero-shot 0.819.
   H103 zero-shot 0.857 used k = 5 on 18 anchors from 9 donors; at k = 15 on the full 160-anchor panel it is **0.777 (fails)**.
   On the 18 B-lineage anchors at the largest allowed k (8) it is 0.831 (passes). The H103 head was fitted on 45 B-lineage
   anchors only, so the 160-anchor value scores it mostly on cell types it never saw.
8. The zero-shot panel's 12 donors are all external donors.
9. Trustworthiness does not compare the model with biology. External "branch-holdout" holds nothing out.
   "60 genes" are coordinates. 2,352× is from the 2,464-number form (98× from the 154-number form).
   "0.973 → 0.123" is classifier accuracy. The "four strongest axes" were chosen by ablation impact.
10. H65 was not a sweep winner; 18 sweep candidates ran; H95 and H103 passed only under a fallback added mid-sweep.

## 6. What I could not do, and limits

- Nothing at the cell level. Per-cell hidden states were not saved, so cells-within-anchor resampling or re-forming
  anchors needs new forward passes (not allowed here).
- No intervals for internal donor-holdout (0.716) or random holdout (0.834), for the compaction chain, or for the
  factor ablation.
- The internal permutation gate uses an in-sample statistic. A held-out statistic would need about 10 re-fits per
  permutation (about 20,000 fits); not run.
- The two-level and training-donor intervals use 300 replicates. Their end points carry Monte-Carlo error up to
  0.020 (SD). With 11–13 donors and one donor holding 64% of training anchors, any donor-level interval is rough.
- Trustworthiness with resampled donors counts copies in the normalising n. In the frozen-head bootstraps the
  bootstrap mean differs from the observed trust by at most 0.008, so this effect is small.
- The 66.2% scGPT figure was not recomputed. Whether MaxToki's pre-training data contains Tabula Sapiens cells was not checked.
- A batched trainer (`v2_common.train_let_batched`) was tried for speed. It differed from the exact trainer by up to
  0.012 in trust, so it was not used for any reported number.

## 7. Source-data files for the paper's manifold table and figure (item 8)

- `OUT/table_gates.csv` — 118 rows, one per (ordering, panel, metric[, branch]). Columns: value, percentile CI,
  basic CI, bootstrap mean, method, replicates, resampling unit, gate, pass on the point value, share of resamples
  below gate, leave-one-out range and its unit, deployed value, k, anchors, donors, definition, note, source file.
- `OUT/fig_gates.csv` — H65 trust and branch per panel (internal, external, zero-shot, lung), plain and
  anchor-weighted, with evaluation-donor, training-donor and two-level intervals, and a column that says which
  branch quantity each value is (re-fit holdout vs frozen within-branch correlation).
- `OUT/task_data/` — anchor-level data for a controlled experiment: for internal, external, zero-shot, lung_nonhema
  and the first lung control: centroids (n × 12 × 1,232), head inputs before and after standardisation
  (n × 2,464), frozen-head outputs, H65 and null rulers, and an anchors table (donor, tissue, cell type, stage,
  branch, depth, cell counts); plus the stage graph, distance table, standardisation, operators, head weights and
  gate spec. README explains how H65 is defined. 108 MB; sha256 of every file in `manifest.json`.

---

## Summary in plain words

I redid every interval for the H65 manifold with donors as the unit. External trustworthiness stays well above the
0.80 pass mark (0.896, interval 0.864–0.918). The external branch number is weaker than the paper says: its interval
now dips below the pass mark. The zero-shot numbers fail their pass marks in about one resample in five, and one
donor carries the zero-shot branch result. The internal results lean on one donor, TSP2, which holds 64% of the
training data. Without it, internal trust falls to 0.763. (Verification found that random panels of the same size
fall about as far, so this drop is mostly panel size.) The internal branch number is 0.370 only if small branches
count as much as large ones; weighted by size it is 0.153.

The fifth pre-set gate (the permutation test) was never run. It passes clearly on the external and zero-shot
panels and fails on the lung control. On the internal panel it passes only by a hair, and the hair is smaller than
the noise from the optimiser's random seed.

The lung negative control fails all its gates, not three of four. Three table cells marked "not tested" were
tested, and one of them failed. H103's zero-shot value fails when measured the same way as the others.

Several labels in the paper are wrong: trustworthiness never looks at biology; the external "branch-holdout" holds
nothing out; the "60 genes" are not genes; the 2,400× factor starts from the wrong object; "0.973 → 0.123" is a
classifier's accuracy; and the zero-shot donors are the same people as the external donors.

---

## Verification notes (independent check, 2026-10-01)

**Verdict: OK after fixes.** Every number I re-derived matches. The resampling units are right (donors; training
donors with the head re-fitted). Three readings need care, and I added them to the text above:
trustworthiness depends on panel size (V2), the plain-mean branch bootstraps mix different branch sets (V3), and the
fifth gate does not tell H65 apart from the run's own null ruler (V4, V5).

New scripts (own code; they do not import `v2_common`): `RUN/scripts/v2_verify_01_checks.py`,
`v2_verify_02_refits.py`, `v2_verify_03_size_matched_trust.py`, `v2_verify_04_null_ruler_permutation.py`.
Outputs and `run_config_v2_verify_0*.json` (sha256 of every input, seeds) are in `RUN/outputs/v2_verify/`.
CPU only, no model forward pass.

### V1. What was re-derived and matched
- Frozen-head values with my own numpy code: external trust 0.89571 (v2 0.89569; the 2e-5 gap is float64 vs float32
  head output), within-branch 0.34644, within-donor 0.88695, global 0.88212. Zero-shot and lung match to 4e-8.
- Anchor-weighted means: internal 0.1535, external 0.3332, zero-shot 0.3112, lung −0.0229. All match.
- Head re-fits replayed with my own copy of the Phase-5 trainer (seed 42, 1 thread): drop TSP2 0.76267; training-donor
  bootstrap replicates 0 and 2: 0.818173 and 0.795832; internal permutation 1662 (the largest null value): 0.986593.
  All equal to v2 to the last digit. H103: internal 0.8601; zero-shot all 160 anchors at k = 15 0.7765; the 18 kept
  anchors at k = 5 0.8567 and k = 8 0.8308; external all 600 0.8844; external kept 106 at k = 15 0.8772. All equal to v2.
- A second trustworthiness code (a loop over rows) on the first 40 donor-bootstrap replicates: largest difference
  6e-5 (external) and 1e-16 (zero-shot).
- Frozen permutation gate with a different random stream (2,000 permutations, `scipy.stats.spearmanr`): on external
  and zero-shot no null value reaches the observed one (same as v2 with 10,000); null means 0.043 and 0.319 (v2 0.043
  and 0.319). Lung p = 0.743 (v2 0.741). The rebuilt ruler equals the saved one on all three panels.
- Internal permutation gate from the saved rows: 2,000 distinct permutations; my replay of the block shuffle gives the
  same number of changed labels for permutations 0–19; 0 null values ≥ observed; margin 0.00024; 250 of the 2,000 null
  values lie within 0.001 of the observed value.
- Facts: 2,352× and 98×; 7,744 bytes; factor shares 18.25% (top 4 by impact) and 8.95% (top 4 by size); stage graph
  34 nodes, 35 edges, 62 cell-type entries, 12 branches; all 12 branches occur in the internal panel, so the branch probe
  has 12 classes (chance 0.083); zero-shot donors are a subset of external donors (only TSP19 missing); 41 of 45
  zero-shot tissues are in external; no internal–external donor overlap; fallback timeline 17:37 → 18:19 (file times and
  `outputs/phase3a_revaluate.log`); the sweep-2 fallback does not check the null (`scripts/phase3a_sweep2_ordinal.py:233-239`).
- The sha256 of all 56 outputs and 14 scripts in `run_config.json`, and of every file in `task_data/manifest.json`,
  match the files on disk. No original run file changed since 2026-09-30 (checked by modification time).

### V2. Trustworthiness depends on panel size (larger issue; numbers unchanged, reading changed)
Deployed head, no re-fit, random anchor subsets drawn without replacement (200 draws each; mean of the draws):

| Panel (full size, full value) | 50 anchors | 103 anchors | 160 anchors | 290 anchors |
|---|---:|---:|---:|---:|
| internal, in-sample (290, 0.811) | 0.750 (98% below 0.80) | 0.782 | 0.786 | — |
| external (600, 0.896) | 0.804 (44% below) | 0.834 | 0.847 | 0.876 |
| zero-shot (160, 0.827) | 0.776 (83% below) | 0.804 | — | — |
| lung control (50, 0.7996) | 0.7996 | — | — | — |

With the head re-fitted on each subset: 12 random 103-anchor subsets of the internal panel give 0.774 on average
(SD 0.023; 10 of 12 below 0.80). Six 103-anchor subsets of TSP2's own anchors give 0.785. Dropping TSP2 gives 0.763.
In the training-donor bootstrap, trust correlates 0.72 with the number of training rows. Replicates without TSP2
average 0.773 and make up 86% of the replicates below 0.80; replicates with 2 or more copies of TSP2 average 0.84–0.85.

Copies of anchors also move the metric. Trust is higher at smaller k (internal 0.851 at k = 5 vs 0.811 at k = 15).
A copied donor fills more of the 15 neighbour slots, which acts like a smaller k. In the zero-shot donor bootstrap,
trust correlates +0.38 with the share of copied rows. Dropping copies instead of masking them changes the share
below 0.80 from 18% to 39% (the lower end barely moves: 0.749 vs 0.753). External: 0.05% either way.

What this means:
- "34% of training-donor resamples below 0.80" and "dropping TSP2 gives 0.763" mostly show that trust falls when a
  panel has fewer anchors. They do not show that the result depends on something specific to donor TSP2.
- Trust values of panels with different sizes cannot be compared directly. The external 0.896 would be about 0.876 at
  290 anchors and 0.847 at 160. At 50 anchors the blood panels score 0.750–0.804, so the lung control's 0.7996 is
  not low for its size. Its "failure" by 0.0004 says nothing about biology.
- Suggestion: report trust next to a size-matched reference, or compare panels at the same number of anchors.

### V3. The plain-mean branch bootstraps average over different branch sets (disclosure)
Internal: 193 of 300 replicates score 6 branches; the 107 without TSP2 score 2 or 3 (numbers added to section 3.3).
External frozen bootstrap: 6 branches in 59% of replicates, 3–5 in the rest. Zero-shot: 6 branches in 28%, 2–5 in the
rest. So the plain-mean intervals mix a mean of 6 values with means of 2–5 values. The anchor-weighted mean is less
affected by this.

### V4. Internal fifth gate: the null ruler would probably pass too
Besides the seed fragility already reported, the run's own null ruler (stage labels shuffled within branch), fitted the
same way, scores 0.98661 in-sample. That is above the largest of the 2,000 H65 permutation values (0.98659). So the
internal gate would very likely not tell H65 apart from its null. Not run: a proper test needs about 2,000 more re-fits.

### V5. Frozen fifth gate does not tell H65 apart from the null ruler
Scored against the run's null ruler with the frozen head: external 0.852, zero-shot 0.874 (H65: 0.882, 0.899). In 2,000
blocked permutations, no null value reaches either (p = 0.0010, the smallest possible), so the null ruler passes too.
The gate shows that the head's distances follow branch-level structure. It does not test the order inside a branch.

### V6. H103 zero-shot
The 0.777 at k = 15 on all 160 anchors is computed correctly, but the H103 head was fitted on 45 B-lineage anchors, so
this value scores it mostly on cell types it never saw. On its 18 B-lineage anchors the largest allowed k (8) gives
0.831, which passes. Both values are now in section 5, item 7.

### V7. Task-data folder
Complete: all five panels, centroids, head inputs before and after scaling, frozen-head outputs, rulers, an anchors table
(donor, tissue, cell type, stage, branch, depth, cell counts), shared files and a README. 108 MB. sha256 values match.
README facts (block layers, 34 nodes, 35 edges, 62 entries, ruler 0–9 and 0–10, 12 branches) check out.
One point for whoever designs the controlled experiment: the README and `lung_control/anchors.csv` (column `note`) say
the first lung control "passed all four gates". If the experiment must not see outcomes, remove that text. I did not change it.

### Edits I made to this report (all small; no CSV, JSON or v2 script was changed)
1. Section 1, plain words: "it never comes near 0.80" → "only 1 of 2,000 resamples fell below 0.80 (to 0.799)"
   (`outputs/v2_intervals/frozen_boot/external.jsonl`, minimum 0.7989).
2. Section 1, section 5 item 3, and the closing summary: added the panel-size caveat for internal trust (V2).
3. Section 3.2: "within 0.005" → "within 0.006" (largest gap 0.0052, external branch upper end).
4. Section 3.3: added the branch-set note and the six-branch-only intervals (V3).
5. Section 3.6: added the null-ruler caveat for the frozen gate (V5).
6. Section 5, item 7: added the H103 value at k = 8 and what the 160-anchor value scores (V6).

### Not checked
The 66.2% source-paper figure; the transformers line numbers for the final RMSNorm; intervals v2 did not compute
(internal donor and random holdout, compaction, factor ablation). I replayed 2 of the 300 training-donor bootstrap
replicates and 1 of the 2,000 internal permutations, not all of them.

### Verification summary in plain words
The numbers in this report are right. I re-computed them with separate code and got the same values. Donors are
the resampling unit, as asked. Three things change how the numbers should be read. First, trustworthiness goes down
when a panel has fewer anchors. So the internal "drop TSP2 gives 0.763" and "one in three resamples below 0.80"
mostly show panel size, and the lung control's 0.7996 is normal for a 50-anchor panel. Second, some branch intervals
average over 6 branches in some resamples and only 2–5 in others. Third, the fifth gate is passed just as well by the
run's own shuffled null ruler, so it cannot be used as evidence for the H65 order.
