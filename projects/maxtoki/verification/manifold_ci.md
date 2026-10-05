# manifold_ci — how the paper's intervals were computed, and a donor-level check

Scope: Reviewer 2's point that the manifold "external" intervals come from 200 repeats of 80% anchor
subsampling without replacement, which is a sensitivity analysis, not uncertainty across donors.
Everything below was read from files or re-run from saved artefacts. I mark inference as **(inferred)**.
The repository was not changed. All new files are under
`<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci/`.

Short names used below:
- RUN = `<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M`
- PAPER = `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`
- SCR = the scratchpad folder above

---

## 1. Bottom line (plain words)

1. The reviewer is right. The external interval [0.884, 0.899] (trust) and [0.270, 0.480] (branch)
   come from 200 random 80% subsets of the 600 **anchors**, drawn **without replacement**
   (`RUN/scripts/audit_a3_bootstrap_h65_gates.py:28-29,126-130`). Donors were never resampled.
   The paper's claim of a "non-parametric bootstrap (1,000 iterations) computed at the unit of
   inference" (PAPER:808-809, 1568-1570) does not match this for any of the three intervals it prints.
2. Taking 80% without replacement makes the spread too small by itself. Theory says the spread of an
   80% subset is about half the true sampling spread (variance factor 1/0.8 − 1 = 0.25). I measured it:
   the published scheme gives SD 0.0041 for trust; a normal anchor bootstrap gives 0.0074; a donor
   bootstrap gives 0.0134 (3.3× the published SD).
3. The saved artefacts **do** allow a proper donor-level analysis with no model forward pass. Each anchor
   has a donor ID, a saved centroid, and a saved ruler row. The frozen head can be re-fit exactly
   (weights identical to the saved `.pt`, max difference 0). I reproduced every published number
   exactly, then ran donor-level bootstraps and leave-one-donor-out.
4. Donor-level results, external panel (13 donors, frozen head):
   - trust 0.896, donor-bootstrap 95% CI **[0.865, 0.918]** (published [0.884, 0.899]). It never falls below the 0.80 gate.
   - branch 0.346, donor-bootstrap 95% CI **[0.142, 0.540]** (published [0.270, 0.480]). The lower end is now below the 0.20 gate; 4.9% of donor resamples fall below 0.20.
   - Adding the uncertainty from which 11 training donors fitted the head (two-level bootstrap, head re-fit each
     time; only 37 replicates, so rough): trust SD 0.014, CI ≈ [0.87, 0.93]; branch SD 0.10, CI ≈ [0.15, 0.55].
     Trust stays above 0.80 in every draw. Branch fell below 0.20 in 1 of 37 draws.
5. Zero-shot panel (12 donors) is weaker than it looks: donor-bootstrap CI trust **[0.751, 0.871]**
   (20% of resamples below 0.80) and branch **[−0.030, 0.471]** (20% below 0.20). Dropping one donor
   (TSP25) moves branch from 0.317 to **0.012**.
6. The zero-shot panel is **not** a separate donor cohort. Its 12 donors are all among the 13 external
   donors (by construction, `RUN/scripts/phase1a_subsample_and_tokenize.py:239-247`). Its cells and anchors are
   disjoint from the external panel, and 41 of its 45 tissues also appear in the external panel. The paper
   presents it as a separate cohort ("12 donors from different tissues", PAPER:1193, 1225). The run summary
   ("disjoint from both internal and external", summary line 38) and the audit ("12 disjoint donors",
   "independent point estimates") say more. Both are wrong about the donors.
7. The internal branch-holdout 0.370 (vs null −0.001) is the plain mean of 6 per-branch values. The three largest
   branches (T 101, B 45, monocyte 34 anchors) give 0.031, 0.154 and −0.075. The mean is carried by three small
   branches (granulocyte 25, erythroid 7, stem 4 anchors). A t-interval over branches is [−0.03, 0.77] (§5.5).
8. Several manifold numbers in the paper are correct as numbers but mislabelled (details in §7):
   "60 genes" are not genes; "0.973 → 0.123" is branch balanced accuracy of a probe, not branch-holdout;
   the 2,400× factor is from the 2,464-dim pooled operator, not the 154-dim one; trustworthiness does not
   compare the model to biology; the lung control trust (0.7996) is below the 0.80 gate, not above it;
   and several "n.t." table cells were in fact tested (one of them failed).

---

## 2. What the units are

### 2.1 Anchor (verified)
An anchor is one group of cells that share **donor × tissue × cell type × stage label**, with at least 5 cells
(`RUN/scripts/phase1a_subsample_and_tokenize.py:65-79`, grouping at line 74). Its centroid is the mean over its
cells of the per-cell hidden state, where each cell's hidden state is the mean over gene-token positions at each
of 12 layer states (`RUN/scripts/phase1bc_hidden_states_and_centroids.py:3-8,174,200`). Saved centroids have shape
(n_anchors, 12, 1232).
Panels keep the **largest** groups first (`phase1a...py:94-106`), capped at 50 cells per anchor (line 49), then
further cut for memory (`phase1bc...py:50-54`): external 20 cells per anchor, zero-shot 32.
The ruler `d_target` is the shortest-path distance between the two anchors' stages on the 34-node stage DAG
(`phase1bc...py:80-97`; values 0–9, no disconnected sentinel 99 in any H65 panel).

### 2.2 Panel sizes (verified from `RUN/artifacts/anchors/anchor_meta_*.csv`)

| Panel | Anchors | Donors | Cells used | Cells per anchor | Anchors per donor |
|---|---:|---:|---:|---|---|
| internal (training) | 290 | 11 | 11,804 | 13–50 (median 50) | TSP2 **187**, TSP4 32, TSP20 17, TSP17 14, TSP15 13, TSP9 7, TSP5 7, TSP3 5, TSP28 4, TSP30 2, TSP12 2 |
| external | 600 | 13 | 12,000 (of 30,000 tokenised) | 20 | TSP25 129, TSP14 125, TSP21 99, TSP27 82, TSP7 61, TSP1 29, TSP10 26, TSP6 15, TSP19 11, TSP26 10, TSP11 8, TSP8 4, TSP13 **1** |
| zero-shot | 160 | 12 | 5,120 (of 7,796) | 32 | TSP14 44, TSP21 34, TSP25 29, TSP27 17, TSP7 11, TSP13 5, TSP10 4, TSP6 4, TSP1 3, TSP26 3, TSP8 3, TSP11 3 |
| lung_nonhema (neg. control) | 50 | 4 | 1,500 | 30 | TSP25 15, TSP14 12, TSP1 12, TSP2 11 |

Checks I ran:
- Internal ∩ external donors = none. Internal ∩ external cells = none.
- **Zero-shot donors ⊂ external donors** (all 12; only TSP19 is missing). Zero-shot ∩ external anchors = 0; cells = 0.
  13 zero-shot anchors share donor, tissue and cell type with an external anchor (they differ only in stage label).
  Zero-shot groups are the next-largest groups (42–75 cells) after the external ones (75–15,803 cells).
- Zero-shot tissues: 45, of which 41 are also in the external panel. The paper's "12 donors spanning different
  tissues" (PAPER:1193, 1225) is not supported.
- The lung control includes TSP2, the donor that supplies 64% of the training anchors.
- One donor dominates training: TSP2 holds 187 of 290 internal anchors (64%). Five donors hold 496 of 600 external anchors (83%).

### 2.3 What each "gate" computes (verified)
- **Trustworthiness** = `sklearn.manifold.trustworthiness(features, z, n_neighbors=15)`
  (`RUN/scripts/phase5_let_anchor.py:236`; `phase7_external_validation.py:94`). It compares the 2,464-dim
  pooled-drift features (model-internal) with the 10-dim head output z. **The biological ruler is not an
  input.** So it asks "does the 10-D projection keep the model's own 15 nearest anchors?", not "are model
  neighbours biological neighbours?" as PAPER:1171-1173 and 1215-1216 say. This explains why the null
  (0.799) and the lung control (0.7996) score almost the same as H65 (0.811).
- **Internal random holdout**: 10 random 80/20 anchor splits with head re-fit (`phase5...py:240`, n_iters=10; the comment says 20).
- **Internal donor / branch holdout**: leave one group out, re-fit head, Spearman(latent arc-cos distance, ruler distance) on
  pairs inside the held-out group; mean over groups with ≥3 anchors (`phase5...py:167-199`).
- **External / zero-shot / lung "holdouts"**: the head is frozen. Nothing is held out. "Branch holdout" is the mean,
  over branches with ≥3 anchors, of the within-branch Spearman (`phase7...py:132-150`). Branches whose anchors all
  share one stage (macrophage, NK) give a constant ruler and are dropped, so external branch = mean of **6** branches,
  zero-shot 6, lung 5. "Donor holdout" is the mean within-donor Spearman (12 donors scored; TSP13 has 1 anchor).
  So the external 0.346 and the internal 0.370 are **different quantities** shown side by side in Table 4 and Fig 7.
- The fifth pre-registered gate, blocked-permutation p ≤ 0.001 with ≥2,000 permutations
  (`RUN/reports/quality_gates_spec.json`, promotion rule "must pass ALL five gates"), is **not computed by any script**
  in `RUN/scripts/` (grep for permutation/blocked finds nothing).

---

## 3. How each interval in the paper was computed

| Paper number | Where in paper | Script | Unit resampled | Replacement | Reps | Fraction | Interval type |
|---|---|---|---|---|---|---|---|
| External trust 0.896 [0.884, 0.899] | PAPER:1191-1192 | `RUN/scripts/audit_a3_bootstrap_h65_gates.py` → `RUN/reports/external_validation_external_bootstrap.json` | anchors (600) | **no** | 200 | 80% (480) | percentile 2.5/97.5; head frozen; seed 42 |
| External branch 0.346 [0.270, 0.480] | PAPER:1572-1573 | same | anchors | no | 200 | 80% | percentile |
| Cross-model Pearson 0.382 [0.380, 0.384] | PAPER:1407-1408 | `runs/spectral-geometry-217M/scripts/audit_a3_bootstrap_cross_model_pearson.py` → `runs/spectral-geometry-217M/outputs/phase9b/bootstrap_pearson.json` | genes (1,500 HVGs; all pairs among kept genes) | **no** (docstring line 5 says "with replacement"; code line 87 is `replace=False`) | 1,000 | 80% (1,200) | percentile |
| CRISPRi 53.42% [52.60%, 54.16%] | PAPER:1099, 1550 | `runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py:207-213` → `outputs/groupkfold_crispri/summary.json` | silenced (source) genes, 248 per-gene accuracies | **yes** | 1,000 | 100% | percentile of the mean |
| CKA 1.000 → 0.991 | PAPER:1464 | — | — | — | — | — | **no interval** |
| Internal, zero-shot, lung, H38/H95/H103, compaction, 18.3% | Table 4, text | — | — | — | — | — | **no interval** |

Notes:
- The audit summary `projects/maxtoki/audits/bootstrap_cis_summary.json` (from `audits/audit_a3_bootstrap_cis.py`) still
  says the manifold and cross-model bootstraps were "SKIPPED — not feasible". Both were done later the same day by
  run-level scripts (report file time May 7 19:14). The audit's "fallback evidence" also calls the zero-shot panel
  "12 disjoint donors" and "independent" — false (§2.2).
- The audit itself prescribed "Resample at the inference unit (TF / donor / cell-sample)"
  (`audits/audit-20260507.md`, action A3). The manifold script resampled anchors, not donors.
- The CRISPRi "five-fold cross-validation so that no gene appears in both training and test" (PAPER:1097-1099) is a
  mis-description: nothing is trained. The script only splits the evaluation pairs into 5 source-gene groups
  (fold means 0.519–0.546, mean 0.5347). The 53.42% is the plain mean of 248 per-gene accuracies, and the CI is a
  bootstrap over genes. The per-gene bootstrap itself is a reasonable unit.

---

## 4. Reproduction (all exact)

`SCR/01_reproduce.py` re-fits the Phase-5 head from saved centroids (same seed and settings). The re-fit weights are
identical to `RUN/artifacts/heads/let_anchor_internal.pt` (max |diff| = 0 for W, b, log β). Output `SCR/01_reproduce.json`:

| Panel | trust (mine / published) | branch (mine / published) | donor (mine / published) |
|---|---|---|---|
| external | 0.895688 / 0.895688 | 0.346443 / 0.346443 | 0.886953 / 0.886953 |
| zero-shot | 0.826724 / 0.826724 | 0.316830 / 0.316830 | 0.870309 / 0.870309 |
| lung_nonhema | 0.799556 / 0.799556 | −0.146871 / −0.146871 | 0.013397 / 0.013397 |

`SCR/02_bootstrap.py` scheme (a) re-runs the published 80% subsample with the same RNG call order and gives
trust [0.88392, 0.89908] and branch [0.26976, 0.47979] — identical to the published JSON.
`SCR/04_crossmodel.py` reproduces the cross-model [0.37974, 0.38434] exactly.

For resampling with replacement I wrote a trustworthiness function (`SCR/common.py:trust_masked`) that treats copies of
the same anchor like the point itself (they are never counted as neighbours). With no copies it matches sklearn to 2e-7.
The within-branch Spearman likewise drops pairs that are two copies of one anchor.

---

## 5. Donor-level prototype results

### 5.1 External panel (frozen head; `SCR/02_bootstrap.json`)

| Scheme | trust 95% CI | trust SD | branch 95% CI | branch SD |
|---|---|---|---|---|
| (a) published: 80% anchors, no replacement, 200 reps | [0.884, 0.899] | 0.0041 | [0.270, 0.480] | 0.059 |
| (b) anchor bootstrap with replacement, 1,000 reps | [0.879, 0.908] | 0.0074 | [0.182, 0.543] | 0.089 |
| (c) **donor bootstrap** with replacement, 2,000 reps | **[0.865, 0.918]** | 0.0134 | **[0.142, 0.540]** | 0.091 |
| (d) leave-one-donor-out jackknife, t(12) interval | [0.867, 0.924] (SE 0.0130) | — | [0.152, 0.541] (SE 0.089) | — |

- Donor bootstrap: resample size ranged 167–1,151 anchors; 3–6 branches scored. 0% of resamples had trust < 0.80.
  4.9% had branch < 0.20.
- Leave-one-donor-out values: trust 0.884 (drop TSP25) to 0.899; branch 0.305 (drop TSP11) to 0.408 (drop TSP27).
- Ratio check: (a) SD / (b) SD = 0.55 for trust and 0.66 for branch. Theory for an 80% subset predicts about 0.5.

### 5.2 Zero-shot panel (frozen head)

| Scheme | trust 95% CI | branch 95% CI |
|---|---|---|
| (b) anchor bootstrap, 1,000 reps | [0.797, 0.862] | [0.027, 0.475] |
| (c) **donor bootstrap**, 2,000 reps | **[0.751, 0.871]** (19.7% < 0.80) | **[−0.030, 0.471]** (20.2% < 0.20) |
| (d) jackknife, t(11) | [0.754, 0.899] (SE 0.033) | [−0.378, 1.012] (SE 0.316) |

Leave-one-donor-out: dropping **TSP25** gives branch **0.012** (trust 0.811); dropping TSP14 gives 0.428; the rest
0.295–0.373. Dropping TSP21 gives trust 0.7995 (below the gate). So the zero-shot branch value rests on one donor.

### 5.3 Which training donors fit the head (`SCR/03_lotdo.json`)
Leave-one-**training**-donor-out (11 re-fits, evaluated on the full external / zero-shot panels):
- External trust 0.894–0.910, branch 0.323–0.442. Dropping TSP2 (training falls from 290 to 103 anchors) gives the
  highest external branch (0.442) and trust (0.910).
- Zero-shot trust 0.826–0.836, branch 0.240 (drop TSP4) to 0.351 (drop TSP15).

### 5.4 Two-level donor bootstrap (training and evaluation donors both resampled)
Each replicate: resample the 11 internal donors with replacement, re-fit the head (same seed and settings as
Phase 5, features re-standardised on the resampled set), then resample the external (and zero-shot) donors with
replacement. `SCR/03_training_donors.py twolevel`, summary `SCR/06_twolevel_summary.json`.
**Only 37 replicates finished** in two 590-s runs (the machine load average was ~100–127 from other jobs).
So these are rough. With 37 draws the 2.5% / 97.5% percentiles are close to the 1st and 37th values; the SD is the
more stable number. Training sets ranged 78–788 anchors; 14 of 37 replicates had no TSP2.

| Quantity | Training donors only (SD) | Both levels: 95% percentile CI (SD) | Normal CI from SD, centred on observed | Share below gate |
|---|---|---|---|---|
| external trust (0.896) | SD 0.0085 | [0.880, 0.928] (0.0138) | [0.869, 0.923] | 0 / 37 below 0.80 |
| external branch (0.346) | SD 0.040 | [0.220, 0.543] (0.102) | [0.147, 0.546] | 1 / 37 below 0.20 |
| zero-shot trust (0.827) | SD 0.0060 | [0.755, 0.865] (0.032) | [0.764, 0.890] | 10 / 37 below 0.80 |
| zero-shot branch (0.317) | SD 0.066 | [0.142, 0.456] (0.097) | [0.127, 0.506] | 4 / 37 below 0.20 |

Reading: for the external branch value, the evaluation donors add more spread (SD 0.091) than the training donors
(SD 0.040); combined ≈ 0.10, which matches the two-level SD. The mean of the two-level draws (0.403) is above the
observed 0.346, so the bootstrap distribution is not centred on the estimate; the head fitted on all 11 donors
(64% TSP2) sits on the low side **(inferred: TSP2's weight pulls branch down; dropping TSP2 gave 0.442)**.
A proper run would use ≥500 replicates (≈1–3 CPU-hours at normal load).

### 5.5 Internal panel branch-holdout 0.370 vs null −0.001
`SCR/05_internal_branch.py` re-fits one head per held-out branch, exactly as `phase5_let_anchor.py:167-199`.
It reproduces 0.370404 (H65) and −0.001012 (null) exactly. Results in `SCR/05_internal_branch_*.json`.

The internal number is an **unweighted mean over 6 branches**. Branches whose anchors all have one stage
(macrophage, NK, lymphoid, dendritic) give no value and are dropped.

| Held-out branch | anchors | stages | H65 rho | null rho |
|---|---:|---:|---:|---:|
| T_lineage | 101 | 6 | 0.031 | −0.072 |
| B_lineage | 45 | 3 | 0.154 | 0.015 |
| monocyte | 34 | 2 | −0.075 | 0.195 |
| granulocyte | 25 | 3 | 0.708 | −0.026 |
| erythroid | 7 | 2 | 0.783 | 0.296 |
| stem | 4 | 2 | 0.621 | −0.414 |
| **mean** | | | **0.370** | **−0.001** |

- The three largest branches (180 of the 216 scored anchors) give 0.031, 0.154 and −0.075 — near zero.
  The 0.370 comes mostly from three small branches (granulocyte 25, erythroid 7, stem 4 anchors; stem has only 6 pairs).
- A t-interval over the 6 branch values is [−0.025, 0.765] for H65 and [−0.259, 0.257] for the null.
  With 6 values this is crude, but it shows the "decisive" gap (PAPER:1185-1186) is not decisive at the branch level.
- Without TSP2 (103 training anchors left): H65 0.393 from only 3 scored branches (T −0.059, B 0.392, granulocyte 0.845);
  null 0.010 from 4 branches.
- For comparison, the frozen-head external per-branch values (`SCR/07_external_per_branch.txt`): T 0.289 (219 anchors),
  B 0.375, granulocyte 0.601, erythroid 0.579, monocyte 0.075, stem 0.159 → mean 0.346. These are higher for T and B
  because in the external test the head was trained on all branches; no branch is held out.
- A full donor bootstrap of the internal branch-holdout was not run. It needs about 10 head re-fits per replicate
  (≈1–4 minutes per replicate at the machine load seen today).

### 5.6 Cross-model Pearson 0.382 (`SCR/04_crossmodel.json`)
Gene bootstrap with replacement (1,000 reps, copy-pairs dropped): [0.3775, 0.3866], SD 0.0024.
Published 80% subsample: [0.3797, 0.3843], SD 0.0012. So the published interval is about half as wide as it should be.
The conclusion does not change. Genes are also not independent units (they share pathways), so even this is optimistic **(inferred)**.

### 5.7 What a donor-level analysis can and cannot do with saved files
- Can (no model run): donor bootstrap / leave-one-donor-out for trust, random, donor and branch on external, zero-shot
  and lung panels; head re-fits on resampled training donors (each fit ≈5–25 s on CPU, depending on machine load);
  internal donor/branch holdouts with donor resampling (slow: one re-fit per held-out group per replicate).
- Cannot: anything at the cell level. Per-cell hidden states were not saved (only per-anchor centroids;
  the run summary says the same). A cell-within-anchor bootstrap, or re-forming anchors, needs new forward passes.
- Cannot: the compaction chain (Phase 10) and factor ablation (Phase 11) intervals without re-running those scripts
  in a loop; possible from saved operators and centroids, but not done here.
- Small-donor limit: with 13 (external) and 12 (zero-shot) donors, and 5 donors holding most anchors, any donor-level
  interval is itself rough. A percentile bootstrap with G=12–13 clusters tends to be a little too narrow **(inferred)**.

---

## 6. What to tell the reviewer (suggested wording, plain)
- Say the published interval resampled anchors (80%, no replacement, 200 repeats) and is a stability check, not a
  donor-level CI. Withdraw "computed at the unit of inference" for this number.
- Report the donor bootstrap: external trust 0.896 [0.865, 0.918]; external branch 0.346 [0.142, 0.540].
- Report the two-level result from §5.4 as the honest interval that includes the 11 training donors.
- Say the zero-shot panel shares donors with the external panel and is not an independent replication.
  Report its donor-level interval (branch [−0.03, 0.47]) and that one donor carries it.
- Say the branch gate passes on the point estimate but the donor-level lower bound is below 0.20.
  This softens the paper's line "None of the resulting intervals overturned a qualitative conclusion"
  (PAPER:1570-1571) and the summary's "all four gate thresholds are crossed by every bootstrap CI 2.5% lower bound".
- Fix the definition of trustworthiness, and label the external "branch-holdout" as a within-branch correlation
  under a frozen head (no branch is held out).

---

## 7. Check of manifold numbers in the paper against sources

Sources: `RUN/reports/quality_gates_let_anchor.json` (QG), `external_validation_external.json` (EXT),
`zeroshot_transfer_anchor_head.json` (ZS), `external_validation_lung_nonhema.json` (LUNG), `compaction_chain.json` (CC),
`factor_ablation.json` (FA), `external_validation_external_bootstrap.json` (BOOT), plus the H-files named below.

| Paper claim (line) | Source value | Verdict |
|---|---|---|
| trust 0.811 vs null 0.799 (1173) | QG 0.810936 / 0.798500 | correct |
| branch 0.370 vs −0.001 (1177) | QG 0.370404 / −0.001012 | correct |
| random 0.834, donor 0.716 (1178-1179) | QG 0.834108 / 0.715888 | correct |
| null random 0.799, donor 0.707 (1184-1185) | QG 0.798593 / 0.707129 | correct |
| "trustworthiness … do cells that neighbour each other inside the model also neighbour each other biologically" (1171-1173, 1215-1216) | code compares 2,464-D features with 10-D head output; ruler not used | **wrong definition** |
| "cells are shuffled within their own developmental branch" (1174) | anchor stage labels permuted within branch (`phase1bc...py:215-230`) | loose wording (anchors, labels) |
| "Internal is the full MaxToki cell panel" (1223) | 290 anchors (largest groups) from 11 donors, 11,804 cells; TSP2 = 187 anchors | **vague / misleading**; units are anchors, not cells |
| 13 donors sharing no cells (1190-1191) | EXT n_donors 13; zero donor/cell overlap with internal | correct |
| trust 0.896 [0.884, 0.899], branch 0.346 (1191-1192) | EXT 0.895688, 0.346443; BOOT [0.88392, 0.89908] | numbers correct; **CI is anchor subsampling, not donor-level** |
| branch CI [0.270, 0.480] (1572-1573) | BOOT [0.26976, 0.47979] | numbers correct; same caveat |
| 12 donors "from different tissues", 0.827 / 0.317 (1193-1194, 1225) | ZS 0.826724 / 0.316830, 12 donors | numbers correct; **donors are a subset of the external 13; 41/45 tissues shared** |
| lung control: 50 anchors, random stage labels, random −0.016, donor 0.013, branch −0.147 (1195-1197) | LUNG 50 anchors, −0.016420 / 0.013397 / −0.146871; stages drawn uniformly (`phase1a_lung_nonhema_panel.py:91`) | correct |
| lung "fail three of the four gates" (1196); trust "clears the gate even on the lung control" (1297-1298); "passes" (1277-1278) | LUNG trust 0.799556 < 0.80; code uses `trust >= 0.8` | **wrong**: fails all four by the code's rule (trust by 0.0004) |
| lung trust shown as 0.800 in Internal column; "negative control is an internal-panel construct only" (1257-1258, 1266) | frozen-head Phase-7 evaluation of a separate 4-donor panel (incl. training donor TSP2) | **mislabelled** |
| 16 features × 60 genes, 7.7 KB (1165-1166) | `phase10_compaction_chain.py:148-165`: 16 SVD factors, 60 of 154 head dims and 60 of 1,232 residual dims; 7,744 bytes = values only | 7.7 KB correct for values; **"genes" wrong**; indices, the LET head and the 217M model are not counted |
| 154-dim dense form → 7.7 KB = 2,400× (1197-1199, 230) | CC: 18.213888 MB (2,464-dim pooled, 3 × 1232² operators) / 0.007744 MB = 2,352×; 154-dim single head 0.758912 MB → 98× | **factor attributed to the wrong starting point**; 2,352 rounds to ~2,400 |
| "costs no measurable quality" (1199) | CC internal-panel only; hard-sparse trust 0.909, branch 0.433 vs single-head 0.882 / 0.467; no CI; same panel used to pick L10H6 | unsupported as stated (no interval, no external test, selection on same panel) |
| 18.3% vs 66.2% (1286-1288) | FA top4 0.182529; 66.2% is the source paper's value (pipeline spec line 26/424) | correct |
| "four strongest axes of variation" (1285-1286) | FA core_factors [1, 4, 11, 9] = top-4 by ablation impact, not top-4 singular values | **wrong description** |
| "branch-holdout collapses from 0.973 to 0.123" (1289-1290) | FA branch_balanced_acc 0.97284 → 0.12340 (probe accuracy) | numbers correct; **it is balanced accuracy, not branch-holdout** |
| H38 internal 0.814 (1251) | `h38_lite_quality_gates.json` 0.813508 (LITE proxy, 279 anchors) | correct |
| H38 external / zero-shot "n.t." (1251) | `external_validation_H38_lite.json` trust 0.883 (4/4 pass); `zeroshot_H38_lite.json` trust **0.787, FAIL** | **tested; zero-shot failure not reported** |
| H95 internal 0.800 (1253) | `hypothesis_registry_3gate.json` 0.800483 (passes by 0.0005; verdict re-classified from INCONCLUSIVE to POSITIVE_3GATE_FALLBACK) | correct, borderline |
| H95 external 0.888 (1253) | `external_validation_h95.json` 0.888464 (441 anchors) | correct |
| H95 zero-shot "n.t." (1253) | `zeroshot_H95.json` 0.818756, category holdout −0.004 (3-gate pass) | **tested but not reported** |
| H103 0.860 / 0.877 / 0.857 (1255) | sweep2 0.860135 (45 anchors, 2 distinct depths); external 0.877207 (106 anchors, 12 donors, depth gate NaN); zero-shot 0.856667 with **k=5** on **18 anchors / 9 donors** | numbers correct; zero-shot uses a different k and tiny n (summary says 7 donors; JSON says 9) |
| "catalogue of 141 candidate biological orderings" (1211) | no 141-item catalogue in RUN; 141 is the topology pipeline; the run tested H65, H38 and 18 sweep candidates (H92–H97, H101–H112) | **unsupported** |
| "A sweep over 12 candidate orderings … H65 is the one that passed all four" (999-1002) | H65 was fixed in advance (source paper), not a sweep winner; sweeps 1–3 had 18 candidates; H95/H103 passed under a 3-gate fallback | **inaccurate** |
| "H115 and H118 were retired after two negatives each" (1615-1617) | no H115 or H118 record anywhere in RUN, `summaries/` or `audits/`; the names appear only in the two paper sources | **unsupported by run artefacts** |
| Fig 7 data (paper-biosystems/main.tex:1204-1205) | 0.811/0.896/0.827/0.800 and 0.370/0.346/0.317/−0.147 | matches sources; no error bars; mixes refit-based and frozen-head branch values |

Not verified: whether MaxToki's pre-training corpus contains Tabula Sapiens cells. If it does, "donors the model was
never shown" (PAPER:997-998) is true only for the small head, not for the foundation model **(open question)**.

---

## 8. Files
- Scripts: `SCR/common.py`, `SCR/01_reproduce.py`, `SCR/02_bootstrap.py`, `SCR/03_training_donors.py`,
  `SCR/04_crossmodel.py`, `SCR/05_internal_branch.py`, `SCR/06_summarize_twolevel.py`
- Results: `SCR/01_reproduce.json`, `SCR/02_bootstrap.json`, `SCR/03_lotdo.json`, `SCR/04_crossmodel.json`,
  `SCR/06_twolevel_summary.json`, `SCR/05_internal_branch_*.json`, `SCR/07_external_per_branch.txt`,
  raw donor-bootstrap draws `SCR/donor_boot_*.npy`,
  two-level logs `SCR/tl11.log`, `SCR/tl12.log`.
