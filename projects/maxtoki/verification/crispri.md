# CRISPRi directional accuracy: re-analysis on the identical corrected records

KEY = crispri. Reviewer 2's point on the 53.42% / 53.48% directional-accuracy claim.

## 1. Bottom line (numbers first)

All numbers below are computed on the **same 1,503,408 corrected records** the paper reports
(`per_pair.parquet`). I also rebuilt the original 1,458,016 records exactly.

| Predictor | Accuracy | Balanced accuracy | MCC |
|---|---:|---:|---:|
| Model (the paper's corrected metric) | **0.5348** | **0.4983** | **-0.0055** |
| Always predict "decrease" | **0.5461** | 0.5000 | n/a |
| Always predict "increase" | 0.4539 | 0.5000 | n/a |
| Chance given the two class rates (pred 89.6% dec, obs 54.6% dec) | 0.5364 | 0.5 | 0 |
| Per-target majority, learned on the other source folds | **0.5915** | 0.5769 | 0.162 |
| Per-source majority, learned on the source's other targets | 0.5643 | 0.5414 | 0.095 |

- The model predicts "decrease" for **89.6%** of pairs. **54.6%** of pairs are observed decreases.
- So the model is **1.13 points worse** than the always-decrease rule.
  Grouped bootstrap over silenced genes (2,000 reps): **-1.13 pts, 95% CI [-1.61, -0.72]**. 0 of 2,000 reps above zero.
- Balanced accuracy is **49.83%**. Balanced accuracy minus 0.5: **-0.17 pts, CI [-0.51, +0.14]**.
  That means no detectable skill once the class imbalance is removed.
- The 53.48% is above 50% only because the model says "decrease" most of the time and decreases are the
  majority class. It is even slightly below what you get from those two class rates alone (53.64%).
- The paper's sentence "the interval excludes 50%, so the effect is real" does not hold. The right
  comparison is the always-decrease rule, and the model loses to it.
- The reviewer is right on every point. The always-decrease rule gives 54.61% on the original records
  (verified exactly) and 54.61% on the corrected records too (0.546059 vs 0.546056).

## 2. Where everything is (verified by reading the files)

Run folder: `<REPO_ROOT>/projects/maxtoki/runs/circuit-tracing-217M/`

| What | Path |
|---|---|
| Circuit edges (the model's "circuit") | `scripts/circuit_trace.py` -> `outputs/circuit_edges.csv` (2,144,011 edges) |
| Original, buggy Phase 11 | `scripts/remaining_phases.py:241-341` -> `outputs/phase11_crispri_validation.json` (n_validated 1,458,016; n_correct 796,159; 0.5461) |
| Corrected Phase 11 ("A2") | `scripts/audit_a2_groupkfold_crispri.py` -> `outputs/groupkfold_crispri/{per_pair.parquet, per_source_accuracy.csv, summary.json}` |
| Per-pair records | `outputs/groupkfold_crispri/per_pair.parquet` (11.9 MB on disk, 1,503,408 rows, 248 sources, 6,324 targets; columns source, target, predicted_inhibitory, actual_lfc, actual_inhibitory, correct) |
| Audit notes | `<REPO_ROOT>/projects/maxtoki/audits/audit_a2_groupkfold_status.md` (lines 62-90, written before the re-run: "BLOCKED"), `audits/audit-20260507-completion-v2.md:68-85` (F8, the re-run), `audits/audit_a3_bootstrap_cis.py:112-140` + `audits/bootstrap_cis_summary.json` (Wilson CI on the buggy 54.61%) |
| Run summary | `<REPO_ROOT>/projects/maxtoki/summaries/circuit-tracing-217M-FINAL_SUMMARY.md:91-115` |
| Paper claims | `paper-plos-one/main.tex:1092-1102` (Results), `:1540-1552` (P3 audit story), `:1585-1590` ("two errors in opposite directions"), `cover-letter.tex:53-54`; same text in `paper-biosystems/main.tex:966-970, 1657-1667` |
| CRISPRi data | Replogle K562, `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad` (30.0 GB), resolved by `projects/maxtoki/setup/dataset_loader.py:21,149` |

## 3. How a prediction (the sign of y-hat) is made — step by step

Verified by reading the code.

1. **Circuit edges** (`circuit_trace.py`). 30 SAE source features at each of layers 0, 3, 6, 9
   (120 source features; chosen as the features with the largest summed -log10 p of enrichment,
   lines 133-145). For each of 200 K562 control cells, one feature is set to zero in SAE space,
   the change is decoded and patched into the residual stream (lines 190-211). The change in every
   downstream SAE feature is averaged over token positions (line 220). Over cells this gives a
   Cohen's d. An edge is kept if |d| > 0.5 and the sign is the same in > 70% of cells (line 246).
   **Edge sign** (line 253): `"inhibitory" if mean_delta < 0 else "excitatory"`.
   So "inhibitory" here means *removing the source feature lowers the target feature*.
   88.17% of all edges are "inhibitory" (`outputs/circuit_summary.json`).
2. **Feature edges -> gene pairs** (`audit_a2...py:60-84`). Each SAE feature has a list of its top
   genes (`runs/sae-atlas-217M/outputs/phase2/layer_XX/feature_catalog.json`, first 10 of
   `top20_genes`). For each edge, every one of the 10 source-feature genes is paired with every one of
   the 10 target-feature genes (self pairs skipped). Per gene pair it counts: evidence (number of
   edges), max |d|, and number of "inhibitory" edges.
3. **Filter** (line 83): keep a gene pair if evidence >= 2 or max |d| > 2.0.
4. **Predicted sign** (line 89): `pred_inhib = n_inhib / n_total > 0.5`.
   `pred_inhib = True` is read as "knocking down the source gene lowers the target gene" (LFC < 0).
   A tie (exactly 0.5) counts as "increase". 43,786 pairs (2.9%) are ties.
   The mapping direction is internally consistent: an "inhibitory" edge (ablation lowers the target)
   predicts a negative LFC. The word "inhibitory" is the reverse of the usual biology word, but the
   comparison itself is the right way round.
5. **Observed sign** (lines 111-161). LFC = mean log1p(CP10k) over K562 cells with that CRISPRi guide
   (at least 10 cells) minus the same mean over 3,000 random non-targeting K562 cells (seed 42).
   `actual_inhibitory = actual_lfc < 0`. No significance test and no minimum effect size.
   LFC == 0 counts as "increase"; only **1** of 1,503,408 pairs has LFC exactly 0.
6. `correct = (pred_inhib == actual_inhib)` (line 162).

**Why the predicted sign is almost always "decrease" (inference, not tested).** Zeroing an SAE
feature removes activation. Most downstream SAE features then go down, so 88% of edges are
"inhibitory". That is a property of the ablation, not of gene regulation. The gene-pair step then
inherits it: 89.6% of pairs are predicted "decrease".

**The original bug** (`remaining_phases.py:322-328`, verified). `predicted_inhibitory` is computed
from `info["max_d"] > 0` (always True, since max_d is an absolute value) and is never used. The
counter goes up whenever `actual_lfc < 0`. So the old 54.61% is exactly the fraction of observed
decreases, i.e. the accuracy of an always-decrease rule.

## 4. What a "source" is

A "source" is a **silenced gene**: a gene that (a) is in the top-10 gene list of one of the 120
source SAE features and (b) has a CRISPRi guide in the Replogle K562 data with at least 10 cells.
There are **248** such genes in the corrected records. Each is paired with 4,570 to 6,323 targets
(median 6,301). In practice nearly every source is paired with nearly every one of the 6,324 targets,
so the "circuit prediction" covers almost the full source x target grid.

The 248 sources are **not independent**. They come from only 110 source features. Genes in the same
feature's top-10 list get identical predictions. There are **139** distinct feature-membership sets and
**54** linked groups (largest group: 95 genes). A cluster bootstrap over the 139 sets gives the same
answer: model minus always-decrease **CI [-1.73, -0.58] pts**; balanced accuracy minus 0.5
**CI [-0.54, +0.16] pts** (`crispri/source_clusters.json`).

## 5. Why the pair counts differ (1,458,016 vs 1,503,408) — verified exactly

- The original script used only edges whose two ends **both have an enrichment label**
  (`has_both_labels`, `remaining_phases.py:134, 207`): **1,416,229** of 2,144,011 edges.
- The corrected script used every edge whose two ends have a **top-gene list**
  (`audit_a2...py:51-58`): all **2,144,011** edges.
- More edges means more gene pairs pass the evidence >= 2 / |d| > 2 filter.

I rebuilt both pair sets from the edge file and the feature catalogues
(`crispri/reconstruct_original.py`, `crispri/original_vs_corrected_records.json`):

- Corrected rule: all 1,503,408 records found; predicted sign matches on **1,503,408 / 1,503,408**.
- Original rule: **1,458,016** pairs and **796,159** decreases. Both match the logged values exactly.
- Every original pair is inside the corrected set (0 missing). The corrected set adds **45,392** pairs,
  all from the same 248 sources. Their decrease rate is 0.5462.
- LFCs are the same for shared pairs (same control sample, same seed, same formula).

On the **original** 1,458,016 records:

| Predictor | Accuracy | Balanced acc |
|---|---:|---:|
| Always decrease (= the old buggy metric) | 0.5461 | 0.500 |
| Corrected sign rule, all edges | 0.5350 | 0.4981 |
| Corrected sign rule, labelled edges only | 0.5324 | 0.4981 |

So the conclusion does not depend on which record set is used.

Note: the audit (`audit-20260507-completion-v2.md:72`) says "966 unique source genes". 966 is the
number of source genes in the original *predictions* (logged in `outputs/remaining_phases.log`), not
the number with CRISPRi data. The corrected records have 248.

## 6. The "grouped 5-fold CV" is not cross-validation

Verified in `audit_a2...py:184-199`. The script shuffles the 248 sources (seed 42), cuts them into 5
groups, and reports accuracy in each group. **Nothing is trained or fit.** The circuit predictions never
see CRISPRi data. So there is no training set to hold out from.

- Paper line 1097-1098: "holding out whole silenced genes in five-fold cross-validation so that no gene
  appears in both training and test". This describes something that did not happen.
- The CI [52.60, 54.16] is a percentile bootstrap of the per-source mean accuracy, 1,000 reps over the
  248 sources (`audit_a2...py:207-213`). It is not a CV result.
- Fold accuracies in `summary.json`: 0.5188, 0.5396, 0.5386, 0.5462, 0.5304; mean **0.53473**.
  The audit and run summary both say "53.55%" (`audit-20260507-completion-v2.md:78`,
  `circuit-tracing-217M-FINAL_SUMMARY.md:103`). That number is not in `summary.json`.
- I did use the same 5 source folds (reproduced exactly; fold sizes match) to cross-fit the
  learned baselines in section 9. There, holding out sources does mean something.

## 7. Pooled results on the corrected records (computed)

Positive class = observed decrease. From `crispri/results.json` -> `pooled`.

| Quantity | Value |
|---|---:|
| Pairs | 1,503,408 |
| Observed decrease (LFC < 0) | 820,950 (**54.61%**) |
| Observed increase (LFC > 0) | 682,457 (45.39%) |
| LFC exactly 0 (counted as increase) | 1 |
| Predicted decrease | 1,346,452 (**89.56%**) |
| Predicted increase | 156,956 (10.44%) |
| Confusion: pred dec & obs dec | 733,979 |
| pred inc & obs dec | 86,971 |
| pred dec & obs inc | 612,473 |
| pred inc & obs inc | 69,985 |
| Accuracy | **0.53476** |
| Always-decrease accuracy | **0.54606** |
| Always-increase accuracy | 0.45394 |
| Majority-class baseline (= always decrease) | 0.54606 |
| Accuracy minus best constant | **-0.01130** |
| Recall on decreases | 0.8941 |
| Recall on increases | 0.1025 |
| Balanced accuracy | **0.49830** |
| Matthews correlation (MCC) | **-0.00552** |
| Cohen's kappa | -0.00363 |
| Expected accuracy if prediction were independent of truth | 0.53644 |

Sensitivity (`crispri/tie_sensitivity.json`, `results.json`):
- Ties predicted "decrease" instead of "increase": accuracy 0.5382, balanced acc 0.4990. Still below 0.5461.
- Ties dropped (1,459,622 pairs): accuracy 0.5376 vs always-decrease 0.5457; balanced acc 0.4989.
- LFC == 0 dropped or counted as decrease: accuracy 0.534761 either way (one pair).

## 8. Per-source results (computed; `crispri/per_source_metrics.csv`)

Means over the 248 silenced genes (median in brackets):

| Quantity | Mean [median] | Range |
|---|---:|---|
| Fraction observed decrease | 0.5457 [0.5459] | 0.363 - 0.728 |
| Fraction predicted decrease | 0.8949 [0.9668] | 0.638 - 0.992 |
| Accuracy | **0.5342** [0.5244] | 0.362 - 0.729 |
| Always-decrease accuracy | **0.5457** | |
| Always-increase accuracy | 0.4543 | |
| Own majority sign (uses the test labels; an upper bound for any constant) | 0.5643 | |
| Balanced accuracy | **0.4986** [0.5027] | 0.432 - 0.522 |
| MCC | 0.0134 [0.0138] | -0.130 - 0.112 |

- Model beats always-decrease in 139 of 248 sources. It beats the source's own majority sign in 104.
- Balanced acc > 0.5 in 146 of 248. MCC > 0 in 146 of 248.
- 185 of 248 sources have more decreases than increases.
- Per-source accuracy is mostly the per-source decrease rate: Pearson r = **0.86** between them.
- No source gets the same predicted sign for all its targets.

## 9. Grouped bootstrap over silenced genes (2,000 reps, seed 42; `crispri/bootstrap_ci.csv`)

Sources resampled with replacement; pooled numbers recomputed from summed confusion counts.

| Statistic | Point | 95% CI | Reps > 0 |
|---|---:|---|---:|
| Pooled accuracy | 0.5348 | [0.5270, 0.5423] | |
| Pooled accuracy - 0.5 | +0.0348 | [+0.0270, +0.0423] | 100% |
| **Pooled accuracy - always-decrease** (best constant; re-choosing it in each rep gives the same) | **-0.0113** | **[-0.0161, -0.0072]** | 0% |
| **Pooled balanced accuracy - 0.5** | **-0.0017** | **[-0.0051, +0.0014]** | 15% |
| Pooled MCC | -0.0055 | [-0.0160, +0.0047] | 15% |
| Pooled accuracy - chance given class rates | -0.0017 | [-0.0050, +0.0014] | 15% |
| Per-source mean accuracy | 0.5342 | [0.5265, 0.5417] | |
| Per-source mean accuracy - always-decrease | -0.0115 | [-0.0163, -0.0074] | 0% |
| Per-source mean accuracy - own majority sign | -0.0301 | [-0.0370, -0.0241] | 0% |
| **Per-source mean balanced accuracy - 0.5** | **-0.0014** | **[-0.0032, +0.0002]** | 4.4% |
| Per-source mean MCC | +0.0134 | [+0.0068, +0.0197] | 100% |

My per-source mean CI [0.5265, 0.5417] reproduces the paper's [0.5260, 0.5416] (different reps and RNG draws).

One small positive: the mean per-source MCC is +0.013 and its CI excludes 0. Within a silenced gene,
the predicted sign has a very weak positive link with the observed sign. But per-source balanced
accuracy does not show it (mean 0.4986, CI crosses 0.5 only barely on the high side). A few sources drag
the mean down (10 sources have balanced acc < 0.47; 9 of them have almost the same
predicted-decrease rate, 0.638-0.641, which suggests they share source features). An MCC of 0.013 means about 0.02% of the variance in sign is shared.
This is not a usable predictor, and it loses to every simple baseline.

## 10. Learned baselines, cross-fitted (`crispri/pooled_baselines.csv`, `groupkfold_by_source_baselines.csv`)

Same 5 source folds as the script (reproduced exactly).

- **Global majority from training folds**: every training fold's majority is "decrease", so this equals
  always-decrease in every fold. Fold accuracies 0.538-0.553; model 0.519-0.546. Model is lower in all 5 folds.
- **Per-target majority from training folds** ("does this target gene usually go down when *other* genes
  are silenced?"): accuracy **0.5915**, balanced acc 0.577, MCC 0.162. Fold accuracies 0.588-0.597.
  Model minus this baseline: **-5.68 pts, CI [-6.39, -5.04]**. Model is better in only 22 of 248 sources.
- **Per-source majority, sign-matched** (the "majority sign of the source's own training fold").
  Under grouping by source, a source has no training fold, so I split each source's *targets* into 5
  folds (target folds fixed across sources, seed 43) and predicted each held-out target with the majority
  sign of that source's other targets. Accuracy **0.5643**, balanced acc 0.541. Model minus this:
  **-2.95 pts, CI [-3.66, -2.32]**. Note: this baseline uses labels from the same knockdown, so it is an
  upper-bound-type reference, not a fair competitor to a label-free predictor.

## 11. By size of the observed change (`crispri/strata_by_abs_lfc.csv`)

| abs(LFC) band | Pairs | Obs decrease | Model acc | Always-dec acc | Balanced acc |
|---|---:|---:|---:|---:|---:|
| < 0.01 | 200,002 | 0.503 | 0.503 | 0.503 | 0.501 |
| 0.01-0.025 | 280,482 | 0.512 | 0.513 | 0.512 | 0.503 |
| 0.025-0.05 | 374,415 | 0.533 | 0.527 | 0.533 | 0.501 |
| 0.05-0.1 | 406,014 | 0.557 | 0.542 | 0.557 | 0.497 |
| 0.1-0.25 | 222,180 | 0.611 | 0.577 | 0.611 | 0.491 |
| >= 0.25 | 20,315 | 0.740 | 0.669 | 0.740 | 0.496 |

- Median abs(LFC) is 0.042 (log1p-CP10k units). Most pairs are tiny changes.
- The larger the change, the more often it is a decrease. The model never beats always-decrease in any
  band, and balanced accuracy is 0.49-0.50 everywhere. Restricting to real changes makes the model look
  worse, not better.

## 12. What drives the predicted and observed signs (`results.json` -> `prediction_structure`)

- Share of variance in the predicted sign explained by target alone: 0.229; by source alone: 0.207.
- Share of variance in the observed sign explained by target alone: 0.050; by source alone: 0.018.
- 55.7% of targets are predicted "decrease" for every source; 0.5% "increase" for every source.

## 13. What this means for the paper

- "53.42% (CI [52.60, 54.16]) excludes 50%, so the effect is real" should be withdrawn. Against the
  majority-class rule the model is **1.1 points worse (CI [-1.6, -0.7])**. Balanced accuracy is
  **49.8% (CI for minus 0.5: [-0.5, +0.1] pts)**. The honest verdict: **no directional skill detected**.
- The "cross-validation" wording (main.tex:1097-1098) describes a training/test split that does not exist.
- The P3 audit story (main.tex:1540-1552) and the "two headline errors in opposite directions" argument
  (main.tex:1585-1590; cover letter lines 53-54) need revising. The audit fixed the code bug but kept the
  wrong chance level. The fixed metric is still inflated by class imbalance. The run summary's line
  "~3.4 percentage points above chance ... cannot be attributed to per-source-gene pseudoreplication"
  (`circuit-tracing-217M-FINAL_SUMMARY.md:106-111`) is also wrong for the same reason.
- This fits the paper's own broader negative on circuit-to-gene regulation. It makes it stronger.

## 14. What was NOT done

- No new model runs and no new LFCs. All numbers use the saved LFCs in `per_pair.parquet`.
  Recomputing them would need the 30 GB Replogle h5ad and the A2 script (about 1-2 h by the audit's
  estimate); not needed, because the records exist and were verified by exact reconstruction.
- No differential-expression test per pair (e.g. only significant knockdown effects). The abs(LFC) bands
  in section 11 are a rough stand-in.
- No check that the guide actually knocked down its own gene (on-target knockdown).
- No other sign rule was tried (e.g. sign of summed Cohen's d, weighting by |d|). The paper's rule was
  scored as written.
- No other datasets (RPE1, Adamson) were scored.
- The reason 54.6% of LFCs are negative (e.g. general effects of silencing essential genes, or the
  normalisation) was not investigated.

## 15. Files written

Folder: `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/crispri/`

- `analyze_crispri.py` — all metrics, per-source table, grouped bootstrap, cross-fitted baselines, strata.
- `results.json` — everything from the script above.
- `pooled_baselines.csv` — model vs 5 baselines (acc, balanced acc, MCC).
- `bootstrap_ci.csv` — all bootstrap CIs.
- `per_source_metrics.csv` — 248 rows, all per-source metrics and confusion counts.
- `groupkfold_by_source_baselines.csv` — per-fold model vs baselines.
- `strata_by_abs_lfc.csv` — by size of change.
- `reconstruct_original.py`, `original_vs_corrected_records.json` — exact rebuild of both record sets.
- `tie_sensitivity.py`, `tie_sensitivity.json` — tie handling.
- `source_clusters.py`, `source_clusters.json`, `source_groups.csv` — source non-independence and cluster bootstrap.

## Plain-words summary

The model says "the gene goes down" for 9 out of 10 pairs. In the real data, 55 out of 100 pairs go
down. So a rule that always says "down" is right 54.6% of the time. The model is right 53.5% of the
time, which is worse than that rule. The gap is about 1 point and the bootstrap interval excludes zero.
Once the imbalance is removed (balanced accuracy), the model scores 49.8%, which is chance. The
"cross-validation" in the paper trained nothing, so it does not change this. The paper should say
there is no detectable directional skill.
