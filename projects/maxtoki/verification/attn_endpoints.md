# attn_endpoints — which endpoint does Figure 2 show?

Reviewer 2 is right. Every number in Figure 2 and in the "6–20 AUROC points" claim comes from the
**per-perturbation knockdown-response endpoint** (pipeline Phase 1), not from the TRRUST endpoint.
The caption says the figure measures "curated regulator–target relationships". That is wrong.
Two other claims in the same paragraph ("z ≈ 0", "collapses to chance") come from the **TRRUST
endpoint** (Phase 3). So one paragraph and one caption mix two different endpoints without saying so.

I also found two new problems while checking:

1. **The degree-preserving (curveball) null never actually shuffled the network.** In the 217M K562
   run, 49 of 50 null draws were identical to the real TRRUST labels. So "z ≈ 0" is built in by the
   code. It is not evidence of anything. (Verified by re-running the exact code; details in §6.)
2. **On the TRRUST endpoint, gene variance does not beat attention in every run.** The pipeline never
   computed a gene-variance baseline on TRRUST. I computed it. Variance clearly beats attention in
   2 of 4 runs, ties in Adamson (0.524 vs 0.520), and is not significant in the 1B run.

Everything below says whether I **verified** it by reading a file / re-running code, or whether I
**infer** it.

---

## 1. Where the Figure 2 numbers live

**Figure source (verified).** Figure 2 is the second `figure` environment in
`<REPO_ROOT>/projects/maxtoki/paper-biosystems/main.tex`
(lines 1013–1062). The PLOS figure build script
`<REPO_ROOT>/projects/maxtoki/paper-plos-one/build/make_figures.sh`
(lines 12–40) pulls each TikZ picture out of that file and renders it to `figures/Fig2.tif`. There is
no separate data file. The numbers are typed straight into the TikZ:

- `paper-biosystems/main.tex:1042-1043` — blue "Model attention": `(k562a,0.512) (rpe1,0.603) (adamson,0.508) (k562b,0.540)`
- `paper-biosystems/main.tex:1044-1045` — orange "Gene-variance baseline": `(k562a,0.596) (rpe1,0.766) (adamson,0.712) (k562b,0.596)`

`make_plos.py` removes the TikZ from the PLOS manuscript and keeps only the caption
(`paper-plos-one/build/make_plos.py:127-136`). So the PLOS text and caption are copies of the
BioSystems ones.

**Where these numbers come from (verified).** They are the Phase 1 mean per-perturbation AUROCs,
rounded to 3 decimals:

| Run | Figure blue | Phase 1 `attention_primary` | Figure orange | Phase 1 `gene_variance` |
|---|---|---|---|---|
| 217M K562 | 0.512 | 0.5119 (`summaries/attention-grn-217M-k562-report.md:23`) | 0.596 | 0.5957 (`…-k562-report.md:20`) |
| 217M RPE1 | 0.603 | 0.6033 (`summaries/attention-grn-217M-rpe1-report.md:21`) | 0.766 | 0.7664 (`…-rpe1-report.md:19`) |
| 217M Adamson | 0.508 | 0.5081 (`summaries/attention-grn-217M-adamson-report.md:24`) | 0.712 | 0.7123 (`…-adamson-report.md:20`) |
| 1B K562 | 0.540 | 0.5398 (`summaries/attention-grn-1B-k562-report.md:22`) | 0.596 | 0.5961 (`…-1B-k562-report.md:19`) |

(All `summaries/…` paths are under `<REPO_ROOT>/projects/maxtoki/`.
Full-precision values are in `runs/attention-grn-217M/outputs/phase1{,_rpe1,_adamson,_k562_1b}/run_config.json`.)

The TRRUST attention AUROCs would have been 0.4829 / 0.6045 / 0.5195 / 0.5554. None of those appear in
the figure. The reviewer's example (0.5119 vs 0.4829) is `…-k562-report.md:23` vs `:74`.

---

## 2. What each endpoint literally measures (verified from code)

Both endpoints use the same attention score: layer 8 (0-indexed), averaged over heads and over the
control cells in which both genes appear (`runs/attention-grn-217M/scripts/phase0_extract.py:291-330`).
Row = the query gene, column = the key gene. The gene set is the top 1,500 most variable genes
(log1p CP10k, control cells) that are in the MaxToki vocabulary (`phase0_extract.py:150-170`).

### (a) Per-perturbation knockdown-response endpoint ("DE endpoint", Phase 1 and Phase 2)
Code: `runs/attention-grn-217M/scripts/phase1_trivial_baselines.py:154-257`.

- One test per perturbation. A perturbation is used only if its target gene is one of the 1,500 genes,
  it has ≥30 cells, and it gives ≥3 responding genes.
- **Positives:** genes that change after the knockdown. Test = Welch t-test, perturbed cells vs up to
  5,000 control cells, log1p CP10k; BH within the perturbation; |log fold change| ≥ 0.5 and q < 0.05
  (lines 49-52, 169-183).
- **Negatives:** every other of the 1,500 genes, minus the perturbed gene itself.
- **Attention score:** `attn[perturbed gene, candidate gene]`.
- **Gene-variance score:** variance of the candidate gene across control cells. It is the same ranking
  for every perturbation. It does not use the model.
- **Summary:** mean of per-perturbation AUROCs; paired Wilcoxon across perturbations; BH across the
  13 tests in that run's table (lines 297-349).
- **Plain words:** "Given a gene that was knocked down, can attention rank which genes will respond?"
  These positives are measured responses, direct or indirect. They are **not** curated regulator–target
  pairs. Gene variance wins here because variable genes are the ones that show up as responders after
  almost any knockdown (inference; this is the known "responsiveness" effect).
- Phase 2 (incremental value) and Phase 3c (propensity matching) use these same labels
  (`phase2_incremental_value.py:176-199`; `phase3_residualization.py:317-391`).

### (b) TRRUST edge endpoint (Phase 3a, 3b, Phase 4)
Code: `runs/attention-grn-217M/scripts/phase3_residualization.py:86-141`.

- Reference: TRRUST v2 human file
  `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv`.
  Only edges where both TF and target are among the 1,500 genes.
- Only TF rows with ≥3 TRRUST targets in the gene set are scored (line 111).
- **Positives:** that TF's TRRUST targets. **Negatives:** all other genes in the set (unknown pairs are
  treated as negatives).
- **Score:** `attn[TF, candidate gene]`. All TF rows are pooled into **one** AUROC (not a per-TF mean).
- **Curveball null (3b):** permute the binary TRRUST matrix while keeping row and column sums; n = 50
  draws (lines 219-269). **Label-shuffle null:** shuffle labels within each TF row; n = 1,000 (lines 271-294).
- **Residualisation (3a):** regress attention on [source mean, source variance, target mean, target
  variance, target dropout] over all gene pairs, 5-fold cross-fit, OLS or GBDT; then TRRUST AUROC of the
  residual (lines 143-214).
- **No gene-variance baseline and no incremental-value test were ever run on this endpoint.**
- **DoRothEA / STRING / other curated databases: not run.** Every run report says "TRRUST only"
  (e.g. `…-k562-report.md:107`).
- **The TRRUST test is small** (verified by my re-run; the edge and TF counts also match
  `runs/attention-grn-217M/outputs/phase3*/run_config.json`):

| Run | TRRUST edges in gene set | TFs scored | Positive pairs scored | Negative pairs |
|---|---|---|---|---|
| 217M K562 | 66 | 8 | 45 | 11,947 |
| 217M RPE1 | 173 | 16 | 138 | 23,846 |
| 217M Adamson | 281 | 32 | 236 | 47,732 |
| 1B K562 | 96 | 12 | 73 | 17,915 |

For comparison, the DE endpoint has 9,741 / 48,245 / 4,438 / 8,587 positive pairs across
174 / 325 / 57 / 155 perturbations. I summed these from `phase1*/per_perturbation_auroc.csv`. They match
`phase2*/run_config.json` n_pairs × positive_rate.

---

## 3. Master table — both endpoints, all four runs

R = `<REPO_ROOT>/projects/maxtoki/summaries/`,
O = `<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs/`.
Report files: K = `R/attention-grn-217M-k562-report.md`, P = `R/attention-grn-217M-rpe1-report.md`,
A = `R/attention-grn-217M-adamson-report.md`, B = `R/attention-grn-1B-k562-report.md`.
"NEW" = computed by me in this check. It is not in any repo file.

### (a) DE endpoint (per-perturbation)

| Quantity | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 | Source |
|---|---|---|---|---|---|
| n perturbations | 174 | 325 | 57 | 155 | K:28, P:27, A:27, B:27 |
| Attention AUROC (mean) | 0.5119 | 0.6033 | 0.5081 | 0.5398 | K:23, P:21, A:24, B:22 |
| Gene-variance AUROC (mean) | 0.5957 | 0.7664 | 0.7123 | 0.5961 | K:20, P:19, A:20, B:19 |
| Gap (variance − attention) | +0.0837 | +0.1632 | +0.2042 | +0.0562 | K:32, P:31, A:31, B:31 |
| Perturbations where variance > attention | 118/174 | 314/325 | 56/57 | 91/155 | NEW (from `O/phase1*/per_perturbation_auroc.csv`) |
| Wilcoxon p, BH (13 tests per run) | 4.03e-10 | 1.51e-51 | 1.38e-10 | 9.47e-05 | K:32, P:31, A:31, B:31; `O/phase1*/wilcoxon_baseline_vs_edges.csv` row 2 |
| Curveball null z | not run | not run | not run | not run | — |
| Incremental ΔAUROC, gene+attn vs gene-only (logreg, cross-pert) | +0.0007 | −0.0003 | −0.0001 | +0.0020 | K:55, P:54, A:54, B:54 |
| Largest incremental Δ over all three splits (logreg) | +0.0007 | −0.0002 | −0.0001 | +0.0021 (cross-gene) | K:55, P:59, A:59, B:49; `O/phase2*/delta_auroc_summary.csv` |
| GBDT incremental Δ | not run (`has_gbdt: false`) | not run | not run | not run | `O/phase2*/run_config.json` |
| Propensity-matched Δ (gene+attn − gene-only) | +0.0046 | +0.0015 | +0.0001 | **+0.0457** | K:82, P:81, A:81, B:81 |
| Propensity-matched attention AUROC (closest analogue to "residualised" on this endpoint) | 0.4949 | 0.5009 | 0.4205 | 0.5174 | `O/phase3*/propensity_matching_results.json` → `matched_mean_per_pert_auroc.attention` |

### (b) TRRUST endpoint (pooled over TF rows)

| Quantity | 217M K562 | 217M RPE1 | 217M Adamson | 1B K562 | Source |
|---|---|---|---|---|---|
| TFs / positive pairs | 8 / 45 | 16 / 138 | 32 / 236 | 12 / 73 | NEW (TF counts also in `O/phase3*/run_config.json`) |
| Attention AUROC | 0.4829 | 0.6045 | 0.5195 | 0.5554 | K:74, P:73, A:73, B:73 (my re-run reproduces all four exactly) |
| Spearman AUROC | 0.5432 | 0.5963 | 0.5389 | 0.5800 | K:69, P:68, A:68, B:68 |
| **Gene-variance AUROC** (target variance) | **0.7252** | **0.6864** | **0.5242** | **0.6292** | NEW |
| Gap (variance − attention) | +0.242 | +0.082 | **+0.005** | +0.074 | NEW |
| TF-level bootstrap 95% CI of gap (2,000 resamples of TFs) | [+0.116, +0.359] | [+0.024, +0.145] | **[−0.057, +0.060]** | **[−0.037, +0.181]** | NEW |
| TFs where variance > attention | 7/8 | 13/16 | **13/32** | 7/12 | NEW |
| Target mean-expression AUROC | 0.5386 | 0.4695 | 0.5894 | 0.6097 | NEW |
| Target (1 − dropout) AUROC | 0.5042 | 0.4323 | 0.5824 | 0.5876 | NEW |
| 3-feature gene-only logreg (GroupKFold by TF, 5 folds) | 0.7631 | 0.7531 | 0.6672 | 0.6834 | NEW |
| Attention-only logreg (same CV) | 0.5167 | 0.5861 | 0.5213 | 0.5293 | NEW |
| Incremental Δ gene+attn vs gene-only (same CV) | 0.0000 | **+0.0139** | +0.0034 | −0.0064 | NEW (noisy: 8–32 TF groups) |
| Curveball null z (as reported) | −0.14 | +0.17 | +0.16 | −0.04 | K:75, P:74, A:74, B:74; `O/phase3*/null_results.json` |
| Curveball null draws identical to real labels | **49/50** | 31/50 | 10/50 | 41/50 | NEW (exact re-run of the repo code) |
| Curveball z with a null that actually mixes (n = 200) | +0.55 (p = 0.30) | +1.87 (p = 0.035) | +2.00 (p = 0.020) | +1.13 (p = 0.13) | NEW |
| Label-shuffle null z | −0.30 | **+4.18** | +1.08 | +1.47 | K:76, P:75, A:75, B:75 |
| Residualised AUROC, OLS | 0.4866 | 0.4922 | 0.5202 | 0.5434 | K:67, P:66, A:66, B:66 |
| Residualised AUROC, GBDT | 0.4970 | 0.5191 | 0.5317 | 0.5111 | K:68, P:67, A:67, B:67 |
| C3 "residual keeps >50% of signal" | FAIL | FAIL | **PASS** (1.03) | **PASS** (0.78) | `O/phase12_verdict*.json` → `C3_residualized_retains_signal` |

The NEW numbers come from these scripts and logs (all in the scratchpad `understand/` folder):
`attn_trrust_baselines.py` / `.log` / `.json`, `attn_verify2.py` / `.log`, `attn_verify3.py` / `.log`.
They only read Phase 0 files (`gene_features.csv`, one layer of `attention_edges_layer_mean.npy`
via mmap, `spearman_edges.npy`) and the TRRUST TSV. No model was run.

---

## 4. Which endpoint each paper claim uses

PLOS text is `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`.
BioSystems line numbers are in brackets.

| Paper claim | Where | Endpoint | Numbers behind it |
|---|---|---|---|
| Figure 2 bars | Fig 2 [biosystems 1042-1045] | **DE** | 0.512/0.603/0.508/0.540 vs 0.596/0.766/0.712/0.596 |
| "recover curated regulator–target relationships" | caption 1145-1146 [1051-1052] | caption says TRRUST-like; **data are DE** | — mislabelled |
| "6–20 AUROC points in every run" | 1084 [955] | **DE** | +8.4, +16.3, +20.4, +5.6 (5.6 rounded up to "6") |
| "5.6 to 20.4 AUROC points" | 1149 [1055] | **DE** | same |
| "p ≤ 10⁻⁴ after correction" | 1085, 1150 [956, 1056] | **DE** | BH p 4.0e-10, 1.5e-51, 1.4e-10, 9.5e-05 (BH over 13 tests within each run, not across runs) |
| "at most +0.002 AUROC in any run" | 1153 [1059] | **DE** (Phase 2 labels are DE) | max +0.0021 (1B, cross-gene logreg). Leaves out the 1B propensity-matched +0.0457 and the fact that GBDT was never run |
| "z ≈ 0 in all four runs" (degree-preserving null) | 1079 [950] | **TRRUST** | −0.14 / +0.17 / +0.16 / −0.04, but the null did not mix (see §6) |
| "once gene-level information is removed… no better than chance" / "residualizing… collapses it to chance" | 1086-1088, 1154-1155 [957-959, 1060-1061] | **TRRUST** | OLS 0.4866 / 0.4922 / 0.5202 / 0.5434. Not true for Adamson (0.5195 → 0.5202, no drop) or 1B (0.5554 → 0.5434, keeps 78%, C3 PASS). In K562 the raw value is already below chance (0.4829), so there is nothing to collapse |
| "Scaling the model tenfold" / "ten-times-larger" | 1151, 1082 [1057, 953] | — | 1B / 217M ≈ 4.6×. `FINAL_SUMMARY.md:16` says "5×" |
| Table row "against curated regulatory databases" | 940-941 | — | Only TRRUST was used |

---

## 5. Runs where the baseline does NOT beat attention on TRRUST

- **217M Adamson:** variance 0.5242 vs attention 0.5195. The gap is +0.005 and the TF-bootstrap CI is
  [−0.057, +0.060]. Variance wins in only 13 of 32 TFs. Mean per-TF AUROC: variance 0.502, attention
  0.523, so attention is slightly ahead on that view. **This is a tie.**
  Other gene-level features do beat attention here: mean expression 0.589, 1−dropout 0.582, and the
  3-feature gene-only model 0.667 in cross-validation.
- **1B K562:** variance 0.6292 vs attention 0.5554. The point gap is +0.074, but the TF-bootstrap CI
  crosses zero ([−0.037, +0.181]). Variance wins in only 7 of 12 TFs. **Not significant at the TF level.**
- 217M K562 (+0.242, CI above 0) and RPE1 (+0.082, CI above 0): variance clearly wins.

So on TRRUST, "one number per gene beats attention in every run" is true for 2 of 4 runs with variance
alone. The broader statement "gene-level features beat attention" holds in all 4 runs if you allow the
3-feature gene model. I computed all of this myself. None of it was in the pipeline outputs.

---

## 6. The curveball null did not shuffle (verified by exact re-run)

- Code: `runs/attention-grn-217M/scripts/phase3_residualization.py:221-263`. It runs 5 × (number of
  edges) trade attempts, e.g. 330 for 217M K562. Each attempt picks **two random rows out of all 1,500**.
  A trade changes anything only when both rows hold TRRUST edges. Only 27 of 1,500 rows (K562) hold any
  edges. So almost every attempt does nothing.
- I re-ran the exact function with the same seeds. I reproduced the reported null exactly (K562: mean
  0.4831, std 0.0011, z = −0.14). **49 of the 50 null draws were identical to the real TRRUST matrix.**
  Identical draws: RPE1 31/50, Adamson 10/50, 1B 41/50.
- So null mean ≈ observed and z ≈ 0 **by construction**. This says nothing about attention.
- The fix: draw trade pairs only among rows that have edges (same trade rule, same row and column
  sums). With that fix and 200 draws, z = +0.55 / +1.87 / +2.00 / +1.13 (one-sided p = 0.30 / 0.035 /
  0.020 / 0.13). None survive a Bonferroni correction across 4 runs (smallest 0.020 × 4 = 0.08). So the
  qualitative negative probably survives. But "z ≈ 0" is not a true description of any working null.
- The run also used 50 draws. The spec asks for 200 (`pipelines/attention-grn-extraction-and-evaluation.md:213`, `:757`).
  The PLOS text quotes a "Step 4" with "1000 iterations" and "z ≥ 3.0" (`paper-plos-one/main.tex:605-610`).
  I could not find that text anywhere in `pipelines/`, `summaries/` or the project README (grep, no hits).
- The internal audit rated this null "clean — exemplary" (`projects/maxtoki/audits/audit-20260507.md:84`, `:178`).
  It missed that the null does not mix.

---

## 7. Other inconsistencies found in the source reports (verified)

1. **1B C3 status disagrees between files.** `outputs/phase12_verdict_k562_1b.json` has C3 `pass: true`
   (fraction retained 0.784). `summaries/attention-grn-217M-FINAL_SUMMARY.md:12` says FAIL. Line 109 says
   "Residualized AUROC collapses to chance after OLS cross-fit in every run". That is not true for
   Adamson or 1B.
2. **RPE1: attention does beat two "trivial" baselines** on the DE endpoint. It beats mean expression
   (0.6033 vs 0.5822, BH p 7.3e-08) and 1−dropout (vs 0.5002, BH p 4.1e-36): `…-rpe1-report.md:34,37`,
   `phase12_verdict_rpe1.json` `attention_won: true`. `FINAL_SUMMARY.md:36` says "No run's attention beats
   any of the three trivial baselines". The paper only claims variance, so it is fine on this point.
3. **The 1B run is mislabelled and not like-for-like.** `…-1B-k562-report.md:1,7` says "MaxToki-217M"
   because `scripts/phase0_any.py:209` hard-codes that model name. The layer/head counts (20 × 16)
   show it is the 1B model. It used **200 control cells, not 2,000** (`:8`). Its "primary layer" is
   L8 of 20 (40% depth), not the ~73% depth used for 217M. Its best DE-endpoint layer is L5 (0.5813;
   `outputs/phase1_k562_1b/attention_per_layer_auroc.csv`). So the 4th bar is not a clean scale comparison.
4. **Copy-paste scope lines.** The RPE1 and Adamson reports say "dataset: Replogle K562 non-targeting
   controls only" (`…-rpe1-report.md:93`, `…-adamson-report.md:93`). They also list Phase 4 as run when
   it was not.
5. **p-values in FINAL_SUMMARY.md:32-33** ("2e-52", "5e-11") are the raw p-values, or neither raw nor BH.
   The BH values are 1.51e-51 and 1.38e-10.
6. `FINAL_SUMMARY.md:11` calls Adamson "CRISPRa". To my knowledge Adamson et al. 2016 is a CRISPRi
   screen. This is an inference; check it. The paper itself does not say CRISPRa.
7. The paper's "pre-registered 0.005 threshold": the 0.005 threshold is the pipeline's C2 criterion
   (`runs/attention-grn-217M/README.md`, "Four mandatory criteria"). I found no pre-registration record.

---

## 8. Recommendation: how the paper should present both endpoints

1. **Split Figure 2 into two panels, each labelled with its endpoint.**
   - Panel A, "Which genes respond to a knockdown (per-perturbation, Replogle/Adamson CRISPR screens)":
     keep the current bars. State n perturbations (174/325/57/155) and n responding pairs.
   - Panel B, "TRRUST TF→target edges": attention 0.483/0.605/0.520/0.555 vs target-variance
     0.725/0.686/0.524/0.629. Add TF-bootstrap 95% CIs and n (8/16/32/12 TFs; 45/138/236/73 positives).
     The Panel B baseline is new work. Add it to `phase3_residualization.py` and the run reports first,
     so every number stays traceable.
2. **Rewrite the caption.** Panel A is not "curated regulator–target relationships". Say "genes whose
   expression changes after CRISPR knockdown of the regulator".
3. **Tie each text claim to its endpoint.**
   - "6–20 points" / "5.6–20.4" / "p ≤ 10⁻⁴": say these are the knockdown-response endpoint.
   - On TRRUST, say: variance beats attention in 2 of 4 runs; ties in Adamson; not significant in 1B;
     a 3-feature gene-only model beats attention in all 4.
4. **Drop "z ≈ 0" until the null is fixed.** Re-run the curveball with a null that mixes (trade only
   among rows with edges; 200 draws as the spec says). Report the new z per run. My quick version gives
   +0.55 to +2.00. That is still weak, but it is not zero. Say it is on the TRRUST endpoint.
5. **State the residualisation result per run.** It falls to about chance in RPE1 only (0.605 → 0.492).
   It does not move in Adamson (0.520 → 0.520). It keeps 78% of the signal in 1B (0.555 → 0.543).
   K562 starts below chance.
6. **Incremental value.** Say "+0.002" is logistic regression on the knockdown-response endpoint, and
   that GBDT was not run. Report the 1B propensity-matched +0.046. Report the TRRUST-endpoint increment
   too (my estimate: 0.000 / +0.014 / +0.003 / −0.006, noisy).
7. **Say what was not done.** Only TRRUST was used. DoRothEA, STRING and ChIP-based references were not
   tested. The 1B run used 200 cells and a non-matched layer. The model is ~4.6× larger, not 10×.
8. **Keep the verdict, but narrow it.** The negative verdict still stands. Attention does not add
   anything that gene-level features cannot already give, on either endpoint. But the strongest,
   cleanest evidence is on the knockdown-response endpoint. The TRRUST evidence is small (45–236
   positive pairs) and its null test was broken.

---

## Plain-words summary

Figure 2 shows how well attention predicts which genes respond to a CRISPR knockdown. It does not show
TRRUST. The caption says it does, so the caption is wrong. The "6–20 points", "5.6–20.4", "p ≤ 1e-4" and
"+0.002" numbers all come from the knockdown endpoint. The "z ≈ 0" and "collapses to chance" claims come
from TRRUST. The z ≈ 0 is fake: the shuffling code almost never shuffled (49 of 50 null draws in K562
were the real network). On TRRUST, gene variance beats attention clearly in 2 runs, ties in Adamson, and
is not significant in the 1B run. Residualisation does not bring attention to chance in Adamson or 1B.
The overall negative verdict survives, but the paper should show both endpoints separately, with their
real sizes, and fix the null before quoting any z.
