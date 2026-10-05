# MaxToki deployment: error ledger and pre-audit snapshot (item C-prep)

> **Read first.** This file describes the ledger as it stood before the blind re-audits of Study C.
> Its 117 errors include 35 that appear only in paper drafts. The final ledger used in the
> manuscript is S1 Table (`projects/maxtoki/paper-plos-one/supporting/S1_Table.csv`): 149 errors in
> the deployment's code, logs, outputs, run summaries and audit reports, 89 from this ledger
> (L ids: the 82 run-output rows described below plus L118-L124, added to error_ledger.csv later) and 60 found in the blind re-audits (R ids). Errors found only in paper drafts are not
> in S1 Table. `error_ledger.csv` holds all rows.

This file lists every distinct error we could confirm in the outputs of the deployed MaxToki-217M study (run code, run summaries, audit files) and in the paper drafts built on them. It also describes the pre-audit snapshot built for the blind re-audit (Study C). The machine-readable ledger is `error_ledger.csv` (same folder). The snapshot manifest is `<EVAL_ROOT>/studyC/SNAPSHOT_MANIFEST.md`.

## 1. Numbers first

- **117 distinct errors.** 82 are in the run outputs (code, run summaries, audit files). 35 appear only in the paper text.
- **Severity:** minor 62, major 46, critical 9. A *critical* error means a headline claim cannot stand.
- **Type:** wrong description 41, unsupported claim 25, missing baseline or null 18, wrong statistic 13, code bug 11, reproducibility 9.
- **Checklist pattern:** 44 of 117 errors fall outside the ten-pattern checklist (P1-P10). Of the 82 run-output errors, 26 fall outside it. Counts by pattern (all rows): outside checklist 44, P10 13, P1 11, P4 11, P6 9, P5 8, P3 6, P8 6, P9 5, P7 3, P2 1.
- **Present before the audit started (2026-05-07 16:57):** 65 errors (28 critical or major).
- **What the original audit sweep caught among those 65:** yes 7, partly 3, no 55. Among the 28 critical or major ones: yes 3, partly 1, no 24. None of the 6 critical pre-audit errors was caught by the sweep (L001, L002, L005, L011, L012, L015).
- **Who found each error first:** independent verification (this investigation) 95, audit sweep agent 10, human (external) 8, repair-round agent 3, executor agent during run 1. No error is recorded as first found by the human who supervised the deployment.
- **Repair status:** wording only 79, pending re-run 30, repaired during deployment 8.
- **Waiting for the MPS re-runs:** 7 rows (L005, L011, L015, L022, L027, L031, L046). They are marked `confirmation pending (MPS re-run)` or `code-level only; impact pending (MPS re-run)`.
- **Coverage of the plan's list E1-E42:** every E-number maps to at least one row. 26 rows are new (not in E1-E42).

## 2. Key numbers re-derived for this ledger

All numbers below come from `ledger_build/verify_ledger_evidence.py` (output `verify_results.json`), which reads only saved files. No model was run.

- **CRISPRi direction (L001, L002).** The deployed Phase 11 code counts `actual_lfc < 0` and never uses the predicted sign (remaining_phases.py line 327). So 54.61% is just the share of genes that went down. On the 1,503,408 corrected pairs from 248 silenced genes: model accuracy 0.5348; always predicting 'decrease' scores 0.5461; balanced accuracy 0.4983; MCC -0.0055. The model says 'decrease' for 89.6% of pairs. Model minus the best constant rule: -1.13 points, 95% CI [-1.60, -0.73]. Balanced accuracy minus 0.5: 95% CI [-0.50, 0.13] points. Intervals: percentile bootstrap, 2,000 resamples of silenced genes with replacement, seed 20261001, pooled confusion counts. Two ways: pooled per-pair arrays and per-gene count sums give the same accuracy to 1e-12. Plain reading: no directional skill; the model does slightly worse than always guessing 'down'.
- **Curveball null (L033).** Re-running the deployed function with its own seeds: 49 of 50 null draws are identical to the real TRRUST matrix on the scored rows in 217M K562 (RPE1 37, Adamson 10, 1B 46). So the reported z of about 0 is built in by the code.
- **Triplets (L012, L013).** 4 triplets were tested, not 2,980; 2,977-2,980 is a count of downstream features. Solving A:B:C from two reported ratios predicts the other three to within the 4-decimal rounding of the JSON values (max residual 1e-4; 3.6e-4 if solved from a different pair) (triplet 0: AC 0.2189 vs 0.2189, three-way 0.1896 vs 0.1896, C given AB 0.2938 vs 0.2938). That only happens if each combined ablation equals its deepest single ablation (ABC = C), so 'zero synergy' is forced by the code.
- **Block-deleting hooks (L005, L011).** Among 2,144,011 circuit edges, the layer right after each source layer has 0 edges (source layers 0, 3, 6, 9: 0, 0, 0, 0 edges). All 1000 exhaustive-mapping features have 0 edges at L6; edges per feature are 4970 +/- 25 (min 4900, max 5400).
- **Steering (L015).** Delta-s at alpha 5 divided by alpha 2 is 0.995-1.047 at every layer; the three features within a layer differ by at most 0.0015. L0/L3 = 5.40 and L0/L11 = 19.98 (the paper's '5x' and '20x'). cos(g_early, g_late) = 0.881 (summary says 0.999).
- **Manifold (L066, L067, L070, L072).** The frozen gate file has five gates; no script computes the permutation gate. The shuffled null passes the random and donor gates (0.799, 0.707). The lung control trust is 0.79956 (< 0.80, so it fails all four gates). All 12 zero-shot donors are among the 13 external donors; 41 of 45 zero-shot tissues are shared.
- **Spectral (L047, L048).** CKA at L0 is 1.0 for every pair (fixed token embedding). The three 'disjoint' 2,000-cell samples share 6, 8, 8 cells (pairs 42-43, 42-44, 43-44) (n = 592,317; two methods agree).

## 3. How the ledger was built

- Sources: the 14 investigation reports in `verification/` (read in full: audit_history, attn_endpoints, crispri, gata1, triplets, manifold_ci, numbers_A/B/C, contract, hardware, repro_inventory; skimmed: heldout_tasks, agent_context), the plan's list E1-E42, and the run files themselves.
- 23 deterministic checks (V01-V23) re-read the saved files. Each row's `deterministic_evidence` names its check or the investigation script it relies on. Rows that rely only on an investigation script say so ('not re-run in this ledger build').
- `first_found_by` uses only what the record supports: file times, script docstrings, the audit reports, and the human prompt log summarised in audit_history. 'human (external)' means a person outside the deployment who read the submitted manuscript. 'repair-round agent' means the same agent session during the two repair rounds on 2026-05-07 (17:50-21:10).
- `detected_by_original_audit_sweep` scores the audit report of 2026-05-07 17:38 (the sweep), not the later repair rounds. 'partly' means it flagged the area but not the actual error (for example, asked for a replication).
- `layer`: run_outputs = the error exists in code, run summaries or audit files; paper_text = only in the manuscript drafts. Where a run-output error is repeated in the paper, the row stays run_outputs and `repeated_in_paper` = yes.
- Hook bugs: the code-level facts and the output fingerprints are confirmed from files. Their final impact waits for the MPS re-runs (D0, D2-D7, D9). An interim D0 hook unit test (3 cells, written 2026-10-01 00:57) already reproduces the triplet overwrite (|AB - B| = 0 and |ABC - C| = 0 under the old hooks) and shows that hidden_states[11] is taken after the final norm.

## 4. Critical errors

| id | short_name | layer | pipeline | first_found_by | detected_by_original_audit_sweep | confirmation_status |
|---|---|---|---|---|---|---|
| L001 | CRISPRi sign never compared | run_outputs | circuit-tracing | repair-round agent | no | confirmed |
| L002 | CRISPRi accuracy judged against 50% | run_outputs | circuit-tracing | human (external) | no | confirmed |
| L005 | Circuit-trace hook deletes the source block | run_outputs | circuit-tracing | independent verification (this investigation) | no | confirmation pending (MPS re-run) |
| L011 | Exhaustive-mapping hooks delete a whole block | run_outputs | exhaustive-mapping | independent verification (this investigation) | no | confirmation pending (MPS re-run) |
| L012 | Triplet hooks overwrite each other (ABC = C) | run_outputs | exhaustive-mapping | independent verification (this investigation) | no | confirmed (output arithmetic) |
| L015 | Steering results are block-deletion artefacts | run_outputs | exhaustive-mapping | independent verification (this investigation) | no | confirmation pending (MPS re-run) |
| L026 | GATA1 'positive' fails a rarity/length-matched null | run_outputs | sae-atlas | independent verification (this investigation) | n.a. | confirmed |
| L090 | Audit matrix 61% -> 75%: no cell table; '2 untouched' wrong | run_outputs | framework (audit) | independent verification (this investigation) | n.a. | confirmed |
| L099 | 'Two errors in opposite directions' is the strongest evidence | paper_text | framework (audit) | independent verification (this investigation) | n.a. | confirmed |

## 5. Full ledger (short form)

Details (where it lives, evidence, who confirmed and repaired it, notes) are in `error_ledger.csv`.

### attention-grn (15)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L033 | Curveball null never shuffles (z = 0 by construction) | run_outputs | major | code bug | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L034 | Curveball used 50 draws; spec asks 200 | run_outputs | minor | reproducibility | outside checklist | yes | executor agent during run | pending re-run | no |
| L035 | Verdict thresholds looser than the spec | run_outputs | minor | reproducibility | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L036 | No gene-level baseline on the TRRUST endpoint | run_outputs | major | missing baseline or null | P2 | yes | independent verification (this investigation) | pending re-run | no |
| L037 | 'Residualised attention collapses to chance in every run' | run_outputs | major | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L038 | 'No run beats any trivial baseline' (RPE1 does) | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L039 | Summary p-values are not the BH values | run_outputs | minor | wrong statistic | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L040 | 1B run not like-for-like and mislabelled | run_outputs | minor | wrong description | P7 | yes | independent verification (this investigation) | wording only | no |
| L041 | RPE1/Adamson reports carry K562 scope text | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L042 | Adamson screen labelled CRISPRa | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L043 | 'Gain <= +0.002 in every run' omits 1B +0.046 | run_outputs | minor | unsupported claim | P5 | yes | independent verification (this investigation) | wording only | no |
| L044 | Fig 2 caption: knockdown endpoint called 'curated pairs' | paper_text | major | wrong description | P3 | no | human (external) | wording only | n.a. |
| L045 | 'Curated regulatory databases' (only TRRUST used) | paper_text | minor | wrong description | P3 | no | independent verification (this investigation) | wording only | n.a. |
| L101 | Quoted 'Step 4' (1,000 iterations, z >= 3, p_BH <= 1e-4) not in spec | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L113 | BH correction credited to the audit | paper_text | minor | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |

### circuit-tracing (10)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L001 | CRISPRi sign never compared | run_outputs | critical | code bug | outside checklist | yes | repair-round agent | repaired during deployment | no |
| L002 | CRISPRi accuracy judged against 50% | run_outputs | critical | missing baseline or null | P1 | partly | human (external) | pending re-run | no |
| L003 | 'Five-fold CV' that trains nothing | run_outputs | major | wrong description | P4 | no | independent verification (this investigation) | wording only | n.a. |
| L004 | Repair write-up numbers not in its JSON (966 genes, 53.55%) | run_outputs | minor | wrong statistic | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L005 | Circuit-trace hook deletes the source block | run_outputs | critical | code bug | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L006 | Edge density lacked a null; repair used a parametric one | run_outputs | minor | missing baseline or null | P1 | yes | audit sweep agent | repaired during deployment | yes |
| L007 | '22x denser' headline contradicts own numbers (41x) | run_outputs | minor | wrong statistic | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L008 | Old and new CRISPRi figures use different record sets | run_outputs | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L009 | Silenced genes treated as independent units | run_outputs | minor | unsupported claim | P4 | no | independent verification (this investigation) | wording only | n.a. |
| L010 | No intervention sanity check or positive control (circuit tracing) | run_outputs | major | missing baseline or null | P6 | yes | audit sweep agent | pending re-run | yes |

### cross-pipeline (13)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L085 | 'Five independent probes converge' on the negative | run_outputs | major | unsupported claim | P9 | partly | independent verification (this investigation) | wording only | no |
| L088 | No checkpoint revision pinned; code pins unverifiable | run_outputs | minor | reproducibility | P10 | yes | independent verification (this investigation) | wording only | no |
| L089 | Environment and intermediates not recorded | run_outputs | minor | reproducibility | P10 | yes | audit sweep agent | repaired during deployment | partly |
| L094 | Hardware: 'single A100-80GB'; 'SAE training on CPU' | paper_text | major | wrong description | P10 | no | human (external) | wording only | n.a. |
| L095 | '~50 GPU-hours' | paper_text | minor | wrong description | P10 | no | independent verification (this investigation) | wording only | n.a. |
| L104 | 'Claude Opus 4.7 ran all eight pipelines' | paper_text | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L105 | 'Checkpoints pinned in each spec' | paper_text | minor | wrong description | P10 | no | independent verification (this investigation) | wording only | n.a. |
| L106 | Data statement: logs and intermediates 'in the public repo' | paper_text | major | wrong description | P10 | no | human (external) | wording only | n.a. |
| L107 | Glossary: 'a whole cell is one position'; Adamson 'thousands silenced' | paper_text | minor | wrong description | outside checklist | partly | independent verification (this investigation) | wording only | no |
| L110 | Table 3 agent hours (topology '4+', manifold '24', caption) | paper_text | minor | wrong description | P10 | no | independent verification (this investigation) | wording only | n.a. |
| L111 | '15 hours of human supervision' has no record | paper_text | minor | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L112 | 'Figures plotted from run artefacts by reproducible code' | paper_text | major | wrong description | P10 | no | independent verification (this investigation) | wording only | n.a. |
| L117 | 'Ten-times-larger' 1B model; 'three audit categories left open' | paper_text | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |

### exhaustive-mapping (9)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L011 | Exhaustive-mapping hooks delete a whole block | run_outputs | critical | code bug | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L012 | Triplet hooks overwrite each other (ABC = C) | run_outputs | critical | code bug | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L013 | '2,980 feature triplets' (4 triplets) | paper_text | major | wrong description | P4 | no | human (external) | wording only | n.a. |
| L014 | Synergy test has no statistic or null; spec design cut | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L015 | Steering results are block-deletion artefacts | run_outputs | critical | code bug | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L016 | Steering has no random-feature null at any layer | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L017 | Exhaustive summary numbers that do not match files | run_outputs | minor | wrong statistic | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L018 | 'Mapped exhaustively' scope overstated | paper_text | minor | wrong description | P9 | no | independent verification (this investigation) | wording only | n.a. |
| L019 | No intervention sanity check or positive control (exhaustive mapping) | run_outputs | major | missing baseline or null | P6 | yes | audit sweep agent | pending re-run | yes |

### framework (audit) (10)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L090 | Audit matrix 61% -> 75%: no cell table; '2 untouched' wrong | run_outputs | critical | wrong statistic | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L091 | Audit graded its own same-day work; one session did everything | run_outputs | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L092 | Audit checklist built after, and from, this project's results | run_outputs | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L093 | Audit bookkeeping errors (A3 'DONE' with 2/5 skipped; 'independent' zero-shot) | run_outputs | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L096 | Audit duration 'two-to-three days' (it took one afternoon) | paper_text | major | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L097 | 'Two of three failures first noticed by human spot-checks' | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L098 | Bug-discovery story ('phrase search, then code', P3) | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L099 | 'Two errors in opposite directions' is the strongest evidence | paper_text | critical | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L109 | Review sets behind the checklist misattributed | paper_text | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L115 | Audit CI repair described as '1,000 bootstraps at the unit of inference' | paper_text | major | wrong description | P8 | no | independent verification (this investigation) | wording only | n.a. |

### framework (loops) (1)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L102 | Retirement rule and H115/H118 example | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |

### framework (specs) (4)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L086 | Deployed specs break the ten-section template | run_outputs | minor | reproducibility | P10 | yes | independent verification (this investigation) | wording only | no |
| L087 | Source-model pre-flight in spec Validation never run | run_outputs | minor | missing baseline or null | P6 | yes | independent verification (this investigation) | wording only | no |
| L100 | 'Template enforced; missing sections rejected' | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L103 | 'Mandatory positive-control slot' in the template | paper_text | minor | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |

### longevity-mechinterp (2)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L083 | Longevity negative has no positive control | run_outputs | minor | missing baseline or null | P6 | yes | audit sweep agent | pending re-run | yes |
| L084 | Longevity summary: 'L5 used by every other pipeline' | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |

### manifold-discovery (17)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L066 | Fifth pre-registered gate (permutation p) never computed | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L067 | Shuffled null passes two of the four gates | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | wording only | no |
| L068 | Trustworthiness described as a biological check | paper_text | major | wrong description | P3 | no | independent verification (this investigation) | wording only | n.a. |
| L069 | External 'branch-holdout' holds nothing out | run_outputs | major | wrong description | P4 | yes | independent verification (this investigation) | wording only | no |
| L070 | Lung negative control fails all four gates, reported as three | run_outputs | minor | wrong description | P6 | yes | independent verification (this investigation) | wording only | no |
| L071 | First negative control passed all gates and was rebuilt (paper silent) | paper_text | minor | unsupported claim | P5 | no | independent verification (this investigation) | wording only | n.a. |
| L072 | Zero-shot panel is not a separate cohort | run_outputs | major | wrong description | P4 | yes | independent verification (this investigation) | wording only | no |
| L073 | Manifold intervals: 80% anchor subsampling, not donor bootstrap | run_outputs | major | wrong statistic | P8 | no | human (external) | pending re-run | n.a. |
| L074 | Internal branch-holdout 0.370 carried by three small branches | run_outputs | major | wrong statistic | P8 | yes | independent verification (this investigation) | pending re-run | no |
| L075 | 3-gate fallback rule added after sweep 1 | run_outputs | major | unsupported claim | P5 | yes | independent verification (this investigation) | wording only | no |
| L076 | H103 zero-shot uses k=5 on 18 anchors | run_outputs | minor | wrong statistic | P8 | yes | independent verification (this investigation) | wording only | no |
| L077 | Two diverging manifold summaries; stale copy hides a failed test | run_outputs | minor | reproducibility | P10 | yes | independent verification (this investigation) | wording only | no |
| L078 | 'Compression costs no measurable quality' untested | run_outputs | minor | unsupported claim | P5 | yes | independent verification (this investigation) | wording only | no |
| L079 | Manifold numbers mislabelled (60 'genes', 2,400x base, 0.973->0.123, 'strongest axes') | paper_text | major | wrong description | P3 | no | independent verification (this investigation) | wording only | n.a. |
| L080 | Manifold sweep and gate count misdescribed (141 catalogue, H65 'winner', four gates) | paper_text | major | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L081 | 'Donors the model was never shown' | paper_text | minor | unsupported claim | P7 | no | independent verification (this investigation) | wording only | n.a. |
| L082 | Manifold run used an unrecorded second Python environment | run_outputs | minor | reproducibility | P10 | yes | independent verification (this investigation) | wording only | no |

### sae-atlas (13)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L020 | 0/48 TF-specificity test null-saturated, no chance baseline | run_outputs | major | missing baseline or null | P1 | yes | audit sweep agent | repaired during deployment | yes |
| L021 | TFs with no responding feature counted as negatives | run_outputs | major | wrong statistic | P6 | yes | repair-round agent | pending re-run | no |
| L022 | Phase 8t selection: layer mismatch, ~5 control cells, token-level test | run_outputs | major | code bug | P4 | yes | independent verification (this investigation) | pending re-run | no |
| L023 | 0/48 result cannot be reproduced (SAE overwritten, IDs not saved) | run_outputs | major | reproducibility | P10 | yes | independent verification (this investigation) | pending re-run | no |
| L024 | 48 TFs chosen alphabetically | run_outputs | minor | wrong description | P5 | yes | independent verification (this investigation) | pending re-run | no |
| L025 | 'Power' numbers are false-positive rates | run_outputs | major | wrong statistic | P6 | no | human (external) | wording only | n.a. |
| L026 | GATA1 'positive' fails a rarity/length-matched null | run_outputs | critical | missing baseline or null | P5 | no | independent verification (this investigation) | pending re-run | n.a. |
| L027 | A8 re-run selection code bugs (layer, controls, knife-edge) | run_outputs | major | code bug | P4 | no | independent verification (this investigation) | pending re-run | n.a. |
| L028 | Threshold search presented as a positive (p = 0.030 uncorrected) | run_outputs | major | unsupported claim | P5 | no | human (external) | wording only | n.a. |
| L029 | 'above_random' column wrong | run_outputs | minor | code bug | outside checklist | no | independent verification (this investigation) | pending re-run | n.a. |
| L030 | p_null computed as b/m while citing Phipson-Smyth | paper_text | minor | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L031 | Phase 6 applies the L5 SAE to hidden_states[6] | run_outputs | major | code bug | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L032 | Phase 6 ratio described as TF-target preference | paper_text | minor | wrong description | P3 | no | independent verification (this investigation) | wording only | n.a. |

### spectral-geometry (7)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L046 | Feature-shuffle null cannot show compression is learned | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L047 | CKA at L0 = 1 by construction; compared with scGPT's L0 | run_outputs | minor | wrong description | P3 | yes | independent verification (this investigation) | pending re-run | no |
| L048 | 'Three disjoint cell samples' overlap | run_outputs | minor | wrong description | P4 | yes | independent verification (this investigation) | wording only | no |
| L049 | Phase 8 JSON verdict contradicts the summary | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | wording only | no |
| L050 | Cross-model CI from 80% subsampling (docstring says bootstrap) | run_outputs | minor | wrong statistic | P8 | no | independent verification (this investigation) | pending re-run | n.a. |
| L051 | L11 effective rank measured after the final norm | run_outputs | minor | wrong description | outside checklist | yes | independent verification (this investigation) | pending re-run | no |
| L052 | Spectral autoloop: timeout passed validation; stop rule counts crashes | run_outputs | minor | code bug | outside checklist | yes | independent verification (this investigation) | wording only | no |

### topology-141 (16)

| id | error | layer | sev. | type | pattern | pre-audit | first found by | repair | sweep caught |
|---|---|---|---|---|---|---|---|---|---|
| L053 | scGPT cross-model CCA has no null (0.40 is at chance) | run_outputs | major | missing baseline or null | P1 | yes | independent verification (this investigation) | pending re-run | no |
| L054 | Procrustes retrieval fitted and scored on the same genes | run_outputs | minor | missing baseline or null | P5 | yes | independent verification (this investigation) | pending re-run | no |
| L055 | scGPT 'alignment breaks' despite retrieval z 19-31 | run_outputs | minor | unsupported claim | P9 | yes | independent verification (this investigation) | wording only | no |
| L056 | Geneformer called autoregressive / same family | run_outputs | minor | wrong description | P7 | yes | independent verification (this investigation) | wording only | no |
| L057 | iter_04 relabelled 'NOVEL POSITIVE' by manual review | run_outputs | major | unsupported claim | P9 | yes | audit sweep agent | repaired during deployment | partly |
| L058 | iter_04 replication verdict contradicts its numbers | run_outputs | minor | wrong description | P6 | no | repair-round agent | wording only | n.a. |
| L059 | H123 cross-validation leaked shared endpoints | run_outputs | minor | wrong statistic | P4 | yes | audit sweep agent | repaired during deployment | yes |
| L060 | H123 GroupKFold repair: all layers below 0.5; leakage claim untested | run_outputs | minor | unsupported claim | P4 | no | independent verification (this investigation) | wording only | n.a. |
| L061 | H123 negative had no positive control; repair control is one easy setting | run_outputs | minor | missing baseline or null | P6 | yes | audit sweep agent | repaired during deployment | yes |
| L062 | Lung H123 'anti-predicts' rests on one cell sample | run_outputs | minor | unsupported claim | P8 | yes | audit sweep agent | repaired during deployment | partly |
| L063 | H123 CIs resample the 12 layers | run_outputs | minor | wrong statistic | P4 | no | independent verification (this investigation) | wording only | n.a. |
| L064 | H139 null uses 8 permutations (spec 100-1,000) | run_outputs | minor | missing baseline or null | P1 | yes | independent verification (this investigation) | wording only | no |
| L065 | Topology 'autoloop' is a scripted batch; decisions changed by hand, unlogged | run_outputs | minor | reproducibility | P10 | yes | independent verification (this investigation) | wording only | no |
| L108 | '141 hypotheses, each screened at every layer' | paper_text | major | wrong description | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L114 | Cross-architecture scope credited to the audit | paper_text | minor | unsupported claim | outside checklist | no | independent verification (this investigation) | wording only | n.a. |
| L116 | Cross-model section overstated (replicates strongly, internal geometry, isolates cleanly) | paper_text | major | wrong description | P9 | no | independent verification (this investigation) | wording only | n.a. |

## 6. What differs from the deployed claims

- The corrected CRISPRi result (53.48%, 'decisively above chance') is at or below chance once the class balance is used as the baseline (L002). The 'five-fold CV' trained nothing (L003).
- The GATA1 'positive' does not survive a null that keeps gene rarity and length, and the same feature matches dozens of other TFs (L026). The 'power' figures are false-positive rates (L025).
- '2,980 triplets' were 4 triplets, and 'zero synergy' plus the 0.190 redundancy ratio are forced by hooks that overwrite each other (L012, L013).
- The 4.97M- and 2.14M-edge maps and the steering results come from hooks that delete a whole transformer block (L005, L011, L015); final numbers wait for the fixed-hook re-runs.
- The curveball null did not shuffle, so 'z = 0' is built in (L033). Residualised attention does not fall to chance in Adamson or 1B (L037).
- The audit took one afternoon, not days (L096); no record supports human spot-checks finding errors (L097); the sign bug was found by the agent during repair, not by the audit sweep (L001, L098); 75% 'fully addressed' has no cell table (L090).
- Hardware was an Apple-silicon laptop (MPS), not an A100 (L094).
- One correction to the investigation reports: audit_history section 4.5 says the 'barely changes' framing first appears at 19:27. The pre-audit EXTENDED_FINDINGS.md (16:37, line 26) already says the feature is 'barely changing at all' (see L057 notes).

## 7. Pre-audit snapshot for the blind re-audit (Study C)

- Location: `<EVAL_ROOT>/studyC/snapshot/maxtoki/` (1,862 files, 221.7 MB; built 2026-10-01T01:27:30; cutoff 2026-05-07T16:57).
- 6 summaries rebuilt by removing audit-added text: `runs/topology-141-217M/FINAL_SUMMARY.md`, `runs/longevity-mechinterp-217M/FINAL_SUMMARY.md`, `summaries/circuit-tracing-217M-FINAL_SUMMARY.md`, `summaries/sae-atlas-217M-FINAL_SUMMARY.md`, `summaries/manifold-discovery-217M-FINAL_SUMMARY.md`, `summaries/spectral-geometry-217M-FINAL_SUMMARY.md`.
- Not included: the project README and requirements.txt (edited or created during the audit; the README cannot be rebuilt), audits/, all paper folders, all audit_* scripts and outputs, and every file changed after the cutoff (269 files, including new re-run files). 343 large or duplicate files are listed (size and shape) in `_large_files_listing.tsv` files instead of copied.
- Full detail, including what was reconstructed and how: `studyC/SNAPSHOT_MANIFEST.md` (kept outside the snapshot).
- Leak scan over 1,708 text files: strong-term hits added 2026-05-07 6, reviewer 13, revision 5, investigation 5, 54.61 / 53.48 / 53.42 132, V2_ / v2 re-run names 43, off-by-one 1, spot-check 7. Each was checked by hand: all are pre-audit wording (spec text about an 'executor-reviewer loop', commit subjects, pre-audit May-7 notes) or number coincidences inside CSVs (for example 0.53422433; 'SV2_SV4' matching 'V2_'). Weak-term hits (for example 'audit' in 'strict max-null audit') are listed and judged in the manifest.

## 8. What was not done

- No model forward pass (GPU in use by the re-runs). The hook bugs' final impact is marked pending.
- The GATA1 matched nulls, the TRRUST variance baseline, the donor bootstrap and the CCA chance level were not re-run here; the rows cite the investigation scripts and outputs.
- The Adamson CRISPRa/CRISPRi label (L042) needs a literature check.
- Session transcripts of the deployment are gone, so 'first found by' for a few items rests on file times and docstrings.
- The pre-audit project README could not be rebuilt; it is left out of the snapshot. Errors that lived only in that README (parts of L085, L107) cannot be found by blind auditors.

## Plain-words summary

We listed 117 separate errors. 82 are in the study's own code, run summaries and audit files; 35 are only in the paper. 9 are critical: a headline claim cannot stand. 65 errors were already there before the audit began. The original audit sweep caught 7 of them fully and 3 in part. It missed all six critical ones: five are code bugs that no checklist pattern covers, and the sixth (judging accuracy against 50% instead of the class balance) is exactly the weak-null problem the checklist's P1 describes. The worst problems are hooks that delete whole model blocks, a sign that was never compared, hooks that overwrite each other, a null that never shuffled, and a 'positive' GATA1 result that fails a fair null. No record shows the human supervisor finding any error first. 7 rows wait for the model re-runs. We also built a pre-audit copy of the project for a blind re-audit: scripts, logs, small outputs and summaries as they were before the audit, with audit-added text removed and all audit and paper files left out.

## Verification notes

Independent check of this item, done on 2026-10-01 by a second agent. CPU only, no model run.

**Verdict: OK after small fixes.** The key numbers reproduce. The snapshot is a faithful, leak-free copy. I fixed six wording or precision issues in the ledger and two in the snapshot manifest. No count in sections 1-8 changed.

### How I checked

- Script: `framework-eval/ledger_verification/verify_independent.py`. Output: `verify_independent_results.json`. Inputs and their sha256: `run_config.json` (same folder).
- I did not reuse the first agent's code. Each number was recomputed a different way: other seed, other resampling routine, other parser, other pair of ratios.
- Before editing, I re-ran `build_ledger.py` in a scratch copy. It rebuilt the CSV and this report byte for byte.

### What I re-derived

- **CRISPRi (L001, L002).** Same 1,503,408 pairs from 248 silenced genes. I rebuilt the "went down" label from `actual_lfc`; it matches the stored label in every row.
  - Accuracy 0.5348. Always guessing "decrease" scores 0.5461. Balanced accuracy 0.4983. MCC -0.0055. The model says "decrease" for 89.6% of pairs.
  - Model minus the best constant rule: -1.13 points, 95% CI [-1.59, -0.72]. None of the 2,000 resamples is above 0.
  - Balanced accuracy minus 0.5: -0.17 points, 95% CI [-0.51, +0.13].
  - Model minus the level expected from the two class rates alone: 95% CI [-0.51, +0.12] points.
  - Method: percentile bootstrap. The resampling unit is the silenced gene (n = 248), drawn with replacement through multinomial weights. 2,000 resamples, seed 777.
  - These match the ledger ([-1.60, -0.73] and [-0.50, +0.13]) within resampling noise. Plain reading: the model has no directional skill.
- **Circuit edges (L005, L007).** I streamed the 79.9 MB CSV with Python's csv module.
  - 2,144,011 edges. 0 edges at the layer right after each source layer (sources 0, 3, 6, 9). The layer two steps on has 146,332, 103,673, 71,274 and 78,943 edges.
  - 2,144,011 / 52,116 = 41.1. Summary line 5 says "22x".
- **Exhaustive mapping (L011, L017).** I summed `edges_by_layer` myself instead of reading `total_edges`.
  - All 1,000 features have 0 edges at L6. L8 has 2,532,553 edges and L11 has 2,437,543.
  - Per feature: mean 4,970.1, SD 25.2, min 4,900, max 5,400.
- **Triplets (L012).** I solved the two free sizes from the AB and AC ratios; the ledger used AB and BC.
  - The other three ratios are predicted to within 6e-5 for triplets 0-2 and 3.6e-4 for triplet 3. That is within the rounding of the 4-decimal JSON values. The ledger's "within 1e-4" and "exactly" were slightly too strong; reworded (see changes).
  - The v1 file (`experiment2/`) stores only the mean pair ratio, so this test is not possible there.
- **Steering (L015, L017).**
  - alpha=5 over alpha=2 is 0.994-1.047 at every layer. The three features within a layer differ by at most 0.0015.
  - L0/L3 = 5.40 and L0/L11 = 19.98.
  - cos(g_early, g_late) = 0.881 recomputed from the npz. The run's own log also prints 0.8810, not 0.999.
- **Spectral samples (L048).** I took n = 592,317 from the run's own log line, not from the h5ad file. The three samples share 6, 8 and 8 cells. Two random 2,000-cell samples would share about 6.75 cells.
- **Curveball null (L033).** I copied the deployed function out of the script by line range (lines 221-247) and ran it with the deployed seeds 42-91. Null draws identical to the real matrix on the scored rows: 49/50 (217M K562), 37/50 (RPE1), 10/50 (Adamson), 46/50 (1B). Same as the ledger.
- **Text facts (L057, L067, L070).**
  - `EXTENDED_FINDINGS.md` line 26 (saved 16:37:40, before the audit) says "barely changing at all". So the ledger's correction to audit_history section 4.5 is right.
  - Lung control trust is 0.79956. The shuffled null scores trust 0.7985, random hold-out 0.799 and donor hold-out 0.707.
- **Ledger bookkeeping.** Every count in section 1 reproduces from the CSV: 117 rows; 65 errors present before the audit; the sweep caught 7 fully and 3 partly; 0 of the 6 critical pre-audit errors. All 15 required columns are present, and every E1-E42 item maps to a row.
- **Snapshot integrity.**
  - 1,862 files, 221.7 MB.
  - Every one of the 1,806 copied files is byte-identical to its source. No copied source was changed at or after 16:57. No file is extra and none is missing. The sha256 list matches. No listing row points to a file changed after the cutoff.
  - No `._*` file sits inside `snapshot/maxtoki`. One, `snapshot/._maxtoki`, sits one level up, outside the folder given to auditors.
- **Leak check.** I used my own term list and also scanned html and other text files (1,709 files). There were 0 hits for:
  - audit events;
  - later work (framework-eval, re-run file names);
  - paper folders;
  - known answers ("53.48%", always-decrease, block deletion, sign bug, null-saturated, "overwrite each other");
  - A100 or Opus;
  - dates after 2026-05-07.
- **Rebuilt summaries.** I also read all six:
  - Circuit and SAE: the cuts remove exactly the audit-marked blocks.
  - Spectral: every remaining difference from the 2026-04-17 copy is a May-3 Phase 8/9b section or the Section 8 validation table. The May-5 autoloop prompts already cite that table, so it is pre-audit.
  - Manifold: the "Phase 12 LITE" section left in the `summaries/` copy also appears in the pre-audit `runs/` copy (16:41).

### Changes I made

All rows below were edited in `ledger_build/ledger_rows.py`, then rebuilt with `build_ledger.py`.

1. L012 evidence: added that a different ratio pair gives residuals up to 3.6e-4, and that both fits are within rounding.
2. Section 2 of this report: "predicts the other three exactly" became "to within the 4-decimal rounding (max residual 1e-4; 3.6e-4 if solved from a different pair)". Changed in `build_ledger.py`.
3. L015 notes: the audit's P9 grade for exhaustive mapping is "acknowledged". The word "correct" is in the cell text, not the grade.
4. L084 evidence: the SAE atlas trains SAEs at every layer but runs its perturbation probes at L5. The old text said it "uses all layers".
5. L115 notes: the manifold part of this mismatch was raised first by an outside reader of the paper (see L073).
6. L007 notes: its pre-audit status is inferred, not proven. Only the post-audit file exists.
7. `SNAPSHOT_MANIFEST.md`, changed through `studyC/build/build_manifest.py` and rebuilt:
   - The area table counts only the 1,806 copied files; added a note on how this reaches 1,862.
   - The topology text said the cut "followed the task's rule literally". Only 1 of the 4 non-audit headings contains the literal marker "(added 2026-05-07". Reworded.
   - Added the evidence that the executor used such markers before the audit (the manifold summary of 16:41).
8. `build_ledger.py` now keeps this "Verification notes" section when it rebuilds the report.

The rebuilt CSV, report, `ledger_build/run_config.json`, manifest and `studyC/build/run_config.json` are on disk. Copies of the pre-edit files are in my scratch folder.

### Problems I did not fix

- **Topology summary choice (judgment call).** The snapshot drops four May-7 sections: H139, Phase 4 vs scGPT, the Phase 8 chain and the Phase 11 autoloop.
  - These are probably pre-audit. None mentions the audit, and their content matches `EXTENDED_FINDINGS.md` and `MANUAL_REVIEW.md` (16:37).
  - I recommend the version that keeps them (`studyC/build/alternatives/topology-141-217M_FINAL_SUMMARY.keep_non_audit_may7_sections.md`).
  - If you swap it in, also update `snapshot_inventory.json`, the sha256 list and the manifest.
- **Possible missing rows.** These are minor and only in the paper text, and none affects a run-output row:
  - Table 3 spec identifiers that match no file (numbers_A, paper lines 893-895).
  - "Surviving candidates" tested with dual-axis CV, when dual-axis CV ran only on H123 (lines 955-957).
  - Table 4 caption details (numbers_B, lines 1224-1231 and 1265-1266):
    - H103 external has 12 donors and 106 anchors;
    - the reasons given for the "three-gate fallback";
    - the lung control is called an "internal-panel construct only".
  - The Fig 1 loop wording (lines 564-568). L065 and L102 cover it in part.
- **L008 first finder.** The outside reader may have raised L008 ("different record sets"). The plan's summary asks for "identical records" but is too short to tell. Left as is.
- **L042.** My recollection is that Adamson et al. 2016 is a CRISPRi (dCas9-KRAB) screen. I did not check a source, so the row stays "needs literature check".
- **File dates.** The 50 listing files and the folders inside the snapshot are dated 2026-10-01, while the rebuilt files are set to 16:56. This shows when the copy was made. It does not reveal any answer.
- **Not re-run by me either:** the GATA1 matched nulls, the TRRUST variance baseline, the donor bootstrap and the CCA chance level.

**Plain-words summary.** I re-ran the main numbers in a different way, and they came out the same. The CRISPRi "skill" is still slightly worse than always guessing "down". The curveball null still copies the real network in 49 of 50 draws. The circuit maps still show no edges at the next layer. The snapshot holds only files from before the audit, each byte-identical to its source, and a word scan found nothing that gives the answers away. I fixed six small wording points in the ledger and two in the manifest. The main open choice is the topology summary: keeping its four May-7 sections looks more correct. A few small paper-only errors may be missing from the ledger.
