# Audit history reconstruction — MaxToki project (KEY = audit_history)

Scope: who found each audit error, when, what was repaired, and whether the
pre-repair files still exist. Also a check of the 8x10 verdict matrix numbers
(61% -> 75%, "2 cells untouched", "3% residual").

Paths below are absolute unless they start with `runs/`, `summaries/`,
`audits/`, `paper*/` or `README.md`; those are relative to
`<REPO_ROOT>/projects/maxtoki/`.

Labels: **[V]** = verified by reading a file or running a check.
**[I]** = my inference from the evidence.

---

## 0. Short answer

1. **The whole audit took one afternoon, not days.** [V] The audit spec, the
   audit, and both repair rounds were written on 2026-05-07 between 16:49
   and 21:10 local time, in one Claude Code session (`2a9284b9`). The audit
   report itself was written within 8 minutes of the prompt (17:30:17 ->
   17:38:25). The paper says "two-to-three days" / "roughly three days of
   agent compute".
2. **No record supports "two errors were first found by human spot-checks".**
   [V] The 83 human prompts in the MaxToki sessions (Apr 15 – May 8) contain
   no spot-check of the directional-accuracy number or of the TF-specificity
   result. The claim first appears in an agent-written file,
   `paper/LLM_USAGE_STATEMENT.txt` (2026-05-08 01:19). It entered the main
   paper text in the JBI version (2026-06-17).
3. **The sign bug was not found by the audit sweep.** [V] The audit report
   (17:38) and the first repair round (17:59, 18:08) treated 54.61% as a
   valid number. The bug first appears in a repair script written at 20:19
   (`runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py`). The
   agent found it while re-implementing Phase 11 to save per-pair records.
   That repair round started after the human asked "So all issues
   identified in the audit are now addressed?" (18:47).
4. **The corrected directional accuracy is not above chance.** [V, new] The
   corrected 53.48% is below the proper chance level. "Always predict
   inhibitory" scores 54.61%. A within-source label shuffle scores 53.59%
   (SD 0.018 points). The paper says the effect is "real but small". Under
   the right null, no effect is left. So the audit's own repair has a
   leftover error.
5. **The 61% (49/80) number can be reproduced. The 75% (60/80) number
   cannot.** [V] No cell-by-cell table exists for the later rounds. At least
   5 of the 8 "unaddressed" cells got no repair at all. So "only 2 cells
   left" cannot be right unless cells were re-graded with no new work.
6. **Pre-repair files mostly survive.** [V] The buggy code and the original
   outputs are all still on disk with April/early-May dates. Most summaries
   were edited in place, but the old text is still inside them, and the
   audit additions are marked "(added 2026-05-07 …)". A few files have
   no pre-repair copy (project README, spectral-geometry summary, longevity
   summary).

---

## 1. Evidence sources and their limits

- **Session transcripts are gone.** [V] No `.jsonl` for session `2a9284b9`
  (audit) or `862b8d39` (first paper) exists anywhere under `~/.claude` or
  `<LOCAL_VOLUME>/MacBook`. So I cannot see what the agent read, whether
  it used subagents, or what it said to the human.
- **Human prompts survive.** [V] `~/.claude/history.jsonl` (2,304 prompts,
  2026-01-06 to 2026-07-17) holds every typed prompt with a time stamp, a
  project path and a session id. Pasted text is sometimes stored and
  sometimes not. It stops on 2026-07-17, so the August PLOS ONE /
  Biosystems prompts are not in it.
- **File modification times survive.** [V] File mtimes are intact. Folder
  mtimes all read "Aug 12 15:52" (the tree was copied then), so folder
  dates mean nothing. The drive is exFAT: access times equal modification
  times, so they cannot show what was read. No Time Machine, no snapshots,
  no git history in `biomi_automation`.
- **Public repo.** [V] `Biodyn-AI/maxtoki-mechinterp` has one commit,
  `89da276`, dated 2026-05-07T22:48:16Z (00:48 local on May 8). That is
  after all repairs. It has `audits/`, `summaries/`, `paper/`. It has no
  `runs/`. So the buggy code is not public.
- Full list of project files changed on May 7:
  `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/may7_files.txt`.
- The 80 audit verdict cells, parsed from the report:
  `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/cells.json`.

---

## 2. Timeline (local time, 2026)

| When | Who | What | Evidence |
|---|---|---|---|
| Apr 15 – May 5 | agents, human prompts | 8 pipelines run in several sessions (`965ec1c8`, `8fb71b84`, `3af9282c`, `8be46b0f`, `b9b1ea22`) | history.jsonl; run mtimes |
| Apr 16 15:44 | agent | attention-grn scripts with BH correction written | `runs/attention-grn-217M/scripts/phase1_trivial_baselines.py` etc. |
| Apr 17 21:43 / 22:20 | agent | sae-atlas Phase 8t code + output "0/48 TF-specific" | `runs/sae-atlas-217M/scripts/remaining_phases.py` L340–470; `runs/sae-atlas-217M/outputs/phase8_true/summary.json` |
| Apr 20 11:05 / 11:13 | agent | circuit-tracing Phase 11 code with the sign bug; output 0.5461 | `runs/circuit-tracing-217M/scripts/remaining_phases.py` L242–338; `runs/circuit-tracing-217M/outputs/phase11_crispri_validation.json` |
| May 7 11:33–15:55 | human -> session `2a9284b9` | "review if everything in [topology-141] is done", "proceed with the previously out-of-scope items" | history.jsonl |
| May 7 16:13–16:20 | agent (`2a9284b9`) | topology-141 Phase 4 vs scGPT: the cross-architecture scope finding | `runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py`; `outputs/phase4_scgpt/phase4_summary.json` |
| May 7 16:24–16:37 | agent (`2a9284b9`) | Phase 11 autoloop; iter_04 re-labelled INCONCLUSIVE -> "NOVEL POSITIVE" | `runs/topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md` (16:37); `EXTENDED_FINDINGS.md` (16:37) |
| May 7 16:49:46 | human | asks for an audit pipeline from `review_plans/` | history.jsonl |
| May 7 16:57 | agent | audit spec written | `<REPO_ROOT>/pipelines/audit-recurring-review-issues.md` |
| May 7 17:30:17 | human | "Please now apply this pipeline to all research findings in projects/maxtoki. The summaries of research are present at projects/maxtoki/summaries" | history.jsonl |
| May 7 17:38:25 | agent | audit report (80 verdicts, A1–A8) | `audits/audit-20260507.md` |
| May 7 17:50:00 | human | "Ok, so please fix now all issues identified in the audit." | history.jsonl |
| May 7 17:52–18:09 | agent | repair round 1: A7, A4 (x2), A3, A1, A8 power, A2 H123, A2 status, completion v1 | see §3 |
| May 7 18:47:31 | human | "So all issues identified in the audit are now addressed?" | history.jsonl |
| May 7 18:48:59 | human | "Ok, so address all the remaining issues." | history.jsonl |
| May 7 18:50–21:10 | agent | repair round 2 (v2), incl. sign-bug fix at 20:19–21:09 | see §3 |
| May 7 21:10:46 | agent | completion v2 (61% -> 75% numbers first appear) | `audits/audit-20260507-completion-v2.md` |
| May 7 23:31 – May 8 01:33 | human -> `2a9284b9`, then `862b8d39` | first paper written; 5 review subagents | `paper/main.tex` (May 8 01:33) |
| May 8 00:48 | agent | public repo commit `89da276` | GitHub API |
| May 8 01:18:28 | human | generic "Write llm usage statement…" prompt | history.jsonl |
| May 8 01:19 | agent | LLM statement with the "first noticed during such spot-checks" claim | `paper/LLM_USAGE_STATEMENT.txt` L144–151 |
| Jun 17 00:13:55 | human | "Regarding A-C: A. We need to compare days of agent-compute. B. Roughly 3 days. C. Please check yourself." | history.jsonl (agent's questions not stored) |
| Jun 17 01:05 | agent | JBI version: spot-check claim moved into main text | `paper-jbi/main.tex` L382–395 |
| Aug 12 15:51 | agent | PLOS ONE version | `paper-plos-one/main.tex` L652–667, L1510–1595 |

---

## 3. How the audit was actually run

**Prompts.** [V] Two human prompts started it. 16:49:46 asked the agent to
read `review_plans/`, find recurring reviewer concerns "focus rather on
'spirit' than on the 'letter'", and write a new pipeline. 17:30:17 asked it to
apply that pipeline to `projects/maxtoki`, pointing it at `summaries/`. Two
more prompts drove the repairs (17:50:00; 18:48:59, after the 18:47:31
question).

**One session.** [V] All of it ran in session `2a9284b9`. The same session
had, earlier that day, done the topology-141 extension work (Phase 4 vs
scGPT, Phase 8 extensions, Phase 11 autoloop, the iter_04 re-labelling). The
audit then graded five topology-141 cells "clean (post-2026-05-07)". So the
auditor graded its own same-day work. The same session also did all
repairs. There was no split between finder, checker and fixer. [I] Whether
it used subagents is unknown (no transcript). Several v2 outputs land within
minutes of each other (19:14, 19:27, 19:28, 19:30), which fits background
jobs launched by one agent.

**Model.** The LLM statement says Claude Opus 4.7. Not checkable (no transcript).

**Duration.** [V]
- Spec: prompt 16:49:46 -> file 16:57.
- Audit report: prompt 17:30:17 -> file 17:38:25 (about 8 minutes, 36.7 KB).
- Repair round 1: 17:50:00 -> completion v1 18:09:25 (about 19 minutes).
- Repair round 2: 18:48:59 -> completion v2 21:10:46 (about 2 h 22 min). The
  heaviest jobs were the lung Phase-0 re-extraction at seed 43 (20:34 ->
  20:49) and the circuit re-run (script 20:19, outputs 21:09).
- Total from audit prompt to v2: **3 h 40 min wall clock**, one day.
- Paper claims: "1–2 days of agent time" (`paper/main.tex` L271) and
  "approximately three days of agent compute (against ~50 hours…), or
  roughly 6% audit overhead" (`paper/main.tex` L827; 72 h vs 50 h is not
  6%). Later: "two-to-three days" (`paper-plos-one/main.tex` L685) and
  "roughly three days" (L1648). The "three days" was first written by the
  agent on May 8. The human later answered "B. Roughly 3 days" (Jun 17). [I]
  The "1–2 days" matches the spec's own planning estimate ("sized to be
  doable in 1–2 days", spec §1 and §6.11), not a measurement.

**What it read.** [V] The report says (L7–11) it read each run's
FINAL_SUMMARY (via `summaries/`), run READMEs, the project README, and
"selected" JSON/scripts. Its own "Open items" (L438–444) say "this audit
stops at the markdown layer for most claims" and "~30 specific
load-bearing claims" were not checked at the JSON/script layer. The spec's
pitfall #2 warns against exactly this. Note: `summaries/` had no longevity
file; longevity was read from its run folder.

**Judgment vs scripts.** [V]
- The 80 verdicts are free-text agent judgments. They use 17 different
  labels ("clean — exemplary", "partial — flagged", "weak (and we did not
  add this in 2026-05-07 work)", …). No script produced them. The 49/23/8
  tally was done by the agent from these labels (mapping in §5).
- The spec says "No code dependencies — the audit is a methodology document
  applied by reading and writing markdown" (spec §5).
- Only the repairs were scripted:

| Action | Script (mtime) | Output (mtime) |
|---|---|---|
| A4 sae null | `runs/sae-atlas-217M/scripts/audit_a4_phase8t_chance_baseline.py` (17:55) | `runs/sae-atlas-217M/outputs/phase8t_chance_baseline/summary.json` (17:55) |
| A4 edge density | `runs/circuit-tracing-217M/scripts/audit_a4_edge_density_chance_baseline.py` (17:56) | `runs/circuit-tracing-217M/outputs/edge_density_chance_baseline/` (17:57) |
| A3 bootstrap v1 | `audits/audit_a3_bootstrap_cis.py` (17:59) | `audits/bootstrap_cis_summary.json` (17:59) |
| A1 synthetic | `runs/topology-141-217M/scripts/audit_a1_synthetic_positive_control.py` (18:01) | `runs/topology-141-217M/outputs/synthetic_positive_control/summary.json` (18:01) |
| A8 power | `runs/sae-atlas-217M/scripts/audit_a8_positive_tf_case_study.py` (18:04) | `runs/sae-atlas-217M/outputs/phase8t_positive_tf_case_study/summary.json` (18:04) |
| A2 H123 | `runs/topology-141-217M/scripts/audit_a2_groupkfold_h123.py` (18:06) | `runs/topology-141-217M/outputs/groupkfold_h123/` (18:07) |
| A3 spectral | `runs/spectral-geometry-217M/scripts/audit_a3_bootstrap_cross_model_pearson.py` (18:53) | `runs/spectral-geometry-217M/outputs/phase9b/bootstrap_pearson.json` (18:54) |
| A3 manifold | `runs/manifold-discovery-217M/scripts/audit_a3_bootstrap_h65_gates.py` (18:56) | `runs/manifold-discovery-217M/reports/external_validation_external_bootstrap.json` (19:14) |
| Residual #9 | `runs/topology-141-217M/scripts/audit_residual9_cross_layer_replication.py` (19:09) | `runs/topology-141-217M/outputs/phase11_autoloop/iter_04_replication/summary.json` (19:27) |
| A8 re-run | `runs/sae-atlas-217M/scripts/audit_a8_rerun_phase8t_targeted.py` (19:21) | `runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun/` (19:30) |
| Residual #6 | `runs/manifold-discovery-217M/scripts/audit_residual6_cell_level_benchmark_lite.py` (19:25) | `runs/manifold-discovery-217M/outputs/cell_level_benchmark_lite/` (19:38) |
| Residual #8 Louvain | `runs/topology-141-217M/scripts/audit_residual8_seed_stability.py` (19:27) | `runs/topology-141-217M/outputs/seed_stability/` (19:28) |
| A2 circuit (sign bug) | `runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py` (20:19) | `runs/circuit-tracing-217M/outputs/groupkfold_crispri/` (21:09) |
| Residual #8 Phase 0 | `runs/topology-141-217M/scripts/audit_residual8_phase0_seed43_lung.py` (20:34) | `runs/topology-141-217M/outputs/seed_stability_phase0/` (20:49) |

**The checklist was not independent of the project.** [V] The spec was
written at 16:57, by the same session, after the topology-141 work. Its
"Common manifestations here" already names this project's cases:
"topology-141-217M reports a clean negative on H123 … without a positive
control" (spec L415–417), "GATA1/MYC/TAL1 in K562 with ChIP-seq direct
targets" (L427–429), and "SAE Stage-2 CRISPRi 54.6% directional accuracy
described as 'near-chance…'" (L269–271, L542–543). It treats 54.6% as a real
number. For a blind re-audit, this spec cannot be used as-is.

**Inputs to the spec.** [V] `review_plans/geometric_structures_reviewer_comments.md`
is 0 bytes (created May 7 16:34). The spec still cites "[PLOS]" items; they
likely came from `review_plans/REVISION_PLAN_PLOS_ONE.md` [I].

---

## 4. Finding by finding

### 4.1 Directional-accuracy sign bug (circuit-tracing Phase 11)

- **Bug.** [V] `runs/circuit-tracing-217M/scripts/remaining_phases.py`
  L316–328: `predicted_inhibitory` is computed and never used; the count is
  `if (actual_lfc < 0): n_correct_direction += 1`. So "directional accuracy"
  = share of targets that went down.
- **First detected by: the repair agent, not the audit sweep, not a human.**
  [V] Evidence:
  - Audit report 17:38: circuit P3 = "acknowledged"; P1 says "CRISPRi 54.6%
    directional accuracy is itself a null comparison"; CT1 uses 54.6% as one
    of five converging negatives. No bug mentioned.
  - A3 17:59 (`audits/bootstrap_cis_summary.json`): Wilson CI on 54.61%,
    "a few SD above chance".
  - A2 status 18:08 (`audits/audit_a2_groupkfold_status.md` L62–90):
    "BLOCKED on per-pair data"; "The substantive finding … is robust".
  - First written mention of the bug: script docstring
    `audit_a2_groupkfold_crispri.py` L4 ("the original phase 11 dropped
    sign"), mtime 20:19:30. Output
    `runs/circuit-tracing-217M/outputs/groupkfold_crispri/summary.json`
    ("original_method": "buggy: just `actual_lfc < 0` regardless of
    predicted sign"), 21:09.
  - No human prompt between 18:48:59 and 23:31.
- **When.** Between 18:49 and 20:19 on 2026-05-07. [I] The agent went back
  to the BLOCKED A2 item, rebuilt per-pair records from `circuit_edges.csv`,
  and saw the sign was never compared.
- **Human role.** The 18:47 question ("So all issues … are now addressed?")
  triggered the round in which the bug was found. That is a nudge about
  completeness, not a spot-check of this number.
- **What the reports say.** v2 F8 (`audits/audit-20260507-completion-v2.md`
  L68–85): 54.61% -> 53.48%, per-source mean 53.42% [52.60, 54.16],
  "decisively above 50% chance".
- **Repair.** New script with sign tracking and GroupKFold by source gene.
  Summary appended at `summaries/circuit-tracing-217M-FINAL_SUMMARY.md`
  L96–115.
- **Numbers that do not match the JSON.** [V] v2 and the summary say "966
  unique source genes"; JSON and `per_source_accuracy.csv` have **248**. They
  say GroupKFold mean **53.55%**; JSON says **53.47%** (0.534730).
  `README.md` L135 (18:50) still says "54.6% near-chance directional".
- **Leftover error in the repair (new check).** [V] From
  `runs/circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet`
  (1,503,408 pairs): predicted inhibitory = 89.56%; observed down = 54.61%.
  - "Always predict inhibitory" accuracy = **54.61%** (higher than the model).
  - Independence expectation = **53.64%**.
  - Within-source shuffle of predicted labels (20 runs) = **53.59%**, SD 0.018 points.
  - Model = **53.48%**. Per-source mean minus per-source independence
    expectation = **-0.13 points**.
  - So the corrected number is at or slightly below chance. The 50% line
    used in v2 and the paper is the wrong chance level. The paper's "real
    but small effect survives" is not supported. Note also that the old
    54.61% equals the observed-down share, as expected from the bug.
- **Pre-repair artefacts.** [V]
  - Code: `runs/circuit-tracing-217M/scripts/remaining_phases.py` (Apr 20 11:05), unchanged.
  - Output: `runs/circuit-tracing-217M/outputs/phase11_crispri_validation.json` (Apr 20 11:13), `runs/circuit-tracing-217M/outputs/remaining_phases.log` (Apr 20 11:13).
  - Raw edges: `runs/circuit-tracing-217M/outputs/circuit_edges.csv` (79.9 MB, Apr 20 02:26).
  - Summary: no clean copy. `summaries/circuit-tracing-217M-FINAL_SUMMARY.md` was edited (last 21:10) but keeps the old lines (L67, L81, L93, L134, L159). Audit blocks are L7–17 and L96–115.
  - Other pre-audit files quoting 54.6%: `summaries/attention-grn-217M-FINAL_SUMMARY.md` (Apr 23), `summaries/topology-141-217M-FINAL_SUMMARY.md` (May 5).
  - Not in the public repo (no `runs/`).

### 4.2 GATA1 / TRRUST: underpowered TF-specificity test (sae-atlas Phase 8t)

- **Original claim.** [V] "0/48 TF-specific" with a ≥2-of-top-20 overlap
  test against TRRUST, 20 cells per target
  (`runs/sae-atlas-217M/scripts/remaining_phases.py` L450–455;
  `runs/sae-atlas-217M/outputs/phase8_true/summary.json`,
  `perturbation_response.csv`, Apr 17 22:20).
- **First flagged by: the audit agent.** [V] Audit report L85 (P1: "0%
  TF-specificity headline lacks an explicit chance-distribution baseline")
  and L202 (P6: "no positive-control TF … e.g., GATA1 with ChIP-seq direct
  targets in K562"), 17:38.
- **Where the idea came from: earlier human peer review.** [V]
  `review_plans/REVISION_PLAN.md` L13–14 (Bioinformatics review R1-M2/R1-M3:
  "use ChIP-seq direct targets in matched cell type, not merged TRRUST",
  file dated Apr 17) and `review_plans/nmi_revision_plan.md` L334 (names
  GATA1, MYC, TAL1; Apr 13). The agent copied this into the spec (L427–429)
  before the audit. So this is human-reviewer knowledge from other papers,
  applied by the agent. Not a human spot-check of this project.
- **Confirmed by: the repair agent.** [V]
  - A4 (17:55): random-target null gives mean n_specific 0.005, q95 = 0 ->
    "null-saturated".
  - A8 power (18:04): TRRUST power at K=5 = 0.004 vs ChIP-seq 0.621.
  - A8 re-run (19:30): GATA1 features [628, 1334, 2006, 2610, 2627]; feature
    2610 has 8/20 overlap with 470 ChIP-seq targets; p_null = 0.030 at ≥5;
    ≥2 gives p_null 0.602.
- **Repair.** Re-framed in `summaries/sae-atlas-217M-FINAL_SUMMARY.md`
  L114–200 ("major reframe"), v2 F7.
- **Caveats found.** [V]
  - The re-run is a different test: 50 GATA1 cells and 200 control cells,
    not 20 cells per target (`phase8t_positive_tf_rerun/run.log`).
  - 2 databases x 4 thresholds were tried; the one "positive" is at the
    strictest threshold, p = 0.030 uncorrected. Bonferroni over 4 thresholds
    gives 0.12. The paper's Limitations do call it "post-hoc-selected".
  - `specificity_with_null.csv` column `above_random` is True even where
    p_null = 0.602. The column is wrong.
  - MYC: "only 0 K562 cells"; TAL1: "not in K562 perturbation list"
    (run.log). So n = 1 TF.
- **Pre-repair artefacts.** [V] Code and outputs above are unchanged.
  `summaries/sae-atlas-217M-FINAL_SUMMARY.md` edited 19:31, but old text is
  kept (L5, L64–71, L86, L94, L109–112). Clean pre-audit copies:
  `summaries/sae-atlas-217M-FINAL_SUMMARY_12layer.md` and
  `summaries/sae-atlas-12layer-global-summary.json` (Apr 19). The original
  run did not save responding-feature IDs, so the exact original features
  cannot be recovered.

### 4.3 Missing bootstrap CIs (P8)

- **First flagged by: the audit agent.** [V] P8 table: 7 of 8 runs
  "partial", only attention-grn "clean" (report L245–258). The paper says
  "Six of the eight". The report's A3 lists 5 target numbers.
- **Repair.** [V] v1 (17:59) computed 3 of 5; spectral and manifold were
  "SKIPPED" (`audits/bootstrap_cis_summary.json`), yet v1 marked A3 "DONE".
  v2 added spectral (18:54) and manifold (19:14).
- **How it was actually done.** [V] Not "1,000 non-parametric bootstrap
  iterations on the unit of inference" (paper):
  - sae Phase 6: 10,000 resamples of 50 features. OK.
  - topology H123: 10,000 resamples of the **12 layers** (not TFs or genes).
  - circuit: a Wilson interval, no bootstrap, on the buggy 54.61%.
  - spectral: 1,000 **80% subsamples without replacement**, not rescaled.
    This makes the CI too narrow [I].
  - manifold: **200** 80% subsamples of anchors.
  - exhaustive-mapping and longevity: nothing.
- **Result change.** None qualitative. [V] Numbers match JSONs (0.382
  [0.380, 0.384]; branch 0.346 [0.270, 0.480]; Phase 6 median 1.015
  [1.011, 1.019]).
- **Pre-repair artefacts.** The original point-estimate outputs are intact.
  The spectral pre-audit summary is not preserved
  (`summaries/spectral-geometry-217M-FINAL_SUMMARY.md` edited 18:54;
  `runs/spectral-geometry-217M/outputs/FINAL_SUMMARY.md`, Apr 17, is older
  than the Phase 9b work). Manifold pre-audit summary is preserved:
  `runs/manifold-discovery-217M/FINAL_SUMMARY.md` (May 7 16:41).

### 4.4 Cross-architecture scope (P7)

- **First found by: the executor agent, before the audit.** [V] Phase 4 vs
  scGPT ran 16:13–16:20, at the human's request to do "the previously
  out-of-scope items" (15:55). Written up in
  `runs/topology-141-217M/EXTENDED_FINDINGS.md` (16:37).
- **Audit role.** [V] CT3 (report L343–355) noticed the qualifier was
  missing from the project README. A5 added it (`README.md`, 18:50).
- **So the paper overstates it.** "The audit re-derived the
  cross-architectural-family scope qualifier" — the audit only copied it up.
- **Caveat.** [V] `runs/topology-141-217M/outputs/phase4_scgpt/phase4_summary.json`:
  scGPT top-1 retrieval z = 19.5–30.8 above null in all 3 domains. Only
  pairwise Pearson is ≈ 0. "Agreement breaks" is too strong for retrieval.
- **Pre-repair artefacts.** Pre-A5 `README.md` is not preserved. The finding's
  own files are intact.

### 4.5 Cross-layer framing (iter_04; P9)

- **Origin of the strong framing.** [V] The executor re-labelled iter_04
  from INCONCLUSIVE to "NOVEL POSITIVE" at 16:37: "single static axis —
  residual stream doesn't reorganise gene similarities through layers"
  (`runs/topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md` L6,
  L49). The script's own decision was "INCONCLUSIVE"
  (`iter_04_cross_layer_consistency/report.json`).
- **Audit role.** [V] The audit listed it as residual exposure #5: "should be
  replicated before entering any paper" (report L429–432). That is a
  replication request, not a language flag.
- **Repair.** [V] v2 replication (19:27). Its auto-written verdict says
  "GENERALISES … All three feature types are above 0.6 … barely changes
  across layers" while its own numbers are 0.645 / 0.561 / 0.275
  (`iter_04_replication/summary.json`). The agent then wrote the correct
  "partially feature-specific" reading in
  `runs/topology-141-217M/FINAL_SUMMARY.md` L315–317 (20:50) and v2 F11.
- **So** the "barely changes" phrase the paper says the audit corrected was
  mostly the repair script's own wrong verdict string. [V] "barely change"
  appears nowhere before 19:27.
- **Leftover issue.** [V] The synthetic check in the same JSON gives ≈ 0.000
  for all three features, yet the verdict says "The cross-layer-Pearson
  metric IS sensitive … (synthetic confirms)". That does not follow.
- **Pre-repair artefacts.** `MANUAL_REVIEW.md` and `EXTENDED_FINDINGS.md`
  (16:37) are intact. `runs/topology-141-217M/FINAL_SUMMARY.md` L474–479
  keep the old text.

### 4.6 H123 lung seed sensitivity (P8)

- **First flagged by: the audit agent, generically.** [V] Residual #4:
  "Single-seed extraction … sensitivity to data sampling is unknown" (L425–428).
  v1 left it "unchanged".
- **Found by: the repair agent.** [V] Louvain seeds 42–46 on immune: max
  range 0.034 (19:28). Phase-0 re-extraction at seed 43 on lung (20:34–20:49):
  mean |ΔAUROC| 0.081, max 0.101; seed-42 mean 0.413 vs seed-43 0.494
  (`runs/topology-141-217M/outputs/seed_stability_phase0/summary.json`).
- **Result change.** The lung "anti-predicts" story becomes "near chance".
  Mean Δ −0.041 called "partly a seed-42 artifact". Only 1 of 3 tissues tested.
- **Pre-repair artefacts.** [V] Seed-42 outputs intact:
  `runs/topology-141-217M/outputs/phase78/h123_signed_motif_degree_strata.csv`
  (May 3 20:31). Pre-audit summary copy: `summaries/topology-141-217M-FINAL_SUMMARY.md`
  (May 5 17:10).

### 4.7 Benjamini–Hochberg correction (attention-grn)

- **Not an audit finding.** [V] BH was in the attention-grn code and outputs
  from 2026-04-16 (scripts 15:44; `phase12_verdict*.json`; `summaries/attention-grn-217M-FINAL_SUMMARY.md`
  L29, L36, Apr 23). The audit rated attention-grn "clean — exemplary …
  Wilcoxon BH-corrected p-values everywhere" (L84, L248). No attention-grn
  file changed on May 7. The paper's "the audit re-derived … an explicit
  Benjamini–Hochberg correction … (P8)" (`paper-plos-one/main.tex` L1581–1583)
  has no support in the audit files.

### 4.8 Other audit changes to results

| Item | Flagged by | Repaired by (time) | Result change | Pre-repair files |
|---|---|---|---|---|
| A2 H123 GroupKFold | audit P4 (L158) | agent 18:07 | immune AUROC 0.505 -> 0.462; "1/12 positive was leakage" | `outputs/phase78/` intact |
| A4 edge density | audit P1/P2/P8 (L86, L108, L250) | agent 17:57 | "41x" -> "14.6x–41x, threshold-sensitive" | `runs/circuit-tracing-217M/outputs/circuit_summary.json` (Apr 20) |
| A4 Phase 8t null | audit P1 (L85) | agent 17:55 | 0/48 "indistinguishable from chance" | as 4.2 |
| A1 synthetic control | audit P6 (L206) | agent 18:01 | none (supports H123 code) | — |
| Residual #6 cell-level lite | audit residual #2 | agent 19:38 | no feature above chance (192 cells) | — |
| A5/A6 wording | audit CT3, P9 | agent 18:50 / 18:54 | wording only | not preserved (in-place edits) |

Caveats [V]: The H123 dual-axis AUROC is below 0.5 at all 12 layers
(0.448–0.476) on only 76 test pairs per layer. A systematic below-chance
value hints at a bias in the split construction, not only removed leakage
[I]. The null-gap metric behind "1/12 positive" was never re-run under
GroupKFold, so "the immune positive layer was leakage" is inferred, not
tested. A4 edge density used a parametric noise model (d ~ N(0, 0.235)),
not the degree-preserving null the audit asked for. The A1 positive control
used one very easy setting (ARI 1.000, AUC 1.000); there is no weaker-signal
test.

---

## 5. Human vs agent: what the record shows

| Error | Human spot-check? | Audit agent (17:38 sweep) | Repair agent | Origin of the idea |
|---|---|---|---|---|
| Sign bug | **No record** | Missed it; used 54.6% as valid | **Found it** 18:49–20:19; fixed 21:09 | re-implementation during A2 |
| TRRUST / GATA1 | **No record** | **Flagged** (P1, P6) | Confirmed (A4, A8), re-tested 19:30 | earlier human peer review (REVISION_PLAN.md R1-M3), copied into spec |
| Bootstrap | No record | **Flagged** (P8) | Computed (partly) | reviewer catalogue |
| Cross-architecture | No record | Found README gap (CT3) | Finding made before audit (16:20) | human asked for "out-of-scope items" |
| Cross-layer framing | No record | Asked for replication | Over-claimed at 16:37 and 19:27, corrected 20:50 | agent's own text |
| H123 seed | No record | Flagged generically | **Found** the sensitivity 20:49 | reviewer catalogue |
| BH | — | Rated already clean | none | pre-existing (Apr 16) |

Human actions that did matter [V]: asking for the audit (16:49, 17:30),
pointing it at `summaries/`, ordering repairs (17:50), and doubting
completeness (18:47) — which led to round 2, where the sign bug and the
seed sensitivity were found.

Where the spot-check claim came from [V]: first written by the agent in
`paper/LLM_USAGE_STATEMENT.txt` L144–151 (May 8 01:19), right after a
generic prompt (01:18:28) that gave no such fact. Not in `paper/main.tex`
(May 8) or `paper-deanon/main.tex` (Jun 16). Added to the main text in
`paper-jbi/main.tex` L382–395 (Jun 17). No prompt in history.jsonl states
it. [I] Pasted text at May 7 23:31 in `2a9284b9` was not stored; it came
after all repairs, so it cannot show earlier human detection. A reviewer's
four-way split (human discovery / agent discovery / agent confirmation /
repair) is answered by the table above. For the two headline errors: sign
bug = agent discovery during repair; TF specificity = agent audit flag
based on earlier human reviewer knowledge, then agent confirmation and repair.

The paper also says "The audit agent searched the run summaries for the
phrase 'directional accuracy' and then read the code behind it"
(`paper-plos-one/main.tex` L1540–1543; first in `paper/main.tex` L730–733).
The audit files contradict this (§4.1). The LLM statement says the audit
"ran autonomously after each pipeline batch" (L122–124). It ran once, after
all pipelines.

---

## 6. The 8x10 verdict matrix

**v0 (before repair): 49 / 23 / 8 is reproducible.** [V] Parsing the 10
tables in `audits/audit-20260507.md` gives exactly 80 cells. This mapping
gives 49 / 23 / 8:

| Bucket | Labels (count) | Total |
|---|---|---|
| "Clean / acknowledged-and-handled" | clean — exemplary (18), clean (12), clean (post-2026-05-07) (5), acknowledged (6), acknowledged with strong handling (2), acknowledged via Stage-3 cross-reference (2), mostly clean (2), strong (1), acknowledged scope (1) | **49** |
| "Acknowledged with gap / partial" | partial (13), partial — flagged (4), acknowledged with gap (4), acknowledged with carryover risk (1), mixed (1) | **23** |
| "Unaddressed" | weak (6), weak (and we did not add…) (1), unaddressed (1) | **8** |

Notes:
- 11 of the 49 are "acknowledged" (limits stated, not fixed). The report
  calls the bucket "clean / acknowledged-and-handled". The paper calls it
  "fully addressed". That is a stronger word.
- 5 of the 49 are topology-141 cells graded "clean (post-2026-05-07)", i.e.
  after same-day work by the same session. The "before audit" bar already
  contains that work.
- The 8 unaddressed cells: circuit P2; P6 for sae, circuit, exhaustive,
  topology, longevity; P7 for circuit and exhaustive.
- The report says these 8 "correspond to 6 distinct action items" (L50–51).
  Only 3 have an action: topology P6 (A1), sae P6 (A8), circuit P2 (A4,
  partly). The other 5 have none.
- P4 text says "Five of eight runs" fail grouping (L161); the table has 6
  non-clean P4 cells.

**v1 (68 / 26 / 6) and v2 (60 / 18 / 2) cannot be checked.** [V]
- No per-cell table exists for v1 or v2 anywhere in the project. The only
  source is two lines in `audits/audit-20260507-completion-v2.md` L187–190.
  The figure in `paper-biosystems/main.tex` L1609–1653 hard-codes these
  percentages (61/68/75, 29/26/22, 10/6/3).
- The v1 report (`audit-20260507-completion.md`) gives no cell counts. It
  gives pattern counts "6/10 clean (was 5), 3/10 partial (was 4), 1/10 weak
  (was 1)". The audit itself said "weak on P4 and P6" (2, not 1).
- 68% x 80 = 54.4, 26% = 20.8, 6% = 4.8. Only 54/21/5 fits after rounding.
  [I] The v1 split was likely estimated, not counted.
- 60/18/2 = 75.0% / 22.5% / 2.5%. Rounding gives 75/22/3 (22.5 rounded
  down, 2.5 rounded up, so it sums to 100).

**v2's "2 unaddressed" conflicts with the repair log.** [V/I]
- At least 5 of the 8 v0 "unaddressed" cells got no repair: circuit P6,
  exhaustive P6, longevity P6, circuit P7, exhaustive P7. So at least 5
  should still be unaddressed, unless they were re-graded with no new work.
- The three "residual" items named for the last 3% (AIDA, scVI/Palantir/
  CellTypist, cross-model H123) sit in cells that v0 already graded
  **clean**: longevity P7 ("acknowledged scope"), manifold P2
  ("acknowledged with strong handling"), topology P7 ("clean (post)").
- Three items are described for "2 cells".

**"2 untouched" vs "0 untouched" vs "20 not fully addressed".** [V]
- v2 L183: "4 items remain partial; 1 closed; 0 untouched".
- v2 L26: "10 fully done, 1 partial, 0 unaddressed" — for 13 items (8
  actions + 5 residuals). 10 + 1 = 11, not 13.
- v2 L30 says "The only remaining partial item (residual #6)", but L183
  says 4 items remain partial.
- v2 table re-grades v1's A3 as "PARTIAL (3/5)"; v1 itself said "DONE".
- The paper's caption: "75% (60 of 80), with only 2 cells left untouched.
  The residual 3% …". "Untouched" is not an audit category. The 2 are the
  "unaddressed" cells. 80 − 60 = **20 cells (25%) are still not fully
  addressed** (18 partial + 2 unaddressed). The caption reads as if only
  2 cells remain open.

---

## 7. Other problems in the audit files (for the re-audit experiment)

1. Corrected directional accuracy is at chance under the right null (§4.1).
2. 966 vs 248 source genes; 53.55% vs 53.47% (§4.1).
3. `above_random` column wrong (§4.2).
4. iter_04 replication verdict string contradicts its own numbers; synthetic
   check does not show sensitivity (§4.5).
5. `README.md` still says 54.6% after the correction.
6. Bootstrap methods differ from what the paper says (§4.3).
7. BH claimed as an audit product; it was not (§4.7).
8. Audit time: hours, not days (§3). "6% overhead" arithmetic in
   `paper/main.tex` L827–829 is wrong.
9. The audit spec already names this project's cases (§3).
10. The auditor graded its own same-day work (§3).
11. scGPT "alignment fails" ignores retrieval z = 19–31 (§4.4).
12. H123 dual-axis values all below 0.5 on 76 pairs (§4.8).

---

## 8. Pre-repair artefacts for a blind re-audit

| Item | Pre-repair file | State |
|---|---|---|
| circuit Phase 11 code (bug) | `runs/circuit-tracing-217M/scripts/remaining_phases.py` | intact, Apr 20 |
| circuit Phase 11 output | `runs/circuit-tracing-217M/outputs/phase11_crispri_validation.json`, `remaining_phases.log` | intact, Apr 20 |
| circuit edges / summary | `runs/circuit-tracing-217M/outputs/circuit_edges.csv`, `circuit_summary.json`, `circuit_trace.log` | intact, Apr 20 |
| circuit summary text | `summaries/circuit-tracing-217M-FINAL_SUMMARY.md` | edited; strip "(added 2026-05-07 …)" blocks L7–17, L96–115 |
| sae Phase 8t code | `runs/sae-atlas-217M/scripts/remaining_phases.py` L340–470 | intact, Apr 17 |
| sae Phase 8t output | `runs/sae-atlas-217M/outputs/phase8_true/` | intact, Apr 17 |
| sae Phase 6 output | `runs/sae-atlas-217M/outputs/phase6/causal_patching.csv` | intact, Apr 18 15:22 |
| sae summary | `summaries/sae-atlas-217M-FINAL_SUMMARY.md` | edited; strip L114–200; clean 12-layer copy (Apr 19) |
| topology H123 seed 42 | `runs/topology-141-217M/outputs/phase78/*` | intact, May 3 |
| topology summary | `summaries/topology-141-217M-FINAL_SUMMARY.md` | clean pre-audit copy (May 5), but older than May 7 morning work |
| topology May-7 pre-audit docs | `runs/topology-141-217M/EXTENDED_FINDINGS.md`, `outputs/phase11_autoloop/MANUAL_REVIEW.md` | intact, 16:37 |
| topology current summary | `runs/topology-141-217M/FINAL_SUMMARY.md` | edited 20:50 |
| manifold summary | `runs/manifold-discovery-217M/FINAL_SUMMARY.md` | clean pre-audit copy, 16:41 |
| attention-grn | `summaries/attention-grn-217M-FINAL_SUMMARY.md`, `runs/attention-grn-217M/outputs/FINAL_SUMMARY.md` | never touched by audit |
| exhaustive-mapping | `summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md` | never touched (Apr 23) |
| spectral summary | `summaries/spectral-geometry-217M-FINAL_SUMMARY.md` | edited in place (A6); **no clean copy** |
| longevity summary | `runs/longevity-mechinterp-217M/FINAL_SUMMARY.md` | 18:50; **no clean copy**; `outputs/stage1_20260505/report.md` (11:40) is pre-audit |
| project README | `README.md` | edited in place 18:50; **no clean copy** |

Files that leak the answers and must be hidden from a blind auditor [V]:
`audits/*`; all `audit_*.py` scripts and their output folders in `runs/`;
`<REPO_ROOT>/pipelines/audit-recurring-review-issues.md`
(names GATA1 ChIP-seq and H123 synthetic control); all `paper*/` folders;
`README.md` L135; the "(added 2026-05-07 …)" blocks in summaries; and the
public repo `89da276` (has audits and post-repair summaries).
Also hide `review_plans/REVISION_PLAN.md` and `nmi_revision_plan.md` if the
test is whether the auditor finds the TRRUST issue on its own.

---

## 9. Plain-words summary

The audit, both repair rounds, and the audit spec were all done by one
agent session in one afternoon (about 3 h 40 min). The paper says days. The
audit sweep itself read mainly the markdown summaries. It missed the sign
bug. The agent found the bug later, while re-doing a blocked repair, after
the human asked whether everything was fixed. The TF-specificity problem was
flagged by the audit agent. But the idea came from earlier human peer
reviews, which the agent had copied into the checklist. No record shows a
human spot-check finding either error. That claim was first written by the
agent in the LLM usage statement. The BH correction was never an audit
finding. The 61% (49/80) figure can be rebuilt from the report. The 75%
(60/80) figure cannot. At least 5 "unaddressed" cells never got a repair,
and 20 cells, not 2, remain less than fully addressed. The repaired
directional accuracy (53.48%) is itself at chance under the correct null.
The original buggy code, the original outputs, and most of the original
summary text are still on disk, so a blind re-audit is possible. The audit
spec, audit reports and papers must be hidden from the blind auditor.
