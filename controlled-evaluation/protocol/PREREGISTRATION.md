# Pre-registration — controlled evaluation of executable specifications and a reviewer-derived audit checklist for agent-run interpretability

Frozen before any subject agent is run. Task-specific answer keys are frozen
separately (one file per task or item); their sha256 hashes are written to
`protocol/FREEZE_LOG.md` before the corresponding study is run.

## 1. Questions

- **Q1 (Study A).** For the same agent model, tools, data, task statement and
  source-method material, does adding the markdown pipeline specification
  ("contract") make the agent's final scientific conclusion more often correct?
- **Q2 (Study A).** Applied to the same executor outputs, does a review using
  the reviewer-derived ten-pattern checklist, followed by repair, produce
  correct conclusions more often than (a) no review and (b) a generic critical
  review of the same budget, followed by the same repair step?
- **Q3 (Study B).** On analyses that were not used to build the checklist,
  how often does a checklist review find a real error, how often does it raise
  a false alarm on a correct analysis, and how does it compare with a generic
  review of the same budget?
- **Q4 (Study C).** On a blind copy of the MaxToki deployment as it stood
  before its audit, which of the errors later documented in the error ledger
  are found by (a) the checklist as deployed (with project-specific examples),
  (b) the generic checklist, (c) a generic review?

## 2. Subjects and context control (all studies)

- Every subject agent (executor, reviewer, repairer, grader) is Claude Opus 5.5
  (`claude-opus-5-5`), run as a Claude Code workflow subagent of the built-in
  read-only type with the model set explicitly to Opus. This agent type starts
  without project instruction files and without a memory index (verified by a
  probe on 2026-10-01). It can read files and run shell commands, but it does
  not write files: analyses run as inline Python, and all deliverables are
  returned as structured output. The harness writes them to disk verbatim from
  the workflow journal.
- Python interpreter: `maxtoki-framework-eval/bin/python` (numpy 1.26.4,
  pandas 2.3.3, scikit-learn 1.8.0, scipy, torch 2.11.0 — CPU only for subjects).
- Each subject is told to read only inside the folders its prompt names. After
  the runs, every transcript is scanned for file reads outside those folders.
  A run that read the repository, another task, another item, an answer key or
  the paper is excluded and repeated once; the number of exclusions is reported.
- Reasoning effort is not overridden; it is the session default for all arms.

## 3. Study A — 2 × 3 factorial on four analysis tasks

### Arms
- Executor factor:
  - `paper`: task brief + data + source-method paper (PDF) of the method's
    original study.
  - `contract`: the same, plus the deployed pipeline specification for that
    method (the version dated 2026-04-15, as used in the deployment).
  The brief, data, deliverable format, compute limits and instructions are
  word-for-word identical across the two arms except for one sentence pointing
  to the specification.
- Review factor (applied to every executor output):
  - `none`: the executor output is final.
  - `generic`: a fresh reviewer reads the task package and the executor's
    deliverable and reviews it critically for errors and weaknesses; a fresh
    repairer then revises the deliverable given the review.
  - `checklist`: the same, except the reviewer applies the generic ten-pattern
    checklist (`protocol/CHECKLIST_GENERIC.md`).
  Reviewer prompts carry the same budget sentence and the same output schema.
  The repair prompt is identical across the two review arms.
- Replicates: 5 independent executor runs per task × executor arm.
  Total executor runs: 4 tasks × 2 arms × 5 = 40. Each yields three final
  deliverables (none / generic / checklist): 120 graded deliverables.

### Tasks (data come from the MaxToki-217M deployment, after correction)
- T1: Do attention-derived scores recover curated regulator–target pairs
  (TRRUST) in the 217M RPE1 run?
- T2: Do SAE-circuit-derived predictions get the direction of change after a
  CRISPRi knockdown right (K562)?
- T3: Are SAE features that respond to a transcription factor's knockdown
  specific to that factor's target genes (K562)?
- T4: Does a developmental (hematopoietic stage) ordering read out from the
  model's internal state transfer to donors not used to fit it?

### Outcomes
- **Primary:** verdict correct (1/0) against the frozen answer key.
- Secondary:
  1. trap score: fraction of the task's pre-listed traps handled correctly;
  2. key numbers within the key's tolerance (deterministic, from the returned
     `results` object where the key defines a matching quantity);
  3. number of materially false statements in the report;
  4. reviews: number of findings; findings judged valid / invalid (false alarm)
     against the key;
  5. repair: wrong→right and right→wrong verdict changes;
  6. reproducibility: the returned analysis script is run by the harness on
     the task data; numbers in `results` are reproduced within tolerance (1/0);
  7. cost: tokens, tool calls, wall time per agent from transcripts;
  8. human interventions: none by design (recorded as 0).

### Grading
- Two independent graders per deliverable (same agent type and model), each
  given the task brief, the answer key and the deliverable; neither is told the
  arm. Before grading, the harness redacts arm-revealing words from the
  deliverable (case-insensitive: specification, spec, contract, checklist,
  audit, reviewer, review, repair, revised, pattern P1–P10 labels). Graders
  score verdict, traps and false statements with written justification.
  Disagreement on the primary outcome → a third grader decides. Agreement is
  reported (Cohen's kappa on the primary outcome; ICC or weighted kappa on the
  trap score).

### Analysis
- Primary contrasts (Holm-corrected across the four):
  1. contract vs paper at review = none (unpaired; Fisher exact test;
     risk difference with Newcombe 95% CI; also a task-stratified
     Cochran–Mantel–Haenszel test);
  2. checklist vs generic (paired on the same executor output; exact McNemar);
  3. checklist vs none (paired; exact McNemar);
  4. generic vs none (paired; exact McNemar).
- Secondary outcomes: descriptive with 95% CIs (Wilson for proportions;
  bootstrap over executor runs, stratified by task, for means).
- All results are reported, whatever their direction. No arm, task or outcome
  is dropped after seeing results.
- Power note: with 20 runs per executor arm the study can detect only large
  differences in the primary outcome. This is stated with the results.

## 4. Study B — held-out audit benchmark

### Items
- Flawed items: analyses from other projects in the author's repository
  (single-cell foundation models other than MaxToki, protein language models,
  text-LLM-based cell models, GRN benchmarks) whose error was found and fixed
  after the fact, rebuilt as self-contained packages (data, code, outputs and
  the results summary as originally written), with hint comments removed.
  Plus a small number of planted causal-overreach items (pattern P9 has no
  natural example in the pool). Each item has exactly one documented error.
- Clean items: the corrected versions of the same analyses where available,
  plus other analyses checked to be correct.
- Every item is verified by code before freezing: a flawed item reproduces its
  documented wrong result; a clean item reproduces the correct one.
- Items are labelled by whether the documented error falls inside the
  checklist's ten patterns or outside them (e.g. pure code bugs).

### Arms
- `checklist` (generic checklist) vs `generic` review; same budget sentence,
  same schema; 3 independent reviews per item per arm.

### Outcomes
- Flawed items: detected (1/0): a finding that names the documented error's
  nature and location well enough that a reader would fix it.
- All items: false alarms: findings that assert a load-bearing error that the
  key marks as not an error. Disputed findings go to a verifier agent that
  checks them by running code on the package; its verdict is final.
- Per arm: sensitivity, false-alarm rate per clean item, precision of
  load-bearing findings; split by in-checklist vs outside-checklist errors.

### Analysis
- Item-level means over replicates; bootstrap over items for CIs; paired
  comparison of arms by item (Wilcoxon signed-rank on item-level detection;
  also a mixed logistic model with item random effect if it converges).

## 5. Study C — blind retrospective re-audit of the deployment

- Snapshot: the MaxToki project as it stood before the audit started
  (2026-05-07 16:57), built from file timestamps; audit files, paper folders and
  every later file excluded; summaries edited later reconstructed by stripping
  the dated audit additions. Manifest kept outside the snapshot.
- Ground truth: rows of the error ledger with `layer = run_outputs` and
  `present_in_pre_audit_state = yes` (frozen before the run).
- Arms (3 independent runs each): `deployed-checklist` (the audit specification
  exactly as used in the deployment, including its project-specific examples),
  `checklist` (generic), `generic` review. Same budget sentence and schema.
- Outcomes: per ledger item, detection by each run (graded by two graders
  with the ledger); critical-item detection; unmatched findings classified by a
  verifier as (a) a real error missing from the ledger, (b) false alarm,
  (c) cannot be decided.
- Comparison: per-arm item-level detection with bootstrap CIs over items, and
  the original audit's detection on the same items (from the ledger).

## 6. Things not done (stated in advance)

- No human baseline (no human analysts or reviewers).
- No other model family as executor or reviewer.
- No measurement of human supervision time (the deployment did not log it).
