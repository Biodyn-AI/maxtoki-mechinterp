# Audit pipeline — recurring reviewer-flagged issues

## 1. Overview

This is a **meta-audit pipeline**, not a research pipeline. It does not produce
new biology; it audits the artefacts of existing research pipelines (any of the
files under `pipelines/` applied to a target model under `projects/<target>/`)
against the **recurring patterns** that peer reviewers have consistently
flagged across the project's lineage of papers.

The audit is built bottom-up from the actual review materials in
`review_plans/`: the BMC Genomics revision (attention/perturbation paper), the
Bioinformatics revision (causal circuit tracing / SAE paper), the PLOS ONE
revision (topology / 141-hypotheses paper), the *Neuron*-style review of the
intelligence-genes paper, the *Biogerontology* revision (longevity
mech-interp), and the BCB2026 review (disagreement-arbitration GRN paper).
Across six independent reviews of six different papers by different reviewers,
the same ten clusters of methodological objections recur. This pipeline turns
those clusters into a checklist + workflow for examining any future run
artefact against that pattern catalogue.

The output is a **per-project audit report** that names which patterns are
present in a given project's run artefacts, where the supporting evidence is,
and what concrete fix or qualifying language is recommended. The audit is
deliberately sized to be doable in 1–2 days per project; it is not a full
re-run.

This pipeline is **complementary** to existing review-style material in this
repo and does not duplicate it (per user direction):
- `prompts/reviewer-1-style-correctness.md`, `prompts/reviewer-2-research-quality.md`,
  `prompts/brainstormer-ideas.md` remain authoritative as composable critique
  prompts. This pipeline cites them as cross-references but does not re-derive
  what they already cover.
- `CLAUDE.md` lists known failure modes (co-expression confound, feature-shuffle
  null trap, annotation-database circularity, selective reporting). This
  pipeline expands those into a structured catalogue grounded in `review_plans/`.

## 2. Source

- **Evidence base:** the six review documents in `review_plans/`, abbreviated
  below as `[NMI]` (BMC Genomics revision), `[BIO]` (Bioinformatics SAE
  revision), `[PLOS]` (PLOS ONE topology revision), `[NEU]` (Neuron-style
  intelligence-genes review), `[BGER]` (Biogerontology longevity revision),
  `[BCB]` (BCB2026 disagreement-arbitration review). The round-2 housekeeping
  PDF (`response_to_reviewers_r2.pdf`) is included for completeness but
  contributes only cross-reference / placeholder fixes.
- **Companion material in this repo:** `prompts/reviewer-1-style-correctness.md`,
  `prompts/reviewer-2-research-quality.md`, `prompts/brainstormer-ideas.md`,
  `CLAUDE.md` "Domain conventions" section. Cross-referenced, not modified.
- **Pinned commit hash:** N/A — this pipeline is methodological. It depends on
  the *content* of `review_plans/` at the time of writing (2026-05-07); when
  new review documents are added to that folder, §6 should be re-checked
  against them.

## 3. Inputs

A target-model project folder, e.g. `projects/maxtoki/`, with the conventional
layout:

```
projects/<target>/
├── README.md                          ← project-level architecture + run order
├── runs/
│   ├── <pipeline-name>-<size>/
│   │   ├── README.md                  ← scoped execution plan + scope deltas
│   │   ├── FINAL_SUMMARY.md           ← cross-pipeline synthesis
│   │   ├── EXTENDED_FINDINGS.md       ← optional: extension-phase additions
│   │   ├── STATUS.md                  ← optional, for autoloop runs
│   │   ├── scripts/                   ← per-phase runners
│   │   ├── outputs/                   ← per-phase results (CSV, JSON, NPY)
│   │   └── logs/                      ← per-phase stdout
│   └── …
└── notes/, planning/                  ← per-project working notes (optional)
```

The audit consumes:
1. Each run's `README.md`, `FINAL_SUMMARY.md`, and `EXTENDED_FINDINGS.md`
   as the primary claim surface.
2. Each run's `outputs/**/*.{csv,json}` for the quantitative evidence behind
   those claims.
3. Each run's `scripts/*.py` to verify claimed methods are implemented and
   to spot-check null/baseline/CV-split logic.
4. The project-level `README.md` for cross-pipeline triangulation (e.g., do
   verdicts from spectral-geometry, attention-GRN, SAE, topology agree?).

The audit does **not** re-extract embeddings, re-train any model, or re-run any
pipeline. It examines what's already on disk.

## 4. Outputs

A single audit report, conventionally placed at:

```
projects/<target>/audits/audit-<YYYYMMDD>.md
```

The report has one section per recurring-issue pattern (see §6) with:
- **Verdict per run:** clean / present-but-acknowledged / present-and-unaddressed.
- **Evidence pointers:** specific file paths + line ranges + claim quotes.
- **Recommended action:** one of (a) re-run with corrected method, (b) add
  qualifying language to FINAL_SUMMARY/EXTENDED_FINDINGS, (c) explicitly
  note the limitation in §"Known pitfalls" of the relevant pipeline file in
  `pipelines/`, (d) no action — already handled.
- **Effort estimate:** ≤30 min / 1–2 hr / ≥half-day.

Plus an executive summary at the top: **N issues found, M acknowledged in run
docs, K requiring action, distribution by severity.**

A second optional output is a per-run `audit-delta.md` written into each
audited run folder, summarizing only that run's audit findings — useful when
the project has many runs and the consolidated report becomes long.

## 5. Dependencies

- No code dependencies — the audit is a methodology document applied by
  reading and writing markdown.
- Conceptual dependencies (cross-references, not modifications):
  - `prompts/reviewer-1-style-correctness.md` — for citation, equation, and
    typesetting checks beyond the scope here.
  - `prompts/reviewer-2-research-quality.md` — for research-design critique
    that overlaps but is broader than recurring-reviewer-pattern matching.
  - `prompts/brainstormer-ideas.md` — for generating fix proposals once an
    issue is identified.
  - `CLAUDE.md` "Recurring failure modes" list — co-expression confound,
    feature-shuffle null trap, annotation-database circularity,
    selective reporting of positives, the "looks great until you control
    properly" pattern, immune-tissue concentration of signal.

## 6. Methodology — the recurring-issue catalogue

Ten patterns recur across the six reviews. Each is named by the *spirit* of
the objection, not by any specific paper's instance of it. For each pattern,
the entry below states:

- **Reviewer signature** — how this objection typically reads
- **Why it recurs** — the structural reason it keeps appearing
- **Diagnostic signature in run artefacts** — what to grep / read for
- **Common manifestations specific to this repo's pipelines**
- **Prescription** — fix vs acknowledge vs qualify
- **Cited by** — which review document(s) flagged this

The audit workflow (apply to a project end-to-end) is in §6.11.

---

### P1. Null model is too weak / not anchored to chance

**Reviewer signature.** *"No baseline given for what `0.53 shared ontology`
means — anchor it to a chance baseline."* *"The expression-matched null may
be insufficient — control for variance and dropout, not just mean."* *"How
much of the effect survives a stricter null?"*

**Why it recurs.** Single-cell foundation-model artefacts are dense, so a
naive shuffle (gene-label permutation, feature-shuffle) produces a null much
weaker than the structure that actually drives the observed result.
Reviewers want to see (i) every quantitative claim *anchored* to a chance
value, and (ii) a hierarchy of nulls that progressively rules out
confounders.

**Diagnostic signature.**
- A claim of the form "X = 0.NN" without an adjacent "vs chance baseline 0.MM
  ± SD".
- A null implemented only as `np.random.permutation(labels)` or
  `np.random.shuffle(features)` with no degree preservation, no
  composition matching, and no coexpression matching.
- A "p-value" without a stated null family.
- A "ΔAUROC" reported only against the trivial 0.5 baseline rather than
  against a coexpression / matched-degree null.

**Common manifestations here.**
- Phase 5 / Phase 6 ΔAUROC tables that only report `delta_vs_coex` without
  also reporting `delta_vs_max_null` (the strict max across feature-shuffle,
  coexpression, rewire).
- H123-family scores reported with sign-shuffle null but without the
  TF-degree-stratified variant.
- "ΔAUROC +0.03" with no statement of effect size relative to the null SD.

**Prescription.**
- For every numeric claim in `FINAL_SUMMARY.md` that is meant to be
  meaningful (i.e., load-bearing), attach a "(null mean ± SD; p95)" suffix.
  If the suffix would be misleading because the relevant null isn't computed,
  compute it (cheap; reuse phase-12-style strict-max-null harness).
- Adopt the strict max-null framing as the default for any claim that
  appears in an abstract or executive summary. Per-domain breakdowns at the
  loose null are fine in supplementary detail.
- Where a single null family was used, name it inline ("…vs feature-shuffle
  null; the same comparison against degree-preserving rewiring is
  in §X").

**Cited by.** [PLOS R2/R3], [BIO E12], [NEU R1.4], [BGER R3.1], [BCB C6].
Already in CLAUDE.md as "feature-shuffle null trap".

---

### P2. Trivial / gene-level baseline absent

**Reviewer signature.** *"What does a simpler model already give you?"*
*"Does the foundation-model embedding add **incremental value** over PCA on
gene expression / marginal correlation / gene-level mean expression?"*

**Why it recurs.** Foundation-model claims are most credible when stated as
"this beats the simple thing." Without a trivial baseline, the reader cannot
distinguish "model encodes X" from "X is recoverable from raw expression
statistics."

**Diagnostic signature.**
- Headline AUROC reported in absolute terms with no comparison to PCA,
  raw-correlation, or gene-mean baselines.
- "Cross-model alignment r=0.78" reported without checking whether r between
  the two models' *static input embeddings* (which are trained independently
  but on similar token sets) already produces r ≈ 0.78 — i.e., the alignment
  is a property of the input layer, not learned dynamics.
- Discussion section claims biological understanding without naming a simpler
  model that would or would not produce the same number.

**Common manifestations here.**
- Cross-model CCA results without a gene-coexpression baseline ("does
  coexpression alone produce comparable r?").
- H123 motif-community ΔAUROC without a trivial coexpression-only baseline.
- Spectral-geometry / SAE positive findings without a PCA-on-expression
  comparator.

**Prescription.**
- Add to the validation step (§9 of any pipeline) an explicit "trivial
  baseline" line: PCA-on-expression, marginal correlation, or gene-mean —
  whichever is the natural simple comparator for that endpoint.
- Convert any absolute-AUROC claim to a "ΔAUROC over trivial baseline" claim
  in `FINAL_SUMMARY.md`. If the delta is zero or negative, that *is* the
  finding and should be stated explicitly.
- For cross-model claims, always also report the static-input-only alignment
  as a floor.

**Cited by.** [NMI R1.1] (BMC's central re-framing: "no regulatory signal" →
"no incremental value over gene-level features"), [BGER R1.1] (PCA matches
scFM in 5/5 cohorts), [BIO E5] (input-size-normalized comparison),
[BIO E7] (partial-correlation test).

---

### P3. Endpoint–object mismatch

**Reviewer signature.** *"Perturbation DE measures direct + indirect; you
can't infer direct regulation from this."* *"AUROC vs TRRUST is an
edge-recovery task; perturbation-target prediction is an effect-magnitude
task; they are not the same thing."* *"Embedding shift is a model-internal
quantity; what makes you think it tracks biological importance?"* *"TRRUST
is curated literature; ChIP-seq in the matched cell type is closer to
truth."*

**Why it recurs.** Mech-interp work routinely conflates (a) what the metric
arithmetically measures, (b) what the model is internally doing, and (c) the
biological claim made in the abstract. Reviewers home in on this gap.

**Diagnostic signature.**
- An abstract sentence "the model encodes regulatory relationships" supported
  only by an AUROC against a curated database.
- "Direct target" used loosely without a ChIP-seq peak or a perturbation +
  time-resolved direct-effect filter.
- Any claim that "the model causes X" supported by a correlational
  embedding-shift analysis.
- Reuse of an annotation database for *both* feature selection and
  evaluation (annotation-database circularity).

**Common manifestations here.**
- Phase-7+8 H123 framed as "regulatory signal" when the actual measurement is
  community co-membership consistent with TRRUST sign annotations.
- Spectral-geometry "B-cell attractor" framed as a causal claim when the
  evidence is a geometric correlate.
- SAE Stage-2 CRISPRi 54.6% directional accuracy described as "near-chance
  causal evidence" without qualifying that the labels themselves include
  indirect targets.

**Prescription.**
- In every pipeline file's §1 Overview and FINAL_SUMMARY's TL;DR, restate
  what the endpoint *literally* measures before the biological claim. Use
  the explicit two-objective framing from [NMI §5.1]: "Objective A
  (mechanistic / edge-recovery against curated database)" vs "Objective B
  (perturbation-response prediction)" — and state which objective each claim
  speaks to.
- Tag annotation databases used as ground truth with the cell-type / context
  in which they were curated. Note explicitly when that context does not
  match the analysis context.
- Replace "encodes" / "causes" with "is consistent with" / "co-varies with"
  unless an intervention-based test was actually done.

**Cited by.** [NMI R1.2/R3.2] (perturb-seq indirect-effect problem),
[BIO R1-M2/R1-M3] (ChIP-seq matched-cell-type required, TRRUST insufficient),
[NEU R1.2] (embedding shift as proxy), [BGER R3.2] ("mechanism-grade
evidence" → "intervention-consistent directional association"),
[BCB C8] (Dixit transfer near-chance buried).
Already in CLAUDE.md as "annotation-database circularity".

---

### P4. Pseudoreplication / wrong unit-of-inference / leaky CV

**Reviewer signature.** *"You have 500 cells but only 3 donors — your
inference unit is donors, not cells, and your power is ~27%."* *"Edge-level
5-fold CV leaks gene-level features; use GroupKFold by TF."* *"Cell-type
labels are imputed from markers; treat as exploratory, not confirmatory."*

**Why it recurs.** Single-cell data is hierarchical (cell ⊂ donor ⊂ cohort,
edge ⊂ TF, etc.) and the natural unit at which the *biological* claim lives
is rarely the same as the unit at which sample size N is large. CV / nulls
that ignore this hierarchy systematically inflate significance.

**Diagnostic signature.**
- A p-value computed over cells when the claim is about donors / cohorts /
  cell-types.
- Reported `n=500` cells with no `n=donors`, `n=cohorts`, or
  `n=independent-conditions` companion.
- StratifiedKFold rather than GroupKFold when features are shared across rows
  (edges sharing a TF; gene pairs sharing an endpoint).
- A bootstrap implemented at the row level when the row-level units are not
  exchangeable.
- Cell-type-stratified AUROC reported without acknowledging that cell-type
  labels are noisy.

**Common manifestations here.**
- H123-family per-pair AUROC computed over thousands of pairs, reported as
  "n=2544 pairs" — but pairs share endpoints, so effective N is much smaller.
- Phase-1 splits implemented as TF-disjoint OR target-disjoint, not always
  as the harder dual-axis disjoint.
- Cell-type analyses reported as confirmatory when based on marker-gene
  thresholds rather than reference-mapped labels.

**Prescription.**
- Whenever a numeric claim is reported, name the inference unit explicitly:
  "n=NN edges over MM TFs over KK cells" with the unit at which the null /
  CV / bootstrap is grouped.
- For shared-endpoint metrics, run GroupKFold (by TF, by target, by both)
  and report the worst of the three. If the conclusion is robust, fine; if
  not, restate.
- For cell-type stratification, report a sensitivity analysis where labels
  are alternatively assigned via reference-mapping (Azimuth / SingleR).

**Cited by.** [NEU R1.1/R2.1] (pseudoreplication; n=3 donors, ~27% power),
[BGER R1.1/R3.4] (donor-aware splits, gate calibration), [BCB C2]
(GroupKFold by TF / target), [NEU R1.3] (cell-type marker fragility).

---

### P5. Selection-driven inflation / double-dipping / confound carry-through

**Reviewer signature.** *"You picked the 120 best-annotated features and
report properties of *those* — that's a biased subset."* *"You ranked genes
on the same data you tested them on."* *"Effect strength is correlated with
expression level — control for it."* *"Tokenization window excluded the
genes you couldn't test."*

**Why it recurs.** Selection rules and confounders have a way of sneaking in
between data acquisition and analysis. The most common version is: select on
property A, test for property B, report the test as if A and B are
independent — but A's selection rule already biases B.

**Diagnostic signature.**
- A "top-N" filter applied before a test, with no random / unselected
  comparison.
- Effect-size statistics that aren't residualized against expression level,
  token frequency, or coexpression.
- A test set that excludes high-expression / high-popularity / high-degree
  items by accident of the pipeline (e.g., scGPT 512-token window dropping
  low-rank genes).
- The same data used for both ranking and confirmatory testing.

**Common manifestations here.**
- SAE atlas / circuit work that selects "annotated" features then reports
  "53% biological coherence" — without a random-feature comparison.
- Phase-6 triangle-defect ΔAUROC without checking whether the signal
  survives coexpression residualization (which, when checked in
  Phase 11 iter_03, it didn't).
- Token-frequency stratified analysis missing — H123 signal could be a
  popularity confound (which we ruled out in Phase 11 iter_05, but the rule-
  out wasn't run in the original pipeline).

**Prescription.**
- For every "selected" feature set claim, run the same analysis on a random
  unselected sample and report both. State the gap honestly: if the random
  set still shows the effect, the selection didn't drive it; if it doesn't,
  the selection did.
- For every effect-size claim, residualize against the most likely confound
  (coexpression, token frequency, expression level) and report the residual
  effect.
- Document tokenization-window dropouts: which genes were excluded and
  whether their inclusion would change the conclusion.
- Avoid double-dipping by holding out a slice for confirmatory testing
  before any feature selection.

**Cited by.** [BIO R2-M1/E6/E7] (annotation bias; partial correlation),
[NEU R1.4/R2.2] (expression-coupling confound), [NEU R2.3]
(tokenization audit), [NEU R2 minor 4] ("double-dipping" bias),
[BGER R3.3] (cohort composition shifts attenuate signal).
Already in CLAUDE.md as "co-expression confound" and "selective reporting".

---

### P6. No positive control for null findings

**Reviewer signature.** *"How do I know your pipeline can detect signal at
all? Show me a positive control where the answer is known."*

**Why it recurs.** A null finding is uninterpretable without evidence that
the pipeline *can* detect a true signal. The strongest reviewer concern in
this project's lineage was [NMI R1.3] requiring a synthetic + real-data
positive-control battery before a null result was acceptable.

**Diagnostic signature.**
- A pipeline that reports a clean negative outcome (e.g., "0/12 layers pass
  null gap") with no companion experiment confirming the pipeline detects
  signal when it should.
- An evaluation framework that's only ever been run on cases where the
  answer is unknown, never on cases where the answer is known.

**Common manifestations here.**
- topology-141-217M reports a clean negative on H123 in MaxToki without a
  positive control demonstrating the same H123 implementation detects
  H123-style signal in a synthetic graph with planted signed-motif structure.
- attention-GRN-217M near-zero "curveball" without a planted-edge synthetic
  control.

**Prescription.**
- For every load-bearing null finding, add a short positive-control phase
  (synthetic graph with planted ground truth + same metrics + same null
  hierarchy) showing the pipeline detects signal at planted SNR. This can
  be quick: a 200-gene synthetic graph with 50 planted regulatory edges +
  one classifier eval per condition.
- Optionally, a real-data positive control on a hand-curated case with
  strong known truth (e.g., GATA1/MYC/TAL1 in K562 with ChIP-seq direct
  targets).
- If the positive control fails, *that's* the finding — don't suppress.

**Cited by.** [NMI R1.3 §3] (full synthetic + real-data positive control),
[BGER R1.1, R3.5] (workflow ablation; what would have been concluded
without each block), [BIO R2-M2] (causal validation tightening).

---

### P7. Single-condition generalization

**Reviewer signature.** *"You used K562 — that's an immortalized cancer line
not in either model's pre-training corpus."* *"Conclusions are based on
three tissue domains; show held-out tissue."* *"Show this works on more than
one architecture."*

**Why it recurs.** Foundation-model interpretability work is often
demonstrated on whatever data was convenient. Reviewers ask: does the
finding generalize beyond the one chosen condition?

**Diagnostic signature.**
- Findings stated for a single tissue / cohort / cell type / model
  architecture / random seed.
- Cross-condition agreement reported only as a within-family measurement
  (e.g., MaxToki ↔ Geneformer, both Llama-style on Ensembl tokens); not
  triangulated against an out-of-family member (e.g., scGPT, an attention-
  only architecture on rank-binned values).
- A "we choose this dataset because it's convenient" justification with no
  acknowledgement of the dataset's idiosyncrasies.

**Common manifestations here.**
- topology-141 / spectral-geometry reporting on lung/immune/external_lung
  alone with no held-out tissue.
- Cross-model claims based only on Geneformer V2-316M when scGPT was
  available (this was addressed in topology-141's Phase 4 vs scGPT, which
  found alignment does NOT generalize across architectural families — a
  single-condition generalization failure caught only by triangulation).
- Single-seed runs without bootstrap stability.

**Prescription.**
- Pick at least one held-out condition (tissue / dataset / architectural
  family) for every load-bearing finding; report the result.
- When triangulating cross-model, include at least one out-of-family member
  (e.g., scGPT for an autoregressive-Ensembl finding; or vice versa).
- For headline claims, run ≥3 random seeds and report mean ± SD.

**Cited by.** [PLOS R1] (held-out kidney tissue), [NMI R1.4] (multi-model
coverage table; scGPT ablation extension), [BIO R1-M1] (non-immortalized,
training-covered cells), [NEU R2.3] (cross-architecture benchmarking),
[BGER R3.3] (cohort-composition attribution).

---

### P8. Stability of headline numbers

**Reviewer signature.** *"What's the bootstrap CI on +0.117?"* *"Does the
result hold at threshold 0.3 instead of 0.5?"* *"What if you change PCA
dim, k, or seed?"*

**Why it recurs.** A point estimate without a stability range invites
suspicion that it's a single-pick of the favorable corner of a parameter
grid.

**Diagnostic signature.**
- Headline ΔAUROC / Δr / Cohen's d reported as a single number without
  a SD, CI, or sweep range.
- Hyperparameters (PCA dim, kNN k, threshold cutoffs, Louvain resolution)
  fixed at single values without sensitivity sweeps.
- DeLong test or bootstrap absent for AUROC differences.

**Common manifestations here.**
- "ΔAUROC +0.094, 6/6 domain-splits" reported as a single point estimate
  without the bootstrap distribution under TF-resampling.
- PCA-20 used everywhere with no documented check at PCA-{10, 30}.
- KNN k=12 fixed; no documented sensitivity sweep.

**Prescription.**
- For headline claims: bootstrap (1000 resamples; resample at the inference
  unit, e.g. by TF), DeLong for AUROC differences, report 95% CI.
- For load-bearing hyperparameters: at minimum a 3-point sweep with
  worst-case reported in main FINAL_SUMMARY.
- Where a sweep reveals threshold-dependent claims, restrict the claim to
  the threshold-stable range and say so.

**Cited by.** [BIO R2-M3/E9, R2-M5/E11] (bootstrap stability; threshold
sensitivity), [PLOS R2.4] (CV + hyperparameter sweep), [BCB C6, C7]
(DeLong + bootstrap CIs; threshold robustness), [NEU R2.5 minor 1]
(sampling stability of ranks).

---

### P9. Causal / regulatory / mechanistic language overreach

**Reviewer signature.** *"'Wiring diagram' is overstated."* *"Replace
'confirms' with 'is consistent with'."* *"Distinguish model-level causality
from biological causality."* *"The paper reads like it was written by AI."*

**Why it recurs.** The evidence base of an interpretability pipeline is
correlational by default; the language slips toward causal/mechanistic
because that's what makes the result feel important. Reviewers consistently
push back, often word-for-word.

**Diagnostic signature.**
- Abstract or §1 contains: "encodes regulation", "causal circuit", "wiring
  diagram", "complete pathway", "the model understands", "the model agrees".
- "Confirms" used where the evidence is at most "is consistent with".
- A negative result understated or buried (e.g., a transfer experiment near
  chance treated as supportive).
- AI-cadence template: "X is not Y. It is the Z." (per [BGER R4]).

**Common manifestations here.**
- spectral-geometry "B-cell germinal-centre attractor" framed as a discovered
  mechanism when the evidence is geometric.
- Stage-2 SAE circuit-tracing language carrying causal weight despite
  CRISPRi 54.6% directional accuracy (near-chance).
- attention-GRN curveball ≈0 — a clean negative — at risk of being framed
  as "negligible signal" instead of "no incremental value over coexpression".

**Prescription.**
- One global pass over each pipeline's `FINAL_SUMMARY.md` and the project
  README, swapping causal/mechanistic verbs for evidence-grade verbs:
  - "encodes regulation" → "is correlationally aligned with curated
    regulatory annotations"
  - "causal circuit" → "model-internal ablation graph"
  - "confirms" → "is consistent with"
  - "agrees with" → "co-varies with"
  - "discovers" → "recovers" (when comparing to a known prior)
- Define an explicit four-level claim taxonomy in any abstract that makes a
  claim about regulation: (1) decoding, (2) cross-model agreement, (3)
  intervention-consistent directional association, (4) causal mechanism.
  State which level the paper claims; do not exceed it.
- Own the negatives — name them as findings, not as gaps.

**Cited by.** [BIO R2-M2/M4/m2] (causal-language audit), [BGER R3.2]
("mechanism-grade evidence" reframe), [BGER R4] (AI-cadence rewrite),
[BIO R1-m3] ("confirms" → "suggests"), [BCB C8] (own the Dixit negative),
[PLOS R5] (anthropomorphic language).
Already in CLAUDE.md as "selective reporting of positives".

---

### P10. Reproducibility scaffolding

**Reviewer signature.** *"Provide environment.yml."* *"Archive code with a
DOI."* *"Confirm every script referenced in the paper is in the public
repo."* *"Report computational footprint."*

**Why it recurs.** This is the cheapest-to-fix and most-frequently-flagged
class. Reviewers expect a frozen, citable, runnable record.

**Diagnostic signature.**
- A pipeline that cites a script path that doesn't exist at the pinned
  commit.
- A run folder with no `requirements.txt` / `environment.yml` / no pinned
  versions, even informally.
- Compute cost / wall time absent — reviewer cannot estimate replication
  effort.
- A repo URL in the README that is private or does not resolve.

**Common manifestations here.**
- Run `scripts/phase*.py` paths cited in `FINAL_SUMMARY.md` should exist;
  always check post-revision that the cited paths still resolve.
- The pinned-commit hash in each pipeline's §2 must still resolve in
  `repos/<name>/PINNED_COMMIT.txt`.
- Per-phase log files (`logs/phase*.log`) should record wall time so the
  scalability profile can be reconstructed without rerunning.

**Prescription.**
- One audit-time check per run: every script path in
  `FINAL_SUMMARY.md` and `README.md` resolves; every pinned-commit hash in
  the pipeline §2 resolves under `repos/<name>/PINNED_COMMIT.txt`.
- Add to each run a `RUNTIME.md` (or extend `STATUS.md`) with: total wall
  time per phase, peak memory if known, hardware used.
- Where a public repo is referenced, verify accessibility and Zenodo DOI in
  the audit; record either as "verified" or "missing" in the audit report.

**Cited by.** [PLOS J2/J5] (code sharing, reference audit), [NMI R2.7]
(GitHub repo audit), [BIO editor] (Zenodo archival), [NEU R2.2]
(environment pinning), [PLOS R4] (computational cost).

---

### 6.11 Audit workflow

For a target project (e.g. `projects/maxtoki/`), the audit proceeds as:

1. **Inventory.** List every run under `projects/<target>/runs/*`. For each
   run, record: pipeline source (which `pipelines/<X>.md` it implements),
   present documents (README / FINAL_SUMMARY / EXTENDED_FINDINGS / STATUS),
   total artefact volume.

2. **Per-pattern sweep (P1–P10 above).** For each of the ten patterns:
   - Read each run's `FINAL_SUMMARY.md` and `EXTENDED_FINDINGS.md` against
     the pattern's diagnostic signature.
   - Spot-check `outputs/**/*_summary.json` for the relevant numeric
     evidence.
   - Spot-check `scripts/*.py` for null / CV / baseline implementation.
   - Record verdict per run: **clean / acknowledged / unaddressed**.

3. **Cross-pipeline triangulation.** Some patterns surface only when
   comparing two runs:
   - P3 (endpoint mismatch) and P9 (causal language) are easier to
     spot when the project's `README.md` aggregates verdicts that
     contradict at the language level even if they agree at the data level.
   - P7 (single-condition generalization) often appears as "this run only
     used Geneformer; another run only used scGPT; the project never ran
     both on the same task".

4. **Severity grading.** Each unaddressed finding is graded:
   - **Critical** — the finding's headline claim cannot stand without action.
   - **Major** — claim survives but with material qualification.
   - **Minor** — language-level fix or scaffolding fix.

5. **Action proposal.** Per finding, propose one of:
   - **(re-run)** specific phase needs to be re-executed with a corrected
     null / baseline / CV / control. Reference the existing pipeline
     phase that is closest to the needed fix; estimate effort.
   - **(qualify)** language fix to `FINAL_SUMMARY.md` and/or pipeline §1
     overview. Provide the exact verb-swap.
   - **(annotate)** add to pipeline's §10 "Known pitfalls" list as a
     forward-looking caution.
   - **(no-op)** already handled, no action needed; record evidence.

6. **Audit report.** Compose `projects/<target>/audits/audit-<YYYYMMDD>.md`
   with: executive summary (counts by severity), per-pattern section with
   per-run findings, action proposal table sorted by effort × severity.

7. **Optional per-run delta.** If actionable findings concentrate in a
   small number of runs, write a short `audit-delta.md` into each affected
   run folder so future readers see the audit's verdict where the run
   itself lives.

The full pass on a single project with ~5–10 runs is a 1–2 day exercise
(reading + judgment, not running). The audit is *not* itself a research
contribution and should not produce new claims; it produces qualifications
and re-run proposals.

## 7. Code references

This pipeline does not run code. It cross-references existing code locations
that are useful when an audit finding suggests a re-run:

- **Strict max-null harness** for P1: any pipeline's
  `scripts/phase12_strict_max_null.py` (e.g.,
  `projects/maxtoki/runs/topology-141-217M/scripts/phase12_strict_max_null.py`).
- **Coexpression-residualization template** for P2/P5:
  `projects/maxtoki/runs/topology-141-217M/scripts/phase11_autoloop.py:iter_03_coex_residual_triangle`.
- **Token-frequency stratification template** for P5:
  `projects/maxtoki/runs/topology-141-217M/scripts/phase11_autoloop.py:iter_05_token_frequency_strata`.
- **Cross-model triangulation template** for P7:
  `projects/maxtoki/runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py`
  (cross-architecture) and `phase14_cross_model_cca.py` (within-family).
- **Annotation-extension chain template** for P5:
  `projects/maxtoki/runs/topology-141-217M/scripts/phase8ext_h124_h138_chain.py`.
- **Bootstrap / stability template** for P8: borrowable from
  `repos/topology-biomechinterp2/iterations/iter_0009/run_iter0009_screen.py`
  (bootstrap-stability harness at the autoloop reference commit) — adapt as
  needed.
- **Composition-matched control template** for P1/P5:
  `repos/longevity-mechinterp/.../stage10_*` and `stage11_*` (when present
  in the local clone) document the "delta-form intervention layered on
  pre-existing SAE artifacts" recipe from [BGER §R1.3].

## 8. Parameters

The audit thresholds below are conservative defaults; they can be relaxed
per-project if explicitly justified in the audit report.

| Parameter | Default | Rationale | Used in pattern |
|---|---:|---|---|
| `min_null_families_for_strict_claim` | 3 | Match the topology-141 pipeline's strict-max-null framing (feature-shuffle, coex-matched, degree-rewiring). | P1 |
| `require_trivial_baseline_in_summary` | yes | Any abstract or executive-summary number must report Δ over a trivial baseline. | P2 |
| `min_groupkfold_axes_for_pair_claim` | 2 | For shared-endpoint metrics (gene-pair edges), require GroupKFold by both endpoints. | P4 |
| `random_unselected_comparison` | yes | For "annotation-selected feature" claims, require a random-unselected companion. | P5 |
| `positive_control_required` | yes-for-load-bearing-negatives | Synthetic graph + planted ground truth at minimum. | P6 |
| `min_held_out_conditions` | 1 | At least one tissue / cohort / model / dataset NOT used during method tuning. | P7 |
| `min_seeds_for_headline` | 3 | Bootstrap or seed sweep for any abstract-level number. | P8 |
| `min_threshold_sweep_points` | 3 | At least 3 hyperparameter values reported with worst-case in main summary. | P8 |
| `causal_language_audit` | every-revision | Run a verb-swap pass on FINAL_SUMMARY before declaring the run complete. | P9 |
| `repo_+_doi` | yes | Public repo + Zenodo DOI for any externally-cited run. | P10 |

## 9. Validation

The audit "passes" for a given project when:

1. The audit report exists at `projects/<target>/audits/audit-<YYYYMMDD>.md`
   with all ten pattern sections populated (each labelled clean / acknowledged
   / unaddressed per run).
2. Every **critical** finding has either a re-run scheduled, a re-run
   completed, or a documented decision-not-to-rerun with reviewer-style
   justification.
3. Every **major** finding has a `FINAL_SUMMARY.md` qualification accepted
   into the run's documentation (verifiable by diff).
4. Every **minor** finding has an action item listed in the relevant
   pipeline file's §10 "Known pitfalls" if the issue is structural to the
   pipeline rather than incidental to one run.
5. Reproducibility scaffolding (P10) reports zero broken script paths and
   zero broken pinned-commit hashes for the project.
6. The audit report's executive summary closes with an explicit statement
   of which patterns the project remains exposed to (i.e., what a future
   reviewer of a paper based on this project would still flag) — even if
   no further action is feasible right now. Honesty about residual
   exposure is itself a deliverable.

If any of these are false, the audit is incomplete; record what's missing
in the audit report's "Open items" section.

## 10. Known pitfalls

The audit itself can fail in characteristic ways. Reviewers of an audit
would themselves invoke patterns like P1–P9; the audit is not exempt.

1. **Audit-driven selective reporting.** It is tempting to record only the
   patterns that have clean prescriptions and skip those that would
   demand expensive re-runs. Force yourself to record every pattern's
   verdict, including "present-and-unaddressed-because-too-expensive".
   That honesty is part of the deliverable.

2. **The "looks great until you control properly" pattern, applied to
   audits.** A run can pass cursory inspection of its `FINAL_SUMMARY.md`
   and fail when the underlying CSVs are checked. Always go from claim to
   numeric-evidence to script-implementation, in that order, for any
   load-bearing finding. Don't stop at the markdown layer.

3. **Recurring-pattern catalogue drift.** This pipeline is grounded in
   `review_plans/` as of the writing date in §2. New review documents will
   add new patterns and may invalidate old ones. The catalogue in §6 is
   not frozen — re-derive it whenever `review_plans/` materially changes
   (new paper review added, new revision round on an existing paper).

4. **False sense of completeness from checking ten patterns.** Reviewers
   raise things outside this catalogue too. The ten patterns are necessary
   conditions for an audit to be useful, not sufficient. If a pipeline's
   methodology has a unique-to-itself failure mode, the audit must invent
   the appropriate pattern slot for it; do not force every objection into
   one of the ten.

5. **Re-runs introduce their own selection bias.** Re-running a phase with
   a corrected null is not free of bias if the correction was chosen to
   recover a desired signal. Pre-register the re-run's expected outcome
   before running; if the re-run's outcome contradicts the pre-registered
   expectation, that *is* the finding.

6. **Audit reports are not papers.** They should be terse, evidence-pointer-
   heavy, and free of biological narrative. The audit's job is to surface
   issues; the pipeline files and FINAL_SUMMARYs are where biological
   claims live (suitably qualified).

7. **The audit can collide with active research.** If a project is
   mid-revision (e.g., responding to a journal review of a paper based on
   that project), the audit findings should feed into that revision rather
   than be filed separately. Coordinate by cross-referencing the relevant
   `review_plans/<paper>_revision_response_*.md` from the audit report.

8. **"This pattern doesn't apply to us" is rarely true.** Each of the ten
   patterns appeared in a different paper by a different reviewer. The
   prior probability that *none* of them apply to a new project is very
   low. If a sweep yields zero unaddressed findings, re-check more
   carefully — the most likely explanation is that the sweep was too
   shallow, not that the project is uniquely clean.
