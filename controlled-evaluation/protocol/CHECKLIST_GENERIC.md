# Methodological audit checklist (ten recurring reviewer objections)

Apply this checklist to the analysis you are given. For each of the ten
patterns below, decide whether the analysis shows the pattern, using the
diagnostic signature, and say what the prescription requires. Go from each
load-bearing claim to its numerical evidence to the code that produced it,
in that order. Record a verdict for every pattern (clean / present but
acknowledged / present and unaddressed), with evidence.

Wherever the text below mentions `FINAL_SUMMARY.md`, `README.md`, a
pipeline section number (§) or a project folder, read it as "the analysis's
results summary", "its method description" or "its files", whatever they
are called in the material you are given.

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

**Prescription.**
- Add to the validation step (§9 of any pipeline) an explicit "trivial
  baseline" line: PCA-on-expression, marginal correlation, or gene-mean —
  whichever is the natural simple comparator for that endpoint.
- Convert any absolute-AUROC claim to a "ΔAUROC over trivial baseline" claim
  in `FINAL_SUMMARY.md`. If the delta is zero or negative, that *is* the
  finding and should be stated explicitly.
- For cross-model claims, always also report the static-input-only alignment
  as a floor.

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

**Prescription.**
- In every summary of results, restate
  what the endpoint *literally* measures before the biological claim. Use
  an explicit two-objective framing: "Objective A
  (mechanistic / edge-recovery against curated database)" vs "Objective B
  (perturbation-response prediction)" — and state which objective each claim
  speaks to.
- Tag annotation databases used as ground truth with the cell-type / context
  in which they were curated. Note explicitly when that context does not
  match the analysis context.
- Replace "encodes" / "causes" with "is consistent with" / "co-varies with"
  unless an intervention-based test was actually done.

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

**Prescription.**
- Whenever a numeric claim is reported, name the inference unit explicitly:
  "n=NN edges over MM TFs over KK cells" with the unit at which the null /
  CV / bootstrap is grouped.
- For shared-endpoint metrics, run GroupKFold (by TF, by target, by both)
  and report the worst of the three. If the conclusion is robust, fine; if
  not, restate.
- For cell-type stratification, report a sensitivity analysis where labels
  are alternatively assigned via reference-mapping (Azimuth / SingleR).

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

---

### P6. No positive control for null findings

**Reviewer signature.** *"How do I know your pipeline can detect signal at
all? Show me a positive control where the answer is known."*

**Why it recurs.** A null finding is uninterpretable without evidence that
the pipeline *can* detect a true signal. Reviewers have repeatedly required a
synthetic + real-data positive-control battery before a null result was
acceptable.

**Diagnostic signature.**
- A pipeline that reports a clean negative outcome (e.g., "0/12 layers pass
  null gap") with no companion experiment confirming the pipeline detects
  signal when it should.
- An evaluation framework that's only ever been run on cases where the
  answer is unknown, never on cases where the answer is known.

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
  (e.g., two models that share a tokenizer and training recipe); not
  triangulated against an out-of-family member (e.g., scGPT, an attention-
  only architecture on rank-binned values).
- A "we choose this dataset because it's convenient" justification with no
  acknowledgement of the dataset's idiosyncrasies.

**Prescription.**
- Pick at least one held-out condition (tissue / dataset / architectural
  family) for every load-bearing finding; report the result.
- When triangulating cross-model, include at least one out-of-family member
  (e.g., a model with a different tokenizer and architecture).
- For headline claims, run ≥3 random seeds and report mean ± SD.

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

**Prescription.**
- For headline claims: bootstrap (1000 resamples; resample at the inference
  unit, e.g. by TF), DeLong for AUROC differences, report 95% CI.
- For load-bearing hyperparameters: at minimum a 3-point sweep with
  worst-case reported in main FINAL_SUMMARY.
- Where a sweep reveals threshold-dependent claims, restrict the claim to
  the threshold-stable range and say so.

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
- AI-cadence template: "X is not Y. It is the Z.".

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

**Prescription.**
- One audit-time check per run: every script path in
  `FINAL_SUMMARY.md` and `README.md` resolves; every pinned-commit hash in
  the pipeline §2 resolves under `repos/<name>/PINNED_COMMIT.txt`.
- Add to each run a `RUNTIME.md` (or extend `STATUS.md`) with: total wall
  time per phase, peak memory if known, hardware used.
- Where a public repo is referenced, verify accessibility and Zenodo DOI in
  the audit; record either as "verified" or "missing" in the audit report.

---
