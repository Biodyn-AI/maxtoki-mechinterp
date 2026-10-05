# Review and Brainstorming Prompts

This directory contains three role-based prompt templates for analysing papers in the **mechanistic interpretability of biological foundation models** domain — the same research area covered by the pipelines in `../pipelines/` and the reference papers in `../references/`.

The three roles are designed to compose: running all three in sequence on a single paper produces a copy-editor pass (Reviewer 1), a scientific referee report (Reviewer 2), and a forward-looking idea portfolio (Brainstormer). They can also be used independently.

## The three roles

| File | Role | Scope | Output |
|---|---|---|---|
| [`reviewer-1-style-correctness.md`](reviewer-1-style-correctness.md) | **Reviewer 1 — Copy & technical editor** | Internal consistency, logical coherence, wording precision, technical correctness (non-object-level), structural clarity. Does the manuscript say what it means, say it consistently across sections, and say it without hype or ambiguity? | Structured copy-editor report: per-issue location + severity + fix |
| [`reviewer-2-research-quality.md`](reviewer-2-research-quality.md) | **Reviewer 2 — Scientific referee** | Object-level critique: are the claims supported by the evidence? Are null models rigorous enough? Are confounds controlled? Do the results actually survive the strict audits the paper itself advocates? Are alternative explanations ruled out? | Formal referee report with decision (accept / minor revise / major revise / reject) and per-section concern list |
| [`brainstormer-ideas.md`](brainstormer-ideas.md) | **Brainstormer — Extension architect** | Forward-looking: what experiments would make this paper stronger? What cross-paper integrations does the project enable? What testable hypotheses fall out of the results? What are the moonshots worth attempting? | Idea portfolio (12–20 items) across 3 ambition tiers, each with hypothesis / test / expected signal / null / cost |

## Intended composition

The roles are designed to run as a pipeline, ideally in this order:

```
┌─────────────────────┐       ┌──────────────────────┐       ┌─────────────────────┐
│  Reviewer 1         │       │  Reviewer 2          │       │  Brainstormer       │
│  (style/correctness)│  ──▶  │  (research quality)  │  ──▶  │  (extensions/ideas) │
└─────────────────────┘       └──────────────────────┘       └─────────────────────┘
     Fix writing                 Fix science                     Propose next work
```

Running them in this order avoids Reviewer 2 flagging scientific issues that disappear once Reviewer 1's clarity fixes are applied, and avoids the Brainstormer proposing extensions to a paper that still has unresolved scientific concerns. But each role is independently useful and can be run alone on a finished paper, a draft, or even on a competing paper in the literature.

## Domain context baked into the prompts

All three prompts are aware of the project's accumulated knowledge base — the seven Kendiukhov papers in `../references/` and the eight pipeline specifications in `../pipelines/`. This means:

- **Reviewer 1** knows the standard notation in the project (ΔAUROC vs null baselines, blocked-permutation *p*, trustworthiness ≥ 0.80, strict max-null margins, etc.) and can catch inconsistencies with the project's conventions.
- **Reviewer 2** knows the failure modes that recur across this literature: the co-expression confound, the feature-shuffle-null trap, the annotation-database-circularity trap, the selective reporting of positive findings, the "looks great until you control properly" pattern, the immune-tissue concentration pattern, and so on. It proactively tests for these.
- **The Brainstormer** knows which methodological facets have already been covered by the existing pipelines (attention-based interpretability, spectral geometry, 141-hypothesis topological screening, SAE atlas + circuit tracing + exhaustive mapping, manifold discovery-extraction-compactification) and can suggest ideas that compose across them rather than reinventing ground already covered.

## When NOT to use these prompts

- **Not for reviewing papers outside this domain.** The domain-specific failure modes, statistical conventions, and accumulated knowledge baked in are specific to mechanistic interpretability of biological foundation models. For generic paper review, strip the domain context and use a general reviewer prompt.
- **Not as a substitute for reading the paper.** These prompts produce structured critiques but do not replace careful reading. Use them to surface issues that structured analysis catches and manual reading might miss (consistency across sections, missing control comparisons, unclaimed opportunities for extension).
- **Not for adversarial reviewing of competitors.** These prompts are designed to be constructive — Reviewer 2 will flag issues but will also credit strengths, and the Brainstormer proposes ideas that help the paper, not ideas that undermine it.

## Usage

Each prompt file is self-contained. Copy the prompt into a new Claude conversation (or equivalent), paste the paper text (or attach the PDF) after the prompt, and the agent will return the structured output specified in the prompt.

For multi-paper analysis (e.g., "review all 7 project papers for consistency across claims"), use the prompts as templates and add a project-level context block that lists the papers and their headline findings at the top of the agent's input.

## Conventions used in the prompts

- **Severity levels** (Reviewer 1): `critical` / `major` / `minor` / `nit`
- **Decision levels** (Reviewer 2): `accept` / `minor-revise` / `major-revise` / `reject` / `reject-and-resubmit`
- **Ambition tiers** (Brainstormer): `cheap-win` (high probability, low cost) / `moderate` (high-value, medium cost) / `moonshot` (high-risk high-reward)
- **Domain shorthand**: the prompts use the project's standard abbreviations (SCFM = single-cell foundation model, BH-*q* = Benjamini-Hochberg corrected *q*-value, TRRUST/STRING/GO/KEGG/Reactome/DoRothEA for the standard ontology databases).
