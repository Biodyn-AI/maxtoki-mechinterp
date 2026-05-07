# Contributing

Thanks for your interest in this repository.

## Scope

This is a research repository containing pipeline specifications, audit
materials, and a paper. We welcome:

- **Reproducibility issues** — if you cannot reproduce a result documented
  under `projects/maxtoki/summaries/`, please open an issue with the
  pipeline name, the step where reproduction diverges, and the artefacts
  you obtained.
- **Pipeline corrections** — typos, ambiguous methodology, missing
  hyperparameters, broken external links.
- **Audit-pattern additions** — if you've encountered a recurring
  reviewer objection that isn't covered by P1–P10, propose an addition
  with: a concrete example, a diagnostic signature (what to grep for),
  and a prescription.
- **Pipeline ports** — if you've successfully run a pipeline against a
  foundation model not currently covered (e.g., scFoundation, UCE, a
  protein language model), a deployment-study PR is welcome under
  `projects/<target>/`.

## What we don't accept (yet)

- Runtime code in this repository. The pipelines are specifications,
  not implementations; they reference pinned external implementations
  by URL + commit hash, and are intended to be re-implemented per
  target model.
- Large run artefacts. Trained SAE checkpoints, per-layer activation
  caches, and trajectory-steering tensors should be hosted externally
  (HuggingFace, Zenodo) and linked from the relevant pipeline file.

## Pipeline file conventions

Every file in `pipelines/` must have all ten sections of the template
(Overview, Source, Inputs, Outputs, Dependencies, Methodology, Code
references, Parameters, Validation, Known pitfalls). PRs that omit a
section will be flagged in review.

When citing code, use the `repos/<name>/path/file.py:LINE` form for
pinned reference implementations, with a commit hash documented in the
pipeline's *Source* section. Do not link to GitHub blob URLs; they rot
on force-push.

## Audit framework

Changes to `pipelines/audit-recurring-review-issues.md` (P1–P10 plus
their diagnostic signatures and prescriptions) require a justification
referencing at least one concrete reviewer comment that motivates the
change.

## Issue tracker

Please use one of the issue labels:

- `reproducibility` — cannot reproduce a documented result
- `pipeline-spec` — error or ambiguity in a pipeline file
- `audit-framework` — addition / correction to P1–P10
- `documentation` — README or top-level documentation
- `question` — clarification request
