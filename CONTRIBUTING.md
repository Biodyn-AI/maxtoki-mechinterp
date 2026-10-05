# Contributing

Thanks for your interest in this repository.

## What is here

- `pipelines/` and `prompts/`: method specifications written for coding agents, and the
  review prompts.
- `projects/maxtoki/`: the MaxToki-217M deployment (code, configs, logs and outputs under
  `runs/`), the checks and the independent verification, the error ledger, and the
  manuscript source.
- `controlled-evaluation/`: the protocol, tasks, answer keys, agent deliverables, reviews,
  grades and analysis code of the controlled studies.
- `RESULTS_MAP.csv`: every number in the manuscript, with the file, field and value it comes
  from.
- Files larger than 2 MB are not in the repository, except files cited by `RESULTS_MAP.csv`,
  up to 50 MB. `ZENODO_MANIFEST.csv` lists them with size and sha256, and
  `python release_tools/fetch_zenodo.py --get` puts them back. `CHECKPOINTS.md` and
  `DATASETS.md` say where to get the model weights and the input data.

## What we welcome

- **Reproducibility issues.** If a number in `RESULTS_MAP.csv` does not match its source
  file, or you cannot reproduce a result under `projects/maxtoki/runs/` or
  `controlled-evaluation/`, please open an issue. Give the row id or file, the step where
  your result differs, and what you obtained.
- **Specification corrections.** Typos, unclear method steps, missing hyperparameters,
  broken links.
- **Checklist additions.** If a recurring methodological objection is not covered by the
  ten checklist items (P1–P10), propose an addition with a concrete example, how to find it
  in an analysis, and how to fix it.
- **Runs on other models.** If you ran a specification on another foundation model, a pull
  request with the run under `projects/<target>/` is welcome.

## Please do not

- Change files under `projects/maxtoki/runs/` or `controlled-evaluation/` in place. They
  are the record behind the manuscript. Add new files next to them instead.
- Add large files. Put files over 2 MB on an external archive and link them.

## Specification file conventions

New files in `pipelines/` should have the ten sections of the template, in this order:
Overview, Source, Inputs, Outputs, Dependencies, Methodology, Code references, Parameters,
Validation, Known pitfalls.

The existing files differ:

- `audit-recurring-review-issues.md` and `longevity-mechinterp-donor-aware.md` have the ten
  sections. The longevity file adds an eleventh, *Porting to a new target model*.
- `attention-grn-extraction-and-evaluation.md`, `manifold-discovery-extraction-compactification.md`,
  `residual-stream-spectral-geometry.md`, `topology-geometry-141-hypotheses.md` and
  `sparse-autoencoders/01-sae-atlas.md`, `02-causal-circuit-tracing.md` and
  `03-exhaustive-mapping-and-steering.md` have no separate *Code references* section. They give
  the code references inside each Methodology step, and end with a *Quick-start* section.
- `sparse-autoencoders/04-atlas-deployment.md` is a deployment guide for the web atlas, with its
  own sections. `sparse-autoencoders/README.md` explains how the three stages fit together.

When citing code, use the `repos/<name>/path/file.py:LINE` form for pinned reference
implementations, with the commit hash in the specification's *Source* section. Do not link
to GitHub blob URLs; they break when a branch is rewritten.

## Checklist

Changes to `pipelines/audit-recurring-review-issues.md` (P1–P10, how to find each issue and
how to fix it) need a justification that cites at least one concrete reviewer comment.

## Issue labels

- `reproducibility`: a documented result or a `RESULTS_MAP.csv` row cannot be reproduced
- `pipeline-spec`: an error or unclear step in a specification
- `audit-framework`: an addition or correction to P1–P10
- `documentation`: README or other documentation
- `question`: a request for clarification
