# maxtoki-mechinterp

Mechanistic-interpretability pipelines for single-cell foundation models, with
a deployment study on **MaxToki-217M** and a reusable audit framework that
codifies recurring peer-reviewer-flagged failure modes.

This repository accompanies the paper *"An Agent-Driven Pipeline for
Mechanistic Interpretability of Single-Cell Foundation Models: Findings on
MaxToki-217M and a Reusable Audit Framework"* (anonymous submission to the
ICML 2026 Mechanistic Interpretability Workshop).
PDF: [`projects/maxtoki/paper/main.pdf`](projects/maxtoki/paper/main.pdf).

---

## What this repository is

This is a **specification hub**, not a software package. There is no build
system, no test suite, and no application to run. The product is the set of
markdown pipeline specifications under [`pipelines/`](pipelines/), which are
written to be consumed by an LLM-based research agent (Claude, Codex) as
executable contracts: given a pipeline file plus a target model checkpoint,
an agent should be able to reproduce the methodology end-to-end without
re-reading the source paper.

The repository also contains:

- **The recurring-issue audit framework** — ten reviewer-derived
  methodological patterns codified as agent-applicable diagnostics.
- **Reviewer prompts** — three role-based prompts (style/correctness,
  research quality, brainstorming) designed to be composed on a draft
  paper.
- **The MaxToki-217M deployment** — per-pipeline final summaries and audit
  reports under [`projects/maxtoki/`](projects/maxtoki/).
- **The submitted paper** — LaTeX source, bibliography, and compiled PDF
  under [`projects/maxtoki/paper/`](projects/maxtoki/paper/).

---

## Repository layout

```
.
├── pipelines/                     # Model-agnostic mech-interp specifications
│   ├── attention-grn-extraction-and-evaluation.md
│   ├── residual-stream-spectral-geometry.md
│   ├── topology-geometry-141-hypotheses.md
│   ├── manifold-discovery-extraction-compactification.md
│   ├── longevity-mechinterp-donor-aware.md
│   ├── audit-recurring-review-issues.md   # The 10-pattern audit framework
│   └── sparse-autoencoders/                # SAE mega-pipeline (3 stages)
│       ├── README.md
│       ├── 01-sae-atlas.md
│       ├── 02-causal-circuit-tracing.md
│       ├── 03-exhaustive-mapping-and-steering.md
│       └── 04-atlas-deployment.md
│
├── prompts/                        # Reviewer / brainstormer agent prompts
│   ├── README.md
│   ├── reviewer-1-style-correctness.md
│   ├── reviewer-2-research-quality.md
│   └── brainstormer-ideas.md
│
└── projects/
    └── maxtoki/                    # MaxToki-217M deployment artefacts
        ├── audits/                 # Audit reports + bootstrap CI artefacts
        ├── summaries/              # Per-pipeline final summaries
        └── paper/                  # ICML 2026 Mech-Interp Workshop submission
            ├── main.tex
            ├── main.pdf
            └── references.bib
```

---

## The 10-section pipeline template

Every file in `pipelines/` follows this template, which is enforced
(missing sections cause an agent to refuse execution):

1. **Overview** (2–4 sentences)
2. **Source** — paper citation, arXiv link, pinned reference-implementation
   commit hash
3. **Inputs** — data types, formats, shapes, preprocessing assumptions
4. **Outputs** — artefacts produced, formats, destinations
5. **Dependencies** — Python/R/system packages, models, datasets, hardware
6. **Methodology** — numbered steps with enough detail (equations,
   hyperparameters inline) that an agent can implement without the paper
7. **Code references** — pointers into pinned reference implementations,
   with critical inline excerpts
8. **Parameters** — table with defaults and valid ranges
9. **Validation** — expected metric ranges, sanity plots, how to tell it
   ran correctly
10. **Known pitfalls** — non-obvious failure modes, environment quirks

The template is itself an agentic affordance: reviewers' recurring
objections cluster around predictable omissions (missing null model,
missing trivial baseline, missing positive control, missing scope
qualifier), each of which gets a named slot with non-empty content.

---

## The audit framework

[`pipelines/audit-recurring-review-issues.md`](pipelines/audit-recurring-review-issues.md)
codifies ten recurring methodological objections extracted from six
independent peer-reviewer comment sets in the authors' prior work:

| ID  | Pattern                                                              |
|-----|----------------------------------------------------------------------|
| P1  | Weak null / not anchored to chance                                   |
| P2  | Trivial / gene-level baseline absent                                 |
| P3  | Endpoint–object mismatch                                             |
| P4  | Pseudoreplication / leaky CV / wrong unit-of-inference               |
| P5  | Selection-driven inflation / double-dipping                          |
| P6  | No positive control for null findings                                |
| P7  | Single-condition generalisation                                      |
| P8  | Stability of headline numbers (CIs, threshold sensitivity, FDR)      |
| P9  | Causal / regulatory language overreach                               |
| P10 | Reproducibility scaffolding                                          |

Each pattern has a diagnostic signature (what to grep / read for in run
artefacts), a prescription (how to fix or qualify), and a citation back
to the review documents that established the pattern.

In the MaxToki-217M deployment the audit detected three load-bearing
flaws (a sign-tracking bug, an unanchored chance baseline, missing
bootstrap CIs) and inverted one previously-stated negative.
See [`projects/maxtoki/audits/`](projects/maxtoki/audits/) for the
verdict matrix and action proposals.

---

## MaxToki-217M deployment

The MaxToki-217M deployment applies eight pipelines to a recently
released Llama-architecture single-cell foundation model with
trajectory-aware pre-training (∼1T gene tokens / ∼175M cells).

| Pipeline                  | Probe                                  |
|---------------------------|----------------------------------------|
| `spectral-geometry`       | residual-stream SVD geometry           |
| `attention-grn` (×4 runs) | attention → GRN claim                  |
| `topology-141`            | 141-hypothesis topology screen         |
| `sae-atlas`               | per-layer TopK SAEs (∼5K feat./layer)  |
| `sae-circuit-tracing`     | 2.1M-edge causal circuit               |
| `sae-exhaustive-mapping`  | 4.97M-edge map + activation steering   |
| `manifold-discovery`      | hematopoietic developmental axis (H65) |
| `longevity-mechinterp`    | donor-aware aging probe                |

Per-pipeline summaries live under
[`projects/maxtoki/summaries/`](projects/maxtoki/summaries/);
audit reports under [`projects/maxtoki/audits/`](projects/maxtoki/audits/);
the final paper under
[`projects/maxtoki/paper/`](projects/maxtoki/paper/).

The deployment yields six interrelated findings that jointly characterise
MaxToki-217M as encoding cell **identity geometry** but not directed
**regulatory mechanism**. See the paper for the full account.

---

## Reproducing the deployment

Run artefacts (raw extracted activations, intermediate SAE checkpoints,
trajectory-steering tensors) are not included in this repository because
of size. To reproduce the deployment:

1. Obtain the pinned model checkpoints (MaxToki-217M, scGPT, Geneformer
   V2-316M) from their public HuggingFace releases. Per-target weight
   download and tokenizer wiring instructions are embedded in each
   pipeline file (Section 5: *Dependencies*).
2. Clone the pinned reference implementations referenced in each
   pipeline's Section 7 (*Code references*). External reference repos
   are cited by URL + commit hash; this repository does not vendor
   them.
3. Execute pipelines in any order; each is self-contained given its
   inputs (typically: target model checkpoint + a public dataset).
4. Run the audit pipeline
   [`pipelines/audit-recurring-review-issues.md`](pipelines/audit-recurring-review-issues.md)
   against the produced artefacts to verify methodological hygiene.

Hardware used in the published deployment: a single A100-80GB for
forward passes; CPU for SAE training and analysis. Per-pipeline
wall-clock is documented in Table 2 of the paper.

---

## Recommended citation

```bibtex
@inproceedings{anon2026maxtokimechinterp,
  title={An Agent-Driven Pipeline for Mechanistic Interpretability
         of Single-Cell Foundation Models: Findings on
         {MaxToki-217M} and a Reusable Audit Framework},
  author={Anonymous},
  booktitle={ICML 2026 Mechanistic Interpretability Workshop},
  year={2026}
}
```

The seven prior single-author works that introduced the pipeline
catalogue (currently anonymised arXiv preprints 2602.17532, 2602.22247,
2602.22289, 2603.01752, 2603.02952, 2603.10261, 2603.11940) will be
listed here on de-anonymisation.

---

## Licence

Code, pipeline specifications, and prompts are released under the
**Apache License 2.0** (see [`LICENSE`](LICENSE)).
The submitted paper PDF and bibliography are released under
**CC BY 4.0**.

---

## Status

Active research repository. The SAE sub-pipeline stubs
(`pipelines/sparse-autoencoders/01-sae-atlas.md`, `02-…`, `03-…`) are
known to be incomplete; the
[`sparse-autoencoders/README.md`](pipelines/sparse-autoencoders/README.md)
is the authoritative composition guide until the stubs are filled in.

Issues and reproducibility questions are welcome via the GitHub issue
tracker.
