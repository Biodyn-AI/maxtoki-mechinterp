# maxtoki-mechinterp

Method specifications for interpretability analyses of single-cell foundation models, written
for coding agents. The repository also holds the full record of one agent-run deployment of
these specifications on MaxToki-217M, an error ledger of that deployment, three controlled
studies of the specifications and of a ten-item audit checklist, and the corrected
MaxToki-217M analyses.

It supports this manuscript (under submission to PLOS ONE):

> Ihor Kendiukhov. *Checking agent-run interpretability of a single-cell foundation model: a
> controlled evaluation of method specifications and a reviewer-derived audit checklist, with a
> MaxToki-217M case study.*

[`RESULTS_MAP.csv`](RESULTS_MAP.csv) links every number in the manuscript to the file, field and
value it comes from.

## Read this first: the original record is not corrected

These files are the deployment's original record (April–May 2026). They are kept as they were
written, and **many of their headline numbers are wrong**. (The only change is a short warning
note at the top of `projects/maxtoki/README.md` and in `projects/maxtoki/summaries/README.md`.)

- [`projects/maxtoki/README.md`](projects/maxtoki/README.md)
- [`projects/maxtoki/summaries/`](projects/maxtoki/summaries/)
- [`projects/maxtoki/audits/`](projects/maxtoki/audits/)
- the `README.md` and `FINAL_SUMMARY.md` files under [`projects/maxtoki/runs/`](projects/maxtoki/runs/)

The 149 confirmed errors in the deployment's code, logs, outputs, run summaries and audit reports
are listed in S1 Table,
[`projects/maxtoki/paper-plos-one/supporting/S1_Table.csv`](projects/maxtoki/paper-plos-one/supporting/S1_Table.csv).
The corrected analyses are described in S3 Appendix,
[`projects/maxtoki/paper-plos-one/supporting/S3_Appendix.pdf`](projects/maxtoki/paper-plos-one/supporting/S3_Appendix.pdf).
Their outputs are the `v2_`, `v2b_` and `v3_` folders under the runs' `outputs/` folders, and the
`V2_`, `V2B_` and `V3_` reports in the run folders (the longevity run has none). Use those, not
the original summaries.

## What is where

### Method specifications

- [`pipelines/`](pipelines/): one markdown file per method, written for a coding agent. See
  *Layout of the pipeline files* below.
  [`pipelines/audit-recurring-review-issues.md`](pipelines/audit-recurring-review-issues.md) is the
  ten-item audit checklist (P1–P10).
  [`pipelines/sparse-autoencoders/README.md`](pipelines/sparse-autoencoders/README.md) explains how
  the three sparse-autoencoder stages fit together.
- [`prompts/`](prompts/): three role prompts for a paper draft (copy editor, scientific referee, and
  ideas for follow-up work). See [`prompts/README.md`](prompts/README.md).

### The MaxToki-217M deployment: [`projects/maxtoki/`](projects/maxtoki/)

| Folder | What it holds |
|---|---|
| [`runs/`](projects/maxtoki/runs/) | one folder per pipeline run (eight in all): scripts, configs, logs and outputs. The `v2_`, `v2b_` and `v3_` outputs and the `V2_`, `V2B_`, `V3_` reports are the corrected analyses. |
| [`checks/`](projects/maxtoki/checks/) | deterministic checks of the specifications and the runs (no model, GPU or agent needed). Built on 2026-10-01; see [`checks/README.md`](projects/maxtoki/checks/README.md). |
| [`framework-eval/`](projects/maxtoki/framework-eval/) | the working error ledger (`error_ledger.csv`, `error_ledger.md`) and the scripts that built and checked it |
| [`verification/`](projects/maxtoki/verification/) | code and data of the independent verification cited in S1 Table |
| [`summaries/`](projects/maxtoki/summaries/), [`audits/`](projects/maxtoki/audits/) | the deployment's own run summaries and audit reports (original record, not corrected) |
| [`setup/`](projects/maxtoki/setup/) | model adapter, hooks, input and data loaders, sparse-autoencoder code and tests, the MaxToki token dictionary and the small model config files (no weights) |
| [`paper-plos-one/`](projects/maxtoki/paper-plos-one/) | manuscript source (`main.tex`, `draft/`, `references.bib`), supporting information (`supporting/`: S1 Table, S2 Appendix, S3 Appendix), figure scripts (`figures-src/`), figure source data (`source-data/`), figures (`submission-figures/`), and the build and check scripts |
| [`paper/`](projects/maxtoki/paper/) | the deployment agent's LLM usage note |

[`projects/maxtoki/requirements.txt`](projects/maxtoki/requirements.txt) is the frozen package list
of the main Python environment.

### The controlled studies: [`controlled-evaluation/`](controlled-evaluation/)

| Folder | What it holds |
|---|---|
| [`protocol/`](controlled-evaluation/protocol/) | the pre-registration, its amendments, the freeze log, the generic and deployed checklists, the Study B items and the Study C key |
| [`tools/`](controlled-evaluation/tools/) | the agent workflow scripts and the analysis and check code |
| [`studyA/`](controlled-evaluation/studyA/) | four known-answer tasks, each with and without the specification (`tasks/`), answer keys (`keys/`), build files (`build/`) and results (`results/`). The copy of each task's source paper that the agents received (`methods/source_method_paper.pdf`, in both versions of each task) is not included. The four papers are the author's arXiv preprints [2602.17532](https://arxiv.org/abs/2602.17532) (T1), [2603.01752](https://arxiv.org/abs/2603.01752) (T2), [2603.02952](https://arxiv.org/abs/2603.02952) (T3) and [2602.22289](https://arxiv.org/abs/2602.22289) (T4). |
| [`studyB/`](controlled-evaluation/studyB/) | held-out analyses from other projects, each in a flawed and a corrected version (`packages/`), keys and results |
| [`studyC/`](controlled-evaluation/studyC/) | the blind re-audit of the deployment: build files and results. The snapshot of project files that the auditors received (`studyC/snapshot/`) is kept on Zenodo (`files_eval_studyC.zip`), except six third-party reference files (see *Large files and Zenodo*); [`SNAPSHOT_MANIFEST.md`](controlled-evaluation/studyC/SNAPSHOT_MANIFEST.md) describes it. |
| [`studyA_haiku2/`](controlled-evaluation/studyA_haiku2/) | the exploratory Study A arm with Claude Haiku 4.5 |
| [`quarantine/`](controlled-evaluation/quarantine/) | three stray files that agents of the first Haiku run wrote into task folders; see [`quarantine/README.md`](controlled-evaluation/quarantine/README.md) |

### Release tools: [`release_tools/`](release_tools/)

- [`fetch_zenodo.py`](release_tools/fetch_zenodo.py) downloads the files kept on Zenodo.
- [`localize_paths.py`](release_tools/localize_paths.py) fills in the path placeholders.

### Top-level documents

| File | What it says |
|---|---|
| [`RESULTS_MAP.csv`](RESULTS_MAP.csv) | every number in the manuscript, with its source file, field and value |
| [`ZENODO_MANIFEST.csv`](ZENODO_MANIFEST.csv) | every file not in git: size, sha256, the Zenodo file that holds it, and a description |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | where to get the model weights (MaxToki, Geneformer, scGPT), with revisions and sha256 |
| [`DATASETS.md`](DATASETS.md) | every input dataset the runs read: public source, version, size, sha256 |
| [`DATASETS_EVAL_INPUTS.csv`](DATASETS_EVAL_INPUTS.csv) | every data file inside the controlled-study packages, with sha256 |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Python versions, package list, and the library versions and devices that runs recorded |
| [`PATH_MAP.md`](PATH_MAP.md) | the path placeholders and what to set for each |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | what is in this tree, size per area, what is missing, and the Zenodo record link |
| [`RELEASE_MANIFEST.csv`](RELEASE_MANIFEST.csv) | every file in the tree, with size and sha256 (and the original sha256 when a path was replaced) |
| [`EXCLUDED.csv`](EXCLUDED.csv) | everything left out of the release, with the reason |
| [`CITATION.cff`](CITATION.cff), [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) | how to cite, the licence, and third-party files and their licences |

## Large files and Zenodo

- Files up to 2 MB (2,000,000 bytes) are in git.
- Files over 2 MB are not in git. [`ZENODO_MANIFEST.csv`](ZENODO_MANIFEST.csv) lists each one.
  They are kept on Zenodo; the record link is in [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
- Exception: files that `RESULTS_MAP.csv` cites stay in git up to 50 MB.
- The snapshot of project files given to the blind Study C auditors
  (`controlled-evaluation/studyC/snapshot/`, 1,862 files) is kept on Zenodo too, whatever the file size.
- The exception is six third-party reference files in that snapshot, under
  `controlled-evaluation/studyC/snapshot/maxtoki/reference_data/`: the GO biological process, KEGG and
  Reactome gene sets, the STRING protein pairs, the TRRUST human table and a Geneformer gene-id
  dictionary. They are not deposited. [`DATASETS.md`](DATASETS.md) and
  [`CHECKPOINTS.md`](CHECKPOINTS.md) say where to get them. So 1,856 snapshot files are deposited.
- 30 large files are not deposited. The `deposited` column of the manifest says why, and how to
  rebuild or re-extract each one.

To put the Zenodo files back at their paths:

```bash
python release_tools/fetch_zenodo.py                      # show what would be downloaded
python release_tools/fetch_zenodo.py --get                # download, check sha256, place files
python release_tools/fetch_zenodo.py --get --only attention-grn-217M   # only matching paths
python release_tools/fetch_zenodo.py --get --cache /big/disk/zenodo_cache
```

A full fetch needs about 83 GB of free space: about 41 GB of downloads, kept in `.zenodo_cache/`,
plus about 42 GB of placed files. `--cache` puts the downloads somewhere else. You can delete
them once the files are placed. If the script says no record id is set, pass `--record <id>`
with the number at the end of the Zenodo DOI in `RELEASE_NOTES.md`.

## Path placeholders

Absolute local paths in copied text files were replaced by placeholders such as `<REPO_ROOT>`,
`<DATA_ROOT>` and `<HF_CACHE>`. [`PATH_MAP.md`](PATH_MAP.md) lists them and says what to set for
each. Scripts will not run until they are set. Text files fetched from Zenodo carry the same
placeholders.

```bash
python release_tools/localize_paths.py                                # show what would change
DATA_ROOT=/data/maxtoki python release_tools/localize_paths.py --apply  # change the files
```

Each value is read from an environment variable with the placeholder's name, without the angle
brackets. `<HOME>` is the exception: it is read from `RELEASE_HOME`. `<REPO_ROOT>` defaults to
this checkout and `<EVAL_ROOT>` to its `controlled-evaluation/` folder.

## Model weights and input data

Model weights and the full input datasets are not in this repository.
[`CHECKPOINTS.md`](CHECKPOINTS.md) and [`DATASETS.md`](DATASETS.md) say where to get each file
and give its sha256.

The folders `repos/` and `references/` that the pipeline files name are not shipped. Neither is
`review_plans/`, which the audit pipeline names as its source material. Nor is `CLAUDE.md`, the
instruction file of the working repository, which the audit pipeline also cites. Its list of
recurring failure modes is repeated in section 5 (*Dependencies*) of
[`pipelines/audit-recurring-review-issues.md`](pipelines/audit-recurring-review-issues.md). To follow a code
reference, clone the repository named in the pipeline file's *Source* section and check out the
pinned commit given there; paths of the form `repos/<name>/...` point into that clone. Get the
papers from their publishers.

## Layout of the pipeline files

New pipeline files should have these ten sections, in this order: 1 Overview, 2 Source, 3 Inputs,
4 Outputs, 5 Dependencies, 6 Methodology, 7 Code references, 8 Parameters, 9 Validation,
10 Known pitfalls. Code is cited as `repos/<name>/path/file.py:LINE`, with the pinned commit in
the *Source* section.

The existing files differ:

| File | Sections |
|---|---|
| [`audit-recurring-review-issues.md`](pipelines/audit-recurring-review-issues.md) | the ten template sections, with *Code references* as section 7 |
| [`longevity-mechinterp-donor-aware.md`](pipelines/longevity-mechinterp-donor-aware.md) | the ten template sections, plus 11 *Porting to a new target model* |
| [`attention-grn-extraction-and-evaluation.md`](pipelines/attention-grn-extraction-and-evaluation.md), [`manifold-discovery-extraction-compactification.md`](pipelines/manifold-discovery-extraction-compactification.md), [`residual-stream-spectral-geometry.md`](pipelines/residual-stream-spectral-geometry.md), [`topology-geometry-141-hypotheses.md`](pipelines/topology-geometry-141-hypotheses.md), [`sparse-autoencoders/01-sae-atlas.md`](pipelines/sparse-autoencoders/01-sae-atlas.md), [`02-causal-circuit-tracing.md`](pipelines/sparse-autoencoders/02-causal-circuit-tracing.md), [`03-exhaustive-mapping-and-steering.md`](pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md) | no separate *Code references* section: code references are given inside each Methodology step. Sections 7–9 are Parameters, Validation and Known pitfalls, and section 10 is a *Quick-start* for a new model. |
| [`sparse-autoencoders/04-atlas-deployment.md`](pipelines/sparse-autoencoders/04-atlas-deployment.md) | a deployment guide for the interactive web atlas, with its own sections |

The manifold-discovery file has no reference repository: its *Source* section says the paper
names none. The audit pipeline has no pinned commit, because it describes a method only.

## Earlier public version

The previous public version of this repository (version 0.1.0, May 2026) is tag `v0.1.0`.

## How to cite

Please cite the manuscript. [`CITATION.cff`](CITATION.cff) has the details.

## Licence

The code and text written for this repository are under the Apache License 2.0
([`LICENSE`](LICENSE)). Some files are copies or subsets of public data, reference networks and
model files. They keep their own licences. [`NOTICE`](NOTICE) lists them with their sources.
