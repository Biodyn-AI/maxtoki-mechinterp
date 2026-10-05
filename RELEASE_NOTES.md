# Release notes

Built by `projects/maxtoki/release/build_release.py` (the builder itself is not part of this tree). Newest source file: 2026-10-05 01:21 (local time).

## What is here

| Area | Files | Size |
|---|---|---|
| `CITATION.cff` | 1 | 1.7 KB |
| `CONTRIBUTING.md` | 1 | 3.8 KB |
| `LICENSE` | 1 | 11.4 KB |
| `NOTICE` | 1 | 14.9 KB |
| `README.md` | 1 | 14.0 KB |
| `controlled-evaluation/protocol` | 11 | 128.1 KB |
| `controlled-evaluation/quarantine` | 1 | 1.1 KB |
| `controlled-evaluation/quarantine/2026-10-02_stray_files_in_task_folders` | 3 | 17.3 KB |
| `controlled-evaluation/studyA/build` | 18 | 252.4 KB |
| `controlled-evaluation/studyA/keys` | 69 | 583.8 KB |
| `controlled-evaluation/studyA/results` | 49 | 11.1 MB |
| `controlled-evaluation/studyA/tasks` | 68 | 10.4 MB |
| `controlled-evaluation/studyA_haiku2` | 1 | 46.1 KB |
| `controlled-evaluation/studyA_haiku2/results` | 6 | 123.4 KB |
| `controlled-evaluation/studyB/keys` | 49 | 376.1 KB |
| `controlled-evaluation/studyB/packages` | 240 | 9.6 MB |
| `controlled-evaluation/studyB/results` | 9 | 4.8 MB |
| `controlled-evaluation/studyC` | 1 | 15.6 KB |
| `controlled-evaluation/studyC/build` | 9 | 1.1 MB |
| `controlled-evaluation/studyC/results` | 10 | 3.0 MB |
| `controlled-evaluation/tools` | 26 | 343.3 KB |
| `pipelines` | 11 | 675.8 KB |
| `projects/maxtoki/README.md` | 1 | 21.3 KB |
| `projects/maxtoki/audits` | 6 | 76.4 KB |
| `projects/maxtoki/checks` | 165 | 7.9 MB |
| `projects/maxtoki/framework-eval` | 14 | 1.5 MB |
| `projects/maxtoki/paper` | 1 | 8.6 KB |
| `projects/maxtoki/paper-plos-one` | 83 | 5.3 MB |
| `projects/maxtoki/requirements.txt` | 1 | 1.2 KB |
| `projects/maxtoki/runs/attention-grn-217M` | 381 | 33.5 MB |
| `projects/maxtoki/runs/circuit-tracing-217M` | 3,068 | 95.8 MB |
| `projects/maxtoki/runs/exhaustive-mapping-217M` | 1,963 | 339.1 MB |
| `projects/maxtoki/runs/longevity-mechinterp-217M` | 14 | 365.1 KB |
| `projects/maxtoki/runs/manifold-discovery-217M` | 872 | 130.8 MB |
| `projects/maxtoki/runs/sae-atlas-217M` | 649 | 109.4 MB |
| `projects/maxtoki/runs/spectral-geometry-217M` | 191 | 22.5 MB |
| `projects/maxtoki/runs/topology-141-217M` | 408 | 4.8 MB |
| `projects/maxtoki/setup` | 10 | 698.2 KB |
| `projects/maxtoki/summaries` | 18 | 2.0 MB |
| `projects/maxtoki/verification` | 107 | 1.6 MB |
| `prompts` | 4 | 88.1 KB |
| **total copied** | **8,542** | **798.0 MB** |

## What is not here, and where to find it

- Model weights: see `CHECKPOINTS.md` (Hugging Face ids, revisions, sha256).
- Input datasets: see `DATASETS.md` (public sources, version ids, sha256).
- 2,146 files over 2 MB (108.4 GB) and the 1,862 files of the Study C snapshot (221.7 MB) are not in the git repository. `ZENODO_MANIFEST.csv` lists each one with its size, sha256, the Zenodo file that holds it
  and a description. `python release_tools/fetch_zenodo.py --get` downloads them and puts them back at their paths
  (a full fetch needs about 83 GB free: about 41 GB of downloads kept in `.zenodo_cache/`, which can be
  deleted afterwards or put elsewhere with `--cache`, plus about 42 GB of placed files).
  36 of them are not deposited (67.0 GB); the `deposited` column says why and how to rebuild or re-extract them.
- Files the manuscript's results map cites stay in the repository up to 50 MB.
- Everything else left out, with the reason: `EXCLUDED.csv` (118 rows, 6.0 GB).
- `projects/maxtoki/paper-plos-one/` holds the manuscript source (`draft/`, `main.tex`, `references.bib`,
  `plos2025.bst`), the figure scripts and their source data, the figures, the supporting information
  and the build and check scripts; not the working files of the manuscript.
- Sources that did not exist at build time (the builder takes them when they appear): none.
- `controlled-evaluation/studyC/snapshot/`, the copy of project files handed to the blind Study C auditors, is on
  Zenodo only (`files_eval_studyC.zip`; its files are rows of ZENODO_MANIFEST.csv). 1,796 of its 1,862 files are also elsewhere in this release (git or ZENODO_MANIFEST.csv) with the same sha256. The other 66 exist only in the snapshot (for example the summaries rebuilt with audit text removed, and large-file listings). Six of them, the third-party reference files under reference_data/, are not deposited (see the deposited column of ZENODO_MANIFEST.csv).
  `SNAPSHOT_MANIFEST.md` and `studyC/build/snapshot_files_sha256.tsv` list what it held.

## Things to know before using this tree

- Absolute local paths in text files were replaced by placeholders (`PATH_MAP.md`). Set them before
  running scripts (`release_tools/localize_paths.py`).
- 510 copied files differ from their source because of that replacement. Their original
  sha256 is in `RELEASE_MANIFEST.csv`.
- `RESULTS_MAP.csv` lists every number in the manuscript with the release file, field and value it comes from.
- `projects/maxtoki/README.md`, `projects/maxtoki/summaries/`, `projects/maxtoki/audits/` and the run README and
  FINAL_SUMMARY files are the deployment's original, uncorrected record. The confirmed errors are in S1 Table;
  the corrected results are in S3 Appendix (see README.md, "Read this first").
- Runs used Apple Silicon (`mps`); no run log or config names CUDA or an A100 (deployment facts).
- The MaxToki revision in CHECKPOINTS.md is a post-hoc pin, checked by sha256.
- No package list was recorded for the second (Python 3.11) environment (`ENVIRONMENT.md`).
