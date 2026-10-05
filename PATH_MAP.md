# Path placeholders

Copied text files had the author's absolute local paths replaced by the placeholders below.
The source files were not changed. The original sha256 of every changed file is in
`RELEASE_MANIFEST.csv` (column `source_sha256`). Hashes recorded inside other files (for example
MANIFEST.sha256 or deployment_facts.json) refer to the original bytes.

Files changed: 510. Replacements: 11,357.

Scripts will not run until the placeholders are set. Either edit the lines, or run
`python release_tools/localize_paths.py` (it reads the values from environment variables of the
same name, e.g. `DATA_ROOT=/data/maxtoki python release_tools/localize_paths.py --apply`; for `<HOME>`
set `RELEASE_HOME`).

The folder `projects/maxtoki/verification/` was called `paper-plos-one/revision/investigation/` on the
author's machine. Text that named the old place now names the new one (50 replacements).

Binary files are not changed. Two `.npz` files keep the original data path in a text field (`h5_path`):
`projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz` and
`projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/cells_tokens.npz` (both on Zenodo).

| Placeholder | What it stood for | What you must set | Replacements | Files | Example file |
|---|---|---|---|---|---|
| `<REPO_ROOT>` | the local copy of the biomi_automation hub repository | the root of this release checkout (the folder holding pipelines/ and projects/). Paths under <REPO_ROOT>/repos/ and <REPO_ROOT>/references/ are not shipped: clone the pinned repos named in the pipeline files, and get the papers from their publishers. | 5,395 | 340 | `projects/maxtoki/runs/attention-grn-217M/outputs/phase0.log` |
| `<EVAL_ROOT>` | the controlled-evaluation workspace (maxtoki-framework-eval), kept outside the hub on purpose | <REPO_ROOT>/controlled-evaluation in this release. | 1,324 | 64 | `projects/maxtoki/framework-eval/error_ledger.md` |
| `<DATA_ROOT>` | the local data folder that holds the input datasets and reference networks | a folder where you put the input files listed in DATASETS.md, keeping the same sub-paths (for example <DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad). | 480 | 192 | `projects/maxtoki/setup/dataset_loader.py` |
| `<CODE_ROOT>` | the folder that holds the author's other code checkouts (not part of this release) | nothing, unless you re-run the controlled studies: then set it to a folder for agent work folders and create <CODE_ROOT>/analysis-tasks/bin/python as a small shell wrapper that exports OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONWARNINGS=ignore and runs the Python environment built from projects/maxtoki/requirements.txt (the original wrapper is not shipped). controlled-evaluation/bin/python, a wrapper that runs the same environment, is not shipped either. | 2,929 | 23 | `projects/maxtoki/paper-plos-one/supporting/build_s2.py` |
| `<LOCAL_VOLUME>` | the external drive that held the project (any other path on it) | nothing; these lines point at files that are not shipped. | 685 | 13 | `projects/maxtoki/paper-plos-one/supporting/build_s2.py` |
| `<HF_CACHE>` | the Hugging Face cache folder (~/.cache/huggingface) | your Hugging Face cache (HF_HOME). Model revisions are in CHECKPOINTS.md. | 41 | 22 | `projects/maxtoki/runs/spectral-geometry-217M/autoloop/iterations/iter_0003/h3del_lid_confound.py` |
| `<CONDA_ROOT>` | an Anaconda install with Python 3.11, used as a second environment for some runs | a Python environment with the packages in ENVIRONMENT.md (its own package list was not recorded). | 94 | 12 | `projects/maxtoki/runs/manifold-discovery-217M/STATUS.md` |
| `<CLAUDE_CLI>` | the Claude Code command-line program called by the spectral-geometry autoloop driver | the path to your `claude` CLI, only if you re-run the autoloop. | 12 | 5 | `projects/maxtoki/runs/spectral-geometry-217M/autoloop/run_maxtoki_autoloop.py` |
| `<CLAUDE_HOME>` | the Claude Code settings and session-transcript folder (~/.claude); its contents are not shipped | nothing; these lines name agent transcripts that are not part of this release. | 22 | 4 | `projects/maxtoki/checks/d12_deployment_facts/run_config.json` |
| `<HOME>` | the author's home folder (anything else under it) | your home folder, if a line needs it. | 6 | 4 | `projects/maxtoki/paper-plos-one/supporting/build_s2.py` |
| `<AGENT_TMP>` | per-session scratch folders of the coding agent (temporary; contents not shipped) | nothing; these are temporary working folders. | 61 | 30 | `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/logs/verify.log` |
| `<SYS_TMP>` | the macOS per-user temporary folder | nothing. | 0 | 0 |  |
| `<TMP>` | the system temporary folder | any temporary folder, e.g. /tmp. | 168 | 27 | `projects/maxtoki/runs/circuit-tracing-217M/scripts/remaining_phases.py` |
| `<HOMEBREW_PREFIX>` | the Homebrew install prefix on macOS | nothing. | 3 | 3 | `projects/maxtoki/runs/attention-grn-217M/README.md` |
| `<SESSION_DIR>` | the name of a coding-agent session folder (made from the local project path) | nothing; these folders are temporary and not shipped | 87 | 37 | `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/logs/verify.log` |

## Absolute paths left in copied text files

These were left on purpose or need a look. Paths under `/scratch/` and `/home/<name>` come from
third-party model metadata (for example scGPT `args.json` and the State SE-600M `config.yaml`) and
were kept unchanged.

| Path prefix | Count | Example file |
|---|---|---|
| `/scratch/ssd004` | 14 | `projects/maxtoki/checks/deployment_facts.json` |
| `/Volumes/...` | 6 | `controlled-evaluation/studyC/results/costs.csv` |
| `/Users/runner` | 4 | `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/logs/evaluate.log` |
| `/home/aadduri` | 2 | `controlled-evaluation/studyB/packages/B2418/models/state_se600m/config.yaml` |
| `/Volumes/|` | 1 | `projects/maxtoki/paper-plos-one/supporting/build_s2.py` |
| `/Volumes/[^\s\` | 1 | `projects/maxtoki/verification/inventory.py` |
