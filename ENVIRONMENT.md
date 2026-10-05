# Environment

## Main environment

- Python 3.12.9 in a virtual environment (`projects/maxtoki/README.md`, section Environment).
- Frozen package list: `projects/maxtoki/requirements.txt` (copied below). The project README says it
  was generated on 2026-05-07, after most deployment runs (runs dated 2026-04-15 to 2026-05-07).
- Rebuild: `python3.12 -m venv .venv && .venv/bin/python -m pip install -r projects/maxtoki/requirements.txt`

## Second environment (Python 3.11, Anaconda)

Some runs used an Anaconda Python 3.11 install (`<CONDA_ROOT>/bin/python`). Its package list was **not
recorded** and cannot be recovered from the files. Evidence that it was used: compiled-bytecode caches
(excluded from this release) carry these interpreter tags:

| Area | Bytecode tags (count of files) |
|---|---|
| `controlled-evaluation/studyC/snapshot` | cpython-312: 1 |
| `projects/maxtoki/checks` | cpython-312: 7 |
| `projects/maxtoki/framework-eval` | cpython-312: 1 |
| `projects/maxtoki/paper-plos-one` | cpython-311: 2, cpython-312: 1 |
| `projects/maxtoki/runs/attention-grn-217M` | cpython-312: 3 |
| `projects/maxtoki/runs/circuit-tracing-217M` | cpython-312: 6 |
| `projects/maxtoki/runs/exhaustive-mapping-217M` | cpython-312: 5 |
| `projects/maxtoki/runs/longevity-mechinterp-217M` | cpython-312: 1 |
| `projects/maxtoki/runs/manifold-discovery-217M` | cpython-311: 6, cpython-312: 4 |
| `projects/maxtoki/runs/sae-atlas-217M` | cpython-312: 5 |
| `projects/maxtoki/runs/spectral-geometry-217M` | cpython-312: 2 |
| `projects/maxtoki/runs/topology-141-217M` | cpython-311: 4, cpython-312: 5 |
| `projects/maxtoki/setup` | cpython-311: 3, cpython-312: 5 |

## Versions recorded by the runs and checks

These rows are read from every copied `*config*.json` file that records an interpreter or library
version or a device. Files named `config.json` that describe a model record the `transformers` version
that **saved** the weights, not the one used; they are marked "model config".

"Era": `deployment` = the original runs (April-May 2026); `later` = the corrected analyses
(V2, V2B and V3 runs), the checks and the evaluation files made after the deployment.

| File | Kind | Era | python | torch | transformers | numpy | device |
|---|---|---|---|---|---|---|---|
| `controlled-evaluation/studyA/tasks/T4-contract/data/geneformer_v2_316m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `controlled-evaluation/studyA/tasks/T4-contract/data/maxtoki_217m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `controlled-evaluation/studyA/tasks/T4-paper/data/geneformer_v2_316m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `controlled-evaluation/studyA/tasks/T4-paper/data/maxtoki_217m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `controlled-evaluation/studyB/packages/B2418/models/geneformer_v2_316m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `controlled-evaluation/studyB/packages/B8833/models/geneformer_v2_316m/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `projects/maxtoki/setup/MaxToki-1B-HF/config.json (not shipped)` | model config |  |  |  | 4.53.3 |  |  |
| `projects/maxtoki/setup/MaxToki-217M-HF/config.json` | model config |  |  |  | 4.44.2 |  |  |
| `projects/maxtoki/setup/MaxToki-217M-HF/generation_config.json` | model config |  |  |  | 4.44.2 |  |  |
| `projects/maxtoki/checks/d12_deployment_facts/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/checks/d12_deployment_facts/verification/run_config.json` | run config | later | 3.12.9 |  |  |  | cpu |
| `projects/maxtoki/checks/results/number_trace_run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/checks/results/run_manifest_run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/checks/results/spec_lint_run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/checks/results/summary_run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/checks/v3_inputs/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 | 1.26.4 | cpu |
| `projects/maxtoki/checks/verification_review/run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/framework-eval/ledger_build/run_config.json` | run config | later | 3.12.9 |  |  |  |  |
| `projects/maxtoki/framework-eval/ledger_verification/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 |  |
| `projects/maxtoki/runs/attention-grn-217M/outputs/phase0/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/attention-grn-217M/outputs/phase0_adamson/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/attention-grn-217M/outputs/phase0_k562_1b/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/attention-grn-217M/outputs/phase0_rpe1/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/attention-grn-217M/outputs/phase0b/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v2_eval/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v2_eval/verification/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 |  |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v2b_attention/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v2b_attention/verification/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v2b_attention/verification2/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/attention-grn-217M/outputs/v3_attention_k562/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 | 1.26.4 | mps |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit/crispri/run_config.json` | run config | later |  |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit/lfc/run_config.json` | run config | later |  |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit/task_data/run_config.json` | run config | later |  |  |  |  | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/annotation/run_config.json` | run config | later |  |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/crispri/run_config.json` | run config | later |  |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/crispri/run_config_bands.json` | run config | later |  |  |  |  | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/lfc/run_config.json` | run config | later |  |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/spotcheck/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/verify/run_config.json` | run config | later |  |  |  |  | cpu |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_steering/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_triplets/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_triplets_followup/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_zero_edit_control/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_steering/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_triplets/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_triplets_followup/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_triplets_followup/v2_replay/run_config.json` | run config | later |  |  |  |  | cpu |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_00_reproduce.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_01_frozen_donor_bootstrap.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_03_aggregate_refit.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_04_permutation_frozen.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_04_permutation_internal.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_05_other_orderings_boot.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_05_other_orderings_fit.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_06_facts.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_07_tables.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_08_task_data.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_10_internal_lotdo.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_intervals/run_config_v2_11_interval_checks.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_verify/run_config_v2_verify_01.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2_verify/run_config_v2_verify_02.json` | run config | later |  | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2b_devorder/run_config_v2b_devorder_01_features_hvg.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2b_devorder/run_config_v2b_devorder_01_features_lookup.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2b_devorder/run_config_v2b_devorder_01_features_maxtoki.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v2b_devorder/run_config_v2b_devorder_01_features_tokenbag.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/run_config.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/run_config_features_hvg.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/run_config_features_lookup.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/run_config_features_maxtoki.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/run_config_features_tokenbag.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 |  |
| `projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/verify/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 | 1.26.4 | mps |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2_sae_site_check/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2_tf_specificity/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  |  |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2_tf_specificity/run_config_extract.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2_tf_specificity/run_config_stats.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2b_annotation_fdr/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2b_annotation_fdr_verify/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v2b_annotation_fdr_verify2/run_config.json` | run config | later | 3.12.9 |  |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/verify/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/verify/run_config_recompute.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  |  |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/run_config_extract.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/run_config_stats.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/phase0/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v2_crossmodel/run_config.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/rand0_A/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/rand0_B/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/rand0_C/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/rand1_A/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/rand2_A/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/trained_A/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/trained_B/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/extract/trained_C/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/prepare/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | cpu |
| `projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/verify/run_config.json` | run config | later | 3.12.9 | 2.11.0 | 5.5.4 |  | mps |
| `projects/maxtoki/runs/topology-141-217M/outputs/phase0/external_lung/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/topology-141-217M/outputs/phase0/lung/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/topology-141-217M/outputs/seed_stability_phase0/seed43/lung/run_config.json` | run config | deployment |  |  |  |  | mps |
| `projects/maxtoki/runs/topology-141-217M/outputs/v2_crossmodel/run_config.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 | cpu |
| `projects/maxtoki/runs/topology-141-217M/outputs/v2_crossmodel/verify/run_config.json` | run config | later | 3.12.9 | 2.11.0 |  | 1.26.4 | cpu |

Summary over run configs (value: number of files):

- **deployment** (9 config files): python not recorded; torch not recorded; transformers not recorded; numpy not recorded; device `mps` (9)
- **later** (88 config files): python `3.12.9` (78); torch `2.11.0` (62); transformers `5.5.4` (35); numpy `1.26.4` (45); device `cpu` (31), `mps` (24)

So the deployment runs record the device but no library versions. The torch and transformers
versions are recorded only by the later runs. `requirements.txt` (2026-05-07) lists the versions
installed in the main environment at the end of the deployment.

No run log prints a torch or transformers version.

## Hardware

- Devices named in run configs: `mps` (33), `cpu` (31) (`mps` is the Apple Silicon GPU); none names CUDA.
- CUDA or A100 named in any run log: False.
- Machine that holds the files: Apple M2 Pro, 32 GB (machine that holds the files on 2026-10-01; run files record only 'MacBook Pro (Apple Silicon, 32 GB)' and device 'mps'; same machine is likely but not proven).

## Atlas web app (JavaScript)

The SAE atlas web app pins its JavaScript packages in `projects/maxtoki/runs/sae-atlas-217M/atlas/package-lock.json`.
`node_modules/` is not shipped; rebuild it with `npm ci` in that folder.

## requirements.txt (verbatim)

```
anndata==0.12.10
annotated-doc==0.0.4
anyio==4.13.0
array-api-compat==1.14.0
certifi==2026.2.25
click==8.3.2
contourpy==1.3.3
cycler==0.12.1
donfig==0.8.1.post1
filelock==3.28.0
fonttools==4.62.1
fsspec==2026.3.0
google-crc32c==1.8.0
h11==0.16.0
h5py==3.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.10.2
idna==3.11
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
legacy-api-wrap==1.5
lightgbm==4.6.0
llvmlite==0.47.0
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.19.0
natsort==8.4.0
networkx==3.6.1
numba==0.65.0
numcodecs==0.16.5
numpy==1.26.4
packaging==26.1
pandas==2.3.3
patsy==1.0.2
pillow==12.2.0
plotly==6.7.0
Pygments==2.20.0
pynndescent==0.6.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.1.post1
PyYAML==6.0.3
regex==2026.4.4
rich==15.0.0
safetensors==0.7.0
scanpy==1.11.5
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
session-info2==0.4.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
statsmodels==0.14.6
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.11.0
tqdm==4.67.3
transformers==5.5.4
typer==0.24.1
typing_extensions==4.15.0
tzdata==2026.1
umap-learn==0.5.12
zarr==3.1.5
```
