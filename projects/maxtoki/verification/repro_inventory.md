# Reproducibility inventory: MaxToki paper (PLOS ONE, rejected)

Scope: `projects/maxtoki/{runs,setup,audits,summaries}`, figure build files in `paper-plos-one/build/` and `paper-biosystems/`, and the public repo `Biodyn-AI/maxtoki-mechinterp@89da276`.
Nothing in the repository was changed. All helper output is in this scratch folder.

Root used below: `M = <REPO_ROOT>/projects/maxtoki`.
"Verified" means I read the file or ran a check. "Inferred" means I reasoned from indirect evidence.

Helper files (scratch only):
- `inventory.py`, `summarize.py`, `missing_refs.py` — the scan scripts.
- `inventory_raw.json` — one record per file (2,314 files; size, type, path/secret/PII hit counts).
- `summary_out.txt` — the full per-group tables.
- `public_tree.tsv` — file list of the public repo at 89da276.
- `maxtoki_sha256.txt` — sha256 of the local MaxToki weights.

---

## 1. Headline numbers

| Item | Value |
|---|---|
| Files scanned (without `node_modules`) | 2,314 |
| `runs/` total on disk | ~10.0 GB (7.2 GB of it is `attention-grn-217M`) |
| `setup/` total | 4.7 GB (of which 4.72 GB = two HF weight files) |
| Analysis scripts (.py/.sh), all folders | 132 files, 1.4 MB (excludes the 2 built JS bundles) |
| Run logs | 116 files, 3.1 MB |
| Configs (run_config.json, planning, package files, model configs) | 40 files, 0.8 MB |
| Docs (.md/.txt/.tex) | 77 files, 0.8 MB |
| Small tables + JSON outputs (<= 50 MB) | 1,448 files, 465.5 MB |
| Small arrays (.npy/.npz <= 50 MB, not .pt) | 169 files, 410.3 MB |
| .pt weight files (all < 50 MB) | 109 files, 603.5 MB |
| Files > 50 MB | 29 files, 12.55 GB (27 in `runs/` = 7.83 GB; 2 HF weight files = 4.72 GB) |
| Files with local absolute paths (not counting `__pycache__`) | 136 files, 506 hits (54 scripts, 68 logs, 13 docs, 1 JSON) |
| Real secrets found | **0** |
| Files carrying the local user name in a path | 13 (scripts, logs, STATUS.md) |

---

## 2. What the public repo holds and why runs/ and setup/ are missing

Verified with `gh api` on `Biodyn-AI/maxtoki-mechinterp` commit `89da27615099c944078e56f4bc112d50143de7fc` ("Initial public release", 2026-05-07T22:48:16Z). It is the only commit.

- 51 files in total: `pipelines/` (11 md), `prompts/` (4 md), `projects/maxtoki/audits/` (6), `projects/maxtoki/paper/` (9, the older ICML-style draft), `projects/maxtoki/summaries/` (16), plus README, LICENSE, CITATION.cff, CONTRIBUTING.md, .gitignore.
- **Not included:** `projects/maxtoki/runs/`, `projects/maxtoki/setup/`, `projects/maxtoki/requirements.txt`, `projects/maxtoki/README.md`, `paper-plos-one/`, `paper-biosystems/` (so no figure sources and no figure build script).
- The `.gitignore` excluded runs on purpose. Exact lines:
  - `# Run artefacts (deliberately excluded from this release for size)` then `runs/`, `projects/*/runs/`, `projects/*/setup/`, `projects/*/.venv/`, `projects/*/checkpoints/`, `*.safetensors`, `*.bin`, `*.h5`, `*.hdf5`, `*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.npz`.
  - Also, under "LaTeX intermediates": `*.log` and `*.out`. **These two lines would also drop all 116 run logs** even if `runs/` were un-ignored. A new release must add `!projects/maxtoki/runs/**/*.log` or change the rule.
  - Also ignored: `repos/`, `references/*.pdf`, `review_plans/`, `.claude/`, `CLAUDE.md`, `._*`.
- The public `projects/maxtoki/audits/audit_a3_bootstrap_cis.py` (9,252 B) differs from the local one (9,317 B) in one line only: public `PROJ = Path(".")`, local `PROJ = Path("<REPO_ROOT>/projects/maxtoki")`. It reads three files under `runs/`, which are not public. Those three files do exist locally:
  - `runs/sae-atlas-217M/outputs/phase6/causal_patching.csv` (3,114 B)
  - `runs/topology-141-217M/outputs/phase78/h123_signed_motif_degree_strata.csv` (9,037 B)
  - `runs/circuit-tracing-217M/outputs/phase11_crispri_validation.json` (113 B)
  So Reviewer 2's point is correct and easy to fix: those files are small.

---

## 3. Per-pipeline inventory (runs/)

Counts are `files / size`. "Tables+JSON" are CSV/JSON outputs <= 50 MB. "Arrays" are .npy/.npz <= 50 MB. ">50 MB" is listed in section 4.

| Pipeline | Scripts | Configs (run_config.json etc.) | Logs | Docs | Tables+JSON | Arrays | .pt weights | >50 MB | Total on disk |
|---|---|---|---|---|---|---|---|---|---|
| attention-grn-217M | 11 / 136 KB | 19 / 22 KB | 19 / 1.4 MB | 7 / 32 KB | 64 / 166.8 MB | 12 / 94.4 MB | 0 | 19 / 6.94 GB | 7.2 GB |
| spectral-geometry-217M | 16 / 173 KB (9 main + 7 autoloop) | 5 | 16 / 0.9 MB | 18 / 133 KB | 54 / 0.6 MB | 6 / 0.9 MB | 0 | 4 / 321.5 MB | 324 MB |
| sae-atlas-217M | 37 / 0.4 MB (12 pipeline .py, 3 atlas .py, 22 web-app TS/JS/config) | 6 | 12 / 98 KB | 0 | 157 / 273.8 MB | 24 / 0.9 MB | 13 / 602.5 MB | 0 | 1.1 GB (+296 MB node_modules) |
| circuit-tracing-217M | 4 / 52 KB | 0 | 2 / 12 KB | 0 | 11 / 11.4 MB | 0 | 0 | 1 / 76.2 MB | 88 MB |
| exhaustive-mapping-217M | 6 / 71 KB | 0 | 4 / 14 KB | 0 | 1,021 / 5.4 MB | 1 / 14.4 MB | 0 | 0 | 20 MB |
| topology-141-217M | 20 / 224 KB | 4 | 26 / 0.4 MB | 4 / 45 KB | 72 / 0.9 MB | 20 / 95.0 MB | 0 | 0 | 97 MB |
| manifold-discovery-217M | 24 / 250 KB (23 .py + 1 .sh) | 1 (+ planning/h65_stage_dag.json, research_plan.md) | 33 / 0.2 MB | 3 / 56 KB | 62 / 6.3 MB | 105 / 190.8 MB | 96 / 1.0 MB (per-head probe .pt) | 3 / 514.0 MB | 713 MB |
| longevity-mechinterp-217M | 4 / 40 KB | 0 | 1 / 10 KB | 4 / 21 KB | 5 / 0.3 MB | 1 / 14.0 MB | 0 | 0 | 14 MB |

Script names per pipeline are in `summary_out.txt` (section "SCRIPTS per group"). Examples: attention-grn `phase0_extract.py … phase12_verdict.py`; topology `phase0_extract.py … phase16_lineage_hub_depth.py` plus 5 `audit_*.py`.

Notes by pipeline (verified unless marked):
- **attention-grn-217M.** Has a `README.md` with a run table (seed 42, N_ctrl 2,000, N_hvg 1,500, max_len 2,048, L8 primary) and a "How to run from scratch" block. The weights download uses `resolve/main` (no revision). 4 datasets: K562 217M, RPE1, Adamson, K562 1B (`*_k562_1b`). Cell manifests: `outputs/phase0*/control_cells.csv` (K562 has barcode + global index; RPE1/Adamson/1B have global index only; 1B uses 200 cells).
- **spectral-geometry-217M.** Has an `autoloop/` folder (agent loop driver `run_maxtoki_autoloop.py`, prompts, per-iteration scripts, `runtime/driver.log`). The driver calls `<CLAUDE_CLI>` with `--permission-mode bypass` (6 hits in `driver.log`). Cell manifest `outputs/phase0/cell_metadata.csv` holds only `sample_idx_local, cell_type` — it cannot be mapped back to source rows without re-running the sampler.
- **sae-atlas-217M.** 12 SAEs `outputs/phase1/layer_XX/sae_final.pt`, 48,597,489 B each, plus `outputs/phase9/sae_multitissue/sae_final.pt`. SAE training settings are recorded in `layer_XX/results.json` (d_sae 4,928, k 32, lr 3e-4, batch 4,096, 4 epochs, n_train 917,996, n_eval 102,000; seed default 42 in `setup/topk_sae.py:71`).
  - **Deleted intermediates (verified missing):** `outputs/phase0/layer_XX_activations.npy`, `layer_XX_cell_ids.npy` and `outputs/phase0/run_config.json`. `phase0_extract_positions.py:133-160` writes them, and `phase1_train_saes.py`, `phases2_to_8.py`, `remaining_phases.py`, `phase9_multitissue.py`, `phases12_13_viz.py` read them. So SAE training can only be rerun after re-extraction. Size if kept (inferred): 1,019,996 positions × 1,232 × 4 B ≈ 5.0 GB per layer, ≈ 60 GB for 12 layers. The list of chosen cells is also gone.
  - `outputs/phase0/layer_00..11/gene_names.json` are **12 identical copies** (same md5 `255861e6…`, 9,440,036 B each; 113 MB total).
  - The atlas data exist **three times**: `atlas-data/` (106 MB), `atlas/public/data/` (106 MB, same names and sizes), `atlas/dist/data/` (106 MB). `atlas/node_modules/` is 296 MB (12,997 files).
  - `full_12layer_pipeline.py:408` writes to `<TMP>/maxtoki-atlas-data-12` (gone). `circuit-tracing-217M/scripts/remaining_phases.py:78` reads `<TMP>/maxtoki-atlas-data-12/cross_layer_graph.json`. A copy exists at `atlas-data/cross_layer_graph.json`.
- **circuit-tracing-217M.** `outputs/circuit_edges.csv` is 79.9 MB. `outputs/phase11_crispri_validation.json` still holds the old buggy value `directional_accuracy 0.5461`. The corrected 53.48 % used in the paper (`paper-plos-one/main.tex:1095`, `:1549`) is in `outputs/groupkfold_crispri/summary.json`. A results map must point to the second file.
- **exhaustive-mapping-217M.** 1,020 small JSON files (per-feature and per-steering records). Steering results in `outputs/experiment3/steering_summary.json`.
- **topology-141-217M.** Logs in `logs/` (26). Cell manifests `outputs/phase0/{immune,lung,external_lung}/cell_metadata.csv` hold only local index + cell type. A seed-43 re-extraction is in `outputs/seed_stability_phase0/seed43/`.
- **manifold-discovery-217M.** Best documented: `STATUS.md`, `planning/research_plan.md`, `planning/h65_stage_dag.json`, `reports/` (hypothesis registries, quality gates). Cell manifests `outputs/phase1/cells_*_obs.csv` carry `obs_label` (source barcode) and `cell_idx`. `STATUS.md:122` says all compute used `<CONDA_ROOT>/bin/python`, not the project venv.
- **longevity-mechinterp-217M.** Only stage 1 has outputs (`outputs/stage1_20260505/`). Stage 2 and 3 outputs referenced by `run_stage2.py` / `run_stage3_sae.py` (`stage2_summary.json`, `stage3_summary.json`, `sae_artifacts.npz`, …) do not exist, which matches the paper saying this pipeline gave no result. `sampled_obs.csv` has `obs_name`. The scripts import from `repos/longevity-mechinterp/` (an external pinned repo, not in runs).

### setup/
| File | Size | Notes |
|---|---|---|
| `maxtoki_adapter.py` | 16,389 B | Model loader + attention extractor. Line 28-30: gene medians from an absolute path to Geneformer's `gene_median_dictionary_gc104M.pkl`. |
| `dataset_loader.py` | 8,354 B | Lines 18-25: absolute paths to K562, RPE1, Adamson h5ad files. |
| `topk_sae.py` | 6,592 B | TopK SAE; `seed=42` default (line 71), `torch.manual_seed(seed)` (line 79). |
| `token_dictionary.json` | 560,316 B | MaxToki Ensembl vocab. Per `runs/attention-grn-217M/README.md`, extracted from `MaxToki-217M-bionemo/context/io.json`. The BioNeMo checkpoint is not on disk under `setup/`. |
| `MaxToki-217M-HF/` | 867,797,760 B safetensors + 2 small JSON | See section 6. |
| `MaxToki-1B-HF/` | 4,196,167,128 B safetensors + 2 small JSON | See section 6. |
| `__pycache__/` | 6 files | Both cpython-311 and cpython-312 → two Python versions were used. |

### audits/
`audit-20260507.md` (36,689 B), `audit-20260507-completion.md`, `audit-20260507-completion-v2.md`, `audit_a2_groupkfold_status.md`, `audit_a3_bootstrap_cis.py`, `bootstrap_cis_summary.json`. All six are already public.

### summaries/
16 files, 2.0 MB. All public. Includes `sae-atlas-interactive.html` (1,880,435 B).

### Figure build files
- `paper-plos-one/build/make_figures.sh` (3,003 B) builds `Fig1.tif`–`Fig9.tif`. It reads `../paper-biosystems/main.tex`, cuts out each `tikzpicture`, and compiles it with `pdflatex` + `pdftoppm` at 300 dpi. Also `make_plos.py` (11,848 B), `check_plos.py` (8,039 B), `plos_latex_template.tex`.
- `paper-biosystems/docx-build/{prep.py, check.py, build.sh, elsevier-harvard.csl}`.
- **The figure data are typed by hand into TikZ `coordinates {…}` blocks in `paper-biosystems/main.tex`** (Fig 1 at line 406 is a diagram; Figs 2–9 at lines 1013, 1176, 1263, 1358, 1424, 1512, 1557, 1610). No script reads run outputs to make a figure. So "figure-generation code" exists but it is a typesetting step. A results map (section 10) is the way to link each number to its source file.
- Output TIFFs: `paper-plos-one/figures/` and a duplicate `paper-plos-one/maxtoki-plos-one-figures/` (+ a `.zip`, 551,443 B). 12 macOS `._*` files sit in `paper-plos-one/` and must not be shipped.

---

## 4. Files over 50 MB (would need Zenodo or similar)

| Size (bytes) | Path (under `M/`) |
|---|---|
| 4,196,167,128 | setup/MaxToki-1B-HF/model.safetensors (re-downloadable from HF; do not re-host) |
| 2,880,000,128 | runs/attention-grn-217M/outputs/phase0_k562_1b/attention_edges_per_head.npy |
| 867,797,760 | setup/MaxToki-217M-HF/model.safetensors (re-downloadable from HF; do not re-host) |
| 792,000,128 | runs/attention-grn-217M/outputs/phase0/attention_edges_per_head.npy |
| 792,000,128 | runs/attention-grn-217M/outputs/phase0b/vw_edges_per_head.npy |
| 792,000,128 | runs/attention-grn-217M/outputs/phase0_rpe1/attention_edges_per_head.npy |
| 792,000,128 | runs/attention-grn-217M/outputs/phase0_adamson/attention_edges_per_head.npy |
| 332,316,988 | runs/manifold-discovery-217M/outputs/phase1/cells_external.npz |
| 180,000,128 | runs/attention-grn-217M/outputs/phase0_k562_1b/attention_edges_layer_mean.npy |
| 123,015,330 | runs/manifold-discovery-217M/outputs/phase1/cells_internal.npz |
| 99,000,128 ×10 | runs/attention-grn-217M/outputs/{phase0,phase0b,phase0_rpe1,phase0_adamson}/…layer_mean.npy and phase4/edges_layer_mean_{baseline,top5_trrust,random5_A,random5_B,random5_C,entropy_matched_5}.npy |
| 90,000,128 ×2 | runs/attention-grn-217M/outputs/phase6_cssi/{per_cluster_edges,per_cluster_pair_counts}.npy |
| 88,704,128 ×3 | runs/spectral-geometry-217M/outputs/phase0/layer_gene_embeddings.npy; phase8/embeddings_seed43.npy; phase8/embeddings_seed44.npy |
| 83,586,200 | runs/manifold-discovery-217M/outputs/phase1/cells_zeroshot.npz |
| 79,873,828 | runs/circuit-tracing-217M/outputs/circuit_edges.csv |
| 70,963,328 | runs/spectral-geometry-217M/autoloop/iterations/iter_0003/h3g_cell_hidden.npy |
| 53,247,769 | runs/attention-grn-217M/outputs/phase2_rpe1/pair_dataset.csv |

Also better placed on Zenodo although each is under 50 MB: the 13 SAE weight files (`sae_final.pt`, 48.6 MB each, 632 MB total) and `manifold-discovery-217M/artifacts/` (150 MB: 88 operator .npy, 96 per-head .pt, anchors). GitHub warns at 50 MB and blocks at 100 MB per file.

---

## 5. Flags

### (a) Size — see section 4.

### (b) Absolute local paths
136 non-cache files, 506 hits. Path roots found:
- `<DATA_ROOT>` (input data; 25+ script hits) and `<DATA_ROOT>/biodyn-nmi-paper`, `…/biodyn-work/single_cell_mechinterp/…`, `…/biodyn-work/subproject_53_scgpt_gpl_replication/…`.
- `<REPO_ROOT>/projects/maxtoki` and `…/repos/longevity-mechinterp/`.
- `<HF_CACHE>/hub/models--ctheodoris--Geneformer` (4 scripts).
- `<CONDA_ROOT>/…` (logs, `STATUS.md`, `phase1bc_run_remaining_panels.sh`).
- `<CLAUDE_CLI>` (autoloop driver).
- `<TMP>/maxtoki-atlas-data-12` (2 scripts, see above).

54 scripts need their paths replaced by a config file or environment variables. Full list is printed in `summary_out.txt` and repeated here by pipeline: attention-grn 8, spectral-geometry 14 (incl. 7 autoloop), sae-atlas 10, circuit-tracing 3, exhaustive-mapping 4, topology 5, manifold-discovery 3, longevity 4, setup 2, audits 1. Also 13 docs (autoloop prompts, `STATUS.md`, `research_plan.md`, longevity `README.md`/`FINAL_SUMMARY.md`), 1 JSON (`topology-141-217M/outputs/phase0/immune/run_config.json`) and 68 logs. Logs can keep paths if we say so, or be cleaned with a simple replace.

### (c) Secrets
Searched for `hf_…`, `sk-…`, `gh[pous]_…`, `AKIA…`, private-key headers, `api_key`, `password`, `HF_TOKEN`, and `token = "…"` assignments. **No real secret found.** False positives only:
- `runs/sae-atlas-217M/atlas/dist/assets/plotly-BnPvoolX.js` — the word "api_key" in the Plotly library.
- `runs/sae-atlas-217M/atlas/dist/assets/index-C3O3WkNU.js` — the word "password" in bundled code.
- `sk-` matches in `runs/manifold-discovery-217M/scripts/phase6_task_probes.py`, `paper-biosystems/README.md`, `audits/audit-20260507.md` are English words ("task-specific", "desk-reject", "task-related").
- The word "token" appears in 142 files. All are about gene tokens / the tokenizer. No token values.
- `node_modules/` was not scanned (third-party code, should not be released anyway).

### (d) Personal information
- Local user name `ihorkendiukhov` inside paths: 13 files (`spectral-geometry-217M/scripts/{audit_a3_bootstrap_cross_model_pearson.py, phase9b_cross_model.py}`, `autoloop/run_maxtoki_autoloop.py`, `autoloop/iterations/iter_0003/h3del_lid_confound.py`, `autoloop/runtime/driver.log`, `topology-141-217M/scripts/phase14_cross_model_cca.py`, `topology-141-217M/logs/{phase12_lung,phase14,phase4_scgpt}.log`, `manifold-discovery-217M/{STATUS.md, scripts/phase1bc_run_remaining_panels.sh, outputs/phase6.log, outputs/phase12_lite.log}`).
- Author name in `runs/sae-atlas-217M/atlas/src/pages/About.tsx` (and its built bundle), `summaries/sae-atlas-interactive.html`, `summaries/topology-141-217M-FINAL_SUMMARY.md`. This is attribution, fine for a non-anonymous release.
- Emails only in paper files (author contact in `main.tex`, cover letters, title pages; template emails in `.bst/.sty`). Expected.
- No phone numbers, IP:port pairs, or donor-level human data beyond public atlas donor IDs (e.g. `TSP2`) in cell manifests.

---

## 6. Checkpoint revisions

| Model | Revision recorded in the project? | What I found |
|---|---|---|
| MaxToki-217M-HF | **No.** Download is `curl …/theodoris-lab/MaxToki/resolve/main/MaxToki-217M-HF/$f` (`runs/attention-grn-217M/README.md:57`). Files dated 2026-04-16. | Local sha256 `9da1fbf8cc489486d158d4653a14e3f44ac49c75a8e1480566eab0e9f99cf898` **equals** the HF LFS oid of the current file (verified via HF API). config.json git-blob `d55a18cd…` equals HF. |
| MaxToki-1B-HF | **No.** Files dated 2026-04-16 14:01–14:03. | Local sha256 `96c6c8f9d61732c9bedfc37a9cfd937359e6ce0bfa860526646d101815a242e6` **equals** HF LFS oid. config.json blob `7b0e9401…` equals HF. |
| Geneformer V2-316M | **Yes, in 4 scripts** as an HF cache snapshot path: `05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28` (`spectral-geometry-217M/scripts/phase9b_cross_model.py:48`, `…/audit_a3_bootstrap_cross_model_pearson.py:27`, `topology-141-217M/scripts/phase14_cross_model_cca.py:47`, `autoloop/iterations/iter_0003/h3del_lid_confound.py`). | HF says that revision is dated 2026-02-06. Its `Geneformer-V2-316M/model.safetensors` blob sha256 is `965ceccea81953d362081ef3843560a0e4fef88d396c28017881f1e94b1246f3`. Local `refs/main` has since moved to `04c2b2e8…`, so "latest" would not reproduce. |
| scGPT | **No HF revision (scGPT is not on HF).** Only a derived file is used: `<DATA_ROOT>/biodyn-work/subproject_53_scgpt_gpl_replication/embeddings/scgpt_whole_human_gene_embeddings.pt` (249,944,756 B), read by `topology-141-217M/scripts/phase4_scgpt_cross_model.py:46-49`. | It was made by `…/subproject_53…/scripts/phase0_extract_embeddings.py:19` from `…/single_cell_mechinterp/external/scGPT_checkpoints/whole-human/best_model.pt` (205,385,258 B). `args.json` there has `save_dir: …/cellxgene_census_human-May23-08-36-2023` → the official "whole-human" release. |

HF commit history of `theodoris-lab/MaxToki` (verified): `21aa7b7f844c146c16fb79ffb3aa75d9b07a3d77` (2026-04-02) was `main` on the download date. Later commits `bf6179a6…`, `9fe1a7a3…`, `581b94c2ba9c7c82761e1ec6eb6b8cf5fba0f8d7` (all 2026-04-23) only "upload config.json for tracking". Since the local files match the current HF files byte for byte, the release can state: "MaxToki weights from `theodoris-lab/MaxToki` revision `21aa7b7f…` (identical files at `581b94c2…`), sha256 as above." This is a post-hoc pin (inferred from dates + verified hashes), and should be said so.

Important mismatch: the PLOS data statement (`paper-plos-one/data-and-code-availability.txt`) says checkpoints were used "at the exact revisions pinned in the Dependencies section of each pipeline specification". I checked all `pipelines/*.md` Dependencies sections. They name `ctheodoris/Geneformer` subfolder `Geneformer-V2-316M` and "scGPT whole-human" but **pin no checkpoint revision**. The 40-character hashes in the pipeline files (`40e6da0a…`, `42ac1796…`, `32ec76d8…`, `9277f5da…`, `4f6bfb4a…`, `6406f07c…`, `5a614646…`) are commits of the source-paper code repos, not model checkpoints.

---

## 7. Datasets used and where they live

All under `B = <DATA_ROOT>`. Sizes and shapes verified with `stat` and `h5py` (metadata only).

| Dataset | Path | Size (B) | Shape (cells × genes) | Version info found | Used by |
|---|---|---|---|---|---|
| Replogle "K562" | `B/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad` | 30,010,775,846 | 643,413 × 6,546 | None in file (`uns` has only `hvg`). **The file holds 4 cell lines**: hepg2 96,616; jurkat 184,470; k562 188,590; rpe1 173,737. | attention-grn, sae-atlas, circuit-tracing, exhaustive-mapping, setup |
| Replogle RPE1 | `B/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad` | 1,236,886,900 | 247,914 × 8,749 | None in `uns`. Name and obs columns match the scPerturb format (inferred). | attention-grn (rpe1) |
| Adamson 2016 | `B/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad` | 601,696,516 | 68,603 × 4,888 | `uns` keys (`non_dropout_gene_idx`, `top_non_dropout_de_20`) match the GEARS processed format (inferred). | attention-grn (adamson) |
| Tabula Sapiens – Immune | `B/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad` | 19,773,999,714 | 592,317 × 60,606 | CELLxGENE dataset `e62b0182-368f-4875-8ac1-f49e3d5eda60`, schema 7.0.0, collection `e5f58829-1a66-40b5-a624-9046778e74f5` | spectral, topology, manifold, sae-atlas phase9, exhaustive-mapping |
| TS Immune subset | `…/data/raw/tabula_sapiens_immune_subset_20000.h5ad` | 1,721,702,998 | 20,000 × 60,606 | same source id as above | longevity-mechinterp |
| Tabula Sapiens – Lung | `…/data/raw/tabula_sapiens_lung.h5ad` | 3,196,847,019 | 65,847 × 60,606 | CELLxGENE `40f8b1a3-9f76-4ac4-8761-32078555ed4e`, schema 7.0.0 | topology, manifold |
| TS Kidney | `…/data/raw/tabula_sapiens_kidney.h5ad` | 449,970,582 | — | CELLxGENE `65ca6e36-73b0-4c88-b0f3-7b23b48844ad` | 1 script reference (sae-atlas area) |
| Krasnow lung Smart-seq2 | `…/data/raw/krasnow_lung_smartsq2.h5ad` | 186,674,175 | 9,409 × 53,514 | CELLxGENE `c88e0403-da93-40f4-99b5-f5fdeb81a82c` | topology (external_lung) — **not named in the data statement** |
| TRRUST (human) | `B/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv` | 297,659 | 9,396 lines | No version string. Format = TRRUST v2 (TF, target, mode, PMID) (inferred). | most pipelines |
| DoRothEA ChIP-seq | `B/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv` | 1,640,587 | 117,836 lines | No version string; columns `source, target, confidence`. | sae-atlas audit A8 |
| STRING pairs | `B/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json` | 91,141 | 2,747 pairs ≥700; 1,675 ≥900 | No STRING version. Pre-filtered to a small gene pool. The sibling `reference_summary.json` wrongly says 0 pairs (stale). | spectral, topology, sae-atlas |
| Reactome / KEGG / GO BP sets | same folder: `reactome_gene_sets.json` (191,888), `kegg_gene_sets.json` (40,567), `go_bp_gene_sets.json` (233,433) | — | — | No version | sae-atlas, spectral, topology |
| GO mapping | `B/biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl` | 9,462,558 | — | No version (GEARS resource, inferred) | sae-atlas, spectral |
| Gene symbol→Ensembl | `…/crispri_validation/data/gene_name_id_dict_gc104M.pkl` | 1,660,882 | — | Geneformer gc104M dictionaries | attention-grn, spectral, setup |
| Gene medians | `…/crispri_validation/data/gene_median_dictionary_gc104M.pkl` | 1,512,661 | — | Geneformer gc104M (used in place of MaxToki's own medians, which are not published) | setup/maxtoki_adapter.py |

Mismatches with the PLOS data statement:
- It says K562/RPE1 matrices came from `https://gwps.wi.mit.edu`. The local K562 file is a 4-cell-line merged file with no provenance note. Its origin is not recorded anywhere I found. (It may be a filtered HepG2/Jurkat/K562/RPE1 compilation; unverified.)
- It names only "Tabula Sapiens". The Krasnow lung Smart-seq2 atlas is also used (topology external_lung).
- No version is recorded for TRRUST, DoRothEA, STRING, Reactome, KEGG, GO.

---

## 8. Seeds, environment, hardware

**Seeds (verified by grep of scripts):** 42 is the default almost everywhere (`SEED = 42`, `default_rng(SEED)`, `random_state=SEED`). Derived seeds exist (e.g. `SEED+1000+trial`, `SEED+li+777`). Other seeds: 43 and 44 (spectral phase8 CKA; topology seed-stability audits), 137 (one exhaustive-mapping script), `seed=0` (2 spectral hits). `run_config.json` files record `"seed": 42` for attention-grn and topology phase0. 14 scripts have no seed at all; most are aggregation-only (e.g. `phase12_verdict.py`, `stage3_synthesize.py`), but `manifold-discovery-217M/scripts/phase4_export_operators.py` and `phase3a_revaluate_3gate.py`, `topology-141-217M/scripts/phase6_manifold_distances.py`, and two autoloop scripts should be checked.

**Environment:**
- `M/requirements.txt`: 71 pinned packages (torch 2.11.0, transformers 5.5.4, numpy 1.26.4, scanpy 1.11.5, anndata 0.12.10, scikit-learn 1.8.0, lightgbm 4.6.0, …). Venv is Python 3.12.9. `M/README.md:66` says it was generated 2026-05-07 — **after** most runs (runs dated 2026-04-15 to 2026-05-07). Not in the public repo.
- A second environment was used: `<CONDA_ROOT>` Python 3.11 (manifold-discovery per `STATUS.md:122`; topology logs `phase12_lung.log`, `phase14.log`, `phase4_scgpt.log`; 2 sae-atlas atlas scripts). `cpython-311` caches exist in manifold-discovery (6), topology (4), setup (3). **Its package versions are not recorded.**
- Model configs say `transformers_version` 4.44.2 (217M) and 4.53.3 (1B) — that is the version that saved the weights, not the one used.
- Web atlas: `atlas/package.json` + `package-lock.json` (228 KB) pin the JS stack.

**Hardware:** every `run_config.json` and log that records a device says `mps` (Apple Silicon): attention-grn 7 files, manifold 8, sae-atlas 8, exhaustive 4, spectral 4, circuit 1. `runs/attention-grn-217M/README.md` says "Device: MPS (Apple Silicon), float32, eager attention". **But the paper says "All runs used a single A100-80GB" (`paper-plos-one/main.tex:901`) and the data statement repeats it.** I found no local evidence of an A100 run. This must be fixed or explained.

---

## 9. Release size estimate

| Release tier | Content | Size |
|---|---|---|
| Code only | 132 scripts + setup + audits + figure build scripts | ~1.5 MB |
| Code + configs + logs + docs | adds 40 configs, 116 logs, 77 docs | ~6 MB |
| + small tables/JSON/figures (<= 50 MB each, deduplicated) | adds CSV/JSON outputs and figures; drop 2 extra atlas copies (−216.5 MB), 11 of 12 duplicate `gene_names.json` (−104 MB), node_modules (−296 MB), `__pycache__`, `._*` | ≈ 380 MB (465.5 + 11.6 MB figures − 104 MB duplicates) |
| + small arrays (<= 50 MB, not weights) | adds 169 .npy/.npz | ≈ 790 MB |
| Zenodo bundle | 27 run files > 50 MB (7.83 GB) + SAE weights (632 MB) + manifold artefacts .pt (≈1 MB) | ≈ 8.4 GB |
| Not re-hosted | MaxToki weights (4.72 GB; HF has them), all input datasets (~57 GB), SAE phase0 activations (deleted; ≈60 GB inferred) | — |

A GitHub release of "code + configs + logs + docs + small tables" at ≈ 380 MB is too big for comfort. Suggested split: GitHub gets code, configs, logs, docs, manifests and the small tables that numbers in the paper come from (a few MB; e.g. `run_report*.md`, `phase12_verdict*.json`, `reports/*.json`, `summary.json` files). Everything else goes to Zenodo as one versioned archive, with a checksum file.

The largest single items are the `attention_edges_per_head.npy` files (792 MB–2.88 GB). They are derived from the model and data and can be rebuilt by `phase0_*.py` (~55 min per dataset on MPS per `run_config.json`: `total_phase_seconds 3286`). Option: ship only the layer-mean files and say how to rebuild the per-head ones.

---

## 10. Proposed release layout

```
maxtoki-mechinterp/                      (GitHub, new tag, e.g. v1.1)
├── pipelines/  prompts/                 (as now)
├── projects/maxtoki/
│   ├── README.md                        (run order, hardware actually used, time per run)
│   ├── environment/
│   │   ├── requirements-venv-py312.txt  (= current requirements.txt)
│   │   ├── requirements-conda-py311.txt (must be regenerated; missing today)
│   │   └── atlas-web/package-lock.json
│   ├── config/
│   │   ├── paths.example.yaml           (DATA_ROOT, MODEL_ROOT, HF_CACHE, OUT_ROOT)
│   │   ├── checkpoints.yaml             (repo id, revision, file, sha256 for each model)
│   │   └── datasets.yaml                (name, source URL/DOI, CELLxGENE id, local file, bytes, sha256, n_obs, n_var)
│   ├── setup/                           (maxtoki_adapter.py, dataset_loader.py, topk_sae.py, token_dictionary.json, download_weights.sh)
│   ├── runs/<pipeline>-217M/
│   │   ├── README.md                    (phases, inputs, outputs, seeds, device)
│   │   ├── scripts/                     (paths read from config/paths.yaml)
│   │   ├── logs/                        (all *.log; path-scrubbed)
│   │   ├── manifests/                   (cell lists with source barcodes; gene lists)
│   │   └── results/                     (small result tables that feed the paper)
│   ├── audits/  summaries/
│   ├── paper/                           (PLOS source, figure TikZ, make_figures.sh)
│   └── RESULTS_MAP.csv                  (section 11)
└── ZENODO.md                            (DOI, file list, sha256, how to place files under runs/)

Zenodo record (one DOI, versioned)
├── runs/<pipeline>-217M/outputs/…       (arrays and CSVs > a few MB, same relative paths)
├── sae_weights/layer_00..11/sae_final.pt, sae_multitissue/sae_final.pt
├── manifold_artifacts/                  (operators, heads, anchors)
└── SHA256SUMS
```

Fixes needed before release (all found above):
1. Replace absolute paths in 54 scripts with a config lookup; fix the two `<TMP>/maxtoki-atlas-data-12` paths.
2. Change `.gitignore` so run logs (`*.log`) and `runs/` text files are kept.
3. Write `checkpoints.yaml` with the MaxToki post-hoc pin + sha256, Geneformer `05fcbeb8…` + blob sha256, scGPT whole-human file size (and sha256, not computed here).
4. Record dataset provenance (above table); find the source of `replogle_concat.h5ad`.
5. Regenerate cell manifests with source barcodes for spectral-geometry and topology (they now hold only local indices), and for sae-atlas (the `cell_ids.npy` files were deleted). All three can be regenerated from the scripts with seed 42 without a forward pass, because the sampling happens before the model call (inferred from `phase0_extract_positions.py:58`, not tested).
6. Correct the hardware statement (MPS vs A100) and the "revisions pinned in Dependencies" statement.
7. Export the conda env package list, or state that it is lost.
8. Drop `node_modules/`, `atlas/dist/`, `atlas/public/data/` duplicates, 11 duplicate `gene_names.json`, `__pycache__`, `._*`.

---

## 11. Results-to-source map format

One row per number (or figure series) in the paper. CSV so reviewers can open it; one file `RESULTS_MAP.csv`.

Columns:
`result_id, paper_location, quantity, value_in_paper, pipeline, source_file, source_field, script, script_line, inputs, seed, device, checkpoint, status`

- `paper_location` = section/figure/table + tex line (e.g. `Fig2 / main.tex:1013`).
- `source_file` = path relative to repo root; `source_field` = JSON key or CSV column/row filter.
- `inputs` = ids from `datasets.yaml` / `checkpoints.yaml`.
- `status` = `verified` (value matches file), `derived` (computed by hand from file), `stale-file` (file holds an older value), `missing`.

Example rows. The file matches below are from a value search I ran; they are **candidate** sources, and each must be checked line by line before release.

| result_id | paper_location | value | source_file (candidate) | status |
|---|---|---|---|---|
| F2-k562a | Fig 2 (biosystems main.tex:1013) | attention 0.512 vs gene-variance 0.596 (K562, 217M) | `runs/attention-grn-217M/outputs/run_report.md` Phase 1 block (`gene_variance 0.5957`, `attention_primary 0.5119`); also `FINAL_SUMMARY.md:22,31` | verified (rounded) |
| F2-k562b | Fig 2 | attention 0.540 vs gene-variance 0.596 (K562, 1B) | `runs/attention-grn-217M/outputs/run_report_k562_1b.md:19,22` (`0.5961`, `0.5398`) | verified (rounded) |
| F2-rpe1 / adamson | Fig 2 | 0.603/0.766; 0.508/0.712 | `run_report_rpe1.md`, `run_report_adamson.md`, `phase12_verdict_{rpe1,adamson}.json`; `cross_dataset_report.md:23` has 0.5119 / 0.6033 | values found; bar order to confirm |
| F3 | Fig 3 (line 1176) | trust 0.811/0.896/0.827/0.800; branch 0.370/0.346/0.317/−0.147 | `runs/manifold-discovery-217M/reports/{quality_gates_let_anchor,external_validation_external,zeroshot_*,external_validation_lung_control}.json` | most values found; −0.147 not found as a literal |
| F4 | Fig 4 (line 1263) | p_null 0.003…; 0.602/0.252/0.096/0.030 | `runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun/specificity_with_null.csv` | 0.030 not found as a literal (likely rounding) |
| F5 | Fig 5 (line 1358) | steering +0.0962 (L0) … −0.0079 (L6) | `runs/exhaustive-mapping-217M/outputs/experiment3/steering_summary.json`, `stage3_summary.json` | verified values present |
| F6 | Fig 6 (line 1424) | CCA 0.78 vs 0.40; pair corr 0.382; top-1 0.425 | `runs/topology-141-217M/outputs/phase14_cross_model/phase14_summary.json`; `runs/spectral-geometry-217M/outputs/phase9b/cross_model_cosine.json` | candidate |
| F7 | Fig 7 (line 1512) | effective rank 560.81 … 93.64 | `runs/spectral-geometry-217M/outputs/phase1/per_layer_metrics.csv` column `effective_rank` | verified (560.809, 332.588, …) |
| F8 | Fig 8 (line 1557) | CKA 1.000 … 0.9914; scGPT ref 0.979/0.779 | `runs/spectral-geometry-217M/outputs/phase8/cka_per_layer.csv` (`cka_mean`); scGPT values are constants in `scripts/phase8_stability.py:241-242` | verified |
| F9 | Fig 9 (line 1610) | 61/68/75, 29/26/22, 10/6/3 | `audits/audit-20260507*.md` | not found as literals → derived by hand; needs a count script |
| T-crispri | main.tex:1095, 1549 | 53.48 % | `runs/circuit-tracing-217M/outputs/groupkfold_crispri/summary.json` | verified; note `phase11_crispri_validation.json` still says 0.5461 (stale-file) |

The same map can be emitted as YAML if a machine check is wanted: a small script can open each `source_file`, read `source_field`, round, and compare with `value_in_paper`.

---

## 12. What was not done

- I did not compute sha256 for the input datasets (57 GB total; too slow for this task) or for the scGPT checkpoint and derived embedding file.
- I did not check every number in the paper. Only the 9 figures were spot-checked, by value search.
- I did not scan `node_modules/` or the `.venv`.
- I did not test whether any script actually runs.
- The origin of `replogle_concat.h5ad` is not resolved.
- The conda (Python 3.11) package list could not be recovered from files; the conda env itself was not inspected.

---

## Plain-words summary

The scripts, configs, logs and most small results do exist on this machine. They were left out of the public repo on purpose: the `.gitignore` drops `runs/` and `setup/`, and its LaTeX rule `*.log` would also drop every run log. The code is small (about 1.5 MB). With logs, docs and the result tables that the paper's numbers come from, a GitHub release can stay at a few MB. About 8.4 GB of large arrays and SAE weights should go to Zenodo. No secrets were found. 54 scripts hard-code paths on this Mac, and 13 files show the local user name. The MaxToki weights on disk match Hugging Face byte for byte, but no revision was ever written down; Geneformer's revision is written in 4 scripts. Seeds are mostly 42 and are in the scripts. Three claims in the data statement are wrong today: runs were on Apple MPS, not an A100; the pipeline files pin no model revisions; and the logs are not in the public repo. Some intermediates are gone: the SAE training activations, and the SAE cell list. The figures are hand-typed numbers in TikZ, so a results-to-source map is needed to link each number to its file.

Report path: <AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/repro_inventory.md
