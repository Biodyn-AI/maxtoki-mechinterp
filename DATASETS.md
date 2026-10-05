# Datasets

The input datasets are **not** in this release. This file lists every input file the runs read,
where to get it, and its sha256. Sizes, shapes, hashes and the `referenced_by` lists come from
`projects/maxtoki/checks/deployment_facts.json` (section `datasets`). Local paths are shown with
the placeholders from PATH_MAP.md. "Inferred" means it was reasoned from the file, not recorded.

## Replogle et al. 2022 genome-wide Perturb-seq, filtered 4-cell-line file (K562 cells used)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad`
- Size: 30,010,775,846 bytes; sha256 `a7992a6109d589f2d49c2eaccff45a9f860418af23125d41b80d8e3929a33d56`
- Shape: 643,413 cells x 6,546 genes
- Cells per cell line: hepg2 96,616, jurkat 184,470, k562 188,590, rpe1 173,737
- Public source: Hugging Face dataset `arcinstitute/State-Replogle-Filtered`, file `replogle_concat.h5ad` (sha256 identical per deployment facts). Original data: Replogle et al., Cell 2022, doi:10.1016/j.cell.2022.05.013; portal https://gwps.wi.mit.edu ; processed data figshare+ doi:10.25452/figshare.plus.20029387; raw reads SRA BioProject PRJNA831566.
- Version: HF dataset main (commits dated 2025-12-06 per deployment facts)
- Read by: attention-grn-217M: `phase0_extract.py`, `phase0b_value_weighted.py`, `phase4_causal_ablation.py`; sae-atlas-217M: `full_12layer_pipeline.py`, `phase0_extract_positions.py`, `phase9_multitissue.py`, `remaining_phases.py`; circuit-tracing-217M: `remaining_phases.py`; setup: `dataset_loader.py`

## Replogle et al. 2022 RPE1 essential-scale Perturb-seq (scPerturb release)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad`
- Size: 1,236,886,900 bytes; sha256 `12d6a0cf9378c4f09411e27f0ce07b59f1d97731bd576d6fadf112f6f088d1a4`
- Shape: 247,914 cells x 8,749 genes
- Cells per cell line: RPE1 247,914
- Public source: scPerturb Zenodo records 7041849 / 7278143 / 7416068 / 10044268 / 13350497 (md5 identical per deployment facts), e.g. https://zenodo.org/records/13350497 . Original data as for K562 (BioProject PRJNA831566).
- Version: scPerturb file versions 1.0-1.4 carry identical bytes
- md5: `cc7f1ec50aeb3a3e1b4a6cfa713d80fa`
- Read by: attention-grn-217M: `cross_dataset_comparison.py`, `phase0_rpe1.py`; setup: `dataset_loader.py`

## Adamson et al. 2016 Perturb-seq (unfolded protein response screen), processed

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad`
- Size: 601,696,516 bytes; sha256 `3b816440acd5f02e55e63df83e697d9e80f3787f5e9056bf965c66acad5d7fc2`
- Shape: 68,603 cells x 4,888 genes
- Public source: Original data GEO GSE90546 (Adamson et al., Cell 2016). The local file matches the GEARS processed format (inferred from its `uns` keys and `GENE+ctrl` labels); byte identity with a public copy was not checkable.
- Version: not recorded
- Read by: setup: `dataset_loader.py`
- Note on modality: the project calls this dataset "CRISPRa" (in setup/dataset_loader.py:96 docstring; summaries/attention-grn-217M-FINAL_SUMMARY.md run table). The deployment check found: knock-down (CRISPRi-like): the perturbed gene falls in its own cells. Adamson et al. 2016 is a CRISPRi screen, so "CRISPRa" is a wrong label.

## Tabula Sapiens - Immune

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad`
- Size: 19,773,999,714 bytes; sha256 `14d41c8e7248319db031940da7347725a9d8b53af539c3392be69dbb66170615`
- Shape: 592,317 cells x 60,606 genes
- CELLxGENE dataset version id: `e62b0182-368f-4875-8ac1-f49e3d5eda60`; collection id `e5f58829-1a66-40b5-a624-9046778e74f5`; schema 7.0.0
- Download: https://datasets.cellxgene.cziscience.com/e62b0182-368f-4875-8ac1-f49e3d5eda60.h5ad
- Publication: https://doi.org/10.1126/science.abl4896
- Read by: spectral-geometry-217M: `h3g_centroid_trajectory.py`, `phase0_extract.py`, `phase8_stability.py`, `remaining_analyses.py`; topology-141-217M: `phase0_extract.py`; sae-atlas-217M: `phase9_multitissue.py`, `remaining_phases.py`; exhaustive-mapping-217M: `experiment3_rerun.py`, `experiments_2_3.py`; manifold-discovery-217M: `phase1a_subsample_and_tokenize.py`

## Tabula Sapiens - Immune, 20,000-cell subset made locally

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad`
- Size: 1,721,702,998 bytes; sha256 `25845071b19bb381fda15f2daf08fded6514f27366fa6f251260fb6c068f8f1a`
- Shape: 20,000 cells x 60,606 genes
- CELLxGENE dataset version id: `e62b0182-368f-4875-8ac1-f49e3d5eda60`; collection id `e5f58829-1a66-40b5-a624-9046778e74f5`; schema 7.0.0
- Download: https://datasets.cellxgene.cziscience.com/e62b0182-368f-4875-8ac1-f49e3d5eda60.h5ad
- Publication: https://doi.org/10.1126/science.abl4896
- Source: a 20,000-cell subset, made locally, of the CELLxGENE dataset named here
- Read by: sae-atlas-217M: `extract_and_enrich_missing_layers.py`; longevity-mechinterp-217M: `maxtoki_runtime.py`, `run_stage1.py`, `run_stage2.py`

## Tabula Sapiens - Lung

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad`
- Size: 3,196,847,019 bytes; sha256 `213eb5b8539af9b508f41078ce40d921c87f2b438cfb1bdf4d4af391068a2595`
- Shape: 65,847 cells x 60,606 genes
- CELLxGENE dataset version id: `40f8b1a3-9f76-4ac4-8761-32078555ed4e`; collection id `e5f58829-1a66-40b5-a624-9046778e74f5`; schema 7.0.0
- Download: https://datasets.cellxgene.cziscience.com/40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad
- Publication: https://doi.org/10.1126/science.abl4896
- Read by: topology-141-217M: `phase0_extract.py`; sae-atlas-217M: `extract_and_enrich_missing_layers.py`; manifold-discovery-217M: `phase1a_lung_nonhema_panel.py`, `phase1a_subsample_and_tokenize.py`

## Tabula Sapiens - Kidney

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad`
- Size: 449,970,582 bytes; sha256 `5653133d835f3f3799e2813bc127187f67ec1d0fb9286eacb72303e61454fc0e`
- Shape: 11,376 cells x 60,606 genes
- CELLxGENE dataset version id: `65ca6e36-73b0-4c88-b0f3-7b23b48844ad`; collection id `e5f58829-1a66-40b5-a624-9046778e74f5`; schema 7.0.0
- Download: https://datasets.cellxgene.cziscience.com/65ca6e36-73b0-4c88-b0f3-7b23b48844ad.h5ad
- Publication: https://doi.org/10.1126/science.abl4896
- Read by: sae-atlas-217M: `extract_and_enrich_missing_layers.py`

## Krasnow Lab Human Lung Cell Atlas, Smart-seq2

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad`
- Size: 186,674,175 bytes; sha256 `2d3097dbc196849afa0cc229db516f439e25389ffa118d941e6c0518b08874a5`
- Shape: 9,409 cells x 53,514 genes
- CELLxGENE dataset version id: `c88e0403-da93-40f4-99b5-f5fdeb81a82c`; collection id `5d445965-6f1a-4b68-ba3a-b8f765155d3a`; schema 7.0.0
- Download: https://datasets.cellxgene.cziscience.com/c88e0403-da93-40f4-99b5-f5fdeb81a82c.h5ad
- Publication: https://doi.org/10.1038/s41586-020-2922-4
- Read by: topology-141-217M: `phase0_extract.py`

## TRRUST human TF-target table

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv`
- Size: 297,659 bytes; sha256 `9b909319ccc8e36588b5a1bd3640e0dfba03936c6fc154cc4304d742bec11bed`
- Public source: TRRUST v2, https://www.grnpedia.org/trrust/ (Han et al., Nucleic Acids Res 2018, 46:D380). Columns TF, target, mode, PMIDs, no header.
- Version: v2 (inferred from the file format; no version string in the file)
- Read by: spectral-geometry-217M: `h02_lid_per_gene.py`, `phase0_extract.py`, `phases_345.py`, `remaining_analyses.py`; attention-grn-217M: `phase0_any.py`, `phase0_extract.py`, `phase0_rpe1.py`, `phase0b_value_weighted.py`, `phase3_residualization.py`, `phase4_causal_ablation.py`, `phase6_cssi.py`; topology-141-217M: `phase0_extract.py`, `phase123_splits_nulls.py`; sae-atlas-217M: `audit_a4_phase8t_chance_baseline.py`, `audit_a8_positive_tf_case_study.py`, `audit_a8_rerun_phase8t_targeted.py`, `full_12layer_pipeline.py`, `phase9_multitissue.py`, `phases2_to_8.py`, `remaining_phases.py`; circuit-tracing-217M: `remaining_phases.py`

## DoRothEA human regulons, ChIP-seq-supported subset (columns source, target, confidence)

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv`
- Size: 1,640,587 bytes; sha256 `c175f174d52540620a70f8de1e2cd56152756b33c164dd045974f9db8a248321`
- Public source: DoRothEA (Garcia-Alonso et al., Genome Res 2019) as served by OmniPath; local copy from an earlier project folder.
- Version: not recorded
- Read by: sae-atlas-217M: `audit_a8_positive_tf_case_study.py`, `audit_a8_rerun_phase8t_targeted.py`

## STRING protein-protein pairs, pre-filtered to a small gene pool (score >= 700 and >= 900 lists)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json`
- Size: 91,141 bytes; sha256 `20cb0e636db323c88489cb5a21bbf958b8c4a93acccfa6e3126dbf3bb8cb5ea6`
- Public source: STRING, https://string-db.org ; filtered locally in an earlier project.
- Version: STRING version not recorded
- Read by: spectral-geometry-217M: `h02_lid_per_gene.py`, `h03_spectral_gap.py`, `h3bc_string_persistence.py`, `h3del_lid_confound.py`, `phase0_extract.py`, `phase9b_cross_model.py`, `phases_345.py`, `remaining_analyses.py`; topology-141-217M: `phase0_extract.py`, `phase123_splits_nulls.py`, `phase8ext_h124_h138_chain.py`; sae-atlas-217M: `phases2_to_8.py`

## Reactome gene sets (derived JSON)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/reactome_gene_sets.json`
- Size: 191,888 bytes; sha256 `ce41f0f52b0390b4133e8d0d2a50c179aa00fc66f32639123091ef5dd3879e11`
- Public source: Enrichr library Reactome_2022 (https://maayanlab.cloud/Enrichr), cut to the highly variable gene list of an earlier project (sets of 5-500 genes) by a script that is not in this release; the sha256 identifies the exact file
- Version: Reactome_2022 (download date not recorded)
- Read by: sae-atlas-217M: `full_12layer_pipeline.py`, `phases2_to_8.py`

## KEGG gene sets (derived JSON)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/kegg_gene_sets.json`
- Size: 40,567 bytes; sha256 `6deba1a07e93b17e0884f49c730c69620e558b08b6588e99cd73c21db7bee677`
- Public source: Enrichr library KEGG_2021_Human (https://maayanlab.cloud/Enrichr), cut the same way; the sha256 identifies the exact file
- Version: KEGG_2021_Human (download date not recorded)
- Read by: sae-atlas-217M: `full_12layer_pipeline.py`, `phases2_to_8.py`

## GO biological-process gene sets (derived JSON)

- Local file: `<DATA_ROOT>/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json`
- Size: 233,433 bytes; sha256 `36ebdf5c7c7f05237acc37b6007ab3da1a71d5862aadc170a6fe8110397aaef6`
- Public source: Enrichr library GO_Biological_Process_2023 (https://maayanlab.cloud/Enrichr), cut the same way; the sha256 identifies the exact file
- Version: GO_Biological_Process_2023 (download date not recorded)
- Read by: spectral-geometry-217M: `phase9_report.py`; topology-141-217M: `phase8ext_h124_h138_chain.py`; sae-atlas-217M: `full_12layer_pipeline.py`, `phases2_to_8.py`, `remaining_phases.py`

## gene-to-GO mapping

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl`
- Size: 9,462,558 bytes; sha256 `f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771`
- Public source: GEARS resource file (inferred)
- Version: not recorded
- Read by: spectral-geometry-217M: `phases_345.py`, `remaining_analyses.py`; sae-atlas-217M: `phases2_to_8.py`, `remaining_phases.py`

## Inputs of the corrected analyses that the deployment did not use

### DoRothEA human regulons, all confidence levels (columns source, target, confidence)

- Local file: `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv`
- Size: 3,835,515 bytes; sha256 `b3972bcfe290abb7df96648d6df0ca6dee85e8bbc365e3e93f03ceeae3f20b89`
- Public source: DoRothEA (Garcia-Alonso et al., Genome Res 2019); local copy from an earlier project folder; how it was downloaded was not recorded.
- Version: not recorded
- Read by: `projects/maxtoki/runs/sae-atlas-217M/scripts/` v2_ and v3_tf_specificity_extract.py and _taskdata.py; `projects/maxtoki/verification/gata1/` inv1.py and inv3.py

### scGPT gene embedding table, extracted in an earlier project from the scGPT whole-human checkpoint

- Local file: `<DATA_ROOT>/biodyn-work/subproject_53_scgpt_gpl_replication/embeddings/scgpt_gene_embeddings.npz`
- Size: 131,585,118 bytes; sha256 `4882d38227aea4edec69f10fd486e2cd1053a42e2000709aab66ad5d5b055cf9`
- Public source: extracted from the scGPT whole-human checkpoint listed in CHECKPOINTS.md; the extraction script is not part of this release.
- Version: not recorded
- Read by: `projects/maxtoki/runs/topology-141-217M/scripts/` v2_crossmodel_prepare.py and v2_crossmodel_common.py

## Perturbation counts used by attention-grn

- `replogle_concat_k562`: {"n_cells": 188590, "n_gene_labels": 1384, "n_perturbed_genes": 1383, "n_nontargeting_cells": 10691}
- `replogle_rpe1`: {"n_cells": 247914, "n_perturbation_labels": 2394, "control_like_labels": ["control"]}
- `attention_grn_k562_217M_perturbations_evaluated`: 174
- `attention_grn_rpe1_perturbations_evaluated`: 325
- `attention_grn_adamson_perturbations_evaluated`: 57
- `attention_grn_k562_1b_perturbations_evaluated`: 155

## Data shipped inside the controlled-evaluation packages

The study packages under `controlled-evaluation/` carry small subsets or copies of public data and of run outputs.
Each package's own `README.md` says where its files came from. All of them, with size and sha256,
are listed in `DATASETS_EVAL_INPUTS.csv` (174 files). Files over the size limit are
listed in ZENODO_MANIFEST.csv instead of being copied.
