"""Build INPUT_ENCODING_AUDIT.csv from the run inventory below + order_metrics_summary.csv."""
import sys, csv
import pandas as pd

OUTDIR = sys.argv[1]       # scratch out dir with order_metrics_summary.csv
DEST = sys.argv[2]         # destination csv path

S = pd.read_csv(OUTDIR + "/order_metrics_summary.csv")

REF_PRIMARY = {
    "K562_ctrl_replogle_concat": "expm1(X)=CP10k",
    "K562_perturbed_replogle_concat": "expm1(X)=CP10k",
    "Adamson_ctrl": "expm1(X)=CP10k",
    "RPE1_ctrl": "X=raw counts",
    "TS_immune_L2048": "raw/X counts",
    "TS_immune_L4096_manifold": "raw/X counts",
    "TS_lung_L4096_manifold": "raw/X counts",
    "TS_lung_L2048_topology": "raw/X counts",
    "TS_immune_sub20k_L1024_longevity": "raw/X counts",
    "Krasnow_lung_SS2_L2048_topology": "raw/X counts",
}
ALT = {"TS_immune_L2048": "expm1(X)=CP10k decontX", "TS_immune_L4096_manifold": "expm1(X)=CP10k decontX",
       "TS_lung_L4096_manifold": "expm1(X)=CP10k decontX", "TS_lung_L2048_topology": "expm1(X)=CP10k decontX",
       "TS_immune_sub20k_L1024_longevity": "expm1(X)=CP10k decontX", "Krasnow_lung_SS2_L2048_topology": "expm1(X)=CPM"}


def nums(dset, which="primary5"):
    if dset is None:
        return {}
    g = S[(S.dataset == dset) & (S.reference == REF_PRIMARY[dset]) & (S.cells == which)].iloc[0]
    out = {"order_check_dataset": dset, "reference_order": REF_PRIMARY[dset], "n_cells": int(g.n_cells),
           "spearman_full_mean": round(g.spearman_full_mean, 3), "spearman_full_min": round(g.spearman_full_min, 3),
           "spearman_as_fed_mean": round(g.spearman_fed_mean, 3),
           "top200_overlap_mean": round(g.top200_overlap_mean, 3), "top200_overlap_min": round(g.top200_overlap_min, 3),
           "top2046_overlap_mean": round(g.top2046_overlap_mean, 3),
           "kept_set_overlap_at_max_len_mean": round(g.kept_set_overlap_mean, 3),
           "same_position_share_mean": round(g.same_position_share_mean, 3),
           "same_position_share_min": round(g.same_position_share_min, 3)}
    if dset in ALT:
        a = S[(S.dataset == dset) & (S.reference == ALT[dset]) & (S.cells == which)].iloc[0]
        out["spearman_full_vs_expm1X_mean"] = round(a.spearman_full_mean, 3)
        out["top200_vs_expm1X_mean"] = round(a.top200_overlap_mean, 3)
    a50 = S[(S.dataset == dset) & (S.reference == REF_PRIMARY[dset]) & (S.cells == "all")].iloc[0]
    out["n_cells_stability_set"] = int(a50.n_cells)
    out["spearman_full_mean_stability_set"] = round(a50.spearman_full_mean, 3)
    out["top200_overlap_mean_stability_set"] = round(a50.top200_overlap_mean, 3)
    return out


B = "<DATA_ROOT>/"
F_K562 = B + "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
F_RPE1 = B + "biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad"
F_ADAM = B + "biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"
F_TSI = B + "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"
F_TSL = B + "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"
F_TSS = B + "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
F_TSK = B + "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"
F_KRA = B + "biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad"
XK = "X = log1p(CP10k) float32, dense, 6,546 genes; sum(expm1(X)) per cell 8.3k-9.4k; no raw counts in file"
XA = "X = log1p(CP10k-like) float32 CSR, 4,888 genes; sum(expm1(X)) per cell about 5.2k; no raw counts in file"
XR = "X = raw integer UMI counts (dense, 8,749 genes)"
XT = "X = log1p(CP10k of decontX counts); raw/X = raw integer counts; layers/decontXcounts = integer counts"
XKR = "X = log1p(CPM) (sum expm1 about 1e6); raw/X = integer Smart-seq2 read counts"
T_X = "none: X passed straight to tokenize_cell, so rank = log1p(CP10k)/median"
T_XK = "none: X passed straight to tokenize_cell, so rank = log1p(CPM)/median"
T_OK = "none needed: raw counts / median"

rows = [
    # run, phase, script, data, xcont, transform, max_len, correct, dset, depends, other, rerun, cost
    ("attention-grn-217M", "Phase 0 attention, 217M K562", "scripts/phase0_extract.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "All 217M K562 attention results: per-perturbation AUROC 0.512, TRRUST 0.483, incremental, curveball null, residualisation, layer sweep, C1-C4 verdict; v2_01..v2_09 numbers; v2b order-share/direction scores",
     "HVG choice and gene features (mean, variance) use log1p(X/sum*1e4) on X that is already log1p (double log); v2 DE labels |delta|>=0.5 on double-log values",
     "yes", "MPS about 55 min (2,000 cells, 1.64 s/cell with attentions) + CPU phases 1-3,12 and v2/v2b about 1-2 h"),
    ("attention-grn-217M", "Phase 0b value-weighted, 217M K562", "scripts/phase0b_value_weighted.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "VW vs raw attention (0.513 vs 0.532), VW layer peak L9-L10", "perturbation features double log", "yes", "MPS about 18 min (1,063 s forward)"),
    ("attention-grn-217M", "Phase 4 head ablation, 217M K562", "scripts/phase4_causal_ablation.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "Phase 4 causal head-ablation results", "", "yes", "MPS about 45 min (12 sweeps x 300 cells, 3.7 min each)"),
    ("attention-grn-217M", "Phase 6 CSSI, 217M K562", "scripts/phase6_cssi.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "CSSI-max/mean/SCENIC AUROCs (0.500/0.504/0.512)", "clustering on double-log HVG expression", "yes", "MPS about 37 min (2,000 cells, 36.5 min logged)"),
    ("attention-grn-217M", "Phase 0 attention, 217M Adamson", "scripts/phase0_any.py (DATASET=adamson)", F_ADAM, XA, T_X, 2048, "NO", "Adamson_ctrl",
     "All Adamson results (AUROC 0.508, TRRUST 0.520, C3 'pass', residualised 0.534)", "gene features and labels double log", "yes", "MPS about 27 min + CPU phases"),
    ("attention-grn-217M", "Phase 0 attention, 1B K562", "scripts/phase0_any.py (MODEL_DIR=MaxToki-1B-HF, N=200)", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "All 1B K562 results (AUROC 0.540, TRRUST 0.555, incremental +0.0021, residual 0.510)", "gene features and labels double log", "yes", "MPS about 95 min (200 cells, 28 s/cell)"),
    ("attention-grn-217M", "Phase 0 attention, 217M RPE1", "scripts/phase0_rpe1.py", F_RPE1, XR, T_OK, 2048, "YES", "RPE1_ctrl",
     "All RPE1 results (AUROC 0.603, TRRUST 0.605)", "none (single log1p of counts for features is correct)", "no", "0"),
    ("attention-grn-217M", "v2 / v2verify re-analyses", "scripts/v2_01..v2_11, v2verify_01..10", "same as Phase 0 runs", "-", "no model; reuse stored attention", "-", "inherits", None,
     "Inherit the Phase 0 encoding of each run (3 of 4 runs affected)", "v2verify_06/09 already measured the input-scale problem (V3)", "yes (CPU)", "CPU only, after re-extraction"),
    ("attention-grn-217M", "v2b direction/rank baselines", "scripts/v2b_attention.py, v2b_attention_verify.py", "same as Phase 0 runs", "-", "rebuilds token order with tokenize_cell(X): same wrong order", "2048", "NO (3 of 4 runs)", "K562_ctrl_replogle_concat",
     "order share F, rank-conditioned edge, sym scores for 217M K562, Adamson, 1B K562", "", "yes (CPU)", "CPU only, after re-extraction"),
    ("sae-atlas-217M", "SAE training activations (12 layers)", "scripts/full_12layer_pipeline.py (earlier 3-layer: phase0_extract_positions.py)", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "Every SAE (outputs/phase1/layer_00..11) and so every SAE-based result: annotation rates, superposition, cross-layer, modules, Phase 6 patching 1.01x, Phase 8/8t TF specificity, Phase 9 multi-tissue, Phase 11 cell type, atlas site; all circuit-tracing, triplet, exhaustive and steering results",
     "", "yes", "MPS about 1.9 h (12 layers x about 9.5 min: 500-cell extraction + SAE training + annotation)"),
    ("sae-atlas-217M", "Phases 6, 8t, 9, 11 (model passes)", "scripts/remaining_phases.py, phase6_patching.py, phase9_multitissue.py, audit_a8_rerun_phase8t_targeted.py", F_K562 + " ; " + F_TSI, XK + " | " + XT, T_X, 2048, "NO", "K562_perturbed_replogle_concat",
     "Patching specificity 1.01x; Phase 8t TF specificity 0/48; GATA1 case study; multi-tissue rescue; cell-type enrichment (TS immune cells also mis-encoded)",
     "Phase 8t/11 perturbation contrasts use double-log expression", "yes", "MPS about 1.5-2.5 h (estimate from cell counts x 0.3-0.6 s per forward)"),
    ("sae-atlas-217M", "v2 TF specificity", "scripts/v2_tf_specificity_extract.py (+ stats, calib, taskdata)", F_K562, XK, T_X, 2048, "NO", "K562_perturbed_replogle_concat",
     "V2_TF_SPECIFICITY_REPORT: GATA1 positive fails, 0 TFs pass BH, fake-TF null floor", "knockdown check uses expm1(X) correctly", "yes", "MPS about 52 min (3,118 s over 7 chunks) + CPU about 15 min"),
    ("sae-atlas-217M", "v2 SAE site check", "scripts/v2_sae_site_check.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "SAE layer convention (VE per site); conclusion about hook sites should not depend on encoding", "", "optional", "MPS < 5 min"),
    ("sae-atlas-217M", "v2b annotation FDR", "scripts/v2b_annotation_fdr.py", "feature_catalog.json (from SAEs)", "-", "no model", "-", "inherits", None,
     "Annotation rates are at chance; per-feature BH counts; all depend on SAEs trained on mis-encoded cells", "", "yes (CPU)", "CPU < 1 h after SAE retrain"),
    ("sae-atlas-217M", "atlas web app helper (Geneformer, not MaxToki)", "atlas/scripts/extract_and_enrich_missing_layers.py", F_TSS + " ; " + F_TSK + " ; " + F_TSL, XT,
     "log1p(X/sum*1e4) on X (double log) then /median: Geneformer V2-316M input", "2048", "NO (but not a MaxToki run)", None,
     "Geneformer atlas layers in the web app, if used", "copied from a Geneformer project; out of scope", "out of scope", "-"),
    ("circuit-tracing-217M", "deployed trace", "scripts/circuit_trace.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "2.14M edges, 88% inhibitory (already superseded by v2: hook bug)", "", "superseded", "-"),
    ("circuit-tracing-217M", "v2 trace (fixed hooks)", "scripts/v2_circuit_trace.py via setup/hooks_v2.load_k562_control_cells", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "272,905 v2 edges, 58.3% inhibitory, CRISPRi direction test on v2 edges", "", "yes", "MPS about 4.3 h (15,316 s, 200 cells x about 73 s)"),
    ("circuit-tracing-217M", "v2 spot check", "scripts/v2_circuit_spotcheck.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "3-edge spot check", "", "yes", "MPS < 15 min"),
    ("circuit-tracing-217M", "CRISPRi LFC labels", "scripts/v2_circuit_lfc.py, audit_a2_groupkfold_crispri.py, remaining_phases.py Phase 11", F_K562, XK,
     "label only: log1p(X/rowsum*1e4) on X that is already log1p (double log)", "-", "NO (labels)", None,
     "CRISPRi direction accuracy 54.6% (deployed) and the v2 direction metrics", "LFC sign is mostly kept by a monotone transform but the pseudobulk mean is of double-log values", "yes (CPU)", "CPU < 30 min"),
    ("exhaustive-mapping-217M", "Experiment 1 exhaustive", "scripts/experiment1_exhaustive.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "4,970,096 edges, density, hub annotation (also hook bug: block offset)", "", "superseded by hook audit", "MPS about 5.5 h if redone at the same size with hooks_v2"),
    ("exhaustive-mapping-217M", "Experiment 2 deployed + rerun", "scripts/experiments_2_3.py, experiment2_rerun.py", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "deployed triplets (hook bug; superseded by v2_triplets)", "", "superseded", "-"),
    ("exhaustive-mapping-217M", "v2 triplets", "scripts/v2_triplets.py via hooks_v2.load_k562_control_cells", F_K562, XK, T_X, 2048, "NO", "K562_ctrl_replogle_concat",
     "V2_TRIPLETS_REPORT: 512 triplets, 0 with BH targets, energy share -0.0000; v2_triplets_followup", "", "yes", "MPS about 1.7 h (5,963 s over 18 chunks)"),
    ("exhaustive-mapping-217M", "Experiment 3 steering (deployed)", "scripts/experiment3_rerun.py, experiments_2_3.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "published steering table (already shown to be a hook artefact)", "pseudotime = PC1 of log1p(X/sum*1e4) on X (double log)", "superseded", "-"),
    ("exhaustive-mapping-217M", "v2 steering", "scripts/v2_steering.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "v2 steering results and random-feature null", "pseudotime and early/late split from double-log expression", "yes", "MPS about 35 min (2,085 s over 5 chunks)"),
    ("exhaustive-mapping-217M", "v2 zero-edit control", "scripts/v2_zero_edit_control.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "'published steering table is a hook artefact' (this conclusion compares hook arithmetic on the same inputs, so it does not depend on the encoding)", "pseudotime double log", "optional", "MPS about 17 min"),
    ("manifold-discovery-217M", "Phase 1A tokenisation (internal, external, zero-shot, lung_control)", "scripts/phase1a_subsample_and_tokenize.py", F_TSI + " ; " + F_TSL, XT, T_X, 4096, "NO", "TS_immune_L4096_manifold",
     "cells_*.npz tokens -> Phase 1B/C hidden-state centroids -> every H65 number (trust 0.811 / 0.896 / 0.827, branch holdouts, head attribution L10H6, compaction, factor ablation), all v2_intervals and v2b_devorder MaxToki and token-bag features",
     "v2b_devorder HVG baseline correctly uses raw/X", "yes", "MPS about 8.3 h (internal 3.05 h, external 3.55 h, zero-shot 1.23 h, lung_control 0.44 h) + CPU phases 3-17, v2, v2b"),
    ("manifold-discovery-217M", "Phase 1A lung_nonhema negative control", "scripts/phase1a_lung_nonhema_panel.py", F_TSL, XT, T_X, 4096, "NO", "TS_lung_L4096_manifold",
     "negative-control panel numbers", "", "yes", "MPS about 22 min (1,321 s)"),
    ("spectral-geometry-217M", "Phase 0 per-gene embeddings", "scripts/phase0_extract.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "All layer >= 1 results: effective-rank collapse 561->94, TwoNN, SV enrichments, STRING, TF-vs-target, B/T compression, GC-plasma angle, BATF/BCL6. Layer-0 results are NOT affected (Llama layer 0 = token embedding, independent of order). Also reused by topology immune domain",
     "HVG list chosen on double-log X", "yes", "MPS about 13 min (2,000 cells)"),
    ("spectral-geometry-217M", "Phase 8 stability seeds", "scripts/phase8_stability.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "cross-seed stability of the spectrum", "", "yes", "MPS about 73 min"),
    ("spectral-geometry-217M", "autoloop H3-G centroid trajectory", "autoloop/iterations/iter_0003/h3g_centroid_trajectory.py", F_TSI, XT, T_X, 2048, "NO", "TS_immune_L2048",
     "H3-G centroid trajectory", "", "yes", "MPS about 10-20 min (estimate)"),
    ("spectral-geometry-217M", "Phase 9b / v2 cross-model", "scripts/phase9b_cross_model.py, audit_a3_*, v2_crossmodel_pearson_ci.py", "model.safetensors embed tables", "-", "no tokenisation (static embed_tokens tables)", "-", "not affected", None,
     "MaxToki-Geneformer Pearson 0.382 and its CI", "gene list comes from Phase 0 HVG choice (double log), a minor effect", "no", "0"),
    ("topology-141-217M", "Phase 0 lung + external lung (+ immune reused)", "scripts/phase0_extract.py", F_TSL + " ; " + F_KRA + " ; immune from spectral phase0", XT + " | " + XKR, T_X + " (Krasnow: log1p(CPM)/median)", 2048, "NO", "TS_lung_L2048_topology",
     "All layer >= 1 topology results: persistent homology, manifold distances, H16, H123, H91, H141 strict max-null, phases 15-16 (verification: Phase 14 reads static tables only, not affected), seed stability", "coexpression null (log_expression_pool.npy) built with double log", "yes", "MPS about 38 min (lung 1,182 s + ext-lung 1,100 s) + immune via spectral + seed-43 lung 20 min"),
    ("topology-141-217M", "Phase 0 external lung (Krasnow SS2)", "scripts/phase0_extract.py", F_KRA, XKR, T_XK, 2048, "NO", "Krasnow_lung_SS2_L2048_topology",
     "external-lung domain numbers", "", "yes", "included above"),
    ("topology-141-217M", "v2 cross-model; Phase 14 (Geneformer) and Phase 4 (scGPT) cross-model [added in verification]", "scripts/v2_crossmodel_*.py; scripts/phase14_cross_model_cca.py; scripts/phase4_scgpt_cross_model.py", "static embedding tables", "-", "no tokenisation", "-", "not affected", None,
     "V2_CROSSMODEL_REPORT (scGPT lookup bug; Pearson 0.27/0.26/0.28); Phase 14 H24/H17/H20 static CCA/Procrustes/Spearman (tap=static only); Phase 4 scGPT static alignment", "", "no", "0"),
    ("longevity-mechinterp-217M", "Stage 1 representations", "scripts/run_stage1.py -> maxtoki_runtime.extract_representations", F_TSS, XT, T_X, 1024, "NO", "TS_immune_sub20k_L1024_longevity",
     "G1 gate fail: L5 balanced accuracy 0.2755 vs null 0.332", "1,022-gene cap: wrong order also changes which genes are kept", "yes", "MPS about 60-90 min (stage 1 took about 92 min in total)"),
    ("setup", "tokeniser", "setup/maxtoki_adapter.py MaxTokiTokenizer.tokenize_cell", "-", "-", "expr/median, argsort; correct for counts or CP10k; no check that input is counts", "caller", "function OK; no guard", None,
     "every run above", "adds no per-cell normalisation (fine for order)", "add a counts guard", "0"),
    ("setup", "dataset loader", "setup/dataset_loader.py (resolve, load_hvg_matrix); setup/hooks_v2.load_k562_control_cells", F_K562 + " ; " + F_ADAM + " ; " + F_RPE1, XK + " | " + XA + " | " + XR, "returns X unchanged; callers treat it as counts", "-", "NO for k562/adamson", None,
     "all K562/Adamson model runs (attention, SAE, circuit, triplets, TF specificity)", "", "fix: expm1(X) for k562/adamson", "0"),
]

fields = ["run", "phase", "script", "data_file", "X_content", "transform_before_median_rank", "max_len", "correct",
          "order_check_dataset", "reference_order", "n_cells", "spearman_full_mean", "spearman_full_min", "spearman_as_fed_mean",
          "top200_overlap_mean", "top200_overlap_min", "top2046_overlap_mean", "kept_set_overlap_at_max_len_mean",
          "same_position_share_mean", "same_position_share_min", "spearman_full_vs_expm1X_mean", "top200_vs_expm1X_mean",
          "n_cells_stability_set", "spearman_full_mean_stability_set", "top200_overlap_mean_stability_set",
          "dependent_results", "other_scale_issues", "rerun_needed", "mps_cost_estimate"]
with open(DEST, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
        d = dict(run=r[0], phase=r[1], script=r[2], data_file=r[3], X_content=r[4], transform_before_median_rank=r[5],
                 max_len=r[6], correct=r[7], dependent_results=r[9], other_scale_issues=r[10], rerun_needed=r[11],
                 mps_cost_estimate=r[12])
        d.update(nums(r[8]))
        w.writerow({k: d.get(k, "") for k in fields})
print("wrote", DEST, len(rows))
