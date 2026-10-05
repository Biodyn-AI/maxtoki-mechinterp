"""v3 circuit tracing, part 4: CRISPRi direction test on v3 edges with single-log LFCs (item V3-3).

Same record rule and same metric code as v2 (imported unchanged from v2_circuit_crispri.py, which
re-implements scripts/audit_a2_groupkfold_crispri.py:51-162 and was checked there against the deployed
records). Only the inputs change:
  * edges:      outputs/v3_circuit/circuit_edges_v3.csv (v3 SAEs, correctly encoded cells, hooks_v2);
  * top genes:  outputs/v3_circuit/annotation/layer_XX/feature_catalog.json (v3 SAEs; first 10 of top20_genes);
  * labels:     outputs/v3_circuit/lfc/lfc_panel.npy  -- single log:
                LFC(s,g) = mean_{KD(s)} log1p(1e4 n_ig / sum_g' n_ig') - mean_{CTRL} log1p(1e4 n_ig / sum_g' n_ig'),
                n = round(expm1(X) / u_i) (integer counts up to one per-cell factor; 6,546 file genes),
                CTRL = 3,000 K562 non-targeting cells (rng 42), KD(s) = K562 cells of guide s (>= 10 cells).
Record rule (the method's rule): pair each top-10 gene of an edge's source feature with each top-10 gene
of its target feature (no self pairs); per gene pair evidence = number of triples, n_inhibitory, max |d|;
keep if evidence >= 2 or max |d| > 2; predicted decrease = n_inhibitory / evidence > 0.5 (tie -> increase);
keep if the source gene has a K562 CRISPRi guide with >= 10 cells and the target gene is measured;
observed decrease = LFC < 0.

Analyses (each on its own identical record set; positive class = observed decrease):
  A  v3 edges + single-log LFC (PRIMARY): full table, grouped bootstrap CIs (silenced genes; source-feature
     groups; connected components), cross-fitted target-direction baseline, per-gene table, |LFC| bands.
  B  v3 edges + file LFC (X = log1p(CP10k) with the full-library total; sensitivity).
  C  v3 edges + v2 double-log LFC (what the old labels say on the same records).
  D  v2 edges (deployed catalogs) + single-log LFC: the v2 records re-scored with corrected labels.
     Check: with the double-log LFC the rebuilt v2 records equal outputs/v2_circuit/per_pair.parquet.
Writes outputs/v3_circuit/crispri/ and outputs/v3_circuit/per_pair.parquet.
Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/circuit-tracing-217M/scripts/v3_circuit_crispri.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_circuit_crispri as C2  # noqa: E402  (record rule + metrics, unchanged)

H = C2.H
PROJ = C2.PROJ
RUN = PROJ / "runs/circuit-tracing-217M"
V3 = RUN / "outputs/v3_circuit"
V2 = RUN / "outputs/v2_circuit"
OUT = V3 / "crispri"
ANN = V3 / "annotation"
LFC = V3 / "lfc"
SEED = C2.SEED


def load_top10_v3():
    top10, labelled = {}, set()
    for li in range(12):
        d = ANN / f"layer_{li:02d}"
        for c in json.load(open(d / "feature_catalog.json")):
            top10[(li, int(c["feature_id"]))] = [g.upper() for g in c["top20_genes"][:10]]
        sig = pd.read_csv(d / "significant_enrichments.csv", usecols=["feature_id"])
        labelled |= {(li, int(f)) for f in sig.feature_id.unique()}
    return top10, labelled


def load_lfc_v3():
    src = pd.read_csv(LFC / "sources.csv")
    src = src[src.has_data].reset_index(drop=True)
    mats = {"panel": np.load(LFC / "lfc_panel.npy"), "file": np.load(LFC / "lfc_file.npy"),
            "v2style": np.load(LFC / "lfc_v2style.npy")}
    var_upper = (LFC / "var_symbols.txt").read_text().split("\n")
    gene_to_var = {s: i for i, s in enumerate(var_upper)}        # last occurrence wins, as deployed
    return src, mats, gene_to_var


def headline(res):
    p = res["pooled"]; bs = res["bootstrap_by_silenced_gene"]["stats"]
    bg = res["bootstrap_by_source_feature_group"]["stats"]; bc = res["bootstrap_by_component"]["stats"]
    tb = [b for b in res["baselines"] if b["predictor"].startswith("per-target")][0]
    return {"pairs": int(p["n"]), "silenced genes": p["n_sources"], "target genes": p["n_targets"],
            "obs decrease": p["frac_obs_dec"], "pred decrease": p["frac_pred_dec"],
            "accuracy": p["accuracy"], "CI accuracy (genes)": bs["pooled_accuracy"]["ci95"],
            "always decrease": p["always_dec_acc"], "always increase": p["always_inc_acc"],
            "majority class": p["majority_acc"],
            "acc - best constant": p["acc_minus_best_constant"],
            "CI acc - best constant (genes)": bs["pooled_acc_minus_best_constant_rechosen_each_rep"]["ci95"],
            "CI acc - best constant (feature groups)": bg["pooled_acc_minus_best_constant_rechosen_each_rep"]["ci95"],
            "CI acc - best constant (components)": bc["pooled_acc_minus_best_constant_rechosen_each_rep"]["ci95"],
            "balanced accuracy": p["balanced_acc"],
            "CI balanced acc - 0.5 (genes)": bs["pooled_balanced_acc_minus_0.5"]["ci95"],
            "CI balanced acc - 0.5 (feature groups)": bg["pooled_balanced_acc_minus_0.5"]["ci95"],
            "CI balanced acc - 0.5 (components)": bc["pooled_balanced_acc_minus_0.5"]["ci95"],
            "MCC": p["mcc"], "CI MCC (genes)": bs["pooled_mcc"]["ci95"], "CI MCC (feature groups)": bg["pooled_mcc"]["ci95"],
            "CI MCC (components)": bc["pooled_mcc"]["ci95"],
            "chance from marginals": p["chance_acc_given_marginals"],
            "acc - chance from marginals": p["acc_minus_chance_given_marginals"],
            "CI acc - chance from marginals (genes)": bs["pooled_acc_minus_chance_given_marginals"]["ci95"],
            "target-direction baseline (cross-fitted)": tb["accuracy"],
            "target baseline balanced acc": tb["balanced_acc"], "target baseline MCC": tb["mcc"],
            "acc - target baseline": bs["pooled_acc_minus_target_direction_baseline"]["point"],
            "CI acc - target baseline (genes)": bs["pooled_acc_minus_target_direction_baseline"]["ci95"],
            "per-gene mean MCC": bs["per_source_mean_mcc"]["point"], "CI per-gene mean MCC": bs["per_source_mean_mcc"]["ci95"],
            "n groups (genes / feature groups / components)": [res["bootstrap_by_silenced_gene"]["n_groups"],
                                                              res["bootstrap_by_source_feature_group"]["n_groups"],
                                                              res["bootstrap_by_component"]["n_groups"]]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    top10, labelled = load_top10_v3()
    feats = json.loads((V3 / "source_features.json").read_text())
    src, mats, gene_to_var = load_lfc_v3()
    S = sorted(src.source)
    lfc_rows = dict(zip(src.source, src.matrix_row))
    edges = pd.read_csv(V3 / "circuit_edges_v3.csv")
    edge_feats = set(zip(edges.src_layer, edges.src_feature))
    results, checks, recs = {}, {}, {}

    # ---------------- A, B, C: v3 edges with three label definitions
    for lab in ["panel", "file", "v2style"]:
        rec, n_used = C2.build_records(edges, top10, S, lfc_rows, mats[lab], gene_to_var)
        recs[lab] = rec
        checks[f"records_v3_edges_{lab}"] = dict(edges_total=int(len(edges)), edges_used=n_used, n_records=int(len(rec)),
                                                 n_sources=int(rec.source.nunique()), n_targets=int(rec.target.nunique()),
                                                 n_lfc_exactly_zero=int((rec.actual_lfc == 0).sum()))
    # identical records across label definitions (only the label differs)
    k = ["source", "target", "evidence", "n_inhibitory", "predicted_inhibitory"]
    checks["same_records_across_labels"] = bool(recs["panel"][k].equals(recs["file"][k]) and recs["panel"][k].equals(recs["v2style"][k]))
    rp = recs["panel"]
    checks["label_agreement_on_v3_records"] = dict(
        sign_agree_panel_vs_v2style=float(((rp.actual_lfc < 0) == (recs["v2style"].actual_lfc < 0)).mean()),
        sign_agree_panel_vs_file=float(((rp.actual_lfc < 0) == (recs["file"].actual_lfc < 0)).mean()),
        pearson_panel_vs_v2style=float(np.corrcoef(rp.actual_lfc, recs["v2style"].actual_lfc)[0, 1]),
        pearson_panel_vs_file=float(np.corrcoef(rp.actual_lfc, recs["file"].actual_lfc)[0, 1]))
    folds_order = sorted(rp.source.unique())
    res_A, df_A, ps_A = C2.evaluate(rp, "v3 edges, single-log LFC", folds_order, top10, feats, edge_feats)
    results["A_v3_edges_single_log"] = res_A
    ps_A.to_csv(OUT / "per_source_metrics_v3_edges.csv")
    res_B, _, _ = C2.evaluate(recs["file"], "v3 edges, file log1p(CP10k) LFC", folds_order, top10, feats, edge_feats)
    results["B_v3_edges_file_lfc"] = res_B
    res_C, _, _ = C2.evaluate(recs["v2style"], "v3 edges, v2 double-log LFC", folds_order, top10, feats, edge_feats)
    results["C_v3_edges_double_log"] = res_C
    # sensitivities on A
    tie_dec = df_A.pred_dec | (df_A.frac_inhibitory == 0.5)
    results["A_sensitivity_ties_as_decrease"] = C2.scal(C2.metrics_from_counts(*C2.conf_counts(tie_dec, df_A.obs_dec)))
    lab_mask = np.array([(a, b) in labelled and (c, d) in labelled for a, b, c, d in
                         zip(edges.src_layer, edges.src_feature, edges.tgt_layer, edges.tgt_feature)])
    rec_lab, _ = C2.build_records(edges, top10, S, lfc_rows, mats["panel"], gene_to_var, edge_mask=lab_mask)
    if len(rec_lab):
        results["A_sensitivity_labelled_edges_only"] = dict(n_edges=int(lab_mask.sum()), n_sources=int(rec_lab.source.nunique()),
                                                            **C2.scal(C2.metrics_from_counts(*C2.conf_counts(rec_lab.predicted_inhibitory, rec_lab.actual_inhibitory))))
    sumd = df_A.sum_d < 0
    results["A_sensitivity_sign_of_summed_d"] = C2.scal(C2.metrics_from_counts(*C2.conf_counts(sumd, df_A.obs_dec)))

    # ---------------- D: v2 edges (deployed catalogs) re-scored with corrected labels
    top10_dep, _ = C2.load_top10()
    v2_edges = pd.read_csv(V2 / "circuit_edges_v2.csv")
    v2_feats = json.loads((V2 / "source_features.json").read_text())
    v2_pp = pd.read_parquet(V2 / "per_pair.parquet")
    rec_v2_old, _ = C2.build_records(v2_edges, top10_dep, S, lfc_rows, mats["v2style"], gene_to_var)
    m = v2_pp.merge(rec_v2_old, on=["source", "target"], how="outer", suffixes=("_pp", "_rb"), indicator=True)
    both = m[m._merge == "both"]
    checks["v2_records_rebuilt"] = dict(n_v2=int(len(v2_pp)), n_rebuilt=int(len(rec_v2_old)), n_both=int(len(both)),
                                        pred_sign_matches=int((both.predicted_inhibitory_pp == both.predicted_inhibitory_rb).sum()),
                                        max_abs_lfc_diff=float((both.actual_lfc_pp - both.actual_lfc_rb).abs().max()))
    checks["v2_records_rebuilt"]["pass"] = bool(len(both) == len(v2_pp) == len(rec_v2_old)
                                                and checks["v2_records_rebuilt"]["pred_sign_matches"] == len(v2_pp)
                                                and checks["v2_records_rebuilt"]["max_abs_lfc_diff"] == 0.0)
    rec_v2_new, _ = C2.build_records(v2_edges, top10_dep, S, lfc_rows, mats["panel"], gene_to_var)
    res_D, _, _ = C2.evaluate(rec_v2_new, "v2 edges, single-log LFC", sorted(rec_v2_new.source.unique()), top10_dep, v2_feats,
                              set(zip(v2_edges.src_layer, v2_edges.src_feature)))
    results["D_v2_edges_single_log"] = res_D
    checks["label_agreement_on_v2_records"] = dict(
        sign_agree_single_vs_double=float(((rec_v2_new.actual_lfc < 0) == (rec_v2_old.actual_lfc < 0)).mean()),
        frac_obs_dec_single=float((rec_v2_new.actual_lfc < 0).mean()), frac_obs_dec_double=float((rec_v2_old.actual_lfc < 0).mean()))

    # ---------------- record overlap v3 vs v2 (gene pairs; the features are different)
    mm = rp.merge(v2_pp, on=["source", "target"], suffixes=("_v3", "_v2"))
    checks["record_overlap_v3_vs_v2"] = dict(n_v3=int(len(rp)), n_v2=int(len(v2_pp)), n_shared_pairs=int(len(mm)),
                                             n_shared_sources=int(mm.source.nunique()),
                                             n_v3_sources=int(rp.source.nunique()), n_v2_sources=int(v2_pp.source.nunique()),
                                             n_sources_in_both=int(len(set(rp.source) & set(v2_pp.source))),
                                             pred_sign_agreement_shared=float((mm.predicted_inhibitory_v3 == mm.predicted_inhibitory_v2).mean()) if len(mm) else None)

    # ---------------- outputs
    grp = ps_A[["source_group", "source_component"]]
    out_pp = df_A.merge(grp, left_on="source", right_index=True, how="left")
    out_pp = out_pp.merge(recs["file"][["source", "target", "actual_lfc"]].rename(columns={"actual_lfc": "lfc_file"}), on=["source", "target"])
    out_pp = out_pp.merge(recs["v2style"][["source", "target", "actual_lfc"]].rename(columns={"actual_lfc": "lfc_v2_double_log"}), on=["source", "target"])
    out_pp = out_pp[["source", "target", "evidence", "n_inhibitory", "frac_inhibitory", "max_abs_d", "sum_d",
                     "predicted_inhibitory", "actual_lfc", "actual_inhibitory", "correct", "src_fold",
                     "pred_target_major", "source_group", "source_component", "lfc_file", "lfc_v2_double_log"]]
    out_pp = out_pp.rename(columns={"pred_target_major": "target_baseline_pred_inhibitory"})
    out_pp.to_parquet(V3 / "per_pair.parquet", index=False)

    v2res = json.loads((V2 / "crispri/results.json").read_text())["results"]
    rows = []
    for name, r in [("v3 edges + single-log LFC (primary)", res_A), ("v3 edges + file log1p(CP10k) LFC", res_B),
                    ("v3 edges + v2 double-log LFC", res_C), ("v2 edges + single-log LFC", res_D),
                    ("v2 edges + v2 double-log LFC (V2 report)", v2res["v2"]),
                    ("deployed edges + double-log LFC (V2 report)", v2res["deployed"])]:
        rows.append({"analysis": name, **headline(r)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "comparison_table.csv", index=False)
    H.write_json(OUT / "results.json", dict(results=results, checks=checks))
    lfc_checks = json.loads((LFC / "checks.json").read_text())
    cfg = dict(script=str(Path(__file__)), script_sha256=H.sha256_file(__file__),
               metric_code=str(HERE / "v2_circuit_crispri.py"), metric_code_sha256=H.sha256_file(HERE / "v2_circuit_crispri.py"),
               device="cpu", seeds=dict(bootstrap=SEED, source_folds=SEED, within_source_target_folds=SEED + 1), n_boot=C2.NBOOT,
               inputs=dict(edges=str(V3 / "circuit_edges_v3.csv"), edges_sha256=H.sha256_file(V3 / "circuit_edges_v3.csv"),
                           catalogs=str(ANN), lfc_panel_sha256=H.sha256_file(LFC / "lfc_panel.npy"),
                           lfc_file_sha256=H.sha256_file(LFC / "lfc_file.npy"), lfc_v2style_sha256=H.sha256_file(LFC / "lfc_v2style.npy"),
                           v2_edges=str(V2 / "circuit_edges_v2.csv"), v2_per_pair=str(V2 / "per_pair.parquet")),
               lfc_cell_counts=lfc_checks["cell_counts"],
               numpy=np.__version__, pandas=pd.__version__, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
               wall_seconds=round(time.time() - t0, 1))
    H.write_json(OUT / "run_config.json", cfg)
    print(json.dumps(checks, indent=1, default=str))
    print(tab.T.to_string())


if __name__ == "__main__":
    main()
