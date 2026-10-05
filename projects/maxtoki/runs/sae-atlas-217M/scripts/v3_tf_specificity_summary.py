"""V3-2: summary numbers, GATA1 detail and bootstrap, v2/deployed comparison, run_config, README.

Reads the outputs of v3_tf_specificity_{extract,taskdata,stats,calib_check,verify,replay_v2}.py and
the v2 outputs, and writes into outputs/v3_tf_specificity/:
  summary.json, summary_tables.md, summary_counts.csv, power_summary.json,
  gata1_bootstrap.json  (cell bootstrap of the GATA1 selection; scipy Mann-Whitney, exact ties,
                         because resampling duplicates cells),
  gata1_detail.json     (GATA1 responding features: effects, top-20 genes, ChIP targets, and the
                         closest deployed (old-SAE) features by decoder cosine),
  v2_vs_v3.json, run_config.json (merged provenance), task_data/README.md
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats as st  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).parent))
import v3_tf_specificity_stats as S  # noqa: E402
from v3_tf_specificity_stats import bh, cap_T, CUTOFFS, Q_FEAT, SEED, D_SAE  # noqa: E402

H = S.H
OUT = S.OUT
TD = S.TD
RUN = S.RUN
V2 = RUN / "v2_tf_specificity"
SCRIPTS = Path(__file__).parent
N_BOOT = 1000


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def gata1_bootstrap(man, MG, rix, ov_chip, feats):
    gpath = OUT / "gata1_bootstrap.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        if g.get("n_boot") == N_BOOT and g.get("seed") == SEED + 2024 and g.get("features") == [int(f) for f in feats]:
            return g
    R = MG[[rix[int(r)] for r in man[man.group == "ref"].row]].astype(np.float64)
    A = MG[[rix[int(r)] for r in man[(man.group == "kd") & (man.tf == "GATA1")].row]].astype(np.float64)
    rb = np.random.default_rng(SEED + 2024)
    sel05 = np.zeros(D_SAE); sel025 = np.zeros(D_SAE)
    K05, T05, M05, M025 = [], [], [], []
    dboot = np.zeros((N_BOOT, len(feats)))
    t0 = time.time()
    for b in range(N_BOOT):
        Ab = A[rb.integers(0, len(A), len(A))]; Rb = R[rb.integers(0, len(R), len(R))]
        p = np.nan_to_num(st.mannwhitneyu(Ab, Rb, axis=0).pvalue, nan=1.0); q = bh(p)
        dl = Ab.mean(0) - Rb.mean(0)
        s05 = (q < Q_FEAT) & (np.abs(dl) > 0.5); s025 = (q < Q_FEAT) & (np.abs(dl) > 0.25)
        sel05 += s05; sel025 += s025
        K05.append(int(s05.sum()))
        M05.append(int(ov_chip[s05].max()) if s05.any() else 0)
        T05.append(int(cap_T(M05[-1])))
        M025.append(int(ov_chip[s025].max()) if s025.any() else 0)
        dboot[b] = dl[feats]
    dl_obs = A.mean(0) - R.mean(0)
    g = {"n_boot": N_BOOT, "seed": SEED + 2024, "features": [int(f) for f in feats],
         "resampling": "cells, knockdown and reference resampled separately with replacement; "
                       "scipy.stats.mannwhitneyu (asymptotic, exact tie correction incl. duplicated cells), BH",
         "delta": {int(f): {"obs": float(dl_obs[f]),
                            "ci95": [float(np.percentile(dboot[:, j], 2.5)), float(np.percentile(dboot[:, j], 97.5))],
                            "frac_boot_selected_cut0.5": float(sel05[f] / N_BOOT),
                            "frac_boot_selected_cut0.25": float(sel025[f] / N_BOOT)} for j, f in enumerate(feats)},
         "K_cut0.5_quantiles_2.5_50_97.5": [float(x) for x in np.percentile(K05, [2.5, 50, 97.5])],
         "M_chip_cut0.5_counts": {int(k): int(v) for k, v in zip(*np.unique(M05, return_counts=True))},
         "T_chip_cut0.5_counts": {int(k): int(v) for k, v in zip(*np.unique(T05, return_counts=True))},
         "M_chip_cut0.25_counts": {int(k): int(v) for k, v in zip(*np.unique(M025, return_counts=True))},
         "features_selected_in_ge_50pct_cut0.5": [int(f) for f in np.where(sel05 / N_BOOT >= 0.5)[0]],
         "features_selected_in_any_boot_cut0.5": int((sel05 > 0).sum()),
         "wall_s": round(time.time() - t0, 1)}
    H.write_json(gpath, g)
    return g


def main():
    t0 = time.time()
    res = pd.read_csv(OUT / "tf_results.csv")
    pw = pd.read_csv(OUT / "power.csv")
    cal = json.load(open(OUT / "calibration.json"))
    calx = json.load(open(OUT / "calibration_extra.json"))
    rcs = json.load(open(OUT / "run_config_stats.json"))
    rce = json.load(open(OUT / "run_config_extract.json"))
    cchk = json.load(open(TD / "catalog_check.json"))
    prep = json.load(open(OUT / "prepare.json"))
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    kdeff = json.load(open(OUT / "knockdown_efficiency.json"))
    old = pd.read_csv(OUT / "old_trrust_test.csv")
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    verify = json.load(open(OUT / "checks/verify.json"))
    replay = json.load(open(OUT / "checks/replay_v2.json"))
    prog = json.load(open(OUT / "extract_progress.json"))
    primary = meta["tfs_primary"]
    res["rank_cm_valid"] = np.where(res.T_obs >= 2, res.rank_cm, np.nan)
    res["rank_frac_valid"] = np.where(res.T_obs >= 2, res.rank_frac, np.nan)
    B = int(rcs["B_fake"])
    Sm = {}

    # ---------------- task data ----------------
    z = np.load(TD / "cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    uni = pd.read_csv(TD / "gene_universe.tsv", sep="\t", keep_default_na=False)
    dcount = dict(zip(uni.gene, uni.detection_count))
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    chip = set(dor[(dor.tf == "GATA1") & (dor.chip_flag.astype(str).str.lower() == "true")].target)
    lists = top.sort_values(["feature_id", "rank"]).groupby("feature_id").gene.apply(list).to_dict()
    ov_chip = np.array([sum(g in chip for g in lists.get(f, [])) for f in range(D_SAE)])
    rfeat = pd.read_csv(TD / "responding_features.tsv", sep="\t")
    g1 = rfeat[rfeat.tf == "GATA1"].set_index("feature_id")
    resp05 = sorted(g1[(g1.bh_q < Q_FEAT) & (g1.delta_mean.abs() > 0.5)].index.tolist())
    gres = res[res.tf == "GATA1"].set_index(["cutoff", "db"])
    best_other = sorted({int(gres.loc[(c, "ChIP")].best_feature) for c in CUTOFFS} - set(resp05))

    # ---------------- bridge to the deployed (old) SAE by decoder cosine ----------------
    def wdec(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        W = ck["W_dec_weight"].float()
        return (W / W.norm(dim=0, keepdim=True)).numpy()
    Wd = wdec(RUN / "phase1/layer_05/sae_final.pt")
    Wv = wdec(RUN / "v3_sae/layer_05/sae_final.pt")
    C = np.abs(Wd.T @ Wv)                     # (old, new)
    v2top = pd.read_csv(V2 / "task_data/feature_top20.tsv", sep="\t", keep_default_na=False)
    v2lists = v2top.sort_values(["feature_id", "rank"]).groupby("feature_id").gene.apply(list).to_dict()
    v2res = pd.read_csv(V2 / "tf_results.csv")
    v2rf = pd.read_csv(V2 / "task_data/responding_features.tsv", sep="\t")
    v2g = v2rf[v2rf.tf == "GATA1"].set_index("feature_id")
    dep_feats = [628, 1334, 2006, 2610, 2627, 3167]
    bridge_old = {}
    for f in dep_feats:
        j = int(np.argmax(C[f])); cs = float(C[f, j])
        # gene-list match: v3 feature whose top-20 shares the most genes with old f's list
        olds = set(v2lists.get(f, []))
        jac = np.array([len(olds & set(lists.get(k, []))) for k in range(D_SAE)])
        jl = int(np.argmax(jac))
        bridge_old[f] = {"old_top20_chip_targets": int(sum(g in chip for g in v2lists.get(f, []))),
                         "old_delta_v2": float(v2g.delta_mean.get(f, np.nan)),
                         "best_v3_by_decoder": j, "abs_cos": cs,
                         "second_best_abs_cos": float(np.sort(C[f])[-2]),
                         "v3_feature_top20_shared_genes_with_old": int(len(olds & set(lists.get(j, [])))),
                         "v3_feature_chip_targets": int(ov_chip[j]),
                         "v3_feature_GATA1_delta": float(g1.delta_mean.get(j, np.nan)) if j in g1.index else None,
                         "v3_feature_GATA1_q": float(g1.bh_q.get(j)) if j in g1.index else None,
                         "best_v3_by_top20_genes": jl, "shared_genes": int(jac[jl]),
                         "best_by_genes_chip_targets": int(ov_chip[jl]),
                         "best_by_genes_GATA1_delta": float(g1.delta_mean.get(jl)) if jl in g1.index else None,
                         "best_by_genes_abs_cos": float(C[f, jl])}
    bridge_new = {int(f): {"best_old_by_decoder": int(np.argmax(C[:, f])), "abs_cos": float(C[:, f].max())}
                  for f in resp05 + best_other}
    rv = np.random.default_rng(SEED + 9).standard_normal((1232, 200)); rv /= np.linalg.norm(rv, axis=0)
    rand_best = np.abs(Wd.T @ rv).max(0)
    Sm["bridge_random_direction_best_abs_cos_median"] = float(np.median(rand_best))

    # ---------------- GATA1 bootstrap + detail ----------------
    feats = resp05 + best_other + [bridge_old[2610]["best_v3_by_decoder"]]
    feats = list(dict.fromkeys(int(f) for f in feats))
    gb = gata1_bootstrap(man, MG, rix, ov_chip, feats)
    Sm["gata1_bootstrap"] = gb
    detail = {}
    for f in feats:
        L = lists.get(f, [])
        tg = [g for g in L if g in chip]
        detail[f] = {"responding_cut0.5": f in resp05,
                     "delta": float(g1.delta_mean.get(f, np.nan)) if f in g1.index else float(MG[[rix[int(r)] for r in man[(man.group == 'kd') & (man.tf == 'GATA1')].row]][:, f].mean() - MG[[rix[int(r)] for r in man[man.group == 'ref'].row]][:, f].mean()),
                     "bh_q": float(g1.bh_q.get(f)) if f in g1.index else None,
                     "cohen_d": float(g1.cohen_d.get(f)) if f in g1.index else None,
                     "ci95": gb["delta"][str(f)]["ci95"] if str(f) in gb["delta"] else gb["delta"][f]["ci95"],
                     "frac_boot_selected_cut0.5": (gb["delta"][str(f)] if str(f) in gb["delta"] else gb["delta"][f])["frac_boot_selected_cut0.5"],
                     "top20": L, "chip_targets_in_top20": tg, "n_chip": len(tg),
                     "detection_counts_of_chip_targets": [int(dcount.get(g, 0)) for g in tg],
                     "median_detection_count_top20": float(np.median([dcount[g] for g in L if g in dcount])) if L else None,
                     "best_old_feature": int(np.argmax(C[:, f])), "best_old_abs_cos": float(C[:, f].max())}
    H.write_json(OUT / "gata1_detail.json", {"responding_cut0.5": resp05, "features": detail, "bridge_old_to_v3": bridge_old,
                                             "bridge_v3_to_old": bridge_new,
                                             "max_chip_overlap_any_v3_feature": int(ov_chip.max()),
                                             "n_v3_features_with_chip_overlap_ge_5": int((ov_chip >= 5).sum()),
                                             "n_v3_features_with_chip_overlap_ge_8": int((ov_chip >= 8).sum()),
                                             "v3_features_chip_ge_6": {int(f): {"n_chip": int(ov_chip[f]),
                                                                                "GATA1_delta": float(g1.delta_mean.get(f)) if f in g1.index else None,
                                                                                "GATA1_q": float(g1.bh_q.get(f)) if f in g1.index else None}
                                                                       for f in np.where(ov_chip >= 6)[0]},
                                             "random_direction_best_abs_cos_median": Sm["bridge_random_direction_best_abs_cos_median"]})
    print("GATA1 detail done", round(time.time() - t0), flush=True)

    # ---------------- per cut-off counts ----------------
    tab_rows = []
    for c in CUTOFFS:
        for db in ["ChIP", "TRRUST"]:
            for scope, tset in [("primary20", primary), ("all87", meta["tfs_all"])]:
                d = res[(res.cutoff == c) & (res.db == db) & res.tf.isin(tset) & (res.n_targets_in_universe > 0)]
                qs = scope.replace("20", "").replace("87", "")
                tab_rows.append({"cutoff": c, "db": db, "scope": scope, "n_tfs_with_set": len(d),
                                 "n_testable": int((d.K > 0).sum()), "n_T_ge2": int((d.T_obs >= 2).sum()),
                                 "median_K_testable": float(d[d.K > 0].K.median()) if (d.K > 0).any() else np.nan,
                                 "n_q_fake_lt05": int((d[f"q_fake_{qs}"] < 0.05).sum()),
                                 "n_q_cm_lt05": int((d[f"q_cm_{qs}"] < 0.05).sum()),
                                 "n_q_cml_lt05": int((d[f"q_cml_{qs}"] < 0.05).sum()),
                                 "n_q_fakeK_lt05": int((d[f"q_fakeK_{qs}"] < 0.05).sum()),
                                 "n_q_okd_lt05": int((d[f"q_okd_{qs}"] < 0.05).sum()),
                                 "n_p_cm_lt05": int((d.p_cm < 0.05).sum()), "n_p_cm_uncapped_lt05": int((d.p_cm_uncapped < 0.05).sum()),
                                 "n_p_cml_lt05": int((d.p_cml < 0.05).sum()),
                                 "n_p_fakeK_lt05": int((d.p_fakeK < 0.05).sum()), "n_p_rf_lt05": int((d.p_rf < 0.05).sum()),
                                 "n_p_okd_lt05": int((d.p_okd < 0.05).sum()),
                                 "n_rank_top5pct": int((d.rank_frac_valid <= 0.05).sum())})
    counts = pd.DataFrame(tab_rows)
    counts.to_csv(OUT / "summary_counts.csv", index=False)

    # ---------------- power with CIs ----------------
    rb = np.random.default_rng(SEED + 2025)
    prow = []
    pcols = ["power_rf", "power_cm", "power_cm_uncapped", "power_cml", "power_cml_uncapped", "power_fake", "power_fakeK", "power_okd"]
    for (c, db, k), d in pw.groupby(["cutoff", "db", "k"]):
        row = {"cutoff": c, "db": db, "k": k, "n_tfs": int(d.tf.nunique()), "tfs": sorted(d.tf.unique())}
        for col in pcols:
            if col not in d:
                continue
            v = d[col].values
            row[col] = float(np.nanmean(v))
            if len(v) > 1:
                bs = [np.nanmean(rb.choice(v, len(v))) for _ in range(2000)]
                row[col + "_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            else:
                kk = int(round(v[0] * int(rcs["R_power"])))
                row[col + "_ci"] = list(wilson(kk, int(rcs["R_power"])))
        prow.append(row)
    pwr = pd.DataFrame(prow)
    pwr.to_json(OUT / "power_summary.json", orient="records", indent=1)
    gp = pw[(pw.tf == "GATA1")]
    gpow = {}
    for (c, db, k), d in gp.groupby(["cutoff", "db", "k"]):
        r = d.iloc[0]
        gpow[f"{c}|{db}|{k}"] = {col: [float(r[col]), *wilson(int(round(r[col] * int(rcs["R_power"]))), int(rcs["R_power"]))]
                                 for col in pcols if col in d and np.isfinite(r[col])}
    Sm["power_gata1"] = gpow

    # ---------------- v2 vs v3 ----------------
    v2s = pd.read_csv(V2 / "task_data/tf_summary.tsv", sep="\t")
    v3s = pd.read_csv(TD / "tf_summary.tsv", sep="\t")
    mm = v3s.merge(v2s, on="tf", suffixes=("_v3", "_v2"))
    cmp = {"spearman_n_bh_sig_87": float(st.spearmanr(mm.n_bh_sig_v3, mm.n_bh_sig_v2).correlation),
           "n_tfs_with_any_bh_feature": {"v3": int((mm.n_bh_sig_v3 > 0).sum()), "v2": int((mm.n_bh_sig_v2 > 0).sum())},
           "K_cut0.5_tfs": {"v3": {t: int(k) for t, k in zip(mm.tf, mm["K_cut_0.5_v3"]) if k > 0},
                            "v2": {t: int(k) for t, k in zip(mm.tf, mm["K_cut_0.5_v2"]) if k > 0}},
           "primary": mm[mm.primary_v3][["tf", "n_kd_cells_v3", "n_bh_sig_v2", "n_bh_sig_v3", "K_cut_0.5_v2", "K_cut_0.5_v3",
                                          "K_cut_0.1_v2", "K_cut_0.1_v3", "K_cut_0.01_v2", "K_cut_0.01_v3",
                                          "n_chip_in_universe_v2", "n_chip_in_universe_v3"]].to_dict("records")}
    v2c = pd.read_csv(V2 / "summary_counts.csv")
    cmp["bh_passes_any_fair_null"] = {
        "v3": int(counts[["n_q_cm_lt05", "n_q_cml_lt05", "n_q_fakeK_lt05", "n_q_okd_lt05"]].values.sum()),
        "v2": int(v2c[["n_q_cm_lt05", "n_q_cml_lt05", "n_q_fakeK_lt05", "n_q_okd_lt05"]].values.sum())}
    keys = ["K", "M_obs", "T_obs", "best_feature", "p_rf", "p_cm", "p_cml", "p_cm_uncapped", "p_fake", "p_fakeK", "p_okd", "n_okd", "rank_cm", "n_rank_cand"]
    cmp["GATA1"] = {}
    for c in CUTOFFS:
        for db in ["ChIP", "TRRUST"]:
            a = v2res[(v2res.tf == "GATA1") & (v2res.cutoff == c) & (v2res.db == db)].iloc[0]
            b = res[(res.tf == "GATA1") & (res.cutoff == c) & (res.db == db)].iloc[0]
            cmp["GATA1"][f"{c}|{db}"] = {"v2": {k: (None if pd.isna(a[k]) else float(a[k])) for k in keys},
                                         "v3": {k: (None if pd.isna(b[k]) else float(b[k])) for k in keys}}
    v2pw = pd.DataFrame(json.load(open(V2 / "power_summary.json")))
    cmp["power_v2_cut0.5"] = v2pw[v2pw.cutoff == 0.5][["db", "k", "power_rf", "power_cm", "power_cm_uncapped", "power_cml", "power_cml_uncapped"]].to_dict("records")
    v2old = pd.read_csv(V2 / "old_trrust_test.csv")
    cmp["old_trrust_v2"] = v2old.to_dict("records")
    H.write_json(OUT / "v2_vs_v3.json", cmp)
    Sm["v2_vs_v3"] = {k: cmp[k] for k in ["spearman_n_bh_sig_87", "n_tfs_with_any_bh_feature", "bh_passes_any_fair_null", "K_cut0.5_tfs"]}

    # ---------------- key numbers ----------------
    Sm["cells"] = {"ref": int((man.group == "ref").sum()), "pool": int((man.group == "pool").sum()),
                   "catalog": int((man.group == "catalog").sum()), "kd": int((man.group == "kd").sum()),
                   "tfs_all": len(meta["tfs_all"]), "tfs_primary": len(primary)}
    ks = [v["KS_p_randomized_pooled"] for k, v in calx["D_calibration_1000_per_tf"].items() if isinstance(v, dict)]
    Sm["checks"] = {
        "manifest_equals_v2": prep["manifest_vs_v2"],
        "encoding": {"cells_checked_before_forward": prog["enc_cells_checked"], "batches": prog["enc_batches"],
                     "max_order_violations": prog["enc_max_order_violations"],
                     "positions_tie_reordered": prog["enc_tie_reordered"],
                     "rows_accepted_unit_one": prog.get("rows_accepted_unit_one"),
                     "old_v2_tokens_fail_check": prep["order_sample"]["old_v2_tokens_fail_check"],
                     "old_vs_new_order_sample": prep["order_sample"]["old_vs_new_agreement_mean"]},
        "bos_code_identical_across_cells": prog["bos_max_abs_diff"],
        "catalog": {k: cchk[k] for k in ["token_check", "codes_A_vs_B", "catalog_A_vs_B", "catalog", "universe"]},
        "fast_mw_vs_scipy": rcs["log"]["fast_mw_check"], "tie_index": rcs["log"].get("tie_index"),
        "swap_null_validation": {nk: {"max_abs_gap_P": rcs["log"]["swap_null_validation"][nk]["max_abs_gap_P"],
                                      "max_gap_in_MC_SE": rcs["log"]["swap_null_validation"][nk]["max_gap_in_MC_SE"],
                                      "max_abs_gap_E": rcs["log"]["swap_null_validation"][nk]["max_abs_gap_E"]}
                                 for nk in ["cm", "cml"]},
        "replay_v2": replay, "verify": verify,
        "repartition": calx["B_repartition"], "pool_vs_ref": calx["A_pool_vs_ref"],
        "fakeK_reproduction": calx["C_fakeK_reproduction_cut0.25"],
        "calibration_summary": {"n_KS_tests": len(ks), "min_KS_p": float(min(ks)), "median_KS_p": float(np.median(ks)),
                                "n_KS_p_lt_0.05": int(sum(x < 0.05 for x in ks)),
                                "max_frac_conservative_p_le_0.05": float(max(v["frac_p_conservative_le_0.05"] for v in calx["D_calibration_1000_per_tf"].values() if isinstance(v, dict))),
                                "chi2_lt_0.05": int(sum(v["chi2_two_sample_n_lt_0.05"] for v in calx["D_calibration_1000_per_tf"].values() if isinstance(v, dict))),
                                "chi2_tests": int(sum(v["n_tfs_with_varying_T"] for v in calx["D_calibration_1000_per_tf"].values() if isinstance(v, dict)))},
        "calibration_250_per_tf_main_run": {k: v for k, v in cal.items() if k.startswith("cutoff_")},
        "feature_level_fixed_pool": cal["feature_level_MW_p_fake_TFs"]}
    Sm["gata1"] = {f"{c}|{db}": {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
                                 for k, v in gres.loc[(c, db)][["K", "M_obs", "T_obs", "best_feature", "p_rf", "p_cm", "p_cml", "p_cm_uncapped",
                                                                "p_cml_uncapped", "E_M_cm", "p_fake", "p_fakeK", "p_okd", "n_okd",
                                                                "rank_cm_valid", "n_rank_cand", "q_fake_primary", "q_cm_primary"]].items()}
                   for c in CUTOFFS for db in ["ChIP", "TRRUST"]}
    Sm["p_fake_floor"] = {"B": B, "min_p": 1 / (B + 1)}
    Sm["old_trrust_test"] = old.to_dict("records")
    Sm["power_primary_cutoff"] = pwr[pwr.cutoff == 0.5].to_dict("records")
    Sm["knockdown_efficiency"] = kdeff
    Sm["nominal_hits"] = res[(res.p_cm < 0.05) | (res.p_cml < 0.05) | (res.p_fakeK < 0.05) | (res.p_okd < 0.05) | (res.p_cm_uncapped < 0.05)][
        ["tf", "cutoff", "db", "K", "M_obs", "T_obs", "p_rf", "p_cm", "p_cml", "p_cm_uncapped", "p_fakeK", "p_okd", "n_okd",
         "rank_cm", "n_rank_cand", "q_cm_all", "q_cm_primary", "q_fakeK_all", "q_fakeK_primary"]].to_dict("records")
    H.write_json(OUT / "summary.json", Sm)

    # ---------------- markdown tables ----------------
    L = []
    L.append("### Table A. Number of TFs per cut-off (primary set = 20 TFs with DoRothEA ChIP-seq targets)\n")
    L.append("| cut-off | db | TFs with set | testable (K>0) | median K | T>=2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | p<0.05 cm (uncapped) | true TF in top 5% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in counts[counts.scope == "primary20"].iterrows():
        L.append(f"| {r.cutoff} | {r.db} | {r.n_tfs_with_set} | {r.n_testable} | {fmt(r.median_K_testable, 0)} | {r.n_T_ge2} | {r.n_q_fake_lt05} | {r.n_q_cm_lt05} | {r.n_q_cml_lt05} | {r.n_q_fakeK_lt05} | {r.n_q_okd_lt05} | {r.n_p_cm_uncapped_lt05} | {r.n_rank_top5pct} |")
    L.append("\n### Table A2. Same, all 87 TFs\n")
    L.append("| cut-off | db | TFs with set | testable | median K | T>=2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | p<0.05 cm uncapped | top 5% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in counts[counts.scope == "all87"].iterrows():
        L.append(f"| {r.cutoff} | {r.db} | {r.n_tfs_with_set} | {r.n_testable} | {fmt(r.median_K_testable, 0)} | {r.n_T_ge2} | {r.n_q_fake_lt05} | {r.n_q_cm_lt05} | {r.n_q_cml_lt05} | {r.n_q_fakeK_lt05} | {r.n_q_okd_lt05} | {r.n_p_cm_uncapped_lt05} | {r.n_rank_top5pct} |")
    L.append("\n### Table B. GATA1, all tests\n")
    L.append("| cut-off | db | K | best overlap M | T | p rf | p cm | p cml | p cm (uncapped) | E[M] cm | p fake | p fakeK | p okd (n) | rank / n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CUTOFFS:
        for db in ["ChIP", "TRRUST"]:
            r = gres.loc[(c, db)]
            rk = f"{r.rank_cm_valid:g}/{int(r.n_rank_cand)}" if np.isfinite(r.rank_cm_valid) else "–"
            L.append(f"| {c} | {db} | {int(r.K)} | {int(r.M_obs)} | {int(r.T_obs)} | {fmt(r.p_rf)} | {fmt(r.p_cm)} | {fmt(r.p_cml)} | {fmt(r.p_cm_uncapped)} | {fmt(r.E_M_cm, 2)} | {fmt(r.p_fake, 4)} | {fmt(r.p_fakeK)} | {fmt(r.p_okd)} ({int(r.n_okd) if np.isfinite(r.n_okd) else 0}) | {rk} |")
    L.append("\n### Table C. Every primary-TF row with T >= 2\n")
    L.append("| TF | cut-off | db | K | M | T | p rf | p cm | p cml | p cm uncapped | p fake | p fakeK | p okd (n) | rank / n | q cm (BH, 20 TFs) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res[(res.T_obs >= 2) & res.primary_tf].sort_values(["tf", "db", "cutoff"], ascending=[True, True, False]).iterrows():
        L.append(f"| {r.tf} | {r.cutoff} | {r.db} | {int(r.K)} | {int(r.M_obs)} | {int(r.T_obs)} | {fmt(r.p_rf)} | {fmt(r.p_cm)} | {fmt(r.p_cml)} | {fmt(r.p_cm_uncapped)} | {fmt(r.p_fake, 4)} | {fmt(r.p_fakeK)} | {fmt(r.p_okd)} ({int(r.n_okd)}) | {r.rank_cm:g}/{int(r.n_rank_cand)} | {fmt(r.q_cm_primary)} |")
    L.append("\n### Table D. Power (planted k true targets into one random responding feature; mean over TFs; 95% CI = bootstrap over TFs, or Wilson over 200 repeats when one TF)\n")
    L.append("| cut-off | db | k | n TFs | rf | cm | cm uncapped | cml | cml uncapped | fake | fakeK | okd |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in pwr[pwr.k.isin([1, 2, 3, 4, 5, 8])].iterrows():
        def cell(col):
            ci = r.get(col + "_ci")
            return f"{r[col]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if isinstance(ci, list) else (f"{r[col]:.2f}" if np.isfinite(r[col]) else "–")
        L.append(f"| {r.cutoff} | {r.db} | {r.k} | {r.n_tfs} | {cell('power_rf')} | {cell('power_cm')} | {cell('power_cm_uncapped')} | {cell('power_cml')} | {cell('power_cml_uncapped')} | {cell('power_fake')} | {cell('power_fakeK')} | {cell('power_okd')} |")
    L.append("\n### Table E. v2 vs v3, primary TFs (BH features = q < 0.05 at cut-off 0)\n")
    L.append("| TF | kd cells | BH features v2 | v3 | K at 0.5 v2 | v3 | K at 0.1 v2 | v3 | K at 0.01 v2 | v3 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(cmp["primary"], key=lambda x: -x["n_bh_sig_v3"]):
        L.append(f"| {r['tf']} | {r['n_kd_cells_v3']} | {r['n_bh_sig_v2']} | {r['n_bh_sig_v3']} | {r['K_cut_0.5_v2']} | {r['K_cut_0.5_v3']} | {r['K_cut_0.1_v2']} | {r['K_cut_0.1_v3']} | {r['K_cut_0.01_v2']} | {r['K_cut_0.01_v3']} |")
    (OUT / "summary_tables.md").write_text("\n".join(L) + "\n")

    # ---------------- run_config.json (merged) ----------------
    scripts = ["v3_tf_specificity_extract.py", "v3_tf_specificity_taskdata.py", "v3_tf_specificity_stats.py",
               "v3_tf_specificity_calib_check.py", "v3_tf_specificity_verify.py", "v3_tf_specificity_replay_v2.py",
               "v3_tf_specificity_summary.py"]
    h = hashlib.sha256()
    for fpath in sorted(glob.glob(str(OUT / "cells/summ_*.npz"))):
        d = np.load(fpath)
        for r, cs, ts in zip(d["row"], d["counts_sha256"], d["tokens_sha256"]):
            h.update(f"{int(r)}:{cs}:{ts};".encode())
    rc = {"item": "V3-2 TF specificity of v3 layer-5 SAE features, correctly encoded K562 inputs",
          "device_model_runs": rce["device"], "torch": rce["torch"], "transformers": rce["transformers"],
          "python": rce["python"], "platform": rce["platform"], "model_dir": rce["model_dir"], "model_dtype": rce["model_dtype"],
          "attention": rce.get("attention"), "batch_size_forward": rce.get("batch_size_forward"),
          "threads": rce.get("threads"),
          "hooks_v2_sha256": rce["hooks_v2_sha256"], "sae_path": rce["sae_path"], "sae_sha256": rce["sae_sha256"],
          "layer_site": rce["layer_site"], "max_len": rce["max_len"],
          "input_encoding": rce["input_encoding"],
          "input_encoding_check": {**rce["encoding_check_result"], "rows_accepted_unit_one": prog.get("rows_accepted_unit_one"),
                                   "old_v2_tokens_fail_check_on_sample": prep["order_sample"]["old_v2_tokens_fail_check"],
                                   "verify_12_cells": verify["model_side"]["encoding_check"]},
          "inputs_v3_note": ("inputs_v3.counts_accept_unit_one was ADDED to setup/inputs_v3.py by this item (nothing "
                             "existing was changed). The first extraction call ran with reader.counts and stopped on row "
                             "282850 (expm1(X) already whole numbers) before any forward pass of that batch; 400 cells "
                             "were done by then with identical code otherwise."),
          "seeds": {"cells": rce["seeds"], "stats": rcs["seeds"],
                    "calib_check": {"repartition": calx["B_repartition"]["seed"], "fakeK_repro": "SEED+4000+n",
                                    "calibration_D": calx["D_calibration_1000_per_tf"]["seed"]},
                    "gata1_bootstrap": SEED + 2024, "power_ci_bootstrap": SEED + 2025, "order_sample": "20261001+77"},
          "cell_ids": "cell_manifest.csv: dataset row index (0-based, X row of replogle_concat.h5ad) + cell_barcode + gem_group; "
                      "groups ref/pool/kd/catalog; identical to v2_tf_specificity/cell_manifest.csv (checked)",
          "dataset_file": rce["input_encoding"]["file"],
          "input_sha256": {"per_cell_counts_and_tokens_combined": h.hexdigest(),
                           "how": "sha256 over 'row:counts_sha256[:16]:tokens_sha256[:16];' for all 9,200 cells (cells/summ_*.npz)",
                           "v3_sae_codes_layer_05": cchk["inputs_sha256"]},
          "feature_ids": "all 4,928 features of runs/sae-atlas-217M/outputs/v3_sae/layer_05/sae_final.pt",
          "wall_time_per_chunk_extract": rce["chunks"],
          "extract_first_call_note": "first extract call: 400 cells in about 1.6 min, then stopped (EncodingError, see inputs_v3_note); not in chunks list",
          "stats_calls": rcs["log"].get("calls"), "calib_check_runs": calx.get("wall_s_by_run"),
          "verify_wall_s": verify.get("wall_s"), "replay_v2_wall_s": replay.get("wall_s"),
          "taskdata_wall_s": cchk.get("wall_s"), "summary_wall_s": round(time.time() - t0, 1),
          "code_sha256": {s: H.sha256_file(SCRIPTS / s) for s in scripts},
          "shared_code_sha256": {"setup/inputs_v3.py": H.sha256_file(S.PROJ / "setup/inputs_v3.py"),
                                 "setup/hooks_v2.py": H.sha256_file(S.PROJ / "setup/hooks_v2.py"),
                                 "setup/topk_sae.py": H.sha256_file(S.PROJ / "setup/topk_sae.py")},
          "parameters": {"B_fake": rcs["B_fake"], "R_power": rcs["R_power"], "B_cal_per_tf": rcs.get("B_cal_per_tf"),
                         "cutoffs": rcs["cutoffs"], "primary_cutoff": rcs["primary_cutoff"], "q_feature": rcs["q_feature"],
                         "thresholds": rcs["thresholds"], "min_candidate_targets": rcs["min_candidate_targets"],
                         "n_boot_gata1": N_BOOT}}
    H.write_json(OUT / "run_config.json", rc)

    shutil.copyfile(OUT / "cell_manifest.csv", TD / "cell_manifest.csv")
    (TD / "README.md").write_text(TD_README.format(n_genes=len(uni), n_tfs=len(meta["tfs_all"])))
    print("done", round(time.time() - t0), "s", flush=True)


TD_README = """# Task data: TF specificity of MaxToki-217M v3 layer-5 SAE features (item V3-2)

Made by `runs/sae-atlas-217M/scripts/v3_tf_specificity_*.py` (see `../run_config.json`). Same layout as
`../../v2_tf_specificity/task_data/`, but everything comes from the v3 SAE
(`outputs/v3_sae/layer_05/sae_final.pt`, 4,928 TopK features, k = 32, retrained on correctly encoded
inputs) applied to the INPUT of decoder block 5 (= `hidden_states[5]`), and from correctly encoded
cells (counts / Geneformer gene median, `setup/inputs_v3.py`). Feature numbers are v3 numbers; they
do not match deployed or v2 feature numbers. Gene symbols are upper case.

## Definitions

* **Gene universe**: the {n_genes} genes that appear as tokens in the 500 K562 non-targeting control
  cells used to train the v3 SAE (and to build its catalog), with the correct encoding.
* **Detection count**: number of those 500 cells in which the gene is a token (max_len 2,048). Max 500.
* **Top-20 genes of a feature** (deployed definition): over every token position of the 500 catalog
  cells, mean activation per (feature, gene) over positions where the feature is active; the 20 genes
  with the highest mean. No minimum count. `<SPECIAL>` (= `<bos>`/`<eos>`) can enter the list; it is
  never a target. Built from `outputs/v3_sae/codes/layer_05_topk.npz`; rebuilt from our own forward
  passes it is identical for all 4,928 features.
* **Per-cell feature value**: mean activation of the feature over the cell's gene tokens.
* **Responding feature** of a TF: two-sided Mann-Whitney U (full tie correction), knockdown cells vs
  400 reference control cells, per feature; BH across 4,928 features; q < 0.05 and
  |mean(knockdown) - mean(reference)| > cut-off (0.5 deployed; also 0.25, 0.1, 0.05, 0.02, 0.01, 0).
* **Count bins**: detection count digitised at 1, 2, 5, 10, 20, 50, 100, 200, 500. **Length tertile**:
  genomic span from `biotensor/data/genemanifold/gene_pos.json`, cut at the 33.3 / 66.7 percentiles of
  the universe (genes without a length go to the middle tertile). `count_x_length_bin` = 10 x bin + tertile.

## Files

| file | content |
|---|---|
| `gene_universe.tsv` | gene, detection_count, gene_length_bp, chr, count_bin, length_tertile, count_x_length_bin |
| `feature_top20.tsv` | feature_id, rank, gene, mean_act_when_active, n_active_positions, gene_detection_count, in_forward_rebuild_top20 |
| `feature_info.tsv` | per feature: n_top20, n_special_in_top20, median detection count / length of top-20, activation_frequency, n_active_positions, forward_rebuild_same_list |
| `targets_trrust.tsv`, `targets_dorothea.tsv` | targets inside the universe (DoRothEA with confidence and chip_flag = ChIP-seq subset) |
| `cell_manifest.csv` | 9,200 cells: row (0-based X row of replogle_concat.h5ad), group, tf, priority, barcode, gem_group, UMI_count, is_primary_tf (same cells as v2) |
| `cell_feature_means.npz` | rows, mean_gene (9,200 x 4,928 float32), n_tokens |
| `responding_features.tsv` | every (TF, feature) with BH q < 0.05: delta_mean, cohen_d, mwu_p, bh_q, n_kd_cells, resp_cut_<c> |
| `tf_summary.tsv` | {n_tfs} TFs: n_kd_cells, n_bh_sig, K_cut_<c>, target-set sizes, kd_remaining_frac_linear |
| `tf_tests.tsv` | all statistics and p-values, TF x cut-off x database (same as `../tf_results.csv`) |
| `catalog_check.json` | token check, code check (stored vs live codes), catalog rebuild check, universe vs v2 |

Column guide for `tf_tests.tsv`: as in `../../v2_tf_specificity/task_data/README.md` (K, M_obs, T_obs,
p_rf, p_cm, p_cml, *_uncapped, E_M_cm, p_fake, p_fakeK, p_okd, n_okd, rank_cm, n_rank_cand, q_<test>_primary,
q_<test>_all).
"""


if __name__ == "__main__":
    main()
