"""D3 (revision): summary numbers, uncertainty, selection stability, run_config and task-data README.

Reads the outputs of v2_tf_specificity_{extract,taskdata,stats,calib_check}.py and writes
  outputs/v2_tf_specificity/summary.json
  outputs/v2_tf_specificity/summary_tables.md     (tables used in V2_TF_SPECIFICITY_REPORT.md)
  outputs/v2_tf_specificity/gata1_bootstrap.json  (cell-bootstrap stability of the GATA1 selection)
  outputs/v2_tf_specificity/run_config.json       (merged provenance for the whole folder)
  outputs/v2_tf_specificity/task_data/README.md, task_data/cell_manifest.csv
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
sys.path.insert(0, str(Path(__file__).parent))
import hooks_v2 as H  # noqa: E402
from v2_tf_specificity_stats import RefMW, bh, cap_T, CUTOFFS, Q_FEAT, SEED, D_SAE  # noqa: E402

OUT = PROJ / "runs/sae-atlas-217M/outputs/v2_tf_specificity"
TD = OUT / "task_data"
SCRIPTS = Path(__file__).parent
N_BOOT = 1000


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cp(k, n):
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def main():
    t0 = time.time()
    res = pd.read_csv(OUT / "tf_results.csv")
    pw = pd.read_csv(OUT / "power.csv")
    cal = json.load(open(OUT / "calibration.json"))
    calx = json.load(open(OUT / "calibration_extra.json"))
    rcs = json.load(open(OUT / "run_config_stats.json"))
    rce = json.load(open(OUT / "run_config_extract.json"))
    cchk = json.load(open(TD / "catalog_check.json"))
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    kdeff = json.load(open(OUT / "knockdown_efficiency.json"))
    old = pd.read_csv(OUT / "old_trrust_test.csv")
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    primary = meta["tfs_primary"]
    # rank is only meaningful when the true TF passes a threshold (T >= 2)
    res["rank_cm_valid"] = np.where(res.T_obs >= 2, res.rank_cm, np.nan)
    res["rank_frac_valid"] = np.where(res.T_obs >= 2, res.rank_frac, np.nan)
    B = int(rcs["B_fake"])
    S = {}

    # ---------------- GATA1 bootstrap over cells ----------------
    z = np.load(TD / "cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]
    R = MG[[rix[int(r)] for r in man[man.group == "ref"].row]]
    A = MG[[rix[int(r)] for r in man[(man.group == "kd") & (man.tf == "GATA1")].row]]
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    chip = set(dor[(dor.tf == "GATA1") & (dor.chip_flag.astype(str).str.lower() == "true")].target)
    ovc = np.zeros(D_SAE, int)
    for f, g in top.groupby("feature_id"):
        ovc[f] = int(g.gene.isin(chip).sum())
    feats = [1334, 2006, 3167, 2627, 628, 2610]
    rb = np.random.default_rng(SEED + 2024)
    gpath = OUT / "gata1_bootstrap.json"
    cached = json.loads(gpath.read_text()) if gpath.exists() else None
    use_cache = bool(cached and cached.get("n_boot") == N_BOOT and cached.get("seed") == SEED + 2024)
    sel05 = np.zeros(D_SAE); sel025 = np.zeros(D_SAE)
    K05, T05, M025 = [], [], []
    dboot = np.zeros((N_BOOT, len(feats)))
    for b in range(0 if use_cache else N_BOOT):
        Ab = A[rb.integers(0, len(A), len(A))]; Rb = R[rb.integers(0, len(R), len(R))]
        m = RefMW(Rb)
        p = m.test(m.contrib(Ab).sum(0), len(Ab), (Ab == 0).sum(0)); q = bh(p)
        dl = Ab.mean(0) - m.mean_ref
        s05 = (q < Q_FEAT) & (np.abs(dl) > 0.5); s025 = (q < Q_FEAT) & (np.abs(dl) > 0.25)
        sel05 += s05; sel025 += s025
        K05.append(int(s05.sum()))
        T05.append(int(cap_T(ovc[s05].max())) if s05.any() else 0)
        M025.append(int(ovc[s025].max()) if s025.any() else 0)
        dboot[b] = dl[feats]
    dl_obs = A.mean(0) - R.mean(0)
    gb = cached if use_cache else {"n_boot": N_BOOT, "seed": SEED + 2024, "resampling": "cells, KD and reference resampled separately with replacement",
          "delta": {int(f): {"obs": float(dl_obs[f]), "ci95": [float(np.percentile(dboot[:, j], 2.5)), float(np.percentile(dboot[:, j], 97.5))],
                              "frac_boot_selected_cut0.5": float(sel05[f] / N_BOOT)} for j, f in enumerate(feats)},
          "K_cut0.5_quantiles": [float(x) for x in np.percentile(K05, [2.5, 50, 97.5])],
          "T_chip_cut0.5_counts": {int(k): int(v) for k, v in zip(*np.unique(T05, return_counts=True))},
          "M_chip_cut0.25_counts": {int(k): int(v) for k, v in zip(*np.unique(M025, return_counts=True))},
          "features_selected_in_ge_50pct_cut0.5": [int(f) for f in np.where(sel05 / N_BOOT >= 0.5)[0]]}
    if not use_cache:
        H.write_json(gpath, gb)
    S["gata1_bootstrap"] = gb
    print("bootstrap done", round(time.time() - t0), flush=True)

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
                                 "n_p_fakeK_lt05": int((d.p_fakeK < 0.05).sum()), "n_p_rf_lt05": int((d.p_rf < 0.05).sum()),
                                 "n_rank_top5pct": int((d.rank_frac_valid <= 0.05).sum())})
    counts = pd.DataFrame(tab_rows)
    counts.to_csv(OUT / "summary_counts.csv", index=False)

    # ---------------- power with CIs ----------------
    prow = []
    for (c, db, k), d in pw.groupby(["cutoff", "db", "k"]):
        row = {"cutoff": c, "db": db, "k": k, "n_tfs": int(d.tf.nunique())}
        for col in ["power_rf", "power_cm", "power_cm_uncapped", "power_cml", "power_cml_uncapped", "power_fake", "power_fakeK", "power_okd"]:
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

    # ---------------- key numbers ----------------
    g = res[res.tf == "GATA1"].set_index(["cutoff", "db"])
    S["cells"] = {"ref": int((man.group == "ref").sum()), "pool": int((man.group == "pool").sum()),
                  "catalog": int((man.group == "catalog").sum()), "kd": int((man.group == "kd").sum()),
                  "tfs_all": len(meta["tfs_all"]), "tfs_primary": len(primary)}
    S["checks"] = {
        "token_check": cchk["token_check"], "catalog_check": cchk["catalog_check"],
        "fast_mw_vs_scipy": rcs["log"]["fast_mw_check"], "swap_null_validation": rcs["log"]["swap_null_validation"],
        "repartition": calx["B_repartition"], "pool_vs_ref": calx["A_pool_vs_ref"],
        "fakeK_reproduction": calx["C_fakeK_reproduction_cut0.25"],
        "calibration_1000_per_tf": calx["D_calibration_1000_per_tf"],
        "calibration_250_per_tf_main_run": {k: v for k, v in cal.items() if k.startswith("cutoff_")},
        "feature_level_fixed_pool": cal["feature_level_MW_p_fake_TFs"]}
    ks = [v["KS_p_randomized_pooled"] for k, v in calx["D_calibration_1000_per_tf"].items() if isinstance(v, dict)]
    S["checks"]["calibration_summary"] = {"n_KS_tests": len(ks), "min_KS_p": float(min(ks)), "median_KS_p": float(np.median(ks)),
                                          "max_frac_conservative_p_le_0.05": float(max(v["frac_p_conservative_le_0.05"] for v in calx["D_calibration_1000_per_tf"].values() if isinstance(v, dict)))}
    S["gata1"] = {f"{c}|{db}": {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
                                for k, v in g.loc[(c, db)][["K", "M_obs", "T_obs", "best_feature", "p_rf", "p_cm", "p_cml", "p_cm_uncapped",
                                                            "p_cml_uncapped", "E_M_cm", "p_fake", "p_fakeK", "p_okd", "n_okd",
                                                            "rank_cm_valid", "n_rank_cand", "q_fake_primary", "q_cm_primary"]].items()}
                  for c in CUTOFFS for db in ["ChIP", "TRRUST"]}
    S["p_fake_floor"] = {"B": B, "min_p": 1 / (B + 1), "cp95_for_0_exceedances": cp(0, B)}
    S["old_trrust_test"] = old.to_dict("records")
    S["power_primary_cutoff"] = pwr[pwr.cutoff == 0.5].to_dict("records")
    S["knockdown_efficiency"] = kdeff
    H.write_json(OUT / "summary.json", S)

    # ---------------- markdown tables ----------------
    L = []
    L.append("### Table A. Number of TFs per cut-off (primary set = 20 TFs with DoRothEA ChIP-seq targets)\n")
    L.append("| cut-off | db | TFs with set | testable (K>0) | median K | T>=2 | BH q<0.05 fake | BH q<0.05 cm | BH q<0.05 cml | BH q<0.05 fakeK | BH q<0.05 okd | p<0.05 cm (uncapped) | true TF in top 5% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in counts[counts.scope == "primary20"].iterrows():
        L.append(f"| {r.cutoff} | {r.db} | {r.n_tfs_with_set} | {r.n_testable} | {fmt(r.median_K_testable,0)} | {r.n_T_ge2} | {r.n_q_fake_lt05} | {r.n_q_cm_lt05} | {r.n_q_cml_lt05} | {r.n_q_fakeK_lt05} | {r.n_q_okd_lt05} | {r.n_p_cm_uncapped_lt05} | {r.n_rank_top5pct} |")
    L.append("\n### Table A2. Same, all 87 TFs\n")
    L.append("| cut-off | db | TFs with set | testable | median K | T>=2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | p<0.05 cm uncapped | top 5% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in counts[counts.scope == "all87"].iterrows():
        L.append(f"| {r.cutoff} | {r.db} | {r.n_tfs_with_set} | {r.n_testable} | {fmt(r.median_K_testable,0)} | {r.n_T_ge2} | {r.n_q_fake_lt05} | {r.n_q_cm_lt05} | {r.n_q_cml_lt05} | {r.n_q_fakeK_lt05} | {r.n_q_okd_lt05} | {r.n_p_cm_uncapped_lt05} | {r.n_rank_top5pct} |")
    L.append("\n### Table B. GATA1, all tests\n")
    L.append("| cut-off | db | K | best overlap M | T | p rf | p cm | p cml | p cm (uncapped) | E[M] cm | p fake | p fakeK | p okd (n) | rank / n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CUTOFFS:
        for db in ["ChIP", "TRRUST"]:
            r = g.loc[(c, db)]
            rk = f"{int(r.rank_cm_valid)}/{int(r.n_rank_cand)}" if np.isfinite(r.rank_cm_valid) else "–"
            L.append(f"| {c} | {db} | {int(r.K)} | {int(r.M_obs)} | {int(r.T_obs)} | {fmt(r.p_rf)} | {fmt(r.p_cm)} | {fmt(r.p_cml)} | {fmt(r.p_cm_uncapped)} | {fmt(r.E_M_cm,2)} | {fmt(r.p_fake,4)} | {fmt(r.p_fakeK)} | {fmt(r.p_okd)} ({int(r.n_okd) if np.isfinite(r.n_okd) else 0}) | {rk} |")
    L.append("\n### Table C. Every primary-TF row with T >= 2 (at least one responding feature has >= 2 targets)\n")
    L.append("| TF | cut-off | db | K | M | T | p rf | p cm | p cml | p cm uncapped | p fake | p fakeK | p okd (n) | rank / n | q cm (BH, 20 TFs) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res[(res.T_obs >= 2) & res.primary_tf].sort_values(["tf", "db", "cutoff"], ascending=[True, True, False]).iterrows():
        L.append(f"| {r.tf} | {r.cutoff} | {r.db} | {int(r.K)} | {int(r.M_obs)} | {int(r.T_obs)} | {fmt(r.p_rf)} | {fmt(r.p_cm)} | {fmt(r.p_cml)} | {fmt(r.p_cm_uncapped)} | {fmt(r.p_fake,4)} | {fmt(r.p_fakeK)} | {fmt(r.p_okd)} ({int(r.n_okd)}) | {int(r.rank_cm)}/{int(r.n_rank_cand)} | {fmt(r.q_cm_primary)} |")
    L.append("\n### Table D. Power (planted k true targets into one random responding feature; mean over TFs; 95% CI = bootstrap over TFs, or Wilson over 200 repeats when one TF)\n")
    L.append("| cut-off | db | k | n TFs | rf | cm | cm uncapped | cml | cml uncapped | fake | fakeK | okd |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in pwr[pwr.k.isin([1, 2, 3, 4, 5, 8])].iterrows():
        def cell(col):
            ci = r.get(col + "_ci")
            return f"{r[col]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if isinstance(ci, list) else f"{r[col]:.2f}"
        L.append(f"| {r.cutoff} | {r.db} | {r.k} | {r.n_tfs} | {cell('power_rf')} | {cell('power_cm')} | {cell('power_cm_uncapped')} | {cell('power_cml')} | {cell('power_cml_uncapped')} | {cell('power_fake')} | {cell('power_fakeK')} | {cell('power_okd')} |")
    (OUT / "summary_tables.md").write_text("\n".join(L) + "\n")

    # ---------------- run_config.json (merged) ----------------
    scripts = ["v2_tf_specificity_extract.py", "v2_tf_specificity_taskdata.py", "v2_tf_specificity_stats.py",
               "v2_tf_specificity_calib_check.py", "v2_tf_specificity_summary.py"]
    rc = {"item": "D3 TF specificity of layer-5 SAE features",
          "device_model_runs": rce["device"], "torch": rce["torch"], "transformers": rce["transformers"],
          "python": rce["python"], "platform": rce["platform"], "model_dir": rce["model_dir"], "model_dtype": rce["model_dtype"],
          "hooks_v2_sha256": rce["hooks_v2_sha256"], "sae_path": rce["sae_path"], "sae_sha256": rce["sae_sha256"],
          "layer_site": rce["layer_site"], "max_len": rce["max_len"],
          "seeds": {"cells": rce["seeds"], "stats": rcs["seeds"], "calib_check": {"repartition": SEED + 31, "fakeK_repro": "SEED+4000+n",
                    "calibration_D": SEED + 8888}, "gata1_bootstrap": SEED + 2024},
          "seed_note": ("run_config_stats.json lists 'calibration': SEED+99; that seed is not used. The calibration "
                        "in v2_tf_specificity_stats.py draws with default_rng(SEED + 555) = 20261558."),
          "cell_ids": "cell_manifest.csv: dataset row index (0-based, h5ad X row) + cell_barcode + gem_group; groups ref/pool/kd/catalog",
          "dataset_path": rce["dataset_path"],
          "feature_ids": "all 4,928 features of runs/sae-atlas-217M/outputs/phase1/layer_05/sae_final.pt",
          "wall_time_per_chunk_extract": rce["chunks"],
          "wall_time_stats_s": rcs["log"].get("wall_s"), "wall_time_summary_s": round(time.time() - t0, 1),
          "calib_check_runs": calx.get("wall_s_by_run"),
          "code_sha256": {s: H.sha256_file(SCRIPTS / s) for s in scripts},
          "parameters": {"B_fake": rcs["B_fake"], "R_power": rcs["R_power"], "B_cal_per_tf": rcs.get("B_cal_per_tf"),
                         "cutoffs": rcs["cutoffs"], "primary_cutoff": rcs["primary_cutoff"], "q_feature": rcs["q_feature"],
                         "thresholds": rcs["thresholds"], "min_candidate_targets": rcs["min_candidate_targets"], "n_boot_gata1": N_BOOT}}
    H.write_json(OUT / "run_config.json", rc)

    # ---------------- task data README + manifest copy ----------------
    shutil.copyfile(OUT / "cell_manifest.csv", TD / "cell_manifest.csv")
    readme = TD_README.format(n_ref=S["cells"]["ref"], n_pool=S["cells"]["pool"], n_kd=S["cells"]["kd"],
                              n_tfs=len(meta["tfs_all"]))
    (TD / "README.md").write_text(readme)
    print("done", round(time.time() - t0), "s", flush=True)


TD_README = """# Task data: TF specificity of MaxToki-217M layer-5 SAE features (revision item D3)

Made by `runs/sae-atlas-217M/scripts/v2_tf_specificity_*.py` (see `../run_config.json`).
Everything here is for MaxToki-217M, layer-5 SAE (`outputs/phase1/layer_05/sae_final.pt`, 4,928
TopK features, k = 32), applied to the INPUT of decoder block 5 (= `hidden_states[5]`), which is
the tensor the SAE was trained on. Gene symbols are upper case.

## Definitions

* **Gene universe**: the 6,324 genes that appear as tokens in the 500 K562 non-targeting control
  cells used to build the deployed catalog (full_12layer_pipeline.py, seed 42).
* **Detection count** of a gene: number of those 500 cells in which the gene is one of the cell's
  tokens (rank-value encoding, max_len 2,048; a gene appears at most once per cell). Maximum 500.
* **Top-20 genes of a feature** (deployed definition, kept unchanged): encode every token position
  of the 500 catalog cells; for each (feature, gene) take the mean activation over the positions
  where the feature is active (> 0) and the token is that gene; the 20 genes with the highest mean.
  No minimum count, so a gene seen once can enter. `<SPECIAL>` (= `<bos>`/`<eos>`) can enter the
  list (59 features); it is never a target. We rebuilt this catalog from our own forward passes:
  4,927 of 4,928 lists are identical (same genes, same order); one differs by one gene.
* **Per-cell feature value**: mean activation of the feature over the cell's gene tokens (all
  positions except `<bos>` and `<eos>`).
* **Responding feature** of a TF: two-sided Mann-Whitney U test, knockdown cells vs {n_ref}
  reference control cells, per feature; BH across the 4,928 features; q < 0.05 and
  |mean(knockdown) - mean(reference)| > cut-off. Deployed cut-off = 0.5; also 0.25, 0.1, 0.05, 0.02,
  0.01, 0.
* **Count bins** (for matched nulls): detection count digitised at 1, 2, 5, 10, 20, 50, 100, 200,
  500 (bins 1..9). **Length tertile**: genomic span (end - start, from
  `biotensor/data/genemanifold/gene_pos.json`) cut at 21,375 and 56,220 bp (0, 1, 2); genes without
  a length (54) are put in tertile 1. `count_x_length_bin` = 10 x count bin + tertile.

## Files

| file | rows | columns |
|---|---|---|
| `gene_universe.tsv` | 6,324 genes | gene, detection_count, gene_length_bp (NaN if unknown), chr, count_bin, length_tertile, count_x_length_bin |
| `feature_top20.tsv` | 97,492 (feature, rank) rows (4,807 features have 20 genes, 121 have fewer) | feature_id, rank (1..20), gene (deployed list), mean_act_when_active_rebuilt (our rebuild), n_active_positions_rebuilt (positions where the feature is active on this gene), gene_detection_count, in_rebuilt_top20 |
| `feature_info.tsv` | 4,928 features | n_top20, n_special_in_top20, median detection count and median length (kb) of the top-20 genes, deployed activation frequency, active positions in the rebuild, rebuilt list = deployed list |
| `targets_trrust.tsv` | TRRUST edges with target in universe | tf, target, mode (Activation/Repression/Unknown, ';'-joined), n_pmid |
| `targets_dorothea.tsv` | DoRothEA edges with target in universe | tf, target, confidence (A..D; a few 'B;D' etc.), chip_flag (edge is in the DoRothEA ChIP-seq subset = the "ChIP" sets) |
| `cell_manifest.csv` | 9,200 cells | row (0-based row of `X` in replogle_concat.h5ad), group (ref / pool / kd / catalog), tf (for kd), priority, barcode, gem_group, UMI_count, is_primary_tf |
| `cell_feature_means.npz` | 9,200 cells | rows (dataset row), mean_gene (cells x 4,928 per-cell feature values, float32), n_tokens |
| `responding_features.tsv` | every (TF, feature) with BH q < 0.05 | tf, feature_id, delta_mean (knockdown - reference, signed), cohen_d, mwu_p, bh_q, n_kd_cells, resp_cut_<c> (passes the effect cut-off c) |
| `tf_summary.tsv` | {n_tfs} TFs | n_kd_cells, n_bh_sig, K_cut_<c> (number of responding features), target-set sizes in universe, kd_remaining_frac_linear (mean expm1 expression of the TF gene in knockdown / reference cells; empty if the TF gene is not in the 6,546-gene panel) |
| `tf_tests.tsv` | TF x cut-off x database | all statistics and p-values (same as `../tf_results.csv`; column guide below) |
| `catalog_check.json` | – | token check and catalog rebuild check |

Cell groups: `ref` = {n_ref} reference controls; `pool` = {n_pool} held-out controls used for fake TFs;
`kd` = {n_kd} knockdown cells (up to 100 per TF, {n_tfs} TFs); `catalog` = the 500 catalog cells.
`ref`, `pool` and `catalog` are disjoint random K562 non-targeting cells.

### Column guide for `tf_tests.tsv`
* K = number of responding features; M_obs = best overlap of a responding feature's top-20 with the
  target set; T_obs = min(M, 5) if M >= 2 else 0 (the deployed thresholds 2..5 as one statistic).
* p_rf: deployed random-feature null (K random catalog features), exact. p_cm / p_cml: gene-swap
  null matched on detection count / on count and length, exact (hypergeometric), with threshold
  search over 2..5; *_uncapped: same with the uncapped M. E_M_cm: expected M under p_cm's null.
* p_fake: selection-aware null, 2,000 fake TFs (random pool subsets of the knockdown group's size,
  whole selection re-run). p_fakeK: fake TFs keep their K most-changed features. p_okd: responding
  features of the other knockdowns (n_okd of them with K > 0); okd_min_attainable_p = 1/(1+n_okd).
* rank_cm: rank of the true TF among all TFs with a set (ChIP: >= 20 targets in universe, 291 sets;
  TRRUST: >= 5, ~109 sets) by p_cm with the same features; n_rank_cand; only meaningful when T_obs >= 2.
* q_<test>_primary / q_<test>_all: BH across the 20 primary TFs / all {n_tfs} TFs that have a target
  set in that database (TFs with K = 0 enter with p = 1).
"""


if __name__ == "__main__":
    main()
