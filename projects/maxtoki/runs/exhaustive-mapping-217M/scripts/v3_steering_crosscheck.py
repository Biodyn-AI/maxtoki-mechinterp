"""v3 steering (item V3-8): key numbers re-derived a second way (no model, CPU, float64).

Separate code from v3_steering_summarize.py. Writes outputs/v3_steering/crosscheck.json.
  X1 delta_s of every candidate and control row recomputed from the stored float32 steered mean logits
     (keep32) and clean references with scipy's cosine distance; compared with the stored delta_s.
  X2 delta_s of every row recomputed from the stored dot products with the signatures and |m|^2
     (no logit vectors needed); compared with the stored delta_s.
  X3 per-unit means and 95% CIs recomputed with scipy.stats.bootstrap (percentile, its own RNG);
     compared with summary.json (CI ends should agree within Monte Carlo error).
  X4 candidate maturity correlations recomputed with scipy.stats.spearmanr per donor (loop, not the
     vectorised code), and the candidate choice re-derived from them.
  X5 axis AUROC recomputed with sklearn.metrics.roc_auc_score.
  X6 empirical p-values recomputed from null_table.csv / candidates_table.csv.

Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering_crosscheck.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import bootstrap, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_steering as V  # noqa: E402
from v3_steering import H  # noqa: E402


def s_of(m, gL, gE):
    return (1 - cos_dist(m, gL)) - (1 - cos_dist(m, gE))


def main():
    selj = V.load_json(V.OUT / "selection.json")
    summ = V.load_json(V.OUT / "summary.json")
    cells = V.load_cells()
    G = {k: v.astype(np.float64) for k, v in np.load(V.OUT / "signatures.npz").items()}
    gE, gL = G["g_early"], G["g_late"]
    names = V.SIG_NAMES
    iE, iL = names.index("g_early"), names.index("g_late")
    nE, nL = np.linalg.norm(gE), np.linalg.norm(gL)
    done = summ["steered_cells"]
    out = {"n_cells": len(done)}
    x1, x2 = [], []
    per = {l: [] for l in V.LAYERS}
    for i in done:
        z = np.load(V.STEER_DIR / f"cell_{i:03d}.npz")
        for l in V.LAYERS:
            ds = z[f"L{l}_delta_s"]
            per[l].append(ds)
            mref = z[f"L{l}_m_ref"].astype(np.float64)
            s_ref = s_of(mref, gL, gE)
            for j, r in enumerate(z[f"L{l}_keep32_rows"]):
                m = z[f"L{l}_keep32"][j].astype(np.float64)
                x1.append(abs((s_of(m, gL, gE) - s_ref) - ds[r]))
            dots, sq = z[f"L{l}_dots"], z[f"L{l}_sq"]
            rd, rs = z[f"L{l}_ref_dots"], float(z[f"L{l}_ref_sq"])
            s_ref2 = rd[iL] / (np.sqrt(rs) * nL) - rd[iE] / (np.sqrt(rs) * nE)
            s_st2 = dots[:, iL] / (np.sqrt(sq) * nL) - dots[:, iE] / (np.sqrt(sq) * nE)
            x2.extend(np.abs((s_st2 - s_ref2) - ds).tolist())
    out["X1_from_stored_logits"] = {"n_rows": len(x1), "max_abs_diff": float(max(x1)),
                                    "note": "m_ref is stored in float32, so differences up to ~1e-7 are expected",
                                    "pass": float(max(x1)) < 1e-6}
    out["X2_from_dot_products"] = {"n_rows": len(x2), "max_abs_diff": float(max(x2)), "pass": float(max(x2)) < 1e-9}
    # X3 bootstrap with scipy
    rng = np.random.default_rng(12345)
    x3 = []
    for l in V.LAYERS:
        A = np.stack(per[l])
        rows = V.unit_rows(selj, l)
        L = summ["per_layer"][f"L{l}"]
        for c in L["candidates"]:
            for a in V.ALPHAS:
                x = A[:, rows.index(("cand", c["feature"], a))]
                b = bootstrap((x,), np.mean, n_resamples=10000, method="percentile", random_state=rng)
                st = c[f"alpha{a:g}"]
                width = st["ci95"][1] - st["ci95"][0]
                x3.append({"layer": l, "feature": c["feature"], "alpha": a, "mean_summary": st["mean_delta_s"],
                           "mean_here": float(x.mean()),
                           "ci_summary": st["ci95"], "ci_scipy": [float(b.confidence_interval.low),
                                                                  float(b.confidence_interval.high)],
                           "max_end_diff_as_frac_of_width": float(max(abs(b.confidence_interval.low - st["ci95"][0]),
                                                                      abs(b.confidence_interval.high - st["ci95"][1]))
                                                                  / width) if width > 0 else 0.0})
    out["X3_bootstrap_scipy"] = {"n_units": len(x3), "max_mean_diff": max(abs(r["mean_summary"] - r["mean_here"]) for r in x3),
                                 "max_ci_end_diff_frac_of_width": max(r["max_end_diff_as_frac_of_width"] for r in x3),
                                 "pass": max(abs(r["mean_summary"] - r["mean_here"]) for r in x3) < 1e-12
                                 and max(r["max_end_diff_as_frac_of_width"] for r in x3) < 0.15,
                                 "rows": x3}
    # X4 rho with a plain loop
    grp, stage, donor = cells["group"], cells["stage"], cells["donor"]
    sel = np.where(grp == "selection")[0]
    x4 = {}
    for l in V.LAYERS:
        A = np.stack([np.load(V.CLEAN_DIR / f"cell_{i:03d}.npz")[f"meanz_L{l}"] for i in sel]).astype(np.float64)
        hsc = sel[stage[sel] == 0]
        act = np.stack([np.load(V.CLEAN_DIR / f"cell_{i:03d}.npz")[f"nact_L{l}"] for i in hsc]) > 0
        alive = np.where(act.mean(0) >= V.ALIVE_MIN_FRAC)[0]
        rho = {}
        for f in alive:
            vals = []
            for d in V.SEL_DONORS:
                m = donor[sel] == d
                r = spearmanr(A[m, f], stage[sel][m]).statistic
                vals.append(0.0 if not np.isfinite(r) else r)
            rho[int(f)] = float(np.mean(vals))
        top = sorted(rho, key=lambda f: (-abs(rho[f]), f))[:V.N_CAND]
        stored = [c["feature"] for c in selj["layers"][str(l)]["candidates"]]
        x4[f"L{l}"] = {"n_alive": len(alive), "n_alive_stored": selj["layers"][str(l)]["n_alive"],
                       "top3_here": top, "top3_stored": stored,
                       "rho_here": [rho[f] for f in top],
                       "rho_stored": [c["rho"] for c in selj["layers"][str(l)]["candidates"]],
                       "pass": top == stored and len(alive) == selj["layers"][str(l)]["n_alive"]
                       and max(abs(a - b) for a, b in zip([rho[f] for f in top],
                                                          [c["rho"] for c in selj["layers"][str(l)]["candidates"]])) < 1e-9}
    out["X4_candidate_choice"] = {"layers": x4, "pass": all(v["pass"] for v in x4.values())}
    # X5 AUROC
    sig = np.load(V.OUT / "signatures.npz")
    s_all = sig["s_clean_all"]
    sd = np.where(grp != "selection")[0]
    y = stage[sd]
    m = (y == 0) | (y == 2)
    auc = roc_auc_score((y[m] == 2).astype(int), s_all[sd][m])
    out["X5_axis_auroc"] = {"sklearn": float(auc), "stored": selj["axis"]["steering_donors"]["auroc_EryP_vs_HSC"],
                            "pass": abs(auc - selj["axis"]["steering_donors"]["auroc_EryP_vs_HSC"]) < 1e-12}
    # X6 empirical p from the csv tables
    ct = pd.read_csv(V.OUT / "candidates_table.csv")
    nt = pd.read_csv(V.OUT / "null_table.csv")
    diffs = []
    for _, r in ct.iterrows():
        nv = nt[nt.layer == r.layer]
        for a in V.ALPHAS:
            pred = np.sign(r.rho) * (1 if a > 1 else -1)
            p = (1 + np.sum(pred * nv[f"ds_a{a:g}"].to_numpy() >= pred * r[f"ds_a{a:g}"])) / 21
            diffs.append(abs(p - r[f"p_dir_a{a:g}"]))
    out["X6_empirical_p"] = {"n": len(diffs), "max_abs_diff": float(max(diffs)), "pass": float(max(diffs)) < 1e-12}
    out["all_pass"] = all(v["pass"] for k, v in out.items() if isinstance(v, dict) and "pass" in v)
    H.write_json(V.OUT / "crosscheck.json", out)
    V.log({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in out.items()})
    V.log({k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "layers")} for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
