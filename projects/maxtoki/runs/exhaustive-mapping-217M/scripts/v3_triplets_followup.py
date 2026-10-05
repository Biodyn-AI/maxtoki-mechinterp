"""v3 triplets follow-up (item V3-4): checks on top of outputs/v3_triplets. No model is run.

Stage `countnull`:
  The aggregate sign-flip test gave 44 of 512 triplets with p < 0.05 (25.6 expected). The triplets
  share cells and conditions, so their p-values are not independent and a binomial count is the wrong
  null. Here the SAME 999 sign-flip vectors (outputs/v3_triplets/signflip_signs.npy) are applied to
  every triplet at once. For each flip s, the flipped statistic of every triplet is ranked inside that
  triplet's own 1,000 values (observed + 999 flips), giving a "pseudo p-value" per triplet; the number
  of triplets with pseudo p < 0.05 is the null count for flip s. This keeps the dependence between
  triplets. Done for I_ABC (512 triplets), all interaction orders (d_ABC - d_A - d_B - d_C) and the
  192 pairwise terms. Also tests whether the hits concentrate on one feature (max over the 24 features
  of the number of its 64 triplets with p < 0.05, against the same null).
  It also records the resolution floor: with 999 flips the smallest p is 0.001, so BH over 512
  triplets can only reject if at least 11 triplets sit at that floor.

Stage `pairs`:
  Pairwise terms I_XY = d_XY - d_X - d_Y for all 192 pairs: noise-corrected energy share of I_XY in the
  joint effect d_XY and rho2 = E[d_XY] / E[d_X + d_Y] with basic bootstrap CIs (same 1,000 cell-weight
  vectors as analyze), per-target BH hits and their size, and two mechanism read-outs for the
  downstream (later-layer) feature Y: its live number of active positions with and without the
  upstream edit, and the cosine between the mean I_XY and Y's own single effect e_Y (a value near -1
  means: removing X leaves less of Y to remove). Also the 4 triplet-level per-target hits.

Stage `replay`:
  Runs the v3 analyze and power code on the stored v2 grid (outputs/v2_triplets) and writes the result
  to outputs/v3_triplets_followup/v2_replay/. Then compares every number with the v2 summary.json and
  v2_triplets_followup/align.json + classnull.json. If they agree, the v2 -> v3 changes come from the
  data (inputs and SAEs), not from changes in the analysis code. Resumable (repeat until DONE).

Usage (from projects/maxtoki):
  OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_triplets_followup.py <stage>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
SRC = RUN / "outputs/v3_triplets"
OUT = RUN / "outputs/v3_triplets_followup"
V2 = RUN / "outputs/v2_triplets"
V2F = RUN / "outputs/v2_triplets_followup"
SCRIPT_PATH = Path(__file__).resolve()
MAIN_SCRIPT = RUN / "scripts/v3_triplets.py"
NONE = -1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def load_main(name="v3_triplets_main"):
    spec = importlib.util.spec_from_file_location(name, MAIN_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def record(stage, t0, extra=None):
    p = OUT / "run_config.json"
    cfg = load_json(p) if p.exists() else {}
    cfg["item"] = "V3-4 follow-up: count null, pairwise mechanism, v2 replay of the analysis code"
    cfg["script"] = str(SCRIPT_PATH)
    cfg["env"] = H.env_info("cpu")
    cfg["inputs"] = {"v3_triplets_dir": str(SRC), "v2_triplets_dir": str(V2),
                     "main_script": str(MAIN_SCRIPT), "main_script_sha256": H.sha256_file(MAIN_SCRIPT)}
    cfg["seeds"] = {"signflip": "outputs/v3_triplets/signflip_signs.npy (seed 20261002, 999 x 20)",
                    "bootstrap": "outputs/v3_triplets/bootstrap_cell_weights.npy (seed 20261001, 1000 x 20)"}
    cfg.setdefault("chunks", []).append(dict({"stage": stage, "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                              "wall_seconds": round(time.time() - t0, 1),
                                              "script_sha256": H.sha256_file(SCRIPT_PATH)}, **(extra or {})))
    H.write_json(p, cfg)


def get_terms(D, cidx, a, b, c):
    keys = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
            "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
    return {k: D[:, cidx[v]].astype(np.float64) for k, v in keys.items()}


# =============================================================================
# countnull
# =============================================================================
def pseudo_counts(Tall, alpha=0.05):
    """Tall: (n_units, 1 + n_flip); column 0 = observed. Returns (counts per column, p per unit per column)."""
    n_u, n_c = Tall.shape
    P = np.empty((n_u, n_c))
    for u in range(n_u):
        row = Tall[u]
        srt = np.sort(row)
        # number of values >= row[s] (including itself)
        P[u] = (n_c - np.searchsorted(srt, row, side="left")) / n_c
    return (P < alpha).sum(0), P


def stage_countnull(args):
    t0 = time.time()
    m = load_main()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = m.build_design(sel)
    D, _, _, _, cidx = m.load_grid(nC, conds, a_groups, with_logits=False)
    Sf = np.load(SRC / "signflip_signs.npy").astype(np.float64) / nC
    pt = load_json(SRC / "per_triplet.json")
    real = [t for t in triplets if t["kind"] == "real"]
    rows_real = [r for r in pt if r["kind"] == "real"]
    T_I, T_tot = [], []
    for t in real:
        d = get_terms(D, cidx, t["a"], t["b"], t["c"])
        Iv = d["ABC"] - d["AB"] - d["AC"] - d["BC"] + d["A"] + d["B"] + d["C"]
        Xt = d["ABC"] - d["A"] - d["B"] - d["C"]
        for X, store in ((Iv, T_I), (Xt, T_tot)):
            obs = np.abs(X.mean(0)).sum()
            nul = np.abs(Sf @ X).sum(1)
            store.append(np.concatenate([[obs], nul]))
    T_I, T_tot = np.array(T_I), np.array(T_tot)
    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    T_P, pair_keys = [], []
    for (lx, ly, fx, fy, nm) in ((0, 1, A, B, "AB"), (0, 2, A, C, "AC"), (1, 2, B, C, "BC")):
        for u in fx:
            for v in fy:
                kx = [NONE] * 3
                kx[lx] = u
                ky = [NONE] * 3
                ky[ly] = v
                kxy = list(kx)
                kxy[ly] = v
                dX, dY, dXY = (D[:, cidx[tuple(k)]].astype(np.float64) for k in (kx, ky, kxy))
                I2 = dXY - dX - dY
                T_P.append(np.concatenate([[np.abs(I2.mean(0)).sum()], np.abs(Sf @ I2).sum(1)]))
                pair_keys.append((nm, int(u), int(v)))
    T_P = np.array(T_P)
    res = {"n_flips": int(Sf.shape[0]), "alpha": 0.05,
           "method": ("same sign-flip vector applied to all units at once; each flip's statistic ranked inside "
                      "its unit's own 1,000 values; null count = units with pseudo p < 0.05 per flip; "
                      "p_count = (1 + #{flips with count >= observed}) / 1000")}
    feats = [("L0", f, 0) for f in A] + [("L5", f, 1) for f in B] + [("L9", f, 2) for f in C]
    for name, Tall, units in (("I_ABC", T_I, [(t["a"], t["b"], t["c"]) for t in real]),
                              ("all_orders", T_tot, [(t["a"], t["b"], t["c"]) for t in real]),
                              ("pairs", T_P, pair_keys)):
        counts, P = pseudo_counts(Tall)
        obs, nul = int(counts[0]), counts[1:]
        r = {"n_units": int(Tall.shape[0]), "observed_count_p05": obs,
             "binomial_expectation": 0.05 * Tall.shape[0],
             "null_count_mean": float(nul.mean()), "null_count_sd": float(nul.std(ddof=1)),
             "null_count_p95": float(np.percentile(nul, 95)), "null_count_max": int(nul.max()),
             "p_count": float((1 + (nul >= obs).sum()) / (len(nul) + 1)),
             "observed_count_p001_floor": int((P[:, 0] <= 0.001 + 1e-12).sum()),
             "smallest_possible_p": 1.0 / Tall.shape[1],
             "smallest_possible_BH_q_single_unit": float(min(1.0, Tall.shape[0] / Tall.shape[1])),
             "units_needed_at_p_floor_for_any_BH_rejection": int(np.ceil(Tall.shape[0] / Tall.shape[1] / 0.05))}
        if name == "I_ABC":
            stored = np.array([q["agg_p_signflip"] for q in rows_real])
            r["max_abs_diff_p_vs_analyze"] = float(np.abs(P[:, 0] - stored).max())
            # concentration on one feature
            per_feat_obs, per_feat_null = {}, []
            sig = P < 0.05
            for lname, f, pos in feats:
                mask = np.array([u[pos] == f for u in units])
                cnt = sig[mask].sum(0)
                per_feat_obs[f"{lname}_{f}"] = int(cnt[0])
                per_feat_null.append(cnt[1:])
            per_feat_null = np.array(per_feat_null)
            max_null = per_feat_null.max(0)
            mx_obs = max(per_feat_obs.values())
            r["per_feature_count_observed"] = per_feat_obs
            r["max_over_features_observed"] = mx_obs
            r["max_over_features_null_mean"] = float(max_null.mean())
            r["max_over_features_null_p95"] = float(np.percentile(max_null, 95))
            r["p_max_over_features"] = float((1 + (max_null >= mx_obs).sum()) / (len(max_null) + 1))
            r["per_feature_p(count >= observed, same flips)"] = {
                k: float((1 + (per_feat_null[i] >= v).sum()) / (per_feat_null.shape[1] + 1))
                for i, (k, v) in enumerate(per_feat_obs.items())}
        if name == "pairs":
            stored = {(q["pair"], q["x"], q["y"]): q["agg_p_signflip"] for q in load_json(SRC / "per_pair.json")}
            r["max_abs_diff_p_vs_analyze"] = float(max(abs(P[i, 0] - stored[k]) for i, k in enumerate(units)))
            r["by_type_observed"] = {nm: int(sum(1 for i, k in enumerate(units) if k[0] == nm and P[i, 0] < 0.05))
                                     for nm in ("AB", "AC", "BC")}
        res[name] = r
        log(f"{name}: observed {obs}, null mean {nul.mean():.1f} (sd {nul.std(ddof=1):.1f}, 95% {np.percentile(nul, 95):.0f}), "
            f"p_count {r['p_count']:.3f}")
    res["wall_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    H.write_json(OUT / "countnull.json", res)
    record("countnull", t0)
    log(json.dumps(res, indent=1)[:4000])


# =============================================================================
# pairs (+ the 4 triplet hits)
# =============================================================================
def stage_pairs(args):
    t0 = time.time()
    m = load_main()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = m.build_design(sel)
    D, _, _, NA, cidx = m.load_grid(nC, conds, a_groups, with_logits=False)
    Wb = np.load(SRC / "bootstrap_cell_weights.npy").astype(np.float64)

    def energy(X):
        return float((X.mean(0) ** 2).sum() - X.var(0, ddof=1).sum() / X.shape[0])

    def plugin(X):
        return float((X.mean(0) ** 2).sum())

    def energy_boot(X):
        n = X.shape[0]
        mu = Wb @ X
        var = np.clip(Wb @ (X * X) - mu * mu, 0, None) * n / (n - 1)
        return (mu * mu).sum(1) - var.sum(1) / n

    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    rows = []
    for (lx, ly, fx, fy, nm) in ((0, 1, A, B, "AB"), (0, 2, A, C, "AC"), (1, 2, B, C, "BC")):
        for u in fx:
            for v in fy:
                kx = [NONE] * 3
                kx[lx] = u
                ky = [NONE] * 3
                ky[ly] = v
                kxy = list(kx)
                kxy[ly] = v
                jx, jy, jxy = cidx[tuple(kx)], cidx[tuple(ky)], cidx[tuple(kxy)]
                dX, dY, dXY = (D[:, j].astype(np.float64) for j in (jx, jy, jxy))
                I2 = dXY - dX - dY
                add = dX + dY
                share = energy(I2) / max(energy(dXY), 1e-30)
                share_pl = plugin(I2) / max(plugin(dXY), 1e-30)
                b_share = energy_boot(I2) / np.maximum(energy_boot(dXY), 1e-30)
                rho2 = energy(dXY) / max(energy(add), 1e-30)
                rho2_pl = plugin(dXY) / max(plugin(add), 1e-30)
                b_rho2 = energy_boot(dXY) / np.maximum(energy_boot(add), 1e-30)
                q = m.bh(m.ttest_p(I2))
                hit = q < 0.05
                I2m, eXY, eY, eX = I2.mean(0), dXY.mean(0), dY.mean(0), dX.mean(0)
                cos = float(I2m @ eY / max(np.linalg.norm(I2m) * np.linalg.norm(eY), 1e-30))
                cosX = float(I2m @ eX / max(np.linalg.norm(I2m) * np.linalg.norm(eX), 1e-30))
                # live active positions of the downstream feature Y, with and without the upstream edit
                nY_alone = NA[:, jy, ly].astype(float)
                nY_after = NA[:, jxy, ly].astype(float)
                rows.append({
                    "pair": nm, "x": int(u), "y": int(v),
                    "energy_share_I2": share, "energy_share_I2_ci95_basic": m.ci_basic(share, b_share, share_pl),
                    "rho2": rho2, "rho2_ci95_basic": m.ci_basic(rho2, b_rho2, rho2_pl),
                    "n_targets_BH_within": int(hit.sum()),
                    "hit_median_abs_I2_over_abs_eXY": float(np.median(np.abs(I2m[hit]) / np.maximum(np.abs(eXY[hit]), 1e-30))) if hit.any() else None,
                    "hit_max_abs_I2": float(np.abs(I2m[hit]).max()) if hit.any() else None,
                    "max_abs_I2_all_targets": float(np.abs(I2m).max()),
                    "cos_I2_vs_eY(downstream single effect)": cos, "cos_I2_vs_eX(upstream single effect)": cosX,
                    "downstream_live_active_positions_alone(mean over cells)": float(nY_alone.mean()),
                    "downstream_live_active_positions_after_upstream_edit(mean)": float(nY_after.mean()),
                    "cells_where_downstream_active_count_changes": int((nY_alone != nY_after).sum()),
                })
    pp = {(r["pair"], r["x"], r["y"]): r for r in load_json(SRC / "per_pair.json")}
    for r in rows:
        r["agg_p_signflip"] = pp[(r["pair"], r["x"], r["y"])]["agg_p_signflip"]
    n_ci = sum(1 for r in rows if r["energy_share_I2_ci95_basic"][0] > 0)
    big = sorted(rows, key=lambda r: -r["n_targets_BH_within"])
    summ = {
        "n_pairs": len(rows),
        "pairs_with_any_BH_target": sum(1 for r in rows if r["n_targets_BH_within"] > 0),
        "pairs_with_any_BH_target_expected_if_no_interaction(<=)": 0.05 * len(rows),
        "pairs_energy_share_I2_ci_above_0": n_ci,
        "pairs_rho2_ci_below_1": sum(1 for r in rows if r["rho2_ci95_basic"][1] < 1),
        "pairs_rho2_ci_above_1": sum(1 for r in rows if r["rho2_ci95_basic"][0] > 1),
        "median_energy_share_I2": float(np.median([r["energy_share_I2"] for r in rows])),
        "median_rho2": float(np.median([r["rho2"] for r in rows])),
        "by_type": {nm: {"pairs_any_BH": sum(1 for r in rows if r["pair"] == nm and r["n_targets_BH_within"] > 0),
                         "pairs_share_ci_above_0": sum(1 for r in rows if r["pair"] == nm and r["energy_share_I2_ci95_basic"][0] > 0),
                         "median_energy_share_I2": float(np.median([r["energy_share_I2"] for r in rows if r["pair"] == nm]))}
                    for nm in ("AB", "AC", "BC")},
        "top_pairs_by_BH_targets": big[:10],
        "pairs_with_share_ci_above_0": [r for r in rows if r["energy_share_I2_ci95_basic"][0] > 0],
    }
    # the triplet-level per-target hits
    pt = [r for r in load_json(SRC / "per_triplet.json") if r["kind"] == "real"]
    rc = [load_json(f) for f in sorted((SRC / "recheck").glob("cell*.json"))]
    hits = []
    for r in pt:
        if r["n_q05_within_t"] == 0:
            continue
        d = get_terms(D, cidx, r["a"], r["b"], r["c"])
        Iv = d["ABC"] - d["AB"] - d["AC"] - d["BC"] + d["A"] + d["B"] + d["C"]
        p = m.ttest_p(Iv)
        q = m.bh(p)
        for j in np.nonzero(q < 0.05)[0]:
            hits.append({"triplet": [r["a"], r["b"], r["c"]], "target": int(j), "p": float(p[j]), "q_within": float(q[j]),
                         "q_across_all_512x4928": None, "mean_I": float(Iv[:, j].mean()), "sd_I": float(Iv[:, j].std(ddof=1)),
                         "mean_e_ABC": float(d["ABC"][:, j].mean()),
                         "abs_I_over_abs_eABC": float(abs(Iv[:, j].mean()) / max(abs(d["ABC"][:, j].mean()), 1e-30)),
                         "max_abs_per_cell_I": float(np.abs(Iv[:, j]).max()),
                         "agg_p_signflip_of_triplet": r["agg_p_signflip"]})
    z = np.load(SRC / "interaction_terms_real.npz")
    trip = [tuple(x) for x in z["triplets"]]
    for h in hits:
        k = trip.index(tuple(h["triplet"]))
        h["q_across_all_512x4928"] = float(z["q_across_t"][k, h["target"]])
    summ["triplet_target_hits"] = hits
    summ["triplet_hits_note"] = (
        "BH within a triplet controls the chance of any false hit in that triplet at 5% when there is no "
        "interaction, so up to 0.05 x 512 = 25.6 triplets with a hit are expected by chance.")
    summ["float32_measurement_floor"] = {
        "recheck_max_abs_dz_diff(stored float32 vs CPU float64 sums)": max(c["max_abs_dz_diff"] for d_ in rc for c in d_["checks"]),
        "note": "per-cell code changes are stored as float32 position means; differences of ~1e-8 to 1e-7 are rounding"}
    summ["wall_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    H.write_json(OUT / "pairs.json", {"summary": summ, "rows": rows})
    record("pairs", t0)
    s2 = dict(summ)
    s2.pop("top_pairs_by_BH_targets")
    log(json.dumps(s2, indent=1, default=str)[:6000])
    for r in big[:6]:
        log(json.dumps(r, default=str))


# =============================================================================
# perfeature: is there a three-way term that involves one feature? (all 24 features, BH over 24)
# =============================================================================
def stage_perfeature(args):
    """For each of the 24 chosen features, pool its 64 triplets:
      * pooled sign-flip test: statistic = sum over the 64 triplets of sum_t |mean_c I_ABC[c, t]|; the same
        999 flips (signflip_signs.npy) are applied to all 64 triplets at once; BH over the 24 features;
      * pooled noise-corrected energy share of I_ABC: sum_triplets E[I] / sum_triplets E[d_ABC], with a basic
        bootstrap CI (same 1,000 cell-weight vectors as analyze, pivot on the uncorrected plug-in value);
      * pooled rho3 = sum E[d_ABC] / sum E[d_A + d_B + d_C], same CI method.
    Added after the analyze stage showed 44 of 512 triplets at p < 0.05 with 15 of them on L5 feature 4369."""
    t0 = time.time()
    m = load_main()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = m.build_design(sel)
    D, _, _, _, cidx = m.load_grid(nC, conds, a_groups, with_logits=False)
    Sf = np.load(SRC / "signflip_signs.npy").astype(np.float64) / nC
    Wb = np.load(SRC / "bootstrap_cell_weights.npy").astype(np.float64)
    real = [t for t in triplets if t["kind"] == "real"]

    def energy(X):
        return float((X.mean(0) ** 2).sum() - X.var(0, ddof=1).sum() / X.shape[0])

    def plugin(X):
        return float((X.mean(0) ** 2).sum())

    def energy_boot(X):
        n = X.shape[0]
        mu = Wb @ X
        var = np.clip(Wb @ (X * X) - mu * mu, 0, None) * n / (n - 1)
        return (mu * mu).sum(1) - var.sum(1) / n

    per = []
    for t in real:
        d = get_terms(D, cidx, t["a"], t["b"], t["c"])
        Iv = d["ABC"] - d["AB"] - d["AC"] - d["BC"] + d["A"] + d["B"] + d["C"]
        add = d["A"] + d["B"] + d["C"]
        per.append({"a": t["a"], "b": t["b"], "c": t["c"],
                    "T_obs": float(np.abs(Iv.mean(0)).sum()), "T_null": np.abs(Sf @ Iv).sum(1),
                    "E_I": energy(Iv), "E_ABC": energy(d["ABC"]), "E_add": energy(add),
                    "P_I": plugin(Iv), "P_ABC": plugin(d["ABC"]), "P_add": plugin(add),
                    "B_I": energy_boot(Iv), "B_ABC": energy_boot(d["ABC"]), "B_add": energy_boot(add)})
    feats = [("L0", f, "a") for f in sel["features"]["0"]] + [("L5", f, "b") for f in sel["features"]["5"]] + \
            [("L9", f, "c") for f in sel["features"]["9"]]
    rows = []
    for lname, f, key in feats:
        grp = [r for r in per if r[key] == f]
        To = sum(r["T_obs"] for r in grp)
        Tn = np.sum([r["T_null"] for r in grp], axis=0)
        p = float((1 + (Tn >= To).sum()) / (len(Tn) + 1))
        sh = sum(r["E_I"] for r in grp) / sum(r["E_ABC"] for r in grp)
        sh_pl = sum(r["P_I"] for r in grp) / sum(r["P_ABC"] for r in grp)
        b_sh = np.sum([r["B_I"] for r in grp], axis=0) / np.sum([r["B_ABC"] for r in grp], axis=0)
        rho = sum(r["E_ABC"] for r in grp) / sum(r["E_add"] for r in grp)
        rho_pl = sum(r["P_ABC"] for r in grp) / sum(r["P_add"] for r in grp)
        b_rho = np.sum([r["B_ABC"] for r in grp], axis=0) / np.sum([r["B_add"] for r in grp], axis=0)
        rows.append({"feature": f"{lname}_{f}", "n_triplets": len(grp), "pooled_T_obs_over_null_median": To / float(np.median(Tn)),
                     "pooled_signflip_p": p,
                     "pooled_energy_share_I": sh, "pooled_energy_share_I_ci95_basic": m.ci_basic(sh, b_sh, sh_pl),
                     "pooled_rho3": rho, "pooled_rho3_ci95_basic": m.ci_basic(rho, b_rho, rho_pl)})
    for r, q in zip(rows, m.bh([r["pooled_signflip_p"] for r in rows])):
        r["pooled_signflip_q_BH_over_24"] = float(q)
    # all 512 pooled (one global test for "any consistent three-way term")
    To = sum(r["T_obs"] for r in per)
    Tn = np.sum([r["T_null"] for r in per], axis=0)
    sh = sum(r["E_I"] for r in per) / sum(r["E_ABC"] for r in per)
    sh_pl = sum(r["P_I"] for r in per) / sum(r["P_ABC"] for r in per)
    b_sh = np.sum([r["B_I"] for r in per], axis=0) / np.sum([r["B_ABC"] for r in per], axis=0)
    res = {"features": rows,
           "n_features_q05": sum(1 for r in rows if r["pooled_signflip_q_BH_over_24"] < 0.05),
           "n_features_p05": sum(1 for r in rows if r["pooled_signflip_p"] < 0.05),
           "n_features_energy_share_ci_above_0": sum(1 for r in rows if r["pooled_energy_share_I_ci95_basic"][0] > 0),
           "all_512_pooled": {"T_obs_over_null_median": To / float(np.median(Tn)),
                              "signflip_p": float((1 + (Tn >= To).sum()) / (len(Tn) + 1)),
                              "energy_share_I": sh, "energy_share_I_ci95_basic": m.ci_basic(sh, b_sh, sh_pl)},
           "note": "post hoc (added after analyze); all 24 features tested so BH over 24 covers the choice of feature",
           "wall_seconds": round(time.time() - t0, 1)}
    H.write_json(OUT / "perfeature.json", res)
    record("perfeature", t0)
    for r in sorted(rows, key=lambda r: r["pooled_signflip_p"])[:8]:
        log(json.dumps(r))
    log(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=1))


# =============================================================================
# replay: v3 analysis code on the v2 grid
# =============================================================================
def stage_replay(args):
    t0 = time.time()
    rdir = OUT / "v2_replay"
    rdir.mkdir(parents=True, exist_ok=True)
    m = load_main("v3_triplets_replay")
    m.SRC = V2
    m.OUT = rdir
    ns = types.SimpleNamespace(max_minutes=args.max_minutes, device="cpu")
    if not (rdir / "summary.json").exists():
        m.stage_analyze(ns)
        if not (rdir / "summary.json").exists():
            record("replay", t0, {"status": "PARTIAL (analyze)"})
            log("PARTIAL: run replay again")
            return
    if not (rdir / "power.json").exists():
        m.stage_power(ns)
    v2 = load_json(V2 / "summary.json")
    rp = load_json(rdir / "summary.json")
    diffs, n = [], [0]

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if k in b:
                    walk(a[k], b[k], path + [k])
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + [str(i)])
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            n[0] += 1
            d = abs(a - b) / max(abs(a), abs(b), 1e-12)
            if d > 1e-6:
                diffs.append({"path": " / ".join(path), "v2": a, "replay": b, "rel_diff": d})
    walk(v2, rp, [])
    only_v2 = []

    def keys(d, path=()):
        out = []
        if isinstance(d, dict):
            for k, v in d.items():
                out.append(path + (k,))
                out += keys(v, path + (k,))
        return out
    k2, kr = set(keys(v2)), set(keys(rp))
    only_v2 = sorted(" / ".join(k) for k in k2 - kr)
    p = load_json(rdir / "power.json")
    v2a = load_json(V2F / "align.json")
    v2c = load_json(V2F / "classnull.json")
    pw = []
    for k, v in p["aggregate_signflip_power_proportional_plant"].items():
        o = v2a["aggregate_signflip_power_proportional_plant"].get(k)
        pw.append({"what": f"aggregate proportional {k}", "v2_followup": o, "replay": v, "equal": o == v})
    for k, v in p["per_target_power"].items():
        o = v2c["per_target_power"].get(k)
        pw.append({"what": f"per-target {k}", "v2_followup": o, "replay": v, "equal": o == v if o is not None else None})
    key = "targets_with_BH_significant_effect_per_triplet(t-test, q<0.05 within triplet)"
    for k in ("A", "B", "C", "ABC", "I"):
        o = v2a.get(key, {}).get(k)
        pw.append({"what": f"BH effect targets {k}", "v2_followup": o, "replay": p[key][k],
                   "equal": (abs(o["mean"] - p[key][k]["mean"]) < 1e-9) if o else None})
    res = {"numeric_values_compared(summary.json)": n[0], "differing(rel > 1e-6)": len(diffs), "diffs": diffs,
           "keys_in_v2_summary_missing_in_replay": only_v2, "power_comparison": pw,
           "power_rows_equal": sum(1 for r in pw if r["equal"]), "power_rows_compared": sum(1 for r in pw if r["equal"] is not None)}
    H.write_json(OUT / "v2_replay_compare.json", res)
    record("replay", t0, {"status": "DONE"})
    log(f"compared {n[0]} numbers; {len(diffs)} differ; power rows equal {res['power_rows_equal']} of {res['power_rows_compared']}")
    for d in diffs[:60]:
        log(json.dumps(d, default=str))
    for r in pw:
        if not r["equal"]:
            log(json.dumps(r, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["countnull", "pairs", "perfeature", "replay"])
    ap.add_argument("--max-minutes", type=float, default=7.5)
    args = ap.parse_args()
    {"countnull": stage_countnull, "pairs": stage_pairs, "perfeature": stage_perfeature,
     "replay": stage_replay}[args.stage](args)


if __name__ == "__main__":
    main()
