"""v3 steering (item V3-8): statistics and tables (no model).

Reads outputs/v3_steering/{cells.npz, selection.json, signatures.npz, steer/cell_*.npz} and writes
  summary.json, candidates_table.csv, null_table.csv, controls_table.csv, comparison.json.
Uses every steered cell that is finished (n is reported). Rules are in design.json.

Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering_summarize.py
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_steering as V  # noqa: E402
from v3_steering import H  # noqa: E402

N_PERM = 100_000
TOP_GENES = 10


def gene_namer():
    from maxtoki_adapter import MaxTokiTokenizer
    from dataset_loader import SYM2ENS_PKL
    tok = MaxTokiTokenizer()
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    ens2sym = {v: k for k, v in sym2ens.items()}
    tok2ens = {v: k for k, v in tok.gene_token_dict.items()}

    def gene(t):
        e = tok2ens.get(int(t))
        return f"<tok{t}>" if e is None else ens2sym.get(e, e)
    return gene


def boot_means(x, idx):
    return np.asarray(x, dtype=np.float64)[idx].mean(1)


def ci(x, idx):
    m = boot_means(x, idx)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_steer(selj, cells):
    steer_idx = [int(i) for i in np.where(cells["group"] == "steer")[0]]
    done = [i for i in steer_idx if (V.STEER_DIR / f"cell_{i:03d}.npz").exists()]
    D = {l: {k: [] for k in ("delta_s", "gap_share", "edit_norm_gene", "n_active_gene", "meanabs_dlogit",
                             "dvec_sum", "ref_vs_full_maxabs", "silent_dvec_max")} for l in V.LAYERS}
    for i in done:
        z = np.load(V.STEER_DIR / f"cell_{i:03d}.npz")
        for l in V.LAYERS:
            for k in ("delta_s", "gap_share", "edit_norm_gene", "n_active_gene", "meanabs_dlogit"):
                D[l][k].append(z[f"L{l}_{k}"])
            dv = z[f"L{l}_dvec16"].astype(np.float64)
            D[l]["dvec_sum"].append(dv)
            D[l]["ref_vs_full_maxabs"].append(float(z[f"L{l}_ref_vs_full_maxabs"]))
            feat_rows = np.array([k in ("cand", "null") for k, f, a in V.unit_rows(selj, l)])
            sil = (z[f"L{l}_n_active_gene"] == 0) & feat_rows
            D[l]["silent_dvec_max"].append(float(np.abs(dv[sil]).max()) if sil.any() else 0.0)
    for l in V.LAYERS:
        for k in ("delta_s", "gap_share", "edit_norm_gene", "n_active_gene", "meanabs_dlogit"):
            D[l][k] = np.stack(D[l][k])                        # (n_cells, n_rows)
        D[l]["dvec_mean"] = np.mean(np.stack(D[l]["dvec_sum"]), axis=0)   # (n_rows, V)
        del D[l]["dvec_sum"]
    return done, D


def main():
    selj = V.load_json(V.OUT / "selection.json")
    design = V.load_json(V.OUT / "design.json")
    cells = V.load_cells()
    G = {k: v for k, v in np.load(V.OUT / "signatures.npz").items()}
    S = selj["axis"]["natural_span_S"]
    gene = gene_namer()
    done, D = load_steer(selj, cells)
    n = len(done)
    rng = np.random.default_rng(V.BOOT_SEED)
    idx = rng.integers(0, n, size=(V.N_BOOT, n))
    checks = {}
    # ---------------------------------------------------------------- checks on stored data
    sil_ds, sil_n = 0.0, 0
    for l in V.LAYERS:
        s = D[l]["n_active_gene"] == 0
        rows = V.unit_rows(selj, l)
        steer_rows = np.array([k in ("cand", "null") for k, f, a in rows])
        s = s & steer_rows[None, :]
        sil_n += int(s.sum())
        if s.any():
            sil_ds = max(sil_ds, float(np.abs(D[l]["delta_s"][s]).max()))
    checks["silent_rows_exactly_zero"] = {"n_cell_feature_alpha_rows_silent": sil_n, "max_abs_delta_s": sil_ds,
                                          "max_abs_logit_change": max(max(D[l]["silent_dvec_max"]) for l in V.LAYERS),
                                          "pass": sil_ds == 0.0 and max(max(D[l]["silent_dvec_max"]) for l in V.LAYERS) == 0.0}
    checks["shortcut_clean_ref_vs_full_pass"] = {
        "max_abs_mean_logit_diff": max(max(D[l]["ref_vs_full_maxabs"]) for l in V.LAYERS),
        "pass": max(max(D[l]["ref_vs_full_maxabs"]) for l in V.LAYERS) < 1e-4}
    ver = V.load_json(V.VERIFY_DIR / "verify.json", {})
    checks["verify_stage"] = ver.get("summary", {"all_pass": False, "note": "verify.json missing"})
    checks["verify_stage"]["pass"] = bool(checks["verify_stage"].get("all_pass"))
    checks["axis_check"] = {"auroc_EryP_vs_HSC_steering_donors": selj["axis"]["steering_donors"]["auroc_EryP_vs_HSC"],
                            "spearman_s_stage_steering_donors": selj["axis"]["steering_donors"]["spearman_s_stage"],
                            "pass": selj["check_axis_pass"]}
    # edits happened where the feature is active
    nz_ok = True
    for l in V.LAYERS:
        rows = V.unit_rows(selj, l)
        for r, (k, f, a) in enumerate(rows):
            if k in ("cand", "null") and a != 1.0:
                act = D[l]["n_active_gene"][:, r] > 0
                if np.any(D[l]["edit_norm_gene"][act, r] <= 0):
                    nz_ok = False
    checks["edit_nonzero_when_active"] = {"pass": nz_ok}
    prep = V.load_json(V.OUT / "prep.json")
    checks["encoding"] = {"prep_all_cells": prep["encoding_check"]["all_pass"],
                          "deployed_path_fails": prep["deployed_path_fails_check"],
                          "steer_chunks_checks_passed": int(sum(c.get("encoding_checks_passed", 0)
                                                                for c in V.load_json(V.OUT / "run_config.json")["chunks"]
                                                                if c.get("stage") == "steer")),
                          "pass": bool(prep["encoding_check"]["all_pass"])}

    # ---------------------------------------------------------------- per layer
    per_layer, cand_rows, null_rows, ctrl_rows = {}, [], [], []
    feat_records = []          # for pooled tests: (layer, feature, kind, rho, alpha -> mean ds)
    for l in V.LAYERS:
        L = selj["layers"][str(l)]
        rows = V.unit_rows(selj, l)
        ds = D[l]["delta_s"]
        gs = D[l]["gap_share"]
        en = D[l]["edit_norm_gene"]

        def col(kind, f, a):
            return rows.index((kind, f, a))
        feats = [(c["feature"], "cand", c["rho"]) for c in L["candidates"]] + \
                [(x["feature"], "null", x["rho"]) for x in L["nulls"]]
        lay = {"null": {}, "candidates": [], "controls": {}, "layer_test": {}, "rho_vs_ds": {}}
        null_vals = {}
        for a in V.ALPHAS:
            nv = np.array([ds[:, col("null", f, a)].mean() for f, k, r in feats if k == "null"])
            null_vals[a] = nv
            lay["null"][f"alpha{a:g}"] = {"mean": float(nv.mean()), "sd": float(nv.std(ddof=1)),
                                          "p2.5": float(np.percentile(nv, 2.5)), "p97.5": float(np.percentile(nv, 97.5)),
                                          "min": float(nv.min()), "max": float(nv.max()),
                                          "mean_abs": float(np.abs(nv).mean()),
                                          "mean_abs_pct_of_span": float(100 * np.abs(nv).mean() / S)}
        lay["null"]["edit_norm_gene_alpha5_mean"] = float(np.mean([en[:, col("null", f, 5.0)].mean()
                                                                   for f, k, r in feats if k == "null"]))
        for f, k, rho in feats:
            if k != "cand":
                continue
            crec = next(c for c in L["candidates"] if c["feature"] == f)
            sg = int(np.sign(rho))
            row = {"feature": f, "rho": rho, "rho_by_donor": crec["rho_by_donor"],
                   "mean_a_by_stage_sel": crec["mean_a_by_stage_sel"],
                   "frac_cells_active": float((D[l]["n_active_gene"][:, col("cand", f, 5.0)] > 0).mean()),
                   "mean_active_gene_positions": float(D[l]["n_active_gene"][:, col("cand", f, 5.0)].mean())}
            for a in V.ALPHAS:
                x = ds[:, col("cand", f, a)]
                g = gs[:, col("cand", f, a)]
                pred = sg if a > 1 else -sg
                m = float(x.mean())
                nv = null_vals[a]
                st = {"mean_delta_s": m, "ci95": ci(x, idx), "pct_of_span": 100 * m / S,
                      "ci95_pct_of_span": [100 * v / S for v in ci(x, idx)],
                      "predicted_sign": pred, "frac_cells_predicted_sign": float((np.sign(x) == pred).mean()),
                      "gap_share_mean": float(g.mean()), "gap_share_ci95": ci(g, idx),
                      "edit_norm_gene_mean": float(en[:, col("cand", f, a)].mean()),
                      "z_vs_null": float((m - nv.mean()) / nv.std(ddof=1)),
                      "p_directional_vs_null": float((1 + np.sum(pred * nv >= pred * m)) / (1 + len(nv))),
                      "p_size_vs_null": float((1 + np.sum(np.abs(nv) >= abs(m))) / (1 + len(nv))),
                      "n_null_bigger_in_predicted_direction": int(np.sum(pred * nv >= pred * m))}
                row[f"alpha{a:g}"] = st
            # ---- added after the first look at the data (not in design.json; labelled in the report):
            # (a) the same null comparison on the second readout (gap share), and
            # (b) size-normalised comparison: effect per unit of mean edit size per gene position
            #     (the null was matched on activity in selection HSCs, not in the steered cells).
            for a in V.ALPHAS:
                pred = sg if a > 1 else -sg
                st = row[f"alpha{a:g}"]
                ng = np.array([gs[:, col("null", f2, a)].mean() for f2, k2, r2 in feats if k2 == "null"])
                g = st["gap_share_mean"]
                st["gap_z_vs_null"] = float((g - ng.mean()) / ng.std(ddof=1))
                st["gap_p_directional_vs_null"] = float((1 + np.sum(pred * ng >= pred * g)) / (1 + len(ng)))
                ne = np.array([en[:, col("null", f2, a)].mean() for f2, k2, r2 in feats if k2 == "null"])
                nds = np.array([ds[:, col("null", f2, a)].mean() for f2, k2, r2 in feats if k2 == "null"])
                e = st["edit_norm_gene_mean"]
                if e > 0:
                    st["ds_per_unit_edit"] = st["mean_delta_s"] / e
                    st["gap_per_unit_edit"] = g / e
                    st["ds_per_unit_edit_p_dir_vs_null"] = float(
                        (1 + np.sum(pred * nds / ne >= pred * st["ds_per_unit_edit"])) / (1 + len(ne)))
                    st["gap_per_unit_edit_p_dir_vs_null"] = float(
                        (1 + np.sum(pred * ng / ne >= pred * st["gap_per_unit_edit"])) / (1 + len(ne)))
                    st["gap_per_unit_edit_z_vs_null"] = float((g / e - (ng / ne).mean()) / (ng / ne).std(ddof=1))
                st["edit_size_ratio_to_null_mean"] = float(e / ne.mean())
                st["readouts_agree_in_sign"] = bool(np.sign(st["mean_delta_s"]) == np.sign(g))
            x2 = ds[:, col("cand", f, 2.0)]; x5 = ds[:, col("cand", f, 5.0)]; x0 = ds[:, col("cand", f, 0.0)]
            row["ratio_a5_over_a2"] = float(x5.mean() / x2.mean()) if x2.mean() != 0 else None
            row["ratio_a0_over_a2"] = float(x0.mean() / x2.mean()) if x2.mean() != 0 else None
            mv = D[l]["dvec_mean"][col("cand", f, 5.0)]
            row["top_up_genes_a5"] = [gene(t) for t in np.argsort(-mv)[:TOP_GENES]]
            row["top_down_genes_a5"] = [gene(t) for t in np.argsort(mv)[:5]]
            lay["candidates"].append(row)
            cand_rows.append({"layer": l, "feature": f, "rho": round(rho, 3),
                              **{f"ds_a{a:g}": row[f"alpha{a:g}"]["mean_delta_s"] for a in V.ALPHAS},
                              **{f"ci_a{a:g}": "[{:+.2e}, {:+.2e}]".format(*row[f"alpha{a:g}"]["ci95"]) for a in V.ALPHAS},
                              **{f"pct_span_a{a:g}": row[f"alpha{a:g}"]["pct_of_span"] for a in V.ALPHAS},
                              **{f"p_dir_a{a:g}": row[f"alpha{a:g}"]["p_directional_vs_null"] for a in V.ALPHAS},
                              **{f"frac_pred_a{a:g}": row[f"alpha{a:g}"]["frac_cells_predicted_sign"] for a in V.ALPHAS},
                              "z_a5": row["alpha5"]["z_vs_null"], "edit_norm_a5": row["alpha5"]["edit_norm_gene_mean"],
                              **{f"gap_a{a:g}": row[f"alpha{a:g}"]["gap_share_mean"] for a in V.ALPHAS},
                              **{f"gap_p_dir_a{a:g}": row[f"alpha{a:g}"]["gap_p_directional_vs_null"] for a in V.ALPHAS},
                              "gap_z_a5": row["alpha5"]["gap_z_vs_null"],
                              "edit_ratio_to_null_a5": row["alpha5"]["edit_size_ratio_to_null_mean"],
                              "gap_per_edit_p_dir_a5": row["alpha5"].get("gap_per_unit_edit_p_dir_vs_null"),
                              "ds_per_edit_p_dir_a5": row["alpha5"].get("ds_per_unit_edit_p_dir_vs_null"),
                              "top_up_a5": ";".join(row["top_up_genes_a5"][:5])})
        for f, k, rho in feats:
            if k == "null":
                xr = next(x for x in L["nulls"] if x["feature"] == f)
                null_rows.append({"layer": l, "feature": f, "matched_to": xr["matched_to"], "rho": rho,
                                  "mean_a_sel_hsc": xr["mean_a_sel_hsc"], "size_ratio": xr["size_ratio_to_matched"],
                                  **{f"ds_a{a:g}": float(ds[:, col('null', f, a)].mean()) for a in V.ALPHAS},
                                  "edit_norm_a5": float(en[:, col("null", f, 5.0)].mean())})
        # positive controls
        for kind in ("pc_full", "pc_matched"):
            r = col(kind, -1, 1.0)
            x = ds[:, r]
            g = gs[:, r]
            mv = D[l]["dvec_mean"][r]
            lay["controls"][kind] = {"mean_delta_s": float(x.mean()), "ci95": ci(x, idx),
                                     "pct_of_span": float(100 * x.mean() / S),
                                     "frac_cells_toward_late": float((x > 0).mean()),
                                     "gap_share_mean": float(g.mean()), "gap_share_ci95": ci(g, idx),
                                     "edit_norm_gene_mean": float(en[:, r].mean()),
                                     "top_up_genes": [gene(t) for t in np.argsort(-mv)[:TOP_GENES]]}
            ctrl_rows.append({"layer": l, "control": kind, "ds": float(x.mean()),
                              "ci": "[{:+.2e}, {:+.2e}]".format(*ci(x, idx)), "pct_span": float(100 * x.mean() / S),
                              "gap_share": float(g.mean()), "edit_norm": float(en[:, r].mean()),
                              "frac_toward_late": float((x > 0).mean())})
        # layer-level tests
        for a in V.ALPHAS:
            sgn_a = 1.0 if a > 1 else -1.0
            mds = np.array([ds[:, col(k, f, a)].mean() for f, k, r in feats])
            rhos = np.array([r for f, k, r in feats])
            dirstat = np.sign(rhos) * sgn_a * mds
            size = np.abs(mds)
            combs = np.array(list(itertools.combinations(range(len(feats)), 3)))
            obs_d, obs_s = dirstat[:3].mean(), size[:3].mean()
            pd_ = dirstat[combs].mean(1); ps_ = size[combs].mean(1)
            lay["layer_test"][f"alpha{a:g}"] = {
                "n_subsets": int(len(combs)),
                "candidates_mean_directional": float(obs_d), "p_directional": float(np.mean(pd_ >= obs_d - 1e-15)),
                "candidates_mean_abs": float(obs_s), "p_size": float(np.mean(ps_ >= obs_s - 1e-15))}
            sr = spearmanr(rhos, mds)
            lay["rho_vs_ds"][f"alpha{a:g}"] = {"spearman": float(sr.statistic), "p": float(sr.pvalue), "n": len(feats)}
            for (f, k, r), m_ in zip(feats, mds):
                feat_records.append({"layer": l, "feature": f, "kind": k, "rho": r, "alpha": a, "ds": m_})
        # how candidates compare with the size-matched ideal direction
        pcm = lay["controls"]["pc_matched"]["mean_delta_s"]
        lay["candidates_vs_matched_control_alpha5"] = {
            "mean_abs_ds_candidates": float(np.mean([abs(c["alpha5"]["mean_delta_s"]) for c in lay["candidates"]])),
            "matched_control_ds": pcm,
            "candidate_edit_norm_mean": float(np.mean([c["alpha5"]["edit_norm_gene_mean"] for c in lay["candidates"]])),
            "matched_control_edit_norm": lay["controls"]["pc_matched"]["edit_norm_gene_mean"]}
        agree = {"cand": [], "null": []}
        for r, (k, f, a) in enumerate(rows):
            if k in agree and ds[:, r].mean() != 0:
                agree[k].append(bool(np.sign(ds[:, r].mean()) == np.sign(gs[:, r].mean())))
        nds5 = np.array([ds[:, col("null", f, 5.0)].mean() for f, k, r in feats if k == "null"])
        ngs5 = np.array([gs[:, col("null", f, 5.0)].mean() for f, k, r in feats if k == "null"])
        lay["readouts"] = {"sign_agreement_candidate_units": float(np.mean(agree["cand"])),
                           "sign_agreement_null_units": float(np.mean(agree["null"])),
                           "spearman_ds_vs_gap_null_alpha5": float(spearmanr(nds5, ngs5).statistic),
                           "null_gap_alpha5_mean": float(ngs5.mean()), "null_gap_alpha5_sd": float(ngs5.std(ddof=1))}
        per_layer[f"L{l}"] = lay
        V.log(f"L{l}: null a5 mean {lay['null']['alpha5']['mean']:+.2e} sd {lay['null']['alpha5']['sd']:.2e}; "
              + "; ".join(f"F{c['feature']} rho {c['rho']:+.2f} a0 {c['alpha0']['mean_delta_s']:+.2e} "
                          f"a5 {c['alpha5']['mean_delta_s']:+.2e} ({c['alpha5']['pct_of_span']:+.2f}% S) "
                          f"pdir5 {c['alpha5']['p_directional_vs_null']:.3f} pdir0 {c['alpha0']['p_directional_vs_null']:.3f}"
                          for c in lay["candidates"])
              + f"; ctrl full {lay['controls']['pc_full']['pct_of_span']:+.1f}% matched {lay['controls']['pc_matched']['pct_of_span']:+.2f}%"
              + f"; layer p_dir a5 {lay['layer_test']['alpha5']['p_directional']:.3f} a0 {lay['layer_test']['alpha0']['p_directional']:.3f}")

    # ---------------------------------------------------------------- pooled tests across layers
    FR = pd.DataFrame(feat_records)
    pooled = {}
    prng = np.random.default_rng(V.BOOT_SEED + 1)
    for a in V.ALPHAS:
        sgn_a = 1.0 if a > 1 else -1.0
        sub = FR[FR.alpha == a]
        # (1) mean over layers of within-layer Spearman(rho, ds); permutation within layer
        obs = np.mean([spearmanr(sub[sub.layer == l].rho, sub[sub.layer == l].ds).statistic for l in V.LAYERS])
        perm = np.zeros(N_PERM)
        from scipy.stats import rankdata
        Rr = {l: rankdata(sub[sub.layer == l].rho.to_numpy()) for l in V.LAYERS}
        Rd = {l: rankdata(sub[sub.layer == l].ds.to_numpy()) for l in V.LAYERS}
        acc = np.zeros(N_PERM)
        for l in V.LAYERS:
            rr = (Rr[l] - Rr[l].mean()); rr = rr / np.sqrt((rr ** 2).sum())
            dd = (Rd[l] - Rd[l].mean()); dd = dd / np.sqrt((dd ** 2).sum())
            P = np.argsort(prng.random((N_PERM, len(dd))), axis=1)
            acc += (dd[P] * rr[None, :]).sum(1)
        perm = acc / len(V.LAYERS)
        p_two = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (1 + N_PERM))
        p_pred = float((1 + np.sum(sgn_a * perm >= sgn_a * obs - 1e-12)) / (1 + N_PERM))
        # (2) stratified candidate-rank test: sum over layers of the mean within-layer rank of the 3 candidates
        #     by directional stat (own sign of rho)
        obs2 = 0.0
        acc2 = np.zeros(N_PERM)
        for l in V.LAYERS:
            s = sub[sub.layer == l]
            d = np.sign(s.rho.to_numpy()) * sgn_a * s.ds.to_numpy()
            rk = rankdata(d)                      # high rank = strong push in predicted direction
            iscand = (s.kind == "cand").to_numpy()
            obs2 += rk[iscand].mean()
            P = np.argsort(prng.random((N_PERM, len(rk))), axis=1)[:, :3]
            acc2 += rk[P].mean(1)
        p2 = float((1 + np.sum(acc2 >= obs2 - 1e-12)) / (1 + N_PERM))
        pooled[f"alpha{a:g}"] = {
            "mean_within_layer_spearman_rho_vs_ds": float(obs), "p_two_sided": p_two,
            "p_in_predicted_direction": p_pred,
            "predicted_direction": "positive" if a > 1 else "negative",
            "candidate_rank_sum_over_layers": float(obs2), "candidate_rank_expected": float(len(V.LAYERS) * 12.0),
            "candidate_rank_p_one_sided": p2, "n_permutations": N_PERM}
        V.log(f"pooled alpha {a:g}: mean within-layer Spearman(rho, ds) {obs:+.3f} (p two-sided {p_two:.4f}); "
              f"candidate rank sum {obs2:.1f} vs {len(V.LAYERS) * 12.0:.0f} expected, p {p2:.4f}")
    checks["all_pass"] = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))
    summary = {"n_steered_cells": n, "steered_cells": done, "natural_span_S": S,
               "units": {"delta_s": "difference of two cosine differences (no unit; range -4..4); "
                                    "S = mean s(Ery) - mean s(HSC) on selection cells",
                         "gap_share": "fraction of the line from g_early to g_late moved by the cell's mean logits",
                         "ci": f"cell-level percentile bootstrap, {V.N_BOOT} resamples, seed {V.BOOT_SEED}, n = {n} cells"},
               "checks": checks, "axis": selj["axis"], "sae_fit_on_these_cells": selj["sae_fit_on_these_cells"],
               "per_layer": per_layer, "pooled": pooled}
    H.write_json(V.OUT / "summary.json", summary)
    pd.DataFrame(cand_rows).to_csv(V.OUT / "candidates_table.csv", index=False)
    pd.DataFrame(null_rows).to_csv(V.OUT / "null_table.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv(V.OUT / "controls_table.csv", index=False)
    comparison(summary)
    V.log(json.dumps(checks, indent=1, default=str))
    V.log(f"n cells {n}; checks all pass: {checks['all_pass']}")


def comparison(summary):
    """Deployed, v2 zero-edit and v2 (unfinished) numbers next to v3."""
    out = {}
    dep = V.load_json(V.DEP3 / "steering_summary.json")
    out["deployed_experiment3"] = {
        "what": "PC1 'pseudotime' of 200 random TS immune cells; block-deletion hook; old encoding; old SAEs",
        "per_layer_alpha5_mean_delta_s": {k: v["mean_delta_s"] for k, v in dep["per_layer_summary_alpha5"].items()},
        "per_layer_alpha2_mean_delta_s": {k: v["mean_delta_s"] for k, v in dep["per_layer_summary_alpha2"].items()}}
    z = V.load_json(V.V2Z / "summary.json")
    out["v2_zero_edit_control"] = {"what": "deployed hook with a zero edit (V2_HOOKS_REPORT.md section 5)",
                                   "summary_keys": list(z.keys())[:20] if z else None}
    units = sorted((V.V2S / "units").glob("L*_F*.npz"))
    sel2 = V.load_json(V.V2S / "selection.json")
    rows = []
    for p in units:
        u = np.load(p)
        l, f = int(u["layer"]), int(u["feature"])
        al = list(u["alphas"])
        rec = {"layer": l, "feature": f, "kind": str(u["kind"])}
        for a in (0.0, 2.0, 5.0):
            if a in al:
                rec[f"ds_a{a:g}"] = float(u["delta_s"][:, al.index(a)].mean())
        rows.append(rec)
    df = pd.DataFrame(rows)
    v2 = {"what": "fixed hooks, PC1 axis, old encoding, deployed SAEs; stopped after 34 of 115 units "
                  "(15 originals + 19 nulls); no summary was ever computed",
          "n_units": len(rows), "per_layer": {}}
    for l in V.LAYERS:
        o = df[(df.layer == l) & (df.kind == "original")]
        nl = df[(df.layer == l) & (df.kind == "null")]
        v2["per_layer"][f"L{l}"] = {
            "originals_ds_a5": o["ds_a5"].round(6).tolist(), "originals_ds_a0": o.get("ds_a0", pd.Series()).round(6).tolist(),
            "null_ds_a5": nl["ds_a5"].round(6).tolist()}
    out["v2_steering_unfinished"] = v2
    out["v3"] = {l: {"cand_ds_a5": [c["alpha5"]["mean_delta_s"] for c in L["candidates"]],
                     "cand_ds_a0": [c["alpha0"]["mean_delta_s"] for c in L["candidates"]],
                     "null_a5_mean_sd": [L["null"]["alpha5"]["mean"], L["null"]["alpha5"]["sd"]],
                     "pc_full_ds": L["controls"]["pc_full"]["mean_delta_s"]}
                 for l, L in summary["per_layer"].items()}
    H.write_json(V.OUT / "comparison.json", out)


if __name__ == "__main__":
    main()
