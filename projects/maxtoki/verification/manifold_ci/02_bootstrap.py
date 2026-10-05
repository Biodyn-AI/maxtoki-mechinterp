"""Step 2: intervals for the frozen-head trust / branch numbers, several resampling schemes.

 (a) exact re-run of the published scheme (80% anchors, no replacement, 200 reps, seed 42,
     same RNG call order as audit_a3_bootstrap_h65_gates.py) -> must give [0.884,0.899] / [0.270,0.480]
 (b) ordinary anchor bootstrap WITH replacement (1000 reps), copies of one anchor never paired
 (c) donor (cluster) bootstrap WITH replacement (2000 reps), copies of one anchor never paired
 (d) leave-one-donor-out (jackknife) values + jackknife SE

The head is FROZEN (weights identical to artifacts/heads/let_anchor_internal.pt); only the
evaluation panel is resampled. Uncertainty from the 11 training donors is NOT included here.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")
import json, time
import numpy as np, pandas as pd
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from common import *

A_e, A_m, A_l, part = load_operators()
c_int, _, _ = load_panel("internal")
f_int_raw = build_pooled_drift(c_int, A_e, A_m, A_l, part)
mu = f_int_raw.mean(0); sd = f_int_raw.std(0) + 1e-6


def pct(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def summarize(arr):
    arr = np.asarray(arr, float)
    ok = arr[~np.isnan(arr)]
    return {"ci95_percentile": pct(ok), "mean": float(ok.mean()), "sd": float(ok.std(ddof=1)),
            "n_valid": int(len(ok)), "n_nan": int(np.isnan(arr).sum())}


results = {}
for panel in ["external", "zeroshot"]:
    t0 = time.time()
    c, d, m = load_panel(panel)
    f = (build_pooled_drift(c, A_e, A_m, A_l, part) - mu) / sd
    z = np.load(OUT / f"z_{panel}_frozen.npy")
    n = len(d)
    DX = euclid(f); DZ = euclid(z); Dh = arccos_dist(z)
    br = branch_labels(m); donors = m["donor_id"].astype(str).to_numpy()
    all_origin = np.arange(n)
    obs_trust = trust_masked(DX, DZ, all_origin)
    obs_branch, _ = group_spearman(Dh, d, br, all_origin)
    R = {"n_anchors": n, "n_donors": int(len(set(donors))),
         "anchors_per_donor": pd.Series(donors).value_counts().to_dict(),
         "observed": {"trust": obs_trust, "branch": obs_branch}}

    # (a) exact published scheme -------------------------------------------------
    if panel == "external":
        rng = np.random.default_rng(SEED)
        sub_n = int(0.8 * n)
        tr, bb = [], []
        for _ in range(200):
            idx = rng.choice(n, size=sub_n, replace=False)
            f_sub, z_sub = f[idx], z[idx]; d_sub = d[np.ix_(idx, idx)]
            tr.append(float(trustworthiness(f_sub, z_sub, n_neighbors=min(15, sub_n - 1))))
            # consume RNG exactly as the original 'rand' gate did (20 permutations)
            for _r in range(20):
                rng.permutation(sub_n)
            b, _ = group_spearman(Dh[np.ix_(idx, idx)], d_sub, br[idx], np.arange(sub_n))
            bb.append(b)
        R["a_published_80pct_subsample_200"] = {"trust": summarize(tr), "branch": summarize(bb)}
        print(panel, "(a)", R["a_published_80pct_subsample_200"])

    # (b) anchor bootstrap with replacement --------------------------------------
    rng = np.random.default_rng(1)
    tr, bb = [], []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        tr.append(trust_masked(DX[np.ix_(idx, idx)], DZ[np.ix_(idx, idx)], idx))
        b, _ = group_spearman(Dh[np.ix_(idx, idx)], d[np.ix_(idx, idx)], br[idx], idx)
        bb.append(b)
    R["b_anchor_bootstrap_1000"] = {"trust": summarize(tr), "branch": summarize(bb)}
    print(panel, "(b)", R["b_anchor_bootstrap_1000"])

    # (c) donor cluster bootstrap --------------------------------------------------
    ud = np.array(sorted(set(donors)))
    members = {u: np.where(donors == u)[0] for u in ud}
    rng = np.random.default_rng(2)
    tr, bb, nn_, nbr = [], [], [], []
    for _ in range(2000):
        pick = rng.choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        tr.append(trust_masked(DX[np.ix_(idx, idx)], DZ[np.ix_(idx, idx)], idx))
        b, k = group_spearman(Dh[np.ix_(idx, idx)], d[np.ix_(idx, idx)], br[idx], idx)
        bb.append(b); nn_.append(len(idx)); nbr.append(k)
    R["c_donor_cluster_bootstrap_2000"] = {"trust": summarize(tr), "branch": summarize(bb),
                                           "resample_size_range": [int(min(nn_)), int(max(nn_))],
                                           "n_branches_scored_range": [int(min(nbr)), int(max(nbr))],
                                           "frac_trust_below_0.80": float(np.mean(np.array(tr) < 0.80)),
                                           "frac_branch_below_0.20": float(np.nanmean(np.array(bb) < 0.20))}
    np.save(OUT / f"donor_boot_{panel}_trust.npy", np.array(tr)); np.save(OUT / f"donor_boot_{panel}_branch.npy", np.array(bb))
    print(panel, "(c)", R["c_donor_cluster_bootstrap_2000"])

    # (d) leave-one-donor-out -------------------------------------------------------
    lodo = {}
    for u in ud:
        idx = np.where(donors != u)[0]
        t = trust_masked(DX[np.ix_(idx, idx)], DZ[np.ix_(idx, idx)], idx)
        b, _ = group_spearman(Dh[np.ix_(idx, idx)], d[np.ix_(idx, idx)], br[idx], idx)
        lodo[u] = {"n_anchors_left": int(len(idx)), "trust": t, "branch": b}
    tt = np.array([v["trust"] for v in lodo.values()]); bb_ = np.array([v["branch"] for v in lodo.values()])
    G = len(ud)
    def jk(vals, full):
        se = float(np.sqrt((G - 1) / G * ((vals - vals.mean()) ** 2).sum()))
        return {"jackknife_se": se, "normal_ci95": [full - 1.96 * se, full + 1.96 * se],
                "t_ci95_df": G - 1,
                "lodo_min": float(vals.min()), "lodo_max": float(vals.max())}
    from scipy.stats import t as tdist
    q = float(tdist.ppf(0.975, G - 1))
    R["d_leave_one_donor_out"] = {"per_donor": lodo, "trust": jk(tt, obs_trust), "branch": jk(bb_, obs_branch),
                                  "t_quantile": q}
    for key, full in [("trust", obs_trust), ("branch", obs_branch)]:
        se = R["d_leave_one_donor_out"][key]["jackknife_se"]
        R["d_leave_one_donor_out"][key]["t_ci95"] = [full - q * se, full + q * se]
    print(panel, "(d)", {k: R["d_leave_one_donor_out"][k] for k in ("trust", "branch")})
    R["elapsed_s"] = time.time() - t0
    results[panel] = R

(OUT / "02_bootstrap.json").write_text(json.dumps(results, indent=2))
print("wrote", OUT / "02_bootstrap.json")
