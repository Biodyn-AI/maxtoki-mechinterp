"""D3 (revision): extra calibration checks for the cell-level selection test.

Why: in the main run, the feature-level Mann-Whitney p-values of fake TFs were not uniform
(KS p down to 4e-38 for one feature) and 10.9% of fake TFs had >= 1 BH-significant feature
at cut-off 0. All fake TFs there share ONE reference group (400 cells) and ONE pool (800
cells), so their tests are not independent: a chance difference between the pool and the
reference is inherited by every fake TF. This script tests that explanation.

Checks
  A. Pool (all 800) vs reference (400): per-feature MW p-values and BH discoveries.
  B. Fresh re-partition: 1,000 times, split the 1,200 control cells (ref + pool) at random into
     a new reference (400) and a fake knockdown group (95, disjoint). Run the same test.
     Per-feature p-values must be uniform (KS) and the share of splits with >= 1 BH discovery
     (family-wise false-positive rate of the selection) must be about 0.05 or less.
  C. fakeK null reproduction: draw 2,000 new fake TFs (new seed) for the three TFs used at
     cut-off 0.25 (GATA1, MAX, TERF2) and compare the distribution of the best overlap with the
     stored null (chi-square); per-TF KS of randomised p-values.
  D. Larger calibration of the selection-aware p-values: 1,000 new fake TFs per primary TF
     (new seed, same size), same selection code, scored with that TF's target sets against the
     stored 2,000-draw null. Because the statistic T takes only a few values, the direct test is
     a two-sample chi-square (new fake TFs vs stored null draws) per TF; the KS test of the
     randomised p-values is also reported (it is sensitive to the Monte-Carlo error of a
     2,000-draw null when T has few values).
Usage: --parts ABCD (default) or any subset. Results are merged into
outputs/v2_tf_specificity/calibration_extra.json
"""
from __future__ import annotations

import glob
import json
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


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="ABCD"); ap.add_argument("--n-d", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    z = np.load(TD / "cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]
    R = MG[[rix[int(r)] for r in man[man.group == "ref"].row]]
    P = MG[[rix[int(r)] for r in man[man.group == "pool"].row]]
    fixed = np.random.default_rng(SEED + 5).choice(D_SAE, 5, replace=False)
    opath = OUT / "calibration_extra.json"
    out = json.loads(opath.read_text()) if opath.exists() else {}
    out["fixed_features"] = [int(x) for x in fixed]

    mw = RefMW(R)
    cpool = mw.contrib(P); zpool = P == 0
    if "A" in args.parts:
        run_A(out, mw, P, fixed)
    if "B" in args.parts:
        run_B(out, R, P, fixed, t0)
    if "C" in args.parts:
        run_C(out, mw, P, cpool, zpool)
    if "D" in args.parts:
        run_D(out, mw, P, cpool, zpool, args.n_d, t0)
    out["script_sha256"] = H.sha256_file(__file__)
    out.setdefault("wall_s_by_run", []).append({"parts": args.parts, "wall_s": round(time.time() - t0, 1)})
    H.write_json(opath, out)


def run_A(out, mw, P, fixed):
    # A. pool vs reference
    p = mw.test(mw.contrib(P).sum(0), len(P), (P == 0).sum(0))
    q = bh(p)
    out["A_pool_vs_ref"] = {"p_fixed_features": [float(p[f]) for f in fixed],
                            "n_bh_sig": int((q < Q_FEAT).sum()),
                            "n_p_lt_0.001": int((p < 0.001).sum()), "expected_p_lt_0.001": D_SAE * 0.001}
    print("A", out["A_pool_vs_ref"], flush=True)


def run_B(out, R, P, fixed, t0):
    # B. fresh re-partition
    C = np.concatenate([R, P])
    rs = np.random.default_rng(SEED + 31)
    NB = 1000
    pf = np.zeros((NB, 5)); nsig = {c: np.zeros(NB, int) for c in CUTOFFS}
    allp_sample = []
    for b in range(NB):
        perm = rs.permutation(len(C))
        Rb = C[perm[:400]]; Ab = C[perm[400:495]]
        m = RefMW(Rb)
        pb = m.test(m.contrib(Ab).sum(0), len(Ab), (Ab == 0).sum(0))
        qb = bh(pb)
        dl = np.abs(Ab.mean(0) - m.mean_ref)
        pf[b] = pb[fixed]
        allp_sample.append(pb[rs.integers(D_SAE)])
        for c in CUTOFFS:
            nsig[c][b] = int(((qb < Q_FEAT) & (dl > c)).sum())
        if b % 200 == 0:
            print(f"  B split {b} ({time.time()-t0:.0f}s)", flush=True)
    out["B_repartition"] = {
        "n_splits": NB,
        "KS_p_fixed_features": [float(st.kstest(pf[:, j], "uniform").pvalue) for j in range(5)],
        "frac_p_lt_0.05_fixed_features": [float((pf[:, j] < 0.05).mean()) for j in range(5)],
        "KS_p_random_feature_each_split": float(st.kstest(np.array(allp_sample), "uniform").pvalue),
        "frac_random_feature_p_lt_0.05": float((np.array(allp_sample) < 0.05).mean()),
        "fwer_selection_by_cutoff": {str(c): float((nsig[c] > 0).mean()) for c in CUTOFFS},
        "mean_n_selected_by_cutoff": {str(c): float(nsig[c].mean()) for c in CUTOFFS}}
    print("B", out["B_repartition"], flush=True)


def load_sets():
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    dor = dor[dor.chip_flag.astype(str).str.lower() == "true"]
    trr = pd.read_csv(TD / "targets_trrust.tsv", sep="\t", keep_default_na=False)
    g2f = top.groupby("feature_id").gene.apply(list).to_dict()
    def ovf(S):
        return np.array([sum(g in S for g in g2f.get(f, [])) for f in range(D_SAE)])
    return dor, trr, ovf


def run_C(out, mw, P, cpool, zpool):
    # C. fakeK null reproduction at cut-off 0.25
    res = pd.read_csv(OUT / "tf_results.csv")
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    dor = dor[dor.chip_flag.astype(str).str.lower() == "true"]
    outC = {}
    for tf in ["GATA1", "MAX", "TERF2"]:
        r = res[(res.tf == tf) & (res.cutoff == 0.25) & (res.db == "ChIP")].iloc[0]
        K = int(r.K); n = int(r.n_cells)
        S = set(dor[dor.tf == tf].target)
        ov = np.zeros(D_SAE, int)
        for f, g in top.groupby("feature_id"):
            ov[f] = int(g.gene.isin(S).sum())
        stored = np.load(OUT / f"fake_null/size_{n:03d}.npz")[f"M|{tf}|0.25|ChIP|fakeK"]
        rn = np.random.default_rng(SEED + 4000 + n)
        new = np.zeros(2000, int)
        for b in range(2000):
            ix = rn.choice(len(P), n, replace=False)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            new[b] = ov[np.argsort(-dl, kind="stable")[:K]].max()
        vals = sorted(set(stored) | set(new))
        tab = np.array([[int((stored == v).sum()), int((new == v).sum())] for v in vals])
        chi = st.chi2_contingency(tab[tab.sum(1) > 0]) if len(vals) > 1 else None
        Tn = cap_T(stored); Tnew = cap_T(new)
        ru = np.random.default_rng(SEED + 77)
        prand = np.array([((Tn > t).sum() + ru.random() * ((Tn == t).sum() + 1)) / (len(Tn) + 1) for t in Tnew])
        outC[tf] = {"K": K, "size": n, "values": [int(v) for v in vals], "counts_stored_vs_new": tab.tolist(),
                    "chi2_p": float(chi.pvalue) if chi is not None else None,
                    "KS_p_randomized_new_vs_stored": float(st.kstest(prand, "uniform").pvalue)}
        print("C", tf, outC[tf], flush=True)
    out["C_fakeK_reproduction_cut0.25"] = outC


def run_D(out, mw, P, cpool, zpool, n_d, t0):
    from collections import defaultdict
    res = pd.read_csv(OUT / "tf_results.csv")
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    dor, trr, ovf = load_sets()
    rd = np.random.default_rng(SEED + 8888)
    prand, pcons, chi = defaultdict(list), defaultdict(list), defaultdict(list)
    Tobs = defaultdict(list)
    for t in meta["tfs_primary"]:
        rr = res[res.tf == t]
        n = int(rr.n_cells.iloc[0])
        K = {c: int(rr[rr.cutoff == c].K.iloc[0]) for c in CUTOFFS}
        sets = {"ChIP": set(dor[dor.tf == t].target), "TRRUST": set(trr[trr.tf == t].target)}
        ov = {db: ovf(S) for db, S in sets.items() if len(S)}
        zc = np.load(OUT / f"fake_null/size_{n:03d}.npz")
        nulls = {(c, db, v): cap_T(zc[f"M|{t}|{c}|{db}|{v}"]) for c in CUTOFFS for db in ov for v in ["fake", "fakeK"]}
        obsT = defaultdict(list)
        for b in range(n_d):
            ix = rd.choice(len(P), n, replace=False)
            p = mw.test(cpool[ix].sum(0), n, zpool[ix].sum(0)); q = bh(p)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            ord_d = np.argsort(-dl, kind="stable"); ord_p = np.lexsort((-dl, p))
            for c in CUTOFFS:
                Rb = np.where((q < Q_FEAT) & (dl > c))[0]
                for db in ov:
                    for v, RR in [("fake", Rb), ("fakeK", (ord_d if c > 0 else ord_p)[:K[c]])]:
                        if v == "fakeK" and K[c] == 0:
                            continue
                        obsT[(c, db, v)].append(int(cap_T(ov[db][RR].max())) if len(RR) else 0)
        for key, Tl in obsT.items():
            Tl = np.array(Tl); Tn = nulls[key]
            u = rd.random(len(Tl))
            gt = np.array([(Tn > x).sum() for x in Tl]); eq = np.array([(Tn == x).sum() for x in Tl])
            prand[key].extend(list((gt + u * (eq + 1)) / (len(Tn) + 1)))
            pcons[key].extend(list(np.where(Tl >= 2, (1 + gt + eq) / (len(Tn) + 1), 1.0)))
            vals = sorted(set(Tl) | set(Tn))
            tab = np.array([[int((Tn == x).sum()), int((Tl == x).sum())] for x in vals])
            if len(vals) > 1:
                chi[key].append(float(st.chi2_contingency(tab).pvalue))
        print(f"  D {t} done ({time.time()-t0:.0f}s)", flush=True)
    outD = {"n_per_tf": n_d, "seed": SEED + 8888}
    for key in sorted(prand, key=str):
        c, db, v = key
        pr = np.array(prand[key]); pc = np.array(pcons[key])
        outD[f"{c}|{db}|{v}"] = {"n": int(len(pr)), "KS_p_randomized_pooled": float(st.kstest(pr, "uniform").pvalue),
                                 "frac_p_conservative_le_0.05": float((pc <= 0.05).mean()),
                                 "n_tfs_with_varying_T": len(chi[key]),
                                 "chi2_two_sample_p_min": float(min(chi[key])) if chi[key] else None,
                                 "chi2_two_sample_n_lt_0.05": int(sum(x < 0.05 for x in chi[key]))}
    out["D_calibration_1000_per_tf"] = outD


if __name__ == "__main__":
    main()
