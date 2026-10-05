"""V3-2: extra calibration checks for the cell-level selection test (same as v2 checks A-D).

All Mann-Whitney tests here use the full tie correction (zeros and tied non-zero values), as in
v3_tf_specificity_stats.py.
  A. Pool (800) vs reference (400): per-feature p-values and BH discoveries.
  B. Fresh re-partition, 1,000 times: split the 1,200 control cells (ref + pool) at random into a
     new reference (400) and a fake knockdown group (95, disjoint). Per-feature p-values must be
     uniform (KS) and the share of splits with >= 1 responding feature (family-wise false-positive
     rate of the selection) must be about 0.05 or less. Resumable in blocks of 250 splits.
  C. fakeK null reproduction: 2,000 new fake TFs (new seed) for every primary TF with K > 0 at
     cut-off 0.25; distribution of the best overlap vs the stored null (chi-square).
  D. 1,000 new fake TFs per primary TF (new seed), same selection code, scored with that TF's
     target sets against the stored 2,000-draw null; KS of randomised p-values and two-sample
     chi-square per TF. Resumable per TF.
Usage: --parts ABCD (default) or a subset; results merged into outputs/v3_tf_specificity/calibration_extra.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import v3_tf_specificity_stats as S  # noqa: E402
from v3_tf_specificity_stats import RefMW, RefMWT, TieIndex, bh, cap_T, CUTOFFS, Q_FEAT, SEED, D_SAE  # noqa: E402

OUT = S.OUT
TD = S.TD
CC = S.CACHE / "calib_extra"


def mw_full(m: RefMW, ties: TieIndex, U1, n1, zeros_a, cnt_ref, cnt_a):
    n2 = m.n2; n = n1 + n2
    mu = n1 * n2 / 2.0
    t0 = (zeros_a + m.zero_ref).astype(np.float64)
    tie = ((t0 ** 3 - t0) + ties.tie_sum(cnt_ref + cnt_a)) / (n * (n - 1))
    var = n1 * n2 / 12.0 * ((n + 1) - tie)
    U = np.maximum(U1, n1 * n2 - U1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (U - mu - 0.5) / np.sqrt(var)
    p = np.clip(2 * st.norm.sf(z), 0, 1)
    p[~(var > 0)] = 1.0
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCD")
    ap.add_argument("--n-d", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=7.0)
    args = ap.parse_args()
    t0 = time.time()
    CC.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    z = np.load(TD / "cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]
    ties = TieIndex(MG)
    R = MG[[rix[int(r)] for r in man[man.group == "ref"].row]]
    P = MG[[rix[int(r)] for r in man[man.group == "pool"].row]]
    fixed = np.random.default_rng(SEED + 5).choice(D_SAE, 5, replace=False)
    opath = OUT / "calibration_extra.json"
    out = json.loads(opath.read_text()) if opath.exists() else {}
    out["fixed_features"] = [int(x) for x in fixed]
    out["mw"] = "full tie correction (zeros and tied non-zero values)"
    mw = RefMWT(R, ties)
    cpool = mw.contrib(P); zpool = P == 0; ipool = ties.indicator(P).astype(np.int32)
    status = "complete"
    if "A" in args.parts:
        p = mw.test_group(P); q = bh(p)
        out["A_pool_vs_ref"] = {"p_fixed_features": [float(p[f]) for f in fixed], "n_bh_sig": int((q < Q_FEAT).sum()),
                                "n_p_lt_0.001": int((p < 0.001).sum()), "expected_p_lt_0.001": D_SAE * 0.001,
                                "n_p_lt_0.05": int((p < 0.05).sum()), "expected_p_lt_0.05": D_SAE * 0.05}
        print("A", out["A_pool_vs_ref"], flush=True)
    if "B" in args.parts:
        st_ = run_B(out, R, P, ties, fixed, t0, args.max_minutes)
        status = st_ if st_ != "complete" else status
    if "C" in args.parts:
        run_C(out, mw, P)
    if "D" in args.parts and status == "complete":
        st_ = run_D(out, mw, P, cpool, zpool, ipool, args.n_d, t0, args.max_minutes)
        status = st_ if st_ != "complete" else status
    out["script_sha256"] = S.H.sha256_file(__file__)
    out.setdefault("wall_s_by_run", []).append({"parts": args.parts, "wall_s": round(time.time() - t0, 1), "status": status})
    S.H.write_json(opath, out)
    print("status:", status, flush=True)
    if status != "complete":
        sys.exit(3)


def run_B(out, R, P, ties, fixed, t0, max_min):
    C = np.concatenate([R, P])
    Cind = ties.indicator(C).astype(np.int32)
    NB, BLOCK = 1000, 250
    for blk in range(NB // BLOCK):
        bp = CC / f"B_block{blk}.pkl"
        if bp.exists():
            continue
        if (time.time() - t0) / 60 > max_min:
            print(f"STOPPED before B block {blk}", flush=True)
            return "incomplete_B"
        rs = np.random.default_rng([SEED + 31, blk])
        pf = np.zeros((BLOCK, 5)); nsig = {c: np.zeros(BLOCK, int) for c in CUTOFFS}; samp = []
        for b in range(BLOCK):
            perm = rs.permutation(len(C))
            ri, ai = perm[:400], perm[400:495]
            Rb = C[ri]; Ab = C[ai]
            m = RefMW(Rb)
            pb = mw_full(m, ties, m.contrib(Ab).sum(0), len(Ab), (Ab == 0).sum(0), Cind[ri].sum(0), Cind[ai].sum(0))
            qb = bh(pb)
            dl = np.abs(Ab.mean(0) - m.mean_ref)
            pf[b] = pb[fixed]
            samp.append(pb[rs.integers(D_SAE)])
            for c in CUTOFFS:
                nsig[c][b] = int(((qb < Q_FEAT) & (dl > c)).sum())
        pickle.dump({"pf": pf, "nsig": nsig, "samp": samp}, open(bp, "wb"))
        print(f"  B block {blk} done ({time.time() - t0:.0f}s)", flush=True)
    blocks = [pickle.load(open(CC / f"B_block{b}.pkl", "rb")) for b in range(NB // BLOCK)]
    pf = np.concatenate([b["pf"] for b in blocks]); samp = np.concatenate([b["samp"] for b in blocks])
    nsig = {c: np.concatenate([b["nsig"][c] for b in blocks]) for c in CUTOFFS}
    out["B_repartition"] = {
        "n_splits": NB, "seed": f"default_rng([{SEED + 31}, block]) for 4 blocks of 250",
        "KS_p_fixed_features": [float(st.kstest(pf[:, j], "uniform").pvalue) for j in range(5)],
        "frac_p_lt_0.05_fixed_features": [float((pf[:, j] < 0.05).mean()) for j in range(5)],
        "KS_p_random_feature_each_split": float(st.kstest(samp, "uniform").pvalue),
        "frac_random_feature_p_lt_0.05": float((samp < 0.05).mean()),
        "fwer_selection_by_cutoff": {str(c): float((nsig[c] > 0).mean()) for c in CUTOFFS},
        "fwer_selection_cp95_by_cutoff": {str(c): [float(x) for x in cp95(int((nsig[c] > 0).sum()), NB)] for c in CUTOFFS},
        "mean_n_selected_by_cutoff": {str(c): float(nsig[c].mean()) for c in CUTOFFS}}
    print("B", out["B_repartition"], flush=True)
    return "complete"


def cp95(k, n):
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def load_sets():
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    dor = dor[dor.chip_flag.astype(str).str.lower() == "true"]
    trr = pd.read_csv(TD / "targets_trrust.tsv", sep="\t", keep_default_na=False)
    g2f = top.groupby("feature_id").gene.apply(list).to_dict()

    def ovf(Sset):   # independent re-implementation of the overlap count (from the TSV, by gene name)
        return np.array([sum(g in Sset for g in g2f.get(f, [])) for f in range(D_SAE)])
    return dor, trr, ovf


def run_C(out, mw, P):
    res = pd.read_csv(OUT / "tf_results.csv")
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    dor, trr, ovf = load_sets()
    outC = {}
    tfsC = [t for t in meta["tfs_primary"]
            if int(res[(res.tf == t) & (res.cutoff == 0.25) & (res.db == "ChIP")].K.iloc[0]) > 0]
    for tf in tfsC:
        r = res[(res.tf == tf) & (res.cutoff == 0.25) & (res.db == "ChIP")].iloc[0]
        K = int(r.K); n = int(r.n_cells)
        ov = ovf(set(dor[dor.tf == tf].target))
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
        outC[tf] = {"K": K, "size": n, "values": [int(v) for v in vals], "counts_stored_vs_new": tab.tolist(),
                    "chi2_p": float(chi.pvalue) if chi is not None else None}
        print("C", tf, outC[tf], flush=True)
    out["C_fakeK_reproduction_cut0.25"] = outC


def run_D(out, mw, P, cpool, zpool, ipool, n_d, t0, max_min):
    res = pd.read_csv(OUT / "tf_results.csv")
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    dor, trr, ovf = load_sets()
    for ti, t in enumerate(meta["tfs_primary"]):
        dp = CC / f"D_{t}.pkl"
        if dp.exists():
            continue
        if (time.time() - t0) / 60 > max_min:
            print(f"STOPPED before D {t}", flush=True)
            return "incomplete_D"
        rd = np.random.default_rng([SEED + 8888, ti])
        rr = res[res.tf == t]
        n = int(rr.n_cells.iloc[0])
        K = {c: int(rr[rr.cutoff == c].K.iloc[0]) for c in CUTOFFS}
        sets = {"ChIP": set(dor[dor.tf == t].target), "TRRUST": set(trr[trr.tf == t].target)}
        ov = {db: ovf(Sx) for db, Sx in sets.items() if len(Sx)}
        zc = np.load(OUT / f"fake_null/size_{n:03d}.npz")
        obsT = defaultdict(list)
        for b in range(n_d):
            ix = rd.choice(len(P), n, replace=False)
            p = mw.test_full(cpool[ix].sum(0), n, zpool[ix].sum(0), ipool[ix].sum(0)); q = bh(p)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            ord_d = np.argsort(-dl, kind="stable"); ord_p = np.lexsort((-dl, p))
            for c in CUTOFFS:
                Rb = np.where((q < Q_FEAT) & (dl > c))[0]
                for db in ov:
                    for v, RR in [("fake", Rb), ("fakeK", (ord_d if c > 0 else ord_p)[:K[c]])]:
                        if v == "fakeK" and K[c] == 0:
                            continue
                        obsT[(c, db, v)].append(int(cap_T(ov[db][RR].max())) if len(RR) else 0)
        recs = {}
        for key, Tl in obsT.items():
            c, db, v = key
            Tl = np.array(Tl); Tn = cap_T(zc[f"M|{t}|{c}|{db}|{v}"])
            u = rd.random(len(Tl))
            gt = np.array([(Tn > x).sum() for x in Tl]); eq = np.array([(Tn == x).sum() for x in Tl])
            vals = sorted(set(Tl) | set(Tn))
            tab = np.array([[int((Tn == x).sum()), int((Tl == x).sum())] for x in vals])
            recs[key] = {"prand": list((gt + u * (eq + 1)) / (len(Tn) + 1)),
                         "pcons": list(np.where(Tl >= 2, (1 + gt + eq) / (len(Tn) + 1), 1.0)),
                         "chi": float(st.chi2_contingency(tab).pvalue) if len(vals) > 1 else None}
        pickle.dump(recs, open(dp, "wb"))
        print(f"  D {t} done ({time.time() - t0:.0f}s)", flush=True)
    prand, pcons, chi = defaultdict(list), defaultdict(list), defaultdict(list)
    for t in meta["tfs_primary"]:
        for key, d in pickle.load(open(CC / f"D_{t}.pkl", "rb")).items():
            prand[key].extend(d["prand"]); pcons[key].extend(d["pcons"])
            if d["chi"] is not None:
                chi[key].append(d["chi"])
    outD = {"n_per_tf": n_d, "seed": f"default_rng([{SEED + 8888}, TF index in primary list])"}
    for key in sorted(prand, key=str):
        c, db, v = key
        pr = np.array(prand[key]); pc = np.array(pcons[key])
        outD[f"{c}|{db}|{v}"] = {"n": int(len(pr)), "KS_p_randomized_pooled": float(st.kstest(pr, "uniform").pvalue),
                                 "frac_p_conservative_le_0.05": float((pc <= 0.05).mean()),
                                 "n_tfs_with_varying_T": len(chi[key]),
                                 "chi2_two_sample_p_min": float(min(chi[key])) if chi[key] else None,
                                 "chi2_two_sample_n_lt_0.05": int(sum(x < 0.05 for x in chi[key]))}
    out["D_calibration_1000_per_tf"] = outD
    return "complete"


if __name__ == "__main__":
    main()
