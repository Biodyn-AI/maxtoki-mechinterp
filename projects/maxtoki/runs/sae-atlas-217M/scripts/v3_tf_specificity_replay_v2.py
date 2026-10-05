"""V3-2 code check: run the v3 statistics code on the v2 inputs and compare with v2 tf_results.csv.

The v3 stats script (v3_tf_specificity_stats.py) is a copy of the v2 one with new paths and
resumable stages. If the copy is faithful, feeding it the v2 per-cell feature means and the v2
task data must give back the v2 numbers. Checked: K (responding features), M_obs, T_obs,
best_feature, p_rf, p_cm, p_cml, p_cm_uncapped, E_M_cm, and p_fake / p_fakeK from the stored v2
fake-TF draws, for every TF x cut-off x database row of v2.
CPU only; reads v2 outputs, writes outputs/v3_tf_specificity/checks/replay_v2.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import v3_tf_specificity_stats as S  # noqa: E402

V2 = S.RUN / "v2_tf_specificity"
OUTF = S.OUT / "checks/replay_v2.json"


def main():
    t0 = time.time()
    man = pd.read_csv(V2 / "cell_manifest.csv", keep_default_na=False)
    meta = json.load(open(V2 / "cell_manifest_meta.json"))
    z = np.load(V2 / "task_data/cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]

    def mat(rr):
        return MG[[rix[int(r)] for r in rr]]

    R = mat(man[man.group == "ref"].row.values)
    tfs = meta["tfs_all"]
    KD = {t: mat(man[(man.group == "kd") & (man.tf == t)].row.values) for t in tfs}
    uni, genes, gidx, NG, top_idx, has_special, sets, TOP = S.load_task(V2 / "task_data")
    swap = {"cm": S.SwapNull(uni.count_bin.values.astype(int), top_idx),
            "cml": S.SwapNull(uni.count_x_length_bin.values.astype(int), top_idx)}
    mw, sel = S.select_all(R, KD, tfs)
    v2 = pd.read_csv(V2 / "tf_results.csv")
    out_rows = []
    fake_cache = {}
    lc_cache = {}
    for _, r in v2.iterrows():
        if not r.get("testable", False) or r.n_targets_in_universe == 0:
            continue
        t, c, db = r.tf, r.cutoff, r.db
        Rr = S.responding(sel[t], c)
        smask = sets[db][t]
        ov = TOP[:, smask].sum(1).astype(np.int64)
        M = int(ov[Rr].max()) if len(Rr) else 0
        T = int(S.cap_T(M))
        row = {"tf": t, "cutoff": c, "db": db, "K": len(Rr), "M_obs": M, "T_obs": T,
               "best_feature": int(Rr[np.argmax(ov[Rr])]) if len(Rr) else -1,
               "p_rf": S.p_randfeat(T, len(Rr), ov)}
        for nk in ["cm", "cml"]:
            if (nk, db, t) not in lc_cache:
                lc_cache[(nk, db, t)] = S.log_cdf_lt(swap[nk].pmf_all(smask))
            lc = lc_cache[(nk, db, t)]
            lsum = lc[Rr].sum(0)
            row[f"p_{nk}"] = S.p_swap(T, lsum)
            row[f"p_{nk}_uncapped"] = S.p_swap(M, lsum) if M >= 2 else 1.0
            row[f"E_M_{nk}"] = float(sum(-np.expm1(lsum[k]) for k in range(1, 21)))
        n = sel[t]["n"]
        if n not in fake_cache:
            fake_cache[n] = np.load(V2 / f"fake_null/size_{n:03d}.npz")
        for v in ["fake", "fakeK"]:
            row[f"p_{v}"] = S.p_emp(T, S.cap_T(fake_cache[n][f"M|{t}|{c}|{db}|{v}"]))
        out_rows.append(row)
    mine = pd.DataFrame(out_rows)
    j = mine.merge(v2, on=["tf", "cutoff", "db"], suffixes=("_v3code", "_v2"))
    rep = {"n_rows_compared": int(len(j))}
    for col in ["K", "M_obs", "T_obs", "best_feature"]:
        rep[f"{col}_n_mismatch"] = int((j[f"{col}_v3code"].astype(float) != j[f"{col}_v2"].astype(float)).sum())
    for col in ["p_rf", "p_cm", "p_cml", "p_cm_uncapped", "E_M_cm", "p_fake", "p_fakeK"]:
        rep[f"{col}_max_abs_diff"] = float(np.nanmax(np.abs(j[f"{col}_v3code"].astype(float) - j[f"{col}_v2"].astype(float))))
    rep["pass"] = bool(all(v == 0 for k, v in rep.items() if k.endswith("_n_mismatch")) and
                       all(v < 1e-9 for k, v in rep.items() if k.endswith("_max_abs_diff")))
    rep["wall_s"] = round(time.time() - t0, 1)
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    S.H.write_json(OUTF, rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
