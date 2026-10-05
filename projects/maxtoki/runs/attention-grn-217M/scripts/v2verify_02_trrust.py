"""Independent verification of the v2 TRRUST endpoint (Endpoint B): raw attention,
variance baseline, 3-feature gene model, OLS-residualised attention, TF-bootstrap CIs.

Does not import v2_common. TRRUST parsed with the csv module. AUROC = sklearn.
Bootstrap = explicit replication: resample evaluated TFs with replacement (2,000 reps,
seed 11+run_index) and concatenate the drawn TFs' pairs; sklearn AUROC on the copy.
Gene model, two routes: (a) GroupKFold(5) by TF (as v2); (b) leave-one-TF-out.
OLS residualisation re-implemented with numpy lstsq (intercept + 5 features) under the
deployed KFold(5, shuffle=True, random_state=42) over all ordered off-diagonal pairs.
Output: outputs/v2_eval/verification/trrust_check.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
VER = OUT / "v2_eval" / "verification"
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
NB = 2000


def trrust_pairs():
    with open(TRRUST) as f:
        return [(r[0].upper(), r[1].upper()) for r in csv.reader(f, delimiter="\t") if len(r) >= 2]


def gene_model(X, y, groups, splitter):
    oof = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    pairs = trrust_pairs()
    agent_tr = json.load(open(OUT / "v2_eval/trrust/trrust_summary.json"))
    agent_rs = json.load(open(OUT / "v2_eval/residualised/residualised_summary.json"))
    res = {}
    for ri, (run, sfx) in enumerate(RUNS.items()):
        gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv")
        G = len(gf)
        idx = {s.upper(): i for i, s in enumerate(gf["symbol"])}
        edges = {(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b}
        outdeg = np.bincount([a for a, _ in edges], minlength=G)
        tfs = np.array(sorted(set(idx[a] for a, _ in pairs if a in idx and outdeg[idx[a]] >= 3)))
        src = np.repeat(tfs, G - 1)
        cand = np.concatenate([np.array([g for g in range(G) if g != t]) for t in tfs])
        grp = np.repeat(np.arange(len(tfs)), G - 1)
        y = np.array([(s, c) in edges for s, c in zip(src, cand)], dtype=int)
        A = np.array(np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float64)
        np.fill_diagonal(A, 0.0)
        gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
        X3 = np.column_stack([gm[cand], gv[cand], gd[cand]])
        sc = {"attention": A[src, cand], "variance": gv[cand],
              "gene3_gkf5": gene_model(X3, y, grp, GroupKFold(5)),
              "gene3_loto": gene_model(X3, y, grp, LeaveOneGroupOut())}
        # OLS residualisation (numpy), deployed fold scheme
        ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        mk = ii != jj
        s_all, t_all = ii[mk], jj[mk]
        del ii, jj
        Xf = np.column_stack([np.ones(len(s_all)), gm[s_all], gv[s_all], gm[t_all], gv[t_all], gd[t_all]])
        yf = A[s_all, t_all]
        resid = np.empty_like(yf)
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xf):
            beta, *_ = np.linalg.lstsq(Xf[tr], yf[tr], rcond=None)
            resid[te] = yf[te] - Xf[te] @ beta
        R = np.zeros((G, G)); R[s_all, t_all] = resid
        del Xf, yf, resid, s_all, t_all
        sc["resid_ols"] = R[src, cand]
        point = {k: float(roc_auc_score(y, v)) for k, v in sc.items()}
        # explicit-replication bootstrap over TFs
        rng = np.random.default_rng(11 + ri)
        nT = len(tfs)
        pos_by_tf = [np.flatnonzero(grp == k) for k in range(nT)]
        boot = {k: [] for k in sc}
        for _ in range(NB):
            d = rng.integers(0, nT, nT)
            ix = np.concatenate([pos_by_tf[k] for k in d])
            yy = y[ix]
            for k, v in sc.items():
                boot[k].append(roc_auc_score(yy, v[ix]))
        boot = {k: np.array(v) for k, v in boot.items()}
        pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        out = {"n_tfs": int(nT), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "n_edges_in_gene_set": len(edges),
               "point": point, "ci": {k: pct(v) for k, v in boot.items()},
               "gap_variance_minus_attention": [point["variance"] - point["attention"], pct(boot["variance"] - boot["attention"])],
               "gap_gene3_minus_attention": [point["gene3_gkf5"] - point["attention"], pct(boot["gene3_gkf5"] - boot["attention"])],
               "gap_gene3loto_minus_attention": [point["gene3_loto"] - point["attention"], pct(boot["gene3_loto"] - boot["attention"])],
               "drop_raw_minus_resid_ols": [point["attention"] - point["resid_ols"], pct(boot["attention"] - boot["resid_ols"])],
               "agent": {"pooled": agent_tr[run]["pooled_auroc"], "ci": agent_tr[run]["pooled_ci95"],
                         "gaps": {k: agent_tr[run]["gaps"][k] for k in ["variance_minus_attention", "gene3_logreg_oof_minus_attention"]},
                         "ols5": agent_rs[run]["ols5"]["residualised_auroc"], "ols5_ci": agent_rs[run]["ols5"]["ci95"]}}
        res[run] = out
        print(run, json.dumps({k: out[k] for k in ["n_tfs", "n_pos", "point", "gap_variance_minus_attention",
                                                    "gap_gene3_minus_attention", "gap_gene3loto_minus_attention"]}), flush=True)
    res["_method"] = {"bootstrap": "explicit replication over TFs, percentile, 2000 reps, seed 11+run_index",
                      "gene3": "logistic regression (standardised, balanced) on candidate mean, variance, dropout",
                      "ols": "numpy lstsq, intercept + [src mean, src var, tgt mean, tgt var, tgt dropout], KFold(5, shuffle, 42)"}
    json.dump(res, open(VER / "trrust_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
