"""Independent check of the v2 degree-preserving null (TRRUST endpoint).

1. Re-derive z and Phipson-Smyth p from the agent's saved null draws
   (outputs/v2_eval/curveball/null_curveball_<run>.npy, column 0 = pooled AUROC).
2. Own sampler, written separately: Curveball on a boolean matrix (rows = evaluated TFs,
   columns = the 1,500 genes; cell (TF, own gene) forbidden). Each draw = 1,000 trades from
   the observed matrix; 1,000 draws; seed 23+run_index. Every draw is checked for row and
   column sums and forbidden cells; identical draws and Jaccard distance are counted.
3. On the same null networks, score other pair scores to see what "beats the degree
   null" can mean:
     attention at every stored layer (max-over-layers test, as v2_08),
     |Spearman| co-expression in the same control cells,
     expression-level proximity  -|mean_expr(TF) - mean_expr(gene)|  (no model; two
     gene-level numbers per pair).
p = (1 + #null >= observed) / (1 + n_draws), one-sided. z uses the null SD (ddof=1).
Output: outputs/v2_eval/verification/null_check.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
VER = OUT / "v2_eval" / "verification"
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
N_DRAWS = 1000
N_TRADES = 1000


def curveball_bool(M, forb_col, n_trades, rng):
    M = M.copy()
    n = M.shape[0]
    for _ in range(n_trades):
        a, b = rng.choice(n, 2, replace=False)
        only_a = M[a] & ~M[b]
        only_b = M[b] & ~M[a]
        # columns that cannot move: a's exclusive column equal to b's own gene, and vice versa
        stay_a = np.zeros_like(only_a); stay_a[forb_col[b]] = only_a[forb_col[b]]
        stay_b = np.zeros_like(only_b); stay_b[forb_col[a]] = only_b[forb_col[a]]
        movable = np.flatnonzero((only_a & ~stay_a) | (only_b & ~stay_b))
        if len(movable) < 2:
            continue
        ka = int(only_a.sum() - stay_a.sum())
        perm = rng.permutation(movable)
        M[a, movable] = False; M[b, movable] = False
        M[a, perm[:ka]] = True; M[b, perm[ka:]] = True
    return M


class PooledAUC:
    def __init__(self, S, tfs):
        G = S.shape[1]
        self.keep = np.ones_like(S, dtype=bool)
        self.keep[np.arange(len(tfs)), tfs] = False
        self.r = rankdata(S[self.keep], method="average")

    def __call__(self, M):
        y = M[self.keep]
        P = y.sum(); N = len(y) - P
        return float((self.r[y].sum() - P * (P + 1) / 2) / (P * N))


def main(selected):
    import sys as _s
    with open(TRRUST) as f:
        pairs = [(r[0].upper(), r[1].upper()) for r in csv.reader(f, delimiter="\t") if len(r) >= 2]
    agent = json.load(open(OUT / "v2_eval/curveball/curveball_summary.json"))
    agent_layers = json.load(open(OUT / "v2_eval/layers/layers_summary.json"))
    res = {}
    for ri, (run, sfx) in enumerate(RUNS.items()):
        part = VER / f"null_check_{run}.json"
        if run not in selected or part.exists():
            continue
        gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv")
        G = len(gf)
        idx = {s.upper(): i for i, s in enumerate(gf["symbol"])}
        edges = {(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b}
        outdeg = np.bincount([a for a, _ in edges], minlength=G)
        tfs = np.array(sorted({idx[a] for a, _ in pairs if a in idx and outdeg[idx[a]] >= 3}))
        nT = len(tfs)
        M0 = np.zeros((nT, G), dtype=bool)
        for k, t in enumerate(tfs):
            M0[k, [b for a, b in edges if a == t]] = True
        rs0, cs0 = M0.sum(1), M0.sum(0)
        # scores
        arr = np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r")
        L = arr.shape[0]
        layer_sc = []
        for li in range(L):
            A = np.array(arr[li], dtype=np.float64); np.fill_diagonal(A, 0.0)
            layer_sc.append(PooledAUC(A[tfs], tfs))
        del arr
        sp = np.abs(np.load(OUT / f"phase0{sfx}/spearman_edges.npy").astype(np.float64)); np.fill_diagonal(sp, 0.0)
        gm = gf["mean_expr"].to_numpy(np.float64)
        prox = -np.abs(gm[tfs][:, None] - gm[None, :])
        other = {"spearman_abs": PooledAUC(sp[tfs], tfs), "expr_proximity": PooledAUC(prox, tfs)}
        obs_layers = np.array([s(M0) for s in layer_sc])
        obs_other = {k: s(M0) for k, s in other.items()}
        # 1. agent's saved draws
        nv = np.load(OUT / f"v2_eval/curveball/null_curveball_{run}.npy")[:, 0]
        o8 = obs_layers[8]
        agent_re = {"n": int(len(nv)), "z": float((o8 - nv.mean()) / nv.std(ddof=1)),
                    "p": float((1 + (nv >= o8).sum()) / (1 + len(nv))),
                    "agent_z": agent[run]["curveball"]["z"], "agent_p": agent[run]["curveball"]["p_one_sided_phipson_smyth"]}
        # 2. own sampler
        rng = np.random.default_rng(23 + ri)
        nl = np.empty((N_DRAWS, L)); no = {k: np.empty(N_DRAWS) for k in other}
        bad = ident = 0; jac = []
        e0 = set(zip(*np.nonzero(M0)))
        for d in range(N_DRAWS):
            M = curveball_bool(M0, tfs, N_TRADES, rng)
            if not (np.array_equal(M.sum(1), rs0) and np.array_equal(M.sum(0), cs0) and not M[np.arange(nT), tfs].any()):
                bad += 1
            e = set(zip(*np.nonzero(M)))
            j = 1 - len(e & e0) / len(e | e0); jac.append(j); ident += int(j == 0)
            nl[d] = [s(M) for s in layer_sc]
            for k, s in other.items():
                no[k][d] = s(M)
        zp = lambda nv_, o: (float((o - nv_.mean()) / nv_.std(ddof=1)), float((1 + (nv_ >= o).sum()) / (1 + len(nv_))))
        z8, p8 = zp(nl[:, 8], o8)
        mu, sd = nl.mean(0), nl.std(0, ddof=1)
        zl = (obs_layers - mu) / sd
        zn = (nl - mu) / sd
        p_max = float((1 + (zn.max(1) >= zl.max()).sum()) / (1 + N_DRAWS))
        res[run] = {
            "n_tfs": int(nT), "n_edges": int(M0.sum()),
            "agent_draws_rederived": agent_re,
            "own_sampler": {"n_draws": N_DRAWS, "n_trades": N_TRADES, "draws_failing_checks": bad,
                            "draws_identical": ident, "mean_jaccard_distance": float(np.mean(jac)),
                            "layer8": {"observed": float(o8), "null_mean": float(nl[:, 8].mean()),
                                       "null_sd": float(nl[:, 8].std(ddof=1)), "z": z8, "p": p8},
                            "per_layer_z": zl.tolist(), "per_layer_observed": obs_layers.tolist(),
                            "max_z_over_layers": float(zl.max()), "argmax_layer": int(np.argmax(zl)),
                            "p_max_over_layers": p_max,
                            "agent_p_max_over_layers": agent_layers[run]["p_max_over_layers"],
                            "other_scores": {k: {"observed": obs_other[k], "null_mean": float(no[k].mean()),
                                                 "null_sd": float(no[k].std(ddof=1)),
                                                 "z": zp(no[k], obs_other[k])[0], "p": zp(no[k], obs_other[k])[1]}
                                             for k in other}},
        }
        print(run, json.dumps(res[run])[:1500], flush=True)
        json.dump(res[run], open(part, "w"), indent=1)


def combine():
    res = {r: json.load(open(VER / f"null_check_{r}.json")) for r in RUNS}
    res["_method"] = {"sampler": "own boolean-matrix Curveball, forbidden (TF, own gene) cells, 1000 trades/draw from observed, 1000 draws, seed 23+run_index",
                      "p": "Phipson-Smyth one-sided", "z": "null SD ddof=1",
                      "max_over_layers": "z per layer standardised by own null; max over layers vs same max on each null draw"}
    json.dump(res, open(VER / "null_check.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a == ["combine"]:
        combine()
    else:
        main(a or list(RUNS))
