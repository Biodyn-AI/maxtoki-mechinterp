"""v2 step 6 — a degree-preserving null for the TRRUST endpoint that actually rewires.

Network = bipartite 0/1 matrix M: rows = the evaluated TFs (>= 3 targets in the gene set),
columns = the run's 1,500 genes, M[k, g] = 1 if TRRUST lists TF_k -> g (g != TF_k).
Row sums (targets per TF) and column sums (how many evaluated TFs list the gene) are kept.
Cell (k, own gene of TF_k) is a structural zero: it is never scored, so it may never
receive an edge.

Null 1 (primary): Curveball (Strona et al. 2014). One trade = pick two TF rows at random;
their non-shared targets are pooled and re-dealt at random, each row keeping its count;
a target that is the other TF's own gene stays where it is. Each null draw is an
independent chain started from the observed M with N_TRADES trades.
Null 2 (check): checkerboard swaps (Gotelli 2000): pick two edges (a,x), (b,y); if (a,y)
and (b,x) are empty and allowed, replace with (a,y), (b,x). SWAPS_PER_EDGE x n_edges
accepted swaps per draw.
Statistic: pooled attention AUROC over all (evaluated TF, candidate) pairs, exactly as in
the deployed evaluation. Secondary statistic: mean over TFs of the per-TF AUROC.
z = (observed - null mean) / null SD; p = (1 + #null >= observed) / (1 + n_draws)
(Phipson & Smyth 2010), one-sided (attention better than the null). BH across the 4 runs.

Verification per run: every draw keeps all row and column sums; no structural zero is
filled; number of draws identical to the observed network; mean Jaccard distance from
the observed edge set; the variance baseline's pooled AUROC must be exactly unchanged in
every draw (it depends only on column sums). A mixing curve (Jaccard and null AUROC vs
number of trades) is written too. The deployed null code is re-run verbatim for contrast.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (RUNS, V2, SEED, load_gene_features, load_attention_layer,  # noqa: E402
                       load_trrust, trrust_matrix, evaluated_tf_rows, auroc_rank, bh, save_json)

OUTD = V2 / "curveball"
OUTD.mkdir(parents=True, exist_ok=True)
N_DRAWS = 1000
N_TRADES = 5000
N_DRAWS_SWAP = 1000
SWAPS_PER_EDGE = 30
MIX_TRADES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
MIX_CHAINS = 100


def curveball(rows: list[set], forb: np.ndarray, n_trades: int, rng) -> list[set]:
    rows = [set(r) for r in rows]
    n = len(rows)
    aa = rng.integers(0, n, n_trades)
    bb = rng.integers(0, n - 1, n_trades)
    bb = bb + (bb >= aa)                      # uniform over ordered pairs a != b
    for a, b in zip(aa.tolist(), bb.tolist()):
        Sa, Sb = rows[a], rows[b]
        Da, Db = Sa - Sb, Sb - Sa
        if not Da or not Db:
            continue
        fixed_a = {c for c in Da if c == forb[b]}
        fixed_b = {c for c in Db if c == forb[a]}
        pool = list((Da - fixed_a) | (Db - fixed_b))
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        ka = len(Da) - len(fixed_a)
        common = Sa & Sb
        rows[a] = common | fixed_a | set(pool[:ka])
        rows[b] = common | fixed_b | set(pool[ka:])
    return rows


def checkerboard(rows: list[set], forb: np.ndarray, n_swaps: int, rng, max_attempts: int) -> list[set]:
    rows = [set(r) for r in rows]
    edges = [(k, c) for k, r in enumerate(rows) for c in r]
    E = len(edges)
    done = att = 0
    buf = []
    while done < n_swaps and att < max_attempts:
        att += 1
        if not buf:
            buf = rng.integers(0, E, (4096, 2)).tolist()
        i, j = buf.pop()
        (a, x), (b, y) = edges[i], edges[j]
        if a == b or x == y or y in rows[a] or x in rows[b] or y == forb[a] or x == forb[b]:
            continue
        rows[a].remove(x); rows[a].add(y); rows[b].remove(y); rows[b].add(x)
        edges[i] = (a, y); edges[j] = (b, x)
        done += 1
    return rows


def deployed_curveball(mat, n_iter, seed):
    """Verbatim copy of phase3_residualization.py:221-246 (for contrast only)."""
    rng = np.random.default_rng(seed)
    rows = [set(np.where(r)[0]) for r in mat.astype(bool)]
    G = len(rows)
    for _ in range(n_iter):
        i, j = rng.choice(G, 2, replace=False)
        a, b = rows[i], rows[j]
        inter = a & b
        sym = (a | b) - inter
        if len(sym) < 2:
            continue
        sym_list = list(sym)
        rng.shuffle(sym_list)
        k = len(a) - len(inter)
        rows[i] = inter | set(sym_list[:k])
        rows[j] = inter | set(sym_list[k:])
    out = np.zeros_like(mat)
    for r_idx, cols in enumerate(rows):
        for c in cols:
            out[r_idx, c] = 1
    return out


class Scorer:
    """Pooled AUROC (and per-TF mean AUROC) of a fixed score for any label set."""

    def __init__(self, score_rows: np.ndarray, tf_rows: np.ndarray, G: int):
        self.tf_rows, self.G = tf_rows, G
        pooled = np.concatenate([np.delete(score_rows[k], tf_rows[k]) for k in range(len(tf_rows))])
        self.rank = rankdata(pooled, method="average")
        self.rank_row = [rankdata(np.delete(score_rows[k], tf_rows[k]), method="average")
                         for k in range(len(tf_rows))]
        self.N = len(pooled)

    def idx(self, k, cols):
        cols = np.asarray(sorted(cols), dtype=np.int64)
        return cols - (cols > self.tf_rows[k])

    def __call__(self, rows: list[set]):
        pos, per_tf = [], []
        for k, r in enumerate(rows):
            ii = self.idx(k, r)
            pos.append(k * (self.G - 1) + ii)
            n1 = len(ii); n0 = self.G - 1 - n1
            per_tf.append((self.rank_row[k][ii].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        pos = np.concatenate(pos)
        P = len(pos); Nn = self.N - P
        return float((self.rank[pos].sum() - P * (P + 1) / 2) / (P * Nn)), float(np.mean(per_tf))


def jaccard_dist(rows, obs_set):
    s = {(k, c) for k, r in enumerate(rows) for c in r}
    return 1.0 - len(s & obs_set) / len(s | obs_set)


def main(selected):
    import json
    trrust = load_trrust()
    for ri, run in enumerate(RUNS):
        if run not in selected or (OUTD / f"curveball_{run}.json").exists():
            continue
        summary, mix_rows = {}, []
        t0 = time.time()
        gf = load_gene_features(run)
        G = len(gf)
        sym = [s.upper() for s in gf["symbol"]]
        E = trrust_matrix(sym, trrust)
        tf_rows = evaluated_tf_rows(sym, trrust, E)
        nT = len(tf_rows)
        M = E[tf_rows].astype(np.int8)          # (nT, G)
        obs_rows = [set(np.flatnonzero(M[k]).tolist()) for k in range(nT)]
        obs_set = {(k, c) for k, r in enumerate(obs_rows) for c in r}
        rs0 = M.sum(1); cs0 = M.sum(0)
        forb = tf_rows.copy()
        A = load_attention_layer(run); np.fill_diagonal(A, 0.0)
        att = Scorer(A[tf_rows].astype(np.float64), tf_rows, G)
        gvar = gf["variance"].to_numpy(np.float64)
        var = Scorer(np.tile(gvar, (nT, 1)), tf_rows, G)
        obs_auc, obs_tfmean = att(obs_rows)
        obs_var, _ = var(obs_rows)
        # sanity: Scorer equals the deployed-style pooled AUROC
        y = np.concatenate([np.delete(M[k], tf_rows[k]) for k in range(nT)]).astype(bool)
        s = np.concatenate([np.delete(A[tf_rows[k]], tf_rows[k]) for k in range(nT)])
        assert abs(auroc_rank(y, s) - obs_auc) < 1e-12

        def check(rows):
            Mm = np.zeros_like(M)
            for k, r in enumerate(rows):
                Mm[k, list(r)] = 1
            return (np.array_equal(Mm.sum(1), rs0) and np.array_equal(Mm.sum(0), cs0)
                    and not Mm[np.arange(nT), forb].any())

        # mixing curve (Curveball)
        rng_mix = np.random.default_rng(SEED + 650 + ri)
        for nt in MIX_TRADES:
            js, aus = [], []
            for _ in range(MIX_CHAINS):
                r = curveball(obs_rows, forb, nt, rng_mix)
                js.append(jaccard_dist(r, obs_set)); aus.append(att(r)[0])
            mix_rows.append({"run": run, "n_trades": nt, "mean_jaccard_distance": float(np.mean(js)),
                             "null_auroc_mean": float(np.mean(aus)), "null_auroc_sd": float(np.std(aus, ddof=1)),
                             "frac_identical": float(np.mean(np.array(js) == 0))})

        # main Curveball null
        rng = np.random.default_rng(SEED + 600 + ri)
        null, null_tf, jac, var_null = [], [], [], []
        n_ident = n_bad = 0
        for _ in range(N_DRAWS):
            r = curveball(obs_rows, forb, N_TRADES, rng)
            if not check(r):
                n_bad += 1
            j = jaccard_dist(r, obs_set)
            n_ident += int(j == 0)
            a_, t_ = att(r)
            null.append(a_); null_tf.append(t_); jac.append(j); var_null.append(var(r)[0])
        null = np.array(null); null_tf = np.array(null_tf); var_null = np.array(var_null)
        np.save(OUTD / f"null_curveball_{run}.npy", np.column_stack([null, null_tf, jac]))

        # check null: checkerboard swaps
        rng2 = np.random.default_rng(SEED + 700 + ri)
        n_edges = len(obs_set)
        sw, sw_tf, sw_jac = [], [], []
        n_bad_sw = 0
        for _ in range(N_DRAWS_SWAP):
            r = checkerboard(obs_rows, forb, SWAPS_PER_EDGE * n_edges, rng2, 10_000 * n_edges)
            if not check(r):
                n_bad_sw += 1
            a_, t_ = att(r)
            sw.append(a_); sw_tf.append(t_); sw_jac.append(jaccard_dist(r, obs_set))
        sw = np.array(sw); sw_tf = np.array(sw_tf)

        # deployed null re-run verbatim (50 draws, seeds 42..91, full HVG matrix)
        n_iter_dep = 5 * int(E.sum())
        dep_vals, dep_ident_eval, dep_ident_full, dep_self = [], 0, 0, 0
        for t in range(50):
            P = deployed_curveball(E, n_iter_dep, 42 + t)
            dep_ident_full += int(np.array_equal(P, E))
            dep_ident_eval += int(np.array_equal(P[tf_rows], E[tf_rows]))
            dep_self += int(P[np.arange(G), np.arange(G)].sum())
            yl = np.concatenate([np.delete(P[tf_rows[k]], tf_rows[k]) for k in range(nT)]).astype(bool)
            dep_vals.append(auroc_rank(yl, s))
        dep_vals = np.array(dep_vals)

        def zp(nv, o):
            return (float((o - nv.mean()) / nv.std(ddof=1)),
                    float((1 + (nv >= o).sum()) / (1 + len(nv))))
        z, p = zp(null, obs_auc)
        z_tf, p_tf = zp(null_tf, obs_tfmean)
        z_sw, p_sw = zp(sw, obs_auc)
        summary[run] = {
            "n_tfs": int(nT), "n_edges": int(n_edges), "n_columns_with_edges": int((cs0 > 0).sum()),
            "observed_pooled_auroc": obs_auc, "observed_per_tf_mean_auroc": obs_tfmean,
            "curveball": {
                "n_draws": N_DRAWS, "n_trades_per_draw": N_TRADES,
                "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                "null_q025_q975": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                "z": z, "p_one_sided_phipson_smyth": p,
                "per_tf_mean_statistic": {"null_mean": float(null_tf.mean()), "null_sd": float(null_tf.std(ddof=1)),
                                          "z": z_tf, "p": p_tf},
                "draws_identical_to_observed": int(n_ident),
                "draws_failing_margin_or_structural_zero_check": int(n_bad),
                "mean_jaccard_distance_from_observed": float(np.mean(jac)),
                "min_jaccard_distance_from_observed": float(np.min(jac)),
                "variance_baseline_auroc_observed": obs_var,
                "variance_baseline_auroc_null_max_abs_change": float(np.max(np.abs(var_null - obs_var))),
            },
            "checkerboard_check": {
                "n_draws": N_DRAWS_SWAP, "accepted_swaps_per_draw": SWAPS_PER_EDGE * n_edges,
                "null_mean": float(sw.mean()), "null_sd": float(sw.std(ddof=1)), "z": z_sw, "p": p_sw,
                "per_tf_mean_null_mean": float(sw_tf.mean()),
                "mean_jaccard_distance_from_observed": float(np.mean(sw_jac)),
                "draws_failing_margin_or_structural_zero_check": int(n_bad_sw),
            },
            "deployed_null_rerun": {
                "n_draws": 50, "n_iter": n_iter_dep, "matrix": "full 1,500 x 1,500 TRRUST matrix",
                "draws_identical_on_evaluated_rows": int(dep_ident_eval),
                "draws_identical_whole_matrix": int(dep_ident_full),
                "self_loops_created_total": int(dep_self),
                "null_mean": float(dep_vals.mean()), "null_sd_ddof0": float(dep_vals.std()),
                "z_as_deployed": float((obs_auc - dep_vals.mean()) / max(dep_vals.std(), 1e-12)),
            },
            "seconds": time.time() - t0,
        }
        print(run, summary[run], flush=True)
        pd.DataFrame(mix_rows).to_csv(OUTD / f"mixing_curve_{run}.csv", index=False)
        save_json(summary[run], OUTD / f"curveball_{run}.json")


def combine():
    import json
    runs = list(RUNS)
    summary = {r: json.load(open(OUTD / f"curveball_{r}.json")) for r in runs}
    q = bh(np.array([summary[r]["curveball"]["p_one_sided_phipson_smyth"] for r in runs]))
    q_tf = bh(np.array([summary[r]["curveball"]["per_tf_mean_statistic"]["p"] for r in runs]))
    for r, qq, qt in zip(runs, q, q_tf):
        summary[r]["curveball"]["p_bh_across_4_runs"] = float(qq)
        summary[r]["curveball"]["per_tf_mean_statistic"]["p_bh_across_4_runs"] = float(qt)
    summary["_method"] = {"seeds": {"curveball": SEED + 600, "checkerboard": SEED + 700, "mixing": SEED + 650},
                          "p": "Phipson-Smyth (1 + #null >= obs) / (1 + n), one-sided upper",
                          "z": "(obs - null mean) / null SD (ddof=1)", "bh": "across the 4 runs"}
    pd.concat([pd.read_csv(OUTD / f"mixing_curve_{r}.csv") for r in runs]).to_csv(OUTD / "mixing_curve.csv", index=False)
    save_json(summary, OUTD / "curveball_summary.json")
    for r in runs:
        c = summary[r]["curveball"]
        print(r, round(summary[r]["observed_pooled_auroc"], 4), "null", round(c["null_mean"], 4), round(c["null_sd"], 4),
              "z", round(c["z"], 2), "p", round(c["p_one_sided_phipson_smyth"], 4), "q", round(c["p_bh_across_4_runs"], 4))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combine"]:
        combine()
    else:
        main(args or list(RUNS))
