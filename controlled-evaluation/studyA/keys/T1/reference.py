"""Reference computation for Study A, task T1 (attention vs TRRUST, MaxToki-217M, RPE1).

Reads ONLY the package data (tasks/T1-paper/data/, identical to tasks/T1-contract/data/).
CPU only. Results are merged into reference_output.json next to this file.

Usage (each section is a separate command so that each stays well under 9 minutes):
    python reference.py main        # sections 1-6: file facts, AUROCs, baselines, TF and
                                    # two-way bootstrap, analyst choices, weak nulls,
                                    # logistic incremental value
    python reference.py null        # 7: degree-preserving null (Curveball), deployed score
    python reference.py null_alt    # 8: the same null for other pair scores
    python reference.py resid       # 9: cross-fitted residualised attention (OLS, HGB)
    python reference.py incr_hgb    # 10: incremental value with a non-linear gene model

Seeds follow the source re-evaluation (master seed 20261001; run index 1 = RPE1). With the
same seeds the TF-bootstrap CIs, the Curveball null of the deployed score and the
residualised scores reproduce the source report exactly.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, norm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "tasks" / "T1-paper" / "data"
OUT = HERE / "reference_output.json"
SEED = 20261001
RI = 1                      # run index of RPE1 in the source re-evaluation
N_BOOT = 2000
SECTIONS = sys.argv[1:] or ["main"]


# ------------------------------------------------------------------ helpers
def auroc(y, s):
    """Mann-Whitney AUROC with average ranks (tie-safe)."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s, method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class WeightedAUROC:
    """AUROC under integer re-weighting of a fixed (y, s) set (bootstrap replication)."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="mergesort")
        ss = np.asarray(s)[order]
        self.order = order
        self.y = np.asarray(y).astype(bool)[order]
        self.starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]

    def __call__(self, w=None):
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)[self.order]
        wp = np.where(self.y, w, 0.0); wn = w - wp
        gp = np.add.reduceat(wp, self.starts); gn = np.add.reduceat(wn, self.starts)
        before = np.cumsum(gn) - gn
        P, N = gp.sum(), gn.sum()
        return float((gp * (before + 0.5 * gn)).sum() / (P * N))


def pci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def r4(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r4(v) for v in x]
    return None if x is None else float(round(float(x), 5))


def save(section, obj):
    allres = json.load(open(OUT)) if OUT.exists() else {}
    allres[section] = obj
    json.dump(allres, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ------------------------------------------------------------------ load (all sections)
t0 = time.time()
A = np.load(DATA / "attention_layer8_headmean.npy").astype(np.float64)
C = np.load(DATA / "pair_cell_counts.npy")
genes = pd.read_csv(DATA / "genes.tsv", sep="\t")
gs = pd.read_csv(DATA / "gene_stats_control_cells.tsv", sep="\t")
tr = pd.read_csv(DATA / "trrust_edges_in_gene_set.tsv", sep="\t")
G = len(genes)
assert A.shape == (G, G) and (gs["index"].to_numpy() == np.arange(G)).all()
gm = gs["mean"].to_numpy(np.float64); gv = gs["variance"].to_numpy(np.float64)
gd = gs["dropout_rate"].to_numpy(np.float64)
A0 = A.copy(); np.fill_diagonal(A0, 0.0)                     # diagonal = self-attention, never scored
Asym = 0.5 * (A0 + A0.T)
ncell = np.diag(C).astype(np.float64)                          # cells in which each gene is present
LIFT = C * 2000.0 / np.outer(ncell, ncell); np.fill_diagonal(LIFT, 0.0)   # co-detection observed/expected

uniq = tr.drop_duplicates(["tf_index", "target_index"])
E = np.zeros((G, G), dtype=np.int8)
for i, j in zip(uniq["tf_index"], uniq["target_index"]):
    if i != j:
        E[i, j] = 1
outdeg = E.sum(1)


def design(min_targets):
    tf_rows = np.flatnonzero(outdeg >= min_targets)            # ascending gene index
    k_of = np.concatenate([np.full(G - 1, k) for k in range(len(tf_rows))])
    src = tf_rows[k_of]
    cand = np.concatenate([np.delete(np.arange(G), r) for r in tf_rows])
    y = E[src, cand].astype(bool)
    return tf_rows, k_of, src, cand, y


tf_rows, k_of, src, cand, y = design(3)
nT = len(tf_rows)
P = int(y.sum()); Nn = len(y) - P


def gene3_oof(extra=None, groups=None, model="lr"):
    X = np.column_stack([gm[cand], gv[cand], gd[cand]]).astype(np.float32)
    if extra is not None:
        X = np.column_stack([X, extra]).astype(np.float32)
    groups = k_of if groups is None else groups
    oof = np.zeros(len(y)); fold = []
    for trn, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
        if model == "lr":
            m = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        else:
            m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                               class_weight="balanced", random_state=0)
        m.fit(X[trn], y[trn])
        oof[te] = m.predict_proba(X[te])[:, 1]
        fold.append(float(roc_auc_score(y[te], oof[te])))
    return oof, fold


def incremental(extra, groups=None, model="lr", seed=3, n=1000):
    base, _ = gene3_oof(None, groups, model); full, _ = gene3_oof(extra, groups, model)
    fb, ff = WeightedAUROC(y, base), WeightedAUROC(y, full)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of]
        d.append(ff(w) - fb(w))
    return {"gene_only_auroc": r4(auroc(y, base)), "gene_plus_score_auroc": r4(auroc(y, full)),
            "delta": r4(auroc(y, full) - auroc(y, base)), "delta_ci95_tf_bootstrap_models_fixed": r4(pci(d))}


# ================================================================== main
if "main" in SECTIONS:
    res: dict = {}
    self_rows = tr["tf_index"] == tr["target_index"]
    res["trrust_file"] = {
        "rows": int(len(tr)), "unique_pairs": int(len(uniq)), "self_rows": int(self_rows.sum()),
        "unique_self_pairs": int((uniq["tf_index"] == uniq["target_index"]).sum()),
        "unique_non_self_edges": int(E.sum()), "tfs_with_any_target": int((outdeg > 0).sum()),
        "tfs_with_ge3_targets": int((outdeg >= 3).sum()),
        "mode_counts_rows": tr["mode"].value_counts().to_dict()}
    res["design"] = {"rule": "TFs with >=3 unique non-self TRRUST targets in the gene set; candidates = the other 1,499 genes; positives = TRRUST targets; all pairs pooled",
                     "n_tfs": int(nT), "tfs": genes["symbol"].to_numpy()[tf_rows].tolist(),
                     "n_positive_pairs": P, "n_negative_pairs": Nn,
                     "n_distinct_target_genes": int(len(np.unique(cand[y]))),
                     "targets_per_tf": E[tf_rows].sum(1).tolist()}
    res["data_facts"] = {
        "diag_mean_attention": r4(np.diag(A).mean()), "offdiag_mean_attention": r4(A0.sum() / (G * G - G)),
        "pairs_never_co_observed": int((C == 0).sum()),
        "frac_offdiag_attention_exact_zero": r4(((A0 == 0).sum() - G) / (G * G - G))}

    scores = {
        "attention_tf_query": A0[src, cand],                    # deployed: row = TF (query), column = candidate (key)
        "attention_tf_key": A0[cand, src],                      # candidate (query) -> TF (key)
        "attention_sym_mean": Asym[src, cand],
        "attention_sym_max": np.maximum(A0[src, cand], A0[cand, src]),
        "variance": gv[cand], "mean": gm[cand], "one_minus_dropout": 1.0 - gd[cand],
        "expr_proximity": -np.abs(gm[src] - gm[cand]),          # no model: closeness of mean expression
        "co_detection_lift": LIFT[src, cand],                   # no model: C_ij * 2000 / (C_ii * C_jj)
        "pair_cell_count": C[src, cand].astype(np.float64),
    }
    oof, fold_auc = gene3_oof()
    scores["gene3_logreg_oof"] = oof
    pooled = {k: auroc(y, s) for k, s in scores.items()}
    per_tf = {k: np.array([auroc(y[k_of == t], s[k_of == t]) for t in range(nT)]) for k, s in scores.items()}

    fast = {k: WeightedAUROC(y, s) for k, s in scores.items()}
    rng = np.random.default_rng(SEED + 500 + RI)                # same draw sequence as the source
    boot = {k: [] for k in scores}; boot_tf = {k: [] for k in scores}
    for _ in range(N_BOOT):
        draw = rng.integers(0, nT, nT)
        w = np.bincount(draw, minlength=nT)[k_of]
        for k in scores:
            boot[k].append(fast[k](w)); boot_tf[k].append(float(np.nanmean(per_tf[k][draw])))
    boot = {k: np.array(v) for k, v in boot.items()}; boot_tf = {k: np.array(v) for k, v in boot_tf.items()}
    res["pooled_auroc"] = {k: {"value": r4(pooled[k]), "ci95_tf_bootstrap": r4(pci(boot[k]))} for k in scores}
    res["per_tf_mean_auroc"] = {k: {"value": r4(np.nanmean(per_tf[k])), "ci95_tf_bootstrap": r4(pci(boot_tf[k]))}
                                for k in scores}
    res["gene3_mean_of_fold_aurocs"] = r4(np.mean(fold_auc))
    gaps = {}
    for b in ["variance", "gene3_logreg_oof"]:
        for a in ["attention_tf_query", "attention_sym_mean"]:
            d = boot[b] - boot[a]
            gaps[f"{b}_minus_{a}"] = {"value": r4(pooled[b] - pooled[a]), "ci95_tf_bootstrap": r4(pci(d)),
                                      "frac_boot_gt0": r4((d > 0).mean())}
            dt = boot_tf[b] - boot_tf[a]
            gaps[f"{b}_minus_{a}_per_tf_mean"] = {"value": r4(np.nanmean(per_tf[b]) - np.nanmean(per_tf[a])),
                                                  "ci95_tf_bootstrap": r4(pci(dt))}
    res["gaps"] = gaps
    res["n_tfs_variance_gt_attention"] = int((per_tf["variance"] > per_tf["attention_tf_query"]).sum())
    res["n_tfs_gene3_gt_attention"] = int((per_tf["gene3_logreg_oof"] > per_tf["attention_tf_query"]).sum())
    res["n_tfs_attention_auroc_gt_0.5"] = int((per_tf["attention_tf_query"] > 0.5).sum())
    res["per_tf_table"] = pd.DataFrame({
        "tf": genes["symbol"].to_numpy()[tf_rows], "n_targets": E[tf_rows].sum(1),
        "attention_tf_query": per_tf["attention_tf_query"].round(4),
        "attention_sym_mean": per_tf["attention_sym_mean"].round(4),
        "variance": per_tf["variance"].round(4), "gene3": per_tf["gene3_logreg_oof"].round(4)}).to_dict(orient="records")

    # ---- other units of resampling
    rng_p = np.random.default_rng(7)
    fa, fv, fg, fs = fast["attention_tf_query"], fast["variance"], fast["gene3_logreg_oof"], fast["attention_sym_mean"]
    pair_b, two_b = [], []
    for _ in range(1000):
        w = np.bincount(rng_p.integers(0, len(y), len(y)), minlength=len(y))
        pair_b.append((fa(w), fv(w) - fa(w), fg(w) - fa(w), fs(w)))
        w2 = np.bincount(rng_p.integers(0, nT, nT), minlength=nT)[k_of] * np.bincount(rng_p.integers(0, G, G), minlength=G)[cand]
        two_b.append((fa(w2), fv(w2) - fa(w2), fg(w2) - fa(w2), fs(w2)))
    pair_b, two_b = np.array(pair_b), np.array(two_b)
    a_ = pooled["attention_tf_query"]
    q1 = a_ / (2 - a_); q2 = 2 * a_ ** 2 / (1 + a_)
    se_hm = np.sqrt((a_ * (1 - a_) + (P - 1) * (q1 - a_ ** 2) + (Nn - 1) * (q2 - a_ ** 2)) / (P * Nn))
    res["other_resampling_units"] = {
        "pair_bootstrap_1000": {"attention_tf_query": r4(pci(pair_b[:, 0])), "variance_minus_attention": r4(pci(pair_b[:, 1])),
                                "gene3_minus_attention": r4(pci(pair_b[:, 2])), "attention_sym_mean": r4(pci(pair_b[:, 3]))},
        "two_way_tf_and_gene_bootstrap_1000": {"attention_tf_query": r4(pci(two_b[:, 0])),
                                               "variance_minus_attention": r4(pci(two_b[:, 1])),
                                               "gene3_minus_attention": r4(pci(two_b[:, 2])),
                                               "attention_sym_mean": r4(pci(two_b[:, 3]))},
        "hanley_mcneil_pairs_independent": {"se": r4(se_hm), "z_vs_0.5": r4((a_ - 0.5) / se_hm),
                                            "p_two_sided": float(2 * norm.sf(abs((a_ - 0.5) / se_hm)))}}

    # ---- analyst choices
    ch = {}
    for mt in [1, 2, 3, 5]:
        tr_, k_, s_, c_, y_ = design(mt)
        ch[f"min_targets_{mt}"] = {"n_tfs": int(len(tr_)), "n_pos": int(y_.sum()),
                                   "attention_tf_query": r4(auroc(y_, A0[s_, c_])),
                                   "attention_tf_key": r4(auroc(y_, A0[c_, s_])),
                                   "attention_sym_mean": r4(auroc(y_, Asym[s_, c_])),
                                   "variance": r4(auroc(y_, gv[c_]))}
    k_all = np.concatenate([np.full(G, k) for k in range(nT)])
    src_all = tf_rows[k_all]; cand_all = np.tile(np.arange(G), nT)
    cnt = tr.groupby(["tf_index", "target_index"]).size()
    lab = np.array([cnt.get((i, j), 0) for i, j in zip(src_all, cand_all)])
    s_all = A[src_all, cand_all]                                 # diagonal kept as stored
    ch["with_self_pairs_scored_by_diagonal"] = r4(auroc(lab > 0, s_all))
    ch["with_self_pairs_and_duplicate_rows_as_weights"] = r4(WeightedAUROC(lab > 0, s_all)(np.where(lab > 0, lab, 1)))
    ch["n_self_positive_pairs_among_evaluated_tfs"] = int(((src_all == cand_all) & (lab > 0)).sum())
    keep_known = set(tr.loc[tr["mode"] != "Unknown"].eval("tf_index * 1500 + target_index").tolist())
    y_known = np.array([(s * 1500 + c) in keep_known for s, c in zip(src, cand)]) & y
    m_known = y_known | ~y                                       # drop positives with only Unknown mode
    ch["direction_known_positives_only"] = {"n_pos": int(y_known.sum()),
                                            "attention_tf_query": r4(auroc(y_known[m_known], scores["attention_tf_query"][m_known])),
                                            "variance": r4(auroc(y_known[m_known], scores["variance"][m_known]))}
    res["analyst_choices"] = ch

    # ---- weak nulls (do not keep target popularity)
    r_att = rankdata(scores["attention_tf_query"], method="average")
    rng_u = np.random.default_rng(SEED + 900 + RI)
    u = np.array([(r_att[rng_u.choice(len(y), P, replace=False)].sum() - P * (P + 1) / 2) / (P * Nn) for _ in range(1000)])
    res["uniform_label_permutation_null"] = {
        "null_mean": r4(u.mean()), "null_sd": r4(u.std(ddof=1)),
        "z": r4((a_ - u.mean()) / u.std(ddof=1)), "p_one_sided": float((1 + (u >= a_).sum()) / 1001)}
    rs = E[tf_rows].sum(1)
    rng_r = np.random.default_rng(SEED + 950 + RI)
    rown = np.array([(r_att[np.concatenate([k * (G - 1) + rng_r.choice(G - 1, rs[k], replace=False) for k in range(nT)])].sum()
                      - P * (P + 1) / 2) / (P * Nn) for _ in range(1000)])
    res["row_only_null_targets_uniform"] = {"null_mean": r4(rown.mean()), "null_sd": r4(rown.std(ddof=1)),
                                            "z": r4((a_ - rown.mean()) / rown.std(ddof=1))}

    # ---- incremental value, logistic gene model, grouped by TF and by candidate gene
    res["incremental_logreg"] = {
        "attention_tf_query_groupTF": incremental(scores["attention_tf_query"]),
        "attention_sym_mean_groupTF": incremental(scores["attention_sym_mean"]),
        "attention_tf_query_groupGene": incremental(scores["attention_tf_query"], groups=cand),
        "attention_sym_mean_groupGene": incremental(scores["attention_sym_mean"], groups=cand),
        "co_detection_lift_groupTF": incremental(scores["co_detection_lift"])}
    res["seconds"] = round(time.time() - t0, 1)
    save("main", res)
    print(json.dumps({k: res[k] for k in ["design", "pooled_auroc", "gaps"]}, indent=1))


# ------------------------------------------------------------------ Curveball null
def curveball(rows, forb, n_trades, rng):
    """Strona et al. 2014. rows = target sets per TF; forb[k] = TF k's own gene (never an edge)."""
    rows = [set(r) for r in rows]
    n = len(rows)
    aa = rng.integers(0, n, n_trades)
    bb = rng.integers(0, n - 1, n_trades)
    bb = bb + (bb >= aa)
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


class Scorer:
    """Pooled and per-TF-mean AUROC of a fixed (nT, G) score matrix for any set of target rows."""

    def __init__(self, S):
        self.rank = rankdata(np.concatenate([np.delete(S[k], tf_rows[k]) for k in range(nT)]), method="average")
        self.rank_row = [rankdata(np.delete(S[k], tf_rows[k]), method="average") for k in range(nT)]
        self.N = len(self.rank)

    def __call__(self, rows):
        pos, per = [], []
        for k, r in enumerate(rows):
            c = np.asarray(sorted(r), dtype=np.int64)
            ii = c - (c > tf_rows[k])
            pos.append(k * (G - 1) + ii)
            m1 = len(ii); m0 = G - 1 - m1
            per.append((self.rank_row[k][ii].sum() - m1 * (m1 + 1) / 2) / (m1 * m0))
        pos = np.concatenate(pos); Pp = len(pos)
        return float((self.rank[pos].sum() - Pp * (Pp + 1) / 2) / (Pp * (self.N - Pp))), float(np.mean(per))


def run_null(score_mats, n_draws, n_trades, seed):
    obs_rows = [set(np.flatnonzero(E[r]).tolist()) for r in tf_rows]
    obs_set = {(k, c) for k, r in enumerate(obs_rows) for c in r}
    sc = {k: Scorer(M) for k, M in score_mats.items()}
    obs = {k: s(obs_rows) for k, s in sc.items()}
    rng_c = np.random.default_rng(seed)
    nulls = {k: [] for k in sc}; jac = []; bad = 0
    rs0 = E[tf_rows].sum(1); cs0 = E[tf_rows].sum(0)
    for _ in range(n_draws):
        r = curveball(obs_rows, tf_rows, n_trades, rng_c)
        M = np.zeros((nT, G), dtype=np.int8)
        for k, rr in enumerate(r):
            M[k, list(rr)] = 1
        bad += int(not (np.array_equal(M.sum(1), rs0) and np.array_equal(M.sum(0), cs0)
                        and not M[np.arange(nT), tf_rows].any()))
        s = {(k, c) for k, rr in enumerate(r) for c in rr}
        jac.append(1 - len(s & obs_set) / len(s | obs_set))
        for k in sc:
            nulls[k].append(sc[k](r))
    out = {"n_draws": n_draws, "n_trades_per_draw": n_trades, "seed": seed,
           "draws_failing_margin_checks": int(bad), "draws_identical_to_observed": int((np.array(jac) == 0).sum()),
           "mean_jaccard_distance": r4(np.mean(jac)), "min_jaccard_distance": r4(np.min(jac)), "scores": {}}
    for k in sc:
        nv = np.array(nulls[k])
        d = {}
        for j, stat in enumerate(["pooled", "per_tf_mean"]):
            o = obs[k][j]; sd = nv[:, j].std(ddof=1)
            d[stat] = {"observed": r4(o), "null_mean": r4(nv[:, j].mean()), "null_sd": r4(sd),
                       "null_central95": r4(pci(nv[:, j])),
                       "z": r4((o - nv[:, j].mean()) / sd) if sd > 1e-12 else None,
                       "p_one_sided": float((1 + (nv[:, j] >= o).sum()) / (1 + n_draws))}
        out["scores"][k] = d
    return out


if "null" in SECTIONS:
    mats = {"attention_tf_query": A0[tf_rows],
            "expr_proximity": -np.abs(gm[tf_rows][:, None] - gm[None, :]),
            "variance": np.tile(gv, (nT, 1))}
    out = run_null(mats, 1000, 5000, SEED + 600 + RI)           # same chain seed as the source
    a = out["scores"]["attention_tf_query"]["pooled"]
    out["share_of_raw_excess_over_0.5_explained_by_null"] = r4((a["null_mean"] - 0.5) / (a["observed"] - 0.5))
    out["method"] = ("Curveball (Strona et al. 2014): rows = 16 evaluated TFs, columns = 1,500 genes; row and column sums "
                     "kept; (TF, own gene) never an edge; independent chains from the observed network; "
                     "z = (obs - null mean)/null SD; p = (1 + #null >= obs)/(1 + n), one-sided")
    out["seconds"] = round(time.time() - t0, 1)
    save("degree_preserving_null", out)
    print(json.dumps(out, indent=1))

if "null_alt" in SECTIONS:
    mats = {"attention_tf_key": A0[:, tf_rows].T,
            "attention_sym_mean": Asym[tf_rows],
            "attention_sym_max": np.maximum(A0[tf_rows], A0[:, tf_rows].T),
            "co_detection_lift": LIFT[tf_rows],
            "pair_cell_count": C[tf_rows].astype(np.float64)}
    out = run_null(mats, 1000, 1000, SEED + 610 + RI)
    out["seconds"] = round(time.time() - t0, 1)
    save("degree_preserving_null_other_scores", out)
    print(json.dumps(out, indent=1))

if "resid" in SECTIONS:
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    mask = ii != jj
    s_all_, t_all_ = ii[mask], jj[mask]
    del ii, jj
    Aoff = A0[s_all_, t_all_]
    flat = src * (G - 1) + cand - (cand > src)
    rng_b = np.random.default_rng(SEED + 800 + RI)
    draws = [np.bincount(rng_b.integers(0, nT, nT), minlength=nT) for _ in range(N_BOOT)]
    f_raw = WeightedAUROC(y, A0[src, cand]); raw_boot = np.array([f_raw(c[k_of]) for c in draws])
    gm32, gv32, gd32 = gm.astype(np.float32), gv.astype(np.float32), gd.astype(np.float32)
    out = {"raw_attention_ci95_tf_bootstrap": r4(pci(raw_boot)),
           "method": "predict attention of every ordered pair (i != j) from gene features only, 5-fold KFold(shuffle, 42) cross-fitting; residual scored like raw attention; CI = TF bootstrap with fitted models held fixed"}

    def fit_resid(X, target, fn):
        R = np.empty(len(target), dtype=np.float64); r2 = []
        for trn, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
            m = fn().fit(X[trn], target[trn])
            R[te] = target[te] - m.predict(X[te])
            pt_ = m.predict(X[trn])
            r2.append(1 - ((target[trn] - pt_) ** 2).sum() / ((target[trn] - target[trn].mean()) ** 2).sum())
        return R, float(np.mean(r2))

    variants = {
        "ols5_float64": (lambda: np.column_stack([gm[s_all_], gv[s_all_], gm[t_all_], gv[t_all_], gd[t_all_]]),
                         lambda: Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())]), Aoff),
        "hgb6": (lambda: np.column_stack([gm32[s_all_], gv32[s_all_], gd32[s_all_], gm32[t_all_], gv32[t_all_], gd32[t_all_]]),
                 lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                       random_state=SEED), Aoff.astype(np.float32)),
    }
    for name, (mk, fn, target) in variants.items():
        R, r2 = fit_resid(mk(), target, fn)
        s = R[flat]
        auc = auroc(y, s); f = WeightedAUROC(y, s); bt = np.array([f(c[k_of]) for c in draws])
        out[name] = {"residualised_auroc": r4(auc), "ci95_tf_bootstrap": r4(pci(bt)),
                     "drop_from_raw": r4(auroc(y, A0[src, cand]) - auc), "drop_ci95_tf_bootstrap": r4(pci(raw_boot - bt)),
                     "share_of_above_chance_signal_kept": r4((auc - 0.5) / (auroc(y, A0[src, cand]) - 0.5)),
                     "train_r2_mean": r4(r2)}
        print(name, out[name], f"[{time.time() - t0:.0f}s]", flush=True)
    # symmetrised attention, OLS on symmetric features of both genes (unordered pairs)
    iu = np.triu_indices(G, 1)
    a3 = np.column_stack([gm, gv, gd])
    Xs = np.column_stack([a3[iu[0]] + a3[iu[1]], np.abs(a3[iu[0]] - a3[iu[1]]), a3[iu[0]] * a3[iu[1]]])
    R, r2 = fit_resid(Xs, Asym[iu], lambda: Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())]))
    Rm = np.zeros((G, G)); Rm[iu] = R; Rm = Rm + Rm.T
    s = Rm[src, cand]; auc = auroc(y, s); f = WeightedAUROC(y, s); bt = np.array([f(c[k_of]) for c in draws])
    raw_s = auroc(y, Asym[src, cand])
    out["sym_mean_ols9_float64"] = {"raw_auroc": r4(raw_s), "residualised_auroc": r4(auc), "ci95_tf_bootstrap": r4(pci(bt)),
                                    "share_of_above_chance_signal_kept": r4((auc - 0.5) / (raw_s - 0.5)),
                                    "train_r2_mean": r4(r2)}
    out["seconds"] = round(time.time() - t0, 1)
    save("residualised_attention", out)
    print(json.dumps(out, indent=1))

if "incr_hgb" in SECTIONS:
    out = {"gene_model": "HistGradientBoostingClassifier on candidate mean, variance, dropout (+ score); GroupKFold(5) by TF",
           "attention_tf_query": incremental(A0[src, cand], model="hgb"),
           "attention_sym_mean": incremental(Asym[src, cand], model="hgb"),
           "attention_tf_query_groupGene": incremental(A0[src, cand], groups=cand, model="hgb")}
    out["seconds"] = round(time.time() - t0, 1)
    save("incremental_hgb", out)
    print(json.dumps(out, indent=1))
