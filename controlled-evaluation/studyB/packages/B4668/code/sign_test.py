"""Does the curated DoRothEA sign (activation / inhibition) predict which way a target gene moves
after its transcription factor (TF) is knocked down by CRISPRi?

Data: Replogle et al. 2022 Perturb-seq pseudo-bulk profiles (K562 genome-wide screen, RPE1 screen),
restricted to knockdowns of DoRothEA TFs and to their signed targets. DoRothEA edges come from OmniPath.

Prediction: knocking down an ACTIVATOR should lower its target (dE < 0); knocking down a REPRESSOR
should raise it (dE > 0).

Every signed edge (TF -> target) with a measured, non-zero response is scored. Edges are split into
three strata by the size of the response: all edges, |dE| top 50%, |dE| top 10%.

Per stratum:
  * 2x2 table: database class (activation / inhibition) x observed direction (up / down).
    Odds ratio OR = odds(up | inhibition edge) / odds(up | activation edge). OR > 1 means the
    database sign and the response agree (0.5 is added to every cell of a table with a zero cell).
    Fisher exact test (two-sided).
  * Sign-shuffle nulls, one-sided p = P(null OR >= observed OR):
      - across edges: the signs are permuted over all edges of the stratum;
      - within TF: the signs are permuted among the edges of the same TF, so each TF keeps its own
        mix of activation and inhibition labels. Agreement that comes only from a TF's overall label
        and its overall response is kept in this null; only edge-by-edge information is tested.
  * TF-cluster bootstrap 95% interval of the OR (TFs resampled with replacement).
  * Balanced accuracy = mean of the per-class accuracies over the two DATABASE classes, with a
    within-class bootstrap 95% interval, next to two baselines (always predict down; random sign).
    The balanced accuracy over the two OBSERVED classes (up / down) is reported as well.
DoRothEA confidence levels (A-E) are reported for the top 10% stratum.

Run from the package root:  python code/sign_test.py
"""
import os, csv, gzip, json, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SEED = 20260801
N_NULL = 2000
N_BOOT = 2000
STRATA = (("all", 0.0), ("|dE| top 50%", 0.50), ("|dE| top 10%", 0.90))
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def load_signed(path):
    """-> ({(tf, target): +1 activation / -1 inhibition}, {(tf, target): DoRothEA level}).
    Edges annotated neither way or both ways are dropped; self-edges are dropped."""
    out = {}
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r.get("is_stimulation") == "True"; i = r.get("is_inhibition") == "True"
            if s == i:
                continue
            tf, tg = r.get("source_genesymbol"), r.get("target_genesymbol")
            if tf and tg and tf != tg:
                out[(tf, tg)] = 1 if s else -1
    lev = {}
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            k = (r.get("source_genesymbol"), r.get("target_genesymbol"))
            if k in out:
                lev[k] = (r.get("dorothea_level") or "?").split(";")[0]
    return out, lev


def load_response(path):
    """-> (R: knockdown x gene mean response, float64; gene names; knockdown names).
    Rows are guide-level profiles; profiles that target the same gene are averaged.
    A TF's response in its own gene is set to NaN."""
    df = pd.read_csv(path, index_col=0)
    X = df.to_numpy(np.float32)
    genes = [str(g) for g in df.columns]
    sym = [q.split("_")[1] if len(q.split("_")) > 2 else q for q in df.index.astype(str)]
    by = defaultdict(list)
    for i, p in enumerate(sym):
        by[p].append(i)
    names = sorted(by)
    R = np.array([X[np.array(by[p])].mean(0) for p in names], np.float64)
    gpos = {g: j for j, g in enumerate(genes)}
    for i, p in enumerate(names):
        if p in gpos:
            R[i, gpos[p]] = np.nan
    return R, genes, names


def table2(S, up):
    a = S > 0
    return np.array([[(a & up).sum(), (a & ~up).sum()],
                     [(~a & up).sum(), (~a & ~up).sum()]], dtype=np.int64)


def odds_ratio(t):
    """odds(up | inhibition) / odds(up | activation). If any cell is zero, 0.5 is added to every
    cell (Haldane-Anscombe), so the value stays finite."""
    t = np.asarray(t, float)
    if (t == 0).any():
        t = t + 0.5
    return float((t[1, 0] * t[0, 1]) / (t[1, 1] * t[0, 0]))


def shuffle_within(S, groups, rng):
    """Permute the entries of S among positions that share a group code."""
    order = np.lexsort((rng.random(len(S)), groups))
    base = np.argsort(groups, kind="stable")
    out = np.empty_like(S)
    out[base] = S[order]
    return out


def stratum_stats(S, D, TF, rng):
    up = D > 0
    t = table2(S, up)
    orr = odds_ratio(t)
    p_f = float(fisher_exact(t)[1])
    one = np.zeros(len(S), np.int64)
    codes = np.unique(TF, return_inverse=True)[1]
    null_all = np.array([odds_ratio(table2(shuffle_within(S, one, rng), up)) for _ in range(N_NULL)])
    null_tf = np.array([odds_ratio(table2(shuffle_within(S, codes, rng), up)) for _ in range(N_NULL)])
    tf_ids = np.unique(codes)
    members = [np.where(codes == c)[0] for c in tf_ids]
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(members), len(members))
        ii = np.concatenate([members[k] for k in pick])
        boot.append(odds_ratio(table2(S[ii], up[ii])))
    boot = np.array(boot)
    pos = boot[np.isfinite(boot) & (boot > 0)]

    def nullsum(v):
        v = v[np.isfinite(v)]
        return dict(median=float(np.median(v)), q95=float(np.percentile(v, 95)),
                    p_one_sided=float((1 + (v >= orr).sum()) / (1 + len(v))))
    mixed = sum(1 for m in members if len(set(S[m])) == 2)
    return dict(n=int(len(S)), n_activating=int((S > 0).sum()), n_inhibiting=int((S < 0).sum()),
                n_tfs=int(len(tf_ids)), n_tfs_with_both_classes=int(mixed),
                table={"activation": {"up": int(t[0, 0]), "down": int(t[0, 1])},
                       "inhibition": {"up": int(t[1, 0]), "down": int(t[1, 1])}},
                frac_up_activation=float(t[0, 0] / t[0].sum()) if t[0].sum() else float("nan"),
                frac_up_inhibition=float(t[1, 0] / t[1].sum()) if t[1].sum() else float("nan"),
                odds_ratio=orr, fisher_p=p_f,
                shuffle_across_edges=nullsum(null_all), shuffle_within_tf=nullsum(null_tf),
                tf_bootstrap_ci=[float(np.percentile(pos, 2.5)), float(np.percentile(pos, 97.5))],
                tf_bootstrap_n_valid=int(len(pos)))


def run_line(line, data_dir, out_dir):
    rng = np.random.default_rng(SEED)
    R, genes, names = load_response(os.path.join(data_dir, f"{line}_tf_knockdown_response.csv.gz"))
    ridx = {p: i for i, p in enumerate(names)}
    gpos = {g: j for j, g in enumerate(genes)}
    say(f"[data] {line}: {R.shape[0]} knocked-down TFs x {R.shape[1]} genes")
    frac_neg = float(np.nanmean(R < 0))
    say(f"[prior] fraction of all responses in this matrix that are NEGATIVE: {frac_neg:.4f}")

    SGN, LEV = load_signed(os.path.join(data_dir, "dorothea_omnipath.tsv.gz"))
    say(f"[db] {len(SGN):,} signed DoRothEA edges ({sum(v > 0 for v in SGN.values()):,} activating, "
        f"{sum(v < 0 for v in SGN.values()):,} inhibiting)")
    rows = []
    for (tf, tg), s in SGN.items():
        if tf not in ridx or tg not in gpos:
            continue
        d = R[ridx[tf], gpos[tg]]
        if not np.isfinite(d) or d == 0:
            continue
        rows.append((tf, tg, s, d, LEV.get((tf, tg), "?")))
    TF = np.array([r[0] for r in rows]); S = np.array([r[2] for r in rows])
    D = np.array([r[3] for r in rows]); LV = np.array([r[4] for r in rows])
    say(f"[map] {len(rows):,} signed edges with a measured response "
        f"({int((S > 0).sum()):,} activating, {int((S < 0).sum()):,} inhibiting; "
        f"{len(set(TF))} TFs, {len(set(r[1] for r in rows))} targets)")
    pd.DataFrame({"tf": TF, "target": [r[1] for r in rows], "db_sign": S, "dorothea_level": LV,
                  "dE": D}).to_csv(os.path.join(out_dir, f"edges_{line}.csv"), index=False,
                                   float_format="%.9g")

    pred_db = -S
    obs = np.sign(D)
    out = {"line": line, "n": len(rows), "n_tfs": len(set(TF)), "frac_neg_matrix": frac_neg,
           "frac_down_edges": float((D < 0).mean())}
    sels = [(nm, np.ones(len(D), bool) if q == 0 else np.abs(D) >= np.quantile(np.abs(D), q))
            for nm, q in STRATA]

    say(f"\n{'stratum':<16}{'predictor':<22}{'balanced acc':>14}{'95% CI':>18}"
        f"{'act':>7}{'inh':>7}{'n':>8}{'bal(up/down)':>14}")
    say("-" * 106)
    for sname, sel in sels:
        out[sname] = {"threshold_abs_dE": float(np.abs(D[sel]).min())}
        for pname, pr in (("DATABASE sign", pred_db),
                          ("always-down (prior)", -np.ones(len(D))),
                          ("random", rng.choice([-1.0, 1.0], len(D)))):
            hit = (pr[sel] == obs[sel]).astype(float)
            a = S[sel] > 0
            acc_a = hit[a].mean() if a.sum() else np.nan
            acc_i = hit[~a].mean() if (~a).sum() else np.nan
            bal = float(np.nanmean([acc_a, acc_i]))
            u = obs[sel] > 0
            bal_obs = float(np.nanmean([hit[u].mean() if u.sum() else np.nan,
                                        hit[~u].mean() if (~u).sum() else np.nan]))
            bs = []
            ha, hi_ = hit[a], hit[~a]
            for _ in range(N_BOOT):
                x = ha[rng.integers(0, len(ha), len(ha))].mean() if len(ha) else np.nan
                y = hi_[rng.integers(0, len(hi_), len(hi_))].mean() if len(hi_) else np.nan
                bs.append(np.nanmean([x, y]))
            lo, hi2 = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0.5 else " "
            say(f"{sname:<16}{pname:<22}{bal:>14.4f}{f' [{lo:.4f},{hi2:.4f}]':>18}{star}"
                f"{acc_a:>6.3f}{acc_i:>7.3f}{int(sel.sum()):>8,}{bal_obs:>14.4f}")
            out[sname][pname] = dict(balanced=bal, ci=[float(lo), float(hi2)],
                                     acc_activating=float(acc_a), acc_inhibiting=float(acc_i),
                                     balanced_over_observed=bal_obs, n=int(sel.sum()))
        say()

    say("2x2 tables (database class x observed direction), odds ratio, nulls")
    say(f"{'stratum':<16}{'act up/down':>14}{'inh up/down':>14}{'OR':>7}{'Fisher p':>10}"
        f"{'shuf-all med':>14}{'p':>8}{'shuf-TF med':>13}{'p':>8}{'TF-boot 95%':>16}{'TFs':>6}")
    say("-" * 126)
    for sname, sel in sels:
        st = stratum_stats(S[sel], D[sel], TF[sel], rng)
        out[sname]["two_by_two"] = st
        t = st["table"]
        ci = "[{:.2f},{:.2f}]".format(*st["tf_bootstrap_ci"])
        say(f"{sname:<16}{t['activation']['up']:>7}/{t['activation']['down']:<6}"
            f"{t['inhibition']['up']:>7}/{t['inhibition']['down']:<6}{st['odds_ratio']:>7.2f}"
            f"{st['fisher_p']:>10.2g}{st['shuffle_across_edges']['median']:>14.2f}"
            f"{st['shuffle_across_edges']['p_one_sided']:>8.4f}{st['shuffle_within_tf']['median']:>13.2f}"
            f"{st['shuffle_within_tf']['p_one_sided']:>8.4f}"
            f"{ci:>16}"
            f"{st['n_tfs']:>6}")

    say("\nDoRothEA confidence level, |dE| top 10% (levels with >= 30 edges)")
    top = sels[-1][1]
    out["levels_top10"] = {}
    for L in ("A", "B", "C", "D", "E"):
        m = top & (LV == L)
        if m.sum() < 30:
            continue
        h = (pred_db[m] == obs[m]).astype(float); aa = S[m] > 0
        acc_a = h[aa].mean() if aa.sum() else np.nan
        acc_i = h[~aa].mean() if (~aa).sum() else np.nan
        ba = float(np.nanmean([acc_a, acc_i]))
        t = table2(S[m], D[m] > 0)
        say(f"   level {L}: balanced {ba:.4f}  act {acc_a:.3f}  inh {acc_i:.3f}  n={int(m.sum())} "
            f"(act {int(aa.sum())}, inh {int((~aa).sum())}, TFs {len(set(TF[m]))})  OR {odds_ratio(t):.2f}")
        out["levels_top10"][L] = dict(balanced=ba, acc_activating=float(acc_a),
                                      acc_inhibiting=float(acc_i), n=int(m.sum()),
                                      n_activating=int(aa.sum()), n_inhibiting=int((~aa).sum()),
                                      n_tfs=len(set(TF[m])), odds_ratio=odds_ratio(t))
    with open(os.path.join(out_dir, f"sign_{line}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    say(f"wrote sign_{line}.json, edges_{line}.csv\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--lines", default="k562,rpe1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for line in args.lines.split(","):
        say(f"===== {line} =====")
        run_line(line, args.data, args.out)
    say("balanced accuracy = mean of (accuracy on activating edges, accuracy on inhibiting edges); "
        "chance = 0.5. bal(up/down) = the same, averaged over the observed up and down classes.")
    say("OR = odds(up | inhibition) / odds(up | activation); 1 = no association.")
    with open(os.path.join(args.out, "run_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
