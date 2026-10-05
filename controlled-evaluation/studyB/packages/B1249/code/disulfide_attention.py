#!/usr/bin/env python
"""Which cysteine pairs are disulfide-bonded? Ranking pairs by ESM-2 attention.

For every pair of cysteines (i < j) inside one protein, each of ESM-2 650M's 660
attention heads gives a score: entry (i, j) of that head's attention map over the
residues (<cls>/<eos> dropped), symmetrised and corrected with the average product
correction (APC). Candidate 660 is the unweighted mean of the 660 corrected maps.
The scores are precomputed in data/attention_scores.npz (see code/extract_attention.py).

Protocol
--------
* Metric: average precision (AP) of a score over all candidate pairs of a split, pooled
  across proteins. The no-skill value is the fraction of pairs that are bonded.
* Selection: AP is computed for every candidate on the SELECT proteins; the candidate
  with the highest select AP is chosen and its AP on the disjoint TEST proteins is the
  reported number. The test-split maximum over all candidates is also reported, next to
  the expected maximum under a null, so the cost of picking a winner is visible.
* Null: bond labels permuted within each test protein (each protein keeps its number of
  bonds; only which pair is bonded is destroyed).
* CI: protein-level bootstrap of the chosen candidate's test AP.
* Baseline: sequence separation |i - j| and |i - j| / protein length, both signs, run
  through the same selection protocol.

Run from the package root:
    python code/disulfide_attention.py               # writes outputs/
    python code/disulfide_attention.py --selftest    # checks the metric helpers
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
N_BOOT = 200
N_NULL_CAND = 256
DETECTOR_LIFT = 2.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _groups_desc(v: np.ndarray, y: np.ndarray):
    """Tie groups of a descending-sorted score vector: (count, positives) per group."""
    if v.size == 0:
        return np.zeros(0), np.zeros(0)
    b = np.concatenate((np.flatnonzero(v[:-1] != v[1:]), [v.size - 1]))
    cnt = np.diff(np.concatenate(([-1], b))).astype(np.float64)
    csy = np.cumsum(y, dtype=np.float64)
    pos = np.diff(np.concatenate(([0.0], csy[b])))
    return cnt, pos


def ap_from_groups(cnt: np.ndarray, pos: np.ndarray) -> float:
    """AP = sum over tie groups (descending score) of (R_k - R_{k-1}) * P_k."""
    ctot = np.cumsum(cnt, dtype=np.float64)
    ptot = np.cumsum(pos, dtype=np.float64)
    n_pos = ptot[-1] if len(ptot) else 0.0
    if n_pos <= 0 or ctot[-1] <= 0:
        return 0.0
    prec = ptot / np.maximum(ctot, 1e-12)
    rec = ptot / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """AP over dense scores; tied scores are treated as one group (as sklearn does)."""
    o = np.argsort(-scores, kind="stable")
    return ap_from_groups(*_groups_desc(scores[o], y[o].astype(np.float64)))


def expected_precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Precision in the top k, averaged over random order within tied scores."""
    n = scores.size
    if n == 0 or k <= 0:
        return float("nan")
    k_used = min(int(k), n)
    o = np.argsort(-scores, kind="stable")
    cnt, pos = _groups_desc(scores[o], y[o].astype(np.float64))
    got, left = 0.0, float(k_used)
    for c, p in zip(cnt, pos):
        if left <= 0:
            break
        take = min(c, left)
        got += p * (take / c)
        left -= take
    return float(got / k_used)


def permute_within_protein(y: np.ndarray, prot: np.ndarray, rng) -> np.ndarray:
    """Shuffle bond labels inside each protein, keeping each protein's bond count."""
    out = np.zeros_like(y)
    order = np.argsort(prot, kind="stable")
    starts = np.flatnonzero(np.concatenate(([True], np.diff(prot[order]) != 0)))
    ends = np.concatenate((starts[1:], [order.size]))
    for a, b in zip(starts, ends):
        blk = order[a:b]
        out[rng.permutation(blk)] = y[blk]
    return out


def expected_best_of_n(null_aps: np.ndarray, n: int, n_draws: int, rng) -> float:
    """E[max AP over n candidates] under the null, from a sample of null APs."""
    if null_aps.size == 0:
        return float("nan")
    m = [null_aps[rng.integers(0, null_aps.size, size=n)].max() for _ in range(n_draws)]
    return float(np.mean(m))


def cluster_bootstrap_ap(scores, y, prot, n_boot, seed):
    """Protein-level bootstrap of AP (pairs in one protein are not independent)."""
    o = np.argsort(-scores, kind="stable")
    s, ys, ps = scores[o], y[o].astype(np.float64), prot[o]
    b = np.concatenate((np.flatnonzero(s[:-1] != s[1:]), [s.size - 1]))
    starts = np.concatenate(([0], b[:-1] + 1))
    uniq, pidx = np.unique(ps, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for t in range(n_boot):
        w = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size)).astype(np.float64)
        wr = w[pidx]
        vals[t] = ap_from_groups(np.add.reduceat(wr, starts), np.add.reduceat(wr * ys, starts))
    return dict(mean=float(vals.mean()), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), n_boot=int(n_boot),
                n_proteins=int(uniq.size))


def per_protein_metrics(scores, prot, y):
    """Per-protein AP, precision of the top-ranked pair, precision at k = n bonds."""
    order = np.argsort(prot, kind="stable")
    starts = np.flatnonzero(np.concatenate(([True], np.diff(prot[order]) != 0)))
    ends = np.concatenate((starts[1:], [order.size]))
    aps, p1, pn, ncand = [], [], [], []
    for a, b in zip(starts, ends):
        blk = order[a:b]
        yb, sb = y[blk], scores[blk]
        ncand.append(blk.size)
        if yb.sum() == 0:
            continue
        aps.append(average_precision(sb, yb))
        p1.append(expected_precision_at_k(sb, yb, 1))
        pn.append(expected_precision_at_k(sb, yb, int(yb.sum())))
    return dict(n_proteins_with_a_bond=len(aps),
                mean_per_protein_ap=float(np.mean(aps)),
                mean_precision_at_1=float(np.mean(p1)),
                mean_precision_at_n_bonds=float(np.mean(pn)),
                median_candidate_pairs_per_protein=float(np.median(ncand)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def protein_codes(acc: np.ndarray) -> np.ndarray:
    """Integer protein codes, numbered in the order proteins first appear in the file."""
    uniq, first, inv = np.unique(acc, return_index=True, return_inverse=True)
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(uniq.size)
    return rank[inv]


def load_data():
    pairs = {}
    for split in ("select", "test"):
        z = np.load(DATA / f"pairs_{split}.npz", allow_pickle=False)
        acc = z["protein"]
        pairs[split] = dict(protein=acc, prot=protein_codes(acc),
                            i=z["pos_i"].astype(np.int64), j=z["pos_j"].astype(np.int64),
                            y=z["bonded"].astype(bool),
                            plen=z["protein_length"].astype(np.int64))
        pairs[split]["sep"] = pairs[split]["j"] - pairs[split]["i"]
    z = np.load(DATA / "attention_scores.npz", allow_pickle=False)
    S = {s: z[f"S_{s}"] for s in ("select", "test")}
    n_layers, n_head = int(z["n_layers"]), int(z["n_head"])
    for s in S:
        if S[s].shape != (n_layers * n_head + 1, pairs[s]["y"].size):
            raise SystemExit(f"attention scores for {s} have shape {S[s].shape}")
    return pairs, S, n_layers, n_head


# ---------------------------------------------------------------------------
# Scorers: each is (name, list of candidate names, column(c, split) -> scores)
# ---------------------------------------------------------------------------

def attention_scorer(S, n_layers, n_head):
    names = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_head)] + ["mean_all_heads"]
    return "attention_esm2", names, (lambda c, split: S[split][c])


def separation_scorer(pairs):
    funcs = [("abs_separation", lambda p: p["sep"].astype(np.float32)),
             ("relative_separation",
              lambda p: (p["sep"] / np.maximum(p["plen"], 1)).astype(np.float32))]
    n = len(funcs)
    names = [f"{f[0]}:+" for f in funcs] + [f"{f[0]}:-" for f in funcs]

    def column(c, split):
        return funcs[c % n][1](pairs[split]) * (1.0 if c < n else -1.0)
    return "sequence_separation", names, column


def score_arm(name, names, column, pairs, seed):
    y_sel, y_test = pairs["select"]["y"], pairs["test"]["y"]
    prot_test = pairs["test"]["prot"]
    prev = float(y_test.mean())
    n_cand = len(names)
    ap_sel = np.array([average_precision(column(c, "select"), y_sel) for c in range(n_cand)])
    ap_test = np.array([average_precision(column(c, "test"), y_test) for c in range(n_cand)])

    idx = int(np.argmax(ap_sel))
    chosen_ap = float(ap_test[idx])
    test_max = float(ap_test.max())

    y_null = permute_within_protein(y_test, prot_test, np.random.default_rng(seed + 22))
    rng = np.random.default_rng(seed + 21)
    pick = np.sort(rng.choice(n_cand, size=min(N_NULL_CAND, n_cand), replace=False))
    null = np.array([average_precision(column(int(c), "test"), y_null) for c in pick])
    ebon = expected_best_of_n(null, n_cand, 200, np.random.default_rng(seed + 33))

    best = column(idx, "test")
    boot = cluster_bootstrap_ap(best, y_test, prot_test, N_BOOT, seed + 2000)
    n_det = int((ap_test >= DETECTOR_LIFT * prev).sum())
    res = dict(
        chosen_candidate=names[idx], chosen_select_ap=float(ap_sel[idx]),
        test_ap=chosen_ap, test_ap_minus_base_rate=chosen_ap - prev,
        test_ap_over_base_rate=chosen_ap / prev,
        test_max_ap_any_candidate=test_max, test_max_minus_chosen=test_max - chosen_ap,
        test_max_ap_expected_under_null=ebon,
        test_max_minus_null_expectation=test_max - ebon,
        n_candidates=n_cand,
        n_candidates_test_ap_at_least_2x_base_rate=n_det,
        bootstrap_ci_test_ap=boot,
        per_protein_test=per_protein_metrics(best, prot_test, y_test),
    )
    return res, ap_sel, ap_test


def selftest():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(30):
        n = int(rng.integers(200, 2000))
        y = rng.random(n) < 0.05
        if y.sum() == 0:
            continue
        s = np.where(rng.random(n) < 0.3, rng.normal(size=n), 0.0)
        worst = max(worst, abs(average_precision_score(y, s) - average_precision(s, y)))
    assert worst < 1e-9, worst
    print(f"[selftest] AP vs sklearn, max |diff| = {worst:.2e}")
    s = np.array([3.0, 1.0, 1.0, 1.0, 0.0]); y = np.array([0, 1, 0, 0, 1], dtype=bool)
    assert abs(expected_precision_at_k(s, y, 2) - (1 / 3) / 2) < 1e-12
    sc = np.zeros(50); yc = np.zeros(50, dtype=bool); yc[:7] = True
    assert abs(expected_precision_at_k(sc, yc, 5) - 7 / 50) < 1e-12
    print("[selftest] precision@k under ties ok")
    prot = np.repeat(np.arange(6), 5); yy = rng.random(30) < 0.3
    yp = permute_within_protein(yy, prot, np.random.default_rng(1))
    assert all(yy[prot == p].sum() == yp[prot == p].sum() for p in range(6))
    print("[selftest] within-protein permutation keeps bond counts")
    L = 40
    a = rng.random(L) + 0.5
    M = np.outer(a, a)
    Sym = 0.5 * (M + M.T)
    C = Sym - np.outer(Sym.sum(1), Sym.sum(0)) / Sym.sum()
    assert np.abs(C).max() < 1e-10
    print("[selftest] APC removes a pure row x column map")
    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    pairs, S, n_layers, n_head = load_data()
    data = {}
    for s in ("select", "test"):
        p = pairs[s]
        data[s] = dict(n_pairs=int(p["y"].size), n_bonded=int(p["y"].sum()),
                       n_proteins=int(np.unique(p["prot"]).size),
                       base_rate=float(p["y"].mean()))
        log(f"{s}: {data[s]['n_pairs']} candidate cysteine pairs in {data[s]['n_proteins']} "
            f"proteins, {data[s]['n_bonded']} bonded (base rate {data[s]['base_rate']:.4f})")

    arms, ap_tables = {}, {}
    for name, names, column in (separation_scorer(pairs), attention_scorer(S, n_layers, n_head)):
        res, ap_sel, ap_test = score_arm(name, names, column, pairs, args.seed)
        arms[name] = res
        ap_tables[name] = (names, ap_sel, ap_test)
        log(f"{name:<20} chosen {res['chosen_candidate']:<16} test AP {res['test_ap']:.4f} "
            f"(base {data['test']['base_rate']:.4f}, x{res['test_ap_over_base_rate']:.2f}) "
            f"CI [{res['bootstrap_ci_test_ap']['lo']:.3f}, {res['bootstrap_ci_test_ap']['hi']:.3f}]  "
            f"test max {res['test_max_ap_any_candidate']:.4f} (null max "
            f"{res['test_max_ap_expected_under_null']:.4f})  "
            f"P@1 {res['per_protein_test']['mean_precision_at_1']:.3f}")

    # the mean over all heads, scored as one candidate without selection
    names, ap_sel, ap_test = ap_tables["attention_esm2"]
    c = len(names) - 1
    y_test, prot_test = pairs["test"]["y"], pairs["test"]["prot"]
    arms["mean_all_heads"] = dict(
        select_ap=float(ap_sel[c]), test_ap=float(ap_test[c]),
        test_ap_over_base_rate=float(ap_test[c]) / data["test"]["base_rate"],
        bootstrap_ci_test_ap=cluster_bootstrap_ap(S["test"][c], y_test, prot_test,
                                                  N_BOOT, args.seed + 2000),
        per_protein_test=per_protein_metrics(S["test"][c], prot_test, y_test),
        note="single candidate, no selection")
    log(f"{'mean_all_heads':<20} no selection           test AP {ap_test[c]:.4f} "
        f"(select AP {ap_sel[c]:.4f})")

    # per-layer view of the 660 heads
    prev = data["test"]["base_rate"]
    head_sel = ap_sel[:-1].reshape(n_layers, n_head)
    head_test = ap_test[:-1].reshape(n_layers, n_head)
    per_layer = []
    for l in range(n_layers):
        h = int(np.argmax(head_sel[l]))
        per_layer.append(dict(layer=l, best_head_by_select=f"L{l}H{h}",
                              its_test_ap=float(head_test[l, h]),
                              max_test_ap=float(head_test[l].max()),
                              n_heads_test_ap_at_least_2x_base_rate=int(
                                  (head_test[l] >= DETECTOR_LIFT * prev).sum())))
    top = np.argsort(-ap_sel[:-1], kind="stable")[:10]
    top_heads = [dict(head=names[t], select_ap=float(ap_sel[t]), test_ap=float(ap_test[t]))
                 for t in top]

    result = dict(
        question="for a pair of cysteines (i, j) in one protein, is this the bonded pair?",
        metric="average precision over all candidate cysteine pairs of a split, pooled over "
               "proteins; no-skill value = base rate (fraction of pairs bonded)",
        selection="candidate with the highest SELECT-split AP; its TEST-split AP is reported",
        null="bond labels permuted within each test protein; used for the expected maximum "
             "over candidates",
        seed=args.seed,
        data=data,
        arms=arms,
        attention_per_layer=per_layer,
        attention_top10_heads_by_select_ap=top_heads,
    )
    with open(out / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "head_ap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "layer", "head", "ap_select", "ap_test"])
        for k, nm in enumerate(names):
            lh = (k // n_head, k % n_head) if k < n_layers * n_head else ("", "")
            w.writerow([nm, lh[0], lh[1], f"{ap_sel[k]:.6f}", f"{ap_test[k]:.6f}"])
    n_layers_with = sum(1 for r in per_layer if r["n_heads_test_ap_at_least_2x_base_rate"] > 0)
    log(f"heads with test AP >= 2x base rate: "
        f"{arms['attention_esm2']['n_candidates_test_ap_at_least_2x_base_rate']} of "
        f"{len(names)} candidates, in {n_layers_with} of {n_layers} layers")
    with open(out / "run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"wrote {out / 'results.json'}, head_ap.csv, run_log.txt")


if __name__ == "__main__":
    main()
