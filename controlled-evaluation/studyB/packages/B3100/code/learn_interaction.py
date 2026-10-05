"""Can anything be trained on Norman? Predict the interaction term from the two single responses.

The question is posed per GENE:

    for a double A_B and a gene h:  I_h = dAB_h - (dA_h + dB_h)
    predict I_h from (dA_h, dB_h) and simple functions of them

where dX is the mean log-normalised expression of the cells carrying perturbation X minus the mean
of the control cells. That is 115 pairs x 1,000 genes = 115,000 training examples. Whole PAIRS are
held out: the double of an evaluation pair is never seen in training. Single perturbations and
measured genes do recur across pairs.

THE HYPOTHESIS THIS TESTS. Much of what looks like "gene-gene interaction" may simply be SATURATION:
if two perturbations push the same gene in the same direction, the combined effect is smaller than
the sum, because expression cannot fall below zero or rise without limit. That is predictable from
(dA_h, dB_h) ALONE - it needs no knowledge of any regulatory network. If a model using only those two
numbers captures most of I, then the interaction term is mostly a ceiling/floor effect.

BASELINES, in increasing order of what they are allowed to know:
  additive        predict I = 0 (the double is exactly the sum of the singles)
  sum-only        linear in (dA + dB) - captures simple shrinkage of large responses
  saturation      adds the product, min, max and sign agreement - no network knowledge at all
  +gene identity  adds a per-gene offset fitted on TRAINING pairs, to see if some genes are simply
                  always non-additive
Ceiling comes from the split-half reliability of the I profile, Spearman-Brown corrected to the
full-data estimate.

Run from the package root:
    python code/learn_interaction.py --data data/norman2019_subset.npz --out outputs/results.json
"""
import json, argparse
import numpy as np

SEED = 20260801


def load(path):
    z = np.load(path, allow_pickle=False)
    return z["counts"], z["totals"], z["labels"].astype(str)


def analyse(counts, totals, lab, args):
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(SEED)

    X = counts.astype(np.float32)                                  # CP10k, then log1p
    X *= (1e4 / np.asarray(totals, dtype=np.float64)).astype(np.float32)[:, None]
    np.log1p(X, out=X)
    low = np.array([x.lower() for x in lab])
    ctrl = np.flatnonzero(np.isin(low, ["control", "ctrl"]))
    gm = X.mean(0); keep = np.argsort(-gm)[: args.top_genes]
    X = X[:, keep]
    cmean = X[ctrl].mean(0)

    def delta(idx):
        return X[idx].mean(0) - cmean

    singles = {p: np.flatnonzero(lab == p) for p in np.unique(lab)
               if "_" not in p and p.lower() not in ("control", "ctrl")}
    singles = {p: i for p, i in singles.items() if len(i) >= args.min_single}
    DA = {p: delta(i) for p, i in singles.items()}
    P = []
    for p in np.unique(lab):
        if "_" not in p or (lab == p).sum() < args.min_double:
            continue
        x, y = p.split("_", 1)
        if x in singles and y in singles:
            P.append((p, x, y))
    print(f"[norman] {len(ctrl)} control cells | {len(singles)} singles | {len(P)} doubles | "
          f"{len(keep)} genes -> {len(P)*len(keep):,} (pair,gene) training examples", flush=True)

    rs = np.random.default_rng(SEED + 1)

    def halves(idx):
        idx = rs.permutation(idx); h = len(idx) // 2
        return idx[:h], idx[h:2 * h]

    c1, c2 = halves(ctrl)
    cm1, cm2 = X[c1].mean(0), X[c2].mean(0)
    SH = {p: halves(i) for p, i in singles.items()}

    I, A_, B_, rel1, rel2 = [], [], [], [], []
    for p, x, y in P:
        idx = np.flatnonzero(lab == p); rng.shuffle(idx); h = len(idx) // 2
        ab1, ab2 = delta(idx[:h]), delta(idx[h:2 * h])
        add = DA[x] + DA[y]
        I.append(0.5 * (ab1 + ab2) - add); A_.append(DA[x]); B_.append(DA[y])
        (x1, x2), (y1, y2) = SH[x], SH[y]
        rel1.append(X[idx[:h]].mean(0) - X[x1].mean(0) - X[y1].mean(0) + cm1)
        rel2.append(X[idx[h:2 * h]].mean(0) - X[x2].mean(0) - X[y2].mean(0) + cm2)
    I = np.stack(I).astype(np.float64); A_ = np.stack(A_).astype(np.float64)
    B_ = np.stack(B_).astype(np.float64)
    r_half = np.mean([np.corrcoef(rel1[i], rel2[i])[0, 1] for i in range(len(P))])
    r_full = 2 * r_half / (1 + r_half)          # Spearman-Brown: reliability of the full-data estimate
    ceiling = np.sqrt(r_full)
    print(f"[ceiling] I profile split-half {r_half:.4f} -> full-data reliability {r_full:.4f} "
          f"-> a perfect model reaches ~{ceiling:.3f}", flush=True)

    def feats(a_, b_, kind):
        s, d = a_ + b_, a_ - b_
        if kind == "sum-only":
            return np.column_stack([s])
        if kind == "saturation":
            return np.column_stack([s, a_ * b_, np.minimum(a_, b_), np.maximum(a_, b_),
                                    np.abs(a_) + np.abs(b_), np.sign(a_) * np.sign(b_), d ** 2])
        raise ValueError(kind)

    fold = rng.integers(0, args.folds, len(P))
    G = len(keep)
    res = {}
    for kind in ("additive", "sum-only", "saturation", "saturation+gene"):
        pred = np.zeros_like(I)
        for f in range(args.folds):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            if len(te) == 0 or len(tr) < 5:
                continue
            if kind == "additive":
                continue                                   # prediction is exactly zero
            Ftr = feats(A_[tr].ravel(), B_[tr].ravel(), kind.replace("+gene", ""))
            Fte = feats(A_[te].ravel(), B_[te].ravel(), kind.replace("+gene", ""))
            if kind.endswith("+gene"):
                # per-gene mean interaction, estimated on TRAINING pairs only
                goff = I[tr].mean(0)
                Ftr = np.column_stack([Ftr, np.tile(goff, len(tr))])
                Fte = np.column_stack([Fte, np.tile(goff, len(te))])
            m = Ridge(alpha=1.0).fit(Ftr, I[tr].ravel())
            pred[te] = m.predict(Fte).reshape(len(te), G)
        per = [np.corrcoef(pred[i], I[i])[0, 1] for i in range(len(P))
               if pred[i].std() > 0 and I[i].std() > 0]
        # variance of I explained, pooled
        ss = 1 - ((I - pred) ** 2).sum() / (I ** 2).sum()
        bs = np.array([np.mean(np.array(per)[rng.integers(0, len(per), len(per))])
                       for _ in range(2000)]) if per else np.array([0.0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        corr = float(np.mean(per)) if per else 0.0
        res[kind] = dict(corr=corr, lo=float(lo), hi=float(hi), var_explained=float(ss), n=len(per),
                         frac_of_ceiling=float(corr / ceiling))
        print(f"  {kind:<18} corr(pred I, true I) = {corr:+.4f} [{lo:+.4f},{hi:+.4f}]   "
              f"variance of I explained = {ss:+.3f}   {100 * corr / ceiling:5.1f}% of ceiling",
              flush=True)
    res["_ceiling"] = dict(split_half=float(r_half), full=float(r_full), ceiling=float(ceiling),
                           n_pairs=len(P), n_genes=int(G), n_control_cells=int(len(ctrl)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/norman2019_subset.npz")
    ap.add_argument("--top-genes", type=int, default=1000)
    ap.add_argument("--min-single", type=int, default=60); ap.add_argument("--min-double", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--out", default="outputs/results.json")
    args = ap.parse_args()
    counts, totals, lab = load(args.data)
    res = analyse(counts, totals, lab, args)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
