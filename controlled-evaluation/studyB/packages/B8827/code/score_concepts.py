"""Best single SAE feature per UniProt residue concept (ESM-2 650M, layer 16, TopK SAE).

Inputs (package data/ folder):
    sae_codes.npz        top-32 SAE feature codes for every residue
    residue_labels.npz   UniProt feature labels for every residue, plus protein ids

Outputs (package outputs/ folder):
    best_feature_per_concept.csv / .json   one row per scored concept
    summary.json                           headline counts

Run from the package root:
    python code/score_concepts.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 42             # protein split
N_THRESHOLDS = 20     # quantiles 0.00 .. 0.95 of a feature's nonzero split-A activations
MIN_NONZERO = 50      # nonzero split-A activations a feature needs to be scored
MIN_PREVALENCE = 1e-3
MIN_POSITIVES = 10    # positive residues a concept needs in each split
MIN_F1 = 0.5          # InterPLM headline threshold
MIN_MARGIN = 0.1      # F1 above the random baseline at the same activation rate


def split_by_protein(protein: np.ndarray, frac: float = 0.5, seed: int = SEED):
    """Residue masks for a 50/50 split of the proteins (no protein in both halves)."""
    rng = np.random.default_rng(seed)
    prots = np.unique(protein)
    rng.shuffle(prots)
    in_a = np.zeros(protein.max() + 1, dtype=bool)
    in_a[prots[: int(frac * len(prots))]] = True
    mask_a = in_a[protein]
    return mask_a, ~mask_a


def expected_random_f1(rate: float, prevalence: float) -> float:
    """F1 of a predictor that fires on a random `rate` fraction of residues: 2ap/(a+p)."""
    a, p = rate, prevalence
    return 2 * a * p / (a + p) if (a + p) > 0 else 0.0


def f1_from_counts(tp, n_pred, n_pos):
    """F1 = 2TP / (2TP + FP + FN) = 2TP / (n_predicted + n_positive)."""
    denom = np.asarray(n_pred + n_pos, dtype=float)
    return np.divide(2.0 * tp, denom, out=np.zeros(denom.shape), where=denom > 0)


def load():
    c = np.load(DATA / "sae_codes.npz")
    lab = np.load(DATA / "residue_labels.npz")
    feat, val = c["feature"].astype(np.int64), c["value"].astype(np.float32)
    n_res, k = feat.shape
    n_feat = int(c["n_features"])
    keep = val.ravel() > 0
    rows = np.repeat(np.arange(n_res), k)[keep]
    codes = sparse.csc_matrix((val.ravel()[keep], (rows, feat.ravel()[keep])),
                              shape=(n_res, n_feat))
    codes.sort_indices()
    return codes, lab["labels"].astype(bool), [str(x) for x in lab["concepts"]], \
        lab["protein"].astype(np.int64)


def score_all(codes, Y, mask_a, mask_b):
    """Every (feature, concept) pair: threshold picked by split-A F1, then scored on split B.

    Returns arrays of shape (n_features, n_concepts); features with fewer than
    MIN_NONZERO nonzero split-A activations are NaN.
    """
    n_feat, n_c = codes.shape[1], Y.shape[1]
    Yi = Y.astype(np.int32)
    pos_a = Y[mask_a].sum(0)
    pos_b = Y[mask_b].sum(0)
    n_b = int(mask_b.sum())
    qs = np.linspace(0.0, 0.95, N_THRESHOLDS)
    names = ("f1", "f1_a", "precision", "recall", "threshold", "activation_rate")
    out = {n: np.full((n_feat, n_c), np.nan) for n in names}

    for f in range(n_feat):
        lo, hi = codes.indptr[f], codes.indptr[f + 1]
        r, v = codes.indices[lo:hi], codes.data[lo:hi]
        in_a = mask_a[r]
        ra, va = r[in_a], v[in_a]
        if va.size < MIN_NONZERO:
            continue
        rb, vb = r[~in_a], v[~in_a]

        thrs = np.unique(np.quantile(va, qs))
        pred_a = (va[:, None] > thrs[None, :]).astype(np.int32)            # (nz_a, T)
        tp_a = pred_a.T @ Yi[ra]                                             # (T, C)
        f1_grid = f1_from_counts(tp_a, pred_a.sum(0)[:, None], pos_a[None, :])
        j = f1_grid.argmax(0)                                                # (C,)
        t = thrs[j]

        pred_b = vb[:, None] > t[None, :]                                    # (nz_b, C)
        tp_b = (pred_b & Y[rb]).sum(0)
        np_b = pred_b.sum(0)
        out["f1"][f] = f1_from_counts(tp_b, np_b, pos_b)
        out["f1_a"][f] = f1_grid[j, np.arange(n_c)]
        out["precision"][f] = np.divide(tp_b, np_b, out=np.zeros(n_c), where=np_b > 0)
        out["recall"][f] = np.divide(tp_b, pos_b, out=np.zeros(n_c), where=pos_b > 0)
        out["threshold"][f] = t
        out["activation_rate"][f] = np_b / n_b
    return out


def main() -> int:
    t0 = time.time()
    codes, Y_all, concepts_all, protein = load()
    mask_a, mask_b = split_by_protein(protein)

    prev_all = Y_all.mean(0)
    pos_a_all, pos_b_all = Y_all[mask_a].sum(0), Y_all[mask_b].sum(0)
    keep = [i for i in range(len(concepts_all))
            if prev_all[i] >= MIN_PREVALENCE
            and min(pos_a_all[i], pos_b_all[i]) >= MIN_POSITIVES]
    concepts = [concepts_all[i] for i in keep]
    Y = Y_all[:, keep]
    print(f"{len(np.unique(protein))} proteins, {len(protein):,} residues "
          f"(split A {int(mask_a.sum()):,} / split B {int(mask_b.sum()):,}); "
          f"{len(concepts)} concepts scored")

    s = score_all(codes, Y, mask_a, mask_b)
    n_scored_feats = int((~np.isnan(s["f1"][:, 0])).sum())
    prev_b = Y[mask_b].mean(0)

    rows = []
    for ci, c in enumerate(concepts):
        best = int(np.nanargmax(s["f1_a"][:, ci]))
        rate = float(s["activation_rate"][best, ci])
        rand = expected_random_f1(rate, float(prev_b[ci]))
        f1 = float(s["f1"][best, ci])
        rows.append({
            "concept": c,
            "prevalence": float(Y[:, ci].mean()),
            "n_pos_a": int(Y[mask_a, ci].sum()),
            "n_pos_b": int(Y[mask_b, ci].sum()),
            "feature": best,
            "f1": f1,
            "precision": float(s["precision"][best, ci]),
            "recall": float(s["recall"][best, ci]),
            "threshold": float(s["threshold"][best, ci]),
            "activation_rate": rate,
            "random_f1": rand,
            "f1_margin": f1 - rand,
            "f1_a": float(s["f1_a"][best, ci]),
            "passes": bool(f1 >= MIN_F1 and f1 - rand >= MIN_MARGIN),
        })
    rows.sort(key=lambda d: -d["f1"])

    OUT.mkdir(exist_ok=True)
    (OUT / "best_feature_per_concept.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "best_feature_per_concept.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for d in rows:
            w.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in d.items()})

    f1s = np.array([d["f1"] for d in rows])
    summary = {
        "n_proteins": int(len(np.unique(protein))),
        "n_residues": int(len(protein)),
        "n_residues_split_a": int(mask_a.sum()),
        "n_residues_split_b": int(mask_b.sum()),
        "n_features": int(codes.shape[1]),
        "n_features_scored": n_scored_feats,
        "n_concepts_scored": len(concepts),
        "gate": {"min_f1": MIN_F1, "min_margin": MIN_MARGIN},
        "n_concepts_passing": int(sum(d["passes"] for d in rows)),
        "concepts_passing": [d["concept"] for d in rows if d["passes"]],
        "mean_best_f1": float(f1s.mean()),
        "median_best_f1": float(np.median(f1s)),
        "mean_best_margin": float(np.mean([d["f1_margin"] for d in rows])),
        "n_concepts_margin_above_0.1": int(sum(d["f1_margin"] >= MIN_MARGIN for d in rows)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'concept':<10} {'feat':>5} {'F1':>6} {'rand':>6} {'margin':>7} {'prev':>7}")
    for d in rows:
        print(f"{d['concept']:<10} {d['feature']:>5} {d['f1']:6.3f} {d['random_f1']:6.3f} "
              f"{d['f1_margin']:+7.3f} {d['prevalence']:7.4f}{'  *' if d['passes'] else ''}")
    print(f"{summary['n_concepts_passing']}/{len(concepts)} concepts pass "
          f"F1>={MIN_F1} and margin>={MIN_MARGIN}; mean best F1 {summary['mean_best_f1']:.3f}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
