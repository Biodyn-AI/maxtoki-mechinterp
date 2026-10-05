"""transfer — frozen K562 -> RPE1 transfer of a cell-cycle phase readout: C2S-Scale-Gemma-2-2B layer-21
activations versus expression encoded the way the model receives it.

A ridge readout of (cos theta, sin theta) is fit on K562 cells and applied unchanged to RPE1 cells.
theta is the oriented cell-cycle phase angle from code/cc_phase.py, computed separately in each cell line.
The same frozen-transfer protocol is run on a ladder of expression encodings that ends at the model's input:

    expr_full      all shared genes, magnitudes
    expr_512_mag   top-512 genes per cell, magnitudes
    expr_512_rank  top-512 genes per cell, rank only      <- the information in a cell sentence
    model          C2S layer-21 cell-summary activations

A C2S cell sentence (code/cell_sentences.py) lists a cell's most highly expressed genes in descending order,
so it carries which genes are present and their order, but not their magnitudes.

Then a PAIRED bootstrap over target cells: the same resampled cell indices are used for every arm, so the
difference between two arms is measured on the same cells and its CI is not widened by between-cell
variance that both arms share.

METRICS. R_diff = |mean(exp(i(pred-true)))|, rotation-invariant and fold-free, plus median angular error
after best rotation. Both are read against a measured constant-predictor floor and a uniform-random floor.
Circular correlation is not used: the RPE1 phase is close to uniform, and in that regime it is unstable.

Run from the package root:   python code/transfer.py
Out: outputs/transfer_results.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle_oriented
from data_io import load_cells


def R_diff(a, b):
    return float(np.abs(np.mean(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def med_err(a, b):
    off = np.angle(np.mean(np.exp(1j * (np.asarray(b) - np.asarray(a)))))
    d = np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b) + off)))
    return float(np.degrees(np.median(np.abs(d))))


def top_k_encode(F, cols, k, values):
    """Top-k encoding of each cell, returned as a (cells x len(cols)) matrix over the shared genes.

    F is the cell line's expression matrix and cols are the shared-gene column indices into F.
    values="magnitude" keeps the expression value of each kept gene; values="rank" stores its position in
    the cell's descending order instead (k for the top gene, 1 for the k-th). Genes not kept are 0.
    """
    col_of = np.full(F.shape[1], -1); col_of[cols] = np.arange(len(cols))
    Y = np.zeros((len(F), len(cols)))
    for i in range(len(F)):
        idx = np.argpartition(-F[i], k)[:k]
        idx = idx[np.argsort(-F[i][idx])]
        vals = F[i, idx] if values == "magnitude" else np.arange(k, 0, -1)
        keep = col_of[idx] >= 0
        Y[i, col_of[idx][keep]] = vals[keep]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--k", type=int, default=512, help="genes per cell sentence, must match extraction")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="outputs/transfer_results.json")
    a = ap.parse_args()
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    src = load_cells(a.data, "k562"); tgt = load_cells(a.data, "rpe1")
    ts, _, _, _ = phase_angle_oriented(src); tt, _, _, _ = phase_angle_oriented(tgt)
    src_dir = os.path.join(a.data, "act_k562"); tgt_dir = os.path.join(a.data, "act_rpe1")
    si_ = np.load(os.path.join(src_dir, "row_cell_ids.npy"))
    ti_ = np.load(os.path.join(tgt_dir, "row_cell_ids.npy"))
    Hs = np.load(os.path.join(src_dir, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    Ht = np.load(os.path.join(tgt_dir, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    ts_m = ts[si_][:len(Hs)]; Hs = Hs[:len(ts_m)]
    tt_m = tt[ti_][:len(Ht)]; Ht = Ht[:len(tt_m)]

    sv = np.char.upper(np.asarray(src.var_names).astype(str))
    tv = np.char.upper(np.asarray(tgt.var_names).astype(str))
    shared = sorted(set(sv) & set(tv))
    sx = {g: i for i, g in enumerate(sv)}; tx = {g: i for i, g in enumerate(tv)}
    sc = np.array([sx[g] for g in shared]); tc = np.array([tx[g] for g in shared])
    Fs = src.X.toarray()[si_][:len(ts_m)]
    Ft = tgt.X.toarray()[ti_][:len(tt_m)]
    Xs = Fs[:, sc]; Xt = Ft[:, tc]
    print(f"source {Hs.shape} | target {Ht.shape} | shared genes {len(shared)} | k={a.k}", flush=True)

    def frozen(A, B):
        Y = np.column_stack([np.cos(ts_m), np.sin(ts_m)])
        s = StandardScaler().fit(A)
        P = Ridge(alpha=1e3).fit(s.transform(A), Y).predict(s.transform(B))
        return np.arctan2(P[:, 1], P[:, 0])

    preds = {"model": frozen(Hs, Ht),
             "expr_full": frozen(Xs, Xt),
             "expr_512_mag": frozen(top_k_encode(Fs, sc, a.k, "magnitude"), top_k_encode(Ft, tc, a.k, "magnitude")),
             "expr_512_rank": frozen(top_k_encode(Fs, sc, a.k, "rank"), top_k_encode(Ft, tc, a.k, "rank"))}
    preds["constant"] = np.full_like(tt_m, np.angle(np.mean(np.exp(1j * tt_m))))
    preds["random"] = np.random.default_rng(0).uniform(0, 2 * np.pi, len(tt_m))

    order = ["expr_full", "expr_512_mag", "expr_512_rank", "model", "constant", "random"]
    print(f"\n{'arm':<16}{'R_diff':>9}{'med err':>10}", flush=True)
    res = {"n_target": int(len(tt_m)), "n_shared_genes": len(shared), "k": a.k, "layer": a.layer,
           "target_phase_R": float(np.abs(np.mean(np.exp(1j * tt_m)))), "point": {}}
    for nm in order:
        res["point"][nm] = dict(R_diff=R_diff(preds[nm], tt_m), median_err_deg=med_err(preds[nm], tt_m))
        print(f"{nm:<16}{res['point'][nm]['R_diff']:>9.3f}{res['point'][nm]['median_err_deg']:>9.0f}d", flush=True)

    # paired bootstrap: the same resampled target cells for every arm
    rng = np.random.default_rng(0); n = len(tt_m)
    boot = {nm: np.empty(a.n_boot) for nm in order}
    for b in range(a.n_boot):
        i = rng.integers(0, n, n)
        for nm in order:
            boot[nm][b] = R_diff(preds[nm][i], tt_m[i])
    print(f"\nPAIRED CONTRASTS ({a.n_boot} draws, same cells per draw)", flush=True)
    print(f"  {'contrast':<34}{'delta R_diff':>14}{'95% CI':>22}{'P(>0)':>9}", flush=True)
    res["contrasts"] = {}
    for x, y in [("model", "expr_512_rank"), ("model", "expr_full"),
                 ("expr_full", "expr_512_rank"), ("expr_full", "expr_512_mag"),
                 ("expr_512_mag", "expr_512_rank")]:
        d = boot[x] - boot[y]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(np.mean(d > 0))
        res["contrasts"][f"{x}_minus_{y}"] = dict(delta=float(d.mean()), ci=[float(lo), float(hi)], p_gt_0=p)
        print(f"  {x + ' - ' + y:<34}{d.mean():>+14.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]{p:>9.3f}", flush=True)

    dm = boot["model"] - boot["expr_512_rank"]
    lo, hi = np.percentile(dm, [2.5, 97.5])
    verdict = ("PARITY — the CI on model minus its own input encoding spans 0"
               if lo <= 0 <= hi else
               ("MODEL ADDS information beyond its input encoding" if lo > 0 else
                "MODEL LOSES information relative to its input encoding"))
    res["verdict"] = verdict
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"  tokenisation loss (expr_full - expr_512_rank) = "
          f"{res['contrasts']['expr_full_minus_expr_512_rank']['delta']:+.4f} "
          f"{res['contrasts']['expr_full_minus_expr_512_rank']['ci']}", flush=True)
    print(f"  model vs its own input               = {dm.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
