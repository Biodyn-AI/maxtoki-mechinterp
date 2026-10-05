"""Cell-cycle loop geometry for one representation: circular decodability and a flat-circle null.

For one representation (scgpt, geneformer or expr) this script:
  1. Reduces the representation to 20 whitened principal components (cc_common.prep).
  2. Circular decodability of the phase phi (circ_r2), 5-fold cross-validated:
       linear: ridge regression (alpha 10) from the 20 PCs to (cos phi, sin phi);
       kNN:    k-nearest-neighbour regression (k = 15) to the same target.
     Predicted phase = atan2 of the two predictions.
     circ-R2 = 1 - mean(1 - cos(error)) / mean(1 - cos(phi - circular mean of phi)).
     1 = perfect, 0 = no better than always predicting the mean phase.
  3. Local tangent of the loop in each of 12 equal phase sectors (loop_tangents): within a sector, ridge
     regression of the phase offset from the sector centre on the 20 PCs; the unit coefficient vector is the
     local direction in which phase increases.
  4. Shape of the tangent field (turning_stats): total turning (sum of angles between neighbouring sector
     tangents, all the way round), planarity (share of the tangent field's energy in its best-fit 2-plane)
     and out-of-plane angle (mean angle by which the sector tangents leave that 2-plane).
  5. Flat-circle null (flat_null): a synthetic representation of the same width whose phase signal lies
     exactly in one random 2-plane, (cos phi, sin phi) @ A, plus isotropic Gaussian noise. The noise level is
     set by binary search so that its linear circ-R2 matches the real one. 20 draws.
     The flat-circle hypothesis is rejected only if the real out-of-plane angle is above the 95th percentile
     of the 20 null draws. One-sided p = (#null >= real + 1) / 21, so the smallest possible p is 0.048.

Run from the package root:
  python code/cc_geometry.py scgpt [--out DIR] [--max-new-draws N]
Writes DIR/cc_geometry_<model>.json and DIR/cc_geometry_<model>.log (default DIR = outputs/).
Each null draw is saved in DIR/null_draws_<model>.json as soon as it finishes. A call reuses the draws already
saved there, so a long run (expr) can be split over several calls with --max-new-draws. The final JSON is
written once all 20 draws are present.
"""
import os, sys, json, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import ROOT, wrap, circ_mean, unit, ang, prep, emb_path, DIM  # noqa: E402

NBINS, SEED = 12, 0
N_NULL = 20
MODELS = ["scgpt", "geneformer", "expr"]


# ------------------------------------------------------------------ circular decodability
def circ_r2(Xz, phi, model="linear", seed=SEED):
    """Circular R^2 for predicting phi from Xz. 1 - E[1-cos(err)] / E[1-cos(phi - circmean)]."""
    Y = np.column_stack([np.cos(phi), np.sin(phi)])
    P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xz):
        m = Ridge(alpha=10.0) if model == "linear" else KNeighborsRegressor(15)
        P[te] = m.fit(Xz[tr], Y[tr]).predict(Xz[te])
    err = 1.0 - np.cos(np.arctan2(P[:, 1], P[:, 0]) - phi)
    base = float(np.mean(1.0 - np.cos(phi - circ_mean(phi))))
    return float(1.0 - err.mean() / (base + 1e-12))


# ------------------------------------------------------------------ tangents around the loop
def loop_tangents(Xz, phi, nbins=NBINS):
    """Local phase-gradient direction in each of `nbins` equal phase sectors.

    Within a sector the target is the wrapped offset wrap(phi - sector centre), so the sector that contains
    the -pi/pi boundary is handled like the others.
    """
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    dirs, ns = [], []
    for b in range(nbins):
        m = (phi >= edges[b]) & (phi < edges[b + 1])
        idx = np.where(m)[0]
        if len(idx) < 20:
            dirs.append(None); ns.append(len(idx)); continue
        c = 0.5 * (edges[b] + edges[b + 1])
        t = wrap(phi[idx] - c)
        dirs.append(unit(Ridge(alpha=10.0).fit(Xz[idx], t).coef_))
        ns.append(len(idx))
    return dirs, ns


def turning_stats(dirs):
    """Total turning round the loop, planarity of the tangent field, mean out-of-plane angle.

    total_turning: sum of angles between consecutive sector tangents, wrapping round (degrees). A smooth
      closed curve traced once in a plane gives 360; noise in the tangents adds to it.
    planarity: share of the tangent field's energy in its best-fit 2-plane (1.0 for a noise-free planar loop).
    out_of_plane_deg: mean angle by which the sector tangents leave that 2-plane.
    """
    D = [d for d in dirs if d is not None]
    if len(D) < 4:
        return dict(total_turning=float("nan"), planarity=float("nan"), out_of_plane_deg=float("nan"))
    D = np.array(D); n = len(D)
    total = float(sum(ang(D[i], D[(i + 1) % n]) for i in range(n)))
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    planarity = float((S[:2] ** 2).sum() / (S ** 2).sum())
    P2 = Vt[:2]
    oop = float(np.mean([np.degrees(np.arccos(np.clip(np.linalg.norm(P2 @ d), 0, 1))) for d in D]))
    return dict(total_turning=total, planarity=planarity, out_of_plane_deg=oop)


# ------------------------------------------------------------------ flat-circle null
def flat_null(phi, d, lin_r2, seed=0):
    """Synthetic d-wide representation whose phase signal lies exactly in one linear 2-plane, with isotropic
    noise chosen so that its linear circ-R2 matches lin_r2. Returned after the same prep() as the real data."""
    rng = np.random.default_rng(seed)
    base = np.column_stack([np.cos(phi), np.sin(phi)])
    A = rng.standard_normal((2, d))
    sig = base @ A
    sig /= sig.std()
    lo, hi = 0.0, 20.0
    for _ in range(18):
        s = 0.5 * (lo + hi)
        E = prep(sig + s * rng.standard_normal(sig.shape), DIM)
        if circ_r2(E, phi, "linear") > lin_r2:
            lo = s
        else:
            hi = s
    return prep(sig + 0.5 * (lo + hi) * rng.standard_normal(sig.shape), DIM)


def run(model, out_dir, max_new_draws=None):
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    z = np.load(emb_path(model), allow_pickle=False)
    X, phi = z["emb"].astype(np.float64), z["phi"].astype(np.float64)
    Xz = prep(X, DIM)

    lin = circ_r2(Xz, phi, "linear")
    knn = circ_r2(Xz, phi, "knn")
    dirs, ns = loop_tangents(Xz, phi)
    real = turning_stats(dirs)
    print(f"[{model}] linear circ-R2 = {lin:.4f} | kNN = {knn:.4f}", flush=True)

    # Each null draw is independent (its own seed) and is saved as soon as it is done, so a long run can be
    # split over several calls; draws already in the file are reused.
    f_draws = os.path.join(out_dir, f"null_draws_{model}.json")
    draws = json.load(open(f_draws)) if os.path.exists(f_draws) else {}
    n_new = 0
    for s in range(N_NULL):
        key = str(100 + s)
        if key in draws:
            continue
        if max_new_draws is not None and n_new >= max_new_draws:
            break
        Xn = flat_null(phi, X.shape[1], lin, seed=100 + s)
        dn, _ = loop_tangents(Xn, phi)
        draws[key] = turning_stats(dn)
        n_new += 1
        json.dump(draws, open(f_draws, "w"), indent=1)
        print(f"  null draw seed {key} done ({len(draws)}/{N_NULL})", flush=True)
    if len(draws) < N_NULL:
        print(f"[{model}] {len(draws)}/{N_NULL} null draws saved in {f_draws}; run again to continue.")
        return None
    nulls = [draws[str(100 + s)] for s in range(N_NULL)]
    def nstat(k):
        v = np.array([n[k] for n in nulls])
        return float(v.mean()), float(v.std())

    tt_m, tt_s = nstat("total_turning")
    pl_m, pl_s = nstat("planarity")
    oo_m, oo_s = nstat("out_of_plane_deg")

    oop_null = np.array([n["out_of_plane_deg"] for n in nulls])
    p_oop = float((np.sum(oop_null >= real["out_of_plane_deg"]) + 1) / (len(oop_null) + 1))
    rejected = bool(real["out_of_plane_deg"] > np.percentile(oop_null, 95))

    out = dict(model=model, n=int(len(phi)), input_dim=int(X.shape[1]), dim=DIM,
               linear_circ_r2=lin, knn_circ_r2=knn, decodability_gap=knn - lin,
               real=real, bin_n=ns,
               flat_null=dict(total_turning_mean=tt_m, total_turning_sd=tt_s,
                              planarity_mean=pl_m, planarity_sd=pl_s,
                              out_of_plane_mean=oo_m, out_of_plane_sd=oo_s,
                              out_of_plane_draws=[float(v) for v in oop_null], n_draws=N_NULL),
               out_of_plane_excess=real["out_of_plane_deg"] - oo_m, p_out_of_plane=p_oop,
               H_flat_rejected=rejected)

    log(f"===== cell-cycle loop geometry / {model}  (n={len(phi)}, input dim {X.shape[1]}, "
        f"{DIM}-PC whitened) =====")
    log(f"  circular decodability:  linear circ-R2 = {lin:.3f} | kNN = {knn:.3f}"
        f" | kNN - linear = {knn-lin:+.3f}")
    log(f"  cells per phase sector: {ns}")
    log(f"  total turning round the loop = {real['total_turning']:.0f} deg"
        f"   (flat-circle null {tt_m:.0f} +- {tt_s:.0f})")
    log(f"  planarity of the tangent field = {real['planarity']:.3f}  (null {pl_m:.3f} +- {pl_s:.3f})")
    log(f"  out-of-plane tangent angle     = {real['out_of_plane_deg']:.1f} deg"
        f"  (null {oo_m:.1f} +- {oo_s:.1f})  excess = {real['out_of_plane_deg']-oo_m:+.1f} deg,"
        f" p = {p_oop:.3f}")
    log(f"  -> flat-circle hypothesis (loop lies in one linear 2-plane): "
        f"{'REJECTED' if rejected else 'NOT REJECTED'}")

    f = os.path.join(out_dir, f"cc_geometry_{model}.json")
    json.dump(out, open(f, "w"), indent=1)
    open(os.path.join(out_dir, f"cc_geometry_{model}.log"), "w").write("\n".join(lines) + "\n")
    print(f"[written] {f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--max-new-draws", type=int, default=None,
                    help="stop after this many new null draws (the rest can be run by a later call)")
    a = ap.parse_args()
    run(a.model, a.out, a.max_new_draws)
