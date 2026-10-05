"""transfer_test - zero-shot cross-cell-line transfer of a cell-cycle phase readout.

    fit the phase readout on K562  ->  apply it, frozen, to RPE1

The readout never sees the target dataset, so a readout that works on RPE1 carries cell-cycle structure that
survives a change of cell line.

DESIGN
  SOURCE  K562 Replogle non-targeting controls (n = 3,000).
  TARGET  RPE1 Replogle non-targeting controls (n = 3,000).
  RPE1 is one cell type, so phase cannot be confounded with cell identity, and 93 of the 94 markers are in
  its panel. RPE1 comes from the same lab and platform as K562, so this tests transfer across BIOLOGY
  (aneuploid erythroleukemia -> near-diploid retinal epithelium) and NOT across batch or platform.

  Both readouts are fit ON SOURCE ONLY and applied frozen to TARGET:
    model       ridge(cos phi, sin phi) on C2S cell-state activations (layer 21)
    expression  ridge(cos phi, sin phi) on expression restricted to the SHARED gene space
  Ground truth in each dataset is its own Tirosh-marker phase, computed independently per dataset and
  oriented the same way (cc_phase.phase_angle_oriented).

METRICS
  within    circular correlation on held-out SOURCE cells (5-fold CV; what each readout achieves at home)
  transfer  circular correlation on TARGET cells, readout frozen
  retention = transfer / within
  The median angular error (deg) is reported next to each.

Out: outputs/transfer_rpe1.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle_oriented

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANCE_ERR_DEG = 90.0


def circ_corr(a, b):
    """Jammalamadaka circular correlation coefficient."""
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def med_err(a, b):
    return float(np.rad2deg(np.median(circ_dist(a, b))))


def fit_apply(Xs, ts, Xt, tt, alpha=1e3, seed=0):
    """Fit ridge(cos,sin) on SOURCE, report within-source (held-out) and frozen TARGET transfer."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    Ys = np.column_stack([np.cos(ts), np.sin(ts)])
    # within-source, out-of-fold
    P = np.zeros_like(Ys)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        sc = StandardScaler().fit(Xs[tr])
        P[te] = Ridge(alpha=alpha).fit(sc.transform(Xs[tr]), Ys[tr]).predict(sc.transform(Xs[te]))
    wh = np.arctan2(P[:, 1], P[:, 0])
    within = circ_corr(wh, ts)
    within_err = med_err(wh, ts)
    # frozen transfer
    sc = StandardScaler().fit(Xs)
    m = Ridge(alpha=alpha).fit(sc.transform(Xs), Ys)
    Pt = m.predict(sc.transform(Xt))
    that = np.arctan2(Pt[:, 1], Pt[:, 0])
    transfer = circ_corr(that, tt)
    transfer_err = med_err(that, tt)
    return dict(within=within, within_median_err_deg=within_err,
                transfer=transfer, transfer_median_err_deg=transfer_err,
                retention=float(transfer / (within + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-states", default=os.path.join(ROOT, "data", "k562", "activations"))
    ap.add_argument("--tgt-states", default=os.path.join(ROOT, "data", "rpe1", "activations"))
    ap.add_argument("--src-h5ad", default=os.path.join(ROOT, "data", "k562", "k562_controls.h5ad"))
    ap.add_argument("--tgt-h5ad", default=os.path.join(ROOT, "data", "rpe1", "rpe1_controls.h5ad"))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "transfer_rpe1.json"))
    a = ap.parse_args()
    import anndata, scipy.sparse as sp

    src = anndata.read_h5ad(a.src_h5ad); tgt = anndata.read_h5ad(a.tgt_h5ad)
    ts_phase, _, _, si = phase_angle_oriented(src)
    tt_phase, _, _, ti = phase_angle_oriented(tgt)
    print(f"SOURCE {src.shape} ({si['n_cc_genes']} cc genes) | TARGET {tgt.shape} ({ti['n_cc_genes']} cc genes)",
          flush=True)
    for nm, i in [("SOURCE", si), ("TARGET", ti)]:
        print(f"  {nm} orientation: flipped={i['flipped']} rot={i['rotation_deg']}deg -> "
              f"S peak {i['s_peak_deg']}deg, G2M peak {i['g2m_peak_deg']}deg (sep {i['s_g2m_separation_deg']}deg)",
              flush=True)

    # ---- MODEL ----
    si_ = np.load(os.path.join(a.src_states, "row_cell_ids.npy"))
    ti_ = np.load(os.path.join(a.tgt_states, "row_cell_ids.npy"))
    Hs = np.load(os.path.join(a.src_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    Ht = np.load(os.path.join(a.tgt_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    ts_m = ts_phase[si_][:len(Hs)]; Hs = Hs[:len(ts_m)]
    tt_m = tt_phase[ti_][:len(Ht)]; Ht = Ht[:len(tt_m)]
    model = fit_apply(Hs, ts_m, Ht, tt_m)

    # ---- EXPRESSION, on the SHARED gene space ----
    sv = np.char.upper(np.asarray(src.var_names).astype(str))
    tv = np.char.upper(np.asarray(tgt.var_names).astype(str))
    shared = sorted(set(sv) & set(tv))
    sidx = {g: i for i, g in enumerate(sv)}; tidx = {g: i for i, g in enumerate(tv)}
    Xs = src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X)
    Xt = tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X)
    Xs = Xs[:, [sidx[g] for g in shared]][si_][:len(ts_m)]
    Xt = Xt[:, [tidx[g] for g in shared]][ti_][:len(tt_m)]
    print(f"shared gene space: {len(shared)} genes", flush=True)
    expr = fit_apply(Xs, ts_m, Xt, tt_m)

    res = dict(n_shared_genes=len(shared), n_source=int(len(Hs)), n_target=int(len(Ht)),
               model=model, expression=expr)
    print(f"\n  {'readout':<12}{'within (src)':>16}{'transfer (tgt)':>18}{'retention':>12}", flush=True)
    for nm, r in [("MODEL", model), ("EXPRESSION", expr)]:
        print(f"  {nm:<12}{r['within']:>9.3f} ({r['within_median_err_deg']:>3.0f}deg)"
              f"{r['transfer']:>11.3f} ({r['transfer_median_err_deg']:>3.0f}deg){r['retention']:>12.2f}", flush=True)
    # VALIDITY GATE: a readout counts as transferring only if it is positively associated with the target
    # phase and its median error is below chance.
    valid = lambda r: (r["transfer"] > 0) and (r["transfer_median_err_deg"] < CHANCE_ERR_DEG)
    res["model_valid"], res["expression_valid"] = valid(model), valid(expr)
    ok = [nm for nm, v in [("MODEL", res["model_valid"]), ("EXPRESSION", res["expression_valid"])] if v]
    if not ok:
        res["verdict"] = "NEITHER - both below chance; target label is invalid for this variable"
        print(f"\n  !! NO VALID TRANSFER: model {model['transfer']:.3f} ({model['transfer_median_err_deg']:.0f}deg), "
              f"expression {expr['transfer']:.3f} ({expr['transfer_median_err_deg']:.0f}deg) -- both worse than "
              f"chance ({CHANCE_ERR_DEG:.0f}deg). The target's phase label is not cell cycle.", flush=True)
    else:
        res["verdict"] = "VALID TRANSFER: " + ", ".join(ok)
        print(f"\n  VALID ZERO-SHOT TRANSFER: {', '.join(ok)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    out_shown = os.path.relpath(os.path.abspath(a.out), ROOT)
    print(f"[done] -> {out_shown if not out_shown.startswith('..') else a.out}", flush=True)


if __name__ == "__main__":
    main()
