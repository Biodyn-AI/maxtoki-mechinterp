"""v2 triplets follow-up (revision item D5): two checks on top of outputs/v2_triplets.

Stage `classnull` (no model; reads outputs/v2_triplets/cells/*.npz):
  * Sign-flip null for the two "superadditive" classification rules (spec rule on Cohen's d_z and the
    deployed 1.1x rule on raw means). Under H0 "no consistent three-way interaction", a cell's
    interaction vector I_c is as likely to be +I_c as -I_c. So ABC*_c = (AB + AC + BC - A - B - C)_c
    + s_c * I_c with random signs s_c has the same distribution as the observed ABC_c. Applying the
    rules to ABC* gives the counts expected from noise alone.
  * Per-target power: plant an interaction into d_ABC of 64 triplets and count how many targets pass
    a one-sample t-test with BH (within triplet, and across the 64 x 4,928 tests).

Stage `readout` (model; 4 cells x 4 triplets x 8 forward passes):
  * Per-position three-way interaction I = ABC - AB - AC - BC + A + B + C of the L11 residual
    (h11, dense), the L11 SAE pre-activation (dense, linear in h11), the L11 SAE code (TopK) and the
    logits. Shows where the per-cell non-additivity of the SAE codes comes from.
  * Cross-check: the position-mean code change of each condition must equal the value stored by
    v2_triplets.py for the same cell and condition.

Usage:
  PY=projects/maxtoki/.venv/bin/python
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets_followup.py classnull
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets_followup.py readout --max-minutes 8   (repeat until DONE)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
SRC = RUN / "outputs/v2_triplets"
OUT = RUN / "outputs/v2_triplets_followup"
SCRIPT_PATH = Path(__file__).resolve()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NONE = -1
SEED = 20261001
N_FLIP_CLASS = 40
MIN_FREE_GB = 3.0
TGT = 11


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def update_run_config(section, entry):
    p = OUT / "run_config.json"
    cfg = load_json(p) if p.exists() else {}
    if section in ("env", "seeds", "cells", "inputs"):
        cfg[section] = entry
    else:
        cfg.setdefault(section, []).append(dict(entry, script_sha256=H.sha256_file(SCRIPT_PATH)))
    cfg["script"] = str(SCRIPT_PATH)
    cfg["script_sha256"] = H.sha256_file(SCRIPT_PATH)
    H.write_json(p, cfg)


def build_design(sel):
    """Identical to v2_triplets.build_design (copied so the main script is not imported)."""
    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    NA, NB, NC = sel["null_features"]["0"], sel["null_features"]["5"], sel["null_features"]["9"]
    triplets = []
    for a, b, c in itertools.product(A, B, C):
        triplets.append({"a": a, "b": b, "c": c, "kind": "real"})
    for a, b, c in itertools.product(NA, NB, NC):
        triplets.append({"a": a, "b": b, "c": c, "kind": "null_all"})
    for i in range(len(A)):
        triplets.append({"a": A[i], "b": B[i], "c": NC[0], "kind": "null_C"})
        triplets.append({"a": A[i], "b": NB[0], "c": C[i], "kind": "null_B"})
        triplets.append({"a": NA[0], "b": B[i], "c": C[i], "kind": "null_A"})
    conds = set()
    for t in triplets:
        a, b, c = t["a"], t["b"], t["c"]
        for keep in itertools.product([0, 1], repeat=3):
            if sum(keep) == 0:
                continue
            conds.add((a if keep[0] else NONE, b if keep[1] else NONE, c if keep[2] else NONE))
    for (a, b, c) in list(conds):
        if b != NONE:
            conds.add((a, b, NONE))
        if a != NONE:
            conds.add((a, NONE, NONE))
    conds = sorted(conds)
    a_groups = [NONE] + list(A) + list(NA)
    return triplets, conds, a_groups


def load_D(n_cells, conds, a_groups):
    cidx = {c: i for i, c in enumerate(conds)}
    D = np.zeros((n_cells, len(conds), 4928), dtype=np.float32)
    seen = np.zeros((n_cells, len(conds)), dtype=bool)
    for ci in range(n_cells):
        for gi in range(len(a_groups)):
            g = np.load(SRC / "cells" / f"cell{ci:02d}_g{gi:02d}.npz")
            for r, k in enumerate(map(tuple, g["conds"])):
                j = cidx[k]
                D[ci, j] = g["dz"][r]
                seen[ci, j] = True
    assert seen.all()
    return D, cidx


def bh(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def ttest_p(x):
    from scipy import stats
    n = x.shape[0]
    m = x.mean(0)
    sd = x.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return np.where(sd > 0, p, np.where(m == 0, 1.0, 0.0))


def classify(Dc, abc):
    """Return (spec_included, spec_strict, spec_band, spec_add, spec_sub, dep_targets, dep_super)
    for one triplet. Dc: dict of (n, T) arrays for A..BC; abc: (n, T) array used as ABC."""
    dd = {}
    allc = dict(Dc, ABC=abc)
    for k, X in allc.items():
        m = X.mean(0)
        sd = X.std(0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd[k] = np.where(sd > 0, m / sd, 0.0)
    incl = np.zeros(abc.shape[1], dtype=bool)
    for k in dd:
        incl |= np.abs(dd[k]) > 0.5
    sum_d = np.abs(dd["A"]) + np.abs(dd["B"]) + np.abs(dd["C"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(sum_d > 0, np.abs(dd["ABC"]) / sum_d, np.inf)
    eA, eB, eC = (np.abs(Dc[k].mean(0)) for k in ("A", "B", "C"))
    eABC = np.abs(abc.mean(0))
    dep_tg = eABC > 0.01
    return np.array([incl.sum(), (incl & (np.abs(dd["ABC"]) > sum_d)).sum(), (incl & (rd > 1.1)).sum(),
                     (incl & (rd >= 0.9) & (rd <= 1.1)).sum(), (incl & (rd < 0.9)).sum(),
                     dep_tg.sum(), (dep_tg & (eABC > 1.1 * (eA + eB + eC))).sum()], dtype=np.int64)


CLASS_NAMES = ["spec_included(|d|>0.5 any condition)", "spec_superadditive_strict(|d_ABC|>sum|d|)",
               "spec_superadditive_band(>1.1x)", "spec_additive(0.9-1.1x)", "spec_subadditive(<0.9x)",
               "deployed_targets(|e_ABC|>0.01)", "deployed_superadditive(>1.1x)"]


# =============================================================================
def stage_classnull(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = build_design(sel)
    D, cidx = load_D(nC, conds, a_groups)
    log(f"loaded D {D.shape}")
    real = [t for t in triplets if t["kind"] == "real"]
    rng = np.random.default_rng(SEED + 10)
    S = rng.choice(np.array([-1.0, 1.0]), size=(N_FLIP_CLASS, nC))
    obs = np.zeros(7, dtype=np.int64)
    null = np.zeros((N_FLIP_CLASS, 7), dtype=np.int64)
    trip_any_obs = np.zeros(2, dtype=np.int64)                 # triplets with >=1 spec-strict / deployed super
    trip_any_null = np.zeros((N_FLIP_CLASS, 2), dtype=np.int64)
    for ti, t in enumerate(real):
        a, b, c = t["a"], t["b"], t["c"]
        J = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
             "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
        Dc = {k: D[:, cidx[v]].astype(np.float64) for k, v in J.items()}
        I = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
        base = Dc["ABC"] - I
        lower = {k: v for k, v in Dc.items() if k != "ABC"}
        o = classify(lower, Dc["ABC"])
        obs += o
        trip_any_obs += np.array([o[1] > 0, o[6] > 0])
        for f in range(N_FLIP_CLASS):
            n_ = classify(lower, base + S[f][:, None] * I)
            null[f] += n_
            trip_any_null[f] += np.array([n_[1] > 0, n_[6] > 0])
        if ti % 64 == 0:
            log(f"classnull triplet {ti}/{len(real)}")
    cls = {}
    for i, name in enumerate(CLASS_NAMES):
        cls[name] = {"observed": int(obs[i]), "signflip_null_mean": float(null[:, i].mean()),
                     "signflip_null_2.5_97.5": [float(np.percentile(null[:, i], 2.5)),
                                                float(np.percentile(null[:, i], 97.5))],
                     "p_one_sided_ge": float((1 + (null[:, i] >= obs[i]).sum()) / (N_FLIP_CLASS + 1))}
    cls["triplets_with_any_spec_strict"] = {"observed": int(trip_any_obs[0]),
                                            "signflip_null_mean": float(trip_any_null[:, 0].mean())}
    cls["triplets_with_any_deployed_super"] = {"observed": int(trip_any_obs[1]),
                                               "signflip_null_mean": float(trip_any_null[:, 1].mean()),
                                               "signflip_null_2.5_97.5": [float(np.percentile(trip_any_null[:, 1], 2.5)),
                                                                          float(np.percentile(trip_any_null[:, 1], 97.5))]}
    log("classification null done")

    # ------------------------------------------------ per-target power (t-test + BH)
    prng = np.random.default_rng(SEED + 2)                      # same 64 triplets as v2_triplets power check
    real_idx = [i for i, t in enumerate(triplets) if t["kind"] == "real"]
    pick = sorted(prng.choice(len(real_idx), size=64, replace=False).tolist())
    power = {}
    for mode in ("constant", "proportional"):
        for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            P = []
            for k in pick:
                t = triplets[real_idx[k]]
                a, b, c = t["a"], t["b"], t["c"]
                J = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
                     "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
                Dc = {q: D[:, cidx[v]].astype(np.float64) for q, v in J.items()}
                I = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
                if mode == "constant":
                    I = I + kappa * Dc["ABC"].mean(0)[None, :]
                else:
                    I = I + kappa * Dc["ABC"]
                P.append(ttest_p(I))
            P = np.stack(P)
            qw = np.stack([bh(p) for p in P])
            qa = bh(P).reshape(P.shape)
            power[f"{mode}_kappa{kappa}"] = {
                "triplets_any_target_BH_within": int((qw < 0.05).any(1).sum()),
                "triplets_any_target_BH_across_64": int((qa < 0.05).any(1).sum()),
                "targets_BH_across_64_total": int((qa < 0.05).sum()),
                "n_triplets": len(pick)}
    log(f"per-target power: {json.dumps(power)}")
    res = {"classification_signflip_null": cls, "n_signflip_draws": N_FLIP_CLASS,
           "per_target_power": power,
           "per_target_power_note": ("constant: kappa x (cell-mean joint effect) added to every cell (no added "
                                     "cell-to-cell spread); proportional: kappa x (that cell's own joint effect "
                                     "d_ABC,c) added to each cell. t-test over 20 cells, BH at q<0.05."),
           "wall_seconds": round(time.time() - t0, 1)}
    H.write_json(OUT / "classnull.json", res)
    update_run_config("inputs", {"source_dir": str(SRC), "source_selection": sel["features"],
                                 "source_script_sha256": load_json(SRC / "run_config.json")["script_sha256"]})
    update_run_config("seeds", {"classnull_signflip_seed": SEED + 10, "power_pick_seed": SEED + 2,
                                "readout_pick_seed": SEED + 20})
    update_run_config("chunks", {"stage": "classnull", "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": res["wall_seconds"]})
    log(json.dumps(cls, indent=1))


# =============================================================================
READ_CELLS = 4
READ_TRIPLETS = 4


@torch.no_grad()
def stage_readout(args):
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rdir = OUT / "readout"
    rdir.mkdir(exist_ok=True)
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    toks = np.load(SRC / "cells_tokens.npz")
    triplets, conds, a_groups = build_design(sel)
    real = [t for t in triplets if t["kind"] == "real"]
    rng = np.random.default_rng(SEED + 20)
    picks = sorted(rng.choice(len(real), size=READ_TRIPLETS, replace=False).tolist())
    todo = [ci for ci in range(READ_CELLS) if not (rdir / f"cell{ci:02d}.json").exists()]
    if not todo:
        log("DONE (readout complete)")
        return
    free_gb = H.check_memory(MIN_FREE_GB)
    torch.manual_seed(0)
    xt = H.load_model(DEVICE)
    model = xt.model
    saes = H.load_saes([0, 5, 9, TGT], device=DEVICE)
    s11 = saes[TGT]
    Wlm = model.lm_head.weight                                       # (V, d)
    assert model.lm_head.bias is None
    budget = args.max_minutes * 60
    durations, done = [], []
    for ci in todo:
        est = max(durations) if durations else 120.0
        if time.time() - t_start + est > budget:
            log("time budget reached; stopping cleanly")
            break
        try:
            H.check_memory(MIN_FREE_GB)
        except H.MemoryGuardError as e:
            log(f"STOP (memory guard): {e}")
            break
        tc = time.time()
        ids = torch.from_numpy(toks[f"cell_{ci:02d}"][None, :]).to(DEVICE)

        def run(edits):
            with H.ResidualEditor(model, saes, edits=edits, capture=[TGT], capture_device=DEVICE) as ed:
                o = model(ids, use_cache=False, return_dict=True)   # all positions reach lm_head
                del o
            return ed.captured[TGT]["pre"][0]                       # (T, d) live lm_head input

        h0 = run([])
        u0 = s11.sae.W_enc(h0 - s11.mu)
        z0 = s11.encode(h0)
        act0 = (z0 > 0).cpu()
        T = h0.shape[0]
        out = {"cell": ci, "row": meta["rows"][ci], "triplets": []}
        g_cache = {}
        for pk in picks:
            t = real[pk]
            a, b, c = t["a"], t["b"], t["c"]
            names = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
                     "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
            dh, du, dz, act = {}, {}, {}, {}
            for k, (x, y, w) in names.items():
                edits = []
                if x != NONE:
                    edits.append(H.Ablate(0, [x]))
                if y != NONE:
                    edits.append(H.Ablate(5, [y]))
                if w != NONE:
                    edits.append(H.Ablate(9, [w]))
                h = run(edits)
                dh[k] = (h - h0).cpu().double()
                u = s11.sae.W_enc(h - s11.mu)
                du[k] = (u - u0).cpu().double()
                z = s11.encode(h)
                dz[k] = (z - z0).cpu().double()
                act[k] = (z > 0).cpu()
                del h, u, z
            sgn = {"ABC": 1, "AB": -1, "AC": -1, "BC": -1, "A": 1, "B": 1, "C": 1}
            rec = {"triplet_index_in_real": pk, "a": a, "b": b, "c": c}
            for nm, dd in (("h11", dh), ("sae_preact", du), ("sae_code", dz)):
                I = sum(sgn[k] * dd[k] for k in sgn)
                tot = dd["ABC"] - dd["A"] - dd["B"] - dd["C"]
                rec[nm] = {
                    "per_position_energy_share_I": float((I ** 2).sum() / max(float((dd["ABC"] ** 2).sum()), 1e-300)),
                    "per_position_energy_share_all_orders": float((tot ** 2).sum() / max(float((dd["ABC"] ** 2).sum()), 1e-300)),
                    "posmean_energy_share_I": float((I.mean(0) ** 2).sum() / max(float((dd["ABC"].mean(0) ** 2).sum()), 1e-300)),
                    "posmean_energy_share_all_orders": float((tot.mean(0) ** 2).sum() / max(float((dd["ABC"].mean(0) ** 2).sum()), 1e-300)),
                }
                if nm == "h11":
                    # logits are lm_head(h11): linear, so their interaction is Wlm @ I
                    Il = (I.float().to(DEVICE) @ Wlm.T).cpu().double()
                    Al = (dd["ABC"].float().to(DEVICE) @ Wlm.T).cpu().double()
                    Il = Il - Il.mean(1, keepdim=True)
                    Al = Al - Al.mean(1, keepdim=True)
                    rec["logits_centered"] = {
                        "per_position_energy_share_I": float((Il ** 2).sum() / max(float((Al ** 2).sum()), 1e-300)),
                        "posmean_energy_share_I": float((Il.mean(0) ** 2).sum() / max(float((Al.mean(0) ** 2).sum()), 1e-300))}
                    del Il, Al
                if nm == "sae_code":
                    stable = act["A"] & act["B"] & act["C"] & act["AB"] & act["AC"] & act["BC"] & act["ABC"] & act0
                    union = act["A"] | act["B"] | act["C"] | act["AB"] | act["AC"] | act["BC"] | act["ABC"] | act0
                    switching = union & ~stable
                    Ie = I ** 2
                    rec[nm]["share_of_I_energy_on_switching_entries"] = float(Ie[switching].sum() / max(float(Ie.sum()), 1e-300))
                    rec[nm]["share_of_dABC_energy_on_switching_entries"] = float((dd["ABC"] ** 2)[switching].sum()
                                                                                 / max(float((dd["ABC"] ** 2).sum()), 1e-300))
                    rec[nm]["positions_with_topk_set_change_vs_clean_ABC"] = float(((act["ABC"] != act0).any(1)).float().mean())
                    rec[nm]["mean_features_switched_per_position_ABC"] = float((act["ABC"] != act0).sum(1).float().mean() / 2)
                    # cross-check with the stored position-mean code change of the main run
                    gi = a_groups.index(a)
                    if gi not in g_cache:
                        g = np.load(SRC / "cells" / f"cell{ci:02d}_g{gi:02d}.npz")
                        g_cache[gi] = {tuple(int(v) for v in k): g["dz"][r] for r, k in enumerate(g["conds"])}
                    stored = g_cache[gi][(a, b, c)]
                    rec[nm]["max_abs_diff_vs_stored_posmean_dz_ABC"] = float(np.abs(dd["ABC"].mean(0).cpu().numpy() - stored).max())
                    rec[nm]["max_abs_stored_posmean_dz_ABC"] = float(np.abs(stored).max())
                del I, tot
            rec["rel_size_dh11_ABC_over_h11"] = float(dh["ABC"].norm() / h0.cpu().double().norm())
            out["triplets"].append(rec)
            del dh, du, dz, act
            H.free_device_cache()
        H.write_json(rdir / f"cell{ci:02d}.json", out)
        del h0, u0, z0, act0
        H.free_device_cache()
        durations.append(time.time() - tc)
        done.append([ci, round(durations[-1], 1)])
        log(f"readout cell {ci} done in {durations[-1]:.0f}s")
    left = sum(1 for ci in range(READ_CELLS) if not (rdir / f"cell{ci:02d}.json").exists())
    update_run_config("cells", {"dataset_path": meta["dataset_path"], "rows": meta["rows"][:READ_CELLS],
                                "barcodes": (meta["barcodes"] or [None] * READ_CELLS)[:READ_CELLS],
                                "note": "first 4 of the 20 v2_triplets cells"})
    update_run_config("env", H.env_info(DEVICE))
    update_run_config("chunks", {"stage": "readout", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                                 "wall_seconds": round(time.time() - t_start, 1), "free_gb_at_start": free_gb,
                                 "cells_done": done, "cells_left": left,
                                 "triplets_read": [[real[p]["a"], real[p]["b"], real[p]["c"]] for p in picks]})
    if left == 0:
        summarize_readout(rdir)
    log(("DONE" if left == 0 else "PARTIAL") + f": {left} cells left")


def summarize_readout(rdir):
    recs = []
    for f in sorted(rdir.glob("cell*.json")):
        d = load_json(f)
        for r in d["triplets"]:
            recs.append(dict(r, cell=d["cell"]))
    s = {"n_cell_triplet_pairs": len(recs)}
    for nm in ("h11", "sae_preact", "sae_code", "logits_centered"):
        s[nm] = {}
        for key in recs[0][nm]:
            v = np.array([r[nm][key] for r in recs], dtype=float)
            s[nm][key] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    s["rel_size_dh11_ABC_over_h11"] = {"median": float(np.median([r["rel_size_dh11_ABC_over_h11"] for r in recs]))}
    H.write_json(OUT / "readout_summary.json", s)
    log(json.dumps(s, indent=1))


# =============================================================================
def stage_align(args):
    """(1) Is the per-cell interaction aligned with the additive part (saturation) or not?
    (2) How many targets have a detectable (BH) single or joint effect at all?
    (3) Power of the aggregate sign-flip test for a planted interaction proportional to each cell's
        own joint effect (the main run planted a constant one)."""
    t0 = time.time()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = build_design(sel)
    D, cidx = load_D(nC, conds, a_groups)
    real = [t for t in triplets if t["kind"] == "real"]
    cos_cell, cos_mean, n_sig = [], [], {k: [] for k in ("A", "B", "C", "ABC", "I")}
    for ti, t in enumerate(real):
        a, b, c = t["a"], t["b"], t["c"]
        J = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
             "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
        Dc = {k: D[:, cidx[v]].astype(np.float64) for k, v in J.items()}
        I = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
        base = Dc["ABC"] - I
        cos_cell.append(float((I * base).sum() / max(np.sqrt((I ** 2).sum() * (base ** 2).sum()), 1e-300)))
        mi, mb = I.mean(0), base.mean(0)
        cos_mean.append(float((mi * mb).sum() / max(np.sqrt((mi ** 2).sum() * (mb ** 2).sum()), 1e-300)))
        for k in ("A", "B", "C", "ABC"):
            n_sig[k].append(int((bh(ttest_p(Dc[k])) < 0.05).sum()))
        n_sig["I"].append(int((bh(ttest_p(I)) < 0.05).sum()))
        if ti % 128 == 0:
            log(f"align triplet {ti}/{len(real)}")

    def sm(x):
        x = np.asarray(x, dtype=float)
        return {"median": float(np.median(x)), "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
                "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
                "n_negative": int((x < 0).sum()), "n": int(x.size)}
    res = {"cos_percell_I_vs_additive_part(cells x targets)": sm(cos_cell),
           "cos_cellmean_I_vs_additive_part": sm(cos_mean),
           "targets_with_BH_significant_effect_per_triplet(t-test, q<0.05 within triplet)": {k: sm(v) for k, v in n_sig.items()},
           "note": ("additive part = AB + AC + BC - A - B - C (= ABC - I). A negative cosine means that within a cell the "
                    "three-way term tends to shrink the joint effect (saturation / redundancy-like); its size is "
                    "given by the energy shares in v2_triplets/summary.json.")}
    # (3) aggregate sign-flip power, proportional plant
    Sf = np.load(SRC / "signflip_signs.npy")                    # (999, nC) same signs as the main run
    prng = np.random.default_rng(SEED + 2)
    real_idx = list(range(len(real)))
    pick = sorted(prng.choice(len(real_idx), size=64, replace=False).tolist())
    powp = {}
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        ps = []
        for k in pick:
            t = real[k]
            a, b, c = t["a"], t["b"], t["c"]
            J = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
                 "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
            Dc = {q: D[:, cidx[v]].astype(np.float64) for q, v in J.items()}
            I = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"] + kappa * Dc["ABC"]
            To = np.abs(I.mean(0)).sum()
            Tn = np.abs((Sf / nC) @ I).sum(1)
            ps.append(float((1 + (Tn >= To).sum()) / (len(Tn) + 1)))
        powp[str(kappa)] = {"detected_p05": int(sum(p < 0.05 for p in ps)),
                            "detected_BH05_within_64": int(sum(q < 0.05 for q in bh(ps))), "n_triplets": len(pick)}
    res["aggregate_signflip_power_proportional_plant"] = powp
    res["wall_seconds"] = round(time.time() - t0, 1)
    H.write_json(OUT / "align.json", res)
    update_run_config("chunks", {"stage": "align", "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": res["wall_seconds"]})
    log(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["classnull", "readout", "align"])
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    {"classnull": stage_classnull, "readout": stage_readout, "align": stage_align}[args.stage](args)


if __name__ == "__main__":
    main()
