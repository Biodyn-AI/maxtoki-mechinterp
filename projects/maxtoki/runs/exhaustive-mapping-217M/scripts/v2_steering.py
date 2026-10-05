"""v2 trajectory steering (revision item D7): fixed hooks + a random-feature null.

What the deployed Experiment 3 did (scripts/experiment3_rerun.py, outputs/experiment3/):
  * 200 Tabula Sapiens immune cells (seed 42 + 3333), "pseudotime" = PC1 of
    log1p(CP10k) expression, early = bottom quartile (50 cells), late = top quartile.
  * g_early / g_late = mean clean logits (averaged over positions) of early / late cells.
  * For the top 3 "switch" features per layer (L0, L3, L6, L9, L11) and alpha in {2, 5}:
    z_f -> alpha * z_f in the 50 early cells, and
    delta_s = cos(z', g_late) - cos(z', g_early) - [same for the clean logits],
    z' = steered logits averaged over positions.
  * Its hook deleted block li (L0-L9) or applied the final norm twice (L11);
    see V2_HOOKS_REPORT.md. This script redoes it with setup/hooks_v2.py.

This script:
  * same cells, same pseudotime, same signatures (recomputed and checked),
    same metric, same features;
  * edit = hooks_v2.Steer(layer, f, alpha, mode="scale"):
    delta = (alpha - 1) * z_f(x) * W_dec[:, f], added at the SAE site of the layer.
    alpha is a MULTIPLIER on the feature code, as in the deployed run. So
    alpha = 1 is the zero edit (adds nothing) and alpha = 0 removes the feature.
    Original features: alpha in {0, 1, 2, 5}. Null features: alpha in {2, 5}.
  * null: 20 random features per layer that are active in the early cells,
    matched on activation frequency (fraction of token positions with z_f > 0
    over the 50 early cells) to the 3 original features.

Stages (run with --stage auto; it continues where it stopped):
  prep      sample cells, pseudotime, tokenize (no model)
  clean     one clean pass per cell (200): mean logits + SAE code stats at 5 layers
  select    signatures, checks, feature stats, null draw (no model)
  steer     one work unit = one (layer, feature): 50 early cells x alphas
  summarize statistics, checks, tables (no model)
Usage:
  .venv/bin/python runs/exhaustive-mapping-217M/scripts/v2_steering.py --max-minutes 7.5
  (repeat until it prints ALL DONE)
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402
from dataset_loader import SYM2ENS_PKL  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
OUT3 = RUN / "outputs/experiment3"                  # deployed outputs (read only)
ZE = RUN / "outputs/v2_zero_edit_control"           # D7a outputs (read only)
OUT = RUN / "outputs/v2_steering"
CLEAN_DIR = OUT / "clean"
UNIT_DIR = OUT / "units"
TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"

SEED = 42                 # deployed PCA seed
TS_SEED = SEED + 3333     # deployed cell-sample seed
N_CELLS = 200
MAX_LEN = 2048
LAYERS = [0, 3, 6, 9, 11]
D_SAE = 4928
N_NULL = 20
NULL_SEED = 20261001      # null draw: default_rng(NULL_SEED + layer)
NULL_K_NEAREST = 50       # candidate pool per original = 50 nearest in log10 frequency
BOOT_SEED = 20261001
N_BOOT = 10000
ALPHAS_ORIG = [0.0, 1.0, 2.0, 5.0]
ALPHAS_NULL = [2.0, 5.0]
TOP_GENES = 10


# ----------------------------------------------------------------------------- helpers
def cosd(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def delta_s(m_st, m_cl, g_late, g_early) -> float:
    """Deployed metric (experiment3_rerun.py:296-302), in float64."""
    return ((cosd(m_st, g_late) - cosd(m_st, g_early))
            - (cosd(m_cl, g_late) - cosd(m_cl, g_early)))


def load_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- stage: prep
def stage_prep():
    if (OUT / "cells.npz").exists():
        return
    import anndata as ad
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    sig = np.load(OUT3 / "state_signatures.npz")
    adata = ad.read_h5ad(str(TS_H5), backed="r")
    rng_ts = np.random.default_rng(TS_SEED)
    ts_idx = np.sort(rng_ts.choice(adata.n_obs, size=N_CELLS, replace=False))
    X = adata[ts_idx].X
    if sp.issparse(X) or hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    var_ens = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
    obs_names = adata.obs_names.to_numpy()[ts_idx].astype(str)
    cell_types = adata.obs["cell_type"].iloc[ts_idx].astype(str).to_numpy()
    adata.file.close()
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_log = np.log1p(X / rs * 1e4)
    pca1 = PCA(n_components=1, random_state=SEED).fit_transform(X_log).flatten()
    pt = (pca1 - pca1.min()) / (pca1.max() - pca1.min() + 1e-12)
    early = pt < np.percentile(pt, 25)
    late = pt > np.percentile(pt, 75)
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping(var_ens.tolist())
    toks = np.empty(N_CELLS, dtype=object)
    for i in range(N_CELLS):
        c = tok.tokenize_cell(X[i], vi, vt, vm, max_len=MAX_LEN)
        toks[i] = None if c is None else np.asarray(c.token_ids, dtype=np.int64)
    ze = np.load(ZE / "early_cells_tokens.npz", allow_pickle=True)
    e_idx = np.where(early)[0]
    checks = {
        "pseudotime_maxabs_vs_deployed_cache": float(np.abs(pt - sig["pseudotime"]).max()),
        "early_mask_equal_deployed": bool((early == sig["early_mask"]).all()),
        "late_mask_equal_deployed": bool((late == sig["late_mask"]).all()),
        "n_early": int(early.sum()), "n_late": int(late.sum()),
        "n_tokenized": int(sum(t is not None for t in toks)),
        "early_positions_equal_zero_edit_control": bool(np.array_equal(e_idx, ze["early_idx"])),
        "early_tokens_equal_zero_edit_control": bool(all(
            np.array_equal(toks[i], np.asarray(t)) for i, t in zip(e_idx, ze["token_ids"]))),
        "early_dataset_rows_equal_zero_edit_control": bool(np.array_equal(ts_idx[e_idx], ze["ts_rows"])),
    }
    log(f"prep checks: {checks}")
    ok = (checks["pseudotime_maxabs_vs_deployed_cache"] == 0.0 and checks["early_mask_equal_deployed"]
          and checks["late_mask_equal_deployed"] and checks["n_tokenized"] == N_CELLS
          and checks["early_tokens_equal_zero_edit_control"]
          and checks["early_dataset_rows_equal_zero_edit_control"])
    if not ok:
        raise SystemExit("PREP CHECK FAILED: cell selection differs from the deployed run")
    np.savez(OUT / "cells.npz", token_ids=toks, ts_rows=ts_idx, obs_names=obs_names,
             cell_types=cell_types, pseudotime=pt, early_mask=early, late_mask=late)
    H.write_json(OUT / "prep_checks.json", checks)


def load_cells():
    c = np.load(OUT / "cells.npz", allow_pickle=True)
    return {k: c[k] for k in c.files}


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, device):
        self.device = device
        xt = H.load_model(device)
        self.model = xt.model
        self.saes = H.load_saes(LAYERS, device=device)


# ----------------------------------------------------------------------------- stage: clean
def stage_clean(M, cells, t_start, max_min, chunk):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(N_CELLS) if not (CLEAN_DIR / f"cell_{i:03d}.npz").exists()]
    for n, i in enumerate(todo):
        if (time.time() - t_start) / 60 > max_min - 0.3:
            return False
        if n % 10 == 0:
            H.check_memory(3.0)
        ids = torch.from_numpy(cells["token_ids"][i][None, :]).to(M.device)
        with torch.no_grad():
            with H.ResidualEditor(M.model, M.saes, edits=[], capture=LAYERS) as ed:
                out = M.model(ids, use_cache=False, return_dict=True)
            mean_logits = out.logits[0].mean(0).float().cpu().numpy()
            del out
            rec = {"mean_logits": mean_logits.astype(np.float32), "seq_len": ids.shape[1]}
            for l in LAYERS:
                z = ed.codes(l, which="pre")          # (T, d_sae) CPU
                rec[f"meanz_L{l}"] = z.mean(0).numpy().astype(np.float32)
                rec[f"nact_L{l}"] = (z > 0).sum(0).numpy().astype(np.int32)
                rec[f"sumz_L{l}"] = z.sum(0).numpy().astype(np.float32)
                del z
            ed.captured.clear()
        tmp = CLEAN_DIR / f"cell_{i:03d}.tmp.npz"
        np.savez(tmp, **rec)
        tmp.replace(CLEAN_DIR / f"cell_{i:03d}.npz")
        chunk.setdefault("clean_cells_done", []).append(i)
        if n % 20 == 0:
            H.free_device_cache()
            log(f"clean cell {i} ({ids.shape[1]} tok)  elapsed {(time.time()-t_start)/60:.1f} min")
    return True


# ----------------------------------------------------------------------------- stage: select
def stage_select(cells):
    if (OUT / "selection.json").exists():
        return
    from scipy.stats import rankdata
    sig = np.load(OUT3 / "state_signatures.npz")
    early = cells["early_mask"]; late = cells["late_mask"]; pt = cells["pseudotime"]
    clean = [np.load(CLEAN_DIR / f"cell_{i:03d}.npz") for i in range(N_CELLS)]
    ML = np.stack([c["mean_logits"] for c in clean]).astype(np.float64)
    g_early = ML[early].mean(0); g_late = ML[late].mean(0)
    checks = {
        "clean_meanlogits_maxabs_vs_deployed_cache": float(np.abs(ML - sig["clean_logits_per_cell"]).max()),
        "g_early_maxabs_vs_deployed": float(np.abs(g_early - sig["g_early_logits"]).max()),
        "g_late_maxabs_vs_deployed": float(np.abs(g_late - sig["g_late_logits"]).max()),
        "cos_g_early_g_late_recomputed": cosd(g_early, g_late),
        "cos_g_early_g_late_deployed_cache": cosd(sig["g_early_logits"], sig["g_late_logits"]),
        "claimed_in_stage3_summary": 0.999,
    }
    np.savez(OUT / "signatures.npz", g_early=g_early, g_late=g_late, clean_mean_logits=ML)

    sw = pd.read_csv(OUT3 / "switch_features.csv")
    e_idx = np.where(early)[0]
    rank_pt = rankdata(pt)
    sel = {"checks": checks, "layers": {}}
    for l in LAYERS:
        nact = np.stack([clean[i][f"nact_L{l}"] for i in e_idx]).astype(np.int64)    # (50, F)
        sumz = np.stack([clean[i][f"sumz_L{l}"] for i in e_idx]).astype(np.float64)
        T = np.array([int(clean[i]["seq_len"]) for i in e_idx])
        freq = nact.sum(0) / T.sum()
        n_cells_active = (nact > 0).sum(0)
        mean_z_active = np.where(nact.sum(0) > 0, sumz.sum(0) / np.maximum(nact.sum(0), 1), 0.0)
        # Spearman rho of per-cell mean code with pseudotime, all 200 cells (as deployed)
        MZ = np.stack([clean[i][f"meanz_L{l}"] for i in range(N_CELLS)]).astype(np.float64)
        R = np.apply_along_axis(rankdata, 0, MZ)
        Rc = R - R.mean(0); rp = rank_pt - rank_pt.mean()
        den = np.sqrt((Rc ** 2).sum(0) * (rp ** 2).sum())
        rho = np.where(den > 0, (Rc * rp[:, None]).sum(0) / np.where(den > 0, den, 1), np.nan)
        orig = sw[sw["layer"] == l].head(3)
        origs = []
        for _, r in orig.iterrows():
            f = int(r["feature"])
            origs.append({"feature": f, "switch_score": float(r["switch_score"]), "direction": r["direction"],
                          "rho_deployed": float(r["rho"]), "rho_recomputed": float(rho[f]),
                          "freq_early": float(freq[f]), "n_early_cells_active": int(n_cells_active[f]),
                          "mean_z_when_active": float(mean_z_active[f])})
        o_feats = [o["feature"] for o in origs]
        sw_l = sw[sw["layer"] == l].set_index("feature")
        pool = np.array([f for f in range(D_SAE) if freq[f] > 0 and f not in o_feats])
        rng = np.random.default_rng(NULL_SEED + l)
        k_per = [7, 7, 6]
        chosen, nulls = set(), []
        for o, k in zip(origs, k_per):
            fo = max(o["freq_early"], 1.0 / T.sum())
            cand = np.array([f for f in pool if f not in chosen])
            dist = np.abs(np.log10(freq[cand]) - np.log10(fo))
            near = cand[np.argsort(dist, kind="stable")[:NULL_K_NEAREST]]
            pick = rng.choice(near, size=k, replace=False)
            for f in pick:
                f = int(f); chosen.add(f)
                nulls.append({"feature": f, "matched_to": o["feature"], "freq_early": float(freq[f]),
                              "freq_ratio_to_matched": float(freq[f] / fo),
                              "n_early_cells_active": int(n_cells_active[f]),
                              "mean_z_when_active": float(mean_z_active[f]),
                              "rho_recomputed": float(rho[f]),
                              "in_deployed_switch_list": bool(f in sw_l.index)})
        sel["layers"][str(l)] = {
            "originals": origs, "nulls": nulls,
            "n_alive_in_early_cells": int((freq > 0).sum()),
            "rho_all_features": [None if not np.isfinite(x) else float(x) for x in rho],
            "freq_all_features": freq.tolist(),
            "mean_z_active_all_features": mean_z_active.tolist(),
        }
        log(f"L{l}: alive {int((freq > 0).sum())}; originals "
            + ", ".join(f"F{o['feature']} freq {o['freq_early']:.2e} rho {o['rho_recomputed']:+.3f}/{o['rho_deployed']:+.3f}" for o in origs)
            + f"; null freq range {min(n['freq_early'] for n in nulls):.2e}-{max(n['freq_early'] for n in nulls):.2e}")
    log(f"select checks: {checks}")
    H.write_json(OUT / "selection.json", sel)


def units_list(sel):
    units = []
    for l in LAYERS:
        for o in sel["layers"][str(l)]["originals"]:
            units.append((l, o["feature"], "original"))
    for j in range(N_NULL):
        for l in LAYERS:
            units.append((l, sel["layers"][str(l)]["nulls"][j]["feature"], "null"))
    return units


# ----------------------------------------------------------------------------- stage: steer
def run_unit(M, cells, sig, l, f, kind):
    e_idx = np.where(cells["early_mask"])[0]
    alphas = ALPHAS_ORIG if kind == "original" else ALPHAS_NULL
    gE, gL, ML = sig["g_early"], sig["g_late"], sig["clean_mean_logits"]
    nA, nC = len(alphas), len(e_idx)
    res = {k: np.zeros((nC, nA)) for k in ("delta_s", "mean_dlogit", "meanabs_dvec", "maxabs_dvec", "delta_norm")}
    res["n_active"] = np.zeros(nC, dtype=np.int64)
    vec_sum = np.zeros((nA, ML.shape[1]))
    per_cell_vec_a5 = np.zeros((nC, ML.shape[1]), dtype=np.float16) if kind == "original" else None
    for ci_n, ci in enumerate(e_idx):
        if ci_n % 10 == 0:
            H.check_memory(3.0)
        ids = torch.from_numpy(cells["token_ids"][ci][None, :]).to(M.device)
        m_cl = ML[ci]
        for a_n, a in enumerate(alphas):
            with torch.no_grad():
                with H.ResidualEditor(M.model, M.saes, edits=[H.Steer(l, feature=f, alpha=a)]) as ed:
                    out = M.model(ids, use_cache=False, return_dict=True)
                m_st = out.logits[0].mean(0).float().cpu().numpy().astype(np.float64)
                del out
            dv = m_st - m_cl
            res["delta_s"][ci_n, a_n] = delta_s(m_st, m_cl, gL, gE)
            res["mean_dlogit"][ci_n, a_n] = dv.mean()
            res["meanabs_dvec"][ci_n, a_n] = np.abs(dv).mean()
            res["maxabs_dvec"][ci_n, a_n] = np.abs(dv).max()
            res["delta_norm"][ci_n, a_n] = ed.edit_log[l]["delta_norm"]
            res["n_active"][ci_n] = ed.edit_log[l]["n_active"][int(f)]
            vec_sum[a_n] += dv
            if per_cell_vec_a5 is not None and a == 5.0:
                per_cell_vec_a5[ci_n] = dv.astype(np.float16)
        if ci_n % 10 == 9:
            H.free_device_cache()
    out = {k: v for k, v in res.items()}
    out.update({"alphas": np.array(alphas), "cells": e_idx, "mean_dvec": (vec_sum / nC).astype(np.float32)})
    if per_cell_vec_a5 is not None:
        out["per_cell_dvec_a5"] = per_cell_vec_a5
    return out


def stage_steer(M, cells, sel, t_start, max_min, chunk):
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    sig = dict(np.load(OUT / "signatures.npz"))
    est = {"original": 1.6, "null": 0.8}
    for (l, f, kind) in units_list(sel):
        path = UNIT_DIR / f"L{l:02d}_F{f:04d}.npz"
        if path.exists():
            continue
        if (time.time() - t_start) / 60 + est[kind] * 1.15 > max_min:
            return False
        t0 = time.time()
        r = run_unit(M, cells, sig, l, f, kind)
        tmp = UNIT_DIR / f"L{l:02d}_F{f:04d}.tmp.npz"
        np.savez(tmp, kind=kind, layer=l, feature=f, **r)
        tmp.replace(path)
        dt = time.time() - t0
        est[kind] = dt / 60
        chunk.setdefault("units_done", []).append({"layer": l, "feature": f, "kind": kind, "wall_s": round(dt, 1)})
        a = list(r["alphas"]); j5 = a.index(5.0); j2 = a.index(2.0)
        log(f"L{l:>2} F{f:>4} {kind:8s} ds(a2) {r['delta_s'][:, j2].mean():+.6f} ds(a5) {r['delta_s'][:, j5].mean():+.6f}"
            + (f" ds(a1) {np.abs(r['delta_s'][:, a.index(1.0)]).max():.1e} ds(a0) {r['delta_s'][:, a.index(0.0)].mean():+.6f}" if kind == "original" else "")
            + f"  {dt:.0f}s")
        H.free_device_cache()
    return True


# ----------------------------------------------------------------------------- stage: summarize
def boot_idx(n):
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def ci(x, idx):
    m = np.asarray(x, dtype=np.float64)[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def gene_namer():
    tok = MaxTokiTokenizer()
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    ens2sym = {v: k for k, v in sym2ens.items()}
    tok2ens = {v: k for k, v in tok.gene_token_dict.items()}

    def gene(t):
        e = tok2ens.get(int(t))
        return f"<tok{t}>" if e is None else ens2sym.get(e, e)
    return gene


def stage_summarize(cells, sel):
    gene = gene_namer()
    dep = json.loads((OUT3 / "steering_summary.json").read_text())
    dep_feat = {(r["layer"], r["feature"]): r for r in dep["features"]}
    sig = np.load(OUT / "signatures.npz")
    U = {}
    for (l, f, kind) in units_list(sel):
        z = np.load(UNIT_DIR / f"L{l:02d}_F{f:04d}.npz")
        U[(l, f)] = {k: z[k] for k in z.files}
    n_cells = int(cells["early_mask"].sum())
    idx = boot_idx(n_cells)
    checks = {}

    # ---- check A: zero edit (alpha = 1) gives exactly 0 for every original feature and cell
    zero_max = max(float(np.abs(U[(l, o["feature"])]["delta_s"][:, 1]).max())
                   for l in LAYERS for o in sel["layers"][str(l)]["originals"])
    zero_vec_max = max(float(U[(l, o["feature"])]["maxabs_dvec"][:, 1].max())
                       for l in LAYERS for o in sel["layers"][str(l)]["originals"])
    zero_norm_max = max(float(U[(l, o["feature"])]["delta_norm"][:, 1].max())
                        for l in LAYERS for o in sel["layers"][str(l)]["originals"])
    checks["A_zero_edit"] = {"max_abs_delta_s": zero_max, "max_abs_mean_logit_change": zero_vec_max,
                             "max_edit_norm": zero_norm_max, "pass": zero_max <= 1e-9 and zero_vec_max <= 1e-6}

    # ---- check B: cells where the feature never fires must give exactly 0 (scale mode)
    worst = 0.0; n_silent = 0
    for (l, f), u in U.items():
        s = u["n_active"] == 0
        n_silent += int(s.sum())
        if s.any():
            worst = max(worst, float(np.abs(u["delta_s"][s]).max()))
    checks["B_silent_cells_zero"] = {"n_cell_feature_pairs_silent": n_silent, "max_abs_delta_s": worst,
                                     "pass": worst <= 1e-9}

    # ---- check C: reproduce the D7a preview (first feature per layer, alpha 5, hooks_v2)
    rep = {}
    for l in LAYERS:
        f0 = sel["layers"][str(l)]["originals"][0]["feature"]
        u = U[(l, f0)]
        mine = u["delta_s"][:, list(u["alphas"]).index(5.0)]
        prev = np.array([json.loads((ZE / "cells" / f"cell_{ci:03d}.json").read_text())[f"L{l}_new_a5_delta_s"]
                         for ci in u["cells"]])
        rep[f"L{l}"] = {"feature": f0, "max_abs_diff_per_cell": float(np.abs(mine - prev).max()),
                        "mean_here": float(mine.mean()), "mean_D7a_preview": float(prev.mean())}
    checks["C_reproduce_D7a_preview"] = {"layers": rep,
                                         "pass": all(v["max_abs_diff_per_cell"] < 1e-5 for v in rep.values())}
    # ---- check D: recomputed rho of originals equals the deployed switch rho
    rd = [abs(o["rho_recomputed"] - o["rho_deployed"]) for l in LAYERS for o in sel["layers"][str(l)]["originals"]]
    checks["D_switch_rho_reproduced"] = {"max_abs_diff": float(max(rd)), "pass": max(rd) < 0.01}
    checks["E_signatures"] = sel["checks"]
    checks["E_signatures"]["pass"] = (sel["checks"]["clean_meanlogits_maxabs_vs_deployed_cache"] < 1e-3
                                      and abs(sel["checks"]["cos_g_early_g_late_recomputed"]
                                              - sel["checks"]["cos_g_early_g_late_deployed_cache"]) < 1e-5)

    layers_out = {}
    table_rows, null_rows = [], []
    all_feat_rows = []
    for l in LAYERS:
        L = sel["layers"][str(l)]
        origs, nulls = L["originals"], L["nulls"]
        rho_all = L["rho_all_features"]
        lay = {"originals": [], "null": {}, "features_differ": {}, "layer_permutation_test": {}}
        null_stats = {}
        for a in (2.0, 5.0):
            vals = []
            fr = []
            for n in nulls:
                u = U[(l, n["feature"])]
                j = list(u["alphas"]).index(a)
                vals.append(float(u["delta_s"][:, j].mean()))
                fr.append(float((u["delta_s"][:, j] > 0).mean()))
            vals = np.array(vals)
            null_stats[a] = vals
            lay["null"][f"alpha{a:g}"] = {
                "n_features": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "p2.5": float(np.percentile(vals, 2.5)), "p97.5": float(np.percentile(vals, 97.5)),
                "min": float(vals.min()), "max": float(vals.max()),
                "mean_abs": float(np.abs(vals).mean()),
                "frac_toward_maturity_mean": float(np.mean(fr)),
                "frac_toward_maturity_range": [float(min(fr)), float(max(fr))],
                "values": vals.tolist(),
            }
        # null alpha scaling
        r52 = []
        for n in nulls:
            u = U[(l, n["feature"])]; a = list(u["alphas"])
            m2 = u["delta_s"][:, a.index(2.0)].mean(); m5 = u["delta_s"][:, a.index(5.0)].mean()
            r52.append(m5 / m2 if abs(m2) > 0 else np.nan)
        lay["null"]["ratio_a5_over_a2_median"] = float(np.nanmedian(r52))
        lay["null"]["ratio_a5_over_a2_range"] = [float(np.nanmin(r52)), float(np.nanmax(r52))]
        lay["null"]["edit_norm_a5_mean"] = float(np.mean([U[(l, n["feature"])]["delta_norm"][:, -1].mean() for n in nulls]))

        o_vecs = {}
        for o in origs:
            f = o["feature"]; u = U[(l, f)]; a = list(u["alphas"])
            row = {"feature": f, "direction": o["direction"], "rho_recomputed": o["rho_recomputed"],
                   "freq_early": o["freq_early"], "n_early_cells_active": o["n_early_cells_active"],
                   "mean_active_positions_per_cell": float(u["n_active"].mean())}
            sgn = 1.0 if o["direction"] == "positive" else -1.0
            for al in ALPHAS_ORIG:
                j = a.index(al); x = u["delta_s"][:, j]
                st = {"mean_delta_s": float(x.mean()), "ci95": ci(x, idx), "sd_cells": float(x.std(ddof=1)),
                      "frac_toward_maturity": float((x > 0).mean()),
                      "edit_norm_mean": float(u["delta_norm"][:, j].mean()),
                      "meanabs_mean_logit_change": float(u["meanabs_dvec"][:, j].mean())}
                if al in null_stats:
                    nv = null_stats[al]; m = x.mean()
                    st["z_vs_null"] = float((m - nv.mean()) / nv.std(ddof=1))
                    st["p_size_vs_null"] = float((1 + np.sum(np.abs(nv) >= abs(m))) / (1 + len(nv)))
                    # one-sided in the direction the switch feature predicts (amplify -> its own end)
                    st["p_directional_vs_null"] = float((1 + np.sum(sgn * nv >= sgn * m)) / (1 + len(nv)))
                    st["rank_among_null_by_size"] = int(np.sum(np.abs(nv) >= abs(m)))
                    st["sign_matches_switch_direction"] = bool(np.sign(m) == sgn)
                row[f"alpha{al:g}"] = st
            # alpha scaling
            x2 = u["delta_s"][:, a.index(2.0)]; x5 = u["delta_s"][:, a.index(5.0)]; x0 = u["delta_s"][:, a.index(0.0)]
            b2 = x2[idx].mean(1); b5 = x5[idx].mean(1); b0 = x0[idx].mean(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = b5 / b2; r02 = b0 / b2
            row["ratio_a5_over_a2"] = {"value": float(x5.mean() / x2.mean()),
                                      "ci95": [float(np.nanpercentile(rr, 2.5)), float(np.nanpercentile(rr, 97.5))],
                                      "expected_if_linear": 4.0}
            row["ratio_a0_over_a2"] = {"value": float(x0.mean() / x2.mean()),
                                      "ci95": [float(np.nanpercentile(r02, 2.5)), float(np.nanpercentile(r02, 97.5))],
                                      "expected_if_linear": -1.0}
            row["abs_a5_gt_abs_a2_cells_frac"] = float(np.mean(np.abs(x5) > np.abs(x2)))
            mv = u["mean_dvec"][a.index(5.0)]
            o_vecs[f] = mv
            row["top_up_genes_a5"] = [{"gene": gene(t), "token": int(t), "mean_dlogit": float(mv[t])}
                                      for t in np.argsort(-mv)[:TOP_GENES]]
            row["top_down_genes_a5"] = [{"gene": gene(t), "token": int(t), "mean_dlogit": float(mv[t])}
                                        for t in np.argsort(mv)[:5]]
            dp = dep_feat.get((l, f))
            if dp:
                row["deployed"] = {"delta_s_a5": dp["alpha_results"]["5"]["mean_delta_s"],
                                   "delta_s_a2": dp["alpha_results"]["2"]["mean_delta_s"],
                                   "frac_toward_maturity_a5": dp["alpha_results"]["5"]["frac_positive_maturity"],
                                   "top_up_genes_a5": [g["gene"] for g in dp["alpha_results"]["5"]["top_upregulated"][:5]]}
            lay["originals"].append(row)
            table_rows.append({"layer": l, "feature": f, "direction": o["direction"],
                               "rho": round(o["rho_recomputed"], 3), "freq_early": o["freq_early"],
                               **{f"ds_a{al:g}": row[f"alpha{al:g}"]["mean_delta_s"] for al in ALPHAS_ORIG},
                               "ds_a5_ci_lo": row["alpha5"]["ci95"][0], "ds_a5_ci_hi": row["alpha5"]["ci95"][1],
                               "frac_mat_a5": row["alpha5"]["frac_toward_maturity"],
                               "z_vs_null_a5": row["alpha5"]["z_vs_null"],
                               "p_size_a5": row["alpha5"]["p_size_vs_null"],
                               "p_dir_a5": row["alpha5"]["p_directional_vs_null"],
                               "ratio_a5_a2": row["ratio_a5_over_a2"]["value"],
                               "top_genes_a5": ";".join(g["gene"] for g in row["top_up_genes_a5"][:5]),
                               "deployed_ds_a5": row.get("deployed", {}).get("delta_s_a5")})
        # features differ: pairwise paired differences + cosine of mean logit-change vectors
        pairs = []
        for (o1, o2) in itertools.combinations(origs, 2):
            u1, u2 = U[(l, o1["feature"])], U[(l, o2["feature"])]
            d = u1["delta_s"][:, 3] - u2["delta_s"][:, 3]
            t1 = {g["gene"] for g in next(r for r in lay["originals"] if r["feature"] == o1["feature"])["top_up_genes_a5"]}
            t2 = {g["gene"] for g in next(r for r in lay["originals"] if r["feature"] == o2["feature"])["top_up_genes_a5"]}
            pc = [cosd(u1["per_cell_dvec_a5"][c].astype(np.float64), u2["per_cell_dvec_a5"][c].astype(np.float64))
                  for c in range(n_cells)]
            pairs.append({"pair": [o1["feature"], o2["feature"]], "diff_delta_s_a5": float(d.mean()),
                          "diff_ci95": ci(d, idx), "diff_ci_excludes_0": bool(ci(d, idx)[0] > 0 or ci(d, idx)[1] < 0),
                          "cos_mean_logit_change_vectors": cosd(o_vecs[o1["feature"]], o_vecs[o2["feature"]]),
                          "median_per_cell_cos": float(np.median(pc)),
                          "top10_gene_overlap": len(t1 & t2)})
        lay["features_differ"]["pairs"] = pairs
        all_feats = [o["feature"] for o in origs] + [n["feature"] for n in nulls]
        top1 = [gene(int(np.argmax(U[(l, f)]["mean_dvec"][list(U[(l, f)]["alphas"]).index(5.0)]))) for f in all_feats]
        lay["features_differ"]["n_distinct_top1_gene_among_23"] = len(set(top1))
        lay["features_differ"]["top1_gene_per_feature"] = dict(zip([str(f) for f in all_feats], top1))
        V = np.stack([U[(l, f)]["mean_dvec"][list(U[(l, f)]["alphas"]).index(5.0)] for f in all_feats]).astype(np.float64)
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        C = Vn @ Vn.T
        iu = np.triu_indices(len(all_feats), 1)
        lay["features_differ"]["pairwise_cos_all23_median"] = float(np.median(C[iu]))
        lay["features_differ"]["pairwise_cos_all23_range"] = [float(C[iu].min()), float(C[iu].max())]
        # cosine of each feature's change vector with the late-minus-early signature
        gdiff = sig["g_late"] - sig["g_early"]
        lay["features_differ"]["cos_with_late_minus_early_orig"] = [cosd(V[k], gdiff) for k in range(3)]
        lay["features_differ"]["cos_with_late_minus_early_null_median"] = float(np.median([cosd(V[k], gdiff) for k in range(3, len(all_feats))]))

        # layer-level permutation test: are the 3 originals exchangeable with the 20 nulls?
        ds5 = np.array([U[(l, f)]["delta_s"][:, list(U[(l, f)]["alphas"]).index(5.0)].mean() for f in all_feats])
        sizes = np.abs(ds5)
        rhos = np.array([rho_all[f] if rho_all[f] is not None else 0.0 for f in all_feats])
        dirstat = np.sign(rhos) * ds5      # push toward the end the feature is associated with
        obs_size = sizes[:3].mean(); obs_dir = dirstat[:3].mean()
        combs = list(itertools.combinations(range(len(all_feats)), 3))
        perm_size = np.array([sizes[list(c)].mean() for c in combs])
        perm_dir = np.array([dirstat[list(c)].mean() for c in combs])
        lay["layer_permutation_test"] = {
            "n_subsets": len(combs),
            "mean_abs_ds_a5_originals": float(obs_size),
            "p_size": float(np.mean(perm_size >= obs_size - 1e-15)),
            "mean_directional_ds_a5_originals": float(obs_dir),
            "p_directional": float(np.mean(perm_dir >= obs_dir - 1e-15)),
        }
        # rho vs delta_s across all 23 features (does amplifying a late-linked feature push to late?)
        from scipy.stats import spearmanr
        sr = spearmanr(rhos, ds5)
        lay["rho_vs_ds_a5_across_23"] = {"spearman": float(sr.statistic), "p": float(sr.pvalue)}
        layers_out[f"L{l}"] = lay
        for n in nulls:
            u = U[(l, n["feature"])]
            null_rows.append({"layer": l, "feature": n["feature"], "matched_to": n["matched_to"],
                              "freq_early": n["freq_early"], "rho": n["rho_recomputed"],
                              "in_switch_list": n["in_deployed_switch_list"],
                              "ds_a2": float(u["delta_s"][:, 0].mean()), "ds_a5": float(u["delta_s"][:, 1].mean()),
                              "frac_mat_a5": float((u["delta_s"][:, 1] > 0).mean()),
                              "edit_norm_a5": float(u["delta_norm"][:, 1].mean()),
                              "top1_gene_a5": gene(int(np.argmax(u["mean_dvec"][1])))})
        for k, f in enumerate(all_feats):
            all_feat_rows.append({"layer": l, "feature": f, "kind": "original" if k < 3 else "null",
                                  "rho": rhos[k], "ds_a5": ds5[k]})

    # pooled across layers: rho vs delta_s (115 features)
    from scipy.stats import spearmanr
    af = pd.DataFrame(all_feat_rows)
    sr = spearmanr(af["rho"], af["ds_a5"])
    pooled = {"spearman_rho_vs_ds_a5_all_115": float(sr.statistic), "p": float(sr.pvalue)}

    # check F: effect grows with alpha (originals): |ds(5)| > |ds(2)| for every original
    grow = [abs(r["alpha5"]["mean_delta_s"]) > abs(r["alpha2"]["mean_delta_s"])
            for L in layers_out.values() for r in L["originals"]]
    ratios = [r["ratio_a5_over_a2"]["value"] for L in layers_out.values() for r in L["originals"]]
    checks["F_effect_grows_with_alpha"] = {"n_originals_abs_a5_gt_abs_a2": int(sum(grow)), "n": len(grow),
                                           "ratio_a5_over_a2_range": [float(min(ratios)), float(max(ratios))],
                                           "pass": all(grow)}
    # check G: features differ (paired difference CI excludes 0 for >=1 pair per layer; top-1 genes not all equal)
    gd = {}
    for k, L in layers_out.items():
        gd[k] = {"pairs_with_ci_excluding_0": int(sum(p["diff_ci_excludes_0"] for p in L["features_differ"]["pairs"])),
                 "distinct_top1_among_3_originals": len({r["top_up_genes_a5"][0]["gene"] for r in L["originals"]}),
                 "max_cos_between_originals": float(max(p["cos_mean_logit_change_vectors"] for p in L["features_differ"]["pairs"]))}
    checks["G_features_differ"] = {"layers": gd, "pass": all(v["pairs_with_ci_excluding_0"] >= 1
                                                             and v["max_cos_between_originals"] < 0.99 for v in gd.values())}
    checks["all_pass"] = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))

    summary = {"what": "D7 trajectory steering with setup/hooks_v2.py and a frequency-matched random-feature null",
               "n_early_cells": n_cells, "layers": LAYERS, "alphas_original": ALPHAS_ORIG, "alphas_null": ALPHAS_NULL,
               "alpha_meaning": "multiplier on the feature code z_f (alpha=1 adds nothing; alpha=0 removes the feature)",
               "resampling": f"cell-level percentile bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}, n = {n_cells} cells; "
                             "null = 20 frequency-matched random features per layer",
               "checks": checks, "per_layer": layers_out, "pooled": pooled}
    H.write_json(OUT / "summary.json", summary)
    pd.DataFrame(table_rows).to_csv(OUT / "originals_table.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "null_features_table.csv", index=False)
    log(json.dumps(checks, indent=1, default=str))
    log(pd.DataFrame(table_rows).drop(columns=["top_genes_a5"]).to_string(index=False))
    for k, L in layers_out.items():
        log(f"{k} null a5: mean {L['null']['alpha5']['mean']:+.6f} sd {L['null']['alpha5']['sd']:.6f} "
            f"range [{L['null']['alpha5']['p2.5']:+.6f}, {L['null']['alpha5']['p97.5']:+.6f}]  "
            f"perm p_size {L['layer_permutation_test']['p_size']:.3f} p_dir {L['layer_permutation_test']['p_directional']:.3f}")
    return checks["all_pass"]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--stage", default="auto", choices=["auto", "summarize"])
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / "run_config.json"
    cfg = load_json(cfg_path, {"chunks": []})
    chunk = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_sha256": H.sha256_file(__file__),
             "free_gb_at_start": round(H.available_memory_gb(), 2)}

    def save_cfg(extra=None):
        cfg.update(H.env_info(args.device))
        cfg.update({
            "script": str(Path(__file__).resolve()), "script_sha256_latest": H.sha256_file(__file__),
            "deployed_script": str(RUN / "scripts/experiment3_rerun.py"),
            "deployed_script_sha256": H.sha256_file(RUN / "scripts/experiment3_rerun.py"),
            "seeds": {"ts_cell_sample": TS_SEED, "pca": SEED, "null_draw": f"{NULL_SEED} + layer",
                      "bootstrap": BOOT_SEED, "torch_numpy_global": SEED},
            "layers": LAYERS, "alphas_original": ALPHAS_ORIG, "alphas_null": ALPHAS_NULL,
            "edit": "hooks_v2.Steer(layer, f, alpha, mode='scale'): delta = (alpha-1) z_f(x) W_dec[:, f], all positions",
            "metric": "delta_s = cos(m', g_late) - cos(m', g_early) - same for clean; m = logits averaged over positions",
            "null": f"{N_NULL} features per layer, active in early cells, nearest {NULL_K_NEAREST} in log10 activation "
                    "frequency to each original (7/7/6), drawn without replacement",
        })
        if (OUT / "cells.npz").exists():
            c = load_cells()
            cfg["cells"] = {"dataset": str(TS_H5), "n_sampled": N_CELLS,
                            "dataset_rows_all_200": [int(x) for x in c["ts_rows"]],
                            "obs_names_all_200": [str(x) for x in c["obs_names"]],
                            "early_positions_in_sample": [int(x) for x in np.where(c["early_mask"])[0]],
                            "late_positions_in_sample": [int(x) for x in np.where(c["late_mask"])[0]],
                            "early_dataset_rows": [int(x) for x in c["ts_rows"][c["early_mask"]]]}
        if (OUT / "selection.json").exists():
            s = load_json(OUT / "selection.json")
            cfg["features"] = {str(l): {"originals": [o["feature"] for o in s["layers"][str(l)]["originals"]],
                                        "nulls": [n["feature"] for n in s["layers"][str(l)]["nulls"]]} for l in LAYERS}
        if extra:
            chunk.update(extra)
        H.write_json(cfg_path, cfg)

    done = False
    try:
        if args.stage == "summarize":
            cells = load_cells(); sel = load_json(OUT / "selection.json")
            done = stage_summarize(cells, sel)
        else:
            stage_prep()
            cells = load_cells()
            need_model = (len(list(CLEAN_DIR.glob("cell_[0-9][0-9][0-9].npz"))) < N_CELLS
                          or not (OUT / "selection.json").exists()
                          or len(list(UNIT_DIR.glob("L*_F*[0-9].npz"))) < 5 * (3 + N_NULL))
            M = None
            if need_model:
                H.check_memory(3.0)
                M = Model(args.device)
            finished = stage_clean(M, cells, t_start, args.max_minutes, chunk) if M else True
            if finished:
                stage_select(cells)
                sel = load_json(OUT / "selection.json")
                finished = stage_steer(M, cells, sel, t_start, args.max_minutes, chunk) if M else True
                if finished:
                    log("all units done; summarizing")
                    ok = stage_summarize(cells, sel)
                    log("ALL DONE" + ("" if ok else " -- BUT A CHECK FAILED"))
                    done = True
            if not done:
                log("time budget reached; run again to continue")
    except H.MemoryGuardError as e:
        log(f"memory guard: {e}")
        chunk["stopped"] = str(e)
    finally:
        chunk["wall_seconds"] = round(time.time() - t_start, 1)
        chunk["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        cfg["chunks"].append(chunk)
        save_cfg()


if __name__ == "__main__":
    main()
