"""v3 steering (item V3-8): verification of the steering computation (model stage).

Checks (all must pass; written to outputs/v3_steering/verify/verify.json):
  V1 shortcut == full pass. For the first 3 steered cells and every layer: candidate 1 at alpha 0, 2, 5,
     the first null feature at alpha 5 and the size-matched positive control are recomputed with the plain
     full forward pass (hooks_v2.run_with_edits, logits for every position, then the mean). delta_s must
     agree with the stored shortcut value within 1e-5 (absolute), and the steered mean logits within 1e-3.
  V2 zero edit. alpha = 1 (Steer adds (1 - 1) z W_dec = 0) on the shortcut and the full path gives
     delta_s = 0 and a logit change of exactly 0.
  V3 silent feature. A feature with z = 0 at every gene position of the cell gives exactly 0 at alpha 5.
  V4 device. One condition (first cell, L9 candidate 1, alpha 5) recomputed on the CPU (float32, full
     pass): delta_s within 1e-5 of the MPS value.
Each cell passes inputs_v3.assert_encoding_batch right before its forward passes.

Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering_verify.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_steering as V  # noqa: E402
from v3_steering import H, I  # noqa: E402

N_CELLS = 3
TOL_DS = 1e-5
TOL_LOGIT = 1e-3


@torch.no_grad()
def full_mean_logits(model, ids, saes, edits):
    logits, ed = H.run_with_edits(model, ids, saes, edits, return_logits=True, logits_device=V.DEVICE)
    return logits[0].mean(0).float().cpu().double().numpy(), ed


def main():
    t_start = time.time()
    V.VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    selj = V.load_json(V.OUT / "selection.json")
    cells = V.load_cells()
    G = {k: v.astype(np.float64) for k, v in np.load(V.OUT / "signatures.npz").items()}
    gE, gL = G["g_early"], G["g_late"]
    steer_idx = [int(i) for i in np.where(cells["group"] == "steer")[0]][:N_CELLS]
    for i in steer_idx:
        if not (V.STEER_DIR / f"cell_{i:03d}.npz").exists():
            raise SystemExit(f"steer output for cell {i} missing")
    M = V.Model()
    res = {"V1": [], "V2": [], "V3": [], "V4": {}, "encoding_checks": []}
    for i in steer_idx:
        H.check_memory(V.MIN_FREE_GB)
        res["encoding_checks"].append(M.check_cell(cells, i, f"verify cell {i}"))
        st = dict(np.load(V.STEER_DIR / f"cell_{i:03d}.npz"))
        tok = cells["token_ids"][i]
        T = len(tok)
        gm = V.gene_mask(T)
        ids = torch.from_numpy(tok[None, :]).to(V.DEVICE)
        m_clean_full, caps, kw = M.clean_full(ids)
        m_clean_full2, _ = full_mean_logits(M.model, ids, M.saes, [])
        res["V2"].append({"cell": i, "what": "two clean full passes", "maxabs": float(np.abs(m_clean_full - m_clean_full2).max())})
        for l in V.LAYERS:
            rows = V.unit_rows(selj, l)
            L = selj["layers"][str(l)]
            c0 = L["candidates"][0]["feature"]
            n0 = L["nulls"][0]["feature"]
            want = [("cand", c0, 0.0), ("cand", c0, 2.0), ("cand", c0, 5.0), ("null", n0, 5.0), ("pc_matched", -1, 1.0)]
            keep_rows = list(st[f"L{l}_keep32_rows"])
            for (kind, f, a) in want:
                r = rows.index((kind, f, a))
                if kind == "pc_matched":
                    v = torch.from_numpy(G[f"v_matched_L{l}"].astype(np.float32)).to(V.DEVICE)
                    edits = [H.AddVector(l, vector=v, positions=gm)]
                else:
                    edits = [H.Steer(l, feature=int(f), alpha=float(a), positions=gm)]
                m_st, _ = full_mean_logits(M.model, ids, M.saes, edits)
                ds_full = V.score(m_st, gL, gE) - V.score(m_clean_full, gL, gE)
                ds_short = float(st[f"L{l}_delta_s"][r])
                rec = {"cell": i, "layer": l, "kind": kind, "feature": int(f), "alpha": a,
                       "ds_full": ds_full, "ds_shortcut": ds_short, "absdiff": abs(ds_full - ds_short)}
                if r in keep_rows:
                    m_short = st[f"L{l}_keep32"][keep_rows.index(r)].astype(np.float64)
                    rec["logit_maxabs_diff"] = float(np.abs(m_short - m_st).max())
                rec["pass"] = rec["absdiff"] <= TOL_DS and rec.get("logit_maxabs_diff", 0.0) <= TOL_LOGIT
                res["V1"].append(rec)
            # V2 zero edit: shortcut and full
            h_in = caps[l].to(V.DEVICE)
            m_ref, _ = M.partial(l, h_in, kw, [])
            m_z, _ = M.partial(l, h_in, kw, [H.Steer(l, feature=int(c0), alpha=1.0, positions=gm)])
            m_zf, _ = full_mean_logits(M.model, ids, M.saes, [H.Steer(l, feature=int(c0), alpha=1.0, positions=gm)])
            res["V2"].append({"cell": i, "layer": l, "shortcut_maxabs": float(np.abs(m_z - m_ref).max()),
                              "full_maxabs": float(np.abs(m_zf - m_clean_full).max()),
                              "shortcut_ds": V.score(m_z, gL, gE) - V.score(m_ref, gL, gE),
                              "full_ds": V.score(m_zf, gL, gE) - V.score(m_clean_full, gL, gE)})
            # V3 silent feature: z = 0 at every gene position, but alive somewhere (z > 0 at some position of
            # some selection cell is not required; any feature works)
            z = M.saes[l].encode(h_in[0])[torch.from_numpy(gm).to(V.DEVICE)]
            silent = torch.where((z > 0).sum(0) == 0)[0].cpu().numpy()
            if len(silent):
                fs = int(silent[len(silent) // 2])
                m_s, elog = M.partial(l, h_in, kw, [H.Steer(l, feature=fs, alpha=5.0, positions=gm)])
                res["V3"].append({"cell": i, "layer": l, "feature": fs, "n_silent_features": int(len(silent)),
                                  "maxabs_logit_change": float(np.abs(m_s - m_ref).max()),
                                  "ds": V.score(m_s, gL, gE) - V.score(m_ref, gL, gE),
                                  "edit_norm": float(elog.get("delta_norm", -1))})
            del h_in, z
        del caps, kw
        H.free_device_cache()
        V.log(f"verify cell {i} done, {(time.time() - t_start) / 60:.1f} min")
    # V4 CPU recompute (first cell, L9 candidate 1, alpha 5)
    i = steer_idx[0]
    l = 9
    c0 = selj["layers"][str(l)]["candidates"][0]["feature"]
    r = V.unit_rows(selj, l).index(("cand", c0, 5.0))
    st = dict(np.load(V.STEER_DIR / f"cell_{i:03d}.npz"))
    del M
    H.free_device_cache()
    xt = H.load_model("cpu")
    saes_cpu = H.load_saes([l], device="cpu", sae_dir=V.SAE_DIR)
    tok = cells["token_ids"][i]
    gm = V.gene_mask(len(tok))
    ids = torch.from_numpy(tok[None, :])
    with torch.no_grad():
        lc, _ = H.run_with_edits(xt.model, ids, saes_cpu, [])
        ls, _ = H.run_with_edits(xt.model, ids, saes_cpu, [H.Steer(l, feature=int(c0), alpha=5.0, positions=gm)])
    mc = lc[0].double().mean(0).numpy()
    ms = ls[0].double().mean(0).numpy()
    ds_cpu = V.score(ms, gL, gE) - V.score(mc, gL, gE)
    res["V4"] = {"cell": i, "layer": l, "feature": int(c0), "alpha": 5.0, "ds_cpu": ds_cpu,
                 "ds_mps_shortcut": float(st[f"L{l}_delta_s"][r]),
                 "absdiff": abs(ds_cpu - float(st[f"L{l}_delta_s"][r])),
                 "clean_logits_cpu_vs_mps_maxabs": float(np.abs(mc - st["m_full"].astype(np.float64)).max())}
    v1_ok = all(x["pass"] for x in res["V1"])
    v2_ok = all(x.get("shortcut_maxabs", 0.0) == 0.0 and x.get("full_maxabs", 0.0) == 0.0
                and x.get("shortcut_ds", 0.0) == 0.0 and x.get("full_ds", 0.0) == 0.0 and x.get("maxabs", 0.0) == 0.0
                for x in res["V2"])
    v3_ok = len(res["V3"]) > 0 and all(x["maxabs_logit_change"] == 0.0 and x["ds"] == 0.0 for x in res["V3"])
    v4_ok = res["V4"]["absdiff"] <= TOL_DS
    res["summary"] = {
        "V1_shortcut_equals_full": {"pass": v1_ok, "n": len(res["V1"]),
                                    "max_abs_ds_diff": max(x["absdiff"] for x in res["V1"]),
                                    "max_logit_diff": max(x.get("logit_maxabs_diff", 0.0) for x in res["V1"]),
                                    "tolerance": {"ds": TOL_DS, "logit": TOL_LOGIT}},
        "V2_zero_edit_exact": {"pass": v2_ok, "n": len(res["V2"])},
        "V3_silent_feature_exact": {"pass": v3_ok, "n": len(res["V3"])},
        "V4_cpu_vs_mps": {"pass": v4_ok, "absdiff": res["V4"]["absdiff"], "tolerance": TOL_DS},
        "all_pass": bool(v1_ok and v2_ok and v3_ok and v4_ok),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    H.write_json(V.VERIFY_DIR / "verify.json", res)
    V.update_run_config("verify", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": res["summary"],
                                   "script_sha256": H.sha256_file(Path(__file__))})
    V.log(res["summary"])


if __name__ == "__main__":
    main()
