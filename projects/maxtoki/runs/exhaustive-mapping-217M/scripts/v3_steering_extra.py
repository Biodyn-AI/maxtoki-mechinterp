"""v3 steering (item V3-8): extra post-hoc checks on stored outputs (no model).

Added after the main summary was read. Not in design.json; the report labels these as post hoc.
  (1) per-donor split of the steered cells for every candidate (alpha 5 and 0);
  (2) each candidate's alpha-5 |delta_s| against ALL 100 null features pooled over layers;
  (3) direction of the mean logit change: cosine with (g_late - g_early), candidates vs nulls;
  (4) are the "top up genes" of the hit also the top up genes of random null features?
Writes outputs/v3_steering/extra.json.

Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering_extra.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_steering as V  # noqa: E402
from v3_steering import H  # noqa: E402
from v3_steering_summarize import gene_namer  # noqa: E402

ERY = ["AHSP", "ALAS2", "HBM", "GYPA", "SLC4A1", "HBG1", "HBG2", "HBZ", "HBD", "GYPB", "KLF1", "HBA1", "HBA2", "HBB"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    selj = V.load_json(V.OUT / "selection.json")
    cells = V.load_cells()
    G = {k: v.astype(np.float64) for k, v in np.load(V.OUT / "signatures.npz").items()}
    gd = G["g_late"] - G["g_early"]
    gene = gene_namer()
    steer_idx = [int(i) for i in np.where(cells["group"] == "steer")[0]]
    donors = np.array([str(cells["donor"][i]) for i in steer_idx])
    Z = [np.load(V.STEER_DIR / f"cell_{i:03d}.npz") for i in steer_idx]
    rng = np.random.default_rng(V.BOOT_SEED)
    out = {"n_cells": len(steer_idx), "donor_counts": {d: int((donors == d).sum()) for d in np.unique(donors)},
           "per_donor": {}, "pooled_null_alpha5": {}, "direction_cosine_alpha5": {}, "gene_lists": {}}
    # top genes of the readout itself
    out["gene_lists"]["top_up_in_g_late_minus_g_early"] = [gene(t) for t in np.argsort(-gd)[:15]]
    all_null_a5 = []
    for l in V.LAYERS:
        rows = V.unit_rows(selj, l)
        ds = np.stack([z[f"L{l}_delta_s"] for z in Z])          # (n, rows)
        for f in [x["feature"] for x in selj["layers"][str(l)]["nulls"]]:
            all_null_a5.append((l, f, float(ds[:, rows.index(("null", f, 5.0))].mean())))
    nulls_abs = np.array([abs(x[2]) for x in all_null_a5])
    for l in V.LAYERS:
        rows = V.unit_rows(selj, l)
        ds = np.stack([z[f"L{l}_delta_s"] for z in Z])
        dv = np.mean(np.stack([z[f"L{l}_dvec16"].astype(np.float64) for z in Z]), axis=0)   # (rows, V)
        nul_cos = [cos(dv[rows.index(("null", x["feature"], 5.0))], gd) for x in selj["layers"][str(l)]["nulls"]]
        for c in selj["layers"][str(l)]["candidates"]:
            f = c["feature"]
            key = f"L{l}_F{f}"
            pdn = {}
            for a in (0.0, 5.0):
                x = ds[:, rows.index(("cand", f, a))]
                for d in np.unique(donors):
                    xd = x[donors == d]
                    bi = rng.integers(0, len(xd), size=(V.N_BOOT, len(xd)))
                    m = xd[bi].mean(1)
                    pdn[f"alpha{a:g}_{d}"] = {"n": int(len(xd)), "mean": float(xd.mean()),
                                               "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
            out["per_donor"][key] = pdn
            m5 = float(ds[:, rows.index(("cand", f, 5.0))].mean())
            out["pooled_null_alpha5"][key] = {
                "abs_delta_s": abs(m5), "n_null_features_pooled": int(len(nulls_abs)),
                "n_null_at_least_as_large": int(np.sum(nulls_abs >= abs(m5))),
                "p_pooled": float((1 + np.sum(nulls_abs >= abs(m5))) / (1 + len(nulls_abs))),
                "ratio_to_largest_null": float(abs(m5) / nulls_abs.max())}
            cc = cos(dv[rows.index(("cand", f, 5.0))], gd)
            out["direction_cosine_alpha5"][key] = {
                "cos_mean_logit_change_with_g_late_minus_g_early": cc,
                "null_same_layer_mean": float(np.mean(nul_cos)), "null_same_layer_sd": float(np.std(nul_cos, ddof=1)),
                "null_same_layer_min_max": [float(np.min(nul_cos)), float(np.max(nul_cos))],
                "n_null_with_larger_abs_cos": int(np.sum(np.abs(nul_cos) >= abs(cc)))}
        for kind in ("pc_full", "pc_matched"):
            out["direction_cosine_alpha5"][f"L{l}_{kind}"] = cos(dv[rows.index((kind, -1, 1.0))], gd)
        # gene lists: hit vs nulls (L9 only is the interesting one, but record every layer)
        tops = {}
        for x in selj["layers"][str(l)]["nulls"]:
            t = [gene(g) for g in np.argsort(-dv[rows.index(("null", x["feature"], 5.0))])[:10]]
            tops[x["feature"]] = t
        n_ery = {f: sum(g in ERY for g in t) for f, t in tops.items()}
        out["gene_lists"][f"L{l}"] = {
            "null_features_with_any_erythroid_gene_in_top10": int(sum(v > 0 for v in n_ery.values())),
            "null_mean_erythroid_genes_in_top10": float(np.mean(list(n_ery.values()))),
            "candidates_erythroid_genes_in_top10": {
                c["feature"]: sum(g in ERY for g in [gene(t) for t in np.argsort(-dv[rows.index(("cand", c["feature"], 5.0))])[:10]])
                for c in selj["layers"][str(l)]["candidates"]},
            "erythroid_list": ERY}
    out["provenance"] = {"script": str(Path(__file__).resolve()), "script_sha256": H.sha256_file(Path(__file__).resolve()),
                         "selection_sha256": H.sha256_file(V.OUT / "selection.json"),
                         "signatures_sha256": H.sha256_file(V.OUT / "signatures.npz"),
                         "steer_files": len(Z), "device": "cpu (no model)", "bootstrap_seed": V.BOOT_SEED,
                         "time": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")}
    H.write_json(V.OUT / "extra.json", out)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
