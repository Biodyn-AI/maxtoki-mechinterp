"""v3 steering (item V3-8): describe the 15 candidate features (model, ~2 min). Descriptive only.

For 8 selection cells per stage (32 cells; the first 8 of each stage in cells.npz order), one clean pass;
for every candidate feature: the gene tokens where it fires most (summed code over cells), and the share of
gene positions where it is active, by stage. Every cell passes the encoding check before its pass.
Writes outputs/v3_steering/describe.json.

Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering_describe.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_steering as V  # noqa: E402
from v3_steering import H  # noqa: E402
from v3_steering_summarize import gene_namer  # noqa: E402

N_PER_STAGE = 8


def main():
    selj = V.load_json(V.OUT / "selection.json")
    cells = V.load_cells()
    gene = gene_namer()
    grp, stage = cells["group"], cells["stage"]
    idx = []
    for s in range(4):
        idx += [int(i) for i in np.where((grp == "selection") & (stage == s))[0][:N_PER_STAGE]]
    cand = {l: [c["feature"] for c in selj["layers"][str(l)]["candidates"]] for l in V.LAYERS}
    tot = {(l, f): defaultdict(float) for l in V.LAYERS for f in cand[l]}
    frac = {(l, f): {s: [] for s in range(4)} for l in V.LAYERS for f in cand[l]}
    M = V.Model()
    checks = []
    for i in idx:
        H.check_memory(V.MIN_FREE_GB)
        checks.append(M.check_cell(cells, i, f"describe cell {i}")["all_pass"])
        tok = cells["token_ids"][i]
        ids = torch.from_numpy(tok[None, :]).to(V.DEVICE)
        _, caps, _ = M.clean_full(ids)
        gm = V.gene_mask(len(tok))
        for l in V.LAYERS:
            z = M.saes[l].encode(caps[l][0].to(V.DEVICE)).cpu().numpy()[gm]
            g = tok[gm]
            for f in cand[l]:
                zf = z[:, f]
                for t, v in zip(g[zf > 0], zf[zf > 0]):
                    tot[(l, f)][int(t)] += float(v)
                frac[(l, f)][int(stage[i])].append(float((zf > 0).mean()))
        del caps
        H.free_device_cache()
    out = {"cells": idx, "encoding_checks_all_pass": all(checks), "features": {}}
    for (l, f), d in tot.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:12]
        totsum = sum(d.values())
        out["features"][f"L{l}_F{f}"] = {
            "rho": next(c["rho"] for c in selj["layers"][str(l)]["candidates"] if c["feature"] == f),
            "top_genes_by_summed_code": [{"gene": gene(t), "share_of_code": v / totsum} for t, v in top],
            "n_distinct_genes_where_active": len(d),
            "share_of_gene_positions_active_by_stage": {V.STAGE_SHORT[s]: float(np.mean(frac[(l, f)][s]))
                                                        for s in range(4)}}
        V.log(f"L{l} F{f}: {len(d)} genes; top " + ", ".join(f"{gene(t)} {v / totsum:.2f}" for t, v in top[:6])
              + " | active share by stage " + " ".join(f"{V.STAGE_SHORT[s]} {np.mean(frac[(l, f)][s]):.4f}" for s in range(4)))
    H.write_json(V.OUT / "describe.json", out)


if __name__ == "__main__":
    main()
