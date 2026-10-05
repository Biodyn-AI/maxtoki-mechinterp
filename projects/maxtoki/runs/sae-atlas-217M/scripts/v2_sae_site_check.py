"""v2 SAE site check: is SAE layer_XX matched to the hooks_v2 site XX?

For K562 control cells (same pool as the hook tests), capture the live input of
every hooks_v2 site (blocks 0..10 and lm_head = site 11) in one clean forward
pass. For each SAE l, compute variance explained (training formula,
topk_sae.py:147-151) on site l and on the neighbouring sites l-1 and l+1.
If the SAE was trained on hidden_states[l] (= site l), VE should be highest at
site l and close to the value in phase1/layer_XX/results.json.

Usage: .venv/bin/python runs/sae-atlas-217M/scripts/v2_sae_site_check.py [--n-cells 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

OUT = PROJ / "runs/sae-atlas-217M/outputs/v2_sae_site_check"
CELL_SEED = 42


def ve(s: H.LoadedSAE, x: torch.Tensor) -> float:
    with torch.no_grad():
        xh = s.sae.decode(s.encode(x), s.mu)
        return float(1.0 - ((x - xh) ** 2).mean() / ((x - s.mu) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cells, rows, h5 = H.load_k562_control_cells(args.n_cells, pool=100, seed=CELL_SEED)
    H.check_memory(args.min_free_gb)
    xt = H.load_model(args.device)
    model = xt.model
    nb = H.n_blocks(model)
    sites = list(range(nb + 1))
    saes = H.load_saes(sites, device=args.device)

    acts = {l: [] for l in sites}
    for c in cells:
        H.check_memory(args.min_free_gb)
        ids = torch.from_numpy(c.token_ids[None, :])
        _, ed = H.run_with_edits(model, ids, saes, edits=[], capture=sites, return_logits=False)
        for l in sites:
            acts[l].append(ed.hidden(l, "pre"))
        H.free_device_cache()
    X = {l: torch.cat(acts[l], 0).to(args.device) for l in sites}

    table = {}
    for l in sites:
        s = saes[l]
        rec = {"results_json_ve": json.loads((H.SAE_DIR / f"layer_{l:02d}/results.json").read_text())["variance_explained"]}
        for m in (l - 1, l, l + 1):
            if 0 <= m <= nb:
                rec[f"ve_at_site_{m}"] = ve(s, X[m])
        best = max((k for k in rec if k.startswith("ve_at_site_")), key=lambda k: rec[k])
        rec["best_site"] = int(best.rsplit("_", 1)[1])
        rec["matches"] = rec["best_site"] == l
        table[f"sae_layer_{l:02d}"] = rec
        print(l, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})

    summary = {"n_cells": len(cells), "n_positions": int(X[0].shape[0]),
               "all_match": all(r["matches"] for r in table.values()), "table": table}
    H.write_json(OUT / "summary.json", summary)
    cfg = H.env_info(args.device)
    cfg.update({"script": str(Path(__file__).resolve()), "script_sha256": H.sha256_file(__file__),
                "seeds": {"cell_selection": CELL_SEED},
                "cells": {"dataset": h5, "dataset_rows": rows},
                "sae_dir": str(H.SAE_DIR), "wall_seconds": round(time.time() - t0, 1)})
    H.write_json(OUT / "run_config.json", cfg)
    print("all_match:", summary["all_match"])


if __name__ == "__main__":
    main()
