"""v2 circuit tracing, resume check (item D2).

The trace (v2_circuit_trace.py) was interrupted after 171 of 200 cells when the
app quit. Before continuing it, this script checks that the finished outputs are
sound:
  (1) every finished cell file: right keys and shapes, finite values, dataset row
      and token count match cells_tokens.npz, zero-edit control exactly 0,
      rows of inactive source features exactly 0;
  (2) recompute finished cells from scratch with the SAME code
      (v2_circuit_trace.process_cell) and compare every saved array.
      Cells: the last finished cell before the interruption, plus one drawn with
      numpy default_rng(20261001) from the other finished cells.
Writes outputs/v2_circuit/checks/resume_check.json.
Usage: .venv/bin/python runs/circuit-tracing-217M/scripts/v2_circuit_resume_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_circuit_trace as T  # noqa: E402

H = T.H
OUT = T.OUT
CELL_DIR = T.CELL_DIR
PICK_SEED = 20261001
KEYS = {"main": (780, T.D_SAE), "zero_v2": (26, T.D_SAE), "legacy_zero": (26, T.D_SAE),
        "legacy_real": (26, T.D_SAE), "n_active": (120,)}


def static_checks(toks, rows):
    files = sorted(p for p in CELL_DIR.glob("cell_*.npz") if not p.name.startswith("._"))
    bad = []
    n_inactive_rows, n_inactive_nonzero = 0, 0
    cells = []
    for p in files:
        ci = int(p.name.split("_")[1])
        cells.append(ci)
        z = np.load(p)
        prob = []
        for k, shp in KEYS.items():
            if k not in z or z[k].shape != shp:
                prob.append(f"{k} shape {z[k].shape if k in z else None}")
        for k in ["main", "zero_v2", "legacy_zero", "legacy_real"]:
            if not np.isfinite(z[k]).all():
                prob.append(f"{k} not finite")
        if np.abs(z["zero_v2"]).max() != 0.0:
            prob.append("zero_v2 not exactly 0")
        if int(z["dataset_row"]) != rows[ci]:
            prob.append("dataset_row mismatch")
        if int(z["n_tokens"]) != len(toks[ci]):
            prob.append("n_tokens mismatch")
        if f"row{rows[ci]}" not in p.name:
            prob.append("file name row mismatch")
        # inactive source features -> their 11-s rows must be exactly zero
        nact = z["n_active"]
        main = z["main"]
        r0 = 0
        fi = 0
        for s in T.SOURCE_LAYERS:
            n_down = T.N_SITES - (s + 1)
            for _ in range(T.N_FEATURES_PER_LAYER):
                if nact[fi] == 0:
                    n_inactive_rows += 1
                    if np.abs(main[r0:r0 + n_down]).max() != 0.0:
                        n_inactive_nonzero += 1
                r0 += n_down
                fi += 1
        if prob:
            bad.append(dict(cell=ci, file=p.name, problems=prob))
    return dict(n_files=len(files), cells=sorted(cells), contiguous_0_to_n=sorted(cells) == list(range(len(files))),
                n_bad_files=len(bad), bad=bad, n_inactive_feature_cases=n_inactive_rows,
                n_inactive_feature_cases_with_nonzero_rows=n_inactive_nonzero)


def main():
    t0 = time.time()
    toks, rows, h5 = T.load_cells()
    st = static_checks(toks, rows)
    print(json.dumps({k: v for k, v in st.items() if k != "cells"}, indent=1))
    done = st["cells"]
    last = max(done)
    rng = np.random.default_rng(PICK_SEED)
    other = int(rng.choice([c for c in done if c != last]))
    recheck = [last, other]

    gb = H.check_memory(T.MIN_FREE_GB)
    feats = T.select_source_features()
    torch.manual_seed(0)
    xt = H.load_model(T.DEVICE)
    model = xt.model
    saes = H.load_saes(range(T.N_SITES), device=T.DEVICE)
    saes_cpu = H.load_saes(T.SOURCE_LAYERS, device="cpu")
    rec = []
    for ci in recheck:
        H.check_memory(T.MIN_FREE_GB)
        t1 = time.time()
        main_arr, z0, lz, lr, nact = T.process_cell(model, saes, saes_cpu, toks[ci], feats)
        dt = time.time() - t1
        p = next(CELL_DIR.glob(f"cell_{ci:03d}_row*.npz"))
        z = np.load(p)
        r = dict(cell=ci, dataset_row=rows[ci], seconds=round(dt, 1))
        for k, a in [("main", main_arr), ("zero_v2", z0), ("legacy_zero", lz), ("legacy_real", lr), ("n_active", nact)]:
            r[f"max_abs_diff_{k}"] = float(np.abs(a.astype(np.float64) - z[k].astype(np.float64)).max())
            r[f"max_abs_{k}"] = float(np.abs(z[k]).max())
        rec.append(r)
        print(r, flush=True)
        H.free_device_cache()
    ok = (st["n_bad_files"] == 0 and st["contiguous_0_to_n"] and st["n_inactive_feature_cases_with_nonzero_rows"] == 0
          and all(r["max_abs_diff_main"] == 0.0 and r["max_abs_diff_n_active"] == 0.0 and r["max_abs_diff_zero_v2"] == 0.0
                  and r["max_abs_diff_legacy_zero"] == 0.0 and r["max_abs_diff_legacy_real"] == 0.0 for r in rec))
    out = dict(description="resume check before continuing the interrupted trace", static=st, recompute=rec,
               recompute_seed=PICK_SEED, pass_=bool(ok), free_memory_gb_at_start=round(gb, 2),
               env=H.env_info(T.DEVICE), script_sha256=H.sha256_file(__file__),
               trace_script_sha256=H.sha256_file(T.__file__), wall_seconds=round(time.time() - t0, 1))
    H.write_json(T.CHECK_DIR / "resume_check.json", out)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
