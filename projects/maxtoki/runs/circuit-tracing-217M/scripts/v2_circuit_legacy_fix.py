"""v2 circuit tracing: repair the legacy zero-edit control in the first cell of each early chunk (item D2).

Problem (found by v2_circuit_resume_check.py; see the docstring of
v2_circuit_trace.install_capture_hooks_first): transformers 5.5.4 installs its
hidden-state capture hooks on the first forward that asks for hidden states. In
the chunks run before the fix, that first forward was legacy_dz(s=0) of the first
cell of the chunk, so hidden_states[1] recorded the PATCHED block-0 output there.
In every other call the unpatched output is recorded (as in the deployed run).
Only 'legacy_zero' row 0 (source L0 -> target L1) of those cells is affected.

This script recomputes 'legacy_zero' (all 26 rows) for those cells with the
capture hooks installed first, checks that rows 1..25 equal the saved values
exactly, and writes the repaired array to a NEW file. The trace files are not
touched. v2_circuit_aggregate.py uses the repaired array when it exists.
Writes outputs/v2_circuit/trace_cells_legacy_fix/cell_XXX.npz and
outputs/v2_circuit/checks/legacy_fix.json. Resumable; --max-minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_circuit_trace as T  # noqa: E402

H = T.H
FIX_DIR = T.OUT / "trace_cells_legacy_fix"


def affected_cells():
    prog = json.loads((T.OUT / "trace_progress.json").read_text())
    return sorted(c["cells_processed"][0] for c in prog["chunks"]
                  if c["cells_processed"] and not c.get("capture_hooks_installed_first", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    t0 = time.time()
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    cells = affected_cells()
    toks, rows, h5 = T.load_cells()
    todo = [ci for ci in cells if not (FIX_DIR / f"cell_{ci:03d}.npz").exists()]
    if todo:
        H.check_memory(T.MIN_FREE_GB)
        torch.manual_seed(0)
        model = H.load_model(T.DEVICE).model
        saes = H.load_saes(range(T.N_SITES), device=T.DEVICE)
        T.install_capture_hooks_first(model, toks[0])
        for ci in todo:
            if time.time() - t0 > args.max_minutes * 60 - 30:
                break
            H.check_memory(T.MIN_FREE_GB)
            ids = torch.from_numpy(toks[ci][None, :]).to(T.DEVICE)
            with torch.no_grad():
                ed = T.run_full(model, saes, ids, [], capture=list(range(T.N_SITES)))
                h_clean = {s: ed.captured[s]["pre"].clone() for s in T.SOURCE_LAYERS}
                z_clean = {l: saes[l].encode(ed.captured[l]["pre"][0]) for l in range(1, T.N_SITES)}
                del ed
                lz = np.concatenate([T.legacy_dz(model, saes, ids, s, h_clean[s].clone(), z_clean)
                                     for s in T.SOURCE_LAYERS])
            np.savez(FIX_DIR / f"cell_{ci:03d}.npz", legacy_zero=lz, dataset_row=np.int64(rows[ci]))
            del h_clean, z_clean
            H.free_device_cache()
            print(f"cell {ci} fixed ({time.time() - t0:.0f}s)", flush=True)
    done = [ci for ci in cells if (FIX_DIR / f"cell_{ci:03d}.npz").exists()]
    if len(done) < len(cells):
        print(f"{len(done)}/{len(cells)} done; run again")
        return
    res = []
    for ci in cells:
        old = np.load(next(T.CELL_DIR.glob(f"cell_{ci:03d}_row*.npz")))["legacy_zero"]
        new = np.load(FIX_DIR / f"cell_{ci:03d}.npz")["legacy_zero"]
        res.append(dict(cell=ci, max_abs_diff_rows_1_to_25=float(np.abs(old[1:] - new[1:]).max()),
                        old_row0_max_abs=float(np.abs(old[0]).max()), new_row0_max_abs=float(np.abs(new[0]).max())))
    ok = all(r["max_abs_diff_rows_1_to_25"] == 0.0 and r["new_row0_max_abs"] == 0.0 for r in res)
    out = dict(description="legacy_zero recomputed with capture hooks installed first; rows 1..25 must equal the "
                           "saved values exactly; row 0 (L0->L1) must be exactly 0 as in every other cell",
               cells=cells, rows=res, pass_=bool(ok), env=H.env_info(T.DEVICE),
               script_sha256=H.sha256_file(__file__), wall_seconds_last_chunk=round(time.time() - t0, 1))
    H.write_json(T.CHECK_DIR / "legacy_fix.json", out)
    print(json.dumps({k: out[k] for k in ["cells", "pass_"]}))
    print("max diff rows 1..25:", max(r["max_abs_diff_rows_1_to_25"] for r in res),
          " max new row0:", max(r["new_row0_max_abs"] for r in res))


if __name__ == "__main__":
    main()
