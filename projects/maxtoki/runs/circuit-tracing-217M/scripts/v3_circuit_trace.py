"""v3 Stage-2 circuit tracing: v3 SAEs + correctly encoded inputs + hooks_v2 (item V3-3, part 1).

What changed from v2 (scripts/v2_circuit_trace.py, not modified):
  * Inputs: the same 200 K562 non-targeting cells (rng(42).choice of the control rows, 200, sorted;
    deployed circuit_trace.py:100-123), now encoded with setup/inputs_v3.py:
    counts = round(expm1(X) / unit) ranked by counts / Geneformer gene median (v2: X = log1p(CP10k)
    ranked as if it were counts). Every cell passes inputs_v3.check_encoding at load time, and the
    cells of each chunk are checked again (assert_encoding_batch) before any forward pass.
  * SAEs: the v3 SAEs (runs/sae-atlas-217M/outputs/v3_sae/layer_XX/sae_final.pt), retrained on the
    correctly encoded inputs. Site = input of block l (= hidden_states[l]); layer 11 = input of lm_head.
  * Source features: the deployed RULE re-applied to the v3 SAEs (v3_circuit_annotate.py):
    30 per layer at 0, 3, 6, 9 with the largest summed -log10 p of enrichment.
  * The deployed-hook controls of v2 (legacy_zero, legacy_real) are dropped: they tested the old hook,
    which is not used here.
Unchanged from v2: edit = hooks_v2.Ablate(s, [f]) (delta = -z_f(x) W_dec[:, f] ADDED to the live
input of block s, every position incl. <bos>/<eos>); read-out = hooks_v2 captures of the input of each
site t = s+1..11 (never output_hidden_states); per cell and target feature: mean over positions of
z_ablated - z_clean; exact speed-ups (partial forward from block s; a source feature that is zero at
every position gives exact zeros without a forward pass); zero-edit control hooks_v2.ZeroDelta(s)
through the same partial forward, compared with the full clean forward.

Outputs (outputs/v3_circuit/):
  cells_tokens.npz       rows, lengths, tokens_flat (v3 tokens), tokens_deployed_flat (old tokens, for comparison only)
  cells_encoding.json    load-time encoding checks + old-vs-new order agreement
  source_features.json   copy of annotation/source_features.json
  combos_main.csv, combos_ctrl.csv
  trace_cells/cell_XXX_rowYYYYYY.npz   main (780, 4928), zero_v2 (26, 4928), n_active (120,), n_tokens, seconds
  trace_progress.json    wall time, memory and encoding check per chunk
Usage:
  OMP_NUM_THREADS=4 .venv/bin/python runs/circuit-tracing-217M/scripts/v3_circuit_trace.py --check-partial
  OMP_NUM_THREADS=4 .venv/bin/python runs/circuit-tracing-217M/scripts/v3_circuit_trace.py --max-minutes 8   (repeat until DONE)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
import inputs_v3 as I  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v3_circuit"
CELL_DIR = OUT / "trace_cells"
CHECK_DIR = OUT / "checks"
ANN = OUT / "annotation"
V3SAE = PROJ / "runs/sae-atlas-217M/outputs/v3_sae"
V2_CELLS = RUN / "outputs/v2_circuit/cells_tokens.npz"

DEVICE = "mps"
SEED = 42                     # deployed cell-selection seed
N_CELLS = 200                 # deployed
N_FEATURES_PER_LAYER = 30     # deployed
SOURCE_LAYERS = [0, 3, 6, 9]  # deployed
N_SITES = 12
D_SAE = 4928
MAX_LEN = 2048
MIN_FREE_GB = 3.0


def load_feats():
    f = json.loads((ANN / "source_features.json").read_text())
    feats = {int(s): [int(x) for x in v] for s, v in f.items()}
    assert sorted(feats) == SOURCE_LAYERS and all(len(v) == N_FEATURES_PER_LAYER for v in feats.values())
    return feats


def main_combos(feats):
    return [(s, f, t) for s in SOURCE_LAYERS for f in feats[s] for t in range(s + 1, N_SITES)]


def ctrl_combos():
    return [(s, t) for s in SOURCE_LAYERS for t in range(s + 1, N_SITES)]


def load_cells():
    """Same 200 rows as the deployed and v2 runs, correctly encoded (inputs_v3)."""
    path = OUT / "cells_tokens.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        offs = np.concatenate([[0], np.cumsum(z["lengths"])])
        toks = [z["tokens_flat"][offs[i]:offs[i + 1]] for i in range(len(z["lengths"]))]
        return toks, z["rows"].tolist(), str(z["h5_path"])
    cells, rows, h5, info = I.load_k562_control_cells(N_CELLS, pool=N_CELLS, seed=SEED, max_len=MAX_LEN, return_info=True)
    assert len(cells) == N_CELLS, len(cells)
    v2rows = np.load(V2_CELLS)["rows"].tolist()
    assert rows == v2rows, "cell rows differ from the v2 / deployed run"
    toks = [np.asarray(c.token_ids, dtype=np.int64) for c in cells]
    old = I.deployed_tokenize_X("k562", rows, MAX_LEN)
    old_toks = [np.asarray(c.token_ids, dtype=np.int64) for c in old]
    v2z = np.load(V2_CELLS)
    same_old_as_v2 = bool(np.array_equal(np.concatenate(old_toks), v2z["tokens_flat"]))
    # order agreement old vs new (audit measures)
    counts, _ = I.read_counts("k562", rows)
    vm = I.get_var_map("k562")
    agree = []
    with I.CountReader("k562", check=False) as r:
        for i, row in enumerate(rows):
            ref = I.full_order(counts[i], vm)
            x = r.stored_X(row)
            fed = I.full_order(x, vm)            # untruncated order the old runs used (stored X as values)
            agree.append(I.order_agreement(fed, ref, MAX_LEN))
    ag = pd.DataFrame(agree)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=np.asarray(rows, np.int64), lengths=np.asarray([len(t) for t in toks]),
             tokens_flat=np.concatenate(toks), lengths_deployed=np.asarray([len(t) for t in old_toks]),
             tokens_deployed_flat=np.concatenate(old_toks), h5_path=np.asarray(h5))
    H.write_json(OUT / "cells_encoding.json", dict(
        load_info={k: v for k, v in info.items() if k != "rows"},
        n_cells=len(rows), rows_equal_v2_and_deployed=True,
        old_tokens_rebuilt_equal_v2_saved_tokens=same_old_as_v2,
        n_cells_cut_at_2046_genes=int(sum(len(t) == MAX_LEN for t in toks)),
        old_vs_new_order=dict(spearman_full_mean=float(ag.spearman_full.mean()), spearman_full_min=float(ag.spearman_full.min()),
                              top200_overlap_mean=float(ag.top200_overlap.mean()), kept_set_overlap_mean=float(ag.kept_set_overlap.mean()),
                              same_position_share_mean=float(ag.same_position_share.mean())),
        counts_sha256=I.array_sha256(counts), tokens_sha256=I.array_sha256(np.concatenate(toks)),
        encoding=I.encoding_record("k562", rows)))
    return toks, rows, h5


def check_chunk_encoding(rows, toks, idx):
    """inputs_v3 encoding check on every cell of this chunk before any forward pass (aborts on failure)."""
    counts, row_checks = I.read_counts("k562", [rows[i] for i in idx])
    res = I.assert_encoding_batch(counts, [toks[i] for i in idx], I.get_var_map("k562"), MAX_LEN, label="v3_circuit chunk")
    res["count_rows_all_ok"] = all(c["expected_ok"] for c in row_checks)
    if not res["count_rows_all_ok"]:
        raise I.EncodingError("a count row failed the count-like check")
    return res


# ----------------------------------------------------------------------------- forward helpers (as v2)
class _Stop(Exception):
    pass


def _stop_hook(module, args):
    raise _Stop()


def run_full(model, saes, ids, edits, capture):
    with H.ResidualEditor(model, saes, edits=edits, capture=capture, capture_device=DEVICE) as ed:
        h = model.lm_head.register_forward_pre_hook(_stop_hook)  # registered after the capture hook
        try:
            with torch.no_grad():
                model(ids, use_cache=False)
        except _Stop:
            pass
        finally:
            h.remove()
    return ed


@torch.no_grad()
def _forward_from(model, h_start, s):
    """Blocks s..10, final norm, then lm_head (stopped by the pre-hook); same calls as
    LlamaModel.forward (transformers 5.5.4, modeling_llama.py:379-421)."""
    mm = model.model
    T = h_start.shape[1]
    position_ids = torch.arange(T, device=h_start.device).unsqueeze(0)
    causal_mask = create_causal_mask(config=mm.config, inputs_embeds=h_start, attention_mask=None,
                                     past_key_values=None, position_ids=position_ids)
    pos_emb = mm.rotary_emb(h_start, position_ids=position_ids)
    h = h_start
    for layer in mm.layers[s: mm.config.num_hidden_layers]:
        h = layer(h, attention_mask=causal_mask, position_embeddings=pos_emb,
                  position_ids=position_ids, past_key_values=None, use_cache=False)
    h = mm.norm(h)
    model.lm_head(h)


def run_from(model, saes, h_start, s, edits, capture):
    with H.ResidualEditor(model, saes, edits=edits, capture=capture, capture_device=DEVICE) as ed:
        hk = model.lm_head.register_forward_pre_hook(_stop_hook)
        try:
            _forward_from(model, h_start, s)
        except _Stop:
            pass
        finally:
            hk.remove()
    return ed


@torch.no_grad()
def dz_from_captures(ed, saes, z_clean, layers):
    rows = [(saes[l].encode(ed.captured[l]["pre"][0]) - z_clean[l]).mean(0) for l in layers]
    return torch.stack(rows).cpu().numpy().astype(np.float32)


@torch.no_grad()
def process_cell(model, saes, tok, feats):
    ids = torch.from_numpy(tok[None, :]).to(DEVICE)
    ed = run_full(model, saes, ids, [], capture=list(range(N_SITES)))
    h_clean = {s: ed.captured[s]["pre"].clone() for s in SOURCE_LAYERS}
    z_clean = {l: saes[l].encode(ed.captured[l]["pre"][0]) for l in range(1, N_SITES)}
    del ed
    main, n_active = [], []
    for s in SOURCE_LAYERS:
        down = list(range(s + 1, N_SITES))
        z_src = saes[s].encode(h_clean[s][0])
        for f in feats[s]:
            na = int((z_src[:, f] > 0).sum().item())
            if na == 0:          # exact: delta = 0 -> every downstream change is exactly 0
                main.append(np.zeros((len(down), D_SAE), np.float32))
                n_active.append(0)
                continue
            e = run_from(model, saes, h_clean[s], s, [H.Ablate(s, [f])], capture=down)
            assert e.edit_log[s]["n_active"][f] == na
            main.append(dz_from_captures(e, saes, z_clean, down))
            n_active.append(na)
            del e
        del z_src
    zero_v2 = []
    for s in SOURCE_LAYERS:
        down = list(range(s + 1, N_SITES))
        e = run_from(model, saes, h_clean[s], s, [H.ZeroDelta(s)], capture=down)
        zero_v2.append(dz_from_captures(e, saes, z_clean, down))
        del e
    del h_clean, z_clean
    return np.concatenate(main), np.concatenate(zero_v2), np.asarray(n_active, np.int64)


def check_partial(model, saes, toks, feats, n_cells=2):
    """Partial forward vs full forward, same hooks_v2 Ablate edit, first two source features per layer."""
    res = []
    for ci in range(n_cells):
        ids = torch.from_numpy(toks[ci][None, :]).to(DEVICE)
        ed = run_full(model, saes, ids, [], capture=list(range(N_SITES)))
        h_clean = {s: ed.captured[s]["pre"].clone() for s in SOURCE_LAYERS}
        z_clean = {l: saes[l].encode(ed.captured[l]["pre"][0]) for l in range(1, N_SITES)}
        del ed
        for s in SOURCE_LAYERS:
            down = list(range(s + 1, N_SITES))
            for f in feats[s][:2]:
                a = run_full(model, saes, ids, [H.Ablate(s, [f])], capture=down)
                b = run_from(model, saes, h_clean[s], s, [H.Ablate(s, [f])], capture=down)
                hmax = max(float((a.captured[l]["pre"] - b.captured[l]["pre"]).abs().max()) for l in down)
                da = dz_from_captures(a, saes, z_clean, down)
                db = dz_from_captures(b, saes, z_clean, down)
                res.append(dict(cell=ci, src_layer=s, feature=f, max_abs_diff_hidden=hmax,
                                max_abs_diff_dz=float(np.abs(da - db).max()), max_abs_dz=float(np.abs(da).max()),
                                n_active=a.edit_log[s]["n_active"][f]))
                del a, b
        H.free_device_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--check-partial", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    budget = args.max_minutes * 60
    OUT.mkdir(parents=True, exist_ok=True)
    CELL_DIR.mkdir(parents=True, exist_ok=True)

    gb = H.check_memory(MIN_FREE_GB)
    feats = load_feats()
    H.write_json(OUT / "source_features.json", {str(s): feats[s] for s in SOURCE_LAYERS})
    mc = main_combos(feats)
    if not (OUT / "combos_main.csv").exists():
        pd.DataFrame(mc, columns=["src_layer", "src_feature", "tgt_layer"]).to_csv(OUT / "combos_main.csv", index_label="row")
        pd.DataFrame(ctrl_combos(), columns=["src_layer", "tgt_layer"]).to_csv(OUT / "combos_ctrl.csv", index_label="row")
    toks, rows, h5 = load_cells()

    done = {int(p.name.split("_")[1]) for p in CELL_DIR.glob("cell_*.npz")}
    todo = [ci for ci in range(N_CELLS) if ci not in done]
    if args.check_partial:
        todo_check = [0, 1]
    else:
        todo_check = todo[:12]           # at most ~8 cells fit in one chunk; check more than that
    if not todo_check:
        print("DONE (nothing to do)")
        return
    enc = check_chunk_encoding(rows, toks, todo_check)   # raises EncodingError before any forward pass

    torch.manual_seed(0)
    xt = H.load_model(DEVICE)
    model = xt.model
    saes = H.load_saes(range(N_SITES), device=DEVICE, sae_dir=V3SAE)

    if args.check_partial:
        CHECK_DIR.mkdir(parents=True, exist_ok=True)
        res = check_partial(model, saes, toks, feats)
        out = dict(description="partial forward (start at block s from the clean input of block s) vs full forward, same "
                               "hooks_v2 Ablate edit; first two source features per layer; 2 cells; v3 SAEs, v3 inputs",
                   max_abs_diff_hidden=max(r["max_abs_diff_hidden"] for r in res),
                   max_abs_diff_dz=max(r["max_abs_diff_dz"] for r in res), rows=res, cell_rows=rows[:2],
                   encoding_check=enc, env=H.env_info(DEVICE), script_sha256=H.sha256_file(__file__),
                   wall_seconds=round(time.time() - t_start, 1))
        H.write_json(CHECK_DIR / "partial_forward_check.json", out)
        print(json.dumps({k: out[k] for k in ["max_abs_diff_hidden", "max_abs_diff_dz"]}))
        for r in res:
            print(r)
        return

    per_cell, processed = [], []
    status = "stopped: time budget"
    gb_min = gb
    for ci in todo:
        elapsed = time.time() - t_start
        est = max(per_cell[-3:]) if per_cell else 70.0
        if elapsed + est > budget:
            break
        if ci not in todo_check:
            status = "stopped: encoding-checked cells used up"
            break
        try:
            gb_now = H.check_memory(MIN_FREE_GB)
            gb_min = min(gb_min, gb_now)
        except H.MemoryGuardError as err:
            status = f"stopped: {err}"
            break
        t0 = time.time()
        main_arr, z0, nact = process_cell(model, saes, toks[ci], feats)
        dt = time.time() - t0
        assert main_arr.shape == (len(mc), D_SAE) and np.isfinite(main_arr).all()
        tmp = CELL_DIR / f"cell_{ci:03d}_row{rows[ci]}.tmp.npz"
        np.savez(tmp, main=main_arr, zero_v2=z0, n_active=nact, n_tokens=np.int64(len(toks[ci])),
                 seconds=np.float64(dt), dataset_row=np.int64(rows[ci]))
        tmp.replace(CELL_DIR / f"cell_{ci:03d}_row{rows[ci]}.npz")
        per_cell.append(dt); processed.append(ci)
        H.free_device_cache()
        print(f"cell {ci} row {rows[ci]} T={len(toks[ci])} {dt:.1f}s  zero_v2 max|dz|={np.abs(z0).max():.3g}  "
              f"active src feats {int((nact > 0).sum())}/120", flush=True)
    else:
        status = "all cells done"
    n_done = len(list(CELL_DIR.glob("cell_*.npz")))
    prog_path = OUT / "trace_progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"chunks": []}
    prog["chunks"].append(dict(start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                               wall_seconds=round(time.time() - t_start, 1), cells_processed=processed,
                               seconds_per_cell=[round(x, 1) for x in per_cell],
                               free_memory_gb_at_start=round(gb, 2), free_memory_gb_min=round(gb_min, 2),
                               mps_driver_gb_end=round(torch.mps.driver_allocated_memory() / 1e9, 2),
                               encoding_check=enc, status=status, n_cells_done_total=n_done,
                               trace_script_sha256=H.sha256_file(__file__)))
    H.write_json(prog_path, prog)
    print(f"{status}; {n_done}/{N_CELLS} cells done; chunk wall {time.time() - t_start:.0f}s")
    if n_done == N_CELLS:
        print("DONE")


if __name__ == "__main__":
    main()
