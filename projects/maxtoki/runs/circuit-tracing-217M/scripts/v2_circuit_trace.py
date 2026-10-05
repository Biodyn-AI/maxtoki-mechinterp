"""v2 Stage-2 circuit tracing with fixed hooks (revision item D2, part 1).

What the deployed script did (scripts/circuit_trace.py, NOT modified):
  * 200 K562 non-targeting cells (rng(42).choice, sorted), 2,048 tokens max.
  * 30 source SAE features at each of layers 0, 3, 6, 9 (largest summed
    -log10 p of enrichment).
  * Per cell and source feature: zero the feature in SAE space at layer s,
    take delta = decode(z_abl) - decode(z), and patch hidden_states[s] + delta
    in as the OUTPUT of block s with a forward hook (bug: block s is deleted).
  * Read the SAE codes at every downstream layer from output_hidden_states,
    take the mean over token positions of (z_ablated - z_clean) -> one 4,928-d
    vector per (cell, source feature, target layer).
  * Edge if |Cohen's d| > 0.5 and consistency > 0.7 over the 200 cells.

What this script does instead (setup/hooks_v2.py):
  * The same cells, the same 120 source features, the same read-out
    (mean over positions of z_ablated - z_clean at each layer s+1..11).
  * The edit is hooks_v2.Ablate(s, [f]): delta = -z_f(x) W_dec[:, f] is ADDED
    to the INPUT of block s (the tensor SAE s was trained on), computed from the
    live input. Read-outs are hooks_v2 captures (never output_hidden_states).
  * Speed-up (exact): for a source at layer s the ablated pass starts at block s
    from the clean input of block s (captured in the clean pass). Blocks 0..s-1
    cannot be changed by an edit at s (hooks test (ii): 0.0 difference). The
    hooks_v2 pre-hook still applies the edit at block s. The forward stops at the
    lm_head pre-hook (the L11 read-out); logits are never needed.
    The zero-edit control below runs through this same partial forward and is
    compared with the full clean forward, so it also checks the speed-up.

Per cell it also runs three controls:
  zero_v2     : hooks_v2.ZeroDelta(s), partial forward, 4 source layers.
                Must give exactly 0 at every downstream layer.
  legacy_zero : the deployed hook logic with delta = 0 (forward hook on
                layers[s] returning clean hidden_states[s]); 4 source layers.
  legacy_real : the deployed hook logic with the first source feature of each
                layer (reproduction check against outputs/circuit_edges.csv).

Outputs (resumable, one file per cell):
  outputs/v2_circuit/cells_tokens.npz        token ids + dataset rows of the 200 cells
  outputs/v2_circuit/source_features.json    the 120 source features (checked = deployed)
  outputs/v2_circuit/combos_main.csv         row order of the 'main' array
  outputs/v2_circuit/combos_ctrl.csv         row order of the control arrays
  outputs/v2_circuit/trace_cells/cell_XXX_rowYYYYYY.npz
        main        (780, 4928) float32  mean_t[z_abl - z_clean] per combo
        zero_v2     (26, 4928)  float32
        legacy_zero (26, 4928)  float32
        legacy_real (26, 4928)  float32
        n_active    (120,) int  positions where the source feature is active
        n_tokens, seconds
  outputs/v2_circuit/trace_progress.json     wall time per chunk
Usage:
  .venv/bin/python runs/circuit-tracing-217M/scripts/v2_circuit_trace.py --check-partial
  .venv/bin/python runs/circuit-tracing-217M/scripts/v2_circuit_trace.py --max-minutes 8   (repeat until DONE)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v2_circuit"
CELL_DIR = OUT / "trace_cells"
CHECK_DIR = OUT / "checks"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OLD_EDGES = RUN / "outputs/circuit_edges.csv"

DEVICE = "mps"
SEED = 42                    # deployed cell-selection seed
N_CELLS = 200                # deployed
N_FEATURES_PER_LAYER = 30    # deployed
SOURCE_LAYERS = [0, 3, 6, 9]  # deployed
N_SITES = 12                 # hidden_states entries 0..11
D_SAE = 4928
MIN_FREE_GB = 3.0


# ----------------------------------------------------------------------------- setup
def select_source_features() -> dict:
    """Deployed Phase 1 (circuit_trace.py:133-145), re-implemented without the
    deprecated groupby.apply; checked against the deployed edge file."""
    feats = {}
    for s in SOURCE_LAYERS:
        e = pd.read_csv(ANN_DIR / f"layer_{s:02d}/significant_enrichments.csv")
        score = e.groupby("feature_id")["p_raw"].apply(
            lambda p: (-np.log10(p.clip(1e-300))).sum()).sort_values(ascending=False)
        feats[s] = [int(x) for x in score.head(N_FEATURES_PER_LAYER).index.tolist()]
    old = pd.read_csv(OLD_EDGES, usecols=["src_layer", "src_feature"]).drop_duplicates()
    for s in SOURCE_LAYERS:
        old_order = old[old.src_layer == s].src_feature.tolist()
        assert old_order == feats[s], f"source features at L{s} differ from the deployed run"
    return feats


def main_combos(feats):
    return [(s, f, t) for s in SOURCE_LAYERS for f in feats[s] for t in range(s + 1, N_SITES)]


def ctrl_combos():
    return [(s, t) for s in SOURCE_LAYERS for t in range(s + 1, N_SITES)]


def load_cells():
    """Same 200 cells as circuit_trace.py:100-123 (rng(42).choice(ctrl_all, 200),
    sorted). hooks_v2.load_k562_control_cells with pool=200 draws exactly this."""
    path = OUT / "cells_tokens.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        rows = z["rows"].tolist()
        lens = z["lengths"]
        flat = z["tokens_flat"]
        offs = np.concatenate([[0], np.cumsum(lens)])
        toks = [flat[offs[i]:offs[i + 1]] for i in range(len(lens))]
        return toks, rows, str(z["h5_path"])
    cells, rows, h5 = H.load_k562_control_cells(N_CELLS, pool=N_CELLS, seed=SEED, max_len=2048)
    assert len(cells) == N_CELLS, len(cells)
    toks = [np.asarray(c.token_ids, dtype=np.int64) for c in cells]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=np.asarray(rows, np.int64), lengths=np.asarray([len(t) for t in toks]),
             tokens_flat=np.concatenate(toks), h5_path=np.asarray(h5))
    return toks, rows, h5


# ----------------------------------------------------------------------------- forward helpers
class _Stop(Exception):
    pass


def _stop_hook(module, args):
    raise _Stop()


def run_full(model, saes, ids, edits, capture):
    """Full forward with hooks_v2 edits/captures (captures kept on the device);
    stops at the lm_head pre-hook after the L11 capture."""
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
    """Blocks s..10, final norm, then lm_head (stopped by the pre-hook).
    Same calls as LlamaModel.forward (transformers 5.5.4, modeling_llama.py:379-421)."""
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
def legacy_dz(model, saes, ids, s, h_patched, z_clean):
    """Deployed hook logic (circuit_trace.py:209-221): forward hook on layers[s]
    REPLACES its output with h_patched; read-out from output_hidden_states."""
    def _hook(module, inp, out, _p=h_patched):
        return _p.to(out.device)
    hk = model.model.layers[s].register_forward_hook(_hook)
    try:
        out = model(ids, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        hk.remove()
    rows = [(saes[l].encode(out.hidden_states[l][0]) - z_clean[l]).mean(0) for l in range(s + 1, N_SITES)]
    del out
    return torch.stack(rows).cpu().numpy().astype(np.float32)


@torch.no_grad()
def install_capture_hooks_first(model, tok):
    """Added 2026-10-01 after cell 170 (resume check, checks/resume_check.json).
    transformers 5.5.4 installs its output_hidden_states capture hooks lazily, on
    the first forward that asks for hidden states (utils/output_capturing.py:182-198).
    Forward hooks run in registration order. In each chunk the first such forward
    was legacy_dz(s=0) of the first cell, AFTER its patch hook was registered, so
    hidden_states[1] recorded the patched tensor there; in every later call the
    capture hook runs first and records the unpatched block-0 output (as in the
    deployed run). Only the legacy_zero row (L0 -> L1) of the first cell of each
    chunk was affected; 'main', 'zero_v2' and 'legacy_real' never read
    output_hidden_states first and are unaffected (resume check: 0.0 difference).
    One short forward here installs the capture hooks before any legacy hook."""
    ids = torch.from_numpy(np.asarray(tok[:16])[None, :]).to(DEVICE)
    out = model(ids, output_hidden_states=True, use_cache=False, return_dict=True)
    assert getattr(model, "_output_capturing_hooks_installed", False) or \
        getattr(model.model, "_output_capturing_hooks_installed", False)
    del out


# ----------------------------------------------------------------------------- one cell
@torch.no_grad()
def process_cell(model, saes, saes_cpu, tok, feats):
    ids = torch.from_numpy(tok[None, :]).to(DEVICE)
    # clean pass: capture all 12 sites
    ed = run_full(model, saes, ids, [], capture=list(range(N_SITES)))
    h_clean = {s: ed.captured[s]["pre"].clone() for s in SOURCE_LAYERS}           # (1,T,D) on device
    z_clean = {l: saes[l].encode(ed.captured[l]["pre"][0]) for l in range(1, N_SITES)}
    del ed
    main, n_active = [], []
    for s in SOURCE_LAYERS:
        down = list(range(s + 1, N_SITES))
        # Exact skip (added after cell 4): a feature with z_f = 0 at every position
        # gives delta = 0, so every downstream change is exactly 0. Checked on cells
        # 0-4, which were run without the skip (all 70 inactive cases gave exact zeros),
        # and by hooks test (iv).
        z_src = saes[s].encode(h_clean[s][0])
        for f in feats[s]:
            if int((z_src[:, f] > 0).sum().item()) == 0:
                main.append(np.zeros((len(down), D_SAE), np.float32))
                n_active.append(0)
                continue
            e = run_from(model, saes, h_clean[s], s, [H.Ablate(s, [f])], capture=down)
            main.append(dz_from_captures(e, saes, z_clean, down))
            n_active.append(e.edit_log[s]["n_active"][f])
            del e
        del z_src
    zero_v2, legacy_zero, legacy_real = [], [], []
    for s in SOURCE_LAYERS:
        down = list(range(s + 1, N_SITES))
        e = run_from(model, saes, h_clean[s], s, [H.ZeroDelta(s)], capture=down)
        zero_v2.append(dz_from_captures(e, saes, z_clean, down))
        del e
        # legacy, zero delta: the hook alone
        legacy_zero.append(legacy_dz(model, saes, ids, s, h_clean[s].clone(), z_clean))
        # legacy, first source feature: deployed delta computed on CPU as deployed
        f0 = feats[s][0]
        h_cpu = h_clean[s][0].cpu()
        sae_c = saes_cpu[s]
        z_src = sae_c.sae.encode(h_cpu, sae_c.mu)
        z_abl = z_src.clone()
        z_abl[:, f0] = 0.0
        delta = sae_c.sae.decode(z_abl, sae_c.mu) - sae_c.sae.decode(z_src, sae_c.mu)
        h_patched = (h_cpu + delta).to(DEVICE).unsqueeze(0)
        legacy_real.append(legacy_dz(model, saes, ids, s, h_patched, z_clean))
    del h_clean, z_clean
    return (np.concatenate(main), np.concatenate(zero_v2), np.concatenate(legacy_zero),
            np.concatenate(legacy_real), np.asarray(n_active, np.int64))


# ----------------------------------------------------------------------------- partial-forward check
def check_partial(model, saes, toks, feats, n_cells=2):
    """Partial forward (start at block s) vs full forward (hooks_v2 standard),
    same Ablate edit, first source feature of each layer."""
    res = []
    for ci in range(n_cells):
        ids = torch.from_numpy(toks[ci][None, :]).to(DEVICE)
        ed = run_full(model, saes, ids, [], capture=list(range(N_SITES)))
        h_clean = {s: ed.captured[s]["pre"].clone() for s in SOURCE_LAYERS}
        z_clean = {l: saes[l].encode(ed.captured[l]["pre"][0]) for l in range(1, N_SITES)}
        del ed
        for s in SOURCE_LAYERS:
            down = list(range(s + 1, N_SITES))
            for f in [feats[s][0], feats[s][1]]:
                a = run_full(model, saes, ids, [H.Ablate(s, [f])], capture=down)
                b = run_from(model, saes, h_clean[s], s, [H.Ablate(s, [f])], capture=down)
                hmax = max(float((a.captured[l]["pre"] - b.captured[l]["pre"]).abs().max()) for l in down)
                da = dz_from_captures(a, saes, z_clean, down)
                db = dz_from_captures(b, saes, z_clean, down)
                res.append(dict(cell=ci, src_layer=s, feature=f,
                                max_abs_diff_hidden=hmax,
                                max_abs_diff_dz=float(np.abs(da - db).max()),
                                max_abs_dz=float(np.abs(da).max()),
                                n_active=a.edit_log[s]["n_active"][f]))
                del a, b
        H.free_device_cache()
    return res


# ----------------------------------------------------------------------------- main
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
    feats = select_source_features()
    H.write_json(OUT / "source_features.json", {str(s): feats[s] for s in SOURCE_LAYERS})
    mc = main_combos(feats)
    if not (OUT / "combos_main.csv").exists():
        pd.DataFrame(mc, columns=["src_layer", "src_feature", "tgt_layer"]).to_csv(OUT / "combos_main.csv", index_label="row")
        pd.DataFrame(ctrl_combos(), columns=["src_layer", "tgt_layer"]).to_csv(OUT / "combos_ctrl.csv", index_label="row")
    toks, rows, h5 = load_cells()

    torch.manual_seed(0)
    xt = H.load_model(DEVICE)
    model = xt.model
    saes = H.load_saes(range(N_SITES), device=DEVICE)
    saes_cpu = H.load_saes(SOURCE_LAYERS, device="cpu")
    install_capture_hooks_first(model, toks[0])

    if args.check_partial:
        CHECK_DIR.mkdir(parents=True, exist_ok=True)
        res = check_partial(model, saes, toks, feats)
        out = dict(description="partial forward (start at block s from the clean input of block s) vs full "
                               "forward, same hooks_v2 Ablate edit; first two source features per layer; 2 cells",
                   max_abs_diff_hidden=max(r["max_abs_diff_hidden"] for r in res),
                   max_abs_diff_dz=max(r["max_abs_diff_dz"] for r in res),
                   rows=res, cell_rows=rows[:2], env=H.env_info(DEVICE),
                   script_sha256=H.sha256_file(__file__), wall_seconds=round(time.time() - t_start, 1))
        H.write_json(CHECK_DIR / "partial_forward_check.json", out)
        print(json.dumps({k: out[k] for k in ["max_abs_diff_hidden", "max_abs_diff_dz"]}))
        for r in res:
            print(r)
        return

    done = {int(p.name.split("_")[1]) for p in CELL_DIR.glob("cell_*.npz")}
    todo = [ci for ci in range(N_CELLS) if ci not in done]
    per_cell = []
    status = "stopped: time budget"
    for ci in todo:
        elapsed = time.time() - t_start
        est = max(per_cell[-3:]) if per_cell else 75.0
        if elapsed + est > budget:
            break
        try:
            H.check_memory(MIN_FREE_GB)
        except H.MemoryGuardError as err:
            status = f"stopped: {err}"
            break
        t0 = time.time()
        main_arr, z0, lz, lr, nact = process_cell(model, saes, saes_cpu, toks[ci], feats)
        dt = time.time() - t0
        assert main_arr.shape == (len(mc), D_SAE)
        tmp = CELL_DIR / f"cell_{ci:03d}_row{rows[ci]}.tmp.npz"
        np.savez(tmp, main=main_arr, zero_v2=z0, legacy_zero=lz, legacy_real=lr, n_active=nact,
                 n_tokens=np.int64(len(toks[ci])), seconds=np.float64(dt), dataset_row=np.int64(rows[ci]))
        tmp.replace(CELL_DIR / f"cell_{ci:03d}_row{rows[ci]}.npz")
        per_cell.append(dt)
        H.free_device_cache()
        print(f"cell {ci} row {rows[ci]} T={len(toks[ci])} {dt:.1f}s  zero_v2 max|dz|={np.abs(z0).max():.3g}  "
              f"legacy_zero max|dz|={np.abs(lz).max():.3g}", flush=True)
    else:
        status = "all cells done"
    n_done = len(list(CELL_DIR.glob("cell_*.npz")))
    prog_path = OUT / "trace_progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"chunks": []}
    prog["chunks"].append(dict(start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                               wall_seconds=round(time.time() - t_start, 1),
                               cells_processed=[ci for ci in todo[:len(per_cell)]],
                               seconds_per_cell=[round(x, 1) for x in per_cell],
                               free_memory_gb_at_start=round(gb, 2),
                               mps_driver_gb_end=round(torch.mps.driver_allocated_memory() / 1e9, 2),
                               status=status, n_cells_done_total=n_done,
                               trace_script_sha256=H.sha256_file(__file__),
                               capture_hooks_installed_first=True))
    H.write_json(prog_path, prog)
    print(f"{status}; {n_done}/{N_CELLS} cells done; chunk wall {time.time() - t_start:.0f}s")
    if n_done == N_CELLS:
        print("DONE")


if __name__ == "__main__":
    main()
