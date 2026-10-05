"""v2 circuit tracing, spot check: recompute 3 edges from scratch (item D2).

Independent of setup/hooks_v2.py and of v2_circuit_trace.py except for the model
weights, the SAE weights and the tokenizer:
  * cells are re-drawn and re-tokenized from the h5ad with the deployed recipe
    (circuit_trace.py:100-123: rng(42).choice of K562 non-targeting rows, 200, sorted);
  * the edit is a forward hook on the module whose OUTPUT is hidden_states[s]
    (embed_tokens for s = 0, decoder block s-1 otherwise) that ADDS
    delta = decode(z with feature f zeroed) - decode(z)   (CPU SAE, as deployed);
  * the read-out is output_hidden_states[t] (t > s) encoded with the CPU SAE t;
  * per cell: dz = mean over positions of (z_ablated[:, j] - z_clean[:, j]);
  * d and consistency are computed with plain numpy (ddof = 1, + 1e-12 as deployed).
The three edges are drawn with rng(20261001) from three (source layer, target layer)
pairs: L0->L1 and L9->L10 (pairs with no deployed edges) and L6->L11.
Writes outputs/v2_circuit/spotcheck/ (resumable per cell; --max-minutes).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import TopKSAE  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402
from dataset_loader import resolve as load_ds, SYM2ENS_PKL  # noqa: E402
import hooks_v2 as H  # noqa: E402  (only for the memory guard, hashing and json writing)

RUN = PROJ / "runs/circuit-tracing-217M"
V2 = RUN / "outputs/v2_circuit"
OUT = V2 / "spotcheck"
SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
DEVICE = "mps"
SEED = 42
N_CELLS = 200
PICK_SEED = 20261001
PAIRS = [(0, 1), (9, 10), (6, 11)]


def load_cpu_sae(l):
    ck = torch.load(SAE_DIR / f"layer_{l:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ck["config"]["d_model"], ck["config"]["d_sae"], ck["config"]["k"])
    sae.W_enc.weight.data = ck["W_enc_weight"]; sae.W_enc.bias.data = ck["W_enc_bias"]
    sae.W_dec.weight.data = ck["W_dec_weight"]; sae.eval()
    return sae, ck["mu"]


def draw_cells():
    ds = load_ds("k562")
    rng = np.random.default_rng(SEED)
    ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    idx = np.sort(rng.choice(ctrl_all, size=N_CELLS, replace=False))
    with h5py.File(ds.h5_path, "r") as f:
        X = np.empty((N_CELLS, ds.n_genes_total), dtype=np.float32)
        for i in range(0, N_CELLS, 100):
            X[i:i + 100] = f["X"][idx[i:i + 100], :]
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    sym2ens = pickle.load(open(SYM2ENS_PKL, "rb"))
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping([sym2ens.get(s) for s in var_symbols])
    toks = [tok.tokenize_cell(X[i], vi, vt, vm, max_len=2048).token_ids for i in range(N_CELLS)]
    return idx, toks


def pick_edges():
    path = OUT / "edges_picked.json"
    if path.exists():
        return json.loads(path.read_text())
    e = pd.read_csv(V2 / "circuit_edges_v2.csv")
    rng = np.random.default_rng(PICK_SEED)
    picked = []
    for s, t in PAIRS:
        sub = e[(e.src_layer == s) & (e.tgt_layer == t)].reset_index(drop=True)
        if len(sub) == 0:
            continue
        r = sub.iloc[int(rng.integers(0, len(sub)))]
        picked.append({k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in sub.columns})
    H.write_json(path, picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    edges = pick_edges()
    cache = OUT / "cells_redrawn.npz"
    if cache.exists():
        z = np.load(cache)
        rows = z["rows"]; offs = np.concatenate([[0], np.cumsum(z["lengths"])])
        toks = [z["flat"][offs[i]:offs[i + 1]] for i in range(N_CELLS)]
    else:
        rows, toks = draw_cells()
        np.savez(cache, rows=rows, lengths=[len(t) for t in toks], flat=np.concatenate(toks))
    # same cells as the main run?
    main_cells = np.load(V2 / "cells_tokens.npz")
    same_rows = bool(np.array_equal(rows, main_cells["rows"]))
    same_tokens = bool(np.array_equal(np.concatenate(toks), main_cells["tokens_flat"]))

    part_dir = OUT / "cells"; part_dir.mkdir(exist_ok=True)
    todo = [ci for ci in range(N_CELLS) if not (part_dir / f"cell_{ci:03d}.npy").exists()]
    if todo:
        H.check_memory(3.0)
        xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=torch.float32)
        model = xt.model
        layers_needed = sorted({e["src_layer"] for e in edges} | {e["tgt_layer"] for e in edges})
        saes = {l: load_cpu_sae(l) for l in layers_needed}
        n_done = 0
        for ci in todo:
            if time.time() - t0 > args.max_minutes * 60 - 25:
                break
            H.check_memory(3.0)
            ids = torch.from_numpy(np.asarray(toks[ci])[None, :]).to(DEVICE)
            with torch.no_grad():
                hs = [h[0].cpu() for h in model(ids, output_hidden_states=True, use_cache=False).hidden_states]
            vals = []
            for e in edges:
                s, f, t, j = int(e["src_layer"]), int(e["src_feature"]), int(e["tgt_layer"]), int(e["tgt_feature"])
                sae_s, mu_s = saes[s]; sae_t, mu_t = saes[t]
                with torch.no_grad():
                    z = sae_s.encode(hs[s], mu_s); za = z.clone(); za[:, f] = 0.0
                    delta = (sae_s.decode(za, mu_s) - sae_s.decode(z, mu_s)).to(DEVICE).unsqueeze(0)
                    mod = model.model.embed_tokens if s == 0 else model.model.layers[s - 1]
                    hk = mod.register_forward_hook(lambda m, i, o, _d=delta: o + _d)
                    try:
                        hs_a = model(ids, output_hidden_states=True, use_cache=False).hidden_states[t][0].cpu()
                    finally:
                        hk.remove()
                    zc = sae_t.encode(hs[t], mu_t)[:, j]
                    zab = sae_t.encode(hs_a, mu_t)[:, j]
                vals.append(float((zab - zc).double().mean()))
            np.save(part_dir / f"cell_{ci:03d}.npy", np.asarray(vals))
            n_done += 1
            if DEVICE == "mps":
                torch.mps.empty_cache()
        print(f"{n_done} cells this chunk; {N_CELLS - len(todo) + n_done}/{N_CELLS} done ({time.time() - t0:.0f}s)")
        prog = OUT / "progress.json"
        p = json.loads(prog.read_text()) if prog.exists() else {"chunks": []}
        p["chunks"].append(dict(start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), cells=n_done,
                                wall_seconds=round(time.time() - t0, 1)))
        H.write_json(prog, p)
        if len(list(part_dir.glob("cell_*.npy"))) < N_CELLS:
            return
    # compare
    V = np.stack([np.load(part_dir / f"cell_{ci:03d}.npy") for ci in range(N_CELLS)])   # (200, 3)
    mc = pd.read_csv(V2 / "combos_main.csv", index_col="row")
    trace_files = sorted((V2 / "trace_cells").glob("cell_*.npz"))
    out = []
    for k, e in enumerate(edges):
        x = V[:, k]
        d = x.mean() / np.sqrt(x.var(ddof=1) + 1e-12)
        pos = int((x > 0).sum()); cons = max(pos, N_CELLS - pos) / N_CELLS
        row = int(mc[(mc.src_layer == e["src_layer"]) & (mc.src_feature == e["src_feature"]) & (mc.tgt_layer == e["tgt_layer"])].index[0])
        main_vals = np.array([np.load(p, mmap_mode="r")["main"][row, int(e["tgt_feature"])] for p in trace_files], np.float64)
        out.append(dict(edge=e, d_recomputed=float(d), consistency_recomputed=cons,
                        sign_recomputed="inhibitory" if x.mean() < 0 else "excitatory",
                        d_main=float(e["cohens_d"]), consistency_main=float(e["consistency"]),
                        abs_diff_d=float(abs(d - e["cohens_d"])),
                        max_abs_diff_per_cell_dz=float(np.abs(main_vals - x).max()),
                        max_abs_per_cell_dz=float(np.abs(x).max()),
                        passes_edge_rule_recomputed=bool(abs(d) > 0.5 and cons > 0.7)))
    res = dict(same_cells_as_main_run=same_rows, same_tokens_as_main_run=same_tokens, edges=out,
               pass_=bool(same_rows and same_tokens and all(o["passes_edge_rule_recomputed"] and
                                                           o["sign_recomputed"] == o["edge"]["sign"] and
                                                           o["abs_diff_d"] < 0.01 for o in out)),
               tolerance="|d recomputed - d main| < 0.01, same sign, recomputed edge passes |d|>0.5 & consistency>0.7",
               env=H.env_info(DEVICE), script_sha256=H.sha256_file(__file__),
               seeds=dict(cells=SEED, edge_pick=PICK_SEED), dataset_rows=[int(r) for r in rows])
    H.write_json(OUT / "spotcheck_result.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("env", "dataset_rows")}, indent=1, default=str))


if __name__ == "__main__":
    main()
