"""Unit tests for setup/hooks_v2.py on MaxToki-217M (MPS, float32, real K562 cells).

Run:  .venv/bin/python setup/test_hooks_v2.py [--n-cells 3] [--device mps] [--max-minutes 8]
Writes per-cell results + a summary to
  runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/
and exits with status 1 if any check fails. Resumable per cell.

Checks
  T0  layer convention: pre-hook input of block l == hidden_states[l] (l=0..10);
      lm_head input == hidden_states[11] == norm(output of block 10);
      hidden_states[11] != output of block 10.
  T1  (i) zero-delta edit at every layer 0..11 gives logits equal to clean
      (max abs diff < 1e-4); Steer(alpha=1) likewise.
  T2  (ii) ablating an active feature at layer l leaves layers < l and the
      pre-edit input at l unchanged, and changes every layer l+1..11 and the logits.
  T3  (iii) stacking: AB != B, ABC != C, BC != C at the L11 read-out and logits;
      contrast: the deployed (legacy) overwrite hooks give AB == B and ABC == C.
  T4  (iv) ablating a never-active feature changes nothing (< 1e-5).
  T5  (v) L11 path applies no norm after the edit: logit change == lm_head(delta)
      (exact linearity); contrast: legacy L11 zero-edit changes logits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SETUP = Path(__file__).resolve().parent
sys.path.insert(0, str(SETUP))
import hooks_v2 as H  # noqa: E402

OUT = SETUP.parent / "runs/exhaustive-mapping-217M/outputs/v2_hooks_tests"
CELL_SEED = 42
VEC_SEED = 0
TOL_ZERO = 1e-4       # (i)
TOL_DEAD = 1e-5       # (iv)
TOL_SAME = 1e-6       # upstream layers must be bit-identical in practice
MIN_CHANGE = 1e-4     # a "change" must exceed this max-abs difference
ABLATE_LAYERS = [0, 3, 5, 9, 10, 11]
STEER_FEATS = {0: 40, 3: 4071, 6: 4138, 9: 1014, 11: 3924}  # deployed experiment3 features
DEAD_TEST_LAYERS = [0, 5, 11]


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def meanabs(a, b):
    return float((a.float() - b.float()).abs().mean().item())


@torch.no_grad()
def forward(model, ids, saes=None, edits=(), capture=(), hidden_states=False, extra_hooks=()):
    handles = [m.register_forward_hook(fn) for m, fn in extra_hooks]
    try:
        with H.ResidualEditor(model, saes, edits=edits, capture=capture) as ed:
            out = model(ids, use_cache=False, return_dict=True, output_hidden_states=hidden_states)
    finally:
        for h in handles:
            h.remove()
    logits = out.logits[0].detach().float().cpu()
    hs = None
    if hidden_states:
        hs = [h[0].detach().float().cpu() for h in out.hidden_states]
    del out
    H.free_device_cache()
    return logits, ed, hs


@torch.no_grad()
def legacy_forward(model, ids, clean_hs, saes_cpu, spec, readout_hidden=True):
    """Deployed hook logic (experiment2_rerun.py:213-236, experiment3_rerun.py:258-282).
    spec = list of (layer, feature, mode, value): mode 'ablate' or 'scale' (value = alpha),
    or 'zero' (delta forced to 0). Hook sits on layers[min(l, n_blocks-1)] and REPLACES
    that block's output with clean hidden_states[l] + delta."""
    nb = H.n_blocks(model)
    dev = ids.device
    handles = []
    for (l, f, mode, val) in spec:
        h_src = clean_hs[l]
        if mode == "zero":
            delta = torch.zeros_like(h_src)
        else:
            s = saes_cpu[l]
            z = s.sae.encode(h_src, s.mu)
            z2 = z.clone()
            if mode == "ablate":
                z2[:, f] = 0.0
            else:
                z2[:, f] = z[:, f] * val
            delta = s.sae.decode(z2, s.mu) - s.sae.decode(z, s.mu)
        patched = (h_src + delta).to(dev).unsqueeze(0)

        def _hook(module, inp, output, _p=patched):
            return _p.to(output.device)
        handles.append(model.model.layers[min(l, nb - 1)].register_forward_hook(_hook))
    try:
        out = model(ids, use_cache=False, return_dict=True, output_hidden_states=readout_hidden)
    finally:
        for h in handles:
            h.remove()
    logits = out.logits[0].float().cpu()
    hs11 = out.hidden_states[nb][0].float().cpu() if readout_hidden else None
    del out
    H.free_device_cache()
    return logits, hs11


def top_active_feature(codes: torch.Tensor, exclude=()) -> int:
    tot = codes.sum(0)
    for f in exclude:
        tot[f] = -1
    return int(tot.argmax().item())


@torch.no_grad()
def run_cell(model, cell, cell_row, saes, saes_cpu, device):
    nb = H.n_blocks(model)
    L = list(range(nb + 1))  # 0..11
    ids = torch.from_numpy(cell.token_ids[None, :]).to(device)
    res = {"dataset_row": cell_row, "seq_len": int(len(cell.token_ids)), "checks": {}}
    C = res["checks"]

    # ---------------- T0: convention ----------------
    out10 = {}

    def grab10(m, a, o):
        out10["x"] = o.detach().float().cpu()[0]
    clean_logits, ed0, hs = forward(model, ids, saes, capture=L, hidden_states=True,
                                    extra_hooks=[(model.model.layers[nb - 1], grab10)])
    clean_pre = {l: ed0.hidden(l, "pre") for l in L}
    conv = {f"pre{l}_vs_hs{l}": maxabs(clean_pre[l], hs[l]) for l in L}
    normed10 = model.model.norm(out10["x"].to(device)[None]).float().cpu()[0]
    conv["hs11_vs_norm(out_block10)"] = maxabs(hs[nb], normed10)
    conv["hs11_vs_out_block10_maxabs"] = maxabs(hs[nb], out10["x"])
    conv["rms_out_block10"] = float(out10["x"].pow(2).mean().sqrt())
    conv["rms_hs11"] = float(hs[nb].pow(2).mean().sqrt())
    conv["pass"] = (max(conv[f"pre{l}_vs_hs{l}"] for l in L) <= TOL_SAME
                    and conv["hs11_vs_norm(out_block10)"] <= 1e-5
                    and conv["hs11_vs_out_block10_maxabs"] > 0.1)
    C["T0_convention"] = conv

    # noise floor: second clean pass without any hook
    clean2, _, _ = forward(model, ids)
    noise = maxabs(clean_logits, clean2)
    C["noise_floor_clean_vs_clean_logits_maxabs"] = noise

    # clean SAE codes per layer (on the pre-edit inputs)
    clean_codes = {l: saes[l].encode(clean_pre[l].to(device)).cpu() for l in L}

    # ---------------- T1: zero-delta ----------------
    t1 = {"zero_delta_maxabs": {}, "steer_alpha1_maxabs": {}}
    for l in L:
        lg, _, _ = forward(model, ids, saes, edits=[H.ZeroDelta(l)])
        t1["zero_delta_maxabs"][l] = maxabs(lg, clean_logits)
    for l, f in STEER_FEATS.items():
        fa = top_active_feature(clean_codes[l])
        for tag, ff in (("deployed_f%d" % f, f), ("active_f%d" % fa, fa)):
            lg, ed, _ = forward(model, ids, saes, edits=[H.Steer(l, ff, 1.0)])
            t1["steer_alpha1_maxabs"][f"L{l}_{tag}"] = maxabs(lg, clean_logits)
    t1["pass"] = (max(t1["zero_delta_maxabs"].values()) < TOL_ZERO
                  and max(t1["steer_alpha1_maxabs"].values()) < TOL_ZERO)
    C["T1_zero_delta"] = t1

    # ---------------- T2: ablation locality ----------------
    t2 = {}
    ok2 = True
    for l in ABLATE_LAYERS:
        f = top_active_feature(clean_codes[l])
        n_act = int((clean_codes[l][:, f] > 0).sum())
        lg, ed, _ = forward(model, ids, saes, edits=[H.Ablate(l, [f])], capture=L)
        up = {m: maxabs(ed.hidden(m, "post" if m < l else "pre"), clean_pre[m]) for m in L if m <= l}
        down = {m: maxabs(ed.hidden(m, "pre"), clean_pre[m]) for m in L if m > l}
        down_codes = {}
        for m in L:
            if m > l:
                zc = ed.codes(m, "pre")
                d = (zc - clean_codes[m]).abs()
                down_codes[m] = {"mean_abs_dz": float(d.mean()),
                                 "n_features_changed": int((d.max(0).values > 1e-6).sum())}
        # the edit itself: post - pre at l equals -z_f d_f
        delta_obs = ed.hidden(l, "post") - ed.hidden(l, "pre")
        delta_exp = -(clean_codes[l][:, f:f + 1] * saes_cpu[l].W_dec[:, f][None, :])
        z_after = saes[l].encode(ed.hidden(l, "post").to(device)).cpu()[:, f]
        logit_change = maxabs(lg, clean_logits)
        entry = {
            "feature": f, "n_active_positions": n_act,
            "upstream_and_same_layer_pre_maxabs": up,
            "downstream_maxabs": down,
            "downstream_sae_codes": down_codes,
            "edit_delta_vs_expected_maxabs": maxabs(delta_obs, delta_exp),
            "edit_delta_rms": float(delta_obs.pow(2).mean().sqrt()),
            "z_f_before_sum": float(clean_codes[l][:, f].sum()),
            "z_f_after_reencode_sum": float(z_after.sum()),
            "logits_maxabs_change": logit_change,
        }
        entry["pass"] = (n_act > 0
                         and max(up.values()) <= TOL_SAME
                         and all(v > MIN_CHANGE for v in down.values())
                         and all(v["n_features_changed"] > 0 for v in down_codes.values())
                         and entry["edit_delta_vs_expected_maxabs"] < 1e-4
                         and logit_change > MIN_CHANGE)
        ok2 &= entry["pass"]
        t2[f"L{l}"] = entry
    t2["pass"] = ok2
    C["T2_locality"] = t2

    # ---------------- T3: stacking ----------------
    fA = top_active_feature(clean_codes[0])
    fB = top_active_feature(clean_codes[5])
    fC = top_active_feature(clean_codes[9])
    E = {"A": H.Ablate(0, [fA]), "B": H.Ablate(5, [fB]), "C": H.Ablate(9, [fC])}
    runs = {}
    for name in ["A", "B", "C", "AB", "BC", "ABC"]:
        lg, ed, _ = forward(model, ids, saes, edits=[E[k] for k in name], capture=[5, 9, 11])
        runs[name] = {"logits": lg, "h11": ed.hidden(11, "pre"),
                      "delta5": (ed.hidden(5, "post") - ed.hidden(5, "pre")) if "B" in name else None,
                      "delta9": (ed.hidden(9, "post") - ed.hidden(9, "pre")) if "C" in name else None}
    t3 = {"features": {"A_L0": fA, "B_L5": fB, "C_L9": fC}}
    for x, y in [("AB", "B"), ("ABC", "C"), ("BC", "C"), ("ABC", "BC"), ("A", None)]:
        ref = runs[y] if y else {"logits": clean_logits, "h11": clean_pre[11]}
        key = f"{x}_vs_{y or 'clean'}"
        t3[key] = {"h11_maxabs": maxabs(runs[x]["h11"], ref["h11"]),
                   "h11_meanabs": meanabs(runs[x]["h11"], ref["h11"]),
                   "logits_maxabs": maxabs(runs[x]["logits"], ref["logits"])}
    # B's delta is computed from the live input, so it differs between B and AB
    t3["deltaB_in_AB_vs_B_maxabs"] = maxabs(runs["AB"]["delta5"], runs["B"]["delta5"])
    t3["deltaC_in_ABC_vs_C_maxabs"] = maxabs(runs["ABC"]["delta9"], runs["C"]["delta9"])
    # legacy contrast
    clean_hs_cpu = {l: hs[l] for l in (0, 5, 9)}
    leg = {}
    for name in ["B", "AB", "C", "ABC"]:
        spec = [({"A": 0, "B": 5, "C": 9}[k], {"A": fA, "B": fB, "C": fC}[k], "ablate", None) for k in name]
        leg[name] = legacy_forward(model, ids, clean_hs_cpu, saes_cpu, spec)
    t3["legacy_AB_vs_B"] = {"h11_maxabs": maxabs(leg["AB"][1], leg["B"][1]),
                            "logits_maxabs": maxabs(leg["AB"][0], leg["B"][0])}
    t3["legacy_ABC_vs_C"] = {"h11_maxabs": maxabs(leg["ABC"][1], leg["C"][1]),
                             "logits_maxabs": maxabs(leg["ABC"][0], leg["C"][0])}
    t3["pass"] = (t3["A_vs_clean"]["h11_maxabs"] > MIN_CHANGE
                  and all(t3[k]["h11_maxabs"] > MIN_CHANGE and t3[k]["logits_maxabs"] > MIN_CHANGE
                          for k in ["AB_vs_B", "ABC_vs_C", "BC_vs_C", "ABC_vs_BC"]))
    t3["legacy_reproduces_overwrite_bug"] = (t3["legacy_AB_vs_B"]["h11_maxabs"] <= TOL_SAME
                                             and t3["legacy_ABC_vs_C"]["h11_maxabs"] <= TOL_SAME)
    C["T3_stacking"] = t3
    del runs, leg

    # ---------------- T4: never-active feature ----------------
    t4 = {}
    ok4 = True
    for l in DEAD_TEST_LAYERS:
        never = (clean_codes[l] > 0).sum(0) == 0
        dead_list = res.setdefault("_dead_candidates", {})
        # choose the lowest-index feature that is inactive at every position of this cell
        f = int(torch.nonzero(never)[0].item())
        lg, ed, _ = forward(model, ids, saes, edits=[H.Ablate(l, [f])], capture=[l])
        dmax = maxabs(ed.hidden(l, "post"), ed.hidden(l, "pre"))
        ch = maxabs(lg, clean_logits)
        t4[f"L{l}"] = {"feature": f, "n_never_active_in_cell": int(never.sum()),
                       "edit_delta_maxabs": dmax, "logits_maxabs_change": ch,
                       "pass": ch < TOL_DEAD and dmax == 0.0}
        ok4 &= t4[f"L{l}"]["pass"]
    res.pop("_dead_candidates", None)
    t4["pass"] = ok4
    C["T4_never_active"] = t4

    # ---------------- T5: L11 path ----------------
    t5 = {}
    W_lm = model.lm_head.weight.detach().float().cpu()  # (V, d)
    g = torch.Generator().manual_seed(VEC_SEED)
    v = torch.randn(hs[nb].shape[1], generator=g)
    v = v / v.norm() * float(hs[nb].norm(dim=1).mean()) * 0.05
    lg, ed, _ = forward(model, ids, saes, edits=[H.AddVector(nb, vector=v)])
    exp = (v @ W_lm.T)[None, :]  # (1, V) broadcast over positions
    t5["addvector_logit_change_vs_lm_head(v)_maxabs"] = maxabs(lg - clean_logits, exp.expand_as(lg))
    t5["addvector_logit_change_scale_maxabs"] = float(exp.abs().max())
    f11 = top_active_feature(clean_codes[nb])
    lg, ed, _ = forward(model, ids, saes, edits=[H.Ablate(nb, [f11])], capture=[nb])
    d = ed.hidden(nb, "post") - ed.hidden(nb, "pre")
    t5["ablate_L11_logit_change_vs_lm_head(delta)_maxabs"] = maxabs(lg - clean_logits, d @ W_lm.T)
    t5["ablate_L11_logit_change_scale_maxabs"] = float((d @ W_lm.T).abs().max())
    t5["ablate_L11_feature"] = f11
    # legacy zero-edit contrasts (deployed experiment3 hook, delta forced to 0)
    for l in (0, 5, 11):
        lg_leg, _ = legacy_forward(model, ids, {l: hs[l]}, saes_cpu, [(l, 0, "zero", None)],
                                   readout_hidden=False)
        t5[f"legacy_zero_edit_L{l}_logits_maxabs"] = maxabs(lg_leg, clean_logits)
        t5[f"legacy_zero_edit_L{l}_logits_meanabs"] = meanabs(lg_leg, clean_logits)
    t5["pass"] = (t5["addvector_logit_change_vs_lm_head(v)_maxabs"] < 1e-3
                  and t5["ablate_L11_logit_change_vs_lm_head(delta)_maxabs"] < 1e-3
                  and t1["zero_delta_maxabs"][nb] < TOL_ZERO
                  and conv[f"pre{nb}_vs_hs{nb}"] <= TOL_SAME)
    C["T5_L11_no_double_norm"] = t5

    res["all_pass"] = all(C[k]["pass"] for k in C if isinstance(C[k], dict) and "pass" in C[k])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(VEC_SEED)

    cells, rows, h5 = H.load_k562_control_cells(args.n_cells, pool=100, seed=CELL_SEED)
    print(f"cells: rows={rows} lens={[len(c.token_ids) for c in cells]}")
    gb = H.check_memory(args.min_free_gb)
    print(f"free memory before model load: {gb:.2f} GB")
    xt = H.load_model(args.device)
    model = xt.model
    saes = H.load_saes(range(12), device=args.device)
    saes_cpu = H.load_saes(range(12), device="cpu")

    cfg_path = OUT / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"chunks": []}
    cfg.update(H.env_info(args.device))
    cfg.update({
        "script": str(Path(__file__).resolve()),
        "script_sha256": H.sha256_file(__file__),
        "seeds": {"cell_selection": CELL_SEED, "random_vector": VEC_SEED},
        "cells": {"dataset": h5, "dataset_rows": rows,
                  "selection": "K562 non-targeting controls; rng(42).choice(pool=100) sorted; first n that tokenize (same pool as experiment2_rerun.py)"},
        "sae_dir": str(H.SAE_DIR),
        "features": {"steer_alpha1_deployed": STEER_FEATS,
                     "others": "chosen per cell as the top total-activation feature (see per-cell JSON)"},
        "tolerances": {"zero": TOL_ZERO, "dead": TOL_DEAD, "same": TOL_SAME, "min_change": MIN_CHANGE},
    })
    chunk = {"start": time.strftime("%H:%M:%S"), "cells_done": []}

    results = []
    for ci, (cell, row) in enumerate(zip(cells, rows)):
        p = OUT / f"cell_{ci}_row{row}.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
            continue
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("time budget reached; re-run to continue")
            break
        gb = H.check_memory(args.min_free_gb)
        t0 = time.time()
        r = run_cell(model, cell, row, saes, saes_cpu, args.device)
        r["wall_seconds"] = round(time.time() - t0, 1)
        r["free_gb_before"] = round(gb, 2)
        H.write_json(p, r)
        results.append(r)
        chunk["cells_done"].append(row)
        H.free_device_cache()
        print(f"cell {ci} row {row}: all_pass={r['all_pass']}  ({r['wall_seconds']} s)")

    chunk["wall_seconds"] = round(time.time() - t_start, 1)
    cfg["chunks"].append(chunk)
    H.write_json(cfg_path, cfg)

    if len(results) < len(cells):
        sys.exit(2)
    summary = summarize(results)
    H.write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["all_pass"] else 1)


def summarize(results):
    s = {"n_cells": len(results), "rows": [r["dataset_row"] for r in results]}
    def mx(fn):
        return max(fn(r) for r in results)
    C = lambda r: r["checks"]
    s["T0_max_pre_vs_hidden_states"] = mx(lambda r: max(v for k, v in C(r)["T0_convention"].items() if k.startswith("pre")))
    s["T0_hs11_vs_norm_out10"] = mx(lambda r: C(r)["T0_convention"]["hs11_vs_norm(out_block10)"])
    s["T0_hs11_vs_out10_min"] = min(C(r)["T0_convention"]["hs11_vs_out_block10_maxabs"] for r in results)
    s["noise_floor_logits"] = mx(lambda r: C(r)["noise_floor_clean_vs_clean_logits_maxabs"])
    s["T1_zero_delta_max"] = mx(lambda r: max(C(r)["T1_zero_delta"]["zero_delta_maxabs"].values()))
    s["T1_steer_alpha1_max"] = mx(lambda r: max(C(r)["T1_zero_delta"]["steer_alpha1_maxabs"].values()))
    s["T2_upstream_max"] = mx(lambda r: max(max(C(r)["T2_locality"][f"L{l}"]["upstream_and_same_layer_pre_maxabs"].values()) for l in ABLATE_LAYERS))
    s["T2_downstream_min"] = min(min(C(r)["T2_locality"][f"L{l}"]["downstream_maxabs"].values())
                                 for r in results for l in ABLATE_LAYERS if l < 11)
    s["T3_AB_vs_B_h11_min"] = min(C(r)["T3_stacking"]["AB_vs_B"]["h11_maxabs"] for r in results)
    s["T3_ABC_vs_C_h11_min"] = min(C(r)["T3_stacking"]["ABC_vs_C"]["h11_maxabs"] for r in results)
    s["T3_legacy_AB_vs_B_h11_max"] = mx(lambda r: C(r)["T3_stacking"]["legacy_AB_vs_B"]["h11_maxabs"])
    s["T3_legacy_ABC_vs_C_h11_max"] = mx(lambda r: C(r)["T3_stacking"]["legacy_ABC_vs_C"]["h11_maxabs"])
    s["T4_never_active_max"] = mx(lambda r: max(C(r)["T4_never_active"][f"L{l}"]["logits_maxabs_change"] for l in DEAD_TEST_LAYERS))
    s["T5_linearity_max"] = mx(lambda r: max(C(r)["T5_L11_no_double_norm"]["addvector_logit_change_vs_lm_head(v)_maxabs"],
                                              C(r)["T5_L11_no_double_norm"]["ablate_L11_logit_change_vs_lm_head(delta)_maxabs"]))
    s["T5_legacy_zero_edit_L11_meanabs"] = [C(r)["T5_L11_no_double_norm"]["legacy_zero_edit_L11_logits_meanabs"] for r in results]
    s["per_check_pass"] = {k: all(C(r)[k]["pass"] for r in results)
                           for k in ["T0_convention", "T1_zero_delta", "T2_locality", "T3_stacking",
                                     "T4_never_active", "T5_L11_no_double_norm"]}
    s["T3_legacy_reproduces_overwrite_bug"] = all(C(r)["T3_stacking"]["legacy_reproduces_overwrite_bug"] for r in results)
    s["all_pass"] = all(s["per_check_pass"].values())
    return s


if __name__ == "__main__":
    main()
