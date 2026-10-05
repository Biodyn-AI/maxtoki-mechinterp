"""Tests for setup/inputs_v3.py (correct MaxToki input encoding).

Run:
  OMP_NUM_THREADS=4 .venv/bin/python setup/test_inputs_v3.py --part cpu
  OMP_NUM_THREADS=4 .venv/bin/python setup/test_inputs_v3.py --part model --device mps --max-minutes 8
  OMP_NUM_THREADS=4 .venv/bin/python setup/test_inputs_v3.py --part summary
Writes to checks/v3_inputs/ and exits 1 if any check fails. The model part is resumable
(one JSON per group in checks/v3_inputs/model/); re-run it until every group is done.

Cells: for every dataset x context used by a MaxToki run, the 5 "primary" cells of the audit
(checks/input_encoding_audit/order_metrics_cells.csv, primary == True) plus the audit's other
cells (up to 50) as an extended set. Same rows, so per-cell numbers can be compared 1:1.

CPU checks (per cell)
  C1  count test: the v3 source is integer counts (RPE1 X, TS/Krasnow raw/X) or whole-number
      multiples of the smallest value (K562/Adamson expm1(X)).
  C2  negative control: the stored X row (log1p) FAILS the count test (except RPE1, where X is counts).
  C3  v3 tokens pass check_encoding (order == counts/median order, length, count-like).
  C4  agreement with tokenize_cell on raw counts: tokens from tokenize_counts (counts/total*1e4)
      equal tokenize_cell(raw integer counts) (K562/Adamson: the integer counts rebuilt as
      round(expm1(X)/unit)); exact, or differing only inside ties (1e-6 relative).
      C4b (K562/Adamson): also tie-equivalent to tokenize_cell(expm1(X)) (float, no rounding).
  C5  negative control: the deployed tokens (tokenize_cell on X) FAIL check_encoding against
      the v3 counts (RPE1: they pass, because RPE1 was already correct).
  C6  the deployed-path function reproduces the tokens saved by the runs (where saved).
  C7  agreement v3 vs deployed (Spearman, top-200, ...) equals the audit's per-cell numbers.
Loader checks
  L1  inputs_v3.load_k562_control_cells returns the same rows as hooks_v2.load_k562_control_cells
      for the three calls used by v2 scripts, and the 200-cell call matches the rows saved by
      v2_circuit_trace; the old loader's tokens equal the saved tokens; new tokens pass the check.
  L2  TS kidney (not used by a MaxToki run): 5 random rows load and pass C1/C3.
  L3  TS decontX alternative: tokens pass the check; order agreement with raw/X reported.
Model check (MPS; only correctly encoded inputs are fed, each batch checked first)
  M1  forward pass on the 5 primary v3 cells per group: finite logits, next-gene NLL and top-1
      accuracy reported as a baseline (by assay); hooks_v2 ZeroDelta edits at all 12 sites leave
      the logits unchanged on a v3 K562 cell.
Summary: means, SD, min/max and bootstrap 95% CIs (cells resampled) -> summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

SETUP = Path(__file__).resolve().parent
PROJ = SETUP.parent
sys.path.insert(0, str(SETUP))
import inputs_v3 as I  # noqa: E402

RUNS = PROJ / "runs"
AUDIT = PROJ / "checks/input_encoding_audit"
OUT = PROJ / "checks/v3_inputs"
SEED = 20261001
# Per-cell agreement numbers must equal the audit's (same rows, same measures). The audit built
# its reference with tokenize_cell on float32 values; v3 uses float64 integer counts. The two can
# swap genes whose counts/median differ by < ~1e-6 (float32 rounding), so allow tiny differences.
TOL_AUDIT_RHO = 1e-5     # Spearman
TOL_AUDIT_SHARE = 1e-3   # overlaps and same-position share (= 2 of 2,046 positions; 0 of 200)

GROUPS = [
    # audit label, dataset, max_len, audit reference
    ("K562_ctrl_replogle_concat", "k562", 2048, "expm1(X)=CP10k"),
    ("K562_perturbed_replogle_concat", "k562", 2048, "expm1(X)=CP10k"),
    ("Adamson_ctrl", "adamson", 2048, "expm1(X)=CP10k"),
    ("RPE1_ctrl", "rpe1", 2048, "X=raw counts"),
    ("TS_immune_L2048", "ts_immune", 2048, "raw/X counts"),
    ("TS_immune_L4096_manifold", "ts_immune", 4096, "raw/X counts"),
    ("TS_lung_L4096_manifold", "ts_lung", 4096, "raw/X counts"),
    ("TS_lung_L2048_topology", "ts_lung", 2048, "raw/X counts"),
    ("TS_immune_sub20k_L1024_longevity", "ts_immune_sub20k", 1024, "raw/X counts"),
    ("Krasnow_lung_SS2_L2048_topology", "krasnow", 2048, "raw/X counts"),
]
METRICS = ["spearman_full", "spearman_fed", "top200_overlap", "top2046_overlap",
           "kept_set_overlap", "same_position_share"]


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(path)


def saved_tokens(label):
    """{row: saved token ids} for the groups whose runs saved tokens (same files as the audit)."""
    if label == "K562_ctrl_replogle_concat":
        z = np.load(RUNS / "circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz")
        off = np.concatenate([[0], np.cumsum(z["lengths"])])
        return {int(r): z["tokens_flat"][off[i]:off[i + 1]] for i, r in enumerate(z["rows"])}
    if label == "K562_perturbed_replogle_concat":
        out = {}
        for p in sorted(glob.glob(str(RUNS / "sae-atlas-217M/outputs/v2_tf_specificity/cells/catalog_*.npz")))[:3]:
            c = np.load(p)
            for i, r in enumerate(c["row"]):
                out[int(r)] = c["tok"][c["offsets"][i]:c["offsets"][i + 1]]
        return out
    if label == "TS_immune_L2048":
        z = np.load(RUNS / "exhaustive-mapping-217M/outputs/v2_steering/cells.npz", allow_pickle=True)
        return {int(r): np.asarray(t) for r, t in zip(z["ts_rows"], z["token_ids"]) if t is not None}
    if label in ("TS_immune_L4096_manifold", "TS_lung_L4096_manifold"):
        stem = "cells_internal" if "immune" in label else "cells_lung_nonhema"
        obs = pd.read_csv(RUNS / f"manifold-discovery-217M/outputs/phase1/{stem}_obs.csv")
        z = np.load(RUNS / f"manifold-discovery-217M/outputs/phase1/{stem}.npz")
        T, Ls = z["token_ids"], z["seq_lens"]
        idx = {int(c): i for i, c in enumerate(obs["cell_idx"])}
        return {"__lazy__": (T, Ls, idx)}
    return {}


def get_saved(sv, row):
    if not sv:
        return None
    if "__lazy__" in sv:
        T, Ls, idx = sv["__lazy__"]
        if row not in idx:
            return None
        i = idx[row]
        return np.asarray(T[i][: int(Ls[i])], dtype=np.int64)
    t = sv.get(row)
    return None if t is None else np.asarray(t, dtype=np.int64)


def tie_equivalent(a, b, counts, vm):
    """Same tokens, or differ only where the counts/median values are exactly tied."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        return False
    if np.array_equal(a, b):
        return True
    med = vm.medians.astype(np.float64)
    cv = counts[vm.var_indices]
    va = np.array([cv[vm.tok2col[int(t)]] / med[vm.tok2col[int(t)]] if t > 3 else -1 for t in a])
    vb = np.array([cv[vm.tok2col[int(t)]] / med[vm.tok2col[int(t)]] if t > 3 else -1 for t in b])
    return bool(np.allclose(va, vb, rtol=1e-6, atol=0))


# =============================================================================
# CPU part
# =============================================================================
def part_cpu():
    t0 = time.time()
    tok = I.get_tokenizer()
    audit = pd.read_csv(AUDIT / "order_metrics_cells.csv")
    rows_out, fails = [], []
    uni_rows = {}
    for label, ds, L, ref in GROUPS:
        a = audit[(audit.dataset == label) & (audit.reference == ref)].copy()
        a = a.sort_values(["primary", "row"], ascending=[False, True])
        vm = I.get_var_map(ds)
        sv = saved_tokens(label)
        umi = None
        if ds == "k562":
            import h5py
            with h5py.File(I.DATASETS["k562"]["path"], "r") as f:
                umi = {int(r): float(f["obs"]["UMI_count"][int(r)]) for r in a.row}
        with I.CountReader(ds) as rd:
            for _, ar in a.iterrows():
                r = int(ar.row)
                d = {"group": label, "dataset": ds, "row": r, "primary": bool(ar.primary), "max_len": L}
                counts, chk = rd.counts(r, return_check=True)            # raises if not count-like
                x = rd.stored_X(r)
                uni_rows.setdefault(ds, set()).add(r)
                # C1
                d["C1_count_kind"] = chk["kind"]
                d["C1_pass"] = bool(chk["expected_ok"])
                d["C1_max_dev"] = chk["max_dev"]
                d["C1_q_max"] = chk["q_max"]
                if chk["kind"] == "integer_multiple":
                    d["implied_total"] = chk["implied_total"]
                    d["sum_multiples"] = chk["sum_multiples"]
                    if umi is not None:
                        d["obs_UMI_count"] = umi[r]
                # C2
                xl = I.check_count_like(x)
                d["C2_storedX_count_like"] = bool(xl["pass"])
                d["C2_pass"] = (xl["pass"] if ds == "rpe1" else not xl["pass"])
                # C3
                cell = I.tokenize_counts(counts, vm, L, check=False)
                ce = I.check_encoding(counts, cell, vm, L)
                d["C3_pass"] = bool(ce["pass"])
                d["C3_n_order_violations"] = ce["n_order_violations"]
                d["C3_tie_reordered"] = ce["n_positions_tie_reordered"]
                d["n_tokens_v3"] = len(cell.token_ids)
                # C4: tokenize_cell on raw integer counts
                int_counts = counts            # integer counts (expm1 sources: rebuilt by the loader)
                assert np.array_equal(int_counts, np.round(int_counts))
                raw_cell = tok.tokenize_cell(int_counts, vm.var_indices, vm.token_ids, vm.medians, max_len=L)
                d["C4_identical"] = bool(np.array_equal(raw_cell.token_ids, cell.token_ids))
                d["C4_tie_equivalent"] = tie_equivalent(raw_cell.token_ids, cell.token_ids, int_counts, vm)
                d["C4_pass"] = d["C4_tie_equivalent"]
                if chk["kind"] == "integer_multiple":
                    xe = np.expm1(x)
                    fl_cell = tok.tokenize_cell(xe, vm.var_indices, vm.token_ids, vm.medians, max_len=L)
                    d["C4b_float_identical"] = bool(np.array_equal(fl_cell.token_ids, cell.token_ids))
                    d["C4b_float_tie_equivalent"] = tie_equivalent(fl_cell.token_ids, cell.token_ids, int_counts, vm)
                    d["C4_pass"] = d["C4_pass"] and d["C4b_float_tie_equivalent"]
                # deployed tokens (old path)
                dep_cell = tok.tokenize_cell(x.astype(np.float32), vm.var_indices, vm.token_ids,
                                             vm.medians, max_len=L)
                # C5
                cd = I.check_encoding(counts, dep_cell, vm, L)
                d["C5_deployed_passes_check"] = bool(cd["pass"])
                d["C5_pass"] = (cd["pass"] if ds == "rpe1" else not cd["pass"])
                d["C5_deployed_order_violations"] = cd["n_order_violations"]
                # C6
                s = get_saved(sv, r)
                if s is not None:
                    s2 = s[s != I.PAD] if len(s) > len(dep_cell.token_ids) else s
                    d["C6_saved_identical"] = bool(np.array_equal(s2, dep_cell.token_ids))
                # C7: agreement deployed vs v3 (full orders), compared with the audit
                dep_full = tok.tokenize_cell(x.astype(np.float32), vm.var_indices, vm.token_ids,
                                             vm.medians, max_len=10 ** 7).token_ids[1:-1]
                v3_full = I.full_order(counts, vm)
                ag = I.order_agreement(dep_full, v3_full, L)
                worst, ok7 = 0.0, True
                for m in METRICS:
                    d[m] = ag[m]
                    d["audit_" + m] = float(ar[m])
                    dm = abs(ag[m] - float(ar[m]))
                    worst = max(worst, dm)
                    ok7 = ok7 and dm <= (TOL_AUDIT_RHO if m.startswith("spearman") else TOL_AUDIT_SHARE)
                d["C7_max_abs_diff_vs_audit"] = worst
                d["C7_exact"] = worst <= 1e-12
                d["C7_pass"] = ok7
                d["truncated_v3"] = ag["truncated_b"]
                rows_out.append(d)
        print(f"[cpu] {label}: {len(a)} cells  {time.time() - t0:.0f}s", flush=True)
        del sv
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "cells_cpu.csv", index=False)

    # group summaries
    summ = []
    for label, ds, L, ref in GROUPS:
        g = df[df.group == label]
        for lab, gg in [("primary5", g[g.primary]), ("all", g)]:
            s = {"group": label, "cells": lab, "n": len(gg)}
            for m in METRICS:
                v = gg[m].to_numpy(float)
                s[m + "_mean"] = float(v.mean())
                s[m + "_min"] = float(v.min())
                s[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                s["audit_" + m + "_mean"] = float(gg["audit_" + m].mean())
            for c in ["C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass", "C7_pass"]:
                s[c] = int(gg[c].sum())
            s["C4_identical"] = int(gg["C4_identical"].sum())
            if "C4b_float_identical" in gg and gg["C4b_float_identical"].notna().any():
                s["C4b_float_identical"] = int(gg["C4b_float_identical"].dropna().astype(bool).sum())
            s["C7_exact"] = int(gg["C7_exact"].sum())
            s["C7_max_abs_diff"] = float(gg["C7_max_abs_diff_vs_audit"].max())
            if "C6_saved_identical" in gg and gg["C6_saved_identical"].notna().any():
                s["C6_checked"] = int(gg["C6_saved_identical"].notna().sum())
                s["C6_identical"] = int(gg["C6_saved_identical"].dropna().astype(bool).sum())
            s["C1_max_dev"] = float(gg["C1_max_dev"].max())
            s["C3_tie_reordered_total"] = int(gg["C3_tie_reordered"].sum())
            summ.append(s)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "summary_cpu.csv", index=False)
    for c in ["C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass", "C7_pass"]:
        bad = df[~df[c].astype(bool)]
        if len(bad):
            fails.append(f"{c}: {len(bad)} cells fail ({bad.group.unique().tolist()})")
    if "C6_saved_identical" in df:
        s6 = df["C6_saved_identical"].dropna()
        if not s6.astype(bool).all():
            fails.append(f"C6: {int((~s6.astype(bool)).sum())} saved token sets not reproduced")

    # ---- L1 K562 drop-in loader
    import hooks_v2 as H
    l1 = {"calls": []}
    for (n, pool, seed) in [(5, 100, 42), (20, 100, 42), (200, 200, 42)]:
        new_c, new_r, h5n, info = I.load_k562_control_cells(n, pool=pool, seed=seed, max_len=2048,
                                                           return_info=True)
        old_c, old_r, h5o = H.load_k562_control_cells(n, pool=pool, seed=seed, max_len=2048)
        rec = {"n_cells": n, "pool": pool, "seed": seed, "rows_equal": new_r == old_r, "h5_equal": h5n == h5o,
               "n_returned": len(new_c),
               "n_tokens_identical_old_vs_new": int(sum(np.array_equal(a.token_ids, b.token_ids)
                                                        for a, b in zip(new_c, old_c))),
               "count_unit_max_dev": info["count_unit_max_dev"]}
        # independent re-check of every new cell
        with I.CountReader("k562") as rd:
            n_ok = 0
            for row, c in zip(new_r, new_c):
                n_ok += int(I.check_encoding(rd.counts(row), c, "k562", 2048)["pass"])
        rec["n_new_pass_check"] = n_ok
        if (n, pool, seed) == (200, 200, 42):
            z = np.load(RUNS / "circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz")
            off = np.concatenate([[0], np.cumsum(z["lengths"])])
            saved = [z["tokens_flat"][off[i]:off[i + 1]] for i in range(len(z["lengths"]))]
            rec["rows_equal_v2_circuit_saved"] = [int(r) for r in z["rows"]] == new_r
            rec["old_tokens_equal_saved"] = int(sum(np.array_equal(a.token_ids, s) for a, s in zip(old_c, saved)))
        ok = (rec["rows_equal"] and rec["h5_equal"] and rec["n_returned"] == n and rec["n_new_pass_check"] == n
              and rec["n_tokens_identical_old_vs_new"] == 0)
        if (n, pool, seed) == (200, 200, 42):
            ok = ok and rec["rows_equal_v2_circuit_saved"] and rec["old_tokens_equal_saved"] == n
        rec["pass"] = bool(ok)
        if not ok:
            fails.append(f"L1 failed for {(n, pool, seed)}: {rec}")
        l1["calls"].append(rec)
        print(f"[cpu] L1 {(n, pool, seed)}: {rec}", flush=True)
    l1["k562_dropin_rows_200"] = new_r

    # ---- L2 TS kidney
    import h5py
    with h5py.File(I.DATASETS["ts_kidney"]["path"], "r") as f:
        nk = int(f["X"].attrs["shape"][0])
    krows = sorted(np.random.default_rng(SEED).choice(nk, 5, replace=False).tolist())
    cells_k, kept_k, info_k, counts_k = I.tokenize_rows("ts_kidney", krows, 2048, return_counts=True)
    l2 = {"rows": krows, "kept": kept_k, "info": info_k,
          "check": I.assert_encoding_batch(counts_k, cells_k, "ts_kidney", 2048, label="ts_kidney")}
    l2["pass"] = bool(len(kept_k) == 5 and info_k["count_checks_all_pass"])
    if not l2["pass"]:
        fails.append("L2 TS kidney failed")

    # ---- L3 decontX alternative on the TS primary cells
    l3 = []
    for label, ds, L, ref in GROUPS:
        if not ds.startswith("ts_"):
            continue
        prim = sorted(df[(df.group == label) & df.primary].row.tolist())
        c_raw, k_raw, _, cnt_raw = I.tokenize_rows(ds, prim, L, return_counts=True)
        c_dx, k_dx, info_dx, cnt_dx = I.tokenize_rows(ds, prim, L, source="decontX", return_counts=True)
        vm = I.get_var_map(ds)
        for r, a, b in zip(k_raw, cnt_raw, cnt_dx):
            ag = I.order_agreement(I.full_order(b, vm), I.full_order(a, vm), L)
            l3.append({"group": label, "row": r, "decontX_vs_raw_spearman_full": ag["spearman_full"],
                       "decontX_vs_raw_top200": ag["top200_overlap"],
                       "decontX_vs_raw_kept_set": ag["kept_set_overlap"]})
    l3df = pd.DataFrame(l3)
    l3df.to_csv(OUT / "decontx_vs_raw.csv", index=False)

    # ---- provenance of the rows used
    prov = {}
    for ds, rs in uni_rows.items():
        rs = sorted(rs)
        cnt, _ = I.read_counts(ds, rs)
        prov[ds] = I.encoding_record(ds, rows=rs, counts_sha256=I.array_sha256(cnt))
        prov[ds]["n_vocab_genes"] = int(len(I.get_var_map(ds).var_indices))
        prov[ds]["n_vars"] = int(I.get_var_map(ds).n_vars)
    res = {"fails": fails, "wall_s": time.time() - t0, "L1": l1, "L2": l2,
           "L3_summary": {g: {c + "_" + st: float(getattr(gg[c], st)()) for c in
                              ["decontX_vs_raw_spearman_full", "decontX_vs_raw_top200", "decontX_vs_raw_kept_set"]
                              for st in ("mean", "min")} for g, gg in l3df.groupby("group")},
           "provenance": prov}
    jdump(OUT / "cpu_results.json", res)
    print(sdf[["group", "cells", "n", "spearman_full_mean", "audit_spearman_full_mean", "top200_overlap_mean",
               "audit_top200_overlap_mean", "C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass", "C7_pass"]]
          .to_string(), flush=True)
    print("[cpu] fails:", fails, flush=True)
    return fails, time.time() - t0


# =============================================================================
# Model part (correctly encoded inputs only)
# =============================================================================
def part_model(device, max_minutes):
    import torch
    import hooks_v2 as H
    torch.manual_seed(SEED)
    t0 = time.time()
    mdir = OUT / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    gb = H.check_memory(3.0)
    print(f"[model] free memory {gb:.1f} GB", flush=True)
    ext = H.load_model(device=device)
    model = ext.model
    df = pd.read_csv(OUT / "cells_cpu.csv")
    chunks = []
    fails = []
    for label, ds, L, ref in GROUPS:
        path = mdir / f"{label}.json"
        if path.exists():
            continue
        if (time.time() - t0) / 60 > max_minutes:
            print("[model] time budget reached; re-run to continue", flush=True)
            break
        gb = H.check_memory(3.0)
        tc = time.time()
        prim = sorted(df[(df.group == label) & df.primary].row.tolist())
        cells, kept, info, counts = I.tokenize_rows(ds, prim, L, return_counts=True)
        chk = I.assert_encoding_batch(counts, cells, ds, L, label=label)   # rule: check before forward
        if ds.startswith("ts_") or ds == "krasnow":
            assay = dict(zip(kept, I.read_obs_column(ds, "assay", kept)))
        else:
            assay = {r: "Perturb-seq 10x (UMI)" for r in kept}
        recs = []
        for r, c in zip(kept, cells):
            ids = torch.from_numpy(c.token_ids[None, :].astype(np.int64)).to(device)
            with torch.no_grad():
                out = model(ids, use_cache=False, return_dict=True)
                lg = out.logits[0].float()
                finite = bool(torch.isfinite(lg).all().item())
                tgt = ids[0, 1:-1]
                lp = torch.log_softmax(lg[:-2], dim=-1)
                nll = -lp.gather(1, tgt[:, None])[:, 0]
                top1 = (lg[:-2].argmax(-1) == tgt).float()
                rec = {"row": r, "assay": str(assay[r]), "n_tokens": int(ids.shape[1]), "finite": finite,
                       "nll_mean": float(nll.mean().item()), "nll_first200": float(nll[:200].mean().item()),
                       "top1": float(top1.mean().item()), "top1_first200": float(top1[:200].mean().item())}
            del out, lg, lp
            recs.append(rec)
            H.free_device_cache()
        zd = None
        if label == "K562_ctrl_replogle_concat":
            ids = torch.from_numpy(cells[0].token_ids[None, :].astype(np.int64))
            clean, _ = H.run_with_edits(model, ids)
            edited, _ = H.run_with_edits(model, ids, edits=[H.ZeroDelta(l) for l in range(12)])
            zd = float((clean - edited).abs().max().item())
            if zd > 1e-4:
                fails.append(f"M1 ZeroDelta changed logits by {zd}")
        ok = all(x["finite"] for x in recs) and len(recs) == 5
        if not ok:
            fails.append(f"M1 {label}: non-finite logits or missing cells")
        res = {"group": label, "dataset": ds, "max_len": L, "rows": kept, "encoding_check": chk,
               "cells": recs, "zero_delta_max_abs_logit_diff": zd, "pass": ok,
               "wall_s": time.time() - tc, "free_gb_before": gb}
        jdump(path, res)
        chunks.append({"group": label, "wall_s": res["wall_s"]})
        print(f"[model] {label}: NLL {np.mean([x['nll_mean'] for x in recs]):.3f}  "
              f"top1 {np.mean([x['top1'] for x in recs]):.3f}  {res['wall_s']:.0f}s", flush=True)
    del model, ext
    H.free_device_cache()
    return fails, chunks, time.time() - t0


def boot_ci(v, n_boot=10000, seed=SEED):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def part_summary():
    """Means with bootstrap 95% CIs (cells resampled, 10,000 draws, seed SEED)."""
    df = pd.read_csv(OUT / "cells_cpu.csv")
    out = {"agreement_v3_vs_deployed": {}, "model_v3": {}, "notes": {
        "unit": "Spearman = rank correlation of gene positions (unitless); overlaps = share of genes; "
                "NLL = nats per gene token (next-gene prediction); top1 = share of positions",
        "ci": f"bootstrap 95% CI of the mean over cells, 10,000 resamples, seed {SEED}"}}
    for label, ds, L, ref in GROUPS:
        g = df[df.group == label]
        rec = {}
        for lab, gg in [("primary5", g[g.primary]), ("all", g)]:
            rec[lab] = {"n": int(len(gg))}
            for m in METRICS:
                v = gg[m].to_numpy(float)
                rec[lab][m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                               "max": float(v.max()), "ci95": boot_ci(v)}
        out["agreement_v3_vs_deployed"][label] = rec
    cells = []
    for p in sorted(q for q in (OUT / "model").glob("*.json") if not q.name.startswith(".")):
        r = json.load(open(p))
        for c in r["cells"]:
            cells.append({"group": r["group"], **c})
    fails = []
    if cells:
        mc = pd.DataFrame(cells)
        mc["full_length"] = mc.assay.str.contains("Smart-seq")
        mc.to_csv(OUT / "model_cells.csv", index=False)
        for key, gg in list(mc.groupby("group")) + [("ALL_UMI", mc[~mc.full_length]),
                                                    ("ALL_SMARTSEQ", mc[mc.full_length])]:
            out["model_v3"][key] = {"n": int(len(gg))}
            for m in ["nll_mean", "nll_first200", "top1", "top1_first200"]:
                v = gg[m].to_numpy(float)
                out["model_v3"][key][m] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                                           "ci95": boot_ci(v) if len(v) > 1 else None}
        from scipy.stats import mannwhitneyu
        a, b = mc[mc.full_length].nll_mean, mc[~mc.full_length].nll_mean
        out["model_v3"]["smartseq_vs_umi_nll"] = {
            "n_smartseq": int(len(a)), "n_umi": int(len(b)), "min_smartseq": float(a.min()),
            "max_umi": float(b.max()), "mannwhitney_p_two_sided": float(mannwhitneyu(a, b).pvalue)}
    jdump(OUT / "summary.json", out)
    print(json.dumps(out["model_v3"].get("smartseq_vs_umi_nll", {}), indent=1))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="cpu", choices=["cpu", "model", "summary"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / "run_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {"chunks": []}
    import torch
    import transformers
    import scipy
    import h5py
    cfg.update({
        "script": str(Path(__file__).resolve()),
        "script_sha256": I.sha256_file(__file__),
        "inputs_v3_sha256": I.sha256_file(I.__file__),
        "maxtoki_adapter_sha256": I.sha256_file(SETUP / "maxtoki_adapter.py"),
        "dataset_loader_sha256": I.sha256_file(SETUP / "dataset_loader.py"),
        "hooks_v2_sha256": I.sha256_file(SETUP / "hooks_v2.py"),
        "audit_cells_csv_sha256": I.sha256_file(AUDIT / "order_metrics_cells.csv"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "h5py": h5py.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "platform": platform.platform()},
        "seeds": {"kidney_rows": SEED, "torch": SEED, "cell_rows": "audit order_metrics_cells.csv rows"},
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    })
    if args.part == "summary":
        t0 = time.time()
        fails = part_summary()
        cfg["chunks"].append({"part": "summary", "wall_s": time.time() - t0,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    elif args.part == "cpu":
        fails, wall = part_cpu()
        cfg["cpu"] = {"wall_s": wall, "fails": fails, "device": "cpu"}
        cfg["chunks"].append({"part": "cpu", "wall_s": wall, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        cpu = json.load(open(OUT / "cpu_results.json"))
        cfg["inputs"] = cpu["provenance"]
        cfg["cell_ids"] = {g: sorted(map(int, rr)) for g, rr in
                           pd.read_csv(OUT / "cells_cpu.csv").groupby("group").row}
        cfg["input_encoding_check"] = {"all_cells_pass_check_encoding": not any(f.startswith("C3") for f in fails),
                                       "fails": fails}
    else:
        fails, chunks, wall = part_model(args.device, args.max_minutes)
        cfg.setdefault("model", {"device": args.device, "model_dir": str(SETUP / "MaxToki-217M-HF"),
                                 "dtype": "float32", "fails": []})
        cfg["model"]["fails"] = cfg["model"].get("fails", []) + fails
        cfg["chunks"].append({"part": "model", "wall_s": wall, "groups": chunks,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    I_json = cfg
    jdump(cfg_path, I_json)
    print("FAILS:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
