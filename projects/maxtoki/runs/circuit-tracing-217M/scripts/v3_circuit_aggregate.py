"""v3 circuit tracing, part 2: per-cell vectors -> edges + checks (item V3-3).

Reads outputs/v3_circuit/trace_cells/*.npz (written by v3_circuit_trace.py: v3 SAEs, correctly
encoded K562 control cells, hooks_v2 edits) and applies the deployed edge rule unchanged
(circuit_trace.py:61-80, 240-256; same code as v2_circuit_aggregate.py):
  d           = mean_cells(dz) / sqrt(var_cells(dz, ddof=1) + 1e-12)
  consistency = max(#cells with dz > 0, n - #cells with dz > 0) / n
  edge        if |d| > 0.5 and consistency > 0.7
  sign        "inhibitory" if mean(dz) < 0 else "excitatory"
A symmetric consistency (max(#pos, #neg) / n) is reported as a sensitivity column only.

The v3 source features are different SAE features from the v2 / deployed ones (the SAEs were
retrained), so v3 and v2 edges cannot be matched edge by edge. The comparison is on counts,
layer pairs, sign shares and |d| distributions.

Writes (outputs/v3_circuit/):
  circuit_edges_v3.csv, edge_stats_main.npz, edges_per_layer_pair.csv, edges_per_source_feature.csv,
  aggregate_summary.json
Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/circuit-tracing-217M/scripts/v3_circuit_aggregate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v3_circuit"
CELL_DIR = OUT / "trace_cells"
V2_EDGES = RUN / "outputs/v2_circuit/circuit_edges_v2.csv"
DEP_EDGES = RUN / "outputs/circuit_edges.csv"
V3SAE = PROJ / "runs/sae-atlas-217M/outputs/v3_sae"
N_CELLS = 200
D_SAE = 4928
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
SOURCE_LAYERS = [0, 3, 6, 9]


class Welford:
    """Same accumulator as circuit_trace.py:56-80 (float64), plus a dz < 0 count."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, np.float64)
        self.M2 = np.zeros(shape, np.float64)
        self.pos = np.zeros(shape, np.int32)
        self.neg = np.zeros(shape, np.int32)

    def update(self, x):
        x = x.astype(np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        self.pos += (x > 0)
        self.neg += (x < 0)

    def finalize(self):
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var + 1e-12)
        d = self.mean / std
        cons = np.maximum(self.pos, self.n - self.pos) / self.n
        cons_sym = np.maximum(self.pos, self.neg) / self.n
        return d, cons, cons_sym


def edges_from(acc, combos):
    d, cons, cons_sym = acc.finalize()
    mask = (np.abs(d) > D_THRESHOLD) & (cons > CONSISTENCY_THRESHOLD)
    r, t = np.nonzero(mask)
    df = pd.DataFrame({
        "src_layer": combos["src_layer"].to_numpy()[r],
        "src_feature": combos["src_feature"].to_numpy()[r] if "src_feature" in combos else -1,
        "tgt_layer": combos["tgt_layer"].to_numpy()[r],
        "tgt_feature": t,
        "cohens_d": d[r, t],
        "consistency": cons[r, t],
        "sign": np.where(acc.mean[r, t] < 0, "inhibitory", "excitatory"),
        "mean_delta": acc.mean[r, t],
        "n_pos": acc.pos[r, t],
        "n_neg": acc.neg[r, t],
        "n_zero": acc.n - acc.pos[r, t] - acc.neg[r, t],
        "consistency_sym": cons_sym[r, t],
    })
    df["passes_symmetric_rule"] = (np.abs(df.cohens_d) > D_THRESHOLD) & (df.consistency_sym > CONSISTENCY_THRESHOLD)
    sym_mask = (np.abs(d) > D_THRESHOLD) & (cons_sym > CONSISTENCY_THRESHOLD)
    return df, d, cons, cons_sym, mask, sym_mask


def special_token_activity(feats):
    """Share of <bos>/<eos> positions (and of gene positions) where each source feature is active,
    in the 500 v3 SAE training cells (stored top-32 codes). The trace edits every position,
    including the special tokens, as the deployed run did."""
    gn = json.load(open(V3SAE / "gene_names_v3.json"))
    spec = np.array([g == "<special>" for g in gn])
    out = []
    for s in SOURCE_LAYERS:
        z = np.load(V3SAE / f"codes/layer_{s:02d}_topk.npz")
        idx, val = z["idx"], z["val"]
        for f in feats[s]:
            act = ((idx == f) & (val > 0)).any(1)
            out.append(dict(src_layer=s, src_feature=f, frac_special_positions_active=float(act[spec].mean()),
                            frac_gene_positions_active=float(act[~spec].mean())))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    files = sorted(CELL_DIR.glob("cell_*.npz"))
    assert len(files) == N_CELLS, f"{len(files)} cell files, need {N_CELLS}"
    mc = pd.read_csv(OUT / "combos_main.csv", index_col="row")
    cc = pd.read_csv(OUT / "combos_ctrl.csv", index_col="row")
    feats = {int(s): [int(x) for x in v] for s, v in json.loads((OUT / "source_features.json").read_text()).items()}
    toks = np.load(OUT / "cells_tokens.npz")

    acc = {"main": Welford((len(mc), D_SAE)), "zero_v2": Welford((len(cc), D_SAE))}
    zero_maxabs, n_active, rows, seconds, ntok = 0.0, [], [], [], []
    for i, p in enumerate(files):
        z = np.load(p)
        ci = int(p.name.split("_")[1])
        assert ci == i, (ci, i)
        assert int(z["dataset_row"]) == int(toks["rows"][ci]) and int(z["n_tokens"]) == int(toks["lengths"][ci])
        assert z["main"].shape == (len(mc), D_SAE) and z["zero_v2"].shape == (len(cc), D_SAE)
        assert np.isfinite(z["main"]).all()
        acc["main"].update(z["main"]); acc["zero_v2"].update(z["zero_v2"])
        zero_maxabs = max(zero_maxabs, float(np.abs(z["zero_v2"]).max()))
        n_active.append(z["n_active"])
        rows.append(int(z["dataset_row"])); seconds.append(float(z["seconds"])); ntok.append(int(z["n_tokens"]))
    n_active = np.stack(n_active)                                   # (200, 120)

    edges, d, cons, cons_sym, mask, sym_mask = edges_from(acc["main"], mc)
    out = edges.copy()
    out["cohens_d"] = out["cohens_d"].round(4)                      # as the deployed CSV (circuit_trace.py:253-254)
    out["consistency"] = out["consistency"].round(4)
    out.to_csv(OUT / "circuit_edges_v3.csv", index=False)
    del out
    np.savez_compressed(OUT / "edge_stats_main.npz", mean=acc["main"].mean.astype(np.float32), d=d.astype(np.float32),
                        consistency=cons.astype(np.float32), consistency_sym=cons_sym.astype(np.float32),
                        n_pos=acc["main"].pos.astype(np.int16), n_neg=acc["main"].neg.astype(np.int16))

    v2 = pd.read_csv(V2_EDGES)
    dep = pd.read_csv(DEP_EDGES)
    lp = pd.DataFrame([(s, t) for s in SOURCE_LAYERS for t in range(s + 1, 12)], columns=["src_layer", "tgt_layer"])
    for name, e in [("v3", edges), ("v2", v2), ("deployed", dep)]:
        g = e.groupby(["src_layer", "tgt_layer"]).agg(n=("sign", "size"),
                                                       inh=("sign", lambda s: int((s == "inhibitory").sum()))).reset_index()
        g = g.rename(columns={"n": f"n_edges_{name}", "inh": f"n_inhibitory_{name}"})
        lp = lp.merge(g, how="left")
    lp = lp.fillna(0)
    for name in ["v3", "v2", "deployed"]:
        lp[f"n_edges_{name}"] = lp[f"n_edges_{name}"].astype(int)
        lp[f"n_inhibitory_{name}"] = lp[f"n_inhibitory_{name}"].astype(int)
        lp[f"share_inhibitory_{name}"] = lp[f"n_inhibitory_{name}"] / lp[f"n_edges_{name}"].replace(0, np.nan)
    lp.to_csv(OUT / "edges_per_layer_pair.csv", index=False)

    per_feat = edges.groupby(["src_layer", "src_feature"]).size()
    feat_index = mc.groupby(["src_layer", "src_feature"], sort=False).size().reset_index()[["src_layer", "src_feature"]]
    per_feat = feat_index.merge(per_feat.rename("n_edges").reset_index(), how="left").fillna(0)
    per_feat["n_edges"] = per_feat.n_edges.astype(int)
    per_feat["frac_cells_active"] = (n_active > 0).mean(0)
    per_feat["mean_active_positions"] = n_active.mean(0)
    inh_f = edges.groupby(["src_layer", "src_feature"]).sign.apply(lambda s: float((s == "inhibitory").mean()))
    per_feat = per_feat.merge(inh_f.rename("share_inhibitory").reset_index(), how="left")
    spa = special_token_activity(feats)
    per_feat = per_feat.merge(spa, how="left")
    per_feat.to_csv(OUT / "edges_per_source_feature.csv", index=False)

    z_edges, *_ = edges_from(acc["zero_v2"], cc.assign(src_feature=-1))
    lplus1 = {f"L{s}->L{s + 1}": int(lp[(lp.src_layer == s) & (lp.tgt_layer == s + 1)].n_edges_v3.iloc[0]) for s in SOURCE_LAYERS}

    def summ(e):
        a = e.cohens_d.abs()
        return dict(total=int(len(e)), share_inhibitory=float((e.sign == "inhibitory").mean()) if len(e) else None,
                    edges_at_next_layer=int((e.tgt_layer == e.src_layer + 1).sum()),
                    abs_d_quantiles={str(q): float(a.quantile(q)) for q in [0.1, 0.5, 0.9, 0.99]} if len(e) else None,
                    frac_abs_d_gt_1=float((a > 1).mean()) if len(e) else None,
                    by_source_layer={str(s): dict(n=int((e.src_layer == s).sum()),
                                                  share_inhibitory=float((e[e.src_layer == s].sign == "inhibitory").mean())
                                                  if (e.src_layer == s).any() else None) for s in SOURCE_LAYERS},
                    edges_per_source_feature_median_by_layer=None)
    top = per_feat.sort_values("n_edges", ascending=False).head(5)
    summary = dict(
        n_cells=acc["main"].n, dataset_rows_first5=rows[:5], n_source_features=int(len(feat_index)),
        v3=summ(edges), v2=summ(v2), deployed=summ(dep),
        v3_symmetric_rule=dict(total=int(sym_mask.sum()), n_v3_edges_failing_symmetric_rule=int((~edges.passes_symmetric_rule).sum())),
        v3_edges_per_source_feature=dict(mean=float(per_feat.n_edges.mean()), median=float(per_feat.n_edges.median()),
                                         max=int(per_feat.n_edges.max()), n_features_with_zero_edges=int((per_feat.n_edges == 0).sum()),
                                         median_by_layer={str(s): float(per_feat[per_feat.src_layer == s].n_edges.median()) for s in SOURCE_LAYERS},
                                         top5=top[["src_layer", "src_feature", "n_edges", "frac_special_positions_active",
                                                   "frac_gene_positions_active"]].to_dict("records")),
        v3_unique_targets=int(edges[["tgt_layer", "tgt_feature"]].drop_duplicates().shape[0]),
        source_feature_activity=dict(median_frac_cells_active=float(per_feat.frac_cells_active.median()),
                                     n_features_never_active=int((per_feat.frac_cells_active == 0).sum()),
                                     n_inactive_cell_feature_cases=int((n_active == 0).sum()),
                                     median_mean_active_positions=float(per_feat.mean_active_positions.median()),
                                     n_features_active_on_special_tokens_gt_50pct=int((per_feat.frac_special_positions_active > 0.5).sum())),
        checks=dict(edges_at_l_plus_1=lplus1, check_edges_at_l_plus_1_pass=all(v > 0 for v in lplus1.values()),
                    zero_edit_max_abs_dz_any_cell=zero_maxabs, zero_edit_n_edges=int(len(z_edges)),
                    check_zero_edit_pass=bool(zero_maxabs == 0.0 and len(z_edges) == 0),
                    cells_in_order_rows_and_lengths_match_tokens=True),
        per_cell_seconds=dict(mean=float(np.mean(seconds)), total_hours=float(np.sum(seconds) / 3600)),
        n_tokens=dict(min=int(min(ntok)), median=float(np.median(ntok)), max=int(max(ntok))),
        wall_seconds=round(time.time() - t0, 1), script_sha256=H.sha256_file(__file__))
    H.write_json(OUT / "aggregate_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items()}, indent=1, default=str))
    print(lp.to_string())


if __name__ == "__main__":
    main()
