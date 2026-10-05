"""v2 Stage-2 circuit tracing, part 2: per-cell vectors -> edges + checks (item D2).

Reads outputs/v2_circuit/trace_cells/*.npz (written by v2_circuit_trace.py) and
applies the deployed edge rule (circuit_trace.py:61-80, 240-256) unchanged:
  d           = mean_cells(dz) / sqrt(var_cells(dz, ddof=1) + 1e-12)
  consistency = max(#cells with dz > 0, n - #cells with dz > 0) / n
  edge        if |d| > 0.5 and consistency > 0.7
  sign        "inhibitory" if mean(dz) < 0 else "excitatory"
Note: the deployed consistency counts dz == 0 together with dz < 0. A symmetric
variant (max(#pos, #neg) / n) is reported as a sensitivity column only.

Writes (outputs/v2_circuit/):
  circuit_edges_v2.csv            the new edge list (columns documented in the report)
  edge_stats_main.npz             mean, d, consistency, n_pos, n_neg per (combo, target feature)
  aggregate_summary.json          counts per layer pair, sign balance, comparisons, checks
  edges_per_layer_pair.csv        new vs old counts per (source layer, target layer)
Usage: .venv/bin/python runs/circuit-tracing-217M/scripts/v2_circuit_aggregate.py
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
OUT = RUN / "outputs/v2_circuit"
CELL_DIR = OUT / "trace_cells"
OLD_EDGES = RUN / "outputs/circuit_edges.csv"
N_CELLS = 200
D_SAE = 4928
D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7


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


def edges_from(acc, combos, extra_cols=None):
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


def main():
    t0 = time.time()
    files = sorted(CELL_DIR.glob("cell_*.npz"))
    assert len(files) == N_CELLS, f"{len(files)} cell files, need {N_CELLS}"
    mc = pd.read_csv(OUT / "combos_main.csv", index_col="row")
    cc = pd.read_csv(OUT / "combos_ctrl.csv", index_col="row")
    feats = json.loads((OUT / "source_features.json").read_text())
    first_feat = {int(s): v[0] for s, v in feats.items()}
    lr_combos = cc.copy()
    lr_combos["src_feature"] = [first_feat[s] for s in cc.src_layer]

    acc = {k: Welford((len(mc) if k == "main" else len(cc), D_SAE))
           for k in ["main", "zero_v2", "legacy_zero", "legacy_real"]}
    zero_v2_maxabs = 0.0
    n_active = []
    rows, seconds, ntok = [], [], []
    n_legacy_fixed = 0
    for p in files:
        z = np.load(p)
        ci = int(p.name.split("_")[1])
        fix = OUT / "trace_cells_legacy_fix" / f"cell_{ci:03d}.npz"   # see v2_circuit_legacy_fix.py
        for k in acc:
            if k == "legacy_zero" and fix.exists():
                acc[k].update(np.load(fix)["legacy_zero"])
                n_legacy_fixed += 1
            else:
                acc[k].update(z[k])
        zero_v2_maxabs = max(zero_v2_maxabs, float(np.abs(z["zero_v2"]).max()))
        n_active.append(z["n_active"])
        rows.append(int(z["dataset_row"])); seconds.append(float(z["seconds"])); ntok.append(int(z["n_tokens"]))
    n_active = np.stack(n_active)                       # (200, 120)

    # ---------------- new edges
    edges, d, cons, cons_sym, mask, sym_mask = edges_from(acc["main"], mc)
    # the deployed script wrote cohens_d and consistency rounded to 4 decimals
    # (circuit_trace.py:253-254); the edge rule itself used unrounded values in both runs
    edges_out = edges.copy()
    edges_out["cohens_d"] = edges_out["cohens_d"].round(4)
    edges_out["consistency"] = edges_out["consistency"].round(4)
    edges_out.to_csv(OUT / "circuit_edges_v2.csv", index=False)
    del edges_out
    np.savez_compressed(OUT / "edge_stats_main.npz", mean=acc["main"].mean.astype(np.float32),
                        d=d.astype(np.float32), consistency=cons.astype(np.float32),
                        consistency_sym=cons_sym.astype(np.float32),
                        n_pos=acc["main"].pos.astype(np.int16), n_neg=acc["main"].neg.astype(np.int16))

    old = pd.read_csv(OLD_EDGES)
    lp_new = edges.groupby(["src_layer", "tgt_layer"]).agg(
        n_edges=("sign", "size"), n_inhibitory=("sign", lambda s: int((s == "inhibitory").sum())),
        mean_abs_d=("cohens_d", lambda x: float(np.abs(x).mean())),
        n_pass_symmetric=("passes_symmetric_rule", "sum")).reset_index()
    lp_old = old.groupby(["src_layer", "tgt_layer"]).agg(
        n_edges_old=("sign", "size"), n_inhibitory_old=("sign", lambda s: int((s == "inhibitory").sum()))).reset_index()
    lp = pd.DataFrame([(s, t) for s in [0, 3, 6, 9] for t in range(s + 1, 12)], columns=["src_layer", "tgt_layer"])
    lp = lp.merge(lp_new, how="left").merge(lp_old, how="left").fillna(0)
    for c in ["n_edges", "n_inhibitory", "n_pass_symmetric", "n_edges_old", "n_inhibitory_old"]:
        lp[c] = lp[c].astype(int)
    lp["share_inhibitory"] = lp.n_inhibitory / lp.n_edges.replace(0, np.nan)
    lp["share_inhibitory_old"] = lp.n_inhibitory_old / lp.n_edges_old.replace(0, np.nan)
    # the symmetric-rule sign balance per layer pair (computed over all combos, not only deployed edges)
    sym_r, sym_t = np.nonzero(sym_mask)
    sym_df = pd.DataFrame({"src_layer": mc.src_layer.to_numpy()[sym_r], "tgt_layer": mc.tgt_layer.to_numpy()[sym_r],
                           "inh": acc["main"].mean[sym_r, sym_t] < 0})
    lp.to_csv(OUT / "edges_per_layer_pair.csv", index=False)

    key = ["src_layer", "src_feature", "tgt_layer", "tgt_feature"]
    both = edges.merge(old, on=key, suffixes=("_new", "_old"))
    union = len(edges) + len(old) - len(both)

    # per source feature
    per_feat = edges.groupby(["src_layer", "src_feature"]).size()
    feat_index = mc.groupby(["src_layer", "src_feature"], sort=False).size().reset_index()[["src_layer", "src_feature"]]
    per_feat = feat_index.merge(per_feat.rename("n_edges").reset_index(), how="left").fillna(0)
    per_feat["n_edges"] = per_feat.n_edges.astype(int)
    per_feat["frac_cells_active"] = (n_active > 0).mean(0)
    per_feat["mean_active_positions"] = n_active.mean(0)
    per_feat.to_csv(OUT / "edges_per_source_feature.csv", index=False)

    # ---------------- controls
    z_edges, *_ = edges_from(acc["zero_v2"], cc.assign(src_feature=-1))
    lz_edges, *_ = edges_from(acc["legacy_zero"], cc.assign(src_feature=-1))
    lr_edges, lr_d, *_ = edges_from(acc["legacy_real"], lr_combos)
    lz_edges.to_csv(OUT / "control_legacy_zero_edges.csv", index=False)
    lr_edges.to_csv(OUT / "control_legacy_real_edges.csv", index=False)

    # legacy_real vs the deployed edges of the same feature (reproduction check)
    repro = []
    for s in [0, 3, 6, 9]:
        f = first_feat[s]
        o = old[(old.src_layer == s) & (old.src_feature == f)]
        n_ = lr_edges[(lr_edges.src_layer == s)]
        m = n_.merge(o, on=key, suffixes=("_v2legacy", "_deployed"))
        # d of the reproduced run at the deployed edges (whether or not it passes now)
        rows_l = [int(cc[(cc.src_layer == s) & (cc.tgt_layer == t)].index[0]) for t in o.tgt_layer]
        d_at_old = lr_d[rows_l, o.tgt_feature.to_numpy()]
        repro.append(dict(src_layer=s, src_feature=f, n_deployed=len(o), n_reproduced=len(n_), n_shared=len(m),
                          jaccard=len(m) / max(1, len(o) + len(n_) - len(m)),
                          max_abs_diff_d_at_deployed_edges=float(np.abs(d_at_old - o.cohens_d.to_numpy()).max()),
                          median_abs_diff_d_at_deployed_edges=float(np.median(np.abs(d_at_old - o.cohens_d.to_numpy()))),
                          sign_agreement_shared=float((m.sign_v2legacy == m.sign_deployed).mean()) if len(m) else None))
    # legacy_zero vs every deployed feature's edge set at the same source layer
    lz_vs_old = []
    for s in [0, 3, 6, 9]:
        lzs = lz_edges[lz_edges.src_layer == s][["tgt_layer", "tgt_feature"]]
        lz_set = set(map(tuple, lzs.to_numpy()))
        jac = []
        for f in feats[str(s)]:
            o = old[(old.src_layer == s) & (old.src_feature == f)][["tgt_layer", "tgt_feature"]]
            oset = set(map(tuple, o.to_numpy()))
            inter = len(lz_set & oset)
            jac.append(dict(feature=f, n_old=len(oset), n_shared=inter,
                            frac_old_explained=inter / max(1, len(oset)),
                            jaccard=inter / max(1, len(lz_set | oset))))
        jdf = pd.DataFrame(jac)
        lz_vs_old.append(dict(src_layer=s, n_legacy_zero_edges=len(lz_set),
                              share_inhibitory_legacy_zero=float((lz_edges[lz_edges.src_layer == s].sign == "inhibitory").mean()) if len(lzs) else None,
                              mean_old_edges_per_feature=float(jdf.n_old.mean()),
                              median_frac_old_edges_also_in_zero_edit=float(jdf.frac_old_explained.median()),
                              min_frac_old_edges_also_in_zero_edit=float(jdf.frac_old_explained.min()),
                              median_jaccard=float(jdf.jaccard.median())))

    # sanity: edges at l+1
    lplus1 = {f"L{s}->L{s + 1}": int(lp[(lp.src_layer == s) & (lp.tgt_layer == s + 1)].n_edges.iloc[0]) for s in [0, 3, 6, 9]}
    lplus1_old = {f"L{s}->L{s + 1}": int(lp[(lp.src_layer == s) & (lp.tgt_layer == s + 1)].n_edges_old.iloc[0]) for s in [0, 3, 6, 9]}

    n_inh = int((edges.sign == "inhibitory").sum())
    summary = dict(
        n_cells=acc["main"].n, dataset_rows_first5=rows[:5],
        n_source_features=len(feat_index),
        total_edges_v2=int(len(edges)), total_edges_deployed=int(len(old)),
        ratio_v2_to_deployed=len(edges) / len(old),
        share_inhibitory_v2=n_inh / max(1, len(edges)),
        share_inhibitory_deployed=float((old.sign == "inhibitory").mean()),
        share_inhibitory_v2_symmetric_rule=float(sym_df.inh.mean()) if len(sym_df) else None,
        total_edges_v2_symmetric_rule=int(sym_mask.sum()),
        n_edges_v2_failing_symmetric_rule=int((~edges.passes_symmetric_rule).sum()),
        mean_abs_d_v2=float(np.abs(edges.cohens_d).mean()) if len(edges) else None,
        median_abs_d_v2=float(np.abs(edges.cohens_d).median()) if len(edges) else None,
        unique_targets_v2=int(edges[["tgt_layer", "tgt_feature"]].drop_duplicates().shape[0]),
        unique_targets_deployed=int(old[["tgt_layer", "tgt_feature"]].drop_duplicates().shape[0]),
        overlap_with_deployed=dict(n_shared=int(len(both)), jaccard=len(both) / max(1, union),
                                   frac_deployed_edges_kept=len(both) / len(old),
                                   frac_v2_edges_in_deployed=len(both) / max(1, len(edges)),
                                   sign_agreement_shared=float((both.sign_new == both.sign_old).mean()) if len(both) else None,
                                   pearson_d_shared=float(np.corrcoef(both.cohens_d_new, both.cohens_d_old)[0, 1]) if len(both) > 2 else None),
        edges_per_source_feature=dict(mean=float(per_feat.n_edges.mean()), median=float(per_feat.n_edges.median()),
                                      min=int(per_feat.n_edges.min()), max=int(per_feat.n_edges.max()),
                                      n_features_with_zero_edges=int((per_feat.n_edges == 0).sum())),
        source_feature_activity=dict(median_frac_cells_active=float(per_feat.frac_cells_active.median()),
                                     n_features_never_active=int((per_feat.frac_cells_active == 0).sum()),
                                     median_mean_active_positions=float(per_feat.mean_active_positions.median())),
        checks=dict(
            edges_at_l_plus_1_v2=lplus1, edges_at_l_plus_1_deployed=lplus1_old,
            check_edges_at_l_plus_1_pass=all(v > 0 for v in lplus1.values()),
            zero_edit_v2_max_abs_dz_any_cell=zero_v2_maxabs,
            zero_edit_v2_n_edges=int(len(z_edges)),
            check_zero_edit_pass=bool(zero_v2_maxabs == 0.0 and len(z_edges) == 0),
            legacy_zero_edit_n_edges=int(len(lz_edges)),
            legacy_zero_n_cells_repaired=n_legacy_fixed,
            legacy_zero_vs_deployed=lz_vs_old,
            legacy_real_reproduction=repro,
        ),
        per_cell_seconds=dict(mean=float(np.mean(seconds)), total_hours=float(np.sum(seconds) / 3600)),
        wall_seconds=round(time.time() - t0, 1),
        script_sha256=H.sha256_file(__file__),
    )
    H.write_json(OUT / "aggregate_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=1, default=str))
    print(json.dumps(summary["checks"], indent=1, default=str))
    print(lp.to_string())


if __name__ == "__main__":
    main()
