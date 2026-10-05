"""v3 circuit tracing, part 5: re-derive the key numbers a second way (item V3-3). CPU only.

1. Edges: d and consistency for every (combo, target feature) from the 200 per-cell files with a plain
   two-pass numpy computation (no Welford), in two halves of the combo rows; compare edge set, signs and
   d with circuit_edges_v3.csv / edge_stats_main.npz.
2. Records: the CRISPRi records rebuilt with a pandas merge / groupby (no bincount code from
   v2_circuit_crispri.build_records); compare with outputs/v3_circuit/per_pair.parquet. The same code
   is also run on the v2 edges and compared with outputs/v2_circuit/per_pair.parquet (a check of this
   second implementation against an independently checked record set).
3. Metrics: accuracy, balanced accuracy and MCC with scikit-learn; accuracy expected from the marginals by
   direct formula; the cross-fitted target-direction baseline with a plain loop; a grouped bootstrap over
   silenced genes with its own loop and its own seed (7), so its CI should match the main CI within
   Monte-Carlo error, not exactly.
4. LFC: 5 silenced genes (rng 20261002) recomputed from the h5ad with separate code (cells found with
   dataset_loader, controls redrawn with rng(42), counts from expm1(X) / smallest value, single log).
Writes outputs/v3_circuit/verify/verify.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from dataset_loader import resolve as load_ds  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
V3 = RUN / "outputs/v3_circuit"
V2 = RUN / "outputs/v2_circuit"
OUT = V3 / "verify"
ANN = V3 / "annotation"
LFC = V3 / "lfc"
DEP_PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
D_SAE = 4928


# ----------------------------------------------------------------------------- 1. edges
def verify_edges():
    mc = pd.read_csv(V3 / "combos_main.csv", index_col="row")
    files = sorted((V3 / "trace_cells").glob("cell_*.npz"))
    n = len(files)
    assert n == 200
    halves = [np.arange(0, len(mc) // 2), np.arange(len(mc) // 2, len(mc))]
    rec = []
    for rows in halves:
        A = np.empty((n, len(rows), D_SAE), np.float32)
        for i, p in enumerate(files):
            A[i] = np.load(p)["main"][rows]
        mean = A.mean(0, dtype=np.float64)
        var = ((A.astype(np.float64) - mean) ** 2).sum(0) / (n - 1)
        d = mean / np.sqrt(var + 1e-12)
        pos = (A > 0).sum(0)
        cons = np.maximum(pos, n - pos) / n
        del A
        m = (np.abs(d) > 0.5) & (cons > 0.7)
        r, t = np.nonzero(m)
        rec.append(pd.DataFrame({"row": rows[r], "tgt_feature": t, "d2": d[r, t], "cons2": cons[r, t],
                                 "sign2": np.where(mean[r, t] < 0, "inhibitory", "excitatory")}))
    e2 = pd.concat(rec)
    e2 = e2.join(mc, on="row")
    e1 = pd.read_csv(V3 / "circuit_edges_v3.csv")
    key = ["src_layer", "src_feature", "tgt_layer", "tgt_feature"]
    m = e1.merge(e2, on=key, how="outer", indicator=True)
    both = m[m._merge == "both"]
    st = np.load(V3 / "edge_stats_main.npz")
    return dict(n_edges_main=int(len(e1)), n_edges_second_way=int(len(e2)), n_both=int(len(both)),
                n_only_main=int((m._merge == "left_only").sum()), n_only_second=int((m._merge == "right_only").sum()),
                sign_agree=float((both.sign == both.sign2).mean()),
                max_abs_diff_d_vs_csv_rounded=float((both.cohens_d - both.d2).abs().max()),
                max_abs_diff_d_vs_npz_float32=float(np.abs(st["d"][both.row.to_numpy(), both.tgt_feature.to_numpy()] - both.d2.to_numpy()).max()),
                share_inhibitory_second=float((e2.sign2 == "inhibitory").mean()),
                pass_=bool((m._merge == "both").all() and (both.sign == both.sign2).all()))


# ----------------------------------------------------------------------------- 2. records
def top10_from(phase_dir):
    rows = []
    for li in range(12):
        for c in json.load(open(phase_dir / f"layer_{li:02d}/feature_catalog.json")):
            for g in c["top20_genes"][:10]:
                rows.append((li, int(c["feature_id"]), g.upper()))
    return pd.DataFrame(rows, columns=["layer", "feature", "gene"])


def records_second_way(edges, ft, lfc, src_tab, var_upper):
    rows_of = dict(zip(src_tab.source, src_tab.matrix_row))
    gene_col = {}
    for i, s in enumerate(var_upper):
        gene_col[s] = i                                   # last occurrence wins, as deployed
    keys = set(zip(ft.layer, ft.feature))
    e = edges[["src_layer", "src_feature", "tgt_layer", "tgt_feature", "sign", "cohens_d"]].copy()
    ok = [(a, b) in keys and (c, d) in keys for a, b, c, d in zip(e.src_layer, e.src_feature, e.tgt_layer, e.tgt_feature)]
    e = e[np.asarray(ok)].reset_index(drop=True)
    e["eid"] = np.arange(len(e))
    sg = e[["eid", "src_layer", "src_feature"]].merge(ft.rename(columns={"layer": "src_layer", "feature": "src_feature", "gene": "source"}),
                                                      on=["src_layer", "src_feature"])
    sg = sg[sg.source.isin(set(rows_of))]
    tg = e[["eid", "tgt_layer", "tgt_feature"]].merge(ft.rename(columns={"layer": "tgt_layer", "feature": "tgt_feature", "gene": "target"}),
                                                      on=["tgt_layer", "tgt_feature"])
    x = sg[["eid", "source"]].merge(tg[["eid", "target"]], on="eid")
    x = x[x.source != x.target]
    x = x.join(e.set_index("eid")[["sign", "cohens_d"]], on="eid")
    x["inh"] = (x.sign == "inhibitory").astype(int)
    x["absd"] = x.cohens_d.abs()
    g = x.groupby(["source", "target"]).agg(evidence=("eid", "size"), n_inhibitory=("inh", "sum"), max_abs_d=("absd", "max")).reset_index()
    g = g[(g.evidence >= 2) | (g.max_abs_d > 2.0)]
    g = g[g.target.isin(set(gene_col))].copy()
    g["predicted_inhibitory"] = g.n_inhibitory / g.evidence > 0.5
    g["actual_lfc"] = [float(lfc[rows_of[s], gene_col[t]]) for s, t in zip(g.source, g.target)]
    return g.sort_values(["source", "target"]).reset_index(drop=True), int(len(e))


def compare_records(a, b):
    m = a.merge(b, on=["source", "target"], how="outer", indicator=True, suffixes=("_a", "_b"))
    both = m[m._merge == "both"]
    return dict(n_a=int(len(a)), n_b=int(len(b)), n_both=int(len(both)),
                evidence_equal=bool((both.evidence_a == both.evidence_b).all()),
                n_inhibitory_equal=bool((both.n_inhibitory_a == both.n_inhibitory_b).all()),
                pred_equal=bool((both.predicted_inhibitory_a == both.predicted_inhibitory_b).all()),
                max_abs_lfc_diff=float((both.actual_lfc_a - both.actual_lfc_b).abs().max()),
                pass_=bool(len(both) == len(a) == len(b) and (both.predicted_inhibitory_a == both.predicted_inhibitory_b).all()
                           and (both.evidence_a == both.evidence_b).all() and float((both.actual_lfc_a - both.actual_lfc_b).abs().max()) == 0.0))


# ----------------------------------------------------------------------------- 3. metrics
def metrics_second_way(pp):
    y = (pp.actual_lfc < 0).to_numpy(); p = pp.predicted_inhibitory.to_numpy()
    po, pr = y.mean(), p.mean()
    # cross-fitted target baseline, plain loop (folds as v2_circuit_crispri: rng(42) permutation of sorted sources, 5 splits)
    srcs = np.asarray(sorted(pp.source.unique()))
    folds = np.array_split(np.random.default_rng(42).permutation(srcs), 5)
    correct = 0
    for k in range(5):
        te_src = set(folds[k])
        tr = pp[~pp.source.isin(te_src)]; te = pp[pp.source.isin(te_src)]
        glob = (tr.actual_lfc < 0).mean() > 0.5
        frac = {}
        for t, v in zip(tr.target, (tr.actual_lfc < 0)):
            a = frac.setdefault(t, [0, 0]); a[0] += int(v); a[1] += 1
        for t, v in zip(te.target, (te.actual_lfc < 0)):
            if t in frac and frac[t][0] * 2 != frac[t][1]:
                pred = frac[t][0] * 2 > frac[t][1]
            else:
                pred = glob
            correct += int(pred == v)
    # grouped bootstrap over silenced genes, own loop, own seed
    cnt = pd.DataFrame({"source": pp.source.to_numpy(), "tp": p & y, "fn": ~p & y, "fp": p & ~y, "tn": ~p & ~y}).groupby("source").sum()
    C = cnt.to_numpy(float)
    rng = np.random.default_rng(7)
    bal, mcc, accmc = [], [], []
    for _ in range(2000):
        s = C[rng.integers(0, len(C), len(C))].sum(0)
        tp, fn, fp, tn = s
        nn = s.sum()
        bal.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
        mcc.append((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        pobs = (tp + fn) / nn
        accmc.append((tp + tn) / nn - max(pobs, 1 - pobs))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(n=int(len(pp)), accuracy=float(accuracy_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
                mcc=float(matthews_corrcoef(y, p)), always_decrease=float(po), always_increase=float(1 - po),
                chance_from_marginals=float(pr * po + (1 - pr) * (1 - po)),
                target_baseline_accuracy=float(correct / len(pp)),
                boot_seed7_ci_balanced_minus_half=ci(np.asarray(bal) - 0.5), boot_seed7_ci_mcc=ci(mcc),
                boot_seed7_ci_acc_minus_best_constant=ci(accmc))


# ----------------------------------------------------------------------------- 4. LFC
def verify_lfc(sources_used):
    src = pd.read_csv(LFC / "sources.csv").set_index("source")
    P = np.load(LFC / "lfc_panel.npy")
    rng = np.random.default_rng(20261002)
    pick = sorted(rng.choice(sorted(sources_used), 5, replace=False).tolist())
    ds = load_ds("k562")
    ctrl_all = np.where(ds.cell_of_interest_mask & np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    ctrl = np.sort(np.random.default_rng(42).choice(ctrl_all, 3000, replace=False))
    same_ctrl = bool(np.array_equal(ctrl, np.load(LFC / "control_rows.npy")))

    def mean_single_log(f, rows):
        acc = np.zeros(ds.n_genes_total)
        for i in range(0, len(rows), 250):
            X = f["X"][rows[i:i + 250], :].astype(np.float64)
            E = np.expm1(X)
            U = np.where(E > 0, E, np.inf).min(1, keepdims=True)
            N = np.round(E / U)
            assert np.abs(E / U - N).max() < 1e-3
            acc += np.log1p(N / N.sum(1, keepdims=True) * 1e4).sum(0)
        return acc / len(rows)
    out = []
    with h5py.File(ds.h5_path, "r") as f:
        cm = mean_single_log(f, ctrl)
        for g in pick:
            code = int(src.loc[g, "pert_code"])
            rows = np.where(ds.cell_of_interest_mask & (ds.perturbation_codes == code))[0]
            lfc = mean_single_log(f, rows) - cm
            out.append(dict(gene=g, n_cells_second_way=int(len(rows)), n_cells_main=int(src.loc[g, "n_cells"]),
                            max_abs_diff=float(np.abs(lfc - P[int(src.loc[g, "matrix_row"])]).max()),
                            self_lfc=float(src.loc[g, "self_lfc_panel"])))
    return dict(control_rows_equal=same_ctrl, genes=out,
                pass_=bool(same_ctrl and all(o["n_cells_second_way"] == o["n_cells_main"] and o["max_abs_diff"] < 1e-9 for o in out)))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    res["edges"] = verify_edges(); print("edges", res["edges"], flush=True)
    src_tab = pd.read_csv(LFC / "sources.csv"); src_tab = src_tab[src_tab.has_data]
    var_upper = (LFC / "var_symbols.txt").read_text().split("\n")
    pp3 = pd.read_parquet(V3 / "per_pair.parquet")
    r3, _ = records_second_way(pd.read_csv(V3 / "circuit_edges_v3.csv"), top10_from(ANN), np.load(LFC / "lfc_panel.npy"), src_tab, var_upper)
    res["records_v3"] = compare_records(pp3, r3); print("records v3", res["records_v3"], flush=True)
    pp2 = pd.read_parquet(V2 / "per_pair.parquet")
    r2, _ = records_second_way(pd.read_csv(V2 / "circuit_edges_v2.csv"), top10_from(DEP_PHASE2), np.load(LFC / "lfc_v2style.npy").astype(np.float64),
                               src_tab, var_upper)
    res["records_v2_second_way_vs_v2_per_pair"] = compare_records(pp2, r2); print("records v2", res["records_v2_second_way_vs_v2_per_pair"], flush=True)
    res["metrics_v3_second_way"] = metrics_second_way(pp3)
    main_res = json.loads((V3 / "crispri/results.json").read_text())["results"]["A_v3_edges_single_log"]
    p = main_res["pooled"]; tb = [b for b in main_res["baselines"] if b["predictor"].startswith("per-target")][0]
    bs = main_res["bootstrap_by_silenced_gene"]["stats"]
    ms = res["metrics_v3_second_way"]
    res["metrics_compare"] = dict(
        accuracy=[p["accuracy"], ms["accuracy"]], balanced=[p["balanced_acc"], ms["balanced_accuracy"]], mcc=[p["mcc"], ms["mcc"]],
        chance=[p["chance_acc_given_marginals"], ms["chance_from_marginals"]],
        target_baseline=[tb["accuracy"], ms["target_baseline_accuracy"]],
        ci_balanced_minus_half=[bs["pooled_balanced_acc_minus_0.5"]["ci95"], ms["boot_seed7_ci_balanced_minus_half"]],
        ci_mcc=[bs["pooled_mcc"]["ci95"], ms["boot_seed7_ci_mcc"]],
        pass_point_estimates=bool(abs(p["accuracy"] - ms["accuracy"]) < 1e-12 and abs(p["balanced_acc"] - ms["balanced_accuracy"]) < 1e-12
                                  and abs(p["mcc"] - ms["mcc"]) < 1e-9 and abs(tb["accuracy"] - ms["target_baseline_accuracy"]) < 1e-12))
    print("metrics", res["metrics_compare"], flush=True)
    res["lfc"] = verify_lfc(set(pp3.source)); print("lfc", res["lfc"], flush=True)
    res["all_pass"] = bool(res["edges"]["pass_"] and res["records_v3"]["pass_"] and res["records_v2_second_way_vs_v2_per_pair"]["pass_"]
                           and res["metrics_compare"]["pass_point_estimates"] and res["lfc"]["pass_"])
    res["script_sha256"] = H.sha256_file(__file__); res["wall_seconds"] = round(time.time() - t0, 1)
    H.write_json(OUT / "verify.json", res)
    print("ALL PASS" if res["all_pass"] else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
