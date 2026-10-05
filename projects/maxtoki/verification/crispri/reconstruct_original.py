"""Rebuild the gene-pair filter of BOTH scripts, restricted to the 248 silenced genes
that have CRISPRi records, and check:
  (1) the corrected rule reproduces per_pair.parquet exactly (pairs + predicted sign);
  (2) the ORIGINAL rule (edges whose two endpoints both carry an enrichment label)
      gives a subset of the corrected records with n = 1,458,016 and 796,159 decreases.
Then score the corrected sign rule and the constant baselines on the ORIGINAL records.
Read-only on the repo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<REPO_ROOT>/projects/maxtoki")
PH2 = REPO / "runs/sae-atlas-217M/outputs/phase2"
OUT = Path(__file__).resolve().parent

pp = pd.read_parquet(REPO / "runs/circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet")
S = sorted(pp["source"].unique()); s_idx = {s: i for i, s in enumerate(S)}

top10 = {}; labelled = set()
for li in range(12):
    d = PH2 / f"layer_{li:02d}"
    for c in json.load(open(d / "feature_catalog.json")):
        top10[(li, c["feature_id"])] = [g.upper() for g in c["top20_genes"][:10]]
    labelled |= {(li, f) for f in pd.read_csv(d / "significant_enrichments.csv", usecols=["feature_id"]).feature_id.unique()}
genes = sorted({g for v in top10.values() for g in v}); g_idx = {g: i for i, g in enumerate(genes)}; G = len(genes)
keys = list(top10.keys()); k_row = {k: i for i, k in enumerate(keys)}
T = np.full((len(keys), 10), -1, np.int64)
for k, v in top10.items():
    T[k_row[k], :len(v)] = [g_idx[g] for g in v]

ed = pd.read_csv(REPO / "runs/circuit-tracing-217M/outputs/circuit_edges.csv")
ed["tgt_row"] = [k_row[(a, b)] for a, b in zip(ed.tgt_layer, ed.tgt_feature)]
ed["lab"] = [(a, b) in labelled and (c, d) in labelled for a, b, c, d in
             zip(ed.src_layer, ed.src_feature, ed.tgt_layer, ed.tgt_feature)]
ed["inh"] = (ed["sign"] == "inhibitory").astype(np.int64)
ed["absd"] = ed["cohens_d"].abs()


def build(mask):
    E = np.zeros((len(S), G), np.int64); NI = np.zeros((len(S), G), np.int64); MX = np.zeros((len(S), G))
    sub = ed[mask]
    for (sl, sf), grp in sub.groupby(["src_layer", "src_feature"]):
        sgs = [g for g in top10.get((sl, sf), []) if g in s_idx]
        if not sgs:
            continue
        ids = T[grp["tgt_row"].to_numpy()]                     # (n_edges, 10)
        ok = ids >= 0
        flat = ids[ok]
        inh = np.repeat(grp["inh"].to_numpy()[:, None], 10, 1)[ok]
        ad = np.repeat(grp["absd"].to_numpy()[:, None], 10, 1)[ok]
        cnt = np.bincount(flat, minlength=G); ni = np.bincount(flat, weights=inh, minlength=G).astype(np.int64)
        mx = np.zeros(G); np.maximum.at(mx, flat, ad)
        for sg in sgs:                                         # duplicates in the list are counted, as in the scripts
            s = s_idx[sg]
            E[s] += cnt; NI[s] += ni; MX[s] = np.maximum(MX[s], mx)
    for sg in S:                                               # self pairs are never created by the scripts
        if sg in g_idx:
            E[s_idx[sg], g_idx[sg]] = 0; NI[s_idx[sg], g_idx[sg]] = 0; MX[s_idx[sg], g_idx[sg]] = 0
    keep = (E >= 2) | (MX > 2.0)
    si, gi = np.nonzero(keep)
    out = pd.DataFrame({"source": np.array(S)[si], "target": np.array(genes)[gi], "evidence": E[si, gi],
                        "n_inhib": NI[si, gi], "max_abs_d": MX[si, gi]})
    out["frac_inhib"] = out.n_inhib / out.evidence
    out["pred_dec"] = out["frac_inhib"] > 0.5
    return out


res = {}
corr = build(np.ones(len(ed), bool))
orig = build(ed["lab"].to_numpy())
res["edges_all"] = int(len(ed)); res["edges_both_labelled"] = int(ed["lab"].sum())

m = pp.merge(corr, on=["source", "target"], how="left", indicator=True)
res["corrected_check"] = dict(
    per_pair_rows=int(len(pp)), found_in_rebuilt=int((m["_merge"] == "both").sum()),
    pred_sign_matches=int((m["pred_dec"] == m["predicted_inhibitory"]).sum()),
    rebuilt_pairs_for_248_sources=int(len(corr)),
    rebuilt_pairs_not_in_per_pair_target_absent_from_replogle=int(len(corr) - len(pp)),
    n_ties_frac_inhib_eq_half=int((m["frac_inhib"] == 0.5).sum()),
)

o = pp.merge(orig[["source", "target", "pred_dec", "frac_inhib"]].rename(columns={"pred_dec": "pred_dec_orig_edges", "frac_inhib": "frac_inhib_orig"}),
             on=["source", "target"], how="inner")
n = len(o); dec = int((o.actual_lfc < 0).sum())
res["original_records_rebuilt"] = dict(
    n_pairs=int(n), n_decrease=dec, frac_decrease=dec / n, n_sources=int(o.source.nunique()),
    matches_logged_n_1458016=bool(n == 1458016), matches_logged_correct_796159=bool(dec == 796159),
    n_orig_pairs_missing_from_corrected=int(len(orig.merge(pp[["source", "target"]], on=["source", "target"], how="inner")) - n),
)
# score on the original records
for name, pred in [("corrected_sign_rule_all_edges", o.predicted_inhibitory.to_numpy()),
                   ("corrected_sign_rule_labelled_edges_only", o.pred_dec_orig_edges.to_numpy()),
                   ("always_decrease", np.ones(n, bool)), ("always_increase", np.zeros(n, bool))]:
    obs = (o.actual_lfc < 0).to_numpy()
    tp = (pred & obs).sum(); tn = (~pred & ~obs).sum(); fp = (pred & ~obs).sum(); fn = (~pred & obs).sum()
    acc = (tp + tn) / n
    bal = 0.5 * (tp / (tp + fn) + (tn / (tn + fp) if (tn + fp) else np.nan))
    res["original_records_rebuilt"][name] = dict(accuracy=float(acc), balanced_acc=float(bal), frac_pred_dec=float(pred.mean()))
extra = pp.merge(o[["source", "target"]], on=["source", "target"], how="left", indicator=True)
extra = extra[extra["_merge"] == "left_only"]
res["pairs_only_in_corrected"] = dict(n=int(len(extra)), frac_decrease=float((extra.actual_lfc < 0).mean()),
                                      n_sources=int(extra.source.nunique()))
(OUT / "original_vs_corrected_records.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
