"""v3 circuit tracing, part 0: feature catalogs, enrichments and the source-feature selection (item V3-3).

Why: the source features of the deployed circuit run (and of v2) were chosen on the deployed SAEs,
which were trained on wrongly encoded inputs (checks/INPUT_ENCODING_AUDIT.md). The v3 SAEs
(runs/sae-atlas-217M/outputs/v3_sae/) have different features, so the deployed feature IDs mean
nothing. This script re-applies the deployed RULE to the v3 SAEs. CPU only, no forward pass.

Deployed chain (re-implemented exactly, then checked against the deployed files):
  1. Catalog (runs/sae-atlas-217M/scripts/full_12layer_pipeline.py:194-235): encode every token
     position of the 500 SAE training cells (incl. <bos>/<eos> = "<SPECIAL>"); for each alive feature
     and gene, mean activation over the positions where the feature is active; top-20 genes = the 20
     highest means (np.argsort(-mean), float32, mean > 0). Here the codes come from the stored top-32
     v3 codes (outputs/v3_sae/codes/layer_XX_topk.npz; = TopK encode of the training activations,
     checked by V3-1) and the v3 token names (outputs/v3_sae/gene_names_v3.json).
  2. Enrichment (full_12layer_pipeline.py:81-98, 238-264): GO_BP, KEGG, Reactome, TRRUST terms;
     each term restricted to the K562 panel symbols (upper case), kept if >= 5 genes;
     for each feature and term: x = |top20 & term|; if x >= 2, p = hypergeom.sf(x-1, N, K, n) with
     N = number of distinct names in the catalog cells (incl. <SPECIAL>), K = term size, n = list
     size; rows with p < 0.1 are BH-corrected together (per layer); significant = q < 0.05.
  3. Source selection (runs/circuit-tracing-217M/scripts/circuit_trace.py:133-145): per feature,
     score = sum over its significant rows of -log10(max(p_raw, 1e-300)); the 30 features with the
     largest score at each of layers 0, 3, 6, 9 (pandas sort_values(ascending=False), as deployed).

Checks written to outputs/v3_circuit/annotation/checks.json:
  * enrichment code fed the DEPLOYED catalogs reproduces every deployed significant_enrichments.csv
    (same rows in the same order, p_raw and q_bh equal);
  * selection code fed the deployed enrichments reproduces the deployed / v2 120 source features;
  * the v3 layer-5 catalog equals the one built independently by item V3-2
    (runs/sae-atlas-217M/outputs/v3_tf_specificity/task_data/feature_top20.tsv).
Outputs: outputs/v3_circuit/annotation/layer_XX/{feature_catalog.json, significant_enrichments.csv,
annotation_summary.json}, source_features.json, selection_detail.csv, checks.json, run_config.json.
Usage: OMP_NUM_THREADS=4 .venv/bin/python runs/circuit-tracing-217M/scripts/v3_circuit_annotate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402  (json writing, hashing)
import inputs_v3 as I  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v3_circuit/annotation"
V3SAE = PROJ / "runs/sae-atlas-217M/outputs/v3_sae"
DEP_PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
V2_FEATS = RUN / "outputs/v2_circuit/source_features.json"
V32_TOP20 = PROJ / "runs/sae-atlas-217M/outputs/v3_tf_specificity/task_data/feature_top20.tsv"
BIOM_ROOT = Path("<DATA_ROOT>")
GO_BP_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"
KEGG_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/kegg_gene_sets.json"
REACTOME_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/reactome_gene_sets.json"
TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
D_SAE = 4928
K_TOP = 32
N_LAYERS = 12
SOURCE_LAYERS = [0, 3, 6, 9]
N_FEATURES_PER_LAYER = 30


# ----------------------------------------------------------------------------- databases (deployed lines 81-98)
def load_terms():
    with h5py.File(I.DATASETS["k562"]["path"], "r") as f:
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    go_bp = json.load(open(GO_BP_JSON)); kegg = json.load(open(KEGG_JSON)); reactome = json.load(open(REACTOME_JSON))
    trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    trrust["tf"] = trrust["tf"].str.upper(); trrust["target"] = trrust["target"].str.upper()
    trrust_tf_sets = {f"TRRUST_TF:{tf}": list(grp["target"].unique()) for tf, grp in trrust.groupby("tf")}
    all_db = {}
    for name, db in [("GO_BP", go_bp), ("KEGG", kegg), ("Reactome", reactome), ("TRRUST_TF", trrust_tf_sets)]:
        for term, genes in db.items():
            all_db[f"{name}:{term}"] = set(g.upper() for g in genes)
    vs = set(g.upper() for g in var_symbols)
    terms = [(k, v & vs, len(v & vs)) for k, v in all_db.items()]
    terms = [(k, v, n) for k, v, n in terms if n >= 5]
    return terms, var_symbols


def enrich(catalog, terms, N_total):
    """Vectorised deployed enrichment (lines 238-255). Rows come out in the deployed order
    (catalog order, then term order). Returns (all rows with p<0.1 + q_bh, significant rows)."""
    genes = sorted({g for _, v, _ in terms for g in v} | {g for c in catalog for g in c["top20_genes"]})
    gi = {g: i for i, g in enumerate(genes)}
    r, c = [], []
    for ti, (_, v, _) in enumerate(terms):
        for g in v:
            r.append(ti); c.append(gi[g])
    Tm = sp.csr_matrix((np.ones(len(r), np.int32), (r, c)), shape=(len(terms), len(genes)))
    r, c = [], []
    for fi, fc in enumerate(catalog):
        for g in set(fc["top20_genes"]):
            r.append(fi); c.append(gi[g])
    Fm = sp.csr_matrix((np.ones(len(r), np.int32), (r, c)), shape=(len(catalog), len(genes)))
    ov = (Fm @ Tm.T).toarray()                                  # (n_features, n_terms) overlaps
    n_list = np.array([len(set(fc["top20_genes"])) for fc in catalog])
    K = np.array([k for _, _, k in terms])
    fi, ti = np.nonzero(ov >= 2)                                # row-major: feature, then term
    x = ov[fi, ti]
    p = hypergeom.sf(x - 1, N_total, K[ti], n_list[fi]).astype(float)
    keep = p < 0.1
    df = pd.DataFrame({"feature_id": np.array([catalog[i]["feature_id"] for i in fi[keep]], dtype=np.int64),
                       "term": [terms[t][0] for t in ti[keep]], "overlap": x[keep].astype(np.int64),
                       "K": K[ti[keep]].astype(np.int64), "n": n_list[fi[keep]].astype(np.int64),
                       "N": np.int64(N_total), "p_raw": p[keep]})
    if len(df):
        _, df["q_bh"], _, _ = multipletests(df["p_raw"], method="fdr_bh")
        sig = df[df["q_bh"] < 0.05]
    else:
        sig = df
    return df, sig


def select(sig: pd.DataFrame, k=N_FEATURES_PER_LAYER):
    """Deployed rule (circuit_trace.py:136-141; same expression as v2_circuit_trace.py:95-97)."""
    score = sig.groupby("feature_id")["p_raw"].apply(
        lambda p: (-np.log10(p.clip(1e-300))).sum()).sort_values(ascending=False)
    return [int(x) for x in score.head(k).index.tolist()], score


# ----------------------------------------------------------------------------- catalog (deployed lines 194-235)
def build_catalog(layer, pos_code, uniq, n_pos):
    z = np.load(V3SAE / f"codes/layer_{layer:02d}_topk.npz")
    idx, val = z["idx"], z["val"]
    assert idx.shape == (n_pos, K_TOP), idx.shape
    NU = len(uniq)
    S = np.zeros(D_SAE * NU, np.float64); C = np.zeros(D_SAE * NU, np.float64)
    fmax = np.zeros(D_SAE, np.float32); ftot = np.zeros(D_SAE, np.int64)
    step = 100_000
    n_nonpos = 0
    for a in range(0, n_pos, step):
        b = min(n_pos, a + step)
        ix = idx[a:b].astype(np.int64); vv = val[a:b]
        act = vv > 0
        n_nonpos += int((~act).sum())
        g = np.repeat(pos_code[a:b, None], K_TOP, 1)
        flat = ix[act] * NU + g[act]
        S += np.bincount(flat, weights=vv[act].astype(np.float64), minlength=D_SAE * NU)
        C += np.bincount(flat, minlength=D_SAE * NU)
        ftot += np.bincount(ix[act], minlength=D_SAE)
        np.maximum.at(fmax, ix[act], vv[act])
    S = S.reshape(D_SAE, NU); C = C.reshape(D_SAE, NU)
    catalog = []
    for fi in np.nonzero(ftot > 0)[0]:
        cnt = C[fi]; s = S[fi]
        nz = cnt > 0
        mpg = np.zeros(NU, dtype=np.float32)
        mpg[nz] = s[nz] / cnt[nz]
        t20i = np.argsort(-mpg)[:20]
        t20 = [uniq[i] for i in t20i if mpg[i] > 0][:20]
        catalog.append({"feature_id": int(fi), "top20_genes": t20, "max_activation": float(fmax[fi]),
                        "activation_frequency": float(ftot[fi] / n_pos)})
    return catalog, dict(n_codes_not_positive=n_nonpos, n_alive=int((ftot > 0).sum()))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    terms, var_symbols = load_terms()
    print(f"{len(terms)} terms after filtering ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- check A: enrichment code on the deployed catalogs = deployed CSVs
    dep_rows = []
    for li in range(N_LAYERS):
        cat = json.load(open(DEP_PHASE2 / f"layer_{li:02d}/feature_catalog.json"))
        old = pd.read_csv(DEP_PHASE2 / f"layer_{li:02d}/significant_enrichments.csv")
        N_dep = int(old["N"].iloc[0])
        _, sig = enrich(cat, terms, N_dep)
        sig = sig.reset_index(drop=True)
        same_rows = len(sig) == len(old) and (sig.feature_id.to_numpy() == old.feature_id.to_numpy()).all() \
            and (sig.term.to_numpy() == old.term.to_numpy()).all()
        dp = float(np.abs(sig.p_raw.to_numpy() - old.p_raw.to_numpy()).max()) if same_rows else None
        dq = float(np.abs(sig.q_bh.to_numpy() - old.q_bh.to_numpy()).max()) if same_rows else None
        ok_int = bool(same_rows and (sig[["overlap", "K", "n", "N"]].to_numpy() == old[["overlap", "K", "n", "N"]].to_numpy()).all())
        dep_rows.append(dict(layer=li, n_deployed=int(len(old)), n_recomputed=int(len(sig)), same_rows_same_order=bool(same_rows),
                             same_overlap_K_n_N=ok_int, max_abs_diff_p_raw=dp, max_abs_diff_q_bh=dq,
                             max_rel_diff_p_raw=float(np.max(np.abs(sig.p_raw.to_numpy() - old.p_raw.to_numpy()) / old.p_raw.to_numpy())) if same_rows else None))
        print("deployed enrichment check", dep_rows[-1], flush=True)
    checks["enrichment_code_vs_deployed_csv"] = dict(
        rows=dep_rows, pass_=all(r["same_rows_same_order"] and r["same_overlap_K_n_N"] and r["max_rel_diff_p_raw"] < 1e-9
                                 and r["max_abs_diff_q_bh"] < 1e-12 for r in dep_rows))

    # ---------------- check B: selection code on the deployed enrichments = deployed / v2 features
    v2f = json.loads(V2_FEATS.read_text())
    selB = {}
    for s in SOURCE_LAYERS:
        selB[s], _ = select(pd.read_csv(DEP_PHASE2 / f"layer_{s:02d}/significant_enrichments.csv"))
    checks["selection_code_vs_deployed_features"] = dict(
        per_layer={str(s): bool(selB[s] == [int(x) for x in v2f[str(s)]]) for s in SOURCE_LAYERS},
        pass_=all(selB[s] == [int(x) for x in v2f[str(s)]] for s in SOURCE_LAYERS))
    print("selection check", checks["selection_code_vs_deployed_features"], flush=True)

    # ---------------- v3 catalogs + enrichments
    gn = json.load(open(V3SAE / "gene_names_v3.json"))
    gn_up = [g.upper() for g in gn]
    uniq = sorted(set(gn_up))
    u_idx = {g: i for i, g in enumerate(uniq)}
    pos_code = np.array([u_idx[g] for g in gn_up], dtype=np.int64)
    N_total = len(uniq)
    summaries, sel_detail, feats = [], [], {}
    for li in range(N_LAYERS):
        t1 = time.time()
        d = OUT / f"layer_{li:02d}"; d.mkdir(parents=True, exist_ok=True)
        cat, info = build_catalog(li, pos_code, uniq, len(gn_up))
        allrows, sig = enrich(cat, terms, N_total)
        sig.to_csv(d / "significant_enrichments.csv", index=False)
        with open(d / "feature_catalog.json", "w") as f:
            json.dump(cat, f)
        n_ann = int(sig.feature_id.nunique()) if len(sig) else 0
        summ = {"layer": li, "n_alive": info["n_alive"], "n_annotated": n_ann,
                "annotation_rate": round(n_ann / max(info["n_alive"], 1), 4), "total_enrichments": int(len(sig)),
                "n_rows_p_lt_0.1": int(len(allrows)), "N_distinct_names": N_total,
                "n_topk_codes_not_positive": info["n_codes_not_positive"],
                "n_lists_with_special": int(sum("<SPECIAL>" in c["top20_genes"] for c in cat)),
                "n_lists_shorter_than_20": int(sum(len(c["top20_genes"]) < 20 for c in cat))}
        H.write_json(d / "annotation_summary.json", summ)
        summaries.append(summ)
        if li in SOURCE_LAYERS:
            top, score = select(sig)
            feats[li] = top
            cmap = {c["feature_id"]: c for c in cat}
            nsig = sig.groupby("feature_id").size()
            for rank, f in enumerate(top):
                sel_detail.append(dict(src_layer=li, rank=rank + 1, feature_id=f, score_sum_neglog10p=float(score.loc[f]),
                                       n_significant_terms=int(nsig.loc[f]),
                                       best_term=sig[sig.feature_id == f].sort_values("p_raw").term.iloc[0],
                                       activation_frequency=cmap[f]["activation_frequency"],
                                       top10_genes=" ".join(cmap[f]["top20_genes"][:10])))
            s30 = float(score.iloc[N_FEATURES_PER_LAYER - 1]); s31 = float(score.iloc[N_FEATURES_PER_LAYER]) if len(score) > 30 else None
            summ["selection"] = dict(score_30th=s30, score_31st=s31, tie_at_cut=bool(s31 is not None and s30 == s31),
                                     n_features_with_score=int(len(score)))
            H.write_json(d / "annotation_summary.json", summ)
        print(f"layer {li}: alive {info['n_alive']} annotated {n_ann} sig rows {len(sig)} ({time.time() - t1:.0f}s)", flush=True)
    H.write_json(OUT / "source_features.json", {str(s): feats[s] for s in SOURCE_LAYERS})
    pd.DataFrame(sel_detail).to_csv(OUT / "selection_detail.csv", index=False)

    # ---------------- check C: layer-5 catalog = V3-2's independently built catalog
    top = pd.read_csv(V32_TOP20, sep="\t", keep_default_na=False)
    ref = top.sort_values(["feature_id", "rank"]).groupby("feature_id")["gene"].apply(list).to_dict()
    mine = {c["feature_id"]: c["top20_genes"] for c in json.load(open(OUT / "layer_05/feature_catalog.json"))}
    same = sum(mine.get(f) == ref.get(f) for f in set(mine) | set(ref))
    checks["layer5_catalog_vs_v3_2"] = dict(n_features_mine=len(mine), n_features_v3_2=len(ref), n_identical_ordered_lists=int(same),
                                            pass_=bool(same == len(set(mine) | set(ref))))
    # ---------------- overlap of v3 source features with the deployed ids (numbers only; different SAEs)
    checks["v3_vs_deployed_feature_ids_same_number"] = {str(s): len(set(feats[s]) & set(int(x) for x in v2f[str(s)])) for s in SOURCE_LAYERS}
    checks["all_pass"] = bool(checks["enrichment_code_vs_deployed_csv"]["pass_"] and checks["selection_code_vs_deployed_features"]["pass_"]
                              and checks["layer5_catalog_vs_v3_2"]["pass_"])
    H.write_json(OUT / "checks.json", checks)
    H.write_json(OUT / "summary.json", dict(layers=summaries, n_terms=len(terms)))
    H.write_json(OUT / "run_config.json", dict(
        script=str(Path(__file__)), script_sha256=H.sha256_file(__file__), device="cpu",
        inputs=dict(v3_sae_codes={li: H.sha256_file(V3SAE / f"codes/layer_{li:02d}_topk.npz") for li in range(N_LAYERS)},
                    gene_names_v3_sha256=H.sha256_file(V3SAE / "gene_names_v3.json"),
                    v3_sae_cells_sha256=H.sha256_file(V3SAE / "cells.npz"),
                    databases={str(p): H.sha256_file(p) for p in [GO_BP_JSON, KEGG_JSON, REACTOME_JSON, TRRUST_TSV]}),
        input_encoding="v3 SAE training cells (500 K562 controls) encoded by inputs_v3 (counts / median); "
                       "encoding check passed on 500/500 cells before every forward pass in V3-1 (outputs/v3_sae/run_config.json)",
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - t0, 1)))
    print(json.dumps({k: v for k, v in checks.items() if k != "enrichment_code_vs_deployed_csv"}, indent=1))
    print(json.dumps({str(s): feats[s] for s in SOURCE_LAYERS}))


if __name__ == "__main__":
    main()
