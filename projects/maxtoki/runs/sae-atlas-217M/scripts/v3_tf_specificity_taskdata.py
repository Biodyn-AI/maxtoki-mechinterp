"""V3-2: gene universe, target sets and the layer-5 top-20 catalog of the v3 SAE (built twice).

Same design as v2_tf_specificity_taskdata.py, with the v3 SAE and correctly encoded inputs.
In v2 the catalog was the deployed one (outputs/phase2/layer_05/feature_catalog.json) and our
forward passes only checked it. The deployed catalog belongs to the old SAE, so here it is
REBUILT from the v3 SAE with the deployed definition (full_12layer_pipeline.py:197-235):
  encode every token position of the 500 catalog cells (incl. <bos>/<eos> = '<SPECIAL>');
  per (feature, gene): mean activation over positions where the feature is active (> 0);
  top-20 genes by that mean (only means > 0), no minimum count.

Two independent sources of the same codes:
  A (primary) outputs/v3_sae/codes/layer_05_topk.npz: top-32 v3 codes for all 1,019,996
    positions, written by the V3-1 SAE item from its stored activations.
  B (check)   our own forward passes of the same 500 cells (v3_tf_specificity_extract.py,
    group 'catalog'), live hook at the input of block 5.
TopK with k = 32 means the top-32 codes ARE the full code (every other feature is 0).

Checks (task_data/catalog_check.json)
  1. our catalog-cell tokens == outputs/v3_sae/cells.npz tokens_v3 (and gene_names_v3.json)
  2. per-position codes A vs B (same active set; value differences)
  3. top-20 lists from A vs from B
Outputs: outputs/v3_tf_specificity/task_data/{gene_universe.tsv, feature_top20.tsv,
feature_info.tsv, targets_trrust.tsv, targets_dorothea.tsv, catalog_check.json}
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M/outputs"
OUT = RUN / "v3_tf_specificity"
TD = OUT / "task_data"
V3 = RUN / "v3_sae"
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
DOROTHEA_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv")
CHIP_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")
GENE_POS = Path("<REPO_ROOT>/projects/biotensor/data/genemanifold/gene_pos.json")
COUNT_EDGES = [1, 2, 5, 10, 20, 50, 100, 200, 500]   # same bins as v2 and the investigation prototype
D_SAE = 4928
K_TOP = 32


def mean_per_gene(idx, val, pos_code, NU):
    """Sum and count of active codes per (feature, gene code). idx/val: (N, 32)."""
    S = np.zeros(D_SAE * NU, dtype=np.float64)
    C = np.zeros(D_SAE * NU, dtype=np.float64)
    step = 100_000
    for a in range(0, idx.shape[0], step):
        b = min(idx.shape[0], a + step)
        ix = idx[a:b].astype(np.int64); vv = val[a:b]
        g = np.repeat(pos_code[a:b], K_TOP).reshape(b - a, K_TOP)
        act = vv > 0
        flat = ix[act] * NU + g[act]
        S += np.bincount(flat, weights=vv[act].astype(np.float64), minlength=D_SAE * NU)
        C += np.bincount(flat, minlength=D_SAE * NU)
    return S.reshape(D_SAE, NU), C.reshape(D_SAE, NU)


def top20(S, C, uniq):
    mpg = np.zeros_like(S, dtype=np.float32)
    nz = C > 0
    mpg[nz] = (S[nz] / C[nz]).astype(np.float32)
    lists = []
    for f in range(D_SAE):
        order = np.argsort(-mpg[f])[:20]
        lists.append([uniq[i] for i in order if mpg[f, i] > 0][:20])
    return lists, mpg


def main():
    t0 = time.time()
    TD.mkdir(parents=True, exist_ok=True)
    check = {}

    # ---------------- universe from the v3 tokens of the 500 catalog cells ----------------
    gn = json.load(open(V3 / "gene_names_v3.json"))
    gn_up = [g.upper() for g in gn]
    cells = np.load(V3 / "cells.npz")
    off = cells["offsets"]
    # a gene appears at most once per cell -> position count = detection count (checked)
    per_cell_dup = 0
    for i in range(len(off) - 1):
        seg = [g for g in gn_up[off[i]:off[i + 1]] if g != "<SPECIAL>"]
        per_cell_dup += len(seg) - len(set(seg))
    cnt = Counter(gn_up); n_special = cnt.pop("<SPECIAL>", 0)
    genes = np.array(sorted(cnt))
    gidx = {g: i for i, g in enumerate(genes)}
    c = np.array([cnt[g] for g in genes])
    pos = json.load(open(GENE_POS))
    L = {g.upper(): (v["end"] - v["start"], v["chr"]) for g, v in pos.items()}
    ln = np.array([L[g][0] if g in L else np.nan for g in genes], float)
    chrom = [L[g][1] if g in L else "" for g in genes]
    cb = np.digitize(c, COUNT_EDGES)
    lq = np.nanpercentile(ln, [33.3, 66.7])
    lb = np.where(np.isfinite(ln), np.digitize(np.nan_to_num(ln, nan=np.nanmedian(ln)), lq), 1)
    uni = pd.DataFrame({"gene": genes, "detection_count": c, "gene_length_bp": ln, "chr": chrom,
                        "count_bin": cb, "length_tertile": lb, "count_x_length_bin": cb * 10 + lb})
    uni.to_csv(TD / "gene_universe.tsv", sep="\t", index=False)
    v2u = pd.read_csv(RUN / "v2_tf_specificity/task_data/gene_universe.tsv", sep="\t", keep_default_na=False)
    m = uni[["gene", "detection_count"]].merge(v2u[["gene", "detection_count"]], on="gene", how="outer",
                                                 suffixes=("_v3", "_v2"))
    check["universe"] = {"n_genes": int(len(genes)), "n_positions": len(gn), "n_special_positions": n_special,
                         "repeated_gene_in_a_cell": int(per_cell_dup),
                         "max_detection_count": int(c.max()), "n_with_length": int(np.isfinite(ln).sum()),
                         "length_tertile_edges_bp": [float(x) for x in lq],
                         "vs_v2_universe": {"n_v2": int(len(v2u)), "n_shared": int(m.dropna().shape[0]),
                                            "only_v3": int(m.detection_count_v2.isna().sum()),
                                            "only_v2": int(m.detection_count_v3.isna().sum()),
                                            "spearman_detection_count_shared": float(
                                                m.dropna()[["detection_count_v3", "detection_count_v2"]].corr("spearman").iloc[0, 1])}}

    # ---------------- target sets restricted to the universe ----------------
    tr = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    tr["tf"] = tr.tf.str.upper(); tr["target"] = tr.target.str.upper()
    tr_u = tr[tr.target.isin(gidx)].groupby(["tf", "target"]).agg(
        mode=("mode", lambda s: ";".join(sorted(set(s)))),
        n_pmid=("pmid", lambda s: len(set(";".join(map(str, s)).split(";"))))).reset_index()
    tr_u.to_csv(TD / "targets_trrust.tsv", sep="\t", index=False)
    dor = pd.read_csv(DOROTHEA_TSV, sep="\t"); chip = pd.read_csv(CHIP_TSV, sep="\t")
    for d in (dor, chip):
        d["source"] = d.source.str.upper(); d["target"] = d.target.str.upper()
    chip_pairs = set(zip(chip.source, chip.target))
    dor["chip_flag"] = [(s, t) in chip_pairs for s, t in zip(dor.source, dor.target)]
    assert dor.chip_flag.sum() == len(chip_pairs), "every ChIP pair must be in the all-levels file"
    dor_u = dor[dor.target.isin(gidx)].rename(columns={"source": "tf"})[["tf", "target", "confidence", "chip_flag"]]
    dor_u.to_csv(TD / "targets_dorothea.tsv", sep="\t", index=False)
    check["targets"] = {"trrust_edges_in_universe": int(len(tr_u)), "trrust_tfs": int(tr_u.tf.nunique()),
                        "dorothea_edges_in_universe": int(len(dor_u)), "dorothea_tfs": int(dor_u.tf.nunique()),
                        "chip_edges_in_universe": int(dor_u.chip_flag.sum()),
                        "chip_tfs": int(dor_u[dor_u.chip_flag].tf.nunique()),
                        "GATA1_trrust_in_universe": int((tr_u.tf == "GATA1").sum()),
                        "GATA1_chip_in_universe": int(((dor_u.tf == "GATA1") & dor_u.chip_flag).sum())}

    # ---------------- source B: our forward-pass codes of the catalog cells ----------------
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    cat_rows = cells["rows"]                      # sorted rows, same order as v3_sae
    assert np.array_equal(np.sort(man[man.group == "catalog"].row.values), cat_rows)
    parts = {}
    for fpath in sorted(glob.glob(str(OUT / "cells/catalog_*.npz"))):
        z = np.load(fpath)
        offs, rws = z["offsets"], z["row"]
        tkA, ixA, vA = z["tok"], z["idx"], z["val"]
        for j, r in enumerate(rws):
            a, b = offs[j], offs[j + 1]
            parts[int(r)] = (tkA[a:b].copy(), ixA[a * K_TOP:b * K_TOP].copy(), vA[a * K_TOP:b * K_TOP].copy())
        del tkA, ixA, vA, z
    missing = [int(r) for r in cat_rows if int(r) not in parts]
    check["catalog_cells_found"] = len(parts); check["catalog_cells_missing"] = len(missing)
    if missing:
        H.write_json(TD / "catalog_check.json", check)
        raise SystemExit(f"{len(missing)} catalog cells not extracted yet")
    tokB = np.concatenate([parts[int(r)][0] for r in cat_rows])
    idxB = np.concatenate([parts[int(r)][1] for r in cat_rows]).reshape(-1, K_TOP)
    valB = np.concatenate([parts[int(r)][2] for r in cat_rows]).reshape(-1, K_TOP)
    del parts
    check["token_check"] = {"n_positions_ours": int(len(tokB)), "n_positions_v3_sae": int(len(cells["tokens_v3"])),
                            "identical": bool(len(tokB) == len(cells["tokens_v3"]) and np.array_equal(tokB, cells["tokens_v3"]))}
    print("token check", check["token_check"], flush=True)
    if not check["token_check"]["identical"]:
        H.write_json(TD / "catalog_check.json", check)
        raise SystemExit("catalog-cell tokens differ from outputs/v3_sae/cells.npz")

    # ---------------- source A: stored v3 codes ----------------
    zA = np.load(V3 / "codes/layer_05_topk.npz")
    idxA, valA = zA["idx"], zA["val"]
    assert idxA.shape == idxB.shape == (len(gn), K_TOP)
    # per-position comparison: same active feature set, value differences
    sa = np.sort(np.where(valA > 0, idxA.astype(np.int32), -1), 1)
    sb = np.sort(np.where(valB > 0, idxB.astype(np.int32), -1), 1)
    same_set = (sa == sb).all(1)
    oa = np.argsort(idxA, 1); ob = np.argsort(idxB, 1)
    va = np.take_along_axis(valA, oa, 1); vb = np.take_along_axis(valB, ob, 1)
    ia = np.take_along_axis(idxA, oa, 1); ib = np.take_along_axis(idxB, ob, 1)
    same_idx_rows = (ia == ib).all(1)
    dv = np.abs(va[same_idx_rows] - vb[same_idx_rows])
    check["codes_A_vs_B"] = {"n_positions": int(len(gn)), "frac_positions_same_active_set": float(same_set.mean()),
                             "n_positions_different_set": int((~same_set).sum()),
                             "max_abs_value_diff_same_set": float(dv.max()) if dv.size else None,
                             "median_abs_value_diff_same_set": float(np.median(dv)) if dv.size else None,
                             "max_value": float(valA.max())}
    print("codes A vs B", check["codes_A_vs_B"], flush=True)

    uniq = sorted(set(gn_up))                     # includes '<SPECIAL>', as in the deployed code
    u_idx = {g: i for i, g in enumerate(uniq)}
    NU = len(uniq)
    pos_code = np.array([u_idx[g] for g in gn_up], dtype=np.int64)
    SA, CA = mean_per_gene(idxA, valA, pos_code, NU)
    listA, mpgA = top20(SA, CA, uniq)
    SB, CB = mean_per_gene(idxB, valB, pos_code, NU)
    listB, _ = top20(SB, CB, uniq)
    del SB, CB, idxB, valB
    same_list = sum(a == b for a, b in zip(listA, listB))
    same_setAB = sum(set(a) == set(b) for a, b in zip(listA, listB))
    jac = [len(set(a) & set(b)) / max(1, len(set(a) | set(b))) for a, b in zip(listA, listB)]
    check["catalog_A_vs_B"] = {"n_features": D_SAE, "n_same_ordered_list": int(same_list), "n_same_set": int(same_setAB),
                               "mean_jaccard": float(np.mean(jac)), "min_jaccard": float(np.min(jac)),
                               "n_features_jaccard_lt_0.9": int(np.sum(np.array(jac) < 0.9)),
                               "pass": bool(np.mean(jac) > 0.99)}
    print("catalog A vs B", check["catalog_A_vs_B"], flush=True)

    freq = (CA.sum(1) / len(gn))
    rows, finfo = [], []
    for f in range(D_SAE):
        d = listA[f]
        for rk, g in enumerate(d):
            i = u_idx[g]
            rows.append((f, rk + 1, g, float(mpgA[f, i]), int(CA[f, i]), int(cnt.get(g, 0)), g in set(listB[f])))
        tg = [g for g in d if g != "<SPECIAL>"]
        finfo.append({"feature_id": f, "n_top20": len(d), "n_special_in_top20": int("<SPECIAL>" in d),
                      "median_detection_count_top20": float(np.median([cnt[g] for g in tg])) if tg else np.nan,
                      "median_length_kb_top20": float(np.nanmedian([L[g][0] / 1e3 if g in L else np.nan for g in tg])) if tg else np.nan,
                      "activation_frequency": float(freq[f]), "n_active_positions": int(CA[f].sum()),
                      "forward_rebuild_same_list": d == listB[f]})
    top = pd.DataFrame(rows, columns=["feature_id", "rank", "gene", "mean_act_when_active", "n_active_positions",
                                      "gene_detection_count", "in_forward_rebuild_top20"])
    top.to_csv(TD / "feature_top20.tsv", sep="\t", index=False)
    fi = pd.DataFrame(finfo)
    fi.to_csv(TD / "feature_info.tsv", sep="\t", index=False)
    check["catalog"] = {"source": "A (outputs/v3_sae/codes/layer_05_topk.npz)",
                        "n_features_with_list": int((fi.n_top20 > 0).sum()),
                        "n_features_20_genes": int((fi.n_top20 == 20).sum()),
                        "n_features_with_special": int(fi.n_special_in_top20.sum()),
                        "median_of_median_detection_count_top20": float(fi.median_detection_count_top20.median()),
                        "share_top20_slots_detection_count_le_5": float((top.gene_detection_count <= 5).mean())}
    v2fi = pd.read_csv(RUN / "v2_tf_specificity/task_data/feature_info.tsv", sep="\t")
    check["catalog"]["v2_deployed_median_of_median_detection_count_top20"] = float(v2fi.median_detection_count_top20.median())
    v2top = pd.read_csv(RUN / "v2_tf_specificity/task_data/feature_top20.tsv", sep="\t", keep_default_na=False)
    check["catalog"]["v2_deployed_share_top20_slots_detection_count_le_5"] = float((v2top.gene_detection_count <= 5).mean())
    check["script_sha256"] = H.sha256_file(__file__)
    check["inputs_sha256"] = {"codes_layer_05_topk.npz": H.sha256_file(V3 / "codes/layer_05_topk.npz"),
                              "gene_names_v3.json": H.sha256_file(V3 / "gene_names_v3.json"),
                              "cells.npz": H.sha256_file(V3 / "cells.npz")}
    check["wall_s"] = round(time.time() - t0, 1)
    H.write_json(TD / "catalog_check.json", check)
    print("done", check["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
