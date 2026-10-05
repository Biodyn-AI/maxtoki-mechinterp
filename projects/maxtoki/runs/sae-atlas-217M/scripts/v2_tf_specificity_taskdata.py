"""D3 (revision): gene universe, target sets, and the layer-5 top-20 catalog (rebuilt and checked).

Inputs
  * deployed catalog  outputs/phase2/layer_05/feature_catalog.json (top-20 genes per feature)
  * deployed universe outputs/phase0/layer_05/gene_names.json (gene at each of the 1,019,996
    token positions of the 500 catalog control cells, '<special>' = <bos>/<eos>)
  * sparse layer-5 SAE codes of the same 500 cells, written by v2_tf_specificity_extract.py
    (group 'catalog'; input of block 5 = hidden_states[5])
  * TRRUST, DoRothEA (all levels) and the DoRothEA ChIP-seq subset; gene_pos.json for gene length

Checks (written to task_data/catalog_check.json)
  1. Token check: our tokenisation of the 500 catalog cells gives exactly the deployed
     gene_names.json sequence.
  2. Catalog check: our rebuilt top-20 (mean activation over the positions where the feature
     is active, top 20 genes, no minimum count) matches the deployed catalog.

Outputs: outputs/v2_tf_specificity/task_data/{gene_universe.tsv, feature_top20.tsv,
feature_info.tsv, targets_trrust.tsv, targets_dorothea.tsv, catalog_check.json}
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402
from dataset_loader import resolve as load_ds, SYM2ENS_PKL  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M/outputs"
OUT = RUN / "v2_tf_specificity"
TD = OUT / "task_data"
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
DOROTHEA_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv")
CHIP_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")
GENE_POS = Path("<REPO_ROOT>/projects/biotensor/data/genemanifold/gene_pos.json")
COUNT_EDGES = [1, 2, 5, 10, 20, 50, 100, 200, 500]   # same bins as the investigation prototype (null2/null3.py)
D_SAE = 4928


def main():
    t0 = time.time()
    TD.mkdir(parents=True, exist_ok=True)
    check = {}

    # ---------------- universe, detection counts, gene length ----------------
    gn = json.load(open(RUN / "phase0/layer_05/gene_names.json"))
    gn_up = [g.upper() for g in gn]
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
    check["universe"] = {"n_genes": int(len(genes)), "n_positions": len(gn), "n_special_positions": n_special,
                         "max_detection_count": int(c.max()), "n_with_length": int(np.isfinite(ln).sum()),
                         "length_tertile_edges_bp": [float(x) for x in lq]}

    # ---------------- target sets restricted to the universe ----------------
    tr = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    tr["tf"] = tr.tf.str.upper(); tr["target"] = tr.target.str.upper()
    tr_u = tr[tr.target.isin(gidx)].groupby(["tf", "target"]).agg(
        mode=("mode", lambda s: ";".join(sorted(set(s)))), n_pmid=("pmid", lambda s: len(set(";".join(map(str, s)).split(";"))))).reset_index()
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

    # ---------------- token check + catalog rebuild ----------------
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    cat_rows = np.sort(man[man.group == "catalog"].row.values)
    files = sorted(glob.glob(str(OUT / "cells/catalog_*.npz")))
    parts = {}
    for fpath in files:
        z = np.load(fpath)
        # read each array ONCE (an NpzFile re-reads the whole array on every access) and
        # copy the slices, so no view keeps a large array alive
        off, rws = z["offsets"], z["row"]
        tkA, ixA, vA = z["tok"], z["idx"], z["val"]
        for j, r in enumerate(rws):
            a, b = off[j], off[j + 1]
            parts[int(r)] = (tkA[a:b].copy(), ixA[a * 32:b * 32].copy(), vA[a * 32:b * 32].copy())
        del tkA, ixA, vA, z
    missing = [int(r) for r in cat_rows if int(r) not in parts]
    check["catalog_cells_found"] = len(parts); check["catalog_cells_missing"] = len(missing)
    if missing:
        H.write_json(TD / "catalog_check.json", check)
        raise SystemExit(f"{len(missing)} catalog cells not extracted yet")

    ds = load_ds("k562")
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    tok = MaxTokiTokenizer()
    vi, vt, _ = tok.make_var_mapping([sym2ens.get(s) for s in ds.var_symbols])
    token_to_gene = {}
    for v_i, t_i in zip(vi, vt):                 # same dict-overwrite order as full_12layer_pipeline.py
        token_to_gene[int(t_i)] = ds.var_symbols[v_i].upper()
    seq = []
    for r in cat_rows:
        seq.extend(token_to_gene.get(int(t), "<special>").upper() for t in parts[int(r)][0])
    n_mis = sum(1 for a, b in zip(seq, gn_up) if a != b)
    check["token_check"] = {"n_positions_ours": len(seq), "n_positions_deployed": len(gn_up),
                            "n_mismatched_positions": int(n_mis + abs(len(seq) - len(gn_up))),
                            "pass": bool(len(seq) == len(gn_up) and n_mis == 0)}
    print("token check", check["token_check"], flush=True)

    uniq = sorted(set(gn_up))                    # includes '<SPECIAL>', as in the deployed code
    u_idx = {g: i for i, g in enumerate(uniq)}
    NU = len(uniq)
    pos_code = np.array([u_idx[g] for g in gn_up], dtype=np.int64)
    S = np.zeros(D_SAE * NU, dtype=np.float64)
    C = np.zeros(D_SAE * NU, dtype=np.float64)
    p0 = 0
    for r in cat_rows:
        tk, ix, vv = parts[int(r)]
        T = len(tk)
        gcode = np.repeat(pos_code[p0:p0 + T], 32)
        act = vv > 0
        flat = ix.astype(np.int64)[act] * NU + gcode[act]
        S += np.bincount(flat, weights=vv[act].astype(np.float64), minlength=D_SAE * NU)
        C += np.bincount(flat, minlength=D_SAE * NU)
        p0 += T
    S = S.reshape(D_SAE, NU); C = C.reshape(D_SAE, NU)
    mpg = np.zeros_like(S, dtype=np.float32)
    nz = C > 0
    mpg[nz] = (S[nz] / C[nz]).astype(np.float32)

    cat = json.load(open(RUN / "phase2/layer_05/feature_catalog.json"))
    dep = {x["feature_id"]: [g.upper() for g in x["top20_genes"]] for x in cat}
    dep_freq = {x["feature_id"]: x["activation_frequency"] for x in cat}
    same_set = same_list = 0
    jacc = []
    rows = []
    finfo = []
    for f in range(D_SAE):
        order = np.argsort(-mpg[f])[:20]
        ours = [uniq[i] for i in order if mpg[f, i] > 0][:20]
        d = dep.get(f, [])
        same_set += set(ours) == set(d); same_list += ours == d
        jacc.append(len(set(ours) & set(d)) / max(1, len(set(ours) | set(d))))
        for rk, g in enumerate(d):
            i = u_idx[g]
            rows.append((f, rk + 1, g, float(mpg[f, i]), int(C[f, i]),
                         int(cnt.get(g, 0)), g in set(ours)))
        tg = [g for g in d if g != "<SPECIAL>"]
        finfo.append({"feature_id": f, "n_top20": len(d), "n_special_in_top20": int("<SPECIAL>" in d),
                      "median_detection_count_top20": float(np.median([cnt[g] for g in tg])) if tg else np.nan,
                      "median_length_kb_top20": float(np.nanmedian([L[g][0] / 1e3 if g in L else np.nan for g in tg])) if tg else np.nan,
                      "activation_frequency_deployed": dep_freq.get(f, np.nan),
                      "n_active_positions_rebuilt": int(C[f].sum()),
                      "rebuilt_top20_same_set": set(ours) == set(d)})
    top = pd.DataFrame(rows, columns=["feature_id", "rank", "gene", "mean_act_when_active_rebuilt",
                                      "n_active_positions_rebuilt", "gene_detection_count", "in_rebuilt_top20"])
    top.to_csv(TD / "feature_top20.tsv", sep="\t", index=False)
    pd.DataFrame(finfo).to_csv(TD / "feature_info.tsv", sep="\t", index=False)
    check["catalog_check"] = {"n_features": D_SAE, "n_same_set": int(same_set), "n_same_ordered_list": int(same_list),
                              "mean_jaccard": float(np.mean(jacc)), "min_jaccard": float(np.min(jacc)),
                              "n_features_jaccard_lt_0.9": int(np.sum(np.array(jacc) < 0.9)),
                              "pass": bool(np.mean(jacc) > 0.99)}
    print("catalog check", check["catalog_check"], flush=True)
    check["script_sha256"] = H.sha256_file(__file__)
    check["wall_s"] = round(time.time() - t0, 1)
    H.write_json(TD / "catalog_check.json", check)


if __name__ == "__main__":
    main()
