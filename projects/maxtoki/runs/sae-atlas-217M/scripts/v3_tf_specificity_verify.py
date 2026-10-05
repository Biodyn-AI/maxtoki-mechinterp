"""V3-2: independent re-derivation of key numbers (second way).

1. Model side (MPS, 12 cells: 6 GATA1 knockdown, 6 reference controls). Per-cell feature means
   recomputed through a DIFFERENT path: full forward pass with output_hidden_states=True, take
   hidden_states[5] (no hook), encode with the v3 SAE loaded directly with topk_sae.TopKSAE (not
   hooks_v2), mean over gene tokens. Tokens rebuilt from counts with inputs_v3.tokenize_rows and
   checked. Compared with the stored per-cell means used in the statistics.
2. Statistic side (CPU): GATA1's responding features, K and the best ChIP / TRRUST overlap at each
   cut-off recomputed from the written task-data tables only (responding_features.tsv,
   feature_top20.tsv, targets_dorothea.tsv, targets_trrust.tsv), matching genes by name.
   Compared with tf_results.csv.
Writes outputs/v3_tf_specificity/checks/verify.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
import inputs_v3 as iv3  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M/outputs"
OUT = RUN / "v3_tf_specificity"
TD = OUT / "task_data"
CUTOFFS = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0]


def model_side(rep):
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    rows = list(man[(man.group == "kd") & (man.tf == "GATA1")].row.values[:6]) + list(man[man.group == "ref"].row.values[:6])
    cells, kept, info, counts = iv3.tokenize_rows("k562", rows, 2048, return_counts=True)
    assert kept == [int(r) for r in rows]
    vm = iv3.get_var_map("k562")
    enc = iv3.assert_encoding_batch(counts, cells, vm, 2048, label="verify 12 cells")
    gb = H.check_memory(3.0)
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(str(PROJ / "setup/MaxToki-217M-HF"), torch_dtype=torch.float32,
                                             attn_implementation="eager").eval().to("mps")
    ck = torch.load(RUN / "v3_sae/layer_05/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ck["config"]["d_model"], ck["config"]["d_sae"], ck["config"]["k"])
    sae.W_enc.weight.data = ck["W_enc_weight"]; sae.W_enc.bias.data = ck["W_enc_bias"]; sae.W_dec.weight.data = ck["W_dec_weight"]
    sae.eval(); mu = ck["mu"].float()
    z = np.load(TD / "cell_feature_means.npz")
    rix = {int(r): i for i, r in enumerate(z["rows"])}
    MG = z["mean_gene"]
    diffs, toks_same = [], []
    # stored tokens of these cells
    import glob
    stored_tok = {}
    for fpath in sorted(glob.glob(str(OUT / "cells/summ_*.npz"))):
        d = np.load(fpath)
        for j, r in enumerate(d["row"]):
            if int(r) in kept:
                stored_tok[int(r)] = d["tokens"][d["tok_offsets"][j]:d["tok_offsets"][j + 1]]
    for r, cell in zip(kept, cells):
        ids = torch.from_numpy(cell.token_ids[None, :].astype(np.int64)).to("mps")
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states[5][0].cpu()
            zz = sae.encode(hs, mu)
        T = zz.shape[0]
        mg = zz[1:T - 1].mean(0).numpy()
        diffs.append(float(np.abs(mg - MG[rix[r]]).max()))
        toks_same.append(bool(np.array_equal(stored_tok[r], cell.token_ids.astype(np.int32))))
        H.free_device_cache()
    rep["model_side"] = {"rows": [int(r) for r in kept], "encoding_check": enc, "mem_gb_before_load": round(gb, 2),
                         "tokens_equal_stored": toks_same,
                         "max_abs_diff_per_cell_mean": diffs, "max_over_cells": max(diffs),
                         "max_stored_value_these_cells": float(np.abs(MG[[rix[r] for r in kept]]).max()),
                         "path": "LlamaForCausalLM(output_hidden_states=True).hidden_states[5] + TopKSAE.encode (no hooks_v2)"}
    print("model side", rep["model_side"]["max_over_cells"], toks_same, flush=True)


def stat_side(rep):
    rf = pd.read_csv(TD / "responding_features.tsv", sep="\t")
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    trr = pd.read_csv(TD / "targets_trrust.tsv", sep="\t", keep_default_na=False)
    chip = set(dor[(dor.tf == "GATA1") & (dor.chip_flag.astype(str).str.lower() == "true")].target)
    trs = set(trr[trr.tf == "GATA1"].target)
    lists = top.groupby("feature_id").gene.apply(list).to_dict()
    res = pd.read_csv(OUT / "tf_results.csv")
    g = rf[rf.tf == "GATA1"]
    out = {}
    ok = True
    for c in CUTOFFS:
        feats = g[(g.bh_q < 0.05) & (g.delta_mean.abs() > c)].feature_id.values
        for db, Sx in [("ChIP", chip), ("TRRUST", trs)]:
            ovs = [sum(x in Sx for x in lists[f]) for f in feats]
            M = max(ovs) if ovs else 0
            r = res[(res.tf == "GATA1") & (res.cutoff == c) & (res.db == db)].iloc[0]
            same = (len(feats) == int(r.K)) and (M == int(r.M_obs))
            ok &= same
            out[f"{c}|{db}"] = {"K": int(len(feats)), "M": int(M), "K_results": int(r.K), "M_results": int(r.M_obs), "same": bool(same)}
    rep["stat_side_GATA1"] = {"all_same": bool(ok), "by_cutoff": out}
    print("stat side all same:", ok, flush=True)


def main():
    t0 = time.time()
    rep = {}
    stat_side(rep)
    model_side(rep)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["script_sha256"] = H.sha256_file(__file__)
    H.write_json(OUT / "checks/verify.json", rep)


if __name__ == "__main__":
    main()
