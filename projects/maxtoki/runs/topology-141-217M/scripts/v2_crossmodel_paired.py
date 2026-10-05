"""v2 cross-model alignment (revision item D10), step 3: paired differences
Geneformer minus scGPT (corrected lookup), on the SAME genes and the SAME gene resamples.

Per topology panel (the Geneformer and scGPT tables cover the same genes in the same order;
checked below):
  * gene-pair cosine Pearson: gene bootstrap with replacement, N_PAIR_BOOT draws, copy pairs dropped;
  * held-out (out-of-bag) canonical r (mean of 10) and held-out top-1: N_PAIR_OOB draws; fit on the
    resample, score on genes the resample did not draw.
Also the gene-pair Pearson difference on the 1,500 spectral-geometry HVGs.
Percentile 2.5/97.5 CIs of the difference; unit = gene. CPU only; no forward pass.
Output: outputs/v2_crossmodel/paired_differences.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning

import v2_crossmodel_common as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_PAIR_BOOT = 2000
N_PAIR_OOB = 500
SEED = 20261201
OUT_F = C.OUT / "paired_differences.json"


def oob_once(A_full, B_full, idx, oob, d, k):
    pA = PCA(n_components=d, random_state=42).fit(A_full[idx])
    pB = PCA(n_components=d, random_state=42).fit(B_full[idx])
    A_in, A_out = pA.transform(A_full[idx]), pA.transform(A_full[oob])
    B_all = pB.transform(B_full); B_in, B_out = pB.transform(B_full[idx]), B_all[oob]
    cca = CCA(n_components=k, max_iter=200).fit(A_in, B_in)
    a_o, b_o = cca.transform(A_out, B_out)
    r = float(np.mean([np.corrcoef(a_o[:, i], b_o[:, i])[0, 1] for i in range(k)]))
    R = C.procrustes_R(A_in.astype(np.float64), B_in.astype(np.float64))
    sim = C.normalise(A_out.astype(np.float64) @ R) @ C.normalise(B_all.astype(np.float64)).T
    return r, float((np.argmax(sim, axis=1) == oob).mean())


def main():
    t0 = time.time()
    deadline = t0 + float(os.environ.get("BUDGET_S", "480"))
    res = json.loads(OUT_F.read_text()) if OUT_F.exists() else {}
    z = C.load_prepared()
    for di, dom in enumerate(C.DOMAINS):
        key = f"panel_{dom}"
        if key in res and res[key].get("complete"):
            continue
        same = bool((z[f"{dom}__gf__ens"] == z[f"{dom}__sc__ens"]).all())
        A = z[f"{dom}__gf__mt"].astype(np.float64)
        assert same and np.array_equal(A, z[f"{dom}__sc__mt"].astype(np.float64))
        G = z[f"{dom}__gf__other"].astype(np.float64); S = z[f"{dom}__sc__fixed"].astype(np.float64)
        n = A.shape[0]; d = C.d_align_for(n); k = min(C.TOP_K_CCA, d)
        Sa, Sg, Ss = C.cos_matrix(A), C.cos_matrix(G), C.cos_matrix(S)
        rng = np.random.default_rng(SEED + di)
        idxs = rng.integers(0, n, size=(N_PAIR_BOOT, n))
        pdiff = np.array([C.offdiag_pearson(Sa, Sg, ix) - C.offdiag_pearson(Sa, Ss, ix) for ix in idxs])
        obs_p = C.offdiag_pearson(Sa, Sg) - C.offdiag_pearson(Sa, Ss)
        part = C.OUT / "jobs" / f"paired_oob_{dom}.npy"
        done = list(np.load(part)) if part.exists() else []
        for b in range(len(done), N_PAIR_OOB):
            ix = idxs[b]; oob = np.setdiff1d(np.arange(n), ix)
            rg, tg = oob_once(A, G, ix, oob, d, k)
            rs, ts = oob_once(A, S, ix, oob, d, k)
            done.append([rg, rs, tg, ts])
            if (b + 1) % 50 == 0:
                np.save(part, np.array(done)); print(f"  {dom} oob {b + 1}/{N_PAIR_OOB}", flush=True)
                if time.time() > deadline and b + 1 < N_PAIR_OOB:
                    print("budget reached; rerun to resume"); OUT_F.write_text(json.dumps(res, indent=2)); return
        np.save(part, np.array(done))
        o = np.array(done)
        res[key] = {
            "complete": True, "same_genes_same_order": same, "n_genes": n, "seed": SEED + di,
            "pearson_gf_minus_scgpt": {"observed": obs_p, "ci95": C.pct_ci(pdiff), "n_boot": N_PAIR_BOOT,
                                       "frac_draws_le_0": float((pdiff <= 0).mean())},
            "heldout_cca_gf_minus_scgpt": {"mean": float((o[:, 0] - o[:, 1]).mean()), "ci95": C.pct_ci(o[:, 0] - o[:, 1]),
                                           "n_boot": N_PAIR_OOB, "frac_draws_le_0": float(((o[:, 0] - o[:, 1]) <= 0).mean())},
            "heldout_top1_gf_minus_scgpt": {"mean": float((o[:, 2] - o[:, 3]).mean()), "ci95": C.pct_ci(o[:, 2] - o[:, 3]),
                                            "n_boot": N_PAIR_OOB, "frac_draws_le_0": float(((o[:, 2] - o[:, 3]) <= 0).mean())},
            "method": "same gene resamples for both comparisons; Pearson drops copy pairs; held-out values are "
                      "out-of-bag (fit on resample, score on undrawn genes); percentile CI of the difference",
        }
        OUT_F.write_text(json.dumps(res, indent=2))
        print(dom, json.dumps(res[key]), flush=True)

    if "hvg1500_pearson_gf_minus_scgpt" not in res:
        sp = C.PROJ / "runs/spectral-geometry-217M"
        g = pd.read_csv(sp / "outputs/phase0/gene_features.csv")
        with safe_open(str(C.MAXTOKI_ST), framework="pt") as f:
            mt = f.get_tensor("model.embed_tokens.weight").float().numpy()
        with safe_open(str(C.GENEFORMER_DIR / "model.safetensors"), framework="pt") as f:
            gf = f.get_tensor("bert.embeddings.word_embeddings.weight").float().numpy()
        tok = pickle.load(open(C.GENEFORMER_TOKEN_DICT, "rb"))
        sc = torch.load(C.SCGPT_PT, map_location="cpu", weights_only=False)
        voc = sc["vocab"]; E = sc["normed_embeddings"].float().numpy()
        ens = g["ensembl_id"].astype(str).tolist(); syms = g["symbol"].astype(str).tolist()
        mt_ids = g["maxtoki_token_id"].astype(int).to_numpy()
        g_ids = np.array([tok.get(e, -1) for e in ens]); s_ids = np.array([int(voc[s]) if s in voc else -1 for s in syms])
        v = (g_ids >= 0) & (s_ids >= 0)
        Sa = C.cos_matrix(mt[mt_ids[v]]); Sg = C.cos_matrix(gf[g_ids[v]]); Ss = C.cos_matrix(E[s_ids[v]])
        n = Sa.shape[0]
        # weighted-pair formula (as in spectral v2_crossmodel_pearson_ci.py) for speed
        def prep(X, Y):
            X = X.copy(); Y = Y.copy(); np.fill_diagonal(X, 0); np.fill_diagonal(Y, 0)
            return X, Y, X * X, Y * Y, X * Y
        def wp(P, c):
            X, Y, XX, YY, XY = P
            W = (c.sum() ** 2 - (c * c).sum()) / 2
            sx, sy = c @ X @ c / 2, c @ Y @ c / 2
            sxx, syy, sxy = c @ XX @ c / 2, c @ YY @ c / 2, c @ XY @ c / 2
            return float((sxy / W - sx * sy / W ** 2) / np.sqrt((sxx / W - (sx / W) ** 2) * (syy / W - (sy / W) ** 2)))
        Pg, Ps = prep(Sa, Sg), prep(Sa, Ss)
        rng = np.random.default_rng(SEED + 10)
        diffs = []
        for _ in range(N_PAIR_BOOT):
            c = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
            diffs.append(wp(Pg, c) - wp(Ps, c))
        ones = np.ones(n)
        res["hvg1500_pearson_gf_minus_scgpt"] = {
            "n_genes": int(n), "gf": wp(Pg, ones), "scgpt_fixed": wp(Ps, ones),
            "observed_diff": wp(Pg, ones) - wp(Ps, ones), "ci95": C.pct_ci(diffs), "n_boot": N_PAIR_BOOT,
            "seed": SEED + 10}
        OUT_F.write_text(json.dumps(res, indent=2))
    res["wall_seconds_last_run"] = time.time() - t0
    OUT_F.write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("hvg1500_pearson_gf_minus_scgpt"), indent=1))


if __name__ == "__main__":
    main()
