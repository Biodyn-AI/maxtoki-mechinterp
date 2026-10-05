"""Independent verification of revision item D10 (cross-model alignment).

Written by a second agent to re-check V2_CROSSMODEL_REPORT.md. It does NOT import the D10
helper module. It rebuilds every panel table from the raw embedding files and uses its own
code paths:
  * PCA by numpy SVD (exact; D10 used sklearn PCA, which picks the randomized solver here);
  * CCA by the exact QR + SVD method (D10 used sklearn's iterative CCA);
  * Procrustes by scipy.linalg.orthogonal_procrustes;
  * gene-pair Pearson bootstrap by direct indexing (D10 used a weighted-pair formula for 1,500 HVGs).

Parts (each writes outputs/v2_crossmodel/verify/<part>.json; rerun skips finished parts):
  lookup     scGPT vocab vs the checkpoint's own vocab.json; rebuild panel rows; compare to D10's npz;
             count rows the deployed lookup gets right.
  insample   in-sample exact CCA (mean of top 10), Pearson, in-sample top-1; chance from
             N_PERM gene-correspondence permutations (CCA refitted, rotation refitted).
  heldout    5-fold held-out exact CCA (mean of 10) and held-out top-1 (N_CV_REP repeats),
             chance from N_CV_PERM permutations; OOB gene bootstrap (N_OOB draws) for CIs.
  pearson    0.382 on 1,500 HVGs: gene bootstrap with replacement by direct indexing (N_PB draws);
             panel Pearson: leave-one-gene-out jackknife SE (second method next to D10's bootstrap).
  bootcheck  why the in-sample CCA has no valid with-replacement bootstrap: exact CCA under gene
             resampling, real pairing vs whole-gene permutation, plus a pure-noise control.
  pcasolver  deployed in-sample CCA / top-1 with sklearn's randomized PCA (as deployed) vs exact PCA.
  permorigin where D10's 'permcheck' noise comes from (randomized PCA solver vs CCA).
  rawscgpt   scGPT raw table (before enc_norm) vs the normed table, correct lookup.
  hvgscgpt   scGPT on the 1,500 HVGs, rows from the checkpoint's vocab.json vs the deployed rule.
  csvcheck   every number in fig_crossmodel.csv against the job JSONs it was built from.

Run: python v2_crossmodel_verify.py [part1,part2,...]   (no argument = all parts)
The held-out part uses an exact Gram-matrix PCA (pca_gram) for speed; other parts use SVD PCA.
Counts were kept small because the machine was heavily loaded during verification.
CPU only. No model forward pass. Seeds fixed below.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys
sys.dont_write_bytecode = True

import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.linalg import orthogonal_procrustes

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/topology-141-217M"
D10 = RUN / "outputs/v2_crossmodel"
OUT = D10 / "verify"
GF_DIR = Path("<HF_CACHE>/hub/models--ctheodoris--Geneformer"
              "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M")
GF_TOK = GF_DIR.parent / "geneformer/token_dictionary_gc104M.pkl"
MT_ST = PROJ / "setup/MaxToki-217M-HF/model.safetensors"
SC_PT = Path("<DATA_ROOT>/biodyn-work/subproject_53_scgpt_gpl_replication/"
             "embeddings/scgpt_whole_human_gene_embeddings.pt")
SC_VOCAB_JSON = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/"
                     "scGPT_checkpoints/whole-human/vocab.json")
HVG = PROJ / "runs/spectral-geometry-217M/outputs/phase0/gene_features.csv"
DOMAINS = ["lung", "immune", "external_lung"]
D = 30
K = 10
N_PERM = 300
N_CV_REP = 3
N_CV_PERM = 20
N_OOB = 100  # kept small: the machine was heavily loaded (load average > 170) during verification
N_PB = 1000
N_BC = 100
SEED = 7_000_000  # all verification seeds derive from this; different from every D10 seed


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tensor(path, name):
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def pca_exact(X_fit, X_apply=None, d=D):
    mu = X_fit.mean(0)
    _, _, Vt = np.linalg.svd(X_fit - mu, full_matrices=False)
    W = Vt[:d].T
    return (X_fit - mu) @ W, (None if X_apply is None else (X_apply - mu) @ W)


def cca_fit(A, B):
    """Exact CCA. Returns canonical correlations and weights (for centred A, B)."""
    ma, mb = A.mean(0), B.mean(0)
    Qa, Ra = np.linalg.qr(A - ma); Qb, Rb = np.linalg.qr(B - mb)
    U, s, Vt = np.linalg.svd(Qa.T @ Qb)
    Wa = np.linalg.solve(Ra, U); Wb = np.linalg.solve(Rb, Vt.T)
    return s, Wa, Wb, ma, mb


def cca_mean_exact(A, B, k=K):
    return float(np.mean(cca_fit(A, B)[0][:k]))


def cosn(X):
    X = X.astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n


def top1(Aq, Bc, R, true_idx):
    sim = cosn(Aq @ R) @ cosn(Bc).T
    return float((np.argmax(sim, 1) == true_idx).mean())


def build_panels():
    """Rebuild the panel rows from the raw tables, independently of D10's npz."""
    mt = tensor(MT_ST, "model.embed_tokens.weight")
    gf = tensor(GF_DIR / "model.safetensors", "bert.embeddings.word_embeddings.weight")
    tok = pickle.load(open(GF_TOK, "rb"))
    sc = torch.load(SC_PT, map_location="cpu", weights_only=False)
    E = sc["normed_embeddings"].float().numpy()
    vj = json.loads(SC_VOCAB_JSON.read_text())  # the checkpoint's own symbol -> token id map
    dep_pos = {s.upper(): i for i, s in enumerate(sc["vocab"])}  # deployed rule
    out = {}
    for dom in DOMAINS:
        g = pd.read_csv(RUN / "outputs/phase0" / dom / "gene_features.csv")
        ens = g["ensembl_id"].astype(str).tolist(); sy = g["symbol"].astype(str).tolist()
        mti = g["maxtoki_token_id"].astype(int).to_numpy()
        gfi = np.array([tok[e] for e in ens])
        sci = np.array([vj[s] for s in sy])
        dpi = np.array([dep_pos[s.upper()] for s in sy])
        out[dom] = {"mt": mt[mti].astype(np.float64), "gf": gf[gfi].astype(np.float64),
                    "scgpt_fixed": E[sci].astype(np.float64), "scgpt_deployed": E[dpi].astype(np.float64),
                    "sym": sy, "sci": sci, "dpi": dpi, "gfi": gfi, "mti": mti}
    return out, sc, vj, mt, gf, E, tok


def part_lookup(P, sc, vj):
    voc = sc["vocab"]
    keys = list(voc.keys())
    res = {"pt_vocab_type": type(voc).__name__, "n_vocab": len(voc),
           "pt_vocab_equals_checkpoint_vocab_json": int(sum(1 for k in vj if voc.get(k) == vj[k])),
           "n_checkpoint_vocab_json": len(vj),
           "dict_position_equals_value": int(sum(1 for i, k in enumerate(keys) if voc[k] == i))}
    z = np.load(D10 / "embeddings_subset.npz")
    for dom in DOMAINS:
        p = P[dom]
        res[dom] = {
            "n": int(len(p["sym"])),
            "deployed_rows_that_are_the_right_gene": int((p["dpi"] == p["sci"]).sum()),
            "gf_id_equals_maxtoki_id": int((p["gfi"] == p["mti"]).sum()),
            "maxdiff_vs_D10_npz": {
                "mt": float(np.abs(p["mt"] - z[f"{dom}__gf__mt"]).max()),
                "gf": float(np.abs(p["gf"] - z[f"{dom}__gf__other"]).max()),
                "scgpt_fixed": float(np.abs(p["scgpt_fixed"] - z[f"{dom}__sc__fixed"]).max()),
                "scgpt_deployed": float(np.abs(p["scgpt_deployed"] - z[f"{dom}__sc__deployed"]).max())},
        }
    return res


def part_insample(P):
    res = {}
    for di, dom in enumerate(DOMAINS):
        A_full = P[dom]["mt"]; A, _ = pca_exact(A_full)
        Sa = cosn(A_full) @ cosn(A_full).T
        n = A.shape[0]; iu = np.triu_indices(n, 1)
        for mi, m in enumerate(["gf", "scgpt_fixed", "scgpt_deployed"]):
            B_full = P[dom][m]; B, _ = pca_exact(B_full)
            Sb = cosn(B_full) @ cosn(B_full).T
            obs_c = cca_mean_exact(A, B)
            R, _ = orthogonal_procrustes(A, B)
            obs_t = top1(A, B, R, np.arange(n))
            obs_p = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
            rng = np.random.default_rng(SEED + 100 * di + 10 * mi + 1)
            nc, nt = [], []
            for _ in range(N_PERM):
                q = rng.permutation(n)
                nc.append(cca_mean_exact(A, B[q]))
                Rq, _ = orthogonal_procrustes(A, B[q])
                nt.append(top1(A, B[q], Rq, np.arange(n)))
            nc = np.array(nc); nt = np.array(nt)
            d10 = json.loads((D10 / "jobs" / f"{m}__{dom}__obs.json").read_text())
            res[f"{m}/{dom}"] = {
                "n": n, "cca_exact_mean10": obs_c, "cca_D10_sklearn": d10["cca_mean_r"],
                "cca_chance_exact_mean": float(nc.mean()), "cca_chance_exact_sd": float(nc.std(ddof=1)),
                "cca_chance_exact_p95": float(np.percentile(nc, 95)),
                "cca_p_one_sided": float((1 + (nc >= obs_c).sum()) / (1 + N_PERM)),
                "pearson": obs_p, "pearson_D10": d10["pairwise_pearson"],
                "top1_insample": obs_t, "top1_D10": d10["top1"],
                "top1_chance_refit_mean": float(nt.mean()), "top1_chance_refit_p95": float(np.percentile(nt, 95)),
                "top1_p_one_sided": float((1 + (nt >= obs_t).sum()) / (1 + N_PERM)),
                "n_perm": N_PERM}
            print(dom, m, {k: round(v, 4) for k, v in res[f"{m}/{dom}"].items() if isinstance(v, float)}, flush=True)
    return res


def pca_gram(X_fit, X_apply, d=D):
    """Exact PCA through the n x n Gram matrix (faster than SVD of n x p when n << p)."""
    mu = X_fit.mean(0); Xc = X_fit - mu
    lam, U = np.linalg.eigh(Xc @ Xc.T)
    top = np.argsort(lam)[::-1][:d]
    W = Xc.T @ U[:, top] / np.sqrt(lam[top])
    return Xc @ W, (X_apply - mu) @ W


def heldout_once(A_full, B_full, tr, te):
    A_tr, A_te = pca_gram(A_full[tr], A_full[te])
    B_tr, B_all = pca_gram(B_full[tr], B_full)
    s, Wa, Wb, ma, mb = cca_fit(A_tr, B_tr)
    a = (A_te - ma) @ Wa[:, :K]; b = (B_all[te] - mb) @ Wb[:, :K]
    r = float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(K)]))
    R, _ = orthogonal_procrustes(A_tr, B_tr)
    t = float((np.argmax(cosn(A_te @ R) @ cosn(B_all).T, 1) == te).mean())
    return r, t


def cv_pass(A_full, B_full, rng):
    n = A_full.shape[0]; order = rng.permutation(n); folds = np.array_split(order, 5)
    rs, hits = [], 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        r, t = heldout_once(A_full, B_full, tr, f)
        rs.append(r); hits += t * len(f)
    return float(np.mean(rs)), hits / n


def part_heldout(P, deadline=None):
    pf = OUT / "heldout_partial.json"
    res = json.loads(pf.read_text()) if pf.exists() else {}
    for di, dom in enumerate(DOMAINS):
        A_full = P[dom]["mt"]; n = A_full.shape[0]
        for mi, m in enumerate(["gf", "scgpt_fixed"]):
            if f"{m}/{dom}" in res:
                continue
            if deadline is not None and time.time() > deadline:
                return None
            B_full = P[dom][m]
            rng = np.random.default_rng(SEED + 1000 + 100 * di + 10 * mi)
            cv = np.array([cv_pass(A_full, B_full, rng) for _ in range(N_CV_REP)])
            rngp = np.random.default_rng(SEED + 2000 + 100 * di + 10 * mi)
            nul = np.array([cv_pass(A_full, B_full[rngp.permutation(n)], rngp) for _ in range(N_CV_PERM)])
            rngo = np.random.default_rng(SEED + 3000 + 100 * di + 10 * mi)
            oob = []
            for _ in range(N_OOB):
                idx = rngo.integers(0, n, n); te = np.setdiff1d(np.arange(n), idx)
                oob.append(heldout_once(A_full, B_full, idx, te))
            oob = np.array(oob)
            d10o = json.loads((D10 / "jobs" / f"{m}__{dom}__oob.json").read_text())
            d10c = json.loads((D10 / "jobs" / f"{m}__{dom}__cv.json").read_text())
            res[f"{m}/{dom}"] = {
                "cv_cca_mean10": float(cv[:, 0].mean()), "cv_cca_D10": d10c["cv_cca_mean_r"],
                "cv_top1": float(cv[:, 1].mean()), "cv_top1_D10": d10c["cv_top1"],
                "cv_chance_cca_mean": float(nul[:, 0].mean()), "cv_chance_cca_p95": float(np.percentile(nul[:, 0], 95)),
                "cv_chance_top1_mean": float(nul[:, 1].mean()), "cv_chance_top1_p95": float(np.percentile(nul[:, 1], 95)),
                "oob_cca_mean": float(oob[:, 0].mean()),
                "oob_cca_ci95": [float(np.percentile(oob[:, 0], 2.5)), float(np.percentile(oob[:, 0], 97.5))],
                "oob_cca_D10": [d10o["oob_cca_mean_r"]["mean"]] + d10o["oob_cca_mean_r"]["ci95"],
                "oob_top1_mean": float(oob[:, 1].mean()),
                "oob_top1_ci95": [float(np.percentile(oob[:, 1], 2.5)), float(np.percentile(oob[:, 1], 97.5))],
                "oob_top1_D10": [d10o["oob_top1"]["mean"]] + d10o["oob_top1"]["ci95"],
                "n_cv_repeats": N_CV_REP, "n_cv_perm": N_CV_PERM, "n_oob": N_OOB}
            print(dom, m, json.dumps(res[f"{m}/{dom}"]), flush=True)
            pf.write_text(json.dumps(res, indent=2))
    return res


def boot_pearson_direct(Sa, Sb, rng, n_boot):
    n = Sa.shape[0]; iu = np.triu_indices(n, 1); out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gi = idx[iu[0]]; gj = idx[iu[1]]; keep = gi != gj
        out.append(float(np.corrcoef(Sa[gi[keep], gj[keep]], Sb[gi[keep], gj[keep]])[0, 1]))
    return np.array(out)


def jackknife_pearson(Sa, Sb):
    """Exact leave-one-gene-out values from pair sums (no weighting code shared with D10)."""
    n = Sa.shape[0]
    X = Sa.copy(); Y = Sb.copy(); np.fill_diagonal(X, 0); np.fill_diagonal(Y, 0)
    tot = lambda M: M.sum() / 2
    rows = lambda M: M.sum(1)
    Sx, Sy, Sxx, Syy, Sxy = tot(X), tot(Y), tot(X * X), tot(Y * Y), tot(X * Y)
    rx, ry, rxx, ryy, rxy = rows(X), rows(Y), rows(X * X), rows(Y * Y), rows(X * Y)
    m = (n - 1) * (n - 2) / 2
    sx, sy, sxx, syy, sxy = Sx - rx, Sy - ry, Sxx - rxx, Syy - ryy, Sxy - rxy
    cov = sxy / m - sx * sy / m ** 2
    jk = cov / np.sqrt((sxx / m - (sx / m) ** 2) * (syy / m - (sy / m) ** 2))
    se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
    return se


def part_pearson(P, mt, gf, tok):
    g = pd.read_csv(HVG)
    ens = g["ensembl_id"].astype(str).tolist(); mti = g["maxtoki_token_id"].astype(int).to_numpy()
    gfi = np.array([tok[e] for e in ens])
    Sa = cosn(mt[mti]) @ cosn(mt[mti]).T; Sb = cosn(gf[gfi]) @ cosn(gf[gfi]).T
    n = Sa.shape[0]; iu = np.triu_indices(n, 1)
    obs = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    rng = np.random.default_rng(SEED + 4000)
    t = time.time()
    bd = boot_pearson_direct(Sa, Sb, rng, N_PB)
    se_jk = jackknife_pearson(Sa, Sb)
    res = {"hvg1500": {"n": n, "observed": obs, "boot_direct_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                       "boot_direct_sd": float(bd.std(ddof=1)), "boot_direct_mean": float(bd.mean()), "n_boot": N_PB,
                       "jackknife_se": se_jk, "jackknife_ci95": [obs - 1.96 * se_jk, obs + 1.96 * se_jk],
                       "seconds": time.time() - t}}
    print(json.dumps(res), flush=True)
    for di, dom in enumerate(DOMAINS):
        A = P[dom]["mt"]; Sa = cosn(A) @ cosn(A).T
        for m in ["gf", "scgpt_fixed"]:
            Sb = cosn(P[dom][m]) @ cosn(P[dom][m]).T
            n = Sa.shape[0]; iu = np.triu_indices(n, 1)
            obs = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1]); se = jackknife_pearson(Sa, Sb)
            d10 = json.loads((D10 / "jobs" / f"{m}__{dom}__boot.json").read_text())["pearson"]
            res[f"{m}/{dom}"] = {"observed": obs, "jackknife_se": se, "jackknife_ci95": [obs - 1.96 * se, obs + 1.96 * se],
                                 "D10_boot_ci95": d10["ci95"], "D10_boot_sd": d10["sd"]}
    return res


def part_bootcheck(P):
    """Exact in-sample CCA on gene resamples (with replacement): real pairing vs whole-gene
    permutation, and a pure-noise control with the same n and dims."""
    res = {}
    dom = "lung"; A_full = P[dom]["mt"]; B_full = P[dom]["gf"]; n = A_full.shape[0]
    rng = np.random.default_rng(SEED + 5000)
    real, null_gene, uniq = [], [], []
    for _ in range(N_BC):
        idx = rng.integers(0, n, n); pi = rng.permutation(n)
        A, _ = pca_exact(A_full[idx]); B, _ = pca_exact(B_full[idx]); Bg, _ = pca_exact(B_full[pi[idx]])
        real.append(cca_mean_exact(A, B)); null_gene.append(cca_mean_exact(A, Bg)); uniq.append(len(np.unique(idx)))
    res["gf/lung"] = {"observed_exact": cca_mean_exact(pca_exact(A_full)[0], pca_exact(B_full)[0]),
                      "boot_real_mean": float(np.mean(real)),
                      "boot_real_ci95": [float(np.percentile(real, 2.5)), float(np.percentile(real, 97.5))],
                      "boot_wholegene_null_mean": float(np.mean(null_gene)), "mean_unique": float(np.mean(uniq)),
                      "n_boot": N_BC}
    # pure noise: independent Gaussian 382 x 30 tables
    rngn = np.random.default_rng(SEED + 5001)
    full, boot, distinct = [], [], []
    for _ in range(N_BC):
        A = rngn.standard_normal((n, D)); B = rngn.standard_normal((n, D))
        full.append(cca_mean_exact(A, B))
        idx = rngn.integers(0, n, n)
        boot.append(cca_mean_exact(A[idx], B[idx]))
        u = np.unique(idx); distinct.append(cca_mean_exact(A[u], B[u]))
    res["noise_382x30"] = {"no_resampling_mean": float(np.mean(full)), "with_replacement_resample_mean": float(np.mean(boot)),
                           "distinct_genes_only_mean": float(np.mean(distinct)), "n": N_BC}
    print(json.dumps(res), flush=True)
    return res


def part_pcasolver(P):
    """How much do the deployed in-sample numbers depend on the PCA solver? sklearn PCA with
    random_state=42 picks the randomized solver for these shapes (as deployed); compare with the
    exact (full) solver. Exact CCA and in-sample top-1 are computed on each."""
    from sklearn.decomposition import PCA
    res = {}
    for dom in DOMAINS:
        A_full = P[dom]["mt"]; n = A_full.shape[0]
        for m in ["gf", "scgpt_fixed", "scgpt_deployed"]:
            B_full = P[dom][m]; row = {}
            for solver in ["auto", "full"]:
                pa = PCA(n_components=D, random_state=42, svd_solver=solver)
                pb = PCA(n_components=D, random_state=42, svd_solver=solver)
                A = pa.fit_transform(A_full - A_full.mean(0)); B = pb.fit_transform(B_full - B_full.mean(0))
                R, _ = orthogonal_procrustes(A, B)
                row[solver] = {"solver_used": getattr(pa, "_fit_svd_solver", solver),
                               "cca_exact_mean10": cca_mean_exact(A, B), "top1_insample": top1(A, B, R, np.arange(n))}
            res[f"{m}/{dom}"] = row
    print(json.dumps(res), flush=True)
    return res


def part_permorigin(P):
    """D10's 'permcheck' found that refitting PCA on permuted rows moves the CCA by up to 0.011 and
    called this noise of sklearn's iterative CCA. Test the other candidate, the PCA solver:
    for 10 permutations, compare CCA(A, PCA(B)[p]) with CCA(A, PCA(B[p])) for the randomized solver
    (deployed) and for the exact solver; also compare sklearn CCA with exact CCA on the same tables."""
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    import warnings
    warnings.filterwarnings("ignore")
    res = {}
    rng = np.random.default_rng(SEED + 6000)
    for m in ["gf", "scgpt_fixed"]:
        A_full = P["lung"]["mt"]; B_full = P["lung"][m]; n = A_full.shape[0]
        row = {}
        for solver in ["randomized", "full"]:
            pc = lambda X: PCA(n_components=D, random_state=42, svd_solver=solver).fit_transform(X - X.mean(0))
            A = pc(A_full); B = pc(B_full); diffs = []
            for _ in range(10):
                q = rng.permutation(n)
                diffs.append(abs(cca_mean_exact(A, B[q]) - cca_mean_exact(A, pc(B_full[q]))))
            row[solver] = {"max_absdiff_exactCCA_rowperm_vs_pca_refit": float(max(diffs))}
            sk = CCA(n_components=K, max_iter=200); a, b = sk.fit_transform(A, B)
            sk_r = float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(K)]))
            row[solver]["sklearn_cca"] = sk_r; row[solver]["exact_cca"] = cca_mean_exact(A, B)
        res[f"{m}/lung"] = row
    print(json.dumps(res), flush=True)
    return res


def part_rawscgpt(P, sc, vj):
    """Extra check D10 listed as not done: scGPT RAW table (before enc_norm LayerNorm), correct lookup.
    Gene-pair Pearson and exact in-sample CCA (exact PCA) next to the normed table."""
    Eraw = sc["embeddings"].float().numpy()
    res = {}
    for dom in DOMAINS:
        A_full = P[dom]["mt"]; Sa = cosn(A_full) @ cosn(A_full).T
        n = A_full.shape[0]; iu = np.triu_indices(n, 1); A, _ = pca_exact(A_full)
        row = {}
        for tag, B_full in [("normed", P[dom]["scgpt_fixed"]), ("raw", Eraw[P[dom]["sci"]].astype(np.float64))]:
            Sb = cosn(B_full) @ cosn(B_full).T; B, _ = pca_exact(B_full)
            row[tag] = {"pearson": float(np.corrcoef(Sa[iu], Sb[iu])[0, 1]), "cca_exact_mean10": cca_mean_exact(A, B)}
        res[dom] = row
    print(json.dumps(res), flush=True)
    return res


def part_hvgscgpt(mt, E, vj, sc):
    """scGPT on the paper's 1,500 HVGs, rows from the checkpoint's vocab.json (correct) and by dict
    position (deployed rule); gene-pair Pearson against MaxToki."""
    g = pd.read_csv(HVG)
    sy = g["symbol"].astype(str).tolist(); mti = g["maxtoki_token_id"].astype(int).to_numpy()
    dep_pos = {s.upper(): i for i, s in enumerate(sc["vocab"])}
    ok = np.array([s in vj for s in sy])
    sci = np.array([vj[s] for s, o in zip(sy, ok) if o]); dpi = np.array([dep_pos[s.upper()] for s, o in zip(sy, ok) if o])
    Sa = cosn(mt[mti[ok]]) @ cosn(mt[mti[ok]]).T; iu = np.triu_indices(Sa.shape[0], 1)
    out = {"n_genes_found": int(ok.sum())}
    for tag, ix in [("fixed_vocab_json", sci), ("deployed_dict_position", dpi)]:
        Sb = cosn(E[ix]) @ cosn(E[ix]).T
        out[tag] = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    print(json.dumps(out), flush=True)
    return out


def part_csvcheck():
    rows = list(csv.DictReader(open(D10 / "fig_crossmodel.csv")))
    J = D10 / "jobs"
    bad = []; checked = 0

    def cmp(r, field, val):
        nonlocal checked
        checked += 1
        s = r[field]
        if s == "" and val is None:
            return
        if s == "" or val is None or abs(float(s) - float(val)) > 5e-6 * max(1, abs(float(val))):
            bad.append({"metric": r["metric"], "model": r["model_key"], "panel": r["panel"], "field": field,
                        "csv": s, "json": val})
    for r in rows:
        m, p, met = r["model_key"], r["panel"], r["metric"]
        if p in DOMAINS and m in ("gf", "scgpt_fixed", "scgpt_deployed"):
            ld = lambda part: json.loads((J / f"{m}__{p}__{part}.json").read_text()) if (J / f"{m}__{p}__{part}.json").exists() else None
            obs, perm, boot, oob, cv, cvp = ld("obs"), ld("perm"), ld("boot"), ld("oob"), ld("cv"), ld("cvperm")
            if met == "cca_insample_mean10":
                cmp(r, "value", obs["cca_mean_r"]); cmp(r, "chance_mean", perm["cca_null"]["mean"]); cmp(r, "chance_p95", perm["cca_null"]["p95"])
            elif met == "pairwise_cosine_pearson":
                cmp(r, "value", obs["pairwise_pearson"]); cmp(r, "chance_mean", perm["pearson_null"]["mean"])
                if boot:
                    cmp(r, "ci_low", boot["pearson"]["ci95"][0]); cmp(r, "ci_high", boot["pearson"]["ci95"][1])
            elif met == "top1_insample":
                cmp(r, "value", obs["top1"]); cmp(r, "chance_mean", perm["top1_null_refit"]["mean"])
            elif met == "cca_heldout_mean10_5fold":
                cmp(r, "value", cv["cv_cca_mean_r"]); cmp(r, "chance_mean", cvp["cv_cca_mean_r_null"]["mean"])
            elif met == "top1_heldout_5fold":
                cmp(r, "value", cv["cv_top1"]); cmp(r, "chance_mean", cvp["cv_top1_null"]["mean"])
            elif met == "cca_heldout_mean10_oob":
                cmp(r, "value", oob["oob_cca_mean_r"]["mean"]); cmp(r, "ci_low", oob["oob_cca_mean_r"]["ci95"][0]); cmp(r, "ci_high", oob["oob_cca_mean_r"]["ci95"][1])
            elif met == "top1_heldout_oob":
                cmp(r, "value", oob["oob_top1"]["mean"]); cmp(r, "ci_low", oob["oob_top1"]["ci95"][0]); cmp(r, "ci_high", oob["oob_top1"]["ci95"][1])
    return {"n_rows": len(rows), "n_fields_checked": checked, "mismatches": bad,
            "metrics": sorted(set(r["metric"] for r in rows)), "panels": sorted(set(r["panel"] for r in rows))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lookup", "insample", "heldout", "pearson", "bootcheck",
                                                               "pcasolver", "permorigin", "rawscgpt", "hvgscgpt", "csvcheck"]
    need_tables = any(p != "csvcheck" for p in parts)
    if need_tables:
        P, sc, vj, mt, gf, E, tok = build_panels()
    for part in parts:
        f = OUT / f"{part}.json"
        if f.exists():
            print("skip", part); continue
        t = time.time()
        if part == "lookup":
            r = part_lookup(P, sc, vj)
        elif part == "insample":
            r = part_insample(P)
        elif part == "heldout":
            r = part_heldout(P, deadline=time.time() + float(os.environ.get("BUDGET_S", "420")))
            if r is None:
                print("heldout: budget reached; rerun to resume"); continue
        elif part == "pearson":
            r = part_pearson(P, mt, gf, tok)
        elif part == "bootcheck":
            r = part_bootcheck(P)
        elif part == "hvgscgpt":
            r = part_hvgscgpt(mt, E, vj, sc)
        elif part == "rawscgpt":
            r = part_rawscgpt(P, sc, vj)
        elif part == "permorigin":
            r = part_permorigin(P)
        elif part == "pcasolver":
            r = part_pcasolver(P)
        elif part == "csvcheck":
            r = part_csvcheck()
        r["_wall_seconds"] = time.time() - t
        f.write_text(json.dumps(r, indent=2))
        print("done", part, round(time.time() - t, 1), flush=True)
    cfg_f = OUT / "run_config.json"
    if not cfg_f.exists():
        inputs = {"maxtoki_model_safetensors": MT_ST, "geneformer_model_safetensors": GF_DIR / "model.safetensors",
                  "geneformer_token_dictionary": GF_TOK, "scgpt_embeddings_pt": SC_PT,
                  "scgpt_checkpoint_vocab_json": SC_VOCAB_JSON, "hvg_gene_features": HVG,
                  "D10_embeddings_subset_npz": D10 / "embeddings_subset.npz",
                  "D10_fig_crossmodel_csv": D10 / "fig_crossmodel.csv"}
        for dom in DOMAINS:
            inputs[f"gene_features_{dom}"] = RUN / "outputs/phase0" / dom / "gene_features.csv"
        cfg = {"what": "independent verification of D10 cross-model item", "script": str(Path(__file__)),
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seeds": {"base": SEED, "rule": "insample SEED+100*domain+10*model+1; heldout cv +1000, cv-perm +2000, "
                                                 "oob +3000; pearson +4000; bootcheck +5000/+5001"},
               "counts": {"N_PERM": N_PERM, "N_CV_REP": N_CV_REP, "N_CV_PERM": N_CV_PERM, "N_OOB": N_OOB, "N_PB": N_PB, "N_BC": N_BC},
               "inputs": {k: {"path": str(p), "resolved": os.path.realpath(p), "sha256": sha(os.path.realpath(p))}
                          for k, p in inputs.items()},
               "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                            "pandas": pd.__version__},
               "device": "CPU only; no model forward pass"}
        cfg_f.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
