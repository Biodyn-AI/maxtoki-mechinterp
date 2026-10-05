"""Reference analysis for Study A task T4 (answer key; NOT part of any package).

Question: is MaxToki-217M's gene-embedding table aligned with scGPT's, and how does that
compare with its alignment to Geneformer V2-316M?

Reads ONLY the package data folder (default: studyA/tasks/T4-paper/data) and prints one JSON
object with every key number. Run:
    maxtoki-framework-eval/bin/python reference.py [path/to/package/root] > reference_output.json

Gene mapping
  MaxToki row     = maxtoki_217m/token_dictionary.json[ensembl_id]
  Geneformer row  = geneformer_v2_316m/token_dictionary.json[ensembl_id]
  scGPT row       = scgpt_whole_human/vocab.json[symbol]   (the dict VALUE), symbol from gene_ids/ensembl_symbol.csv
  scGPT vector    = LayerNorm(encoder_embedding_weight[row]) with enc_norm weight/bias, eps 1e-5
  "positional"    = the row taken as the POSITION of the symbol among vocab.json keys (the deployed rule);
                    computed only to give the numbers a wrong lookup produces.

Metrics (settings copy the deployed run: PCA to 30 dims per model, 10 CCA components)
  pearson        Pearson r between the upper triangles of the two gene x gene cosine matrices (full rows)
  cca_insample   mean of the 10 in-sample canonical correlations on PCA-30 of each table
                 (a) deployed: sklearn PCA(random_state=42) + sklearn CCA(10, max_iter=200)
                 (b) exact: SVD PCA + closed-form CCA (QR + SVD)
  top1_insample  orthogonal Procrustes MaxToki-PCA30 -> other-PCA30 fitted and scored on the same genes
  heldout_cca    5-fold split of genes, PCA and CCA fitted on 80%, canonical r on the held-out 20%, mean of 10
  heldout_top1   same split; Procrustes fitted on training genes; held-out gene's nearest neighbour among ALL panel genes
  chance         permutations of the gene correspondence; CCA / rotation refitted on every permutation
  intervals      unit = gene. Pearson: gene bootstrap with replacement, pairs of two copies dropped.
                 Held-out metrics: out-of-bag (OOB) bootstrap: fit on the resample, score on undrawn genes.
                 Paired differences (Geneformer minus scGPT) use the same resamples for both.
CPU only. Runs in parts (see main()); each part takes a few minutes on one core.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]
ROOT = Path(ARGS[0]) if ARGS else Path(
    "<EVAL_ROOT>/studyA/tasks/T4-paper")
PARTS_DIR = Path(__file__).resolve().parent / "parts"
DATA = ROOT / "data"
PANELS = ["lung", "immune", "external_lung"]
D = 30
K = 10
N_PERM = 1000
N_BOOT = 1000
N_PAIR_BOOT = 2000
N_OOB = 500
N_CV_REP = 10
N_CV_PERM = 200
SEED = 4242


# ----------------------------------------------------------------------------- loading
def load():
    mt = np.load(DATA / "maxtoki_217m/embed_tokens.npy", mmap_mode="r")
    gf = np.load(DATA / "geneformer_v2_316m/word_embeddings.npy", mmap_mode="r")
    sc = np.load(DATA / "scgpt_whole_human/encoder_embedding_weight.npy", mmap_mode="r")
    ln = np.load(DATA / "scgpt_whole_human/encoder_enc_norm.npz")
    mt_tok = json.loads((DATA / "maxtoki_217m/token_dictionary.json").read_text())
    gf_tok = json.loads((DATA / "geneformer_v2_316m/token_dictionary.json").read_text())
    vocab = json.loads((DATA / "scgpt_whole_human/vocab.json").read_text())
    es = pd.read_csv(DATA / "gene_ids/ensembl_symbol.csv")
    sym = dict(zip(es.ensembl_id, es.symbol))
    panels = {p: pd.read_csv(DATA / "panels" / f"{p}.csv")["ensembl_id"].astype(str).tolist() for p in PANELS}
    return mt, gf, sc, ln, mt_tok, gf_tok, vocab, sym, panels


def layernorm(X, w, b, eps):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(1, keepdims=True); var = X.var(1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps) * w + b


# ----------------------------------------------------------------------------- metrics
def normalise(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n


def cosm(X):
    Xn = normalise(np.asarray(X, dtype=np.float64))
    return Xn @ Xn.T


def pearson_tri(Sa, Sb, idx=None):
    if idx is None:
        iu = np.triu_indices(Sa.shape[0], 1)
        return float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    iu = np.triu_indices(len(idx), 1)
    gi, gj = idx[iu[0]], idx[iu[1]]
    keep = gi != gj
    return float(np.corrcoef(Sa[gi[keep], gj[keep]], Sb[gi[keep], gj[keep]])[0, 1])


def thin(X):
    """n x n coordinates with the same inner products as X (X = U S Vt -> U S). Every metric
    here (cosines, centred PCA of any row subset, CCA, Procrustes) is unchanged."""
    U, s, _ = np.linalg.svd(np.asarray(X, dtype=np.float64), full_matrices=False)
    return U * s


def pca_fit(X, d=D):
    """Exact PCA (top-d eigenvectors of the scatter matrix; same subspace as the SVD)."""
    mu = X.mean(0)
    Xc = X - mu
    p = Xc.shape[1]
    _, V = sla.eigh(Xc.T @ Xc, subset_by_index=[p - d, p - 1])
    return mu, V[:, ::-1]


def pca_scores(X, d=D):
    mu, P = pca_fit(X, d)
    return (X - mu) @ P


def cca_fit(A, B, k=K):
    ma, mb = A.mean(0), B.mean(0)
    Qa, Ra = np.linalg.qr(A - ma); Qb, Rb = np.linalg.qr(B - mb)
    U, s, Vt = np.linalg.svd(Qa.T @ Qb)
    Wa = np.linalg.solve(Ra, U[:, :k]); Wb = np.linalg.solve(Rb, Vt.T[:, :k])
    return ma, mb, Wa, Wb, s[:k]


def cca_insample_exact(A, B, k=K):
    Qa, _ = np.linalg.qr(A - A.mean(0)); Qb, _ = np.linalg.qr(B - B.mean(0))
    return float(np.linalg.svd(Qa.T @ Qb, compute_uv=False)[:k].mean())


def cca_insample_sklearn(A, B, k=K):
    a, b = CCA(n_components=k, max_iter=200).fit_transform(A, B)
    return float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(k)]))


def procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def top1(Aq, Bc, R, q_ids, c_ids):
    sim = normalise(Aq @ R) @ normalise(Bc).T
    return float((c_ids[np.argmax(sim, 1)] == q_ids).mean())


def deployed_insample(A_full, B_full):
    """Deployed settings: sklearn PCA(30, random_state=42) on centred rows, sklearn CCA(10)."""
    A = PCA(n_components=D, random_state=42).fit_transform(A_full - A_full.mean(0))
    B = PCA(n_components=D, random_state=42).fit_transform(B_full - B_full.mean(0))
    ids = np.arange(len(A))
    return {"cca": cca_insample_sklearn(A, B),
            "top1": top1(A, B, procrustes(A, B), ids, ids)}


def heldout_pass(Az, Bz, tr, te, Apca=None):
    """Fit PCA (each model), CCA and rotation on training genes; score held-out genes."""
    muA, PA = Apca if Apca is not None else pca_fit(Az[tr])
    muB, PB = pca_fit(Bz[tr])
    A_tr, A_te = (Az[tr] - muA) @ PA, (Az[te] - muA) @ PA
    B_all = (Bz - muB) @ PB
    B_tr, B_te = (Bz[tr] - muB) @ PB, B_all[te]
    ma, mb, Wa, Wb, _ = cca_fit(A_tr, B_tr)
    a, b = (A_te - ma) @ Wa, (B_te - mb) @ Wb
    r = float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(K)]))
    R = procrustes(A_tr, B_tr)
    n = Bz.shape[0]
    t = top1(A_te, B_all, R, te, np.arange(n))
    return r, t


def kfold_splits(n, rng, k=5):
    p = rng.permutation(n)
    folds = np.array_split(p, k)
    out = []
    for i in range(k):
        te = np.sort(folds[i]); tr = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((tr, te))
    return out


def cv_once(Az, Bz, splits, Apcas=None):
    rs, hits = [], 0
    for f, (tr, te) in enumerate(splits):
        r, t = heldout_pass(Az, Bz, tr, te, None if Apcas is None else Apcas[f])
        rs.append(r); hits += t * len(te)
    return float(np.mean(rs)), hits / Bz.shape[0]


def ci(v):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def null_summary(obs, null):
    null = np.asarray(null, dtype=float)
    return {"mean": float(null.mean()), "sd": float(null.std(ddof=1)), "p95": float(np.percentile(null, 95)),
            "p_one_sided": float((1 + (null >= obs).sum()) / (1 + len(null))),
            "excess_over_mean": float(obs - null.mean())}


# ----------------------------------------------------------------------------- per panel
def analyse_panel(name, A_full, tabs_full, rng):
    """A_full: MaxToki rows; tabs_full: dict model -> other-model rows (same genes, same order)."""
    n = A_full.shape[0]
    out = {"n_genes": n}
    Az = thin(A_full)
    Z = {m: thin(X) for m, X in tabs_full.items()}
    Sa = cosm(Az)
    S = {m: cosm(Z[m]) for m in Z}
    Apc = pca_scores(Az)
    # observed values
    for m in tabs_full:
        o = {"pearson": pearson_tri(Sa, S[m])}
        dep = deployed_insample(np.asarray(A_full, dtype=np.float64), np.asarray(tabs_full[m], dtype=np.float64))
        o["cca_insample_deployed_settings"] = dep["cca"]
        o["top1_insample_deployed_settings"] = dep["top1"]
        Bpc = pca_scores(Z[m])
        o["cca_insample_exact"] = cca_insample_exact(Apc, Bpc)
        ids = np.arange(n)
        o["top1_insample_exact"] = top1(Apc, Bpc, procrustes(Apc, Bpc), ids, ids)
        out[m] = o
    # chance levels, intervals, held-out values for the two main comparisons
    for m in ["geneformer", "scgpt"]:
        o = out[m]
        Bpc = pca_scores(Z[m])
        iu = np.triu_indices(n, 1); sa = Sa[iu]
        R0 = procrustes(Apc, Bpc)
        sim0 = normalise(Apc @ R0) @ normalise(Bpc).T
        c_n, t_n, t_dep, p_n = [], [], [], []
        ids = np.arange(n)
        for _ in range(N_PERM):
            p = rng.permutation(n)
            Bp = Bpc[p]
            c_n.append(cca_insample_exact(Apc, Bp))
            t_n.append(top1(Apc, Bp, procrustes(Apc, Bp), ids, ids))
            t_dep.append(float((np.argmax(sim0[:, p], 1) == ids).mean()))
            p_n.append(float(np.corrcoef(sa, S[m][np.ix_(p, p)][iu])[0, 1]))
        o["chance_cca_insample"] = null_summary(o["cca_insample_exact"], c_n)
        o["chance_top1_insample_rotation_refit"] = null_summary(o["top1_insample_exact"], t_n)
        o["chance_top1_insample_rotation_not_refit"] = null_summary(o["top1_insample_exact"], t_dep)
        o["chance_pearson"] = null_summary(o["pearson"], p_n)
        # gene bootstrap for Pearson
        pb = [pearson_tri(Sa, S[m], rng.integers(0, n, n)) for _ in range(N_BOOT)]
        o["pearson_gene_bootstrap_ci95"] = ci(pb)
        # with-replacement bootstrap of the in-sample CCA (to show it is biased upward)
        cb = []
        for _ in range(200):
            ix = rng.integers(0, n, n)
            cb.append(cca_insample_exact(pca_scores(Az[ix]), pca_scores(Z[m][ix])))
        o["cca_insample_with_replacement_bootstrap_mean_and_ci95"] = [float(np.mean(cb))] + ci(cb)
        # held-out 5-fold, 10 repeats
        reps = []
        for r in range(N_CV_REP):
            reps.append(cv_once(Az, Z[m], kfold_splits(n, np.random.default_rng(SEED + 100 + r))))
        reps = np.array(reps)
        o["heldout_cca_5fold_mean"] = float(reps[:, 0].mean())
        o["heldout_cca_5fold_range"] = [float(reps[:, 0].min()), float(reps[:, 0].max())]
        o["heldout_top1_5fold_mean"] = float(reps[:, 1].mean())
        o["heldout_top1_5fold_range"] = [float(reps[:, 1].min()), float(reps[:, 1].max())]
        splits0 = kfold_splits(n, np.random.default_rng(SEED + 100))
        Apcas = [pca_fit(Az[tr]) for tr, _ in splits0]
        cvn = np.array([cv_once(Az, Z[m][rng.permutation(n)], splits0, Apcas) for _ in range(N_CV_PERM)])
        o["chance_heldout_cca"] = null_summary(o["heldout_cca_5fold_mean"], cvn[:, 0])
        o["chance_heldout_top1"] = null_summary(o["heldout_top1_5fold_mean"], cvn[:, 1])
    # paired differences on the same resamples
    idxs = rng.integers(0, n, size=(N_PAIR_BOOT, n))
    pdiff = [pearson_tri(Sa, S["geneformer"], ix) - pearson_tri(Sa, S["scgpt"], ix) for ix in idxs]
    oob = []
    for b in range(N_OOB):
        ix = idxs[b]; te = np.setdiff1d(np.arange(n), ix)
        Apca = pca_fit(Az[ix])
        rg, tg = heldout_pass(Az, Z["geneformer"], ix, te, Apca)
        rs, ts = heldout_pass(Az, Z["scgpt"], ix, te, Apca)
        oob.append([rg, rs, tg, ts])
    oob = np.array(oob)
    out["paired_geneformer_minus_scgpt"] = {
        "pearson_diff": out["geneformer"]["pearson"] - out["scgpt"]["pearson"],
        "pearson_diff_ci95": ci(pdiff),
        "heldout_cca_oob_diff_mean": float((oob[:, 0] - oob[:, 1]).mean()),
        "heldout_cca_oob_diff_ci95": ci(oob[:, 0] - oob[:, 1]),
        "heldout_top1_oob_diff_mean": float((oob[:, 2] - oob[:, 3]).mean()),
        "heldout_top1_oob_diff_ci95": ci(oob[:, 2] - oob[:, 3]),
        "frac_draws_le_0": {"pearson": float((np.array(pdiff) <= 0).mean()),
                            "heldout_cca": float(((oob[:, 0] - oob[:, 1]) <= 0).mean()),
                            "heldout_top1": float(((oob[:, 2] - oob[:, 3]) <= 0).mean())},
    }
    for j, m in enumerate(["geneformer", "scgpt"]):
        out[m]["heldout_cca_oob_mean_ci95"] = [float(oob[:, j].mean())] + ci(oob[:, j])
        out[m]["heldout_top1_oob_mean_ci95"] = [float(oob[:, 2 + j].mean())] + ci(oob[:, 2 + j])
    return out


def main():
    """Parts: --part=lung | --part=immune | --part=external_lung | --part=extras  (each writes parts/<part>.json);
    --combine prints the merged JSON. With no flag, every part is run in turn and the merged JSON is printed."""
    t0 = time.time()
    part = next((f.split("=", 1)[1] for f in FLAGS if f.startswith("--part=")), None)
    combine = "--combine" in FLAGS
    PARTS_DIR.mkdir(exist_ok=True)
    todo = [part] if part else ([] if combine else PANELS + ["extras"])
    mt, gf, sc, ln, mt_tok, gf_tok, vocab, sym, panels = load()
    keys = list(vocab.keys())
    pos = {k: i for i, k in enumerate(keys)}

    def sc_rows(rows):
        return layernorm(np.asarray(sc[rows]), ln["weight"], ln["bias"], float(ln["eps"]))

    for name in todo:
        rng = np.random.default_rng(SEED + 1 + (PANELS + ["extras"]).index(name))
        res = {}
        if name in PANELS:
            ens = panels[name]
            mt_rows = np.array([mt_tok[e] for e in ens]); gf_rows = np.array([gf_tok[e] for e in ens])
            syms = [sym[e] for e in ens]
            sc_val = np.array([vocab[s] for s in syms]); sc_pos = np.array([pos[s] for s in syms])
            A = np.asarray(mt[mt_rows], dtype=np.float64)
            tabs = {"geneformer": np.asarray(gf[gf_rows], dtype=np.float64),
                    "scgpt": sc_rows(sc_val),
                    "scgpt_raw_no_layernorm": np.asarray(sc[sc_val], dtype=np.float64),
                    "scgpt_positional_lookup": sc_rows(sc_pos)}
            res = analyse_panel(name, A, tabs, rng)
            res["n_scgpt_rows_same_under_positional_lookup"] = int((sc_val == sc_pos).sum())
        else:
            res["lookup_checks"] = {
                "scgpt_vocab_size": len(vocab),
                "scgpt_vocab_key_position_equals_value": int(sum(1 for i, k in enumerate(keys) if vocab[k] == i)),
                "maxtoki_vs_geneformer_gene_ids_identical": bool(all(gf_tok.get(e) == v for e, v in mt_tok.items()
                                                                     if e.startswith("ENSG"))),
            }
            pairs = [("RPL3", "RPL5"), ("CD3D", "CD3E"), ("CD79A", "CD79B"), ("HBA1", "HBB"), ("RPS3", "RPS6")]
            sp = {}
            for a, b in pairs:
                va, vb = sc_rows([vocab[a], vocab[b]])
                pa, pb = sc_rows([pos[a], pos[b]])
                sp[f"{a}-{b}"] = {"value_lookup": float(normalise(va[None])[0] @ normalise(vb[None])[0]),
                                  "positional_lookup": float(normalise(pa[None])[0] @ normalise(pb[None])[0])}
            rr = np.random.default_rng(0).integers(0, len(vocab) - 3, size=(2000, 2))
            allc = [float(normalise(x[None])[0] @ normalise(y[None])[0]) for x, y in
                    zip(sc_rows(rr[:, 0]), sc_rows(rr[:, 1]))]
            sp["random_pairs_mean_sd"] = [float(np.mean(allc)), float(np.std(allc))]
            res["scgpt_sanity_pairs_cosine"] = sp
            union = sorted(set().union(*[set(v) for v in panels.values()]))
            L, I, E = (set(panels[x]) for x in PANELS)
            res["panel_overlap"] = {"lung&immune": len(L & I), "lung&external_lung": len(L & E),
                                    "immune&external_lung": len(I & E), "all_three": len(L & I & E),
                                    "union": len(union)}

            def pearson_set(ens, nb=500):
                mt_rows = np.array([mt_tok[e] for e in ens]); gf_rows = np.array([gf_tok[e] for e in ens])
                sc_val = np.array([vocab[sym[e]] for e in ens])
                Sa = cosm(thin(np.asarray(mt[mt_rows], dtype=np.float64)))
                Sg = cosm(thin(np.asarray(gf[gf_rows], dtype=np.float64)))
                Ss = cosm(thin(sc_rows(sc_val)))
                n = len(ens)
                o = {"n_genes": n, "pearson_geneformer": pearson_tri(Sa, Sg), "pearson_scgpt": pearson_tri(Sa, Ss)}
                if n <= 900:
                    d = []
                    for _ in range(nb):
                        ix = rng.integers(0, n, n)
                        d.append(pearson_tri(Sa, Sg, ix) - pearson_tri(Sa, Ss, ix))
                    o["pearson_diff_ci95_gene_bootstrap"] = ci(d)
                return o
            res["union_of_panels"] = pearson_set(union)
            shared = [e for e in mt_tok if e.startswith("ENSG") and e in gf_tok and e in sym and sym[e] in vocab]
            bg = sorted(np.random.default_rng(SEED + 7).choice(shared, size=2000, replace=False).tolist())
            res["random_2000_shared_genes"] = pearson_set(bg)
            res["random_2000_shared_genes"]["n_shared_genes_all_three_vocabularies"] = len(shared)
        res["wall_seconds"] = time.time() - t0
        (PARTS_DIR / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[{name}] done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    if combine or not part:
        out = {"package_root": ROOT.name,
               "settings": {"pca_dim": D, "cca_components": K, "n_perm": N_PERM, "n_boot": N_BOOT,
                            "n_pair_boot": N_PAIR_BOOT, "n_oob": N_OOB, "cv_repeats": N_CV_REP,
                            "cv_perm": N_CV_PERM, "seed": SEED}}
        ex = json.loads((PARTS_DIR / "extras.json").read_text())
        out.update({k: v for k, v in ex.items() if k != "wall_seconds"})
        for name in PANELS:
            out[name] = json.loads((PARTS_DIR / f"{name}.json").read_text())
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
