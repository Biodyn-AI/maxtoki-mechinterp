"""Verifier's check that the key verdict does not depend on the choice of metric.
The two inline checks run during verification, merged into one file and re-run (output: robust.json).
Reads only the package data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = Path("<EVAL_ROOT>/studyA/tasks/T4-paper/data")
mt = np.load(D / "maxtoki_217m/embed_tokens.npy", mmap_mode="r")
gf = np.load(D / "geneformer_v2_316m/word_embeddings.npy", mmap_mode="r")
sc = np.load(D / "scgpt_whole_human/encoder_embedding_weight.npy", mmap_mode="r")
ln = np.load(D / "scgpt_whole_human/encoder_enc_norm.npz")
mtok = json.load(open(D / "maxtoki_217m/token_dictionary.json"))
voc = json.load(open(D / "scgpt_whole_human/vocab.json"))
es = pd.read_csv(D / "gene_ids/ensembl_symbol.csv")
sym = dict(zip(es.ensembl_id, es.symbol))


def lnorm(X):
    X = np.asarray(X, np.float64); m = X.mean(1, keepdims=True); v = X.var(1, keepdims=True)
    return (X - m) / np.sqrt(v + 1e-5) * ln["weight"] + ln["bias"]


def cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True); return Xn @ Xn.T


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def pcs(X, k):
    Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False); return U[:, :k] * s[:k]


def cca_mean(A, B, k=10):
    Qa, _ = np.linalg.qr(A - A.mean(0)); Qb, _ = np.linalg.qr(B - B.mean(0))
    return float(np.linalg.svd(Qa.T @ Qb, compute_uv=False)[:k].mean())


out = {}
rng = np.random.default_rng(1)
for p in ["lung", "immune", "external_lung"]:
    g = pd.read_csv(D / f"panels/{p}.csv").ensembl_id.tolist(); n = len(g)
    A = np.asarray(mt[[mtok[e] for e in g]], np.float64)
    G = np.asarray(gf[[mtok[e] for e in g]], np.float64)
    S = lnorm(sc[[voc[sym[e]] for e in g]])
    iu = np.triu_indices(n, 1); Ca, Cg, Cs = cos(A), cos(G), cos(S)
    r = {"spearman_gf": float(spearmanr(Ca[iu], Cg[iu])[0]), "spearman_sc": float(spearmanr(Ca[iu], Cs[iu])[0]),
         "cka_gf": cka(A, G), "cka_sc": cka(A, S),
         "cka_chance_mean": float(np.mean([cka(A, S[rng.permutation(n)]) for _ in range(50)]))}
    P20 = [pcs(X, 20) for X in (A, G, S)]
    r["cca_pca20_gf"] = cca_mean(P20[0], P20[1]); r["cca_pca20_sc"] = cca_mean(P20[0], P20[2])
    r["cca_pca20_chance"] = float(np.mean([cca_mean(P20[0], P20[1][rng.permutation(n)]) for _ in range(200)]))
    # neighbour overlap, with a paired per-gene bootstrap of the Geneformer - scGPT difference
    for M in (Ca, Cg, Cs):
        np.fill_diagonal(M, -9)
    for k in (10, 20):
        nA, nG, nS = (np.argsort(-M, 1)[:, :k] for M in (Ca, Cg, Cs))
        og = np.array([len(set(nA[i]) & set(nG[i])) / k for i in range(n)])
        os_ = np.array([len(set(nA[i]) & set(nS[i])) / k for i in range(n)])
        d = og - os_; b = np.random.default_rng(0)
        bs = [d[b.integers(0, n, n)].mean() for _ in range(2000)]
        r[f"knn{k}"] = {"gf": float(og.mean()), "sc": float(os_.mean()), "diff": float(d.mean()),
                        "diff_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "chance": k / (n - 1)}
    out[p] = r
print(json.dumps(out, indent=1))
