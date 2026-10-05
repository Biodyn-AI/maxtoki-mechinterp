"""Verifier's own re-derivation of the T4 key numbers from the package data only.

Usage: python verify_metrics.py <panel> <stage>
  stage = insample | heldout | oob | union
Writes JSON to stdout.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

D = Path("<EVAL_ROOT>/studyA/tasks/T4-paper/data")
PANEL, STAGE = sys.argv[1], sys.argv[2]
T0 = time.time()


def load_all():
    mt = np.load(D / "maxtoki_217m/embed_tokens.npy", mmap_mode="r")
    gf = np.load(D / "geneformer_v2_316m/word_embeddings.npy", mmap_mode="r")
    sc = np.load(D / "scgpt_whole_human/encoder_embedding_weight.npy", mmap_mode="r")
    ln = np.load(D / "scgpt_whole_human/encoder_enc_norm.npz")
    mtok = json.loads((D / "maxtoki_217m/token_dictionary.json").read_text())
    gtok = json.loads((D / "geneformer_v2_316m/token_dictionary.json").read_text())
    voc = json.loads((D / "scgpt_whole_human/vocab.json").read_text())
    es = pd.read_csv(D / "gene_ids/ensembl_symbol.csv")
    return mt, gf, sc, ln, mtok, gtok, voc, dict(zip(es.ensembl_id, es.symbol))


mt, gf, sc, ln, mtok, gtok, voc, sym = load_all()


def lnorm(X):
    X = np.asarray(X, dtype=np.float64)
    m = X.mean(1, keepdims=True); v = X.var(1, keepdims=True)
    return (X - m) / np.sqrt(v + float(ln["eps"])) * ln["weight"].astype(np.float64) + ln["bias"].astype(np.float64)


def genes_for(panel):
    if panel == "union":
        g = []
        for p in ["lung", "immune", "external_lung"]:
            g += pd.read_csv(D / f"panels/{p}.csv")["ensembl_id"].tolist()
        return sorted(set(g))
    return pd.read_csv(D / f"panels/{panel}.csv")["ensembl_id"].tolist()


genes = genes_for(PANEL)
n = len(genes)
pos_rule = {s.upper(): i for i, s in enumerate(voc)}  # deployed rule: key position
T = {
    "mt": np.asarray(mt[[mtok[e] for e in genes]], dtype=np.float64),
    "gf": np.asarray(gf[[gtok[e] for e in genes]], dtype=np.float64),
    "sc": lnorm(sc[[voc[sym[e]] for e in genes]]),
    "sc_raw": np.asarray(sc[[voc[sym[e]] for e in genes]], dtype=np.float64),
    "sc_pos": lnorm(sc[[pos_rule[sym[e].upper()] for e in genes]]),
}


def cosmat(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 0.0)
    return S


def pearson_ut(Sa, Sb):
    iu = np.triu_indices(Sa.shape[0], 1)
    return float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])


def coords(X):
    """n x n coordinates that keep every inner product among rows (exact)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt.T


def pca_fit(Z, k=30):
    from scipy.linalg import eigh
    mu = Z.mean(0)
    Xc = Z - mu
    d = Xc.shape[1]
    w, V = eigh(Xc.T @ Xc, subset_by_index=[d - k, d - 1])
    return mu, V[:, ::-1]


def cca_exact_corrs(A, B, k=10):
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)[:k]


def cca_fit(A, B, k=10):
    ma, mb = A.mean(0), B.mean(0)
    Qa, Ra = np.linalg.qr(A - ma); Qb, Rb = np.linalg.qr(B - mb)
    U, s, Vt = np.linalg.svd(Qa.T @ Qb)
    Wa = np.linalg.solve(Ra, U[:, :k]); Wb = np.linalg.solve(Rb, Vt.T[:, :k])
    return ma, mb, Wa, Wb


def colcorr(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (X * Y).sum(0) / np.sqrt((X * X).sum(0) * (Y * Y).sum(0))


def rot(A, B):
    U, s, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def top1_insample(A, B):
    R = rot(A, B)
    sim = unit(A @ R) @ unit(B).T
    return float((sim.argmax(1) == np.arange(A.shape[0])).mean())


res = {"panel": PANEL, "n": n, "stage": STAGE}
Z = {k: coords(v) for k, v in T.items()}

if STAGE == "insample":
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    S = {k: cosmat(v) for k, v in T.items()}
    for k in ["gf", "sc", "sc_raw", "sc_pos"]:
        res[f"pearson_{k}"] = pearson_ut(S["mt"], S[k])
    # chance for Pearson
    rng = np.random.default_rng(101)
    iu = np.triu_indices(n, 1)
    a = S["mt"][iu]
    null = []
    for _ in range(1000):
        p = rng.permutation(n)
        null.append(np.corrcoef(a, S["gf"][p][:, p][iu])[0, 1])
    res["pearson_chance_gf_mean_sd"] = [float(np.mean(null)), float(np.std(null))]
    # gene bootstrap, copy pairs dropped, paired across gf and sc; weights c_g c_h
    def wsums(C, M):
        return 0.5 * ((C @ M) * C).sum(1)
    X = S["mt"]
    stats = {}
    rng = np.random.default_rng(202)
    B = 2000
    C = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    W = 0.5 * (C.sum(1) ** 2 - (C ** 2).sum(1))
    Sx = wsums(C, X); Sxx = wsums(C, X * X)
    for k in ["gf", "sc"]:
        Y = S[k]
        Sy = wsums(C, Y); Syy = wsums(C, Y * Y); Sxy = wsums(C, X * Y)
        cov = Sxy / W - (Sx / W) * (Sy / W)
        vx = Sxx / W - (Sx / W) ** 2; vy = Syy / W - (Sy / W) ** 2
        stats[k] = cov / np.sqrt(vx * vy)
    # check weighted formula against direct indexing on 3 draws
    chk = []
    for b in range(3):
        idx = np.repeat(np.arange(n), C[b].astype(int))
        ii, jj = np.triu_indices(len(idx), 1)
        keep = idx[ii] != idx[jj]
        r = np.corrcoef(X[idx[ii[keep]], idx[jj[keep]]], S["gf"][idx[ii[keep]], idx[jj[keep]]])[0, 1]
        chk.append(abs(r - stats["gf"][b]))
    res["boot_formula_check_maxdiff"] = float(max(chk))
    for k in ["gf", "sc"]:
        res[f"pearson_{k}_ci"] = [float(np.percentile(stats[k], 2.5)), float(np.percentile(stats[k], 97.5)), float(stats[k].mean())]
    d = stats["gf"] - stats["sc"]
    res["pearson_diff"] = res["pearson_gf"] - res["pearson_sc"]
    res["pearson_diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    # in-sample CCA and top-1, deployed sklearn setting (on full-dim centred tables)
    P = {}
    for k in ["mt", "gf", "sc", "sc_pos", "sc_raw"]:
        X32 = T[k].astype(np.float32)
        P[k] = PCA(n_components=30, random_state=42).fit_transform(X32 - X32.mean(0))
    for k in ["gf", "sc", "sc_pos"]:
        Ac, Bc = CCA(n_components=10, max_iter=200).fit_transform(P["mt"], P[k])
        res[f"cca_sklearn_{k}"] = float(np.mean([np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(10)]))
        res[f"top1_sklearnpca_{k}"] = top1_insample(P["mt"], P[k])
    # exact PCA versions
    E = {}
    for k in ["mt", "gf", "sc", "sc_pos", "sc_raw"]:
        mu, V = pca_fit(Z[k]); E[k] = (Z[k] - mu) @ V
    for k in ["gf", "sc", "sc_pos", "sc_raw"]:
        res[f"cca_exact_{k}"] = float(cca_exact_corrs(E["mt"], E[k]).mean())
        res[f"top1_exactpca_{k}"] = top1_insample(E["mt"], E[k])
    # chance: permute correspondence, refit CCA / rotation
    for k in ["gf", "sc"]:
        rng = np.random.default_rng(303)
        Qa, _ = np.linalg.qr(E["mt"] - E["mt"].mean(0))
        Qb, _ = np.linalg.qr(E[k] - E[k].mean(0))
        cc, tt = [], []
        for i in range(1000):
            p = rng.permutation(n)
            cc.append(np.linalg.svd(Qa.T @ Qb[p], compute_uv=False)[:10].mean())
            if i < 1000:
                tt.append(top1_insample(E["mt"], E[k][p]))
        res[f"cca_chance_{k}"] = [float(np.mean(cc)), float(np.std(cc)), float(np.percentile(cc, 95))]
        res[f"top1_chance_refit_{k}"] = [float(np.mean(tt)), float(np.percentile(tt, 95))]
    # non-refitted null (deployed style) for top-1, gf
    R = rot(E["mt"], E["gf"]); sim = unit(E["mt"] @ R) @ unit(E["gf"]).T
    rng = np.random.default_rng(404)
    res["top1_null_norefit_gf_mean"] = float(np.mean([(sim[:, rng.permutation(n)].argmax(1) == np.arange(n)).mean() for _ in range(200)]))
    # wrong-bootstrap demo for in-sample CCA (with replacement, Geneformer)
    rng = np.random.default_rng(505)
    bb = []
    for _ in range(100):
        idx = rng.integers(0, n, n)
        a = Z["mt"][idx]; b = Z["gf"][idx]
        mua, Va = pca_fit(a); mub, Vb = pca_fit(b)
        bb.append(cca_exact_corrs((a - mua) @ Va, (b - mub) @ Vb).mean())
    res["insample_cca_withreplacement_boot_gf"] = [float(np.mean(bb)), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]

elif STAGE == "heldout":
    NPERM_HO = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    def heldout(ZA, ZB, rng, nsplit):
        ccs, hits, tot = [], 0, 0
        for s in range(nsplit):
            perm = rng.permutation(n)
            folds = np.array_split(perm, 5)
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                mua, Va = pca_fit(ZA[tr]); mub, Vb = pca_fit(ZB[tr])
                A = (ZA - mua) @ Va; B = (ZB - mub) @ Vb
                ma, mb, Wa, Wb = cca_fit(A[tr], B[tr])
                ccs.append(colcorr((A[f] - ma) @ Wa, (B[f] - mb) @ Wb).mean())
                R = rot(A[tr], B[tr])
                sim = unit(A[f] @ R) @ unit(B).T
                hits += int((sim.argmax(1) == f).sum()); tot += len(f)
        return float(np.mean(ccs)), hits / tot
    for k in ["gf", "sc"]:
        rng = np.random.default_rng(606)
        res[f"heldout_{k}"] = heldout(Z["mt"], Z[k], rng, 10)
    for k in ["gf", "sc"]:
        rng = np.random.default_rng(707)
        cc, tt = [], []
        for _ in range(NPERM_HO):
            p = rng.permutation(n)
            c, t = heldout(Z["mt"], Z[k][p], rng, 1)
            cc.append(c); tt.append(t)
        res[f"heldout_chance_{k}"] = {"cca_mean": float(np.mean(cc)), "cca_p95": float(np.percentile(cc, 95)),
                                     "top1_mean": float(np.mean(tt)), "top1_p95": float(np.percentile(tt, 95))}

elif STAGE == "oob":
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    rng = np.random.default_rng(808)
    vals = {"gf": [], "sc": []}
    for b in range(NB):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        for k in ["gf", "sc"]:
            ZA, ZB = Z["mt"], Z[k]
            mua, Va = pca_fit(ZA[idx]); mub, Vb = pca_fit(ZB[idx])
            A = (ZA - mua) @ Va; B = (ZB - mub) @ Vb
            ma, mb, Wa, Wb = cca_fit(A[idx], B[idx])
            c = colcorr((A[oob] - ma) @ Wa, (B[oob] - mb) @ Wb).mean()
            R = rot(A[idx], B[idx])
            sim = unit(A[oob] @ R) @ unit(B).T
            t = float((sim.argmax(1) == oob).mean())
            vals[k].append((c, t))
    g = np.array(vals["gf"]); s = np.array(vals["sc"]); d = g - s
    res["oob_draws"] = NB
    for name, arr in [("gf", g), ("sc", s), ("diff", d)]:
        res[f"oob_cca_{name}"] = [float(arr[:, 0].mean()), float(np.percentile(arr[:, 0], 2.5)), float(np.percentile(arr[:, 0], 97.5))]
        res[f"oob_top1_{name}"] = [float(arr[:, 1].mean()), float(np.percentile(arr[:, 1], 2.5)), float(np.percentile(arr[:, 1], 97.5))]

elif STAGE == "union":
    S = {k: cosmat(v) for k, v in T.items() if k in ("mt", "gf", "sc")}
    res["pearson_gf"] = pearson_ut(S["mt"], S["gf"]); res["pearson_sc"] = pearson_ut(S["mt"], S["sc"])
    # sanity pairs
    def cos_pair(tab_rows, a, b):
        x, y = tab_rows(a), tab_rows(b)
        return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    good = lambda s: lnorm(sc[[voc[s]]])[0]
    bad = lambda s: lnorm(sc[[pos_rule[s.upper()]]])[0]
    res["pairs"] = {f"{a}-{b}": [cos_pair(good, a, b), cos_pair(bad, a, b)] for a, b in [("CD3D", "CD3E"), ("HBA1", "HBB"), ("RPL3", "RPL5")]}

res["seconds"] = round(time.time() - T0, 1)
print(json.dumps(res, indent=1))
