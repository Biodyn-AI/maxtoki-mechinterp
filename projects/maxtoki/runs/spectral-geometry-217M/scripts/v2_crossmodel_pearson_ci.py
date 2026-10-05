"""v2 cross-model alignment (revision item D10): proper interval for the published
MaxToki-vs-Geneformer gene-pair Pearson 0.382 (spectral-geometry Phase 9b).

Representation: static input tables only -- MaxToki `model.embed_tokens.weight` and
Geneformer V2-316M `bert.embeddings.word_embeddings.weight`, rows for the 1,500 HVGs in
outputs/phase0/gene_features.csv (as in scripts/phase9b_cross_model.py:72-136).
Statistic: Pearson over the 1,124,250 gene pairs (upper triangle) of the two cosine matrices.

Intervals (unit = gene in all cases):
  A. published scheme reproduced: 80% of genes WITHOUT replacement, 1,000 draws, seed 42
     (scripts/audit_a3_bootstrap_cross_model_pearson.py:82-92);
  B. gene bootstrap WITH replacement, N_BOOT draws; pairs made of two copies of one gene are
     dropped. Computed with a weighted-pair formula (pair (g,h) gets weight c_g*c_h, c = copy
     counts), checked against direct indexing on the first 5 draws;
  C. leave-one-gene-out jackknife standard error (exact, same weighted formula), normal CI;
  D. published subsample SD rescaled by sqrt(m/(n-m)) (m = 1,200, n = 1,500), normal CI.
Chance level: permute the Geneformer gene labels (1,000 draws).
Also: the same statistic for scGPT on the same 1,500 HVGs (symbols), with the corrected
row lookup and with the deployed (dict-position) lookup, for a like-for-like figure bar.
CPU only; no forward pass. Outputs: outputs/v2_crossmodel/pearson_1500hvg_ci.json, run_config.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import sys
sys.dont_write_bytecode = True

import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/spectral-geometry-217M"
OUT = RUN / "outputs/v2_crossmodel"
GF_DIR = Path("<HF_CACHE>/hub/models--ctheodoris--Geneformer"
              "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M")
GF_TOK = GF_DIR.parent / "geneformer/token_dictionary_gc104M.pkl"
MT_ST = PROJ / "setup/MaxToki-217M-HF/model.safetensors"
SC_PT = Path("<DATA_ROOT>/biodyn-work/subproject_53_scgpt_gpl_replication/"
             "embeddings/scgpt_whole_human_gene_embeddings.pt")
GENES = RUN / "outputs/phase0/gene_features.csv"
PUBLISHED = RUN / "outputs/phase9b/bootstrap_pearson.json"

N_BOOT = 2000
N_PERM = 1000
N_BOOT_SC = 1000
N_PERM_SC = 200
SEED_BOOT = 20261101
SEED_PERM = 20261102
SEED_BOOT_SC = 20261103
SEED_PERM_SC = 20261104


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_key(path, subs):
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if all(s in k for s in subs):
                return k, f.get_tensor(k).float().numpy()
    raise KeyError(subs)


def cosm(X):
    X = X.astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    X = X / n
    return X @ X.T


class PairStats:
    """Weighted Pearson over distinct gene pairs g<h with weight c_g*c_h."""

    def __init__(self, Sa, Sb):
        self.X = Sa.copy(); np.fill_diagonal(self.X, 0)
        self.Y = Sb.copy(); np.fill_diagonal(self.Y, 0)
        self.XX = self.X * self.X; self.YY = self.Y * self.Y; self.XY = self.X * self.Y

    def pearson(self, c):
        c = c.astype(np.float64)
        W = (c.sum() ** 2 - (c * c).sum()) / 2
        sx = c @ self.X @ c / 2; sy = c @ self.Y @ c / 2
        sxx = c @ self.XX @ c / 2; syy = c @ self.YY @ c / 2; sxy = c @ self.XY @ c / 2
        cov = sxy / W - (sx / W) * (sy / W)
        vx = sxx / W - (sx / W) ** 2; vy = syy / W - (sy / W) ** 2
        return float(cov / np.sqrt(vx * vy))


def direct_boot_pearson(Sa, Sb, idx):
    n = len(idx); iu = np.triu_indices(n, 1)
    gi = idx[iu[0]]; gj = idx[iu[1]]; keep = gi != gj
    return float(np.corrcoef(Sa[gi[keep], gj[keep]], Sb[gi[keep], gj[keep]])[0, 1])


def boot_block(Sa, Sb, n_boot, seed, part_file, check_direct=0, deadline=float("inf")):
    ps = PairStats(Sa, Sb)
    n = Sa.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, n, size=(n_boot, n))  # all draws fixed up front
    done = list(np.load(part_file)) if part_file.exists() else []
    checks = []
    for b in range(len(done), n_boot):
        c = np.bincount(idxs[b], minlength=n)
        done.append(ps.pearson(c))
        if b < check_direct:
            checks.append([done[-1], direct_boot_pearson(Sa, Sb, idxs[b])])
        if (b + 1) % 200 == 0:
            np.save(part_file, np.array(done))
            if time.time() > deadline:
                return None, checks
    np.save(part_file, np.array(done))
    return np.array(done), checks


def perm_null(Sa, Sb, n_perm, seed):
    n = Sa.shape[0]; iu = np.triu_indices(n, 1); a = Sa[iu]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        out.append(float(np.corrcoef(a, Sb[np.ix_(p, p)][iu])[0, 1]))
    return np.array(out)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    t0 = time.time()
    deadline = t0 + float(os.environ.get("BUDGET_S", "480"))
    OUT.mkdir(parents=True, exist_ok=True)
    res_f = OUT / "pearson_1500hvg_ci.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}

    _, gf = load_key(GF_DIR / "model.safetensors", ["embeddings.word_embeddings"])
    _, mt = load_key(MT_ST, ["embed_tokens"])
    g = pd.read_csv(GENES)
    ens = g["ensembl_id"].astype(str).tolist(); syms = g["symbol"].astype(str).tolist()
    mt_ids = g["maxtoki_token_id"].astype(int).to_numpy()
    tok = pickle.load(open(GF_TOK, "rb"))
    g_ids = np.array([tok.get(e, -1) for e in ens]); ok = (g_ids >= 0) & (g_ids < gf.shape[0])
    Sa = cosm(mt[mt_ids[ok]]); Sb = cosm(gf[g_ids[ok]])
    n = Sa.shape[0]; iu = np.triu_indices(n, 1)
    obs = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    res.update({"n_genes": int(n), "n_pairs": int(len(iu[0])), "observed_pearson": obs,
                "observed_spearman": float(pd.Series(Sa[iu]).corr(pd.Series(Sb[iu]), method="spearman")),
                "representation": "static input tables: MaxToki model.embed_tokens.weight vs Geneformer V2-316M "
                                  "bert.embeddings.word_embeddings.weight; 1,500 HVGs from spectral-geometry phase0 "
                                  "(Tabula Sapiens immune cells)"})

    # A. published subsample scheme, reproduced
    if "A_published_subsample_reproduced" not in res:
        rng = np.random.default_rng(42); m = int(0.8 * n); ius = np.triu_indices(m, 1); v = []
        for _ in range(1000):
            idx = rng.choice(n, size=m, replace=False)
            v.append(np.corrcoef(Sa[np.ix_(idx, idx)][ius], Sb[np.ix_(idx, idx)][ius])[0, 1])
        v = np.array(v)
        pub = json.loads(PUBLISHED.read_text())
        res["A_published_subsample_reproduced"] = {
            "method": "80% of genes without replacement, 1,000 draws, seed 42, percentile",
            "ci95": ci(v), "sd": float(v.std(ddof=1)), "published_ci95": pub["bootstrap_ci95"],
            "max_abs_diff_vs_published": float(max(abs(ci(v)[0] - pub["bootstrap_ci95"][0]),
                                                   abs(ci(v)[1] - pub["bootstrap_ci95"][1]))),
            "subsample_size_m": m}
        res_f.write_text(json.dumps(res, indent=2))

    # B. gene bootstrap with replacement
    if "B_gene_bootstrap_with_replacement" not in res:
        draws, checks = boot_block(Sa, Sb, N_BOOT, SEED_BOOT, OUT / "boot_gf_partial.npy", check_direct=5,
                                   deadline=deadline)
        if draws is None:
            print("budget reached in bootstrap; rerun to resume"); return
        res["B_gene_bootstrap_with_replacement"] = {
            "method": "resample 1,500 genes with replacement; drop pairs of two copies of one gene; weighted-pair "
                      "formula; percentile 2.5/97.5", "n_boot": N_BOOT, "seed": SEED_BOOT,
            "ci95": ci(draws), "mean": float(draws.mean()), "sd": float(draws.std(ddof=1)),
            "formula_vs_direct_indexing_first_draws": checks}
        res_f.write_text(json.dumps(res, indent=2))

    # C. jackknife
    if "C_jackknife" not in res:
        ps = PairStats(Sa, Sb)
        c1 = np.ones(n)
        assert abs(ps.pearson(c1) - obs) < 1e-9
        jk = []
        for i in range(n):
            c = c1.copy(); c[i] = 0; jk.append(ps.pearson(c))
        jk = np.array(jk)
        se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
        res["C_jackknife"] = {"method": "leave-one-gene-out jackknife SE; normal 95% CI obs +/- 1.96 SE",
                              "se": se, "ci95": [obs - 1.96 * se, obs + 1.96 * se]}
        res_f.write_text(json.dumps(res, indent=2))

    # D. rescaled subsample
    A = res["A_published_subsample_reproduced"]; m = A["subsample_size_m"]
    sd_r = A["sd"] * np.sqrt(m / (n - m))
    res["D_rescaled_subsample"] = {"method": "published subsample SD x sqrt(m/(n-m)) (finite-population "
                                             "correction for m-of-n without replacement); normal CI",
                                   "sd": float(sd_r), "ci95": [obs - 1.96 * sd_r, obs + 1.96 * sd_r]}

    # chance level
    if "chance_permutation" not in res:
        nul = perm_null(Sa, Sb, N_PERM, SEED_PERM)
        res["chance_permutation"] = {"method": "permute Geneformer gene labels", "n_perm": N_PERM, "seed": SEED_PERM,
                                     "mean": float(nul.mean()), "sd": float(nul.std(ddof=1)),
                                     "p95": float(np.percentile(nul, 95)), "max": float(nul.max()),
                                     "p_empirical_one_sided": float((1 + (nul >= obs).sum()) / (1 + N_PERM))}
        res_f.write_text(json.dumps(res, indent=2))

    # scGPT on the same HVGs
    if "scgpt_same_1500_hvg" not in res:
        sc = torch.load(SC_PT, map_location="cpu", weights_only=False)
        voc = sc["vocab"]; E = sc["normed_embeddings"].float().numpy()
        dep = {s.upper(): i for i, s in enumerate(voc)}
        up = {k.upper(): int(v) for k, v in voc.items()}
        fix_idx = np.array([int(voc[s]) if s in voc else up.get(s.upper(), -1) for s in syms])
        dep_idx = np.array([dep.get(s.upper(), -1) for s in syms])
        v = (dep_idx >= 0) & (fix_idx >= 0)
        Sm = cosm(mt[mt_ids[v]])
        out = {"n_genes": int(v.sum())}
        for tag, ix, seeds in [("fixed_lookup", fix_idx, (SEED_BOOT_SC, SEED_PERM_SC)),
                               ("deployed_dict_position_lookup", dep_idx, (SEED_BOOT_SC + 10, SEED_PERM_SC + 10))]:
            Ss = cosm(E[ix[v]])
            ius = np.triu_indices(Sm.shape[0], 1)
            o = float(np.corrcoef(Sm[ius], Ss[ius])[0, 1])
            draws, _ = boot_block(Sm, Ss, N_BOOT_SC, seeds[0], OUT / f"boot_scgpt_{tag}_partial.npy")
            nul = perm_null(Sm, Ss, N_PERM_SC, seeds[1])
            out[tag] = {"observed_pearson": o, "boot_ci95": ci(draws), "boot_sd": float(draws.std(ddof=1)),
                        "n_boot": N_BOOT_SC, "chance_mean": float(nul.mean()), "chance_sd": float(nul.std(ddof=1)),
                        "chance_p95": float(np.percentile(nul, 95)), "n_perm": N_PERM_SC}
        res["scgpt_same_1500_hvg"] = out
        res_f.write_text(json.dumps(res, indent=2))

    res["wall_seconds_last_run"] = time.time() - t0
    res_f.write_text(json.dumps(res, indent=2))
    cfg = {"item": "D10", "script": str(Path(__file__)), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seeds": {"boot": SEED_BOOT, "perm": SEED_PERM, "boot_scgpt": SEED_BOOT_SC, "perm_scgpt": SEED_PERM_SC,
                     "published_subsample": 42},
           "inputs": {}, "software": {"python": platform.python_version(), "numpy": np.__version__,
                                      "torch": torch.__version__, "pandas": pd.__version__},
           "device": "CPU only; no model forward pass"}
    for k, p in {"maxtoki_model_safetensors": MT_ST, "geneformer_model_safetensors": GF_DIR / "model.safetensors",
                 "geneformer_token_dictionary": GF_TOK, "scgpt_embeddings_pt": SC_PT, "hvg_gene_features": GENES,
                 "published_bootstrap_json": PUBLISHED}.items():
        rp = os.path.realpath(p)
        cfg["inputs"][k] = {"path": str(p), "resolved": rp, "sha256": sha256_file(rp)}
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "B_gene_bootstrap_with_replacement"}, indent=1))
    print(json.dumps(res.get("B_gene_bootstrap_with_replacement"), indent=1))


if __name__ == "__main__":
    main()
