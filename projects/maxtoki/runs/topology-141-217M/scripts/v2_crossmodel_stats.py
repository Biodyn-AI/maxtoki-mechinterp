"""v2 cross-model alignment (revision item D10), step 2: chance levels, gene-bootstrap CIs,
held-out (cross-validated) versions. CPU only. Resumable: each job writes its own JSON under
outputs/v2_crossmodel/jobs/, the bootstrap job also writes partial chunks.

Jobs per (model, domain), model in {gf, scgpt_fixed, scgpt_deployed}:
  obs       observed values with the deployed settings
  perm      chance level: permute the gene correspondence (rows of the other model),
            N_PERM draws. CCA and Pearson are recomputed; Procrustes is REFITTED for top-1
            (the deployed null did not refit). Deployed-style top-1 null also kept.
  permcheck 10 draws that refit PCA on the permuted full rows, to show that permuting the
            PCA rows (used in 'perm') gives the same CCA as permuting before PCA.
  boot      gene bootstrap WITH replacement, N_BOOT draws. Unit = gene. PCA, CCA and
            Procrustes are refitted on each resample. Pearson drops pairs made of two copies
            of one gene. Each draw also gets one permuted pairing inside the same resample
            (paired chance, rows permuted).
  bootgene  same resamples; chance inside each resample with WHOLE-GENE permutation, so copies
            stay paired with copies (shows that in-sample CCA / top-1 rise with copies even at chance).
  oob       same resamples; held-out (out-of-bag) canonical r and top-1: fit on the resample, score on
            genes the resample did not draw. This is the gene-bootstrap CI we recommend for CCA.
  cv        held-out versions: 10 x 5-fold split of genes; PCA, CCA and rotation fitted on
            training genes only; canonical r and top-1 measured on held-out genes. Chance level
            from N_CV_PERM permuted pairings (1 x 5-fold each).

Usage:  python v2_crossmodel_stats.py --all --budget 480
        python v2_crossmodel_stats.py --model gf --domain lung --part boot
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

import v2_crossmodel_common as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_PERM = 1000
N_BOOT = 1000
BOOT_CHUNK = 100
N_CV_REPEATS = 10
N_CV_PERM = 200
N_FOLDS = 5
MODEL_IDX = {"gf": 0, "scgpt_fixed": 1, "scgpt_deployed": 2}
PART_IDX = {"perm": 1, "permcheck": 2, "boot": 3, "cv": 4, "cvperm": 5, "bootgene": 6, "oob": 7}
JOBS = C.OUT / "jobs"


def seed_for(model, domain, part):
    return 20261001 + 1000 * MODEL_IDX[model] + 100 * C.DOMAINS.index(domain) + PART_IDX[part]


def get_pair(z, model, domain):
    if model == "gf":
        return z[f"{domain}__gf__mt"], z[f"{domain}__gf__other"]
    if model == "scgpt_fixed":
        return z[f"{domain}__sc__mt"], z[f"{domain}__sc__fixed"]
    if model == "scgpt_deployed":
        return z[f"{domain}__sc__mt"], z[f"{domain}__sc__deployed"]
    raise KeyError(model)


def job_obs(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    A = C.pca_reduce(A_full, d); B = C.pca_reduce(B_full, d)
    m, rs, nconv = C.cca_mean_r(A, B, k=min(C.TOP_K_CCA, d))
    Sa = C.cos_matrix(A_full); Sb = C.cos_matrix(B_full)
    iu = np.triu_indices(n, 1)
    return {"n_genes": n, "d_align": d, "cca_mean_r": m, "cca_all_r": rs, "cca_convergence_warnings": nconv,
            "pairwise_pearson": float(np.corrcoef(Sa[iu], Sb[iu])[0, 1]),
            "pairwise_spearman": float(pd.Series(Sa[iu]).corr(pd.Series(Sb[iu]), method="spearman")),
            "top1": C.top1_insample(A, B), "top1_uniform_chance_1_over_n": 1.0 / n}


def job_perm(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    A = C.pca_reduce(A_full, d); B = C.pca_reduce(B_full, d)
    obs = job_obs(A_full, B_full, model, domain)
    Sa = C.cos_matrix(A_full); Sb = C.cos_matrix(B_full)
    iu = np.triu_indices(n, 1); sa = Sa[iu]
    # deployed-style top-1 null: rotation fitted once on the true pairing, columns shuffled
    R = C.procrustes_R(A.astype(np.float64), B.astype(np.float64))
    sim_full = C.normalise(A.astype(np.float64) @ R) @ C.normalise(B.astype(np.float64)).T
    rng = np.random.default_rng(seed_for(model, domain, "perm"))
    cca_n, top1_n, top1_dep_n, pear_n = [], [], [], []
    for _ in range(N_PERM):
        p = rng.permutation(n)
        Bp = B[p]
        cca_n.append(C.cca_mean_r(A, Bp, k=min(C.TOP_K_CCA, d))[0])
        top1_n.append(C.top1_insample(A, Bp))
        top1_dep_n.append(float((np.argmax(sim_full[:, p], axis=1) == np.arange(n)).mean()))
        pear_n.append(float(np.corrcoef(sa, Sb[np.ix_(p, p)][iu])[0, 1]))
    return {
        "seed": seed_for(model, domain, "perm"), "n_perm": N_PERM,
        "method": "permute rows of the other model's PCA-30 table (gene correspondence); CCA refitted; "
                  "Procrustes refitted for top-1; Pearson on permuted full cosine matrix",
        "observed": obs,
        "cca_null": C.summarise_null(obs["cca_mean_r"], np.array(cca_n)),
        "top1_null_refit": C.summarise_null(obs["top1"], np.array(top1_n)),
        "top1_null_deployed_style_no_refit": C.summarise_null(obs["top1"], np.array(top1_dep_n)),
        "pearson_null": C.summarise_null(obs["pairwise_pearson"], np.array(pear_n)),
        "draws": {"cca": cca_n, "top1_refit": top1_n},
    }


def job_permcheck(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    A = C.pca_reduce(A_full, d); B = C.pca_reduce(B_full, d)
    rng = np.random.default_rng(seed_for(model, domain, "permcheck"))
    diffs = []
    for _ in range(10):
        p = rng.permutation(n)
        a = C.cca_mean_r(A, B[p], k=min(C.TOP_K_CCA, d))[0]
        b = C.cca_mean_r(A, C.pca_reduce(B_full[p], d), k=min(C.TOP_K_CCA, d))[0]
        diffs.append([a, b, abs(a - b)])
    return {"seed": seed_for(model, domain, "permcheck"),
            "rows_[pca_row_perm, pca_refit_on_permuted_rows, absdiff]": diffs,
            "max_abs_diff": float(max(x[2] for x in diffs))}


def job_boot(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    Sa = C.cos_matrix(A_full); Sb = C.cos_matrix(B_full)
    part_dir = JOBS / f"{model}__{domain}__boot_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(seed_for(model, domain, "boot"))
    # pre-draw all per-chunk seeds so chunks are independent of stopping points
    chunk_seeds = rng_master.integers(0, 2**31 - 1, size=N_BOOT // BOOT_CHUNK)
    for ci, cs in enumerate(chunk_seeds):
        f = part_dir / f"chunk_{ci:03d}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng(int(cs))
        rec = {k: [] for k in ("cca", "cca_null", "top1", "top1_null", "pearson", "n_unique")}
        for _ in range(BOOT_CHUNK):
            idx = rng.integers(0, n, size=n)
            q = rng.permutation(n)
            Ab = C.pca_reduce(A_full[idx], d); Bb = C.pca_reduce(B_full[idx], d)
            rec["cca"].append(C.cca_mean_r(Ab, Bb, k=min(C.TOP_K_CCA, d))[0])
            rec["top1"].append(C.top1_insample(Ab, Bb, ids=idx))
            rec["pearson"].append(C.offdiag_pearson(Sa, Sb, idx))
            Bq = Bb[q]
            rec["cca_null"].append(C.cca_mean_r(Ab, Bq, k=min(C.TOP_K_CCA, d))[0])
            Aq = Ab.astype(np.float64); Bq64 = Bq.astype(np.float64)
            R = C.procrustes_R(Aq, Bq64)
            rec["top1_null"].append(C.top1_from(Aq, Bq64, R, ids_q=idx[q], ids_c=idx[q]))
            rec["n_unique"].append(len(np.unique(idx)))
        np.savez(f, **{k: np.array(v) for k, v in rec.items()})
        print(f"  boot {model}/{domain} chunk {ci + 1}/{len(chunk_seeds)}", flush=True)
        if time.time() > DEADLINE:
            return None  # resume later
    parts = sorted(part_dir.glob("chunk_*.npz"))
    if len(parts) < len(chunk_seeds):
        return None
    agg = {k: np.concatenate([np.load(p)[k] for p in parts]) for k in
           ("cca", "cca_null", "top1", "top1_null", "pearson", "n_unique")}
    obs = job_obs(A_full, B_full, model, domain)
    ex_cca = agg["cca"] - agg["cca_null"]; ex_t1 = agg["top1"] - agg["top1_null"]
    out = {"seed": seed_for(model, domain, "boot"), "n_boot": int(len(agg["cca"])),
           "unit": "gene", "method": "resample genes with replacement (n draws of n); PCA-30, CCA and Procrustes "
                                     "refitted on each resample; Pearson drops pairs of two copies of the same gene; "
                                     "percentile 2.5/97.5",
           "observed": obs, "mean_unique_genes_per_resample": float(agg["n_unique"].mean())}
    for k in ("cca", "top1", "pearson", "cca_null", "top1_null"):
        v = agg[k]
        out[k] = {"ci95": C.pct_ci(v), "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
    out["cca_excess_over_paired_chance"] = {"ci95": C.pct_ci(ex_cca),
                                            "mean": float(np.nanmean(ex_cca))}
    out["top1_excess_over_paired_chance"] = {"ci95": C.pct_ci(ex_t1), "mean": float(np.nanmean(ex_t1))}
    return out


def job_bootgene(A_full, B_full, model, domain):
    """Chance level INSIDE each bootstrap resample, with the permutation done on whole genes
    (gene g of the other model is replaced by gene pi(g) everywhere), so copies made by the
    bootstrap stay paired with copies, as they are in the real pairing. This matters because
    in-sample CCA and top-1 rise when there are fewer distinct genes. Uses exactly the same
    resamples as job_boot (same chunk seeds, same RNG call order)."""
    n = A_full.shape[0]; d = C.d_align_for(n); k = min(C.TOP_K_CCA, d)
    part_dir = JOBS / f"{model}__{domain}__boot_parts"
    rng_master = np.random.default_rng(seed_for(model, domain, "boot"))
    chunk_seeds = rng_master.integers(0, 2**31 - 1, size=N_BOOT // BOOT_CHUNK)
    for ci, cs in enumerate(chunk_seeds):
        f = part_dir / f"genenull_{ci:03d}.npz"
        if f.exists():
            continue
        stored = np.load(part_dir / f"chunk_{ci:03d}.npz")
        rng = np.random.default_rng(int(cs))
        rng_g = np.random.default_rng(seed_for(model, domain, "bootgene") * 1000 + ci)
        rec = {"cca_null_gene": [], "top1_null_gene": [], "check_cca_recomputed": [], "check_cca_stored": []}
        for r in range(BOOT_CHUNK):
            idx = rng.integers(0, n, size=n)
            _q = rng.permutation(n)  # keep the same call order as job_boot
            pi = rng_g.permutation(n)
            Ab = C.pca_reduce(A_full[idx], d)
            Bg = C.pca_reduce(B_full[pi[idx]], d)
            rec["cca_null_gene"].append(C.cca_mean_r(Ab, Bg, k=k)[0])
            A64 = Ab.astype(np.float64); B64 = Bg.astype(np.float64)
            R = C.procrustes_R(A64, B64)
            rec["top1_null_gene"].append(C.top1_from(A64, B64, R, ids_q=idx, ids_c=idx))
            if r == 0:  # verify the resample is the same one job_boot used
                Bb = C.pca_reduce(B_full[idx], d)
                rec["check_cca_recomputed"].append(C.cca_mean_r(Ab, Bb, k=k)[0])
                rec["check_cca_stored"].append(float(stored["cca"][0]))
        np.savez(f, **{kk: np.array(v) for kk, v in rec.items()})
        print(f"  bootgene {model}/{domain} chunk {ci + 1}/{len(chunk_seeds)}", flush=True)
        if time.time() > DEADLINE:
            return None
    parts = sorted(part_dir.glob("genenull_*.npz"))
    if len(parts) < len(chunk_seeds):
        return None
    boot = {kk: np.concatenate([np.load(part_dir / f"chunk_{i:03d}.npz")[kk] for i in range(len(chunk_seeds))])
            for kk in ("cca", "top1")}
    gn = {kk: np.concatenate([np.load(p)[kk] for p in parts]) for kk in
          ("cca_null_gene", "top1_null_gene", "check_cca_recomputed", "check_cca_stored")}
    ex_c = boot["cca"] - gn["cca_null_gene"]; ex_t = boot["top1"] - gn["top1_null_gene"]
    return {"seed_gene_perm_base": seed_for(model, domain, "bootgene"), "n_boot": int(len(ex_c)),
            "method": "same gene-bootstrap resamples as 'boot'; chance per resample = CCA / top-1 after replacing each "
                      "gene of the other model by a randomly chosen gene (whole-gene permutation, copies stay paired); "
                      "one permutation per resample; percentile 2.5/97.5 of (resample value - resample chance)",
            "resample_check_max_abs_diff": float(np.max(np.abs(gn["check_cca_recomputed"] - gn["check_cca_stored"]))),
            "cca_null_gene": {"mean": float(gn["cca_null_gene"].mean()), "ci95": C.pct_ci(gn["cca_null_gene"])},
            "top1_null_gene": {"mean": float(gn["top1_null_gene"].mean()), "ci95": C.pct_ci(gn["top1_null_gene"])},
            "cca_excess": {"mean": float(ex_c.mean()), "ci95": C.pct_ci(ex_c), "sd": float(ex_c.std(ddof=1))},
            "top1_excess": {"mean": float(ex_t.mean()), "ci95": C.pct_ci(ex_t), "sd": float(ex_t.std(ddof=1))}}


def job_oob(A_full, B_full, model, domain):
    """Out-of-bag gene bootstrap of the HELD-OUT canonical correlation and held-out top-1.
    Same resamples as job_boot. PCA, CCA and Procrustes are fitted on the resample (with
    copies); scores are measured on the genes the resample did not draw (about 37%), so a copy
    of a gene can never be on both sides. Top-1 candidates = all n genes of the other model.
    Chance level for these held-out values comes from 'cvperm' (about 0 and about 1/n)."""
    n = A_full.shape[0]; d = C.d_align_for(n); k = min(C.TOP_K_CCA, d)
    part_dir = JOBS / f"{model}__{domain}__boot_parts"
    rng_master = np.random.default_rng(seed_for(model, domain, "boot"))
    chunk_seeds = rng_master.integers(0, 2**31 - 1, size=N_BOOT // BOOT_CHUNK)
    for ci, cs in enumerate(chunk_seeds):
        f = part_dir / f"oob_{ci:03d}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng(int(cs))
        rec = {"oob_cca_mean_r": [], "oob_cca_first_r": [], "oob_top1": [], "n_oob": []}
        for _ in range(BOOT_CHUNK):
            idx = rng.integers(0, n, size=n)
            _q = rng.permutation(n)  # keep the same call order as job_boot
            oob = np.setdiff1d(np.arange(n), idx)
            pA = PCA(n_components=d, random_state=42).fit(A_full[idx])
            pB = PCA(n_components=d, random_state=42).fit(B_full[idx])
            A_in, A_out = pA.transform(A_full[idx]), pA.transform(A_full[oob])
            B_all = pB.transform(B_full); B_in, B_out = pB.transform(B_full[idx]), B_all[oob]
            cca = CCA(n_components=k, max_iter=200).fit(A_in, B_in)
            a_o, b_o = cca.transform(A_out, B_out)
            rs = [float(np.corrcoef(a_o[:, i], b_o[:, i])[0, 1]) for i in range(k)]
            rec["oob_cca_mean_r"].append(float(np.mean(rs))); rec["oob_cca_first_r"].append(rs[0])
            R = C.procrustes_R(A_in.astype(np.float64), B_in.astype(np.float64))
            sim = C.normalise(A_out.astype(np.float64) @ R) @ C.normalise(B_all.astype(np.float64)).T
            rec["oob_top1"].append(float((np.argmax(sim, axis=1) == oob).mean()))
            rec["n_oob"].append(len(oob))
        np.savez(f, **{kk: np.array(v) for kk, v in rec.items()})
        print(f"  oob {model}/{domain} chunk {ci + 1}/{len(chunk_seeds)}", flush=True)
        if time.time() > DEADLINE:
            return None
    parts = sorted(part_dir.glob("oob_*.npz"))
    if len(parts) < len(chunk_seeds):
        return None
    agg = {kk: np.concatenate([np.load(p)[kk] for p in parts]) for kk in
           ("oob_cca_mean_r", "oob_cca_first_r", "oob_top1", "n_oob")}
    out = {"n_boot": int(len(agg["n_oob"])), "mean_n_oob_genes": float(agg["n_oob"].mean()),
           "method": "gene bootstrap with replacement (same resamples as 'boot'); fit on the resample, score on "
                     "out-of-bag genes; percentile 2.5/97.5. Trains on ~63% distinct genes, so values sit a little "
                     "below the 5-fold held-out values (which train on 80%)."}
    for kk in ("oob_cca_mean_r", "oob_cca_first_r", "oob_top1"):
        v = agg[kk]
        out[kk] = {"mean": float(np.nanmean(v)), "ci95": C.pct_ci(v), "sd": float(np.nanstd(v, ddof=1))}
    return out


def cv_once(A_full, B_full, d, kf_seed):
    """One 5-fold held-out pass. Returns (mean held-out canonical r, first held-out canonical r,
    held-out top-1 among all n candidates)."""
    n = A_full.shape[0]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=kf_seed)
    fold_mean, fold_first, hits = [], [], 0
    k = min(C.TOP_K_CCA, d)
    for tr, te in kf.split(np.arange(n)):
        pA = PCA(n_components=d, random_state=42).fit(A_full[tr])
        pB = PCA(n_components=d, random_state=42).fit(B_full[tr])
        A_tr, A_te = pA.transform(A_full[tr]), pA.transform(A_full[te])
        B_all = pB.transform(B_full); B_tr, B_te = B_all[tr], B_all[te]
        cca = CCA(n_components=k, max_iter=200).fit(A_tr, B_tr)
        a_te, b_te = cca.transform(A_te, B_te)
        rs = [float(np.corrcoef(a_te[:, i], b_te[:, i])[0, 1]) for i in range(k)]
        fold_mean.append(float(np.mean(rs))); fold_first.append(rs[0])
        R = C.procrustes_R(A_tr.astype(np.float64), B_tr.astype(np.float64))
        sim = C.normalise(A_te.astype(np.float64) @ R) @ C.normalise(B_all.astype(np.float64)).T
        hits += int((np.argmax(sim, axis=1) == te).sum())
    return float(np.mean(fold_mean)), float(np.mean(fold_first)), hits / n


def job_cv(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    base = seed_for(model, domain, "cv")
    obs = [cv_once(A_full, B_full, d, base + r) for r in range(N_CV_REPEATS)]
    obs = np.array(obs)
    return {"seed_base": base, "n_repeats": N_CV_REPEATS, "n_folds": N_FOLDS,
            "method": "5-fold split of genes; PCA-30 of each model, CCA(10, max_iter=200) and Procrustes fitted on "
                      "training genes only; canonical r = correlation of held-out canonical scores (mean of 10, "
                      "averaged over folds); top-1 = held-out gene's nearest neighbour among ALL n genes of the "
                      "other model; values are means over repeats, range = min-max over repeats",
           "cv_cca_mean_r": float(obs[:, 0].mean()), "cv_cca_mean_r_range": [float(obs[:, 0].min()), float(obs[:, 0].max())],
           "cv_cca_first_r": float(obs[:, 1].mean()),
           "cv_top1": float(obs[:, 2].mean()), "cv_top1_range": [float(obs[:, 2].min()), float(obs[:, 2].max())],
           "top1_uniform_chance_1_over_n": 1.0 / n}


def job_cvperm(A_full, B_full, model, domain):
    n = A_full.shape[0]; d = C.d_align_for(n)
    f = JOBS / f"{model}__{domain}__cvperm_partial.npy"
    rng = np.random.default_rng(seed_for(model, domain, "cvperm"))
    perms = [rng.permutation(n) for _ in range(N_CV_PERM)]
    done = list(np.load(f)) if f.exists() else []
    for i in range(len(done), N_CV_PERM):
        done.append(cv_once(A_full, B_full[perms[i]], d, seed_for(model, domain, "cv")))
        if (i + 1) % 20 == 0:
            np.save(f, np.array(done))
            print(f"  cvperm {model}/{domain} {i + 1}/{N_CV_PERM}", flush=True)
            if time.time() > DEADLINE and i + 1 < N_CV_PERM:
                return None
    np.save(f, np.array(done))
    arr = np.array(done)
    cvj = json.loads((JOBS / f"{model}__{domain}__cv.json").read_text()) if (JOBS / f"{model}__{domain}__cv.json").exists() else None
    out = {"seed": seed_for(model, domain, "cvperm"), "n_perm": N_CV_PERM,
           "method": "same 5-fold held-out pass (first split seed) after permuting the gene correspondence"}
    for j, key in enumerate(["cv_cca_mean_r", "cv_cca_first_r", "cv_top1"]):
        obs_v = cvj[key] if cvj else float("nan")
        out[key + "_null"] = C.summarise_null(obs_v, arr[:, j])
    return out


FUNCS = {"obs": job_obs, "perm": job_perm, "permcheck": job_permcheck, "boot": job_boot,
         "cv": job_cv, "cvperm": job_cvperm, "bootgene": job_bootgene, "oob": job_oob}
DEADLINE = float("inf")


def run_job(z, model, domain, part):
    out_f = JOBS / f"{model}__{domain}__{part}.json"
    if out_f.exists():
        return True
    A_full, B_full = get_pair(z, model, domain)
    t = time.time()
    res = FUNCS[part](A_full.astype(np.float64), B_full.astype(np.float64), model, domain)
    if res is None:
        return False
    res["wall_seconds"] = time.time() - t
    res["model"] = model; res["domain"] = domain; res["part"] = part
    C.write_json(out_f, res)
    print(f"[done] {model}/{domain}/{part} in {res['wall_seconds']:.1f}s", flush=True)
    return True


def main():
    global DEADLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--domain"); ap.add_argument("--part")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parts", default="obs,perm,permcheck,boot,bootgene,oob,cv,cvperm")
    ap.add_argument("--models", default="gf,scgpt_fixed,scgpt_deployed")
    ap.add_argument("--budget", type=float, default=480.0)
    a = ap.parse_args()
    DEADLINE = time.time() + a.budget
    JOBS.mkdir(parents=True, exist_ok=True)
    z = C.load_prepared()
    if a.all:
        todo = [(m, d, p) for p in a.parts.split(",") for m in a.models.split(",") for d in C.DOMAINS]
    else:
        todo = [(a.model, a.domain, a.part)]
    for m, d, p in todo:
        if p in ("cv", "cvperm", "boot", "bootgene", "oob") and m == "scgpt_deployed":
            continue  # wrong-gene table; its chance level is covered by 'perm'
        if time.time() > DEADLINE:
            print("budget reached; rerun to resume"); break
        ok = run_job(z, m, d, p)
        if not ok:
            print("budget reached inside job; rerun to resume"); break
    left = [(m, d, p) for m, d, p in todo if not (p in ("cv", "cvperm", "boot", "bootgene", "oob") and m == "scgpt_deployed")
            and not (JOBS / f"{m}__{d}__{p}.json").exists()]
    print("remaining jobs:", len(left), left[:6])


if __name__ == "__main__":
    main()
