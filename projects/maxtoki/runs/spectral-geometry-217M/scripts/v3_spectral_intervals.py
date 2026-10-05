"""v3 spectral (item V3-5): jackknife intervals and report tables.

Why a second script
-------------------
scripts/v3_spectral.py computed the point estimates and percentile bootstraps. Both bootstraps
turned out to be shifted for this design:
  * gene bootstrap (genes drawn with replacement): duplicated genes are identical rows in both
    matrices, which pushes CKA UP (the point estimate falls at or below the lower CI edge);
  * cell-block bootstrap (10 blocks of 200 cells drawn with replacement): a resample holds only
    ~63% distinct cells, so per-gene means are noisier, which pushes CKA DOWN and ER UP.
So the main intervals here are delete-a-group jackknife intervals, which never duplicate a gene
or a cell:
  * over genes: 30 random groups of 50 genes (fixed seed), delete one group at a time;
  * over cells: the 10 random blocks of 200 cells of each sample, delete one block at a time
    (for the 3-sample CKA the same block index is deleted in all three samples).
  95% CI = estimate +/- t(0.975, g-1) * SE_jack, SE_jack = sqrt((g-1)/g * sum((theta_i - mean)^2)).
The bootstrap SDs from v3_spectral.py are reported next to them as a second estimate of the
interval width. v3_spectral.py is not changed (its extraction outputs depend on it).

Usage (from projects/maxtoki):
  OMP_NUM_THREADS=4 .venv/bin/python runs/spectral-geometry-217M/scripts/v3_spectral_intervals.py <part>
  parts: er | cka | half | tables      (er, cka, half are resumable; each fits in one call)
"""
from __future__ import annotations

import argparse
import json

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_spectral as V  # noqa: E402

H = V.H
ANA = V.ANA
OUT_JSON = ANA / "intervals.json"
SCRIPT_PATH = Path(__file__).resolve()
JK_SEED = 20261007
N_GENE_GROUPS = 30
ALL_JOBS = V.JOBS + V.OPTIONAL_JOBS


def tq(g: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, g - 1))


def jk(est: float, reps) -> dict:
    reps = np.asarray(reps, dtype=np.float64)
    g = len(reps)
    se = float(np.sqrt((g - 1) / g * np.sum((reps - reps.mean()) ** 2)))
    q = tq(g)
    return {"est": float(est), "se": se, "ci": [float(est - q * se), float(est + q * se)], "g": g,
            "jk_bias": float((g - 1) * (reps.mean() - est))}


def er_eig(M: np.ndarray) -> float:
    """Effective rank from the eigenvalues of the float64 covariance (= exp entropy of sigma^2 shares)."""
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(Mc.T @ Mc), 0, None)
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def cka_gram(Ka: np.ndarray, Kb: np.ndarray) -> float:
    def c(K):
        a = K.mean(axis=0)
        return K - a[:, None] - a[None, :] + a.mean()
    Ca, Cb = c(Ka), c(Kb)
    return float(np.vdot(Ca, Cb) / np.sqrt(np.vdot(Ca, Ca) * np.vdot(Cb, Cb)))


def gene_groups(n: int, seed: int = JK_SEED) -> list:
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[k::N_GENE_GROUPS]) for k in range(N_GENE_GROUPS)]


def load_res() -> dict:
    return V.read_json(OUT_JSON, {}) or {}


def save_res(res):
    res["code"] = {"script": str(SCRIPT_PATH), "script_sha256": V.sha256_file(SCRIPT_PATH),
                   "v3_spectral_sha256": V.sha256_file(V.SCRIPT_PATH)}
    V.write_json(OUT_JSON, res)


def block_means(acc, cnt, genes, layer, w):
    """Per-gene means for slot weights w (N_SLOTS,), one layer, float64."""
    c = (w[:, None] * cnt[:, genes]).sum(axis=0)
    tot = np.zeros((len(genes), V.HIDDEN))
    for sl in np.nonzero(w)[0]:
        tot += w[sl] * np.asarray(acc[sl][layer][genes], dtype=np.float64)
    ok = c > 0
    M = np.zeros_like(tot)
    M[ok] = tot[ok] / c[ok][:, None]
    return M, ok


# =============================================================================
def part_er(args, t0):
    import pandas as pd
    res = load_res()
    R = res.setdefault("er_jackknife_cells", {})
    panel = V.read_json(V.PREP / "panel.json")
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    # deployed-code metrics for the optional jobs (v3_spectral.py analyze --part metrics covers JOBS only)
    opt_csv = ANA / "per_layer_metrics_optional.csv"
    have = set(pd.read_csv(opt_csv)["label"]) if opt_csv.exists() else set()
    rows = pd.read_csv(opt_csv).to_dict("records") if opt_csv.exists() else []
    for job in V.OPTIONAL_JOBS:
        lab = f"v3_{job}_panelD_all"
        if lab in have or not V.job_done(job):
            continue
        H.check_memory(V.MIN_FREE_GB)
        E, c = V._job_means(job, V.W_ALL, D_in_U)
        r = V.layer_metrics(E, c, lab)
        rows.extend(r)
        pd.DataFrame(rows).to_csv(opt_csv, index=False)
        print(f"  {lab}: ER " + " ".join(f"{x['effective_rank']:.1f}" for x in r)
              + f" | shuffle L11 {r[-1]['feature_shuffle_er']:.1f} ({time.time() - t0:.0f}s)")
    for job in ALL_JOBS:
        if job in R or not V.job_done(job):
            continue
        if time.time() - t0 > args.max_minutes * 60 - 90:
            print("[er] time budget reached; run again")
            break
        H.check_memory(V.MIN_FREE_GB)
        acc, cnt = V.load_job(job)
        full = np.r_[np.ones(V.N_BLOCKS), 0.0]
        out = {}
        reps_all = np.zeros((V.N_BLOCKS, V.N_STATES))
        est_all = np.zeros(V.N_STATES)
        for li in range(V.N_STATES):
            M, ok = block_means(acc, cnt, D_in_U, li, full)
            est_all[li] = er_eig(M[ok])
            for k in range(V.N_BLOCKS):
                w = full.copy()
                w[k] = 0.0
                Mk, okk = block_means(acc, cnt, D_in_U, li, w)
                reps_all[k, li] = er_eig(Mk[okk])
        for li in range(V.N_STATES):
            out[str(li)] = jk(est_all[li], reps_all[:, li])
        out["ratio_L1_over_L11"] = jk(est_all[1] / est_all[11], reps_all[:, 1] / reps_all[:, 11])
        out["ratio_L0_over_L11"] = jk(est_all[0] / est_all[11], reps_all[:, 0] / reps_all[:, 11])
        out["drop_L1_to_L11"] = jk(1 - est_all[11] / est_all[1], 1 - reps_all[:, 11] / reps_all[:, 1])
        out["method"] = "delete-one-block jackknife over the 10 random 200-cell blocks; ER by the eigenvalue route"
        R[job] = out
        np.save(ANA / f"er_jackknife_reps_{job}.npy", reps_all)
        save_res(res)
        print(f"  {job}: ER L1 {est_all[1]:.1f} ±{out['1']['se']:.2f}  L11 {est_all[11]:.1f} ±{out['11']['se']:.2f}  "
              f"L1/L11 {out['ratio_L1_over_L11']['est']:.3f} CI {np.round(out['ratio_L1_over_L11']['ci'], 3)} "
              f"({time.time() - t0:.0f}s)")
    save_res(res)


def part_cka(args, t0):
    """3-sample CKA: gene jackknife (deployed, v3_trained, v3_rand0) and cell jackknife (v3 sets)."""
    res = load_res()
    R = res.setdefault("cka3_jackknife", {})
    panel = V.read_json(V.PREP / "panel.json")
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    for name, jobs in V.CKA_SETS.items():
        if name in R and R[name].get("complete"):
            continue
        if jobs is not None and not all(V.job_done(j) for j in jobs):
            print(f"  {name}: jobs not finished; skipped")
            continue
        H.check_memory(V.MIN_FREE_GB)
        S = R.setdefault(name, {"per_layer": {}})
        if jobs is None:
            Es = [np.load(V.DEP0 / "layer_gene_embeddings.npy", mmap_mode="r"),
                  np.load(V.DEP8 / "embeddings_seed43.npy", mmap_mode="r"),
                  np.load(V.DEP8 / "embeddings_seed44.npy", mmap_mode="r")]
            loaded = None
            keep = np.ones(V.N_HVG, dtype=bool)
        else:
            loaded = [V.load_job(j) for j in jobs]
            keep = np.all([c[:V.N_BLOCKS].sum(axis=0)[D_in_U] > 0 for _, c in loaded], axis=0)
        groups = gene_groups(int(keep.sum()))
        full = np.r_[np.ones(V.N_BLOCKS), 0.0]
        for li in range(1, V.N_STATES):
            if str(li) in S["per_layer"]:
                continue
            if time.time() - t0 > args.max_minutes * 60 - 60:
                save_res(res)
                print("[cka] time budget reached; run again")
                return
            if loaded is None:
                Ms = [np.asarray(E[li], dtype=np.float64)[keep] for E in Es]
            else:
                Ms = [block_means(a, c, D_in_U, li, full)[0][keep] for a, c in loaded]
            Ks = [M @ M.T for M in Ms]
            est_p = [cka_gram(Ks[a], Ks[b]) for a, b in V.PAIRS]
            est = float(np.mean(est_p))
            reps = []
            n = Ks[0].shape[0]
            for gi in groups:
                m = np.ones(n, dtype=bool)
                m[gi] = False
                reps.append(np.mean([cka_gram(Ks[a][np.ix_(m, m)], Ks[b][np.ix_(m, m)]) for a, b in V.PAIRS]))
            rec = {"est_mean": est, "est_pairs": est_p, "gene_jk": jk(est, reps),
                   "one_minus_cka_gene_jk": jk(1 - est, 1 - np.array(reps))}
            if loaded is not None:
                creps = []
                for k in range(V.N_BLOCKS):
                    w = full.copy()
                    w[k] = 0.0
                    Mk = [block_means(a, c, D_in_U, li, w) for a, c in loaded]
                    okk = keep & Mk[0][1] & Mk[1][1] & Mk[2][1]
                    Kk = [M[okk] @ M[okk].T for M, _ in Mk]
                    creps.append(np.mean([cka_gram(Kk[a], Kk[b]) for a, b in V.PAIRS]))
                rec["cell_jk"] = jk(est, creps)
            S["per_layer"][str(li)] = rec
            save_res(res)
            print(f"  {name} L{li}: CKA {est:.5f} gene-jk CI {np.round(rec['gene_jk']['ci'], 5)}"
                  + (f" cell-jk CI {np.round(rec['cell_jk']['ci'], 5)}" if "cell_jk" in rec else "")
                  + f" ({time.time() - t0:.0f}s)")
        S["complete"] = True
        S["genes_used"] = int(keep.sum())
        S["method"] = ("CKA via centred gram matrices (float64; equals phase8 linear_cka); gene jackknife: 30 random "
                       "groups of 50 genes; cell jackknife: delete block k (200 cells) in all three samples")
        save_res(res)


def part_half(args, t0):
    """Split-half CKA (blocks 0-4 vs 5-9): gene jackknife and cell jackknife."""
    res = load_res()
    R = res.setdefault("cka_half_jackknife", {})
    panel = V.read_json(V.PREP / "panel.json")
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    for job in ALL_JOBS:
        if (job in R and R[job].get("complete")) or not V.job_done(job):
            continue
        H.check_memory(V.MIN_FREE_GB)
        acc, cnt = V.load_job(job)
        S = R.setdefault(job, {"per_layer": {}})
        wh = []
        for hb in V.HALVES:
            w = np.zeros(V.N_SLOTS)
            w[hb] = 1.0
            wh.append(w)
        keep = np.all([(w[:, None] * cnt).sum(axis=0)[D_in_U] > 0 for w in wh], axis=0)
        groups = gene_groups(int(keep.sum()))
        for li in range(1, V.N_STATES):
            if str(li) in S["per_layer"]:
                continue
            if time.time() - t0 > args.max_minutes * 60 - 45:
                save_res(res)
                print("[half] time budget reached; run again")
                return
            Ms = [block_means(acc, cnt, D_in_U, li, w)[0][keep] for w in wh]
            Ks = [M @ M.T for M in Ms]
            est = cka_gram(Ks[0], Ks[1])
            n = Ks[0].shape[0]
            greps = []
            for gi in groups:
                m = np.ones(n, dtype=bool)
                m[gi] = False
                greps.append(cka_gram(Ks[0][np.ix_(m, m)], Ks[1][np.ix_(m, m)]))
            creps = []
            for k in range(V.N_BLOCKS):
                ws = [w.copy() for w in wh]
                for w in ws:
                    w[k] = 0.0
                Mk = [block_means(acc, cnt, D_in_U, li, w) for w in ws]
                okk = keep & Mk[0][1] & Mk[1][1]
                creps.append(cka_gram(Mk[0][0][okk] @ Mk[0][0][okk].T, Mk[1][0][okk] @ Mk[1][0][okk].T))
            S["per_layer"][str(li)] = {"est": est, "gene_jk": jk(est, greps), "cell_jk": jk(est, creps)}
            save_res(res)
        S["complete"] = True
        save_res(res)
        print(f"  {job}: split-half CKA L1 {S['per_layer']['1']['est']:.4f} L11 {S['per_layer']['11']['est']:.4f} "
              f"gene-jk CI {np.round(S['per_layer']['11']['gene_jk']['ci'], 4)} cell-jk CI "
              f"{np.round(S['per_layer']['11']['cell_jk']['ci'], 4)} ({time.time() - t0:.0f}s)")


# =============================================================================
def part_tables(args, t0):
    import pandas as pd
    from scipy.stats import spearmanr
    res = load_res()
    met = pd.read_csv(ANA / "per_layer_metrics.csv")
    if (ANA / "per_layer_metrics_optional.csv").exists():
        met = pd.concat([met, pd.read_csv(ANA / "per_layer_metrics_optional.csv")], ignore_index=True)
    cka = V.read_json(ANA / "cka.json", {}) or {}

    erj = res.get("er_jackknife_cells", {})
    ckj = res.get("cka3_jackknife", {})
    hj = res.get("cka_half_jackknife", {})
    L = []

    def er(label):
        d = met[met["label"] == label].sort_values("layer")
        return d["effective_rank"].to_numpy(), d

    # ---- Table 1: ER per layer ----
    labs = [("Deployed encoding, sample A (phase0 file)", "deployed_A_phase0", None),
            ("v3 trained, sample A", "v3_trained_A_panelD_all", "trained_A"),
            ("v3 trained, sample B", "v3_trained_B_panelD_all", "trained_B"),
            ("v3 trained, sample C", "v3_trained_C_panelD_all", "trained_C"),
            ("v3 random init seed 0, sample A", "v3_rand0_A_panelD_all", "rand0_A"),
            ("v3 random init seed 1, sample A", "v3_rand1_A_panelD_all", "rand1_A"),
            ("v3 random init seed 2, sample A", "v3_rand2_A_panelD_all", "rand2_A"),
            ("v3 random init seed 0, sample B", "v3_rand0_B_panelD_all", "rand0_B"),
            ("v3 random init seed 0, sample C", "v3_rand0_C_panelD_all", "rand0_C")]
    L.append("Table 1. Effective rank per layer (panel D, all cells, deployed code)\n")
    L.append("| Model / sample | " + " | ".join(f"L{i}" for i in range(12)) + " |")
    L.append("|---|" + "---|" * 12)
    for name, lab, _ in labs:
        if lab not in set(met["label"]):
            continue
        e, _ = er(lab)
        L.append(f"| {name} | " + " | ".join(f"{x:.0f}" for x in e) + " |")
    L.append("")
    L.append("Table 1b. Jackknife SE over cells (delete one 200-cell block of 10)\n")
    L.append("| Model / sample | " + " | ".join(f"L{i}" for i in range(12)) + " |")
    L.append("|---|" + "---|" * 12)
    for name, lab, job in labs:
        if job in erj:
            L.append(f"| {name} | " + " | ".join(f"{erj[job][str(i)]['se']:.1f}" for i in range(12)) + " |")
    L.append("")

    # ---- Table 2: compression summary ----
    L.append("Table 2. Compression summary\n")
    L.append("| Model / sample | ER L0 | ER L1 | ER L11 | L1/L11 [95% CI] | L0/L11 [95% CI] | Spearman rho (L0-L11) | "
             "Spearman rho (L1-L11) | Feature-shuffle ER at L11 | ER L11 / shuffle |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for name, lab, job in labs + [("Deployed encoding, Phase 8 seed 43", "deployed_seed43_phase8", None),
                                  ("Deployed encoding, Phase 8 seed 44", "deployed_seed44_phase8", None)]:
        if lab not in set(met["label"]):
            continue
        e, d = er(lab)
        r0, p0 = spearmanr(np.arange(12), e)
        r1, p1 = spearmanr(np.arange(1, 12), e[1:])
        shuf = float(d["feature_shuffle_er"].iloc[-1])
        c1 = erj[job]["ratio_L1_over_L11"]["ci"] if job in erj else None
        c0 = erj[job]["ratio_L0_over_L11"]["ci"] if job in erj else None
        fmt = (lambda v, c: f"{v:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else ""))
        L.append(f"| {name} | {e[0]:.1f} | {e[1]:.1f} | {e[11]:.1f} | {fmt(e[1] / e[11], c1)} | {fmt(e[0] / e[11], c0)} | "
                 f"{r0:.3f} (p={p0:.1g}) | {r1:.3f} (p={p1:.1g}) | {shuf:.1f} | {e[11] / shuf:.3f} |")
        summ[lab] = {"er": e.tolist(), "rho0": [float(r0), float(p0)], "rho1": [float(r1), float(p1)], "shuffle_L11": shuf}
    L.append("")
    # trained vs random contrast
    tr = [summ[f"v3_trained_{s}_panelD_all"]["er"] for s in "ABC" if f"v3_trained_{s}_panelD_all" in summ]
    rd = [summ[f"v3_rand{s}_A_panelD_all"]["er"] for s in (0, 1, 2) if f"v3_rand{s}_A_panelD_all" in summ]
    contrast = {}
    if tr and rd:
        tr, rd = np.array(tr), np.array(rd)
        rt = tr[:, 1] / tr[:, 11]
        rr = rd[:, 1] / rd[:, 11]
        r0t = tr[:, 0] / tr[:, 11]
        r0r = rd[:, 0] / rd[:, 11]
        contrast = {"trained_L1_over_L11_samples": rt.tolist(), "random_L1_over_L11_seeds": rr.tolist(),
                    "trained_L0_over_L11_samples": r0t.tolist(), "random_L0_over_L11_seeds": r0r.tolist(),
                    "trained_drop_L1_L11": (1 - tr[:, 11] / tr[:, 1]).tolist(),
                    "random_drop_L1_L11": (1 - rd[:, 11] / rd[:, 1]).tolist(),
                    "trained_ER_L11_samples": tr[:, 11].tolist(), "random_ER_L11_seeds": rd[:, 11].tolist()}
        L.append("Trained vs random init (sample A for random; A, B, C for trained):")
        L.append(f"- ER L1/L11: trained {rt.min():.2f}-{rt.max():.2f} (3 samples); random {rr.min():.2f}-{rr.max():.2f} (3 seeds)")
        L.append(f"- ER L0/L11: trained {r0t.min():.2f}-{r0t.max():.2f}; random {r0r.min():.2f}-{r0r.max():.2f}")
        L.append(f"- Share of ER lost from L1 to L11: trained {100 * (1 - tr[:, 11] / tr[:, 1]).min():.1f}-"
                 f"{100 * (1 - tr[:, 11] / tr[:, 1]).max():.1f}%; random {100 * (1 - rd[:, 11] / rd[:, 1]).min():.1f}-"
                 f"{100 * (1 - rd[:, 11] / rd[:, 1]).max():.1f}%")
        L.append("")
    # sensitivity rows
    L.append("Table 3. Sensitivity: ER at L1 / L6 / L11 for other cell sets and panels\n")
    L.append("| Label | ER L1 | ER L6 | ER L11 | L1/L11 |")
    L.append("|---|---|---|---|---|")
    for lab in sorted(met["label"].unique()):
        if lab.endswith("_umi") or "panelS" in lab:
            e, _ = er(lab)
            L.append(f"| {lab} | {e[1]:.1f} | {e[6]:.1f} | {e[11]:.1f} | {e[1] / e[11]:.2f} |")
    L.append("")

    # ---- Table 4: CKA ----
    L.append("Table 4. Linear CKA across three disjoint 2,000-cell samples (mean of 3 pairs), L1-L11\n")
    L.append("| Layer | Deployed encoding (seeds 42/43/44; 6-8 shared cells per pair) | gene-jk 95% CI | "
             "v3 trained (A/B/C, disjoint) | gene-jk 95% CI | cell-jk 95% CI | gene-boot SD | cell-boot SD | "
             "v3 random init seed 0 (A/B/C) | gene-jk 95% CI | cell-jk 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for li in range(1, 12):
        row = [f"L{li}"]
        for name in ("deployed", "v3_trained", "v3_rand0"):
            p = ckj.get(name, {}).get("per_layer", {}).get(str(li))
            if p is None:
                row += ["—", "—"] + (["—"] if name != "deployed" else []) + (["—", "—"] if name == "v3_trained" else [])
                continue
            row.append(f"{p['est_mean']:.4f}")
            row.append(f"[{p['gene_jk']['ci'][0]:.4f}, {p['gene_jk']['ci'][1]:.4f}]")
            if name != "deployed":
                row.append(f"[{p['cell_jk']['ci'][0]:.4f}, {p['cell_jk']['ci'][1]:.4f}]")
            if name == "v3_trained":
                b = cka.get("v3_trained", {}).get("per_layer", {}).get(str(li), {})
                row.append(f"{b.get('gene_boot_sd_mean', float('nan')):.5f}")
                row.append(f"{b.get('cell_boot_sd_mean', float('nan')):.5f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Table 5. Split-half CKA inside one 2,000-cell sample (1,000 vs 1,000 cells), L1, L6, L11\n")
    L.append("| Job | L1 | L6 | L11 [gene-jk 95% CI] [cell-jk 95% CI] |")
    L.append("|---|---|---|---|")
    for job in ALL_JOBS:
        if job not in hj:
            continue
        p = hj[job]["per_layer"]
        L.append(f"| {job} | {p['1']['est']:.4f} | {p['6']['est']:.4f} | {p['11']['est']:.4f} "
                 f"[{p['11']['gene_jk']['ci'][0]:.4f}, {p['11']['gene_jk']['ci'][1]:.4f}] "
                 f"[{p['11']['cell_jk']['ci'][0]:.4f}, {p['11']['cell_jk']['ci'][1]:.4f}] |")
    L.append("")
    (ANA / "report_tables.md").write_text("\n".join(L))
    res["contrast"] = contrast
    save_res(res)
    print("\n".join(L))


def part_index(args, t0):
    """Top-level outputs/v3_spectral/run_config.json: what is where, devices, versions, seeds, wall times."""
    jobs = {}
    for job in ALL_JOBS:
        rc = V.read_json(V.job_paths(job)["dir"] / "run_config.json", {}) or {}
        if not rc:
            continue
        jobs[job] = {"model": rc["model"], "sample": rc["sample"], "n_cells": len(rc["cells"]),
                     "script_sha256": rc["code"]["script_sha256"], "wall_seconds_total": rc["wall_seconds_total"],
                     "n_chunks": len(rc["chunks"]),
                     "min_free_gb": min(c["min_free_gb"] for c in rc["chunks"]),
                     "max_mps_driver_gb": max(c["mps_driver_peak_gb"] for c in rc["chunks"]),
                     "encoding_check": rc["input_encoding"]["per_forward_check"]}
    prc = V.read_json(V.PREP / "run_config.json")
    idx = {
        "item": "V3-5 spectral geometry with correct inputs and a random-initialised control",
        "time": V.now(), "env": prc["env"],
        "folders": {"prepare": "cells, panels, tokens, order agreement with the deployed tokens",
                    "extract/<job>": "per-job block sums of hidden states (MPS forward passes)",
                    "analysis": "metrics, CKA, split-half CKA, bootstraps, jackknife intervals, report tables",
                    "verify": "second-way checks (V1-V7)",
                    "code_versions": "exact copies of scripts/v3_spectral.py for each sha256 used by an extraction job"},
        "seeds": prc["seeds"], "random_init_seeds": list(V.RANDOM_INIT_SEEDS), "boot_seed": V.BOOT_SEED,
        "jackknife_seed": JK_SEED,
        "samples": {s: {"n": len(prc["cells"][s]), "counts_sha256": prc["input_encoding"][s]["counts_sha256"],
                        "tokens_sha256": prc["input_encoding"][s]["encoding_check"]["tokens_sha256"]} for s in V.SAMPLES},
        "deployed_sample_overlaps": prc["deployed_sample_overlaps"],
        "jobs": jobs,
        "mps_wall_seconds_total": round(sum(j["wall_seconds_total"] for j in jobs.values()), 1),
        "code": {"v3_spectral.py": V.sha256_file(V.SCRIPT_PATH), "v3_spectral_intervals.py": V.sha256_file(SCRIPT_PATH),
                 "inputs_v3.py": V.sha256_file(V.PROJ / "setup/inputs_v3.py"),
                 "hooks_v2.py": V.sha256_file(V.PROJ / "setup/hooks_v2.py")},
    }
    V.write_json(V.OUT / "run_config.json", idx)
    print(json.dumps({k: idx[k] for k in ("mps_wall_seconds_total", "code")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["er", "cka", "half", "tables", "index"])
    ap.add_argument("--max-minutes", type=float, default=7.0)
    args = ap.parse_args()
    t0 = time.time()
    {"er": part_er, "cka": part_cka, "half": part_half, "tables": part_tables, "index": part_index}[args.part](args, t0)
    V.update_run_config(ANA, f"intervals_{args.part}", {"time": V.now(), "script": str(SCRIPT_PATH),
                                                       "script_sha256": V.sha256_file(SCRIPT_PATH),
                                                       "jk_seed": JK_SEED, "n_gene_groups": N_GENE_GROUPS,
                                                       "wall_s": round(time.time() - t0, 1)})
    print(f"[{args.part}] {time.time() - t0:.0f}s; peak RSS {V.peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
