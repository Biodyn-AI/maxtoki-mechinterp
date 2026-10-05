"""Independent check of v2b_annotation_fdr.py (item N01, SAE feature annotation rates).

Written separately from v2b_annotation_fdr.py. It does not import it or read its outputs,
except at the very end (`compare`), where the two sets of numbers are put side by side.

What is done differently on purpose:
  * term sets are rebuilt from the raw GO / KEGG / Reactome / TRRUST files and the Replogle var list
  * hypergeometric tail p-values come from my own log-gamma sum (checked against scipy at the end)
  * overlaps come from a sparse (features x genes) times dense (genes x terms) product
  * BH q-values come from my own code (sorted p * m / rank, then running minimum from the end),
    with the deployed convention "q < 0.05"; I also count the "<= 0.05" form
  * random-list nulls use other samplers and other seeds:
      es     frequency-weighted lists, Efraimidis-Spirakis keys (exact successive sampling,
             each next gene drawn with probability proportional to its count among genes not yet drawn)
      slot   exact margins: pool every gene slot of the real lists, shuffle, cut into lists of the
             real sizes, then remove within-list duplicates by random swaps that keep both margins
  * the nulls are also scored under the bio-sae rule (overlap >= 2, one pooled BH), which the first
    run did not do

CPU only. No model forward pass.

Usage (project venv python):
  python v2b_annotation_fdr_verify.py prep
  python v2b_annotation_fdr_verify.py real --layers 0-11
  python v2b_annotation_fdr_verify.py null --layers 0-3 --reps 10
  python v2b_annotation_fdr_verify.py summary
  python v2b_annotation_fdr_verify.py compare
  python v2b_annotation_fdr_verify.py extra
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.stats import hypergeom

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/sae-atlas-217M"
OUTPUTS = RUN / "outputs"
OUT = OUTPUTS / "v2b_annotation_fdr_verify"
FIRST = OUTPUTS / "v2b_annotation_fdr"
BIOM = Path("<DATA_ROOT>/biodyn-nmi-paper")
REPLOGLE_H5 = BIOM / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
DB_FILES = {
    "GO_BP": BIOM / "results/biological_impact/reference_edge_sets/go_bp_gene_sets.json",
    "KEGG": BIOM / "results/biological_impact/reference_edge_sets/kegg_gene_sets.json",
    "Reactome": BIOM / "results/biological_impact/reference_edge_sets/reactome_gene_sets.json",
}
TRRUST_TSV = BIOM / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
ALPHA = 0.05
SEED = 4242017  # different from the first run on purpose
KIND_OFF = {"es": 11, "slot": 23}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ------------------------------------------------------------------ terms
def build_terms():
    raw = {}
    for name, p in DB_FILES.items():
        d = json.loads(p.read_text())
        for term, genes in d.items():
            raw[f"{name}:{term}"] = {g.upper() for g in genes}
    tr = pd.read_csv(TRRUST_TSV, sep="\t", header=None, usecols=[0, 1])
    tr.columns = ["tf", "target"]
    for tf, sub in tr.groupby(tr["tf"].str.upper()):
        raw[f"TRRUST_TF:TRRUST_TF:{tf}"] = set(sub["target"].str.upper())
    with h5py.File(REPLOGLE_H5, "r") as f:
        var = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["gene_name_index"][:]]
    varU = {v.upper() for v in var}
    terms = {}
    for k, s in raw.items():
        inv = s & varU
        if len(inv) >= 5:
            terms[k] = sorted(inv)
    return terms, var


def cmd_prep():
    OUT.mkdir(parents=True, exist_ok=True)
    terms, var = build_terms()
    (OUT / "terms_rebuilt.json").write_text(json.dumps(terms))
    inputs = {str(p): sha256_file(p) for p in list(DB_FILES.values()) + [TRRUST_TSV]}
    for li in range(12):
        for p in [OUTPUTS / f"phase0/layer_{li:02d}/gene_names.json",
                  OUTPUTS / f"phase2/layer_{li:02d}/feature_catalog.json",
                  OUTPUTS / f"phase2/layer_{li:02d}/significant_enrichments.csv",
                  OUTPUTS / f"phase2/layer_{li:02d}/annotation_summary.json"]:
            inputs[str(p)] = sha256_file(p)
    inputs[str(OUTPUTS / "phase6/causal_patching.csv")] = sha256_file(OUTPUTS / "phase6/causal_patching.csv")
    cfg = {
        "purpose": "independent verification of scripts/v2b_annotation_fdr.py (item N01)",
        "script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "platform": platform.platform(), "device": "cpu, no model forward pass",
        "alpha": ALPHA, "annotated_definition": "feature has >= 1 term with BH q < 0.05 (deployed convention)",
        "seed_rule": "null rep r of kind k at layer L: np.random.default_rng(4242017 + 100000*L + 1000*KIND_OFF[k] + r)",
        "kind_offsets": KIND_OFF, "n_terms": len(terms),
        "replogle_var_n": len(var),
        "replogle_var_list_sha256": hashlib.sha256("\n".join(var).encode()).hexdigest(),
        "terms_rebuilt_sha256": sha256_file(OUT / "terms_rebuilt.json"),
        "interval_methods": {
            "null_range": "min and max over the null catalogs (unit: annotated features per catalog)",
            "z": "(real count - null mean) / null SD (ddof=1) over the null catalogs; unit = null SD",
            "wilson": "Wilson score 95% interval, features as units, one trained SAE",
        },
        "commands_run_in_order": ["prep", "real --layers 5 (timing)", "real --layers 0-11",
                                  "null --layers 5 --reps 2 (timing)", "null --layers 0-3 --reps 20",
                                  "null --layers 4-7 --reps 20", "null --layers 8-11 --reps 20", "summary", "compare",
                                  "extra", "prep (last, to record the final script hash; terms rebuilt identically)"],
        "null_reps_per_kind_per_layer": 20,
        "extra_seed_rule": "uniform rep r at layer L: default_rng(99000 + 100*L + r), 3 reps at L0, L5, L11",
        "input_sha256": inputs,
    }
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print("terms", len(terms), "var", len(var))


# ------------------------------------------------------------------ core
def log_choose(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def tail_table(N, K, n):
    """P(X >= x) for x = 0..20 under Hypergeom(N, K, n), own log-gamma sum."""
    out = np.zeros(21)
    lo, hi = max(0, n - (N - K)), min(K, n)
    i = np.arange(lo, hi + 1)
    pmf = np.exp(log_choose(K, i) + log_choose(N - K, n - i) - log_choose(N, n))
    # sum from the top (small terms first)
    cs = np.cumsum(pmf[::-1])[::-1]
    for x in range(21):
        if x <= lo:
            out[x] = 1.0
        elif x > hi:
            out[x] = 0.0
        else:
            out[x] = min(1.0, cs[x - lo])
    return out


class Layer:
    def __init__(self, li, terms):
        self.li = li
        names = json.loads((OUTPUTS / f"phase0/layer_{li:02d}/gene_names.json").read_text())
        self.U = sorted({g.upper() for g in names})
        del names
        self.N = len(self.U)
        self.idx = {g: i for i, g in enumerate(self.U)}
        cat = json.loads((OUTPUTS / f"phase2/layer_{li:02d}/feature_catalog.json").read_text())
        self.fid = np.array([c["feature_id"] for c in cat])
        self.lists = [np.array(sorted({self.idx[g.upper()] for g in c["top20_genes"]}), dtype=np.int64) for c in cat]
        self.tnames = list(terms.keys())
        self.T = len(self.tnames)
        G = np.zeros((self.N, self.T), dtype=np.float32)
        for t, k in enumerate(self.tnames):
            for g in terms[k]:
                j = self.idx.get(g)
                if j is not None:
                    G[j, t] = 1.0
        self.G = G
        self.Kvar = np.array([len(terms[k]) for k in self.tnames], dtype=np.int64)
        self._tab = {}

    def tables(self, n):
        if n not in self._tab:
            uK, inv = np.unique(self.Kvar, return_inverse=True)
            tab = np.stack([tail_table(self.N, int(k), int(n)) for k in uK])
            self._tab[n] = (tab, inv)
        return self._tab[n]

    def tests(self, lists):
        F = len(lists)
        r = np.repeat(np.arange(F), [len(L) for L in lists])
        c = np.concatenate(lists)
        M = sp.csr_matrix((np.ones(len(c), dtype=np.float32), (r, c)), shape=(F, self.N))
        X = np.rint(M @ self.G).astype(np.int16)
        n = np.array([len(L) for L in lists])
        P = np.empty(X.shape)
        for nv in np.unique(n):
            rows = np.where(n == nv)[0]
            tab, inv = self.tables(int(nv))
            P[rows] = tab[inv[None, :], X[rows].astype(np.int64)]
        return X, n, P


def bh_q(p):
    m = p.size
    o = np.argsort(p, kind="mergesort")
    s = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(s[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out


def score(X, P):
    F, T = P.shape
    res = {}
    # deployed rule and bio-sae rule
    for name, keep in [("deployed", (X >= 2) & (P < 0.1)), ("biosae", X >= 2)]:
        fr, tc = np.nonzero(keep)
        q = bh_q(P[fr, tc])
        sig = q < ALPHA
        cnt = np.bincount(fr[sig], minlength=F)
        res[name] = {"kept": int(fr.size), "sig_rows": int(sig.sum()), "annot": int((cnt > 0).sum()),
                     "ge3": int((cnt >= 3).sum())}
        if name == "deployed":
            res["_dep_rows"] = (fr[sig], tc[sig], P[fr[sig], tc[sig]], q[sig])
            res["_dep_cnt"] = cnt
    # per-feature BH over all T terms
    Ps = np.sort(P, axis=1)
    Q = np.minimum.accumulate((Ps * (T / np.arange(1, T + 1)))[:, ::-1], axis=1)[:, ::-1]
    kpf = (Q < ALPHA).sum(axis=1)
    kpf_le = (Q <= ALPHA).sum(axis=1)
    res["perfeature"] = {"sig_rows": int(kpf.sum()), "annot": int((kpf > 0).sum()), "ge3": int((kpf >= 3).sum()),
                         "annot_le": int((kpf_le > 0).sum())}
    # cutoff p per row: the kpf-th smallest p
    cut = np.where(kpf > 0, Ps[np.arange(F), np.maximum(kpf - 1, 0)], -1.0)
    res["_pf_k"] = kpf; res["_pf_cut"] = cut
    del Ps, Q
    # global BH
    flat = np.sort(P, axis=None)
    M = flat.size
    qs = np.minimum.accumulate((flat * (M / np.arange(1, M + 1)))[::-1])[::-1]
    k = int((qs < ALPHA).sum())
    k_le = int((qs <= ALPHA).sum())
    cutg = flat[k - 1] if k > 0 else -1.0
    sigG = P <= cutg if k > 0 else np.zeros(P.shape, bool)
    cg = sigG.sum(axis=1)
    res["global"] = {"sig_rows": int(sigG.sum()), "annot": int((cg > 0).sum()), "ge3": int((cg >= 3).sum()),
                     "p_cut": float(cutg), "k_le": k_le}
    res["_g_cnt"] = cg; res["_g_sig"] = sigG
    return res


# ------------------------------------------------------------------ real
def cmd_real(layers):
    terms = json.loads((OUT / "terms_rebuilt.json").read_text())
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    gl_rows = []
    for li in layers:
        t0 = time.time()
        L = Layer(li, terms)
        X, n, P = L.tests(L.lists)
        res = score(X, P)
        F = len(L.lists)
        out = {"layer": li, "F": F, "N": L.N, "T": L.T, "n_tests": F * L.T,
               "n_lt20": int((n < 20).sum())}
        for k in ["deployed", "biosae", "perfeature", "global"]:
            out[k] = res[k]
        # (1) own p-values vs scipy, over every (K, n, x) cell actually used
        maxrel = 0.0
        for nv, (tab, inv) in L._tab.items():
            uK = np.unique(L.Kvar)
            ref = np.stack([hypergeom.sf(np.arange(21) - 1, L.N, int(k), int(nv)) for k in uK])
            ref[:, nv + 1:] = 0.0
            m = ref > 1e-300
            maxrel = max(maxrel, float(np.max(np.abs(tab[m] - ref[m]) / ref[m])))
        out["own_p_vs_scipy_max_rel_diff"] = maxrel
        # (2) reproduction of deployed files
        dep = json.loads((OUTPUTS / f"phase2/layer_{li:02d}/annotation_summary.json").read_text())
        csv = pd.read_csv(OUTPUTS / f"phase2/layer_{li:02d}/significant_enrichments.csv")
        fr, tc, pk, qk = res["_dep_rows"]
        mine = pd.DataFrame({"feature_id": L.fid[fr], "term": [L.tnames[t] for t in tc], "p2": pk, "q2": qk})
        mg = csv.merge(mine, on=["feature_id", "term"], how="outer", indicator=True)
        b = mg[mg["_merge"] == "both"]
        out["repro"] = {"dep_annot": dep["n_annotated"], "dep_rows": dep["total_enrichments"],
                        "dep_alive": dep["n_alive"], "only_dep": int((mg["_merge"] == "left_only").sum()),
                        "only_mine": int((mg["_merge"] == "right_only").sum()),
                        "max_rel_p": float((np.abs(b["p_raw"] - b["p2"]) / b["p_raw"]).max()),
                        "max_abs_q": float((b["q_bh"] - b["q2"]).abs().max()),
                        "csv_N": sorted(csv["N"].unique().tolist()),
                        "csv_max_p": float(csv["p_raw"].max()),
                        "csv_frac_overlap2": float((csv["overlap"] == 2).mean())}
        # (3) per-feature BH rows: overlap and list size of surviving rows
        kpf, cut = res["_pf_k"], res["_pf_cut"]
        rr, cc = np.nonzero(P <= cut[:, None])
        ov = X[rr, cc]; nn = n[rr]
        out["perfeature"]["rows_check"] = int(rr.size)
        out["perfeature"]["frac_rows_overlap2"] = float((ov == 2).mean()) if rr.size else None
        out["perfeature"]["rows_overlap2_with_n20"] = int(((ov == 2) & (nn == 20)).sum())
        out["perfeature"]["rows_overlap2_with_n_lt20"] = int(((ov == 2) & (nn < 20)).sum())
        # smallest attainable p for overlap 2, n = 20, smallest term
        out["min_p_overlap2_n20"] = float(L.tables(20)[0][0, 2])
        # (4) Wilson interval for the per-feature rate
        k_ = out["perfeature"]["annot"]; z = 1.959964; ph = k_ / F
        den = 1 + z * z / F; c = (ph + z * z / (2 * F)) / den
        h = z * np.sqrt(ph * (1 - ph) / F + z * z / (4 * F * F)) / den
        out["perfeature"]["wilson95"] = [c - h, c + h]
        # (5) features passing global BH: their terms and genes
        cg, sigG = res["_g_cnt"], res["_g_sig"]
        for f in np.where(cg > 0)[0]:
            ts = np.where(sigG[f])[0]
            ts = ts[np.argsort(P[f, ts])]
            gl_rows.append({"layer": li, "feature_id": int(L.fid[f]), "n_terms_global": int(cg[f]),
                            "n_terms_perfeature": int(kpf[f]), "n_terms_deployed": int(res["_dep_cnt"][f]),
                            "min_p": float(P[f, ts[0]]), "top_terms": " | ".join(L.tnames[t] for t in ts[:3]),
                            "top20_genes": ",".join(L.U[j] for j in L.lists[f])})
        # (6) per-feature flags for L5 phase 6 check
        if li == 5:
            p6 = pd.read_csv(OUTPUTS / "phase6/causal_patching.csv")["feature_id"].to_numpy()
            pos = {f: i for i, f in enumerate(L.fid)}
            ix = np.array([pos[f] for f in p6])
            depc = res["_dep_cnt"]
            order = sorted(range(F), key=lambda i: (-depc[i], i))
            top50 = [L.fid[i] for i in order if depc[i] >= 3][:50]
            out["phase6"] = {"n": int(len(p6)), "in_top50_of_rebuilt_deployed": int(len(set(p6) & set(top50))),
                             "dep_count_range": [int(depc[ix].min()), int(depc[ix].max())],
                             "boundary_count_50th": int(depc[order[49]]), "count_51st": int(depc[order[50]]),
                             "perfeature_annot": int((kpf[ix] > 0).sum()), "perfeature_ge3": int((kpf[ix] >= 3).sum()),
                             "global_annot": int((cg[ix] > 0).sum()),
                             "global_annot_ids": [int(x) for x in p6[cg[ix] > 0]]}
        out["seconds"] = round(time.time() - t0, 1)
        (OUT / f"real/layer_{li:02d}.json").write_text(json.dumps(out, indent=1))
        print(f"L{li} F={F} dep={out['deployed']['annot']}/{dep['n_annotated']} rows {out['deployed']['sig_rows']}/"
              f"{dep['total_enrichments']} only_dep={out['repro']['only_dep']} only_mine={out['repro']['only_mine']} "
              f"bio={out['biosae']['annot']} pf={out['perfeature']['annot']} (le {out['perfeature']['annot_le']}) "
              f"glob={out['global']['annot']} prel={maxrel:.1e} {out['seconds']}s", flush=True)
    if gl_rows:
        p = OUT / "real/global_bh_features.csv"
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        new = pd.DataFrame(gl_rows)
        if len(old):
            old = old[~old["layer"].isin(layers)]
        pd.concat([old, new]).sort_values(["layer", "min_p"]).to_csv(p, index=False)


# ------------------------------------------------------------------ nulls
def null_es(L: Layer, rng):
    cnt = np.zeros(L.N)
    for x in L.lists:
        cnt[x] += 1
    sup = np.where(cnt > 0)[0]
    w = cnt[sup]
    sizes = [len(x) for x in L.lists]
    out = []
    for a in range(0, len(sizes), 400):
        sz = sizes[a:a + 400]
        keys = rng.exponential(size=(len(sz), sup.size)) / w[None, :]
        part = np.argpartition(keys, 20, axis=1)[:, :20]
        for i, s in enumerate(sz):
            sel = part[i][np.argsort(keys[i, part[i]])][:s]
            out.append(np.sort(sup[sel]))
    return out


def null_slot(L: Layer, rng):
    sizes = np.array([len(x) for x in L.lists])
    slots = rng.permutation(np.concatenate(L.lists))
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    lists = [list(slots[bounds[i]:bounds[i + 1]]) for i in range(len(sizes))]
    sets = [set(x) for x in lists]
    F = len(lists)
    for it in range(200):
        bad = [i for i in range(F) if len(sets[i]) < len(lists[i])]
        if not bad:
            break
        for i in bad:
            seen = set()
            for pos, g in enumerate(lists[i]):
                if g not in seen:
                    seen.add(g); continue
                # duplicate at pos: swap with a random slot elsewhere that keeps both lists duplicate-free
                for _ in range(1000):
                    j = int(rng.integers(F))
                    if j == i:
                        continue
                    pj = int(rng.integers(len(lists[j])))
                    h = lists[j][pj]
                    if h in seen or h in sets[i] or g in sets[j]:
                        continue
                    lists[j][pj] = g; lists[i][pos] = h
                    sets[j] = set(lists[j]); seen.add(h)
                    break
            sets[i] = set(lists[i])
    assert all(len(sets[i]) == len(lists[i]) for i in range(F)), "duplicate repair failed"
    return [np.array(sorted(x), dtype=np.int64) for x in lists]


def cmd_null(layers, reps):
    terms = json.loads((OUT / "terms_rebuilt.json").read_text())
    (OUT / "null").mkdir(parents=True, exist_ok=True)
    for li in layers:
        L = Layer(li, terms)
        real_cnt = np.zeros(L.N)
        for x in L.lists:
            real_cnt[x] += 1
        path = OUT / f"null/layer_{li:02d}.json"
        done = json.loads(path.read_text()) if path.exists() else []
        have = {(d["kind"], d["rep"]) for d in done}
        for kind in ["es", "slot"]:
            for r in range(reps):
                if (kind, r) in have:
                    continue
                t0 = time.time()
                seed = SEED + 100000 * li + 1000 * KIND_OFF[kind] + r
                rng = np.random.default_rng(seed)
                fake = null_es(L, rng) if kind == "es" else null_slot(L, rng)
                X, n, P = L.tests(fake)
                res = score(X, P)
                c = np.zeros(L.N)
                for x in fake:
                    c[x] += 1
                d = {"layer": li, "kind": kind, "rep": r, "seed": seed,
                     "count_corr_real": float(np.corrcoef(c, real_cnt)[0, 1]),
                     "count_max_abs_diff": float(np.abs(c - real_cnt).max()),
                     "sizes_equal": bool(np.array_equal(n, [len(x) for x in L.lists]))}
                for k in ["deployed", "biosae", "perfeature", "global"]:
                    d[k] = res[k]
                d["seconds"] = round(time.time() - t0, 1)
                done.append(d)
                path.write_text(json.dumps(done, indent=1))
                print(f"L{li} {kind} r{r}: dep={d['deployed']['annot']} bio={d['biosae']['annot']} "
                      f"pf={d['perfeature']['annot']} glob={d['global']['annot']} corr={d['count_corr_real']:.3f} "
                      f"{d['seconds']}s", flush=True)


# ------------------------------------------------------------------ summary
def cmd_summary():
    rows = []
    for li in range(12):
        r = json.loads((OUT / f"real/layer_{li:02d}.json").read_text())
        nl = json.loads((OUT / f"null/layer_{li:02d}.json").read_text())
        F = r["F"]
        row = {"layer": li, "F": F,
               "repro_exact": r["deployed"]["annot"] == r["repro"]["dep_annot"] and r["deployed"]["sig_rows"] == r["repro"]["dep_rows"]
               and r["repro"]["only_dep"] == 0 and r["repro"]["only_mine"] == 0,
               "own_p_max_rel": r["own_p_vs_scipy_max_rel_diff"], "kept_deployed": r["deployed"]["kept"],
               "csv_max_p": r["repro"]["csv_max_p"], "csv_frac_ov2": r["repro"]["csv_frac_overlap2"],
               "pf_frac_rows_ov2": r["perfeature"]["frac_rows_overlap2"],
               "pf_ov2_n20": r["perfeature"]["rows_overlap2_with_n20"],
               "pf_ov2_nlt20": r["perfeature"]["rows_overlap2_with_n_lt20"],
               "pf_wilson_lo": r["perfeature"]["wilson95"][0], "pf_wilson_hi": r["perfeature"]["wilson95"][1]}
        for rule in ["deployed", "biosae", "perfeature", "global"]:
            row[f"{rule}_real"] = r[rule]["annot"]
            row[f"{rule}_real_rate"] = r[rule]["annot"] / F
            for kind in ["es", "slot"]:
                v = np.array([d[rule]["annot"] for d in nl if d["kind"] == kind], dtype=float)
                row[f"{rule}_{kind}_reps"] = int(v.size)
                row[f"{rule}_{kind}_mean"] = float(v.mean())
                row[f"{rule}_{kind}_min"] = float(v.min())
                row[f"{rule}_{kind}_max"] = float(v.max())
                sd = float(v.std(ddof=1))
                row[f"{rule}_{kind}_z"] = (r[rule]["annot"] - v.mean()) / sd if sd > 0 else None
                row[f"{rule}_{kind}_n_null_ge_real"] = int((v >= r[rule]["annot"]).sum())
                if rule == "perfeature":
                    row[f"pf_ge3_{kind}_mean"] = float(np.mean([d[rule]["ge3"] for d in nl if d["kind"] == kind]))
        row["pf_ge3_real"] = r["perfeature"]["ge3"]
        for kind in ["es", "slot"]:
            row[f"count_corr_{kind}_min"] = float(min(d["count_corr_real"] for d in nl if d["kind"] == kind))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    extra = {}
    for rule in ["deployed", "biosae", "perfeature"]:
        for kind in ["es", "slot"]:
            extra[f"pearson_layers_{rule}_real_vs_{kind}"] = float(np.corrcoef(df[f"{rule}_real_rate"],
                                                                               df[f"{rule}_{kind}_mean"] / df["F"])[0, 1])
    extra["global_total_real"] = int(df["global_real"].sum())
    extra["global_total_null_mean_es"] = float(df["global_es_mean"].sum())
    extra["global_total_null_mean_slot"] = float(df["global_slot_mean"].sum())
    extra["alive_total"] = int(df["F"].sum())
    extra["phase6"] = json.loads((OUT / "real/layer_05.json").read_text())["phase6"]
    (OUT / "summary_extra.json").write_text(json.dumps(extra, indent=1))
    pd.set_option("display.width", 250)
    cols = ["layer", "F", "repro_exact", "deployed_real", "deployed_es_min", "deployed_es_max", "deployed_slot_min",
            "deployed_slot_max", "biosae_real", "biosae_es_mean", "biosae_slot_mean", "perfeature_real",
            "perfeature_es_mean", "perfeature_es_max", "perfeature_slot_mean", "perfeature_slot_max", "perfeature_es_z",
            "perfeature_slot_z", "global_real", "global_es_mean", "global_slot_mean"]
    print(df[cols].to_string(index=False))
    print(json.dumps(extra, indent=1))


# ------------------------------------------------------------------ compare with first run
def cmd_compare():
    mine = pd.read_csv(OUT / "summary.csv")
    first = pd.read_csv(FIRST / "summary.csv")
    m = mine.merge(first, on="layer")
    out = pd.DataFrame({
        "layer": m["layer"],
        "deployed_mine": m["deployed_real"], "deployed_first": m["A_n"],
        "biosae_mine": m["biosae_real"], "biosae_first": m["A2_n"],
        "pf_mine": m["perfeature_real"], "pf_first": m["B_n"],
        "global_mine": m["global_real"], "global_first": m["C_n"],
        "pf_ge3_mine": m["pf_ge3_real"], "pf_ge3_first": m["B_ge3"],
        "pf_null_es_mean": m["perfeature_es_mean"], "pf_null_freq_first": m["null_freq_B_n_mean"],
        "pf_null_slot_mean": m["perfeature_slot_mean"],
        "pf_null_curveball_first": m["null_curveball_B_mean"] * m["F"],
        "dep_null_es_mean_rate": m["deployed_es_mean"] / m["F"], "dep_null_freq_first_rate": m["null_freq_A_mean"],
    })
    out.to_csv(OUT / "compare_with_first_run.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


def cmd_extra():
    """Two side checks: (a) term size K counted inside the universe; (b) uniform random lists."""
    terms = json.loads((OUT / "terms_rebuilt.json").read_text())
    out = {}
    for li in range(12):
        L = Layer(li, terms)
        Kvar = L.Kvar.copy()
        L.Kvar = L.G.sum(axis=0).astype(np.int64); L._tab = {}
        X, n, P = L.tests(L.lists)
        r = score(X, P)
        row = {"perfeature_annot_K_universe": r["perfeature"]["annot"], "global_annot_K_universe": r["global"]["annot"],
               "perfeature_rows_K_universe": r["perfeature"]["sig_rows"]}
        L.Kvar = Kvar; L._tab = {}
        if li in (0, 5, 11):
            u = []
            for rep in range(3):
                rng = np.random.default_rng(99000 + 100 * li + rep)
                fake = [np.sort(rng.choice(L.N, size=len(x), replace=False)) for x in L.lists]
                X, n, P = L.tests(fake)
                s_ = score(X, P)
                u.append({"deployed_rate": s_["deployed"]["annot"] / len(fake), "biosae_rate": s_["biosae"]["annot"] / len(fake),
                          "perfeature_rate": s_["perfeature"]["annot"] / len(fake), "global_annot": s_["global"]["annot"]})
            row["uniform_null"] = u
        out[li] = row
        print(li, row, flush=True)
    (OUT / "extra_checks.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "real", "null", "summary", "compare", "extra"])
    ap.add_argument("--layers", default="0-11")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep()
    elif a.cmd == "real":
        cmd_real(parse_layers(a.layers))
    elif a.cmd == "null":
        cmd_null(parse_layers(a.layers), a.reps)
    elif a.cmd == "summary":
        cmd_summary()
    elif a.cmd == "extra":
        cmd_extra()
    else:
        cmd_compare()


if __name__ == "__main__":
    main()
