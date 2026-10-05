"""V2b item N01: SAE feature annotation rates with a correct multiple-testing family.

The deployed Phase 2 code (full_12layer_pipeline.py:243-252, same as phases2_to_8.py:176-197)
kept a feature x term test only if overlap >= 2 AND raw p < 0.1, then ran one pooled BH over the
kept rows. The spec (pipelines/sparse-autoencoders/01-sae-atlas.md, Phase 2 step 3) asks for BH
across all terms tested for each feature.

This script rebuilds every feature x term hypergeometric test from saved files only
(feature_catalog.json top-20 lists, phase0 gene_names.json universe, the same GO/KEGG/Reactome/TRRUST
files, the Replogle var symbols). No model forward pass. CPU only.

Rules computed (alpha = 0.05 everywhere):
  A   deployed rule: overlap >= 2 and p < 0.1, pooled BH over kept rows (must reproduce the deployed numbers)
  A2  diagnostic: overlap >= 2 only, pooled BH over kept rows (shows what the p < 0.1 filter adds)
  B   spec rule: BH per feature over ALL T terms (x = 0 tests have p = 1 and still count in m)
  C   global BH over ALL feature x term tests of the layer
  Bu, Cu  sensitivity: same as B, C but with the term size K counted inside the test universe
          (the deployed K counts term genes in the full Replogle var list, 6,546 genes, while N is the
          6,325 names seen at token positions; that makes deployed p-values slightly conservative)

Random-feature nulls (same code path, same rules):
  freq       each fake top-20 list = n_f distinct genes drawn without replacement with probability
             proportional to the gene's number of appearances across the layer's real top-20 lists
  curveball  exact-margin null: the real lists are shuffled by curveball trades (Strona et al. 2014),
             which keeps every list's size and every gene's total count exactly
  uniform    n_f distinct genes drawn uniformly from the universe (context only)

Usage:
  python v2b_annotation_fdr.py prep
  python v2b_annotation_fdr.py real   --layers 0-11
  python v2b_annotation_fdr.py null   --layers 0-3 --kinds freq,curveball,uniform --reps 20
  python v2b_annotation_fdr.py summary
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
import statsmodels
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/sae-atlas-217M"
OUTPUTS = RUN / "outputs"
OUT = OUTPUTS / "v2b_annotation_fdr"
BIOM_ROOT = Path("<DATA_ROOT>")
REPLOGLE_H5 = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
GO_BP_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"
KEGG_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/kegg_gene_sets.json"
REACTOME_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/reactome_gene_sets.json"
TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"

ALPHA = 0.05
SEED = 20261001
KIND_OFFSET = {"freq": 100, "curveball": 200, "uniform": 300}
CURVEBALL_ROUNDS = 30
SPECIAL = "<SPECIAL>"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ---------------------------------------------------------------- prep
def cmd_prep():
    OUT.mkdir(parents=True, exist_ok=True)
    # Same construction as full_12layer_pipeline.py:79-94
    with open(GO_BP_JSON) as f:
        go_bp = json.load(f)
    with open(KEGG_JSON) as f:
        kegg = json.load(f)
    with open(REACTOME_JSON) as f:
        reactome = json.load(f)
    trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    trrust["tf"] = trrust["tf"].str.upper()
    trrust["target"] = trrust["target"].str.upper()
    trrust_tf_sets = {}
    for tf, grp in trrust.groupby("tf"):
        trrust_tf_sets[f"TRRUST_TF:{tf}"] = list(grp["target"].unique())
    all_databases = {}
    for name, db in [("GO_BP", go_bp), ("KEGG", kegg), ("Reactome", reactome), ("TRRUST_TF", trrust_tf_sets)]:
        for term, genes in db.items():
            all_databases[f"{name}:{term}"] = set(g.upper() for g in genes)
    with h5py.File(REPLOGLE_H5, "r") as f:
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    var_upper = set(g.upper() for g in var_symbols)
    terms = []
    for k, v in all_databases.items():
        inv = v & var_upper
        if len(inv) >= 5:
            terms.append({"term": k, "genes": sorted(inv), "K_var": len(inv)})
    print(f"terms after filtering: {len(terms)} (deployed log says 2377)")
    var_list_sha = hashlib.sha256("\n".join(var_symbols).encode()).hexdigest()
    (OUT / "terms.json").write_text(json.dumps(terms))

    inputs = {}
    for p in [GO_BP_JSON, KEGG_JSON, REACTOME_JSON, TRRUST_TSV]:
        inputs[str(p)] = sha256(p)
    for li in range(12):
        for p in [OUTPUTS / f"phase0/layer_{li:02d}/gene_names.json",
                  OUTPUTS / f"phase2/layer_{li:02d}/feature_catalog.json",
                  OUTPUTS / f"phase2/layer_{li:02d}/significant_enrichments.csv",
                  OUTPUTS / f"phase2/layer_{li:02d}/annotation_summary.json"]:
            inputs[str(p)] = sha256(p)
    st = REPLOGLE_H5.stat()
    cfg = {
        "item": "N01 SAE feature annotation rates: correct BH family + random-feature null",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "statsmodels": statsmodels.__version__, "platform": platform.platform(),
        "device": "cpu (no model forward pass)",
        "alpha": ALPHA,
        "seed_base": SEED,
        "seed_rule": "null rep r of kind k at layer L uses default_rng(seed_base + 1000*L + KIND_OFFSET[k] + r)",
        "kind_offset": KIND_OFFSET,
        "curveball_rounds": CURVEBALL_ROUNDS,
        "n_terms": len(terms),
        "replogle_var": {"path": str(REPLOGLE_H5), "n_genes": len(var_symbols),
                         "sha256_of_newline_joined_var_gene_name_index": var_list_sha,
                         "file_size_bytes": st.st_size, "file_mtime": st.st_mtime,
                         "note": "the 30 GB h5ad was not hashed whole; the extracted var symbol list was"},
        "terms_json_sha256": sha256(OUT / "terms.json"),
        "commands_run_in_order": [
            "prep", "real --layers 0-11",
            "null --layers 5 --reps 2 (timing), then null --reps 20 for layers 0-2, 3-4, 5, 6-7, 8, 9, 10, 11 "
            "(existing reps are skipped, so every layer/kind has reps 0-19)",
            "summary", "prep (re-run last to record the final script hash; terms.json is rebuilt identically)"],
        "null_reps_per_kind_per_layer": 20,
        "interval_methods": {
            "null_range": "2.5th-97.5th percentile of the annotation rate over 20 null catalogs (unit: fraction of alive features)",
            "wilson": "Wilson score 95% interval for the real per-feature-BH rate, features as units (one trained SAE; does not cover SAE retraining)",
            "z": "(real rate - null mean) / null SD over 20 catalogs",
            "emp_p": "(1 + #null catalogs with rate >= real) / (20 + 1); smallest possible value 0.048"},
        "input_sha256": inputs,
    }
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print("wrote", OUT / "run_config.json")


# ---------------------------------------------------------------- core
class LayerData:
    def __init__(self, li: int, terms: list[dict]):
        self.li = li
        with open(OUTPUTS / f"phase0/layer_{li:02d}/gene_names.json") as f:
            names = json.load(f)
        self.universe = sorted(set(g.upper() for g in names))
        del names
        self.N = len(self.universe)
        self.g2i = {g: i for i, g in enumerate(self.universe)}
        with open(OUTPUTS / f"phase2/layer_{li:02d}/feature_catalog.json") as f:
            self.catalog = json.load(f)
        self.feature_ids = np.array([c["feature_id"] for c in self.catalog])
        self.lists = [sorted(set(c["top20_genes"])) for c in self.catalog]
        # term x universe membership
        T = len(terms)
        rows, cols = [], []
        for t, d in enumerate(terms):
            for g in d["genes"]:
                j = self.g2i.get(g)
                if j is not None:
                    rows.append(j)
                    cols.append(t)
        self.B = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.N, T))
        self.K_var = np.array([d["K_var"] for d in terms], dtype=np.int64)
        self.K_u = np.asarray(self.B.sum(axis=0)).ravel().astype(np.int64)
        self.term_names = [d["term"] for d in terms]
        self.T = T
        self._lut_cache = {}

    def lut(self, K: np.ndarray, kname: str, n_values) -> dict:
        """p-value table: tab[n][k_index, x] = hypergeom.sf(x-1, N, K, n)."""
        uK, kidx = np.unique(K, return_inverse=True)
        out = {"kidx": kidx}
        for n in sorted(set(n_values)):
            key = (kname, n)
            if key not in self._lut_cache:
                xs = np.arange(0, 21)
                tab = np.empty((len(uK), 21), dtype=np.float64)
                for a, k in enumerate(uK):
                    tab[a] = hypergeom.sf(xs - 1, self.N, int(k), int(n))
                tab[:, n + 1:] = 0.0  # x > n is impossible
                self._lut_cache[key] = tab
            out[n] = self._lut_cache[key]
        return out

    def overlap(self, lists: list[list[str]]):
        rows, cols = [], []
        for f, L in enumerate(lists):
            for g in L:
                rows.append(f)
                cols.append(self.g2i[g])
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lists), self.N))
        AB = A @ self.B
        X = np.rint(AB.toarray() if sp.issparse(AB) else np.asarray(AB)).astype(np.int16)
        n = np.array([len(L) for L in lists], dtype=np.int64)
        return X, n

    def pvals(self, X: np.ndarray, n: np.ndarray, K: np.ndarray, kname: str) -> np.ndarray:
        lt = self.lut(K, kname, n.tolist())
        kidx = lt["kidx"]
        P = np.empty(X.shape, dtype=np.float64)
        for nv in np.unique(n):
            rows = np.where(n == nv)[0]
            tab = lt[int(nv)]
            P[rows] = tab[kidx[None, :], X[rows].astype(np.int64)]
        return P


def bh_perfeature(P: np.ndarray, alpha=ALPHA):
    """BH within each row over all columns. Returns (n_rejected per row, cutoff per row)."""
    F, T = P.shape
    Ps = np.sort(P, axis=1)
    thr = (np.arange(1, T + 1) / T) * alpha  # same form as statsmodels fdr_bh
    ok = Ps <= thr[None, :]
    anyok = ok.any(axis=1)
    last = T - 1 - np.argmax(ok[:, ::-1], axis=1)
    k = np.where(anyok, last + 1, 0)
    cutoff = np.where(anyok, Ps[np.arange(F), last], -1.0)
    return k, cutoff


def bh_global(P: np.ndarray, alpha=ALPHA):
    flat = np.sort(P, axis=None)
    M = flat.size
    thr = (np.arange(1, M + 1) / M) * alpha
    ok = flat <= thr
    if not ok.any():
        return 0, -1.0
    last = M - 1 - np.argmax(ok[::-1])
    return int(last + 1), float(flat[last])


def rules(ld: LayerData, lists, want_rows=False, sensitivity=True):
    X, n = ld.overlap(lists)
    F = len(lists)
    res = {"n_features": F, "n_terms": ld.T, "n_tests": int(F * ld.T)}
    P = ld.pvals(X, n, ld.K_var, "var")
    # A: deployed
    keep = (X >= 2) & (P < 0.1)
    fr, tc = np.nonzero(keep)
    pk = P[fr, tc]
    if pk.size:
        _, q, _, _ = multipletests(pk, method="fdr_bh")
        sigA = q < ALPHA
    else:
        q = np.array([]); sigA = np.array([], dtype=bool)
    cntA = np.bincount(fr[sigA], minlength=F)
    res["A_deployed"] = {"n_kept_tests": int(pk.size), "n_sig_rows": int(sigA.sum()),
                         "n_annotated": int((cntA > 0).sum()), "n_ge3": int((cntA >= 3).sum()),
                         "max_raw_p_sig": float(pk[sigA].max()) if sigA.any() else None,
                         "frac_sig_rows_overlap2": float((X[fr[sigA], tc[sigA]] == 2).mean()) if sigA.any() else None}
    # A2: overlap >= 2 only
    keep2 = X >= 2
    fr2, tc2 = np.nonzero(keep2)
    pk2 = P[fr2, tc2]
    if pk2.size:
        _, q2, _, _ = multipletests(pk2, method="fdr_bh")
        sig2 = q2 < ALPHA
        cnt2 = np.bincount(fr2[sig2], minlength=F)
        res["A2_overlap2_only"] = {"n_kept_tests": int(pk2.size), "n_sig_rows": int(sig2.sum()),
                                   "n_annotated": int((cnt2 > 0).sum()), "n_ge3": int((cnt2 >= 3).sum())}
    # B: per-feature BH over all terms
    kB, cutB = bh_perfeature(P)
    res["B_perfeature_bh"] = {"n_sig_rows": int(kB.sum()), "n_annotated": int((kB > 0).sum()),
                              "n_ge3": int((kB >= 3).sum())}
    # C: global BH
    kC, cutC = bh_global(P)
    sigC = P <= cutC if kC > 0 else np.zeros_like(P, dtype=bool)
    cntC = sigC.sum(axis=1)
    res["C_global_bh"] = {"n_sig_rows": int(kC), "n_annotated": int((cntC > 0).sum()),
                          "n_ge3": int((cntC >= 3).sum()), "p_cutoff": cutC}
    for key in ["A_deployed", "A2_overlap2_only", "B_perfeature_bh", "C_global_bh"]:
        if key in res:
            res[key]["rate"] = res[key]["n_annotated"] / F
    extra = {"X": X, "n": n, "P": P, "kB": kB, "cutB": cutB, "sigC": sigC,
             "A_rows": (fr[sigA], tc[sigA], pk[sigA], q[sigA])}
    if sensitivity:
        Pu = ld.pvals(X, n, ld.K_u, "u")
        kBu, _ = bh_perfeature(Pu)
        kCu, cutCu = bh_global(Pu)
        cntCu = (Pu <= cutCu).sum(axis=1) if kCu > 0 else np.zeros(F, dtype=int)
        res["Bu_perfeature_bh_Kuniverse"] = {"n_annotated": int((kBu > 0).sum()), "rate": float((kBu > 0).mean()),
                                             "n_sig_rows": int(kBu.sum()), "n_ge3": int((kBu >= 3).sum())}
        res["Cu_global_bh_Kuniverse"] = {"n_annotated": int((cntCu > 0).sum()), "rate": float((cntCu > 0).mean()),
                                         "n_sig_rows": int(kCu), "n_ge3": int((cntCu >= 3).sum())}
        del Pu
    return res, (extra if want_rows else None)


# ---------------------------------------------------------------- real
def cmd_real(layers):
    terms = json.loads((OUT / "terms.json").read_text())
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    for li in layers:
        t0 = time.time()
        ld = LayerData(li, terms)
        res, ex = rules(ld, ld.lists, want_rows=True)
        res["layer"] = li
        res["N_universe"] = ld.N
        # reproduction check against deployed files
        dep = json.loads((OUTPUTS / f"phase2/layer_{li:02d}/annotation_summary.json").read_text())
        csv = pd.read_csv(OUTPUTS / f"phase2/layer_{li:02d}/significant_enrichments.csv")
        fr, tc, pk, qk = ex["A_rows"]
        mine = pd.DataFrame({"feature_id": ld.feature_ids[fr], "term": [ld.term_names[t] for t in tc],
                             "p_mine": pk, "q_mine": qk})
        m = csv.merge(mine, on=["feature_id", "term"], how="outer", indicator=True)
        both = m[m["_merge"] == "both"]
        res["repro_check"] = {
            "deployed_n_alive": dep["n_alive"], "deployed_n_annotated": dep["n_annotated"],
            "deployed_total_enrichments": dep["total_enrichments"], "deployed_rate": dep["annotation_rate"],
            "rows_only_in_deployed_csv": int((m["_merge"] == "left_only").sum()),
            "rows_only_in_rebuild": int((m["_merge"] == "right_only").sum()),
            "max_abs_diff_p": float((both["p_raw"] - both["p_mine"]).abs().max()),
            "max_abs_diff_q": float((both["q_bh"] - both["q_mine"]).abs().max()),
            "deployed_csv_max_raw_p": float(csv["p_raw"].max()),
            "deployed_csv_frac_overlap2": float((csv["overlap"] == 2).mean()),
            "N_in_csv": sorted(csv["N"].unique().tolist()),
            "exact_match": bool(dep["n_annotated"] == res["A_deployed"]["n_annotated"]
                                and dep["total_enrichments"] == res["A_deployed"]["n_sig_rows"]
                                and dep["n_alive"] == res["n_features"]
                                and (m["_merge"] != "both").sum() == 0),
        }
        # self-test of the vectorised BH against statsmodels on 200 features
        P = ex["P"]; kB = ex["kB"]
        rng = np.random.default_rng(SEED + li)
        test_rows = np.unique(np.concatenate([np.where(kB > 0)[0][:100],
                                              rng.choice(P.shape[0], size=min(100, P.shape[0]), replace=False)]))
        mism = 0
        for r in test_rows:
            rej, _, _, _ = multipletests(P[r], alpha=ALPHA, method="fdr_bh")
            mism += int(rej.sum() != kB[r])
        res["selftest_perfeature_bh_vs_statsmodels"] = {"rows_checked": int(len(test_rows)), "mismatches": mism}
        # per-feature flags (for downstream re-selection)
        X = ex["X"]; cutB = ex["cutB"]; sigC = ex["sigC"]
        cntA = np.bincount(fr, minlength=len(ld.lists))
        minp = P.min(axis=1)
        best = P.argmin(axis=1)
        flags = pd.DataFrame({"feature_id": ld.feature_ids, "n_top_genes": ex["n"],
                              "n_sig_deployed": cntA, "n_sig_perfeature_bh": kB,
                              "n_sig_global_bh": sigC.sum(axis=1), "min_raw_p": minp,
                              "best_term": [ld.term_names[t] for t in best],
                              "best_term_overlap": X[np.arange(len(best)), best]})
        flags.to_csv(OUT / f"real/layer_{li:02d}_feature_flags.csv", index=False)
        # significant rows under B, with per-feature BH q
        rows = []
        for r in np.where(kB > 0)[0]:
            _, qv, _, _ = multipletests(P[r], method="fdr_bh")
            for t in np.where(P[r] <= cutB[r])[0]:
                rows.append({"feature_id": int(ld.feature_ids[r]), "term": ld.term_names[t], "overlap": int(X[r, t]),
                             "K": int(ld.K_var[t]), "n": int(ex["n"][r]), "N": ld.N, "p_raw": float(P[r, t]),
                             "q_perfeature_bh": float(qv[t]), "sig_global_bh": bool(sigC[r, t])})
        dfB = pd.DataFrame(rows)
        dfB.to_csv(OUT / f"real/layer_{li:02d}_perfeature_bh_enrichments.csv", index=False)
        if len(dfB):
            res["B_perfeature_bh"]["overlap_min"] = int(dfB["overlap"].min())
            res["B_perfeature_bh"]["overlap_median"] = float(dfB["overlap"].median())
            res["B_perfeature_bh"]["by_database"] = dfB["term"].str.split(":").str[0].value_counts().to_dict()
        # smallest p for a 2-gene overlap at this layer (smallest K, n = 20)
        p2 = hypergeom.sf(1, ld.N, int(ld.K_var.min()), 20)
        res["smallest_possible_p_overlap2"] = float(p2)
        res["perfeature_bh_rank1_threshold"] = ALPHA / ld.T
        res["seconds"] = round(time.time() - t0, 1)
        (OUT / f"real/layer_{li:02d}.json").write_text(json.dumps(res, indent=2))
        print(f"L{li}: F={res['n_features']} deployed={res['A_deployed']['n_annotated']} "
              f"(file {dep['n_annotated']}, exact={res['repro_check']['exact_match']}) "
              f"A2={res['A2_overlap2_only']['n_annotated']} B={res['B_perfeature_bh']['n_annotated']} "
              f"C={res['C_global_bh']['n_annotated']} Bu={res['Bu_perfeature_bh_Kuniverse']['n_annotated']} "
              f"Cu={res['Cu_global_bh_Kuniverse']['n_annotated']} selftest_mism={mism} ({res['seconds']}s)")


# ---------------------------------------------------------------- nulls
def curveball(lists, rng, rounds):
    L = [list(x) for x in lists]
    F = len(L)
    for _ in range(rounds):
        perm = rng.permutation(F)
        for a, b in zip(perm[0::2], perm[1::2]):
            sa, sb = set(L[a]), set(L[b])
            shared = sa & sb
            pool = sorted((sa - shared) | (sb - shared))
            if not pool:
                continue
            na = len(sa) - len(shared)
            pool = list(rng.permutation(pool))
            L[a] = sorted(shared) + pool[:na]
            L[b] = sorted(shared) + pool[na:]
    return L


def make_null(ld: LayerData, kind: str, rng):
    sizes = [len(x) for x in ld.lists]
    if kind == "curveball":
        return curveball(ld.lists, rng, CURVEBALL_ROUNDS)
    if kind == "freq":
        cnt = np.zeros(ld.N)
        for L in ld.lists:
            for g in L:
                cnt[ld.g2i[g]] += 1
        w = cnt / cnt.sum()
        return [[ld.universe[j] for j in rng.choice(ld.N, size=s, replace=False, p=w)] for s in sizes]
    if kind == "uniform":
        return [[ld.universe[j] for j in rng.choice(ld.N, size=s, replace=False)] for s in sizes]
    raise ValueError(kind)


def cmd_null(layers, kinds, reps, rep_start):
    terms = json.loads((OUT / "terms.json").read_text())
    (OUT / "null").mkdir(parents=True, exist_ok=True)
    for li in layers:
        ld = LayerData(li, terms)
        real_cnt = np.zeros(ld.N)
        for L in ld.lists:
            for g in L:
                real_cnt[ld.g2i[g]] += 1
        for kind in kinds:
            path = OUT / f"null/layer_{li:02d}_{kind}.json"
            done = json.loads(path.read_text()) if path.exists() else []
            have = {d["rep"] for d in done}
            for r in range(rep_start, rep_start + reps):
                if r in have:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(SEED + 1000 * li + KIND_OFFSET[kind] + r)
                fake = make_null(ld, kind, rng)
                res, _ = rules(ld, fake, want_rows=False, sensitivity=False)
                cnt = np.zeros(ld.N)
                for L in fake:
                    for g in L:
                        cnt[ld.g2i[g]] += 1
                same_slots = sum(len(set(a) & set(b)) for a, b in zip(fake, ld.lists)) / sum(len(a) for a in ld.lists)
                res.update({"layer": li, "kind": kind, "rep": r,
                            "seed": SEED + 1000 * li + KIND_OFFSET[kind] + r,
                            "gene_count_corr_with_real": float(np.corrcoef(cnt, real_cnt)[0, 1]),
                            "gene_count_max_abs_diff": float(np.abs(cnt - real_cnt).max()),
                            "frac_gene_slots_same_as_real_list": float(same_slots),
                            "seconds": round(time.time() - t0, 1)})
                done.append(res)
                path.write_text(json.dumps(done, indent=1))
                print(f"L{li} {kind} rep{r}: A={res['A_deployed']['rate']:.3f} B={res['B_perfeature_bh']['rate']:.4f} "
                      f"C={res['C_global_bh']['n_annotated']} corr={res['gene_count_corr_with_real']:.3f} "
                      f"same={same_slots:.3f} ({res['seconds']}s)", flush=True)


# ---------------------------------------------------------------- summary
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def cmd_summary():
    rows = []
    for li in range(12):
        r = json.loads((OUT / f"real/layer_{li:02d}.json").read_text())
        F = r["n_features"]
        row = {"layer": li, "n_alive": F,
               "deployed_file_rate": r["repro_check"]["deployed_rate"],
               "repro_exact": r["repro_check"]["exact_match"]}
        for key, short in [("A_deployed", "A"), ("A2_overlap2_only", "A2"), ("B_perfeature_bh", "B"),
                           ("C_global_bh", "C"), ("Bu_perfeature_bh_Kuniverse", "Bu"), ("Cu_global_bh_Kuniverse", "Cu")]:
            row[f"{short}_n"] = r[key]["n_annotated"]
            row[f"{short}_rate"] = r[key]["n_annotated"] / F
            row[f"{short}_ge3"] = r[key]["n_ge3"]
        lo, hi = wilson(r["B_perfeature_bh"]["n_annotated"], F)
        row["B_rate_wilson_lo"], row["B_rate_wilson_hi"] = lo, hi
        for kind in ["freq", "curveball", "uniform"]:
            path = OUT / f"null/layer_{li:02d}_{kind}.json"
            if not path.exists():
                continue
            nl = json.loads(path.read_text())
            row[f"null_{kind}_reps"] = len(nl)
            for short, key in [("A", "A_deployed"), ("B", "B_perfeature_bh"), ("C", "C_global_bh")]:
                v = np.array([d[key]["rate"] for d in nl])
                real = row[f"{short}_rate"]
                row[f"null_{kind}_{short}_mean"] = float(v.mean())
                row[f"null_{kind}_{short}_p2.5"] = float(np.percentile(v, 2.5))
                row[f"null_{kind}_{short}_p97.5"] = float(np.percentile(v, 97.5))
                row[f"null_{kind}_{short}_max"] = float(v.max())
                row[f"excess_{kind}_{short}"] = real - float(v.mean())
                row[f"emp_p_{kind}_{short}"] = (1 + int((v >= real).sum())) / (len(v) + 1)
                row[f"null_{kind}_{short}_sd"] = float(v.std(ddof=1))
                row[f"z_{kind}_{short}"] = (real - float(v.mean())) / float(v.std(ddof=1)) if v.std(ddof=1) > 0 else None
                row[f"frac_null_reps_below_real_{kind}_{short}"] = float((v < real).mean())
                if short == "A":
                    row[f"null_{kind}_A_ge3_mean"] = float(np.mean([d[key]["n_ge3"] for d in nl]))
                if short == "B":
                    row[f"null_{kind}_B_n_mean"] = float(np.mean([d[key]["n_annotated"] for d in nl]))
                    row[f"null_{kind}_B_ge3_mean"] = float(np.mean([d[key]["n_ge3"] for d in nl]))
                    row[f"chance_share_{kind}_B"] = row[f"null_{kind}_B_n_mean"] / max(row["B_n"], 1)
                if short == "C":
                    row[f"null_{kind}_C_n_mean"] = float(np.mean([d[key]["n_annotated"] for d in nl]))
                    row[f"null_{kind}_C_n_max"] = int(max(d[key]["n_annotated"] for d in nl))
            row[f"null_{kind}_count_corr_min"] = float(min(d["gene_count_corr_with_real"] for d in nl))
            row[f"null_{kind}_same_slots_mean"] = float(np.mean([d["frac_gene_slots_same_as_real_list"] for d in nl]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    # markdown tables
    lines = ["| Layer | Alive | Deployed rule (file) | Deployed rule, freq null mean [2.5-97.5%] | Deployed excess over freq null "
             "| Per-feature BH (spec) | Wilson 95% | Per-feature BH, freq null mean [max] | Global BH features | Global BH, freq null mean |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def g(k, d=np.nan):
            return r.get(k, d)
        lines.append(
            f"| L{r['layer']} | {r['n_alive']} | {r['A_rate']*100:.1f}% ({r['A_n']}) | "
            f"{g('null_freq_A_mean')*100:.1f}% [{g('null_freq_A_p2.5')*100:.1f}-{g('null_freq_A_p97.5')*100:.1f}] | "
            f"{g('excess_freq_A')*100:+.1f} pts | {r['B_rate']*100:.2f}% ({r['B_n']}) | "
            f"{r['B_rate_wilson_lo']*100:.2f}-{r['B_rate_wilson_hi']*100:.2f} | "
            f"{g('null_freq_B_mean')*100:.3f}% [{g('null_freq_B_max')*100:.3f}] | {r['C_n']} | "
            f"{g('null_freq_C_mean')*r['n_alive']:.2f} |")
    (OUT / "summary_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # layer profile: does the null reproduce the deployed layer-to-layer shape?
    extra = {}
    for kind in ["freq", "curveball", "uniform"]:
        a = df["A_rate"].to_numpy(); b = df[f"null_{kind}_A_mean"].to_numpy()
        extra[f"pearson_layer_profile_deployed_real_vs_{kind}_null"] = float(np.corrcoef(a, b)[0, 1])
        a = df["B_rate"].to_numpy(); b = df[f"null_{kind}_B_mean"].to_numpy()
        extra[f"pearson_layer_profile_perfeature_bh_real_vs_{kind}_null"] = float(np.corrcoef(a, b)[0, 1])
    extra["global_bh_features_total_all_layers"] = int(df["C_n"].sum())
    extra["global_bh_null_freq_mean_total_all_layers"] = float(df["null_freq_C_n_mean"].sum())

    # downstream: the 50 Phase-6 "richly annotated" picks (phase6_patching.py:56-66, layer 5)
    p6 = pd.read_csv(OUTPUTS / "phase6/causal_patching.csv")
    ids = p6["feature_id"].tolist()
    dep = pd.read_csv(OUTPUTS / "phase2/layer_05/significant_enrichments.csv")
    cnt = dep.groupby("feature_id").size()
    cat = json.loads((OUTPUTS / "phase2/layer_05/feature_catalog.json").read_text())
    rich_now = sorted([c["feature_id"] for c in cat if cnt.get(c["feature_id"], 0) >= 3],
                      key=lambda f: -cnt.get(f, 0))[:50]
    fl = pd.read_csv(OUT / "real/layer_05_feature_flags.csv").set_index("feature_id")
    sel = fl.loc[ids]
    extra["phase6_picks"] = {
        "n_picks": len(ids),
        "picks_equal_top50_of_current_deployed_file": int(len(set(ids) & set(rich_now))),
        "deployed_enrichment_count_range": [int(cnt.reindex(ids).min()), int(cnt.reindex(ids).max())],
        "annotated_perfeature_bh": int((sel["n_sig_perfeature_bh"] > 0).sum()),
        "ge3_perfeature_bh": int((sel["n_sig_perfeature_bh"] >= 3).sum()),
        "annotated_global_bh": int((sel["n_sig_global_bh"] > 0).sum()),
        "layer5_features_ge3_perfeature_bh": int((fl["n_sig_perfeature_bh"] >= 3).sum()),
    }
    (OUT / "summary_extra.json").write_text(json.dumps(extra, indent=2))
    print(json.dumps(extra, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "real", "null", "summary"])
    ap.add_argument("--layers", default="0-11")
    ap.add_argument("--kinds", default="freq,curveball,uniform")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rep_start", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep()
    elif a.cmd == "real":
        cmd_real(parse_layers(a.layers))
    elif a.cmd == "null":
        cmd_null(parse_layers(a.layers), a.kinds.split(","), a.reps, a.rep_start)
    else:
        cmd_summary()


if __name__ == "__main__":
    main()
