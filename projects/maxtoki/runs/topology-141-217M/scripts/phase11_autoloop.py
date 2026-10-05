"""Phase 11 — autoloop substitute (Claude-as-brainstormer batch).

The source paper ran 53 iterations of Codex 5.3-xhigh in an executor-brainstormer
autoloop, producing ~141 hypotheses across 9 families with retirement of
2-consecutive-negatives. We don't have Codex; instead, this script runs a
curated batch of 6 additional hypotheses chosen to extend coverage of families
not yet exhausted on MaxToki, applies the retirement policy explicitly, and
emits per-iteration reports plus a final synthesis.

The 6 hypotheses (one per "iteration"):

  iter_01  H-orc       Ollivier-Ricci curvature direction (companion to H23-Forman)
  iter_02  H-hyp       4-point Gromov hyperbolicity (companion to paper's H30)
  iter_03  H-coex-res  Coexpression-residualized triangle-defect (does H70 survive?)
  iter_04  H-cross     Cross-layer correlation of H123 motif score (single axis?)
  iter_05  H-tokfreq   H123 stratified by token frequency (popularity confound?)
  iter_06  H-pooled    Pooled multi-tissue kNN — does H123 strengthen?

Each outputs to outputs/phase11_autoloop/iter_NN/ with a JSON result + decision.
A final outputs/phase11_autoloop/autoloop_summary.json compiles all decisions.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P11 = RUN / "outputs/phase11_autoloop"
P11.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase78_community_signed_motif import (
    louvain_communities, comembership_matrix,
)
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature, coexpression_matrix,
    adaptive_knn_graph,
)
from phase123_splits_nulls import load_trrust_signed

DOMAINS = ["lung", "immune", "external_lung"]
KNN = 12
N_NULL = 50


# ---------- shared utils -------------------------------------------------------

def per_layer_pair_data(domain: str):
    """Yield (layer, E_kept, log_expr_kept, P_lay, kept_syms) for each layer."""
    d_dir = P0 / domain
    emb = np.load(d_dir / "layer_gene_embeddings_pca20.npy")
    log_expr = np.load(d_dir / "log_expression_pool.npy")
    gf = pd.read_csv(d_dir / "gene_features.csv")
    pairs = pd.read_csv(P1 / domain / "pair_table.csv")
    n_layers = emb.shape[0]
    for li in range(n_layers):
        E = emb[li]
        keep = np.linalg.norm(E, axis=1) > 0
        if keep.sum() < 50:
            continue
        E_kept = E[keep]
        mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
        new_idx = -np.ones(len(gf), dtype=np.int64)
        new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
        valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
        P_lay = pairs[valid].reset_index(drop=True).copy()
        P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
        P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
        log_kept = log_expr[:, mask_g]
        kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(),
                                        mask_g.tolist()) if m]
        yield li, E_kept, log_kept, P_lay, kept_syms


def auc_or_nan(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def decision_for_layered(by_domain_layer: list[dict]) -> str:
    """Apply retirement policy to a list of {domain, layer, delta_vs_null_p95} rows.

    PROMOTE if positive null-gap in ≥4/12 layers AND ≥2/3 domains
    INCONCLUSIVE if positive in 1 domain only
    RETIRE otherwise.
    """
    df = pd.DataFrame(by_domain_layer)
    if df.empty or "delta_vs_null_p95" not in df:
        return "RETIRE_no_signal"
    pos = df[df["delta_vs_null_p95"] > 0]
    by_domain = pos.groupby("domain").size()
    n_domains_with_4plus = int((by_domain >= 4).sum())
    n_domains_with_any = int((by_domain >= 1).sum())
    if n_domains_with_4plus >= 2:
        return "PROMOTE"
    if n_domains_with_any >= 1:
        return "INCONCLUSIVE"
    return "RETIRE"


# ---------- iter_01 — Ollivier-Ricci curvature direction ----------------------

def iter_01_ollivier_ricci():
    """For each (i,j) edge in the kNN graph, compute Ollivier-Ricci curvature
    by Wasserstein-1 distance between uniform measures on Ni and Nj. Test:
    are HIGH-curvature edges LESS regulatory (paper's H23-Forman direction)?"""
    rows = []
    for domain in DOMAINS:
        for li, E, _, P_lay, _ in per_layer_pair_data(domain):
            D = pairwise_distances(E)
            n = D.shape[0]
            np.fill_diagonal(D, np.inf)
            nbr = np.argsort(D, axis=1)[:, :KNN]
            # per-pair ORC via simplified W1: 1 - mean_{a in Ni, b in Nj} (d(a,b)/D(i,j))
            pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
            y = P_lay["is_trrust"].to_numpy()
            orc = np.zeros(len(P_lay), dtype=np.float32)
            for e in range(len(P_lay)):
                i = int(pi[e]); j = int(pj[e])
                d_ij = D[i, j] if np.isfinite(D[i, j]) else float(D[D < np.inf].max())
                Ni = nbr[i]; Nj = nbr[j]
                inter_d = D[np.ix_(Ni, Nj)]
                inter_d = inter_d[np.isfinite(inter_d)]
                if inter_d.size == 0 or d_ij == 0:
                    orc[e] = 0.0
                else:
                    orc[e] = float(1.0 - inter_d.mean() / max(1e-9, d_ij))
            # Direction: HIGH curvature should mean LESS regulatory (paper's pattern)
            auc_high_less = auc_or_nan(y, -orc)  # negate so high orc → low score
            rows.append(dict(domain=domain, layer=li,
                              auc_high_less_regulatory=auc_high_less,
                              direction_replicates_paper=bool(auc_high_less < 0.5),
                              delta_vs_null_p95=0.5 - auc_high_less,  # null is 0.5
                              ))
    return rows


# ---------- iter_02 — 4-point Gromov hyperbolicity ----------------------------

def iter_02_gromov_hyperbolicity():
    """Sample 4-point subsets and compute Gromov delta. Compare distribution
    to a euclidean reference (delta=0). Layer-wise."""
    rng = np.random.default_rng(42)
    rows = []
    for domain in DOMAINS:
        for li, E, _, P_lay, _ in per_layer_pair_data(domain):
            D = pairwise_distances(E)
            n = D.shape[0]
            n_sample = 200
            deltas = []
            for _ in range(n_sample):
                idx = rng.choice(n, size=4, replace=False)
                d = D[np.ix_(idx, idx)]
                d12 = d[0, 1] + d[2, 3]
                d13 = d[0, 2] + d[1, 3]
                d14 = d[0, 3] + d[1, 2]
                sums = sorted([d12, d13, d14])
                deltas.append((sums[2] - sums[1]) / 2.0)
            deltas = np.array(deltas)
            # Normalise by mean pairwise distance: delta_norm = mean delta / mean D
            mean_d = float(np.nanmean(D[D < np.inf]))
            delta_norm = float(deltas.mean() / max(1e-9, mean_d))
            # Lower delta_norm = more hyperbolic. Euclidean reference ≈ 0.05–0.10 typically.
            # Decision: is the manifold meaningfully more hyperbolic than euclidean?
            rows.append(dict(domain=domain, layer=li,
                              mean_delta_norm=delta_norm,
                              meaningfully_hyperbolic=bool(delta_norm < 0.05),
                              delta_vs_null_p95=0.05 - delta_norm,  # negative → not hyperbolic
                              ))
    return rows


# ---------- iter_03 — coexpression-residualized triangle defect ---------------

def iter_03_coex_residual_triangle():
    """Recompute triangle-defect ΔAUROC after regressing the triangle feature
    against coexpression. Does any signal survive once explicit coexp is removed?"""
    rows = []
    for domain in DOMAINS:
        for li, E, log_expr, P_lay, _ in per_layer_pair_data(domain):
            tri = sum(triangle_defect_feature(E, k=k) for k in [8, 12, 16])
            coex = coexpression_matrix(log_expr)
            pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
            y = P_lay["is_trrust"].to_numpy()
            t = tri[pi, pj].astype(np.float64)
            c = coex[pi, pj].astype(np.float64)
            # Residualise t against c (linear)
            A = np.column_stack([c, np.ones_like(c)])
            beta, *_ = np.linalg.lstsq(A, t, rcond=None)
            t_resid = t - A @ beta
            auc_resid = auc_or_nan(y, t_resid)
            auc_raw = auc_or_nan(y, t)
            # Permutation null on residual
            rng = np.random.default_rng(42)
            null_aucs = []
            for _ in range(N_NULL):
                null_aucs.append(auc_or_nan(rng.permutation(y), t_resid))
            null_p95 = float(np.percentile(null_aucs, 95))
            rows.append(dict(domain=domain, layer=li,
                              auc_resid=auc_resid, auc_raw=auc_raw,
                              null_p95=null_p95,
                              delta_vs_null_p95=auc_resid - null_p95,
                              ))
    return rows


# ---------- iter_04 — cross-layer H123 motif-score correlation ----------------

def iter_04_cross_layer_consistency():
    """Compute the H123 signed-motif-community feature per layer, then
    correlate the per-pair feature vectors across layers. Cross-layer stability
    high → motif signal is a single axis; low → layer-specific."""
    trrust = load_trrust_signed()
    rows = []
    for domain in DOMAINS:
        d_dir = P0 / domain
        emb = np.load(d_dir / "layer_gene_embeddings_pca20.npy")
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        n_layers = emb.shape[0]
        per_layer_feat = []
        keep_layer = []
        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            if keep.sum() < 50:
                continue
            E_kept = E[keep]
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
            kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(),
                                            mask_g.tolist()) if m]
            sym_to_idx = {s: i for i, s in enumerate(kept_syms)}
            D = pairwise_distances(E_kept)
            n = D.shape[0]
            np.fill_diagonal(D, np.inf)
            nbr = np.argsort(D, axis=1)[:, :KNN]
            g = nx.Graph(); g.add_nodes_from(range(n))
            for i in range(n):
                for j in nbr[i]:
                    g.add_edge(int(i), int(j))
            labels = louvain_communities(g, seed=42)
            com = comembership_matrix(labels)
            trrust_in = trrust[
                trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
            ].reset_index(drop=True)
            if len(trrust_in) == 0:
                continue
            feat = np.zeros((n, n), dtype=np.float32)
            for sign, row in zip(trrust_in["sign"].to_numpy(), trrust_in.itertuples()):
                tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
                same_com = bool(com[tf, tgt])
                consistency = 1.0 if (
                    (sign > 0 and same_com) or (sign < 0 and not same_com)
                ) else (-1.0 if sign != 0 else 0.0)
                feat[tf, tgt] += consistency
                feat[tgt, tf] += consistency
            pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
            f = feat[pi, pj]
            per_layer_feat.append((li, f, P_lay))
            keep_layer.append(li)
        # Compute pairwise Pearson correlations between layers (on common pair indexing)
        # Use first layer's P_lay as reference; assume P_lay same across layers (it is, since pairs is global)
        if len(per_layer_feat) < 2:
            continue
        feats = np.stack([f for _, f, _ in per_layer_feat], axis=0)  # (L, n_pairs)
        corr = np.corrcoef(feats)
        mean_off_diag = float((corr.sum() - np.trace(corr)) /
                              (corr.shape[0] * (corr.shape[0] - 1)))
        # Per-layer AUROC against TRRUST labels
        y = per_layer_feat[0][2]["is_trrust"].to_numpy()
        per_layer_auc = [auc_or_nan(y, f) for _, f, _ in per_layer_feat]
        rows.append(dict(domain=domain,
                          n_layers=int(len(per_layer_feat)),
                          mean_cross_layer_pearson=mean_off_diag,
                          per_layer_auc=per_layer_auc,
                          # Decision: signal IS layer-specific if mean cross-layer pearson < 0.3
                          single_axis_likely=bool(mean_off_diag > 0.5),
                          delta_vs_null_p95=mean_off_diag - 0.3,
                          ))
    # Pivot to per-domain-layer-equivalent: rows here are per-domain summary
    return rows


# ---------- iter_05 — H123 stratified by token frequency ----------------------

def iter_05_token_frequency_strata():
    """Stratify gene pairs by total observation count (n_observations from
    gene_features.csv, summed across the pair). Test whether the H123 motif
    signal is concentrated in the high-popularity stratum (tokenization
    confound) or distributed."""
    trrust = load_trrust_signed()
    rows = []
    for domain in DOMAINS:
        gf = pd.read_csv(P0 / domain / "gene_features.csv")
        for li, E, _, P_lay, kept_syms in per_layer_pair_data(domain):
            sym_to_idx = {s: i for i, s in enumerate(kept_syms)}
            sym_to_obs = dict(zip(gf["symbol"].astype(str).str.upper(),
                                    gf["n_observations"]))
            n_obs_kept = np.array([sym_to_obs.get(s, 0) for s in kept_syms],
                                   dtype=np.float64)
            pi = P_lay["i"].to_numpy(); pj = P_lay["j"].to_numpy()
            pair_pop = np.log1p(n_obs_kept[pi]) + np.log1p(n_obs_kept[pj])
            try:
                strata = pd.qcut(pair_pop, q=3, duplicates="drop", labels=False)
            except Exception:
                continue
            if strata is None:
                continue
            y = P_lay["is_trrust"].to_numpy()
            # Compute H123-like motif score (no triangle, no degree strata for speed)
            D = pairwise_distances(E)
            n = D.shape[0]
            np.fill_diagonal(D, np.inf)
            nbr = np.argsort(D, axis=1)[:, :KNN]
            g = nx.Graph(); g.add_nodes_from(range(n))
            for i in range(n):
                for j in nbr[i]:
                    g.add_edge(int(i), int(j))
            labels = louvain_communities(g, seed=42)
            com = comembership_matrix(labels)
            trrust_in = trrust[
                trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
            ].reset_index(drop=True)
            if len(trrust_in) == 0:
                continue
            feat = np.zeros((n, n), dtype=np.float32)
            for sign, row in zip(trrust_in["sign"].to_numpy(), trrust_in.itertuples()):
                tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
                same_com = bool(com[tf, tgt])
                consistency = 1.0 if (
                    (sign > 0 and same_com) or (sign < 0 and not same_com)
                ) else (-1.0 if sign != 0 else 0.0)
                feat[tf, tgt] += consistency
                feat[tgt, tf] += consistency
            f = feat[pi, pj]
            per_stratum = {}
            for s in [0, 1, 2]:
                mask = strata == s
                if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                    continue
                per_stratum[f"q{s}"] = auc_or_nan(y[mask], f[mask])
            rows.append(dict(domain=domain, layer=li, **per_stratum,
                              auc_overall=auc_or_nan(y, f),
                              # Decision rule: significant difference high vs low stratum?
                              delta_vs_null_p95=(per_stratum.get("q2", 0.5)
                                                  - per_stratum.get("q0", 0.5)) - 0.05,
                              ))
    return rows


# ---------- iter_06 — pooled multi-tissue kNN ---------------------------------

def iter_06_pooled_multi_tissue():
    """Pool genes from all 3 domains into one common kNN; check if H123 motif
    score is stronger than per-domain (i.e., does the model encode tissue-
    invariant gene geometry that helps regulatory signal)?"""
    trrust = load_trrust_signed()
    # Find common genes across all 3 domains
    common = None
    domain_emb = {}
    domain_pairs = {}
    for domain in DOMAINS:
        d_dir = P0 / domain
        gf = pd.read_csv(d_dir / "gene_features.csv")
        emb = np.load(d_dir / "layer_gene_embeddings_pca20.npy")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        syms_set = set(gf["symbol"].astype(str).str.upper().tolist())
        if common is None:
            common = syms_set
        else:
            common &= syms_set
        domain_emb[domain] = (emb, gf)
        domain_pairs[domain] = pairs
    common_syms = sorted(common)
    sym_to_idx_common = {s: i for i, s in enumerate(common_syms)}
    n_common = len(common_syms)
    if n_common < 50:
        return [dict(error=f"only {n_common} common genes")]

    rows = []
    n_layers = next(iter(domain_emb.values()))[0].shape[0]
    for li in range(n_layers):
        per_domain_E = []
        for domain in DOMAINS:
            emb, gf = domain_emb[domain]
            sym_in_domain = gf["symbol"].astype(str).str.upper().tolist()
            sym_to_dom_idx = {s: i for i, s in enumerate(sym_in_domain)}
            E = emb[li]
            keep_idx = np.array([sym_to_dom_idx[s] for s in common_syms if s in sym_to_dom_idx],
                                 dtype=int)
            if len(keep_idx) != n_common:
                # alignment issue, skip
                break
            per_domain_E.append(E[keep_idx])
        if len(per_domain_E) != 3:
            continue
        # Average across domains (mean embedding for each common gene)
        E_pooled = np.stack(per_domain_E, axis=0).mean(axis=0)
        keep = np.linalg.norm(E_pooled, axis=1) > 0
        if keep.sum() < 50:
            continue
        E_pooled_kept = E_pooled[keep]
        kept_syms = [s for s, m in zip(common_syms, keep.tolist()) if m]
        sym_to_idx = {s: i for i, s in enumerate(kept_syms)}

        # Build pair_table on common pool
        # Use TRRUST membership for labels
        trrust_in = trrust[
            trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
        ]
        # Construct pair table: for each TF-target in trrust_in, plus equally
        # many random non-TRRUST pairs
        n_pos = len(trrust_in)
        if n_pos < 10:
            continue
        pairs_pos = [(sym_to_idx[r.tf], sym_to_idx[r.target], 1) for r in trrust_in.itertuples()]
        rng = np.random.default_rng(42)
        n = len(kept_syms)
        pos_set = set((min(i, j), max(i, j)) for i, j, _ in pairs_pos)
        pairs_neg = []
        while len(pairs_neg) < n_pos:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if (min(i, j), max(i, j)) in pos_set:
                continue
            pairs_neg.append((int(i), int(j), 0))
        all_pairs = pairs_pos + pairs_neg
        pi = np.array([p[0] for p in all_pairs])
        pj = np.array([p[1] for p in all_pairs])
        y = np.array([p[2] for p in all_pairs])

        D = pairwise_distances(E_pooled_kept)
        np.fill_diagonal(D, np.inf)
        nbr = np.argsort(D, axis=1)[:, :KNN]
        g = nx.Graph(); g.add_nodes_from(range(n))
        for i in range(n):
            for j in nbr[i]:
                g.add_edge(int(i), int(j))
        labels = louvain_communities(g, seed=42)
        com = comembership_matrix(labels)
        feat = np.zeros((n, n), dtype=np.float32)
        for sign, row in zip(trrust_in["sign"].to_numpy(), trrust_in.itertuples()):
            tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
            same_com = bool(com[tf, tgt])
            consistency = 1.0 if (
                (sign > 0 and same_com) or (sign < 0 and not same_com)
            ) else (-1.0 if sign != 0 else 0.0)
            feat[tf, tgt] += consistency
            feat[tgt, tf] += consistency
        f = feat[pi, pj]
        auc = auc_or_nan(y, f)
        # null: shuffle signs
        null_aucs = []
        for _ in range(N_NULL):
            shuffled = trrust_in["sign"].sample(frac=1, random_state=rng.integers(1e9)).to_numpy()
            feat_n = np.zeros((n, n), dtype=np.float32)
            for sign, row in zip(shuffled, trrust_in.itertuples()):
                tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
                same_com = bool(com[tf, tgt])
                consistency = 1.0 if (
                    (sign > 0 and same_com) or (sign < 0 and not same_com)
                ) else (-1.0 if sign != 0 else 0.0)
                feat_n[tf, tgt] += consistency
                feat_n[tgt, tf] += consistency
            null_aucs.append(auc_or_nan(y, feat_n[pi, pj]))
        null_p95 = float(np.percentile(null_aucs, 95))
        rows.append(dict(domain="pooled_3tissue", layer=li,
                          n_common_genes=int(n_common), n_pairs=int(len(y)),
                          auc=auc, null_p95=null_p95,
                          delta_vs_null_p95=auc - null_p95,
                          ))
    return rows


# ---------- driver ------------------------------------------------------------

ITERATIONS = [
    ("iter_01_ollivier_ricci", iter_01_ollivier_ricci, "H-orc",
     "Ollivier-Ricci curvature direction (high curvature = less regulatory)"),
    ("iter_02_gromov_hyperbolicity", iter_02_gromov_hyperbolicity, "H-hyp",
     "4-point Gromov hyperbolicity per layer"),
    ("iter_03_coex_residual_triangle", iter_03_coex_residual_triangle, "H-coex-res",
     "Coexpression-residualized triangle-defect ΔAUROC"),
    ("iter_04_cross_layer_consistency", iter_04_cross_layer_consistency, "H-cross",
     "Cross-layer Pearson of H123 motif score per domain"),
    ("iter_05_token_frequency_strata", iter_05_token_frequency_strata, "H-tokfreq",
     "H123 motif AUROC stratified by token-popularity tertile"),
    ("iter_06_pooled_multi_tissue", iter_06_pooled_multi_tissue, "H-pooled",
     "H123 on pooled multi-tissue kNN over common gene set"),
]


def main():
    autoloop_summary = []
    consecutive_negatives = 0
    for tag, fn, hyp_id, desc in ITERATIONS:
        print(f"\n{'='*72}\n[{tag}] {hyp_id}: {desc}\n{'='*72}")
        rows = fn()
        decision = decision_for_layered(rows) if rows else "RETIRE_no_signal"
        out_dir = P11 / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rows.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)
        report = dict(iteration=tag, hypothesis_id=hyp_id, description=desc,
                       n_rows=len(rows), decision=decision,
                       brief=summarize_rows(rows))
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        autoloop_summary.append(report)
        print(f"  decision: {decision}")
        print(f"  brief:    {report['brief']}")
        if decision.startswith("RETIRE"):
            consecutive_negatives += 1
        else:
            consecutive_negatives = 0
        if consecutive_negatives >= 3:
            print(f"\n[autoloop] 3 consecutive RETIRE — stopping early")
            break

    final = dict(
        n_iterations=len(autoloop_summary),
        per_iter=autoloop_summary,
        n_promote=sum(1 for r in autoloop_summary if r["decision"] == "PROMOTE"),
        n_inconclusive=sum(1 for r in autoloop_summary if r["decision"] == "INCONCLUSIVE"),
        n_retire=sum(1 for r in autoloop_summary
                      if r["decision"].startswith("RETIRE")),
    )
    with open(P11 / "autoloop_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print("\n" + "=" * 72)
    print("AUTOLOOP DONE")
    print("=" * 72)
    print(json.dumps(final, indent=2, default=str))


def summarize_rows(rows: list[dict]) -> str:
    if not rows:
        return "no rows"
    df = pd.DataFrame(rows)
    if "delta_vs_null_p95" not in df.columns:
        return f"{len(rows)} rows, no delta_vs_null_p95"
    by_domain = df.groupby("domain")["delta_vs_null_p95"].agg(["mean", "count"])
    pos = df[df["delta_vs_null_p95"] > 0]
    return (f"{len(rows)} rows, mean Δ_vs_null={df['delta_vs_null_p95'].mean():.3f}, "
            f"positive {len(pos)}/{len(rows)}, by-domain {by_domain.to_dict()}")


if __name__ == "__main__":
    main()
