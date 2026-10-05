"""Phase 8 extensions — H124 → H127 → H130 → H138 annotation-extension chain.

Closes the README's deliberately-skipped item: the source paper's degradation
chain that progressively adds biological annotations to H123, with the
counterintuitive result that raw effect size grows while null-gap robustness
collapses (H123 6/6 → H124 <3/6 → H127 2/9 → H130 0/9 → H138 0/9).

Each variant extends H123 with one additional feature concatenated into the
combined classifier (z-summed with the existing motif × triangle stack):

  - H124: + STRING-PPI co-membership (binary, threshold 700)
  - H127: + GO Biological Process co-membership (binary; share ≥1 BP term)
  - H130: + GO BP semantic similarity (continuous; Jaccard of BP term sets)
  - H138: + ontology-sheaf approximation (continuous: weighted Jaccard with
          IC weighting + endpoint-degree term)

Null: TF-identity-preserving sign shuffle stratified by (TF-degree quantile,
target-degree quantile) — identical to H123. The extension features are NOT
shuffled (they are static gene-pair properties), so they enter both observed
and null AUROC and the comparison cleanly tests "does the extension help the
*null* explain the data more than it helps the observed signal".

Outputs:
  outputs/phase8_extensions/h124_string.csv
  outputs/phase8_extensions/h127_go_comem.csv
  outputs/phase8_extensions/h130_go_semantic.csv
  outputs/phase8_extensions/h138_ontology_sheaf.csv
  outputs/phase8_extensions/phase8ext_summary.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P8X = RUN / "outputs/phase8_extensions"
P8X.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))

from phase123_splits_nulls import load_trrust_signed
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature,
)
from phase78_community_signed_motif import (
    louvain_communities, comembership_matrix,
)

DOMAINS = ["lung", "immune", "external_lung"]
KNN = 12
N_NULL = 50

BIOM = Path("<DATA_ROOT>")
STRING_JSON = BIOM / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"
GO_BP_JSON = BIOM / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"


# ---------- annotation loaders --------------------------------------------------

def load_string_pairs(threshold: int = 700) -> set[tuple[str, str]]:
    with open(STRING_JSON) as f:
        data = json.load(f)
    key = f"pairs_{threshold}"
    edges = set()
    for a, b in data[key]:
        edges.add(tuple(sorted([a.upper(), b.upper()])))
    return edges


def load_go_bp() -> dict[str, set[str]]:
    """Returns {term: set_of_gene_symbols}, upper-cased."""
    with open(GO_BP_JSON) as f:
        data = json.load(f)
    return {term: set(g.upper() for g in genes) for term, genes in data.items()}


def gene_to_go_terms(go_bp: dict[str, set[str]]) -> dict[str, set[str]]:
    """Inverse map: gene -> set of BP terms."""
    out: dict[str, set[str]] = {}
    for term, genes in go_bp.items():
        for g in genes:
            out.setdefault(g, set()).add(term)
    return out


def go_term_ic(go_bp: dict[str, set[str]]) -> dict[str, float]:
    """Information content per term: -log(|term| / total_universe_size)."""
    universe = set()
    for genes in go_bp.values():
        universe |= genes
    n = max(1, len(universe))
    return {term: -math.log(max(1, len(genes)) / n) for term, genes in go_bp.items()}


# ---------- pair-feature builders ----------------------------------------------

def build_string_comem(syms: list[str], string_edges: set[tuple[str, str]]) -> np.ndarray:
    n = len(syms)
    feat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if (tuple(sorted([syms[i], syms[j]]))) in string_edges:
                feat[i, j] = feat[j, i] = 1.0
    return feat


def build_go_comem(syms: list[str], gene2terms: dict[str, set[str]]) -> np.ndarray:
    n = len(syms)
    feat = np.zeros((n, n), dtype=np.float32)
    sets = [gene2terms.get(s, set()) for s in syms]
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            if sets[j] & si:
                feat[i, j] = feat[j, i] = 1.0
    return feat


def build_go_jaccard(syms: list[str], gene2terms: dict[str, set[str]]) -> np.ndarray:
    n = len(syms)
    feat = np.zeros((n, n), dtype=np.float32)
    sets = [gene2terms.get(s, set()) for s in syms]
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets[j]
            if not sj:
                continue
            inter = len(si & sj); union = len(si | sj)
            feat[i, j] = feat[j, i] = inter / max(1, union)
    return feat


def build_go_ic_weighted(syms: list[str], gene2terms: dict[str, set[str]],
                         ic: dict[str, float]) -> np.ndarray:
    n = len(syms)
    feat = np.zeros((n, n), dtype=np.float32)
    sets = [gene2terms.get(s, set()) for s in syms]
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets[j]
            if not sj:
                continue
            inter_ic = sum(ic.get(t, 0.0) for t in (si & sj))
            union_ic = sum(ic.get(t, 0.0) for t in (si | sj))
            feat[i, j] = feat[j, i] = inter_ic / max(1e-9, union_ic)
    return feat


def build_ontology_sheaf(syms: list[str], gene2terms: dict[str, set[str]],
                         ic: dict[str, float], string_edges: set[tuple[str, str]]) -> np.ndarray:
    """Approximation of an ontology-sheaf section: IC-weighted GO Jaccard +
    a STRING-degree consistency term (does each gene's STRING degree match?)."""
    sheaf = build_go_ic_weighted(syms, gene2terms, ic)
    # STRING-degree per gene
    deg = np.zeros(len(syms), dtype=np.float32)
    sym_set = set(syms)
    for a, b in string_edges:
        if a in sym_set and b in sym_set:
            ia = syms.index(a); ib = syms.index(b)
            deg[ia] += 1; deg[ib] += 1
    # log-degree similarity: 1 / (1 + |log(da+1) - log(db+1)|)
    log_deg = np.log1p(deg)
    diff = np.abs(log_deg[:, None] - log_deg[None, :])
    deg_sim = 1.0 / (1.0 + diff)
    return 0.5 * sheaf + 0.5 * deg_sim.astype(np.float32)


# ---------- scoring -----------------------------------------------------------

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    s = x.std() + 1e-12
    return (x - x.mean()) / s


def signed_motif_with_extension(
    emb: np.ndarray,
    pairs: pd.DataFrame,
    sym_to_idx: dict[str, int],
    trrust: pd.DataFrame,
    triangle_baseline: np.ndarray,
    extension_pair_feat: np.ndarray | None,
    k: int = KNN,
    n_null: int = N_NULL,
    null_seed_base: int = 100,
) -> dict:
    D = pairwise_distances(emb)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    labels = louvain_communities(g, seed=42)
    com = comembership_matrix(labels)

    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    trrust_in = trrust[
        trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
    ].reset_index(drop=True)
    if len(trrust_in) == 0:
        return dict(error="no TRRUST edges in pool")
    obs_signs = trrust_in["sign"].to_numpy()

    def compute_motif_feature(sign_array: np.ndarray) -> np.ndarray:
        feat = np.zeros((n, n), dtype=np.float32)
        for sign, row in zip(sign_array, trrust_in.itertuples()):
            tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
            same_com = bool(com[tf, tgt])
            consistency = 1.0 if (
                (sign > 0 and same_com) or (sign < 0 and not same_com)
            ) else (-1.0 if sign != 0 else 0.0)
            feat[tf, tgt] += consistency
            feat[tgt, tf] += consistency
        return feat[pi, pj]

    y = pairs["is_trrust"].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return dict(error="degenerate labels")

    motif_obs = compute_motif_feature(obs_signs)
    tri_pair = triangle_baseline[pi, pj]
    if extension_pair_feat is not None:
        ext_pair = extension_pair_feat[pi, pj]
        combo_obs = zscore(motif_obs) + zscore(tri_pair) + zscore(ext_pair)
    else:
        ext_pair = None
        combo_obs = zscore(motif_obs) + zscore(tri_pair)

    auc_motif = float(roc_auc_score(y, motif_obs))
    auc_tri = float(roc_auc_score(y, tri_pair))
    auc_combo = float(roc_auc_score(y, combo_obs))

    # Degree-stratified sign shuffle (same as H123)
    tf_deg = trrust_in.groupby("tf").size()
    tgt_deg = trrust_in.groupby("target").size()
    tf_q = pd.qcut(trrust_in["tf"].map(tf_deg), q=4, duplicates="drop", labels=False)
    tgt_q = pd.qcut(trrust_in["target"].map(tgt_deg), q=4, duplicates="drop", labels=False)
    stratum = (tf_q.fillna(0).astype(int) * 10 + tgt_q.fillna(0).astype(int)).to_numpy()

    rng = np.random.default_rng(null_seed_base)
    null_motif_aucs = []
    null_combo_aucs = []
    for _ in range(n_null):
        shuffled = obs_signs.copy()
        for s_val in np.unique(stratum):
            idxs = np.where(stratum == s_val)[0]
            perm = rng.permutation(idxs)
            shuffled[idxs] = obs_signs[perm]
        motif_null = compute_motif_feature(shuffled)
        if extension_pair_feat is not None:
            combo_null = zscore(motif_null) + zscore(tri_pair) + zscore(ext_pair)
        else:
            combo_null = zscore(motif_null) + zscore(tri_pair)
        try:
            null_motif_aucs.append(roc_auc_score(y, motif_null))
            null_combo_aucs.append(roc_auc_score(y, combo_null))
        except Exception:
            pass

    null_motif = np.array(null_motif_aucs)
    null_combo = np.array(null_combo_aucs)

    return dict(
        auc_motif=auc_motif,
        auc_triangle_baseline=auc_tri,
        auc_combined=auc_combo,
        delta_combo_vs_baseline=auc_combo - auc_tri,
        null_motif_p95=float(np.percentile(null_motif, 95)),
        null_combo_p95=float(np.percentile(null_combo, 95)),
        delta_motif_vs_null_p95=auc_motif - float(np.percentile(null_motif, 95)),
        delta_combo_vs_null_p95=auc_combo - float(np.percentile(null_combo, 95)),
        null_motif_mean=float(null_motif.mean()),
        null_combo_mean=float(null_combo.mean()),
        n_trrust_in_pool=int(len(trrust_in)),
    )


# ---------- main --------------------------------------------------------------

def main():
    print("[load] TRRUST, STRING, GO BP")
    trrust = load_trrust_signed()
    string_edges = load_string_pairs(threshold=700)
    go_bp = load_go_bp()
    gene2terms = gene_to_go_terms(go_bp)
    ic = go_term_ic(go_bp)
    print(f"  string pairs (700): {len(string_edges)}")
    print(f"  GO BP terms: {len(go_bp)}, gene2terms: {len(gene2terms)}")

    extensions = ["H124_string", "H127_go_comem", "H130_go_semantic", "H138_ontology_sheaf"]
    rows_per_ext: dict[str, list] = {e: [] for e in extensions}

    for domain in DOMAINS:
        d_dir = P0 / domain
        emb = np.load(d_dir / "layer_gene_embeddings_pca20.npy")
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        n_layers = emb.shape[0]
        print(f"\n[{domain}] emb {emb.shape}  pairs {len(pairs)}")

        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            if keep.sum() < 50:
                continue
            E = E[keep]
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]

            kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(),
                                           mask_g.tolist()) if m]
            sym_to_idx_kept = {s: i for i, s in enumerate(kept_syms)}

            # Static gene-pair feature matrices (computed once per layer)
            string_feat = build_string_comem(kept_syms, string_edges)
            go_comem_feat = build_go_comem(kept_syms, gene2terms)
            go_jaccard_feat = build_go_jaccard(kept_syms, gene2terms)
            sheaf_feat = build_ontology_sheaf(kept_syms, gene2terms, ic, string_edges)

            tri = sum(triangle_defect_feature(E, k=k) for k in [8, 12, 16])

            tag_to_feat = {
                "H124_string": string_feat,
                "H127_go_comem": go_comem_feat,
                "H130_go_semantic": go_jaccard_feat,
                "H138_ontology_sheaf": sheaf_feat,
            }
            for tag, ext_feat in tag_to_feat.items():
                r = signed_motif_with_extension(
                    E, P_lay, sym_to_idx_kept, trrust, tri, ext_feat,
                    k=KNN, n_null=N_NULL,
                )
                if "error" in r:
                    continue
                r.update(domain=domain, layer=li, hypothesis=tag)
                rows_per_ext[tag].append(r)

            tags_short = "  ".join(
                f"{t.split('_')[0]} Δ={rows_per_ext[t][-1].get('delta_combo_vs_null_p95', float('nan')):+.3f}"
                if rows_per_ext[t] and rows_per_ext[t][-1]["domain"] == domain and rows_per_ext[t][-1]["layer"] == li
                else "" for t in extensions
            )
            print(f"  L{li:02d}  {tags_short}")

    summary = {}
    for tag in extensions:
        df = pd.DataFrame(rows_per_ext[tag])
        out_csv = P8X / f"{tag.lower()}.csv"
        df.to_csv(out_csv, index=False)
        per_domain = {}
        if not df.empty:
            for domain, sub in df.groupby("domain"):
                col = "delta_combo_vs_null_p95"
                per_domain[domain] = dict(
                    n_layers=int(len(sub)),
                    mean_delta=float(sub[col].mean()),
                    n_layers_positive_null_gap=int((sub[col] > 0).sum()),
                    fraction_layers_positive=float((sub[col] > 0).mean()),
                    mean_auc_combo=float(sub["auc_combined"].mean()),
                    mean_null_combo_mean=float(sub["null_combo_mean"].mean()),
                )
        summary[tag] = per_domain

    # Compose degradation table comparing to existing H123
    h123_csv = RUN / "outputs/phase78/h123_signed_motif_degree_strata.csv"
    if h123_csv.exists():
        h123 = pd.read_csv(h123_csv)
        per_domain_h123 = {}
        col = "delta_combo_vs_null_p95"
        for domain, sub in h123.groupby("domain"):
            per_domain_h123[domain] = dict(
                n_layers=int(len(sub)),
                mean_delta=float(sub[col].mean()),
                n_layers_positive_null_gap=int((sub[col] > 0).sum()),
                fraction_layers_positive=float((sub[col] > 0).mean()),
                mean_auc_combo=float(sub["auc_combined"].mean()),
                mean_null_combo_mean=float(sub["null_combo_mean"].mean()),
            )
        summary["H123_baseline"] = per_domain_h123

    with open(P8X / "phase8ext_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[phase 8 extensions] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
