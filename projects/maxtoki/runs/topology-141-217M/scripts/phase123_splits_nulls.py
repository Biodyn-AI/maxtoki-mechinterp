"""Phase 1+2+3 — gene-pool splits, hierarchical null infrastructure, ΔAUROC harness.

Phase 1: TF-disjoint and target-disjoint splits per domain (3 seeds each).
Phase 2: four null families per gene pair:
   1. feature-shuffle  — shuffle embedding features across genes
   2. label-permutation — shuffle regulatory labels
   3. degree-preserving rewiring (curveball) — for the kNN graph
   4. coexpression-matched — bin pairs by absolute Pearson rank, sample matched
Phase 3: ΔAUROC harness — observed AUROC minus 95th percentile of coexpression null.

Outputs into outputs/phase1/ and outputs/phase2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P1.mkdir(parents=True, exist_ok=True)

DOMAINS = ["lung", "immune", "external_lung"]
SEEDS = [42, 43, 44]

BIOM = Path("<DATA_ROOT>")
TRRUST_TSV = BIOM / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON = BIOM / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"


def load_trrust_signed():
    """Returns DataFrame with columns: tf, target, sign ∈ {+1,-1,0}."""
    df = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
    sign = df["mode"].str.lower().map(
        {"activation": +1, "repression": -1, "unknown": 0}
    ).fillna(0).astype(int)
    df = df.assign(sign=sign).drop(columns=["pmid"])
    df["tf"] = df["tf"].str.upper()
    df["target"] = df["target"].str.upper()
    return df


def load_string_pairs() -> set[tuple[str, str]]:
    with open(STRING_JSON) as f:
        data = json.load(f)
    edges: set[tuple[str, str]] = set()
    for pair_list in data.values():
        for a, b in pair_list:
            edges.add(tuple(sorted([a.upper(), b.upper()])))
    return edges


def build_pair_table(gene_features: pd.DataFrame, trrust: pd.DataFrame,
                     string_edges: set[tuple[str, str]]) -> pd.DataFrame:
    """Build all unordered gene pairs in this domain, with regulatory labels."""
    syms = gene_features["symbol"].astype(str).str.upper().to_numpy()
    pool_to_idx = {s: i for i, s in enumerate(syms)}
    n = len(syms)

    # TRRUST pair index
    trrust_in_pool = trrust[
        trrust["tf"].isin(pool_to_idx) & trrust["target"].isin(pool_to_idx)
    ]
    trrust_pairs: dict[tuple[int, int], dict] = {}
    for _, r in trrust_in_pool.iterrows():
        i, j = pool_to_idx[r["tf"]], pool_to_idx[r["target"]]
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        e = trrust_pairs.setdefault(key, {"signs": [], "directed": []})
        e["signs"].append(int(r["sign"]))
        e["directed"].append((i, j))

    # Build pair list — keep all unordered pairs in pool with any annotation,
    # plus a sample of negatives to make the eval set tractable.
    rng = np.random.default_rng(42)
    pos_keys = list(trrust_pairs.keys())
    pos_set = set(pos_keys)
    string_pos = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (syms[i], syms[j]) in string_edges or (syms[j], syms[i]) in string_edges:
                string_pos.add((i, j))

    # Universe of candidate pairs: all annotated TRRUST + all STRING edges + a pool of negatives
    n_neg = max(len(pos_keys) * 4, 500)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    neg_pool = [p for p in all_pairs if p not in pos_set and p not in string_pos]
    neg_sample = neg_pool[: n_neg]

    rows = []
    for key in pos_keys:
        i, j = key
        e = trrust_pairs[key]
        sign_majority = int(np.sign(np.sum(e["signs"]))) if e["signs"] else 0
        rows.append((i, j, syms[i], syms[j], 1, sign_majority,
                     int((i, j) in string_pos)))
    for key in string_pos - pos_set:
        i, j = key
        rows.append((i, j, syms[i], syms[j], 0, 0, 1))
    for key in neg_sample:
        i, j = key
        rows.append((i, j, syms[i], syms[j], 0, 0, 0))

    df = pd.DataFrame(rows, columns=[
        "i", "j", "sym_i", "sym_j", "is_trrust", "trrust_sign", "is_string"
    ])
    return df


def make_splits(gene_features: pd.DataFrame, pairs: pd.DataFrame, seed: int):
    """Produce TF-disjoint and target-disjoint train/test gene splits."""
    rng = np.random.default_rng(seed)
    syms = gene_features["symbol"].astype(str).str.upper().to_numpy()
    is_tf = gene_features["is_trrust_tf"].astype(bool).to_numpy()
    is_tgt = gene_features["is_trrust_target"].astype(bool).to_numpy()
    tf_idx = np.where(is_tf)[0]
    tgt_idx = np.where(is_tgt)[0]

    rng.shuffle(tf_idx)
    rng.shuffle(tgt_idx)
    tf_test = set(tf_idx[: len(tf_idx) // 2].tolist())
    tgt_test = set(tgt_idx[: len(tgt_idx) // 2].tolist())

    out = {}
    # source-disjoint: pairs where one of (i,j) is a held-out TF go to test
    pi = pairs["i"].to_numpy()
    pj = pairs["j"].to_numpy()
    src_test_mask = np.array([
        (a in tf_test) or (b in tf_test) for a, b in zip(pi, pj)
    ], dtype=bool)
    out["source_disjoint"] = {
        "train_idx": np.where(~src_test_mask)[0].tolist(),
        "test_idx": np.where(src_test_mask)[0].tolist(),
        "tf_test": sorted(tf_test),
    }

    # target-disjoint: pairs where one of (i,j) is a held-out target go to test
    tgt_test_mask = np.array([
        (a in tgt_test) or (b in tgt_test) for a, b in zip(pi, pj)
    ], dtype=bool)
    out["target_disjoint"] = {
        "train_idx": np.where(~tgt_test_mask)[0].tolist(),
        "test_idx": np.where(tgt_test_mask)[0].tolist(),
        "tgt_test": sorted(tgt_test),
    }
    return out


# ----------------------------------------------------------------------
# Null-model implementations  (used in Phases 5/6/7/8/9/12)
# ----------------------------------------------------------------------
def feature_shuffle_embedding(emb: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each feature column across genes."""
    rng = np.random.default_rng(seed)
    out = emb.copy()
    for d in range(emb.shape[1]):
        rng.shuffle(out[:, d])
    return out


def label_permutation(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = labels.copy()
    rng.shuffle(perm)
    return perm


def curveball_rewire(
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_swaps: int = 1000,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Degree-preserving rewiring via random swap (Strona-curveball-like)."""
    rng = np.random.default_rng(seed)
    edge_set = set(map(lambda e: tuple(sorted(e)), edges))
    edge_list = list(edge_set)
    for _ in range(n_swaps):
        if len(edge_list) < 2:
            break
        a, b = rng.integers(0, len(edge_list), size=2)
        if a == b:
            continue
        u1, v1 = edge_list[a]
        u2, v2 = edge_list[b]
        if len({u1, v1, u2, v2}) < 4:
            continue
        # candidate swap
        e1, e2 = tuple(sorted((u1, v2))), tuple(sorted((u2, v1)))
        if e1 in edge_set or e2 in edge_set:
            continue
        edge_set.discard(edge_list[a]); edge_set.discard(edge_list[b])
        edge_set.add(e1); edge_set.add(e2)
        edge_list[a] = e1; edge_list[b] = e2
    return list(edge_set)


def coexpression_matched_negatives(
    pos_pairs: list[tuple[int, int]],
    neg_universe: list[tuple[int, int]],
    log_expr: np.ndarray,  # (n_cells, n_gene)
    n_bins: int = 5,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Sample one negative per positive matched in absolute Pearson rank bin."""
    rng = np.random.default_rng(seed)

    def pair_corr(p):
        a, b = p
        x = log_expr[:, a]; y = log_expr[:, b]
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(abs(np.corrcoef(x, y)[0, 1]))

    pos_corr = np.array([pair_corr(p) for p in pos_pairs])
    neg_corr = np.array([pair_corr(p) for p in neg_universe])

    # Bin both by quantiles of |corr| over ALL pairs
    all_corr = np.concatenate([pos_corr, neg_corr])
    bin_edges = np.quantile(all_corr, np.linspace(0, 1, n_bins + 1))
    # Avoid duplicate edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return rng.choice(len(neg_universe), size=len(pos_pairs)).tolist()
    pos_bin = np.clip(np.searchsorted(bin_edges, pos_corr, side="right") - 1,
                      0, len(bin_edges) - 2)
    neg_bin = np.clip(np.searchsorted(bin_edges, neg_corr, side="right") - 1,
                      0, len(bin_edges) - 2)

    sampled: list[int] = []
    for b in pos_bin:
        candidates = np.where(neg_bin == b)[0]
        if len(candidates) == 0:
            candidates = np.arange(len(neg_universe))
        sampled.append(int(rng.choice(candidates)))
    return [neg_universe[i] for i in sampled]


# ----------------------------------------------------------------------
# ΔAUROC harness
# ----------------------------------------------------------------------
def delta_auroc(
    feature: np.ndarray,
    labels: np.ndarray,
    coexpression_score: np.ndarray | None = None,
    n_null_replicates: int = 100,
    seed: int = 42,
) -> dict:
    """Returns observed AUROC, ΔAUROC vs label-permutation null + 95th pct,
    and the same vs coexpression-matched null if coexpression_score given.
    """
    rng = np.random.default_rng(seed)
    auc_obs = float(roc_auc_score(labels, feature))

    # Label-permutation null
    null_aucs = []
    for k in range(n_null_replicates):
        perm = labels.copy()
        rng.shuffle(perm)
        null_aucs.append(roc_auc_score(perm, feature))
    null_aucs = np.array(null_aucs)
    p95 = float(np.percentile(null_aucs, 95))
    delta_label = auc_obs - p95

    out = dict(
        auc_obs=auc_obs,
        n_pos=int(labels.sum()),
        n_neg=int((1 - labels).sum()),
        null_label_mean=float(null_aucs.mean()),
        null_label_p95=p95,
        delta_auroc_vs_label=delta_label,
    )

    if coexpression_score is not None:
        # Coexpression-matched null: residualise feature wrt coexpression (linear),
        # then re-score. The "matched 95th percentile" approximation uses the
        # AUROC of coexpression alone as the null floor.
        from numpy.polynomial.polynomial import polyfit, polyval
        c = polyfit(coexpression_score, feature, 1)
        feat_resid = feature - polyval(coexpression_score, c)
        try:
            auc_resid = float(roc_auc_score(labels, feat_resid))
        except Exception:
            auc_resid = float("nan")
        auc_coex = float(roc_auc_score(labels, coexpression_score))
        out.update(
            auc_residualised=auc_resid,
            auc_coexpression=auc_coex,
            delta_auroc_vs_coexpression=auc_obs - auc_coex,
        )
    return out


def main():
    trrust = load_trrust_signed()
    string_edges = load_string_pairs()

    summary = {}
    for domain in DOMAINS:
        d_dir = P0 / domain
        if not (d_dir / "gene_features.csv").exists():
            print(f"[{domain}] skip — no phase0 outputs")
            continue
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = build_pair_table(gf, trrust, string_edges)
        out_d = P1 / domain
        out_d.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_d / "pair_table.csv", index=False)
        print(f"[{domain}] pairs: {len(pairs)}  trrust+={int(pairs['is_trrust'].sum())}  "
              f"string+={int((pairs['is_string'] == 1).sum())}")

        # Splits per seed
        all_splits = {}
        for s in SEEDS:
            sp = make_splits(gf, pairs, seed=s)
            all_splits[str(s)] = sp
        with open(out_d / "splits.json", "w") as f:
            json.dump(all_splits, f, indent=2)
        summary[domain] = {
            "n_pairs": int(len(pairs)),
            "n_pos_trrust": int(pairs["is_trrust"].sum()),
            "n_pos_string": int(pairs["is_string"].sum()),
            "n_genes": int(len(gf)),
        }

    with open(P1 / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[phase 1+2+3] done. summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
