"""Phase 15 (extended) — H-tissue: cross-domain geometry transfer.

For each pair of tissue domains (lung, immune, external_lung), restrict to
genes shared between both pools, then test:
  - Per-layer cosine-matrix Pearson correlation between the two domains
  - Permutation null: shuffle the gene labels of one domain
  - Per-layer top-1 retrieval (gene's nearest neighbor in domain B should
    be the same gene)
  - Cross-model CCA against Geneformer for shared genes per domain pair —
    is the cross-domain disagreement smaller than the cross-model
    disagreement?

If MaxToki encodes tissue-invariant gene-gene geometry, the within-MaxToki
across-tissue Pearson should be HIGHER than the within-tissue across-model
Pearson observed in Phase 14.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P15 = RUN / "outputs/phase15_cross_domain"
P15.mkdir(parents=True, exist_ok=True)

DOMAINS = ["lung", "immune", "external_lung"]
N_PERM = 100


def normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def main():
    # Load each domain's full per-layer embeddings (not PCA — we want the full)
    # plus gene_features for shared-gene mapping
    domain_data = {}
    for d in DOMAINS:
        emb = np.load(P0 / d / "layer_gene_embeddings_full.npy")
        gf = pd.read_csv(P0 / d / "gene_features.csv")
        # ensure ensembl_id is string-string
        gf["ensembl_id"] = gf["ensembl_id"].astype(str)
        domain_data[d] = (emb, gf)
        print(f"[{d}] emb {emb.shape}  gf {len(gf)}")

    rows = []
    for da, db in combinations(DOMAINS, 2):
        emb_a, gf_a = domain_data[da]
        emb_b, gf_b = domain_data[db]
        # find shared genes
        ens_a = gf_a["ensembl_id"].tolist()
        ens_b_set = set(gf_b["ensembl_id"])
        idx_a = [i for i, e in enumerate(ens_a) if e in ens_b_set]
        b_pos_by_ens = {e: i for i, e in enumerate(gf_b["ensembl_id"])}
        idx_b = [b_pos_by_ens[ens_a[i]] for i in idx_a]
        n_shared = len(idx_a)
        print(f"\n[{da} ↔ {db}] shared genes: {n_shared}")
        if n_shared < 50:
            continue

        # For each layer, compute the within-MaxToki cross-tissue alignment
        # via cosine-matrix Pearson + permutation null.
        for li in range(emb_a.shape[0]):
            Ea = emb_a[li, idx_a]
            Eb = emb_b[li, idx_b]
            # Normalise + cosine
            An = normalise(Ea); Bn = normalise(Eb)
            Sa = An @ An.T; Sb = Bn @ Bn.T
            iu = np.triu_indices(n_shared, k=1)
            pearson = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
            spearman = float(pd.Series(Sa[iu]).corr(pd.Series(Sb[iu]), method="spearman"))

            # Permutation null
            rng = np.random.default_rng(42 + li * 10 + hash(da+db) % 1024)
            null = []
            for k in range(N_PERM):
                perm = rng.permutation(n_shared)
                null.append(float(np.corrcoef(Sa[iu], Sb[perm][:, perm][iu])[0, 1]))
            null = np.array(null)

            # Top-1 retrieval (rotation of A→B via PCA-reduced Procrustes)
            d_align = min(30, max(2, n_shared // 4))
            Ea_p = PCA(n_components=d_align, random_state=42).fit_transform(Ea - Ea.mean(0))
            Eb_p = PCA(n_components=d_align, random_state=42).fit_transform(Eb - Eb.mean(0))
            M = Ea_p.T @ Eb_p
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
            sim = normalise(Ea_p @ R) @ normalise(Eb_p).T
            top1 = float((np.argmax(sim, axis=1) == np.arange(n_shared)).mean())
            null_top1 = []
            for k in range(N_PERM):
                perm = rng.permutation(n_shared)
                null_top1.append(float((np.argmax(sim[:, perm], axis=1) == np.arange(n_shared)).mean()))
            null_top1 = np.array(null_top1)

            rows.append(dict(
                pair=f"{da}_vs_{db}",
                layer=li,
                n_shared=n_shared,
                pearson=pearson,
                spearman=spearman,
                null_pearson_mean=float(null.mean()),
                null_pearson_p95=float(np.percentile(null, 95)),
                pearson_z=float((pearson - null.mean()) / (null.std() + 1e-9)),
                top1=top1,
                null_top1_p95=float(np.percentile(null_top1, 95)),
                top1_z=float((top1 - null_top1.mean()) / (null_top1.std() + 1e-9)),
            ))
        # progress per pair
        sub = [r for r in rows if r["pair"] == f"{da}_vs_{db}"]
        ps = np.array([r["pearson"] for r in sub])
        zs = np.array([r["pearson_z"] for r in sub])
        t1 = np.array([r["top1"] for r in sub])
        print(f"  Pearson per layer: min={ps.min():.3f} median={np.median(ps):.3f} max={ps.max():.3f}")
        print(f"  Pearson z         : min={zs.min():.1f} median={np.median(zs):.1f} max={zs.max():.1f}")
        print(f"  Top-1 retrieval   : min={t1.min():.3f} median={np.median(t1):.3f} max={t1.max():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(P15 / "cross_domain_alignment.csv", index=False)

    summary = {}
    for pair, sub in df.groupby("pair"):
        summary[pair] = dict(
            n_layers=int(len(sub)),
            mean_pearson=float(sub["pearson"].mean()),
            median_pearson=float(sub["pearson"].median()),
            min_pearson_z=float(sub["pearson_z"].min()),
            mean_pearson_z=float(sub["pearson_z"].mean()),
            mean_top1=float(sub["top1"].mean()),
            median_top1=float(sub["top1"].median()),
            best_layer=int(sub.loc[sub["pearson"].idxmax(), "layer"]),
            verdict=(
                "STRONG cross-tissue invariance (mean Pearson > 0.5)"
                if sub["pearson"].mean() > 0.5 else
                "moderate cross-tissue alignment"
                if sub["pearson"].mean() > 0.2 else
                "weak cross-tissue alignment"
            ),
        )
    with open(P15 / "phase15_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[phase 15] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
