"""Phase 4 (extended) — H17/H20/H24 cross-model alignment between
MaxToki-217M and scGPT (whole-human static gene embeddings, d_model=512).

Triangulates the existing Phase 14 result (vs Geneformer V2-316M). Per the
source-paper Layer-1 hierarchy claim, cross-model agreement is the most robust
property — verifying it against a second independent foundation model
strengthens (or weakens) the partial-replication finding from Phase 14.

scGPT static embeddings come from the cached extract at
  biodyn-work/subproject_53_scgpt_gpl_replication/embeddings/
  scgpt_whole_human_gene_embeddings.pt

Vocabulary is HUGO gene symbols (60,697 genes). Mapping: MaxToki
gene_features.csv → 'symbol' column, then case-insensitive match into scGPT
vocab.

Tests (mirror Phase 14):
  H24 — CCA on per-gene (MaxToki static, scGPT normed) embeddings
  H17 — pairwise gene-gene cosine-similarity Spearman + Pearson
  H20 — orthogonal Procrustes alignment + top-1 retrieval

Permutation null (50 draws): shuffle scGPT row identities.

Outputs to outputs/phase4_scgpt/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P4 = RUN / "outputs/phase4_scgpt"
P4.mkdir(parents=True, exist_ok=True)

MAXTOKI_DIR = PROJ / "setup/MaxToki-217M-HF"
SCGPT_PT = Path(
    "<DATA_ROOT>/biodyn-work/"
    "subproject_53_scgpt_gpl_replication/embeddings/"
    "scgpt_whole_human_gene_embeddings.pt"
)

DOMAINS = ["lung", "immune", "external_lung"]
N_PERM = 50
TOP_K_CCA = 10
PCA_DIM_FOR_ALIGN = 30


def load_safetensor_key(path: Path, key_substrs: list[str]) -> np.ndarray:
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if all(s in k for s in key_substrs):
                return f.get_tensor(k).cpu().float().numpy()
    raise KeyError(f"no key with {key_substrs} in {path}")


def normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cca_score(A: np.ndarray, B: np.ndarray, k: int = 10) -> dict:
    n = min(A.shape[0], B.shape[0])
    A = A[:n]; B = B[:n]
    k = min(k, A.shape[1], B.shape[1], n - 1)
    if k < 1:
        return {"mean_canon_r": float("nan"), "top_canon_r": float("nan"),
                "all_canon_r": []}
    cca = CCA(n_components=k, max_iter=200)
    try:
        Ac, Bc = cca.fit_transform(A, B)
    except Exception:
        return {"mean_canon_r": float("nan"), "top_canon_r": float("nan"),
                "all_canon_r": []}
    rs = []
    for i in range(k):
        a, b = Ac[:, i], Bc[:, i]
        if a.std() == 0 or b.std() == 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    if not rs:
        return {"mean_canon_r": float("nan"), "top_canon_r": float("nan"),
                "all_canon_r": []}
    return {"mean_canon_r": float(np.mean(rs)),
            "top_canon_r": float(rs[0]),
            "all_canon_r": rs}


def gene_level_top1_retrieval(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0])
    A = A[:n].astype(np.float64); B = B[:n].astype(np.float64)
    d = min(A.shape[1], B.shape[1])
    A = A[:, :d]; B = B[:, :d]
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    A_rot = A @ R
    sim = normalise(A_rot) @ normalise(B).T
    return float((np.argmax(sim, axis=1) == np.arange(n)).mean())


def main():
    print("=" * 72)
    print("PHASE 4 — cross-model alignment (MaxToki-217M ↔ scGPT)")
    print("=" * 72)

    print("\n[1] Loading scGPT static embeddings...")
    scgpt = torch.load(SCGPT_PT, map_location="cpu", weights_only=False)
    scgpt_vocab = [s.upper() for s in scgpt["vocab"]]
    scgpt_emb = scgpt["normed_embeddings"].cpu().float().numpy()
    sym_to_scgpt_idx = {s: i for i, s in enumerate(scgpt_vocab)}
    print(f"  scGPT: {scgpt_emb.shape}  vocab len {len(scgpt_vocab)}")

    print("\n[2] Loading MaxToki static embeddings...")
    mt_static_full = load_safetensor_key(
        MAXTOKI_DIR / "model.safetensors", ["embed_tokens"],
    )
    print(f"  MaxToki static: {mt_static_full.shape}")

    rows = []
    for domain in DOMAINS:
        gf = pd.read_csv(P0 / domain / "gene_features.csv")
        syms = gf["symbol"].astype(str).str.upper().tolist()
        mt_tids = gf["maxtoki_token_id"].astype(int).to_numpy()
        sc_idx = np.array(
            [sym_to_scgpt_idx.get(s, -1) for s in syms], dtype=np.int64
        )
        valid = (sc_idx >= 0) & (mt_tids >= 0) & (mt_tids < mt_static_full.shape[0])
        n = int(valid.sum())
        if n < 30:
            print(f"[{domain}] too few shared genes ({n}); skip")
            continue
        sc_vec_full = scgpt_emb[sc_idx[valid]]
        mt_vec_full = mt_static_full[mt_tids[valid]]
        d_align = min(PCA_DIM_FOR_ALIGN, max(2, n // 4))
        sc_vec = PCA(n_components=d_align, random_state=42).fit_transform(
            sc_vec_full - sc_vec_full.mean(0))
        mt_vec = PCA(n_components=d_align, random_state=42).fit_transform(
            mt_vec_full - mt_vec_full.mean(0))
        print(f"\n[{domain}] shared symbols: {n}/{len(gf)}  PCA-aligned dim: {d_align}")

        cca_out = cca_score(mt_vec, sc_vec, k=min(TOP_K_CCA, d_align))

        # Pairwise similarity comparison on FULL static embeddings
        Sa = normalise(mt_vec_full) @ normalise(mt_vec_full).T
        Sb = normalise(sc_vec_full) @ normalise(sc_vec_full).T
        iu = np.triu_indices(n, k=1)
        sa_off = Sa[iu]; sb_off = Sb[iu]
        pearson = float(np.corrcoef(sa_off, sb_off)[0, 1])
        spearman = float(pd.Series(sa_off).corr(pd.Series(sb_off), method="spearman"))

        top1 = gene_level_top1_retrieval(mt_vec, sc_vec)

        # Permutation null on top-1 + pearson
        d_min = min(mt_vec.shape[1], sc_vec.shape[1])
        Aproc = mt_vec[:, :d_min].astype(np.float64)
        Bproc = sc_vec[:, :d_min].astype(np.float64)
        M = Aproc.T @ Bproc
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        sim_full = normalise(Aproc @ R) @ normalise(Bproc).T

        rng = np.random.default_rng(42 + hash(domain) % 1024)
        null_top1, null_pearson = [], []
        for _ in range(N_PERM):
            perm = rng.permutation(n)
            shuffled_sim = sim_full[:, perm]
            null_top1.append(float((np.argmax(shuffled_sim, axis=1) == np.arange(n)).mean()))
            null_pearson.append(float(np.corrcoef(sa_off, Sb[perm][:, perm][iu])[0, 1]))
        null_top1 = np.array(null_top1); null_pearson = np.array(null_pearson)

        rows.append(dict(
            domain=domain, tap="static", n_genes=n,
            cca_mean_r=cca_out["mean_canon_r"],
            cca_top_r=cca_out["top_canon_r"],
            pairwise_pearson=pearson,
            pairwise_spearman=spearman,
            null_pearson_mean=float(null_pearson.mean()),
            null_pearson_std=float(null_pearson.std()),
            null_pearson_p95=float(np.percentile(null_pearson, 95)),
            pearson_z=float((pearson - null_pearson.mean()) / (null_pearson.std() + 1e-9)),
            top1_retrieval=top1,
            null_top1_mean=float(null_top1.mean()),
            null_top1_p95=float(np.percentile(null_top1, 95)),
            top1_z=float((top1 - null_top1.mean()) / (null_top1.std() + 1e-9)),
        ))
        print(f"  CCA mean_r={cca_out['mean_canon_r']:.3f}  "
              f"Spearman={spearman:.3f}  Pearson={pearson:.3f} (z={rows[-1]['pearson_z']:.1f})")
        print(f"  Top-1 retrieval={top1*100:.1f}%  (null p95 {np.percentile(null_top1,95)*100:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(P4 / "cross_model_alignment_scgpt.csv", index=False)
    summary = {}
    for domain, sub in df.groupby("domain"):
        s = sub.iloc[0]
        summary[domain] = dict(
            n_genes=int(s["n_genes"]),
            cca_mean_r=float(s["cca_mean_r"]),
            pairwise_pearson=float(s["pairwise_pearson"]),
            pairwise_spearman=float(s["pairwise_spearman"]),
            pairwise_pearson_z=float(s["pearson_z"]),
            top1_retrieval=float(s["top1_retrieval"]),
            top1_retrieval_z=float(s["top1_z"]),
        )
    # Triangulation table comparing scGPT vs Geneformer Phase 14 result
    p14_csv = RUN / "outputs/phase14_cross_model/cross_model_alignment.csv"
    if p14_csv.exists():
        p14 = pd.read_csv(p14_csv)
        triangulation = {}
        for domain, sub in p14.groupby("domain"):
            s = sub.iloc[0]
            scgpt_row = summary.get(domain, {})
            triangulation[domain] = dict(
                geneformer=dict(
                    cca_mean_r=float(s.get("cca_mean_r", float("nan"))),
                    pairwise_pearson=float(s.get("pairwise_pearson", float("nan"))),
                    top1_retrieval=float(s.get("top1_retrieval", float("nan"))),
                ),
                scgpt=dict(
                    cca_mean_r=scgpt_row.get("cca_mean_r"),
                    pairwise_pearson=scgpt_row.get("pairwise_pearson"),
                    top1_retrieval=scgpt_row.get("top1_retrieval"),
                ),
            )
        summary["triangulation_vs_phase14"] = triangulation

    with open(P4 / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[phase 4 vs scGPT] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
