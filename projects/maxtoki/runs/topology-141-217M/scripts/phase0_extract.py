"""Phase 0 — per-layer per-gene embedding extraction across 3 tissue domains.

For each domain (lung, immune, external-lung), runs MaxToki-217M-HF over a
sample of cells, averages per-layer hidden states per gene at the gene's
token position, restricts to the regulatory-annotated gene pool (~320 genes
per the source pipeline), and PCA-reduces to 20 components per layer.

Output (per domain):
  outputs/phase0/{domain}/
    layer_gene_embeddings_full.npy   # [n_layer_states, n_gene, hidden]
    layer_gene_embeddings_pca20.npy  # [n_layer_states, n_gene, 20]
    gene_features.csv                # symbol, ensembl_id, var_idx, ...
    cell_metadata.csv
    run_config.json

For the immune domain we reuse the existing spectral-geometry-217M phase0
extraction (sub-selected to the topology gene pool) to save compute.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[3]
BIOM = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor

# ----------------------------------------------------------------------
# Domain registry
# ----------------------------------------------------------------------
DATA_ROOT = BIOM / "biodyn-work/single_cell_mechinterp/data/raw"
DOMAINS = {
    "lung": {
        "h5ad": DATA_ROOT / "tabula_sapiens_lung.h5ad",
        "n_cells": 1500,
    },
    "immune": {
        "h5ad": DATA_ROOT / "tabula_sapiens_immune.h5ad",
        "n_cells": 1500,
        # Reuse already-extracted embeddings from spectral-geometry-217M
        "reuse_phase0": PROJ / "runs/spectral-geometry-217M/outputs/phase0",
    },
    "external_lung": {
        "h5ad": DATA_ROOT / "krasnow_lung_smartsq2.h5ad",
        "n_cells": 1500,
    },
}

# ----------------------------------------------------------------------
# Reference annotations  → defines the "regulatory-annotated" gene pool
# ----------------------------------------------------------------------
TRRUST_TSV = BIOM / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON = BIOM / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"

OUT_ROOT = PROJ / "runs/topology-141-217M/outputs/phase0"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PCA_DIM = 20
MAX_LEN = int(os.environ.get("PHASE0_MAX_LEN", "2048"))
DEVICE = os.environ.get("PHASE0_DEVICE",
                        "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42


def load_annotation_gene_pool() -> dict:
    """Build the regulatory-annotated symbol pool used to filter HVGs."""
    trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                         names=["tf", "target", "mode", "pmid"])
    trrust_tfs = set(trrust["tf"].str.upper())
    trrust_targets = set(trrust["target"].str.upper())
    with open(STRING_JSON) as f:
        string_data = json.load(f)
    string_symbols = set()
    for pair_list in string_data.values():
        for a, b in pair_list:
            string_symbols.add(a.upper())
            string_symbols.add(b.upper())
    canonical_markers = {
        "CD19", "CD79A", "MS4A1", "PAX5", "BCL6", "PRDM1", "IRF4", "IRF8",
        "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "TBX21", "GATA3", "FOXP3",
        "NCAM1", "KLRD1", "NKG7", "GNLY",
        "CD68", "CD163", "MRC1", "CD14", "LYZ", "S100A8", "S100A9",
        "CLEC9A", "CD1C", "FCER1A", "ELANE", "MPO",
        "EPCAM", "KRT8", "KRT18", "VIM", "PECAM1", "CDH5", "VWF",
        "ACTA2", "MYH11", "COL1A1", "COL3A1", "FN1",
        "SFTPC", "SFTPB", "SCGB1A1", "FOXJ1", "MUC5B", "MUC5AC", "AGER",
    }
    return {
        "trrust_tfs": trrust_tfs,
        "trrust_targets": trrust_targets,
        "trrust_all": trrust_tfs | trrust_targets,
        "string": string_symbols,
        "markers": canonical_markers,
        "any": trrust_tfs | trrust_targets | string_symbols | canonical_markers,
        "trrust_df": trrust,
    }


def select_topology_gene_pool(
    var_ens: np.ndarray,
    var_symbols: np.ndarray,
    expr: np.ndarray,
    tokenizer: MaxTokiTokenizer,
    annot: dict,
    target_n: int = 350,
) -> np.ndarray:
    """Pick ~320–350 genes that are (a) in MaxToki vocab, (b) in regulatory
    annotations, (c) have non-trivial expression in the sampled cells.

    Priority order:
      1. TRRUST TFs ∩ vocab ∩ expressed
      2. TRRUST targets ∩ vocab ∩ expressed
      3. STRING-PPI ∩ vocab ∩ expressed (top-variance fill)
      4. Canonical markers (always include if present)
    """
    vocab = set(tokenizer.gene_token_dict.keys())
    in_vocab = np.array([e in vocab for e in var_ens], dtype=bool)
    syms_upper = np.array([str(s).upper() for s in var_symbols])
    nz_pct = (expr > 0).mean(axis=0)
    var_arr = expr.var(axis=0)
    expressed = (nz_pct >= 0.01) & (var_arr > 0)

    is_tf = np.array([s in annot["trrust_tfs"] for s in syms_upper])
    is_tgt = np.array([s in annot["trrust_targets"] for s in syms_upper])
    is_string = np.array([s in annot["string"] for s in syms_upper])
    is_marker = np.array([s in annot["markers"] for s in syms_upper])

    base = in_vocab & expressed
    chosen = set()

    # 1. TRRUST TFs first
    tf_pool = np.where(base & is_tf)[0]
    tf_sorted = tf_pool[np.argsort(-var_arr[tf_pool])]
    chosen.update(tf_sorted[: min(80, len(tf_sorted))].tolist())

    # 2. TRRUST targets
    tgt_pool = np.where(base & is_tgt & ~is_tf)[0]
    tgt_sorted = tgt_pool[np.argsort(-var_arr[tgt_pool])]
    chosen.update(tgt_sorted[: min(150, len(tgt_sorted))].tolist())

    # 3. STRING-only fillers up to target_n
    str_pool = np.where(base & is_string & ~is_tf & ~is_tgt)[0]
    str_sorted = str_pool[np.argsort(-var_arr[str_pool])]
    needed = max(0, target_n - len(chosen))
    chosen.update(str_sorted[:needed].tolist())

    # 4. Canonical markers (always include)
    mk_pool = np.where(base & is_marker)[0]
    chosen.update(mk_pool.tolist())

    return np.sort(np.fromiter(chosen, dtype=np.int64))


def extract_embeddings(
    domain: str,
    h5ad: Path,
    n_cells: int,
    tokenizer: MaxTokiTokenizer,
    extractor: MaxTokiAttentionExtractor,
    annot: dict,
    out_dir: Path,
):
    rng = np.random.default_rng(SEED)
    print(f"\n[{domain}] reading {h5ad.name} ...")
    adata = ad.read_h5ad(str(h5ad), backed="r")
    n_total = adata.n_obs
    print(f"[{domain}]   adata: {adata.shape}")

    # var arrays
    var_ens = np.array([str(v).split(".")[0] for v in adata.var_names])
    var_symbols = adata.var["feature_name"].to_numpy().astype(str) \
        if "feature_name" in adata.var.columns else var_ens

    # Sample cells
    sample_idx = np.sort(rng.choice(n_total, size=min(n_cells, n_total),
                                    replace=False))
    print(f"[{domain}]   sampled {len(sample_idx)}/{n_total} cells")

    # Pull expression
    X = adata[sample_idx].X
    if sp.issparse(X) or hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    cell_types = adata.obs["cell_type"].iloc[sample_idx].astype(str).to_numpy() \
        if "cell_type" in adata.obs.columns else np.array(["unknown"] * len(sample_idx))
    adata.file.close()

    # Pick the topology gene pool
    gene_idx = select_topology_gene_pool(var_ens, var_symbols, X,
                                         tokenizer, annot, target_n=350)
    print(f"[{domain}]   topology gene pool: {len(gene_idx)} genes")

    gene_ens = var_ens[gene_idx]
    gene_sym = var_symbols[gene_idx]
    gene_token_ids = np.array([tokenizer.gene_token_dict[e] for e in gene_ens],
                              dtype=np.int64)

    # Compute log-normalised expression for diagnostics + later coexpression null
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_log = np.log1p(X / rs * 1e4)

    syms_upper = np.array([s.upper() for s in gene_sym])
    annot_mask = {
        "is_trrust_tf": np.array([s in annot["trrust_tfs"] for s in syms_upper]),
        "is_trrust_target": np.array([s in annot["trrust_targets"] for s in syms_upper]),
        "is_string": np.array([s in annot["string"] for s in syms_upper]),
        "is_marker": np.array([s in annot["markers"] for s in syms_upper]),
    }

    # Build var->gene_pool map
    var_to_pool = -np.ones(len(var_ens), dtype=np.int64)
    for hi, vi in enumerate(gene_idx):
        var_to_pool[vi] = hi
    n_gene = len(gene_idx)

    # Save the (cell × gene) log-normalised matrix for downstream coexpression null
    np.save(out_dir / "log_expression_pool.npy", X_log[:, gene_idx].astype(np.float32))
    pd.DataFrame({
        "sample_idx_local": np.arange(len(sample_idx)),
        "cell_type": cell_types,
    }).to_csv(out_dir / "cell_metadata.csv", index=False)

    n_layers = extractor.n_layers
    hidden = extractor.hidden_size
    n_layer_states = n_layers + 1

    hidden_sum = np.zeros((n_layer_states, n_gene, hidden), dtype=np.float32)
    gene_counts = np.zeros(n_gene, dtype=np.int32)

    var_idx_full, var_tok_full, var_med_full = tokenizer.make_var_mapping(
        var_ens.tolist())
    pool_idx_for_var = var_to_pool[var_idx_full]

    print(f"[{domain}]   extracting {len(sample_idx)} cells × {n_layer_states} layers ...")
    t0 = time.time()
    for ci in tqdm(range(len(sample_idx)), ncols=80):
        cell = tokenizer.tokenize_cell(
            X[ci], var_idx_full, var_tok_full, var_med_full, max_len=MAX_LEN)
        if cell is None:
            continue
        pos = cell.gene_positions
        pool_pos = np.full(len(pos), -1, dtype=np.int64)
        in_v = pos >= 0
        pool_pos[in_v] = pool_idx_for_var[pos[in_v]]
        keep = pool_pos >= 0
        if not keep.any():
            continue
        seq_idx = np.where(keep)[0]
        gids = pool_pos[seq_idx]

        input_ids = torch.from_numpy(cell.token_ids[None, :])
        _, hidden_states = extractor.forward_with_hidden_states(input_ids)
        for li in range(n_layer_states):
            sub = hidden_states[li][0, seq_idx].numpy()  # (k, H)
            np.add.at(hidden_sum[li], gids, sub)
        gene_counts[gids] += 1

    denom = gene_counts.astype(np.float32)
    nonzero = denom > 0
    denom[denom == 0] = np.nan
    full = (hidden_sum / denom[None, :, None]).astype(np.float32)
    full = np.nan_to_num(full, nan=0.0)
    print(f"[{domain}]   forward done in {time.time()-t0:.1f}s; "
          f"genes with ≥1 obs: {int(nonzero.sum())}/{n_gene}")

    # PCA per layer to 20 dims
    pca = np.zeros((n_layer_states, n_gene, PCA_DIM), dtype=np.float32)
    pca_var_explained = np.zeros((n_layer_states, PCA_DIM), dtype=np.float32)
    for li in range(n_layer_states):
        layer = full[li]
        # drop genes that never had observations
        mask = nonzero
        Z = layer[mask]
        Z_c = Z - Z.mean(axis=0, keepdims=True)
        n_components = min(PCA_DIM, Z_c.shape[0], Z_c.shape[1])
        p = PCA(n_components=n_components, random_state=SEED)
        proj = p.fit_transform(Z_c)
        pca[li, mask, :n_components] = proj.astype(np.float32)
        pca_var_explained[li, :n_components] = p.explained_variance_ratio_

    np.save(out_dir / "layer_gene_embeddings_full.npy", full)
    np.save(out_dir / "layer_gene_embeddings_pca20.npy", pca)
    np.save(out_dir / "pca_explained_variance.npy", pca_var_explained)
    np.save(out_dir / "gene_counts.npy", gene_counts)

    gene_features = pd.DataFrame({
        "pool_idx": np.arange(n_gene),
        "symbol": gene_sym,
        "ensembl_id": gene_ens,
        "maxtoki_token_id": gene_token_ids,
        "var_idx": gene_idx,
        "n_observations": gene_counts,
        **annot_mask,
    })
    gene_features.to_csv(out_dir / "gene_features.csv", index=False)

    return {
        "n_cells": int(len(sample_idx)),
        "n_gene_pool": int(n_gene),
        "n_layer_states": int(n_layer_states),
        "hidden_size": int(hidden),
        "n_genes_with_obs": int(nonzero.sum()),
        "pca_dim": PCA_DIM,
        "max_len": MAX_LEN,
        "device": DEVICE,
        "seed": SEED,
    }


def reuse_immune_phase0(
    out_dir: Path,
    tokenizer: MaxTokiTokenizer,
    annot: dict,
):
    """Reuse spectral-geometry-217M phase0 immune extraction by sub-selecting
    its 1500-gene HVG to the topology pool.

    Note: spectral-geometry's gene_features used 1500 HVG over 2000 cells.
    We re-select to keep only genes that meet the topology pool criteria
    (TRRUST/STRING/marker annotated, in MaxToki vocab) and re-PCA per layer.
    """
    src = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
    full = np.load(src / "layer_gene_embeddings.npy")
    counts = np.load(src / "gene_counts.npy")
    gf = pd.read_csv(src / "gene_features.csv")
    print(f"[immune-reuse]   spectral-geometry full embeddings: {full.shape}")

    syms_upper = gf["symbol"].astype(str).str.upper().to_numpy()
    is_tf = np.array([s in annot["trrust_tfs"] for s in syms_upper])
    is_tgt = np.array([s in annot["trrust_targets"] for s in syms_upper])
    is_string = np.array([s in annot["string"] for s in syms_upper])
    is_marker = np.array([s in annot["markers"] for s in syms_upper])
    annotated = is_tf | is_tgt | is_string | is_marker
    has_obs = counts > 0
    keep_mask = annotated & has_obs

    pool_local = np.where(keep_mask)[0]
    # Cap at 350 for parity with the other domains, prioritising TF > target > STRING > marker-only
    if len(pool_local) > 350:
        priority = np.zeros(len(pool_local), dtype=np.int8)
        for i, pi in enumerate(pool_local):
            if is_tf[pi]:
                priority[i] = 0
            elif is_tgt[pi]:
                priority[i] = 1
            elif is_string[pi]:
                priority[i] = 2
            else:
                priority[i] = 3
        pool_local = pool_local[np.argsort(priority)][:350]
        pool_local = np.sort(pool_local)
    n_gene = len(pool_local)
    print(f"[immune-reuse]   topology pool: {n_gene} genes")

    sub_full = full[:, pool_local, :].astype(np.float32)
    sub_counts = counts[pool_local]
    n_layer_states, _, hidden = sub_full.shape

    # PCA per layer
    pca = np.zeros((n_layer_states, n_gene, PCA_DIM), dtype=np.float32)
    pca_var_explained = np.zeros((n_layer_states, PCA_DIM), dtype=np.float32)
    for li in range(n_layer_states):
        layer = sub_full[li]
        Z_c = layer - layer.mean(axis=0, keepdims=True)
        n_components = min(PCA_DIM, n_gene, hidden)
        p = PCA(n_components=n_components, random_state=SEED)
        proj = p.fit_transform(Z_c)
        pca[li, :, :n_components] = proj.astype(np.float32)
        pca_var_explained[li, :n_components] = p.explained_variance_ratio_

    np.save(out_dir / "layer_gene_embeddings_full.npy", sub_full)
    np.save(out_dir / "layer_gene_embeddings_pca20.npy", pca)
    np.save(out_dir / "pca_explained_variance.npy", pca_var_explained)
    np.save(out_dir / "gene_counts.npy", sub_counts)

    sub_gf = gf.iloc[pool_local].copy().reset_index(drop=True)
    sub_gf["pool_idx"] = np.arange(n_gene)
    sub_gf["is_trrust_tf"] = is_tf[pool_local]
    sub_gf["is_trrust_target"] = is_tgt[pool_local]
    sub_gf["is_string"] = is_string[pool_local]
    sub_gf["is_marker"] = is_marker[pool_local]
    sub_gf["n_observations"] = sub_counts
    cols = ["pool_idx", "symbol", "ensembl_id", "maxtoki_token_id", "var_idx",
            "n_observations", "is_trrust_tf", "is_trrust_target", "is_string", "is_marker"]
    sub_gf[cols].to_csv(out_dir / "gene_features.csv", index=False)

    # Approximation: we don't have the original log-expression matrix saved;
    # generate a placeholder zeros file with right shape so that downstream
    # coexpression-null computation can detect the absence and use a sub-sample
    # re-extracted from the immune h5ad on demand.
    # Instead, lazily re-extract log-expression for the chosen genes here:
    print("[immune-reuse]   re-extracting log-expression for coexpression null ...")
    a = ad.read_h5ad(str(DOMAINS["immune"]["h5ad"]), backed="r")
    var_ens = np.array([str(v).split(".")[0] for v in a.var_names])
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(a.n_obs, size=2000, replace=False))
    Xs = a[sample_idx].X
    if sp.issparse(Xs) or hasattr(Xs, "toarray"):
        Xs = Xs.toarray()
    Xs = np.asarray(Xs, dtype=np.float32)
    a.file.close()
    rs = Xs.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_log = np.log1p(Xs / rs * 1e4)
    var_idx_for_pool = sub_gf["var_idx"].to_numpy().astype(np.int64)
    np.save(out_dir / "log_expression_pool.npy",
            X_log[:, var_idx_for_pool].astype(np.float32))
    pd.DataFrame({
        "sample_idx_local": np.arange(len(sample_idx)),
        "cell_type": a.obs["cell_type"].iloc[sample_idx].astype(str).to_numpy()
                     if False else "reused-spectral-geometry",
    }).to_csv(out_dir / "cell_metadata.csv", index=False)

    return {
        "reused": True,
        "n_gene_pool": int(n_gene),
        "n_layer_states": int(n_layer_states),
        "hidden_size": int(hidden),
        "n_genes_with_obs": int((sub_counts > 0).sum()),
        "pca_dim": PCA_DIM,
        "source": str(src),
    }


def main(domain_filter: list[str] | None = None):
    annot = load_annotation_gene_pool()
    print(f"[init] TRRUST TFs: {len(annot['trrust_tfs'])} targets: {len(annot['trrust_targets'])}  "
          f"STRING symbols: {len(annot['string'])}")

    tokenizer = MaxTokiTokenizer()
    print(f"[init] MaxToki vocab: {len(tokenizer.gene_token_dict)} genes")

    summary: dict[str, dict] = {}
    extractor = None
    for name, cfg in DOMAINS.items():
        if domain_filter and name not in domain_filter:
            continue
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if "reuse_phase0" in cfg and cfg["reuse_phase0"].exists():
            info = reuse_immune_phase0(out_dir, tokenizer, annot)
        else:
            if extractor is None:
                print(f"\n[init] loading MaxToki-217M on {DEVICE} ...")
                extractor = MaxTokiAttentionExtractor(device=DEVICE,
                                                     dtype=torch.float32)
                print(f"[init]   L={extractor.n_layers} H={extractor.n_heads} "
                      f"D={extractor.hidden_size}")
            t0 = time.time()
            info = extract_embeddings(name, cfg["h5ad"], cfg["n_cells"],
                                      tokenizer, extractor, annot, out_dir)
            info["wall_seconds"] = float(time.time() - t0)
        summary[name] = info
        with open(out_dir / "run_config.json", "w") as f:
            json.dump(info, f, indent=2)

    # Top-level summary
    with open(OUT_ROOT / "phase0_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n[phase 0 done] domains: " + ", ".join(summary.keys()))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=None)
    args = ap.parse_args()
    main(domain_filter=args.domains)
