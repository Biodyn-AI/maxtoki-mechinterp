"""MaxToki contextual representation runtime for the longevity-mechinterp pipeline.

Mirrors the interface of `ScGPTRuntime` / `GeneformerRuntime` in
repos/longevity-mechinterp/implementation/scripts/run_stage1_longevity_mechinterp.py:
``extract_representations(adata, max_genes_per_cell, batch_size, layer_indices)``
returns ``dict[name -> ndarray(n_cells, hidden_dim)]`` of per-cell embeddings.

Per-cell pooling: mean over the non-special token positions of the requested
layer's residual-stream output. Special tokens (BOS/EOS) are excluded by
selecting positions where ``TokenizedCell.gene_positions >= 0``.

Layer naming convention: ``maxtoki217m_layer_NN`` where NN is the index into
``hidden_states`` returned by the HF model (0 = post-embedding,
N = post-block-N-1). MaxToki-217M has 11 blocks, so valid NN ∈ [0, 11].
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETUP_DIR = PROJECT_ROOT / "setup"
sys.path.insert(0, str(SETUP_DIR))

from maxtoki_adapter import (  # type: ignore  # noqa: E402
    MaxTokiAttentionExtractor,
    MaxTokiTokenizer,
)


@dataclass
class _CellExtractionStats:
    n_cells_in: int
    n_cells_out: int
    n_skipped_zero_expr: int
    mean_seq_len: float
    mean_fwd_seconds: float
    layer_indices: tuple


class MaxTokiContextualRuntime:
    """Per-cell mean-pooled residual-stream extractor for MaxToki HF safetensors.

    Usage:
        runtime = MaxTokiContextualRuntime(
            model_dir=SETUP_DIR / "MaxToki-217M-HF",
            tokenizer=MaxTokiTokenizer(),
        )
        reps, stats = runtime.extract_representations(
            adata, max_genes_per_cell=1024, layer_indices=(5,)
        )
    """

    def __init__(
        self,
        model_dir: Path,
        tokenizer: MaxTokiTokenizer | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        model_tag: str = "maxtoki217m",
    ):
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer or MaxTokiTokenizer()
        self.extractor = MaxTokiAttentionExtractor(
            model_dir=self.model_dir, dtype=dtype, device=device
        )
        self.n_layers = self.extractor.n_layers
        self.hidden_size = self.extractor.hidden_size
        self.model_tag = model_tag

    # ---- gene-mapping helpers ------------------------------------------------
    def _build_var_mapping(self, adata: ad.AnnData):
        """Return (var_indices, token_ids, medians) over `adata.var_names`."""
        var_ensembl = list(adata.var_names)
        return self.tokenizer.make_var_mapping(var_ensembl)

    # ---- main entry ----------------------------------------------------------
    @torch.no_grad()
    def extract_representations(
        self,
        adata: ad.AnnData,
        max_genes_per_cell: int = 1024,
        layer_indices: Sequence[int] = (5,),
        batch_size: int = 1,  # MPS does not benefit from padding-batching for eager attention
        progress_every: int = 200,
        cache_path: Path | None = None,
    ) -> tuple[Dict[str, np.ndarray], _CellExtractionStats]:
        """Extract per-cell mean-pooled residual-stream embeddings.

        Returns:
            reps: dict {f"{model_tag}_layer_{NN:02d}": ndarray(n_cells, hidden_dim)}.
                  Cells where tokenization fails (all-zero expression after
                  Ensembl-to-MaxToki-vocab filter) are filled with NaN rows; the
                  caller is expected to drop them.
            stats: extraction stats (counts, timings, layer indices).
        """
        n_cells = int(adata.n_obs)
        layer_idx_tuple = tuple(int(li) for li in layer_indices)
        for li in layer_idx_tuple:
            if li < 0 or li > self.n_layers:
                raise ValueError(
                    f"layer_index {li} outside valid range [0, {self.n_layers}] "
                    f"for MaxToki-{self.model_tag}"
                )

        # Try cache first.
        if cache_path is not None and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if (
                    cached.shape[0] == n_cells
                    and cached.shape[1] == len(layer_idx_tuple)
                    and cached.shape[2] == self.hidden_size
                ):
                    print(f"[runtime] cache hit: {cache_path} shape {cached.shape}")
                    reps = {
                        f"{self.model_tag}_layer_{li:02d}": np.ascontiguousarray(
                            cached[:, i].astype(np.float32)
                        )
                        for i, li in enumerate(layer_idx_tuple)
                    }
                    stats = _CellExtractionStats(
                        n_cells_in=n_cells,
                        n_cells_out=int(np.sum(np.isfinite(cached[:, 0, 0]))),
                        n_skipped_zero_expr=int(np.sum(~np.isfinite(cached[:, 0, 0]))),
                        mean_seq_len=float("nan"),
                        mean_fwd_seconds=float("nan"),
                        layer_indices=layer_idx_tuple,
                    )
                    return reps, stats
            except Exception as e:
                print(f"[runtime] cache load failed ({e!r}), re-extracting")

        var_indices, token_ids, medians = self._build_var_mapping(adata)
        if var_indices.size == 0:
            raise ValueError(
                "No genes from adata.var_names mapped to MaxToki vocab — check var "
                "are Ensembl IDs"
            )
        print(
            f"[runtime] mapped {var_indices.size}/{adata.n_vars} adata genes onto "
            f"MaxToki vocab"
        )

        # Materialize expression matrix once; tabula sapiens immune subset is 20k×60k
        # sparse, so this is < 200 MB after slicing to mapped genes.
        X = adata.X
        if sp.issparse(X):
            X_dense = np.asarray(X.toarray(), dtype=np.float32)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
        # Filter to mapped genes only — make tokenize_cell index work
        # against the same column ordering.

        out = np.full(
            (n_cells, len(layer_idx_tuple), self.hidden_size),
            np.nan,
            dtype=np.float32,
        )

        per_cell_times: list[float] = []
        per_cell_seq_lens: list[int] = []
        n_skipped = 0

        t_phase = time.time()
        for ci in range(n_cells):
            t0 = time.time()
            cell = self.tokenizer.tokenize_cell(
                X_dense[ci], var_indices, token_ids, medians, max_len=max_genes_per_cell
            )
            if cell is None:
                n_skipped += 1
                continue

            input_ids = torch.from_numpy(cell.token_ids[None, :])
            _, hidden = self.extractor.forward_with_hidden_states(input_ids)

            # Gene-token positions (exclude BOS/EOS)
            gene_mask = cell.gene_positions >= 0
            if not gene_mask.any():
                n_skipped += 1
                continue
            pos_idx = np.where(gene_mask)[0]

            for slot, li in enumerate(layer_idx_tuple):
                hl = hidden[li][0].numpy()  # (seq, H)
                pooled = hl[pos_idx].mean(axis=0)  # (H,)
                out[ci, slot] = pooled

            per_cell_times.append(time.time() - t0)
            per_cell_seq_lens.append(int(gene_mask.sum()))

            if (ci + 1) % progress_every == 0 or (ci + 1) == n_cells:
                elapsed = time.time() - t_phase
                eta = (elapsed / max(ci + 1, 1)) * (n_cells - ci - 1)
                print(
                    f"[runtime] {ci+1}/{n_cells}  mean_fwd={np.mean(per_cell_times[-progress_every:]):.2f}s  "
                    f"seq_len={np.mean(per_cell_seq_lens[-progress_every:]):.0f}  "
                    f"skipped={n_skipped}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, out)
            print(f"[runtime] cached extraction to {cache_path}")

        reps = {
            f"{self.model_tag}_layer_{li:02d}": np.ascontiguousarray(out[:, slot])
            for slot, li in enumerate(layer_idx_tuple)
        }
        stats = _CellExtractionStats(
            n_cells_in=n_cells,
            n_cells_out=n_cells - n_skipped,
            n_skipped_zero_expr=n_skipped,
            mean_seq_len=float(np.mean(per_cell_seq_lens)) if per_cell_seq_lens else float("nan"),
            mean_fwd_seconds=float(np.mean(per_cell_times)) if per_cell_times else float("nan"),
            layer_indices=layer_idx_tuple,
        )
        return reps, stats


def smoke_test(n_cells: int = 10):
    import json

    adata = ad.read_h5ad(
        "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
        "data/raw/tabula_sapiens_immune_subset_20000.h5ad",
        backed="r",
    ).to_memory()
    sub = adata[:n_cells].copy()
    runtime = MaxTokiContextualRuntime(
        model_dir=SETUP_DIR / "MaxToki-217M-HF",
    )
    reps, stats = runtime.extract_representations(
        sub, max_genes_per_cell=1024, layer_indices=(5,)
    )
    arr = next(iter(reps.values()))
    nz = np.isfinite(arr[:, 0])
    info = {
        "n_layers": runtime.n_layers,
        "hidden_size": runtime.hidden_size,
        "rep_shape": list(arr.shape),
        "n_finite_rows": int(nz.sum()),
        "rep_mean_finite": float(arr[nz].mean()) if nz.any() else None,
        "rep_std_finite": float(arr[nz].std()) if nz.any() else None,
        "stats": {
            "n_cells_in": stats.n_cells_in,
            "n_cells_out": stats.n_cells_out,
            "n_skipped_zero_expr": stats.n_skipped_zero_expr,
            "mean_seq_len": stats.mean_seq_len,
            "mean_fwd_seconds": stats.mean_fwd_seconds,
        },
    }
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    smoke_test()
