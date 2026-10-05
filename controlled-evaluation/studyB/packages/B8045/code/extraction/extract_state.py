"""Per-cell STATE SE-600M layer-11 embeddings for the cells of an h5ad (reference; not runnable here).

Needs the SE-600M weights with its protein embeddings, a small loader for them (state_loader.py), and the
arc-state package (0.11.1). None of these are included. Hooks transformer_encoder.layers[11] and mean-pools the
residual stream over the non-CLS positions of expressed genes, one vector per cell.

Run:  python code/extraction/extract_state.py controls_for_state.h5ad data/state_se_L11.npz
Out:  emb (N, 2048) float32 : per-cell mean-pooled L11 residual ; cell_idx (N,)
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import anndata as ad
import torch

from state_loader import load_state_se, load_protein_embeds  # noqa: E402

LAYER = int(os.environ.get("STATE_LAYER", "11"))
BATCH = int(os.environ.get("STATE_BATCH", "8"))
DEVICE = os.environ.get("STATE_DEVICE", "cpu")


def main(h5ad, out):
    model, cfg, genes, info = load_state_se(device=DEVICE, dtype=torch.float32)
    pe_dict, gene_list, _ = load_protein_embeds()
    print(f"[load] STATE SE d={info['d_model']} L={info['nlayers']} on {DEVICE}; hooking layer {LAYER}", flush=True)

    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)

    adata = ad.read_h5ad(h5ad)
    n = adata.n_obs
    rows = np.arange(n)
    n_map = sum(1 for g in adata.var_names if g in pe_dict)
    print(f"[data] N={n} | {n_map}/{adata.n_vars} genes in STATE protein-embedding vocab", flush=True)

    adata = inferer._convert_to_csr(adata)
    gene_col = inferer._auto_detect_gene_column(adata)
    cfg.model.batch_size = BATCH
    dl = create_dataloader(cfg, adata=adata, adata_name="controls", shape_dict=None,
                           data_dir=str(Path(h5ad).parent), shuffle=False,
                           protein_embeds=pe_dict, precision=None, gene_column=gene_col)

    captured = {}
    h = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", o.detach()))

    embs = np.zeros((n, info["d_model"]), np.float32)
    filled = np.zeros(n, bool)
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            model._compute_embedding_for_batch(batch)
            res = captured["x"].float().cpu().numpy()          # (B, T, d)
            bs = batch[0].cpu().numpy()                        # (B, T) tokens
            counts = batch[7].cpu().numpy() if batch[7] is not None else None
            idxs = batch[3].cpu().numpy()                      # (B,) cell rows
            B, T = bs.shape
            for i in range(B):
                ci = int(idxs[i])
                valid = np.zeros(T, bool); valid[1:T] = True   # skip CLS
                if counts is not None:
                    valid &= (counts[i, :T] > 0)
                pos = np.nonzero(valid)[0]
                if len(pos) and ci < n:
                    embs[ci] = res[i, pos].mean(0)
                    filled[ci] = True
            if bi % 10 == 0:
                print(f"  batch {bi}: filled={int(filled.sum())}/{n}", flush=True)
    h.remove()
    np.savez_compressed(out, emb=embs[filled], cell_idx=rows[filled])
    print(f"saved {out}  kept={int(filled.sum())}/{n}  emb norm mean={np.linalg.norm(embs[filled], axis=1).mean():.2f}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
