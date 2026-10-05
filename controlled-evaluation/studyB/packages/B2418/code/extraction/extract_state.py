"""Extract per-cell mean-pooled layer-11 residual activations from STATE SE-600M.

Builds the model with state_loader.py (safetensors weights) and uses STATE's own dataloader for the
cell-sentence construction. Hooks transformer_encoder.layers[11] and mean-pools over the non-CLS positions
of expressed genes in each cell. Input: raw counts in X, gene symbols as var names.

Output: data/embeddings/state_gut.npz
  emb (N,2048) f32 : per-cell mean-pooled L11 residual ; pseudotime ; clusters ; cell_idx

Needs arc-state 0.11.1 and the SE-600M files (model.safetensors, protein_embeddings.pt) in
models/state_se600m/ (only config.yaml is included). Run from the package root:
  python code/extraction/extract_state.py [n_cells]
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import anndata as ad
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from state_loader import load_state_se, load_protein_embeds  # noqa: E402

H5AD = os.environ.get("BP_H5AD", str(ROOT / "data" / "raw" / "gut_epithelium_counts.h5ad"))
OUT = os.environ.get("BP_OUT", str(ROOT / "data" / "embeddings" / "state_gut.npz"))
LAYER = int(os.environ.get("STATE_LAYER", "11"))
BATCH = int(os.environ.get("STATE_BATCH", "8"))
DEVICE = os.environ.get("STATE_DEVICE", "mps")


def main(n_cells):
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, cfg, genes, info = load_state_se(device=dev, dtype=torch.float32)
    pe_dict, gene_list, _ = load_protein_embeds()
    print(f"[load] STATE SE d={info['d_model']} L={info['nlayers']} on {dev}; hooking layer {LAYER}", flush=True)

    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)

    adata = ad.read_h5ad(H5AD)
    idx_file = os.environ.get("BP_IDX")
    if idx_file:
        rows = np.load(idx_file)
        adata = adata[rows].copy()
        print(f"[subset] {len(rows)} cells from {idx_file}", flush=True)
    else:
        n0 = min(n_cells, adata.n_obs)
        rows = np.arange(n0)
        adata = adata[:n0].copy()
    n = adata.n_obs
    # align labels
    cats = adata.obs["clusters"].astype(str).values.copy() if "clusters" in adata.obs else None
    if cats is None:  # old-format categorical may need cat codes
        import h5py
        with h5py.File(H5AD, "r") as f:
            cc = np.array([x.decode() if isinstance(x, bytes) else x for x in f["obs"]["__categories"]["clusters"][:]])
            cats = cc[f["obs"]["clusters"][:n]]
    pt = adata.obs["palantir_pseudotime"].astype(float).values.copy()

    n_map = sum(1 for g in adata.var_names if g in pe_dict)
    print(f"[data] N={n} | {n_map}/{adata.n_vars} genes in STATE protein-embedding vocab", flush=True)

    adata = inferer._convert_to_csr(adata)
    gene_col = inferer._auto_detect_gene_column(adata)
    cfg.model.batch_size = BATCH
    dl = create_dataloader(cfg, adata=adata, adata_name="gut", shape_dict=None,
                           data_dir=str(Path(H5AD).parent), shuffle=False,
                           protein_embeds=pe_dict, precision=None, gene_column=gene_col)

    captured = {}
    h = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", o.detach()))

    embs = np.zeros((n, info["d_model"]), np.float32)
    filled = np.zeros(n, bool)
    # resume: pre-fill already-saved cells (matched by original row) so a restart never loses them
    row_to_local = {int(r): i for i, r in enumerate(np.asarray(rows))}
    if os.path.exists(OUT):
        try:
            z = np.load(OUT, allow_pickle=True)
            if z["emb"].shape[1] == info["d_model"]:
                for e, ci in zip(z["emb"], z["cell_idx"]):
                    loc = row_to_local.get(int(ci))
                    if loc is not None:
                        embs[loc] = e.astype(np.float32); filled[loc] = True
                print(f"[resume] pre-filled {int(filled.sum())} cells from {OUT}", flush=True)
        except Exception as ex:
            print(f"[resume] ignored existing ({ex})", flush=True)

    def flush():
        keep = filled
        if keep.sum() == 0:
            return
        np.savez(OUT, emb=embs[keep], pseudotime=pt[keep], clusters=cats[keep].astype(str),
                 cell_idx=np.asarray(rows)[keep])

    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            model._compute_embedding_for_batch(batch)
            res = captured["x"].float().cpu().numpy()          # (B,R,d)
            bs = batch[0].cpu().numpy()                        # (B,T) tokens
            counts = batch[7].cpu().numpy() if batch[7] is not None else None
            idxs = batch[3].cpu().numpy()                      # (B,) cell rows
            B, T = bs.shape
            for i in range(B):
                ci = int(idxs[i])
                if ci < n and filled[ci]:
                    continue                                   # already have this cell (resume)
                valid = np.zeros(T, bool); valid[1:T] = True   # skip CLS
                if counts is not None:
                    valid &= (counts[i, :T] > 0)
                pos = np.nonzero(valid)[0]
                if len(pos) and ci < n:
                    embs[ci] = res[i, pos].mean(0)
                    filled[ci] = True
            seen += B
            if bi % 10 == 0:
                print(f"  batch {bi}: cells={seen}/{n} filled={int(filled.sum())}", flush=True)
            if bi % 20 == 0 and bi > 0:
                flush(); print(f"  [ckpt] {int(filled.sum())} cells written", flush=True)
    h.remove()
    flush()
    print(f"saved {OUT}  kept={int(filled.sum())}/{n}  emb norm mean={np.linalg.norm(embs[filled],axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
