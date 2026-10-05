"""Extract per-cell mean-pooled layer-11 residual activations from frozen scGPT whole-human.

Reads a cell x gene count matrix (h5ad: CSR matrix in X, gene symbols in var/index, cell type codes in
obs/clusters, pseudotime in obs/palantir_pseudotime). Each cell is tokenized (expressed genes that are in
the scGPT vocabulary, sorted by expression, first MAX_SEQ kept), run through the frozen model, and the
output of transformer_encoder.layers[11] is mean-pooled over the non-pad gene tokens.

Output: data/embeddings/scgpt_gut.npz
  emb        (N, 512) float32 : per-cell mean-pooled L11 residual
  pseudotime (N,)     float64 : obs/palantir_pseudotime
  clusters   (N,)     str     : cell type label
  cell_idx   (N,)     int     : row in the h5ad

Needs the scGPT source tree (SCGPT_REPO) and models/scgpt_whole_human/best_model.pt (not included).
Run from the package root:
  python code/extraction/extract_scgpt.py [n_cells]
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCGPT_REPO = os.environ.get("SCGPT_REPO", os.path.join(ROOT, "external", "scGPT"))
MODEL_DIR = os.environ.get("SCGPT_MODEL_DIR", os.path.join(ROOT, "models", "scgpt_whole_human"))
SCGPT_CKPT = os.path.join(MODEL_DIR, "best_model.pt")
SCGPT_VOCAB = os.path.join(MODEL_DIR, "vocab.json")
H5AD = os.environ.get("BP_H5AD", os.path.join(ROOT, "data", "raw", "gut_epithelium_counts.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data", "embeddings", "scgpt_gut.npz"))

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11


def load_h5ad():
    with h5py.File(H5AD, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in f["obs"]["__categories"]["clusters"][:]])
        codes = f["obs"]["clusters"][:]
        clusters = cats[codes]
        pt = f["obs"]["palantir_pseudotime"][:].astype(np.float64)
        shape = f["X"].attrs["shape"]
    return gn, clusters, pt, int(shape[0]), int(shape[1])


def sparse_row(fX, i, ncols):
    ip = fX["indptr"]; a, b = int(ip[i]), int(ip[i + 1])
    row = np.zeros(ncols, np.float32)
    row[fX["indices"][a:b]] = fX["data"][a:b]
    return row


def tokenize(expr, gene_names, vocab, pad_id):
    nz = np.where(expr > 0)[0]
    ids, val = [], []
    for idx in nz:
        g = gene_names[idx]
        if g in vocab:
            ids.append(vocab[g]); val.append(expr[idx])
    if not ids:
        return None
    ids = np.array(ids, np.int64); val = np.array(val, np.float32)
    o = np.argsort(-val)[:MAX_SEQ]
    ids, val = ids[o], val[o]
    n = len(ids)
    gi = np.pad(ids, (0, MAX_SEQ - n), constant_values=pad_id)
    gv = np.pad(val, (0, MAX_SEQ - n), constant_values=-2)
    mask = np.zeros(MAX_SEQ, bool); mask[n:] = True
    return gi, gv, mask, n


def build_model(vocab):
    sys.path.insert(0, SCGPT_REPO)
    import scgpt  # noqa
    from scgpt.model.model import TransformerModel
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL, nlayers=N_LAYERS,
                         vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=-2,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False, do_dab=False,
                         use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main(n_cells):
    _dev = os.environ.get("SCGPT_DEVICE")
    dev = torch.device(_dev) if _dev else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    vocab = json.load(open(SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    gn, clusters, pt, N, G = load_h5ad()
    N = min(n_cells, N)
    n_in_vocab = sum(1 for g in gn if g in vocab)
    print(f"scGPT extraction | N={N} cells | {n_in_vocab}/{G} genes in scGPT vocab | dev={dev}", flush=True)
    model = build_model(vocab).to(dev)

    layer_out = {}
    hooks = [model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda mod, inp, out: layer_out.__setitem__("h", out.detach()))]

    embs = np.zeros((N, D_MODEL), np.float32)
    keep = np.zeros(N, bool)
    t0 = time.time()
    with h5py.File(H5AD, "r") as f:
        fX = f["X"]
        for i in range(N):
            expr = sparse_row(fX, i, G)
            tk = tokenize(expr, gn, vocab, pad_id)
            if tk is None:
                continue
            gi, gv, mask, n = tk
            with torch.no_grad():
                layer_out.clear()
                model._encode(src=torch.tensor(gi)[None].to(dev),
                              values=torch.tensor(gv)[None].to(dev),
                              src_key_padding_mask=torch.tensor(mask)[None].to(dev))
                embs[i] = layer_out["h"][0, :n].float().mean(0).cpu().numpy()
            keep[i] = True
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{N} | {(i+1)/(time.time()-t0):.1f} c/s", flush=True)
    for hk in hooks:
        hk.remove()
    embs, pt, clusters = embs[keep], pt[:N][keep], clusters[:N][keep]
    idx = np.where(keep)[0]
    np.savez(OUT, emb=embs, pseudotime=pt, clusters=clusters.astype(str), cell_idx=idx)
    print(f"saved {OUT}  kept={keep.sum()}/{N}  emb norm mean={np.linalg.norm(embs,axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
