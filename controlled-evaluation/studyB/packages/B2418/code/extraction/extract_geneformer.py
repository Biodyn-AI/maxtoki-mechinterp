"""Extract per-cell mean-pooled layer-11 residual activations from Geneformer V2-316M.

Rank-value tokenization: gene symbols -> Ensembl id (gene_name_id_dict) -> Geneformer token. Each cell is
normalized log1p(counts / rowsum * 1e4), genes are ranked by expression / gene median (descending), and the
sequence is [CLS] genes... [EOS]. The output of transformer layer 11 (hidden_states[12]) is mean-pooled over
the gene positions.

Output: data/embeddings/geneformer_gut.npz
  emb (N,1152) f32 : per-cell mean-pooled L11 residual ; pseudotime ; clusters ; cell_idx

Needs the Geneformer token / median / name-id dictionaries in models/geneformer_v2_316m/ (not included) and
the ctheodoris/Geneformer weights from the Hugging Face hub. Run from the package root:
  python code/extraction/extract_geneformer.py [n_cells]
"""
import os, sys, json, time, pickle, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_DIR = os.environ.get("GF_TOK_DIR", os.path.join(ROOT, "models", "geneformer_v2_316m"))
H5AD = os.environ.get("BP_H5AD", os.path.join(ROOT, "data", "raw", "gut_epithelium_counts.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data", "embeddings", "geneformer_gut.npz"))
MODEL_NAME, MODEL_SUBFOLDER = "ctheodoris/Geneformer", "Geneformer-V2-316M"
HIDDEN_DIM, MAX_SEQ_LEN, LAYER = 1152, 2048, 11


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


def tokenize(expr_norm, var_idx, token_ids, medians):
    """expr_norm already log1p-normalized. Return token seq [CLS,...,EOS] or None."""
    e = expr_norm[var_idx]
    nz = e > 0
    if nz.sum() == 0:
        return None
    e_nz, t_nz, m_nz = e[nz], token_ids[nz], medians[nz]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.nan_to_num(e_nz / m_nz, nan=0.0, posinf=0.0)
    order = np.argsort(-norm)
    ranked = t_nz[order][:MAX_SEQ_LEN - 2]
    return np.concatenate([[2], ranked, [3]]).astype(np.int64)


def main(n_cells):
    from transformers import BertForMaskedLM
    _gd = os.environ.get("GF_DEVICE")   # override: 'cpu' avoids MPS memory growth on long loops
    if _gd:
        dev = torch.device(_gd)
    else:
        dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    token_dict = pickle.load(open(os.path.join(TOK_DIR, "token_dictionary_gc104M.pkl"), "rb"))
    median_dict = pickle.load(open(os.path.join(TOK_DIR, "gene_median_dictionary_gc104M.pkl"), "rb"))
    name_id = pickle.load(open(os.path.join(TOK_DIR, "gene_name_id_dict_gc104M.pkl"), "rb"))

    gn, clusters, pt, N, G = load_h5ad()
    idx_file = os.environ.get("BP_IDX")
    if idx_file:
        rows = np.load(idx_file)
        print(f"[subset] {len(rows)} cells from {idx_file}", flush=True)
    else:
        rows = np.arange(min(n_cells, N))
    # symbol -> ensembl -> token/median
    var_idx, tok_ids, meds = [], [], []
    for i in range(G):
        ens = name_id.get(gn[i])
        if ens and ens in token_dict:
            var_idx.append(i); tok_ids.append(token_dict[ens]); meds.append(median_dict.get(ens, 1.0))
    var_idx = np.array(var_idx); tok_ids = np.array(tok_ids); meds = np.array(meds)
    M = len(rows)
    print(f"Geneformer | M={M} cells | {len(var_idx)}/{G} genes mapped to tokens | dev={dev}", flush=True)

    # output_hidden_states=False + a single-layer hook keeps memory low (avoids storing all 19 hidden
    # states) -> much lighter on MPS, which mitigates the long-loop memory growth.
    attn_impl = os.environ.get("GF_ATTN", "sdpa")   # sdpa: identical hidden states, faster on MPS than eager
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, subfolder=MODEL_SUBFOLDER,
                                            output_hidden_states=False, output_attentions=False,
                                            attn_implementation=attn_impl).to(dev).eval()
    cap = {}
    # encoder.layer[LAYER] output == hidden_states[LAYER+1] == output of transformer layer LAYER
    model.bert.encoder.layer[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0].detach()))

    BATCH = int(os.environ.get("GF_BATCH", "16"))
    CKPT = int(os.environ.get("CKPT", "250"))
    acc = {}   # row -> emb (float32)  (resumable, keyed by original cell row)
    if os.path.exists(OUT):
        try:
            z = np.load(OUT, allow_pickle=True)
            for e, ci in zip(z["emb"], z["cell_idx"]):
                acc[int(ci)] = e.astype(np.float32)
            print(f"[resume] loaded {len(acc)} cells from existing {OUT}", flush=True)
        except Exception as ex:
            print(f"[resume] could not load checkpoint ({ex}); starting fresh", flush=True)

    def flush():
        idx = np.array(sorted(acc), dtype=np.int64)
        if len(idx) == 0:
            return
        np.savez(OUT, emb=np.stack([acc[i] for i in idx]),
                 pseudotime=pt[idx], clusters=clusters[idx].astype(str), cell_idx=idx)

    # tokenize all pending cells first (cheap), then batch by similar length to minimize padding
    pending = []  # (row, tokens)
    with h5py.File(H5AD, "r") as f:
        fX = f["X"]
        for i in rows:
            i = int(i)
            if i in acc:
                continue
            expr = sparse_row(fX, i, G)
            rs = expr.sum(); rs = rs if rs > 0 else 1.0
            tk = tokenize(np.log1p(expr / rs * 1e4), var_idx, tok_ids, meds)
            if tk is not None:
                pending.append((i, tk))
    pending.sort(key=lambda x: len(x[1]))
    print(f"[tokenized] {len(pending)} pending cells; batching {BATCH} (device {dev})", flush=True)

    t0 = time.time(); n_new = 0
    for b0 in range(0, len(pending), BATCH):
        chunk = pending[b0:b0 + BATCH]
        maxlen = max(len(tk) for _, tk in chunk)
        B = len(chunk)
        ii = np.zeros((B, maxlen), np.int64)      # pad token = 0
        am = np.zeros((B, maxlen), np.int64)
        for r, (_, tk) in enumerate(chunk):
            ii[r, :len(tk)] = tk; am[r, :len(tk)] = 1
        input_ids = torch.from_numpy(ii).to(dev)
        attn = torch.from_numpy(am).to(dev)
        gmask = torch.from_numpy((ii >= 4).astype(np.float32)).to(dev)   # gene tokens: not pad/mask/CLS/EOS
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)
            h = cap["h"].float()                                          # (B, maxlen, 1152)
            gm = gmask.unsqueeze(-1)
            pooled = (h * gm).sum(1) / gm.sum(1).clamp(min=1.0)           # (B, 1152)
            pooled = pooled.cpu().numpy()
        for r, (row_i, _) in enumerate(chunk):
            acc[row_i] = pooled[r]
        n_new += B
        del input_ids, attn, gmask
        if dev.type == "mps":
            torch.mps.empty_cache()
        if (b0 // BATCH) % 5 == 0:
            print(f"  {len(acc)}/{M} ({n_new} new) | {n_new/(time.time()-t0):.1f} c/s | maxlen={maxlen}", flush=True)
        if n_new % CKPT < BATCH:
            flush()
    flush()
    embs = np.stack([acc[i] for i in sorted(acc)]) if acc else np.zeros((0, HIDDEN_DIM), np.float32)
    print(f"saved {OUT}  kept={len(acc)}/{M}  emb norm mean={np.linalg.norm(embs,axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
