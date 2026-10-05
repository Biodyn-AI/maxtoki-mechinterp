"""Step 4: cross-model Pearson 0.382 -- reproduce the published 80%-gene subsample CI, then
compare with an ordinary gene bootstrap WITH replacement (pairs of copies of one gene excluded).
Setup copied from runs/spectral-geometry-217M/scripts/audit_a3_bootstrap_cross_model_pearson.py:49-75.
Only the two embedding tensors are read (safe_open), not the full checkpoints."""
import sys; sys.dont_write_bytecode = True
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from safetensors.torch import safe_open

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
OUT = Path("<AGENT_TMP>/<SESSION_DIR>/"
           "d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")
GF = Path("<HF_CACHE>/hub/models--ctheodoris--Geneformer/snapshots/"
          "05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M")


def key(path, subs):
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if all(s in k for s in subs):
                return f.get_tensor(k).float().numpy()


gfm = key(GF / "model.safetensors", ["embeddings.word_embeddings"])
mt = key(PROJ / "setup/MaxToki-217M-HF/model.safetensors", ["embed_tokens"])
gf_feat = pd.read_csv(PROJ / "runs/spectral-geometry-217M/outputs/phase0/gene_features.csv")
ens = gf_feat["ensembl_id"].astype(str).tolist(); mt_ids = gf_feat["maxtoki_token_id"].astype(int).to_numpy()
tok = pickle.load(open(GF.parent / "geneformer/token_dictionary_gc104M.pkl", "rb"))
g_ids = np.array([tok.get(e, -1) for e in ens]); ok = (g_ids >= 0) & (g_ids < gfm.shape[0])
A = mt[mt_ids[ok]]; B = gfm[g_ids[ok]]
nz = lambda X: X / np.where(np.linalg.norm(X, axis=1, keepdims=True) == 0, 1, np.linalg.norm(X, axis=1, keepdims=True))
S1 = nz(A) @ nz(A).T; S2 = nz(B) @ nz(B).T
n = S1.shape[0]; iu = np.triu_indices(n, 1)
obs = float(np.corrcoef(S1[iu], S2[iu])[0, 1])
res = {"n_genes": int(n), "observed": obs}

rng = np.random.default_rng(42); sub = int(0.8 * n); iu_s = np.triu_indices(sub, 1); v = []
for _ in range(1000):
    idx = rng.choice(n, size=sub, replace=False)
    v.append(np.corrcoef(S1[np.ix_(idx, idx)][iu_s], S2[np.ix_(idx, idx)][iu_s])[0, 1])
v = np.array(v)
res["published_80pct_subsample_1000"] = {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "sd": float(v.std(ddof=1))}

rng = np.random.default_rng(7); w = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    a = S1[np.ix_(idx, idx)][iu]; b = S2[np.ix_(idx, idx)][iu]
    keep = idx[iu[0]] != idx[iu[1]]
    w.append(np.corrcoef(a[keep], b[keep])[0, 1])
w = np.array(w)
res["gene_bootstrap_with_replacement_1000"] = {"ci95": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))], "sd": float(w.std(ddof=1))}
print(json.dumps(res, indent=1))
(OUT / "04_crossmodel.json").write_text(json.dumps(res, indent=2))
