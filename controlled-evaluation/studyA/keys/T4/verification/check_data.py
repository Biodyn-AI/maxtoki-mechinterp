"""Independent data checks for the T4 package (verifier's own code)."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path("<EVAL_ROOT>/studyA/tasks/T4-paper")
D = PKG / "data"
PROJ = Path("<REPO_ROOT>/projects/maxtoki")
out = {}

# manifest
man = {}
for line in (D / "MANIFEST.sha256").read_text().splitlines():
    h, p = line.split(None, 1)
    man[p.strip()] = h
files = sorted(str(f.relative_to(D)) for f in D.rglob("*") if f.is_file() and not f.name.startswith("._") and f.name != "MANIFEST.sha256")
out["manifest_lists_all_files"] = sorted(man) == files
bad = []
for p in files:
    h = hashlib.sha256(open(D / p, "rb").read()).hexdigest()
    if man.get(p) != h:
        bad.append(p)
out["manifest_mismatch"] = bad

mt = np.load(D / "maxtoki_217m/embed_tokens.npy", mmap_mode="r")
gf = np.load(D / "geneformer_v2_316m/word_embeddings.npy", mmap_mode="r")
sc = np.load(D / "scgpt_whole_human/encoder_embedding_weight.npy", mmap_mode="r")
ln = np.load(D / "scgpt_whole_human/encoder_enc_norm.npz")
out["shapes"] = {"mt": mt.shape, "gf": gf.shape, "sc": sc.shape}
out["dtypes"] = {"mt": str(mt.dtype), "gf": str(gf.dtype), "sc": str(sc.dtype)}
out["ln"] = {k: (ln[k].shape, str(ln[k].dtype)) for k in ln.files}
out["ln_eps"] = float(ln["eps"])
out["finite"] = {k: bool(np.isfinite(np.asarray(v)).all()) for k, v in [("mt", mt), ("gf", gf), ("sc", sc)]}

mtok = json.loads((D / "maxtoki_217m/token_dictionary.json").read_text())
gtok = json.loads((D / "geneformer_v2_316m/token_dictionary.json").read_text())
voc = json.loads((D / "scgpt_whole_human/vocab.json").read_text())
ens_m = {k: v for k, v in mtok.items() if k.startswith("ENSG")}
spec_m = {k: v for k, v in mtok.items() if k.startswith("<")}
num_m = {k: v for k, v in mtok.items() if not k.startswith("ENSG") and not k.startswith("<")}
out["maxtoki_dict"] = {"n": len(mtok), "ensg": len(ens_m), "special": spec_m, "numeric": len(num_m),
                       "numeric_min": min(int(k) for k in num_m), "numeric_max": max(int(k) for k in num_m),
                       "ids_ge_20275": sorted(set(k for k, v in mtok.items() if v >= 20275) - set(num_m)),
                       "numeric_all_ge_20275": all(v >= 20275 for v in num_m.values()),
                       "ensg_max_id": max(ens_m.values()), "ids_unique": len(set(mtok.values())) == len(mtok)}
ens_g = {k: v for k, v in gtok.items() if k.startswith("ENSG")}
out["gf_dict"] = {"n": len(gtok), "ensg": len(ens_g), "special": {k: v for k, v in gtok.items() if not k.startswith("ENSG")},
                  "max_id": max(gtok.values())}
out["mt_gf_same_ids_all_genes"] = (set(ens_m) == set(ens_g)) and all(ens_m[k] == ens_g[k] for k in ens_m)
vals = list(voc.values())
out["scgpt_vocab"] = {"n": len(voc), "ids_perm": sorted(vals) == list(range(len(vals))),
                      "pos_eq_id": int(sum(i == v for i, v in enumerate(vals))),
                      "special": {k: voc[k] for k in voc if k.startswith("<")}}
up = {}
for k in voc:
    up.setdefault(k.upper(), []).append(k)
out["scgpt_case_collisions"] = int(sum(len(v) > 1 for v in up.values()))

es = pd.read_csv(D / "gene_ids/ensembl_symbol.csv")
out["ens_sym"] = {"rows": len(es), "cols": list(es.columns), "uniq_ens": es.ensembl_id.nunique(),
                  "uniq_sym": es.symbol.nunique(), "na": int(es.isna().sum().sum())}
sym = dict(zip(es.ensembl_id, es.symbol))

panels = {p: pd.read_csv(D / f"panels/{p}.csv")["ensembl_id"].tolist() for p in ["lung", "immune", "external_lung"]}
out["panels"] = {}
for p, g in panels.items():
    out["panels"][p] = {"n": len(g), "uniq": len(set(g)), "in_mt": sum(e in mtok for e in g), "in_gf": sum(e in gtok for e in g),
                        "has_symbol": sum(e in sym for e in g), "sym_in_scgpt_exact": sum(sym.get(e) in voc for e in g),
                        "mt_id_lt_20275": sum(mtok[e] < 20275 for e in g)}
S = {p: set(g) for p, g in panels.items()}
out["overlaps"] = {"lung_immune": len(S["lung"] & S["immune"]), "lung_ext": len(S["lung"] & S["external_lung"]),
                   "immune_ext": len(S["immune"] & S["external_lung"]), "all3": len(S["lung"] & S["immune"] & S["external_lung"]),
                   "union": len(S["lung"] | S["immune"] | S["external_lung"])}

# deployed panels: same genes, same symbols?
for p, g in panels.items():
    dep = pd.read_csv(PROJ / f"runs/topology-141-217M/outputs/phase0/{p}/gene_features.csv")
    out["panels"][p]["same_as_deployed_order"] = dep["ensembl_id"].astype(str).tolist() == g
    out["panels"][p]["sym_matches_deployed"] = int(sum(sym.get(e) == s for e, s in zip(dep["ensembl_id"], dep["symbol"])))

# compare tables to checkpoints (independent read)
from safetensors import safe_open
with safe_open(str(PROJ / "setup/MaxToki-217M-HF/model.safetensors"), framework="np") as f:
    t = f.get_tensor("model.embed_tokens.weight")
out["mt_eq_ckpt"] = float(np.abs(t.astype(np.float32) - mt).max()); del t
gfp = Path("<HF_CACHE>/hub/models--ctheodoris--Geneformer/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M/model.safetensors")
with safe_open(str(gfp), framework="np") as f:
    t = f.get_tensor("bert.embeddings.word_embeddings.weight")
out["gf_eq_ckpt"] = float(np.abs(t.astype(np.float32) - gf).max()); del t
import pickle
gtp = gfp.parent.parent / "geneformer/token_dictionary_gc104M.pkl"
gt = pickle.load(open(gtp, "rb"))
out["gf_dict_eq_pkl"] = {str(k): int(v) for k, v in gt.items()} == gtok
nmp = pickle.load(open(gfp.parent.parent / "geneformer/gene_name_id_dict_gc104M.pkl", "rb"))
out["ens_sym_eq_pkl"] = dict(zip(es.symbol, es.ensembl_id)) == {str(k): str(v) for k, v in nmp.items()}
import torch
ck = torch.load("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/whole-human/best_model.pt", map_location="cpu", weights_only=False)
if not any(k.startswith("encoder.") for k in ck):
    for c in ["model_state_dict", "state_dict", "model"]:
        if c in ck:
            ck = ck[c]; break
out["sc_eq_ckpt"] = float(np.abs(ck["encoder.embedding.weight"].float().numpy() - sc).max())
out["ln_w_eq"] = float(np.abs(ck["encoder.enc_norm.weight"].float().numpy() - ln["weight"]).max())
out["ln_b_eq"] = float(np.abs(ck["encoder.enc_norm.bias"].float().numpy() - ln["bias"]).max())
del ck

# D10's subset table (rows chosen by the corrected lookup) vs my lookup
sub = np.load(PROJ / "runs/topology-141-217M/outputs/v2_crossmodel/embeddings_subset.npz")
out["d10_subset_keys"] = sub.files[:20]
def lnorm(X):
    X = X.astype(np.float64)
    m = X.mean(1, keepdims=True); v = X.var(1, keepdims=True)
    return ((X - m) / np.sqrt(v + float(ln["eps"]))) * ln["weight"] + ln["bias"]
cmp = {}
for p, g in panels.items():
    rows = np.array([voc[sym[e]] for e in g])
    mine = lnorm(np.asarray(sc[rows]))
    for k in sub.files:
        if p in k and "scgpt" in k.lower():
            arr = sub[k]
            if arr.shape == mine.shape:
                cmp[k] = float(np.abs(arr - mine).max())
            else:
                cmp[k] = str(arr.shape)
out["d10_subset_compare"] = cmp
print(json.dumps(out, indent=1, default=str))
