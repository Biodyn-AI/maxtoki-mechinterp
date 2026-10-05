"""v2 cross-model alignment (revision item D10), step 1: prepare inputs and check them.

What it does (CPU only, no forward pass):
  1. Reads the three static embedding tables (MaxToki embed_tokens, Geneformer V2-316M
     word_embeddings, scGPT whole-human normed gene embeddings) and the three topology
     gene panels (outputs/phase0/{lung,immune,external_lung}/gene_features.csv).
  2. Builds the gene rows exactly as the deployed scripts do, plus a CORRECTED scGPT
     lookup (the deployed one used dict position instead of the vocab index value).
  3. Checks the mappings (MaxToki token ids, Geneformer token ids, scGPT rows against an
     independently saved npz and against the original checkpoint).
  4. Reproduces the deployed per-domain numbers (CCA, Pearson, Spearman, top-1).
  5. Writes outputs/v2_crossmodel/embeddings_subset.npz, prepare_checks.json, run_config.json.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import platform
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open

import v2_crossmodel_common as C


def load_key(path, subs):
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if all(s in k for s in subs):
                return k, f.get_tensor(k).float().numpy()
    raise KeyError(subs)


def main():
    t0 = time.time()
    C.OUT.mkdir(parents=True, exist_ok=True)
    checks = {}

    gf_key, gf_emb = load_key(C.GENEFORMER_DIR / "model.safetensors", ["embeddings.word_embeddings"])
    mt_key, mt_emb = load_key(C.MAXTOKI_ST, ["embed_tokens"])
    checks["tensor_keys"] = {"maxtoki": mt_key, "maxtoki_shape": list(mt_emb.shape),
                             "geneformer": gf_key, "geneformer_shape": list(gf_emb.shape)}
    with open(C.GENEFORMER_TOKEN_DICT, "rb") as f:
        gf_tok = pickle.load(f)
    mt_tok = json.loads(C.MAXTOKI_TOKDICT.read_text())

    sc = torch.load(C.SCGPT_PT, map_location="cpu", weights_only=False)
    sc_vocab = sc["vocab"]                       # dict {symbol: row index}
    sc_emb = sc["normed_embeddings"].float().numpy()
    sc_raw = sc["embeddings"].float().numpy()
    keys = list(sc_vocab.keys())
    checks["scgpt_table"] = {
        "shape": list(sc_emb.shape),
        "n_vocab": len(keys),
        "n_keys_whose_dict_position_equals_index_value": int(sum(1 for i, k in enumerate(keys) if sc_vocab[k] == i)),
        "first_5_keys_with_index_values": [[k, int(sc_vocab[k])] for k in keys[:5]],
    }
    # deployed lookup (phase4_scgpt_cross_model.py:119-121)
    dep_vocab = [s.upper() for s in sc_vocab]
    dep_map = {s: i for i, s in enumerate(dep_vocab)}
    # corrected lookup: exact symbol first, then upper-case (last wins, as in deployed)
    up_map = {}
    for k, v in sc_vocab.items():
        up_map[k.upper()] = int(v)
    n_case_collisions = len(keys) - len(up_map)
    checks["scgpt_table"]["n_upper_case_collisions_in_vocab"] = int(n_case_collisions)
    pos_to_key = {i: k for i, k in enumerate(keys)}
    idx_to_key = {int(v): k for k, v in sc_vocab.items()}

    # independent check: npz saved by the extraction script with the right lookup
    znpz = np.load(C.SCGPT_NPZ, allow_pickle=True)
    npz_names = [str(x) for x in znpz["gene_names"]]
    npz_row = {g: i for i, g in enumerate(npz_names)}
    npz_emb = znpz["embeddings"]

    # original checkpoint check: raw table == encoder.embedding.weight, normed == LayerNorm(raw)
    ck = {}
    if C.SCGPT_CKPT.exists():
        st = torch.load(C.SCGPT_CKPT, map_location="cpu", weights_only=False)
        st = st.get("model_state_dict", st.get("state_dict", st.get("model", st))) if isinstance(st, dict) else st
        ek = [k for k in st if "encoder" in k and "embedding" in k and "weight" in k][0]
        W = st[ek].float()
        ck["embedding_key"] = ek
        ck["raw_equals_checkpoint_max_abs_diff"] = float((W - torch.from_numpy(sc_raw)).abs().max())
        nw = [k for k in st if "enc_norm" in k and "weight" in k]
        nb = [k for k in st if "enc_norm" in k and "bias" in k]
        if nw and nb:
            ln = torch.nn.functional.layer_norm(W, st[nw[0]].shape, st[nw[0]].float(), st[nb[0]].float())
            ck["enc_norm_keys"] = [nw[0], nb[0]]
            ck["normed_equals_layernorm_raw_max_abs_diff"] = float((ln - torch.from_numpy(sc_emb)).abs().max())
        del st, W
    checks["scgpt_checkpoint"] = ck

    # scGPT sanity pairs over the whole vocab: fixed vs deployed lookup
    pairs = [("RPL3", "RPL5"), ("RPS3", "RPS6"), ("CD3D", "CD3E"), ("CD79A", "CD79B"),
             ("HBA1", "HBB"), ("STAT1", "STAT2"), ("TP53", "GAPDH")]
    scn = C.normalise(sc_emb)
    sp = []
    for a, b in pairs:
        if a in up_map and b in up_map:
            cf = float(scn[up_map[a]] @ scn[up_map[b]])
            cd = float(scn[dep_map[a]] @ scn[dep_map[b]])
            sp.append({"pair": f"{a}-{b}", "cos_fixed": cf, "cos_deployed": cd,
                       "deployed_row_is_gene": [idx_to_key[dep_map[a]], idx_to_key[dep_map[b]]]})
    rnd = np.random.default_rng(0).integers(0, len(keys), size=(2000, 2))
    checks["scgpt_sanity_pairs"] = {
        "pairs": sp,
        "random_pair_cos_mean_sd": [float(np.mean([scn[i] @ scn[j] for i, j in rnd])),
                                    float(np.std([scn[i] @ scn[j] for i, j in rnd]))],
    }

    save = {}
    dom_checks = {}
    genes_by_domain = {}
    for dom in C.DOMAINS:
        gfeat = pd.read_csv(C.RUN / "outputs/phase0" / dom / "gene_features.csv")
        ens = gfeat["ensembl_id"].astype(str).to_numpy()
        syms = gfeat["symbol"].astype(str).to_numpy()
        mt_tids = gfeat["maxtoki_token_id"].astype(int).to_numpy()
        genes_by_domain[dom] = set(ens.tolist())
        dc = {"n_rows_gene_features": int(len(gfeat))}
        dc["maxtoki_token_id_matches_token_dictionary"] = int(sum(mt_tok.get(e, -9) == t for e, t in zip(ens, mt_tids)))
        # --- Geneformer (phase14:155-165)
        gf_tids = np.array([gf_tok.get(e, -1) for e in ens], dtype=np.int64)
        v_gf = (gf_tids >= 0) & (gf_tids < gf_emb.shape[0]) & (mt_tids >= 0) & (mt_tids < mt_emb.shape[0])
        dc["n_gf"] = int(v_gf.sum())
        dc["gf_token_id_equals_maxtoki_token_id"] = int((gf_tids[v_gf] == mt_tids[v_gf]).sum())
        save[f"{dom}__gf__mt"] = mt_emb[mt_tids[v_gf]]
        save[f"{dom}__gf__other"] = gf_emb[gf_tids[v_gf]]
        save[f"{dom}__gf__ens"] = ens[v_gf].astype("U20")
        # --- scGPT (phase4:132-144)
        syms_up = np.array([s.upper() for s in syms])
        dep_idx = np.array([dep_map.get(s, -1) for s in syms_up], dtype=np.int64)
        v_sc = (dep_idx >= 0) & (mt_tids >= 0) & (mt_tids < mt_emb.shape[0])
        fix_idx = np.array([int(sc_vocab[s]) if s in sc_vocab else up_map.get(s.upper(), -1) for s in syms],
                           dtype=np.int64)
        dc["n_scgpt"] = int(v_sc.sum())
        dc["scgpt_fixed_lookup_exact_case"] = int(sum(1 for s, ok in zip(syms, v_sc) if ok and s in sc_vocab))
        dc["scgpt_fixed_available_for_all_deployed_genes"] = bool((fix_idx[v_sc] >= 0).all())
        dc["scgpt_deployed_row_equals_fixed_row"] = int((dep_idx[v_sc] == fix_idx[v_sc]).sum())
        wrong = [(s, idx_to_key[int(d)]) for s, d, ok in zip(syms, dep_idx, v_sc) if ok][:8]
        dc["scgpt_deployed_examples_symbol_to_gene_actually_read"] = wrong
        # independent check vs npz (built with the right lookup by the extraction script)
        fixed_names = [idx_to_key[int(i)] for i in fix_idx[v_sc]]
        diffs = [float(np.abs(sc_emb[int(i)] - npz_emb[npz_row[g]]).max()) for i, g in zip(fix_idx[v_sc], fixed_names)
                 if g in npz_row]
        dc["scgpt_fixed_rows_vs_npz_max_abs_diff"] = float(max(diffs)) if diffs else None
        dc["scgpt_fixed_rows_checked_vs_npz"] = len(diffs)
        save[f"{dom}__sc__mt"] = mt_emb[mt_tids[v_sc]]
        save[f"{dom}__sc__fixed"] = sc_emb[fix_idx[v_sc]]
        save[f"{dom}__sc__deployed"] = sc_emb[dep_idx[v_sc]]
        save[f"{dom}__sc__fixed_raw"] = sc_raw[fix_idx[v_sc]]
        save[f"{dom}__sc__ens"] = ens[v_sc].astype("U20")
        save[f"{dom}__sc__sym"] = syms[v_sc].astype("U30")
        dom_checks[dom] = dc

    # gene overlap between the three panels
    ov = {}
    for i, a in enumerate(C.DOMAINS):
        for b in C.DOMAINS[i + 1:]:
            A, B = genes_by_domain[a], genes_by_domain[b]
            ov[f"{a}|{b}"] = {"shared": len(A & B), "jaccard": len(A & B) / len(A | B)}
    allg = set.intersection(*genes_by_domain.values())
    ov["shared_by_all_three"] = len(allg)
    ov["union"] = len(set.union(*genes_by_domain.values()))
    checks["panel_gene_overlap"] = ov
    checks["domains"] = dom_checks

    np.savez_compressed(C.PREP, **save)

    # --- reproduce deployed numbers
    dep14 = pd.read_csv(C.RUN / "outputs/phase14_cross_model/cross_model_alignment.csv").set_index("domain")
    dep4 = pd.read_csv(C.RUN / "outputs/phase4_scgpt/cross_model_alignment_scgpt.csv").set_index("domain")
    rep = {}
    timing = {}
    for dom in C.DOMAINS:
        for tag, mt_k, ot_k, ref in [("gf", "gf__mt", "gf__other", dep14), ("scgpt_deployed", "sc__mt", "sc__deployed", dep4)]:
            A_full = save[f"{dom}__{mt_k}"]; B_full = save[f"{dom}__{ot_k}"]
            n = A_full.shape[0]; d = C.d_align_for(n)
            t1 = time.time()
            A = C.pca_reduce(A_full, d); B = C.pca_reduce(B_full, d)
            t2 = time.time()
            m, rs, nconv = C.cca_mean_r(A, B, k=min(C.TOP_K_CCA, d))
            t3 = time.time()
            Sa = C.cos_matrix(A_full); Sb = C.cos_matrix(B_full)
            iu = np.triu_indices(n, 1)
            pear = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
            spear = float(pd.Series(Sa[iu]).corr(pd.Series(Sb[iu]), method="spearman"))
            t1r = C.top1_insample(A, B)
            timing[f"{dom}/{tag}"] = {"pca_s": t2 - t1, "cca_s": t3 - t2}
            r = ref.loc[dom]
            rep[f"{dom}/{tag}"] = {
                "n": n, "d_align": d,
                "cca_mean_r": m, "deployed_cca_mean_r": float(r["cca_mean_r"]),
                "cca_top_r": rs[0], "deployed_cca_top_r": float(r["cca_top_r"]),
                "cca_all_r": rs, "cca_convergence_warnings": nconv,
                "pairwise_pearson": pear, "deployed_pairwise_pearson": float(r["pairwise_pearson"]),
                "pairwise_spearman": spear, "deployed_pairwise_spearman": float(r["pairwise_spearman"]),
                "top1": t1r, "deployed_top1": float(r["top1_retrieval"]),
            }
            rep[f"{dom}/{tag}"]["max_abs_diff_vs_deployed"] = max(
                abs(m - float(r["cca_mean_r"])), abs(pear - float(r["pairwise_pearson"])),
                abs(spear - float(r["pairwise_spearman"])), abs(t1r - float(r["top1_retrieval"])))
    checks["reproduction_of_deployed"] = rep
    checks["timing_seconds"] = timing
    C.write_json(C.OUT / "prepare_checks.json", checks)

    # --- run_config with input hashes
    inputs = {
        "maxtoki_model_safetensors": C.MAXTOKI_ST,
        "maxtoki_token_dictionary": C.MAXTOKI_TOKDICT,
        "geneformer_v2_316m_model_safetensors": C.GENEFORMER_DIR / "model.safetensors",
        "geneformer_token_dictionary_gc104M": C.GENEFORMER_TOKEN_DICT,
        "scgpt_whole_human_gene_embeddings_pt": C.SCGPT_PT,
        "scgpt_gene_embeddings_npz_crosscheck": C.SCGPT_NPZ,
        "scgpt_checkpoint_crosscheck": C.SCGPT_CKPT,
        "deployed_phase14_csv": C.RUN / "outputs/phase14_cross_model/cross_model_alignment.csv",
        "deployed_phase4_csv": C.RUN / "outputs/phase4_scgpt/cross_model_alignment_scgpt.csv",
    }
    for dom in C.DOMAINS:
        inputs[f"gene_features_{dom}"] = C.RUN / "outputs/phase0" / dom / "gene_features.csv"
    hashes = {}
    for k, p in inputs.items():
        rp = os.path.realpath(p)
        hashes[k] = {"path": str(p), "resolved": rp, "bytes": os.path.getsize(rp), "sha256": C.sha256_file(rp)}
    hashes["prepared_embeddings_subset_npz"] = {"path": str(C.PREP), "sha256": C.sha256_file(C.PREP)}
    cfg = {
        "item": "D10 cross-model alignment",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scripts": ["scripts/v2_crossmodel_common.py", "scripts/v2_crossmodel_prepare.py",
                    "scripts/v2_crossmodel_stats.py", "scripts/v2_crossmodel_summarize.py"],
        "settings_copied_from_deployed": {"pca_dim": C.PCA_DIM_FOR_ALIGN, "pca_random_state": 42,
                                          "cca": "sklearn CCA(n_components=10, max_iter=200), mean of 10 in-sample canonical r",
                                          "top1": "orthogonal Procrustes on PCA-30, fitted and scored on same genes"},
        "inputs": hashes,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": __import__("sklearn").__version__, "torch": torch.__version__,
                     "pandas": pd.__version__},
        "device": "CPU only; no model forward pass",
        "wall_seconds_prepare": time.time() - t0,
    }
    C.write_json(C.OUT / "run_config.json", cfg)
    print(json.dumps({k: v for k, v in checks.items() if k in ("domains", "scgpt_table", "scgpt_checkpoint",
                                                               "panel_gene_overlap", "timing_seconds")}, indent=1))
    for k, v in rep.items():
        print(k, {kk: v[kk] for kk in ("cca_mean_r", "deployed_cca_mean_r", "pairwise_pearson", "top1",
                                       "deployed_top1", "max_abs_diff_vs_deployed", "cca_convergence_warnings")})
    print(json.dumps(checks["scgpt_sanity_pairs"], indent=1))
    print("done in", time.time() - t0)


if __name__ == "__main__":
    main()
