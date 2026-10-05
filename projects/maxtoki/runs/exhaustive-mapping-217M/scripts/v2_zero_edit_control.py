"""v2 zero-edit control for the deployed Experiment 3 (steering) hook.

Question: does the published steering table (Delta-s per layer at alpha=5:
+0.0962, +0.0178, -0.0079, +0.0141, +0.0048 for L0, L3, L6, L9, L11) survive
when the feature edit is exactly zero?

The deployed code (experiment3_rerun.py:250-305) scales the feature code
z_f -> alpha * z_f and patches decode(z') - decode(z) into the model. At
alpha = 1 that delta is exactly zero. The deployed hook still REPLACES the
output of block min(li, 10) with clean hidden_states[li] (+ delta). So with a
zero delta any change in the output is caused by the hook itself (block li is
skipped; at li = 11 the final RMSNorm is applied twice).

Per early-pseudotime cell (same 50 cells as the deployed run) this script runs:
  clean       : deployed clean pass (output_hidden_states=True)
  old_zero    : deployed hook logic, alpha = 1 (delta asserted exactly 0)
  old_a5      : deployed hook logic, alpha = 5, first feature of the layer
                (reproduction check against the published per-feature Delta-s)
  new_zero    : setup/hooks_v2.py Steer(alpha=1) (expected exactly 0)
  new_a5      : setup/hooks_v2.py Steer(alpha=5), same feature (preview only;
                one feature per layer, no null -- not a steering result)
Delta-s is computed exactly as deployed (logit space, cos to cached
g_late / g_early signatures, minus the clean baseline).

The original script is NOT modified; its logic is copied here.
Resumable: one JSON + one NPZ per cell under outputs/v2_zero_edit_control/cells/.
Usage: .venv/bin/python runs/exhaustive-mapping-217M/scripts/v2_zero_edit_control.py --max-minutes 8
       (repeat until it prints DONE; then it writes summary.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402
from dataset_loader import SYM2ENS_PKL  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
OUT3 = RUN / "outputs/experiment3"               # deployed outputs (read only)
OUT = RUN / "outputs/v2_zero_edit_control"
CELL_DIR = OUT / "cells"
TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"
SEED = 42                      # deployed
TS_SEED = SEED + 3333          # deployed cell sample seed
N_CELLS = 200                  # deployed
STEERING_LAYERS = [0, 3, 6, 9, 11]
BOOT_SEED = 20261001
N_BOOT = 10000
PUBLISHED_A5 = {0: 0.096226, 3: 0.017829, 6: -0.007897, 9: 0.014076, 11: 0.004817}


def cos(a, b):  # deployed (experiment3_rerun.py:157-161)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def delta_s(steered_logits, clean_logits, g_late, g_early):  # deployed lines 297-302
    z_clean = clean_logits.mean(axis=0)
    z_st = steered_logits.mean(axis=0)
    return ((cos(z_st, g_late) - cos(z_st, g_early))
            - (cos(z_clean, g_late) - cos(z_clean, g_early)))


def load_sae_cpu(li):  # deployed lines 58-69
    ckpt = torch.load(SAE_DIR / f"layer_{li:02d}/sae_final.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["config"]["d_model"], ckpt["config"]["d_sae"], ckpt["config"]["k"])
    sae.W_enc.weight.data = ckpt["W_enc_weight"]
    sae.W_enc.bias.data = ckpt["W_enc_bias"]
    sae.W_dec.weight.data = ckpt["W_dec_weight"]
    sae.eval()
    return sae, ckpt["mu"]


def prepare_cells():
    """Deployed cell sample + pseudotime (experiment3_rerun.py:111-134); cached."""
    cache = OUT / "early_cells_tokens.npz"
    sig = np.load(OUT3 / "state_signatures.npz")
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        return (list(c["token_ids"]), c["early_idx"], c["ts_rows"], sig, json.loads(str(c["checks"])))
    import anndata as ad
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    adata = ad.read_h5ad(str(TS_H5), backed="r")
    rng_ts = np.random.default_rng(TS_SEED)
    ts_idx = np.sort(rng_ts.choice(adata.n_obs, size=N_CELLS, replace=False))
    X_ts = adata[ts_idx].X
    if sp.issparse(X_ts) or hasattr(X_ts, "toarray"):
        X_ts = X_ts.toarray()
    X_ts = np.asarray(X_ts, dtype=np.float32)
    var_ens_ts = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
    obs_names = adata.obs_names.to_numpy()[ts_idx].astype(str)
    adata.file.close()
    rs = X_ts.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    X_log = np.log1p(X_ts / rs * 1e4)
    pca1 = PCA(n_components=1, random_state=SEED).fit_transform(X_log).flatten()
    pseudotime = (pca1 - pca1.min()) / (pca1.max() - pca1.min() + 1e-12)
    early_mask = pseudotime < np.percentile(pseudotime, 25)
    late_mask = pseudotime > np.percentile(pseudotime, 75)
    checks = {
        "pseudotime_maxabs_vs_cache": float(np.abs(pseudotime - sig["pseudotime"]).max()),
        "early_mask_equal_cache": bool((early_mask == sig["early_mask"]).all()),
        "late_mask_equal_cache": bool((late_mask == sig["late_mask"]).all()),
        "n_early": int(early_mask.sum()),
    }
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping(var_ens_ts.tolist())
    early_idx = np.where(early_mask)[0]
    token_ids = []
    for ci in early_idx:
        cell = tok.tokenize_cell(X_ts[ci], vi, vt, vm, max_len=2048)
        token_ids.append(None if cell is None else cell.token_ids)
    checks["n_early_tokenized"] = int(sum(t is not None for t in token_ids))
    ts_rows = ts_idx[early_idx]
    tok_arr = np.empty(len(token_ids), dtype=object)  # 1-D object array even if lengths match
    for j, t in enumerate(token_ids):
        tok_arr[j] = None if t is None else np.asarray(t, dtype=np.int64)
    np.savez(cache, token_ids=tok_arr, early_idx=early_idx,
             ts_rows=ts_rows, obs_names=obs_names[early_idx], checks=json.dumps(checks))
    return token_ids, early_idx, ts_rows, sig, checks


def steer_features():
    """Deployed selection: top 3 switch features per layer (experiment3_rerun.py:80-90)."""
    df = pd.read_csv(OUT3 / "switch_features.csv")
    out = {}
    for li in STEERING_LAYERS:
        out[li] = [int(f) for f in df[df["layer"] == li].head(3)["feature"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()
    t_start = time.time()
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    token_ids, early_idx, ts_rows, sig, checks = prepare_cells()
    g_early = sig["g_early_logits"]; g_late = sig["g_late_logits"]
    clean_cache = sig["clean_logits_per_cell"]
    feats = steer_features()
    print(f"early cells: {len(early_idx)}  checks: {checks}")
    assert checks["early_mask_equal_cache"] and checks["late_mask_equal_cache"], "cell selection differs from deployed run"

    cfg_path = OUT / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"chunks": []}
    cfg.update(H.env_info(args.device))
    cfg.update({
        "script": str(Path(__file__).resolve()),
        "script_sha256": H.sha256_file(__file__),
        "deployed_script_copied_from": str(RUN / "scripts/experiment3_rerun.py"),
        "deployed_script_sha256": H.sha256_file(RUN / "scripts/experiment3_rerun.py"),
        "seeds": {"ts_cell_sample": TS_SEED, "pca": SEED, "bootstrap": BOOT_SEED},
        "cells": {"dataset": str(TS_H5), "n_sampled": N_CELLS,
                  "early_positions_in_sample": [int(x) for x in early_idx],
                  "early_dataset_rows": [int(x) for x in ts_rows],
                  "selection_checks": checks},
        "features_steered_by_deployed_run": {str(k): v for k, v in feats.items()},
        "features_used_here_for_alpha5": {str(k): v[0] for k, v in feats.items()},
        "signatures": str(OUT3 / "state_signatures.npz"),
    })

    chunk = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "cells_done": [], "free_gb": H.available_memory_gb(),
             "script_sha256": H.sha256_file(__file__)}
    cfg.setdefault("script_sha256_chunks_1_2", cfg.get("script_sha256"))
    todo = [i for i in range(len(early_idx)) if token_ids[i] is not None
            and not (CELL_DIR / f"cell_{int(early_idx[i]):03d}.json").exists()]
    if todo:
        H.check_memory(args.min_free_gb)
        xt = MaxTokiAttentionExtractor(device=args.device, dtype=torch.float32)
        model = xt.model
        nb = xt.n_layers
        saes_cpu = {li: load_sae_cpu(li) for li in STEERING_LAYERS}
        saes_dev = H.load_saes(STEERING_LAYERS, device=args.device)
        vocab = int(model.config.vocab_size)

    for i in todo:
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("time budget reached; run again to continue")
            break
        try:
            gb = H.check_memory(args.min_free_gb)
        except H.MemoryGuardError as e:
            print(f"memory guard: {e}")
            chunk["stopped"] = str(e)
            break
        ci = int(early_idx[i])
        t0 = time.time()
        input_ids = torch.from_numpy(np.asarray(token_ids[i])[None, :]).to(args.device)
        rec = {"cell_in_sample": ci, "ts_row": int(ts_rows[i]), "seq_len": int(input_ids.shape[1])}
        vecs = {}
        with torch.no_grad():
            out_clean = model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            clean_logits = out_clean.logits[0].cpu().numpy().astype(np.float32)
            h_all = {li: out_clean.hidden_states[li][0].cpu() for li in STEERING_LAYERS}
            del out_clean
        rec["clean_vs_cached_meanlogit_maxabs"] = float(np.abs(clean_logits.mean(0) - clean_cache[ci]).max())

        for li in STEERING_LAYERS:
            hl = min(li, nb - 1)
            sae, mu = saes_cpu[li]
            f0 = feats[li][0]
            for tag, alpha in (("old_zero", 1.0), ("old_a5", 5.0)):
                with torch.no_grad():  # deployed lines 258-266
                    h_li = h_all[li]
                    z = sae.encode(torch.from_numpy(h_li.numpy().astype(np.float32)), mu)
                    z_st = z.clone(); z_st[:, f0] = z[:, f0] * alpha
                    delta = sae.decode(z_st, mu) - sae.decode(z, mu)
                    if tag == "old_zero":
                        nz = int(torch.count_nonzero(delta))
                        assert nz == 0, f"zero edit has {nz} non-zero entries"
                        rec[f"L{li}_old_zero_delta_nonzero"] = nz
                    else:
                        rec[f"L{li}_old_a5_delta_rms"] = float(delta.pow(2).mean().sqrt())
                        rec[f"L{li}_n_active_f{f0}"] = int((z[:, f0] > 0).sum())
                    h_patched = (h_li + delta.cpu()).to(args.device).unsqueeze(0)

                def _hook(module, inp, output, _p=h_patched):  # deployed lines 268-271
                    if isinstance(output, tuple):
                        return (_p.to(output[0].device),) + output[1:]
                    return _p.to(output.device)
                hook = model.model.layers[hl].register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        o = model(input_ids, output_hidden_states=False, use_cache=False, return_dict=True)
                        st = o.logits[0].cpu().numpy().astype(np.float32)
                        del o
                finally:
                    hook.remove()
                ld = (st - clean_logits).mean(axis=0)
                rec[f"L{li}_{tag}_delta_s"] = delta_s(st, clean_logits, g_late, g_early)
                rec[f"L{li}_{tag}_mean_logit_delta"] = float(np.mean(ld))
                rec[f"L{li}_{tag}_meanabs_logit_change"] = float(np.abs(st - clean_logits).mean())
                if tag == "old_zero":
                    vecs[f"L{li}_old_zero"] = ld.astype(np.float32)
                del st

            for tag, alpha in (("new_zero", 1.0), ("new_a5", 5.0)):
                lg, ed = H.run_with_edits(model, input_ids, saes_dev, [H.Steer(li, f0, alpha)])
                st = lg[0].numpy()
                rec[f"L{li}_{tag}_delta_s"] = delta_s(st, clean_logits, g_late, g_early)
                rec[f"L{li}_{tag}_meanabs_logit_change"] = float(np.abs(st - clean_logits).mean())
                rec[f"L{li}_{tag}_maxabs_logit_change"] = float(np.abs(st - clean_logits).max())
                del lg, st
            H.free_device_cache()

        rec["wall_seconds"] = round(time.time() - t0, 2)
        rec["free_gb_before"] = round(gb, 2)
        np.savez_compressed(CELL_DIR / f"cell_{ci:03d}_vecs.npz", **vecs)
        H.write_json(CELL_DIR / f"cell_{ci:03d}.json", rec)
        chunk["cells_done"].append(ci)
        print(f"cell {ci} ({rec['seq_len']} tok) {rec['wall_seconds']} s  "
              + "  ".join(f"L{li}:{rec[f'L{li}_old_zero_delta_s']:+.4f}" for li in STEERING_LAYERS))

    chunk["wall_seconds"] = round(time.time() - t_start, 1)
    cfg["chunks"].append(chunk)
    H.write_json(cfg_path, cfg)

    done = sorted(CELL_DIR.glob("cell_*[0-9].json"))
    n_expected = sum(t is not None for t in token_ids)
    print(f"{len(done)}/{n_expected} cells done")
    if len(done) == n_expected:
        summarize(done, feats)
        print("DONE")


def boot_ci(x, rng):
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(files, feats):
    recs = [json.loads(p.read_text()) for p in files]
    rng = np.random.default_rng(BOOT_SEED)
    tok = MaxTokiTokenizer()
    with open(SYM2ENS_PKL, "rb") as f:
        sym2ens = pickle.load(f)
    ens2sym = {v: k for k, v in sym2ens.items()}
    tok2ens = {v: k for k, v in tok.gene_token_dict.items()}

    def gene(t):
        e = tok2ens.get(int(t))
        return f"<tok{t}>" if e is None else ens2sym.get(e, e)

    dep = json.loads((OUT3 / "steering_summary.json").read_text())
    dep_feat = {(r["layer"], r["feature"]): r for r in dep["features"]}
    layers = {}
    for li in STEERING_LAYERS:
        f0 = feats[li][0]
        row = {"n_cells": len(recs), "feature_for_alpha5": f0}
        for tag in ("old_zero", "old_a5", "new_zero", "new_a5"):
            v = np.array([r[f"L{li}_{tag}_delta_s"] for r in recs])
            row[tag] = {"mean_delta_s": float(v.mean()), "ci95_cell_bootstrap": boot_ci(v, rng),
                        "sd": float(v.std(ddof=1)), "frac_positive": float((v > 0).mean()),
                        "meanabs_logit_change": float(np.mean([r[f"L{li}_{tag}_meanabs_logit_change"] for r in recs]))}
        oz = np.array([r[f"L{li}_old_zero_delta_s"] for r in recs])
        oa = np.array([r[f"L{li}_old_a5_delta_s"] for r in recs])
        row["old_a5_minus_old_zero"] = {"mean": float((oa - oz).mean()), "ci95_cell_bootstrap": boot_ci(oa - oz, rng),
                                        "max_abs": float(np.abs(oa - oz).max())}
        row["share_of_old_a5_explained_by_zero_edit"] = float(oz.mean() / oa.mean()) if oa.mean() != 0 else None
        vec = np.mean([np.load(CELL_DIR / f"cell_{r['cell_in_sample']:03d}_vecs.npz")[f"L{li}_old_zero"] for r in recs], axis=0)
        row["old_zero_top_up_genes"] = [gene(t) for t in np.argsort(-vec)[:5]]
        row["old_zero_frac_mean_logit_delta_positive"] = float(np.mean([r[f"L{li}_old_zero_mean_logit_delta"] > 0 for r in recs]))
        pub = dep_feat[(li, f0)]["alpha_results"]
        row["published"] = {
            "layer_mean_delta_s_alpha5": PUBLISHED_A5[li],
            "feature_delta_s_alpha5": pub["5"]["mean_delta_s"],
            "feature_delta_s_alpha2": pub["2"]["mean_delta_s"],
            "feature_frac_positive_maturity_alpha5": pub["5"]["frac_positive_maturity"],
            "feature_top_up_gene_alpha5": pub["5"]["top_upregulated"][0]["gene"],
            "all_3_features_delta_s_alpha5": [dep_feat[(li, f)]["alpha_results"]["5"]["mean_delta_s"] for f in feats[li]],
        }
        row["reproduction_old_a5_vs_published_feature_abs_diff"] = abs(row["old_a5"]["mean_delta_s"] - pub["5"]["mean_delta_s"])
        layers[f"L{li}"] = row
    summary = {
        "what": "Deployed experiment3 steering hook run with a zero feature edit (alpha=1)",
        "n_cells": len(recs),
        "resampling": f"cell-level percentile bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}",
        "max_clean_vs_cached_meanlogit_maxabs": float(max(r["clean_vs_cached_meanlogit_maxabs"] for r in recs)),
        "layers": layers,
    }
    H.write_json(OUT / "summary.json", summary)
    rows = []
    for li in STEERING_LAYERS:
        r = layers[f"L{li}"]
        rows.append({"layer": li, "published_layer_mean_a5": PUBLISHED_A5[li],
                     "published_feature_a5": r["published"]["feature_delta_s_alpha5"],
                     "old_a5_repro": r["old_a5"]["mean_delta_s"],
                     "old_zero_edit": r["old_zero"]["mean_delta_s"],
                     "old_zero_ci_lo": r["old_zero"]["ci95_cell_bootstrap"][0],
                     "old_zero_ci_hi": r["old_zero"]["ci95_cell_bootstrap"][1],
                     "old_zero_frac_pos": r["old_zero"]["frac_positive"],
                     "new_zero_edit": r["new_zero"]["mean_delta_s"],
                     "new_a5_preview": r["new_a5"]["mean_delta_s"],
                     "new_a5_ci_lo": r["new_a5"]["ci95_cell_bootstrap"][0],
                     "new_a5_ci_hi": r["new_a5"]["ci95_cell_bootstrap"][1]})
    pd.DataFrame(rows).to_csv(OUT / "zero_edit_table.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
