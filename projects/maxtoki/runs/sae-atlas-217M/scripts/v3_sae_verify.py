"""v3_sae_verify -- independent checks of the v3 SAE retrain (scripts/v3_sae.py).

Sub-commands (outputs in outputs/v3_sae/verify/):
  recompute    CPU, float64 sums, own code (not v3_sae.py). For every layer, from the kept held-out rows
               (outputs/v3_sae/activations_eval/layer_XX.npy) and SAEs loaded with hooks_v2.load_sae:
                 * v3 and deployed SAE: FVE vs the held-out mean (two-pass), FVE vs the SAE's own mu,
                   mean L0, dead features on held-out tokens; compared with eval/summary.json.
                 * deployed SAE re-centred: its mu replaced by the v3 train mean (v3 SAE mu; no
                   held-out data used). Shows how much of the deployed SAE's loss is a mean shift.
                 * mean shift: ||held-out mean - mu||^2 / (per-token variance around the held-out mean).
                 * decoder matching: for each v3 feature alive on held-out tokens, max |cos| with any
                   deployed decoder column; same for a random-direction null (seeded).
  l0_build     CPU: layer-0 activations of the DEPLOYED tokens = rows of model.embed_tokens.weight
               (hidden_states[0] of a Llama model is the embedding lookup; no forward pass). Written to
               --scratch. Also checks that the same lookup on the v3 tokens equals the stored v3
               layer_00.npy byte for byte.
  l0_train     MPS: v3_sae.train_one (the exact code used for the v3 SAEs) on those deployed-token
               layer-0 rows -> verify/l0_protocol/layer_00/. Resumable.
  l0_compare   CPU: the reproduced SAE vs the deployed layer-0 SAE (results, epoch losses, weights).
The deployed tokens are used ONLY in l0_build/l0_train, as a control that the training code reproduces
the deployed protocol. They never enter a forward pass and no result is built on them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/sae-atlas-217M"
OUT = RUN / "outputs/v3_sae"
VER = OUT / "verify"
DEP = RUN / "outputs/phase1"
MODEL = PROJ / "setup/MaxToki-217M-HF/model.safetensors"
sys.path.insert(0, str(PROJ / "setup"))
sys.path.insert(0, str(RUN / "scripts"))

import hooks_v2 as hv2  # noqa: E402
import inputs_v3 as iv3  # noqa: E402

N_LAYERS, HIDDEN, D_SAE, K = 12, 1232, 4928, 32
NULL_SEED = 7


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def rj(p, d=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else d


@torch.no_grad()
def sae_pass(sae, X: np.ndarray, mu: torch.Tensor, m_eval: np.ndarray, chunk=8192):
    """float64 sums of one SAE over X (CPU). mu may differ from the SAE's own (re-centring)."""
    W_enc, b_enc, W_dec = sae.sae.W_enc.weight, sae.sae.W_enc.bias, sae.sae.W_dec.weight
    sse = sst_mean = sst_mu = 0.0
    l0 = 0
    counts = np.zeros(D_SAE, np.int64)
    m = torch.from_numpy(m_eval.astype(np.float32))
    for a in range(0, X.shape[0], chunk):
        x = torch.from_numpy(np.ascontiguousarray(X[a:a + chunk], dtype=np.float32))
        pre = (x - mu) @ W_enc.T + b_enc
        v, i = pre.topk(K, dim=1)
        v = v.clamp(min=0)
        h = torch.zeros_like(pre).scatter_(1, i, v)
        xh = h @ W_dec.T + mu
        sse += float(((x.double() - xh.double()) ** 2).sum())
        sst_mean += float(((x.double() - m.double()) ** 2).sum())
        sst_mu += float(((x.double() - mu.double()) ** 2).sum())
        act = (v > 0).numpy()
        l0 += int(act.sum())
        counts += np.bincount(i.numpy()[act], minlength=D_SAE)
    n = X.shape[0]
    return {"fve_heldout_mean": 1 - sse / sst_mean, "fve_own_mu": 1 - sse / sst_mu, "mean_l0": l0 / n,
            "dead_heldout": int((counts == 0).sum()), "alive_mask": counts > 0,
            "sse_per_token": sse / n, "var_per_token": sst_mean / n}


def max_abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (d, nA), B (d, nB) -> for each column of A, max |cos| with any column of B."""
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    out = np.empty(A.shape[1])
    for a in range(0, A.shape[1], 1024):
        out[a:a + 1024] = np.abs(A[:, a:a + 1024].T @ B).max(axis=1)
    return out


def cmd_recompute(args):
    VER.mkdir(parents=True, exist_ok=True)
    path = VER / "recompute.json"
    res = rj(path, {}) or {}
    summ = rj(OUT / "eval/summary.json")
    t0 = time.time()
    rng = np.random.default_rng(NULL_SEED)
    for l in range(N_LAYERS):
        key = f"layer_{l:02d}"
        if key in res:
            continue
        if (time.time() - t0) / 60 > args.max_minutes - 1.5:
            log("time budget used; run again")
            break
        hv2.check_memory(3.0)
        X = np.load(OUT / f"activations_eval/layer_{l:02d}.npy")
        m_eval = X.astype(np.float64).mean(axis=0)
        v3 = hv2.load_sae(l, "cpu", OUT)
        dep = hv2.load_sae(l, "cpu", DEP)
        r_v3 = sae_pass(v3, X, v3.mu, m_eval)
        r_dep = sae_pass(dep, X, dep.mu, m_eval)
        r_rc = sae_pass(dep, X, v3.mu, m_eval)      # deployed SAE, re-centred on the v3 train mean
        var = r_v3["var_per_token"]
        mu_v3, mu_dep = v3.mu.double().numpy(), dep.mu.double().numpy()
        shift = {
            "heldout_mean_vs_v3_mu": float(((m_eval - mu_v3) ** 2).sum() / var),
            "heldout_mean_vs_deployed_mu": float(((m_eval - mu_dep) ** 2).sum() / var),
            "cos_v3_mu_deployed_mu": float(mu_v3 @ mu_dep / np.linalg.norm(mu_v3) / np.linalg.norm(mu_dep)),
            "norm_v3_mu": float(np.linalg.norm(mu_v3)), "norm_deployed_mu": float(np.linalg.norm(mu_dep)),
            "var_per_token_heldout": float(var),
        }
        Wv3 = v3.W_dec.numpy().astype(np.float64)
        Wdp = dep.W_dec.numpy().astype(np.float64)
        alive = r_v3["alive_mask"]
        mc = max_abs_cos(Wv3[:, alive], Wdp[:, r_dep["alive_mask"]]) if alive.any() else np.zeros(0)
        Rn = rng.standard_normal((HIDDEN, D_SAE))
        mc_null = max_abs_cos(Wv3[:, alive], Rn) if alive.any() else np.zeros(0)
        same_idx = np.abs((Wv3 * Wdp).sum(0) / np.linalg.norm(Wv3, axis=0) / np.linalg.norm(Wdp, axis=0))

        def q(v):
            return {"median": float(np.median(v)), "share_ge_0.9": float(np.mean(v >= 0.9)),
                    "share_ge_0.7": float(np.mean(v >= 0.7)), "share_ge_0.5": float(np.mean(v >= 0.5))} if v.size else {}

        s = summ["layers"][key] if summ else None
        rec = {
            "v3": {k: v for k, v in r_v3.items() if k != "alive_mask"},
            "deployed": {k: v for k, v in r_dep.items() if k != "alive_mask"},
            "deployed_recentred_on_v3_mu": {k: r_rc[k] for k in ("fve_heldout_mean", "mean_l0", "dead_heldout")},
            "mean_shift_over_per_token_variance": shift,
            "decoder_match": {
                "n_v3_alive_heldout": int(alive.sum()),
                "v3_alive_vs_deployed_alive_max_abs_cos": q(mc),
                "v3_alive_vs_random_null_max_abs_cos": q(mc_null),
                "same_index_abs_cos_v3_alive": q(same_idx[alive]),
                "same_index_abs_cos_dead_in_both": q(same_idx[~alive & ~r_dep["alive_mask"]]),
                "note": "both SAEs start from the same seed-42 init, so a feature that never fires keeps "
                        "its init in both (same-index cos ~1); only alive features are informative",
            },
        }
        if s:
            rec["vs_summary_absdiff"] = {
                "v3_fve": abs(s["v3"]["fve_heldout_mean_baseline"] - r_v3["fve_heldout_mean"]),
                "v3_fve_own_mu": abs(s["v3"]["fve_own_mu_baseline"] - r_v3["fve_own_mu"]),
                "dep_fve": abs(s["deployed_on_v3_tokens"]["fve_heldout_mean_baseline"] - r_dep["fve_heldout_mean"]),
                "dep_fve_own_mu": abs(s["deployed_on_v3_tokens"]["fve_own_mu_baseline"] - r_dep["fve_own_mu"]),
                "v3_l0": abs(s["v3"]["mean_l0"] - r_v3["mean_l0"]),
                "dep_l0": abs(s["deployed_on_v3_tokens"]["mean_l0"] - r_dep["mean_l0"]),
                "v3_dead_equal": s["v3"]["dead_heldout"] == r_v3["dead_heldout"],
                "dep_dead_equal": s["deployed_on_v3_tokens"]["dead_heldout"] == r_dep["dead_heldout"],
            }
        res[key] = rec
        hv2.write_json(path, res)
        log(f"L{l}: v3 FVE {r_v3['fve_heldout_mean']:.4f} dep {r_dep['fve_heldout_mean']:.4f} "
            f"dep recentred {r_rc['fve_heldout_mean']:.4f}; shift/var dep {shift['heldout_mean_vs_deployed_mu']:.3f} "
            f"v3 {shift['heldout_mean_vs_v3_mu']:.4f}; match>=0.9 {q(mc).get('share_ge_0.9', 0):.3f} "
            f"(null {q(mc_null).get('share_ge_0.9', 0):.3f}); diffs {rec.get('vs_summary_absdiff')}")
        del X
    if all(f"layer_{l:02d}" in res for l in range(N_LAYERS)):
        cfg = {"script": str(Path(__file__).resolve()), "script_sha256": iv3.sha256_file(__file__),
               "env": hv2.env_info("cpu"), "null_seed": NULL_SEED,
               "inputs": "outputs/v3_sae/activations_eval/layer_XX.npy (held-out rows of the v3 extraction; "
                         "encoding checked in v3_sae extract)",
               "eval_rows_sha256": {f"layer_{l:02d}": iv3.sha256_file(OUT / f"activations_eval/layer_{l:02d}.npy")
                                    for l in range(N_LAYERS)}}
        hv2.write_json(VER / "run_config_recompute.json", cfg)
        log("recompute DONE")


# ----------------------------------------------------------------------------- layer-0 protocol check
def _embed_table() -> np.ndarray:
    from safetensors import safe_open
    with safe_open(str(MODEL), "np") as f:
        return f.get_tensor("model.embed_tokens.weight").astype(np.float32)


def cmd_l0_build(args):
    t0 = time.time()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    VER.mkdir(parents=True, exist_ok=True)
    E = _embed_table()
    z = np.load(OUT / "cells.npz")
    tok_dep, tok_v3 = z["tokens_deployed"].astype(np.int64), z["tokens_v3"].astype(np.int64)
    n = len(tok_dep)
    # (1) lookup on the v3 tokens == stored v3 layer-0 activations, every row
    stored = np.load(OUT / "activations/layer_00.npy", mmap_mode="r")
    assert stored.shape == (n, HIDDEN)
    n_bad = 0
    for a in range(0, n, 65536):
        b = min(n, a + 65536)
        n_bad += int((~np.all(E[tok_v3[a:b]] == np.asarray(stored[a:b]), axis=1)).sum())
    log(f"embedding lookup of v3 tokens vs stored v3 layer_00: {n_bad} rows differ of {n}")
    if n_bad:
        raise SystemExit("hidden_states[0] is not the plain embedding lookup; stop")
    # (2) deployed-token layer-0 rows -> scratch .npy
    p = scratch / "deployed_tokens_layer_00.npy"
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n, HIDDEN))
    for a in range(0, n, 65536):
        b = min(n, a + 65536)
        mm[a:b] = E[tok_dep[a:b]]
    mm.flush()
    del mm
    rec = {"file": str(p), "n_rows": n, "v3_lookup_rows_differing_from_stored": n_bad,
           "tokens_deployed_sha256": iv3.array_sha256(z["tokens_deployed"]),
           "deployed_token_reproduction": rj(OUT / "prepare.json")["deployed_token_reproduction"],
           "seconds": time.time() - t0}
    hv2.write_json(VER / "l0_build.json", rec)
    log(f"l0_build DONE: {rec}")


def _patched_v3_module(scratch: Path):
    import v3_sae as vs
    d = VER / "l0_protocol"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("layout.json", "extract_progress.json"):
        if not (d / f).exists():
            shutil.copy(OUT / f, d / f)
    src = scratch / "deployed_tokens_layer_00.npy"
    vs.OUT = d
    vs.act_path = lambda layer: src
    vs.CACHE_DIR = scratch / "cache"
    return vs, d


def cmd_l0_train(args):
    vs, d = _patched_v3_module(Path(args.scratch))
    st = vs.train_one(0, args.max_minutes, time.time())
    log(f"l0_train: {st}")


def cmd_l0_compare(args):
    d = VER / "l0_protocol/layer_00"
    r_rep, r_dep = rj(d / "results.json"), rj(DEP / "layer_00/results.json")
    a = torch.load(d / "sae_final.pt", map_location="cpu", weights_only=False)
    b = torch.load(DEP / "layer_00/sae_final.pt", map_location="cpu", weights_only=False)
    w = {}
    for k in ("W_enc_weight", "W_enc_bias", "W_dec_weight", "mu"):
        x, y = a[k].double(), b[k].double()
        w[k] = {"max_abs_diff": float((x - y).abs().max()), "rel_fro_diff": float((x - y).norm() / y.norm())}
    Wa, Wb = a["W_dec_weight"].double().numpy(), b["W_dec_weight"].double().numpy()
    same = np.abs((Wa * Wb).sum(0))
    # the reproduction's held-out alive mask is not stored; use 'columns that moved from init' instead
    init = torch.manual_seed(42)
    from topk_sae import TopKSAE
    s0 = TopKSAE(HIDDEN, D_SAE, K)
    W0 = s0.W_dec.weight.detach().double().numpy()
    moved = np.abs((W0 * Wb).sum(0)) < 0.999999
    losses = [(x["loss"], y["loss"]) for x, y in zip(r_rep["training_log"], r_dep["training_log"])]
    out = {
        "reproduced": {k: r_rep[k] for k in ("variance_explained", "n_dead", "n_alive", "mean_l0_eval")},
        "deployed": {k: r_dep[k] for k in ("variance_explained", "n_dead", "n_alive")},
        "fve_abs_diff": abs(r_rep["variance_explained"] - r_dep["variance_explained"]),
        "epoch_losses_reproduced_vs_deployed": losses,
        "epoch_loss_max_rel_diff": max(abs(x - y) / y for x, y in losses),
        "weights": w,
        "deployed_columns_moved_from_init": int(moved.sum()),
        "same_index_abs_cos_moved_columns": {"median": float(np.median(same[moved])),
                                             "min": float(same[moved].min()),
                                             "share_ge_0.99": float(np.mean(same[moved] >= 0.99))},
        "same_index_abs_cos_unmoved_columns_min": float(same[~moved].min()) if (~moved).any() else None,
        "init_check": "TopKSAE(1232, 4928, 32) after torch.manual_seed(42) on CPU, as train_sae",
    }
    hv2.write_json(VER / "l0_protocol_compare.json", out)
    log(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recompute", "l0_build", "l0_train", "l0_compare"])
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()
    {"recompute": cmd_recompute, "l0_build": cmd_l0_build, "l0_train": cmd_l0_train,
     "l0_compare": cmd_l0_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
