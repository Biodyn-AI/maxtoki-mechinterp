"""Independent verification of item D8, part 3: how much of each trustworthiness value is panel SIZE.

Trustworthiness (k = 15) is scored on panels of 50 (lung), 160 (zero-shot), 290 (internal) and 600 (external)
anchors. This script scores the DEPLOYED head (saved weights, no re-fit) on random anchor subsets of fixed size
drawn without replacement from each hematopoietic panel (internal: in-sample head), so that panels can be
compared at the same size. 200 draws per (panel, size); rng default_rng([31415, panel_code, size]).
Own code; does not import v2_common. Writes outputs/v2_verify/v2_verify_03_size_matched_trust.json.
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.manifold import trustworthiness

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART = RUN / "artifacts"; OUTV = RUN / "outputs/v2_verify"; OUTV.mkdir(parents=True, exist_ok=True)
T0 = time.time()
op = np.load(ART / "operators/pooled_drift_components.npz")
part = json.loads((ART / "operators/operator_index.json").read_text())["block_partition"]
sd_ = torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")
W = sd_["W.weight"].numpy(); b = sd_["b"].numpy()


def raw(p):
    c = np.load(ART / f"anchors/centroids_{p}.npy").astype(np.float32)
    xs = [c[:, [i + 1 for i in part[k]], :].mean(1) for k in ("early", "mid", "late")]
    ys = [xs[0] @ op["A_early"], xs[1] @ op["A_mid"], xs[2] @ op["A_late"]]
    return np.concatenate([ys[0] - ys[1], ys[1] - ys[2]], 1).astype(np.float32)


ri = raw("internal"); MU = ri.mean(0); SD = ri.std(0) + 1e-6
R = {"method": "deployed head, no re-fit; random anchor subsets without replacement; k = 15; 200 draws; "
               "percentile 2.5/97.5 of the draws (this is a spread over subsets, not a confidence interval)"}
codes = {"internal": 1, "external": 2, "zeroshot": 3, "lung_nonhema": 4}
for p in ["internal", "external", "zeroshot", "lung_nonhema"]:
    f = ((raw(p) - MU) / SD).astype(np.float32); z = ((f - b) @ W.T).astype(np.float32)
    R[p] = {"n_anchors": int(len(f)), "trust_full": float(trustworthiness(f, z, n_neighbors=15))}
    for n in [50, 103, 160, 290]:
        if n >= len(f):
            continue
        g = np.random.default_rng([31415, codes[p], n])
        v = np.array([trustworthiness(f[s], z[s], n_neighbors=15)
                      for s in (np.sort(g.choice(len(f), n, replace=False)) for _ in range(200))])
        R[p][f"subsets_n{n}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "p2.5_p97.5": [float(x) for x in np.percentile(v, [2.5, 97.5])],
                                 "frac_below_0.80": float((v < 0.80).mean())}
    print(p, json.dumps(R[p]), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


R["elapsed_s"] = time.time() - T0
(OUTV / "v2_verify_03_size_matched_trust.json").write_text(json.dumps(R, indent=2))
ins = [ART / f"anchors/centroids_{p}.npy" for p in codes] + [ART / "operators/pooled_drift_components.npz",
       ART / "operators/operator_index.json", ART / "heads/let_anchor_internal.pt"]
(OUTV / "run_config_v2_verify_03.json").write_text(json.dumps({
    "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "seeds": "default_rng([31415, panel_code, size]); codes internal 1, external 2, zeroshot 3, lung_nonhema 4",
    "inputs": {str(p): sha(p) for p in ins}}, indent=2))
