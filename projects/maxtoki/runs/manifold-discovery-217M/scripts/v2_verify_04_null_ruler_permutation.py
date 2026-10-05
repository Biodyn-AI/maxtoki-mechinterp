"""Independent verification of item D8, part 4: does the fifth gate (blocked permutation) tell H65 apart from the
run's own null ruler?

The run's null ruler (d_target_<panel>_null_shuffled.npy) shuffles stage labels WITHIN each branch
(phase1bc_hidden_states_and_centroids.py:215-230, rng default_rng(42)). It keeps branch structure and breaks the
order inside a branch. Here the frozen deployed head is scored against that null ruler on the external and zero-shot
panels, and the same blocked-permutation test as v2_04 (stage labels permuted within donor x tissue blocks,
statistic = Spearman(arc-cos latent distance, ruler) over all pairs, two-sided doubled p with +1) is applied to it.
2,000 permutations, rng default_rng([5151, panel_code]). Own code; does not import v2_common.
Writes outputs/v2_verify/v2_verify_04_null_ruler_permutation.json
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART = RUN / "artifacts"; OUTV = RUN / "outputs/v2_verify"
dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
nodes = list(dag["stage_to_branch"].keys()); ni = {s: i for i, s in enumerate(nodes)}
adj = {s: set() for s in nodes}
for a, b in dag["edges"]:
    adj[a].add(b); adj[b].add(a)
Tt = np.full((len(nodes),) * 2, 99.0)
for s in nodes:
    dist = {s: 0}; fr = [s]
    while fr:
        nx = []
        for u in fr:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; nx.append(v)
        fr = nx
    for t, dd in dist.items():
        Tt[ni[s], ni[t]] = dd
op = np.load(ART / "operators/pooled_drift_components.npz")
part = json.loads((ART / "operators/operator_index.json").read_text())["block_partition"]
hs = torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu"); W = hs["W.weight"].numpy(); bb = hs["b"].numpy()


def raw(p):
    c = np.load(ART / f"anchors/centroids_{p}.npy").astype(np.float32)
    xs = [c[:, [i + 1 for i in part[k]], :].mean(1) for k in ("early", "mid", "late")]
    ys = [xs[0] @ op["A_early"], xs[1] @ op["A_mid"], xs[2] @ op["A_late"]]
    return np.concatenate([ys[0] - ys[1], ys[1] - ys[2]], 1).astype(np.float32)


ri = raw("internal"); MU = ri.mean(0); SD = ri.std(0) + 1e-6


def ruler(st):
    si = np.array([ni[s] for s in st]); D = Tt[np.ix_(si, si)].copy(); np.fill_diagonal(D, 0); return D


R = {}
for p, code in [("external", 1), ("zeroshot", 2)]:
    t0 = time.time()
    m = pd.read_csv(ART / f"anchors/anchor_meta_{p}.csv"); st = m["hema_stage"].astype(str).to_numpy()
    # rebuild the run's within-branch shuffle exactly (phase1bc:215-230)
    g = np.random.default_rng(42); sh = st.copy()
    br = np.array([s2b.get(s, "_unknown") for s in st])
    for x in np.unique(br):
        ii = np.where(br == x)[0]; sh[ii] = g.permutation(st[ii])
    Dn = ruler(sh); saved = np.load(ART / f"anchors/d_target_{p}_null_shuffled.npy")
    f = ((raw(p) - MU) / SD).astype(np.float32); z = (f - bb) @ W.T
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    Dh = np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)); iu = np.triu_indices(len(st), 1); x = Dh[iu]
    out = {"null_ruler_rebuilt_equals_saved": bool(np.allclose(Dn, saved)),
           "labels_changed_by_within_branch_shuffle": int((sh != st).sum()), "n_anchors": int(len(st))}
    blocks = (m["donor_id"].astype(str) + "|" + m["tissue"].astype(str)).to_numpy()
    groups = [np.where(blocks == q)[0] for q in np.unique(blocks)]
    for lab, labels in [("H65", st), ("null_within_branch_shuffle", sh)]:
        obs = float(spearmanr(x, ruler(labels)[iu])[0])
        gg = np.random.default_rng([5151, code]); B = 2000; nul = np.empty(B)
        for k in range(B):
            s2 = labels.copy()
            for ii in groups:
                if len(ii) > 1:
                    s2[ii] = labels[gg.permutation(ii)]
            nul[k] = spearmanr(x, ruler(s2)[iu])[0]
        ph = (1 + (nul >= obs).sum()) / (B + 1); pl = (1 + (nul <= obs).sum()) / (B + 1)
        out[lab] = {"observed_global_spearman": obs, "null_mean": float(nul.mean()), "null_max": float(nul.max()),
                    "n_null_ge_obs": int((nul >= obs).sum()), "p_two_sided_doubled": float(min(1, 2 * min(ph, pl))),
                    "passes_p_le_0.001": bool(min(1, 2 * min(ph, pl)) <= 0.001)}
    out["seconds"] = time.time() - t0
    R[p] = out
    print(p, json.dumps(out), flush=True)


def sha(q):
    h = hashlib.sha256()
    with open(q, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


(OUTV / "v2_verify_04_null_ruler_permutation.json").write_text(json.dumps(R, indent=2))
ins = [ART / f"anchors/{n}" for n in ["centroids_internal.npy", "centroids_external.npy", "centroids_zeroshot.npy",
       "anchor_meta_external.csv", "anchor_meta_zeroshot.csv", "d_target_external_null_shuffled.npy",
       "d_target_zeroshot_null_shuffled.npy"]] + [ART / "operators/pooled_drift_components.npz",
       ART / "operators/operator_index.json", ART / "heads/let_anchor_internal.pt", RUN / "planning/h65_stage_dag.json"]
(OUTV / "run_config_v2_verify_04.json").write_text(json.dumps({
    "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "seeds": {"within-branch shuffle (as deployed)": 42, "block permutations": "default_rng([5151, panel_code])"},
    "inputs": {str(q): sha(q) for q in ins}}, indent=2))
