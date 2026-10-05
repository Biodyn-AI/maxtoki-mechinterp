"""Independent verification of item D8, part 2: checks that need the LET head re-fitted (CPU, small).

Own code (does not import v2_common): the trainer is re-typed from scripts/phase5_let_anchor.py:83-135
(same maths, same torch seed 42 and RNG order), trust is sklearn or an own masked version.
Tasks (one JSON line each in outputs/v2_verify/refits.jsonl; resumable):
  lotdo_TSP2         drop donor TSP2 (103 anchors left), re-standardise, re-fit, in-sample trust  -> v2 says 0.7627
  rand103|i          i = 0..11: 103 anchors drawn at random from all 290 (rng default_rng([8080, i])), re-fit, trust
                     (size-matched control for 'dropping TSP2 gives 0.763')
  tsp2only103|i      i = 0..5: 103 anchors drawn at random from TSP2's 187 anchors, re-fit, trust
  boot_rep|r         r = 0, 2: replay of the v2 training-donor bootstrap replicate r (default_rng([20261001, r]))
                     -> v2 says 0.818173 (r=0), 0.795832 (r=2)
  perm|b             b = 1662: replay of the v2 internal permutation with the largest null value -> v2 says 0.986593
  h103               H103 head (45 internal B-lineage anchors, as phase7_h103_external.py), trust on the zero-shot
                     panel: 18 kept anchors at k=5 and k=8, all 160 anchors at k=15; external all 600 at k=15
usage: python v2_verify_02_refits.py run <n_workers> <budget_s> | summarize
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART, PLAN = RUN / "artifacts", RUN / "planning"
OUTV = RUN / "outputs/v2_verify"; OUTV.mkdir(parents=True, exist_ok=True)
RES = OUTV / "refits.jsonl"
G = {}


def setup():
    torch.set_num_threads(1)
    op = np.load(ART / "operators/pooled_drift_components.npz")
    part = json.loads((ART / "operators/operator_index.json").read_text())["block_partition"]

    def raw(p):
        c = np.load(ART / f"anchors/centroids_{p}.npy").astype(np.float32)
        xs = [c[:, [i + 1 for i in part[k]], :].mean(1) for k in ("early", "mid", "late")]
        ys = [xs[0] @ op["A_early"], xs[1] @ op["A_mid"], xs[2] @ op["A_late"]]
        return np.concatenate([ys[0] - ys[1], ys[1] - ys[2]], 1).astype(np.float32)
    G["raw"] = {p: raw(p) for p in ["internal", "external", "zeroshot"]}
    G["d"] = np.load(ART / "anchors/d_target_internal.npy")
    G["m"] = {p: pd.read_csv(ART / f"anchors/anchor_meta_{p}.csv") for p in ["internal", "external", "zeroshot"]}
    G["don"] = G["m"]["internal"]["donor_id"].astype(str).to_numpy()
    dag = json.loads((PLAN / "h65_stage_dag.json").read_text()); G["dag"] = dag


class Head(nn.Module):  # phase5_let_anchor.LETHead, re-typed
    def __init__(self, d, k, beta0):
        super().__init__()
        self.W = nn.Linear(d, k, bias=False)
        self.b = nn.Parameter(torch.zeros(d))
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(beta0))))
        nn.init.normal_(self.W.weight, std=0.01)

    def forward(self, x):
        z = self.W(x - self.b)
        return z, (self.W.weight.T @ z.T).T + self.b


def fit(F, D):
    torch.manual_seed(42)
    head = Head(F.shape[1], 10, max(1.0, float(D.max()) / (np.pi / 2.0 + 1e-6)))
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    x = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)); Dt = torch.from_numpy(np.ascontiguousarray(D, dtype=np.float32))
    for _ in range(1500):
        opt.zero_grad()
        z, rec = head(x)
        zn = z / (z.norm(dim=1, keepdim=True) + 1e-9)
        dh = torch.exp(head.log_beta) * torch.arccos((zn @ zn.T).clamp(-1 + 1e-7, 1 - 1e-7))
        loss = ((dh - Dt) ** 2).mean() + 0.1 * ((rec - x) ** 2).mean()
        loss.backward(); opt.step()
    return head


def zof(head, F):
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)))[0].numpy()


def masked_trust(f, z, origin, k=15):
    n = len(origin); f64 = f.astype(np.float64); z64 = z.astype(np.float64)
    sq = (f64 ** 2).sum(1); DX = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * f64 @ f64.T, 0))
    sz = (z64 ** 2).sum(1); DZ = np.sqrt(np.maximum(sz[:, None] + sz[None] - 2 * z64 @ z64.T, 0))
    t = 0.0
    for i in range(n):
        cand = np.where(origin != origin[i])[0]
        ox = cand[np.argsort(DX[i, cand], kind="stable")]
        rank = np.empty(n, np.int64); rank[ox] = np.arange(1, len(ox) + 1)
        r = rank[cand[np.argsort(DZ[i, cand], kind="stable")][:k]] - k
        t += r[r > 0].sum()
    return 1.0 - t * 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))


def std_fit_trust(idx):
    raw = G["raw"]["internal"][idx]; mu = raw.mean(0); sd = raw.std(0) + 1e-6
    F = ((raw - mu) / sd).astype(np.float32); D = G["d"][np.ix_(idx, idx)]
    h = fit(F, D); z = zof(h, F)
    if len(np.unique(idx)) == len(idx):
        return float(trustworthiness(F, z, n_neighbors=15)), h, mu, sd
    return float(masked_trust(F, z, idx)), h, mu, sd


def run_task(t):
    torch.set_num_threads(1); t0 = time.time(); kind = t["kind"]; out = {"id": t["id"], "kind": kind}
    don = G["don"]
    if kind == "lotdo_TSP2":
        idx = np.where(don != "TSP2")[0]
        out["trust"], *_ = std_fit_trust(idx); out["n"] = int(len(idx))
    elif kind == "rand103":
        idx = np.sort(np.random.default_rng([8080, t["i"]]).choice(len(don), 103, replace=False))
        out["trust"], *_ = std_fit_trust(idx); out["n_TSP2"] = int((don[idx] == "TSP2").sum())
    elif kind == "tsp2only103":
        pool = np.where(don == "TSP2")[0]
        idx = np.sort(np.random.default_rng([9090, t["i"]]).choice(pool, 103, replace=False))
        out["trust"], *_ = std_fit_trust(idx)
    elif kind == "boot_rep":
        ud = np.array(sorted(set(don)))
        pick = np.random.default_rng([20261001, t["r"]]).choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([np.where(don == u)[0] for u in pick])
        out["trust"], *_ = std_fit_trust(idx); out["n"] = int(len(idx))
    elif kind == "perm":
        m = G["m"]["internal"]; stg = m["hema_stage"].astype(str).to_numpy()
        blk = (m["donor_id"].astype(str) + "|" + m["tissue"].astype(str)).to_numpy()
        g = np.random.default_rng([777, t["b"]]); s = stg.copy()
        for gg in pd.unique(blk):
            ii = np.where(blk == gg)[0]
            if len(ii) > 1:
                s[ii] = stg[g.permutation(ii)]
        dag = G["dag"]; nodes = list(dag["stage_to_branch"].keys()); ni = {x: i for i, x in enumerate(nodes)}
        adj = {x: set() for x in nodes}
        for a, b in dag["edges"]:
            adj[a].add(b); adj[b].add(a)
        Tt = np.full((len(nodes),) * 2, 99.0)
        for s0 in nodes:
            dist = {s0: 0}; fr = [s0]
            while fr:
                nx = []
                for u in fr:
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1; nx.append(v)
                fr = nx
            for q, dd in dist.items():
                Tt[ni[s0], ni[q]] = dd
        si = np.array([ni[x] for x in s]); D = Tt[np.ix_(si, si)]; np.fill_diagonal(D, 0)
        raw = G["raw"]["internal"]; mu = raw.mean(0); sd = raw.std(0) + 1e-6; F = ((raw - mu) / sd).astype(np.float32)
        z = zof(fit(F, D), F)
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        dh = np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)); iu = np.triu_indices(len(si), 1)
        out["stat"] = float(spearmanr(dh[iu], D[iu])[0]); out["n_changed"] = int((s != stg).sum())
    elif kind == "h103":
        sys.path.insert(0, str(RUN / "scripts"))
        from phase3a_sweep2_ordinal import H103_DEPTH  # read-only import of the depth map (no bytecode written)
        def depths(meta):
            ct = meta["cell_type"].astype(str).str.strip().str.lower()
            return ct.map(H103_DEPTH).fillna(-1).astype(int).to_numpy()
        raw = G["raw"]["internal"]; mu = raw.mean(0); sd = raw.std(0) + 1e-6
        dep = depths(G["m"]["internal"]); keep = np.where(dep >= 0)[0]
        D = np.abs(dep[keep][:, None] - dep[keep][None, :]).astype(np.float32)
        F = ((raw[keep] - mu) / sd).astype(np.float32); h = fit(F, D)
        out["internal_n_kept"] = int(len(keep)); out["internal_trust_kept_k15"] = float(trustworthiness(F, zof(h, F), n_neighbors=15))
        out["internal_depth_values"] = sorted(set(dep[keep].tolist()))
        for p in ["zeroshot", "external"]:
            f = ((G["raw"][p] - mu) / sd).astype(np.float32); z = zof(h, f); dp = depths(G["m"][p]); kk = np.where(dp >= 0)[0]
            out[p] = {"n_all": int(len(f)), "trust_all_k15": float(trustworthiness(f, z, n_neighbors=15)), "n_kept": int(len(kk)),
                      "n_kept_donors": int(G["m"][p].loc[kk, "donor_id"].nunique()), "depth_values": sorted(set(dp[kk].tolist())),
                      "trust_kept_k5": float(trustworthiness(f[kk], z[kk], n_neighbors=5)),
                      "trust_kept_k8": float(trustworthiness(f[kk], z[kk], n_neighbors=8)),
                      "trust_kept_k15": float(trustworthiness(f[kk], z[kk], n_neighbors=15)) if len(kk) > 30 else None}
    out["seconds"] = time.time() - t0
    return out


def tasks():
    T = [{"id": "h103", "kind": "h103"}, {"id": "lotdo_TSP2", "kind": "lotdo_TSP2"}]
    T += [{"id": f"boot_rep|{r}", "kind": "boot_rep", "r": r} for r in (2, 0)]
    T += [{"id": f"rand103|{i}", "kind": "rand103", "i": i} for i in range(12)]
    T += [{"id": f"tsp2only103|{i}", "kind": "tsp2only103", "i": i} for i in range(6)]
    T += [{"id": "perm|1662", "kind": "perm", "b": 1662}]
    return T


def run(nw, budget):
    T0 = time.time(); setup()
    done = {json.loads(l)["id"] for l in RES.read_text().splitlines() if l.strip()} if RES.exists() else set()
    todo = [t for t in tasks() if t["id"] not in done]
    print("todo", len(todo), flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(nw, initializer=setup) as pool, open(RES, "a") as fh:
        pend = []; it = iter(todo); ex = False
        while True:
            while not ex and len(pend) < nw and time.time() - T0 < budget:
                try:
                    pend.append(pool.apply_async(run_task, (next(it),)))
                except StopIteration:
                    ex = True
            if not pend:
                break
            keep = []
            for a in pend:
                if a.ready():
                    fh.write(json.dumps(a.get()) + "\n"); fh.flush()
                else:
                    keep.append(a)
            pend = keep
            if time.time() - T0 > budget + 60:
                pool.terminate(); break
            if pend:
                pend[0].wait(0.5)
    print("elapsed", round(time.time() - T0), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def summarize():
    rows = {r["id"]: r for r in (json.loads(l) for l in RES.read_text().splitlines() if l.strip())}
    v2 = {json.loads(l)["id"]: json.loads(l) for l in (RUN / "outputs/v2_intervals/refit_pool/results.jsonl").read_text().splitlines() if l.strip()}
    lot = json.loads((RUN / "outputs/v2_intervals/v2_10_internal_lotdo.json").read_text())
    fitj = json.loads((RUN / "outputs/v2_intervals/v2_05_other_orderings_fit.json").read_text())
    r103 = [rows[f"rand103|{i}"]["trust"] for i in range(12) if f"rand103|{i}" in rows]
    t2 = [rows[f"tsp2only103|{i}"]["trust"] for i in range(6) if f"tsp2only103|{i}" in rows]
    S = {"lotdo_TSP2": {"verify": rows.get("lotdo_TSP2", {}).get("trust"), "v2": lot["per_donor"]["TSP2"]["trust"]},
         "boot_rep_0": {"verify": rows.get("boot_rep|0", {}).get("trust"), "v2": v2["boot_full|0"]["trust"]},
         "boot_rep_2": {"verify": rows.get("boot_rep|2", {}).get("trust"), "v2": v2["boot_full|2"]["trust"]},
         "perm_1662": {"verify": rows.get("perm|1662", {}).get("stat"), "v2": v2["perm|1662"]["stat"]},
         "rand103_refit_trust": {"n": len(r103), "mean": float(np.mean(r103)) if r103 else None, "sd": float(np.std(r103, ddof=1)) if len(r103) > 1 else None,
                                 "min": float(np.min(r103)) if r103 else None, "max": float(np.max(r103)) if r103 else None,
                                 "frac_below_0.80": float(np.mean(np.array(r103) < 0.8)) if r103 else None, "values": r103,
                                 "n_TSP2_in_draw": [rows[f"rand103|{i}"]["n_TSP2"] for i in range(12) if f"rand103|{i}" in rows]},
         "tsp2only103_refit_trust": {"n": len(t2), "mean": float(np.mean(t2)) if t2 else None, "min": float(np.min(t2)) if t2 else None,
                                     "max": float(np.max(t2)) if t2 else None, "values": t2},
         "h103": rows.get("h103"),
         "h103_v2": {"zeroshot_all_k15": fitj["H103"]["zeroshot"]["trust_all_anchors_k15"], "zeroshot_kept_k5": fitj["H103"]["zeroshot"]["trust_kept_k5"],
                     "zeroshot_kept_k8": fitj["H103"]["zeroshot"]["trust_kept_k8"], "external_all_k15": fitj["H103"]["external"]["trust_all_anchors_k15"],
                     "external_kept_k15": fitj["H103"]["external"]["trust_kept_k15"], "internal_kept_k15": fitj["H103"]["internal"]["trust_in_sample_k15_kept"]}}
    (OUTV / "v2_verify_02_refits.json").write_text(json.dumps(S, indent=2))
    ins = [ART / "anchors/centroids_internal.npy", ART / "anchors/centroids_external.npy", ART / "anchors/centroids_zeroshot.npy",
           ART / "anchors/d_target_internal.npy", ART / "anchors/anchor_meta_internal.csv", ART / "anchors/anchor_meta_external.csv",
           ART / "anchors/anchor_meta_zeroshot.csv", ART / "operators/pooled_drift_components.npz", ART / "operators/operator_index.json",
           PLAN / "h65_stage_dag.json", RUN / "scripts/phase3a_sweep2_ordinal.py", RES,
           RUN / "outputs/v2_intervals/refit_pool/results.jsonl", RUN / "outputs/v2_intervals/v2_10_internal_lotdo.json",
           RUN / "outputs/v2_intervals/v2_05_other_orderings_fit.json"]
    (OUTV / "run_config_v2_verify_02.json").write_text(json.dumps({
        "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1,
        "seeds": {"head": 42, "rand103": "default_rng([8080, i])", "tsp2only103": "default_rng([9090, i])",
                  "boot replay": "default_rng([20261001, r])", "perm replay": "default_rng([777, b])"},
        "inputs": {str(p): sha(p) for p in ins}}, indent=2))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), float(sys.argv[3]))
    else:
        summarize()
