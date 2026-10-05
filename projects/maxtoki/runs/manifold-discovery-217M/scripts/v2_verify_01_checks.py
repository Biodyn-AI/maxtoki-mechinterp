"""Independent verification of item D8 (v2 manifold intervals), part 1: checks that need no head re-fit.

Written by the verifying agent. Does NOT import v2_common (own code for features, head output, metrics),
so that every number here is a second derivation, not a re-run of the same code.
Reads: artifacts/*, reports/*, planning/*, outputs/v2_intervals/* (read only).
Writes: outputs/v2_verify/v2_verify_01_checks.json and run_config_v2_verify_01.json only.

Sections
  A  frozen-head numbers re-derived (own numpy code; deployed head weights)
  B  trustworthiness as a function of k (frozen head) - what copies of anchors do to the metric
  C  frozen donor bootstrap: share of copied rows vs trust; a second copy treatment (drop copies)
  D  frozen-panel blocked permutation re-done with a different RNG stream and scipy.spearmanr
  E  internal permutation gate re-summarised from the saved re-fit rows
  F  internal training-donor bootstrap: trust vs donor TSP2; plain-mean branch value vs number of branches
  G  facts (compaction bytes, factor shares, DAG counts, cohort overlaps)
  H  report / table consistency checks
usage: python v2_verify_01_checks.py
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, rankdata
from sklearn.manifold import trustworthiness

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART, REP, PLAN = RUN / "artifacts", RUN / "reports", RUN / "planning"
V2 = RUN / "outputs/v2_intervals"
OUTV = RUN / "outputs/v2_verify"; OUTV.mkdir(parents=True, exist_ok=True)
J = lambda p: json.loads(Path(p).read_text())
T0 = time.time()
INPUTS = []


def inp(p):
    INPUTS.append(Path(p)); return Path(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ own data code
op = np.load(inp(ART / "operators/pooled_drift_components.npz"))
part = J(inp(ART / "operators/operator_index.json"))["block_partition"]
dag = J(inp(PLAN / "h65_stage_dag.json"))
s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
head_sd = torch.load(inp(ART / "heads/let_anchor_internal.pt"), map_location="cpu")
W = head_sd["W.weight"].numpy().astype(np.float64); bvec = head_sd["b"].numpy().astype(np.float64)


def feats_raw(panel):
    c = np.load(inp(ART / f"anchors/centroids_{panel}.npy")).astype(np.float32)
    xs = [c[:, [i + 1 for i in part[k]], :].mean(1) for k in ("early", "mid", "late")]
    ys = [xs[0] @ op["A_early"], xs[1] @ op["A_mid"], xs[2] @ op["A_late"]]
    return np.concatenate([ys[0] - ys[1], ys[1] - ys[2]], 1).astype(np.float32)


F_int_raw = feats_raw("internal")
MU = F_int_raw.mean(0); SD = F_int_raw.std(0) + 1e-6


def panel(p):
    f = ((feats_raw(p) - MU) / SD).astype(np.float32)
    z = ((f.astype(np.float64) - bvec) @ W.T)
    d = np.load(inp(ART / f"anchors/d_target_{p}.npy"))
    m = pd.read_csv(inp(ART / f"anchors/anchor_meta_{p}.csv"))
    return f, z, d, m


def arc(z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7))


def within(Dh, D, labels, min_n=3):
    out = []
    for g, ii in pd.Series(np.arange(len(labels))).groupby(np.asarray(labels)):
        ii = ii.to_numpy()
        if g == "_unk" or len(ii) < min_n:
            continue
        a, b = np.triu_indices(len(ii), 1)
        y = D[ii[a], ii[b]]; x = Dh[ii[a], ii[b]]
        if np.ptp(y) == 0:
            continue
        out.append((str(g), len(ii), float(spearmanr(x, y)[0])))
    return out


R = {}
# ------------------------------------------------------------------ A frozen reproduction
pub = {"external": J(inp(REP / "external_validation_external.json")), "zeroshot": J(inp(REP / "zeroshot_transfer_anchor_head.json")),
       "lung_nonhema": J(inp(REP / "external_validation_lung_nonhema.json"))}
P = {}
A = {}
for p in ["external", "zeroshot", "lung_nonhema"]:
    f, z, d, m = panel(p); P[p] = (f, z, d, m)
    Dh = arc(z); iu = np.triu_indices(len(d), 1)
    br = m["hema_stage"].map(s2b).fillna("_unk").to_numpy()
    wb = within(Dh, d, br); wd = within(Dh, d, m["donor_id"].astype(str).to_numpy())
    v = np.array([x[2] for x in wb]); w = np.array([x[1] for x in wb], float)
    A[p] = {"trust_k15": float(trustworthiness(f, z, n_neighbors=15)),
            "branch_plain": float(v.mean()), "branch_anchor_weighted": float((w * v).sum() / w.sum()),
            "branches": wb, "donor_plain": float(np.mean([x[2] for x in wd])),
            "global": float(spearmanr(Dh[iu], d[iu])[0]),
            "deployed": {k: pub[p][k] for k in ["trustworthiness", "branch_holdout", "donor_holdout", "global_correlation"]}}
    A[p]["max_abs_diff_vs_deployed"] = max(abs(A[p]["trust_k15"] - pub[p]["trustworthiness"]),
                                           abs(A[p]["branch_plain"] - pub[p]["branch_holdout"]),
                                           abs(A[p]["donor_plain"] - pub[p]["donor_holdout"]),
                                           abs(A[p]["global"] - pub[p]["global_correlation"]))
R["A_frozen_reproduction"] = A

# ------------------------------------------------------------------ B trust vs k
f_int = ((F_int_raw - MU) / SD).astype(np.float32); z_int = (f_int.astype(np.float64) - bvec) @ W.T
B = {"internal_in_sample_saved_head": {k: float(trustworthiness(f_int, z_int, n_neighbors=k)) for k in [3, 5, 8, 10, 15, 20, 30]}}
for p in ["external", "zeroshot", "lung_nonhema"]:
    f, z, d, m = P[p]
    B[p] = {k: float(trustworthiness(f, z, n_neighbors=k)) for k in [3, 5, 8, 10, 15, 20] if k < len(d) / 2}
# n-dependence at fixed head and fixed k: random anchor subsets (no re-fit)
rng = np.random.default_rng(424242)
sub = {}
for n_sub in [103, 150, 200, 250]:
    vals = []
    for _ in range(40):
        ii = np.sort(rng.choice(len(f_int), n_sub, replace=False))
        vals.append(float(trustworthiness(f_int[ii], z_int[ii], n_neighbors=15)))
    sub[n_sub] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "min": float(np.min(vals)), "max": float(np.max(vals))}
B["internal_saved_head_random_anchor_subsets_k15_40_draws"] = sub
don_int = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")["donor_id"].astype(str).to_numpy()
nt = np.where(don_int != "TSP2")[0]; t2 = np.where(don_int == "TSP2")[0]
B["internal_saved_head_without_TSP2_no_refit_k15"] = float(trustworthiness(f_int[nt], z_int[nt], n_neighbors=15))
B["internal_saved_head_TSP2_only_no_refit_k15"] = float(trustworthiness(f_int[t2], z_int[t2], n_neighbors=15))
R["B_trust_vs_k_and_n"] = B


# ------------------------------------------------------------------ C copies in the frozen donor bootstrap
def own_masked_trust(f, z, origin, k=15):
    """Own implementation (loop over rows) of trustworthiness where rows with the same origin are not neighbours."""
    n = len(origin)
    f64 = f.astype(np.float64)
    sq = (f64 ** 2).sum(1); DX = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * f64 @ f64.T, 0))
    sqz = (z ** 2).sum(1); DZ = np.sqrt(np.maximum(sqz[:, None] + sqz[None] - 2 * z @ z.T, 0))
    t = 0.0
    for i in range(n):
        ok = origin != origin[i]
        cand = np.where(ok)[0]
        ox = cand[np.argsort(DX[i, cand], kind="stable")]
        rank = np.empty(n, np.int64); rank[ox] = np.arange(1, len(ox) + 1)
        nz = cand[np.argsort(DZ[i, cand], kind="stable")][:k]
        r = rank[nz] - k
        t += r[r > 0].sum()
    return 1.0 - t * 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))


C = {}
for p, code in [("external", 1), ("zeroshot", 2)]:
    f, z, d, m = P[p]
    don = m["donor_id"].astype(str).to_numpy(); ud = np.array(sorted(set(don)))
    mem = {u: np.where(don == u)[0] for u in ud}
    saved = pd.DataFrame([json.loads(l) for l in inp(V2 / f"frozen_boot/{p}.jsonl").read_text().splitlines() if l.strip()]).set_index("rep")
    rows = []
    for r in range(len(saved)):
        g = np.random.default_rng([4242, code, r])
        pick = g.choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([mem[u] for u in pick])
        uniq = np.unique(idx)
        row = {"rep": r, "n_rows": len(idx), "n_distinct": len(uniq), "share_copied": 1 - len(uniq) / len(idx),
               "trust_saved": float(saved.loc[r, "trust"]),
               "trust_distinct_only": float(trustworthiness(f[uniq], z[uniq], n_neighbors=15)) if len(uniq) > 30 else np.nan}
        if r < 40:  # own masked-trust implementation on the first 40 replicates (second code path)
            row["trust_own_masked"] = float(own_masked_trust(f[idx], z[idx], idx))
        rows.append(row)
    df = pd.DataFrame(rows)
    q = pd.qcut(df.share_copied, 3, labels=["low", "mid", "high"])
    C[p] = {"n_reps": int(len(df)), "corr_trust_vs_share_copied": float(np.corrcoef(df.trust_saved, df.share_copied)[0, 1]),
            "mean_trust_by_share_copied_tercile": df.groupby(q, observed=True).trust_saved.mean().round(4).to_dict(),
            "own_masked_vs_saved_max_abs_diff_first40": float((df.trust_own_masked - df.trust_saved).abs().max()),
            "copies_masked_percentile_ci": [float(x) for x in np.percentile(df.trust_saved, [2.5, 97.5])],
            "copies_masked_frac_below_0.80": float((df.trust_saved < 0.80).mean()),
            "copies_dropped_percentile_ci": [float(x) for x in np.nanpercentile(df.trust_distinct_only, [2.5, 97.5])],
            "copies_dropped_frac_below_0.80": float((df.trust_distinct_only.dropna() < 0.80).mean()),
            "copies_dropped_n_distinct_range": [int(df.n_distinct.min()), int(df.n_distinct.max())],
            "note": "copies dropped = trust on the distinct anchors of each donor resample (a random subset of donors, "
                    "no copies); it is a different estimator, shown only to see how much copy handling moves the tail"}
R["C_copies_in_frozen_bootstrap"] = C

# ------------------------------------------------------------------ D frozen permutation, own RNG stream
nodes = list(dag["stage_to_branch"].keys()); ni = {s: i for i, s in enumerate(nodes)}
adj = {s: set() for s in nodes}
for a, b in dag["edges"]:
    adj[a].add(b); adj[b].add(a)
Tt = np.full((len(nodes), len(nodes)), 99.0)
for s in nodes:  # BFS
    dist = {s: 0}; frontier = [s]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; nxt.append(v)
        frontier = nxt
    for t, dd in dist.items():
        Tt[ni[s], ni[t]] = dd
D_ = {}
for p in ["external", "zeroshot", "lung_nonhema"]:
    f, z, d, m = P[p]
    st = m["hema_stage"].astype(str).to_numpy(); si = np.array([ni[s] for s in st])
    D0 = Tt[np.ix_(si, si)]; np.fill_diagonal(D0, 0)
    same_ruler = bool(np.allclose(D0, d))
    iu = np.triu_indices(len(d), 1); x = arc(z)[iu]
    obs = float(spearmanr(x, d[iu])[0])
    blocks = (m["donor_id"].astype(str) + "|" + m["tissue"].astype(str)).to_numpy()
    groups = [np.where(blocks == g)[0] for g in np.unique(blocks)]
    g = np.random.default_rng(20261002)
    Bn = 2000; null = np.empty(Bn)
    for bb in range(Bn):
        s2 = si.copy()
        for ii in groups:
            if len(ii) > 1:
                s2[ii] = si[g.permutation(ii)]
        Dp = Tt[np.ix_(s2, s2)]
        null[bb] = spearmanr(x, Dp[iu])[0]
    p_hi = (1 + (null >= obs).sum()) / (Bn + 1); p_lo = (1 + (null <= obs).sum()) / (Bn + 1)
    sizes = pd.Series(blocks).value_counts()
    D_[p] = {"rebuilt_ruler_equals_saved": same_ruler, "observed": obs, "B": Bn, "null_mean": float(null.mean()),
             "null_max": float(null.max()), "n_null_ge_obs": int((null >= obs).sum()),
             "p_two_sided_doubled": float(min(1, 2 * min(p_hi, p_lo))), "n_blocks": int(len(groups)),
             "n_singleton_blocks": int((sizes == 1).sum()),
             "v2_saved": {k: J(V2 / "v2_04_permutation_frozen.json")[p][k] for k in ["observed", "null_mean", "null_max", "p_two_sided_doubled", "n_blocks", "n_singleton_blocks"]}}
inp(V2 / "v2_04_permutation_frozen.json")
R["D_frozen_permutation_own_stream"] = D_

# ------------------------------------------------------------------ E internal permutation from saved rows
rows = [json.loads(l) for l in inp(V2 / "refit_pool/results.jsonl").read_text().splitlines() if l.strip()]
perm = [r for r in rows if r["kind"] == "perm"]
null = np.array([r["stat"] for r in perm]); obs = [r for r in rows if r["id"] == "repro_full|H65"][0]["global_rho"]
obs4 = [r for r in rows if r["id"] == "t4|repro_full|H65"][0]["global_rho"]
E = {"B": len(null), "distinct_b": len({r["b"] for r in perm}), "observed_1thread": obs, "observed_4threads": obs4,
     "null_mean": float(null.mean()), "null_max": float(null.max()), "n_null_ge_obs": int((null >= obs).sum()),
     "p_two_sided_doubled": float(min(1, 2 * (1 + (null >= obs).sum()) / (len(null) + 1))),
     "smallest_p_possible_with_B": float(2 / (len(null) + 1)),
     "margin_obs_minus_null_max": float(obs - null.max()),
     "null_values_within_0.001_of_obs": int((null >= obs - 0.001).sum()),
     "perm_threads": sorted({r.get("threads", 1) for r in perm})}
# block permutation reproduction (their algorithm) for b = 0..19: does n_changed match?
mi = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
stg = mi["hema_stage"].astype(str).to_numpy(); blk = (mi["donor_id"].astype(str) + "|" + mi["tissue"].astype(str)).to_numpy()
okc = 0
byb = {r["b"]: r for r in perm}
for bb in range(20):
    g = np.random.default_rng([777, bb]); s = stg.copy()
    for gg in pd.unique(blk):
        ii = np.where(blk == gg)[0]
        if len(ii) > 1:
            s[ii] = stg[g.permutation(ii)]
    okc += int((s != stg).sum() == byb[bb]["n_changed"])
E["n_changed_reproduced_for_b0_19"] = okc
E["blocks_internal"] = int(len(pd.unique(blk))); E["singleton_blocks_internal"] = int((pd.Series(blk).value_counts() == 1).sum())
R["E_internal_permutation"] = E

# ------------------------------------------------------------------ F internal training-donor bootstrap structure
bi = pd.read_csv(inp(V2 / "v2_03_boot_internal_reps.csv"))
F = {"corr_trust_vs_n_train": float(np.corrcoef(bi.trust, bi.n_train)[0, 1]),
     "trust_by_TSP2_copies": {int(k): {"n_reps": int(len(g)), "mean_trust": float(g.trust.mean()), "frac_below_0.80": float((g.trust < 0.8).mean())}
                              for k, g in bi.groupby("n_TSP2_copies")},
     "share_of_below_0.80_reps_without_TSP2": float(((bi.trust < 0.8) & (bi.n_TSP2_copies == 0)).sum() / (bi.trust < 0.8).sum()),
     "branch_plain_by_n_branches": {int(k): {"n_reps": int(len(g)), "mean": float(g.branch.mean())} for k, g in bi.groupby("n_branches")},
     "branch_plain_6_branch_reps_only": {"n": int((bi.n_branches == 6).sum()),
                                        "ci": [float(x) for x in np.percentile(bi.branch[bi.n_branches == 6], [2.5, 97.5])],
                                        "frac_below_0.20": float((bi.branch[bi.n_branches == 6] < 0.2).mean())},
     "branch_aw_6_branch_reps_only": {"ci": [float(x) for x in np.percentile(bi.branch_aw[bi.n_branches == 6], [2.5, 97.5])],
                                     "frac_below_0.20": float((bi.branch_aw[bi.n_branches == 6] < 0.2).mean())}}
lot = J(inp(V2 / "v2_10_internal_lotdo.json"))
F["lotdo_branch_n_scored"] = {u: v["n_branches_scored"] for u, v in lot["per_donor"].items()}
F["lotdo_trust"] = {u: v["trust"] for u, v in lot["per_donor"].items()}
F["lotdo_n_train"] = {u: v["n_train"] for u, v in lot["per_donor"].items()}
# anchor-weighted internal mean re-derived from the per-branch table
pb = pd.read_csv(inp(V2 / "v2_03_internal_per_branch.csv")); pb = pb[(pb.ruler == "H65") & (pb.threads == 4)]
F["internal_anchor_weighted_rederived"] = float((pb.rho * pb.n_anchors).sum() / pb.n_anchors.sum())
F["internal_plain_rederived"] = float(pb.rho.mean())
R["F_internal_bootstrap_structure"] = F

# ------------------------------------------------------------------ G facts
L = np.load(inp(ART / "operators/layer10_head6.npy"))
fa = J(inp(REP / "factor_ablation.json")); pe = pd.read_csv(inp(REP / "factor_ablation_per_endpoint.csv"))
cc = J(inp(REP / "compaction_chain.json"))
sv = np.linalg.svd(L.astype(np.float64), compute_uv=False)
tot = pe.pooled_drop.sum()
G = {"pooled_operator_bytes": int(op["A_early"].nbytes + op["A_mid"].nbytes + op["A_late"].nbytes),
     "L10H6_shape": list(L.shape), "L10H6_bytes": int(L.nbytes), "sparse_values_bytes": 16 * (60 + 60) * 4 + 16 * 4,
     "compaction_chain_mb": {r["label"]: r["mb_size"] for r in cc}}
G["ratio_pooled_over_sparse"] = G["pooled_operator_bytes"] / G["sparse_values_bytes"]
G["ratio_L10H6_over_sparse"] = G["L10H6_bytes"] / G["sparse_values_bytes"]
G["factor_core"] = fa["core_factors"]; G["top4_share_json"] = fa["top4_pct_of_total_impact"]
G["top4_share_csv"] = float(pe.sort_values("pooled_drop", ascending=False).head(4).pooled_drop.sum() / tot)
G["factors0to3_share"] = float(pe[pe.factor.isin([0, 1, 2, 3])].pooled_drop.sum() / tot)
G["singular_values_descending"] = bool(np.all(np.diff(sv) <= 1e-6))
G["branch_bal_acc"] = [fa["intact_metrics"]["branch_balanced_acc"], fa["core_metrics"]["branch_balanced_acc"]]
G["dag_nodes"] = len(nodes); G["dag_edges"] = len(dag["edges"]); G["cell_type_to_stage_entries"] = len(dag["cell_type_to_stage"])
G["dag_branches"] = sorted(set(s2b.values()))
G["internal_branches_present"] = sorted(set(mi["hema_stage"].map(s2b).dropna()))
G["ruler_range"] = {p: [float(np.load(ART / f"anchors/d_target_{p}.npy").min()), float(np.load(ART / f"anchors/d_target_{p}.npy").max())]
                    for p in ["internal", "external", "zeroshot", "lung_nonhema"]}
metas = {p: pd.read_csv(ART / f"anchors/anchor_meta_{p}.csv") for p in ["internal", "external", "zeroshot", "lung_nonhema", "lung_control"]}
don = {p: set(m.donor_id.astype(str)) for p, m in metas.items()}
G["cohort"] = {p: {"anchors": len(m), "donors": len(don[p]), "tissues": int(m.tissue.nunique()),
                   "cells_centroided": int(m.n_cells_centroided.sum()), "top_donor": m.donor_id.value_counts().index[0],
                   "top_donor_share": float(m.donor_id.value_counts().iloc[0] / len(m))} for p, m in metas.items()}
G["zeroshot_donors_subset_external"] = don["zeroshot"] <= don["external"]
G["external_not_in_zeroshot"] = sorted(don["external"] - don["zeroshot"])
G["internal_external_donor_overlap"] = sorted(don["internal"] & don["external"])
G["zeroshot_tissues_in_external"] = [len(set(metas["zeroshot"].tissue)), len(set(metas["zeroshot"].tissue) & set(metas["external"].tissue))]
G["lung_nonhema_donors"] = sorted(don["lung_nonhema"]); G["lung_nonhema_in_internal"] = sorted(don["lung_nonhema"] & don["internal"])
G["external_top5_donor_anchors"] = int(metas["external"].donor_id.value_counts().iloc[:5].sum())
R["G_facts"] = G

# ------------------------------------------------------------------ H report/table consistency
tab = pd.read_csv(inp(V2 / "table_gates.csv")); fig = pd.read_csv(inp(V2 / "fig_gates.csv"))
fb = J(inp(V2 / "v2_01_frozen_donor_bootstrap.json"))
H = {"table_rows": int(len(tab)), "fig_rows": int(len(fig)),
     "external_trust_frozen_boot_n_below_0.80": int(sum(json.loads(l)["trust"] < 0.8 for l in (V2 / "frozen_boot/external.jsonl").read_text().splitlines() if l.strip())),
     "external_trust_frozen_boot_min": float(min(json.loads(l)["trust"] for l in (V2 / "frozen_boot/external.jsonl").read_text().splitlines() if l.strip())),
     "lung_branch_frac_below": fb["lung_nonhema"]["bootstrap"]["branch"]["frac_below_0.2"]}
# every numeric table value that has a twin in a JSON: compare
chk = []
def cmpv(name, a, b, tol=5e-4):
    chk.append({"check": name, "a": a, "b": b, "ok": bool(abs(a - b) <= tol)})
t = tab.set_index(["ordering", "panel", "metric"])
cmpv("ext trust value vs own A", float(t.loc[("H65", "external", "trustworthiness"), "value"]), A["external"]["trust_k15"])
cmpv("ext branch value vs own A", float(t.loc[("H65", "external", "within_branch_corr_plain_mean"), "value"]), A["external"]["branch_plain"])
cmpv("ext branch_aw vs own A", float(t.loc[("H65", "external", "within_branch_corr_anchor_weighted"), "value"]), A["external"]["branch_anchor_weighted"])
cmpv("zs branch_aw vs own A", float(t.loc[("H65", "zero-shot", "within_branch_corr_anchor_weighted"), "value"]), A["zeroshot"]["branch_anchor_weighted"])
cmpv("lung branch_aw vs own A", float(t.loc[("lung_control_random_labels", "lung control", "within_branch_corr_anchor_weighted"), "value"]), A["lung_nonhema"]["branch_anchor_weighted"])
cmpv("internal aw vs re-derived", float(t.loc[("H65", "internal", "branch_holdout_anchor_weighted"), "value"]), F["internal_anchor_weighted_rederived"])
pf_saved = J(V2 / "v2_04_permutation_frozen.json")
for p, lab in [("external", "external"), ("zeroshot", "zero-shot"), ("lung_nonhema", "lung control")]:
    o = "H65" if p != "lung_nonhema" else "lung_control_random_labels"
    if p == "lung_nonhema":
        cmpv(f"{p} perm p vs own stream (B 10,000 vs 2,000)", float(t.loc[(o, lab, "permutation_p_two_sided"), "value"]), D_[p]["p_two_sided_doubled"], tol=0.05)
    else:  # both runs: no null value reaches the observed one, so p is the smallest possible for each B
        cmpv(f"{p} perm: null values >= observed (v2 B=10,000 vs own B=2,000)", float(pf_saved[p]["n_null_ge_obs"]), float(D_[p]["n_null_ge_obs"]), tol=0)
H["value_checks"] = chk
H["all_value_checks_ok"] = all(c["ok"] for c in chk)
R["H_consistency"] = H

R["elapsed_s"] = time.time() - T0
(OUTV / "v2_verify_01_checks.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
(OUTV / "run_config_v2_verify_01.json").write_text(json.dumps({
    "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
    "seeds": {"random anchor subsets (B)": 424242, "frozen permutation own stream (D)": 20261002,
              "replay of v2 donor picks (C)": "default_rng([4242, panel_code, rep]) as in v2_01"},
    "inputs": {str(p): sha(p) for p in sorted(set(INPUTS))}}, indent=2))
print(json.dumps({k: R[k] for k in ["A_frozen_reproduction", "B_trust_vs_k_and_n", "C_copies_in_frozen_bootstrap",
                                     "D_frozen_permutation_own_stream", "E_internal_permutation", "F_internal_bootstrap_structure"]},
                 indent=1, default=str)[:20000])
print(json.dumps(R["G_facts"], indent=1, default=str)[:6000]); print(json.dumps(R["H_consistency"], indent=1, default=str))
print("elapsed", round(time.time() - T0))
