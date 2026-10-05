"""v2b step 3: tables, structured nulls, lung structured control, donor bootstraps, random-projection control.

Usage: python v2b_devorder_03_analyze.py <part>
  part = tables | evalnull | lung | boot_frozen <panel> | boot_internal | n60 | config

Reads outputs/v2b_devorder/pool/results.jsonl (+ heads/), writes JSON/CSV under outputs/v2b_devorder/.
CPU only, no model forward pass.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, beta as beta_dist
from v2b_devorder_common import *  # noqa

RES = OUT / "pool" / "results.jsonl"
LOOK5 = [f"lookup_ct_s{k}" for k in range(5)]
LOOKC = [f"lookup_cls_s{k}" for k in range(3)]
MAIN_REPS = ["maxtoki", "maxtoki_pca64"] + LOOK5 + LOOKC + ["tokenbag_pca64", "hvg_pca64"]
PERM_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"]


def load_results():
    d = {}
    for l in RES.read_text().splitlines():
        if l.strip():
            j = json.loads(l)
            if j.get("error") is None:
                d[j["key"]] = j["res"]
    return d


def nm(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def internal_summary(d, fam, rep, variant, gpart):
    pre = f"{fam}|{rep}|{variant}|"
    ks = [k for k in d if k.startswith(pre)]
    if pre + "full|-" not in d:
        return None
    g = lambda part, f: nm([d[k][f] for k in ks if k.split("|")[3] == part])
    cnt = lambda part, f: int(sum(1 for k in ks if k.split("|")[3] == part and d[k][f] == d[k][f]))
    out = {"trust": d[pre + "full|-"]["trust"]}
    for part, name in (("rand", "random"), ("donor", "donor"), (gpart, gpart)):
        out[name] = g(part, "rho"); out[name + "_diff_ct"] = g(part, "rho_diff_ct")
        out[name + "_n"] = cnt(part, "rho"); out[name + "_diff_ct_n"] = cnt(part, "rho_diff_ct")
    out["per_group"] = {k.split("|")[4]: {"n": d[k]["n"], "rho": d[k]["rho"], "rho_diff_ct": d[k]["rho_diff_ct"]}
                        for k in ks if k.split("|")[3] == gpart}
    out["frozen"] = d[pre + "full|-"].get("frozen", {})
    return out


def passes(x, keys):
    ok = x["trust"] >= GATE_TRUST
    for k in keys:
        ok = ok and (x[k] == x[k]) and x[k] >= GATE_CORR
    return bool(ok)


# ------------------------------------------------------------------------------------------- tables
def part_tables():
    d = load_results()
    rows, per_branch = [], []
    S = {}
    for fam, gpart in (("h65", "branch"), ("h38", "cat"), ("h95", "cat"), ("h103", "cat")):
        for rep in MAIN_REPS:
            for variant in ("pos", "runnull"):
                s = internal_summary(d, fam, rep, variant, gpart)
                if s is None:
                    continue
                S[f"{fam}|{rep}|{variant}"] = s
                r = {"ordering": fam, "rep": rep, "variant": variant, "panel": "internal (head re-fitted)",
                     "trust": s["trust"], "random": s["random"], "donor": s["donor"], "group": s[gpart],
                     "random_diff_ct": s["random_diff_ct"], "donor_diff_ct": s["donor_diff_ct"],
                     "group_diff_ct": s[gpart + "_diff_ct"], "group_n": s[gpart + "_n"],
                     "group_diff_ct_n": s[gpart + "_diff_ct_n"],
                     "passes_4": passes({"trust": s["trust"], "r": s["random"], "d": s["donor"], "g": s[gpart]}, ["r", "d", "g"]),
                     "passes_3_no_group": passes({"trust": s["trust"], "r": s["random"], "d": s["donor"]}, ["r", "d"])}
                rows.append(r)
                for p, e in s["frozen"].items():
                    gk = "branch" if fam == "h65" else "category"
                    rows.append({"ordering": fam, "rep": rep, "variant": variant, "panel": p + " (frozen head)",
                                 "trust": e.get("trust"), "random": e.get("random"), "donor": e.get("donor"),
                                 "group": e.get(gk), "global": e.get("global"),
                                 "random_diff_ct": e.get("random_diff_ct"), "donor_diff_ct": e.get("donor_diff_ct"),
                                 "group_diff_ct": e.get(gk + "_diff_ct"), "global_diff_ct": e.get("global_diff_ct"),
                                 "group_n": e.get(gk + "_n"), "group_diff_ct_n": e.get(gk + "_diff_ct_n"),
                                 "trust_k5": e.get("trust_k5"), "trust_k8": e.get("trust_k8"),
                                 "passes_4": passes({"trust": e["trust"] if e.get("trust") == e.get("trust") else -1,
                                                     "r": e.get("random"), "d": e.get("donor"), "g": e.get(gk)}, ["r", "d", "g"])
                                 if e.get("trust") is not None else None})
                    if fam == "h65" and variant == "pos":
                        for g, v in e.get("branch_per_group", {}).items():
                            per_branch.append({"rep": rep, "panel": p, "branch": g, **v})
                if fam == "h65" and variant == "pos":
                    for g, v in s["per_group"].items():
                        per_branch.append({"rep": rep, "panel": "internal (held out)", "branch": g, **v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_gates_by_representation.csv", index=False)
    pb = pd.DataFrame(per_branch)
    pb.to_csv(OUT / "table_h65_per_branch.csv", index=False)
    # lookup seed summaries
    look = {}
    for fam in ("h65", "h38", "h95", "h103"):
        for grp, reps in (("lookup_ct", LOOK5), ("lookup_cls", LOOKC)):
            sub = df[(df.ordering == fam) & (df.rep.isin(reps)) & (df.variant == "pos")]
            if len(sub) == 0:
                continue
            for panel, ss in sub.groupby("panel"):
                look[f"{fam}|{grp}|{panel}"] = {c: {"mean": float(ss[c].mean()), "min": float(ss[c].min()),
                                                    "max": float(ss[c].max()), "n_seeds": int(ss[c].notna().sum())}
                                                for c in ["trust", "random", "donor", "group", "global", "group_diff_ct",
                                                          "global_diff_ct", "random_diff_ct", "donor_diff_ct"]
                                                if c in ss and ss[c].notna().any()}
                look[f"{fam}|{grp}|{panel}"]["n_seeds_pass_4"] = int(ss["passes_4"].fillna(False).sum())
    # ruler structure per branch: distinct ruler values among pairs of DIFFERENT classes / cell types
    struct = {}
    for p in BLOOD:
        m = meta(p); D = np.load(ART / f"anchors/d_target_{p}.npy")
        br = branches_of(m["hema_stage"].astype(str)); ct = m["cell_type"].astype(str).to_numpy()
        cl = np.array([a + "|" + b for a, b in anchor_classes(m)])
        for g in sorted(set(br)):
            idx = np.where(br == g)[0]
            if len(idx) < 3:
                continue
            iu = np.triu_indices(len(idx), 1); a = idx[iu[0]]; b = idx[iu[1]]
            dif_cls = cl[a] != cl[b]; dif_ct = ct[a] != ct[b]
            struct[f"{p}|{g}"] = {"n_anchors": int(len(idx)), "stages": sorted(set(m["hema_stage"].iloc[idx])),
                                  "ruler_values_all_pairs": sorted(set(D[a, b].tolist())),
                                  "ruler_values_diff_class_pairs": sorted(set(D[a, b][dif_cls].tolist())),
                                  "ruler_values_diff_ct_pairs": sorted(set(D[a, b][dif_ct].tolist())),
                                  "n_pairs": int(len(a)), "n_diff_ct_pairs": int(dif_ct.sum())}
    # structured null (refit)
    null = {}
    for rep in PERM_REPS:
        obs = S.get(f"h65|{rep}|pos")
        if obs is None:
            continue
        draws = sorted({int(k.split("|")[2][4:]) for k in d if k.startswith(f"h65|{rep}|perm") and k.endswith("|full|-")})
        vals = {}
        for dd in draws:
            fu = d[f"h65|{rep}|perm{dd}|full|-"]
            v = {"trust": fu["trust"]}
            for p in ("external", "zeroshot"):
                e = fu["frozen"][p]
                for k in ("trust", "branch", "branch_diff_ct", "global", "global_diff_ct", "random", "donor"):
                    v[f"{p}_{k}"] = e[k]
            bk = [k for k in d if k.startswith(f"h65|{rep}|perm{dd}|branch|")]
            if bk:
                v["internal_branch"] = nm([d[k]["rho"] for k in bk])
                v["internal_branch_diff_ct"] = nm([d[k]["rho_diff_ct"] for k in bk])
                v["internal_branch_n"] = len(bk)
            vals[dd] = v
        o = {"trust": obs["trust"], "internal_branch": obs["branch"], "internal_branch_diff_ct": obs["branch_diff_ct"]}
        for p in ("external", "zeroshot"):
            for k in ("trust", "branch", "branch_diff_ct", "global", "global_diff_ct", "random", "donor"):
                o[f"{p}_{k}"] = obs["frozen"][p][k]
        res = {}
        for k, ov in o.items():
            nv = np.array([v[k] for v in vals.values() if k in v and v[k] == v[k]], float)
            if len(nv) == 0:
                continue
            res[k] = {"observed": ov, "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)),
                      "null_min": float(nv.min()), "null_max": float(nv.max()), "n_draws": int(len(nv)),
                      "p_one_sided": float((1 + (nv >= ov).sum()) / (1 + len(nv))),
                      "z": float((ov - nv.mean()) / nv.std(ddof=1)) if nv.std() > 0 else None,
                      "excess_over_null_mean": float(ov - nv.mean()),
                      "null_pass_rate_gate": float(np.mean(nv >= (GATE_TRUST if "trust" in k else GATE_CORR)))}
        null[rep] = res
    # H38 / H95 structured null (refit of the full head only; frozen external / zero-shot gates)
    catnull = {}
    for fam in ("h38", "h95"):
        for rep in ("maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"):
            obs = S.get(f"{fam}|{rep}|pos")
            draws = sorted({int(k.split("|")[2][4:]) for k in d if k.startswith(f"{fam}|{rep}|perm") and k.endswith("|full|-")})
            if obs is None or not draws:
                continue
            o = {"internal_trust": obs["trust"]}
            for p in ("external", "zeroshot"):
                for k in ("trust", "random", "donor", "category", "category_diff_ct", "global", "global_diff_ct"):
                    o[f"{p}_{k}"] = obs["frozen"][p][k]
            res = {}
            for k, ov in o.items():
                nv = []
                for dd in draws:
                    fu = d[f"{fam}|{rep}|perm{dd}|full|-"]
                    v = fu["trust"] if k == "internal_trust" else fu["frozen"][k.split("_", 1)[0]][k.split("_", 1)[1]]
                    if v == v:
                        nv.append(v)
                nv = np.array(nv, float)
                if len(nv) == 0 or ov != ov:
                    continue
                res[k] = {"observed": ov, "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)),
                          "null_max": float(nv.max()), "n_draws": int(len(nv)),
                          "p_one_sided": float((1 + (nv >= ov).sum()) / (1 + len(nv))),
                          "null_pass_rate_gate": float(np.mean(nv >= (GATE_TRUST if "trust" in k else GATE_CORR)))}
            catnull[f"{fam}|{rep}"] = res
    out = {"summaries": S, "lookup_seed_summary": look, "ruler_structure": struct, "structured_null_refit": null,
           "structured_null_refit_h38_h95": catnull}
    (OUT / "v2b_tables.json").write_text(json.dumps(out, indent=1, default=float))
    print("tables written; rows", len(df))


# ------------------------------------------------------------------------------------------- eval-only null
def part_evalnull(n_draws=2000):
    """Internal branch-holdout: heads fitted without the held-out branch on the TRUE ruler (saved z), held-out
    ruler re-drawn from the class -> stage map permuted within branch. 2,000 draws, default_rng([2027, d])."""
    d = load_results()
    m = meta("internal"); cls = h65_classes()
    acls = anchor_classes(m); ct = m["cell_type"].astype(str).to_numpy()
    reps = [r for r in MAIN_REPS if any(k.startswith(f"h65|{r}|pos|branch|") for k in d)]
    held = {}
    for r in reps:
        held[r] = {}
        for k in d:
            if k.startswith(f"h65|{r}|pos|branch|"):
                g = k.split("|")[4]; t = np.array(d[k]["test_idx"]); z = np.array(d[k]["z"], dtype=np.float32)
                Dh = arccos_dist(z); iu = np.triu_indices(len(t), 1)
                held[r][g] = (t, Dh[iu], iu, (ct[t][iu[0]] != ct[t][iu[1]]))
    def score(stages):
        D = h65_ruler(stages)
        out = {}
        for r in reps:
            va, vd = [], []
            for g, (t, x, iu, dif) in held[r].items():
                y = D[np.ix_(t, t)][iu]
                if np.ptp(y) > 0:
                    va.append(spearmanr(x, y)[0])
                if dif.sum() >= 2 and np.ptp(y[dif]) > 0:
                    vd.append(spearmanr(x[dif], y[dif])[0])
            out[r] = (nm(va), nm(vd))
        return out
    obs = score(m["hema_stage"].astype(str).tolist())
    nulls = {r: [] for r in reps}
    for dd in range(n_draws):
        mp_ = permuted_stage_map(cls, np.random.default_rng([2027, dd]))
        sc = score([mp_[c] for c in acls])
        for r in reps:
            nulls[r].append(sc[r])
    res = {}
    for r in reps:
        a = np.array(nulls[r], float)
        res[r] = {}
        for j, name in enumerate(("branch_all_pairs", "branch_diff_ct")):
            nv = a[:, j]; nv = nv[~np.isnan(nv)]; ov = obs[r][j]
            res[r][name] = {"observed": ov, "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)),
                            "p_one_sided": float((1 + (nv >= ov).sum()) / (1 + len(nv))), "n_draws": int(len(nv)),
                            "null_q95": float(np.percentile(nv, 95))}
    (OUT / "v2b_evalnull_internal_branch.json").write_text(json.dumps(
        {"method": "eval-only structured null; heads fixed (fitted without the held-out branch on the true H65 ruler); "
                   "held-out ruler from the class->stage map permuted within branch; class = (cell_type, hema_stage) "
                   "over all blood panels; p = (1 + #null >= obs) / (1 + N)", "rng": "default_rng([2027, d])",
         "n_draws": n_draws, "results": res}, indent=1, default=float))
    for r in reps:
        print(r, {k: (round(v["observed"], 3), round(v["null_mean"], 3), round(v["p_one_sided"], 4)) for k, v in res[r].items()})


# ------------------------------------------------------------------------------------------- lung structured control
def part_lung(design_sel, n_draws=2000):
    """Frozen H65 head of each representation on the 50 lung non-blood anchors. Deployed design: one random stage
    per ANCHOR (as phase1a_lung_nonhema_panel.py:91). Structured control: one random stage per lung CELL TYPE, so
    anchors of the same cell type share a stage. Both drawn uniformly from the 34 stage nodes, 2,000 draws each
    (rng default_rng([6161, d]) and default_rng([6262, d]))."""
    d = load_results()
    m = meta("lung_nonhema"); ct = m["cell_type"].astype(str).to_numpy(); donors = m["donor_id"].astype(str).to_numpy()
    uct = np.unique(ct)
    reps = [r for r in MAIN_REPS if (HEADS / f"z_h65_{r}_pos_lung_nonhema.npy").exists()]
    Z = {r: np.load(HEADS / f"z_h65_{r}_pos_lung_nonhema.npy") for r in reps}
    Dh = {r: arccos_dist(Z[r]) for r in reps}
    trust = {r: d[f"h65|{r}|pos|full|-"]["frozen"]["lung_nonhema"]["trust"] for r in reps}
    rng0 = np.random.default_rng(SEED); n = len(m)
    splits = []
    for _ in range(20):
        idx = rng0.permutation(n); splits.append(np.array(sorted(idx[:max(2, int(round(n * 0.2)))])))
    def gates(r, stages):
        D = h65_ruler(stages); X = Dh[r]
        def rho(ii):
            iu = np.triu_indices(len(ii), 1); y = D[np.ix_(ii, ii)][iu]
            return spearmanr(X[np.ix_(ii, ii)][iu], y)[0] if np.ptp(y) > 0 else np.nan
        rnd = nm([rho(s) for s in splits])
        don = nm([rho(np.where(donors == g)[0]) for g in np.unique(donors) if (donors == g).sum() >= 3])
        br = branches_of(stages)
        bra = nm([rho(np.where(br == g)[0]) for g in np.unique(br) if g != "_unk" and (br == g).sum() >= 3])
        iu = np.triu_indices(n, 1)
        glo = spearmanr(X[iu], D[iu])[0]
        return rnd, don, bra, glo
    res = {}
    for design, tag in (("per_anchor_random (deployed)", 6161), ("per_cell_type_random (structured)", 6262)):
        if str(tag) != design_sel:
            continue
        res[design] = {}
        vals = {r: [] for r in reps}
        for dd in range(n_draws):
            rng = np.random.default_rng([tag, dd])
            if tag == 6161:
                stages = list(rng.choice(STAGE_NODES, size=n))
            else:
                mp_ = dict(zip(uct, rng.choice(STAGE_NODES, size=len(uct))))
                stages = [mp_[c] for c in ct]
            for r in reps:
                vals[r].append(gates(r, stages))
        for r in reps:
            a = np.array(vals[r], float)
            rnd, don, bra, glo = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            t_ok = trust[r] >= GATE_TRUST
            res[design][r] = {
                "trust": trust[r], "trust_passes": bool(t_ok),
                "mean": {"random": float(np.nanmean(rnd)), "donor": float(np.nanmean(don)), "branch": float(np.nanmean(bra)),
                         "global": float(np.nanmean(glo))},
                "pass_rate": {"random": float(np.mean(rnd >= GATE_CORR)), "donor": float(np.mean(don >= GATE_CORR)),
                              "branch": float(np.mean(bra >= GATE_CORR)),
                              "random_donor_branch": float(np.mean((rnd >= GATE_CORR) & (don >= GATE_CORR) & (bra >= GATE_CORR))),
                              "all_four": float(np.mean((rnd >= GATE_CORR) & (don >= GATE_CORR) & (bra >= GATE_CORR)) if t_ok else 0.0)},
                "branch_ge_external_maxtoki_0.346": float(np.mean(bra >= 0.346443)),
                "branch_nan_draws": int(np.isnan(bra).sum())}
            print(design, r, json.dumps(res[design][r]["pass_rate"]), round(res[design][r]["mean"]["branch"], 3), flush=True)
    (OUT / f"v2b_lung_control_{design_sel}.json").write_text(json.dumps({"n_draws": n_draws, "results": res}, indent=1, default=float))


# ------------------------------------------------------------------------------------------- donor bootstraps
def grp_rho(X, Y, idx, groups, ct, diff):
    """Mean over groups (>=3 members) of Spearman on within-group pairs; copies of one anchor never paired."""
    vals = []
    for g in pd.unique(groups):
        if g == "_unk":
            continue
        mm = np.where(groups == g)[0]
        if len(mm) < 3:
            continue
        iu = np.triu_indices(len(mm), 1); a = mm[iu[0]]; b = mm[iu[1]]
        keep = idx[a] != idx[b]
        if diff:
            keep &= ct[a] != ct[b]
        if keep.sum() < 2:
            continue
        x = X[idx[a[keep]], idx[b[keep]]]; y = Y[idx[a[keep]], idx[b[keep]]]
        if np.ptp(y) == 0 or np.ptp(x) == 0:
            continue
        vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return float(np.mean(vals)) if vals else np.nan


def glob_rho(X, Y, idx, ct, diff, sub=None):
    n = len(idx)
    iu = np.triu_indices(n, 1)
    keep = idx[iu[0]] != idx[iu[1]]
    if diff:
        keep &= ct[iu[0]] != ct[iu[1]]
    a = idx[iu[0][keep]]; b = idx[iu[1][keep]]
    return float(np.corrcoef(rankdata(X[a, b]), rankdata(Y[a, b]))[0, 1])


BOOT_REPS = ["maxtoki"] + LOOK5 + ["tokenbag_pca64", "hvg_pca64"]


def contrasts(vals):
    """vals: dict rep -> value. Returns MaxToki minus lookup (mean of 5 seeds), minus token bag, minus HVG."""
    lk = np.nanmean([vals[r] for r in LOOK5])
    return {"maxtoki": vals["maxtoki"], "lookup_ct_mean5": lk, "tokenbag_pca64": vals["tokenbag_pca64"],
            "hvg_pca64": vals["hvg_pca64"],
            "maxtoki_minus_lookup": vals["maxtoki"] - lk, "maxtoki_minus_tokenbag": vals["maxtoki"] - vals["tokenbag_pca64"],
            "maxtoki_minus_hvg": vals["maxtoki"] - vals["hvg_pca64"]}


def summarize_boot(obs, reps_list):
    out = {}
    for k in obs:
        a = np.array([r[k] for r in reps_list], float); a = a[~np.isnan(a)]
        out[k] = {"observed": obs[k], "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                  "boot_mean": float(a.mean()), "n_valid": int(len(a)),
                  "share_le_0": float(np.mean(a <= 0)) if "minus" in k else None}
    return out


def part_boot_frozen(panel, n_boot=2000, with_global=True):
    m = meta(panel); ct = m["cell_type"].astype(str).to_numpy(); donors = m["donor_id"].astype(str).to_numpy()
    br = branches_of(m["hema_stage"].astype(str)); D = np.load(ART / f"anchors/d_target_{panel}.npy")
    X = {r: arccos_dist(np.load(HEADS / f"z_h65_{r}_pos_{panel}.npy")) for r in BOOT_REPS}
    ud = np.array(sorted(set(donors))); mem = {u: np.where(donors == u)[0] for u in ud}
    metrics = ["branch", "branch_diff_ct"] + (["global", "global_diff_ct"] if with_global else [])
    if with_global == "only":
        metrics = ["global", "global_diff_ct"]
    def one(idx):
        out = {}
        for mt in metrics:
            vals = {}
            for r in BOOT_REPS:
                if mt.startswith("branch"):
                    vals[r] = grp_rho(X[r], D, idx, br[idx], ct[idx], mt.endswith("diff_ct"))
                else:
                    vals[r] = glob_rho(X[r], D, idx, ct[idx], mt.endswith("diff_ct"))
            for k, v in contrasts(vals).items():
                out[f"{mt}|{k}"] = v
        return out
    obs = one(np.arange(len(m)))
    reps_list = []
    T0 = time.time()
    for b in range(n_boot):
        rng = np.random.default_rng([8080, BLOOD.index(panel), b])
        pick = rng.choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([mem[u] for u in pick])
        reps_list.append(one(idx))
    res = summarize_boot(obs, reps_list)
    (OUT / f"v2b_boot_{panel}{'_global' if with_global == 'only' else ''}.json").write_text(json.dumps(
        {"method": "donor cluster bootstrap (donors drawn with replacement, all their anchors kept), frozen heads, "
                   "copies of one anchor never paired; percentile 2.5/97.5; lookup = mean of 5 seeds per replicate",
         "rng": f"default_rng([8080, {BLOOD.index(panel)}, b])", "n_boot": n_boot, "n_donors": int(len(ud)),
         "seconds": time.time() - T0, "results": res}, indent=1, default=float))
    for k, v in res.items():
        if "minus" in k:
            print(panel, k, round(v["observed"], 3), np.round(v["ci95"], 3), "share<=0", round(v["share_le_0"], 3))


def _wsetup(x, y):
    ox = np.argsort(x, kind="stable"); xs = x[ox]
    gx = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1]
    gid = np.repeat(np.arange(len(gx)), np.diff(np.r_[gx, len(xs)]))
    uy, yinv = np.unique(y, return_inverse=True)
    return ox, gx, gid, yinv, len(uy)


def _wspearman(w, st):
    """Spearman (average ranks) of the sample in which pair p appears w[p] times; exact, no expansion."""
    ox, gx, gid, yinv, ny = st
    ws = w[ox]
    gsum = np.add.reduceat(ws, gx)
    rank_g = (np.cumsum(gsum) - gsum) + (gsum + 1) / 2
    rx = np.empty_like(ws); rx[ox] = rank_g[gid]
    wy = np.bincount(yinv, weights=w, minlength=ny)
    ry = ((np.cumsum(wy) - wy) + (wy + 1) / 2)[yinv]
    W = w.sum(); mx = (w * rx).sum() / W; my = (w * ry).sum() / W
    return float((w * (rx - mx) * (ry - my)).sum() / np.sqrt((w * (rx - mx) ** 2).sum() * (w * (ry - my) ** 2).sum()))


def part_boot_global(panel, n_boot=2000):
    """Global Spearman (all anchor pairs; and pairs of different cell types) under the same donor bootstrap as
    part_boot_frozen (same rng streams). A drawn donor's anchors appear k times; pair (i, j) of distinct anchors then
    appears k_d(i) * k_d(j) times and copies of one anchor are never paired, which is what the weights encode."""
    m = meta(panel); ct = m["cell_type"].astype(str).to_numpy(); donors = m["donor_id"].astype(str).to_numpy()
    D = np.load(ART / f"anchors/d_target_{panel}.npy")
    n = len(m); iu = np.triu_indices(n, 1); dif = ct[iu[0]] != ct[iu[1]]
    ud = np.array(sorted(set(donors))); dix = np.searchsorted(ud, donors)
    ST = {}
    for r in BOOT_REPS:
        X = arccos_dist(np.load(HEADS / f"z_h65_{r}_pos_{panel}.npy"))[iu]
        ST[(r, "global")] = _wsetup(X, D[iu]); ST[(r, "global_diff_ct")] = _wsetup(X[dif], D[iu][dif])
    def one(k):
        w_all = (k[dix[iu[0]]] * k[dix[iu[1]]]).astype(float)
        out = {}
        for mt, w in (("global", w_all), ("global_diff_ct", w_all[dif])):
            vals = {r: _wspearman(w, ST[(r, mt)]) for r in BOOT_REPS}
            for kk, v in contrasts(vals).items():
                out[f"{mt}|{kk}"] = v
        return out
    obs = one(np.ones(len(ud), dtype=int))
    reps_list = []; T0 = time.time()
    for b in range(n_boot):
        rng = np.random.default_rng([8080, BLOOD.index(panel), b])
        pick = rng.choice(ud, size=len(ud), replace=True)
        k = np.array([(pick == u).sum() for u in ud])
        reps_list.append(one(k))
    res = summarize_boot(obs, reps_list)
    (OUT / f"v2b_boot_{panel}_global.json").write_text(json.dumps(
        {"method": "donor cluster bootstrap, global Spearman via pair weights k_d(i)*k_d(j) (identical to expanding "
                   "the resample; copies of one anchor never paired); same rng as v2b_boot_<panel>.json; percentile 2.5/97.5",
         "rng": f"default_rng([8080, {BLOOD.index(panel)}, b])", "n_boot": n_boot, "seconds": time.time() - T0,
         "results": res}, indent=1, default=float))
    for kk, v in res.items():
        if "minus" in kk or kk.endswith("maxtoki"):
            print(panel, kk, round(v["observed"], 4), np.round(v["ci95"], 4), v["share_le_0"])


def part_boot_internal(n_boot=2000):
    """Internal branch-holdout: held-out heads fixed (fitted once, all training donors); evaluation donors resampled.
    This interval covers evaluation-donor variation only, not re-fitting."""
    d = load_results()
    m = meta("internal"); ct = m["cell_type"].astype(str).to_numpy(); donors = m["donor_id"].astype(str).to_numpy()
    br = branches_of(m["hema_stage"].astype(str)); D = np.load(ART / "anchors/d_target_internal.npy")
    n = len(m)
    X = {}
    for r in BOOT_REPS:
        Xr = np.full((n, n), np.nan)
        for k in d:
            if k.startswith(f"h65|{r}|pos|branch|"):
                t = np.array(d[k]["test_idx"]); Dh = arccos_dist(np.array(d[k]["z"], dtype=np.float32))
                Xr[np.ix_(t, t)] = Dh
        X[r] = Xr
    held = set(k.split("|")[4] for k in d if k.startswith("h65|maxtoki|pos|branch|"))
    brm = np.where(np.isin(br, list(held)), br, "_unk")
    ud = np.array(sorted(set(donors))); mem = {u: np.where(donors == u)[0] for u in ud}
    def one(idx):
        out = {}
        for mt in ("branch", "branch_diff_ct"):
            vals = {r: grp_rho(X[r], D, idx, brm[idx], ct[idx], mt.endswith("diff_ct")) for r in BOOT_REPS}
            for k, v in contrasts(vals).items():
                out[f"{mt}|{k}"] = v
        return out
    obs = one(np.arange(n))
    reps_list = []
    for b in range(n_boot):
        rng = np.random.default_rng([8080, 9, b])
        pick = rng.choice(ud, size=len(ud), replace=True)
        reps_list.append(one(np.concatenate([mem[u] for u in pick])))
    res = summarize_boot(obs, reps_list)
    (OUT / "v2b_boot_internal_branch.json").write_text(json.dumps(
        {"method": "internal branch-holdout with held-out heads fixed; evaluation donors drawn with replacement; "
                   "copies never paired; plain mean over scored branches; percentile 2.5/97.5",
         "rng": "default_rng([8080, 9, b])", "n_boot": n_boot, "results": res}, indent=1, default=float))
    for k, v in res.items():
        print("internal", k, round(v["observed"], 3), np.round(v["ci95"], 3), v["share_le_0"])


# ------------------------------------------------------------------------------------------- N60
def part_n60():
    d = load_results()
    def comp(variant, arg0):
        pre = f"n60|{variant}|{arg0}|"
        ks = [k for k in d if k.startswith(pre)]
        if pre + "full|-" not in d:
            return None
        tr = d[pre + "full|-"]["trust"]
        rnd = nm([d[k]["rho"] for k in ks if k.split("|")[3] == "rand"])
        bra = nm([d[k]["rho"] for k in ks if k.split("|")[3] == "branch"])
        brd = nm([d[k]["rho_diff_ct"] for k in ks if k.split("|")[3] == "branch"])
        nb = sum(1 for k in ks if k.split("|")[3] == "branch"); nr = sum(1 for k in ks if k.split("|")[3] == "rand")
        if nb < 6 or nr < 3:
            return None
        return {"trust": tr, "random": rnd, "branch": bra, "branch_diff_ct": brd,
                "composite": 0.5 * tr + 0.25 * rnd + 0.25 * bra}
    head = comp("head", "11_6")
    r11 = [c for c in (comp("rand", f"11_{s}") for s in range(30)) if c]
    r1 = [c for c in (comp("rand", f"1_{s}") for s in range(10)) if c]
    screen = pd.read_csv(REP / "head_layer_screen.csv")
    out = {"L10H6_recomputed": head,
           "L10H6_deployed": screen[(screen["layer"] == 10) & (screen["head"] == 6)].iloc[0][
               ["trustworthiness", "random_holdout", "branch_holdout", "score"]].to_dict(),
           "screen_88_heads": {"composite_mean": float(screen.score.mean()), "composite_max": float(screen.score.max()),
                               "branch_mean": float(screen.branch_holdout.mean()), "trust_mean": float(screen.trustworthiness.mean())}}
    for name, R in (("random_154_proj_of_centroid11", r11), ("random_154_proj_of_centroid1", r1)):
        if not R:
            continue
        df = pd.DataFrame(R)
        o = {"n": int(len(df))}
        for k in ("trust", "random", "branch", "branch_diff_ct", "composite"):
            v = df[k].to_numpy()
            o[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
            if head:
                c = head[k]
                n_below = int((v < c).sum()); N = len(v)
                F = n_below / N
                lo = beta_dist.ppf(0.025, n_below, N - n_below + 1) if n_below > 0 else 0.0
                hi = beta_dist.ppf(0.975, n_below + 1, N - n_below) if n_below < N else 1.0
                o[k]["L10H6_value"] = c
                o[k]["share_random_below_L10H6"] = F
                o[k]["share_random_ge_L10H6"] = 1 - F
                o[k]["P_best_of_88_random_ge_L10H6"] = float(1 - F ** 88)
                o[k]["P_best_of_88_random_ge_L10H6_interval"] = [float(1 - hi ** 88), float(1 - lo ** 88)]
                o[k]["P_best_of_8_random_ge_L10H6"] = float(1 - F ** 8)
        out[name] = o
        out[name + "_draws"] = R
    out["method"] = ("phase9 scoring (features standardised per column; LET head; trust k=15 in-sample; random holdout "
                     "3 splits; branch holdout over 6 scorable branches; composite 0.5*trust + 0.25*random + 0.25*branch). "
                     "Random projections: Gaussian 154 x 1232, default_rng([6060, centroid index, seed]). "
                     "Best-of-88: P(max of 88 independent random draws >= L10H6) = 1 - F^88, F = share of random draws "
                     "below L10H6; interval from a Clopper-Pearson 95% interval on F.")
    (OUT / "v2b_n60_random_projection.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_draws")}, indent=1, default=float)[:4000])


def part_n60dim():
    """Why does L10H6 have higher trust? Effective dimension (participation ratio of the eigenvalues of the
    correlation matrix of the standardised 154 features) of the L10H6 slice vs the random projections."""
    import v2b_devorder_02_pool as POOL
    POOL.C()
    def pr(F):
        ev = np.clip(np.linalg.eigvalsh(np.cov(F.T)), 0, None)
        return float(ev.sum() ** 2 / (ev ** 2).sum()), float(ev[::-1][:10].sum() / ev.sum())
    out = {"L10H6": pr(POOL.n60_features("head", "11_6"))}
    r = [pr(POOL.n60_features("rand", f"11_{s}")) for s in range(30)]
    out["random_centroid11"] = {"pr_mean": float(np.mean([x[0] for x in r])), "pr_min": float(np.min([x[0] for x in r])),
                                "pr_max": float(np.max([x[0] for x in r])),
                                "top10_share_mean": float(np.mean([x[1] for x in r]))}
    out["note"] = "pr = participation ratio (sum ev)^2 / sum ev^2; top10_share = share of variance in the 10 largest eigenvalues"
    (OUT / "v2b_n60_effective_dimension.json").write_text(json.dumps(out, indent=1))
    print(out)


def part_config():
    files = [RES, OUT / "pool" / "task_list.json", OUT / "thread_check" / "results.jsonl", OUT / "features_manifest.json"] + \
        sorted(HEADS.glob("*")) + sorted(FEAT.glob("*.npy")) + sorted(OUT.glob("v2b_*.json")) + sorted(OUT.glob("table_*.csv"))
    scripts = sorted(f for f in SCR.glob("v2b_devorder_*.py") if not f.name.startswith("._"))
    inputs = [ART / f"anchors/{f}" for f in ["centroids_internal.npy", "centroids_external.npy", "centroids_zeroshot.npy",
                                             "centroids_lung_nonhema.npy", "d_target_internal.npy", "d_target_external.npy",
                                             "d_target_zeroshot.npy", "d_target_lung_nonhema.npy",
                                             "d_target_internal_null_shuffled.npy", "d_target_external_null_shuffled.npy",
                                             "d_target_zeroshot_null_shuffled.npy", "anchor_meta_internal.csv",
                                             "anchor_meta_external.csv", "anchor_meta_zeroshot.csv", "anchor_meta_lung_nonhema.csv"]]
    inputs += [ART / "operators/pooled_drift_components.npz", ART / "operators/operator_index.json",
               PLAN / "h65_stage_dag.json", REP / "quality_gates_spec.json", REP / "head_layer_screen.csv",
               REP / "quality_gates_let_anchor.json", REP / "h38_lite_quality_gates.json",
               REP / "hypothesis_registry_3gate.json", REP / "hypothesis_registry_sweep2.json"]
    inputs += [SCR / f for f in ["phase5_let_anchor.py", "phase7_external_validation.py", "phase8_zeroshot_transfer.py",
                                 "phase9_head_attribution.py", "phase13_h38_lite.py", "phase3a_manifold_sweep.py",
                                 "phase3a_revaluate_3gate.py", "phase3a_sweep2_ordinal.py", "phase15_validate_candidate.py",
                                 "phase1bc_hidden_states_and_centroids.py", "phase1a_lung_nonhema_panel.py"]]
    inputs += [ART / f"operators/layer{l:02d}_head{h}.npy" for l in [10] for h in [6]]
    PH1 = RUN / "outputs/phase1"
    inputs += [PH1 / f"cells_{p}_obs.csv" for p in PANELS] + [PH1 / f"anchors_{p}.csv" for p in PANELS] + \
              [PH1 / f"cells_{p}.npz" for p in PANELS]
    cfg = {"item": "D8b developmental ordering beyond cell-type identity",
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "inputs_sha256": {str(p): sha256(p) for p in inputs},
           "scripts_sha256": {str(p): sha256(p) for p in scripts},
           "outputs_sha256": {str(p): sha256(p) for p in files if p.is_file() and not p.name.startswith("._")},
           "raw_h5ad_not_hashed": {"tabula_sapiens_immune.h5ad": 19773999714, "tabula_sapiens_lung.h5ad": 3196847019},
           "seeds": {"LET head": "torch seed 42 inside phase5 train_let", "torch_threads": 1,
                     "lookup_ct": "codes default_rng([7001,k]) k=0..4; noise default_rng([7001,k,panel])",
                     "lookup_cls": "codes default_rng([7002,k]) k=0..2; noise default_rng([7002,k,panel])",
                     "pca": "sklearn PCA random_state 42",
                     "random holdout splits": "default_rng(42) as phase5",
                     "H65 structured null (refit)": "default_rng([2026, d]) d=0..39 (branch holdout d=0..19)",
                     "H65 structured null (eval-only)": "default_rng([2027, d]) d=0..1999",
                     "H38/H95 structured null": "default_rng([3838|9595, d]) d=0..29",
                     "lung per-anchor / per-cell-type": "default_rng([6161, d]) / default_rng([6262, d]) d=0..1999",
                     "donor bootstrap": "default_rng([8080, panel_index, b]) b=0..1999 (internal panel index 9)",
                     "random projections": "default_rng([6060, centroid_index, seed])"},
           "hardware": "CPU only; no model forward pass; torch 1 thread per worker"}
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print("run_config.json:", len(cfg["inputs_sha256"]), "inputs,", len(cfg["outputs_sha256"]), "outputs,", len(scripts), "scripts")


from scipy.stats import spearmanr  # noqa: E402

if __name__ == "__main__":
    part = sys.argv[1]
    if part == "tables":
        part_tables()
    elif part == "evalnull":
        part_evalnull()
    elif part == "lung":
        part_lung(sys.argv[2])
    elif part == "boot_frozen":
        wg = True
        if len(sys.argv) > 4:
            wg = {"noglobal": False, "globalonly": "only"}.get(sys.argv[4], True)
        part_boot_frozen(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000, with_global=wg)
    elif part == "boot_global":
        part_boot_global(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif part == "boot_internal":
        part_boot_internal()
    elif part == "n60":
        part_n60()
    elif part == "n60dim":
        part_n60dim()
    elif part == "config":
        part_config()
