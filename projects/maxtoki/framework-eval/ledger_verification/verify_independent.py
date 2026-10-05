"""Independent re-check of the error ledger and the Study C pre-audit snapshot.

Written by the verifier of item C-prep ('ledger'). It does NOT reuse the
first agent's code paths: every key number is recomputed a different way
(other seed, other resampling routine, other parser, other pair of ratios).

Reads only saved files. CPU only. No model forward pass. Writes
verify_independent_results.json and run_config.json next to this script.

Seeds: bootstrap seed 777 (first agent used 20261001); curveball seeds are
the deployed ones (42 + trial), because the point is to reproduce the
deployed null exactly.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MT = Path("<REPO_ROOT>/projects/maxtoki")
R = MT / "runs"
S = MT / "summaries"
FE = MT / "framework-eval"
STUDYC = Path("<EVAL_ROOT>/studyC")
SNAP = STUDYC / "snapshot" / "maxtoki"
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
OUT = Path(__file__).resolve().parent / "verify_independent_results.json"
CFG = Path(__file__).resolve().parent / "run_config.json"
SEED = 777
N_BOOT = 2000
CUTOFF = dt.datetime(2026, 5, 7, 16, 57, 0).timestamp()

INPUTS: dict[str, str] = {}
RES: dict[str, dict] = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def reg(p: Path) -> Path:
    if str(p) not in INPUTS:
        INPUTS[str(p)] = sha(p)
    return p


def run(cid, fn):
    t0 = time.time()
    try:
        out = fn()
    except Exception as e:  # keep going
        out = {"error": repr(e)}
    out["seconds"] = round(time.time() - t0, 1)
    RES[cid] = out
    print(cid, json.dumps(out, default=str)[:600], flush=True)


# ---------------------------------------------------------------- C1 CRISPRi
def c1():
    p = reg(R / "circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet")
    df = pd.read_parquet(p)
    pred = df["predicted_inhibitory"].astype(bool).to_numpy()
    lfc = df["actual_lfc"].to_numpy(float)
    obs = lfc < 0  # derive the label from the raw log fold change, not the stored label
    stored_obs_agrees = bool((obs == df["actual_inhibitory"].astype(bool).to_numpy()).all())
    stored_correct_agrees = bool(((pred == obs) == df["correct"].astype(bool).to_numpy()).all())
    n = len(df)
    acc = float(np.mean(pred == obs))
    p_dec = float(obs.mean())
    sens = float(np.mean(pred[obs]))          # predicted down among observed down
    spec = float(np.mean(~pred[~obs]))        # predicted up among observed up
    bal = 0.5 * (sens + spec)
    # MCC via correlation of 0/1 vectors (a different formula from the confusion-count one)
    mcc = float(np.corrcoef(pred.astype(float), obs.astype(float))[0, 1])
    # per-source counts via bincount
    codes, uniq = pd.factorize(df["source"])
    k = len(uniq)
    tp = np.bincount(codes, weights=(pred & obs), minlength=k)
    tn = np.bincount(codes, weights=(~pred & ~obs), minlength=k)
    fp = np.bincount(codes, weights=(pred & ~obs), minlength=k)
    fn = np.bincount(codes, weights=(~pred & obs), minlength=k)
    rng = np.random.default_rng(SEED)
    d_acc = np.empty(N_BOOT); d_bal = np.empty(N_BOOT); d_ind = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = rng.multinomial(k, np.full(k, 1.0 / k))  # bootstrap weights = counts of each silenced gene
        TP, TN, FP, FN = (w @ tp, w @ tn, w @ fp, w @ fn)
        T = TP + TN + FP + FN
        a = (TP + TN) / T
        dec = (TP + FN) / T
        pdec = (TP + FP) / T
        d_acc[b] = a - max(dec, 1 - dec)
        d_bal[b] = 0.5 * (TP / (TP + FN) + TN / (TN + FP)) - 0.5
        d_ind[b] = a - (pdec * dec + (1 - pdec) * (1 - dec))
    q = lambda x: [round(float(np.quantile(x, 0.025)), 5), round(float(np.quantile(x, 0.975)), 5)]
    orig = json.load(open(reg(R / "circuit-tracing-217M/outputs/phase11_crispri_validation.json")))
    return {"n_pairs": n, "n_sources": int(k), "label_from_lfc_matches_stored": stored_obs_agrees,
            "correct_column_matches": stored_correct_agrees,
            "accuracy": acc, "always_decrease": p_dec, "pred_decrease_rate": float(pred.mean()),
            "balanced_accuracy": bal, "mcc": mcc,
            "acc_minus_best_constant_pts": round(100 * (acc - max(p_dec, 1 - p_dec)), 3),
            "ci95_acc_minus_best_constant_pts": [round(100 * x, 2) for x in q(d_acc)],
            "share_boot_above0_acc": float((d_acc > 0).mean()),
            "balacc_minus_half_pts": round(100 * (bal - 0.5), 3),
            "ci95_balacc_minus_half_pts": [round(100 * x, 2) for x in q(d_bal)],
            "ci95_acc_minus_independence_pts": [round(100 * x, 2) for x in q(d_ind)],
            "bootstrap": {"unit": "silenced (source) gene, n=%d" % k, "method": "percentile; multinomial resampling weights (= sampling genes with replacement); pooled confusion counts", "reps": N_BOOT, "seed": SEED},
            "original_phase11_json": orig}


# ---------------------------------------------------------------- C2 circuit edges
def c2():
    p = reg(R / "circuit-tracing-217M/outputs/circuit_edges.csv")
    cnt = Counter(); n = 0; inh = 0
    with open(p, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        i_s, i_t, i_g = hdr.index("src_layer"), hdr.index("tgt_layer"), hdr.index("sign")
        for row in r:
            n += 1
            cnt[(int(row[i_s]), int(row[i_t]))] += 1
            inh += row[i_g] == "inhibitory"
    src = sorted({s for s, _ in cnt})
    nxt = {s: cnt.get((s, s + 1), 0) for s in (0, 3, 6, 9)}
    second = {s: cnt.get((s, s + 2), 0) for s in (0, 3, 6, 9)}
    txt = open(reg(S / "circuit-tracing-217M-FINAL_SUMMARY.md"), encoding="utf-8").read().splitlines()
    line5 = [i + 1 for i, l in enumerate(txt) if "22× denser" in l]
    return {"n_edges": n, "inhibitory_share": round(inh / n, 4), "source_layers_seen": src,
            "edges_at_src_plus_1": nxt, "edges_at_src_plus_2": second,
            "ratio_to_52116": round(n / 52116, 2), "summary_lines_with_22x": line5}


# ---------------------------------------------------------------- C3 Exp-1 features
def c3():
    d = R / "exhaustive-mapping-217M/outputs/experiment1"
    tot, l6, l8, l11 = [], 0, 0, 0
    n_l6_zero = 0
    for f in sorted(d.iterdir()):
        if not re.fullmatch(r"feature_F\d+\.json", f.name):
            continue
        j = json.loads(f.read_text())
        e = j["edges_by_layer"]
        tot.append(sum(int(v) for v in e.values()))
        n_l6_zero += int(e.get("6", 0)) == 0
        l6 += int(e.get("6", 0)); l8 += int(e.get("8", 0)); l11 += int(e.get("11", 0))
    t = np.array(tot)
    return {"n_features": len(t), "n_with_zero_L6": n_l6_zero, "edges_L6": l6, "edges_L8": l8, "edges_L11": l11,
            "total": int(t.sum()), "mean": round(float(t.mean()), 2), "sd": round(float(t.std(ddof=1)), 2),
            "min": int(t.min()), "max": int(t.max()), "note": "per-feature totals recomputed from edges_by_layer, not from total_edges"}


# ---------------------------------------------------------------- C4 triplets
def c4():
    out = {}
    for tag in ("experiment2", "experiment2_v2"):
        j = json.load(open(reg(R / f"exhaustive-mapping-217M/outputs/{tag}/combinatorial_summary.json")))
        rows = []
        if "pairwise_ratio_AB" not in j["triplets"][0]:
            out[tag] = {"n_triplets": j.get("n_triplets", len(j["triplets"])), "note": "v1 file stores only the mean pairwise ratio; overwrite identity not testable",
                        "threeway": [t["threeway_ratio"] for t in j["triplets"]], "n_superadditive": [t["n_superadditive"] for t in j["triplets"]],
                        "n_targets_with_effect": [t["n_targets_with_effect"] for t in j["triplets"]]}
            continue
        for t in j["triplets"]:
            ab, ac = t["pairwise_ratio_AB"], t["pairwise_ratio_AC"]
            # overwrite model with C = 1: AB = B/(A+B), AC = 1/(A+1)  ->  A, then B
            A = 1 / ac - 1
            B = ab * A / (1 - ab)
            pred = {"pairwise_ratio_BC": 1 / (B + 1), "threeway_ratio": 1 / (A + B + 1), "marginal_C_given_AB": 1 - B}
            res = {k: round(abs(t[k] - v), 5) for k, v in pred.items()}
            rows.append({"triplet": t["triplet"], "A": round(A, 3), "B": round(B, 3), "abs_residuals": res,
                         "n_targets_with_effect": t["n_targets_with_effect"], "n_superadditive": t["n_superadditive"]})
        out[tag] = {"n_triplets": j.get("n_triplets", len(j["triplets"])), "rows": rows,
                    "max_abs_residual": max(max(r["abs_residuals"].values()) for r in rows)}
    out["note"] = "solved from AB and AC (first agent used AB and BC); predicts BC, three-way and marginal"
    return out


# ---------------------------------------------------------------- C5 steering
def c5():
    j = json.load(open(reg(R / "exhaustive-mapping-217M/outputs/experiment3/steering_summary.json")))
    a5 = {L: v["mean_delta_s"] for L, v in j["per_layer_summary_alpha5"].items()}
    a2 = {L: v["mean_delta_s"] for L, v in j["per_layer_summary_alpha2"].items()}
    per = defaultdict(list)
    for f in j["features"]:
        per[f"L{f['layer']}"].append(f["alpha_results"]["5"]["mean_delta_s"])
    z = np.load(reg(R / "exhaustive-mapping-217M/outputs/experiment3/state_signatures.npz"))
    a, b = z["g_early_logits"].astype(np.float64), z["g_late_logits"].astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    log = open(reg(R / "exhaustive-mapping-217M/outputs/experiment3_rerun.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"cos\(g_early, g_late\) = ([0-9.]+)", log)
    return {"alpha5_over_alpha2": {L: round(a5[L] / a2[L], 4) for L in a5},
            "max_spread_within_layer_alpha5": {L: round(float(np.ptp(v)), 5) for L, v in per.items()},
            "L0_over_L3": round(a5["L0"] / a5["L3"], 3), "L0_over_L11": round(a5["L0"] / a5["L11"], 3),
            "cos_g_early_g_late_recomputed": round(cos, 4), "cos_in_run_log": m.group(1) if m else None}


# ---------------------------------------------------------------- C6 spectral sample overlap
def c6():
    log = open(reg(R / "spectral-geometry-217M/outputs/phase0.log"), encoding="utf-8").read()
    n_total = int(re.search(r"sampled 2000/(\d+) cells", log).group(1))
    samp = {}
    for s in (42, 43, 44):
        rng = np.random.default_rng(s)
        samp[s] = set(rng.choice(n_total, size=2000, replace=False).tolist())
    ov = {f"{a}-{b}": len(samp[a] & samp[b]) for a, b in ((42, 43), (42, 44), (43, 44))}
    # expected overlap of two random 2000-subsets of n_total
    exp = 2000 * 2000 / n_total
    return {"n_total_from_log": n_total, "overlap": ov, "expected_overlap_random": round(exp, 2),
            "note": "n_total read from the run's own log line, not from the h5ad; draws copy phase0_extract.py:64-65 and phase8_stability.py:102-103"}


# ---------------------------------------------------------------- C7 curveball
def _load_curveball():
    src = open(reg(R / "attention-grn-217M/scripts/phase3_residualization.py"), encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("def curveball_permute"))
    end = start + 1
    while end < len(src) and (src[end].startswith(" ") or src[end].strip() == ""):
        end += 1
    ns = {"np": np}
    exec("\n".join(src[start:end]), ns)
    return ns["curveball_permute"], start + 1, end


def c7():
    cb, l0, l1 = _load_curveball()
    tf, tg = [], []
    with open(reg(TRRUST), newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            tf.append(row[0].upper()); tg.append(row[1].upper())
    tfset = set(tf)
    out = {"function_lines": [l0, l1]}
    for suf in ("", "_rpe1", "_adamson", "_k562_1b"):
        gf = pd.read_csv(reg(R / f"attention-grn-217M/outputs/phase0{suf}/gene_features.csv"))
        sym = [s.upper() for s in gf["symbol"]]
        ix = {s: i for i, s in enumerate(sym)}
        n = len(sym)
        M = np.zeros((n, n), dtype=np.int8)
        for a, b in zip(tf, tg):
            i, j = ix.get(a), ix.get(b)
            if i is not None and j is not None and i != j:
                M[i, j] = 1
        ne = int(M.sum())
        rows = np.array([i for i in range(n) if sym[i] in tfset and M[i].sum() >= 3])
        same_rows = same_all = 0
        nonempty_rows = int((M.sum(1) > 0).sum())
        for t in range(50):
            P = cb(M, n_iter=5 * ne, seed=42 + t)
            same_rows += int(np.array_equal(P[rows], M[rows]))
            same_all += int(np.array_equal(P, M))
        nr = json.load(open(reg(R / f"attention-grn-217M/outputs/phase3{suf}/null_results.json")))
        out[suf or "_k562"] = {"n_genes": n, "n_edges": ne, "n_nonempty_rows": nonempty_rows,
                               "n_scored_rows": int(len(rows)), "identical_on_scored_rows_of_50": same_rows,
                               "identical_full_matrix_of_50": same_all,
                               "reported": nr.get("curveball")}
    return out


# ---------------------------------------------------------------- C8 text facts used in the ledger
def c8():
    ef = reg(R / "topology-141-217M/EXTENDED_FINDINGS.md")
    ef_lines = [i + 1 for i, l in enumerate(open(ef, encoding="utf-8")) if "barely changing at all" in l]
    lung = json.load(open(reg(R / "manifold-discovery-217M/reports/external_validation_lung_nonhema.json")))
    qg = json.load(open(reg(R / "manifold-discovery-217M/reports/quality_gates_let_anchor.json")))
    return {"extended_findings_barely_changing_lines": ef_lines,
            "extended_findings_mtime": dt.datetime.fromtimestamp(ef.stat().st_mtime).isoformat(timespec="seconds"),
            "lung_nonhema_trust": lung.get("trustworthiness"),
            "null_shuffled": {k: qg["null_shuffled"].get(k) for k in ("trustworthiness", "random_holdout", "donor_holdout", "branch_holdout")}}


# ---------------------------------------------------------------- C9 ledger bookkeeping
def c9():
    d = pd.read_csv(reg(FE / "error_ledger.csv"))
    pre = d[d.present_in_pre_audit_state.isin(["yes", "partly"])]
    crit_pre = pre[pre.severity == "critical"]
    required = ["id", "short_name", "pipeline", "error_type", "audit_pattern", "severity", "where_it_lives",
                "deterministic_evidence", "present_in_pre_audit_state", "first_found_by", "confirmed_by",
                "repaired_by", "repair_status", "notes", "layer"]
    e_refs = set()
    for r in d.revision_plan_ref.fillna(""):
        e_refs |= {int(x) for x in re.findall(r"E(\d+)", r)}
    # a paper_text row can only be present pre-audit if the claim also sits in a pre-audit run file
    paper_pre = d[(d.layer == "paper_text") & (d.present_in_pre_audit_state != "no")][["id", "present_in_pre_audit_state"]].values.tolist()
    return {"n_rows": len(d), "missing_required_columns": [c for c in required if c not in d.columns],
            "layer": d.layer.value_counts().to_dict(), "severity": d.severity.value_counts().to_dict(),
            "n_pre_audit": len(pre), "n_pre_audit_crit_or_major": int(pre.severity.isin(["critical", "major"]).sum()),
            "sweep_on_pre_audit": pre.detected_by_original_audit_sweep.value_counts().to_dict(),
            "sweep_on_pre_audit_crit_major": pre[pre.severity.isin(["critical", "major"])].detected_by_original_audit_sweep.value_counts().to_dict(),
            "critical_pre_audit_ids": crit_pre.id.tolist(),
            "critical_pre_audit_sweep": crit_pre.detected_by_original_audit_sweep.tolist(),
            "first_found_by": d.first_found_by.value_counts().to_dict(),
            "repair_status": d.repair_status.value_counts().to_dict(),
            "E_numbers_missing": sorted(set(range(1, 43)) - e_refs),
            "n_rows_without_E_ref": int((d.revision_plan_ref.fillna("") == "").sum()),
            "pending_mps_rows": d[d.confirmation_status.str.contains("MPS", na=False)].id.tolist(),
            "run_outputs_outside_checklist": int(((d.layer == "run_outputs") & (d.audit_pattern == "outside checklist")).sum()),
            "paper_text_rows_marked_pre_audit": paper_pre}


# ---------------------------------------------------------------- C10 snapshot integrity
def c10():
    inv = json.load(open(reg(STUDYC / "build/snapshot_inventory.json")))
    tsv = {}
    with open(reg(STUDYC / "build/snapshot_files_sha256.tsv"), newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tsv[row["path"]] = row["sha256"]
    snap_files = {str(p.relative_to(SNAP)): p for p in SNAP.rglob("*") if p.is_file()}
    included = {x["dst"]: x for x in inv["included"]}
    recon = {x["dst"]: x for x in inv["reconstructed"]}
    listings = {k for k in snap_files if k.endswith("_large_files_listing.tsv")}
    stray = sorted(set(snap_files) - set(included) - set(recon) - listings)
    missing = sorted((set(included) | set(recon)) - set(snap_files))
    bad_copy, post_cutoff_src, tsv_mismatch = [], [], []
    for dst, x in included.items():
        sp = Path(x["src"])
        h_dst = sha(snap_files[dst]) if dst in snap_files else None
        if sp.stat().st_mtime >= CUTOFF:
            post_cutoff_src.append(dst)
        if h_dst != sha(sp):
            bad_copy.append(dst)
        if tsv.get(dst) not in (None, h_dst):
            tsv_mismatch.append(dst)
    for dst, x in recon.items():
        if sha(snap_files[dst]) != x["dst_sha256"]:
            tsv_mismatch.append(dst)
    # listing rows that point at post-cutoff files, and listing mtimes
    post_rows, listing_mtimes = [], set()
    for k in listings:
        p = snap_files[k]
        listing_mtimes.add(dt.datetime.fromtimestamp(p.stat().st_mtime).date().isoformat())
        with open(p, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["mtime"] and row["mtime"] >= "2026-05-07 16:57":
                    post_rows.append(row["path_in_project"])
    total = sum(p.stat().st_size for p in snap_files.values())
    appledouble = [k for k in snap_files if Path(k).name.startswith("._")]
    return {"n_files": len(snap_files), "bytes": total, "n_included": len(included), "n_reconstructed": len(recon),
            "n_listings": len(listings), "stray_files": stray[:20], "missing_files": missing[:20],
            "copies_not_byte_identical": bad_copy[:20], "copied_sources_at_or_after_cutoff": post_cutoff_src[:20],
            "hash_list_mismatches": tsv_mismatch[:20], "listing_rows_pointing_to_post_cutoff_files": post_rows[:20],
            "listing_file_dates": sorted(listing_mtimes), "appledouble_files": appledouble[:5]}


# ---------------------------------------------------------------- C11 leak scan (independent term list)
LEAK = {
    "audit_event": r"audit (action|residual)|audits/|audit-2026|audit_a\d|audit_residual|audit-recurring",
    "later_work": r"framework-eval|studyC|error ledger|revision plan|/revision/|investigation report|hooks_v2|v2_hooks|zero_edit|v2_eval|v2_intervals|v2_crossmodel",
    "paper_folders": r"paper-(plos-one|biosystems|jbi|cbac|deanon)|projects/maxtoki/paper/",
    "known_answers": r"53\.4[28] ?%|53\.55 ?%|always[- ]decrease|always predict .?(decrease|inhibitory)|block[- ]deletion|deletes? (a|the) (whole )?(transformer )?block|sign bug|ignored predicted sign|null[- ]saturated|detection-count|length-matched|overwrite each other",
    "hardware_model": r"\bA100\b|Opus 4\.7|claude-opus",
    "dates_after_audit": r"2026-05-0[89]|2026-0[6-9]-|2026-1[0-2]-",
}


def c11():
    hits = defaultdict(list)
    n = 0
    for p in sorted(SNAP.rglob("*")):
        if not p.is_file() or p.suffix.lower() in (".npy", ".npz", ".pt", ".pkl"):
            continue
        n += 1
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for k, rx in LEAK.items():
            for m in re.finditer(rx, txt, flags=re.I):
                a = max(0, m.start() - 60)
                hits[k].append({"file": str(p.relative_to(SNAP)), "text": txt[a:m.end() + 60].replace("\n", " ")})
    return {"files_scanned": n, "n_hits": {k: len(hits.get(k, [])) for k in LEAK}, "hits": {k: v[:40] for k, v in hits.items()}}


if __name__ == "__main__":
    t0 = time.time()
    for cid, fn in (("C1_crispri", c1), ("C2_circuit_edges", c2), ("C3_exp1_features", c3), ("C4_triplets", c4),
                    ("C5_steering", c5), ("C6_spectral_overlap", c6), ("C7_curveball", c7), ("C8_text_facts", c8),
                    ("C9_ledger_counts", c9), ("C10_snapshot_integrity", c10), ("C11_leak_scan", c11)):
        run(cid, fn)
    OUT.write_text(json.dumps({"generated": dt.datetime.now().isoformat(timespec="seconds"), "results": RES}, indent=1, default=str))
    CFG.write_text(json.dumps({"script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__).resolve()),
                               "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                               "bootstrap_seed": SEED, "n_boot": N_BOOT, "curveball_seeds": "42..91 (deployed)",
                               "cutoff": "2026-05-07T16:57 local", "inputs_sha256": INPUTS,
                               "snapshot_root": str(SNAP), "wall_seconds": round(time.time() - t0, 1)}, indent=1))
    print("wrote", OUT)
