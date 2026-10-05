"""Write error_ledger.csv, error_ledger.md and run_config.json for item C-prep.

Inputs: ledger_rows.py (row content), verify_results.json (deterministic checks),
the Study C snapshot inventory and leak scan (if present), and the
investigation reports (hashed for provenance).
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FE = HERE.parent
sys.path.insert(0, str(HERE))
from ledger_rows import ROWS  # noqa: E402

MT = Path("<REPO_ROOT>/projects/maxtoki")
INV = MT / "verification"
PLAN = MT / "paper-plos-one/revision/REVISION_PLAN.md"
STUDYC = Path("<EVAL_ROOT>/studyC")
SNAP_INV = STUDYC / "build/snapshot_inventory.json"
LEAK = STUDYC / "build/leak_scan_results.json"

COLS = ["id", "short_name", "layer", "pipeline", "error_type", "audit_pattern", "severity",
        "where_it_lives", "deterministic_evidence", "present_in_pre_audit_state", "first_found_by",
        "confirmed_by", "repaired_by", "repair_status", "confirmation_status",
        "detected_by_original_audit_sweep", "revision_plan_ref", "repeated_in_paper", "notes"]
VOCAB = {
    "layer": {"run_outputs", "paper_text"},
    "error_type": {"code bug", "wrong statistic", "missing baseline or null", "wrong description",
                   "unsupported claim", "reproducibility"},
    "severity": {"critical", "major", "minor"},
    "present_in_pre_audit_state": {"yes", "no", "partly"},
    "repair_status": {"repaired during deployment", "pending re-run", "wording only"},
    "detected_by_original_audit_sweep": {"yes", "partly", "no", "n.a."},
    "repeated_in_paper": {"yes", "no", "partly"},
}
FINDERS = {"executor agent during run", "audit sweep agent", "repair-round agent",
           "independent verification (this investigation)", "human (external)", "unknown", "none"}
PATTERNS = {f"P{i}" for i in range(1, 11)} | {"outside checklist"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def validate():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in ROWS:
        assert set(r) == set(COLS), (r["id"], set(COLS) ^ set(r))
        for k, allowed in VOCAB.items():
            assert r[k] in allowed, (r["id"], k, r[k])
        assert r["audit_pattern"] in PATTERNS, (r["id"], r["audit_pattern"])
        for k in ("first_found_by", "confirmed_by", "repaired_by"):
            assert r[k] in FINDERS, (r["id"], k, r[k])
        if r["layer"] == "paper_text" and r["present_in_pre_audit_state"] == "no":
            assert r["detected_by_original_audit_sweep"] == "n.a.", r["id"]
        if r["present_in_pre_audit_state"] == "no":
            assert r["detected_by_original_audit_sweep"] == "n.a.", r["id"]


def counts(key, rows):
    return dict(collections.Counter(r[key] for r in rows).most_common())


def fmt(d):
    return ", ".join(f"{k} {v}" for k, v in d.items())


def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def main():
    validate()
    # ------------------------------------------------------------------ CSV
    with open(FE / "error_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)

    V = json.load(open(HERE / "verify_results.json"))["checks"]
    run = [r for r in ROWS if r["layer"] == "run_outputs"]
    pap = [r for r in ROWS if r["layer"] == "paper_text"]
    pre = [r for r in ROWS if r["present_in_pre_audit_state"] in ("yes", "partly")]
    pre_cm = [r for r in pre if r["severity"] in ("critical", "major")]
    det = collections.Counter(r["detected_by_original_audit_sweep"] for r in pre)
    det_cm = collections.Counter(r["detected_by_original_audit_sweep"] for r in pre_cm)
    pending = [r for r in ROWS if r["confirmation_status"].startswith(("confirmation pending", "code-level only"))]
    crit = [r for r in ROWS if r["severity"] == "critical"]

    # E1-E42 coverage
    ecov = collections.defaultdict(list)
    for r in ROWS:
        for e in [x.strip() for x in r["revision_plan_ref"].split(";") if x.strip()]:
            ecov[e].append(r["id"])
    missing_e = [f"E{i}" for i in range(1, 43) if f"E{i}" not in ecov]

    snap = json.load(open(SNAP_INV)) if SNAP_INV.exists() else None
    leak = json.load(open(LEAK)) if LEAK.exists() else None

    v2 = V["V02"]; v9 = V["V09"]; v7 = V["V07"]; v8 = V["V08"]; v5 = V["V05"]; v6 = V["V06"]
    v14 = V["V14"]; v15 = V["V15"]; v13 = V["V13"]

    L = []
    A = L.append
    A("# MaxToki deployment: error ledger and pre-audit snapshot (item C-prep)")
    A("")
    A("This file lists every distinct error we could confirm in the outputs of the deployed MaxToki-217M study "
      "(run code, run summaries, audit files) and in the paper drafts built on them. "
      "It also describes the pre-audit snapshot built for the blind re-audit (Study C). "
      "The machine-readable ledger is `error_ledger.csv` (same folder). "
      "The snapshot manifest is `<EVAL_ROOT>/studyC/SNAPSHOT_MANIFEST.md`.")
    A("")
    A("## 1. Numbers first")
    A("")
    A(f"- **{len(ROWS)} distinct errors.** {len(run)} are in the run outputs (code, run summaries, audit files). "
      f"{len(pap)} appear only in the paper text.")
    A(f"- **Severity:** {fmt(counts('severity', ROWS))}. A *critical* error means a headline claim cannot stand.")
    A(f"- **Type:** {fmt(counts('error_type', ROWS))}.")
    A(f"- **Checklist pattern:** {sum(1 for r in ROWS if r['audit_pattern']=='outside checklist')} of {len(ROWS)} "
      f"errors fall outside the ten-pattern checklist (P1-P10). Of the {len(run)} run-output errors, "
      f"{sum(1 for r in run if r['audit_pattern']=='outside checklist')} fall outside it. Counts by pattern (all rows): {fmt(counts('audit_pattern', ROWS))}.")
    A(f"- **Present before the audit started (2026-05-07 16:57):** {len(pre)} errors ({sum(1 for r in pre if r['severity']!='minor')} critical or major).")
    A(f"- **What the original audit sweep caught among those {len(pre)}:** yes {det.get('yes',0)}, partly {det.get('partly',0)}, "
      f"no {det.get('no',0)}. Among the {len(pre_cm)} critical or major ones: yes {det_cm.get('yes',0)}, partly {det_cm.get('partly',0)}, no {det_cm.get('no',0)}. "
      f"None of the {sum(1 for r in pre if r['severity']=='critical')} critical pre-audit errors was caught by the sweep "
      f"({', '.join(r['id'] for r in pre if r['severity']=='critical')}).")
    A(f"- **Who found each error first:** {fmt(counts('first_found_by', ROWS))}. "
      "No error is recorded as first found by the human who supervised the deployment.")
    A(f"- **Repair status:** {fmt(counts('repair_status', ROWS))}.")
    A(f"- **Waiting for the MPS re-runs:** {len(pending)} rows ({', '.join(r['id'] for r in pending)}). "
      "They are marked `confirmation pending (MPS re-run)` or `code-level only; impact pending (MPS re-run)`.")
    A(f"- **Coverage of the plan's list E1-E42:** every E-number maps to at least one row"
      + ("." if not missing_e else f", except {missing_e}.") +
      f" {len(ROWS) - sum(1 for r in ROWS if r['revision_plan_ref'])} rows are new (not in E1-E42).")
    A("")
    A("## 2. Key numbers re-derived for this ledger")
    A("")
    A("All numbers below come from `ledger_build/verify_ledger_evidence.py` (output `verify_results.json`), "
      "which reads only saved files. No model was run.")
    A("")
    A(f"- **CRISPRi direction (L001, L002).** The deployed Phase 11 code counts `actual_lfc < 0` and never uses the predicted sign "
      f"(remaining_phases.py line {V['V01']['count_line'][0]}). So 54.61% is just the share of genes that went down. "
      f"On the {v2['n_pairs']:,} corrected pairs from {v2['n_sources']} silenced genes: model accuracy {v2['accuracy']:.4f}; "
      f"always predicting 'decrease' scores {v2['always_decrease']:.4f}; balanced accuracy {v2['balanced_accuracy']:.4f}; MCC {v2['mcc']:.4f}. "
      f"The model says 'decrease' for {100*v2['pred_decrease_rate']:.1f}% of pairs. "
      f"Model minus the best constant rule: {100*v2['acc_minus_best_constant']:.2f} points, "
      f"95% CI [{100*v2['acc_minus_best_constant_ci95'][0]:.2f}, {100*v2['acc_minus_best_constant_ci95'][1]:.2f}]. "
      f"Balanced accuracy minus 0.5: 95% CI [{100*v2['balacc_minus_half_ci95'][0]:.2f}, {100*v2['balacc_minus_half_ci95'][1]:.2f}] points. "
      "Intervals: percentile bootstrap, 2,000 resamples of silenced genes with replacement, seed 20261001, pooled confusion counts. "
      "Two ways: pooled per-pair arrays and per-gene count sums give the same accuracy to 1e-12. "
      "Plain reading: no directional skill; the model does slightly worse than always guessing 'down'.")
    k = v9["_k562"]
    A(f"- **Curveball null (L033).** Re-running the deployed function with its own seeds: {k['identical_on_scored_rows']} of 50 null draws "
      f"are identical to the real TRRUST matrix on the scored rows in 217M K562 (RPE1 {v9['_rpe1']['identical_on_scored_rows']}, "
      f"Adamson {v9['_adamson']['identical_on_scored_rows']}, 1B {v9['_k562_1b']['identical_on_scored_rows']}). "
      "So the reported z of about 0 is built in by the code.")
    r0 = v7["experiment2_v2"]["rows"][0]
    A(f"- **Triplets (L012, L013).** {v7['experiment2_v2']['n_triplets']} triplets were tested, not 2,980; 2,977-2,980 is a count of downstream features. "
      f"Solving A:B:C from two reported ratios predicts the other three to within the 4-decimal rounding of the JSON values (max residual 1e-4; 3.6e-4 if solved from a different pair) (triplet 0: AC {r0['AC_pred']} vs {r0['AC_obs']}, "
      f"three-way {r0['threeway_pred']} vs {r0['threeway_obs']}, C given AB {r0['marg_pred']} vs {r0['marg_obs']}). "
      "That only happens if each combined ablation equals its deepest single ablation (ABC = C), so 'zero synergy' is forced by the code.")
    A(f"- **Block-deleting hooks (L005, L011).** Among {v5['n_edges']:,} circuit edges, the layer right after each source layer has 0 edges "
      f"(source layers 0, 3, 6, 9: {', '.join(str(v) for v in v5['edges_at_src_plus_1'].values())} edges). All {v6['n_feature_files']} exhaustive-mapping features have 0 edges at L6; edges per feature are "
      f"{v6['mean']:.0f} +/- {v6['sd']:.0f} (min {v6['min']}, max {v6['max']}).")
    A(f"- **Steering (L015).** Delta-s at alpha 5 divided by alpha 2 is {min(v8['alpha5_over_alpha2'].values()):.3f}-{max(v8['alpha5_over_alpha2'].values()):.3f} at every layer; "
      f"the three features within a layer differ by at most {max(v8['max_spread_within_layer_a5'].values()):.4f}. "
      f"L0/L3 = {v8['L0_over_L3']:.2f} and L0/L11 = {v8['L0_over_L11']:.2f} (the paper's '5x' and '20x'). "
      f"cos(g_early, g_late) = {v8['cos_g_early_g_late']:.3f} (summary says 0.999).")
    A(f"- **Manifold (L066, L067, L070, L072).** The frozen gate file has five gates; no script computes the permutation gate. "
      f"The shuffled null passes the random and donor gates ({v13['null']['random_holdout']:.3f}, {v13['null']['donor_holdout']:.3f}). "
      f"The lung control trust is {v14['lung_trust']:.5f} (< 0.80, so it fails all four gates). "
      f"All {v14['n_zs_donors']} zero-shot donors are among the {v14['n_ext_donors']} external donors; "
      f"{v14['zs_tissues_shared_with_ext']} of {v14['zs_tissues']} zero-shot tissues are shared.")
    A(f"- **Spectral (L047, L048).** CKA at L0 is {v15['cka_L0']['cka_mean']} for every pair (fixed token embedding). "
      f"The three 'disjoint' 2,000-cell samples share {', '.join(str(v) for v in v15['overlap_intersect1d'].values())} cells (pairs 42-43, 42-44, 43-44) (n = {v15['n_total_cells']:,}; two methods agree).")
    A("")
    A("## 3. How the ledger was built")
    A("")
    A("- Sources: the 14 investigation reports in `verification/` (read in full: audit_history, "
      "attn_endpoints, crispri, gata1, triplets, manifold_ci, numbers_A/B/C, contract, hardware, repro_inventory; skimmed: "
      "heldout_tasks, agent_context), the plan's list E1-E42, and the run files themselves.")
    A("- 23 deterministic checks (V01-V23) re-read the saved files. Each row's `deterministic_evidence` names its check or the "
      "investigation script it relies on. Rows that rely only on an investigation script say so ('not re-run in this ledger build').")
    A("- `first_found_by` uses only what the record supports: file times, script docstrings, the audit reports, and the human prompt log "
      "summarised in audit_history. 'human (external)' means a person outside the deployment who read the submitted manuscript. "
      "'repair-round agent' means the same agent session during the two repair rounds on 2026-05-07 (17:50-21:10).")
    A("- `detected_by_original_audit_sweep` scores the audit report of 2026-05-07 17:38 (the sweep), not the later repair rounds. "
      "'partly' means it flagged the area but not the actual error (for example, asked for a replication).")
    A("- `layer`: run_outputs = the error exists in code, run summaries or audit files; paper_text = only in the manuscript drafts. "
      "Where a run-output error is repeated in the paper, the row stays run_outputs and `repeated_in_paper` = yes.")
    A("- Hook bugs: the code-level facts and the output fingerprints are confirmed from files. Their final impact waits for the MPS "
      "re-runs (D0, D2-D7, D9). An interim D0 hook unit test (3 cells, written 2026-10-01 00:57) already reproduces the triplet overwrite "
      "(|AB - B| = 0 and |ABC - C| = 0 under the old hooks) and shows that hidden_states[11] is taken after the final norm.")
    A("")
    A("## 4. Critical errors")
    A("")
    A(md_table(crit, ["id", "short_name", "layer", "pipeline", "first_found_by", "detected_by_original_audit_sweep", "confirmation_status"]))
    A("")
    A("## 5. Full ledger (short form)")
    A("")
    A("Details (where it lives, evidence, who confirmed and repaired it, notes) are in `error_ledger.csv`.")
    A("")
    for pipe in sorted(set(r["pipeline"] for r in ROWS)):
        rows = [r for r in ROWS if r["pipeline"] == pipe]
        A(f"### {pipe} ({len(rows)})")
        A("")
        A(md_table(rows, ["id", "short_name", "layer", "severity", "error_type", "audit_pattern", "present_in_pre_audit_state",
                          "first_found_by", "repair_status", "detected_by_original_audit_sweep"],
                   ["id", "error", "layer", "sev.", "type", "pattern", "pre-audit", "first found by", "repair", "sweep caught"]))
        A("")
    A("## 6. What differs from the deployed claims")
    A("")
    A("- The corrected CRISPRi result (53.48%, 'decisively above chance') is at or below chance once the class balance is used as the baseline (L002). The 'five-fold CV' trained nothing (L003).")
    A("- The GATA1 'positive' does not survive a null that keeps gene rarity and length, and the same feature matches dozens of other TFs (L026). The 'power' figures are false-positive rates (L025).")
    A("- '2,980 triplets' were 4 triplets, and 'zero synergy' plus the 0.190 redundancy ratio are forced by hooks that overwrite each other (L012, L013).")
    A("- The 4.97M- and 2.14M-edge maps and the steering results come from hooks that delete a whole transformer block (L005, L011, L015); final numbers wait for the fixed-hook re-runs.")
    A("- The curveball null did not shuffle, so 'z = 0' is built in (L033). Residualised attention does not fall to chance in Adamson or 1B (L037).")
    A("- The audit took one afternoon, not days (L096); no record supports human spot-checks finding errors (L097); the sign bug was found by the agent during repair, not by the audit sweep (L001, L098); 75% 'fully addressed' has no cell table (L090).")
    A("- Hardware was an Apple-silicon laptop (MPS), not an A100 (L094).")
    A("- One correction to the investigation reports: audit_history section 4.5 says the 'barely changes' framing first appears at 19:27. "
      "The pre-audit EXTENDED_FINDINGS.md (16:37, line 26) already says the feature is 'barely changing at all' (see L057 notes).")
    A("")
    A("## 7. Pre-audit snapshot for the blind re-audit (Study C)")
    A("")
    if snap:
        t = snap["totals"]
        A(f"- Location: `<EVAL_ROOT>/studyC/snapshot/maxtoki/` "
          f"({t['files']:,} files, {t['bytes']/1e6:.1f} MB; built {t['built']}; cutoff {t['cutoff']}).")
        A(f"- {len(snap['reconstructed'])} summaries rebuilt by removing audit-added text: "
          + ", ".join(f"`{x['dst']}`" for x in snap["reconstructed"]) + ".")
        A(f"- Not included: the project README and requirements.txt (edited or created during the audit; the README cannot be rebuilt), "
          f"audits/, all paper folders, all audit_* scripts and outputs, and every file changed after the cutoff "
          f"({len(snap['excluded'])} files, including new re-run files). {len(snap['listed_not_copied'])} large or duplicate files are "
          "listed (size and shape) in `_large_files_listing.tsv` files instead of copied.")
        A("- Full detail, including what was reconstructed and how: `studyC/SNAPSHOT_MANIFEST.md` (kept outside the snapshot).")
    if leak:
        strong = {k: v["n_hits"] for k, v in leak["strong"].items() if v["n_hits"]}
        A(f"- Leak scan over {leak['files_scanned']:,} text files: strong-term hits {fmt(strong) if strong else 'none'}. "
          "Each was checked by hand: all are pre-audit wording (spec text about an 'executor-reviewer loop', commit subjects, "
          "pre-audit May-7 notes) or number coincidences inside CSVs (for example 0.53422433; 'SV2_SV4' matching 'V2_'). "
          "Weak-term hits (for example 'audit' in 'strict max-null audit') are listed and judged in the manifest.")
    A("")
    A("## 8. What was not done")
    A("")
    A("- No model forward pass (GPU in use by the re-runs). The hook bugs' final impact is marked pending.")
    A("- The GATA1 matched nulls, the TRRUST variance baseline, the donor bootstrap and the CCA chance level were not re-run here; "
      "the rows cite the investigation scripts and outputs.")
    A("- The Adamson CRISPRa/CRISPRi label (L042) needs a literature check.")
    A("- Session transcripts of the deployment are gone, so 'first found by' for a few items rests on file times and docstrings.")
    A("- The pre-audit project README could not be rebuilt; it is left out of the snapshot. Errors that lived only in that README "
      "(parts of L085, L107) cannot be found by blind auditors.")
    A("")
    A("## Plain-words summary")
    A("")
    A(f"We listed {len(ROWS)} separate errors. {len(run)} are in the study's own code, run summaries and audit files; "
      f"{len(pap)} are only in the paper. {len(crit)} are critical: a headline claim cannot stand. "
      f"{len(pre)} errors were already there before the audit began. The original audit sweep caught {det.get('yes',0)} of them fully and "
      f"{det.get('partly',0)} in part. It missed all six critical ones: five are code bugs that no checklist pattern covers, "
      "and the sixth (judging accuracy against 50% instead of the class balance) is exactly the weak-null problem the checklist's P1 describes. "
      "The worst problems are hooks that delete whole model blocks, a sign that was never compared, hooks that overwrite each other, "
      "a null that never shuffled, and a 'positive' GATA1 result that fails a fair null. "
      "No record shows the human supervisor finding any error first. "
      f"{len(pending)} rows wait for the model re-runs. "
      "We also built a pre-audit copy of the project for a blind re-audit: scripts, logs, small outputs and summaries as they were "
      "before the audit, with audit-added text removed and all audit and paper files left out.")
    # keep an appended "## Verification notes" section (added by the independent verifier) across rebuilds
    old = (FE / "error_ledger.md").read_text() if (FE / "error_ledger.md").exists() else ""
    keep = old[old.index("\n## Verification notes"):] if "\n## Verification notes" in old else ""
    (FE / "error_ledger.md").write_text("\n".join(L) + "\n" + keep)

    # ------------------------------------------------------------------ run_config.json
    inputs = {}
    for p in [PLAN] + sorted(INV.glob("*.md")) + [HERE / "ledger_rows.py", HERE / "verify_results.json",
                                                   HERE / "verify_ledger_evidence.py", HERE / "build_ledger.py"]:
        if p.exists() and not p.name.startswith("._"):
            inputs[str(p)] = sha(p)
    for extra in (SNAP_INV, LEAK, STUDYC / "build/build_snapshot.py", STUDYC / "build/leak_scan.py",
                  STUDYC / "build/build_manifest.py", STUDYC / "build/snapshot_files_sha256.tsv",
                  STUDYC / "build/run_config.json", STUDYC / "SNAPSHOT_MANIFEST.md"):
        if extra.exists():
            inputs[str(extra)] = sha(extra)
    vr = json.load(open(HERE / "verify_results.json"))
    cfg = {"task": "C-prep error ledger + pre-audit snapshot", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seeds": {"crispri_grouped_bootstrap": 20261001, "curveball_rerun": "42+trial (deployed seeds)",
                     "spectral_sample_redraw": [42, 43, 44]},
           "cutoff_local_time": "2026-05-07 16:57",
           "python": sys.version.split()[0], "n_rows": len(ROWS),
           "inputs_sha256_ledger": inputs, "inputs_sha256_verification_checks": vr["inputs_sha256"]}
    (HERE / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print("rows", len(ROWS), "run", len(run), "paper", len(pap), "pre", len(pre), "det", dict(det), "det_cm", dict(det_cm),
          "crit", len(crit), "pending", len(pending), "missingE", missing_e)


if __name__ == "__main__":
    main()
