#!/usr/bin/env python3
"""Collect the headline numbers of all checks into results/checker_summary.json and
results/checker_summary.md. CHECKER_REPORT.md quotes only numbers that appear here."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJ, RESULTS, SEED, run_config, write_json  # noqa: E402


def load_csv(name):
    p = RESULTS / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (round(max(0, c - w), 4), round(min(1, c + w), 4))


def main():
    out, md = {}, []
    # ---------------- spec lint
    lint = json.loads((RESULTS / "spec_lint.json").read_text())
    dep = [r for r in lint if r["deployed_on_maxtoki"]]
    s = {}
    for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        s[k] = dict(Counter(r["verdicts"][k] for r in dep))
    refs = [r["code_refs"] for r in dep]
    s["template_reject"] = sum(r["template_verdict"] == "REJECT" for r in dep)
    s["pin_status"] = dict(Counter(r["pin"]["status"].split(";")[0] for r in dep))
    s["code_refs_total"] = sum(c["n_refs"] for c in refs)
    s["code_refs_required_form"] = sum(c["n_required_form"] for c in refs)
    s["code_refs_exist"] = sum(c["n_exists"] for c in refs)
    s["code_refs_exist_as_written"] = sum(c["n_exists_as_written"] for c in refs)
    s["code_refs_missing"] = sum(len(c["missing"]) for c in refs)
    s["code_refs_pattern_unchecked"] = sum(c["n_pattern_unchecked"] for c in refs)
    s["code_refs_with_line"] = sum(c["n_with_line"] for c in refs)
    s["code_refs_line_in_range"] = sum(c["n_line_in_range"] for c in refs)
    s["code_refs_in_clone_without_git"] = sum(c["n_resolved_in_clone_without_git"] for c in refs)
    s["symbols_checked"] = sum(c["n_symbols_checked"] for c in refs)
    s["symbols_found_in_file"] = sum(c["n_symbols_found_in_file"] for c in refs)
    s["symbols_not_found"] = {r["spec"]: r["code_refs"]["symbols_not_found"] for r in dep if r["code_refs"]["symbols_not_found"]}
    s["missing_refs"] = {r["spec"]: r["code_refs"]["missing"] for r in dep if r["code_refs"]["missing"]}
    s["param_rows"] = sum(r["parameters"].get("n_rows", 0) for r in dep)
    s["param_default_numeric"] = sum(r["parameters"].get("n_default_machine_readable", 0) for r in dep)
    s["param_range_numeric"] = sum(r["parameters"].get("n_range_machine_readable", 0) for r in dep)
    s["val_items"] = sum(r["validation"].get("n_items", 0) for r in dep)
    s["val_threshold"] = sum(r["validation"].get("numeric-threshold", 0) for r in dep)
    s["val_mention"] = sum(r["validation"].get("numeric-mention", 0) for r in dep)
    s["val_qualitative"] = sum(r["validation"].get("qualitative", 0) for r in dep)
    s["val_if_broken"] = sum(r["validation"].get("n_if_broken_rule", 0) for r in dep)
    s["val_machine_readable_blocks"] = sum(bool(r["validation"].get("machine_readable_block")) for r in dep)
    s["val_framed_as_source_replication"] = sum(bool(r["validation"].get("framed_as_source_replication")) for r in dep)
    s["slots_labelled"] = {k: sum(r["slots"][k]["has_labelled_slot"] for r in dep)
                           for k in ("null_model", "trivial_baseline", "positive_control", "scope_statement")}
    s["slots_all_four"] = sum(r["verdicts"]["S7"] == "PASS" for r in dep)
    out["spec_lint"] = s
    md.append("## spec_lint (8 deployed specs)\n")
    md.append("| spec | template | S1 | S2 | S3 pin | S4 refs | S5 params | S6 val block | S7 slots | refs (required form / exist as written / total) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in dep:
        v, c = r["verdicts"], r["code_refs"]
        md.append(f"| {r['spec'].split('/')[-1]} | {r['template_verdict']} | {v['S1']} | {v['S2']} | {r['pin']['status'].split(';')[0]} "
                  f"| {v['S4']} | {v['S5']} | {v['S6']} | {v['S7']} | {c['n_required_form']} / {c['n_exists_as_written']} / {c['n_refs']} |")
    # ---------------- number trace
    summ = load_csv("number_trace_summary.csv")
    nums = load_csv("number_trace_numbers.csv")
    nt = {"docs": [], "paper": [], "classes": []}
    md.append("\n## number_trace\n")
    md.append("| scope | document | numbers | traced (share, Wilson 95% CI) | mean chance rate (bootstrap 95% CI) | excess over chance (bootstrap 95% CI) | weak traces | untraced |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in summ:
        if r["scope"] in ("run_summary",) or r["scope"].startswith("paper") or r["scope"].startswith("run_summaries_class"):
            key = "docs" if r["scope"] == "run_summary" else ("paper" if r["scope"].startswith("paper") else "classes")
            nt[key].append(r)
            f = lambda x: f"{float(x):.2f}" if x not in ("", None) else "n/a"
            md.append(f"| {r['scope']} | {r['doc'].split('/')[-1] if r['doc'] else ''} | {r['n_numbers']} | {r['n_traced']} ({f(r['traced_share'])}, {f(r['traced_ci_lo'])}–{f(r['traced_ci_hi'])}) "
                      f"| {f(r['mean_chance'])} ({f(r['chance_ci_lo'])}–{f(r['chance_ci_hi'])}) | {f(r['excess'])} ({f(r['excess_ci_lo'])}–{f(r['excess_ci_hi'])}) "
                      f"| {r['n_weak']} | {r['n_untraced']} |")
    rs = [r for r in nums if r["scope"] == "run_summary"]
    pooled_n = len(rs)
    pooled_t = sum(r["traced"] == "True" for r in rs)
    pooled_w = sum(r["weak"] == "True" for r in rs)
    ch = np.array([float(r["chance_rate"]) for r in rs])
    tr = np.array([r["traced"] == "True" for r in rs], float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rs), size=(2000, len(rs)))
    ex = tr - ch
    out["number_trace"] = dict(
        summary_rows_all_docs=pooled_n, traced=pooled_t, traced_share=round(pooled_t / pooled_n, 4),
        traced_wilson=wilson(pooled_t, pooled_n), weak=pooled_w, strong=pooled_t - pooled_w,
        mean_chance=round(float(ch.mean()), 4),
        mean_chance_ci=[round(float(np.percentile(ch[idx].mean(1), q)), 4) for q in (2.5, 97.5)],
        excess=round(float(ex.mean()), 4),
        excess_ci=[round(float(np.percentile(ex[idx].mean(1), q)), 4) for q in (2.5, 97.5)],
        untraced=pooled_n - pooled_t,
        untraced_not_in_spec=sum(r["traced"] == "False" and r["also_in_spec"] == "False" for r in rs),
        interval_note="Wilson over numbers; bootstrap = 2,000 resamples of numbers (rows pooled over the 13 FINAL_SUMMARY documents; a number written in two copies of a summary counts twice)",
        docs=nt["docs"], paper=nt["paper"], classes=nt["classes"])
    # Distinct numbers per run (added in the 2026-10-01 verification pass). The pooled rows above count a
    # number twice when it is written in two copies of the same run summary (summaries/ and the run folder).
    # Here each (run, value, half-width, %) counts once. Intervals: Wilson over distinct numbers;
    # percentile bootstrap over distinct numbers (2,000 resamples); and a run-level cluster bootstrap
    # (2,000 resamples of the 8 runs, numbers kept together), because numbers in one run share one corpus.
    dmap = {}
    for r in rs:
        dmap.setdefault((r["run"], round(float(r["value"]), 12), round(float(r["half_width"]), 15), r["pct"]), r)
    du = list(dmap.values())
    dtr = np.array([r["traced"] == "True" for r in du], float)
    dch = np.array([float(r["chance_rate"]) for r in du])
    dex = dtr - dch
    didx = np.random.default_rng(SEED).integers(0, len(du), size=(2000, len(du)))
    druns = sorted({r["run"] for r in du})
    by_run = {R: np.array([i for i, r in enumerate(du) if r["run"] == R]) for R in druns}
    crng = np.random.default_rng(SEED)
    cl = []
    for _ in range(2000):
        ii = np.concatenate([by_run[druns[j]] for j in crng.integers(0, len(druns), len(druns))])
        cl.append((dtr[ii].mean(), dch[ii].mean(), dex[ii].mean()))
    cl = np.array(cl)
    pc = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    out["number_trace"]["distinct_per_run"] = dict(
        n_numbers=len(du), traced=int(dtr.sum()), traced_share=round(float(dtr.mean()), 4),
        traced_wilson=wilson(int(dtr.sum()), len(du)), traced_cluster_ci=pc(cl[:, 0]),
        weak=sum(r["weak"] == "True" for r in du),
        mean_chance=round(float(dch.mean()), 4), mean_chance_ci=pc(dch[didx].mean(1)), mean_chance_cluster_ci=pc(cl[:, 1]),
        excess=round(float(dex.mean()), 4), excess_ci=pc(dex[didx].mean(1)), excess_cluster_ci=pc(cl[:, 2]),
        untraced=len(du) - int(dtr.sum()),
        untraced_not_in_spec=sum(r["traced"] == "False" and r["also_in_spec"] == "False" for r in du),
        n_runs=len(druns), n_rows_pooled=pooled_n, n_rows_that_repeat_a_number=pooled_n - len(du),
        interval_note="unit = distinct number per run (run, value, half-width, %); Wilson over numbers; "
                      "'_ci' = percentile bootstrap over numbers (2,000); '_cluster_ci' = percentile bootstrap "
                      "over the 8 runs (2,000), which keeps the numbers of one run together")
    pap = load_csv("number_trace_paper_numbers.csv")
    out["number_trace"]["paper_not_chain"] = [dict(line=r["line"], raw=r["raw"], in_summaries=r["in_summaries"],
                                                   traced=r["traced"], context=r["context"][:100]) for r in pap if r["chain_trace"] == "False"]
    for f in ("verify_number_trace.json", "verify_chance_rate.json", "crosscheck_manual_audit.json", "verify_spec_lint.json",
              "verify_manifest_deviations.json"):
        p = RESULTS / f
        if p.exists():
            d = json.loads(p.read_text())
            d.pop("checks", None)
            d.pop("rows", None)
            d.pop("disagreements", None)
            d.pop("per_doc", None)
            out[f.replace(".json", "")] = d
    # ---------------- run manifest
    man = json.loads((RESULTS / "run_manifest.json").read_text())
    params = load_csv("run_manifest_params.csv")
    gates = load_csv("run_manifest_gates.csv")
    out["run_manifest"] = dict(
        param_rows=len(params), rows_with_evidence=sum(int(p["n_evidence"]) > 0 for p in params),
        rows_param_map=sum(p["link"] == "param_map" for p in params),
        rows_auto=sum(p["link"] == "auto_exact_name" for p in params),
        status_counts=dict(Counter(p["status"] for p in params)),
        deviations=[dict(run=p["run"], param=p["param"], used=p["used_values"], spec_default=p["spec_default"],
                         spec_range=p["spec_range"], status=p["status"], reason=p["reason_recorded"]) for p in params if p["deviation"] == "True"],
        gate_status_counts=dict(Counter(g["status"] for g in gates)),
        gate_flags=[dict(run=g["run"], file=g["file"].split("/")[-1], object=g["object"], status=g["status"],
                         missing=g["missing_gates"], boundary=g["boundary"]) for g in gates if g["status"] != "CONSISTENT"],
        phases=man["phases"], repro=man["repro"], project_repro=man["project_repro"],
    )
    rep = man["repro"]
    cis = load_csv("run_manifest_ci_scope.csv")
    wd = load_csv("run_manifest_wording.csv")
    out["run_manifest"]["m4_totals"] = dict(
        scripts=sum(int(x["n_scripts"]) for x in rep),
        scripts_using_randomness=sum(int(x["scripts_using_randomness"]) for x in rep),
        scripts_using_randomness_with_seed=sum(int(x["scripts_using_randomness_with_seed"]) for x in rep),
        summary_py_paths_ok=sum(int(x["summary_py_paths_ok"]) for x in rep),
        summary_py_paths_missing=sum(bool(x["summary_py_paths_missing"]) for x in rep),
        runs_with_device_recorded=sum(bool(x["devices_recorded"]) for x in rep),
        runs_with_timing_json=sum(int(x["json_files_with_timing"]) > 0 for x in rep),
        runs_with_decision_log=sum(x["machine_readable_decision_log"] != "none" for x in rep),
        runs_with_run_manifest=sum(x["run_manifest_file"] != "none" for x in rep),
        runs_with_manual_reclassification=sum(bool(x["manual_reclassification_files"]) for x in rep),
        runs_with_phase_record=sum(bool(p_["machine_readable_phase_record"]) for p_ in man["phases"]),
    )
    out["run_manifest"]["m5_totals"] = dict(
        summary_docs=len(cis), interval_lines=sum(int(r["lines_with_interval"]) for r in cis),
        interval_lines_naming_method=sum(int(r["interval_lines_naming_method"]) for r in cis),
        interval_lines_naming_method_and_unit=sum(int(r["interval_lines_naming_method_and_unit"]) for r in cis),
        docs_with_scope_heading=sum(r["scope_or_limitation_heading_lines"] != "none" for r in cis),
        p9_word_lines=sum(int(r["n_lines"]) for r in wd),   # term-line hits: a line with two terms counts twice
        # distinct lines (added in the verification pass); exact because no term has more than 15 lines per document
        p9_distinct_lines=len({(r["doc"], ln) for r in wd for ln in r["lines"].split(";") if ln}),
        p9_max_lines_per_term=max((int(r["n_lines"]) for r in wd), default=0),
    )
    md.append("\n## run_manifest_check: flagged parameter deviations\n")
    md.append("| run | spec parameter | value(s) used | spec default | spec range | status | reason written down? |")
    md.append("|---|---|---|---|---|---|---|")
    for d in out["run_manifest"]["deviations"]:
        md.append(f"| {d['run']} | {d['param']} | {d['used']} | {d['spec_default']} | {d['spec_range']} | {d['status']} | {d['reason'] or 'no'} |")
    md.append("\n## run_manifest_check: gate objects not simply consistent\n")
    md.append("| run | file | object | status | declared gates with no value | within 0.005 of threshold |")
    md.append("|---|---|---|---|---|---|")
    for g in out["run_manifest"]["gate_flags"]:
        md.append(f"| {g['run']} | {g['file']} | {g['object']} | {g['status']} | {g['missing']} | {g['boundary']} |")
    # ---------------- repair queue: one row per deterministic failure, with who must act
    rq = []
    for r in dep:
        v = r["verdicts"]
        if v["S1"] == "FAIL":
            rq.append(dict(check="spec_lint S1", target=r["spec"], location="section headings",
                           finding=f"missing: {r['sections']['missing']}; numbering mismatches: {len(r['sections']['heading_number_mismatches'])}",
                           action="add the missing section in a versioned copy of the spec", owner="agent (edit) + human (approve)"))
        if v["S3"] in ("WARN", "FAIL", "WAIVER"):
            rq.append(dict(check="spec_lint S3", target=r["spec"], location="Source section",
                           finding=r["pin"]["status"], action=("restore .git in the clone or record a content hash" if v["S3"] == "WARN"
                                                               else "human approves the no-repository waiver" if v["S3"] == "WAIVER"
                                                               else "add a resolvable pinned commit"),
                           owner="human" if v["S3"] == "WAIVER" else "agent"))
        c = r["code_refs"]
        if c["n_required_form"] < c["n_refs"]:
            rq.append(dict(check="spec_lint S4", target=r["spec"], location="code references",
                           finding=f"{c['n_refs'] - c['n_required_form']} of {c['n_refs']} not in repos/<name>/path:LINE form",
                           action="rewrite as repos/<name>/path:LINE", owner="agent"))
        for m in c["missing"]:
            rq.append(dict(check="spec_lint S4", target=r["spec"], location=m, finding="path does not resolve in the pinned clone",
                           action="fix the path", owner="agent"))
        for sname in c["symbols_not_found"]:
            rq.append(dict(check="spec_lint S4", target=r["spec"], location=sname, finding="named function not defined in the cited file",
                           action="point to the file that defines it", owner="agent"))
        if v["S6"] == "FAIL":
            rq.append(dict(check="spec_lint S6", target=r["spec"], location="Validation", finding="no machine-readable validation block",
                           action="add a validation block with kind/artefact/field/op/value/on_fail", owner="agent (draft) + human (thresholds)"))
        missing_slots = [k for k in ("null_model", "trivial_baseline", "positive_control", "scope_statement") if not r["slots"][k]["has_labelled_slot"]]
        if missing_slots:
            rq.append(dict(check="spec_lint S7", target=r["spec"], location="whole spec", finding=f"no labelled slot for {missing_slots}",
                           action="add labelled slots", owner="agent (draft) + human (content)"))
    for r in rs:
        if r["traced"] == "False" and r["also_in_spec"] == "False":
            rq.append(dict(check="number_trace", target=r["doc"], location=f"line {r['line']}: {r['raw']}",
                           finding="number not found in this run's outputs within rounding", action="trace to a file or correct the text",
                           owner="agent"))
    for d in out["run_manifest"]["deviations"]:
        rq.append(dict(check="run_manifest M1", target=d["run"], location=d["param"],
                       finding=f"used {d['used']} vs spec {d['spec_default']} (range {d['spec_range']})",
                       action=("reason is written down; human approves or orders a re-run" if d["reason"] else
                               "write the reason; human approves or orders a re-run"), owner="human"))
    for g in out["run_manifest"]["gate_flags"]:
        if g["status"] == "CONSISTENT; GATE_SET_FOR_PANEL_UNSTATED":
            action, owner = "decide which frozen gates apply to this panel (the gate file names only internal and external)", "human"
        else:
            action, owner = "compute the missing gate or correct the recorded verdict", "agent (compute) + human (accept)"
        rq.append(dict(check="run_manifest M2", target=g["run"], location=f"{g['file']}:{g['object']}", finding=g["status"],
                       action=action, owner=owner))
    with open(RESULTS / "repair_queue.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "target", "location", "finding", "action", "owner"])
        w.writeheader()
        w.writerows(rq)
    out["repair_queue"] = dict(n_items=len(rq), by_check=dict(Counter(x["check"] for x in rq)),
                               by_owner=dict(Counter(x["owner"] for x in rq)))
    write_json(RESULTS / "checker_summary.json", out)
    (RESULTS / "checker_summary.md").write_text("\n".join(md) + "\n")
    inv = PROJ / "paper-plos-one" / "revision" / "investigation"
    ins = sorted(p for p in RESULTS.glob("*") if p.is_file() and not p.name.startswith("._")
                 and p.name not in ("checker_summary.json", "checker_summary.md", "repair_queue.csv", "summary_run_config.json"))
    ins += [inv / "numbers_A.md", inv / "numbers_B.md", inv / "numbers_C.md", inv / "contract" / "spec_lint_summary.csv"]
    ins += sorted((Path(__file__).resolve().parent / "verify").glob("*.py"))
    write_json(RESULTS / "summary_run_config.json", run_config("summarize_results.py", ins, extra={
        "note": "inputs = all result files of the three checks and the verify scripts, the manual audits used by "
                "verify/crosscheck_manual_audit.py, and the prototype linter output used by verify/verify_spec_lint.py"}))
    print(json.dumps({k: v for k, v in out["spec_lint"].items() if not isinstance(v, dict) or len(v) < 8}, indent=1, default=str)[:3000])
    print(json.dumps({k: v for k, v in out["number_trace"].items() if k not in ("docs", "paper", "classes")}, indent=1)[:2500])
    print(json.dumps({k: v for k, v in out["run_manifest"].items() if k not in ("phases", "repro")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
