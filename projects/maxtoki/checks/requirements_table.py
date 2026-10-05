#!/usr/bin/env python3
"""Write REQUIREMENTS_TABLE.md and requirements_table.csv from one list of rows.

Each row: one requirement the framework imposes -> what was in place in the deployment ->
how it is checked now (code / agent judgment / human decision) -> how non-compliance is detected ->
what triggers repair -> what the check does NOT establish.
Numbers in the rows are read from results/checker_summary.json (this package's outputs). Numbers
marked [inv:<report>] come from the investigation reports in verification/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "checker_summary.json").read_text())
L, N, M = S["spec_lint"], S["number_trace"], S["run_manifest"]
D = N["distinct_per_run"]
M4, M5 = M["m4_totals"], M["m5_totals"]
XC = S.get("crosscheck_manual_audit", {})
X = XC.get("by_manual_status", {})
wrong = X.get("MISMATCH", {}).get("n", 0) + X.get("DIFFERENT ENDPOINT", {}).get("n", 0)
wrong_traced = X.get("MISMATCH", {}).get("traced_any_run", 0) + X.get("DIFFERENT ENDPOINT", {}).get("traced_any_run", 0)
paper_union = next(r for r in N["paper"] if r["scope"].startswith("paper_vs_union_of"))
paper_chain = next(r for r in N["paper"] if r["scope"].startswith("paper_chain"))
n_dev = len(M["deviations"])
n_dev_reason = sum(bool(d["reason"]) for d in M["deviations"])
n_dev_params = len({(d["run"], d["param"]) for d in M["deviations"]})
gate_unc = sum("PASS_WITH_UNCOMPUTED_GATES" in g["status"] for g in M["gate_flags"])

CODE, JUDG, HUMAN = "code", "agent judgment", "human decision"

ROWS = [
    # --- spec template
    dict(id="T1", group="Spec template", requirement="All ten template sections present, in order, numbered 1-10",
         source="CLAUDE.md:45; AGENTS.md:21; paper main.tex:587-588 says 'enforced'",
         deployment="Prose instruction to the agent only. No code read the specs. All 8 specs were executed.",
         deployment_result=f"{L['template_reject']} of 8 deployed specs lack the 'Code references' section (and their later headings are numbered one lower than the template position); only longevity complies.",
         checked_by=CODE, detection="spec_lint.py S1 = FAIL (template_verdict REJECT); with --strict the exit code is 1.",
         repair_trigger="A row in results/repair_queue.csv; the agent edits a versioned copy of the spec, a human approves. Deployed specs are not edited retroactively.",
         not_established="That a present section is correct or sufficient."),
    dict(id="T2", group="Spec template", requirement="Every section substantive (not empty, no TBD)",
         source="CLAUDE.md:45 ('present and substantive')",
         deployment="Agent judgment when writing the spec; nothing checked it.",
         deployment_result=f"All 8 deployed specs pass the crude test (>= 200 characters, no TBD) [S2: {L['S2']}].",
         checked_by=CODE + " (length/marker only) + " + JUDG + " (substance)", detection="spec_lint.py S2 = FAIL.",
         repair_trigger="repair_queue row; agent fills the section.",
         not_established="Substance. 200 characters of wrong text passes."),
    dict(id="T3", group="Pinned code", requirement="Source section names a pinned commit that resolves",
         source="CLAUDE.md Source section rule; repos/<name>/PINNED_COMMIT.txt",
         deployment="Agent wrote the hash into the spec and PINNED_COMMIT.txt. No check. Clones copied without .git.",
         deployment_result=f"Pin status over 8 specs: {L['pin_status']}. For the {L['pin_status'].get('TEXT_MATCH_ONLY_NO_GIT', 0)} text-only cases the files on disk cannot be proven to be that commit.",
         checked_by=CODE + "; waiver = " + HUMAN, detection="spec_lint.py S3: FAIL (no/unknown hash), WARN (text match, clone has no .git), WAIVER (explicit 'no repository').",
         repair_trigger="WARN: restore .git or record a content hash (agent). WAIVER: a human accepts or rejects the no-repository reason.",
         not_established="That the pinned code is the code the executor actually read."),
    dict(id="T4", group="Pinned code", requirement="Code pointers in the form repos/<name>/path:LINE; the path and line exist",
         source="CLAUDE.md:58",
         deployment="Convention only.",
         deployment_result=(f"{L['code_refs_required_form']} of {L['code_refs_total']} code pointers use the required form; "
                            f"{L['code_refs_exist_as_written']} resolve as written, {L['code_refs_exist'] - L['code_refs_exist_as_written']} only by path suffix or file name, "
                            f"{L['code_refs_missing']} not at all, {L['code_refs_pattern_unchecked']} are wildcards. {L['symbols_found_in_file']} of {L['symbols_checked']} named functions are defined in the cited file."),
         checked_by=CODE, detection="spec_lint.py S4 = FAIL; spec_lint_coderefs.csv lists every pointer with its status; --strict exits 1 on a path that does not resolve.",
         repair_trigger="repair_queue rows (format, missing path, function not in file); agent fixes.",
         not_established="That the cited code implements what the spec says; line checks run on the working tree, which is unproven for clones without .git."),
    dict(id="T5", group="Parameters", requirement="Parameters table with defaults and valid ranges",
         source="CLAUDE.md template item 8",
         deployment="Tables written by the agent; free-text cells; ranges optional.",
         deployment_result=f"{L['param_rows']} rows over 8 specs; {L['param_default_numeric']} defaults and {L['param_range_numeric']} ranges can be read as numbers; {L['S5'].get('FAIL', 0)} spec has a duplicate parameter name; one table (02) has unescaped pipe characters inside a cell, which breaks its columns.",
         checked_by=CODE, detection="spec_lint.py S5; spec_lint_params.csv shows how each cell was parsed.",
         repair_trigger="repair_queue row; agent rewrites cells as numbers/ranges; human sets ranges.",
         not_established="That a range is scientifically sensible."),
    dict(id="T6", group="Parameters", requirement="Values used in the run equal the spec defaults, or the deviation is justified",
         source="Paper main.tex:617-621 ('no latitude over what counts as the null')",
         deployment="No run manifest. The agent chose values; a few deviations were noted in code comments or a README. No check.",
         deployment_result=(f"run_manifest_check.py linked {M['rows_with_evidence']} of {M['param_rows']} rows to run evidence. "
                            f"{n_dev_params} parameters deviate ({n_dev} spec rows incl. a duplicate row); a written reason was found for {n_dev_reason}. Examples: Curveball null 50 vs 200; "
                            "label-permutation null 8 and 50 vs 100-1,000; rewiring null 8 and 12 vs >= 24; 50 vs 200 cells per ablation condition."),
         checked_by=CODE + " (comparison) + " + JUDG + " (param_map.json links) + " + HUMAN + " (approve deviation)",
         detection="run_manifest_params.csv status OUT_OF_RANGE or DIFFERS_NO_RANGE_GIVEN; --strict exits 1 if no reason is written.",
         repair_trigger="repair_queue row owned by a human: approve with the written reason, or order a re-run.",
         not_established=f"Coverage: {M['param_rows'] - M['rows_with_evidence']} rows have no linked evidence. A constant in a script may have been edited after the run."),
    # --- analysis content
    dict(id="A1", group="Nulls", requirement="Every claim has a null of the stated family and size, and the null really randomises",
         source="Specs' Methodology; audit P1",
         deployment="Spec text + audit agent reading (P1). No code.",
         deployment_result="The Curveball null left the network unchanged in 49 of 50 draws and was not caught by the audit [inv:attn_endpoints]. Null sizes below spec (see T6).",
         checked_by=CODE + " (size only, M1) + " + JUDG + " (correctness)",
         detection="Size: run_manifest_check M1. Correctness: not detected by this package. Proposed: implementation items in a machine-readable validation block (e.g. 'fraction of null draws that differ from the input == 1').",
         repair_trigger="Size: repair_queue (human). Correctness: audit finding by an agent, then re-run.",
         not_established="That any null is valid. This is the largest gap between code and judgment."),
    dict(id="A2", group="Baselines", requirement="A trivial / gene-level baseline on the same records (P2)",
         source="audit P2; spec prescriptions",
         deployment="In some spec phases (attention Phase 1); audit P2 by reading.",
         deployment_result="The constant 'always decrease' baseline for CRISPRi sign (54.61% vs the reported 53.42%) was missing and not caught [inv:crispri].",
         checked_by=CODE + " (labelled slot present in spec, S7) + " + JUDG,
         detection=f"spec_lint S7: {L['slots_labelled']['trivial_baseline']} of 8 specs have a labelled trivial-baseline slot. Run level: not detected by code.",
         repair_trigger="repair_queue (S7) for the spec; run-level gaps only through an audit finding.",
         not_established="That the baseline used is the right one."),
    dict(id="A3", group="Positive controls", requirement="A positive control for every load-bearing negative (P6)",
         source="audit P6; paper main.tex:594-600, 1605-1609 ('mandatory slot')",
         deployment="No template slot existed. The topology control was added after the run as audit action A1.",
         deployment_result=f"spec_lint S7: {L['slots_labelled']['positive_control']} of 8 specs have a labelled positive-control slot.",
         checked_by=CODE + " (slot present) + " + JUDG + " (adequacy)", detection="spec_lint S7 = FAIL.",
         repair_trigger="repair_queue (agent drafts, human decides content).", not_established="That a control can detect the effect size of interest."),
    dict(id="A4", group="Intervals", requirement="Every headline number has a CI with a stated method and resampling unit (P8, P4)",
         source="audit P8/P4; revision rules",
         deployment="Audit P8 by reading; action A3 computed intervals for some numbers.",
         deployment_result=(f"In the 13 FINAL_SUMMARY files, {M5['interval_lines']} lines mention an interval; {M5['interval_lines_naming_method']} name a method; "
                            f"{M5['interval_lines_naming_method_and_unit']} name a method and a unit. The manifold 'bootstrap' was 200 draws of 80% anchor subsampling [inv:manifold_ci]."),
         checked_by=CODE + " (words present) + " + JUDG + " (method correct)",
         detection="run_manifest_ci_scope.csv counts; a line with an interval and no method word is a pointer.",
         repair_trigger="Reader or agent follows the pointer; no automatic repair.",
         not_established="That the method or unit is right. A text match is weak evidence."),
    dict(id="A5", group="Scope", requirement="Scope statement: what was run, what was not, which model",
         source="audit P7; paper 'scope qualifier slot'",
         deployment="Agent prose in summaries; the attention verdict JSON had a phases_run list.",
         deployment_result=(f"{M5['docs_with_scope_heading']} of {M5['summary_docs']} FINAL_SUMMARY files have a scope/limitation heading. Only {M4['runs_with_phase_record']} of 8 runs has a machine-readable record of phases run "
                            "(attention: 8 of 14 spec phases not run)."),
         checked_by=CODE + " (presence, phase coverage) + " + JUDG + " (adequacy)",
         detection="run_manifest_phases.csv (phases_not_run); run_manifest_ci_scope.csv; spec_lint S7 scope slot.",
         repair_trigger="Missing record -> agent adds phases_run to the run output.", not_established="That the stated scope matches the claims made elsewhere."),
    dict(id="A6", group="Wording", requirement="No causal / mechanistic overreach (P9)",
         source="audit P9",
         deployment="Audit P9 by reading; verb swaps in summaries.",
         deployment_result=f"{M5['p9_distinct_lines']} distinct lines in the 13 FINAL_SUMMARY files contain a word from the P9 list ({M5['p9_word_lines']} term hits; pointers only).",
         checked_by=CODE + " (word list) + " + JUDG + " + " + HUMAN + " (final wording)",
         detection="run_manifest_wording.csv lists term and line.", repair_trigger="Reader decides; no automatic repair.",
         not_established="Whether a flagged word is wrong in context; overreach without the listed words."),
    dict(id="A7", group="Gates", requirement="Pre-registered gates computed from artefacts; never loosened; all declared gates evaluated",
         source="manifold spec 'Critical design rules'; runs/manifold-discovery-217M/reports/quality_gates_spec.json",
         deployment="Manifold: a frozen gate file (agent-written) and pass flags computed by run scripts. Attention: phase12_verdict.py computed C1-C4 by code. Nothing re-checked the flags.",
         deployment_result=(f"{gate_unc} manifold internal-panel objects record PASS while the declared permutation gate (required on the internal panel) was never computed; "
                            "the external-panel verdicts need only four gates under the frozen rules; the first lung negative control (lung_control) "
                            "passed, and the run itself rebuilt it as lung_nonhema, which failed as expected (FINAL_SUMMARY lines 39-40); "
                            "H103 used k=5 for trustworthiness vs the frozen k=15 (reason written in the file: 18 anchors); "
                            "attention K562 C3 stores a 'fraction' of -13,421."),
         checked_by=CODE, detection="run_manifest_gates.csv status INCONSISTENT / PASS_WITH_UNCOMPUTED_GATES / NEGATIVE_CONTROL_PASSED / PROTOCOL_K_DIFFERS / IMPLAUSIBLE_FRACTION; --strict exits 1.",
         repair_trigger="repair_queue row: agent computes the missing gate or corrects the verdict; human accepts.",
         not_established="Gates stored in prose or in non-standard JSON; the attention C4 object is ambiguous and not recomputed."),
    dict(id="A8", group="Validation", requirement="Validation signatures usable as pass/fail checks on the target model",
         source="CLAUDE.md template item 9",
         deployment="Validation sections written as source-model replication targets; no MaxToki pre-flight was run.",
         deployment_result=f"{L['val_items']} items; {L['val_threshold']} have a numeric threshold; {L['val_framed_as_source_replication']} of 8 sections are framed as source-model replication; {L['val_machine_readable_blocks']} machine-readable blocks.",
         checked_by=CODE + " (block present) + " + HUMAN + " (thresholds)", detection="spec_lint S6 = FAIL (no block).",
         repair_trigger="repair_queue: agent drafts a block (kind: implementation / source_replication / new_model_gate), human sets thresholds.",
         not_established="That the thresholds are right."),
    # --- numbers
    dict(id="N1", group="Number provenance", requirement="Every number in a run summary comes from that run's outputs",
         source="Paper main.tex:652-656 (human spot-check); audit pitfall 2",
         deployment="Claimed human spot-checks; no record of any [inv:audit_history]. The audit 'stopped at the markdown layer'.",
         deployment_result=(f"{D['traced']} of {D['n_numbers']} distinct summary numbers (per run) are found within rounding (Wilson 95% CI {D['traced_wilson'][0]:.3f}-{D['traced_wilson'][1]:.3f}); "
                            f"but random numbers of the same format are found {D['mean_chance']:.2f} of the time (95% CI over runs {D['mean_chance_cluster_ci'][0]:.2f}-{D['mean_chance_cluster_ci'][1]:.2f}); "
                            f"excess over chance {D['excess']:.2f} (95% CI over runs {D['excess_cluster_ci'][0]:.2f}-{D['excess_cluster_ci'][1]:.2f}). "
                            f"{D['untraced_not_in_spec']} distinct numbers that are not spec quotes are untraced."),
         checked_by=CODE, detection="number_trace_numbers.csv traced=False; WEAK flag when the number's own chance rate >= 0.5; --strict exits 1.",
         repair_trigger="repair_queue row per untraced number; agent traces it or corrects the text.",
         not_established=(f"Correctness or meaning. {wrong_traced} of {wrong} paper number mentions "
                          f"({XC.get('n_distinct_values_wrong_in_meaning_traced', '?')} of {XC.get('n_distinct_values_wrong_in_meaning', '?')} distinct values) "
                          "that a manual audit judged wrong in meaning are still 'found' in the outputs.")),
    dict(id="N2", group="Number provenance", requirement="Every number in the paper matches a run summary and that run's outputs",
         source="Revision plan section 7 (RESULTS_MAP)",
         deployment="None.",
         deployment_result=(f"Paper vs union of 8 runs: {paper_union['n_traced']}/{paper_union['n_numbers']} found, chance {float(paper_union['mean_chance']):.2f} (uninformative). "
                            f"Paper -> summary -> same run's outputs: {paper_chain['n_traced']}/{paper_chain['n_numbers']}, chance {float(paper_chain['mean_chance']):.2f}."),
         checked_by=CODE, detection="number_trace_paper_numbers.csv (chain_trace, in_summaries).",
         repair_trigger="Row without a chain -> trace or fix; the future RESULTS_MAP.csv replaces this heuristic.", not_established="As N1."),
    # --- reproducibility
    dict(id="R1", group="Reproducibility", requirement="Seeds fixed in every script that uses randomness",
         source="audit P10; CLAUDE.md", deployment="Scripts define SEED = 42; no check.",
         deployment_result=f"{M4['scripts_using_randomness_with_seed']} of {M4['scripts_using_randomness']} scripts that use randomness also set a seed-like value.",
         checked_by=CODE, detection="run_manifest_repro.csv scripts_random_without_seed.", repair_trigger="Agent adds the seed.",
         not_established="That every random call uses the seed."),
    dict(id="R2", group="Reproducibility", requirement="Environment pinned", source="audit P10",
         deployment="projects/maxtoki/requirements.txt, written on the audit day after the runs.",
         deployment_result=f"{M['project_repro'].get('project_requirements_mtime')} file date; all lines use ==; no per-run environment file.",
         checked_by=CODE, detection="run_manifest_repro.csv; run_manifest.json project_repro.", repair_trigger="Agent records the environment at run time (manifest).",
         not_established="That the runs used this environment."),
    dict(id="R3", group="Reproducibility", requirement="Model checkpoint pinned (revision or hash)", source="Paper claim E34; audit P10",
         deployment="Weights downloaded with curl from the Hugging Face 'main' branch; loaded from a local folder.",
         deployment_result=f"revision pins: {M['project_repro'].get('revision_pins')}; download refs: {M['project_repro'].get('download_refs_in_readmes')}; recorded weight hashes: {M['project_repro'].get('files_mentioning_sha256')}.",
         checked_by=CODE, detection="run_manifest.json project_repro.", repair_trigger="Agent records the commit id and sha256; human confirms.",
         not_established="Which commit was downloaded (the investigation matched the sha256 separately)."),
    dict(id="R4", group="Reproducibility", requirement="Device / hardware recorded", source="audit P10; paper hardware statement",
         deployment="Some run_config.json files record 'device'.",
         deployment_result=f"{M4['runs_with_device_recorded']} of 8 runs record a device (mps, cpu); none records an NVIDIA GPU. The paper said A100 [inv:hardware].",
         checked_by=CODE + " (what the files say) + " + HUMAN + " (paper text)", detection="run_manifest_repro.csv devices_recorded.",
         repair_trigger="Agent adds device to every run manifest.", not_established="Paper claims are not parsed by this package."),
    dict(id="R5", group="Reproducibility", requirement="Script paths cited in summaries resolve (P10)", source="audit P10 prescription",
         deployment="Intended as 'one audit-time check', done by reading.",
         deployment_result=f"{M4['summary_py_paths_ok']} cited script paths, {M4['summary_py_paths_missing']} missing; most summaries cite no script at all.",
         checked_by=CODE, detection="run_manifest_repro.csv summary_py_paths_missing.", repair_trigger="Agent fixes path.",
         not_established="That the cited script produced the cited number."),
    dict(id="R6", group="Reproducibility", requirement="Wall time recorded per phase (P10)", source="audit P10",
         deployment="Some JSON files record seconds; the paper's agent hours were typed by hand.",
         deployment_result=f"{M4['runs_with_timing_json']} of 8 runs have at least one JSON file with a timing field.",
         checked_by=CODE, detection="run_manifest_repro.csv json_files_with_timing.", repair_trigger="Agent adds timing to the manifest.",
         not_established="Total compute time."),
    dict(id="R7", group="Reproducibility", requirement="Run manifest (spec hash, parameters used, seeds, phases, device)",
         source="Proposed in verification/contract.md section 8 (B1)", deployment="Did not exist.",
         deployment_result=f"{M4['runs_with_run_manifest']} of 8 runs have a run_manifest*.json.", checked_by=CODE,
         detection="run_manifest_repro.csv run_manifest_file = none.", repair_trigger="Future runs must write one; the checker then compares by exact name.",
         not_established="Honesty of a self-written manifest (it still needs M2 and N1)."),
    dict(id="R8", group="Reproducibility", requirement="Code, configs, logs and intermediates public (P10)", source="Paper data statement",
         deployment="Public repo excluded runs/ and setup/ [inv:repro_inventory].", deployment_result="Not checkable from local files.",
         checked_by=HUMAN, detection="Not detected by this package.", repair_trigger="Release decision (outward-facing; human).",
         not_established="-"),
    # --- process
    dict(id="P-loop", group="Process", requirement="Retire a hypothesis family after 2 consecutive negatives",
         source="Paper main.tex:640-641; spec autoloop_retirement_policy",
         deployment="Not implemented for topology or manifold. The spectral loop's '2 in a row' rule counted crashed turns; topology used a coded 3-in-a-row rule, then a manual review changed 4 of 6 decisions [inv:contract].",
         deployment_result=f"{M4['runs_with_decision_log']} of 8 runs have a machine-readable decision/retirement log; {M4['runs_with_manual_reclassification']} run has a MANUAL_REVIEW file that overrode coded decisions.",
         checked_by="not implemented (code only checks whether a decision log exists)", detection="run_manifest_repro.csv machine_readable_decision_log / manual_reclassification_files.",
         repair_trigger="Proposed: the loop driver applies the rule and writes retired.json; overrides go to decisions.jsonl with who/why.",
         not_established="Whether any retirement was correct."),
    dict(id="P-gate", group="Process", requirement="Human go/no-go on expensive operations", source="Paper main.tex:572-575, 645-660",
         deployment="Claimed; not logged. One recorded human decision in run files (manifold STATUS.md:123) [inv:contract].",
         deployment_result="No decision log in any run.", checked_by=HUMAN, detection="Only absence of a log is detectable (R7/P-loop).",
         repair_trigger="Proposed: decisions.jsonl written at each go/no-go.", not_established="That decisions were made at all."),
    dict(id="P-repair", group="Process", requirement="Audit findings lead to repair and re-check",
         source="audit spec section 6.11 and section 9",
         deployment="Two human prompts started two repair rounds in one afternoon; no re-check by code; the 75% 'addressed' figure has no cell table [inv:audit_history].",
         deployment_result=f"This package's repair queue has {S['repair_queue']['n_items']} items: {S['repair_queue']['by_check']}.",
         checked_by=CODE + " (queue) + " + HUMAN + " (closing items that need approval)",
         detection="results/repair_queue.csv regenerated on every run; an item closes only when its check passes on re-run.",
         repair_trigger="Each row names an owner (agent / human).", not_established="That a repair is correct (a re-run can still be wrong)."),
]
# --- audit patterns P1-P10
P = [
    ("P1", "Null too weak / not anchored to chance", CODE + " (null size vs spec, M1) + " + JUDG,
     "Null size deviations (T6); nothing tests that a null randomises (A1)."),
    ("P2", "Trivial / gene-level baseline absent", JUDG + " (+ spec slot presence, S7)", "S7 slot only."),
    ("P3", "Endpoint-object mismatch", JUDG, "Not detectable by code; number_trace cannot see meaning (N1)."),
    ("P4", "Pseudoreplication / wrong unit / leaky CV", JUDG + " (+ unit words near intervals, M5)", "Only word presence."),
    ("P5", "Selection-driven inflation / double dipping", JUDG, "Not detected by code."),
    ("P6", "No positive control for null findings", JUDG + " (+ spec slot presence, S7)", "S7 slot only."),
    ("P7", "Single-condition generalisation", JUDG + " (+ phases-run coverage, M3)", "Only phase coverage where a phases_run list exists."),
    ("P8", "Stability of headline numbers (CIs, sweeps)", JUDG + " (+ interval/method words, M5)", "Only word presence."),
    ("P9", "Causal / mechanistic language overreach", JUDG + " (+ word list, M5)", "Pointers only."),
    ("P10", "Reproducibility scaffolding", CODE + " (S3, S4, M4) + " + HUMAN + " (release)", "Paths, pins, seeds, env, device, timing, manifest, checkpoint pin."),
]
for pid, name, how, adds in P:
    ROWS.append(dict(id=pid, group="Audit pattern", requirement=f"{pid}: {name}", source="pipelines/audit-recurring-review-issues.md section 6",
                     deployment="One agent session applied all ten patterns by reading, in about 4 hours, with free-text verdicts (17 labels); no code [inv:audit_history].",
                     deployment_result="See the finding-level rows above and the error list in the revision plan (E1-E42).",
                     checked_by=how, detection=f"New code covers: {adds}",
                     repair_trigger="Code-detected parts: repair_queue. Judgment parts: an audit finding, then a human decides re-run vs re-word.",
                     not_established="Whether the pattern is present; that remains a judgment."))

COLS = ["id", "group", "requirement", "source", "deployment", "deployment_result", "checked_by", "detection",
        "repair_trigger", "not_established"]
HEAD = ["ID", "Group", "Requirement", "Where it is stated", "What was in place in the deployment", "What the files show",
        "Checked now by", "How non-compliance is detected (new)", "What triggers repair (new)", "What this does NOT establish"]


def main():
    with open(HERE / "requirements_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(ROWS)
    md = ["# Requirements table: what is checked by code, by agent judgment, or by a human",
          "",
          "Generated by `requirements_table.py` from `results/checker_summary.json`. One row per requirement the",
          "framework imposes. 'Deployment' = what existed when the eight MaxToki runs and the audit were done",
          "(April-May 2026). 'Checked now by' = with this checker package. `[inv:X]` = number taken from the",
          "investigation report `verification/X.md`, not recomputed here.",
          "",
          "Counts of how each requirement is checked now:",
          ""]
    cnt = {}
    for r in ROWS:
        k = ("not implemented" if r["checked_by"].startswith("not implemented") else
             "code only" if r["checked_by"] == CODE else
             "code + judgment/human" if CODE in r["checked_by"] else
             "judgment only" if r["checked_by"] == JUDG else
             "human only" if r["checked_by"] == HUMAN else "judgment (+ weak code pointers)" if JUDG in r["checked_by"] else r["checked_by"])
        cnt[k] = cnt.get(k, 0) + 1
    for k, v in cnt.items():
        md.append(f"- {k}: {v}")
    md += ["", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    for r in ROWS:
        md.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in COLS) + " |")
    md += ["", "## Plain-words summary", "",
           "During the deployment, nothing in this table was checked by code except a few things inside single runs:",
           "the manifold and attention runs computed their own pass/fail flags from their numbers, one loop driver",
           "checked that output files existed, and one scripted topology batch applied a coded promote/retire rule",
           "that a later manual review overrode for 4 of 6 hypotheses. Everything else was an instruction to the",
           "agent, or the agent's own reading during a one-afternoon audit. The new package checks the mechanical parts by code:",
           "sections, pins, code pointers, parameter values against the spec, gate flags, seeds, paths and whether",
           "numbers can be found in the outputs. It cannot check whether a null is valid, whether a baseline is the",
           "right one, whether a number means what the text says, or whether wording is too strong. Those stay",
           "agent judgment, and approving deviations, waivers and thresholds stays a human decision."]
    (HERE / "REQUIREMENTS_TABLE.md").write_text("\n".join(md) + "\n")
    print(f"{len(ROWS)} rows written; {cnt}")


if __name__ == "__main__":
    main()
