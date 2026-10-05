#!/usr/bin/env python3
"""run_manifest_check.py -- what did each deployed run actually use, and does it match its spec?

Deployed runs wrote no run manifest, so this check rebuilds one from the files (deployment
snapshot only; see common.in_deployment_snapshot) and compares it with the spec.

M1  Parameter conformance. For every row of the spec's Parameters table: find the value(s) the run
    used -- (a) links listed in param_map.json (hand-curated, reviewed, with a reason) and
    (b) automatic exact-name matches (spec name == script constant or JSON key, ignoring case and
    underscores). Recorded JSON values take priority over script constants; constants inside an
    `if` block (e.g. smoke-test settings) are listed but never judged. Each value is compared with
    the spec's parsed default and valid range:
        MATCH_DEFAULT | IN_RANGE_NOT_DEFAULT | OUT_OF_RANGE | DIFFERS_NO_RANGE_GIVEN |
        SPEC_NOT_MACHINE_READABLE | DESCRIPTOR_NOT_JUDGED | NO_EVIDENCE_FOUND
    A deviation (OUT_OF_RANGE or DIFFERS_NO_RANGE_GIVEN on a `setting`/`threshold` row) gets a
    `reason_recorded` flag: does any line of the run's scripts/README mention both the used and the
    spec value next to a word like vs/instead/scoped/budget/reduced?
M2  Gate recomputation. Every JSON object that stores a pass/fail flag together with thresholds (or
    that sits in a run with a frozen gates file) is re-evaluated from its own numbers. Flags:
    recorded flag disagrees with the recomputed one; a declared gate has no value (never computed);
    a value within 0.005 of its threshold ("boundary"). Which declared gates are required depends on
    the panel when the frozen file says so (manifold: five gates on the internal panel, four on the
    external panel); a panel the frozen rules do not name gets GATE_SET_FOR_PANEL_UNSTATED (a pointer
    for a human, not blocking).
M3  Phase coverage: phases named in the spec's Methodology vs a machine-readable `phases_run` list
    in the run outputs (if any).
M4  Reproducibility records: seeds set in scripts that use randomness; a pinned environment file;
    device recorded; model checkpoint revision pinned in code; script paths cited in the
    FINAL_SUMMARY resolve; wall time recorded; machine-readable decision/retirement log present.
M5  Wording flags: causal / mechanistic words from the audit checklist's P9 list, counted per
    FINAL_SUMMARY. These are pointers for a reader, not errors.

What it does NOT establish: that a matched constant was the value in force when the reported
number was produced (scripts can be edited after running); that the param_map links are complete;
that an in-range value is scientifically adequate; that a flagged word is wrong in context.

Exit code: 0 (report mode). With --strict: 1 if any deviation has no written reason or any gate object is
INCONSISTENT / PASS_WITH_UNCOMPUTED_GATES / NEGATIVE_CONTROL_PASSED / IMPLAUSIBLE_FRACTION.

Outputs (checks/results/): run_manifest_params.csv, run_manifest_gates.csv, run_manifest_repro.csv,
run_manifest_wording.csv, run_manifest.json, run_manifest_run_config.json
"""
from __future__ import annotations

import csv
import fnmatch
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEPLOYED, PROJ, RESULTS, ROOT, in_deployment_snapshot, rel,  # noqa: E402
                    run_config, summary_docs, write_json)
from run_evidence import collect, norm, run_files  # noqa: E402
from spec_parse import describe, parse_param_table, satisfies, section_map  # noqa: E402

SEVERITY = ["MATCH_DEFAULT", "IN_RANGE_NOT_DEFAULT", "SPEC_NOT_MACHINE_READABLE", "DIFFERS_NO_RANGE_GIVEN",
            "OUT_OF_RANGE"]
REASON_WORDS = re.compile(r"\b(vs\.?|versus|instead|scoped|budget|reduced|compute|because|down from|paper uses|"
                          r"source paper|smaller|fewer)\b", re.I)


# ------------------------------------------------------------------ M1
def value_status(v, default, rng):
    """Compare one used value (number or list) with the spec default and range."""
    if isinstance(v, list):
        if default is not None and default["kind"] == "set":
            return "MATCH_DEFAULT" if sorted(v) == sorted(default["vals"]) else "DIFFERS_NO_RANGE_GIVEN"
        return "SPEC_NOT_MACHINE_READABLE"
    if default is None and rng is None:
        return "SPEC_NOT_MACHINE_READABLE"
    if default is not None and satisfies(v, default):
        return "MATCH_DEFAULT"
    if rng is not None:
        return "IN_RANGE_NOT_DEFAULT" if satisfies(v, rng) else "OUT_OF_RANGE"
    return "DIFFERS_NO_RANGE_GIVEN"


def threshold_status(v, default):
    """For cut-off parameters compare the cut-off value itself (e.g. spec '> 0.7' vs code 0.7)."""
    if default is None or isinstance(v, list):
        return "SPEC_NOT_MACHINE_READABLE"
    k = default["kind"]
    ref = {"value": default.get("v"), "min": default.get("lo"), "max": default.get("hi")}.get(k)
    if ref is None:
        return value_status(v, default, None)
    return "MATCH_DEFAULT" if abs(v - ref) <= 1e-12 + 1e-9 * abs(ref) else "DIFFERS_NO_RANGE_GIVEN"


def find_evidence(ev, spec_name, mapping):
    items = []
    consts, leaves = ev["constants"], ev["json_leaves"]
    used = set()
    for e in (mapping or {}).get("evidence", []):
        glob = e.get("file_glob")
        if e["source"] == "json":
            for it in leaves:
                if (it["name"] == e["key"] or it["name"].endswith("." + e["key"])) and (not glob or fnmatch.fnmatch(it["file"], glob)):
                    items.append(dict(it, link="param_map"))
                    used.add(id(it))
        else:
            for it in consts:
                if it["name"].lower() == e["name"].lower() and (not glob or fnmatch.fnmatch(it["file"], glob)) \
                        and (not e.get("kind_prefix") or it["kind"].startswith(e["kind_prefix"])):
                    items.append(dict(it, link="param_map"))
                    used.add(id(it))
    if not mapping:
        # automatic exact-name matches (only when the row has no curated entry)
        for it in consts + leaves:
            last = it["name"].split(".")[-1]
            if norm(last) == norm(spec_name) and id(it) not in used:
                items.append(dict(it, link="auto_exact_name"))
    return items


def reason_recorded(run_dir: Path, used, spec_vals):
    """Is there a line in the run's scripts/README that names both values and a reason word?"""
    if not spec_vals:
        return None
    def fmt(x):
        return {("%g" % x), (f"{int(x):,}" if float(x).is_integer() else "%g" % x)}
    u = fmt(used)
    s = set().union(*[fmt(x) for x in spec_vals])
    files = run_files(run_dir, ".py") + [p for p in run_dir.glob("README*.md") if in_deployment_snapshot(p)]
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for k, ln in enumerate(lines):
            nums = set(re.findall(r"\d[\d,]*(?:\.\d+)?", ln))
            nums |= {n.replace(",", "") for n in nums}
            if (u & nums) and (s & nums) and REASON_WORDS.search(ln):
                return f"{rel(f)}:{k + 1}"
    return None


def spec_values(c):
    if c is None:
        return []
    return {"value": [c.get("v")], "set": c.get("vals", []), "range": [c.get("lo"), c.get("hi")],
            "min": [c.get("lo")], "max": [c.get("hi")]}.get(c["kind"], [])


def check_params(run, spec, ev, pmap):
    text = (ROOT / spec).read_text(encoding="utf-8")
    _, _, found = section_map(text)
    rows, _ = parse_param_table(found[7]["body"]) if 7 in found else ([], [])
    smap = pmap.get(spec, {})
    out = []
    for r in rows:
        m = smap.get(r["name"])
        role = m["role"] if m else "unreviewed"
        default, rng = r["default"], r["range"]
        override = (m or {}).get("spec_override")
        if override:
            c = {k: v for k, v in override.items() if k != "why"}
            if c["kind"] == "range" and default is not None and default["kind"] != "range":
                rng = c
            elif c["kind"] == "range":
                rng = c
            else:
                default = c
        items = find_evidence(ev, r["name"], m) if role != "not_same_parameter" else []
        judged = [it for it in items if "(conditional)" not in it["kind"]]
        recorded = [it for it in judged if it["source"] == "json"]
        basis = recorded if recorded else judged
        item_rows = []
        for it in items:
            if "(conditional)" in it["kind"]:
                st = "CONDITIONAL_NOT_JUDGED"
            elif role == "descriptor":
                st = "DESCRIPTOR_NOT_JUDGED"
            elif role == "threshold":
                st = threshold_status(it["value"], default)
            else:
                st = value_status(it["value"], default, rng)
            item_rows.append(dict(it, status=st))
        if role == "not_same_parameter":
            overall = "NOT_SAME_PARAMETER (suppressed)"
        elif not items:
            overall = "NO_EVIDENCE_FOUND"
        elif role == "descriptor":
            overall = "DESCRIPTOR_NOT_JUDGED"
        else:
            sts = [ir["status"] for ir in item_rows if any(ir is x or (ir["file"] == x["file"] and ir["name"] == x["name"]
                                                                        and ir["line"] == x["line"]) for x in basis)]
            sts = [s for s in sts if s in SEVERITY] or ["SPEC_NOT_MACHINE_READABLE"]
            overall = max(sts, key=SEVERITY.index)
        deviation = overall in ("OUT_OF_RANGE", "DIFFERS_NO_RANGE_GIVEN") and role in ("setting", "threshold", "unreviewed")
        reason = None
        if deviation:
            bad = [ir for ir in item_rows if ir["status"] in ("OUT_OF_RANGE", "DIFFERS_NO_RANGE_GIVEN")
                   and not isinstance(ir["value"], list)]
            if bad:
                reason = reason_recorded(PROJ / "runs" / run, bad[0]["value"], spec_values(default))
        used_vals = sorted({(json.dumps(ir["value"]) if isinstance(ir["value"], list) else "%g" % ir["value"])
                            for ir in item_rows if ir["status"] != "CONDITIONAL_NOT_JUDGED"})
        out.append(dict(
            run=run, spec=spec, param=r["name"], qualifier=r["qualifier"], role=role,
            link=("param_map" if m else ("auto_exact_name" if items else "")),
            spec_default_raw=r["default_raw"], spec_range_raw=r["range_raw"],
            spec_default=describe(default), spec_range=describe(rng),
            spec_override=(override or {}).get("why", ""),
            used_values=";".join(used_vals), n_evidence=len(items),
            evidence=";".join(f"{ir['file'].split('/runs/')[-1]}:{ir['line'] or ir['name']}={ir['value'] if isinstance(ir['value'], list) else '%g' % ir['value']}[{ir['kind']}|{ir['status']}]"
                              for ir in item_rows[:12]),
            status=overall, deviation=deviation, reason_recorded=reason or "",
            note=(m or {}).get("note", ""),
        ))
    return out


# ------------------------------------------------------------------ M2 gates
METRIC_TO_GATE = {   # metric key in a result object -> gate key in a thresholds object (documented alias)
    "trustworthiness": "trustworthiness_min",
    "trustworthiness_k5": "trustworthiness_min",
    "random_holdout": "random_holdout_correlation_min",
    "donor_holdout": "donor_holdout_correlation_min",
    "branch_holdout": "clade_branch_holdout_correlation_min",
    "category_holdout": "clade_branch_holdout_correlation_min",
    "blocked_permutation_p": "blocked_permutation_p_max",
    "permutation_p": "blocked_permutation_p_max",
    "n_permutations": "n_permutations_min",
}
PASS_KEYS = ("pass", "passes", "passed", "all_pass", "all_gates_pass")


def cmp_gate(v, thr_key, thr):
    return v <= thr if thr_key.endswith("_max") else v >= thr


def walk_dicts(o, path=""):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk_dicts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            yield from walk_dicts(v, f"{path}[{i}]")


def check_gates(run):
    run_dir = PROJ / "runs" / run
    jfiles = run_files(run_dir, ".json")
    frozen, frozen_proto = None, None
    four_gate_outside_internal = False   # frozen promotion rule: only four gates on the external panel
    for jp in jfiles:
        if re.search(r"gates?_spec", jp.name):
            try:
                obj = json.loads(jp.read_text())
                if isinstance(obj.get("gates"), dict):
                    frozen = (rel(jp), obj["gates"])
                    frozen_proto = obj.get("evaluation_protocol")
                    rules = " ".join(map(str, obj.get("promotion_rules", [])))
                    four_gate_outside_internal = bool(re.search(r"four gates on the [\w\s-]*external panel", rules, re.I))
            except (OSError, ValueError):
                pass
    rows = []
    for jp in jfiles:
        if jp.stat().st_size > 2_000_000:
            continue
        try:
            obj = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        # thresholds declared anywhere in the same file (e.g. gates_thresholds)
        file_thr = None
        for _, d in walk_dicts(obj):
            for k in ("gates_thresholds", "gates"):
                if isinstance(d.get(k), dict) and any(kk.endswith(("_min", "_max")) for kk in d[k]):
                    file_thr = (f"{rel(jp)}:{k}", d[k])
        for path, d in walk_dicts(obj):
            pk = next((k for k in PASS_KEYS if isinstance(d.get(k), bool)), None)
            if pk is None:
                proto_k = next((v for k, v in (frozen_proto or {}).items() if k.endswith("_k_neighbors")), None)
                kdev = [k for k in d if k in METRIC_TO_GATE and re.search(r"_k(\d+)$", k) and proto_k is not None
                        and int(re.search(r"_k(\d+)$", k).group(1)) != int(proto_k)]
                if kdev:
                    rows.append(dict(run=run, file=rel(jp), object=path or "(root)", pattern="protocol only",
                                     recorded_pass=None, recomputed_pass=None,
                                     status=f"PROTOCOL_K_DIFFERS({kdev} vs k={proto_k})",
                                     detail=", ".join(f"{k}={d[k]}" for k in kdev) + (f"; note: {d.get('note')}" if d.get("note") else ""),
                                     missing_gates="", boundary=""))
                continue
            # pattern A: value + threshold in the same object
            if isinstance(d.get("threshold"), (int, float)) and not isinstance(d.get("threshold"), bool):
                cands = [k for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                         and k != "threshold" and not k.startswith(("n_", "baseline"))]
                named = [k for k in cands if re.search(r"fraction|delta|ratio|retained", k)]
                if len(cands) > 1 and len(named) == 1:
                    cands = named          # name rule: prefer the one key that names a fraction/delta/ratio
                if len(cands) == 1:
                    v, thr = float(d[cands[0]]), float(d["threshold"])
                    ge, gt = v >= thr, v > thr
                    rec = d[pk]
                    status = "CONSISTENT" if (ge == rec and gt == rec) else ("BOUNDARY_DIRECTION_UNKNOWN" if ge != gt else "INCONSISTENT")
                    if "fraction" in cands[0] and not (-0.5 <= v <= 1.5):
                        status += "; IMPLAUSIBLE_FRACTION"
                    rows.append(dict(run=run, file=rel(jp), object=path or "(root)", pattern="value+threshold",
                                     recorded_pass=rec, recomputed_pass=ge, status=status,
                                     detail=f"{cands[0]}={v:.4g} vs threshold {thr:g}", missing_gates="", boundary=""))
                else:
                    rows.append(dict(run=run, file=rel(jp), object=path or "(root)", pattern="value+threshold",
                                     recorded_pass=d[pk], recomputed_pass=None, status="AMBIGUOUS_NOT_RECOMPUTED",
                                     detail=f"candidate value keys: {cands}", missing_gates="", boundary=""))
                continue
            # pattern B: metrics named like gates, thresholds from the file or the run's frozen gate file
            metrics = {k: float(v) for k, v in d.items() if k in METRIC_TO_GATE and isinstance(v, (int, float))
                       and not isinstance(v, bool)}
            if not metrics:
                continue
            thr_src, thr = (file_thr or frozen or (None, None))
            if not thr:
                rows.append(dict(run=run, file=rel(jp), object=path or "(root)", pattern="named metrics",
                                 recorded_pass=d[pk], recomputed_pass=None, status="NO_THRESHOLDS_FOUND",
                                 detail=str(metrics), missing_gates="", boundary=""))
                continue
            results, boundary, used = [], [], set()
            for mk, v in metrics.items():
                gk = METRIC_TO_GATE[mk]
                if gk in thr:
                    ok = cmp_gate(v, gk, float(thr[gk]))
                    results.append(ok)
                    used.add(gk)
                    if abs(v - float(thr[gk])) < 0.005 and not gk.startswith("n_"):
                        boundary.append(f"{mk}={v:.4f} vs {thr[gk]}")
            declared = [g for g in thr if g.endswith(("_min", "_max"))]
            # Which gates does the frozen file require for THIS panel? (added in the 2026-10-01 verification
            # pass). The manifold gate file says: all five gates on the internal panel, all four gates on the
            # external panel. An object with no panel key, or panel "internal", needs every declared gate.
            panel = d.get("panel", obj.get("panel") if isinstance(obj, dict) else None)
            perm_gates = [g for g in declared if g in ("blocked_permutation_p_max", "n_permutations_min")]
            panel_rule = "all declared gates (internal panel or no panel key)"
            not_required, rule_unstated = [], False
            if four_gate_outside_internal and isinstance(panel, str) and panel != "internal":
                if "external" in panel or "expected_to_fail" in d:
                    not_required = perm_gates
                    panel_rule = f"panel '{panel}': frozen rule requires four gates outside the internal panel"
                else:
                    rule_unstated = True
                    panel_rule = f"panel '{panel}': the frozen rules name only the internal and external panels"
            missing = [g for g in declared if g not in used and g not in not_required]
            recomputed = all(results) if results else None
            rec = d[pk]
            # protocol check: metric keys that encode a neighbour count different from the frozen protocol
            proto_k = None
            if frozen_proto:
                proto_k = next((v for k, v in frozen_proto.items() if k.endswith("_k_neighbors")), None)
            kdev = [k for k in d if re.search(r"_k(\d+)$", k) and proto_k is not None
                    and int(re.search(r"_k(\d+)$", k).group(1)) != int(proto_k)]
            if recomputed is None:
                status = "NOT_RECOMPUTED"
            elif rec != recomputed:
                status = "INCONSISTENT"
            elif rec and missing and rule_unstated:
                status = "CONSISTENT; GATE_SET_FOR_PANEL_UNSTATED"
            elif rec and missing:
                status = "PASS_WITH_UNCOMPUTED_GATES"
            else:
                status = "CONSISTENT"
            if d.get("expected_to_fail") is True and rec:
                status += "; NEGATIVE_CONTROL_PASSED"
            if kdev:
                status += f"; PROTOCOL_K_DIFFERS({kdev} vs k={proto_k})"
            rows.append(dict(run=run, file=rel(jp), object=path or "(root)", pattern="named metrics",
                             recorded_pass=rec, recomputed_pass=recomputed, status=status,
                             detail=f"thresholds from {thr_src}; " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                             + f"; required gates: {panel_rule}",
                             missing_gates=";".join(missing), boundary=";".join(boundary)))
    return rows, frozen


# ------------------------------------------------------------------ M3 phases
def check_phases(run, spec, ev):
    text = (ROOT / spec).read_text(encoding="utf-8")
    spec_phases = sorted(set(re.findall(r"^#{3,4}\s+(?:Phase|Stage)\s+([0-9]+[a-z]?)", text, re.M)),
                         key=lambda x: (int(re.match(r"\d+", x).group()), x))
    recorded = set()
    src = []
    for jp in run_files(PROJ / "runs" / run, ".json"):
        if jp.stat().st_size > 2_000_000:
            continue
        try:
            obj = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        for _, d in walk_dicts(obj):
            v = d.get("phases_run")
            if isinstance(v, list):
                recorded |= {str(x) for x in v}
                src.append(rel(jp))
    missing = [p for p in spec_phases if recorded and p not in recorded and p.rstrip("ab") not in recorded]
    return dict(run=run, spec_phases=";".join(spec_phases), n_spec_phases=len(spec_phases),
                machine_readable_phase_record=bool(recorded), recorded_phases=";".join(sorted(recorded)),
                record_files=";".join(sorted(set(src))[:3]), phases_not_run=";".join(missing))


# ------------------------------------------------------------------ M4 reproducibility
RAND_USE = re.compile(r"np\.random\.(?!seed|default_rng)\w+|default_rng\(|torch\.rand|random\.(?:shuffle|sample|choice|random|randint)|"
                      r"\.permutation\(|\.shuffle\(|\.choice\(|sample\(|random_state\s*=|ShuffleSplit|KFold\(|"
                      r"train_test_split\(|RandomForest|GradientBoosting|LogisticRegression\(|\bUMAP\(|TSNE\(|leiden\(|louvain\(")
SEED_SET = re.compile(r"np\.random\.seed\(|default_rng\(\s*[A-Za-z0-9_]|torch\.manual_seed\(|random\.seed\(|"
                      r"random_state\s*=\s*[A-Za-z0-9_]|RandomState\(\s*[A-Za-z0-9_]|\bSEED\s*=", re.I)
HF_LOAD = re.compile(r"from_pretrained\(|hf_hub_download\(|snapshot_download\(")
HF_REV = re.compile(r"revision\s*=")
PY_PATH = re.compile(r"`?([A-Za-z0-9_./\-]+\.py)`?")


def check_repro(run):
    run_dir = PROJ / "runs" / run
    scripts = run_files(run_dir, ".py")
    rand_no_seed, n_rand, n_seeded, hf_load, hf_rev = [], 0, 0, [], []
    for py in scripts:
        t = py.read_text(encoding="utf-8", errors="replace")
        uses = bool(RAND_USE.search(t))
        seeded = bool(SEED_SET.search(t))
        n_rand += uses
        n_seeded += uses and seeded
        if uses and not seeded:
            rand_no_seed.append(rel(py))
        if HF_LOAD.search(t):
            hf_load.append(rel(py))
            if HF_REV.search(t):
                hf_rev.append(rel(py))
    envs = [p for p in list(run_dir.glob("requirements*.txt")) + list(run_dir.glob("environment*.y*ml"))
            if in_deployment_snapshot(p)]
    proj_req = PROJ / "requirements.txt"
    pinned = None
    if proj_req.exists():
        lines = [l.strip() for l in proj_req.read_text().splitlines() if l.strip() and not l.startswith("#")]
        pinned = f"{sum('==' in l for l in lines)}/{len(lines)} lines use =="
    devices, walltime = set(), 0
    for jp in run_files(run_dir, ".json"):
        if jp.stat().st_size > 2_000_000:
            continue
        try:
            t = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        devices |= set(re.findall(r'"device"\s*:\s*"([^"]+)"', t))
        walltime += bool(re.search(r'"(?:elapsed|wall|total_\w*seconds|\w*_seconds|elapsed_s)\w*"\s*:', t))
    cited_ok, cited_missing = 0, []
    for doc in summary_docs(run):
        for m in PY_PATH.finditer(doc.read_text(encoding="utf-8")):
            relp = m.group(1)
            cands = [run_dir / relp, run_dir / "scripts" / Path(relp).name, PROJ / relp, ROOT / relp,
                     run_dir / "autoloop" / relp]
            if any(c.exists() for c in cands):
                cited_ok += 1
            else:
                cited_missing.append(relp)
    decision_logs = [rel(p) for p in run_dir.rglob("*") if re.search(r"(decisions?|retired|retirement)\.(jsonl?|csv)$", p.name)
                     and in_deployment_snapshot(p)]
    manual_reviews = [rel(p) for p in run_dir.rglob("MANUAL_REVIEW*.md") if in_deployment_snapshot(p)]
    manifest = [rel(p) for p in run_dir.rglob("run_manifest*.json") if in_deployment_snapshot(p)]
    return dict(run=run, n_scripts=len(scripts), scripts_using_randomness=n_rand,
                scripts_using_randomness_with_seed=n_seeded, scripts_random_without_seed=";".join(rand_no_seed),
                run_level_env_file=";".join(rel(p) for p in envs), project_requirements_pinning=pinned or "none",
                devices_recorded=";".join(sorted(devices)), json_files_with_timing=walltime,
                scripts_loading_hf_checkpoints=len(hf_load), hf_loads_with_revision_pin=len(hf_rev),
                summary_py_paths_ok=cited_ok, summary_py_paths_missing=";".join(sorted(set(cited_missing))),
                machine_readable_decision_log=";".join(decision_logs) or "none",
                manual_reclassification_files=";".join(manual_reviews), run_manifest_file=";".join(manifest) or "none")


def check_project_repro():
    """Project-level items shared by all runs: how the model checkpoint was fetched and pinned."""
    setup = [p for p in sorted((PROJ / "setup").glob("*.py")) if in_deployment_snapshot(p) and not p.name.startswith("._")]
    loads, revs = [], []
    for py in setup:
        t = py.read_text(encoding="utf-8", errors="replace")
        loads += [f"{rel(py)}:{k + 1}" for k, ln in enumerate(t.splitlines()) if HF_LOAD.search(ln)]
        revs += [f"{rel(py)}:{k + 1}" for k, ln in enumerate(t.splitlines()) if HF_REV.search(ln)]
    readmes = [p for p in [PROJ / "README.md"] + sorted((PROJ / "runs").glob("*/README*.md")) if p.exists()
               and in_deployment_snapshot(p)]
    refs, hashes = set(), []
    for rm in readmes:
        t = rm.read_text(encoding="utf-8", errors="replace")
        refs |= set(re.findall(r"huggingface\.co/[^\s\"']*/resolve/([^/\s\"']+)/", t))
        hashes += [rel(rm) for ln in t.splitlines() if re.search(r"sha256", ln, re.I)]
    for py in setup:
        if re.search(r"sha256", py.read_text(encoding="utf-8", errors="replace"), re.I):
            hashes.append(rel(py))
    req = PROJ / "requirements.txt"
    import time as _t
    return dict(setup_scripts=len(setup), checkpoint_load_lines=";".join(loads),
                project_requirements_mtime=(_t.strftime("%Y-%m-%d %H:%M", _t.localtime(req.stat().st_mtime)) if req.exists() else None),
                revision_pins=";".join(revs) or "none", download_refs_in_readmes=";".join(sorted(refs)) or "none",
                files_mentioning_sha256=";".join(sorted(set(hashes))) or "none",
                note="download ref 'main' is a moving branch; no revision= argument and no recorded weight hash means "
                     "the exact checkpoint cannot be proven from the project files")


# ------------------------------------------------------------------ M5 wording
P9_TERMS = ["encodes regulation", "causal circuit", "wiring diagram", "complete pathway", "understands",
            "the model agrees", "confirms", "confirmed", "discovers", "discovered", "mechanism", "mechanistic",
            "causes", "causally", "causal", "regulatory logic", "proves", "demonstrates"]


CI_RX = re.compile(r"\b95\s*%?\s*CI\b|\bCI\b|confidence interval|±", re.I)
CI_METHOD = re.compile(r"bootstrap|permutation|wilson|delong|percentile|subsampl|clopper|jackknife|t-interval", re.I)
CI_UNIT = re.compile(r"\b(donors?|cells?|genes?|TFs?|perturbations?|pairs?|features?|anchors?|seeds?|splits?|layers?|"
                     r"branch(?:es)?|silenced genes?|triplets?|resampl\w* (?:of|over|by) \w+)\b", re.I)
SCOPE_RX = re.compile(r"^\s*(#+|\*\*|[-*]\s+\*\*)[^\n]{0,60}\b(scope|limitation|caveat|not run|not tested|what was not)", re.I)


def check_ci_scope(run):
    rows = []
    for doc in summary_docs(run):
        lines = doc.read_text(encoding="utf-8").splitlines()
        ci_lines = [k for k, ln in enumerate(lines) if CI_RX.search(ln)]
        with_method = [k for k in ci_lines if CI_METHOD.search(" ".join(lines[max(0, k - 1):k + 2]))]
        with_unit = [k for k in with_method if CI_UNIT.search(" ".join(lines[max(0, k - 1):k + 2]))]
        scope = [k + 1 for k, ln in enumerate(lines) if SCOPE_RX.search(ln)]
        rows.append(dict(run=run, doc=rel(doc), lines_with_interval=len(ci_lines),
                         interval_lines_naming_method=len(with_method),
                         interval_lines_naming_method_and_unit=len(with_unit),
                         scope_or_limitation_heading_lines=";".join(map(str, scope[:10])) or "none"))
    return rows


def check_wording(run):
    rows = []
    for doc in summary_docs(run):
        lines = doc.read_text(encoding="utf-8").splitlines()
        for term in P9_TERMS:
            rx = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
            hits = [k + 1 for k, ln in enumerate(lines) if rx.search(ln)]
            if hits:
                rows.append(dict(run=run, doc=rel(doc), term=term, n_lines=len(hits), lines=";".join(map(str, hits[:15]))))
    return rows


def main():
    pmap = json.loads((Path(__file__).parent / "param_map.json").read_text())
    params, gates, phases, repro, wording, inputs, ciscope = [], [], [], [], [], [], []
    frozen_files = {}
    for run, spec in DEPLOYED:
        print(f"[{run}]", flush=True)
        ev = collect(PROJ / "runs" / run)
        inputs += [ROOT / s for s in ev["scripts"]] + [ROOT / spec]
        params += check_params(run, spec, ev, pmap)
        g, frozen = check_gates(run)
        gates += g
        if frozen:
            frozen_files[run] = frozen[0]
        phases.append(check_phases(run, spec, ev))
        repro.append(check_repro(run))
        wording += check_wording(run)
        ciscope += check_ci_scope(run)
        inputs += summary_docs(run)
    RESULTS.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        if not rows:
            return
        with open(RESULTS / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    dump("run_manifest_params.csv", params)
    dump("run_manifest_gates.csv", gates)
    dump("run_manifest_phases.csv", phases)
    dump("run_manifest_repro.csv", repro)
    dump("run_manifest_wording.csv", wording)
    dump("run_manifest_ci_scope.csv", ciscope)
    summ = {}
    for run, _ in DEPLOYED:
        pr = [p for p in params if p["run"] == run]
        summ[run] = dict(
            param_rows=len(pr),
            rows_with_evidence=sum(p["n_evidence"] > 0 for p in pr),
            linked_by_param_map=sum(p["link"] == "param_map" for p in pr),
            linked_by_exact_name=sum(p["link"] == "auto_exact_name" for p in pr),
            status_counts={s: sum(p["status"] == s for p in pr) for s in sorted({p["status"] for p in pr})},
            deviations=[f"{p['param']}: used {p['used_values']} vs spec {p['spec_default']} (range {p['spec_range']})"
                        + (f" [reason noted at {p['reason_recorded']}]" if p["reason_recorded"] else " [no reason found]")
                        for p in pr if p["deviation"]],
            gate_objects=sum(g["run"] == run for g in gates),
            gate_status_counts={s: sum(g["status"] == s and g["run"] == run for g in gates)
                                for s in sorted({g["status"] for g in gates if g["run"] == run})},
        )
    project = check_project_repro()
    write_json(RESULTS / "run_manifest.json", dict(summary=summ, frozen_gate_files=frozen_files, phases=phases,
                                                   repro=repro, project_repro=project))
    write_json(RESULTS / "run_manifest_run_config.json",
               run_config("run_manifest_check.py", inputs + [Path(__file__).parent / "param_map.json"],
                          extra={"helper_sha256": {f: __import__("common").sha256_file(Path(__file__).parent / f)
                                                   for f in ("run_evidence.py", "spec_parse.py", "param_map.json")}}))
    strict = "--strict" in sys.argv
    for run, s in summ.items():
        print(f"{run:28s} rows={s['param_rows']:3d} with_evidence={s['rows_with_evidence']:2d} "
              f"(map {s['linked_by_param_map']}, auto {s['linked_by_exact_name']}) deviations={len(s['deviations'])} "
              f"gates={s['gate_status_counts']}")
        for d in s["deviations"]:
            print("     -", d)
    if strict:
        # blocking rule when used before a summary is accepted: a deviation with no written reason, or a
        # gate object whose recorded pass flag is inconsistent, rests on an uncomputed gate, or is a passed negative control
        bad_p = [p for p in params if p["deviation"] and not p["reason_recorded"]]
        bad_g = [g for g in gates if any(t in g["status"] for t in ("INCONSISTENT", "PASS_WITH_UNCOMPUTED_GATES",
                                                                    "NEGATIVE_CONTROL_PASSED", "IMPLAUSIBLE"))]
        print(f"--strict: {len(bad_p)} unexplained parameter deviation(s), {len(bad_g)} gate problem(s)")
        return 1 if (bad_p or bad_g) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
