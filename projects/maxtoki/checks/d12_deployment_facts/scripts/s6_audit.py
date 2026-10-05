"""D12 step 6: audit timing on 2026-05-07.

Combines human prompt times (out/human_prompts.json, from history.jsonl) with
file write times of the audit spec, audit reports and repair scripts/outputs.
Durations are prompt -> last file of that step (wall clock, local time).
Outputs: out/audit_timing.json
"""
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, REPO, OUT, iso, dump, eff_time

hp = json.load(open(os.path.join(OUT, "human_prompts.json")))
P = {a["time"]: a for a in hp["audit_session_2a9284b9"]}
ts = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
prompt = {
    "spec_request": ts("2026-05-07 16:49:46"),
    "apply_request": ts("2026-05-07 17:30:17"),
    "fix_request": ts("2026-05-07 17:50:00"),
    "done_question": ts("2026-05-07 18:47:31"),
    "fix_remaining_request": ts("2026-05-07 18:48:59"),
}
for k, v in prompt.items():
    assert iso(v) in P and P[iso(v)]["category"] == "audit", k


def mt(p):
    return eff_time(os.stat(p))[0]


f = {
    "audit_spec": REPO + "/pipelines/audit-recurring-review-issues.md",
    "audit_report": MT + "/audits/audit-20260507.md",
    "completion_v1": MT + "/audits/audit-20260507-completion.md",
    "completion_v2": MT + "/audits/audit-20260507-completion-v2.md",
}
t = {k: mt(v) for k, v in f.items()}

# repair artefacts: audit_* scripts and the outputs they wrote
repair = []
for s in glob.glob(MT + "/runs/*/scripts/audit_*.py") + [MT + "/audits/audit_a3_bootstrap_cis.py"]:
    if "/._" in s:
        continue
    repair.append((mt(s), os.path.relpath(s, MT)))
outs = ["runs/sae-atlas-217M/outputs/phase8t_chance_baseline", "runs/circuit-tracing-217M/outputs/edge_density_chance_baseline",
        "audits/bootstrap_cis_summary.json", "runs/topology-141-217M/outputs/synthetic_positive_control",
        "runs/sae-atlas-217M/outputs/phase8t_positive_tf_case_study", "runs/topology-141-217M/outputs/groupkfold_h123",
        "audits/audit_a2_groupkfold_status.md", "runs/spectral-geometry-217M/outputs/phase9b/bootstrap_pearson.json",
        "runs/manifold-discovery-217M/reports/external_validation_external_bootstrap.json",
        "runs/topology-141-217M/outputs/phase11_autoloop/iter_04_replication", "runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun",
        "runs/manifold-discovery-217M/outputs/cell_level_benchmark_lite", "runs/topology-141-217M/outputs/seed_stability",
        "runs/circuit-tracing-217M/outputs/groupkfold_crispri", "runs/topology-141-217M/outputs/seed_stability_phase0"]
for o in outs:
    p = os.path.join(MT, o)
    files = [p] if os.path.isfile(p) else [x for x in glob.glob(p + "/**/*", recursive=True) if os.path.isfile(x) and "/._" not in x]
    last = max(mt(x) for x in files)
    repair.append((last, o + ("/" if os.path.isdir(p) else "")))
repair.sort()
r1 = [(iso(a), b) for a, b in repair if prompt["fix_request"] <= a < prompt["done_question"]]
r2 = [(iso(a), b) for a, b in repair if prompt["fix_remaining_request"] <= a <= t["completion_v2"] + 60]
other = [(iso(a), b) for a, b in repair if not (prompt["fix_request"] <= a <= t["completion_v2"] + 60)]

mins = lambda a, b: round((b - a) / 60, 1)
res = dict(
    prompts={k: iso(v) for k, v in prompt.items()},
    files={k: iso(v) for k, v in t.items()},
    steps=dict(
        spec_writing=dict(start=iso(prompt["spec_request"]), end=iso(t["audit_spec"]), minutes=mins(prompt["spec_request"], t["audit_spec"])),
        audit_sweep=dict(start=iso(prompt["apply_request"]), end=iso(t["audit_report"]), minutes=mins(prompt["apply_request"], t["audit_report"])),
        repair_round_1=dict(start=iso(prompt["fix_request"]), end=iso(t["completion_v1"]), minutes=mins(prompt["fix_request"], t["completion_v1"]), artefacts=r1),
        repair_round_2=dict(start=iso(prompt["fix_remaining_request"]), end=iso(t["completion_v2"]), minutes=mins(prompt["fix_remaining_request"], t["completion_v2"]), artefacts=r2),
    ),
    total_from_spec_request_h=round((t["completion_v2"] - prompt["spec_request"]) / 3600, 2),
    total_from_apply_request_h=round((t["completion_v2"] - prompt["apply_request"]) / 3600, 2),
    repair_artefacts_outside_rounds=other,
    model_forward_passes_during_repair=[
        "runs/topology-141-217M/outputs/seed_stability_phase0/seed43/lung/run_config.json (device mps, wall_seconds 893.5)",
        "runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun/run.log (device=mps)",
        "runs/manifold-discovery-217M/scripts/audit_residual6_cell_level_benchmark_lite.py (MaxToki forward on MPS)",
    ],
    session="2a9284b9 (the same session also ran the topology-141 extensions earlier that day)",
    n_human_prompts_in_session_that_day=sum(1 for a in hp["audit_session_2a9284b9"] if a["time"].startswith("2026-05-07")),
    n_audit_prompts=hp["audit_prompts"]["n"],
)
print(dump("audit_timing.json", res))
print(json.dumps({k: res[k] for k in ("steps", "total_from_spec_request_h", "total_from_apply_request_h")}, indent=1)[:3000])
