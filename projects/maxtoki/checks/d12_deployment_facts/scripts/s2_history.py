"""D12 step 2: human prompts during the MaxToki deployment and audit.

Source: ~/.claude/history.jsonl (read only). It stores each typed prompt with a
time stamp, the project folder and the session id. It does NOT store the agent
model, the agent's replies, or how long the person spent reading or thinking.

Privacy: no prompt text is written to the output. The output holds counts,
times, session ids, a hand-assigned category per prompt, and a short neutral
gist only for the prompts in the audit session (2a9284b9).
Outputs: out/human_prompts.json
"""
import collections
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import dump, sha256_file

HIST = os.path.expanduser("~/.claude/history.jsonl")
PROJECT = "<REPO_ROOT>"

# Session roles, assigned by reading the prompts (see DEPLOYMENT_FACTS.md).
SESSIONS = {
    "17b19da0": "spec authoring (pipelines/*.md), last prompt starts the MaxToki project",
    "965ec1c8": "MaxToki runs: setup, attention-grn, spectral, sae-atlas (+web atlas), circuit-tracing",
    "8fb71b84": "MaxToki runs: exhaustive-mapping, atlas deployment, spectral extension + autoloop",
    "3af9282c": "MaxToki runs: topology-141",
    "8be46b0f": "MaxToki runs: manifold-discovery",
    "b9b1ea22": "MaxToki runs: longevity",
    "2a9284b9": "topology-141 completion + extensions, audit, repairs, start of paper",
    "58d03bad": "MaxToki runs: longevity completion",
    "5137b56d": "MaxToki runs: manifold-discovery continuation",
    "862b8d39": "paper drafting and public repo",
}
NOT_MAXTOKI = {
    "6fa7aa59": "AIDO.Cell project (different model)",
    "9c9a1e82": "journal correspondence for an unrelated paper",
    "91ff0dbe": "download of a different model",
}

# category, pipeline for every prompt in the MaxToki sessions (key = ms timestamp)
C = {
    1776270666674: ("spec_authoring", None), 1776270744247: ("spec_authoring", None),
    1776270881436: ("spec_authoring", None), 1776271585463: ("spec_authoring", None),
    1776272811011: ("spec_authoring", None), 1776274855311: ("spec_authoring", None),
    1776275543988: ("spec_authoring", None), 1776276558714: ("spec_authoring", None),
    1776277613175: ("spec_authoring", None), 1776278597870: ("spec_authoring", None),
    1776279627139: ("spec_authoring", None), 1776282064256: ("spec_authoring", None),
    1776282825544: ("spec_authoring", None),
    1776287882931: ("setup", "setup"),
    1776288274672: ("session_admin", None), 1776288302978: ("session_admin", None),
    1776288351453: ("continue", "setup"),
    1776288682555: ("spec_authoring", "longevity"),
    1776289395288: ("setup", "setup"), 1776289514862: ("setup", "setup"),
    1776289587944: ("run_request", "attention-grn"), 1776290243624: ("continue", "attention-grn"),
    1776329553048: ("continue", "attention-grn"), 1776340558111: ("continue", "attention-grn"),
    1776358628276: ("run_request", "spectral-geometry"),
    1776413742963: ("docs", None),
    1776413930229: ("scope_check", "spectral-geometry"), 1776414033226: ("continue", "spectral-geometry"),
    1776415195813: ("scope_check", "spectral-geometry"), 1776415270455: ("stop_decision", "spectral-geometry"),
    1776433714618: ("run_request", "sae-atlas"),
    1776454473106: ("scope_check", "sae-atlas"), 1776454527203: ("continue", "sae-atlas"),
    1776516028098: ("continue", "sae-atlas"), 1776597027796: ("continue", "sae-atlas"),
    1776597723613: ("atlas_webapp", "sae-atlas"), 1776599159916: ("atlas_webapp", "sae-atlas"),
    1776599839550: ("atlas_webapp", "sae-atlas"), 1776601525437: ("atlas_webapp", "sae-atlas"),
    1776604876346: ("atlas_webapp", "sae-atlas"), 1776605252139: ("continue", "sae-atlas"),
    1776624504386: ("atlas_webapp", "sae-atlas"), 1776624789986: ("atlas_webapp", "sae-atlas"),
    1776625561738: ("continue", "sae-atlas"),
    1776630041234: ("run_request", "circuit-tracing"),
    1776675781937: ("scope_check", "circuit-tracing"), 1776675827006: ("continue", "circuit-tracing"),
    1776687250004: ("scope_check", "circuit-tracing"), 1776687315780: ("run_request", "exhaustive-mapping"),
    1776965220732: ("session_admin", None), 1776965248906: ("session_admin", None),
    1776965263604: ("session_admin", None), 1776965302121: ("session_admin", None),
    1776965327685: ("session_admin", None),
    1776965383099: ("scope_check", "exhaustive-mapping"), 1776965500750: ("continue", "exhaustive-mapping"),
    1776968722216: ("scope_check", "exhaustive-mapping"), 1776968766534: ("continue", "exhaustive-mapping"),
    1776969905937: ("status_check", "exhaustive-mapping"),
    1777031361751: ("docs", "exhaustive-mapping"),
    1777827312567: ("atlas_webapp", "sae-atlas"), 1777827897015: ("atlas_webapp", "sae-atlas"),
    1777828313913: ("atlas_webapp", "sae-atlas"),
    1777828381357: ("run_request", "spectral-geometry"),
    1777828519663: ("run_request", "topology-141"),
    1777829894980: ("run_request", "manifold-discovery"), 1777830660778: ("scope_decision", "manifold-discovery"),
    1777990054165: ("continue", "manifold-discovery"), 1777990093270: ("continue", "topology-141"),
    1777990103371: ("scope_check", "spectral-geometry"), 1777990165363: ("continue", "spectral-geometry"),
    1777990214292: ("run_request", "longevity"),
    1778146418205: ("session_admin", None), 1778146542353: ("session_admin", None),
    1778146620503: ("session_admin", "longevity"), 1778146660499: ("session_admin", "manifold-discovery"),
    1778146683041: ("continue", "longevity"),
    1778146716095: ("scope_check", "topology-141"), 1778146757433: ("continue", "manifold-discovery"),
    1778146832163: ("continue", "topology-141"), 1778162095492: ("continue", "manifold-discovery"),
    1778162124488: ("continue", "topology-141"),
    1778165386338: ("audit", "audit"), 1778167817508: ("audit", "audit"),
    1778169000791: ("audit", "audit"), 1778172451620: ("audit", "audit"),
    1778172539003: ("audit", "audit"),
    1778189468887: ("paper", "paper"),
    1778190151408: ("paper", "paper"), 1778190416291: ("paper", "paper"),
    1778190678761: ("paper", "paper"), 1778192025971: ("paper", "paper"),
    1778192919787: ("paper", "paper"), 1778193984298: ("paper", "paper"),
    1778194265296: ("paper", "paper"), 1778195908682: ("paper", "paper"),
    1778196645658: ("paper", "paper"),
}

# short neutral gists, audit session only
GIST_2A9284B9 = {
    1778146418205: "asks the agent to find the parallel sessions of the last days",
    1778146542353: "picks one of the options the agent offered",
    1778146716095: "asks to check whether the topology-141 run is complete",
    1778146832163: "approves the agent's proposed completion work",
    1778162124488: "asks to run the items the topology run had left out of scope",
    1778165386338: "asks for a new audit pipeline built from earlier peer-review comments",
    1778167817508: "asks to apply that audit to all MaxToki results (summaries folder)",
    1778169000791: "asks to fix all issues the audit found",
    1778172451620: "asks whether all audit issues are now addressed",
    1778172539003: "asks to address the remaining issues",
    1778189468887: "pasted text (content not stored); starts the paper work",
}


def ts(ms):
    return datetime.datetime.fromtimestamp(ms / 1000)


rows = [json.loads(l) for l in open(HIST)]
lo, hi = datetime.datetime(2026, 4, 15), datetime.datetime(2026, 5, 9)
win = [r for r in rows if lo <= ts(r["timestamp"]) < hi and r["project"] == PROJECT]
mx = [r for r in win if r["sessionId"][:8] in SESSIONS]
other = [r for r in win if r["sessionId"][:8] not in SESSIONS]

missing = [r["timestamp"] for r in mx if r["timestamp"] not in C]
assert not missing, f"unclassified prompts: {missing}"
assert len(C) == len(mx), (len(C), len(mx))

recs = []
for r in mx:
    cat, pipe = C[r["timestamp"]]
    recs.append(dict(time=ts(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"), session=r["sessionId"][:8],
                     category=cat, pipeline=pipe,
                     n_pasted_blocks=len(r.get("pastedContents") or {})))

deploy_start, deploy_end = "2026-04-15 23:18:00", "2026-05-07 21:10:46"   # MaxToki project start -> audit v2
dep = [x for x in recs if x["category"] not in ("spec_authoring", "paper")]
per_day = collections.OrderedDict()
for d in sorted({x["time"][:10] for x in recs}):
    xs = [x for x in recs if x["time"][:10] == d]
    ds = [x for x in dep if x["time"][:10] == d]
    per_day[d] = dict(
        all_maxtoki_session_prompts=len(xs),
        deployment_prompts=len(ds),
        by_category=dict(collections.Counter(x["category"] for x in xs)),
        sessions=sorted({x["session"] for x in xs}),
        first=min(x["time"] for x in xs)[11:16], last=max(x["time"] for x in xs)[11:16],
    )

audit_sess = [dict(time=x["time"], category=x["category"], gist=GIST_2A9284B9[r["timestamp"]])
              for x, r in zip(recs, mx) if x["session"] == "2a9284b9"]
audit_only = [a for a in audit_sess if a["category"] == "audit"]

res = dict(
    source=dict(path=HIST, sha256=sha256_file(HIST), n_rows_total=len(rows),
                first=ts(rows[0]["timestamp"]).isoformat(), last=ts(rows[-1]["timestamp"]).isoformat(),
                note="stores typed prompts only; no agent model, no replies, no reading/thinking time"),
    window=dict(start=lo.isoformat(), end=hi.isoformat(), project=PROJECT),
    sessions=SESSIONS, excluded_sessions_same_folder=NOT_MAXTOKI,
    n_prompts_same_folder_in_window=len(win),
    n_prompts_excluded_other_projects_same_folder=len(other),
    n_prompts_maxtoki_sessions=len(mx),
    by_category=dict(collections.Counter(x["category"] for x in recs)),
    n_deployment_prompts_excl_spec_and_paper=len(dep),
    deployment_window=dict(start=deploy_start, end=deploy_end),
    per_day=per_day,
    by_pipeline=dict(collections.Counter(x["pipeline"] for x in dep if x["pipeline"])),
    audit_session_2a9284b9=audit_sess,
    audit_prompts=dict(n=len(audit_only), first=audit_only[0]["time"], last=audit_only[-1]["time"]),
    prompts=recs,
    supervision_time="NOT LOGGED. history.jsonl records when a prompt was typed, not how long the person read, checked or thought. No stopwatch or time log exists.",
)
print(dump("human_prompts.json", res))
print(json.dumps({k: res[k] for k in ("n_prompts_maxtoki_sessions", "by_category", "n_deployment_prompts_excl_spec_and_paper", "by_pipeline", "audit_prompts")}, indent=1))
for d, v in per_day.items():
    print(d, v["all_maxtoki_session_prompts"], v["deployment_prompts"], v["first"], v["last"], v["sessions"])
