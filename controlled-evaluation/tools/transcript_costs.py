#!/usr/bin/env python3
"""Per-agent cost and file-access summary for one workflow run.

Usage: transcript_costs.py <workflow_transcript_dir> <out_csv> [allowed_prefix ...]

For each agent started by the workflow (labels from journal.jsonl) this sums
token usage over the agent's assistant messages, counts tool calls, measures
wall time from the first to the last transcript timestamp, and flags any file
path read or named in a shell command that falls outside the allowed prefixes.
"""
import json, sys, csv, os, re, glob
from datetime import datetime

d, out = sys.argv[1], sys.argv[2]
allowed = sys.argv[3:]
labels = {}
for line in open(os.path.join(d, 'journal.jsonl')):
    e = json.loads(line)
    if e.get('type') == 'started':
        labels[e['agentId']] = e.get('label', '')

path_re = re.compile(r"(/(?:Volumes|Users|private|tmp)[^\s'\"<>|;&)]*)")
rows = []
for f in glob.glob(os.path.join(d, 'agent-*.jsonl')):
    aid = os.path.basename(f)[len('agent-'):-len('.jsonl')]
    tin = tout = tcr = tcc = 0
    ncalls = 0
    seen_msg = set()
    ts = []
    outside = set()
    for line in open(f):
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get('timestamp')
        if t:
            ts.append(datetime.fromisoformat(t.replace('Z', '+00:00')))
        m = e.get('message') or {}
        if e.get('type') == 'assistant' and isinstance(m, dict):
            mid = m.get('id')
            u = m.get('usage') or {}
            if mid not in seen_msg:  # one usage record per API message
                seen_msg.add(mid)
                tin += u.get('input_tokens', 0) or 0
                tout += u.get('output_tokens', 0) or 0
                tcr += u.get('cache_read_input_tokens', 0) or 0
                tcc += u.get('cache_creation_input_tokens', 0) or 0
            for c in m.get('content') or []:
                if isinstance(c, dict) and c.get('type') == 'tool_use':
                    ncalls += 1
                    inp = json.dumps(c.get('input', {}))
                    for p in path_re.findall(inp):
                        p = p.rstrip('\\').rstrip('.,')
                        if allowed and not any(p.startswith(a) for a in allowed):
                            outside.add(p[:200])
    wall = (max(ts) - min(ts)).total_seconds() if ts else 0
    rows.append(dict(agent_id=aid, label=labels.get(aid, ''), input_tokens=tin, output_tokens=tout,
                     cache_read_tokens=tcr, cache_creation_tokens=tcc, tool_calls=ncalls,
                     wall_seconds=round(wall, 1), n_outside_paths=len(outside),
                     outside_paths=' | '.join(sorted(outside))[:2000]))
rows.sort(key=lambda r: r['label'])
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'{len(rows)} agents -> {out}')
