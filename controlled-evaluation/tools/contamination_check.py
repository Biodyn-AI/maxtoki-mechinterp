#!/usr/bin/env python3
"""Scan subject-agent tool calls for access to forbidden locations.

Usage: contamination_check.py <workflow_transcript_dir> <label_prefix> <out_json> <forbidden_substring> [...]

Only tool-call INPUTS (commands, file paths) are scanned, not the agent's own
final text. Reading the agent's own saved tool results under
~/.claude/projects/.../tool-results/ is allowed and ignored.
"""
import json, glob, os, sys
d, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
forbidden = sys.argv[4:]
labels = {}
for l in open(os.path.join(d, 'journal.jsonl')):
    e = json.loads(l)
    if e.get('type') == 'started':
        labels[e['agentId']] = e.get('label', '')
report = {}
for f in glob.glob(os.path.join(d, 'agent-*.jsonl')):
    aid = os.path.basename(f)[6:-6]
    lab = labels.get(aid, '')
    if not lab.startswith(prefix):
        continue
    hits = []
    for l in open(f):
        e = json.loads(l); m = e.get('message') or {}
        if e.get('type') != 'assistant' or not isinstance(m, dict):
            continue
        for c in m.get('content') or []:
            if isinstance(c, dict) and c.get('type') == 'tool_use' and c.get('name') != 'StructuredOutput':
                s = json.dumps(c.get('input', {}))
                s_clean = s.replace('/tool-results/', '')
                for pat in forbidden:
                    if pat in s and '/tool-results/' not in s:
                        hits.append({'pattern': pat, 'call': s[:400]})
    report[lab] = hits
json.dump(report, open(out, 'w'), indent=1)
bad = {k: len(v) for k, v in report.items() if v}
print(f'{len(report)} agents scanned; with hits: {bad}')
