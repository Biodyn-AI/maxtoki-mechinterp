#!/usr/bin/env python3
"""Two remaining shared locations: plain <TMP>, and the harness folder of background-task outputs.

(1) Any subject agent whose tool calls touch a <TMP> path outside the harness folders.
(2) Any subject agent that reads tasks/<id>.output for a background task it did not start itself.
"""
import json, os, glob, re, collections
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/')
WFS = {'wf_28496082-f75': 'A', 'wf_59bae9e3-04a': 'A', 'wf_7e9ab73d-e56': 'B', 'wf_6ef77bf7-9cd': 'C', 'wf_655a226a-c12': 'AH'}
TMP = re.compile(r'(?<![\w/])(?:/private)?<TMP>/(?!claude-501)[^\s"\'\;|&)]+')
TASK = re.compile(r'tasks/(\w{9})\.output')
OWN = re.compile(r'(?:with ID|ID): (\w{9})')
def subject(lab, st):
    if st == 'C': return len(lab.split(':')) == 3
    return bool(re.search(r':(exec|review|review-generic|review-checklist|repair-generic|repair-checklist)$', lab))
res = collections.defaultdict(list)
for wf, st in WFS.items():
    labels = {}
    for l in open(W + wf + '/journal.jsonl'):
        e = json.loads(l)
        if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
    for f in glob.glob(W + wf + '/agent-*.jsonl'):
        lab = labels.get(os.path.basename(f)[6:-6], '?')
        if not subject(lab, st): continue
        own = set(); calls = []
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            m = e.get('message') or {}
            for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
                if not isinstance(c, dict): continue
                if c.get('type') == 'tool_use': calls.append(json.dumps(c.get('input', {})))
                if c.get('type') == 'tool_result':
                    out = c.get('content'); out = out if isinstance(out, str) else json.dumps(out)
                    own.update(OWN.findall(out))
        tmp = sorted({p for s in calls for p in TMP.findall(s)})
        foreign = sorted({t for s in calls for t in TASK.findall(s)} - own)
        if tmp: res[(st, 'tmp')].append((lab, tmp[:6]))
        if foreign: res[(st, 'foreign_task_output')].append((lab, foreign[:6]))
for k in sorted(res):
    print(k, len(res[k]))
    for x in res[k][:15]: print('   ', x)
