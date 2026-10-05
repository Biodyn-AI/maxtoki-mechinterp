#!/usr/bin/env python3
"""Did any subject agent read a scratch file that another agent wrote?

Subject agents could write temporary files into the session scratch folder
(the harness tells every agent its path). For each subject agent, collect the
scratch paths that appear in its tool-call inputs, split into paths it wrote
(redirect, cat >, open(..,'w'), np.save, pickle.dump, to_csv) and all paths it
mentions. A cross-read is a path an agent mentions that it never wrote itself
but some other agent (any study, any role) wrote.
Usage: scratch_crossread_check.py <out_json>
"""
import json, re, os, glob, sys, collections
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/')
SCR = 'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad'
PATH = re.compile(re.escape(SCR) + r'(/[^\s"\'\\)]*)?')
WRITE = re.compile(r'(?:>>?|cat\s*>|tee\s+(?:-a\s+)?|open\(\s*|np\.save\w*\(\s*|to_csv\(\s*|savefig\(\s*|mkdir\s+-p\s+)[\"\']?[^\s\"\']*' + re.escape(SCR) + r'(/[^\s"\'\\)]*)?')
SUBJ = re.compile(r'^(A|B|C|AH):.*?(:exec|:review|:review-generic|:review-checklist|:repair-generic|:repair-checklist)?$')

def is_subject(lab):
    p = lab.split(':')
    if lab.startswith('C:'): return len(p) == 3
    return bool(re.search(r':(exec|review|review-generic|review-checklist|repair-generic|repair-checklist)$', lab))

writes = collections.defaultdict(set); mentions = collections.defaultdict(set); role = {}
for d in sorted(glob.glob(W + 'wf_*')):
    labels = {}
    jp = os.path.join(d, 'journal.jsonl')
    if not os.path.exists(jp): continue
    for l in open(jp):
        e = json.loads(l)
        if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
    for f in glob.glob(os.path.join(d, 'agent-*.jsonl')):
        aid = os.path.basename(f)[6:-6]; lab = labels.get(aid, '?')
        key = os.path.basename(d) + '|' + lab + '|' + aid
        role[key] = is_subject(lab)
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            m = e.get('message') or {}
            for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
                if isinstance(c, dict) and c.get('type') == 'tool_use':
                    s = json.dumps(c.get('input', {})).replace('\\"', '"')
                    for mm in PATH.finditer(s):
                        p = (mm.group(1) or '/').rstrip('/;,') or '/'
                        if '/tool-results' in s[mm.start()-0:mm.end()]: continue
                        mentions[key].add(p)
                    for mm in WRITE.finditer(s):
                        writes[key].add((mm.group(1) or '/').rstrip('/;,') or '/')
writer_of = collections.defaultdict(set)
for k, ps in writes.items():
    for p in ps: writer_of[p].add(k)
cross = []
for k, ps in mentions.items():
    if not role.get(k): continue
    for p in ps:
        if p in ('/', ''): continue
        others = {w for w in writer_of.get(p, set()) if w != k}
        if others and p not in writes[k]:
            cross.append({'agent': k, 'path': p, 'written_by': sorted(others)})
subj = [k for k in role if role[k]]
n_w = sum(1 for k in subj if writes[k])
n_m = sum(1 for k in subj if mentions[k])
print(f'subject agents: {len(subj)}; that wrote scratch files: {n_w}; that mention any scratch path: {n_m}')
print(f'cross-reads (subject agent mentions a scratch path another agent wrote, not itself): {len(cross)}')
for x in cross[:40]: print(' ', x['agent'], x['path'], '<-', x['written_by'][:3])
json.dump({'cross': cross, 'writes': {k: sorted(v) for k, v in writes.items() if role.get(k)}}, open(sys.argv[1], 'w'), indent=1)
