#!/usr/bin/env python3
"""Cross-agent file sharing check for all subject agents (Studies A, B, C and the exploratory Haiku arm).

Collect every file path any agent wrote (shell redirects, cat >, tee, open(...,'w'),
np.save*, pickle.dump(open(...)), to_csv, savefig, Write/Edit tools), resolving
relative names against a 'cd <dir>' in the same command. Then, for every subject
agent, list paths it mentions after another agent wrote them and that it did not
write itself. /dev/*, the agent's own tool-results and the shared python wrapper
are ignored.
Usage: crossread_all.py <out_json>
"""
import json, os, glob, re, sys, collections
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/')
WFS = {'wf_28496082-f75': 'A', 'wf_59bae9e3-04a': 'A', 'wf_7e9ab73d-e56': 'B', 'wf_6ef77bf7-9cd': 'C', 'wf_655a226a-c12': 'AH'}
FN = r"[\w\-./ ]*?[\w\-]+\.(?:py|txt|json|csv|npy|npz|md|tsv|pkl|pickle|log|out|sh|png|parquet)"
WRITE_PATS = [re.compile(p) for p in [
    r"(?:cat\s*>|>>|(?<![<>=!\-])>(?![=>]))\s*[\"']?(" + FN + r")[\"']?",
    r"tee\s+(?:-a\s+)?[\"']?(" + FN + r")",
    r"open\(\s*[\"'](" + FN + r")[\"']\s*,\s*[\"'][wax]",
    r"(?:np\.save\w*|savefig|to_csv|to_parquet|write_text)\(\s*[\"'](" + FN + r")[\"']"]]
CD = re.compile(r"cd\s+[\"']?([^\"'&;\n]+?)[\"']?\s*(?:&&|;|\n)")

def subject(lab, st):
    if st == 'C': return len(lab.split(':')) == 3
    return bool(re.search(r':(exec|review|review-generic|review-checklist|repair-generic|repair-checklist)$', lab))

def norm(p, cwd):
    p = p.strip().replace('\\ ', ' ')
    if not p.startswith('/'):
        if not cwd: return None
        p = os.path.normpath(os.path.join(cwd.replace('\\ ', ' '), p))
    return p

agents = {}
for wf, st in WFS.items():
    labels = {}
    for l in open(W + wf + '/journal.jsonl'):
        e = json.loads(l)
        if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
    for f in glob.glob(W + wf + '/agent-*.jsonl'):
        aid = os.path.basename(f)[6:-6]; lab = labels.get(aid, '?')
        A = agents[wf + '|' + lab + '|' + aid] = {'study': st, 'subject': subject(lab, st), 'writes': {}, 'texts': []}
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            ts = e.get('timestamp', '')
            m = e.get('message') or {}
            for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
                if not (isinstance(c, dict) and c.get('type') == 'tool_use'): continue
                inp = c.get('input', {})
                if c.get('name') in ('Write', 'Edit', 'NotebookEdit') and inp.get('file_path'):
                    A['writes'].setdefault(inp['file_path'], ts)
                cmd = inp.get('command', '') if isinstance(inp, dict) else ''
                s = cmd if cmd else json.dumps(inp)
                A['texts'].append((ts, s))
                if cmd:
                    cds = CD.findall(cmd); cwd = cds[0].strip() if cds else None
                    for p in WRITE_PATS:
                        for mm in p.finditer(cmd):
                            q = norm(mm.group(1), cwd)
                            if q and not q.startswith('/dev/') and '/tool-results/' not in q:
                                A['writes'].setdefault(q, ts)
written = collections.defaultdict(list)
for k, A in agents.items():
    for p, ts in A['writes'].items(): written[p].append((ts, k))
cross = []
for k, A in agents.items():
    if not A['subject']: continue
    for p, ws in written.items():
        others = [(ts, w) for ts, w in ws if w != k]
        if not others or p in A['writes']: continue
        base = os.path.basename(p); parent = os.path.dirname(p)
        for ts, s in A['texts']:
            sn = s.replace('\\ ', ' ')
            hit = p in sn or (base in sn and os.path.basename(parent) in sn)
            if hit and any(t0 < ts for t0, _ in others):
                cross.append({'reader': k, 'path': p, 'read_at': ts, 'writers': [w for t0, w in others if t0 < ts][:3]})
                break
by = collections.Counter((agents[x['reader']]['study'], 'cross') for x in cross)
print('subject agents per study:', collections.Counter(A['study'] for A in agents.values() if A['subject']))
print('subject agents that read a file another agent had written:', collections.Counter(agents[x['reader']]['study'] for x in {x['reader']: x for x in cross}.values()))
for x in cross:
    if agents[x['reader']]['study'] != 'AH':
        print(' ', x['reader'], x['path'], x['read_at'][:19], '<-', [w.split('|')[1] for w in x['writers']])
json.dump(cross, open(sys.argv[1], 'w'), indent=1)
