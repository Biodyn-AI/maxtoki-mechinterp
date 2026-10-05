#!/usr/bin/env python3
"""For the Haiku re-run (wf_6fde3184-e61): did any agent read or run a <TMP> file whose most
recent write (before that moment) was made by a DIFFERENT agent?  Also: whose folder did the
five reviewers open, when it was not their own?
Events come from tool-call inputs with their timestamps. A tool call that both writes and then
uses the same path counts as the agent's own write (the read follows within the same command).
Writes into <TMP> done by a script running inside python (open('<TMP>/..','w')) are counted too.
Writes studyA_haiku2/results/tmp_race_check.json.
"""
import json, os, glob, re, collections
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/wf_6fde3184-e61')
EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMAP = json.load(open(os.path.join(EV, 'studyA_haiku2', 'folder_map.json')))
OWNER = {os.path.basename(v): k for k, v in FMAP.items()}
P = r'(?:/private)?<TMP>/(?!claude-501)[\w.\-/]+'
WR = re.compile(r'(?:cat\s*>|>>?|tee\s+(?:-a\s+)?|open\(\s*[\'"])\s*[\'"]?(' + P + ')')
ANY = re.compile(P)
labels = {}
for l in open(os.path.join(W, 'journal.jsonl')):
    e = json.loads(l)
    if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
events = []; other = []
for f in glob.glob(os.path.join(W, 'agent-*.jsonl')):
    lab = labels.get(os.path.basename(f)[6:-6], '?')
    for l in open(f):
        try: e = json.loads(l)
        except Exception: continue
        ts = e.get('timestamp', ''); m = e.get('message') or {}
        for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
            if not (isinstance(c, dict) and c.get('type') == 'tool_use'): continue
            s = json.dumps(c.get('input', {})).replace('\\"', '"').replace('\\n', '\n')
            w = {x.replace('/private', '').rstrip('.') for x in WR.findall(s)}
            a = {x.replace('/private', '').rstrip('.') for x in ANY.findall(s)}
            for p in w: events.append((ts, lab, p, 'write'))
            for p in a - w: events.append((ts, lab, p, 'read'))
            for u in set(re.findall(r'analysis-tasks/(u\d{6})', s)):
                if not FMAP.get(lab, '').endswith(u):
                    other.append((lab, u, OWNER.get(u, '?')))
events.sort()
last = {}; cross = []
for ts, lab, p, kind in events:
    if kind == 'write':
        last[p] = (lab, ts)
    elif p in last and last[p][0] != lab:
        cross.append(dict(time=ts, reader=lab, path=p, last_writer=last[p][0], written=last[p][1]))
print('tmp events', len(events), '| reads whose last writer was another agent:', len(cross))
for x in cross[:30]: print('  ', x['time'][:19], x['reader'], x['path'], '<-', x['last_writer'], x['written'][:19])
print('other folders opened:')
for x in sorted(set(other)): print('  ', x)
json.dump(dict(cross=cross, other_folders=sorted(set(other))), open(os.path.join(EV, 'studyA_haiku2', 'results', 'tmp_race_check.json'), 'w'), indent=1)
