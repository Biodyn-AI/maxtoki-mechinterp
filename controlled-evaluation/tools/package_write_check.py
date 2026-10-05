#!/usr/bin/env python3
"""Which agents wrote files into the Study A task folders, and which agents later touched them.

For each Study A workflow (Opus batches and the exploratory Haiku arm), scan
every agent's tool calls for paths into a task folder that are not part of the
package source, and report write/read events with timestamps.
"""
import json, os, glob, re, collections, sys
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/')
AT = '<CODE_ROOT>/analysis-tasks/'
EXTRA = ['w6252/analysis_script.py', 'w6252/verify_analysis.py', 'w3318/analysis_script.py']
# also catch any other relative writes inside a package: cd <pkg> && cat > x / open('x','w')
REL_WRITE = re.compile(r"(?:cat\s*>|>\s*|open\(\s*[\"'])\s*\"?((?:\./)?[A-Za-z_][\w\-]*\.(?:py|txt|json|csv|npy|npz|md|tsv|pkl))")
events = []
for wf in ['wf_59bae9e3-04a', 'wf_28496082-f75', 'wf_655a226a-c12']:
    labels = {}
    for l in open(W + wf + '/journal.jsonl'):
        e = json.loads(l)
        if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
    for f in glob.glob(W + wf + '/agent-*.jsonl'):
        lab = labels.get(os.path.basename(f)[6:-6], '?')
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            ts = e.get('timestamp', '')
            m = e.get('message') or {}
            for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
                if not (isinstance(c, dict) and c.get('type') == 'tool_use'): continue
                s = json.dumps(c.get('input', {})).replace('\\"', '"')
                for x in EXTRA:
                    if x.split('/')[1] in s and x.split('/')[0] in s:
                        kind = 'write' if re.search(r"(cat\s*>|>\s*\S*" + re.escape(x.split('/')[1]) + r"|open\([^)]*" + re.escape(x.split('/')[1]) + r"[^)]*[\"']w)", s) or c.get('name') in ('Write', 'Edit') else 'mention'
                        events.append((ts, wf, lab, x, kind, s[:240]))
                if AT in s or 'cd "' in s:
                    for mm in REL_WRITE.finditer(s):
                        fn = mm.group(1)
                        if fn.startswith('./'): fn = fn[2:]
                        if '/dev/' in s[max(0, mm.start()-5):mm.end()]: continue
                        pk = re.findall(r'analysis-tasks/(w\d+)', s)
                        if pk and ('cd ' in s) and not re.search(r'scratchpad|<TMP>/|/private/', s[max(0, mm.start()-200):mm.end()]):
                            events.append((ts, wf, lab, pk[0] + '/' + fn, 'possible-rel-write', s[max(0, mm.start()-120):mm.end()+40]))
events.sort()
for ev in events:
    if ev[4] != 'possible-rel-write' or True:
        print(ev[0][:19], ev[1][:11], ev[2], ev[3], ev[4], '|', ev[5][:200].replace('\n', ' '))
