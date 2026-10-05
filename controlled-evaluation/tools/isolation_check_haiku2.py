#!/usr/bin/env python3
"""Isolation check for the exploratory Haiku re-run (Amendment 3), workflow wf_6fde3184-e61.

For each subject agent (executor, reviewer, repairer): any tool call that touches plain <TMP>,
the session scratch folder, the shared task folders (analysis-tasks/w####), another agent's
own folder (analysis-tasks/u######), the answer keys, the evaluation workspace (other than the
checklist file) or the project repository. Graders are checked only for other agents' folders.
Writes studyA_haiku2/results/isolation_check.json.
"""
import json, os, glob, re, collections
W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows/wf_6fde3184-e61')
EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMAP = json.load(open(os.path.join(EV, 'studyA_haiku2', 'folder_map.json')))
SUBJ = re.compile(r':(exec|review-generic|review-checklist|repair-generic|repair-checklist)$')
pats = {'tmp': re.compile(r'(?<![\w/])(?:/private)?<TMP>/(?!claude-501)'), 'scratch': re.compile(r'scratchpad'),
        'shared_pkg_any': re.compile(r'analysis-tasks/w\d{4}'), 'keys': re.compile(r'studyA/keys|/keys/'),
        'repo': re.compile(r'biomi_automation|paper-plos-one')}
WRITE = re.compile(r"(cat\s*>|>>|(?<![<>=!\-])>\s*[\"']?/|tee\s|open\([^)]*[\"'][wa]|np\.save|to_csv\(|mkdir\s)")
labels = {}
for l in open(os.path.join(W, 'journal.jsonl')):
    e = json.loads(l)
    if e.get('type') == 'started': labels[e['agentId']] = e.get('label', '')
res = {}; summ = collections.Counter()
for f in glob.glob(os.path.join(W, 'agent-*.jsonl')):
    lab = labels.get(os.path.basename(f)[6:-6], '?')
    subj = bool(SUBJ.search(lab)); own = FMAP.get(lab, '')
    hits = collections.defaultdict(list)
    for l in open(f):
        try: e = json.loads(l)
        except Exception: continue
        m = e.get('message') or {}
        for c in (m.get('content') if isinstance(m, dict) and isinstance(m.get('content'), list) else []):
            if not (isinstance(c, dict) and c.get('type') == 'tool_use'): continue
            s = json.dumps(c.get('input', {})).replace('\\"', '"')
            for other in set(re.findall(r'analysis-tasks/(u\d{6})', s)):
                if not own.endswith(other): hits['other_agent_folder'].append(s[:160])
            if subj:
                for k, p in pats.items():
                    if p.search(s):
                        if k == 'shared_pkg_any':
                            k = 'shared_pkg_write' if WRITE.search(s) else 'shared_pkg_read'
                        hits[k].append(s[:160])
                if 'maxtoki-framework-eval' in s and 'CHECKLIST_GENERIC.md' not in s:
                    hits['eval_workspace'].append(s[:160])
    res[lab] = {k: len(v) for k, v in hits.items()}
    res[lab]['_examples'] = {k: v[:2] for k, v in hits.items()}
    for k, v in hits.items():
        if v: summ[(('subject' if subj else 'grader'), k)] += 1
print('agents:', collections.Counter('subject' if SUBJ.search(l) else 'grader' for l in res))
for k, v in sorted(summ.items()): print(k, v)
json.dump(res, open(os.path.join(EV, 'studyA_haiku2', 'results', 'isolation_check.json'), 'w'), indent=1)
