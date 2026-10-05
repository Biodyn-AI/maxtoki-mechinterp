#!/usr/bin/env python3
"""Strict isolation check for the subject agents of Studies A, B and C.

For every subject agent (executor, reviewer, repairer, re-auditor; graders are
allowed to read keys and are skipped) scan every tool-call input and flag:
  - any path into the project repository, the evaluation workspace (keys,
    protocol, results), the manuscript, or Claude Code project/memory files
    (the agent's own saved tool results under .../tool-results/ are allowed);
  - any task/item folder under analysis-tasks/ other than the agent's own
    package (the shared bin/ python wrapper is allowed).
Usage: contamination_strict.py <out_json>
"""
import json, glob, os, re, sys, collections

W = os.path.expanduser('~/.claude/projects/<SESSION_DIR>/'
                       'd422dc33-6d6f-4b2b-bbf7-f274d8a82fae/subagents/workflows')
EV = '<EVAL_ROOT>'
STUDIES = {'A': ['wf_28496082-f75', 'wf_59bae9e3-04a'], 'B': ['wf_7e9ab73d-e56'], 'C': ['wf_6ef77bf7-9cd']}
FORBID = ['biomi_automation', 'maxtoki-framework-eval', 'paper-plos-one', 'MEMORY.md', 'CLAUDE.md', '.claude/projects']
SUBJ = re.compile(r':(exec|review|review-generic|review-checklist|repair-generic|repair-checklist)$')
amap = json.load(open(os.path.join(EV, 'protocol/STUDYA_PACKAGE_MAP.json')))
own_A = {(v['task'], v['arm']): k for k, v in amap.items()}
PKG = re.compile(r'analysis-tasks/([A-Za-z0-9_\-]+)')
# Reviewers were given the checklist by path; Study C auditors were given the snapshot by path.
EVPATH = re.compile(r'maxtoki-framework-eval/?([^\s"\\\']*)')
ALLOWED_EV = ('protocol/CHECKLIST_GENERIC.md', 'protocol/CHECKLIST_DEPLOYED.md', 'studyC/snapshot', 'bin/python')


def tool_inputs(path):
    for l in open(path):
        try:
            e = json.loads(l)
        except Exception:
            continue
        msg = e.get('message') or {}
        content = msg.get('content') if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get('type') == 'tool_use':
                yield json.dumps(c.get('input', {}))


def own_package(study, label):
    p = label.split(':')
    if study == 'A':
        return {own_A[(p[1], p[2])]}
    if study == 'B':
        return {p[1]}
    return None  # Study C: own package found from the snapshot location below


report = {}
for study, wfs in STUDIES.items():
    for wf in wfs:
        d = os.path.join(W, wf)
        labels = {}
        for l in open(os.path.join(d, 'journal.jsonl')):
            e = json.loads(l)
            if e.get('type') == 'started':
                labels[e['agentId']] = e.get('label', '')
        for f in glob.glob(os.path.join(d, 'agent-*.jsonl')):
            lab = labels.get(os.path.basename(f)[6:-6], '')
            if not lab.startswith(study + ':'):
                continue
            if study != 'C' and not SUBJ.search(lab):
                continue
            if study == 'C' and len(lab.split(':')) != 3:
                continue
            own = own_package(study, lab)
            hits, pk = [], collections.Counter()
            for s in tool_inputs(f):
                if '/tool-results/' in s:
                    s2 = s.replace('/tool-results/', '')
                else:
                    s2 = s
                for pat in FORBID:
                    if pat == 'maxtoki-framework-eval':
                        bad = [m for m in EVPATH.findall(s) if not m.startswith(ALLOWED_EV)]
                        if bad:
                            hits.append({'pattern': pat, 'paths': sorted(set(bad))[:5], 'call': s[:300]})
                    elif pat in s2 and not ('.claude/projects' == pat and '/tool-results/' in s):
                        hits.append({'pattern': pat, 'call': s[:300]})
                for m in PKG.findall(s):
                    pk[m] += 1
            other = {k: v for k, v in pk.items() if k != 'bin' and (own is None or k not in own)}
            report[f'{wf}|{lab}'] = {'forbidden': hits, 'packages': dict(pk), 'other_packages': other}

summ = collections.defaultdict(lambda: collections.Counter())
for k, v in report.items():
    st = k.split('|')[1][0]
    summ[st]['agents'] += 1
    summ[st]['forbidden'] += bool(v['forbidden'])
    if st != 'C':
        summ[st]['other_pkg'] += bool(v['other_packages'])
    else:
        summ[st]['distinct_pkgs'] += len(v['packages'])
for st in sorted(summ):
    print(st, dict(summ[st]))
for k, v in report.items():
    if v['forbidden'] or (k.split('|')[1][0] != 'C' and v['other_packages']):
        print('HIT', k, v['other_packages'], [(h['pattern'], h.get('paths'), h['call'][:200]) for h in v['forbidden'][:3]])
    if k.split('|')[1][0] == 'C':
        print('C', k, v['packages'])
json.dump(report, open(sys.argv[1], 'w'), indent=1)
