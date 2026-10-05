#!/usr/bin/env python3
"""Build main.tex from the parts in draft/.

Order: preamble, front matter (title, abstract), introduction, methods,
results (framework evaluation, then the MaxToki-217M case study),
discussion, back matter. Each table is inserted where its part has a
'% TABLE: <name> (tab:<name>) ...' marker, which sits right after the
paragraph that first cites it (PLOS style). Tables live in
draft/25_tables_methods.tex and draft/55_table_scope.tex, one block per
'%% ---- tab:<name>' header.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'draft')
PARTS = ['05_front.tex', '10_intro.tex', '20_methods.tex', '30_results_part1.tex',
         '32_results_studyA.tex', '33_results_studyB.tex', '34_results_checks.tex',
         '50_results_casestudy.tex', '40_discussion.tex', '60_back.tex']
PENDING_DEF = r'\newcommand{\PENDING}[1]{\textcolor{red}{\textbf{[PENDING: #1]}}}'

tables = {}
for f in ['25_tables_methods.tex', '55_table_scope.tex']:
    txt = open(os.path.join(D, f)).read()
    for blk in re.split(r'(?m)^%% -+ (?=tab:)', txt)[1:]:
        name, body = blk.split('\n', 1)
        tables[name.strip()[4:]] = body.strip() + '\n'

pre = open(os.path.join(D, '00_preamble.tex')).read()
pre = pre.replace('%% END MACROS SECTION', PENDING_DEF + '\n\n%% END MACROS SECTION')
out = [pre]
used = []
for p in PARTS:
    txt = open(os.path.join(D, p)).read()
    def sub(m):
        name = m.group(1)
        used.append(name)
        return tables[name]
    txt = re.sub(r'(?m)^% TABLE: (\w+) \(tab:\w+\).*$', sub, txt)
    out.append(txt.rstrip('\n') + '\n')
missing = set(tables) - set(used)
if missing:
    sys.exit(f'tables never placed: {sorted(missing)}')
open(os.path.join(HERE, 'main.tex'), 'w').write('\n'.join(out))
n_pending = '\n'.join(out).count(r'\PENDING{') - 1
print(f'main.tex written; tables placed: {used}; PENDING markers: {n_pending}')
