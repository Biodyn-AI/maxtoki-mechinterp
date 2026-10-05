#!/usr/bin/env python3
"""Check the PLOS ONE submission (main.tex, main.pdf, figures) against the journal's rules,
and export the figures under their submission names.

Run from paper-plos-one/ after building:
    python3 assemble.py && pdflatex main && bibtex main && pdflatex main && pdflatex main
    python3 check_plos.py            # add --allow-pending while results are still missing
Figures: each label fig:<name> is drawn by figures-src/fig_<name>.py into figures/Fig_<name>.tif;
this script copies them to submission-figures/Fig<k>.tif in order of first citation.
Exits non-zero on any failure.
"""
import os, re, shutil, subprocess, sys

ALLOW_PENDING = '--allow-pending' in sys.argv
FAIL = []
def chk(ok, label, detail=''):
    print(('  PASS  ' if ok else '  FAIL  ') + label + (('  | ' + detail) if detail else ''))
    if not ok:
        FAIL.append(label + ((' | ' + detail) if detail else ''))

tex = open('main.tex', encoding='utf-8').read()
pdf = subprocess.run(['pdftotext', 'main.pdf', '-'], capture_output=True, text=True).stdout
log = open('main.log', encoding='latin-1').read() if os.path.exists('main.log') else ''
live = '\n'.join(l for l in tex.split('\n') if not l.lstrip().startswith('%'))
body = live[live.index('\\begin{document}'):]

print('1. TEMPLATE AND FORMATTING')
chk('\\documentclass[10pt,letterpaper]{article}' in live, 'official PLOS template class')
chk('\\bibliographystyle{plos2025}' in live and os.path.exists('plos2025.bst'), 'PLOS Vancouver style (plos2025.bst)')
chk('\\linenumbers' in live and '\\nolinenumbers' in live, 'continuous line numbering')
chk('\\doublespacing' in live, 'double spacing')
chk('\\usepackage{cite}' in live, 'numeric citations')
chk('Undefined' not in log and 'undefined references' not in log, 'no undefined references or citations in the LaTeX log')
chk(not re.findall(r'\[\?+\]', pdf), 'no unresolved [?] in the PDF')

print('2. NO GRAPHICS IN THE MANUSCRIPT')
chk('includegraphics' not in live and 'tikzpicture' not in live, 'no \\includegraphics or TikZ')
imgs = subprocess.run(['pdfimages', '-list', 'main.pdf'], capture_output=True, text=True).stdout
n_img = len([l for l in imgs.strip().split('\n') if re.match(r'^\s*\d', l)])
chk(n_img == 0, 'manuscript PDF embeds zero images', f'{n_img} found')

print('3. TITLE AND ABSTRACT')
m = re.search(r'\{\\Large\s*\\textbf\{(.*?)\}\s*\}', body, re.S)
title = re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
chk(0 < len(title) <= 250, 'title <= 250 characters', f'{len(title)}')
abstract = body[body.index('\\section*{Abstract}') + 19:body.index('\\clearpage')]
words = len([w for w in abstract.split() if any(c.isalnum() for c in w)])
chk(words <= 300, 'abstract <= 300 words', f'{words}')
chk('\\cite' not in abstract, 'abstract has no citations')

print('4. SECTION STRUCTURE')
order = [x.group(1) for x in re.finditer(r'\\section\*\{([^}]*)\}', live)]
want = ['Abstract', 'Introduction', 'Materials and methods', 'Results', 'Discussion', 'Conclusion',
        'Supporting information', 'Acknowledgments']
chk(order == want, 'sections present and in PLOS order', f'{order}')
chk('\\section{' not in live and '\\subsection{' not in live, 'all headings unnumbered')
chk(re.search(r'\\subsection\*\{Declaration of generative AI', live[:live.index('\\section*{Results}')]) is not None,
    'AI disclosure is a subsection of Materials and methods')
for bad in ['CRediT', 'Declaration of competing interest', '\\section*{Funding}', '\\section*{Data and code availability}']:
    chk(bad not in live, f'form-field statement absent from manuscript: {bad}')

print('5. FIGURES')
labels = re.findall(r'\\begin\{figure\}.*?\\label\{(fig:[^}]*)\}.*?\\end\{figure\}', body, re.S)
cite_pos = {l: body.find('\\ref{%s}' % l) for l in labels}
chk(all(p >= 0 for p in cite_pos.values()), 'every figure is cited', str([l for l, p in cite_pos.items() if p < 0]))
by_cite = sorted(labels, key=lambda l: cite_pos[l])
chk(by_cite == labels, 'figure captions appear in order of first citation', f'{labels} vs {by_cite}')
late = [l for l in labels if cite_pos[l] > body.find('\\label{%s}' % l)]
chk(not late, 'each figure cited before its caption', str(late))
nb = sum(1 for b in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', body, re.S) if re.search(r'\\caption\{\s*\\textbf', b))
chk(nb == len(labels), 'every caption opens with a bold title', f'{nb}/{len(labels)}')
chk('Figure~\\ref' not in live and 'Figures~\\ref' not in live, "in-text style is 'Fig'")
os.makedirs('submission-figures', exist_ok=True)
for f in os.listdir('submission-figures'):
    if re.match(r'Fig\d+\.tif$', f):
        os.remove(os.path.join('submission-figures', f))
missing, outspec = [], []
try:
    from PIL import Image
except ImportError:
    Image = None
for k, lab in enumerate(labels, 1):
    src = os.path.join('figures', 'Fig_' + lab[4:] + '.tif')
    if not os.path.exists(src):
        missing.append(src); continue
    dst = os.path.join('submission-figures', f'Fig{k}.tif')
    shutil.copyfile(src, dst)
    if Image:
        im = Image.open(dst); w, h = im.size; mb = os.path.getsize(dst) / 1e6
        if not (789 <= w <= 2250 and h <= 2625 and mb <= 10 and im.mode in ('RGB', 'L')):
            outspec.append(f'Fig{k} {lab} ({w}x{h},{im.mode},{mb:.1f}MB)')
chk(not missing, 'figure file present for every caption', str(missing))
chk(not outspec, 'all figures within PLOS pixel and size limits', str(outspec))
print('        figure order: ' + ', '.join(f'Fig{k}={l}' for k, l in enumerate(labels, 1)))

print('6. TABLES')
tlabels = re.findall(r'\\begin\{longtable\}.*?\\label\{(tab:[^}]*)\}', body, re.S)
tpos = {l: body.find('\\ref{%s}' % l) for l in tlabels}
chk(all(p >= 0 for p in tpos.values()), 'every table is cited', str([l for l, p in tpos.items() if p < 0]))
chk(sorted(tlabels, key=lambda l: tpos[l]) == tlabels, 'tables appear in order of first citation', str(tlabels))
chk(all(tpos[l] < body.find('\\label{%s}' % l) for l in tlabels), 'each table cited before it appears')
chk(live.count('\\makecell') == 0 and all('\\newline' not in t for t in re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}', live, re.S)),
    'no nested cells or forced line breaks in tables')

print('7. SUPPORTING INFORMATION')
for s in ['S1 Table', 'S2 Appendix', 'S3 Appendix']:
    key = s.replace(' ', '~')
    chk(key in body[:body.index('\\section*{Supporting information}')], f'{s} cited in the text')
    chk(re.search(r'\\paragraph\*\{' + s + r'\.\}', body) is not None, f'{s} has a caption in Supporting information')

print('8. REFERENCES')
nbib = open('main.bbl', encoding='utf-8').read().count('\\bibitem') if os.path.exists('main.bbl') else 0
chk(nbib >= 30, 'reference list populated', f'{nbib} entries')
chk(not re.findall(r'arXiv:\d{7,}', pdf), 'arXiv identifiers keep their decimal point')
chk(not re.findall(r'\d{4}\. \.', pdf), 'no reference ends with a stray period')
chk('\\footnote{' not in live, 'no footnotes')

print('9. NO DRAFT OR REVISION ARTEFACTS')
n_pend = len(re.findall(r'\\PENDING\{', body))
chk(ALLOW_PENDING or n_pend == 0, 'no PENDING markers', f'{n_pend} left')
flat = re.sub(r'\s+', ' ', re.sub(r'%.*', '', body))
terms = ['review round', 'resubmi', 'rebuttal', 'response to review', 'response to the review', 'as requested', 'we now', 'now report', 'updated',
         'previously', 'earlier version', 'prior version', 'this version', 'originally reported', 'TODO', 'TBD',
         'XXX', 'in response', 'revised manuscript', 'first submission']
hits = [t for t in terms if t.lower() in flat.lower()]
chk(not hits, 'no revision wording', str(hits))
rev = [x.group(0) for x in re.finditer(r'.{40}\b(revision|revised|reviewers? (?:asked|noted|requested|pointed))\b.{40}', flat, re.I)]
ok_rev = [r for r in rev if any(k in r for k in ('revision \\texttt', 'revised the deliverable', 'revised a deliverable', 'MaxToki revision', 'exact revision', 'revisions, so the hashes'))]  # model revisions on Hugging Face
chk(len(rev) == len(ok_rev), "'revision' only as a model revision or the repair step", str([r for r in rev if r not in ok_rev]))

print('10. RESULTS MAP')
if os.path.exists('RESULTS_MAP.csv'):
    import csv
    rmap = list(csv.DictReader(open('RESULTS_MAP.csv', encoding='utf-8', newline='')))
    flat_tex = re.sub(r'\s+', ' ', tex)
    stale = [r['id'] for r in rmap if re.sub(r'\s+', ' ', r['quote']).strip() not in flat_tex]
    chk(not stale, 'every RESULTS_MAP.csv quote appears verbatim in main.tex', f'{len(stale)} stale: {stale[:10]}')
    chk(not [r for r in rmap if r['status'] == 'mismatch'], 'no RESULTS_MAP.csv row marked mismatch')
else:
    chk(False, 'RESULTS_MAP.csv present')

print('=' * 70)
if FAIL:
    print(f'RESULT: {len(FAIL)} CHECK(S) FAILED')
    for f in FAIL:
        print('   - ' + f)
    sys.exit(1)
print('RESULT: ALL CHECKS PASSED')
