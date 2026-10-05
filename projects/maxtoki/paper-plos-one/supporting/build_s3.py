#!/usr/bin/env python3
"""Build S3 Appendix (corrected MaxToki-217M analyses) from the section files in s3/.

The sections are written as Markdown, one per analysis (00_common.md, 01_... , 10_...).
They are joined in file-name order under a short introduction, and converted with the same
pandoc + xelatex settings as S2 Appendix (build_s2.run_pandoc). Writes S3_Appendix.md and
S3_Appendix.pdf next to this script.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_s2 as b

b.TITLE = "S3 Appendix. Corrected MaxToki-217M analyses"
# Sections are numbered like their files (00, 01, ...), because the text refers to them that way.
b.HEADER_TEX = b.HEADER_TEX + r"""
\setcounter{section}{-1}
\renewcommand{\thesection}{\ifnum\value{section}<10 0\fi\arabic{section}}
\setlength{\cftsecnumwidth}{2.6em}
"""
INTRO = """This appendix describes each corrected analysis of MaxToki-217M reported in the paper: the data, the method in enough detail to repeat it, the checks that were run, the full results, how the key numbers were derived a second time, and what was not tested. Section 00 describes the input encoding and the intervention code shared by several analyses. Paths are relative to the root of the code and data release.
"""


def main():
    parts = sorted(p for p in (HERE / 's3').glob('[0-9][0-9]_*.md') if not p.name.startswith('._'))
    if not parts:
        sys.exit('no section files in s3/')
    out = ['# Contents of this appendix {-}', '', INTRO]
    for p in parts:
        txt = p.read_text(encoding='utf-8').strip('\n')
        # each section file starts with '## Title' -> make it a top-level section of the appendix
        txt = re.sub(r'(?m)^(#{2,6}) ', lambda m: '#' * (len(m.group(1)) - 1) + ' ', txt)
        out += ['', txt, '']
        print(f'[part] {p.name}')
    md = HERE / 'S3_Appendix.md'
    pdf = HERE / 'S3_Appendix.pdf'
    md.write_text('\n'.join(out) + '\n', encoding='utf-8')
    r = b.run_pandoc(md, pdf)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(f'pandoc failed with code {r.returncode}')
    missing = [l for l in r.stderr.split('\n') if 'Missing character' in l]
    print(f'[check] missing-character warnings: {len(missing)}')
    for l in missing[:10]:
        print('   ', l)
    print(f'[ok] wrote {pdf}')
    b.page_report(pdf)


if __name__ == '__main__':
    main()
