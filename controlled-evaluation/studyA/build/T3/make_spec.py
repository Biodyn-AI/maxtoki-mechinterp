"""Copy the deployed SAE-atlas pipeline spec (file dated 2026-04-15) to contract/SPEC.md.
Only local file-system paths are rewritten (package-relative or repository-relative); no other text changes."""
from pathlib import Path
SRC = Path("<REPO_ROOT>/pipelines/sparse-autoencoders/01-sae-atlas.md")
DST = Path("<EVAL_ROOT>/studyA/build/T3/SPEC.md")
txt = SRC.read_text()
rep = [
    ("`../../references/2603.02952_Kendiukhov_SAE_atlas.pdf`", "`methods/source_method_paper.pdf` (in this package)"),
    ("Local clone at `../../repos/bio-sae/`.", "The repository is not included in this package; code paths below are relative to its root."),
    ("`../../repos/bio-sae/", "`"),
    ("`repos/bio-sae/", "`"),
    ("`../attention-grn-extraction-and-evaluation.md`", "`attention-grn-extraction-and-evaluation.md`"),
]
for a, b in rep:
    n = txt.count(a); assert n > 0, a
    txt = txt.replace(a, b); print(n, "x", a)
assert "../" not in txt and "repos/" not in txt and "/Volumes" not in txt
DST.write_text(txt)
