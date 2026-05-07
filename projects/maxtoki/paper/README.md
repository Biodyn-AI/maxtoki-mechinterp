# Paper — agent-driven mech-interp on MaxToki-217M

ICML-workshop-style paper integrating the project's best scientific findings
across all 8 maxtoki pipelines + the audit framework.

## Files

- `main.tex` — paper source (article class, ~10 pages compiled)
- `references.bib` — bibliography (28 entries: SCFMs, mech-interp foundations,
  the project's source papers (the prior work ×7), autonomous-agent literature)
- `main.pdf` — compiled output

## Compile

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

(Requires TeX Live 2024+ or equivalent with natbib + plainnat + xspace.)

## Contributions claimed

1. **Methodological:** an agent-executable pipeline architecture for biological
   foundation-model mech-interp, incl. a recurring-issue audit pipeline that
   operationalizes peer-reviewer-flagged methodological patterns.
2. **Scientific:** six interrelated findings on MaxToki-217M, including one
   audit-driven inversion of a previously-stated negative.
3. **Honesty:** explicit failure-mode documentation (the agentic system needed
   audit-driven correction on several headline numbers before being defensible).

## Source materials

- Per-pipeline FINAL_SUMMARYs: `../summaries/`
- Audit reports: `../audits/audit-20260507{,-completion,-completion-v2}.md`
- Pipeline catalogue: `../../../pipelines/`
- Audit-pattern catalogue: `../../../pipelines/audit-recurring-review-issues.md`
- Run artefacts: `../runs/<pipeline>-<size>/`

## Notes for revision

- The paper currently runs 10 pages including refs. Many ICML workshops
  cap at 4 pages (excluding refs/appendix) or 8 pages total. Trimming
  candidates: §5.1 bullet list → prose, §5.3/5.4 prose → tighter, drop
  §6 (audit framework summary) and let it stand on §3.4 + the supplement.
- Author info is currently single-author (the authors, ).
- Reproducibility statement at end of §7 — fill in repository URL on
  de-anonymization.
- The Boyeau et al. citation (`boyeau2024scaling`) currently maps to Liu
  et al. 2024 SCFM evaluation; verify intent and adjust if needed.
- All the prior work source-paper citations use placeholder arXiv
  numbers; cross-check against the published versions before submission.
