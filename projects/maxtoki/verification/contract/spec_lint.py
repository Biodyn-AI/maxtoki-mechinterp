#!/usr/bin/env python3
"""Read-only linter for the ten-section pipeline template.

Checks, per spec file:
  1. which of the ten canonical sections are present as level-2 (##) headings;
  2. whether the present ones are in canonical order;
  3. whether the heading number matches the canonical position;
  4. body size of each section (chars) -- a crude "non-empty" test;
  5. extra level-2 sections not in the template;
  6. Validation section: split into top-level items, classify each as
       - numeric-threshold (has a comparator / range / tolerance with a number)
       - numeric-mention (has a number but no comparator/range)
       - qualitative (no number)
  7. Parameters section: number of table rows; whether a "range" column exists;
  8. code citations: how many use the required `repos/<name>/path:LINE` form,
     and whether the cited file exists and has >= LINE lines (working tree only).

Writes JSON + CSV next to this script. Never writes into the repository.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path("<REPO_ROOT>")
OUT = Path(__file__).resolve().parent

SPECS = [
    # (file, deployed-on-MaxToki as one of the paper's eight?)
    ("pipelines/attention-grn-extraction-and-evaluation.md", True),
    ("pipelines/residual-stream-spectral-geometry.md", True),
    ("pipelines/topology-geometry-141-hypotheses.md", True),
    ("pipelines/manifold-discovery-extraction-compactification.md", True),
    ("pipelines/longevity-mechinterp-donor-aware.md", True),
    ("pipelines/sparse-autoencoders/01-sae-atlas.md", True),
    ("pipelines/sparse-autoencoders/02-causal-circuit-tracing.md", True),
    ("pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md", True),
    ("pipelines/audit-recurring-review-issues.md", False),
    ("pipelines/sparse-autoencoders/04-atlas-deployment.md", False),
]

CANON = [
    ("Overview", r"^overview\b"),
    ("Source", r"^sources?\b"),
    ("Inputs", r"^inputs?\b"),
    ("Outputs", r"^outputs?\b"),
    ("Dependencies", r"^dependenc"),
    ("Methodology", r"^methodology\b"),
    ("Code references", r"^code\s+ref"),
    ("Parameters", r"^parameters?\b"),
    ("Validation", r"^validation\b"),
    ("Known pitfalls", r"^(known\s+)?pitfalls?\b"),
]

H2 = re.compile(r"^##\s+(?!#)(.*)$")
NUM_PREFIX = re.compile(r"^\s*(\d+)\.\s*(.*)$")

# numbers that are NOT measurements: phase/step/layer/hypothesis/figure labels
LABEL_NUM = re.compile(
    r"(phase|step|stage|layer|iter(ation)?|fig(ure)?|table|supp(lementary)?|note|§|section|"
    r"experiment|family|test family|h|l|sv|k562|rpe1|v)\s*[-_]?\d+",
    re.I,
)
NUMBER = re.compile(r"(?<![A-Za-z_])[-+−]?\d+(?:[.,]\d+)?(?:\s*[×x]\s*10[⁻-]?\d+)?%?")
COMPARATOR = re.compile(
    r"(≥|≤|>=|<=|(?<![-=])>\s*[-+−]?\d|<\s*[-+−]?\d|±|"            # symbols
    r"\d\s*[–—-]\s*\d|"                                              # ranges 0.6–0.8
    r"\bbetween\s+[-+]?\d|\bwithin\s+[-+±]?\d|\btolerance\b|"
    r"\bat least\s+\d|\bat most\s+\d|\bmore than\s+\d|\bless than\s+\d|"
    r"\babove\s+\d|\bbelow\s+\d|\bexceed)",         # k/n counts like 22/22
    re.I,
)
ITEM_START = re.compile(r"^(\d+\.|[-*])\s+")
TABLE_ROW = re.compile(r"^\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\|\s*:?-{2,}")

CITE_REPOS = re.compile(r"repos/([A-Za-z0-9_.\-]+)/([^\s`:)\]]+?)(?::(\d+)(?:[-–](\d+))?)?(?=[\s`),\];]|$)")
CITE_BARE_PY = re.compile(r"`((?:src|iterations|iter_\d+|loop|scripts|notebooks)/[^`\s]+?\.(?:py|md|ipynb|sh))`")


def split_sections(text: str):
    lines = text.splitlines()
    sections = []  # (raw_title, start_line_idx, end_line_idx)
    in_code = False
    starts = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = H2.match(ln)
        if m:
            starts.append((m.group(1).strip(), i))
    for k, (title, i) in enumerate(starts):
        j = starts[k + 1][1] if k + 1 < len(starts) else len(lines)
        sections.append((title, i, j))
    return lines, sections


def canon_index(title: str):
    t = title
    num = None
    m = NUM_PREFIX.match(t)
    if m:
        num = int(m.group(1))
        t = m.group(2)
    t = re.sub(r"[*_`]", "", t).strip().lower()
    for idx, (_, pat) in enumerate(CANON):
        if re.search(pat, t):
            return idx, num
    return None, num


def split_items(body_lines):
    """Top-level items: numbered/bulleted lines at column 0, or table body rows."""
    items = []
    cur = None
    in_code = False
    for ln in body_lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            if cur is not None:
                cur.append(ln)
            continue
        if in_code:
            if cur is not None:
                cur.append(ln)
            continue
        if ITEM_START.match(ln):  # column 0 only
            if cur:
                items.append("\n".join(cur))
            cur = [ln]
        elif TABLE_ROW.match(ln.strip()) and not TABLE_SEP.match(ln.strip()):
            if cur:
                items.append("\n".join(cur))
                cur = None
            items.append(ln.strip())
        elif ln.strip() == "" or ln.startswith("#"):
            if cur and ln.startswith("#"):
                items.append("\n".join(cur))
                cur = None
            elif cur is not None:
                cur.append(ln)
        else:
            if cur is not None:
                cur.append(ln)
    if cur:
        items.append("\n".join(cur))
    # drop table header rows (first row of each table): heuristic = no digits and contains 'Metric'/'Check' etc.
    return [it for it in items if it.strip()]


SOURCE_ANCHOR = re.compile(r"(geneformer|scgpt|source paper|the paper|paper'?s|table s\d|supp(lementary)? note|fig\. ?s\d|in the source)", re.I)
BUG_RULE = re.compile(r"\bif\b[^.]{0,200}?(broken|bug|wrong|not actually|mis-?|is not|are not|check|indicat|suspicious|too weak|too strict|flipped|leak|fail)", re.I)


def flags(item: str):
    return dict(source_anchored=bool(SOURCE_ANCHOR.search(item)), bug_rule=bool(BUG_RULE.search(item)))


def classify(item: str):
    item = re.sub(r"^\s*(\d+\.|[-*])\s+", "", item)
    s = LABEL_NUM.sub(" ", item)
    s = re.sub(r"\(Phase[^)]*\)", " ", s, flags=re.I)
    has_num = bool(NUMBER.search(s))
    has_cmp = bool(COMPARATOR.search(s)) and has_num
    if has_cmp:
        return "numeric-threshold"
    if has_num:
        return "numeric-mention"
    return "qualitative"


def check_citations(text: str):
    out = {"repos_form": 0, "repos_form_with_line": 0, "repos_file_missing": [],
           "repos_line_out_of_range": [], "bare_paths": 0}
    for m in CITE_REPOS.finditer(text):
        name, path, line = m.group(1), m.group(2).rstrip(".,"), m.group(3)
        out["repos_form"] += 1
        f = REPO / "repos" / name / path
        if line:
            out["repos_form_with_line"] += 1
        if not f.exists():
            out["repos_file_missing"].append(f"repos/{name}/{path}")
            continue
        if line and f.is_file():
            try:
                with f.open("rb") as fh:
                    n = sum(1 for _ in fh)
                if int(line) > n:
                    out["repos_line_out_of_range"].append(f"repos/{name}/{path}:{line} (file has {n} lines)")
            except Exception:
                pass
    out["bare_paths"] = len(CITE_BARE_PY.findall(text))
    out["repos_file_missing"] = sorted(set(out["repos_file_missing"]))
    return out


def lint(relpath: str, deployed: bool):
    p = REPO / relpath
    text = p.read_text(encoding="utf-8")
    lines, sections = split_sections(text)
    found = {}
    extras = []
    heading_order = []
    number_mismatch = []
    for title, i, j in sections:
        idx, num = canon_index(title)
        body = "\n".join(lines[i + 1 : j]).strip()
        if idx is None:
            extras.append(title)
            continue
        heading_order.append(idx)
        if idx not in found:
            found[idx] = dict(title=title, line=i + 1, chars=len(body), body=lines[i + 1 : j])
        if num is not None and num != idx + 1:
            number_mismatch.append(f"'{title}' numbered {num}, canonical position {idx + 1}")
    missing = [CANON[k][0] for k in range(10) if k not in found]
    in_order = all(a < b for a, b in zip(heading_order, heading_order[1:]))
    empty = [CANON[k][0] for k, v in found.items() if v["chars"] < 200]

    # validation items
    v_items = []
    if 8 in found:
        for it in split_items(found[8]["body"]):
            first = it.strip().splitlines()[0]
            if TABLE_ROW.match(first) and not re.search(r"\d", first):
                continue  # header row
            f = flags(it)
            v_items.append((classify(it), first[:160], f["source_anchored"], f["bug_rule"]))
    counts = {"numeric-threshold": 0, "numeric-mention": 0, "qualitative": 0}
    for c, *_ in v_items:
        counts[c] += 1
    counts["source_anchored"] = sum(1 for v in v_items if v[2])
    counts["has_if_bug_rule"] = sum(1 for v in v_items if v[3])
    counts["threshold_and_not_source_anchored"] = sum(1 for v in v_items if v[0] == "numeric-threshold" and not v[2])

    # parameters table
    param_rows, has_range_col = 0, False
    if 7 in found:
        body = found[7]["body"]
        hdr_seen = False
        for ln in body:
            s = ln.strip()
            if TABLE_ROW.match(s):
                if TABLE_SEP.match(s):
                    continue
                if not hdr_seen:
                    hdr_seen = True
                    if re.search(r"range|valid|allowed|bounds", s, re.I):
                        has_range_col = True
                    continue
                if re.search(r"range|valid|allowed", s, re.I) and "---" not in s and param_rows == 0:
                    has_range_col = True
                param_rows += 1
            elif s == "":
                continue
            else:
                # a new table starts after prose -> next row is a header
                if hdr_seen and not TABLE_ROW.match(s):
                    hdr_seen = False

    cites = check_citations(text)
    return dict(
        file=relpath,
        deployed_on_maxtoki=deployed,
        chars=len(text),
        h2_headings=[t for t, _, _ in sections],
        present=[CANON[k][0] for k in sorted(found)],
        missing=missing,
        present_in_order=in_order,
        number_mismatch=number_mismatch,
        extra_sections=extras,
        short_sections_lt200chars=empty,
        validation_items=len(v_items),
        validation_counts=counts,
        validation_item_detail=v_items,
        parameters_table_rows=param_rows,
        parameters_has_range_column=has_range_col,
        citations=cites,
        verdict=("PASS" if not missing and in_order else "FAIL"),
    )


def main():
    results = [lint(f, d) for f, d in SPECS]
    (OUT / "spec_lint_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    with (OUT / "spec_lint_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "deployed", "verdict", "n_present", "missing", "in_order",
                    "number_mismatch", "extra_sections", "val_items", "val_numeric_threshold",
                    "val_numeric_mention", "val_qualitative", "param_rows", "param_range_col",
                    "cite_repos_form", "cite_repos_with_line", "cite_repos_missing_files",
                    "cite_bare_paths"])
        for r in results:
            w.writerow([r["file"], r["deployed_on_maxtoki"], r["verdict"], len(r["present"]),
                        "; ".join(r["missing"]), r["present_in_order"], len(r["number_mismatch"]),
                        "; ".join(r["extra_sections"]), r["validation_items"],
                        r["validation_counts"]["numeric-threshold"],
                        r["validation_counts"]["numeric-mention"],
                        r["validation_counts"]["qualitative"], r["parameters_table_rows"],
                        r["parameters_has_range_column"], r["citations"]["repos_form"],
                        r["citations"]["repos_form_with_line"],
                        len(r["citations"]["repos_file_missing"]), r["citations"]["bare_paths"]])
    with (OUT / "validation_items.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "source_anchored", "has_if_bug_rule", "item_first_line"])
        for r in results:
            for c, t, a, b in r["validation_item_detail"]:
                w.writerow([r["file"], c, a, b, t])
    for r in results:
        print(f"{r['verdict']:4s} {r['file']}")
        print(f"     present={len(r['present'])}/10 missing={r['missing']} in_order={r['present_in_order']}")
        print(f"     extras={r['extra_sections']}")
        if r["number_mismatch"]:
            print(f"     numbering: {r['number_mismatch']}")
        print(f"     validation items={r['validation_items']} {r['validation_counts']}")
        print(f"     parameters rows={r['parameters_table_rows']} range_col={r['parameters_has_range_column']}")
        c = r["citations"]
        print(f"     cites repos/-form={c['repos_form']} with :LINE={c['repos_form_with_line']} "
              f"missing_files={len(c['repos_file_missing'])} line_oob={len(c['repos_line_out_of_range'])} "
              f"bare_paths={c['bare_paths']}")
    n_fail = sum(r["verdict"] == "FAIL" for r in results)
    print(f"\n{n_fail}/{len(results)} specs FAIL the presence+order check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
