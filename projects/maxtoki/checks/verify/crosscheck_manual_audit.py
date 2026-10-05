#!/usr/bin/env python3
"""Compare number_trace.py's paper results with the manual number audits (read-only).

The investigation reports numbers_A.md, numbers_B.md and numbers_C.md hold hand-made tables with
columns "Paper line", "Value in paper" and "Status" (numbers_C has an extra leading "#" column and a
trailing "Notes" column). Status is one of MATCH, MISMATCH, NOT FOUND, DIFFERENT ENDPOINT, AMBIGUOUS
(plus free text). This script reads the column positions from each table's header row, joins every
numeric "Value in paper" to a number the tracer extracted from main.tex at ANY occurrence line within
+-3 of the row's line range (same value within rounding), and cross-tabulates the manual status
against the tracer's flags for that number.

Changed in the 2026-10-01 verification pass: the first version took columns 0/2/last for every table
(wrong for numbers_C, whose rows were therefore never joined) and matched only the first line on
which a number occurs in the paper (so repeated numbers mostly failed to join).

It answers two questions: does "found in the outputs" agree with a human's MATCH; and how many
numbers a human judged wrong (MISMATCH / DIFFERENT ENDPOINT) still look fine to a presence check?
Output: results/crosscheck_manual_audit.json and .csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import PAPER, PROJ, RESULTS, write_json  # noqa: E402
from numtext import extract  # noqa: E402

INV = PROJ / "paper-plos-one" / "revision" / "investigation"
STATUSES = ["NOT FOUND", "DIFFERENT ENDPOINT", "MISMATCH", "AMBIGUOUS", "MATCH"]


def status_of(text):
    t = text.upper()
    pos = [(t.find(s), s) for s in STATUSES if s in t]
    return min(pos)[1] if pos else "OTHER"


def _cells(ln):
    return [c.strip() for c in ln.strip().strip("|").split("|")]


def main():
    paper = list(csv.DictReader(open(RESULTS / "number_trace_paper_numbers.csv")))
    prow = {(round(float(p["value"]), 12), round(float(p["half_width"]), 15), p["pct"] == "True"): p for p in paper}
    occ = {}
    for n in extract(PAPER.read_text(encoding="utf-8"), "latex"):
        if not n.excluded:
            occ.setdefault(n.line, []).append(n)
    joined, unjoined, no_number_rows, rows_read = [], 0, 0, 0
    for fn in ("numbers_A.md", "numbers_B.md", "numbers_C.md"):
        cols = None
        for ln in (INV / fn).read_text(encoding="utf-8").splitlines():
            if not ln.lstrip().startswith("|"):
                cols = None          # a table ends at the first non-table line
                continue
            cells = _cells(ln)
            low = [c.lower() for c in cells]
            if any(c.startswith("paper line") for c in low) and any(c.startswith("value in paper") for c in low):
                cols = dict(line=next(i for i, c in enumerate(low) if c.startswith("paper line")),
                            value=next(i for i, c in enumerate(low) if c.startswith("value in paper")),
                            status=next(i for i, c in enumerate(low) if c.startswith("status")))
                continue
            if cols is None or re.match(r"^\|\s*:?-{2,}", ln.strip()) or len(cells) <= max(cols.values()):
                continue
            lines = [int(x) for x in re.findall(r"\d+", cells[cols["line"]])[:2]]
            if not lines:
                continue
            lo, hi = lines[0], lines[-1]
            if hi < lo or hi - lo > 200:
                continue
            rows_read += 1
            status = status_of(cells[cols["status"]])
            nums = [n for n in extract(cells[cols["value"]], "markdown") if not n.excluded]
            if not nums:
                no_number_rows += 1
            for n in nums:
                cand = [m for L in range(lo - 3, hi + 4) for m in occ.get(L, [])
                        if abs(m.value - n.value) <= max(m.h, n.h) + 1e-12]
                p = prow.get(cand[0].key()) if cand else None
                if p is None:
                    unjoined += 1
                    continue
                joined.append(dict(report=fn, paper_line=cand[0].line, value=p["raw"], manual_status=status,
                                   traced_any_run=p["traced"], chain=p["chain_trace"], in_summaries=p["in_summaries"],
                                   chance_union=p["chance_rate"], manual_text=cells[cols["status"]][:120]))
    # dedupe by (paper line, value, status)
    seen, rows = set(), []
    for r in joined:
        k = (r["paper_line"], r["value"], r["manual_status"])
        if k not in seen:
            seen.add(k)
            rows.append(r)
    tab = {}
    for r in rows:
        k = r["manual_status"]
        t = tab.setdefault(k, dict(n=0, traced_any_run=0, chain=0, in_summaries=0))
        t["n"] += 1
        t["traced_any_run"] += r["traced_any_run"] == "True"
        t["chain"] += r["chain"] == "True"
        t["in_summaries"] += r["in_summaries"] == "True"
    with open(RESULTS / "crosscheck_manual_audit.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    write_json(RESULTS / "crosscheck_manual_audit.json",
               dict(n_joined=len(rows), n_manual_numbers_not_joined=unjoined, n_manual_rows_read=rows_read,
                    n_manual_rows_without_a_number=no_number_rows,
                    n_distinct_values_wrong_in_meaning=len({r["value"] for r in rows
                                                            if r["manual_status"] in ("MISMATCH", "DIFFERENT ENDPOINT")}),
                    n_distinct_values_wrong_in_meaning_traced=len({r["value"] for r in rows
                                                                   if r["manual_status"] in ("MISMATCH", "DIFFERENT ENDPOINT")
                                                                   and r["traced_any_run"] == "True"}),
                    by_manual_status=tab,
                    note="traced = present in outputs within rounding; it does not test meaning. A number the "
                         "manual audit marks MISMATCH or DIFFERENT ENDPOINT can still be present in the outputs."))
    for k, t in sorted(tab.items()):
        print(f"{k:20s} n={t['n']:3d} traced_any_run={t['traced_any_run']:3d} chain={t['chain']:3d} in_summaries={t['in_summaries']:3d}")
    print(f"rows read={rows_read} (without a number: {no_number_rows}) joined={len(rows)} manual numbers not joined={unjoined}")


if __name__ == "__main__":
    main()
