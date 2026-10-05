#!/usr/bin/env python3
"""Second, independent way to get the section verdict (checks spec_lint.py S1 and S6 counts).

(1) Re-derives which of the ten template sections are present using plain string tests on '## '
    lines (no shared parsing code), and compares with spec_lint.json.
(2) Compares with the prototype linter written during the investigation
    (verification/contract/spec_lint_summary.csv), read-only.
Output: results/verify_spec_lint.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEPLOYED, OTHER_SPECS, PROJ, RESULTS, ROOT, write_json  # noqa: E402

WORDS = [("Overview", "overview"), ("Source", "source"), ("Inputs", "input"), ("Outputs", "output"),
         ("Dependencies", "dependenc"), ("Methodology", "methodolog"), ("Code references", "code ref"),
         ("Parameters", "parameter"), ("Validation", "validation"), ("Known pitfalls", "pitfall")]


def simple_sections(path: Path):
    heads, fence = [], False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("```"):
            fence = not fence
        if not fence and ln.startswith("## "):
            heads.append(ln[3:].lower())
    present = []
    for name, w in WORDS:
        # heading text after an optional number like "7. "
        if any(h.split(". ", 1)[-1].startswith(w) or h.split(". ", 1)[-1].startswith("known " + w) for h in heads):
            present.append(name)
    return present


def main():
    lint = {r["spec"]: r for r in json.loads((RESULTS / "spec_lint.json").read_text())}
    proto_path = PROJ / "verification/contract/spec_lint_summary.csv"
    proto = {r["file"]: r for r in csv.DictReader(open(proto_path))} if proto_path.exists() else {}
    rows = []
    for spec in [s for _, s in DEPLOYED] + OTHER_SPECS:
        present = simple_sections(ROOT / spec)
        missing = [n for n, _ in WORDS if n not in present]
        r = lint[spec]
        p = proto.get(spec, {})
        rows.append(dict(
            spec=spec, simple_missing=missing, lint_missing=r["sections"]["missing"],
            agree_missing=(missing == r["sections"]["missing"]),
            prototype_missing=p.get("missing", "").split("; ") if p.get("missing") else [],
            agree_with_prototype=((p.get("missing", "").split("; ") if p.get("missing") else []) == r["sections"]["missing"]),
            lint_val_items=r["validation"].get("n_items"), prototype_val_items=int(p["val_items"]) if p else None,
            lint_val_threshold=r["validation"].get("numeric-threshold"),
            prototype_val_threshold=int(p["val_numeric_threshold"]) if p else None,
        ))
        print(f"{spec.split('/')[-1][:44]:44s} simple_missing={missing} agree={rows[-1]['agree_missing']} "
              f"proto_agree={rows[-1]['agree_with_prototype']} val {rows[-1]['lint_val_items']}/{rows[-1]['prototype_val_items']} "
              f"thr {rows[-1]['lint_val_threshold']}/{rows[-1]['prototype_val_threshold']}")
    write_json(RESULTS / "verify_spec_lint.json", dict(rows=rows,
               all_agree_simple=all(r["agree_missing"] for r in rows),
               all_agree_prototype=all(r["agree_with_prototype"] for r in rows)))


if __name__ == "__main__":
    main()
