#!/usr/bin/env python3
"""Second way to read the values behind every flagged parameter deviation (checks run_manifest_check.py).

run_manifest_check.py reads script constants with Python's ast module and JSON with json.load.
Here each flagged value is re-read from raw text: the cited script line is matched with a plain
regex (`NAME = <number>` or an environment default `"..."`), and JSON values are found by a regex
on the raw file text. Output: results/verify_manifest_deviations.json
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS, ROOT, write_json  # noqa: E402

EV = re.compile(r"(?P<path>[^;:]+?):(?P<loc>[^=;]+)=(?P<val>[^\[;]+)\[(?P<kind>[^|\]]+)\|(?P<status>[^\]]+)\]")


def main():
    rows = [r for r in csv.DictReader(open(RESULTS / "run_manifest_params.csv")) if r["deviation"] == "True"]
    checks = []
    for r in rows:
        for m in EV.finditer(r["evidence"]):
            if m.group("status") not in ("OUT_OF_RANGE", "DIFFERS_NO_RANGE_GIVEN"):
                continue
            path = ROOT / "projects/maxtoki/runs" / m.group("path").strip()
            loc, val, kind = m.group("loc").strip(), float(m.group("val")), m.group("kind")
            raw = None
            if kind == "json":
                key = loc.split(".")[-1]
                txt = path.read_text(encoding="utf-8", errors="replace")
                found = [float(x) for x in re.findall(r'"' + re.escape(key) + r'"\s*:\s*(-?[\d.eE+-]+)', txt)]
                ok = any(abs(x - val) < 1e-9 for x in found)
                raw = f"raw JSON values for '{key}': {found[:6]}"
            else:
                line = path.read_text(encoding="utf-8", errors="replace").splitlines()[int(loc) - 1]
                nums = [float(x.replace("_", "")) for x in re.findall(r"(?<![\w.])(\d[\d_]*(?:\.\d+)?(?:e-?\d+)?)", line)]
                ok = any(abs(x - val) < 1e-9 for x in nums)
                raw = line.strip()[:140]
            checks.append(dict(run=r["run"], param=r["param"], file=str(path.relative_to(ROOT)), loc=loc, value=val,
                               kind=kind, confirmed_from_raw_text=ok, raw=raw))
    n_ok = sum(c["confirmed_from_raw_text"] for c in checks)
    write_json(RESULTS / "verify_manifest_deviations.json", dict(n_values=len(checks), n_confirmed=n_ok, checks=checks))
    for c in checks:
        print(f"{'OK ' if c['confirmed_from_raw_text'] else 'BAD'} {c['run'][:22]:22s} {c['param']:34s} {c['value']:g}  <- {c['raw'][:90]}")
    print(f"{n_ok}/{len(checks)} flagged values confirmed from raw text")


if __name__ == "__main__":
    main()
