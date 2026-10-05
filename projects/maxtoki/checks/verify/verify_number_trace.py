#!/usr/bin/env python3
"""Second, independent way to trace FINAL_SUMMARY numbers (checks number_trace.py).

number_trace.py matches by an interval test on sorted float values. This script instead
(1) tokenises the same evidence files with a different, simpler regex and float(),
(2) formats every distinct artefact value to the written number of decimals as a STRING
    (Python round-half-even on the binary value, like the prototype checker did), and
(3) calls a number traced when its own string form is in that set.
Only plain decimals and integers are compared (numbers written with 10^k or M/K suffixes are skipped).
The two methods should agree except at exact rounding ties. Output: results/verify_number_trace.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEPLOYED, PROJ, RESULTS, evidence_files, write_json  # noqa: E402

TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def distinct_values(files):
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            buf = []
            for line in fh:
                # skip digits glued to letters/underscores (identifiers) by blanking them first
                line = re.sub(r"(?<![\d.])[A-Za-z_][A-Za-z_0-9]*", " ", line)
                buf.extend(TOKEN.findall(line))
                if len(buf) > 2_000_000:
                    parts.append(np.unique(np.abs(np.array(buf, dtype=float))))
                    buf = []
            if buf:
                parts.append(np.unique(np.abs(np.array(buf, dtype=float))))
    return np.unique(np.concatenate(parts)) if parts else np.empty(0)


def main():
    rows = list(csv.DictReader(open(RESULTS / "number_trace_numbers.csv")))
    out = {}
    tot_agree = tot = 0
    disagreements = []
    for run, _ in DEPLOYED:
        rr = [r for r in rows if r["run"] == run and r["cls"] in ("decimal_1dp", "decimal_2+dp", "integer>=10")
              and "e" not in r["raw"].rstrip("%")]
        if not rr:
            continue
        vals = distinct_values(evidence_files(PROJ / "runs" / run))
        need = {}
        for r in rr:
            hw = float(r["half_width"])
            d = max(0, int(round(-np.log10(2 * hw))))
            if abs(2 * hw - 10.0 ** -d) > 1e-12 * max(1, 10.0 ** -d):
                continue          # scaled numbers (M/K) are skipped
            need.setdefault(d, []).append(r)
        agree = n = 0
        for d, group in need.items():
            fmt = "{:.%df}" % d
            strings = set(fmt.format(v) for v in vals)
            strings_pct = set(fmt.format(v * 100) for v in vals)
            for r in group:
                x = fmt.format(float(r["value"]))
                traced2 = x in strings or (r["pct"] == "True" and x in strings_pct)
                traced1 = r["traced"] == "True"
                n += 1
                agree += traced1 == traced2
                if traced1 != traced2:
                    disagreements.append(dict(run=run, doc=r["doc"], raw=r["raw"], number_trace=traced1,
                                              string_method=traced2))
        out[run] = dict(compared=n, agree=agree, agreement=round(agree / n, 4) if n else None,
                        distinct_values_string_method=int(len(vals)))
        tot += n
        tot_agree += agree
        print(f"{run:28s} compared={n:4d} agree={agree:4d}")
    res = dict(per_run=out, total_compared=tot, total_agree=tot_agree,
               agreement=round(tot_agree / tot, 4) if tot else None, disagreements=disagreements[:200])
    write_json(RESULTS / "verify_number_trace.json", res)
    print(f"TOTAL {tot_agree}/{tot} agree; {len(disagreements)} disagreements")


if __name__ == "__main__":
    main()
