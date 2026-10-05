#!/usr/bin/env python3
"""Second way to get the chance-match rate (checks number_trace.py).

number_trace.py estimates each number's chance rate from 50 random grid points in [0.5x, 1.5x].
Here the rate is computed exactly: every grid point in that window (same step, x itself excluded)
is tested against the same cached corpus, and the matched share is the exact coverage.
We compare the per-document means. Numbers whose window has > 200,000 grid points are skipped.
Output: results/verify_chance_rate.json
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS, write_json  # noqa: E402


def exact_coverage(vals, x, h, pct):
    step = 2 * h
    if x <= 0:
        grid = step * np.arange(1, 51)
    else:
        lo = max(step, math.ceil(0.5 * x / step) * step)
        hi = math.floor(1.5 * x / step) * step
        n = int(round((hi - lo) / step)) + 1 if hi >= lo else 0
        if n > 200_000:
            return None
        grid = lo + step * np.arange(max(n, 0))
        grid = grid[np.abs(grid - x) > step / 2]
        if len(grid) < 10:
            ks = np.arange(1, 11)
            extra = np.concatenate([x + step * ks, x - step * ks])
            grid = np.unique(np.concatenate([grid, extra[extra > 0]]))
            grid = grid[np.abs(grid - x) > step / 2]
    if len(grid) == 0:
        return None
    slack = h * 1e-9 + 1e-12 * np.maximum(1.0, np.abs(grid))
    hit = np.zeros(len(grid), bool)
    for s in ((1.0, 100.0) if pct else (1.0,)):
        lo_, hi_ = (grid - h - slack) / s, (grid + h + slack) / s
        i = np.searchsorted(vals, lo_, side="left")
        ok = i < len(vals)
        hit |= ok & (vals[np.minimum(i, len(vals) - 1)] <= hi_)
    return float(hit.mean())


def main():
    rows = [r for r in csv.DictReader(open(RESULTS / "number_trace_numbers.csv")) if r["scope"] == "run_summary"]
    by_doc = {}
    cache = {}
    for r in rows:
        run = r["run"]
        if run not in cache:
            cache[run] = np.load(RESULTS / "cache" / f"{run}.values.npy")
        ex = exact_coverage(cache[run], float(r["value"]), float(r["half_width"]), r["pct"] == "True")
        if ex is None:
            continue
        by_doc.setdefault(r["doc"], []).append((float(r["chance_rate"]), ex))
    out = {}
    all_s, all_e = [], []
    for doc, pairs in by_doc.items():
        s = np.array([p[0] for p in pairs])
        e = np.array([p[1] for p in pairs])
        all_s += list(s)
        all_e += list(e)
        out[doc] = dict(n=len(pairs), mean_sampled=round(float(s.mean()), 4), mean_exact=round(float(e.mean()), 4),
                        max_abs_diff_per_number=round(float(np.abs(s - e).max()), 4))
        print(f"{doc.split('/')[-1][:48]:48s} n={len(pairs):4d} sampled={s.mean():.3f} exact={e.mean():.3f}")
    all_s, all_e = np.array(all_s), np.array(all_e)
    res = dict(per_doc=out, n_numbers=int(len(all_s)), mean_sampled=float(all_s.mean()), mean_exact=float(all_e.mean()),
               corr=float(np.corrcoef(all_s, all_e)[0, 1]), mean_abs_diff=float(np.abs(all_s - all_e).mean()),
               note="sampled = 50 random grid points (number_trace.py); exact = all grid points in [0.5x,1.5x]")
    write_json(RESULTS / "verify_chance_rate.json", res)
    print(f"ALL n={len(all_s)} sampled={all_s.mean():.4f} exact={all_e.mean():.4f} r={res['corr']:.3f} "
          f"mean|diff|={res['mean_abs_diff']:.4f}")


if __name__ == "__main__":
    main()
