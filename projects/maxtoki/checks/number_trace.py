#!/usr/bin/env python3
"""number_trace.py -- can each number in a FINAL_SUMMARY (and in the paper) be found in run outputs?

For every number written in
  (a) each run's FINAL_SUMMARY file(s)  -> searched in THAT run's machine-readable outputs, and
  (b) the paper (main.tex body)          -> searched in the union of all eight runs' outputs
                                            (and, separately, an extended set that adds audits/,
                                            summaries/ and setup/ json/csv/log/txt),
the tracer asks: is there an artefact value v with |v - x| <= half a unit of the last written
digit? (Percent numbers may also match 100*v.) Markdown, HTML and .py files never count as
evidence, so prose cannot vouch for itself.

Chance matches. A short number like 0.5 or 12 will be "found" in almost any large CSV. To measure
this, every number x is replaced by K random numbers of the same written precision and similar
size (grid points in [0.5x, 1.5x], never x itself) and those are searched in the same corpus.
The share of them that match is that number's chance-match rate. For a document we report:
  traced share (Wilson 95% CI; unit = distinct number),
  mean chance-match rate (percentile bootstrap 95% CI, 2,000 resamples of numbers),
  excess = traced share - mean chance rate (paired bootstrap over numbers).
A traced number whose own chance rate is >= 0.5 is labelled a WEAK trace.

What it does NOT establish: that a traced number was computed correctly, belongs to the claim it
supports, or came from the file where it was found; an untraced number may still be correct
(e.g. computed only in memory, stored in .npy, or quoted from another run or the source paper).

Outputs (checks/results/): number_trace_summary.csv, number_trace_numbers.csv,
number_trace_paper_numbers.csv, number_trace.json, number_trace_run_config.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
from common import (DEPLOYED, PAPER, PROJ, RESULTS, ROOT, SEED, evidence_files, rel,  # noqa: E402
                    run_config, summary_docs, write_json)
from numtext import dedupe, extract  # noqa: E402

K_PERTURB = 50
N_BOOT = 2000
WEAK = 0.5


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def boot_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def stable_seed(*parts):
    h = hashlib.sha256(("|".join(map(str, parts)) + f"|{SEED}").encode()).hexdigest()
    return int(h[:15], 16)


def perturbations(n, k=K_PERTURB):
    """K values on the same precision grid as n, in [0.5x, 1.5x], excluding x itself."""
    step = 2 * n.h
    x = n.value
    rng = np.random.default_rng(stable_seed(n.raw, n.value, n.h))
    if x <= 0:
        grid = step * np.arange(1, 51)
    else:
        lo = max(step, math.ceil(0.5 * x / step) * step)
        hi = math.floor(1.5 * x / step) * step
        npts = int(round((hi - lo) / step)) + 1 if hi >= lo else 0
        if npts > 200_000:
            grid = None
        else:
            grid = lo + step * np.arange(max(npts, 0))
            grid = grid[np.abs(grid - x) > step / 2]
            if len(grid) < 10:
                ks = np.arange(1, 11)
                extra = np.concatenate([x + step * ks, x - step * ks])
                grid = np.unique(np.concatenate([grid, extra[extra > 0]]))
                grid = grid[np.abs(grid - x) > step / 2]
    if grid is None:   # very fine grid: sample uniformly then snap to the grid
        u = rng.uniform(0.5 * x, 1.5 * x, size=k * 3)
        g = np.round(u / step) * step
        g = g[np.abs(g - x) > step / 2]
        return g[:k]
    return rng.choice(grid, size=k, replace=True)


def trace_one(n, vals, fids, files):
    scales = (1.0, 100.0) if n.pct else (1.0,)
    i, s = corpus.match(vals, n.value, n.h, scales)
    traced = i >= 0
    first_file = files[fids[i]] if traced else None
    matched_value = float(vals[i]) if traced else None
    relaxed = None
    if not traced and n.trailing_zero_h:
        j, _ = corpus.match(vals, n.value, n.trailing_zero_h, scales)
        relaxed = j >= 0
    pert = perturbations(n)
    hits = sum(corpus.match(vals, float(p), n.h, scales)[0] >= 0 for p in pert)
    chance = hits / len(pert) if len(pert) else float("nan")
    return dict(traced=traced, matched_value=matched_value, matched_scale=s, first_file=first_file,
                relaxed_trailing_zero=relaxed, chance_rate=chance, weak=bool(traced and chance >= WEAK))


def classify(n):
    if n.kind == "sci":
        return "sci"
    if n.kind == "integer":
        return "integer>=10"
    return "decimal_1dp" if n.decimals == 1 else "decimal_2+dp"


def doc_stats(rows, rng):
    n = len(rows)
    k = sum(r["traced"] for r in rows)
    ch = [r["chance_rate"] for r in rows if not math.isnan(r["chance_rate"])]
    ex = [float(r["traced"]) - r["chance_rate"] for r in rows if not math.isnan(r["chance_rate"])]
    lo, hi = wilson(k, n)
    clo, chi = boot_ci(ch, rng)
    elo, ehi = boot_ci(ex, rng)
    return dict(n_numbers=n, n_traced=k, traced_share=(k / n if n else float("nan")), traced_ci_lo=lo,
                traced_ci_hi=hi, mean_chance=(float(np.mean(ch)) if ch else float("nan")),
                chance_ci_lo=clo, chance_ci_hi=chi, excess=(float(np.mean(ex)) if ex else float("nan")),
                excess_ci_lo=elo, excess_ci_hi=ehi, n_weak=sum(r["weak"] for r in rows),
                n_untraced=n - k, n_strong_traced=sum(r["traced"] and not r["weak"] for r in rows))


def spec_keys(spec_rel):
    nums = dedupe(extract((ROOT / spec_rel).read_text(encoding="utf-8"), "markdown"))
    return {n.key() for n in nums}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="subset of run names")
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any FINAL_SUMMARY number that is not also a spec number is untraced")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    runs = [(r, s) for r, s in DEPLOYED if not args.runs or r in args.runs]
    all_rows, summary, inputs = [], [], []
    run_corpora = {}
    for run, spec in runs:
        files = evidence_files(PROJ / "runs" / run)
        inputs += files
        print(f"[{run}] building corpus from {len(files)} files ...", flush=True)
        vals, fids, flist = corpus.build(run, files, rebuild=args.rebuild_cache)
        run_corpora[run] = (vals, fids, flist)
        skeys = spec_keys(spec)
        run_union = {}
        for doc in summary_docs(run):
            inputs.append(doc)
            nums = dedupe(extract(doc.read_text(encoding="utf-8"), "markdown"))
            rows = []
            for n in nums:
                t = trace_one(n, vals, fids, flist)
                r = dict(scope="run_summary", run=run, doc=rel(doc), raw=n.raw, value=n.value, half_width=n.h,
                         cls=classify(n), pct=n.pct, approx=n.approx, line=n.line,
                         n_occurrences=len(n.extra["occurrences"]), also_in_spec=(n.key() in skeys),
                         context=n.context[:160], **t)
                rows.append(r)
                run_union.setdefault(n.key(), r)
            all_rows += rows
            st = doc_stats(rows, rng)
            st.update(scope="run_summary", run=run, doc=rel(doc), corpus_files=len(files),
                      corpus_values=int(len(vals)))
            summary.append(st)
            ns = [r for r in rows if not r["also_in_spec"]]
            st2 = doc_stats(ns, rng)
            st2.update(scope="run_summary_excluding_spec_numbers", run=run, doc=rel(doc),
                       corpus_files=len(files), corpus_values=int(len(vals)))
            summary.append(st2)
            print(f"   {rel(doc)}: {st['n_traced']}/{st['n_numbers']} traced "
                  f"(chance {st['mean_chance']:.2f}, excess {st['excess']:.2f}, weak {st['n_weak']})", flush=True)
        urows = list(run_union.values())
        st = doc_stats(urows, rng)
        st.update(scope="run_union_of_summaries", run=run, doc="(all FINAL_SUMMARY files)",
                  corpus_files=len(files), corpus_values=int(len(vals)))
        summary.append(st)
    # ---------------------------------------------------------------- paper
    paper_rows = []
    if not args.skip_paper:
        inputs.append(PAPER)
        print("[paper] building union corpus ...", flush=True)
        uv = np.unique(np.concatenate([run_corpora[r][0] for r in run_corpora]))
        ufid = np.zeros(len(uv), dtype=np.int32)
        extra_files = []
        for sub in ("audits", "summaries", "setup"):
            extra_files += [p for p in evidence_files(PROJ / sub)]
        inputs += extra_files
        ev, efid, eflist = corpus.build("extended_audits_summaries_setup", extra_files, rebuild=args.rebuild_cache)
        xv = np.unique(np.concatenate([uv, ev]))
        # values written in any FINAL_SUMMARY (markdown) -- for the paper-vs-summary transcription check
        sv, sv_run = [], {}
        for r in run_corpora:
            vals_r = []
            for doc in summary_docs(r):
                vals_r += [n.value for n in dedupe(extract(doc.read_text(encoding="utf-8"), "markdown"))]
            sv_run[r] = np.unique(np.array(vals_r, dtype=float))
            sv += vals_r
        sv = np.unique(np.array(sv, dtype=float))
        all_spec_keys = set()
        for _, spec in DEPLOYED:
            all_spec_keys |= spec_keys(spec)
        text = PAPER.read_text(encoding="utf-8")
        nums = dedupe(extract(text, "latex"))
        sec_lines = sorted((i + 1, l.split("{")[1].split("}")[0]) for i, l in enumerate(text.splitlines())
                           if l.startswith("\\section*{"))
        stat_rows = {"union": [], "chain": [], "summaries": [], "extended": []}
        for n in nums:
            scales = (1.0, 100.0) if n.pct else (1.0,)
            pert = perturbations(n)
            t = trace_one(n, uv, ufid, ["(union of 8 runs)"])
            per_run = {}
            for r, (v, f, fl) in run_corpora.items():
                i, _ = corpus.match(v, n.value, n.h, scales)
                hit = np.array([corpus.match(v, float(p), n.h, scales)[0] >= 0 for p in pert])
                per_run[r] = dict(traced=i >= 0, file=(fl[f[i]] if i >= 0 else None), chance=float(hit.mean()), hits=hit)
            # chain: the number is written in run r's FINAL_SUMMARY and found in run r's outputs
            chain_runs = []
            chain_hits = np.zeros(len(pert), bool)
            for r in per_run:
                in_sum = corpus.match(sv_run[r], n.value, n.h, scales)[0] >= 0
                if in_sum and per_run[r]["traced"]:
                    chain_runs.append(r)
                sum_hits = np.array([corpus.match(sv_run[r], float(p), n.h, scales)[0] >= 0 for p in pert])
                chain_hits |= (sum_hits & per_run[r]["hits"])
            summary_runs = [r for r in per_run if corpus.match(sv_run[r], n.value, n.h, scales)[0] >= 0]
            s_i, _ = corpus.match(sv, n.value, n.h, scales)
            s_hits = np.array([corpus.match(sv, float(p), n.h, scales)[0] >= 0 for p in pert])
            x_i, _ = corpus.match(xv, n.value, n.h, scales)
            x_hits = np.array([corpus.match(xv, float(p), n.h, scales)[0] >= 0 for p in pert])
            in_runs = [r for r in per_run if per_run[r]["traced"]]
            sec = [s_ for ln, s_ in sec_lines if ln <= n.line]
            paper_rows.append(dict(
                scope="paper", run="(union)", doc=rel(PAPER), raw=n.raw, value=n.value, half_width=n.h,
                cls=classify(n), pct=n.pct, approx=n.approx, line=n.line,
                section=sec[-1] if sec else "(front matter)", n_occurrences=len(n.extra["occurrences"]),
                also_in_spec=(n.key() in all_spec_keys), context=n.context[:160],
                traced_in_runs=";".join(in_runs),
                first_files=";".join(per_run[r]["file"] for r in in_runs[:3]),
                summary_runs=";".join(summary_runs), chain_trace=bool(chain_runs), chain_runs=";".join(chain_runs),
                chain_chance=float(chain_hits.mean()),
                in_summaries=s_i >= 0, summaries_chance=float(s_hits.mean()),
                traced_extended=x_i >= 0, extended_chance=float(x_hits.mean()), **t))
            stat_rows["union"].append(dict(traced=t["traced"], chance_rate=t["chance_rate"], weak=t["weak"]))
            stat_rows["chain"].append(dict(traced=bool(chain_runs), chance_rate=float(chain_hits.mean()), weak=False))
            stat_rows["summaries"].append(dict(traced=s_i >= 0, chance_rate=float(s_hits.mean()), weak=False))
            stat_rows["extended"].append(dict(traced=x_i >= 0, chance_rate=float(x_hits.mean()), weak=False))
        labels = {"union": "paper_vs_union_of_8_runs (found in any run)",
                  "chain": "paper_chain (written in run R's FINAL_SUMMARY and found in run R's outputs)",
                  "summaries": "paper_vs_numbers_written_in_FINAL_SUMMARY_files (transcription check)",
                  "extended": "paper_vs_extended_corpus (runs + audits + summaries + setup json/csv/log/txt)"}
        for key, rows_ in stat_rows.items():
            st = doc_stats(rows_, rng)
            st.update(scope=labels[key], run="(paper)", doc=rel(PAPER), corpus_files=None,
                      corpus_values=int({"union": len(uv), "chain": len(uv), "summaries": len(sv),
                                         "extended": len(xv)}[key]))
            summary.append(st)
            print(f"   paper {key}: {st['n_traced']}/{st['n_numbers']} (chance {st['mean_chance']:.2f}, "
                  f"excess {st['excess']:.2f})", flush=True)
        ns = [r for r in paper_rows if not r["also_in_spec"]]
        st2 = doc_stats(ns, rng)
        st2.update(scope="paper_vs_union_excluding_spec_numbers", run="(paper)", doc=rel(PAPER),
                   corpus_files=None, corpus_values=int(len(uv)))
        summary.append(st2)
        for c in sorted({r["cls"] for r in paper_rows}):
            st4 = doc_stats([r for r in paper_rows if r["cls"] == c], rng)
            st4.update(scope=f"paper_class:{c}", run="(paper)", doc=rel(PAPER), corpus_files=None,
                       corpus_values=int(len(uv)))
            summary.append(st4)
    # per-class stats pooled over run summaries
    for c in sorted({r["cls"] for r in all_rows}):
        st = doc_stats([r for r in all_rows if r["cls"] == c], rng)
        st.update(scope=f"run_summaries_class:{c}", run="(all runs)", doc="(all FINAL_SUMMARY files)",
                  corpus_files=None, corpus_values=None)
        summary.append(st)
    # ---------------------------------------------------------------- write
    RESULTS.mkdir(parents=True, exist_ok=True)
    cols = ["scope", "run", "doc", "n_numbers", "n_traced", "traced_share", "traced_ci_lo", "traced_ci_hi",
            "mean_chance", "chance_ci_lo", "chance_ci_hi", "excess", "excess_ci_lo", "excess_ci_hi",
            "n_weak", "n_strong_traced", "n_untraced", "corpus_files", "corpus_values"]
    with open(RESULTS / "number_trace_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    ncols = ["scope", "run", "doc", "line", "raw", "value", "half_width", "cls", "pct", "approx", "also_in_spec",
             "traced", "weak", "chance_rate", "matched_value", "first_file", "relaxed_trailing_zero",
             "n_occurrences", "context"]
    with open(RESULTS / "number_trace_numbers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ncols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    if paper_rows:
        pcols = ["line", "section", "raw", "value", "half_width", "cls", "pct", "approx", "also_in_spec", "traced",
                 "weak", "chance_rate", "traced_in_runs", "summary_runs", "chain_trace", "chain_runs", "chain_chance",
                 "in_summaries", "summaries_chance", "traced_extended", "extended_chance", "first_files",
                 "relaxed_trailing_zero", "n_occurrences", "context"]
        with open(RESULTS / "number_trace_paper_numbers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=pcols, extrasaction="ignore")
            w.writeheader()
            w.writerows(paper_rows)
    write_json(RESULTS / "number_trace.json", {"summary": summary, "k_perturb": K_PERTURB, "n_boot": N_BOOT,
                                               "weak_threshold": WEAK, "seed": SEED})
    write_json(RESULTS / "number_trace_run_config.json",
               run_config("number_trace.py", inputs, extra={
                   "k_perturb": K_PERTURB, "n_boot": N_BOOT, "weak_threshold": WEAK,
                   "helper_sha256": {f: __import__("common").sha256_file(Path(__file__).parent / f)
                                     for f in ("numtext.py", "corpus.py")},
                   "interval_methods": {
                       "traced_share": "Wilson score 95% interval; unit = distinct number in the document",
                       "mean_chance": "percentile bootstrap 95%, 2,000 resamples of numbers (with replacement)",
                       "excess": "paired percentile bootstrap 95%, 2,000 resamples of numbers"}}))
    if args.strict:
        bad = [r for r in all_rows if not r["traced"] and not r["also_in_spec"]]
        print(f"--strict: {len(bad)} untraced FINAL_SUMMARY number(s) that are not spec quotes")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
