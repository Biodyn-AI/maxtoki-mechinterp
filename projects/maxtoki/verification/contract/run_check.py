#!/usr/bin/env python3
"""Read-only run-artefact checks for projects/maxtoki (a prototype of the deterministic layer).

Check A  script-path resolution: every `*.py` path cited in a run's FINAL_SUMMARY resolves.
Check B  number traceability: every decimal number in a FINAL_SUMMARY that is NOT also in the
         pipeline spec (i.e. a MaxToki-side number, not a quoted source-paper value) appears,
         after rounding to the same number of decimals, somewhere in that run's machine-readable
         artefacts (json/csv/tsv/log/txt; markdown excluded so prose cannot vouch for itself).
Check C  pin consistency: spec §2 commit hash == repos/<name>/PINNED_COMMIT.txt == git HEAD
         (HEAD only when the clone still has .git).

A match in B is only evidence the number *could* come from an artefact (coincidences happen,
e.g. 0.50); a miss is a concrete item for a human or agent to trace. Streams files; caps size.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path("<REPO_ROOT>")
PROJ = ROOT / "projects/maxtoki"
OUT = Path(__file__).resolve().parent
MAX_FILE = 20_000_000
SKIP_DIRS = {"node_modules", "atlas", "atlas-data", "dist", ".git", "__pycache__"}
ART_EXT = {".json", ".csv", ".tsv", ".log", ".txt"}

RUNS = {
    "attention-grn-217M": ("summaries/attention-grn-217M-FINAL_SUMMARY.md", "pipelines/attention-grn-extraction-and-evaluation.md"),
    "spectral-geometry-217M": ("summaries/spectral-geometry-217M-FINAL_SUMMARY.md", "pipelines/residual-stream-spectral-geometry.md"),
    "topology-141-217M": ("summaries/topology-141-217M-FINAL_SUMMARY.md", "pipelines/topology-geometry-141-hypotheses.md"),
    "manifold-discovery-217M": ("summaries/manifold-discovery-217M-FINAL_SUMMARY.md", "pipelines/manifold-discovery-extraction-compactification.md"),
    "longevity-mechinterp-217M": ("runs/longevity-mechinterp-217M/FINAL_SUMMARY.md", "pipelines/longevity-mechinterp-donor-aware.md"),
    "sae-atlas-217M": ("summaries/sae-atlas-217M-FINAL_SUMMARY.md", "pipelines/sparse-autoencoders/01-sae-atlas.md"),
    "circuit-tracing-217M": ("summaries/circuit-tracing-217M-FINAL_SUMMARY.md", "pipelines/sparse-autoencoders/02-causal-circuit-tracing.md"),
    "exhaustive-mapping-217M": ("summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md", "pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md"),
}

DEC = re.compile(r"(?<![\w.])[-−]?\d+\.(\d{2,})(?![\d])")          # 0.382, 54.61, 1.000
INT_BIG = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\d,])")      # 52,116  1,393,850
FLOAT_ANY = re.compile(rb"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+|(?<![\d.])\d+(?![\d.])")
PY_PATH = re.compile(r"`?([A-Za-z0-9_./\-]+\.py)`?")


def summary_numbers(text: str):
    """(string as written, decimals) for decimal numbers; ints written with thousands commas."""
    text = re.sub(r"`[^`]*`", " ", text)                  # drop code spans (paths, identifiers)
    text = re.sub(r"\b20\d\d-\d\d-\d\d\b", " ", text)       # dates
    nums = set()
    for m in DEC.finditer(text):
        s = m.group(0).replace("−", "-").lstrip("-")
        nums.add((s, len(m.group(1))))
    for m in INT_BIG.finditer(text):
        nums.add((m.group(0).replace(",", ""), 0))
    return nums


def artefact_files(run_dir: Path):
    for p in run_dir.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name.startswith("._") or not p.is_file():
            continue
        if p.suffix.lower() in ART_EXT:
            try:
                if p.stat().st_size <= MAX_FILE:
                    yield p
            except OSError:
                pass


def trace_numbers(targets, run_dir: Path):
    by_dec = {}
    for s, d in targets:
        by_dec.setdefault(d, set()).add(s)
    found = set()
    n_files = 0
    for f in artefact_files(run_dir):
        n_files += 1
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4_000_000), b""):
                for tok in FLOAT_ANY.findall(chunk):
                    try:
                        v = abs(float(tok))
                    except ValueError:
                        continue
                    for d, pool in by_dec.items():
                        if d == 0:
                            c = str(int(round(v))) if v == int(v) else None
                            if c and c in pool:
                                found.add((c, 0))
                            continue
                        for cand in (f"{v:.{d}f}", f"{v * 100:.{d}f}"):
                            if cand in pool:
                                found.add((cand, d))
                # (chunk boundary may split a token: acceptable for a prototype)
    return found, n_files


def check_paths(text: str, run_dir: Path):
    missing, ok = [], 0
    for m in PY_PATH.finditer(text):
        rel = m.group(1)
        if rel.startswith("http"):
            continue
        cands = [run_dir / rel, run_dir / "scripts" / Path(rel).name, PROJ / rel, ROOT / rel,
                 run_dir / "autoloop" / rel]
        if any(c.exists() for c in cands):
            ok += 1
        else:
            missing.append(rel)
    return ok, sorted(set(missing))


def check_pins():
    rows = []
    for spec in sorted((ROOT / "pipelines").rglob("*.md")):
        if spec.name.startswith("._"):
            continue
        txt = spec.read_text(encoding="utf-8")
        m = re.search(r"^## 2\. Source(.*?)^## 3\.", txt, re.S | re.M)
        sec = m.group(1) if m else ""
        hashes = set(re.findall(r"\b[0-9a-f]{40}\b", sec))
        for h in sorted(hashes) or [None]:
            match_repo, head = None, None
            for pc in (ROOT / "repos").glob("*/PINNED_COMMIT.txt"):
                if h and h in pc.read_text(errors="ignore"):
                    match_repo = pc.parent.name
                    r = subprocess.run(["git", "-C", str(pc.parent), "rev-parse", "HEAD"],
                                       capture_output=True, text=True)
                    head = r.stdout.strip() if r.returncode == 0 else "no .git (cannot verify tree)"
            rows.append(dict(spec=str(spec.relative_to(ROOT)), source_hash=h,
                             pinned_file_repo=match_repo, git_head=head,
                             head_matches=(head == h) if head and not head.startswith("no .git") else None))
    return rows


def main():
    report = {"runs": {}, "pins": check_pins()}
    for run, (summ_rel, spec_rel) in RUNS.items():
        run_dir = PROJ / "runs" / run
        summ = (PROJ / summ_rel).read_text(encoding="utf-8")
        spec = (ROOT / spec_rel).read_text(encoding="utf-8")
        s_nums = summary_numbers(summ)
        spec_strs = {s for s, _ in summary_numbers(spec)}
        targets = {(s, d) for s, d in s_nums if s not in spec_strs}
        found, n_files = trace_numbers(targets, run_dir)
        ok, missing = check_paths(summ, run_dir)
        untraced = sorted(targets - found, key=lambda x: (x[1], x[0]))
        report["runs"][run] = dict(
            summary=summ_rel, artefact_files_scanned=n_files,
            summary_numbers=len(s_nums), also_in_spec=len(s_nums) - len(targets),
            maxtoki_numbers=len(targets), traced=len(found),
            traced_frac=round(len(found) / len(targets), 3) if targets else None,
            untraced_examples=[s for s, _ in untraced[:25]],
            py_paths_ok=ok, py_paths_missing=missing,
        )
        r = report["runs"][run]
        print(f"{run:28s} files={n_files:5d} nums={r['summary_numbers']:4d} "
              f"maxtoki-side={r['maxtoki_numbers']:4d} traced={r['traced']:4d} "
              f"({r['traced_frac']}) py_ok={ok} py_missing={len(missing)}")
    (OUT / "run_check_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nPins:")
    for p in report["pins"]:
        print(f"  {p['spec']:62s} {str(p['source_hash'])[:10]:10s} file={p['pinned_file_repo']} head_ok={p['head_matches']} ({str(p['git_head'])[:24]})")


if __name__ == "__main__":
    main()
