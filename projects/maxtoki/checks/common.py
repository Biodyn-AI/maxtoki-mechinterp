"""Shared paths, spec/run lists and small helpers for the MaxToki checker package.

Nothing in this package writes outside `checks/results/`. All checks only read the
repository. See README.md for what each check does and does NOT establish.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path("<REPO_ROOT>")
PROJ = ROOT / "projects" / "maxtoki"
CHECKS = PROJ / "checks"
RESULTS = CHECKS / "results"
SEED = 20261001  # fixed seed for every random step in this package

# The eight pipeline specs that were executed on MaxToki-217M, with their run folder.
DEPLOYED = [
    ("attention-grn-217M", "pipelines/attention-grn-extraction-and-evaluation.md"),
    ("spectral-geometry-217M", "pipelines/residual-stream-spectral-geometry.md"),
    ("topology-141-217M", "pipelines/topology-geometry-141-hypotheses.md"),
    ("manifold-discovery-217M", "pipelines/manifold-discovery-extraction-compactification.md"),
    ("longevity-mechinterp-217M", "pipelines/longevity-mechinterp-donor-aware.md"),
    ("sae-atlas-217M", "pipelines/sparse-autoencoders/01-sae-atlas.md"),
    ("circuit-tracing-217M", "pipelines/sparse-autoencoders/02-causal-circuit-tracing.md"),
    ("exhaustive-mapping-217M", "pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md"),
]
# Specs that are linted for completeness but were not one of the eight research pipelines.
OTHER_SPECS = [
    "pipelines/audit-recurring-review-issues.md",   # the audit checklist itself (P1-P10)
    "pipelines/sparse-autoencoders/04-atlas-deployment.md",  # not used on MaxToki
]

PAPER = PROJ / "paper-plos-one" / "main.tex"

# Folders inside a run that are NOT scanned as machine-readable evidence:
# web-app source and its duplicated data copies, dependency trees, caches.
SKIP_DIR_NAMES = {"node_modules", "atlas", "dist", ".git", "__pycache__", ".ipynb_checkpoints"}
# Text artefact types that count as evidence. Markdown/HTML/py are excluded so that prose
# (or a hard-coded number in a script) cannot vouch for itself.
EVIDENCE_EXT = {".json", ".csv", ".tsv", ".log", ".txt", ".jsonl"}

# Deployment snapshot. Every file written by the deployed runs has a modification time on or
# before 2026-05-07; files added later by the correction work live in `v2_*` folders or
# `scripts/v2_*` and are dated 2026-10-01. The checks look only at the deployment snapshot
# (mtime before this cutoff AND no `v2_` path component), so the deployed claims are traced
# against the deployed outputs. Pass cutoff=False to include everything.
DEPLOYMENT_CUTOFF_LOCAL = "2026-05-08 00:00:00"
DEPLOYMENT_CUTOFF_EPOCH = time.mktime(time.strptime(DEPLOYMENT_CUTOFF_LOCAL, "%Y-%m-%d %H:%M:%S"))


def in_deployment_snapshot(p: Path) -> bool:
    try:
        if p.stat().st_mtime >= DEPLOYMENT_CUTOFF_EPOCH:
            return False
    except OSError:
        return False
    return not any(part.startswith("v2_") for part in p.parts)


def summary_docs(run: str) -> list[Path]:
    """Every FINAL_SUMMARY file that belongs to a run (summaries/ copy + in-run copies)."""
    docs = sorted((PROJ / "summaries").glob(f"{run}-FINAL_SUMMARY*.md"))
    docs += sorted(p for p in (PROJ / "runs" / run).rglob("FINAL_SUMMARY*.md")
                   if not p.name.startswith("._") and not any(s in p.parts for s in SKIP_DIR_NAMES))
    return [d for d in docs if not d.name.startswith("._") and in_deployment_snapshot(d)]


def evidence_files(run_dir: Path, cutoff: bool = True):
    """Machine-readable text artefacts of a run, sorted for determinism (deployment snapshot)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith("._"))
        for fn in sorted(filenames):
            if fn.startswith("._"):
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() in EVIDENCE_EXT and (not cutoff or in_deployment_snapshot(p)):
                out.append(p)
    return out


def sha256_file(p: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def run_config(tool: str, inputs: list[Path], extra: dict | None = None,
               hash_limit_bytes: int = 200_000_000) -> dict:
    """Provenance record: tool, time, python, seed, and sha256 of every input file.

    Files larger than `hash_limit_bytes` are recorded with size + mtime only (hashing
    hundreds of MB on every run would dominate the run time); this is stated in the record.
    """
    rows = []
    for p in sorted(set(Path(x) for x in inputs)):
        try:
            st = p.stat()
        except OSError:
            rows.append({"path": rel(p), "missing": True})
            continue
        row = {"path": rel(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}
        if st.st_size <= hash_limit_bytes:
            row["sha256"] = sha256_file(p)
        else:
            row["sha256"] = None
            row["note"] = f"not hashed (> {hash_limit_bytes} bytes); size+mtime recorded"
        rows.append(row)
    cfg = {
        "tool": tool,
        "tool_sha256": sha256_file(CHECKS / tool) if (CHECKS / tool).exists() else None,
        "common_sha256": sha256_file(CHECKS / "common.py"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": SEED,
        "n_inputs": len(rows),
        "inputs": rows,
    }
    if extra:
        cfg.update(extra)
    return cfg
