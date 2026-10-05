"""Build (and cache) the set of numeric values that appear in a run's machine-readable outputs.

For each run we stream every .json/.csv/.tsv/.log/.txt/.jsonl file (markdown, html and .py are
excluded on purpose), pull out every numeric token, and keep the sorted unique absolute values
together with the id of the first file (in sorted path order) that contains each value.

Memory: values are float64 + int32 file ids; files are read in 8 MB chunks. The cache in
results/cache/ is keyed by the file list, sizes and mtimes, so a re-run is fast and a changed
input forces a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from common import RESULTS, rel

CHUNK = 8 << 20
# number not preceded by a letter/underscore/digit/dot (skips identifiers like layer_05, phase12)
TOK = re.compile(rb"(?<![A-Za-z_0-9.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TOK_THOUSANDS = re.compile(rb"(?<![A-Za-z_0-9.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")


def _parse(tokens):
    if not tokens:
        return np.empty(0)
    try:
        arr = np.array(tokens, dtype="S").astype(np.float64)
    except ValueError:
        vals = []
        for t in tokens:
            try:
                vals.append(float(t))
            except ValueError:
                pass
        arr = np.array(vals, dtype=np.float64)
    arr = np.abs(arr[np.isfinite(arr)])
    return np.unique(arr)


def file_values(path: Path) -> np.ndarray:
    parts = []
    thousands = path.suffix.lower() in (".log", ".txt")
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                buf, tail = tail, b""
            else:
                buf = tail + chunk
                cut = max(buf.rfind(b"\n"), buf.rfind(b","), buf.rfind(b" "))
                if cut <= 0:
                    tail = buf
                    continue
                buf, tail = buf[:cut + 1], buf[cut + 1:]
            if buf:
                parts.append(_parse(TOK.findall(buf)))
                if thousands:
                    parts.append(_parse([t.replace(b",", b"") for t in TOK_THOUSANDS.findall(buf)]))
            if not chunk:
                break
    if not parts:
        return np.empty(0)
    return np.unique(np.concatenate(parts))


def _merge(vals, fids):
    order = np.lexsort((fids, vals))           # sort by value, then by file id
    vals, fids = vals[order], fids[order]
    keep = np.ones(len(vals), bool)
    keep[1:] = vals[1:] != vals[:-1]           # first (lowest file id) occurrence of each value
    return vals[keep], fids[keep]


def build(name: str, files: list[Path], rebuild: bool = False, log=print):
    """Return (values, file_ids, file_list). Cached under results/cache/<name>.*"""
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta = [(rel(p), p.stat().st_size, int(p.stat().st_mtime)) for p in files]
    key = hashlib.sha256(json.dumps(meta).encode()).hexdigest()
    vpath, fpath, mpath = cache / f"{name}.values.npy", cache / f"{name}.fids.npy", cache / f"{name}.meta.json"
    if not rebuild and mpath.exists() and vpath.exists() and fpath.exists():
        m = json.loads(mpath.read_text())
        if m.get("key") == key:
            return np.load(vpath), np.load(fpath), [x[0] for x in meta]
    vals = np.empty(0)
    fids = np.empty(0, dtype=np.int32)
    pend_v, pend_f, pend_n = [], [], 0
    for i, p in enumerate(files):
        v = file_values(p)
        pend_v.append(v)
        pend_f.append(np.full(len(v), i, dtype=np.int32))
        pend_n += len(v)
        if pend_n > 5_000_000 or i == len(files) - 1:
            vals, fids = _merge(np.concatenate([vals] + pend_v), np.concatenate([fids] + pend_f))
            pend_v, pend_f, pend_n = [], [], 0
    np.save(vpath, vals)
    np.save(fpath, fids)
    mpath.write_text(json.dumps({"key": key, "n_files": len(files), "n_values": int(len(vals)),
                                 "files": meta}, indent=1))
    log(f"  corpus {name}: {len(files)} files, {len(vals):,} distinct values")
    return vals, fids, [x[0] for x in meta]


def match(vals: np.ndarray, x: float, h: float, scales=(1.0,)):
    """Index of an artefact value v with |v*s - x| <= h (inclusive, tiny float slack), else -1."""
    slack = h * 1e-9 + 1e-12 * max(1.0, abs(x))
    for s in scales:
        lo, hi = (x - h - slack) / s, (x + h + slack) / s
        i = int(np.searchsorted(vals, lo, side="left"))
        if i < len(vals) and vals[i] <= hi:
            return i, s
    return -1, None
