"""Shared helpers for the D11/D12 deployment-facts ledger.

Read-only with respect to the project: every function here only reads files.
All times are local machine time (Europe, CEST = UTC+2 in April-May 2026).
"""
import datetime
import hashlib
import json
import os

MT = "<REPO_ROOT>/projects/maxtoki"
REPO = "<REPO_ROOT>"
HERE = os.path.join(MT, "checks", "d12_deployment_facts")
OUT = os.path.join(HERE, "out")
INP = os.path.join(HERE, "inputs")
os.makedirs(OUT, exist_ok=True)

PIPELINES = [
    "spectral-geometry-217M",
    "attention-grn-217M",
    "topology-141-217M",
    "sae-atlas-217M",
    "circuit-tracing-217M",
    "exhaustive-mapping-217M",
    "manifold-discovery-217M",
    "longevity-mechinterp-217M",
]

# Files written after this moment are revision-era files (e.g. new v2_* scripts
# other agents add in 2026-10). They are not part of the April-May deployment.
CUTOFF = datetime.datetime(2026, 5, 8, 6, 0, 0).timestamp()
SKIP_DIRS = {"node_modules", ".git", "__pycache__"}


def iso(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def day(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")


def sha256_file(path, bufsize=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path):
    data = open(path, "rb").read()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def eff_time(st):
    """Effective write time of a file in this tree.

    Normal files: birth <= mtime, so the last write is mtime.
    Files copied in with an old preserved mtime (only the sae-atlas web-app
    folder) have birth > mtime; for those the copy-in time (birth) is used.
    """
    b = getattr(st, "st_birthtime", st.st_mtime)
    if b > st.st_mtime + 60:
        return b, True
    return st.st_mtime, False


def walk(top):
    for dp, dn, fn in os.walk(top):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.startswith("._"):
                continue
            yield os.path.join(dp, f)


def union(intervals, gap=0.0):
    """Merge [a, b] intervals; intervals closer than `gap` seconds are merged."""
    iv = sorted([list(x) for x in intervals])
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total_hours(iv):
    return sum(b - a for a, b in iv) / 3600.0


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return p
