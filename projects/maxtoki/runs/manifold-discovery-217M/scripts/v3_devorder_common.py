"""V3-7 shared paths and helpers: manifold hidden-state centroids with the CORRECT MaxToki input encoding, and
the v2b developmental-ordering analysis re-run on them.

Why v3: checks/INPUT_ENCODING_AUDIT.md found that every deployed and v2/v2b manifold run fed MaxToki genes ranked by
log1p(CP10k) / gene median (the stored Tabula Sapiens X) instead of counts / gene median. v3 re-extracts the anchor
centroids from raw/X integer counts with setup/inputs_v3.py, then re-runs the v2b_devorder analysis unchanged.

Nothing here writes outside outputs/v3_devorder/. The v2b analysis code is IMPORTED, not copied:
patch_v2b() points the module globals of scripts/v2b_devorder_common.py (ART, OUT, FEAT, HEADS) at the v3 folders
before any v2b module that uses them is imported, so the v2b gate code runs unchanged on the v3 artefacts.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
SETUP = PROJ / "setup"
RUN = PROJ / "runs/manifold-discovery-217M"
SCR = RUN / "scripts"
DEP_ART = RUN / "artifacts"                 # deployed artefacts (read only)
PH1 = RUN / "outputs/phase1"                # deployed cell lists and (wrong) tokens (read only)
V2B = RUN / "outputs/v2b_devorder"          # v2b outputs (read only)
V3 = RUN / "outputs/v3_devorder"            # everything v3 writes
V3ART = V3 / "artifacts"
V3ANCH = V3ART / "anchors"
V3OPS = V3ART / "operators"
V3FEAT = V3 / "features"
V3HEADS = V3 / "heads"
PERCELL = V3 / "percell"
CELLS = V3 / "cells"
for _d in (V3, V3ART, V3ANCH, V3OPS, V3FEAT, V3HEADS, PERCELL, CELLS):
    _d.mkdir(parents=True, exist_ok=True)
for _p in (str(SETUP), str(SCR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_DIR = SETUP / "MaxToki-217M-HF"
MAX_LEN = 4096                               # phase1a: tokenizer.model_input_size
N_STATES, HIDDEN = 12, 1232
# order of extraction: everything the v2b analysis needs first, the replaced first lung control last
PANELS_ALL = ["internal", "external", "zeroshot", "lung_nonhema", "lung_control"]
DATASET = {"internal": "ts_immune", "external": "ts_immune", "zeroshot": "ts_immune",
           "lung_control": "ts_lung", "lung_nonhema": "ts_lung"}
# phase1bc_hidden_states_and_centroids.py:50-54 (MAX_CELLS_PER_PANEL)
CAP = {"zeroshot": 5000, "lung_control": 2124, "external": 12000}


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def selected_cells(panel: str):
    """Rows of cells_<panel>_obs.csv that phase1bc averaged into the deployed centroids.
    Exact replay of phase1bc_hidden_states_and_centroids.py:121-144 (default_rng(42), same loop), checked against
    the deployed anchor_meta n_cells_centroided. Returns (keep positions into cells_<panel>_obs.csv, sub-table)."""
    cells = pd.read_csv(PH1 / f"cells_{panel}_obs.csv")
    anchors = pd.read_csv(PH1 / f"anchors_{panel}.csv")
    n_full = len(cells)
    cap = CAP.get(panel, n_full)
    if n_full > cap:
        rng = np.random.default_rng(42)
        tgt = max(3, int(np.ceil(cap / max(1, len(anchors)))))
        a_col = cells["anchor_id"].astype(str).to_numpy()
        keep = []
        for aid in anchors["anchor_id"].astype(str):
            idx = np.where(a_col == aid)[0]
            if len(idx) > tgt:
                idx = rng.choice(idx, size=tgt, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
    else:
        keep = np.arange(n_full)
    sub = cells.iloc[keep].reset_index(drop=True)
    dep = pd.read_csv(DEP_ART / f"anchors/anchor_meta_{panel}.csv")
    cnt = sub.groupby("anchor_id").size()
    chk = dep.set_index("anchor_id")["n_cells_centroided"]
    if not (cnt.reindex(chk.index).fillna(0).astype(int) == chk.astype(int)).all():
        raise RuntimeError(f"{panel}: replayed cell subsample does not match deployed n_cells_centroided")
    return keep, sub


def patch_v2b():
    """Point scripts/v2b_devorder_common.py at the v3 folders. Must run before importing v2b_devorder_01_features,
    v2b_devorder_02_pool or v2b_devorder_03_analyze (they copy these names at import)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import v2b_devorder_common as V
    if getattr(V, "_V3_PATCHED", False):
        return V
    for mod in ("v2b_devorder_01_features", "v2b_devorder_02_pool", "v2b_devorder_03_analyze"):
        if mod in sys.modules:
            raise RuntimeError(f"{mod} was imported before patch_v2b(); its copied paths would point at v2b")
    V._V2B_OUT = V.OUT
    V.ART = V3ART
    V.OUT = V3
    V.FEAT = V3FEAT
    V.HEADS = V3HEADS
    V._V3_PATCHED = True
    assert V.OUT != V._V2B_OUT and str(V.OUT).endswith("v3_devorder")
    return V


def assert_paths_v3(mod):
    """Every path-like global of an imported v2b module that points inside outputs/ or artifacts/ must be a v3 path."""
    bad = []
    for k, v in vars(mod).items():
        if isinstance(v, Path) and (("outputs" in v.parts) or ("artifacts" in v.parts)):
            if "v3_devorder" not in v.parts and v not in (PH1,):
                bad.append((k, str(v)))
    if bad:
        raise RuntimeError(f"{mod.__name__}: globals still point at non-v3 paths: {bad}")
