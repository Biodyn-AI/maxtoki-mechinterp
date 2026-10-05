"""v2 circuit tracing, part 3: K562 CRISPRi pseudobulk LFCs for every candidate source gene (item D2).

Same LFC as the corrected deployed metric (scripts/audit_a2_groupkfold_crispri.py:101-151):
  control  = 3,000 K562 non-targeting cells, np.random.default_rng(42).choice(..., replace=False), sorted
  per cell = log1p(counts / row_sum * 1e4)   (float32, row_sum 0 -> 1)
  LFC(source, target) = mean over cells with the source's CRISPRi guide (>= 10 K562 cells)
                        - mean over the control cells
  source gene -> guide: first perturbation category whose upper-case name equals the gene
  target gene -> column: last var column whose upper-case symbol equals the gene (dict build order)
Candidate sources = every gene in the top-10 list of any of the 120 source features
(not only the genes that end up in a prediction), so the same LFC table serves
both the deployed edges and the v2 edges.

Writes outputs/v2_circuit/lfc/:
  lfc_matrix.npy      (n_sources_with_data, n_var) float32
  sources.csv         source gene, perturbation code, n_cells, row in lfc_matrix
  var_symbols.txt     upper-case var symbols (column order of lfc_matrix)
  control_rows.npy    dataset rows of the 3,000 control cells
  check_vs_deployed_records.json   LFC reproduction check vs per_pair.parquet
  run_config.json
Resumable: per-source vectors are cached in lfc/parts/ ; use --max-minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from dataset_loader import resolve as load_ds  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v2_circuit/lfc"
PARTS = OUT / "parts"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OLD_PP = RUN / "outputs/groupkfold_crispri/per_pair.parquet"
SEED = 42


def candidate_sources():
    feats = json.loads((RUN / "outputs/v2_circuit/source_features.json").read_text())
    genes = []
    for s, fl in feats.items():
        cat = {c["feature_id"]: c for c in json.load(open(PHASE2 / f"layer_{int(s):02d}/feature_catalog.json"))}
        for f in fl:
            genes += [g.upper() for g in cat[f]["top20_genes"][:10]]
    return sorted(set(genes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    cands = candidate_sources()
    ds = load_ds("k562")
    with h5py.File(ds.h5_path, "r") as f:
        pg_cats = [s.decode() if isinstance(s, bytes) else s for s in f["obs"]["gene"]["categories"][:]]
        pg_codes = f["obs"]["gene"]["codes"][:]
        cl_codes = f["obs"]["cell_line"]["codes"][:]
        cl_cats = [(s.decode() if isinstance(s, bytes) else s).lower() for s in f["obs"]["cell_line"]["categories"][:]]
        k562 = (cl_codes == cl_cats.index("k562"))
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
        var_upper = [s.upper() for s in var_symbols]
        n_var = len(var_symbols)

        # control mean (cached)
        cm_path = OUT / "control_mean.npy"
        if cm_path.exists():
            ctrl_mean = np.load(cm_path)
            nt_idx = np.load(OUT / "control_rows.npy")
        else:
            nt_codes = {i for i, g in enumerate(pg_cats) if "non-targeting" in g.lower() or "nontargeting" in g.lower()}
            nt_mask = k562 & np.isin(pg_codes, list(nt_codes))
            nt_idx = np.where(nt_mask)[0]
            rng = np.random.default_rng(SEED)
            if len(nt_idx) > 3000:
                nt_idx = np.sort(rng.choice(nt_idx, 3000, replace=False))
            X_nt = np.empty((len(nt_idx), n_var), dtype=np.float32)
            for i in range(0, len(nt_idx), 500):
                X_nt[i:i + 500] = f["X"][nt_idx[i:i + 500], :]
            rs = X_nt.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            X_nt = np.log1p(X_nt / rs * 1e4)
            ctrl_mean = X_nt.mean(axis=0)
            del X_nt
            np.save(cm_path, ctrl_mean)
            np.save(OUT / "control_rows.npy", nt_idx)
            print(f"control mean from {len(nt_idx)} cells ({time.time() - t0:.0f}s)", flush=True)

        k562_cells = np.where(k562)[0]
        k562_codes = pg_codes[k562]
        status = "all sources done"
        for gi, g in enumerate(cands):
            part = PARTS / f"{g}.npz"
            if part.exists():
                continue
            if time.time() - t0 > args.max_minutes * 60 - 30:
                status = "stopped: time budget"
                break
            pert_code = None
            for i, cat in enumerate(pg_cats):
                if cat.upper() == g:
                    pert_code = i
                    break
            if pert_code is None:
                np.savez(part, has_data=False, pert_code=-1, n_cells=0)
                continue
            pert_cells = k562_cells[k562_codes == pert_code]
            if len(pert_cells) < 10:
                np.savez(part, has_data=False, pert_code=pert_code, n_cells=len(pert_cells))
                continue
            X_pert = np.empty((len(pert_cells), n_var), dtype=np.float32)
            for i in range(0, len(pert_cells), 300):
                X_pert[i:i + 300] = f["X"][pert_cells[i:i + 300], :]
            rs_p = X_pert.sum(axis=1, keepdims=True); rs_p[rs_p == 0] = 1.0
            X_pert = np.log1p(X_pert / rs_p * 1e4)
            lfc = X_pert.mean(axis=0) - ctrl_mean
            np.savez(part, has_data=True, pert_code=pert_code, n_cells=len(pert_cells), lfc=lfc.astype(np.float32))
            if gi % 25 == 0:
                print(f"  {gi + 1}/{len(cands)} {g} n={len(pert_cells)} ({time.time() - t0:.0f}s)", flush=True)

    done = [g for g in cands if (PARTS / f"{g}.npz").exists()]
    print(f"{status}: {len(done)}/{len(cands)} candidate sources processed")
    if len(done) < len(cands):
        return
    # assemble
    rows, mats = [], []
    for g in cands:
        z = np.load(PARTS / f"{g}.npz")
        rows.append(dict(source=g, pert_code=int(z["pert_code"]), n_cells=int(z["n_cells"]), has_data=bool(z["has_data"])))
        if bool(z["has_data"]):
            mats.append(z["lfc"])
    src = pd.DataFrame(rows)
    src["matrix_row"] = -1
    src.loc[src.has_data, "matrix_row"] = np.arange(int(src.has_data.sum()))
    lfc = np.stack(mats).astype(np.float32)
    np.save(OUT / "lfc_matrix.npy", lfc)
    src.to_csv(OUT / "sources.csv", index=False)
    (OUT / "var_symbols.txt").write_text("\n".join(var_upper))

    # reproduction check against the deployed corrected records
    pp = pd.read_parquet(OLD_PP)
    gene_to_var = {s: i for i, s in enumerate(var_upper)}
    srow = dict(zip(src.source, src.matrix_row))
    miss_src = sorted(set(pp.source) - {s for s, r in srow.items() if r >= 0})
    r_idx = pp.source.map(srow).to_numpy()
    c_idx = pp.target.map(gene_to_var).to_numpy()
    ok = (r_idx >= 0)
    rec = lfc[r_idx[ok].astype(int), c_idx[ok].astype(int)]
    diff = np.abs(rec.astype(np.float64) - pp.actual_lfc.to_numpy()[ok])
    chk = dict(n_records=int(len(pp)), n_sources_deployed=int(pp.source.nunique()),
               deployed_sources_missing_here=miss_src, n_compared=int(ok.sum()),
               max_abs_diff=float(diff.max()), n_exact=int((diff == 0).sum()),
               n_candidate_sources=len(cands), n_sources_with_ge10_cells=int(src.has_data.sum()),
               n_candidates_without_guide=int((src.pert_code < 0).sum()),
               n_candidates_with_lt10_cells=int(((src.pert_code >= 0) & ~src.has_data).sum()),
               pass_=bool(ok.all() and diff.max() == 0.0))
    H.write_json(OUT / "check_vs_deployed_records.json", chk)
    cfg = dict(script=str(Path(__file__)), script_sha256=H.sha256_file(__file__), device="cpu",
               dataset=str(ds.h5_path), seeds=dict(control_sample=SEED), n_control_cells=int(len(nt_idx)),
               control_rows_file="control_rows.npy", min_cells_per_guide=10,
               numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds_last_chunk=round(time.time() - t0, 1))
    H.write_json(OUT / "run_config.json", cfg)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
