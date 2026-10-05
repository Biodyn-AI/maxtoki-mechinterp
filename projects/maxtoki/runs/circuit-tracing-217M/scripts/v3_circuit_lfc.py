"""v3 circuit tracing, part 3: K562 CRISPRi knockdown log-fold-changes with a SINGLE log (item V3-3).

The bug (checks/INPUT_ENCODING_AUDIT.md section 6): the deployed and v2 LFCs computed
log1p(X / rowsum(X) * 1e4) on X = log1p(CP10k), so the values were logged twice.
Here the same cells and the same steps are used, with counts in place of X:

  counts     n_ig = round(expm1(X_ig) / u_i), u_i = smallest non-zero expm1(X) of cell i
             (expm1(X) is checked to be whole-number multiples of u_i in every cell, inputs_v3.check_count_like;
             a row whose expm1(X) is already whole numbers (u_i = 1) is accepted as counts, as in
             inputs_v3.counts_accept_unit_one)
  per cell   v_ig = log1p(1e4 * n_ig / sum_g' n_ig')      (sum over the 6,546 genes in the file)  -- PRIMARY
  LFC(s, g)  = mean_{i in KD(s)} v_ig - mean_{i in CTRL} v_ig
  CTRL       = 3,000 K562 non-targeting cells, np.random.default_rng(42).choice(..., replace=False), sorted (as v2)
  KD(s)      = all K562 cells whose perturbation category (first one whose upper-case name equals s) is s;
               used only if >= 10 cells (as v2)
Second definition (sensitivity): w_ig = X_ig = log1p(CP10k_ig) with the file's own CP10k (normalised to the
cell's full-transcriptome total before the file kept 6,546 genes); LFC_file = mean_KD X - mean_CTRL X.
Reproduction: the v2 (double-log) value is recomputed from the same arrays with the v2 float32 code and
compared with outputs/v2_circuit/lfc/ (must be identical) -- this proves the cell selection is the same.

Candidate source genes = top-10 genes of the 120 v3 source features (outputs/v3_circuit/annotation) UNION the
v2 candidates (outputs/v2_circuit/lfc/sources.csv), so v2 edges can be re-scored with corrected labels.
Writes outputs/v3_circuit/lfc/: lfc_panel.npy, lfc_file.npy, lfc_v2style.npy (n_sources_with_data, 6546),
sources.csv, var_symbols.txt, control_rows.npy, checks.json, run_config.json. Resumable (parts/), --max-minutes.
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
import inputs_v3 as I  # noqa: E402
from dataset_loader import resolve as load_ds  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
OUT = RUN / "outputs/v3_circuit/lfc"
PARTS = OUT / "parts"
ANN = RUN / "outputs/v3_circuit/annotation"
V2LFC = RUN / "outputs/v2_circuit/lfc"
SEED = 42
MIN_CELLS = 10


def candidate_sources():
    feats = json.loads((ANN / "source_features.json").read_text())
    genes = []
    for s, fl in feats.items():
        cat = {c["feature_id"]: c for c in json.load(open(ANN / f"layer_{int(s):02d}/feature_catalog.json"))}
        for f in fl:
            genes += [g.upper() for g in cat[f]["top20_genes"][:10]]
    v3 = sorted(set(genes))
    v2 = sorted(pd.read_csv(V2LFC / "sources.csv").source.tolist())
    return sorted(set(v3) | set(v2)), v3, v2


def counts_block(X):
    """X: (n, n_var) float32 block of stored log1p(CP10k). Returns integer counts (float64) and row checks."""
    C = np.empty(X.shape, np.float64)
    info = dict(n_rows=X.shape[0], n_integer_multiple=0, n_unit_one=0, max_dev=0.0)
    for i in range(X.shape[0]):
        x = X[i].astype(np.float64)
        if x.max() > 30:
            raise I.EncodingError("X max > 30; not log1p values")
        c = np.expm1(x)
        chk = I.check_count_like(c)
        if chk["pass"] and chk["kind"] == "integer_multiple":
            C[i] = np.round(c / chk["unit"]); info["n_integer_multiple"] += 1
        elif chk["pass"] and chk["kind"] == "integer" and float(chk.get("unit", 0)) == 1.0:
            C[i] = np.round(c); info["n_unit_one"] += 1
        else:
            raise I.EncodingError(f"row {i} of block is not count-like: {chk}")
        info["max_dev"] = max(info["max_dev"], float(chk["max_dev"]))
    return C, info


def three_means(X):
    """Means over cells of: v2 double-log (exact v2 float32 code), panel log1p(CP10k) (primary), file X."""
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    v2 = np.log1p(X / rs * 1e4).mean(axis=0)                       # v2_circuit_lfc.py:92-94 / 125-127
    C, info = counts_block(X)
    tot = C.sum(axis=1, keepdims=True)
    panel = np.log1p(C / tot * 1e4).mean(axis=0)                     # float64
    file_ = X.astype(np.float64).mean(axis=0)
    # second derivation of the panel value from expm1(X) directly (no rounding): must agree to float noise
    E = np.expm1(X.astype(np.float64))
    panel_b = np.log1p(E / E.sum(axis=1, keepdims=True) * 1e4).mean(axis=0)
    info["max_abs_diff_panel_rounded_vs_expm1"] = float(np.abs(panel - panel_b).max())
    info["mean_panel_total_counts"] = float(tot.mean())
    info["mean_implied_full_total"] = float(np.mean(1e4 / (np.expm1(X.astype(np.float64)) / np.where(C > 0, C, np.inf)).max(axis=1)))
    return v2, panel, file_, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True); PARTS.mkdir(parents=True, exist_ok=True)
    cands, cands_v3, cands_v2 = candidate_sources()
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
        cm_path = OUT / "control_means.npz"
        if cm_path.exists():
            z = np.load(cm_path); ctrl = dict(v2=z["v2"], panel=z["panel"], file=z["file_x"])
            nt_idx = np.load(OUT / "control_rows.npy")
        else:
            nt_codes = {i for i, g in enumerate(pg_cats) if "non-targeting" in g.lower() or "nontargeting" in g.lower()}
            nt_idx = np.where(k562 & np.isin(pg_codes, list(nt_codes)))[0]
            n_nt_all = int(len(nt_idx))
            rng = np.random.default_rng(SEED)
            if len(nt_idx) > 3000:
                nt_idx = np.sort(rng.choice(nt_idx, 3000, replace=False))
            X_nt = np.empty((len(nt_idx), n_var), dtype=np.float32)
            for i in range(0, len(nt_idx), 500):
                X_nt[i:i + 500] = f["X"][nt_idx[i:i + 500], :]
            v2m, pm, fm, info = three_means(X_nt)
            del X_nt
            np.savez(cm_path, v2=v2m, panel=pm, file_x=fm)
            np.save(OUT / "control_rows.npy", nt_idx)
            H.write_json(OUT / "control_info.json", dict(n_k562_nontargeting_cells=n_nt_all, n_used=int(len(nt_idx)), **info))
            ctrl = dict(v2=v2m, panel=pm, file=fm)
            print(f"controls: {len(nt_idx)} cells ({time.time() - t0:.0f}s)", flush=True)
        k562_cells = np.where(k562)[0]
        k562_codes = pg_codes[k562]
        status = "all sources done"
        for gi, g in enumerate(cands):
            part = PARTS / f"{g}.npz"
            if part.exists():
                continue
            if time.time() - t0 > args.max_minutes * 60 - 30:
                status = "stopped: time budget"; break
            pert_code = next((i for i, cat in enumerate(pg_cats) if cat.upper() == g), None)
            if pert_code is None:
                np.savez(part, has_data=False, pert_code=-1, n_cells=0); continue
            pert_cells = k562_cells[k562_codes == pert_code]
            if len(pert_cells) < MIN_CELLS:
                np.savez(part, has_data=False, pert_code=pert_code, n_cells=len(pert_cells)); continue
            X_p = np.empty((len(pert_cells), n_var), dtype=np.float32)
            for i in range(0, len(pert_cells), 300):
                X_p[i:i + 300] = f["X"][pert_cells[i:i + 300], :]
            v2m, pm, fm, info = three_means(X_p)
            np.savez(part, has_data=True, pert_code=pert_code, n_cells=len(pert_cells),
                     lfc_v2style=(v2m - ctrl["v2"]).astype(np.float32), lfc_panel=pm - ctrl["panel"], lfc_file=fm - ctrl["file"],
                     info=json.dumps(info))
            if gi % 50 == 0:
                print(f"  {gi + 1}/{len(cands)} {g} n={len(pert_cells)} ({time.time() - t0:.0f}s)", flush=True)
    done = [g for g in cands if (PARTS / f"{g}.npz").exists()]
    print(f"{status}: {len(done)}/{len(cands)} candidate sources processed ({time.time() - t0:.0f}s)")
    if len(done) < len(cands):
        return
    # ---------------- assemble
    rows, M = [], {"v2style": [], "panel": [], "file": []}
    infos = []
    for g in cands:
        z = np.load(PARTS / f"{g}.npz")
        rows.append(dict(source=g, pert_code=int(z["pert_code"]), n_cells=int(z["n_cells"]), has_data=bool(z["has_data"]),
                         in_v3_candidates=g in cands_v3, in_v2_candidates=g in cands_v2))
        if bool(z["has_data"]):
            M["v2style"].append(z["lfc_v2style"]); M["panel"].append(z["lfc_panel"]); M["file"].append(z["lfc_file"])
            infos.append(json.loads(str(z["info"])))
    src = pd.DataFrame(rows)
    src["matrix_row"] = -1
    src.loc[src.has_data, "matrix_row"] = np.arange(int(src.has_data.sum()))
    gene_to_var = {s: i for i, s in enumerate(var_upper)}
    # on-target knockdown: LFC of the silenced gene itself (if it is a column)
    src["self_lfc_panel"] = np.nan
    P = np.stack(M["panel"])
    for i, r in src[src.has_data].iterrows():
        if r.source in gene_to_var:
            src.loc[i, "self_lfc_panel"] = P[int(r.matrix_row), gene_to_var[r.source]]
    np.save(OUT / "lfc_panel.npy", P)
    np.save(OUT / "lfc_file.npy", np.stack(M["file"]))
    np.save(OUT / "lfc_v2style.npy", np.stack(M["v2style"]).astype(np.float32))
    src.to_csv(OUT / "sources.csv", index=False)
    (OUT / "var_symbols.txt").write_text("\n".join(var_upper))
    # ---------------- checks
    chk = {}
    v2src = pd.read_csv(V2LFC / "sources.csv")
    v2mat = np.load(V2LFC / "lfc_matrix.npy")
    v2rows = dict(zip(v2src.source, v2src.matrix_row))
    myrows = dict(zip(src.source, src.matrix_row))
    V2S = np.stack(M["v2style"]).astype(np.float32)
    both = [g for g in v2src.source[v2src.has_data] if myrows.get(g, -1) >= 0]
    diffs = [float(np.abs(V2S[myrows[g]] - v2mat[v2rows[g]]).max()) for g in both]
    same_ncells = all(int(src.set_index("source").n_cells[g]) == int(v2src.set_index("source").n_cells[g]) for g in v2src.source)
    chk["v2_lfc_reproduced"] = dict(n_v2_sources_with_data=int(v2src.has_data.sum()), n_compared=len(both),
                                    max_abs_diff=max(diffs) if diffs else None, same_cell_counts=bool(same_ncells),
                                    control_rows_equal=bool(np.array_equal(np.load(OUT / "control_rows.npy"), np.load(V2LFC / "control_rows.npy"))),
                                    pass_=bool(len(both) == int(v2src.has_data.sum()) and max(diffs) == 0.0 and same_ncells))
    Fm = np.stack(M["file"])
    flat_p, flat_f, flat_v2 = P.ravel(), Fm.ravel(), V2S.ravel().astype(np.float64)
    nzm = (flat_p != 0)
    chk["agreement_over_all_source_x_gene_cells"] = dict(
        n=int(flat_p.size),
        sign_agree_panel_vs_v2style=float(np.mean(np.sign(flat_p) == np.sign(flat_v2))),
        sign_agree_panel_vs_file=float(np.mean(np.sign(flat_p) == np.sign(flat_f))),
        pearson_panel_vs_v2style=float(np.corrcoef(flat_p, flat_v2)[0, 1]),
        pearson_panel_vs_file=float(np.corrcoef(flat_p, flat_f)[0, 1]),
        n_panel_exactly_zero=int((~nzm).sum()))
    chk["count_rows"] = dict(n_sources=len(infos), rows_integer_multiple=int(sum(i["n_integer_multiple"] for i in infos)),
                             rows_unit_one=int(sum(i["n_unit_one"] for i in infos)),
                             max_dev=float(max(i["max_dev"] for i in infos)),
                             max_abs_diff_panel_rounded_vs_expm1=float(max(i["max_abs_diff_panel_rounded_vs_expm1"] for i in infos)),
                             control=json.loads((OUT / "control_info.json").read_text()))
    sl = src.self_lfc_panel.dropna()
    chk["on_target_knockdown"] = dict(n_sources_with_own_column=int(len(sl)), median_self_lfc=float(sl.median()),
                                      frac_self_lfc_below_0=float((sl < 0).mean()), frac_self_lfc_below_minus_0p5=float((sl < -0.5).mean()))
    nc = src[src.has_data].n_cells
    chk["cell_counts"] = dict(n_candidates=len(cands), n_with_ge10_cells=int(src.has_data.sum()),
                              n_without_guide=int((src.pert_code < 0).sum()),
                              n_with_lt10_cells=int(((src.pert_code >= 0) & ~src.has_data).sum()),
                              kd_cells_min=int(nc.min()), kd_cells_median=float(nc.median()), kd_cells_max=int(nc.max()),
                              kd_cells_total=int(nc.sum()), n_control_cells=int(len(nt_idx)))
    H.write_json(OUT / "checks.json", chk)
    H.write_json(OUT / "run_config.json", dict(
        script=str(Path(__file__)), script_sha256=H.sha256_file(__file__), inputs_v3_sha256=H.sha256_file(I.__file__), device="cpu",
        dataset=I.file_fingerprint(ds.h5_path), seeds=dict(control_sample=SEED), n_control_cells=int(len(nt_idx)),
        min_cells_per_guide=MIN_CELLS,
        formula_primary="LFC(s,g) = mean_{KD(s)} log1p(1e4*n_ig/sum_g' n_ig') - mean_{CTRL} log1p(1e4*n_ig/sum_g' n_ig'), "
                        "n = round(expm1(X)/u) integer counts (sum over the 6,546 file genes)",
        formula_sensitivity="LFC_file(s,g) = mean_{KD(s)} X_ig - mean_{CTRL} X_ig, X = log1p(CP10k) of the file (full-library total)",
        formula_v2_double_log="mean log1p(X / rowsum(X) * 1e4) (float32), as v2_circuit_lfc.py -- reproduced, not used for v3 labels",
        input_encoding_check="every count row checked with inputs_v3.check_count_like (expm1(X) whole-number multiples of the cell unit)",
        numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds_last_chunk=round(time.time() - t0, 1)))
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
