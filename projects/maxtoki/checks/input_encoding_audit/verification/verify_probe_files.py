# Independent probe: what X, raw/X and layers hold. Reads single rows only.
import h5py, numpy as np, json, sys
B="<DATA_ROOT>"
FILES = {
 "replogle_concat": f"{B}/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad",
 "rpe1": f"{B}/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad",
 "adamson": f"{B}/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad",
 "ts_immune": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad",
 "ts_immune_sub20k": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
 "ts_lung": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
 "ts_kidney": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
 "krasnow": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad",
}
def shape(node):
    if isinstance(node, h5py.Dataset): return tuple(node.shape), "dense"
    return tuple(int(x) for x in node.attrs["shape"]), node.attrs.get("encoding-type", b"?")
def dense_row(node, i, ncol):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[i], dtype=np.float64)
    ip = node["indptr"]; a, b = int(ip[i]), int(ip[i+1])
    out = np.zeros(ncol); out[np.asarray(node["indices"][a:b])] = np.asarray(node["data"][a:b], dtype=np.float64)
    return out
rng = np.random.default_rng(20261001)
res = {}
for name, p in FILES.items():
    with h5py.File(p, "r") as f:
        (n, g), kind = shape(f["X"])
        r = {"shape": [n, g], "X_kind": str(kind), "has_raw": "raw" in f, "layers": list(f["layers"].keys()) if "layers" in f else []}
        rows = sorted(rng.choice(n, size=3, replace=False).tolist())
        r["rows"] = rows
        per = []
        for i in rows:
            x = dense_row(f["X"], i, g); nz = x[x > 0]
            d = {"row": i, "nnz": int(nz.size), "frac_int": float(np.mean(nz == np.round(nz))), "max": float(nz.max()), "sum": float(nz.sum())}
            if nz.max() < 30:
                e = np.expm1(nz); d["sum_expm1"] = float(e.sum())
                # if X = log1p(c * s), expm1 values are integer multiples of s (s = scale/total)
                s = e.min(); q = e / s
                d["expm1_multiple_of_min_frac_int"] = float(np.mean(np.abs(q - np.round(q)) < 1e-3))
                d["implied_total_counts"] = float(1e4 / s)
            if "raw" in f:
                (rn, rg), _ = shape(f["raw/X"])
                rx = dense_row(f["raw/X"], i, rg); rnz = rx[rx > 0]
                d["raw_frac_int"] = float(np.mean(rnz == np.round(rnz))); d["raw_sum"] = float(rnz.sum()); d["raw_nnz"] = int(rnz.size)
                if rg == g:
                    for scale in (1e4, 1e6):
                        pred = np.log1p(rx / rx.sum() * scale)
                        d[f"maxabs_X_minus_log1p_raw_x{int(scale)}"] = float(np.max(np.abs(pred - x)))
            for L in r["layers"]:
                node = f["layers"][L]
                try:
                    lx = dense_row(node, i, g)
                except Exception as ex:
                    continue
                lnz = lx[lx != 0]
                if lnz.size == 0: continue
                d[f"layer_{L}_frac_int"] = float(np.mean(lnz == np.round(lnz))); d[f"layer_{L}_sum"] = float(lnz.sum())
                if L.lower().startswith("decontx"):
                    pred = np.log1p(lx / lx.sum() * 1e4)
                    d[f"maxabs_X_minus_log1p_{L}_x1e4"] = float(np.max(np.abs(pred - x)))
            per.append(d)
        r["per_row"] = per
        if "raw" in f:
            def names(grp):
                k = grp.attrs.get("_index", "_index"); k = k.decode() if isinstance(k, bytes) else k
                return [s.decode() if isinstance(s, bytes) else s for s in grp[k][:]]
            try:
                r["raw_var_equals_var"] = names(f["raw/var"]) == names(f["var"])
            except Exception as ex:
                r["raw_var_equals_var"] = repr(ex)
            r["raw_shape"] = list(shape(f["raw/X"])[0])
    res[name] = r
    print(name, json.dumps(r, default=str)[:1500], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
