import h5py, numpy as np, json, sys
B="<DATA_ROOT>"
FILES = {
 "replogle_concat": f"{B}/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad",
 "rpe1": f"{B}/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad",
 "adamson_symbols": f"{B}/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad",
 "ts_immune": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad",
 "ts_immune_sub20k": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
 "ts_lung": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
 "ts_kidney": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
 "krasnow": f"{B}/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad",
}
def describe(node):
    if isinstance(node, h5py.Dataset):
        return {"kind":"dense","shape":list(node.shape),"dtype":str(node.dtype)}
    enc = node.attrs.get("encoding-type", b"")
    enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    sh = node.attrs.get("shape", None)
    return {"kind":enc or "group","shape":[int(x) for x in sh] if sh is not None else None,
            "dtype":str(node["data"].dtype) if "data" in node else None}
def row(node, i):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[i], dtype=np.float64)
    ip = node["indptr"]; a,b = int(ip[i]), int(ip[i+1])
    d = np.asarray(node["data"][a:b], dtype=np.float64)
    return d
def stats(vals):
    nz = vals[vals!=0]
    if nz.size==0: return {"nnz":0}
    return {"nnz":int(nz.size),"sum":float(nz.sum()),"max":float(nz.max()),"min_nz":float(nz.min()),
            "frac_int":float(np.mean(np.isclose(nz, np.round(nz)))),
            "expm1_sum":float(np.expm1(nz).sum()) if nz.max()<30 else None}
out={}
for name,p in FILES.items():
    with h5py.File(p,"r") as f:
        r={"top_keys":list(f.keys())}
        r["X"]=describe(f["X"])
        n = r["X"]["shape"][0]
        idx = np.linspace(0, n-1, 6).astype(int)[1:-1].tolist() + [0]
        r["X_rows"]={int(i):stats(row(f["X"],i)) for i in idx}
        if "raw" in f:
            r["raw_keys"]=list(f["raw"].keys())
            if "X" in f["raw"]:
                r["raw/X"]=describe(f["raw"]["X"])
                r["raw/X_rows"]={int(i):stats(row(f["raw"]["X"],i)) for i in idx}
                if "var" in f["raw"]:
                    r["raw_var_n"]= int(f["raw"]["X"].attrs["shape"][1]) if "shape" in f["raw"]["X"].attrs else None
        if "layers" in f:
            r["layers"]={}
            for k in f["layers"].keys():
                r["layers"][k]=describe(f["layers"][k])
                r["layers"][k]["rows"]={int(i):stats(row(f["layers"][k],i)) for i in idx[:2]}
        if "uns" in f:
            r["uns_keys"]=list(f["uns"].keys())[:30]
    out[name]=r
    print(name, json.dumps(r, indent=1)[:3000], flush=True)
json.dump(out, open(sys.argv[1],"w"), indent=1)
