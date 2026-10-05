import h5py, numpy as np, pandas as pd, json
H="<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
with h5py.File(H,"r") as f:
    print(list(f["obs"].keys()))
    print("X", f["X"].shape, f["X"].dtype, type(f["X"]))
    cl=[s.decode() if isinstance(s,bytes) else s for s in f["obs"]["cell_line"]["categories"][:]]
    clc=f["obs"]["cell_line"]["codes"][:]
    pg=[s.decode() if isinstance(s,bytes) else s for s in f["obs"]["gene"]["categories"][:]]
    pgc=f["obs"]["gene"]["codes"][:]
print("cell lines", cl, np.bincount(clc))
k=clc==[c.lower() for c in cl].index("k562")
vc=pd.Series(pgc[k]).value_counts()
names=pd.Series(pg)
cnt=pd.DataFrame({"gene":names[vc.index].values,"n":vc.values})
cnt.to_csv("k562_pert_counts.csv",index=False)
print("K562 perturbations", len(cnt), "with >=20 cells", (cnt.n>=20).sum(), "with >=50", (cnt.n>=50).sum())
print(cnt[cnt.gene.str.upper().isin(["GATA1","TAL1","MYC","LMO2","KLF1","NFE2","ZFPM1","RUNX1","SPI1","MAX","YY1","E2F4","CTCF","REST","MED1","CDC5L"])])
nt=cnt[cnt.gene.str.lower().str.contains("non-targeting|nontargeting")]; print("NT categories", len(nt), nt.n.sum())
