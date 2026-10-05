import numpy as np, pandas as pd
P="<EVAL_ROOT>/studyA/tasks/T3-paper/data/"
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
do=pd.read_csv(P+"targets_dorothea.tsv",sep="\t"); tr=pd.read_csv(P+"targets_trrust.tsv",sep="\t")
chip=set(do[do.chip_flag].tf); trs=set(tr.tf)
cuts=[0.5,0.25,0.1,0.05,0.02,0.01,0.0]
g=tfs.index("GATA1")
print("GATA1 BH sig", (Q[g]<0.05).sum())
print("TFs any BH", ((Q<0.05).any(1)).sum())
for c in cuts:
    R=(Q<0.05)&(np.abs(E)>c)
    K=R.sum(1)
    print(c, "TFs responding", (K>0).sum(), "chip testable", sum(K[i]>0 for i,t in enumerate(tfs) if t in chip), "trrust testable", sum(K[i]>0 for i,t in enumerate(tfs) if t in trs))
    if c==0.5: print({tfs[i]:int(K[i]) for i in np.where(K>0)[0]}, "GATA1 feats", np.where(R[g])[0].tolist())
    if c in (0.25,): print("GATA1 K", K[g])
