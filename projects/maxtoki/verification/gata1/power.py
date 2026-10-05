import json, numpy as np, pandas as pd, collections
RUN="<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs"
cat=json.load(open(f"{RUN}/phase2/layer_05/feature_catalog.json"))
gn=json.load(open(f"{RUN}/phase0/layer_05/gene_names.json"))
cnt=collections.Counter(g.upper() for g in gn); cnt.pop("<SPECIAL>",None)
genes=np.array(sorted(cnt)); gidx={g:i for i,g in enumerate(genes)}; NG=len(genes)
feats=[np.array([gidx[g.upper()] for g in x["top20_genes"] if g.upper() in gidx]) for x in cat]
chip=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",sep="\t")
tr=pd.read_csv("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv",sep="\t",header=None,names=["tf","target","mode","pmid"])
C=set(chip[chip.source.str.upper()=="GATA1"].target.str.upper()); T=set(tr[tr.tf.str.upper()=="GATA1"].target.str.upper())
mask={"ChIP":np.array([g in C for g in genes]),"TRRUST":np.array([g in T for g in genes])}
print({k:int(v.sum()) for k,v in mask.items()}, "overlap", int((mask["ChIP"]&mask["TRRUST"]).sum()))
crit={"TRRUST":2,"ChIP":5}   # alpha=0.05 critical values under random-feature null (P>=2 TRRUST=0.003, P>=5 ChIP=0.031, P>=4 ChIP=0.087)
rng=np.random.default_rng(7); NS=20000; K=5
def sim(truth, evaldb, k, r=1.0):
    tm=mask[truth]; em=mask[evaldb]; tpool=np.where(tm)[0]; npool=np.where(~em)[0]
    det=0
    for _ in range(NS):
        fs=rng.choice(len(feats),K,replace=False)
        mx=0
        for j,fi in enumerate(fs):
            g=feats[fi]
            if j==0 and k>0:
                g=g.copy(); kk=min(k,len(tpool),len(g))
                pos=rng.choice(len(g),kk,replace=False)
                new=rng.choice(tpool,kk,replace=False)
                if r<1.0:  # each planted true target is in eval DB only with prob r; else replaced by non-DB gene
                    keep=rng.random(kk)<r
                    new=np.where(keep,new,rng.choice(npool,kk))
                g[pos]=new
            ov=int(em[np.unique(g)].sum()); mx=max(mx,ov)
        det+= mx>=crit[evaldb]
    return det/NS
ks=[0,1,2,3,4,5,6,8,10]
res={}
for truth,ev,r in [("TRRUST","TRRUST",1.0),("ChIP","ChIP",1.0),("ChIP","ChIP",0.5),("ChIP","ChIP",0.25),("ChIP","TRRUST",1.0),("TRRUST","ChIP",1.0)]:
    key=f"truth={truth}|eval={ev}|recall={r}"
    res[key]={k:round(sim(truth,ev,k,r),4) for k in ks}
    print(key,res[key],flush=True)
json.dump({"critical_values":crit,"n_sim":NS,"K_responding":K,"in_universe_sizes":{k:int(v.sum()) for k,v in mask.items()},"power":res},open("power_results.json","w"),indent=1)
