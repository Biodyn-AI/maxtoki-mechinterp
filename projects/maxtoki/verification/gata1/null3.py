# exact without-replacement gene-swap null (rarity / rarity x length matched) for feature 2610 and the 5 responding features
import json, numpy as np, pandas as pd, collections
RUN="<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs"
cat=json.load(open(f"{RUN}/phase2/layer_05/feature_catalog.json"))
gn=json.load(open(f"{RUN}/phase0/layer_05/gene_names.json"))
cnt=collections.Counter(g.upper() for g in gn); cnt.pop("<SPECIAL>",None)
pos=json.load(open("<REPO_ROOT>/projects/biotensor/data/genemanifold/gene_pos.json"))
L={g.upper():v["end"]-v["start"] for g,v in pos.items()}
genes=np.array(sorted(cnt)); gidx={g:i for i,g in enumerate(genes)}
c=np.array([cnt[g] for g in genes]); ln=np.array([L.get(g,np.nan) for g in genes],float)
chip=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",sep="\t")
G=set(chip[chip.source.str.upper()=="GATA1"].target.str.upper()); inG=np.array([g in G for g in genes])
top={x["feature_id"]:[gidx[g.upper()] for g in x["top20_genes"] if g.upper() in gidx] for x in cat}
cb=np.digitize(c,[1,2,5,10,20,50,100,200,500])
lq=np.nanpercentile(ln,[33.3,66.7]); lb=np.where(np.isfinite(ln),np.digitize(np.nan_to_num(ln,nan=np.nanmedian(ln)),lq),1)
rng=np.random.default_rng(11); NS=100000; RESP=[628,1334,2006,2610,2627]
out={}
for label,key in [("count_only",cb),("count_x_length",cb*10+lb)]:
    pools={k:np.where(key==k)[0] for k in np.unique(key)}
    sims={}
    for f in RESP:
        ks=key[top[f]]; tot=np.zeros(NS,int)
        for k,nk in collections.Counter(ks).items():
            P=pools[k]
            # without replacement within feature: random keys argsort, chunked
            for s in range(0,NS,5000):
                e=min(NS,s+5000)
                if len(P)<=400:
                    idx=np.argsort(rng.random((e-s,len(P))),axis=1)[:,:nk]
                    tot[s:e]+=inG[P[idx]].sum(1)
                else:
                    tot[s:e]+=inG[P[rng.integers(0,len(P),size=(e-s,nk))]].sum(1)
        sims[f]=tot
    mx=np.max(np.stack([sims[f] for f in RESP]),0)
    out[label]={"bin_sizes_used_by_2610":{int(k):int(len(pools[k])) for k in np.unique(key[top[2610]])},
                "E_overlap_2610":round(float(sims[2610].mean()),3),"P_2610_ge8":float((sims[2610]>=8).mean()),
                "P_max5_ge8":float((mx>=8).mean()),"P_max5_ge5":float((mx>=5).mean()),
                "E_overlap_other4":{int(f):round(float(sims[f].mean()),3) for f in RESP if f!=2610}}
print(json.dumps(out,indent=1)); json.dump(out,open("null3_exact_geneswap.json","w"),indent=1)
