import json, numpy as np, pandas as pd, collections
from scipy.stats import spearmanr
RUN="<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs"
cat=json.load(open(f"{RUN}/phase2/layer_05/feature_catalog.json"))
gn=json.load(open(f"{RUN}/phase0/layer_05/gene_names.json"))
cnt=collections.Counter(g.upper() for g in gn); cnt.pop("<SPECIAL>",None)
pos=json.load(open("<REPO_ROOT>/projects/biotensor/data/genemanifold/gene_pos.json"))
L={g.upper():v["end"]-v["start"] for g,v in pos.items()}
genes=np.array(sorted(cnt)); gidx={g:i for i,g in enumerate(genes)}
c=np.array([cnt[g] for g in genes]); ln=np.array([L.get(g,np.nan) for g in genes],float)
print("universe",len(genes),"with length",np.isfinite(ln).sum())
chip=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",sep="\t")
chip["source"]=chip.source.str.upper(); chip["target"]=chip.target.str.upper()
chipsets={t:set(g.target) for t,g in chip.groupby("source")}
G=chipsets["GATA1"]; inG=np.array([g in G for g in genes])
out={}
m=np.isfinite(ln)
r=spearmanr(ln[m],inG[m]); out["spearman_genelength_vs_GATA1chip"]=[float(r.correlation),float(r.pvalue)]
qs=np.nanpercentile(ln,[0,25,50,75,90,100]); rows=[]
for lo,hi in zip(qs[:-1],qs[1:]):
    mm=m&(ln>=lo)&(ln<=hi); rows.append({"len_bin_kb":f"{lo/1e3:.0f}-{hi/1e3:.0f}","n":int(mm.sum()),"frac_GATA1_chip":round(float(inG[mm].mean()),4)})
out["genelength_quantile_bins_vs_GATA1chip"]=rows
r2=spearmanr(ln[m],c[m]); out["spearman_genelength_vs_count"]=[float(r2.correlation),float(r2.pvalue)]
top={x["feature_id"]:[g.upper() for g in x["top20_genes"]] for x in cat}
t2610=top[2610]
out["feature2610_top20"]=[{"gene":g,"count_in_500_ctrl_cells":int(cnt.get(g,0)),"length_kb":round(L.get(g,np.nan)/1e3,1) if g in L else None,"GATA1_chip":g in G} for g in t2610]
out["median_length_kb_2610_top20"]=float(np.nanmedian([L.get(g,np.nan) for g in t2610])/1e3)
out["median_length_kb_universe"]=float(np.nanmedian(ln)/1e3)
out["median_length_kb_GATA1chip_in_universe"]=float(np.nanmedian(ln[inG&m])/1e3)
# covariate-preserving gene swap: bins = joint (count quintile-ish bins) x (length tertile)
cb=np.digitize(c,[1,2,5,10,20,50,100,200,500])  # count bins
lq=np.nanpercentile(ln,[33.3,66.7]); lb=np.where(np.isfinite(ln),np.digitize(np.nan_to_num(ln,nan=np.nanmedian(ln)),lq),1)
key=cb*10+lb
bins=collections.defaultdict(list)
for i,k in enumerate(key): bins[k].append(i)
bins={k:np.array(v) for k,v in bins.items()}
rng=np.random.default_rng(1)
RESP=[628,1334,2006,2610,2627]
def swap_overlap(feat_genes, S_mask, nsim, keyarr):
    idx=[gidx[g] for g in feat_genes if g in gidx]
    ks=keyarr[idx]; res=np.zeros(nsim,int)
    # sample per gene from its bin (with replacement across genes; small chance duplicates)
    for k in np.unique(ks):
        nk=(ks==k).sum(); pool=bins_k[k]
        draws=pool[rng.integers(0,len(pool),size=(nsim,nk))]
        res+=S_mask[draws].sum(1)
    return res
for label,keyarr in [("count_only",cb),("count_x_length",key)]:
    bins_k=collections.defaultdict(list)
    for i,k in enumerate(keyarr): bins_k[k].append(i)
    bins_k={k:np.array(v) for k,v in bins_k.items()}
    NS=100000
    sims={f:swap_overlap(top[f],inG,NS,keyarr) for f in RESP}
    mx=np.max(np.stack([sims[f] for f in RESP]),0)
    out[f"gene_swap_null_{label}"]={"expected_overlap_2610":float(sims[2610].mean()),"P(2610_overlap>=8)":float((sims[2610]>=8).mean()),
       "P(max5>=8)":float((mx>=8).mean()),"P(max5>=5)":float((mx>=5).mean()),"expected_overlap_by_feature":{int(f):round(float(sims[f].mean()),3) for f in RESP}}
    # TF swap under this null for feature 2610: z per TF
    rows=[]
    for tf,S in chipsets.items():
        Sm=np.array([g in S for g in genes])
        if Sm.sum()<20: continue
        s=swap_overlap(t2610,Sm,20000,keyarr); o=sum(1 for g in t2610 if g in S)
        rows.append({"tf":tf,"n_in_uni":int(Sm.sum()),"obs_2610":o,"exp":float(s.mean()),"excess":o-float(s.mean()),"p_upper":float(((s>=o).sum()+1)/(len(s)+1))})
    d=pd.DataFrame(rows).sort_values("p_upper"); d.to_csv(f"tf_swap_2610_{label}.csv",index=False)
    gr=d[d.tf=="GATA1"].iloc[0]
    out[f"tf_swap_2610_{label}"]={"n_tfs":len(d),"GATA1_obs":int(gr.obs_2610),"GATA1_exp":round(gr.exp,2),"GATA1_p":gr.p_upper,
        "GATA1_rank_by_p":int((d.p_upper<gr.p_upper).sum()+1),"n_TFs_p<=GATA1":int((d.p_upper<=gr.p_upper).sum()),
        "GATA1_rank_by_excess":int((d.excess>gr.excess).sum()+1),"top10":d.head(10)[["tf","n_in_uni","obs_2610","exp","p_upper"]].round(4).to_dict("records")}
json.dump(out,open("null2_results.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
