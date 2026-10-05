import json, numpy as np, pandas as pd, collections
from math import comb
from scipy.stats import hypergeom, spearmanr, mannwhitneyu
RUN="<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs"
cat=json.load(open(f"{RUN}/phase2/layer_05/feature_catalog.json"))
gn=json.load(open(f"{RUN}/phase0/layer_05/gene_names.json"))
cnt=collections.Counter(g.upper() for g in gn); cnt.pop("<SPECIAL>",None)
uni=set(cnt)
fids=np.array([c["feature_id"] for c in cat]); top={c["feature_id"]:set(g.upper() for g in c["top20_genes"]) for c in cat}
N=len(fids); RESP=[628,1334,2006,2610,2627]; K=5
chip=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",sep="\t")
chip["source"]=chip.source.str.upper(); chip["target"]=chip.target.str.upper()
tr=pd.read_csv("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv",sep="\t",header=None,names=["tf","target","mode","pmid"])
tr["tf"]=tr.tf.str.upper(); tr["target"]=tr.target.str.upper()
chipsets={t:set(g.target) for t,g in chip.groupby("source")}
G=chipsets["GATA1"]; T=set(tr[tr.tf=="GATA1"].target)
def ov_all(S): return np.array([len(top[f]&S) for f in fids])
def p_max_ge(ovs,t,k=K):
    m=(ovs>=t).sum(); return 1-comb(N-m,k)/comb(N,k)
out={}
og=ov_all(G); ot=ov_all(T)
out["gata1_chip_overlap_distribution_all_features"]={int(k):int(v) for k,v in sorted(collections.Counter(og).items())}
out["gata1_trrust_overlap_distribution_all_features"]={int(k):int(v) for k,v in sorted(collections.Counter(ot).items())}
obs=max(len(top[f]&G) for f in RESP); out["observed_max_overlap_chip"]=obs
out["exact_random_feature_null_chip"]={f">={t}":round(p_max_ge(og,t),5) for t in [2,3,4,5,6,7,8]}
out["exact_random_feature_null_trrust"]={f">={t}":round(p_max_ge(ot,t),5) for t in [1,2,3]}
out["n_features_with_chip_overlap_ge8"]=int((og>=8).sum())
out["rank_of_2610_among_all_features_by_chip_overlap"]=int((og>og[fids==2610][0]).sum()+1)
# threshold-search-aware (min-p over nested thresholds 2..5) under random-feature null
# null min-p = p(thr=min(m,5)) if m>=2 ; P(minp<=pobs) where pobs=p(>=5)
# simulate
rng=np.random.default_rng(0); thr=[2,3,4,5]; pthr={t:p_max_ge(og,t) for t in thr}
sims=np.array([og[rng.choice(N,K,replace=False)].max() for _ in range(200000)])
def minp(m): 
    ps=[pthr[t] for t in thr if m>=t]; return min(ps) if ps else 1.0
mp=np.array([minp(m) for m in sims]); out["threshold_search_adjusted_p_random_feature_null"]=float((mp<=pthr[5]+1e-12).mean())
out["simulated_P(max>=8)_200k"]=float((sims>=8).mean())
# --- gene rarity vs ChIP membership
genes=sorted(uni); c=np.array([cnt[g] for g in genes]); inG=np.array([g in G for g in genes]); union=set().union(*chipsets.values()); inU=np.array([g in union for g in genes])
bins=[0,1,2,5,10,50,200,100000]; rows=[]
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(c>lo)&(c<=hi); rows.append({"count_bin":f"({lo},{hi}]","n_genes":int(m.sum()),"frac_GATA1_chip":round(inG[m].mean(),4) if m.sum() else None,"frac_any_chip_TF":round(inU[m].mean(),4) if m.sum() else None})
out["gene_rarity_vs_chip_membership"]=rows
rho=spearmanr(c,inG); out["spearman_count_vs_GATA1chip"]=[float(rho.correlation),float(rho.pvalue)]
# feature rarity: median count of top20 genes
med=np.array([np.median([cnt.get(g,0) for g in top[f]]) if top[f] else np.nan for f in fids])
out["median_top20_gene_count_resp_features"]={int(f):float(med[fids==f][0]) for f in RESP}
out["feature_rarity_quantiles_all"]=[float(np.nanpercentile(med,q)) for q in [5,25,50,75,95]]
rho2=spearmanr(med,og,nan_policy="omit"); out["spearman_feature_rarity_vs_GATA1chip_overlap"]=[float(rho2.correlation),float(rho2.pvalue)]
# rarity-matched null for a single feature: features with median top20 count <= 2610's
m2610=med[fids==2610][0]; mask=med<=m2610
out["n_features_as_rare_as_2610"]=int(mask.sum()); out["mean_chip_overlap_rare_features"]=float(og[mask].mean()); out["mean_chip_overlap_all"]=float(og.mean())
out["P_single_feature_overlap>=8_all"]=float((og>=8).mean()); out["P_single_feature_overlap>=8_rare_matched"]=float((og[mask]>=8).mean())
# max over 5 where 1 feature rarity-matched + 4 random (approx): P = 1-(1-p_rare8)*(1-p_all8)^4
pr=float((og[mask]>=8).mean()); pa=float((og>=8).mean()); out["approx_P_max5>=8_one_rare_matched"]=1-(1-pr)*(1-pa)**4
# --- TF-swap null: fix the 5 GATA1 responding features, swap ChIP set across all TFs
rows=[]
for tf,S in chipsets.items():
    Su=S&uni
    if len(Su)<20: continue
    o=ov_all(S); obsj=max(len(top[f]&S) for f in RESP)
    ptab={t:p_max_ge(o,t) for t in range(1,21)}
    pj=ptab[obsj] if obsj>=1 else 1.0
    # min-p over thresholds 2..5 as in paper
    pp=[ptab[t] for t in [2,3,4,5] if obsj>=t]; pmin=min(pp) if pp else 1.0
    ov2610=len(top[2610]&S)
    # hypergeom p for 2610 against this set (universe = uni)
    n20=len(top[2610]); hp=hypergeom.sf(ov2610-1,len(uni),len(Su),n20)
    rows.append({"tf":tf,"n_targets":len(S),"n_in_universe":len(Su),"max_overlap_5resp":obsj,"overlap_2610":ov2610,"p_randfeat_maxge_obs":pj,"p_minover_thr2to5":pmin,"hypergeom_p_2610":hp})
sw=pd.DataFrame(rows).sort_values("p_randfeat_maxge_obs"); sw.to_csv("tf_swap_null.csv",index=False)
g=sw[sw.tf=="GATA1"].iloc[0]
out["tf_swap"]={"n_tfs_with_>=20_in_universe":len(sw),
 "GATA1_rank_by_p_randfeat_maxge_obs":int((sw.p_randfeat_maxge_obs<g.p_randfeat_maxge_obs).sum()+1),
 "GATA1_p_randfeat_maxge_obs":float(g.p_randfeat_maxge_obs),
 "frac_TFs_with_p<=GATA1":float((sw.p_randfeat_maxge_obs<=g.p_randfeat_maxge_obs).mean()),
 "GATA1_rank_by_minp_thr2to5":int((sw.p_minover_thr2to5<g.p_minover_thr2to5).sum()+1),
 "frac_TFs_with_minp<=GATA1_minp":float((sw.p_minover_thr2to5<=g.p_minover_thr2to5).mean()),
 "n_TFs_where_2610_overlap>=8":int((sw.overlap_2610>=8).sum()),
 "n_TFs_where_2610_overlap>=5":int((sw.overlap_2610>=5).sum()),
 "GATA1_rank_by_2610_hypergeom":int((sw.hypergeom_p_2610<g.hypergeom_p_2610).sum()+1),
 "frac_TFs_with_max_overlap>=5":float((sw.max_overlap_5resp>=5).mean()),
 "size_matched_150to300_in_uni":None}
sm=sw[(sw.n_in_universe>=150)&(sw.n_in_universe<=300)]
out["tf_swap"]["size_matched_150to300_in_uni"]={"n":len(sm),"frac_maxoverlap>=8":float((sm.max_overlap_5resp>=8).mean()),"GATA1_rank_by_p":int((sm.p_randfeat_maxge_obs<g.p_randfeat_maxge_obs).sum()+1),"frac_p<=GATA1":float((sm.p_randfeat_maxge_obs<=g.p_randfeat_maxge_obs).mean())}
json.dump(out,open("null1_results.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
print(sw.head(25).to_string(index=False))
