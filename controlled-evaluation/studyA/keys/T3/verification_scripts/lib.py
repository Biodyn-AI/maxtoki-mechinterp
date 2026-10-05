import numpy as np, pandas as pd
from scipy.stats import hypergeom
from scipy.special import comb
P="<EVAL_ROOT>/studyA/tasks/T3-paper/data/"
gu=pd.read_csv(P+"gene_universe.tsv",sep="\t")
G=list(gu.gene); gidx={g:i for i,g in enumerate(G)}; NG=len(G)
det=gu.detection_count.values; L=gu.gene_length_bp.values
t20=pd.read_csv(P+"feature_top20.tsv",sep="\t")
t20=t20[t20.gene!="<SPECIAL>"].sort_values(["feature_id","rank"])
NF=4928
lists=[[] for _ in range(NF)]
for f,g in zip(t20.feature_id.values,t20.gene.values): lists[f].append(gidx[g])
lists=[np.array(l,dtype=int) for l in lists]
do=pd.read_csv(P+"targets_dorothea.tsv",sep="\t"); tr=pd.read_csv(P+"targets_trrust.tsv",sep="\t")
def sets_from(df):
    out={}
    for tf,sub in df.groupby("tf"):
        v=np.zeros(NG,bool); v[[gidx[g] for g in sub.target]]=True; out[tf]=v
    return out
CHIP=sets_from(do[do.chip_flag]); TRR=sets_from(tr)
def overlaps(setv): return np.array([setv[l].sum() for l in lists])
EDGES=[1,2,5,10,20,50,100,200,500]
def count_bins(edges=EDGES, last_closed=True):
    # bins [e_i, e_{i+1}); last bin includes 500
    b=np.searchsorted(np.array(edges[1:-1]),det,side="right")
    return b
def len_tert():
    lc=np.where(np.isnan(L),np.nan,L)
    t=np.full(NG,1)
    t[L<21375]=0; t[L>=56220]=2; t[np.isnan(L)]=1
    return t
def dist_feature(lst,setv,bins):
    # distribution of overlap when each gene swapped for random gene in same bin w/o replacement within bin
    d=np.array([1.0])
    ub=np.unique(bins[lst])
    for b in ub:
        n=(bins[lst]==b).sum(); Nb=(bins==b).sum(); Tb=(setv&(bins==b)).sum()
        k=np.arange(0,n+1); pk=hypergeom.pmf(k,Nb,Tb,n)
        d=np.convolve(d,pk)
    return d
def p_max_ge(feats,setv,m,bins):
    if m<=0: return 1.0
    prod=1.0
    for f in feats:
        d=dist_feature(lists[f],setv,bins)
        prod*=d[:m].sum() if m<len(d) else 1.0
    return 1-prod
def p_rf(feats_K,ov,m):
    if m<=0: return 1.0
    K=feats_K; n=(ov>=m).sum(); N=NF
    return 1-comb(N-n,K,exact=True)/comb(N,K,exact=True)
def Tcap(M): return min(M,5) if M>=2 else 0
