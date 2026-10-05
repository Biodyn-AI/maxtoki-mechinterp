import numpy as np, pandas as pd
from scipy.stats import rankdata
D='<EVAL_ROOT>/studyA/tasks/T1-paper/data/'
A=np.load(D+'attention_layer8_headmean.npy').astype(np.float64)
C=np.load(D+'pair_cell_counts.npy').astype(np.float64)
G=pd.read_csv(D+'genes.tsv',sep='\t'); S=pd.read_csv(D+'gene_stats_control_cells.tsv',sep='\t')
T=pd.read_csv(D+'trrust_edges_in_gene_set.tsv',sep='\t')
N=1500
mean=S['mean'].values; var=S['variance'].values; drop=S['dropout_rate'].values

def edges(min_t=3, keep_self=False, dedup=True):
    t=T[['tf_index','target_index']]
    if dedup: t=t.drop_duplicates()
    if not keep_self: t=t[t.tf_index!=t.target_index]
    cnt=t.groupby('tf_index').size()
    tfs=np.array(sorted(cnt[cnt>=min_t].index))
    M=np.zeros((len(tfs),N),bool)
    pos={tf:k for k,tf in enumerate(tfs)}
    for a,b in t.values:
        if a in pos: M[pos[a],b]=True
    return tfs,M

def auc(y,s):
    y=np.asarray(y,bool); r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)

def pairmat(tfs, score_fn):
    """returns score matrix (len(tfs), N) with TF's own column masked (nan)."""
    Sm=np.array([score_fn(tf) for tf in tfs],float)
    for k,tf in enumerate(tfs): Sm[k,tf]=np.nan
    return Sm

def pooled(M,Sm):
    m=~np.isnan(Sm); return auc(M[m],Sm[m])

def per_tf(M,Sm):
    v=[]
    for k in range(M.shape[0]):
        m=~np.isnan(Sm[k]); v.append(auc(M[k][m],Sm[k][m]))
    return np.array(v)
