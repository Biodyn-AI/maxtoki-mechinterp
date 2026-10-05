import json, numpy as np, pandas as pd, collections
RUN="<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs"
gn=json.load(open(f"{RUN}/phase0/layer_05/gene_names.json"))
uni=set(g.upper() for g in gn)-{"<SPECIAL>"}
print("gene universe (genes seen in 500 NT control cells, L5 phase0)", len(uni))
chip=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",sep="\t")
dor=pd.read_csv("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv",sep="\t")
tr=pd.read_csv("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv",sep="\t",header=None,names=["tf","target","mode","pmid"])
for d,a,b in [(chip,"source","target"),(dor,"source","target"),(tr,"tf","target")]:
    d[a]=d[a].str.upper(); d[b]=d[b].str.upper()
print("dorothea_human conf", dor.confidence.value_counts().to_dict(), "sources", dor.source.nunique())
p8=pd.read_csv(f"{RUN}/phase8_true/perturbation_response.csv")
tfs48=p8[p8.is_tf].target.str.upper().tolist(); print("n TFs in phase8_true", len(tfs48))
pc=pd.read_csv("k562_pert_counts.csv"); pc["gene"]=pc.gene.str.upper(); ncell=dict(zip(pc.gene,pc.n))
rows=[]
def sizes(d,a,tf,conf=None):
    s=d[d[a]==tf]
    if conf is not None: s=s[s.confidence.isin(conf)]
    t=set(s.target if "target" in s else [])
    return len(t), len(t&uni)
for tf in tfs48:
    r={"tf":tf,"k562_cells":ncell.get(tf,0)}
    r["trrust"],r["trrust_in_uni"]=sizes(tr,"tf",tf)
    r["dor_any"],r["dor_any_in_uni"]=sizes(dor,"source",tf)
    r["dor_AB"],r["dor_AB_in_uni"]=sizes(dor,"source",tf,["A","B"])
    r["dor_ABC"],r["dor_ABC_in_uni"]=sizes(dor,"source",tf,["A","B","C"])
    r["chip"],r["chip_in_uni"]=sizes(chip,"source",tf)
    r["n_resp_phase8t"]=int(p8[p8.target.str.upper()==tf].n_responding.iloc[0])
    rows.append(r)
df=pd.DataFrame(rows); df.to_csv("tf48_target_inventory.csv",index=False)
pd.set_option("display.width",200); print(df.to_string(index=False))
print("TFs with any DoRothEA targets:", (df.dor_any>0).sum(), " with ChIP-seq targets:", (df.chip>0).sum(), " with DoRothEA A/B:", (df.dor_AB>0).sum(), " TRRUST in-universe>=2:", (df.trrust_in_uni>=2).sum(), " chip_in_uni>=20:",(df.chip_in_uni>=20).sum())
# full K562 perturbation list testable
allp=pc[~pc.gene.str.contains("NON-TARGETING")]
trtfs=set(tr.tf); dortfs=set(dor.source); chiptfs=set(chip.source)
print("K562 perturbed genes:", len(allp), " TRRUST TFs:", allp.gene.isin(trtfs).sum(), " DoRothEA TFs (any):", allp.gene.isin(dortfs).sum(), " ChIP TFs:", allp.gene.isin(chiptfs).sum(), " union TRRUST|DoRothEA:", allp.gene.isin(trtfs|dortfs).sum())
cl=allp[allp.gene.isin(chiptfs)].copy()
cl["chip_in_uni"]=[len(set(chip[chip.source==g].target)&uni) for g in cl.gene]
cl.sort_values("gene").to_csv("k562_chip_tfs.csv",index=False)
print("ChIP TFs perturbed in K562 (gene, cells, chip_in_uni):"); print(cl.sort_values("gene").to_string(index=False))
