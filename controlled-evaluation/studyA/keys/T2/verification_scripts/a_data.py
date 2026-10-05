"""Independent data checks for the T2 package, plus a rebuild of the gene pairs.
Reads only the package data folder. Writes a small cache to the scratch folder."""
import os, sys, json, time
import numpy as np, pandas as pd

PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
res = {}

E = pd.read_csv(f"{PKG}/circuit_edges.csv")
SF = pd.read_csv(f"{PKG}/source_features.tsv", sep="\t")
FT = pd.read_csv(f"{PKG}/feature_top_genes.tsv", sep="\t")
SG = pd.read_csv(f"{PKG}/silenced_genes.tsv", sep="\t")
GP = pd.read_parquet(f"{PKG}/gene_pairs.parquet")
Z = np.load(f"{PKG}/knockdown_lfc.npz", allow_pickle=False)

res["n_edges"] = len(E)
res["edge_cols"] = list(E.columns)
res["sign_values"] = E["sign"].value_counts().to_dict()
res["frac_inhibitory"] = float((E["sign"] == "inhibitory").mean())
# sign consistent with d
res["sign_vs_d_mismatch"] = int(((E["cohens_d"] < 0) != (E["sign"] == "inhibitory")).sum())
res["min_abs_d"] = float(E["cohens_d"].abs().min())
res["min_consistency"] = float(E["consistency"].min())
res["dup_edges"] = int(E.duplicated(["src_layer", "src_feature", "tgt_layer", "tgt_feature"]).sum())
res["tgt_gt_src_layer_all"] = bool((E["tgt_layer"] > E["src_layer"]).all())
res["src_layers"] = sorted(E["src_layer"].unique().tolist())
res["n_source_features"] = len(SF)
res["source_per_layer"] = SF["layer"].value_counts().sort_index().to_dict()
src_keys = set(zip(SF["layer"], SF["feature"]))
edge_src = set(zip(E["src_layer"], E["src_feature"]))
res["edge_sources_subset_of_SF"] = edge_src <= src_keys
res["n_source_features_without_edges"] = len(src_keys - edge_src)
res["n_features_toplist"] = len(FT)
ft_keys = set(zip(FT["layer"], FT["feature"]))
edge_feats = edge_src | set(zip(E["tgt_layer"], E["tgt_feature"]))
res["n_edge_features"] = len(edge_feats)
res["edge_feats_equal_FT"] = edge_feats == ft_keys
# source feature top lists agree with feature_top_genes
ftd = {(l, f): g for l, f, g in zip(FT["layer"], FT["feature"], FT["top10_genes"])}
res["sf_top_mismatch"] = int(sum(1 for l, f, g in zip(SF["layer"], SF["feature"], SF["top10_genes"]) if (l, f) in ftd and ftd[(l, f)] != g))
ntop = FT["top10_genes"].str.split(",").map(len)
res["toplist_len_counts"] = ntop.value_counts().to_dict()
res["n_lists_with_SPECIAL"] = int(FT["top10_genes"].str.contains("<SPECIAL>").sum())
res["any_lowercase_gene"] = bool(FT["top10_genes"].str.contains("[a-z]", regex=True).any())

# npz
lfc = Z["lfc"]; sil = Z["silenced_genes"]; meas = Z["measured_genes"]; ncell = Z["n_crispri_cells"]; cm = Z["control_mean"]
res["lfc_shape"] = list(lfc.shape)
res["lfc_finite"] = bool(np.isfinite(lfc).all())
res["sil_order_matches_tsv"] = bool((sil == SG["silenced_gene"].values).all())
res["ncell_matches_tsv"] = bool((ncell == SG["n_crispri_cells"].values).all())
res["ncell_range"] = [int(ncell.min()), int(ncell.max())]
res["n_measured"] = len(meas); res["measured_unique"] = len(set(meas)) == len(meas)
res["all_silenced_measured"] = bool(np.isin(sil, meas).all())
res["control_mean_range"] = [float(cm.min()), float(cm.max())]
mi = {g: i for i, g in enumerate(meas)}
own = np.array([lfc[i, mi[g]] for i, g in enumerate(sil)])
res["own_lfc_all_negative"] = bool((own < 0).all())
res["own_lfc_median"] = float(np.median(own))
res["own_lfc_n_below_m05"] = int((own < -0.5).sum())

# gene pairs
res["n_pairs"] = len(GP)
res["gp_cols"] = list(GP.columns)
res["n_silenced_in_pairs"] = GP["silenced_gene"].nunique()
res["silenced_set_equal"] = set(GP["silenced_gene"]) == set(SG["silenced_gene"])
res["n_targets"] = GP["target_gene"].nunique()
res["targets_all_measured"] = bool(GP["target_gene"].isin(meas).all())
res["self_pairs"] = int((GP["silenced_gene"] == GP["target_gene"]).sum())
res["dup_pairs"] = int(GP.duplicated(["silenced_gene", "target_gene"]).sum())
res["sorted"] = bool((GP[["silenced_gene", "target_gene"]].apply(tuple, axis=1).is_monotonic_increasing))
frac = GP["n_inhibitory_evidence"] / GP["evidence"]
res["pred_rule_ok"] = bool(((frac > 0.5) == GP["predicted_decrease"]).all())
res["keep_rule_ok"] = bool(((GP["evidence"] >= 2) | (GP["max_abs_d"] > 2.0)).all())
res["n_ties"] = int((frac == 0.5).sum())
rows = GP["silenced_gene"].map({g: i for i, g in enumerate(sil)}).values
cols = GP["target_gene"].map(mi).values
lfc_from_npz = lfc[rows, cols].astype(np.float64)
res["lfc_match_maxabs"] = float(np.abs(lfc_from_npz - GP["lfc"].values).max())
res["lfc_zero"] = int((GP["lfc"] == 0).sum())
res["gp_lfc_dtype"] = str(GP["lfc"].dtype)

# ---------- independent rebuild of the gene pairs ----------
# gene vocabulary
lists = {k: [g for g in v.split(",")] for k, v in ftd.items()}
genes = sorted({g for v in lists.values() for g in v})
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
fidx = {k: i for i, k in enumerate(lists.keys())}
M = np.full((len(fidx), 10), -1, dtype=np.int32)
for k, i in fidx.items():
    v = [gi[g] for g in lists[k]]
    M[i, :len(v)] = v
s_i = np.array([fidx[(a, b)] for a, b in zip(E["src_layer"], E["src_feature"])])
t_i = np.array([fidx[(a, b)] for a, b in zip(E["tgt_layer"], E["tgt_feature"])])
d = E["cohens_d"].values.astype(np.float64)
inh = (E["sign"] == "inhibitory").values
parts = []
for a in range(10):
    sa = M[s_i, a]
    for b in range(10):
        tb = M[t_i, b]
        ok = (sa >= 0) & (tb >= 0) & (sa != tb)
        key = sa[ok].astype(np.int64) * G + tb[ok]
        parts.append((key, inh[ok], np.abs(d[ok]), d[ok]))
key = np.concatenate([p[0] for p in parts]); ih = np.concatenate([p[1] for p in parts])
ad = np.concatenate([p[2] for p in parts]); sd = np.concatenate([p[3] for p in parts])
del parts
res["n_triples"] = int(len(key))
order = np.argsort(key, kind="stable")
key = key[order]; ih = ih[order]; ad = ad[order]; sd = sd[order]
uk, start = np.unique(key, return_index=True)
ev = np.diff(np.append(start, len(key)))
ninh = np.add.reduceat(ih.astype(np.int64), start)
mx = np.maximum.reduceat(ad, start)
sumd = np.add.reduceat(sd, start)
keep = (ev >= 2) | (mx > 2.0)
uk, ev, ninh, mx, sumd = uk[keep], ev[keep], ninh[keep], mx[keep], sumd[keep]
sg = np.array(genes)[uk // G]; tg = np.array(genes)[uk % G]
R = pd.DataFrame({"silenced_gene": sg, "target_gene": tg, "evidence": ev, "n_inh": ninh, "max_abs_d": mx, "sum_d": sumd})
R = R[R["silenced_gene"].isin(set(sil)) & R["target_gene"].isin(set(meas))]
res["n_rebuilt"] = len(R)
J = GP.merge(R, on=["silenced_gene", "target_gene"], how="outer", indicator=True)
res["rebuild_merge"] = J["_merge"].value_counts().to_dict()
both = J[J["_merge"] == "both"]
res["rebuild_evidence_mismatch"] = int((both["evidence_x"] != both["evidence_y"]).sum())
res["rebuild_ninh_mismatch"] = int((both["n_inhibitory_evidence"] != both["n_inh"]).sum())
res["rebuild_maxd_maxdiff"] = float((both["max_abs_d_x"] - both["max_abs_d_y"]).abs().max())
# cache for later scripts: pairs with sum_d
C = GP.merge(R[["silenced_gene", "target_gene", "sum_d"]], on=["silenced_gene", "target_gene"], how="left")
C.to_parquet(f"{OUT}/pairs_with_sumd.parquet", index=False)
res["seconds"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1, default=str))
