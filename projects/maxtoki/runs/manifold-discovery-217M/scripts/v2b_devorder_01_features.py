"""v2b step 1: build every anchor representation that is scored by the run's gates.

Usage: python v2b_devorder_01_features.py <part>
  part = maxtoki | lookup | tokenbag | hvg | manifest

Representations (one row per anchor; all standardised with the INTERNAL panel's mean and SD, so a head fitted on
the internal panel can be applied unchanged to the other panels):
  maxtoki         the deployed 2,464-number pooled-drift feature from MaxToki hidden-state centroids
                  (phase5_let_anchor.build_pooled_drift, internal mean/SD as phase7/phase8).
  maxtoki_pca64   PCA-64 of maxtoki (fitted on internal anchors) - same model, same size as the baselines.
  lookup_ct_s<k>  cell-type-label lookup: one random 64-number Gaussian code per Tabula Sapiens cell_type string
                  + 0.5 x Gaussian noise per anchor. Knows the label and nothing else. Seeds k = 0..4.
  lookup_cls_s<k> the same, but the label is (cell_type, hema_stage), the finest label the anchors were built
                  from (stage comes from free_annotation > cell_type). Seeds k = 0..2.
  tokenbag_pca64  model-free: for each cell, every gene token at rank position r of n gets weight 1 - r/n
                  (the MaxToki tokenizer's own rank order); anchor mean; log1p; genes with SD > 0 on internal;
                  standardise; PCA-64 fitted on internal anchors. Same cells as the MaxToki centroids.
  hvg_pca64       model-free: raw UMI/read counts (raw/X of the Tabula Sapiens h5ad), counts per 10k + log1p per
                  cell, anchor mean, 2,000 highly variable genes (Seurat-v1 normalised dispersion, 20 mean bins,
                  computed on internal cells only), standardise, PCA-64 fitted on internal anchors.

Cells: internal and lung_nonhema use all tokenised cells; external and zero-shot use exactly the subsample that
phase1bc_hidden_states_and_centroids.py averaged (rng default_rng(42), same loop), checked against
anchor_meta n_cells_centroided.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time, math
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from v2b_devorder_common import *  # noqa

PH1 = RUN / "outputs/phase1"
DATA_RAW = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw")
H5 = {"internal": DATA_RAW / "tabula_sapiens_immune.h5ad", "external": DATA_RAW / "tabula_sapiens_immune.h5ad",
      "zeroshot": DATA_RAW / "tabula_sapiens_immune.h5ad", "lung_nonhema": DATA_RAW / "tabula_sapiens_lung.h5ad"}
CAP = {"zeroshot": 5000, "external": 12000}           # phase1bc_hidden_states_and_centroids.py:50-54


def standardise_all(F: dict) -> dict:
    mu = F["internal"].mean(0); sd = F["internal"].std(0) + 1e-6
    return {p: ((F[p] - mu) / sd).astype(np.float32) for p in F}


def save(rep, F):
    for p, x in F.items():
        np.save(FEAT / f"{rep}__{p}.npy", x.astype(np.float32))
    print(rep, {p: x.shape for p, x in F.items()}, flush=True)


def pca_rep(Fstd: dict, k=64) -> dict:
    pca = PCA(n_components=k, random_state=SEED).fit(Fstd["internal"])
    S = {p: pca.transform(x) for p, x in Fstd.items()}
    return standardise_all(S), [float(v) for v in pca.explained_variance_ratio_]


def selected_cells(panel):
    """Rows of cells_<panel>.npz / _obs.csv that went into the centroids (phase1bc lines 124-144)."""
    cells = pd.read_csv(PH1 / f"cells_{panel}_obs.csv")
    anchors = pd.read_csv(PH1 / f"anchors_{panel}.csv")
    n_full = len(cells)
    cap = CAP.get(panel, n_full)
    if n_full > cap:
        rng = np.random.default_rng(42)
        tgt = max(3, int(np.ceil(cap / max(1, len(anchors)))))
        keep = []
        a_col = cells["anchor_id"].astype(str).to_numpy()
        for aid in anchors["anchor_id"].astype(str):
            idx = np.where(a_col == aid)[0]
            if len(idx) > tgt:
                idx = rng.choice(idx, size=tgt, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
    else:
        keep = np.arange(n_full)
    sub = cells.iloc[keep].reset_index(drop=True)
    m = meta(panel)
    cnt = sub.groupby("anchor_id").size()
    chk = m.set_index("anchor_id")["n_cells_centroided"]
    assert (cnt.reindex(chk.index).fillna(0).astype(int) == chk.astype(int)).all(), f"cell subsample mismatch {panel}"
    return keep, sub


def part_maxtoki():
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l, part = op["A_early"], op["A_mid"], op["A_late"], op_idx["block_partition"]
    raw = {p: P5.build_pooled_drift(np.load(ART / f"anchors/centroids_{p}.npy"), A_e, A_m, A_l, part) for p in PANELS}
    mu = raw["internal"].mean(0); sd = raw["internal"].std(0) + 1e-6       # phase7 lines 58-59
    F = {p: ((raw[p] - mu) / sd).astype(np.float32) for p in PANELS}
    save("maxtoki", F)
    S, evr = pca_rep(F)
    save("maxtoki_pca64", S)
    return {"maxtoki_pca64_explained_variance_ratio_sum": float(np.sum(evr))}


def part_lookup():
    M = {p: meta(p) for p in PANELS}
    info = {}
    for kind, seeds, tag in (("ct", range(5), 7001), ("cls", range(3), 7002)):
        def lab(p):
            m = M[p]
            if kind == "ct" or p == "lung_nonhema":
                return m["cell_type"].astype(str).to_numpy()
            return (m["cell_type"].astype(str) + "|" + m["hema_stage"].astype(str)).to_numpy()
        labels = sorted(set(np.concatenate([lab(p) for p in PANELS])))
        for k in seeds:
            rng = np.random.default_rng([tag, k])
            code = {c: rng.standard_normal(64) for c in labels}
            F = {}
            for pi, p in enumerate(PANELS):
                nrng = np.random.default_rng([tag, k, pi])
                L = lab(p)
                F[p] = np.stack([code[c] for c in L]) + 0.5 * nrng.standard_normal((len(L), 64))
            save(f"lookup_{kind}_s{k}", standardise_all(F))
        info[f"lookup_{kind}_n_labels"] = len(labels)
    return info


def part_tokenbag():
    V = 0
    for p in PANELS:
        V = max(V, int(np.load(PH1 / f"cells_{p}.npz")["token_ids"].max()) + 1)
    bags = {}
    for p in PANELS:
        t0 = time.time()
        keep, sub = selected_cells(p)
        z = np.load(PH1 / f"cells_{p}.npz")
        tok = z["token_ids"]; L = z["seq_lens"]
        m = meta(p); idx_of = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        bag = np.zeros((len(m), V)); cnt = np.zeros(len(m))
        a_col = sub["anchor_id"].astype(str).to_numpy()
        for r, c in enumerate(keep):
            l = int(L[c]); ai = idx_of[a_col[r]]; cnt[ai] += 1
            if l < 3:
                continue
            g = tok[c, 1:l - 1]; n = len(g)
            np.add.at(bag[ai], g, 1.0 - np.arange(n) / max(n, 1))
        bags[p] = np.log1p(bag / cnt[:, None])
        del tok, z
        print(p, "tokenbag cells", len(keep), f"{time.time()-t0:.0f}s", flush=True)
    ok = bags["internal"].std(0) > 0
    F = standardise_all({p: b[:, ok] for p, b in bags.items()})
    S, evr = pca_rep(F)
    save("tokenbag_pca64", S)
    return {"tokenbag_vocab": V, "tokenbag_genes_kept": int(ok.sum()), "tokenbag_pca64_evr_sum": float(np.sum(evr))}


def read_rows(h5path, positions):
    """Yield (position, gene indices, counts) from raw/X (CSR) for the given cell positions."""
    import h5py
    order = np.argsort(positions)
    with h5py.File(h5path, "r", rdcc_nbytes=64 << 20) as f:
        X = f["raw/X"]; ip = X["indptr"]; dat = X["data"]; ind = X["indices"]
        for o in order:
            pos = int(positions[o]); a, b = int(ip[pos]), int(ip[pos + 1])
            yield o, ind[a:b], dat[a:b]


def part_hvg():
    import h5py
    with h5py.File(H5["internal"], "r") as f:
        G = int(f["raw/X"].attrs["shape"][1])
        genes = f["raw/var/_index"][:].astype(str)
    with h5py.File(H5["lung_nonhema"], "r") as f:
        assert (f["raw/var/_index"][:].astype(str) == genes).all()
    means = {}; s1 = np.zeros(G); s2 = np.zeros(G); n_int = 0
    lib = {}
    for p in PANELS:
        t0 = time.time()
        keep, sub = selected_cells(p)
        pos = sub["cell_idx"].to_numpy().astype(np.int64)
        m = meta(p); idx_of = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        a_of = np.array([idx_of[a] for a in sub["anchor_id"].astype(str)])
        acc = np.zeros((len(m), G), dtype=np.float64); cnt = np.zeros(len(m))
        libs = []
        for o, gi, x in read_rows(H5[p], pos):
            x = x.astype(np.float64); tot = x.sum(); libs.append(tot)
            cp = x / max(tot, 1.0) * 1e4
            acc[a_of[o], gi] += np.log1p(cp); cnt[a_of[o]] += 1
            if p == "internal":
                s1[gi] += cp; s2[gi] += cp * cp; n_int += 1
        means[p] = acc / cnt[:, None]
        lib[p] = {"median_counts_per_cell": float(np.median(libs)), "n_cells": int(len(libs))}
        print(p, "hvg cells", len(pos), f"{time.time()-t0:.0f}s", flush=True)
    mean = s1 / n_int; var = (s2 - n_int * mean ** 2) / (n_int - 1)
    expr = mean > 0
    disp = np.full(G, np.nan); disp[expr] = var[expr] / mean[expr]
    ldisp = np.log(np.where(disp > 0, disp, np.nan)); lmean = np.log1p(mean)
    bins = pd.cut(lmean, 20)
    df = pd.DataFrame({"ld": ldisp, "bin": bins})
    st = df.groupby("bin", observed=False)["ld"].agg(["mean", "std"])
    nd = (df["ld"] - st.loc[df["bin"], "mean"].to_numpy()) / st.loc[df["bin"], "std"].to_numpy()
    nd = nd.to_numpy(); nd[~np.isfinite(nd)] = -np.inf
    hvg = np.argsort(-nd)[:2000]
    np.save(FEAT / "hvg_gene_index.npy", hvg)
    F = standardise_all({p: x[:, hvg] for p, x in means.items()})
    S, evr = pca_rep(F)
    save("hvg_pca64", S)
    return {"hvg_n_genes_total": G, "hvg_selected": 2000, "hvg_first10": genes[hvg[:10]].tolist(),
            "hvg_pca64_evr_sum": float(np.sum(evr)), "library_size": lib, "n_internal_cells_for_hvg": n_int}


def part_manifest():
    files = sorted(f for f in FEAT.glob("*.npy") if not f.name.startswith("._"))
    man = {f.name: {"sha256": sha256(f), "shape": list(np.load(f, mmap_mode="r").shape)} for f in files}
    (OUT / "features_manifest.json").write_text(json.dumps(man, indent=1))
    print(len(man), "feature files")


if __name__ == "__main__":
    part = sys.argv[1]
    t0 = time.time()
    info = {"maxtoki": part_maxtoki, "lookup": part_lookup, "tokenbag": part_tokenbag, "hvg": part_hvg,
            "manifest": part_manifest}[part]()
    if part != "manifest":
        inputs = [ART / f"anchors/anchor_meta_{p}.csv" for p in PANELS]
        if part == "maxtoki":
            inputs += [ART / f"anchors/centroids_{p}.npy" for p in PANELS] + [
                ART / "operators/pooled_drift_components.npz", ART / "operators/operator_index.json"]
        if part in ("tokenbag", "hvg"):
            inputs += [PH1 / f"cells_{p}_obs.csv" for p in PANELS] + [PH1 / f"anchors_{p}.csv" for p in PANELS]
        if part == "tokenbag":
            inputs += [PH1 / f"cells_{p}.npz" for p in PANELS]
        extra = {"info": info, "seconds": time.time() - t0,
                 "seeds": {"pca": SEED, "lookup_ct": "default_rng([7001,k]) codes, default_rng([7001,k,panel_index]) noise",
                           "lookup_cls": "default_rng([7002,k]) codes, default_rng([7002,k,panel_index]) noise",
                           "cell_subsample": "default_rng(42) as phase1bc"}}
        if part == "hvg":
            extra["raw_h5ad_note"] = ("raw counts read from raw/X of tabula_sapiens_immune.h5ad (19.8 GB) and "
                                      "tabula_sapiens_lung.h5ad; sha256 not computed for these two files (size); "
                                      "file sizes recorded instead")
            extra["raw_h5ad_bytes"] = {str(v): v.stat().st_size for v in set(H5.values())}
        write_run_config(f"v2b_devorder_01_features_{part}", inputs, extra)
    print(part, "done", f"{time.time()-t0:.0f}s", info if part != "manifest" else "")
