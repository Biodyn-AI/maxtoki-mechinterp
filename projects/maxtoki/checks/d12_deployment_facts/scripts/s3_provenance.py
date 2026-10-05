"""D12 step 3: checkpoint and dataset provenance.

- sha256 of local model weight files, compared with Hugging Face LFS ids that
  were saved to inputs/ (fetched 2026-10-01 with the public HF API).
- git-blob sha1 of small config files, compared with HF blob ids.
- size, sha256 and embedded metadata (h5ad `uns`) of every input dataset.
- which run scripts reference each dataset file.
Hashing is resumable: results are cached in out/sha256_cache.json.
Usage: python s3_provenance.py [hash-small|hash-big|report]
"""
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, OUT, INP, PIPELINES, sha256_file, git_blob_sha1, dump, iso

import h5py

B = "<DATA_ROOT>"
HF = os.path.expanduser("~/.cache/huggingface/hub")
GF_SNAP = HF + "/models--ctheodoris--Geneformer/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M"

MODELS = {
    "MaxToki-217M-HF/model.safetensors": MT + "/setup/MaxToki-217M-HF/model.safetensors",
    "MaxToki-217M-HF/config.json": MT + "/setup/MaxToki-217M-HF/config.json",
    "MaxToki-217M-HF/generation_config.json": MT + "/setup/MaxToki-217M-HF/generation_config.json",
    "MaxToki-1B-HF/model.safetensors": MT + "/setup/MaxToki-1B-HF/model.safetensors",
    "MaxToki-1B-HF/config.json": MT + "/setup/MaxToki-1B-HF/config.json",
    "MaxToki-1B-HF/generation_config.json": MT + "/setup/MaxToki-1B-HF/generation_config.json",
    "Geneformer-V2-316M/model.safetensors": GF_SNAP + "/model.safetensors",
    "Geneformer-V2-316M/config.json": GF_SNAP + "/config.json",
    "scGPT whole-human best_model.pt": B + "/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/whole-human/best_model.pt",
    "scGPT whole-human args.json": B + "/biodyn-work/single_cell_mechinterp/external/scGPT_checkpoints/whole-human/args.json",
    "scGPT derived gene embeddings (.pt)": B + "/biodyn-work/subproject_53_scgpt_gpl_replication/embeddings/scgpt_whole_human_gene_embeddings.pt",
    "MaxToki token_dictionary.json": MT + "/setup/token_dictionary.json",
    "Geneformer gene_median_dictionary_gc104M.pkl": B + "/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_median_dictionary_gc104M.pkl",
    "Geneformer gene_name_id_dict_gc104M.pkl": B + "/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl",
}

DATA = {
    "replogle_concat.h5ad": B + "/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad",
    "ReplogleWeissman2022_rpe1.h5ad": B + "/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad",
    "adamson perturb_processed_symbols.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad",
    "tabula_sapiens_immune.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad",
    "tabula_sapiens_immune_subset_20000.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
    "tabula_sapiens_lung.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
    "tabula_sapiens_kidney.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
    "krasnow_lung_smartsq2.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/raw/krasnow_lung_smartsq2.h5ad",
    "trrust_human.tsv": B + "/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv",
    "dorothea_chipseq_human.tsv": B + "/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv",
    "string_ppi_edges.json": B + "/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json",
    "reactome_gene_sets.json": B + "/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/reactome_gene_sets.json",
    "kegg_gene_sets.json": B + "/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/kegg_gene_sets.json",
    "go_bp_gene_sets.json": B + "/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json",
    "gene2go_all.pkl": B + "/biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl",
}

CACHE = os.path.join(OUT, "sha256_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def hashed(path, budget_bytes=None):
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if key in cache:
        return cache[key]["sha256"]
    t0 = time.time()
    h = sha256_file(path)
    cache[key] = dict(sha256=h, seconds=round(time.time() - t0, 1))
    json.dump(cache, open(CACHE, "w"), indent=1)
    print(f"  hashed {os.path.basename(path)} {st.st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    return h


mode = sys.argv[1] if len(sys.argv) > 1 else "report"
ALL = dict(MODELS, **DATA)
if mode in ("hash-small", "hash-big"):
    for name, p in sorted(ALL.items(), key=lambda kv: os.path.getsize(kv[1])):
        big = os.path.getsize(p) > 5e9
        if (mode == "hash-small") == (not big):
            hashed(p)
    sys.exit(0)

# ------------------------------------------------------------------ report
def hf_tree(fn):
    d = json.load(open(os.path.join(INP, fn)))
    return {e["path"]: dict(size=e.get("size"), lfs_sha256=(e.get("lfs") or {}).get("oid"), blob_sha1=e.get("oid")) for e in d}


def get_hash(p):
    st = os.stat(p)
    key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
    return cache.get(key, {}).get("sha256")


mt_old = hf_tree("hf_maxtoki_tree_21aa7b7f_MaxToki-217M-HF.json")
mt_old.update(hf_tree("hf_maxtoki_tree_21aa7b7f_MaxToki-1B-HF.json"))
mt_main = hf_tree("hf_maxtoki_tree_main_MaxToki-217M-HF.json")
mt_main.update(hf_tree("hf_maxtoki_tree_main_MaxToki-1B-HF.json"))
commits = json.load(open(os.path.join(INP, "hf_maxtoki_commits_main.json")))
gf = hf_tree("hf_geneformer_tree_05fcbeb8_V2-316M.json")
gf_rev = json.load(open(os.path.join(INP, "hf_geneformer_rev_05fcbeb8.json")))
state = hf_tree("hf_state_replogle_tree_main.json")

models = {}
for name, p in MODELS.items():
    st = os.stat(p)
    rec = dict(path=p, size=st.st_size, mtime=iso(st.st_mtime), sha256=get_hash(p))
    if name.endswith(".json"):
        rec["git_blob_sha1"] = git_blob_sha1(p)
    if name.startswith("MaxToki-"):
        o, m = mt_old.get(name, {}), mt_main.get(name, {})
        rec["hf_21aa7b7f"] = o
        rec["hf_main_2026_10_01"] = m
        if o.get("lfs_sha256"):
            rec["matches_hf_21aa7b7f"] = rec["sha256"] == o["lfs_sha256"] and st.st_size == o["size"]
        else:
            rec["matches_hf_21aa7b7f"] = rec.get("git_blob_sha1") == o.get("blob_sha1")
    if name.startswith("Geneformer-V2-316M/"):
        key = "Geneformer-V2-316M/" + os.path.basename(p)
        g = gf.get(key, {})
        rec["hf_05fcbeb8"] = g
        rec["cache_blob_name"] = os.path.basename(os.path.realpath(p))
        rec["matches_hf_05fcbeb8"] = (rec["sha256"] == g.get("lfs_sha256")) if g.get("lfs_sha256") else (rec.get("git_blob_sha1") == g.get("blob_sha1"))
    models[name] = rec

dl = min(os.stat(MODELS["MaxToki-217M-HF/model.safetensors"]).st_mtime,
         os.stat(MODELS["MaxToki-1B-HF/model.safetensors"]).st_mtime)
import datetime as _dt
def cdate(c):
    return _dt.datetime.fromisoformat(c["date"].replace("Z", "+00:00")).timestamp()
before = [c for c in commits if cdate(c) <= dl]
head_at_download = max(before, key=cdate)
maxtoki_rev = dict(
    local_download_time=iso(dl),
    hf_commit_that_was_main_at_download=dict(id=head_at_download["id"], date=head_at_download["date"], title=head_at_download["title"]),
    later_commits=[dict(id=c["id"], date=c["date"], title=c["title"]) for c in commits if cdate(c) > dl],
    all_weight_and_config_files_match_21aa7b7f=all(v.get("matches_hf_21aa7b7f") for k, v in models.items() if k.startswith("MaxToki-")),
    note="post-hoc pin: no revision was written down at download time; the download used resolve/main",
)
gf_info = dict(revision="05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28", hf_lastModified=gf_rev.get("lastModified"),
               cache_refs_main_now=open(HF + "/models--ctheodoris--Geneformer/refs/main").read().strip())

# which scripts reference which files / snapshot hashes
scripts = []
for p in PIPELINES:
    scripts += [s for s in glob.glob(f"{MT}/runs/{p}/**/*.py", recursive=True)
                if os.stat(s).st_mtime < _dt.datetime(2026, 5, 8, 6).timestamp() and "/node_modules/" not in s]
scripts += glob.glob(f"{MT}/setup/*.py")
scripts = [s for s in scripts if os.stat(s).st_mtime < _dt.datetime(2026, 5, 8, 6).timestamp()]


def refs(token):
    hits = {}
    for s in scripts:
        if "/._" in s:
            continue
        txt = open(s, errors="ignore").read()
        if token in txt:
            key = os.path.relpath(s, MT).split("/")[1] if s.startswith(MT + "/runs") else "setup"
            hits.setdefault(key, []).append(os.path.relpath(s, MT))
    return {k: sorted(v) for k, v in hits.items()}


gf_refs = refs("05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28")
scgpt_refs = refs("scgpt_whole_human_gene_embeddings.pt")
args = json.load(open(MODELS["scGPT whole-human args.json"]))

datasets = {}
for name, p in DATA.items():
    st = os.stat(p)
    rec = dict(path=p, size=st.st_size, mtime=iso(st.st_mtime), sha256=get_hash(p),
               referenced_by=refs(os.path.basename(p)))
    if p.endswith(".h5ad"):
        with h5py.File(p, "r") as f:
            X = f["X"]
            shape = list(X.attrs["shape"]) if isinstance(X, h5py.Group) else list(X.shape)
            rec["shape_cells_x_genes"] = [int(x) for x in shape]
            u = f["uns"]
            for k in ("title", "schema_version", "citation"):
                if k in u:
                    v = u[k][()]
                    rec[k] = v.decode() if isinstance(v, bytes) else str(v)
            if "citation" in rec:
                m = re.search(r"cellxgene\.cziscience\.com/([0-9a-f-]{36})\.h5ad", rec["citation"])
                c = re.search(r"collections/([0-9a-f-]{36})", rec["citation"])
                rec["cellxgene_dataset_version_id"] = m.group(1) if m else None
                rec["cellxgene_collection_id"] = c.group(1) if c else None
            if "cell_line" in f["obs"]:
                g = f["obs"]["cell_line"]
                if isinstance(g, h5py.Group):
                    cats = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]
                    import numpy as np
                    codes = g["codes"][:]
                    rec["cells_per_cell_line"] = {cats[i]: int((codes == i).sum()) for i in range(len(cats))}
    if name == "replogle_concat.h5ad":
        s = state.get("replogle_concat.h5ad", {})
        rec["hf_arcinstitute_State-Replogle-Filtered_main"] = s
        rec["matches_hf_state_replogle"] = (rec["sha256"] == s.get("lfs_sha256")) if rec["sha256"] else None
        rec["size_matches_hf_state_replogle"] = st.st_size == s.get("size")
    if name == "ReplogleWeissman2022_rpe1.h5ad":
        import hashlib
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(8 << 20), b""):
                h.update(b)
        rec["md5"] = h.hexdigest()
        zs = json.load(open(os.path.join(INP, "zenodo_search_scperturb.json")))
        z = [dict(record=x["id"], version=x["metadata"].get("version"), size=f["size"], checksum=f.get("checksum"))
             for x in zs["hits"]["hits"] for f in x.get("files", []) if f["key"] == "ReplogleWeissman2022_rpe1.h5ad"]
        rec["zenodo_scperturb_records"] = z
        rec["matches_zenodo_scperturb_md5"] = any(("md5:" + rec["md5"]) == r["checksum"] and r["size"] == st.st_size for r in z)
    datasets[name] = rec

# which named perturbation dataset each pipeline asks setup/dataset_loader for
loader_calls = {}
for sp in scripts:
    if "/._" in sp:
        continue
    t = open(sp, errors="ignore").read()
    names = set(re.findall(r"load_ds\(\s*[\"']([a-z0-9_]+)[\"']", t))
    names |= set(re.findall(r'os\.environ\.get\("DATASET", "([a-z0-9_]+)"\)', t))
    if names:
        key = os.path.relpath(sp, MT).split("/")[1] if sp.startswith(MT + "/runs") else "setup"
        loader_calls.setdefault(key, set()).update(names)
loader_calls = {k: sorted(v) for k, v in loader_calls.items()}
# attention-grn also runs DATASET=rpe1 / adamson via env var: evidenced by its output folders
loader_calls["attention-grn-217M_output_suffixes"] = sorted({d.split("_", 1)[1] for d in os.listdir(MT + "/runs/attention-grn-217M/outputs") if d.startswith("phase0_") and not d.startswith("._") and os.path.isdir(os.path.join(MT, "runs/attention-grn-217M/outputs", d))})
with h5py.File(DATA["ReplogleWeissman2022_rpe1.h5ad"], "r") as f:
    g = f["obs"]["perturbation_type"]
    rpe1_ptype = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]

res = dict(models=models, dataset_loader_names_by_pipeline=loader_calls, rpe1_obs_perturbation_type_categories=rpe1_ptype,
           hf_state_replogle_commits=[dict(id=c["id"], date=c["date"], title=c["title"]) for c in json.load(open(os.path.join(INP, "hf_state_replogle_commits_main.json")))], maxtoki_revision=maxtoki_rev, geneformer=gf_info,
           geneformer_snapshot_referenced_by=gf_refs, scgpt_embedding_file_referenced_by=scgpt_refs,
           scgpt_args_save_dir=args.get("save_dir"), datasets=datasets,
           hf_metadata_inputs=sorted(os.listdir(INP)))
print(dump("provenance.json", res))
print(json.dumps(maxtoki_rev, indent=1))
for k, v in models.items():
    print(k, v["size"], (v["sha256"] or "")[:16], v.get("matches_hf_21aa7b7f", v.get("matches_hf_05fcbeb8")))
for k, v in datasets.items():
    print(k, v["size"], (v["sha256"] or "NOHASH")[:16], v.get("shape_cells_x_genes"), v.get("cellxgene_dataset_version_id"), v.get("matches_hf_state_replogle"), list(v["referenced_by"].keys()))
