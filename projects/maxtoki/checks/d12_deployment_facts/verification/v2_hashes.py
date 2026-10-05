"""Independent re-hash of the checkpoint and dataset files named in the D12
ledger, compared with (a) the original agent's cached hashes in
out/provenance.json and (b) public metadata that this verifier fetched again on
2026-10-01 into verification/inputs/ (HF API trees at four MaxToki revisions,
Geneformer 05fcbeb8 tree, arcinstitute/State-Replogle-Filtered tree, Zenodo
record 13350497).

Question checked beyond the original: does the hash match single out MaxToki
revision 21aa7b7f, or are the same bytes present at other revisions too?
Output: verification/out/v2_hashes.json
"""
import hashlib
import json
import os
import time

MT = "<REPO_ROOT>/projects/maxtoki"
V = os.path.join(MT, "checks/d12_deployment_facts/verification")
INP = os.path.join(V, "inputs")
B = "<DATA_ROOT>"
HF = os.path.expanduser("~/.cache/huggingface/hub")

FILES = {
    "MaxToki-217M-HF/model.safetensors": MT + "/setup/MaxToki-217M-HF/model.safetensors",
    "MaxToki-1B-HF/model.safetensors": MT + "/setup/MaxToki-1B-HF/model.safetensors",
    "MaxToki-217M-HF/config.json": MT + "/setup/MaxToki-217M-HF/config.json",
    "MaxToki-1B-HF/config.json": MT + "/setup/MaxToki-1B-HF/config.json",
    "MaxToki-217M-HF/generation_config.json": MT + "/setup/MaxToki-217M-HF/generation_config.json",
    "MaxToki-1B-HF/generation_config.json": MT + "/setup/MaxToki-1B-HF/generation_config.json",
    "Geneformer-V2-316M/model.safetensors": HF + "/models--ctheodoris--Geneformer/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M/model.safetensors",
    "replogle_concat.h5ad": B + "/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad",
    "ReplogleWeissman2022_rpe1.h5ad": B + "/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad",
    "adamson perturb_processed_symbols.h5ad": B + "/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad",
}


def digests(path, want_md5=False):
    h, m = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
            if want_md5:
                m.update(b)
    return h.hexdigest(), (m.hexdigest() if want_md5 else None)


def blob_sha1(path):
    d = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()


prov = json.load(open(os.path.join(MT, "checks/d12_deployment_facts/out/provenance.json")))
orig = {**{k: v.get("sha256") for k, v in prov["models"].items()},
        **{k: v.get("sha256") for k, v in prov["datasets"].items()}}

res = {}
for name, p in FILES.items():
    t0 = time.time()
    sha, md5 = digests(p, want_md5=name.startswith("ReplogleWeissman"))
    rec = dict(path=p, size=os.path.getsize(p), sha256=sha, seconds=round(time.time() - t0, 1),
               equals_original_agent_hash=(sha == orig.get(name)))
    if md5:
        rec["md5"] = md5
    if name.endswith(".json"):
        rec["git_blob_sha1"] = blob_sha1(p)
    res[name] = rec
    print(name, rec["size"], sha[:16], rec["equals_original_agent_hash"], rec["seconds"], "s", flush=True)

# which MaxToki revisions carry these exact bytes?
revs = {}
for fn in sorted(os.listdir(INP)):
    if fn.startswith("hf_maxtoki_tree_") and not fn.startswith("._"):
        rev, sub = fn[len("hf_maxtoki_tree_"):-5].split("_", 1)
        d = json.load(open(os.path.join(INP, fn)))
        if not isinstance(d, list):
            revs.setdefault(rev, {})[sub] = "folder absent at this revision"
            continue
        ok = {}
        for e in d:
            if e["path"] in res:
                r = res[e["path"]]
                ok[e["path"]] = (r["sha256"] == (e.get("lfs") or {}).get("oid")) if e.get("lfs") else (r.get("git_blob_sha1") == e["oid"])
        revs.setdefault(rev, {})[sub] = ok
gf = {e["path"]: e for e in json.load(open(os.path.join(INP, "hf_geneformer_tree_05fcbeb8_V2-316M.json")))}
st = {e["path"]: e for e in json.load(open(os.path.join(INP, "hf_state_replogle_tree_main.json")))}
z = json.load(open(os.path.join(INP, "zenodo_record_13350497.json")))
zf = [f for f in z.get("files", []) if f.get("key") == "ReplogleWeissman2022_rpe1.h5ad"]
out = dict(
    files=res,
    maxtoki_revisions_with_identical_files=revs,
    geneformer_05fcbeb8_match=res["Geneformer-V2-316M/model.safetensors"]["sha256"] == gf["Geneformer-V2-316M/model.safetensors"]["lfs"]["oid"],
    replogle_concat_matches_state_replogle_main=res["replogle_concat.h5ad"]["sha256"] == st["replogle_concat.h5ad"]["lfs"]["oid"],
    rpe1_zenodo_13350497=dict(checksum=zf[0]["checksum"] if zf else None, size=zf[0]["size"] if zf else None,
                              match=bool(zf) and zf[0]["checksum"] == "md5:" + res["ReplogleWeissman2022_rpe1.h5ad"]["md5"]
                              and zf[0]["size"] == res["ReplogleWeissman2022_rpe1.h5ad"]["size"]),
    inputs_sha256={fn: hashlib.sha256(open(os.path.join(INP, fn), "rb").read()).hexdigest()
                   for fn in sorted(os.listdir(INP)) if not fn.startswith("._")},
)
json.dump(out, open(os.path.join(V, "out/v2_hashes.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("files", "inputs_sha256")}, indent=1))
