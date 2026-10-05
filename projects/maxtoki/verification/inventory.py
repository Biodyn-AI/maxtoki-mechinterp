import os, re, json, sys, collections
M = "<REPO_ROOT>/projects/maxtoki"
OUT = "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand"
ROOTS = ["runs", "setup", "audits", "summaries", "paper-plos-one", "paper-biosystems", "paper"]
SKIP_DIRS = {"node_modules", ".venv", ".git"}
SCRIPT = {".py", ".sh", ".R", ".r", ".ipynb", ".js", ".mjs", ".ts", ".tsx"}
LOG = {".log", ".out", ".err"}
DOC = {".md", ".txt", ".tex", ".bib", ".markdown"}
TABLE = {".csv", ".tsv", ".parquet", ".feather"}
JSON = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
ARRAY = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".h5", ".h5ad", ".pkl", ".pickle", ".bin", ".joblib"}
FIG = {".png", ".pdf", ".svg", ".html", ".jpg", ".jpeg", ".tif", ".tiff", ".eps"}
TEXT_EXT = SCRIPT | LOG | DOC | TABLE | JSON | {".html", ".svg", ".css", ".sty", ".cls", ".bst"}
BIG = 50 * 1024 * 1024
SCAN_MAX = 60 * 1024 * 1024

def cat(ext, name):
    if name == "__pycache__" or ext == ".pyc": return "pycache"
    for c, s in [("script", SCRIPT), ("log", LOG), ("doc", DOC), ("table", TABLE), ("json/config", JSON),
                 ("array/binary", ARRAY), ("figure", FIG)]:
        if ext in s: return c
    return "other"

abs_re = re.compile(rb"(/Volumes/[^\s\"'`,)\]]+|/Users/[^\s\"'`,)\]]+|/private/(?:tmp|var)/[^\s\"'`,)\]]+|/home/[^\s\"'`,)\]]+|/workspace/[^\s\"'`,)\]]+|/root/[^\s\"'`,)\]]+|/runpod-volume/[^\s\"'`,)\]]+)")
secret_res = {
    "hf_token_pattern": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "sk_key_pattern": re.compile(rb"sk-[A-Za-z0-9_\-]{20,}"),
    "api_key_word": re.compile(rb"api[_-]?key", re.I),
    "password_word": re.compile(rb"passw(or)?d", re.I),
    "HF_TOKEN_env": re.compile(rb"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN"),
    "token_assign": re.compile(rb"token\s*[=:]\s*[\"'][A-Za-z0-9_\-]{16,}[\"']", re.I),
    "aws_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "ghp_token": re.compile(rb"gh[pous]_[A-Za-z0-9]{30,}"),
}
pii_res = {
    "email": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "username_ihorkendiukhov": re.compile(rb"ihorkendiukhov", re.I),
    "name_kendiukhov": re.compile(rb"kendiukhov", re.I),
    "ip_addr": re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b"),
}

records = []
dir_skipped = []
for r in ROOTS:
    base = os.path.join(M, r)
    for dp, dns, fns in os.walk(base):
        keep = []
        for d in dns:
            if d in SKIP_DIRS:
                full = os.path.join(dp, d)
                n = 0; s = 0
                for dp2, _, f2 in os.walk(full):
                    for f in f2:
                        try:
                            s += os.lstat(os.path.join(dp2, f)).st_size; n += 1
                        except OSError: pass
                dir_skipped.append((os.path.relpath(full, M), n, s))
            else:
                keep.append(d)
        dns[:] = keep
        for fn in fns:
            p = os.path.join(dp, fn)
            try: st = os.lstat(p)
            except OSError: continue
            rel = os.path.relpath(p, M)
            ext = os.path.splitext(fn)[1].lower()
            c = "pycache" if "__pycache__" in rel else cat(ext, fn)
            rec = {"rel": rel, "size": st.st_size, "ext": ext, "cat": c, "abs": 0, "abs_examples": [],
                   "secrets": {}, "pii": {}}
            if (ext in TEXT_EXT or c == "pycache") and st.st_size <= SCAN_MAX:
                try:
                    with open(p, "rb") as fh: data = fh.read()
                    m = abs_re.findall(data)
                    rec["abs"] = len(m)
                    rec["abs_examples"] = sorted(set(x.decode("utf-8", "replace")[:140] for x in m))[:5]
                    for k, rx in secret_res.items():
                        n = len(rx.findall(data))
                        if n: rec["secrets"][k] = n
                    for k, rx in pii_res.items():
                        n = len(rx.findall(data))
                        if n: rec["pii"][k] = n
                    del data
                except Exception as e:
                    rec["err"] = str(e)
            records.append(rec)

json.dump({"records": records, "skipped_dirs": dir_skipped}, open(os.path.join(OUT, "inventory_raw.json"), "w"))
print("files", len(records))
print("skipped", dir_skipped)
