import os, re, collections
M = "<REPO_ROOT>/projects/maxtoki"
runs = os.path.join(M, "runs")
rx = re.compile(r"[\"']([A-Za-z0-9_{}\-./]+\.(?:npy|npz|csv|json|pt|parquet|tsv|pkl|h5ad|md|png|pdf|txt))[\"']")
for p in sorted(os.listdir(runs)):
    rd = os.path.join(runs, p)
    names = set()
    for dp, dns, fns in os.walk(rd):
        dns[:] = [d for d in dns if d != "node_modules"]
        names.update(fns)
    missing = collections.defaultdict(list)
    for dp, dns, fns in os.walk(rd):
        dns[:] = [d for d in dns if d not in ("node_modules", "atlas")]
        for fn in fns:
            if not fn.endswith(".py"): continue
            src = open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()
            for m in rx.findall(src):
                if "{" in m: continue
                b = os.path.basename(m)
                if b not in names:
                    missing[b].append(fn)
    print("===", p, "referenced-but-absent basenames:", len(missing))
    for b, fs in sorted(missing.items()):
        print("   ", b, "<-", ",".join(sorted(set(fs)))[:150])
