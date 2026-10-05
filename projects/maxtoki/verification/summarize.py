import json, os, collections
OUT = "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand"
d = json.load(open(os.path.join(OUT, "inventory_raw.json")))
R = d["records"]
BIG = 50 * 1024 * 1024
def grp(rel):
    parts = rel.split("/")
    if parts[0] == "runs": return "runs/" + parts[1]
    return parts[0]
def h(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"
groups = collections.OrderedDict()
for r in R:
    groups.setdefault(grp(r["rel"]), []).append(r)
lines = []
lines.append("| group | category | n files | total size | n > 50 MB |")
lines.append("|---|---|---|---|---|")
tot_small = 0; tot_big = 0; tot_all = 0
per_group_small = {}
for g, rs in groups.items():
    cats = collections.defaultdict(lambda: [0, 0, 0])
    for r in rs:
        c = cats[r["cat"]]; c[0] += 1; c[1] += r["size"]; c[2] += r["size"] > BIG
    for c, v in sorted(cats.items()):
        lines.append(f"| {g} | {c} | {v[0]} | {h(v[1])} | {v[2]} |")
    small = sum(r["size"] for r in rs if r["size"] <= BIG and r["cat"] != "pycache")
    big = sum(r["size"] for r in rs if r["size"] > BIG)
    per_group_small[g] = (small, big, sum(r["size"] for r in rs))
print("\n".join(lines))
print()
print("| group | files <=50MB (no pycache) | files >50MB | all |")
for g, (s, b, a) in per_group_small.items():
    print(f"| {g} | {h(s)} | {h(b)} | {h(a)} |")
print("TOTAL small", h(sum(v[0] for v in per_group_small.values())), "big", h(sum(v[1] for v in per_group_small.values())))
# release estimates: scripts + configs(json/yaml) + logs + docs + tables + figures <= 50 MB; exclude setup model weights
cats_release = {"script", "json/config", "log", "doc", "table", "figure", "other"}
rel_size = sum(r["size"] for r in R if r["size"] <= BIG and r["cat"] in cats_release and r["cat"] != "pycache")
rel_arr_small = sum(r["size"] for r in R if r["size"] <= BIG and r["cat"] == "array/binary")
print("release text+figs <=50MB:", h(rel_size), " small arrays <=50MB:", h(rel_arr_small))
# code-only size
code = [r for r in R if r["cat"] == "script"]
print("scripts:", len(code), h(sum(r["size"] for r in code)))
# abs paths
absf = [r for r in R if r["abs"]]
print("\nFILES WITH ABS PATHS:", len(absf))
bycat = collections.Counter((grp(r["rel"]), r["cat"]) for r in absf)
for k, v in sorted(bycat.items()): print(" ", k, v)
ex = collections.Counter()
for r in absf:
    for e in r["abs_examples"]:
        pref = "/".join(e.split("/")[:6])
        ex[pref] += 1
print("path prefixes:", ex.most_common(25))
print("\nSECRETS:")
for r in R:
    if r["secrets"]: print(" ", r["rel"], r["secrets"])
print("\nPII:")
pii = collections.Counter()
for r in R:
    if r["pii"]:
        for k in r["pii"]: pii[(grp(r["rel"]), k)] += 1
for k, v in sorted(pii.items()): print(" ", k, v)
print("\nemail files:")
for r in R:
    if "email" in r["pii"]: print(" ", r["rel"], r["pii"]["email"])
print("\nBIG FILES:")
for r in sorted(R, key=lambda x: -x["size"]):
    if r["size"] > BIG: print(f"  {r['size']:>12d}  {h(r['size']):>10}  {r['rel']}")
print("\nSCRIPTS per group:")
for g, rs in groups.items():
    sc = [r for r in rs if r["cat"] == "script"]
    print(" ", g, len(sc), h(sum(r["size"] for r in sc)))
    for r in sc: print("     ", r["rel"], r["size"])
