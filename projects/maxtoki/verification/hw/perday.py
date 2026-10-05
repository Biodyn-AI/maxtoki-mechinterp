import os, datetime, collections
root = "<REPO_ROOT>/projects/maxtoki"
def fmt(t): return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
targets = [os.path.join(root,"runs",d) for d in sorted(os.listdir(os.path.join(root,"runs")))] + [os.path.join(root,x) for x in ["summaries","audits","setup"]]
allrecs=[]
for t in targets:
    per=collections.defaultdict(list)
    for dp, dn, fn in os.walk(t):
        dn[:] = [d for d in dn if d not in ("node_modules",".git","__pycache__")]
        for f in fn:
            if f.startswith("._"): continue
            p=os.path.join(dp,f)
            st=os.stat(p)
            per[fmt(st.st_mtime)[:10]].append(st.st_mtime)
            allrecs.append((st.st_mtime, os.path.relpath(p,root)))
    print("==", os.path.relpath(t,root))
    for d in sorted(per):
        v=sorted(per[d]); print("  ",d, fmt(v[0])[11:], "->", fmt(v[-1])[11:], "n=",len(v))
# global hourly-activity: count distinct hours with any write
hours=set(fmt(r[0])[:13] for r in allrecs)
print("distinct clock-hours with >=1 file write:", len(hours))
days=sorted(set(h[:10] for h in hours)); print("active days:", days)
