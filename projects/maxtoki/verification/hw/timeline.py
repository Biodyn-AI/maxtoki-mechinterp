import os, datetime, collections, sys
root = "<REPO_ROOT>/projects/maxtoki"
def fmt(t): return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
targets = [os.path.join(root,"runs",d) for d in sorted(os.listdir(os.path.join(root,"runs")))] + [os.path.join(root,x) for x in ["summaries","audits","setup"]]
for t in targets:
    recs=[]
    for dp, dn, fn in os.walk(t):
        dn[:] = [d for d in dn if d not in ("node_modules",".git","__pycache__")]
        for f in fn:
            if f.startswith("._"): continue
            p=os.path.join(dp,f)
            try: st=os.stat(p)
            except: continue
            recs.append((st.st_mtime, getattr(st,"st_birthtime",0), p))
    if not recs: continue
    recs.sort()
    days=collections.Counter(fmt(r[0])[:10] for r in recs)
    bdays=collections.Counter(fmt(r[1])[:10] for r in recs)
    print("=="*10, os.path.relpath(t,root), "n=",len(recs))
    print(" mtime first:", fmt(recs[0][0]), os.path.relpath(recs[0][2],t))
    print(" mtime last :", fmt(recs[-1][0]), os.path.relpath(recs[-1][2],t))
    print(" mtime by day:", dict(sorted(days.items())))
    print(" birth by day:", dict(sorted(bdays.items())))
