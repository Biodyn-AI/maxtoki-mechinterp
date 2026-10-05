import os, re, datetime, json
root = "<REPO_ROOT>/projects/maxtoki/runs"
tq = re.compile(r"(\d+)/(\d+) \[(\d+(?::\d+){1,2})<")
tsec = re.compile(r"(?:in|elapsed|took|time:?|wall(?:_seconds)?\"?:?)\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|min|m|h)\b", re.I)
def tosec(s):
    p=[int(x) for x in s.split(":")]
    return p[0]*60+p[1] if len(p)==2 else p[0]*3600+p[1]*60+p[2]
def fmt(t): return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
rows=[]
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ("node_modules",".git")]
    for f in fn:
        if not f.endswith(".log") or f.startswith("._"): continue
        p=os.path.join(dp,f)
        sz=os.path.getsize(p)
        if sz>200e6: continue
        txt=open(p,errors="ignore").read()
        segs=re.split(r"[\r\n]",txt)
        # tqdm bars: track max elapsed per (total) bar sequence; a bar ends when n==total
        bars=[]; cur=None
        for s in segs:
            for m in tq.finditer(s):
                n,tot,el=int(m.group(1)),int(m.group(2)),tosec(m.group(3))
                if cur is None or tot!=cur[0] or n<cur[1]:
                    if cur: bars.append(cur)
                    cur=[tot,n,el]
                else:
                    cur[1]=n; cur[2]=el
        if cur: bars.append(cur)
        tq_total=sum(b[2] for b in bars)
        exp=[]
        for s in segs:
            for m in tsec.finditer(s):
                v=float(m.group(1)); u=m.group(2).lower()
                v = v*60 if u in("min","m") else v*3600 if u=="h" else v
                exp.append(v)
        st=os.stat(p)
        rows.append(dict(log=os.path.relpath(p,root), mtime=fmt(st.st_mtime), birth=fmt(st.st_birthtime), size=sz, n_bars=len(bars), tqdm_sum_min=round(tq_total/60,1), longest_bar_min=round(max([b[2] for b in bars],default=0)/60,1), explicit_sum_min=round(sum(exp)/60,1)))
rows.sort(key=lambda r:r["log"])
for r in rows: print(json.dumps(r))
