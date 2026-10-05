import os, datetime
root = "<REPO_ROOT>/projects/maxtoki/runs"
def fmt(t): return datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M")
def union(iv):
    iv=sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
allv=[]
for pl in sorted(os.listdir(root)):
    iv=[]
    for dp,dn,fn in os.walk(os.path.join(root,pl)):
        dn[:]=[d for d in dn if d!="node_modules"]
        for f in fn:
            if f.endswith(".log") and not f.startswith("._"):
                st=os.stat(os.path.join(dp,f)); iv.append((st.st_birthtime, st.st_mtime))
    u=union(iv); tot=sum(b-a for a,b in u)/3600
    allv+=iv
    print(f"{pl:30s} logs={len(iv):3d}  union of log [create,last-write] = {tot:5.2f} h ; windows: " + "; ".join(f"{fmt(a)}-{fmt(b)[6:]}" for a,b in u if b-a>600))
u=union(allv); print("ALL pipelines union:", round(sum(b-a for a,b in u)/3600,2),"h")
print("sum of per-pipeline unions (double counts overlap) computed above")
