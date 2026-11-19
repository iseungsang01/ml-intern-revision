import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
SHOTS = sorted(int(x) for x in g["shot"].unique())
bes_cols=[c for c in g.columns if c.startswith("BES_")]
info={}
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float).copy()
    prev=np.r_[np.nan,ti[:-1]]; ti[ti==prev]=np.nan; ti[ti>3000]=np.nan
    n=len(ti)
    # future-frame reference: median of the finite values among rows t+1..t+3 that lie within 35 ms of t (>=2 values)
    z=np.full(n,np.nan); z1=np.full(n,np.nan)
    for k in range(n):
        w=[]
        for j in (k+1,k+2,k+3):
            if j<n and t[j]-t[k]<0.035+1e-4 and np.isfinite(ti[j]): w.append(ti[j])
        if len(w)>=2: z[k]=np.median(w)
        if k+1<n and abs(t[k+1]-t[k]-0.01)<1e-4 and np.isfinite(ti[k+1]): z1[k]=ti[k+1]
    bes=d[bes_cols].values.astype(float).mean(1)
    # BES relative to the median of the previous 10 rows
    br=np.full(n,np.nan)
    for k in range(n):
        a=max(0,k-10)
        if k-a>=5:
            m=np.median(bes[a:k])
            if m>0: br[k]=bes[k]/m
    info[shot]=(t,ti,z,z1,br)
out=[]
for r in (2,3,4):
    for f in range(5):
        a=np.load(os.path.join(repo,"data/.b70p2/b54/rows_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        b=np.load(os.path.join(repo,"data/.b70/stage2_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        y=a["y"]; fidx=a["fidx"]; grow=a["grow"]
        Z=np.full(len(y),np.nan); Z1=np.full(len(y),np.nan); BR=np.full(len(y),np.nan); SH=np.zeros(len(y),int)
        for j in np.unique(fidx):
            m=fidx==j; t,ti,z,z1,br=info[SHOTS[j]]; idx=grow[m]-1
            Z[m]=z[idx]; Z1[m]=z1[idx]; BR[m]=br[idx]; SH[m]=SHOTS[j]
        out.append(pd.DataFrame({"r":r,"shot":SH,"y":y,"z":Z,"z1":Z1,"br":BR,"c2":b["est_c2"],"t2":b["est_t2"],"c":b["est_c"],"t":b["est_t"],"hold":a["hold_val"],"se_gp":a["se_gp"]}))
D=pd.concat(out,ignore_index=True)
D.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1z.pkl"))
print("rows",len(D),"z defined %.3f"%np.isfinite(D.z).mean(),"z1 defined %.3f"%np.isfinite(D.z1).mean())
def sse(est,ref,m): return ((D.loc[m,est]-D.loc[m,ref])**2).sum()
for ref in ("z","z1"):
    m=np.isfinite(D[ref]).values
    print("\n== reference = %s (rows %d per re-partition, share %.3f)"%("median of the next 2-3 frames" if ref=="z" else "the next frame alone",m.sum()/3,m.mean()))
    for A,B,lab in [("c2","t2","two-stage vs two-stage twin"),("c","t","network vs twin"),("c2","hold","two-stage vs hold"),("t2","hold","two-stage twin vs hold"),("c2","c","two-stage vs network")]:
        ry=sse(B,"y",m)-sse(A,"y",m); rz=sse(B,ref,m)-sse(A,ref,m)
        print("  %-28s skill vs own frame %+.3f | skill vs reference %+.3f | SSE reduction against the reference / against the own frame = %.3f"%(lab,1-sse(A,"y",m)/sse(B,"y",m),1-sse(A,ref,m)/sse(B,ref,m),rz/ry))
    print("  RMSE against own frame: two-stage %.1f, twin2 %.1f, hold %.1f ; against reference: two-stage %.1f, twin2 %.1f, hold %.1f ; reference vs own frame RMS %.1f"%(np.sqrt(sse("c2","y",m)/m.sum()),np.sqrt(sse("t2","y",m)/m.sum()),np.sqrt(sse("hold","y",m)/m.sum()),np.sqrt(sse("c2",ref,m)/m.sum()),np.sqrt(sse("t2",ref,m)/m.sum()),np.sqrt(sse("hold",ref,m)/m.sum()),np.sqrt(sse(ref,"y",m)/m.sum())))
# bootstrap of the ratio for the two-stage increment, reference z
rng=np.random.default_rng(7)
vals=[]
per={}
for r in (2,3,4):
    d=D[(D.r==r)&np.isfinite(D.z)]
    q=pd.DataFrame({"shot":d.shot.values,"ry":(d.t2-d.y)**2-(d.c2-d.y)**2,"rz":(d.t2-d.z)**2-(d.c2-d.z)**2,"ny":(d.t-d.y)**2-(d.c-d.y)**2,"nz":(d.t-d.z)**2-(d.c-d.z)**2,
                    "c2z":(d.c2-d.z)**2,"t2z":(d.t2-d.z)**2,"cz":(d.c-d.z)**2,"tz":(d.t-d.z)**2,"hz":(d.hold-d.z)**2}).groupby("shot").sum()
    per[r]=q.values; cols=list(q.columns)
B={k:[] for k in ["ratio2","ratio1","sk2z","sk1z","c2_hold_z","c_hold_z"]}
for i in range(2000):
    r=rng.choice([2,3,4]); M=per[r]; s=M[rng.integers(0,len(M),len(M))].sum(0); v=dict(zip(cols,s))
    B["ratio2"].append(v["rz"]/v["ry"]); B["ratio1"].append(v["nz"]/v["ny"]); B["sk2z"].append(1-v["c2z"]/v["t2z"]); B["sk1z"].append(1-v["cz"]/v["tz"]); B["c2_hold_z"].append(1-v["c2z"]/v["hz"]); B["c_hold_z"].append(1-v["cz"]/v["hz"])
for k,v in B.items(): print("  %-10s %+.3f [%+.3f, %+.3f]"%(k,np.median(v),np.quantile(v,.025),np.quantile(v,.975)))
# BES level of the frame relative to the 10 rows before, by how the target sits
m=np.isfinite(D.br).values
print("\nBES sample / median of the 10 rows before, scored rows: quantiles 1,5,50,95,99%:",np.round(np.nanquantile(D.br[m],[.01,.05,.5,.95,.99]),3))
