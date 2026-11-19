import os, glob
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
SHOTS = sorted(int(x) for x in g["shot"].unique())
info={}
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float).copy()
    prev=np.r_[np.nan,ti[:-1]]; ti[ti==prev]=np.nan; ti[ti>3000]=np.nan
    n=len(ti); z=np.full(n,np.nan); zp=np.full(n,np.nan)
    lt=np.log(ti); s=np.full(n,np.nan)
    if n>=3:
        ok=(np.abs(t[1:-1]-t[:-2]-0.01)<1e-4)&(np.abs(t[2:]-t[1:-1]-0.01)<1e-4)
        v=lt[1:-1]-0.5*(lt[:-2]+lt[2:]); v[~ok]=np.nan; s[1:-1]=v
    for k in range(n):
        w=[ti[j] for j in (k+1,k+2,k+3) if j<n and t[j]-t[k]<0.035+1e-4 and np.isfinite(ti[j])]
        if len(w)>=2: z[k]=np.median(w)
    info[shot]=(t,ti,z,s)
# signed causal-GP estimates from the stored replay tables (delay L0 = the ordinary condition)
gp={}
for fn in glob.glob(os.path.join(repo,"data/.b70p2/replay/s*.npz")):
    a=np.load(fn,allow_pickle=True)
    shot=int(os.path.basename(fn)[1:-4])
    gg=a["g"]; di=a["d_idx"]
    gp[shot]=dict(zip(gg[di].tolist(), a["d_L0_gp"].tolist()))
print("replay files",len(gp))
out=[]
for r in (2,3,4):
    for f in range(5):
        a=np.load(os.path.join(repo,"data/.b70p2/b54/rows_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        b=np.load(os.path.join(repo,"data/.b70/stage2_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        k=np.load(os.path.join(repo,"data/.b70p2/b56/r%d_f%d.npz"%(r,f)),allow_pickle=True)
        y=a["y"]; fidx=a["fidx"]; grow=a["grow"]
        assert np.allclose(k["y"],y,rtol=1e-5,atol=1e-3)
        Z=np.full(len(y),np.nan); S=np.full(len(y),np.nan); SH=np.zeros(len(y),int); GP=np.full(len(y),np.nan)
        for j in np.unique(fidx):
            m=fidx==j; t,ti,z,s=info[SHOTS[j]]; idx=grow[m]-1
            Z[m]=z[idx]; S[m]=s[idx]; SH[m]=SHOTS[j]
            mp=gp.get(SHOTS[j],{})
            GP[m]=[mp.get(int(q),np.nan) for q in grow[m]]
        d=pd.DataFrame({"r":r,"shot":SH,"y":y,"z":Z,"S":S,"c2":b["est_c2"],"t2":b["est_t2"],"c":b["est_c"],"t":b["est_t"],"hold":a["hold_val"],"gp":GP,"se_gp":a["se_gp"],"kf0":k["est_kf0"],"kfx":k["est_kfx"],"kg":k["est_kg"]})
        out.append(d)
D=pd.concat(out,ignore_index=True)
ok=np.isfinite(D.gp)
print("rows",len(D),"GP estimate found for %.3f"%ok.mean(), "max |(gp-y)^2 - se_gp| rel:", np.nanmax(np.abs((D.gp[ok]-D.y[ok])**2-D.se_gp[ok])/(1+D.se_gp[ok])))
D.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1all.pkl"))
m=(np.isfinite(D.z)&ok).values
print("rows with future reference and signed GP: %d per re-partition (%.3f of D1)"%(m.sum()/3,m.mean()))
def sse(e,ref,mm): return ((D.loc[mm,e]-D.loc[mm,ref])**2).sum()
pairs=[("c2","gp","two-stage vs causal GP"),("c2","t2","two-stage vs two-stage twin"),("c","t","network vs twin"),("c","gp","network vs causal GP"),("t2","gp","two-stage twin vs causal GP"),("t","gp","twin vs causal GP"),("kfx","kf0","filter with BES vs filter without"),("kg","kf0","filter radial-difference only vs without"),("kf0","gp","filter without BES vs causal GP"),("c2","kfx","two-stage vs filter with BES"),("gp","hold","causal GP vs hold"),("c2","hold","two-stage vs hold")]
print("%-40s  skill(own frame)  skill(next 2-3 frames)  reduction ratio"%"pair")
for A,B,lab in pairs:
    ry=sse(B,"y",m)-sse(A,"y",m); rz=sse(B,"z",m)-sse(A,"z",m)
    print("%-40s  %+.3f            %+.3f                  %.3f"%(lab,1-sse(A,"y",m)/sse(B,"y",m),1-sse(A,"z",m)/sse(B,"z",m),rz/ry))
print("RMSE vs own frame / vs next-frames reference (eV):")
for e in ["hold","gp","kf0","kfx","t","c","t2","c2"]:
    print("   %-5s %.1f / %.1f"%(e,np.sqrt(sse(e,"y",m)/m.sum()),np.sqrt(sse(e,"z",m)/m.sum())))
# two-level bootstrap
rng=np.random.default_rng(11)
per={}
names=["hold","gp","kf0","kfx","t","c","t2","c2"]
for r in (2,3,4):
    d=D[(D.r==r)&m]
    q={"shot":d.shot.values}
    for e in names:
        q[e+"_y"]=(d[e]-d.y).values**2; q[e+"_z"]=(d[e]-d.z).values**2
    per[r]=pd.DataFrame(q).groupby("shot").sum()
res={lab:{"y":[],"z":[],"ratio":[]} for _,_,lab in pairs}
for i in range(2000):
    r=rng.choice([2,3,4]); P=per[r]; s=P.iloc[rng.integers(0,len(P),len(P))].sum(0)
    for A,B,lab in pairs:
        res[lab]["y"].append(1-s[A+"_y"]/s[B+"_y"]); res[lab]["z"].append(1-s[A+"_z"]/s[B+"_z"]); res[lab]["ratio"].append((s[B+"_z"]-s[A+"_z"])/(s[B+"_y"]-s[A+"_y"]))
print("\n95% two-level intervals:")
for _,_,lab in pairs:
    v=res[lab]
    print("%-40s own %+.3f [%+.3f, %+.3f] | next frames %+.3f [%+.3f, %+.3f] | ratio %.3f [%.3f, %.3f]"%(lab,np.median(v["y"]),np.quantile(v["y"],.025),np.quantile(v["y"],.975),np.median(v["z"]),np.quantile(v["z"],.025),np.quantile(v["z"],.975),np.median(v["ratio"]),np.quantile(v["ratio"],.025),np.quantile(v["ratio"],.975)))
