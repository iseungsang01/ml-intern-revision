import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1])
repo = sys.argv[2]
# per-file cut series and row classes computed from the measured series only
cls = {}
for shot, d in g.groupby("shot", sort=False):
    t = d["time"].values; ti = d["CES_TI"].values.astype(float).copy()
    prev = np.r_[np.nan, ti[:-1]]
    ti[(ti==prev)] = np.nan          # held repeats
    ti[ti>3000] = np.nan             # cut
    n=len(ti)
    lt = np.log(ti)
    s = np.full(n,np.nan)
    if n>=3:
        ok = (np.abs(t[1:-1]-t[:-2]-0.01)<1e-4)&(np.abs(t[2:]-t[1:-1]-0.01)<1e-4)
        v = lt[1:-1]-0.5*(lt[:-2]+lt[2:])
        v[~ok]=np.nan
        s[1:-1]=v
    # ratio to the median of the other values in the +-5 row window
    ratio = np.full(n,np.nan)
    for k in range(n):
        a=max(0,k-5); b=min(n,k+6)
        w = np.r_[ti[a:k],ti[k+1:b]]
        w = w[np.isfinite(w)]
        if len(w)>=4 and np.isfinite(ti[k]): ratio[k]=ti[k]/np.median(w)
    cls["s%d.csv"%shot]=(t,ti,s,ratio)
out=[]
SHOTS = sorted(int(x) for x in g['shot'].unique())
for r in (2,3,4):
    for f in range(5):
        a = np.load(os.path.join(repo,"data/.b70p2/b54/rows_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        b = np.load(os.path.join(repo,"data/.b70/stage2_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        assert np.array_equal(a["fidx"],b["fidx"]) and np.array_equal(a["grow"],b["grow"])
        y=a["y"]; files=a["test_files"]; fidx=a["fidx"]; grow=a["grow"]
        S=np.full(len(y),np.nan); R=np.full(len(y),np.nan); Sprev=np.full(len(y),np.nan); Rprev=np.full(len(y),np.nan); chk=np.full(len(y),np.nan)
        SHOT=np.zeros(len(y),int)
        for j in np.unique(fidx):
            fn = "s%d.csv" % SHOTS[j]
            m = fidx==j
            t,ti,s,ratio = cls[fn]
            idx = grow[m]-1
            assert np.allclose(ti[idx], y[m], rtol=1e-4, atol=1e-2), (fn, ti[idx[:3]], y[m][:3])
            S[m]=s[idx]; R[m]=ratio[idx]; chk[m]=ti[idx]
            pi = np.clip(idx-1,0,None)
            Sprev[m]=np.where(idx>0,s[pi],np.nan); Rprev[m]=np.where(idx>0,ratio[pi],np.nan)
            SHOT[m]=SHOTS[j]
        d = pd.DataFrame({"r":r,"f":f,"shot":SHOT,"y":y,"S":S,"R":R,"Sprev":Sprev,"Rprev":Rprev,
            "se_c":a["se_c"],"se_t":a["se_t"],"se_hold":a["se_hold"],"se_gp":a["se_gp"],"se_off":a["se_off"],
            "se_c2":(b["est_c2"]-y)**2,"se_t2":(b["est_t2"]-y)**2,"e_c2":b["est_c2"]-y,"e_t2":b["est_t2"]-y,"e_c":b["est_c"]-y,"e_t":b["est_t"]-y})
        # consistency: stored se_c equals (est_c-y)^2
        assert np.allclose(d["se_c"],(b["est_c"]-y)**2,rtol=1e-3,atol=1e-1), "se_c mismatch"
        out.append(d)
D = pd.concat(out,ignore_index=True)
D.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1rows.pkl"))
print("rows per re-partition", D.groupby("r").size().to_dict())
def skill(a,b,m): return 1-D.loc[m,a].sum()/D.loc[m,b].sum()
allm = np.ones(len(D),bool)
print("ALL rows: two-stage vs GP %.3f ; two-stage vs two-stage twin %.3f ; net vs twin %.3f ; net vs GP %.3f ; two-stage vs offline GP %.3f"%(skill("se_c2","se_gp",allm),skill("se_c2","se_t2",allm),skill("se_c","se_t",allm),skill("se_c","se_gp",allm),skill("se_c2","se_off",allm)))
print("RMSE: two-stage %.1f net %.1f twin %.1f GP %.1f hold %.1f"%tuple(np.sqrt(D[k].mean()) for k in ["se_c2","se_c","se_t","se_gp","se_hold"]))
