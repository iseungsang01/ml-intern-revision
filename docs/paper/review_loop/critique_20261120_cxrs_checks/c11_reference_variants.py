import os, glob
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
SHOTS = sorted(int(x) for x in g["shot"].unique())
info={}; nabs={}
allsp=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float).copy()
    prev=np.r_[np.nan,ti[:-1]]; ti[ti==prev]=np.nan; ti[ti>3000]=np.nan
    n=len(ti)
    dt=np.diff(t); ab=np.flatnonzero((dt>0.015)&(dt<0.045))
    nabs[shot]=len(ab)
    if len(ab)>1:
        sp=t[ab[1:]]-t[ab[:-1]+1]; allsp+=list(sp[sp<5])
    za=np.full(n,np.nan); zb=np.full(n,np.nan); zc=np.full(n,np.nan)
    for k in range(n):
        w=[ti[j] for j in (k+1,k+2,k+3) if j<n and t[j]-t[k]<0.035+1e-4 and np.isfinite(ti[j])]
        if len(w)>=2: za[k]=np.mean(w)
        w=[ti[j] for j in (k+2,k+3,k+4) if j<n and t[j]-t[k]<0.045+1e-4 and np.isfinite(ti[j])]
        if len(w)>=2: zb[k]=np.median(w)
        if k+2<n and abs(t[k+2]-t[k]-0.02)<1e-4 and np.isfinite(ti[k+2]): zc[k]=ti[k+2]
    info[shot]=(za,zb,zc)
print("discharges with no absent frame:",sum(v==0 for v in nabs.values()),"of",len(nabs),"; absent steps total",sum(nabs.values()))
print("spacing between consecutive absent frames: n=%d, median %.3f s, quartiles %.3f..%.3f"%(len(allsp),np.median(allsp),np.quantile(allsp,.25),np.quantile(allsp,.75)))
D=pd.read_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1all.pkl"))
out=[]; pos=0
ZA=np.full(len(D),np.nan); ZB=np.full(len(D),np.nan); ZC=np.full(len(D),np.nan); BL=np.zeros(len(D),bool); NA=np.zeros(len(D),int)
for r in (2,3,4):
    for f in range(5):
        a=np.load(os.path.join(repo,"data/.b70p2/b54/rows_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        fidx=a["fidx"]; grow=a["grow"]; n=len(fidx)
        sl=slice(pos,pos+n)
        assert np.allclose(D.y.values[sl],a["y"])
        za=np.full(n,np.nan); zb=np.full(n,np.nan); zc=np.full(n,np.nan); na=np.zeros(n,int)
        for j in np.unique(fidx):
            m=fidx==j; A,B,C=info[SHOTS[j]]; idx=grow[m]-1
            za[m]=A[idx]; zb[m]=B[idx]; zc[m]=C[idx]; na[m]=nabs[SHOTS[j]]
        ZA[sl]=za; ZB[sl]=zb; ZC[sl]=zc; BL[sl]=a["blip13"]; NA[sl]=na
        pos+=n
D["za"]=ZA; D["zb"]=ZB; D["zc"]=ZC; D["blip13"]=BL; D["nabs"]=NA
ok=np.isfinite(D.gp).values
def tab(mm,refs,pairs,title):
    print(title)
    for ref,rl in refs:
        m=mm&np.isfinite(D[ref]).values
        def sse(e,rf): return ((D.loc[m,e]-D.loc[m,rf])**2).sum()
        line="   ref=%-28s rows/rep %7d |"%(rl,m.sum()/3)
        for A,B,lab in pairs:
            line+=" %s: own %+.3f, ref %+.3f, ratio %.2f |"%(lab,1-sse(A,"y")/sse(B,"y"),1-sse(A,ref)/sse(B,ref),(sse(B,ref)-sse(A,ref))/(sse(B,"y")-sse(A,"y")))
        print(line)
pairs=[("c2","t2","2stage/twin2"),("c","t","net/twin"),("kfx","kf0","KFX/KF0"),("c2","gp","2stage/GP"),("t2","gp","twin2/GP")]
refs=[("z","median of t+1..t+3"),("za","mean of t+1..t+3"),("zb","median of t+2..t+4"),("zc","t+2 alone")]
tab(ok,refs,pairs,"ALL rows, different future references")
tab(ok&D.blip13.values,refs[:1],pairs,"rows 1-3 after an absent frame (blip13)")
tab(ok&~D.blip13.values,refs[:1],pairs,"all other rows")
tab(ok&(D.nabs.values==0),refs[:1],pairs,"discharges without any absent frame")
tab(ok&(D.nabs.values>0),refs[:1],pairs,"discharges with at least one absent frame")
print("share of D1 rows: blip13 %.4f ; in discharges without absent frame %.4f"%(D.blip13.mean(),(D.nabs==0).mean()))
D.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1all.pkl"))
