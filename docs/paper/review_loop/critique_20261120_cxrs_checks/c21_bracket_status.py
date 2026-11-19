import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
SHOTS=sorted(int(x) for x in g["shot"].unique())
stat={}  # per shot: array of status per grid row (0 before first absent frame of the segment, 1 between two, 2 after the last, 3 discharge without any absent frame)
rows=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float).copy(); n=len(t)
    raw=ti.copy(); ti[ti>3000]=np.nan
    st=np.zeros(n,int)
    if n>1:
        dt=np.diff(t); seg=np.r_[0,np.cumsum(dt>=0.5)]
        anyabs=((dt>0.015)&(dt<0.045)).any()
        for s in np.unique(seg):
            idx=np.flatnonzero(seg==s); tt=t[idx]; d2=np.diff(tt); ab=np.flatnonzero((d2>0.015)&(d2<0.045))
            if len(ab)==0: st[idx]=0
            else:
                first=idx[ab[0]+1]; last=idx[ab[-1]]
                st[idx]=1; st[idx[idx<first]]=0; st[idx[idx>last]]=2
        if not anyabs: st[:]=3
    else: st[:]=3
    stat[shot]=st
    lt=np.log(ti); S=np.full(n,np.nan)
    if n>=3:
        ok=(np.abs(t[1:-1]-t[:-2]-0.01)<1e-4)&(np.abs(t[2:]-t[1:-1]-0.01)<1e-4)
        v=lt[1:-1]-0.5*(lt[:-2]+lt[2:]); v[~ok]=np.nan; S[1:-1]=v
    rows.append(pd.DataFrame({"shot":shot,"st":st,"ti":ti,"raw":raw,"S":S}))
A=pd.concat(rows,ignore_index=True)
names={0:"before the first absent frame of the segment",1:"between two absent frames",2:"after the last absent frame",3:"discharge without any absent frame"}
print("status | grid rows | T_i missing | above 3 keV (of values) | below 100 eV (of values) | median T_i | robust relative one-frame scatter (1.4826 MAD of S / sqrt(1.5)) | share |S|>0.3")
for k in range(4):
    a=A[A.st==k]; s=a.S.dropna()
    print("  %-46s %7d  %.3f  %.4f  %.4f  %5.0f  %.4f  %.4f"%(names[k],len(a),a.raw.isna().mean(),(a.raw>3000).sum()/a.raw.notna().sum(),(a.raw<100).sum()/a.raw.notna().sum(),a.ti.median(),1.4826*np.median(np.abs(s-s.median()))/np.sqrt(1.5),(np.abs(s)>0.3).mean()))
# within-discharge comparison: discharges that have >=100 S values in status 1 and in status 0 or 2
res=[]
for shot,a in A.groupby("shot"):
    s1=a.S[a.st==1].dropna()
    for k in (0,2):
        sk=a.S[a.st==k].dropna()
        if len(s1)>=100 and len(sk)>=100:
            f=lambda s:1.4826*np.median(np.abs(s-s.median()))
            res.append((k,f(sk)/f(s1),(np.abs(sk)>0.3).mean(),(np.abs(s1)>0.3).mean()))
R=pd.DataFrame(res,columns=["k","scatter_ratio","exc_k","exc_1"])
for k in (0,2):
    r=R[R.k==k]
    print("within discharge, %s vs between two: n=%d discharges, median ratio of the one-frame scatter %.3f [quartiles %.3f, %.3f]; share of >0.3 excursions %.4f vs %.4f"%(names[k],len(r),r.scatter_ratio.median(),r.scatter_ratio.quantile(.25),r.scatter_ratio.quantile(.75),r.exc_k.median(),r.exc_1.median()))
# estimator skills by status on the D1 rows
D=pd.read_pickle(os.path.join(os.path.dirname(sys.argv[1]),"d1all.pkl"))
ST=np.zeros(len(D),int); pos=0
for r in (2,3,4):
    for f in range(5):
        a=np.load(os.path.join(repo,"data/.b70p2/b54/rows_r%d_f%d.npz"%(r,f)),allow_pickle=True)
        fidx=a["fidx"]; grow=a["grow"]; n=len(fidx); st=np.zeros(n,int)
        for j in np.unique(fidx):
            m=fidx==j; st[m]=stat[SHOTS[j]][grow[m]-1]
        ST[pos:pos+n]=st; pos+=n
D["st"]=ST
se=lambda e,m:((D.loc[m,e]-D.loc[m,"y"])**2).sum()
print("status | share of D1 rows | two-stage vs causal GP | two-stage vs two-stage twin | network vs twin | RMSE two-stage (eV)")
for k in range(4):
    m=(D.st.values==k)
    print("  %-46s %.3f  %+.3f  %+.3f  %+.3f  %.1f"%(names[k],m.mean(),1-se("c2",m)/D.loc[m,"se_gp"].sum(),1-se("c2",m)/se("t2",m),1-se("c",m)/se("t",m),np.sqrt(se("c2",m)/m.sum())))
