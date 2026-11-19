import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
rows=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; vt=d["CES_VT"].values.astype(float); ti=d["CES_TI"].values.astype(float); n=len(t)
    st=np.zeros(n,int)
    if n>1:
        dt=np.diff(t); seg=np.r_[0,np.cumsum(dt>=0.5)]; anyabs=((dt>0.015)&(dt<0.045)).any()
        for s in np.unique(seg):
            idx=np.flatnonzero(seg==s); tt=t[idx]; d2=np.diff(tt); ab=np.flatnonzero((d2>0.015)&(d2<0.045))
            if len(ab)==0: st[idx]=0
            else:
                first=idx[ab[0]+1]; last=idx[ab[-1]]; st[idx]=1; st[idx[idx<first]]=0; st[idx[idx>last]]=2
        if not anyabs: st[:]=3
    else: st[:]=3
    pv=np.r_[np.nan,vt[:-1]]; held=np.isfinite(vt)&(vt==pv)
    rows.append(pd.DataFrame({"shot":shot,"st":st,"held":held,"vobs":np.isfinite(vt),"tobs":np.isfinite(ti)}))
A=pd.concat(rows,ignore_index=True)
names={0:"before the first absent frame of the segment",1:"between two absent frames",2:"after the last absent frame",3:"discharge without any absent frame"}
print("status | grid rows | rotation value present | of those, repeat of the previous value | T_i present")
for k in range(4):
    a=A[A.st==k]; print("  %-46s %7d  %.3f  %.3f  %.3f"%(names[k],len(a),a.vobs.mean(),a.held.sum()/max(1,a.vobs.sum()),a.tobs.mean()))
# per discharge: is the rotation fresh only between absent frames?
ps=A.groupby("shot").apply(lambda a: pd.Series({"held_share":a.held.sum()/max(1,a.vobs.sum()),"nabs_rows":(a.st==1).mean(),"st3":(a.st==3).all()}))
print("discharges without any absent frame: n=%d, median share of repeated rotation values %.3f ; with absent frames: n=%d, median %.3f"%(ps.st3.sum(),ps.held_share[ps.st3].median(),(~ps.st3).sum(),ps.held_share[~ps.st3].median()))
