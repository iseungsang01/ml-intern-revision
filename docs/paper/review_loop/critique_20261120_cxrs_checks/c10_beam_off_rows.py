import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
bes_cols=[c for c in g.columns if c.startswith("BES_")]
rows=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float); vt=d["CES_VT"].values.astype(float)
    B=d[bes_cols].values.astype(float); n=len(t)
    if n<30: continue
    dt=np.r_[np.nan,np.diff(t)]
    for k in range(12,n-1):
        w=slice(k-10,k)
        if np.nanmax(dt[k-10:k+1])>0.0101: continue     # ten regular rows before and a regular step into the row (no absent frame just before)
        med=np.median(B[w],axis=0)
        if np.any(med<=0.02): continue                    # beam clearly on before (all nine channels well above zero)
        ratio=B[k]/med
        tb=ti[w]; tb=tb[np.isfinite(tb)&(tb<=3000)]
        if len(tb)<6: continue
        rows.append((shot,t[k],ratio.max(),ratio.mean(),ti[k],np.median(tb),np.std(tb)/np.median(tb),vt[k],dt[k+1] if k+1<n else np.nan))
R=pd.DataFrame(rows,columns=["shot","t","rmax","rmean","ti","tbefore","cv_before","vt","dt_next"])
print("rows examined (10 regular beam-on rows before, T_i level defined):",len(R))
off=R[R.rmax<0.4]
print("rows whose nine BES channels are all below 0.4 of their level in the 10 rows before (beam-off-like sample at the stamp):",len(off),"in",off.shot.nunique(),"discharges")
print("   of them with a reported T_i: %d (%.3f) ; share for all examined rows %.3f"%(np.isfinite(off.ti).sum(),np.isfinite(off.ti).mean(),np.isfinite(R.ti).mean()))
o=off[np.isfinite(off.ti)]
q=o.ti/o.tbefore
print("   reported T_i / median of the 10 rows before: quantiles 5,25,50,75,95 %%:",np.round(np.quantile(q,[.05,.25,.5,.75,.95]),3))
print("   share within 0.67..1.5: %.3f ; within 0.8..1.25: %.3f ; above 3 keV: %.3f ; below 100 eV: %.3f"%(((q>0.67)&(q<1.5)).mean(),((q>0.8)&(q<1.25)).mean(),(o.ti>3000).mean(),(o.ti<100).mean()))
on=R[(R.rmean>0.9)&(R.rmean<1.1)&np.isfinite(R.ti)]
qq=on.ti/on.tbefore
print("reference, rows with BES within 0.9..1.1 of the level before (n=%d): quantiles 5,25,50,75,95 %%: %s ; within 0.67..1.5: %.3f ; within 0.8..1.25: %.3f"%(len(on),np.round(np.quantile(qq,[.05,.25,.5,.75,.95]),3),((qq>0.67)&(qq<1.5)).mean(),((qq>0.8)&(qq<1.25)).mean()))
# deeper beam-off: all channels below 0.3
off2=R[R.rmax<0.3]; o2=off2[np.isfinite(off2.ti)]; q2=o2.ti/o2.tbefore
print("all nine channels below 0.3: rows %d, with T_i %d (%.3f); ratio quantiles 5,25,50,75,95: %s ; within 0.67..1.5: %.3f"%(len(off2),len(o2),len(o2)/max(1,len(off2)),np.round(np.quantile(q2,[.05,.25,.5,.75,.95]),3),((q2>0.67)&(q2<1.5)).mean()))
# is the next row an absent frame (i.e. the stamp caught the start of a programmed notch)?
print("   beam-off-like rows followed by a step above 15 ms (start of an absent frame): %.3f"%((off.dt_next>0.015).mean()))
onx=off[(off.dt_next<0.0101)]
ox=onx[np.isfinite(onx.ti)]; qx=ox.ti/ox.tbefore
print("   not followed by an absent frame: rows %d, with T_i %d (%.3f); ratio quantiles 5,25,50,75,95: %s; within 0.67..1.5: %.3f"%(len(onx),len(ox),len(ox)/max(1,len(onx)),np.round(np.quantile(qx,[.05,.25,.5,.75,.95]),3),((qx>0.67)&(qx<1.5)).mean()))
print("   discharges with most such rows:",onx.shot.value_counts().head(8).to_dict())
# rotation in those rows
print("   rotation value present in beam-off-like rows: %.3f (all examined rows %.3f)"%(np.isfinite(off.vt).mean(),np.isfinite(R.vt).mean()))
R.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"beamoff.pkl"))
