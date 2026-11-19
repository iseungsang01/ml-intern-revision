import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
bes_cols=[c for c in g.columns if c.startswith("BES_")]; ecei_cols=[c for c in g.columns if c.startswith("ECEI_")]
rows=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float).copy(); vt=d["CES_VT"].values.astype(float).copy()
    n=len(t)
    if n<30: continue
    pv=np.r_[np.nan,vt[:-1]]; vt[vt==pv]=np.nan; vt[np.abs(vt)>1000]=np.nan
    ti[ti>3000]=np.nan
    b=d[bes_cols].values.astype(float).mean(1); e=d[ecei_cols].values.astype(float).mean(1)
    dt=np.r_[np.nan,np.diff(t)]
    # centred windows are not used: everything is relative to the 10 rows before (input-side conditioning)
    for k in range(12,n):
        if np.nanmax(dt[k-10:k+1])>0.0101: continue
        bb=np.median(b[k-10:k]); 
        if bb<=0.02: continue
        tw=ti[k-10:k]; tw=tw[np.isfinite(tw)]; vw=vt[k-10:k]; vw=vw[np.isfinite(vw)]; ee=np.median(e[k-10:k])
        tr=ti[k]/np.median(tw) if len(tw)>=6 and np.isfinite(ti[k]) else np.nan
        vr=vt[k]/np.median(vw) if len(vw)>=6 and np.isfinite(vt[k]) and np.median(vw)>20 else np.nan
        er=e[k]/ee if ee>0.02 else np.nan
        rows.append((shot,b[k]/bb,tr,vr,er))
R=pd.DataFrame(rows,columns=["shot","br","tr","vr","er"])
R.to_pickle(os.path.join(os.path.dirname(sys.argv[1]),"dose.pkl"))
bins=[0.4,0.6,0.8,0.9,0.97,1.03,1.1,1.2,1.4,9]
print("BES at the stamp / level of the 10 rows before | n(T_i) | median T_i ratio | n(V) | median fresh V_rot ratio [quartiles] | median ECEI ratio")
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(R.br>=lo)&(R.br<hi); d=R[m]
    tq=d.tr.dropna(); vq=d.vr.dropna(); eq=d.er.dropna()
    print("  %.2f-%.2f | %6d | %.3f | %6d | %.3f [%.3f, %.3f] | %.3f"%(lo,hi,len(tq),tq.median(),len(vq),vq.median(),vq.quantile(.25),vq.quantile(.75),eq.median()))
# same restricted to rows that have both a T_i ratio and a fresh V ratio (same frames, same fit)
both=R.dropna(subset=["tr","vr"])
print("rows with both a T_i ratio and a fresh V_rot ratio:",len(both),"in",both.shot.nunique(),"discharges")
for lo,hi in zip(bins[:-1],bins[1:]):
    d=both[(both.br>=lo)&(both.br<hi)]
    if len(d)<30: print("  %.2f-%.2f | n %d (too few)"%(lo,hi,len(d))); continue
    print("  %.2f-%.2f | n %6d | median T_i ratio %.3f | median V_rot ratio %.3f"%(lo,hi,len(d),d.tr.median(),d.vr.median()))
# slopes in log-log on rows with both (robust: Theil-like via binned medians is above; here least squares on |log br|<0.5)
bb=both[(both.br>0.6)&(both.br<1.6)&(both.tr>0)&(both.vr>0)]
x=np.log(bb.br.values); 
for nm,yv in [("T_i",np.log(bb.tr.values)),("V_rot",np.log(bb.vr.values))]:
    A=np.c_[x,np.ones(len(x))]; coef=np.linalg.lstsq(A,np.clip(yv,-1.5,1.5),rcond=None)[0]
    print("least-squares slope of log(%s ratio) on log(BES ratio), same rows: %.3f"%(nm,coef[0]))
# bootstrap over discharges for the two slopes
rng=np.random.default_rng(5); us=bb.shot.unique(); idx={s:np.flatnonzero(bb.shot.values==s) for s in us}
yT=np.clip(np.log(bb.tr.values),-1.5,1.5); yV=np.clip(np.log(bb.vr.values),-1.5,1.5); sT=[];sV=[]
for i in range(500):
    ii=np.concatenate([idx[s] for s in rng.choice(us,len(us))]); A=np.c_[x[ii],np.ones(len(ii))]
    sT.append(np.linalg.lstsq(A,yT[ii],rcond=None)[0][0]); sV.append(np.linalg.lstsq(A,yV[ii],rcond=None)[0][0])
print("95%% intervals (discharges resampled): T_i slope [%.3f, %.3f]; V_rot slope [%.3f, %.3f]"%(np.quantile(sT,.025),np.quantile(sT,.975),np.quantile(sV,.025),np.quantile(sV,.975)))
