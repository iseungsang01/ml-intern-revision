import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
bes_cols=[c for c in g.columns if c.startswith("BES_")]
ev=[]; allr=[]
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float); vt=d["CES_VT"].values.astype(float)
    B=d[bes_cols].values.astype(float); n=len(t)
    if n<60: continue
    med=np.median(B,axis=0)
    if np.any(med<=0.02): continue
    rel=(B/med)
    on=rel.mean(1)>0.6; off=rel.max(1)<0.3
    # tail: the final run of rows that are all 'off', preceded by at least 20 'on' rows among the 25 before it
    k=n
    while k>0 and off[k-1]: k-=1
    if k==n or n-k<3: continue
    if on[max(0,k-25):k].sum()<20: continue
    before=ti[max(0,k-15):k]; before=before[np.isfinite(before)&(before<=3000)]
    if len(before)<8: continue
    lvl=np.median(before); sc=1.4826*np.median(np.abs(before-lvl))/lvl
    tail=ti[k:]; tt=t[k:]-t[k-1]
    ev.append((shot,n-k,np.isfinite(tail).sum(),lvl,sc))
    for a,b in zip(tt,tail): allr.append((shot,a,b/lvl if np.isfinite(b) else np.nan))
E=pd.DataFrame(ev,columns=["shot","tail_rows","with_ti","level_before","rel_scatter_before"]); A=pd.DataFrame(allr,columns=["shot","dt","ratio"])
print("discharges whose file ends with the beam emission at its beam-off level (>=3 rows):",len(E),"; tail rows",int(E.tail_rows.sum()),"; with a reported T_i",int(E.with_ti.sum()),"(%.3f)"%(E.with_ti.sum()/E.tail_rows.sum()))
print("tail length (rows): quantiles 25,50,75,max:",np.quantile(E.tail_rows,[.25,.5,.75,1.0]))
for lo,hi in [(0,0.015),(0.015,0.035),(0.035,0.055),(0.055,0.105),(0.105,0.205),(0.205,0.505),(0.505,9)]:
    a=A[(A.dt>lo)&(A.dt<=hi)]; r=a.ratio.dropna()
    if len(a)==0: continue
    print("  %4.0f-%4.0f ms after the last beam-on row: rows %4d, with T_i %.3f, T_i / level before: median %.3f [quartiles %.3f, %.3f], share within 0.67..1.5: %.3f"%(lo*1e3,hi*1e3,len(a),a.ratio.notna().mean(),r.median() if len(r) else np.nan,r.quantile(.25) if len(r) else np.nan,r.quantile(.75) if len(r) else np.nan,((r>0.67)&(r<1.5)).mean() if len(r) else np.nan))
print("longest tails:",E.sort_values("tail_rows",ascending=False).head(8).to_string(index=False))
d=g[g.shot==31747]; d=d[(d.time>7.93)&(d.time<8.50)]
bb=d[bes_cols].values.mean(1)
print("31747: rows after 8.0 s: %d, with T_i %d; T_i median %.0f eV [quartiles %.0f, %.0f]; before 8.0 s (7.93-8.0): median %.0f eV; BES level before %.3f, after %.3f"%((d.time>8.0).sum(),np.isfinite(d.CES_TI[d.time>8.0]).sum(),np.nanmedian(d.CES_TI[d.time>8.0]),np.nanquantile(d.CES_TI[d.time>8.0],.25),np.nanquantile(d.CES_TI[d.time>8.0],.75),np.nanmedian(d.CES_TI[d.time<8.0]),np.median(bb[d.time.values<8.0]),np.median(bb[d.time.values>8.0])))
