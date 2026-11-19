import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
bes_cols=[c for c in g.columns if c.startswith("BES_")]
pd.set_option("display.width",200)
for shot in (31873,31747):
    d=g[g.shot==shot].reset_index(drop=True); d["bes"]=d[bes_cols].values.mean(1)
    print(shot,"last 14 rows"); print(d[["time","CES_TI","CES_VT","bes"]].tail(14).to_string(index=False,float_format=lambda v:"%.4f"%v))
# campaign-wide: rows where the beam emission is at its beam-off level relative to the discharge (below 0.3 of the discharge median of the 9-channel mean,
# and every channel below 0.4 of its own discharge median) -- how often is a T_i written, and what does it look like?
tot=0; withti=0; vals=[]; lvl=[]; nsh=0; where=[]
for shot,d in g.groupby("shot",sort=False):
    B=d[bes_cols].values.astype(float); ti=d["CES_TI"].values.astype(float); t=d["time"].values
    if len(t)<50: continue
    med=np.median(B,axis=0)
    if np.any(med<=0.02): continue
    off=(B/med).max(1)<0.3
    if not off.any(): continue
    nsh+=1
    tm=np.nanmedian(ti[~off&(ti<=3000)])
    tot+=off.sum(); withti+=np.isfinite(ti[off]).sum()
    vals+=list(ti[off][np.isfinite(ti[off])]); lvl+=[tm]*int(np.isfinite(ti[off]).sum())
    # position: fraction of the off rows in the first 10 % / last 10 % of the file
    pos=np.flatnonzero(off)/len(t); where+=list(pos)
vals=np.array(vals); lvl=np.array(lvl); where=np.array(where)
print("\nrows with every BES channel below 0.3 of its discharge median: %d in %d discharges; with a reported T_i: %d (%.3f)"%(tot,nsh,withti,withti/tot))
print("position in the file: first tenth %.3f, last tenth %.3f, elsewhere %.3f"%((where<0.1).mean(),(where>0.9).mean(),((where>=0.1)&(where<=0.9)).mean()))
print("reported T_i in those rows: quantiles 5,25,50,75,95 %%: %s eV ; relative to the discharge median T_i: %s"%(np.round(np.quantile(vals,[.05,.25,.5,.75,.95]),0),np.round(np.quantile(vals/lvl,[.05,.25,.5,.75,.95]),2)))
print("share above 3 keV %.3f ; below 100 eV %.3f ; between 0.5 and 2 times the discharge median %.3f"%((vals>3000).mean(),(vals<100).mean(),((vals/lvl>0.5)&(vals/lvl<2)).mean()))
