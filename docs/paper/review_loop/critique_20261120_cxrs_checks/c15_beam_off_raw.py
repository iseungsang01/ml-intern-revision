import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
R=pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
off=R[R.rmax<0.4]
raw_shots=[31097,31114,31357,31359,31747,31873,31902,31921,31923,32027,32092,32097]
sel=off[off.shot.isin(raw_shots)]
print("beam-off-like rows in the 12 discharges with raw streams:",len(sel), sel.shot.value_counts().to_dict())
rows=[]
for shot in sorted(sel.shot.unique()):
    z=np.load(os.path.join(repo,"data/raw_mus/.cache/s%d_BES.npz"%shot),allow_pickle=True)
    import json
    x=z["x"]; meta=json.loads(str(z["meta"].item() if z["meta"].shape==() else z["meta"]))
    t0=meta["t0"]; dt=1.0/meta["fs_hz"]
    lvl=x.mean(1)
    for _,r in sel[sel.shot==shot].iterrows():
        i0=int(round((r.t-t0)/dt)); n5=int(round(0.005/dt)); n10=int(round(0.010/dt)); n200=int(round(0.2/dt))
        if i0-n200<0 or i0+n10>len(lvl): 
            rows.append((shot,r.t,np.nan,np.nan,np.nan,r.ti,r.tbefore)); continue
        base=np.median(lvl[i0-n200:i0-int(round(0.02/dt))])
        f5=np.mean(lvl[i0:i0+n5]<0.5*base); f10=np.mean(lvl[i0:i0+n10]<0.5*base); m5=np.mean(lvl[i0:i0+n5])/base
        rows.append((shot,r.t,f5,f10,m5,r.ti,r.tbefore))
T=pd.DataFrame(rows,columns=["shot","t","share of first 5 ms below half level","share of 10 ms below half level","mean level first 5 ms / level before","T_i reported","median T_i before"])
pd.set_option("display.width",250)
print(T.to_string(index=False,float_format=lambda v:"%.3f"%v))
