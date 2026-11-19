import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1])
ages=[]; status=[]  # status: 0 before first absent frame of the segment, 1 between two, 2 after the last
tot=0
for shot,d in g.groupby("shot",sort=False):
    t=d["time"].values; ti=d["CES_TI"].values.astype(float)
    n=len(t)
    if n<2: 
        continue
    dt=np.diff(t)
    seg=np.r_[0,np.cumsum(dt>=0.5)]
    for s in np.unique(seg):
        idx=np.flatnonzero(seg==s); tt=t[idx]; yy=ti[idx]
        d2=np.diff(tt); ab=np.flatnonzero((d2>0.015)&(d2<0.045))   # absent step between idx[ab] and idx[ab+1]
        ta_before=tt[ab] if len(ab) else np.array([])   # stamp of the last frame before each absent frame
        tb_after=tt[ab+1] if len(ab) else np.array([])
        for k in range(len(tt)):
            if not np.isfinite(yy[k]): continue
            tot+=1
            prev=np.flatnonzero(tb_after<=tt[k]+1e-6); nxt=np.flatnonzero(ta_before>=tt[k]-1e-6)
            if len(prev)==0: status.append(0); ages.append(np.nan)
            else:
                ages.append(tt[k]-ta_before[prev[-1]])
                status.append(1 if len(nxt) else 2)
ages=np.array(ages); status=np.array(status)
print("rows with a T_i value:",tot)
print("share before the first absent frame of their segment: %.3f ; between two absent frames: %.3f ; after the last: %.3f"%((status==0).mean(),(status==1).mean(),(status==2).mean()))
a=ages[np.isfinite(ages)]
print("for rows with an absent frame earlier in the segment (n=%d): age of the newest value an analysis that waits for the next beam-off frame could have delivered"%len(a))
edges=[0,0.05,0.1,0.3,1.0,3.0,99]
h=np.histogram(a,edges)[0]
for lo,hi,c in zip(edges[:-1],edges[1:],h): print("   %.2f-%.2f s: %.3f"%(lo,hi,c/len(a)))
print("   median age %.2f s"%np.median(a))
