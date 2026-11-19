import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
g=pd.read_pickle(sys.argv[1])
shot=g["shot"].values; tm=g["time"].values
ti=g["CES_TI"].values.astype(float).copy(); vt=g["CES_VT"].values.astype(float).copy()
same_prev=np.r_[False,shot[1:]==shot[:-1]]
pv=np.r_[np.nan,vt[:-1]]; held=same_prev&np.isfinite(vt)&(vt==pv)
vt_f=vt.copy(); vt_f[held]=np.nan           # fresh rotation values only
vt_f[np.abs(vt_f)>1000]=np.nan              # the authors' 1,000 km/s rule
ti[ti>3000]=np.nan
n=len(ti); i=np.arange(1,n-1)
ok=(shot[i-1]==shot[i])&(shot[i+1]==shot[i])&(np.abs(tm[i]-tm[i-1]-0.01)<1e-4)&(np.abs(tm[i+1]-tm[i]-0.01)<1e-4)
for a in (ti,vt_f):
    ok&=np.isfinite(a[i-1])&np.isfinite(a[i])&np.isfinite(a[i+1])
i=i[ok]
print("triples with T_i and a fresh rotation value in all three rows:",len(i),"in",len(np.unique(shot[i])),"discharges")
sT=np.log(ti[i])-0.5*(np.log(ti[i-1])+np.log(ti[i+1]))
vn=0.5*(vt_f[i-1]+vt_f[i+1])
dV=vt_f[i]-vn
m=(vn>20)&(ti[i]>=100)&(ti[i-1]>=100)&(ti[i+1]>=100)
rel=dV[m]/vn[m]; s=sT[m]
print("with neighbour rotation above 20 km/s and T_i above 100 eV:",m.sum())
print("Spearman(T_i excursion, relative V excursion), same frame: %.4f"%spearmanr(s,rel).correlation)
# per-discharge distribution of the correlation
sh=shot[i][m]; rr=[]
for q in np.unique(sh):
    k=sh==q
    if k.sum()>=100: rr.append(spearmanr(s[k],rel[k]).correlation)
rr=np.array(rr)
print("per discharge (>=100 triples, n=%d): median rho %.3f, quartiles %.3f..%.3f, share negative %.3f"%(len(rr),np.median(rr),np.quantile(rr,.25),np.quantile(rr,.75),(rr<0).mean()))
bins=[(-9,-0.5),(-0.5,-0.3),(-0.3,-0.1),(-0.1,0.1),(0.1,0.3),(0.3,0.5),(0.5,9)]
print("T_i excursion bin      n     median rel. V excursion   mean    share V down>10%%   share V up>10%%")
for a,b in bins:
    k=(s>a)&(s<=b)
    print("  (%+.1f,%+.1f]   %7d   %+.4f   %+.4f   %.3f   %.3f"%(a,b,k.sum(),np.median(rel[k]),np.mean(np.clip(rel[k],-3,3)),(rel[k]<-0.1).mean(),(rel[k]>0.1).mean()))
# bootstrap over discharges for the pooled correlation and for the median V excursion on up-spikes
rng=np.random.default_rng(3)
us=np.unique(sh); idx={q:np.flatnonzero(sh==q) for q in us}
B=[];U=[];Dn=[]
for b in range(400):
    pick=rng.choice(us,len(us)); ii=np.concatenate([idx[q] for q in pick])
    B.append(spearmanr(s[ii],rel[ii]).correlation)
    U.append(np.median(rel[ii][s[ii]>0.3])); Dn.append(np.median(rel[ii][s[ii]<-0.3]))
print("pooled Spearman 95%% interval [%.3f, %.3f]; median rel. V excursion when T_i up>0.3: %.4f [%.4f, %.4f]; when T_i down<-0.3: %.4f [%.4f, %.4f]"%(np.quantile(B,.025),np.quantile(B,.975),np.median(rel[s>0.3]),np.quantile(U,.025),np.quantile(U,.975),np.median(rel[s<-0.3]),np.quantile(Dn,.025),np.quantile(Dn,.975)))
# the same for |V| excursion size: is the rotation of the frame disturbed (either sign) when T_i is?
print("share of frames with |rel V excursion|>0.2 : quiet T_i (|S|<=0.05) %.3f ; T_i up>0.3 %.3f ; T_i down<-0.3 %.3f"%((np.abs(rel[np.abs(s)<=0.05])>0.2).mean(),(np.abs(rel[s>0.3])>0.2).mean(),(np.abs(rel[s<-0.3])>0.2).mean()))
