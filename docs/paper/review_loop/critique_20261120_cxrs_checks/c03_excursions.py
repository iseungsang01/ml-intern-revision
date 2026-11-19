import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
g = pd.read_pickle(sys.argv[1])
shot = g["shot"].values; tm = g["time"].values
ti = g["CES_TI"].values.astype(float)
bes_cols = [c for c in g.columns if c.startswith("BES_")]; ecei_cols=[c for c in g.columns if c.startswith("ECEI_")]
B = g[bes_cols].values.astype(float); E = g[ecei_cols].values.astype(float)
bes = B.mean(1); ecei = E.mean(1)
z = np.load(os.path.join(os.path.dirname(sys.argv[1]),"triples.npz")); i=z["i"]
# single-frame log excursion of the centre row relative to the geometric mean of the neighbours (inclusive population first)
lt = np.log(ti)
s = lt[i]-0.5*(lt[i-1]+lt[i+1])
print("triples", len(i))
for thr in [0.1,0.2,0.3,0.5,1.0]:
    up=(s>thr).mean(); dn=(s<-thr).mean()
    print("  |log excursion|>%.1f : up %.4f  down %.4f  up/down %.2f"%(thr,up,dn,up/dn))
# same, cut population (all three <= 3 keV) and all three >= 100 eV to exclude the unphysical ends
m = (ti[i-1]<=3000)&(ti[i]<=3000)&(ti[i+1]<=3000)&(ti[i-1]>=100)&(ti[i]>=100)&(ti[i+1]>=100)
print("triples within 100 eV..3 keV:", m.sum())
for thr in [0.1,0.2,0.3,0.5,1.0]:
    up=(s[m]>thr).mean(); dn=(s[m]<-thr).mean()
    print("  |log excursion|>%.1f : up %.4f  down %.4f  up/down %.2f"%(thr,up,dn,up/dn))
# is the excursion reversible? value at t+1 relative to t-1 for big excursions
big_up = m&(s>0.3); big_dn = m&(s<-0.3)
r_nb = lt[i+1]-lt[i-1]
print("big up-excursions (n=%d): median log(T[t+1]/T[t-1]) %.3f, IQR %.3f..%.3f ; centre/neighbour-mean median factor %.2f"%(big_up.sum(),np.median(r_nb[big_up]),np.quantile(r_nb[big_up],.25),np.quantile(r_nb[big_up],.75),np.exp(np.median(s[big_up]))))
print("big down-excursions (n=%d): median log(T[t+1]/T[t-1]) %.3f, IQR %.3f..%.3f ; centre/neighbour-mean median factor %.2f"%(big_dn.sum(),np.median(r_nb[big_dn]),np.quantile(r_nb[big_dn],.25),np.quantile(r_nb[big_dn],.75),np.exp(np.median(s[big_dn]))))
# same-frame fast-channel excursions (BES positive values only for log)
pos = (bes[i-1]>0)&(bes[i]>0)&(bes[i+1]>0)
sb = np.full(len(i),np.nan); sb[pos]=np.log(bes[i][pos])-0.5*(np.log(bes[i-1][pos])+np.log(bes[i+1][pos]))
pe = (ecei[i-1]>0)&(ecei[i]>0)&(ecei[i+1]>0)
se = np.full(len(i),np.nan); se[pe]=np.log(ecei[i][pe])-0.5*(np.log(ecei[i-1][pe])+np.log(ecei[i+1][pe]))
mm = m&np.isfinite(sb)
print("Spearman(Ti excursion, BES excursion) same frame: %.4f (n=%d)"%(spearmanr(s[mm],sb[mm]).correlation,mm.sum()))
me = m&np.isfinite(se)
print("Spearman(Ti excursion, ECEI excursion) same frame: %.4f (n=%d)"%(spearmanr(s[me],se[me]).correlation,me.sum()))
# conditional: BES excursion sign/size vs Ti excursion
q = np.nanquantile(sb[mm],[0.01,0.05,0.25,0.75,0.95,0.99])
print("BES excursion quantiles 1,5,25,75,95,99%:",np.round(q,3))
bins=[-np.inf,q[0],q[1],q[2],q[3],q[4],q[5],np.inf]
lab=["<1%","1-5%","5-25%","25-75%","75-95%","95-99%",">99%"]
for k in range(7):
    c = mm&(sb>bins[k])&(sb<=bins[k+1])
    print("  BES excursion bin %-7s n %6d: median Ti excursion %+.4f, mean %+.4f, share Ti up>0.3 %.4f, down<-0.3 %.4f"%(lab[k],c.sum(),np.median(s[c]),np.mean(s[c]),(s[c]>0.3).mean(),(s[c]<-0.3).mean()))
# What do BES and ECEI do on the frames with big Ti excursions?
for name,c in [("Ti up>0.3",big_up),("Ti down<-0.3",big_dn),("|Ti exc|<0.05",m&(np.abs(s)<0.05))]:
    cb=c&np.isfinite(sb); ce=c&np.isfinite(se)
    print("  %-14s n %6d: BES excursion median %+.4f (share<-0.1: %.3f, >+0.1: %.3f) ; ECEI excursion median %+.4f"%(name,c.sum(),np.median(sb[cb]),(sb[cb]<-0.1).mean(),(sb[cb]>0.1).mean(),np.median(se[ce])))
# spikes by Ti band: share of values in each band that exceed 1.5x (2x) the median of the 10 neighbours (+-5 rows, same shot)
n=len(ti)
ser = pd.Series(ti)
grp = pd.Series(shot)
med11 = ser.groupby(grp).transform(lambda x: x.rolling(11,center=True,min_periods=4).median()).values
ratio = ti/med11
bands=[(0,25),(25,50),(50,100),(100,300),(300,1000),(1000,1500),(1500,2000),(2000,2500),(2500,3000),(3000,4000),(4000,6000),(6000,20000)]
print("band(eV)      n     median ratio to 11-row median   share>1.5x   share>2x   share<0.67x  share<0.5x")
for a,b in bands:
    c=np.isfinite(ti)&(ti>=a)&(ti<b)&np.isfinite(ratio)
    print("%6d-%-6d %7d   %.2f   %.3f   %.3f   %.3f   %.3f"%(a,b,c.sum(),np.median(ratio[c]),(ratio[c]>1.5).mean(),(ratio[c]>2).mean(),(ratio[c]<0.67).mean(),(ratio[c]<0.5).mean()))
c=np.isfinite(ratio)&(ti<=3000)
print("at or below 3 keV: rows with value > 2x local median: %d (%.4f); >1.5x: %d (%.4f); < 0.5x: %d (%.4f); <0.67x: %d (%.4f)"%((ratio[c]>2).sum(),(ratio[c]>2).mean(),(ratio[c]>1.5).sum(),(ratio[c]>1.5).mean(),(ratio[c]<0.5).sum(),(ratio[c]<0.5).mean(),(ratio[c]<0.67).sum(),(ratio[c]<0.67).mean()))
