import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1])
shot = g["shot"].values; tm = g["time"].values
ti = g["CES_TI"].values.astype(float); vt = g["CES_VT"].values.astype(float)
bes_cols = [c for c in g.columns if c.startswith("BES_")]; ecei_cols=[c for c in g.columns if c.startswith("ECEI_")]
bes = g[bes_cols].values.mean(1); ecei = g[ecei_cols].values.mean(1)
n=len(g)
same_prev = np.r_[False, shot[1:]==shot[:-1]]
dt_prev = np.r_[np.nan, np.diff(tm)]
dt_prev[~same_prev]=np.nan
prev_vt = np.r_[np.nan, vt[:-1]]; prev_ti=np.r_[np.nan, ti[:-1]]
held_vt = same_prev & np.isfinite(vt) & (vt==prev_vt)
fresh_vt = np.isfinite(vt) & ~held_vt
# ---- (a) per-shot classification
df = pd.DataFrame({"shot":shot,"held":held_vt,"vobs":np.isfinite(vt),"tiobs":np.isfinite(ti),"ti":ti,"bes":bes})
ps = df.groupby("shot").agg(held=("held","sum"),vobs=("vobs","sum"),tiobs=("tiobs","sum"),rows=("ti","size"),ti_med=("ti","median"),bes_med=("bes","median"))
ps["share"]=ps["held"]/ps["vobs"].clip(lower=1)
ps["vmiss"]=1-ps["vobs"]/ps["rows"]
print("shots by held share: <0.05:",int((ps.share<0.05).sum())," 0.05-0.9:",int(((ps.share>=0.05)&(ps.share<=0.9)).sum())," >0.9:",int((ps.share>0.9).sum()))
for name,m in [("low(<0.05)",ps.share<0.05),("mid",(ps.share>=0.05)&(ps.share<=0.9)),("high(>0.9)",ps.share>0.9)]:
    s=ps[m]
    print(name,"n",len(s),"shot range",s.index.min(),s.index.max(),"median shot",int(np.median(s.index)),"Ti median of medians %.0f"%s.ti_med.median(),"Vt missing share median %.3f"%s.vmiss.median(),"rows",int(s.rows.sum()))
# shot-number distribution of high-held shots in blocks of 200
bins=np.arange(30800,32801,200)
hh=np.histogram(ps.index[ps.share>0.9],bins)[0]; aa=np.histogram(ps.index,bins)[0]
print("shot blocks",list(zip(bins[:-1],aa,hh)))
# ---- triples with 10 ms steps
i=np.arange(1,n-1)
ok = (shot[i-1]==shot[i])&(shot[i+1]==shot[i])&(np.abs(tm[i]-tm[i-1]-0.01)<1e-4)&(np.abs(tm[i+1]-tm[i]-0.01)<1e-4)&np.isfinite(ti[i-1])&np.isfinite(ti[i])&np.isfinite(ti[i+1])
i=i[ok]
print("triples",len(i))
sd = ti[i]-0.5*(ti[i-1]+ti[i+1])   # second difference/2 : centre minus mean of neighbours
lvl = np.median(np.c_[ti[i-1],ti[i],ti[i+1]],axis=1)
cut = (ti[i-1]<=3000)&(ti[i]<=3000)&(ti[i+1]<=3000)
def rob(x): 
    x=x[np.isfinite(x)]; return 1.4826*np.median(np.abs(x-np.median(x)))
print("all triples (cut): robust sd of centre-minus-neighbour-mean %.1f eV -> white sigma = that/sqrt(1.5) = %.1f eV; rms-> %.1f eV"%(rob(sd[cut]), rob(sd[cut])/np.sqrt(1.5), np.sqrt(np.mean(sd[cut]**2)/1.5)))
# (a2) held vs fresh Vt at the centre row
for name,m in [("centre Vt fresh",fresh_vt[i]),("centre Vt held",held_vt[i]),("centre Vt missing",np.isnan(vt[i]))]:
    mm=m&cut
    rel = sd[mm]/lvl[mm]
    print("%-18s n %7d  median level %.0f eV  white(robust) %.1f eV  relative(robust) %.4f  rms white %.1f eV  share |rel|>0.3: %.4f"%(name,mm.sum(),np.median(lvl[mm]),rob(sd[mm])/np.sqrt(1.5),rob(rel)/np.sqrt(1.5),np.sqrt(np.mean(sd[mm]**2)/1.5),np.mean(np.abs(rel)>0.3)))
# within-shot comparison for mixed shots
mixed = ps.index[(ps.share>0.2)&(ps.share<0.8)]
res=[]
for s in mixed:
    m=(shot[i]==s)&cut
    a=m&fresh_vt[i]; b=m&held_vt[i]
    if a.sum()>=50 and b.sum()>=50:
        res.append((s,rob(sd[a]/lvl[a]),rob(sd[b]/lvl[b]),np.median(lvl[a]),np.median(lvl[b]),np.median(bes[i][a]),np.median(bes[i][b])))
res=np.array(res)
if len(res):
    r=res[:,2]/res[:,1]
    print("within-shot (mixed shots n=%d): ratio relative white scatter held/fresh: median %.3f, IQR %.3f-%.3f, share>1: %.3f; level ratio held/fresh median %.3f; BES ratio held/fresh median %.3f"%(len(res),np.median(r),np.quantile(r,.25),np.quantile(r,.75),np.mean(r>1),np.median(res[:,4]/res[:,3]),np.median(res[:,6]/res[:,5])))
# missing / >3keV / <50 eV rates by Vt status of the row
for name,m in [("Vt fresh",fresh_vt),("Vt held",held_vt),("Vt missing",np.isnan(vt))]:
    print("%-11s rows %7d  Ti missing %.3f  Ti>3keV (of present) %.4f  Ti<50eV (of present) %.4f  median Ti %.0f  median BES %.4f"%(name,m.sum(),np.isnan(ti[m]).mean(),np.nanmean(np.where(np.isfinite(ti[m]),ti[m]>3000,np.nan)),np.nanmean(np.where(np.isfinite(ti[m]),ti[m]<50,np.nan)),np.nanmedian(ti[m]),np.median(bes[m])))
np.savez(os.path.join(os.path.dirname(sys.argv[1]),"triples.npz"),i=i,sd=sd,lvl=lvl,cut=cut)
