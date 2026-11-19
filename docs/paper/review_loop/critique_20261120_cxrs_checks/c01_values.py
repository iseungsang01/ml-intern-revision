import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g = pd.read_pickle(sys.argv[1])
ti = g["CES_TI"].values; vt = g["CES_VT"].values
n = len(g)
print("grid rows", n, "Ti present", np.isfinite(ti).sum(), "missing %.2f%%" % (100*np.isnan(ti).mean()), "Vt missing %.2f%%" % (100*np.isnan(vt).mean()))
t = ti[np.isfinite(ti)]
print("Ti min/max", t.min(), t.max())
print("quantiles eV", {q: round(float(np.quantile(t,q)),1) for q in [0.001,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999]})
edges = [0,25,50,75,100,150,200,300,500,750,1000,1500,2000,2500,3000,4000,6000,10000,20000,1e9]
h,_ = np.histogram(t, bins=edges)
for a,b,c in zip(edges[:-1],edges[1:],h): print("  %7g-%-7g %7d  %.4f%%" % (a,b,c,100*c/len(t)))
print("negative or zero Ti:", (t<=0).sum())
# pile-ups: most common exact values
vc = pd.Series(t).value_counts()
print("most common exact Ti values:\n", vc.head(8))
# decimals
s = pd.Series(t).head(20).tolist(); print("sample values", s[:10])
# held values (same as previous observed within shot, consecutive rows)
shot = g["shot"].values
same_shot = np.r_[False, shot[1:]==shot[:-1]]
prev_ti = np.r_[np.nan, ti[:-1]]; prev_vt = np.r_[np.nan, vt[:-1]]
held_ti = same_shot & np.isfinite(ti) & (ti==prev_ti)
held_vt = same_shot & np.isfinite(vt) & (vt==prev_vt)
print("held Ti rows", held_ti.sum(), "held Vt rows", held_vt.sum(), "of Vt observed", np.isfinite(vt).sum(), "share %.3f" % (held_vt.sum()/np.isfinite(vt).sum()))
# Vt held: is Ti fresh at those rows?
print("rows with Vt held and Ti fresh(finite, != prev):", (held_vt & np.isfinite(ti) & (ti!=prev_ti)).sum())
print("rows with Vt held and Ti missing:", (held_vt & np.isnan(ti)).sum())
# run lengths of held Vt
runs=[]; c=0
for hflag in held_vt:
    if hflag: c+=1
    else:
        if c>0: runs.append(c)
        c=0
runs=np.array(runs)
print("held-Vt runs", len(runs), "run length quantiles", {q:int(np.quantile(runs,q)) for q in [0.5,0.75,0.9,0.99,1.0]})
# per shot share of held Vt
ps = pd.DataFrame({"shot":shot,"held":held_vt,"obs":np.isfinite(vt)}).groupby("shot").sum()
sh = (ps["held"]/ps["obs"].clip(lower=1))
print("per-shot held-Vt share quantiles", {q: round(float(sh.quantile(q)),3) for q in [0.1,0.25,0.5,0.75,0.9]}, "shots with share>0.9:", int((sh>0.9).sum()), "shots with share<0.05:", int((sh<0.05).sum()), "n shots", len(sh))
# Vt value distribution
v = vt[np.isfinite(vt)]
print("Vt quantiles km/s", {q: round(float(np.quantile(v,q)),1) for q in [0.001,0.01,0.1,0.5,0.9,0.99,0.999]}, "min", v.min(), "max", v.max())
print("Vt most common exact values:\n", pd.Series(v).value_counts().head(8))
