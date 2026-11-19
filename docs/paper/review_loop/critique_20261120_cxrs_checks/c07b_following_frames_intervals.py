import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
D=pd.read_pickle(sys.argv[1])
ok=np.isfinite(D.gp).values
diff=np.abs((D.gp-D.y)**2-D.se_gp)
bad = ok & (diff.values > np.maximum(1.0, 0.01*D.se_gp.values))
print("rows where the replay GP differs from the scorer GP by more than 1 %% of its squared error: %d of %d (%.5f)"%(bad.sum(),ok.sum(),bad.sum()/ok.sum()))
m=(np.isfinite(D.z).values)&ok&~bad
print("rows used per re-partition:",m.sum()/3,"share of D1 %.3f"%m.mean())
names=["hold","gp","kf0","kfx","kg","t","c","t2","c2"]
pairs=[("c2","gp","two-stage vs causal GP"),("c2","t2","two-stage vs two-stage twin"),("c","t","network vs twin"),("c","gp","network vs causal GP"),("t2","gp","two-stage twin vs causal GP"),("t","gp","twin vs causal GP"),("kfx","kf0","filter with BES vs filter without"),("kg","kf0","filter, radial difference only, vs without"),("gp","hold","causal GP vs hold"),("c2","hold","two-stage vs hold"),("t2","hold","two-stage twin vs hold")]
rng=np.random.default_rng(11)
per={}
for r in (2,3,4):
    d=D[(D.r==r)&m]
    q={"shot":d.shot.values}
    for e in names:
        q[e+"_y"]=(d[e]-d.y).values**2; q[e+"_z"]=(d[e]-d.z).values**2
    per[r]=pd.DataFrame(q).groupby("shot").sum()
cols=list(per[2].columns)
pt={}
for A,B,lab in pairs:
    sy=[];sz=[];rr=[]
    for r in (2,3,4):
        s=per[r].sum(axis=0)
        sy.append(1-s[A+"_y"]/s[B+"_y"]); sz.append(1-s[A+"_z"]/s[B+"_z"]); rr.append((s[B+"_z"]-s[A+"_z"])/(s[B+"_y"]-s[A+"_y"]))
    pt[lab]=(np.mean(sy),np.mean(sz),np.mean(rr))
res={lab:{"y":[],"z":[],"ratio":[]} for _,_,lab in pairs}
arr={r:per[r].values for r in per}
ix={c:i for i,c in enumerate(cols)}
for i in range(4000):
    r=rng.choice([2,3,4]); P=arr[r]; s=P[rng.integers(0,len(P),len(P))].sum(axis=0)
    for A,B,lab in pairs:
        ay=s[ix[A+"_y"]]; by=s[ix[B+"_y"]]; az=s[ix[A+"_z"]]; bz=s[ix[B+"_z"]]
        res[lab]["y"].append(1-ay/by); res[lab]["z"].append(1-az/bz); res[lab]["ratio"].append((bz-az)/(by-ay))
print("pair | skill against the frame's own value | skill against the median of the next 2-3 frames | share of the squared-error reduction that remains")
for _,_,lab in pairs:
    v=res[lab]; p=pt[lab]
    print("%-42s %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] | %.3f [%.3f, %.3f]"%(lab,p[0],np.quantile(v["y"],.025),np.quantile(v["y"],.975),p[1],np.quantile(v["z"],.025),np.quantile(v["z"],.975),p[2],np.quantile(v["ratio"],.025),np.quantile(v["ratio"],.975)))
print("RMSE (eV) against own frame / against next-frames reference, mean of r2-r4:")
for e in names:
    a=np.mean([np.sqrt(per[r][e+"_y"].sum()/((D.r==r)&m).sum()) for r in (2,3,4)]); b=np.mean([np.sqrt(per[r][e+"_z"].sum()/((D.r==r)&m).sum()) for r in (2,3,4)])
    print("   %-5s %.1f / %.1f"%(e,a,b))
# same test restricted to rows whose own value is not a >0.3 single-frame excursion (to see whether the loss is all on those rows)
S=D.S.values
for nm,mm in [("rows with |S|<=0.3",m&(np.abs(S)<=0.3)),("rows with |S|>0.3",m&(np.abs(S)>0.3))]:
    def sse(e,ref): return ((D.loc[mm,e]-D.loc[mm,ref])**2).sum()
    print(nm,"share of rows %.3f"%(mm.sum()/m.sum()))
    for A,B,lab in pairs[:3]+pairs[6:7]:
        print("   %-40s own %+.3f | next frames %+.3f | ratio %.3f"%(lab,1-sse(A,"y")/sse(B,"y"),1-sse(A,"z")/sse(B,"z"),(sse(B,"z")-sse(A,"z"))/(sse(B,"y")-sse(A,"y"))))
