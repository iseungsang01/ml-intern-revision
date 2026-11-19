import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
d=os.path.dirname(sys.argv[1])
D=pd.read_pickle(os.path.join(d,"d1all.pkl")); Z=pd.read_pickle(os.path.join(d,"d1z.pkl")); g=pd.read_pickle(sys.argv[1])
assert len(D)==len(Z) and np.allclose(D.y.values,Z.y.values)
br=Z.br.values
se=lambda e,m:((D.loc[m,e]-D.loc[m,"y"])**2).sum()
tot=se("c2",np.ones(len(D),bool))
print("D1 scored rows per re-partition with a BES ratio:",np.isfinite(br).sum()/3)
for lab,m in [("BES sample below 0.4 of the 10 rows before",br<0.4),("0.4-0.8",(br>=0.4)&(br<0.8)),("0.8-0.9",(br>=0.8)&(br<0.9)),("0.9-1.1",(br>=0.9)&(br<1.1)),("1.1-1.2",(br>=1.1)&(br<1.2)),("above 1.2",br>=1.2)]:
    print("  %-44s rows/rep %8.0f (%.4f) | share of two-stage SSE %.3f | two-stage vs GP %+.3f | two-stage vs twin2 %+.3f | network vs twin %+.3f | share of the (twin2 - two-stage) reduction %.3f"%(lab,m.sum()/3,m.mean(),se("c2",m)/tot,1-se("c2",m)/D.loc[m,"se_gp"].sum(),1-se("c2",m)/se("t2",m),1-se("c",m)/se("t",m),(se("t2",m)-se("c2",m))/(se("t2",np.ones(len(D),bool))-tot)))
# the tail of 31747 after 8.0 s: are those values scored rows of D1?
a=g[(g.shot==31747)&(g.time>8.0)].CES_TI.dropna().values
yy=D[(D.shot==31747)&(D.r==2)].y.values
hit=sum(np.any(np.isclose(yy,v,rtol=1e-6,atol=1e-3)) for v in a)
print("31747 after 8.0 s: %d reported values, of them %d are scored rows of D1 (re-partition r2)"%(len(a),hit))
m=(D.shot.values==31747)&(D.r.values==2)
tail=np.array([np.any(np.isclose(a,v,rtol=1e-6,atol=1e-3)) for v in D.y.values[m]])
dd=D[m][tail]
print("   on those rows: RMSE two-stage %.1f, twin2 %.1f, causal GP %.1f, hold %.1f eV"%(np.sqrt(((dd.c2-dd.y)**2).mean()),np.sqrt(((dd.t2-dd.y)**2).mean()),np.sqrt(dd.se_gp.mean()),np.sqrt(((dd.hold-dd.y)**2).mean())))
