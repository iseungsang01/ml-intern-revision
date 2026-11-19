import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
d=os.path.dirname(sys.argv[1])
D=pd.read_pickle(os.path.join(d,"d1all.pkl")); Z=pd.read_pickle(os.path.join(d,"d1z.pkl"))
assert len(D)==len(Z) and np.allclose(D.y.values,Z.y.values)
br=Z.br.values; ok=np.isfinite(D.z.values)&np.isfinite(br)
def sse(e,ref,m): return ((D.loc[m,e]-D.loc[m,ref])**2).sum()
print("BES sample / median of the 10 rows before | rows share | two-stage vs twin2: own frame, next frames, share of reduction remaining | network vs twin: own, next | KFX vs KF0: own, next")
for lab,m0 in [("below 0.8",br<0.8),("0.8-0.9",(br>=0.8)&(br<0.9)),("0.9-0.97",(br>=0.9)&(br<0.97)),("0.97-1.03",(br>=0.97)&(br<1.03)),("1.03-1.1",(br>=1.03)&(br<1.1)),("1.1-1.2",(br>=1.1)&(br<1.2)),("above 1.2",br>=1.2),("all",np.ones(len(D),bool))]:
    m=m0&ok
    ry=sse("t2","y",m)-sse("c2","y",m); rz=sse("t2","z",m)-sse("c2","z",m)
    print("  %-10s %.4f | %+.3f, %+.3f, %.2f | %+.3f, %+.3f | %+.3f, %+.3f"%(lab,m.sum()/ok.sum(),1-sse("c2","y",m)/sse("t2","y",m),1-sse("c2","z",m)/sse("t2","z",m),rz/ry,1-sse("c","y",m)/sse("t","y",m),1-sse("c","z",m)/sse("t","z",m),1-sse("kfx","y",m)/sse("kf0","y",m),1-sse("kfx","z",m)/sse("kf0","z",m)))
