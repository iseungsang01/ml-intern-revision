import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
d=os.path.dirname(sys.argv[1])
D=pd.read_pickle(os.path.join(d,"d1all.pkl")); Z=pd.read_pickle(os.path.join(d,"d1z.pkl"))
br=Z.br.values; ok=np.isfinite(D.z.values)&np.isfinite(br)
def red(A,B,ref,m): return ((D.loc[m,B]-D.loc[m,ref])**2).sum()-((D.loc[m,A]-D.loc[m,ref])**2).sum()
bands={"BES below 0.9":br<0.9,"BES 0.9-0.97":(br>=0.9)&(br<0.97),"BES 0.97-1.03":(br>=0.97)&(br<1.03),"BES above 1.03":br>=1.03}
for A,B,lab in [("c2","t2","two-stage vs twin2"),("c","t","network vs twin"),("kfx","kf0","KFX vs KF0")]:
    ty=red(A,B,"y",ok); tz=red(A,B,"z",ok)
    print(lab,": squared-error reduction per row against own frame %.0f eV^2, against next frames %.0f eV^2"%(ty/ok.sum(),tz/ok.sum()))
    for bl,bm in bands.items():
        m=bm&ok
        print("   %-15s rows %.3f | share of the own-frame reduction %.3f | reduction against next frames as a share of the total own-frame reduction %+.3f"%(bl,m.sum()/ok.sum(),red(A,B,"y",m)/ty,red(A,B,"z",m)/ty))
