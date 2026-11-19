import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
g=pd.read_pickle(sys.argv[1]); repo=sys.argv[2]
bes_cols=[c for c in g.columns if c.startswith("BES_")]; ecei_cols=[c for c in g.columns if c.startswith("ECEI_")]
d=g[g.shot==32027].reset_index(drop=True)
d["bes"]=d[bes_cols].mean(1); d["ecei"]=d[ecei_cols].mean(1)
for a,b in [(5.05,5.22),(5.30,5.45)]:
    k=(d.time>=a)&(d.time<=b)
    print(d.loc[k,["time","CES_TI","CES_VT","bes","ecei"]].to_string(index=False, float_format=lambda x:"%.4f"%x))
    print()
# raw BES of 32027 around the two interruptions
z=np.load(os.path.join(repo,"data/raw_mus/.cache/s32027_BES.npz"),allow_pickle=True)
print("raw keys",z.files,[z[k].shape for k in z.files])
