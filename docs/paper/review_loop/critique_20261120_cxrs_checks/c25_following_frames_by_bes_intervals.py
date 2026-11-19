import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
d=os.path.dirname(sys.argv[1])
D=pd.read_pickle(os.path.join(d,"d1all.pkl")); Z=pd.read_pickle(os.path.join(d,"d1z.pkl"))
br=Z.br.values; ok=np.isfinite(D.z.values)&np.isfinite(br)
bands={"BES below 0.9":br<0.9,"BES 0.9-0.97":(br>=0.9)&(br<0.97),"BES 0.97-1.03":(br>=0.97)&(br<1.03),"BES above 1.03":br>=1.03}
pairs=[("c2","t2","two-stage vs twin2"),("c","t","network vs twin"),("kfx","kf0","KFX vs KF0")]
rng=np.random.default_rng(21)
for bl,bm in bands.items():
    m=bm&ok
    per={}
    for r in (2,3,4):
        dd=D[(D.r.values==r)&m]
        q={"shot":dd.shot.values}
        for A,B,_ in pairs:
            for e in (A,B):
                q[e+"_y"]=(dd[e]-dd.y).values**2; q[e+"_z"]=(dd[e]-dd.z).values**2
        per[r]=pd.DataFrame(q).groupby("shot").sum()
    cols=list(per[2].columns); ix={c:i for i,c in enumerate(cols)}; arr={r:per[r].values for r in per}
    out={lab:([],[]) for _,_,lab in pairs}
    for i in range(3000):
        r=rng.choice([2,3,4]); P=arr[r]; s=P[rng.integers(0,len(P),len(P))].sum(axis=0)
        for A,B,lab in pairs:
            out[lab][0].append(1-s[ix[A+"_y"]]/s[ix[B+"_y"]]); out[lab][1].append(1-s[ix[A+"_z"]]/s[ix[B+"_z"]])
    line="%-15s rows %.3f |"%(bl,m.sum()/ok.sum())
    for A,B,lab in pairs:
        py=np.mean([1-per[r][A+"_y"].sum()/per[r][B+"_y"].sum() for r in (2,3,4)]); pz=np.mean([1-per[r][A+"_z"].sum()/per[r][B+"_z"].sum() for r in (2,3,4)])
        line+=" %s: own %+.3f [%+.3f, %+.3f], next %+.3f [%+.3f, %+.3f] |"%(lab,py,np.quantile(out[lab][0],.025),np.quantile(out[lab][0],.975),pz,np.quantile(out[lab][1],.025),np.quantile(out[lab][1],.975))
    print(line)
