import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
D=pd.read_pickle(sys.argv[1])
m=np.isfinite(D.z).values&np.isfinite(D.gp).values
d=D[m]
for A,B,lab in [("c2","t2","two-stage minus two-stage twin"),("c","t","network minus twin"),("kfx","kf0","KFX minus KF0"),("t2","gp","two-stage twin minus causal GP")]:
    dl=(d[A]-d[B]).values
    b0=np.sum(dl*(d.y-d[B]).values)/np.sum(dl**2); b1=np.sum(dl*(d.z-d[B]).values)/np.sum(dl**2)
    print("%-34s rms correction %.1f eV | slope on the own frame %.3f | slope on the next-frames median %.3f | ratio %.3f | (2*b1-1)/(2*b0-1) = %.3f"%(lab,np.sqrt(np.mean(dl**2)),b0,b1,b1/b0,(2*b1-1)/(2*b0-1)))
# how much of the own-frame departure (y - z) is what the correction lines up with
fs=(d.y-d.z).values
for A,B,lab in [("c2","t2","two-stage minus two-stage twin"),("c","t","network minus twin"),("kfx","kf0","KFX minus KF0")]:
    dl=(d[A]-d[B]).values
    print("%-34s corr(correction, y - z) = %.3f ; corr(correction, z - reference estimator) = %.3f"%(lab,np.corrcoef(dl,fs)[0,1],np.corrcoef(dl,(d.z-d[B]).values)[0,1]))
print("rms of y - z (frame-specific departure plus the noise of z): %.1f eV ; rows %d"%(np.sqrt(np.mean(fs**2)),len(d)))
