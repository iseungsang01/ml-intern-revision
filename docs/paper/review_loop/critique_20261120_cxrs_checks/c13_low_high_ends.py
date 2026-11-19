import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
D=pd.read_pickle(sys.argv[1]); y=D.y.values
se={k:(D[k]-D.y).values**2 for k in ["c2","t2","c","t","hold","kfx","kf0"]}
se["gp"]=D.se_gp.values
tot={k:v.sum() for k,v in se.items()}
print("D1 scored rows per re-partition:",len(D)/3)
for lab,m in [("y < 50 eV",y<50),("y < 94 eV (Doppler FWHM below the 1.14 A instrument width)",y<94),("y < 200 eV",y<200),("y >= 200 eV",y>=200),("y > 2000 eV",y>2000),("y > 1500 eV",y>1500)]:
    print("%-58s rows %.4f | SSE share two-stage %.3f, GP %.3f | two-stage vs GP inside %+.3f | two-stage vs twin2 inside %+.3f | relative RMS error of two-stage %.2f"%(lab,m.mean(),se["c2"][m].sum()/tot["c2"],se["gp"][m].sum()/tot["gp"],1-se["c2"][m].sum()/se["gp"][m].sum(),1-se["c2"][m].sum()/se["t2"][m].sum(),np.sqrt(np.mean(se["c2"][m]/y[m]**2))))
# median relative error by level
for lo,hi in [(0,50),(50,100),(100,200),(200,500),(500,1000),(1000,3000)]:
    m=(y>=lo)&(y<hi)
    print("  y in [%d,%d): rows %.3f, median |err|/y two-stage %.3f, hold %.3f, twin2 %.3f"%(lo,hi,m.mean(),np.median(np.sqrt(se["c2"][m])/y[m]),np.median(np.sqrt(se["hold"][m])/y[m]),np.median(np.sqrt(se["t2"][m])/y[m])))
