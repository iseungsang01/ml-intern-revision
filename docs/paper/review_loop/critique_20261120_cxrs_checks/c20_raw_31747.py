import numpy as np, json, sys, os
repo=sys.argv[1]
z=np.load(os.path.join(repo,"data/raw_mus/.cache/s31747_BES.npz"),allow_pickle=True)
meta=json.loads(str(z["meta"].item())); x=z["x"]; t0=meta["t0"]; fs=meta["fs_hz"]
lvl=x.mean(1)
def seg(a,b): return lvl[int(round((a-t0)*fs)):int(round((b-t0)*fs))]
base=np.median(seg(7.80,7.99))
tail=seg(8.001,8.491)
print("31747 raw BES (9-channel mean): median level 7.80-7.99 s %.4f ; 8.001-8.491 s: median %.4f (ratio %.3f) ; share of raw samples below half the earlier level %.4f ; share below 0.35 %.4f"%(base,np.median(tail),np.median(tail)/base,(tail<0.5*base).mean(),(tail<0.35*base).mean()))
for a in (4.0003,5.0003,6.0003,7.0002):
    s=seg(a+0.001,a+0.009); print("  programmed notch at %.4f s: median level inside / before = %.3f"%(a,np.median(s)/np.median(seg(a-0.2,a-0.02))))
# 100 ms bins of the tail
for a in np.arange(8.0,8.45,0.1):
    s=seg(a+0.001,a+0.1); print("  %.1f-%.1f s: median ratio %.3f, share below half %.4f"%(a,a+0.1,np.median(s)/base,(s<0.5*base).mean()))
