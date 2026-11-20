"""N1 (unregistered, after the fact): how much of C1 (learned history-only estimator
over the causal GP, target L5, design V1) does a trivial robust causal filter already give?

Reads only stored B.73 arrays (data/.b73/rows.npz and data/.b73/trees/V1_r{r}_k*__T0__L5.npz).
Estimators compared on identical rows (rows where T0, GP, hold, m5 and tg_L5 are all finite):
  hold, causal GP, m5 (median of the last five measured values, stored by B.73), T0 (B.73 tree).
Interval: discharge bootstrap (1,000 draws), one level, re-partition r fixed. ASCII output.
CPU: numpy only, single thread.
"""
import glob
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

R = np.load("data/.b73/rows.npz")
y = R["tg_L5"]
shot = R["shot"]
cols = {"hold": R["hold"], "gp": R["gp"], "m5": R["m5"]}


def run(rep):
    t0 = np.full(y.shape, np.nan)
    files = sorted(glob.glob("data/.b73/trees/V1_%s_k*__T0__L5.npz" % rep))
    for f in files:
        d = np.load(f)
        t0[d["idx"]] = d["est"]
    est = dict(cols)
    est["T0"] = t0
    ok = np.isfinite(y)
    for v in est.values():
        ok &= np.isfinite(v)
    print("rep %s: folds %d, rows %d, discharges %d" % (rep, len(files), ok.sum(), len(np.unique(shot[ok]))))
    se = {k: (v[ok] - y[ok]) ** 2 for k, v in est.items()}
    s = shot[ok]
    us, inv = np.unique(s, return_inverse=True)
    sums = {k: np.bincount(inv, weights=v, minlength=len(us)) for k, v in se.items()}
    cnt = np.bincount(inv, minlength=len(us)).astype(float)
    rng = np.random.default_rng(20261120)
    B = 1000
    draws = rng.integers(0, len(us), size=(B, len(us)))
    for k in se:
        print("  RMSE %-4s %.1f eV" % (k, np.sqrt(se[k].mean())))

    def skill(a, b):
        pt = 1 - se[a].mean() / se[b].mean()
        bs = []
        for dr in draws:
            bs.append(1 - sums[a][dr].sum() / sums[b][dr].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return pt, lo, hi

    for a, b in [("m5", "gp"), ("T0", "gp"), ("T0", "m5"), ("gp", "hold"), ("m5", "hold")]:
        pt, lo, hi = skill(a, b)
        print("  skill %s over %s: %+.3f [%+.3f, %+.3f]" % (a, b, pt, lo, hi))
    pt_t, _, _ = skill("T0", "gp")
    pt_m, _, _ = skill("m5", "gp")
    print("  share of T0-over-GP MSE reduction already given by m5: %.2f" % (pt_m / pt_t if pt_t else float("nan")))


for rep in sys.argv[1:] or ["r2"]:
    run(rep)
