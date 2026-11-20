"""N2 (unregistered, after the fact): does a linear blend of causal history summaries with
a handful of coefficients give most of T0's gain over the causal GP (target L5, V1 rows)?

Blend inputs (all causal, stored in data/.b73/rows.npz): causal GP, hold, m5 (median of the
last five measured values) and an intercept -> 4 coefficients. (A first version also used the
row mean of `thist`; `thist` holds five different history scalars, not five values, so that
column was meaningless and was dropped; its output is not used.) Fitted by least squares on
one half of the discharges and scored on the other half (two-way cross-fit by a seeded random
split of discharges). T0 from the stored B.73 trees (re-partition r2). Interval: discharge
bootstrap, 1,000 draws. ASCII output, numpy only.
"""
import glob
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

R = np.load("data/.b73/rows.npz")
y = R["tg_L5"]
shot = R["shot"]
th = R["thist"]
t0 = np.full(y.shape, np.nan)
for f in sorted(glob.glob("data/.b73/trees/V1_r2_k*__T0__L5.npz")):
    d = np.load(f)
    t0[d["idx"]] = d["est"]
X = np.column_stack([np.ones_like(y), R["gp"], R["hold"], R["m5"]])
ok = np.isfinite(y) & np.isfinite(t0) & np.all(np.isfinite(X), axis=1)
print("rows %d, discharges %d" % (ok.sum(), len(np.unique(shot[ok]))))
X, yy, s, t0o, gp = X[ok], y[ok], shot[ok], t0[ok], R["gp"][ok]
us = np.unique(s)
rng = np.random.default_rng(7)
half = set(rng.permutation(us)[: len(us) // 2].tolist())
a = np.array([v in half for v in s])
blend = np.empty_like(yy)
for tr, te in [(a, ~a), (~a, a)]:
    coef, *_ = np.linalg.lstsq(X[tr], yy[tr], rcond=None)
    blend[te] = X[te] @ coef
    print("coef (intercept, gp, hold, m5):", " ".join("%+.3f" % c for c in coef))
se = {"gp": (gp - yy) ** 2, "blend": (blend - yy) ** 2, "T0": (t0o - yy) ** 2}
ui, inv = np.unique(s, return_inverse=True)
sums = {k: np.bincount(inv, weights=v) for k, v in se.items()}
draws = np.random.default_rng(20261120).integers(0, len(ui), size=(1000, len(ui)))
for k in se:
    print("RMSE %-5s %.1f eV" % (k, np.sqrt(se[k].mean())))
for a_, b_ in [("blend", "gp"), ("T0", "gp"), ("T0", "blend")]:
    pt = 1 - se[a_].mean() / se[b_].mean()
    bs = [1 - sums[a_][d].sum() / sums[b_][d].sum() for d in draws]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("skill %s over %s: %+.3f [%+.3f, %+.3f]" % (a_, b_, pt, lo, hi))
