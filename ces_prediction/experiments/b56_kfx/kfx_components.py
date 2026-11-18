"""Unregistered look (read-only, CPU): a robust state-space filter with an exogenous BES input.

State x_t = log T_i (random walk), observation log y_t = x_t + e_t, Huber-weighted update.
  predict: x- = x + b_c * dc_t + b_g * dg_t ;  P- = P + q * dt/10ms
  output : exp(x-)  (the estimate of row t from CES values before t and BES up to t)
  update : v = log y - x-, S = P- + r, w = min(1, k sqrt(S)/|v|), x = x- + (P-/S) w v, P = (1 - w P-/S) P-
c = mean of the logs of the nine BES channels (common mode), g = log radial difference (col 07 - col 05),
as in B.51.  KF0: b = 0 (CES history only).  KFX: b free.  5 or 3 parameters, fitted on the training
discharges of each pooled fold of repartition R by minimising the eV mean squared error of the output
on observed rows (values above 3 keV are missing).  Scored on the identical test rows of the pooled run
(data/.b39p_v2_cut_r{R}_f{k}/comparison_errors_test.npz), beside the estimator, its twin and the causal GP.
Run from the repo root.
"""
import glob
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
import os
OUT_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"kfx_rows_r{R}.npz")
CUT = 3000.0
SEG = 0.5
files = sorted(glob.glob("data/s*.csv"))
hdr = pd.read_csv(files[0], nrows=0).columns.tolist()
BES = [c for c in hdr if c.startswith("BES_")]
C05 = [c for c in BES if c[6:8] == "05"]
C07 = [c for c in BES if c[6:8] == "07"]

# ---------------------------------------------------------------- grid per file
grid = []
for f in files:
    df = pd.read_csv(f)
    df = df.iloc[1:-1].reset_index(drop=True)
    t = df["time"].to_numpy(float)
    ti = df["CES_TI"].to_numpy(float).copy()
    ti[ti > CUT] = np.nan
    X = df[BES].to_numpy(float)
    L = np.log(np.where(X > 0, X, np.nan))
    c = np.nanmean(L, 1)
    g = np.log(np.clip(df[C07].to_numpy(float), 1e-9, None)).mean(1) - np.log(np.clip(df[C05].to_numpy(float), 1e-9, None)).mean(1)
    grid.append(dict(t=t, ti=ti, c=c, g=g, ti_raw=df["CES_TI"].to_numpy(float)))
NF = len(grid)
TMAX = max(len(d["t"]) for d in grid)


def pad(key, fill=np.nan):
    out = np.full((NF, TMAX), fill)
    for i, d in enumerate(grid):
        out[i, :len(d[key])] = d[key]
    return out


Y = pad("ti")
T = pad("t")
Cc = pad("c")
Gg = pad("g")
DT = np.diff(T, axis=1, prepend=np.nan)
RESET = ~(DT < SEG)                       # first row of a segment (or padding)
dC = np.diff(Cc, axis=1, prepend=np.nan)
dG = np.diff(Gg, axis=1, prepend=np.nan)
dC[RESET | ~np.isfinite(dC)] = 0.0
dG[RESET | ~np.isfinite(dG)] = 0.0
STEPS = np.where(RESET | ~np.isfinite(DT), 1.0, np.clip(DT / 0.01, 0.5, 50))
LY = np.log(np.where(Y > 0, Y, np.nan))


CLIP = [np.inf]


def run(theta, rows_mask, use_x):
    lq, lr, lk = theta[:3]
    bc, bg = (theta[3], theta[4]) if use_x else (0.0, 0.0)
    q, r, k = np.exp(lq), np.exp(lr), np.exp(lk)
    n = NF
    x = np.full(n, np.nan)
    P = np.full(n, np.nan)
    out = np.full((n, TMAX), np.nan)
    for j in range(TMAX):
        res = RESET[:, j]
        x = np.where(res, np.nan, x)
        P = np.where(res, np.nan, P)
        xm = x + bc * np.clip(dC[:, j], -CLIP[0], CLIP[0]) + bg * np.clip(dG[:, j], -CLIP[0], CLIP[0])
        Pm = P + q * STEPS[:, j]
        out[:, j] = np.exp(xm)
        ly = LY[:, j]
        obs = np.isfinite(ly)
        start = obs & ~np.isfinite(xm)
        v = ly - xm
        S = Pm + r
        w = np.minimum(1.0, k * np.sqrt(S) / np.maximum(np.abs(v), 1e-12))
        K = Pm / S
        upd = obs & np.isfinite(xm)
        x = np.where(upd, xm + K * w * v, xm)
        P = np.where(upd, (1 - w * K) * Pm, Pm)
        x = np.where(start, ly, x)
        P = np.where(start, r, P)
    return out


def loss(theta, fidx, use_x):
    out = run(theta, None, use_x)[fidx]
    y = Y[fidx]
    m = np.isfinite(y) & np.isfinite(out)
    return float(np.mean((out[m] - y[m]) ** 2))


# ---------------------------------------------------------------- map npz rows -> grid rows (value match)
def map_rows(fi, rows, yv):
    """npz rows are indices into the file's rows with >= 1 target after cleaning; match the T_i values in
    order against the grid's observed T_i (subsequence match)."""
    ti = grid[fi]["ti"]
    cand = np.flatnonzero(np.isfinite(ti))
    order = np.argsort(rows)
    gi = np.full(len(rows), -1)
    p = 0
    for o in order:
        while p < len(cand) and abs(ti[cand[p]] - yv[o]) > 1e-2:
            p += 1
        if p < len(cand):
            gi[o] = cand[p]
            p += 1
    return gi



CLIP[0] = 0.2
rng = np.random.default_rng(3)
res = {"K0": [], "Kc": [], "Kg": [], "KX": [], "G": [], "fi": []}
for k in range(5):
    a = np.load(f"data/.b39p_v2_cut_r{R}_f{k}/comparison_errors_test.npz")
    test_f = np.unique(a["CES_TI_shot"])
    train_f = np.setdiff1d(np.arange(NF), test_f)
    f0 = minimize(lambda th: loss(np.r_[th, 0, 0], train_f, False), np.array([np.log(0.002), np.log(0.01), np.log(2.0)]),
                  method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-1))
    fc = minimize(lambda b_: loss(np.r_[f0.x, b_[0], 0.0], train_f, True), np.zeros(1), method="Nelder-Mead", options=dict(maxiter=200, xatol=1e-4, fatol=1e-1))
    fg = minimize(lambda b_: loss(np.r_[f0.x, 0.0, b_[0]], train_f, True), np.zeros(1), method="Nelder-Mead", options=dict(maxiter=200, xatol=1e-4, fatol=1e-1))
    fx = minimize(lambda b_: loss(np.r_[f0.x, b_], train_f, True), np.zeros(2), method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-4, fatol=1e-1))
    print(f"fold {k}: b_c only {fc.x.round(3)}, b_g only {fg.x.round(3)}, both {fx.x.round(3)}", flush=True)
    outs = {"K0": run(np.r_[f0.x, 0, 0], None, False), "Kc": run(np.r_[f0.x, fc.x[0], 0.0], None, True),
            "Kg": run(np.r_[f0.x, 0.0, fg.x[0]], None, True), "KX": run(np.r_[f0.x, fx.x], None, True)}
    for fi in test_f:
        m = a["CES_TI_shot"] == fi
        gi = map_rows(fi, a["CES_TI_row"][m], a["CES_TI_y_true"][m])
        ok = gi >= 0
        y = a["CES_TI_y_true"][m][ok].astype(float)
        res["fi"].append(np.full(ok.sum(), fi))
        res["G"].append(a["CES_TI_se_gp_causal"][m][ok])
        for kk, o in outs.items():
            res[kk].append((o[fi, gi[ok]] - y) ** 2)
D = {k: np.concatenate(v).astype(float) for k, v in res.items()}
uf = np.unique(D["fi"]); idx = np.searchsorted(uf, D["fi"])
S = {k: np.bincount(idx, D[k], len(uf)) for k in ("K0", "Kc", "Kg", "KX", "G")}
W = rng.multinomial(len(uf), np.ones(len(uf)) / len(uf), size=4000).astype(float)
for a_, r_ in (("Kc", "K0"), ("Kg", "K0"), ("KX", "K0"), ("KX", "Kc"), ("KX", "Kg")):
    pt = 1 - S[a_].sum() / S[r_].sum(); bs = 1 - (W @ S[a_]) / (W @ S[r_]); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"skill {a_} over {r_}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("median abs:", {k: round(float(np.median(np.sqrt(D[k]))), 1) for k in S})
