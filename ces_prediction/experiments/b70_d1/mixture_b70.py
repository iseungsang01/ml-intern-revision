"""B.70 D1-6 (PREREGISTRATION_B70.md section 4 and Addendum 1): the good-and-bad Gaussian mixture of Box and Tiao
(1968), as used for KSTAR profile fitting by Kim et al. (2026, Nucl. Fusion 66 096038), as a scoring-only flag.

Residual of a measured T_i value at grid row i:
    r_i = log T_i - median{ log T_j : j in [i-5, i+5], j in the same block as i, T_j finite }
The window is the centred 11 grid rows (10 ms) of the row's block (blocks cut at a time step >= 0.5 s or <= 0, as
seq_data / run_b30.block_spans); it is truncated at the block edges (no padding, no reflection) and holds the row
itself.  The residual is defined when the window holds at least three finite values; otherwise the row has no
residual, does not enter the fit and is never flagged.  The first and last grid row of every file are excluded.

Mixture: two zero-mean Gaussians (good: sd s_g; bad: sd s_b > s_g) and the bad fraction w, fitted by EM on the
residuals of the TRAINING discharges of a fold; a row is flagged when its posterior bad probability exceeds 0.5.
"""
from __future__ import annotations

import numpy as np

HALF = 5            # +-5 rows
MIN_FINITE = 3      # finite values in the window for a residual to be defined
GAP = 0.5           # s, dataset.STUCK_GAP_SECONDS / seq_data block rule
EM_TOL, EM_MAX = 1e-12, 5000


def block_ids(t):
    d = np.diff(t)
    return np.concatenate(([0], np.cumsum((d >= GAP) | (d <= 0))))


def residuals(t, ti):
    """(residual, defined) per grid row of one file; t in s, ti in eV (NaN where not measured or withheld)."""
    n = len(t)
    res = np.full(n, np.nan)
    if n < 3:
        return res, np.zeros(n, bool)
    blk = block_ids(t.astype(np.float64))
    lg = np.full(n, np.nan)
    ok = np.isfinite(ti) & (ti > 0)
    lg[ok] = np.log(ti[ok].astype(np.float64))
    idx = np.arange(n)
    for b in np.unique(blk):
        rows = idx[blk == b]
        a, e = rows[0], rows[-1] + 1
        seg = lg[a:e]
        m = len(seg)
        # windows truncated at the block edges: pad with NaN, take nanmedian over 11 columns
        pad = np.concatenate([np.full(HALF, np.nan), seg, np.full(HALF, np.nan)])
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * HALF + 1)[:m]
        cnt = np.isfinite(win).sum(1)
        good = np.isfinite(seg) & (cnt >= MIN_FINITE)
        if good.any():
            with np.errstate(all="ignore"):
                med = np.nanmedian(win[good], axis=1)
            res[a:e][good] = seg[good] - med
    keep = np.isfinite(res)
    keep[0] = keep[-1] = False          # first and last grid row of the file (D1 item 2)
    res[~keep] = np.nan
    return res, keep


def npdf(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))


def fit_em(r):
    """Zero-mean two-Gaussian mixture by EM.  Returns dict(w, s_g, s_b, loglik, iterations, n)."""
    r = np.asarray(r, np.float64)
    r = r[np.isfinite(r)]
    mad = float(np.median(np.abs(r)))
    s_g = 1.4826 * mad if mad > 0 else float(np.std(r)) * 0.5
    s_b, w = 10.0 * s_g, 0.01
    r2 = r * r
    prev = -np.inf
    it = 0
    for it in range(1, EM_MAX + 1):
        pg = (1 - w) * npdf(r, s_g)
        pb = w * npdf(r, s_b)
        tot = pg + pb
        ll = float(np.sum(np.log(tot)))
        post = pb / tot
        w = float(post.mean())
        s_b = float(np.sqrt(np.sum(post * r2) / np.sum(post)))
        s_g = float(np.sqrt(np.sum((1 - post) * r2) / np.sum(1 - post)))
        if abs(ll - prev) <= EM_TOL * max(1.0, abs(ll)):
            break
        prev = ll
    if s_b < s_g:                         # keep the labels: bad = wide
        s_g, s_b, w = s_b, s_g, 1 - w
    return {"w": w, "s_g": s_g, "s_b": s_b, "loglik": ll, "iterations": it, "n": int(len(r)),
            "init": {"s_g": 1.4826 * mad, "s_b": 10.0 * 1.4826 * mad, "w": 0.01}}


def posterior_bad(r, p):
    r = np.asarray(r, np.float64)
    out = np.full(r.shape, np.nan)
    ok = np.isfinite(r)
    pg = (1 - p["w"]) * npdf(r[ok], p["s_g"])
    pb = p["w"] * npdf(r[ok], p["s_b"])
    out[ok] = pb / (pg + pb)
    return out


def single_frames(flag, ti, t):
    """A flagged row is a single frame when the previous and the next measured value of its block are not flagged
    (a missing neighbour, i.e. the block edge, counts as not flagged)."""
    blk = block_ids(t.astype(np.float64))
    meas = np.flatnonzero(np.isfinite(ti))
    out = np.zeros(len(ti), bool)
    for pos, i in enumerate(meas):
        if not flag[i]:
            continue
        prev_f = pos > 0 and blk[meas[pos - 1]] == blk[i] and flag[meas[pos - 1]]
        next_f = pos + 1 < len(meas) and blk[meas[pos + 1]] == blk[i] and flag[meas[pos + 1]]
        out[i] = not (prev_f or next_f)
    return out
