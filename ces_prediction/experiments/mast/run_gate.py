"""MAST gate (PREREGISTRATION_MAST_GATE.md): CPU, numpy/scipy/pandas only, deterministic.

Reads ONLY the shots of mast/gate_shots.txt from data/mast/act/{shot}.npz and writes
docs/paper/mast_gate.json / .md.  Run from the repo root:

    py ces_prediction/experiments/mast/run_gate.py

Every arm is linear in the visible values.  Each arm function returns (W, idx): W[h, c, k] is the
weight that the estimate for hidden block frame hid[h] on chord c puts on block frame idx[h, k]
(idx = -1: unused slot, weight 0).  Predictions and the reduced-chi-square denominator
sigma_t^2 + sum_i w_i^2 sigma_i^2 are both computed from these weights.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261116
N_BOOT = 2000
STEP_MS = 5.0
ROOT = Path(__file__).resolve().parents[3]
ACT_DIR = Path(os.environ.get("MAST_ACT_DIR", ROOT / "data" / "mast" / "act"))
META = ROOT / "data" / "mast" / "meta" / "l1_shots.parquet"
SHOTS_TXT = Path(__file__).resolve().parent / "gate_shots.txt"
OUT_JSON = ROOT / "docs" / "paper" / "mast_gate.json"
OUT_MD = ROOT / "docs" / "paper" / "mast_gate.md"

TARGETS = ("T_i", "V_i")
POPS = {"G": 0.1, "W": 0.5}
ARMS = ("persistence", "extrapolation", "running_mean", "slope_filter", "causal_gp",
        "offline_interp", "offline_gp")
CAUSAL_ARMS = ARMS[:5]
TAUS = (5.0, 10.0, 15.0, 25.0, 40.0, 70.0)
GP_ELLS = (10.0, 20.0, 40.0, 80.0, 160.0)
GP_AMPS = (0.5, 1.0, 2.0, 4.0)
Q_LEVEL_EXP = np.linspace(-2.0, 1.5, 6)
Q_SLOPE_EXP = np.linspace(-4.0, 1.0, 6)
CONST_FACTOR = 1e6      # constant-kernel variance = CONST_FACTOR x amplitude^2
MIN_TRIPLES = 12
MIN_AC_PAIRS = 8
GAUSS_MED_CHI2_1 = 0.455


# --------------------------------------------------------------------------- arms (weights)

def hidden_positions(L: int) -> np.ndarray:
    """Odd block positions j with j - 3 >= 0 and j + 1 <= L - 1."""
    return np.arange(3, L - 1, 2)


def arm_persistence(t, G, S2, hid):
    W = G[hid - 1].astype(float)[:, :, None]
    return W, (hid - 1)[:, None]


def arm_extrapolation(t, G, S2, hid):
    a = (t[hid] - t[hid - 1]) / (t[hid - 1] - t[hid - 3])
    ok = (G[hid - 1] & G[hid - 3]).astype(float)
    W = np.stack([(1.0 + a)[:, None] * ok, (-a)[:, None] * ok], axis=-1)
    return W, np.stack([hid - 1, hid - 3], axis=1)


def arm_offline_interp(t, G, S2, hid):
    ok = (G[hid - 1] & G[hid + 1]).astype(float)
    W = np.stack([0.5 * ok, 0.5 * ok], axis=-1)
    return W, np.stack([hid - 1, hid + 1], axis=1)


def arm_running_mean(t, G, S2, hid, tau):
    vis = np.arange(0, G.shape[0], 2)
    age = t[hid][:, None] - t[vis][None, :]
    E = np.where(vis[None, :] < hid[:, None], np.exp(-np.maximum(age, 0.0) / tau), 0.0)
    W = E[:, None, :] * G[vis].T[None, :, :]
    norm = W.sum(-1, keepdims=True)
    W = W / np.where(norm > 0, norm, 1.0)
    return W, np.tile(vis, (len(hid), 1))


def arm_slope_filter(t, G, S2, hid, q_level, q_slope, Y0=None):
    """Kalman filter, state (level, slope), one predict per frame interval with
    F = [[1, dt], [0, 1]], Q = diag(q_level dt, q_slope dt), dt in 5 ms steps; update at visible
    (even) frames where the sample is good.  Exact diffuse initialisation: after the first good
    sample the level is that sample and the slope is unknown; the second good sample fixes the
    slope (no large-number prior).  Y0 None: returns (W, idx) for scalar q.
    Y0 given: returns the predictions directly, shape q.shape + (H, C) (used for the grid)."""
    L, C = G.shape
    ql = np.asarray(q_level, float)[..., None]
    qs = np.asarray(q_slope, float)[..., None]
    bs = ql.shape[:-1]
    vis = np.arange(0, L, 2)
    m = 1 if Y0 is not None else len(vis)
    b0 = np.zeros(bs + (C, m))
    b1 = np.zeros(bs + (C, m))
    p11 = np.zeros(bs + (C,))
    p12 = np.zeros(bs + (C,))
    p22 = np.zeros(bs + (C,))
    n = np.zeros(C, int)          # good visible samples absorbed so far (0, 1, 2 = two or more)
    d0 = np.zeros(C)              # time since the first good sample, in steps (diffuse direction)
    hidset = set(int(j) for j in hid)
    out = []
    for i in range(L):
        if i > 0:
            dt = (t[i] - t[i - 1]) / STEP_MS
            b0 = b0 + dt * b1
            p11 = p11 + 2.0 * dt * p12 + dt * dt * p22 + ql * dt
            p12 = p12 + dt * p22
            p22 = p22 + qs * dt
            d0 = d0 + dt
        if i % 2 == 0:
            g = G[i]
            s2 = S2[i]
            c0 = g & (n == 0)
            c1 = g & (n == 1)
            c2 = g & (n == 2)
            if Y0 is not None:
                z = Y0[i][:, None]
            else:
                z = np.zeros(m)
                z[i // 2] = 1.0
            r = z - b0
            # two or more samples absorbed: standard update
            s = p11 + np.where(c2, s2, 1.0)
            k1 = np.where(c2, p11 / s, 0.0)
            k2 = np.where(c2, p12 / s, 0.0)
            # second sample: it fixes the diffuse slope
            g1 = np.where(c1, 1.0 / np.where(c1, d0, 1.0), 0.0)
            reset = (c0 | c1)[:, None]
            nb0 = np.where(reset, z + 0.0 * b0, b0 + k1[..., None] * r)
            nb1 = np.where(c0[:, None], 0.0, b1 + (k2 + g1)[..., None] * r)
            n22 = np.where(c1, g1 * g1 * p11 - 2.0 * g1 * p12 + p22 + g1 * g1 * s2,
                           np.where(c0, 0.0, p22 - k2 * p12))
            n12 = np.where(c1, g1 * s2, np.where(c0, 0.0, (1.0 - k1) * p12))
            n11 = np.where(c0 | c1, s2, (1.0 - k1) * p11)
            b0, b1, p11, p12, p22 = nb0, nb1, n11, n12, n22
            d0 = np.where(c0, 0.0, d0)
            n = np.where(g, np.minimum(n + 1, 2), n)
        elif i in hidset:
            out.append(b0)
    if not out:
        if Y0 is not None:
            return np.zeros(bs + (0, C))
        return np.zeros((0, C, m)), np.tile(vis, (0, 1))
    res = np.stack(out, axis=len(bs))
    if Y0 is not None:
        return res[..., 0]
    return res, np.tile(vis, (len(hid), 1))


def _matern32(r):
    x = math.sqrt(3.0) * r
    return (1.0 + x) * np.exp(-x)


def arm_gp(t, G, S2, hid, ell, amp, causal, mask=None):
    """GP posterior-mean weights w = k*^T (K + R)^-1, K = amp^2 Matern-3/2 + c, c = 1e6 amp^2,
    R = diag(reported error^2).  Window: the 8 visible frames before the hidden frame (causal) or
    4 visible frames on each side (offline); bad samples dropped.  The constant part is applied
    through the Sherman-Morrison identity (the same matrix, solved stably)."""
    L, C = G.shape
    H = len(hid)
    offs = -(2 * np.arange(8) + 1) if causal else np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    idx = hid[:, None] + offs[None, :]
    inw = (idx >= 0) & (idx < L)
    idc = np.clip(idx, 0, L - 1)
    tw = t[idc]
    a2 = float(amp) ** 2
    Kmat = a2 * _matern32(np.abs(tw[:, :, None] - tw[:, None, :]) / ell)
    k0 = a2 * _matern32(np.abs(t[hid][:, None] - tw) / ell)
    valid = inw[:, None, :] & np.transpose(G[idc], (0, 2, 1))
    s2w = np.transpose(S2[idc], (0, 2, 1))
    W = np.zeros((H, C, 8))
    if mask is None:
        mask = np.ones((H, C), bool)
    hh, cc = np.nonzero(mask)
    if len(hh):
        v = valid[hh, cc]
        A = Kmat[hh] * (v[:, :, None] & v[:, None, :])
        d = np.arange(8)
        A[:, d, d] += np.where(v, s2w[hh, cc], 1.0)
        rhs = np.stack([k0[hh] * v, v.astype(float)], axis=-1)
        sol = np.linalg.solve(A, rhs)
        u, one = sol[..., 0], sol[..., 1]
        c = CONST_FACTOR * a2
        beta = c * (1.0 - u.sum(-1)) / (1.0 + c * one.sum(-1))
        W[hh, cc] = u + beta[:, None] * one
    return W, np.where(inw, idx, -1)


def apply_weights(W, idx, Y0, S2):
    """Prediction sum_i w_i y_i and weight variance sum_i w_i^2 sigma_i^2, shape (H, C)."""
    use = idx >= 0
    idc = np.where(use, idx, 0)
    Wm = W * use[:, None, :]
    Yg = np.transpose(Y0[idc], (0, 2, 1))
    Sg = np.transpose(S2[idc], (0, 2, 1))
    return (Wm * Yg).sum(-1), (Wm * Wm * Sg).sum(-1)


# --------------------------------------------------------------------------- gate (a) statistic

def second_difference_s2(Y, G, k):
    """Per chord: (1.4826 MAD of d_k)^2 / 6 over triples of good samples; NaN below 12 triples."""
    L, C = Y.shape
    out = np.full(C, np.nan)
    if L <= 2 * k:
        return out
    d = Y[2 * k:] - 2.0 * Y[k:L - k] + Y[:L - 2 * k]
    ok = G[2 * k:] & G[k:L - k] & G[:L - 2 * k]
    d = np.where(ok, d, np.nan)
    n = ok.sum(0)
    keep = n >= MIN_TRIPLES
    if keep.any():
        dk = d[:, keep]
        med = np.nanmedian(dk, axis=0)
        mad = np.nanmedian(np.abs(dk - med[None, :]), axis=0)
        out[keep] = (1.4826 * mad) ** 2 / 6.0
    return out


def t_a_statistic(Y, G):
    """T_a = (S_1 - S_2 + (S_4 - S_2) / 2) / S_2 with S_k the median over chords of s2_k."""
    S = {}
    for k in (1, 2, 4):
        s2 = second_difference_s2(Y, G, k)
        S[k] = float(np.nanmedian(s2)) if np.isfinite(s2).any() else float("nan")
    if not all(np.isfinite(S[k]) for k in (1, 2, 4)) or not S[2] > 0:
        return float("nan"), S
    return (S[1] - S[2] + (S[4] - S[2]) / 2.0) / S[2], S


def regime_number(Y, G):
    """Median over chords of the lag-1 autocorrelation of 10 ms increments of the visible series."""
    Yv, Gv = Y[0::2], G[0::2]
    inc = Yv[1:] - Yv[:-1]
    ok = Gv[1:] & Gv[:-1]
    a, b = inc[:-1], inc[1:]
    okp = ok[:-1] & ok[1:]
    rs = []
    for c in range(Y.shape[1]):
        sel = okp[:, c]
        if sel.sum() < MIN_AC_PAIRS:
            continue
        x, y = a[sel, c], b[sel, c]
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
    return float(np.median(rs)) if rs else float("nan")


# --------------------------------------------------------------------------- data

def read_gate_shots():
    return [int(s.strip()) for s in SHOTS_TXT.read_text().splitlines() if s.strip()]


def load_shot(shot):
    p = ACT_DIR / f"{shot}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        d = {
            "time_ms": z["act_time"].astype(float) * 1000.0,
            "R": z["act_majorradius"].astype(float),
            "T": z["act_ss_temperature"].astype(float),
            "Te": z["act_ss_temperature_error"].astype(float),
            "V": z["act_ss_velocity"].astype(float) / 1000.0,
            "Ve": z["act_ss_velocity_error"].astype(float) / 1000.0,
            "chisq_cols": int(z["act_ss_chisq"].shape[-1]) if "act_ss_chisq" in z.files else -1,
        }
    return d


def good_masks(d, rel):
    T, Te, V, Ve = d["T"], d["Te"], d["V"], d["Ve"]
    with np.errstate(all="ignore"):
        gT = (np.isfinite(T) & np.isfinite(Te) & (T > 0) & (Te > 0) & (T < 5000.0)
              & (Te / T < rel))
        gV = gT & np.isfinite(V) & np.isfinite(Ve) & (Ve > 0)
    return {"T_i": gT, "V_i": gV}


def prepare(d, target, rel):
    """Block, scored chords and scored rows of one shot for one target and population."""
    good = good_masks(d, rel)[target]
    y = d["T"] if target == "T_i" else d["V"]
    e = d["Te"] if target == "T_i" else d["Ve"]
    frames = np.nonzero(good.sum(1) >= 10)[0]
    if len(frames) == 0:
        return {"status": "no_block"}
    a, b = int(frames[0]), int(frames[-1])
    L = b - a + 1
    if L < 20:
        return {"status": "no_block", "L_short": L}
    gb = good[a:b + 1]
    chords = np.nonzero(gb.mean(0) >= 0.9)[0]
    if len(chords) == 0:
        return {"status": "no_scored_chord", "L": L}
    G = gb[:, chords]
    Y0 = np.where(G, y[a:b + 1][:, chords], 0.0)
    S2 = np.where(G, e[a:b + 1][:, chords] ** 2, 0.0)
    t_stored = d["time_ms"][a:b + 1]
    t = STEP_MS * np.arange(L)          # nominal frame grid; stored times enter the usable rule only
    hid = hidden_positions(L)
    rows = G[hid] & G[hid - 1] & G[hid - 3] & G[hid + 1]
    R = d["R"][chords] if d["R"].shape[0] == y.shape[1] else np.full(len(chords), np.nan)
    dts = np.diff(t_stored)
    return {"status": "ok", "L": L, "t": t, "G": G, "Y0": Y0, "S2": S2, "hid": hid, "rows": rows,
            "R": R, "n_chords": len(chords), "n_rows": int(rows.sum()),
            "n_bad_dt": int((~(np.abs(dts - STEP_MS) <= 0.1)).sum()),
            "n_negative_dt": int((dts <= 0).sum()), "t_finite": bool(np.isfinite(t_stored).all())}


def arm_outputs(p, arm, params, mask=None):
    t, G, S2, hid = p["t"], p["G"], p["S2"], p["hid"]
    if arm == "persistence":
        W, idx = arm_persistence(t, G, S2, hid)
    elif arm == "extrapolation":
        W, idx = arm_extrapolation(t, G, S2, hid)
    elif arm == "offline_interp":
        W, idx = arm_offline_interp(t, G, S2, hid)
    elif arm == "running_mean":
        W, idx = arm_running_mean(t, G, S2, hid, params["tau_ms"])
    elif arm == "slope_filter":
        W, idx = arm_slope_filter(t, G, S2, hid, params["q_level"], params["q_slope"])
    elif arm == "causal_gp":
        W, idx = arm_gp(t, G, S2, hid, params["gp_causal"][0], params["gp_causal"][1], True, mask)
    elif arm == "offline_gp":
        W, idx = arm_gp(t, G, S2, hid, params["gp_offline"][0], params["gp_offline"][1], False, mask)
    else:
        raise ValueError(arm)
    return apply_weights(W, idx, p["Y0"], S2)


# --------------------------------------------------------------------------- tuning

def tune(prep_tune, log):
    """Hyperparameters by pooled MSE on TUNE scored rows (population G) for one target."""
    ps = [p for p in prep_tune if p["status"] == "ok" and p["n_rows"] > 0]
    sig = np.concatenate([np.sqrt(p["S2"][p["hid"]][p["rows"]]) for p in ps])
    scale = float(np.median(sig))
    n = int(sum(p["n_rows"] for p in ps))
    out = {"scale_median_reported_error": scale, "n_tune_rows": n, "n_tune_shots_with_rows": len(ps)}
    sse = np.zeros(len(TAUS))
    for p in ps:
        yt = p["Y0"][p["hid"]]
        for i, tau in enumerate(TAUS):
            W, idx = arm_running_mean(p["t"], p["G"], p["S2"], p["hid"], tau)
            pred, _ = apply_weights(W, idx, p["Y0"], p["S2"])
            sse[i] += ((pred - yt)[p["rows"]] ** 2).sum()
    i = int(np.argmin(sse))
    out["running_mean"] = {"grid_tau_ms": list(TAUS), "tune_mse": (sse / n).tolist(),
                           "selected_tau_ms": TAUS[i], "on_grid_edge": i in (0, len(TAUS) - 1)}
    qL = scale ** 2 * 10.0 ** Q_LEVEL_EXP
    qS = scale ** 2 * 10.0 ** Q_SLOPE_EXP
    QL, QS = np.meshgrid(qL, qS, indexing="ij")
    sse = np.zeros(QL.shape)
    for p in ps:
        pred = arm_slope_filter(p["t"], p["G"], p["S2"], p["hid"], QL, QS, Y0=p["Y0"])
        yt = p["Y0"][p["hid"]]
        sse += (((pred - yt) ** 2) * p["rows"]).sum((-1, -2))
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    out["slope_filter"] = {
        "unit": "(median reported error of TUNE rows)^2 per 5 ms step",
        "grid_q_level_over_scale2": (10.0 ** Q_LEVEL_EXP).tolist(),
        "grid_q_slope_over_scale2": (10.0 ** Q_SLOPE_EXP).tolist(),
        "tune_mse_rows_q_level_cols_q_slope": (sse / n).tolist(),
        "selected_q_level_over_scale2": float(10.0 ** Q_LEVEL_EXP[i]),
        "selected_q_slope_over_scale2": float(10.0 ** Q_SLOPE_EXP[j]),
        "selected_q_level": float(qL[i]), "selected_q_slope": float(qS[j]),
        "on_grid_edge_q_level": int(i) in (0, 5), "on_grid_edge_q_slope": int(j) in (0, 5)}
    log("  slope filter tuned")
    for name, causal in (("gp_causal", True), ("gp_offline", False)):
        sse = np.zeros((len(GP_ELLS), len(GP_AMPS)))
        for p in ps:
            yt = p["Y0"][p["hid"]]
            for a, ell in enumerate(GP_ELLS):
                for b, amp in enumerate(GP_AMPS):
                    W, idx = arm_gp(p["t"], p["G"], p["S2"], p["hid"], ell, amp * scale, causal,
                                    mask=p["rows"])
                    pred, _ = apply_weights(W, idx, p["Y0"], p["S2"])
                    sse[a, b] += ((pred - yt)[p["rows"]] ** 2).sum()
        a, b = np.unravel_index(int(np.argmin(sse)), sse.shape)
        out[name] = {"grid_length_ms": list(GP_ELLS), "grid_amp_over_scale": list(GP_AMPS),
                     "tune_mse_rows_length_cols_amp": (sse / n).tolist(),
                     "selected_length_ms": GP_ELLS[a],
                     "selected_amp_over_scale": GP_AMPS[b], "selected_amp": GP_AMPS[b] * scale,
                     "on_grid_edge_length": int(a) in (0, len(GP_ELLS) - 1),
                     "on_grid_edge_amp": int(b) in (0, len(GP_AMPS) - 1)}
        log(f"  {name} tuned")
    params = {"tau_ms": out["running_mean"]["selected_tau_ms"],
              "q_level": out["slope_filter"]["selected_q_level"],
              "q_slope": out["slope_filter"]["selected_q_slope"],
              "gp_causal": (out["gp_causal"]["selected_length_ms"], out["gp_causal"]["selected_amp"]),
              "gp_offline": (out["gp_offline"]["selected_length_ms"], out["gp_offline"]["selected_amp"])}
    return out, params


# --------------------------------------------------------------------------- statistics

def interval(point, boots, ref=0.0):
    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0 or point is None or not np.isfinite(point):
        return {"point": None, "lo": None, "hi": None, "ref": ref, "excludes_ref": None, "side": None}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    side = "above" if lo > ref else ("below" if hi < ref else "includes")
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "ref": ref,
            "excludes_ref": side != "includes", "side": side}


def boot_weighted_median(values, shot_idx, counts):
    """Median of the rows of each shot-resample (weighted median with the shot multiplicities)."""
    out = np.full(counts.shape[0], np.nan)
    if len(values) == 0:
        return out
    order = np.argsort(values, kind="stable")
    vs = values[order]
    ss = shot_idx[order]
    for b in range(counts.shape[0]):
        cw = np.cumsum(counts[b][ss])
        tot = cw[-1]
        if tot > 0:
            out[b] = vs[min(np.searchsorted(cw, tot / 2.0), len(vs) - 1)]
    return out


def shot_sums(x, shot_idx, n):
    return np.bincount(shot_idx, weights=x, minlength=n)


def readouts(tab, counts, n_shots, with_median=True):
    """Skill vs persistence, mean and median reduced chi-square of every arm on a row table."""
    sid = tab["shot"]
    N = len(sid)
    res = {"n_rows": int(N), "n_shots": int(len(np.unique(sid))), "arms": {}}
    if N == 0:
        return res
    nrow_s = np.bincount(sid, minlength=n_shots).astype(float)
    nb = counts @ nrow_s
    sse_p = shot_sums(tab["r2"]["persistence"], sid, n_shots)
    bp = counts @ sse_p
    for arm in ARMS:
        r2 = tab["r2"][arm]
        chi = r2 / tab["den"][arm]
        s = shot_sums(r2, sid, n_shots)
        bs = counts @ s
        with np.errstate(all="ignore"):
            sk = interval(1.0 - s.sum() / sse_p.sum(), 1.0 - bs / bp, 0.0)
            bmean = (counts @ shot_sums(chi, sid, n_shots)) / nb
        a = {"mse": float(r2.mean()), "rmse": float(math.sqrt(r2.mean())),
             "skill_vs_persistence": sk, "chi2_mean": interval(chi.mean(), bmean, 1.0),
             "chi2_mean_vs_1p5": interval(chi.mean(), bmean, 1.5)}
        if with_median:
            bmed = boot_weighted_median(chi, sid, counts)
            a["chi2_median"] = interval(float(np.median(chi)), bmed, 1.0)
            with np.errstate(all="ignore"):
                a["chi2_mean_over_median"] = interval(chi.mean() / np.median(chi), bmean / bmed, 2.2)
        res["arms"][arm] = a
    return res


def subset(tab, sel):
    return {"shot": tab["shot"][sel], "R": tab["R"][sel], "sig2": tab["sig2"][sel],
            "r2": {a: v[sel] for a, v in tab["r2"].items()},
            "den": {a: v[sel] for a, v in tab["den"].items()}}


# --------------------------------------------------------------------------- formatting

def fmt(e, nd=3, signed=True):
    if e is None or e.get("point") is None:
        return "n/a"
    f = "{:" + ("+" if signed else "") + "." + str(nd) + "f}"
    return f"{f.format(e['point'])} [{f.format(e['lo'])}, {f.format(e['hi'])}]"


def side_txt(e):
    if e is None or e.get("point") is None:
        return "n/a"
    ref = f"{e['ref']:g}"
    return {"above": f"excludes {ref} (above)", "below": f"excludes {ref} (below)",
            "includes": f"includes {ref}"}[e["side"]]


# --------------------------------------------------------------------------- main

def main():
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    shots = read_gate_shots()
    meta = pd.read_parquet(META, columns=["shot_id", "campaign"])
    camp = dict(zip(meta["shot_id"].astype(int), meta["campaign"].astype(str)))
    data = {}
    for s in shots:
        d = load_shot(s)
        if d is not None:
            data[s] = d
    have = sorted(data)
    split = {}
    camps = sorted(set(camp[s] for s in have))
    for c in camps:
        for i, s in enumerate(sorted(s for s in have if camp[s] == c)):
            split[s] = "TUNE" if i % 2 == 0 else "TEST"
    usable, med_dt = {}, {}
    for s in have:
        tt = data[s]["time_ms"]
        dd = np.diff(tt[np.isfinite(tt)])
        m = float(np.median(dd)) if len(dd) else float("nan")
        med_dt[s] = m
        usable[s] = bool(np.isfinite(m) and abs(m - STEP_MS) <= 0.1)
    test_shots = [s for s in have if split[s] == "TEST"]
    tune_shots = [s for s in have if split[s] == "TUNE"]
    tpos = {s: i for i, s in enumerate(test_shots)}
    nT = len(test_shots)
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, nT, size=(N_BOOT, nT))
    counts = np.zeros((N_BOOT, nT))
    for b in range(N_BOOT):
        counts[b] = np.bincount(bidx[b], minlength=nT)
    log(f"gate shots {len(shots)}, with data {len(have)}, usable {sum(usable.values())}, "
        f"TUNE {len(tune_shots)}, TEST {len(test_shots)}")

    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_MAST_GATE.md",
           "seed": SEED, "n_boot": N_BOOT,
           "counts": {"gate_shots": len(shots), "with_data": len(have),
                      "no_data": len(shots) - len(have),
                      "usable": int(sum(usable.values())),
                      "usable_fraction_of_with_data": sum(usable.values()) / len(have),
                      "unusable_shots_median_dt_ms": {str(s): med_dt[s] for s in have if not usable[s]},
                      "tune_shots": len(tune_shots), "test_shots": len(test_shots),
                      "usable_tune": int(sum(usable[s] for s in tune_shots)),
                      "usable_test": int(sum(usable[s] for s in test_shots)),
                      "by_campaign": {c: {"with_data": int(sum(camp[s] == c for s in have)),
                                          "usable": int(sum(camp[s] == c and usable[s] for s in have))}
                                      for c in camps},
                      "chord_counts": {str(k): int(v) for k, v in zip(*np.unique(
                          [data[s]["T"].shape[1] for s in have], return_counts=True))},
                      "per_target_population": {}},
           "tuning": {}, "gate_a": {}, "gate_b": {}, "readouts": {}, "regime": {}, "audit": {},
           "radius": {}, "sensitivity_irregular_shots_dropped": {}}

    prep = {}
    for tg in TARGETS:
        for pop, rel in POPS.items():
            pp = {s: (prepare(data[s], tg, rel) if usable[s] else {"status": "unusable"}) for s in have}
            prep[tg, pop] = pp
            ok = [p for p in pp.values() if p["status"] == "ok"]
            Ls = np.array([p["L"] for p in pp.values() if p["status"] in ("ok", "no_scored_chord")])
            c = {"shots_no_block": int(sum(p["status"] == "no_block" for p in pp.values())),
                 "shots_no_scored_chord": int(sum(p["status"] == "no_scored_chord" for p in pp.values())),
                 "shots_with_block_and_chords": len(ok),
                 "shots_with_block_and_chords_no_scored_row": int(sum(p["n_rows"] == 0 for p in ok)),
                 "block_frames_median": float(np.median(Ls)),
                 "block_frames_p25_p75": [float(x) for x in np.percentile(Ls, [25, 75])],
                 "block_frames_min_max": [int(Ls.min()), int(Ls.max())],
                 "scored_chords_total": int(sum(p["n_chords"] for p in ok)),
                 "scored_chords_median_per_shot": float(np.median([p["n_chords"] for p in ok])),
                 "block_intervals_outside_5pm0p1_ms": int(sum(p["n_bad_dt"] for p in ok)),
                 "shots_with_such_intervals": int(sum(p["n_bad_dt"] > 0 for p in ok)),
                 "shots_with_non_increasing_block_time": int(sum(p["n_negative_dt"] > 0 for p in ok)),
                 "irregular_shots": {str(s): split[s] for s in have
                                     if pp[s]["status"] == "ok" and pp[s]["n_bad_dt"] > 0},
                 "shots_with_nonfinite_block_time": int(sum(not p["t_finite"] for p in ok))}
            for sp, lst in (("TUNE", tune_shots), ("TEST", test_shots)):
                o = [pp[s] for s in lst if pp[s]["status"] == "ok"]
                c[f"scored_rows_{sp}"] = int(sum(p["n_rows"] for p in o))
                c[f"shots_with_rows_{sp}"] = int(sum(p["n_rows"] > 0 for p in o))
                c[f"scored_chords_{sp}"] = int(sum(p["n_chords"] for p in o))
            res["counts"]["per_target_population"][f"{tg}|{pop}"] = c
            log(f"prepared {tg} {pop}: rows TUNE {c['scored_rows_TUNE']} TEST {c['scored_rows_TEST']}")

    for tg in TARGETS:
        log(f"tuning {tg}")
        tinfo, params = tune([prep[tg, "G"][s] for s in tune_shots], log)
        res["tuning"][tg] = tinfo

        tabs = {}
        for pop in POPS:
            acc = {"shot": [], "R": [], "sig2": [], "r2": {a: [] for a in ARMS},
                   "den": {a: [] for a in ARMS}}
            for s in test_shots:
                p = prep[tg, pop][s]
                if p["status"] != "ok" or p["n_rows"] == 0:
                    continue
                rows = p["rows"]
                yt = p["Y0"][p["hid"]][rows]
                s2t = p["S2"][p["hid"]][rows]
                acc["shot"].append(np.full(len(yt), tpos[s]))
                acc["R"].append(np.broadcast_to(p["R"][None, :], rows.shape)[rows])
                acc["sig2"].append(s2t)
                for arm in ARMS:
                    pred, vw = arm_outputs(p, arm, params, mask=rows)
                    acc["r2"][arm].append((pred[rows] - yt) ** 2)
                    acc["den"][arm].append(s2t + vw[rows])
            tab = {"shot": np.concatenate(acc["shot"]).astype(np.int64),
                   "R": np.concatenate(acc["R"]), "sig2": np.concatenate(acc["sig2"]),
                   "r2": {a: np.concatenate(v) for a, v in acc["r2"].items()},
                   "den": {a: np.concatenate(v) for a, v in acc["den"].items()}}
            tabs[pop] = tab
            log(f"TEST rows {tg} {pop}: {len(tab['shot'])}")
            res["readouts"][f"{tg}|{pop}"] = readouts(tab, counts, nT)
        rG = res["readouts"][f"{tg}|G"]
        best = min(CAUSAL_ARMS, key=lambda a: rG["arms"][a]["mse"])

        # ---------------- gate (a), population G, TEST
        ta = np.full(nT, np.nan)
        Sk = {}
        q_rows, q_shot = [], []
        for s in test_shots:
            p = prep[tg, "G"][s]
            if p["status"] != "ok":
                continue
            G, S2 = p["G"], p["S2"]
            ta[tpos[s]], Sk[s] = t_a_statistic(np.where(G, p["Y0"], np.nan), G)
            d1 = p["Y0"][2:] - 2.0 * p["Y0"][1:-1] + p["Y0"][:-2]
            ok = G[2:] & G[1:-1] & G[:-2]
            var = 4.0 * S2[1:-1] + S2[2:] + S2[:-2]
            q_rows.append((d1[ok] ** 2) / var[ok])
            q_shot.append(np.full(int(ok.sum()), tpos[s]))
        q_rows = np.concatenate(q_rows)
        q_shot = np.concatenate(q_shot).astype(np.int64)
        bta = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            x = ta[bidx[b]]
            if np.isfinite(x).any():
                bta[b] = np.nanmedian(x)
        e_ta = interval(float(np.nanmedian(ta)), bta, 0.0)
        e_q = interval(float(np.median(q_rows)) / GAUSS_MED_CHI2_1,
                       boot_weighted_median(q_rows, q_shot, counts) / GAUSS_MED_CHI2_1, 1.0)
        fail_a = e_ta["side"] == "below"
        Sarr = np.array([[Sk[s][k] for k in (1, 2, 4)] for s in Sk], float)
        ga = {"T_a_median_over_TEST_shots": e_ta, "n_shots_with_T_a": int(np.isfinite(ta).sum()),
              "T_a_per_shot_quartiles": [float(x) for x in np.nanpercentile(ta, [25, 50, 75])],
              "share_of_shots_with_T_a_below_0": float(np.mean(ta[np.isfinite(ta)] < 0)),
              "median_S1_S2_S4_over_shots": [float(x) for x in np.nanmedian(Sarr, axis=0)],
              "verdict": "FAIL" if fail_a else "PASS",
              "rho1_bound": (-0.75 * e_ta["lo"]) if e_ta["lo"] < 0 else None,
              "coreported_ratio": e_q,
              "coreported_median_raw": float(np.median(q_rows)), "coreported_n_rows": int(len(q_rows)),
              "errors_larger_than_scatter_reading_applies": bool((not fail_a) and e_q["side"] == "below")}
        res["gate_a"][tg] = ga

        # ---------------- gate (b)
        gb = {"best_causal_arm": best,
              "test_mse_causal_arms_G": {a: rG["arms"][a]["mse"] for a in CAUSAL_ARMS}}
        for pop in POPS:
            a = res["readouts"][f"{tg}|{pop}"]["arms"][best]
            gb[f"chi2_mean_{pop}"] = a["chi2_mean_vs_1p5"]
            gb[f"chi2_median_{pop}"] = a["chi2_median"]
            if ga["errors_larger_than_scatter_reading_applies"]:
                f = e_q["point"]
                m = a["chi2_mean_vs_1p5"]
                lo, hi = m["lo"] / f, m["hi"] / f
                side = "above" if lo > 1.5 else ("below" if hi < 1.5 else "includes")
                gb[f"chi2_mean_{pop}_errors_rescaled"] = {
                    "point": m["point"] / f, "lo": lo, "hi": hi, "ref": 1.5, "side": side,
                    "excludes_ref": side != "includes", "variance_factor": f}
        gb["verdict"] = "PASS" if gb["chi2_mean_G"]["side"] == "above" else "FAIL"
        res["gate_b"][tg] = gb

        # ---------------- sensitivity: TEST shots with a block frame interval outside 5 +- 0.1 ms dropped
        sens = {}
        for pop in POPS:
            irr = np.array([prep[tg, pop][s]["status"] == "ok" and prep[tg, pop][s]["n_bad_dt"] > 0
                            for s in test_shots])
            tab = tabs[pop]
            kept = readouts(subset(tab, ~irr[tab["shot"]]), counts, nT, with_median=False)
            sens[pop] = {"test_shots_dropped": int(irr.sum()), "n_rows": kept["n_rows"],
                         "skill_vs_persistence": {a: kept["arms"][a]["skill_vs_persistence"] for a in ARMS},
                         "chi2_mean_best_causal_arm_vs_1p5": kept["arms"][best]["chi2_mean_vs_1p5"]}
            if pop == "G":
                ta2 = np.where(irr, np.nan, ta)
                b2 = np.full(N_BOOT, np.nan)
                for b in range(N_BOOT):
                    x = ta2[bidx[b]]
                    if np.isfinite(x).any():
                        b2[b] = np.nanmedian(x)
                sens[pop]["T_a_median"] = interval(float(np.nanmedian(ta2)), b2, 0.0)
        res["sensitivity_irregular_shots_dropped"][tg] = sens

        # ---------------- regime number, R1, R2 (population G)
        reg = np.full(nT, np.nan)
        for s in test_shots:
            p = prep[tg, "G"][s]
            if p["status"] == "ok":
                reg[tpos[s]] = regime_number(np.where(p["G"], p["Y0"], np.nan), p["G"])
        tab = tabs["G"]
        sp = shot_sums(tab["r2"]["persistence"], tab["shot"], nT)
        with np.errstate(all="ignore"):
            dsk = (shot_sums(tab["r2"]["running_mean"], tab["shot"], nT)
                   - shot_sums(tab["r2"]["slope_filter"], tab["shot"], nT)) / sp
        okr = np.isfinite(reg) & np.isfinite(dsk)
        rho = stats.spearmanr(reg[okr], dsk[okr]).statistic if okr.sum() > 2 else np.nan
        brho = np.full(N_BOOT, np.nan)
        breg = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            ii = bidx[b][okr[bidx[b]]]
            if len(np.unique(ii)) > 2:
                brho[b] = stats.spearmanr(reg[ii], dsk[ii]).statistic
            x = reg[bidx[b]]
            if np.isfinite(x).any():
                breg[b] = np.nanmedian(x)
        sf = rG["arms"]["slope_filter"]["skill_vs_persistence"]
        rm = rG["arms"]["running_mean"]["skill_vs_persistence"]
        e_rho = interval(rho, brho, 0.0)
        res["regime"][tg] = {
            "regime_number_median_over_TEST_shots": interval(float(np.nanmedian(reg)), breg, -0.5),
            "regime_number_quartiles": [float(x) for x in np.nanpercentile(reg, [25, 50, 75])],
            "n_shots": int(np.isfinite(reg).sum()),
            "R1": {"slope_filter_skill": sf, "running_mean_skill": rm,
                   "holds": bool(sf["side"] == "above" and rm["side"] != "above")},
            "R2": {"spearman": e_rho, "n_shots": int(okr.sum()),
                   "holds": bool(e_rho["side"] == "above")}}

        # ---------------- audit and radius (both populations)
        for pop in POPS:
            tab = tabs[pop]
            N = len(tab["shot"])
            r2b = tab["r2"][best]
            ntop = int(math.ceil(0.001 * N))
            top = np.zeros(N, bool)
            top[np.argsort(r2b, kind="stable")[N - ntop:]] = True
            sb = shot_sums(r2b, tab["shot"], nT)
            stop = shot_sums(r2b * top, tab["shot"], nT)
            with np.errstate(all="ignore"):
                share = interval(stop.sum() / sb.sum(), (counts @ stop) / (counts @ sb), 0.001)
            kept = readouts(subset(tab, ~top), counts, nT, with_median=False)
            res["audit"][f"{tg}|{pop}"] = {
                "best_causal_arm": best,
                "chi2_mean_over_median": res["readouts"][f"{tg}|{pop}"]["arms"][best]["chi2_mean_over_median"],
                "top_rows_dropped": ntop, "top_0p1pct_share_of_sse": share,
                "top_rows_n_shots": int(len(np.unique(tab["shot"][top]))),
                "skill_with_top_rows_dropped": {a: kept["arms"][a]["skill_vs_persistence"] for a in ARMS},
                "chi2_mean_with_top_rows_dropped": {a: kept["arms"][a]["chi2_mean"] for a in ARMS}}
            fin = np.isfinite(tab["R"])
            rad = {"n_rows_without_radius": int((~fin).sum())}
            if fin.sum() >= 3:
                e1, e2 = np.percentile(tab["R"][fin], [100 / 3, 200 / 3])
                rad["tertile_edges_m"] = [float(e1), float(e2)]
                sels = {"inner": fin & (tab["R"] <= e1),
                        "middle": fin & (tab["R"] > e1) & (tab["R"] <= e2),
                        "outer": fin & (tab["R"] > e2)}
                for name, sel in sels.items():
                    rad[name] = readouts(subset(tab, sel), counts, nT)
            res["radius"][f"{tg}|{pop}"] = rad
        log(f"{tg}: gate (a) {ga['verdict']}, gate (b) {gb['verdict']} (best causal arm {best})")

    res["expectations"] = expectations(res)
    n_eq = int(sum(data[s]["chisq_cols"] == data[s]["T"].shape[1] for s in have))
    pairs = sorted(set((data[s]["T"].shape[1], data[s]["chisq_cols"]) for s in have))
    res["ss_chisq"] = (f"not reported: the column count of ss_chisq equals the chord count in {n_eq} of "
                       f"{len(have)} gate files (chords / ss_chisq columns: "
                       + ", ".join(f"{a} / {b}" for a, b in pairs)
                       + "), so its columns are not mapped to chords")
    res["deviations_ambiguities"] = DEVIATIONS
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    log(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    return res


DEVIATIONS = [
    "Block, scored chords and scored rows are built separately per target and population from that "
    "target's good-sample rule (the registration defines the block through 'good chords' without "
    "saying which target fixes it).",
    "Frames are addressed by block position on the nominal 5 ms grid: 't - 5 ms', 't - 15 ms', "
    "'t + 5 ms' are block positions j - 1, j - 3, j + 1, and every time-dependent weight "
    "(extrapolation, running-mean age, filter dt, GP kernel) uses position x 5 ms. Stored frame "
    "times enter the usable-shot rule only. Reason: a few usable shots carry block frame intervals "
    "outside 5.0 +- 0.1 ms, some of them negative (time stamps out of order), which the "
    "registration did not anticipate; these shots stay in the row set as registered, their count "
    "is in the counts table, and section 5b repeats the gate statistics with them dropped.",
    "Running mean: all good visible samples of the block before the hidden frame, weights "
    "exp(-age / tau), normalised to sum 1.",
    "Slope-state filter: discrete model per frame interval, F = [[1, dt], [0, 1]], "
    "Q = diag(q_level dt, q_slope dt) with dt in 5 ms steps; exact diffuse initialisation (level "
    "from the first good sample, slope fixed by the second); bad visible samples get no update. Grid (not "
    "fixed numerically in the registration): q_level / scale^2 = 10^(-2 .. +1.5), q_slope / scale^2 "
    "= 10^(-4 .. +1), 6 log-spaced values each.",
    "GP window: 'last 8 visible samples' = the 8 visible block frames before the hidden frame "
    "(fewer at the block start), offline 4 visible frames on each side; bad samples inside the "
    "window are dropped, the window is not extended. Constant-kernel variance = 1e6 x amplitude^2 "
    "(not stated in the registration), applied through the Sherman-Morrison identity.",
    "Target scale for the GP amplitude and the filter grid: median reported error of the hidden "
    "sample over TUNE scored rows, population G.",
    "Population W uses the hyperparameters tuned on population G (the registration tunes on G only) "
    "and the best causal arm chosen in population G.",
    "Gate (a): the chords entering S_k are the scored chords with at least 12 triples at that "
    "spacing (the chord set may differ between k). Co-reported ratio: rows = all good d_1 triples "
    "of scored chords in TEST shots, population G.",
    "Regime number: Pearson correlation of consecutive 10 ms increments (both increments from good "
    "visible samples), chords with at least 8 such pairs. R2 per-shot skills are 1 - SSE_arm / "
    "SSE_persistence within the shot.",
    "Radius tertiles: edges are the 1/3 and 2/3 quantiles of major radius over the TEST scored rows "
    "of that target and population (pooled over shots).",
    "Top-0.1 % rows: ceil(0.001 N) rows with the largest squared residual of the best causal arm; "
    "the same rows are dropped from every arm. The interval of the share keeps that row set fixed.",
    "Bootstrap medians over rows are weighted medians with the shot multiplicities (lower median); "
    "point estimates use numpy.median. Intervals are percentile intervals (2.5, 97.5).",
    "Section 6 expectations stated without an interval rule are judged on the point estimate; "
    "'about -0.4' is judged as within +-0.1; 'no better than 0' as the interval not lying above 0.",
    "If gate (a) passes and the co-reported ratio interval lies below 1, the rescaled floor is the "
    "mean reduced chi-square divided by the point estimate of the ratio (interval scaled by the "
    "same factor).",
]


def expectations(res):
    out = []

    def add(name, expected, found, met):
        out.append({"expectation": name, "expected": expected, "found": found, "met": bool(met)})

    c = res["counts"]
    f = c["usable_fraction_of_with_data"]
    add("usable shots", "85-95 % of the shots with data",
        f"{c['usable']} / {c['with_data']} = {100 * f:.1f} %", 0.85 <= f <= 0.95)
    mb = c["per_target_population"]["T_i|G"]["block_frames_median"]
    add("median block length (T_i, G)", "45-60 frames", f"{mb:g}", 45 <= mb <= 60)
    ga = res["gate_a"]["T_i"]
    e = ga["T_a_median_over_TEST_shots"]
    add("gate (a) T_i: T_a interval includes 0 (pass)", "includes 0", fmt(e) + ", " + side_txt(e),
        e["side"] == "includes")
    q = ga["coreported_ratio"]
    add("gate (a) T_i: co-reported ratio below 1", "< 1 (pilot 0.68)",
        fmt(q, signed=False) + ", " + side_txt(q), q["point"] < 1)
    r = res["readouts"]["T_i|G"]["arms"]
    x = r["extrapolation"]["skill_vs_persistence"]
    add("T_i G: extrapolation vs persistence", "about -0.4", fmt(x), abs(x["point"] + 0.4) <= 0.1)
    tau = res["tuning"]["T_i"]["running_mean"]["selected_tau_ms"]
    add("T_i G: running mean best tau", "5 ms", f"{tau:g} ms", tau == 5.0)
    x = r["running_mean"]["skill_vs_persistence"]
    add("T_i G: running mean no better than 0", "interval not above 0", fmt(x) + ", " + side_txt(x),
        x["side"] != "above")
    x = r["slope_filter"]["skill_vs_persistence"]
    add("T_i G: slope-state filter vs persistence", "+0.05 to +0.25", fmt(x), 0.05 <= x["point"] <= 0.25)
    x = r["offline_interp"]["skill_vs_persistence"]
    add("T_i G: offline interpolation vs persistence", "+0.35 to +0.60", fmt(x), 0.35 <= x["point"] <= 0.60)
    gb = res["gate_b"]["T_i"]
    x = gb["chi2_mean_G"]
    add("T_i G: mean reduced chi-square, best causal arm", "2 to 5",
        fmt(x, 2, False) + f" ({gb['best_causal_arm']})", 2 <= x["point"] <= 5)
    add("T_i: gate (b) passes", "PASS", gb["verdict"], gb["verdict"] == "PASS")
    x = gb["chi2_median_G"]
    add("T_i G: median reduced chi-square, best causal arm", "< 1", fmt(x, 3, False), x["point"] < 1)
    r = res["readouts"]["V_i|G"]["arms"]
    x = r["slope_filter"]["skill_vs_persistence"]
    add("V_i G: slope-state filter vs persistence", "+0.2 to +0.5", fmt(x), 0.2 <= x["point"] <= 0.5)
    x = r["offline_interp"]["skill_vs_persistence"]
    add("V_i G: offline interpolation vs persistence", "+0.55 to +0.75", fmt(x), 0.55 <= x["point"] <= 0.75)
    gb = res["gate_b"]["V_i"]
    x = gb["chi2_mean_G"]
    add("V_i G: mean reduced chi-square, best causal arm", "> 10",
        fmt(x, 2, False) + f" ({gb['best_causal_arm']})", x["point"] > 10)
    add("V_i: gate (b) passes", "PASS", gb["verdict"], gb["verdict"] == "PASS")
    for tg in TARGETS:
        x = res["audit"][f"{tg}|G"]["chi2_mean_over_median"]
        add(f"{tg} G: mean-to-median reduced chi-square ratio, best causal arm", "> 4",
            fmt(x, 2, False), x["point"] > 4)
    return out


def render_md(res):
    L = []
    A = L.append
    c = res["counts"]
    A("# MAST gate - results")
    A("")
    A("Generated by `ces_prediction/experiments/mast/run_gate.py` from the registration "
      "`ces_prediction/experiments/PREREGISTRATION_MAST_GATE.md` (seed 20261116, 2,000 shot-clustered "
      "bootstrap resamples of TEST shots, the same resample indices for every arm). Every number is "
      "on TEST unless marked TUNE. Entries are point [95 % interval]. T_i in eV, V_i in km/s. "
      "Populations: G = relative T_i error < 0.1 (primary), W = < 0.5.")
    A("")
    A("## 1. Verdicts")
    A("")
    A("| target | gate (a): median T_a over TEST shots | vs 0 | gate (a) | best causal arm | "
      "mean reduced chi-square, G | vs 1.5 | gate (b) |")
    A("|---|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        ga, gb = res["gate_a"][tg], res["gate_b"][tg]
        A(f"| {tg} | {fmt(ga['T_a_median_over_TEST_shots'])} | {side_txt(ga['T_a_median_over_TEST_shots'])} | "
          f"**{ga['verdict']}** | {gb['best_causal_arm']} | {fmt(gb['chi2_mean_G'], 2, False)} | "
          f"{side_txt(gb['chi2_mean_G'])} | **{gb['verdict']}** |")
    A("")
    A("Gate (a) fails when the interval lies entirely below 0; gate (b) passes when the interval of "
      "the mean reduced chi-square of the best causal arm lies above 1.5 in population G.")
    A("")
    A("## 2. Counts (section 2 of the registration)")
    A("")
    A(f"- Gate shots {c['gate_shots']}; with `act` arrays {c['with_data']}; without {c['no_data']}.")
    A(f"- Usable (median frame spacing 5.0 +- 0.1 ms): {c['usable']} / {c['with_data']} = "
      f"{100 * c['usable_fraction_of_with_data']:.1f} %. TUNE {c['tune_shots']} shots "
      f"({c['usable_tune']} usable), TEST {c['test_shots']} shots ({c['usable_test']} usable).")
    A("- By campaign (with data / usable): " + ", ".join(
        f"{k} {v['with_data']} / {v['usable']}" for k, v in c["by_campaign"].items()) + ".")
    A("- Chord count per file: " + ", ".join(
        f"{k} chords in {v} shots" for k, v in c["chord_counts"].items()) + ".")
    A("")
    A("| target, population | no block | block, no scored chord | block and chords | of those, no "
      "scored row | block frames median [p25, p75] (min-max) | scored chords total (median per shot) | "
      "scored rows TUNE (shots) | scored rows TEST (shots) | block intervals outside 5 +- 0.1 ms (shots) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in c["per_target_population"].items():
        A(f"| {k.replace('|', ', ')} | {v['shots_no_block']} | {v['shots_no_scored_chord']} | "
          f"{v['shots_with_block_and_chords']} | {v['shots_with_block_and_chords_no_scored_row']} | "
          f"{v['block_frames_median']:g} [{v['block_frames_p25_p75'][0]:g}, {v['block_frames_p25_p75'][1]:g}] "
          f"({v['block_frames_min_max'][0]}-{v['block_frames_min_max'][1]}) | "
          f"{v['scored_chords_total']} ({v['scored_chords_median_per_shot']:g}) | "
          f"{v['scored_rows_TUNE']} ({v['shots_with_rows_TUNE']}) | "
          f"{v['scored_rows_TEST']} ({v['shots_with_rows_TEST']}) | "
          f"{v['block_intervals_outside_5pm0p1_ms']} ({v['shots_with_such_intervals']}; "
          f"{v['shots_with_non_increasing_block_time']} with a non-increasing time) |")
    A("")
    A("Block lengths are over usable shots that have a block (TUNE and TEST together).")
    A("")
    A("## 3. Gate (a): independence of the hidden frame (population G, TEST)")
    A("")
    A("| target | shots with T_a | median T_a | vs 0 | per-shot quartiles of T_a | share of shots "
      "with T_a < 0 | median S_1, S_2, S_4 | implied bound on rho_1 | co-reported ratio: (median "
      "d_1^2 / reported variance) / 0.455 | vs 1 | rows |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        g = res["gate_a"][tg]
        q = g["coreported_ratio"]
        S = g["median_S1_S2_S4_over_shots"]
        qs = g["T_a_per_shot_quartiles"]
        rb = ("n/a (lower bound not negative)" if g["rho1_bound"] is None
              else f"rho_1 <= {g['rho1_bound']:.3f}")
        A(f"| {tg} | {g['n_shots_with_T_a']} | {fmt(g['T_a_median_over_TEST_shots'])} | "
          f"{side_txt(g['T_a_median_over_TEST_shots'])} | {qs[0]:+.3f} / {qs[1]:+.3f} / {qs[2]:+.3f} | "
          f"{100 * g['share_of_shots_with_T_a_below_0']:.1f} % | {S[0]:.4g}, {S[1]:.4g}, {S[2]:.4g} | "
          f"{rb} | {fmt(q, 3, False)} | {side_txt(q)} | {g['coreported_n_rows']} |")
    A("")
    for tg in TARGETS:
        g = res["gate_a"][tg]
        A(f"- {tg}: gate (a) **{g['verdict']}**. " + (
            "The interval lies entirely below 0: a hidden frame shares noise with its neighbours; "
            "decimation scores on MAST are not quoted as a measured answer at the filled instant, "
            "and MAST is used for the regime and coupling study only."
            if g["verdict"] == "FAIL" else
            "No adjacent-frame noise correlation detected at this power (one-sided, conservative test).")
          + (" The reading 'reported errors are larger than the frame-to-frame scatter at the median "
             "row' applies (gate (a) passes and the ratio interval lies below 1); the floor of "
             "section 4 is quoted raw and rescaled."
             if g["errors_larger_than_scatter_reading_applies"] else
             " The rescaled-floor clause does not apply (it requires gate (a) to pass and the ratio "
             "interval to lie below 1)."))
    A("")
    A("## 4. Gate (b): CES-only arms")
    A("")
    A("### Tuning (TUNE scored rows, population G)")
    A("")
    A("| target | scale (median reported error) | TUNE rows | running mean tau | filter q_level / "
      "scale^2 | filter q_slope / scale^2 | causal GP length, amplitude / scale | offline GP length, "
      "amplitude / scale | optimum on a grid edge |")
    A("|---|---|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        t = res["tuning"][tg]
        edges = []
        if t["running_mean"]["on_grid_edge"]:
            edges.append("tau")
        if t["slope_filter"]["on_grid_edge_q_level"]:
            edges.append("q_level")
        if t["slope_filter"]["on_grid_edge_q_slope"]:
            edges.append("q_slope")
        for n in ("gp_causal", "gp_offline"):
            if t[n]["on_grid_edge_length"]:
                edges.append(n + " length")
            if t[n]["on_grid_edge_amp"]:
                edges.append(n + " amplitude")
        A(f"| {tg} | {t['scale_median_reported_error']:.4g} | {t['n_tune_rows']} | "
          f"{t['running_mean']['selected_tau_ms']:g} ms | "
          f"{t['slope_filter']['selected_q_level_over_scale2']:.4g} | "
          f"{t['slope_filter']['selected_q_slope_over_scale2']:.4g} | "
          f"{t['gp_causal']['selected_length_ms']:g} ms, {t['gp_causal']['selected_amp_over_scale']:g} | "
          f"{t['gp_offline']['selected_length_ms']:g} ms, {t['gp_offline']['selected_amp_over_scale']:g} | "
          f"{', '.join(edges) if edges else 'none'} |")
    A("")
    A("Filter grid: q_level / scale^2 in {" + ", ".join(f"{10 ** x:.3g}" for x in Q_LEVEL_EXP) +
      "}, q_slope / scale^2 in {" + ", ".join(f"{10 ** x:.3g}" for x in Q_SLOPE_EXP) +
      "} per 5 ms step. The TUNE MSE of every grid point is in the JSON.")
    A("")
    for tg in TARGETS:
        for pop in POPS:
            r = res["readouts"][f"{tg}|{pop}"]
            A(f"### {tg}, population {pop} (TEST: {r['n_rows']} rows, {r['n_shots']} shots)")
            A("")
            A("| arm | RMSE | skill vs persistence | vs 0 | mean reduced chi-square | vs 1 | median "
              "reduced chi-square | vs 1 | mean / median |")
            A("|---|---|---|---|---|---|---|---|---|")
            for arm in ARMS:
                a = r["arms"][arm]
                star = " (best causal)" if arm == res["gate_b"][tg]["best_causal_arm"] else ""
                sk = "0 (reference)" if arm == "persistence" else fmt(a["skill_vs_persistence"])
                sks = "-" if arm == "persistence" else side_txt(a["skill_vs_persistence"])
                A(f"| {arm}{star} | {a['rmse']:.4g} | {sk} | {sks} | {fmt(a['chi2_mean'], 2, False)} | "
                  f"{side_txt(a['chi2_mean'])} | {fmt(a['chi2_median'], 3, False)} | "
                  f"{side_txt(a['chi2_median'])} | {fmt(a['chi2_mean_over_median'], 2, False)} |")
            A("")
    A("### Gate (b) statistic")
    A("")
    A("| target | best causal arm (lowest TEST MSE, G) | mean reduced chi-square G | vs 1.5 | median G | "
      "mean reduced chi-square W | vs 1.5 | G with errors rescaled | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        g = res["gate_b"][tg]
        rs = g.get("chi2_mean_G_errors_rescaled")
        rs_t = (fmt(rs, 2, False) + ", " + side_txt(rs)) if rs else "not applicable"
        A(f"| {tg} | {g['best_causal_arm']} | {fmt(g['chi2_mean_G'], 2, False)} | "
          f"{side_txt(g['chi2_mean_G'])} | {fmt(g['chi2_median_G'], 3, False)} | "
          f"{fmt(g['chi2_mean_W'], 2, False)} | {side_txt(g['chi2_mean_W'])} | {rs_t} | **{g['verdict']}** |")
    A("")
    A("## 5. Regime number and audit readouts (not gating)")
    A("")
    A("| target | regime number, median over TEST shots (white noise on a flat signal: -0.5) | quartiles | "
      "shots | R1: slope-filter skill | R1: running-mean skill | R1 holds | R2: Spearman(regime number, "
      "slope-filter skill - running-mean skill) | vs 0 | R2 holds |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        g = res["regime"][tg]
        qs = g["regime_number_quartiles"]
        A(f"| {tg} | {fmt(g['regime_number_median_over_TEST_shots'])} | "
          f"{qs[0]:+.3f} / {qs[1]:+.3f} / {qs[2]:+.3f} | {g['n_shots']} | "
          f"{fmt(g['R1']['slope_filter_skill'])} ({side_txt(g['R1']['slope_filter_skill'])}) | "
          f"{fmt(g['R1']['running_mean_skill'])} ({side_txt(g['R1']['running_mean_skill'])}) | "
          f"{'yes' if g['R1']['holds'] else 'no'} | {fmt(g['R2']['spearman'])} (n = {g['R2']['n_shots']}) | "
          f"{side_txt(g['R2']['spearman'])} | {'yes' if g['R2']['holds'] else 'no'} |")
    A("")
    A("### Audit: heavy tail of the best causal arm")
    A("")
    A("| target, population | best causal arm | mean / median reduced chi-square (Gaussian 2.2) | "
      "rows dropped (0.1 %) | shots they come from | their share of the SSE |")
    A("|---|---|---|---|---|---|")
    for k, a in res["audit"].items():
        A(f"| {k.replace('|', ', ')} | {a['best_causal_arm']} | {fmt(a['chi2_mean_over_median'], 2, False)} | "
          f"{a['top_rows_dropped']} | {a['top_rows_n_shots']} | {fmt(a['top_0p1pct_share_of_sse'], 3, False)} |")
    A("")
    A("Skill against persistence with those rows dropped from every arm (mean reduced chi-square in "
      "parentheses):")
    A("")
    A("| target, population | " + " | ".join(a for a in ARMS if a != "persistence") + " |")
    A("|---|" + "---|" * (len(ARMS) - 1))
    for k, a in res["audit"].items():
        A(f"| {k.replace('|', ', ')} | " + " | ".join(
            f"{fmt(a['skill_with_top_rows_dropped'][arm])} "
            f"({a['chi2_mean_with_top_rows_dropped'][arm]['point']:.2f})"
            for arm in ARMS if arm != "persistence") + " |")
    A("")
    A("### Radius tertiles (major radius over the TEST scored rows)")
    A("")
    for k, rad in res["radius"].items():
        if "tertile_edges_m" not in rad:
            A(f"- {k.replace('|', ', ')}: no finite radius.")
            continue
        A(f"**{k.replace('|', ', ')}** - edges {rad['tertile_edges_m'][0]:.3f} m, "
          f"{rad['tertile_edges_m'][1]:.3f} m; rows without a radius: {rad['n_rows_without_radius']}; rows "
          + ", ".join(f"{n} {rad[n]['n_rows']}" for n in ("inner", "middle", "outer")) + ".")
        A("")
        A("| arm | inner: skill | inner: mean / median chi-square | middle: skill | middle: mean / "
          "median chi-square | outer: skill | outer: mean / median chi-square |")
        A("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            cells = []
            for name in ("inner", "middle", "outer"):
                a = rad[name]["arms"].get(arm)
                if a is None:
                    cells += ["n/a", "n/a"]
                    continue
                cells.append("0 (reference)" if arm == "persistence" else fmt(a["skill_vs_persistence"]))
                cells.append(f"{fmt(a['chi2_mean'], 2, False)} / {fmt(a['chi2_median'], 3, False)}")
            A(f"| {arm} | " + " | ".join(cells) + " |")
        A("")
    A("## 5b. Sensitivity: TEST shots with a block frame interval outside 5.0 +- 0.1 ms dropped (not registered)")
    A("")
    A("| target, population | TEST shots dropped | rows | median T_a (G only) | mean reduced chi-square, best "
      "causal arm, vs 1.5 | " + " | ".join(a for a in ARMS if a != "persistence") + " |")
    A("|---|---|---|---|---|" + "---|" * (len(ARMS) - 1))
    for tg in TARGETS:
        for pop in POPS:
            v = res["sensitivity_irregular_shots_dropped"][tg][pop]
            ta_t = fmt(v["T_a_median"]) if "T_a_median" in v else "-"
            m = v["chi2_mean_best_causal_arm_vs_1p5"]
            A(f"| {tg}, {pop} | {v['test_shots_dropped']} | {v['n_rows']} | {ta_t} | "
              f"{fmt(m, 2, False)}, {side_txt(m)} | " + " | ".join(
                  fmt(v["skill_vs_persistence"][arm]) for arm in ARMS if arm != "persistence") + " |")
    A("")
    A("Arm columns are skill against persistence. Irregular shots (target, population: shot split): " + "; ".join(
        f"{k.replace('|', ', ')}: " + (", ".join(f"{s} {sp}" for s, sp in v["irregular_shots"].items()) or "none")
        for k, v in res["counts"]["per_target_population"].items()) + ".")
    A("")
    A("## 6. Numerical expectations of the registration (section 6)")
    A("")
    A("| expectation | registered | found | met |")
    A("|---|---|---|---|")
    for e in res["expectations"]:
        A(f"| {e['expectation']} | {e['expected']} | {e['found']} | {'met' if e['met'] else 'not met'} |")
    A("")
    A("## 7. Deviations and ambiguities (most literal reading taken)")
    A("")
    for d in res["deviations_ambiguities"]:
        A(f"- {d}")
    A(f"- `ss_chisq`: {res['ss_chisq']}.")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
    sys.exit(0)
