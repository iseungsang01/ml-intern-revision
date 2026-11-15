"""MAST coupling study: lagged association of charge-exchange increments with beam power and
D-alpha.  Registration: ces_prediction/experiments/PREREGISTRATION_MAST_COUPLING.md.

CPU only, one process; shots are streamed one at a time and the per-shot sums live in a float32
memmap under data/mast/ (gitignored).  Run from the repo root:
    py ces_prediction/experiments/mast/run_coupling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as rg  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_MAST_COUPLING.md"
HERE = Path(__file__).resolve().parent
SHOTS_TXT = HERE / "coupling_shots.txt"
DALPHA_DIR = rg.ROOT / "data" / "mast" / "dalpha"
SUMS = rg.ROOT / "data" / "mast" / ".coupling_sums.dat"
OUT_JSON = rg.ROOT / "docs" / "paper" / "mast_coupling.json"
OUT_MD = rg.ROOT / "docs" / "paper" / "mast_coupling.md"

EDGES = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
BINS = tuple(f"{a:g}-{b:g}" for a, b in zip(EDGES[:-1], EDGES[1:])) + ("all",)
NB = len(BINS)                      # 5 radial bins + "all"
MAXLAG = 20
LAGS = np.arange(-MAXLAG, MAXLAG + 1)
NL = len(LAGS)
POS = LAGS >= 0
NEG = LAGS < 0
PAD = MAXLAG + 2                    # driver grid starts PAD frames before the block
DRIVERS = ("ss", "tot", "ss_nostep", "dalpha")
KINDS = ("diff", "level")
POPS = ("G", "W")
NSTAT = 5                           # sxy, sxx, syy, snoise, n
N_NULL = 500
STEP_FRAC = 0.25
STEP_HALF = 2
MIN_SHOTS = 8
F_MIN = 0.05
CAMPS = ("M6", "M7", "M8", "M9")
KSTAR_BOUND = 0.044


# --------------------------------------------------------------------------- pure helpers

def frame_means(t_ms, v, t_a, n_lo, n_hi):
    """Mean of the driver over [t_a + 5n - 2.5, t_a + 5n + 2.5) ms for n = n_lo .. n_hi."""
    n = np.arange(n_lo, n_hi + 1)
    out = np.full(len(n), np.nan)
    ok = np.isfinite(t_ms) & np.isfinite(v)
    if ok.sum() < 2:
        return out
    t, x = t_ms[ok], v[ok].astype(float)
    if np.any(np.diff(t) < 0):
        o = np.argsort(t, kind="stable")
        t, x = t[o], x[o]
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(t, t_a + rg.STEP_MS * n - rg.STEP_MS / 2, side="left")
    hi = np.searchsorted(t, t_a + rg.STEP_MS * n + rg.STEP_MS / 2, side="left")
    cnt = hi - lo
    has = cnt > 0
    out[has] = (cs[hi[has]] - cs[lo[has]]) / cnt[has]
    return out


def centred(a, valid):
    """a with its mean over `valid` removed and 0 where invalid; (values, mask as float)."""
    m = valid & np.isfinite(a)
    out = np.zeros(a.shape)
    if m.any():
        out[m] = a[m] - a[m].mean()
    return out, m.astype(float)


def lag_index(L_rows, first_row, n0):
    """Driver-grid index of the driver sample paired with target row j at each lag:
    idx[l, j] = (j - lag) - n0, for j = first_row .. first_row + L_rows - 1."""
    j = first_row + np.arange(L_rows)
    return (j[None, :] - LAGS[:, None]) - n0


def lag_sums(ysum, y2, ycnt, ynoise, xc, xm, idx):
    """Sums for every lag and bin. ysum, y2, ycnt, ynoise: (rows, bins); xc, xm: driver grid;
    idx: (lags, rows) grid index (out-of-range pairs are dropped). Returns (bins, lags, 5)."""
    ok = (idx >= 0) & (idx < len(xc))
    ii = np.where(ok, idx, 0)
    X = np.where(ok, xc[ii], 0.0)
    M = np.where(ok, xm[ii], 0.0)
    out = np.empty((ysum.shape[1], idx.shape[0], NSTAT))
    out[:, :, 0] = (X @ ysum).T
    out[:, :, 1] = ((X * X) @ ycnt).T
    out[:, :, 2] = (M @ y2).T
    out[:, :, 3] = (M @ ynoise).T
    out[:, :, 4] = (M @ ycnt).T
    return out


def corr(s):
    """Pearson r from sums (..., 5)."""
    with np.errstate(all="ignore"):
        return s[..., 0] / np.sqrt(s[..., 1] * s[..., 2])


def sigfrac(s):
    with np.errstate(all="ignore"):
        return 1.0 - s[..., 3] / s[..., 2]


def rho_bin(rho):
    r = np.asarray(rho, float)
    out = np.full(r.shape, -1, int)
    ok = np.isfinite(r) & (r > 0.0) & (r < 1.0)
    out[ok] = np.searchsorted(np.asarray(EDGES), r[ok], side="right") - 1
    return out


def membership(cb):
    B = np.zeros((len(cb), NB))
    for k in range(NB - 1):
        B[cb == k, k] = 1.0
    B[cb >= 0, NB - 1] = 1.0
    return B


# --------------------------------------------------------------------------- per shot

def block_of(d, target, rel):
    good = rg.good_masks(d, rel)[target]
    frames = np.nonzero(good.sum(1) >= 10)[0]
    if len(frames) == 0:
        return None
    a, b = int(frames[0]), int(frames[-1])
    return (a, b) if b - a + 1 >= 20 else None


def load_drivers(shot):
    out = {}
    with np.load(rg.ACT_DIR / f"{shot}.npz") as z:
        if "anb_time" in z.files:
            tb = z["anb_time"].astype(float) * 1000.0
            for k, name in (("ss", "anb_ss_sum_power"), ("tot", "anb_tot_sum_power")):
                if name in z.files and z[name].shape == tb.shape:
                    out[k] = (tb, z[name].astype(float))
    p = DALPHA_DIR / f"{shot}.npz"
    if p.exists():
        with np.load(p) as z:
            out["dalpha"] = (z["time"].astype(float) * 1000.0, z["dalpha"].astype(float))
    return out


def driver_grid(drv, t_a, L):
    """Frame-grid driver series for n = -PAD .. L + PAD - 1: levels and increments, centred."""
    n_hi = L + PAD - 1
    g = {}
    for k in ("ss", "tot", "dalpha"):
        if k not in drv:
            continue
        x = frame_means(drv[k][0], drv[k][1], t_a, -PAD, n_hi)
        dx = np.full(len(x), np.nan)
        dx[1:] = x[1:] - x[:-1]
        g[k] = (x, dx)
    if "ss" in g:
        x, dx = g["ss"]
        top = np.nanmax(x) if np.isfinite(x).any() else np.nan
        dxn = dx.copy()
        if np.isfinite(top) and top > 0:
            step = np.nonzero(np.abs(np.nan_to_num(dx)) >= STEP_FRAC * top)[0]
            for s in step:
                dxn[max(0, s - STEP_HALF): s + STEP_HALF + 1] = np.nan
        g["ss_nostep"] = (x, dxn)
    return g


def process(shot, geo, null_offsets=None):
    """Sums of one shot: array (kind, target, pop, driver, bin, lag, 5) or a status string.
    With null_offsets (confirmation shots): also (target, driver, bin, rep, lag, 3) for pop G."""
    d = rg.load_shot(shot)
    if d is None:
        return "no_file", None, None
    drv = load_drivers(shot)
    ax, ro = geo
    out = np.zeros((len(KINDS), len(rg.TARGETS), len(POPS), len(DRIVERS), NB, NL, NSTAT), np.float32)
    nul = None
    if null_offsets is not None:
        nul = np.zeros((len(rg.TARGETS), len(DRIVERS), NB, N_NULL, NL, 3))
    status, any_ok = "no_block", False
    for ti, tg in enumerate(rg.TARGETS):
        for pi, pop in enumerate(POPS):
            p = rg.prepare(d, tg, rg.POPS[pop])
            if p["status"] != "ok":
                continue
            if p["n_bad_dt"] > 0 or not p["t_finite"]:
                status = "irregular_times"
                continue
            a, _ = block_of(d, tg, rg.POPS[pop])
            t_a = float(d["time_ms"][a])
            L, G, Y, S2 = p["L"], p["G"], p["Y0"], p["S2"]
            cb = rho_bin((p["R"] - ax) / (ro - ax))
            B = membership(cb)
            if not B.any():
                continue
            any_ok = True
            grid = driver_grid(drv, t_a, L)
            # increments, rows j = 1 .. L-1
            v = G[1:] & G[:-1]
            dy = np.where(v, Y[1:] - Y[:-1], 0.0)
            cnt = np.maximum(v.sum(0), 1)
            dyc = np.where(v, dy - dy.sum(0) / cnt, 0.0)
            noise = np.where(v, S2[1:] + S2[:-1], 0.0)
            ysum, y2, ycnt, yn = dyc @ B, (dyc * dyc) @ B, v.astype(float) @ B, noise @ B
            idx = lag_index(L - 1, 1, -PAD)
            # levels, rows j = 0 .. L-1
            cg = np.maximum(G.sum(0), 1)
            yc = np.where(G, Y - Y.sum(0) / cg, 0.0)
            lsum, l2, lcnt, ln = yc @ B, (yc * yc) @ B, G.astype(float) @ B, S2 @ B
            idx0 = lag_index(L, 0, -PAD)
            for di, k in enumerate(DRIVERS):
                if k not in grid:
                    continue
                x, dx = grid[k]
                xc, xm = centred(dx, np.isfinite(dx))
                out[0, ti, pi, di] = lag_sums(ysum, y2, ycnt, yn, xc, xm, idx)
                lc, lm = centred(x, np.isfinite(x))
                out[1, ti, pi, di] = lag_sums(lsum, l2, lcnt, ln, lc, lm, idx0)
                if nul is not None and pop == "G":
                    N1 = len(dx) - 1                       # increments live at grid index 1 .. N1
                    span = N1 - 2 * (MAXLAG + 1)
                    off = MAXLAG + 1 + (null_offsets % max(span, 1))
                    ok = (idx >= 1) & (idx <= N1)
                    base = np.where(ok, idx, 1) - 1
                    rot = 1 + (base[None, :, :] + off[:, None, None]) % N1      # (rep, lag, row)
                    X = np.where(ok[None], xc[rot], 0.0)
                    M = np.where(ok[None], xm[rot], 0.0)
                    nul[ti, di, :, :, :, 0] = np.einsum("rlj,jb->brl", X, ysum)
                    nul[ti, di, :, :, :, 1] = np.einsum("rlj,jb->brl", X * X, ycnt)
                    nul[ti, di, :, :, :, 2] = np.einsum("rlj,jb->brl", M, y2)
    return ("ok" if any_ok else status), out, nul


# --------------------------------------------------------------------------- statistics

def ci(point, boots, ref=0.0):
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if not np.isfinite(point) or len(b) < 100:
        return {"point": None if not np.isfinite(point) else float(point), "lo": None, "hi": None, "side": "n/a"}
    lo, hi = np.percentile(b, [2.5, 97.5])
    side = "above" if lo > ref else ("below" if hi < ref else "includes")
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "side": side}


def boot_weights(n, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=rg.N_BOOT).astype(np.float32)


def cell_stats(S, W):
    """S: (shots, bins, lags, 5) float32 sums of one (kind, target, pop, driver) cell.
    Returns point sums (bins, lags, 5) and bootstrap sums (boot, bins, lags, 5)."""
    flat = S.reshape(S.shape[0], -1).astype(np.float64)
    point = flat.sum(0).reshape(S.shape[1:])
    boots = (W.astype(np.float64) @ flat).reshape((W.shape[0],) + S.shape[1:])
    return point, boots


def analyse_cell(Sd, Sc, Wc, kind):
    """One (kind, target, pop, driver): discovery picks l*, confirmation gives the numbers."""
    pd_ = Sd.astype(np.float64).sum(0)
    rd = corr(pd_)
    pc, bc = cell_stats(Sc, Wc)
    rc, rb = corr(pc), corr(bc)
    fc, fb = sigfrac(pc), sigfrac(bc)
    cell = {"bins": {}}
    for k, name in enumerate(BINS):
        prof_d, prof_c = rd[k], rc[k]
        e = {"r_discovery": [None if not np.isfinite(x) else float(x) for x in prof_d],
             "r_confirmation": [None if not np.isfinite(x) else float(x) for x in prof_c],
             "pairs_lag0_confirmation": float(pc[k, MAXLAG, 4]),
             "shots_confirmation": int((Sc[:, k, MAXLAG, 4] > 0).sum()),
             "shots_discovery": int((Sd[:, k, MAXLAG, 4] > 0).sum())}
        absd = np.where(POS & np.isfinite(prof_d), np.abs(prof_d), -1.0)
        if absd.max() < 0 or e["shots_confirmation"] < MIN_SHOTS:
            e["l_star_ms"] = None
            cell["bins"][name] = e
            continue
        li = int(np.argmax(absd))
        e["l_star_ms"] = float(LAGS[li] * rg.STEP_MS)
        e["l_star_index"] = li
        e["r_at_l_star"] = ci(rc[k, li], rb[:, k, li])
        e["abs_r_boot_index"] = li
        e["signal_fraction"] = ci(fc[k, li], fb[:, k, li], ref=F_MIN)
        if kind == "diff" and e["signal_fraction"]["side"] == "above":
            with np.errstate(all="ignore"):
                e["r_corr_at_l_star"] = ci(rc[k, li] / np.sqrt(fc[k, li]), rb[:, k, li] / np.sqrt(fb[:, k, li]))
        else:
            e["r_corr_at_l_star"] = None
        absc = np.where(POS & np.isfinite(prof_c), np.abs(prof_c), -1.0)
        lc = int(np.argmax(absc))
        e["lag_of_max_confirmation_ms"] = float(LAGS[lc] * rg.STEP_MS)
        e["max_abs_r_pos_confirmation"] = float(absc[lc])
        absn = np.where(NEG & np.isfinite(prof_c), np.abs(prof_c), -1.0)
        e["max_abs_r_neg_confirmation"] = float(absn.max())
        e["r_lag0"] = ci(rc[k, MAXLAG], rb[:, k, MAXLAG])
        cell["bins"][name] = e
    cell["_rb"] = rb
    cell["_rc"] = rc
    return cell


def null_pvalues(nul, cell):
    """nul: (bins, rep, lags, 3) pooled null sums; adds p-values for the max over lag sets."""
    with np.errstate(all="ignore"):
        rn = np.abs(nul[..., 0] / np.sqrt(nul[..., 1] * nul[..., 2]))
    for k, name in enumerate(BINS):
        e = cell["bins"][name]
        if e.get("l_star_ms") is None:
            continue
        for tag, sel, obs in (("pos", POS, e["max_abs_r_pos_confirmation"]),
                              ("neg", NEG, e["max_abs_r_neg_confirmation"])):
            m = np.nanmax(rn[k][:, sel], axis=1)
            m = m[np.isfinite(m)]
            e[f"null_p_{tag}"] = float((1 + (m >= obs).sum()) / (1 + len(m))) if len(m) else None
            e[f"null_q95_{tag}"] = float(np.percentile(m, 95)) if len(m) else None


def strip(cell):
    return {k: v for k, v in cell.items() if not k.startswith("_")}


def fmt(e, nd=3):
    if not e or e.get("point") is None:
        return "n/a"
    if e.get("lo") is None:
        return f"{e['point']:+.{nd}f}"
    return f"{e['point']:+.{nd}f} [{e['lo']:+.{nd}f}, {e['hi']:+.{nd}f}]"


def side(e, ref="0"):
    if not e or e.get("side") in (None, "n/a"):
        return "n/a"
    return {"above": f"excludes {ref} (above)", "below": f"excludes {ref} (below)",
            "includes": f"includes {ref}"}[e["side"]]


# --------------------------------------------------------------------------- main

def main():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    shots = [int(x) for x in SHOTS_TXT.read_text().split()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "campaign", "heating", "rmp_coil",
                                            "radii_magnetic__radius_efit", "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")

    def geo_of(s):
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        ro = float(cat.loc[s, "radii_router_radius_efit"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            return ax, ro
        ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
        return (ax, ro) if np.isfinite(ax) and np.isfinite(ro) and ro > ax else None

    shape = (len(shots), len(KINDS), len(rg.TARGETS), len(POPS), len(DRIVERS), NB, NL, NSTAT)
    S = np.lib.format.open_memmap(SUMS, mode="w+", dtype=np.float32, shape=shape)
    NULL = np.zeros((len(rg.TARGETS), len(DRIVERS), NB, N_NULL, NL, 3))
    status = {}
    used = np.zeros(len(shots), bool)
    has_da = np.zeros(len(shots), bool)
    for i, s in enumerate(shots):
        geo = geo_of(s)
        if geo is None:
            status["no_geometry"] = status.get("no_geometry", 0) + 1
            continue
        offs = None
        if s % 2 == 0:
            offs = np.random.default_rng([rg.SEED, s]).integers(0, 10 ** 6, N_NULL)
        st, out, nul = process(s, geo, offs)
        status[st] = status.get(st, 0) + 1
        if out is not None and st == "ok":
            S[i] = out
            used[i] = True
            has_da[i] = (DALPHA_DIR / f"{s}.npz").exists()
            if nul is not None:
                NULL += np.nan_to_num(nul)
        if (i + 1) % 500 == 0:
            log(f"{i + 1}/{len(shots)} shots, used {int(used.sum())}")
    S.flush()
    sh = np.array(shots)
    disc = np.nonzero(used & (sh % 2 == 1))[0]
    conf = np.nonzero(used & (sh % 2 == 0))[0]
    log(f"used {int(used.sum())}: discovery {len(disc)}, confirmation {len(conf)}; status {status}")
    rng = np.random.default_rng(rg.SEED)
    Wc = boot_weights(len(conf), rng)

    res = {"registration": REG, "seed": rg.SEED, "n_boot": rg.N_BOOT, "n_null": N_NULL,
           "lags_ms": [float(x * rg.STEP_MS) for x in LAGS], "bins": list(BINS),
           "counts": {"listed": len(shots), "status": status, "used": int(used.sum()),
                      "discovery": int(len(disc)), "confirmation": int(len(conf)),
                      "with_dalpha_file_used": int(has_da.sum()),
                      "dalpha_files_total": len(list(DALPHA_DIR.glob("*.npz")))},
           "cells": {}}
    keep = {}
    for ki, kind in enumerate(KINDS):
        for ti, tg in enumerate(rg.TARGETS):
            for pi, pop in enumerate(POPS):
                for di, dr in enumerate(DRIVERS):
                    Sd = np.asarray(S[disc, ki, ti, pi, di])
                    Sc = np.asarray(S[conf, ki, ti, pi, di])
                    cell = analyse_cell(Sd, Sc, Wc, kind)
                    if kind == "diff" and pop == "G":
                        null_pvalues(NULL[ti, di], cell)
                    key = f"{kind}|{tg}|{pop}|{dr}"
                    keep[key] = cell
                    res["cells"][key] = strip(cell)
        log(f"analysed kind {kind}")

    # ---------------- statements
    def get(tg, dr, b, pop="G", kind="diff"):
        return keep[f"{kind}|{tg}|{pop}|{dr}"]["bins"][b]

    st = {}
    v = get("V_i", "ss", "all")
    c1 = v.get("l_star_ms") is not None and 5.0 <= v["l_star_ms"] <= 100.0 and v["r_at_l_star"]["side"] == "above"
    st["C1"] = {"holds": bool(c1), "l_star_ms": v.get("l_star_ms"), "r": v.get("r_at_l_star"),
                "lag_of_max_confirmation_ms": v.get("lag_of_max_confirmation_ms"),
                "r_corr": v.get("r_corr_at_l_star"), "null_p_pos": v.get("null_p_pos")}
    t = get("T_i", "ss", "all")
    c2a = t.get("l_star_ms") is not None and t["r_at_l_star"]["side"] == "above"
    c2b = (t.get("l_star_ms") is not None and v.get("l_star_ms") is not None
           and t["l_star_ms"] <= v["l_star_ms"]
           and t["lag_of_max_confirmation_ms"] <= v["lag_of_max_confirmation_ms"])
    st["C2"] = {"holds": bool(c2a and c2b), "association": bool(c2a), "not_later_than_rotation": bool(c2b),
                "l_star_ms": t.get("l_star_ms"), "r": t.get("r_at_l_star"),
                "lag_of_max_confirmation_ms": t.get("lag_of_max_confirmation_ms"),
                "null_p_pos": t.get("null_p_pos")}
    cellTD = keep["diff|T_i|G|dalpha"]
    core = cellTD["bins"]["0-0.3"]
    c3 = {"bins": {}, "holds": False}
    for b in ("0.7-0.9", "0.9-1"):
        e = cellTD["bins"][b]
        row = {"l_star_ms": e.get("l_star_ms"), "r": e.get("r_at_l_star"), "shots": e["shots_confirmation"]}
        if e.get("l_star_ms") is not None and core.get("l_star_ms") is not None:
            kb, kc = BINS.index(b), BINS.index("0-0.3")
            rb = cellTD["_rb"]
            dboot = np.abs(rb[:, kb, e["l_star_index"]]) - np.abs(rb[:, kc, core["l_star_index"]])
            dpoint = abs(e["r_at_l_star"]["point"]) - abs(core["r_at_l_star"]["point"])
            row["abs_r_minus_core"] = ci(dpoint, dboot)
            ok = (e["r_at_l_star"]["side"] in ("above", "below") and e["l_star_ms"] <= 5.0
                  and row["abs_r_minus_core"]["side"] == "above")
            row["meets"] = bool(ok)
            c3["holds"] = c3["holds"] or bool(ok)
        c3["bins"][b] = row
    c3["core"] = {"l_star_ms": core.get("l_star_ms"), "r": core.get("r_at_l_star")}
    st["C3"] = c3
    outer = None
    for b in ("0.9-1", "0.7-0.9", "0.5-0.7", "0.3-0.5"):
        e = cellTD["bins"][b]
        if e.get("l_star_ms") is not None and e["r_at_l_star"]["side"] in ("above", "below"):
            outer = b
            break
    c4 = {"outer_bin": outer, "holds": False}
    if outer is not None and core.get("l_star_ms") is not None:
        o = cellTD["bins"][outer]
        c4.update({"core_l_star_ms": core["l_star_ms"], "outer_l_star_ms": o["l_star_ms"],
                   "core_lag_of_max_confirmation_ms": core["lag_of_max_confirmation_ms"],
                   "outer_lag_of_max_confirmation_ms": o["lag_of_max_confirmation_ms"]})
        c4["holds"] = bool(core["l_star_ms"] > o["l_star_ms"]
                           and core["lag_of_max_confirmation_ms"] > o["lag_of_max_confirmation_ms"])
    st["C4"] = c4
    pp, pn = v.get("null_p_pos"), v.get("null_p_neg")
    st["C5"] = {"holds": bool(pp is not None and pn is not None and pp < 0.05 and pn > 0.05),
                "null_p_pos": pp, "null_p_neg": pn,
                "max_abs_r_pos": v.get("max_abs_r_pos_confirmation"),
                "max_abs_r_neg": v.get("max_abs_r_neg_confirmation")}
    res["statements"] = st

    # ---------------- strata (confirmation, at the discovery l*)
    camp = cat["campaign"].astype(str)
    heat = cat["heating"].astype(str)
    rmp = cat["rmp_coil"]
    strata = {}
    for c in CAMPS:
        strata[f"campaign {c}"] = [i for i in conf if camp.get(shots[i]) == c]
    strata["heating SS only"] = [i for i in conf if heat.get(shots[i]) == "SS Beam"]
    strata["heating 2 beams"] = [i for i in conf if heat.get(shots[i]) == "2 Beams,SS Beam,SW Beam"]
    strata["RMP on"] = [i for i in conf if rmp.get(shots[i]) is True or rmp.get(shots[i]) == True]  # noqa: E712
    strata["RMP off"] = [i for i in conf if rmp.get(shots[i]) is False or rmp.get(shots[i]) == False]  # noqa: E712
    rows = [("V_i", "ss", "all"), ("T_i", "ss", "all"), ("T_i", "dalpha", "0.7-0.9"),
            ("T_i", "dalpha", "0.9-1"), ("V_i", "dalpha", "0.7-0.9")]
    res["strata"] = {}
    for name, idxs in strata.items():
        ent = {"shots": len(idxs), "rows": {}}
        if len(idxs) >= MIN_SHOTS:
            W = boot_weights(len(idxs), np.random.default_rng([rg.SEED, len(idxs)]))
            for tg, dr, b in rows:
                base = get(tg, dr, b)
                if base.get("l_star_ms") is None:
                    continue
                ti, di, kb = rg.TARGETS.index(tg), DRIVERS.index(dr), BINS.index(b)
                Ss = np.asarray(S[np.array(idxs), 0, ti, 0, di, kb, base["l_star_index"]])
                n_sh = int((Ss[:, 4] > 0).sum())
                if n_sh < MIN_SHOTS:
                    ent["rows"][f"{tg}|{dr}|{b}"] = {"shots": n_sh, "r": None}
                    continue
                pt = Ss.astype(np.float64).sum(0)
                bt = W.astype(np.float64) @ Ss.astype(np.float64)
                ent["rows"][f"{tg}|{dr}|{b}"] = {"shots": n_sh, "r": ci(corr(pt), corr(bt))}
        res["strata"][name] = ent

    res["expectations"] = expectations(res)
    res["notes"] = [
        "Positive lag = the driver increment comes first. A frame stamp is taken as the centre of "
        "its 5 ms exposure; the convention is not in the metadata (half a frame of uncertainty).",
        "Target increments are centred per chord and driver increments per shot over their valid "
        "samples, not over the set of pairs of each lag.",
        "The circular-shift null rotates the driver increments of each confirmation shot on its own "
        "frame grid by a random offset of at least 21 frames; it is computed for the increments in "
        "population G only.",
        "Shots whose block holds a frame interval outside 5.0 +- 0.1 ms or a non-finite stamp are "
        "excluded (status irregular_times counts target-population cells that hit the rule in a "
        "shot with no usable cell).",
        "Charge exchange exists only while the viewed beam is on: beam power, the existence of the "
        "measurement and its fit quality are tied, so a beam association is not only a plasma response.",
    ]
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    log(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    print("C1", st["C1"]["holds"], "C2", st["C2"]["holds"], "C3", st["C3"]["holds"],
          "C4", st["C4"]["holds"], "C5", st["C5"]["holds"])


def expectations(res):
    st = res["statements"]
    out = []

    def add(name, expected, found, met):
        out.append({"item": name, "expected": expected, "found": found, "met": bool(met)})

    c1 = st["C1"]
    r = (c1.get("r") or {}).get("point")
    add("C1 l* (V_i, beam)", "10-40 ms", c1.get("l_star_ms"),
        c1.get("l_star_ms") is not None and 10 <= c1["l_star_ms"] <= 40)
    add("C1 r", "+0.02 to +0.12", r, r is not None and 0.02 <= r <= 0.12)
    c2 = st["C2"]
    add("C2 l* (T_i, beam)", "5-30 ms", c2.get("l_star_ms"),
        c2.get("l_star_ms") is not None and 5 <= c2["l_star_ms"] <= 30)
    for b, row in st["C3"]["bins"].items():
        rr = (row.get("r") or {}).get("point")
        add(f"C3 sign at {b}", "negative", rr, rr is not None and rr < 0)
        add(f"C3 |r| at {b}", "0.02-0.10", None if rr is None else abs(rr),
            rr is not None and 0.02 <= abs(rr) <= 0.10)
    for k in ("C1", "C2", "C3", "C4", "C5"):
        add(f"{k} holds", "yes", st[k]["holds"], st[k]["holds"])
    return out


def render_md(res):
    L = []
    c = res["counts"]
    st = res["statements"]
    cells = res["cells"]
    L.append("# MAST coupling study -- lagged association of charge-exchange increments with beam power and D-alpha")
    L.append("")
    L.append(f"Generated by `ces_prediction/experiments/mast/run_coupling.py`. Registration: `{res['registration']}`. "
             "Every number below is from the confirmation set (even shot numbers); the lag l* at which a "
             "coupling is read was chosen on the discovery set (odd shot numbers). r = pooled within-shot "
             "correlation of 5 ms increments; interval = 95 % shot-cluster bootstrap "
             f"({res['n_boot']} resamples); positive lag = the driver comes first.")
    L.append("")
    L.append("## 1. Counts")
    L.append("")
    L.append(f"- Listed shots {c['listed']}; per-shot status {c['status']}.")
    L.append(f"- Used {c['used']}: discovery {c['discovery']}, confirmation {c['confirmation']}. "
             f"Used shots with a D-alpha file: {c['with_dalpha_file_used']} "
             f"(D-alpha files on disk when the run started: {c['dalpha_files_total']}).")
    L.append("")
    L.append("## 2. Statements")
    L.append("")
    L.append("| statement | verdict | read-out |")
    L.append("|---|---|---|")
    c1 = st["C1"]
    L.append(f"| C1 rotation follows the beam with a delay (V_i, ss_sum_power, all radii, G) | "
             f"{'holds' if c1['holds'] else 'does not hold'} | l* = {c1['l_star_ms']} ms; r = {fmt(c1['r'])}, "
             f"{side(c1['r'])}; lag of the maximum on the confirmation set {c1['lag_of_max_confirmation_ms']} ms; "
             f"noise-corrected {fmt(c1['r_corr']) if c1['r_corr'] else 'not quotable'}; circular-shift p (positive lags) {c1['null_p_pos']} |")
    c2 = st["C2"]
    L.append(f"| C2 T_i follows the beam, no later than rotation | {'holds' if c2['holds'] else 'does not hold'} | "
             f"l* = {c2['l_star_ms']} ms; r = {fmt(c2['r'])}, {side(c2['r'])}; association {c2['association']}, "
             f"not later than rotation {c2['not_later_than_rotation']}; lag of the maximum on the confirmation set "
             f"{c2['lag_of_max_confirmation_ms']} ms; circular-shift p {c2['null_p_pos']} |")
    c3 = st["C3"]
    parts = []
    for b, row in c3["bins"].items():
        parts.append(f"{b}: l* = {row['l_star_ms']} ms, r = {fmt(row.get('r'))}, {side(row.get('r'))}, "
                     f"|r| minus core {fmt(row.get('abs_r_minus_core'))}, {row['shots']} shots")
    parts.append(f"core 0-0.3: l* = {c3['core']['l_star_ms']} ms, r = {fmt(c3['core']['r'])}")
    L.append(f"| C3 D-alpha association at the edge and short lag (T_i, G) | {'holds' if c3['holds'] else 'does not hold'} | "
             + "; ".join(parts) + " |")
    c4 = st["C4"]
    L.append(f"| C4 the lag grows inward (T_i, D-alpha) | {'holds' if c4['holds'] else 'does not hold'} | "
             f"outermost bin with an interval excluding 0: {c4['outer_bin']}; l* core {c4.get('core_l_star_ms')} ms, "
             f"outer {c4.get('outer_l_star_ms')} ms; lag of the maximum on the confirmation set core "
             f"{c4.get('core_lag_of_max_confirmation_ms')} ms, outer {c4.get('outer_lag_of_max_confirmation_ms')} ms |")
    c5 = st["C5"]
    L.append(f"| C5 direction check (V_i, beam) | {'holds' if c5['holds'] else 'does not hold'} | "
             f"max abs r over positive lags {c5['max_abs_r_pos']:.4f} (p {c5['null_p_pos']}), over negative lags "
             f"{c5['max_abs_r_neg']:.4f} (p {c5['null_p_neg']}) |")
    L.append("")
    L.append("## 3. Increments, population G: r at the discovery lag, per radius")
    L.append("")
    for tg in rg.TARGETS:
        for dr in DRIVERS:
            cell = cells[f"diff|{tg}|G|{dr}"]
            L.append(f"**{tg} vs {dr}**")
            L.append("")
            L.append("| rho | shots | l* (ms) | r at l* | vs 0 | r at lag 0 | lag of max, confirmation (ms) | "
                     "signal fraction | noise-corrected r | null p (pos / neg) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for b in BINS:
                e = cell["bins"][b]
                if e.get("l_star_ms") is None:
                    L.append(f"| {b} | {e['shots_confirmation']} | too few | | | | | | | |")
                    continue
                L.append(f"| {b} | {e['shots_confirmation']} | {e['l_star_ms']:.0f} | {fmt(e['r_at_l_star'])} | "
                         f"{side(e['r_at_l_star'])} | {fmt(e['r_lag0'])} | {e['lag_of_max_confirmation_ms']:.0f} | "
                         f"{fmt(e['signal_fraction'])} | {fmt(e['r_corr_at_l_star']) if e['r_corr_at_l_star'] else 'not quotable'} | "
                         f"{e.get('null_p_pos')} / {e.get('null_p_neg')} |")
            L.append("")
    L.append("## 4. Lag profiles, all radii, population G (confirmation set)")
    L.append("")
    L.append("| lag (ms) | " + " | ".join(f"{tg} vs {dr}" for tg in rg.TARGETS for dr in DRIVERS) + " |")
    L.append("|---|" + "---|" * (len(rg.TARGETS) * len(DRIVERS)))
    for li, lag in enumerate(res["lags_ms"]):
        vals = []
        for tg in rg.TARGETS:
            for dr in DRIVERS:
                x = cells[f"diff|{tg}|G|{dr}"]["bins"]["all"]["r_confirmation"][li]
                vals.append("n/a" if x is None else f"{x:+.4f}")
        L.append(f"| {lag:+.0f} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("## 5. Co-reported: population W and the level read-out (all radii)")
    L.append("")
    L.append("| read-out | target | driver | l* (ms) | r at l* | vs 0 |")
    L.append("|---|---|---|---|---|---|")
    for kind, pop in (("diff", "W"), ("level", "G")):
        for tg in rg.TARGETS:
            for dr in DRIVERS:
                e = cells[f"{kind}|{tg}|{pop}|{dr}"]["bins"]["all"]
                if e.get("l_star_ms") is None:
                    continue
                L.append(f"| {'increments, W' if kind == 'diff' else 'levels, G'} | {tg} | {dr} | "
                         f"{e['l_star_ms']:.0f} | {fmt(e['r_at_l_star'])} | {side(e['r_at_l_star'])} |")
    L.append("")
    L.append("## 6. Strata (confirmation set, increments, G, at the discovery l*; not registered tests)")
    L.append("")
    keys = ["V_i|ss|all", "T_i|ss|all", "T_i|dalpha|0.7-0.9", "T_i|dalpha|0.9-1", "V_i|dalpha|0.7-0.9"]
    L.append("| stratum | shots | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for name, ent in res["strata"].items():
        vals = []
        for k in keys:
            row = ent["rows"].get(k)
            vals.append("too few" if not row or not row.get("r") else fmt(row["r"]))
        L.append(f"| {name} | {ent['shots']} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("## 7. Next to the KSTAR bound")
    L.append("")
    v = cells["diff|V_i|G|ss"]["bins"]["all"]
    vd = cells["diff|V_i|G|dalpha"]["bins"]["all"]
    L.append(f"KSTAR, one edge-side chord: rotation increments vs co-located fast-channel increments, noise-corrected "
             f"|r| <= {KSTAR_BOUND} over lags 0-200 ms. MAST, all radii, population G: rotation increments vs "
             f"beam-power increments r = {fmt(v.get('r_at_l_star'))} at {v.get('l_star_ms')} ms; vs mid-plane "
             f"D-alpha increments r = {fmt(vd.get('r_at_l_star'))} at {vd.get('l_star_ms')} ms. Different device, "
             "different drivers, different radii; the MAST numbers are not support for the KSTAR claim.")
    L.append("")
    L.append("## 8. Expectation check")
    L.append("")
    L.append("| item | expected | found | met |")
    L.append("|---|---|---|---|")
    for e in res["expectations"]:
        f_ = e["found"]
        f_ = f"{f_:+.4f}" if isinstance(f_, float) else str(f_)
        L.append(f"| {e['item']} | {e['expected']} | {f_} | {'yes' if e['met'] else 'no'} |")
    L.append("")
    L.append("## 9. Notes")
    L.append("")
    for n in res["notes"]:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
