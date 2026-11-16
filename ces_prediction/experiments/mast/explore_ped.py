"""MAST exploration at the pedestal top (discovery shots only): which inputs lower the error of a
hidden 5 ms charge-exchange frame near the edge, against hold and interpolation.

Protocol note: ces_prediction/experiments/MAST_FAST_EXPLORATION.md.  Only the odd-numbered shots of
coupling_shots.txt are read.  Inputs added to the charge-exchange past (explore_fast.py):
  ts    Thomson scattering T_e and n_e at the chord major radius (core system ayc, or atm), frame
        windows of the hidden frame and the three frames before it
  dens  CO2 interferometer line-integrated density, frame-window means
  lock  saddle-coil locked-mode amplitude and the H-mode rating
  da    mid-plane D-alpha (as in explore_fast.py)
The edge position is taken from each Thomson density profile (outermost radius where n_e is at
least half of the 90th percentile of the outboard profile), so that a chord is placed by its
distance to the measured edge at that time instead of one outer radius per shot.

CPU only.  Run from the repo root:
    py ces_prediction/experiments/mast/explore_ped.py build
    py ces_prediction/experiments/mast/explore_ped.py fit
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
import run_coupling as rc  # noqa: E402
import run_edge as re_  # noqa: E402
import explore_fast as ef  # noqa: E402

EXTRA = rg.ROOT / "data" / "mast" / "extra"
OUT = rg.ROOT / "data" / "mast"
POP, FRAC = "W", 0.5
OFFS = (0, -1, -2, -3)              # frame offsets of the Thomson / density / coil windows
MAX_R_GAP = 0.03                    # m: a chord takes T_e only between valid points this close
EDGE_BANDS = ((-0.02, 0.015, "edge -2..1.5 cm (gradient side)"),
              (0.015, 0.05, "edge 1.5..5 cm (pedestal top)"),
              (0.05, 0.10, "edge 5..10 cm"), (0.10, 0.20, "edge 10..20 cm"),
              (0.20, 9.0, "edge > 20 cm"))
SEED = 20261118
N_BOOT = 400


def window_mean(t_ms, v, t_frames, off):
    """Mean of v over [t + 5 off - 2.5, t + 5 off + 2.5) ms for each frame time t (rows of v)."""
    lo = np.searchsorted(t_ms, t_frames + rg.STEP_MS * off - 2.5, side="left")
    hi = np.searchsorted(t_ms, t_frames + rg.STEP_MS * off + 2.5, side="left")
    out = np.full((len(t_frames),) + v.shape[1:], np.nan)
    for i in range(len(t_frames)):
        if hi[i] > lo[i]:
            with np.errstate(all="ignore"):
                out[i] = np.nanmean(v[lo[i]:hi[i]], axis=0)
    return out


def to_chords(Rc, Rv, v):
    if len(Rv) < 2:
        return np.full(len(Rc), np.nan)
    o = np.argsort(Rv)
    Rv, v = Rv[o], v[o]
    out = np.interp(Rc, Rv, v, left=np.nan, right=np.nan)
    i = np.searchsorted(Rv, Rc)
    i0, i1 = np.clip(i - 1, 0, len(Rv) - 1), np.clip(i, 0, len(Rv) - 1)
    out[(Rc - Rv[i0] > MAX_R_GAP) | (Rv[i1] - Rc > MAX_R_GAP)] = np.nan
    return out


def edge_radius(R, ne, r_axis):
    """Outermost radius where n_e >= half of the 90th percentile of the outboard profile."""
    ok = np.isfinite(R) & np.isfinite(ne) & (R > r_axis)
    if ok.sum() < 6:
        return np.nan
    o = np.argsort(R[ok])
    r, n = R[ok][o], ne[ok][o]
    half = 0.5 * np.quantile(n, 0.9)
    above = np.flatnonzero(n >= half)
    if len(above) == 0:
        return np.nan
    j = above[-1]
    if j + 1 < len(r) and n[j + 1] < half:      # linear crossing to the next outer point
        return float(r[j] + (r[j + 1] - r[j]) * (n[j] - half) / (n[j] - n[j + 1]))
    return float(r[j])


def load_extra(shot):
    p = EXTRA / f"{shot}.npz"
    if not p.exists():
        return {}
    out = {}
    with np.load(p) as z:
        g = "ayc" if "ayc_te" in z.files else "atm" if "atm_te" in z.files else None
        if g:
            t = z[g + "_time"].astype(float) * 1000.0
            te, er, ne = (z[g + "_" + k].astype(float) for k in ("te", "te_error", "ne"))
            R = z[g + "_radius"].astype(float)
            if R.ndim == 1:
                R = np.broadcast_to(R, te.shape).copy()
            with np.errstate(all="ignore"):
                ok = (np.isfinite(te) & np.isfinite(er) & np.isfinite(ne) & np.isfinite(R)
                      & (te > 0) & (er > 0) & (ne > 0) & (er / te < 1.0))
            keep = np.isfinite(t) & (ok.sum(1) >= 6)
            o = np.argsort(t[keep], kind="stable")
            out["ts"] = (t[keep][o], R[keep][o], np.where(ok, te, np.nan)[keep][o],
                         np.where(ok, ne, np.nan)[keep][o], g)
        for key, tn, vn in (("dens", "ane_time", "ane_density"), ("lock", "asm_out_time", "asm_out_signal"),
                            ("hm", "asm_hm_time", "asm_hm_rating")):
            if vn in z.files and z[vn].shape == z[tn].shape:
                tt, vv = z[tn].astype(float) * 1000.0, z[vn].astype(float)
                okk = np.isfinite(tt) & np.isfinite(vv)
                if okk.sum() > 10:
                    oo = np.argsort(tt[okk], kind="stable")
                    out[key] = (tt[okk][oo], vv[okk][oo])
    return out


def add_extra(r, shot, r_axis):
    """Extra columns for the row table r of one shot (explore_fast.shot_rows)."""
    n = len(r["y"])
    ex = load_extra(shot)
    frames, fi = np.unique(r["t0"], return_inverse=True)
    chords, ci = np.unique(r["R"], return_inverse=True)
    frames = frames.astype(float)
    cols = {}
    for k in OFFS:
        for q in ("te", "ne"):
            cols[f"{q}{-k}"] = np.full(n, np.nan, np.float32)
    cols["d_edge"] = np.full(n, np.nan, np.float32)
    cols["ts_sys"] = np.zeros(n, np.int8)
    if "ts" in ex:
        t, R, te, ne, g = ex["ts"]
        TE = np.stack([to_chords(chords, R[j][np.isfinite(te[j])], te[j][np.isfinite(te[j])])
                       for j in range(len(t))])
        NE = np.stack([to_chords(chords, R[j][np.isfinite(ne[j])], ne[j][np.isfinite(ne[j])])
                       for j in range(len(t))])
        for k in OFFS:
            cols[f"te{-k}"] = window_mean(t, TE, frames, k)[fi, ci].astype(np.float32)
            cols[f"ne{-k}"] = (window_mean(t, NE, frames, k)[fi, ci] / 1e19).astype(np.float32)
        edge = np.array([edge_radius(R[j], ne[j], r_axis) for j in range(len(t))])
        # edge position of a frame: median of the profiles within three frames of it
        lo = np.searchsorted(t, frames - 7.5, side="left")
        hi = np.searchsorted(t, frames + 7.5, side="right")
        ef_ = np.full(len(frames), np.nan)
        for i in range(len(frames)):
            if hi[i] > lo[i] and np.isfinite(edge[lo[i]:hi[i]]).any():
                ef_[i] = np.nanmedian(edge[lo[i]:hi[i]])
        cols["d_edge"] = (ef_[fi] - chords[ci]).astype(np.float32)
        cols["ts_sys"][:] = 2 if g == "ayc" else 1
    for key, pre in (("dens", "de"), ("lock", "lk"), ("hm", "hm")):
        for k in OFFS[:3]:
            cols[f"{pre}{-k}"] = np.full(n, np.nan, np.float32)
        if key in ex:
            tt, vv = ex[key]
            w = [window_mean(tt, vv, frames, k) for k in OFFS[:3]]
            sc = 1.0
            if key != "hm":
                blk = np.abs(w[0][np.isfinite(w[0])])
                sc = np.median(blk) if len(blk) and np.median(blk) > 0 else np.nan
            for k, wk in zip(OFFS[:3], w):
                cols[f"{pre}{-k}"] = (wk[fi] / sc).astype(np.float32)
    r.update(cols)
    return r


def build():
    t0 = time.time()
    listed = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    disc = [s for s in listed if s % 2 == 1 and (EXTRA / f"{s}.npz").exists()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit",
                                            "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")
    add1 = json.loads(re_.ADD1_JSON.read_text(encoding="utf-8"))

    def geo_of(s):
        if s not in cat.index:
            return None
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        for col in ("radii_router_radius_efit", "radii_router_radius_d_alpha"):
            ro = float(cat.loc[s, col])
            if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
                return ax, ro
        return None

    print(f"{len(disc)} discovery shots with an extra file", flush=True)
    for tg in rg.TARGETS:
        params = re_.params_from(add1, tg)
        parts = []
        for i, s in enumerate(disc):
            geo = geo_of(s)
            r = ef.shot_rows(s, geo, params, tg, POP, FRAC)
            if r is not None:
                parts.append(add_extra(r, s, geo[0]))
            if (i + 1) % 300 == 0:
                print(f"[{time.time() - t0:5.0f}s] {tg} {i + 1}/{len(disc)} shots, "
                      f"{len(parts)} with rows", flush=True)
        tab = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(OUT / f".ped_rows_{tg}.npz", **tab)
        print(f"{tg}: {len(parts)} shots, {len(tab['y'])} rows, Thomson at the hidden and the last "
              f"visible frame on {int((np.isfinite(tab['te0']) & np.isfinite(tab['te1'])).sum())} rows, "
              f"edge distance on {int(np.isfinite(tab['d_edge']).sum())}", flush=True)


# --------------------------------------------------------------------------- inputs and models

def group_features(t, groups):
    y1 = t["y1"]
    f = {"y1": y1, "d13": y1 - t["y3"], "d35": t["y3"] - t["y5"], "d57": t["y5"] - t["y7"],
         "sf": t["sf"] - y1, "rm": t["rm"] - y1, "s1": t["s1"], "s3": t["s3"], "rho": t["rho"]}
    if "da" in groups:
        f.update({"da01": t["da0"] - t["da1"], "da12": t["da1"] - t["da2"], "da23": t["da2"] - t["da3"],
                  "dmx0": t["dmx0"] - t["da1"], "dmx1": t["dmx1"] - t["da2"], "da0": t["da0"]})
    if "ts" in groups:
        with np.errstate(all="ignore"):
            f.update({"te0": t["te0"], "te1": t["te1"], "dte01": t["te0"] - t["te1"],
                      "dte12": t["te1"] - t["te2"], "dte13": t["te1"] - t["te3"],
                      "rte01": t["te0"] / t["te1"], "y1_rte": y1 * (t["te0"] / t["te1"] - 1.0),
                      "y1_te1": y1 / t["te1"], "ne0": t["ne0"], "dne01": t["ne0"] - t["ne1"],
                      "dne12": t["ne1"] - t["ne2"], "rne01": t["ne0"] / t["ne1"],
                      "d_edge": t["d_edge"]})
    if "dens" in groups:
        f.update({"de0": t["de0"], "dde01": t["de0"] - t["de1"], "dde12": t["de1"] - t["de2"]})
    if "lock" in groups:
        f.update({"lk0": t["lk0"], "dlk01": t["lk0"] - t["lk1"], "hm0": t["hm0"],
                  "dhm01": t["hm0"] - t["hm1"]})
    X = np.column_stack(list(f.values())).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


MODELS = (("gb_ces", ()), ("gb_da", ("da",)), ("gb_ts", ("ts",)), ("gb_dens", ("dens",)),
          ("gb_lock", ("lock",)), ("gb_all", ("da", "ts", "dens", "lock")),
          ("gb_all_no_ts", ("da", "dens", "lock")))
PAIRS = (("gb_ces", "persistence"), ("gb_ts", "gb_ces"), ("gb_da", "gb_ces"), ("gb_dens", "gb_ces"),
         ("gb_lock", "gb_ces"), ("gb_all", "gb_all_no_ts"), ("gb_all", "gb_ces"),
         ("gb_all", "persistence"), ("gb_all", "interp"), ("interp", "persistence"))


def fit():
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    for tg in rg.TARGETS:
        z = np.load(OUT / f".ped_rows_{tg}.npz")
        t = {k: z[k] for k in z.files}
        n = len(t["y"])
        shots = np.unique(t["shot"])
        fold_of = dict(zip(shots.tolist(), rng.permutation(len(shots)) % 5))
        fold = np.array([fold_of[s] for s in t["shot"].tolist()])
        resid = (t["y"] - t["y1"]).astype(np.float64)
        scale = float(np.median(np.sqrt(t["s0"])))
        target = np.clip(resid, -20 * scale, 20 * scale)
        preds = {"persistence": np.zeros(n),
                 "slope_filter": (t["sf"] - t["y1"]).astype(np.float64),
                 "interp": (0.5 * (t["y1"] + t["yn"]) - t["y1"]).astype(np.float64)}
        for name, groups in MODELS:
            X = group_features(t, groups)
            pr = np.zeros(n)
            for k in range(5):
                tr, te = fold != k, fold == k
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                  min_samples_leaf=200, l2_regularization=1.0,
                                                  random_state=SEED)
                m.fit(X[tr], target[tr])
                pr[te] = m.predict(X[te])
            preds[name] = pr
            print(f"  fitted {tg} {name}", flush=True)
        np.savez(OUT / f".ped_preds_{tg}.npz", fold=fold, **preds)
        report(tg, t, resid, preds, rng)


def report(tg, t, resid, preds, rng):
    shots, inv = np.unique(t["shot"], return_inverse=True)
    ns = len(shots)
    W = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=N_BOOT).astype(float)
    se = {k: (resid - v) ** 2 for k, v in preds.items()}
    has_ts = np.isfinite(t["te0"]) & np.isfinite(t["te1"])
    d = t["d_edge"]
    excess = resid ** 2 - (t["s0"] + t["s1"])
    with np.errstate(all="ignore"):
        dte = np.abs(t["te0"] / t["te1"] - 1.0)
    hm = t["hm0"]

    def skill(sel, a, b):
        sa = np.bincount(inv[sel], se[a][sel], ns)
        sb = np.bincount(inv[sel], se[b][sel], ns)
        pt = 1.0 - sa.sum() / sb.sum()
        bs = 1.0 - (W @ sa) / (W @ sb)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]"

    print(f"\n==== {tg} pop {POP} frac {FRAC}: {len(resid)} rows, {ns} discovery shots; Thomson at "
          f"both frames on {int(has_ts.sum())} rows; edge distance on {int(np.isfinite(d).sum())}")
    cells = [("all", np.ones(len(resid), bool)), ("Thomson available", has_ts)]
    cells += [(lab, has_ts & (d >= a) & (d < b)) for a, b, lab in EDGE_BANDS]
    top = has_ts & (d >= 0.015) & (d < 0.05)
    cells += [("pedestal top, H-mode rating >= 0.5", top & (hm >= 0.5)),
              ("pedestal top, H-mode rating < 0.5", top & (hm < 0.5)),
              ("pedestal top, T_e changed >= 15 %", top & (dte >= 0.15)),
              ("pedestal top, T_e changed < 15 %", top & (dte < 0.15)),
              ("pedestal top, core system ayc", top & (t["ts_sys"] == 2)),
              ("pedestal top, core system atm", top & (t["ts_sys"] == 1))]
    print("| cell | rows | shots | rmse pers | noise share | "
          + " | ".join(f"{a} vs {b}" for a, b in PAIRS) + " |")
    print("|" + "---|" * (5 + len(PAIRS)))
    for name, sel in cells:
        if sel.sum() < 300:
            print(f"| {name} | {int(sel.sum())} | - | - | - |" + " - |" * len(PAIRS))
            continue
        mse = se["persistence"][sel].mean()
        print(f"| {name} | {int(sel.sum())} | {len(np.unique(inv[sel]))} | {np.sqrt(mse):.1f} | "
              f"{1.0 - excess[sel].mean() / mse:.2f} | "
              + " | ".join(skill(sel, a, b) for a, b in PAIRS) + " |")
    if top.sum() >= 300:
        print("median absolute error, pedestal top: " + ", ".join(
            f"{k} {np.median(np.sqrt(v[top])):.2f}" for k, v in se.items()))


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
