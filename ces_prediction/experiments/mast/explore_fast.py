"""MAST exploration (discovery shots only): what do beam power and mid-plane D-alpha add to the
charge-exchange past when a hidden 5 ms frame is estimated, by radius and by D-alpha event class.

Protocol note: ces_prediction/experiments/MAST_FAST_EXPLORATION.md.  Only the odd-numbered shots of
coupling_shots.txt are read (the discovery half of the coupling study).  The even-numbered shots
are not opened by this script; a frozen model and its statements are registered before they are.

CPU only.  Run from the repo root:
    py ces_prediction/experiments/mast/explore_fast.py build     # row table -> data/mast/.explore_rows_{T_i,V_i}.npz
    py ces_prediction/experiments/mast/explore_fast.py fit       # shot-level 5-fold models, tables
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

OUT = rg.ROOT / "data" / "mast"
POP = "G"
BEAM_LAGS = 13                      # beam increments at frame lags 0 .. 12 (0 .. 60 ms)
RHO_EDGES = (0.0, 0.7, 0.8, 0.9, 1.0)
RHO_NAMES = ("0-0.7", "0.7-0.8", "0.8-0.9", "0.9-1")
SEED = 20261117
N_BOOT = 400


def frame_max(t_ms, v, t_a, n_lo, n_hi):
    """Maximum of the driver over [t_a + 5n - 2.5, t_a + 5n + 2.5) ms for n = n_lo .. n_hi."""
    n = np.arange(n_lo, n_hi + 1)
    out = np.full(len(n), np.nan)
    ok = np.isfinite(t_ms) & np.isfinite(v)
    if ok.sum() < 2:
        return out
    t, x = t_ms[ok], v[ok].astype(float)
    lo = np.searchsorted(t, t_a + rg.STEP_MS * n - 2.5, side="left")
    hi = np.searchsorted(t, t_a + rg.STEP_MS * n + 2.5, side="left")
    for i in range(len(n)):
        if hi[i] > lo[i]:
            out[i] = x[lo[i]:hi[i]].max()
    return out


def shot_rows(shot, geo, params, target):
    d = rg.load_shot(shot)
    if d is None or geo is None:
        return None
    tt = d["time_ms"]
    dd = np.diff(tt[np.isfinite(tt)])
    if not (len(dd) and abs(float(np.median(dd)) - rg.STEP_MS) <= 0.1):
        return None
    if d["R"].shape[0] != d["T"].shape[1]:
        return None
    p = rg.prepare(d, target, rg.POPS[POP])
    if p["status"] != "ok" or p["n_rows"] == 0 or p["n_bad_dt"] > 0 or not p["t_finite"]:
        return None
    ax, ro = geo
    a, _ = rc.block_of(d, target, rg.POPS[POP])
    t_a = float(d["time_ms"][a])
    L, G, Y, S2, hid, rows = p["L"], p["G"], p["Y0"], p["S2"], p["hid"], p["rows"]
    rho = (p["R"] - ax) / (ro - ax)
    nh, nc = rows.shape
    sf, _ = rg.arm_outputs(p, "slope_filter", params)
    rm, _ = rg.arm_outputs(p, "running_mean", params)

    def at(off, arr=Y, need_good=True):
        idx = hid + off
        ok = (idx >= 0) & (idx < L)
        v = np.full((nh, nc), np.nan)
        v[ok] = arr[idx[ok]]
        if need_good:
            g = np.zeros((nh, nc), bool)
            g[ok] = G[idx[ok]]
            v[~g] = np.nan
        return v

    y, y1, y3, y5, y7 = at(0), at(-1), at(-3), at(-5), at(-7)
    s0, s1, s3 = at(0, S2), at(-1, S2), at(-3, S2)
    # neighbours: mean of the good chords within two columns, at the last visible frame
    Yn = np.where(G, Y, 0.0)
    num = np.zeros_like(Yn)
    den = np.zeros_like(Yn)
    for k in (-2, -1, 1, 2):
        sh = np.roll(Yn, k, axis=1)
        gs = np.roll(G, k, axis=1).astype(float)
        if k > 0:
            sh[:, :k] = 0.0
            gs[:, :k] = 0.0
        else:
            sh[:, k:] = 0.0
            gs[:, k:] = 0.0
        num += sh
        den += gs
    with np.errstate(all="ignore"):
        nbm = np.where(den > 0, num / den, np.nan)
    nb1, nb3 = at(-1, nbm, False), at(-3, nbm, False)

    drv = rc.load_drivers(shot)
    n_lo, n_hi = -(BEAM_LAGS + 3), L
    fast = {}
    if "dalpha" in drv:
        tdv, dv = drv["dalpha"]
        m = rc.frame_means(tdv, dv, t_a, n_lo, n_hi)
        mx = frame_max(tdv, dv, t_a, n_lo, n_hi)
        blk = m[-n_lo:-n_lo + L]
        sc = np.nanmedian(np.abs(blk)) if np.isfinite(blk).any() else np.nan
        if np.isfinite(sc) and sc > 0:
            m, mx = m / sc, mx / sc
            j = hid - n_lo
            fast["da0"], fast["da1"], fast["da2"], fast["da3"] = m[j], m[j - 1], m[j - 2], m[j - 3]
            fast["dmx0"], fast["dmx1"], fast["dmx2"] = mx[j], mx[j - 1], mx[j - 2]
    if "ss" in drv:
        tb, bv = drv["ss"]
        b = rc.frame_means(tb, bv, t_a, n_lo, n_hi)
        top = np.nanmax(b) if np.isfinite(b).any() else np.nan
        if np.isfinite(top) and top > 0:
            b = b / top
            j = hid - n_lo
            fast["bm0"] = b[j]
            for k in range(BEAM_LAGS):
                fast[f"bd{k}"] = b[j - k] - b[j - k - 1]
    names = (["da0", "da1", "da2", "da3", "dmx0", "dmx1", "dmx2", "bm0"]
             + [f"bd{k}" for k in range(BEAM_LAGS)])
    sel = rows
    n = int(sel.sum())
    hh = np.broadcast_to(np.arange(nh)[:, None], rows.shape)[sel]
    cc = np.broadcast_to(np.arange(nc)[None, :], rows.shape)[sel]
    out = {"shot": np.full(n, shot, np.int32), "frame": hid[hh].astype(np.int16),
           "rho": rho[cc].astype(np.float32)}
    for k, v in (("y", y), ("y1", y1), ("y3", y3), ("y5", y5), ("y7", y7), ("s0", s0), ("s1", s1),
                 ("s3", s3), ("sf", sf), ("rm", rm), ("nb1", nb1), ("nb3", nb3)):
        out[k] = v[sel].astype(np.float32)
    for k in names:
        out[k] = (fast[k][hh] if k in fast else np.full(n, np.nan)).astype(np.float32)
    return out


def build():
    t0 = time.time()
    listed = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    disc = [s for s in listed if s % 2 == 1]
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

    for tg in rg.TARGETS:
        params = re_.params_from(add1, tg)
        parts, used = [], 0
        for i, s in enumerate(disc):
            r = shot_rows(s, geo_of(s), params, tg)
            if r is not None:
                parts.append(r)
                used += 1
            if (i + 1) % 200 == 0:
                print(f"[{time.time() - t0:5.0f}s] {tg} {i + 1}/{len(disc)} shots, {used} with rows",
                      flush=True)
        tab = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(OUT / f".explore_rows_{tg}.npz", **tab)
        print(f"{tg}: {used} shots, {len(tab['y'])} rows", flush=True)


# --------------------------------------------------------------------------- models

def features(t, group):
    y1 = t["y1"]
    f = {"y1": y1, "d13": y1 - t["y3"], "d35": t["y3"] - t["y5"], "d57": t["y5"] - t["y7"],
         "sf": t["sf"] - y1, "rm": t["rm"] - y1, "s1": t["s1"], "s3": t["s3"], "rho": t["rho"]}
    if "nb" in group:
        f["nb1"] = t["nb1"] - y1
        f["nb3"] = t["nb3"] - t["y3"]
    if "fast" in group:
        f["da01"] = t["da0"] - t["da1"]
        f["da12"] = t["da1"] - t["da2"]
        f["da23"] = t["da2"] - t["da3"]
        f["dmx0"] = t["dmx0"] - t["da1"]
        f["dmx1"] = t["dmx1"] - t["da2"]
        f["da0"] = t["da0"]
        f["bm0"] = t["bm0"]
        for k in range(BEAM_LAGS):
            f[f"bd{k}"] = t[f"bd{k}"]
    names = list(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names


def fit():
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    for tg in rg.TARGETS:
        z = np.load(OUT / f".explore_rows_{tg}.npz")
        t = {k: z[k] for k in z.files}
        n = len(t["y"])
        shots = np.unique(t["shot"])
        fold_of = dict(zip(shots.tolist(), rng.permutation(len(shots)) % 5))
        fold = np.array([fold_of[s] for s in t["shot"].tolist()])
        resid = (t["y"] - t["y1"]).astype(np.float64)
        scale = float(np.median(np.sqrt(t["s0"])))
        preds = {"persistence": np.zeros(n), "slope_filter": (t["sf"] - t["y1"]).astype(np.float64),
                 "running_mean": (t["rm"] - t["y1"]).astype(np.float64)}
        # clip the regression target at 20 median reported errors: fit failures stored as values
        # would otherwise set the trees; scoring uses the unclipped value
        target = np.clip(resid, -20 * scale, 20 * scale)
        for name, group in (("gb_ces", ()), ("gb_ces_nb", ("nb",)), ("gb_ces_fast", ("fast",)),
                            ("gb_all", ("nb", "fast"))):
            X, _ = features(t, group)
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
        np.savez(OUT / f".explore_preds_{tg}.npz", fold=fold, **preds)
        report(tg, t, resid, preds, rng)


def report(tg, t, resid, preds, rng):
    shots, inv = np.unique(t["shot"], return_inverse=True)
    ns = len(shots)
    W = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=N_BOOT).astype(float)
    se = {k: (resid - v) ** 2 for k, v in preds.items()}
    ev = np.abs(t["da0"] - t["da1"])
    fin = np.isfinite(ev)
    thr = np.quantile(ev[fin], 0.9)
    rb = np.searchsorted(np.asarray(RHO_EDGES), t["rho"], side="right") - 1
    rb[~((t["rho"] > 0) & (t["rho"] < 1))] = -1
    excess = resid ** 2 - (t["s0"] + t["s1"])

    def skill(sel, a, b):
        sa = np.bincount(inv[sel], se[a][sel], ns)
        sb = np.bincount(inv[sel], se[b][sel], ns)
        pt = 1.0 - sa.sum() / sb.sum()
        bs = 1.0 - (W @ sa) / (W @ sb)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]"

    print(f"\n==== {tg}: {len(resid)} rows, {ns} discovery shots; D-alpha step threshold "
          f"(90th pct of |frame-mean change| / block median) = {thr:.3f}")
    cells = [("all", np.ones(len(resid), bool))]
    cells += [(f"rho {RHO_NAMES[k]}", rb == k) for k in range(4)]
    cells += [("D-alpha step rows", fin & (ev >= thr)), ("quiet rows", fin & (ev < thr)),
              ("edge 0.8-1 step", fin & (ev >= thr) & (rb >= 2)),
              ("edge 0.8-1 quiet", fin & (ev < thr) & (rb >= 2))]
    pairs = [("slope_filter", "persistence"), ("gb_ces", "persistence"), ("gb_ces", "slope_filter"),
             ("gb_ces_fast", "gb_ces"), ("gb_ces_nb", "gb_ces"), ("gb_all", "gb_ces_nb"),
             ("gb_all", "persistence")]
    print("| cell | rows | rmse pers | noise share of pers MSE | "
          + " | ".join(f"{a} vs {b}" for a, b in pairs) + " |")
    print("|" + "---|" * (4 + len(pairs)))
    for name, sel in cells:
        if sel.sum() < 500:
            continue
        mse = se["persistence"][sel].mean()
        share = 1.0 - excess[sel].mean() / mse
        print(f"| {name} | {int(sel.sum())} | {np.sqrt(mse):.1f} | {share:.2f} | "
              + " | ".join(skill(sel, a, b) for a, b in pairs) + " |")
    print("median absolute error: " + ", ".join(
        f"{k} {np.median(np.sqrt(v)):.2f}" for k, v in se.items()))


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
