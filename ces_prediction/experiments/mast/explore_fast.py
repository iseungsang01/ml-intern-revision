"""MAST exploration (discovery shots only): what do beam power and mid-plane D-alpha add to the
charge-exchange past when a hidden 5 ms frame is estimated, by radius and by D-alpha event class.

Protocol note: ces_prediction/experiments/MAST_FAST_EXPLORATION.md.  Only the odd-numbered shots of
coupling_shots.txt are read (the discovery half of the coupling study).  The even-numbered shots
are not opened by this script; a frozen model and its statements are registered before they are.

CPU only.  Run from the repo root:
    py ces_prediction/experiments/mast/explore_fast.py build [POP FRAC]   # row tables
    py ces_prediction/experiments/mast/explore_fast.py fit SET [POP FRAC] # shot-level 5-fold models
POP is G (relative error < 0.1) or W (< 0.5); FRAC is the share of block frames in which a chord
has to be good to be scored (0.9 = the gate rule).  SET is "groups" (gradient-boosted trees, input
groups) or "small" (linear and one-hidden-layer forms against the trees).
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
BEAM_LAGS = 13                      # beam increments at frame lags 0 .. 12 (0 .. 60 ms)
RHO_EDGES = (0.0, 0.7, 0.8, 0.9, 0.95, 1.0)
RHO_NAMES = ("0-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1")
SEED = 20261117
N_BOOT = 400
FAST_NAMES = (["da0", "da1", "da2", "da3", "dmx0", "dmx1", "dmx2", "bm0"]
              + [f"bd{k}" for k in range(BEAM_LAGS)])


def tag_of(pop, frac):
    return "" if (pop, frac) == ("G", 0.9) else f"_{pop}{int(round(frac * 100))}"


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


def prepare(d, target, rel, frac):
    """rg.prepare with the scored-chord share as an argument (0.9 reproduces the gate rule)."""
    if frac == 0.9:
        return rg.prepare(d, target, rel)
    good = rg.good_masks(d, rel)[target]
    y = d["T"] if target == "T_i" else d["V"]
    e = d["Te"] if target == "T_i" else d["Ve"]
    frames = np.nonzero(good.sum(1) >= 10)[0]
    if len(frames) == 0:
        return {"status": "no_block"}
    a, b = int(frames[0]), int(frames[-1])
    L = b - a + 1
    if L < 20:
        return {"status": "no_block"}
    gb = good[a:b + 1]
    chords = np.nonzero(gb.mean(0) >= frac)[0]
    if len(chords) == 0:
        return {"status": "no_scored_chord"}
    G = gb[:, chords]
    Y0 = np.where(G, y[a:b + 1][:, chords], 0.0)
    S2 = np.where(G, e[a:b + 1][:, chords] ** 2, 0.0)
    t_stored = d["time_ms"][a:b + 1]
    hid = rg.hidden_positions(L)
    rows = G[hid] & G[hid - 1] & G[hid - 3] & G[hid + 1]
    dts = np.diff(t_stored)
    return {"status": "ok", "L": L, "t": rg.STEP_MS * np.arange(L), "G": G, "Y0": Y0, "S2": S2,
            "hid": hid, "rows": rows, "R": d["R"][chords], "n_rows": int(rows.sum()),
            "n_bad_dt": int((~(np.abs(dts - rg.STEP_MS) <= 0.1)).sum()),
            "t_finite": bool(np.isfinite(t_stored).all())}


def shot_rows(shot, geo, params, target, pop, frac):
    d = rg.load_shot(shot)
    if d is None or geo is None:
        return None
    tt = d["time_ms"]
    dd = np.diff(tt[np.isfinite(tt)])
    if not (len(dd) and abs(float(np.median(dd)) - rg.STEP_MS) <= 0.1):
        return None
    if d["R"].shape[0] != d["T"].shape[1]:
        return None
    p = prepare(d, target, rg.POPS[pop], frac)
    if p["status"] != "ok" or p["n_rows"] == 0 or p["n_bad_dt"] > 0 or not p["t_finite"]:
        return None
    ax, ro = geo
    a, _ = rc.block_of(d, target, rg.POPS[pop])
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
    yn = at(1)                      # next visible frame: offline interpolation reference only
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
    sel = rows
    n = int(sel.sum())
    hh = np.broadcast_to(np.arange(nh)[:, None], rows.shape)[sel]
    cc = np.broadcast_to(np.arange(nc)[None, :], rows.shape)[sel]
    out = {"shot": np.full(n, shot, np.int32), "frame": hid[hh].astype(np.int16),
           "rho": rho[cc].astype(np.float32), "R": p["R"][cc].astype(np.float32),
           "t0": (t_a + rg.STEP_MS * hid[hh]).astype(np.float32)}   # hidden-frame time, ms
    for k, v in (("y", y), ("y1", y1), ("y3", y3), ("y5", y5), ("y7", y7), ("yn", yn), ("s0", s0),
                 ("s1", s1),
                 ("s3", s3), ("sf", sf), ("rm", rm), ("nb1", nb1), ("nb3", nb3)):
        out[k] = v[sel].astype(np.float32)
    for k in FAST_NAMES:
        out[k] = (fast[k][hh] if k in fast else np.full(n, np.nan)).astype(np.float32)
    return out


def build(pop="G", frac=0.9):
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
        parts = []
        for i, s in enumerate(disc):
            r = shot_rows(s, geo_of(s), params, tg, pop, frac)
            if r is not None:
                parts.append(r)
            if (i + 1) % 400 == 0:
                print(f"[{time.time() - t0:5.0f}s] {tg} {i + 1}/{len(disc)} shots, "
                      f"{len(parts)} with rows", flush=True)
        tab = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(OUT / f".explore_rows_{tg}{tag_of(pop, frac)}.npz", **tab)
        print(f"{tg} {pop} {frac}: {len(parts)} shots, {len(tab['y'])} rows", flush=True)


# --------------------------------------------------------------------------- inputs

def tree_features(t, group):
    y1 = t["y1"]
    f = {"y1": y1, "d13": y1 - t["y3"], "d35": t["y3"] - t["y5"], "d57": t["y5"] - t["y7"],
         "sf": t["sf"] - y1, "rm": t["rm"] - y1, "s1": t["s1"], "s3": t["s3"], "rho": t["rho"]}
    if "nb" in group:
        f["nb1"] = t["nb1"] - y1
        f["nb3"] = t["nb3"] - t["y3"]
    if "da" in group:
        f["da01"] = t["da0"] - t["da1"]
        f["da12"] = t["da1"] - t["da2"]
        f["da23"] = t["da2"] - t["da3"]
        f["dmx0"] = t["dmx0"] - t["da1"]
        f["dmx1"] = t["dmx1"] - t["da2"]
        f["da0"] = t["da0"]
    if "bm" in group:
        f["bm0"] = t["bm0"]
        for k in range(BEAM_LAGS):
            f[f"bd{k}"] = t[f"bd{k}"]
    return np.column_stack(list(f.values())).astype(np.float32)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def small_features(t, scale, with_da):
    """Inputs of the small forms, in units of the median reported error; missing values -> 0 with
    a flag.  With D-alpha: signed-log frame-mean and frame-maximum changes, alone and multiplied
    by the last visible value and by the normalized radius."""
    y1 = t["y1"].astype(np.float64)
    z = lambda v: np.nan_to_num(v.astype(np.float64) / scale)  # noqa: E731
    lv = y1 / scale
    cols = [z(t["sf"] - t["y1"]), z(t["rm"] - t["y1"]), z(t["y1"] - t["y3"]),
            z(t["y3"] - t["y5"]), np.isfinite(t["y5"]).astype(float), lv,
            np.nan_to_num(t["rho"].astype(np.float64)),
            np.sqrt(t["s1"].astype(np.float64)) / scale]
    if with_da:
        has = np.isfinite(t["da0"]) & np.isfinite(t["da2"]) & np.isfinite(t["dmx0"])
        u01 = np.where(has, slog(t["da0"] - t["da1"]), 0.0)
        u12 = np.where(has, slog(t["da1"] - t["da2"]), 0.0)
        um0 = np.where(has, slog(t["dmx0"] - t["da1"]), 0.0)
        um1 = np.where(has, slog(t["dmx1"] - t["da2"]), 0.0)
        lvl = np.where(has, slog(t["da0"]), 0.0)
        rho = np.nan_to_num(t["rho"].astype(np.float64))
        cols += [has.astype(float), u01, u12, um0, um1, lvl, lv * u01, lv * um0, lv * u12,
                 rho * u01, rho * um0]
    return np.column_stack(cols)


# --------------------------------------------------------------------------- models

def fit(which="groups", pop="G", frac=0.9):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    tag = tag_of(pop, frac)
    for tg in rg.TARGETS:
        z = np.load(OUT / f".explore_rows_{tg}{tag}.npz")
        t = {k: z[k] for k in z.files}
        n = len(t["y"])
        shots = np.unique(t["shot"])
        fold_of = dict(zip(shots.tolist(), rng.permutation(len(shots)) % 5))
        fold = np.array([fold_of[s] for s in t["shot"].tolist()])
        resid = (t["y"] - t["y1"]).astype(np.float64)
        scale = float(np.median(np.sqrt(t["s0"])))
        preds = {"persistence": np.zeros(n), "slope_filter": (t["sf"] - t["y1"]).astype(np.float64)}
        # regression target clipped at 20 median reported errors (fit failures stored as values
        # would otherwise set the fit); scoring uses the unclipped value
        target = np.clip(resid, -20 * scale, 20 * scale)
        n_par = {}

        def gb(X):
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                 max_leaf_nodes=31, min_samples_leaf=200,
                                                 l2_regularization=1.0, random_state=SEED), X, 1.0

        def lin(X):
            n_par[X.shape[1]] = X.shape[1] + 1
            return Ridge(alpha=10.0), X, scale

        def mlp(X):
            return MLPRegressor(hidden_layer_sizes=(16,), activation="tanh", batch_size=4096,
                                learning_rate_init=2e-3, max_iter=25, early_stopping=True,
                                n_iter_no_change=4, alpha=1e-4, random_state=SEED), X, scale

        if which == "groups":
            models = [("gb_ces", gb, tree_features(t, ())), ("gb_nb", gb, tree_features(t, ("nb",))),
                      ("gb_da", gb, tree_features(t, ("da",))), ("gb_bm", gb, tree_features(t, ("bm",))),
                      ("gb_da_bm", gb, tree_features(t, ("da", "bm"))),
                      ("gb_all", gb, tree_features(t, ("nb", "da", "bm")))]
            pairs = [("gb_ces", "persistence"), ("gb_da", "gb_ces"), ("gb_bm", "gb_ces"),
                     ("gb_da_bm", "gb_bm"), ("gb_nb", "gb_ces"), ("gb_all", "persistence")]
        else:
            Xc, Xd = small_features(t, scale, False), small_features(t, scale, True)
            models = [("gb_ces", gb, tree_features(t, ())), ("gb_da", gb, tree_features(t, ("da",))),
                      ("lin_ces", lin, Xc), ("lin_da", lin, Xd), ("mlp_ces", mlp, Xc),
                      ("mlp_da", mlp, Xd)]
            pairs = [("gb_da", "gb_ces"), ("lin_da", "lin_ces"), ("mlp_da", "mlp_ces"),
                     ("gb_da", "persistence"), ("lin_da", "persistence"), ("mlp_da", "persistence"),
                     ("lin_da", "slope_filter"), ("mlp_da", "gb_da")]
            print(f"{tg}: inputs lin_ces {Xc.shape[1]}, lin_da {Xd.shape[1]} (+1 intercept each); "
                  f"mlp parameters {Xc.shape[1] * 16 + 16 + 17} / {Xd.shape[1] * 16 + 16 + 17}")
        for name, make, X in models:
            pr = np.zeros(n)
            for k in range(5):
                tr, te = fold != k, fold == k
                m, XX, div = make(X)
                if make is not gb:
                    sc = StandardScaler().fit(XX[tr])
                    m.fit(sc.transform(XX[tr]), target[tr] / div)
                    pr[te] = m.predict(sc.transform(XX[te])) * div
                else:
                    m.fit(XX[tr], target[tr])
                    pr[te] = m.predict(XX[te])
            preds[name] = pr
            print(f"  fitted {tg} {name}", flush=True)
        np.savez(OUT / f".explore_preds_{tg}{tag}_{which}.npz", fold=fold, **preds)
        report(f"{tg} pop {pop} frac {frac}", t, resid, preds, pairs, rng)


def report(title, t, resid, preds, pairs, rng):
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
    bd = np.column_stack([t[f"bd{k}"] for k in range(BEAM_LAGS)])
    bfin = np.isfinite(bd).all(1)
    steady = bfin & (np.abs(bd).max(1) < 0.05) & (t["bm0"] > 0.5)

    def skill(sel, a, b):
        sa = np.bincount(inv[sel], se[a][sel], ns)
        sb = np.bincount(inv[sel], se[b][sel], ns)
        pt = 1.0 - sa.sum() / sb.sum()
        bs = 1.0 - (W @ sa) / (W @ sb)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]"

    print(f"\n==== {title}: {len(resid)} rows, {ns} discovery shots, {int(fin.sum())} rows with "
          f"D-alpha; step threshold (90th pct of |frame-mean change| / block median) = {thr:.3f}")
    cells = [("all", np.ones(len(resid), bool))]
    cells += [(f"rho {RHO_NAMES[k]}", rb == k) for k in range(len(RHO_NAMES))]
    cells += [("rho 0.8-1", rb >= 2),
              ("beam steady 60 ms", steady), ("beam steady, D-alpha step", steady & fin & (ev >= thr)),
              ("beam steady, quiet", steady & fin & (ev < thr)),
              ("beam changed in 60 ms", bfin & (np.abs(bd).max(1) >= 0.05)),
              ("D-alpha step rows", fin & (ev >= thr)), ("quiet rows", fin & (ev < thr)),
              ("rho 0.8-1 step", fin & (ev >= thr) & (rb >= 2)),
              ("rho 0.8-1 quiet", fin & (ev < thr) & (rb >= 2))]
    print("| cell | rows | shots | rmse pers | noise share of pers MSE | "
          + " | ".join(f"{a} vs {b}" for a, b in pairs) + " |")
    print("|" + "---|" * (5 + len(pairs)))
    for name, sel in cells:
        if sel.sum() < 500:
            continue
        mse = se["persistence"][sel].mean()
        share = 1.0 - excess[sel].mean() / mse
        print(f"| {name} | {int(sel.sum())} | {len(np.unique(inv[sel]))} | {np.sqrt(mse):.1f} | "
              f"{share:.2f} | " + " | ".join(skill(sel, a, b) for a, b in pairs) + " |")
    print("median absolute error: " + ", ".join(
        f"{k} {np.median(np.sqrt(v)):.2f}" for k, v in se.items()))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build(*((a[1], float(a[2])) if len(a) > 1 else ()))
    else:
        fit(a[1], *((a[2], float(a[3])) if len(a) > 2 else ()))
