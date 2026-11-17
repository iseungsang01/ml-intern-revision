"""B.40 (PREREGISTRATION_B40.md): a bracket for the white-noise variance of the CES T_i value.

y_t = s_t + e_t, e_t white of variance sigma^2; over triples of consecutive measured rows
    N_lo = -cov(d_t, d_{t+1})                 (<= sigma^2 when signal increments are not anti-correlated)
    N_hi = var(y_{t+1} - 2 y_t + y_{t-1}) / 6  (>= sigma^2)
in three versions: moments, moments with clipped differences, robust (1.4826 MAD, primary):
    N_hi = S^2(sd) / 6,  N_lo = [S^2(sd) - S^2(span)] / 4,  sd = y+ - 2y + y-,  span = y+ - y-.

M1  MAST (reported fit errors): the median squared error lies inside [N_lo, N_hi] by radius bin.
K1  KSTAR: N_hi / N_lo <= 1.30, upper end of its interval <= 1.45.
K2  KSTAR: N_lo in the lowest fifth of the BES level over N_lo in the highest fifth, interval above 1.

CPU only, no model, no training.  Run from the repo root:
    py ces_prediction/experiments/b40_noise/run_b40.py
Output: docs/paper/b40_noise.json / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b37_plan"))
sys.path.insert(0, str(HERE.parent / "mast"))
import run_b30 as R  # noqa: E402
from run_b37 import Ctx  # noqa: E402
import run_gate as rg  # noqa: E402
import run_coupling as rc  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b40_noise.json"
OUT_MD = R.REPO / "docs" / "paper" / "b40_noise.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B40.md"
SEED, N_BOOT = 20261118, 2000
K_HIST = 2000             # quantile bins of the resampling approximation of medians
MAD_C = 1.4826
MIN_TRIPLES_SHOT = 100
VERSIONS = ("robust", "moments", "clipped")
EST = ("hold", "gp", "hist80", "causal")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [float("nan")] * 2


# ------------------------------------------------------------------------------ robust scale
def s2_exact(x):
    med = np.median(x)
    return float((MAD_C * np.median(np.abs(x - med))) ** 2)


class Hist:
    """Per-discharge histograms on pooled-quantile edges; resampled medians and MADs by linear
    interpolation of the resampled CDF (point estimates are exact, see s2_exact)."""

    def __init__(self, x, f, n):
        e = np.unique(np.quantile(x, np.linspace(0, 1, K_HIST + 1)))
        self.e = e
        k = len(e) - 1
        j = np.clip(np.searchsorted(e, x, side="right") - 1, 0, k - 1)
        self.H = np.bincount(f * k + j, minlength=n * k).reshape(n, k).astype(float)

    def cdf_rows(self, W):
        Hb = W @ self.H
        C = np.concatenate([np.zeros((len(W), 1)), np.cumsum(Hb, 1)], 1)
        return C / C[:, -1:]

    def F(self, C, v):
        e = self.e
        vv = np.clip(v, e[0], e[-1])
        j = np.clip(np.searchsorted(e, vv, side="right"), 1, len(e) - 1)
        w = np.where(e[j] > e[j - 1], (vv - e[j - 1]) / np.maximum(e[j] - e[j - 1], 1e-300), 0.0)
        r = np.arange(len(C))
        return C[r, j - 1] + w * (C[r, j] - C[r, j - 1])

    def quantile(self, C, q):
        e = self.e
        j = np.clip((C < q).sum(1), 1, len(e) - 1)
        r = np.arange(len(C))
        lo, hi = C[r, j - 1], C[r, j]
        w = np.where(hi > lo, (q - lo) / np.maximum(hi - lo, 1e-300), 0.0)
        return e[j - 1] + w * (e[j] - e[j - 1])

    def s2(self, W):
        out = []
        for a in range(0, len(W), 250):
            C = self.cdf_rows(W[a:a + 250])
            med = self.quantile(C, 0.5)
            lo, hi = np.zeros(len(C)), np.full(len(C), self.e[-1] - self.e[0])
            for _ in range(60):
                r = 0.5 * (lo + hi)
                ok = self.F(C, med + r) - self.F(C, med - r) >= 0.5
                hi, lo = np.where(ok, r, hi), np.where(ok, lo, r)
            out.append((MAD_C * hi) ** 2)
        return np.concatenate(out)

    def median(self, W):
        return np.concatenate([self.quantile(self.cdf_rows(W[a:a + 250]), 0.5) for a in range(0, len(W), 250)])


# ------------------------------------------------------------------------------ the bracket
def bracket(sd, span, d1, d2, f, n, W):
    """Point and resampled N_lo, N_hi in three versions; f: discharge index 0..n-1, W: (B, n)."""
    out = {}
    hs, hp = Hist(sd, f, n), Hist(span, f, n)
    a, b = s2_exact(sd), s2_exact(span)
    bs, bp = hs.s2(W), hp.s2(W)
    out["robust"] = {"lo": ((a - b) / 4, (bs - bp) / 4), "hi": (a / 6, bs / 6)}
    out["robust"]["_hist_check"] = {"S2_sd_exact": a, "S2_sd_hist": float(hs.s2(np.ones((1, n)))[0]),
                                    "S2_span_exact": b, "S2_span_hist": float(hp.s2(np.ones((1, n)))[0])}
    cnt = np.bincount(f, minlength=n).astype(float)
    Nb = W @ cnt

    def mom(u, v, s):
        su, sv, suv, ss, ss2 = (np.bincount(f, z, n) for z in (u, v, u * v, s, s * s))
        lo = -(suv.sum() / cnt.sum() - su.sum() * sv.sum() / cnt.sum() ** 2)
        hi = (ss2.sum() / cnt.sum() - (ss.sum() / cnt.sum()) ** 2) / 6
        blo = -((W @ suv) / Nb - (W @ su) * (W @ sv) / Nb ** 2)
        bhi = ((W @ ss2) / Nb - ((W @ ss) / Nb) ** 2) / 6
        return {"lo": (float(lo), blo), "hi": (float(hi), bhi)}

    out["moments"] = mom(d1, d2, sd)
    dl, dh = np.quantile(np.concatenate([d1, d2]), [0.01, 0.99])
    sl, sh = np.quantile(sd, [0.01, 0.99])
    out["clipped"] = mom(np.clip(d1, dl, dh), np.clip(d2, dl, dh), np.clip(sd, sl, sh))
    return out


def summarize(br):
    res = {}
    for v in VERSIONS:
        lo, blo = br[v]["lo"]
        hi, bhi = br[v]["hi"]
        with np.errstate(all="ignore"):
            o = {"N_lo": [float(lo), *ci(blo)], "N_hi": [float(hi), *ci(bhi)],
                 "sigma_lo_eV": [float(np.sqrt(lo)) if lo > 0 else float("nan"), *ci(np.sqrt(np.where(blo > 0, blo, np.nan)))],
                 "sigma_hi_eV": [float(np.sqrt(hi)), *ci(np.sqrt(bhi))],
                 "ratio_hi_over_lo": [float(hi / lo), *ci(bhi / blo)],
                 "resamples_N_lo_nonpositive": int((blo <= 0).sum())}
        res[v] = o
    res["hist_check"] = br["robust"]["_hist_check"]
    return res


def boot_W(n, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)


# ------------------------------------------------------------------------------ KSTAR
def kstar_triples(grid, names, nb):
    """Triples of consecutive measured rows, both steps 10 ms (first and last row of a file dropped)."""
    cols = {k: [] for k in ("f", "d1", "d2", "bes", "lvl", "row")}
    for i, nm in enumerate(names):
        arr = grid[nm][1:-1]
        t, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
        b = arr[:, 3:3 + nb].astype(np.float64).mean(1)
        med = np.median(b)
        if len(y) < 3:
            continue
        dt = np.diff(t)
        m = (np.isfinite(y[:-2]) & np.isfinite(y[1:-1]) & np.isfinite(y[2:])
             & (np.abs(dt[:-1] - 0.010) < 1e-4) & (np.abs(dt[1:] - 0.010) < 1e-4))
        if not m.any():
            continue
        c = np.flatnonzero(m) + 1
        cols["f"].append(np.full(len(c), i))
        cols["d1"].append(y[c] - y[c - 1]); cols["d2"].append(y[c + 1] - y[c])
        cols["bes"].append(b[c] / med if med > 0 else np.full(len(c), np.nan))
        cols["lvl"].append((y[c - 1] + y[c] + y[c + 1]) / 3)
        cols["row"].append(c + 1)          # row index in the full grid array
    return {k: np.concatenate(v) for k, v in cols.items()}


def kstar_population(grid, names, nb, rng, label):
    tr = kstar_triples(grid, names, nb)
    uf, f = np.unique(tr["f"], return_inverse=True)
    n = len(uf)
    sd, span = tr["d2"] - tr["d1"], tr["d2"] + tr["d1"]
    W = boot_W(n, rng)
    log(f"KSTAR {label}: {len(f):,} triples, {n} discharges")

    b_all = bracket(sd, span, tr["d1"], tr["d2"], f, n, W)
    o = {"triples": int(len(f)), "discharges": n, "overall": summarize(b_all)}
    # identity on the real data: moment N_lo as -cov equals [var(sd) - var(span)] / 4
    o["identity_check"] = {"minus_cov": float(-np.cov(tr["d1"], tr["d2"], bias=True)[0, 1]),
                           "var_form": float((sd.var() - span.var()) / 4)}
    # fifths of the BES level of the centre row relative to the discharge median
    okb = np.isfinite(tr["bes"])
    q5 = np.quantile(tr["bes"][okb], [0.2, 0.4, 0.6, 0.8])
    fifth = np.where(okb, np.searchsorted(q5, tr["bes"], side="right"), -1)
    fifths, bl, lv5 = [], {}, {}
    for k in range(5):
        sel = fifth == k
        b = bracket(sd[sel], span[sel], tr["d1"][sel], tr["d2"][sel], f[sel], n, W)
        bl[k] = b
        lv5[k] = (float(np.median(tr["lvl"][sel])), Hist(tr["lvl"][sel], f[sel], n).median(W))
        fifths.append({"triples": int(sel.sum()), "bes_rel_range": [float(tr["bes"][sel].min()), float(tr["bes"][sel].max())],
                       **summarize(b)})
    o["bes_fifths"] = fifths
    o["bes_fifth_edges"] = q5.tolist()
    k2 = {}
    for v in VERSIONS:
        lo0, blo0 = bl[0][v]["lo"]
        lo4, blo4 = bl[4][v]["lo"]
        with np.errstate(all="ignore"):
            k2[v] = [float(lo0 / lo4), *ci(blo0 / blo4)]
    o["K2_ratio_lowest_over_highest"] = k2
    # after the fact: the same ratio with N_lo divided by the squared median T_i level of the fifth
    lo0, blo0 = bl[0]["robust"]["lo"]
    lo4, blo4 = bl[4]["robust"]["lo"]
    o["after_fact_K2_relative"] = {
        "median_level_eV_by_fifth": [[lv5[k][0], *ci(lv5[k][1])] for k in range(5)],
        "ratio_of_N_lo_over_level2": [float((lo0 / lv5[0][0] ** 2) / (lo4 / lv5[4][0] ** 2)),
                                      *ci((blo0 / lv5[0][1] ** 2) / (blo4 / lv5[4][1] ** 2))]}
    # thirds of the T_i level (mean of the three rows)
    q3 = np.quantile(tr["lvl"], [1 / 3, 2 / 3])
    third = np.searchsorted(q3, tr["lvl"], side="right")
    thirds = []
    for k in range(3):
        sel = third == k
        b = bracket(sd[sel], span[sel], tr["d1"][sel], tr["d2"][sel], f[sel], n, W)
        hl = Hist(tr["lvl"][sel], f[sel], n)
        lv, blv = float(np.median(tr["lvl"][sel])), hl.median(W)
        s = summarize(b)
        rel = {}
        for v in VERSIONS:
            rel[v] = {}
            for side in ("lo", "hi"):
                p, bb = b[v][side]
                with np.errstate(all="ignore"):
                    rel[v][f"sigma_{side}_over_level"] = [float(np.sqrt(p) / lv) if p > 0 else float("nan"),
                                                          *ci(np.sqrt(np.where(bb > 0, bb, np.nan)) / blv)]
        thirds.append({"triples": int(sel.sum()), "level_range_eV": [float(tr["lvl"][sel].min()), float(tr["lvl"][sel].max())],
                       "median_level_eV": [lv, *ci(blv)], **s, "sigma_over_level": rel})
    o["ti_thirds"] = thirds
    o["ti_third_edges_eV"] = q3.tolist()
    # per-discharge sigma (robust), discharges with at least 100 triples
    per = []
    for j in range(n):
        sel = f == j
        if sel.sum() < MIN_TRIPLES_SHOT:
            continue
        a, b = s2_exact(sd[sel]), s2_exact(span[sel])
        per.append(((a - b) / 4, a / 6))
    per = np.array(per)
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    o["per_discharge"] = {"discharges": int(len(per)), "quantiles": list(qs),
                          "N_lo_nonpositive": int((per[:, 0] <= 0).sum()),
                          "sigma_lo_eV_quantiles": [float(x) for x in np.quantile(np.sqrt(np.maximum(per[:, 0], 0)), qs)],
                          "sigma_hi_eV_quantiles": [float(x) for x in np.quantile(np.sqrt(per[:, 1]), qs)],
                          "ratio_hi_over_lo_quantiles": [float(x) for x in np.quantile(per[per[:, 0] > 0, 1] / per[per[:, 0] > 0, 0], qs)]}
    return o, tr


def row_tables(tr, rng):
    """Mean squared errors on the test rows of the frozen row tables (natural cadence, k = 1) in
    units of the bracket of the same discharges."""
    res = {}
    sd_all, span_all = tr["d2"] - tr["d1"], tr["d2"] + tr["d1"]
    for s in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        ok = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0)
        uf = np.unique(tab["file"][ok].astype(int))
        n = len(uf)
        pos = {int(i): j for j, i in enumerate(uf)}
        f = np.array([pos[int(i)] for i in tab["file"][ok]])
        m = np.isin(tr["f"], uf)
        ft = np.array([pos[int(i)] for i in tr["f"][m]])
        W = boot_W(n, rng)
        br = bracket(sd_all[m], span_all[m], tr["d1"][m], tr["d2"][m], ft, n, W)
        y = tab["y"][ok].astype(np.float64)
        cnt = np.bincount(f, minlength=n).astype(float)
        mse = {}
        for e in EST:
            se = np.bincount(f, (tab[e][ok].astype(np.float64) - y) ** 2, n)
            mse[e] = (se.sum() / cnt.sum(), (W @ se) / (W @ cnt))
        o = {"test_discharges": n, "test_rows": int(ok.sum()), "triples": int(m.sum()),
             "rmse_eV": {e: float(np.sqrt(mse[e][0])) for e in EST}, "bracket": summarize(br)}
        for v in VERSIONS:
            o[v] = {}
            for side in ("lo", "hi"):
                p, bb = br[v][side]
                with np.errstate(all="ignore"):
                    o[v][f"floors_{side}"] = {e: [float(mse[e][0] / p), *ci(mse[e][1] / bb)] for e in EST}
                    o[v][f"largest_skill_over_hold_{side}"] = [float(1 - p / mse["hold"][0]), *ci(1 - bb / mse["hold"][1])]
        res[str(s)] = o
        log(f"rows s{s}: robust floors causal lo {o['robust']['floors_lo']['causal'][0]:.2f} hi {o['robust']['floors_hi']['causal'][0]:.2f}")
    return res


# ------------------------------------------------------------------------------ MAST
def mast_triples(step=1, pops=tuple(rg.POPS)):
    shots = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit", "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")

    def geo_of(s):                                   # run_coupling.main / run_edge.main
        if s not in cat.index:
            return None
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        ro = float(cat.loc[s, "radii_router_radius_efit"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            return ax, ro
        ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
        return (ax, ro) if np.isfinite(ax) and np.isfinite(ro) and ro > ax else None

    acc = {pop: {k: [] for k in ("shot", "bin", "d1", "d2", "s2c", "s2all")} for pop in pops}
    status = {}
    for i, s in enumerate(shots):
        geo = geo_of(s)
        d = rg.load_shot(s) if geo is not None else None
        st = "no_geometry" if geo is None else ("no_file" if d is None else "ok")
        if d is not None:
            ax, ro = geo
            for pop in pops:
                rel = rg.POPS[pop]
                p = rg.prepare(d, "T_i", rel)
                if p["status"] != "ok":
                    continue
                a, _ = rc.block_of(d, "T_i", rel)
                L, G, Y, S2 = p["L"], p["G"], p["Y0"], p["S2"]
                tm = d["time_ms"][a:a + L]
                k = step
                if L < 2 * k + 1:
                    continue
                dt = np.diff(tm)
                step_ok = np.abs(dt - rg.STEP_MS) <= 0.1           # every frame step inside the triple
                run = np.convolve(step_ok.astype(int), np.ones(k, int), "valid") == k   # run[j]: steps j..j+k-1
                dtok = run[:-k] & run[k:]
                cb = rc.rho_bin((p["R"] - ax) / (ro - ax))
                ok = G[:-2 * k] & G[k:L - k] & G[2 * k:] & dtok[:, None] & (cb >= 0)[None, :]
                if not ok.any():
                    continue
                tt, cc = np.nonzero(ok)
                c = tt + k
                acc[pop]["shot"].append(np.full(len(c), s))
                acc[pop]["bin"].append(cb[cc])
                acc[pop]["d1"].append(Y[c, cc] - Y[c - k, cc]); acc[pop]["d2"].append(Y[c + k, cc] - Y[c, cc])
                acc[pop]["s2c"].append(S2[c, cc])
                acc[pop]["s2all"].append((S2[c - k, cc] + S2[c, cc] + S2[c + k, cc]) / 3)
        status[st] = status.get(st, 0) + 1
        if (i + 1) % 500 == 0:
            log(f"MAST {i + 1}/{len(shots)} shots")
    out = {pop: {k: np.concatenate(v) for k, v in acc[pop].items()} for pop in acc}
    return out, status, len(shots)


def mast_population(tr, rng):
    us, f = np.unique(tr["shot"], return_inverse=True)
    n = len(us)
    W = boot_W(n, rng)
    sd, span = tr["d2"] - tr["d1"], tr["d2"] + tr["d1"]
    cnt = np.bincount(f, minlength=n).astype(float)
    res = {"shots": n, "triples": int(len(f)), "bins": {}}
    names = list(rc.BINS[:-1]) + ["all"]
    for k, nm in enumerate(names):
        sel = np.ones(len(f), bool) if nm == "all" else tr["bin"] == k
        if sel.sum() < 50:
            res["bins"][nm] = None
            continue
        br = bracket(sd[sel], span[sel], tr["d1"][sel], tr["d2"][sel], f[sel], n, W)
        href = Hist(tr["s2c"][sel], f[sel], n)
        ref, bref = float(np.median(tr["s2c"][sel])), href.median(W)
        sm = np.bincount(f[sel], tr["s2c"][sel], n)
        cs = np.bincount(f[sel], minlength=n).astype(float)
        mref, bmref = float(sm.sum() / cs.sum()), (W @ sm) / (W @ cs)
        o = {"triples": int(sel.sum()), "shots": int((cs > 0).sum()), **summarize(br),
             "ref_median_s2": [ref, *ci(bref)], "ref_mean_s2": [mref, *ci(bmref)],
             "ref_median_s2_three_rows": float(np.median(tr["s2all"][sel])),
             "ref_median_sigma_eV": float(np.sqrt(ref))}
        for v in VERSIONS:
            with np.errstate(all="ignore"):
                o[v]["lo_over_ref"] = [br[v]["lo"][0] / ref, *ci(br[v]["lo"][1] / bref)]
                o[v]["hi_over_ref"] = [br[v]["hi"][0] / ref, *ci(br[v]["hi"][1] / bref)]
                o[v]["lo_over_mean_ref"] = [br[v]["lo"][0] / mref, *ci(br[v]["lo"][1] / bmref)]
                o[v]["hi_over_mean_ref"] = [br[v]["hi"][0] / mref, *ci(br[v]["hi"][1] / bmref)]
                o[v]["ref_inside"] = bool(br[v]["lo"][0] <= ref <= br[v]["hi"][0])
        res["bins"][nm] = o
        log(f"MAST bin {nm}: {o['triples']:,} triples; robust lo/ref {o['robust']['lo_over_ref'][0]:.3f} "
            f"hi/ref {o['robust']['hi_over_ref'][0]:.3f}")
    del cnt
    return res


def m1(res, v="robust"):
    b = res["bins"]
    inside = [nm for nm in list(rc.BINS[:-1]) if b.get(nm) and b[nm][v]["ref_inside"]]
    a = b["all"][v]
    pooled = bool(a["lo_over_ref"][2] < 1 and a["hi_over_ref"][1] > 1)
    return {"bins_inside": inside, "n_inside": len(inside), "pooled_ok": pooled, "holds": bool(len(inside) >= 4 and pooled)}


# ------------------------------------------------------------------------------ expectations
def expectations(res):
    k = res["kstar"]["cut"]
    r = k["overall"]["robust"]
    E = []
    s = r["sigma_lo_eV"][0]
    E.append(("KSTAR robust sqrt(N_lo) 40 to 60 eV", f"{s:.1f} eV", bool(40 <= s <= 60)))
    q = r["ratio_hi_over_lo"][0]
    E.append(("N_hi / N_lo 1.05 to 1.25 (K1 holds)", f"{q:.3f}, K1 {'holds' if res['K1']['holds'] else 'does not hold'}",
              bool(1.05 <= q <= 1.25 and res["K1"]["holds"])))
    k2 = k["K2_ratio_lowest_over_highest"]["robust"]
    E.append(("K2 ratio 1.3 to 3 (holds)", f"{k2[0]:.3f}, K2 {'holds' if res['K2']['holds'] else 'does not hold'}",
              bool(1.3 <= k2[0] <= 3 and res["K2"]["holds"])))
    rel = [t["sigma_over_level"]["robust"]["sigma_lo_over_level"][0] for t in k["ti_thirds"]]
    E.append(("sigma / level 0.06 to 0.15 (robust sqrt(N_lo), each third)", ", ".join(f"{x:.3f}" for x in rel),
              bool(all(0.06 <= x <= 0.15 for x in rel))))
    E.append(("sigma / level largest in the lowest third", ", ".join(f"{x:.3f}" for x in rel), bool(np.nanargmax(rel) == 0)))
    fl = [res["rows"][str(s)]["robust"]["floors_lo"]["causal"][0] for s in R.SEEDS]
    E.append(("causal variant at 2.5 to 5 floors of the robust bound (MSE / N_lo, each split)", ", ".join(f"{x:.2f}" for x in fl),
              bool(all(2.5 <= x <= 5 for x in fl))))
    E.append(("MAST: M1 holds in at least four bins", f"{res['M1']['n_inside']} bins, M1 {'holds' if res['M1']['holds'] else 'does not hold'}",
              bool(res["M1"]["holds"])))
    inner = res["mast"]["G"]["bins"]["0-0.3"]
    v = inner["robust"]["N_lo"][0] if inner else float("nan")
    E.append(("MAST: N_lo negative in the innermost bin (0-0.3)", f"{v:.1f} eV^2", bool(v < 0)))
    return [{"expectation": a, "observed": b, "met": c} for a, b, c in E]


# ------------------------------------------------------------------------------ render
def f3(x, nd=3):
    return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"


def f1(x):
    return f3(x, 1)


def bracket_rows(label, o):
    r = []
    for v in VERSIONS:
        b = o[v]
        r.append(f"| {label} | {v} | {f1(b['N_lo'])} | {f1(b['N_hi'])} | {f1(b['sigma_lo_eV'])} | {f1(b['sigma_hi_eV'])} | "
                 f"{f3(b['ratio_hi_over_lo'])} |")
    return r


HDR = ["| set | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |",
       "|---|---|---|---|---|---|---|"]


def render(res):
    k = res["kstar"]["cut"]
    ki = res["kstar"]["incl"]
    hs = lambda h: "holds" if h else "does not hold"                         # noqa: E731
    L = [f"# B.40 — a bracket for the white-noise variance of the CES T_i value (registered: {REG})", "",
         "CPU, no model. y_t = s_t + e_t; over triples of consecutive measured rows (both steps one frame: 10 ms on KSTAR, "
         "5 ms on MAST), N_lo = -cov(d_t, d_t+1), N_hi = var(second difference) / 6, in three versions: robust (primary; "
         "S = 1.4826 MAD, N_hi = S^2(sd)/6, N_lo = [S^2(sd) - S^2(span)]/4), moments, and moments with the differences "
         "clipped at their 1st and 99th percentiles. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST); "
         "point values are exact, resampled medians use per-discharge histograms on 2,000 pooled-quantile bins.", "",
         "## Registered statements", "",
         f"- **M1** (MAST, population G: the median squared reported error lies between N_lo and N_hi in at least four of five "
         f"radius bins, and pooled N_lo / reference has an interval below 1 and N_hi / reference above 1): bins inside "
         f"{res['M1']['n_inside']} of 5 ({', '.join(res['M1']['bins_inside']) or 'none'}); pooled N_lo / ref "
         f"{f3(res['mast']['G']['bins']['all']['robust']['lo_over_ref'])}, N_hi / ref "
         f"{f3(res['mast']['G']['bins']['all']['robust']['hi_over_ref'])} -> **{hs(res['M1']['holds'])}**.",
         f"- **K1** (KSTAR, cut population, robust: N_hi / N_lo at most 1.30 with the upper end at most 1.45): "
         f"{f3(k['overall']['robust']['ratio_hi_over_lo'])} -> **{hs(res['K1']['holds'])}**.",
         f"- **K2** (N_lo in the lowest fifth of the BES level over N_lo in the highest fifth, interval above 1): "
         f"{f3(k['K2_ratio_lowest_over_highest']['robust'])} -> **{hs(res['K2']['holds'])}**.", "",
         "## KSTAR: the bracket over all discharges", "",
         f"Cut population: {k['triples']:,} triples of {k['discharges']} discharges; inclusive population: "
         f"{ki['triples']:,} triples of {ki['discharges']} discharges.", "", *HDR,
         *bracket_rows("cut", k["overall"]), *bracket_rows("inclusive", ki["overall"]), ""]
    L += ["## KSTAR: by fifth of the BES level (centre row, mean of the nine channels over the discharge median; cut)", "",
          f"Fifth edges: {', '.join(f'{x:.3f}' for x in k['bes_fifth_edges'])}.", "", *HDR]
    for j, o in enumerate(k["bes_fifths"]):
        L += bracket_rows(f"fifth {j + 1} ({o['triples']:,})", o)
    L += ["", "| version | N_lo lowest fifth / highest fifth |", "|---|---|"]
    for v in VERSIONS:
        L.append(f"| {v} | {f3(k['K2_ratio_lowest_over_highest'][v])} |")
    L += ["", "Inclusive population, same ratio: " + ", ".join(
        f"{v} {f3(ki['K2_ratio_lowest_over_highest'][v])}" for v in VERSIONS) + ".", ""]
    L += ["## KSTAR: by third of the T_i level (mean of the three rows; cut)", "",
          f"Third edges: {k['ti_third_edges_eV'][0]:.0f} and {k['ti_third_edges_eV'][1]:.0f} eV.", "", *HDR]
    for j, o in enumerate(k["ti_thirds"]):
        L += bracket_rows(f"third {j + 1} ({o['triples']:,})", o)
    L += ["", "| third | median level (eV) | robust sqrt(N_lo) / level | robust sqrt(N_hi) / level | moments sqrt(N_lo) / level | "
          "moments sqrt(N_hi) / level |", "|---|---|---|---|---|---|"]
    for j, o in enumerate(k["ti_thirds"]):
        r, m = o["sigma_over_level"]["robust"], o["sigma_over_level"]["moments"]
        L.append(f"| {j + 1} | {f1(o['median_level_eV'])} | {f3(r['sigma_lo_over_level'])} | {f3(r['sigma_hi_over_level'])} | "
                 f"{f3(m['sigma_lo_over_level'])} | {f3(m['sigma_hi_over_level'])} |")
    for lab, kk in (("cut", k), ("inclusive", ki)):
        p = kk["per_discharge"]
        L += ["", f"Per discharge (robust, {lab}; {p['discharges']} discharges with at least {MIN_TRIPLES_SHOT} triples, "
              f"N_lo <= 0 in {p['N_lo_nonpositive']}): quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of sqrt(N_lo) "
              f"{' / '.join(f'{x:.1f}' for x in p['sigma_lo_eV_quantiles'])} eV; of sqrt(N_hi) "
              f"{' / '.join(f'{x:.1f}' for x in p['sigma_hi_eV_quantiles'])} eV; of N_hi / N_lo (N_lo > 0) "
              f"{' / '.join(f'{x:.2f}' for x in p['ratio_hi_over_lo_quantiles'])}."]
    L += ["", "## KSTAR: estimators in units of the noise floor (test rows of the frozen row tables, natural cadence)", "",
          "Mean squared error over N of the same test discharges (robust); the largest skill over hold an estimator of the "
          "signal could reach is 1 - N / MSE_hold.", "",
          "| split | test discharges / rows | robust sqrt(N_lo), sqrt(N_hi) (eV) | RMSE hold / GP / twin / causal (eV) | "
          "hold / N_lo | GP / N_lo | twin / N_lo | causal / N_lo | causal / N_hi | largest skill over hold (N_lo; N_hi) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in R.SEEDS:
        o = res["rows"][str(s)]
        b, r = o["bracket"]["robust"], o["robust"]
        L.append(f"| {s} | {o['test_discharges']} / {o['test_rows']:,} | {b['sigma_lo_eV'][0]:.1f}, {b['sigma_hi_eV'][0]:.1f} | "
                 f"{' / '.join(f'{o["rmse_eV"][e]:.1f}' for e in EST)} | "
                 f"{f3(r['floors_lo']['hold'], 2)} | {f3(r['floors_lo']['gp'], 2)} | {f3(r['floors_lo']['hist80'], 2)} | "
                 f"{f3(r['floors_lo']['causal'], 2)} | {f3(r['floors_hi']['causal'], 2)} | "
                 f"{f3(r['largest_skill_over_hold_lo'])}; {f3(r['largest_skill_over_hold_hi'])} |")
    L += ["", "Same, moments version (causal / N_lo; causal / N_hi): " + "; ".join(
        f"split {s}: {f3(res['rows'][str(s)]['moments']['floors_lo']['causal'], 2)}, "
        f"{f3(res['rows'][str(s)]['moments']['floors_hi']['causal'], 2)}" for s in R.SEEDS) + ".", ""]
    L += ["## MAST: the bracket beside the reported fit errors", "",
          f"Shots as in the MAST edge study (run_coupling.SHOTS_TXT, {res['mast_status']['listed']} listed; status "
          f"{json.dumps(res['mast_status']['status'])}); blocks and scored chords from run_gate.prepare; bins of normalized "
          "radius as in run_coupling. Reference: median over the triple centre rows of the squared reported error.", ""]
    for pop in ("G", "W"):
        M = res["mast"][pop]
        L += [f"### Population {pop} (relative T_i error below {rg.POPS[pop]}): {M['triples']:,} triples of {M['shots']} shots", "",
              "| bin | triples / shots | reference median sigma (eV) | robust N_lo / ref | robust N_hi / ref | ref inside (robust) | "
              "moments N_lo / ref | moments N_hi / ref | robust N_lo / mean ref | robust N_hi / mean ref |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for nm in list(rc.BINS[:-1]) + ["all"]:
            o = M["bins"].get(nm)
            if not o:
                L.append(f"| {nm} | too few triples | | | | | | | | |")
                continue
            r, m = o["robust"], o["moments"]
            L.append(f"| {nm} | {o['triples']:,} / {o['shots']} | {o['ref_median_sigma_eV']:.1f} | {f3(r['lo_over_ref'])} | "
                     f"{f3(r['hi_over_ref'])} | {'yes' if r['ref_inside'] else 'no'} | {f3(m['lo_over_ref'])} | {f3(m['hi_over_ref'])} | "
                     f"{f3(r['lo_over_mean_ref'])} | {f3(r['hi_over_mean_ref'])} |")
        L += ["", "| bin | version | N_lo (eV^2) | N_hi (eV^2) | sqrt(N_lo) (eV) | sqrt(N_hi) (eV) | N_hi / N_lo |",
              "|---|---|---|---|---|---|---|"]
        for nm in list(rc.BINS[:-1]) + ["all"]:
            if M["bins"].get(nm):
                L += bracket_rows(nm, M["bins"][nm])
        L.append("")
    L += [f"M1 in the other versions (population G): moments {res['M1_versions']['moments']['n_inside']} bins inside, "
          f"pooled {res['M1_versions']['moments']['pooled_ok']}; clipped {res['M1_versions']['clipped']['n_inside']} bins inside, "
          f"pooled {res['M1_versions']['clipped']['pooled_ok']}. Population W (reported), robust: "
          f"{res['M1_W']['n_inside']} bins inside, pooled {res['M1_W']['pooled_ok']}.", ""]
    A = res["after_fact_mast_10ms"]
    ak = k["after_fact_K2_relative"]
    L += ["## After the fact (not registered; not used for any verdict)", "",
          "1. MAST population G with triples two frames apart (10 ms, the KSTAR frame and the cadence of the regime number); "
          f"{A['triples']:,} triples of {A['shots']} shots. Robust: bins inside {A['M1_rule']['n_inside']} of 5, pooled interval "
          f"rule {A['M1_rule']['pooled_ok']}.", "",
          "| bin | reference median sigma (eV) | robust N_lo / ref | robust N_hi / ref | robust sqrt(N_hi) (eV) | moments N_lo / ref | moments N_hi / ref |",
          "|---|---|---|---|---|---|---|"]
    for nm in list(rc.BINS[:-1]) + ["all"]:
        o = A["bins"].get(nm)
        if o:
            L.append(f"| {nm} | {o['ref_median_sigma_eV']:.1f} | {f3(o['robust']['lo_over_ref'])} | {f3(o['robust']['hi_over_ref'])} | "
                     f"{f1(o['robust']['sigma_hi_eV'])} | {f3(o['moments']['lo_over_ref'])} | {f3(o['moments']['hi_over_ref'])} |")
    L += ["", "2. KSTAR K2 with the T_i level divided out: median T_i level by BES fifth (eV) " +
          ", ".join(f1(x) for x in ak["median_level_eV_by_fifth"]) +
          f"; (robust N_lo / level^2) lowest fifth over highest fifth {f3(ak['ratio_of_N_lo_over_level2'])}.", ""]
    L += ["## Checks before the read-out", ""]
    for c in res["checks"]:
        L.append(f"- {c}")
    L += ["", "## Expectations of the registration (section 7)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in res["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    n_met = sum(e["met"] for e in res["expectations"])
    L += ["", f"{n_met} of {len(res['expectations'])} expectations met.", "", "## Deviations and interpretations", ""]
    for d in res["interpretations"]:
        L.append(f"- {d}")
    L += ["", f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


INTERP = [
    "No deviation from the registered definitions, thresholds or populations.",
    "Interpretation (not stated in the registration): a triple is three consecutive grid rows with all three T_i values "
    "measured and both time steps equal to one frame (10 ms on KSTAR within 0.1 ms, as in B.37 part Q; 5 ms on MAST within "
    "0.1 ms of the stored times). Rows separated by an absent frame (beam blip) do not form a triple.",
    "Interpretation: the clipped version clips the first differences at the 1st and 99th percentiles of both increments "
    "pooled (for N_lo, as in B.37 Q) and the second difference at its own percentiles (for N_hi), within each set of triples.",
    "Interpretation: the BES level of a row is the mean of the nine BES channels divided by the median of that mean over the "
    "rows of its discharge (first and last row dropped); fifths and T_i thirds are pooled quantiles over all triples of the population.",
    "Interpretation: the MAST reference uses the centre row of each triple (the median over all three rows is stored beside it "
    "in the json); MAST discharges are resampled among the shots that contribute triples to the population.",
    "Interpretation: the shot list is run_coupling.SHOTS_TXT (3,686 shots), which is the primary set of the MAST edge study.",
    "Interpretation: in the row-table section the bracket is computed on the triples of the test discharges of the split, "
    "from the cut-population grid, and resampled together with the test rows (same discharge weights).",
    "Interpretation: the expectation 'sigma / level 0.06 to 0.15' is checked with the robust sqrt(N_lo); the expectation "
    "'2.5 to 5 floors of the robust bound' with MSE of the causal variant over the robust N_lo.",
]


def main():
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT}
    checks = []
    C = Ctx()
    log(f"KSTAR cut grid: {len(C.names)} files, BES channels {C.nb}")
    if C.nb != 9:
        raise SystemExit("FATAL: expected nine BES channels")
    rng = np.random.default_rng(SEED)
    k_cut, tr_cut = kstar_population(C.grid, C.names, C.nb, rng, "cut")
    _t, _B, _ls, load_grid_files, *_ = R.imports()
    grid_incl, _dims = load_grid_files(R.DATA, True, R.POPS["incl"])
    names_incl = sorted(grid_incl)
    k_incl, _tr = kstar_population(grid_incl, names_incl, C.nb, np.random.default_rng(SEED + 1), "incl")
    res["kstar"] = {"cut": k_cut, "incl": k_incl}
    ic = k_cut["identity_check"]
    checks.append(f"Identity on the KSTAR cut triples: -cov(d_t, d_t+1) = {ic['minus_cov']:.4f} eV^2 and "
                  f"[var(sd) - var(span)] / 4 = {ic['var_form']:.4f} eV^2 (the moment N_lo equals the variance form).")
    hc = k_cut["overall"]["hist_check"]
    checks.append(f"Histogram approximation of the robust scale with unit weights against the exact value (KSTAR cut): "
                  f"S^2(sd) {hc['S2_sd_hist']:.2f} vs {hc['S2_sd_exact']:.2f}; S^2(span) {hc['S2_span_hist']:.2f} vs "
                  f"{hc['S2_span_exact']:.2f} eV^2.")
    res["rows"] = row_tables(tr_cut, np.random.default_rng(SEED + 2))
    del C, grid_incl
    tr_m, status, listed = mast_triples()
    res["mast_status"] = {"listed": listed, "status": status}
    res["mast"] = {pop: mast_population(tr_m[pop], np.random.default_rng(SEED + 3 + j)) for j, pop in enumerate(("G", "W"))}
    tr2, _st2, _n2 = mast_triples(step=2, pops=("G",))
    res["after_fact_mast_10ms"] = mast_population(tr2["G"], np.random.default_rng(SEED + 9))
    res["after_fact_mast_10ms"]["M1_rule"] = m1(res["after_fact_mast_10ms"])
    del tr2
    hm = res["mast"]["G"]["bins"]["all"]["hist_check"]
    checks.append(f"Same check on MAST population G pooled: S^2(sd) {hm['S2_sd_hist']:.1f} vs {hm['S2_sd_exact']:.1f}; "
                  f"S^2(span) {hm['S2_span_hist']:.1f} vs {hm['S2_span_exact']:.1f} eV^2.")
    # registered statements
    res["M1"] = m1(res["mast"]["G"])
    res["M1_versions"] = {v: m1(res["mast"]["G"], v) for v in ("moments", "clipped")}
    res["M1_W"] = m1(res["mast"]["W"])
    r = k_cut["overall"]["robust"]["ratio_hi_over_lo"]
    res["K1"] = {"ratio": r, "holds": bool(r[0] <= 1.30 and r[2] <= 1.45)}
    k2 = k_cut["K2_ratio_lowest_over_highest"]["robust"]
    res["K2"] = {"ratio": k2, "holds": bool(k2[1] > 1)}
    res["checks"] = checks
    res["expectations"] = expectations(res)
    res["interpretations"] = INTERP
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"M1 {res['M1']['holds']} K1 {res['K1']['holds']} K2 {res['K2']['holds']}; written")


if __name__ == "__main__":
    main()
