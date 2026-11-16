"""B.32, read after the fact (NOT registered; written after docs/paper/b32_two_sources.md was read).

Three readouts that the registered run called for:

A'  KSTAR, all files.  The registered table of the blip rows read once more (D2 against D1) and
    the same comparison, measured T_i change against the proportional reading of 50 eV per
    standard deviation, in rows that are not within ten rows after a blip.
B'  KSTAR, twelve discharges with a raw stream.  The departures beyond three standard deviations
    counted per channel value (the threshold of the 20-parameter filter acts per channel); the
    registered row-level count (10 rows) was below the 20 rows the registered test needed.
C'  MAST.  One event per switch (the registered definition counted one physical switch at several
    adjacent frame boundaries: 741 events in 122 shots) and, as a control for the programmed
    evolution of the discharge, windows without a switch matched on the frame index since the
    start of the charge-exchange block.

CPU only; no model.  Run from the repo root:
    py ces_prediction/experiments/b32_two_sources/posthoc_b32.py
Output: docs/paper/b32_posthoc.json / .md   (cache of the MAST frames: data/.b32_mast.pkl)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b32 as R  # noqa: E402  (adds b31_bursts and mast to sys.path)

OUT_JSON = R.REPO / "docs" / "paper" / "b32_posthoc.json"
OUT_MD = R.REPO / "docs" / "paper" / "b32_posthoc.md"
CACHE = R.DATA / ".b32_mast.pkl"
EDGES = (-9.0, -3.0, -2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0, 3.0, 9.0)
KS = np.arange(-9, 17)
BIN_FRAMES = 5
MIN_CTRL = 20
CHUNK = 200


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def wmed(v, w):
    """Weighted (lower) median of v (n,) for every row of the weights w (reps, n)."""
    o = np.argsort(v, kind="stable")
    vs = v[o]
    out = np.full(w.shape[0], np.nan)
    for a in range(0, w.shape[0], CHUNK):
        c = np.cumsum(w[a:a + CHUNK][:, o], 1)
        tot = c[:, -1]
        k = (c >= tot[:, None] / 2).argmax(1)
        r = vs[k]
        r[tot <= 0] = np.nan
        out[a:a + CHUNK] = r
    return out


# ------------------------------------------------------------------------------------------ A'
def part_a(rng):
    files = sorted(R.DATA.glob("s*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files, found {len(files)}")
    nb = len(EDGES) - 1
    S = np.zeros((len(files), nb, 3))          # rows not after a blip: n, sum of change, sum of d
    blip = []                                  # first row after a blip: file, d, ratio
    for fi, f in enumerate(files):
        k = R.kstar_file(f)
        if k is None:
            continue
        t, ti, d, after = k["t"], k["ti"], k["d"], k["after"]
        n = len(t)
        for j in k["first"]:
            i = j + 1
            pre = ti[j - 5:j + 1]
            pre = pre[np.isfinite(pre) & (pre <= 3000)]
            y = ti[i]
            if len(pre) < 3 or not np.isfinite(y) or y > 3000 or not np.isfinite(d[i]):
                continue
            lastv = ti[j] if np.isfinite(ti[j]) and ti[j] <= 3000 else np.nan
            blip.append((fi, d[i], y / float(np.median(pre)), y - float(np.median(pre)), y - lastv,
                         lastv - float(np.median(pre))))
        seg = np.concatenate(([0], np.cumsum(np.diff(t) >= R.SEG_BREAK_S)))
        last = np.full(n, np.nan)
        cur = np.nan
        for i in range(n):
            last[i] = cur
            if np.isfinite(ti[i]):
                cur = ti[i]
            if i + 1 < n and seg[i + 1] != seg[i]:
                cur = np.nan
        m = np.isfinite(d) & np.isfinite(ti) & np.isfinite(last) & (ti <= 3000) & (last <= 3000) & (after == 0)
        b = np.searchsorted(EDGES, d[m], side="right") - 1
        ok = (b >= 0) & (b < nb)
        np.add.at(S[fi, :, 0], b[ok], 1.0)
        np.add.at(S[fi, :, 1], b[ok], (ti[m] - last[m])[ok])
        np.add.at(S[fi, :, 2], b[ok], d[m][ok])
    W = rng.multinomial(len(files), np.full(len(files), 1.0 / len(files)), size=R.N_BOOT).astype(float)
    out = {"other_rows": []}
    for b in range(nb):
        nrow = S[:, b, 0].sum()
        if nrow < 50:
            continue
        den = W @ S[:, b, 0]
        ch = (W @ S[:, b, 1]) / den
        md = (W @ S[:, b, 2]) / den
        p_ch, p_d = S[:, b, 1].sum() / nrow, S[:, b, 2].sum() / nrow
        out["other_rows"].append({
            "from": EDGES[b], "to": EDGES[b + 1], "rows": int(nrow), "mean_departure_sd": float(p_d),
            "mean_change_eV": float(p_ch), "mean_change_ci95": ci(ch),
            "proportional_reading_eV": float(R.GAIN_EV_PER_SD * p_d),
            "measured_minus_proportional_eV": float(p_ch - R.GAIN_EV_PER_SD * p_d),
            "measured_minus_proportional_ci95": ci(ch - R.GAIN_EV_PER_SD * md)})
    r = np.array(blip)
    uf, inv = np.unique(r[:, 0].astype(int), return_inverse=True)
    Wb = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=R.N_BOOT).astype(float)
    g1, g2, g3 = r[:, 1] <= -3, (r[:, 1] > -3) & (r[:, 1] <= -2), r[:, 1] > -2
    med, groups = {}, {}
    for name, m in (("D1", g1), ("D2", g2), ("D3", g3)):
        med[name] = (float(np.median(r[m, 2])), wmed(r[m, 2], Wb[:, inv[m]]))
        ok = m & np.isfinite(r[:, 4])
        groups[name] = {"rows": int(m.sum()), "median_change_from_median_of_six_eV": float(np.median(r[m, 3])),
                        "median_change_ci95": ci(wmed(r[m, 3], Wb[:, inv[m]])),
                        "rows_with_the_last_row_observed": int(ok.sum()),
                        "mean_change_from_the_last_row_eV": float(r[ok, 4].mean()),
                        "mean_last_row_minus_median_of_six_eV": float(r[ok, 5].mean())}
    out["blip_rows"] = {
        "rows": int(len(r)), "groups": groups,
        "D1_minus_D2_median_ratio": med["D1"][0] - med["D2"][0],
        "D1_minus_D2_ci95": ci(med["D1"][1] - med["D2"][1]),
        "D2_minus_D3_median_ratio": med["D2"][0] - med["D3"][0],
        "D2_minus_D3_ci95": ci(med["D2"][1] - med["D3"][1])}
    return out


# ------------------------------------------------------------------------------------------ B'
def channel_departures(path):
    df = pd.read_csv(path).iloc[1:-1].reset_index(drop=True)
    t = df["time"].to_numpy(float)
    X = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float)
    z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    n = len(t)
    dt = np.diff(t)
    seg = np.concatenate(([0], np.cumsum(dt >= R.SEG_BREAK_S)))
    cs = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z, 0)])
    d = np.full_like(z, np.nan)
    for i in range(10, n):
        if seg[i] == seg[i - 10]:
            d[i] = z[i] - (cs[i] - cs[i - 10]) / 10.0
    after = np.zeros(n, int)
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        for k in range(1, 11):
            i = j + k
            if i < n and seg[i] == seg[j + 1] and after[i] == 0:
                after[i] = k
    return t, d, after


def part_b(rng):
    import run_b31 as B31
    per = []
    for shot in B31.SHOTS:
        t, d, after = channel_departures(R.DATA / f"s{shot}.csv")
        raw = B31.bursts_and_blips(shot)
        inside = (t >= raw["t0"] + 0.02) & (t <= raw["t_end"] - 0.02) & (after == 0) & np.isfinite(d[:, 0])
        ev = raw["bursts"]
        near1 = (np.searchsorted(ev, t + 0.001, side="right") - np.searchsorted(ev, t - 0.001, side="left")) > 0
        near3 = (np.searchsorted(ev, t + 0.001, side="right") - np.searchsorted(ev, t - 0.003, side="left")) > 0
        t1, lvl = raw["level_1ms"]
        run = pd.Series(lvl).rolling(200, center=True, min_periods=50).median().to_numpy()
        idx = np.clip(np.searchsorted(t1, t), 0, len(t1) - 1)
        dip = lvl[idx] < 0.8 * run[idx]
        up, dn = d > 3, d < -3
        row = lambda m: int((inside & m).sum())                       # noqa: E731
        val = lambda m, f: int((m & (inside & f)[:, None]).sum())     # noqa: E731
        alltrue = np.ones(len(t), bool)
        per.append({"shot": shot, "rows": row(alltrue), "rows_near1": row(near1), "rows_near3": row(near3),
                    "rows_dip": row(dip), "up": val(up, alltrue), "up_near1": val(up, near1),
                    "up_near3": val(up, near3), "down": val(dn, alltrue), "down_dip": val(dn, dip),
                    "down_near1": val(dn, near1),
                    "rows_any_up": row(up.any(1)), "rows_any_up_near1": row(up.any(1) & near1)})
    keys = [k for k in per[0] if k != "shot"]
    a = np.array([[p[k] for k in keys] for p in per], float)
    ix = {k: i for i, k in enumerate(keys)}
    tot = a.sum(0)
    W = rng.multinomial(len(per), np.full(len(per), 1.0 / len(per)), size=R.N_BOOT).astype(float)
    s = W @ a

    def ratio(num, den, bnum, bden):
        p = (tot[ix[num]] / tot[ix[den]]) / (tot[ix[bnum]] / tot[ix[bden]])
        with np.errstate(invalid="ignore", divide="ignore"):
            bb = (s[:, ix[num]] / s[:, ix[den]]) / (s[:, ix[bnum]] / s[:, ix[bden]])
        return {"share": float(tot[ix[num]] / tot[ix[den]]), "base_share": float(tot[ix[bnum]] / tot[ix[bden]]),
                "ratio": float(p), "ratio_ci95": ci(bb)}

    out = {"per_discharge": per, "totals": {k: int(tot[ix[k]]) for k in keys},
           "discharges_with_an_upward_value": int((a[:, ix["up"]] > 0).sum()),
           "discharges_with_a_downward_value": int((a[:, ix["down"]] > 0).sum()),
           "up_burst_within_1ms": ratio("up_near1", "up", "rows_near1", "rows"),
           "up_burst_onset_up_to_3ms_before": ratio("up_near3", "up", "rows_near3", "rows"),
           "rows_with_any_upward_value_burst_within_1ms": ratio("rows_any_up_near1", "rows_any_up", "rows_near1", "rows"),
           "down_raw_dip": ratio("down_dip", "down", "rows_dip", "rows"),
           "down_burst_within_1ms": ratio("down_near1", "down", "rows_near1", "rows")}
    return out


# ------------------------------------------------------------------------------------------ C'
def mast_frames():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    import run_coupling as rc
    import run_gate as rg
    shots = [int(x) for x in rc.SHOTS_TXT.read_text().split()]
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit", "radii_router_radius_efit",
                                            "radii_router_radius_d_alpha"]).set_index("shot_id")
    out = []
    for s in shots:
        d = rg.load_shot(s)
        if d is None:
            continue
        p = rg.prepare(d, "T_i", rg.POPS["G"])
        if p["status"] != "ok" or p["n_bad_dt"] > 0 or not p["t_finite"]:
            continue
        with np.load(rg.ACT_DIR / f"{s}.npz") as z:
            if not {"anb_time", "anb_ss_sum_power", "anb_sw_sum_power"} <= set(z.files):
                continue
            tb = z["anb_time"].astype(float) * 1000.0
            ss, sw = z["anb_ss_sum_power"].astype(float), z["anb_sw_sum_power"].astype(float)
        if ss.shape != tb.shape or sw.shape != tb.shape:
            continue
        a, _b = rc.block_of(d, "T_i", rg.POPS["G"])
        t_a = float(d["time_ms"][a])
        L = p["L"]
        try:
            ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
            ro = float(cat.loc[s, "radii_router_radius_efit"])
            if not (np.isfinite(ro) and ro > ax):
                ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
            good = np.isfinite(ax) and np.isfinite(ro) and ro > ax
            rho = (p["R"] - ax) / (ro - ax) if good else np.full(len(p["R"]), np.nan)
        except KeyError:
            rho = np.full(len(p["R"]), np.nan)
        out.append({"shot": s, "ss": rc.frame_means(tb, ss, t_a, 0, L - 1), "sw": rc.frame_means(tb, sw, t_a, 0, L - 1),
                    "G": p["G"], "Y": p["Y0"].astype(np.float32), "rho": rho})
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def classify(fss, fsw, n):
    seg_ss, seg_sw = fss[n - 9:n + 17], fsw[n - 9:n + 17]
    if not (np.isfinite(seg_ss).all() and np.isfinite(seg_sw).all()):
        return None
    m_ss = fss[n - 5:n - 1].mean()
    if not (m_ss >= 0.3 and np.all(seg_ss >= 0.3) and np.all(np.abs(seg_ss - m_ss) <= 0.2 * m_ss)):
        return None
    pre, post = fsw[n - 9:n - 1], fsw[n + 1:n + 17]
    P, Q = fsw[n - 5:n - 1].mean(), post.mean()
    if P >= 0.3 and np.all(pre >= 0.8 * P) and fsw[n - 1] >= 0.5 * P and fsw[n] < 0.5 * P and Q <= 0.2 * P:
        return "down"
    if np.all(pre <= 0.05) and Q >= 0.3 and fsw[n - 1] < 0.5 * Q and fsw[n] >= 0.5 * Q:
        return "up"
    if P >= 0.3 and np.all(np.abs(seg_sw - P) <= 0.2 * P):
        return "on"
    if np.all(seg_sw <= 0.05):
        return "off"
    return None


def trace(G, Y, n, cols):
    gpre = G[n - 5:n - 1][:, cols]
    ok = gpre.sum(0) >= 3
    if not ok.any():
        return None
    c = cols[ok]
    base = np.array([np.median(Y[n - 5:n - 1, j][G[n - 5:n - 1, j]]) for j in c])
    out = np.full(len(KS), np.nan)
    for q, k in enumerate(KS):
        g = G[n + k, c]
        if g.any():
            out[q] = np.median(Y[n + k, c][g] / base[g])
    return out


def part_c(rng):
    frames = mast_frames()
    lists = {}                       # (kind, region) -> list of (shot index, n, trace)
    for si, fr in enumerate(frames):
        L = len(fr["ss"])
        C = fr["G"].shape[1]
        regions = {"all": np.arange(C), "core": np.flatnonzero(fr["rho"] < 0.5), "edge": np.flatnonzero(fr["rho"] >= 0.7)}
        for n in range(9, L - 16):
            kind = classify(fr["ss"], fr["sw"], n)
            if kind is None or (kind in ("on", "off") and n % 4):
                continue
            for region, cols in regions.items():
                if len(cols) == 0:
                    continue
                tr = trace(fr["G"], fr["Y"], n, cols)
                if tr is not None:
                    lists.setdefault((kind, region), []).append((si, n, tr))
    W = rng.multinomial(len(frames), np.full(len(frames), 1.0 / len(frames)), size=R.N_BOOT).astype(float)
    i0, i1 = int(np.flatnonzero(KS == 0)[0]), int(np.flatnonzero(KS == 1)[0])
    late, pre = (KS >= 10) & (KS <= 16), (KS >= -9) & (KS <= -6)
    out = {"shots_with_frames": len(frames), "frames": [int(k) for k in KS], "kinds": {}}
    for kind, ctrl in (("down", "on"), ("up", "off")):
        out["kinds"][kind] = {}
        for region in ("all", "core", "edge"):
            E, Cn = lists.get((kind, region), []), lists.get((ctrl, region), [])
            if not E:
                continue
            fe, ne, Te = np.array([e[0] for e in E]), np.array([e[1] for e in E]), np.array([e[2] for e in E])
            cell = {"events": int(len(E)), "shots": int(len(set(fe))),
                    "frame_index_median": float(np.median(ne)),
                    "frame_index_quartiles": [float(np.quantile(ne, 0.25)), float(np.quantile(ne, 0.75))]}
            We = W[:, fe]
            med = np.array([np.nanmedian(Te[:, q]) for q in range(len(KS))])
            boots = np.array([wmed(Te[np.isfinite(Te[:, q]), q], We[:, np.isfinite(Te[:, q])]) for q in range(len(KS))])
            r_late, b_late = float(med[late].mean()), np.nanmean(boots[late], 0)
            cell.update({"median_ratio": [float(x) for x in med], "median_ratio_ci95": [ci(b) for b in boots],
                         "R_0": float(med[i0]), "R_0_ci95": ci(boots[i0]), "R_1": float(med[i1]), "R_1_ci95": ci(boots[i1]),
                         "R_late": r_late, "R_late_ci95": ci(b_late),
                         "delay_statistic": float(abs(1 - r_late) - 2 * abs(1 - med[i1])),
                         "delay_ci95": ci(np.abs(1 - b_late) - 2 * np.abs(1 - boots[i1]))})
            half = None
            for q in np.flatnonzero(KS >= 0):
                if abs(1 - med[q]) >= 0.5 * abs(1 - r_late):
                    half = int(KS[q])
                    break
            cell["half_response_frame"] = half
            if Cn:
                fc, nc, Tc = np.array([e[0] for e in Cn]), np.array([e[1] for e in Cn]), np.array([e[2] for e in Cn])
                be, bc = ne // BIN_FRAMES, nc // BIN_FRAMES
                nbin = int(max(be.max(), bc.max())) + 1
                cb = np.bincount(bc, minlength=nbin)
                keep = cb[be] >= MIN_CTRL
                keepc = np.isin(bc, np.unique(be[keep]))
                fe2, be2, Te2 = fe[keep], be[keep], Te[keep]
                fc2, bc2, Tc2 = fc[keepc], bc[keepc], Tc[keepc]
                cell.update({"matched_events": int(keep.sum()), "control_windows": int(keepc.sum()),
                             "control_shots": int(len(set(fc2)))})
                if keep.sum() >= 5:
                    mb = np.bincount(be2, minlength=nbin).astype(float)
                    cb2 = np.bincount(bc2, minlength=nbin).astype(float)
                    wpoint = (mb / np.maximum(cb2, 1))[bc2][None, :]
                    Oe = np.zeros((len(be2), nbin))
                    Oe[np.arange(len(be2)), be2] = 1.0
                    Oc = np.zeros((len(bc2), nbin))
                    Oc[np.arange(len(bc2)), bc2] = 1.0
                    We2, Wc2 = W[:, fe2], W[:, fc2]
                    MB, CB = We2 @ Oe, Wc2 @ Oc
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = np.where(CB > 0, MB / CB, 0.0)
                    Wc2 = Wc2 * ratio[:, bc2]
                    D, Db, Rc = [], [], []
                    for q in range(len(KS)):
                        oe, oc = np.isfinite(Te2[:, q]), np.isfinite(Tc2[:, q])
                        pe = float(np.median(Te2[oe, q]))
                        pc = float(wmed(Tc2[oc, q], wpoint[:, oc])[0])
                        D.append(pe - pc)
                        Rc.append(pc)
                        Db.append(wmed(Te2[oe, q], We2[:, oe]) - wmed(Tc2[oc, q], Wc2[:, oc]))
                    D, Db = np.array(D), np.array(Db)
                    d_late, db_late = float(D[late].mean()), np.nanmean(Db[late], 0)
                    d_pre, db_pre = float(D[pre].mean()), np.nanmean(Db[pre], 0)
                    halfd = None
                    for q in np.flatnonzero(KS >= 0):
                        if abs(D[q]) >= 0.5 * abs(d_late):
                            halfd = int(KS[q])
                            break
                    cell.update({"control_median_ratio": [float(x) for x in Rc],
                                 "net": [float(x) for x in D], "net_ci95": [ci(b) for b in Db],
                                 "net_0": float(D[i0]), "net_0_ci95": ci(Db[i0]),
                                 "net_1": float(D[i1]), "net_1_ci95": ci(Db[i1]),
                                 "net_late": d_late, "net_late_ci95": ci(db_late),
                                 "net_placebo_frames_-9_to_-6": d_pre, "net_placebo_ci95": ci(db_pre),
                                 "net_delay_statistic": float(abs(d_late) - 2 * abs(D[i1])),
                                 "net_delay_ci95": ci(np.abs(db_late) - 2 * np.abs(Db[i1])),
                                 "net_half_response_frame": halfd})
            out["kinds"][kind][region] = cell
    return out


def f3(x, lo_hi=None, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    if lo_hi is None:
        return s
    a, b = lo_hi
    return s + (f" [{a:+.3f}, {b:+.3f}]" if sign else f" [{a:.3f}, {b:.3f}]")


def render(res):
    A, B, C = res["A"], res["B"], res["C"]
    L = ["# B.32, read after the fact (not registered)", "",
         "Written after the registered readout (`b32_two_sources.md`) had been read. CPU, no model. "
         "Intervals: 95 %, 2,000 resamples of discharges (shots on MAST).", "",
         "## A'. KSTAR, all files: measured T_i change against a proportional reading", "",
         "Rows that are not within ten rows after a blip. Change of the measured T_i from its last observed value, "
         "by the mean departure d of the nine BES channels of the row; proportional reading = 50 eV per standard "
         "deviation times the mean d of the bin. Values above 3 keV left out.", "",
         "| mean BES departure (sd) | rows | mean d (sd) | mean T_i change (eV) | proportional reading (eV) | measured minus proportional (eV) |",
         "|---|---|---|---|---|---|"]
    for r in A["other_rows"]:
        L.append(f"| {r['from']:+.1f} to {r['to']:+.1f} | {r['rows']:,} | {r['mean_departure_sd']:+.2f} | "
                 f"{r['mean_change_eV']:+.1f} [{r['mean_change_ci95'][0]:+.1f}, {r['mean_change_ci95'][1]:+.1f}] | "
                 f"{r['proportional_reading_eV']:+.1f} | {r['measured_minus_proportional_eV']:+.1f} "
                 f"[{r['measured_minus_proportional_ci95'][0]:+.1f}, {r['measured_minus_proportional_ci95'][1]:+.1f}] |")
    bl = A["blip_rows"]
    L += ["", f"First row after a blip ({bl['rows']:,} rows, the registered table): median T_i ratio of D1 minus D2 = "
          f"{f3(bl['D1_minus_D2_median_ratio'], bl['D1_minus_D2_ci95'], True)}; D2 minus D3 = "
          f"{f3(bl['D2_minus_D3_median_ratio'], bl['D2_minus_D3_ci95'], True)}.", "",
          "| group | rows | median T_i change from the median of the six rows before (eV) | rows with the last row before observed | "
          "mean change from that last row (eV) | mean of that last row minus the median of six (eV) |", "|---|---|---|---|---|---|"]
    for name, lab in (("D1", "D1 (d <= -3)"), ("D2", "D2 (-3 < d <= -2)"), ("D3", "D3 (d > -2)")):
        g = bl["groups"][name]
        L.append(f"| {lab} | {g['rows']} | {g['median_change_from_median_of_six_eV']:+.1f} "
                 f"[{g['median_change_ci95'][0]:+.1f}, {g['median_change_ci95'][1]:+.1f}] | {g['rows_with_the_last_row_observed']} | "
                 f"{g['mean_change_from_the_last_row_eV']:+.1f} | {g['mean_last_row_minus_median_of_six_eV']:+.1f} |")
    L += ["",
          "## B'. KSTAR, twelve discharges with a raw stream: departures beyond three standard deviations per channel value", "",
          "Rows inside the raw window and not within ten rows after a blip; nine BES channel values per row. "
          "Burst: the B.31 detector, unchanged. Ratio = share among the channel values (or rows) named, over the share among all rows.", ""]
    T = B["totals"]

    def line(label, key, n_key):
        v = B[key]
        return (f"- {label}: {T[n_key]:,} values; share {v['share']:.3f} against {v['base_share']:.3f} over all rows; "
                f"ratio {v['ratio']:.1f} [{v['ratio_ci95'][0]:.1f}, {v['ratio_ci95'][1]:.1f}].")

    L += [f"Rows: {T['rows']:,}. Channel values above +3: {T['up']:,} in {B['discharges_with_an_upward_value']} discharges "
          f"(rows with at least one: {T['rows_any_up']:,}); below -3: {T['down']:,} in "
          f"{B['discharges_with_a_downward_value']} discharges.", "",
          line("above +3, a detected burst within 1 ms of the time stamp", "up_burst_within_1ms", "up"),
          line("above +3, a burst onset from 3 ms before to 1 ms after the stamp", "up_burst_onset_up_to_3ms_before", "up"),
          f"- rows with at least one channel above +3, a burst within 1 ms: share "
          f"{B['rows_with_any_upward_value_burst_within_1ms']['share']:.3f}; ratio "
          f"{B['rows_with_any_upward_value_burst_within_1ms']['ratio']:.1f} "
          f"[{B['rows_with_any_upward_value_burst_within_1ms']['ratio_ci95'][0]:.1f}, "
          f"{B['rows_with_any_upward_value_burst_within_1ms']['ratio_ci95'][1]:.1f}].",
          line("below -3, raw BES level at the stamp below 0.8 of its 200 ms running median", "down_raw_dip", "down"),
          line("below -3, a detected burst within 1 ms", "down_burst_within_1ms", "down"), "",
          "| discharge | rows | rows with a burst within 1 ms | values above +3 | with a burst within 1 ms | values below -3 | with a raw dip |",
          "|---|---|---|---|---|---|---|"]
    for p in B["per_discharge"]:
        L.append(f"| {p['shot']} | {p['rows']} | {p['rows_near1']} | {p['up']} | {p['up_near1']} | {p['down']} | {p['down_dip']} |")
    L += ["", "## C'. MAST: one event per switch of the SW beam, and windows without a switch as a control", "",
          "Event at frame 0: the first 5 ms frame in which the SW power is on the other side of half its level; the eight "
          "frames from -9 to -2 are in the state before (frame -1 may hold the switch), frames +1 to +16 in the state "
          "after, and the SS power is steady from -9 to +16. Ratio of T_i to its level in frames -5 to -2 (median over "
          "chords, then over events). Control: windows in which the SW beam stays as it was before the switch, weighted "
          f"to the events' distribution of the frame index since the start of the block (bins of {BIN_FRAMES} frames, at "
          f"least {MIN_CTRL} control windows). Net = events minus control. "
          f"Shots with a block and both beam traces: {C['shots_with_frames']:,}.", ""]
    for kind, lab in (("down", "SW switched off"), ("up", "SW switched on")):
        for region in ("all", "core", "edge"):
            c = C["kinds"].get(kind, {}).get(region)
            if not c:
                continue
            s = (f"- **{lab}, {region}**: {c['events']} events in {c['shots']} shots (frame index median "
                 f"{c['frame_index_median']:.0f}, quartiles {c['frame_index_quartiles'][0]:.0f} to "
                 f"{c['frame_index_quartiles'][1]:.0f}). Ratio at frame 0 {f3(c['R_0'], c['R_0_ci95'])}, at +1 "
                 f"{f3(c['R_1'], c['R_1_ci95'])}, at 50 to 80 ms {f3(c['R_late'], c['R_late_ci95'])}; delay statistic "
                 f"{f3(c['delay_statistic'], c['delay_ci95'], True)}; half response at frame {c['half_response_frame']}.")
            if "net" in c:
                s += (f" Against {c['control_windows']:,} control windows of {c['control_shots']} shots "
                      f"({c['matched_events']} events matched): net at frame 0 {f3(c['net_0'], c['net_0_ci95'], True)}, at +1 "
                      f"{f3(c['net_1'], c['net_1_ci95'], True)}, at 50 to 80 ms {f3(c['net_late'], c['net_late_ci95'], True)}; "
                      f"frames -9 to -6 (no response expected) {f3(c['net_placebo_frames_-9_to_-6'], c['net_placebo_ci95'], True)}; "
                      f"net delay statistic {f3(c['net_delay_statistic'], c['net_delay_ci95'], True)}; "
                      f"half of the net response at frame {c['net_half_response_frame']}.")
            L.append(s)
    L += ["", "| frame | " + " | ".join(f"{k:+d}" for k in C["frames"]) + " |", "|---|" + "---|" * len(C["frames"])]
    for kind, lab in (("down", "SW off"), ("up", "SW on")):
        for region in ("core", "edge"):
            c = C["kinds"].get(kind, {}).get(region)
            if not c:
                continue
            L.append(f"| {lab}, {region}: events | " + " | ".join(f"{x:.3f}" for x in c["median_ratio"]) + " |")
            if "net" in c:
                L.append(f"| {lab}, {region}: control | " + " | ".join(f"{x:.3f}" for x in c["control_median_ratio"]) + " |")
                L.append(f"| {lab}, {region}: net | " + " | ".join(f"{x:+.3f}" for x in c["net"]) + " |")
    return "\n".join(L) + "\n"


def main():
    rng = np.random.default_rng(R.SEED + 1)
    res = {"note": "read after the fact; not registered"}
    res["A"] = part_a(rng)
    print("A done", flush=True)
    res["B"] = part_b(rng)
    print("B done", flush=True)
    res["C"] = part_c(rng)
    print("C done", flush=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
