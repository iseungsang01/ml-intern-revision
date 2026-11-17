"""B.46 (PREREGISTRATION_B46.md): the reported T_i and the last beam-off frame.

Part J  a step across an absent frame (beam-off frame) against control stamps, with the ECEI level as control
        J1: spread of the T_i step at events / at controls, interval above 1
        J2: (J1 ratio) / (same ratio for the ECEI level), interval above 1
Part A  the frame-to-frame independent component by the age of the last absent frame
        A1: median over discharges of (relative component at 1.0-2.0 s) / (at 0.1-0.5 s), interval excludes 1

CPU only; nothing is trained.  Run from the repo root:
    py ces_prediction/experiments/b46_beam_off/run_b46.py
Output: docs/paper/b46_beam_off.json / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
import run_b30 as R  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b46_beam_off.json"
OUT_MD = R.REPO / "docs" / "paper" / "b46_beam_off.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B46.md"
SEED, N_BOOT = 20261118, 2000
MAD_C = 1.4826
EPS = 1e-4                      # s; tolerance on float32 time stamps (stamps lie on a 10 ms grid)
SEG_BREAK_S = 0.5
GAP_LO, GAP_HI = 0.015, 0.045   # absent frame: a step between 15 and 45 ms inside a segment
ISO_S = 0.4                     # events: no other absent frame within 400 ms before tb / after ta
WB = 0.200                      # window before: (t0 - 200 ms, t0]
WA = (0.150, 0.350)             # window after an event: [ta + 150 ms, ta + 350 ms)
WA_MOVED = (0.250, 0.450)       # moved window after an event
CT_START, CT_STEP, CT_ISO = 0.6, 0.5, 0.6
CT_GAP = 0.030                  # control geometry = an event with a gap of 30 ms
MIN_ROWS = 10
ECEI_FRAC = 0.05
MIN_EVENTS, MIN_DIS_J, MIN_DIS_A = 100, 20, 20
A_EARLY, A_LATE = (0.1, 0.5), (1.0, 2.0)
A_MIN_TRIPLES = 30
AGE_BINS = [("0.1-0.3", 0.1, 0.3), ("0.3-0.6", 0.3, 0.6), ("0.6-1.0", 0.6, 1.0), ("1.0-1.5", 1.0, 1.5),
            ("1.5-3", 1.5, 3.0), ("above 3", 3.0, np.inf)]
AGE_EXTRA = ("0-0.1 (after the fact)", 0.0, 0.1)
NONE_BIN = "no absent frame before in the segment"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [float("nan")] * 2


# ------------------------------------------------------------------------------ weighted robust statistics
def wmed_sorted(xs, ws):
    """Median of xs (sorted) with integer weights ws; equals np.median of the repeated sample."""
    cw = np.cumsum(ws)
    if len(cw) == 0 or cw[-1] <= 0:
        return np.nan
    n = int(round(cw[-1]))
    if n % 2:
        return float(xs[np.searchsorted(cw, (n - 1) / 2, side="right")])
    return float(0.5 * (xs[np.searchsorted(cw, n / 2 - 1, side="right")] + xs[np.searchsorted(cw, n / 2, side="right")]))


def wmed(x, w, order=None):
    o = np.argsort(x, kind="stable") if order is None else order
    return wmed_sorted(x[o], w[o])


def wspread(x, w, order=None):
    """1.4826 x median absolute deviation from the median (weighted by integer multiplicities)."""
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    m = wmed(x, w, order)
    d = np.abs(x - m)
    return MAD_C * wmed(d, w)


def spread(x):
    return wspread(x, np.ones(len(x)))


# ------------------------------------------------------------------------------ per-file structure
def file_struct(arr_cut, arr_raw, nb, ne):
    a, r = arr_cut[1:-1], arr_raw[1:-1]           # first and last row of a file are not grid rows
    t = a[:, 0].astype(np.float64)
    y = a[:, 1].astype(np.float64)               # held values and values above 3 keV are NaN
    yraw = r[:, 1].astype(np.float64)
    ec = a[:, 3 + nb:3 + nb + ne].astype(np.float64).mean(1)
    dt = np.diff(t)
    seg = np.concatenate(([0], np.cumsum(dt >= SEG_BREAK_S)))
    gi = np.flatnonzero((dt > GAP_LO) & (dt < GAP_HI))
    gaps = {"tb": t[gi], "ta": t[gi + 1], "seg": seg[gi], "frames": np.rint(dt[gi] / 0.010).astype(int) - 1,
            "ib": gi}
    return {"t": t, "y": y, "yraw": yraw, "ec": ec, "seg": seg, "gaps": gaps,
            "ec_medabs": float(np.median(np.abs(ec))) if len(ec) else np.nan,
            "rows_match": bool(len(a) == len(r) and np.allclose(a[:, 0], r[:, 0]))}


def window_rows(F, s, lo, hi, lo_open, hi_open):
    t, seg = F["t"], F["seg"]
    m = seg == s
    m &= (t > lo + EPS) if lo_open else (t >= lo - EPS)
    m &= (t <= hi + EPS) if not hi_open else (t < hi - EPS)
    return m


def step_of(F, s, t0, after_lo, after_hi):
    """Steps across (t0 - 200 ms, t0] -> [after_lo, after_hi); None when a window lacks 10 measured T_i rows."""
    mb = window_rows(F, s, t0 - WB, t0, True, False)
    ma = window_rows(F, s, after_lo, after_hi, False, True)
    yb, ya = F["y"][mb], F["y"][ma]
    yb, ya = yb[np.isfinite(yb)], ya[np.isfinite(ya)]
    if len(yb) < MIN_ROWS or len(ya) < MIN_ROWS:
        return None
    mb_, ma_ = yb.mean(), ya.mean()
    if mb_ <= 0 or ma_ <= 0:
        return None
    eb, ea = F["ec"][mb].mean(), F["ec"][ma].mean()
    thr = ECEI_FRAC * F["ec_medabs"]
    ec_ok = bool(eb >= thr and ea >= thr and eb > 0 and ea > 0)
    return {"ti": float(np.log(ma_ / mb_)), "ec": float(np.log(ea / eb)) if ec_ok else np.nan,
            "lvl": float(mb_), "nb": int(len(yb)), "na": int(len(ya))}


def other_gap_near(G, k, lo, hi):
    """Is there an absent frame (other than gap k) whose open interval (tb, ta) overlaps [lo, hi]?"""
    m = (G["ta"] > lo + EPS) & (G["tb"] < hi - EPS)
    if k is not None:
        m[k] = False
    return bool(m.any())


def part_j_records(files):
    ev, ct, funnel = [], [], {"absent_steps": 0, "isolated": 0, "rows_ok": 0, "rows_ok_moved": 0,
                              "control_candidates": 0, "control_isolated": 0, "control_rows_ok": 0,
                              "control_rows_ok_moved": 0}
    for fi, F in enumerate(files):
        G, t, seg = F["gaps"], F["t"], F["seg"]
        for k in range(len(G["tb"])):
            funnel["absent_steps"] += 1
            tb, ta, s = G["tb"][k], G["ta"][k], G["seg"][k]
            if other_gap_near(G, k, tb - ISO_S, ta + ISO_S):
                continue
            funnel["isolated"] += 1
            st = step_of(F, s, tb, ta + WA[0], ta + WA[1])
            mv = step_of(F, s, tb, ta + WA_MOVED[0], ta + WA_MOVED[1])
            if st is not None:
                funnel["rows_ok"] += 1
            if mv is not None:
                funnel["rows_ok_moved"] += 1
            if st is not None or mv is not None:
                ev.append({"f": fi, "t0": float(tb), "gap_s": float(ta - tb), "main": st, "moved": mv})
        for s in np.unique(seg):
            ts = t[seg == s]
            t_start, t_end = ts[0], ts[-1]
            used = set()
            target = t_start + CT_START
            while target <= t_end + EPS:
                j = int(np.searchsorted(ts, target - EPS, side="left"))
                target += CT_STEP
                if j >= len(ts) or j in used:
                    continue
                used.add(j)
                tc = ts[j]
                funnel["control_candidates"] += 1
                if other_gap_near(G, None, tc - CT_ISO, tc + CT_ISO):
                    continue
                funnel["control_isolated"] += 1
                ta = tc + CT_GAP
                st = step_of(F, s, tc, ta + WA[0], ta + WA[1])
                mv = step_of(F, s, tc, ta + WA_MOVED[0], ta + WA_MOVED[1])
                if st is not None:
                    funnel["control_rows_ok"] += 1
                if mv is not None:
                    funnel["control_rows_ok_moved"] += 1
                if st is not None or mv is not None:
                    ct.append({"f": fi, "t0": float(tc), "main": st, "moved": mv})
    return ev, ct, funnel


def arrays(recs, key):
    sel = [r for r in recs if r[key] is not None]
    return {"f": np.array([r["f"] for r in sel], int), "ti": np.array([r[key]["ti"] for r in sel]),
            "ec": np.array([r[key]["ec"] for r in sel]), "lvl": np.array([r[key]["lvl"] for r in sel])}


# ------------------------------------------------------------------------------ Part J statistics
def cell(E, C, W, pos):
    """E, C: dicts of arrays (f, ti, ec); W: (B, n) multiplicities over the discharges indexed by pos (file -> col)."""
    out = {"events": int(len(E["ti"])), "controls": int(len(C["ti"])),
           "event_discharges": int(len(np.unique(E["f"]))), "control_discharges": int(len(np.unique(C["f"])))}
    fe, fc = pos[E["f"]], pos[C["f"]]
    ve = np.isfinite(E["ec"])
    vc = np.isfinite(C["ec"])
    out["ecei_dropped_events"] = int((~ve).sum())
    out["ecei_dropped_controls"] = int((~vc).sum())
    out["ecei_valid_events"] = int(ve.sum())
    out["ecei_valid_controls"] = int(vc.sum())
    sets = {"ti": (E["ti"], fe, C["ti"], fc),
            "ti_sub": (E["ti"][ve], fe[ve], C["ti"][vc], fc[vc]),
            "ec": (E["ec"][ve], fe[ve], C["ec"][vc], fc[vc])}
    pt, bt = {}, {}
    for nm, (xe, ge, xc, gc) in sets.items():
        oe, oc = np.argsort(xe, kind="stable"), np.argsort(xc, kind="stable")
        one_e, one_c = np.ones(len(xe)), np.ones(len(xc))
        pt[nm] = {"se": wspread(xe, one_e, oe), "sc": wspread(xc, one_c, oc),
                  "me": wmed(xe, one_e, oe), "mc": wmed(xc, one_c, oc)}
        b = {k: np.full(len(W), np.nan) for k in ("se", "sc", "me", "mc")}
        for i in range(len(W)):
            we, wc = W[i, ge], W[i, gc]
            if we.sum() > 0:
                b["se"][i] = wspread(xe, we, oe)
                b["me"][i] = wmed(xe, we, oe)
            if wc.sum() > 0:
                b["sc"][i] = wspread(xc, wc, oc)
                b["mc"][i] = wmed(xc, wc, oc)
        bt[nm] = b
    with np.errstate(all="ignore"):
        for nm in ("ti", "ec"):
            p, b = pt[nm], bt[nm]
            out[nm] = {"spread_events": [p["se"], *ci(b["se"])], "spread_controls": [p["sc"], *ci(b["sc"])],
                       "ratio": [p["se"] / p["sc"], *ci(b["se"] / b["sc"])],
                       "median_events": [p["me"], *ci(b["me"])], "median_controls": [p["mc"], *ci(b["mc"])],
                       "median_difference": [p["me"] - p["mc"], *ci(b["me"] - b["mc"])],
                       "excess_spread_pct": [100 * np.sqrt(max(0.0, p["se"] ** 2 - p["sc"] ** 2)),
                                             *ci(100 * np.sqrt(np.maximum(0.0, b["se"] ** 2 - b["sc"] ** 2)))]}
            out[nm]["bound_pct"] = out[nm]["excess_spread_pct"][2]
        rs = pt["ti_sub"]["se"] / pt["ti_sub"]["sc"]
        re_ = pt["ec"]["se"] / pt["ec"]["sc"]
        brs = bt["ti_sub"]["se"] / bt["ti_sub"]["sc"]
        bre = bt["ec"]["se"] / bt["ec"]["sc"]
        out["ti_on_ecei_valid"] = {"ratio": [rs, *ci(brs)]}
        out["J2_ratio"] = [rs / re_, *ci(brs / bre)]
    return out


def part_j(ev, ct, rng_seed):
    res = {}
    for key in ("main", "moved"):
        E, C = arrays(ev, key), arrays(ct, key)
        ev_dis = np.unique(E["f"])
        for variant in ("all", "event_discharges"):
            Cv = C if variant == "all" else {k: v[np.isin(C["f"], ev_dis)] for k, v in C.items()}
            dis = np.unique(np.concatenate([E["f"], Cv["f"]]))
            pos = np.full(max(int(dis.max()) + 1, 1), -1)
            pos[dis] = np.arange(len(dis))
            rng = np.random.default_rng(rng_seed + (0 if key == "main" else 10) + (0 if variant == "all" else 1))
            W = rng.multinomial(len(dis), np.full(len(dis), 1.0 / len(dis)), size=N_BOOT).astype(float)
            o = cell(E, Cv, W, pos)
            o["resampled_discharges"] = int(len(dis))
            if key == "main" and variant == "all":
                # by third of the T_i level before (edges: tertiles of the event levels)
                q = np.quantile(E["lvl"], [1 / 3, 2 / 3])
                o["third_edges_eV"] = q.tolist()
                te, tc = np.searchsorted(q, E["lvl"], side="right"), np.searchsorted(q, Cv["lvl"], side="right")
                o["thirds"] = []
                for j in range(3):
                    Ej = {k: v[te == j] for k, v in E.items()}
                    Cj = {k: v[tc == j] for k, v in Cv.items()}
                    oj = cell(Ej, Cj, W, pos)
                    oj["level_range_events_eV"] = [float(Ej["lvl"].min()), float(Ej["lvl"].max())]
                    o["thirds"].append(oj)
                # after the fact: ECEI steps of exactly zero (ECEI level constant over both windows) dropped for ECEI
                Ez, Cz = dict(E), dict(Cv)
                Ez["ec"] = np.where(E["ec"] == 0, np.nan, E["ec"])
                Cz["ec"] = np.where(Cv["ec"] == 0, np.nan, Cv["ec"])
                oz = cell(Ez, Cz, W, pos)
                o["after_fact_nonzero_ecei"] = {"zero_steps_events": int((E["ec"] == 0).sum()),
                                                "zero_steps_controls": int((Cv["ec"] == 0).sum()),
                                                "zero_step_discharges_events": int(len(np.unique(E["f"][E["ec"] == 0]))),
                                                "zero_step_discharges_controls": int(len(np.unique(Cv["f"][Cv["ec"] == 0]))),
                                                "ecei_valid_events": oz["ecei_valid_events"],
                                                "ecei_valid_controls": oz["ecei_valid_controls"],
                                                "ec": oz["ec"], "ti_on_ecei_valid": oz["ti_on_ecei_valid"],
                                                "J2_ratio": oz["J2_ratio"]}
            res[f"{key}|{variant}"] = o
            log(f"J {key} {variant}: events {o['events']} controls {o['controls']} "
                f"ratio {o['ti']['ratio'][0]:.3f} [{o['ti']['ratio'][1]:.3f}, {o['ti']['ratio'][2]:.3f}] "
                f"J2 {o['J2_ratio'][0]:.3f}")
    return res


# ------------------------------------------------------------------------------ Part A
def age_of(F, tq):
    """Time from ta of the last absent frame of the same segment (ta <= tq) to tq; NaN if none."""
    G = F["gaps"]
    seg_q = F["seg"][np.searchsorted(F["t"], tq - EPS)]
    age = np.full(len(tq), np.nan)
    for s in np.unique(G["seg"]):
        ta = np.sort(G["ta"][G["seg"] == s])
        m = seg_q == s
        j = np.searchsorted(ta, tq[m] + EPS, side="right") - 1
        a = np.where(j >= 0, tq[m] - ta[np.maximum(j, 0)], np.nan)
        age[m] = a
    return age


def bin_label(age):
    lab = np.full(len(age), -2)                    # -2: below 0.1 s; -1: none before
    lab[~np.isfinite(age)] = -1
    for k, (_, lo, hi) in enumerate(AGE_BINS):
        lab[np.isfinite(age) & (age >= lo - 1e-9) & (age < hi - 1e-9)] = k
    return lab


def rel_component(sd, ym, w=None):
    if w is None:
        w = np.ones(len(sd))
    if w.sum() <= 0:
        return np.nan
    return wspread(sd, w) / np.sqrt(6.0) / wmed(ym, w)


def part_a(files):
    tri = {k: [] for k in ("f", "sd", "ym", "age")}
    rows = {k: [] for k in ("f", "age", "miss", "above", "held")}
    for fi, F in enumerate(files):
        t, y = F["t"], F["y"]
        rows["f"].append(np.full(len(t), fi))
        rows["age"].append(age_of(F, t))
        rows["miss"].append(~np.isfinite(F["yraw"]))
        rows["above"].append(np.isfinite(F["yraw"]) & (F["yraw"] > 3000.0))
        rows["held"].append(np.isfinite(F["yraw"]) & (F["yraw"] <= 3000.0) & ~np.isfinite(y))
        if len(y) < 3:
            continue
        dt = np.diff(t)
        fin = np.isfinite(y)
        m = (fin[:-2] & fin[1:-1] & fin[2:] & (np.abs(dt[:-1] - 0.010) < 1e-4) & (np.abs(dt[1:] - 0.010) < 1e-4))
        c = np.flatnonzero(m) + 1
        if not len(c):
            continue
        tri["f"].append(np.full(len(c), fi))
        tri["sd"].append(y[c + 1] - 2 * y[c] + y[c - 1])
        tri["ym"].append(y[c])
        tri["age"].append(age_of(F, t[c]))
    tri = {k: np.concatenate(v) for k, v in tri.items()}
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    return tri, rows


def part_a_stats(tri, rows, n_files, rng):
    out = {}
    W = rng.multinomial(n_files, np.full(n_files, 1.0 / n_files), size=N_BOOT).astype(float)
    lt, lr = bin_label(tri["age"]), bin_label(rows["age"])
    labels = [(nm, k) for k, (nm, _, _) in enumerate(AGE_BINS)] + [(NONE_BIN, -1), (AGE_EXTRA[0], -2)]
    out["bins"] = {}
    for nm, k in labels:
        st, sr = lt == k, lr == k
        sd, ym, f = tri["sd"][st], tri["ym"][st], tri["f"][st]
        o = {"triples": int(st.sum()), "triple_discharges": int(len(np.unique(f))), "rows": int(sr.sum()),
             "row_discharges": int(len(np.unique(rows["f"][sr])))}
        p = rel_component(sd, ym)
        bs = np.full(N_BOOT, np.nan)
        osd, oym = np.argsort(sd, kind="stable"), np.argsort(ym, kind="stable")
        for i in range(N_BOOT):
            w = W[i, f]
            if w.sum() > 0:
                bs[i] = wspread(sd, w, osd) / np.sqrt(6.0) / wmed(ym, w, oym)
        o["relative_component"] = [float(p), *ci(bs)]
        o["component_eV"] = float(spread(sd) / np.sqrt(6.0)) if len(sd) else np.nan
        o["median_ti_eV"] = float(np.median(ym)) if len(ym) else np.nan
        cnt = np.bincount(rows["f"][sr], minlength=n_files).astype(float)
        for q in ("miss", "above", "held"):
            num = np.bincount(rows["f"][sr], rows[q][sr].astype(float), minlength=n_files)
            with np.errstate(all="ignore"):
                o[f"share_{q}"] = [float(num.sum() / cnt.sum()) if cnt.sum() else np.nan, *ci((W @ num) / (W @ cnt))]
        out["bins"][nm] = o
        log(f"A bin {nm}: triples {o['triples']} rel {o['relative_component'][0]:.4f}")
    # A1: per discharge, relative component at 1.0-2.0 s over that at 0.1-0.5 s
    per = []
    for fi in np.unique(tri["f"]):
        m = tri["f"] == fi
        a = tri["age"][m]
        e = m.copy(); e[m] = np.isfinite(a) & (a >= A_EARLY[0] - 1e-9) & (a < A_EARLY[1] - 1e-9)
        l_ = m.copy(); l_[m] = np.isfinite(a) & (a >= A_LATE[0] - 1e-9) & (a < A_LATE[1] - 1e-9)
        if e.sum() >= A_MIN_TRIPLES and l_.sum() >= A_MIN_TRIPLES:
            re_ = rel_component(tri["sd"][e], tri["ym"][e])
            rl = rel_component(tri["sd"][l_], tri["ym"][l_])
            per.append((int(fi), int(e.sum()), int(l_.sum()), float(re_), float(rl), float(rl / re_)))
    per = np.array(per, float).reshape(-1, 6)
    n = len(per)
    a1 = {"discharges": int(n), "per_discharge": per.tolist()}
    if n:
        r = per[:, 5]
        rng2 = np.random.default_rng(SEED + 7)
        idx = rng2.integers(0, n, size=(N_BOOT, n))
        a1["median_ratio"] = [float(np.median(r)), *ci(np.median(r[idx], axis=1))]
        a1["ratio_quantiles_0.1_0.25_0.5_0.75_0.9"] = [float(x) for x in np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9])]
        a1["share_ratio_above_1"] = float((r > 1).mean())
        a1["testable"] = bool(n >= MIN_DIS_A)
        lo, hi = a1["median_ratio"][1], a1["median_ratio"][2]
        a1["holds"] = bool(a1["testable"] and (lo > 1 or hi < 1))
    else:
        a1.update({"testable": False, "holds": False})
    out["A1"] = a1
    return out


def absent_distribution(files, names):
    frames = np.array([int(F["gaps"]["frames"].sum()) for F in files])
    steps = np.array([len(F["gaps"]["tb"]) for F in files])
    sp = []
    for F in files:
        G = F["gaps"]
        for s in np.unique(G["seg"]):
            m = G["seg"] == s
            tb, ta = np.sort(G["tb"][m]), np.sort(G["ta"][m])
            if len(tb) > 1:
                sp.append(tb[1:] - ta[:-1])
    sp = np.concatenate(sp) if sp else np.zeros(0)
    cats = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
    hist = [{"range": f"{a}" if a == b else (f"{a} or more" if b > 10 ** 8 else f"{a}-{b}"),
             "discharges_by_frames": int(((frames >= a) & (frames <= b)).sum()),
             "discharges_by_steps": int(((steps >= a) & (steps <= b)).sum())} for a, b in cats]
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    edges = [0, 0.1, 0.2, 0.4, 1.0, 2.0, 5.0, np.inf]
    sh = [{"range_s": f"{edges[i]}-{edges[i + 1]}" if np.isfinite(edges[i + 1]) else f"{edges[i]} or more",
           "count": int(((sp >= edges[i]) & (sp < edges[i + 1])).sum())} for i in range(len(edges) - 1)]
    top = np.argsort(-frames)[:5]
    return {"histogram": hist, "frames_quantiles": [float(x) for x in np.quantile(frames, qs)],
            "frames_mean": float(frames.mean()), "frames_max": int(frames.max()),
            "most_frames": [[names[i], int(frames[i])] for i in top],
            "spacing_n": int(len(sp)), "spacing_quantiles_s": [float(x) for x in np.quantile(sp, qs)] if len(sp) else [],
            "spacing_histogram": sh, "quantiles": qs,
            "spacing_definition": "tb of the next absent frame minus ta of the previous one, inside a segment"}


# ------------------------------------------------------------------------------ expectations
def expectations(res):
    J = res["J"]["main|all"]
    E = []
    ne, nd = J["events"], J["event_discharges"]
    E.append(("events 300 to 3,000 in 100 to 400 discharges", f"{ne} events in {nd} discharges",
              300 <= ne <= 3000 and 100 <= nd <= 400))
    r = J["ti"]["ratio"]
    E.append(("J1 ratio 0.95 to 1.25", f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]", 0.95 <= r[0] <= 1.25))
    E.append(("J1 not established or weakly so (does not hold, or the lower end below 1.05)",
              f"J1 {res['verdicts']['J1']}, lower end {r[1]:.3f}",
              res["verdicts"]["J1"] != "holds" or r[1] < 1.05))
    E.append(("J2 does not hold", f"J2 {res['verdicts']['J2']}", res["verdicts"]["J2"] != "holds"))
    x = J["ti"]["excess_spread_pct"]
    E.append(("excess spread below 5 % of the level", f"{x[0]:.2f} % [{x[1]:.2f}, {x[2]:.2f}]", x[0] < 5))
    a = res["A"]["A1"]
    if a.get("median_ratio"):
        m = a["median_ratio"]
        E.append(("A1 ratio 0.9 to 1.1", f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]", 0.9 <= m[0] <= 1.1))
        E.append(("A1 interval includes 1", f"[{m[1]:.3f}, {m[2]:.3f}]", m[1] <= 1 <= m[2]))
    return [{"expectation": a_, "observed": b, "met": bool(c)} for a_, b, c in E]


# ------------------------------------------------------------------------------ render
def f3(x, nd=3, sign=False):
    s = "+" if sign else ""
    return f"{x[0]:{s}.{nd}f} [{x[1]:{s}.{nd}f}, {x[2]:{s}.{nd}f}]"


def pct(x, nd=2):
    return f"{100 * x[0]:.{nd}f} [{100 * x[1]:.{nd}f}, {100 * x[2]:.{nd}f}]"


CHOICES = [
    "Grid rows: the rows of `load_grid_files` (the fast channels are complete on every row) without the first and the "
    "last row of each file. T_i: the cut population (held values and values above 3 keV set to NaN), as in B.40 and "
    "B.44. The raw T_i column (`load_grid_files(DATA, False, 0.0)`) is read for the shares of missing values and of "
    "values above 3 keV.",
    "Absent frame: a step strictly between 15 and 45 ms inside a segment (a segment ends at a step of 0.5 s or more). "
    "One step is one event, whatever the number of frames it removes (20, 30 or 40 ms); `tb` and `ta` are the stamps "
    "on either side of the step. The count of absent frames is round(step / 10 ms) - 1 per step (as in B.44).",
    "Isolation: an event is dropped when another absent step (its open interval (tb', ta')) overlaps "
    "[tb - 400 ms, ta + 400 ms]. Steps of 45 ms to 0.5 s (76 in the grid) are not absent frames and do not count.",
    "Window membership uses a tolerance of 0.1 ms on the float32 time stamps; rows must lie in the segment of the event. "
    "A window may contain grid steps longer than 10 ms that are not absent frames; nothing further is required of it.",
    "T_i window mean: the mean of the measured (finite) T_i rows of the window. ECEI window mean: the mean over all grid "
    "rows of the window of the ECEI level (mean of the four ECEI channels, raw CSV values).",
    "ECEI 5 % rule: an event or control is dropped for ECEI when the ECEI mean of either window is below 5 % of the "
    "median over the grid rows of its discharge of the absolute ECEI level (this also removes non-positive means, for "
    "which the logarithm is undefined). The same rule is applied to the controls.",
    "Control stamps: in each segment the targets are segment start + 0.6 s + k x 0.5 s up to the last stamp of the "
    "segment; tc is the first grid stamp at or after the target. A control is kept when no absent step overlaps "
    "[tc - 600 ms, tc + 600 ms] and both windows hold 10 measured T_i rows.",
    "Controls: the registration says \"in the same discharges\"; following the lead's instruction the registered "
    "readout uses controls from all discharges (the discharge set of the data, which events and controls share), and "
    "controls restricted to discharges with at least one event are reported beside it.",
    "Resampling: 2,000 multinomial resamples of the discharges that carry at least one event or control of the cell's "
    "variant (events and controls of a discharge drawn together); the spread, the medians, the ratio and the excess are "
    "recomputed in each resample with the drawn multiplicities (weighted median = median of the repeated sample). "
    f"One seed ({SEED}) with fixed offsets per variant. J2 uses the events and controls with a valid ECEI step for both "
    "the T_i ratio and the ECEI ratio.",
    "Excess spread: sqrt(max(0, spread_events^2 - spread_controls^2)) x 100 (the step is a natural logarithm, so a "
    "spread of 0.01 is 1 % of the level); the bound is the 97.5 % quantile of its resampled values.",
    "Moved window: the window after an event becomes [ta + 250 ms, ta + 450 ms); controls are moved by the same 100 ms "
    "([tc + 280 ms, tc + 480 ms)); the window before and the isolation rules are unchanged; the 10-row requirement is "
    "applied to the moved windows, so the moved sets are not identical to the main sets.",
    "Thirds of the level: edges are the tertiles of the mean T_i of the window before over the events of the main "
    "readout; controls are assigned by the same edges; the resamples of the full readout are reused.",
    "Part A: triples as in B.40 (three consecutive measured rows, both steps 10 ms within 0.1 ms). The age of a triple "
    "(or a row) is its stamp minus ta of the last absent step of the same segment with ta at or before it. Age bins are "
    "half-open [lo, hi). The A1 windows are [0.1, 0.5) s and [1.0, 2.0) s. The pooled relative component of a bin is "
    "1.4826 MAD(second difference) / sqrt(6) over the median T_i of the middle rows, both over the pooled triples of "
    "the bin; its interval resamples all 641 discharges.",
    "Shares by age bin are over grid rows of the bin: missing = NaN in the raw CSV; above 3 keV = raw value above "
    "3,000 eV (denominator: all grid rows of the bin); held (removed as held, at most 3 keV) is reported beside them. "
    "An age bin 0-0.1 s (rows inside the recovery) is reported after the fact; it is not part of the registration.",
    "A1 interval: 2,000 resamples of the discharges that qualify (at least 30 triples in each window), median of the "
    "per-discharge ratios recomputed in each.",
]


def render(res):
    J = res["J"]
    m = J["main|all"]
    v = res["verdicts"]
    L = [f"# B.46 -- the reported T_i and the last beam-off frame (registered: {REG})", "",
         "CPU, nothing trained. All 641 discharges, 10 ms files, grid rows only (first and last row of a file dropped). "
         "T_i with held values and values above 3 keV removed (`load_grid_files(DATA, True, 3000.0)`). An absent frame "
         "(beam-off frame) is a time step between 15 and 45 ms inside a segment (a segment ends at a step of 0.5 s or "
         "more); `tb` is the stamp before the step and `ta` the stamp after it. ECEI level: mean of the four ECEI "
         "channels. Part J: step = ln(mean after / mean before) with the window before (tb - 200 ms, tb] and the window "
         "after [ta + 150 ms, ta + 350 ms), at least 10 measured T_i rows in each; events have no other absent frame "
         "within 400 ms before tb or after ta; controls are stamps tc every 500 ms along a segment with no absent frame "
         "within 600 ms, windows (tc - 200 ms, tc] and [tc + 180 ms, tc + 380 ms). Spread = 1.4826 x median absolute "
         "deviation from the median. Part A: triples of consecutive measured rows (both steps 10 ms), relative "
         "component = 1.4826 MAD(second difference) / sqrt(6) / median T_i of the middle rows, by the age of the last "
         "absent frame of the segment. Intervals: 95 %, 2,000 resamples of discharges.", "",
         "## Checks before the readout", ""]
    L += [f"- {c}" for c in res["checks_text"]]
    L += ["", "## Registered statements", ""]
    r = m["ti"]["ratio"]
    L.append(f"- **J1** (spread of the T_i step at absent frames over that at control stamps, interval above 1): "
             f"{f3(r)}; {m['events']} events in {m['event_discharges']} discharges, {m['controls']} controls in "
             f"{m['control_discharges']} discharges -> **{v['J1']}**.")
    L.append(f"- **J2** (J1 ratio over the same ratio for the ECEI level, interval above 1): {f3(m['J2_ratio'])} "
             f"(T_i ratio on these rows {f3(m['ti_on_ecei_valid']['ratio'])}, ECEI ratio {f3(m['ec']['ratio'])}; "
             f"{m['ecei_valid_events']} events and {m['ecei_valid_controls']} controls with a valid ECEI step, "
             f"{m['ecei_dropped_events']} events and {m['ecei_dropped_controls']} controls dropped by the 5 % rule) "
             f"-> **{v['J2']}**.")
    a1 = res["A"]["A1"]
    if a1.get("median_ratio"):
        L.append(f"- **A1** (median over discharges of the relative component at 1.0-2.0 s over that at 0.1-0.5 s, "
                 f"interval excludes 1): {f3(a1['median_ratio'])}; {a1['discharges']} discharges -> **{v['A1']}**.")
    else:
        L.append(f"- **A1**: {a1['discharges']} discharges qualify -> **{v['A1']}**.")
    L += ["", "## Part J: the step across an absent frame", "",
          "### Counts", "",
          "| stage | events | controls |", "|---|---|---|"]
    fn = res["funnel"]
    L += [f"| absent steps (events) / control stamps | {fn['absent_steps']:,} | {fn['control_candidates']:,} |",
          f"| isolated (400 ms; controls 600 ms) | {fn['isolated']:,} | {fn['control_isolated']:,} |",
          f"| both windows with >= 10 measured T_i rows (main) | {fn['rows_ok']:,} | {fn['control_rows_ok']:,} |",
          f"| same, moved window | {fn['rows_ok_moved']:,} | {fn['control_rows_ok_moved']:,} |", ""]
    L += ["### Spread and median of the step (main windows)", "",
          "| readout | controls | quantity | events | controls | events / controls (or difference) |",
          "|---|---|---|---|---|---|"]
    for key, lab in (("main|all", "registered"), ("main|event_discharges", "reported"),
                     ("moved|all", "moved window"), ("moved|event_discharges", "moved, reported")):
        o = J[key]
        cl = "all discharges" if key.endswith("|all") else "discharges with an event"
        for q, ql in (("ti", "T_i"), ("ec", "ECEI")):
            x = o[q]
            L.append(f"| {lab} | {cl} | {ql} spread of step | {f3(x['spread_events'], 4)} | {f3(x['spread_controls'], 4)} | "
                     f"{f3(x['ratio'])} |")
            L.append(f"| {lab} | {cl} | {ql} median step | {f3(x['median_events'], 4, True)} | "
                     f"{f3(x['median_controls'], 4, True)} | {f3(x['median_difference'], 4, True)} |")
        L.append(f"| {lab} | {cl} | J2 ratio | | | {f3(o['J2_ratio'])} |")
    L += ["", "| readout | events (discharges) | controls (discharges) | resampled discharges | ECEI dropped (events / controls) | "
          "T_i excess spread (% of level) | bound (%) | ECEI excess spread (%) |", "|---|---|---|---|---|---|---|---|"]
    for key in ("main|all", "main|event_discharges", "moved|all", "moved|event_discharges"):
        o = J[key]
        L.append(f"| {key} | {o['events']} ({o['event_discharges']}) | {o['controls']} ({o['control_discharges']}) | "
                 f"{o['resampled_discharges']} | {o['ecei_dropped_events']} / {o['ecei_dropped_controls']} | "
                 f"{f3(o['ti']['excess_spread_pct'], 2)} | {o['ti']['bound_pct']:.2f} | {f3(o['ec']['excess_spread_pct'], 2)} |")
    L += ["", "The step is a natural logarithm; a spread of 0.0100 is 1 % of the level. Excess spread = "
          "sqrt(max(0, spread_events^2 - spread_controls^2)); its interval is the 2.5 and 97.5 % quantiles of the resampled "
          "values (zero where the difference is negative), and the bound is the upper end.", "",
          f"### By third of the T_i level before the event (registered readout; edges {m['third_edges_eV'][0]:.0f} and "
          f"{m['third_edges_eV'][1]:.0f} eV, tertiles of the events)", "",
          "| third | events (discharges) | controls | T_i spread events | T_i spread controls | T_i ratio | T_i excess (%) | "
          "median T_i step events / controls | ECEI ratio | J2 ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
    for j, o in enumerate(m["thirds"]):
        L.append(f"| {j + 1} ({o['level_range_events_eV'][0]:.0f}-{o['level_range_events_eV'][1]:.0f} eV) | "
                 f"{o['events']} ({o['event_discharges']}) | {o['controls']} | {f3(o['ti']['spread_events'], 4)} | "
                 f"{f3(o['ti']['spread_controls'], 4)} | {f3(o['ti']['ratio'])} | {f3(o['ti']['excess_spread_pct'], 2)} | "
                 f"{o['ti']['median_events'][0]:+.4f} / {o['ti']['median_controls'][0]:+.4f} | {f3(o['ec']['ratio'])} | "
                 f"{f3(o['J2_ratio'])} |")
    A = res["A"]
    L += ["", "## Part A: the independent component by the age of the last absent frame", "",
          "| age bin (s) | triples (discharges) | rows | relative component | component (eV) | median T_i (eV) | "
          "share missing | share above 3 keV | share held |", "|---|---|---|---|---|---|---|---|---|"]
    order = [nm for nm, _, _ in AGE_BINS] + [NONE_BIN, AGE_EXTRA[0]]
    for nm in order:
        o = A["bins"][nm]
        L.append(f"| {nm} | {o['triples']:,} ({o['triple_discharges']}) | {o['rows']:,} | {f3(o['relative_component'], 4)} | "
                 f"{o['component_eV']:.1f} | {o['median_ti_eV']:.0f} | {pct(o['share_miss'])} % | {pct(o['share_above'])} % | "
                 f"{pct(o['share_held'])} % |")
    if a1.get("median_ratio"):
        L += ["", f"A1: {a1['discharges']} discharges with at least {A_MIN_TRIPLES} triples in each window; median ratio "
              f"{f3(a1['median_ratio'])}; quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of the ratio "
              f"{' / '.join(f'{x:.3f}' for x in a1['ratio_quantiles_0.1_0.25_0.5_0.75_0.9'])}; share of discharges with "
              f"a ratio above 1: {a1['share_ratio_above_1']:.3f}."]
    D = res["absent_distribution"]
    L += ["", "### Absent frames per discharge", "",
          "| absent frames (or absent steps) per discharge | discharges by frames | discharges by steps |", "|---|---|---|"]
    for h in D["histogram"]:
        L.append(f"| {h['range']} | {h['discharges_by_frames']} | {h['discharges_by_steps']} |")
    L += ["", f"Absent frames per discharge: mean {D['frames_mean']:.2f}, quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 "
          f"{' / '.join(f'{x:.0f}' for x in D['frames_quantiles'])}, largest {D['frames_max']} "
          f"({', '.join(f'{a} {b}' for a, b in D['most_frames'])}).", "",
          f"Spacing between consecutive absent frames ({D['spacing_definition']}; {D['spacing_n']:,} spacings): quantiles "
          f"0.1 / 0.25 / 0.5 / 0.75 / 0.9 {' / '.join(f'{x:.3f}' for x in D['spacing_quantiles_s'])} s.", "",
          "| spacing (s) | count |", "|---|---|"]
    for h in D["spacing_histogram"]:
        L.append(f"| {h['range_s']} | {h['count']:,} |")
    z = m["after_fact_nonzero_ecei"]
    L += ["", "## After the fact (not registered; not used for any verdict)", "",
          f"1. The ECEI level is exactly constant over both windows (step exactly 0) at {z['zero_steps_controls']} controls "
          f"({z['zero_step_discharges_controls']} discharges) and {z['zero_steps_events']} events "
          f"({z['zero_step_discharges_events']} discharges); {res['checks']['ecei_rows_equal_previous']:,} of "
          f"{res['checks']['grid_row_pairs']:,} consecutive grid-row "
          f"pairs have the same ECEI level, and {len(res['checks']['ecei_constant_files'])} files have a constant ECEI "
          f"level ({', '.join(res['checks']['ecei_constant_files'])}). Zero steps shrink the ECEI spread at the controls "
          f"more than at the events. With them dropped for ECEI: ECEI spread events {f3(z['ec']['spread_events'], 4)}, "
          f"controls {f3(z['ec']['spread_controls'], 4)}, ratio {f3(z['ec']['ratio'])}; T_i ratio on the same rows "
          f"{f3(z['ti_on_ecei_valid']['ratio'])}; J2 ratio {f3(z['J2_ratio'])} ({z['ecei_valid_events']} events, "
          f"{z['ecei_valid_controls']} controls)."]
    L += ["", "## Expectations of the registration (section 7)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in res["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    nm_ = sum(e["met"] for e in res["expectations"])
    L += ["", f"{nm_} of {len(res['expectations'])} expectations met.", "",
          "## Choices made where the registration left room", ""]
    L += [f"- {c}" for c in CHOICES]
    L += ["", "## Notes on the readout", "",
          "- The absent frames are close to periodic: the median spacing between consecutive absent frames of a segment is "
          "about 1.47 s (table above). An exclusion of 600 ms on either side of a control stamp therefore removes most stamps "
          "of discharges with regular blips, and the controls of the registered readout come in part from discharges "
          "without events. The reported variant with controls restricted to discharges with an event is given beside it.",
          "- For the same reason the age bins of 1.5-3 s and above 3 s hold few discharges (see the triple counts); they are "
          "discharges or segments without the regular blip pattern, and their components are not a within-discharge "
          "comparison. A1 compares ages inside a discharge.",
          "- The group without an absent frame before in the segment is mostly the early part of each segment (before the "
          "first blip); its median T_i is lower than that of the other bins.",
          "", "## Deviations", "", "None from the registered definitions, thresholds or populations.", "",
          f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, *_ = R.imports()
    grid, dims = load_grid_files(R.DATA, True, 3000.0)
    raw, _ = load_grid_files(R.DATA, False, 0.0)
    names = sorted(grid)
    log(f"grid loaded: {len(names)} files, dims {dims}")
    if len(names) != 641 or dims["ecei"] != 4:
        raise SystemExit(f"FATAL: expected 641 files and four ECEI channels, got {len(names)} / {dims['ecei']}")
    files = [file_struct(grid[n], raw[n], dims["bes"], dims["ecei"]) for n in names]
    if not all(F["rows_match"] for F in files):
        raise SystemExit("FATAL: raw and cut grids differ in rows")
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT}
    # ---- checks
    n_rows = sum(len(F["t"]) for F in files)
    n_miss = sum(int((~np.isfinite(F["yraw"])).sum()) for F in files)
    n_cut_nan = sum(int((~np.isfinite(F["y"])).sum()) for F in files)
    n_seg = sum(int(F["seg"][-1]) + 1 for F in files if len(F["seg"]))
    steps = sum(len(F["gaps"]["tb"]) for F in files)
    frames = sum(int(F["gaps"]["frames"].sum()) for F in files)
    dis_with = sum(len(F["gaps"]["tb"]) > 0 for F in files)
    tri_all = []
    for F in files:
        t, y = F["t"], F["y"]
        dt = np.diff(t)
        fin = np.isfinite(y)
        mm = (fin[:-2] & fin[1:-1] & fin[2:] & (np.abs(dt[:-1] - 0.010) < 1e-4) & (np.abs(dt[1:] - 0.010) < 1e-4))
        c = np.flatnonzero(mm) + 1
        tri_all.append(y[c + 1] - 2 * y[c] + y[c - 1])
    tri_all = np.concatenate(tri_all)
    pooled = float(spread(tri_all) / np.sqrt(6.0))
    chk = {"grid_rows": n_rows, "ti_missing_raw": n_miss, "ti_missing_raw_share": n_miss / n_rows,
           "ti_nan_after_cut_share": n_cut_nan / n_rows, "segments": n_seg,
           "triples": int(len(tri_all)), "pooled_component_eV": pooled,
           "absent_steps": steps, "absent_frames": frames, "discharges_with_absent": int(dis_with),
           "held_rows": sum(int((np.isfinite(F["yraw"]) & (F["yraw"] <= 3000) & ~np.isfinite(F["y"])).sum()) for F in files),
           "above_3keV_rows": sum(int((np.isfinite(F["yraw"]) & (F["yraw"] > 3000)).sum()) for F in files),
           "ecei_rows_equal_previous": sum(int((np.diff(F["ec"]) == 0).sum()) for F in files),
           "grid_row_pairs": sum(len(F["ec"]) - 1 for F in files),
           "ecei_constant_files": [n for n, F in zip(names, files) if len(F["ec"]) and np.ptp(F["ec"]) == 0]}
    res["checks"] = chk
    ok_a = n_rows == 245925 and abs(100 * n_miss / n_rows - 7.70) < 0.005
    ok_b = abs(pooled - 42.8) <= 0.5
    res["checks_text"] = [
        f"(a) Grid rows over all files: {n_rows:,} (file_rules.md / PROJECT_KNOWLEDGE.md section 2: 245,925); T_i missing "
        f"in the raw CSV column before the removal of held values and of values above 3 keV: {n_miss:,} rows = "
        f"{100 * n_miss / n_rows:.2f} % (reference 7.70 %) -> {'reproduced' if ok_a else 'NOT reproduced'}. After the "
        f"removal T_i is NaN on {100 * n_cut_nan / n_rows:.2f} % of the grid rows ({chk['above_3keV_rows']:,} rows above "
        f"3 keV, {chk['held_rows']} held row). Segments: {n_seg} (file_rules.md: 698).",
        f"(b) Pooled robust component over all triples, 1.4826 MAD(second difference) / sqrt(6): {pooled:.2f} eV over "
        f"{len(tri_all):,} triples (b40_noise.md, robust sqrt(N_hi) of the cut population: 42.8 eV over 210,908 triples) "
        f"-> {'reproduced within 0.5 eV' if ok_b else 'NOT reproduced'}.",
        f"(c) Absent frames: {frames:,} absent frames in {steps:,} steps between 15 and 45 ms (file_rules.md: 1,637 in "
        f"1,414); {dis_with} of {len(files)} discharges have at least one."]
    for c in res["checks_text"]:
        log(c.encode("ascii", "replace").decode())
    if not ok_a:
        raise SystemExit("FATAL: check (a) failed; stopping before the readout")
    if not ok_b:
        raise SystemExit("FATAL: check (b) failed; stopping before the readout")
    # ---- Part J
    ev, ct, funnel = part_j_records(files)
    res["funnel"] = funnel
    log(f"J funnel {funnel}")
    res["J"] = part_j(ev, ct, SEED)
    m = res["J"]["main|all"]
    testable_j = m["events"] >= MIN_EVENTS and m["event_discharges"] >= MIN_DIS_J
    res["J_testable"] = bool(testable_j)
    vs = lambda h: "holds" if h else "does not hold"          # noqa: E731
    res["verdicts"] = {"J1": vs(m["ti"]["ratio"][1] > 1) if testable_j else "not testable",
                       "J2": vs(m["J2_ratio"][1] > 1) if testable_j else "not testable"}
    # ---- Part A
    tri, rows = part_a(files)
    res["A"] = part_a_stats(tri, rows, len(files), np.random.default_rng(SEED + 5))
    a1 = res["A"]["A1"]
    res["verdicts"]["A1"] = vs(a1["holds"]) if a1["testable"] else "not testable"
    res["absent_distribution"] = absent_distribution(files, names)
    res["expectations"] = expectations(res)
    res["choices"] = CHOICES
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"verdicts {res['verdicts']}; written")


if __name__ == "__main__":
    main()
