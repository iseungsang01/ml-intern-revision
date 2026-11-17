"""B.41 (PREREGISTRATION_B41.md): where the gain sits.

G  share of the squared-error reduction on rows that pass the four value rules (plain) and on the rest
Q  distribution of the absolute error (percentiles, signed error by third of hold, calibration slope)
U  T_i, ECEI and BES around edge-burst onsets (twelve raw-stream discharges) and the estimators there
B  the estimators in the rows after a beam blip (test discharges of the four splits)

CPU only; nothing is trained.  Reads the frozen B.30 row tables (data/.b30/rows_cut_s*.npz), the shot
files and the raw BES stream.  Run from the repo root:
    py ces_prediction/experiments/b41_events/run_b41.py
Output: docs/paper/b41_events.json / .md and the figure data docs/paper/b41_events_fig.json
(the figure itself: fig_events.py).
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b31_bursts"))
import run_b30 as R  # noqa: E402
from run_b31 import bursts_and_blips, SHOTS as RAW_SHOTS  # noqa: E402
from sensitivity_b30 import LOW_EV, JUMP_EV, BLIP_S  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b41_events.json"
OUT_MD = R.REPO / "docs" / "paper" / "b41_events.md"
OUT_FIG = R.REPO / "docs" / "paper" / "b41_events_fig.json"
HIGH_EV = 2500.0
EST = ("hold", "gp", "hist80", "causal")
LABEL = {"hold": "hold", "gp": "causal GP", "hist80": "twin", "causal": "causal variant"}
PCTS = (50, 75, 90, 95, 99)
CHUNK = 100
# U1
ISO_S, ABS_AFTER_S = 0.050, 0.100
PRE_LO, PRE_HI, POST_HI = -0.050, -0.010, 0.030
BINS_MS = np.arange(-50, 100, 10)          # left edges, 15 bins of 10 ms
MIN_EVENTS = 30
# blips (B.35 event rule)
KB, PRE_ROWS = 15, 5
EPS = 1e-6


def boot_w(rng, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=R.N_BOOT).astype(np.float64)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def wquant(v, f, W, qs):
    """Weighted quantiles of v (n,) under discharge weights W (reps, n_files); f = file index of each row.
    Returns (reps, len(qs))."""
    o = np.argsort(v, kind="stable")
    vs, fo = v[o], f[o]
    out = np.empty((W.shape[0], len(qs)))
    for a in range(0, W.shape[0], CHUNK):
        c = np.cumsum(W[a:a + CHUNK][:, fo], 1)
        for j, q in enumerate(qs):
            out[a:a + CHUNK, j] = vs[(c >= q * c[:, -1:]).argmax(1)]
    return out


def pin_ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


# ------------------------------------------------------------------------------------------ setup
class Ctx:
    def __init__(self):
        R.pin_env()
        _t, _B, _ls, load_grid_files, _bb, self.GAP, _m = R.imports()
        self.grid, self.dims = load_grid_files(R.DATA, True, R.POPS["cut"])
        self.names = sorted(self.grid)
        nb, ne = self.dims["bes"], self.dims["ecei"]
        self.t, self.ti, self.bes, self.ecei = {}, {}, {}, {}
        self.after_low_bes = {}
        for i, nm in enumerate(self.names):
            arr = self.grid[nm]
            t = arr[:, 0].astype(np.float64)
            self.t[i], self.ti[i] = t, arr[:, 1].astype(np.float64)
            self.bes[i] = arr[:, 3:3 + nb].astype(np.float64).mean(1)
            self.ecei[i] = arr[:, 3 + nb:3 + nb + ne].astype(np.float64).mean(1)
            # sensitivity_b30.py, rule (d), verbatim
            b = self.bes[i]
            low = np.flatnonzero(b < 0.5 * np.median(b))
            flag = np.zeros(len(t), bool)
            for j in low:
                flag |= (t >= t[j]) & (t <= t[j] + BLIP_S)
            self.after_low_bes[i] = flag

    def manifest(self, s):
        return json.loads((R.DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json").read_text(encoding="utf-8"))

    def table(self, s):
        z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
        return {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}


def plain_mask(C, tab):
    y, h = tab["y"].astype(np.float64), tab["hold"].astype(np.float64)
    blip = np.array([C.after_low_bes[int(f)][int(r)] for f, r in zip(tab["file"], tab["row"])])
    return ((y <= HIGH_EV) & (h <= HIGH_EV) & (y >= LOW_EV) & (h >= LOW_EV)
            & (np.abs(y - h) <= JUMP_EV) & ~blip)


def population(tab):
    return (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0)


# ------------------------------------------------------------------------------------------ check
def precheck(C):
    """Plain share on the population of sensitivity_b30.py (has_future, pchip and every network finite)."""
    out = {}
    for s in R.SEEDS:
        tab = C.table(s)
        ests = ["hold", "gp", "pchip", *R.NETS["cut"]]
        fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0) & (tab["k"] == 1)
        pm = plain_mask(C, tab)
        pop = population(tab)
        out[str(s)] = {"sensitivity_population_rows": int(fin.sum()),
                       "plain_share_sensitivity_population": float((fin & pm).sum() / fin.sum()),
                       "b41_population_rows": int(pop.sum()),
                       "plain_share_b41_population": float((pop & pm).sum() / pop.sum())}
        print("check", s, out[str(s)], flush=True)
    return out


# ------------------------------------------------------------------------------------------ G and Q
def part_gq(C, rng):
    res = {}
    for s in R.SEEDS:
        tab = C.table(s)
        ok = population(tab)
        plain = plain_mask(C, tab)[ok]
        uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
        n = len(uf)
        W = boot_w(rng, n)
        y = tab["y"][ok].astype(np.float64)
        v = {e: tab[e][ok].astype(np.float64) for e in EST}
        se = {e: (v[e] - y) ** 2 for e in EST}
        ae = {e: np.abs(v[e] - y) for e in EST}
        fs = lambda x: np.bincount(f, x, n)                                     # noqa: E731

        def share(mask, ref):
            d = se[ref] - se["causal"]
            a, t = fs(d * mask), fs(d)
            return [float(a.sum() / t.sum()), *ci((W @ a) / (W @ t))]

        def skill(mask, a, b):
            sa, sb = fs(se[a] * mask), fs(se[b] * mask)
            return [float(1 - sa.sum() / sb.sum()), *ci(1 - (W @ sa) / (W @ sb))]

        top = se["hold"] >= np.quantile(se["hold"], 0.99)
        o = {"rows": int(ok.sum()), "discharges": n}
        for cname, m in (("plain", plain), ("flagged", ~plain), ("top1_hold", top), ("rest_hold", ~top)):
            mf = m.astype(np.float64)
            o[cname] = {"rows": int(m.sum()),
                        "row_share": [float(m.mean()), *ci((W @ fs(mf)) / (W @ fs(np.ones(len(y)))))],
                        "share_of_reduction_vs_twin": share(mf, "hist80"),
                        "share_of_reduction_vs_gp": share(mf, "gp"),
                        "skill_causal_vs_twin": skill(mf, "causal", "hist80"),
                        "skill_causal_vs_gp": skill(mf, "causal", "gp"),
                        "skill_causal_vs_hold": skill(mf, "causal", "hold")}
        o["total_reduction_vs_twin_eV2_per_row"] = float((se["hist80"] - se["causal"]).mean())
        o["total_reduction_vs_gp_eV2_per_row"] = float((se["gp"] - se["causal"]).mean())
        o["G1_holds"] = bool(o["plain"]["share_of_reduction_vs_twin"][1] > 0.5)
        o["G2_holds"] = bool(o["plain"]["share_of_reduction_vs_gp"][1] > 0.5)
        # Q: percentiles
        qs = [p / 100 for p in PCTS]
        pt = {e: np.percentile(ae[e], PCTS) for e in EST}
        bq = {e: wquant(ae[e], f, W, qs) for e in EST}
        o["abs_error_percentiles_eV"] = {e: {str(p): float(x) for p, x in zip(PCTS, pt[e])} for e in EST}
        o["ratio_causal_over"] = {}
        for ref in ("hold", "gp", "hist80"):
            o["ratio_causal_over"][ref] = {str(p): [float(pt["causal"][j] / pt[ref][j]),
                                                    *ci(bq["causal"][:, j] / bq[ref][:, j])]
                                           for j, p in enumerate(PCTS)}
        j90 = PCTS.index(90)
        o["Q1_p90_ratio_vs_gp"] = o["ratio_causal_over"]["gp"]["90"]
        o["Q1_p90_ratio_vs_hold"] = o["ratio_causal_over"]["hold"]["90"]
        o["Q1_holds"] = bool(o["Q1_p90_ratio_vs_gp"][2] < 1 and o["Q1_p90_ratio_vs_hold"][2] < 1)
        assert np.isclose(o["Q1_p90_ratio_vs_gp"][0], pt["causal"][j90] / pt["gp"][j90])
        # signed error by third of the last visible sample
        h = v["hold"]
        t1, t2 = np.quantile(h, [1 / 3, 2 / 3])
        thirds = {"low": h <= t1, "middle": (h > t1) & (h <= t2), "high": h > t2}
        o["hold_thirds_eV"] = [float(t1), float(t2)]
        o["mean_signed_error_eV"] = {}
        for tn, tm in thirds.items():
            cnt = fs(tm.astype(np.float64))
            o["mean_signed_error_eV"][tn] = {
                e: [float((v[e] - y)[tm].mean()), *ci((W @ fs((v[e] - y) * tm)) / (W @ cnt))] for e in EST}
        # slope of the least-squares line of the measurement on each estimate
        o["slope_measurement_on_estimate"] = {}
        for e in EST:
            x = v[e]
            S = [fs(np.ones(len(y))), fs(x), fs(y), fs(x * x), fs(x * y)]
            pts = [a.sum() for a in S]
            bs = [W @ a for a in S]

            def slope(N, sx, sy, sxx, sxy):
                return (sxy / N - sx / N * sy / N) / (sxx / N - (sx / N) ** 2)
            o["slope_measurement_on_estimate"][e] = [float(slope(*pts)), *ci(slope(*bs))]
        res[str(s)] = o
        print("GQ", s, "G1", o["G1_holds"], o["plain"]["share_of_reduction_vs_twin"], "G2", o["G2_holds"],
              o["plain"]["share_of_reduction_vs_gp"], "Q1", o["Q1_holds"], flush=True)
    return res


# ------------------------------------------------------------------------------------------ U
def raw_assign(C):
    """Raw-stream discharges that are test discharges: the first split whose test files hold them
    (b37_plan/run_b37.py::part_s)."""
    assign = {}
    for s in R.SEEDS:
        tf = C.manifest(s)["test_files"]
        for sh in RAW_SHOTS:
            if sh not in assign and f"s{sh}.csv" in tf:
                assign[sh] = s
    return assign


def burst_events(C):
    ev = []
    info = {}
    for sh in RAW_SHOTS:
        i = C.names.index(f"s{sh}.csv")
        t, ti = C.t[i], C.ti[i]
        raw = bursts_and_blips(sh)
        on = np.asarray(raw["bursts"], float)
        dt = np.diff(t)
        g = np.flatnonzero((dt > 0.015) & (dt < 0.045))
        n_iso = n_abs = n_ti = 0
        for k, o in enumerate(on):
            if k > 0 and o - on[k - 1] <= ISO_S:
                continue
            n_iso += 1
            # an absent frame lies between t[j] and t[j+1]; its last stamp is t[j+1] - 10 ms
            if np.any((o > t[g]) & (o <= t[g + 1] - 0.010 + ABS_AFTER_S + EPS)):
                continue
            n_abs += 1
            rel = t - o
            pre = (rel >= PRE_LO - EPS) & (rel <= PRE_HI + EPS)
            post = (rel >= -EPS) & (rel <= POST_HI + EPS)
            if np.isfinite(ti[pre]).sum() < 2 or np.isfinite(ti[post]).sum() < 1:
                continue
            n_ti += 1
            ev.append({"shot": sh, "file": i, "onset": float(o),
                       "ti_pre": float(np.nanmean(ti[pre])), "ti_post": float(np.nanmean(ti[post])),
                       "ecei_pre": float(C.ecei[i][pre].mean()),
                       "ecei_post": float(C.ecei[i][post].mean()),
                       "bes_pre": float(C.bes[i][pre].mean()), "bes_post": float(C.bes[i][post].mean())})
        info[str(sh)] = {"onsets": int(len(on)), "isolated": n_iso, "outside_absent_frames": n_abs,
                         "with_ti_rows": n_ti}
        print("bursts", sh, info[str(sh)], flush=True)
    return ev, info


def part_u(C, rng):
    ev, info = burst_events(C)
    shots = np.array([e["shot"] for e in ev])
    us = sorted(set(shots.tolist()))
    by = {sh: np.flatnonzero(shots == sh) for sh in us}
    reps = [rng.integers(0, len(us), len(us)) for _ in range(R.N_BOOT)]
    res = {"per_discharge": info, "events": len(ev), "discharges_with_events": len(us),
           "events_per_discharge": {str(sh): int(len(by[sh])) for sh in us}}

    def med_ratio(a, b):
        r = np.array([e[a] / e[b] for e in ev])
        p = float(np.median(r))
        bs = [np.median(r[np.concatenate([by[us[k]] for k in rr])]) for rr in reps]
        return [p, *ci(bs)], r

    for q in ("ti", "ecei", "bes"):
        res[f"median_ratio_{q}"], r = med_ratio(f"{q}_post", f"{q}_pre")
        res[f"mean_ratio_{q}"] = float(r.mean())
        res[f"share_below_1_{q}"] = float((r < 1).mean())
    res["after_the_fact_level_before_quantiles"] = {
        "q": [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
        **{q: [float(x) for x in np.quantile([e[f"{q}_pre"] for e in ev], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])]
           for q in ("ti", "ecei", "bes")},
        "ti_ratio": [float(x) for x in np.quantile([e["ti_post"] / e["ti_pre"] for e in ev],
                                                   [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])]}
    res["U1_testable"] = bool(len(ev) >= MIN_EVENTS)
    res["U1_holds"] = bool(res["U1_testable"] and res["median_ratio_ti"][2] < 1)

    # traces by 10 ms bin, ratio to the event's level 50 to 10 ms before
    nbin = len(BINS_MS)
    W = np.array([np.bincount(rr, minlength=len(us)) for rr in reps], float)    # (reps, discharges)
    ui = {sh: k for k, sh in enumerate(us)}

    # after the fact: median traces (own generator, so the registered numbers above are unchanged)
    rng_med = np.random.default_rng(R.BOOT_SEED + 3)

    def med_trace(vals, nunit):
        """vals[b] = list of (unit, value); median over events with a discharge-resample interval."""
        reps_m = [rng_med.integers(0, nunit, nunit) for _ in range(R.N_BOOT)]
        out = {"median": [], "lo": [], "hi": []}
        for b in range(nbin):
            u = np.array([a for a, _ in vals[b]], int)
            x = np.array([v for _, v in vals[b]], float)
            idx = [np.flatnonzero(u == k) for k in range(nunit)]
            bs = [np.median(x[np.concatenate([idx[k] for k in rr])]) if len(x) else np.nan for rr in reps_m]
            out["median"].append(float(np.median(x)) if len(x) else np.nan)
            lo, hi = ci(bs) if len(x) else (np.nan, np.nan)
            out["lo"].append(lo)
            out["hi"].append(hi)
        return out

    def trace(vals_fn):
        S = np.zeros((len(us), nbin))
        N = np.zeros((len(us), nbin))
        V = [[] for _ in range(nbin)]
        for e in ev:
            i = e["file"]
            rel_ms = (C.t[i] - e["onset"]) * 1000.0
            for b, lo in enumerate(BINS_MS):
                m = (rel_ms >= lo - 1e-3) & (rel_ms < lo + 10 - 1e-3)
                x = vals_fn(e, i, m)
                x = x[np.isfinite(x)]
                if x.size:
                    S[ui[e["shot"]], b] += x.mean()
                    N[ui[e["shot"]], b] += 1
                    V[b].append((ui[e["shot"]], float(x.mean())))
        p = S.sum(0) / np.maximum(N.sum(0), 1)
        bs = pin_ratio(W @ S, W @ N)
        return {"mean": [float(x) for x in p], "lo": [float(np.nanquantile(bs[:, b], 0.025)) for b in range(nbin)],
                "hi": [float(np.nanquantile(bs[:, b], 0.975)) for b in range(nbin)],
                "events": [int(x) for x in N.sum(0)], "after_the_fact_median": med_trace(V, len(us))}

    tr = {"bins_ms": [int(x) for x in BINS_MS],
          "T_i": trace(lambda e, i, m: C.ti[i][m] / e["ti_pre"]),
          "ECEI": trace(lambda e, i, m: C.ecei[i][m] / e["ecei_pre"]),
          "BES": trace(lambda e, i, m: C.bes[i][m] / e["bes_pre"])}

    # estimators on the raw-stream discharges that are test discharges
    assign = raw_assign(C)
    tabs = {s: C.table(s) for s in sorted(set(assign.values()))}
    lut = {}
    for sh, s in assign.items():
        tab = tabs[s]
        i = C.names.index(f"s{sh}.csv")
        m = population(tab) & (tab["file"] == i)
        lut[sh] = {int(r): (float(y), float(c), float(h), float(hh), float(g))
                   for r, y, c, h, hh, g in zip(tab["row"][m], tab["y"][m], tab["causal"][m], tab["hist80"][m],
                                                tab["hold"][m], tab["gp"][m])}
    evt = [e for e in ev if e["shot"] in assign]
    ts = sorted({e["shot"] for e in evt})

    def est_trace(col):
        S = np.zeros((len(ts), nbin))
        N = np.zeros((len(ts), nbin))
        V = [[] for _ in range(nbin)]
        for e in evt:
            i, L = e["file"], lut[e["shot"]]
            rel_ms = (C.t[i] - e["onset"]) * 1000.0
            for b, lo in enumerate(BINS_MS):
                rows = np.flatnonzero((rel_ms >= lo - 1e-3) & (rel_ms < lo + 10 - 1e-3))
                x = [L[r][col] / e["ti_pre"] for r in rows if r in L]
                if x:
                    S[ts.index(e["shot"]), b] += np.mean(x)
                    N[ts.index(e["shot"]), b] += 1
                    V[b].append((ts.index(e["shot"]), float(np.mean(x))))
        Wt = boot_w(rng, len(ts))
        p = S.sum(0) / np.maximum(N.sum(0), 1)
        bs = pin_ratio(Wt @ S, Wt @ N)
        return {"mean": [float(x) for x in p], "lo": [float(np.nanquantile(bs[:, b], 0.025)) for b in range(nbin)],
                "hi": [float(np.nanquantile(bs[:, b], 0.975)) for b in range(nbin)],
                "events": [int(x) for x in N.sum(0)], "after_the_fact_median": med_trace(V, len(ts))}

    tr_test = {"measured_same_rows": est_trace(0), "causal": est_trace(1), "twin": est_trace(2)}
    # squared errors in the rows 0 to 30 ms after an onset (each row once)
    sse = np.zeros((len(ts), 4))
    nrow = np.zeros(len(ts))
    for sh in ts:
        i = C.names.index(f"s{sh}.csv")
        rows = set()
        for e in evt:
            if e["shot"] != sh:
                continue
            rel = C.t[i] - e["onset"]
            rows |= set(np.flatnonzero((rel >= -EPS) & (rel <= POST_HI + EPS)).tolist())
        L = lut[sh]
        for r in rows:
            if r in L:
                y, c, h, hh, g = L[r]
                k = ts.index(sh)
                sse[k] += [(c - y) ** 2, (h - y) ** 2, (hh - y) ** 2, (g - y) ** 2]
                nrow[k] += 1
    Wt = boot_w(rng, len(ts))
    bs = Wt @ sse
    post = {"test_discharges": {str(sh): int(assign[sh]) for sh in ts},
            "events": len(evt), "rows": int(nrow.sum()),
            "rmse_eV": {k: float(np.sqrt(sse[:, j].sum() / nrow.sum())) for j, k in
                        enumerate(("causal", "twin", "hold", "gp"))},
            "skill_causal_vs_twin": [float(1 - sse[:, 0].sum() / sse[:, 1].sum()), *ci(1 - bs[:, 0] / bs[:, 1])],
            "per_discharge_rows": {str(sh): int(nrow[k]) for k, sh in enumerate(ts)}}
    res["test_discharge_estimates"] = post
    res["traces"] = tr
    res["traces_test_discharges"] = tr_test
    res["events_list"] = [{k: e[k] for k in ("shot", "onset", "ti_pre", "ti_post")} for e in ev]
    print("U1", res["events"], res["median_ratio_ti"], "holds", res["U1_holds"], flush=True)
    return res


# ------------------------------------------------------------------------------------------ blips
def part_b(C, rng):
    """Rows 1 to 15 after an absent frame (B.35 event rule), test discharges, frozen row tables."""
    ks = np.arange(-PRE_ROWS, KB + 1)                  # -5..0 = the six rows before; 1..15 after
    nk = len(ks)
    cols = ("y", "causal", "hist80", "hold", "gp")
    res = {"rows_k": [int(k) for k in ks]}
    pooled = {c: ([], []) for c in (*cols, "ECEI", "BES")}
    for s in R.SEEDS:
        tab = C.table(s)
        ok = population(tab)
        files = np.unique(tab["file"][ok].astype(int))
        S = {c: np.zeros((len(files), nk)) for c in (*cols, "ECEI", "BES")}
        N = {c: np.zeros((len(files), nk)) for c in (*cols, "ECEI", "BES")}
        sse = {g: np.zeros((len(files), 2)) for g in ("1-3", "4-10")}
        n_ev = 0
        for fi, i in enumerate(files):
            m = ok & (tab["file"] == i)
            L = {int(r): j for j, r in zip(np.flatnonzero(m), tab["row"][m])}
            t, ti = C.t[i], C.ti[i]
            dt = np.diff(t)
            n = len(t)
            seen = {"1-3": set(), "4-10": set()}
            for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
                if j < PRE_ROWS or not np.all(np.abs(dt[j - PRE_ROWS:j] - 0.010) < 1e-4):
                    continue
                pre = ti[j - PRE_ROWS:j + 1]
                pre = pre[np.isfinite(pre)]
                if len(pre) < 3:
                    continue
                lvl = float(np.median(pre))
                elvl = float(np.median(C.ecei[i][j - PRE_ROWS:j + 1]))
                blvl = float(np.median(C.bes[i][j - PRE_ROWS:j + 1]))
                n_ev += 1
                for a, k in enumerate(ks):
                    r = j + k
                    if k >= 1:
                        if r >= n or (k > 1 and abs(dt[r - 1] - 0.010) > 1e-4):
                            break
                    S["ECEI"][fi, a] += C.ecei[i][r] / elvl
                    N["ECEI"][fi, a] += 1
                    S["BES"][fi, a] += C.bes[i][r] / blvl
                    N["BES"][fi, a] += 1
                    q = L.get(r)
                    if q is None:
                        continue
                    for c in cols:
                        S[c][fi, a] += float(tab[c][q]) / lvl
                        N[c][fi, a] += 1
                    for g, (lo, hi) in (("1-3", (1, 3)), ("4-10", (4, 10))):
                        if lo <= k <= hi and r not in seen[g]:
                            seen[g].add(r)
                            yv = float(tab["y"][q])
                            sse[g][fi] += [(float(tab["causal"][q]) - yv) ** 2, (float(tab["hist80"][q]) - yv) ** 2]
        W = boot_w(rng, len(files))
        o = {"discharges": int(len(files)), "events": n_ev, "traces": {}}
        for c in (*cols, "ECEI", "BES"):
            p = S[c].sum(0) / np.maximum(N[c].sum(0), 1)
            bs = pin_ratio(W @ S[c], W @ N[c])
            o["traces"][c] = {"mean": [float(x) for x in p],
                              "lo": [float(np.nanquantile(bs[:, a], 0.025)) for a in range(nk)],
                              "hi": [float(np.nanquantile(bs[:, a], 0.975)) for a in range(nk)],
                              "rows": [int(x) for x in N[c].sum(0)]}
            pooled[c][0].append(S[c])
            pooled[c][1].append(N[c])
        for g in sse:
            bs = W @ sse[g]
            o[f"skill_causal_vs_twin_rows_{g}"] = [float(1 - sse[g][:, 0].sum() / sse[g][:, 1].sum()),
                                                   *ci(1 - bs[:, 0] / bs[:, 1])]
        o["rows_1-3"] = int(N["y"][:, PRE_ROWS + 1:PRE_ROWS + 4].sum())
        o["rows_4-10"] = int(N["y"][:, PRE_ROWS + 4:PRE_ROWS + 11].sum())
        res[str(s)] = o
        print("blip", s, n_ev, o["skill_causal_vs_twin_rows_1-3"], o["skill_causal_vs_twin_rows_4-10"], flush=True)
    # pooled over the four splits for the figure (unit = discharge within split)
    fig = {}
    nunit = sum(len(x) for x in pooled["y"][0])
    W = boot_w(rng, nunit)
    for c in pooled:
        S, N = np.vstack(pooled[c][0]), np.vstack(pooled[c][1])
        p = S.sum(0) / np.maximum(N.sum(0), 1)
        bs = pin_ratio(W @ S, W @ N)
        fig[c] = {"mean": [float(x) for x in p], "lo": [float(np.nanquantile(bs[:, a], 0.025)) for a in range(nk)],
                  "hi": [float(np.nanquantile(bs[:, a], 0.975)) for a in range(nk)],
                  "rows": [int(x) for x in N.sum(0)]}
    res["pooled_for_figure"] = fig
    return res


# ------------------------------------------------------------------------------------------ render
def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def s1(x):
    return f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"


def render(res):
    G, U, B = res["G"], res["U"], res["B"]
    S = R.SEEDS
    n = lambda k: sum(G[str(s)][k] for s in S)                                   # noqa: E731
    v = lambda c: "holds" if c else "does not hold"                              # noqa: E731
    L = ["# B.41 — where the gain sits (registered: PREREGISTRATION_B41.md)", "",
         "CPU, nothing trained. T_i, cut population, natural cadence (k = 1), frozen B.30 row tables; rows with hold, "
         "the causal GP, the history-only twin, the causal variant and the measurement finite. Intervals: 95 %, "
         "2,000 resamples of test discharges; a statement holds if it holds on at least three of the four splits. "
         "Plain = passes all four exclusions of sensitivity_b30.py (scored value and last visible sample between "
         f"{LOW_EV:.0f} eV and {HIGH_EV / 1000:.1f} keV, jump between them at most {JUMP_EV / 1000:.0f} keV, not within "
         f"{BLIP_S * 1000:.0f} ms after a row whose BES level is below half the discharge median); flagged = the rest.", "",
         "## Check before the readout", "",
         "| split | rows (sensitivity population) | plain share there | rows (B.41 population) | plain share there |",
         "|---|---|---|---|---|"]
    for s in S:
        c = res["check"][str(s)]
        L.append(f"| {s} | {c['sensitivity_population_rows']:,} | {c['plain_share_sensitivity_population']:.3f} | "
                 f"{c['b41_population_rows']:,} | {c['plain_share_b41_population']:.3f} |")
    L += ["", "The sensitivity population additionally requires a later visible sample (has_future) and finite PCHIP and "
          "network values; docs/paper/b30_sensitivity.md reports 0.885 to 0.898 for it.", "",
          "## Operational choices made before the run (where the registration leaves room)", "",
          "- Population: the registered one (k = 1; hold, GP, twin, causal variant and measurement finite), without the "
          "has_future and PCHIP requirement of the sensitivity script (about 100 more rows per split).",
          "- Top 1 %: rows whose hold squared error is at or above its 99th percentile within the split.",
          "- Thirds: tertiles of the last visible sample (hold) within the split.",
          "- Bursts: an absent frame is a 20 to 40 ms step t_j -> t_(j+1) in the 10 ms file; an onset is excluded when it "
          "falls in (t_j, t_(j+1) + 90 ms], i.e. on or within 100 ms after the last absent stamp t_(j+1) - 10 ms. "
          "Windows by row stamp: 50 to 10 ms before ([-50, -10] ms), 0 to 30 ms after ([0, 30] ms).",
          "- Burst traces: mean over events of the per-event ratio in each 10 ms bin of stamp minus onset (bins [-50, -40) "
          "to [90, 100) ms); ECEI and BES levels are channel means of the 10 ms snapshots.",
          "- Raw-stream test discharges: each assigned to the first split whose test files contain it (B.37 part S).",
          "- Blips: the B.35 event rule (five 10 ms steps before, at least three measured T_i of the six rows before); "
          "ratios averaged over scored rows of the row table.", "",
          "## Statements", "",
          f"- **G1** (plain rows hold more than half of the reduction of the causal variant against the twin; interval above 0.5): "
          f"{n('G1_holds')} of 4 -> {v(n('G1_holds') >= 3)}. Shares: "
          + "; ".join(f3(G[str(s)]['plain']['share_of_reduction_vs_twin']) for s in S) + ".",
          f"- **G2** (the same against the causal GP): {n('G2_holds')} of 4 -> {v(n('G2_holds') >= 3)}. Shares: "
          + "; ".join(f3(G[str(s)]['plain']['share_of_reduction_vs_gp']) for s in S) + ".",
          f"- **Q1** (90th percentile of the absolute error, causal variant over causal GP and over hold, both intervals below 1): "
          f"{n('Q1_holds')} of 4 -> {v(n('Q1_holds') >= 3)}. Over the GP: "
          + "; ".join(f3(G[str(s)]['Q1_p90_ratio_vs_gp']) for s in S) + ". Over hold: "
          + "; ".join(f3(G[str(s)]['Q1_p90_ratio_vs_hold']) for s in S) + ".",
          f"- **U1** (median ratio of T_i 0 to 30 ms after a burst onset to 50 to 10 ms before, interval below 1; "
          f"{U['events']} events of {U['discharges_with_events']} discharges, testable at 30 or more): "
          f"{f3(U['median_ratio_ti'])} -> {v(U['U1_holds']) if U['U1_testable'] else 'not testable'}.", "",
          "## G. Rows by the value rules (reported)", "",
          "| split | class | rows | share of rows | share of the reduction vs twin | vs causal GP | skill causal vs twin | vs causal GP | vs hold |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for cn, lab in (("plain", "plain"), ("flagged", "flagged"), ("top1_hold", "top 1 % of hold error"),
                        ("rest_hold", "other 99 %")):
            c = G[str(s)][cn]
            L.append(f"| {s} | {lab} | {c['rows']:,} | {f3(c['row_share'])} | {f3(c['share_of_reduction_vs_twin'])} | "
                     f"{f3(c['share_of_reduction_vs_gp'])} | {s3(c['skill_causal_vs_twin'])} | {s3(c['skill_causal_vs_gp'])} | "
                     f"{s3(c['skill_causal_vs_hold'])} |")
    L += ["", "Shares of the reduction are sums over rows of (reference error squared minus causal error squared) in the "
          "class over the same sum over all rows; a share above 1 means the other class has a net increase.", "",
          "## Q. Distribution of the absolute error (reported)", "",
          "| split | estimator | p50 | p75 | p90 | p95 | p99 (eV) |", "|---|---|---|---|---|---|---|"]
    for s in S:
        for e in EST:
            p = G[str(s)]["abs_error_percentiles_eV"][e]
            L.append(f"| {s} | {LABEL[e]} | " + " | ".join(f"{p[str(q)]:.1f}" for q in PCTS) + " |")
    L += ["", "| split | causal variant over | p50 | p75 | p90 | p95 | p99 |", "|---|---|---|---|---|---|---|"]
    for s in S:
        for ref in ("hold", "gp", "hist80"):
            r = G[str(s)]["ratio_causal_over"][ref]
            L.append(f"| {s} | {LABEL[ref]} | " + " | ".join(f3(r[str(q)]) for q in PCTS) + " |")
    L += ["", "Mean signed error (estimate minus measurement, eV) by third of the last visible sample:", "",
          "| split | third (hold, eV) | " + " | ".join(LABEL[e] for e in EST) + " |", "|---|---|" + "---|" * len(EST)]
    for s in S:
        o = G[str(s)]
        t1, t2 = o["hold_thirds_eV"]
        for tn, rng_ in (("low", f"<= {t1:.0f}"), ("middle", f"{t1:.0f} to {t2:.0f}"), ("high", f"> {t2:.0f}")):
            L.append(f"| {s} | {tn} ({rng_}) | " + " | ".join(s1(o['mean_signed_error_eV'][tn][e]) for e in EST) + " |")
    L += ["", "Slope of the least-squares line of the measurement on each estimate (1 = calibrated, below 1 = estimates "
          "spread too wide):", "", "| split | " + " | ".join(LABEL[e] for e in EST) + " |", "|---|" + "---|" * len(EST)]
    for s in S:
        L.append(f"| {s} | " + " | ".join(f3(G[str(s)]['slope_measurement_on_estimate'][e]) for e in EST) + " |")
    L += ["", "## U. Edge-burst onsets (twelve raw-stream discharges, B.31 detector unchanged)", "",
          f"Event: onset with no other onset in the 50 ms before, not in (t_j, t_(j+1) + 90 ms] for any absent-frame step "
          f"t_j -> t_(j+1) of 20 to 40 ms (i.e. not within 100 ms after the last absent frame stamp), with at least two measured "
          f"T_i rows stamped 50 to 10 ms before and at least one 0 to 30 ms after. Level = mean of the rows 50 to 10 ms before; "
          f"ECEI and BES levels = mean over channels of the 10 ms snapshots in the same windows.", "",
          "| discharge | onsets | isolated | outside absent frames | with T_i rows (events) |", "|---|---|---|---|---|"]
    for sh, d in U["per_discharge"].items():
        L.append(f"| {sh} | {d['onsets']} | {d['isolated']} | {d['outside_absent_frames']} | {d['with_ti_rows']} |")
    L += ["", "| quantity | median ratio after / before | mean ratio | share of events below 1 |", "|---|---|---|---|"]
    for q, lab in (("ti", "T_i"), ("ecei", "ECEI level"), ("bes", "BES level")):
        L.append(f"| {lab} | {f3(U[f'median_ratio_{q}'])} | {U[f'mean_ratio_{q}']:.3f} | {U[f'share_below_1_{q}']:.3f} |")
    tr = U["traces"]
    L += ["", "Traces (mean over events of the ratio to the level before; bins of 10 ms by stamp minus onset; interval over discharges):", "",
          "| bin (ms) | events | T_i | ECEI | BES |", "|---|---|---|---|---|"]
    for b, lo in enumerate(tr["bins_ms"]):
        g = lambda k: f"{tr[k]['mean'][b]:.3f} [{tr[k]['lo'][b]:.3f}, {tr[k]['hi'][b]:.3f}]"   # noqa: E731
        L.append(f"| {lo} to {lo + 10} | {tr['T_i']['events'][b]} | {g('T_i')} | {g('ECEI')} | {g('BES')} |")
    P = U["test_discharge_estimates"]
    tt = U["traces_test_discharges"]
    L += ["", f"Raw-stream discharges that are test discharges (split of the first test assignment, as in B.37 part S): "
          + ", ".join(f"{sh} (split {s})" for sh, s in P["test_discharges"].items())
          + f"; {P['events']} events. Ratios to the measured level before, on the scored rows of the row table:", "",
          "| bin (ms) | events | measured | causal variant | twin |", "|---|---|---|---|---|"]
    for b, lo in enumerate(tr["bins_ms"]):
        g = lambda k: f"{tt[k]['mean'][b]:.3f} [{tt[k]['lo'][b]:.3f}, {tt[k]['hi'][b]:.3f}]"   # noqa: E731
        L.append(f"| {lo} to {lo + 10} | {tt['causal']['events'][b]} | {g('measured_same_rows')} | {g('causal')} | {g('twin')} |")
    L += ["", f"Rows 0 to 30 ms after an onset (each row once): {P['rows']} rows; RMSE (eV) causal variant "
          f"{P['rmse_eV']['causal']:.1f}, twin {P['rmse_eV']['twin']:.1f}, hold {P['rmse_eV']['hold']:.1f}, causal GP "
          f"{P['rmse_eV']['gp']:.1f}; skill causal vs twin {s3(P['skill_causal_vs_twin'])} (interval over "
          f"{len(P['test_discharges'])} discharges).", "",
          "### After the fact: median traces around burst onsets (not registered)", "",
          "The mean traces above are carried by a few events: the per-event T_i ratio spans "
          + ", ".join(f"{x:.2f}" for x in U["after_the_fact_level_before_quantiles"]["ti_ratio"])
          + " (quantiles 0, 0.05, 0.25, 0.5, 0.75, 0.95, 1), the T_i level before spans "
          + ", ".join(f"{x:.0f}" for x in U["after_the_fact_level_before_quantiles"]["ti"])
          + " eV, and the channel-mean ECEI level before spans "
          + ", ".join(f"{x:.3f}" for x in U["after_the_fact_level_before_quantiles"]["ecei"])
          + " (it reaches zero on some discharges, so its ratio is ill-defined there). Medians over events of the same "
          "per-event values (interval over discharges); the figure's burst panel uses these:", "",
          "| bin (ms) | T_i | ECEI | BES | test subset: measured | causal variant | twin |", "|---|---|---|---|---|---|---|"]
    for b, lo in enumerate(tr["bins_ms"]):
        g = lambda d: (f"{d['after_the_fact_median']['median'][b]:.3f} [{d['after_the_fact_median']['lo'][b]:.3f}, "   # noqa: E731
                       f"{d['after_the_fact_median']['hi'][b]:.3f}]")
        L.append(f"| {lo} to {lo + 10} | {g(tr['T_i'])} | {g(tr['ECEI'])} | {g(tr['BES'])} | {g(tt['measured_same_rows'])} | "
                 f"{g(tt['causal'])} | {g(tt['twin'])} |")
    L += ["", "## Beam blips (test discharges of the four splits, reported)", "",
          "Event rule of B.35: a 20 to 40 ms step, five 10 ms steps before it, T_i measured in at least three of the six rows "
          "before; level = median of those measured values. Rows k = 1 to 15 after the absent frame(s) while the 10 ms steps "
          "continue; a row enters when it is a scored row of the row table. Mean over rows of value / level.", ""]
    ks = B["rows_k"]
    for s in S:
        o = B[str(s)]
        L += [f"### split {s}: {o['events']} events, {o['discharges']} discharges", "",
              "| row k | rows | measured T_i | causal variant | twin | hold | causal GP |", "|---|---|---|---|---|---|---|"]
        for a, k in enumerate(ks):
            if k < 1:
                continue
            tr = o["traces"]
            L.append(f"| {k} | {tr['y']['rows'][a]} | " + " | ".join(
                f"{tr[c]['mean'][a]:.3f} [{tr[c]['lo'][a]:.3f}, {tr[c]['hi'][a]:.3f}]"
                for c in ("y", "causal", "hist80", "hold", "gp")) + " |")
        L += ["", f"Skill causal vs twin: rows 1 to 3 {s3(o['skill_causal_vs_twin_rows_1-3'])} ({o['rows_1-3']} rows); "
              f"rows 4 to 10 {s3(o['skill_causal_vs_twin_rows_4-10'])} ({o['rows_4-10']} rows).", ""]
    L += ["## Figure", "",
          "docs/paper/b41_events_fig.pdf / .png from docs/paper/b41_events_fig.json (fig_events.py). Blip panel: the four "
          "splits pooled, mean ratios, resampling unit = discharge within split (a presentation choice, not a registered "
          "readout). Burst panel: after-the-fact median traces; measured T_i, ECEI and BES on all events, the two networks "
          "and the measurement on the same rows on the test-discharge subset.", ""]
    return "\n".join(L) + "\n"


def main():
    t0 = _time.time()
    C = Ctx()
    print(f"[{_time.time() - t0:.0f}s] grid loaded", flush=True)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B41.md", "boot_seed": R.BOOT_SEED,
           "n_boot": R.N_BOOT}
    res["check"] = precheck(C)
    res["G"] = part_gq(C, np.random.default_rng(R.BOOT_SEED))
    res["U"] = part_u(C, np.random.default_rng(R.BOOT_SEED + 1))
    res["B"] = part_b(C, np.random.default_rng(R.BOOT_SEED + 2))
    n = lambda k: sum(res["G"][str(s)][k] for s in R.SEEDS)                     # noqa: E731
    res["verdicts"] = {"G1": {"splits": n("G1_holds"), "holds": n("G1_holds") >= 3},
                       "G2": {"splits": n("G2_holds"), "holds": n("G2_holds") >= 3},
                       "Q1": {"splits": n("Q1_holds"), "holds": n("Q1_holds") >= 3},
                       "U1": {"testable": res["U"]["U1_testable"], "holds": res["U"]["U1_holds"],
                              "median_ratio": res["U"]["median_ratio_ti"], "events": res["U"]["events"]}}
    res["runtime_s"] = round(_time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    fig = {"note": "data of docs/paper/b41_events_fig.pdf; written by run_b41.py",
           "blip": {"rows_k": res["B"]["rows_k"], **res["B"]["pooled_for_figure"]},
           "burst": {"bins_ms": res["U"]["traces"]["bins_ms"], **{k: res["U"]["traces"][k] for k in ("T_i", "ECEI", "BES")},
                     **res["U"]["traces_test_discharges"], "events": res["U"]["events"],
                     "test_events": res["U"]["test_discharge_estimates"]["events"]}}
    OUT_FIG.write_text(json.dumps(fig, indent=1), encoding="utf-8")
    print(json.dumps(res["verdicts"], indent=1))
    print(f"written in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
