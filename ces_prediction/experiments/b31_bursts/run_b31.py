"""B.31 (PREREGISTRATION_B31.md): edge bursts in the raw 500 kHz BES stream of the twelve
discharges with a raw delivery; where the CES gaps, the large changes, the hold error and the
fast-channel share sit.

CPU only.  Reads data/raw_mus/.cache/s{shot}_BES.npz, the shot files and the frozen B.30 row
tables (data/.b30/rows_cut_s*.npz).  Run from the repo root:
    py ces_prediction/experiments/b31_bursts/run_b31.py
Output: docs/paper/b31_bursts.json / .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b30_thinned"))
import run_b30 as R  # noqa: E402

RAW = R.DATA / "raw_mus" / ".cache"
OUT_JSON = R.REPO / "docs" / "paper" / "b31_bursts.json"
OUT_MD = R.REPO / "docs" / "paper" / "b31_bursts.md"
SHOTS = (31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097)
DISCOVERY = 31921
BIN_S = 0.0002
MED_BINS = 251
K_SIGMA = 6.0
MERGE_S = 0.002
NEAR_S, PHASE_S, BLIP_S, ACTIVE_MIN = 0.010, 0.200, 0.030, 3


def bursts_and_blips(shot):
    with np.load(RAW / f"s{shot}_BES.npz", allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        x = z["x"].mean(1).astype(np.float64)
    fs, t0 = float(meta["fs_hz"]), float(meta["t0"])
    w = int(round(fs * BIN_S))
    nb = len(x) // w
    b = x[:nb * w].reshape(nb, w).mean(1)
    tb = t0 + (np.arange(nb) + 0.5) * BIN_S
    pad = np.pad(b, (MED_BINS // 2, MED_BINS - 1 - MED_BINS // 2), mode="edge")
    base = np.median(sliding_window_view(pad, MED_BINS), axis=1)
    res = b - base
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    peak = res > K_SIGMA * sig
    d = np.diff(np.concatenate(([0], peak.astype(int), [0])))
    ev = tb[np.flatnonzero(d == 1)]
    if len(ev):
        ev = ev[np.concatenate(([True], np.diff(ev) > MERGE_S))]
    w1 = int(round(fs * 0.001))
    n1 = len(x) // w1
    lvl = x[:n1 * w1].reshape(n1, w1).mean(1)
    t1 = t0 + (np.arange(n1) + 0.5) * 0.001
    blip = t1[lvl < 0.5 * np.median(lvl)]
    return {"t0": t0, "t_end": t0 + len(x) / fs, "bursts": ev, "blips": blip,
            "level_1ms": (t1, lvl / np.median(lvl))}


def labels(t, raw):
    """0 unlabelled, 1 quiet, 2 sparse, 3 active; near flag; blip flag."""
    ev, bl = raw["bursts"], raw["blips"]
    lab = np.zeros(len(t), np.int8)
    near = np.zeros(len(t), bool)
    blip = np.zeros(len(t), bool)
    ok = (t - PHASE_S >= raw["t0"]) & (t <= raw["t_end"])
    n_phase = np.searchsorted(ev, t, side="right") - np.searchsorted(ev, t - PHASE_S, side="right")
    n_near = np.searchsorted(ev, t, side="right") - np.searchsorted(ev, t - NEAR_S, side="right")
    n_blip = np.searchsorted(bl, t, side="right") - np.searchsorted(bl, t - BLIP_S, side="right")
    lab[ok & (n_phase == 0)] = 1
    lab[ok & (n_phase > 0) & (n_phase < ACTIVE_MIN)] = 2
    lab[ok & (n_phase >= ACTIVE_MIN)] = 3
    near[ok] = n_near[ok] > 0
    blip[ok] = n_blip[ok] > 0
    lab[blip] = 0
    near[blip] = False
    return lab, near, blip


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, _bb, _gap, _m = R.imports()
    grid, _dims = load_grid_files(R.DATA, True, R.POPS["cut"])
    names = sorted(grid)
    rng = np.random.default_rng(R.BOOT_SEED)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B31.md", "discovery": DISCOVERY,
           "discharges": {}, "model": {"entries": []}}
    raw, lab = {}, {}
    for s in SHOTS:
        nm = f"s{s}.csv"
        arr = grid[nm]
        t = arr[:, 0].astype(np.float64)
        ti = arr[:, 1].astype(np.float64)
        raw[s] = bursts_and_blips(s)
        lab[s] = labels(t, raw[s])
        L, near, blip = lab[s]
        miss = np.isnan(ti)
        both = ~miss[1:] & ~miss[:-1] & (np.diff(t) < 0.015)
        dti = np.abs(np.diff(ti))
        Lp = L[1:]
        d = {"bursts": int(len(raw[s]["bursts"])), "raw_span_s": [raw[s]["t0"], raw[s]["t_end"]],
             "blips_ms": int(len(raw[s]["blips"])), "rows_blip_excluded": int(blip.sum()),
             "rows": {k: int((L == v).sum()) for k, v in (("quiet", 1), ("sparse", 2), ("active", 3))},
             "rows_near": int(near.sum())}
        q = L == 1
        d["missing_share"] = {"near": float(miss[near].mean()) if near.sum() else None,
                              "quiet": float(miss[q].mean()) if q.sum() else None,
                              "active": float(miss[L == 3].mean()) if (L == 3).sum() else None,
                              "n_near": int(near.sum()), "n_quiet": int(q.sum())}
        pa, pq = both & (Lp == 3), both & (Lp == 1)
        d["abs_step_median_eV"] = {"active": float(np.median(dti[pa])) if pa.sum() else None,
                                   "quiet": float(np.median(dti[pq])) if pq.sum() else None,
                                   "n_active": int(pa.sum()), "n_quiet": int(pq.sum())}
        res["discharges"][str(s)] = d
        print(s, d["bursts"], d["rows"], "near", d["rows_near"], flush=True)

    # ------------------------------------------------------------ model rows (B.30, cut, k = 1)
    ests = ["hold", "gp", "pchip", *R.NETS["cut"]]
    for seed in R.SEEDS:
        z = np.load(R.ROWS_DIR / f"rows_cut_s{seed}.npz")
        tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
        fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0) & (tab["k"] == 1)
        for s in SHOTS:
            f = names.index(f"s{s}.csv")
            sel = fin & (tab["file"] == f)
            if sel.sum() == 0:
                continue
            rows = tab["row"][sel]
            L, near, _blip = lab[s]
            se = {e: (tab[e][sel] - tab["y"][sel]).astype(np.float64) ** 2 for e in ests}
            ent = {"split": seed, "shot": s, "cells": {}}
            for name, v in (("quiet", 1), ("sparse", 2), ("active", 3)):
                m = L[rows] == v
                ent["cells"][name] = {"rows": int(m.sum()),
                                      **{f"sse_{e}": float(se[e][m].sum()) for e in ests}}
            m = near[rows]
            ent["cells"]["near"] = {"rows": int(m.sum()), **{f"sse_{e}": float(se[e][m].sum()) for e in ests}}
            res["model"]["entries"].append(ent)
            print("model", seed, s, {k: c["rows"] for k, c in ent["cells"].items()}, flush=True)

    # ------------------------------------------------------------ statements
    v = {}
    oth = [s for s in SHOTS if s != DISCOVERY]
    el = [s for s in oth if res["discharges"][str(s)]["missing_share"]["n_near"] >= 20
          and res["discharges"][str(s)]["missing_share"]["n_quiet"] >= 20]
    ok = [s for s in el if res["discharges"][str(s)]["missing_share"]["near"]
          > res["discharges"][str(s)]["missing_share"]["quiet"]]
    v["H31-1"] = {"eligible": el, "larger_in_near": ok, "readable": len(el) >= 4,
                  "holds": bool(len(el) >= 4 and len(ok) >= 0.8 * len(el))}
    el = [s for s in oth if res["discharges"][str(s)]["abs_step_median_eV"]["n_active"] >= 20
          and res["discharges"][str(s)]["abs_step_median_eV"]["n_quiet"] >= 20]
    ok = [s for s in el if res["discharges"][str(s)]["abs_step_median_eV"]["active"]
          > res["discharges"][str(s)]["abs_step_median_eV"]["quiet"]]
    v["H31-2"] = {"eligible": el, "larger_in_active": ok, "readable": len(el) >= 4,
                  "holds": bool(len(el) >= 4 and len(ok) >= 0.8 * len(el))}

    ents = [e for e in res["model"]["entries"] if e["shot"] != DISCOVERY]
    by_shot = {}
    for e in ents:
        by_shot.setdefault(e["shot"], []).append(e)

    def rmse(cells, name, est):
        n = sum(c[name]["rows"] for c in cells)
        return (sum(c[name][f"sse_{est}"] for c in cells) / n) ** 0.5 if n else None

    el, ok, per = [], [], {}
    for s, es in by_shot.items():
        cells = [e["cells"] for e in es]
        na = min(c["active"]["rows"] for c in cells)
        nq = min(c["quiet"]["rows"] for c in cells)
        ra, rq = rmse(cells, "active", "hold"), rmse(cells, "quiet", "hold")
        per[str(s)] = {"hold_rmse_active": ra, "hold_rmse_quiet": rq, "rows_active": na, "rows_quiet": nq}
        if na >= 30 and nq >= 30:
            el.append(s)
            if ra > rq:
                ok.append(s)
    v["H31-3"] = {"eligible": el, "larger_in_active": ok, "per_discharge": per, "readable": len(el) >= 3,
                  "holds": bool(len(el) >= 3 and len(ok) >= 0.8 * len(el))}

    shots_m = sorted(by_shot)

    def pooled(weights):
        """Sums over entries with one weight per discharge."""
        out = {}
        for name in ("quiet", "sparse", "active"):
            a = b = n = 0.0
            for s, w in zip(shots_m, weights):
                for e in by_shot[s]:
                    c = e["cells"][name]
                    a += w * c["sse_bb80"]
                    b += w * c["sse_hist80"]
                    n += w * c["rows"]
            out[name] = (a, b, n)
        return out

    def stats(weights):
        p = pooled(weights)
        gain = {k: p[k][1] - p[k][0] for k in p}
        tot_gain, tot_rows = sum(gain.values()), sum(p[k][2] for k in p)
        return {"skill_active": 1 - p["active"][0] / p["active"][1] if p["active"][1] else np.nan,
                "skill_quiet": 1 - p["quiet"][0] / p["quiet"][1] if p["quiet"][1] else np.nan,
                "gain_share_active": gain["active"] / tot_gain if tot_gain else np.nan,
                "row_share_active": p["active"][2] / tot_rows if tot_rows else np.nan}

    pt = stats(np.ones(len(shots_m)))
    boots = [stats(w) for w in rng.multinomial(len(shots_m), np.full(len(shots_m), 1.0 / len(shots_m)),
                                               size=R.N_BOOT).astype(float)]

    def ci(key):
        x = np.array([b[key] for b in boots], dtype=float)
        x = x[np.isfinite(x)]
        return [float(pt[key]), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    v["H31-4"] = {"discharges": shots_m, "skill_active": ci("skill_active"),
                  "gain_share_active": ci("gain_share_active"), "row_share_active": ci("row_share_active")}
    v["H31-4"]["holds"] = bool(v["H31-4"]["gain_share_active"][0] > v["H31-4"]["row_share_active"][0]
                               and v["H31-4"]["skill_active"][1] > 0)
    v["H31-5"] = {"skill_quiet": ci("skill_quiet")}
    v["H31-5"]["reading"] = ("present without bursts as well" if v["H31-5"]["skill_quiet"][1] > 0
                             else "established in burst-active phases only"
                             if v["H31-5"]["skill_quiet"][1] <= 0 <= v["H31-5"]["skill_quiet"][2]
                             else "below zero in quiet rows")
    res["verdicts"] = v
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print(json.dumps({k: {a: b for a, b in d.items() if a != "per_discharge"} for k, d in v.items()}, indent=1))


def render(res):
    L = ["# B.31 edge bursts in the raw BES stream (registration PREREGISTRATION_B31.md)", "",
         "Labels of a 10 ms row from the 500 kHz BES stream up to the row time: near = a burst in the "
         "preceding 10 ms; active = at least 3 bursts in the preceding 200 ms; quiet = none; sparse = "
         "1 or 2. Discharge 31921 was the discovery discharge and enters no rule.", "",
         "## Per discharge", "",
         "| discharge | raw span (s) | bursts | quiet / sparse / active rows | near rows | T_i missing: near | quiet | active | median abs step (eV): active | quiet |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    for s, d in res["discharges"].items():
        L.append(f"| {s}{' (discovery)' if int(s) == res['discovery'] else ''} | "
                 f"{d['raw_span_s'][0]:.2f}-{d['raw_span_s'][1]:.2f} | {d['bursts']} | "
                 f"{d['rows']['quiet']} / {d['rows']['sparse']} / {d['rows']['active']} | {d['rows_near']} | "
                 f"{f(d['missing_share']['near'])} | {f(d['missing_share']['quiet'])} | "
                 f"{f(d['missing_share']['active'])} | {f(d['abs_step_median_eV']['active'], 1)} | "
                 f"{f(d['abs_step_median_eV']['quiet'], 1)} |")
    L += ["", "## Model rows (B.30 row tables, cut population, k = 1)", "",
          "| split | discharge | rows quiet / sparse / active / near | hold RMSE quiet | active | network vs history-only: quiet | active |",
          "|---|---|---|---|---|---|---|"]
    for e in res["model"]["entries"]:
        c = e["cells"]

        def r(name, est):
            return (c[name][f"sse_{est}"] / c[name]["rows"]) ** 0.5 if c[name]["rows"] else None

        def sk(name):
            return 1 - c[name]["sse_bb80"] / c[name]["sse_hist80"] if c[name]["sse_hist80"] else None

        L.append(f"| {e['split']} | {e['shot']}{' (discovery)' if e['shot'] == res['discovery'] else ''} | "
                 f"{c['quiet']['rows']} / {c['sparse']['rows']} / {c['active']['rows']} / {c['near']['rows']} | "
                 f"{f(r('quiet', 'hold'), 1)} | {f(r('active', 'hold'), 1)} | {f(sk('quiet'), 3)} | {f(sk('active'), 3)} |")
    L += ["", "## Registered statements", ""]
    for h, d in res["verdicts"].items():
        L.append(f"- **{h}**: " + json.dumps({k: v for k, v in d.items() if k != "per_discharge"}))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
