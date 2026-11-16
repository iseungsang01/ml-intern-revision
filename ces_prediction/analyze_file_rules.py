"""How the 10 ms discharge files relate to the raw signals (descriptive; CPU; no model).

Everything here is read from the files themselves, by setting the 10 ms files against the raw
1 us deliveries of twelve discharges (data/raw_mus/.cache) and against each other:

 1. grid      time stamps, step sizes, the leading and trailing row of every file (no CES value;
              the provider's note says to ignore them), absent frames;
 2. missing   T_i missing rate by quintile of the BES level with those two rows left out (the
              earlier ledger counted them as missing rows);
 3. raw       every fast-channel value of a row against the raw sample at the time stamp;
 4. blips     beam blips in the raw BES against the absent frames, and beam interruptions that
              removed no frame;
 5. response  what T_i, fresh V_rot, ECEI, BES and Mirnov do in the rows after an absent frame,
              all files, with discharge-level bootstrap intervals;
 6. recovery  raw BES level around the blips at 1 ms.

Run from the repo root:  py ces_prediction/analyze_file_rules.py
(add --render to rewrite the .md from the stored .json without reading the data again)
Output: docs/paper/file_rules.json / .md
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO / "data"))
RAW = DATA / "raw_mus" / ".cache"
OUT_JSON = REPO / "docs" / "paper" / "file_rules.json"
OUT_MD = REPO / "docs" / "paper" / "file_rules.md"

SEG_BREAK_S = 0.5            # a step of at least this length starts a new segment
ABSENT_MAX_S = 0.045         # steps of 20, 30, 40 ms = one, two, three absent frames
PRINT_TOL = 0.6e-5           # the files print five decimals
N_BOOT, BOOT_SEED = 2000, 20261116
RAW_SHOTS = (31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097)
K_BEFORE, K_AFTER = 6, 15    # rows before / after an absent frame
RESP_ROWS = (1, 2, 3, 4, 5, 6)   # rows over which the T_i and ECEI deficits are compared


def load(path):
    """A discharge file without its leading (t = 0) and trailing (t = 30 s) row."""
    df = pd.read_csv(path)
    t = df["time"].to_numpy(float)
    first = bool(t[0] == 0.0 and np.isnan(df["CES_TI"].iloc[0]) and np.isnan(df["CES_VT"].iloc[0]))
    last = bool(len(t) > 1 and t[-1] - t[-2] >= SEG_BREAK_S and np.isnan(df["CES_TI"].iloc[-1])
                and np.isnan(df["CES_VT"].iloc[-1]))
    return df.iloc[int(first):len(df) - int(last)].reset_index(drop=True), first, last, float(t[-1])


def absent_groups(t):
    """(index of the last row before the step, stamps of the absent frames) for steps of 20-40 ms."""
    dt = np.diff(t)
    out = []
    for j in np.flatnonzero((dt > 0.015) & (dt < ABSENT_MAX_S)):
        n = int(round(dt[j] / 0.010)) - 1
        out.append((int(j), [float(t[j] + 0.010 * (i + 1)) for i in range(n)]))
    return out


# ------------------------------------------------------------------------------------------
# 1. grid, 2. missing rate by BES level, 5. response to absent frames  (all files, one pass)
# ------------------------------------------------------------------------------------------
def all_files(files):
    steps, stamp_mod, last_times = Counter(), Counter(), Counter()
    n_raw_rows = n_rows = n_first = n_last = n_segments = 0
    miss = {"CES_TI": 0, "CES_VT": 0}
    bes_z, ti_missing, file_of_row = [], [], []
    ks = [k for k in range(-K_BEFORE, K_AFTER + 1) if k != 0]
    ev = {q: {k: [] for k in ks} for q in ("ti", "vt", "ecei", "bes", "mc", "ti_obs")}
    ev_file, ev_tipre, ev_frames = [], [], []
    for fi, f in enumerate(files):
        df, first, last, t_last = load(f)
        n_raw_rows += len(df) + int(first) + int(last)
        n_first += int(first)
        n_last += int(last)
        last_times[round(t_last, 3)] += 1
        t = df["time"].to_numpy(float)
        n_rows += len(t)
        if len(t) == 0:
            continue
        dt = np.diff(t)
        for k, v in Counter(np.rint(dt * 1000).astype(int).tolist()).items():
            steps[k] += v
        for k, v in Counter((np.rint(t * 1000).astype(int) % 10).tolist()).items():
            stamp_mod[k] += v
        n_segments += 1 + int((dt >= SEG_BREAK_S).sum())
        ti, vt = df["CES_TI"].to_numpy(float), df["CES_VT"].to_numpy(float)
        miss["CES_TI"] += int(np.isnan(ti).sum())
        miss["CES_VT"] += int(np.isnan(vt).sum())
        bes = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float).mean(1)
        ecei = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
        mc = np.abs(df[[c for c in df.columns if c.startswith("MC")]].to_numpy(float)).mean(1)
        sd = bes.std()
        bes_z.append((bes - bes.mean()) / sd if sd > 0 else np.full(len(bes), np.nan))
        ti_missing.append(np.isnan(ti))
        file_of_row.append(np.full(len(t), fi))
        fresh = np.concatenate(([True], vt[1:] != vt[:-1])) & np.isfinite(vt)
        for j, stamps in absent_groups(t):
            if j < K_BEFORE - 1:
                continue
            pre = slice(j - K_BEFORE + 1, j + 1)
            if not np.all(np.abs(np.diff(t[pre]) - 0.010) < 1e-4):
                continue
            tip = np.nanmedian(ti[pre]) if np.isfinite(ti[pre]).any() else np.nan
            vfresh = vt[pre][fresh[pre]]
            vp = float(np.median(vfresh)) if len(vfresh) else np.nan
            ep, bp, mp = np.median(ecei[pre]), np.median(bes[pre]), np.median(mc[pre])
            ev_file.append(fi)
            ev_tipre.append(tip)
            ev_frames.append(len(stamps))
            t_first_after = t[j + 1]
            for k in ks:
                i = j + k + 1 if k < 0 else j + k
                want = t[j] + 0.010 * (k + 1) if k < 0 else t_first_after + 0.010 * (k - 1)
                okrow = 0 <= i < len(t) and abs(t[i] - want) < 1e-3
                ev["ti"][k].append(ti[i] / tip if okrow and np.isfinite(tip) and tip > 0 else np.nan)
                ev["vt"][k].append(vt[i] / vp if okrow and fresh[i] and np.isfinite(vp) and abs(vp) > 5 else np.nan)
                ev["ecei"][k].append(ecei[i] / ep if okrow and abs(ep) > 1e-9 else np.nan)
                ev["bes"][k].append(bes[i] / bp if okrow and abs(bp) > 1e-9 else np.nan)
                ev["mc"][k].append(mc[i] / mp if okrow and mp > 1e-12 else np.nan)
                ev["ti_obs"][k].append(float(np.isfinite(ti[i])) if okrow else np.nan)
    grid = {
        "files": len(files), "rows_in_files": n_raw_rows, "leading_rows_t0_without_ces": n_first,
        "trailing_rows_without_ces": n_last,
        "trailing_row_times_s": {str(k): v for k, v in sorted(last_times.items())[-3:]},
        "grid_rows": n_rows, "segments": n_segments,
        "stamp_ms_modulo_10": {str(k): v for k, v in sorted(stamp_mod.items())},
        "steps_ms": {str(k): v for k, v in sorted(steps.items()) if k <= 60},
        "steps_60_to_500_ms": int(sum(v for k, v in steps.items() if 60 < k < 500)),
        "steps_ge_500_ms": int(sum(v for k, v in steps.items() if k >= 500)),
        "share_of_steps_10_ms": steps[10] / max(1, sum(steps.values())),
        "absent_frames": int(steps[20] + 2 * steps[30] + 3 * steps[40]),
        "missing_pct": {c: 100 * m / n_rows for c, m in miss.items()},
    }
    # --- 2. missing rate of T_i by quintile of the per-discharge standardised BES level
    z, m, fr = np.concatenate(bes_z), np.concatenate(ti_missing), np.concatenate(file_of_row)
    ok = np.isfinite(z)
    z, m, fr = z[ok], m[ok], fr[ok]
    q = np.searchsorted(np.quantile(z, [0.2, 0.4, 0.6, 0.8]), z, side="right")
    rng = np.random.default_rng(BOOT_SEED)
    uf, inv = np.unique(fr, return_inverse=True)
    n_tab = np.zeros((len(uf), 5))
    k_tab = np.zeros((len(uf), 5))
    np.add.at(n_tab, (inv, q), 1.0)
    np.add.at(k_tab, (inv, q), m.astype(float))
    W = rng.multinomial(len(uf), np.full(len(uf), 1.0 / len(uf)), size=N_BOOT).astype(float)
    rb = (W @ k_tab) / (W @ n_tab)
    quint = {"rows": int(len(z)),
             "missing_pct_by_quintile": [float(100 * k_tab[:, i].sum() / n_tab[:, i].sum()) for i in range(5)],
             "ci95_pct": [[float(100 * np.quantile(rb[:, i], 0.025)), float(100 * np.quantile(rb[:, i], 0.975))]
                          for i in range(5)]}
    # --- 5. response to absent frames
    ev_file = np.array(ev_file)
    uf2, finv = np.unique(ev_file, return_inverse=True)
    W2 = rng.multinomial(len(uf2), np.full(len(uf2), 1.0 / len(uf2)), size=N_BOOT).astype(float)

    def med(vals, sel=None):
        v = np.asarray(vals, float)
        okv = np.isfinite(v) if sel is None else np.isfinite(v) & sel
        v, f = v[okv], finv[okv]
        if len(v) == 0:
            return None, None, 0
        o = np.argsort(v)
        v, f = v[o], f[o]
        c = np.cumsum(W2[:, f], 1)
        b = v[(c >= c[:, -1:] / 2).argmax(1)]
        return float(np.median(v)), b, int(len(v))

    def mean(vals):
        v = np.asarray(vals, float)
        okv = np.isfinite(v)
        v, f = v[okv], finv[okv]
        w = W2[:, f]
        b = (w * v).sum(1) / w.sum(1)
        return float(v.mean()), b, int(len(v))

    def ci(b):
        return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]

    tipre = np.array(ev_tipre)
    resp = {"events": int(len(ev_file)), "discharges": int(len(uf2)),
            "events_by_absent_frames": {str(k): int(v) for k, v in sorted(Counter(ev_frames).items())},
            "t_i_before_eV_quantiles_10_50_90": [float(x) for x in np.nanquantile(tipre, [0.1, 0.5, 0.9])],
            "rows": {}}
    boot = {"ti": {}, "ecei": {}}
    for k in ks:
        row = {}
        for qn in ("ti", "vt", "ecei", "bes", "mc"):
            p, b, n = med(ev[qn][k])
            row[qn] = {"median": p, "ci95": ci(b), "n": n}
            if qn in boot:
                boot[qn][k] = (p, b)
        p, b, n = mean(ev["ti_obs"][k])
        row["ti_observed_share"] = {"mean": p, "ci95": ci(b), "n": n}
        for name, sel in (("ti_before_ge_700eV", tipre >= 700), ("ti_before_lt_400eV", tipre < 400)):
            p, b, n = med(ev["ti"][k], sel)
            row[name] = {"median": p, "ci95": ci(b), "n": n}
        resp["rows"][str(k)] = row
    d_ti = np.mean([1 - boot["ti"][k][0] for k in RESP_ROWS])
    d_ec = np.mean([1 - boot["ecei"][k][0] for k in RESP_ROWS])
    b_ti = np.mean([1 - boot["ti"][k][1] for k in RESP_ROWS], 0)
    b_ec = np.mean([1 - boot["ecei"][k][1] for k in RESP_ROWS], 0)
    resp["mean_deficit_rows_1_to_6"] = {"t_i": float(d_ti), "t_i_ci95": ci(b_ti), "ecei": float(d_ec),
                                        "ecei_ci95": ci(b_ec), "ratio": float(d_ti / d_ec), "ratio_ci95": ci(b_ti / b_ec)}
    with np.errstate(invalid="ignore"):
        a = np.nanmean(np.array([ev["ti"][k] for k in (1, 2, 3, 4)], float), 0)
        e = np.nanmean(np.array([ev["ecei"][k] for k in (1, 2, 3, 4)], float), 0)
    okp = np.isfinite(a) & np.isfinite(e)
    a, e, fp = a[okp], e[okp], finv[okp]
    rho = float(spearmanr(a, e)[0])
    rb = []
    for w in W2[:400].astype(int):
        idx = np.repeat(np.arange(len(a)), w[fp])
        rb.append(spearmanr(a[idx], e[idx])[0])
    resp["rank_correlation_ti_ecei_rows_1_to_4"] = {"rho": rho, "ci95": ci(np.array(rb)), "events": int(len(a)),
                                                    "bootstrap_resamples": 400}
    edges = (0.0, 0.85, 0.95, 1.05, 99.0)
    resp["ti_ratio_by_ecei_ratio"] = [{"ecei_ratio_from": lo, "to": hi, "events": int(((e >= lo) & (e < hi)).sum()),
                                       "median_ti_ratio": float(np.median(a[(e >= lo) & (e < hi)]))}
                                      for lo, hi in zip(edges[:-1], edges[1:])]
    return grid, quint, resp


# ------------------------------------------------------------------------------------------
# 3. raw samples, 4. blips, 6. recovery  (twelve discharges with raw data)
# ------------------------------------------------------------------------------------------
def raw_npz(shot, diag):
    z = np.load(RAW / f"s{shot}_{diag}.npz", allow_pickle=True)
    m = json.loads(str(z["meta"]))
    return z["x"], float(m["fs_hz"]), float(m["t0"]), list(m["channels"])


def raw_side():
    match, blip_rows, loose, prof = {}, [], [], []
    for shot in RAW_SHOTS:
        df, _f, _l, _t = load(DATA / f"s{shot}.csv")
        t = df["time"].to_numpy(float)
        ti = df["CES_TI"].to_numpy(float)
        match[str(shot)] = {}
        for diag in ("BES", "ECEI", "MC"):
            x, fs, t0, chans = raw_npz(shot, diag)
            chs = [c for c in chans if c in df.columns]
            ci_ = [chans.index(c) for c in chs]
            v = df[chs].to_numpy(float)
            idx = np.rint((t - t0) * fs).astype(np.int64)
            okr = (idx >= 1) & (idx < x.shape[0] - 1)
            hit = np.stack([np.abs(x[idx[okr] + s][:, ci_].astype(np.float64) - v[okr]).max(1) <= PRINT_TOL
                            for s in (-1, 0, 1)], 1)
            match[str(shot)][diag] = {"fs_hz": fs, "t0_s": t0, "rows": int(okr.sum()), "channels": len(chs),
                                      "share_equal_within_one_sample": float(hit.any(1).mean()),
                                      "share_at_offset_-1_0_+1": [float(h) for h in hit.mean(0)]}
            if diag == "BES":
                lvl_x, lvl_fs, lvl_t0 = x.mean(1).astype(np.float64), fs, t0
            del x
        # blips at 0.1 ms
        w = int(round(lvl_fs * 1e-4))
        n1 = len(lvl_x) // w
        lvl = lvl_x[:n1 * w].reshape(n1, w).mean(1)
        tb = lvl_t0 + np.arange(n1) * 1e-4
        ref = pd.Series(lvl).rolling(2000, center=True, min_periods=200).median().to_numpy()
        low = lvl < 0.5 * ref
        d = np.diff(np.concatenate(([0], low.astype(int), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        keep = (en - st) >= 30
        events = [(float(tb[a]), float(tb[min(b, n1 - 1)])) for a, b in zip(st[keep], en[keep])
                  if tb[a] > lvl_t0 + 0.05 and tb[min(b, n1 - 1)] < tb[-1] - 0.05]
        groups = [g for g in absent_groups(t) if lvl_t0 + 0.05 < g[1][0] and g[1][-1] < tb[-1] - 0.05]
        used = set()
        for j, stamps in groups:
            hit = [i for i, (a, b) in enumerate(events) if a < stamps[-1] + 0.010 and b > stamps[0] - 0.005]
            if hit:
                a, b = events[hit[0]][0], events[hit[-1]][1]
                used.update(hit)
                blip_rows.append({"shot": shot, "absent_frames": len(stamps), "first_absent_stamp_s": stamps[0],
                                  "start_minus_first_stamp_ms": (a - stamps[0]) * 1e3,
                                  "end_minus_last_stamp_ms": (b - stamps[-1]) * 1e3, "duration_ms": (b - a) * 1e3})
            else:
                blip_rows.append({"shot": shot, "absent_frames": len(stamps), "first_absent_stamp_s": stamps[0],
                                  "start_minus_first_stamp_ms": None, "end_minus_last_stamp_ms": None,
                                  "duration_ms": None})
        for i, (a, b) in enumerate(events):
            if i in used:
                continue
            inside = np.flatnonzero((t >= a + 0.002 - 1e-4) & (t + 0.008 <= b + 1e-4))
            before = np.flatnonzero((t < a - 0.002) & (t > a - 0.07))
            loose.append({"shot": shot, "start_s": a, "end_s": b, "duration_ms": (b - a) * 1e3,
                          "rows_exposed_without_beam": [float(x_) for x_ in t[inside]],
                          "t_i_in_those_rows_eV": [None if np.isnan(x_) else float(x_) for x_ in ti[inside]],
                          "t_i_median_before_eV": float(np.nanmedian(ti[before])) if np.isfinite(ti[before]).any() else None})
        # recovery profile at 1 ms around single absent frames
        w1 = int(round(lvl_fs * 1e-3))
        n2 = len(lvl_x) // w1
        l1 = lvl_x[:n2 * w1].reshape(n2, w1).mean(1)
        for j, stamps in groups:
            if len(stamps) != 1:
                continue
            i0 = int(round((stamps[0] - 0.002 - lvl_t0) * 1e3))
            if i0 - 60 >= 0 and i0 + 160 <= n2:
                seg = l1[i0 - 60:i0 + 160]
                prof.append(seg / np.median(seg[:50]))
        del lvl_x
        print(f"raw side: shot {shot} done", flush=True)
    prof = np.array(prof)
    rec = {"blips": int(len(prof)),
           "ms_from_the_first_ms_without_beam": [-10, 0, 2, 5, 9, 10, 11, 12, 13, 15, 17, 20, 25, 30, 40, 50, 60, 80, 100, 150],
           "median_level": [], "q25": [], "q75": []}
    for a in rec["ms_from_the_first_ms_without_beam"]:
        col = prof[:, 60 + a]
        rec["median_level"].append(float(np.median(col)))
        rec["q25"].append(float(np.quantile(col, 0.25)))
        rec["q75"].append(float(np.quantile(col, 0.75)))
    return match, blip_rows, loose, rec


# ------------------------------------------------------------------------------------------
def render(res):
    g, qn, rp = res["grid"], res["missing_by_bes_level"], res["response"]
    L = ["# How the 10 ms files relate to the raw signals", "",
         "Descriptive, read from the files (`ces_prediction/analyze_file_rules.py`); no model is involved. "
         "Intervals are 95 % intervals over 2,000 resamples of discharges.", "",
         "## 1. Grid", "",
         f"- {g['files']} files, {g['rows_in_files']:,} rows as delivered. Every file opens with a row at t = 0 "
         f"({g['leading_rows_t0_without_ces']} files) and closes with a row at t = 30 s "
         f"({g['trailing_rows_without_ces']} files) that carry no CES value; the provider's note says to ignore "
         f"them. Without them the grid has **{g['grid_rows']:,} rows** in {g['segments']:,} segments "
         f"(breaks of at least {SEG_BREAK_S} s).",
         "- Time stamp modulo 10 ms: " + ", ".join(f"{k} ms on {v:,} rows" for k, v in g["stamp_ms_modulo_10"].items())
         + ". Every time stamp lies 2 ms after a multiple of 10 ms.",
         "- Steps between rows: " + "; ".join(f"{k} ms: {v:,}" for k, v in g["steps_ms"].items())
         + f"; {g['steps_60_to_500_ms']} steps between 60 and 500 ms, {g['steps_ge_500_ms']} of 500 ms or more. "
         f"{100 * g['share_of_steps_10_ms']:.2f} % of the steps are 10 ms. Steps of 20, 30 and 40 ms are one, two "
         f"and three absent frames: {g['absent_frames']:,} absent frames in "
         f"{g['steps_ms']['20'] + g['steps_ms']['30'] + g['steps_ms']['40']:,} steps.",
         f"- Missing on this grid: T_i {g['missing_pct']['CES_TI']:.2f} %, V_rot {g['missing_pct']['CES_VT']:.2f} % "
         "(the earlier ledger divided by the rows as delivered and counted the two edge rows as missing: 8.18 % "
         "and 23.91 %).", "",
         "## 2. T_i missing rate by quintile of the BES level", "",
         f"BES level = mean of the nine channels, standardised per discharge; {qn['rows']:,} rows.", "",
         "| quintile | missing | 95 % interval |", "|---|---|---|"]
    for i in range(5):
        L.append(f"| {i + 1} | {qn['missing_pct_by_quintile'][i]:.1f} % | "
                 f"[{qn['ci95_pct'][i][0]:.1f}, {qn['ci95_pct'][i][1]:.1f}] |")
    L += ["", "## 3. Fast-channel values of a row against the raw sample at its time stamp", "",
          "Share of rows whose values, all channels of the diagnostic at once, equal a raw sample within one raw "
          "sample of the time stamp, at the five decimals the files print.", "",
          "| discharge | BES (rate, rows, share) | ECEI | Mirnov |", "|---|---|---|---|"]
    for shot, m in res["raw_match"].items():
        L.append(f"| {shot} | " + " | ".join(
            f"{m[d]['fs_hz'] / 1e3:.0f} kHz, {m[d]['rows']}, {m[d]['share_equal_within_one_sample']:.3f}"
            for d in ("BES", "ECEI", "MC")) + " |")
    tot = {d: (sum(m[d]["rows"] * m[d]["share_equal_within_one_sample"] for m in res["raw_match"].values()),
               sum(m[d]["rows"] for m in res["raw_match"].values())) for d in ("BES", "ECEI", "MC")}
    L += ["", "All discharges: " + ", ".join(
        f"{d} {round(a):,} of {b:,} rows ({100 * a / b:.1f} %)" for d, (a, b) in tot.items()) + ". "
          "The fast channels of a row are single raw samples taken at the time stamp, not averages over the "
          "CES exposure.", "",
          "## 4. Beam blips in the raw BES against the absent frames", "",
          "A blip is a stretch of at least 3 ms in which the mean BES level (0.1 ms bins) is below half of its "
          "200 ms running median.", ""]
    br = res["blips"]
    one = [b for b in br if b["absent_frames"] == 1 and b["duration_ms"] is not None]
    more = [b for b in br if b["absent_frames"] > 1 and b["duration_ms"] is not None]
    none = [b for b in br if b["duration_ms"] is None]

    def qs(key, rows):
        v = np.array([r[key] for r in rows])
        return f"median {np.median(v):+.1f} ms (range {v.min():+.1f} to {v.max():+.1f})"

    L += [f"- Absent frames inside the raw windows: {len(br)} groups ({len(one) + len(more)} with a blip, "
          f"{len(none)} without).",
          f"- Single absent frames ({len(one)}): the blip starts {qs('start_minus_first_stamp_ms', one)} from the "
          f"absent time stamp and ends {qs('end_minus_last_stamp_ms', one)} from it; duration "
          f"{qs('duration_ms', one).replace('+', '')}."]
    if more:
        L.append(f"- Groups of two or three absent frames ({len(more)}): start "
                 f"{qs('start_minus_first_stamp_ms', more)} from the first absent stamp, end "
                 f"{qs('end_minus_last_stamp_ms', more)} from the last; duration {qs('duration_ms', more).replace('+', '')}.")
    L += ["- The frame that is absent is the one exposed without beam, so the time stamp of a row lies within "
          "about 2 ms of the start of its 10 ms exposure.", "",
          "Beam interruptions that removed no frame (rows whose exposure, stamp - 2 ms to stamp + 8 ms, lies "
          "inside the interruption):", "",
          "| discharge | interruption (s) | duration | T_i in rows exposed without beam (eV) | median T_i in the 70 ms before |",
          "|---|---|---|---|---|"]
    for x in res["interruptions_without_absent_frame"]:
        vals = " / ".join("none" if v is None else f"{v:,.0f}" for v in x["t_i_in_those_rows_eV"]) or "(no row)"
        bef = "n/a" if x["t_i_median_before_eV"] is None else f"{x['t_i_median_before_eV']:,.0f}"
        L.append(f"| {x['shot']} | {x['start_s']:.4f} to {x['end_s']:.4f} | {x['duration_ms']:.1f} ms | {vals} | {bef} |")
    L += ["", "## 5. Rows after an absent frame, all files", "",
          f"{rp['events']:,} absent-frame events with six contiguous rows before them, in {rp['discharges']} "
          "discharges (by number of absent frames: " + ", ".join(
              f"{k}: {v:,}" for k, v in rp["events_by_absent_frames"].items()) + "). Each value is the ratio to "
          "the median of the six rows before the event; V_rot uses values that differ from the previous row "
          "only.", "",
          "| row | T_i | ECEI level | fresh V_rot | BES level | T_i observed |", "|---|---|---|---|---|---|"]

    def cell(c, key="median"):
        return f"{c[key]:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"

    for k, r in rp["rows"].items():
        L.append(f"| {int(k):+d} | {cell(r['ti'])} | {cell(r['ecei'])} | {cell(r['vt'])} | {cell(r['bes'])} | "
                 f"{cell(r['ti_observed_share'], 'mean')} |")
    md_ = rp["mean_deficit_rows_1_to_6"]
    rc = rp["rank_correlation_ti_ecei_rows_1_to_4"]
    L += ["", f"- Mean deficit over rows +1 to +6: T_i {md_['t_i']:.3f} [{md_['t_i_ci95'][0]:.3f}, "
          f"{md_['t_i_ci95'][1]:.3f}], ECEI {md_['ecei']:.3f} [{md_['ecei_ci95'][0]:.3f}, {md_['ecei_ci95'][1]:.3f}]; "
          f"ratio {md_['ratio']:.2f} [{md_['ratio_ci95'][0]:.2f}, {md_['ratio_ci95'][1]:.2f}].",
          f"- Across events, rank correlation of the T_i ratio and the ECEI ratio (means of rows +1 to +4): "
          f"{rc['rho']:+.3f} [{rc['ci95'][0]:+.3f}, {rc['ci95'][1]:+.3f}] ({rc['events']:,} events).",
          "- T_i ratio (rows +1 to +4) by ECEI ratio: " + "; ".join(
              f"{x['ecei_ratio_from']:.2f}-{x['to']:.2f}: {x['median_ti_ratio']:.3f} (n {x['events']})"
              for x in rp["ti_ratio_by_ecei_ratio"]) + ".",
          "- By T_i before the event, rows +1/+2/+3: at least 700 eV " + " / ".join(
              f"{rp['rows'][str(k)]['ti_before_ge_700eV']['median']:.3f}" for k in (1, 2, 3))
          + "; below 400 eV " + " / ".join(
              f"{rp['rows'][str(k)]['ti_before_lt_400eV']['median']:.3f}" for k in (1, 2, 3)) + ".", "",
          "## 6. Raw BES level around a blip (1 ms bins, ratio to the 50 ms before)", "",
          f"{res['recovery']['blips']} blips with a single absent frame.", "",
          "| ms from the first ms without beam | median level | quartiles |", "|---|---|---|"]
    rv = res["recovery"]
    for a, m_, q1, q3 in zip(rv["ms_from_the_first_ms_without_beam"], rv["median_level"], rv["q25"], rv["q75"]):
        L.append(f"| {a:+d} | {m_:.3f} | {q1:.3f} to {q3:.3f} |")
    return "\n".join(L) + "\n"


def main():
    if "--render" in sys.argv:
        OUT_MD.write_text(render(json.loads(OUT_JSON.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
        print("rendered", OUT_MD)
        return
    files = sorted(DATA.glob("s*.csv"))
    if len(files) != 641:
        raise SystemExit(f"FATAL: expected 641 CSV files in {DATA}, found {len(files)}")
    grid, quint, resp = all_files(files)
    print("all files done", flush=True)
    match, blip_rows, loose, rec = raw_side()
    res = {"note": "descriptive; read from the 10 ms files and the raw 1 us deliveries; no model",
           "grid": grid, "missing_by_bes_level": quint, "raw_match": match, "blips": blip_rows,
           "interruptions_without_absent_frame": loose, "response": resp, "recovery": rec}
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
