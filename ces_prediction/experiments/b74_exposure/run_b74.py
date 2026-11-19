"""B.74 (PREREGISTRATION_B74.md, PREREGISTRATION_B74_ADDENDUM1.md): what is inside the exposure of a frame
whose reported T_i departs for one frame, read on the raw BES stream (with the raw ECEI and Mirnov streams beside
it) of the twelve discharges with a raw delivery.  CPU only, read-only; nothing is trained, no estimator is used.

Run from the repo root, one stage at a time (each stage resumes from data/.b74/):
    py ces_prediction/experiments/b74_exposure/run_b74.py check     # registration section 7
    py ces_prediction/experiments/b74_exposure/run_b74.py prep      # per-discharge row tables
    py ces_prediction/experiments/b74_exposure/run_b74.py readout   # docs/paper/b74_exposure.{json,md}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b31_bursts"))
import run_b31 as R31  # noqa: E402  (detector, blip rule and constants of B.31)

REPO = R31.R.REPO
DATA = R31.R.DATA
RAW = R31.RAW
CACHE = DATA / ".b74"
OUT_JSON = REPO / "docs" / "paper" / "b74_exposure.json"
OUT_MD = REPO / "docs" / "paper" / "b74_exposure.md"
B31_JSON = REPO / "docs" / "paper" / "b31_bursts.json"

SHOTS = R31.SHOTS
DISCOVERY = R31.DISCOVERY
STAT_SHOTS = tuple(s for s in SHOTS if s != DISCOVERY)
SEED, N_BOOT, CHUNK = 20261174, 10000, 250
WINDOWS = {"W10": (-0.002, 0.008), "W5": (0.0, 0.005)}
P_BEFORE, Q_BEFORE = 0.010, 0.030
TI_CUT, BLOCK_GAP = 3000.0, 0.5
STEP, STEP_TOL, N_BEFORE = 0.010, 0.001, 10
MIN_TI_BEFORE, MIN_VT_BEFORE, VT_LEVEL_MIN, ECEI_LEVEL_MIN = 6, 3, 20.0, 0.02
SPAN_BEFORE, SPAN_AFTER = 0.200, 0.050
UP, DOWN, BACK = 1.35, 0.74, 0.15
FOLLOW_ROWS, FOLLOW_MIN, FOLLOW_TOL = 5, 3, 0.003
D_STARTS = tuple(range(-10, 15))
D_LENGTHS = (1, 2, 5)
D_SPAN = (-0.010, 0.019)
PRINT_TOL = 0.6e-5
MC_BIN = 1e-4
CLS = {"B": 1, "P": 2, "Q": 3, "O": 0}
CACHE_VERSION = 1


# ---------------------------------------------------------------------------------------------------- helpers
def save_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def idx_at(a, t0, fs):
    """Index of the first raw sample at or after time a (sample i lies at t0 + i / fs)."""
    return np.ceil((np.asarray(a, dtype=np.float64) - t0) * fs - 1e-6).astype(np.int64)


def read_file(shot):
    """Grid rows of one 10 ms file: first and last row dropped (analyze_file_rules.load), D1 T_i."""
    with open(DATA / f"s{shot}.csv", newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        a = np.array([[float(v) if v != "" else np.nan for v in r] for r in rd], dtype=np.float64)
    col = {c: i for i, c in enumerate(head)}
    bes = [c for c in head if c.startswith("BES_")]
    ecei = [c for c in head if c.startswith("ECEI_")]
    mc = [c for c in head if c.startswith("MC")]
    t, ti, vt = a[:, col["time"]], a[:, col["CES_TI"]], a[:, col["CES_VT"]]
    first = bool(t[0] == 0.0 and np.isnan(ti[0]) and np.isnan(vt[0]))
    last = bool(len(t) > 1 and t[-1] - t[-2] >= BLOCK_GAP and np.isnan(ti[-1]) and np.isnan(vt[-1]))
    a = a[int(first):len(a) - int(last)]
    fast = [col[c] for c in bes + ecei + mc]
    keep = np.isfinite(a[:, col["time"]]) & np.all(np.isfinite(a[:, fast]), axis=1)
    n_dropped_fast = int((~keep).sum())
    a = a[keep]
    t = a[:, col["time"]].copy()
    ti_raw = a[:, col["CES_TI"]].copy()
    vt = a[:, col["CES_VT"]].copy()
    # D1: 3 keV cut, then held repeats (KSTAR_CES_Dataset._stuck_repeat_mask, same comparison)
    ti = ti_raw.copy()
    ti[ti > TI_CUT] = np.nan
    block = np.concatenate(([0], np.cumsum(np.diff(t) >= BLOCK_GAP)))
    obs = np.flatnonzero(np.isfinite(ti))
    n_held = 0
    if obs.size > 1:
        dup = (ti[obs[1:]] == ti[obs[:-1]]) & (block[obs[1:]] == block[obs[:-1]])
        n_held = int(dup.sum())
        ti[obs[1:][dup]] = np.nan
    prev = np.concatenate(([np.nan], vt[:-1]))
    fresh = np.isfinite(vt) & ~(vt == prev)
    return {"t": t, "ti": ti, "ti_raw": ti_raw, "vt": vt, "fresh": fresh, "block": block,
            "bes": a[:, [col[c] for c in bes]], "bes_cols": bes,
            "ecei_s": a[:, [col[c] for c in ecei]].mean(1), "first_dropped": first, "last_dropped": last,
            "n_dropped_fast": n_dropped_fast, "n_ti_cut": int(np.sum(ti_raw > TI_CUT)), "n_ti_held": n_held}


def bins_of(lvl32, fs, t0):
    """The binned array mean, its running median and residual exactly as run_b31.bursts_and_blips."""
    x = lvl32.astype(np.float64)
    w = int(round(fs * R31.BIN_S))
    nb = len(x) // w
    b = x[:nb * w].reshape(nb, w).mean(1)
    tb = t0 + (np.arange(nb) + 0.5) * R31.BIN_S
    pad = np.pad(b, (R31.MED_BINS // 2, R31.MED_BINS - 1 - R31.MED_BINS // 2), mode="edge")
    base = np.median(sliding_window_view(pad, R31.MED_BINS), axis=1)
    res = b - base
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    peak = res > R31.K_SIGMA * sig
    d = np.diff(np.concatenate(([0], peak.astype(int), [0])))
    ev = tb[np.flatnonzero(d == 1)]
    if len(ev):
        ev = ev[np.concatenate(([True], np.diff(ev) > R31.MERGE_S))]
    return tb, base, res, ev


# ---------------------------------------------------------------------------------------------------- check
def stage_check():
    CACHE.mkdir(parents=True, exist_ok=True)
    ref = json.loads(B31_JSON.read_text(encoding="utf-8"))
    out = {}
    for s in SHOTS:
        pj = CACHE / f"s{s}_check.json"
        if pj.exists():
            out[str(s)] = json.loads(pj.read_text(encoding="utf-8"))
            print(s, "check cached", out[str(s)]["bursts"], out[str(s)]["bes_sample_share"], flush=True)
            continue
        raw = R31.bursts_and_blips(s)
        bursts, blips = np.asarray(raw["bursts"], np.float64), np.asarray(raw["blips"], np.float64)
        t0_r, t_end_r = float(raw["t0"]), float(raw["t_end"])
        del raw
        gc.collect()
        f = read_file(s)
        with np.load(RAW / f"s{s}_BES.npz", allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            x = z["x"]
        fs, t0 = float(meta["fs_hz"]), float(meta["t0"])
        chans = list(meta["channels"])
        ci = [chans.index(c) for c in f["bes_cols"]]
        idx = np.rint((f["t"] - t0) * fs).astype(np.int64)
        ok = (idx >= 1) & (idx < x.shape[0] - 1)
        hit = np.stack([np.abs(x[idx[ok] + o][:, ci].astype(np.float64) - f["bes"][ok]).max(1) <= PRINT_TOL
                        for o in (-1, 0, 1)], 1)
        lvl32 = x.mean(1)
        n_raw = int(x.shape[0])
        del x
        gc.collect()
        _tb, _base, _res, ev = bins_of(lvl32, fs, t0)
        same = bool(len(ev) == len(bursts) and np.array_equal(ev, bursts))
        save_npz(CACHE / f"s{s}_raw.npz", lvl32=lvl32, bursts=bursts, blips=blips,
                 meta=np.array(json.dumps({"fs": fs, "t0": t0, "t_end": t_end_r, "t0_b31": t0_r, "n": n_raw})))
        del lvl32
        gc.collect()
        d = {"bursts": int(len(bursts)), "bursts_b31_md": int(ref["discharges"][str(s)]["bursts"]),
             "counts_equal": bool(len(bursts) == ref["discharges"][str(s)]["bursts"]),
             "recomputed_onsets_equal": same, "bes_rows_in_raw_span": int(ok.sum()),
             "bes_sample_share": float(hit.any(1).mean()) if ok.sum() else None,
             "bes_share_at_offset_-1_0_+1": [float(h) for h in hit.mean(0)] if ok.sum() else None,
             "fs_hz": fs, "t0": t0, "t_end": t_end_r, "blip_ms_bins": int(len(blips))}
        save_json(pj, d)
        out[str(s)] = d
        print(s, "bursts", d["bursts"], "b31", d["bursts_b31_md"], "recomputed equal", same,
              "BES sample share", d["bes_sample_share"], flush=True)
    bad = [s for s, d in out.items() if not (d["counts_equal"] and d["recomputed_onsets_equal"])]
    bad2 = [s for s, d in out.items() if d["bes_sample_share"] is None or d["bes_sample_share"] < 0.99]
    save_json(CACHE / "checks.json", {"discharges": out, "burst_check_failed": bad, "sample_check_failed": bad2,
                                     "passed": not bad and not bad2})
    if bad or bad2:
        raise SystemExit(f"STOP: burst check failed for {bad}; sample check failed for {bad2}")
    print("checks passed", flush=True)


# ---------------------------------------------------------------------------------------------------- prep
def window_classes(t, ev, lo, hi):
    a, b = t + lo, t + hi
    n_in = np.searchsorted(ev, b, "left") - np.searchsorted(ev, a, "left")
    n_p = np.searchsorted(ev, a, "left") - np.searchsorted(ev, a - P_BEFORE, "left")
    n_q = np.searchsorted(ev, a, "left") - np.searchsorted(ev, a - Q_BEFORE, "left")
    cls = np.full(len(t), CLS["O"], np.int8)
    cls[n_in > 0] = CLS["B"]
    cls[(n_in == 0) & (n_p > 0)] = CLS["P"]
    cls[(n_in == 0) & (n_q == 0)] = CLS["Q"]
    first = np.full(len(t), np.nan)
    j = np.searchsorted(ev, a, "left")
    m = n_in > 0
    first[m] = ev[j[m]]
    return cls, first


def stage_prep():
    chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    if not chk["passed"]:
        raise SystemExit("STOP: checks did not pass")
    for s in SHOTS:
        out_p = CACHE / f"s{s}_rows.npz"
        if out_p.exists():
            with np.load(out_p) as z:
                if int(z["version"]) == CACHE_VERSION:
                    print(s, "rows cached", flush=True)
                    continue
        f = read_file(s)
        t, ti, vt, fresh, block = f["t"], f["ti"], f["vt"], f["fresh"], f["block"]
        n = len(t)
        # ---- ratios on the 10 ms rows
        okstep = np.abs(np.diff(t) - STEP) < STEP_TOL
        cs = np.concatenate(([0], np.cumsum(okstep)))
        regular = np.zeros(n, bool)
        k = np.arange(N_BEFORE, n)
        regular[k] = (cs[k] - cs[k - N_BEFORE]) == N_BEFORE
        ti_lb, vt_lb, ec_lb = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        n_ti_before = np.zeros(n, np.int16)
        n_vt_before = np.zeros(n, np.int16)
        for i in np.flatnonzero(regular):
            w = ti[i - N_BEFORE:i]
            w = w[np.isfinite(w)]
            n_ti_before[i] = len(w)
            if len(w) >= MIN_TI_BEFORE:
                ti_lb[i] = np.median(w)
            w = vt[i - N_BEFORE:i][fresh[i - N_BEFORE:i]]
            n_vt_before[i] = len(w)
            if len(w) >= MIN_VT_BEFORE:
                m_ = np.median(w)
                if m_ > VT_LEVEL_MIN:
                    vt_lb[i] = m_
            m_ = np.median(f["ecei_s"][i - N_BEFORE:i])
            if m_ > ECEI_LEVEL_MIN:
                ec_lb[i] = m_
        ti_ratio = ti / ti_lb
        vt_ratio = np.where(fresh, vt / vt_lb, np.nan)
        ec_ratio = f["ecei_s"] / ec_lb
        later = np.full((n, FOLLOW_ROWS), np.nan)
        for d in range(1, FOLLOW_ROWS + 1):
            i = np.arange(0, n - d)
            dt = t[i + d] - t[i]
            j = np.rint(dt / STEP).astype(np.int64)
            ok = (block[i + d] == block[i]) & (j >= 1) & (j <= FOLLOW_ROWS) & (np.abs(dt - STEP * j) < FOLLOW_TOL)
            later[i[ok], j[ok] - 1] = ti[i[ok] + d]
        n_later = np.isfinite(later).sum(1)
        l5 = np.full(n, np.nan)
        m = n_later >= FOLLOW_MIN
        l5[m] = np.nanmedian(later[m], 1)
        nxt = np.full(n, np.nan)
        for j in range(FOLLOW_ROWS - 1, -1, -1):
            v = later[:, j]
            nxt = np.where(np.isfinite(v), v, nxt)
        l5_ratio = l5 / ti_lb
        next_ratio = nxt / ti_lb

        # ---- raw BES: bursts, blips, span, classes, excess light, part D
        with np.load(CACHE / f"s{s}_raw.npz") as z:
            lvl32 = z["lvl32"]
            bursts, blips = z["bursts"], z["blips"]
            meta = json.loads(str(z["meta"]))
        fs, t0, t_end = meta["fs"], meta["t0"], meta["t_end"]
        raw31 = {"t0": meta["t0_b31"], "t_end": t_end, "bursts": bursts, "blips": blips}
        _lab, _near, blip = R31.labels(t, raw31)
        in_span = (t - SPAN_BEFORE >= t0) & (t + SPAN_AFTER <= t_end)
        rowset = in_span & ~blip
        blip_in_w10 = np.zeros(n, bool)
        blip_in_dspan = np.zeros(n, bool)
        for arr, (lo, hi) in ((blip_in_w10, WINDOWS["W10"]), (blip_in_dspan, D_SPAN)):
            # a 1 ms blip bin (centre c) overlaps [t+lo, t+hi) when c + 0.5 ms > t+lo and c - 0.5 ms < t+hi
            nb_ = np.searchsorted(blips, t + hi + 0.0005, "left") - np.searchsorted(blips, t + lo - 0.0005, "right")
            arr[:] = nb_ > 0
        tb, base, res, ev = bins_of(lvl32, fs, t0)
        if not np.array_equal(ev, bursts):
            raise SystemExit(f"STOP: recomputed onsets differ from bursts_and_blips on {s}")
        cpos = np.concatenate(([0.0], np.cumsum(np.maximum(res, 0.0))))
        cbase = np.concatenate(([0.0], np.cumsum(base)))
        cls, excess, first_onset = {}, {}, {}
        for wn, (lo, hi) in WINDOWS.items():
            cls[wn], first_onset[wn] = window_classes(t, bursts, lo, hi)
            ia = np.searchsorted(tb, t + lo, "left")
            ib = np.searchsorted(tb, t + hi, "left")
            with np.errstate(invalid="ignore", divide="ignore"):
                excess[wn] = (cpos[ib] - cpos[ia]) / (cbase[ib] - cbase[ia])
            excess[wn][~in_span] = np.nan
        del tb, base, res, cpos, cbase
        lvl = lvl32.astype(np.float64)
        del lvl32
        csl = np.concatenate(([0.0], np.cumsum(lvl)))
        del lvl
        gc.collect()
        n_raw = len(csl) - 1
        dr = np.full((n, len(D_STARTS), len(D_LENGTHS)), np.nan)
        rows_d = np.flatnonzero(rowset & regular)

        def wmean(a_, L_):
            ia = idx_at(a_, t0, fs)
            ib = idx_at(a_ + L_, t0, fs)
            if np.any(ia < 0) or np.any(ib > n_raw) or np.any(ib <= ia):
                raise SystemExit(f"STOP: window outside the raw BES span on {s}")
            return (csl[ib] - csl[ia]) / (ib - ia)

        if len(rows_d):
            for si, st in enumerate(D_STARTS):
                for li, L in enumerate(D_LENGTHS):
                    own = wmean(t[rows_d] + st * 1e-3, L * 1e-3)
                    refm = np.mean([wmean(t[rows_d - j] + st * 1e-3, L * 1e-3) for j in range(1, N_BEFORE + 1)], axis=0)
                    dr[rows_d, si, li] = own / refm
        del csl
        gc.collect()

        # ---- raw ECEI: change of the array mean across the first onset inside the window
        with np.load(RAW / f"s{s}_ECEI.npz", allow_pickle=True) as z:
            em = json.loads(str(z["meta"]))
            xe = z["x"]
        e_fs, e_t0 = float(em["fs_hz"]), float(em["t0"])
        lv = xe.mean(1, dtype=np.float64)
        del xe
        ce = np.concatenate(([0.0], np.cumsum(lv)))
        e_n = len(lv)
        del lv
        gc.collect()

        def ecei_step(tc):
            out = np.full(len(tc), np.nan)
            fin = np.isfinite(tc)
            tc = np.where(fin, tc, e_t0)
            ia, im, ib = idx_at(tc - 0.001, e_t0, e_fs), idx_at(tc, e_t0, e_fs), idx_at(tc + 0.001, e_t0, e_fs)
            ok_ = fin & (ia >= 0) & (ib <= e_n) & (im > ia) & (ib > im)
            pre = (ce[im[ok_]] - ce[ia[ok_]]) / (im[ok_] - ia[ok_])
            post = (ce[ib[ok_]] - ce[im[ok_]]) / (ib[ok_] - im[ok_])
            v = np.where(pre > ECEI_LEVEL_MIN, (post - pre) / np.where(pre > ECEI_LEVEL_MIN, pre, 1.0), np.nan)
            out[np.flatnonzero(ok_)] = v
            return out

        ecei_onset, ecei_stamp = {}, ecei_step(t)
        for wn in WINDOWS:
            ecei_onset[wn] = ecei_step(first_onset[wn])
        del ce
        gc.collect()

        # ---- raw Mirnov: RMS over the window (window mean removed) against the ten windows before
        with np.load(RAW / f"s{s}_MC.npz", allow_pickle=True) as z:
            mm = json.loads(str(z["meta"]))
            xm = z["x"]
        m_fs, m_t0 = float(mm["fs_hz"]), float(mm["t0"])
        wbin = int(round(m_fs * MC_BIN))
        nbm = xm.shape[0] // wbin
        s1 = np.zeros((nbm, xm.shape[1]))
        s2 = np.zeros((nbm, xm.shape[1]))
        step_b = max(1, 1_000_000 // wbin)
        for c in range(xm.shape[1]):
            for b0 in range(0, nbm, step_b):
                b1 = min(b0 + step_b, nbm)
                seg = xm[b0 * wbin:b1 * wbin, c].astype(np.float64).reshape(b1 - b0, wbin)
                s1[b0:b1, c] = seg.sum(1)
                s2[b0:b1, c] = (seg * seg).sum(1)
                del seg
        del xm
        gc.collect()
        c1 = np.vstack([np.zeros((1, s1.shape[1])), np.cumsum(s1, 0)])
        c2 = np.vstack([np.zeros((1, s2.shape[1])), np.cumsum(s2, 0)])
        del s1, s2
        off = (np.asarray(t) - m_t0) / MC_BIN
        mc_misalign = float(np.max(np.abs(off - np.rint(off))[rowset])) if rowset.any() else 0.0
        if mc_misalign > 0.05:
            raise SystemExit(f"STOP: Mirnov 0.1 ms bins not aligned with the stamps on {s} ({mc_misalign})")
        mc_ratio = {}
        for wn, (lo, hi) in WINDOWS.items():
            out = np.full(n, np.nan)
            nlo, nhi = int(round(lo / MC_BIN)), int(round(hi / MC_BIN))

            def rms(ib0):
                ib1 = ib0 + (nhi - nlo)
                cnt = (ib1 - ib0) * wbin
                mu = (c1[ib1] - c1[ib0]) / cnt
                return np.sqrt(np.maximum((c2[ib1] - c2[ib0]) / cnt - mu * mu, 0.0))

            for i in np.flatnonzero(rowset & regular):
                ib_own = int(round(off[i])) + nlo
                ib_first = int(round(off[i - N_BEFORE])) + nlo
                if ib_first < 0 or ib_own + (nhi - nlo) > nbm:
                    continue
                r_own = rms(ib_own)
                r_ref = np.sqrt(np.mean([rms(int(round(off[i - j])) + nlo) ** 2 for j in range(1, N_BEFORE + 1)], 0))
                if np.all(r_ref > 0):
                    out[i] = float(np.mean(r_own / r_ref))
            mc_ratio[wn] = out
        del c1, c2
        gc.collect()

        save_npz(out_p, version=np.array(CACHE_VERSION), t=t, ti=ti, vt=vt, fresh=fresh, regular=regular,
                 in_span=in_span, blip=blip, rowset=rowset, blip_in_w10=blip_in_w10, blip_in_dspan=blip_in_dspan,
                 ti_lb=ti_lb, vt_lb=vt_lb, ec_lb=ec_lb, n_ti_before=n_ti_before, n_vt_before=n_vt_before,
                 ti_ratio=ti_ratio, vt_ratio=vt_ratio, ec_ratio=ec_ratio, l5=l5, l5_ratio=l5_ratio,
                 next_ratio=next_ratio, n_later=n_later,
                 cls_W10=cls["W10"], cls_W5=cls["W5"], excess_W10=excess["W10"], excess_W5=excess["W5"],
                 onset_W10=first_onset["W10"], onset_W5=first_onset["W5"], d_ratio=dr,
                 ecei_onset_W10=ecei_onset["W10"], ecei_onset_W5=ecei_onset["W5"], ecei_stamp=ecei_stamp,
                 mc_W10=mc_ratio["W10"], mc_W5=mc_ratio["W5"], bursts=bursts,
                 info=np.array(json.dumps({
                     "rows_file": int(n), "first_dropped": f["first_dropped"], "last_dropped": f["last_dropped"],
                     "rows_dropped_missing_fast": f["n_dropped_fast"], "ti_above_3keV": f["n_ti_cut"],
                     "ti_held": f["n_ti_held"], "fs_bes": fs, "t0_bes": t0, "t_end_bes": t_end,
                     "ecei_span": [e_t0, e_t0 + e_n / e_fs], "mc_span": [m_t0, m_t0 + nbm * MC_BIN],
                     "mc_bin_misalignment_max": mc_misalign,
                     "bursts_in_rowset_span": int(np.sum((bursts >= t[in_span].min() - 0.032) & (bursts < t[in_span].max() + 0.008))) if in_span.any() else 0})))
        print(s, "rows", n, "rowset", int(rowset.sum()), "regular", int((rowset & regular).sum()),
              "with T_i ratio", int(np.sum(rowset & np.isfinite(ti_ratio))), flush=True)
        del f, dr
        gc.collect()


# ---------------------------------------------------------------------------------------------------- statistics
class Boot:
    """Discharge resampling: one multinomial weight matrix for every statistic."""

    def __init__(self, n_units):
        rng = np.random.default_rng(SEED)
        self.W = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=N_BOOT).astype(np.float64)

    def share(self, num, den):
        """num, den: weighted counts per unit -> (point, draws)."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num.sum() / den.sum() if den.sum() else np.nan), (self.W @ num) / (self.W @ den)

    def median(self, v, g):
        v, g = np.asarray(v, float), np.asarray(g, int)
        nd = self.W.shape[0]
        if len(v) == 0:
            return np.nan, np.full(nd, np.nan)
        o = np.argsort(v, kind="stable")
        vs, gs = v[o], g[o]
        out = np.empty(nd)
        for c0 in range(0, nd, CHUNK):
            rw = self.W[c0:c0 + CHUNK][:, gs]
            cw = np.cumsum(rw, 1)
            tot = cw[:, -1]
            p1, p2 = np.floor((tot - 1) / 2), np.floor(tot / 2)
            i1 = np.minimum((cw <= p1[:, None]).sum(1), len(vs) - 1)
            i2 = np.minimum((cw <= p2[:, None]).sum(1), len(vs) - 1)
            r = 0.5 * (vs[i1] + vs[i2])
            r[tot == 0] = np.nan
            out[c0:c0 + CHUNK] = r
        return float(np.median(v)), out

    def spearman(self, x, y, g):
        x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g, int)
        pt = spearman_point(x, y)
        ox, oy = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
        ix, iy = np.empty_like(ox), np.empty_like(oy)
        ix[ox] = np.arange(len(x))
        iy[oy] = np.arange(len(y))
        nd = self.W.shape[0]
        out = np.empty(nd)
        for c0 in range(0, nd, CHUNK):
            rw = self.W[c0:c0 + CHUNK][:, g]
            wx = rw[:, ox]
            rx = (np.cumsum(wx, 1) - 0.5 * (wx - 1.0))[:, ix]
            wy = rw[:, oy]
            ry = (np.cumsum(wy, 1) - 0.5 * (wy - 1.0))[:, iy]
            tot = rw.sum(1)
            mu = 0.5 * (tot + 1.0)
            dx, dy = rx - mu[:, None], ry - mu[:, None]
            num = (rw * dx * dy).sum(1)
            den = np.sqrt((rw * dx * dx).sum(1) * (rw * dy * dy).sum(1))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c0:c0 + CHUNK] = num / den
        return pt, out


def avg_rank(v):
    o = np.argsort(v, kind="stable")
    vs = v[o]
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and vs[j + 1] == vs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman_point(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = avg_rank(x), avg_rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def pack(pt, draws, per=None, n=None):
    d = np.asarray(draws, float)
    ok = np.isfinite(d)
    out = {"point": None if pt is None or not np.isfinite(pt) else float(pt),
           "lo": float(np.quantile(d[ok], 0.025)) if ok.sum() else None,
           "hi": float(np.quantile(d[ok], 0.975)) if ok.sum() else None,
           "draws_used": int(ok.sum())}
    if n is not None:
        out["n"] = int(n)
    if per is not None:
        out["per_discharge"] = per
    return out


# ---------------------------------------------------------------------------------------------------- readout
def stage_readout():
    tabs = {}
    for s in SHOTS:
        with np.load(CACHE / f"s{s}_rows.npz") as z:
            if int(z["version"]) != CACHE_VERSION:
                raise SystemExit(f"STOP: stale row table for {s}")
            tabs[s] = {k: z[k] for k in z.files}
            tabs[s]["info"] = json.loads(str(z["info"]))
    chk = json.loads((CACHE / "checks.json").read_text(encoding="utf-8"))
    U = len(STAT_SHOTS)
    bt = Boot(U)
    res = {"registration": "ces_prediction/experiments/PREREGISTRATION_B74.md",
           "addendum": "ces_prediction/experiments/PREREGISTRATION_B74_ADDENDUM1.md",
           "discharges_in_statements": list(STAT_SHOTS), "discovery_discharge": DISCOVERY,
           "seed": SEED, "draws": N_BOOT, "checks": chk}

    def cat(key, shots=STAT_SHOTS):
        return np.concatenate([tabs[s][key] for s in shots])

    gidx = np.concatenate([np.full(len(tabs[s]["t"]), u) for u, s in enumerate(STAT_SHOTS)])
    A = {k: cat(k) for k in ("ti", "rowset", "regular", "ti_ratio", "vt_ratio", "ec_ratio", "l5_ratio",
                             "next_ratio", "cls_W10", "cls_W5", "excess_W10", "excess_W5", "mc_W10", "mc_W5",
                             "ecei_onset_W10", "ecei_onset_W5", "ecei_stamp", "blip_in_w10", "blip_in_dspan")}
    A["d_ratio"] = cat("d_ratio")
    rowset = A["rowset"].astype(bool)
    popB = rowset & np.isfinite(A["ti_ratio"])

    # ---------------- self-check of the resampling helpers: the first 20 draws against the expanded samples
    sub = Boot.__new__(Boot)
    sub.W = bt.W[:20]
    mchk = popB & (A["cls_W10"] == CLS["B"])
    xv = A["d_ratio"][:, D_STARTS.index(0), D_LENGTHS.index(5)]
    mspr = popB & np.isin(A["cls_W10"], (CLS["B"], CLS["Q"])) & np.isfinite(xv)
    _p, dm = sub.median(A["ti_ratio"][mchk], gidx[mchk])
    _p, dsp = sub.spearman(xv[mspr], A["ti_ratio"][mspr], gidx[mspr])
    dev_m = dev_s = 0.0
    for r in range(20):
        rep = np.repeat(np.flatnonzero(mchk), sub.W[r][gidx[mchk]].astype(int))
        dev_m = max(dev_m, abs(dm[r] - np.median(A["ti_ratio"][rep])))
        rep = np.repeat(np.flatnonzero(mspr), sub.W[r][gidx[mspr]].astype(int))
        dev_s = max(dev_s, abs(dsp[r] - spearman_point(xv[rep], A["ti_ratio"][rep])))
    res["resampling_self_check"] = {"draws": 20, "max_abs_dev_median": float(dev_m), "max_abs_dev_spearman": float(dev_s)}
    print("self-check", res["resampling_self_check"], flush=True)
    if dev_m > 1e-12 or dev_s > 1e-3:
        raise SystemExit("STOP: resampling helpers disagree with the expanded samples")

    def per_disc(fn):
        """fn(table of one discharge) -> number; for the eleven and for 31921."""
        out = {}
        for s in SHOTS:
            try:
                v = fn(tabs[s])
            except (ValueError, IndexError, ZeroDivisionError):
                v = None
            out[str(s)] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
        return out

    def counts_per_unit(mask):
        return np.bincount(gidx[mask], minlength=U).astype(float)

    # ---------------- per-discharge helpers working on one table
    def T_popB(T):
        return T["rowset"].astype(bool) & np.isfinite(T["ti_ratio"])

    def T_med(T, vkey, m):
        m = m & np.isfinite(T[vkey])
        return float(np.median(T[vkey][m])) if m.sum() else None

    # ================= Part A
    partA = {}
    for wn in WINDOWS:
        c = A[f"cls_{wn}"]
        partA[wn] = {}
        draws = {}
        for cn in ("B", "P", "Q"):
            m = rowset & (c == CLS[cn])
            num = counts_per_unit(m & ~np.isfinite(A["ti"]))
            den = counts_per_unit(m)
            pt, dr = bt.share(num, den)
            draws[cn] = (pt, dr)
            partA[wn][cn] = pack(pt, dr, per_disc(lambda T, cn=cn, wn=wn: (
                float(np.mean(~np.isfinite(T["ti"][T["rowset"].astype(bool) & (T[f"cls_{wn}"] == CLS[cn])])))
                if np.sum(T["rowset"].astype(bool) & (T[f"cls_{wn}"] == CLS[cn])) else None)), n=den.sum())
        partA[wn]["B_minus_Q"] = pack(draws["B"][0] - draws["Q"][0], draws["B"][1] - draws["Q"][1])
        partA[wn]["rows_class_O"] = int(np.sum(rowset & (c == CLS["O"])))
    res["A"] = partA

    # ================= Part B
    partB = {}
    qty = {"ti_ratio": "T_i ratio", "vt_ratio": "fresh V_rot ratio", "ec_ratio": "ECEI ratio",
           "l5_ratio": "following-level ratio"}
    for wn in WINDOWS:
        c = A[f"cls_{wn}"]
        partB[wn] = {}
        for vk in qty:
            partB[wn][vk] = {}
            meds = {}
            for cn in ("B", "P", "Q"):
                m = popB & (c == CLS[cn]) & np.isfinite(A[vk])
                pt, dr = bt.median(A[vk][m], gidx[m])
                meds[cn] = (pt, dr)
                partB[wn][vk][cn] = pack(pt, dr, per_disc(lambda T, cn=cn, wn=wn, vk=vk: T_med(
                    T, vk, T_popB(T) & (T[f"cls_{wn}"] == CLS[cn]))), n=m.sum())
            partB[wn][vk]["B_minus_Q"] = pack(meds["B"][0] - meds["Q"][0], meds["B"][1] - meds["Q"][1],
                                              per_disc(lambda T, wn=wn, vk=vk: (
                                                  None if T_med(T, vk, T_popB(T) & (T[f"cls_{wn}"] == CLS["B"])) is None
                                                  or T_med(T, vk, T_popB(T) & (T[f"cls_{wn}"] == CLS["Q"])) is None
                                                  else T_med(T, vk, T_popB(T) & (T[f"cls_{wn}"] == CLS["B"]))
                                                  - T_med(T, vk, T_popB(T) & (T[f"cls_{wn}"] == CLS["Q"])))))
        partB[wn]["shares"] = {}
        for cn in ("B", "P", "Q"):
            m = popB & (c == CLS[cn])
            den = counts_per_unit(m)
            for nm_, cond in (("above_1.35", A["ti_ratio"] > UP), ("below_0.74", A["ti_ratio"] < DOWN)):
                pt, dr = bt.share(counts_per_unit(m & cond), den)
                partB[wn]["shares"][f"{cn}_{nm_}"] = pack(pt, dr, per_disc(
                    lambda T, cn=cn, wn=wn, nm_=nm_: (lambda mm: float(np.mean(
                        (T["ti_ratio"][mm] > UP) if nm_ == "above_1.35" else (T["ti_ratio"][mm] < DOWN)))
                        if mm.sum() else None)(T_popB(T) & (T[f"cls_{wn}"] == CLS[cn]))), n=den.sum())
    res["B"] = partB

    # ================= Part C
    partC = {}
    for wn in WINDOWS:
        ex = A[f"excess_{wn}"]
        m0 = popB & np.isfinite(ex)
        pos = ex[m0 & (ex > 0)]
        qe = np.quantile(pos, [0.25, 0.5, 0.75]) if len(pos) else np.array([np.nan] * 3)
        edges = [("zero", lambda e: e == 0), ("Q1", lambda e: (e > 0) & (e <= qe[0])),
                 ("Q2", lambda e: (e > qe[0]) & (e <= qe[1])), ("Q3", lambda e: (e > qe[1]) & (e <= qe[2])),
                 ("Q4", lambda e: e > qe[2])]
        partC[wn] = {"quartile_edges_of_positive": [float(q) for q in qe], "rows": int(m0.sum()),
                     "rows_zero": int(np.sum(m0 & (ex == 0))), "bins": {}}
        for bn, fn in edges:
            mb = m0 & fn(np.where(np.isfinite(ex), ex, -1.0))
            cell = {"rows": int(mb.sum())}
            for vk in ("ti_ratio", "l5_ratio"):
                mm = mb & np.isfinite(A[vk])
                pt, dr = bt.median(A[vk][mm], gidx[mm])
                cell[vk] = pack(pt, dr, per_disc(lambda T, vk=vk, wn=wn, fn=fn: T_med(
                    T, vk, T_popB(T) & np.isfinite(T[f"excess_{wn}"])
                    & fn(np.where(np.isfinite(T[f"excess_{wn}"]), T[f"excess_{wn}"], -1.0)))), n=mm.sum())
            partC[wn]["bins"][bn] = cell
    res["C"] = partC

    # ================= Part D
    partD = {}
    for cw in ("W10", "W5"):
        mD = popB & np.isin(A[f"cls_{cw}"], (CLS["B"], CLS["Q"]))
        partD[cw] = {"rows": int(np.sum(mD & np.isfinite(A["d_ratio"][:, 0, 0]))), "cells": {}}
        draws_all = {}
        for li, L in enumerate(D_LENGTHS):
            for si, st in enumerate(D_STARTS):
                xv = A["d_ratio"][:, si, li]
                mm = mD & np.isfinite(xv)
                pt, dr = bt.spearman(xv[mm], A["ti_ratio"][mm], gidx[mm])
                draws_all[(L, st)] = (pt, dr)
                partD[cw]["cells"][f"L{L}_s{st:+d}"] = pack(pt, dr, per_disc(
                    lambda T, si=si, li=li, cw=cw: (lambda m_: spearman_point(
                        T["d_ratio"][m_, si, li], T["ti_ratio"][m_]) if m_.sum() >= 10 else None)(
                        T_popB(T) & np.isin(T[f"cls_{cw}"], (CLS["B"], CLS["Q"])) & np.isfinite(T["d_ratio"][:, si, li]))),
                    n=mm.sum())
            n_ties = int(len(A["ti_ratio"][mD]) - len(np.unique(A["ti_ratio"][mD])))
            partD[cw]["ti_ratio_tied_values"] = n_ties
        pts5 = np.array([draws_all[(5, st)][0] for st in D_STARTS])
        s_star = D_STARTS[int(np.nanargmax(pts5))]
        d5 = np.stack([draws_all[(5, st)][1] for st in D_STARTS], 1)
        okd = np.all(np.isfinite(d5), 1)
        am = np.array(D_STARTS)[np.argmax(np.where(np.isfinite(d5), d5, -np.inf), 1)]
        partD[cw]["J5"] = {
            "s_star_ms": int(s_star), "rho_s_star": float(draws_all[(5, s_star)][0]),
            "rho_minus10": float(draws_all[(5, -10)][0]), "rho_plus10": float(draws_all[(5, 10)][0]),
            "diff_s_star_minus_m10": pack(draws_all[(5, s_star)][0] - draws_all[(5, -10)][0],
                                          draws_all[(5, s_star)][1] - draws_all[(5, -10)][1]),
            "diff_s_star_minus_p10": pack(draws_all[(5, s_star)][0] - draws_all[(5, 10)][0],
                                          draws_all[(5, s_star)][1] - draws_all[(5, 10)][1]),
            "share_draws_argmax_in_-2_+5": float(np.mean((am[okd] >= -2) & (am[okd] <= 5))),
            "argmax_draw_quantiles_ms": [float(q) for q in np.quantile(am[okd], [0.025, 0.5, 0.975])]}
        for L in (1, 2):
            pts = np.array([draws_all[(L, st)][0] for st in D_STARTS])
            partD[cw][f"argmax_L{L}_ms"] = int(D_STARTS[int(np.nanargmax(pts))])
    res["D"] = partD

    # ================= Part E
    partE = {}
    for wn in WINDOWS:
        c = A[f"cls_{wn}"]
        isB = c == CLS["B"]
        den_all = counts_per_unit(popB)
        pt_all, dr_all = bt.share(counts_per_unit(popB & isB), den_all)
        partE[wn] = {"share_B_all_rows": pack(pt_all, dr_all, n=den_all.sum())}
        for nm_, cond in (("up", A["ti_ratio"] > UP), ("down", A["ti_ratio"] < DOWN)):
            dep = popB & cond & np.isfinite(A["next_ratio"]) & (np.abs(A["next_ratio"] - 1.0) <= BACK)
            den = counts_per_unit(dep)
            pt, dr = bt.share(counts_per_unit(dep & isB), den)
            partE[wn][nm_] = {
                "departures": int(dep.sum()), "departures_in_B": int(np.sum(dep & isB)),
                "share_B": pack(pt, dr, per_disc(lambda T, cond_up=(nm_ == "up"), wn=wn: (
                    lambda mm: float(np.mean(T[f"cls_{wn}"][mm] == CLS["B"])) if mm.sum() else None)(
                    T_popB(T) & ((T["ti_ratio"] > UP) if cond_up else (T["ti_ratio"] < DOWN))
                    & np.isfinite(T["next_ratio"]) & (np.abs(T["next_ratio"] - 1.0) <= BACK)))),
                "share_ratio": pack(pt / pt_all if pt_all else np.nan, dr / dr_all)}
        partE[wn]["share_B_all_rows"]["per_discharge"] = per_disc(
            lambda T, wn=wn: float(np.mean(T[f"cls_{wn}"][T_popB(T)] == CLS["B"])) if T_popB(T).sum() else None)
    res["E"] = partE

    # ================= Part F
    partF = {"per_discharge": {}}
    for s in SHOTS:
        T = tabs[s]
        rs = T["rowset"].astype(bool)
        d = {"rows_file": T["info"]["rows_file"], "rows_in_set": int(rs.sum()),
             "rows_regular": int(np.sum(rs & T["regular"].astype(bool))),
             "rows_with_ti_ratio": int(T_popB(T).sum()),
             "rows_left_out_blip": int(np.sum(T["in_span"].astype(bool) & T["blip"].astype(bool))),
             "rows_kept_blip_in_W10": int(np.sum(rs & T["blip_in_w10"].astype(bool))),
             "rows_kept_blip_in_partD_span": int(np.sum(rs & T["blip_in_dspan"].astype(bool))),
             "bursts_raw_stream": int(len(T["bursts"])), "bursts_near_row_set": T["info"]["bursts_in_rowset_span"],
             "info": T["info"]}
        for wn in WINDOWS:
            d[f"classes_{wn}"] = {cn: int(np.sum(rs & (T[f"cls_{wn}"] == CLS[cn]))) for cn in ("B", "P", "Q", "O")}
        partF["per_discharge"][str(s)] = d
    for wn in WINDOWS:
        c = A[f"cls_{wn}"]
        partF[wn] = {}
        for nm_, vk, cn in (("mirnov_B", f"mc_{wn}", "B"), ("mirnov_Q", f"mc_{wn}", "Q"),
                            ("ecei_onset_B", f"ecei_onset_{wn}", "B"), ("ecei_stamp_Q", "ecei_stamp", "Q")):
            m = rowset & (c == CLS[cn]) & np.isfinite(A[vk])
            pt, dr = bt.median(A[vk][m], gidx[m])
            partF[wn][nm_] = pack(pt, dr, per_disc(lambda T, vk=vk, cn=cn, wn=wn: T_med(
                T, vk, T["rowset"].astype(bool) & (T[f"cls_{wn}"] == CLS[cn]))), n=m.sum())
            partF[wn][nm_]["rows_without_value"] = int(np.sum(rowset & (c == CLS[cn]) & ~np.isfinite(A[vk])))
    res["F"] = partF

    # ================= statements
    def above0(p):
        return p["lo"] is not None and p["lo"] > 0

    J = {}
    p = partA["W10"]["B_minus_Q"]
    J["J1"] = {"statistic": "W10, share without a usable T_i, class B minus class Q", **p,
               "expectation": "holds", "holds": bool(above0(p))}
    p = partB["W10"]["ti_ratio"]["B_minus_Q"]
    J["J2"] = {"statistic": "W10, median T_i ratio, class B minus class Q", **{k: v for k, v in p.items() if k != "per_discharge"},
               "expectation": "uncertain; holds if bursts are what raises the width for one frame",
               "holds": bool(above0(p))}
    p = partB["W10"]["l5_ratio"]["B_minus_Q"]
    J["J3"] = {"statistic": "W10, median following-level ratio, class B minus class Q", **{k: v for k, v in p.items() if k != "per_discharge"},
               "expectation": "holds if J2 holds", "holds": bool(p["lo"] is not None and p["lo"] <= 0)}
    p = partB["W10"]["vt_ratio"]["B_minus_Q"]
    J["J4"] = {"statistic": "W10, median fresh V_rot ratio, class B minus class Q", **{k: v for k, v in p.items() if k != "per_discharge"},
               "expectation": "holds", "holds": bool(p["lo"] is not None and p["lo"] >= -0.03 and p["hi"] <= 0.03)}
    j5 = partD["W10"]["J5"]
    J["J5"] = {"statistic": "part D, 5 ms windows, classes of W10: start of the largest rank correlation; differences to the starts -10 and +10 ms",
               "s_star_ms": j5["s_star_ms"], "rho_s_star": j5["rho_s_star"],
               "diff_minus10": j5["diff_s_star_minus_m10"], "diff_plus10": j5["diff_s_star_minus_p10"],
               "expectation": "holds",
               "holds": bool(-2 <= j5["s_star_ms"] <= 5 and above0(j5["diff_s_star_minus_m10"])
                             and above0(j5["diff_s_star_minus_p10"]))}
    p = partE["W10"]["up"]["share_ratio"]
    J["J6"] = {"statistic": "W10, share of class B among upward one-frame departures / share of class B among all rows with a T_i ratio",
               **p, "expectation": "uncertain", "holds": bool(p["lo"] is not None and p["lo"] > 1)}
    res["statements"] = J
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a not in ("per_discharge",)} for k, v in J.items()}, indent=1))


# ---------------------------------------------------------------------------------------------------- render
def fm(x, n=3, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ci(p, n=3, sign=False):
    if p is None or p.get("point") is None:
        return "-"
    return f"{fm(p['point'], n, sign)} [{fm(p['lo'], n, sign)}, {fm(p['hi'], n, sign)}]"


def render(res):
    S = [str(s) for s in STAT_SHOTS]
    J = res["statements"]
    L = ["# B.74 what is inside the exposure of a frame whose reported T_i departs (registration PREREGISTRATION_B74.md, addendum 1)", "",
         "Raw BES stream (with raw ECEI and Mirnov) of the twelve discharges with a raw delivery; nothing is trained, no "
         "estimator is involved. Eleven discharges enter the statements; 31921 (B.31's discovery discharge) is reported "
         "apart. Every interval is a 95 % interval over 10,000 resamples of the eleven discharges (seed 20261174); with eleven "
         "units the intervals are coarse. Classes of a row from the raw stream: B = a burst onset inside the window; "
         "P = none inside, one in the 10 ms before; Q = none inside and none in the 30 ms before. W10 = [t - 2 ms, t + 8 ms), "
         "W5 = [t, t + 5 ms). Ratio = value of the row / median of the ten preceding regular rows.", "",
         "## Statements", "",
         "| Id | Statement (as registered) | Statistic | Point [95 % interval] | Expectation | Verdict |",
         "|---|---|---|---|---|---|"]
    reg = {"J1": "W10: share of rows without a usable value higher in class B than in class Q (interval of the difference above zero)",
           "J2": "W10, rows with a value: median T_i ratio of class B above that of class Q (interval of the difference above zero)",
           "J3": "W10: in class B the median following-level ratio does not lie above the class-Q value (interval of the difference contains zero or lies below)",
           "J4": "W10: median ratio of fresh V_rot in class B within 0.03 of that of class Q (interval inside +-0.03)",
           "J5": "Part D, 5 ms windows: largest rank correlation at a start between -2 and +5 ms, exceeding the values at -10 and +10 ms (intervals of both differences above zero)",
           "J6": "Part E: class B over-represented among the upward one-frame departures (interval of the share ratio above 1)"}
    for k in ("J1", "J2", "J3", "J4", "J6"):
        j = J[k]
        L.append(f"| {k} | {reg[k]} | {j['statistic']} | {ci(j, 3, True) if k != 'J6' else ci(j, 2)} | {j['expectation']} | "
                 f"{'holds' if j['holds'] else 'does not hold'} |")
    j = J["J5"]
    L.insert(len(L) - 1, f"| J5 | {reg['J5']} | {j['statistic']} | start {j['s_star_ms']:+d} ms, rho {fm(j['rho_s_star'])}; "
             f"minus rho(-10 ms): {ci(j['diff_minus10'], 3, True)}; minus rho(+10 ms): {ci(j['diff_plus10'], 3, True)} | "
             f"{j['expectation']} | {'holds' if j['holds'] else 'does not hold'} |")
    L += ["", "Reading of the registration: J2 with J3 and J4 is the reading \"a burst of light in the exposure raises the "
          "reported width for that frame only, and not the shift\". If J2 does not hold, the one-frame rises are not "
          "explained by bursts as B.31 defines them.", ""]

    # A
    L += ["## A. Share of rows without a usable T_i, by class", "",
          "| window | class B | class P | class Q | B minus Q | rows B / P / Q / O |", "|---|---|---|---|---|---|"]
    for wn in WINDOWS:
        a = res["A"][wn]
        L.append(f"| {wn} | {ci(a['B'])} | {ci(a['P'])} | {ci(a['Q'])} | {ci(a['B_minus_Q'], 3, True)} | "
                 f"{a['B']['n']} / {a['P']['n']} / {a['Q']['n']} / {a['rows_class_O']} |")
    L += ["", "Per discharge (W10): share without a usable T_i", "", "| discharge | B | P | Q |", "|---|---|---|---|"]
    for s in S + [str(DISCOVERY)]:
        a = res["A"]["W10"]
        L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | {fm(a['B']['per_discharge'][s])} | "
                 f"{fm(a['P']['per_discharge'][s])} | {fm(a['Q']['per_discharge'][s])} |")

    # B
    L += ["", "## B. Rows with a value: median ratios by class", ""]
    for wn in WINDOWS:
        b = res["B"][wn]
        L += [f"### {wn}", "", "| quantity | class B | class P | class Q | B minus Q | rows B / P / Q |", "|---|---|---|---|---|---|"]
        for vk, nm in (("ti_ratio", "T_i ratio"), ("l5_ratio", "following level / level before"),
                       ("vt_ratio", "fresh V_rot ratio"), ("ec_ratio", "ECEI ratio (sample at the stamp)")):
            c = b[vk]
            L.append(f"| {nm} | {ci(c['B'])} | {ci(c['P'])} | {ci(c['Q'])} | {ci(c['B_minus_Q'], 3, True)} | "
                     f"{c['B']['n']} / {c['P']['n']} / {c['Q']['n']} |")
        sh = b["shares"]
        L += ["", "| share of rows | class B | class P | class Q |", "|---|---|---|---|",
              f"| T_i ratio above 1.35 | {ci(sh['B_above_1.35'])} | {ci(sh['P_above_1.35'])} | {ci(sh['Q_above_1.35'])} |",
              f"| T_i ratio below 0.74 | {ci(sh['B_below_0.74'])} | {ci(sh['P_below_0.74'])} | {ci(sh['Q_below_0.74'])} |", ""]
    L += ["Per discharge (W10), medians of class B / class Q", "",
          "| discharge | T_i ratio | following-level ratio | fresh V_rot ratio | ECEI ratio | above 1.35 B / Q |", "|---|---|---|---|---|---|"]
    b = res["B"]["W10"]
    for s in S + [str(DISCOVERY)]:
        def bq(vk):
            return f"{fm(b[vk]['B']['per_discharge'][s])} / {fm(b[vk]['Q']['per_discharge'][s])}"
        L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | {bq('ti_ratio')} | {bq('l5_ratio')} | {bq('vt_ratio')} | "
                 f"{bq('ec_ratio')} | {fm(b['shares']['B_above_1.35']['per_discharge'][s])} / "
                 f"{fm(b['shares']['Q_above_1.35']['per_discharge'][s])} |")

    # C
    L += ["", "## C. Dose: medians by bins of the excess light (all classes, rows with a value)", ""]
    for wn in WINDOWS:
        c = res["C"][wn]
        qe = ", ".join(fm(q, 4) for q in c["quartile_edges_of_positive"])
        L += [f"### {wn} (quartile edges of the positive values: {qe}; rows {c['rows']}, of which zero {c['rows_zero']})", "",
              "| bin | rows | median T_i ratio | median following-level ratio |", "|---|---|---|---|"]
        for bn, cell in c["bins"].items():
            L.append(f"| {bn} | {cell['rows']} | {ci(cell['ti_ratio'])} | {ci(cell['l5_ratio'])} |")
        L.append("")

    # D
    for cw in ("W10", "W5"):
        d = res["D"][cw]
        L += [f"## D. Where the association sits: rank correlation of the T_i ratio and the window-mean BES ratio "
              f"(class Q and B rows of {cw}; {d['rows']} rows){' (beside)' if cw == 'W5' else ''}", "",
              "| start (ms from the stamp) | 1 ms window | 2 ms window | 5 ms window |", "|---|---|---|---|"]
        for st in D_STARTS:
            L.append(f"| {st:+d} | " + " | ".join(ci(d["cells"][f"L{Ln}_s{st:+d}"], 3, True) for Ln in D_LENGTHS) + " |")
        j5 = d["J5"]
        L += ["", f"- 5 ms windows: largest at start {j5['s_star_ms']:+d} ms (rho {fm(j5['rho_s_star'], 3, True)}); "
              f"minus rho(-10 ms) {ci(j5['diff_s_star_minus_m10'], 3, True)}; minus rho(+10 ms) {ci(j5['diff_s_star_minus_p10'], 3, True)}; "
              f"share of draws whose own maximum lies in [-2, +5] ms: {fm(j5['share_draws_argmax_in_-2_+5'], 3)}; "
              f"quantiles 2.5/50/97.5 % of that start: {', '.join(f'{q:+.0f}' for q in j5['argmax_draw_quantiles_ms'])} ms.",
              f"- Largest point estimate for 1 ms windows at {d['argmax_L1_ms']:+d} ms, for 2 ms windows at {d['argmax_L2_ms']:+d} ms.",
              f"- Tied values of the T_i ratio among these rows: {d['ti_ratio_tied_values']}.", ""]
        if cw == "W10":
            L += ["Per discharge (5 ms windows, W10 classes): rank correlation at starts -10 / s* / +10 ms", "",
                  "| discharge | -10 ms | s* | +10 ms |", "|---|---|---|---|"]
            for s in S + [str(DISCOVERY)]:
                L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | "
                         + " | ".join(fm(d["cells"][f"L5_s{st:+d}"]["per_discharge"][s], 3, True)
                                      for st in (-10, j5["s_star_ms"], 10)) + " |")
            L.append("")

    # E
    L += ["## E. One-frame departures (ratio beyond the threshold, next measured value back within 15 % of the level before)", "",
          "| window | direction | departures (in class B) | share of class B among them | share of class B among all rows with a value | share ratio |",
          "|---|---|---|---|---|---|"]
    for wn in WINDOWS:
        e = res["E"][wn]
        for dn in ("up", "down"):
            x = e[dn]
            L.append(f"| {wn} | {dn} | {x['departures']} ({x['departures_in_B']}) | {ci(x['share_B'])} | "
                     f"{ci(e['share_B_all_rows'])} | {ci(x['share_ratio'], 2)} |")
    L += ["", "This conditions on the reported value on purpose: it describes the target; no estimator is judged.", "",
          "Per discharge (W10): share of class B among upward departures / among all rows with a value", "",
          "| discharge | upward | all rows |", "|---|---|---|"]
    e = res["E"]["W10"]
    for s in S + [str(DISCOVERY)]:
        L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | {fm(e['up']['share_B']['per_discharge'][s])} | "
                 f"{fm(e['share_B_all_rows']['per_discharge'][s])} |")

    # F
    L += ["", "## F. Reported beside", "",
          "| discharge | rows (file / set / with T_i ratio) | bursts (raw stream) | W10 classes B / P / Q / O | W5 classes B / P / Q / O | rows left out (beam interruption) | kept rows with an interruption bin in W10 / in [-10, +19] ms |",
          "|---|---|---|---|---|---|---|"]
    for s, d in res["F"]["per_discharge"].items():
        c10, c5 = d["classes_W10"], d["classes_W5"]
        L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | {d['rows_file']} / {d['rows_in_set']} / {d['rows_with_ti_ratio']} | "
                 f"{d['bursts_raw_stream']} | {c10['B']} / {c10['P']} / {c10['Q']} / {c10['O']} | "
                 f"{c5['B']} / {c5['P']} / {c5['Q']} / {c5['O']} | {d['rows_left_out_blip']} | "
                 f"{d['rows_kept_blip_in_W10']} / {d['rows_kept_blip_in_partD_span']} |")
    L += ["", "| window | Mirnov RMS ratio, class B | class Q | ECEI change across the burst onset, class B | ECEI change across the stamp, class Q |",
          "|---|---|---|---|---|"]
    for wn in WINDOWS:
        f = res["F"][wn]
        L.append(f"| {wn} | {ci(f['mirnov_B'])} (n {f['mirnov_B']['n']}) | {ci(f['mirnov_Q'])} (n {f['mirnov_Q']['n']}) | "
                 f"{ci(f['ecei_onset_B'], 3, True)} (n {f['ecei_onset_B']['n']}) | {ci(f['ecei_stamp_Q'], 3, True)} (n {f['ecei_stamp_Q']['n']}) |")
    L += ["", "Per discharge (W10): Mirnov RMS ratio B / Q; ECEI change across the onset (B) / across the stamp (Q)", "",
          "| discharge | Mirnov B / Q | ECEI B / Q |", "|---|---|---|"]
    f = res["F"]["W10"]
    for s in S + [str(DISCOVERY)]:
        L.append(f"| {s}{' (discovery)' if s == str(DISCOVERY) else ''} | {fm(f['mirnov_B']['per_discharge'][s])} / "
                 f"{fm(f['mirnov_Q']['per_discharge'][s])} | {fm(f['ecei_onset_B']['per_discharge'][s], 3, True)} / "
                 f"{fm(f['ecei_stamp_Q']['per_discharge'][s], 3, True)} |")

    # checks
    ck = res["checks"]["discharges"]
    L += ["", "## Checks (registration section 7)", "",
          "| discharge | bursts (this run) | bursts (b31_bursts.md) | onsets from the recomputed residual equal | BES rows in raw span | share equal to a raw sample at the stamp |",
          "|---|---|---|---|---|---|"]
    for s in map(str, SHOTS):
        c = ck[s]
        L.append(f"| {s} | {c['bursts']} | {c['bursts_b31_md']} | {'yes' if c['recomputed_onsets_equal'] else 'no'} | "
                 f"{c['bes_rows_in_raw_span']} | {fm(c['bes_sample_share'], 4)} |")
    L += ["", "## Limits", "",
          "- Eleven discharges in the statements, one campaign, selected earlier for their published context; not a sample of the 639.",
          "- A burst is defined on the beam-emission light; it is not confirmed as an ELM by D-alpha, which is not in the files.",
          "- The exposure window is not known to better than a few ms; two windows are carried and part D reports where the association peaks.",
          "- No spectrum, fit amplitude or fit error is available: an association with bursts says in which frames the width departs, not why.",
          "- Exploratory: the reviewers have looked at these streams for other quantities."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "check":
        stage_check()
    elif stage == "prep":
        stage_prep()
    elif stage == "readout":
        stage_readout()
    else:
        raise SystemExit("usage: run_b74.py check|prep|readout")
