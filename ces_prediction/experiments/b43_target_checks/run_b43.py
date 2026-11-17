"""B.43 (PREREGISTRATION_B43.md): checks of the reported T_i value that the files allow.

S  co-variation of block-mean T_i changes with block-mean ECEI-level changes (200 ms blocks)
   against a circularly shifted null                                               (S1)
F  where the values above 3 keV sit, by fifth of the BES level                    (F1)
H  the published time course of discharge 31923 around the beam-power step at 4.0 s (H1)
D  the fall of the estimate in panel (c) of the trace figure (descriptive)

CPU only; no model is trained.  Part D runs the frozen checkpoint of the trace figure
(data/.b1_seqv2_s1_i1) forward on the CPU for one discharge.  Run from the repo root:
    py ces_prediction/experiments/b43_target_checks/run_b43.py
Output: docs/paper/b43_target_checks.json / .md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CES_TRACE_DEVICE"] = "cpu"          # read by fig_trace at import; GPU is never touched

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "docs" / "paper" / "ppcf"))
sys.path.insert(0, str(HERE.parent / "b32_two_sources"))
import fig_trace as FT  # noqa: E402  (sets the frozen-protocol env, CPU device)
import run_b32 as B32  # noqa: E402
from seq_data import load_grid_files  # noqa: E402  (path inserted by fig_trace)

import torch  # noqa: E402

DATA = REPO / "data"
OUT_JSON = REPO / "docs" / "paper" / "b43_target_checks.json"
OUT_MD = REPO / "docs" / "paper" / "b43_target_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B43.md"
SEED, N_BOOT = 20261118, 2000
SEG_BREAK = 0.5
CUT_EV = 3000.0
T0 = time.time()

# block designs of part S: rows per block, min measured T_i rows, min pairs per discharge.
# 20 rows / 10 / 40 is registered; the 100 ms and 500 ms versions keep half the rows measured
# and the same minimum span of data per discharge (40 pairs x 200 ms = 8 s) -- fixed before the run.
BLOCKS = {"200ms": (20, 10, 40), "100ms": (10, 5, 80), "500ms": (50, 25, 16)}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [float("nan")] * 2


def boot_W(n, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)


def absent_stamps(t):
    """Stamps of absent frames: steps of 20, 30, 40 ms are one, two, three absent frames."""
    dt = np.diff(t)
    out = []
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        n = int(round(dt[j] / 0.010)) - 1
        out += [float(t[j] + 0.010 * (i + 1)) for i in range(n)]
    return np.array(out)


# ------------------------------------------------------------------------------------------ S
def s_blocks(t, ti, x, B, min_meas):
    """Pairs of consecutive used blocks inside each contiguous segment: (dT, dX)."""
    seg = np.concatenate(([0], np.cumsum(np.diff(t) >= SEG_BREAK)))
    dT, dX = [], []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        nb = len(idx) // B                      # trailing partial block dropped
        mt, mx, used = np.full(nb, np.nan), np.full(nb, np.nan), np.zeros(nb, bool)
        for k in range(nb):
            r = idx[k * B:(k + 1) * B]
            ok = np.isfinite(ti[r])
            if ok.sum() >= min_meas:
                used[k] = True
                mt[k] = ti[r][ok].mean()
                mx[k] = x[r].mean()
        p = np.flatnonzero(used[:-1] & used[1:])
        dT.append(mt[p + 1] - mt[p])
        dX.append(mx[p + 1] - mx[p])
    return np.concatenate(dT) if dT else np.zeros(0), np.concatenate(dX) if dX else np.zeros(0)


def part_s(grid, names, nb, ne, rng):
    out = {}
    med_ti = {}
    for nm in names:
        a = grid[nm][1:-1]
        v = a[:, 1].astype(float)
        med_ti[nm] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    for bname, (B, mm, mp) in BLOCKS.items():
        rows = []
        for nm in names:
            a = grid[nm][1:-1]
            t, ti = a[:, 0].astype(float), a[:, 1].astype(float)
            bes = a[:, 3:3 + nb].astype(float).mean(1)
            ecei = a[:, 3 + nb:3 + nb + ne].astype(float).mean(1)
            rec = {"name": nm}
            for lab, x in (("ecei", ecei), ("bes", bes)):
                dT, dX = s_blocks(t, ti, x, B, mm)
                n = len(dT)
                rec["pairs"] = n
                if n < mp:
                    rec[lab] = None
                    continue
                r = spearmanr(dT, dX)[0]
                r0 = spearmanr(dT, np.roll(dX, n // 2))[0]
                rec[lab] = (float(r), float(r0))
            if rec.get("ecei") is not None or rec.get("bes") is not None:
                rows.append(rec)
        res = {"discharges_with_enough_pairs": len(rows)}
        for lab in ("ecei", "bes"):
            rr = [(r["name"], *r[lab]) for r in rows if r.get(lab) is not None and np.isfinite(r[lab][0]) and np.isfinite(r[lab][1])]
            n_nan = sum(1 for r in rows if r.get(lab) is not None and not (np.isfinite(r[lab][0]) and np.isfinite(r[lab][1])))
            res[lab] = summarize_s(rr, med_ti, rng, n_nan)
        out[bname] = res
        log(f"S {bname}: {res['ecei']['discharges']} discharges; ECEI median rho {res['ecei']['median_rho'][0]:+.3f}, "
            f"null {res['ecei']['median_null'][0]:+.3f}, diff {res['ecei']['diff'][0]:+.3f} "
            f"[{res['ecei']['diff'][1]:+.3f}, {res['ecei']['diff'][2]:+.3f}]")
    return out


def summarize_s(rr, med_ti, rng, n_nan):
    n = len(rr)
    if n == 0:
        return {"discharges": 0}
    names = [r[0] for r in rr]
    rho = np.array([r[1] for r in rr])
    nul = np.array([r[2] for r in rr])
    f = np.arange(n)
    W = boot_W(n, rng)
    p1, b1 = B32.wmedian(rho, f, W)
    p0, b0 = B32.wmedian(nul, f, W)
    o = {"discharges": n, "discharges_dropped_nan_rho": n_nan,
         "median_rho": [p1, *ci(b1)], "median_null": [p0, *ci(b0)], "diff": [p1 - p0, *ci(b1 - b0)],
         "share_rho_positive": float((rho > 0).mean()),
         "rho_quantiles_10_25_50_75_90": [float(x) for x in np.quantile(rho, [0.1, 0.25, 0.5, 0.75, 0.9])]}
    mt = np.array([med_ti[nm] for nm in names])
    e = np.quantile(mt, [1 / 3, 2 / 3])
    th = np.searchsorted(e, mt, side="right")
    o["thirds_edges_eV"] = e.tolist()
    o["thirds"] = []
    for k in range(3):
        sel = th == k
        m = int(sel.sum())
        if m < 3:
            o["thirds"].append({"discharges": m})
            continue
        Wk = boot_W(m, rng)
        fk = np.arange(m)
        q1, c1 = B32.wmedian(rho[sel], fk, Wk)
        q0, c0 = B32.wmedian(nul[sel], fk, Wk)
        o["thirds"].append({"discharges": m, "median_ti_range_eV": [float(mt[sel].min()), float(mt[sel].max())],
                            "median_rho": [q1, *ci(c1)], "median_null": [q0, *ci(c0)], "diff": [q1 - q0, *ci(c1 - c0)]})
    return o


def part_s_sensitivity(grid, names, nb, ne, rng):
    """After the fact (not registered): the 200 ms read-out with lower minimum numbers of pairs,
    and the distribution of the number of pairs per discharge."""
    npairs = []
    per = []
    for nm in names:
        a = grid[nm][1:-1]
        t, ti = a[:, 0].astype(float), a[:, 1].astype(float)
        ecei = a[:, 3 + nb:3 + nb + ne].astype(float).mean(1)
        dT, dX = s_blocks(t, ti, ecei, 20, 10)
        npairs.append(len(dT))
        per.append((nm, dT, dX))
    o = {"pairs_per_discharge_quantiles_10_50_90": [float(x) for x in np.quantile(npairs, [0.1, 0.5, 0.9])],
         "discharges_with_ge": {str(k): int((np.array(npairs) >= k).sum()) for k in (10, 20, 40)}, "by_min_pairs": {}}
    med_ti = {nm: float(np.nanmedian(grid[nm][1:-1, 1])) if np.isfinite(grid[nm][1:-1, 1]).any() else np.nan for nm in names}
    for mp in (10, 20):
        rr = []
        for nm, dT, dX in per:
            n = len(dT)
            if n < mp:
                continue
            r = spearmanr(dT, dX)[0]
            r0 = spearmanr(dT, np.roll(dX, n // 2))[0]
            if np.isfinite(r) and np.isfinite(r0):
                rr.append((nm, float(r), float(r0)))
        s = summarize_s(rr, med_ti, rng, 0)
        o["by_min_pairs"][str(mp)] = {k: s[k] for k in ("discharges", "median_rho", "median_null", "diff", "share_rho_positive")}
        log(f"after the fact S min pairs {mp}: {s['discharges']} discharges, diff {s['diff'][0]:+.3f} [{s['diff'][1]:+.3f}, {s['diff'][2]:+.3f}]")
    return o


# ------------------------------------------------------------------------------------------ F
def after_rows(t):
    """Rows 1..10 after an absent frame inside the segment: the rule of run_b32.kstar_file, for every
    file (kstar_file returns None for files with fewer than 12 grid rows)."""
    n = len(t)
    dt = np.diff(t)
    seg = np.concatenate(([0], np.cumsum(dt >= B32.SEG_BREAK_S)))
    after = np.zeros(n, int)
    for j in np.flatnonzero((dt > 0.015) & (dt < 0.045)):
        for k in range(1, 11):
            i = j + k
            if i < n and seg[i] == seg[j + 1] and after[i] == 0:
                after[i] = k
    return after


def part_f(grid_incl, grid_raw, names, nb, rng):
    rows = {k: [] for k in ("f", "rel", "z", "ti", "raw_missing", "after")}
    align_bad, b32_compared, b32_mismatch = 0, 0, 0
    for i, nm in enumerate(names):
        a = grid_incl[nm][1:-1]
        r = grid_raw[nm][1:-1]
        t = a[:, 0].astype(float)
        if len(r) != len(a) or np.max(np.abs(r[:, 0].astype(float) - t)) > 1e-6:
            align_bad += 1
            continue
        after = after_rows(t)
        k = B32.kstar_file(DATA / nm)
        if k is not None:
            b32_compared += 1
            same = (len(k["t"]) == len(t) and np.max(np.abs(k["t"] - t)) < 1e-4 and np.array_equal(k["after"], after)
                    and np.array_equal(np.isfinite(k["ti"]), np.isfinite(r[:, 1])))
            b32_mismatch += int(not same)
        b = a[:, 3:3 + nb].astype(float).mean(1)
        med, sd = np.median(b), b.std()
        rows["f"].append(np.full(len(a), i))
        rows["rel"].append(b / med if med > 0 else np.full(len(a), np.nan))
        rows["z"].append((b - b.mean()) / sd if sd > 0 else np.full(len(a), np.nan))
        rows["ti"].append(a[:, 1].astype(float))
        rows["raw_missing"].append(~np.isfinite(r[:, 1].astype(float)))
        rows["after"].append(after)
    R = {k: np.concatenate(v) for k, v in rows.items()}
    uf, f = np.unique(R["f"], return_inverse=True)
    n = len(uf)
    W = boot_W(n, rng)
    out = {"discharges": n, "files_misaligned": align_bad, "grid_rows": int(len(f)),
           "b32_reader_files_compared": b32_compared, "b32_reader_files_differing": b32_mismatch,
           "discharges_with_ti": int(len(np.unique(f[np.isfinite(R["ti"])])))}

    def fifth_tables(level, sel, flag):
        ok = sel & np.isfinite(level)
        q = np.quantile(level[ok], [0.2, 0.4, 0.6, 0.8])
        fi = np.searchsorted(q, level[ok], side="right")
        nt = np.zeros((n, 5))
        kt = np.zeros((n, 5))
        np.add.at(nt, (f[ok], fi), 1.0)
        np.add.at(kt, (f[ok], fi), flag[ok].astype(float))
        return q, nt, kt, int(ok.sum())

    def shares(nt, kt):
        with np.errstate(all="ignore"):
            pb = (W @ kt) / (W @ nt)
        return [[float(kt[:, j].sum() / nt[:, j].sum()), *ci(pb[:, j])] for j in range(5)], pb

    # registered F: rows with a T_i value (held removed, no 3 keV cut), BES relative to the discharge median
    has = np.isfinite(R["ti"])
    above = R["ti"] > CUT_EV
    q, nt, kt, nrows = fifth_tables(R["rel"], has, above)
    sh, pb = shares(nt, kt)
    with np.errstate(all="ignore"):
        rb = pb[:, 0] / pb[:, 4]
    p = (kt[:, 0].sum() / nt[:, 0].sum()) / (kt[:, 4].sum() / nt[:, 4].sum())
    out["F"] = {"rows_with_ti": nrows, "rows_above_3keV": int(above[has].sum()), "fifth_edges_rel": q.tolist(),
                "rows_by_fifth": nt.sum(0).astype(int).tolist(), "above_by_fifth": kt.sum(0).astype(int).tolist(),
                "share_above_by_fifth": sh, "ratio_lowest_over_highest": [float(p), *ci(rb)],
                "resamples_highest_zero": int((pb[:, 4] == 0).sum())}
    out["F1"] = {"holds": bool(ci(rb)[0] > 1)}
    # missing T_i by fifth, registered definition (BES / discharge median), all grid rows, missing = empty in the file
    allrows = np.ones(len(f), bool)
    q2, nt2, kt2, nr2 = fifth_tables(R["rel"], allrows, R["raw_missing"])
    out["missing_rel_median"] = {"rows": nr2, "share_by_fifth": shares(nt2, kt2)[0], "fifth_edges_rel": q2.tolist()}
    # check: the definition of file_rules.md section 2 (per-discharge z-score)
    q3, nt3, kt3, nr3 = fifth_tables(R["z"], allrows, R["raw_missing"])
    out["missing_zscore_check"] = {"rows": nr3, "share_by_fifth": shares(nt3, kt3)[0]}
    # the values above 3 keV within ten rows after an absent frame
    aft = (R["after"] >= 1) & (R["after"] <= 10)
    sa = has & above
    kk = np.bincount(f[sa], aft[sa].astype(float), n)
    nn = np.bincount(f[sa], minlength=n).astype(float)
    kb = np.bincount(f[has], aft[has].astype(float), n)
    nb_ = np.bincount(f[has], minlength=n).astype(float)
    with np.errstate(all="ignore"):
        out["after_blip"] = {
            "above_3keV_rows": int(sa.sum()), "above_within_10_rows_after": int(aft[sa].sum()),
            "share_of_above_within_10_rows_after": [float(kk.sum() / nn.sum()), *ci((W @ kk) / (W @ nn))],
            "reference_share_of_all_ti_rows_within_10_rows_after": [float(kb.sum() / nb_.sum()), *ci((W @ kb) / (W @ nb_))],
            "ratio_to_reference": [float((kk.sum() / nn.sum()) / (kb.sum() / nb_.sum())),
                                   *ci(((W @ kk) / (W @ nn)) / ((W @ kb) / (W @ nb_)))]}
    log(f"F: shares above 3 keV by fifth {[round(x[0], 4) for x in sh]}, ratio {p:.2f}; "
        f"missing z-check {[round(100 * x[0], 1) for x in out['missing_zscore_check']['share_by_fifth']]}")
    return out


# ------------------------------------------------------------------------------------------ H
def part_h(grid_cut, grid_incl, rng):
    a = grid_cut["s31923.csv"][1:-1]
    t, ti = a[:, 0].astype(float), a[:, 1].astype(float)
    ab = absent_stamps(t)
    win = ab[(ab >= 3.0) & (ab <= 5.5)]
    steps = np.diff(t)
    odd = [[float(t[j]), float(steps[j])] for j in np.flatnonzero(np.abs(steps - 0.010) > 1e-4) if 3.0 <= t[j] <= 5.5]
    near35 = win[np.abs(win - 3.5) <= 0.030 + 1e-9]
    near50 = win[np.abs(win - 5.0) <= 0.030 + 1e-9]

    def window(lo, hi):
        m = (t >= lo - 1e-6) & (t <= hi + 1e-6)
        blk = np.floor((t[m] - lo) / 0.1 + 1e-9).astype(int)
        return t[m], ti[m], blk

    t1, y1, b1 = window(3.60, 3.95)
    t2, y2, b2 = window(4.10, 4.90)

    def bstats(y, blk):
        ub = np.unique(blk)
        s = np.array([np.nansum(y[blk == u]) for u in ub])
        c = np.array([np.isfinite(y[blk == u]).sum() for u in ub], float)
        return s, c

    s1, c1 = bstats(y1, b1)
    s2, c2 = bstats(y2, b2)
    m1, m2 = s1.sum() / c1.sum(), s2.sum() / c2.sum()
    i1 = rng.integers(0, len(s1), size=(N_BOOT, len(s1)))
    i2 = rng.integers(0, len(s2), size=(N_BOOT, len(s2)))
    bm1 = s1[i1].sum(1) / c1[i1].sum(1)
    bm2 = s2[i2].sum(1) / c2[i2].sum(1)
    ratio = [float(m2 / m1), *ci(bm2 / bm1)]
    means100 = []
    for k in range(15):
        lo = 3.5 + 0.1 * k
        m = (t >= lo - 1e-6) & (t < lo + 0.1 - 1e-6)
        v = ti[m][np.isfinite(ti[m])]
        means100.append({"from_s": round(lo, 2), "rows": int(m.sum()), "measured": int(len(v)),
                         "mean_eV": float(v.mean()) if len(v) else None, "sd_eV": float(v.std(ddof=1)) if len(v) > 1 else None})
    # 31921 at the two published times
    a21 = grid_cut["s31921.csv"][1:-1]
    a21i = grid_incl["s31921.csv"][1:-1]
    p21 = {}
    for tc in (5.40, 8.05):
        m = np.abs(a21[:, 0].astype(float) - tc) <= 0.050 + 1e-6
        v = a21[m, 1].astype(float)
        v = v[np.isfinite(v)]
        vi = a21i[m, 1].astype(float)
        p21[f"{tc:.2f}"] = {"rows": int(m.sum()), "measured": int(len(v)), "mean_eV": float(v.mean()) if len(v) else None,
                            "sd_eV": float(v.std(ddof=1)) if len(v) > 1 else None,
                            "values_above_3keV_removed": int((vi > CUT_EV).sum())}
    o = {"first_grid_stamp_s": float(t[0]), "last_grid_stamp_s": float(t[-1]),
         "absent_stamps_3_to_5p5_s": win.tolist(), "non_10ms_steps_3_to_5p5_s": odd,
         "absent_within_30ms_of_3p5": near35.tolist(), "absent_within_30ms_of_5p0": near50.tolist(),
         "early": {"window_s": [3.60, 3.95], "rows": int(len(y1)), "measured": int(np.isfinite(y1).sum()),
                   "blocks": int(len(s1)), "mean_eV": [float(m1), *ci(bm1)]},
         "late": {"window_s": [4.10, 4.90], "rows": int(len(y2)), "measured": int(np.isfinite(y2).sum()),
                  "blocks": int(len(s2)), "mean_eV": [float(m2), *ci(bm2)]},
         "ratio_late_over_early": ratio, "means_100ms_3p5_to_5p0": means100, "s31921": p21}
    o["H1"] = {"blip_near_3p5": bool(len(near35)), "blip_near_5p0": bool(len(near50)), "ratio_interval_above_1": bool(ratio[1] > 1)}
    o["H1"]["holds"] = bool(o["H1"]["blip_near_3p5"] and o["H1"]["blip_near_5p0"] and o["H1"]["ratio_interval_above_1"])
    log(f"H: absent {win.tolist()}; ratio {ratio}")
    return o


# ------------------------------------------------------------------------------------------ D
def part_d():
    name = "s31343.csv"
    metrics = json.loads((FT.RUN / "metrics.json").read_text(encoding="utf-8"))
    stats = FT._load_stats(metrics)
    grid, _ = load_grid_files(DATA, True)          # env CES_TI_SPIKE_CUT_EV=3000, as in fig_trace
    valid_names = sorted(grid)
    torch.set_num_threads(4)
    with torch.no_grad():
        lookup = FT.build_pred_lookup(DATA, [name], stats, torch.device("cpu"))["full"]
    time_, ti, model, pers, gp, seg_id = FT.estimates(grid[name], lookup[name])
    npz = np.load(FT.RUN / "comparison_errors_test.npz")
    chk = FT.check(name, time_, ti, model, pers, gp, npz, valid_names)
    lo, hi, n_gap = FT.pick_window(time_, ti, seg_id)
    sl = slice(lo, hi)
    t = time_[sl]
    obs = np.isfinite(ti[sl])
    vv = ti[sl][obs]
    vmin, vmax = float(vv.min()), float(vv.max())
    below = np.flatnonzero(model[sl] < vmin)
    k = B32.kstar_file(DATA / name)          # rows 1:-1 of the file; grid row r -> k row r - 1
    raw = grid[name]
    nb, ne = 9, 4
    ecei = raw[:, 3 + nb:3 + nb + ne].astype(float).mean(1)
    ab = absent_stamps(raw[1:-1, 0].astype(float))
    rows = []
    for j in below:
        r = lo + j
        tr = float(time_[r])
        prev_ab = ab[ab < tr]
        e_prev = ecei[max(r - 10, 0):r]
        nxt = np.flatnonzero(np.isfinite(ti[r + 1:]))
        prv = np.flatnonzero(np.isfinite(ti[:r]))
        rows.append({"time_s": round(tr, 4), "estimate_eV": float(model[r]), "measured_at_row_eV": float(ti[r]) if np.isfinite(ti[r]) else None,
                     "previous_measured_eV": float(ti[prv[-1]]) if len(prv) else None,
                     "previous_measured_time_s": round(float(time_[prv[-1]]), 4) if len(prv) else None,
                     "next_measured_eV": float(ti[r + 1 + nxt[0]]) if len(nxt) else None,
                     "next_measured_time_s": round(float(time_[r + 1 + nxt[0]]), 4) if len(nxt) else None,
                     "ms_since_last_absent_frame": round(float(1000 * (tr - prev_ab[-1])), 1) if len(prev_ab) else None,
                     "last_absent_frame_s": round(float(prev_ab[-1]), 4) if len(prev_ab) else None,
                     "mean_bes_departure_sd": float(k["d"][r - 1]) if 0 <= r - 1 < len(k["d"]) else None,
                     "rows_since_blip_b32": int(k["after"][r - 1]) if 0 <= r - 1 < len(k["after"]) else None,
                     "ecei_over_100ms_before": float(ecei[r] / e_prev.mean()) if len(e_prev) and abs(e_prev.mean()) > 1e-12 else None})
    # context: the same quantities over all rows of the window (after the fact, for scale)
    d_win = np.array([k["d"][lo + j - 1] for j in range(hi - lo) if 0 <= lo + j - 1 < len(k["d"])], float)
    ctx = {"window_mean_bes_departure_sd_quantiles_10_50_90": [float(x) for x in np.nanquantile(d_win, [0.1, 0.5, 0.9])],
           "window_absent_frames": [float(x) for x in ab[(ab >= t[0] - 0.2) & (ab <= t[-1])]]}
    fj = json.loads((HERE.parents[2] / "docs" / "paper" / "ppcf" / "fig_trace.json").read_text(encoding="utf-8"))[name]
    o = {"discharge": name, "estimator": "recurrent estimator of the figure (data/.b1_seqv2_s1_i1, whole-discharge standardization), CPU",
         "window_s": [float(t[0]), float(t[-1])], "window_s_in_fig_trace_json": fj["window_s"],
         "measured_range_eV": [vmin, vmax], "measured_range_in_fig_trace_json": fj["measured_min_max_eV_in_window"],
         "check_vs_stored_errors": chk, "rows_below_measured_range": rows,
         "min_estimate_in_window_eV": float(np.nanmin(model[sl])), "context": ctx}
    o["check_passed"] = bool(chk["model"]["max_abs_dev_of_error_eV"] <= 0.5 and chk["persistence"]["share_within_1e-3"] >= 0.999
                             and chk["causal_gp"]["share_within_1e-3"] >= 0.999 and chk["max_value_mismatch_eV"] < 0.05
                             and abs(o["window_s"][0] - fj["window_s"][0]) < 1e-4 and abs(o["window_s"][1] - fj["window_s"][1]) < 1e-4
                             and abs(round(vmin, 1) - fj["measured_min_max_eV_in_window"][0]) < 0.11
                             and abs(round(vmax, 1) - fj["measured_min_max_eV_in_window"][1]) < 0.11)
    log(f"D: {len(rows)} rows below {vmin:.1f} eV; check {o['check_passed']}")
    return o


# ------------------------------------------------------------------------------------------ render
def f3(x, nd=3, sign=False):
    s = "+" if sign else ""
    return f"{x[0]:{s}.{nd}f} [{x[1]:{s}.{nd}f}, {x[2]:{s}.{nd}f}]"


def pct(x):
    return f"{100 * x[0]:.2f} % [{100 * x[1]:.2f}, {100 * x[2]:.2f}]"


def expectations(res):
    S = res["S"]["200ms"]["ecei"]
    E = []
    E.append(("S1: median correlation +0.15 to +0.40", f"{S['median_rho'][0]:+.3f}", bool(0.15 <= S["median_rho"][0] <= 0.40)))
    E.append(("S1: null within 0.03 of zero", f"{S['median_null'][0]:+.3f}", bool(abs(S["median_null"][0]) <= 0.03)))
    E.append(("S1 holds", "holds" if res["S1"]["holds"] else "does not hold", bool(res["S1"]["holds"])))
    r = res["F"]["F"]["ratio_lowest_over_highest"][0]
    E.append(("F1: ratio 2 to 8", f"{r:.2f}", bool(2 <= r <= 8)))
    E.append(("F1 holds", "holds" if res["F"]["F1"]["holds"] else "does not hold", bool(res["F"]["F1"]["holds"])))
    H = res["H"]
    E.append(("H1: both blips found", f"near 3.5 s: {', '.join(f'{x:.3f}' for x in H['absent_within_30ms_of_3p5']) or 'none'} "
              f"(file begins at {H['first_grid_stamp_s']:.3f} s); near 5.0 s: "
              f"{', '.join(f'{x:.3f}' for x in H['absent_within_30ms_of_5p0']) or 'none'}",
              bool(H["H1"]["blip_near_3p5"] and H["H1"]["blip_near_5p0"])))
    q = H["ratio_late_over_early"][0]
    E.append(("H1: ratio 1.05 to 1.40", f"{q:.3f}", bool(1.05 <= q <= 1.40)))
    E.append(("H1 holds", "holds" if H["H1"]["holds"] else "does not hold", bool(H["H1"]["holds"])))
    ms = [x["ms_since_last_absent_frame"] for x in res["D"]["rows_below_measured_range"]]
    E.append(("D: the rows follow an absent frame by 10 to 30 ms", ", ".join("none" if m is None else f"{m:.0f} ms" for m in ms) or "no rows",
              bool(ms and all(m is not None and 10 <= m <= 30 for m in ms))))
    return [{"expectation": a, "observed": b, "met": c} for a, b, c in E]


def render(res):
    hs = lambda h: "holds" if h else "does not hold"  # noqa: E731
    S, F, H, D = res["S"], res["F"], res["H"], res["D"]
    s2 = S["200ms"]["ecei"]
    L = [f"# B.43 — checks of the reported T_i value that the files allow (registered: {REG})", "",
         "CPU; no model trained. Part D runs the frozen network of the trace figure forward on the CPU for one discharge. "
         "Intervals: 95 %, 2,000 resamples of discharges unless stated. Grid = rows of the 641 files without the first (t = 0) "
         "and last (t = 30 s) row. T_i with held values removed; values above 3 keV removed except in part F.", "",
         "## Registered statements", "",
         f"- **S1** (median over discharges of the Spearman correlation of 200 ms block-mean changes of T_i and of the ECEI level, "
         f"minus the median of the circularly shifted null, interval above zero): {f3(s2['median_rho'], sign=True)} minus "
         f"{f3(s2['median_null'], sign=True)} = **{f3(s2['diff'], sign=True)}** ({s2['discharges']} discharges) -> **{hs(res['S1']['holds'])}**.",
         f"- **F1** (share of rows above 3 keV in the lowest fifth of the BES level over that in the highest fifth, interval above 1): "
         f"**{f3(F['F']['ratio_lowest_over_highest'], 2)}** -> **{hs(F['F1']['holds'])}**.",
         f"- **H1** (discharge 31923: an absent frame within 30 ms of 3.5 s and of 5.0 s, and mean T_i over 4.10-4.90 s over that "
         f"over 3.60-3.95 s with an interval above 1): absent frames within 30 ms of 3.5 s: "
         f"{', '.join(f'{x:.3f} s' for x in H['absent_within_30ms_of_3p5']) or 'none'} (the grid of the file begins at "
         f"{H['first_grid_stamp_s']:.3f} s); within 30 ms of 5.0 s: "
         f"{', '.join(f'{x:.3f} s' for x in H['absent_within_30ms_of_5p0']) or 'none'}; ratio **{f3(H['ratio_late_over_early'])}** (2,000 resamples of 100 ms blocks of rows) "
         f"-> **{hs(H['H1']['holds'])}**.", "",
         "## S. Co-variation with the ECEI level and the BES level", "",
         "Blocks of consecutive grid rows inside each contiguous segment (breaks of at least 0.5 s); a block is used when at least "
         "half its rows have a measured T_i; block means of T_i over its measured rows, of the ECEI level (mean of the four "
         "channels) and of the BES level (mean of the nine channels) over all its rows; one value per discharge with enough pairs "
         "of consecutive used blocks. Null: the level changes shifted circularly by half the number of pairs.", "",
         "| blocks | level | discharges | median rho | median null | difference | share of discharges with rho > 0 |",
         "|---|---|---|---|---|---|---|"]
    for bn in ("200ms", "100ms", "500ms"):
        B, mm, mp = BLOCKS[bn]
        for lab in ("ecei", "bes"):
            o = S[bn][lab]
            L.append(f"| {B} rows ({bn}), min {mm} measured, min {mp} pairs | {lab.upper()} | {o['discharges']} | "
                     f"{f3(o['median_rho'], sign=True)} | {f3(o['median_null'], sign=True)} | {f3(o['diff'], sign=True)} | "
                     f"{o['share_rho_positive']:.3f} |")
    L += ["", "By third of the discharge median T_i (200 ms blocks):", "",
          "| level | third | discharges | median T_i range (eV) | median rho | median null | difference |", "|---|---|---|---|---|---|---|"]
    for lab in ("ecei", "bes"):
        for j, o in enumerate(S["200ms"][lab]["thirds"]):
            if "median_rho" not in o:
                L.append(f"| {lab.upper()} | {j + 1} | {o['discharges']} | | | | |")
                continue
            L.append(f"| {lab.upper()} | {j + 1} | {o['discharges']} | {o['median_ti_range_eV'][0]:.0f}-{o['median_ti_range_eV'][1]:.0f} | "
                     f"{f3(o['median_rho'], sign=True)} | {f3(o['median_null'], sign=True)} | {f3(o['diff'], sign=True)} |")
    L += ["", f"Quantiles 0.1/0.25/0.5/0.75/0.9 of the per-discharge rho (200 ms, ECEI): "
          f"{' / '.join(f'{x:+.3f}' for x in s2['rho_quantiles_10_25_50_75_90'])}.", ""]
    FF = F["F"]
    L += ["## F. Where the values above 3 keV sit", "",
          f"Grid rows with a T_i value (held removed, no 3 keV cut): {FF['rows_with_ti']:,} in {F['discharges_with_ti']} discharges, "
          f"{FF['rows_above_3keV']:,} above 3 keV. BES level of a row = mean of the nine channels over the discharge median "
          f"(as in B.40); fifths are pooled quantiles over these rows (edges {', '.join(f'{x:.3f}' for x in FF['fifth_edges_rel'])}).", "",
          "| fifth of the BES level | rows with T_i | above 3 keV | share above 3 keV | missing T_i (all grid rows, same definition) |",
          "|---|---|---|---|---|"]
    for j in range(5):
        L.append(f"| {j + 1} | {FF['rows_by_fifth'][j]:,} | {FF['above_by_fifth'][j]:,} | {pct(FF['share_above_by_fifth'][j])} | "
                 f"{pct(F['missing_rel_median']['share_by_fifth'][j])} |")
    ab = F["after_blip"]
    L += ["", f"Ratio lowest over highest fifth: {f3(FF['ratio_lowest_over_highest'], 2)} (resamples with no value above 3 keV in "
          f"the highest fifth: {FF['resamples_highest_zero']}).", "",
          f"Values above 3 keV within ten rows after an absent frame: {ab['above_within_10_rows_after']} of {ab['above_3keV_rows']}, "
          f"share {pct(ab['share_of_above_within_10_rows_after'])}; the same share for all rows with a T_i value "
          f"{pct(ab['reference_share_of_all_ti_rows_within_10_rows_after'])}; ratio {f3(ab['ratio_to_reference'], 2)}.", ""]
    L += ["## H. The published time course of discharge 31923", "",
          f"Grid of the file: {H['first_grid_stamp_s']:.3f} to {H['last_grid_stamp_s']:.3f} s. "
          f"Absent frames between 3.0 and 5.5 s (stamps, s): {', '.join(f'{x:.3f}' for x in H['absent_stamps_3_to_5p5_s']) or 'none'}. "
          f"Steps other than 10 ms in that range (time of the row before, step in s): "
          f"{'; '.join(f'{a:.3f}, {b:.3f}' for a, b in H['non_10ms_steps_3_to_5p5_s']) or 'none'}.", "",
          f"Mean T_i 3.60-3.95 s: {f3(H['early']['mean_eV'], 1)} eV ({H['early']['measured']} of {H['early']['rows']} rows measured, "
          f"{H['early']['blocks']} blocks); 4.10-4.90 s: {f3(H['late']['mean_eV'], 1)} eV ({H['late']['measured']} of "
          f"{H['late']['rows']} rows, {H['late']['blocks']} blocks). Intervals of the means: the same block resampling.", "",
          "| 100 ms from (s) | rows | measured | mean T_i (eV) | sd (eV) |", "|---|---|---|---|---|"]
    for m in H["means_100ms_3p5_to_5p0"]:
        L.append(f"| {m['from_s']:.1f} | {m['rows']} | {m['measured']} | "
                 f"{'' if m['mean_eV'] is None else f'{m['mean_eV']:.0f}'} | {'' if m['sd_eV'] is None else f'{m['sd_eV']:.0f}'} |")
    L += ["", "Discharge 31921, rows within 50 ms of the published profile times:", "",
          "| time (s) | rows | measured | mean T_i (eV) | sd (eV) | values above 3 keV removed |", "|---|---|---|---|---|---|"]
    for tc, o in H["s31921"].items():
        L.append(f"| {tc} | {o['rows']} | {o['measured']} | {'' if o['mean_eV'] is None else f'{o['mean_eV']:.0f}'} | "
                 f"{'' if o['sd_eV'] is None else f'{o['sd_eV']:.0f}'} | {o['values_above_3keV_removed']} |")
    L += ["", "## D. The fall of the estimate in panel (c) of the trace figure (descriptive)", "",
          f"Discharge {D['discharge']}, window {D['window_s'][0]:.3f}-{D['window_s'][1]:.3f} s (fig_trace.json: "
          f"{D['window_s_in_fig_trace_json'][0]:.3f}-{D['window_s_in_fig_trace_json'][1]:.3f}), measured range "
          f"{D['measured_range_eV'][0]:.1f}-{D['measured_range_eV'][1]:.1f} eV. Estimator: {D['estimator']}. Check against the "
          f"stored errors of the frozen evaluation ({D['check_vs_stored_errors']['scored_rows']} scored rows): largest deviation of "
          f"the absolute error {D['check_vs_stored_errors']['model']['max_abs_dev_of_error_eV']:.2e} eV (network), passed: "
          f"{D['check_passed']}. Lowest estimate in the window {D['min_estimate_in_window_eV']:.1f} eV.", "",
          "| time (s) | estimate (eV) | measured at the row (eV) | previous measured (eV, s) | next measured (eV, s) | ms since the last absent frame | mean BES departure (sd, B.32) | rows since blip (B.32) | ECEI / mean of 100 ms before |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in D["rows_below_measured_range"]:
        g = lambda v, fmt: "" if v is None else format(v, fmt)  # noqa: E731
        L.append(f"| {r['time_s']:.3f} | {r['estimate_eV']:.1f} | {g(r['measured_at_row_eV'], '.1f')} | "
                 f"{g(r['previous_measured_eV'], '.1f')}, {g(r['previous_measured_time_s'], '.3f')} | "
                 f"{g(r['next_measured_eV'], '.1f')}, {g(r['next_measured_time_s'], '.3f')} | {g(r['ms_since_last_absent_frame'], '.0f')} | "
                 f"{g(r['mean_bes_departure_sd'], '+.2f')} | {r['rows_since_blip_b32']} | {g(r['ecei_over_100ms_before'], '.3f')} |")
    c = D["context"]
    L += ["", f"Context: absent frames from 0.2 s before the window to its end: "
          f"{', '.join(f'{x:.3f}' for x in c['window_absent_frames']) or 'none'}; quantiles 0.1/0.5/0.9 of the mean BES departure "
          f"over the window rows {' / '.join(f'{x:+.2f}' for x in c['window_mean_bes_departure_sd_quantiles_10_50_90'])} sd.", "",
          f"Caption sentence: {res['D_sentence']}", ""]
    A = res["after_fact_S"]
    L += ["## After the fact (not registered; not used for any verdict)", "",
          f"1. Part S, 200 ms blocks, ECEI: pairs of used blocks per discharge, quantiles 0.1/0.5/0.9 "
          f"{' / '.join(f'{x:.0f}' for x in A['pairs_per_discharge_quantiles_10_50_90'])}; discharges with at least 10, 20, 40 "
          f"pairs: {A['discharges_with_ge']['10']}, {A['discharges_with_ge']['20']}, {A['discharges_with_ge']['40']}. The registered "
          "minimum of 40 pairs leaves few discharges; the same read-out with lower minima:", "",
          "| min pairs | discharges | median rho | median null | difference | share rho > 0 |", "|---|---|---|---|---|---|"]
    for mp, o in A["by_min_pairs"].items():
        L.append(f"| {mp} | {o['discharges']} | {f3(o['median_rho'], sign=True)} | {f3(o['median_null'], sign=True)} | "
                 f"{f3(o['diff'], sign=True)} | {o['share_rho_positive']:.3f} |")
    L.append("")
    L += ["## Checks before the read-out", ""] + [f"- {x}" for x in res["checks"]]
    L += ["", "## Expectations of the registration (section 5)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in res["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    L += ["", f"{sum(e['met'] for e in res['expectations'])} of {len(res['expectations'])} expectations met.", "",
          "## Deviations and choices left open by the registration", ""] + [f"- {x}" for x in res["interpretations"]]
    L += ["", f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


INTERP = [
    "Choice fixed before the run: part S uses only whole blocks; the trailing partial block of a segment is dropped. Block means "
    "of T_i are over the measured rows of the block, block means of the ECEI and BES levels over all its rows. The pairs of a "
    "discharge are taken over all its segments in time order, and the null shifts that vector circularly by floor(n/2).",
    "Choice fixed before the run: the 100 ms and 500 ms versions (reported, no statement) use blocks of 10 and 50 rows, at least "
    "half the rows measured, and at least 80 and 16 pairs (the same 8 s of data per discharge as 40 pairs of 200 ms).",
    "Choice fixed before the run: thirds of the discharge median T_i are taken over the discharges that enter the S read-out "
    "(median over the measured T_i rows of the cut grid).",
    "Choice fixed before the run: in part F 'missing' for the check is an empty CES_TI cell in the file (held values not counted "
    "as missing), over all grid rows. The 18.0 % and 3.4 % of file_rules.md were computed with the BES level standardized per "
    "discharge (z-score), not divided by the discharge median; the check is made with that definition, and the missing shares "
    "under the B.40 definition are reported beside the shares above 3 keV.",
    "Choice fixed before the run: 'within ten rows after an absent frame' is the B.32 rule (rows 1 to 10 after a step of more "
    "than 15 and less than 45 ms, inside the segment); the same share over all rows with a T_i value is given as the reference.",
    "Choice fixed before the run: part H windows include their end stamps (3.60 <= t <= 3.95 s, 4.10 <= t <= 4.90 s); 100 ms "
    "blocks start at the window start (the early window has a last block of 50 ms); blocks are resampled with replacement "
    "within each window independently, and a resampled mean is the sum of T_i over the measured rows of the drawn blocks over "
    "their count. Values above 3 keV are removed (cut grid).",
    "Correction after the first run (no definition, threshold or population changed): the first run took the rows of part F "
    "from the B.32 reader, which returns nothing for the 28 files with fewer than 12 grid rows, so 613 files (245,727 rows) "
    "entered. The registration says all grid rows, so the run was repeated with the same B.32 rule applied to all 641 files "
    "(checked identical to the reader on the 613). F1 changed from 3.18 [2.51, 4.03] to 3.18 [2.49, 4.04]; the missing-share "
    "check from 17.95 / 3.37 % to 17.96 / 3.37 %. Part D's expectation check now rounds the time since the absent frame to "
    "0.1 ms (the first run read 9.9999 ms as below 10).",
    "Observation on H1 (verdict kept as registered): the 10 ms file of 31923 begins at 3.512 s, so no absent frame can lie "
    "before or at 3.5 s in it; the published blip at 3.5 s coincides with the start of the delivered grid (12 ms after 3.5 s). "
    "The rule was applied as written and H1 does not hold on that clause; the clause near 5.0 s and the ratio interval are met.",
    "Deviation: the registration names 'the estimate of the causal variant' in panel (c). The figure (fig_trace.py and the "
    "caption) draws the recurrent estimator with whole-discharge standardization (data/.b1_seqv2_s1_i1), not the causal "
    "variant. Part D describes the estimate that is drawn, recomputed on the CPU with the figure's own functions and checked "
    "against the stored errors; the causal variant was not run.",
]


def d_sentence(D):
    rows = D["rows_below_measured_range"]
    if not rows:
        return "No estimate in the window lies below the measured range."
    ms = [r["ms_since_last_absent_frame"] for r in rows]
    dd = [r["mean_bes_departure_sd"] for r in rows]
    ee = [r["ecei_over_100ms_before"] for r in rows]
    parts = [f"{len(rows)} row(s) at {', '.join(f'{r['time_s']:.3f}' for r in rows)} s with estimates "
             f"{', '.join(f'{r['estimate_eV']:.0f}' for r in rows)} eV (measured range {D['measured_range_eV'][0]:.0f} to "
             f"{D['measured_range_eV'][1]:.0f} eV)"]
    parts.append("time since the last absent frame " + ", ".join("none in the file before" if m is None else f"{m:.0f} ms" for m in ms))
    parts.append("mean BES departure " + ", ".join("n/a" if d is None else f"{d:+.2f} sd" for d in dd))
    parts.append("ECEI level over the 100 ms before " + ", ".join("n/a" if e is None else f"{e:.3f}" for e in ee))
    return "; ".join(parts) + "."


def main():
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT}
    checks = []
    rng = np.random.default_rng(SEED)
    grid_cut, dims = load_grid_files(DATA, True, CUT_EV)
    names = sorted(grid_cut)
    nb, ne = dims["bes"], dims["ecei"]
    if len(names) != 641 or nb != 9 or ne != 4:
        raise SystemExit(f"FATAL: expected 641 files, 9 BES and 4 ECEI channels; got {len(names)}, {nb}, {ne}")
    bad_edges = [nm for nm in names if not (abs(grid_cut[nm][0, 0]) < 1e-6 and abs(grid_cut[nm][-1, 0] - 30.0) < 1e-4)]
    if bad_edges:
        raise SystemExit(f"FATAL: files without the t = 0 / t = 30 s edge rows: {bad_edges[:5]}")
    checks.append("641 files, 9 BES and 4 ECEI channels; every file opens at t = 0 and closes at t = 30 s, and these two "
                  "rows are dropped in every tally.")
    log("cut grid loaded")
    res["S"] = part_s(grid_cut, names, nb, ne, np.random.default_rng(SEED + 1))
    d = res["S"]["200ms"]["ecei"]["diff"]
    res["S1"] = {"diff": d, "holds": bool(d[1] > 0)}
    res["after_fact_S"] = part_s_sensitivity(grid_cut, names, nb, ne, np.random.default_rng(SEED + 11))
    grid_incl, _ = load_grid_files(DATA, True, 0.0)
    grid_raw, _ = load_grid_files(DATA, False, 0.0)
    log("inclusive and raw grids loaded")
    res["F"] = part_f(grid_incl, grid_raw, names, nb, np.random.default_rng(SEED + 2))
    del grid_raw
    mz = res["F"]["missing_zscore_check"]["share_by_fifth"]
    ok18 = abs(100 * mz[0][0] - 18.0) <= 0.5 and abs(100 * mz[4][0] - 3.4) <= 0.5
    checks.append(f"Missing T_i by fifth of the BES level, file_rules.md definition (per-discharge z-score, "
                  f"{res['F']['missing_zscore_check']['rows']:,} rows): lowest {100 * mz[0][0]:.2f} %, highest {100 * mz[4][0]:.2f} % "
                  f"(to reproduce 18.0 % and 3.4 % within 0.5 points): {'reproduced' if ok18 else 'NOT reproduced'}. "
                  f"All five: {', '.join(f'{100 * x[0]:.1f}' for x in mz)} %.")
    checks.append(f"Part F covers {res['F']['discharges']} files ({res['F']['grid_rows']:,} grid rows; "
                  f"{res['F']['discharges_with_ti']} with at least one T_i value); raw and held-free grids misaligned in "
                  f"{res['F']['files_misaligned']} files. The B.32 reader (kstar_file, None for files under 12 rows) was "
                  f"compared on {res['F']['b32_reader_files_compared']} files: times, rows-after-blip and missing T_i differ "
                  f"in {res['F']['b32_reader_files_differing']}.")
    res["H"] = part_h(grid_cut, grid_incl, np.random.default_rng(SEED + 3))
    del grid_cut, grid_incl
    res["D"] = part_d()
    cd = res["D"]
    checks.append(f"Part D: CPU recomputation against the stored squared errors of the frozen evaluation for {cd['discharge']} "
                  f"({cd['check_vs_stored_errors']['scored_rows']} rows): network max deviation "
                  f"{cd['check_vs_stored_errors']['model']['max_abs_dev_of_error_eV']:.2e} eV, persistence and causal GP share within "
                  f"1e-3 {cd['check_vs_stored_errors']['persistence']['share_within_1e-3']:.3f} / "
                  f"{cd['check_vs_stored_errors']['causal_gp']['share_within_1e-3']:.3f}; window and measured range equal to "
                  f"fig_trace.json: {cd['check_passed']}.")
    res["D_sentence"] = d_sentence(cd)
    res["checks"] = checks
    res["expectations"] = expectations(res)
    res["interpretations"] = INTERP
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"S1 {res['S1']['holds']} F1 {res['F']['F1']['holds']} H1 {res['H']['H1']['holds']}; written")


if __name__ == "__main__":
    main()
