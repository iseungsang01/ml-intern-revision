"""B.58 (PREREGISTRATION_B58.md): beam blips as a calibration of how the reported T_i responds to the beam (part S),
and the lag of the association away from blips (part L).  CPU only, no model, real 10 ms files (and the raw BES of
the twelve discharges with raw streams, for a reported add-on that is not a statement).

The definitions and thresholds were fixed after an exploratory look at the same data.

The exploratory scripts uc_blip_shape.py and uc_lag_quiet.py are copied unchanged into this folder; their SHA-256 is
written into the output.  Importing them runs them (they are top-level scripts): uc_blip_shape provides the event
arrays and the profile statistic; uc_lag_quiet provides the all-discharge lag medians, which this runner's per-
discharge re-implementation (needed for halves and paired bootstrap; the code is the same as in uc_lag_quiet) must
reproduce exactly (check written to the output).

Run from the repo root:  py ces_prediction/experiments/b58_blip_calibration/run_b58.py
Outputs: docs/paper/b58_blip_calibration.{json,md}, docs/paper/b58_blip_profiles.{png,pdf}; cache data/.b58/
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import contextlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

T0 = _time.time()
NB = 2000
SEED = 20261118
CACHE = REPO / "data" / ".b58"
PAPER = REPO / "docs" / "paper"
OUT_JSON = PAPER / "b58_blip_calibration.json"
OUT_MD = PAPER / "b58_blip_calibration.md"
FIG = PAPER / "b58_blip_profiles"
REG = "ces_prediction/experiments/PREREGISTRATION_B58.md"
FROZEN = ("uc_blip_shape.py", "uc_lag_quiet.py")
LAGS = list(range(-4, 5))


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


# ---------------------------------------------------------------------------------- import the frozen scripts
def import_frozen(name):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = __import__(name)
    return mod, buf.getvalue()


log("importing uc_blip_shape (runs the exploratory shape script)")
SH, sh_out = import_frozen("uc_blip_shape")
log("importing uc_lag_quiet (runs the exploratory lag script)")
LQ, lq_out = import_frozen("uc_lag_quiet")
files = sorted(F.DATA.glob("s*.csv"))
assert [p.name for p in files] == [p.name for p in sorted(SH.files)] == [p.name for p in sorted(LQ.files)]

# ---------------------------------------------------------------------------------------------- part S, shape
K = SH.K
ev = SH.ev
ctl = SH.ctl
evf, ctf, evn = np.asarray(SH.evf), np.asarray(SH.ctf), np.asarray(SH.evn)
HALF = {"all": lambda f: np.ones(len(f), bool), "half_even": lambda f: f % 2 == 0, "half_odd": lambda f: f % 2 == 1}


def boot_profiles(d, fidx, sel, seed):
    """Event-median profiles (SH.stat) on the full selection and on NB resamples of discharges."""
    idx = np.flatnonzero(sel)
    f = fidx[idx]
    uf = np.unique(f)
    pos = {u: idx[f == u] for u in uf}
    r = np.random.default_rng(seed)
    full = SH.stat(d, fidx, idx)
    res = []
    for _ in range(NB):
        pick = r.choice(uf, size=len(uf))
        res.append(SH.stat(d, fidx, np.concatenate([pos[u] for u in pick])))
    return full, res, int(len(idx)), int(len(uf))


def unexplained(m, q):
    y, x = 1 - m["ti"][:8], 1 - m[q][:8]
    a = (x @ y) / (x @ x)
    return float(np.sum((y - a * x) ** 2) / np.sum(y ** 2))


def two_reg(m):
    y = 1 - m["ti"][:8]
    X = np.c_[1 - m["ecei"][:8], 1 - m["bes"][:8]]
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b), float(b * (1 - m["bes"][0]))


def part_s(sel_ev, sel_ct, seed):
    full, B, n_ev, n_dis = boot_profiles(ev, evf, sel_ev, seed)
    cfull = SH.stat(ctl, ctf, np.flatnonzero(sel_ct))
    out = {"events": n_ev, "discharges": n_dis, "control_pseudo_events": int(sel_ct.sum()),
           "profiles": {q: [float(v) for v in full[q]] for q in ("ti", "ecei", "bes")},
           "control_profiles": {q: [float(v) for v in cfull[q]] for q in ("ti", "ecei", "bes")}}
    out["profiles_ci95"] = {q: [ci([m[q][k] for m in B]) for k in range(len(K))] for q in ("ti", "ecei", "bes")}
    out["shape_index"] = {q: {"point": float(SH.shape(full, q)), "ci95": ci([SH.shape(m, q) for m in B])}
                          for q in ("ti", "ecei", "bes")}
    for q in ("ecei", "bes"):
        out[f"unexplained_{q}"] = {"point": unexplained(full, q), "ci95": ci([unexplained(m, q) for m in B])}
    ab = np.array([two_reg(m) for m in B])
    a0, b0, p0 = two_reg(full)
    d1 = float(1 - full["ti"][0])
    out["a"] = {"point": a0, "ci95": ci(ab[:, 0])}
    out["b"] = {"point": b0, "ci95": ci(ab[:, 1])}
    out["b_times_first_row_bes_deficit"] = {"point": p0, "ci95": ci(ab[:, 2])}
    out["first_row_ti_deficit"] = {"point": d1, "ci95": ci([1 - m["ti"][0] for m in B])}
    out["share_of_first_row_deficit_proportional_to_bes"] = {
        "point": p0 / d1, "ci95": ci(ab[:, 2] / np.array([1 - m["ti"][0] for m in B]))}
    # statements
    s1 = out["unexplained_ecei"]["ci95"][1] < 0.10 and out["unexplained_bes"]["ci95"][0] > 0.20
    s2 = out["b_times_first_row_bes_deficit"]["ci95"][1] < d1 / 3.0
    cdev = {q: float(np.max(np.abs(np.array(cfull[q]) - 1))) for q in ("ti", "ecei", "bes")}
    s3 = cdev["ecei"] <= 0.015 and cdev["bes"] <= 0.015 and cdev["ti"] <= 0.02
    out["control_max_abs_deviation"] = cdev
    out["S1_holds"], out["S2_holds"], out["S3_holds"] = bool(s1), bool(s2), bool(s3)
    out["S2_threshold_one_third_of_first_row_deficit"] = d1 / 3.0
    return out, full


def event_assoc(sel, seed):
    """Event by event, rows 4-6: T_i deficit vs remaining BES deficit, plain and partial on ECEI (rank)."""
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.nanmean(1 - ev["ti"][:, 3:6], 1)
            e = np.nanmean(1 - ev["ecei"][:, 3:6], 1)
            bb = np.nanmean(1 - ev["bes"][:, 3:6], 1)
    ok = sel & np.isfinite(a) & np.isfinite(e) & np.isfinite(bb)
    a, e, bb, f = a[ok], e[ok], bb[ok], evf[ok]

    def resid(y, x):
        X = np.c_[np.ones_like(x), x]
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    def stats(ii):
        ra, re_, rb_ = rankdata(a[ii]), rankdata(e[ii]), rankdata(bb[ii])
        return (spearmanr(a[ii], bb[ii])[0], np.corrcoef(resid(ra, re_), resid(rb_, re_))[0, 1],
                spearmanr(a[ii], e[ii])[0], np.corrcoef(resid(ra, rb_), resid(re_, rb_))[0, 1])

    full = stats(np.arange(len(a)))
    uf = np.unique(f)
    pos = {u: np.flatnonzero(f == u) for u in uf}
    r = np.random.default_rng(seed)
    bs = np.array([stats(np.concatenate([pos[u] for u in r.choice(uf, size=len(uf))])) for _ in range(NB)])
    names = ("rank_ti_bes", "partial_ti_bes_given_ecei", "rank_ti_ecei", "partial_ti_ecei_given_bes")
    return {"events": int(len(a)), **{n: {"point": float(full[i]), "ci95": ci(bs[:, i])} for i, n in enumerate(names)}}


log("part S")
S = {}
full_profiles = {}
for h, fn in HALF.items():
    sel = (evn == 1) & fn(evf)
    S[h], full_profiles[h] = part_s(sel, fn(ctf), SEED + len(h))
    S[h]["event_by_event_rows_4_to_6"] = event_assoc(sel, SEED + 7 + len(h))
    log(f"  S {h}: S1 {S[h]['S1_holds']} S2 {S[h]['S2_holds']} S3 {S[h]['S3_holds']}")
S_apart = {}
for n in (2, 3):
    o, _ = part_s(evn == n, np.ones(len(ctf), bool), SEED + 100 + n)
    S_apart[f"{n}_absent_frames"] = {k: o[k] for k in ("events", "discharges", "profiles", "shape_index",
                                                       "unexplained_ecei", "unexplained_bes", "a", "b")}

# kernel of the heating route: normalized event-median T_i deficit profile of part S (registered set, all discharges)
H_RUN = 1 - np.array(full_profiles["all"]["ti"])
H_RUN = H_RUN / H_RUN.sum()
H_FROZEN = np.asarray(LQ.H, float)

# ------------------------------------------------------------------------------------------------ part L, lag
BLK = LQ.BLK


def per_file(f, H_list):
    """Per-discharge lag correlations on quiet stretches -- the code of uc_lag_quiet, per file -- plus the kernel
    predictions for each H in H_list, the lag-0 partials, the s.d. of 30 ms changes of the BES common mode, the
    median T_i, and the positive control on windows that contain a one-frame blip (alignments 0, 1, 2)."""
    df, *_ = F.load(f)
    t = df["time"].to_numpy(float)
    res = {"rho": {}, "null": {}, "ctrl": {}, "partial": None, "sd_c": np.nan, "ti_med": np.nan}
    if len(t) < 200:
        return res
    ti = df["CES_TI"].to_numpy(float).copy()
    ti[ti > 3000] = np.nan
    B = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float)
    E = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(np.clip(B, 1e-4, None))
    c = L.mean(1)
    cols = [c_ for c_ in df.columns if c_.startswith("BES_")]
    col07 = [i for i, n in enumerate(cols) if n.endswith("07")]
    col05 = [i for i, n in enumerate(cols) if n.endswith("05")]
    g = L[:, col07].mean(1) - L[:, col05].mean(1)
    res["ti_med"] = float(np.nanmedian(ti)) if np.isfinite(ti).any() else np.nan
    bad = np.zeros(len(t), bool)
    dt = np.diff(t)
    for j in np.flatnonzero(dt > 0.015):
        bad[max(0, j - 3):j + 16] = True
    ok = ~bad
    runs, start = [], None
    for i in range(len(t)):
        if ok[i] and (start is None or abs(t[i] - t[i - 1] - 0.010) < 1e-4):
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 30:
                runs.append((start, i))
            start = i if ok[i] else None
    if start is not None and len(t) - start >= 30:
        runs.append((start, len(t)))
    keys = ["c", "g", "e"] + [f"k{i}" for i in range(len(H_list))]
    pairs = {q: {l: ([], []) for l in LAGS} for q in keys}
    trip = ([], [], [])
    dcs = []
    import warnings
    for a, b in runs:
        n = (b - a) // BLK
        if n < 8:
            continue
        sl = slice(a, a + n * BLK)

        def bm(x):
            x = x[sl].reshape(n, BLK)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmean(x, 1)
        T = bm(ti)
        X = {"c": bm(c), "g": bm(g), "e": bm(E)}
        dT = np.diff(T)
        cc = c[a:a + n * BLK]
        Yk = {}
        for i, H in enumerate(H_list):
            kp = np.convolve(cc, H)[:len(cc)]
            kp[:len(H)] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Yk[f"k{i}"] = np.diff(np.nanmean(kp.reshape(n, BLK), 1))
            X[f"k{i}"] = X["c"]
        dcs.extend(np.diff(X["c"]).tolist())
        trip[0].extend(dT.tolist())
        trip[1].extend(np.diff(X["c"]).tolist())
        trip[2].extend(np.diff(X["e"]).tolist())
        for q in keys:
            dX = np.diff(X[q])
            dY = Yk[q] if q.startswith("k") else dT
            for l in LAGS:
                if l >= 0:
                    xx, yy = dX[:len(dX) - l], dY[l:]
                else:
                    xx, yy = dX[-l:], dY[:len(dY) + l]
                pairs[q][l][0].extend(xx.tolist())
                pairs[q][l][1].extend(yy.tolist())
    for q in keys:
        for l in LAGS:
            x, y = np.array(pairs[q][l][0]), np.array(pairs[q][l][1])
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 20 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                res["rho"][(q, l)] = spearmanr(x[m], y[m])[0]
                yy = np.roll(y[m], m.sum() // 2)
                res["null"][(q, l)] = spearmanr(x[m], yy)[0]
    d = np.array(dcs, float)
    d = d[np.isfinite(d)]
    if len(d) >= 20:
        res["sd_c"] = float(np.std(d))
    yT, xc, xe = (np.array(v, float) for v in trip)
    m = np.isfinite(yT) & np.isfinite(xc) & np.isfinite(xe)
    if m.sum() >= 20 and np.std(xc[m]) > 0 and np.std(xe[m]) > 0 and np.std(yT[m]) > 0:
        ry, rc, re_ = rankdata(yT[m]), rankdata(xc[m]), rankdata(xe[m])

        def resid(y, x):
            Xm = np.c_[np.ones_like(x), x]
            return y - Xm @ np.linalg.lstsq(Xm, y, rcond=None)[0]
        res["partial"] = (float(np.corrcoef(resid(ry, re_), resid(rc, re_))[0, 1]),
                          float(np.corrcoef(resid(ry, rc), resid(re_, rc))[0, 1]))
    # positive control: windows containing a one-frame blip.  Regular 10 ms grid of 45 positions, the absent stamp
    # at position 15 + shift (shift = 0 registered reading: the absent frame opens a block; 1, 2 reported), values
    # NaN there; 15 - (0) ... rows before must be contiguous 10 ms, rows after contiguous 10 ms.
    for shift in (0, 1, 2):
        pc = {l: ([], []) for l in LAGS}
        npos = 45
        p_abs = 15 + shift
        for j in np.flatnonzero((dt > 0.015) & (dt < 0.025)):      # one absent frame (20 ms step)
            nb_ = p_abs                    # rows before (positions 0 .. p_abs-1): rows j-p_abs+1 .. j
            na_ = npos - p_abs - 1         # rows after: j+1 .. j+na_
            if j - nb_ + 1 < 0 or j + na_ >= len(t):
                continue
            if not np.all(np.abs(dt[j - nb_ + 1:j] - 0.010) < 1e-4):
                continue
            if not np.all(np.abs(dt[j + 1:j + na_] - 0.010) < 1e-4):
                continue
            idx = np.r_[np.arange(j - nb_ + 1, j + 1), -1, np.arange(j + 1, j + na_ + 1)]

            def grid(x):
                v = x[np.clip(idx, 0, None)].astype(float)
                v[idx < 0] = np.nan
                return v
            n = npos // BLK
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Tb = np.nanmean(grid(ti).reshape(n, BLK), 1)
                Cb = np.nanmean(grid(c).reshape(n, BLK), 1)
            dT, dX = np.diff(Tb), np.diff(Cb)
            for l in LAGS:
                if l >= 0:
                    xx, yy = dX[:len(dX) - l], dT[l:]
                else:
                    xx, yy = dX[-l:], dT[:len(dT) + l]
                pc[l][0].extend(xx.tolist())
                pc[l][1].extend(yy.tolist())
        for l in LAGS:
            x, y = np.array(pc[l][0]), np.array(pc[l][1])
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 20 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                res["ctrl"][(shift, l)] = spearmanr(x[m], y[m])[0]
    return res


cache = CACHE / "lag_per_file.npz"
log("part L: per-discharge lag analysis (all files)")
PF = [per_file(f, [H_FROZEN, H_RUN]) for f in files]
log("part L done")


def collect(getter):
    """(file rank array, value array) over discharges where the value exists."""
    fi, v = [], []
    for i, r in enumerate(PF):
        x = getter(r)
        if x is not None and np.isfinite(x):
            fi.append(i)
            v.append(x)
    return np.array(fi, int), np.array(v, float)


# reproduction check against the imported exploratory script (all discharges, frozen kernel = k0)
repro = {}
for q, qm in (("c", "c"), ("g", "g"), ("e", "e"), ("k0", "k")):
    for l in LAGS:
        _, v = collect(lambda r: r["rho"].get((q, l)))
        mine = float(np.median(v))
        theirs = float(np.median(np.array(LQ.per[qm][l])))
        repro[f"{q}_lag{l:+d}"] = [mine, theirs, len(v), len(LQ.per[qm][l])]
repro_ok = all(abs(a - b) < 1e-12 and n1 == n2 for a, b, n1, n2 in repro.values())
log(f"reproduction of uc_lag_quiet medians: {repro_ok}")
if not repro_ok:
    raise SystemExit("FATAL: the per-discharge re-implementation does not reproduce uc_lag_quiet")


def lag_block(sel_fn, seed, key_fn=lambda q, l: ("rho", (q, l)), qs=("c", "g", "e", "k0", "k1")):
    """Median over discharges of the per-discharge correlation at each lag, with NB paired resamples of discharges;
    ratio lag(+1)/lag(0) per resample."""
    r = np.random.default_rng(seed)
    nd = len(PF)
    keep = sel_fn(np.arange(nd))
    kind0 = key_fn(qs[0], 0)[0]
    U = np.array([i for i in range(nd) if keep[i] and len(PF[i][kind0]) > 0], int)   # discharges with values
    Wu = r.multinomial(len(U), np.full(len(U), 1.0 / len(U)), size=NB)
    W = np.zeros((NB, nd), np.int64)
    W[:, U] = Wu                                                     # counts per discharge rank, paired across lags
    out = {}
    for q in qs:
        o = {"lags": {}}
        meds_b = {}
        for l in LAGS:
            kind, key = key_fn(q, l)
            fi, v = collect(lambda rr: rr[kind].get(key))
            mk = keep[fi]
            fi, v = fi[mk], v[mk]
            _, vn = collect(lambda rr: rr["null"].get(key)) if kind == "rho" else (None, None)
            o["lags"][str(l)] = {"median": float(np.median(v)), "n": int(len(v))}
            if kind == "rho":
                fin, vn = collect(lambda rr: rr["null"].get(key))
                vn = vn[keep[fin]]
                o["lags"][str(l)]["null_median"] = float(np.median(vn))
            # weighted median under each resample
            order = np.argsort(v)
            vs, fs = v[order], fi[order]
            cw = np.cumsum(W[:, fs], 1).astype(float)
            tot = cw[:, -1:]
            bm = vs[(cw >= tot / 2).argmax(1)]
            bm[tot[:, 0] == 0] = np.nan
            meds_b[l] = bm
            o["lags"][str(l)]["ci95"] = ci(bm)
        rat = meds_b[1] / meds_b[0]
        o["ratio_lag_plus30_over_lag0"] = {"point": o["lags"]["1"]["median"] / o["lags"]["0"]["median"],
                                           "ci95": ci(rat)}
        out[q] = o
    return out


L = {}
for h, fn in HALF.items():
    blk = lag_block(fn, SEED + 200 + len(h))
    ctrl = {}
    for shift in (0, 1, 2):
        ctrl[f"shift{shift}"] = lag_block(fn, SEED + 300 + shift + len(h),
                                          key_fn=lambda q, l, s=shift: ("ctrl", (s, l)), qs=("c",))["c"]
    # partial at lag 0
    part = {}
    for i, nm in enumerate(("ti_bes_given_ecei", "ti_ecei_given_bes")):
        fi, v = collect(lambda r, i=i: None if r["partial"] is None else r["partial"][i])
        v = v[fn(fi)]
        r_ = np.random.default_rng(SEED + 400 + i)
        bs = [np.median(r_.choice(v, len(v))) for _ in range(NB)]
        part[nm] = {"median": float(np.median(v)), "ci95": ci(bs), "n": int(len(v))}
    L1 = (blk["c"]["ratio_lag_plus30_over_lag0"]["ci95"][1] < 0.5
          and blk["k1"]["ratio_lag_plus30_over_lag0"]["ci95"][0] > 0.7)
    L1_frozen_kernel = (blk["c"]["ratio_lag_plus30_over_lag0"]["ci95"][1] < 0.5
                        and blk["k0"]["ratio_lag_plus30_over_lag0"]["ci95"][0] > 0.7)
    L2 = blk["e"]["ratio_lag_plus30_over_lag0"]["ci95"][1] < 0.5
    c0 = ctrl["shift0"]
    L3 = c0["lags"]["1"]["ci95"][0] > 0 and c0["ratio_lag_plus30_over_lag0"]["ci95"][0] > 0.5
    L[h] = {"quiet": blk, "positive_control": ctrl, "partial_lag0": part, "L1_holds": bool(L1),
            "L1_holds_with_frozen_kernel": bool(L1_frozen_kernel), "L2_holds": bool(L2), "L3_holds": bool(L3)}
    log(f"  L {h}: L1 {L1} L2 {L2} L3 {L3}")

# S2 bound in eV on quiet rows: upper end of b (registered set, all discharges) x sd of 30 ms changes of the BES
# common mode x the T_i level, per discharge
b_up = S["all"]["b"]["ci95"][1]
fi_sd, sd = collect(lambda r: r["sd_c"])
fi_ti, tim = collect(lambda r: r["ti_med"])
common = np.intersect1d(fi_sd, fi_ti)
sdd = dict(zip(fi_sd, sd))
tid = dict(zip(fi_ti, tim))
ev_bound = np.array([b_up * sdd[i] * tid[i] for i in common])
bound_ev = {"b_upper": b_up, "discharges": int(len(common)),
            "sd_30ms_change_bes_common_median": float(np.median([sdd[i] for i in common])),
            "ti_level_median_eV": float(np.median([tid[i] for i in common])),
            "bound_eV_median": float(np.median(ev_bound)),
            "bound_eV_quartiles": [float(np.quantile(ev_bound, 0.25)), float(np.quantile(ev_bound, 0.75))],
            "bound_eV_p90": float(np.quantile(ev_bound, 0.9))}
log(f"S2 bound in eV: median {bound_ev['bound_eV_median']:.2f}")


# ----------------------------------------------------------------------- raw-stream add-on (twelve discharges)
def raw_addon():
    rows = []
    for shot in F.RAW_SHOTS:
        df, *_ = F.load(F.DATA / f"s{shot}.csv")
        t = df["time"].to_numpy(float)
        ti = df["CES_TI"].to_numpy(float).copy()
        ti[ti > 3000] = np.nan
        ecei = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float).mean(1)
        bes = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float).mean(1)
        x, fs, t0, _ch = F.raw_npz(shot, "BES")
        lvl_x = x.mean(1).astype(np.float64)
        del x
        w = int(round(fs * 1e-4))
        n1 = len(lvl_x) // w
        lvl = lvl_x[:n1 * w].reshape(n1, w).mean(1)
        tb = t0 + np.arange(n1) * 1e-4
        ref = pd.Series(lvl).rolling(2000, center=True, min_periods=200).median().to_numpy()
        low = lvl < 0.5 * ref
        d = np.diff(np.concatenate(([0], low.astype(int), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        keep = (en - st) >= 30
        events = [(float(tb[a]), float(tb[min(b, n1 - 1)])) for a, b in zip(st[keep], en[keep])
                  if tb[a] > t0 + 0.05 and tb[min(b, n1 - 1)] < tb[-1] - 0.05]
        for j, stamps in F.absent_groups(t):
            if not (t0 + 0.05 < stamps[0] and stamps[-1] < tb[-1] - 0.05):
                continue
            hit = [i for i, (a, b) in enumerate(events) if a < stamps[-1] + 0.010 and b > stamps[0] - 0.005]
            if not hit or j < 5:
                continue
            pre = slice(j - 5, j + 1)
            if not np.all(np.abs(np.diff(t[pre]) - 0.010) < 1e-4) or j + 1 >= len(t):
                continue
            tip = np.nanmedian(ti[pre]) if np.isfinite(ti[pre]).sum() >= 3 else np.nan
            a, b = events[hit[0]][0], events[hit[-1]][1]
            rows.append({"shot": shot, "absent_frames": len(stamps), "duration_ms": (b - a) * 1e3,
                         "end_minus_last_stamp_ms": (b - stamps[-1]) * 1e3,
                         "ti_ratio_row1": float(ti[j + 1] / tip) if np.isfinite(tip) else None,
                         "ecei_ratio_row1": float(ecei[j + 1] / np.median(ecei[pre])),
                         "bes_ratio_row1": float(bes[j + 1] / np.median(bes[pre]))})
        log(f"  raw {shot}: {sum(r['shot'] == shot for r in rows)} blips")
    one = [r for r in rows if r["absent_frames"] == 1]
    summ = {"blips": len(rows), "one_absent_frame": len(one)}
    for q in ("ti_ratio_row1", "ecei_ratio_row1", "bes_ratio_row1"):
        pr = [(r["duration_ms"], r[q]) for r in one if r[q] is not None and np.isfinite(r[q])]
        if len(pr) >= 5:
            dd, vv = np.array(pr).T
            summ[f"rank_duration_vs_{q}"] = {"rho": float(spearmanr(dd, vv)[0]), "n": int(len(pr))}
        pr2 = [(r["end_minus_last_stamp_ms"], r[q]) for r in one if r[q] is not None and np.isfinite(r[q])]
        if len(pr2) >= 5:
            dd, vv = np.array(pr2).T
            summ[f"rank_beam_return_vs_{q}"] = {"rho": float(spearmanr(dd, vv)[0]), "n": int(len(pr2))}
    return rows, summ


log("raw-stream add-on")
RAW_ROWS, RAW_SUMM = raw_addon()


# ------------------------------------------------------------------------------------------------ figure
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, c = S["all"]["profiles"], S["all"]["control_profiles"]
    pci = S["all"]["profiles_ci95"]
    k = np.array(K)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    cols = {"ti": "#1f4e9a", "ecei": "#c0392b", "bes": "#7f8c8d"}
    lab = {"ti": "reported $T_i$", "ecei": "ECEI level", "bes": "BES level"}
    for q in ("bes", "ecei", "ti"):
        lo = [x[0] for x in pci[q]]
        hi = [x[1] for x in pci[q]]
        ax.fill_between(k, lo, hi, color=cols[q], alpha=0.15, lw=0)
        ax.plot(k, p[q], "-o", color=cols[q], ms=3, lw=1.4)
        ax.plot(k, c[q], ":", color=cols[q], lw=1.0)
    ax.text(k[0] + 0.2, p["ti"][0] - 0.005, lab["ti"], color=cols["ti"], fontsize=8, va="top")
    ax.text(k[1] + 0.2, p["ecei"][1] - 0.012, lab["ecei"], color=cols["ecei"], fontsize=8, va="top")
    ax.text(k[0] + 0.25, p["bes"][0], lab["bes"], color=cols["bes"], fontsize=8, va="center")
    ax.text(k[2], 1.04, "dotted: controls (no absent frame)", color="0.3", fontsize=7, ha="left")
    ax.axhline(1.0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("row after the absent frame (10 ms frames)")
    ax.set_ylabel("ratio to the median of the six rows before")
    ax.set_xticks(k)
    ax.set_ylim(0.6, 1.06)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".png"))
    fig.savefig(FIG.with_suffix(".pdf"))
    plt.close(fig)


figure()

# ------------------------------------------------------------------------------------------------ outputs
res = {"registration": REG,
       "note": "The definitions and thresholds were fixed after an exploratory look at the same data.",
       "frozen_scripts_sha256": {n: sha(HERE / n) for n in FROZEN},
       "runner_sha256": sha(Path(__file__)),
       "bootstrap_resamples": NB, "seed": SEED,
       "halves": "parity of the rank of the discharge file in the sorted list of all 641 files (even / odd)",
       "kernel_frozen_in_uc_lag_quiet": [float(x) for x in H_FROZEN],
       "kernel_registered_part_S_one_frame": [float(x) for x in H_RUN],
       "reproduction_of_uc_lag_quiet": {"ok": repro_ok, "median_mine_theirs_n_mine_n_theirs": repro},
       "S": S, "S_two_and_three_absent_frames": S_apart, "L": L, "S2_bound_eV_quiet_rows": bound_ev,
       "raw_addon": {"rows": RAW_ROWS, "summary": RAW_SUMM},
       "runtime_s": _time.time() - T0}
CACHE.mkdir(exist_ok=True)
OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


def f3(x):
    return f"{x:+.3f}" if x is not None else "-"


def iv(d):
    return f"{d['point']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def hv(x):
    return "holds" if x else "does not hold"


H3 = ("all", "half_even", "half_odd")
M = ["# B.58 Beam blips as a calibration of the reported value, and the lag away from blips", "",
     "**The definitions and thresholds were fixed after an exploratory look at the same data.** "
     f"Registration: `{REG}`. Runner: `ces_prediction/experiments/b58_blip_calibration/run_b58.py`. "
     "CPU only, no model, real files. Intervals: 95 %, 2,000 resamples of discharges. Halves: parity of the rank of "
     "the discharge file in the sorted list of 641 files.", "",
     "Frozen definitions (copied unchanged): " + ", ".join(f"`{n}` sha256 `{res['frozen_scripts_sha256'][n]}`" for n in FROZEN)
     + f". The per-discharge re-implementation of `uc_lag_quiet.py` in the runner reproduces its all-discharge medians exactly: {repro_ok}.", "",
     "## Verdicts", "",
     "| statement | all discharges | even half | odd half | verdict | expected |", "|---|---|---|---|---|---|"]


def srow(name, key, exp):
    v = [S[h][key] for h in H3]
    return f"| {name} | {hv(v[0])} | {hv(v[1])} | {hv(v[2])} | **{hv(all(v))}** | {exp} |"


def lrow(name, key, exp):
    v = [L[h][key] for h in H3]
    return f"| {name} | {hv(v[0])} | {hv(v[1])} | {hv(v[2])} | **{hv(all(v))}** | {exp} |"


l3_all = all(L[h]["L3_holds"] for h in H3)
M += [srow("S1 shape", "S1_holds", "0.01 and 0.34, holds, also in each half"),
      srow("S2 signal-level part < 1/3 of the first-row deficit", "S2_holds",
           "upper end 0.046 against 0.061, holds on all; a half may reach 1/3"),
      srow("S3 control", "S3_holds", "holds"),
      lrow("L1 lag ratio, BES common < 0.5; heating-route prediction > 0.7", "L1_holds", "0.15 against 0.90, holds"),
      lrow("L2 lag ratio, ECEI < 0.5", "L2_holds", "about 0.07, holds"),
      lrow("L3 positive control (windows with a blip)", "L3_holds", "holds, ratio 0.6 to 1.0"), ""]
if not l3_all:
    M += ["L3 does not hold everywhere: by the registration, **L1 and L2 are withdrawn as uninformative**.", ""]
M += ["## Part S, shape (registered set: one absent frame)", "",
      "| quantity | all | even half | odd half | threshold |", "|---|---|---|---|---|"]
M.append("| events (discharges) | " + " | ".join(f"{S[h]['events']} ({S[h]['discharges']})" for h in H3) + " | |")
M.append("| unexplained share, T_i proportional to ECEI | " + " | ".join(iv(S[h]["unexplained_ecei"]) for h in H3) + " | upper end < 0.10 |")
M.append("| unexplained share, T_i proportional to BES | " + " | ".join(iv(S[h]["unexplained_bes"]) for h in H3) + " | lower end > 0.20 |")
M.append("| b x first-row BES deficit | " + " | ".join(iv(S[h]["b_times_first_row_bes_deficit"]) for h in H3) + " | upper end < 1/3 of first-row T_i deficit |")
M.append("| one third of the first-row T_i deficit | " + " | ".join(f"{S[h]['S2_threshold_one_third_of_first_row_deficit']:.3f}" for h in H3) + " | |")
M.append("| first-row T_i deficit | " + " | ".join(iv(S[h]["first_row_ti_deficit"]) for h in H3) + " | |")
M.append("| share of first-row deficit proportional to BES | " + " | ".join(iv(S[h]["share_of_first_row_deficit_proportional_to_bes"]) for h in H3) + " | reported |")
M.append("| a (per unit ECEI deficit) | " + " | ".join(iv(S[h]["a"]) for h in H3) + " | reported |")
M.append("| b (per unit BES deficit) | " + " | ".join(iv(S[h]["b"]) for h in H3) + " | reported (an upper bound) |")
for q in ("ti", "ecei", "bes"):
    M.append(f"| shape index {q} (rows 4-6 / row 1) | " + " | ".join(iv(S[h]["shape_index"][q]) for h in H3) + " | reported |")
M.append("| control: max deviation from 1, T_i / ECEI / BES | " + " | ".join(
    "{ti:.4f} / {ecei:.4f} / {bes:.4f}".format(**S[h]["control_max_abs_deviation"]) for h in H3) + " | 0.02 / 0.015 / 0.015 |")
M += ["", "### Event-median profiles, all discharges (ratio to the median of the six rows before)", "",
      "| row | T_i | ECEI level | BES level | control T_i | control ECEI | control BES |", "|---|---|---|---|---|---|---|"]
p, c, pc_ = S["all"]["profiles"], S["all"]["control_profiles"], S["all"]["profiles_ci95"]
for i, k in enumerate(K):
    M.append(f"| +{k} | {p['ti'][i]:.3f} [{pc_['ti'][i][0]:.3f}, {pc_['ti'][i][1]:.3f}] | {p['ecei'][i]:.3f} "
             f"[{pc_['ecei'][i][0]:.3f}, {pc_['ecei'][i][1]:.3f}] | {p['bes'][i]:.3f} [{pc_['bes'][i][0]:.3f}, "
             f"{pc_['bes'][i][1]:.3f}] | {c['ti'][i]:.3f} | {c['ecei'][i]:.3f} | {c['bes'][i]:.3f} |")
M += ["", "### Event by event, rows 4 to 6 (reported)", "",
      "| association | all | even half | odd half |", "|---|---|---|---|"]
for nm in ("rank_ti_bes", "partial_ti_bes_given_ecei", "rank_ti_ecei", "partial_ti_ecei_given_bes"):
    M.append(f"| {nm} | " + " | ".join(iv(S[h]["event_by_event_rows_4_to_6"][nm]) for h in H3) + " |")
M += ["", "### Two and three absent frames (reported apart, not statements)", "",
      "| events | n (discharges) | T_i row 1 | ECEI row 1 | BES row 1 | shape T_i / ECEI / BES | unexplained ECEI / BES | b |",
      "|---|---|---|---|---|---|---|---|"]
for nm, o in S_apart.items():
    M.append(f"| {nm} | {o['events']} ({o['discharges']}) | {o['profiles']['ti'][0]:.3f} | {o['profiles']['ecei'][0]:.3f} | "
             f"{o['profiles']['bes'][0]:.3f} | {o['shape_index']['ti']['point']:.2f} / {o['shape_index']['ecei']['point']:.2f} / "
             f"{o['shape_index']['bes']['point']:.2f} | {iv(o['unexplained_ecei'])} / {iv(o['unexplained_bes'])} | {iv(o['b'])} |")
M += ["", "## Part L, lag on quiet stretches (30 ms blocks; + lag = X leads T_i)", "",
      "Median over discharges of the per-discharge rank correlation of first differences of block means.", "",
      "| X | lag -30 ms | lag 0 | lag +30 ms | lag +60 ms | ratio +30/0 (all) | ratio even half | ratio odd half |",
      "|---|---|---|---|---|---|---|---|"]
names = {"c": "BES common mode", "g": "BES radial difference", "e": "ECEI level",
         "k1": "prediction, heating route (registered kernel, part S)", "k0": "prediction, kernel frozen in uc_lag_quiet"}
for q in ("c", "g", "e", "k1", "k0"):
    o = L["all"]["quiet"][q]
    cells = []
    for l in (-1, 0, 1, 2):
        d = o["lags"][str(l)]
        cells.append(f"{d['median']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    rr = [L[h]["quiet"][q]["ratio_lag_plus30_over_lag0"] for h in H3]
    M.append(f"| {names[q]} | " + " | ".join(cells) + " | " + " | ".join(
        f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in rr) + " |")
M += ["", "Nulls (circular shift), all discharges, lag 0: " + ", ".join(
    f"{names[q]} {L['all']['quiet'][q]['lags']['0']['null_median']:+.3f}" for q in ("c", "g", "e")) + ".", "",
      "### Positive control: windows that contain a one-frame blip (BES common mode)", "",
      "45 positions on the 10 ms grid, the absent frame at position 15 + shift, NaN there. Registered reading: shift 0 "
      "(the absent frame opens a block). Shifts 1 and 2 are reported.", "",
      "| shift | half | discharges | lag 0 | lag +30 ms | ratio +30/0 |", "|---|---|---|---|---|---|"]
for s_ in (0, 1, 2):
    for h in H3:
        o = L[h]["positive_control"][f"shift{s_}"]
        d0, d1 = o["lags"]["0"], o["lags"]["1"]
        x = o["ratio_lag_plus30_over_lag0"]
        M.append(f"| {s_} | {h} | {d0['n']} | {d0['median']:+.3f} [{d0['ci95'][0]:+.3f}, {d0['ci95'][1]:+.3f}] | "
                 f"{d1['median']:+.3f} [{d1['ci95'][0]:+.3f}, {d1['ci95'][1]:+.3f}] | "
                 f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] |")
M += ["", "### Partial association at lag 0 (reported)", "", "| | all | even half | odd half |", "|---|---|---|---|"]
for nm in ("ti_bes_given_ecei", "ti_ecei_given_bes"):
    M.append(f"| {nm} | " + " | ".join(
        f"{L[h]['partial_lag0'][nm]['median']:+.3f} [{L[h]['partial_lag0'][nm]['ci95'][0]:+.3f}, "
        f"{L[h]['partial_lag0'][nm]['ci95'][1]:+.3f}] (n {L[h]['partial_lag0'][nm]['n']})" for h in H3) + " |")
be = bound_ev
M += ["", "## S2 bound in eV on quiet rows (reported)", "",
      f"Upper end of b ({be['b_upper']:.3f}) x per-discharge s.d. of the 30 ms changes of the BES common mode "
      f"(median {be['sd_30ms_change_bes_common_median']:.4f}) x per-discharge median T_i (median {be['ti_level_median_eV']:.0f} eV), "
      f"{be['discharges']} discharges: median {be['bound_eV_median']:.2f} eV, quartiles "
      f"{be['bound_eV_quartiles'][0]:.2f} to {be['bound_eV_quartiles'][1]:.2f} eV, 90th percentile {be['bound_eV_p90']:.2f} eV "
      "(a bound on the r.m.s. size of a signal-level bias per 30 ms change; not a statement).", "",
      "## Raw-stream add-on, twelve discharges (not a statement)", "",
      f"Blips matched to absent frames: {RAW_SUMM['blips']} ({RAW_SUMM['one_absent_frame']} with one absent frame). "
      "Duration of the interruption from the raw BES (0.1 ms level below half of its 200 ms running median, at least 3 ms).", "",
      "| association (one absent frame) | rank correlation | n |", "|---|---|---|"]
for k_, v_ in RAW_SUMM.items():
    if isinstance(v_, dict):
        M.append(f"| {k_} | {v_['rho']:+.3f} | {v_['n']} |")
M += ["", "| shot | absent frames | duration (ms) | beam return after last stamp (ms) | T_i row 1 | ECEI row 1 | BES row 1 |",
      "|---|---|---|---|---|---|---|"]
for r_ in RAW_ROWS:
    tir = "-" if r_["ti_ratio_row1"] is None or not np.isfinite(r_["ti_ratio_row1"]) else f"{r_['ti_ratio_row1']:.3f}"
    M.append(f"| {r_['shot']} | {r_['absent_frames']} | {r_['duration_ms']:.1f} | {r_['end_minus_last_stamp_ms']:+.1f} | "
             f"{tir} | {r_['ecei_ratio_row1']:.3f} | {r_['bes_ratio_row1']:.3f} |")
M += ["", "## Interpretations made in running the registration", "",
      "1. Kernel of the heating route: the registration names the event-average T_i deficit profile of part S; "
      "`uc_lag_quiet.py` hard-codes the exploratory profile of all events (one to three absent frames). The verdict of L1 "
      "uses the profile of the registered set (one absent frame, all discharges, event medians, rows 1-10, normalized to sum 1); "
      "the frozen kernel is reported beside it (`L1_holds_with_frozen_kernel` in the json).",
      "2. Positive control windows: the registration does not fix the window; it was fixed in the runner before any "
      "result was read: one-frame blips, 15 rows before the absent frame and 29 after, all 10 ms contiguous, the absent "
      "position as NaN, absent frame opening a block (shift 0). Shifts 1 and 2 are reported.",
      "3. L3 'above zero' and 'above 0.5' are read as the lower ends of the intervals (as for the other statements).",
      "4. S2: the upper end of the interval of b x first-row BES deficit is compared with one third of the point "
      "first-row T_i deficit of the same set (as in the expectation '0.046 against 0.061'); the ratio with its interval is reported.",
      "5. S3: point event-median control profiles, rows 1 to 10; controls are not split by number of absent frames.",
      "6. Halves for part S use the file rank of each event's discharge; the S2 eV bound uses b from all discharges.",
      "", f"Run time {res['runtime_s']:.0f} s."]
OUT_MD.write_text("\n".join(M) + "\n", encoding="utf-8")
log(f"wrote {OUT_JSON.name}, {OUT_MD.name}, {FIG.name}.png/.pdf")
