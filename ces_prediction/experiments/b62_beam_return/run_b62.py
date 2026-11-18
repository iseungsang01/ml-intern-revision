"""B.62 (PREREGISTRATION_B62.md with addendum 1): what happens at the return of the beam.
Part E: raw ECEI and BES levels in 1 ms bins through the interruptions of one absent frame (twelve discharges).
Part X: first-frame excess of the T_i deficit over the shape shared with ECEI (all files, B.58 events).
Part V: fresh rotation values after the same events.
CPU only, no model, real data.  Event and ratio code: b58_blip_calibration/uc_blip_shape.py (imported unchanged);
raw blip detection: the code of b58_blip_calibration/run_b58.py::raw_addon (B.58), repeated here because importing
run_b58 would rerun B.58 and rewrite its outputs.

Run from the repo root:  py ces_prediction/experiments/b62_beam_return/run_b62.py
Outputs: docs/paper/b62_beam_return.{json,md,png,pdf}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
B58 = HERE.parent / "b58_blip_calibration"
sys.path.insert(0, str(B58))
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

T0 = _time.time()
NB = 2000
SEED = 20261120
REG = "ces_prediction/experiments/PREREGISTRATION_B62.md"
PAPER = REPO / "docs" / "paper"
OUT_JSON = PAPER / "b62_beam_return.json"
OUT_MD = PAPER / "b62_beam_return.md"
FIG = PAPER / "b62_beam_return"
PRE_MS, POST_MS = 30, 60          # bins from 30 ms before the beam goes off to 60 ms after it returns
REF_MS = 20                       # reference: mean of the 20 ms before the beam goes off
WINDOWS = {"inside_last_3ms": ("ret", -3, 0), "return_0_3ms": ("ret", 0, 3),
           "return_10_13ms": ("ret", 10, 13), "return_20_23ms": ("ret", 20, 23)}
CTRL_PER_SHOT = 10
warnings.simplefilter("ignore", category=RuntimeWarning)


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


log("importing b58_blip_calibration/uc_blip_shape.py (runs it)")
with contextlib.redirect_stdout(io.StringIO()):
    import uc_blip_shape as SH  # noqa: E402


# ================================================================================== part E (raw streams)
def level_1ms(x, fs, t0):
    w = int(round(fs * 1e-3))
    n = len(x) // w
    return x[:n * w].reshape(n, w).mean(1), t0 + np.arange(n) * 1e-3   # bin start times


def raw_events(shot):
    """B.58 raw_addon: interruptions in the raw BES (0.1 ms level below half of its 200 ms running median for at
    least 3 ms), matched to the absent frames of the 10 ms file, with the B.58 conditions (six contiguous rows)."""
    df, *_ = F.load(F.DATA / f"s{shot}.csv")
    t = df["time"].to_numpy(float)
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
    out = []
    for j, stamps in F.absent_groups(t):
        if not (t0 + 0.05 < stamps[0] and stamps[-1] < tb[-1] - 0.05):
            continue
        hit = [i for i, (a, b) in enumerate(events) if a < stamps[-1] + 0.010 and b > stamps[0] - 0.005]
        if not hit or j < 5:
            continue
        pre = slice(j - 5, j + 1)
        if not np.all(np.abs(np.diff(t[pre]) - 0.010) < 1e-4) or j + 1 >= len(t):
            continue
        out.append({"shot": shot, "absent_frames": len(stamps), "off_s": events[hit[0]][0],
                    "return_s": events[hit[-1]][1]})
    all_int = events
    return out, all_int, t


def window_vals(L, tbin, off, ret):
    """ratios to the mean of the 20 ms before off: bins and the four windows.
    L = (cumulative sum of the per-sample level, fs, t0) -- windows and 1 ms bins are anchored at the event times
    with sample resolution (correction after the first run, which used a global 1 ms grid; see the md)."""
    cs, fs, t0 = L

    def mean_between(a, b):
        i0 = int(round((a - t0) * fs))
        i1 = int(round((b - t0) * fs))
        if i0 < 0 or i1 >= len(cs) or i1 <= i0:
            return np.nan
        return float((cs[i1] - cs[i0]) / (i1 - i0))
    ref = mean_between(off - REF_MS * 1e-3, off)
    refbins = np.array([mean_between(off + k * 1e-3, off + (k + 1) * 1e-3) for k in range(-REF_MS, 0)])
    r = {"ref": ref, "ref_sd": float(np.nanstd(refbins))}
    for k, (base, a, b) in WINDOWS.items():
        r[k] = mean_between(ret + a * 1e-3, ret + b * 1e-3) / ref
    # profiles: off-aligned bins -30..-1 and inside, return-aligned bins 0..59; also return-aligned -45..59
    r["ret_aligned"] = np.array([mean_between(ret + k * 1e-3, ret + (k + 1) * 1e-3) / ref for k in range(-45, POST_MS)])
    r["off_aligned"] = np.array([mean_between(off + k * 1e-3, off + (k + 1) * 1e-3) / ref for k in range(-PRE_MS, 12)])
    return r


def part_e():
    ev, ct = [], []
    rng = np.random.default_rng(SEED)
    for shot in F.RAW_SHOTS:
        evs, ints, t10 = raw_events(shot)
        lv = {}
        for diag in ("BES", "ECEI"):
            x, fs, t0, _ch = F.raw_npz(shot, diag)
            lvl = x.mean(1).astype(np.float64)
            del x
            lv[diag] = (np.concatenate(([0.0], np.cumsum(lvl))), fs, t0)
            lv[diag + "_t"] = (t0 + 0.0, t0 + len(lvl) / fs)
        for e in evs:
            for diag in ("BES", "ECEI"):
                e[diag] = window_vals(lv[diag], None, e["off_s"], e["return_s"])
            ev.append(e)
        # control pseudo-times: no raw interruption and no absent frame within 200 ms, inside the raw window
        lo, hi = lv["BES_t"][0] + 0.1, lv["BES_t"][1] - 0.1
        dtt = np.diff(t10)
        absent_t = t10[:-1][(dtt > 0.015) & (dtt < 0.5)]
        cand = rng.uniform(lo, hi, size=400)
        ok = [c for c in cand if all(abs(c - a) > 0.2 and abs(c - b) > 0.2 for a, b in ints)
              and np.all(np.abs(absent_t - c) > 0.2)]
        for c in ok[:CTRL_PER_SHOT]:
            e = {"shot": shot, "off_s": c, "return_s": c + 0.0105}
            for diag in ("BES", "ECEI"):
                e[diag] = window_vals(lv[diag], None, c, c + 0.0105)
            ct.append(e)
        log(f"  raw {shot}: {len(evs)} matched interruptions, {min(len(ok), CTRL_PER_SHOT)} control times")
    return ev, ct


def ecei_ok(e):
    return e["ECEI"]["ref"] > 0 and e["ECEI"]["ref"] >= 10 * e["ECEI"]["ref_sd"]


def boot_e(events, seed):
    """event medians of the four windows and of the differences, with NB resamples of discharges."""
    shots = np.array([e["shot"] for e in events])
    us = np.unique(shots)
    pos = {u: np.flatnonzero(shots == u) for u in us}
    rng = np.random.default_rng(seed)

    def summ(idx, diag):
        V = {k: np.array([events[i][diag][k] for i in idx]) for k in WINDOWS}
        o = {k: float(np.nanmedian(V[k])) for k in WINDOWS}
        o["E2_diff_of_medians"] = o["return_0_3ms"] - o["inside_last_3ms"]
        o["E3_diff_of_medians"] = o["return_10_13ms"] - o["return_0_3ms"]
        o["E2_median_paired"] = float(np.nanmedian(V["return_0_3ms"] - V["inside_last_3ms"]))
        o["E3_median_paired"] = float(np.nanmedian(V["return_10_13ms"] - V["return_0_3ms"]))
        return o

    out = {}
    for diag in ("ECEI", "BES"):
        idx_all = np.arange(len(events))
        full = summ(idx_all, diag)
        bs = []
        for _ in range(NB):
            idx = np.concatenate([pos[u] for u in rng.choice(us, size=len(us))])
            bs.append(summ(idx, diag))
        out[diag] = {k: {"point": full[k], "ci95": ci([b[k] for b in bs])} for k in full}
        prof = np.array([e[diag]["ret_aligned"] for e in events])
        proff = np.array([e[diag]["off_aligned"] for e in events])
        out[diag]["profile_return_aligned_ms"] = list(range(-45, POST_MS))
        out[diag]["profile_return_aligned_median"] = [float(v) for v in np.nanmedian(prof, 0)]
        out[diag]["profile_off_aligned_ms"] = list(range(-PRE_MS, 12))
        out[diag]["profile_off_aligned_median"] = [float(v) for v in np.nanmedian(proff, 0)]
        pb = []
        for _ in range(400):
            idx = np.concatenate([pos[u] for u in rng.choice(us, size=len(us))])
            pb.append(np.nanmedian(prof[idx], 0))
        pb = np.array(pb)
        out[diag]["profile_return_aligned_ci95"] = [ci(pb[:, i]) for i in range(pb.shape[1])]
    out["events"] = len(events)
    out["discharges"] = int(len(us))
    return out


log("part E: raw streams")
EV_ALL, CT_ALL = part_e()
one = [e for e in EV_ALL if e["absent_frames"] == 1]
off_ms = np.array([(e["return_s"] - e["off_s"]) * 1e3 for e in one])
checkE = {"interruptions_one_frame": len(one), "expected": 41,
          "median_off_ms": float(np.median(off_ms)), "expected_off_ms": 10.5}
checkE["ok"] = (len(one) == 41) and abs(checkE["median_off_ms"] - 10.5) < 0.05
log(f"check E: {len(one)} interruptions, median off {checkE['median_off_ms']:.2f} ms -> {checkE['ok']}")
if not checkE["ok"]:
    raise SystemExit("FATAL: part E check failed; stopping before any statement")
one_ecei = [e for e in one if ecei_ok(e)]
excluded = [{"shot": e["shot"], "off_s": e["off_s"], "ecei_ref": e["ECEI"]["ref"], "ecei_ref_sd": e["ECEI"]["ref_sd"]}
            for e in one if not ecei_ok(e)]
ct_ecei = [e for e in CT_ALL if ecei_ok(e)]
log(f"ECEI readout keeps {len(one_ecei)} of {len(one)} interruptions; excluded {len(excluded)}")
E_ecei = boot_e(one_ecei, SEED + 1)
E_bes = boot_e(one, SEED + 2)
C_ecei = boot_e(ct_ecei, SEED + 3)
C_bes = boot_e(CT_ALL, SEED + 4)
ctrl_dev = {"ECEI": max(abs(C_ecei["ECEI"][k]["point"] - 1) for k in WINDOWS),
            "BES": max(abs(C_bes["BES"][k]["point"] - 1) for k in WINDOWS)}
ctrl_ok = ctrl_dev["ECEI"] <= 0.02 and ctrl_dev["BES"] <= 0.02
log(f"check E control within 0.02: {ctrl_ok} {ctrl_dev}")
multi = [e for e in EV_ALL if e["absent_frames"] > 1]
multi_rows = [{"shot": e["shot"], "absent_frames": e["absent_frames"],
               "off_ms": (e["return_s"] - e["off_s"]) * 1e3,
               **{f"ECEI_{k}": (e["ECEI"][k] if ecei_ok(e) else None) for k in WINDOWS},
               **{f"BES_{k}": e["BES"][k] for k in WINDOWS}} for e in multi]
E1 = E_ecei["ECEI"]["inside_last_3ms"]["ci95"][1] < 1
E2 = E_ecei["ECEI"]["E2_diff_of_medians"]["ci95"][1] < 0
E3 = E_ecei["ECEI"]["E3_diff_of_medians"]["ci95"][1] < 0


# ================================================================================== parts X and V (10 ms files)
log("parts X and V")
files = sorted(F.DATA.glob("s*.csv"))
K = list(range(1, 11))
vt_ev, vt_f, vt_n, ti_chk = [], [], [], []
for fi, f in enumerate(files):
    df, *_ = F.load(f)
    t = df["time"].to_numpy(float)
    if len(t) < 30:
        continue
    ti = df["CES_TI"].to_numpy(float).copy()
    ti[ti > 3000] = np.nan
    vt = df["CES_VT"].to_numpy(float)
    fresh = np.concatenate(([True], vt[1:] != vt[:-1])) & np.isfinite(vt)
    for j, stamps in F.absent_groups(t):
        pre = slice(j - 5, j + 1)
        if j < 5 or not np.all(np.abs(np.diff(t[pre]) - 0.010) < 1e-4):      # B.58 event rule (uc_blip_shape)
            continue
        tip = np.nanmedian(ti[pre]) if np.isfinite(ti[pre]).sum() >= 3 else np.nan
        vf = vt[pre][fresh[pre]]
        vp = float(np.median(vf)) if len(vf) else np.nan
        rv, rt = [], []
        for k in K:
            i = j + k
            ok = i < len(t) and abs(t[i] - (t[j + 1] + 0.010 * (k - 1))) < 1e-3
            rv.append(vt[i] / vp if ok and fresh[i] and np.isfinite(vp) and abs(vp) > 5 else np.nan)
            rt.append(ti[i] / tip if ok and np.isfinite(tip) and tip > 0 else np.nan)
        vt_ev.append(rv)
        ti_chk.append(rt)
        vt_f.append(fi)
        vt_n.append(len(stamps))
vt_ev, ti_chk = np.array(vt_ev, float), np.array(ti_chk, float)
vt_f, vt_n = np.array(vt_f), np.array(vt_n)
evf, evn = np.asarray(SH.evf), np.asarray(SH.evn)
same_events = (len(vt_f) == len(evf)) and np.array_equal(vt_f, evf) and np.array_equal(vt_n, evn) \
    and np.allclose(ti_chk, SH.ev["ti"], equal_nan=True)
b58 = json.loads((PAPER / "b58_blip_calibration.json").read_text(encoding="utf-8"))
first = SH.stat(SH.ev, evf, np.flatnonzero(evn == 1))
checkX = {q: [float(first[q][0]), float(b58["S"]["all"]["profiles"][q][0])] for q in ("ti", "ecei", "bes")}
checkX_ok = all(abs(a - b) < 1e-12 for a, b in checkX.values()) and bool(same_events)
log(f"check X (B.58 first-row ratios reproduced, same event set for V): {checkX_ok}")
if not checkX_ok:
    raise SystemExit("FATAL: part X/V check failed; stopping before any statement")


def fit_x(m):
    y = 1 - np.asarray(m["ti"][:8])
    Xm = np.c_[1 - np.asarray(m["ecei"][:8]), (np.arange(8) == 0).astype(float)]
    c, d = np.linalg.lstsq(Xm, y, rcond=None)[0]
    return float(c), float(d), float(d / y[0])


def part_x(n, seed):
    sel = np.flatnonzero(evn == n)
    full = SH.stat(SH.ev, evf, sel)
    uf = np.unique(evf[sel])
    pos = {u: sel[evf[sel] == u] for u in uf}
    rng = np.random.default_rng(seed)
    bs = np.array([fit_x(SH.stat(SH.ev, evf, np.concatenate([pos[u] for u in rng.choice(uf, size=len(uf))])))
                   for _ in range(NB)])
    c, d, s = fit_x(full)
    return {"events": int(len(sel)), "discharges": int(len(uf)),
            "c": {"point": c, "ci95": ci(bs[:, 0])}, "d": {"point": d, "ci95": ci(bs[:, 1])},
            "d_share_of_first_row_deficit": {"point": s, "ci95": ci(bs[:, 2])},
            "first_row_ti_deficit": float(1 - full["ti"][0])}


def part_v(n, seed):
    sel = np.flatnonzero(vt_n == n)
    uf = np.unique(vt_f[sel])
    pos = {u: sel[vt_f[sel] == u] for u in uf}
    rng = np.random.default_rng(seed)

    def prof(idx):
        return np.array([np.nanmedian(vt_ev[idx, k]) for k in range(len(K))])

    def shp(p):
        dd = 1 - p
        return dd[3:6].mean() / dd[0]
    full = prof(sel)
    bs = [prof(np.concatenate([pos[u] for u in rng.choice(uf, size=len(uf))])) for _ in range(NB)]
    bs = np.array(bs)
    return {"events": int(len(sel)), "discharges": int(len(uf)),
            "fresh_values_per_row": [int(np.isfinite(vt_ev[sel, k]).sum()) for k in range(len(K))],
            "profile": [float(v) for v in full], "profile_ci95": [ci(bs[:, k]) for k in range(len(K))],
            "shape_index": {"point": float(shp(full)), "ci95": ci([shp(p) for p in bs])}}


X = {str(n): part_x(n, SEED + 10 + n) for n in (1, 2, 3)}
V = {str(n): part_v(n, SEED + 20 + n) for n in (1, 2, 3)}
X1 = X["1"]["d"]["ci95"][0] > 0
V1 = (V["1"]["profile_ci95"][0][1] < 1) or (V["1"]["profile_ci95"][0][0] > 1)
log(f"E1 {E1} E2 {E2} E3 {E3} X1 {X1} V1 {V1}")


# ================================================================================== figure
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = np.array(E_ecei["ECEI"]["profile_return_aligned_ms"]) + 0.5
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
    med_off = checkE["median_off_ms"]
    ax.axvspan(-med_off, 0, color="0.88", lw=0)
    ax.text(-med_off / 2, 0.08, "beam off\n(median)", ha="center", fontsize=7, color="0.35")
    for diag, E_, col, lab in (("BES", E_bes, "#7f8c8d", "BES level"), ("ECEI", E_ecei, "#c0392b", "ECEI level")):
        p = np.array(E_[diag]["profile_return_aligned_median"])
        c_ = np.array(E_[diag]["profile_return_aligned_ci95"], float)
        ax.fill_between(ms, c_[:, 0], c_[:, 1], color=col, alpha=0.15, lw=0)
        ax.plot(ms, p, color=col, lw=1.3)
        ax.text(ms[-1] + 1, p[-1], lab, color=col, fontsize=8, va="center")
    ax.axhline(1.0, color="0.8", lw=0.6, zorder=0)
    ax.set_xlabel("time from the return of the beam (ms), 1 ms bins")
    ax.set_ylabel("ratio to the 20 ms before beam-off")
    ax.set_xlim(ms[0] - 1, ms[-1] + 14)
    ax.set_ylim(0, 1.15)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".png"))
    fig.savefig(FIG.with_suffix(".pdf"))
    plt.close(fig)


figure()

# ================================================================================== outputs
res = {"registration": REG + " (with addendum 1)",
       "uc_blip_shape_sha256": hashlib.sha256((B58 / "uc_blip_shape.py").read_bytes()).hexdigest(),
       "bootstrap_resamples": NB, "seed": SEED,
       "checks": {"E": checkE, "E_control_max_deviation": ctrl_dev, "E_control_ok": bool(ctrl_ok),
                  "X_first_row_ratios_b61_b58": checkX, "X_V_same_event_set": bool(same_events), "X_ok": bool(checkX_ok)},
       "E": {"ecei_readout_interruptions": len(one_ecei),
             "ecei_readout_discharges": int(len({e['shot'] for e in one_ecei})),
             "bes_readout_interruptions": len(one), "bes_readout_discharges": int(len({e['shot'] for e in one})),
             "excluded_from_ecei": excluded, "ECEI": E_ecei["ECEI"], "BES": E_bes["BES"],
             "control": {"ECEI": C_ecei["ECEI"], "BES": C_bes["BES"], "times_ecei": len(ct_ecei), "times_bes": len(CT_ALL)},
             "two_and_three_absent_frames": multi_rows},
       "X": X, "V": V,
       "verdicts": {"E1": bool(E1), "E2": bool(E2), "E3": bool(E3), "X1": bool(X1), "V1": bool(V1)},
       "runtime_s": _time.time() - T0}
OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def iv(d, f="{:.3f}"):
    lo, hi = d["ci95"]
    return f"{f.format(d['point'])} [{f.format(lo)}, {f.format(hi)}]"


def hv(x):
    return "**holds**" if x else "**does not hold**"


Ee, Eb = E_ecei["ECEI"], E_bes["BES"]
w_e2 = Ee["E2_diff_of_medians"]["ci95"][1] - Ee["E2_diff_of_medians"]["ci95"][0]
w_e1 = Ee["inside_last_3ms"]["ci95"][1] - Ee["inside_last_3ms"]["ci95"][0]
w_e3 = Ee["E3_diff_of_medians"]["ci95"][1] - Ee["E3_diff_of_medians"]["ci95"][0]
M = ["# B.62 What happens at the return of the beam", "",
     f"Registration: `{REG}` with addendum 1. Runner: `ces_prediction/experiments/b62_beam_return/run_b62.py`. "
     "CPU only, no model, real data. Intervals: 95 %, 2,000 resamples of discharges; window values are event medians.", "",
     "## Checks (before any statement)", "",
     f"- Part E: {checkE['interruptions_one_frame']} interruptions of one absent frame (expected 41), median off-time "
     f"{checkE['median_off_ms']:.2f} ms (expected 10.5): {'passed' if checkE['ok'] else 'FAILED'}.",
     f"- Part E control ({res['E']['control']['times_ecei']} pseudo-times for ECEI, {res['E']['control']['times_bes']} for BES; "
     f"no raw interruption and no absent frame within 200 ms, a 10.5 ms pseudo-interruption): largest deviation of a window "
     f"median from 1: ECEI {ctrl_dev['ECEI']:.4f}, BES {ctrl_dev['BES']:.4f} (limit 0.02): {'passed' if ctrl_ok else 'FAILED'}.",
     "- Part X: first-row ratios of B.58 reproduced exactly (T_i, ECEI, BES " + ", ".join(
         f"{v[0]:.3f}" for v in checkX.values()) + f"); the rotation readout uses the same event set: {checkX_ok}.", "",
     "## Addendum 1: interruptions kept for the ECEI readout", "",
     f"ECEI readout: **{len(one_ecei)} interruptions in {res['E']['ecei_readout_discharges']} discharges** "
     f"(BES readout: {len(one)} in {res['E']['bes_readout_discharges']}). Excluded (mean of the 20 ms before not positive "
     "or below ten times the s.d. of its 1 ms bins):", "",
     "| shot | beam off at (s) | ECEI mean before | s.d. of 1 ms bins |", "|---|---|---|---|"]
for x_ in excluded:
    M.append(f"| {x_['shot']} | {x_['off_s']:.4f} | {x_['ecei_ref']:+.5f} | {x_['ecei_ref_sd']:.5f} |")
M += ["", f"With this many events the intervals are wide: E1 spans {w_e1:.3f} in the ratio, E2 {w_e2:.3f} and E3 "
      f"{w_e3:.3f} in the difference; a step of a few hundredths, the size expected, is near the resolution of this readout.", "",
      "## Statements", "",
      "| statement | finding | verdict | expected |", "|---|---|---|---|",
      f"| E1 ECEI inside the interruption (last 3 ms before return) below 1 | {iv(Ee['inside_last_3ms'])} | {hv(E1)} | 0.97 to 1.01, fails |",
      f"| E2 ECEI step at the return: (0-3 ms after) minus (last 3 ms before) below 0 | {iv(Ee['E2_diff_of_medians'], '{:+.3f}')} | {hv(E2)} | -0.03 to -0.08, holds |",
      f"| E3 ECEI deepens: (10-13 ms after) minus (0-3 ms after) below 0 | {iv(Ee['E3_diff_of_medians'], '{:+.3f}')} | {hv(E3)} | -0.02 to +0.02, fails |",
      f"| X1 first-frame excess d above 0 | d = {iv(X['1']['d'], '{:+.3f}')} | {hv(X1)} | 0.01 to 0.04, holds or fails narrowly |",
      f"| V1 fresh rotation, first row, ratio excludes 1 | {iv({'point': V['1']['profile'][0], 'ci95': V['1']['profile_ci95'][0]})} | {hv(V1)} | 0.96 to 0.99, holds |",
      "", "Differences are differences of event medians within a draw (the verdict); the median of the paired per-event "
      "differences is reported beside them.", ""]
_fr = REPO / "data" / ".b62" / "first_run_globalbins.json"
if _fr.exists():
    fr = json.loads(_fr.read_text(encoding="utf-8"))
    fe = fr["E"]["ECEI"]
    M += ["### First run (implementation error, kept for the record)", "",
          "The first run of this runner, read before the correction, averaged whole bins of a global 1 ms grid of the raw "
          "stream (bins starting at the raw start time) that fell inside each window, so a 3 ms window held two or three "
          "bins and was displaced by up to 1 ms from the event times; its 1 ms profiles kept only the events whose times "
          "fell on that grid. The run above anchors every window and every 1 ms bin at the beam-off and beam-return "
          "times of each event with sample resolution, which is what the registration describes. First-run values "
          f"(ECEI, {fr['E']['ecei_readout_interruptions']} interruptions): E1 {iv(fe['inside_last_3ms'])}, "
          f"E2 {iv(fe['E2_diff_of_medians'], '{:+.3f}')}, E3 {iv(fe['E3_diff_of_medians'], '{:+.3f}')}; verdicts "
          + ", ".join(f"{k} {'holds' if v else 'does not hold'}" for k, v in fr["verdicts"].items()) + ".", ""]
M += ["## Part E, the four windows (event medians, ratio to the 20 ms before the beam goes off)", "",
      "| window | ECEI (kept interruptions) | BES (all) | ECEI control | BES control |", "|---|---|---|---|---|"]
for k in WINDOWS:
    M.append(f"| {k} | {iv(Ee[k])} | {iv(Eb[k])} | {C_ecei['ECEI'][k]['point']:.3f} | {C_bes['BES'][k]['point']:.3f} |")
for k in ("E2_diff_of_medians", "E2_median_paired", "E3_diff_of_medians", "E3_median_paired"):
    M.append(f"| {k} | {iv(Ee[k], '{:+.3f}')} | {iv(Eb[k], '{:+.3f}')} | | |")
M += ["", "### 1 ms profiles aligned to the return of the beam (event medians; figure `docs/paper/b62_beam_return.png`)", "",
      "| ms from return | ECEI | BES |", "|---|---|---|"]
msr = Ee["profile_return_aligned_ms"]
for i, k in enumerate(msr):
    if k in (-45, -35, -25, -20, -15, -12, -10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10, 13, 15, 20, 23, 30, 40, 50, 59):
        M.append(f"| {k} | {Ee['profile_return_aligned_median'][i]:.3f} | {Eb['profile_return_aligned_median'][i]:.3f} |")
M += ["", "### 1 ms profiles aligned to the beam going off", "", "| ms from off | ECEI | BES |", "|---|---|---|"]
mso = Ee["profile_off_aligned_ms"]
for i, k in enumerate(mso):
    if k in (-30, -20, -10, -5, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        M.append(f"| {k} | {Ee['profile_off_aligned_median'][i]:.3f} | {Eb['profile_off_aligned_median'][i]:.3f} |")
M += ["", "### Interruptions of two and three absent frames (raw; reported, not statements)", "",
      "| shot | absent frames | off (ms) | ECEI inside / 0-3 / 10-13 / 20-23 | BES inside / 0-3 / 10-13 / 20-23 |",
      "|---|---|---|---|---|"]
for r_ in multi_rows:
    ee = " / ".join("-" if r_[f"ECEI_{k}"] is None else f"{r_[f'ECEI_{k}']:.3f}" for k in WINDOWS)
    bb = " / ".join(f"{r_[f'BES_{k}']:.3f}" for k in WINDOWS)
    M.append(f"| {r_['shot']} | {r_['absent_frames']} | {r_['off_ms']:.1f} | {ee} | {bb} |")
M += ["", "## Part X, first-frame excess (event-median T_i deficit, rows 1-8 = c x ECEI deficit + d x [row 1])", "",
      "| absent frames | events (discharges) | c | d | d / first-row deficit | first-row deficit |", "|---|---|---|---|---|---|"]
for n in ("1", "2", "3"):
    x_ = X[n]
    M.append(f"| {n} | {x_['events']} ({x_['discharges']}) | {iv(x_['c'])} | {iv(x_['d'], '{:+.3f}')} | "
             f"{iv(x_['d_share_of_first_row_deficit'], '{:+.3f}')} | {x_['first_row_ti_deficit']:.3f} |")
M += ["", "## Part V, fresh rotation values (ratio to the median of the fresh values of the six rows before)", "",
      "| row | 1 absent frame | fresh values | 2 absent frames | 3 absent frames |", "|---|---|---|---|---|"]
for k in range(10):
    M.append(f"| +{k + 1} | {V['1']['profile'][k]:.3f} [{V['1']['profile_ci95'][k][0]:.3f}, {V['1']['profile_ci95'][k][1]:.3f}] | "
             f"{V['1']['fresh_values_per_row'][k]} | {V['2']['profile'][k]:.3f} | {V['3']['profile'][k]:.3f} |")
M += ["", "Events (discharges): " + ", ".join(f"{n} absent frame(s) {V[n]['events']} ({V[n]['discharges']})" for n in ("1", "2", "3"))
      + ". Shape index (rows 4-6 over row 1): " + ", ".join(f"{n}: {iv(V[n]['shape_index'])}" for n in ("1", "2", "3")) + ".", "",
      "## Interpretations made in running the registration", "",
      "1. Window values are event medians (as in B.58); E2 and E3 are differences of the event medians within a draw; the median of "
      "paired per-event differences is reported beside them.",
      "2. Window edges and 1 ms bins are anchored at the B.58 beam-off and beam-return times of each interruption (0.1 ms "
      "resolution) and averaged over the raw samples with sample resolution; the reference is the 20 ms before the beam goes off; "
      "the s.d. of addendum 1 is that of the twenty 1 ms bins of that window.",
      "3. The figure aligns to the return of the beam (45 ms before to 60 ms after it) and shades the median off interval; a profile "
      "aligned to the beam going off is tabulated.",
      "4. Control pseudo-times: up to ten per discharge drawn uniformly inside the raw window with no raw interruption and no step "
      "above 15 ms of the 10 ms file within 200 ms; a pseudo-interruption of 10.5 ms; the ECEI rule of addendum 1 applies to them too.",
      "5. The rotation readout repeats the event rule of `uc_blip_shape.py` in this runner (that script does not compute the "
      "rotation) and is checked to give the identical event set and identical T_i ratios; fresh = differs from the previous row "
      "and is finite; the level is the median of the fresh values among the six rows before, used if its magnitude exceeds 5 km/s "
      "(the rule of `analyze_file_rules.py`).",
      "6. V1 'differs from its level before' is read as the interval of the first-row ratio excluding 1.", "",
      f"Run time {res['runtime_s']:.0f} s."]
OUT_MD.write_text("\n".join(M) + "\n", encoding="utf-8")
log("wrote outputs")
