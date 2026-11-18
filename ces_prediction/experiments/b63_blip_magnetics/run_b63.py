"""B.63 (PREREGISTRATION_B63.md): the integrated raw Mirnov signal (MC1T03, MC1T16) through the beam interruptions
of the twelve discharges with raw streams.  CPU only, no model, real data.

Addendum 1 (drift correction with a correction half and a held-out half of up to 40 pseudo-events per discharge).
Order (as instructed): without a flag the runner computes and writes the repeated check (b) (and (a), (d)) only;
with --statements it then computes M1-M4 and the reported quantities.  The first, uncorrected checks are kept in
data/.b63/first_checks.{md,json} and quoted in the md; check (c) is not recomputed (addendum 1).

Event code: the raw interruption detection of B.58 (run_b58.raw_addon) as repeated in b62_beam_return/run_b62.py
(raw_events); it is repeated here verbatim because importing either runner would rerun it and rewrite its outputs.
Raw arrays: data/raw_mus/.cache/s{shot}_{BES,ECEI,MC}.npz (float32 copies of data/raw_mus/s{shot}_*.csv, read by
analyze_file_rules.raw_npz); a spot check against the CSV is written into the output.  Per-event windows are cached
under data/.b63/.

Run from the repo root:
    py ces_prediction/experiments/b63_blip_magnetics/run_b63.py               # checks only
    py ces_prediction/experiments/b63_blip_magnetics/run_b63.py --statements  # checks + statements
Outputs: docs/paper/b63_blip_magnetics.{json,md,png,pdf}
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

warnings.simplefilter("ignore", category=RuntimeWarning)
T0 = _time.time()
STATEMENTS = "--statements" in sys.argv
NB = 2000
SEED = 20261121
REG = "ces_prediction/experiments/PREREGISTRATION_B63.md"
PAPER = REPO / "docs" / "paper"
OUT_JSON = PAPER / "b63_blip_magnetics.json"
OUT_MD = PAPER / "b63_blip_magnetics.md"
FIG = PAPER / "b63_blip_magnetics"
CACHE = REPO / "data" / ".b63"
BASE_MS = 30                      # baseline: mean of the 30 ms before t_off
PRE_MS, POST_MS = 30, 80          # profile from 30 ms before t_off to 80 ms after t_on
WIN = {"inside_last_3ms": (-3, 0), "return_0_3ms": (0, 3), "return_10_13ms": (10, 13),
       "return_20_23ms": (20, 23), "return_40_43ms": (40, 43)}
CTRL_PER_SHOT = 40                # addendum 1: up to 40 per discharge, alternate halves
CTRL_LEN_S = 0.0105               # median off-time
N_EXC = 200
EXC_MIN_SEP_MS = 20               # interpretation: excursions closer than 20 ms are one excursion
EXC_MIN_DIST_INT_S = 0.050


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


# ------------------------------------------------------------------------------------------- events (B.58/B.62)
def raw_events(shot):
    """verbatim from b62_beam_return/run_b62.py::raw_events (B.58 raw_addon rule)."""
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
    return out, events, t


# ------------------------------------------------------------------------------------------- raw helpers
class Raw:
    def __init__(self, shot, diag):
        x, fs, t0, ch = F.raw_npz(shot, diag)
        self.fs, self.t0, self.ch = fs, t0, ch
        self.x = x
        self.t1 = t0 + len(x) / fs

    def idx(self, t):
        return int(round((t - self.t0) * self.fs))

    def covers(self, a, b):
        return a >= self.t0 + 1e-6 and b <= self.t1 - 1e-6


def integrated(mc, t_ref, t_from, t_to):
    """Phi_c for both coils on [t_from, t_to), zero at t_ref, baseline = mean over the 30 ms before t_ref.
    Returns (time of each sample relative to t_ref, Phi (n, 2)) in V*s (units of the coil signal times seconds)."""
    i_ref = mc.idx(t_ref)
    i_b0 = mc.idx(t_ref - BASE_MS * 1e-3)
    base = mc.x[i_b0:i_ref].astype(np.float64).mean(0)
    i0, i1 = mc.idx(t_from), mc.idx(t_to)
    seg = mc.x[i0:i1].astype(np.float64) - base
    dt = 1.0 / mc.fs
    # integral from t_ref: forward for samples >= t_ref, backward (negative) for samples before it
    k = i_ref - i0
    phi = np.empty_like(seg)
    phi[k:] = np.cumsum(seg[k:], 0) * dt
    if k > 0:
        phi[:k] = -np.cumsum(seg[:k][::-1], 0)[::-1] * dt
    tt = (np.arange(i0, i1) - i_ref) / mc.fs
    return tt, phi


def win_mean(tt, y, a_s, b_s):
    m = (tt >= a_s) & (tt < b_s)
    return float(y[m].mean()) if m.any() else np.nan


def event_record(mc, t_off, t_on):
    """C, D, Phi_03, Phi_16 (V*s): window means relative to t_on, 1 ms profile from t_off-30 ms to t_on+80 ms."""
    tt, phi = integrated(mc, t_off, t_off - PRE_MS * 1e-3, t_on + POST_MS * 1e-3)
    tr = tt - (t_on - t_off)                        # time relative to t_on
    C = phi.mean(1)
    D = (phi[:, 0] - phi[:, 1]) / 2.0
    r = {}
    for nm, y in (("C", C), ("D", D), ("P03", phi[:, 0]), ("P16", phi[:, 1])):
        r[nm] = {k: win_mean(tr, y, a * 1e-3, b * 1e-3) for k, (a, b) in WIN.items()}
    # profile in 1 ms bins aligned to t_on: from -(30 + off) to +80 ms; we store bins -45..+79 relative to t_on
    edges = np.arange(-45, POST_MS) * 1e-3
    r["prof_C"] = [win_mean(tr, C, e, e + 1e-3) for e in edges]
    r["prof_D"] = [win_mean(tr, D, e, e + 1e-3) for e in edges]
    r["prof_P03"] = [win_mean(tr, phi[:, 0], e, e + 1e-3) for e in edges]
    r["prof_P16"] = [win_mean(tr, phi[:, 1], e, e + 1e-3) for e in edges]
    return r


def burst_change(mc, te):
    """C change between the 5 ms before and the 5 ms after te (Phi zero at te, baseline the 30 ms before te)."""
    tt, phi = integrated(mc, te, te - 5e-3, te + 5e-3)
    C = phi.mean(1)
    return win_mean(tt, C, 0, 5e-3) - win_mean(tt, C, -5e-3, 0)


# ------------------------------------------------------------------------------------------- collect (addendum 1)
PROF_MS = list(range(-45, POST_MS))


def collect():
    rng = np.random.default_rng(SEED)
    EV, PS, per_shot = [], [], {}
    for shot in F.RAW_SHOTS:
        evs, ints, t10 = raw_events(shot)
        mc = Raw(shot, "MC")
        ec = Raw(shot, "ECEI")
        ecs = np.concatenate(([0.0], np.cumsum(ec.x.mean(1).astype(np.float64))))

        def emean(a, b):
            i0, i1 = ec.idx(a), ec.idx(b)
            return (ecs[i1] - ecs[i0]) / (i1 - i0) if 0 <= i0 < i1 < len(ecs) else np.nan
        cov = 0
        for e in evs:
            a, b = e["off_s"] - (BASE_MS + PRE_MS) * 1e-3, e["return_s"] + POST_MS * 1e-3
            e["mc_covered"] = mc.covers(a, b)
            if e["mc_covered"]:
                cov += 1
                e.update(event_record(mc, e["off_s"], e["return_s"]))
            ref = emean(e["off_s"] - 0.020, e["off_s"])
            bins = np.array([emean(e["off_s"] + k * 1e-3, e["off_s"] + (k + 1) * 1e-3) for k in range(-20, 0)])
            e["ecei_ok"] = bool(ref > 0 and ref >= 10 * np.nanstd(bins))
            e["ecei_fall_10_13"] = float(1 - emean(e["return_s"] + 0.010, e["return_s"] + 0.013) / ref) if e["ecei_ok"] else None
            EV.append(e)
        # pseudo-events: B.62 rule + Mirnov coverage, up to 40, in time order, alternate halves
        lo, hi = mc.t0 + 0.1, mc.t1 - 0.1
        dtt = np.diff(t10)
        absent_t = t10[:-1][(dtt > 0.015) & (dtt < 0.5)]
        cand = rng.uniform(lo, hi, size=4000)
        ok = [c for c in cand if all(abs(c - a_) > 0.2 and abs(c - b_) > 0.2 for a_, b_ in ints)
              and np.all(np.abs(absent_t - c) > 0.2)
              and mc.covers(c - (BASE_MS + PRE_MS) * 1e-3, c + CTRL_LEN_S + POST_MS * 1e-3)]
        chosen = sorted(ok[:CTRL_PER_SHOT])
        for i, c in enumerate(chosen):
            r = {"shot": shot, "off_s": float(c), "return_s": float(c + CTRL_LEN_S),
                 "half": "correction" if i % 2 == 0 else "heldout"}
            r.update(event_record(mc, c, c + CTRL_LEN_S))
            PS.append(r)
        per_shot[str(shot)] = {"events_all": len(evs), "events_mc_covered": cov,
                               "mc_window_s": [mc.t0, mc.t1], "pseudo_events": len(chosen),
                               "correction": sum(1 for i in range(len(chosen)) if i % 2 == 0),
                               "heldout": sum(1 for i in range(len(chosen)) if i % 2 == 1)}
        del mc, ec
        log(f"  {shot}: {len(evs)} interruptions, {cov} covered, {len(chosen)} pseudo-events")
    return EV, PS, per_shot


cache_f = CACHE / "collected_addendum1.json"
if cache_f.exists():
    log("loading cached event windows (addendum 1)")
    _c = json.loads(cache_f.read_text(encoding="utf-8"))
    EV, PS, PER_SHOT = _c["EV"], _c["PS"], _c["PER_SHOT"]
else:
    log("collecting event windows from the raw streams (addendum 1)")
    EV, PS, PER_SHOT = collect()
    cache_f.write_text(json.dumps({"EV": EV, "PS": PS, "PER_SHOT": PER_SHOT}), encoding="utf-8")

FIRST = json.loads((CACHE / "first_checks.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------------------------------------- drift correction
CORR = {}
for shot in F.RAW_SHOTS:
    cs = [p for p in PS if p["shot"] == shot and p["half"] == "correction"]
    CORR[shot] = {c: {k: float(np.nanmedian([p[c][k] for p in cs])) for k in WIN} for c in ("P03", "P16")}
    for c in ("P03", "P16"):
        CORR[shot]["prof_" + c] = np.nanmedian(np.array([p["prof_" + c] for p in cs], float), 0)


def corrected(rec):
    s = rec["shot"]
    o = {}
    for c in ("P03", "P16"):
        o[c] = {k: rec[c][k] - CORR[s][c][k] for k in WIN}
        o["prof_" + c] = np.array(rec["prof_" + c], float) - CORR[s]["prof_" + c]
    o["C"] = {k: (o["P03"][k] + o["P16"][k]) / 2 for k in WIN}
    o["D"] = {k: (o["P03"][k] - o["P16"][k]) / 2 for k in WIN}
    o["prof_C"] = (o["prof_P03"] + o["prof_P16"]) / 2
    o["prof_D"] = (o["prof_P03"] - o["prof_P16"]) / 2
    return o


SCALE = {}
for shot in F.RAW_SHOTS:
    cs = [corrected(p) for p in PS if p["shot"] == shot and p["half"] == "correction"]
    SCALE[shot] = float(np.median([abs(c["C"]["return_0_3ms"]) for c in cs]))


def unit(rec):
    o = corrected(rec)
    sc = SCALE[rec["shot"]]
    u = {nm: {k: o[nm][k] / sc for k in WIN} for nm in ("C", "D", "P03", "P16")}
    u.update({"prof_" + nm: o["prof_" + nm] / sc for nm in ("C", "D")})
    u["shot"] = rec["shot"]
    return u


one_all = [e for e in EV if e["absent_frames"] == 1]
one = [unit(e) for e in one_all if e["mc_covered"]]
one_raw = [e for e in one_all if e["mc_covered"]]
HO = [unit(p) for p in PS if p["half"] == "heldout"]
SHOTS = list(F.RAW_SHOTS)


def joint_boot(fn_ev, fn_ho, seed):
    """event median of fn_ev over interruptions, held-out median of fn_ho, and their difference, each with NB
    draws of discharges (one draw used for both)."""
    ve = np.array([fn_ev(r) for r in one], float)
    se = np.array([r["shot"] for r in one])
    vh = np.array([fn_ho(r) for r in HO], float)
    sh = np.array([r["shot"] for r in HO])
    rng = np.random.default_rng(seed)
    be, bh = [], []
    pe = {u: np.flatnonzero(se == u) for u in SHOTS}
    ph = {u: np.flatnonzero(sh == u) for u in SHOTS}
    for _ in range(NB):
        d = rng.choice(SHOTS, size=len(SHOTS))
        ie = np.concatenate([pe[u] for u in d])
        ih = np.concatenate([ph[u] for u in d])
        be.append(np.nanmedian(ve[ie]) if len(ie) else np.nan)
        bh.append(np.nanmedian(vh[ih]) if len(ih) else np.nan)
    be, bh = np.array(be), np.array(bh)
    return ({"point": float(np.nanmedian(ve)), "ci95": ci(be), "n": int(np.isfinite(ve).sum())},
            {"point": float(np.nanmedian(vh)), "ci95": ci(bh), "n": int(np.isfinite(vh).sum())},
            {"point": float(np.nanmedian(ve) - np.nanmedian(vh)), "ci95": ci(be - bh)}, be, bh)


# ------------------------------------------------------------------------------------------- checks (repeated (b))
chk_a = {"interruptions_one_frame": len(one_all),
         "median_off_ms": float(np.median([(e["return_s"] - e["off_s"]) * 1e3 for e in one_all]))}
chk_a["ok"] = chk_a["interruptions_one_frame"] == 41 and abs(chk_a["median_off_ms"] - 10.5) < 0.05
HOB = {}
for i, k in enumerate(WIN):
    vh = np.array([r["C"][k] for r in HO], float)
    sh = np.array([r["shot"] for r in HO])
    ph = {u: np.flatnonzero(sh == u) for u in SHOTS}
    rng = np.random.default_rng(SEED + 50 + i)
    bs = [np.nanmedian(vh[np.concatenate([ph[u] for u in rng.choice(SHOTS, size=len(SHOTS))])]) for _ in range(NB)]
    HOB[k] = {"point": float(np.nanmedian(vh)), "ci95": ci(bs), "passes": bool(abs(np.nanmedian(vh)) <= 0.5)}
chk_d = {"events_one_frame_used": len(one), "per_discharge": PER_SHOT}
log(f"check (a) {chk_a['ok']}; repeated check (b): " + ", ".join(f"{k} {v['point']:+.3f} {'pass' if v['passes'] else 'FAIL'}" for k, v in HOB.items()))
log(f"check (d): {len(one)} interruptions; held-out pseudo-events {len(HO)}")


def iv(d, f="{:+.3f}"):
    return f"{f.format(d['point'])} [{f.format(d['ci95'][0])}, {f.format(d['ci95'][1])}]"


def md_checks():
    fc = FIRST["checks"]
    L = ["# B.63 The integrated Mirnov signal through the beam interruptions", "",
         f"Registration: `{REG}` with addendum 1. Runner: `ces_prediction/experiments/b63_blip_magnetics/run_b63.py`. "
         "CPU only, no model, real data (raw 2 MHz Mirnov MC1T03 and MC1T16, raw BES and ECEI of the twelve discharges "
         "with raw streams). Intervals: 95 %, 2,000 resamples of discharges; event medians. Quantities after the drift "
         "correction of addendum 1, in units of the control scatter of the discharge (median |corrected C| of its "
         "correction half in the window 0 to 3 ms after the return).", "",
         "## Checks with the drift correction (addendum 1; written before any statement)", "",
         f"- **(a)** {chk_a['interruptions_one_frame']} interruptions of one absent frame, median off-time "
         f"{chk_a['median_off_ms']:.2f} ms: **{'passed' if chk_a['ok'] else 'FAILED'}**.",
         f"- **(b), repeated on the held-out half** ({len(HO)} held-out pseudo-events; correction half "
         f"{sum(1 for p in PS if p['half'] == 'correction')}): event median of corrected C per window (limit |median| <= 0.5): "
         + "; ".join(f"{k} {iv(v)} **{'passes' if v['passes'] else 'fails'}**" for k, v in HOB.items()) + ".",
         "- **(c)** not recomputed (addendum 1): ratio 2.33 at the 200 BES excursions, excursion median "
         f"{iv(fc['c']['median_abs_change_C_excursions'], '{:.3f}')} against {iv(fc['c']['median_abs_change_C_pseudo'], '{:.3f}')} "
         "control scatters (first-run units).",
         f"- **(d)** {len(one)} of {len(one_all)} one-frame interruptions inside the raw Mirnov coverage. Per discharge "
         "(used / all; pseudo-events correction / held-out): " + ", ".join(
             f"{s} {PER_SHOT[str(s)]['events_mc_covered']}/{PER_SHOT[str(s)]['events_all']} "
             f"({PER_SHOT[str(s)]['correction']}/{PER_SHOT[str(s)]['heldout']})" for s in SHOTS) + ".",
         "", "Control scatter per discharge (V s): " + ", ".join(f"{k} {v:.3e}" for k, v in SCALE.items()) + ".", "",
         "## First checks, without the drift correction (kept for the record)", "",
         f"- (a) passed: {fc['a']['interruptions_one_frame']} interruptions, median off-time {fc['a']['median_off_ms']:.2f} ms.",
         f"- (b) **failed** ({fc['b']['pseudo_events']} pseudo-events, ten per discharge): event median of C "
         + ", ".join(f"{k} {v:+.3f}" for k, v in fc["b"]["control_event_median_C_scatter_units"].items())
         + " control scatters; the last window outside 0.5. Cause (pseudo-events only): the integrated signal drifts, with a "
         "sign that is the same within a discharge and a size growing with the time since the start of the integration.",
         f"- (c) passed: ratio {fc['c']['ratio']:.2f}. (d) {fc['d']['events_one_frame_used']} interruptions covered.",
         "- Code of that run: `data/.b63/run_b63_first_checks.py`; output `data/.b63/first_checks.{md,json}`.", ""]
    return L


INTERP = ["## Interpretations made in running the registration", "",
          "1. Raw arrays are the float32 caches `data/raw_mus/.cache/s{shot}_*.npz` of the raw CSVs (spot-checked against the CSV in the first run: largest difference 4.8e-7).",
          "2. Phi before t_off (for the profile) is the integral from t_off backward, so Phi is zero at t_off on both sides.",
          "3. Pseudo-events: the B.62 rule (uniform in the raw window, no raw interruption and no step above 15 ms of the 10 ms file "
          "within 200 ms), inside the raw Mirnov window, length 10.5 ms; the first 40 drawn, then put in time order and split alternately.",
          "4. Joint intervals: one draw of the twelve discharges gives the event median (over its interruptions), the held-out median "
          "(over its held-out pseudo-events) and their difference.",
          "5. M2 is read if the window 0-3 ms passes check (b) (the check is on C); the added condition is applied to Phi_03 and Phi_16 separately.",
          "6. M3: (median at the later window - median at 0-3 ms) x sign of the full-sample median at 0-3 ms; the added condition uses the "
          "same quantity of the held-out pseudo-events.",
          "7. M4 (D includes zero) is a statement of no change; the added condition of addendum 1 is written for a deviation. The "
          "verdict applies the addendum literally (the interval of event minus held-out must exclude zero), and the reading that "
          "M4 is met when both intervals include zero is reported beside it."]

if not STATEMENTS:
    res = {"registration": REG + " (addendum 1)", "stage": "checks only (addendum 1)",
           "checks": {"a": chk_a, "b_heldout": HOB, "d": chk_d, "control_scatter_Vs": SCALE}, "first_checks": FIRST["checks"],
           "runtime_s": _time.time() - T0}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text("\n".join(md_checks() + ["Statements not yet computed (stage: checks only).", ""] + INTERP) + "\n",
                      encoding="utf-8")
    log("checks written; statements not computed")
    sys.exit(0)

# ------------------------------------------------------------------------------------------- statements
if not chk_a["ok"]:
    raise SystemExit("FATAL: check (a) failed")
if not (HOB["inside_last_3ms"]["passes"] and HOB["return_0_3ms"]["passes"]):
    raise SystemExit("STOP: the repeated check (b) fails in a window that M1 or M2 needs")
log("statements")
ex = lambda d: d["ci95"][0] > 0 or d["ci95"][1] < 0  # noqa: E731
R = {}
i = 0
for nm in ("C", "D", "P03", "P16"):
    for k in WIN:
        e_, h_, d_, _, _ = joint_boot(lambda r, nm=nm, k=k: r[nm][k], lambda r, nm=nm, k=k: r[nm][k], SEED + 100 + i)
        R[f"{nm}_{k}"] = {"interruptions": e_, "heldout": h_, "difference": d_}
        i += 1
r1 = R["C_inside_last_3ms"]
M1 = ex(r1["interruptions"]) and ex(r1["difference"])
a3, a6 = R["P03_return_0_3ms"], R["P16_return_0_3ms"]
M2 = (np.sign(a3["interruptions"]["point"]) == np.sign(a6["interruptions"]["point"]) and ex(a3["interruptions"])
      and ex(a6["interruptions"]) and ex(a3["difference"]) and ex(a6["difference"]))
late = "return_40_43ms" if HOB["return_40_43ms"]["passes"] else ("return_20_23ms" if HOB["return_20_23ms"]["passes"] else None)
if late:
    sgn = np.sign(R["C_return_0_3ms"]["interruptions"]["point"])
    e0, h0, _, be0, bh0 = joint_boot(lambda r: r["C"]["return_0_3ms"], lambda r: r["C"]["return_0_3ms"], SEED + 300)
    eL, hL, _, beL, bhL = joint_boot(lambda r: r["C"][late], lambda r: r["C"][late], SEED + 300)
    m3e = {"point": float((eL["point"] - e0["point"]) * sgn), "ci95": ci((beL - be0) * sgn)}
    m3h = {"point": float((hL["point"] - h0["point"]) * sgn), "ci95": ci((bhL - bh0) * sgn)}
    m3d = {"point": m3e["point"] - m3h["point"], "ci95": ci((beL - be0) * sgn - (bhL - bh0) * sgn)}
    M3 = m3e["ci95"][1] < 0 and ex(m3d)
else:
    m3e = m3h = m3d = None
    M3 = None
r4 = R["D_return_0_3ms"]
incl = lambda d: d["ci95"][0] <= 0 <= d["ci95"][1]  # noqa: E731
M4_own = incl(r4["interruptions"])
M4_literal = M4_own and ex(r4["difference"])
M4_alt = M4_own and incl(r4["difference"])
log(f"M1 {M1} M2 {M2} M3 {M3} (window {late}) M4 literal {M4_literal} / alt {M4_alt}")

# reported
PE = np.array([r["prof_C"] for r in one]); PEd = np.array([r["prof_D"] for r in one])
PH = np.array([r["prof_C"] for r in HO]); PHd = np.array([r["prof_D"] for r in HO])
q = lambda A: (np.nanmedian(A, 0), np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0))  # noqa: E731
pC, pCl, pCh = q(PE)
pD, pDl, pDh = q(PEd)
hC, hCl, hCh = q(PH)
hD, hDl, hDh = q(PHd)
imax = int(np.nanargmax(np.abs(pC)))
t_max = PROF_MS[imax] + 0.5
b62 = json.loads((PAPER / "b62_beam_return.json").read_text(encoding="utf-8"))
ecei_prof = b62["E"]["ECEI"]["profile_return_aligned_median"]
bes_prof = b62["E"]["BES"]["profile_return_aligned_median"]
em = b62["E"]["ECEI"]["profile_return_aligned_ms"]
multi = [unit(e) | {"absent_frames": e["absent_frames"], "off_ms": (e["return_s"] - e["off_s"]) * 1e3}
         for e in EV if e["absent_frames"] > 1 and e["mc_covered"]]
pairs = [(unit(e)["C"]["return_0_3ms"], e["ecei_fall_10_13"]) for e in one_raw if e["ecei_fall_10_13"] is not None]
rc = np.array(pairs, float)
rho = {"rho_C_return_vs_ecei_fall": float(spearmanr(rc[:, 0], rc[:, 1])[0]),
       "rho_absC_return_vs_ecei_fall": float(spearmanr(np.abs(rc[:, 0]), rc[:, 1])[0]), "n": int(len(rc))}
per_dis = {}
for s in SHOTS:
    ee = [r for r in one if r["shot"] == s]
    hh = [r for r in HO if r["shot"] == s]
    if ee:
        per_dis[str(s)] = {"events": len(ee), **{f"C_{k}": float(np.median([r["C"][k] for r in ee])) for k in WIN},
                           "D_return_0_3ms": float(np.median([r["D"]["return_0_3ms"] for r in ee])),
                           "P03_return_0_3ms": float(np.median([r["P03"]["return_0_3ms"] for r in ee])),
                           "P16_return_0_3ms": float(np.median([r["P16"]["return_0_3ms"] for r in ee])),
                           "heldout_C_return_0_3ms": float(np.median([r["C"]["return_0_3ms"] for r in hh]))}


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = np.array(PROF_MS) + 0.5
    off = float(np.median([(e["return_s"] - e["off_s"]) * 1e3 for e in one_raw]))
    fig, axs = plt.subplots(3, 1, figsize=(5.8, 6.6), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.9]})
    for ax in axs:
        ax.axvspan(-off, 0, color="0.9", lw=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax, (m, lo, hi, mh, loh, hih, lab) in zip(axs[:2], ((pC, pCl, pCh, hC, hCl, hCh, "C (common), control scatters"),
                                                            (pD, pDl, pDh, hD, hDl, hDh, "D (difference), control scatters"))):
        ax.axhline(0, color="0.8", lw=0.6)
        ax.fill_between(ms, loh, hih, color="0.6", alpha=0.25, lw=0)
        ax.plot(ms, mh, color="0.4", lw=1.0, ls="--")
        ax.fill_between(ms, lo, hi, color="#1f4e9a", alpha=0.18, lw=0)
        ax.plot(ms, m, color="#1f4e9a", lw=1.3)
        ax.text(ms[-1] + 1, m[-1], "interruptions", color="#1f4e9a", fontsize=7, va="center")
        ax.text(ms[-1] + 1, mh[-1] - 0.25 * (np.nanmax(hih) - np.nanmin(loh)) / 2, "held-out\npseudo-events", color="0.35", fontsize=7, va="center")
        ax.set_ylabel(lab, fontsize=7)
    a2 = axs[2]
    a2.axhline(1, color="0.8", lw=0.6)
    emx = np.array(em) + 0.5
    a2.plot(emx, bes_prof, color="#7f8c8d", lw=1.2)
    a2.plot(emx, ecei_prof, color="#c0392b", lw=1.2)
    a2.text(emx[-1] + 1, bes_prof[-1] + 0.03, "BES level", color="#7f8c8d", fontsize=7, va="center")
    a2.text(emx[-1] + 1, ecei_prof[-1] - 0.04, "ECEI level", color="#c0392b", fontsize=7, va="center")
    a2.set_ylabel("ratio to before (B.62)", fontsize=7)
    a2.set_xlabel("time from the return of the beam (ms); shaded: beam off; bands: interquartile", fontsize=8)
    a2.set_xlim(ms[0] - 1, ms[-1] + 24)
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".png"))
    fig.savefig(FIG.with_suffix(".pdf"))
    plt.close(fig)


figure()


def hv(x):
    return "not read" if x is None else ("**holds**" if x else "**does not hold**")


res = {"registration": REG + " (addendum 1)", "stage": "checks and statements (addendum 1)",
       "checks": {"a": chk_a, "b_heldout": HOB, "d": chk_d, "control_scatter_Vs": SCALE}, "first_checks": FIRST["checks"],
       "windows": R, "M3": {"later_window": late, "interruptions": m3e, "heldout": m3h, "difference": m3d},
       "verdicts": {"M1": bool(M1), "M2": bool(M2), "M3": None if M3 is None else bool(M3), "M4_literal": bool(M4_literal),
                    "M4_both_intervals_include_zero": bool(M4_alt)},
       "profile_ms_from_return": PROF_MS, "profile_C": pC.tolist(), "profile_C_iqr": [pCl.tolist(), pCh.tolist()],
       "profile_D": pD.tolist(), "profile_heldout_C": hC.tolist(), "profile_heldout_D": hD.tolist(),
       "time_of_largest_abs_C_ms_from_return": t_max, "largest_abs_C": float(pC[imax]),
       "longer_interruptions": [{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in m.items() if not k.startswith("prof")} for m in multi],
       "rank_correlation": rho, "per_discharge": per_dis, "runtime_s": _time.time() - T0}
OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

L = md_checks()
L += ["## Statements", "",
      "Each statement needs its own interval and the interval of the difference interruptions minus held-out pseudo-events "
      "(addendum 1). Units: control scatters.", "",
      "| statement | interruptions | held-out pseudo-events | difference | verdict | expected |", "|---|---|---|---|---|---|",
      f"| M1 C, last 3 ms before the return, differs from 0 | {iv(r1['interruptions'])} | {iv(r1['heldout'])} | {iv(r1['difference'])} | {hv(M1)} | met, 1 to 4 |",
      f"| M2 Phi_03, 0-3 ms after the return | {iv(a3['interruptions'])} | {iv(a3['heldout'])} | {iv(a3['difference'])} | {hv(M2)} | met (same sign, both exclude 0) |",
      f"| M2 Phi_16, 0-3 ms after the return | {iv(a6['interruptions'])} | {iv(a6['heldout'])} | {iv(a6['difference'])} | | |"]
if late:
    L.append(f"| M3 C decays ({late.replace('return_', '')} minus 0-3 ms, x sign at 0-3 ms) below 0 | {iv(m3e)} | {iv(m3h)} | {iv(m3d)} | {hv(M3)} | met (a half or more smaller) |")
else:
    L.append("| M3 | - | - | - | not read (neither later window passes check (b)) | |")
L += [f"| M4 D, 0-3 ms after the return, includes 0 | {iv(r4['interruptions'])} | {iv(r4['heldout'])} | {iv(r4['difference'])} | "
      f"literal: {hv(M4_literal)}; both intervals include 0: {hv(M4_alt)} | met |", "",
      f"Events: {r1['interruptions']['n']} interruptions of one absent frame; held-out pseudo-events: {r1['heldout']['n']}.", "",
      "## All windows (medians, control scatters)", "",
      "| quantity | window | interruptions | held-out | difference |", "|---|---|---|---|---|"]
for nm in ("C", "D", "P03", "P16"):
    for k in WIN:
        x = R[f"{nm}_{k}"]
        L.append(f"| {nm} | {k} | {iv(x['interruptions'])} | {iv(x['heldout'])} | {iv(x['difference'])} |")
L += ["", f"Largest |event-median C| in the 1 ms profile: {pC[imax]:+.3f} control scatters at {t_max:+.1f} ms from the return of the beam.", "",
      "### 1 ms profiles (medians, control scatters; ECEI and BES levels of B.62 beside)", "",
      "| ms from return | C | D | held-out C | held-out D | ECEI (B.62) | BES (B.62) |", "|---|---|---|---|---|---|---|"]
for i_, k in enumerate(PROF_MS):
    if k in (-45, -40, -35, -30, -25, -20, -15, -12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 5, 7, 10, 13, 15, 20, 25, 30, 40, 50, 60, 70, 79):
        j = em.index(k) if k in em else None
        L.append(f"| {k} | {pC[i_]:+.3f} | {pD[i_]:+.3f} | {hC[i_]:+.3f} | {hD[i_]:+.3f} | "
                 f"{'-' if j is None else f'{ecei_prof[j]:.3f}'} | {'-' if j is None else f'{bes_prof[j]:.3f}'} |")
L += ["", "## Longer interruptions (two and three absent frames; reported)", "",
      "| shot | absent frames | off (ms) | C inside / 0-3 / 10-13 / 20-23 / 40-43 | D 0-3 |", "|---|---|---|---|---|"]
for m in multi:
    L.append(f"| {m['shot']} | {m['absent_frames']} | {m['off_ms']:.1f} | " + " / ".join(f"{m['C'][k]:+.2f}" for k in WIN)
             + f" | {m['D']['return_0_3ms']:+.2f} |")
L += ["", f"Rank correlation over one-frame interruptions with an ECEI readout (n {rho['n']}): corrected C at 0-3 ms after the return "
      f"vs the ECEI fall 10-13 ms after it {rho['rho_C_return_vs_ecei_fall']:+.3f}; |C| vs the fall {rho['rho_absC_return_vs_ecei_fall']:+.3f}.", "",
      "## Per discharge (medians over its one-frame interruptions, control scatters)", "",
      "| shot | events | C inside | C 0-3 | C 10-13 | C 40-43 | D 0-3 | Phi_03 0-3 | Phi_16 0-3 | held-out C 0-3 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for s, d in per_dis.items():
    L.append(f"| {s} | {d['events']} | {d['C_inside_last_3ms']:+.2f} | {d['C_return_0_3ms']:+.2f} | {d['C_return_10_13ms']:+.2f} | "
             f"{d['C_return_40_43ms']:+.2f} | {d['D_return_0_3ms']:+.2f} | {d['P03_return_0_3ms']:+.2f} | {d['P16_return_0_3ms']:+.2f} | "
             f"{d['heldout_C_return_0_3ms']:+.2f} |")
L += [""] + INTERP + ["", f"Run time {res['runtime_s']:.0f} s (statement stage)."]
OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
log("wrote outputs")
