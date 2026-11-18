"""B.63 addendum 2 (PREREGISTRATION_B63.md): relative polarity of the two Mirnov coils (MC1T03, MC1T16), read at the
200 BES common-mode excursions of check (c) (candidate edge bursts).  Exploratory, decided after the statements.
CPU only, no model, real data.

For each excursion: the change of Phi_03 and of Phi_16 between the 5 ms before and the 5 ms after it, with the
integration and baseline of check (c) (Phi zero at the excursion, baseline the mean of the coil signal over the 30 ms
before it; the code of run_b63.py::integrated / burst_change, applied per coil instead of to C).  The same quantity
at the held-out pseudo-events of addendum 1 (taken at the middle of the pseudo-interruption, as check (c) did at its
pseudo-events).  For comparison: the 36 one-frame interruptions, Phi_03 and Phi_16 0 to 3 ms after the return with
Phi zero at t_off, drift-corrected and in control-scatter units exactly as in the statements.

Quantities: share of events with opposite signs, Spearman rank correlation over events, event median of
Phi_16/Phi_03; 95 % intervals from 2,000 resamples of discharges.  Also event medians of the two changes in
control-scatter units (addendum 1 scale of the discharge).

The registered runner is not imported (importing it would rerun it and rewrite its outputs); its helpers are
copied.  The excursion list is read from data/.b63/collected.json (first-check cache, the list check (c) used); the
held-out pseudo-events and the interruptions from data/.b63/collected_addendum1.json.

Run from the repo root:
    py ces_prediction/experiments/b63_blip_magnetics/polarity_b63.py
Appends "## Addendum 2 ..." to docs/paper/b63_blip_magnetics.md (replacing an earlier such section) and the key
addendum2_polarity to docs/paper/b63_blip_magnetics.json.  Per-event values cached in data/.b63/polarity_events.json.
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
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "ces_prediction"))
import analyze_file_rules as F  # noqa: E402

warnings.simplefilter("ignore", category=RuntimeWarning)
T0 = _time.time()
NB = 2000
SEED = 20261122
BASE_MS = 30
CTRL_LEN_S = 0.0105
WIN = ("inside_last_3ms", "return_0_3ms", "return_10_13ms", "return_20_23ms", "return_40_43ms")
PAPER = REPO / "docs" / "paper"
OUT_JSON = PAPER / "b63_blip_magnetics.json"
OUT_MD = PAPER / "b63_blip_magnetics.md"
CACHE = REPO / "data" / ".b63"
HEAD = "## Addendum 2: relative polarity of the coils (exploratory, decided after the statements)"


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


# ------------------------------------------------------------------------------- helpers copied from run_b63.py
class Raw:
    def __init__(self, shot, diag):
        x, fs, t0, ch = F.raw_npz(shot, diag)
        self.fs, self.t0, self.ch = fs, t0, ch
        self.x = x
        self.t1 = t0 + len(x) / fs

    def idx(self, t):
        return int(round((t - self.t0) * self.fs))


def integrated(mc, t_ref, t_from, t_to):
    i_ref = mc.idx(t_ref)
    i_b0 = mc.idx(t_ref - BASE_MS * 1e-3)
    base = mc.x[i_b0:i_ref].astype(np.float64).mean(0)
    i0, i1 = mc.idx(t_from), mc.idx(t_to)
    seg = mc.x[i0:i1].astype(np.float64) - base
    dt = 1.0 / mc.fs
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


def burst_change_per_coil(mc, te):
    """check (c) quantity, per coil: Phi(5 ms after) - Phi(5 ms before), Phi zero at te, 30 ms baseline (V s)."""
    tt, phi = integrated(mc, te, te - 5e-3, te + 5e-3)
    return [win_mean(tt, phi[:, j], 0, 5e-3) - win_mean(tt, phi[:, j], -5e-3, 0) for j in (0, 1)]


# ------------------------------------------------------------------------------- inputs
first = json.loads((CACHE / "collected.json").read_text(encoding="utf-8"))
EXC = first["EXC"]
a1 = json.loads((CACHE / "collected_addendum1.json").read_text(encoding="utf-8"))
EV, PS = a1["EV"], a1["PS"]
RES = json.loads(OUT_JSON.read_text(encoding="utf-8"))
SCALE = {int(k): float(v) for k, v in RES["checks"]["control_scatter_Vs"].items()}
SHOTS = list(F.RAW_SHOTS)
assert len(EXC) == 200, len(EXC)

cache_f = CACHE / "polarity_events.json"
if cache_f.exists():
    log("loading cached per-coil changes")
    PC = json.loads(cache_f.read_text(encoding="utf-8"))
else:
    log("computing per-coil changes from the raw Mirnov streams")
    PC = {"excursions": [], "heldout": []}
    for shot in SHOTS:
        mc = Raw(shot, "MC")
        assert list(mc.ch) == ["MC1T03", "MC1T16"], mc.ch
        for r in EXC:
            if r["shot"] == shot:
                d03, d16 = burst_change_per_coil(mc, r["t_s"])
                # consistency with check (c): C change = mean of the two
                assert abs((d03 + d16) / 2 - r["change_C_Vs"]) <= 1e-9 + 1e-6 * abs(r["change_C_Vs"]), (shot, r["t_s"])
                PC["excursions"].append({"shot": shot, "t_s": r["t_s"], "d03_Vs": d03, "d16_Vs": d16})
        for p in PS:
            if p["shot"] == shot and p["half"] == "heldout":
                d03, d16 = burst_change_per_coil(mc, p["off_s"] + CTRL_LEN_S / 2)
                PC["heldout"].append({"shot": shot, "t_s": p["off_s"] + CTRL_LEN_S / 2, "d03_Vs": d03, "d16_Vs": d16})
        del mc
        log(f"  {shot} done")
    cache_f.write_text(json.dumps(PC), encoding="utf-8")

# interruptions: drift-corrected Phi_03, Phi_16 at 0-3 ms after the return, control-scatter units (as the statements)
CORR = {}
for shot in SHOTS:
    cs = [p for p in PS if p["shot"] == shot and p["half"] == "correction"]
    CORR[shot] = {c: {k: float(np.nanmedian([p[c][k] for p in cs])) for k in WIN} for c in ("P03", "P16")}
INT = []
for e in EV:
    if e["absent_frames"] == 1 and e["mc_covered"]:
        s = e["shot"]
        u03 = (e["P03"]["return_0_3ms"] - CORR[s]["P03"]["return_0_3ms"]) / SCALE[s]
        u16 = (e["P16"]["return_0_3ms"] - CORR[s]["P16"]["return_0_3ms"]) / SCALE[s]
        INT.append({"shot": s, "u03": u03, "u16": u16,
                    "raw03": e["P03"]["return_0_3ms"] / SCALE[s], "raw16": e["P16"]["return_0_3ms"] / SCALE[s]})
assert len(INT) == 36, len(INT)
# consistency with the statements: per-discharge medians of Phi_03 0-3 ms
for s, d in RES["per_discharge"].items():
    v = float(np.median([r["u03"] for r in INT if r["shot"] == int(s)]))
    assert abs(v - d["P03_return_0_3ms"]) < 1e-6 * max(1, abs(v)), (s, v, d["P03_return_0_3ms"])
for grp in ("excursions", "heldout"):
    for r in PC[grp]:
        r["u03"], r["u16"] = r["d03_Vs"] / SCALE[r["shot"]], r["d16_Vs"] / SCALE[r["shot"]]


# ------------------------------------------------------------------------------- quantities
def stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return (float(np.mean(np.sign(a) * np.sign(b) < 0)), float(spearmanr(a, b)[0]) if len(a) > 2 else np.nan,
            float(np.median(b / a)), float(np.median(a)), float(np.median(b)))


NAMES = ("share_opposite", "spearman", "median_ratio_16_over_03", "median_phi03_scatter", "median_phi16_scatter")


def boot(recs, k03, k16, seed):
    a = np.array([r[k03] for r in recs], float)
    b = np.array([r[k16] for r in recs], float)
    sh = np.array([r["shot"] for r in recs])
    us = [u for u in SHOTS if (sh == u).any()]
    pos = {u: np.flatnonzero(sh == u) for u in us}
    pt = stats(a, b)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(NB):
        ii = np.concatenate([pos[u] for u in rng.choice(us, size=len(us))])
        B.append(stats(a[ii], b[ii]))
    B = np.array(B)
    out = {nm: {"point": pt[j], "ci95": ci(B[:, j])} for j, nm in enumerate(NAMES)}
    out["n"] = int(len(a))
    out["discharges"] = len(us)
    per = {}
    for u in us:
        i = pos[u]
        s_ = stats(a[i], b[i])
        per[str(u)] = {"n": int(len(i)), "share_opposite": s_[0], "median_ratio_16_over_03": s_[2],
                       "spearman": s_[1] if len(i) >= 5 else None}
    out["per_discharge"] = per
    return out


R = {"excursions": boot(PC["excursions"], "u03", "u16", SEED + 1),
     "heldout_pseudo_events": boot(PC["heldout"], "u03", "u16", SEED + 2),
     "interruptions_corrected": boot(INT, "u03", "u16", SEED + 3),
     "interruptions_uncorrected": boot(INT, "raw03", "raw16", SEED + 4)}
ex = R["excursions"]
OPP = ex["share_opposite"]["ci95"][0] > 0.5 and ex["spearman"]["ci95"][1] < 0
VERDICT = ("the two coils have opposite effective polarity for such changes; read as exploratory: the wall field "
           "changes alike at two toroidal positions from the moment the beam goes off and for about 50 ms after its "
           "return") if OPP else (
           "the two coils change in opposite directions at the interruptions only, which a purely axisymmetric change "
           "does not explain")
for k, v in R.items():
    log(f"{k}: n {v['n']}, opposite {v['share_opposite']['point']:.3f} {v['share_opposite']['ci95']}, "
        f"rho {v['spearman']['point']:+.3f} {v['spearman']['ci95']}, ratio {v['median_ratio_16_over_03']['point']:+.3f} "
        f"{v['median_ratio_16_over_03']['ci95']}")
log("branch: " + ("opposite polarity" if OPP else "not opposite"))

# ------------------------------------------------------------------------------- outputs
RES["addendum2_polarity"] = {
    "registration": "PREREGISTRATION_B63.md addendum 2 (exploratory, decided after the statements)",
    "runner": "ces_prediction/experiments/b63_blip_magnetics/polarity_b63.py",
    "units": "per-coil changes divided by the addendum-1 control scatter of the discharge (V s)",
    "results": R, "opposite_polarity_branch": bool(OPP), "verdict": VERDICT, "runtime_s": _time.time() - T0}
OUT_JSON.write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")


def iv(d, f="{:+.3f}"):
    return f"{f.format(d['point'])} [{f.format(d['ci95'][0])}, {f.format(d['ci95'][1])}]"


LAB = {"excursions": "BES excursions (candidate edge bursts), change 5 ms after minus 5 ms before",
       "heldout_pseudo_events": "held-out pseudo-events, same quantity at the middle of the pseudo-interruption",
       "interruptions_corrected": "one-frame interruptions, Phi 0-3 ms after the return (t_off baseline), drift-corrected (as the statements)",
       "interruptions_uncorrected": "one-frame interruptions, same, without the drift correction"}
L = [HEAD, "",
     "Exploratory check of addendum 2, decided after the statements were read; not a statement. Runner: "
     "`ces_prediction/experiments/b63_blip_magnetics/polarity_b63.py` (the registered runner is untouched). At each of "
     "the 200 BES excursions of check (c) the change of Phi_03 and of Phi_16 between the 5 ms before and the 5 ms after "
     "the excursion, with the integration and baseline of check (c) (Phi zero at the excursion, baseline the mean over "
     "the 30 ms before; the mean of the two changes reproduces the check-(c) change of C for every excursion). Changes "
     "in units of the addendum-1 control scatter of the discharge. Share of events with opposite signs, Spearman rank "
     "correlation over events, event median of Phi_16/Phi_03; 95 % intervals, 2,000 resamples of discharges.", "",
     "| events | n (discharges) | share opposite signs | rank correlation | median Phi_16/Phi_03 | median Phi_03 | median Phi_16 |",
     "|---|---|---|---|---|---|---|"]
for k, v in R.items():
    L.append(f"| {LAB[k]} | {v['n']} ({v['discharges']}) | {iv(v['share_opposite'])} | {iv(v['spearman'])} | "
             f"{iv(v['median_ratio_16_over_03'])} | {iv(v['median_phi03_scatter'])} | {iv(v['median_phi16_scatter'])} |")
L += ["", "Per discharge (n, share opposite, median Phi_16/Phi_03):", "",
      "| shot | excursions | held-out pseudo-events | interruptions (corrected) |", "|---|---|---|---|"]


def cell(v, s):
    d = v["per_discharge"].get(s)
    return "-" if d is None else f"{d['n']}, {d['share_opposite']:.2f}, {d['median_ratio_16_over_03']:+.2f}"


for s in map(str, SHOTS):
    L.append(f"| {s} | {cell(R['excursions'], s)} | {cell(R['heldout_pseudo_events'], s)} | {cell(R['interruptions_corrected'], s)} |")
L += ["", "Rule of addendum 2: opposite effective polarity if, at the excursions, the interval of the share of opposite "
      "signs lies above 0.5 and the interval of the rank correlation below 0.", "",
      f"**Branch of addendum 2: {VERDICT}.**", "",
      "In either case no size of a displacement is given and the fall of T_i is not attributed to it.", ""]
md = OUT_MD.read_text(encoding="utf-8")
if HEAD in md:
    md = md[:md.index(HEAD)].rstrip("\n") + "\n"
OUT_MD.write_text(md.rstrip("\n") + "\n\n" + "\n".join(L) + "\n", encoding="utf-8")
log("wrote md and json")
