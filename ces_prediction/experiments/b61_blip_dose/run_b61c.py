"""B.61 part C (PREREGISTRATION_B61.md): dose of a beam interruption.  CPU only, no model, all 641 files, cut
population, event and ratio definitions of B.58 (imported from b58_blip_calibration/uc_blip_shape.py, unchanged).

Run from the repo root:  py ces_prediction/experiments/b61_blip_dose/run_b61c.py
Outputs: docs/paper/b61_blip_dose.{json,md}
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
B58 = HERE.parent / "b58_blip_calibration"
sys.path.insert(0, str(B58))
T0 = _time.time()
NB = 2000
SEED = 20261119
REG = "ces_prediction/experiments/PREREGISTRATION_B61.md"
PAPER = REPO / "docs" / "paper"
OUT_JSON = PAPER / "b61_blip_dose.json"
OUT_MD = PAPER / "b61_blip_dose.md"
Q = ("ti", "ecei", "bes")


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


log("importing b58_blip_calibration/uc_blip_shape.py (runs it; event and ratio code of B.58)")
with contextlib.redirect_stdout(io.StringIO()):
    import uc_blip_shape as SH  # noqa: E402
ev, evf, evn, K = SH.ev, np.asarray(SH.evf), np.asarray(SH.evn), SH.K
CLASSES = (1, 2, 3)


def summ(m):
    """first-row ratio and mean ratio of rows 1-3 of an event-median profile (SH.stat)."""
    return {q: (float(m[q][0]), float(np.mean(m[q][:3]))) for q in Q}


# --- full sample
full = {n: SH.stat(ev, evf, np.flatnonzero(evn == n)) for n in CLASSES}
counts = {n: {"events": int((evn == n).sum()), "discharges": int(len(np.unique(evf[evn == n])))} for n in CLASSES}

# --- check against B.58 (one absent frame)
b58 = json.loads((PAPER / "b58_blip_calibration.json").read_text(encoding="utf-8"))
ref = {q: b58["S"]["all"]["profiles"][q][0] for q in Q}
check = {q: {"b61": float(full[1][q][0]), "b58": float(ref[q])} for q in Q}
check_ok = all(abs(v["b61"] - v["b58"]) < 1e-12 for v in check.values())
log(f"check: first-row ratios of one-frame events reproduce B.58: {check_ok} "
    + ", ".join(f"{q} {check[q]['b61']:.3f}" for q in Q))
if not check_ok:
    raise SystemExit("FATAL: B.58 first-row ratios not reproduced; stopping before any statement")

# --- bootstrap: one draw of discharges (over all discharges with any event), all classes from the same draw
uf = np.unique(evf)
pos = {u: np.flatnonzero(evf == u) for u in uf}
rng = np.random.default_rng(SEED)
draws = {n: [] for n in CLASSES}
for _ in range(NB):
    sel = np.concatenate([pos[u] for u in rng.choice(uf, size=len(uf))])
    for n in CLASSES:
        s = sel[evn[sel] == n]
        draws[n].append(SH.stat(ev, evf, s) if len(s) else None)
log("bootstrap done")

res_cls = {}
for n in CLASSES:
    o = {**counts[n], "first_row_ratio": {}, "mean_ratio_rows_1_to_3": {}, "profile_rows_1_to_10": {},
         "profile_ci95": {}}
    for q in Q:
        o["first_row_ratio"][q] = {"point": float(full[n][q][0]),
                                   "ci95": ci([m[q][0] for m in draws[n] if m is not None])}
        o["mean_ratio_rows_1_to_3"][q] = {"point": float(np.mean(full[n][q][:3])),
                                          "ci95": ci([np.mean(m[q][:3]) for m in draws[n] if m is not None])}
        o["profile_rows_1_to_10"][q] = [float(v) for v in full[n][q]]
        o["profile_ci95"][q] = [ci([m[q][k] for m in draws[n] if m is not None]) for k in range(len(K))]
    res_cls[str(n)] = o

diff = {}
for n in (2, 3):
    d = {}
    for q in Q:
        pt = (1 - full[n][q][0]) - (1 - full[1][q][0])
        b = [(1 - m[q][0]) - (1 - m1[q][0]) for m, m1 in zip(draws[n], draws[1]) if m is not None and m1 is not None]
        d[q] = {"point": float(pt), "ci95": ci(b), "draws_used": len(b)}
        pt3 = np.mean(1 - full[n][q][:3]) - np.mean(1 - full[1][q][:3])
        b3 = [np.mean(1 - m[q][:3]) - np.mean(1 - m1[q][:3]) for m, m1 in zip(draws[n], draws[1])
              if m is not None and m1 is not None]
        d[q + "_mean_rows_1_to_3"] = {"point": float(pt3), "ci95": ci(b3)}
    diff[f"{n}_minus_1"] = d
C1 = diff["2_minus_1"]["ti"]["ci95"][0] > 0
C2 = diff["2_minus_1"]["ecei"]["ci95"][0] > 0
log(f"C1 {C1}  C2 {C2}")

# --- arithmetic: beam-off duration per class on the twelve raw discharges (from the B.58 raw add-on)
raw = b58["raw_addon"]["rows"]
dur = {}
for n in CLASSES:
    v = [r["duration_ms"] for r in raw if r["absent_frames"] == n]
    dur[str(n)] = {"blips": len(v), "duration_ms_median": float(np.median(v)) if v else None,
                   "duration_ms_range": [float(min(v)), float(max(v))] if v else None}

res = {"registration": REG, "part": "C", "bootstrap_resamples": NB, "seed": SEED,
       "b58_shape_script_sha256": hashlib.sha256((B58 / "uc_blip_shape.py").read_bytes()).hexdigest(),
       "check_b58_first_row_one_frame": {"ok": check_ok, "values": check},
       "classes": res_cls, "differences_first_row_deficit": diff,
       "C1_holds": bool(C1), "C2_holds": bool(C2),
       "arithmetic_beam_off_duration_raw_twelve_discharges": dur,
       "runtime_s": _time.time() - T0}
OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")


def iv(d, f="{:.3f}"):
    return f"{f.format(d['point'])} [{f.format(d['ci95'][0])}, {f.format(d['ci95'][1])}]"


nm = {"ti": "T_i", "ecei": "ECEI level", "bes": "BES level"}
M = ["# B.61 part C: dose of a beam interruption", "",
     f"Registration: `{REG}` (part C only; parts A and B are run separately). Runner: "
     "`ces_prediction/experiments/b61_blip_dose/run_b61c.py`, event and ratio code imported unchanged from "
     "`ces_prediction/experiments/b58_blip_calibration/uc_blip_shape.py`. CPU only, no model, all 641 files, cut "
     "population. Ratios are event medians of the value over the median of the six rows before. Intervals: 95 %, "
     "2,000 resamples of the discharges that have any event; every class and every difference is computed within "
     "the same draw.", "",
     "## Check before any statement", "",
     "First-row ratios of events with one absent frame, against B.58: " + ", ".join(
         f"{nm[q]} {check[q]['b61']:.3f} (B.58 {check[q]['b58']:.3f})" for q in Q)
     + f". Reproduced exactly: {check_ok}.", "",
     "## Statements", "",
     "| statement | difference of the first-row deficit, two minus one absent frame | verdict | expected |",
     "|---|---|---|---|",
     f"| C1 T_i deficit larger after two absent frames than after one | {iv(diff['2_minus_1']['ti'], '{:+.3f}')} | "
     f"**{'holds' if C1 else 'does not hold'}** | uncertain, expected to fail |",
     f"| C2 the same for the ECEI level | {iv(diff['2_minus_1']['ecei'], '{:+.3f}')} | "
     f"**{'holds' if C2 else 'does not hold'}** | uncertain |", "",
     "## Per class", "",
     "| absent frames | events (discharges) | quantity | first-row ratio | mean ratio rows 1-3 |", "|---|---|---|---|---|"]
for n in CLASSES:
    o = res_cls[str(n)]
    for q in Q:
        M.append(f"| {n} | {o['events']} ({o['discharges']}) | {nm[q]} | {iv(o['first_row_ratio'][q])} | "
                 f"{iv(o['mean_ratio_rows_1_to_3'][q])} |")
M += ["", "## Differences of the deficit (one minus the ratio), within a draw (reported)", "",
      "| difference | quantity | first row | mean of rows 1-3 |", "|---|---|---|---|"]
for k_, d in diff.items():
    for q in Q:
        M.append(f"| {k_.replace('_', ' ')} | {nm[q]} | {iv(d[q], '{:+.3f}')} | {iv(d[q + '_mean_rows_1_to_3'], '{:+.3f}')} |")
M += ["", "## Event-median profiles, rows 1 to 10 (ratio to the median of the six rows before)", "",
      "| row | " + " | ".join(f"{nm[q]}, {n} frame{'s' if n > 1 else ''}" for q in Q for n in CLASSES) + " |",
      "|---|" + "---|" * 9]
for i, k in enumerate(K):
    M.append(f"| +{k} | " + " | ".join(f"{res_cls[str(n)]['profile_rows_1_to_10'][q][i]:.3f}" for q in Q for n in CLASSES) + " |")
M += ["", "## Arithmetic, not a statement: how long the beam is off in each class", "",
      "From the raw BES of the twelve discharges with raw streams (B.58 add-on: 0.1 ms level below half of its 200 ms "
      "running median for at least 3 ms, matched to the absent frames).", "",
      "| absent frames | blips in the raw windows | beam off, median (ms) | range (ms) |", "|---|---|---|---|"]
for n in CLASSES:
    d = dur[str(n)]
    if d["blips"]:
        M.append(f"| {n} | {d['blips']} | {d['duration_ms_median']:.1f} | {d['duration_ms_range'][0]:.1f} to {d['duration_ms_range'][1]:.1f} |")
    else:
        M.append(f"| {n} | 0 | - | - |")
M += ["", "Interpretations: the first-row ratio and the mean of rows 1-3 are read from the event-median profile "
      "(the B.58 statistic `stat`), so 'mean ratio of rows 1 to 3' is the mean of three event medians; the "
      "bootstrap draws discharges once from all discharges with an event of any class and computes every class "
      "from that draw (a draw with no event of a class is skipped for that class).", "",
      f"Run time {res['runtime_s']:.0f} s."]
OUT_MD.write_text("\n".join(M) + "\n", encoding="utf-8")
log("wrote " + OUT_JSON.name + ", " + OUT_MD.name)
