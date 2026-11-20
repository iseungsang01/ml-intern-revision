# -*- coding: utf-8 -*-
"""B.86, recomputation by the session lead with code that shares nothing with the batch.

    py ces_prediction/experiments/b86_mast_radius/lead_checks/verify_b86.py

A  Targets and history values from the per-shot value arrays: for 300 random shots, the level of the
   following frames (L5), and the newest visible value and its age under D5 and D100, recomputed with
   plain loops from the value array Y and the good mask Gm of the shot, against the stored rows.
B  Statement values from the stored estimates: RMSE of hold, causal GP, T0 and TX, the skills of R1 to
   R4, the cost of staleness, the share within 10 per cent and the points of R5 to R7, recomputed from
   rows_{band}.npz and fits/{band}/*.npz, against docs/paper/b86_mast_radius.json.
C  Session intervals of R1.E, R3.E, R5 and R7 with an own bootstrap (4,000 draws, another seed).

Writes docs/paper/b86_lead_checks.md.  ASCII only on the console.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
C86 = ROOT / "data" / ".b86"
OUT = ROOT / "docs" / "paper" / "b86_lead_checks.md"
J = json.loads((ROOT / "docs" / "paper" / "b86_mast_radius.json").read_text(encoding="utf-8"))
BANDS = ("E", "B2", "B3", "B4", "B5")
lines = ["# B.86: recomputation by the session lead", "",
         "Code: `ces_prediction/experiments/b86_mast_radius/lead_checks/verify_b86.py` (no import from the batch).", ""]


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


# ---------------------------------------------------------------- A
rng = np.random.default_rng(8601)
files = sorted((C86 / "shots").glob("*.npz"))
pick = rng.choice(len(files), size=300, replace=False)
n_rows = 0
bad = {"L5": 0, "D5_H": 0, "D5_age": 0, "D100_H": 0, "D100_age": 0}
n_shots = 0
for i in pick:
    z = np.load(files[i], allow_pickle=True)
    if str(z["status"]) != "ok" or "Y" not in z.files or len(z["j"]) == 0:
        continue
    Y, Gm = z["Y"], z["Gm"].astype(bool)
    n_shots += 1
    cc, jj = z["c"].astype(int), z["j"].astype(int)
    for r in range(len(jj)):
        c, j = cc[r], jj[r]
        col_y, col_g = Y[:, c], Gm[:, c]
        nxt = [col_y[k] for k in range(j + 1, min(j + 6, len(col_y))) if col_g[k]]
        l5 = float(np.median(nxt)) if len(nxt) >= 3 else np.nan
        out = {"L5": l5}
        for name, lag in (("D5", 1), ("D100", 20)):
            h, age = np.nan, np.nan
            for k in range(j - lag, -1, -1):
                if col_g[k]:
                    h, age = col_y[k], 5.0 * (j - k)
                    break
            out[name + "_H"], out[name + "_age"] = h, age
        for key, val in out.items():
            st = float(z[key][r])
            if not ((st == val) or (np.isnan(st) and np.isnan(val)) or abs(st - val) <= 1e-9 * max(1.0, abs(val))):
                bad[key] += 1
        n_rows += 1
print("A: shots %d rows %d mismatches %s" % (n_shots, n_rows, bad))
lines += ["## A. Targets and history values from the value arrays", "",
          "%d shots drawn at random (seed 8601), %s stored rows. L5 = median of the good values of the chord at frames j+1 to j+5 (at least three); the newest visible value under D5 is the newest good value at frames up to j-1, under D100 at frames up to j-20; age = 5 ms times the frame distance." % (n_shots, format(n_rows, ",")), "",
          "| quantity | rows that differ from the stored value |", "|---|---|"]
lines += ["| %s | %d |" % (k, v) for k, v in bad.items()]
lines.append("")

# ---------------------------------------------------------------- B
meta = json.loads((C86 / "shots.json").read_text(encoding="utf-8"))["meta"]


def session_of(shot):
    m = meta[str(int(shot))] if isinstance(meta, dict) else None
    if isinstance(m, dict):
        for key in ("session", "sess", "date", "day"):
            if key in m:
                return m[key]
    return m


def load_est(band, unit, est, rule, target, n):
    z = np.load(C86 / "fits" / band / ("%s__%s__%s__%s.npz" % (unit, est, rule, target)))
    v = np.full(n, np.nan)
    v[z["idx"]] = z["est"]
    return v


def pooled(band, design):
    r = np.load(C86 / ("rows_%s.npz" % band), allow_pickle=True)
    n = len(r["y"])
    units = [u["name"] for u in json.loads((C86 / ("units_%s.json" % band)).read_text(encoding="utf-8"))["units"]
             if u["design"] == design]
    est = {}
    for e in ("T0", "TX"):
        for rule in ("D5", "D100"):
            v = np.full(n, np.nan)
            for u in units:
                w = load_est(band, u, e, rule, "L5", n)
                m = np.isfinite(w)
                if np.any(np.isfinite(v[m])):
                    raise SystemExit("FATAL: a row scored in two units of %s %s" % (band, design))
                v[m] = w[m]
            est[(e, rule)] = v
    m = np.isfinite(est[("T0", "D5")])
    for k in est:
        if not np.array_equal(np.isfinite(est[k]), m):
            raise SystemExit("FATAL: estimators scored on different rows in %s %s" % (band, design))
    return r, est, m


def sse(a, y):
    return float(np.sum((a - y) ** 2))


res = {}
tab = ["## B. Statement values from the stored estimates", "",
       "| band | quantity | own code | readout of the batch | equal to 1e-9 |", "|---|---|---|---|---|"]
st = {s["id"]: s for s in J["statements"]}
worst = 0.0
cache = {}
for b in BANDS:
    r, est, m = pooled(b, "M1")
    y = r["L5"]
    if not np.array_equal(m, r["common_L5"].astype(bool)):
        raise SystemExit("FATAL: M1 does not score the L5 common set of band " + b)
    yy = y[m]
    q = {
        "R1": 1 - sse(est[("TX", "D100")][m], yy) / sse(est[("T0", "D100")][m], yy),
        "R2": 1 - sse(est[("TX", "D5")][m], yy) / sse(est[("T0", "D5")][m], yy),
        "R4": 1 - sse(est[("T0", "D5")][m], yy) / sse(r["D5_G"][m], yy),
    }
    r3, est3, m3 = pooled(b, "M3")
    q["R3"] = 1 - sse(est3[("TX", "D100")][m3], r3["L5"][m3]) / sse(est3[("T0", "D100")][m3], r3["L5"][m3])
    for k in ("R1", "R2", "R3", "R4"):
        ref = st["%s.%s" % (k, b)]["value"]
        d = abs(q[k] - ref)
        worst = max(worst, d)
        tab.append("| %s | %s | %+.6f | %+.6f | %s |" % (b, k, q[k], ref, d < 1e-9))
    rm = {"H": np.sqrt(sse(r["D5_H"][m], yy) / m.sum()), "G": np.sqrt(sse(r["D5_G"][m], yy) / m.sum()),
          "T0": np.sqrt(sse(est[("T0", "D5")][m], yy) / m.sum()), "TX": np.sqrt(sse(est[("TX", "D5")][m], yy) / m.sum())}
    jb = J["bands"][b]["M1"]["D5"]["estimators"]
    for k in ("H", "G", "T0", "TX"):
        d = abs(rm[k] - jb[k]["rmse_eV"])
        worst = max(worst, d)
        tab.append("| %s | RMSE %s, D5 (eV) | %.4f | %.4f | %s |" % (b, k, rm[k], jb[k]["rmse_eV"], d < 1e-9))
    share = float(np.mean(np.abs(est[("TX", "D5")][m] - yy) <= 0.1 * yy))
    d = abs(share - jb["TX"]["share_within_10pct"]["point"])
    worst = max(worst, d)
    tab.append("| %s | share of TX within 10 %% of the level, D5 | %.6f | %.6f | %s |" % (b, share, jb["TX"]["share_within_10pct"]["point"], d < 1e-9))
    cost = sse(est[("T0", "D100")][m], yy) / sse(est[("T0", "D5")][m], yy)
    res[b] = {"q": q, "share": share, "cost": cost}
    cache[b] = (r["shot"][m], yy, {k: v[m] for k, v in est.items()}, r3["shot"][m3], r3["L5"][m3], {k: v[m3] for k, v in est3.items()})
inner = J["innermost_band_with_statements"]
pts = {"R5": res["E"]["cost"] - res[inner]["cost"], "R6": res["E"]["q"]["R1"] - res[inner]["q"]["R1"],
       "R7": res[inner]["share"] - res["E"]["share"]}
for k, v in pts.items():
    d = abs(v - st[k]["value"])
    worst = max(worst, d)
    tab.append("| E, %s | %s | %+.6f | %+.6f | %s |" % (inner, k, v, st[k]["value"], d < 1e-9))
print("B: largest difference %.3g" % worst)
lines += tab + ["", "Largest absolute difference: %.3g." % worst, ""]

# ---------------------------------------------------------------- C
def per_session(shots, cols):
    sess = np.array([str(session_of(s)) for s in np.unique(shots)])
    ush = np.unique(shots)
    smap = dict(zip(ush.tolist(), sess.tolist()))
    lab = np.array([smap[int(s)] for s in shots])
    names, inv = np.unique(lab, return_inverse=True)
    return names, np.stack([np.bincount(inv, weights=c, minlength=len(names)) for c in cols], axis=1)


def boot(stat, tables, n=4000, seed=8602):
    """tables: dict name -> (session names, sums per session); sessions drawn once per draw for all tables."""
    allnames = sorted(set().union(*[set(t[0].tolist()) for t in tables.values()]))
    pos = {s: i for i, s in enumerate(allnames)}
    full = {}
    for k, (names, sums) in tables.items():
        a = np.zeros((len(allnames), sums.shape[1]))
        a[[pos[s] for s in names.tolist()]] = sums
        full[k] = a
    g = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        w = np.bincount(g.integers(0, len(allnames), len(allnames)), minlength=len(allnames)).astype(float)
        out[i] = stat({k: w @ a for k, a in full.items()})
    return np.percentile(out, [2.5, 97.5])


ci = ["## C. Session intervals with an own bootstrap (4,000 draws, seed 8602)", "",
      "| statement | own interval | interval of the batch (20,000 draws) | largest difference of an end |", "|---|---|---|---|"]
worst_ci = 0.0
tabs = {}
for b in ("E", inner):
    sh, yy, e, sh3, y3, e3 = cache[b]
    tabs[b] = per_session(sh, [(e[("TX", "D100")] - yy) ** 2, (e[("T0", "D100")] - yy) ** 2, (e[("T0", "D5")] - yy) ** 2,
                               (np.abs(e[("TX", "D5")] - yy) <= 0.1 * yy).astype(float), np.ones(len(yy))])
    tabs[b + "3"] = per_session(sh3, [(e3[("TX", "D100")] - y3) ** 2, (e3[("T0", "D100")] - y3) ** 2])
jobs = {
    "R1.E": (lambda s: 1 - s["E"][0] / s["E"][1], {"E": tabs["E"]}),
    "R3.E": (lambda s: 1 - s["E3"][0] / s["E3"][1], {"E3": tabs["E3"]}),
    "R5": (lambda s: s["E"][1] / s["E"][2] - s[inner][1] / s[inner][2], {"E": tabs["E"], inner: tabs[inner]}),
    "R7": (lambda s: s[inner][3] / s[inner][4] - s["E"][3] / s["E"][4], {"E": tabs["E"], inner: tabs[inner]}),
}
for k, (f, t) in jobs.items():
    lo, hi = boot(f, t)
    ref = st[k]["session_interval"]
    d = max(abs(lo - ref[0]), abs(hi - ref[1]))
    worst_ci = max(worst_ci, d)
    ci.append("| %s | [%+.3f, %+.3f] | [%+.3f, %+.3f] | %.3f |" % (k, lo, hi, ref[0], ref[1], d))
print("C: largest difference of an interval end %.3f" % worst_ci)
lines += ci + [""]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written", OUT.relative_to(ROOT).as_posix())
sys.exit(0 if (sum(bad.values()) == 0 and worst < 1e-9) else 1)
