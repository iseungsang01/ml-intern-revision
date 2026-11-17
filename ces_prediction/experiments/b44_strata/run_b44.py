"""B.44 (PREREGISTRATION_B44.md): in which discharges the margins hold -- strata of the test discharges.

Per split, the test discharges of the frozen B.30 row table (T_i, cut population, natural cadence k = 1,
rows with hold, causal GP, history-only twin, causal variant and the measurement finite) are divided into
thirds by five quantities computed from the 10 ms file of each discharge:
    1 median measured T_i
    2 robust white component over the median T_i (1.4826 MAD of the second difference / sqrt(6) / median)
    3 share of rows with at least one BES channel more than +3 SD above its 100 ms mean (B.32)
    4 absent frames (beam blips) per second of measured time
    5 shot number
Inside each third: skill of the causal variant over the twin, the causal GP and hold (95 % intervals,
2,000 resamples of the discharges of that third) and the highest-minus-lowest difference.

CPU only; nothing is trained.  Run from the repo root:
    py ces_prediction/experiments/b44_strata/run_b44.py
Output: docs/paper/b44_strata.json / .md
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b30_thinned"))
sys.path.insert(0, str(HERE.parent / "b32_two_sources"))
import run_b30 as R  # noqa: E402
from posthoc_b32 import channel_departures  # noqa: E402

OUT_JSON = R.REPO / "docs" / "paper" / "b44_strata.json"
OUT_MD = R.REPO / "docs" / "paper" / "b44_strata.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B44.md"
SEED, N_BOOT = 20261118, 2000
MAD_C = 1.4826
MIN_TRIPLES = 30          # fewest triples for the white component of one discharge (fixed before the run)
SEG_BREAK_S = 0.5
EST = ("hold", "gp", "hist80", "causal")
REFS = ("hist80", "gp", "hold")
LABEL = {"hist80": "twin", "gp": "causal GP", "hold": "hold"}
STRATA = ("median_ti", "white_rel", "bes_up_share", "absent_per_s", "shot")
SLAB = {"median_ti": "median measured T_i (eV)",
        "white_rel": "robust white component / median T_i",
        "bes_up_share": "share of rows with a BES channel > +3 SD above its 100 ms mean",
        "absent_per_s": "absent frames per second of measured time",
        "shot": "shot number"}
THIRDS = ("low", "middle", "high")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def boot_w(rng, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


# ------------------------------------------------------------------------------ per-discharge quantities
def discharge_quantities(grid, names, nb):
    out = {}
    checks = {"bes_rows_mismatch": 0, "bes_time_maxdiff_s": 0.0}
    for i, nm in enumerate(names):
        arr = grid[nm][1:-1]                       # first and last row are not grid rows
        t = arr[:, 0].astype(np.float64)
        y = arr[:, 1].astype(np.float64)           # held values and values above 3 keV are NaN here
        q = {"shot": float(re.search(r"(\d+)", nm).group(1))}
        fin = np.isfinite(y)
        q["n_measured"] = int(fin.sum())
        med = float(np.median(y[fin])) if fin.any() else np.nan
        q["median_ti"] = med
        # robust white component (B.40 triples: three consecutive measured rows, both steps 10 ms)
        dt = np.diff(t)
        if len(y) >= 3:
            m = (fin[:-2] & fin[1:-1] & fin[2:] & (np.abs(dt[:-1] - 0.010) < 1e-4) & (np.abs(dt[1:] - 0.010) < 1e-4))
            c = np.flatnonzero(m) + 1
            sd = y[c + 1] - 2 * y[c] + y[c - 1]
        else:
            sd = np.zeros(0)
        q["triples"] = int(len(sd))
        if len(sd) >= MIN_TRIPLES and np.isfinite(med) and med > 0:
            q["white_eV"] = float(MAD_C * np.median(np.abs(sd - np.median(sd))) / np.sqrt(6.0))
            q["white_rel"] = q["white_eV"] / med
        else:
            q["white_eV"] = q["white_rel"] = np.nan
        # absent frames per second of measured time
        blip = (dt > 0.015) & (dt < 0.045)
        q["absent_frames"] = int(np.rint(dt[blip] / 0.010).sum() - blip.sum())
        q["blip_steps"] = int(blip.sum())
        q["measured_time_s"] = float(dt[dt < SEG_BREAK_S].sum())
        q["absent_per_s"] = q["absent_frames"] / q["measured_time_s"] if q["measured_time_s"] > 0 else np.nan
        # BES departures (B.32 definition, read from the CSV by posthoc_b32.channel_departures)
        tc, d, after = channel_departures(R.DATA / nm)
        if len(tc) != len(t):
            checks["bes_rows_mismatch"] += 1
        else:
            checks["bes_time_maxdiff_s"] = max(checks["bes_time_maxdiff_s"], float(np.max(np.abs(tc - t))))
        ok = np.isfinite(d[:, 0]) & (after == 0)
        q["bes_rows"] = int(ok.sum())
        q["bes_up_share"] = float((d[ok] > 3).any(1).mean()) if ok.any() else np.nan
        out[i] = q
        if (i + 1) % 100 == 0:
            log(f"quantities {i + 1}/{len(names)}")
    return out, checks


# ------------------------------------------------------------------------------ strata
def split_thirds(vals, shots):
    """Rank-based thirds: sort by (value, shot); np.array_split gives sizes differing by at most one
    (the larger thirds first)."""
    o = np.lexsort((shots, vals))
    return np.array_split(o, 3)


def skill_cells(sse, idx, rng):
    """sse: dict est -> per-discharge SSE (n,); idx: discharge positions of the cell."""
    W = boot_w(rng, len(idx))
    out, boots = {}, {}
    for ref in REFS:
        a, b = sse["causal"][idx], sse[ref][idx]
        bs = 1 - (W @ a) / (W @ b)
        out[ref] = [float(1 - a.sum() / b.sum()), *ci(bs)]
        boots[ref] = bs
    return out, boots


def run_split(s, Q, names, rng):
    z = np.load(R.ROWS_DIR / f"rows_cut_s{s}.npz")
    tab = {c.split("__")[1]: z[c] for c in z.files if c.startswith("CES_TI__")}
    ok = (tab["k"] == 1) & np.all([np.isfinite(tab[e]) for e in (*EST, "y")], axis=0)
    uf, f = np.unique(tab["file"][ok].astype(int), return_inverse=True)
    n = len(uf)
    y = tab["y"][ok].astype(np.float64)
    sse = {e: np.bincount(f, (tab[e][ok].astype(np.float64) - y) ** 2, n) for e in EST}
    rows = np.bincount(f, minlength=n)
    manifest = json.loads((R.DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json").read_text(encoding="utf-8"))
    test = set(manifest["test_files"])
    not_test = [names[i] for i in uf if names[i] not in test]
    if not_test:
        raise SystemExit(f"FATAL: split {s}: row-table files outside the test set: {not_test[:3]}")
    res = {"rows": int(ok.sum()), "discharges": n, "test_files_in_manifest": len(test)}
    res["overall"], _ = skill_cells(sse, np.arange(n), rng)
    shots = np.array([Q[i]["shot"] for i in uf])
    res["strata"] = {}
    for st in STRATA:
        v = np.array([Q[i][st] for i in uf], float)
        have = np.flatnonzero(np.isfinite(v))
        parts = [have[p] for p in split_thirds(v[have], shots[have])]
        o = {"excluded_discharges": [names[uf[j]] for j in np.flatnonzero(~np.isfinite(v))], "thirds": {}}
        boots = {}
        for tn, idx in zip(THIRDS, parts):
            sk, bs = skill_cells(sse, idx, rng)
            boots[tn] = bs
            o["thirds"][tn] = {"discharges": int(len(idx)), "rows": int(rows[idx].sum()),
                               "value_range": [float(v[idx].min()), float(v[idx].max())],
                               "value_median": float(np.median(v[idx])),
                               "skill_causal_vs": sk}
        o["edges"] = [o["thirds"]["low"]["value_range"][1], o["thirds"]["middle"]["value_range"][1]]
        # ties that straddle an edge: discharges with the edge value on both sides of it (assigned by shot number)
        lab = np.full(n, -1)
        for k, idx in enumerate(parts):
            lab[idx] = k
        o["ties_across_edges"] = []
        for k, e in enumerate(o["edges"]):
            same = np.flatnonzero(np.isfinite(v) & (v == e))
            sides = set(lab[same].tolist())
            o["ties_across_edges"].append({"edge": float(e), "discharges_with_edge_value": int(len(same)),
                                           "straddles": bool(len(sides) > 1)})
        if st == "bes_up_share":
            # after the fact: discharges with no upward departure against those with at least one
            zero = np.flatnonzero(np.isfinite(v) & (v == 0))
            pos = np.flatnonzero(np.isfinite(v) & (v > 0))
            rng_af = np.random.default_rng(SEED + 100 + s)
            sz, bz = skill_cells(sse, zero, rng_af)
            sp_, bp = skill_cells(sse, pos, rng_af)
            o["after_fact_zero_vs_positive"] = {
                "zero": {"discharges": int(len(zero)), "rows": int(rows[zero].sum()), "skill_causal_vs": sz},
                "positive": {"discharges": int(len(pos)), "rows": int(rows[pos].sum()), "skill_causal_vs": sp_},
                "positive_minus_zero": {ref: [float(sp_[ref][0] - sz[ref][0]), *ci(bp[ref] - bz[ref])] for ref in REFS}}
        o["high_minus_low"] = {}
        for ref in REFS:
            p = o["thirds"]["high"]["skill_causal_vs"][ref][0] - o["thirds"]["low"]["skill_causal_vs"][ref][0]
            iv = ci(boots["high"][ref] - boots["low"][ref])
            o["high_minus_low"][ref] = [float(p), *iv]
        o["all_thirds_above_zero"] = {ref: bool(all(o["thirds"][tn]["skill_causal_vs"][ref][1] > 0 for tn in THIRDS))
                                      for ref in REFS}
        # after the fact: rank correlation of this quantity with the median T_i over the discharges of the split
        mt = np.array([Q[i]["median_ti"] for i in uf], float)
        g = np.isfinite(v) & np.isfinite(mt)
        o["after_fact_spearman_with_median_ti"] = float(np.corrcoef(np.argsort(np.argsort(v[g])),
                                                                    np.argsort(np.argsort(mt[g])))[0, 1])
        res["strata"][st] = o
    res["W1_holds"] = res["strata"]["median_ti"]["all_thirds_above_zero"]["hist80"]
    res["W2_holds"] = res["strata"]["median_ti"]["all_thirds_above_zero"]["gp"]
    return res, tab


def precheck_b30(tabs):
    """Skill on the B.30 population (has_future, PCHIP and every network finite, k = 1) against the stored json."""
    ref = json.loads((R.REPO / "docs" / "paper" / "b30_thinned.json").read_text(encoding="utf-8"))
    out = {}
    for s, tab in tabs.items():
        ests = ["hold", "gp", "pchip", *R.NETS["cut"]]
        fin = tab["has_future"] & np.all([np.isfinite(tab[e]) for e in ests], axis=0) & (tab["k"] == 1)
        y = tab["y"][fin].astype(np.float64)
        se = {e: ((tab[e][fin].astype(np.float64) - y) ** 2).sum() for e in ("hold", "gp", "hist80", "causal")}
        c = ref["cells"][f"cut|s{s}|CES_TI"]["by_k"]["1"]
        out[str(s)] = {"rows_mine": int(fin.sum()), "rows_b30": c["rows"],
                       "causal_vs_gp_mine": float(1 - se["causal"] / se["gp"]),
                       "causal_vs_gp_b30": c["skill"]["causal_vs_gp"][0] if isinstance(c["skill"]["causal_vs_gp"], list)
                       else c["skill"]["causal_vs_gp"],
                       "causal_vs_hold_mine": float(1 - se["causal"] / se["hold"]),
                       "causal_vs_hold_b30": c["skill"]["causal_vs_hold"][0] if isinstance(c["skill"]["causal_vs_hold"], list)
                       else c["skill"]["causal_vs_hold"],
                       "causal_vs_twin_mine": float(1 - se["causal"] / se["hist80"]),
                       "causal_vs_twin_from_b30_rmse": float(1 - (c["rmse"]["causal"] / c["rmse"]["hist80"]) ** 2)}
    return out


# ------------------------------------------------------------------------------ expectations
def expectations(res):
    S = [str(s) for s in R.SEEDS]
    sp = res["splits"]
    mt = {s: sp[s]["strata"]["median_ti"] for s in S}
    E = []
    E.append(("W1 does not hold", f"W1 {'holds' if res['verdicts']['W1']['holds'] else 'does not hold'} "
              f"({res['verdicts']['W1']['splits']} of 4)", not res["verdicts"]["W1"]["holds"]))
    low = [mt[s]["thirds"]["low"]["skill_causal_vs"]["hist80"] for s in S]
    nz = sum(x[1] <= 0 for x in low)
    E.append(("lowest third of median T_i: increment over the twin not separated from zero on two or more splits",
              f"{nz} of 4 splits", nz >= 2))
    E.append(("lowest third: increment +0.00 to +0.06 (each split)", ", ".join(f"{x[0]:+.4f}" for x in low),
              all(0.0 <= x[0] <= 0.06 for x in low)))
    hi = [mt[s]["thirds"]["high"]["skill_causal_vs"]["hist80"] for s in S]
    E.append(("highest third: increment +0.05 to +0.15 (each split)", ", ".join(f"{x[0]:+.4f}" for x in hi),
              all(0.05 <= x[0] <= 0.15 for x in hi)))
    w2 = res["verdicts"]["W2"]["splits"]
    E.append(("W2 holds on two or three splits", f"{w2} of 4 splits", w2 in (2, 3)))
    bd = [sp[s]["strata"]["bes_up_share"]["high_minus_low"]["hist80"] for s in S]
    npos = sum(x[0] > 0 for x in bd)
    nex = sum(x[1] > 0 for x in bd)
    E.append(("increment over the twin grows with the BES upward-departure share (high minus low > 0 on at least "
              "three splits)", f"points {', '.join(f'{x[0]:+.3f}' for x in bd)}; interval above 0 on {nex} of 4",
              npos >= 3))
    sd = [sp[s]["strata"]["shot"]["high_minus_low"]["hist80"] for s in S]
    ninc = sum(x[1] <= 0 <= x[2] for x in sd)
    E.append(("increment over the twin does not depend on the shot number (high minus low interval contains 0 on at "
              "least three splits)", f"{ninc} of 4 contain 0; points {', '.join(f'{x[0]:+.3f}' for x in sd)}", ninc >= 3))
    return [{"expectation": a, "observed": b, "met": bool(c)} for a, b, c in E]


# ------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def fmtv(st, x):
    if st == "median_ti":
        return f"{x:.0f}"
    if st == "shot":
        return f"{x:.0f}"
    if st == "absent_per_s":
        return f"{x:.2f}"
    return f"{x:.3f}"


CHOICES = [
    "Population: rows of the frozen table with k = 1 and hold, causal GP, twin (hist80), causal variant and the "
    "measurement finite (as B.41); the discharges of a split are the test discharges that have at least one such row.",
    "All five quantities come from the cut-population grid (seq_data.load_grid_files with held values and T_i above "
    "3 keV set to NaN) with the first and last row of the file dropped; the BES departures are read by "
    "posthoc_b32.channel_departures from the same CSV (first and last row dropped there too).",
    "Median T_i: median over the measured (finite) T_i rows of the discharge.",
    f"White component: B.40 triples (three consecutive rows with T_i measured and both steps 10 ms within 0.1 ms); "
    f"sigma = 1.4826 MAD(second difference) / sqrt(6), divided by the median T_i. A discharge with fewer than "
    f"{MIN_TRIPLES} triples has no value and is left out of this stratum only (fixed before the run; listed below).",
    "BES share: rows with a finite departure (ten earlier rows in the same segment) and not within ten rows after an "
    "absent frame (the denominator of B.32 B'); a row counts when at least one of the nine channels departs by more "
    "than +3 (per-discharge standard deviations) from the mean of the ten rows before it.",
    "Absent frames: a step of more than 15 and less than 45 ms between grid rows holds round(step / 10 ms) - 1 absent "
    "frames; measured time = sum of the grid steps shorter than 0.5 s.",
    "Thirds: rank-based within each split over the discharges with a value, sorted by (value, shot number) so ties go "
    "by shot number; sizes differ by at most one (np.array_split, larger thirds first). Edges = the largest value of "
    "the low and of the middle third.",
    "Skill = 1 - SSE(causal variant) / SSE(reference) over the rows of the cell. Intervals: 2,000 multinomial "
    "resamples of the discharges of that third; the highest-minus-lowest interval is the 2.5 and 97.5 % quantiles of "
    "the difference of independent resamples of the two thirds. One generator (seed 20261118) in a fixed order.",
    "W1 (W2) holds on a split when the lower ends of all three intervals of the median-T_i thirds against the twin "
    "(causal GP) are above zero; it holds when that is so on at least three of the four splits.",
    "Section 5 rule: a stratum is reported as a condition on which a margin depends when its highest-minus-lowest "
    "interval excludes zero on at least three splits, read for each of the three references.",
]


def render(res):
    S = [str(s) for s in R.SEEDS]
    sp = res["splits"]
    hs = lambda h: "holds" if h else "does not hold"                                   # noqa: E731
    v = res["verdicts"]
    L = [f"# B.44 -- in which discharges the margins hold: strata of the test discharges (registered: {REG})", "",
         "CPU, nothing trained. T_i, cut population, natural cadence (k = 1), frozen B.30 row tables. Skill of the causal "
         "variant over the history-only twin, the causal GP and hold inside thirds of the test discharges of each split. "
         "Intervals: 95 %, 2,000 resamples of the discharges of the cell. A statement holds if it holds on at least three "
         "of the four splits.", "",
         "## Registered statements", ""]
    for w, ref in (("W1", "hist80"), ("W2", "gp")):
        parts = []
        for s in S:
            t = sp[s]["strata"]["median_ti"]["thirds"]
            parts.append(f"split {s}: " + "; ".join(f"{tn} {s3(t[tn]['skill_causal_vs'][ref])}" for tn in THIRDS))
        L.append(f"- **{w}** (every third of the median T_i: causal variant over the {LABEL[ref]}, interval above zero): "
                 f"{v[w]['splits']} of 4 splits -> **{hs(v[w]['holds'])}**. " + " | ".join(parts) + ".")
    L += ["", "## Section 5 rule: strata whose highest-minus-lowest difference excludes zero on at least three splits", "",
          "| stratum | vs twin | vs causal GP | vs hold |", "|---|---|---|---|"]
    for st in STRATA:
        c = res["conditions"][st]
        L.append(f"| {SLAB[st]} | " + " | ".join(
            f"{c[ref]['splits_excluding_zero']} of 4 ({'condition' if c[ref]['condition'] else 'no'})" for ref in REFS) + " |")
    L += ["", "## Check before the readout", "",
          "| split | rows (B.44 population) | discharges | skill vs twin (all thirds) | vs causal GP | vs hold |",
          "|---|---|---|---|---|---|"]
    for s in S:
        o = sp[s]
        L.append(f"| {s} | {o['rows']:,} | {o['discharges']} | " + " | ".join(s3(o["overall"][r]) for r in REFS) + " |")
    L += ["", "Same estimators on the B.30 population (has_future, PCHIP and every network finite) against "
          "docs/paper/b30_thinned.json:", "",
          "| split | rows (mine / B.30) | causal vs GP (mine / B.30) | causal vs hold (mine / B.30) | "
          "causal vs twin (mine / from B.30 RMSE) |", "|---|---|---|---|---|"]
    for s in S:
        c = res["check"]["b30"][s]
        L.append(f"| {s} | {c['rows_mine']:,} / {c['rows_b30']:,} | {c['causal_vs_gp_mine']:+.4f} / {c['causal_vs_gp_b30']:+.4f} | "
                 f"{c['causal_vs_hold_mine']:+.4f} / {c['causal_vs_hold_b30']:+.4f} | "
                 f"{c['causal_vs_twin_mine']:+.4f} / {c['causal_vs_twin_from_b30_rmse']:+.4f} |")
    bc = res["check"]["bes_csv_vs_grid"]
    L += ["", f"BES read from the CSV against the grid: files with a different row count {bc['bes_rows_mismatch']}, "
          f"largest time difference {bc['bes_time_maxdiff_s']:.2e} s. Sum of the rows of the three thirds equals the "
          f"population on every split and stratum: {res['check']['thirds_partition']}.", "",
          "## Operational choices made before the run (where the registration leaves room)", ""]
    L += [f"- {c}" for c in CHOICES]
    L += ["", "## Tables by stratum", ""]
    for st in STRATA:
        L += [f"### {SLAB[st]}", "",
              "| split | third | discharges | rows | range | vs twin | vs causal GP | vs hold |",
              "|---|---|---|---|---|---|---|---|"]
        for s in S:
            o = sp[s]["strata"][st]
            for tn in THIRDS:
                t = o["thirds"][tn]
                rg = f"{fmtv(st, t['value_range'][0])} to {fmtv(st, t['value_range'][1])}"
                L.append(f"| {s} | {tn} | {t['discharges']} | {t['rows']:,} | {rg} | "
                         + " | ".join(s3(t["skill_causal_vs"][r]) for r in REFS) + " |")
            L.append(f"| {s} | high minus low | | | edges {fmtv(st, o['edges'][0])}, {fmtv(st, o['edges'][1])} | "
                     + " | ".join(s3(o["high_minus_low"][r]) for r in REFS) + " |")
        tie = [(s, k, x) for s in S for k, x in enumerate(sp[s]["strata"][st]["ties_across_edges"]) if x["straddles"]]
        if tie:
            L += ["", "Ties across an edge (discharges with the edge value on both sides, assigned by shot number): "
                  + "; ".join(f"split {s} {'low/middle' if k == 0 else 'middle/high'} edge {fmtv(st, x['edge'])}: "
                              f"{x['discharges_with_edge_value']} discharges" for s, k, x in tie) + "."]
        ex = {s: sp[s]["strata"][st]["excluded_discharges"] for s in S}
        if any(ex.values()):
            L += ["", "Discharges without a value (left out of this stratum): "
                  + "; ".join(f"split {s}: {', '.join(x) if x else 'none'}" for s, x in ex.items()) + "."]
        L.append("")
    L += ["## After the fact (not registered; not used for any verdict)", "",
          "Rank correlation (Spearman) of each quantity with the median T_i over the test discharges of the split, to read "
          "how far the strata are separate conditions:", "",
          "| stratum | " + " | ".join(f"split {s}" for s in S) + " |", "|---|" + "---|" * len(S)]
    for st in STRATA[1:]:
        L.append(f"| {SLAB[st]} | " + " | ".join(f"{sp[s]['strata'][st]['after_fact_spearman_with_median_ti']:+.2f}" for s in S) + " |")
    L += ["", "The BES share is zero on many discharges, so its low third (and part of the middle third) is a set of "
          "zero-share discharges divided by shot number. Read once more after the fact: discharges with no upward "
          "departure against those with at least one (own generator, the registered numbers above are unchanged):", "",
          "| split | group | discharges | rows | vs twin | vs causal GP | vs hold |", "|---|---|---|---|---|---|---|"]
    for s in S:
        a = sp[s]["strata"]["bes_up_share"]["after_fact_zero_vs_positive"]
        for g in ("zero", "positive"):
            L.append(f"| {s} | {g} | {a[g]['discharges']} | {a[g]['rows']:,} | "
                     + " | ".join(s3(a[g]["skill_causal_vs"][r]) for r in REFS) + " |")
        L.append(f"| {s} | positive minus zero | | | " + " | ".join(s3(a["positive_minus_zero"][r]) for r in REFS) + " |")
    L += ["", "## Expectations of the registration (section 6)", "", "| expectation | observed | met |", "|---|---|---|"]
    for e in res["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {'yes' if e['met'] else 'no'} |")
    nm = sum(e["met"] for e in res["expectations"])
    L += ["", f"{nm} of {len(res['expectations'])} expectations met.", "", "## Deviations", "",
          "None from the registered definitions, thresholds or populations. The registration leaves the treatment of ties, "
          "of discharges without a value and the exact counting of absent frames open; the choices above were fixed before "
          "the run.", "", "## Limits", "",
          "About 32 discharges per third of a split, so the intervals are wide. The strata are computed from the same files "
          "as the scores. They are not confinement modes.", "", f"Run time: {res['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


def main():
    R.pin_env()
    _t, _B, _ls, load_grid_files, *_ = R.imports()
    grid, dims = load_grid_files(R.DATA, True, R.POPS["cut"])
    names = sorted(grid)
    log(f"grid loaded: {len(names)} files")
    Q, bes_check = discharge_quantities(grid, names, dims["bes"])
    log("quantities done")
    rng = np.random.default_rng(SEED)
    res = {"registration": REG, "seed": SEED, "n_boot": N_BOOT, "min_triples": MIN_TRIPLES, "splits": {}}
    tabs = {}
    part_ok = True
    for s in R.SEEDS:
        o, tab = run_split(s, Q, names, rng)
        tabs[s] = tab
        for st in STRATA:
            n_have = o["discharges"] - len(o["strata"][st]["excluded_discharges"])
            part_ok &= sum(o["strata"][st]["thirds"][t]["discharges"] for t in THIRDS) == n_have
        res["splits"][str(s)] = o
        log(f"split {s}: W1 {o['W1_holds']} W2 {o['W2_holds']}")
    # rows of the thirds sum to the population when no discharge is excluded
    for s in R.SEEDS:
        o = res["splits"][str(s)]
        for st in STRATA:
            if not o["strata"][st]["excluded_discharges"]:
                part_ok &= sum(o["strata"][st]["thirds"][t]["rows"] for t in THIRDS) == o["rows"]
    res["check"] = {"b30": precheck_b30(tabs), "bes_csv_vs_grid": bes_check, "thirds_partition": bool(part_ok)}
    S = [str(s) for s in R.SEEDS]
    n1 = sum(res["splits"][s]["W1_holds"] for s in S)
    n2 = sum(res["splits"][s]["W2_holds"] for s in S)
    res["verdicts"] = {"W1": {"splits": int(n1), "holds": bool(n1 >= 3)}, "W2": {"splits": int(n2), "holds": bool(n2 >= 3)}}
    res["conditions"] = {}
    for st in STRATA:
        res["conditions"][st] = {}
        for ref in REFS:
            k = sum(not (res["splits"][s]["strata"][st]["high_minus_low"][ref][1] <= 0
                         <= res["splits"][s]["strata"][st]["high_minus_low"][ref][2]) for s in S)
            res["conditions"][st][ref] = {"splits_excluding_zero": int(k), "condition": bool(k >= 3)}
    res["discharge_quantities"] = {names[i]: q for i, q in Q.items()}
    res["expectations"] = expectations(res)
    res["choices"] = CHOICES
    res["runtime_s"] = time.time() - T0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(res), encoding="utf-8", newline="\n")
    log(f"W1 {res['verdicts']['W1']} W2 {res['verdicts']['W2']}; written")


if __name__ == "__main__":
    main()
