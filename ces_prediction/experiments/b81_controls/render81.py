"""B.81: docs/paper/b81_controls.md from the readout dictionary (readout81.py).  No number is computed here."""
from __future__ import annotations

DES = ("V1", "V2", "V3", "V4a", "V4b")
DLAB = {"V1": "V1 random (r2-r4)", "V2": "V2 cluster-blocked (3 assignments)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4a chronological: fit to 31991, score from 32312", "V4b": "V4b chronological: fit to 32310, score from 32312"}
RULES = ("D650", "S")
RLAB = {"D650": "D650 (history 0.65 s old)", "S": "S (beam-modulation schedule)"}


def s3(x):
    return f"{x:+.3f}"


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def cellf(v):
    return f"{s3(v['point'])}<br>c {iv(v['cluster'])}<br>d {iv(v['discharge'])}"


def render(o):
    L = ["# B.81 — the two controls and the two halves of the fast inputs under the stale-history rules, every design", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before any run. CPU only; "
         "boosted regressions of the kind of B.75, no network. Target L5 (the median of the measured values among the next "
         "five rows). Rows, inputs, units, regression and target transform of B.75; rule D650 on B.75's common set, rule S on "
         "B.75's S set. Exploratory on used data (registration section 1).", "",
         "Estimators. T0: the ten history scalars. TX: T0's inputs, the eight fast scalars of row t and the six anchor scalars. "
         "TXp: TX's inputs with the fourteen fast and anchor columns permuted jointly over rows (within the fit rows and within "
         "the test rows) and the estimator refitted. TXo: the ten history scalars, the eight fast scalars of the anchor row (the "
         "row of the newest visible CES value) and a1 of BES and of ECEI. TE (TB): T0's inputs and the ECEI (BES) half of TX's "
         "fast inputs. T0, TX and, under V1 and V4, TE and TB are the stored estimates of B.75; the rest is fitted here.", "",
         "Conventions. Skill of A over R = 1 - MSE(A) / MSE(R) on identical rows. Each value: point; 95 % interval with shot "
         "clusters resampled (c); with discharges resampled (d); 20,000 draws (V1: a re-partition drawn first; V2: an assignment "
         "drawn first; V3, V4: the scored units). Verdicts on the cluster interval. Points under V1 and V2: mean over the three "
         "re-partitions or assignments. RMSE in eV. Checks: `docs/paper/b81_checks.md` (all passed before this readout: "
         + ", ".join(f"{k} {v}" for k, v in o["checks"].items()) + ").", ""]
    # ---- statements
    L += ["## Statements (target L5, cluster interval)", "",
          "| Id | statement | point | cluster interval | discharge interval | verdict | registered expectation | as expected | note |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in o["statements"].items():
        v = s["value"]
        verdict = "holds" if s["holds"] else "does not hold"
        met = o["expectations"][k]["verdict_as_expected"]
        met = "n/a (uncertain)" if met is None else ("yes" if met else "no")
        L.append(f"| {k} | {s['text']} | {s3(v['point'])} | {iv(v['cluster'])} | {iv(v['discharge'])} | {verdict} | "
                 f"{s['expected']} | {met} | {s['note']} |")
    L += ["", "C1 and C2 hold when the lower end of the cluster interval is at most zero; the others when it is above zero.", ""]
    # ---- controls
    L += ["## The two controls by design and rule", "",
          "TXp over T0 (fast inputs permuted over rows), TXo over T0 (fast inputs of the anchor row), TX over TXo (the present "
          "samples beside those of the anchor row), with TX over T0 (the fast-channel increment of B.75, recomputed here on "
          "this batch's draws) beside them.", ""]
    for r in RULES:
        L += [f"### {RLAB[r]}", "", "| design | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |", "|---|---|---|---|---|"]
        for d in DES:
            m = o["main"][d][r]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(m[c]) for c in ("TX/T0", "TXp/T0", "TXo/T0", "TX/TXo")) + " |")
        L.append("")
    # ---- halves
    L += ["## Which diagnostic: the ECEI half (TE) and the BES half (TB) over T0", "",
          "Under V1 and V4 from B.75's stored estimates (intervals recomputed on this batch's draws), under V2 and V3 fitted here.", ""]
    for r in RULES:
        L += [f"### {RLAB[r]}", "", "| design | TE/T0 | TB/T0 | TX/T0 |", "|---|---|---|---|"]
        for d in DES:
            m = o["main"][d][r]
            L.append(f"| {DLAB[d]} | " + " | ".join(cellf(m[c]) for c in ("TE/T0", "TB/T0", "TX/T0")) + " |")
        L.append("")
    # ---- RMSE and counts
    L += ["## RMSE (eV) and counts", "", "| design | rule | rows | discharges | clusters | T0 | TX | TXp | TXo | TE | TB |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in DES:
        for r in RULES:
            m = o["main"][d][r]
            c = m["counts"]
            L.append(f"| {d} | {r} | {c['rows']:,} | {c['discharges']} | {c['clusters']} | " +
                     " | ".join(f"{m['rmse_eV'][x]:.1f}" for x in ("T0", "TX", "TXp", "TXo", "TE", "TB")) + " |")
    L += ["", "Fit rows per unit and TXo rows whose anchor-row fast scalars are not all finite (kept, left to the regression):", "",
          "| design | rule | fit rows per unit (min-max) | TXo: such test rows (total) | TXo: such fit rows per unit (max) |",
          "|---|---|---|---|---|"]
    for d in DES:
        for r in RULES:
            c = o["counts"][d][r]
            nf = c["TXo_rows_with_nonfinite_anchor_row_scalars"]
            L.append(f"| {d} | {r} | {c['fit_rows_per_unit']['min']:,}-{c['fit_rows_per_unit']['max']:,} | {nf['test_total']:,} | "
                     f"{nf['fit_per_unit_max']:,} |")
    L.append("")
    # ---- ECEI moving
    L += ["## V1 with the rows left out where the ECEI is constant (a description, defined after the fact)", "",
          "A row is left out when the four ECEI channels have not changed for at least ten rows at row t or at the anchor row "
          "(definition of `b75_availability/lead_checks/stuck_ecei.py`). The stratum was defined after B.75 had been read; no "
          "statement uses it.", "",
          "| rule | rows kept | rows left out | discharges | clusters | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |", "|---|---|---|---|---|---|---|---|---|"]
    for r in RULES:
        m = o["ecei_moving"][r]
        c = m["counts"]
        L.append(f"| {r} | {c['rows']:,} | {m['rows_left_out']:,} | {c['discharges']} | {c['clusters']} | " +
                 " | ".join(cellf(m[x]) for x in ("TX/T0", "TXp/T0", "TXo/T0", "TX/TXo")) + " |")
    L += ["", "Beside it, V1 on all rows (from the tables above):", "", "| rule | TX/T0 | TXp/T0 | TXo/T0 | TX/TXo |", "|---|---|---|---|---|"]
    for r in RULES:
        m = o["main"]["V1"][r]
        L.append(f"| {r} | " + " | ".join(cellf(m[x]) for x in ("TX/T0", "TXp/T0", "TXo/T0", "TX/TXo")) + " |")
    L += ["", f"Readout run time {o['runtime_readout_s']} s.", ""]
    return "\n".join(L)
