"""B.86: docs/paper/b86_mast_radius.md from the readout dictionary (readout86.py).  No number is computed here."""
from __future__ import annotations

BANDS = ("E", "B2", "B3", "B4", "B5")
DLAB = {"M1": "M1 random (re-partition 1, five folds of whole shots)", "M3": "M3 rolling origin (fifths 2-5 pooled)"}
ELAB = {"H": "H", "G": "G", "T0": "T0", "TX": "TX", "O": "O", "PCHIP": "PCHIP", "OGP": "offline GP"}


def iv(v, signed=True, f=".3f"):
    s = "+" if signed else ""
    return f"[{v[0]:{s}{f}}, {v[1]:{s}{f}}]"


def val(v, signed=True):
    s = "+" if signed else ""
    return f"{v['point']:{s}.3f} S {iv(v['session'], signed)}"


def render(o):
    C = o["counts"]
    L = ["# B.86 — MAST from the core to the edge: the same estimator at every radius", "",
         f"Registered: `{o['registration']}`; implementation choices in `{o['addendum']}`, written before the row build, the "
         "checks and any readout. CPU only; public FAIR-MAST files on disk; T_i of the charge-exchange chords in five bands by "
         "the distance d of the chord inside the edge of the Thomson density profile. Exploratory on used data (registration "
         "section 1). No statement here is transferred to the KSTAR core.", "",
         "Conventions. Skill of A over R = 1 - SSE(A) / SSE(R) on identical rows. Each value: point estimate, then the 95 % "
         f"interval with sessions (calendar days) resampled (S), {o['n_boot']:,} draws; the interval with shots resampled is "
         "in the statement table and in the JSON. Verdicts on the session interval. M1 is re-partition 1 (one split, five "
         "folds of whole shots); M3 pools the scored fifths 2 to 5. Each band is fitted and scored on its rows complete under "
         "D5 and D100. RMSE, MAE and median absolute error in eV. Target L5 (median of the good values of the next five "
         "frames) unless F is named. Checks: `docs/paper/b86_checks.md` (all four passed before this readout).", ""]
    # counts
    L += ["## Counts per band (L5 common set)", "",
          "| band | d (cm) | candidate rows | rows | shots | sessions | median rho | median T_i (eV) | median d (cm) | rows with rho < 0 | statements |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in BANDS:
        c = C[b]
        x = c["L5_common"]
        L.append(f"| {b} | {c['distance_cm'][0]:g} to {c['distance_cm'][1]:g} | {c['candidate_rows']:,} | {x['rows']:,} | "
                 f"{x['shots']:,} | {x['sessions']} | {x['median_rho']:.3f} | {x['median_Ti_eV']:.0f} | {x['median_dband_cm']:.1f} | "
                 f"{x['share_rho_below_0']:.3f} | {'yes' if c['statements'] else 'no (fewer than 5,000 rows)'} |")
    L += ["", f"Innermost band with statements: {o['innermost_band_with_statements']}.", ""]
    # statements
    L += ["## Statements (target L5; judged on the lower end of the session interval)", "",
          "| id | statement | value | session interval | shot interval | verdict | registered expectation | expectation | note |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in o["statements"]:
        L.append(f"| {s['id']} | {s['text']} | {s['value']:+.3f} | {iv(s['session_interval'])} | {iv(s['shot_interval'])} | "
                 f"**{s['verdict']}** | {s['expectation']} | {s['expectation_met']} | {s['note']} |")
    L += [""]
    # the table of the paper
    L += ["## MAST from the core to the edge (summary per band)", "",
          "Error with the previous frame visible (M1, D5): RMSE of TX and its share within 10 % of the level; cost of "
          "staleness of T0 at 100 ms (M1); the Thomson increment (TX over T0) under the two rules and the two designs.", "",
          "| band | RMSE TX, D5 | TX within 10 %, D5 | cost of staleness T0, D100 | TX over T0, M1 D5 | TX over T0, M1 D100 | TX over T0, M3 D5 | TX over T0, M3 D100 |",
          "|---|---|---|---|---|---|---|---|"]
    for b in BANDS:
        t = o["bands"][b]
        m1, m3 = t["M1"], t["M3"]
        L.append(f"| {b} | {m1['D5']['estimators']['TX']['rmse_eV']:.1f} | {val(m1['D5']['estimators']['TX']['share_within_10pct'], False)} | "
                 f"{val(m1['cost of staleness']['T0'], False)} | {val(m1['D5']['A over R']['TX over T0'])} | "
                 f"{val(m1['D100']['A over R']['TX over T0'])} | {val(m3['D5']['A over R']['TX over T0'])} | "
                 f"{val(m3['D100']['A over R']['TX over T0'])} |")
    L += [""]
    # per band and design
    L += ["## Per band, design and rule", "",
          "H hold of the newest visible value; G causal GP of the last up to 16 visible values; T0 G + boosted correction on "
          "the history scalars and rho; TX with the nine Thomson scalars added; O the frame's own reading (against L5).", ""]
    for b in BANDS:
        for d in ("M1", "M3"):
            t = o["bands"][b][d]
            c = t["counts"]
            L += [f"### Band {b}, {DLAB[d]} ({c['rows']:,} rows / {c['shots']:,} shots / {c['sessions']} sessions)", "",
                  "| rule | estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in ("D5", "D100"):
                for e, m in t[r]["estimators"].items():
                    L.append(f"| {r} | {ELAB[e]} | {m['rmse_eV']:.1f} | {m['MAE_eV']:.1f} | {m['median_abs_error_eV']:.1f} | "
                             f"{m['median_rel_error']:.3f} | {val(m['share_within_10pct'], False)} | {val(m['share_within_20pct'], False)} |")
            L += ["", "| skill | D5 | D100 |", "|---|---|---|"]
            for k in t["D5"]["A over R"]:
                L.append(f"| {k} | {val(t['D5']['A over R'][k])} | {val(t['D100']['A over R'][k])} |")
            L += ["", "| quantity | value |", "|---|---|"]
            for k, v in t["cost of staleness"].items():
                L.append(f"| cost of staleness {k} (SSE D100 / SSE D5) | {val(v, False)} |")
            L.append(f"| recovered share at D100 | {val(t['recovered share D100'])} |")
            L.append(f"| TX over T0, D100 minus D5 | {val(t['TX over T0, D100 minus D5'])} |")
            L.append("")
    # value of the frame
    L += ["## The value of the frame (target F, D5, M1)", "",
          "PCHIP and the offline GP wait for the next good value of the chord (frame t left out); they are scored only on rows "
          "where a good value exists before and after frame t in the block. T0 and TX are fitted to F.", ""]
    for b in BANDS:
        v = o["value_of_frame"][b]
        c = v["counts"]
        L += [f"### Band {b} ({c['rows']:,} rows / {c['shots']:,} shots / {c['sessions']} sessions; left out without a "
              f"reference {c['rows_left_out_no_reference']:,}; offline GP fell back to hold on {c['offline_gp_fit_failed_fell_back_to_hold']:,})", "",
              "| estimator | RMSE | MAE | median abs. error | median rel. error | within 10 % | within 20 % |", "|---|---|---|---|---|---|---|"]
        for e, m in v["estimators"].items():
            L.append(f"| {ELAB[e]} | {m['rmse_eV']:.1f} | {m['MAE_eV']:.1f} | {m['median_abs_error_eV']:.1f} | {m['median_rel_error']:.3f} | "
                     f"{val(m['share_within_10pct'], False)} | {val(m['share_within_20pct'], False)} |")
        L += ["", "| skill | value |", "|---|---|"]
        for k, x in v["A over R"].items():
            L.append(f"| {k.replace('OGP', 'offline GP')} | {val(x)} |")
        L.append("")
    L += [f"Readout time {o['runtime_readout_s']} s.", ""]
    return "\n".join(L)
