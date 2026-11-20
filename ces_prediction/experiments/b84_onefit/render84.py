"""B.84: docs/paper/b84_onefit.md from the readout dictionary (readout84.py).  No number is computed here."""
from __future__ import annotations

DES = ("V1", "V3", "V4a")
RULES = ("D10", "D50", "D100", "D300", "D650", "D1000", "S", "K10")
EST = ("T0", "TX", "T0a", "TXa", "T0all", "TXall", "T0raw", "TXraw", "TXrel")
SKILLS = ("TXall over TX", "T0all over T0", "TXraw over TX", "T0raw over T0", "TXall over T0all", "TX over T0",
          "TXraw over TXall", "T0raw over T0all", "TXraw over T0all", "TXall over TXa", "T0all over T0a", "TXraw over TXa",
          "TXrel over TX", "TXrel over T0", "TXrel over TXall")
RLAB = {"D10": "D10", "D50": "D50", "D100": "D100", "D300": "D300", "D650": "D650", "D1000": "D1000",
        "S": "S (schedule)", "K10": "K10 (not in the mixture)"}


def iv(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def sk(v):
    return f"{v['point']:+.3f} c {iv(v['cluster'])} d {iv(v['discharge'])}"


def render(res):
    L = ["# B.84 — one fit for every scenario, raw windows, and the scalars without absolute levels", "",
         f"Registration `{res['registration']}`; implementation choices `{res['addendum']}`. Target L5. "
         f"Mixture: {', '.join(res['mixture_rules'])}; K10 is a scenario the mixture learners were not fitted on. "
         f"Intervals: 95 %, {res['n_boot']:,} draws, shot clusters (c) and discharges (d) resampled, seed {res['seed']}; "
         "skill of A over R = 1 - MSE(A)/MSE(R) on identical rows. Verdicts on the cluster interval. "
         "Checks: `docs/paper/b84_checks.md` (all passed before this readout)." , ""]
    L += ["Estimators: T0, TX = fitted per rule (B.75; under K10, B.78); T0a, TXa = fitted with the previous frame visible "
          "and applied (B.75; under S the fresh-history fit on the S set); T0all, TXall = T0's and TX's inputs fitted once on "
          "the mixture; T0raw, TXraw = raw windows fitted once on the mixture; TXrel = TX without the four absolute levels, "
          "fitted per rule (D650, S).", ""]
    # statements
    L += ["## Statements", "", "| Id | Statement | Point | Cluster interval | Discharge interval | Verdict | Expectation |",
          "|---|---|---|---|---|---|---|"]
    for k, s in res["statements"].items():
        v = s["value"]
        L.append(f"| {k} | {s['text']} | {v['point']:+.3f} | {iv(v['cluster'])} | {iv(v['discharge'])} | "
                 f"{'holds' if s['holds'] else 'does not hold'} | {s['expected']} |")
    L += ["", "Rule of judgement: 'lies above zero' = lower end of the cluster interval above zero; 'not worse by more "
          "than 5 %' = lower end above -0.05.", ""]
    # RMSE tables
    for d in DES:
        cd = res["cells"][d]
        L += [f"## {d}: {res['designs'][d]['label']}", "", "### RMSE (eV)", "",
              "| Rule | Rows | Discharges | Clusters | " + " | ".join(EST) + " |", "|---|---|---|---|" + "---|" * len(EST)]
        for r in RULES:
            c = cd[r]
            cells = [f"{c['rmse_eV'][e]:.1f}" if e in c["rmse_eV"] else "-" for e in EST]
            L.append(f"| {RLAB[r]} | {c['counts']['rows']:,} | {c['counts']['discharges']} | {c['counts']['clusters']} | "
                     + " | ".join(cells) + " |")
        L += ["", "### Skills (point, cluster interval c, discharge interval d)", ""]
        for r in RULES:
            c = cd[r]
            L.append(f"**{RLAB[r]}**")
            L.append("")
            L += ["| Comparison | Point | Cluster interval | Discharge interval |", "|---|---|---|---|"]
            for nm in SKILLS:
                if nm in c["skills"]:
                    v = c["skills"][nm]
                    L.append(f"| {nm} | {v['point']:+.3f} | {iv(v['cluster'])} | {iv(v['discharge'])} |")
            L.append("")
        L += ["### Mean absolute error (eV) and share within 10 % of the level", "",
              "| Rule | Estimator | MAE | MAE cluster interval | Share within 10 % | Share cluster interval |",
              "|---|---|---|---|---|---|"]
        for r in ("D10", "D650", "S"):
            for e in ("TX", "TXall", "TXraw", "T0", "T0all", "TXrel"):
                if e in cd[r]["estimators"]:
                    m = cd[r]["estimators"][e]
                    a, w = m["MAE"], m["share within 10 %"]
                    L.append(f"| {r} | {e} | {a['point']:.1f} | [{a['cluster'][0]:.1f}, {a['cluster'][1]:.1f}] | "
                             f"{w['point']:.3f} | [{w['cluster'][0]:.3f}, {w['cluster'][1]:.3f}] |")
        L.append("")
    # increment along the rules
    L += ["## TXall over T0all along the rules (the fast-channel increment inside the one fit)", "",
          "| Rule | " + " | ".join(DES) + " |", "|---|" + "---|" * len(DES)]
    for r in RULES:
        L.append(f"| {RLAB[r]} | " + " | ".join(sk(res["cells"][d][r]["skills"]["TXall over T0all"]) for d in DES) + " |")
    L += ["", "## Stacked fit rows per rule", "", "| Unit | Learner | " + " | ".join(res["mixture_rules"]) + " | total | iterations |",
          "|---|---|" + "---|" * (len(res["mixture_rules"]) + 2)]
    for un, v in res["fit_rows"].items():
        for Ln in ("T0all", "TXall", "T0raw", "TXraw"):
            f = v[Ln]
            L.append(f"| {un} | {Ln} | " + " | ".join(f"{f['stacked_rows_per_rule'][r]:,}" for r in res["mixture_rules"])
                     + f" | {f['stacked_rows']:,} | {f['n_iter']} |")
    L += ["", "TXrel fit rows: " + "; ".join(f"{un} D650 {v['TXrel|D650']['fit_rows']:,}, S {v['TXrel|S']['fit_rows']:,}"
                                             for un, v in res["fit_rows"].items()), ""]
    L += [f"Readout time {res['runtime_readout_s']} s.", ""]
    return "\n".join(L)
