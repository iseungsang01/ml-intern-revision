"""MAST gate, addendum 1 (PREREGISTRATION_MAST_GATE_ADDENDUM1.md): wider tuning grids.

CPU, numpy/scipy/pandas, deterministic, single process.  Run from the repo root:

    py ces_prediction/experiments/mast/run_gate_addendum1.py

The whole pipeline is run_gate.main(), unchanged.  This script only (1) replaces the tuning grids
that run_gate.tune() reads from its module globals and wraps tune() in the registered stopping rule
(edge -> three more log-steps, at most two further rounds), (2) redirects main()'s own output to a
temporary directory, (3) writes docs/paper/mast_gate_addendum1.json / .md with the retuned numbers
next to those of the first run (docs/paper/mast_gate.json, read, never written).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as rg  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_MAST_GATE_ADDENDUM1.md"
FIRST_JSON = rg.ROOT / "docs" / "paper" / "mast_gate.json"
OUT_JSON = rg.ROOT / "docs" / "paper" / "mast_gate_addendum1.json"
OUT_MD = rg.ROOT / "docs" / "paper" / "mast_gate_addendum1.md"

MAX_EXTRA_ROUNDS = 2
N_STEPS = 3
Q_LEVEL_STEP = 0.7
Q_SLOPE_STEP = 1.0
TAU_UP = (110.0, 180.0, 300.0)
RETUNED_ARMS = ("running_mean", "slope_filter", "causal_gp", "offline_gp")
UNCHANGED_ARMS = ("persistence", "extrapolation", "offline_interp")

ORIG_TUNE = rg.tune
FIRST_TAUS, FIRST_ELLS, FIRST_AMPS = tuple(rg.TAUS), tuple(rg.GP_ELLS), tuple(rg.GP_AMPS)
FIRST_Q_SLOPE_EXP = [float(x) for x in rg.Q_SLOPE_EXP]     # captured before any grid is replaced
ROUNDS = {}          # target -> list of tuning rounds (filled in call order: T_i, then V_i)


def initial_grids():
    """Section 3 of the addendum registration; the same starting grids for every target."""
    return {"taus": [2.5] + list(FIRST_TAUS),
            "q_level_exp": [-2.0 + Q_LEVEL_STEP * k for k in range(9)],
            "q_slope_exp": list(FIRST_Q_SLOPE_EXP),
            "ells": [10.0 / 2 ** (k / 2) for k in (4, 3, 2, 1)] + list(FIRST_ELLS),
            "amps": list(FIRST_AMPS) + [8.0, 16.0, 32.0]}


def set_grids(g):
    rg.TAUS = tuple(g["taus"])
    rg.Q_LEVEL_EXP = np.array(g["q_level_exp"])
    rg.Q_SLOPE_EXP = np.array(g["q_slope_exp"])
    rg.GP_ELLS = tuple(g["ells"])
    rg.GP_AMPS = tuple(g["amps"])


def last_step_change(mse, idx, axis):
    """Relative TUNE-MSE change of the selected point against its inward neighbour on one axis."""
    mse = np.asarray(mse, float)
    n = mse.shape[axis]
    j = list(idx)
    j[axis] = 1 if idx[axis] == 0 else n - 2
    return float(mse[tuple(idx)] / mse[tuple(j)] - 1.0)


def edge_report(tinfo):
    """Selected index, edge side (None / 'lower' / 'upper') and last-step change per hyperparameter."""
    out = {}

    def put(name, mse, idx, axis, value):
        n = np.asarray(mse).shape[axis]
        side = "lower" if idx[axis] == 0 else ("upper" if idx[axis] == n - 1 else None)
        out[name] = {"selected": value, "edge": side,
                     "last_step_relative_tune_mse_change":
                         last_step_change(mse, idx, axis) if side else None}

    m = np.asarray(tinfo["running_mean"]["tune_mse"])
    i = (int(np.argmin(m)),)
    put("tau_ms", m, i, 0, tinfo["running_mean"]["selected_tau_ms"])
    m = np.asarray(tinfo["slope_filter"]["tune_mse_rows_q_level_cols_q_slope"])
    ij = tuple(int(x) for x in np.unravel_index(int(np.argmin(m)), m.shape))
    put("q_level_over_scale2", m, ij, 0, tinfo["slope_filter"]["selected_q_level_over_scale2"])
    put("q_slope_over_scale2", m, ij, 1, tinfo["slope_filter"]["selected_q_slope_over_scale2"])
    for n in ("gp_causal", "gp_offline"):
        m = np.asarray(tinfo[n]["tune_mse_rows_length_cols_amp"])
        ij = tuple(int(x) for x in np.unravel_index(int(np.argmin(m)), m.shape))
        put(n + "_length_ms", m, ij, 0, tinfo[n]["selected_length_ms"])
        put(n + "_amp_over_scale", m, ij, 1, tinfo[n]["selected_amp_over_scale"])
    return out


def extend(g, edges):
    """Registered extension: three more steps on every edge reached (tau lower edge excepted)."""
    g = {k: list(v) for k, v in g.items()}
    done = []
    if edges["tau_ms"]["edge"] == "upper":
        for _ in range(N_STEPS):
            nxt = [t for t in TAU_UP if t > g["taus"][-1]]
            g["taus"].append(nxt[0] if nxt else g["taus"][-1] * 1.6)
        done.append("tau upper")
    for key, name, step in (("q_level_exp", "q_level_over_scale2", Q_LEVEL_STEP),
                            ("q_slope_exp", "q_slope_over_scale2", Q_SLOPE_STEP)):
        if edges[name]["edge"] == "upper":
            g[key] += [g[key][-1] + step * k for k in range(1, N_STEPS + 1)]
            done.append(name + " upper")
        elif edges[name]["edge"] == "lower":
            g[key] = [g[key][0] - step * k for k in range(N_STEPS, 0, -1)] + g[key]
            done.append(name + " lower")
    sides = {edges[n + "_amp_over_scale"]["edge"] for n in ("gp_causal", "gp_offline")}
    if "upper" in sides:
        g["amps"] += [g["amps"][-1] * 2.0 ** k for k in range(1, N_STEPS + 1)]
        done.append("GP amplitude upper")
    if "lower" in sides:
        g["amps"] = [g["amps"][0] / 2.0 ** k for k in range(N_STEPS, 0, -1)] + g["amps"]
        done.append("GP amplitude lower")
    sides = {edges[n + "_length_ms"]["edge"] for n in ("gp_causal", "gp_offline")}
    if "upper" in sides:
        g["ells"] += [g["ells"][-1] * 2.0 ** k for k in range(1, N_STEPS + 1)]
        done.append("GP length upper")
    if "lower" in sides:
        g["ells"] = [g["ells"][0] / 2.0 ** (k / 2) for k in range(N_STEPS, 0, -1)] + g["ells"]
        done.append("GP length lower")
    return g, done


def tune_with_stopping_rule(prep_tune, log):
    """Drop-in replacement of run_gate.tune: the same function on the addendum grids, repeated
    under the registered stopping rule.  Uses TUNE rows only."""
    target = rg.TARGETS[len(ROUNDS)]
    grids = initial_grids()
    rounds = []
    while True:
        for v in grids.values():
            assert all(x < y for x, y in zip(v, v[1:])), "grid not strictly increasing"
        set_grids(grids)
        log(f"  addendum tuning round {len(rounds) + 1} ({target}): "
            f"{len(grids['q_level_exp'])} x {len(grids['q_slope_exp'])} filter, "
            f"{len(grids['ells'])} x {len(grids['amps'])} GP, {len(grids['taus'])} tau")
        tinfo, params = ORIG_TUNE(prep_tune, log)
        for key in ("running_mean", "slope_filter", "gp_causal", "gp_offline"):
            for k, v in tinfo[key].items():
                if k.startswith("tune_mse") and not np.isfinite(np.asarray(v)).all():
                    raise RuntimeError(f"non-finite TUNE error in {key} ({target})")
        edges = edge_report(tinfo)
        new_grids, extended = extend(grids, edges)
        rounds.append({"grids": grids, "edges": edges, "tinfo": tinfo, "extended_after": extended})
        if not extended or len(rounds) == 1 + MAX_EXTRA_ROUNDS:
            rounds[-1]["extended_after"] = []
            break
        grids = new_grids
    # the hard-coded 6 x 6 edge flags of run_gate.tune are replaced by the actual ones
    tinfo["slope_filter"]["on_grid_edge_q_level"] = edges["q_level_over_scale2"]["edge"] is not None
    tinfo["slope_filter"]["on_grid_edge_q_slope"] = edges["q_slope_over_scale2"]["edge"] is not None
    ROUNDS[target] = rounds
    return tinfo, params


# --------------------------------------------------------------------------- report

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_edges(first, tg):
    return edge_report(first["tuning"][tg])


def expectations(first, res, final_edges):
    out = []

    def add(name, expected, found, met):
        out.append({"expectation": name, "expected": expected, "found": found, "met": bool(met)})

    def sk(tg, arm):
        return res["readouts"][f"{tg}|G"]["arms"][arm]["skill_vs_persistence"]

    for tg, lo, hi in (("T_i", 0.07, 0.14), ("V_i", 0.08, 0.15)):
        x = sk(tg, "slope_filter")
        add(f"{tg} G: slope-state filter vs persistence", f"{lo:+.2f} to {hi:+.2f}", rg.fmt(x),
            lo <= x["point"] <= hi)
    x = sk("T_i", "slope_filter")
    add("T_i G: slope-filter interval excludes 0 (above)", "excludes 0 (above)",
        rg.fmt(x) + ", " + rg.side_txt(x), x["side"] == "above")
    add("T_i G: slope-filter interval lower bound", "between 0 and +0.04", f"{x['lo']:+.3f}",
        0 < x["lo"] <= 0.04)
    x = sk("V_i", "slope_filter")
    add("V_i G: slope-filter interval excludes 0 (above)", "excludes 0 (above)",
        rg.fmt(x) + ", " + rg.side_txt(x), x["side"] == "above")
    for tg, lo, hi in (("T_i", -0.05, 0.12), ("V_i", -0.05, 0.13)):
        x = sk(tg, "causal_gp")
        add(f"{tg} G: causal GP vs persistence", f"{lo:+.2f} to {hi:+.2f}", rg.fmt(x),
            lo <= x["point"] <= hi)
        d = x["point"] - sk(tg, "slope_filter")["point"]
        add(f"{tg} G: causal-GP point minus slope-filter point", "<= +0.03", f"{d:+.3f}", d <= 0.03)
    for tg in rg.TARGETS:
        b = res["gate_b"][tg]["best_causal_arm"]
        add(f"{tg}: best causal arm", "slope_filter", b, b == "slope_filter")
    for tg, lo, hi in (("T_i", 2.2, 3.3), ("V_i", 16.0, 26.0)):
        gb = res["gate_b"][tg]
        add(f"{tg}: gate (b) re-read passes", "PASS", gb["verdict"], gb["verdict"] == "PASS")
        x = gb["chi2_mean_G"]
        add(f"{tg} G: mean reduced chi-square, best causal arm", f"{lo:g} to {hi:g}",
            rg.fmt(x, 2, False) + f" ({gb['best_causal_arm']})", lo <= x["point"] <= hi)
    for tg, lo, hi in (("T_i", 0.50, 0.62), ("V_i", 0.60, 0.70)):
        x = sk(tg, "offline_gp")
        add(f"{tg} G: offline GP vs persistence", f"{lo:+.2f} to {hi:+.2f}", rg.fmt(x),
            lo <= x["point"] <= hi)
    for tg in rg.TARGETS:
        tau = res["tuning"][tg]["running_mean"]["selected_tau_ms"]
        add(f"{tg}: running-mean tau selected", "2.5 ms", f"{tau:g} ms", tau == 2.5)
        x = sk(tg, "running_mean")
        add(f"{tg} G: running mean vs persistence", "-0.03 to 0", rg.fmt(x), -0.03 <= x["point"] <= 0)
    for tg in rg.TARGETS:
        e = final_edges[tg]
        add(f"{tg}: q_level interior after the stopping rule", "interior",
            e["q_level_over_scale2"]["edge"] or "interior", e["q_level_over_scale2"]["edge"] is None)
        amp = [e[n + "_amp_over_scale"]["edge"] for n in ("gp_causal", "gp_offline")]
        add(f"{tg}: GP amplitude edge-bound for at least one GP", "edge-bound",
            f"causal {amp[0] or 'interior'}, offline {amp[1] or 'interior'}", any(amp))
        add(f"{tg}: offline GP length interior", "interior",
            e["gp_offline_length_ms"]["edge"] or "interior", e["gp_offline_length_ms"]["edge"] is None)
    return out


def edge_txt(e):
    if e["edge"] is None:
        return "interior"
    return f"{e['edge']} edge ({100 * e['last_step_relative_tune_mse_change']:+.2f} % last step)"


def render_md(doc):
    first, res = doc["_first"], doc["retuned"]
    L = []
    A = L.append
    A("# MAST gate, addendum 1 (wider tuning grids) - results next to the first run")
    A("")
    A("Generated by `ces_prediction/experiments/mast/run_gate_addendum1.py` from the registration "
      f"`{REG}`. Single change against the first run (`docs/paper/mast_gate.json` / `.md`, the "
      "registered primary): the tuning grids. Shot list, TUNE / TEST split, populations, row sets, "
      "nominal 5 ms grid, arms, readouts and bootstrap (seed 20261116, 2,000 shot-clustered "
      "resamples of TEST shots) are the functions of `run_gate.py`, unchanged. Every number is on "
      "TEST unless marked TUNE. Entries are point [95 % interval]. T_i in eV, V_i in km/s.")
    A("")
    A("## 1. Gate (b) re-read (registered statement 1; threshold 1.5, population G)")
    A("")
    A("| target | run | best causal arm (lowest TEST MSE, G) | mean reduced chi-square G | vs 1.5 | "
      "median G | mean reduced chi-square W | vs 1.5 | G with errors rescaled | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for tg in rg.TARGETS:
        for label, r in (("first run", first), ("addendum 1", res)):
            g = r["gate_b"][tg]
            rs = g.get("chi2_mean_G_errors_rescaled")
            rs_t = (rg.fmt(rs, 2, False) + ", " + rg.side_txt(rs)) if rs else "not applicable"
            A(f"| {tg} | {label} | {g['best_causal_arm']} | {rg.fmt(g['chi2_mean_G'], 2, False)} | "
              f"{rg.side_txt(g['chi2_mean_G'])} | {rg.fmt(g['chi2_median_G'], 3, False)} | "
              f"{rg.fmt(g['chi2_mean_W'], 2, False)} | {rg.side_txt(g['chi2_mean_W'])} | {rs_t} | "
              f"**{g['verdict']}** |")
    A("")
    A("TEST MSE of the causal arms in G (the best causal arm is the lowest):")
    A("")
    A("| target | run | " + " | ".join(rg.CAUSAL_ARMS) + " |")
    A("|---|---|" + "---|" * len(rg.CAUSAL_ARMS))
    for tg in rg.TARGETS:
        for label, r in (("first run", first), ("addendum 1", res)):
            m = r["gate_b"][tg]["test_mse_causal_arms_G"]
            A(f"| {tg} | {label} | " + " | ".join(f"{m[a]:.5g}" for a in rg.CAUSAL_ARMS) + " |")
    A("")
    A("## 2. Selected hyperparameters and edge flags (TUNE scored rows, population G)")
    A("")
    A("In units of the target scale (median reported error of the TUNE rows; unchanged). 'last "
      "step' = relative change of the pooled TUNE MSE between the selected value and its inward "
      "neighbour on that axis, the other hyperparameter of the family held at its selected value.")
    A("")
    A("| target | hyperparameter | first run: selected | first run: position | addendum 1: selected | "
      "addendum 1: position | status |")
    A("|---|---|---|---|---|---|---|")
    names = (("tau_ms", "running-mean tau (ms)"), ("q_level_over_scale2", "filter q_level / scale^2"),
             ("q_slope_over_scale2", "filter q_slope / scale^2"),
             ("gp_causal_length_ms", "causal GP length (ms)"),
             ("gp_causal_amp_over_scale", "causal GP amplitude / scale"),
             ("gp_offline_length_ms", "offline GP length (ms)"),
             ("gp_offline_amp_over_scale", "offline GP amplitude / scale"))
    for tg in rg.TARGETS:
        fe, ae = doc["first_run_edges"][tg], doc["final_edges"][tg]
        for key, label in names:
            if ae[key]["edge"] is None:
                st = "interior"
            elif key == "tau_ms" and ae[key]["edge"] == "lower":
                st = "**edge-bound** (persistence limit, not extended by rule)"
            else:
                st = "**edge-bound** (after the last allowed round)"
            A(f"| {tg} | {label} | {fe[key]['selected']:.4g} | {edge_txt(fe[key])} | "
              f"{ae[key]['selected']:.4g} | {edge_txt(ae[key])} | {st} |")
    A("")
    A("Tuning rounds (stopping rule: every edge reached is extended by three log-steps, at most two "
      "further rounds):")
    A("")
    A("| target | round | q_level / scale^2 grid | q_slope / scale^2 grid | GP length grid (ms) | GP "
      "amplitude / scale grid | tau grid (ms) | selected (tau; q_level, q_slope; causal GP length, "
      "amplitude; offline GP length, amplitude) | on an edge | extended for the next round |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for tg in rg.TARGETS:
        for k, r in enumerate(doc["tuning_rounds"][tg]):
            g, e = r["grids"], r["edges"]
            sel = (f"{e['tau_ms']['selected']:g}; {e['q_level_over_scale2']['selected']:.4g}, "
                   f"{e['q_slope_over_scale2']['selected']:.4g}; "
                   f"{e['gp_causal_length_ms']['selected']:.4g}, {e['gp_causal_amp_over_scale']['selected']:g}; "
                   f"{e['gp_offline_length_ms']['selected']:.4g}, {e['gp_offline_amp_over_scale']['selected']:g}")
            on = ", ".join(f"{n} ({v['edge']})" for n, v in e.items() if v["edge"]) or "none"
            A(f"| {tg} | {k + 1} | 10^({g['q_level_exp'][0]:g} .. {g['q_level_exp'][-1]:g}), "
              f"{len(g['q_level_exp'])} values | 10^({g['q_slope_exp'][0]:g} .. {g['q_slope_exp'][-1]:g}), "
              f"{len(g['q_slope_exp'])} values | " + ", ".join(f"{x:.3g}" for x in g["ells"]) + " | "
              + ", ".join(f"{x:g}" for x in g["amps"]) + " | " + ", ".join(f"{x:g}" for x in g["taus"])
              + f" | {sel} | {on} | {', '.join(r['extended_after']) or 'none (stop)'} |")
    A("")
    A("The TUNE MSE of every grid point of every round is in the JSON (`tuning_rounds`).")
    A("")
    A("## 3. Arms: addendum 1 next to the first run")
    A("")
    for tg in rg.TARGETS:
        for pop in rg.POPS:
            k = f"{tg}|{pop}"
            r1, r2 = first["readouts"][k], res["readouts"][k]
            A(f"### {tg}, population {pop} (TEST: {r2['n_rows']} rows, {r2['n_shots']} shots; "
              f"first run {r1['n_rows']} rows, {r1['n_shots']} shots)")
            A("")
            A("| arm | RMSE first / addendum | skill vs persistence, first run | vs 0 | skill vs "
              "persistence, addendum 1 | vs 0 | mean reduced chi-square, first run | addendum 1 | vs 1 | "
              "median reduced chi-square, first run | addendum 1 | vs 1 |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for arm in rg.ARMS:
                a1, a2 = r1["arms"][arm], r2["arms"][arm]
                tag = ""
                if arm == res["gate_b"][tg]["best_causal_arm"]:
                    tag = " (best causal, addendum)"
                if arm in UNCHANGED_ARMS:
                    tag += " (no hyperparameter)"
                ref = arm == "persistence"
                A(f"| {arm}{tag} | {a1['rmse']:.4g} / {a2['rmse']:.4g} | "
                  f"{'0 (reference)' if ref else rg.fmt(a1['skill_vs_persistence'])} | "
                  f"{'-' if ref else rg.side_txt(a1['skill_vs_persistence'])} | "
                  f"{'0 (reference)' if ref else rg.fmt(a2['skill_vs_persistence'])} | "
                  f"{'-' if ref else rg.side_txt(a2['skill_vs_persistence'])} | "
                  f"{rg.fmt(a1['chi2_mean'], 2, False)} | {rg.fmt(a2['chi2_mean'], 2, False)} | "
                  f"{rg.side_txt(a2['chi2_mean'])} | {rg.fmt(a1['chi2_median'], 3, False)} | "
                  f"{rg.fmt(a2['chi2_median'], 3, False)} | {rg.side_txt(a2['chi2_median'])} |")
            A("")
    A("## 4. Regime statements, audit, sensitivity (not gating)")
    A("")
    A("| target | run | R1: slope-filter skill | R1: running-mean skill | R1 holds | R2: Spearman(regime "
      "number, slope-filter skill - running-mean skill) | vs 0 | R2 holds |")
    A("|---|---|---|---|---|---|---|---|")
    for tg in rg.TARGETS:
        for label, r in (("first run", first), ("addendum 1", res)):
            g = r["regime"][tg]
            A(f"| {tg} | {label} | {rg.fmt(g['R1']['slope_filter_skill'])} "
              f"({rg.side_txt(g['R1']['slope_filter_skill'])}) | {rg.fmt(g['R1']['running_mean_skill'])} "
              f"({rg.side_txt(g['R1']['running_mean_skill'])}) | {'yes' if g['R1']['holds'] else 'no'} | "
              f"{rg.fmt(g['R2']['spearman'])} (n = {g['R2']['n_shots']}) | "
              f"{rg.side_txt(g['R2']['spearman'])} | {'yes' if g['R2']['holds'] else 'no'} |")
    A("")
    A("Heavy tail of the best causal arm, and skill with the 0.1 % of rows with its largest squared "
      "residual dropped from every arm (mean reduced chi-square in parentheses):")
    A("")
    A("| target, population | run | best causal arm | mean / median reduced chi-square (Gaussian 2.2) | "
      "rows dropped (shots) | their share of the SSE | " + " | ".join(RETUNED_ARMS) + " |")
    A("|---|---|---|---|---|---|" + "---|" * len(RETUNED_ARMS))
    for k in res["audit"]:
        for label, r in (("first run", first), ("addendum 1", res)):
            a = r["audit"][k]
            A(f"| {k.replace('|', ', ')} | {label} | {a['best_causal_arm']} | "
              f"{rg.fmt(a['chi2_mean_over_median'], 2, False)} | {a['top_rows_dropped']} "
              f"({a['top_rows_n_shots']}) | {rg.fmt(a['top_0p1pct_share_of_sse'], 3, False)} | "
              + " | ".join(f"{rg.fmt(a['skill_with_top_rows_dropped'][arm])} "
                           f"({a['chi2_mean_with_top_rows_dropped'][arm]['point']:.2f})"
                           for arm in RETUNED_ARMS) + " |")
    A("")
    A("TEST shots with a block frame interval outside 5.0 +- 0.1 ms dropped (not registered in "
      "either registration; skill against persistence):")
    A("")
    A("| target, population | run | TEST shots dropped | rows | mean reduced chi-square, best causal "
      "arm, vs 1.5 | " + " | ".join(RETUNED_ARMS) + " |")
    A("|---|---|---|---|---|" + "---|" * len(RETUNED_ARMS))
    for tg in rg.TARGETS:
        for pop in rg.POPS:
            for label, r in (("first run", first), ("addendum 1", res)):
                v = r["sensitivity_irregular_shots_dropped"][tg][pop]
                m = v["chi2_mean_best_causal_arm_vs_1p5"]
                A(f"| {tg}, {pop} | {label} | {v['test_shots_dropped']} | {v['n_rows']} | "
                  f"{rg.fmt(m, 2, False)}, {rg.side_txt(m)} | "
                  + " | ".join(rg.fmt(v["skill_vs_persistence"][arm]) for arm in RETUNED_ARMS) + " |")
    A("")
    A("### Radius tertiles (retuned arms; skill vs persistence, mean reduced chi-square in parentheses)")
    A("")
    for k, rad in res["radius"].items():
        if "tertile_edges_m" not in rad:
            continue
        A(f"**{k.replace('|', ', ')}** - edges {rad['tertile_edges_m'][0]:.3f} m, "
          f"{rad['tertile_edges_m'][1]:.3f} m.")
        A("")
        A("| arm | run | inner | middle | outer |")
        A("|---|---|---|---|---|")
        for arm in RETUNED_ARMS:
            for label, r in (("first run", first), ("addendum 1", res)):
                cells = []
                for name in ("inner", "middle", "outer"):
                    a = r["radius"][k][name]["arms"][arm]
                    cells.append(f"{rg.fmt(a['skill_vs_persistence'])} ({a['chi2_mean']['point']:.2f})")
                A(f"| {arm} | {label} | " + " | ".join(cells) + " |")
        A("")
    A("## 5. Integrity of the reuse (registered statement 3)")
    A("")
    for name, ok in doc["integrity"].items():
        A(f"- {name}: {'identical to the first run' if ok else '**DIFFERS from the first run**'}")
    A(f"- `docs/paper/mast_gate.json` SHA-256 before and after this run: "
      f"`{doc['first_run_sha256_before']}` / `{doc['first_run_sha256_after']}`.")
    A("")
    A("## 6. Numerical expectations of the addendum (section 6)")
    A("")
    A("| expectation | registered | found | met |")
    A("|---|---|---|---|")
    for e in doc["expectations"]:
        A(f"| {e['expectation']} | {e['expected']} | {e['found']} | {'met' if e['met'] else 'not met'} |")
    n_met = sum(e["met"] for e in doc["expectations"])
    A("")
    A(f"{n_met} of {len(doc['expectations'])} met.")
    A("")
    A("## 7. Notes")
    A("")
    for n in doc["notes"]:
        A(f"- {n}")
    A("")
    return "\n".join(L)


NOTES = [
    "The first run is the registered primary; this addendum is reported next to it and does not "
    "replace it. The TEST tables and the TUNE surfaces of the first run had been seen when the "
    "addendum was registered.",
    "The pipeline is `run_gate.main()` itself: the addendum replaces the grid constants read by "
    "`run_gate.tune()` and wraps it in the stopping rule; `run_gate.py` is not edited.",
    "Population W uses the hyperparameters tuned on population G and the best causal arm chosen in "
    "population G, as in the first run.",
    "The JSON key `retuned` holds the complete output of the pipeline with the retuned arms (same "
    "structure as `mast_gate.json`); its `expectations` entry is section 6 of the FIRST registration "
    "re-evaluated on the retuned arms, and its `deviations_ambiguities` entry is the first-run list, "
    "which applies unchanged except for the grids.",
    "'Last step' changes are TUNE quantities. An edge-bound hyperparameter means the pooled TUNE MSE "
    "has no interior minimum inside the allowed extensions; the skill of that family is then a lower "
    "bound for the family only up to the reported last-step change.",
    "V_i offline GP: the selected point (640 ms, 2048 x scale) lies on a flat ridge of the TUNE "
    "surface where length and amplitude grow together (pooled TUNE MSE 19.27 to 19.60 for every "
    "length from 7.07 ms to 320 ms at that amplitude); the amplitude is edge-bound after the two "
    "allowed extensions with a last-step change of -1.5 %.",
    "A first execution of this script started the V_i tuning from the grids T_i had ended with "
    "(the starting grids were read from module globals already replaced), which is not the "
    "registered procedure; its TEST output was printed before the defect was found. Fixed by "
    "capturing the first-run grids at import; every number here is from the corrected run. The "
    "T_i numbers of the two executions are identical; for V_i the filter and causal-GP selections "
    "are identical (158.5 / 10; 40 ms, 16) and the offline-GP selection differs (160 ms, 256 then; "
    "640 ms, 2048 now, one more extension round).",
]


def main():
    sha_before = sha256(FIRST_JSON)
    first = json.loads(FIRST_JSON.read_text(encoding="utf-8"))
    rg.tune = tune_with_stopping_rule
    with tempfile.TemporaryDirectory() as tmp:
        rg.OUT_JSON = Path(tmp) / "pipeline.json"
        rg.OUT_MD = Path(tmp) / "pipeline.md"
        res = rg.main()
    res = json.loads(json.dumps(res))        # plain JSON types, as the first run was stored
    res["registration"] = REG
    final_edges = {tg: ROUNDS[tg][-1]["edges"] for tg in rg.TARGETS}
    integrity = {
        "counts": res["counts"] == first["counts"],
        "gate (a), both targets": res["gate_a"] == first["gate_a"],
        "regime number": all(
            res["regime"][tg]["regime_number_median_over_TEST_shots"]
            == first["regime"][tg]["regime_number_median_over_TEST_shots"] for tg in rg.TARGETS),
    }
    for arm in UNCHANGED_ARMS:
        integrity[f"{arm}, every readout, 4 target x population cells"] = all(
            res["readouts"][k]["arms"][arm] == first["readouts"][k]["arms"][arm]
            for k in res["readouts"])
    doc = {
        "registration": REG,
        "first_run": "docs/paper/mast_gate.json",
        "first_run_sha256_before": sha_before,
        "first_run_sha256_after": sha256(FIRST_JSON),
        "stopping_rule": {"max_extra_rounds": MAX_EXTRA_ROUNDS, "steps_per_extension": N_STEPS},
        "first_run_edges": {tg: first_edges(first, tg) for tg in rg.TARGETS},
        "final_edges": final_edges,
        "tuning_rounds": {tg: [{"grids": r["grids"], "edges": r["edges"],
                                "extended_after": r["extended_after"],
                                "tune_mse": {
                                    "running_mean": r["tinfo"]["running_mean"]["tune_mse"],
                                    "slope_filter_rows_q_level_cols_q_slope":
                                        r["tinfo"]["slope_filter"]["tune_mse_rows_q_level_cols_q_slope"],
                                    "gp_causal_rows_length_cols_amp":
                                        r["tinfo"]["gp_causal"]["tune_mse_rows_length_cols_amp"],
                                    "gp_offline_rows_length_cols_amp":
                                        r["tinfo"]["gp_offline"]["tune_mse_rows_length_cols_amp"]}}
                               for r in ROUNDS[tg]] for tg in rg.TARGETS},
        "gate_b_reread": {tg: {"first_run": first["gate_b"][tg], "addendum1": res["gate_b"][tg]}
                          for tg in rg.TARGETS},
        "skill_vs_persistence_G": {tg: {arm: {
            "first_run": first["readouts"][f"{tg}|G"]["arms"][arm]["skill_vs_persistence"],
            "addendum1": res["readouts"][f"{tg}|G"]["arms"][arm]["skill_vs_persistence"]}
            for arm in rg.ARMS if arm != "persistence"} for tg in rg.TARGETS},
        "integrity": integrity,
        "expectations": expectations(first, res, final_edges),
        "notes": NOTES,
        "retuned": res,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    doc["_first"] = first
    OUT_MD.write_text(render_md(doc), encoding="utf-8")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}", flush=True)
    for tg in rg.TARGETS:
        print(tg, "rounds", len(ROUNDS[tg]), "edges",
              {k: v["edge"] for k, v in final_edges[tg].items() if v["edge"]}, flush=True)
    print("integrity", integrity, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
