"""B.67 read-outs (PREREGISTRATION_B67.md sections 3, 4 and 6).  CPU; reads comparison_errors_test.npz of the runs
and the gate files b67_gates_test.npz of arm S.

Statements (re-partitions r2, r3, r4 pooled, cut population; two-level interval: a re-partition is drawn among the
three, then its discharges are resampled, 20,000 draws -- the interval of analyze_b39.arm_p, as in analyze_b59):
  G1  S (v2s, gated concatenation) over the network of B.39 (v2): interval above zero
  G2  S over the twin of B.39 (v2h): interval above zero
  G3  A (v2a, attention pooling) over the network: interval above zero
  G4  arm S: the BES gate is higher in rows 1 to 3 after an absent frame than in rows right after a measured frame;
      the interval of the difference is above zero
Reported: every network over the causal GP, hold and the offline GP; S over A, the network over the twin; in rows 1
to 3 after an absent frame and in the other rows, S, A and the network over the twin and S and A over the network;
median absolute errors; epochs; the gate readout of arm S by row class with intervals; the expectations of
section 6 beside the findings.

Row classes (rows of the stored test rows, which every run shares with the network of its fold):
  rows 1 to 3 after an absent frame   B.54 row table (data/.b54/rows_r{r}_f{k}.npz, flag blip13; run_b41 event rule)
  rows right after a measured frame   not in the class above, and the T_i frame 10 ms before the row is measured:
                                      the previous kept row is 10 ms earlier (|age_ms - 10| < 0.5) with a finite
                                      T_i (B.54 age_ms, hold_val); cross-checked against the network's own input
                                      (T_i staleness of 10 ms in the gate file)
  burst phase / inter-burst phase     run_b66.phase_masks on the B.54 rows (0 to 30 ms after an onset / at least
                                      50 ms from every onset; rows of detector classes 3 to 6), discharges of B.66
                                      classes B and C as in B.66 R3; the classes A, B, C themselves are also read
  temperature thirds                  thirds of the discharges of each re-partition by median measured T_i
                                      (run_b44.split_thirds, as B.54 and B.66)
Gate intervals: the row-weighted mean gate of a class; a re-partition drawn, its discharges resampled (20,000 draws,
the same draws for every class, so differences between classes are formed draw by draw).

    py ces_prediction/experiments/b67_gated/analyze_b67.py [--partial] [--out-dir DIR] [--selftest]
Output: docs/paper/b67_gated.json / .md.  Without --partial it stops, listing what is missing, unless all 30 runs, the
15 gate files and the 30 B.39 references are present; with --partial a network is read on the re-partitions whose five
folds it has (zero B.67 runs: everything of B.67 is 'not read', exit 0).  --selftest (check_b67 only) reads the B.39
network in place of S, the twin in place of A, and synthetic gates; it must be written outside docs/paper.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "b39_primary"))
sys.path.insert(0, str(EXP / "b66_regimes"))
import run_b67 as R  # noqa: E402
import analyze_b39 as A39  # noqa: E402  (load, same_rows, sums, ci)
import run_b66 as B66  # noqa: E402  (load_detect, phase_masks, CLASSES; B54, B44)

B54, B44 = B66.B54, B66.B44
REPO = HERE.parents[2]
DATA = R.DATA
B54C, B66C = DATA / ".b54", DATA / ".b66"
PAPER = REPO / "docs" / "paper"
REG = "ces_prediction/experiments/PREREGISTRATION_B67.md"
NB, SEED = 20_000, 20261167
K = 5
NETS = ("est", "twin", "S", "A")
LAB = {"est": "network v2 (B.39)", "twin": "twin v2h (B.39)", "S": "S: gated concatenation (v2s)",
       "A": "A: attention pooling (v2a)"}
REFS = {"gp_causal": "causal GP", "persistence": "hold", "gp": "offline GP"}
SOURCES = ("BES", "ECEI", "Mirnov", "CES-derived")      # model_seq_v2gate.SOURCE_LABELS (checked against each file)
STATEMENTS = {"G1": ("S", "est"), "G2": ("S", "twin"), "G3": ("A", "est")}
STEXT = {"G1": "S over the network (v2), interval above zero", "G2": "S over the twin, interval above zero",
         "G3": "A over the network, interval above zero",
         "G4": "arm S: BES gate in rows 1 to 3 after an absent frame minus that in rows right after a measured frame, "
               "interval above zero"}
EXPECT = {"G1": ("-0.01 to +0.02, not met", -0.01, 0.02, False), "G2": ("+0.03 to +0.06, met", 0.03, 0.06, True),
          "G3": ("-0.03 to +0.01, not met", -0.03, 0.01, False), "G4": ("met", None, None, True)}
AGE_MS, AGE_TOL_MS = 10.0, 0.5
STALE_S, STALE_TOL_S = 0.010, 0.0005
CELLS = ("all", "blip13", "after_measured", "other", "burst_BC", "inter_BC", "class_A", "class_B", "class_C",
         "third_low", "third_middle", "third_high")
CELL_LAB = {"all": "all scored rows", "blip13": "rows 1 to 3 after an absent frame",
            "after_measured": "rows right after a measured frame", "other": "other rows (neither of the two above)",
            "burst_BC": "burst phase (0 to 30 ms after an onset), classes B and C",
            "inter_BC": "inter-burst phase (>= 50 ms from every onset), classes B and C",
            "class_A": "class A (no burst)", "class_B": "class B (burst rate up to the median)",
            "class_C": "class C (burst rate above the median)",
            "third_low": "coldest third of the discharges", "third_middle": "middle third", "third_high": "hottest third"}


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


# ------------------------------------------------------------------------------ paths
SELFTEST = {"on": False}


def run_path(net, r, k):
    if net == "est" or (SELFTEST["on"] and net == "S"):
        return R.estimator_dir(r, k)
    if net == "twin" or (SELFTEST["on"] and net == "A"):
        return R.twin_dir(r, k)
    return R.run_dir(net, r, k)


def have(net, r, k):
    return (run_path(net, r, k) / "comparison_errors_test.npz").exists()


def have_gates(r, k):
    return SELFTEST["on"] or (R.run_dir("S", r, k) / R.GATES_NPZ).exists()


def load_gates(r, k, est):
    """(gates (n, 4), T_i staleness (n,), has (n,), check dict) of arm S on fold (r, k), rows checked against the
    network's rows.  --selftest: Dirichlet(1, 1, 1, 1) gates (seeded), staleness from nothing (NaN)."""
    n = len(est["shot"])
    if SELFTEST["on"]:
        g = rng_for(f"selftest|r{r}f{k}").dirichlet(np.ones(len(SOURCES)), size=n)
        return g, np.full(n, np.nan), np.zeros(n, bool), {"passes": True, "selftest": True}
    z = np.load(R.run_dir("S", r, k) / R.GATES_NPZ)
    if tuple(str(s) for s in z["sources"]) != SOURCES:
        raise SystemExit(f"FATAL: r{r} f{k}: gate file sources {list(z['sources'])} != {list(SOURCES)}")
    if not (np.array_equal(z["shot"], est["shot"]) and np.array_equal(z["row"], est["row"])):
        raise SystemExit(f"FATAL: r{r} f{k}: the gate file is not on the rows of the network")
    chk = json.loads(str(z["check"]))
    if not chk.get("passes"):
        raise SystemExit(f"FATAL: r{r} f{k}: the gate file's alignment check did not pass: {chk}")
    return z["gates"].astype(np.float64), z["ti_stale_s"].astype(np.float64), z["ti_has"].astype(bool), chk


# ------------------------------------------------------------------------------ data
def repeat_data(r, nets, with_gates, Dt, lab, F):
    cat = {q: [] for q in (*nets, *REFS, "shot", "blip", "age", "hold", "fidx", "grow", "cls", "g", "stale", "has")}
    epochs, gate_checks = {}, {}
    for k in range(K):
        z = {n: A39.load(run_path(n, r, k)) for n in nets}
        for n in nets:
            A39.same_rows(z["est"], z[n], f"r{r} f{k} {n} vs the network")
        t = np.load(B54C / f"rows_r{r}_f{k}.npz")
        if not (np.array_equal(t["shot"], z["est"]["shot"]) and np.array_equal(t["row"], z["est"]["row"])):
            raise SystemExit(f"FATAL: r{r} f{k}: the B.54 row table is not on the rows of the network")
        cat["shot"].append(z["est"]["shot"])
        for q, src in (("blip", "blip13"), ("age", "age_ms"), ("hold", "hold_val"), ("fidx", "fidx"), ("grow", "grow"),
                       ("cls", "cls")):
            cat[q].append(t[src])
        for n in nets:
            cat[n].append(z[n]["model"])
        for q in REFS:
            cat[q].append(z["est"][q])
        if with_gates:
            g, st, hs, chk = load_gates(r, k, z["est"])
            cat["g"].append(g)
            cat["stale"].append(st)
            cat["has"].append(hs)
            gate_checks[f"f{k}"] = {q: chk.get(q) for q in ("passes", "rows", "identical_rows", "p99_abs_err_diff_eV",
                                                            "max_abs_gate_sum_minus_one", "selftest") if q in chk}
        for n in nets:
            m = json.loads((run_path(n, r, k) / "metrics.json").read_text(encoding="utf-8"))
            epochs[f"{n} f{k}"] = [m["epochs_run"], m["best_epoch"]]
    c = {q: np.concatenate(v) for q, v in cat.items() if v}
    shot, blip = c["shot"], c["blip"].astype(bool)
    se = {q: c[q] for q in (*nets, *REFS)}
    sums = {"all": {}, "blip": {}, "other": {}}
    for q, v in se.items():
        sums["all"][q] = A39.sums(shot, v)
        sums["blip"][q] = A39.sums(shot, np.where(blip, v, 0.0))
        sums["other"][q] = A39.sums(shot, np.where(blip, 0.0, v))
    out = {"sums": sums, "abs": {n: np.sqrt(se[n]) for n in nets}, "abs_ref": {q: np.sqrt(se[q]) for q in REFS},
           "rows": int(len(shot)), "discharges": int(len(np.unique(shot))), "blip_rows": int(blip.sum()),
           "epochs": epochs, "nets": list(nets), "gates": with_gates, "gate_checks": gate_checks}
    if not with_gates:
        return out
    # row classes
    after = ~blip & (np.abs(c["age"] - AGE_MS) < AGE_TOL_MS) & np.isfinite(c["hold"])
    fidx, grow, cls = c["fidx"], c["grow"], c["cls"]
    burst, inter = B66.phase_masks(Dt, fidx, grow, cls)
    L = lab[fidx]
    u, f = np.unique(fidx, return_inverse=True)
    v = F["median_ti"][u]
    hv = np.flatnonzero(np.isfinite(v))
    third = np.full(len(u), -1)
    for j, idx in enumerate(B44.split_thirds(v[hv], F["shot"][u][hv])):
        third[hv[idx]] = j
    BC = np.isin(L, ("B", "C"))
    masks = {"all": np.ones(len(shot), bool), "blip13": blip, "after_measured": after, "other": ~blip & ~after,
             "burst_BC": burst & BC, "inter_BC": inter & BC, "class_A": L == "A", "class_B": L == "B",
             "class_C": L == "C", "third_low": third[f] == 0, "third_middle": third[f] == 1, "third_high": third[f] == 2}
    g = c["g"]
    out["n_dis"] = int(len(u))
    out["gcell"] = {}
    out["cell_rows"], out["cell_discharges"] = {}, {}
    for name, m in masks.items():
        gs = np.stack([np.bincount(f[m], weights=g[m, s], minlength=len(u)) for s in range(len(SOURCES))], axis=1)
        out["gcell"][name] = (gs, np.bincount(f[m], minlength=len(u)).astype(np.float64))
        out["cell_rows"][name] = int(m.sum())
        out["cell_discharges"][name] = int(len(np.unique(f[m])))
    out["gate_quantiles"] = {s: [float(np.quantile(g[:, j], q)) for q in (0.05, 0.5, 0.95)] for j, s in enumerate(SOURCES)}
    if not SELFTEST["on"]:
        mine = ~blip & c["has"] & (np.abs(c["stale"] - STALE_S) < STALE_TOL_S)
        out["after_measured_agreement"] = {"rows": int(len(shot)), "agree": int(np.sum(mine == after)),
                                           "b54_rule": int(after.sum()), "network_input_rule": int(mine.sum())}
    return out


# ------------------------------------------------------------------------------ intervals
def hier(per, a, b, repeats, group="all"):
    """Copied from analyze_b39.arm_p.hier (a closure there; also analyze_b59.hier), with the repeats and the row
    group as arguments and one generator per quantity: a repeat is drawn, then its discharges are resampled with
    replacement.  Point = mean over repeats."""
    rng = rng_for(f"skill|{a}|{b}|{group}|{','.join(map(str, repeats))}")
    draws = rng.choice(np.array(repeats), size=NB)
    o = np.empty(NB)
    for r in repeats:
        sel = np.flatnonzero(draws == r)
        m, bb = per[r]["sums"][group][a], per[r]["sums"][group][b]
        idx = rng.integers(0, len(m), size=(len(sel), len(m)))
        o[sel] = 1.0 - m[idx].sum(1) / bb[idx].sum(1)
    pr = {str(r): float(1 - per[r]["sums"][group][a].sum() / per[r]["sums"][group][b].sum()) for r in repeats}
    return [float(np.mean(list(pr.values()))), *A39.ci(o)], pr


def gate_boot(per, repeats):
    """Row-weighted mean gate per class and source: a repeat drawn, its discharges resampled; the same draws for
    every class.  Returns (draws per class (NB, 4), points per class (4,), per-repeat means per class)."""
    rng = rng_for(f"gates|{','.join(map(str, repeats))}")
    draws = rng.choice(np.array(repeats), size=NB)
    out = {c: np.full((NB, len(SOURCES)), np.nan) for c in CELLS}
    for r in repeats:
        sel = np.flatnonzero(draws == r)
        n = per[r]["n_dis"]
        for c0 in range(0, len(sel), 500):
            ss = sel[c0:c0 + 500]
            idx = rng.integers(0, n, size=(len(ss), n))
            for c in CELLS:
                gs, ct = per[r]["gcell"][c]
                den = ct[idx].sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[c][ss] = gs[idx].sum(1) / den[:, None]
    per_rep = {c: {str(r): (per[r]["gcell"][c][0].sum(0) / per[r]["gcell"][c][1].sum()).tolist()
                   if per[r]["gcell"][c][1].sum() > 0 else [float("nan")] * len(SOURCES) for r in repeats} for c in CELLS}
    points = {c: np.nanmean(np.array(list(per_rep[c].values()), float), axis=0) for c in CELLS}
    return out, points, per_rep


def ci_or_nan(x):
    x = np.asarray(x, float)
    return A39.ci(x) if np.isfinite(x).any() else [float("nan"), float("nan")]


# ------------------------------------------------------------------------------ readout
def readout(partial):
    avail = {n: [r for r in R.REPEATS if all(have(n, r, k) for k in range(K))] for n in NETS}
    gates_avail = [r for r in avail["S"] if all(have_gates(r, k) for k in range(K))]
    missing = [run_path(n, r, k).name for n in NETS for r in R.REPEATS for k in range(K) if not have(n, r, k)]
    missing += [f"{R.run_dir('S', r, k).name}/{R.GATES_NPZ}" for r in R.REPEATS for k in range(K)
                if have("S", r, k) and not have_gates(r, k)]
    if missing:
        print(f"[b67] missing ({len(missing)}): {', '.join(missing)}")
        if not partial:
            raise SystemExit("[b67] not every run is present; nothing written (use --partial to read what is complete)")
    base = [r for r in R.REPEATS if r in avail["est"] and r in avail["twin"]]
    if not base:
        raise SystemExit("[b67] the B.39 network and twin are not complete on any re-partition")
    need = [B54C / f"rows_r{r}_f{k}.npz" for r in base for k in range(K)] + [B66C / "detect.npz", B66C / "classes.npz",
                                                                             B54C / "files.npz"]
    for p in need:
        if not p.exists():
            raise SystemExit(f"FATAL: {p} missing (B.54 / B.66 caches)")
    Dt, lab, F = B66.load_detect(), np.load(B66C / "classes.npz")["lab"], B54.load_files()
    per = {}
    for r in base:
        nets = [n for n in NETS if r in avail[n]]
        per[r] = repeat_data(r, nets, r in gates_avail, Dt, lab, F)
        print(f"[b67] r{r}: {per[r]['rows']:,} rows, {per[r]['discharges']} discharges, networks {nets}, gates {r in gates_avail}",
              flush=True)

    def reps(*nets):
        return [r for r in base if all(n in per[r]["nets"] for n in nets)]

    def H(a, b, group="all"):
        rr = reps(*(x for x in (a, b) if x not in REFS))
        if not rr:
            return None
        v, p = hier(per, a, b, rr, group)
        return {"skill": v, "per_repeat": p, "repeats": rr}

    res = {"registration": REG, "seed": SEED, "draws": NB, "selftest": SELFTEST["on"], "partial": bool(missing),
           "missing": missing, "repeats_read": {n: reps(n) for n in NETS}, "gate_repeats_read": gates_avail,
           "rows": {str(r): {"rows": per[r]["rows"], "discharges": per[r]["discharges"], "blip_rows": per[r]["blip_rows"]}
                    for r in base}}
    st = {}
    for name, (a, b) in STATEMENTS.items():
        h = H(a, b)
        if h is None:
            st[name] = {"read": False}
            continue
        txt, lo, hi, exp_holds = EXPECT[name]
        st[name] = {"read": True, "numerator": a, "denominator": b, **h, "holds": bool(h["skill"][1] > 0),
                    "complete": len(h["repeats"]) == len(R.REPEATS), "expectation": txt,
                    "point_inside_expected_range": bool(lo <= h["skill"][0] <= hi),
                    "verdict_as_expected": bool((h["skill"][1] > 0) == exp_holds)}
    # gate readout (arm S)
    gr = {"read": False}
    if gates_avail:
        draws, points, per_rep = gate_boot(per, gates_avail)
        cells = {}
        for c in CELLS:
            cells[c] = {"label": CELL_LAB[c],
                        "mean_gate": {s: [float(points[c][j]), *ci_or_nan(draws[c][:, j])] for j, s in enumerate(SOURCES)},
                        "per_repeat": per_rep[c],
                        "rows_per_repeat": {str(r): per[r]["cell_rows"][c] for r in gates_avail},
                        "discharges_per_repeat": {str(r): per[r]["cell_discharges"][c] for r in gates_avail}}

        def diff(c1, c2):
            o = {}
            for j, s in enumerate(SOURCES):
                pr = {str(r): float(per_rep[c1][str(r)][j] - per_rep[c2][str(r)][j]) for r in gates_avail}
                o[s] = {"value": [float(np.mean(list(pr.values()))), *ci_or_nan(draws[c1][:, j] - draws[c2][:, j])],
                        "per_repeat": pr}
            return o

        d_blip = diff("blip13", "after_measured")
        g4 = d_blip["BES"]
        txt, _, _, exp_holds = EXPECT["G4"]
        st["G4"] = {"read": True, "value": g4["value"], "per_repeat": g4["per_repeat"], "repeats": gates_avail,
                    "holds": bool(g4["value"][1] > 0), "complete": len(gates_avail) == len(R.REPEATS),
                    "expectation": txt, "verdict_as_expected": bool((g4["value"][1] > 0) == exp_holds)}
        gr = {"read": True, "repeats": gates_avail, "cells": cells,
              "difference_blip13_minus_after_measured": d_blip,
              "difference_burst_minus_inter_BC": diff("burst_BC", "inter_BC"),
              "difference_class_C_minus_class_A": diff("class_C", "class_A"),
              "difference_third_high_minus_third_low": diff("third_high", "third_low"),
              "gate_quantiles_over_rows": {str(r): per[r]["gate_quantiles"] for r in gates_avail},
              "gate_file_checks": {str(r): per[r]["gate_checks"] for r in gates_avail},
              "after_measured_agreement": {str(r): per[r].get("after_measured_agreement") for r in gates_avail}}
    else:
        st["G4"] = {"read": False}
    res["statements"] = st
    res["gate_readout"] = gr
    rep = {"over_refs": {}, "pairs": {}, "blip_rows": {}, "other_rows": {}}
    for n in NETS:
        if reps(n):
            rep["over_refs"][n] = {q: H(n, q) for q in REFS}
    for a, b in (("S", "A"), ("est", "twin"), ("A", "twin")):
        rep["pairs"][f"{a}_over_{b}"] = H(a, b)
    for a, b in (("S", "twin"), ("A", "twin"), ("est", "twin"), ("S", "est"), ("A", "est")):
        rep["blip_rows"][f"{a}_over_{b}"] = H(a, b, "blip")
        rep["other_rows"][f"{a}_over_{b}"] = H(a, b, "other")
    rep["median_abs_error_eV"] = {LAB[n]: float(np.mean([np.median(per[r]["abs"][n]) for r in reps(n)]))
                                  for n in NETS if reps(n)}
    rep["median_abs_error_eV"].update({REFS[q]: float(np.mean([np.median(per[r]["abs_ref"][q]) for r in base])) for q in REFS})
    rep["epochs_run_best"] = {str(r): per[r]["epochs"] for r in base}
    res["reported"] = rep
    return res


# ------------------------------------------------------------------------------ render
def s3(x):
    return "not read" if x is None else f"{x['skill'][0]:+.3f} [{x['skill'][1]:+.3f}, {x['skill'][2]:+.3f}]"


def v3(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def g3(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def pr(x):
    return "-" if x is None else ", ".join(f"r{r} {v:+.3f}" for r, v in x["per_repeat"].items())


def render(res):
    S, P, G = res["statements"], res["reported"], res["gate_readout"]
    L = ["# B.67 — weighing the sources before the recurrent layer: source-gated fusion", ""]
    if res["selftest"]:
        L += ["**SELFTEST** (check_b67): the B.39 network stands in for S, the twin for A, the gates are synthetic. "
              "Not a result.", ""]
    L += [f"Registered: {REG}. GPU runs (one training per fold, initialisation seed of the fold); read-outs on a CPU. T_i, "
          "cut population, the fifteen folds of re-partitions r2, r3, r4 of the pooled cross-validation of B.39. Skill of "
          "X over Y = 1 - sum of squared errors of X / that of Y on the same rows; 95 % two-level interval (a re-partition "
          "drawn among those read, then its discharges resampled; 20,000 draws); point = mean over re-partitions.", ""]
    if res["partial"]:
        L += [f"**Partial read-out**: {len(res['missing'])} items missing. Re-partitions read per network: "
              + "; ".join(f"{LAB[n]} {v}" for n, v in res["repeats_read"].items())
              + f"; gate readout {res['gate_repeats_read']}.", ""]
    L += ["## Statements", "", "| statement | value (interval) | per re-partition | holds | expectation (section 6) | "
          "point inside the expected range |", "|---|---|---|---|---|---|"]
    for name in ("G1", "G2", "G3", "G4"):
        x = S[name]
        if not x["read"]:
            L.append(f"| **{name}** {STEXT[name]} | not read | - | - | {EXPECT[name][0]} | - |")
            continue
        val = v3(x["value"]) if name == "G4" else s3(x)
        per_r = ", ".join(f"r{r} {v:+.3f}" for r, v in x["per_repeat"].items())
        inside = "-" if name == "G4" else x["point_inside_expected_range"]
        L.append(f"| **{name}** {STEXT[name]} | {val} | {per_r} | {'holds' if x['holds'] else 'does not hold'}"
                 f"{'' if x['complete'] else ' (partial)'} | {x['expectation']} | {inside} |")
    L += ["", "## Every network over the references", "", "| network | over the causal GP | over hold | over the offline GP |",
          "|---|---|---|---|"]
    for n, d in P["over_refs"].items():
        L.append(f"| {LAB[n]} | {s3(d['gp_causal'])} | {s3(d['persistence'])} | {s3(d['gp'])} |")
    names = {"S_over_A": "S over A", "est_over_twin": "network over the twin", "A_over_twin": "A over the twin"}
    L += ["", "| comparison | skill (interval) | per re-partition |", "|---|---|---|"]
    for key, x in P["pairs"].items():
        L.append(f"| {names[key]} | {s3(x)} | {pr(x)} |")
    L += ["", "## By rows", "", "Rows 1 to 3 after an absent frame: B.54 row table (flag blip13) on the rows of the network. "
          "Rows per re-partition: " + "; ".join(f"r{r}: {v['rows']:,} rows, {v['blip_rows']:,} after an absent frame, "
                                               f"{v['discharges']} discharges" for r, v in res["rows"].items()) + ".", "",
          "| comparison | rows 1 to 3 after an absent frame | other rows |", "|---|---|---|"]
    lab2 = {"S": "S", "A": "A", "est": "network", "twin": "twin"}
    for key in P["blip_rows"]:
        a, b = key.split("_over_")
        L.append(f"| {lab2[a]} over {lab2[b]} | {s3(P['blip_rows'][key])} | {s3(P['other_rows'][key])} |")
    L += ["", "## Median absolute errors (eV, mean over re-partitions)", "", "| network or reference | median absolute error |",
          "|---|---|"]
    for n, v in P["median_abs_error_eV"].items():
        L.append(f"| {n} | {v:.1f} |")
    L += ["", "## Gate readout, arm S", ""]
    if not G["read"]:
        L.append("Not read: arm S has no re-partition with all five gate files.")
    else:
        L += [f"Row-weighted mean gate of each source on the test rows (re-partitions {G['repeats']}); interval: a "
              "re-partition drawn, its discharges resampled, the same draws for every class. The gates of a row sum to one; "
              "uniform gates are 0.25 each. A gate is which source the trained network weighs, not a physical cause; gates "
              "near uniform do not show that the sources are used equally (registration section 7).", "",
              "| rows | rows (per re-partition) | " + " | ".join(SOURCES) + " |", "|---|---|" + "---|" * len(SOURCES)]
        for c, v in G["cells"].items():
            rows = ", ".join(f"{x:,}" for x in v["rows_per_repeat"].values())
            L.append(f"| {v['label']} | {rows} | " + " | ".join(g3(v["mean_gate"][s]) for s in SOURCES) + " |")
        L += ["", "| difference of the mean gate | " + " | ".join(SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
        for key, nm in (("difference_blip13_minus_after_measured", "rows 1 to 3 after an absent frame minus rows right after a measured frame (G4: BES)"),
                        ("difference_burst_minus_inter_BC", "burst phase minus inter-burst phase (classes B and C)"),
                        ("difference_class_C_minus_class_A", "class C minus class A"),
                        ("difference_third_high_minus_third_low", "hottest third minus coldest third")):
            L.append(f"| {nm} | " + " | ".join(v3(G[key][s]["value"]) for s in SOURCES) + " |")
        L += ["", "Gate quantiles over rows (5 %, 50 %, 95 %) per re-partition: " + "; ".join(
            f"r{r}: " + ", ".join(f"{s} {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}" for s, q in d.items())
            for r, d in G["gate_quantiles_over_rows"].items()) + "."]
        ag = {r: a for r, a in G["after_measured_agreement"].items() if a}
        if ag:
            L += ["", "Rows right after a measured frame, B.54 rule against the network's own input (T_i staleness 10 ms, "
                  "outside rows 1 to 3 after an absent frame): " + "; ".join(
                      f"r{r}: agree on {a['agree']:,} of {a['rows']:,} rows (B.54 rule {a['b54_rule']:,}, input rule "
                      f"{a['network_input_rule']:,})" for r, a in ag.items()) + "."]
        L += ["", "Alignment checks of the gate files (squared errors reproduced in the scoring process): " + "; ".join(
            f"r{r}: " + ", ".join(f"{f} {'pass' if c.get('passes') else 'FAIL'}" for f, c in d.items())
            for r, d in G["gate_file_checks"].items()) + "."]
    L += ["", "## Epochs run (best epoch)", "", "| re-partition | runs |", "|---|---|"]
    for r, d in P["epochs_run_best"].items():
        L.append(f"| r{r} | " + "; ".join(f"{k} {v[0]} ({v[1]})" for k, v in d.items()) + " |")
    L += ["", "## Choices fixed in the implementation", "",
          "- Networks of B.39 on the same folds: `.b39p_v2_cut_r{r}_f{k}` (the network) and `.b39p_v2h_cut_r{r}_f{k}` (the twin).",
          "- Rows right after a measured frame: outside rows 1 to 3 after an absent frame, the previous kept row 10 ms "
          "earlier (|age - 10 ms| < 0.5 ms) with a finite T_i (B.54 `age_ms`, `hold_val`).",
          "- Burst and inter-burst phases: `run_b66.phase_masks` on the B.54 rows, restricted to the discharges of B.66 "
          "classes B and C (B.66 R3); classes A, B, C read as whole classes as well.",
          "- Temperature thirds: within each re-partition, by the median measured T_i of a discharge (`run_b44.split_thirds`).",
          "- Gate intervals: one generator for all classes (the same draws), so the differences are formed draw by draw.",
          ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="read the re-partitions each network has complete")
    ap.add_argument("--out-dir", default=str(PAPER))
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.selftest:
        if out_dir.resolve() == PAPER.resolve():
            raise SystemExit("FATAL: --selftest output must not go to docs/paper")
        SELFTEST["on"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    res = readout(args.partial)
    (out_dir / "b67_gated.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "b67_gated.md").write_text(render(res), encoding="utf-8", newline="\n")
    print("written", out_dir / "b67_gated.md", "| partial:", res["partial"], "| selftest:", res["selftest"])


if __name__ == "__main__":
    main()
