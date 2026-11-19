"""B.70 phase 1 readout (PREREGISTRATION_B70.md sections 3 (c), 4, 6 and Addendum 1).  CPU only; nothing is trained.

Reads the D1 re-scorings (data/.b70_rescore/, runs_b70.py / batch_b70.py) and the second stage on the D1 rows
(data/.b70/stage2_r{r}_f{k}.npz, stage2_b70.py); fits the good-and-bad mixture of D1-6 (mixture_b70.py) per fold.

    py ces_prediction/experiments/b70_d1/analyze_b70.py
Output: docs/paper/b70_d1.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b64_two_stage"))
sys.path.insert(0, str(HERE))
import stage2_b64 as S2  # noqa: E402
import stage2_b70 as ST  # noqa: E402
import runs_b70 as RB  # noqa: E402
import mixture_b70 as MX  # noqa: E402

A39, B61 = S2.A39, S2.B61
REPO = HERE.parents[2]
PAPER = REPO / "docs" / "paper"
DATA = RB.DATA
OUT_JSON, OUT_MD = PAPER / "b70_d1.json", PAPER / "b70_d1.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B70.md"
REPEATS = (2, 3, 4)
K = 5
N_BOOT = 20_000
SEED = 20261170
FLAG_P = 0.5
LOW_EV, HIGH_EV = 75.0, 3000.0
TOL_EXPECT = 0.01
T0 = _time.time()

LAB = {"c2": "two-stage estimator", "t2": "two-stage twin", "c": "network (first stage)", "t": "history-only twin",
       "gp": "causal GP", "hold": "hold", "off": "offline GP", "pchip": "PCHIP",
       "c2nan": "two-stage estimator (boosted model's own missing-input handling)",
       "t2nan": "two-stage twin (boosted model's own missing-input handling)",
       "L": "B.59 L: fast channels one row old", "N": "B.59 N: no Mirnov", "E": "B.59 E: ECEI + CES history",
       "B": "B.59 B: BES + CES history", "S": "B.67 S: gated concatenation", "A": "B.67 A: attention pooling",
       "X": "B.68 X: auxiliary target"}
MAIN = ("c2", "t2", "c", "t", "gp", "hold", "off", "pchip")
OTHER = {"b59": ("L", "N", "E", "B"), "b67": ("S", "A"), "b68": ("X",)}
OTHER_EST = tuple(a for v in OTHER.values() for a in v)
PAIRS = [("c2", "c"), ("c2", "gp"), ("c", "gp"), ("c2", "t2"), ("t2", "t"), ("c", "t"), ("t2", "gp"), ("t", "gp"),
         *[(a, b) for a in ("c2", "t2", "c") for b in ("hold", "off", "pchip")], ("t", "hold"), ("gp", "hold")]
OTHER_PAIRS = [("L", "t"), ("c", "L"), ("N", "t"), ("c", "N"), ("E", "t"), ("B", "t"), ("c", "B"), ("c", "E"), ("B", "E"),
               ("S", "c"), ("S", "t"), ("A", "c"), ("A", "t"), ("S", "A"), ("X", "c"), ("X", "t"),
               *[(a, b) for a in OTHER_EST for b in ("gp", "hold", "off")]]
MED_PAIRS = [(e, "hold") for e in MAIN if e != "hold"] + [("c2", "c"), ("c2", "gp"), ("c", "gp"), ("c2", "t2"),
                                                           ("c", "t"), ("X", "gp")]
NAN_PAIRS = [("c2nan", "c"), ("c2nan", "gp"), ("c2nan", "t2nan"), ("t2nan", "t"), ("c2nan", "hold")]


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


# ------------------------------------------------------------------------------------------ data
def load_fold(r, k):
    st = np.load(ST.CACHE / f"stage2_r{r}_f{k}.npz")
    zc = ST.load_npz(RB.new_dir("b39", "v2", r, k) / "comparison_errors_test.npz")
    zt = ST.load_npz(RB.new_dir("b39", "v2h", r, k) / "comparison_errors_test.npz")
    same = lambda a, b: np.array_equal(a["shot"], b["shot"]) and np.array_equal(a["row"], b["row"])  # noqa: E731
    if not (same(zc, zt) and np.array_equal(st["shot"], zc["shot"]) and np.array_equal(st["row"], zc["row"])):
        raise SystemExit(f"FATAL: r{r} f{k}: network, twin and second stage are not on the same rows")
    y = zc["y"]
    se = {"c": zc["se"], "t": zt["se"], "hold": zc["hold"], "gp": zc["gp"], "off": zc["off"], "pchip": zc["pchip"]}
    for q in ("hold", "gp", "off", "pchip"):
        if not np.array_equal(zc[q], zt[q]):
            raise SystemExit(f"FATAL: r{r} f{k}: reference {q} differs between network and twin")
    for e in ("c2", "t2", "c2nan", "t2nan"):
        se[e] = (st[f"est_{e}"] - y) ** 2
    dev1 = {e: float(np.max(np.abs(np.sqrt((st[f"est_{e}"] - y) ** 2) - np.sqrt(se[e])))) for e in ("c", "t")}
    for fam, arms in OTHER.items():
        for a in arms:
            z = ST.load_npz(RB.new_dir(fam, a, r, k) / "comparison_errors_test.npz")
            if not same(z, zc) or not np.array_equal(z["gp"], zc["gp"]) or not np.array_equal(z["y"], y):
                raise SystemExit(f"FATAL: r{r} f{k}: {fam} {a} is not on the rows of the network")
            se[a] = z["se"]
    return {"fidx": st["fidx"], "grow": st["grow"], "shot": zc["shot"], "y": y, "se": se,
            "fb_c2": st["fallback_c2"], "fb_t2": st["fallback_t2"], "fold": np.full(len(y), k),
            "first_stage_dev_eV": dev1, "info": json.loads(str(st["info"]))}


def load_all():
    D, infos = {}, {}
    for r in REPEATS:
        parts = [load_fold(r, k) for k in range(K)]
        d = {kk: np.concatenate([p[kk] for p in parts]) for kk in ("fidx", "grow", "shot", "y", "fb_c2", "fb_t2", "fold")}
        d["se"] = {e: np.concatenate([p["se"][e] for p in parts]) for e in parts[0]["se"]}
        u, f = np.unique(d["fidx"], return_inverse=True)
        d["u"], d["f"], d["n"] = u, f, len(u)
        d["keep_trim"] = d["se"]["hold"] < np.quantile(d["se"]["hold"], 1 - B61.TRIM)      # run_b61 / stage2_b64
        d["fold_rows"] = [int(len(p["y"])) for p in parts]
        d["first_stage_dev_eV"] = {e: max(p["first_stage_dev_eV"][e] for p in parts) for e in ("c", "t")}
        D[r] = d
        for k, p in enumerate(parts):
            infos[f"r{r}f{k}"] = p["info"]
        log(f"r{r}: {len(d['y']):,} rows, {d['n']} discharges")
    return D, infos


# ------------------------------------------------------------------------------------------ mixture (D1-6)
def grid_inclusive():
    sys.path.insert(0, str(RB.SEQ_DIR))
    sys.path.insert(1, str(REPO / "ces_prediction"))
    from seq_data import load_grid_files
    g, _ = load_grid_files(DATA, True, 0.0)
    return g


def file_residuals(grid, names):
    out = {}
    for i, nm in enumerate(names):
        arr = grid[nm]
        out[i] = MX.residuals(arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64))
    return out


def mixture(C, D):
    names = C.names
    res_cut = file_residuals(C.grid, names)
    g_inc = grid_inclusive()
    if sorted(g_inc) != list(names):
        raise SystemExit("FATAL: the inclusive grid has another file list")
    res_inc = file_residuals(g_inc, names)
    # values above 3 keV on the inclusive grid (first and last grid row excluded)
    n_above = 0
    for nm in names:
        ti = g_inc[nm][1:-1, 1].astype(np.float64)
        n_above += int(np.sum(np.isfinite(ti) & (ti > HIGH_EV)))
    out = {"values_above_3keV_inclusive_grid": n_above, "registered_count": 1197, "fits": {}, "per_repeat": {}}
    flags = {}
    for r in REPEATS:
        d = D[r]
        flag = np.zeros(len(d["y"]), bool)
        undefined = np.zeros(len(d["y"]), bool)
        cnt = {"scored_rows": int(len(d["y"])), "scored_rows_flagged": 0, "scored_flagged_below_75eV": 0,
               "scored_flagged_single_frames": 0, "scored_rows_without_residual": 0,
               "test_values_flagged_cut": 0, "test_values_cut": 0,
               "inc_values": 0, "inc_values_flagged": 0, "inc_above_3keV": 0, "inc_above_3keV_flagged": 0,
               "inc_above_3keV_without_residual": 0, "inc_flagged_at_or_below_3keV": 0, "inc_flagged_below_75eV": 0,
               "inc_flagged_single_frames": 0, "inc_above_3keV_flagged_single_frames": 0}
        for k in range(K):
            man = S2.manifest(r, k)
            tr = S2.idx_of(C, man["train_files"])
            te = S2.idx_of(C, man["test_files"])
            fit_c = MX.fit_em(np.concatenate([res_cut[i][0][res_cut[i][1]] for i in tr]))
            fit_i = MX.fit_em(np.concatenate([res_inc[i][0][res_inc[i][1]] for i in tr]))
            out["fits"][f"r{r}f{k}"] = {"cut": fit_c, "before_cut": fit_i}
            sel = np.flatnonzero(d["fold"] == k)
            for i in te:
                nm = names[i]
                arr = C.grid[nm]
                t, ti = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
                pf = MX.posterior_bad(res_cut[i][0], fit_c)
                ff = np.nan_to_num(pf, nan=0.0) > FLAG_P
                single = MX.single_frames(ff, ti, t)
                meas = np.isfinite(ti)
                meas[0] = meas[-1] = False
                cnt["test_values_cut"] += int(meas.sum())
                cnt["test_values_flagged_cut"] += int(ff[meas].sum())
                rows = sel[d["fidx"][sel] == i]
                if len(rows):
                    gr = d["grow"][rows]
                    flag[rows] = ff[gr]
                    undefined[rows] = ~res_cut[i][1][gr]
                    cnt["scored_flagged_single_frames"] += int((ff[gr] & single[gr]).sum())
                # before the cut
                a2 = g_inc[nm]
                t2, ti2 = a2[:, 0].astype(np.float64), a2[:, 1].astype(np.float64)
                p2 = MX.posterior_bad(res_inc[i][0], fit_i)
                f2 = np.nan_to_num(p2, nan=0.0) > FLAG_P
                s2 = MX.single_frames(f2, ti2, t2)
                m2 = np.isfinite(ti2)
                m2[0] = m2[-1] = False
                hi = m2 & (ti2 > HIGH_EV)
                cnt["inc_values"] += int(m2.sum())
                cnt["inc_values_flagged"] += int(f2[m2].sum())
                cnt["inc_above_3keV"] += int(hi.sum())
                cnt["inc_above_3keV_flagged"] += int(f2[hi].sum())
                cnt["inc_above_3keV_without_residual"] += int((hi & ~res_inc[i][1]).sum())
                cnt["inc_flagged_at_or_below_3keV"] += int((f2 & m2 & ~hi).sum())
                cnt["inc_flagged_below_75eV"] += int((f2 & m2 & (ti2 < LOW_EV)).sum())
                cnt["inc_flagged_single_frames"] += int((f2 & m2 & s2).sum())
                cnt["inc_above_3keV_flagged_single_frames"] += int((f2 & hi & s2).sum())
        cnt["scored_rows_flagged"] = int(flag.sum())
        cnt["scored_flagged_below_75eV"] = int((flag & (d["y"] < LOW_EV)).sum())
        cnt["scored_flagged_above_3keV"] = int((flag & (d["y"] > HIGH_EV)).sum())
        cnt["scored_rows_without_residual"] = int(undefined.sum())
        cnt["inc_share_above_3keV_flagged"] = cnt["inc_above_3keV_flagged"] / max(cnt["inc_above_3keV"], 1)
        cnt["scored_flagged_share"] = cnt["scored_rows_flagged"] / max(cnt["scored_rows"], 1)
        out["per_repeat"][str(r)] = cnt
        flags[r] = flag
        log(f"mixture r{r}: scored flagged {cnt['scored_rows_flagged']:,}; before the cut: "
            f"{cnt['inc_above_3keV_flagged']}/{cnt['inc_above_3keV']} above 3 keV flagged, "
            f"{cnt['inc_flagged_at_or_below_3keV']:,} at or below")
    pr = out["per_repeat"].values()
    out["mean_over_repeats"] = {kk: float(np.mean([v[kk] for v in pr])) for kk in next(iter(pr))}
    fc = [v["cut"] for v in out["fits"].values()]
    fi = [v["before_cut"] for v in out["fits"].values()]
    out["fit_ranges"] = {nm: {p: [float(min(x[p] for x in L_)), float(max(x[p] for x in L_))] for p in ("w", "s_g", "s_b")}
                         for nm, L_ in (("cut", fc), ("before_cut", fi))}
    return out, flags


# ------------------------------------------------------------------------------------------ old values
def old_values():
    B64 = json.loads((PAPER / "b64_two_stage.json").read_text(encoding="utf-8"))
    B39 = json.loads((PAPER / "b39_primary.json").read_text(encoding="utf-8"))["P"]
    B61j = json.loads((PAPER / "b61_absolute_robust.json").read_text(encoding="utf-8"))
    B59 = json.loads((PAPER / "b59_inputs.json").read_text(encoding="utf-8"))
    B67 = json.loads((PAPER / "b67_gated.json").read_text(encoding="utf-8"))
    B68 = json.loads((PAPER / "b68_auxtarget.json").read_text(encoding="utf-8"))
    return B64, B39, B61j["part_a_r2_r9"], B59, B67, B68


def comparisons(A, Q, st, B64, B39, a29, B59):
    """Every main-text number that phase 1 recomputes: (label, D1 value, old value, old source)."""
    ar, md = B64["all_rows"], B64["medians"]
    rows = []

    def add(label, new, old, src, kind="skill"):
        if new is None or old is None:
            return
        rows.append({"quantity": label, "d1": new, "old": old, "old_source": src, "kind": kind,
                     "diff": float(new[0] - old[0])})

    ref39 = {"hold": "persistence", "gp": "gp_causal", "pchip": "pchip", "off": "gp"}
    for b in ("hold", "gp", "pchip", "off"):
        add(f"Table main (a): network over {LAB[b]}, MSE skill", A[f"skill_mse|c|{b}"],
            B39["causal_over"][ref39[b]] if b != "gp" else B39["P1_causal_over_gp"], "b39_primary.json P (r2-r9, old subset)")
        add(f"  same, against its r2-r4 value of B.64", A[f"skill_mse|c|{b}"], ar.get(f"skill_mse|c|{b}"),
            "b64_two_stage.json (r2-r4, old subset)")
    add("Table main (a): network over the twin, MSE skill", A["skill_mse|c|t"], B39["P2_causal_over_twin"],
        "b39_primary.json P (r2-r9, old subset)")
    add("  same, against its r2-r4 value of B.64", A["skill_mse|c|t"], ar["skill_mse|c|t"], "b64_two_stage.json (r2-r4)")
    for b, kk in (("hold", "hold"), ("gp", "gp"), ("t", "t"), ("off", "off")):
        add(f"Table main (a): network over {LAB[b]}, Huber skill", A[f"skill_huber|c|{b}"], a29.get(f"skill_huber|c|{kk}"),
            "b61_absolute_robust.json part_a_r2_r9")
        add(f"Table main (a): network over {LAB[b]}, 1 %-trimmed skill", A[f"skill_trim|c|{b}"], a29.get(f"skill_trim|c|{kk}"),
            "b61_absolute_robust.json part_a_r2_r9")
    for b in ("hold", "gp", "t2", "c", "pchip", "off"):
        for kind, nm in (("mse", "MSE"), ("huber", "Huber"), ("trim", "1 %-trimmed")):
            add(f"Table main (a): two-stage estimator over {LAB[b]}, {nm} skill", A[f"skill_{kind}|c2|{b}"],
                ar.get(f"skill_{kind}|c2|{b}"), "b64_two_stage.json (r2-r4, old subset)")
    for a, b in (("t2", "t"), ("c", "t"), ("c", "off")):
        for kind, nm in (("mse", "MSE"), ("huber", "Huber"), ("trim", "1 %-trimmed")):
            add(f"Table main (a): {LAB[a]} over {LAB[b]}, {nm} skill", A[f"skill_{kind}|{a}|{b}"],
                ar.get(f"skill_{kind}|{a}|{b}"), "b64_two_stage.json (r2-r4, old subset)")
    for e in ("hold", "gp", "off", "pchip", "t", "c", "t2", "c2"):
        add(f"Table main (c): {LAB[e]}, RMSE (eV)", A[f"rmse|{e}"], ar.get(f"rmse|{e}"), "b64_two_stage.json", "eV")
        add(f"Table main (c): {LAB[e]}, median |e| (eV)", Q[f"median_eV|{e}"], md.get(f"median_eV|{e}"), "b64_two_stage.json", "eV")
        if e != "hold":
            add(f"Table main (c): {LAB[e]}, median |e| / hold", Q[f"median_ratio|{e}|hold"], md.get(f"median_ratio|{e}|hold"),
                "b64_two_stage.json", "ratio")
        add(f"Table main (c): {LAB[e]}, share within 20 %", A[f"share20|{e}"], ar.get(f"share20|{e}"), "b64_two_stage.json", "share")
        if e != "gp":
            add(f"Table main (c): {LAB[e]}, within 20 % minus causal GP", A.get(f"share20_diff|{e}|gp"),
                ar.get(f"share20_diff|{e}|gp"), "b64_two_stage.json", "diff")
    s64 = B64["statements"]
    for name, new, kind in (("T1", A["skill_mse|c2|c"], "skill"), ("T2", A["skill_mse|c2|gp"], "skill"),
                            ("T3", A["skill_mse|c2|t2"], "skill"), ("T4", Q["median_ratio|c2|hold"], "ratio"),
                            ("T5", A["share20_diff|c2|gp"], "diff"), ("T6", A["skill_mse|t2|t"], "skill")):
        add(f"B.64 {name} ({s64[name]['text']})", new, s64[name]["value"], "b64_two_stage.json statements", kind)
    d = A["rmse_diff_eV|c2|gp"]
    add("Macro TwoRmseAbs: RMSE of the causal GP minus that of the two-stage estimator (eV)", [-d[0], -d[2], -d[1]],
        [-B64["all_rows"]["rmse_diff_eV|c2|gp"][0], -B64["all_rows"]["rmse_diff_eV|c2|gp"][2],
         -B64["all_rows"]["rmse_diff_eV|c2|gp"][1]], "b64_two_stage.json", "eV")
    add("Median |e| (eV), hold (B.39 P, r2-r9)", Q["median_eV|hold"], [B39["median_abs_error_eV"]["hold"]] * 3,
        "b39_primary.json P (point only)", "eV")
    add("Median |e| (eV), network (B.39 P, r2-r9)", Q["median_eV|c"], [B39["median_abs_error_eV"]["causal variant"]] * 3,
        "b39_primary.json P (point only)", "eV")
    add("Median |e| (eV), twin (B.39 P, r2-r9)", Q["median_eV|t"], [B39["median_abs_error_eV"]["twin"]] * 3,
        "b39_primary.json P (point only)", "eV")
    st9, rf9, pr9 = B59["statements"], B59["reported"]["over_refs"], B59["reported"]["pairs"]
    for arm, key in (("L", "L1"), ("B", "B1"), ("E", "E1"), ("N", "N1")):
        add(f"Table carry: {LAB[arm]} over the twin ({key})", A[f"skill_mse|{arm}|t"], st9[key]["skill"], "b59_inputs.json")
        add(f"Table carry: {LAB[arm]} over the causal GP", A[f"skill_mse|{arm}|gp"], rf9[arm]["gp_causal"]["skill"],
            "b59_inputs.json")
    add("Table carry: network over L (L2)", A["skill_mse|c|L"], st9["L2"]["skill"], "b59_inputs.json")
    add("Table carry: network over B", A["skill_mse|c|B"], pr9["est_over_B"]["skill"], "b59_inputs.json")
    add("Table carry: network over N (N2)", A["skill_mse|c|N"], st9["N2"]["skill"], "b59_inputs.json")
    return rows


# ------------------------------------------------------------------------------------------ readout
def main():
    C = S2.ctx()
    D, infos = load_all()
    ests = MAIN + OTHER_EST
    A, Aper = S2.cell(D, lambda r: np.ones(len(D[r]["y"]), bool), ests, PAIRS + OTHER_PAIRS, rng_for("all"), nboot=N_BOOT)
    log("cell (all rows) done")
    Q = S2.medians(D, MAIN + ("X",), MED_PAIRS, rng_for("medians"), nboot=N_BOOT)
    log("medians done")
    An, _ = S2.cell(D, lambda r: np.ones(len(D[r]["y"]), bool), ("c2nan", "t2nan", "c", "t", "gp", "hold"), NAN_PAIRS,
                    rng_for("nan"), nboot=N_BOOT)
    mix, flags = mixture(C, D)
    keep = {r: ~flags[r] for r in REPEATS}
    M6, M6per = S2.cell(D, lambda r: keep[r], ("c2", "t2", "c", "t", "gp", "hold"),
                        [("c2", "gp"), ("c2", "t2"), ("c", "gp"), ("c", "t"), ("c2", "c"), ("c2", "hold")],
                        rng_for("D1-6"), nboot=N_BOOT)
    log("D1-6 done")
    B64, B39, a29, B59, B67, B68 = old_values()
    old64 = B64["all_rows"]
    st = {
        "D1-1": {"text": "two-stage estimator over the causal GP: MSE skill interval above zero",
                 "value": A["skill_mse|c2|gp"], "per_repeat": Aper["skill_mse|c2|gp"], "old_b64_r2_r4": old64["skill_mse|c2|gp"]},
        "D1-2": {"text": "two-stage estimator over the two-stage twin: MSE skill interval above zero",
                 "value": A["skill_mse|c2|t2"], "per_repeat": Aper["skill_mse|c2|t2"], "old_b64_r2_r4": old64["skill_mse|c2|t2"]},
        "D1-3": {"text": "network over the twin: MSE skill interval above zero",
                 "value": A["skill_mse|c|t"], "per_repeat": Aper["skill_mse|c|t"], "old_b64_r2_r4": old64["skill_mse|c|t"]},
        "D1-4": {"text": "network over the causal GP: MSE skill interval above zero",
                 "value": A["skill_mse|c|gp"], "per_repeat": Aper["skill_mse|c|gp"], "old_b64_r2_r4": old64["skill_mse|c|gp"]},
        "D1-5": {"text": "median absolute error, two-stage estimator over hold: interval of the ratio below 1",
                 "value": Q["median_ratio|c2|hold"], "old_b64_r2_r4": B64["medians"]["median_ratio|c2|hold"]},
        "D1-6": {"text": "D1-1 and D1-2 hold on the D1 rows without the rows the good-and-bad mixture flags",
                 "value_D1_1": M6["skill_mse|c2|gp"], "value_D1_2": M6["skill_mse|c2|t2"],
                 "per_repeat_D1_1": M6per["skill_mse|c2|gp"], "per_repeat_D1_2": M6per["skill_mse|c2|t2"]},
    }
    for k_ in ("D1-1", "D1-2", "D1-3", "D1-4"):
        s = st[k_]
        s["holds"] = bool(s["value"][1] > 0)
        s["expected"] = "holds; point within 0.01 of the r2-r4 value of B.64"
        s["point_minus_old"] = float(s["value"][0] - s["old_b64_r2_r4"][0])
        s["within_0.01_of_old"] = bool(abs(s["point_minus_old"]) <= TOL_EXPECT)
    s = st["D1-5"]
    s["holds"] = bool(s["value"][2] < 1)
    s["expected"] = "holds; point within 0.01 of the r2-r4 value of B.64"
    s["point_minus_old"] = float(s["value"][0] - s["old_b64_r2_r4"][0])
    s["within_0.01_of_old"] = bool(abs(s["point_minus_old"]) <= TOL_EXPECT)
    s = st["D1-6"]
    s["holds"] = bool(s["value_D1_1"][1] > 0 and s["value_D1_2"][1] > 0)
    s["expected"] = "holds; the mixture flags most values above 3 keV and about one to two thousand rows below them"
    # rows and discharges
    rows = {}
    for r in REPEATS:
        d = D[r]
        cnt = np.bincount(d["f"], minlength=d["n"])
        old_rows = sum(int(len(S2.B54.load_rows(r, k)["y"])) for k in range(K))
        rows[str(r)] = {"rows": int(len(d["y"])), "discharges": int(d["n"]), "rows_per_fold": d["fold_rows"],
                        "max_rows_one_discharge": int(cnt.max()), "median_rows_per_discharge": float(np.median(cnt)),
                        "old_rows_40000_cap": old_rows, "old_share_of_d1": old_rows / len(d["y"]),
                        "fallback_share_c2": float(d["fb_c2"].mean()), "first_stage_max_dev_eV": d["first_stage_dev_eV"]}
    universe = {str(r): sorted(set(n for k in range(K) for n in S2.manifest(r, k)["test_files"])) for r in REPEATS}
    names_scored = {str(r): sorted(C.names[i] for i in D[r]["u"]) for r in REPEATS}
    for r in REPEATS:
        rows[str(r)]["universe"] = len(universe[str(r)])
        rows[str(r)]["never_scored"] = sorted(set(universe[str(r)]) - set(names_scored[str(r)]))
    comp = comparisons(A, Q, st, B64, B39, a29, B59)
    other = {
        "B.67 G1: S over the network": (A["skill_mse|S|c"], B67["statements"]["G1"]["skill"]),
        "B.67 G2: S over the twin": (A["skill_mse|S|t"], B67["statements"]["G2"]["skill"]),
        "B.67 G3: A over the network": (A["skill_mse|A|c"], B67["statements"]["G3"]["skill"]),
        "B.68 X1: X over the network": (A["skill_mse|X|c"], B68["statements"]["X1"]["skill"]),
        "B.68 X2: X over the twin": (A["skill_mse|X|t"], B68["statements"]["X2"]["skill"]),
        "B.68 X3: median |e| of X over that of the causal GP": (Q["median_ratio|X|gp"], B68["statements"]["X3"]["ratio"]),
    }
    b59_st = {}
    for key, (a, b) in {"L1": ("L", "t"), "L2": ("c", "L"), "N1": ("N", "t"), "N2": ("c", "N"),
                        "E1": ("E", "t"), "B1": ("B", "t")}.items():
        v = A[f"skill_mse|{a}|{b}"]
        b59_st[key] = {"value": v, "old": B59["statements"][key]["skill"],
                       "holds": bool(-0.01 <= v[1] and v[2] <= 0.01) if key == "N2" else bool(v[1] > 0),
                       "old_holds": B59["statements"][key]["holds"]}
    recs = [json.loads((RB.new_dir(*x) / "b70_rescore.json").read_text(encoding="utf-8")) for x in RB.plan()]
    own = [q for q in recs if q.get("status") == "ok" and "seconds" in q]
    shared = [q for q in recs if q.get("status") == "ok" and "seconds_arm" in q]
    if len(own) + len(shared) != len(RB.plan()):
        raise SystemExit("FATAL: a re-scored run has no record")
    harness_s = {(q["repeat"], q["fold"]): q["seconds_harness_of_fold"] for q in shared}
    rescoring = {"own_script_runs": len(own), "own_script_process_seconds": float(sum(q["seconds"] for q in own)),
                 "own_script_seconds_median": float(np.median([q["seconds"] for q in own])) if own else None,
                 "shared_scorer_runs": len(shared), "shared_scorer_folds": len(harness_s),
                 "shared_scorer_harness_seconds": float(sum(harness_s.values())),
                 "shared_scorer_arm_seconds": float(sum(q["seconds_arm"] for q in shared)),
                 "shared_scorer_harness_seconds_median": float(np.median(list(harness_s.values()))) if harness_s else None,
                 "shared_scorer_arm_seconds_median": float(np.median([q["seconds_arm"] for q in shared])) if shared else None}
    o = {"registration": REG, "n_boot": N_BOOT, "seed": SEED, "repeats": list(REPEATS),
         "statements": st, "rows": rows, "all_rows": A, "all_rows_per_repeat": Aper, "medians": Q,
         "missing_inputs_by_the_boosted_model": An, "mixture": mix, "unflagged_rows": M6, "unflagged_rows_per_repeat": M6per,
         "main_text_numbers": comp, "b59_on_d1": b59_st,
         "b67_b68_on_d1": {k_: {"d1": v[0], "old": v[1]} for k_, v in other.items()},
         "second_stage_info": infos,
         "seconds_per_row_second_stage": {a: float(np.mean([v[a]["seconds_per_d1_row_second_stage"] for v in infos.values()]))
                                          for a in ("v2", "v2h")},
         "rescoring": rescoring,
         "runtime_readout_s": round(_time.time() - T0, 1)}
    OUT_JSON.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(o), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD) + " | " + ", ".join(f"{k_} {'holds' if v['holds'] else 'does not hold'}" for k_, v in st.items()))


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def e1(x):
    return f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"


def mse(x):
    return f"{x[0] ** 2:,.0f} [{x[1] ** 2:,.0f}, {x[2] ** 2:,.0f}]"


def fmt(kind, x):
    return {"skill": s3, "ratio": f3, "share": f3, "eV": e1, "diff": s3}[kind](x)


def se1(x):
    return f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"


def per(p):
    return ", ".join(f"r{r} {v:+.3f}" for r, v in p.items())


def render(o):
    A, Q, st, rows, mix = o["all_rows"], o["medians"], o["statements"], o["rows"], o["mixture"]
    L = ["# B.70 — one data definition (D1): phase 1", "",
         f"Registered: {REG} (with Addendum 1). CPU only; nothing trained except the registered refit of the B.64 second "
         "stage. T_i, cut population, re-partitions r2-r4 of the pooled cross-validation (15 test folds). Rows: every "
         "eligible row of `evaluate.build_clean_val_subset` (no random cap, `CES_MAX_VAL_SAMPLES=0`): T_i measured and "
         "withheld, the previous kept row carries T_i, every reference defined; every estimator of a table on the same rows. "
         "Skill of A over B = 1 - sum of squared errors of A / that of B. Point = mean over r2-r4; 95 % two-level interval "
         f"(a re-partition drawn, then its test discharges resampled), {o['n_boot']:,} draws. Checks: `docs/paper/b70_checks.md`.", "",
         "## Statements", "",
         "| statement | value [95 % interval] | per re-partition | verdict | old value (B.64, r2-r4, 40,000-sample subset) | "
         "difference | registered expectation |", "|---|---|---|---|---|---|---|"]
    for k_ in ("D1-1", "D1-2", "D1-3", "D1-4"):
        s = st[k_]
        L.append(f"| **{k_}** {s['text']} | {s3(s['value'])} | {per(s['per_repeat'])} | "
                 f"**{'holds' if s['holds'] else 'does not hold'}** | {s3(s['old_b64_r2_r4'])} | {s['point_minus_old']:+.3f} | "
                 f"{s['expected']} (within 0.01: {s['within_0.01_of_old']}) |")
    s = st["D1-5"]
    L.append(f"| **D1-5** {s['text']} | {f3(s['value'])} | - | **{'holds' if s['holds'] else 'does not hold'}** | "
             f"{f3(s['old_b64_r2_r4'])} | {s['point_minus_old']:+.3f} | {s['expected']} (within 0.01: {s['within_0.01_of_old']}) |")
    s = st["D1-6"]
    L.append(f"| **D1-6** {s['text']} | D1-1 form {s3(s['value_D1_1'])}; D1-2 form {s3(s['value_D1_2'])} | "
             f"D1-1 form {per(s['per_repeat_D1_1'])}; D1-2 form {per(s['per_repeat_D1_2'])} | "
             f"**{'holds' if s['holds'] else 'does not hold'}** | - | - | {s['expected']} |")
    L += ["", "## The six models and PCHIP on the D1 rows (r2-r4)", "",
          "| estimator | MSE (eV^2) | RMSE (eV) | median absolute error (eV) | median / hold | mean absolute error (eV) | "
          "share within 20 % |", "|---|---|---|---|---|---|---|"]
    for e in ("c2", "c", "t", "gp", "hold", "off", "pchip", "t2"):
        mr = Q.get(f"median_ratio|{e}|hold")
        L.append(f"| {LAB[e]} | {mse(A[f'rmse|{e}'])} | {e1(A[f'rmse|{e}'])} | {e1(Q[f'median_eV|{e}'])} | "
                 f"{f3(mr) if mr else '1'} | {e1(A[f'mae|{e}'])} | {f3(A[f'share20|{e}'])} |")
    L += ["", "MSE intervals are the squares of the RMSE bounds on the same draws. The two-stage twin is the reference of "
          "D1-2 and is listed last.", "",
          "## Paired comparisons (the pairs of B.64's tables, KFX excepted)", "",
          "| A over B | MSE skill | MSE reduction (%) | RMSE difference A - B (eV) | Huber skill (200 eV) | 1 %-trimmed skill | "
          "per re-partition (MSE skill) |", "|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        q = f"{a}|{b}"
        sk = A[f"skill_mse|{q}"]
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(sk)} | {100 * sk[0]:+.1f} [{100 * sk[1]:+.1f}, {100 * sk[2]:+.1f}] | "
                 f"{se1(A[f'rmse_diff_eV|{q}'])} | {s3(A[f'skill_huber|{q}'])} | {s3(A[f'skill_trim|{q}'])} | "
                 f"{per(o['all_rows_per_repeat'][f'skill_mse|{q}'])} |")
    L += ["", "Median absolute error ratios: " + "; ".join(
        f"{LAB[k_.split('|')[1]]} / {LAB[k_.split('|')[2]]} {f3(v)}" for k_, v in Q.items() if k_.startswith("median_ratio")) + ".", "",
          "## Every main-text number next to its old value", "",
          "Old values: as cited (source and treatment named). The network rows of the main table were cited on r2-r9; their "
          "r2-r4 values of B.64 are given on the next line. Difference = D1 point minus old point.", "",
          "| quantity | D1 | old | difference | old source |", "|---|---|---|---|---|"]
    for c in o["main_text_numbers"]:
        f = (lambda x: fmt(c["kind"], x))
        old = c["old"]
        olds = f(old) if old[1] != old[0] or old[2] != old[0] else (f"{old[0]:.1f}" if c["kind"] == "eV" else f"{old[0]:+.3f}")
        dif = f"{c['diff']:+.1f}" if c["kind"] == "eV" else f"{c['diff']:+.3f}"
        L.append(f"| {c['quantity']} | {f(c['d1'])} | {olds} | {dif} | {c['old_source']} |")
    L += ["", "B.64 verdicts on D1 (T1-T6 forms): " + "; ".join(
        f"{c['quantity'].split(' (')[0].replace('B.64 ', '')} "
        f"{'holds' if (c['d1'][2] < 1 if c['kind'] == 'ratio' else c['d1'][1] > 0) else 'does not hold'}"
        for c in o["main_text_numbers"] if c["quantity"].startswith("B.64 T")) + ".", "",
          "B.59 statements on D1: " + "; ".join(f"{k_} {s3(v['value'])} ({'holds' if v['holds'] else 'does not hold'}; old "
                                                f"{s3(v['old'])}, {'held' if v['old_holds'] else 'did not hold'})"
                                                for k_, v in o["b59_on_d1"].items()) + ".", "",
          "B.67 and B.68 on D1 (not yet in the text): " + "; ".join(
              f"{k_} {(f3 if 'median' in k_ else s3)(v['d1'])} (old {(f3 if 'median' in k_ else s3)(v['old'])})"
              for k_, v in o["b67_b68_on_d1"].items()) + ".", "",
          "## Scored rows and discharges", "",
          "| re-partition | D1 rows | rows per fold | discharges with rows | test discharges (universe) | never scored | "
          "largest rows of one discharge | median rows per discharge | old rows (40,000 cap) | old / D1 | rows where the "
          "two-stage estimator keeps its first stage |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, v in rows.items():
        L.append(f"| r{r} | {v['rows']:,} | {', '.join(f'{x:,}' for x in v['rows_per_fold'])} | {v['discharges']} | "
                 f"{v['universe']} | {', '.join(v['never_scored'])} | {v['max_rows_one_discharge']:,} | "
                 f"{v['median_rows_per_discharge']:.0f} | {v['old_rows_40000_cap']:,} | {v['old_share_of_d1']:.3f} | "
                 f"{v['fallback_share_c2']:.4f} |")
    mr = mix["mean_over_repeats"]
    L += ["", "## D1-6: the good-and-bad mixture", "",
          "Residual: ln T_i minus the median of ln T_i over the finite values of the 11 grid rows centred on the row, "
          "truncated at the block edges (Addendum 1, A3). Two zero-mean Gaussians and the bad fraction, fitted by EM on the "
          "residuals of each fold's training discharges; a row is flagged when its posterior bad probability exceeds 0.5. "
          "Scoring-only.", "",
          f"Fitted parameters over the 15 folds (cut grid): bad fraction {mix['fit_ranges']['cut']['w'][0]:.4f}-"
          f"{mix['fit_ranges']['cut']['w'][1]:.4f}, good sd {mix['fit_ranges']['cut']['s_g'][0]:.4f}-"
          f"{mix['fit_ranges']['cut']['s_g'][1]:.4f}, bad sd {mix['fit_ranges']['cut']['s_b'][0]:.3f}-"
          f"{mix['fit_ranges']['cut']['s_b'][1]:.3f} (ln units). Before the cut: bad fraction "
          f"{mix['fit_ranges']['before_cut']['w'][0]:.4f}-{mix['fit_ranges']['before_cut']['w'][1]:.4f}, good sd "
          f"{mix['fit_ranges']['before_cut']['s_g'][0]:.4f}-{mix['fit_ranges']['before_cut']['s_g'][1]:.4f}, bad sd "
          f"{mix['fit_ranges']['before_cut']['s_b'][0]:.3f}-{mix['fit_ranges']['before_cut']['s_b'][1]:.3f}.", "",
          "| quantity | r2 | r3 | r4 | mean |", "|---|---|---|---|---|"]
    keys = [("scored_rows", "D1 scored rows"), ("scored_rows_flagged", "scored rows flagged (cut grid)"),
            ("scored_flagged_share", "share of scored rows flagged"),
            ("scored_flagged_below_75eV", "flagged scored rows below 75 eV"),
            ("scored_flagged_above_3keV", "flagged scored rows above 3 keV (none exist after the cut)"),
            ("scored_flagged_single_frames", "flagged scored rows that are single isolated frames"),
            ("scored_rows_without_residual", "scored rows without a residual (fewer than 3 finite values in the window)"),
            ("test_values_cut", "measured values of the test discharges (cut grid)"),
            ("test_values_flagged_cut", "of them flagged"),
            ("inc_values", "measured values of the test discharges before the cut"),
            ("inc_values_flagged", "of them flagged (mixture run before the cut)"),
            ("inc_above_3keV", "values above 3 keV"), ("inc_above_3keV_flagged", "values above 3 keV flagged"),
            ("inc_share_above_3keV_flagged", "share of the values above 3 keV flagged"),
            ("inc_above_3keV_without_residual", "values above 3 keV without a residual"),
            ("inc_above_3keV_flagged_single_frames", "flagged values above 3 keV that are single isolated frames"),
            ("inc_flagged_at_or_below_3keV", "flagged values at or below 3 keV"),
            ("inc_flagged_below_75eV", "flagged values below 75 eV"),
            ("inc_flagged_single_frames", "flagged values that are single isolated frames")]
    for kk, lab in keys:
        vals = [mix["per_repeat"][str(r)][kk] for r in REPEATS]
        fm = (lambda x: f"{x:.4f}") if "share" in kk else (lambda x: f"{x:,.0f}")
        L.append(f"| {lab} | " + " | ".join(fm(x) for x in vals) + f" | {fm(mr[kk])} |")
    L += ["", f"Values above 3 keV on the inclusive grid (first and last rows excluded): "
          f"{mix['values_above_3keV_inclusive_grid']:,} (registration: {mix['registered_count']:,}).", "",
          "On the D1 rows without the flagged rows:", "", "| A over B | MSE skill | per re-partition |", "|---|---|---|"]
    for a, b in (("c2", "gp"), ("c2", "t2"), ("c", "gp"), ("c", "t"), ("c2", "c"), ("c2", "hold")):
        q = f"skill_mse|{a}|{b}"
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(o['unflagged_rows'][q])} | {per(o['unflagged_rows_per_repeat'][q])} |")
    N_ = o["missing_inputs_by_the_boosted_model"]
    L += ["", "## Reported beside (B.64 addendum 1): the second stage on every row, its own handling of missing inputs", "",
          f"Two-stage estimator over the network {s3(N_['skill_mse|c2nan|c'])}; over the causal GP {s3(N_['skill_mse|c2nan|gp'])}; "
          f"over the two-stage twin {s3(N_['skill_mse|c2nan|t2nan'])}; two-stage twin over the twin {s3(N_['skill_mse|t2nan|t'])}.",
          "", "## Run time (CPU)", ""]
    rs = o["rescoring"]
    L += [f"- Fold r2 f0, each arm by its own scoring script (Addendum 1, A1): {rs['own_script_runs']} runs, "
          f"{rs['own_script_process_seconds'] / 3600:.1f} process-hours (median {rs['own_script_seconds_median'] or 0:.0f} s per "
          "run; six at once on a machine on battery, see Addendum 1, A6).",
          f"- The other {rs['shared_scorer_folds']} folds, shared-reference scorer (A6): rows and references "
          f"{rs['shared_scorer_harness_seconds'] / 3600:.1f} process-hours (median {rs['shared_scorer_harness_seconds_median'] or 0:.0f} s "
          f"per fold), {rs['shared_scorer_runs']} network lookups {rs['shared_scorer_arm_seconds'] / 3600:.2f} process-hours "
          f"(median {rs['shared_scorer_arm_seconds_median'] or 0:.0f} s per arm).",
          "- Interruption: the machine (on battery) went to sleep at 19:48 and woke at 21:47; the three scoring processes "
          "died in the reference pass of r2 f4, r3 f4 and r4 f4. Nothing of those five unfinished folds had been written; "
          "they (r2 f4, r3 f0, r3 f4, r4 f0, r4 f4) were scored again from the start. Every other output was checked to "
          "load and to carry its run record before the relaunch.",
          f"- Second stage per D1 row: network {o['seconds_per_row_second_stage']['v2'] * 1e6:.1f} us, twin "
          f"{o['seconds_per_row_second_stage']['v2h'] * 1e6:.1f} us (batch prediction). Readout {o['runtime_readout_s']:.0f} s."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
