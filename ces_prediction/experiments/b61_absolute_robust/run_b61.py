"""B.61 parts A and B (PREREGISTRATION_B61.md): absolute errors, robust skills, sensitivity to the discharges, and the
overlap of the four file-level test sets.  Part C (dose of a beam interruption) is run elsewhere.

CPU only; no network is run.  Rows: the pooled rows of B.39 (row tables of B.54, r2 to r9) and the filter outputs of
B.56 (`data/.b56/r{r}_f{k}.npz`, r3 to r9; r2, their development set, reported apart).

    py ces_prediction/experiments/b61_absolute_robust/run_b61.py
Output: docs/paper/b61_absolute_robust.json / .md
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b54_pooled_rows"))
import run_b54 as B54  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_B61.md"
OUT_JSON = B54.PAPER / "b61_absolute_robust.json"
OUT_MD = B54.PAPER / "b61_absolute_robust.md"
B56 = B54.DATA / ".b56"
ALL = tuple(B54.REPEATS)                 # r2..r9
CONFIRM, DEV = tuple(range(3, 10)), (2,)
N_BOOT = 2000
HUBER = 200.0                            # eV
TRIM = 0.01
KS = (10, 32, 64)
SPLITS = (42, 1, 7, 123)
BASE5 = ("c", "t", "gp", "hold", "off")
LAB = {**B54.LAB, "kfx": "KFX", "kf0": "KF0"}
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def huber(e):
    """Huber loss of the error e (eV), threshold 200 eV: e^2 / 2 below, 200 (|e| - 100) above (continuous)."""
    a = np.abs(e)
    return np.where(a <= HUBER, 0.5 * a * a, HUBER * (a - 0.5 * HUBER))


# ------------------------------------------------------------------------------------------ data
def load_data():
    D = {r: B54.repeat_data(r) for r in ALL}
    rows_chk = {}
    for r in ALL:
        est = {"kfx": [], "kf0": []}
        for k in range(B54.K):
            z = np.load(B56 / f"r{r}_f{k}.npz")
            q = B54.load_rows(r, k)
            same = len(z["shot"]) == len(q["shot"]) and np.array_equal(z["shot"], q["shot"]) and np.array_equal(z["row"], q["row"])
            dy = float(np.max(np.abs(z["y"] - q["y"]))) if same else np.inf
            rows_chk[f"r{r}_f{k}"] = {"same_rows": bool(same), "max_abs_value_diff_eV": dy, "passes": bool(same and dy <= 1e-3)}
            for e in est:
                est[e].append(z[f"est_{e}"].astype(np.float64))
        for e in est:
            D[r]["se"][e] = (np.concatenate(est[e]) - D[r]["y"]) ** 2
    for r in ALL:
        d = D[r]
        d["keep_trim"] = d["se"]["hold"] < np.quantile(d["se"]["hold"], 1 - TRIM)   # run_b41: top = se_hold >= q99
    return D, rows_chk


# ------------------------------------------------------------------------------------------ part A engine
def sums_of(d, ests):
    """Per-discharge sums of a repeat for every estimator: squared error, absolute error, Huber loss, squared error
    on the trimmed rows; counts."""
    f, n = d["f"], d["n"]
    S = {"cnt": np.bincount(f, None, n).astype(float),
         "cnt_trim": np.bincount(f, d["keep_trim"].astype(float), n)}
    for e in ests:
        se = d["se"][e]
        ae = np.sqrt(se)
        S[f"se|{e}"] = np.bincount(f, se, n)
        S[f"ae|{e}"] = np.bincount(f, ae, n)
        S[f"hub|{e}"] = np.bincount(f, huber(ae), n)
        S[f"set|{e}"] = np.bincount(f, se * d["keep_trim"], n)
    return S


def stats(S, ests, pairs):
    """Quantities from summed values (scalars for a point, vectors over draws)."""
    o = {}
    for e in ests:
        o[f"rmse|{e}"] = np.sqrt(S[f"se|{e}"] / S["cnt"])
        o[f"mae|{e}"] = S[f"ae|{e}"] / S["cnt"]
        o[f"rmse_trim|{e}"] = np.sqrt(S[f"set|{e}"] / S["cnt_trim"])
    for e in ests:
        if e == "gp":
            continue
        for q in ("rmse", "mae"):
            o[f"{q}_minus_gp_eV|{e}"] = o[f"{q}|{e}"] - o[f"{q}|gp"]
            o[f"{q}_minus_gp_pct|{e}"] = 100.0 * (o[f"{q}|{e}"] / o[f"{q}|gp"] - 1.0)
    for a, b in pairs:
        o[f"skill_mse|{a}|{b}"] = 1 - S[f"se|{a}"] / S[f"se|{b}"]
        o[f"skill_huber|{a}|{b}"] = 1 - S[f"hub|{a}"] / S[f"hub|{b}"]
        o[f"skill_trim|{a}|{b}"] = 1 - S[f"set|{a}"] / S[f"set|{b}"]
    return o


def part_a_cell(D, reps, ests, pairs, rng):
    """Two-level bootstrap (a repeat drawn, then its discharges; run_b54.hier): every quantity on the same draws."""
    draws = rng.choice(np.array(reps), size=N_BOOT)
    boot, per = {}, {}
    for r in reps:
        S = sums_of(D[r], ests)
        per[r] = {k: float(v) for k, v in stats({k: v.sum() for k, v in S.items()}, ests, pairs).items()}
        sel = np.flatnonzero(draws == r)
        n = D[r]["n"]
        idx = rng.integers(0, n, size=(len(sel), n))
        W = np.zeros((len(sel), n))
        for j in range(len(sel)):
            W[j] = np.bincount(idx[j], minlength=n)
        st = stats({k: W @ v for k, v in S.items()}, ests, pairs)
        for k, v in st.items():
            boot.setdefault(k, np.empty(N_BOOT))[sel] = v
    return {k: [float(np.mean([per[r][k] for r in reps])), *B54.ci(boot[k])] for k in boot}, \
        {k: {str(r): per[r][k] for r in reps} for k in boot}


# ------------------------------------------------------------------------------------------ checks
def checks(D, rows_chk):
    b39 = json.loads((B54.PAPER / "b39_primary.json").read_text(encoding="utf-8"))["P"]
    b54 = json.loads((B54.PAPER / "b54_pooled_rows.json").read_text(encoding="utf-8"))["reported"]["abs_error_percentiles_eV_mean_over_repeats"]
    out = {}
    p1 = float(np.mean([1 - D[r]["se"]["c"].sum() / D[r]["se"]["gp"].sum() for r in ALL]))
    p2 = float(np.mean([1 - D[r]["se"]["c"].sum() / D[r]["se"]["t"].sum() for r in ALL]))
    out["pooled_skills"] = {"P1": {"mine": p1, "recorded": b39["P1_causal_over_gp"][0]},
                            "P2": {"mine": p2, "recorded": b39["P2_causal_over_twin"][0]}}
    for v in out["pooled_skills"].values():
        v["equal_3dp"] = bool(abs(v["mine"] - v["recorded"]) < 5e-4)
    pc = {}
    for e in BASE5:
        for p in (50, 75, 90, 95, 99):
            mine = float(np.mean([np.percentile(np.sqrt(D[r]["se"][e]), p) for r in ALL]))
            rec = b54[LAB[e]][str(p)]
            pc[f"{LAB[e]}|p{p}"] = {"mine": mine, "recorded": rec, "equal_1dp": bool(round(mine, 1) == round(rec, 1))}
    out["b54_percentiles"] = pc
    out["filter_rows"] = {"folds": len(rows_chk), "passing": int(sum(v["passes"] for v in rows_chk.values())),
                          "max_abs_value_diff_eV": float(max(v["max_abs_value_diff_eV"] for v in rows_chk.values()))}
    out["passes"] = bool(all(v["equal_3dp"] for v in out["pooled_skills"].values())
                         and all(v["equal_1dp"] for v in pc.values()) and out["filter_rows"]["passing"] == len(rows_chk))
    return out


# ------------------------------------------------------------------------------------------ part B
def part_b(D):
    res = {}
    for k in KS:
        keep, conc = {}, {}
        for r in ALL:
            d = D[r]
            red = np.bincount(d["f"], d["se"]["gp"] - d["se"]["c"], d["n"])
            o = np.argsort(red)[::-1]
            km = np.ones(d["n"], bool)
            km[o[:k]] = False
            keep[r] = km
            conc[str(r)] = float(red[o[:k]].sum() / red.sum())
        o = {"removed": k, "share_of_reduction_vs_gp_removed": conc,
             "share_mean": float(np.mean(list(conc.values())))}
        for ref in ("gp", "t", "hold"):
            num = {r: np.bincount(D[r]["f"], D[r]["se"]["c"], D[r]["n"])[keep[r]] for r in ALL}
            den = {r: np.bincount(D[r]["f"], D[r]["se"][ref], D[r]["n"])[keep[r]] for r in ALL}
            # same generator names as run_b54's 'without_top10' cell, so k = 10 reproduces B.54
            o[ref], o[ref + "_per_repeat"] = B54.hier(num, den, B54.SKILL, B54.B_SKILL, B54.rng_for(f"without_top{k}|{ref}"))
        o["remaining_discharges_per_repeat"] = {str(r): int(keep[r].sum()) for r in ALL}
        res[str(k)] = o
        log(f"B k={k}: over GP {o['gp']}, over twin {o['t']}")
    # overlap of the four file-level test sets
    sets = {s: set(json.loads((B54.DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json").read_text(encoding="utf-8"))["test_files"])
            for s in SPLITS}
    allf = set.union(*sets.values())
    cnt = {nm: sum(nm in v for v in sets.values()) for nm in allf}
    # the same from the stored runs (discharges with scored rows)
    stored = {}
    for s in SPLITS:
        z = np.load(B54.DATA / f".b39_v2_cut_s{s}_i{s}" / "comparison_errors_test.npz")
        stored[s] = set(np.unique(z["CES_TI_shot"]).tolist())
    scnt = {}
    for s, v in stored.items():
        for u in v:
            scnt[u] = scnt.get(u, 0) + 1
    universe = int(json.loads((B54.DATA / ".b27_split_r2_f0" / "split_manifest.json").read_text())["universe"])
    res["overlap"] = {
        "test_files_per_split": {str(s): len(v) for s, v in sets.items()},
        "discharges_in_any_test_set": len(allf),
        "by_number_of_splits": {str(j): int(sum(c == j for c in cnt.values())) for j in (1, 2, 3, 4)},
        "never_a_test_discharge_of_the_639": universe - len(allf),
        "stored_rows_by_number_of_splits": {str(j): int(sum(c == j for c in scnt.values())) for j in (1, 2, 3, 4)},
        "stored_discharges_in_any": len(scnt)}
    log(f"overlap {res['overlap']}")
    return res


# ------------------------------------------------------------------------------------------ main
def main():
    D, rows_chk = load_data()
    log("data loaded")
    ck = checks(D, rows_chk)
    log(f"check passes: {ck['passes']}; P1 {ck['pooled_skills']['P1']['mine']:.4f}, P2 {ck['pooled_skills']['P2']['mine']:.4f}")
    if not ck["passes"]:
        OUT_JSON.write_text(json.dumps({"check": ck, "stopped": True}, indent=1, default=float), encoding="utf-8")
        raise SystemExit("FATAL: the check fails; no statement is read")
    pairs5 = [("c", "gp"), ("c", "t"), ("c", "hold"), ("c", "off"), ("t", "gp")]
    A, Aper = part_a_cell(D, ALL, BASE5, pairs5, B54.rng_for("b61|A|r2-r9"))
    log("part A, r2-r9 done")
    ests7 = (*BASE5, "kf0", "kfx")
    pairs7 = pairs5 + [("kfx", "kf0"), ("kfx", "gp"), ("kf0", "gp"), ("kfx", "t"), ("kfx", "c"), ("kfx", "hold"), ("kfx", "off")]
    F, Fper = part_a_cell(D, CONFIRM, ests7, pairs7, B54.rng_for("b61|A|r3-r9"))
    log("part A, r3-r9 done")
    F2, _ = part_a_cell(D, DEV, ests7, pairs7, B54.rng_for("b61|A|r2"))
    B = part_b(D)
    st = {"A1": {"value": A["skill_huber|c|gp"], "text": "Huber skill (200 eV) of the estimator over the causal GP"},
          "A2": {"value": A["skill_huber|c|t"], "text": "Huber skill of the estimator over the twin"},
          "A3": {"value": A["skill_trim|c|gp"], "text": "trimmed skill (1 % of rows by hold error removed) over the causal GP"},
          "A4": {"value": A["skill_trim|c|t"], "text": "trimmed skill over the twin"},
          "A5": {"value": F["skill_huber|kfx|kf0"], "text": "Huber skill of KFX over KF0 (r3 to r9)"},
          "B1": {"value": B["64"]["gp"], "text": "the 64 discharges with the largest reduction removed: over the causal GP"},
          "B2": {"value": B["64"]["t"], "text": "the 64 removed: over the twin"}}
    for s in st.values():
        s["holds"] = bool(s["value"][1] > 0)
    E = []
    for text, val, lo, hi, key, want in (
            ("RMSE of the estimator 180 to 190 eV", A["rmse|c"][0], 180, 190, None, None),
            ("RMSE of the causal GP 4 to 5 % higher than the estimator's", 100 * (A["rmse|gp"][0] / A["rmse|c"][0] - 1), 4, 5, None, None),
            ("RMSE of the twin 2 to 3 % higher than the estimator's", 100 * (A["rmse|t"][0] / A["rmse|c"][0] - 1), 2, 3, None, None),
            ("MAE of the estimator within 1 % of the causal GP's", A["mae_minus_gp_pct|c"][0], -1, 1, None, None),
            ("A1 +0.01 to +0.04, holds", st["A1"]["value"][0], 0.01, 0.04, "A1", True),
            ("A2 +0.005 to +0.03, holds or fails by a small margin", st["A2"]["value"][0], 0.005, 0.03, "A2", None),
            ("A3 +0.02 to +0.07, holds", st["A3"]["value"][0], 0.02, 0.07, "A3", True),
            ("A4 +0.01 to +0.05, holds", st["A4"]["value"][0], 0.01, 0.05, "A4", True),
            ("A5 +0.02 to +0.05, holds", st["A5"]["value"][0], 0.02, 0.05, "A5", True),
            ("B1 +0.03 to +0.06, holds", st["B1"]["value"][0], 0.03, 0.06, "B1", True),
            ("B2 +0.01 to +0.04, holds", st["B2"]["value"][0], 0.01, 0.04, "B2", True)):
        E.append({"expectation": text, "observed": f"{val:+.4f}", "range_met": bool(lo <= val <= hi),
                  "verdict": None if key is None else ("holds" if st[key]["holds"] else "does not hold"),
                  "verdict_met": None if want is None else bool(st[key]["holds"] == want)})
    out = {"registration": REG, "parts": "A and B (part C run separately)", "n_boot": N_BOOT, "huber_threshold_eV": HUBER,
           "trim_share": TRIM, "check": ck, "statements": st,
           "part_a_r2_r9": A, "part_a_r2_r9_per_repeat": Aper, "part_a_r3_r9_with_filters": F,
           "part_a_r3_r9_per_repeat": Fper, "part_a_r2_filters_development": F2, "part_b": B, "expectations": E,
           "rows_trimmed_per_repeat": {str(r): int((~D[r]["keep_trim"]).sum()) for r in ALL},
           "runtime_s": round(_time.time() - T0, 1)}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(out), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD))


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def e1(x):
    return f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"


def d1(x):
    return f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"


def abs_table(L, A, ests):
    L += ["| estimator | RMSE (eV) | RMSE minus causal GP (eV) | RMSE vs causal GP (%) | MAE (eV) | MAE minus causal GP (eV) | MAE vs causal GP (%) | RMSE, trimmed rows (eV) |",
          "|---|---|---|---|---|---|---|---|"]
    for e in ests:
        if e == "gp":
            L.append(f"| {LAB[e]} | {e1(A['rmse|gp'])} | | | {e1(A['mae|gp'])} | | | {e1(A['rmse_trim|gp'])} |")
            continue
        L.append(f"| {LAB[e]} | {e1(A[f'rmse|{e}'])} | {d1(A[f'rmse_minus_gp_eV|{e}'])} | {d1(A[f'rmse_minus_gp_pct|{e}'])} | "
                 f"{e1(A[f'mae|{e}'])} | {d1(A[f'mae_minus_gp_eV|{e}'])} | {d1(A[f'mae_minus_gp_pct|{e}'])} | {e1(A[f'rmse_trim|{e}'])} |")


def skill_table(L, A, pairs):
    L += ["| A over R | MSE skill | Huber skill (200 eV) | trimmed skill |", "|---|---|---|---|"]
    for a, b in pairs:
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(A[f'skill_mse|{a}|{b}'])} | {s3(A[f'skill_huber|{a}|{b}'])} | {s3(A[f'skill_trim|{a}|{b}'])} |")


def render(o):
    ck, st, A, F, F2, B = o["check"], o["statements"], o["part_a_r2_r9"], o["part_a_r3_r9_with_filters"], o["part_a_r2_filters_development"], o["part_b"]
    L = [f"# B.61 parts A and B -- absolute errors, robust skills, sensitivity to the discharges (registered: {o['registration']})", "",
         "CPU, no network run. T_i, cut population. Rows: the pooled rows of B.39 (80 runs, re-partitions r2 to r9, row "
         "tables of B.54) for the estimator, the twin, the causal GP, hold and the offline GP; the filters KF0 and KFX of "
         "B.56 on r3 to r9 (r2, their development set, reported apart), with every other estimator recomputed on those "
         "rows. Intervals: 95 %, 2,000 draws of the two-level bootstrap (a re-partition drawn, then its discharges "
         "resampled); every quantity and every difference of a table on the same draws. Point = mean over re-partitions of "
         "the per-re-partition value. Huber loss with threshold 200 eV: e^2/2 for |e| <= 200 eV, 200 (|e| - 100) above. "
         "Trimmed: within each re-partition the rows whose squared error of hold is at or above its 99th percentile are "
         "removed (the same rows for every estimator). Part C of the registration is run separately.", "",
         "## Check before the readout", ""]
    ps = ck["pooled_skills"]
    L += [f"- Pooled skills of B.39: P1 {ps['P1']['mine']:+.4f} (recorded {ps['P1']['recorded']:+.4f}), P2 {ps['P2']['mine']:+.4f} "
          f"(recorded {ps['P2']['recorded']:+.4f}); equal to three decimals: {'yes' if all(v['equal_3dp'] for v in ps.values()) else 'NO'}.",
          f"- Percentiles of the absolute error of B.54 (five estimators, p50 to p99, 25 values): "
          f"{sum(v['equal_1dp'] for v in ck['b54_percentiles'].values())} of {len(ck['b54_percentiles'])} equal to one decimal "
          f"(largest difference {max(abs(v['mine'] - v['recorded']) for v in ck['b54_percentiles'].values()):.1e} eV).",
          f"- Filter rows (B.56 cache) identical to the B.54 rows with the same reported values in {ck['filter_rows']['passing']} of "
          f"{ck['filter_rows']['folds']} folds (largest value difference {ck['filter_rows']['max_abs_value_diff_eV']:.1e} eV).",
          f"- The check {'passes' if ck['passes'] else 'FAILS'}.", "",
          "## Statements", "", "| statement | point [95 % interval] | reference | verdict |", "|---|---|---|---|"]
    for k, s in st.items():
        L.append(f"| **{k}**: {s['text']} | {s3(s['value'])} | 0 | **{'holds' if s['holds'] else 'does not hold'}** |")
    L += ["", "## Part A: absolute errors (pooled rows, r2 to r9)", ""]
    abs_table(L, A, ("c", "t", "gp", "hold", "off"))
    L += ["", "Robust skills (r2 to r9):", ""]
    skill_table(L, A, [("c", "gp"), ("c", "t"), ("c", "hold"), ("c", "off"), ("t", "gp")])
    L += ["", f"Rows trimmed per re-partition: " + ", ".join(f"r{r} {v:,}" for r, v in o["rows_trimmed_per_repeat"].items()) + ".", "",
          "## Part A with the filters (r3 to r9, every estimator on the same rows)", ""]
    abs_table(L, F, ("c", "t", "gp", "hold", "off", "kf0", "kfx"))
    L += [""]
    skill_table(L, F, [("kfx", "kf0"), ("kfx", "gp"), ("kf0", "gp"), ("kfx", "t"), ("kfx", "c"), ("kfx", "hold"), ("kfx", "off"),
                       ("c", "gp"), ("c", "t")])
    L += ["", "r2 (development set of the filters, reported apart; interval over its discharges only): RMSE KFX "
          f"{e1(F2['rmse|kfx'])} eV, KF0 {e1(F2['rmse|kf0'])}, causal GP {e1(F2['rmse|gp'])}; Huber skill KFX over KF0 "
          f"{s3(F2['skill_huber|kfx|kf0'])}, KFX over the causal GP {s3(F2['skill_huber|kfx|gp'])}; trimmed skill KFX over KF0 "
          f"{s3(F2['skill_trim|kfx|kf0'])}.", "",
          "## Part B: discharges", "",
          "| removed (largest reduction against the causal GP, per re-partition) | share of the reduction they hold | over causal GP | over twin | over hold |",
          "|---|---|---|---|---|"]
    for k, v in B.items():
        if k == "overlap":
            continue
        L.append(f"| {k} | {v['share_mean']:.3f} | {s3(v['gp'])} | {s3(v['t'])} | {s3(v['hold'])} |")
    ov = B["overlap"]
    L += ["", "Part B skills: run_b54.hier, 20,000 draws, resampling the remaining discharges only (the registration's 'pooled "
          "interval on the remaining discharges'). The k = 10 point equals B.54's; B.54's interval resampled every "
          "discharge with the removed ones contributing nothing, so its interval differs slightly (B.54: over the causal "
          "GP [+0.050, +0.098], over the twin [+0.016, +0.064]). Remaining discharges per re-partition for k = 64: "
          + ", ".join(f"r{r} {v}" for r, v in B["64"]["remaining_discharges_per_repeat"].items()) + ".", "",
          "Overlap of the test sets of the four file-level splits (42, 1, 7, 123; `split_manifest.json`): test discharges per "
          f"split {', '.join(f'{s}: {v}' for s, v in ov['test_files_per_split'].items())}; {ov['discharges_in_any_test_set']} "
          "distinct discharges are a test discharge of at least one split, of them "
          + ", ".join(f"{v} in exactly {j}" for j, v in ov["by_number_of_splits"].items())
          + f" of the four; {ov['never_a_test_discharge_of_the_639']} of the 639 are a test discharge of none. Counted from the "
          f"stored scored rows of the four estimator runs instead: {ov['stored_discharges_in_any']} discharges, "
          + ", ".join(f"{v} in exactly {j}" for j, v in ov["stored_rows_by_number_of_splits"].items()) + ".", "",
          "## Expectations of the registration (section 6) beside the findings", "",
          "| expectation | observed | range met | verdict | verdict met |", "|---|---|---|---|---|"]
    yn = lambda x: "" if x is None else ("yes" if x else "no")                    # noqa: E731
    for e in o["expectations"]:
        L.append(f"| {e['expectation']} | {e['observed']} | {yn(e['range_met'])} | {e['verdict'] or ''} | {yn(e['verdict_met'])} |")
    L += ["", "## Choices made where the registration left room", "",
          "- Huber loss in its usual form, e^2/2 below the threshold and 200 (|e| - 100) above, so that it is continuous; the "
          "skill is a ratio, so the factor 1/2 does not change it.",
          "- Trimming: the 1 % of rows with the largest squared error of hold is taken as the rows at or above the 99th "
          "percentile of that error within each re-partition (the rule of run_b41 for 'top 1 % of the hold error'); on the "
          "filter rows (r3 to r9) the same rule within each re-partition.",
          "- The five estimators are read on r2 to r9 (the statements A1 to A4); the filters and A5 on r3 to r9, where every "
          "estimator is recomputed on the same rows; r2 for the filters reported apart.",
          "- Differences to the causal GP: in eV, RMSE_A - RMSE_GP; in per cent, 100 (RMSE_A / RMSE_GP - 1); likewise for "
          "the MAE; both within a draw.",
          "- Part B: per re-partition the discharges are ranked by their summed squared-error reduction against the causal GP "
          "(as B.54) and the top k removed for every reference; skills with run_b54.hier (20,000 draws) on the remaining "
          "discharges.",
          "- Overlap: from the test_files of the four split manifests (`data/.b1_w2cut_split_s{seed}`); the count from the "
          "stored runs is given beside it.",
          "- Generators: run_b54.rng_for (default_rng([20261154, crc32(name)])).", "",
          "## Limits", "", "The Huber threshold and the trimming share are choices fixed before the readout. Removing "
          "discharges by their reduction selects on the outcome and gives a lower bound.", "",
          f"Run time: {o['runtime_s']:.0f} s."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
