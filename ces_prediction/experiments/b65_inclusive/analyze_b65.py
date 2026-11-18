"""B.65 readout (PREREGISTRATION_B65.md): statements I1..I4 on the inclusive population, r2..r4 pooled.

CPU; reads the 30 runs data/.b65_{v2,v2h}_incl_r{r}_f{k} and the CPU references data/.b65_refs_incl_r{r}_f{k}
(refs_b65.py).  Interval: the two-level bootstrap of analyze_b39.arm_p (a re-partition drawn uniformly, then its
discharges resampled with replacement), 20,000 draws, every quantity on the same draws.  Huber loss (200 eV) and the
1 % trimmed skill (rows with the top 1 % of the hold squared error of a re-partition removed) as in run_b61.py.
Every quantity also on the rows with a value at or below 3 keV.

Trimming: the top 1 % of the hold squared error is found within each re-partition and within the rows of the cell
(all inclusive rows, or the rows at or below 3 keV), so each cell is trimmed as a population of its own.

    py ces_prediction/experiments/b65_inclusive/analyze_b65.py
    py ces_prediction/experiments/b65_inclusive/analyze_b65.py --validate-cut   # engine test, no B.65 run needed
Output: docs/paper/b65_inclusive.{json,md}

--validate-cut runs the same engine on the cut population (the stored B.39 runs .b39p_{v2,v2h}_cut_r{r}_f{k} and the
CPU references .b65_refs_cut_*) and requires the per-re-partition MSE, Huber and trimmed skills, RMSE and MAE to equal
those of B.61 (docs/paper/b61_absolute_robust.json, part A) and the MSE skill over the twin those of B.59; it writes
only data/.b65_validate_cut.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b61_absolute_robust"))
import run_b65 as R65  # noqa: E402
from run_b65 import B27, B39, DATA  # noqa: E402
import run_b61 as B61  # noqa: E402  (huber)

REG = "ces_prediction/experiments/PREREGISTRATION_B65.md"
PAPER = DATA.parent / "docs" / "paper"
OUT_JSON = PAPER / "b65_inclusive.json"
OUT_MD = PAPER / "b65_inclusive.md"
T = "CES_TI"
REPS = R65.REPEATS
NBOOT = 20_000
SEED = 20261165
TRIM = 0.01
CUT = 3000.0
EST = ("c", "t", "gp", "hold", "off", "pchip")
LAB = {"c": "network", "t": "twin", "gp": "causal GP", "hold": "hold", "off": "offline GP", "pchip": "PCHIP"}
SRC = {"gp": "gp_causal", "hold": "persistence", "off": "gp", "pchip": "pchip"}
PAIRS = [("c", "gp"), ("c", "t"), ("c", "hold"), ("c", "off"), ("c", "pchip"), ("t", "gp")]
T0 = _time.time()


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def load(d):
    z = np.load(d / "comparison_errors_test.npz")
    o = {"shot": z[f"{T}_shot"], "row": z[f"{T}_row"], "y": z[f"{T}_y_true"].astype(np.float64),
         "model": z[f"{T}_se_model"].astype(np.float64)}
    for k, q in SRC.items():
        o[k] = z[f"{T}_se_{q}"].astype(np.float64)
    return o


def net_dir(pop, model, r, k):
    return R65.run_dir(model, r, k) if pop == "incl" else B39.pooled_run_dir(model, r, k)


def env_check(pop, r, k):
    """The env each B.65 run recorded (b65_env.json, written by run_b65.execute) equals the registered one
    (run_b65.run_env).  Not applicable to the stored B.39 runs (validation)."""
    if pop != "incl":
        return True
    for m in ("v2", "v2h"):
        f = net_dir(pop, m, r, k) / "b65_env.json"
        if not f.exists():
            return False
        if json.loads(f.read_text(encoding="utf-8")) != R65.ces_vars(R65.run_env(m, r, k)):
            return False
    return True


def load_all(pop="incl"):
    chk = {"population": pop, "folds": {}}
    ok = True
    refchk = json.loads((DATA / ".b65_refs_checks.json").read_text(encoding="utf-8"))
    chk["refs_cut_reproduce_b39"] = bool(refchk["passes"])
    ok &= chk["refs_cut_reproduce_b39"]
    D = {}
    for r in REPS:
        parts = []
        for k in range(B27.K):
            c, t = load(net_dir(pop, "v2", r, k)), load(net_dir(pop, "v2h", r, k))
            ref = load(DATA / f".b65_refs_{pop}_r{r}_f{k}")
            same = all(len(x["shot"]) == len(c["shot"]) and np.array_equal(x["shot"], c["shot"]) and
                       np.array_equal(x["row"], c["row"]) for x in (t, ref))
            refs_eq = same and all(np.array_equal(c[q], ref[q]) and np.array_equal(t[q], ref[q]) for q in SRC)
            m1 = json.loads((net_dir(pop, "v2", r, k) / "metrics.json").read_text(encoding="utf-8"))
            m2 = json.loads((net_dir(pop, "v2h", r, k) / "metrics.json").read_text(encoding="utf-8"))
            overlap = R65.overlap_check(r, k)
            rec = {"same_rows_net_twin_refs": bool(same), "refs_equal_cpu_refs": bool(refs_eq),
                   "models": [m1["seq_model"], m2["seq_model"]], "rows": int(len(c["y"])),
                   "rows_above_3keV": int((c["y"] > CUT).sum()), "split_sizes": overlap,
                   "epochs_run_best": [[m1["epochs_run"], m1["best_epoch"]], [m2["epochs_run"], m2["best_epoch"]]]}
            rec["env_recorded_matches_registration"] = bool(env_check(pop, r, k))
            rec["passes"] = bool(same and refs_eq and rec["models"] == ["v2", "v2h"] and
                                 rec["env_recorded_matches_registration"])
            ok &= rec["passes"]
            chk["folds"][f"r{r}f{k}"] = rec
            parts.append({"shot": c["shot"], "y": c["y"], "c": c["model"], "t": t["model"],
                          **{q: ref[q] for q in SRC}})
        d = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        u, f = np.unique(d["shot"], return_inverse=True)
        d["f"], d["n"] = f, len(u)
        D[r] = d
    chk["passes"] = bool(ok)
    return D, chk


def sums_of(d, m):
    """Per-discharge sums on the rows m; the trimmed rows are found within m (run_b61: top = se_hold >= q99)."""
    f, n = d["f"][m], d["n"]
    hold = d["hold"][m]
    keep_trim = hold < np.quantile(hold, 1 - TRIM)
    S = {"cnt": np.bincount(f, None, n).astype(float), "cnt_trim": np.bincount(f, keep_trim.astype(float), n)}
    for e in EST:
        se = d[e][m]
        ae = np.sqrt(se)
        S[f"se|{e}"] = np.bincount(f, se, n)
        S[f"ae|{e}"] = np.bincount(f, ae, n)
        S[f"hub|{e}"] = np.bincount(f, B61.huber(ae), n)
        S[f"set|{e}"] = np.bincount(f, se * keep_trim, n)
    return S


def stats(S):
    o = {}
    for e in EST:
        o[f"rmse|{e}"] = np.sqrt(S[f"se|{e}"] / S["cnt"])
        o[f"mae|{e}"] = S[f"ae|{e}"] / S["cnt"]
        o[f"rmse_trim|{e}"] = np.sqrt(S[f"set|{e}"] / S["cnt_trim"])
    for a, b in PAIRS:
        o[f"mse|{a}|{b}"] = 1 - S[f"se|{a}"] / S[f"se|{b}"]
        o[f"huber|{a}|{b}"] = 1 - S[f"hub|{a}"] / S[f"hub|{b}"]
        o[f"trim|{a}|{b}"] = 1 - S[f"set|{a}"] / S[f"set|{b}"]
    return o


def cell(D, mask_fn, rng):
    draws = rng.choice(np.array(REPS), size=NBOOT)
    boot, per, med = {}, {}, {}
    for r in REPS:
        m = mask_fn(D[r])
        S = sums_of(D[r], m)
        per[r] = {k: float(v) for k, v in stats({k: v.sum() for k, v in S.items()}).items()}
        med[r] = {e: float(np.median(np.sqrt(D[r][e][m]))) for e in EST}
        sel = np.flatnonzero(draws == r)
        n = D[r]["n"]
        for c0 in range(0, len(sel), 1000):
            ss = sel[c0:c0 + 1000]
            idx = rng.integers(0, n, size=(len(ss), n))
            W = np.zeros((len(ss), n))
            for j in range(len(ss)):
                W[j] = np.bincount(idx[j], minlength=n)
            st = stats({k: W @ v for k, v in S.items()})
            for k, v in st.items():
                boot.setdefault(k, np.empty(NBOOT))[ss] = v
    out = {k: [float(np.mean([per[r][k] for r in REPS])), *ci(boot[k])] for k in boot}
    out["per_repeat"] = {k: {str(r): per[r][k] for r in REPS} for k in boot}
    out["median_abs_eV"] = {e: float(np.mean([med[r][e] for r in REPS])) for e in EST}
    out["rows_per_repeat"] = {str(r): int(mask_fn(D[r]).sum()) for r in REPS}
    out["rows_above_3keV_per_repeat"] = {str(r): int((D[r]["y"][mask_fn(D[r])] > CUT).sum()) for r in REPS}
    out["discharges_per_repeat"] = {str(r): int(D[r]["n"]) for r in REPS}
    return out


VAL_TOL = 1e-9
B59_OVER_TWIN = {"2": 0.045, "3": 0.051, "4": 0.062}   # docs/paper/b59_inputs.md, estimator over the twin (3 decimals)


def validate_cut():
    """Engine test on the cut population: per-re-partition quantities must equal those B.61 and B.59 stored."""
    D, chk = load_all("cut")
    if not chk["passes"]:
        raise SystemExit(f"FATAL: validation checks fail: {json.dumps(chk)[:800]}")
    rng = np.random.default_rng(SEED)
    A = cell(D, lambda d: np.ones(len(d["y"]), bool), rng)
    b61 = json.loads((PAPER / "b61_absolute_robust.json").read_text(encoding="utf-8"))["part_a_r2_r9_per_repeat"]
    cmp, worst = {}, 0.0
    for a, b in PAIRS:
        for q in ("mse", "huber", "trim"):
            k61 = f"skill_{q}|{a}|{b}"
            if k61 not in b61:
                continue
            for r in REPS:
                mine, ref = A["per_repeat"][f"{q}|{a}|{b}"][str(r)], b61[k61][str(r)]
                cmp[f"{q}|{a}|{b}|r{r}"] = [mine, ref]
                worst = max(worst, abs(mine - ref))
    for e in EST:
        for q in ("rmse", "mae", "rmse_trim"):
            k61 = f"{q}|{e}"
            if k61 not in b61:
                continue
            for r in REPS:
                mine, ref = A["per_repeat"][k61][str(r)], b61[k61][str(r)]
                cmp[f"{k61}|r{r}"] = [mine, ref]
                worst = max(worst, abs(mine - ref) / max(abs(ref), 1.0))
    b59_ok = all(abs(A["per_repeat"]["mse|c|t"][r] - v) < 5e-4 for r, v in B59_OVER_TWIN.items())
    out = {"validation": "engine on the cut population vs B.61 part A and B.59", "max_diff_vs_b61": worst,
           "compared": len(cmp), "passes_b61": bool(worst < VAL_TOL and len(cmp) > 0), "passes_b59": bool(b59_ok),
           "pooled_r2_r4": {k: A[k] for k in ("mse|c|gp", "mse|c|t", "huber|c|gp", "huber|c|t", "trim|c|gp", "trim|c|t")},
           "b59_estimator_over_twin_r2_r4": "+0.053 [+0.027, +0.076] (B.59, other draw seed)", "pairs": cmp}
    (DATA / ".b65_validate_cut.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (DATA / ".b65_validate_cut.md").write_text(render({"registration": REG, "statements": {}, "expectations": {},
                                                        "all_rows": A}, title_pop="cut"),
                                               encoding="utf-8", newline="\n")
    log(f"validation: {len(cmp)} per-re-partition quantities vs B.61, max diff {worst:.2e} -> "
        f"{'equal' if out['passes_b61'] else 'DIFFERENT'}; MSE skill over the twin vs B.59 per re-partition: "
        f"{'equal to 3 decimals' if b59_ok else 'DIFFERENT'}")
    log(f"pooled r2-r4 (cut): MSE over causal GP {s3(A['mse|c|gp'])}, over twin {s3(A['mse|c|t'])} "
        f"(B.59: +0.053 [+0.027, +0.076]); Huber over causal GP {s3(A['huber|c|gp'])}, over twin {s3(A['huber|c|t'])}")
    if not (out["passes_b61"] and b59_ok):
        raise SystemExit("FATAL: the engine does not reproduce B.61 / B.59 on the cut population")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-cut", action="store_true", help="engine test on the cut population (stored B.39 runs)")
    if ap.parse_args().validate_cut:
        validate_cut()
        return
    missing = [R65.run_dir(m, r, k).name for r in REPS for k in range(B27.K) for m in ("v2", "v2h")
               if not (R65.run_dir(m, r, k) / "comparison_errors_test.npz").exists()]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} of 30 B.65 runs have no test errors yet (first: {missing[:3]}); "
                         "nothing is read")
    D, chk = load_all()
    if not chk["passes"]:
        (PAPER / "b65_inclusive_checks.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
        raise SystemExit("FATAL: a check fails; no statement is read (docs/paper/b65_inclusive_checks.json)")
    log("checks pass")
    rng = np.random.default_rng(SEED)
    A = cell(D, lambda d: np.ones(len(d["y"]), bool), rng)
    L3 = cell(D, lambda d: d["y"] <= CUT, rng)
    st = {"I1": ("network over causal GP, MSE skill", A["mse|c|gp"]),
          "I2": ("network over twin, MSE skill", A["mse|c|t"]),
          "I3": ("network over causal GP, Huber skill", A["huber|c|gp"]),
          "I4": ("network over twin, Huber skill", A["huber|c|t"])}
    st = {k: {"text": t, "value": v, "holds": bool(v[1] > 0)} for k, (t, v) in st.items()}
    exp = {"I1": "+0.00 to +0.05, uncertain", "I2": "+0.00 to +0.04, uncertain", "I3": "+0.04 to +0.08, met",
           "I4": "+0.02 to +0.05, met"}
    out = {"registration": REG, "seed": SEED, "n_boot": NBOOT, "huber_eV": B61.HUBER, "trim": TRIM, "checks": chk,
           "statements": st, "expectations": exp, "all_rows": A, "rows_at_or_below_3keV": L3}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    OUT_MD.write_text(render(out), encoding="utf-8", newline="\n")
    for k, v in st.items():
        log(f"{k} {v['text']}: {s3(v['value'])} {'holds' if v['holds'] else 'does not hold'}")
    log("written " + str(OUT_MD))


def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def render(o, title_pop="incl"):
    head = ("# B.65 -- the pooled result on the population that keeps the values above 3 keV" if title_pop == "incl" else
            "# B.65 engine validation on the cut population (stored B.39 runs) -- not a B.65 result")
    L = [head, "",
         f"Registration: `{REG}`. Re-partitions r2 to r4 (15 folds), network `v2` and twin `v2h` trained and scored with "
         "CES_TI_SPIKE_CUT_EV=0; references by the B.39 code on the same rows. Interval: re-partition drawn, then "
         f"discharges resampled, {NBOOT:,} draws.", "", "## Statements", "",
         "| statement | value [95 %] | holds | expectation |", "|---|---|---|---|"]
    for k, v in o["statements"].items():
        L.append(f"| **{k}**: {v['text']} | {s3(v['value'])} | {'holds' if v['holds'] else 'does not hold'} | {o['expectations'][k]} |")
    for name, key in (("All inclusive rows", "all_rows"), ("Rows with a value at or below 3 keV", "rows_at_or_below_3keV")):
        if key not in o:
            continue
        A = o[key]
        L += ["", f"## {name}", "", f"Rows per re-partition: {', '.join(f'r{r} {v:,}' for r, v in A['rows_per_repeat'].items())}; "
              f"of them above 3 keV: {', '.join(f'r{r} {v:,}' for r, v in A['rows_above_3keV_per_repeat'].items())}; "
              f"discharges: {', '.join(f'r{r} {v}' for r, v in A['discharges_per_repeat'].items())}.", "",
              "| A over R | MSE skill | Huber skill (200 eV) | 1 % trimmed skill |", "|---|---|---|---|"]
        for a, b in PAIRS:
            L.append(f"| {LAB[a]} over {LAB[b]} | {s3(A[f'mse|{a}|{b}'])} | {s3(A[f'huber|{a}|{b}'])} | {s3(A[f'trim|{a}|{b}'])} |")
        L += ["", "Per re-partition (r2, r3, r4): " + "; ".join(
            f"{k} " + ", ".join(f"{v:+.3f}" for v in A["per_repeat"][k].values())
            for k in ("mse|c|gp", "mse|c|t", "huber|c|gp", "huber|c|t")) + ".", "",
              "| estimator | RMSE (eV) [95 %] | mean abs. error (eV) | RMSE, trimmed rows (eV) | median abs. error (eV) |",
              "|---|---|---|---|---|"]
        for e in EST:
            rm = A[f"rmse|{e}"]
            L.append(f"| {LAB[e]} | {rm[0]:.1f} [{rm[1]:.1f}, {rm[2]:.1f}] | {A[f'mae|{e}'][0]:.1f} | "
                     f"{A[f'rmse_trim|{e}'][0]:.1f} | {A['median_abs_eV'][e]:.1f} |")
        L += ["", "Point values: mean over the three re-partitions; the median absolute error is the mean of the three "
              "per-re-partition medians (no interval)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
