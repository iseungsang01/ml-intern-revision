"""B.84 checks (registration section 6, Addendum 1 section 4).  Each check is run once; its result is written to
data/.b84/checks/{name}.json and a second run of a check refuses.  No error, MSE or skill is computed here.

    py ces_prediction/experiments/b84_onefit/checks84.py reuse|mixture|raw|rows
    py ces_prediction/experiments/b84_onefit/checks84.py md          # docs/paper/b84_checks.md from the four results
"""
from __future__ import annotations

import json
import sys
import warnings

import common84 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
K75 = M.K75
UNIT = "V1_r2_k0"


def guard(name):
    p = M.CHK / f"{name}.json"
    if p.exists():
        raise SystemExit(f"check {name} was already run: {p}")
    return p


def unit_ctx():
    ctx = M.Ctx()
    us, _, _ = M.units()
    u = next(x for x in us if x["name"] == UNIT)
    return ctx, u, M.UnitArrays(ctx, u)


# ------------------------------------------------------------------------------------------ reuse
def check_reuse():
    p = guard("reuse")
    ctx, u, ua = unit_ctx()
    rm = M.RuleMats(ua, "D650", need_raw=False)
    fr, te = ctx.rows_of(u, "D650", "fit"), ctx.rows_of(u, "D650", "test")
    out = {"unit": UNIT, "rule": "D650", "target": "L5", "fit_rows": int(len(fr)), "test_rows": int(len(te))}
    ok = True
    for L in ("T0", "TX"):
        m = M.gbr().fit(rm.X(L, fr), rm.target(fr))
        est = rm.G[te] + m.predict(rm.X(L, te))
        z = np.load(K75.FITS / f"{UNIT}__{L}__D650__L5.npz")
        same_idx = bool(np.array_equal(z["idx"], te))
        dev = float(np.max(np.abs(z["est"] - est))) if same_idx else None
        bit = bool(same_idx and np.array_equal(z["est"], est))
        out[L] = {"test_rows_identical": same_idx, "largest_abs_difference_eV": dev, "bit_for_bit": bit,
                  "stored_fit_rows": int(z["fit_rows"]), "fit_rows_here": int(len(fr))}
        ok &= bit and int(z["fit_rows"]) == len(fr)
    out["pass"] = bool(ok)
    M.save_json(p, out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ mixture
def check_mixture():
    p = guard("mixture")
    sys.path.insert(0, str(K75.HERE))
    import stage_fits as SF  # B.75's learner matrices (Lazy), used as the independent reference
    ctx, u, ua = unit_ctx()
    R = ctx.R
    fr = ctx.rows_of(u, "D650", "fit")
    I = K75.load_inputs("D650")
    H10 = K75.history_matrix(R["tstep"], I)
    A = K75.anchor_scalars(R["gi"], I["anc_g"], ua.ms_b, ua.ms_e, ctx.mr_b, ctx.mr_e)
    lazy = SF.Lazy(H10, ua.wf[R["gi"]], A)
    sm = ctx.sets75["common_L5"]
    fr75 = np.isin(R["fidx"], u["fit"]) & sm
    tgt75 = (R["tg_L5"] - I["G"])[fr75]
    out = {"unit": UNIT, "learners": {}}
    ok = bool(np.array_equal(np.flatnonzero(fr75), fr))
    out["d650_fit_rows_equal_b75_rule"] = ok
    test_f = set(int(x) for x in u["test"])
    for L in M.MIX_LEARNERS:
        Xs, ys, counts, src = M.build_stack(ctx, ua, u, L)
        start = 0
        for r in M.MIX:
            if r == "D650":
                break
            start += counts[r]
        sl = slice(start, start + counts["D650"])
        rows_eq = bool(np.array_equal(src[sl], fr))
        if L in ("T0all", "TXall"):
            ref = lazy["T0" if L == "T0all" else "TX"][fr75]
        else:
            ref = M.RuleMats(ua, "D650", need_raw=True).X(L, fr)
        x_eq = bool(Xs[sl].shape == ref.shape and np.array_equal(Xs[sl], ref, equal_nan=True))
        y_eq = bool(np.array_equal(ys[sl], tgt75))
        n_test = int(sum(int(f) in test_f for f in R["fidx"][src]))
        # every block: its rows are the unit's fit rows of the rule's set, in order
        blocks_ok = True
        a = 0
        for r in M.MIX:
            blocks_ok &= bool(np.array_equal(src[a:a + counts[r]], ctx.rows_of(u, r, "fit")))
            a += counts[r]
        out["learners"][L] = {"stacked_rows_per_rule": counts, "stacked_rows": int(len(ys)), "columns": int(Xs.shape[1]),
                              "d650_block_rows_equal": rows_eq, "d650_block_inputs_bit_for_bit": x_eq,
                              "d650_block_targets_bit_for_bit": y_eq, "rows_of_test_discharges_in_stack": n_test,
                              "every_block_is_the_fit_rows_of_its_rule": blocks_ok}
        ok &= rows_eq and x_eq and y_eq and n_test == 0 and blocks_ok
        print(L, counts, len(ys), rows_eq, x_eq, y_eq, n_test, blocks_ok, flush=True)
        del Xs, ys, src
    out["pass"] = bool(ok)
    M.save_json(p, out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ raw windows
def check_raw():
    p = guard("raw")
    ctx, u, ua = unit_ctx()
    R = ctx.R
    G = K75.grid_lean(("off", "time", "ti", "seg", "fast"))
    off, tm, ti, seg, fast = G["off"], G["time"], G["ti"], G["seg"], G["fast"]
    mu, sd = ctx.FE.stats(u["fit"])
    te = ctx.rows_of(u, "D650", "test")
    pick = np.sort(M.rng_for("raw windows").choice(te, 200, replace=False))
    rm = M.RuleMats(ua, "D650", need_raw=True)
    Xr = rm.X("TXraw", pick)
    I = K75.load_inputs("D650")
    n_fast_ok = n_hist_ok = 0
    worst_fast = 0.0
    bad = []
    for q, j in enumerate(pick):
        gi = int(R["gi"][j])
        i = int(R["fidx"][j])
        a, b = off[i], off[i + 1]
        # (a) the twenty fast values from the grid rows of the file
        vals = []
        for lo, hi in ((0, C.NB), (C.NB, C.NB + C.NE)):
            for k in range(10):
                g = gi - k
                if not (a <= g < b) or seg[g] != seg[gi]:
                    raise SystemExit("FATAL: a fast row outside the block")
                s = (float(fast[g, lo]) - float(mu[lo])) / float(sd[lo])
                for c in range(lo + 1, hi):
                    s += (float(fast[g, c]) - float(mu[c])) / float(sd[c])
                vals.append(s / (hi - lo))
        vals = np.array(vals)
        got = Xr[q, 23:43]
        fok = bool(np.array_equal(vals, got))
        worst_fast = max(worst_fast, float(np.max(np.abs(vals - got))))
        n_fast_ok += fok
        # (b) history part from a copy with every value not visible under D650 raised by 1000 eV
        tmf, tif, sgf = tm[a:b], ti[a:b].copy(), seg[a:b]
        t_loc = int(R["grow"][j])
        lim = float(I["lim"][j])
        vis = (tmf <= lim + K75.TOL) & (sgf == sgf[t_loc])
        raise_ = np.isfinite(tif) & ~vis
        tif[raise_] += 1000.0
        rows = np.array([t_loc])
        V, Ag, anc = M.raw_hist_file(tmf, tif, sgf, rows, "D650")
        Fi = K75.file_inputs(tmf, tif, sgf, rows, "D650")
        H = V[0, 0]
        hist = np.concatenate([[R["tstep"][j], Fi["age"][0], Fi["G"][0] - H], V[0] - H, Ag[0]])
        hok = bool(np.array_equal(hist, Xr[q, :23], equal_nan=True) and anc[0] + a == I["anc_g"][j]
                   and Fi["H"][0] == H)
        n_hist_ok += hok
        if not (fok and hok):
            bad.append({"row": int(j), "fast_identical": fok, "history_identical": hok})
    out = {"unit": UNIT, "rule": "D650", "rows": 200, "fast_values_identical_rows": int(n_fast_ok),
           "fast_largest_abs_difference": worst_fast, "history_identical_rows": int(n_hist_ok),
           "values_raised_note": "every T_i value of the file with stamp above the D650 limit + 3 ms or in another block",
           "failures": bad[:20], "pass": bool(n_fast_ok == 200 and n_hist_ok == 200)}
    M.save_json(p, out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ rows
DESIGN_UNITS = {"V1": tuple(f"V1_r2_k{k}" for k in range(5)), "V3": tuple(f"V3_q{q}" for q in range(1, 5)), "V4a": ("V4a",)}


def ref_files(un, r):
    """(estimator, path) of every estimator that enters a comparison under rule r."""
    out = []
    if r == "K10":
        for L in ("T0", "TX"):
            out.append((L, M.M78.FITS / f"{un}__{L}__K10__L5.npz"))
    else:
        for L in ("T0", "TX"):
            out.append((L, K75.FITS / f"{un}__{L}__{r}__L5.npz"))
        if r != "D10":
            for L in ("T0a", "TXa"):
                out.append((L, K75.FITS / f"{un}__{L}__{r}__L5.npz"))
    for L in M.MIX_LEARNERS:
        out.append((L, M.FITS / f"{un}__{L}__{r}.npz"))
    if r in M.REL_RULES:
        out.append(("TXrel", M.FITS / f"{un}__TXrel__{r}.npz"))
    return out


def check_rows():
    p = guard("rows")
    ctx = M.Ctx()
    us, _, _ = M.units()
    by = {u["name"]: u for u in us}
    R = ctx.R
    out = {"cells": {}}
    ok = True
    for d, names in DESIGN_UNITS.items():
        for r in M.ALL_RULES:
            per_est = {}
            ref_idx = []
            for un in names:
                te = ctx.rows_of(by[un], r, "test")
                ref_idx.append(te)
                for L, path in ref_files(un, r):
                    z = np.load(path)
                    per_est.setdefault(L, []).append(bool(np.array_equal(z["idx"], te)))
            rows = np.concatenate(ref_idx)
            same = {L: all(v) for L, v in per_est.items()}
            cell_ok = all(same.values())
            ok &= cell_ok
            out["cells"][f"{d}|{r}"] = {"rows": int(len(rows)), "discharges": int(len(np.unique(R["fidx"][rows]))),
                                        "clusters": int(len(np.unique(R["clus"][rows]))),
                                        "estimators_on_identical_rows": same, "pass": bool(cell_ok)}
            print(f"{d} {r}: rows {len(rows):,}, discharges {len(np.unique(R['fidx'][rows]))}, "
                  f"clusters {len(np.unique(R['clus'][rows]))}, identical {cell_ok}", flush=True)
    out["pass"] = bool(ok)
    M.save_json(p, out)


# ------------------------------------------------------------------------------------------ md
def render_md():
    res = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("reuse", "mixture", "raw", "rows")}
    L = ["# B.84 checks", "",
         f"Registration `{M.REG}`, Addendum 1 `{M.ADD}` section 4. Each check was run once, after the fits and before any "
         "readout; no error, MSE or skill of this batch had been computed when they ran.", ""]
    L += ["| Check | Result | Pass |", "|---|---|---|"]
    r = res["reuse"]
    L.append(f"| reuse | `{UNIT}`, D650, L5, refitted with this batch's code: T0 test rows identical {r['T0']['test_rows_identical']}, "
             f"largest difference {r['T0']['largest_abs_difference_eV']} eV; TX test rows identical {r['TX']['test_rows_identical']}, "
             f"largest difference {r['TX']['largest_abs_difference_eV']} eV; fit rows {r['fit_rows']:,} (B.75 {r['T0']['stored_fit_rows']:,}) "
             f"| {'yes' if r['pass'] else 'NO'} |")
    m = res["mixture"]
    parts = []
    for Ln, v in m["learners"].items():
        parts.append(f"{Ln}: D650 block rows / inputs / targets equal {v['d650_block_rows_equal']} / "
                     f"{v['d650_block_inputs_bit_for_bit']} / {v['d650_block_targets_bit_for_bit']}, test-discharge rows "
                     f"{v['rows_of_test_discharges_in_stack']}, {v['stacked_rows']:,} stacked rows")
    L.append(f"| mixture | {'; '.join(parts)} | {'yes' if m['pass'] else 'NO'} |")
    w = res["raw"]
    L.append(f"| raw windows | 200 test rows of `{UNIT}` under D650: twenty fast values identical on {w['fast_values_identical_rows']} "
             f"(largest difference {w['fast_largest_abs_difference']}); 23 history columns identical on "
             f"{w['history_identical_rows']} after raising every value not visible under D650 by 1000 eV "
             f"| {'yes' if w['pass'] else 'NO'} |")
    ro = res["rows"]
    L.append(f"| rows | every estimator of every comparison on the test rows of B.75's (B.78's) stored TX, "
             f"{len(ro['cells'])} cells | {'yes' if ro['pass'] else 'NO'} |")
    L += ["", "## Stacked fit rows per rule (unit `V1_r2_k0`)", "", "| Learner | " + " | ".join(M.MIX) + " | total |",
          "|---|" + "---|" * (len(M.MIX) + 1)]
    for Ln, v in m["learners"].items():
        L.append(f"| {Ln} | " + " | ".join(f"{v['stacked_rows_per_rule'][x]:,}" for x in M.MIX) + f" | {v['stacked_rows']:,} |")
    L += ["", "## Rows per design and rule", "", "| Design | Rule | Rows | Discharges | Clusters | Same rows |", "|---|---|---|---|---|---|"]
    for k, v in ro["cells"].items():
        d, r = k.split("|")
        L.append(f"| {d} | {r} | {v['rows']:,} | {v['discharges']} | {v['clusters']} | {'yes' if v['pass'] else 'NO'} |")
    M.save_text(M.CHECKS_MD, "\n".join(L) + "\n")
    print("written", M.CHECKS_MD)


if __name__ == "__main__":
    {"reuse": check_reuse, "mixture": check_mixture, "raw": check_raw, "rows": check_rows, "md": render_md}[sys.argv[1]]()
