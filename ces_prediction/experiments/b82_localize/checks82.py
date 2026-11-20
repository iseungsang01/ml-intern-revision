"""B.82 checks (registration section 8, Addendum 1 section 9).  Each run once, after the fits and before any readout.
Only idx arrays, matrices, inputs and estimates are compared, except check 'classes', which the registration defines on
the pooled increment (it computes the pooled points of the stored estimates and the class counts, nothing per class).

    py ces_prediction/experiments/b82_localize/checks82.py reuse
    py ces_prediction/experiments/b82_localize/checks82.py tau
    py ces_prediction/experiments/b82_localize/checks82.py classes
    py ces_prediction/experiments/b82_localize/checks82.py swap
    py ces_prediction/experiments/b82_localize/checks82.py mirnov
    py ces_prediction/experiments/b82_localize/checks82.py write        # docs/paper/b82_checks.md
Output: data/.b82/checks/{name}.json, docs/paper/b82_checks.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common82 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C, K75 = M.C, M.K75
CHK = M.CHK


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def nan_bitsame(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b))) and bitsame(np.nan_to_num(a, nan=-7.25e300),
                                                                       np.nan_to_num(b, nan=-7.25e300))


# ------------------------------------------------------------------------------------------ reuse
def check_reuse():
    ctx = M.B81.Ctx()
    us, _, _ = M.units()
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    rule = "D650"
    wf, ms_b, ms_e = ctx.unit_arrays(u)
    H10, f8, A, I = ctx.matrices(rule, wf, ms_b, ms_e)
    fr, te = ctx.rows_of(u, rule)
    tgt = ctx.R["tg_L5"]
    out = {"unit": u["name"], "rule": rule, "target": "L5", "random_state": C.TREE_SEED}
    ok = True
    for L in ("T0", "TX"):
        Xf = M.matrix(None, L, u, rule, H10, f8, A, I, wf, ms_b, ms_e, fr, {})
        Xt = M.matrix(None, L, u, rule, H10, f8, A, I, wf, ms_b, ms_e, te, {})
        m = M.gbr().fit(Xf, (tgt - I["G"])[fr])
        est = I["G"][te] + m.predict(Xt)
        z = np.load(M.B75FITS / f"{u['name']}__{L}__{rule}__L5.npz")
        r = {"same_idx": bool(np.array_equal(z["idx"], te)), "est_bit_identical": bitsame(z["est"], est),
             "rows": int(len(te)), "fit_rows": int(len(fr)), "stored_fit_rows": int(z["fit_rows"]),
             "n_iter": int(m.n_iter_), "stored_n_iter": int(z["n_iter"])}
        out[L] = r
        ok &= r["same_idx"] and r["est_bit_identical"]
        M.log(f"reuse {L}: idx {r['same_idx']}, est bit-identical {r['est_bit_identical']}")
    out["passes"] = bool(ok)
    M.save_json(CHK / "reuse.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ tau
def tau_from_csv(path, local_row):
    """tau of one row from the shot file alone (time column): stamps as the grid stores them (float32), blocks cut at a
    step of 0.5 s, absent step = step above 15 and below 45 ms inside a block, the most recent one that ends at or before
    the row."""
    import pandas as pd
    t = pd.read_csv(path, usecols=["time"])["time"].to_numpy(np.float64).astype(np.float32).astype(np.float64)
    dt = np.diff(t)
    blk = np.concatenate(([0], np.cumsum(dt >= C.SEG_GAP)))
    t_row = local_row
    best = -1
    for j in range(t_row):                       # step between j and j + 1, ending at j + 1 <= row
        if blk[j] == blk[j + 1] == blk[t_row] and K75.AB_LO < dt[j] < K75.AB_HI:
            best = j
    if best < 0:
        return np.nan
    return (t[t_row] - t[best + 1]) * 1000.0


def check_tau():
    R = K75.rows_lean()
    S = M.load_npz(K75.CACHE / "sets.npz")
    G = K75.grid_lean(("names", "off"))
    tz = M.load_npz(M.CACHE / "tau.npz")
    tau, cls = tz["tau"], tz["cls"]
    sset = np.flatnonzero(S["S_L5"])
    pick = np.sort(M.rng_for("tau-check").choice(sset, 500, replace=False))
    n_same, worst = 0, []
    for j in pick:
        f = int(R["fidx"][j])
        local = int(R["gi"][j] - G["off"][f])
        v = tau_from_csv(C.DATA / str(G["names"][f]), local)
        same = bitsame(np.array([v]), np.array([tau[j]]))
        n_same += int(same)
        if not same and len(worst) < 10:
            worst.append({"row": int(j), "file": str(G["names"][f]), "csv": float(v), "batch": float(tau[j])})
    nn = {"S_L5": int((cls[S["S_L5"]] == 3).sum()), "common_L5": int((cls[S["common_L5"]] == 3).sum())}
    cnt = {rs: {M.TAU_CLASSES[k]: int((cls[S[rs]] == k).sum()) for k in range(4)} for rs in ("common_L5", "S_L5")}
    # rows whose tau lies within 1 ms of a class edge plus or minus 3 ms (the tolerance changes nothing there)
    edge = {e: int(np.sum(np.abs(tau[np.isfinite(tau)] - e) < 1.0)) for e in (103.0, 303.0)}
    out = {"rows_drawn": int(len(pick)), "rows_identical": n_same, "mismatches": worst,
           "S_set_rows_with_no_notch_before": nn["S_L5"], "common_set_rows_with_no_notch_before": nn["common_L5"],
           "class_counts": cnt, "rows_within_1_ms_of_103_or_303_ms": edge}
    out["passes"] = bool(n_same == len(pick) and nn["S_L5"] == 0)
    M.save_json(CHK / "tau.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ classes
def stored_points():
    """The stored increments (points) of B.75, B.81 and B.79 as printed (3 decimals)."""
    j75 = json.loads((C.PAPER / "b75_availability.json").read_text(encoding="utf-8"))
    j81 = json.loads((C.PAPER / "b81_controls.json").read_text(encoding="utf-8"))
    j79 = json.loads((C.PAPER / "b79_beam_inputs.json").read_text(encoding="utf-8"))
    out = {}
    for d in ("V1", "V2", "V3", "V4a", "V4b"):
        for r, kind in (("D650", "common"), ("S", "S")):
            for block in (("all", "block from 32312") if d == "V1" else ("all",)):
                m75 = j75["main"][f"{d}|L5|{kind}"][block][r]
                m79 = j79["main"][f"{d}|L5|{kind}"][block][r]
                key = (d, r, block)
                out[(*key, "TX/T0")] = ("B.75", m75["TX/T0"]["point"])
                if "TE/T0" in m75:
                    out[(*key, "TE/T0")] = ("B.75", m75["TE/T0"]["point"])
                    out[(*key, "TB/T0")] = ("B.75", m75["TB/T0"]["point"])
                elif block == "all":
                    out[(*key, "TE/T0")] = ("B.81", j81["main"][d][r]["TE/T0"]["point"])
                    out[(*key, "TB/T0")] = ("B.81", j81["main"][d][r]["TB/T0"]["point"])
                out[(*key, "TXN/TN")] = ("B.79", m79["TXN/TN"]["point"])
    return out


def check_classes():
    import readout82 as RO
    D = RO.Data()
    A = RO.part_a(D, points_only=True)
    stored = stored_points()
    res, ok = [], True
    for (d, r, block, comp), (src, val) in stored.items():
        cell = A[d][r][block]
        pooled = cell["pooled_points"][comp]
        same = f"{pooled:+.3f}" == f"{val:+.3f}"
        e = {"cell": f"{d}|{r}|{block}|{comp}", "source": src, "stored_printed": f"{val:+.3f}", "pooled_printed": f"{pooled:+.3f}",
             "same_printed": same}
        ok &= same
        # the classes: counts sum to the rows of the set; class sums of squared errors add to the pooled ones
        Lr = comp.split("/")
        for sname in ("tau", "c"):
            s = cell[sname]
            nrows = cell["counts"]["rows"] if not comp.endswith("/TN") else cell["counts_TN_TXN_rows"]["rows"]
            per_split = None
            tot = 0
            for cname in RO.CLASSES[sname]:
                c = s[cname]
                cc = c["counts"] if not comp.endswith("/TN") else c["counts_TN_TXN_rows"]
                tot += cc["rows"] if "rows_per_split" not in cc else 0
                if "rows_per_split" in cc:
                    per_split = [a + b for a, b in zip(per_split or [0] * len(cc["rows_per_split"]), cc["rows_per_split"])]
            sums_ok = (tot == nrows) if per_split is None else all(x == nrows for x in per_split)
            # class sums of squared errors
            sse_ok = True
            for L in Lr:
                pooled_sse = cell["pooled_sse_per_split"][L]
                cs = [0.0] * len(pooled_sse)
                for cname in RO.CLASSES[sname]:
                    c = s[cname]
                    if c.get("empty"):
                        continue
                    for si, v in enumerate(c["sse_per_split"][L]):
                        cs[si] += v
                sse_ok &= all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(cs, pooled_sse))
            e[f"{sname}: class counts sum to the rows"] = bool(sums_ok)
            e[f"{sname}: class sums of squared errors add to the pooled"] = bool(sse_ok)
            ok &= sums_ok and sse_ok
        res.append(e)
    out = {"cells": res, "passes": bool(ok)}
    M.save_json(CHK / "classes.json", out)
    print(json.dumps({"passes": out["passes"], "cells": len(res),
                      "printed_equal": sum(x["same_printed"] for x in res)}, indent=1))


# ------------------------------------------------------------------------------------------ swap
def check_swap():
    ctx = M.Ctx()
    us, _, _ = M.units()
    R = ctx.R
    res, ok = {}, True
    # beside: the anchors of every grid row against B.75's anchor rows of the D1 rows
    anc_ok = {}
    for r in M.RULES:
        anc_ok[r] = bool(np.array_equal(ctx.anc_all(r)[R["gi"]], ctx.inputs(r)["anc_g"]))
        ok &= anc_ok[r]
    for u in M.fit_units(us):
        wf, ms_b, ms_e = ctx.unit_arrays(u)
        fitset, testset = set(int(x) for x in u["fit"]), set(int(x) for x in u["test"])
        for r in M.RULES:
            H10, f8, A, I = ctx.matrices(r, wf, ms_b, ms_e)
            fr, te = ctx.rows_of(u, r)
            z = np.load(M.path82(u["name"], "TXs", r))
            cell = {}
            for part, rows, side, dg_key, df_key in (("fit", fr, fitset, "donor_grid_fit", "donor_file_fit"),
                                                     ("test", te, testset, "donor_grid_test", "donor_file_test")):
                T = M.matrix(ctx, "TX", u, r, H10, f8, A, I, wf, ms_b, ms_e, rows, {})
                S_ = M.matrix(ctx, "TXs", u, r, H10, f8, A, I, wf, ms_b, ms_e, rows, {})
                _, dg, dfile = ctx.swap14(u, r, wf, ms_b, ms_e, rows)
                own = R["fidx"][rows]
                has = dg >= 0
                dg_file = ctx.G["fidx"][np.maximum(dg, 0)]
                c = {"rows": int(len(rows)), "history_bit_identical": bitsame(T[:, :10], S_[:, :10]),
                     "columns": int(S_.shape[1]),
                     "no_row_with_its_own_discharge": bool(np.all(dfile != own) and np.all(dg_file[has] != own[has])),
                     "donor_rows_in_the_donor_file": bool(np.all(dg_file[has] == dfile[has])),
                     "donors_on_the_same_side": bool(all(int(x) in side for x in np.unique(dfile))),
                     "same_frame_and_segment": bool(np.all(ctx.fr[dg[has]] == ctx.fr[R["gi"][rows][has]]) and
                                                    np.all(ctx.sord[dg[has]] == ctx.sord[R["gi"][rows][has]])),
                     "stored_donors_equal": bool(np.array_equal(z[dg_key], dg) and np.array_equal(z[df_key], dfile)),
                     "rows_without_a_donor_row": int((~has).sum()),
                     "rows_with_a_missing_swapped_input": int((~np.isfinite(S_[:, 10:24]).all(1)).sum()),
                     "donor_files": int(len(np.unique(dfile)))}
                okc = (c["history_bit_identical"] and c["no_row_with_its_own_discharge"] and c["donor_rows_in_the_donor_file"]
                       and c["donors_on_the_same_side"] and c["same_frame_and_segment"] and c["stored_donors_equal"]
                       and c["columns"] == 24)
                ok &= okc
                cell[part] = c
            res[f"{u['name']}|{r}"] = cell
        M.log(f"swap {u['name']}: done")
    out = {"anchors_of_all_grid_rows_equal_B75_on_D1_rows": anc_ok, "cells": res, "passes": bool(ok)}
    M.save_json(CHK / "swap.json", out)
    print(json.dumps({"passes": out["passes"], "cells": len(res), "anchors": anc_ok}, indent=1))


# ------------------------------------------------------------------------------------------ mirnov
def check_mirnov():
    import pandas as pd
    ctx = M.Ctx()
    us, _, _ = M.units()
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    G = K75.grid_lean(("names", "off"))
    mu, sd = ctx.FEmc.stats(u["fit"])
    out = {"unit": u["name"], "standardisation": "mean and sd of each Mirnov column over the grid rows of the unit's fit "
                                                "discharges (Feats.stats), taken from the batch",
           "mu": mu.tolist(), "sd": sd.tolist()}
    ok = True
    for r in M.RULES:
        I = ctx.inputs(r)
        batch = ctx.mirnov7(u, I["anc_g"])
        _, te = ctx.rows_of(u, r)
        pick = np.sort(M.rng_for(f"mirnov|{r}").choice(te, 200, replace=False))
        n_same, n_nan = 0, 0
        for j in pick:
            f = int(ctx.R["fidx"][j])
            a = int(G["off"][f])
            df = pd.read_csv(C.DATA / str(G["names"][f]), usecols=["time", *M.MC_COLS])
            t = df["time"].to_numpy(np.float64).astype(np.float32).astype(np.float64)
            x = df[list(M.MC_COLS)].to_numpy(np.float64).astype(np.float32).astype(np.float64)
            seg = np.concatenate(([0], np.cumsum(np.diff(t) >= C.SEG_GAP)))
            z = ((x[:, 0] - mu[0]) / sd[0])
            z += (x[:, 1] - mu[1]) / sd[1]
            z /= 2
            raw = x[:, 0].copy()
            raw += x[:, 1]
            raw /= 2
            w4 = C.window_feats(z, seg)
            ms, mr = K75.run_mean10(z, seg), K75.run_mean10(raw, seg)
            t_loc = int(ctx.R["gi"][j]) - a
            anc = int(I["anc_g"][j]) - a
            ref = np.full(7, np.nan)
            ref[:4] = w4[t_loc]
            if anc >= 0:
                ref[4] = ms[anc]
                ref[5] = ms[t_loc] - ms[anc]
                if mr[t_loc] > 0 and mr[anc] > 0:
                    ref[6] = np.log(mr[t_loc] / mr[anc])
            same = nan_bitsame(ref, batch[j])
            n_same += int(same)
            n_nan += int(np.isnan(batch[j]).any())
        out[r] = {"rows": int(len(pick)), "rows_identical": n_same, "rows_with_an_undefined_scalar": n_nan}
        ok &= n_same == len(pick)
        M.log(f"mirnov {r}: {n_same}/{len(pick)} identical")
    out["passes"] = bool(ok)
    M.save_json(CHK / "mirnov.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ md
def write():
    j = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("reuse", "tau", "classes", "swap", "mirnov")}
    extra = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("mc_join", "wq_inputs")
             if (CHK / f"{n}.json").exists()}
    ru, ta, cl, sw, mi = (j[k] for k in ("reuse", "tau", "classes", "swap", "mirnov"))
    L = ["# B.82 checks", "",
         f"Registration: `{M.REG}` section 8; implementation: `{M.ADD}` section 9. Each check was run once, after the fits "
         "and before any readout. No tolerance was changed after a run. Checks reuse, tau, swap and Mirnov compare idx arrays, "
         "inputs and estimates only; check classes forms, as registered, the pooled increments of the stored estimates of "
         "B.75, B.79 and B.81 (known numbers) and the class counts; no value of a class was formed.", "",
         "## 1. Reuse", "",
         f"Unit {ru['unit']}, rule {ru['rule']}, target {ru['target']}, random_state {ru['random_state']}: refitted with this "
         "batch's code and compared with `data/.b75/fits/`.", "",
         "| learner | test rows | same rows | estimates bit-identical | fit rows (this batch / B.75) | iterations (this batch / B.75) |",
         "|---|---|---|---|---|---|"]
    for Lr in ("T0", "TX"):
        x = ru[Lr]
        L.append(f"| {Lr} | {x['rows']:,} | {x['same_idx']} | {x['est_bit_identical']} | {x['fit_rows']:,} / {x['stored_fit_rows']:,} | "
                 f"{x['n_iter']} / {x['stored_n_iter']} |")
    L += ["", f"Check passes: **{ru['passes']}**", "",
          "## 2. tau", "",
          f"{ta['rows_drawn']} rows drawn from the S set (L5); tau recomputed from the row's shot file alone (time column, "
          f"stamps as float32 as the grid stores them): identical for {ta['rows_identical']} of {ta['rows_drawn']}. "
          f"S-set rows in the class 'no notch before': {ta['S_set_rows_with_no_notch_before']} (common set: "
          f"{ta['common_set_rows_with_no_notch_before']:,}). Rows within 1 ms of a class edge (103, 303 ms): "
          f"{ta['rows_within_1_ms_of_103_or_303_ms']}.", "",
          "| set | " + " | ".join(M.TAU_CLASSES) + " |", "|---|---|---|---|---|"]
    for rs, c in ta["class_counts"].items():
        L.append(f"| {rs} | " + " | ".join(f"{c[k]:,}" for k in M.TAU_CLASSES) + " |")
    L += ["", f"Check passes: **{ta['passes']}**", "",
          "## 3. Classes", "",
          "For every design, rule (D650, S), block (V1: all rows and the block from 32312) and comparison that a stored "
          "readout holds: the pooled increment of the stored estimates on this batch's rows, printed to three decimals, "
          "against the printed value of B.75 (TX/T0; TE/T0 and TB/T0 of V1 and V4), B.81 (TE/T0 and TB/T0 of V2 and V3) or "
          "B.79 (TXN/TN). For each stratification (tau, c): the class counts sum to the rows of the set (under V1 in each "
          "re-partition), and the class sums of squared errors add to the pooled ones (relative tolerance 1e-9).", "",
          "| cell | source | stored | pooled here | tau: counts / sums | c: counts / sums |", "|---|---|---|---|---|---|"]
    for e in cl["cells"]:
        L.append(f"| {e['cell']} | {e['source']} | {e['stored_printed']} | {e['pooled_printed']} | "
                 f"{e['tau: class counts sum to the rows']} / {e['tau: class sums of squared errors add to the pooled']} | "
                 f"{e['c: class counts sum to the rows']} / {e['c: class sums of squared errors add to the pooled']} |")
    L += ["", f"Check passes: **{cl['passes']}**", "",
          "## 4. Swap", "",
          f"Anchor rows of every grid row (needed for the donor rows) against B.75's anchor rows on the D1 rows: "
          f"{sw['anchors_of_all_grid_rows_equal_B75_on_D1_rows']}. For every unit of part C and both rules, fit and test rows: "
          "the ten history columns of TXs bit-identical to TX's; no row carries the inputs of its own discharge (donor file "
          "and donor row's file differ from the row's); every donor lies on the row's side (fit or test) of the unit; the "
          "donor row has the row's segment ordinal and frame index; the donors stored with the fit equal the rebuilt ones.", "",
          "| unit, rule | rows (fit / test) | history identical | own discharge never | same side | same frame | rows without a donor row (fit / test) | donor files (fit / test) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, x in sw["cells"].items():
        f_, t_ = x["fit"], x["test"]
        L.append(f"| {k} | {f_['rows']:,} / {t_['rows']:,} | {f_['history_bit_identical']} / {t_['history_bit_identical']} | "
                 f"{f_['no_row_with_its_own_discharge']} / {t_['no_row_with_its_own_discharge']} | "
                 f"{f_['donors_on_the_same_side']} / {t_['donors_on_the_same_side']} | "
                 f"{f_['same_frame_and_segment']} / {t_['same_frame_and_segment']} | "
                 f"{f_['rows_without_a_donor_row']:,} / {t_['rows_without_a_donor_row']:,} | {f_['donor_files']} / {t_['donor_files']} |")
    L += ["", f"Check passes: **{sw['passes']}**", "",
          "## 5. Mirnov", "",
          f"Unit {mi['unit']}: 200 test rows per rule; the seven Mirnov scalars recomputed from the row's shot file alone "
          "(time and the two MC columns as float32, blocks, standardisation with the unit's statistics, window scalars, "
          "running means, anchor scalars with B.75's anchor row) against the batch's.", "",
          "| rule | rows | identical | rows with an undefined scalar |", "|---|---|---|---|"]
    for r in M.RULES:
        x = mi[r]
        L.append(f"| {r} | {x['rows']} | {x['rows_identical']} | {x['rows_with_an_undefined_scalar']} |")
    L += ["", f"Check passes: **{mi['passes']}**", ""]
    if "mc_join" in extra:
        x = extra["mc_join"]
        L += ["## Beside the checks: the Mirnov columns joined to the grid", "",
              f"{x['files']} files, {x['rows']:,} rows; files with another row count {x['files_other_row_count']}, with other "
              f"stamps {x['files_other_time']}, with other BES or MC headers {x['files_other_bes_or_mc_header']}; non-finite "
              f"MC values {x['nonfinite_mc_values']}. Passes: {x['passes']}.", ""]
    if "wq_inputs" in extra:
        x = extra["wq_inputs"]
        L += ["## Beside the checks: the weighted quantile on an input", "",
              f"Weighted quantiles (0.5, 0.75, 0.9) of c over the test rows of {x['unit']} under D650 against the quantiles of "
              f"the explicitly resampled rows, {x['draws']} draws per form: largest absolute difference "
              f"{x['largest_abs_difference']:.3g} (identical: {x['identical']}).", ""]
    M.save_text(M.CHECKS_MD, "\n".join(L))
    print("written", M.CHECKS_MD)


def check_wq_inputs():
    """Not a registered check: the weighted quantile of boot82 on an input (c of the test rows of V1_r2_k0, D650), against
    the quantile of the explicitly resampled rows; no target or estimate enters."""
    import boot82 as BT
    R = K75.rows_lean()
    cc = M.load_npz(M.CACHE / "cclass.npz")
    S = M.load_npz(K75.CACHE / "sets.npz")
    c = cc["V1_r2_k0|D650|c_test"]
    rows = np.flatnonzero(np.isfinite(c) & S["common_L5"])
    _, ud = np.unique(R["fidx"], return_inverse=True)
    _, uc = np.unique(R["clus"], return_inverse=True)
    U = BT.Units(rows, ud, uc)
    rng = M.rng_for("wq-inputs")
    dev = 0.0
    for form in BT.FORMS:
        inv, nu = U.idx[form]
        wq = BT.WQ(c[rows], inv, nu)
        Wm = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=20).astype(np.float64)
        a = wq.draws(Wm, (0.5, 0.75, 0.9))
        b = np.array([wq.direct(Wm[i], (0.5, 0.75, 0.9)) for i in range(20)])
        dev = max(dev, float(np.max(np.abs(a - b))))
    out = {"unit": "V1_r2_k0", "draws": 20, "largest_abs_difference": dev, "identical": dev == 0.0}
    M.save_json(CHK / "wq_inputs.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    {"reuse": check_reuse, "tau": check_tau, "classes": check_classes, "swap": check_swap, "mirnov": check_mirnov,
     "wq_inputs": check_wq_inputs, "write": write}[sys.argv[1]]()
    M.log(f"done ({time.time() - t0:.0f}s)")
