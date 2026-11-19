"""B.71 checks (PREREGISTRATION_B71.md section 6; checks 4b and 5 and the D1 mapping from addendum 1) and the timing.

  1   with every sample available, the rebuilt inputs equal build_blocks (np.allclose, 1e-6) -- both networks
  2   U of the network equals the B.70 D1 re-scoring within 0.5 eV on >= 99.9 % of the D1 T_i rows (absolute errors:
      B.70 stores squared errors, not predictions)
  3   with a cut after the block end, F, R and G equal U exactly
  4   R stepped row by row with every sample available (no feedback) equals the batched U forward within 1e-4 eV
  4b  the batched F/G continuation from the U state, with every sample available, equals U within 1e-4 eV
  5   the causal GP fitted once per cut equals baselines_interpolation.predict_gp_causal row by row within 1e-6 eV
No check computes an error against the measured T_i by scenario.

    py ces_prediction/experiments/b71_scenarios/run_b71.py check  [--fold r2f0] [--files 5] [--out DIR]
    py ces_prediction/experiments/b71_scenarios/run_b71.py timing [--fold r2f0] [--files 5]
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np

import run_b71 as M

TOL4_EV, TOL5_EV = 1e-4, 1e-6


def parse_fold(s):
    return int(s[1:].split("f")[0]), int(s.split("f")[1])


def fold_checks(X, r, k, nfiles):
    """Checks 3, 4, 4b and 5 on the first `nfiles` test files of fold r k."""
    T, base = X.T, X.base
    nets = X.nets(r, k)
    test = X.test_files(r, k, nfiles)
    c3 = {"blocks": 0, "equal": 0}
    c4 = {key: {"rows": 0, "max_dev_ti_eV": 0.0, "max_dev_vt": 0.0} for key in nets}
    c4b = {key: {"rows": 0, "max_dev_ti_eV": 0.0, "max_dev_vt": 0.0} for key in nets}
    c5 = {"rows": 0, "max_dev_eV": 0.0, "cuts": 0}
    for nm in test:
        F = X.file_prep(nm, nets)
        n = len(F["tm"])
        kept = [c for c in F["cuts"] if c["kept"]]
        for key, net in nets.items():
            blks = F["P"][key]["blks"]
            U = X.u_pass(net, blks, F["spans"], n)
            # check 3: a cut after the end of every block -> the assembled F, R, G block outputs are U's
            after = []
            for bi, (a, b) in enumerate(F["spans"]):
                tq = F["tq"][a:b]
                L = b - a
                c = {"block": bi, "a": a, "b": b, "cq": int(tq[-1]) + M.CUT_STEP_Q, "t0": L, "hF": L,
                     "hG": {g: L for g in M.GAPS_MS}}
                after.append(c)
                ok_in = True
                obs = F["obs"][a:b]
                for av in (M.avail_F(obs, tq, c["cq"]), *(M.avail_G(obs, tq, c["cq"], g) for g in M.GAPS_MS)):
                    ok_in &= bool(np.array_equal(av, obs))
                after[-1]["ok_in"] = ok_in
            RP = X.replay(F, net, key, after)
            for ci, c in enumerate(after):
                a, b, t0 = c["a"], c["b"], c["t0"]
                outs = [RP["F"][ci], RP["R"][ci], *(RP["G"][g][ci] for g in M.GAPS_MS)]
                eq = c["ok_in"]
                for o in outs:
                    blk = U[a:b].copy()
                    blk[t0:t0 + len(o)] = o
                    eq &= bool(np.array_equal(blk, U[a:b])) and len(o) == 0
                c3["blocks"] += 1
                c3["equal"] += int(eq)
            # checks 4 and 4b: the stepped (R engine, no feedback) and the batched (F/G engine) continuation with the
            # U inputs over the F horizon of every kept cut, against the batched U forward
            xs, dts, sts, idx = [], [], [], []
            with T.no_grad():
                for bi, (blk, (a, b)) in enumerate(zip(blks, F["spans"])):
                    kc = [c for c in kept if c["block"] == bi]
                    if not kc:
                        continue
                    st = M.u_states(T, net, blk["x"], [c["t0"] for c in kc])
                    for c in kc:
                        t0, h = c["t0"], c["hF"]
                        xs.append(blk["x"][t0:h])
                        dts.append(np.diff(F["tm"][a + t0 - 1:a + h]))
                        sts.append(st[t0])
                        idx.append(np.arange(a + t0, a + h))
                o4 = M.run_r(T, net, xs, dts, sts, base, feedback=False)
                o4b = M.run_cont(T, net, xs, sts)
            std64 = net.std32.astype(np.float64)
            for rec, outs in ((c4[key], o4), (c4b[key], o4b)):
                for o, rows in zip(outs, idx):
                    # the network outputs (float32, normalised) converted to eV in float64: the replay difference
                    # itself; the float32 conversion (net.phys) adds its own rounding, 1.2e-4 eV above 1,024 eV
                    d = np.abs(o.astype(np.float64) - U[rows].astype(np.float64)) * std64
                    d32 = np.abs(net.phys(o).astype(np.float64) - net.phys(U[rows]).astype(np.float64))
                    rec["rows"] += len(rows)
                    if len(rows):
                        rec["max_dev_ti_eV"] = max(rec["max_dev_ti_eV"], float(d[:, 0].max()))
                        rec["max_dev_vt"] = max(rec["max_dev_vt"], float(d[:, 1].max()))
                        rec["max_dev_ti_eV_float32_conversion"] = max(rec.get("max_dev_ti_eV_float32_conversion", 0.0),
                                                                      float(d32[:, 0].max()))
                        rec["max_dev_normalised_ti"] = max(rec.get("max_dev_normalised_ti", 0.0),
                                                           float(np.abs(o[:, 0] - U[rows, 0]).max()))
                        rec["rows_ti_above_1e-4_eV"] = rec.get("rows_ti_above_1e-4_eV", 0) + int((d[:, 0] > TOL4_EV).sum())
        # check 5: GP fitted once per cut vs predict_gp_causal per row, on every row of the F horizon with T_i measured
        y = F["tgt"][:, 0].astype(np.float64)
        for c in kept:
            a, t0 = c["a"], c["t0"]
            o = np.flatnonzero(F["obs"][a:a + t0, 0] & (F["tq"][a:a + t0] < c["cq"])) + a
            rows = np.arange(a + t0, a + c["hF"])
            rows = rows[np.isfinite(y[rows])]
            if not len(rows):
                continue
            fast = M.gp_causal_fixed(X.C.B, F["tm"][o], y[o], F["tm"][rows])
            ref = np.array([X.C.B.predict_gp_causal(F["tm"][o], y[o], float(F["tm"][t])) for t in rows])
            c5["rows"] += len(rows)
            c5["cuts"] += 1
            c5["max_dev_eV"] = max(c5["max_dev_eV"], float(np.max(np.abs(fast - ref))))
    res = {"files": test, "check3": c3, "check4": c4, "check4b": c4b, "check5": c5}
    res["check3"]["passes"] = bool(c3["blocks"] > 0 and c3["equal"] == c3["blocks"])
    for kk in ("check4", "check4b"):
        res[kk]["passes"] = bool(all(v["rows"] > 0 and v["max_dev_ti_eV"] <= TOL4_EV for v in res[kk].values()
                                     if isinstance(v, dict)))
    res["check5"]["passes"] = bool(c5["rows"] > 0 and c5["max_dev_eV"] <= TOL5_EV)
    res["passes"] = all(res[kk]["passes"] for kk in ("check3", "check4", "check4b", "check5"))
    return res


def count_all(X):
    """Kept cuts and replayed rows of every test file of the 15 folds (no network run)."""
    per = {}
    for r, k in M.fold_list(None):
        kc = rowsF = rowsG = files = 0
        for nm in X.test_files(r, k):
            arr = X.C.grid[nm]
            tm = arr[:, 0].astype(np.float64)
            tq = M.tq_of(tm)
            obs = np.isfinite(arr[:, 1])
            for a, b in M.R.block_spans(tm, X.C.GAP):
                for c in M.block_cuts(tq[a:b], obs[a:b]):
                    if c["kept"]:
                        kc += 1
                        rowsF += c["hF"] - c["t0"]
                        rowsG += sum(c["hG"][g] - c["t0"] for g in M.GAPS_MS)
            files += 1
        per[f"r{r}f{k}"] = {"files": files, "kept_cuts": kc, "rows_F": rowsF, "rows_G": rowsG}
    return per


def run_timing(args):
    t_all = _time.time()
    X = M.B71()
    t_ctx = _time.time() - t_all
    r, k = parse_fold(args.fold)
    test = X.test_files(r, k, args.files)
    t1 = _time.time()
    nets = X.nets(r, k)
    d1, _ = X.d1_rows(r, k)
    t_setup = _time.time() - t1
    t2 = _time.time()
    tr = X.train_range(r, k)
    t_range = _time.time() - t2
    times, cuts5, rf5 = [], 0, 0
    for nm in test:
        t3 = _time.time()
        fu, gu, cuts, U = X.file_units(nm, nets, d1)
        times.append(_time.time() - t3)
        kept = [c for c in cuts if c["kept"]]
        cuts5 += len(kept)
        rf5 += sum(c["hF"] - c["t0"] for c in kept)
    t_c34 = _time.time()
    fold_checks(X, r, k, args.files)
    t_c34 = _time.time() - t_c34
    allc = count_all(X)
    K_all = sum(v["kept_cuts"] for v in allc.values())
    RF_all = sum(v["rows_F"] for v in allc.values())
    T5 = float(sum(times))
    proj_cut = T5 / max(cuts5, 1) * K_all
    proj_row = T5 / max(rf5, 1) * RF_all
    fixed = t_ctx + 15 * (t_setup + t_range + t_c34)
    res = {"fold": args.fold, "files": test, "seconds_per_file": [round(x, 1) for x in times],
           "kept_cuts_5_files": cuts5, "F_rows_5_files": rf5, "seconds_5_files": round(T5, 1),
           "seconds_context_load": round(t_ctx, 1), "seconds_fold_setup": round(t_setup, 1),
           "seconds_train_range": round(t_range, 1), "seconds_checks_3_to_5_on_5_files": round(t_c34, 1),
           "all_15_folds": {"kept_cuts": K_all, "rows_F": RF_all, "per_fold": allc},
           "projection_hours_by_cuts": round((proj_cut + fixed) / 3600, 2),
           "projection_hours_by_F_rows": round((proj_row + fixed) / 3600, 2)}
    M.CACHE.mkdir(exist_ok=True)
    (M.CACHE / "timing.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({kk: v for kk, v in res.items() if kk != "all_15_folds"}, indent=1), flush=True)
    print("all 15 folds:", K_all, "kept cuts,", RF_all, "F rows", flush=True)
    return res


def run_checks(args):
    t0 = _time.time()
    X = M.B71()
    r, k = parse_fold(args.fold)
    res = {"fold": args.fold, "files": X.test_files(r, k, args.files)}
    b70 = M.B70_ROOT / f"b39p_v2_cut_r{r}_f{k}" / "comparison_errors_test.npz"
    if b70.exists():
        out, meta = X.rows_fold(r, k, limit=args.files, t_start=t0)      # checks 1 and 2 (+ D1 mapping)
        res["d1_mapping"] = meta["d1"]
        res["check1"] = meta["check1"]
        res["check2"] = meta["check2"]
        res["units"] = {"F": int(len(out["F__row"])), "G": int(len(out["G__row"])), "kept_cuts": meta["kept_cuts"]}
        res["train_range"] = meta["train_range"]
        if args.out:
            d = Path(args.out)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"rows_r{r}_f{k}.npz", **out)
    else:
        res["check2"] = {"pending": f"{b70} does not exist yet"}
        nets = X.nets(r, k)
        c1 = {"files": 0, "failed": []}
        for nm in res["files"]:
            for key, net in nets.items():
                blks, spans, tm, tgt, tz = M.file_inputs(X.C, X.C.grid[nm], (net.model, net.stats), net.per_shot)
                if not M.input_check(X.C, blks, spans, tm, tgt, tz):
                    c1["failed"].append(f"{key} {nm}")
            c1["files"] += 1
        res["check1"] = c1
    res["check1"]["passes"] = not res["check1"]["failed"]
    M.log(t0, "checks 1-2 done")
    res["checks_3_to_5"] = fold_checks(X, r, k, args.files)
    M.log(t0, "checks 3-5 done")
    M.CACHE.mkdir(exist_ok=True)
    (M.CACHE / f"checks_{args.fold}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({kk: res[kk] for kk in res if kk not in ("files", "train_range")}, indent=1), flush=True)
    return res
