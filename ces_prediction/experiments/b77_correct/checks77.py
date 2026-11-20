"""B.77 checks (registration section 9, Addendum 1 section 11).  Run before any readout.

    py ces_prediction/experiments/b77_correct/checks77.py data      # check 3 (inputs, target)
    py ces_prediction/experiments/b77_correct/checks77.py b75       # check 1b (this code on B.75's rows: bit-identical)
    py ces_prediction/experiments/b77_correct/checks77.py compare   # check 1a (after the fits)
    py ces_prediction/experiments/b77_correct/checks77.py frame     # check 1c (after the fits)
    py ces_prediction/experiments/b77_correct/checks77.py leakage   # check 2 and 2+ (after the fits)
    py ces_prediction/experiments/b77_correct/checks77.py write     # docs/paper/b77_checks.md
Output: data/.b77/checks/{name}.json, docs/paper/b77_checks.md.  No score against a target is computed here.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common77 as M

import numpy as np  # noqa: E402

import stage_fits as SF  # noqa: E402
import stage_trees as ST  # noqa: E402

warnings.filterwarnings("ignore")
K, C = M.K, M.C
REPRO_UNITS = ("V1_r2_k0", "V4a")
FRAME_UNITS = ("V1_r2_k0", "V2_s1_k0", "V3_q1", "V4a", "V4b")
LEAK_UNIT = "V1_r2_k0"


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def unit(R, name):
    us, _, _ = M.units(R)
    return next(x for x in us if x["name"] == name)


# ------------------------------------------------------------------------------------------ check 3
def check_data():
    R = M.rows()
    S = K.load_npz(K.CACHE / "sets.npz")
    inset = M.row_set(R, S)
    I = K.load_inputs(M.RULE)
    G = K.grid_lean(("time", "ti", "seg", "fidx"))
    later = C.later_values(G["time"], G["ti"], G["seg"], R["gi"], G["fidx"], R["fidx"])
    l5 = C.targets_from_later(R["y"], later)["L5"]
    out = {"row_set": int(inset.sum()), "discharges": int(len(np.unique(R["fidx"][inset]))),
           "clusters": int(len(np.unique(R["clus"][inset]))),
           "L5_recomputed_bit_identical": bool(np.array_equal(l5, R["tg_L5"], equal_nan=True)),
           "H_equals_anchor_value": bool(np.array_equal(I["H"][inset], G["ti"][I["anc_g"][inset]])),
           "min_earlier_values_kept": float(np.min(I["th"][inset, 4])),
           "L5_above_zero": bool(np.all(R["tg_L5"][inset] > 0)),
           "y_max_abs_dev_from_grid_eV": float(np.max(np.abs(R["y"] - G["ti"][R["gi"]]))),
           "anchor_is_row_t_minus_1_share": float(np.mean(I["anc_g"][inset] == R["gi"][inset] - 1))}
    m3 = M.med3(R["y"][inset], I["anc_g"][inset], G["ti"], G["seg"])
    out["MED3_defined_rows"] = int(np.isfinite(m3).sum())
    out["passes"] = bool(out["L5_recomputed_bit_identical"] and out["H_equals_anchor_value"]
                         and out["min_earlier_values_kept"] >= 3 and out["L5_above_zero"]
                         and out["MED3_defined_rows"] == out["row_set"])
    M.save_json(M.CHK / "data.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 1b
def check_b75():
    import fits77 as F
    B = F.Base()
    R = B.R
    common = B.ctx.S["common_L5"]
    out, worst = {}, 0.0
    for un in REPRO_UNITS:
        u = unit(R, un)
        UI = M.UnitInputs(B.ctx, u, B.F4)
        fit = np.flatnonzero(np.isin(R["fidx"], u["fit"]) & common)
        te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & common)
        resid = R["tg_L5"] - UI.G
        for L in ("T0", "TX"):
            m = SF.gbr().fit(UI.X(L, fit), resid[fit])
            est = UI.G[te] + m.predict(UI.X(L, te))
            z = M.load_npz(K.FITS / f"{un}__{L}__D10__L5.npz")
            same_idx = bool(np.array_equal(z["idx"], te))
            dev = float(np.max(np.abs(z["est"] - est))) if same_idx else float("inf")
            worst = max(worst, dev)
            out[f"{un}|{L}"] = {"same_test_rows": same_idx, "max_abs_dev_eV": dev, "bit_identical": bool(same_idx and bitsame(z["est"], est)),
                                "fit_rows": int(len(fit)), "fit_rows_b75": int(z["fit_rows"]),
                                "n_iter": int(m.n_iter_), "n_iter_b75": int(z["n_iter"])}
            M.log(f"b75 repro {un} {L}: fit rows {len(fit):,}")
    out["passes"] = bool(all(v["bit_identical"] for v in out.values() if isinstance(v, dict)))
    M.save_json(M.CHK / "b75.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 1a
def check_compare():
    R = M.rows()
    us, _, _ = M.units(R)
    out = {}
    for L in ("T0", "TX"):
        per = {}
        alld = []
        for u in us:
            z7 = M.load_npz(M.fit_path(u, L))
            z5 = M.load_npz(K.FITS / f"{u['name']}__{L}__D10__L5.npz")
            com, i7, i5 = np.intersect1d(z7["idx"], z5["idx"], return_indices=True)
            d = np.abs(z7["est"][i7] - z5["est"][i5])
            alld.append(d)
            per[u["name"]] = {"rows_both": int(len(com)), "rows_b77": int(len(z7["idx"])), "rows_b75": int(len(z5["idx"])),
                              "b75_rows_all_in_b77": bool(len(com) == len(z5["idx"])),
                              "max_abs_dev_eV": float(d.max()), "median_abs_dev_eV": float(np.median(d)),
                              "share_identical": float(np.mean(d == 0))}
        d = np.concatenate(alld)
        out[L] = {"units": per, "all": {"rows_both": int(len(d)), "max_abs_dev_eV": float(d.max()),
                                        "median_abs_dev_eV": float(np.median(d)), "p90_abs_dev_eV": float(np.quantile(d, 0.9)),
                                        "rows_above_100_eV": int(np.sum(d > 100)), "rows_above_300_eV": int(np.sum(d > 300)),
                                        "share_identical": float(np.mean(d == 0))}}
        M.log(f"compare {L}: rows {len(d):,}")
    out["note"] = "reported; the row sets differ (B.75: common set; B.77: complete under D10), so equality is not required"
    M.save_json(M.CHK / "compare.json", out)
    print(json.dumps({L: out[L]["all"] for L in ("T0", "TX")}, indent=1))


# ------------------------------------------------------------------------------------------ check 1c
def check_frame():
    import fits77 as F
    B = F.Base()
    R = B.R
    out = {}
    for un in FRAME_UNITS:
        u = unit(R, un)
        UI = M.UnitInputs(B.ctx, u, B.F4)
        fit = np.flatnonzero(np.isin(R["fidx"], u["fit"]) & B.inset)
        te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & B.inset)
        resid = R["tg_L5"] - UI.G
        Xf = UI.X("C0", fit, frame=False)
        m = SF.gbr().fit(Xf, resid[fit])
        est = UI.G[te] + m.predict(UI.X("C0", te, frame=False))
        z = M.load_npz(M.fit_path(u, "T0"))
        same_idx = bool(np.array_equal(z["idx"], te))
        out[un] = {"columns": int(Xf.shape[1]), "same_test_rows": same_idx,
                   "bit_identical": bool(same_idx and bitsame(z["est"], est)),
                   "max_abs_dev_eV": float(np.max(np.abs(z["est"] - est))) if same_idx else float("inf"),
                   "n_iter": int(m.n_iter_), "n_iter_T0": int(z["n_iter"])}
        M.log(f"frame {un}: test rows {len(te):,}")
    out["passes"] = bool(all(v["bit_identical"] for v in out.values() if isinstance(v, dict)))
    M.save_json(M.CHK / "frame.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 2
class FileFeats(ST.Feats):
    """B.73's Feats restricted to one file, with the unit's standardisation passed in (the fit discharges are not part of
    the file), so that window(), zmeans() and run_mean10 run B.73/B.75's arithmetic on a modified copy of the file."""

    def __init__(self, fast_file, seg_file, mu, sd):  # noqa: D107  (no grid load)
        self.off = np.array([0, len(seg_file)])
        self.seg = seg_file
        self.fid = np.zeros(len(seg_file), np.int64)
        self.fast = fast_file
        self._musd = (mu, sd)

    def stats(self, fit_files):
        return self._musd


def file_fast(fast_file, seg_file, mu, sd, fit):
    FE1 = FileFeats(fast_file, seg_file, mu, sd)
    wf = FE1.window(fit)
    bz, ez = K.zmeans(FE1, fit)
    ms_b, ms_e = K.run_mean10(bz, seg_file), K.run_mean10(ez, seg_file)
    mr_b, mr_e = (K.run_mean10(x, seg_file) for x in K.raw_means(fast_file))
    return wf, ms_b, ms_e, mr_b, mr_e


def check_leakage():
    import fits77 as F
    B = F.Base()
    R = B.R
    u = unit(R, LEAK_UNIT)
    UI = M.UnitInputs(B.ctx, u, B.F4)
    Gd = K.grid_lean(("off", "time", "ti", "seg", "fast"))
    off, tm, ti, seg, fast = Gd["off"], Gd["time"], Gd["ti"], Gd["seg"], Gd["fast"]
    I = K.load_inputs(M.RULE)
    fit = np.flatnonzero(np.isin(R["fidx"], u["fit"]) & B.inset)
    te = np.flatnonzero(np.isin(R["fidx"], u["test"]) & B.inset)
    resid = R["tg_L5"] - UI.G
    models, stored = {}, {}
    refit_same = {}
    for L in ("C0", "CX"):
        m = SF.gbr().fit(UI.X(L, fit), resid[fit])
        est = UI.G[te] + m.predict(UI.X(L, te))
        z = M.load_npz(M.fit_path(u, L))
        refit_same[L] = bool(np.array_equal(z["idx"], te) and bitsame(z["est"], est))
        models[L] = m
        stored[L] = dict(zip(z["idx"].tolist(), z["est"].tolist()))
    mu, sd = B.ctx.FE.stats(u["fit"])
    med_main = {}
    pick = np.sort(M.rng_for("leakage").choice(te, 200, replace=False))
    m3 = M.med3(R["y"][pick], I["anc_g"][pick], ti, seg)
    for q, j in enumerate(pick):
        med_main[int(j)] = m3[q]
    rng = M.rng_for("leakage|modify")
    modes = ("removed", "raised", "changed with fast", "control: y of row t raised", "control: fast of row t x 1.5")
    res = {mo: {"rows": 0, "inputs_identical": 0, "estimates_identical": 0, "frame_scalars_changed": 0,
                "fast_scalars_changed": 0, "ti_values_modified": 0, "fast_rows_modified": 0} for mo in modes}
    file_path_same = 0
    for j in pick:
        j = int(j)
        i = int(R["fidx"][j])
        a, b = off[i], off[i + 1]
        t = int(R["grow"][j])
        tif0, ff0, sf = ti[a:b].copy(), fast[a:b].copy(), seg[a:b]
        later = np.arange(b - a) > t
        # the file-local path on the unmodified file reproduces the main pass (sanity of the check itself)
        wf0, msb0, mse0, mrb0, mre0 = file_fast(ff0, sf, mu, sd, u["fit"])
        A0 = K.anchor_scalars(np.array([t]), np.array([int(I["anc_g"][j]) - a]), msb0, mse0, mrb0, mre0)
        file_path_same += int(bitsame(np.concatenate([wf0[t], A0[0]]), UI.fast14([j])[0]))
        main = {"H10": UI.XS.H10[j:j + 1], "F4": B.F4[j:j + 1], "f14": UI.fast14([j]), "G": UI.G[j:j + 1],
                "M3": np.array([med_main[j]])}
        for mo in modes:
            tif, ff, y = tif0.copy(), ff0.copy(), float(R["y"][j])
            if mo == "removed":
                sel = later & np.isfinite(tif)
                tif[sel] = np.nan
            elif mo == "raised":
                sel = later & np.isfinite(tif)
                tif[sel] += 1000.0
            elif mo == "changed with fast":
                fin = later & np.isfinite(tif)
                tif[fin] *= rng.uniform(0.5, 1.5, int(fin.sum()))
                miss = np.flatnonzero(later & ~np.isfinite(tif0))
                give = miss[rng.random(len(miss)) < 0.5]
                tif[give] = 500.0 * rng.uniform(0.5, 1.5, len(give))
                sel = fin.copy()
                sel[give] = True
                fr = np.flatnonzero(later)
                ff[fr] = (ff[fr].astype(np.float64) * rng.uniform(0.5, 1.5, (len(fr), ff.shape[1]))).astype(ff.dtype)
                res[mo]["fast_rows_modified"] += len(fr)
            elif mo == "control: y of row t raised":
                sel = np.zeros(b - a, bool)
                y += 1000.0
            else:
                sel = np.zeros(b - a, bool)
                ff[t] = (ff[t].astype(np.float64) * 1.5).astype(ff.dtype)
            res[mo]["ti_values_modified"] += int(sel.sum())
            J = K.file_inputs(tm[a:b], tif, sf, np.array([t]), M.RULE)
            h10 = K.history_matrix(R["tstep"][j:j + 1], J)
            f4 = M.frame_scalars(np.array([y]), J)
            m3j = M.med3(np.array([y]), J["anc"], tif, sf)
            if mo in ("changed with fast", "control: fast of row t x 1.5"):
                wf, msb, mse, mrb, mre = file_fast(ff, sf, mu, sd, u["fit"])
            else:
                wf, msb, mse, mrb, mre = wf0, msb0, mse0, mrb0, mre0
            A = K.anchor_scalars(np.array([t]), J["anc"], msb, mse, mrb, mre)
            f14 = np.concatenate([wf[t], A[0]])[None, :]
            same_in = (bitsame(h10, main["H10"]) and bitsame(f4, main["F4"]) and bitsame(f14, main["f14"])
                       and bitsame(J["G"], main["G"]) and bitsame(m3j, main["M3"]) and int(J["anc"][0]) + a == int(I["anc_g"][j]))
            e0 = J["G"][0] + models["C0"].predict(np.column_stack([h10, f4]))[0]
            ex = J["G"][0] + models["CX"].predict(np.column_stack([h10, f4, f14]))[0]
            same_est = bitsame(np.array([e0, ex]), np.array([stored["C0"][j], stored["CX"][j]]))
            r_ = res[mo]
            r_["rows"] += 1
            r_["inputs_identical"] += int(same_in)
            r_["estimates_identical"] += int(same_est)
            r_["frame_scalars_changed"] += int(not bitsame(f4, main["F4"]))
            r_["fast_scalars_changed"] += int(not bitsame(f14, main["f14"]))
    n = len(pick)
    main_ok = all(res[mo]["inputs_identical"] == n and res[mo]["estimates_identical"] == n for mo in modes[:3])
    ctl_ok = res[modes[3]]["frame_scalars_changed"] == n and res[modes[4]]["fast_scalars_changed"] == n
    out = {"unit": LEAK_UNIT, "rows": int(n), "candidates": int(len(te)), "refit_equals_stored": refit_same,
           "file_local_path_reproduces_main_pass_rows": int(file_path_same), "modes": res,
           "passes_leakage": bool(main_ok and all(refit_same.values()) and file_path_same == n),
           "passes_positive_control": bool(ctl_ok)}
    out["passes"] = bool(out["passes_leakage"] and out["passes_positive_control"])
    M.save_json(M.CHK / "leakage.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ md
def write():
    names = ("data", "b75", "compare", "frame", "leakage")
    j = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    d, b, c, f, lk = (j[n] for n in names)
    L = ["# B.77 checks", "",
         f"Registration: `{M.REG}` section 9; implementation: `{M.ADD}` section 11. Run before any readout; no score "
         "against a target is computed here. No tolerance was changed after a run.", "",
         "## 3. Inputs and target (added)", "",
         f"Row set {d['row_set']:,} rows, {d['discharges']} discharges, {d['clusters']} shot clusters. "
         f"L5 recomputed from the grid bit-identical to `rows.npz`: {d['L5_recomputed_bit_identical']}. "
         f"H equals the grid value of the anchor row: {d['H_equals_anchor_value']}. Fewest earlier values kept on a row: "
         f"{d['min_earlier_values_kept']:.0f}. L5 above zero on every row: {d['L5_above_zero']}. MED3 defined on "
         f"{d['MED3_defined_rows']:,} rows. The reported y differs from the grid value by at most {d['y_max_abs_dev_from_grid_eV']:.1e} eV. "
         f"Anchor row = row t - 1 on {100 * d['anchor_is_row_t_minus_1_share']:.1f} % of the rows.", "",
         f"Check 3 passes: **{d['passes']}**", "",
         "## 1b. This batch's code on B.75's common set: T0 and TX against B.75's stored D10 L5 estimates (added)", "",
         "| unit / learner | same test rows | bit-identical | largest deviation (eV) | fit rows (B.75) | iterations (B.75) |",
         "|---|---|---|---|---|---|"]
    for k, x in b.items():
        if isinstance(x, dict):
            L.append(f"| {k.replace('|', ' / ')} | {x['same_test_rows']} | {x['bit_identical']} | {x['max_abs_dev_eV']:.1e} | "
                     f"{x['fit_rows']:,} ({x['fit_rows_b75']:,}) | {x['n_iter']} ({x['n_iter_b75']}) |")
    L += ["", f"Check 1b passes: **{b['passes']}**", "",
          "## 1a. T0 and TX of this batch (fitted on the B.77 row set) against B.75's D10 L5 estimates on the rows both have", "",
          c["note"] + ".", "",
          "| learner | rows both have (all units) | share identical | median abs. deviation (eV) | 90th percentile (eV) | rows above 100 eV | rows above 300 eV | largest (eV) | every B.75 test row in B.77 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for Lr in ("T0", "TX"):
        x = c[Lr]["all"]
        allin = all(v["b75_rows_all_in_b77"] for v in c[Lr]["units"].values())
        L.append(f"| {Lr} | {x['rows_both']:,} | {x['share_identical']:.3f} | {x['median_abs_dev_eV']:.2f} | {x['p90_abs_dev_eV']:.2f} | "
                 f"{x['rows_above_100_eV']:,} | {x['rows_above_300_eV']:,} | {x['max_abs_dev_eV']:.1f} | {allin} |")
    L += ["", "The fit rows differ (B.77 adds about 49,000 rows that are complete under D10 but not under every longer delay). "
              "The largest deviations lie on rows with a high and scattered history (values about 2 keV, standard deviation of "
              "the last ten values 330-690 eV), where the trees of the two fits differ. Check 1b shows that this code reproduces "
              "B.75 bit for bit on B.75's rows. Per unit: `data/.b77/checks/compare.json`.", "",
          "## 1c. C0 with the four frame scalars removed against T0", "",
          "| unit | columns | same test rows | bit-identical | largest deviation (eV) | iterations (T0) |", "|---|---|---|---|---|---|"]
    for k, x in f.items():
        if isinstance(x, dict):
            L.append(f"| {k} | {x['columns']} | {x['same_test_rows']} | {x['bit_identical']} | {x['max_abs_dev_eV']:.1e} | "
                     f"{x['n_iter']} ({x['n_iter_T0']}) |")
    L += ["", f"Check 1c passes: **{f['passes']}**", "",
          "## 2. No input of C0 or CX contains a value of a row later than t", "",
          f"Unit {lk['unit']}, {lk['rows']} test rows drawn from {lk['candidates']:,}. The C0 and CX models refitted in the check "
          f"process equal the stored ones bit for bit: {lk['refit_equals_stored']}. The file-local recomputation of the fourteen "
          f"fast scalars (B.73/B.75 code on a one-file copy, the unit's standardisation passed in) reproduces the main pass on "
          f"{lk['file_local_path_reproduces_main_pass_rows']} of {lk['rows']} rows. Compared bit for bit per row: the ten history "
          "scalars, G, the four frame scalars, MED3, the fourteen fast scalars, the anchor row, and the C0 and CX estimates.", "",
          "| modification | rows | inputs identical | estimates identical | frame scalars changed | fast scalars changed | T_i values modified | fast rows modified |",
          "|---|---|---|---|---|---|---|---|"]
    for mo, x in lk["modes"].items():
        L.append(f"| {mo} | {x['rows']} | {x['inputs_identical']} | {x['estimates_identical']} | {x['frame_scalars_changed']} | "
                 f"{x['fast_scalars_changed']} | {x['ti_values_modified']:,} | {x['fast_rows_modified']:,} |")
    L += ["", "The first three rows modify rows after t only (removed; raised by 1000 eV; T_i multiplied by factors in [0.5, 1.5], "
              "half of the missing values given a value, and the fast channels multiplied by factors in [0.5, 1.5]). The last two "
              "rows are the positive control (row t itself modified): the comparison detects a change there.", "",
          f"Check 2 passes: **{lk['passes_leakage']}**; positive control: **{lk['passes_positive_control']}**", ""]
    M.save_text(M.CHECKS_MD, "\n".join(L))
    print("written", M.CHECKS_MD)


if __name__ == "__main__":
    t0 = time.time()
    {"data": check_data, "b75": check_b75, "compare": check_compare, "frame": check_frame, "leakage": check_leakage,
     "write": write}[sys.argv[1]]()
    M.log(f"done ({time.time() - t0:.0f}s)")
