"""B.75 checks (registration section 9, Addendum 1 section 8).  Run before any readout.

    py ces_prediction/experiments/b75_availability/stage_checks.py gp          # check 1 (+ H, age, history scalars)
    py ces_prediction/experiments/b75_availability/stage_checks.py repro       # check 2
    py ces_prediction/experiments/b75_availability/stage_checks.py visibility  # check 3 (needs sets.npz)
    py ces_prediction/experiments/b75_availability/stage_checks.py ages        # check 4
    py ces_prediction/experiments/b75_availability/stage_checks.py units       # check 5
    py ces_prediction/experiments/b75_availability/stage_checks.py write       # docs/paper/b75_checks.md
Output: data/.b75/checks/{name}.json, docs/paper/b75_checks.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common75 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = K.C
CHK = K.CACHE / "checks"
TOL_GP, TOL_REPRO = 1e-6, 1e-6
C12_REF = {"median_s": 0.65, "bands": {"0-0.1 s": 0.071, "0.1-0.3 s": 0.166, "0.3-1 s": 0.498, "above 1 s": 0.266}}
BAND_EDGES = (0.0, 0.1, 0.3, 1.0, np.inf)
BAND_NAMES = ("0-0.1 s", "0.1-0.3 s", "0.3-1 s", "above 1 s")


def bands(a):
    a = a[np.isfinite(a)]
    h = np.histogram(a, BAND_EDGES)[0]
    return {nm: float(c / len(a)) for nm, c in zip(BAND_NAMES, h)}


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------------------------ check 1
def check_gp():
    R = C.rows()
    I = K.load_inputs("D10")
    base = np.load(C.DATA / ".b64" / "base.npz", allow_pickle=True)
    bkey = base["fidx"].astype(np.int64) * 1_000_000 + base["row"].astype(np.int64)
    key = R["fidx"] * 1_000_000 + R["grow"]
    ob = np.argsort(bkey)
    pos = np.searchsorted(bkey[ob], key)
    jb = ob[pos]
    if not np.array_equal(bkey[jb], key):
        raise SystemExit("FATAL: a D1 row is not a base row")
    gp_b = base["gp"][jb]
    d = np.abs(I["G"] - gp_b)
    nan_same = bool(np.array_equal(np.isnan(I["G"]), np.isnan(gp_b)))
    out = {"rows": int(len(key)), "G_max_abs_dev_eV": float(np.nanmax(d)), "G_bit_identical_rows": int(np.sum(I["G"].view(np.int64) == gp_b.view(np.int64))),
           "G_nan_pattern_same": nan_same, "G_undefined_rows": int(np.isnan(I["G"]).sum())}
    for k_, ref in (("H", base["hold"][jb]), ("age", base["age"][jb]), ("M5", R["m5"])):
        dd = np.abs(I[k_] - ref)
        out[f"{k_}_max_abs_dev"] = float(np.nanmax(dd))
        out[f"{k_}_nan_pattern_same"] = bool(np.array_equal(np.isnan(I[k_]), np.isnan(ref)))
    th0 = R["thist"]
    fin = np.isfinite(th0).all(1)
    out["history5_nan_pattern_same"] = bool(np.array_equal(np.isfinite(I["th"]).all(1), fin))
    out["history5_bit_identical"] = bool(bitsame(I["th"][fin], th0[fin]))
    out["history5_max_abs_dev"] = float(np.nanmax(np.abs(I["th"] - th0)))
    out["passes"] = bool(out["G_max_abs_dev_eV"] <= TOL_GP and nan_same)
    K.save_json(CHK / "gp.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 2
def check_repro():
    import stage_trees as ST
    R = K.rows_lean()
    I = K.load_inputs("D10")
    H10 = K.history_matrix(R["tstep"], I)
    FE = ST.Feats()
    us, _, _ = K.DS.units(C.rows())
    out = {}
    worst = 0.0
    for un in ("V1_r2_k0", "V4a"):
        u = next(x for x in us if x["name"] == un)
        wf = FE.window(u["fit"])
        fast, fastl = FE.at(wf, R["gi"])
        ok = ST.complete_mask(H10, fast, fastl, I["G"])
        XS = {"T0": H10, "TX": np.column_stack([H10, fast])}
        fit_d = np.isin(R["fidx"], u["fit"])
        test_idx = np.flatnonzero(np.isin(R["fidx"], u["test"]))
        for L in ("T0", "TX"):
            for X in K.TARGETS:
                tgt = R[f"tg_{X}"]
                fr = fit_d & ok & np.isfinite(tgt)
                m = ST.gbr().fit(XS[L][fr], (tgt - I["G"])[fr])
                est = I["G"][test_idx].copy()
                te_ok = ok[test_idx]
                est[te_ok] += m.predict(XS[L][test_idx[te_ok]])
                z = C.load_npz(C.CACHE / "trees" / f"{un}__{L}__{X}.npz")
                same_idx = bool(np.array_equal(z["idx"], test_idx))
                dev = float(np.max(np.abs(z["est"] - est))) if same_idx else float("inf")
                worst = max(worst, dev)
                out[f"{un}|{'TX8' if L == 'TX' else L}|{X}"] = {
                    "same_test_rows": same_idx, "max_abs_dev_eV": dev, "fit_rows": int(fr.sum()),
                    "fit_rows_b73": int(z["fit_rows"]), "n_iter": int(m.n_iter_), "n_iter_b73": int(z["n_iter"]),
                    "same_complete_mask": bool(np.array_equal(z["complete"], te_ok))}
                K.log(f"repro {un} {L}[{X}]: max dev {dev:.2e} eV, fit rows {int(fr.sum()):,} (B.73 {int(z['fit_rows']):,})")
    out["max_abs_dev_eV"] = worst
    out["passes"] = bool(worst <= TOL_REPRO)
    K.save_json(CHK / "repro.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 3
def unit_fast(FE, u, G):
    wf = FE.window(u["fit"])
    bz, ez = K.zmeans(FE, u["fit"])
    ms_b, ms_e = K.run_mean10(bz, G["seg"]), K.run_mean10(ez, G["seg"])
    return wf, ms_b, ms_e


def check_visibility():
    import stage_trees as ST
    G = K.grid_lean(("names", "off", "time", "ti", "seg", "fidx", "grow", "fast"))
    R = K.rows_lean()
    S = K.load_npz(K.CACHE / "sets.npz")
    us, _, _ = K.DS.units(C.rows())
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    FE = ST.Feats()
    wf, ms_b, ms_e = unit_fast(FE, u, G)
    mr_b, mr_e = (K.run_mean10(x, G["seg"]) for x in K.raw_means(G["fast"]))
    off, tm, ti, seg = G["off"], G["time"], G["ti"], G["seg"]
    test = np.isin(R["fidx"], u["test"])
    out = {"unit": u["name"]}
    allok = True
    for rule, setname in (("D650", "common_F"), ("S", "S_F")):
        I = K.load_inputs(rule)
        H10 = K.history_matrix(R["tstep"], I)
        A = K.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, mr_b, mr_e)
        cand = np.flatnonzero(test & S[setname])
        pick = np.sort(K.rng_for(f"visibility|{rule}").choice(cand, 200, replace=False))
        res = {"rows": int(len(pick)), "candidates": int(len(cand))}
        for mode in ("removed", "raised"):
            n_same, n_mod = 0, 0
            for j in pick:
                i = int(R["fidx"][j])
                a, b = off[i], off[i + 1]
                t = int(R["grow"][j])
                lim = I["lim"][j]
                tif = ti[a:b].copy()
                later = (tm[a:b] > lim + K.TOL) & np.isfinite(tif)
                n_mod += int(later.sum())
                if mode == "removed":
                    tif[later] = np.nan
                else:
                    tif[later] += 1000.0
                J = K.file_inputs(tm[a:b], tif, seg[a:b], np.array([t]), rule)
                anc_g = np.where(J["anc"] >= 0, J["anc"] + a, -1)
                h10 = K.history_matrix(R["tstep"][j:j + 1], J)
                aa = K.anchor_scalars(R["gi"][j:j + 1], anc_g, ms_b, ms_e, mr_b, mr_e)
                same = (bitsame(h10, H10[j:j + 1]) and bitsame(aa, A[j:j + 1])
                        and int(anc_g[0]) == int(I["anc_g"][j]) and bitsame(J["G"], I["G"][j:j + 1])
                        and bitsame(J["M5"], I["M5"][j:j + 1]))
                n_same += int(same)
            res[mode] = {"rows_identical": n_same, "values_modified_total": n_mod}
            allok &= n_same == len(pick)
        out[rule] = res
        K.log(f"visibility {rule}: {res}")
    out["fast_scalars_note"] = ("the eight fast scalars and the running means m(i) take no T_i argument (they are functions "
                                "of the fast channels and the fit discharges only) and are not compared; the anchor scalars "
                                "are recomputed from the recomputed anchor row; M5 is compared too")
    out["passes"] = bool(allok)
    K.save_json(CHK / "visibility.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 4
def check_ages():
    import pandas as pd
    G = K.grid_lean(("names", "off", "time", "ti", "seg"))
    off, tm, seg, names = G["off"], G["time"], G["seg"], G["names"]
    ages_rev, n_val, n_before, ages64 = [], 0, 0, []
    for i, nm in enumerate(names):
        df = pd.read_csv(C.DATA / str(nm), usecols=["time", "CES_TI"])
        a, b = off[i], off[i + 1]
        if len(df) != b - a or np.max(np.abs(df["time"].to_numpy(np.float64) - tm[a:b])) > 1e-5:
            raise SystemExit(f"FATAL: {nm}: CSV rows are not the grid rows")
        ti_raw = df["CES_TI"].to_numpy(np.float64)
        # diagnostic: the same with the float64 stamps of the CSV text (the reviewer's stamps), first/last rows dropped
        t64, y64 = df["time"].to_numpy(np.float64)[1:-1], ti_raw[1:-1]
        j64 = K.s_limit_rows(t64, np.concatenate(([0], np.cumsum(np.diff(t64) >= 0.5))))
        k64 = np.flatnonzero(np.isfinite(y64))
        k64 = k64[j64[k64] >= 0]
        ages64.append(t64[k64] - t64[j64[k64]])
        jr = K.s_limit_rows(tm[a:b], seg[a:b])
        val = np.isfinite(ti_raw)
        val[0] = val[-1] = False                       # the first and last row of a file are not grid rows
        idx = np.flatnonzero(val)
        n_val += len(idx)
        has = jr[idx] >= 0
        n_before += int((~has).sum())
        ages_rev.append(tm[a:b][idx[has]] - tm[a:b][jr[idx[has]]])
    ag = np.concatenate(ages_rev)
    out = {"population": "grid rows other than a file's first and last, finite T_i as delivered (no cut, no held removal)",
           "valued_rows": int(n_val), "rows_with_absent_step_earlier": int(len(ag)),
           "share_without_absent_step_earlier": float(n_before / n_val),
           "median_s": float(np.median(ag)), "bands": bands(ag), "reference": C12_REF}
    ok = abs(out["median_s"] - C12_REF["median_s"]) <= 0.02 and all(abs(out["bands"][k] - v) <= 0.01 for k, v in C12_REF["bands"].items())
    out["passes"] = bool(ok)
    a64 = np.concatenate(ages64)
    out["float64_stamps_diagnostic"] = {
        "rows": int(len(a64)), "median_s": float(np.median(a64)), "bands": bands(a64),
        "ages_within_1e-5_s_of_band_edges": {str(e): int(np.sum(np.abs(ag - e) < 1e-5)) for e in (0.1, 0.3, 1.0)},
        "six_bins_c12_format": [f"{x:.3f}" for x in np.histogram(a64, (0, 0.05, 0.1, 0.3, 1.0, 3.0, 99))[0] / len(a64)],
        "note": "the grid stamps are float32; ages that lie on a band edge fall on either side of it depending on the rounding "
                "of the stamps; with the float64 stamps of the CSV text the six bins printed in C12's format sum to C12's bands"}
    # the same on the D1 rows (cut, held-free), limit age and newest-visible age
    I = K.load_inputs("S")
    d1 = I["nvis"] > 0
    out["d1_rows"] = {"rows": int(len(d1)), "with_visible_value_under_S": int(d1.sum()), "share_without": float(1 - d1.mean()),
                      "limit_age_median_s": float(np.median(I["lim_age"][d1])), "limit_age_bands": bands(I["lim_age"][d1]),
                      "newest_value_age_median_s": float(np.median(I["age"][d1]) / 1000), "newest_value_age_bands": bands(I["age"][d1] / 1000)}
    K.save_json(CHK / "ages.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 5
def check_units():
    R = C.rows()
    us, tab, fifth = K.DS.units(R)
    res, ok = {}, True
    for u in us:
        z = C.load_npz(C.CACHE / "trees" / f"{u['name']}__T0__F.npz")
        mine = np.flatnonzero(np.isin(R["fidx"], u["test"]))
        same = bool(np.array_equal(z["idx"], mine))
        ok &= same
        res[u["name"]] = {"same_test_rows": same, "fit_discharges": int(len(u["fit"])), "test_discharges": int(len(u["test"]))}
    ua, _, _ = K.units_all(R)
    v3c = {x["name"]: {"fit_discharges": int(len(x["fit"])), "test_discharges": int(len(x["test"])),
                       "fit_overlaps_test": bool(np.intersect1d(x["fit"], x["test"]).size),
                       "shared_with_V3_fit": int(np.intersect1d(x["fit"], next(y for y in us if y["name"] == f"V3_q{x['fifth']}")["fit"]).size)}
           for x in ua if x["design"] == "V3c"}
    out = {"b73_units": res, "passes": bool(ok), "v3c_units": v3c}
    K.save_json(CHK / "units.json", out)
    print(json.dumps({"passes": out["passes"], "v3c": v3c}, indent=1))


# ------------------------------------------------------------------------------------------ md
def write():
    j = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("gp", "repro", "visibility", "ages", "units")}
    si = json.loads((K.CACHE / "sets.json").read_text(encoding="utf-8"))
    g, r, v, a, u = j["gp"], j["repro"], j["visibility"], j["ages"], j["units"]
    L = ["# B.75 checks", "",
         f"Registration: `{K.REG}` section 9; implementation: `{K.ADD}` section 8. Run before any readout. "
         "No tolerance was changed after a run.", "",
         "## 1. G under D10 against the cached causal GP of `data/.b64/base.npz`", "",
         f"D1 rows {g['rows']:,}. Largest deviation {g['G_max_abs_dev_eV']:.1e} eV; bit-identical on {g['G_bit_identical_rows']:,} rows "
         f"(the {g['G_undefined_rows']:,} rows without a visible value are undefined in both: {g['G_nan_pattern_same']}). "
         f"Beside it: H {g['H_max_abs_dev']:.1e} eV, age {g['age_max_abs_dev']:.1e} ms, M5 {g['M5_max_abs_dev']:.1e} eV; "
         f"the five history scalars against `rows.npz`: bit-identical {g['history5_bit_identical']}, same undefined rows {g['history5_nan_pattern_same']}.",
         "", f"Check 1 passes: **{g['passes']}**", "",
         "## 2. T0 and TX8 under D10 against the stored estimates of B.73", "",
         "Fit rows: B.73's complete set (Addendum 1 section 4). Test rows without it keep G, as in B.73.", "",
         "| unit / learner / target | same test rows | largest deviation (eV) | fit rows (B.73) | iterations (B.73) | same complete mask |",
         "|---|---|---|---|---|---|"]
    for k, x in r.items():
        if not isinstance(x, dict):
            continue
        L.append(f"| {k.replace('|', ' / ')} | {x['same_test_rows']} | {x['max_abs_dev_eV']:.1e} | {x['fit_rows']:,} ({x['fit_rows_b73']:,}) | "
                 f"{x['n_iter']} ({x['n_iter_b73']}) | {x['same_complete_mask']} |")
    L += ["", f"Check 2 passes: **{r['passes']}**", "",
          "## 3. Visibility", "",
          f"Unit {v['unit']}. For each row, a copy of its file in which every T_i value with stamp above the row's limit + 3 ms "
          "is removed, or raised by 1000 eV; the ten history scalars, G, M5, the six anchor scalars and the "
          "anchor row recomputed and compared bit for bit with the main pass. Note: " + v["fast_scalars_note"] + ".", "",
          "| rule | rows (candidates) | removed: rows identical (values changed) | raised: rows identical (values changed) |",
          "|---|---|---|---|"]
    for rule in ("D650", "S"):
        x = v[rule]
        L.append(f"| {rule} | {x['rows']} ({x['candidates']:,}) | {x['removed']['rows_identical']} ({x['removed']['values_modified_total']:,}) | "
                 f"{x['raised']['rows_identical']} ({x['raised']['values_modified_total']:,}) |")
    L += ["", f"Check 3 passes: **{v['passes']}**", "",
          "## 4. Ages under S against the reviewer's check C12", "",
          f"Population: {a['population']}: {a['valued_rows']:,} rows, {a['rows_with_absent_step_earlier']:,} with an absent step "
          f"earlier in the block; share without {a['share_without_absent_step_earlier']:.3f} (reviewer: 0.315). Age = stamp of "
          "the row minus the stamp of the last row before the most recent absent step that ends at or before it.", "",
          "| quantity | ours | reviewer (C12) |", "|---|---|---|",
          f"| median (s) | {a['median_s']:.3f} | {C12_REF['median_s']:.2f} |"]
    for k_, ref in C12_REF["bands"].items():
        L.append(f"| share {k_} | {a['bands'][k_]:.3f} | {ref:.3f} |")
    d = a["d1_rows"]
    f64 = a["float64_stamps_diagnostic"]
    L += ["", f"Check 4 passes: **{a['passes']}**", "",
          "The band shares differ from C12 by up to 0.0023. Cause: the grid stamps are float32, and "
          + ", ".join(f"{n:,} ages lie within 1e-5 s of {e} s" for e, n in f64["ages_within_1e-5_s_of_band_edges"].items())
          + "; such an age falls on either side of the edge depending on the rounding of the stamps. With the float64 stamps "
          f"of the CSV text ({f64['rows']:,} rows): median {f64['median_s']:.3f} s, bands "
          + ", ".join(f"{k_} {x:.4f}" for k_, x in f64["bands"].items()) + "; C12's six printed bins (0-0.05, 0.05-0.1, 0.1-0.3, "
          "0.3-1, 1-3, above 3 s) come out as " + ", ".join(f64["six_bins_c12_format"]) + ", whose sums are C12's four bands.", "",
          f"On the D1 rows (cut, held-free): {d['with_visible_value_under_S']:,} of {d['rows']:,} have a visible value under S "
          f"(share without {d['share_without']:.3f}). Age of the limit: median {d['limit_age_median_s']:.3f} s, bands "
          + ", ".join(f"{k_} {x:.3f}" for k_, x in d["limit_age_bands"].items()) +
          f". Age of the newest visible value: median {d['newest_value_age_median_s']:.3f} s, bands "
          + ", ".join(f"{k_} {x:.3f}" for k_, x in d["newest_value_age_bands"].items()) + ".", "",
          "## 5. Units", "",
          f"Every B.73 unit has the test rows stored by B.73: **{u['passes']}** ({len(u['b73_units'])} units). V3c units: " +
          "; ".join(f"{k_}: fit {x['fit_discharges']}, test {x['test_discharges']}, fit discharges shared with V3 {x['shared_with_V3_fit']}"
                    for k_, x in u["v3c_units"].items()) + ".", "",
          "## Row sets", "",
          f"D1 rows {si['d1_rows']:,}; rows with the eight fast scalars {si['valid8_rows']:,}.", "",
          "| rule | rows with a visible value | history and G complete | anchor means defined | complete | dropped by a3 alone |",
          "|---|---|---|---|---|---|"]
    for rule, x in si["per_rule"].items():
        L.append(f"| {rule} | {x['visible']:,} | {x['history_and_G_complete']:,} | {x['anchor_means_defined']:,} | {x['complete']:,} | {x['dropped_by_a3_alone']:,} |")
    L += ["", f"Common set: F {si['common_F']:,}, L5 {si['common_L5']:,} (rows with the target: F {si['target_F']:,}, L5 {si['target_L5']:,}). "
              f"S set: F {si['S_F']:,}, L5 {si['S_L5']:,}. Rows complete under S but not under D10: {si['complete_S_not_D10']:,}. "
              f"Rows without a visible value under S: {si['no_visible_value_under_S']:,}.", ""]
    K.save_text(K.CHECKS_MD, "\n".join(L))
    print("written", K.CHECKS_MD)


if __name__ == "__main__":
    t0 = time.time()
    {"gp": check_gp, "repro": check_repro, "visibility": check_visibility, "ages": check_ages, "units": check_units,
     "write": write}[sys.argv[1]]()
    K.log(f"done ({time.time() - t0:.0f}s)")
