"""B.78 checks (registration section 10, Addendum 1 section 8).  Run before any readout.

    py ces_prediction/experiments/b78_sparse/checks78.py frames      # frame index facts; scored rows on no kept grid
    py ces_prediction/experiments/b78_sparse/checks78.py k1          # K1 inputs = B.75 D10 inputs on the scored rows
    py ces_prediction/experiments/b78_sparse/checks78.py visibility  # K10, 200 test rows, removed / raised copies
    py ces_prediction/experiments/b78_sparse/checks78.py units       # units = B.73 units
    py ces_prediction/experiments/b78_sparse/checks78.py write       # docs/paper/b78_checks.md
Output: data/.b78/checks/{name}.json, docs/paper/b78_checks.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common78 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
CHK = M.CHK


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------------------------ frames
def check_frames():
    G = M.grid_lean(("time", "ti", "seg", "fidx", "grow"))
    R = M.rows_lean()
    F = M.load_frames()
    fr, sc = F["fr"], F["scored"]
    info = json.loads((M.CACHE / "frames.json").read_text(encoding="utf-8"))
    fr2, n_off = M.frame_index(G)
    meas = np.isfinite(G["ti"])
    out = {**info, "recomputed_identical": bool(np.array_equal(fr2, fr)),
           "measured_rows": int(meas.sum()), "measured_rows_without_index": int((meas & (fr < 0)).sum())}
    # largest distance of a stamp from its multiple of 10 ms (rows with an index)
    tm, seg, grow = G["time"], G["seg"], G["grow"]
    ok = fr >= 0
    # the stamp of the index-0 row of each block (s0) recovered as the stamp minus 10 ms x index of the row with index 0
    blk_first = {}
    for i in np.flatnonzero(ok & (fr == 0)):
        blk_first.setdefault(int(seg[i]), tm[i])
    s0 = np.array([blk_first.get(int(s), np.nan) for s in seg[ok]])
    dev = np.abs(tm[ok] - s0 - M.LAT * fr[ok])
    out["max_stamp_deviation_ms"] = float(np.nanmax(dev) * 1000)
    out["rows_without_block_origin"] = int(np.isnan(s0).sum())
    f_sc = fr[R["gi"][sc]]
    out["scored_on_a_kept_grid"] = {f"K{k}": int((f_sc % k == 0).sum()) for k in M.KS[1:]}
    out["scored_rows_by_position_mod_10"] = {str(v): int(c) for v, c in zip(*np.unique(f_sc % 10, return_counts=True))}
    out["passes"] = bool(out["rows_off_lattice"] == 0 and out["measured_rows_without_index"] == 0 and
                         all(v == 0 for v in out["scored_on_a_kept_grid"].values()) and out["recomputed_identical"])
    M.save_json(CHK / "frames.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ K1 = D10
def check_k1():
    F = M.load_frames()
    sc = F["scored"]
    I = M.load_inputs("K1")
    D = M.load_npz(C.DATA / ".b75" / "inputs" / "D10.npz")
    out = {"scored_rows": int(sc.sum())}
    ok = True
    for key in ("H", "age", "M5", "G", "th", "anc_g", "nvis"):
        a, b = I[key][sc], D[key][sc]
        if a.dtype.kind == "f":
            same = bitsame(a, b)
            dev = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
            nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            out[key] = {"bit_identical": same, "max_abs_dev": dev, "same_undefined_rows": nan_same,
                        "undefined_rows": int(np.isnan(a).any(axis=-1).sum() if a.ndim > 1 else np.isnan(a).sum())}
        else:
            same = bool(np.array_equal(a, b))
            out[key] = {"identical": same}
        ok &= same
    out["passes"] = bool(ok)
    M.save_json(CHK / "k1.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ visibility
def check_visibility():
    import stage_trees as ST
    G = M.grid_lean(("off", "time", "ti", "seg", "fidx", "grow", "fast"))
    R = M.rows_lean()
    F = M.load_frames()
    S = M.load_npz(M.CACHE / "sets.npz")
    us, _, _ = M.units(C.rows())
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    FE = ST.Feats()
    bz, ez = M.zmeans(FE, u["fit"])
    ms_b, ms_e = M.run_mean10(bz, G["seg"]), M.run_mean10(ez, G["seg"])
    mr_b, mr_e = (M.run_mean10(x, G["seg"]) for x in M.raw_means(G["fast"]))
    off, tm, ti, seg, fr = G["off"], G["time"], G["ti"], G["seg"], F["fr"]
    test = np.isin(R["fidx"], u["test"])
    rule, k = "K10", 10
    I = M.load_inputs(rule)
    H10 = M.history_matrix(R["tstep"], I)
    A = M.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, mr_b, mr_e)
    cand = np.flatnonzero(test & S["common_L5"])
    pick = np.sort(M.rng_for(f"visibility|{rule}").choice(cand, 200, replace=False))
    res = {"unit": u["name"], "rule": rule, "rows": int(len(pick)), "candidates": int(len(cand))}
    allok = True
    for mode in ("removed", "raised"):
        n_same, n_mod = 0, 0
        for j in pick:
            i = int(R["fidx"][j])
            a, b = off[i], off[i + 1]
            t = int(R["grow"][j])
            tif = ti[a:b].copy()
            frf = fr[a:b]
            mod = np.isfinite(tif) & ((frf < 0) | (frf % k != 0) | (np.arange(b - a) >= t))
            n_mod += int(mod.sum())
            if mode == "removed":
                tif[mod] = np.nan
            else:
                tif[mod] += 1000.0
            J = M.file_inputs_k(tm[a:b], tif, seg[a:b], frf, np.array([t]), k)
            anc_g = np.where(J["anc"] >= 0, J["anc"] + a, -1)
            h10 = M.history_matrix(R["tstep"][j:j + 1], J)
            aa = M.anchor_scalars(R["gi"][j:j + 1], anc_g, ms_b, ms_e, mr_b, mr_e)
            same = (bitsame(h10, H10[j:j + 1]) and bitsame(aa, A[j:j + 1]) and int(anc_g[0]) == int(I["anc_g"][j])
                    and bitsame(J["G"], I["G"][j:j + 1]) and bitsame(J["M5"], I["M5"][j:j + 1])
                    and int(J["nvis"][0]) == int(I["nvis"][j]))
            n_same += int(same)
        res[mode] = {"rows_identical": n_same, "values_modified_total": n_mod}
        allok &= n_same == len(pick)
        M.log(f"visibility {mode}: {n_same}/{len(pick)} identical")
    res["note"] = ("the eight fast scalars and the running means m(i) take no T_i argument and are not recomputed; the "
                   "anchor scalars are recomputed from the recomputed anchor row; M5 and the number of visible values are "
                   "compared too")
    res["passes"] = bool(allok)
    M.save_json(CHK / "visibility.json", res)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------ units
def check_units():
    R = C.rows()
    us, _, _ = M.units(R)
    res, ok = {}, True
    for u in us:
        z = C.load_npz(C.CACHE / "trees" / f"{u['name']}__T0__F.npz")
        mine = np.flatnonzero(np.isin(R["fidx"], u["test"]))
        same = bool(np.array_equal(z["idx"], mine))
        ok &= same
        res[u["name"]] = {"same_test_rows": same, "fit_discharges": int(len(u["fit"])), "test_discharges": int(len(u["test"]))}
    out = {"b73_units": res, "passes": bool(ok)}
    M.save_json(CHK / "units.json", out)
    print(json.dumps({"passes": out["passes"], "units": len(res)}, indent=1))


# ------------------------------------------------------------------------------------------ md
def write():
    j = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("frames", "k1", "visibility", "units")}
    si = json.loads((M.CACHE / "sets.json").read_text(encoding="utf-8"))
    f, k1, v, u = j["frames"], j["k1"], j["visibility"], j["units"]
    L = ["# B.78 checks", "",
         f"Registration: `{M.REG}` section 10; implementation: `{M.ADD}` section 8. Run before any readout. "
         "No tolerance was changed after a run.", "",
         "## 1. Frame index and scored rows", "",
         f"Grid rows {f['grid_rows']:,}; rows without a frame index {f['grid_rows_without_index']:,} (each file's first and last "
         f"row, which are not grid rows); rows off the 10 ms lattice by more than 3 ms: {f['rows_off_lattice']}. Largest distance "
         f"of a stamp from the first grid row of its block plus 10 ms x its index: {f['max_stamp_deviation_ms']:.4f} ms. "
         f"Measured rows {f['measured_rows']:,}, of them without an index: {f['measured_rows_without_index']}. "
         f"D1 rows {f['d1_rows']:,}; scored rows (index odd and not a multiple of 5) {f['scored_rows']:,}; by index modulo 10: "
         + ", ".join(f"{k_}: {c:,}" for k_, c in f["scored_rows_by_position_mod_10"].items()) + ". Scored rows whose index is "
         "a multiple of k: " + ", ".join(f"{k_} {c}" for k_, c in f["scored_on_a_kept_grid"].items()) + ".", "",
         f"Check passes: **{f['passes']}**", "",
         "## 2. Inputs under K1 against B.75's D10 inputs (`data/.b75/inputs/D10.npz`) on the scored rows", "",
         f"Scored rows {k1['scored_rows']:,}.", "", "| input | bit-identical | largest deviation | same undefined rows |", "|---|---|---|---|"]
    for key in ("H", "age", "M5", "G", "th", "anc_g", "nvis"):
        x = k1[key]
        if "bit_identical" in x:
            L.append(f"| {key} | {x['bit_identical']} | {x['max_abs_dev']:.1e} | {x['same_undefined_rows']} ({x['undefined_rows']:,} rows undefined) |")
        else:
            L.append(f"| {key} | {x['identical']} | - | - |")
    L += ["", "The six anchor scalars are functions of the anchor row (`anc_g`) and of the unit only, so they are identical too. "
              "(`th` = the five history scalars; `anc_g` = the anchor row; `nvis` = the number of visible values of the block.)", "",
          f"Check passes: **{k1['passes']}**", "",
          "## 3. Visibility under K10", "",
          f"Unit {v['unit']}; {v['rows']} rows drawn from its {v['candidates']:,} test rows in the common set (L5). For each row, a copy "
          "of its file in which every T_i value off the kept grid (index not a multiple of 10) or not before row t is removed, "
          "or raised by 1000 eV; the ten history scalars, G, M5, the number of visible values, the anchor row and the six "
          "anchor scalars recomputed and compared bit for bit with the main pass. Note: " + v["note"] + ".", "",
          "| copy | rows identical | values changed (total) |", "|---|---|---|",
          f"| removed | {v['removed']['rows_identical']} / {v['rows']} | {v['removed']['values_modified_total']:,} |",
          f"| raised by 1000 eV | {v['raised']['rows_identical']} / {v['rows']} | {v['raised']['values_modified_total']:,} |", "",
          f"Check passes: **{v['passes']}**", "",
          "## 4. Units", "",
          f"Every B.73 unit has the test rows stored by B.73: **{u['passes']}** ({len(u['b73_units'])} units).", "",
          "## Row sets", "",
          f"D1 rows {si['d1_rows']:,}; scored rows {si['scored_rows']:,}; scored rows with the eight fast scalars {si['scored_valid8_rows']:,}.", "",
          "| rule | scored rows with a visible value | of them fewer than three | history and G complete | anchor means defined | complete | dropped by a3 alone |",
          "|---|---|---|---|---|---|---|"]
    for rule, x in si["per_rule"].items():
        L.append(f"| {rule} | {x['visible']:,} | {x['fewer_than_three_visible']:,} | {x['history_and_G_complete']:,} | "
                 f"{x['anchor_means_defined']:,} | {x['complete']:,} | {x['dropped_by_a3_alone']:,} |")
    L += ["", f"Complete under all six rules: {si['complete_under_all_six']:,}. Common set: F {si['common_F']:,}, "
              f"L5 {si['common_L5']:,} (scored rows with the target: F {si['scored_with_target_F']:,}, L5 {si['scored_with_target_L5']:,}).", ""]
    M.save_text(M.CHECKS_MD, "\n".join(L))
    print("written", M.CHECKS_MD)


if __name__ == "__main__":
    t0 = time.time()
    {"frames": check_frames, "k1": check_k1, "visibility": check_visibility, "units": check_units, "write": write}[sys.argv[1]]()
    M.log(f"done ({time.time() - t0:.0f}s)")
