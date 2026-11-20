"""B.86 checks (registration section 6, Addendum 1 section 9).  Each run once, before any readout.  No error, skill or
interval is computed here.

    py ces_prediction/experiments/b86_mast_radius/stage_checks86.py band_e
    py ces_prediction/experiments/b86_mast_radius/stage_checks86.py visibility
    py ces_prediction/experiments/b86_mast_radius/stage_checks86.py references
    py ces_prediction/experiments/b86_mast_radius/stage_checks86.py rows
    py ces_prediction/experiments/b86_mast_radius/stage_checks86.py write
Output: data/.b86/checks/{name}.json, docs/paper/b86_checks.md.
"""
from __future__ import annotations

import copy
import json
import sys

import common86 as C
import stage_fits86 as F
import stage_refs86 as RF

import numpy as np  # noqa: E402

K = C.K
B76 = K.CACHE


def bitsame(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def bitsame64(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def done(name, res):
    C.save_json(C.CHK / f"{name}.json", res)
    C.log(f"check {name}: passes {res['passes']}")
    if not res["passes"]:
        sys.exit(2)


# ------------------------------------------------------------------------------------------ band E
def check_band_e():
    fields = [k for k in C.ROW_KEYS]
    mine = C.load_band("E", fields)
    old = np.load(B76 / "rows.npz", allow_pickle=False)
    res = {"rows_this_build": int(len(mine["shot"])), "rows_b76": int(len(old["shot"])), "fields": {}}
    ok = res["rows_this_build"] == res["rows_b76"]
    for k in fields:
        same = bitsame(mine[k], old[k])
        res["fields"][k] = {"identical": bool(same), "dtype_this": str(mine[k].dtype), "dtype_b76": str(old[k].dtype)}
        ok &= same
    del old
    # (b) T0 and TX of M1_s1_k0 under D100, L5, refitted here with B.76's set and unit
    R = C.load_band("E", F.KEYS[:7] + [k for k in F.KEYS if k.startswith("D100_")] + [k for k in F.KEYS if k.startswith("D5_")])
    S = np.load(B76 / "sets.npz", allow_pickle=False)
    us = json.loads((B76 / "units.json").read_text(encoding="utf-8"))["units"]
    u = next(x for x in us if x["name"] == "M1_s1_k0")
    sm = S["common_L5"]
    fr = np.flatnonzero(np.isin(R["shot"], np.array(u["fit"])) & sm)
    te = np.flatnonzero(np.isin(R["shot"], np.array(u["test"])) & sm)
    res["refit"] = {}
    for L in ("T0", "TX"):
        z = np.load(B76 / "fits" / f"M1_s1_k0__{L}__D100__L5.npz")
        est, nit = F.fit_estimate(R, L, "D100", "L5", fr, te)
        a = bitsame(te.astype(np.int64), z["idx"].astype(np.int64))
        bq = bitsame64(est, z["est"])
        res["refit"][L] = {"fit_rows": int(len(fr)), "fit_rows_b76": int(z["fit_rows"]), "test_rows": int(len(te)),
                           "idx_identical": bool(a), "estimates_identical": bool(bq), "n_iter": nit,
                           "n_iter_b76": int(z["n_iter"])}
        ok &= a and bq and len(fr) == int(z["fit_rows"])
    res["passes"] = bool(ok)
    done("band_e", res)


# ------------------------------------------------------------------------------------------ visibility
def check_visibility():
    b, rule = "B4", "D100"
    dl = C.DELAYS[rule]
    keys = ["shot", "craw", "c", "j", "t", "R", "common_L5", "te_t", "ne_t", "dedge_t"] + \
           [f"{rule}_{k}" for k in ("H", "age", "M5", "G", "th", "anc", "nvis", "te_a", "ne_a")]
    R = C.load_band(b, keys)
    us, _ = C.units(b)
    u = next(x for x in us if x["name"] == "M1_s1_k0")
    cand = np.flatnonzero(R["common_L5"] & np.isin(R["shot"], np.array(u["test"])))
    pick = np.sort(C.rng_for("visibility|B4|D100").choice(cand, 200, replace=False))
    cat = K.meta_tables()
    ts_old = K.ts_matrix(R, rule)
    same, changed, fails = 0, 0, []
    cache = {}
    for i in pick:
        s = int(R["shot"][i])
        if s not in cache:
            cache.clear()
            st, P = K.prepare_shot(s, cat)
            Rc = P["d"]["R"][P["chords"]].astype(np.float64)
            t_a0 = float(P["d"]["time_ms"][P["a"]])
            cache[s] = (P, Rc, t_a0)
        P, Rc, t_a = cache[s]
        d, a = P["d"], P["a"]
        if abs(t_a + C.STEP * R["j"][i] - R["t"][i]) > 1e-9:
            raise SystemExit("FATAL: frame time mismatch")
        lim = R["t"][i] - dl
        nfr = d["T"].shape[0]
        tn = t_a + C.STEP * (np.arange(nfr) - a)
        later = tn > lim + C.TOL_MS
        cr = int(R["craw"][i])
        d2 = copy.deepcopy(d)
        changed += int(np.isfinite(d2["T"][later]).sum())
        d2["T"][later] = d2["T"][later] + 1000.0
        good2 = K.rg.good_masks(d2, K.REL)["T_i"]
        fr10 = np.nonzero(good2.sum(1) >= 10)[0]
        ok = len(fr10) > 0 and int(fr10[0]) == a
        L0 = P["p"]["L"]
        y2 = d2["T"][a:a + L0, cr].astype(np.float64)
        g2 = good2[a:a + L0, cr]
        tf2 = C.STEP * np.arange(L0)
        I = K.chord_history(tf2, y2, g2, np.array([R["j"][i]]), dl)
        anc = int(I["anc"][0])
        TEw, NEw, dedge, *_ = K.thomson_grids(P["ex"], Rc, t_a + C.STEP * np.arange(L0), P["geo"][0])
        c, j = int(R["c"][i]), int(R["j"][i])
        one = {"te_t": np.array([TEw[j, c]]), "ne_t": np.array([NEw[j, c]]), "dedge_t": np.array([dedge[j, c]]),
               f"{rule}_te_a": np.array([TEw[anc, c] if anc >= 0 else np.nan]),
               f"{rule}_ne_a": np.array([NEw[anc, c] if anc >= 0 else np.nan])}
        ts_new = K.ts_matrix(one, rule)[0]
        vals = {"anc": (anc, R[f"{rule}_anc"][i]), "nvis": (I["nvis"][0], R[f"{rule}_nvis"][i]),
                "H": (I["H"][0], R[f"{rule}_H"][i]), "age": (I["age"][0], R[f"{rule}_age"][i]),
                "M5": (I["M5"][0], R[f"{rule}_M5"][i]), "G": (I["G"][0], R[f"{rule}_G"][i]),
                "th": (I["th"][0], R[f"{rule}_th"][i]), "thomson_scalars": (ts_new, ts_old[i])}
        diff = [k for k, (x, y) in vals.items() if not bitsame64(x, y)]
        if ok and not diff:
            same += 1
        else:
            fails.append({"row": int(i), "shot": s, "block_start_same": bool(ok), "differs": diff})
    res = {"band": b, "rule": rule, "unit": u["name"], "candidates": int(len(cand)), "rows": int(len(pick)),
           "rows_identical": same, "values_raised": changed, "failures": fails[:20], "passes": bool(same == len(pick))}
    done("visibility", res)


# ------------------------------------------------------------------------------------------ references
def check_references():
    pool, tabs = [], {}
    for b in C.BAND_NAMES:
        R = C.load_band(b, ["shot", "c", "craw", "j", "common_F"])
        Z = C.load_npz(C.CACHE / f"refs_{b}.npz")
        rows = np.flatnonzero(R["common_F"] & Z["formed"])
        pool += [(b, int(i)) for i in rows]
        tabs[b] = (R, Z)
    pick = sorted(C.rng_for("references").choice(len(pool), 200, replace=False).tolist())
    full = {}
    for k in pick:
        b, i = pool[k]
        R, Z = tabs[b]
        full[k] = (b, i, int(R["shot"][i]), int(R["c"][i]), int(R["craw"][i]), int(R["j"][i]), float(Z["P"][i]), float(Z["O"][i]))
    cat = K.meta_tables()
    same, fails = 0, []
    for k in pick:
        b, i, s, c, cr, j, P0, O0 = full[k]
        st, P = K.prepare_shot(s, cat)
        d, a, L0 = P["d"], P["a"], P["p"]["L"]
        if int(P["chords"][c]) != cr:
            raise SystemExit("FATAL: chord index mismatch")
        d2 = copy.deepcopy(d)
        d2["T"][a + j, cr] = d2["T"][a + j, cr] + 1000.0
        good2 = K.rg.good_masks(d2, K.REL)["T_i"]
        ycol = d2["T"][a:a + L0, cr].astype(np.float64)
        gcol = good2[a:a + L0, cr]
        tf = C.STEP * np.arange(L0)
        p, g, formed, _ = RF.reference_of(tf, ycol, gcol, j)
        ok = bitsame64(p, P0) and bitsame64(g, O0) and formed
        same += ok
        if not ok:
            fails.append({"band": b, "row": i, "shot": s, "pchip_same": bool(bitsame64(p, P0)), "gp_same": bool(bitsame64(g, O0))})
    res = {"pool_rows": len(pool), "rows": len(pick), "rows_identical": int(same), "failures": fails[:20],
           "rows_by_band": {b: int(sum(1 for k in pick if pool[k][0] == b)) for b in C.BAND_NAMES},
           "passes": bool(same == len(pick))}
    done("references", res)


# ------------------------------------------------------------------------------------------ rows
def check_rows():
    res = {"bands": {}}
    ok_all = True
    for b in C.BAND_NAMES:
        R = C.load_band(b, ["shot", "y", "L5", "common_F", "common_L5"] + [f"{r}_{k}" for r in C.RULES for k in ("H", "G")])
        Z = np.load(C.CACHE / f"refs_{b}.npz")
        us, tab = C.units(b)
        shots = np.array(tab["shots"])
        n = len(R["shot"])
        out = {}
        m1 = [u for u in us if u["design"] == "M1"]
        tests = np.concatenate([np.array(u["test"]) for u in m1])
        u_ok = bool(len(tests) == len(shots) and np.array_equal(np.sort(tests), shots)
                    and all(not (set(u["fit"]) & set(u["test"])) and len(u["fit"]) + len(u["test"]) == len(shots) for u in m1))
        m3 = [u for u in us if u["design"] == "M3"]
        u3 = bool(all(max(u["fit"]) < min(u["test"]) for u in m3))
        out["units"] = {"M1_every_shot_tested_once_and_disjoint": u_ok, "M3_fit_before_test": u3,
                        "design_shots": int(len(shots))}
        ok = u_ok and u3
        for dsg, uu in (("M1", m1), ("M3", m3)):
            scored_s = np.isin(R["shot"], np.array(sorted(set().union(*[set(u["test"]) for u in uu]))))
            for X in C.TARGETS:
                rules = C.RULES if X == "L5" else (("D5",) if dsg == "M1" else ())
                for r in rules:
                    rows = np.flatnonzero(scored_s & R[f"common_{X}"])
                    est = {"H": R[f"{r}_H"], "G": R[f"{r}_G"]}
                    if X == "L5":
                        est["O"] = R["y"]
                    for L in ("T0", "TX"):
                        a = np.full(n, np.nan)
                        cover = np.zeros(n, np.int64)
                        for u in uu:
                            z = np.load(C.fit_path(b, u["name"], L, r, X))
                            a[z["idx"]] = z["est"]
                            cover[z["idx"]] += 1
                        if not (np.all(cover[rows] == 1) and cover.sum() == len(rows)):
                            ok = False
                        est[L] = a
                    cell = {"rows": int(len(rows)), "shots": int(len(np.unique(R["shot"][rows])))}
                    fin = {e: bool(np.all(np.isfinite(v[rows]))) for e, v in est.items()}
                    tgt = (R["y"] if X == "F" else R["L5"])[rows]
                    fin["target"] = bool(np.all(np.isfinite(tgt)))
                    if X == "F":
                        vr = rows[Z["formed"][rows]]
                        cell["value_of_frame_rows"] = int(len(vr))
                        cell["rows_left_out_no_reference"] = int(len(rows) - len(vr))
                        for e in ("P", "O"):
                            fin["ref_" + e] = bool(np.all(np.isfinite(Z[e][vr])))
                        for e in ("H", "G", "T0", "TX"):
                            fin[e + "_on_value_of_frame_rows"] = bool(np.all(np.isfinite(est[e][vr])))
                    cell["all_finite"] = fin
                    ok &= all(fin.values()) and len(rows) > 0
                    out[f"{dsg}|{X}|{r}"] = cell
                    C.log(f"rows {b} {dsg} {X} {r}: {cell['rows']} rows / {cell['shots']} shots"
                          + (f", value-of-frame rows {cell['value_of_frame_rows']}" if X == "F" else ""))
        out["passes"] = bool(ok)
        res["bands"][b] = out
        ok_all &= ok
    res["passes"] = bool(ok_all)
    done("rows", res)


# ------------------------------------------------------------------------------------------ markdown
def write():
    Cc = {n: json.loads((C.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("band_e", "visibility", "references", "rows")}
    L = ["# B.86 checks", "",
         f"Registration: `{C.REG}` section 6; implementation: `{C.ADD}` section 9. Each check was run once, before any "
         "readout; no error, skill or interval was computed by a check. No tolerance was changed after a run.", ""]
    e = Cc["band_e"]
    L += ["## 1. Band E against B.76", "",
          f"Rows: this build {e['rows_this_build']:,}, B.76 `data/.b76/rows.npz` {e['rows_b76']:,} (same order required).", "",
          "| field | identical bit for bit | dtype |", "|---|---|---|"]
    for k, v in e["fields"].items():
        L.append(f"| {k} | {v['identical']} | {v['dtype_this']} |")
    L += ["", "T0 and TX of unit M1_s1_k0 under D100, target L5, refitted by this batch's fit code on this batch's band-E rows "
              "with B.76's L5 common set and unit, against B.76's stored estimates:", "",
          "| learner | fit rows (B.76) | test rows | row indices identical | estimates identical | iterations (B.76) |",
          "|---|---|---|---|---|---|"]
    for k, v in e["refit"].items():
        L.append(f"| {k} | {v['fit_rows']:,} ({v['fit_rows_b76']:,}) | {v['test_rows']:,} | {v['idx_identical']} | "
                 f"{v['estimates_identical']} | {v['n_iter']} ({v['n_iter_b76']}) |")
    L += ["", f"Check passes: **{e['passes']}**", ""]
    v = Cc["visibility"]
    L += ["## 2. Visibility (band B4, D100)", "",
          f"Unit {v['unit']} of band {v['band']}, {v['rows']} rows drawn from {v['candidates']:,} test rows of the L5 common set. "
          "For each, a copy of its shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms "
          "(all chords) is raised by 1000 eV; good mask and block start recomputed; anchor frame, number of visible values, H, "
          "age, M5, G, the five scalars of the last ten values and the nine Thomson scalars recomputed and compared bit for "
          f"bit. Rows identical: {v['rows_identical']} of {v['rows']}; T_i values raised: {v['values_raised']:,}."]
    if v["failures"]:
        L += ["", "Failures (first 20): " + json.dumps(v["failures"])]
    L += ["", f"Check passes: **{v['passes']}**", ""]
    r = Cc["references"]
    L += ["## 3. References that wait", "",
          f"{r['rows']} rows drawn from the {r['pool_rows']:,} value-of-the-frame rows of all bands (by band: "
          + ", ".join(f"{b} {n}" for b, n in r["rows_by_band"].items()) + "). For each, the T_i value of the row's chord at "
          "frame t raised by 1000 eV in a copy of the shot, good mask recomputed, PCHIP and offline GP recomputed with frame t "
          f"left out: identical bit for bit on {r['rows_identical']} of {r['rows']}."]
    if r["failures"]:
        L += ["", "Failures (first 20): " + json.dumps(r["failures"])]
    L += ["", f"Check passes: **{r['passes']}**", ""]
    w = Cc["rows"]
    L += ["## 4. Rows", "",
          "For every band, design, rule and target: scored rows and shots; every estimator (H, G, T0, TX, O; for F also PCHIP "
          "and the offline GP on the value-of-the-frame rows) finite on every scored row, every scored row estimated exactly "
          "once by T0 and TX; M1 tests every design shot once with fit and test disjoint; M3 fits only shots before its test shots.", "",
          "| band | design shots | M1 units | M3 units | cell | rows | shots | value-of-frame rows (left out) | all finite |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b, o in w["bands"].items():
        uo = o["units"]
        for k, c in o.items():
            if k in ("units", "passes"):
                continue
            vf = f"{c['value_of_frame_rows']:,} ({c['rows_left_out_no_reference']:,})" if "value_of_frame_rows" in c else "-"
            L.append(f"| {b} | {uo['design_shots']:,} | {uo['M1_every_shot_tested_once_and_disjoint']} | {uo['M3_fit_before_test']} | "
                     f"{k} | {c['rows']:,} | {c['shots']:,} | {vf} | {all(c['all_finite'].values())} |")
    L += ["", f"Check passes: **{w['passes']}**", ""]
    C.save_text(C.CHECKS_MD, "\n".join(L))
    C.log("written b86_checks.md")


if __name__ == "__main__":
    {"band_e": check_band_e, "visibility": check_visibility, "references": check_references, "rows": check_rows,
     "write": write}[sys.argv[1]]()
