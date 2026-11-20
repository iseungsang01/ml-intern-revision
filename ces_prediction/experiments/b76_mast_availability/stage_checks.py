"""B.76 checks (registration section 12, Addendum 1 section 9).  Run before any readout.

    py ces_prediction/experiments/b76_mast_availability/stage_checks.py hold        # check 1
    py ces_prediction/experiments/b76_mast_availability/stage_checks.py visibility  # check 2
    py ces_prediction/experiments/b76_mast_availability/stage_checks.py units       # check 3
    py ces_prediction/experiments/b76_mast_availability/stage_checks.py thomson     # check 4
    py ces_prediction/experiments/b76_mast_availability/stage_checks.py write       # docs/paper/b76_checks.md
Output: data/.b76/checks/{name}.json, docs/paper/b76_checks.md.  No estimate or score is computed here.
"""
from __future__ import annotations

import json
import sys
import warnings

import common76 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
OLD = (K.DATA / "mast" / ".ped_rows_T_i.npz", K.DATA / "mast" / ".ped_conf_rows_T_i.npz")


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def keys_of(shot, frame, R32):
    return ((np.asarray(shot, np.int64) << 47) | (np.asarray(frame, np.int64) << 32)
            | np.asarray(R32, np.float32).view(np.uint32).astype(np.int64))


# ------------------------------------------------------------------------------------------ check 1
def check_hold():
    R = K.load_rows(["shot", "j", "R", "rho", "t", "te_t", "ne_t", "dband", "prevgood", "D5_H", "D5_anc"])
    mine = keys_of(R["shot"], R["j"], R["R"].astype(np.float32))
    uk, ucnt = np.unique(mine, return_counts=True)
    dup = np.isin(mine, uk[ucnt > 1])          # two chords of one shot at the same radius: not matched
    mine = np.where(dup, -1, mine)
    out = {"rows_this_build": int(len(mine)), "rows_with_a_shared_key_not_matched": int(dup.sum()), "tables": {}}
    allm = {"H": [], "y1": [], "te": [], "te0": [], "ne": [], "ne0": [], "db": [], "de": [], "rho": [], "rho0": [], "t": [], "t0": []}
    for p in OLD:
        z = np.load(p, allow_pickle=False)
        o = {k: z[k] for k in ("shot", "frame", "R", "y1", "te0", "ne0", "d_edge", "rho", "t0")}
        ko = keys_of(o["shot"], o["frame"], o["R"])
        uk, ucnt = np.unique(ko, return_counts=True)
        ko = np.where(np.isin(ko, uk[ucnt > 1]), -2, ko)
        srt = np.argsort(ko)
        pos = np.searchsorted(ko[srt], mine)
        pos = np.minimum(pos, len(ko) - 1)
        hit = ko[srt][pos] == mine
        jo = srt[pos[hit]]
        mi = np.flatnonzero(hit)
        pg = R["prevgood"][mi]
        out["tables"][p.name] = {"old_rows": int(len(ko)), "rows_in_both": int(hit.sum()),
                                 "rows_in_both_prev_frame_good": int(pg.sum()),
                                 "anchor_is_previous_frame": bool(np.all(R["D5_anc"][mi][pg] == R["j"][mi][pg] - 1))}
        mi, jo = mi[pg], jo[pg]
        allm["H"].append(R["D5_H"][mi]); allm["y1"].append(o["y1"][jo])
        allm["te"].append(R["te_t"][mi]); allm["te0"].append(o["te0"][jo])
        allm["ne"].append(R["ne_t"][mi]); allm["ne0"].append(o["ne0"][jo])
        allm["db"].append(R["dband"][mi]); allm["de"].append(o["d_edge"][jo])
        allm["rho"].append(R["rho"][mi]); allm["rho0"].append(o["rho"][jo])
        allm["t"].append(R["t"][mi]); allm["t0"].append(o["t0"][jo])
        del z, o
    A = {k: np.concatenate(v) for k, v in allm.items()}
    H32 = A["H"].astype(np.float32)
    out["compared_rows"] = int(len(H32))
    out["H_float32_equal_rows"] = int(np.sum(H32 == A["y1"]))
    out["H_max_abs_dev_eV_float64_vs_stored_float32"] = float(np.max(np.abs(A["H"] - A["y1"].astype(np.float64)))) if len(H32) else None
    out["H_max_abs_dev_eV_float32"] = float(np.max(np.abs(H32 - A["y1"]))) if len(H32) else None
    diag = {}
    for a, b, nm in (("te", "te0", "T_e(t) vs te0"), ("ne", "ne0", "n_e(t) vs ne0"), ("db", "de", "band distance vs d_edge"),
                     ("rho", "rho0", "normalised radius vs rho"), ("t", "t0", "frame time vs t0")):
        x32 = A[a].astype(np.float32)
        y = A[b]
        fin = np.isfinite(x32) & np.isfinite(y)
        diag[nm] = {"both_finite": int(fin.sum()), "finite_pattern_same": int(np.sum(np.isfinite(x32) == np.isfinite(y))),
                    "float32_equal": int(np.sum(x32[fin] == y[fin])),
                    "max_abs_dev": float(np.max(np.abs(x32[fin].astype(np.float64) - y[fin].astype(np.float64)))) if fin.any() else None}
    out["diagnostics"] = diag
    out["passes"] = bool(out["compared_rows"] > 0 and out["H_float32_equal_rows"] == out["compared_rows"])
    K.save_json(K.CHK / "hold.json", out)
    K.log(f"check 1: compared {out['compared_rows']}, equal {out['H_float32_equal_rows']}, passes {out['passes']}")


def diag_thomson_vs_old():
    """Diagnostic beside check 1 (added after check 1 ran, before any fit): the cause of the rows where T_e(t) differs
    from te0 of the earlier tables.  The earlier build stored the frame time in float32 (t0) and formed the windows from it."""
    R = K.load_rows(["shot", "j", "R", "t", "te_t", "dband", "prevgood"])
    mine = keys_of(R["shot"], R["j"], R["R"].astype(np.float32))
    tot = flip = btot = bflip = 0
    for p in OLD:
        z = np.load(p, allow_pickle=False)
        o = {k: z[k] for k in ("shot", "frame", "R", "te0", "t0", "d_edge")}
        ko = keys_of(o["shot"], o["frame"], o["R"])
        srt = np.argsort(ko)
        pos = np.minimum(np.searchsorted(ko[srt], mine), len(ko) - 1)
        hit = (ko[srt][pos] == mine) & R["prevgood"]
        mi, jo = np.flatnonzero(hit), srt[pos[hit]]
        dif = R["te_t"][mi].astype(np.float32) != o["te0"][jo]
        bdif = R["dband"][mi].astype(np.float32) != o["d_edge"][jo]
        for i, jj, a_, b_ in zip(mi[dif | bdif], jo[dif | bdif], dif[dif | bdif], bdif[dif | bdif]):
            t = K.ep.load_extra(int(R["shot"][i]))["ts"][0]
            tf, t32 = R["t"][i], float(o["t0"][jj])
            if a_:
                tot += 1
                flip += bool(np.any(((t >= tf - 2.5) & (t < tf + 2.5)) != ((t >= t32 - 2.5) & (t < t32 + 2.5))))
            if b_:
                btot += 1
                bflip += bool(np.any(((t >= tf - 7.5) & (t <= tf + 7.5)) != ((t >= t32 - 7.5) & (t <= t32 + 7.5))))
    out = {"rows_te_differs": tot, "rows_where_a_laser_time_changes_side_of_the_window_edge": flip,
           "rows_band_distance_differs": btot, "rows_where_a_laser_time_changes_side_of_the_7.5ms_window": bflip}
    K.save_json(K.CHK / "hold_diag.json", out)
    K.log(f"diag: {out}")


# ------------------------------------------------------------------------------------------ check 2
def check_visibility():
    import copy
    rule, dl = "D100", K.DELAYS["D100"]
    keys = ["shot", "craw", "c", "j", "t", "R"] + [f"{rule}_{k}" for k in ("H", "age", "M5", "G", "th", "anc", "nvis", "te_a", "ne_a",
                                                                           "da_a", "dmx_a")]
    R = K.load_rows(keys)
    S = K.load_npz(K.CACHE / "sets.npz")
    us, _ = K.units()
    u = next(x for x in us if x["name"] == "M1_s1_k0")
    cand = np.flatnonzero(S["common_L5"] & np.isin(R["shot"], np.array(u["test"])))
    pick = np.sort(K.rng_for("visibility|D100").choice(cand, 200, replace=False))
    cat = K.meta_tables()
    res = {"rule": rule, "unit": u["name"], "candidates": int(len(cand)), "rows": int(len(pick)), "modes": {}}
    fails = []
    changed = {"removed": 0, "raised": 0}
    same = {"removed": 0, "raised": 0}
    cache = {}
    for i in pick:
        s = int(R["shot"][i])
        if s not in cache:
            cache.clear()
            st, P = K.prepare_shot(s, cat)
            cache[s] = P
        P = cache[s]
        d, a = P["d"], P["a"]
        t_a = float(d["time_ms"][a])
        if abs(t_a + K.STEP * R["j"][i] - R["t"][i]) > 1e-9:
            raise SystemExit("FATAL: frame time mismatch")
        lim = R["t"][i] - dl
        nfr = d["T"].shape[0]
        tn = t_a + K.STEP * (np.arange(nfr) - a)
        later = tn > lim + K.TOL_MS
        cr = int(R["craw"][i])
        for mode in ("removed", "raised"):
            d2 = copy.deepcopy(d)
            fin = np.isfinite(d2["T"][later])
            changed[mode] += int(fin.sum())
            if mode == "removed":
                d2["T"][later] = np.nan
            else:
                d2["T"][later] = d2["T"][later] + 1000.0
            good2 = K.rg.good_masks(d2, K.REL)["T_i"]
            fr10 = np.nonzero(good2.sum(1) >= 10)[0]          # block start: first frame with at least ten good chords
            ok = len(fr10) > 0 and int(fr10[0]) == a
            y2 = d2["T"][a:, cr].astype(np.float64)
            g2 = good2[a:, cr]
            tf2 = K.STEP * np.arange(len(y2))
            I = K.chord_history(tf2, y2, g2, np.array([R["j"][i]]), dl)
            anc = int(I["anc"][0])
            # anchor-frame Thomson and D-alpha, recomputed from the recomputed anchor frame
            L0 = P["p"]["L"]
            Rc = d["R"][P["chords"]].astype(np.float64)
            TEw, NEw, *_ = K.thomson_grids(P["ex"], Rc, t_a + K.STEP * np.arange(L0), P["geo"][0])
            da, dmx = K.dalpha_grid(s, t_a, L0)
            c = int(R["c"][i])
            vals = {"anc": (anc, R[f"{rule}_anc"][i]), "nvis": (I["nvis"][0], R[f"{rule}_nvis"][i]),
                    "H": (I["H"][0], R[f"{rule}_H"][i]), "age": (I["age"][0], R[f"{rule}_age"][i]),
                    "M5": (I["M5"][0], R[f"{rule}_M5"][i]), "G": (I["G"][0], R[f"{rule}_G"][i]),
                    "th": (I["th"][0], R[f"{rule}_th"][i]),
                    "te_a": (TEw[anc, c] if anc >= 0 else np.nan, R[f"{rule}_te_a"][i]),
                    "ne_a": (NEw[anc, c] if anc >= 0 else np.nan, R[f"{rule}_ne_a"][i]),
                    "da_a": (da[anc] if anc >= 0 else np.nan, R[f"{rule}_da_a"][i]),
                    "dmx_a": (dmx[anc] if anc >= 0 else np.nan, R[f"{rule}_dmx_a"][i])}
            diff = [k for k, (x, y) in vals.items() if not bitsame(x, y)]
            if ok and not diff:
                same[mode] += 1
            else:
                fails.append({"row": int(i), "shot": s, "mode": mode, "block_start_same": bool(ok), "differs": diff})
    for mode in ("removed", "raised"):
        res["modes"][mode] = {"rows_identical": same[mode], "values_changed": changed[mode]}
    res["failures"] = fails[:20]
    res["passes"] = bool(same["removed"] == len(pick) and same["raised"] == len(pick))
    K.save_json(K.CHK / "visibility.json", res)
    K.log(f"check 2: identical removed {same['removed']}, raised {same['raised']} of {len(pick)}; passes {res['passes']}")


# ------------------------------------------------------------------------------------------ check 3
def check_units():
    us, tab = K.units()
    shots = np.array(tab["shots"])
    sess = dict(zip(tab["shots"], tab["session"]))
    res = {"designs": {}}
    ok_all = True
    for dsg in ("M1", "M2"):
        for seed in K.SEEDS:
            uu = [u for u in us if u["design"] == dsg and u["sub"] == seed]
            tests = np.concatenate([np.array(u["test"]) for u in uu])
            once = bool(len(tests) == len(shots) and np.array_equal(np.sort(tests), shots))
            disjoint = all(len(set(u["fit"]) & set(u["test"])) == 0 and len(u["fit"]) + len(u["test"]) == len(shots) for u in uu)
            whole = True
            if dsg == "M2":
                fold_of_sess = {}
                for k, u in enumerate(uu):
                    for s in u["test"]:
                        fold_of_sess.setdefault(sess[s], set()).add(k)
                whole = all(len(v) == 1 for v in fold_of_sess.values())
            res["designs"][f"{dsg} seed {seed}"] = {"every_shot_tested_once": once, "fit_test_disjoint": disjoint,
                                                    "sessions_whole": whole, "test_shots": [len(u["test"]) for u in uu]}
            ok_all &= once and disjoint and whole
    for u in us:
        if u["design"] in ("M3", "M4"):
            ok = max(u["fit"]) < min(u["test"])
            res["designs"][u["name"]] = {"fit_shots": len(u["fit"]), "test_shots": len(u["test"]),
                                         "last_fit_shot": int(max(u["fit"])), "first_test_shot": int(min(u["test"])),
                                         "fit_before_test": bool(ok)}
            ok_all &= ok
    m4 = next(u for u in us if u["name"] == "M4")
    q4 = next(u for u in us if u["name"] == "M3_q4")
    res["M4_equals_M3_q4"] = bool(m4["fit"] == q4["fit"] and m4["test"] == q4["test"])
    res["passes"] = bool(ok_all)
    K.save_json(K.CHK / "units.json", res)
    K.log(f"check 3: passes {res['passes']}")


# ------------------------------------------------------------------------------------------ check 4
def check_thomson():
    R = K.load_rows(["shot", "R", "t", "te_t", "ne_t", "nts", "nlas"])
    S = K.load_npz(K.CACHE / "sets.npz")
    cand = np.flatnonzero(S["common_F"])
    pick = np.sort(K.rng_for("thomson|window").choice(cand, 2000, replace=False))
    same_te = same_ne = 0
    maxdev = 0.0
    fails = []
    ex_cache = {}
    for i in pick:
        s = int(R["shot"][i])
        if s not in ex_cache:
            ex_cache.clear()
            ex_cache[s] = K.ep.load_extra(s)
        t, Rv, te, ne, g = ex_cache[s]["ts"]
        tf = R["t"][i]
        inside = np.flatnonzero((t >= tf - 2.5) & (t < tf + 2.5))
        Rc = np.array([R["R"][i]])
        vte = np.array([K.ep.to_chords(Rc, Rv[j][np.isfinite(te[j])], te[j][np.isfinite(te[j])])[0] for j in inside])
        vne = np.array([K.ep.to_chords(Rc, Rv[j][np.isfinite(ne[j])], ne[j][np.isfinite(ne[j])])[0] for j in inside])
        x_te = np.nanmean(vte) if np.isfinite(vte).any() else np.nan
        x_ne = (np.nanmean(vne) if np.isfinite(vne).any() else np.nan) / 1e19
        a_te, a_ne = bitsame(x_te, R["te_t"][i]), bitsame(x_ne, R["ne_t"][i])
        same_te += a_te
        same_ne += a_ne
        if np.isfinite(x_te) and np.isfinite(R["te_t"][i]):
            maxdev = max(maxdev, abs(x_te - R["te_t"][i]))
        if not (a_te and a_ne) and len(fails) < 20:
            fails.append({"row": int(i), "shot": s, "laser_times_inside": int(len(inside)), "te_same": bool(a_te), "ne_same": bool(a_ne)})
    res = {"rows": int(len(pick)), "candidates": int(len(cand)), "te_identical": int(same_te), "ne_identical": int(same_ne),
           "te_max_abs_dev_eV": float(maxdev), "failures": fails, "samples": {}}
    for X in K.TARGETS:
        m = S[f"common_{X}"]
        nts = R["nts"][m]
        res["samples"][X] = {"rows": int(m.sum()), "one": float(np.mean(nts == 1)), "two": float(np.mean(nts == 2)),
                             "three_or_more": float(np.mean(nts >= 3)), "zero": float(np.mean(nts == 0)),
                             "laser_times_in_window": {str(k): float(np.mean(R["nlas"][m] == k)) for k in range(0, 5)}}
    res["passes"] = bool(same_te == len(pick) and same_ne == len(pick))
    K.save_json(K.CHK / "thomson.json", res)
    K.log(f"check 4: identical te {same_te}, ne {same_ne} of {len(pick)}; passes {res['passes']}")


# ------------------------------------------------------------------------------------------ markdown
def write():
    C = {n: json.loads((K.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("hold", "visibility", "units", "thomson")}
    L = ["# B.76 checks", "",
         f"Registration: `{K.REG}` section 12; implementation: `{K.ADD}` section 9. Run before any readout. "
         "No tolerance was changed after a run.", ""]
    h = C["hold"]
    L += ["## 1. H under D5 against the earlier MAST row tables", "",
          "Earlier tables `data/mast/.ped_rows_T_i.npz` (odd-numbered shots) and `.ped_conf_rows_T_i.npz` (even-numbered), "
          "matched by shot, block frame and chord radius (float32). Only rows whose frame j - 1 is good are compared (the "
          "earlier build hid every odd frame; here no frame is hidden).", "",
          "| table | rows in the table | rows in both builds | of them with frame j - 1 good | anchor = frame j - 1 on all of them |",
          "|---|---|---|---|---|"]
    for k, v in h["tables"].items():
        L.append(f"| {k} | {v['old_rows']:,} | {v['rows_in_both']:,} | {v['rows_in_both_prev_frame_good']:,} | {v['anchor_is_previous_frame']} |")
    L += ["", f"Compared rows {h['compared_rows']:,}; float32(H) equal to `y1` on {h['H_float32_equal_rows']:,}. Largest "
              f"deviation: {h['H_max_abs_dev_eV_float32']:.3g} eV in float32; {h['H_max_abs_dev_eV_float64_vs_stored_float32']:.3g} eV "
              "between this build's float64 value and the stored float32 value (rounding of the stored table).", "",
          "Diagnostics (no pass rule):", "", "| quantity | both finite | finite pattern same (rows) | float32 equal | largest deviation |",
          "|---|---|---|---|---|"]
    for k, v in h["diagnostics"].items():
        md = "-" if v["max_abs_dev"] is None else f"{v['max_abs_dev']:.3g}"
        L.append(f"| {k} | {v['both_finite']:,} | {v['finite_pattern_same']:,} | {v['float32_equal']:,} | {md} |")
    dg = K.CHK / "hold_diag.json"
    if dg.exists():
        x = json.loads(dg.read_text(encoding="utf-8"))
        L += ["", f"Cause of the T_e differences (diagnostic added after check 1 ran, before any fit): of the "
                  f"{x['rows_te_differs']} rows where T_e(t) differs from `te0`, on "
                  f"{x['rows_where_a_laser_time_changes_side_of_the_window_edge']} a laser time lies on the other side of a "
                  "window edge when the window is formed from the earlier build's float32 frame time `t0` instead of the "
                  f"float64 frame time used here. The band distance differs on {x['rows_band_distance_differs']} rows; on "
                  f"{x['rows_where_a_laser_time_changes_side_of_the_7.5ms_window']} of them a laser time changes side of the "
                  "±7.5 ms edge window in the same way."]
    L += ["", f"Check 1 passes: **{h['passes']}**", ""]
    v = C["visibility"]
    L += ["## 2. Visibility", "",
          f"Unit {v['unit']}, rule {v['rule']}, {v['rows']} rows drawn from {v['candidates']:,} test rows of the L5 common set. "
          "For each row a copy of its shot in which every T_i value at a frame with nominal time above the row's limit + 0.5 ms "
          "(all chords) is removed or raised by 1000 eV; good mask and block start recomputed; anchor frame, number of visible "
          "values, H, age, M5, G, the five scalars of the last ten values, T_e and n_e at the anchor and the D-alpha anchor "
          "scalars recomputed and compared bit for bit with the main pass.", "",
          "| modification | rows identical | T_i values changed |", "|---|---|---|"]
    for mode, x in v["modes"].items():
        L.append(f"| {mode} | {x['rows_identical']} of {v['rows']} | {x['values_changed']:,} |")
    if v["failures"]:
        L += ["", "Failures (first 20): " + json.dumps(v["failures"])]
    L += ["", f"Check 2 passes: **{v['passes']}**", ""]
    u = C["units"]
    L += ["## 3. Design units", "", "| design | every shot tested once | fit and test disjoint | sessions whole | test shots per fold |",
          "|---|---|---|---|---|"]
    for k, x in u["designs"].items():
        if "every_shot_tested_once" in x:
            L.append(f"| {k} | {x['every_shot_tested_once']} | {x['fit_test_disjoint']} | {x['sessions_whole']} | {x['test_shots']} |")
    L += ["", "| unit | fit shots | test shots | last fit shot | first test shot | fit before test |", "|---|---|---|---|---|---|"]
    for k, x in u["designs"].items():
        if "fit_before_test" in x:
            L.append(f"| {k} | {x['fit_shots']} | {x['test_shots']} | {x['last_fit_shot']} | {x['first_test_shot']} | {x['fit_before_test']} |")
    L += ["", f"M4 has the fit and test shots of M3_q4: {u['M4_equals_M3_q4']}.", "", f"Check 3 passes: **{u['passes']}**", ""]
    t = C["thomson"]
    L += ["## 4. Thomson value at frame t from the window of frame t only", "",
          f"{t['rows']:,} rows drawn from the F common set ({t['candidates']:,}); T_e(t) and n_e(t) recomputed from the extra "
          "file with only the laser times in [t - 2.5, t + 2.5) ms: identical bit for bit on "
          f"{t['te_identical']:,} (T_e) and {t['ne_identical']:,} (n_e) rows; largest T_e deviation {t['te_max_abs_dev_eV']:.3g} eV.", "",
          "Laser samples finite at the chord in the window of frame t, share of rows:", "",
          "| set | rows | one | two | three or more | laser times in the window: 0 / 1 / 2 / 3 / 4 |", "|---|---|---|---|---|---|"]
    for X, s in t["samples"].items():
        lw = " / ".join(f"{s['laser_times_in_window'][str(k)]:.3f}" for k in range(5))
        L.append(f"| common {X} | {s['rows']:,} | {s['one']:.3f} | {s['two']:.3f} | {s['three_or_more']:.3f} | {lw} |")
    if t["failures"]:
        L += ["", "Failures (first 20): " + json.dumps(t["failures"])]
    L += ["", f"Check 4 passes: **{t['passes']}**", ""]
    K.save_text(K.CHECKS_MD, "\n".join(L))
    K.log("written b76_checks.md")


if __name__ == "__main__":
    {"hold": check_hold, "visibility": check_visibility, "units": check_units, "thomson": check_thomson,
     "diag": diag_thomson_vs_old, "write": write}[sys.argv[1]]()
