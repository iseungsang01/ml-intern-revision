"""B.83 checks (registration section 5, Addendum 1 section 7), each run once, before any readout.

    py ces_prediction/experiments/b83_interp/checks83.py references|no_look|rows|stored
    py ces_prediction/experiments/b83_interp/checks83.py render          # docs/paper/b83_checks.md
Output: data/.b83/checks/<name>.json.  A check that exists is not run again.
"""
from __future__ import annotations

import json
import sys

import common83 as M

import numpy as np  # noqa: E402

C = M.C
sys.path.insert(0, str(C.REPO / "ces_prediction"))
B70_RMSE = {"hold": "232.6", "causal GP": "192.8", "PCHIP": "206.0", "offline GP": "180.8"}
B73_RMSE = {"T0[F]": "182.9", "TX[F]": "174.8", "H": "232.6", "G": "192.7"}


def _once(name):
    p = M.CHK / f"{name}.json"
    if p.exists():
        raise SystemExit(f"check {name} was already run (each check runs once): {p}")
    return p


def references():
    p = _once("references")
    R = C.rows()
    refs = M.load_npz(M.REFS)
    key = R["fidx"] * 1_000_000 + R["grow"]
    o = np.argsort(key)
    per, mism, rows_cmp = {}, {"pchip": 0, "gp": 0}, 0
    for r in C.REPEATS:
        se = {"hold": [], "causal GP": [], "PCHIP": [], "offline GP": []}
        for k in range(C.K):
            s = np.load(C.DATA / ".b70" / f"stage2_r{r}_f{k}.npz", allow_pickle=True)
            c = np.load(C.DATA / ".b70_rescore" / f"b39p_v2_cut_r{r}_f{k}" / "comparison_errors_test.npz")
            if not np.array_equal(s["row"], c["CES_TI_row"]):
                raise SystemExit(f"FATAL: r{r} f{k}: stage2 rows are not the comparison rows")
            kk = s["fidx"].astype(np.int64) * 1_000_000 + s["grow"].astype(np.int64)
            pos = np.searchsorted(key[o], kk)
            j = o[pos]
            if not np.array_equal(key[j], kk):
                raise SystemExit(f"FATAL: r{r} f{k}: a comparison row is not a D1 row")
            y = c["CES_TI_y_true"].astype(np.float64)
            sp = (refs["P"][j] - y) ** 2
            sg = (refs["O"][j] - y) ** 2
            mism["pchip"] += int(np.sum(sp != c["CES_TI_se_pchip"]))
            mism["gp"] += int(np.sum(sg != c["CES_TI_se_gp"]))
            rows_cmp += len(y)
            se["hold"].append(c["CES_TI_se_persistence"])
            se["causal GP"].append(c["CES_TI_se_gp_causal"])
            se["PCHIP"].append(sp)
            se["offline GP"].append(sg)
        per[f"r{r}"] = {e: float(np.sqrt(np.mean(np.concatenate(v)))) for e, v in se.items()}
        per[f"r{r}"]["rows"] = int(sum(len(x) for x in se["hold"]))
    pooled = {e: float(np.mean([per[f"r{r}"][e] for r in C.REPEATS])) for e in B70_RMSE}
    printed = {e: f"{v:.1f}" for e, v in pooled.items()}
    rep = {e: printed[e] == B70_RMSE[e] for e in B70_RMSE}
    ok = mism["pchip"] == 0 and mism["gp"] == 0 and all(rep.values())
    out = {"passes": bool(ok), "rows_compared_over_15_files": rows_cmp, "values_not_bit_identical": mism,
           "per_repartition_rmse_eV": per, "pooled_rmse_eV": pooled, "printed": printed, "b70_d1_md": B70_RMSE,
           "reproduced": rep,
           "note": "values compared through the stored squared errors: (v - float64(y_true))**2 == se, numpy exact equality"}
    M.save_json(p, out)
    M.log(f"references: passes {ok}; not bit-identical pchip {mism['pchip']}, gp {mism['gp']} of {rows_cmp:,}; printed {printed}")
    return ok


def no_look():
    p = _once("no_look")
    R = C.rows()
    refs = M.load_npz(M.REFS)
    names, arrs, _ = M.load_arrays()
    rng = M.rng_for("no-look")
    pick = np.sort(rng.choice(len(R["y"]), 200, replace=False))
    bad = {"removed": 0, "raised": 0}
    detail = []
    for i in pick:
        fa0 = arrs[names[int(R["fidx"][i])]]
        row = int(refs["drow"][i])
        if not np.isfinite(fa0[row, 1]):
            raise SystemExit("FATAL: the value of a D1 row is missing in its dataset array")
        for case in ("removed", "raised"):
            fa = fa0.copy()
            if case == "removed":
                fa[row, 1] = np.nan
            else:
                fa[row, 1] = fa[row, 1] + np.float32(1000.0)
            pv, gv, *_ = M.reference_of(fa, row)
            same = (pv == refs["P"][i]) and (gv == refs["O"][i])
            if not same:
                bad[case] += 1
                detail.append({"row": int(i), "case": case})
    ok = bad["removed"] == 0 and bad["raised"] == 0
    M.save_json(p, {"passes": bool(ok), "rows": 200, "changed": bad, "detail": detail[:20],
                    "stream": "[20261183, crc32('no-look')]"})
    M.log(f"no_look: passes {ok}; changed {bad}")
    return ok


def rows():
    p = _once("rows")
    R = C.rows()
    refs = M.load_npz(M.REFS)
    us, _, _ = M.units()
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    out, ok = {}, True
    for d in M.DESIGNS:
        subs, scored = M.assemble(d, R, us)
        ref = scored & refs["formed"]
        rr = np.flatnonzero(ref)
        fin = {}
        for sub in subs:
            est = M.estimates(sub, R, refs)
            fin[sub["name"]] = {e: bool(np.all(np.isfinite(v[rr]))) for e, v in est.items()}
            fin[sub["name"]]["all_scored_rows_T0_TX_H_G"] = bool(all(np.all(np.isfinite(est[e][scored])) for e in ("H", "G", "T0", "TX")))
        good = all(all(v.values()) for v in fin.values())
        ok &= good
        sc = np.flatnonzero(scored)
        out[d] = {"subs": [s["name"] for s in subs], "scored_rows": int(scored.sum()), "reference_rows": int(ref.sum()),
                  "rows_left_out_missing_reference": int(scored.sum() - ref.sum()),
                  "discharges": int(len(np.unique(unit_d[rr]))), "clusters": int(len(np.unique(unit_c[rr]))),
                  "discharges_scored": int(len(np.unique(unit_d[sc]))), "clusters_scored": int(len(np.unique(unit_c[sc]))),
                  "every_estimator_finite_on_the_same_rows": fin, "passes": bool(good),
                  "masks_equal_in_every_sub": True}
        M.log(f"rows {d}: scored {out[d]['scored_rows']:,}, reference rows {out[d]['reference_rows']:,}, left out "
              f"{out[d]['rows_left_out_missing_reference']:,}, discharges {out[d]['discharges']}, clusters {out[d]['clusters']}; "
              f"finite {good}")
    info = json.loads((M.CACHE / "refs_info.json").read_text(encoding="utf-8"))
    M.save_json(p, {"passes": bool(ok), "designs": out, "refs_info": info})
    return ok


def stored():
    p = _once("stored")
    R = C.rows()
    us, _, _ = M.units()
    subs, scored = M.assemble("V1", R, us)
    if not scored.all():
        raise SystemExit("FATAL: V1 does not score every D1 row")
    y = R["tg_F"]
    per = {}
    for sub in subs:
        est = {"T0[F]": sub["T0"], "TX[F]": sub["TX"], "H": R["hold"], "G": R["gp"]}
        per[f"r{sub['name']}"] = {e: float(np.sqrt(np.mean((v - y) ** 2))) for e, v in est.items()}
    pooled = {e: float(np.mean([per[k][e] for k in per])) for e in B73_RMSE}
    printed = {e: f"{v:.1f}" for e, v in pooled.items()}
    rep = {e: printed[e] == B73_RMSE[e] for e in B73_RMSE}
    ok = all(rep.values())
    M.save_json(p, {"passes": bool(ok), "rows": int(len(y)), "per_repartition_rmse_eV": per, "pooled_rmse_eV": pooled,
                    "printed": printed, "b73_targets_md": B73_RMSE, "reproduced": rep})
    M.log(f"stored: passes {ok}; printed {printed}")
    return ok


def render():
    J = {nm: json.loads((M.CHK / f"{nm}.json").read_text(encoding="utf-8")) for nm in M.CHECK_NAMES
         if (M.CHK / f"{nm}.json").exists()}
    L = ["# B.83 checks", "",
         f"Registration `{M.REG}`, section 5; implementation `{M.ADD}`, section 7. Each check was run once, before any "
         "readout of this batch.", ""]
    if "references" in J:
        x = J["references"]
        L += ["## References (PCHIP and the offline GP against B.70)", "",
              f"- Rows compared over the 15 comparison files (r2-r4 x 5 folds): {x['rows_compared_over_15_files']:,}.",
              f"- Values not bit-identical (through the stored squared errors): PCHIP {x['values_not_bit_identical']['pchip']}, "
              f"offline GP {x['values_not_bit_identical']['gp']}.", "",
              "| estimator | r2 | r3 | r4 | pooled (mean of r2-r4) | printed | b70_d1.md | reproduced |", "|---|---|---|---|---|---|---|---|"]
        for e in B70_RMSE:
            L.append(f"| {e} | " + " | ".join(f"{x['per_repartition_rmse_eV'][f'r{r}'][e]:.3f}" for r in C.REPEATS)
                     + f" | {x['pooled_rmse_eV'][e]:.3f} | {x['printed'][e]} | {x['b70_d1_md'][e]} | {x['reproduced'][e]} |")
        L += ["", "Hold and causal GP: B.70's own squared errors; PCHIP and offline GP: this batch's values against B.70's `y_true`.",
              "", f"References passes: **{x['passes']}**", ""]
    if "no_look" in J:
        x = J["no_look"]
        L += ["## No look at row t", "",
              f"{x['rows']} D1 rows (stream {x['stream']}); the value of row t (a) set to missing, (b) raised by 1000 eV, in a "
              f"copy of the dataset array; PCHIP and the offline GP recomputed. Rows whose value changed: removed "
              f"{x['changed']['removed']}, raised {x['changed']['raised']}.", "", f"No look passes: **{x['passes']}**", ""]
    if "rows" in J:
        x = J["rows"]
        ri = x["refs_info"]
        L += ["## Rows", "",
              f"D1 rows {ri['rows']:,}; reference formed (a measured value before and after row t in B.70's block) on "
              f"{ri['formed']:,}; not formed {ri['not_formed']:,} (no value before {ri['no_past']}, no value after "
              f"{ri['no_future']:,}). Offline GP fits that failed on every grid point and fell back to hold: "
              f"{ri['gp_fit_failed_fell_back_to_hold']}. PCHIP on exactly two neighbour times (linear): "
              f"{ri['pchip_two_points_linear']:,}.", "",
              "| design | scored rows | reference rows | left out (no reference) | discharges | clusters | every estimator finite on the same rows |",
              "|---|---|---|---|---|---|---|"]
        for d, v in x["designs"].items():
            L.append(f"| {M.DESIGN_LABEL[d]} | {v['scored_rows']:,} | {v['reference_rows']:,} | "
                     f"{v['rows_left_out_missing_reference']:,} | {v['discharges']} | {v['clusters']} | {v['passes']} |")
        L += ["", f"Rows passes: **{x['passes']}**", ""]
    if "stored" in J:
        x = J["stored"]
        L += ["## Stored estimates of B.73 (V1, target F, all D1 rows)", "",
              "| estimator | r2 | r3 | r4 | pooled | printed | b73_targets.md | reproduced |", "|---|---|---|---|---|---|---|---|"]
        for e in B73_RMSE:
            L.append(f"| {e} | " + " | ".join(f"{x['per_repartition_rmse_eV'][f'r{r}'][e]:.3f}" for r in C.REPEATS)
                     + f" | {x['pooled_rmse_eV'][e]:.3f} | {x['printed'][e]} | {x['b73_targets_md'][e]} | {x['reproduced'][e]} |")
        L += ["", f"Stored estimates passes: **{x['passes']}**", ""]
    ok, failed = M.checks_passed()
    L += [f"All four checks pass: **{ok}**" + ("" if ok else f" (first not passed or not run: {failed})"), ""]
    M.save_text(M.CHECKS_MD, "\n".join(L))
    M.log(f"written {M.CHECKS_MD.name}")


if __name__ == "__main__":
    fn = {"references": references, "no_look": no_look, "rows": rows, "stored": stored, "render": render}[sys.argv[1]]
    r = fn()
    sys.exit(0 if r is None or r else 1)
