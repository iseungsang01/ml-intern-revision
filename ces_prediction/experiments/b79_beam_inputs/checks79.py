"""B.79 checks (registration section 9, Addendum 1 section 6).  Run before any readout.

    py ces_prediction/experiments/b79_beam_inputs/checks79.py repro       # T0, TX on B.75's sets vs B.75's stored estimates
    py ces_prediction/experiments/b79_beam_inputs/checks79.py causality   # beam scalars of row t unchanged by later rows
    py ces_prediction/experiments/b79_beam_inputs/checks79.py fits        # after the fits: B.79's T0, TX files vs B.75's
    py ces_prediction/experiments/b79_beam_inputs/checks79.py write       # docs/paper/b79_checks.md (main + refit section)
The join check is written by stage_beam79.py join (data/.b79/checks/join.json).
B79_VARIANT=main (default): the run with the beam columns as delivered, checks in data/.b79/checks/;
B79_VARIANT=refit: the run under the session lead's validity rule, checks in data/.b79/refit_valid/checks/.
Output: {name}.json of the run, docs/paper/b79_checks.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import common79 as M

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
TOL_REPRO = 1e-6


def bitsame(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------------------------ check 2
def check_repro():
    import fits79 as F
    ctx = F.SF75.Ctx()
    S75 = M.load_npz(M.K75.CACHE / "sets.npz")
    sets = M.load_sets()
    us, _, _ = M.units(M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    out, worst = {}, 0.0
    for un in ("V1_r2_k0", "V4a"):
        u = next(x for x in us if x["name"] == un)
        fast8, ms_b, ms_e = ctx.unit_arrays(u)
        z, Gr = ctx.matrices("D650", fast8, ms_b, ms_e)
        B = M.load_beam("D650")
        for L in ("T0", "TX"):
            for X in M.TARGETS:
                te, est, nf, ni = F.fit_one(ctx, u, L, "D650", X, z, Gr, B, S75[f"common_{X}"])
                ref = M.load_npz(M.K75.FITS / f"{un}__{L}__D650__{X}.npz")
                same = bool(np.array_equal(ref["idx"], te))
                dev = float(np.max(np.abs(ref["est"] - est))) if same else float("inf")
                worst = max(worst, dev)
                out[f"{un}|{L}|D650|{X}"] = {"same_test_rows": same, "max_abs_dev_eV": dev, "bit_identical": bool(same and bitsame(ref["est"], est)),
                                             "fit_rows": int(nf), "fit_rows_b75": int(ref["fit_rows"]), "n_iter": ni,
                                             "n_iter_b75": int(ref["n_iter"])}
                M.log(f"repro {un} {L} D650 {X}: max dev {dev:.1e} eV")
    same_sets = {k: bool(np.array_equal(sets[k], S75[k])) for k in ("common_F", "common_L5", "S_F", "S_L5")}
    out["b79_sets_equal_b75_sets"] = same_sets
    out["max_abs_dev_eV"] = worst
    out["passes"] = bool(worst <= TOL_REPRO)
    M.save_json(M.VCHK / "repro.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ check 3
def check_causality():
    G = M.grid_lean(("off", "seg", "grow", "fidx", "time"))
    off, seg = G["off"], G["seg"]
    ok = M.grid_row_mask(G["grow"], G["fidx"])
    bc = M.load_npz(M.CACHE / "beam_cols.npz")
    R = M.rows_lean()
    sets = M.load_sets()
    out, allok = {}, True
    for rule in M.RULES:
        B = M.load_beam(rule)
        anc_g = M.K75.load_inputs(rule)["anc_g"]
        cand = np.flatnonzero(sets["S_F" if rule == "S" else "common_F"])
        pick = np.sort(M.rng_for(f"causality|{rule}").choice(cand, 200, replace=False))
        res = {"rows": int(len(pick)), "candidates": int(len(cand))}
        for mode in ("raised", "emptied", "positive control: raised from row t on"):
            n_same, n_mod = 0, 0
            for j in pick:
                i = int(R["fidx"][j])
                a, b = off[i], off[i + 1]
                t = int(R["grow"][j])
                anc = int(anc_g[j]) - a
                if not (0 <= anc < t):
                    raise SystemExit(f"FATAL: anchor {anc} not before row {t}")
                P, V = bc["P"][a:b].copy(), bc["V"][a:b].copy()
                first = t if mode.startswith("positive") else t + 1
                if mode == "emptied":
                    P[first:] = np.nan
                    V[first:] = np.nan
                else:
                    P[first:] = np.where(np.isfinite(P[first:]), P[first:], 0.0) + 1.0
                    V[first:] = V[first:] + 10.0
                n_mod += int(b - a - first)
                cols = M.beam_cols(P, V, seg[a:b], ok[a:b])
                x = M.beam_scalars(cols, np.array([t]), np.array([anc]))
                n_same += int(bitsame(x, B[j:j + 1]))
            res[mode] = {"rows_identical": n_same, "later_rows_modified_total": n_mod}
            if mode.startswith("positive"):
                allok &= n_same == 0
            else:
                allok &= n_same == len(pick)
        out[rule] = res
        M.log(f"causality {rule}: {res}")
    out["passes"] = bool(allok)
    M.save_json(M.VCHK / "causality.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ after the fits
def check_fits():
    if M.VARIANT == "refit":
        return check_fits_refit()
    return check_fits_main()


def check_fits_main():
    """Every T0 and TX file of B.79 against B.75's file of the same unit, rule and target (the row sets are equal)."""
    us, _, _ = M.units(M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    n, n_bit, worst, bad = 0, 0, 0.0, []
    for u in us:
        for L in ("T0", "TX"):
            for r in M.RULES:
                for X in M.TARGETS:
                    a = M.load_npz(M.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
                    b = M.load_npz(M.K75.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
                    n += 1
                    if not np.array_equal(a["idx"], b["idx"]):
                        bad.append(f"{u['name']} {L} {r} {X}: other rows")
                        continue
                    worst = max(worst, float(np.max(np.abs(a["est"] - b["est"]))))
                    n_bit += int(bitsame(a["est"], b["est"]))
    out = {"files_compared": n, "bit_identical": n_bit, "max_abs_dev_eV": worst, "other_rows": bad,
           "passes": bool(not bad and worst <= TOL_REPRO)}
    M.save_json(M.VCHK / "fits_vs_b75.json", out)
    print(json.dumps(out, indent=1))


def check_fits_refit():
    """Refit run: every T0 and TX file against B.75's file of the same unit, rule and target.  The row sets of the refit
    are B.75's without the rows of the invalid files, so the fits differ; reported: test rows of B.75 not in the refit
    (all of them rows of the invalid files), and the deviation of the estimates on the shared test rows."""
    R = M.rows_lean()
    inval = np.isin(R["shot"], M.invalid_files())
    us, _, _ = M.units(M.load_keys(C.CACHE / "rows.npz", ("fidx", "shot", "clus", "r2_k", "r3_k", "r4_k")))
    per, n, worst, other = {}, 0, 0.0, []
    for u in us:
        for L in ("T0", "TX"):
            for r in M.RULES:
                for X in M.TARGETS:
                    a = M.load_npz(M.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
                    b = M.load_npz(M.K75.FITS / f"{u['name']}__{L}__{r}__{X}.npz")
                    n += 1
                    if np.setdiff1d(a["idx"], b["idx"]).size:
                        other.append(f"{u['name']} {L} {r} {X}: a test row not in B.75's")
                    gone = np.setdiff1d(b["idx"], a["idx"])
                    if not inval[gone].all():
                        other.append(f"{u['name']} {L} {r} {X}: a removed row outside the invalid files")
                    shared, ia, ib = np.intersect1d(a["idx"], b["idx"], return_indices=True)
                    dd = a["est"][ia] - b["est"][ib]
                    dev = float(np.max(np.abs(dd))) if len(shared) else 0.0
                    worst = max(worst, dev)
                    per[f"{u['name']}|{L}|{r}|{X}"] = {"test_rows": int(len(a["idx"])), "test_rows_b75": int(len(b["idx"])),
                                                       "removed_test_rows": int(len(gone)), "max_abs_dev_shared_eV": dev,
                                                       "rms_dev_shared_eV": float(np.sqrt(np.mean(dd ** 2))) if len(shared) else 0.0,
                                                       "fit_rows": int(a["fit_rows"]), "fit_rows_b75": int(b["fit_rows"])}
    out = {"files_compared": n, "max_abs_dev_on_shared_test_rows_eV": worst,
           "median_of_per_file_max_dev_eV": float(np.median([x["max_abs_dev_shared_eV"] for x in per.values()])),
           "median_of_per_file_rms_dev_eV": float(np.median([x["rms_dev_shared_eV"] for x in per.values()])),
           "largest_per_file_rms_dev_eV": float(np.max([x["rms_dev_shared_eV"] for x in per.values()])),
           "problems": other, "passes": bool(not other), "per_file": per}
    M.save_json(M.VCHK / "fits_vs_b75.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_file"}, indent=1))


# ------------------------------------------------------------------------------------------ md
def main_text():
    """The checks of the main run (columns as delivered), as written at 17:04 on 2026-11-20."""
    names = ("join", "beam_facts", "repro", "causality", "fits_vs_b75")
    j = {n: json.loads((M.CHK / f"{n}.json").read_text(encoding="utf-8")) for n in names if (M.CHK / f"{n}.json").exists()}
    si = json.loads((M.CACHE / "sets.json").read_text(encoding="utf-8"))
    jo, fa, rp, ca = j["join"], j["beam_facts"], j["repro"], j["causality"]
    L = ["# B.79 checks", "",
         f"Registration: `{M.REG}` section 9; implementation: `{M.ADD}`. Run before any readout. No tolerance was changed "
         "after a run.", "",
         "## 1. The join of the beam rows to the grid rows", "",
         f"Shot files {jo['files']}; without an NBI file: {len(jo['files_without_nbi_file'])}; with another number of rows: "
         f"{len(jo['files_with_other_row_count'])}. Rows {jo['rows_all']:,} (grid rows without each file's first and last row: "
         f"{jo['rows_grid']:,}). Rows whose beam stamp differs from the grid stamp by more than {jo['tolerance_s']:.0e} s or "
         f"that have no beam row: **{jo['mismatches_all_rows']}** (grid rows: {jo['mismatches_grid_rows']}). Largest stamp "
         f"difference {jo['max_abs_stamp_difference_s']:.1e} s; rows whose beam stamp rounded to float32 equals the grid stamp: "
         f"{jo['rows_float32_stamp_equal']:,}.", "", f"Check 1 passes: **{jo['passes']}**", "",
         "## 2. T0 and TX on B.75's row sets against B.75's stored estimates (D650)", "",
         "| unit / learner / rule / target | same test rows | largest deviation (eV) | bit-identical | fit rows (B.75) | iterations (B.75) |",
         "|---|---|---|---|---|---|"]
    for k, x in rp.items():
        if isinstance(x, dict) and "same_test_rows" in x:
            L.append(f"| {k.replace('|', ' / ')} | {x['same_test_rows']} | {x['max_abs_dev_eV']:.1e} | {x['bit_identical']} | "
                     f"{x['fit_rows']:,} ({x['fit_rows_b75']:,}) | {x['n_iter']} ({x['n_iter_b75']}) |")
    L += ["", "The row sets of this batch equal B.75's: " + ", ".join(f"{k} {v}" for k, v in rp["b79_sets_equal_b75_sets"].items()) +
          " (no row of B.75's sets lacks a beam scalar), so there is no difference on the rows of this batch to report.", "",
          f"Check 2 passes: **{rp['passes']}**", ""]
    if "fits_vs_b75" in j:
        fv = j["fits_vs_b75"]
        L += [f"After the fits: every T0 and TX file of this batch (36 units x D10, D650, S x F, L5) against B.75's file of the "
              f"same unit, rule and target: {fv['files_compared']} files, bit-identical {fv['bit_identical']}, largest deviation "
              f"{fv['max_abs_dev_eV']:.1e} eV, files with other rows {len(fv['other_rows'])}.", ""]
    L += ["## 3. No beam scalar of row t uses a row after t", "",
          "For each row, a copy of its file's beam columns in which every row after t is changed (raised: every power + 1 MW "
          "with an empty power read as zero first, NB1B voltage + 10 kV; emptied: every power and the voltage made empty); "
          "the nineteen scalars of row t recomputed with the row's anchor and compared bit for bit with the main pass. "
          "Positive control: the same raise applied from row t on must change the scalars of every row.", "",
          "| rule | rows (candidates) | raised: rows identical | emptied: rows identical | positive control: rows identical (must be 0) |",
          "|---|---|---|---|---|"]
    for rule in M.RULES:
        x = ca[rule]
        L.append(f"| {rule} | {x['rows']} ({x['candidates']:,}) | {x['raised']['rows_identical']} | {x['emptied']['rows_identical']} | "
                 f"{x['positive control: raised from row t on']['rows_identical']} |")
    L += ["", f"Check 3 passes: **{ca['passes']}**", "",
          "## The beam columns (input facts, no target and no estimate)", "",
          f"Grid rows {fa['grid_rows']:,}. An empty power is read as zero.", "",
          "| source | empty rows | largest (MW) | smallest (MW) | 99th percentile (MW) | rows above 3 MW (files) | rows below -0.1 MW (files) |",
          "|---|---|---|---|---|---|---|"]
    for s, x in fa["per_source"].items():
        L.append(f"| {s} | {x['empty_rows']:,} | {x['max_MW']:.2f} | {x['min_MW']:.2f} | {x['p99_MW']:.3f} | "
                 f"{x['rows_above_3MW']:,} ({len(x['files_above_3MW'])}) | {x['rows_below_minus_0.1MW']:,} ({len(x['files_below_minus_0.1MW'])}) |")
    L += ["", f"Every power above 3 MW lies in thirteen files (shots {', '.join(str(s) for s in fa['d1_files_with_a_power_above_3MW'])}), "
              f"all NBI-2 columns; {fa['d1_rows_with_a_power_above_3MW']:,} D1 rows carry one. In the same files the NBI-2 powers "
              "read as low as -3.5 MW while the source is off. The NB1B voltage is empty on "
              f"{fa['NB1B_V_empty_while_on']} rows with NB1B on and on {fa['NB1B_V_empty_while_off']} rows with it off.", "",
          "## Row sets", "",
          f"D1 rows {si['d1_rows']:,}. Rows whose nineteen beam scalars are defined: " +
          ", ".join(f"{r} {v:,}" for r, v in si["beam_complete"].items()) + ".", "",
          "| set | B.75 | B.79 | dropped by the beam scalars | rows of the thirteen files |", "|---|---|---|---|---|"]
    for X in M.TARGETS:
        c, s = si[f"common_{X}"], si[f"S_{X}"]
        L.append(f"| common, {X} | {c['b75']:,} | {c['b79']:,} | D10 {c['dropped_by_beam_D10']}, D650 {c['dropped_by_beam_D650']} | {si[f'common_{X}_rows_of_the_13_files']:,} |")
        L.append(f"| S, {X} | {s['b75']:,} | {s['b79']:,} | {s['dropped_by_beam_S']} | {si[f'S_{X}_rows_of_the_13_files']:,} |")
    L += ["", "Undefined beam scalars over all D1 rows (they lie outside B.75's sets, which already require the anchor and its "
              "running means): " + "; ".join(f"{r}: " + ", ".join(f"{k} {v:,}" for k, v in si[f"undefined_scalars_{r}"].items())
                                              for r in M.RULES) + ".", ""]
    return "\n".join(L)


def refit_text():
    """The checks of the refit without the thirteen files (Addendum 1 sections 11 to 13), appended to the main checks."""
    RD = M.REFIT_DIR
    rj = lambda n: json.loads((RD / "checks" / f"{n}.json").read_text(encoding="utf-8"))   # noqa: E731
    va = json.loads((M.CHK / "validity.json").read_text(encoding="utf-8"))
    sv, rp, ca, fv = rj("scalars_vs_as_delivered"), rj("repro"), rj("causality"), rj("fits_vs_b75")
    si = json.loads((RD / "sets.json").read_text(encoding="utf-8"))
    pf = fv["per_file"]
    L = ["## Refit without the thirteen files (sensitivity; Addendum 1 sections 11 to 13)", "",
         "Run after the main readout had been read, under the session lead's validity rule. Caches: `data/.b79/refit_valid/`. "
         "Not judged as the registered statements.", "",
         "### R1. The validity rule selects exactly the thirteen files", "",
         f"Rule: {va['rule']}. A file with at least one invalid pair has no valid beam scalars: all nineteen are non-finite on "
         "every row of the file, and by the registration's row rule (complete for TXN) its rows are in no fit and no score, "
         "for every learner.", "",
         f"Files selected: {len(va['invalid_files'])} ({', '.join(str(s) for s in va['invalid_files'])}); exactly the thirteen "
         f"files named beforehand: **{va['selects_exactly_the_expected_files']}**. Invalid file-source pairs: {len(va['invalid_pairs'])}.", "",
         "| shot | source | power column, smallest to largest (MW) | rows above 5 MW | rows below -0.1 MW | rows with power above 0.1 MW | median V x I / 1000 on those rows |",
         "|---|---|---|---|---|---|---|"]
    for p in va["invalid_pairs"]:
        med = "-" if p["median_VI_over_1000_on_rows"] is None else f"{p['median_VI_over_1000_on_rows']:.2f}"
        L.append(f"| {p['shot']} | {p['source']} | {p['power_min_MW']:.2f} to {p['power_max_MW']:.2f} | {p['above_5MW_rows']} | "
                 f"{p['below_minus_0.1MW_rows']} | {p['on_rows']} | {med} |")
    L += ["", "Rows over the file's grid rows (first and last row excluded). Beam scalars of the refit against the main run: " +
          "; ".join(f"{r}: rows of the other files bit-identical {x['valid_files_bit_identical_to_as_delivered']} ({x['rows_of_valid_files']:,} rows), "
                    f"rows of the thirteen files all non-finite {x['invalid_file_rows_all_nonfinite']} ({x['rows_of_invalid_files']:,} rows)"
                    for r, x in sv.items()) + ".", "",
          "### R2. Row sets of the refit", "",
          f"D1 rows of the thirteen files: {si['d1_rows_of_invalid_files']:,} in {si['d1_discharges_of_invalid_files']} discharges.", "",
          "| set | main (= B.75) rows (discharges) | refit rows (discharges) | rows removed: of the thirteen files / otherwise | discharges removed |",
          "|---|---|---|---|---|"]
    for X in M.TARGETS:
        for nm in ("common", "S"):
            c = si[f"{nm}_{X}"]
            L.append(f"| {nm}, {X} | {c['b75']:,} ({c['discharges_b75']}) | {c['b79']:,} ({c['discharges_b79']}) | "
                     f"{c['rows_removed_of_invalid_files']:,} / {c['rows_removed_otherwise']} | {c['discharges_removed']} |")
    L += ["", "### R3. Check 2 rerun (on B.75's sets, as registered) and check 3 rerun (on the refit's sets)", "",
          f"Check 2: largest deviation {rp['max_abs_dev_eV']:.1e} eV over "
          f"{sum(1 for x in rp.values() if isinstance(x, dict) and 'same_test_rows' in x)} fits; passes **{rp['passes']}**.", "",
          "| rule | rows (candidates) | raised: rows identical | emptied: rows identical | positive control: rows identical (must be 0) |",
          "|---|---|---|---|---|"]
    for rule in M.RULES:
        x = ca[rule]
        L.append(f"| {rule} | {x['rows']} ({x['candidates']:,}) | {x['raised']['rows_identical']} | {x['emptied']['rows_identical']} | "
                 f"{x['positive control: raised from row t on']['rows_identical']} |")
    L += ["", f"Check 3 passes: **{ca['passes']}**", "",
          "### R4. The refit's T0 and TX against B.75's", "",
          f"Every T0 and TX file of the refit (36 units x D10, D650, S x F, L5; {fv['files_compared']} files) against B.75's file "
          "of the same unit, rule and target. Every test row of the refit is a test row of B.75's; every B.75 test row not in "
          f"the refit is a row of the thirteen files (problems found: {len(fv['problems'])}). Because the fit rows lose the rows "
          "of the thirteen files, the estimates differ on the shared test rows: largest deviation "
          f"{fv['max_abs_dev_on_shared_test_rows_eV']:.1f} eV, median over files of the per-file largest deviation "
          f"{fv['median_of_per_file_max_dev_eV']:.1f} eV; root mean square deviation per file: median "
          f"{fv['median_of_per_file_rms_dev_eV']:.1f} eV, largest {fv['largest_per_file_rms_dev_eV']:.1f} eV. Units V1_r2_k0 and V4a under D650:", "",
          "| unit / learner / rule / target | test rows (B.75) | fit rows (B.75) | largest deviation on shared test rows (eV) | root mean square deviation (eV) |",
          "|---|---|---|---|---|"]
    for un in ("V1_r2_k0", "V4a"):
        for Lr in ("T0", "TX"):
            for X in M.TARGETS:
                x = pf[f"{un}|{Lr}|D650|{X}"]
                L.append(f"| {un} / {Lr} / D650 / {X} | {x['test_rows']:,} ({x['test_rows_b75']:,}) | {x['fit_rows']:,} ({x['fit_rows_b75']:,}) | "
                         f"{x['max_abs_dev_shared_eV']:.1f} | {x['rms_dev_shared_eV']:.1f} |")
    L.append("")
    return "\n".join(L)


def write():
    text = main_text()
    if (M.REFIT_DIR / "checks" / "fits_vs_b75.json").exists():
        text += "\n" + refit_text()
    M.save_text(M.CHECKS_MD, text)
    print("written", M.CHECKS_MD)


if __name__ == "__main__":
    t0 = time.time()
    {"repro": check_repro, "causality": check_causality, "fits": check_fits, "write": write}[sys.argv[1]]()
    M.log(f"done ({time.time() - t0:.0f}s)")
