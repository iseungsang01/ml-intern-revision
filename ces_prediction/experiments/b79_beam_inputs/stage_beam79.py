"""B.79 stage 1: the beam columns joined to the grid (check 1 of registration section 9), the nineteen beam scalars of
every D1 row under D10, D650 and S (Addendum 1 sections 2-3), and the row sets (section 4).

    py ces_prediction/experiments/b79_beam_inputs/stage_beam79.py join      # beam columns + join check + input facts
    py ces_prediction/experiments/b79_beam_inputs/stage_beam79.py scalars   # beam_{rule}.npz for D10, D650, S
    py ces_prediction/experiments/b79_beam_inputs/stage_beam79.py sets      # sets.npz, sets.json
Output: data/.b79/beam_cols.npz, data/.b79/beam_{rule}.npz (B: D1 rows x 19, rows.npz order), data/.b79/sets.npz,
        data/.b79/sets.json, data/.b79/checks/join.json, data/.b79/checks/beam_facts.json
"""
from __future__ import annotations

import sys
import time

import common79 as M

import numpy as np  # noqa: E402

C = M.C


def run_join():
    G = M.grid_lean(("names", "shot", "off", "time", "grow", "fidx"))
    names, shots, off, tm = G["names"], G["shot"], G["off"], G["time"]
    N = len(tm)
    P = np.full((N, len(M.SRC)), np.nan)
    V = np.full(N, np.nan)
    tn = np.full(N, np.nan)
    missing, rowcount_bad = [], []
    for i, nm in enumerate(names):
        sh = int(shots[i])
        if not (M.NBI / f"s{sh}_NBI.csv").exists():
            missing.append(sh)
            continue
        t_, P_, V_ = M.read_nbi(sh)
        a, b = off[i], off[i + 1]
        if len(t_) != b - a:
            rowcount_bad.append([sh, int(len(t_)), int(b - a)])
            continue
        tn[a:b], P[a:b], V[a:b] = t_, P_, V_
    gr = M.grid_row_mask(G["grow"], G["fidx"])
    d = np.abs(tn - tm)
    mism = ~(d <= M.JOIN_TOL)                       # NaN (no beam row) counts as a mismatch
    f32 = np.float32(tn).astype(np.float64) == tm
    out = {"files": int(len(names)), "files_without_nbi_file": missing, "files_with_other_row_count": rowcount_bad,
           "rows_all": int(N), "rows_grid": int(gr.sum()),
           "mismatches_all_rows": int(mism.sum()), "mismatches_grid_rows": int((mism & gr).sum()),
           "max_abs_stamp_difference_s": float(np.nanmax(d)), "rows_float32_stamp_equal": int(f32.sum()),
           "tolerance_s": M.JOIN_TOL}
    out["passes"] = bool(not missing and not rowcount_bad and out["mismatches_all_rows"] == 0)
    M.save_json(M.CHK / "join.json", out)
    M.save_npz(M.CACHE / "beam_cols.npz", P=P, V=V, t=tn)
    # ---- facts of the input columns (no target, no estimate)
    Pg, Vg = P[gr], V[gr]
    Pz = np.where(np.isfinite(Pg), Pg, 0.0)
    fid = G["fidx"][gr]
    facts = {"grid_rows": int(gr.sum()), "per_source": {}}
    for k, s in enumerate(M.SRC):
        x = Pz[:, k]
        big = x > 3.0
        neg = x < -M.ON_MW
        facts["per_source"][s] = {
            "empty_rows": int(np.isnan(Pg[:, k]).sum()), "max_MW": float(x.max()), "min_MW": float(x.min()),
            "p99_MW": float(np.percentile(x, 99)), "rows_above_3MW": int(big.sum()),
            "files_above_3MW": [int(shots[f]) for f in np.unique(fid[big])],
            "rows_below_minus_0.1MW": int(neg.sum()), "files_below_minus_0.1MW": [int(shots[f]) for f in np.unique(fid[neg])]}
    on1b = Pz[:, 0] > M.ON_MW
    facts["NB1B_V_empty_while_on"] = int(np.isnan(Vg[on1b]).sum())
    facts["NB1B_V_empty_while_off"] = int(np.isnan(Vg[~on1b]).sum())
    R = M.rows_lean()
    big_any = np.zeros(N, bool)
    big_any[gr] = (Pz > 3.0).any(1)
    facts["d1_rows_with_a_power_above_3MW"] = int(big_any[R["gi"]].sum())
    facts["d1_files_with_a_power_above_3MW"] = sorted(int(shots[f]) for f in np.unique(G["fidx"][big_any]))
    M.save_json(M.CHK / "beam_facts.json", facts)
    M.log(f"join: mismatches {out['mismatches_all_rows']} of {N:,} rows, passes {out['passes']}")


def run_validity():
    """Validity rule of the beam inputs (session lead, 2026-11-20; Addendum 1 section 11): a file-source pair is invalid
    if its power column is above 5 MW or below -0.1 MW in any grid row of the file; a file with an invalid pair has no
    valid beam scalars."""
    import pandas as pd
    G = M.grid_lean(("names", "shot", "off", "grow", "fidx"))
    shots, off = G["shot"], G["off"]
    gr = M.grid_row_mask(G["grow"], G["fidx"])
    P = M.load_npz(M.CACHE / "beam_cols.npz")["P"]
    pairs, files = [], []
    for i in range(len(shots)):
        a, b = off[i], off[i + 1]
        m = gr[a:b]
        bad_here = []
        for k, s in enumerate(M.SRC):
            x = P[a:b, k][m]
            fin = np.isfinite(x)
            if not fin.any():
                continue
            hi, lo = float(x[fin].max()), float(x[fin].min())
            if hi > M.INVALID_HI_MW or lo < M.INVALID_LO_MW:
                bad_here.append((k, s, lo, hi))
        if bad_here:
            files.append(int(shots[i]))
            df = pd.read_csv(M.NBI / f"s{int(shots[i])}_NBI.csv")
            for k, s, lo, hi in bad_here:
                p = df[s + "_P"].to_numpy(np.float64)[1:-1]
                vi = (df[s + "_V"].to_numpy(np.float64) * df[s + "_I"].to_numpy(np.float64))[1:-1] / 1000.0
                on = np.where(np.isfinite(p), p, 0.0) > M.ON_MW
                pairs.append({"shot": int(shots[i]), "source": s, "power_min_MW": lo, "power_max_MW": hi,
                              "above_5MW_rows": int(np.sum(p > M.INVALID_HI_MW)), "below_minus_0.1MW_rows": int(np.sum(p < M.INVALID_LO_MW)),
                              "on_rows": int(on.sum()),
                              "median_VI_over_1000_on_rows": float(np.nanmedian(vi[on])) if on.any() else None})
    expected = [30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909]
    out = {"rule": f"a file-source pair (sources {', '.join(M.SRC)}) is invalid if its power column is above "
                   f"{M.INVALID_HI_MW} MW or below {M.INVALID_LO_MW} MW in any grid row of the file",
           "invalid_files": files, "invalid_pairs": pairs, "expected_files": expected,
           "selects_exactly_the_expected_files": files == expected}
    M.save_json(M.CHK / "validity.json", out)
    M.log(f"validity: {len(files)} files, {len(pairs)} pairs; exactly the expected 13: {out['selects_exactly_the_expected_files']}")


def file_scalars(cols_file, grow_rows, anc_local):
    return M.beam_scalars(cols_file, grow_rows, anc_local)


def run_scalars():
    """Beam scalars of the run (B79_VARIANT): main = the columns as delivered; refit = the validity rule applied (the
    rows of the invalid files get no beam scalar), compared with the main run's scalars."""
    G = M.grid_lean(("off", "seg", "grow", "fidx"))
    off, seg = G["off"], G["seg"]
    ok = M.grid_row_mask(G["grow"], G["fidx"])
    bc = M.load_npz(M.CACHE / "beam_cols.npz")
    R = M.rows_lean()
    files = np.unique(R["fidx"])
    refit = M.VARIANT == "refit"
    invalid = set(M.invalid_files()) if refit else set()
    bad_rows = np.isin(R["shot"], sorted(invalid))
    vs = {}
    t0 = time.time()
    for rule in M.RULES:
        out = M.VDIR / f"beam_{rule}.npz"
        if out.exists():
            M.log(f"{rule}: cached")
            continue
        anc_g = M.K75.load_inputs(rule)["anc_g"]
        B = np.full((len(R["gi"]), M.NB), np.nan)
        for i in files:
            a, b = off[i], off[i + 1]
            q = np.flatnonzero(R["fidx"] == i)
            if int(R["shot"][q[0]]) in invalid:          # validity rule: no valid beam scalar in this file
                continue
            cols = M.beam_cols(bc["P"][a:b], bc["V"][a:b], seg[a:b], ok[a:b])
            anc = anc_g[q]
            B[q] = file_scalars(cols, R["grow"][q], np.where(anc >= 0, anc - a, -1))
        M.save_npz(out, B=B)
        if refit and (M.CACHE / f"beam_{rule}.npz").exists():    # against the main run (columns as delivered)
            Bo = np.load(M.CACHE / f"beam_{rule}.npz")["B"]
            vs[rule] = {"rows_of_valid_files": int((~bad_rows).sum()),
                        "valid_files_bit_identical_to_as_delivered": bool(
                            np.ascontiguousarray(B[~bad_rows]).tobytes() == np.ascontiguousarray(Bo[~bad_rows]).tobytes()),
                        "rows_of_invalid_files": int(bad_rows.sum()),
                        "invalid_file_rows_all_nonfinite": bool((~np.isfinite(B[bad_rows])).all())}
        M.log(f"{rule}: written ({time.time() - t0:.0f}s); complete rows {int(np.isfinite(B).all(1).sum()):,} of {len(B):,}")
    if vs:
        M.save_json(M.VCHK / "scalars_vs_as_delivered.json", vs)
        M.log(f"scalars against the as-delivered computation: {vs}")


def run_sets():
    S75 = M.load_npz(M.K75.CACHE / "sets.npz")
    R = M.rows_lean()
    comp = {r: np.isfinite(M.load_beam(r)).all(1) for r in M.RULES}
    arr, info = {}, {"d1_rows": int(len(R["gi"])), "beam_complete": {r: int(comp[r].sum()) for r in M.RULES}}
    early = np.isin(R["shot"], M.invalid_files())       # D1 rows of the thirteen files (validity.json)
    arr["files13"] = early
    for X in M.TARGETS:
        c75, s75 = S75[f"common_{X}"], S75[f"S_{X}"]
        arr[f"common_{X}"] = c75 & comp["D10"] & comp["D650"]
        arr[f"S_{X}"] = s75 & comp["S"]
        info[f"common_{X}"] = {"b75": int(c75.sum()), "b79": int(arr[f"common_{X}"].sum()),
                               "dropped_by_beam_D10": int((c75 & ~comp["D10"]).sum()),
                               "dropped_by_beam_D650": int((c75 & ~comp["D650"]).sum())}
        info[f"S_{X}"] = {"b75": int(s75.sum()), "b79": int(arr[f"S_{X}"].sum()), "dropped_by_beam_S": int((s75 & ~comp["S"]).sum())}
        for nm, b75 in (("common", c75), ("S", s75)):
            b79 = arr[f"{nm}_{X}"]
            gone = b75 & ~b79
            info[f"{nm}_{X}"].update({"rows_removed": int(gone.sum()), "rows_removed_of_invalid_files": int((gone & early).sum()),
                                      "rows_removed_otherwise": int((gone & ~early).sum()),
                                      "discharges_b75": int(np.unique(R["fidx"][b75]).size),
                                      "discharges_b79": int(np.unique(R["fidx"][b79]).size),
                                      "discharges_removed": int(np.unique(R["fidx"][b75]).size - np.unique(R["fidx"][b79]).size)})
            info[f"{nm}_{X}_rows_of_the_13_files"] = int((b79 & early).sum())
    # dropped rows by which beam scalar is undefined (all sets pooled)
    for r in M.RULES:
        B = M.load_beam(r)
        info[f"undefined_scalars_{r}"] = {M.BEAM_NAMES[k]: int((~np.isfinite(B[:, k])).sum()) for k in range(M.NB)
                                          if (~np.isfinite(B[:, k])).any()}
    import json
    info["variant"] = M.VARIANT
    info["invalid_files"] = M.invalid_files()
    info["d1_rows_of_the_13_files"] = int(early.sum())
    info["d1_rows_of_invalid_files"] = int(early.sum())
    info["d1_discharges_of_invalid_files"] = int(np.unique(R["fidx"][early]).size)
    M.save_npz(M.VDIR / "sets.npz", **arr)
    M.save_json(M.VDIR / "sets.json", info)
    M.log(f"sets: {json.dumps(info)}")


if __name__ == "__main__":
    {"join": run_join, "validity": run_validity, "scalars": run_scalars, "sets": run_sets}[sys.argv[1]]()
