"""B.73 stage 4: the filters of B.56 (K0: three parameters; KX: the two gains of the beam-emission drive added, clip
0.2) refitted per unit of every design and per target (F, L5), by minimising the eV mean squared error of the output
against the target on the fit discharges (registration section 4, Addendum 1).

The frozen definitions of `b56_kfx/kfx_model.py` (filter `run`, its own loader) are executed by `run_b56.ns()`; the held
repeats are removed from its grid as in B.70 phase 2 (`p2_b56_fit.ns_held`), here from the cached seq grid
(data/.b73/grid.npz) with the same checks.  Starts, Nelder-Mead options and fit order are those of
`p2_b56_fit.fit_fold` (KF0, then the two gains with the noise parameters held at the KF0 fit, clip 0.2).

    py ces_prediction/experiments/b73_targets/stage_filters.py [--design V1,...] [--units NAME,...] [--check]
Output: data/.b73/filters/{unit}__{target}.npz and .json; --check: data/.b73/filters/CHECK_r2_k0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import common as C
import designs as DS

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(C.EXP / "b56_kfx"))
import run_b56 as B56  # noqa: E402

OUT = C.CACHE / "filters"
NM = lambda maxiter, xatol: dict(maxiter=maxiter, xatol=xatol, fatol=1e-1)      # noqa: E731
_S = {}


def ns_held():
    """run_b56.ns() with the held repeats removed from Y and LY (p2_b56_fit.ns_held, with the cached grid)."""
    if "g" in _S:
        return _S["g"]
    g = B56.ns()
    G = C.grid()
    names = [str(x) for x in G["names"]]
    from pathlib import Path
    if [Path(f).name for f in g["files"]] != names:
        raise SystemExit("FATAL: the filter's file list differs from the grid's")
    removed = 0
    kinds = {"missing_in_file": 0, "above_3keV": 0, "held_repeat": 0}
    for i, nm in enumerate(names):
        a, b = G["off"][i], G["off"][i + 1]
        nk = len(g["grid"][i]["t"])
        if (b - a) != nk + 2 or not np.allclose(G["time"][a + 1:b - 1], g["grid"][i]["t"], atol=1e-6):
            raise SystemExit(f"FATAL: {nm}: filter rows are not grid rows 1..n-2")
        tis = G["ti"][a + 1:b - 1]
        tik = g["Y"][i, :nk]
        if (np.isfinite(tis) & ~np.isfinite(tik)).any():
            raise SystemExit(f"FATAL: {nm}: a grid value the filter does not have")
        both = np.isfinite(tik) & np.isfinite(tis)
        if both.any() and np.max(np.abs(tik[both] - tis[both])) > 1e-3:
            raise SystemExit(f"FATAL: {nm}: values differ between the filter grid and the seq grid")
        held = np.isfinite(tik) & ~np.isfinite(tis)
        if held.any():
            g["Y"][i, :nk][held] = np.nan
            g["LY"][i, :nk][held] = np.nan
            removed += int(held.sum())
    _S["g"], _S["removed"] = g, removed
    C.log(f"held repeats removed from the filter grid: {removed}")
    return g


def label_kinds(g):
    """Kinds of the positive labels of the classifier rows: missing in the file, above 3 keV, held repeat."""
    out = C.CACHE / "label_kinds.json"
    if out.exists():
        return
    G = C.grid()
    Q = C.clf_rows()
    pos = Q["gidx"][Q["label"] == 1]
    fid = G["fidx"][pos]
    j = G["grow"][pos] - 1                      # filter row
    raw = np.array([g["grid"][int(f)]["ti_raw"][int(r)] for f, r in zip(fid, j)])
    k = {"positives": int(len(pos)), "missing_in_file": int(np.sum(~np.isfinite(raw))),
         "above_3keV": int(np.sum(np.isfinite(raw) & (raw > C.CUT_EV))),
         "held_repeat": int(np.sum(np.isfinite(raw) & (raw <= C.CUT_EV)))}
    C.save_json(out, k)
    C.log(f"label kinds: {k}")


def target_arrays(g):
    """NF x TMAX target arrays on the filter rows: F = Y (the frozen loss); L5 = the following level of the row where the
    row's own value is measured (Addendum 1)."""
    if "T" in _S:
        return _S["T"]
    G = C.grid()
    tm, ti, seg, off, fid = G["time"], G["ti"], G["seg"], G["off"], G["fidx"]
    gi = np.flatnonzero(np.isfinite(ti))
    later = C.later_values(tm, ti, seg, gi, fid, fid[gi])
    L5 = C.targets_from_later(ti[gi], later)["L5"]
    T5 = np.full_like(g["Y"], np.nan)
    rr = gi - off[fid[gi]] - 1                  # filter row
    okr = (rr >= 0) & (rr < np.array([len(g["grid"][int(f)]["t"]) for f in range(len(off) - 1)])[fid[gi]])
    T5[fid[gi][okr], rr[okr]] = L5[okr]
    T5[~np.isfinite(g["Y"])] = np.nan
    _S["T"] = {"F": g["Y"], "L5": T5}
    return _S["T"]


def make_loss(g, TGT):
    run = g["run"]

    def loss(theta, fidx, use_x):
        out = run(theta, None, use_x)[fidx]
        y = TGT[fidx]
        m = np.isfinite(y) & np.isfinite(out)
        return float(np.mean((out[m] - y[m]) ** 2))
    return loss


def fit(g, TGT, fit_files):
    loss = make_loss(g, TGT)
    CLIP = g["CLIP"]
    CLIP[0] = B56.CLIP_MAIN
    t0 = time.time()
    f0 = minimize(lambda th: loss(np.r_[th, 0, 0], fit_files, False),
                  np.array([np.log(0.002), np.log(0.01), np.log(2.0)]), method="Nelder-Mead", options=NM(250, 1e-3))
    th0 = f0.x
    fx = minimize(lambda b_: loss(np.r_[th0, b_], fit_files, True), np.zeros(2), method="Nelder-Mead", options=NM(300, 1e-4))
    return {"theta_k0": th0.tolist(), "fit_mse_k0": float(f0.fun), "b_kx": fx.x.tolist(), "fit_mse_kx": float(fx.fun),
            "clip": B56.CLIP_MAIN, "seconds": round(time.time() - t0, 1), "nfev": [int(f0.nfev), int(fx.nfev)]}


def estimates(g, par, R, test_rows):
    """K0 and KX outputs at the given D1 rows (filter row = grid row - 1)."""
    run, CLIP = g["run"], g["CLIP"]
    CLIP[0] = par["clip"]
    o0 = run(np.r_[par["theta_k0"], 0, 0], None, False)
    ox = run(np.r_[par["theta_k0"], par["b_kx"]], None, True)
    f, j = R["fidx"][test_rows], R["grow"][test_rows] - 1
    return o0[f, j], ox[f, j]


def do_unit(g, R, u, X):
    path = OUT / f"{u['name']}__{X}.npz"
    if path.exists():
        return
    TGT = target_arrays(g)[X]
    fit_files = np.array(sorted(int(i) for i in u["fit"]))
    par = fit(g, TGT, fit_files)
    test_rows = np.flatnonzero(np.isin(R["fidx"], u["test"]))
    e0, ex = estimates(g, par, R, test_rows)
    par.update({"unit": u["name"], "target": X, "fit_files": int(len(fit_files)), "test_rows": int(len(test_rows)),
                "test_rows_without_output": int(np.sum(~np.isfinite(e0) | ~np.isfinite(ex)))})
    C.save_npz(path, idx=test_rows, k0=e0, kx=ex)
    C.save_json(path.with_suffix(".json"), par)
    C.log(f"  {u['name']} [{X}]: theta {np.round(par['theta_k0'], 3).tolist()} b {np.round(par['b_kx'], 3).tolist()} "
          f"({par['seconds']} s, nfev {par['nfev']})")


def check_b70(g):
    """The F fit on the training discharges of r2 f0 (manifest train_files) against data/.b70/kf_params_r2_f0.json."""
    out = OUT / "CHECK_r2_k0.json"
    if out.exists():
        return
    names = [str(x) for x in C.grid()["names"]]
    man = json.loads((C.DATA / ".b27_split_r2_f0" / "split_manifest.json").read_text(encoding="utf-8"))
    train_f = np.array(sorted(names.index(n) for n in man["train_files"]))
    par = fit(g, target_arrays(g)["F"], train_f)
    ref = json.loads((C.DATA / ".b70" / "kf_params_r2_f0.json").read_text(encoding="utf-8"))
    dev = {"KF0_theta_max_abs_dev": float(np.max(np.abs(np.array(par["theta_k0"]) - np.array(ref["KF0"])))),
           "KFX_b_max_abs_dev": float(np.max(np.abs(np.array(par["b_kx"]) - np.array(ref["KFX"][3:]))))}
    C.save_json(out, {"ours": par, "b70": {"KF0": ref["KF0"], "KFX": ref["KFX"]}, **dev,
                      "held_removed": _S["removed"]})
    C.log(f"check against B.70 r2 f0: {dev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="")
    ap.add_argument("--units", default="")
    ap.add_argument("--targets", default="F,L5")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    g = ns_held()
    R = C.rows()
    label_kinds(g)
    if a.check:
        check_b70(g)
        return
    designs = set(a.design.split(",")) if a.design else None
    names = set(a.units.split(",")) if a.units else None
    us, _, _ = DS.units(R)
    for X in a.targets.split(","):
        for u in us:
            if designs and u["design"] not in designs:
                continue
            if names and u["name"] not in names:
                continue
            do_unit(g, R, u, X)
    C.log("filters: done")


if __name__ == "__main__":
    main()
