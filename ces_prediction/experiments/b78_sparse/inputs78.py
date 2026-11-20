"""B.78 stage 1: frame index and scored rows, the estimator inputs of every scored row under each rate rule, and the
row sets (registration sections 3-5, Addendum 1 sections 2-4).

    py ces_prediction/experiments/b78_sparse/inputs78.py frames        # frame index of the grid, scored rows
    py ces_prediction/experiments/b78_sparse/inputs78.py rule K10      # one rule
    py ces_prediction/experiments/b78_sparse/inputs78.py sets          # after all six rules
Output: data/.b78/frames.npz, data/.b78/inputs/{rule}.npz (arrays over the D1 rows in rows.npz order; rows that are not
scored stay undefined), data/.b78/sets.npz, data/.b78/sets.json
"""
from __future__ import annotations

import sys
import time

import common78 as M

import numpy as np  # noqa: E402

C = M.C


def run_frames():
    out = M.CACHE / "frames.npz"
    G = M.grid_lean(("time", "seg", "fidx", "grow"))
    R = M.rows_lean()
    fr, n_off = M.frame_index(G)
    sc = M.scored_mask(fr, R["gi"])
    info = {"grid_rows": int(len(fr)), "rows_off_lattice": n_off, "grid_rows_without_index": int((fr < 0).sum()),
            "d1_rows": int(len(R["gi"])), "d1_rows_without_index": int((fr[R["gi"]] < 0).sum()),
            "scored_rows": int(sc.sum())}
    M.save_npz(out, fr=fr, scored=sc)
    M.save_json(M.CACHE / "frames.json", info)
    M.log(f"frames: {info}")


def run_rule(rule):
    out = M.INPUTS / f"{rule}.npz"
    if out.exists():
        M.log(f"{rule}: cached")
        return
    k = M.KOF[rule]
    G = M.grid_lean(("off", "time", "ti", "seg"))
    R = M.rows_lean()
    F = M.load_frames()
    fr, sc = F["fr"], F["scored"]
    off, tm, ti, seg = G["off"], G["time"], G["ti"], G["seg"]
    n = len(R["gi"])
    H, age, M5, Gp = (np.full(n, np.nan) for _ in range(4))
    th = np.full((n, 5), np.nan)
    anc_g = np.full(n, -1, np.int64)
    nvis = np.zeros(n, np.int64)
    files = np.unique(R["fidx"][sc])
    t0 = time.time()
    for j, i in enumerate(files):
        a, b = off[i], off[i + 1]
        q = np.flatnonzero((R["fidx"] == i) & sc)
        rows = R["grow"][q]
        I = M.file_inputs_k(tm[a:b], ti[a:b], seg[a:b], fr[a:b], rows, k)
        H[q], age[q], M5[q], Gp[q], th[q] = I["H"], I["age"], I["M5"], I["G"], I["th"]
        anc_g[q] = np.where(I["anc"] >= 0, I["anc"] + a, -1)
        nvis[q] = I["nvis"]
        if (j + 1) % 100 == 0:
            M.log(f"{rule}: {j + 1}/{len(files)} files ({time.time() - t0:.0f}s)")
    M.save_npz(out, H=H, age=age, M5=M5, G=Gp, th=th, anc_g=anc_g, nvis=nvis)
    M.log(f"{rule}: written ({time.time() - t0:.0f}s); visible on {int((nvis[sc] > 0).sum()):,} of {int(sc.sum()):,} scored rows")


def run_sets():
    G = M.grid_lean(("grow", "seg", "fast"))
    R = M.rows_lean()
    F = M.load_frames()
    sc = F["scored"]
    gi = R["gi"]
    mrb, mre = (M.run_mean10(x, G["seg"]) for x in M.raw_means(G["fast"]))
    v8 = M.valid8(G, gi)
    comp = {}
    info = {"d1_rows": int(len(gi)), "scored_rows": int(sc.sum()), "scored_valid8_rows": int((sc & v8).sum()), "per_rule": {}}
    for rule in M.RULES:
        I = M.load_inputs(rule)
        H10 = M.history_matrix(R["tstep"], I)
        A = M.anchor_scalars(gi, I["anc_g"], mrb, mre, mrb, mre)      # finiteness only (raw in place of std)
        vis = sc & (I["nvis"] > 0)
        h_ok = sc & np.isfinite(H10).all(1) & np.isfinite(I["G"])
        a_fin = np.isfinite(A[:, [0, 1, 3, 4]]).all(1)
        a3_ok = np.isfinite(A[:, [2, 5]]).all(1)
        comp[rule] = h_ok & v8 & a_fin & a3_ok
        info["per_rule"][rule] = {"visible": int(vis.sum()),
                                  "fewer_than_three_visible": int((vis & (I["nvis"] < 3)).sum()),
                                  "history_and_G_complete": int(h_ok.sum()),
                                  "anchor_means_defined": int((sc & a_fin).sum()),
                                  "complete": int(comp[rule].sum()),
                                  "dropped_by_a3_alone": int((h_ok & v8 & a_fin & ~a3_ok).sum())}
    arr = {f"complete_{r}": comp[r] for r in M.RULES}
    all_rules = np.logical_and.reduce([comp[r] for r in M.RULES])
    info["complete_under_all_six"] = int(all_rules.sum())
    for X in M.TARGETS:
        tg = np.isfinite(R[f"tg_{X}"])
        arr[f"common_{X}"] = all_rules & tg
        info[f"common_{X}"] = int(arr[f"common_{X}"].sum())
        info[f"scored_with_target_{X}"] = int((sc & tg).sum())
    M.save_npz(M.CACHE / "sets.npz", **arr)
    M.save_json(M.CACHE / "sets.json", info)
    M.log(f"sets: {info}")


if __name__ == "__main__":
    if sys.argv[1] == "frames":
        run_frames()
    elif sys.argv[1] == "rule":
        run_rule(sys.argv[2])
    elif sys.argv[1] == "sets":
        run_sets()
    else:
        raise SystemExit("usage: inputs78.py frames | rule <RULE> | sets")
