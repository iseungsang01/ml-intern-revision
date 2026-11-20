"""B.75 stage 1: the estimator inputs of every D1 row under each availability rule (Addendum 1 sections 2-3), and the
row sets (section 4).

    py ces_prediction/experiments/b75_availability/stage_inputs.py rule D650      # one rule (about 150-300 s)
    py ces_prediction/experiments/b75_availability/stage_inputs.py sets           # after all seven rules
Output: data/.b75/inputs/{rule}.npz (arrays over the D1 rows in rows.npz order), data/.b75/sets.npz, data/.b75/sets.json
"""
from __future__ import annotations

import sys
import time

import common75 as K

import numpy as np  # noqa: E402

C = K.C


def run_rule(rule):
    out = K.INPUTS / f"{rule}.npz"
    if out.exists():
        K.log(f"{rule}: cached")
        return
    G = K.grid_lean()
    R = K.rows_lean()
    off, tm, ti, seg = G["off"], G["time"], G["ti"], G["seg"]
    n = len(R["gi"])
    H, age, M5, Gp, lim, lim_age = (np.full(n, np.nan) for _ in range(6))
    th = np.full((n, 5), np.nan)
    anc_g = np.full(n, -1, np.int64)
    jS_g = np.full(n, -1, np.int64)
    nvis = np.zeros(n, np.int64)
    files = np.unique(R["fidx"])
    t0 = time.time()
    for k, i in enumerate(files):
        a, b = off[i], off[i + 1]
        q = np.flatnonzero(R["fidx"] == i)
        rows = R["grow"][q]
        I = K.file_inputs(tm[a:b], ti[a:b], seg[a:b], rows, rule)
        H[q], age[q], M5[q], Gp[q], th[q] = I["H"], I["age"], I["M5"], I["G"], I["th"]
        anc_g[q] = np.where(I["anc"] >= 0, I["anc"] + a, -1)
        jS_g[q] = np.where(I["jS"] >= 0, I["jS"] + a, -1)
        nvis[q] = I["nvis"]
        lim[q] = I["lim"]
        lim_age[q] = tm[a:b][rows] - I["lim"]
        if (k + 1) % 50 == 0:
            K.log(f"{rule}: {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)")
    K.save_npz(out, H=H, age=age, M5=M5, G=Gp, th=th, anc_g=anc_g, jS_g=jS_g, nvis=nvis, lim=lim, lim_age=lim_age)
    K.log(f"{rule}: written ({time.time() - t0:.0f}s); visible on {int((nvis > 0).sum()):,} of {n:,} rows")


def run_sets():
    G = K.grid_lean(("grow", "seg", "fast"))
    R = K.rows_lean()
    gi = R["gi"]
    mrb, mre = (K.run_mean10(x, G["seg"]) for x in K.raw_means(G["fast"]))
    v8 = K.valid8(G, gi)
    comp, info = {}, {"d1_rows": int(len(gi)), "valid8_rows": int(v8.sum()), "per_rule": {}}
    for rule in K.RULES:
        I = K.load_inputs(rule)
        H10 = K.history_matrix(R["tstep"], I)
        A = K.anchor_scalars(gi, I["anc_g"], mrb, mre, mrb, mre)      # finiteness only (raw in place of std)
        h_ok = np.isfinite(H10).all(1) & np.isfinite(I["G"])
        a_fin = np.isfinite(A[:, [0, 1, 3, 4]]).all(1)
        a3_ok = np.isfinite(A[:, [2, 5]]).all(1)
        comp[rule] = h_ok & v8 & a_fin & a3_ok
        info["per_rule"][rule] = {"visible": int((I["nvis"] > 0).sum()), "history_and_G_complete": int(h_ok.sum()),
                                  "anchor_means_defined": int(a_fin.sum()),
                                  "complete": int(comp[rule].sum()),
                                  "dropped_by_a3_alone": int((h_ok & v8 & a_fin & ~a3_ok).sum())}
    arr = {f"complete_{r}": comp[r] for r in K.RULES}
    all_fixed = np.logical_and.reduce([comp[r] for r in K.FIXED])
    s_ok = comp["S"] & comp["D10"]
    info["complete_S_not_D10"] = int((comp["S"] & ~comp["D10"]).sum())
    info["no_visible_value_under_S"] = int((K.load_inputs("S")["nvis"] == 0).sum())
    for X in K.TARGETS:
        tg = np.isfinite(R[f"tg_{X}"])
        arr[f"common_{X}"] = all_fixed & tg
        arr[f"S_{X}"] = s_ok & tg
        info[f"common_{X}"] = int(arr[f"common_{X}"].sum())
        info[f"S_{X}"] = int(arr[f"S_{X}"].sum())
        info[f"target_{X}"] = int(tg.sum())
    K.save_npz(K.CACHE / "sets.npz", **arr)
    K.save_json(K.CACHE / "sets.json", info)
    K.log(f"sets: {info}")


if __name__ == "__main__":
    if sys.argv[1] == "rule":
        run_rule(sys.argv[2])
    elif sys.argv[1] == "sets":
        run_sets()
    else:
        raise SystemExit("usage: stage_inputs.py rule <RULE> | sets")
