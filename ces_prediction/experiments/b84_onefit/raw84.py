"""B.84 stage 1: the raw history windows of every D1 row under each rule (Addendum 1 section 2).

    py ces_prediction/experiments/b84_onefit/raw84.py            # all eight rules
Output: data/.b84/raw/{rule}.npz: V (n x 10, values newest first), A (n x 10, ages in ms), anc_g (global anchor row)
over the D1 rows in rows.npz order.  Stops if the newest visible value is not B.75's (B.78's) anchor row and H on every
row of the rule's set.
"""
from __future__ import annotations

import time

import common84 as M

import numpy as np  # noqa: E402


def run_rule(rule, G, R, fr_all, sets75, sets78):
    out = M.raw_path(rule)
    if out.exists():
        M.log(f"{rule}: cached")
        return
    t0 = time.time()
    off, tm, ti, seg = G["off"], G["time"], G["ti"], G["seg"]
    n = len(R["gi"])
    V = np.full((n, 10), np.nan)
    A = np.full((n, 10), np.nan)
    anc_g = np.full(n, -1, np.int64)
    for i in np.unique(R["fidx"]):
        a, b = off[i], off[i + 1]
        q = np.flatnonzero(R["fidx"] == i)
        rows = R["grow"][q]
        v, ag, anc = M.raw_hist_file(tm[a:b], ti[a:b], seg[a:b], rows, rule, fr_all[a:b] if rule == "K10" else None)
        V[q], A[q] = v, ag
        anc_g[q] = np.where(anc >= 0, anc + a, -1)
    I = M.load_inputs(rule)
    sm = M.set_of(rule, sets75, sets78)
    bad_anc = int((anc_g[sm] != I["anc_g"][sm]).sum())
    bad_h = int((V[sm, 0] != I["H"][sm]).sum())
    bad_age = int((A[sm, 0] != I["age"][sm]).sum())
    if bad_anc or bad_h or bad_age:
        raise SystemExit(f"FATAL: {rule}: anchor rows differ on {bad_anc}, H on {bad_h}, age on {bad_age} rows of the set")
    M.save_npz(out, V=V, A=A, anc_g=anc_g)
    M.log(f"{rule}: written ({time.time() - t0:.0f}s); set rows {int(sm.sum()):,}; anchor, H and age equal B.75/B.78 on all")


def main():
    G = M.K75.grid_lean(("off", "time", "ti", "seg"))
    R = M.K75.rows_lean()
    fr_all = M.M78.load_frames()["fr"]
    sets75 = M.load_npz(M.K75.CACHE / "sets.npz")
    sets78 = M.load_npz(M.M78.CACHE / "sets.npz")
    for rule in M.ALL_RULES:
        run_rule(rule, G, R, fr_all, sets75, sets78)


if __name__ == "__main__":
    main()
