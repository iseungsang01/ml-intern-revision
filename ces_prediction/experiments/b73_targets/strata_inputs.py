"""B.73, the session lead's Addendum 1 (A1-A4) as implemented by Addendum 2 section 12: strata defined on the inputs and
the time structure only, the rotation from the same fit, rows exposed without beam emission.

    py ces_prediction/experiments/b73_targets/strata_inputs.py
Output: data/.b73/strata_inputs.npz (per D1 row: r_B, r_B class, absent-frame class, beam-off flag),
        data/.b73/a3_a4.json (the A3 table and slopes, the A4 counts)
"""
from __future__ import annotations

import common as C

import numpy as np  # noqa: E402

RB_EDGES = (0.9, 0.97, 1.03)                                  # A1 classes
RB_LAB = ("below 0.9", "0.9-0.97", "0.97-1.03", "above 1.03", "r_B undefined")
A3_EDGES = (0.4, 0.6, 0.8, 0.9, 0.97, 1.03, 1.1, 1.2, 1.4, np.inf)
AB_LAB = ("absent frames earlier and later", "only later (before the first)", "only earlier (after the last)",
          "file without absent frames")


def grid_arrays():
    G = C.grid()
    N = len(G["time"])
    off, tm, seg, fid = G["off"], G["time"], G["seg"], G["fidx"]
    bes = G["fast"][:, :C.NB].astype(np.float64)
    ecei = G["fast"][:, C.NB:].astype(np.float64)
    first = np.zeros(N, bool)
    first[off[:-1]] = True
    last = np.zeros(N, bool)
    last[off[1:] - 1] = True
    step = np.full(N, np.nan)                          # step into row g from g-1 (same block)
    same = (fid[1:] == fid[:-1]) & (seg[1:] == seg[:-1])
    step[1:][same] = np.diff(tm)[same]
    reg_step = np.abs(step - 0.010) < 0.003
    regular = np.zeros(N, bool)                        # rows t-10 .. t consecutive at 10 ms, same block
    ok = np.ones(N, bool)
    for j in range(10):
        src = np.arange(N) - j
        ok &= (src >= 1) & reg_step[np.maximum(src, 0)]
    regular[:] = ok & ~first & ~last
    return G, N, tm, seg, fid, bes, ecei, step, regular, first, last


def prev10(x, N):
    """N x 10 matrix of x at rows t-10 .. t-1 (NaN before the array start)."""
    M = np.full((N, 10), np.nan)
    for j in range(1, 11):
        M[j:, 10 - j] = x[:-j]
    return M


def main():
    out = C.CACHE / "strata_inputs.npz"
    G, N, tm, seg, fid, bes, ecei, step, regular, first, last = grid_arrays()
    R = C.rows()
    gi = R["gi"]
    bmean, emean = bes.mean(1), ecei.mean(1)
    # ---- r_B (A1, A3)
    Pb = prev10(bmean, N)
    medb = np.median(Pb, 1)
    rb = np.where(regular & (medb > 0.02), bmean / np.where(medb > 0.02, medb, 1.0), np.nan)
    cls = np.full(N, 4, np.int64)
    fin = np.isfinite(rb)
    cls[fin] = np.searchsorted(np.array(RB_EDGES), rb[fin], "right")
    # ---- absent frames (A2): steps of 15-45 ms inside a block; class of each D1 row within its file
    absent = (step > 0.015) & (step < 0.045)          # absent frame between rows g-1 and g
    acls = np.full(len(gi), -1, np.int64)
    for i in np.unique(R["fidx"]):
        a, b = G["off"][i], G["off"][i + 1]
        pos = np.flatnonzero(absent[a:b]) + a
        q = np.flatnonzero(R["fidx"] == i)
        if len(pos) == 0:
            acls[q] = 3
            continue
        g = gi[q]
        earlier = pos.min() <= g
        later = pos.max() > g
        acls[q] = np.where(earlier & later, 0, np.where(later, 1, 2))
    # ---- beam-off-like rows (A4)
    beamoff = np.zeros(N, bool)
    okp = np.ones(N, bool)
    for j in range(1, 11):
        src = np.arange(N) - j
        okp &= (src >= 0) & (seg[np.maximum(src, 0)] == seg) & (fid[np.maximum(src, 0)] == fid)
    idx = np.flatnonzero(okp & ~first & ~last)
    for a in range(0, len(idx), 50_000):
        ii = idx[a:a + 50_000]
        P = np.stack([bes[ii - j] for j in range(1, 11)], 1)     # n x 10 x 9
        med = np.median(P, 1)
        good = np.all(med > 0, 1)
        beamoff[ii] = good & np.all(bes[ii] < 0.4 * np.where(med > 0, med, np.inf), 1)
    ti = G["ti"]
    d1 = np.zeros(N, bool)
    d1[gi] = True
    a4 = {"rows_exposed_without_beam": int(beamoff.sum()), "with_T_i_value": int((beamoff & np.isfinite(ti)).sum()),
          "D1_rows": int((beamoff & d1).sum()),
          "discharges": int(len(np.unique(fid[beamoff]))),
          "rows_examined": int((okp & ~first & ~last).sum())}
    # ---- A3: ratios to the ten rows before
    vt_raw = C.load_npz(C.CACHE / "vt_raw.npz")["vt"]
    vprev = np.concatenate(([np.nan], vt_raw[:-1]))
    samef = np.concatenate(([False], fid[1:] == fid[:-1]))
    vprev[~samef] = np.nan
    fresh = np.isfinite(vt_raw) & (vt_raw != vprev) & (np.abs(vt_raw) <= 1000.0)
    vf = np.where(fresh, vt_raw, np.nan)
    rep_den = np.isfinite(vt_raw) & np.isfinite(vprev)
    a3 = {"rotation_values_equal_to_previous_row_share": float(np.mean(vt_raw[rep_den] == vprev[rep_den])),
          "rotation_values_delivered": int(np.isfinite(vt_raw[~first & ~last]).sum())}
    rows3 = np.flatnonzero(np.isfinite(rb))
    Pt = prev10(ti, N)[rows3]
    Pv = prev10(vf, N)[rows3]
    Pe = prev10(emean, N)[rows3]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mt = np.nanmedian(Pt, 1)
        mv = np.nanmedian(Pv, 1)
    nt, nv = np.isfinite(Pt).sum(1), np.isfinite(Pv).sum(1)
    me = np.median(Pe, 1)
    tr = np.where((nt >= 6) & np.isfinite(ti[rows3]), ti[rows3] / mt, np.nan)
    vr = np.where((nv >= 6) & np.isfinite(vf[rows3]) & (mv > 20), vf[rows3] / np.where(mv > 20, mv, 1.0), np.nan)
    er = np.where(me > 0.02, emean[rows3] / np.where(me > 0.02, me, 1.0), np.nan)
    br = rb[rows3]
    tab = []
    for lo, hi in zip(A3_EDGES[:-1], A3_EDGES[1:]):
        m = (br >= lo) & (br < hi)
        t_, v_, e_ = tr[m & np.isfinite(tr)], vr[m & np.isfinite(vr)], er[m & np.isfinite(er)]
        tab.append({"class": f"{lo:.2f}-{hi:.2f}" if np.isfinite(hi) else f"above {lo:.2f}",
                    "rows_T_i": int(len(t_)), "median_T_i_ratio": float(np.median(t_)) if len(t_) else None,
                    "rows_V_rot": int(len(v_)), "median_V_rot_ratio": float(np.median(v_)) if len(v_) else None,
                    "rows_ECEI": int(len(e_)), "median_ECEI_ratio": float(np.median(e_)) if len(e_) else None})
    a3["rows_regular_with_r_B"] = int(len(rows3))
    a3["below_0.40"] = int(np.sum(br < 0.4))
    a3["table"] = tab
    both = np.isfinite(tr) & np.isfinite(vr) & (tr > 0) & (vr > 0)
    shot = G["shot"][fid[rows3]]
    a3["slopes"] = {}
    for tag, m, clip in (("C18 form: r_B 0.6-1.6, log ratios clipped at 1.5", both & (br > 0.6) & (br < 1.6), 1.5),
                         ("all rows with both positive ratios, no clip", both, None)):
        x = np.log(br[m])
        yT, yV = np.log(tr[m]), np.log(vr[m])
        if clip:
            yT, yV = np.clip(yT, -clip, clip), np.clip(yV, -clip, clip)
        u, inv = np.unique(shot[m], return_inverse=True)
        nu = len(u)
        S = {k: np.bincount(inv, w, nu) for k, w in (("n", np.ones_like(x)), ("x", x), ("xx", x * x), ("yT", yT),
                                                       ("xyT", x * yT), ("yV", yV), ("xyV", x * yV))}

        def slope(s, yk):
            n_, sx, sxx = s["n"], s["x"], s["xx"]
            return (s["xy" + yk] - sx * s["y" + yk] / n_) / (sxx - sx * sx / n_)
        pt = {k: float(slope({kk: v.sum() for kk, v in S.items()}, k)) for k in ("T", "V")}
        rng = C.rng_for("A3|slopes" + ("" if clip else "|noclip"))
        W = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=C.N_BOOT).astype(np.float64)
        Sd = {k: W @ v for k, v in S.items()}
        dr = {k: slope(Sd, k) for k in ("T", "V")}
        a3["slopes"][tag] = {"rows": int(m.sum()), "discharges": int(nu),
                             "T_i": {"point": pt["T"], "discharge": [float(np.quantile(dr["T"], .025)), float(np.quantile(dr["T"], .975))]},
                             "V_rot": {"point": pt["V"], "discharge": [float(np.quantile(dr["V"], .025)), float(np.quantile(dr["V"], .975))]}}
    C.save_npz(out, rb=rb[gi], rb_class=cls[gi], absent_class=acls, beamoff=beamoff[gi])
    C.save_json(C.CACHE / "a3_a4.json", {"A3": a3, "A4": a4,
                                         "rb_class_share_d1": {RB_LAB[k]: float(np.mean(cls[gi] == k)) for k in range(5)},
                                         "absent_class_share_d1": {AB_LAB[k]: float(np.mean(acls == k)) for k in range(4)}})
    C.log(f"strata written; A4 {a4}; r_B classes {[float(np.mean(cls[gi] == k)) for k in range(5)]}")


if __name__ == "__main__":
    main()
