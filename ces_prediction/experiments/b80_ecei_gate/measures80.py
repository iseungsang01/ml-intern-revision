"""B.80 stage 1 (registration section 3, Addendum 1 section 2): the input-only measures per discharge.

    py ces_prediction/experiments/b80_ecei_gate/measures80.py
Writes data/.b80/measures.npz (per file of the grid) and data/.b80/measures.json (distribution over discharges).
Reads only the grid's time stamps, block numbers, row numbers and the four ECEI columns; no CES value, no estimate.
"""
from __future__ import annotations

import common80 as M

import numpy as np  # noqa: E402

C, K75 = M.C, M.K75
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def coherence(E):
    """Mean of the six pairwise Pearson correlations of the four columns of E (rows x 4, float64), the six values, and a
    status: 'ok' or 'constant channel' (a column with zero variance: its correlations, and so C_E, are undefined)."""
    sd = E.std(0)
    if np.any(sd == 0):
        return np.nan, np.full(6, np.nan), "constant channel"
    c = np.corrcoef(E.T)
    six = np.array([c[a, b] for a, b in PAIRS])
    if not np.all(np.isfinite(six)):
        return np.nan, six, "non-finite correlation"
    return float(six.mean()), six, "ok"


def main():
    G = K75.load_keys(C.CACHE / "grid.npz", ("shot", "off", "time", "seg", "fidx", "grow", "fast"))
    off, tm, seg, fid, grow = G["off"], G["time"], G["seg"], G["fidx"], G["grow"]
    fast = G["fast"]
    if not np.all(np.isfinite(fast[:, C.NB:C.NB + C.NE])):
        raise SystemExit("FATAL: a non-finite ECEI value in the grid (Addendum 1 section 2 assumes none)")
    nf = len(G["shot"])
    last = np.concatenate((fid[1:] != fid[:-1], [True]))
    okg = (grow > 0) & ~last                                   # grid rows: a file's first and last row excluded
    ece_mean = K75.raw_means(fast)[1]                          # raw ECEI array mean (sequential sum of the four / 4)

    n_grid = np.zeros(nf, np.int64)
    ce, ce_e = np.full(nf, np.nan), np.full(nf, np.nan)
    six, six_e = np.full((nf, 6), np.nan), np.full((nf, 6), np.nan)
    st, st_e = [], []
    n_early = np.zeros(nf, np.int64)
    for i in range(nf):
        a, b = off[i], off[i + 1]
        r = np.flatnonzero(okg[a:b]) + a                       # grid rows of the file, in time order
        n_grid[i] = len(r)
        if len(r) < M.MIN_ROWS:
            st.append("fewer than 50 grid rows")
            st_e.append("fewer than 50 grid rows")
            continue
        E = fast[r, C.NB:C.NB + C.NE].astype(np.float64)
        ce[i], six[i], s = coherence(E)
        st.append(s)
        re = r[:M.EARLY_ROWS]
        n_early[i] = len(re)
        ce_e[i], six_e[i], s = coherence(E[:M.EARLY_ROWS])
        st_e.append(s)

    # N_E: absent steps (j, j + 1) inside a block, with rows j - 4 .. j grid rows of the same block
    dt = np.diff(tm)
    ab = np.flatnonzero((seg[1:] == seg[:-1]) & (dt > K75.AB_LO) & (dt < K75.AB_HI))
    steps = [[] for _ in range(nf)]
    n_steps_all, n_steps_pre, n_dropped_ref = np.zeros(nf, np.int64), np.zeros(nf, np.int64), np.zeros(nf, np.int64)
    for j in ab:
        f = int(fid[j])
        n_steps_all[f] += 1
        if not okg[j + 1]:
            raise SystemExit("FATAL: an absent step ends on a row that is not a grid row")
        if j - (M.PRE_ROWS - 1) < 0:
            continue
        pre = np.arange(j - (M.PRE_ROWS - 1), j + 1)
        if not (np.all(seg[pre] == seg[j]) and np.all(okg[pre])):
            continue
        n_steps_pre[f] += 1
        ref = np.median(ece_mean[pre])
        if not ref > 0:
            n_dropped_ref[f] += 1
            continue
        steps[f].append(ece_mean[j + 1] / ref)
    ne = np.full(nf, np.nan)
    n_used = np.array([len(s) for s in steps], np.int64)
    for f in range(nf):
        if n_grid[f] >= M.MIN_ROWS and n_used[f] >= M.MIN_STEPS:
            ne[f] = float(np.median(steps[f]))

    shot = G["shot"]
    blk = np.full(nf, -1, np.int64)
    for k, (_, lo, hi) in enumerate(C.BLOCKS):
        blk[(shot >= lo) & (shot <= hi)] = k
    status_codes = {"ok": 0, "fewer than 50 grid rows": 1, "constant channel": 2, "non-finite correlation": 3}
    M.save_npz(M.CACHE / "measures.npz", shot=shot, blk=blk, n_grid=n_grid, ce=ce, ce_six=six, ce_status=np.array([status_codes[s] for s in st]),
               ce_early=ce_e, ce_early_six=six_e, ce_early_status=np.array([status_codes[s] for s in st_e]), n_early=n_early,
               ne=ne, ne_steps_used=n_used, ne_steps_with_five_rows=n_steps_pre, ne_steps_all=n_steps_all,
               ne_steps_ref_not_positive=n_dropped_ref, cls=M.class_of(ce), cls_early=M.class_of(ce_e))
    M.save_json(M.CACHE / "measures_status_codes.json", status_codes)
    M.log(f"measures written: {nf} files")


def _q(x, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = x[np.isfinite(x)]
    return {str(p): (float(np.quantile(x, p)) if len(x) else None) for p in ps}


def distribution(meas, d1_fidx):
    """Distribution of C_E, C_E early and N_E over discharges, per block of shots (inputs only).  Population: the D1
    discharges (file indices d1_fidx); the same over all grid files beside it."""
    codes = {0: "ok", 1: "fewer than 50 grid rows", 2: "constant channel", 3: "non-finite correlation"}
    nf = len(meas["shot"])
    d1 = np.zeros(nf, bool)
    d1[d1_fidx] = True
    out = {}
    pops = {"D1 discharges": d1, "all grid files": np.ones(nf, bool)}
    for pname, pm in pops.items():
        o = {}
        for bname, bm in [("all blocks", np.ones(nf, bool))] + [(f"block {C.BLOCKS[k][0]}", meas["blk"] == k) for k in range(3)] + \
                [("no block", meas["blk"] == -1)]:
            m = pm & bm
            if not m.any():
                continue
            ce, cls, ce_e, cls_e, ne = meas["ce"][m], meas["cls"][m], meas["ce_early"][m], meas["cls_early"][m], meas["ne"][m]
            both = (cls >= 0) & (cls_e >= 0)
            same_rows = meas["n_early"][m] == meas["n_grid"][m]
            o[bname] = {
                "discharges": int(m.sum()),
                "with C_E": int(np.isfinite(ce).sum()),
                "without C_E, by reason": {codes[k]: int((meas["ce_status"][m] == k).sum()) for k in (1, 2, 3)},
                "per class": {M.CLASS_NAMES[k]: int((cls == k).sum()) for k in range(3)},
                "share low among those with C_E": float((cls == 0).sum() / max(1, (cls >= 0).sum())),
                "C_E quantiles": _q(ce),
                "C_E early: defined": int(np.isfinite(ce_e).sum()),
                "C_E early: discharges with fewer than 200 grid rows (early = whole)": int((same_rows & np.isfinite(ce)).sum()),
                "C_E early: same class as C_E (share of discharges with both)": (float((cls[both] == cls_e[both]).mean()) if both.any() else None),
                "C_E early: discharges with both": int(both.sum()),
                "C_E early: same class, discharges with at least 200 grid rows only": (
                    float((cls[both & ~same_rows] == cls_e[both & ~same_rows]).mean()) if (both & ~same_rows).any() else None),
                "C_E early quantiles": _q(ce_e),
                "N_E: defined": int(np.isfinite(ne).sum()),
                "N_E quantiles": _q(ne),
                "N_E: absent steps used (sum)": int(meas["ne_steps_used"][m].sum()),
                "N_E: steps with five rows before, reference not above zero (sum)": int(meas["ne_steps_ref_not_positive"][m].sum()),
            }
        out[pname] = o
    return out


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        meas = M.load_npz(M.CACHE / "measures.npz")
        R = M.rows()
        print(json.dumps(distribution(meas, np.unique(R["fidx"])), indent=1))
    else:
        main()
