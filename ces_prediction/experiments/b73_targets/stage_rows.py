"""B.73 stage 2: the D1 rows with their causal inputs, targets and stored estimates; the rows of the usability
classifier.

    py ces_prediction/experiments/b73_targets/stage_rows.py
Output: data/.b73/rows.npz, data/.b73/clf_rows.npz, data/.b73/rows_info.json
"""
from __future__ import annotations

import common as C

import numpy as np  # noqa: E402


def d1_union():
    """D1 rows of r2-r4 (data/.b70/stage2_r{r}_f{k}.npz) in one canonical order (fidx, grow), with each re-partition's
    fold and stored estimates; the stored causal GP / hold errors of the comparison files for the B.70 check."""
    per = {}
    for r in C.REPEATS:
        parts = []
        for k in range(C.K):
            s = np.load(C.DATA / ".b70" / f"stage2_r{r}_f{k}.npz", allow_pickle=True)
            c = np.load(C.DATA / ".b70_rescore" / f"b39p_v2_cut_r{r}_f{k}" / "comparison_errors_test.npz")
            t = np.load(C.DATA / ".b70_rescore" / f"b39p_v2h_cut_r{r}_f{k}" / "comparison_errors_test.npz")
            y = s["y"].astype(np.float64)
            if len(y) != len(c["CES_TI_y_true"]) or np.max(np.abs(y - c["CES_TI_y_true"])) > 1e-2:
                raise SystemExit(f"FATAL: r{r} f{k}: stage2 and comparison rows differ")
            if not (np.array_equal(c["CES_TI_row"], t["CES_TI_row"]) and np.array_equal(c["CES_TI_shot"], t["CES_TI_shot"])):
                raise SystemExit(f"FATAL: r{r} f{k}: network and twin rows differ")
            if not np.array_equal(s["row"], c["CES_TI_row"]):
                raise SystemExit(f"FATAL: r{r} f{k}: stage2 rows are not the comparison rows")
            parts.append({"fidx": s["fidx"].astype(np.int64), "grow": s["grow"].astype(np.int64), "y": y,
                          "k": np.full(len(y), k, np.int64), "c": s["est_c"], "t": s["est_t"], "c2": s["est_c2"],
                          "t2": s["est_t2"], "fb": s["fallback_c2"].astype(bool),
                          "se_gp": c["CES_TI_se_gp_causal"].astype(np.float64),
                          "se_hold": c["CES_TI_se_persistence"].astype(np.float64),
                          "se_c": c["CES_TI_se_model"].astype(np.float64), "se_t": t["CES_TI_se_model"].astype(np.float64)})
        d = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        key = d["fidx"] * 1_000_000 + d["grow"]
        o = np.argsort(key, kind="stable")
        if len(np.unique(key)) != len(key):
            raise SystemExit(f"FATAL: r{r}: a D1 row is scored twice")
        per[r] = {kk: v[o] for kk, v in d.items()}
        per[r]["key"] = key[o]
    for r in C.REPEATS[1:]:
        if not np.array_equal(per[r]["key"], per[C.REPEATS[0]]["key"]):
            raise SystemExit(f"FATAL: the D1 rows of r{r} differ from those of r{C.REPEATS[0]}")
    return per


def main():
    out, outc = C.CACHE / "rows.npz", C.CACHE / "clf_rows.npz"
    if out.exists() and outc.exists():
        C.log("rows: cached")
        return
    G = C.grid()
    names, off, tm, ti, seg = G["names"], G["off"], G["time"], G["ti"], G["seg"]
    nF = len(names)
    fid_all = G["fidx"]
    per = d1_union()
    r0 = per[C.REPEATS[0]]
    fidx, grow, y = r0["fidx"], r0["grow"], r0["y"]
    gi = off[fidx] + grow
    n = len(y)
    if np.max(np.abs(ti[gi] - y)) > 1e-2 or not np.all(np.isfinite(ti[gi])):
        raise SystemExit("FATAL: a D1 value is not the grid value")
    C.log(f"D1 rows {n:,}, discharges {len(np.unique(fidx))}")
    # ---- base rows (hold, causal GP, age) of B.64
    base = np.load(C.DATA / ".b64" / "base.npz", allow_pickle=True)
    bkey = base["fidx"].astype(np.int64) * 1_000_000 + base["row"].astype(np.int64)
    ob = np.argsort(bkey)
    pos = np.searchsorted(bkey[ob], r0["key"])
    if np.any(pos >= len(bkey)) or not np.array_equal(bkey[ob][np.minimum(pos, len(bkey) - 1)], r0["key"]):
        raise SystemExit("FATAL: a D1 row is not a base row")
    jb = ob[pos]
    hold, gp, age = base["hold"][jb], base["gp"][jb], base["age"][jb]
    if np.max(np.abs(base["y"][jb] - y)) > 1e-3:
        raise SystemExit("FATAL: base values differ from the D1 values")
    # ---- time step (rows of 10 ms) to the previous grid row of the same file
    prev_same_file = (gi >= 1) & (fid_all[np.maximum(gi - 1, 0)] == fidx)
    tstep = np.where(prev_same_file, (tm[gi] - tm[np.maximum(gi - 1, 0)]) / 0.010, np.nan)
    # ---- CES-history scalars (per file, causal)
    thist = np.full((n, 5), np.nan)
    m5 = np.full(n, np.nan)
    hold2, age2 = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.unique(fidx):
        a, b = off[i], off[i + 1]
        q = np.flatnonzero(fidx == i)
        h_, a_, th_, m_ = C.file_history(tm[a:b], ti[a:b], seg[a:b], grow[q])
        hold2[q], age2[q], thist[q], m5[q] = h_, a_, th_, m_
    dev_h = float(np.nanmax(np.abs(hold2 - hold)))
    dev_a = float(np.nanmax(np.abs(age2 - age)))
    C.log(f"history scalars: complete share {np.isfinite(thist).all(1).mean():.4f}; own hold vs base max dev {dev_h:.2e} eV, "
          f"own age vs base max dev {dev_a:.2e} ms")
    # ---- targets: values at 10 j ms after t (j = 1..10), same block
    later = C.later_values(tm, ti, seg, gi, fid_all, fidx)
    tg = C.targets_from_later(y, later)
    # ---- strata predictors and units
    bes_level = G["fast"][gi, :C.NB].astype(np.float64).mean(1)
    shot_f = G["shot"]
    shot = shot_f[fidx]
    blk = np.full(n, -1, np.int64)
    for b_, (_, lo, hi) in enumerate(C.BLOCKS):
        blk[(shot >= lo) & (shot <= hi)] = b_
    clus_f, _ = C.clusters_of_files(names)
    arr = {"fidx": fidx, "grow": grow, "gi": gi, "y": y, "shot": shot, "hold": hold, "gp": gp, "age": age,
           "tstep": tstep, "thist": thist, "m5": m5, "bes_level": bes_level, "blk": blk, "clus": clus_f[fidx],
           "later": later, **{f"tg_{t}": v for t, v in tg.items()}}
    for r in C.REPEATS:
        p = per[r]
        dv = float(np.max(np.abs(p["y"] - y)))
        C.log(f"r{r}: max value difference to r{C.REPEATS[0]} {dv:.2e} eV")
        if dv > 1e-2:
            raise SystemExit(f"FATAL: r{r}: values differ")
        for kk in ("k", "c", "t", "c2", "t2", "fb", "se_gp", "se_hold", "se_c", "se_t"):
            arr[f"r{r}_{kk}"] = p[kk]
    C.save_npz(out, **arr)
    # ---- usability classifier rows: grid rows (not the first or last row of a file) with >= 1 measured value among
    # the ten grid rows before (same block)
    meas = np.isfinite(ti)
    N = len(tm)
    first = np.zeros(N, bool)
    first[off[:-1]] = True
    last = np.zeros(N, bool)
    last[off[1:] - 1] = True
    n10 = np.zeros(N, np.int64)
    for d in range(1, 11):
        src = np.arange(N) - d
        srcc = np.maximum(src, 0)
        n10 += (src >= 0) & (seg[srcc] == seg) & meas[srcc]
    prev_has = np.zeros(N, bool)
    prev_has[1:] = meas[:-1] & (seg[:-1] == seg[1:])
    idx_meas = np.where(meas, np.arange(N), -1)
    lastm = np.maximum.accumulate(idx_meas)                       # last measured index at or before g
    lastb = np.concatenate(([-1], lastm[:-1]))                    # strictly before g
    okage = (lastb >= 0) & (seg[np.maximum(lastb, 0)] == seg)
    agec = np.where(okage, (tm - tm[np.maximum(lastb, 0)]) * 1000.0, np.nan)
    ts_all = np.full(N, np.nan)
    same = fid_all[1:] == fid_all[:-1]
    ts_all[1:][same] = (np.diff(tm) / 0.010)[same]
    sel = np.flatnonzero(~first & ~last & (n10 >= 1))
    C.save_npz(outc, gidx=sel, label=(~meas[sel]).astype(np.int8), tstep=ts_all[sel], age=agec[sel],
               n10=n10[sel], prev_has=prev_has[sel].astype(np.int8), fidx=fid_all[sel], shot=shot_f[fid_all[sel]],
               clus=clus_f[fid_all[sel]])
    dts = np.diff(tm)[same]
    info = {"d1_rows": int(n), "d1_discharges": int(len(np.unique(fidx))), "files": int(nF),
            "target_rows": {t: int(np.isfinite(v).sum()) for t, v in tg.items()},
            "history_complete_share": float(np.isfinite(thist).all(1).mean()),
            "own_hold_vs_base_max_dev_eV": dev_h, "own_age_vs_base_max_dev_ms": dev_a,
            "clf_rows": int(len(sel)), "clf_positive_share": float((~meas[sel]).mean()),
            "blocks_rows": {C.BLOCKS[b][0]: int((blk == b).sum()) for b in range(3)}, "rows_outside_blocks": int((blk < 0).sum()),
            "grid_time_steps_ms": {"share_within_3ms_of_10ms_multiple": float(np.mean(np.abs(dts / 0.01 - np.rint(dts / 0.01)) * 10 < 3)),
                                   "share_10ms": float(np.mean(np.abs(dts - 0.01) < 0.003))}}
    C.save_json(C.CACHE / "rows_info.json", info)
    C.log(f"rows written: {info}")


if __name__ == "__main__":
    main()
