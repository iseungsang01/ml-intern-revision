"""B.86 stage 1: candidate rows of the five bands with their inputs under D5 and D100 (Addendum 1 sections 2-4), then the
per-band row tables, the row sets and the design units.

    py ces_prediction/experiments/b86_mast_radius/stage_rows86.py build --part K/N   # per shot, cached
    py ces_prediction/experiments/b86_mast_radius/stage_rows86.py concat             # data/.b86/rows_{band}.npz
    py ces_prediction/experiments/b86_mast_radius/stage_rows86.py sets               # sets.npz-like masks in the band tables, units_{band}.json, sets.json
Progress lines carry counts only.
"""
from __future__ import annotations

import argparse
import json
import time

import common86 as C

import numpy as np  # noqa: E402


def build(part):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    k, n = (int(x) for x in part.split("/"))
    cat = C.K.meta_tables()
    shots = C.K.listed_shots()[k::n]
    t0 = time.time()
    done = 0
    for i, s in enumerate(shots):
        path = C.SHOTS / f"{s}.npz"
        if path.exists():
            continue
        out = C.build_shot86(s, cat)
        C.save_npz(path, **out)
        done += 1
        if (i + 1) % 50 == 0:
            C.log(f"part {k}/{n}: {i + 1}/{len(shots)} shots ({time.time() - t0:.0f}s)")
    C.log(f"part {k}/{n}: done, {done} shots built ({time.time() - t0:.0f}s)")


def concat():
    """Per band: band E keeps every candidate row (check of section 6 compares them with B.76's rows); B2-B5 keep the
    candidate rows complete under D5 and under D100 (the only rows any fit or score uses)."""
    cat = C.K.meta_tables()
    shots = C.K.listed_shots()
    status, meta = {}, {}
    counts = np.zeros((len(C.BANDS), 4), np.int64)
    shots_cnt = np.zeros((len(C.BANDS), 4), np.int64)
    parts = {b: [] for b in C.BAND_NAMES}
    ncand = {b: 0 for b in C.BAND_NAMES}
    ncand_shots = {b: 0 for b in C.BAND_NAMES}
    for s in shots:
        z = np.load(C.SHOTS / f"{s}.npz", allow_pickle=False)
        st = str(z["status"])
        status[st] = status.get(st, 0) + 1
        if st != "ok":
            continue
        cnt = z["counts"]
        counts += cnt
        shots_cnt += (cnt > 0)
        meta[s] = {"L": int(z["L"]), "t_a": float(z["t_a"]), "ts_sys": str(z["ts_sys"]), "n_chords": int(z["n_chords"]),
                   "session": C.K.session_of(cat, s)}
        if not len(z["j"]):
            continue
        Z = {k: z[k] for k in C.ROW_KEYS + ["band"]}
        comp = C.complete_mask(Z, "D5") & C.complete_mask(Z, "D100")
        for k, b in enumerate(C.BAND_NAMES):
            inb = Z["band"] == k
            ncand[b] += int(inb.sum())
            ncand_shots[b] += int(inb.any())
            keep = inb if b == "E" else inb & comp
            if keep.any():
                parts[b].append({kk: v[keep] for kk, v in Z.items()})
    info = {"listed_shots": len(shots), "shot_status": status, "shots_ok": len(meta), "bands": {}}
    for k, b in enumerate(C.BAND_NAMES):
        tab = {kk: np.concatenate([q[kk] for q in parts[b]]) for kk in parts[b][0]}
        C.save_npz(C.band_path(b), **tab)
        info["bands"][b] = {"rows_good_all_bands": int(counts[k, 0]), "rows_good_in_band": int(counts[k, 1]),
                            "rows_band_with_thomson_t": int(counts[k, 2]), "rows_candidate": int(counts[k, 3]),
                            "shots_band": int(shots_cnt[k, 1]), "shots_band_ts": int(shots_cnt[k, 2]),
                            "shots_candidate": int(ncand_shots[b]), "rows_stored": int(len(tab["shot"]))}
        if ncand[b] != counts[k, 3]:
            raise SystemExit(f"FATAL: band {b}: candidate counts disagree")
        C.log(f"concat {b}: candidates {ncand[b]}, stored {len(tab['shot'])}")
        del tab
        parts[b] = None
    C.save_json(C.CACHE / "shots.json", {"info": info, "meta": {str(k): v for k, v in meta.items()}})


def make_units(shots, sess):
    """Re-partition 1 of B.76 (fold = default_rng(1).permutation(n) % 5 over the band's design shots in increasing
    order) and the rolling-origin units M3_q1..q4 (B.76 Addendum 1 section 6), per band."""
    ns = len(shots)
    out = []
    fold = np.random.default_rng(C.M1_SEED).permutation(ns) % C.NF
    for k in range(C.NF):
        out.append({"name": f"M1_s{C.M1_SEED}_k{k}", "design": "M1", "sub": C.M1_SEED, "fit": shots[fold != k].tolist(),
                    "test": shots[fold == k].tolist()})
    fifth = np.zeros(ns, np.int64)
    for q, part in enumerate(np.array_split(np.arange(ns), C.NF)):
        fifth[part] = q
    for q in range(1, C.NF):
        out.append({"name": f"M3_q{q}", "design": "M3", "sub": q, "fit": shots[fifth < q].tolist(),
                    "test": shots[fifth == q].tolist()})
    return {"units": out, "table": {"shots": shots.tolist(), "session": sess.tolist(), "fifth": fifth.tolist()}}


def sets():
    meta = json.loads((C.CACHE / "shots.json").read_text(encoding="utf-8"))["meta"]
    info = {"bands": {}}
    for b in C.BAND_NAMES:
        R = C.load_band(b)
        sh = R["shot"]
        comp = {r: C.complete_mask(R, r) for r in C.RULES}
        both = comp["D5"] & comp["D100"]
        bi = {"rows_stored": int(len(sh)), "per_rule": {r: {"complete": int(comp[r].sum()),
                                                            "complete_shots": int(len(np.unique(sh[comp[r]])))}
                                                        for r in C.RULES},
              "complete_both": int(both.sum()), "complete_both_shots": int(len(np.unique(sh[both])))}
        masks = {}
        for X in C.TARGETS:
            tg = np.isfinite(R["y"] if X == "F" else R["L5"])
            masks[f"common_{X}"] = both & tg
            m = masks[f"common_{X}"]
            ss = np.array([meta[str(int(s))]["session"] for s in np.unique(sh[m])])
            bi[f"common_{X}"] = {"rows": int(m.sum()), "shots": int(len(np.unique(sh[m]))),
                                 "sessions": int(len(np.unique(ss))),
                                 "median_rho": float(np.median(R["rho"][m])) if m.any() else None,
                                 "median_Ti_eV": float(np.median(R["y"][m])) if m.any() else None,
                                 "median_dband_cm": float(100 * np.median(R["dband"][m])) if m.any() else None,
                                 "share_rho_below_0": float(np.mean(R["rho"][m] < 0)) if m.any() else None}
        if not np.all(masks["common_L5"] <= masks["common_F"]):
            raise SystemExit("FATAL: the L5 common set is not inside the F common set")
        R.update(masks)
        R["complete_D5"], R["complete_D100"] = comp["D5"], comp["D100"]
        C.save_npz(C.band_path(b), **R)
        shots = np.unique(sh[masks["common_F"]]).astype(int)
        sess = np.array([meta[str(s)]["session"] for s in shots])
        us = make_units(shots, sess)
        C.save_json(C.CACHE / f"units_{b}.json", us)
        bi["design_shots"] = int(len(shots))
        bi["design_sessions"] = int(len(np.unique(sess)))
        bi["statements"] = bool(bi["common_L5"]["rows"] >= C.MIN_ROWS)
        info["bands"][b] = bi
        C.log(f"sets {b}: complete both {bi['complete_both']}, common F {bi['common_F']['rows']}, "
              f"common L5 {bi['common_L5']['rows']} rows / {bi['common_L5']['shots']} shots / {bi['common_L5']['sessions']} sessions")
        del R
    C.save_json(C.CACHE / "sets.json", info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    {"build": lambda: build(a.part), "concat": concat, "sets": sets}[a.what]()
