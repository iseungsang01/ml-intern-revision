"""B.76 stage 1: candidate rows with their inputs under every rule (Addendum 1 sections 1-5), then the row table, the row
sets and the design units.

    py ces_prediction/experiments/b76_mast_availability/stage_rows.py build --part K/N   # per shot, cached
    py ces_prediction/experiments/b76_mast_availability/stage_rows.py concat             # data/.b76/rows.npz
    py ces_prediction/experiments/b76_mast_availability/stage_rows.py sets               # sets.npz, sets.json, units.json
Progress lines carry counts only.
"""
from __future__ import annotations

import argparse
import time
import warnings

import common76 as K

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def build(part):
    k, n = (int(x) for x in part.split("/"))
    cat = K.meta_tables()
    shots = K.listed_shots()[k::n]
    t0 = time.time()
    done = 0
    for i, s in enumerate(shots):
        path = K.SHOTS / f"{s}.npz"
        if path.exists():
            continue
        out = K.build_shot(s, cat)
        K.save_npz(path, **out)
        done += 1
        if (i + 1) % 100 == 0:
            K.log(f"part {k}/{n}: {i + 1}/{len(shots)} shots ({time.time() - t0:.0f}s)")
    K.log(f"part {k}/{n}: done, {done} shots built ({time.time() - t0:.0f}s)")


def concat():
    cat = K.meta_tables()
    shots = K.listed_shots()
    status, parts, counts, meta = {}, [], {}, {}
    for s in shots:
        z = np.load(K.SHOTS / f"{s}.npz", allow_pickle=False)
        st = str(z["status"])
        status[st] = status.get(st, 0) + 1
        if st != "ok":
            continue
        counts[s] = z["counts"].tolist()
        meta[s] = {"L": int(z["L"]), "t_a": float(z["t_a"]), "ts_sys": str(z["ts_sys"]), "n_chords": int(z["n_chords"]),
                   "session": K.session_of(cat, s)}
        if len(z["j"]):
            parts.append({k: z[k] for k in z.files if k not in ("status", "counts", "L", "t_a", "ts_sys", "n_chords")})
    tab = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    K.save_npz(K.CACHE / "rows.npz", **tab)
    cnt = np.array(list(counts.values()))
    info = {"listed_shots": len(shots), "shot_status": status, "shots_ok": len(counts),
            "shots_with_candidate_rows": int(sum(1 for v in counts.values() if v[3] > 0)),
            "rows_good": int(cnt[:, 0].sum()), "rows_good_in_band": int(cnt[:, 1].sum()),
            "rows_band_with_thomson_t": int(cnt[:, 2].sum()), "rows_candidate": int(cnt[:, 3].sum()),
            "shots_good": int((cnt[:, 0] > 0).sum()), "shots_band": int((cnt[:, 1] > 0).sum()),
            "shots_band_ts": int((cnt[:, 2] > 0).sum())}
    K.save_json(K.CACHE / "shots.json", {"info": info, "meta": {str(k): v for k, v in meta.items()}})
    K.log(f"concat: {info}")


def complete_mask(R, r):
    H = K.hist_matrix(R, r)
    h_ok = np.isfinite(H).all(1) & np.isfinite(R[f"{r}_G"])
    both = np.isfinite(R["te_t"]) & np.isfinite(R["ne_t"]) & np.isfinite(R[f"{r}_te_a"]) & np.isfinite(R[f"{r}_ne_a"])
    with np.errstate(all="ignore"):
        pos = (R["te_t"] / R[f"{r}_te_a"] > 0) & (R["ne_t"] / R[f"{r}_ne_a"] > 0)
    T = K.ts_matrix(R, r)
    t_ok = np.isfinite(T).all(1)
    return h_ok, both, both & ~pos, h_ok & t_ok


def sets():
    R = K.load_rows()
    n = len(R["shot"])
    sh = R["shot"]
    info = {"candidate_rows": int(n), "candidate_shots": int(len(np.unique(sh))), "per_rule": {}}
    comp = {}
    for r in K.RULES:
        h_ok, both, nonpos, c = complete_mask(R, r)
        comp[r] = c
        D = K.da_matrix(R, r)
        info["per_rule"][r] = {"visible": int((R[f"{r}_nvis"] > 0).sum()),
                               "history_and_G_complete": int(h_ok.sum()),
                               "thomson_at_anchor": int(both.sum()),
                               "nonpositive_ratio": int(nonpos.sum()),
                               "complete": int(c.sum()), "complete_shots": int(len(np.unique(sh[c]))),
                               "complete_with_dalpha": int((c & np.isfinite(D).all(1)).sum())}
    allr = np.logical_and.reduce([comp[r] for r in K.RULES])
    info["complete_all_rules"] = int(allr.sum())
    info["complete_all_rules_shots"] = int(len(np.unique(sh[allr])))
    arr = {f"complete_{r}": comp[r] for r in K.RULES}
    da_ok = np.logical_and.reduce([np.isfinite(K.da_matrix(R, r)).all(1) for r in K.DA_RULES])
    for X in K.TARGETS:
        tg = np.isfinite(R["y"] if X == "F" else R["L5"])
        arr[f"common_{X}"] = allr & tg
        arr[f"DA_{X}"] = allr & tg & da_ok
        info[f"target_{X}_defined"] = int(tg.sum())
        for nm in ("common", "DA"):
            m = arr[f"{nm}_{X}"]
            info[f"{nm}_{X}"] = {"rows": int(m.sum()), "shots": int(len(np.unique(sh[m])))}
    if not np.all(arr["common_L5"] <= arr["common_F"]):
        raise SystemExit("FATAL: the L5 common set is not inside the F common set")
    K.save_npz(K.CACHE / "sets.npz", **arr)
    meta = __import__("json").loads((K.CACHE / "shots.json").read_text(encoding="utf-8"))["meta"]
    us, dinfo = make_units(R, arr["common_F"], meta)
    info["designs"] = dinfo
    K.save_json(K.CACHE / "units.json", us)
    K.save_json(K.CACHE / "sets.json", info)
    K.log(f"sets: candidate {n}, complete under all rules {info['complete_all_rules']}, "
          f"common F {info['common_F']}, common L5 {info['common_L5']}, DA L5 {info['DA_L5']}")


def make_units(R, mask, meta):
    shots = np.unique(R["shot"][mask]).astype(int)
    sess = np.array([meta[str(s)]["session"] for s in shots])
    out = []
    ns = len(shots)
    for seed in K.SEEDS:
        fold = np.random.default_rng(seed).permutation(ns) % K.NF
        for k in range(K.NF):
            out.append({"name": f"M1_s{seed}_k{k}", "design": "M1", "sub": seed, "fit": shots[fold != k].tolist(),
                        "test": shots[fold == k].tolist()})
    su, sinv, scnt = np.unique(sess, return_inverse=True, return_counts=True)
    loads = {}
    for seed in K.SEEDS:
        order = np.random.default_rng(seed).permutation(len(su))
        load = np.zeros(K.NF, np.int64)
        fs = np.zeros(len(su), np.int64)
        for j in order:
            f = int(np.argmin(load))
            fs[j] = f
            load[f] += scnt[j]
        loads[seed] = load.tolist()
        fold = fs[sinv]
        for k in range(K.NF):
            out.append({"name": f"M2_s{seed}_k{k}", "design": "M2", "sub": seed, "fit": shots[fold != k].tolist(),
                        "test": shots[fold == k].tolist()})
    fifth = np.zeros(ns, np.int64)
    for q, part in enumerate(np.array_split(np.arange(ns), K.NF)):
        fifth[part] = q
    for q in range(1, K.NF):
        out.append({"name": f"M3_q{q}", "design": "M3", "sub": q, "fit": shots[fifth < q].tolist(),
                    "test": shots[fifth == q].tolist()})
    out.append({"name": "M4", "design": "M4", "sub": "a", "fit": shots[fifth < 4].tolist(), "test": shots[fifth == 4].tolist()})
    straddle = 0
    for q in range(1, K.NF):
        straddle += int(len(set(sess[fifth == q - 1]) & set(sess[fifth == q])))
    dinfo = {"shots": ns, "sessions": int(len(su)), "session_shots": {"min": int(scnt.min()), "median": float(np.median(scnt)),
                                                                      "max": int(scnt.max())},
             "m2_fold_shots": {str(k): v for k, v in loads.items()},
             "fifths": [{"shots": [int(shots[fifth == q].min()), int(shots[fifth == q].max())], "n": int((fifth == q).sum()),
                         "sessions": int(len(set(sess[fifth == q])))} for q in range(K.NF)],
             "sessions_straddling_a_fifth_boundary": straddle}
    tab = {"shots": shots.tolist(), "session": sess.tolist(), "fifth": fifth.tolist()}
    return {"units": out, "table": tab}, dinfo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    {"build": lambda: build(a.part), "concat": concat, "sets": sets}[a.what]()
