"""B.6 H1 verdict: the pre-registered joint rule, executed and nothing else.

Implements PREREGISTRATION_B6.md sec. 4.2 for `CES_VT`, A1 - A0 paired on identical
rows. PASS requires ALL of (a) pooled shot-clustered CI clearing zero in the improving
direction in BOTH populations, (b) >= 3/4 per-seed CIs in the same direction,
(c) the generality guard (majority of test discharges improved AND the pooled sign
survives removing the single best-contributing discharge), (d) the secondary
shot x 500 ms block bootstrap not significantly reversed. Anything partial is 미정,
everything reversed is 기각 -- reported as measured, never rounded up.

Also reported (never part of the verdict): the same table for `CES_TI` (sec. 5:
descriptive only -- no confirmatory language exists for it at this shot count),
the H1-m activity stratification (mechanism sentence, not a criterion), the
spike-row SSE share (|V_rot| > 1000 km/s, the sec. 8ab audit convention), and PR2
fallback fractions from the run reports.

Usage (repo root, after run_b6.py):
  py ces_prediction/experiments/b6_mus/verdict_b6.py
  py ces_prediction/experiments/b6_mus/verdict_b6.py --prefix data/.b6_plumb --split val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[0] / "hires_shots"))
from folds import BOOTSTRAP, TEST  # noqa: E402
from select_hires_shots import held_mask, blocks_of  # noqa: E402

DATA = REPO_ROOT / "data"
SEEDS = (42, 1, 7, 123)
B = BOOTSTRAP["primary"]["B"]
BOOT_SEED = BOOTSTRAP["primary"]["seed"]
ACTIVITY_HALF_S = 0.25          # H1-m: +-250 ms of genuine V_rot spread (sec. 4.2)
BLOCK_S = 0.5                   # secondary bootstrap cluster width
SPIKE_VT = 1000.0               # |V_rot| spike proxy (sec. 8ab audit convention)


def a(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def load_arm(prefix, pop, arm, seed, split):
    d = np.load(Path(f"{prefix}_{pop}_{arm}_refit_s{seed}") / f"comparison_errors_{split}.npz",
                allow_pickle=True)
    return {k: d[k] for k in d.files}


def pairing_check(runs, target):
    ref = runs[0]
    for r in runs[1:]:
        for key in ("shot", "dt_ms", "y_true", "se_pchip", "row"):
            k = f"{target}_{key}"
            if k not in ref or k not in r:
                raise SystemExit(f"FATAL: pairing key {k} missing")
            if not np.array_equal(ref[k], r[k]):
                raise SystemExit(f"FATAL: {k} differs between runs -- rows are not paired")


def cluster_ci(d0, d1, clusters, rng):
    """95% CI of paired skill 1 - sum(d1)/sum(d0) under cluster resampling."""
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    s0 = np.array([d0[idx[c]].sum() for c in uniq])
    s1 = np.array([d1[idx[c]].sum() for c in uniq])
    n = len(uniq)
    draws = rng.integers(0, n, size=(B, n))
    t0 = s0[draws].sum(axis=1)
    t1 = s1[draws].sum(axis=1)
    ok = t0 > 0
    sk = 1.0 - t1[ok] / t0[ok]
    point = float(1.0 - s1.sum() / s0.sum()) if s0.sum() > 0 else float("nan")
    return point, float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))


def row_times(shot_index_to_name, shots, rows, data_dir, cache):
    t = np.zeros(len(shots))
    for i, (s, r) in enumerate(zip(shots, rows)):
        name = shot_index_to_name[int(s)]
        if name not in cache:
            cache[name] = pd.read_csv(data_dir / name, usecols=["time"])["time"].to_numpy(float)
        t[i] = cache[name][int(r)]
    return t


def local_vt_activity(shots, rows, shot_index_to_name, data_dir):
    """Per scored row: std of GENUINE V_rot observations within +-250 ms, target row excluded."""
    out = np.zeros(len(shots))
    per_file = {}
    for name in {shot_index_to_name[int(s)] for s in shots}:
        df = pd.read_csv(data_dir / name, usecols=["time", "CES_VT"])
        t = df["time"].to_numpy(float)
        v = df["CES_VT"].to_numpy(float)
        genuine = np.isfinite(v) & ~held_mask(v, blocks_of(t))
        per_file[name] = (t, v, genuine)
    for i, (s, r) in enumerate(zip(shots, rows)):
        t, v, genuine = per_file[shot_index_to_name[int(s)]]
        r = int(r)
        sel = genuine & (np.abs(t - t[r]) <= ACTIVITY_HALF_S)
        sel[r] = False
        out[i] = float(np.std(v[sel])) if sel.sum() >= 2 else 0.0
    return out


def analyze(prefix, pop, split, target, rng):
    a0 = [load_arm(prefix, pop, "a0", s, split) for s in SEEDS]
    a1 = [load_arm(prefix, pop, "a1", s, split) for s in SEEDS]
    pairing_check(a0 + a1, target)
    ref = a0[0]
    shots = ref[f"{target}_shot"]
    rows = ref[f"{target}_row"]
    y = ref[f"{target}_y_true"]

    data_dir = DATA / ".b6_data"
    names = sorted(p.name for p in data_dir.glob("s*.csv"))
    idx_to_name = dict(enumerate(names))
    cache = {}
    times = row_times(idx_to_name, shots, rows, data_dir, cache)

    # pooled across seeds: concatenate rows, cluster stays the physical discharge
    d0 = np.concatenate([r[f"{target}_se_model"] for r in a0])
    d1 = np.concatenate([r[f"{target}_se_model"] for r in a1])
    shot_rep = np.tile(shots, len(SEEDS))
    time_rep = np.tile(times, len(SEEDS))

    point, lo, hi = cluster_ci(d0, d1, shot_rep, rng)
    a_pass = lo > 0.0

    seed_rows = []
    n_pos = n_neg = 0
    for i, s in enumerate(SEEDS):
        p_, l_, h_ = cluster_ci(a0[i][f"{target}_se_model"], a1[i][f"{target}_se_model"],
                                shots, rng)
        seed_rows.append({"seed": s, "skill": p_, "ci": [l_, h_]})
        n_pos += l_ > 0
        n_neg += h_ < 0
    b_pass = n_pos >= 3

    per_shot = {}
    for s in np.unique(shots):
        m = shot_rep == s
        if d0[m].sum() > 0:
            per_shot[int(s)] = float(1.0 - d1[m].sum() / d0[m].sum())
    wins = sum(v > 0 for v in per_shot.values())
    best = max(per_shot, key=per_shot.get) if per_shot else None
    keep = shot_rep != best
    minus_top1 = (1.0 - d1[keep].sum() / d0[keep].sum()) if d0[keep].sum() > 0 else float("nan")
    c_pass = (wins > len(per_shot) / 2) and (np.sign(minus_top1) == np.sign(point))

    blocks = shot_rep.astype(np.int64) * 1_000_000 + (time_rep / BLOCK_S).astype(np.int64)
    pb, lb, hb = cluster_ci(d0, d1, blocks, rng)
    d_pass = not (hb < 0.0 if point > 0 else lb > 0.0)

    verdict = ("PASS" if (a_pass and b_pass and c_pass and d_pass)
               else "REJECT" if (hi < 0 and n_neg >= 3) else "UNDECIDED")

    act = local_vt_activity(shots, rows, idx_to_name, data_dir)
    strata = {}
    tercile = np.zeros(len(shots), dtype=int)
    for s in np.unique(shots):
        m = shots == s
        if m.sum() >= 3:
            qs = np.quantile(act[m], [1 / 3, 2 / 3])
            tercile[m] = np.digitize(act[m], qs)
    ter_rep = np.tile(tercile, len(SEEDS))
    for t_id, label in ((0, "quiet"), (1, "middle"), (2, "active")):
        m = ter_rep == t_id
        strata[label] = (float(1.0 - d1[m].sum() / d0[m].sum())
                         if m.any() and d0[m].sum() > 0 else None)
    h1m = (strata.get("active") is not None and strata.get("quiet") is not None
           and strata["active"] > strata["quiet"])

    spike = np.abs(np.tile(y, len(SEEDS))) > SPIKE_VT
    spike_share = {"a0": float(d0[spike].sum() / max(d0.sum(), 1e-12)),
                   "a1": float(d1[spike].sum() / max(d1.sum(), 1e-12))}

    return {
        "n_rows": int(len(shots)), "n_shots": int(len(per_shot)),
        "pooled": {"skill": point, "ci": [lo, hi], "pass": bool(a_pass)},
        "per_seed": seed_rows, "seed_consistency_pass": bool(b_pass),
        "per_shot_skill": per_shot, "wins": int(wins),
        "minus_top1": float(minus_top1), "generality_pass": bool(c_pass),
        "block_ci": [lb, hb], "block_not_reversed": bool(d_pass),
        "verdict": verdict,
        "h1m_strata": strata, "h1m_mechanism_holds": bool(h1m),
        "spike_sse_share": spike_share,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(DATA / ".b6"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--pops", nargs="*", default=["cut", "incl"])
    args = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

    out = {"split": args.split, "test_shots": list(TEST), "targets": {}}
    for target in ("CES_VT", "CES_TI"):
        role = "confirmatory (H1)" if target == "CES_VT" else "descriptive only (sec. 5)"
        node = {"role": role, "populations": {}}
        for pop in args.pops:
            r = analyze(args.prefix, pop, args.split, target, rng)
            node["populations"][pop] = r
            print(a(f"\n== {target} [{pop}] ({role}) =="))
            print(a(f"  pooled A1-A0 skill {r['pooled']['skill']:+.4f} "
                    f"CI [{r['pooled']['ci'][0]:+.4f}, {r['pooled']['ci'][1]:+.4f}]"))
            print(a(f"  seeds >=3/4 same dir: {r['seed_consistency_pass']}   "
                    f"shots won {r['wins']}/{r['n_shots']}  -top1 {r['minus_top1']:+.4f}"))
            print(a(f"  block CI [{r['block_ci'][0]:+.4f}, {r['block_ci'][1]:+.4f}]   "
                    f"strata {r['h1m_strata']}"))
            print(a(f"  verdict: {r['verdict']}"))
        if target == "CES_VT":
            both = [node["populations"][p]["verdict"] for p in args.pops]
            node["h1"] = ("PASS" if all(v == "PASS" for v in both)
                          else "REJECT" if all(v == "REJECT" for v in both) else "UNDECIDED")
            print(a(f"\nH1 ({'+'.join(args.pops)}): {node['h1']}"))
        out["targets"][target] = node

    dst = DATA / (".b6_verdict.json" if args.split == "test" else ".b6_verdict_val.json")
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(a(f"\n[b6v] wrote {dst}"))


if __name__ == "__main__":
    main()
