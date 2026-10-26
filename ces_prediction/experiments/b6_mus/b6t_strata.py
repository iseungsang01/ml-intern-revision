"""B.6-T report (PREREGISTRATION_B6T.md sec. 3): paired T1 - A0 on TEST, stratified by the
time since the last raw-detected burst (the feature table's own timing channel), plus the
same strata for each arm against PCHIP. Descriptive only -- no verdict language.

Usage (repo root, after `run_b6.py --arms t1`):
  py ces_prediction/experiments/b6_mus/b6t_strata.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
SEEDS = (42, 1, 7, 123)
POPS = ("cut", "incl")
STRATA = (("burst in window (0-10 ms)", 0.0, 10.0), ("10-50 ms", 10.0, 50.0),
          ("50-200 ms", 50.0, 200.0), (">200 ms", 200.0, np.inf), ("no raw cover", np.nan, np.nan))
B = 10000
BOOT_SEED = 12345


def load(prefix, pop, arm, seed, target):
    d = np.load(DATA / f"{prefix}_{pop}_{arm}_refit_s{seed}" / "comparison_errors_test.npz",
                allow_pickle=True)
    return {k[len(target) + 1:]: d[k] for k in d.files if k.startswith(target + "_")}


def shot_ci(d0, d1, shots, rng):
    uniq = np.unique(shots)
    s0 = np.array([d0[shots == c].sum() for c in uniq])
    s1 = np.array([d1[shots == c].sum() for c in uniq])
    n = len(uniq)
    draws = rng.integers(0, n, size=(B, n))
    t0 = s0[draws].sum(axis=1)
    t1 = s1[draws].sum(axis=1)
    ok = t0 > 0
    sk = 1.0 - t1[ok] / t0[ok]
    return (float(1.0 - s1.sum() / s0.sum()) if s0.sum() > 0 else float("nan"),
            float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5)))


def main():
    rng = np.random.default_rng(BOOT_SEED)
    names = sorted(p.name for p in (DATA / ".b6_data").glob("s*.csv"))
    shot_of = {i: int(n[1:].split(".")[0]) for i, n in enumerate(names)}
    times = {n: pd.read_csv(DATA / ".b6_data" / n, usecols=["time"])["time"].to_numpy(float)
             for n in names}
    tsb = {}
    for n in names:
        z = np.load(DATA / ".b6_features_ti" / f"mus_features_s{n[1:-4]}.npz")
        f = z["feat"]
        cover = f[:, :26].any(axis=1)
        ms = np.expm1(f[:, 26])
        ms[~cover] = np.nan
        tsb[n] = dict(zip(np.round(z["time"], 4), ms))
    out = {}
    for target in ("CES_TI", "CES_VT"):
        out[target] = {}
        for pop in POPS:
            a0 = [load(".b6", pop, "a0", s, target) for s in SEEDS]
            t1 = [load(".b6", pop, "t1", s, target) for s in SEEDS]
            for key in ("shot", "row", "y_true", "se_pchip"):
                for r in t1:
                    if not np.array_equal(a0[0][key], r[key]):
                        raise SystemExit(f"FATAL: rows not paired ({target}/{pop}/{key})")
            shots = a0[0]["shot"]
            rows = a0[0]["row"]
            ms = np.array([tsb[names[int(s)]].get(round(float(times[names[int(s)]][int(r)]), 4), np.nan)
                           for s, r in zip(shots, rows)])
            d0 = np.concatenate([r["se_model"] for r in a0])
            d1 = np.concatenate([r["se_model"] for r in t1])
            dp = np.concatenate([r["se_pchip"] for r in a0])
            shot_rep = np.tile(shots, len(SEEDS))
            ms_rep = np.tile(ms, len(SEEDS))
            node = {}
            p, lo, hi = shot_ci(d0, d1, shot_rep, rng)
            node["pooled"] = {"t1_vs_a0": p, "ci": [lo, hi], "n_rows": int(len(shots)),
                              "a0_vs_pchip": float(1 - d0.sum() / dp.sum()),
                              "t1_vs_pchip": float(1 - d1.sum() / dp.sum())}
            per_seed = []
            for i in range(len(SEEDS)):
                ps, ls, hs = shot_ci(a0[i]["se_model"], t1[i]["se_model"], shots, rng)
                per_seed.append({"seed": SEEDS[i], "skill": ps, "ci": [ls, hs]})
            node["per_seed"] = per_seed
            node["per_shot"] = {}
            for s in np.unique(shots):
                m = shot_rep == s
                if d0[m].sum() > 0:
                    node["per_shot"][shot_of[int(s)]] = float(1 - d1[m].sum() / d0[m].sum())
            node["strata"] = {}
            for label, lo_ms, hi_ms in STRATA:
                m = np.isnan(ms_rep) if np.isnan(lo_ms) else (ms_rep >= lo_ms) & (ms_rep < hi_ms)
                if m.sum() == 0 or d0[m].sum() == 0:
                    continue
                node["strata"][label] = {
                    "n_rows": int(m.sum() // len(SEEDS)),
                    "t1_vs_a0": float(1 - d1[m].sum() / d0[m].sum()),
                    "a0_vs_pchip": float(1 - d0[m].sum() / dp[m].sum()),
                    "t1_vs_pchip": float(1 - d1[m].sum() / dp[m].sum()),
                    "share_a0_sse": float(d0[m].sum() / d0.sum())}
            out[target][pop] = node
            print(f"\n== {target} [{pop}] n={node['pooled']['n_rows']} ==")
            print(f"  pooled T1-A0 {p:+.4f} CI [{lo:+.4f},{hi:+.4f}]   A0 vs PCHIP "
                  f"{node['pooled']['a0_vs_pchip']:+.3f}  T1 vs PCHIP {node['pooled']['t1_vs_pchip']:+.3f}")
            print("  per seed " + "  ".join(f"{r['seed']}:{r['skill']:+.3f}[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                                            for r in per_seed))
            print("  per shot " + "  ".join(f"{k}:{v:+.3f}" for k, v in node["per_shot"].items()))
            for label, s in node["strata"].items():
                print(f"  {label:28s} n={s['n_rows']:5d}  T1-A0 {s['t1_vs_a0']:+.3f}  A0/PCHIP "
                      f"{s['a0_vs_pchip']:+.3f}  T1/PCHIP {s['t1_vs_pchip']:+.3f}  A0 SSE share {s['share_a0_sse']:.2f}")
    dst = DATA / ".b6t_report.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[b6t] wrote {dst}")


if __name__ == "__main__":
    main()
