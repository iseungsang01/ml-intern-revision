"""B.6 sec. 4.5: the one time-ordered re-split inside the pool (report only, no verdict).

The pool is split by shot number once -- train = the three earliest discharges
(#31097, #31359, #31747), val = the three latest (#32027, #32092, #32097) -- and A0 and
A1 are trained on the early half (init seed 42, the fold epoch cap and patience) and
scored on the late half's rows (split tag val). The paired A1 - A0 skill on those rows
is reported beside the TEST verdict as the generalisation-in-time check the
pre-registration asks for; it decides nothing.

Usage (repo root, after run_b6.py):
  py ces_prediction/experiments/b6_mus/run_b6_timesplit.py            # train + score
  py ces_prediction/experiments/b6_mus/run_b6_timesplit.py --analyze
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b6 import (DATA, POOL, TEST, FOLD_EPOCH_CAP, POPULATIONS, base_env,  # noqa: E402
                    write_manifest, train_run, eval_run, check_cap, stage_csvs, FEATURE_DIR)

EARLY = tuple(sorted(POOL)[:3])
LATE = tuple(sorted(POOL)[3:])
LOG = DATA / ".b6_timesplit.log"
B = 10000
BOOT_SEED = 12345


def run():
    stage_csvs()
    for pop, cut_ev in POPULATIONS.items():
        control_metrics = DATA / f".b6_{pop}_a0_refit_s42" / "metrics.json"
        for arm in ("a0", "a1"):
            env = base_env(arm, cut_ev, FEATURE_DIR)
            split_dir = DATA / f".b6_time_{pop}_{arm}_split"
            out_dir = DATA / f".b6_time_{pop}_{arm}"
            write_manifest(split_dir, EARLY, LATE, TEST)
            tag = f"time/{pop}/{arm}"
            if not train_run(tag, env, split_dir, out_dir, 42, 42, FOLD_EPOCH_CAP, False, LOG):
                raise SystemExit(f"FATAL: {tag} failed -- see {LOG}")
            check_cap(out_dir, FOLD_EPOCH_CAP)
            if not eval_run(f"{tag}/eval", env, split_dir, out_dir, control_metrics, "val", LOG):
                raise SystemExit(f"FATAL: {tag} eval failed -- see {LOG}")
    print("[time] done -- run with --analyze")


def _load(path, target):
    d = np.load(path, allow_pickle=True)
    return {k[len(target) + 1:]: d[k] for k in d.files if k.startswith(target + "_")}


def _shot_ci(d0, d1, shots, rng):
    uniq = np.unique(shots)
    s0 = np.array([d0[shots == c].sum() for c in uniq])
    s1 = np.array([d1[shots == c].sum() for c in uniq])
    n = len(uniq)
    draws = rng.integers(0, n, size=(B, n))
    t0 = s0[draws].sum(axis=1)
    t1 = s1[draws].sum(axis=1)
    ok = t0 > 0
    sk = 1.0 - t1[ok] / t0[ok]
    point = float(1.0 - s1.sum() / s0.sum()) if s0.sum() > 0 else float("nan")
    return point, float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))


def analyze():
    rng = np.random.default_rng(BOOT_SEED)
    names = dict(enumerate(sorted(p.name for p in (DATA / ".b6_data").glob("s*.csv"))))
    shot_of = {i: int(n[1:].split(".")[0]) for i, n in names.items()}
    out = {"train": list(EARLY), "val": list(LATE), "targets": {}}
    for target in ("CES_VT", "CES_TI"):
        node = {}
        for pop in POPULATIONS:
            a0 = _load(DATA / f".b6_time_{pop}_a0" / "comparison_errors_val.npz", target)
            a1 = _load(DATA / f".b6_time_{pop}_a1" / "comparison_errors_val.npz", target)
            for key in ("shot", "row", "y_true", "se_pchip"):
                if not np.array_equal(a0[key], a1[key]):
                    raise SystemExit(f"FATAL: rows not paired ({target}/{pop}/{key})")
            p, lo, hi = _shot_ci(a0["se_model"], a1["se_model"], a0["shot"], rng)
            per_shot = {}
            for s in np.unique(a0["shot"]):
                m = a0["shot"] == s
                if a0["se_model"][m].sum() > 0:
                    per_shot[shot_of[int(s)]] = float(1.0 - a1["se_model"][m].sum() / a0["se_model"][m].sum())
            node[pop] = {"n_rows": int(len(a0["shot"])), "skill_a1_vs_a0": p, "ci": [lo, hi],
                         "per_shot": per_shot,
                         "a0_skill_vs_pchip": float(1.0 - a0["se_model"].sum() / a0["se_pchip"].sum()),
                         "a1_skill_vs_pchip": float(1.0 - a1["se_model"].sum() / a1["se_pchip"].sum())}
            print(f"{target} [{pop}] time-split A1-A0 {p:+.4f} CI [{lo:+.4f},{hi:+.4f}] "
                  f"n={node[pop]['n_rows']} per-shot {per_shot}")
        out["targets"][target] = node
    dst = DATA / ".b6_timesplit.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[time] wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run()


if __name__ == "__main__":
    main()
