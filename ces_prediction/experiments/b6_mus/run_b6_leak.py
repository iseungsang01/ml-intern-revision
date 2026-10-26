"""B.6 H3: session leakage measured directly with the companion pairs (sec. 6).

Not a test -- a measurement. Two isolated arms, run ONLY after the confirmatory TEST
verdict (`verdict_b6.py`) has been written, with their own output prefix
(`data/.b6_leak_*`) so nothing here can touch the sec. 4 artifacts:

  L1a  the A0 protocol with companion #31923 added to EVERY train set (folds and the
       four refits); the partner #31921 is a TEST shot, so the paired skill change on
       #31921's rows against the A0 refits is the leakage in gate units.
  L1b  fold 2 only (val = #31359): train = the other five pool shots + companion #31357;
       the paired skill change on #31359's rows against the plain A0 fold-2 model is the
       within-pool leakage. The plain A0 fold is scored on val here as well (the sec. 4
       batch never scored a fold on val), so both sides of the pair come from this file.

Both measurements are reported with a 500 ms block bootstrap (one discharge, so the
shot-clustered primary is undefined) and alongside the 100 Hz summary distance that put
these pairs at the 0.03rd percentile (SELECTION.md).

Usage (repo root, after verdict_b6.py):
  py ces_prediction/experiments/b6_mus/run_b6_leak.py            # train + score
  py ces_prediction/experiments/b6_mus/run_b6_leak.py --analyze  # the numbers
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b6 import (DATA, FOLDS, POOL, TEST, COMPANIONS, REFIT_SEEDS, FOLD_EPOCH_CAP,  # noqa: E402
                    POPULATIONS, base_env, write_manifest, train_run, eval_run, check_cap,
                    stage_csvs, FEATURE_DIR)

LEAK_A = DATA / ".b6_leak_a"       # L1a prefix
LEAK_B = DATA / ".b6_leak_b"       # L1b prefix
MAIN = DATA / ".b6"                # the sec. 4 batch prefix (read-only here)
LOG = DATA / ".b6_leak.log"
BLOCK_S = 0.5
B = 10000
BOOT_SEED = 12345
L1B_FOLD = next(f for f in FOLDS if f["val"] == (31359,))


def verdict_locked():
    return (DATA / ".b6_verdict.json").exists()


def run_l1a(pop, cut_ev):
    env = base_env("a0", cut_ev, FEATURE_DIR)
    control_metrics = DATA / f".b6_{pop}_a0_refit_s42" / "metrics.json"
    best = []
    for f in FOLDS:
        tag = f"leak_a/{pop}/fold{f['fold']}"
        split_dir = DATA / f".b6_leak_a_{pop}_fold{f['fold']}_split"
        out_dir = DATA / f".b6_leak_a_{pop}_fold{f['fold']}"
        write_manifest(split_dir, (*f["train"], 31923), f["val"], TEST)
        if not train_run(tag, env, split_dir, out_dir, 42, 42, FOLD_EPOCH_CAP, False, LOG):
            raise SystemExit(f"FATAL: {tag} failed -- see {LOG}")
        best.append(check_cap(out_dir, FOLD_EPOCH_CAP))
    median_epoch = int(statistics.median(best))
    print(f"[leak] L1a {pop}: fold best epochs {best} -> refit at {median_epoch}")
    refit_split = DATA / f".b6_leak_a_{pop}_refit_split"
    write_manifest(refit_split, (*POOL, 31923), (*POOL, 31923), TEST)
    for init in REFIT_SEEDS:
        tag = f"leak_a/{pop}/refit_s{init}"
        out_dir = DATA / f".b6_leak_a_{pop}_refit_s{init}"
        if not train_run(tag, env, refit_split, out_dir, 42, init, median_epoch, True, LOG):
            raise SystemExit(f"FATAL: {tag} failed -- see {LOG}")
        if not eval_run(f"{tag}/eval", env, refit_split, out_dir, control_metrics, "test", LOG):
            raise SystemExit(f"FATAL: {tag} eval failed -- see {LOG}")


def run_l1b(pop, cut_ev):
    env = base_env("a0", cut_ev, FEATURE_DIR)
    control_metrics = DATA / f".b6_{pop}_a0_refit_s42" / "metrics.json"
    f = L1B_FOLD
    # the leaked fold: train + 31357
    split_dir = DATA / f".b6_leak_b_{pop}_fold{f['fold']}_split"
    out_dir = DATA / f".b6_leak_b_{pop}_fold{f['fold']}"
    write_manifest(split_dir, (*f["train"], 31357), f["val"], TEST)
    tag = f"leak_b/{pop}/fold{f['fold']}"
    if not train_run(tag, env, split_dir, out_dir, 42, 42, FOLD_EPOCH_CAP, False, LOG):
        raise SystemExit(f"FATAL: {tag} failed -- see {LOG}")
    if not eval_run(f"{tag}/eval", env, split_dir, out_dir, control_metrics, "val", LOG):
        raise SystemExit(f"FATAL: {tag} eval failed -- see {LOG}")
    # the plain A0 fold, scored on the same val shot (its training is the sec. 4 run)
    a0_split = DATA / f".b6_{pop}_a0_fold{f['fold']}_split"
    a0_out = DATA / f".b6_{pop}_a0_fold{f['fold']}"
    if not eval_run(f"leak_b/{pop}/a0_fold{f['fold']}/eval", env, a0_split, a0_out,
                    control_metrics, "val", LOG):
        raise SystemExit("FATAL: A0 fold val scoring failed")


def _load(path, target):
    d = np.load(path, allow_pickle=True)
    return {k[len(target) + 1:]: d[k] for k in d.files if k.startswith(target + "_")}


def _block_ci(d0, d1, times, rng):
    blocks = (times / BLOCK_S).astype(np.int64)
    uniq = np.unique(blocks)
    s0 = np.array([d0[blocks == c].sum() for c in uniq])
    s1 = np.array([d1[blocks == c].sum() for c in uniq])
    n = len(uniq)
    draws = rng.integers(0, n, size=(B, n))
    t0 = s0[draws].sum(axis=1)
    t1 = s1[draws].sum(axis=1)
    ok = t0 > 0
    sk = 1.0 - t1[ok] / t0[ok]
    point = float(1.0 - s1.sum() / s0.sum()) if s0.sum() > 0 else float("nan")
    return point, float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5)), int(n)


def _row_times(names, shots, rows):
    import pandas as pd
    cache = {}
    t = np.zeros(len(shots))
    for i, (s, r) in enumerate(zip(shots, rows)):
        name = names[int(s)]
        if name not in cache:
            cache[name] = pd.read_csv(DATA / ".b6_data" / name, usecols=["time"])["time"].to_numpy(float)
        t[i] = cache[name][int(r)]
    return t


def analyze():
    rng = np.random.default_rng(BOOT_SEED)
    names = dict(enumerate(sorted(p.name for p in (DATA / ".b6_data").glob("s*.csv"))))
    shot_of = {i: int(n[1:].split(".")[0]) for i, n in names.items()}
    out = {}
    for target in ("CES_TI", "CES_VT"):
        node = {}
        for pop in POPULATIONS:
            # L1a: TEST rows of #31921, pooled over the four refit seeds
            d0s, d1s, ts = [], [], []
            for init in REFIT_SEEDS:
                a0 = _load(DATA / f".b6_{pop}_a0_refit_s{init}" / "comparison_errors_test.npz", target)
                lk = _load(DATA / f".b6_leak_a_{pop}_refit_s{init}" / "comparison_errors_test.npz", target)
                for key in ("shot", "row", "y_true"):
                    if not np.array_equal(a0[key], lk[key]):
                        raise SystemExit(f"FATAL: L1a rows not paired ({target}/{pop}/{init}/{key})")
                m = np.array([shot_of[int(s)] == 31921 for s in a0["shot"]])
                d0s.append(a0["se_model"][m])
                d1s.append(lk["se_model"][m])
                ts.append(_row_times(names, a0["shot"][m], a0["row"][m]))
            d0 = np.concatenate(d0s)
            d1 = np.concatenate(d1s)
            t = np.concatenate(ts)
            p, lo, hi, nb = _block_ci(d0, d1, t, rng) if len(d0) else (float("nan"),) * 3 + (0,)
            l1a = {"partner": 31921, "n_rows": int(len(d0) // max(len(REFIT_SEEDS), 1)),
                   "delta_skill": p, "block_ci": [lo, hi], "n_blocks": nb}
            # L1b: val rows of #31359, fold 2, seed 42 only
            a0 = _load(DATA / f".b6_{pop}_a0_fold{L1B_FOLD['fold']}" / "comparison_errors_val.npz", target)
            lk = _load(DATA / f".b6_leak_b_{pop}_fold{L1B_FOLD['fold']}" / "comparison_errors_val.npz", target)
            for key in ("shot", "row", "y_true"):
                if not np.array_equal(a0[key], lk[key]):
                    raise SystemExit(f"FATAL: L1b rows not paired ({target}/{pop}/{key})")
            t = _row_times(names, a0["shot"], a0["row"])
            p, lo, hi, nb = (_block_ci(a0["se_model"], lk["se_model"], t, rng)
                             if len(t) else (float("nan"),) * 3 + (0,))
            l1b = {"partner": 31359, "n_rows": int(len(t)), "delta_skill": p,
                   "block_ci": [lo, hi], "n_blocks": nb}
            node[pop] = {"L1a": l1a, "L1b": l1b}
            print(f"{target} [{pop}]  L1a(31923->31921) dSkill {l1a['delta_skill']:+.4f} "
                  f"CI [{l1a['block_ci'][0]:+.4f},{l1a['block_ci'][1]:+.4f}] n={l1a['n_rows']}   "
                  f"L1b(31357->31359) dSkill {l1b['delta_skill']:+.4f} CI [{l1b['block_ci'][0]:+.4f},"
                  f"{l1b['block_ci'][1]:+.4f}] n={l1b['n_rows']}")
        out[target] = node
    dst = DATA / ".b6_leak.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[leak] wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not verdict_locked():
        raise SystemExit("REFUSED: H3 runs only after the sec. 4 TEST verdict is written "
                         "(data/.b6_verdict.json missing) -- PREREGISTRATION_B6.md sec. 6.")
    if args.analyze:
        analyze()
        return
    stage_csvs()
    for pop, cut_ev in POPULATIONS.items():
        run_l1a(pop, cut_ev)
        run_l1b(pop, cut_ev)
    print("[leak] done -- run with --analyze")


if __name__ == "__main__":
    main()
