"""Offline baselines on the B.12 fixed hidden points (CPU only).

For every hidden observed row the context is the same as the arms see: the observed,
non-hidden rows of the same block and target. Arms: linear, PCHIP and the acausal
Matern-3/2 GP (`baselines_interpolation.predict_gp`, the strongest offline arm, §8p),
all from the shared module so their definitions cannot drift.

Usage: py ces_prediction/experiments/b12/baselines_b12.py --seed 42 --cut 3000
Output: data/.b12_base_cut{cut}_s{seed}/{val,test}_points.npz with keys
shot, row, target, truth, and one prediction array per baseline (mean = train mean),
plus test_sparse{k}_points.npz for the label-sparsified sets.
"""

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from b12_data import prepare, split_blocks  # noqa: E402
from baselines_interpolation import predict_linear, predict_pchip, predict_gp  # noqa: E402

DATA = HERE.parents[2] / "data"
METHODS = {"linear": predict_linear, "pchip": predict_pchip, "gp": predict_gp}


def train_mean(grid, names):
    """Per-target mean of observed train values: the only baseline with zero shot context."""
    return [float(np.nanmean(np.concatenate([grid[n][:, 1 + t] for n in names]))) for t in range(2)]


def score_split(grid, names, masks, means):
    out = {k: [] for k in ("shot", "row", "target", "truth", "mean", *METHODS)}
    for name in names:
        arr, hid = grid[name], masks[name]
        time = arr[:, 0].astype(np.float64)
        for a, b in split_blocks(arr):
            for t in range(2):
                v = arr[a:b, 1 + t].astype(np.float64)
                h = hid[a:b, t]
                ctx = ~np.isnan(v) & ~h
                ct, cv = time[a:b][ctx], v[ctx]
                for r in np.flatnonzero(h):
                    out["shot"].append(name)
                    out["row"].append(a + r)
                    out["target"].append(t)
                    out["truth"].append(v[r])
                    out["mean"].append(means[t])
                    for m, fn in METHODS.items():
                        out[m].append(fn(ct, cv, time[a + r]) if len(ct) else means[t])  # no context: train mean
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cut", type=float, required=True)
    args = ap.parse_args()
    t0 = _time.time()
    prep = prepare(DATA, DATA / f".b1_manifest_s{args.seed}", args.cut, DATA / ".b12_masks", args.seed)
    out = DATA / f".b12_base_cut{int(args.cut)}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    means = train_mean(prep["grid"], prep["train"])
    sets = [("val", prep["val"], prep["val_masks"]), ("test", prep["test"], prep["test_masks"])]
    sets += [(f"test_sparse{k}", prep["test"], m) for k, m in prep["sparse_masks"].items()]
    for split, names, masks in sets:
        pts = score_split(prep["grid"], names, masks, means)
        np.savez_compressed(out / f"{split}_points.npz", **pts)
        n = [(pts["target"] == t).sum() for t in range(2)]
        print(f"[b12-base] seed={args.seed} cut={args.cut:g} {split}: hidden TI={n[0]} VT={n[1]} "
              f"({_time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
