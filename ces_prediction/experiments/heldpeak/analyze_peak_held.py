"""§8r: cross the two facts about `CES_VT` that have never been crossed.

Fact 1 (§1): 54% of *observed* `CES_VT` values are held -- forward-filled instrument
repeats, not measurements. `CES_TI` has ~0%.
Fact 2 (peak analysis): `CES_VT` PASSes at high-activity peaks (+0.69) while the global
`CES_VT` result is n.s.

A held row's correct answer *is* the previous value, so persistence is exactly right
there and no model can beat it. If held rows are concentrated in the quiet bulk, then
part of the global `CES_VT` n.s. is a property of the scored population rather than of
the model -- and the peak result is partly "peaks are where the instrument was actually
updating". This script measures that directly, with no retraining and no model change:
the 2x2 crosstab (is_peak x is_held) and the skill of each cell.

Scored on the HELD-KEPT (`stuck0`) npz, because under the headline `genuine` treatment
held values are already NaN'd out of scoring and the held column would be empty by
construction.

The held flag is recomputed from the raw CSVs with `dataset._stuck_repeat_mask` -- the
same definition the loader uses, never a second one -- and aligned to the npz rows by
`{target}_shot` / `{target}_row`, with the alignment proved by matching the `time`
column value for value.

Usage (repo root):
  py ces_prediction/experiments/heldpeak/rerun_compare_stuck0.py   # writes the npz first
  py ces_prediction/experiments/heldpeak/analyze_peak_held.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
sys.path.insert(0, str(CES_DIR))
sys.path.insert(0, str(CES_DIR / "experiments"))
from dataset import (  # noqa: E402
    KSTAR_CES_Dataset,
    STUCK_GAP_SECONDS,
    TARGET_COLUMNS,
    TIME_COLUMN,
)
from runner_common import BASELINE_OUT, SPLIT_SRC  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
NPZ_NAME = "comparison_errors_test__test_stuck0_held.npz"
OUT_PATH = REPO_ROOT / "data" / ".peakheld_analysis.json"
B = 10000
BOOTSTRAP_SEED = 12345
WINDOW_SIZE = 4
CELLS = (("peak", "held"), ("peak", "genuine"), ("bulk", "held"), ("bulk", "genuine"))


def _per_shot_sums(shot, se):
    order = np.argsort(shot, kind="stable")
    s, e = shot[order], se[order]
    shots, starts = np.unique(s, return_index=True)
    sums = np.add.reduceat(e, starts) if len(s) else np.zeros(0)
    counts = np.diff(np.append(starts, len(s))) if len(s) else np.zeros(0)
    return shots, sums, counts


def _bootstrap(shot, se_a, se_b, rng):
    """95% CI of skill = 1 - MSE_a/MSE_b under whole-shot resampling (PR4).

    Identical to the harness used in §8g/§8i/§8p so cell skills are comparable to
    every other number in the record.
    """
    if len(shot) == 0:
        return {"n": 0, "n_shots": 0, "skill_point": float("nan"),
                "skill_ci95": [float("nan"), float("nan")], "pass": False}
    _, sa, _ = _per_shot_sums(shot, se_a)
    _, sb, _ = _per_shot_sums(shot, se_b)
    n = len(sa)
    skill_point = (1.0 - sa.sum() / sb.sum()) if sb.sum() > 0 else float("nan")
    idx = rng.integers(0, n, size=(B, n))
    skill = 1.0 - sa[idx].sum(axis=1) / sb[idx].sum(axis=1)
    finite = np.isfinite(skill)
    lo, hi = (np.percentile(skill[finite], [2.5, 97.5]) if finite.any()
              else (float("nan"), float("nan")))
    return {"n": int(len(shot)), "n_shots": int(n), "skill_point": float(skill_point),
            "skill_ci95": [float(lo), float(hi)], "pass": bool(lo > 0.0)}


def held_masks(dataset, file_idx):
    """Per-target held mask for one shot file, aligned to `dataset.file_arrays[file_idx]`.

    Recomputed from the raw CSV rather than from `file_arrays`, because the loader
    computes held-ness BEFORE dropping rows where both targets are missing, and those
    dropped rows can in principle change the contiguous-block segmentation. The `time`
    assertion below turns "in principle" into a proof for this dataset.
    """
    path = dataset.valid_files[file_idx]
    input_cols = list(dataset._input_columns())
    df = pd.read_csv(path, usecols=[TIME_COLUMN] + list(TARGET_COLUMNS) + input_cols)
    df = df.dropna(subset=input_cols).reset_index(drop=True)
    time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    block = (np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
             if len(time) > 1 else np.zeros(len(time), dtype=np.int64))
    masks = {c: KSTAR_CES_Dataset._stuck_repeat_mask(
        df[c].to_numpy(dtype=np.float64), block) for c in TARGET_COLUMNS}
    keep = df[list(TARGET_COLUMNS)].notna().any(axis=1).to_numpy()

    fa = dataset.file_arrays[file_idx]
    fa_time = fa[:, int(dataset._column_slices["time"])].astype(np.float64)
    kept_time = time[keep]
    if kept_time.shape != fa_time.shape or not np.allclose(kept_time, fa_time, atol=1e-6):
        raise SystemExit(
            f"FATAL: row alignment failed for {Path(path).name}: "
            f"{kept_time.shape} vs {fa_time.shape} -- the held flag cannot be trusted")
    return {c: masks[c][keep] for c in TARGET_COLUMNS}


def analyze_seed(seed, rng):
    out_dir = BASELINE_OUT[seed]
    npz_path = out_dir / NPZ_NAME
    if not npz_path.exists():
        raise SystemExit(f"FATAL: missing {npz_path} -- run rerun_compare_stuck0.py first")
    d = np.load(npz_path)

    manifest = json.loads((SPLIT_SRC[seed] / "split_manifest.json").read_text(encoding="utf-8"))
    test_files = set(manifest["test_files"])

    dataset = KSTAR_CES_Dataset(
        data_dir=Path(os.getenv("CES_DATA_DIR", REPO_ROOT / "data")),
        window_size=WINDOW_SIZE,
        temporal_subset_augmentation=False,
        drop_stuck_targets=False,   # held KEPT -- explicit, never inherited
    )
    wanted = {i for i, p in enumerate(dataset.valid_files) if Path(p).name in test_files}
    cache = {i: held_masks(dataset, i) for i in sorted(wanted)}

    rec = {"n_test_files": len(wanted), "per_target": {}}
    for t in TARGETS:
        shot, row = d[f"{t}_shot"], d[f"{t}_row"]
        se_model, se_pchip = d[f"{t}_se_model"], d[f"{t}_se_pchip"]
        se_persist = d[f"{t}_se_persistence"]
        is_peak = d[f"{t}_is_peak"].astype(bool)
        unknown = sorted(set(int(s) for s in shot) - wanted)
        if unknown:
            raise SystemExit(f"FATAL: npz references non-test shots {unknown[:5]}")
        is_held = np.array([bool(cache[int(s)][t][int(r)]) for s, r in zip(shot, row)])

        blk = {
            "n": int(len(shot)),
            "held_fraction": float(is_held.mean()),
            "peak_fraction": float(is_peak.mean()),
            # the crosstab itself: is the held rate different inside peaks?
            "crosstab": {f"{a}_{b}": int(((is_peak == (a == "peak")) &
                                         (is_held == (b == "held"))).sum())
                         for a, b in CELLS},
            "held_fraction_in_peak": float(is_held[is_peak].mean()) if is_peak.any() else None,
            "held_fraction_in_bulk": float(is_held[~is_peak].mean()) if (~is_peak).any() else None,
            "cells": {},
            "subsets": {},
        }
        for a, b in CELLS:
            sel = (is_peak == (a == "peak")) & (is_held == (b == "held"))
            blk["cells"][f"{a}_{b}"] = {
                **_bootstrap(shot[sel], se_model[sel], se_pchip[sel], rng),
                "vs_persistence": _bootstrap(shot[sel], se_model[sel], se_persist[sel], rng),
            }
        for label, sel in (("all", np.ones(len(shot), bool)),
                           ("genuine_only", ~is_held),
                           ("held_only", is_held)):
            blk["subsets"][label] = {
                **_bootstrap(shot[sel], se_model[sel], se_pchip[sel], rng),
                "vs_persistence": _bootstrap(shot[sel], se_model[sel], se_persist[sel], rng),
            }
        rec["per_target"][t] = blk
    return rec


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"npz": NPZ_NAME, "treatment": "stuck0 (held KEPT in scoring)",
           "bootstrap_resamples": B, "seed": BOOTSTRAP_SEED,
           "held_definition": "dataset._stuck_repeat_mask on the raw CSV (loader's own rule)",
           "per_seed": {}}
    for seed in SEEDS:
        out["per_seed"][seed] = analyze_seed(seed, rng)
        print(f"[heldpeak] seed {seed} done")

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}\n")

    for t in TARGETS:
        print(f"=== {t} ===")
        print(f"{'seed':>5} {'held%':>7} {'held%@peak':>11} {'held%@bulk':>11} "
              f"{'skill(all)':>11} {'skill(genuine)':>15} {'skill(held)':>12}")
        for seed in SEEDS:
            b = out["per_seed"][seed]["per_target"][t]
            s = b["subsets"]
            def f(k):
                e = s[k]
                return f"{e['skill_point']:+.3f}{'*' if e['pass'] else ' '}"
            print(f"{seed:>5} {b['held_fraction']*100:>6.1f}% "
                  f"{(b['held_fraction_in_peak'] or 0)*100:>10.1f}% "
                  f"{(b['held_fraction_in_bulk'] or 0)*100:>10.1f}% "
                  f"{f('all'):>11} {f('genuine_only'):>15} {f('held_only'):>12}")
        print()
    print("* = shot-clustered 95% CI excludes 0 (PR4). skill is vs PCHIP.")


if __name__ == "__main__":
    main()
