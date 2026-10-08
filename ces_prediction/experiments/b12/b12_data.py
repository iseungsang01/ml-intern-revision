"""B.12 offline gap-filling data (PREREGISTRATION_B12.md §3).

Every arm sees the same thing: a shot file split into contiguous 10 ms blocks, the 15
fast diagnostics (per-shot z-scored, §8s), and the CES targets of which some observed
rows are HIDDEN. Hidden rows are what the arm must fill and what the loss / score use;
the remaining observed rows are context. Hiding is per target, because CES_TI and
CES_VT go missing independently.

Hidden spans are drawn from the empirical missing-run-length distribution of the
TRAIN files (per target), so the test gaps look like the gaps the data actually has
(§8at: V_rot median 1 / mean 18.8 / p90 11 steps). Test and val masks are generated
once with a fixed seed and written to disk; training draws fresh masks every epoch.

Per-step features (N_FEATURES = 32), all computable offline from context only:
  [ fast z (15) | log1p dt_prev |
    per target: ctx value, ctx flag, left ctx value, log1p dist left, has left,
                right ctx value, log1p dist right, has right ]
"left/right" = nearest OTHER context row in the same block. A hidden row's own value
never enters any feature.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "seq"), str(HERE.parents[1])]
from seq_data import load_grid_files, fit_stats  # noqa: E402
from dataset import STUCK_GAP_SECONDS  # noqa: E402

N_FAST = 15
N_SLOW_PER_TARGET = 8
N_FEATURES = N_FAST + 1 + 2 * N_SLOW_PER_TARGET  # 32
HIDE_FRACTION = 0.20  # share of each (file, target)'s observed rows hidden per mask
MAX_SPAN = 400        # steps; caps the heavy V_rot tail at 4 s so one span cannot eat a block
# Label-sparsified TEST sets (H-np, §4.1): in shots with plenty of V_rot labels keep only
# k of them as context and hide the rest. Real low-label shots carry too few points to
# score (the 1-20-label stratum has < 80 hidden V_rot points pooled over 4 splits), so
# the low-context regime is reproduced where the truth is dense. T_i is left untouched.
SPARSE_KS = (0, 3, 10)
SPARSE_MIN_VT = 50


def load_population(data_dir, ti_spike_cut_ev):
    """Held-free grid for one co-primary population (PREREGISTRATION_W2.md §1)."""
    return load_grid_files(data_dir, drop_stuck_targets=True, ti_spike_cut_ev=ti_spike_cut_ev)


def split_blocks(arr):
    """Row-index bounds of contiguous blocks (same rule as seq_data.build_blocks)."""
    time = arr[:, 0].astype(np.float64)
    bounds = [0]
    for i in range(1, len(time)):
        d = time[i] - time[i - 1]
        if d >= STUCK_GAP_SECONDS or d <= 0:
            bounds.append(i)
    bounds.append(len(time))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def missing_run_lengths(grid, names):
    """Per-target empirical lengths of missing runs inside blocks, train files only."""
    runs = {0: [], 1: []}
    for n in names:
        arr = grid[n]
        for a, b in split_blocks(arr):
            for t in range(2):
                miss = np.isnan(arr[a:b, 1 + t])
                i = 0
                while i < len(miss):
                    if miss[i]:
                        j = i
                        while j < len(miss) and miss[j]:
                            j += 1
                        # interior runs only: a run touching the block edge is censored
                        if i > 0 and j < len(miss):
                            runs[t].append(j - i)
                        i = j
                    else:
                        i += 1
    return {t: np.minimum(np.asarray(r, dtype=np.int64), MAX_SPAN) for t, r in runs.items()}


def draw_hidden(arr, run_lengths, rng, fraction=HIDE_FRACTION):
    """(rows, 2) bool: observed rows hidden by spans with empirical lengths."""
    obs = ~np.isnan(arr[:, 1:3])
    hidden = np.zeros_like(obs)
    blocks = split_blocks(arr)
    block_of = np.full(len(arr), -1)
    for k, (a, b) in enumerate(blocks):
        block_of[a:b] = k
    for t in range(2):
        cand = np.flatnonzero(obs[:, t] & (block_of >= 0))
        goal = int(round(fraction * len(cand)))
        tries = 0
        while hidden[:, t].sum() < goal and tries < 10 * max(goal, 1):
            tries += 1
            s = int(rng.choice(cand))
            span = int(rng.choice(run_lengths[t]))
            _, b = blocks[block_of[s]]
            hidden[s:min(s + span, b), t] = True
        hidden[:, t] &= obs[:, t]
    return hidden


def fixed_masks(grid, names, run_lengths, seed, path):
    """Generate once, then reload: the scored population must never drift."""
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        masks = {n: z[n] for n in z.files}
        if sorted(masks) != sorted(names):
            raise SystemExit(f"FATAL: {path} covers different files than the split -- delete it to regenerate.")
        return masks
    masks = {}
    for i, n in enumerate(sorted(names)):
        rng = np.random.default_rng([seed, i])
        masks[n] = draw_hidden(grid[n], run_lengths, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **masks)
    return masks


def sparse_masks(grid, names, k, seed, path):
    """V_rot hidden everywhere except k random kept labels, shots with >= SPARSE_MIN_VT labels."""
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {n: z[n] for n in z.files}
    masks = {}
    for i, n in enumerate(sorted(names)):
        arr = grid[n]
        hidden = np.zeros((len(arr), 2), dtype=bool)
        vt = np.flatnonzero(~np.isnan(arr[:, 2]))
        if len(vt) >= SPARSE_MIN_VT:
            keep = np.random.default_rng([seed, k, i]).choice(vt, size=k, replace=False)
            hidden[vt, 1] = True
            hidden[keep, 1] = False
        masks[n] = hidden
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **masks)
    return masks


def _nearest(ctx, side):
    """Value index of the nearest OTHER context row on one side (-1 if none)."""
    L = len(ctx)
    idx = np.where(ctx, np.arange(L), -1 if side == "left" else L)
    if side == "left":
        prev = np.maximum.accumulate(idx)
        out = np.concatenate(([-1], prev[:-1]))
    else:
        nxt = np.minimum.accumulate(idx[::-1])[::-1]
        out = np.concatenate((nxt[1:], [L]))
        out = np.where(out >= L, -1, out)
    return out


def file_features(arr, hidden, stats):
    """One file -> list of blocks {x (L,32), y (L,2), obs (L,2), hid (L,2), row0}.

    Fast inputs are per-shot z-scored (CES_PER_SHOT_NORM=1, the confirmed protocol);
    targets use the global train-only stats so physical units invert identically for
    every arm and baseline.
    """
    time = arr[:, 0].astype(np.float64)
    fast = arr[:, 3:3 + N_FAST].astype(np.float64)
    mean, std = fast.mean(axis=0), fast.std(axis=0)
    std[std < 1e-6] = 1.0
    fast_z = (fast - mean) / std
    t_mean = stats["target"]["mean"].astype(np.float64)
    t_std = stats["target"]["std"].astype(np.float64)
    tgt_z = (arr[:, 1:3].astype(np.float64) - t_mean) / t_std
    obs = ~np.isnan(arr[:, 1:3])
    ctx = obs & ~hidden

    out = []
    for a, b in split_blocks(arr):
        L = b - a
        tb = time[a:b]
        x = np.zeros((L, N_FEATURES), dtype=np.float32)
        x[:, :N_FAST] = fast_z[a:b]
        dt = np.zeros(L)
        dt[1:] = np.diff(tb)
        x[:, N_FAST] = np.log1p(dt)
        yz = np.nan_to_num(tgt_z[a:b], nan=0.0)
        for t in range(2):
            c = ctx[a:b, t]
            base = N_FAST + 1 + N_SLOW_PER_TARGET * t
            x[:, base] = np.where(c, yz[:, t], 0.0)
            x[:, base + 1] = c
            for k, side in enumerate(("left", "right")):
                j = _nearest(c, side)
                has = j >= 0
                jj = np.where(has, j, 0)
                x[:, base + 2 + 3 * k] = np.where(has, yz[jj, t], 0.0)
                x[:, base + 3 + 3 * k] = np.where(has, np.log1p(np.abs(tb - tb[jj])), 0.0)
                x[:, base + 4 + 3 * k] = has
        out.append({"x": x, "y": yz.astype(np.float32),
                    "obs": obs[a:b].astype(np.float32), "hid": hidden[a:b].astype(np.float32),
                    "row0": a})
    return out


def load_manifest(split_dir):
    m = json.loads((Path(split_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    return list(m["train_files"]), list(m["val_files"]), list(m["test_files"])


def prepare(data_dir, split_dir, ti_spike_cut_ev, mask_dir, seed):
    """Grid, stats, train run-lengths and the fixed val/test masks for one split."""
    grid, dims = load_population(data_dir, ti_spike_cut_ev)
    if dims["n_fast"] != N_FAST:
        raise SystemExit(f"FATAL: expected {N_FAST} fast channels, data has {dims['n_fast']}")
    train, val, test = (([n for n in s if n in grid]) for s in load_manifest(split_dir))
    stats = fit_stats(grid, dims, train)
    runs = missing_run_lengths(grid, train)
    tag = f"cut{int(ti_spike_cut_ev)}_s{seed}"
    val_masks = fixed_masks(grid, val, runs, seed * 1000 + 1, Path(mask_dir) / f"val_{tag}.npz")
    test_masks = fixed_masks(grid, test, runs, seed * 1000 + 2, Path(mask_dir) / f"test_{tag}.npz")
    sparse = {k: sparse_masks(grid, test, k, seed * 1000 + 3, Path(mask_dir) / f"test_sparse{k}_{tag}.npz")
              for k in SPARSE_KS}
    return {"grid": grid, "stats": stats, "runs": runs, "train": train, "val": val, "test": test,
            "val_masks": val_masks, "test_masks": test_masks, "sparse_masks": sparse}
