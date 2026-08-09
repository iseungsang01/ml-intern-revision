"""Full-grid sequence data for the seq-LSTM experiment.

The key reframing (승상님, 2026-07-30): the per-sample pipeline dropped rows with no
observed target, which made the time axis irregular and forced the window/augmentation
machinery. But the fast diagnostics are 100% dense on the 10 ms grid -- so keep EVERY
input-complete row (labelled or not) as sequence context, and handle sparse labels in
the LOSS with a per-target observed mask instead of in the dataset with row dropping.

Per file we produce contiguous blocks (split where the time delta >= 0.5 s, the same
constant the main dataset uses). Per step the features are strictly causal:

  [ bes(9) ecei(4) mc(2) z-scored | log1p(dt_prev) |
    per target (TI, VT): last-observed value (z, from rows < t), log1p(staleness s),
    has-any-past-observation flag ]                                  -> 22 channels

Targets are z-scored with NaN -> 0 and a parallel observed mask (loss-side masking).
Normalization stats are fit train-files-only and NaN-aware, mirroring dataset.py.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ces_prediction/
from dataset import KSTAR_CES_Dataset, TIME_COLUMN, TARGET_COLUMNS, STUCK_GAP_SECONDS  # noqa: E402

FAST_PREFIXES = ("BES_", "ECEI_", "MC")
N_FEATURES = 22
# BES(9) + ECEI(4) + MC(2). The remaining 7 channels are dt plus the per-target
# (carried, staleness, has-observation) triples -- the "slow" state the V_rot branch of
# model_seq_v2 is restricted to. Kept as a named constant so the routing slice and the
# feature layout can never drift apart silently.
N_FAST_CHANNELS = 15


def infer_columns(csv_path):
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    bes = [c for c in header if c.startswith("BES_")]
    ecei = [c for c in header if c.startswith("ECEI_")]
    mc = [c for c in header if c.startswith("MC")]
    return bes, ecei, mc


def load_grid_files(data_dir, drop_stuck_targets):
    """name -> float32 array (rows, [time, TI, VT, bes..., ecei..., mc...]).

    Rows are kept whenever the INPUTS are complete (they always are -- the fast
    diagnostics have no missing values); target columns keep their NaNs so that
    unlabelled rows still provide sequence context. `drop_stuck_targets` NaNs
    held (bit-identical forward-filled) target values exactly like dataset.py.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("s*.csv"))
    if not files:
        raise SystemExit(f"FATAL: no shot CSVs under {data_dir} -- real data required.")
    bes, ecei, mc = infer_columns(files[0])
    usecols = [TIME_COLUMN, *TARGET_COLUMNS, *bes, *ecei, *mc]
    n_fast = len(bes) + len(ecei) + len(mc)

    out = {}
    for fp in files:
        df = pd.read_csv(fp, usecols=usecols)[usecols]
        df = df.dropna(subset=[TIME_COLUMN, *bes, *ecei, *mc]).reset_index(drop=True)
        if df.empty:
            continue
        if drop_stuck_targets:
            time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
            if time.shape[0] >= 2:
                block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
                for col in TARGET_COLUMNS:
                    vals = df[col].to_numpy(dtype=np.float64, copy=True)
                    stuck = KSTAR_CES_Dataset._stuck_repeat_mask(vals, block)
                    if stuck.any():
                        vals[stuck] = np.nan
                        df[col] = vals
        out[fp.name] = df.to_numpy(dtype=np.float32)
    if not out:
        raise SystemExit("FATAL: no usable shot files after loading.")
    return out, {"bes": len(bes), "ecei": len(ecei), "mc": len(mc), "n_fast": n_fast}


def fit_stats(grid, dims, train_names):
    """Train-files-only, NaN-aware normalization stats (dataset.py conventions)."""
    b, e, m = dims["bes"], dims["ecei"], dims["mc"]
    groups = {"bes": [], "ecei": [], "mc": [], "target": []}
    for name in train_names:
        arr = grid.get(name)
        if arr is None:
            continue
        groups["target"].append(arr[:, 1:3])
        groups["bes"].append(arr[:, 3:3 + b])
        groups["ecei"].append(arr[:, 3 + b:3 + b + e])
        groups["mc"].append(arr[:, 3 + b + e:3 + b + e + m])
    if not groups["target"]:
        raise SystemExit("FATAL: none of the manifest train files are present on disk.")
    return {g: KSTAR_CES_Dataset._channel_stats(a) for g, a in groups.items()}


def build_blocks(arr, dims, stats, per_shot_norm=False):
    """One file -> list of blocks {x (L,22), y (L,2), mask (L,2), time (L,)}.

    `per_shot_norm` z-scores the fast diagnostics with THIS shot's own statistics instead
    of the train-file-only global ones (§8s). Targets always keep the global stats: a
    per-shot target scale would make the physical-unit inverse transform shot-dependent
    and break every baseline comparison.
    """
    time = arr[:, 0].astype(np.float64)
    tgt = arr[:, 1:3].astype(np.float64)
    fast = arr[:, 3:3 + dims["n_fast"]].astype(np.float64)

    if per_shot_norm:
        mean = fast.mean(axis=0)
        std = fast.std(axis=0)
        std[std < 1e-6] = 1.0
    else:
        mean = np.concatenate([stats[g]["mean"] for g in ("bes", "ecei", "mc")]).astype(np.float64)
        std = np.concatenate([stats[g]["std"] for g in ("bes", "ecei", "mc")]).astype(np.float64)
    fast_z = (fast - mean) / std
    t_mean = stats["target"]["mean"].astype(np.float64)
    t_std = stats["target"]["std"].astype(np.float64)
    tgt_z = (tgt - t_mean) / t_std

    bounds = [0]
    for i in range(1, len(time)):
        d = time[i] - time[i - 1]
        if d >= STUCK_GAP_SECONDS or d <= 0:
            bounds.append(i)
    bounds.append(len(time))

    blocks = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        L = b - a
        if L < 2:
            continue
        x = np.zeros((L, N_FEATURES), dtype=np.float32)
        x[:, :dims["n_fast"]] = fast_z[a:b]
        dt_prev = np.zeros(L)
        dt_prev[1:] = time[a + 1:b] - time[a:b - 1]
        x[:, dims["n_fast"]] = np.log1p(dt_prev)

        base = dims["n_fast"] + 1
        obs = ~np.isnan(tgt[a:b])
        y = np.nan_to_num(tgt_z[a:b], nan=0.0).astype(np.float32)
        for t in range(2):
            carried = np.zeros(L)
            stale = np.zeros(L)
            has = np.zeros(L)
            last_v, last_t = 0.0, None
            for i in range(L):
                # features at i use observations from rows < i only (strictly causal)
                if last_t is not None:
                    carried[i] = last_v
                    stale[i] = np.log1p(time[a + i] - last_t)
                    has[i] = 1.0
                if obs[i, t]:
                    last_v = tgt_z[a + i, t]
                    last_t = time[a + i]
            x[:, base + 3 * t + 0] = carried
            x[:, base + 3 * t + 1] = stale
            x[:, base + 3 * t + 2] = has
        blocks.append({
            "x": x, "y": y, "mask": obs.astype(np.float32),
            "time": arr[a:b, 0].copy(),  # float32, bit-comparable to dataset arrays
        })
    return blocks
