"""B.73 (PREREGISTRATION_B73.md with Addendum 1): shared paths, constants and helpers.

Every stage caches under data/.b73/ (git-ignored) and writes whole files, then renames, so a stopped process resumes.
Run every stage from the repo root with `py`.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import re  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
DATA = REPO / "data"
CACHE = DATA / ".b73"
PAPER = REPO / "docs" / "paper"
OUT_JSON, OUT_MD, CHECKS_MD = PAPER / "b73_targets.json", PAPER / "b73_targets.md", PAPER / "b73_checks.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B73.md"

SEED = 20261173                    # bootstrap streams (registration section 10)
TREE_SEED = 20261116               # run_b37.gbr
N_BOOT = 20_000
REPEATS = (2, 3, 4)
K = 5
CUT_EV = 3000.0
SEG_GAP = 0.5                      # blocks cut at a time step of 0.5 s (seq_data / dataset STUCK_GAP_SECONDS)
CLUSTER_GAP = 3                    # shot cluster: consecutive delivered shot numbers differ by at most 3
V2_SEEDS = (1, 2, 3)
V4_FIT_A, V4_FIT_B, V4_TEST_FROM = 31991, 32310, 32312
BLOCKS = (("up to 31991", -1, 31991), ("32002-32310", 32002, 32310), ("from 32312", 32312, 10 ** 9))
NB, NE = 9, 4                      # BES, ECEI channels of the grid (columns 3 .. 3+NB+NE)
TARGETS = ("F", "N1", "L3", "L5", "L10")
T0_ = _time.time()


def log(m):
    print(f"[{_time.time() - T0_:7.0f}s] {m}", flush=True)


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def save_npz(path, **arrays):
    """Write whole file, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def shot_of(name):
    return int(re.search(r"(\d+)", str(name)).group(1))


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------ grid (cached)
def grid():
    """Concatenated cut, held-free grid of seq_data.load_grid_files (stage_grid.py)."""
    return load_npz(CACHE / "grid.npz")


def rows():
    return load_npz(CACHE / "rows.npz")


def clf_rows():
    return load_npz(CACHE / "clf_rows.npz")


# ------------------------------------------------------------------------------------------ clusters
def clusters_of_files(names):
    """Cluster id per file (all delivered files, sorted by shot): a new cluster starts where the shot number
    exceeds the previous delivered shot number by more than CLUSTER_GAP."""
    shots = np.array([shot_of(n) for n in names])
    order = np.argsort(shots, kind="stable")
    lab = np.zeros(len(shots), np.int64)
    lab[order] = np.concatenate(([0], np.cumsum(np.diff(shots[order]) > CLUSTER_GAP)))
    return lab, shots


def skill(a, b):
    return 1.0 - np.sum(a) / np.sum(b)


# ------------------------------------------------------------------------------------------ per-file causal inputs
def file_history(tm, ti, seg, rows):
    """Causal CES-history inputs of the given rows of one file (rows must carry a measured value; arrays file-local).
    Returns hold, age (ms), the five scalars of the last ten measured values of the same block (mean, sd, last - mean,
    last increment, number present; NaN when fewer than three), and M5 (median of the last up to five measured values of
    the same block).  Only values of rows before each row enter."""
    meas = np.isfinite(ti)
    obs = np.flatnonzero(meas)
    pos_of = {int(g): p for p, g in enumerate(obs)}
    yo, so, to = ti[obs], seg[obs], tm[obs]
    n = len(rows)
    hold, age, m5 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    th = np.full((n, 5), np.nan)
    for q, g in enumerate(rows):
        p = pos_of[int(g)]
        prev = [u for u in range(max(p - 10, 0), p) if so[u] == so[p]]
        if prev:
            hold[q] = yo[prev[-1]]
            age[q] = (tm[g] - to[prev[-1]]) * 1000.0
            m5[q] = np.median(yo[prev[-5:]])
        if len(prev) >= 3:
            v = yo[prev]
            th[q] = (v.mean(), v.std(), v[-1] - v.mean(), v[-1] - v[-2], len(prev))
    return hold, age, th, m5


def later_values(tm, ti, seg, gi, fid, fidx_rows, time_tol=0.003):
    """Values of the grid rows of the same block at 10 j ms after each row (j = 1..10; NaN where that row does not exist
    or carries no value).  gi: global grid indices of the rows; fid: file index of every grid row."""
    n = len(gi)
    later = np.full((n, 10), np.nan)
    for d in range(1, 11):
        g2 = gi + d
        ok = g2 < len(tm)
        g2c = np.minimum(g2, len(tm) - 1)
        ok &= (fid[g2c] == fidx_rows) & (seg[g2c] == seg[gi])
        dt = tm[g2c] - tm[gi]
        j = np.rint(dt / 0.010).astype(np.int64)
        ok &= (j >= 1) & (j <= 10) & (np.abs(dt - 0.010 * j) < time_tol)
        idx = np.flatnonzero(ok)
        later[idx, j[idx] - 1] = ti[g2c[idx]]
    return later


def targets_from_later(y, later):
    nm5 = np.isfinite(later[:, :5]).sum(1)
    nm10 = np.isfinite(later).sum(1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l5 = np.nanmedian(later[:, :5], 1)
        l10 = np.nanmedian(later, 1)
    return {"F": np.asarray(y, np.float64).copy(),
            "N1": later[:, 0].copy(),
            "L3": np.where(np.isfinite(later[:, :3]).all(1), later[:, :3].mean(1), np.nan),
            "L5": np.where(nm5 >= 3, l5, np.nan),
            "L10": np.where(nm10 >= 5, l10, np.nan)}


def window_feats(x, seg):
    """run_b33.window_feats (copied): level, departure from the mean of the preceding ten rows, their sd, one-row
    increment; NaN unless row i and row i - 10 are in the same block (seg is global and monotone)."""
    n = len(x)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    out = np.full((n, 4), np.nan)
    i = np.arange(10, n)
    ok = seg[i] == seg[i - 10]
    i = i[ok]
    m = (c1[i] - c1[i - 10]) / 10.0
    v = np.maximum((c2[i] - c2[i - 10]) / 10.0 - m * m, 0.0)
    out[i, 0], out[i, 1], out[i, 2], out[i, 3] = x[i], x[i] - m, np.sqrt(v), x[i] - x[i - 1]
    return out
