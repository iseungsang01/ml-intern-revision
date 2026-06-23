"""Conventional interpolation baselines for CES gap-filling comparison.

These are the *bar* the nowcasting model must beat (thesis claim): pure,
deterministic per-target predictors over a sample's filtered per-file array
(`KSTAR_CES_Dataset.file_arrays[file_idx]`).

Fairness / leakage rules (pre-registration PR1/PR2 in
`.omc/plans/ces-interpolation-comparison-consensus.md`):

- A baseline for target `t` at `row_index` may use past+future *observed* CES
  neighbors WITHIN the same contiguous block (consecutive time delta < 0.5 s,
  mirroring `dataset.py` block detection), but NEVER the target's own value at
  `row_index`.
- Acausal methods (linear, pchip, gp) use bracketing past+future neighbors.
  Where no future observed neighbor exists in-block, they FALL BACK to
  persistence (last past observed) so the arm is defined everywhere the model is
  scored. Causal references (persistence, ar_local) use past-only.
- All predictions are in physical CES units (the raw file values), matching the
  denormalized model predictions and targets in `evaluate.py`.
"""

import numpy as np

try:  # scipy provides the canonical monotone-cubic (Fritsch-Carlson) interpolant
    from scipy.interpolate import PchipInterpolator

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only in scipy-less envs
    _HAVE_SCIPY = False

try:  # GP is an optional baseline; skipped+logged by the harness if unavailable
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAVE_SKLEARN = False

GAP_SECONDS = 0.5  # contiguous-block boundary, mirrors dataset.py:271

# Method classification (for reporting causal vs acausal).
ACAUSAL_METHODS = ("linear", "pchip") + (("gp",) if _HAVE_SKLEARN else ())
CAUSAL_METHODS = ("persistence", "ar_local")


def contiguous_block_bounds(times, row_index, gap=GAP_SECONDS):
    """Inclusive [start, end] row span of the contiguous block containing
    row_index (consecutive time deltas < gap). Mirrors dataset.py block logic so
    a baseline never bridges a discontinuity the model's window also never crosses.
    """
    n = len(times)
    start = row_index
    while start > 0 and (times[start] - times[start - 1]) < gap:
        start -= 1
    end = row_index
    while end < n - 1 and (times[end + 1] - times[end]) < gap:
        end += 1
    return start, end


def build_neighbor_set(file_array, time_col, target_col, row_index, gap=GAP_SECONDS):
    """Observed (non-NaN) neighbor times/values for `target_col` within the
    contiguous block containing row_index, EXCLUDING row_index itself.

    Returns (neighbor_times: float64[k], neighbor_values: float64[k],
    target_time: float). The exclusion of row_index is the no-leakage guarantee.
    """
    times_all = file_array[:, time_col].astype(np.float64)
    target_time = float(times_all[row_index])
    start, end = contiguous_block_bounds(times_all, row_index, gap)
    idx = np.arange(start, end + 1)
    idx = idx[idx != row_index]  # never read the target's own row
    vals = file_array[idx, target_col].astype(np.float64)
    keep = ~np.isnan(vals)
    return times_all[idx][keep], vals[keep], target_time


def predict_persistence(times, values, target_time):
    """Most-recent PAST observed value (causal). NaN if no past observation."""
    past = times < target_time
    if not np.any(past):
        return np.nan
    past_times, past_vals = times[past], values[past]
    return float(past_vals[int(np.argmax(past_times))])


def predict_linear(times, values, target_time):
    """Linear interpolation with bracketing past+future neighbors (acausal).
    Falls back to persistence when no future neighbor exists (PR2)."""
    if not (np.any(times < target_time) and np.any(times > target_time)):
        return predict_persistence(times, values, target_time)
    order = np.argsort(times)
    return float(np.interp(target_time, times[order], values[order]))


def predict_pchip(times, values, target_time):
    """Monotone cubic (PCHIP) interpolation with past+future neighbors (acausal).
    Falls back to linear (2 unique pts / no scipy) or persistence (no future)."""
    if not (np.any(times < target_time) and np.any(times > target_time)):
        return predict_persistence(times, values, target_time)
    order = np.argsort(times)
    t, uniq = np.unique(times[order], return_index=True)
    v = values[order][uniq]
    if len(t) < 2:
        return predict_persistence(times, values, target_time)
    if len(t) == 2 or not _HAVE_SCIPY:
        return float(np.interp(target_time, t, v))
    return float(PchipInterpolator(t, v)(target_time))


def predict_ar_local(times, values, target_time):
    """Causal local-slope linear extrapolation from the two most-recent past
    observations (past-only reference). Falls back to persistence with <2 past."""
    past = times < target_time
    if not np.any(past):
        return np.nan
    pt, pv = times[past], values[past]
    order = np.argsort(pt)
    pt, pv = pt[order], pv[order]
    if len(pt) < 2 or pt[-1] == pt[-2]:
        return float(pv[-1])
    slope = (pv[-1] - pv[-2]) / (pt[-1] - pt[-2])
    return float(pv[-1] + slope * (target_time - pt[-1]))


def predict_gp(times, values, target_time):
    """Optional Matern-3/2 GP regression (acausal). Returns NaN if sklearn is
    unavailable; the harness logs and drops it. Falls back to persistence when
    no future neighbor exists (PR2)."""
    if not _HAVE_SKLEARN:
        return np.nan
    if not (np.any(times < target_time) and np.any(times > target_time)):
        return predict_persistence(times, values, target_time)
    order = np.argsort(times)
    t = times[order].reshape(-1, 1)
    v = values[order]
    if len(t) < 2:
        return predict_persistence(times, values, target_time)
    kernel = Matern(length_scale=1e-2, nu=1.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
    gp.fit(t, v)
    return float(gp.predict(np.asarray([[target_time]]))[0])


PREDICTORS = {
    "persistence": predict_persistence,
    "linear": predict_linear,
    "pchip": predict_pchip,
    "ar_local": predict_ar_local,
    "gp": predict_gp,
}


def predict(method, file_array, time_col, target_col, row_index, gap=GAP_SECONDS):
    """Predict physical CES for `target_col` at `row_index` using `method`."""
    times, values, target_time = build_neighbor_set(
        file_array, time_col, target_col, row_index, gap
    )
    return PREDICTORS[method](times, values, target_time)
