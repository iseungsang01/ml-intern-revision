"""High-variability ("peak") CES reconstruction evaluation, examples & causes.

This module *layers* on top of the existing interpolation-comparison machinery
(`compare_baselines.py` + `bootstrap_compare.py`) WITHOUT retraining or changing
the model. It asks a sharper question than the global thesis result:

  Does the same uniformly-trained model reconstruct held-out real CES values
  better than conventional interpolation ESPECIALLY in high-local-activity
  (high-variability) neighborhoods?

See `.omc/plans/ces-peak-recon-plan.md` for the authoritative plan (AC1-AC9).

Two DISTINCT peak families (never merge them):

- **Family (i) INPUT-DEFINED (headline, non-circular).** A timestep is flagged
  using ONLY the no-leakage neighbor/input set that EXCLUDES `row_index`
  (`baselines_interpolation.build_neighbor_set`). The target's own realized CES
  value is NEVER read, so a model win here is a fair test of the thesis claim.
  Two SEPARATE sub-signals (reported independently):
    * (i-a) `ces_activity` -- large neighbor-bracket slope across `row_index`
      and/or high local CES-neighbor variance. Measures high CES variability.
      THIS is the headline sub-signal written to the npz by `compare_baselines.py`.
      It selects high-local-activity NEIGHBORHOODS (a conservative regional
      proxy), NOT pointwise extrema.
    * (i-b) `fast_activity` -- high local BES/ECEI input variance in the window.
      Measures high DIAGNOSTIC activity (a different claim); reported separately.
- **Family (ii) OBSERVED-TARGET-DEFINED (secondary, caveated).** Flags a local
  extremum / large |dCES| at `idx` USING `CES[idx]`. Carries an explicit
  selection-bias caveat: selecting peaks by deviation-from-neighbors mechanically
  penalizes neighbor-smooth interpolation, so this number is optimistic and is
  NOT the thesis headline.

Detection is pure numpy, deterministic (no RNG), and operates strictly WITHIN
contiguous time blocks (consecutive `time` delta < `GAP_SECONDS`), mirroring
`dataset.py` / `baselines_interpolation.py` block logic so no peak signal ever
bridges a discontinuity the model's window also never crosses.
"""

import json
import os
from pathlib import Path

import numpy as np

import baselines_interpolation as B
import bootstrap_compare as BC

GAP_SECONDS = B.GAP_SECONDS  # 0.5 s contiguous-block boundary (single source)
TARGET_NAMES = ("CES_TI", "CES_VT")
BOOTSTRAP_SEED = 12345  # mirrors bootstrap_compare.BOOTSTRAP_SEED (deterministic)

# Dual sufficiency guard for a valid shot-clustered peak CI. The BINDING gate is
# the distinct-peak-SHOT count (the bootstrap resamples whole shots, so cluster
# count -- not row count -- governs percentile-CI stability); the peak-ROW count
# is secondary. Below either floor we emit insufficient_* and force pass=False
# (never crash). Explicitly NOT the >50 ROW guard in bootstrap_compare.py:96.
N_MIN_PEAK_SHOTS = 15   # binding
N_MIN_PEAK_ROWS = 200   # secondary

# ---------------------------------------------------------------------------
# Pinned, conservative peak-detection defaults (logged into JSON; the sensitivity
# sweep is in THESIS_RESULTS.md). No RNG anywhere -- detection is fully deterministic.
# ---------------------------------------------------------------------------
PEAK_PARAMS = {
    # Family (i-a) CES-neighbor activity (headline, input-only).
    "ces_activity": {
        "slope_z": 2.5,        # |z-score of neighbor-bracket slope| threshold
        "neigh_var_pct": 0.10, # top fraction by local CES-neighbor variance
        "min_neighbors": 2,    # need >=2 observed neighbors to define activity
    },
    # Family (i-b) fast-diagnostic activity (separate, input-only).
    "fast_activity": {
        "bes_ecei_var_pct": 0.10,  # top fraction by local BES/ECEI window variance
    },
    # Family (ii) observed-target peaks (secondary, reads CES[idx]).
    "observed_target": {
        "diff_z": 2.5,       # |z-score of |dCES|| threshold
        "roll_window": 5,    # rolling-std window (rows)
        "top_pct": 0.10,     # top fraction by rolling std
    },
    "gap": GAP_SECONDS,
}


def _block_id_array(times, gap=GAP_SECONDS):
    """Integer block label per row: increments whenever the gap to the previous
    row is >= `gap`. NaN times start a new block (and are never grouped)."""
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    block = np.zeros(n, dtype=np.int64)
    if n == 0:
        return block
    cur = 0
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if not np.isfinite(dt) or dt >= gap:
            cur += 1
        block[i] = cur
    return block


def _top_pct_threshold(values, pct):
    """Lower threshold so that the top `pct` fraction of FINITE values is >= it.

    Deterministic; uses the (1-pct) quantile. Non-finite values never selected.
    Returns +inf when there are no finite values OR when the distribution has no
    spread (max == min) -- a flat/degenerate signal is not a "peak" of anything,
    so it must select nothing rather than the whole array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.inf
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:  # no spread -> not a meaningful top-pct selection
        return np.inf
    q = float(np.clip(1.0 - pct, 0.0, 1.0))
    return float(np.quantile(finite, q))


def _robust_z(values):
    """Per-array z-score of FINITE entries using mean/std over finite values.

    Non-finite entries map to 0 (never flagged). Zero/degenerate std -> all 0."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out
    mu = float(np.mean(values[finite]))
    sd = float(np.std(values[finite]))
    if not np.isfinite(sd) or sd <= 0.0:
        return out
    out[finite] = (values[finite] - mu) / sd
    return out


def _bracket_slope(times, values, target_time):
    """Local slope of observed CES neighbors across `target_time`, in CES/s.

    Prefers the bracketing (immediately-past, immediately-future) observed
    neighbors so a smooth-but-fast ramp is flagged at interior points (the
    documented "high-local-activity neighborhood, not pointwise extremum"
    behavior). Falls back to a 2-point past slope, else NaN.
    """
    if times.size < 2:
        return np.nan
    order = np.argsort(times)
    t = times[order]
    v = values[order]
    past = t < target_time
    fut = t > target_time
    if np.any(past) and np.any(fut):
        i0 = int(np.flatnonzero(past)[-1])  # latest past
        i1 = int(np.flatnonzero(fut)[0])    # earliest future
        dt = t[i1] - t[i0]
        if dt > 0:
            return float((v[i1] - v[i0]) / dt)
    # no future bracket: 2 most-recent past points
    if np.sum(past) >= 2:
        tp = t[past][-2:]
        vp = v[past][-2:]
        dt = tp[1] - tp[0]
        if dt > 0:
            return float((vp[1] - vp[0]) / dt)
    return np.nan


def detect_peak_rows_input_only(
    file_array,
    time_col,
    target_col,
    *,
    bes_cols=None,
    ecei_cols=None,
    gap=GAP_SECONDS,
    slope_z=None,
    neigh_var_pct=None,
    bes_ecei_var_pct=None,
    min_neighbors=None,
):
    """Family (i) INPUT-DEFINED peak detection -- the HEADLINE, non-circular.

    Classifies each row using ONLY the no-leakage neighbor/input set that
    EXCLUDES `row_index` (never reads ``CES[idx]``). Operates strictly within
    contiguous blocks (``time`` delta < `gap`).

    Returns a dict with TWO DISTINCT boolean masks (do NOT merge them), each of
    shape ``(len(file_array),)``:
      - ``"ces_activity"`` (i-a): large neighbor-bracket slope across the row
        AND/OR high local CES-neighbor variance. Headline sub-signal (high CES
        variability). Selects high-local-activity NEIGHBORHOODS, not pointwise
        extrema.
      - ``"fast_activity"`` (i-b): high local BES/ECEI input variance in the
        contiguous block. Measures high DIAGNOSTIC activity (separate claim).

    Deterministic; thresholds are computed per file_array over its finite
    candidate rows. Robust to NaN interior rows; invariant to row reordering
    within a block (each row's signal depends only on its in-block neighbor set,
    not on storage order). No leakage across ``>= gap`` boundaries.
    """
    p_a = PEAK_PARAMS["ces_activity"]
    p_b = PEAK_PARAMS["fast_activity"]
    slope_z = p_a["slope_z"] if slope_z is None else slope_z
    neigh_var_pct = p_a["neigh_var_pct"] if neigh_var_pct is None else neigh_var_pct
    min_neighbors = p_a["min_neighbors"] if min_neighbors is None else min_neighbors
    bes_ecei_var_pct = (
        p_b["bes_ecei_var_pct"] if bes_ecei_var_pct is None else bes_ecei_var_pct
    )

    n = len(file_array)
    ces_activity = np.zeros(n, dtype=bool)
    fast_activity = np.zeros(n, dtype=bool)
    if n == 0:
        return {"ces_activity": ces_activity, "fast_activity": fast_activity}

    times_all = file_array[:, time_col].astype(np.float64)
    block = _block_id_array(times_all, gap)

    # --- (i-a) CES-neighbor activity: per-row slope + local neighbor variance ---
    slope = np.full(n, np.nan, dtype=np.float64)
    neigh_var = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        nt, nv, tt = B.build_neighbor_set(file_array, time_col, target_col, i, gap)
        if nv.size < min_neighbors:
            continue
        slope[i] = _bracket_slope(nt, nv, tt)
        neigh_var[i] = float(np.var(nv))

    slope_z_arr = _robust_z(np.abs(slope))
    var_thr = _top_pct_threshold(neigh_var, neigh_var_pct)
    ces_activity = (slope_z_arr >= slope_z) | (
        np.isfinite(neigh_var) & (neigh_var >= var_thr)
    )

    # --- (i-b) fast-diagnostic activity: local BES/ECEI window variance ---
    feat_cols = []
    if bes_cols is not None:
        feat_cols.extend(int(c) for c in np.asarray(bes_cols).ravel())
    if ecei_cols is not None:
        feat_cols.extend(int(c) for c in np.asarray(ecei_cols).ravel())
    if feat_cols:
        feats = file_array[:, feat_cols].astype(np.float64)  # (n, F)
        fast_var = np.full(n, np.nan, dtype=np.float64)
        # local variance over the contiguous block containing each row.
        for b in np.unique(block):
            sel = np.flatnonzero(block == b)
            if sel.size < 2:
                continue
            blk = feats[sel, :]
            # mean over channels of the per-channel in-block variance, evaluated
            # per row as squared deviation from the block mean (high when a row
            # sits in a locally-active stretch of the fast diagnostics).
            mu = np.nanmean(blk, axis=0)
            dev = blk - mu[None, :]
            row_energy = np.nanmean(dev * dev, axis=1)
            fast_var[sel] = row_energy
        fast_thr = _top_pct_threshold(fast_var, bes_ecei_var_pct)
        fast_activity = np.isfinite(fast_var) & (fast_var >= fast_thr)

    return {"ces_activity": ces_activity, "fast_activity": fast_activity}


def detect_peak_rows_target(
    file_array,
    time_col,
    target_col,
    *,
    gap=GAP_SECONDS,
    diff_z=None,
    roll_window=None,
    top_pct=None,
):
    """Family (ii) OBSERVED-TARGET peak detection -- SECONDARY, caveated.

    Flags a row as a peak using the realized observed ``CES[idx]``: large
    |dCES| across the row (z-scored) OR high local rolling std. Operates within
    contiguous blocks; rows whose own CES is NaN are never flagged.

    Returns ``(mask: bool[n], caveat: str)``. The caveat MUST be carried into the
    JSON/report: selecting peaks by deviation-from-neighbors mechanically
    penalizes neighbor-smooth interpolation, so the resulting skill is optimistic
    and is NOT the thesis headline.
    """
    p = PEAK_PARAMS["observed_target"]
    diff_z = p["diff_z"] if diff_z is None else diff_z
    roll_window = p["roll_window"] if roll_window is None else roll_window
    top_pct = p["top_pct"] if top_pct is None else top_pct

    n = len(file_array)
    mask = np.zeros(n, dtype=bool)
    caveat = (
        "SELECTION BIAS: Family (ii) peaks are chosen using the observed CES[idx]"
        " (deviation from neighbors), which mechanically penalizes neighbor-smooth"
        " interpolation. This skill is an optimistic upper bound and is NOT the"
        " thesis headline (headline = input-only Family (i-a) ces_activity)."
    )
    if n == 0:
        return mask, caveat

    times_all = file_array[:, time_col].astype(np.float64)
    ces = file_array[:, target_col].astype(np.float64)
    block = _block_id_array(times_all, gap)

    abs_diff = np.full(n, np.nan, dtype=np.float64)
    roll_std = np.full(n, np.nan, dtype=np.float64)
    half = max(1, int(roll_window) // 2)
    for b in np.unique(block):
        sel = np.flatnonzero(block == b)
        for pos, i in enumerate(sel):
            if not np.isfinite(ces[i]):
                continue
            # |dCES| across the row using nearest finite in-block neighbors.
            left = np.nan
            for j in range(pos - 1, -1, -1):
                if np.isfinite(ces[sel[j]]):
                    left = ces[sel[j]]
                    break
            right = np.nan
            for j in range(pos + 1, len(sel)):
                if np.isfinite(ces[sel[j]]):
                    right = ces[sel[j]]
                    break
            diffs = []
            if np.isfinite(left):
                diffs.append(abs(ces[i] - left))
            if np.isfinite(right):
                diffs.append(abs(ces[i] - right))
            if diffs:
                abs_diff[i] = max(diffs)
            # rolling std over a centered in-block window of observed CES.
            lo = max(0, pos - half)
            hi = min(len(sel), pos + half + 1)
            win = ces[sel[lo:hi]]
            win = win[np.isfinite(win)]
            if win.size >= 2:
                roll_std[i] = float(np.std(win))

    diff_z_arr = _robust_z(abs_diff)
    roll_thr = _top_pct_threshold(roll_std, top_pct)
    mask = (diff_z_arr >= diff_z) | (np.isfinite(roll_std) & (roll_std >= roll_thr))
    # only rows with an observed CES[idx] qualify as observed-target peaks.
    mask &= np.isfinite(ces)
    return mask, caveat


# ---------------------------------------------------------------------------
# Step 3 -- peak-subset metric + shot-clustered bootstrap CI, dual guard (AC3/AC4)
# ---------------------------------------------------------------------------


def _rmse_from_se(se):
    """Physical-unit RMSE from per-sample squared errors (already physical)."""
    se = np.asarray(se, dtype=np.float64)
    return float(np.sqrt(np.mean(se))) if se.size else float("nan")


def _skill(se_model, se_base):
    """skill = 1 - MSE_model / MSE_base (positive => model better)."""
    mb = float(np.mean(se_base)) if se_base.size else 0.0
    mm = float(np.mean(se_model)) if se_model.size else 0.0
    return (1.0 - mm / mb) if mb > 0 else float("nan")


def _peak_skill_block(shot_peak, se_model_peak, se_base_peak, *, seed=BOOTSTRAP_SEED):
    """Per-target peak-subset skill + shot-clustered bootstrap CI vs one baseline.

    Reuses `bootstrap_compare._bootstrap` (whole-shot resampling, B=10000) for the
    CI, applies the dual sufficiency guard, and never crashes. Returns a dict with
    point skill, physical RMSEs, the 95% skill CI, paired-diff CI, sufficiency
    flags, and a sufficiency-gated `pass`.
    """
    shot_peak = np.asarray(shot_peak)
    se_model_peak = np.asarray(se_model_peak, dtype=np.float64)
    se_base_peak = np.asarray(se_base_peak, dtype=np.float64)

    n_rows = int(se_model_peak.size)
    n_shots = int(np.unique(shot_peak).size) if n_rows else 0
    insufficient_rows = n_rows < N_MIN_PEAK_ROWS
    insufficient_shots = n_shots < N_MIN_PEAK_SHOTS

    block = {
        "n_peak_rows": n_rows,
        "n_peak_shots": n_shots,
        "peak_skill_vs_pchip": float("nan"),
        "peak_rmse_model": _rmse_from_se(se_model_peak),
        "peak_rmse_pchip": _rmse_from_se(se_base_peak),
        "peak_skill_ci95": [float("nan"), float("nan")],
        "paired_diff_ci95": [float("nan"), float("nan")],
        "insufficient_rows": bool(insufficient_rows),
        "insufficient_shots": bool(insufficient_shots),
        "pass": False,
    }
    if n_rows == 0:
        return block

    block["peak_skill_vs_pchip"] = _skill(se_model_peak, se_base_peak)
    # CI is only meaningful with >= 2 shots; below that leave NaN bounds.
    if n_shots >= 2:
        try:
            res = BC._bootstrap(
                shot_peak, se_model_peak, se_base_peak, np.random.default_rng(seed)
            )
            block["peak_skill_ci95"] = list(res["skill_ci95"])
            block["paired_diff_ci95"] = list(res["paired_diff_ci95"])
            ci_pass = bool(res["skill_ci95"][0] > 0.0)
        except Exception as exc:  # never let a CI failure crash the eval
            block["bootstrap_error"] = str(exc)
            ci_pass = False
    else:
        ci_pass = False

    # Binding gate: sufficiency must hold AND the CI lower bound must exceed 0.
    block["pass"] = bool(
        ci_pass and not insufficient_shots and not insufficient_rows
    )
    return block


def _resolve_output_dir(env):
    """CES_OUTPUT_DIR from `env` (falling back to os.environ, then this dir)."""
    src = env if env is not None else os.environ
    val = src.get("CES_OUTPUT_DIR") if hasattr(src, "get") else None
    if not val:
        val = os.environ.get("CES_OUTPUT_DIR")
    if not val:
        val = str(Path(__file__).resolve().parent)
    return Path(val)


def run_peak_eval(env=None, *, split_tag="val"):
    """Cheap headline peak block per target -- the in-loop AC8 return (no forward pass).

    Loads the val-split `comparison_errors_<split_tag>.npz` that `run_comparison`
    wrote (CES_SPLIT_TAG='val' on a copy of env => `comparison_errors_val.npz` in
    CES_OUTPUT_DIR), masks each target's byte-identical headline SE/shot arrays by
    the npz `{name}_is_peak` flag (Family i-a CES-activity), and computes the
    per-target peak skill + shot-clustered bootstrap CI vs PCHIP (and linear),
    with the dual sufficiency guard.

    Returns ``{name: {"input_only": {"ces_activity": {...}}, "params": {...}}}``
    where the headline block exposes ``peak_skill_vs_pchip``, ``peak_skill_ci95``,
    ``pass``, ``n_peak_rows``/``n_peak_shots``, ``insufficient_*``, physical RMSEs,
    a ``vs_linear`` sub-block, and a ``definition`` note. Degrades gracefully:
    missing npz / missing keys -> ``{}`` (never raises into the loop).
    """
    output_dir = _resolve_output_dir(env)
    npz_path = output_dir / f"comparison_errors_{split_tag}.npz"
    if not npz_path.exists():
        print(f"[peak_analysis] npz not found: {npz_path} -> peak eval skipped.")
        return {}

    data = np.load(npz_path)
    out = {}
    for name in TARGET_NAMES:
        peak_key = f"{name}_is_peak"
        if peak_key not in data.files or f"{name}_se_model" not in data.files:
            continue
        is_peak = np.asarray(data[peak_key], dtype=bool)
        shot = np.asarray(data[f"{name}_shot"])
        se_model = np.asarray(data[f"{name}_se_model"], dtype=np.float64)
        se_pchip = np.asarray(data[f"{name}_se_pchip"], dtype=np.float64)
        se_linear = (
            np.asarray(data[f"{name}_se_linear"], dtype=np.float64)
            if f"{name}_se_linear" in data.files
            else None
        )

        head = _peak_skill_block(shot[is_peak], se_model[is_peak], se_pchip[is_peak])
        head["definition"] = (
            "high-local-activity NEIGHBORHOOD proxy (input-only Family i-a "
            "ces_activity: large neighbor-bracket slope and/or high local "
            "CES-neighbor variance, EXCLUDING CES[idx]); NOT pointwise extrema."
        )
        head["caveat"] = (
            "Non-circularity rests on the per-row SIGNAL, which is target-independent"
            " (excludes row_index / CES[idx]); the boolean FLAG additionally uses a"
            " file-global top-percentile threshold, so a row's flag CAN shift if other"
            " rows change -- the precise defensible quantity is the signal, not the flag."
        )
        if se_linear is not None:
            lin = _peak_skill_block(shot[is_peak], se_model[is_peak], se_linear[is_peak])
            head["vs_linear"] = {
                "peak_skill_vs_linear": lin["peak_skill_vs_pchip"],
                "peak_rmse_linear": lin["peak_rmse_pchip"],
                "peak_skill_ci95": lin["peak_skill_ci95"],
                "pass": lin["pass"],
            }
        out[name] = {
            "input_only": {"ces_activity": head},
            "params": PEAK_PARAMS["ces_activity"],
            "split": split_tag,
            "mnar_caveat": "skill measured on observed CES points only (MNAR optimistic bound)",
        }
    return out


# ---------------------------------------------------------------------------
# Step 5 -- best/median/worst example selection, small-pool guarded (AC5)
# ---------------------------------------------------------------------------


def select_examples(per_sample_skill, shot_ids, row_indices, k=3):
    """Pick worst/median/best peak examples by per-sample skill (deterministic).

    Per-sample skill = ``SE_pchip - SE_model`` (positive => model beats PCHIP).
    Examples are sorted ASCENDING by the stable composite key
    ``(skill, shot_id, row_index)`` so ties break reproducibly; ``order`` is that
    argsort, ``n = n_peak``, ``k = k_eff``, ``m = n // 2``:

        worst  = order[0:k]            (most negative skill incl. failures)
        best   = order[n-k:n]          (largest skill)
        median = order[m-(k//2) : m-(k//2)+k]

    Small-pool overlap guard (NIT 1): if ``n_peak < 3*k`` the effective k is
    reduced to ``k_eff = max(1, n_peak // 3)`` (so worst/median/best never draw
    from a pool too small to keep the bands disjoint) and a ``bands_overlap_guard``
    note is recorded. Example selection may run BELOW the Step-3 sufficiency floor
    -- examples are illustrative, not a verdict.

    Returns ``{"bands": {"worst": [...], "median": [...], "best": [...]},
    "order": [...], "bands_overlap_guard": {...}|None, "k_effective": int}`` where
    each band entry is a dict ``{idx, shot_id, row_index, skill}`` (idx = position
    in the input peak arrays). All three bands always emitted (possibly empty when
    n_peak == 0).
    """
    skill = np.asarray(per_sample_skill, dtype=np.float64)
    shot_ids = np.asarray(shot_ids)
    row_indices = np.asarray(row_indices)
    n = int(skill.size)
    k_req = int(k)

    guard = None
    k_eff = k_req
    if n < 3 * k_req:
        k_eff = max(1, n // 3)
        guard = {"n_peak": n, "k_requested": k_req, "k_effective": k_eff}

    bands = {"worst": [], "median": [], "best": []}
    if n == 0:
        return {
            "bands": bands,
            "order": [],
            "bands_overlap_guard": guard,
            "k_effective": k_eff,
        }

    # Stable ascending sort by (skill, shot_id, row_index). np.lexsort uses the
    # LAST key as primary, so order keys from least- to most-significant.
    order = np.lexsort((row_indices, shot_ids, skill))
    m = n // 2
    kk = k_eff
    band_idx = {
        "worst": order[0:kk],
        "best": order[n - kk:n],
        "median": order[m - (kk // 2): m - (kk // 2) + kk],
    }

    def _descr(i):
        i = int(i)
        return {
            "idx": i,
            "shot_id": int(shot_ids[i]),
            "row_index": int(row_indices[i]),
            "skill": float(skill[i]),
        }

    for band, idxs in band_idx.items():
        bands[band] = [_descr(i) for i in idxs]

    return {
        "bands": bands,
        "order": [int(i) for i in order],
        "bands_overlap_guard": guard,
        "k_effective": k_eff,
    }


# ---------------------------------------------------------------------------
# Step 4 -- AC7 ablation (flag-gated own forward pass, paired-shot delta) (AC7)
# ---------------------------------------------------------------------------

OOD_ABLATION_CAVEAT = (
    "OOD CONFOUND: the no_fast arm ZEROES BES/ECEI/MC, an out-of-distribution "
    "intervention off the training manifold. A skill drop conflates 'model uses "
    "the fast signal' with 'model was fed OOD zeros'; a single ablation cannot "
    "fully separate them. Reported honestly. A manifold-respecting (noise/"
    "permutation) ablation is a follow-up."
)


def _compare_forward_pass(env, ablate):
    """Reproduce compare_baselines' clean-val forward loop for a given ablation.

    Returns per-target dicts of full-sample arrays in the EXACT compare order
    (same DataLoader, no shuffle, same keep/valid masks), so the ablated arm can
    be paired sample-for-sample to the npz the headline path wrote. The model is
    run with `evaluate.apply_ablation(ablate=...)`; persistence/interp arms always
    use the real data (ablation only zeroes BES/ECEI/MC), mirroring evaluate.py.

    Returns ``{name: {"shot", "row", "valid", "is_peak", "se_model"}}`` (all
    arrays full length, target-major); SE is in physical CES units squared.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    from evaluate import (
        _load_stats,
        _persistence_from_history,
        apply_ablation,
        build_clean_val_subset,
    )
    from model import MultimodalCESPredictor

    src = env if env is not None else os.environ
    def _get(key, default):
        v = src.get(key) if hasattr(src, "get") else None
        return v if v is not None else os.environ.get(key, default)

    root_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(_get("CES_DATA_DIR", str(root_dir / "data")))
    output_dir = _resolve_output_dir(env)
    split_dir = Path(_get("CES_SPLIT_DIR", str(root_dir / "data" / "splits")))
    window_size = int(_get("CES_WINDOW_SIZE", "4"))
    seed = int(_get("CES_SEED", "42"))
    max_val_samples = int(_get("CES_MAX_VAL_SAMPLES", "40000"))
    batch_size = int(_get("CES_BATCH_SIZE", "512"))
    split_tag = "val"  # match run_comparison's CES_SPLIT_TAG copy

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(manifest[f"{split_tag}_files"])

    dataset, eval_indices, _ = build_clean_val_subset(
        data_dir, window_size, stats, eval_files, max_val_samples, seed
    )
    path_to_idx = {p: i for i, p in enumerate(dataset.valid_files)}
    time_col = int(dataset._column_slices["time"])
    target_cols = [int(c) for c in dataset._column_slices["target"]]
    bes_cols = [int(c) for c in dataset._column_slices["bes"]]
    ecei_cols = [int(c) for c in dataset._column_slices["ecei"]]

    methods = ["persistence", "linear", "pchip", "ar_local"]
    if B._HAVE_SKLEARN:
        methods.append("gp")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCESPredictor.from_dataset(dataset, window_size=window_size)
    model.load_state_dict(
        torch.load(output_dir / "weights" / "multimodal_ces.pth", map_location="cpu")
    )
    model.to(device)
    model.eval()

    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    def to_phys(t):
        return (t * target_std + target_mean).numpy()

    loader = DataLoader(Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False)

    model_chunks, target_chunks, keep_chunks = [], [], []
    base_chunks = {m: [] for m in methods}
    shot_chunks, row_chunks = [], []

    peak_cache = {}

    def _ces_activity_mask(file_idx, target_col):
        key = (file_idx, target_col)
        cached = peak_cache.get(key)
        if cached is None:
            fa = dataset.file_arrays[file_idx]
            cached = detect_peak_rows_input_only(
                fa, time_col, target_col, bes_cols=bes_cols, ecei_cols=ecei_cols
            )["ces_activity"]
            peak_cache[key] = cached
        return cached

    with torch.no_grad():
        for batch in loader:
            bes = batch["bes"].to(device, non_blocking=True)
            ecei = batch["ecei"].to(device, non_blocking=True)
            mc = batch["mc"].to(device, non_blocking=True)
            tf = batch["time_features"].to(device, non_blocking=True)
            hist = batch["ces_history"].to(device, non_blocking=True)

            # Persistence baseline always uses the REAL history, independent of ablation.
            persistence, has_obs = _persistence_from_history(hist)
            a_bes, a_ecei, a_mc, a_hist = apply_ablation(ablate, bes, ecei, mc, hist)
            out = model(a_bes, a_ecei, a_mc, tf, a_hist)

            model_chunks.append(to_phys(out.cpu()))
            target_chunks.append(to_phys(batch["target"]))
            keep_chunks.append(((batch["target_mask"] > 0.5) & has_obs.cpu()).numpy())

            persist_phys = to_phys(persistence.cpu())
            files = batch["file"]
            rows = [int(r) for r in batch["row_index"].tolist()]
            b = len(rows)
            arr = {m: np.full((b, 2), np.nan, dtype=np.float64) for m in methods}
            for j in range(b):
                fa = dataset.file_arrays[path_to_idx[files[j]]]
                ri = rows[j]
                for t in range(2):
                    tc = target_cols[t]
                    times, values, target_time = B.build_neighbor_set(fa, time_col, tc, ri)
                    for m in methods:
                        if m == "persistence":
                            arr[m][j, t] = float(persist_phys[j, t])
                        else:
                            arr[m][j, t] = B.PREDICTORS[m](times, values, target_time)
            for m in methods:
                base_chunks[m].append(arr[m])
            shot_chunks.append(np.array([path_to_idx[f] for f in files], dtype=np.int64))
            row_chunks.append(np.array(rows, dtype=np.int64))

    model_phys = np.concatenate(model_chunks)
    target_phys = np.concatenate(target_chunks)
    keep = np.concatenate(keep_chunks)
    base_phys = {m: np.concatenate(base_chunks[m]) for m in methods}
    shot_ids = np.concatenate(shot_chunks)
    row_ids = np.concatenate(row_chunks)

    result = {}
    for t, name in enumerate(TARGET_NAMES):
        valid = keep[:, t].copy()
        for m in methods:
            valid &= ~np.isnan(base_phys[m][:, t])
        tc = target_cols[t]
        is_peak = np.zeros(len(shot_ids), dtype=bool)
        for s in range(len(shot_ids)):
            is_peak[s] = bool(_ces_activity_mask(int(shot_ids[s]), tc)[int(row_ids[s])])
        y = target_phys[valid, t]
        se_model = (model_phys[valid, t] - y) ** 2
        result[name] = {
            "shot": shot_ids[valid],
            "row": row_ids[valid],
            "valid": valid,
            "is_peak": is_peak[valid],
            "se_model": se_model,
            "model_pred": model_phys[valid, t],
            "truth": y,
        }
    return result


def run_ablation_eval(env=None, *, split_tag="val", seed=BOOTSTRAP_SEED):
    """AC7 cause-analysis ablation -- FLAG-GATED (CES_PEAK_ABLATION=1) (AC7).

    Compares the headline FULL arm against a `no_fast` arm (BES/ECEI/MC zeroed) on
    the IDENTICAL input-only peak subset, per target. `se_full` is the byte-
    identical npz `{name}_se_model` (NOT re-forwarded); `se_no_fast` is this
    module's own `no_fast` forward pass over the same samples.

    Pairing guard: the recomputed full-arm `shot`/`is_peak`/`se_model` are asserted
    element-wise equal (`np.array_equal`) to the npz arrays; ANY mismatch fails
    loudly (no silent desync, incl. split-tag mismatch).

    Sign convention (pinned): `_bootstrap(shot, se_model=se_no_fast, se_base=se_full)`
    => paired_diff = SE_no_fast - SE_full. Significance of "removing fast
    diagnostics hurts (full better)" is judged ONLY by ``paired_diff_ci95[0] > 0``.
    `_bootstrap`'s own `skill`/`pass` test the OPPOSITE direction with this
    argument order and MUST be disregarded for the ablation arm.

    Returns ``{name: {"full": {...}, "no_fast": {...}, "paired_diff_point",
    "paired_diff_ci95", "ablation_significant", "ood_caveat", ...}}``.
    """
    output_dir = _resolve_output_dir(env)
    npz_path = output_dir / f"comparison_errors_{split_tag}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"[peak_analysis] ablation requires {npz_path} (run compare_baselines first)."
        )

    data = np.load(npz_path)
    forward = _compare_forward_pass(env, "no_fast")

    out = {}
    for name in TARGET_NAMES:
        if f"{name}_is_peak" not in data.files:
            continue
        npz_shot = np.asarray(data[f"{name}_shot"])
        npz_peak = np.asarray(data[f"{name}_is_peak"], dtype=bool)
        npz_se_full = np.asarray(data[f"{name}_se_model"], dtype=np.float64)

        fp = forward[name]
        # Pairing guard: recomputed full-arm arrays MUST match the npz exactly.
        if not np.array_equal(fp["shot"], npz_shot):
            raise AssertionError(
                f"[peak_analysis] ablation pairing mismatch for {name}: shot arrays "
                f"differ (len {len(fp['shot'])} vs {len(npz_shot)}). Split/index desync."
            )
        if not np.array_equal(fp["is_peak"], npz_peak):
            raise AssertionError(
                f"[peak_analysis] ablation pairing mismatch for {name}: is_peak arrays differ."
            )
        # `valid`/`shot`/`is_peak` are ablation-INDEPENDENT (they depend only on
        # data + target masks + interp arms, never on model output), so the
        # shot+is_peak equality above proves index alignment to the npz order.
        # se_full = npz value (byte-identical headline); se_no_fast = this module's
        # no_fast forward over the SAME valid order (fp["se_model"] here is no_fast SE).
        se_no_fast = fp["se_model"]
        if len(se_no_fast) != len(npz_se_full):
            raise AssertionError(
                f"[peak_analysis] ablation length mismatch for {name}: "
                f"{len(se_no_fast)} != {len(npz_se_full)}"
            )

        peak = npz_peak
        shot_peak = npz_shot[peak]
        se_full_peak = npz_se_full[peak]
        se_no_fast_peak = se_no_fast[peak]

        n_rows = int(se_full_peak.size)
        n_shots = int(np.unique(shot_peak).size) if n_rows else 0
        # Sanity field: did the no_fast forward actually diverge from full? A stuck or
        # duplicated forward leaves se_no_fast == se_full everywhere. CES_VT may
        # LEGITIMATELY be all-equal ([0,0] CI -- the VT head ignores fast inputs), so
        # this is REPORTED (not hard-failed); CES_TI all-equal across targets would be
        # the suspicious signature of a stuck forward.
        forward_varies = bool(np.any(se_no_fast != npz_se_full))
        block = {
            "n_peak_rows": n_rows,
            "n_peak_shots": n_shots,
            "forward_varies": forward_varies,
            "full": {"peak_rmse": _rmse_from_se(se_full_peak)},
            "no_fast": {"peak_rmse": _rmse_from_se(se_no_fast_peak)},
            # skill of each arm vs PCHIP is reported elsewhere; here we report the
            # arm-vs-arm relationship the ablation is about.
            "full_better_skill": _skill(se_full_peak, se_no_fast_peak),
            "paired_diff_point": float("nan"),
            "paired_diff_ci95": [float("nan"), float("nan")],
            "ablation_significant": False,
            "ood_caveat": OOD_ABLATION_CAVEAT,
            "sign_convention": (
                "paired_diff = SE_no_fast - SE_full; significant iff "
                "paired_diff_ci95[0] > 0 (full better). _bootstrap skill/pass DISREGARDED."
            ),
        }
        if n_rows > 0 and n_shots >= 2:
            # Sign convention: se_model=se_no_fast, se_base=se_full.
            res = BC._bootstrap(
                shot_peak, se_no_fast_peak, se_full_peak, np.random.default_rng(seed)
            )
            block["paired_diff_point"] = float(res["paired_diff_point"])
            block["paired_diff_ci95"] = list(res["paired_diff_ci95"])
            # ONLY the diff lower bound > 0 signals "removing fast diagnostics hurts".
            block["ablation_significant"] = bool(res["paired_diff_ci95"][0] > 0.0)
        out[name] = block
    return out


# ---------------------------------------------------------------------------
# Step 6 -- example time-series figures (matplotlib [viz] extra) (AC6)
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Lazy, guarded matplotlib import with the Agg (headless) backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: never opens a window, no plt.show needed
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover - exercised only without [viz]
        raise ImportError(
            "matplotlib is required for peak figures. Install the optional extra: "
            'pip install -e ".[viz]"  (figures are skippable with --no-figures).'
        ) from exc


def plot_example(
    file_array,
    time_col,
    target_col,
    row_index,
    model_pred,
    out_path,
    *,
    title=None,
    gap=GAP_SECONDS,
):
    """Save a CES time-series figure for one peak example (AC6).

    Overlays, over the example's contiguous block (`time` delta < gap):
      - observed truth (scatter, the real CES values incl. the highlighted target),
      - the model prediction at the eval timestep (single marker),
      - PCHIP and linear interpolation curves (from the no-leakage neighbor set,
        EXCLUDING `row_index`, mirroring `baselines_interpolation`),
      - the peak/target timestep highlighted with a vertical line.

    Uses the Agg backend (no `plt.show`); writes `out_path` via `savefig`. Returns
    the written path. Raises a clear ImportError if `[viz]` is not installed.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times_all = file_array[:, time_col].astype(np.float64)
    start, end = B.contiguous_block_bounds(times_all, int(row_index), gap)
    blk = np.arange(start, end + 1)
    bt = times_all[blk]
    bv = file_array[blk, target_col].astype(np.float64)
    obs = np.isfinite(bv)

    # No-leakage interp curves over the block (neighbor set excludes row_index).
    nt, nv, target_time = B.build_neighbor_set(file_array, time_col, target_col, int(row_index), gap)
    grid = np.linspace(bt.min(), bt.max(), 200) if bt.size > 1 else bt
    pchip_curve = np.array([B.predict_pchip(nt, nv, g) for g in grid])
    linear_curve = np.array([B.predict_linear(nt, nv, g) for g in grid])

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.scatter(bt[obs], bv[obs], c="black", s=24, zorder=5, label="observed CES")
    if grid is not bt:
        ax.plot(grid, pchip_curve, "-", color="tab:blue", lw=1.4, label="PCHIP")
        ax.plot(grid, linear_curve, "--", color="tab:green", lw=1.2, label="linear")
    ax.scatter(
        [target_time], [model_pred], marker="*", s=180, color="tab:red",
        zorder=6, label="model prediction",
    )
    # highlight the peak/target timestep.
    ax.axvline(target_time, color="tab:red", alpha=0.3, lw=1.0)
    truth = float(file_array[int(row_index), target_col])
    if np.isfinite(truth):
        ax.scatter([target_time], [truth], facecolors="none", edgecolors="black",
                   s=120, lw=1.5, zorder=7, label="target (truth)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("CES (physical units)")
    ax.set_title(title or f"row {int(row_index)}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI -- assemble peak_metrics.json (+ figures), flag-gated ablation (AC3-7)
# ---------------------------------------------------------------------------


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def main(argv=None):
    """Standalone peak analysis: peak_metrics.json + headline (i-a) figures.

    Reads the val-split `comparison_errors_val.npz` (run compare_baselines.py
    first) for the byte-identical headline SEs, recovers per-sample row indices +
    model predictions via a paired forward pass (`np.array_equal` guard), selects
    best/median/worst examples, writes `peak_metrics.json`, and (unless
    --no-figures / no matplotlib) renders the headline example figures.

    With `CES_PEAK_ABLATION=1`, the AC7 `no_fast` ablation block is added.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Peak (high-variability) CES analysis")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    parser.add_argument("--k", type=int, default=3, help="examples per band (default 3)")
    args = parser.parse_args(argv)

    output_dir = _resolve_output_dir(None)
    split_tag = "val"
    npz_path = output_dir / f"comparison_errors_{split_tag}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} missing. Run `python ces_prediction/compare_baselines.py` first."
        )

    report = {
        "split": split_tag,
        "params": PEAK_PARAMS,
        "sufficiency": {"N_MIN_PEAK_SHOTS": N_MIN_PEAK_SHOTS, "N_MIN_PEAK_ROWS": N_MIN_PEAK_ROWS},
        "per_target": {},
    }

    # Headline cheap peak metrics (reuses the npz; no forward pass).
    peak_block = run_peak_eval(None, split_tag=split_tag)

    # Forward pass to recover row indices + model predictions for examples/figures
    # and to validate alignment to the npz (the same pairing guard as the ablation).
    forward = _compare_forward_pass(None, "none")
    data = np.load(npz_path)

    # Dataset handle for figures (file_arrays + column slices).
    dataset = None
    figures = []
    do_figures = not args.no_figures
    fig_dir = output_dir / "peak_figures"

    for name in TARGET_NAMES:
        if f"{name}_is_peak" not in data.files:
            continue
        t = TARGET_NAMES.index(name)
        npz_shot = np.asarray(data[f"{name}_shot"])
        npz_peak = np.asarray(data[f"{name}_is_peak"], dtype=bool)
        npz_se_model = np.asarray(data[f"{name}_se_model"], dtype=np.float64)
        npz_se_pchip = np.asarray(data[f"{name}_se_pchip"], dtype=np.float64)

        fp = forward[name]
        if not (np.array_equal(fp["shot"], npz_shot) and np.array_equal(fp["is_peak"], npz_peak)):
            raise AssertionError(
                f"[peak_analysis] CLI pairing mismatch for {name}: forward arrays "
                "differ from npz (split/index desync)."
            )

        peak = npz_peak
        # per-sample skill on the peak subset (SE_pchip - SE_model; >0 model better)
        skill_peak = npz_se_pchip[peak] - npz_se_model[peak]
        shot_peak = npz_shot[peak]
        row_peak = fp["row"][peak]
        pred_peak = fp["model_pred"][peak]

        examples = select_examples(skill_peak, shot_peak, row_peak, k=args.k)

        target_block = dict(peak_block.get(name, {}))
        target_block["examples"] = examples

        report["per_target"][name] = target_block

        if do_figures and examples["bands"]["worst"]:
            try:
                plt = _require_matplotlib()  # fail-fast guard; honored below
            except ImportError as exc:
                print(f"[peak_analysis] figures skipped: {exc}")
                do_figures = False
                continue
            if dataset is None:
                # lazy-build the dataset once (figures need file_arrays).
                from evaluate import _load_stats, build_clean_val_subset
                root = Path(__file__).resolve().parents[1]
                ddir = Path(os.environ.get("CES_DATA_DIR", str(root / "data")))
                sdir = Path(os.environ.get("CES_SPLIT_DIR", str(root / "data" / "splits")))
                ws = int(os.environ.get("CES_WINDOW_SIZE", "4"))
                seedv = int(os.environ.get("CES_SEED", "42"))
                mvs = int(os.environ.get("CES_MAX_VAL_SAMPLES", "40000"))
                metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
                stats = _load_stats(metrics)
                manifest = json.loads((sdir / "split_manifest.json").read_text(encoding="utf-8"))
                dataset, _, _ = build_clean_val_subset(
                    ddir, ws, stats, set(manifest["val_files"]), mvs, seedv
                )
            time_col = int(dataset._column_slices["time"])
            target_col = int(dataset._column_slices["target"][t])
            for band, descrs in examples["bands"].items():
                for rank, d in enumerate(descrs):
                    fa = dataset.file_arrays[int(d["shot_id"])]
                    out_png = fig_dir / f"ces_activity_{name}_{band}_{rank}.png"
                    try:
                        plot_example(
                            fa, time_col, target_col, int(d["row_index"]),
                            float(pred_peak[d["idx"]]), out_png,
                            title=f"{name} {band} #{rank} skill={d['skill']:.1f}",
                        )
                        figures.append(str(out_png))
                    except Exception as exc:
                        print(f"[peak_analysis] figure failed ({name}/{band}/{rank}): {exc}")

    # Flag-gated AC7 ablation.
    if _env_flag("CES_PEAK_ABLATION"):
        print("[peak_analysis] CES_PEAK_ABLATION=1 -> running no_fast ablation...")
        try:
            ablation = run_ablation_eval(None, split_tag=split_tag)
            for name, blk in ablation.items():
                if name in report["per_target"]:
                    report["per_target"][name].setdefault("input_only", {})
                    report["per_target"][name]["input_only"]["full"] = {
                        "peak_rmse": blk["full"]["peak_rmse"]
                    }
                    report["per_target"][name]["input_only"]["no_fast"] = blk["no_fast"]
                    report["per_target"][name]["ablation"] = {
                        "paired_diff_point": blk["paired_diff_point"],
                        "paired_diff_ci95": blk["paired_diff_ci95"],
                        "ablation_significant": blk["ablation_significant"],
                        "ood_caveat": blk["ood_caveat"],
                        "sign_convention": blk["sign_convention"],
                    }
        except Exception as exc:
            print(f"[peak_analysis] ablation failed: {exc}")
            report["ablation_error"] = str(exc)

    report["figures"] = figures
    out_path = output_dir / "peak_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPeak metrics saved to {out_path}")
    if figures:
        print(f"{len(figures)} figures written to {fig_dir}")
    for name in TARGET_NAMES:
        b = report["per_target"].get(name, {}).get("input_only", {}).get("ces_activity")
        if b:
            print(
                f"  {name} PEAK skill_vs_pchip={b['peak_skill_vs_pchip']:.4f} "
                f"CI={[round(x, 4) for x in b['peak_skill_ci95']]} "
                f"pass={b['pass']} (rows={b['n_peak_rows']}, shots={b['n_peak_shots']})"
            )
    return report


if __name__ == "__main__":
    main()
