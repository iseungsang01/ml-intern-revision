"""Tests for the high-variability (peak) CES reconstruction evaluation (Step 8 / AC1, AC2, AC5, AC7, AC8, AC9).

CPU-only, synthetic numpy (mirrors tests/test_baselines_interpolation.py style). No
real data, no GPU, no matplotlib. Run from repo root: ``python -m pytest -q``.

This file is built in two waves (the work is split across teammates):
  * READY NOW (depend only on existing code I own/read): the AC8 in-loop real-path
    guards (automl_agent_loop.py + a stubbed peak_analysis) and the AC7 sign-convention
    guard (bootstrap_compare._bootstrap only).
  * AFTER peak_analysis.py LANDS: detection (two families), example selection +
    small-pool guard, and the dual-sufficiency guard, which must import and run the
    real peak_analysis functions. Those are written defensively with pytest.importorskip
    so the suite stays green before the module exists and auto-activates once it does.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

CES_DIR = Path(__file__).resolve().parents[1] / "ces_prediction"
sys.path.insert(0, str(CES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIME, TGT = 0, 1  # column layout for the synthetic single-target arrays


def _arr(times, vals):
    return np.array(list(zip(times, vals)), dtype=np.float32)


def _peak_block(skill, ci, *, pass_, n_rows=512, n_shots=22,
                insufficient_shots=False, insufficient_rows=False):
    """Build a per-target peak block matching the documented interface:
    headline at peak["input_only"]["ces_activity"]."""
    return {
        "input_only": {
            "ces_activity": {
                "peak_skill_vs_pchip": skill,
                "peak_skill_ci95": ci,
                "pass": pass_,
                "insufficient_shots": insufficient_shots,
                "insufficient_rows": insufficient_rows,
                "n_peak_rows": n_rows,
                "n_peak_shots": n_shots,
            }
        }
    }


def _install_peak_stub(run_peak_eval):
    """Install a stub peak_analysis module exposing run_peak_eval, and drop any
    already-imported real module so the lazy import inside automl picks up the stub.
    Returns the previous module (or None) for restoration."""
    prev = sys.modules.get("peak_analysis")
    mod = types.ModuleType("peak_analysis")
    mod.run_peak_eval = run_peak_eval
    sys.modules["peak_analysis"] = mod
    return prev


def _restore_peak(prev):
    if prev is not None:
        sys.modules["peak_analysis"] = prev
    else:
        sys.modules.pop("peak_analysis", None)


# ===========================================================================
# AC8 — loop integration, REAL in-loop path (depends only on automl_agent_loop.py)
# ===========================================================================

def test_ac8_merge_populates_eval_report_in_memory():
    """run_evaluation's peak merge sets eval_report["per_target"][name]["peak"]
    IN MEMORY by reusing the val-split npz (here stubbed via run_peak_eval)."""
    import automl_agent_loop as L

    peak = {
        "CES_TI": _peak_block(0.1234, [0.05, 0.20], pass_=True),
        "CES_VT": _peak_block(-0.02, [-0.12, 0.08], pass_=False),
    }
    prev = _install_peak_stub(lambda env: peak)
    try:
        agent = L.EvaluationAgent(run_smoke_test=False)
        eval_report = {
            "per_target": {
                "CES_TI": {"n": 100, "skill_vs_persistence": 0.5},
                "CES_VT": {"n": 80, "skill_vs_persistence": 0.2},
            }
        }
        agent._merge_peak_block({"CES_SPLIT_TAG": "val"}, eval_report)
        assert eval_report["per_target"]["CES_TI"]["peak"] is peak["CES_TI"]
        assert eval_report["per_target"]["CES_VT"]["peak"] is peak["CES_VT"]
    finally:
        _restore_peak(prev)


def test_ac8_build_briefing_contains_peak_fields():
    """build_briefing(result) returns a STRING containing the per-target peak
    skill + 95% CI + verdict tag for the headline (input-only CES-activity)."""
    import automl_agent_loop as L

    eval_report = {
        "per_target": {
            "CES_TI": {
                "n": 100, "skill_vs_persistence": 0.5, "r2_vs_mean": 0.4,
                "rmse_model": 1.2,
                "peak": _peak_block(0.1234, [0.05, 0.20], pass_=True,
                                    n_rows=512, n_shots=22),
            },
            "CES_VT": {
                "n": 80, "skill_vs_persistence": 0.2, "r2_vs_mean": 0.1,
                "rmse_model": 3.4,
                "peak": _peak_block(-0.02, [-0.12, 0.08], pass_=False,
                                    n_rows=300, n_shots=18),
            },
        }
    }
    result = {
        "eval": eval_report, "score": 0.2, "error_stage": None, "error": None,
        "final_train_loss": 0.5, "final_val_loss": 0.6,
    }
    tracker = L.ExperimentTracker(_tmp_state_dir())
    briefing = tracker.build_briefing(1, result, "kept")

    assert isinstance(briefing, str)
    # Headline peak skill + CI for both targets reach the briefing string.
    assert "CES_TI PEAK" in briefing
    assert "CES_VT PEAK" in briefing
    assert "skill_vs_pchip=0.1234" in briefing
    assert "CI95=[0.0500, 0.2000]" in briefing
    assert "PASS" in briefing            # CES_TI significant
    assert "n.s." in briefing            # CES_VT not significant
    assert "n_shots=22" in briefing


def test_ac8_briefing_omits_peak_line_when_absent():
    """No peak block (e.g. peak eval failed / None) => the briefing simply omits
    the peak line; it must NOT crash and must NOT emit a PEAK line."""
    import automl_agent_loop as L

    eval_report = {"per_target": {"CES_TI": {"n": 100, "skill_vs_persistence": 0.5,
                                             "peak": None}}}
    result = {"eval": eval_report, "score": 0.5, "error_stage": None, "error": None,
              "final_train_loss": None, "final_val_loss": None}
    tracker = L.ExperimentTracker(_tmp_state_dir())
    briefing = tracker.build_briefing(2, result, "kept")
    assert "PEAK" not in briefing


def test_ac8_merge_is_graceful_on_peak_failure():
    """A peak failure (exception in run_peak_eval) is non-fatal: it logs, leaves
    no peak key set (stays absent/None), and does NOT raise or flip error_stage."""
    import automl_agent_loop as L

    def boom(env):
        raise RuntimeError("comparison_errors_val.npz missing")

    prev = _install_peak_stub(boom)
    try:
        agent = L.EvaluationAgent(run_smoke_test=False)
        eval_report = {"per_target": {"CES_TI": {"n": 100, "skill_vs_persistence": 0.5}}}
        # Must not raise.
        agent._merge_peak_block({}, eval_report)
        assert eval_report["per_target"]["CES_TI"].get("peak") is None
    finally:
        _restore_peak(prev)


def test_ac8_gate_unchanged_with_vs_without_peak_keys():
    """comparison_skill (the keep/discard gate) reads ONLY comparison_report and
    is byte-identical whether or not peak keys are present anywhere."""
    import automl_agent_loop as L

    comp = {"per_target": {"CES_TI": {"skill_vs_pchip": 0.3},
                           "CES_VT": {"skill_vs_pchip": 0.1}}}
    comp_with_peak = {"per_target": {
        "CES_TI": {"skill_vs_pchip": 0.3, "peak": {"input_only": {"ces_activity": {"peak_skill_vs_pchip": 0.9}}}},
        "CES_VT": {"skill_vs_pchip": 0.1, "peak": {"input_only": {"ces_activity": {"peak_skill_vs_pchip": -0.5}}}},
    }}
    s1 = L.comparison_skill(comp)
    s2 = L.comparison_skill(comp_with_peak)
    assert s1 == s2 == pytest.approx(0.2)


def test_ac8_run_peak_eval_real_path_from_synthetic_npz(tmp_path):
    """AC8 REAL-path integration: drive the actual peak_analysis.run_peak_eval over a
    synthetic comparison_errors_val.npz (no stub), so npz-key renames and return-shape
    regressions are caught (the stub tests above cannot see them).

    Writes the byte-identical npz keys run_peak_eval reads for EACH TARGET_NAMES entry
    ({name}_shot/_se_model/_se_pchip/_se_linear/_is_peak), with enough distinct peak
    shots/rows to clear both sufficiency floors (N_MIN_PEAK_SHOTS=15, N_MIN_PEAK_ROWS=200)
    and se_model < se_pchip on every peak row so peak_skill_vs_pchip > 0 and pass=True.
    Pure numpy/CPU; run_peak_eval forces split_tag='val' and resolves the npz from
    CES_OUTPUT_DIR."""
    import peak_analysis as pa

    assert pa.N_MIN_PEAK_SHOTS == 15 and pa.N_MIN_PEAK_ROWS == 200

    rng = np.random.default_rng(20240623)
    n_shots = 30                       # > 15 distinct peak shots
    rows_per = 20
    shot = np.repeat(np.arange(n_shots), rows_per)
    n = shot.size
    # is_peak True on a clear subset spanning EVERY shot: 12 of 20 rows/shot -> 360
    # peak rows across all 30 shots (>= 200 rows, >= 15 shots).
    is_peak = (np.arange(n) % rows_per) < 12
    assert int(is_peak.sum()) > pa.N_MIN_PEAK_ROWS
    assert int(np.unique(shot[is_peak]).size) > pa.N_MIN_PEAK_SHOTS

    se_pchip = np.abs(rng.normal(2.0, 0.3, size=n))
    se_linear = se_pchip * 1.1
    se_model = se_pchip.copy()
    # Model clearly + consistently better on the peak subset (mild jitter so the
    # bootstrap CI is non-degenerate but still strictly above 0).
    se_model[is_peak] = se_pchip[is_peak] * rng.uniform(0.20, 0.30, size=int(is_peak.sum()))

    npz_path = tmp_path / "comparison_errors_val.npz"
    arrays = {}
    for name in pa.TARGET_NAMES:
        arrays[f"{name}_shot"] = shot
        arrays[f"{name}_se_model"] = se_model
        arrays[f"{name}_se_pchip"] = se_pchip
        arrays[f"{name}_se_linear"] = se_linear
        arrays[f"{name}_is_peak"] = is_peak
    np.savez(npz_path, **arrays)

    env = {"CES_OUTPUT_DIR": str(tmp_path)}
    result = pa.run_peak_eval(env)

    assert set(result.keys()) == set(pa.TARGET_NAMES)
    for name in pa.TARGET_NAMES:
        ca = result[name]["input_only"]["ces_activity"]
        for key in ("peak_skill_vs_pchip", "peak_skill_ci95", "pass",
                    "n_peak_rows", "n_peak_shots"):
            assert key in ca, f"{name} missing headline key {key!r}"
        assert len(ca["peak_skill_ci95"]) == 2
        assert ca["n_peak_rows"] == int(is_peak.sum())
        assert ca["n_peak_shots"] == n_shots
        assert ca["peak_skill_vs_pchip"] > 0.0   # se_model < se_pchip on peak rows
        assert ca["peak_skill_ci95"][0] > 0.0    # CI lower bound strictly positive
        assert ca["pass"] is True                # sufficient shots+rows + significant


def _tmp_state_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="automl_state_")


# ===========================================================================
# AC7 — ablation sign convention (depends only on bootstrap_compare._bootstrap)
# ===========================================================================

def test_ac7_sign_convention_paired_diff_lower_bound_is_signal():
    """The AC7 ablation significance signal is paired_diff_ci95[0] > 0, where the
    pinned argument order is _bootstrap(shot, se_model=se_no_fast, se_base=se_full).
    paired_diff = SE_no_fast - SE_full; > 0 means removing fast diagnostics hurts
    (full is better). With this order _bootstrap's own skill/pass test the OPPOSITE
    direction and MUST be disregarded for the ablation arm."""
    import bootstrap_compare as BC

    rng = np.random.default_rng(0)
    n_shots = 60
    rows_per_shot = 20
    shot = np.repeat(np.arange(n_shots), rows_per_shot)
    n = shot.size
    # Construct se_no_fast > se_full per sample across many shots: removing fast
    # diagnostics consistently increases error (the model genuinely uses them).
    se_full = np.abs(rng.normal(1.0, 0.1, size=n))
    se_no_fast = se_full + np.abs(rng.normal(0.5, 0.05, size=n))  # strictly larger

    res = BC._bootstrap(shot, se_no_fast, se_full, np.random.default_rng(12345))

    # The SIGNAL: paired-diff lower bound strictly above 0 => full better, significant.
    assert res["paired_diff_ci95"][0] > 0.0
    assert res["paired_diff_point"] > 0.0
    # With this argument order, _bootstrap's skill = 1 - MSE_no_fast/MSE_full < 0
    # and its pass = (skill_lo > 0) is False -- it tests the WRONG direction, so the
    # ablation code must NOT key off it. Pin that here so a regression is caught.
    assert res["skill_point"] < 0.0
    assert res["pass"] is False


def test_ac7_sign_convention_null_when_no_fast_equals_full():
    """When the ablated arm matches the full arm (fast diagnostics add nothing),
    the paired-diff CI must straddle 0 -- i.e. no significant drop is reported."""
    import bootstrap_compare as BC

    rng = np.random.default_rng(1)
    n_shots = 60
    shot = np.repeat(np.arange(n_shots), 20)
    n = shot.size
    base = np.abs(rng.normal(1.0, 0.1, size=n))
    # Symmetric jitter around equality => no systematic full-vs-no_fast advantage.
    jitter = rng.normal(0.0, 0.05, size=n)
    se_full = base + np.maximum(jitter, 0)
    se_no_fast = base + np.maximum(-jitter, 0)

    res = BC._bootstrap(shot, se_no_fast, se_full, np.random.default_rng(12345))
    lo, hi = res["paired_diff_ci95"]
    assert lo <= 0.0 <= hi  # straddles zero -> not significant (correct null)


# ===========================================================================
# AC2 — peak detection (two families, input-only is headline)
# ===========================================================================

def test_ac2_detection_flat_series_has_no_peaks():
    """A flat (constant CES) series has zero slope and zero neighbor variance =>
    no headline (i-a ces_activity) peaks. _top_pct_threshold returns +inf on a
    degenerate (no-spread) signal, so nothing is selected."""
    import peak_analysis as pa

    times = np.arange(40) * 0.01            # one contiguous block, 10 ms grid
    vals = np.full(40, 3.0, dtype=np.float64)  # perfectly flat
    a = _arr(times, vals)
    masks = pa.detect_peak_rows_input_only(a, TIME, TGT)
    assert masks["ces_activity"].dtype == bool
    assert masks["ces_activity"].shape == (40,)
    assert not masks["ces_activity"].any()  # nothing flagged on a flat series

    # Family (ii) observed-target: a flat series also has no |dCES| spread.
    mask_t, caveat = pa.detect_peak_rows_target(a, TIME, TGT)
    assert not mask_t.any()
    assert "SELECTION BIAS" in caveat


def test_ac2_detection_input_only_family_and_target_family():
    """A series with a localized large excursion is flagged by BOTH families in its
    block; a SEPARATE flat block stays quiet; the input-only family never reads
    CES[idx] (poisoning the target row's own value must not change its i-a flag).

    Note: the input-only neighbor variance is over the WHOLE contiguous block
    (build_neighbor_set spans the block, not a sliding window), so "high-local-
    activity neighborhood" is a per-block property. To get a genuinely quiet region
    it must be in a DIFFERENT block (separated by a >= 0.5 s gap)."""
    import peak_analysis as pa

    # Block A (rows 0-9): perfectly flat -> quiet. Gap 1.0 s. Block B (rows 10-19):
    # contains a sharp single-row spike -> high-activity neighborhood. A single-row
    # spike (vs a symmetric triangle) makes the spike row itself carry by far the
    # largest |dCES|, so Family (ii)'s z-test flags the extremum row directly.
    times = np.concatenate([np.arange(10) * 0.01, 1.0 + np.arange(10) * 0.01])
    vals = np.full(20, 1.0, dtype=np.float64)
    vals[15] = 12.0  # lone spike against a flat block-B baseline
    a = _arr(times, vals)

    masks = pa.detect_peak_rows_input_only(a, TIME, TGT)
    # Excursion block flagged (headline i-a); separate flat block stays quiet.
    assert masks["ces_activity"][10:20].any()
    assert not masks["ces_activity"][0:10].any()

    # Family (ii) flags the observed extremum at idx using CES[idx].
    mask_t, _ = pa.detect_peak_rows_target(a, TIME, TGT)
    assert mask_t[15]


def test_ac2_input_only_signal_excludes_target_row_value():
    """Non-circularity guarantee (the headline's defense): a row's OWN input-only
    SIGNAL (neighbor-bracket slope + local CES-neighbor variance) is computed from
    build_neighbor_set, which EXCLUDES row_index -- so it never reads CES[idx].

    We test the guarantee at the signal level (not the thresholded flag): poisoning
    ONLY row idx's own CES value leaves row idx's own slope/neighbor-variance
    unchanged. (The downstream boolean flag also depends on a file-global top-pct
    threshold, so poisoning a row CAN shift other rows' thresholds; the per-row
    SIGNAL is the precise, defensible non-circular quantity.)"""
    import peak_analysis as pa
    import baselines_interpolation as B

    times = np.arange(40) * 0.01
    vals = 1.0 + np.sin(np.arange(40) * 0.4)  # smooth, varied, all observed
    a = _arr(times, vals)

    for probe in (5, 20, 33):
        nt0, nv0, tt0 = B.build_neighbor_set(a, TIME, TGT, probe)
        slope0 = pa._bracket_slope(nt0, nv0, tt0)
        var0 = float(np.var(nv0))

        poisoned = a.copy()
        poisoned[probe, TGT] = 999.0  # poison ONLY this row's own value
        nt1, nv1, tt1 = B.build_neighbor_set(poisoned, TIME, TGT, probe)
        slope1 = pa._bracket_slope(nt1, nv1, tt1)
        var1 = float(np.var(nv1))

        assert 999.0 not in nv1                       # own value never a neighbor
        assert slope1 == pytest.approx(slope0)        # signal unaffected by CES[idx]
        assert var1 == pytest.approx(var0)


def test_ac2_detection_invariant_to_row_reorder_within_block():
    """Each row's input-only flag depends only on its in-block neighbor set, not on
    storage order. Permuting rows within a single contiguous block (carrying their
    time+value together) and un-permuting must reproduce the same per-row flags."""
    import peak_analysis as pa

    rng = np.random.default_rng(7)
    times = np.arange(30) * 0.01
    vals = 1.0 + rng.normal(0, 0.05, size=30)
    vals[15] = 8.0  # one clear local excursion
    vals[14] = 4.0
    vals[16] = 4.0
    a = _arr(times, vals)
    base = pa.detect_peak_rows_input_only(a, TIME, TGT)["ces_activity"]

    # Permute rows (build_neighbor_set sorts by time internally, so a contiguous
    # block reconstructs identically); map base flags through the permutation.
    perm = rng.permutation(30)
    a_perm = a[perm]
    permuted = pa.detect_peak_rows_input_only(a_perm, TIME, TGT)["ces_activity"]
    # permuted[i] corresponds to original row perm[i].
    np.testing.assert_array_equal(permuted, base[perm])

    # Determinism: identical inputs -> identical output (no RNG in detection).
    again = pa.detect_peak_rows_input_only(a, TIME, TGT)["ces_activity"]
    np.testing.assert_array_equal(again, base)


def test_ac2_detection_robust_to_nan_interior_rows():
    """NaN interior CES rows must not crash detection and must not be flagged as
    observed-target peaks; the input-only family treats them as missing neighbors."""
    import peak_analysis as pa

    times = np.arange(30) * 0.01
    vals = 1.0 + np.zeros(30)
    vals[10] = np.nan
    vals[11] = np.nan
    vals[20] = 9.0  # an excursion elsewhere
    vals[19] = 4.0
    vals[21] = 4.0
    a = _arr(times, vals)

    masks = pa.detect_peak_rows_input_only(a, TIME, TGT)  # must not raise
    assert masks["ces_activity"].shape == (30,)
    assert masks["ces_activity"].dtype == bool  # bool mask, all defined

    mask_t, _ = pa.detect_peak_rows_target(a, TIME, TGT)
    # Rows whose own CES is NaN can never be observed-target peaks.
    assert not mask_t[10]
    assert not mask_t[11]


def test_ac2_detection_no_leakage_across_half_second_gap():
    """A spike isolated in a LATER contiguous block must not flag rows in an
    EARLIER block separated by a >= 0.5 s gap (block boundary respected)."""
    import peak_analysis as pa

    # Block A: rows 0-9 (flat). Gap of 1.0 s. Block B: rows 10-19 with a spike.
    times = np.concatenate([np.arange(10) * 0.01, 1.0 + np.arange(10) * 0.01])
    vals = np.full(20, 1.0, dtype=np.float64)
    vals[15] = 20.0   # large spike entirely inside block B
    vals[14] = 10.0
    vals[16] = 10.0
    a = _arr(times, vals)

    masks = pa.detect_peak_rows_input_only(a, TIME, TGT)
    # Block A (the flat, pre-gap block) must be entirely unflagged: the spike in B
    # cannot leak across the >=0.5 s boundary.
    assert not masks["ces_activity"][0:10].any()
    # Block B near the spike should be flagged.
    assert masks["ces_activity"][10:20].any()


# ===========================================================================
# AC5 — example selection (exact indices, deterministic, small-pool guard)
# ===========================================================================

def test_ac5_example_selection_exact_indices():
    """For fixed per-sample skills, select_examples returns the exact expected
    worst/median/best indices using the documented index math, with a stable
    (skill, shot_id, row_index) tie-break."""
    import peak_analysis as pa

    # 9 samples, distinct skills => deterministic ascending order is identity.
    skills = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    shots = np.arange(9)
    rows = np.arange(9) * 10
    res = pa.select_examples(skills, shots, rows, k=3)

    assert res["k_effective"] == 3
    assert res["bands_overlap_guard"] is None
    assert res["order"] == list(range(9))  # already ascending

    worst = [d["idx"] for d in res["bands"]["worst"]]
    best = [d["idx"] for d in res["bands"]["best"]]
    median = [d["idx"] for d in res["bands"]["median"]]
    # n=9, k=3, m=n//2=4, m-(k//2)=4-1=3 => median = order[3:6]
    assert worst == [0, 1, 2]
    assert best == [6, 7, 8]
    assert median == [3, 4, 5]
    # descriptors carry shot_id/row_index/skill for the selected indices.
    assert res["bands"]["best"][-1]["skill"] == pytest.approx(3.0)
    assert res["bands"]["worst"][0]["row_index"] == 0


def test_ac5_example_selection_stable_tie_break():
    """All-equal skills => order falls back to (shot_id, row_index); selection is
    deterministic rather than dependent on input order."""
    import peak_analysis as pa

    skills = np.zeros(6)
    shots = np.array([2, 2, 1, 1, 0, 0])
    rows = np.array([5, 1, 9, 2, 7, 3])
    res = pa.select_examples(skills, shots, rows, k=2)
    # ascending (shot, row): (0,3),(0,7),(1,2),(1,9),(2,1),(2,5)
    # original indices:        5,    4,    3,    2,    1,    0
    assert res["order"] == [5, 4, 3, 2, 1, 0]


def test_ac5_small_pool_overlap_guard():
    """When n_peak < 3*k the effective k is reduced (k_eff = max(1, n//3)) so the
    worst/median/best bands never silently overlap, and a guard note is recorded."""
    import peak_analysis as pa

    skills = np.array([-1.0, 0.0, 1.0, 2.0])  # n=4, k_req=3 -> 4 < 9
    shots = np.arange(4)
    rows = np.arange(4)
    res = pa.select_examples(skills, shots, rows, k=3)

    assert res["k_effective"] == max(1, 4 // 3)  # == 1
    assert res["bands_overlap_guard"] == {"n_peak": 4, "k_requested": 3, "k_effective": 1}
    # bands disjoint with k_eff=1: worst=order[0:1], best=order[3:4], median=order[2:3]
    all_idx = (
        [d["idx"] for d in res["bands"]["worst"]]
        + [d["idx"] for d in res["bands"]["median"]]
        + [d["idx"] for d in res["bands"]["best"]]
    )
    assert len(all_idx) == len(set(all_idx))  # no overlap across bands

    # Empty pool: all bands empty, no crash.
    empty = pa.select_examples(np.array([]), np.array([]), np.array([]), k=3)
    assert empty["bands"] == {"worst": [], "median": [], "best": []}


# ===========================================================================
# AC4 — dual-sufficiency guard (shot-bound, forces pass=False)
# ===========================================================================

def test_ac4_dual_sufficiency_guard_forces_pass_false_below_min_shots():
    """Below N_MIN_PEAK_SHOTS the headline block sets insufficient_shots and forces
    pass=False even when the model is strictly better on every sample (so the point
    skill is large positive). Above both floors with a clear win, pass=True."""
    import peak_analysis as pa

    assert pa.N_MIN_PEAK_SHOTS == 15
    assert pa.N_MIN_PEAK_ROWS == 200

    rng = np.random.default_rng(3)

    # --- too FEW shots (5) but many rows: insufficient_shots -> pass False ---
    n_shots_small = 5
    rows_per = 60  # 300 rows total (>= N_MIN_PEAK_ROWS) but only 5 shots
    shot = np.repeat(np.arange(n_shots_small), rows_per)
    n = shot.size
    se_pchip = np.abs(rng.normal(2.0, 0.2, size=n))
    se_model = se_pchip * 0.25  # model strictly + clearly better
    block = pa._peak_skill_block(shot, se_model, se_pchip)
    assert block["n_peak_shots"] == 5
    assert block["insufficient_shots"] is True
    assert block["insufficient_rows"] is False  # 300 >= 200
    assert block["pass"] is False               # forced false by shot guard
    assert block["peak_skill_vs_pchip"] > 0.5   # point estimate still reported

    # --- too FEW rows (below 200) even with enough shots: insufficient_rows ---
    n_shots_ok = 20
    shot2 = np.repeat(np.arange(n_shots_ok), 5)  # 100 rows < 200
    n2 = shot2.size
    se_pchip2 = np.abs(rng.normal(2.0, 0.2, size=n2))
    se_model2 = se_pchip2 * 0.25
    block2 = pa._peak_skill_block(shot2, se_model2, se_pchip2)
    assert block2["n_peak_shots"] == 20
    assert block2["insufficient_rows"] is True
    assert block2["pass"] is False

    # --- sufficient shots AND rows with a clear win -> pass True ---
    n_shots_big = 30
    shot3 = np.repeat(np.arange(n_shots_big), 20)  # 600 rows, 30 shots
    n3 = shot3.size
    se_pchip3 = np.abs(rng.normal(2.0, 0.2, size=n3))
    se_model3 = se_pchip3 * 0.25
    block3 = pa._peak_skill_block(shot3, se_model3, se_pchip3)
    assert block3["insufficient_shots"] is False
    assert block3["insufficient_rows"] is False
    assert block3["peak_skill_ci95"][0] > 0.0
    assert block3["pass"] is True


def test_ac4_empty_peak_subset_is_safe():
    """A peak subset with zero rows must not crash and reports pass=False."""
    import peak_analysis as pa

    block = pa._peak_skill_block(np.array([]), np.array([]), np.array([]))
    assert block["n_peak_rows"] == 0
    assert block["pass"] is False
    assert block["insufficient_shots"] is True
