"""B.29 (PREREGISTRATION_B29.md): the contracts the batch was registered with.

- `v2h`: seq_v2 whose T_i encoder cannot read a fast channel (no weight exists for one), causal,
  V_rot branch = seq_v2's, 347,970 parameters.
- `mlp22`: memoryless -- the output at row t depends on row t alone; V_rot blind to fast channels.
- the expanding-window standardizer: no row after i influences the standardized value at i
  (bit-identical), while the offline per-discharge standardizer does depend on later rows.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(HERE, "../ces_prediction")))
sys.path.append(os.path.abspath(os.path.join(HERE, "../ces_prediction/experiments/seq")))
sys.path.append(os.path.abspath(os.path.join(HERE, "../ces_prediction/experiments/b29_causal")))

import seq_data  # noqa: E402
from seq_data import N_FAST_CHANNELS, N_FEATURES, build_blocks, expanding_standardize  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402


def _x(L=80, seed=0):
    """A random feature block in seq_data's layout, used only to test wiring (no result)."""
    return torch.randn(2, L, N_FEATURES, generator=torch.Generator().manual_seed(seed))


def test_v2h_contract_shape_params_and_registry():
    torch.manual_seed(0)
    m = SEQ_MODELS["v2h"]().eval()
    assert m.n_params == 347_970                       # backbone 357,570 minus 4 * 160 * 15
    assert SEQ_MODELS["v2"]().n_params - m.n_params == 4 * 160 * N_FAST_CHANNELS
    assert m.lstm_ti.input_size == N_FEATURES - N_FAST_CHANNELS == 7
    out = m(_x(), torch.tensor([80, 37]))
    assert out.shape == (2, 80, 2)
    assert not hasattr(m, "needs_obs") and not hasattr(m, "dist")   # plain (x, lengths) contract


def test_v2h_ti_and_vrot_never_see_fast_channels_and_are_causal():
    torch.manual_seed(0)
    m = SEQ_MODELS["v2h"]().eval()
    x = _x()
    with torch.no_grad():
        base = m(x)
        fast = x.clone()
        fast[..., :N_FAST_CHANNELS] += 7.0             # move every fast channel at every step
        assert torch.equal(m(fast), base)              # both targets: bit-identical
        future = x.clone()
        future[:, 51:] += 5.0
        assert torch.equal(m(future)[:, :51], base[:, :51])
        slow = x.clone()
        slow[:, 10, N_FAST_CHANNELS + 1] += 1.0        # the carried T_i value does matter
        assert not torch.equal(m(slow)[:, 10:, 0], base[:, 10:, 0])


def test_v2h_vrot_branch_is_the_backbone_branch():
    """Same V_rot modules and shapes as seq_v2, so a V_rot difference is optimisation, not design."""
    a, b = SEQ_MODELS["v2h"](), SEQ_MODELS["v2"]()
    for name in ("lstm_vt", "norm_vt", "head_vt", "norm_ti", "head_ti"):
        sa = {k: v.shape for k, v in getattr(a, name).state_dict().items()}
        sb = {k: v.shape for k, v in getattr(b, name).state_dict().items()}
        assert sa == sb, name


def test_mlp22_is_memoryless_and_routes_like_seq_v2():
    torch.manual_seed(0)
    m = SEQ_MODELS["mlp22"]().eval()
    x = _x()
    with torch.no_grad():
        base = m(x, torch.tensor([80, 37]))
        assert base.shape == (2, 80, 2)
        other = x.clone()
        other[:, :40] += 3.0                           # every row except 40..79
        assert torch.equal(m(other)[:, 40:], base[:, 40:])
        fast = x.clone()
        fast[..., :N_FAST_CHANNELS] += 3.0
        out = m(fast)
        assert torch.equal(out[..., 1], base[..., 1])  # V_rot blind to the fast channels
        assert not torch.equal(out[..., 0], base[..., 0])


def _fast(rows=400, seed=1):
    rng = np.random.default_rng(seed)
    return 3.0 + 2.0 * rng.standard_normal((rows, N_FAST_CHANNELS)).cumsum(axis=0) / 10.0


def test_expanding_standardizer_never_reads_a_future_row():
    fast = _fast()
    mu, sd = np.full(N_FAST_CHANNELS, 2.5), np.full(N_FAST_CHANNELS, 1.7)
    z = expanding_standardize(fast, mu, sd)
    for t in (0, 1, 49, 50, 200, 398):
        moved = fast.copy()
        moved[t + 1:] = moved[t + 1:] * -4.0 + 100.0   # rewrite every later row
        assert np.array_equal(expanding_standardize(moved, mu, sd)[:t + 1], z[:t + 1]), t
        if t > 0:                                      # and row t does depend on the rows before it
            past = fast.copy()
            past[:t] += 1.0
            assert not np.array_equal(expanding_standardize(past, mu, sd)[t], z[t])


def test_expanding_standardizer_limits():
    fast = _fast(rows=3000)
    mu, sd = fast[:200].mean(axis=0), fast[:200].std(axis=0)
    # first row: 50 prior rows against one observed row
    u0 = (fast[0] - mu) / sd
    z = expanding_standardize(fast, mu, sd)
    m0 = u0 / 51.0
    assert np.allclose(z[0], (u0 - m0) / np.sqrt((50.0 + u0 ** 2) / 51.0 - m0 ** 2))
    # no prior: the last row is the per-discharge z-score of the whole record
    z0 = expanding_standardize(fast, mu, sd, n0=0)
    assert np.allclose(z0[-1], (fast[-1] - fast.mean(axis=0)) / fast.std(axis=0))
    # a heavy prior: the train-file z-score everywhere
    assert np.allclose(expanding_standardize(fast, mu, sd, n0=1e12), (fast - mu) / sd, atol=1e-6)


def _arr_and_stats(rows=300, seed=2):
    """A file array in load_grid_files' layout [time, TI, VT, fast...] (wiring only)."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((rows, 3 + N_FAST_CHANNELS), dtype=np.float32)
    arr[:, 0] = 0.01 * np.arange(rows)
    arr[150:, 0] += 1.0                                # a >= 0.5 s gap: two blocks
    arr[:, 1:3] = rng.standard_normal((rows, 2))
    arr[rng.random(rows) < 0.5, 1] = np.nan
    arr[:, 3:] = _fast(rows, seed)
    dims = {"bes": 9, "ecei": 4, "mc": 2, "n_fast": N_FAST_CHANNELS}

    def one(n):
        return {"mean": np.zeros(n, dtype=np.float32), "std": np.ones(n, dtype=np.float32)}

    stats = {"bes": one(9), "ecei": one(4), "mc": one(2), "target": one(2)}
    return arr, dims, stats


def _rows(blocks):
    return np.concatenate([b["x"] for b in blocks])


def test_build_blocks_expanding_is_causal_and_offline_per_shot_is_not(monkeypatch):
    monkeypatch.delenv("CES_SEQ_NORM_EXPANDING", raising=False)
    monkeypatch.delenv("CES_SEQ_ZERO_MC", raising=False)
    arr, dims, stats = _arr_and_stats()
    t = 200                                            # inside the second block
    moved = arr.copy()
    moved[t + 1:, 3:] = moved[t + 1:, 3:] * 3.0 - 20.0  # later fast-channel rows only
    exp_a = _rows(build_blocks(arr, dims, stats, per_shot_norm=True, expanding=True))
    exp_b = _rows(build_blocks(moved, dims, stats, per_shot_norm=True, expanding=True))
    assert np.array_equal(exp_a[:t + 1], exp_b[:t + 1])
    assert not np.array_equal(exp_a[t + 1:], exp_b[t + 1:])
    off_a = _rows(build_blocks(arr, dims, stats, per_shot_norm=True, expanding=False))
    off_b = _rows(build_blocks(moved, dims, stats, per_shot_norm=True, expanding=False))
    assert not np.array_equal(off_a[:t + 1, :N_FAST_CHANNELS], off_b[:t + 1, :N_FAST_CHANNELS])
    # the slow channels (all the V_rot branch reads) are the same under every standardizer
    assert np.array_equal(exp_a[:, N_FAST_CHANNELS:], off_a[:, N_FAST_CHANNELS:])
    # the statistics run across the gap: the second block is not a fresh start
    fresh = expanding_standardize(arr[150:, 3:].astype(np.float64), np.zeros(N_FAST_CHANNELS), np.ones(N_FAST_CHANNELS))
    assert not np.allclose(exp_a[150:, :N_FAST_CHANNELS], fresh, atol=1e-3)


def test_expanding_knob_env_default_off_and_requires_per_shot(monkeypatch):
    arr, dims, stats = _arr_and_stats()
    monkeypatch.delenv("CES_SEQ_NORM_EXPANDING", raising=False)
    monkeypatch.delenv("CES_SEQ_ZERO_MC", raising=False)
    assert not seq_data.expanding_norm_enabled()
    assert np.array_equal(_rows(build_blocks(arr, dims, stats, per_shot_norm=True)),
                          _rows(build_blocks(arr, dims, stats, per_shot_norm=True, expanding=False)))
    monkeypatch.setenv("CES_SEQ_NORM_EXPANDING", "1")
    assert seq_data.expanding_norm_enabled()
    assert np.array_equal(_rows(build_blocks(arr, dims, stats, per_shot_norm=True)),
                          _rows(build_blocks(arr, dims, stats, per_shot_norm=True, expanding=True)))
    with pytest.raises(SystemExit):
        build_blocks(arr, dims, stats, per_shot_norm=False)


def test_runner_plan_and_env_pin_every_treatment(monkeypatch):
    monkeypatch.setenv("CES_ABLATE", "no_fast")        # ambient leaks that must not survive
    monkeypatch.setenv("CES_SEQ_NORM_EXPANDING", "1")
    import run_b29
    runs = run_b29.plan(list(run_b29.ARMS), list(run_b29.POPS), list(run_b29.SEEDS), ["split", "camp"], False)
    assert len(runs) == 48 and len(set(runs)) == 48
    assert len({run_b29.arm_dir(*r) for r in runs}) == 48
    assert all(run_b29.arm_dir(*r).name.startswith(".b29_") for r in runs)
    assert all(run_b29.arm_dir(*r, smoke=True).name.startswith(".smoke_b29_") for r in runs)
    want = {"v2h": ("v2h", "1", "0"), "norm0": ("v2", "0", "0"),
            "expand": ("v2", "1", "1"), "mlp22": ("mlp22", "1", "0")}
    for arm, (model, ps, ex) in want.items():
        for pop, cut in (("cut", "3000"), ("incl", "0")):
            env = run_b29.run_env(arm, "split", pop, 7, Path("x"), smoke=False)
            assert "CES_ABLATE" not in env
            assert (env["CES_SEQ_MODEL"], env["CES_PER_SHOT_NORM"], env["CES_SEQ_NORM_EXPANDING"]) == (model, ps, ex)
            assert env["CES_TI_SPIKE_CUT_EV"] == cut
            assert env["CES_DROP_STUCK_TARGETS"] == "1" and env["CES_WINDOW_SIZE"] == "2"
            assert env["CES_MAX_SAMPLES_PER_FILE"] == "500" and env["CES_SEQ_EPOCHS"] == "30"
            assert env["CES_SEED"] == env["CES_INIT_SEED"] == "7"


def test_addendum1_changes_the_epoch_cap_and_nothing_else(monkeypatch):
    """PREREGISTRATION_B29_ADDENDUM1.md: 8 runs, cap 30 -> 80, every other variable arm A's."""
    monkeypatch.setenv("CES_ABLATE", "no_fast")
    monkeypatch.setenv("CES_SEQ_EPOCHS", "5")
    import run_b29
    runs = run_b29.add1_plan(list(run_b29.SEEDS), False)
    assert len(runs) == 8 and len(set(runs)) == 8
    assert [m for m, _ in runs] == ["v2h"] * 4 + ["v2"] * 4          # arm A first: (ii) pairs against it
    names = {run_b29.add1_dir(*r).name for r in runs}
    assert len(names) == 8 and all(n.startswith(".b29a1_") and "_cut_s" in n for n in names)
    registered = {run_b29.arm_dir(*r).name for r in
                  run_b29.plan(list(run_b29.ARMS), list(run_b29.POPS), list(run_b29.SEEDS), ["split", "camp"], False)}
    assert len(registered) == 48 and not (names & registered)        # the 48 registered dirs are untouched
    assert run_b29.add1_plan(list(run_b29.SEEDS), True) == [("v2h", 42), ("v2", 42)]
    for s in run_b29.SEEDS:
        base = run_b29.run_env("v2h", "split", "cut", s, Path("x"), smoke=False)
        assert base["CES_SEQ_EPOCHS"] == "30"                         # the registered env is unchanged
        arm = run_b29.add1_env("v2h", s, Path("x"), smoke=False)
        bb = run_b29.add1_env("v2", s, Path("x"), smoke=False)
        assert "CES_ABLATE" not in arm and "CES_ABLATE" not in bb
        assert {k for k in base if arm[k] != base[k]} == {"CES_SEQ_EPOCHS"} and arm.keys() == base.keys()
        assert arm["CES_SEQ_EPOCHS"] == bb["CES_SEQ_EPOCHS"] == "80"
        assert {k for k in arm if arm[k] != bb[k]} == {"CES_SEQ_MODEL"}
        assert (arm["CES_SEQ_MODEL"], bb["CES_SEQ_MODEL"]) == ("v2h", "v2")
        assert bb["CES_PER_SHOT_NORM"] == "1" and bb["CES_SEQ_NORM_EXPANDING"] == "0"
        assert bb["CES_TI_SPIKE_CUT_EV"] == "3000" and bb["CES_SEED"] == bb["CES_INIT_SEED"] == str(s)
        assert bb["CES_SPLIT_DIR"] == base["CES_SPLIT_DIR"] and bb["CES_CONTROL_METRICS"] == base["CES_CONTROL_METRICS"]


def test_addendum1_identity_check_flags_any_field_but_the_cap():
    import analyze_b29_addendum1 as a1
    frozen = {k: 1 for k in a1.IDENTITY_FIELDS}
    frozen.update({"epochs_run": 30, "best_epoch": 29, "best_val_masked_mse": 0.3})
    retrain = dict(frozen, epochs_run=47, best_epoch=41, best_val_masked_mse=0.29, select_metric="mse")
    assert a1.identity_mismatches(frozen, retrain) == []              # training length may differ
    for field in a1.IDENTITY_FIELDS:
        assert a1.identity_mismatches(frozen, dict(retrain, **{field: 2})) == [field]
    assert "seq_model" in a1.identity_mismatches(frozen, {k: v for k, v in retrain.items() if k != "seq_model"})
