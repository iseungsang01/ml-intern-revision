"""B.47 checks that need no training and no GPU (CPU only; nothing is written outside the console).

  1. the patch: every reference to build_blocks inside train_seq goes through the module global that
     train_delayed.install rebinds;
  2. the wrapper on the train, validation and test files of one split (both standardizations):
     L = 0 returns blocks equal to seq_data.build_blocks; L = 2 changes only the six CES channels, which
     equal run_b42.ces_channels(..., 2, "true") and an independent brute-force implementation of the rule;
  3. one CPU forward pass of frozen networks on blocks built by the wrapper with L = 2;
  4. the run_b39.py process guard of run_b47.py.

    py ces_prediction/experiments/b47_after_b39/check_b47.py [--split 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
for k in [k for k in os.environ if k.startswith("CES_") or k.startswith("B47_")]:
    del os.environ[k]
# the data treatment of every B.47 cut run (run_b29.run_env / GATE_ENV)
os.environ.update({"CES_DATA_DIR": str(DATA), "CES_TI_SPIKE_CUT_EV": "3000", "CES_DROP_STUCK_TARGETS": "1",
                   "CES_SEQ_ZERO_MC": "0", "CES_SEQ_NORM_EXPANDING": "0", "CES_PER_SHOT_NORM": "0"})
sys.path.insert(0, str(HERE))
import train_delayed as TD  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def brute(time_, tz, obs, L):
    """The delayed rule written out directly: at row i the last u with obs[u] and u <= i - 1 - L."""
    n = len(time_)
    out = np.zeros((n, 6))
    for t in range(2):
        for i in range(n):
            us = [u for u in range(0, i - L) if obs[u, t]]
            if us:
                u = us[-1]
                out[i, 3 * t] = tz[u, t]
                out[i, 3 * t + 1] = np.log1p(max(time_[i] - time_[u], 0.0))
                out[i, 3 * t + 2] = 1.0
    return out


def code_refs(mod, name):
    """Every code object of a module (nested ones included) that names `name`, with how it names it."""
    hits = []

    def walk(co, where):
        if name in co.co_names:
            hits.append((where + co.co_name, "global/attribute lookup"))
        if name in co.co_varnames or name in co.co_cellvars or name in co.co_freevars:
            hits.append((where + co.co_name, "LOCAL binding"))
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk(c, where + co.co_name + ".")

    for attr, obj in vars(mod).items():
        if isinstance(obj, types.FunctionType) and obj.__module__ == mod.__name__:
            walk(obj.__code__, "")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=42)
    ap.add_argument("--brute-files", type=int, default=12, help="files per role checked against the brute-force rule")
    args = ap.parse_args()
    s = args.split

    # ---------------------------------------------------------------- 1. the patch
    import seq_data
    original = seq_data.build_blocks
    train_seq, state2 = TD.install(2)
    wrap2 = train_seq.build_blocks
    from dataset import STUCK_GAP_SECONDS
    from run_b42 import ces_channels, R as R30
    wrap0, state0 = TD.make_wrapper(original, 0, ces_channels, STUCK_GAP_SECONDS)
    refs = code_refs(train_seq, "build_blocks")
    log(f"1. train_seq.build_blocks is the wrapper: {train_seq.build_blocks is wrap2}; "
        f"seq_data.build_blocks is the wrapper: {seq_data.build_blocks is wrap2}; references in train_seq: {refs}")
    if train_seq.build_blocks is not wrap2 or any(k != "global/attribute lookup" for _, k in refs) or not refs:
        raise SystemExit("FATAL: check 1 failed")

    # ---------------------------------------------------------------- 2. the wrapper on one split
    man = json.loads((DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json").read_text(encoding="utf-8"))
    grid, dims = seq_data.load_grid_files(DATA, True)
    stats = seq_data.fit_stats(grid, dims, [n for n in man["train_files"] if n in grid])
    base = dims["n_fast"] + 1
    log(f"grid {len(grid)} files, n_fast {dims['n_fast']}, CES channels {base}..{base + 5}")
    summary = {}
    for per_shot in (False, True):
        for role in ("train_files", "val_files", "test_files"):
            names = [n for n in man[role] if n in grid]
            c = {"files": 0, "blocks": 0, "L0_allclose": 0, "L0_array_equal": 0, "L2_ces_equal_ces_channels": 0,
                 "L2_other_channels_unchanged": 0, "L2_y_mask_time_unchanged": 0, "L2_ces_changed_somewhere": 0,
                 "brute_files": 0, "brute_blocks_equal": 0, "brute_blocks": 0, "max_abs_dev_L0": 0.0}
            for j, nm in enumerate(names):
                arr = grid[nm]
                ref = original(arr, dims, stats, per_shot_norm=per_shot)
                b0 = wrap0(arr, dims, stats, per_shot_norm=per_shot)
                b2 = wrap2(arr, dims, stats, per_shot_norm=per_shot)
                if not (len(ref) == len(b0) == len(b2)):
                    raise SystemExit(f"FATAL: block counts differ on {nm}")
                c["files"] += 1
                tm = arr[:, 0].astype(np.float64)
                tgt = arr[:, 1:3].astype(np.float64)
                tz = (tgt - stats["target"]["mean"].astype(np.float64)) / stats["target"]["std"].astype(np.float64)
                spans = R30.block_spans(tm, STUCK_GAP_SECONDS)   # run_b30's span rule, independent of the wrapper
                for k, (r, z0, z2, (a, b)) in enumerate(zip(ref, b0, b2, spans)):
                    c["blocks"] += 1
                    c["L0_allclose"] += int(all(np.allclose(r[q], z0[q], rtol=1e-6, atol=1e-6) for q in ("x", "y", "mask", "time")))
                    c["L0_array_equal"] += int(all(np.array_equal(r[q], z0[q]) for q in ("x", "y", "mask", "time")))
                    c["max_abs_dev_L0"] = max(c["max_abs_dev_L0"], float(np.max(np.abs(r["x"] - z0["x"]))))
                    exp = ces_channels(tm[a:b], tz[a:b], ~np.isnan(tgt[a:b]), 2, "true").astype(np.float32)
                    c["L2_ces_equal_ces_channels"] += int(np.array_equal(z2["x"][:, base:base + 6], exp))
                    other = np.ones(r["x"].shape[1], bool)
                    other[base:base + 6] = False
                    c["L2_other_channels_unchanged"] += int(np.array_equal(z2["x"][:, other], r["x"][:, other]))
                    c["L2_y_mask_time_unchanged"] += int(all(np.array_equal(r[q], z2[q]) for q in ("y", "mask", "time")))
                    c["L2_ces_changed_somewhere"] += int(not np.array_equal(z2["x"], r["x"]))
                    if j < args.brute_files:
                        bf = brute(tm[a:b], tz[a:b], ~np.isnan(tgt[a:b]), 2).astype(np.float32)
                        c["brute_blocks"] += 1
                        c["brute_blocks_equal"] += int(np.allclose(bf, z2["x"][:, base:base + 6], rtol=1e-6, atol=1e-6))
                if j < args.brute_files:
                    c["brute_files"] += 1
            key = f"{'per-discharge' if per_shot else 'train-file'} standardization, {role}"
            summary[key] = c
            log(f"2. {key}: {c}")
            n = c["blocks"]
            ok = (c["L0_allclose"] == n and c["L2_ces_equal_ces_channels"] == n and c["L2_other_channels_unchanged"] == n
                  and c["L2_y_mask_time_unchanged"] == n and c["brute_blocks_equal"] == c["brute_blocks"])
            if not ok:
                raise SystemExit(f"FATAL: check 2 failed ({key})")
    log(f"2. wrapper calls during the check: L = 2 {state2['calls']}, L = 0 {state0['calls']}")

    # ---------------------------------------------------------------- 3. forward pass of frozen networks
    import torch
    from evaluate import _load_stats
    from seq_models import SEQ_MODELS
    torch.set_num_threads(1)
    test = [n for n in man["test_files"] if n in grid]
    for run, model_name in ((f".b39_v2_cut_s{s}_i{s}", "v2"), (f".b29a1_v2h_cut_s{s}", "v2h"), (f".b39_v2b_cut_s{s}_i{s}", "v2b")):
        m = json.loads((DATA / run / "metrics.json").read_text(encoding="utf-8"))
        st = _load_stats(m)
        same = all(np.allclose(st[g][q], stats[g][q]) for g in stats for q in stats[g])
        net = SEQ_MODELS[model_name]()
        net.load_state_dict(torch.load(DATA / run / "weights" / "seq_lstm.pth", map_location="cpu"))
        net.eval()
        blk = max((b for nm in test[:10] for b in wrap2(grid[nm], dims, st, per_shot_norm=bool(m["per_shot_input_norm"]))),
                  key=lambda b: len(b["x"]))
        xin = torch.from_numpy(blk["x"]).unsqueeze(0)
        with torch.no_grad():
            out = net(xin)
        log(f"3. {run} ({model_name}, per-discharge {bool(m['per_shot_input_norm'])}): input {tuple(xin.shape)} {xin.dtype} -> "
            f"output {tuple(out.shape)} {out.dtype}, finite {bool(torch.isfinite(out).all())}; "
            f"its stored statistics equal fit_stats on the manifest train files: {same}")

    # ---------------------------------------------------------------- 4. the process guard
    sys.path.insert(0, str(HERE))
    import run_b47
    procs = run_b47.b39_processes()
    log(f"4. run_b39.py processes seen by the guard: {[p for p, _ in procs]} "
        f"(refuse_if_b39_alive would {'REFUSE' if procs else 'allow'} a start); GPU memory used {run_b47.gpu_used_mib()} MiB")
    log("all checks passed")


if __name__ == "__main__":
    main()
