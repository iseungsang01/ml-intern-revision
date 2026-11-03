"""B.27 -- shot-level repeated 5-fold replicate of the seq_v2 headline
(PREREGISTRATION_B27.md; THESIS_RESULTS.md sec. 8bv named it).

The headline (sec. 8ab) rests on four random 15 % TEST splits whose TRAIN sets overlap,
read by counting splits whose shot-cluster CI excludes 0. B.27 replaces the count with
one interval: every shot is tested exactly once per repeat (5 folds x 2 repeats = 10
replicates per population), and the mean skill gets the Nadeau-Bengio corrected
repeated-CV t-interval, which inflates the variance for the TRAIN overlap.

One variable against the headline: the split design. Model (seq_v2), env (GATE_ENV:
W = 2, held-free, per-file cap 500), per-shot input norm ON, both populations
(cut 3000 eV / inclusive 0) are unchanged. The eval harness needs train-file-only
window normalization stats (CES_CONTROL_METRICS); B.27 has no window control, so the
`stats` stage writes them directly with dataset.fit_normalization_stats -- verified
bit-identical to the frozen controls' stats before any fold is built (stage `verify`).

Stages (sequential, resumable):
  verify   stats builder == frozen .b1_w2cut_s42 / .wsweep_hf_w2_s42 stats (CPU)
  folds    write data/.b27_split_r{r}_f{k}/split_manifest.json (CPU)
  stats    write data/.b27_ctrl_{pop}_r{r}_f{k}/metrics.json (CPU)
  runs     train_seq -> eval_seq(test) -> bootstrap, per pop x repeat x fold (GPU)
  analyze  corrected interval per pop x target x reference -> data/.b27_summary.json

Usage (repo root):
  py ces_prediction/experiments/b27_kfold/run_b27.py --smoke
  py ces_prediction/experiments/b27_kfold/run_b27.py --resume
  py ces_prediction/experiments/b27_kfold/run_b27.py --stages analyze
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR / "experiments" / "b1_gate"))
sys.path.insert(0, str(CES_DIR))
from run_b1_gate import GATE_ENV, AMBIENT_VARS, SMOKE_OVERRIDES, run_step  # noqa: E402

K = 5
REPEAT_SEEDS = (2701, 2702)        # fold-partition seeds, one per repeat
POPS = {"cut": 3000.0, "incl": 0.0}
VAL_FRACTION = 0.2                 # of the shot universe, as in the headline manifests
TARGETS = ("CES_TI", "CES_VT")
REFS = ("pchip", "persistence", "gp_causal", "gp")
UNIVERSE_FROM = DATA / ".b1_manifest_s42" / "split_manifest.json"   # inclusive, 639 shots
VERIFY = {"cut": (".b1_w2cut_s42", ".b1_w2cut_split_s42"),
          "incl": (".wsweep_hf_w2_s42", ".b1_manifest_s42")}


def tag(smoke):
    return "_smoke" if smoke else ""


def split_dir(r, k, smoke=False):
    return DATA / f".b27_split_r{r}_f{k}{tag(smoke)}"


def ctrl_dir(pop, r, k, smoke=False):
    return DATA / f".b27_ctrl_{pop}_r{r}_f{k}{tag(smoke)}"


def run_dir(pop, r, k, smoke=False):
    return DATA / f".b27_seqv2_{pop}_r{r}_f{k}{tag(smoke)}"


def init_seed(r, k):
    return 27000 + 10 * r + k


def cells(smoke):
    if smoke:
        return [("cut", 0, 0)]
    return [(pop, r, k) for pop in POPS for r in range(len(REPEAT_SEEDS)) for k in range(K)]


# ------------------------------------------------------------------ CPU stages
def _dataset(cut):
    from dataset import KSTAR_CES_Dataset  # noqa: PLC0415
    return KSTAR_CES_Dataset(data_dir=DATA, window_size=int(GATE_ENV["CES_WINDOW_SIZE"]),
                             temporal_subset_augmentation=False, drop_stuck_targets=True,
                             ti_spike_cut_ev=cut, per_shot_input_norm=False)


def _fit(ds, names):
    st = ds.fit_normalization_stats({str(DATA / n) for n in names})
    return {g: {k: np.asarray(v, dtype=np.float32).tolist() for k, v in gs.items()} for g, gs in st.items()}


def stage_verify():
    for pop, (ctrl, man) in VERIFY.items():
        ds = _dataset(POPS[pop])
        m = json.loads((DATA / man / "split_manifest.json").read_text(encoding="utf-8"))
        got = _fit(ds, m["train_files"])
        ref = json.loads((DATA / ctrl / "metrics.json").read_text(encoding="utf-8"))["normalization"]["stats"]
        for g, gs in got.items():
            for key, v in gs.items():
                if not np.array_equal(np.asarray(ref[g][key], dtype=np.float32), np.asarray(v, dtype=np.float32)):
                    raise SystemExit(f"FATAL: stats builder differs from {ctrl} at {g}.{key}")
        print(f"[b27] verify {pop}: stats bit-identical to {ctrl}", flush=True)


def stage_folds(smoke):
    m = json.loads(UNIVERSE_FROM.read_text(encoding="utf-8"))
    universe = sorted(set(m["train_files"]) | set(m["val_files"]) | set(m["test_files"]))
    n_val = int(round(VAL_FRACTION * len(universe)))
    for r, seed in enumerate(REPEAT_SEEDS):
        perm = [universe[i] for i in np.random.default_rng(seed).permutation(len(universe))]
        folds = np.array_split(np.arange(len(perm)), K)
        tested = []
        for k, idx in enumerate(folds):
            test = sorted(perm[i] for i in idx)
            rest = [f for i, f in enumerate(perm) if i not in set(idx.tolist())]
            val, train = sorted(rest[:n_val]), sorted(rest[n_val:])
            assert not (set(test) & set(val)) and not (set(test) & set(train)) and not (set(val) & set(train))
            tested += test
            if smoke and (r, k) != (0, 0):
                continue
            d = split_dir(r, k, smoke)
            d.mkdir(parents=True, exist_ok=True)
            (d / "split_manifest.json").write_text(json.dumps({
                "design": "B.27 shot-level repeated K-fold", "repeat": r, "fold": k, "K": K,
                "partition_seed": seed, "val_fraction_of_universe": VAL_FRACTION,
                "split_unit": "csv_shot_file", "universe": len(universe),
                "train_file_count": len(train), "val_file_count": len(val), "test_file_count": len(test),
                "train_files": train, "val_files": val, "test_files": test,
            }, indent=2), encoding="utf-8")
        assert sorted(tested) == universe, "each shot must be tested exactly once per repeat"
    print(f"[b27] folds: universe {len(universe)} shots, K={K}, repeats {len(REPEAT_SEEDS)}, "
          f"val {n_val} per fold", flush=True)


def stage_stats(smoke):
    for pop, cut in POPS.items():
        todo = [(r, k) for (p, r, k) in cells(smoke) if p == pop]
        if not todo:
            continue
        ds = _dataset(cut)
        for r, k in todo:
            man = json.loads((split_dir(r, k, smoke) / "split_manifest.json").read_text(encoding="utf-8"))
            d = ctrl_dir(pop, r, k, smoke)
            d.mkdir(parents=True, exist_ok=True)
            (d / "metrics.json").write_text(json.dumps({
                "source": "B.27 stats-only control (dataset.fit_normalization_stats on TRAIN files)",
                "population": pop, "CES_TI_SPIKE_CUT_EV": cut,
                "normalization": {"stats": _fit(ds, man["train_files"])},
            }), encoding="utf-8")
        print(f"[b27] stats {pop}: {len(todo)} folds", flush=True)


# ------------------------------------------------------------------ GPU stage
def _env(pop, r, k, smoke):
    import os  # noqa: PLC0415
    env = os.environ.copy()
    for var in AMBIENT_VARS + ("CES_MODEL_FILE",):
        env.pop(var, None)
    env.update(GATE_ENV)
    if smoke:
        env.update(SMOKE_OVERRIDES)
        env.update({"CES_SEQ_EPOCHS": "2", "CES_SEQ_MAX_FILES": "40"})
    env.update({
        "CES_TI_SPIKE_CUT_EV": str(int(POPS[pop])),
        "CES_SEED": str(REPEAT_SEEDS[r]),
        "CES_INIT_SEED": str(init_seed(r, k)),
        "CES_SPLIT_DIR": str(split_dir(r, k, smoke)),
        "CES_OUTPUT_DIR": str(run_dir(pop, r, k, smoke)),
        "CES_CONTROL_METRICS": str(ctrl_dir(pop, r, k, smoke) / "metrics.json"),
        "CES_SEQ_MODEL": "v2",
        "CES_PER_SHOT_NORM": "1",
        "CES_SPLIT_TAG": "test",
    })
    return env


def stage_runs(smoke, resume):
    bad = []
    for pop, r, k in cells(smoke):
        out = run_dir(pop, r, k, smoke)
        done = out / ("comparison_errors_test.npz" if smoke else "bootstrap_summary.json")
        if resume and done.exists():
            print(f"[b27] === {out.name}: complete, skipping", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = out / "run.log"
        env = _env(pop, r, k, smoke)
        print(f"[b27] === {out.name} (init {init_seed(r, k)})", flush=True)
        t0 = time.time()
        trained = resume and (out / "weights" / "seq_lstm.pth").exists() and (out / "metrics.json").exists()
        ok = trained or run_step([SEQ_DIR / "train_seq.py"], env, log, "train",
                                 artifacts=(out / "weights" / "seq_lstm.pth", out / "metrics.json"))
        ok = ok and run_step([SEQ_DIR / "eval_seq.py"], env, log, "eval(test)",
                             artifacts=(out / "comparison_errors_test.npz", out / "comparison_metrics.json"))
        if ok and not smoke:
            ok = run_step([CES_DIR / "bootstrap_compare.py"], env, log, "bootstrap",
                          artifacts=(out / "bootstrap_summary.json",))
        print(f"[b27]   {out.name}: {'ok' if ok else 'FAILED'} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if not ok:
            bad.append(out.name)
    return bad


# ------------------------------------------------------------------ analysis
def corrected_interval(values, test_over_train):
    """Nadeau & Bengio (2003) corrected repeated-CV t-interval on the mean."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    mean, var = float(v.mean()), float(v.var(ddof=1))
    q = float(sps.t.ppf(0.975, n - 1))
    half_c = q * np.sqrt((1.0 / n + test_over_train) * var)
    half_u = q * np.sqrt(var / n)
    return {"n": n, "mean": mean, "sd": float(np.sqrt(var)),
            "ci95_corrected": [mean - half_c, mean + half_c],
            "ci95_uncorrected": [mean - half_u, mean + half_u]}


def verdict(ci):
    lo, hi = ci
    return "beats" if lo > 0 else ("loses" if hi < 0 else "undecided")


def stage_analyze():
    summary = {"design": {"K": K, "repeat_seeds": REPEAT_SEEDS, "env": GATE_ENV,
                          "model": "seq_v2", "per_shot_norm": 1}, "cells": {}}
    ratios = []
    for r in range(len(REPEAT_SEEDS)):
        for k in range(K):
            m = json.loads((split_dir(r, k) / "split_manifest.json").read_text(encoding="utf-8"))
            ratios.append(m["test_file_count"] / m["train_file_count"])
    ratio = float(np.mean(ratios))
    summary["design"]["test_over_train_files"] = ratio
    for pop in POPS:
        for t in TARGETS:
            for ref in REFS:
                vals, passes = [], 0
                for r in range(len(REPEAT_SEEDS)):
                    for k in range(K):
                        out = run_dir(pop, r, k)
                        z = np.load(out / "comparison_errors_test.npz")
                        vals.append(float(1.0 - z[f"{t}_se_model"].mean() / z[f"{t}_se_{ref}"].mean()))
                        g = json.loads((out / "bootstrap_summary.json").read_text(encoding="utf-8"))["splits"]["test"][t]
                        passes += int(bool(g.get(ref, {}).get("pass", False)))
                iv = corrected_interval(vals, ratio)
                iv.update({"per_fold": vals, "per_fold_pass": passes,
                           "verdict": verdict(iv["ci95_corrected"])})
                key = f"{pop}/{t}/{ref}"
                summary["cells"][key] = iv
                lo, hi = iv["ci95_corrected"]
                print(f"[b27] {key:28s} mean {iv['mean']:+.3f} sd {iv['sd']:.3f} "
                      f"CI_NB [{lo:+.3f}, {hi:+.3f}] -> {iv['verdict']:9s} fold-pass {passes}/10", flush=True)
    out = DATA / ".b27_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[b27] summary -> {out}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="plumbing: cut, repeat 0, fold 0, tiny caps")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stages", default="verify,folds,stats,runs,analyze")
    args = ap.parse_args()
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    stages = args.stages.split(",")
    t0 = time.time()
    if "verify" in stages:
        stage_verify()
    if "folds" in stages:
        stage_folds(args.smoke)
    if "stats" in stages:
        stage_stats(args.smoke)
    if "runs" in stages:
        bad = stage_runs(args.smoke, args.resume)
        print(f"[b27] runs wall {(time.time() - t0) / 3600:.2f} h; failed: {bad or 'none'}", flush=True)
        if bad:
            sys.exit(1)
    if "analyze" in stages and not args.smoke:
        stage_analyze()


if __name__ == "__main__":
    main()
