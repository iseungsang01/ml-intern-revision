"""Skill vs. latency for every arm that could run ONLINE (THESIS_RESULTS.md sec. 8ac).

The claim this prices (승상님, 2026-08-17): on the 10 ms CES grid, when the target is
missing, current practice is to HOLD the last value (persistence). Anything proposed as
its replacement must (a) beat it, and (b) fit in the 10 ms budget using past data only.

sec. 8x named `gp_causal` the strongest deployable causal baseline and sec. 8ac adopts it
as the reach headline -- but "deployable" was a statement about its INFORMATION (past
only), never about its cost. It refits an exact Matern-3/2 GP per row, selecting
hyperparameters by log marginal likelihood over a 5 x 4 grid (`baselines_interpolation`
`_GP_LS_GRID` x `_GP_NOISE_GRID`) on up to `_GP_MAX_SIDE` = 16 past neighbours. That is a
per-sample cost, so it belongs in the same table as the network's.

Latency is measured on the REAL neighbour sets of the frozen TEST files under the
confirmed protocol (held-free, 3 keV cut), because GP cost scales with how many past
observations exist -- synthetic neighbourhoods would not reproduce it. Tail latency, not
the mean, decides whether a real-time loop holds its deadline (sec. 8l), so p95/p99 are
reported. Skill comes from the sec. 8ac reach artifacts, so both columns describe the
same rows.

The network's own numbers are read from the frozen sec. 8l benchmark
(`data/.latency_benchmark.json`, same machine) rather than re-measured here.

Usage (repo root):  py ces_prediction/experiments/reach/bench_causal_arms.py
Writes: data/.reach_pareto.json
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR))
sys.path.insert(1, str(CES_DIR / "experiments" / "seq"))

import baselines_interpolation as B  # noqa: E402
from seq_data import load_grid_files  # noqa: E402

SEEDS = (42, 1, 7, 123)
GRID_MS = 10.0            # the CES cadence an online gap-filler must keep up with
N_SAMPLES = 1500          # neighbour sets drawn per target
WARMUP = 100
SAMPLE_SEED = 20260817

# Causal arms only for the deadline verdict; the acausal ones are shown for reference
# because they are what sec. 8's headline used -- they cannot run online at all.
CAUSAL_ARMS = ("persistence", "ar_local", "gp_causal")
ACAUSAL_ARMS = ("linear", "pchip", "gp")


def collect_neighbor_sets(rng):
    """Real (times, values, target_time) triples from the frozen TEST files.

    Protocol-pinned exactly as B.1/sec. 8ac: held-free plus the 3 keV fit-failure cut,
    so the neighbour counts -- and therefore the GP cost -- match the scored population.
    """
    manifests = [json.loads((DATA / f".b1_w2cut_split_s{s}" / "split_manifest.json")
                            .read_text(encoding="utf-8")) for s in SEEDS]
    test_names = sorted({n for m in manifests for n in m["test_files"]})
    grid, _ = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=3000.0)

    pool = {t: [] for t in (1, 2)}     # column 1 = CES_TI, 2 = CES_VT
    for name in test_names:
        arr = grid.get(Path(name).name)
        if arr is None:
            continue
        for tcol in (1, 2):
            obs = np.flatnonzero(~np.isnan(arr[:, tcol]))
            for ri in obs:
                pool[tcol].append((name, int(ri)))

    out = {}
    for tcol, items in pool.items():
        pick = rng.choice(len(items), size=min(N_SAMPLES, len(items)), replace=False)
        sets = []
        for i in pick:
            name, ri = items[i]
            arr = grid[Path(name).name]
            times, values, target_time = B.build_neighbor_set(arr, 0, tcol, ri)
            sets.append((times, values, target_time))
        out[tcol] = sets
    return out


def bench_arm(fn, sets):
    for times, values, tt in sets[:WARMUP]:
        fn(times, values, tt)
    samples = []
    for times, values, tt in sets:
        t0 = time.perf_counter()
        fn(times, values, tt)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": samples[int(0.95 * n) - 1],
        "p99_ms": samples[int(0.99 * n) - 1],
        "max_ms": samples[-1],
        "n_calls": n,
        "grid_budget_used_pct_p99": 100.0 * samples[int(0.99 * n) - 1] / GRID_MS,
        "fits_10ms_budget_p99": samples[int(0.99 * n) - 1] < GRID_MS,
    }


def skill_table():
    """Mean skill vs persistence over the four sec. 8ac reach runs (same rows).

    Read from the metrics report rather than the npz: `ar_local` is scored by the
    harness but is not among the keys `eval_seq` writes to `comparison_errors`, and
    leaving the strongest cheap causal arm out of the table would beg the question.
    skill = 1 - (rmse_arm / rmse_persistence)^2 is identical to the MSE-ratio form.
    """
    arms = (*CAUSAL_ARMS, *ACAUSAL_ARMS)
    acc = {t: {a: [] for a in (*arms, "seq_v2")} for t in ("CES_TI", "CES_VT")}
    for s in SEEDS:
        path = DATA / f".reach_s{s}" / "comparison_metrics_test.json"
        if not path.exists():
            raise SystemExit(f"FATAL: run run_reach.py first -- missing {path}")
        rep = json.loads(path.read_text(encoding="utf-8"))["per_target"]
        for t in acc:
            base = rep[t]["baselines"]
            ref = base["persistence"]["rmse"]
            for a in arms:
                if a in base:
                    acc[t][a].append(1.0 - (base[a]["rmse"] / ref) ** 2)
            acc[t]["seq_v2"].append(1.0 - (rep[t]["rmse_model"] / ref) ** 2)
    return {t: {a: (float(np.mean(v)) if v else None) for a, v in d.items()}
            for t, d in acc.items()}


def _fmt(x):
    return f"{x:>+9.4f}" if x is not None else f"{'n/a':>9}"


def main():
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required, aborting.")
    if not B._HAVE_SKLEARN:
        raise SystemExit("FATAL: sklearn unavailable -- the GP arms cannot be benchmarked.")

    rng = np.random.default_rng(SAMPLE_SEED)
    print(f"[pareto] collecting real neighbour sets from the frozen TEST files ...", flush=True)
    sets = collect_neighbor_sets(rng)
    sizes = {t: [len(s[0]) for s in v] for t, v in sets.items()}
    for tcol, name in ((1, "CES_TI"), (2, "CES_VT")):
        q = np.percentile(sizes[tcol], [50, 95])
        print(f"[pareto] {name}: {len(sets[tcol])} neighbour sets, "
              f"|neighbours| median {q[0]:.0f} p95 {q[1]:.0f}", flush=True)

    latency = {}
    for tcol, name in ((1, "CES_TI"), (2, "CES_VT")):
        latency[name] = {}
        for arm in (*CAUSAL_ARMS, *ACAUSAL_ARMS):
            latency[name][arm] = bench_arm(B.PREDICTORS[arm], sets[tcol])
            r = latency[name][arm]
            print(f"[pareto] {name:>7} {arm:>12}: median {r['median_ms']:8.4f} ms  "
                  f"p99 {r['p99_ms']:8.4f} ms  ({r['grid_budget_used_pct_p99']:6.2f}% of budget)",
                  flush=True)

    net = {}
    bench_path = DATA / ".latency_benchmark.json"
    if bench_path.exists():
        for r in json.loads(bench_path.read_text(encoding="utf-8"))["runs"]:
            if r.get("batch") == 1 and r.get("device") == "cpu" and r["model"] in (
                    "seq_v2_step", "window_iter009"):
                key = r["model"] + (f"_w{r['window']}" if r.get("window") else "")
                net[key] = {"median_ms": r["median_ms"], "p99_ms": r["p99_ms"],
                            "grid_budget_used_pct_p99": 100.0 * r["p99_ms"] / GRID_MS,
                            "fits_10ms_budget_p99": r["p99_ms"] < GRID_MS}

    skills = skill_table()
    out = {
        "question": ("under a 10 ms deadline, using past data only, what is the best "
                     "available replacement for holding the last CES value?"),
        "grid_ms": GRID_MS,
        "protocol": {"held_free": True, "ti_spike_cut_ev": 3000, "seeds": list(SEEDS),
                     "n_neighbour_sets": N_SAMPLES, "sample_seed": SAMPLE_SEED,
                     "device": "cpu (single call, batch 1)",
                     "gp_config": {"max_past_neighbours": B._GP_MAX_SIDE,
                                   "length_scales": list(B._GP_LS_GRID),
                                   "noise_grid": list(B._GP_NOISE_GRID),
                                   "refit_per_row": True}},
        "neighbour_set_sizes": {n: {"median": float(np.median(sizes[c])),
                                    "p95": float(np.percentile(sizes[c], 95))}
                                for c, n in ((1, "CES_TI"), (2, "CES_VT"))},
        "latency": latency, "network_latency_sec8l": net,
        "skill_vs_persistence_mean_over_seeds": skills,
        "note": ("baseline latency is the predictor call on an already-built neighbour set; "
                 "network latency is the forward pass only (sec. 8l scope). Feature assembly "
                 "from the acquisition system is excluded for BOTH, so the comparison is fair."),
    }
    (DATA / ".reach_pareto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== online arms against the {GRID_MS:.0f} ms CES budget "
          f"(skill vs persistence = what holding the last value achieves)")
    print(f"{'arm':>14} {'causal':>7} {'TI skill':>9} {'VT skill':>9} "
          f"{'median ms':>10} {'p99 ms':>9} {'budget p99':>11} {'deployable':>11}")
    for arm in (*CAUSAL_ARMS, *ACAUSAL_ARMS):
        r = latency["CES_TI"][arm]
        causal = "yes" if arm in CAUSAL_ARMS else "NO"
        dep = "yes" if (arm in CAUSAL_ARMS and r["fits_10ms_budget_p99"]) else "no"
        print(f"{arm:>14} {causal:>7} {_fmt(skills['CES_TI'][arm])} "
              f"{_fmt(skills['CES_VT'][arm])} {r['median_ms']:>10.4f} {r['p99_ms']:>9.4f} "
              f"{r['grid_budget_used_pct_p99']:>10.2f}% {dep:>11}")
    for key, r in net.items():
        seq = key == "seq_v2_step"
        dep = "yes" if r["fits_10ms_budget_p99"] else "no"
        print(f"{key:>14} {'yes':>7} {_fmt(skills['CES_TI']['seq_v2'] if seq else None)} "
              f"{_fmt(skills['CES_VT']['seq_v2'] if seq else None)} "
              f"{r['median_ms']:>10.4f} {r['p99_ms']:>9.4f} "
              f"{r['grid_budget_used_pct_p99']:>10.2f}% {dep:>11}")
    print(f"\n[pareto] saved {DATA / '.reach_pareto.json'}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
