"""Is this actually real-time? Measure it instead of asserting it.

The paper motivates the nowcaster by "ultimately, real-time use" but never reports a
latency. That is a claim a control-systems reader will check first and we should not
make them guess. The relevant budget is hard: CES sits on a 10 ms grid, so an online
gap-filler has 10 ms per step minus whatever the acquisition and control loop already
spend.

What is measured here is the *inference* cost of the pinned thesis architecture, at
batch 1 (the online case) and at batch 512 (the offline reprocessing case), on CPU and
GPU. Tail latency, not the mean, is what decides whether a real-time loop holds its
deadline, so p95/p99 are reported and the mean is not reported alone. Normalization
statistics are applied outside the model, so the number below is the network only;
feature assembly from the acquisition system is not included and is stated as such.

Both selected windows are measured: W=2 (the window the selection rule of the paper
returns) and W=4 (the incumbent), since the history length changes the sequence length
the GRU walks.

Run this with nothing else on the GPU -- a concurrent training job invalidates the
tail numbers, which are the point.

Usage (repo root):  py ces_prediction/experiments/latency/bench_latency.py
Writes: data/.latency_benchmark.json
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ces_prediction"))
from model import MultimodalCESPredictor  # noqa: E402  (the published architecture)

WINDOWS = (2, 4)
BATCHES = (1, 512)
WARMUP = 50
ITERS = 1000
GRID_MS = 10.0          # the CES cadence the model must keep up with
CHANNELS = {"bes": 9, "ecei": 4, "mc": 2, "time": 4, "ces_history": 4}


def make_inputs(batch, window, device):
    def t(c):
        return torch.randn(batch, window, c, device=device)
    return (t(CHANNELS["bes"]), t(CHANNELS["ecei"]), t(CHANNELS["mc"]),
            t(CHANNELS["time"]), t(CHANNELS["ces_history"]))


def bench_steadystate(model, inputs, device, iters=ITERS, warmup=WARMUP):
    """Amortized cost with ONE synchronize around the whole loop.

    Reported alongside the per-call numbers because they answer different questions
    and, on a laptop GPU, disagree sharply. Per-call timing forces the device idle
    between iterations, so clocks ramp down and launch overhead is paid every step --
    which is exactly what a 10 ms control loop does, so that number is the honest one
    for the online case. This one is the fair number for offline reprocessing, where
    calls are back to back.
    """
    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
        total = (time.perf_counter() - t0) * 1000.0
    return {"amortized_ms": total / iters, "iters": iters}


def bench(model, inputs, device, iters=ITERS, warmup=WARMUP):
    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(*inputs)
            if cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": samples[int(0.95 * len(samples)) - 1],
        "p99_ms": samples[int(0.99 * len(samples)) - 1],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "iters": len(samples),
    }


def main():

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results = {
        "_env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_threads": torch.get_num_threads(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "grid_ms": GRID_MS,
        },
        "_scope": ("network forward pass only; feature assembly and normalization "
                   "happen outside the model and are not included"),
        "runs": [],
    }

    for window in WINDOWS:
        model = MultimodalCESPredictor(
            window_size=window,
            bes_channels=CHANNELS["bes"], ecei_channels=CHANNELS["ecei"],
            mc_channels=CHANNELS["mc"], time_channels=CHANNELS["time"],
            ces_history_channels=CHANNELS["ces_history"],
        ).eval()
        n_params = sum(p.numel() for p in model.parameters())
        for device in devices:
            model.to(device)
            for batch in BATCHES:
                inputs = make_inputs(batch, window, device)
                stats = bench(model, inputs, device)
                steady = bench_steadystate(model, inputs, device)
                rec = {
                    "window": window, "device": device.type, "batch": batch,
                    "params": n_params, **stats,
                    "amortized_ms": steady["amortized_ms"],
                    "per_sample_median_ms": stats["median_ms"] / batch,
                    "throughput_samples_per_s": batch / (steady["amortized_ms"] / 1000.0),
                    "grid_budget_used_pct_p99": 100.0 * stats["p99_ms"] / GRID_MS,
                }
                results["runs"].append(rec)
                print(f"[latency] W={window} {device.type:>4} batch={batch:>4} "
                      f"per-call median={stats['median_ms']:8.3f} p99={stats['p99_ms']:8.3f} | "
                      f"amortized={steady['amortized_ms']:8.3f} ms "
                      f"({rec['throughput_samples_per_s']:,.0f} samples/s)")

    out = REPO_ROOT / "data" / ".latency_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print(f"\n=== online case (batch 1) against the {GRID_MS:.0f} ms CES budget")
    print(f"{'W':>3} {'device':>7} {'params':>9} {'median':>10} {'p99':>10} {'budget p99':>12}")
    for r in results["runs"]:
        if r["batch"] != 1:
            continue
        print(f"{r['window']:>3} {r['device']:>7} {r['params']:>9,} "
              f"{r['median_ms']:>9.3f}m {r['p99_ms']:>9.3f}m "
              f"{r['grid_budget_used_pct_p99']:>11.2f}%")


if __name__ == "__main__":
    main()
