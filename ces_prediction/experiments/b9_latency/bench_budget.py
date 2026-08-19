"""B.9 axis C: which arms survive a 1 ms budget, measured to the strengthened protocol.

§8ac priced the causal arms against the 10 ms CES grid and found the budget never binds —
every arm fits, so the argument for the backbone was skill-per-millisecond, not feasibility.
A 1 ms budget (a control-cycle deadline rather than a diagnostic-cadence one) is a different
question, and it is the first place where structure decides admissibility: `seq_v2`'s own
stateful step sat at p99 1.49 ms, so it does not obviously survive its own tightening.

**Why this needs its own protocol.** §8ac concluded "quote the ordering, not the absolutes"
after two sessions disagreed by 4.2× on one arm. A 1 ms verdict is decided BY an absolute,
so that rule is not enough. PREREGISTRATION_B9.md §4 fixes the replacement, implemented
here: 5 independent sessions (separate processes), 200 warm-up + 2,000 timed iterations
each, single-threaded batch-1 CPU, and a pass only when **every** session's p99 clears the
budget. A max-p99 landing in [0.8 B, 1.25 B] is reported as boundary/undecided rather than
resolved either way, and the environment is recorded with every session.

**Cost does not depend on trained weights**, so the arms are instantiated fresh — the same
choice §8l and §8ac made. What it does depend on is nothing else running, which is why the
sessions must not be launched alongside a training batch.

Each network arm is timed in its ONLINE form: `seq_v2` and its width variants as a stateful
LSTM step, the TCN through its ring-buffer cache, the transformer through its KV cache, and
the window family as the full forward it has to redo every step. Where an arm has no
streaming implementation it is timed re-running its whole receptive field and labelled
`window_recompute`, so the table never credits an efficiency that does not exist.

Usage (repo root, with nothing else running):
  py ces_prediction/experiments/b9_latency/bench_budget.py --all
  py ces_prediction/experiments/b9_latency/bench_budget.py --session 3
  py ces_prediction/experiments/b9_latency/bench_budget.py --aggregate
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(CES_DIR))
sys.path.insert(1, str(CES_DIR / "experiments" / "seq"))
sys.path.insert(2, str(CES_DIR / "experiments" / "latency"))
from model import MultimodalCESPredictor  # noqa: E402  (the window family)
from seq_models import SEQ_MODELS  # noqa: E402
from seq_data import N_FEATURES  # noqa: E402
from bench_latency import CHANNELS, SeqV2Step, make_inputs  # noqa: E402

SESSIONS = 5
WARMUP = 200
ITERS = 2000
BUDGETS_MS = (10.0, 1.0)     # CES grid cadence; control-cycle deadline
BOUNDARY = (0.8, 1.25)       # PREREGISTRATION_B9.md §4.5

# arm -> (kind, spec). "seq" = a SEQ_MODELS variant, "window" = the iter009 family,
# "baseline" = a predictor from baselines_interpolation timed on REAL neighbour sets.
#
# The two baselines are re-measured here rather than quoted from §8ac because a 1 ms
# verdict is not robust to that section's own caveat: it warns absolutes can move up to
# 4x between sessions, and `gp_causal` at p99 2.34 ms would flip from fail to pass under
# that much movement. `persistence` is far enough below the band to be safe either way,
# and is carried as the protocol's own sanity floor.
ARMS = (
    ("persistence", "baseline", "predict_persistence"),
    ("gp_causal", "baseline", "predict_gp_causal"),
    ("seq_v2", "seq", "v2"),
    ("v2m7k", "seq", "v2m7k"),
    ("v2m2k", "seq", "v2m2k"),
    ("tcn15", "seq", "tcn15"),
    ("tcn63", "seq", "tcn63"),
    ("xfmr63", "seq", "xfmr63"),
    ("window_w2", "window", 2),
    # Fused deployment forms of the three recurrent arms — the only ones within reach of
    # 1 ms in eager mode, so the only ones where fusion can change a verdict.
    ("seq_v2_jit", "seq_fused", "v2"),
    ("v2m7k_jit", "seq_fused", "v2m7k"),
    ("v2m2k_jit", "seq_fused", "v2m2k"),
)


def _percentile(sorted_samples, q):
    return sorted_samples[min(len(sorted_samples) - 1, int(q * len(sorted_samples)))]


def _time_loop(call, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"median_ms": statistics.median(samples),
            "p95_ms": _percentile(samples, 0.95),
            "p99_ms": _percentile(samples, 0.99),
            "max_ms": samples[-1],
            "iters": len(samples)}


_NEIGHBOR_SETS = None


def _neighbor_sets():
    """Real past-neighbour sets from the frozen TEST files (§8ac); loaded once per process.

    GP cost scales with how many past observations a row actually has, so synthetic
    neighbourhoods would not reproduce it — the same reason `bench_causal_arms` insists on
    the real ones.
    """
    global _NEIGHBOR_SETS
    if _NEIGHBOR_SETS is None:
        import numpy as np
        sys.path.insert(0, str(CES_DIR / "experiments" / "reach"))
        from bench_causal_arms import collect_neighbor_sets, SAMPLE_SEED
        _NEIGHBOR_SETS = collect_neighbor_sets(np.random.default_rng(SAMPLE_SEED))
    return _NEIGHBOR_SETS


class _V2StepPure(torch.nn.Module):
    """seq_v2's online step with the recurrent state passed in and out as plain tensors.

    `SeqV2Step` keeps its state as Python attributes, which `torch.jit.trace` cannot see —
    tracing it would silently freeze the first step's state into the graph. Making the state
    explicit is what makes the fused arm both traceable and *correct*, and the bench asserts
    it reproduces the eager step before timing it.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.n_fast = m.n_fast

    def forward(self, x_t, h_ti, c_ti, h_vt, c_vt):
        m = self.m
        out_ti, (h_ti2, c_ti2) = m.lstm_ti(x_t, (h_ti, c_ti))
        out_vt, (h_vt2, c_vt2) = m.lstm_vt(x_t[..., self.n_fast:], (h_vt, c_vt))
        y = torch.cat([m.head_ti(m.norm_ti(out_ti)), m.head_vt(m.norm_vt(out_vt))], dim=-1)
        return y, h_ti2, c_ti2, h_vt2, c_vt2


def _zero_state(lstm):
    return (torch.zeros(lstm.num_layers, 1, lstm.hidden_size),
            torch.zeros(lstm.num_layers, 1, lstm.hidden_size))


def make_arm(kind, spec):
    """-> (callable, mode, n_params). The callable performs ONE online step."""
    torch.set_grad_enabled(False)
    if kind == "seq_fused":
        # PREREGISTRATION_B9.md §2.3: a deployment-optimisation LATENCY column, never a
        # scored artifact. §8ac measured torch.jit fusion at 1.73x on the window model with
        # outputs identical to 1e-5; at a 1 ms budget that factor decides admissibility, so
        # it has to be measured rather than quoted.
        model = SEQ_MODELS[spec]().eval()
        pure = _V2StepPure(model).eval()
        x_t = torch.randn(1, 1, N_FEATURES)
        state = (*_zero_state(model.lstm_ti), *_zero_state(model.lstm_vt))
        eager = pure(x_t, *state)[0]
        traced = torch.jit.freeze(torch.jit.trace(pure, (x_t, *state)))
        for _ in range(3):                       # let the fuser specialise
            traced(x_t, *state)
        fused = traced(x_t, *state)[0]
        if not torch.allclose(eager, fused, atol=1e-5):
            raise SystemExit(f"FATAL: fused {spec} differs from eager by "
                             f"{float((eager - fused).abs().max()):.2e}; refusing to time it")
        return (lambda: traced(x_t, *state)), "jit_fused_lstm", model.n_params

    if kind == "baseline":
        import baselines_interpolation as B
        fn = getattr(B, spec)
        sets = _neighbor_sets()[1]                     # column 1 = CES_TI
        counter = {"i": 0}

        def call():
            times, values, tt = sets[counter["i"] % len(sets)]
            counter["i"] += 1
            fn(times, values, tt)
        return call, "per_row_baseline", 0

    if kind == "window":
        # Channel counts are the dataset's, not the constructor defaults (§8l does the
        # same); the defaults declare 3 history channels and the contract carries 4.
        model = MultimodalCESPredictor(
            window_size=spec,
            bes_channels=CHANNELS["bes"], ecei_channels=CHANNELS["ecei"],
            mc_channels=CHANNELS["mc"], time_channels=CHANNELS["time"],
            ces_history_channels=CHANNELS["ces_history"],
        ).eval()
        inputs = make_inputs(1, spec, torch.device("cpu"))
        n = sum(p.numel() for p in model.parameters())
        # No streaming form exists: the window family re-runs three sensor CNNs over the
        # whole window every step, which is the 3.0x tail §8ac measured.
        return (lambda: model(*inputs)), "window_recompute", n

    model = SEQ_MODELS[spec]().eval()
    x_t = torch.randn(1, 1, N_FEATURES)
    if hasattr(model, "stream_step"):
        state = model.stream_init()
        return (lambda: model.stream_step(state, x_t)), "stream_cache", model.n_params
    if hasattr(model, "lstm_ti"):                      # seq_v2 and its width variants
        step = SeqV2Step(model).eval()
        return (lambda: step(x_t)), "stateful_lstm", model.n_params
    reach = getattr(model, "receptive_field", None) or 63
    window = torch.randn(1, reach, N_FEATURES)
    return (lambda: model(window)), "window_recompute", model.n_params


def run_session(index):
    torch.set_num_threads(1)
    out = {"session": index,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "env": {"platform": platform.platform(),
                   "processor": platform.processor(),
                   "torch": torch.__version__,
                   "threads": torch.get_num_threads(),
                   "cpu_count": os.cpu_count()},
           "warmup": WARMUP, "iters": ITERS, "arms": {}}
    for name, kind, spec in ARMS:
        call, mode, n_params = make_arm(kind, spec)
        stats = _time_loop(call)
        stats.update({"mode": mode, "params": n_params})
        out["arms"][name] = stats
        print(f"[b9c] s{index} {name:>10} {mode:>16} params={n_params:>7,} "
              f"median={stats['median_ms']:7.3f} p99={stats['p99_ms']:7.3f} ms", flush=True)
    path = DATA / f".b9_latency_s{index}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[b9c] wrote {path}", flush=True)
    return out


def verdict(max_p99, budget):
    if max_p99 < BOUNDARY[0] * budget:
        return "pass"
    if max_p99 > BOUNDARY[1] * budget:
        return "fail"
    return "boundary"          # §4.5: no deployment claim either way


def aggregate():
    sessions = []
    for i in range(1, SESSIONS + 1):
        path = DATA / f".b9_latency_s{i}.json"
        if path.exists():
            sessions.append(json.loads(path.read_text()))
    if not sessions:
        raise SystemExit("FATAL: no session files; run --session 1..5 first")
    if len(sessions) < SESSIONS:
        # ASCII only in prints: this console is cp949 (see b9_reach/run_b9_reach.py).
        print(f"[b9c] WARNING: {len(sessions)}/{SESSIONS} sessions present; the sec.4 rule "
              f"requires all {SESSIONS}; verdicts below are provisional", flush=True)

    summary = {"question": "which arms clear a 10 ms and a 1 ms budget?",
               "protocol": {"sessions": len(sessions), "required_sessions": SESSIONS,
                            "warmup": WARMUP, "iters": ITERS, "batch": 1, "device": "cpu",
                            "threads": 1, "budgets_ms": list(BUDGETS_MS),
                            "boundary_band": list(BOUNDARY),
                            "rule": "pass iff EVERY session p99 < budget; "
                                    "max p99 in [0.8B, 1.25B] = boundary/undecided",
                            "prereg": "experiments/PREREGISTRATION_B9.md §4",
                            "span": [sessions[0]["started"], sessions[-1]["started"]],
                            "env": sessions[0]["env"]},
               "arms": {}}

    print("\n" + "=" * 96)
    print(f"{len(sessions)} sessions, batch 1, 1 thread, CPU: max p99 over sessions decides")
    print("arm".rjust(11) + "mode".rjust(17) + "params".rjust(9) + "med".rjust(9)
          + "p99".rjust(9) + "max p99".rjust(10) + "10 ms".rjust(10) + "1 ms".rjust(10))
    for name, _, _ in ARMS:
        per = [s["arms"][name] for s in sessions if name in s["arms"]]
        if not per:
            continue
        node = {"mode": per[0]["mode"], "params": per[0]["params"],
                "median_ms": statistics.fmean(a["median_ms"] for a in per),
                "mean_p99_ms": statistics.fmean(a["p99_ms"] for a in per),
                "max_p99_ms": max(a["p99_ms"] for a in per),
                "session_p99_spread": max(a["p99_ms"] for a in per) /
                                      max(1e-9, min(a["p99_ms"] for a in per))}
        for b in BUDGETS_MS:
            node[f"verdict_{b:g}ms"] = verdict(node["max_p99_ms"], b)
            node[f"budget_used_pct_{b:g}ms"] = 100.0 * node["max_p99_ms"] / b
        summary["arms"][name] = node
        print(name.rjust(11) + node["mode"].rjust(17) + f"{node['params']:,}".rjust(9)
              + f"{node['median_ms']:.3f}".rjust(9) + f"{node['mean_p99_ms']:.3f}".rjust(9)
              + f"{node['max_p99_ms']:.3f}".rjust(10)
              + node["verdict_10ms"].rjust(10) + node["verdict_1ms"].rjust(10))

    spread = max((n["session_p99_spread"] for n in summary["arms"].values()), default=1.0)
    summary["worst_session_p99_spread"] = spread
    print(f"\n[b9c] worst session-to-session p99 spread: {spread:.2f}x "
          f"(§8ac saw 4.2x when a session was contaminated)")
    out = DATA / ".b9_latency.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[b9c] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, help="run one session (1..5)")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--all", action="store_true", help="5 sessions in separate processes")
    args = ap.parse_args()
    if args.session:
        run_session(args.session)
    elif args.all:
        for i in range(1, SESSIONS + 1):
            # Separate processes: a session must not inherit the previous one's warmed
            # allocator, thread pool or CPU frequency state.
            subprocess.run([sys.executable, str(Path(__file__)), "--session", str(i)],
                           cwd=REPO_ROOT, check=True)
        aggregate()
    elif args.aggregate:
        aggregate()
    else:
        ap.error("one of --session / --all / --aggregate")


if __name__ == "__main__":
    main()
