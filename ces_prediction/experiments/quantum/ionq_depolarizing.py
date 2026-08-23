"""Measure the depolarizing coefficient of IonQ Forte on the trained CES circuit.

Why
---
`THESIS_RESULTS.md` §8ap and the experiment log §12.1 explain an apparent paradox: the
hardware's `<Z_0>` deviation looks small even though the full 256-state distribution sits
1.5-2.0x further from ideal than a perfect sampler. The proposed explanation is that
depolarizing-type error pulls the state toward maximally mixed, whose `<Z_0>` is exactly 0,
and our operating points already sit near 0 -- so there is little for the error to move.

That is an *inference*. This script measures it.

Under a depolarizing channel of strength lambda the measured expectation is a pure contraction:

    <Z_0>_measured = (1 - lambda) * <Z_0>_exact

so a straight-line fit over operating points spanning a wide range of `<Z_0>_exact` returns
lambda directly from the slope. The circuit is the trained VQC (8 qubits, 4 layers, identical
weights), so the error budget matches the shot ladder exactly and the two results compose.

The operating points are NOT val samples. Real val inputs span only `<Z_0>` in
[-0.31, +0.48]; optimizing the encoded angles inside the same +-pi/2 box the PCA squash
produces reaches [-0.57, +0.79], which is 1.7x the leverage for the same money. These are
device-characterization points, and the script says so in its output.

Falsifiable either way: a slope near 1 with scattered residuals means the error is NOT
depolarizing (coherent or calibration drift instead), which is equally worth knowing and
would invalidate the §12.1 explanation rather than confirm it.

Cost: one job per point at 400 shots, the $25.79 tier. 18 points = $464.22.

Run from the repo root::

    py ces_prediction/experiments/quantum/ionq_depolarizing.py                    # free sim check
    py ces_prediction/experiments/quantum/ionq_depolarizing.py --hardware --yes --submit-only
    py ces_prediction/experiments/quantum/ionq_depolarizing.py --hardware --collect
    py ces_prediction/experiments/quantum/ionq_depolarizing.py --analyze
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "ces_prediction"))

from ionq_hw_ladder import (  # noqa: E402
    IonQ, build_circuit, exact_z0, expval_z0, load_api_key, PRICE_LOW,
)

ANGLE_LIMIT = np.pi / 2   # the box apply_pca's tanh squash maps into


def solve_angles(target_z, weights, n_qubits, seed, iters=400):
    """Find encoded angles inside the +-pi/2 box whose noiseless <Z_0> equals target_z."""
    import pennylane as qml
    n_layers = weights.shape[0]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inp, w):
        for layer in range(n_layers):
            qml.AngleEmbedding(inp, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(w[layer:layer + 1], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    w = torch.as_tensor(weights)
    best = None
    for trial in range(4):
        g = torch.Generator().manual_seed(seed * 100 + trial)
        raw = ((torch.rand(n_qubits, generator=g) * 2 - 1) * 0.8).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=0.15)
        for _ in range(iters):
            ang = torch.tanh(raw) * ANGLE_LIMIT
            loss = (circuit(ang, w) - target_z) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        ang = (torch.tanh(raw) * ANGLE_LIMIT).detach()
        err = abs(float(circuit(ang, w)) - target_z)
        if best is None or err < best[0]:
            best = (err, ang.numpy().copy())
    return best[1], best[0]


def build_points(n_points, weights, n_qubits, lo, hi):
    targets = np.linspace(lo, hi, n_points)
    pts = []
    for i, t in enumerate(targets):
        ang, err = solve_angles(float(t), weights, n_qubits, seed=i)
        got = exact_z0(ang, weights)
        pts.append({"index": i, "target_z": float(t), "exact_z0": float(got),
                    "solve_error": float(err), "angles": [float(a) for a in ang]})
        print("  point %2d  target %+.4f -> exact %+.4f  (miss %.1e)" % (i, t, got, err))
    return pts


def label_for(i):
    return "dep%02d_400sh" % i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--submit-only", dest="submit_only", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--points", type=int, default=18)
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--lo", type=float, default=-0.55)
    ap.add_argument("--hi", type=float, default=0.77)
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--backend", type=str, default="qpu.forte-1")
    ap.add_argument("--out", type=str, default=str(HERE / "ionq_depolarizing_result.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    ckpt = torch.load(HERE / "quantum_vqc_weights.pt", map_location="cpu", weights_only=False)
    weights = ckpt["weights"].numpy()
    n_qubits = int(ckpt["n_qubits"])

    if args.analyze:
        return analyze(out_path, n_qubits)

    state = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    points = state.get("points")
    if not points:
        print("Solving %d operating points spanning <Z_0> in [%.2f, %.2f] ..."
              % (args.points, args.lo, args.hi))
        points = build_points(args.points, weights, n_qubits, args.lo, args.hi)
        state = {"points": points, "shots": args.shots, "backend": args.backend,
                 "n_qubits": n_qubits, "measurements": state.get("measurements", {})}
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    api = IonQ(load_api_key())
    target = args.backend if args.hardware else "simulator"
    meas = state.get("measurements", {})

    # Reclaim anything already paid for, from disk and from IonQ's job history.
    if args.hardware:
        want = {label_for(p["index"]): p["index"] for p in points}
        _, hist = api.call("GET", "/jobs?limit=100")
        for j in hist.get("jobs", []):
            name = str(j.get("name") or "")
            if (j.get("dry_run") or j.get("status") != "completed"
                    or name not in want or str(want[name]) in meas):
                continue
            full = api.call("GET", "/jobs/" + j["id"])[1]
            if full.get("backend") != args.backend:
                continue
            probs = api.probabilities(full)
            if not probs:
                continue
            i = want[name]
            meas[str(i)] = {"job_id": j["id"], "backend": full.get("backend"),
                            "z0": expval_z0(probs, n_qubits, True),
                            "exec_ms": full.get("execution_duration_ms"),
                            "cost_usd": api.cost(j["id"])}
            print("  reclaimed %s (point %d) from IonQ history" % (name, i))

    todo = [p for p in points if str(p["index"]) not in meas]
    print("\n%d points total, %d measured, %d to buy (est $%.2f, cap $%.2f)"
          % (len(points), len(points) - len(todo), len(todo), len(todo) * PRICE_LOW, args.budget))

    if args.collect:
        state["measurements"] = meas
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("Collected %d of %d ($%.2f accounted)."
              % (len(meas), len(points), sum(float(m.get("cost_usd") or 0) for m in meas.values())))
        return

    if args.hardware:
        status, degraded = api.backend_status(args.backend)
        print("Backend status: %s (degraded=%s)" % (status, degraded))
        if status != "available":
            raise SystemExit("%s is %s -- doing nothing." % (args.backend, status))
        if len(todo) * PRICE_LOW > args.budget:
            raise SystemExit("estimate exceeds --budget")
        if not args.yes:
            raise SystemExit("Dry stop: re-run with --yes to spend.")

    committed = 0.0
    for p in todo:
        circ = build_circuit(np.array(p["angles"]), weights)
        if args.hardware and args.submit_only:
            if committed + PRICE_LOW > args.budget:
                print("STOP: budget cap reached"); break
            jid = api.submit(circ, n_qubits, args.shots, target, False, label_for(p["index"]))
            committed += PRICE_LOW
            print("  queued %-14s %s (committed $%.2f)" % (label_for(p["index"]), jid, committed))
            continue
        jid = api.submit(circ, n_qubits, args.shots, target, False, label_for(p["index"]))
        job = api.wait(jid)
        probs = api.probabilities(job)
        meas[str(p["index"])] = {
            "job_id": jid, "backend": job.get("backend"),
            "z0": expval_z0(probs, n_qubits, True) if probs else float("nan"),
            "exec_ms": job.get("execution_duration_ms"),
            "cost_usd": (api.cost(jid) or 0.0) if args.hardware else 0.0}
        state["measurements"] = meas
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("  point %2d exact %+.4f  measured %+.4f  backend=%s"
              % (p["index"], p["exact_z0"], meas[str(p["index"])]["z0"], job.get("backend")))

    state["measurements"] = meas
    out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if args.submit_only:
        print("\nQueued %d job(s), committing $%.2f. Machine can be shut down; collect later with"
              % (len(todo), committed))
        print("  --hardware --collect      then      --analyze")


def analyze(out_path, n_qubits):
    state = json.loads(out_path.read_text(encoding="utf-8"))
    pts = {p["index"]: p for p in state["points"]}
    rows = [(pts[int(k)]["exact_z0"], m["z0"]) for k, m in state["measurements"].items()
            if int(k) in pts and np.isfinite(m.get("z0", float("nan")))]
    if len(rows) < 3:
        raise SystemExit("only %d measurements; need at least 3" % len(rows))
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    n = len(x)

    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    s_res = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1)))
    sxx = float(((x - x.mean()) ** 2).sum())
    se_slope = s_res / np.sqrt(sxx) if sxx > 0 else float("nan")
    shot_sigma = 1.0 / np.sqrt(state["shots"])

    print("=== depolarizing fit:  <Z_0>_measured = (1 - lambda) * <Z_0>_exact + c ===")
    print("  n points            : %d   spanning exact <Z_0> [%+.4f, %+.4f]" % (n, x.min(), x.max()))
    print("  slope (1 - lambda)  : %.4f  +- %.4f" % (slope, se_slope))
    print("  lambda              : %.4f  +- %.4f" % (1 - slope, se_slope))
    print("  intercept           : %+.4f" % intercept)
    print("  residual sigma      : %.4f   (pure shot noise at %d shots = %.4f)"
          % (s_res, state["shots"], shot_sigma))
    print()
    # Test lambda > 0 directionally. A two-sided |1 - slope| test calls lambda < 0 a
    # "confirmed contraction", which is backwards: a slope above 1 is not a contraction at all.
    lam = 1 - slope
    z = lam / se_slope if se_slope > 0 else float("inf")
    if lam <= 2 * se_slope:
        print("  VERDICT: lambda is not positive at 2 sigma -- the error is NOT a contraction.")
        print("  The depolarizing explanation in the log's section 12.1 does not survive; the")
        print("  excess scatter must come from a non-contracting mechanism (coherent error or")
        print("  drift). Update that section rather than citing it.")
    else:
        print("  VERDICT: contraction confirmed at %.1f sigma. lambda = %.3f, i.e. the hardware"
              % (z, 1 - slope))
        print("  returns %.0f%% of the circuit's intended signal amplitude." % (slope * 100))
        print("  This is a MULTIPLICATIVE degradation: it shrinks the usable output window")
        print("  rather than adding noise to it, which is why <Z_0> near 0 looked unharmed.")
    if s_res > 1.5 * shot_sigma:
        print("\n  NOTE: residuals (%.4f) exceed shot noise (%.4f); a contraction alone does not"
              % (s_res, shot_sigma))
        print("  explain everything, consistent with the ladder's a = 1.546.")


if __name__ == "__main__":
    main()
