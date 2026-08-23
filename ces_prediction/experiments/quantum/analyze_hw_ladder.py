"""Decide the shot ladder: is the hardware error shot-limited, or is there a gate-error floor?

Reads ``ionq_hw_ladder_result.json`` (written by ``ionq_hw_ladder.py``) and fits the measured
deviation of the hardware <Z_0> from the exact noiseless value against shot count.

The model
---------
Two independent error sources add in quadrature:

    rms_deviation(N)^2  =  a^2 / N  +  b^2

``a`` is the sampling coefficient. For an unbiased estimator of <Z> from N shots,
Var = (1 - <Z>^2)/N, so a ~= 1 at our near-zero operating points -- a is therefore a
*prediction*, not a free knob, and a fitted a far from 1 means something else is wrong.

``b`` is the part more shots cannot remove: gate error, readout error, drift. It is the number
that decides the whole question.

    b ~ 0                -> shot-limited. Buying more shots keeps helping (offline only).
    b >= the task signal -> permanent. No shot count ever resolves this task on this hardware.

The verdict compares b against what the task actually needs, converted through the trained
circuit's own output scaling into physical CES_TI units.

Run from the repo root::

    py ces_prediction/experiments/quantum/analyze_hw_ladder.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def quadrature_fit(shots, rms):
    """Least-squares fit of rms^2 = a^2/N + b^2, linear in (1/N, 1).

    Fitting the squares keeps the model linear, so this is a closed-form solve rather than an
    optimiser with a starting guess.
    """
    x = np.asarray([1.0 / n for n in shots], dtype=float)
    y = np.asarray(rms, dtype=float) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a2, b2), *_ = np.linalg.lstsq(A, y, rcond=None)
    return math.sqrt(max(a2, 0.0)), math.sqrt(max(b2, 0.0))


def population_window(n_qubits, n_points=1500, seed=42):
    """p1-p99 span of the noiseless <Z_0> over many real val operating points.

    Measured on a handful of points this is badly biased low -- a 3-point range gave 0.088
    where 1,500 points give 0.578. Always take the window from the population.
    """
    import sys
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(ROOT / "ces_prediction"))
    import pennylane as qml
    from evaluate import _load_stats
    from quantum_vqc import apply_pca, fit_pca, load_split_tensors

    ckpt = torch.load(HERE / "quantum_vqc_weights.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((ROOT / "ces_prediction" / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    _, tr, va, _, _ = load_split_tensors(ROOT / "data", int(ckpt["window_size"]),
                                         _load_stats(metrics), manifest, 2000, 4000, seed)
    mean, comps, scale = fit_pca(tr["x"], n_qubits)
    z = apply_pca(va["x"], mean, comps, scale).numpy()

    w = ckpt["weights"]
    n_layers = int(ckpt["n_layers"])
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inp, ww):
        for layer in range(n_layers):
            qml.AngleEmbedding(inp, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(ww[layer:layer + 1], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    vals = np.array([float(circuit(torch.as_tensor(z[i]), w))
                     for i in range(min(n_points, len(z)))])
    return float(np.percentile(vals, 99) - np.percentile(vals, 1))


def main():
    res_path = HERE / "ionq_hw_ladder_result.json"
    if not res_path.exists():
        raise SystemExit("no result file yet: %s" % res_path)
    data = json.loads(res_path.read_text(encoding="utf-8"))
    recs = [r for r in data["records"] if r.get("on_hardware") and r.get("status") == "completed"]
    if not recs:
        raise SystemExit("no completed hardware records in %s" % res_path)

    ckpt = torch.load(HERE / "quantum_vqc_weights.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((ROOT / "ces_prediction" / "metrics.json").read_text(encoding="utf-8"))
    vqc_res = json.loads((HERE / "quantum_vqc_result.json").read_text(encoding="utf-8"))
    out_scale = abs(float(ckpt["out_scale"]))
    ti_std = float(metrics["normalization"]["stats"]["target"]["std"][0])

    def to_ev(dz):
        return dz * out_scale * ti_std

    # ---- per-shot-level statistics -------------------------------------------------
    by_shots = {}
    for r in recs:
        by_shots.setdefault(r["shots"], []).append(r["z0_little_endian"] - r["exact_z0"])

    print("=== measured deviation of hardware <Z_0> from the exact value ===")
    print("%-8s %-4s %-11s %-11s %-11s %-11s" % ("shots", "n", "mean", "rms", "theory 1/sqrtN", "rms in eV"))
    levels = sorted(by_shots)
    rms_by_level = []
    for s in levels:
        d = np.asarray(by_shots[s], dtype=float)
        rms = float(np.sqrt((d ** 2).mean()))
        rms_by_level.append(rms)
        print("%-8d %-4d %+11.5f %-11.5f %-11.5f %-11.1f"
              % (s, len(d), d.mean(), rms, 1.0 / math.sqrt(s), to_ev(rms)))

    # A two-parameter model fitted to two levels is an exact fit with zero residual: it will
    # report some a and b whatever the data says. Refuse a verdict until the fit is actually
    # over-determined and each level has enough samples for its rms to mean anything.
    min_per_level = min(len(by_shots[s]) for s in levels)
    enough = len(levels) >= 3 and min_per_level >= 5
    if not enough:
        print("\nTOO THIN FOR A VERDICT: %d shot level(s), %d sample(s) at the smallest level."
              % (len(levels), min_per_level))
        print("Need >=3 levels and >=5 samples/level. With 2 levels the quadrature fit is exactly")
        print("determined (zero residual), so any a and b it prints are artefacts, not findings.")
        if len(levels) < 2:
            return

    a, b = quadrature_fit(levels, rms_by_level)
    print("\n=== quadrature fit  rms^2 = a^2/N + b^2 ===")
    print("  a (sampling coefficient) = %.4f   [prediction ~1.0; far off means a problem]" % a)
    print("  b (irreducible floor)    = %.5f   = %.1f eV" % (b, to_ev(b)))

    # Bootstrap over samples within each level, so the reported floor carries an interval.
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        r = []
        for s in levels:
            d = np.asarray(by_shots[s], dtype=float)
            r.append(float(np.sqrt((rng.choice(d, size=len(d), replace=True) ** 2).mean())))
        boots.append(quadrature_fit(levels, r))
    ba = np.array([x[0] for x in boots]); bb = np.array([x[1] for x in boots])
    print("  bootstrap 95%% CI: a [%.4f, %.4f]   b [%.5f, %.5f] = [%.1f, %.1f] eV"
          % (np.percentile(ba, 2.5), np.percentile(ba, 97.5),
             np.percentile(bb, 2.5), np.percentile(bb, 97.5),
             to_ev(np.percentile(bb, 2.5)), to_ev(np.percentile(bb, 97.5))))
    floor_resolved = float(np.percentile(bb, 2.5)) > 0.0

    # Consistency with the pure-shot-noise null (b = 0, a = 1): is a floor needed at all?
    chi2 = sum(len(by_shots[s]) * (r ** 2) / ((1.0 - np.mean(np.asarray(by_shots[s])) ** 2) / s)
               for s, r in zip(levels, rms_by_level))
    dof = sum(len(by_shots[s]) for s in levels)
    print("  vs pure shot noise (b=0, a=1): chi2/dof = %.3f over %d measurements" % (chi2 / dof, dof))
    if chi2 / dof < 1.0:
        print("    -> measured scatter is BELOW the sampling prediction; no floor is indicated,")
        print("       and an under-dispersed chi2 usually means too few samples, not a good QPU.")

    # ---- what the task needs -------------------------------------------------------
    # The output window must come from the POPULATION, not from the handful of measured
    # points: a 3-sample range understated it by 7x once already. p1-p99 over many val
    # points is the honest figure.
    span = population_window(int(ckpt["n_qubits"]))
    persistence = float(vqc_res["vqc"]["rmse_persistence"])
    mlp_rmse = float(vqc_res["matched_mlp"]["rmse_model"])

    print("\n=== what the task needs ===")
    print("  circuit output window, p1-p99       : %.4f  = %.1f eV" % (span, to_ev(span)))
    print("  persistence baseline RMSE           : %.1f eV" % persistence)
    print("  classical matched MLP RMSE          : %.1f eV" % mlp_rmse)
    print("  VQC RMSE (simulator, exact probs)   : %.1f eV" % float(vqc_res["vqc"]["rmse_model"]))
    print("  floor b as a fraction of the window : %.0f%%" % (b / span * 100 if span else float("nan")))

    print("\n=== verdict ===")
    if not enough:
        print("  WITHHELD -- see the thinness warning above. Collect the full ladder first.")
    elif not floor_resolved:
        print("  SHOT-LIMITED: the bootstrap interval for b reaches 0, so no floor is resolved")
        print("  above sampling noise at these shot counts. More shots would keep helping, which")
        print("  is an offline-only consolation: each 4x in precision costs 4x the shots.")
    else:
        print("  GATE-ERROR FLOOR at b = %.5f (%.1f eV), CI excludes zero." % (b, to_ev(b)))
        print("  Sampling stops dominating past ~%.0f shots; beyond that more shots buy nothing."
              % ((a / b) ** 2))
        print("  The floor alone is %.0f%% of the circuit's entire output window."
              % (b / span * 100 if span else float("nan")))
        print("  This is a property of the hardware, not of the shot budget.")

    print("\n  (n = %d hardware measurements across %d shot levels, %s)"
          % (len(recs), len(levels), ", ".join(str(s) for s in levels)))


if __name__ == "__main__":
    main()
