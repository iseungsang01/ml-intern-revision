"""Run the trained VQC on IonQ hardware (inference only) and quantify what shot noise costs.

Why inference only: training on a QPU is arithmetically out of reach here. Parameter-shift
needs 2 circuit evaluations per parameter per sample; at batch 32 with 98 parameters that is
6,272 circuits per gradient step, and at 100 shots each ~= 627,200 shots. At IonQ Forte gate
speeds (1q 130 us, 2q ~600 us) one shot of this circuit takes ~40 ms, so a SINGLE gradient
step costs roughly 7 hours of QPU time -- more than an entire research-credit allocation.
So the circuit is trained classically (``quantum_vqc.py``) and only *evaluated* on hardware.

What this script measures: the gap between the noiseless simulator prediction and the real
hardware prediction, on identical inputs and identical circuit parameters. That gap is the
combination of (a) finite-shot sampling noise and (b) gate/readout error. It is the number
that decides whether a QPU can resolve this task's effect size at all.

SAFETY: defaults to IonQ's cloud **simulator** (no QPU time consumed). Real hardware requires
``--hardware`` AND a printed budget you must confirm with ``--yes``.

Run (from repo root)::

    py ces_prediction/ionq_infer.py                       # cloud simulator, free
    py ces_prediction/ionq_infer.py --hardware            # prints budget, then stops
    py ces_prediction/ionq_infer.py --hardware --yes      # actually spends QPU time

Environment::

    IONQ_API_KEY      required (loaded from .env if present)
    IONQ_N_SAMPLES    samples to run   (default 24)
    IONQ_SHOTS        shots per sample (default 500)
    IONQ_QPU_BACKEND  hardware target  (default forte-1)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pennylane as qml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))  # ces_prediction/

from dataset import KSTAR_CES_Dataset, select_seeded_random_indices  # noqa: E402,F401
from evaluate import _load_stats  # noqa: E402
from quantum_vqc import (  # noqa: E402
    VQCRegressor, apply_pca, load_split_tensors, predict_batched, score, TARGET_NAMES,
)

# IonQ Forte gate durations (published): 1-qubit 130 us, 2-qubit ~600 us.
GATE_TIME_1Q_S = 130e-6
GATE_TIME_2Q_S = 600e-6


def load_dotenv(root):
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def circuit_gate_counts(n_qubits, n_layers):
    """Gates in one shot of the data-re-uploading circuit.

    Per layer: AngleEmbedding = n_qubits 1q rotations; StronglyEntanglingLayers = n_qubits
    Rot gates (3 rotations each) + n_qubits CNOTs.
    """
    one_q = n_layers * (n_qubits + 3 * n_qubits)
    two_q = n_layers * n_qubits
    return one_q, two_q


def estimate_budget(n_qubits, n_layers, n_samples, shots):
    one_q, two_q = circuit_gate_counts(n_qubits, n_layers)
    shot_seconds = one_q * GATE_TIME_1Q_S + two_q * GATE_TIME_2Q_S
    total_shots = n_samples * shots
    qpu_seconds = total_shots * shot_seconds
    # AWS Braket IonQ Forte list price: $0.30/task + $0.08/shot (1 task == 1 sample here).
    usd = n_samples * 0.30 + total_shots * 0.08
    return {
        "gates_1q_per_shot": one_q,
        "gates_2q_per_shot": two_q,
        "seconds_per_shot": shot_seconds,
        "total_shots": total_shots,
        "qpu_seconds": qpu_seconds,
        "qpu_minutes": qpu_seconds / 60.0,
        "braket_list_usd": usd,
        "shot_noise_sigma": 1.0 / np.sqrt(shots),
    }


def build_qnode(device, n_qubits, n_layers):
    @qml.qnode(device)
    def circuit(inputs, weights):
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[layer: layer + 1], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))
    return circuit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true", help="target real QPU instead of cloud simulator")
    ap.add_argument("--yes", action="store_true", help="confirm spending QPU time")
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir)
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        raise SystemExit("IONQ_API_KEY not set (checked environment and .env).")

    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", root_dir / "data" / "splits"))
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))

    n_samples = int(os.getenv("IONQ_N_SAMPLES", "24"))
    shots = int(os.getenv("IONQ_SHOTS", "500"))
    qpu_backend = os.getenv("IONQ_QPU_BACKEND", "forte-1")

    ckpt_path = output_dir / "quantum_vqc_weights.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing {ckpt_path}. Run quantum_vqc.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_qubits, n_layers = ckpt["n_qubits"], ckpt["n_layers"]
    target_name = ckpt["target"]
    t_idx = TARGET_NAMES.index(target_name)
    window_size = ckpt["window_size"]

    budget = estimate_budget(n_qubits, n_layers, n_samples, shots)
    print(f"Circuit: {n_qubits} qubits x {n_layers} layers  "
          f"({budget['gates_1q_per_shot']} 1q + {budget['gates_2q_per_shot']} 2q gates/shot)")
    print(f"Plan:    {n_samples} samples x {shots} shots = {budget['total_shots']:,} shots")
    print(f"Est. QPU time: {budget['qpu_minutes']:.1f} min   "
          f"(Braket list price ~= ${budget['braket_list_usd']:,.2f})")
    print(f"Shot-noise sigma on <Z>: {budget['shot_noise_sigma']:.4f}")

    if args.hardware and not args.yes:
        print("\n--hardware requested but --yes not given. Nothing submitted. "
              "Re-run with --yes to spend QPU time.")
        return

    # --- data (identical val samples as evaluate.py / quantum_vqc.py)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    print("\nBuilding val tensors...")
    _, _, val, _, _ = load_split_tensors(
        data_dir, window_size, stats, manifest, 1, max_val, seed
    )
    zval = apply_pca(val["x"], ckpt["pca_mean"], ckpt["pca_comps"], ckpt["pca_scale"])

    # Restrict to rows where the target is observed AND persistence exists, so every
    # hardware shot is spent on a sample that actually contributes to the metric.
    usable = ((val["mask"][:, t_idx] > 0.5) & val["persist_obs"][:, t_idx]).nonzero().squeeze(-1)
    g = torch.Generator().manual_seed(seed + 777)
    pick = usable[torch.randperm(usable.numel(), generator=g)[:n_samples]]
    print(f"  selected {pick.numel()} of {usable.numel()} usable val samples")

    weights = ckpt["weights"]
    out_scale, out_bias = ckpt["out_scale"], ckpt["out_bias"]

    # --- reference: exact, noiseless
    exact_dev = qml.device("default.qubit", wires=n_qubits)
    exact_circuit = build_qnode(exact_dev, n_qubits, n_layers)
    exact_exp = torch.stack([
        torch.as_tensor(float(exact_circuit(zval[i], weights))) for i in pick
    ])

    # --- IonQ
    backend_label = f"ionq.qpu ({qpu_backend})" if args.hardware else "ionq.simulator"
    if args.hardware:
        dev = qml.device("ionq.qpu", backend=qpu_backend, wires=n_qubits,
                         shots=shots, api_key=api_key)
    else:
        dev = qml.device("ionq.simulator", wires=n_qubits, shots=shots, api_key=api_key)
    ionq_circuit = build_qnode(dev, n_qubits, n_layers)

    print(f"\nSubmitting {pick.numel()} circuits to {backend_label}...")
    ionq_exp, t0 = [], time.time()
    for j, i in enumerate(pick):
        val_exp = float(ionq_circuit(zval[i], weights))
        ionq_exp.append(val_exp)
        print(f"  [{j + 1:3d}/{pick.numel()}] <Z0>_hw={val_exp:+.4f}  "
              f"exact={float(exact_exp[j]):+.4f}  diff={val_exp - float(exact_exp[j]):+.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    ionq_exp = torch.tensor(ionq_exp, dtype=torch.float32)
    elapsed = time.time() - t0

    dev_exp = (ionq_exp - exact_exp)
    print(f"\n<Z0> deviation hardware vs exact: mean {float(dev_exp.mean()):+.4f}, "
          f"std {float(dev_exp.std()):.4f}  (theoretical shot noise {budget['shot_noise_sigma']:.4f})")

    # --- score both in physical units on the SAME picked samples
    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)
    sub = {k: v[pick] for k, v in val.items()}
    res_exact = score(out_scale * exact_exp + out_bias, sub, t_idx, target_mean, target_std)
    res_ionq = score(out_scale * ionq_exp + out_bias, sub, t_idx, target_mean, target_std)

    hdr = f"{'run':<22} {'n':>5} {'RMSE':>11} {'RMSE_persist':>13} {'skill_vs_persist':>17}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, r in (("simulator (exact)", res_exact), (backend_label, res_ionq)):
        print(f"{label:<22} {r['n']:>5} {r['rmse_model']:>11.4f} "
              f"{r['rmse_persistence']:>13.4f} {r['skill_vs_persistence']:>17.4f}")

    report = {
        "backend": backend_label,
        "hardware": bool(args.hardware),
        "target": target_name,
        "n_samples": int(pick.numel()),
        "shots_per_sample": shots,
        "budget": budget,
        "elapsed_seconds": elapsed,
        "expval_deviation_mean": float(dev_exp.mean()),
        "expval_deviation_std": float(dev_exp.std()),
        "exact": res_exact,
        "ionq": res_ionq,
    }
    out_path = output_dir / ("ionq_hardware_result.json" if args.hardware
                             else "ionq_simulator_result.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
