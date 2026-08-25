"""Free pre-verification: is a QUANTUM feature map worth buying hardware time for?

The rule this exists to satisfy
-------------------------------
THESIS_RESULTS.md §8ap closed the variational-circuit (VQC) route and stated the condition for
reopening the quantum arm on performance: *a circuit family must beat a parameter-matched
classical model in simulation first*. Simulation is free; hardware is $25.79 per circuit. So
that check belongs here, before any money moves.

Why this structure and not another VQC
--------------------------------------
§8z established that the backbone's `T_i` skill decomposes exactly as

    prediction = persistence anchor + sum_k w_k z_k + b,   z bounded in [-1, 1], K = 8

Quantum expectation values are natively bounded in [-1, 1], so a quantum feature map slots into
that same shape. Two consequences make it a better candidate than the VQC:

  * the readout is LINEAR, so it is fitted by ridge regression in closed form -- no gradients,
    which removes the parameter-shift wall that made VQC training impossible on hardware;
  * the circuit is FIXED (not trained), so there is nothing to optimise on the device.

The control that decides it
---------------------------
A quantum feature map must be compared against a CLASSICAL RANDOM feature map of the same width
on the same inputs with the same readout. Without that control the experiment only measures
"does having K features help", which is not the question. Arms:

    quantum   : x -> angles -> fixed random circuit -> K expectation values
    classical : x -> fixed random projection -> tanh -> K features
    anchor    : persistence alone (K = 0), the floor both must clear

Everything downstream is identical: same rows, same anchor, same ridge readout, same metric
(`skill_vs_persistence` in physical CES units on the val split `evaluate.py` uses).

Run from the repo root::

    py ces_prediction/experiments/quantum/qfeature_probe.py
    py ces_prediction/experiments/quantum/qfeature_probe.py --qubits 10 --layers 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "ces_prediction"))

from evaluate import _load_stats  # noqa: E402
from quantum_vqc import apply_pca, fit_pca, load_split_tensors  # noqa: E402


def quantum_features(z, n_qubits, n_layers, seed, batch=512):
    """Fixed (untrained) circuit -> K expectation values per row.

    Data re-uploading with a random-but-fixed entangling block. The weights are frozen: this is
    a feature map, not a model, so nothing here is optimised.
    """
    import pennylane as qml
    g = torch.Generator().manual_seed(seed)
    w = (torch.rand(n_layers, n_qubits, 3, generator=g) * 2 - 1) * np.pi
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inp):
        for layer in range(n_layers):
            qml.AngleEmbedding(inp, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(w[layer:layer + 1], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    out = []
    for i in range(0, len(z), batch):
        out.append(torch.stack(circuit(torch.as_tensor(z[i:i + batch])), dim=-1))
    return torch.cat(out).numpy()


def classical_features(z, k, seed):
    """Fixed random projection + tanh -> k features. The honest control for the quantum map."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(z.shape[1]), size=(z.shape[1], k))
    b = rng.normal(0, 0.3, size=k)
    return np.tanh(z @ W + b)


def ridge_on_anchor(feat_tr, anch_tr, y_tr, feat_va, anch_va, alpha):
    """Fit y - anchor = feat @ w + b by ridge; return val predictions. b3's decomposition shape."""
    if feat_tr is None:
        return np.full(len(anch_va), 0.0) + anch_va
    A = np.hstack([feat_tr, np.ones((len(feat_tr), 1))])
    r = y_tr - anch_tr
    reg = alpha * np.eye(A.shape[1]); reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ r)
    B = np.hstack([feat_va, np.ones((len(feat_va), 1))])
    return anch_va + B @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--target", type=str, default="CES_TI")
    args = ap.parse_args()

    t_idx = ("CES_TI", "CES_VT").index(args.target)
    metrics = json.loads((ROOT / "ces_prediction" / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    tstd = float(stats["target"]["std"][t_idx])

    _, tr, va, _, _ = load_split_tensors(
        ROOT / "data", args.window, stats, manifest, args.train, args.val, 42)

    K = args.qubits
    mean, comps, scale = fit_pca(tr["x"], K)
    ztr = apply_pca(tr["x"], mean, comps, scale).numpy()
    zva = apply_pca(va["x"], mean, comps, scale).numpy()

    def prep(split):
        y = split["target"][:, t_idx].numpy()
        m = (split["mask"][:, t_idx] > 0.5).numpy()
        anchor = split["persist"][:, t_idx].numpy()
        obs = (split["persist_obs"][:, t_idx] > 0.5).numpy()
        keep = m & obs & np.isfinite(anchor) & np.isfinite(y)
        return y, anchor, keep

    ytr, atr, ktr = prep(tr)
    yva, ava, kva = prep(va)
    print("target %s   window=%d   qubits/features K=%d   layers=%d"
          % (args.target, args.window, K, args.layers))
    print("usable rows: train %d / %d, val %d / %d" % (ktr.sum(), len(ktr), kva.sum(), len(kva)))

    rmse_persist = float(np.sqrt(np.mean((ava[kva] - yva[kva]) ** 2)) * tstd)
    print("persistence RMSE: %.2f eV  (the floor both arms must clear)\n" % rmse_persist)

    print("computing quantum features (fixed circuit, no training) ...")
    qtr = quantum_features(ztr, K, args.layers, seed=7)
    qva = quantum_features(zva, K, args.layers, seed=7)

    ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    rows = []
    for name, ftr, fva, seeds in (
            ("anchor only", None, None, [0]),
            ("quantum map", qtr, qva, [7]),
            ("classical random", None, None, list(range(args.seeds)))):
        best = None
        for s in seeds:
            if name == "classical random":
                ftr_s, fva_s = classical_features(ztr, K, s), classical_features(zva, K, s)
            else:
                ftr_s, fva_s = ftr, fva
            for a in ALPHAS:
                if ftr_s is None:
                    p = ava[kva]
                else:
                    p = ridge_on_anchor(ftr_s[ktr], atr[ktr], ytr[ktr], fva_s[kva], ava[kva], a)
                rmse = float(np.sqrt(np.mean((p - yva[kva]) ** 2)) * tstd)
                sk = 1.0 - (rmse / rmse_persist) ** 2
                if best is None or sk > best[0]:
                    best = (sk, rmse, a, s)
                if ftr_s is None:
                    break
        rows.append((name, best))
        print("%-18s skill %+0.4f   RMSE %7.2f eV   (alpha %g, seed %s)"
              % (name, best[0], best[1], best[2], best[3]))

    q = dict((r[0], r[1]) for r in rows)["quantum map"]
    c = dict((r[0], r[1]) for r in rows)["classical random"]
    print("\n=== verdict ===")
    gap = q[0] - c[0]
    if gap > 0.01:
        print("  quantum map beats the classical random control by %+0.4f skill." % gap)
        print("  This clears §8ap's condition -- worth pricing hardware for.")
    else:
        print("  quantum map does NOT beat the classical random control (%+0.4f)." % gap)
        print("  Both are fixed nonlinear maps of the same width into the same linear readout,")
        print("  so any quantum advantage would have to show up here, where it is free to look.")
        print("  §8ap's condition is NOT met: do not spend hardware credit on this family.")


if __name__ == "__main__":
    main()
