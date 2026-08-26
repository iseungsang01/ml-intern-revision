"""Free sweep of every remaining QML family against a matched classical control.

Context
-------
§8ap closed variational circuits on hardware. Addendum 3 closed fixed feature maps for free and
found the dominant effect: **freezing the map costs ~92% of the achievable skill** (+0.0095
against a trained encoder's +0.126 on identical inputs). That finding dictates what is left to
test, and this runs all of it in simulation, where it costs nothing.

Every arm predicts `T_i` in the §8z decomposition — `anchor + correction` — is scored by
`skill_vs_persistence` in physical eV on the same val rows, and is paired with a classical
control of matched width and matched training budget. A quantum arm only counts if it beats
*its own control*, not merely the anchor.

Arms
----
    reservoir  fixed quantum dynamics carrying state across timesteps -> K observables
               control: echo-state network (fixed random recurrent tanh), same K
               -- the one family Addendum 3 left untested rather than closed

    kernel     quantum kernel ridge, K(x,x') = |<phi(x)|phi(x')>|^2
               control: RBF kernel ridge on identical rows
               -- the other major QML paradigm; O(N^2), so run on a subsample

    encoder    TRAINED quantum circuit -> K bounded latents -> linear readout on the anchor
               control: trained classical MLP encoder, same K, same epochs
               -- the strongest untried candidate, because it restores the learning that
                  Addendum 3 showed is where the skill lives, in the b3k8 shape

Run from the repo root::

    py ces_prediction/experiments/quantum/qfamilies_probe.py --arms reservoir,kernel
    py ces_prediction/experiments/quantum/qfamilies_probe.py --arms encoder
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "ces_prediction"))

from evaluate import _load_stats  # noqa: E402
from quantum_vqc import apply_pca, fit_pca, load_split_tensors  # noqa: E402

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- shared scoring
def ridge_fit_predict(ftr, rtr, fva, alpha):
    A = np.hstack([ftr, np.ones((len(ftr), 1))])
    reg = alpha * np.eye(A.shape[1]); reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ rtr)
    return np.hstack([fva, np.ones((len(fva), 1))]) @ coef


def best_skill(ftr, atr, ytr, fva, ava, yva, tstd, rmse_p):
    best = -9e9
    for a in ALPHAS:
        pred = ava + ridge_fit_predict(ftr, ytr - atr, fva, a)
        rmse = float(np.sqrt(np.mean((pred - yva) ** 2)) * tstd)
        best = max(best, 1.0 - (rmse / rmse_p) ** 2)
    return best


# --------------------------------------------------------------------------- reservoirs
def quantum_reservoir(seq, n_qubits, seed, batch=512):
    """Fixed quantum dynamics with state carried across timesteps -> K observables.

    Each step re-encodes the input into the SAME register without resetting it, so the register
    keeps a memory of earlier steps. That memory is the only thing distinguishing this from the
    per-row feature map already closed in Addendum 3.
    """
    import pennylane as qml
    g = torch.Generator().manual_seed(seed)
    w = (torch.rand(2, n_qubits, 3, generator=g) * 2 - 1) * np.pi
    dev = qml.device("lightning.qubit", wires=n_qubits)
    T = seq.shape[1]

    @qml.qnode(dev, interface="torch")
    def circuit(x):
        for t in range(T):
            qml.AngleEmbedding(x[:, t, :], wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(w[0:1], wires=range(n_qubits))
            qml.StronglyEntanglingLayers(w[1:2], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    out = []
    for i in range(0, len(seq), batch):
        out.append(torch.stack(circuit(torch.as_tensor(seq[i:i + batch])), dim=-1))
    return torch.cat(out).numpy()


def classical_reservoir(seq, k, seed, leak=0.6, rho=0.9):
    """Echo-state network: fixed random recurrent tanh state. The control for the above."""
    rng = np.random.default_rng(seed)
    din = seq.shape[2]
    Win = rng.normal(0, 1.0, size=(din, k)) / np.sqrt(din)
    W = rng.normal(0, 1.0, size=(k, k))
    W *= rho / max(abs(np.linalg.eigvals(W)).max(), 1e-9)
    h = np.zeros((len(seq), k))
    for t in range(seq.shape[1]):
        h = (1 - leak) * h + leak * np.tanh(seq[:, t, :] @ Win + h @ W)
    return h


# --------------------------------------------------------------------------- kernels
def quantum_kernel(a, b, n_qubits, batch=256):
    """K(x,x') = |<phi(x)|phi(x')>|^2 via the statevectors of the encoding circuit."""
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state(x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Z")
        return qml.state()

    def states(z):
        return np.stack([np.asarray(state(torch.as_tensor(r))) for r in z])

    sa, sb = states(a), states(b)
    return np.abs(sa.conj() @ sb.T) ** 2


def rbf_kernel(a, b, gamma):
    d = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T)
    return np.exp(-gamma * np.maximum(d, 0))


def kernel_ridge_skill(Ktr, Kva, atr, ytr, ava, yva, tstd, rmse_p):
    best = -9e9
    r = ytr - atr
    for a in ALPHAS:
        alpha_mat = Ktr + a * np.eye(len(Ktr))
        try:
            dual = np.linalg.solve(alpha_mat, r)
        except np.linalg.LinAlgError:
            continue
        pred = ava + Kva @ dual
        rmse = float(np.sqrt(np.mean((pred - yva) ** 2)) * tstd)
        best = max(best, 1.0 - (rmse / rmse_p) ** 2)
    return best


# --------------------------------------------------------------------------- trained encoders
def train_encoder_skill(kind, ztr, atr, ytr, zva, ava, yva, tstd, rmse_p,
                        k, epochs, seed, layers=2):
    """Trained encoder -> k bounded latents -> linear readout on the anchor (the b3k8 shape).

    Zero-init readout so training starts exactly at persistence, as §8z's recipe requires.
    """
    import pennylane as qml
    torch.manual_seed(seed)
    n_in = ztr.shape[1]

    if kind == "quantum":
        dev = qml.device("lightning.qubit", wires=k)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circ(inp, w):
            for L in range(layers):
                qml.AngleEmbedding(inp, wires=range(k), rotation="Y")
                qml.StronglyEntanglingLayers(w[L:L + 1], wires=range(k))
            return [qml.expval(qml.PauliZ(i)) for i in range(k)]

        theta = (0.1 * torch.randn(layers, k, 3)).requires_grad_(True)
        params = [theta]
        enc = lambda x: torch.stack(circ(x, theta), dim=-1)
    else:
        net = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, k), torch.nn.Tanh())
        params = list(net.parameters())
        enc = net

    w_out = torch.zeros(k, requires_grad=True)
    b_out = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(params + [w_out, b_out], lr=0.02)

    Xtr = torch.as_tensor(ztr, dtype=torch.float32)
    Rtr = torch.as_tensor(ytr - atr, dtype=torch.float32)
    n, bs = len(Xtr), 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            z = enc(Xtr[idx])
            loss = ((z @ w_out + b_out - Rtr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        zv = enc(torch.as_tensor(zva, dtype=torch.float32))
        pred = ava + (zv @ w_out + b_out).numpy()
    rmse = float(np.sqrt(np.mean((pred - yva) ** 2)) * tstd)
    return 1.0 - (rmse / rmse_p) ** 2


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="reservoir,kernel,encoder")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--kernel-n", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    metrics = json.loads((ROOT / "ces_prediction" / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    tstd = float(stats["target"]["std"][0])
    _, tr, va, _, _ = load_split_tensors(ROOT / "data", args.window, stats, manifest,
                                         args.train, args.val, 42)
    K = args.qubits
    mean, comps, scale = fit_pca(tr["x"], K)
    ztr = apply_pca(tr["x"], mean, comps, scale).numpy()
    zva = apply_pca(va["x"], mean, comps, scale).numpy()

    def prep(sp):
        y = sp["target"][:, 0].numpy(); m = (sp["mask"][:, 0] > 0.5).numpy()
        a = sp["persist"][:, 0].numpy(); o = (sp["persist_obs"][:, 0] > 0.5).numpy()
        return y, a, m & o & np.isfinite(a) & np.isfinite(y)

    ytr, atr, ktr = prep(tr); yva, ava, kva = prep(va)
    rmse_p = float(np.sqrt(np.mean((ava[kva] - yva[kva]) ** 2)) * tstd)
    print("window=%d  K=%d  train %d  val %d   persistence RMSE %.2f eV"
          % (args.window, K, ktr.sum(), kva.sum(), rmse_p))
    print("reference: a TRAINED classical encoder on these inputs reaches roughly +0.36 (W=4)\n")

    res = {}

    if "reservoir" in arms:
        print("[reservoir] fixed dynamics with memory across timesteps ...")
        str_ = tr["x"].numpy().reshape(len(ztr), args.window, -1)
        sva = va["x"].numpy().reshape(len(zva), args.window, -1)
        # project each timestep to K dims so both reservoirs see the same width
        pm, pc, ps = fit_pca(torch.as_tensor(str_.reshape(-1, str_.shape[2])), K)
        pj = lambda s: apply_pca(torch.as_tensor(s.reshape(-1, s.shape[2])), pm, pc, ps
                                 ).numpy().reshape(s.shape[0], s.shape[1], K)
        qtr, qva = quantum_reservoir(pj(str_), K, 7), quantum_reservoir(pj(sva), K, 7)
        res["reservoir - quantum"] = best_skill(qtr[ktr], atr[ktr], ytr[ktr], qva[kva],
                                                ava[kva], yva[kva], tstd, rmse_p)
        best_c = max(best_skill(classical_reservoir(pj(str_), K, s)[ktr], atr[ktr], ytr[ktr],
                                classical_reservoir(pj(sva), K, s)[kva], ava[kva], yva[kva],
                                tstd, rmse_p) for s in range(args.seeds))
        res["reservoir - classical ESN"] = best_c

    if "kernel" in arms:
        print("[kernel] quantum kernel ridge vs RBF, subsampled to %d rows ..." % args.kernel_n)
        rng = np.random.default_rng(0)
        it = rng.choice(np.flatnonzero(ktr), size=min(args.kernel_n, ktr.sum()), replace=False)
        iv = rng.choice(np.flatnonzero(kva), size=min(args.kernel_n, kva.sum()), replace=False)
        Kq = quantum_kernel(ztr[it], ztr[it], K); Kqv = quantum_kernel(zva[iv], ztr[it], K)
        res["kernel - quantum"] = kernel_ridge_skill(Kq, Kqv, atr[it], ytr[it], ava[iv], yva[iv],
                                                     tstd, rmse_p)
        best_r = -9e9
        for gamma in (0.05, 0.1, 0.3, 1.0):
            best_r = max(best_r, kernel_ridge_skill(rbf_kernel(ztr[it], ztr[it], gamma),
                                                    rbf_kernel(zva[iv], ztr[it], gamma),
                                                    atr[it], ytr[it], ava[iv], yva[iv], tstd, rmse_p))
        res["kernel - classical RBF"] = best_r

    if "encoder" in arms:
        print("[encoder] TRAINED quantum circuit -> %d latents -> linear readout ..." % K)
        t0 = time.time()
        res["encoder - quantum (trained)"] = train_encoder_skill(
            "quantum", ztr[ktr], atr[ktr], ytr[ktr], zva[kva], ava[kva], yva[kva],
            tstd, rmse_p, K, args.epochs, 7)
        print("   quantum encoder trained in %.0f s" % (time.time() - t0))
        res["encoder - classical (trained)"] = max(
            train_encoder_skill("classical", ztr[ktr], atr[ktr], ytr[ktr], zva[kva], ava[kva],
                                yva[kva], tstd, rmse_p, K, args.epochs, s)
            for s in range(args.seeds))

    print("\n%-32s %s" % ("arm", "skill_vs_persistence"))
    print("-" * 56)
    for k_, v in res.items():
        print("%-32s %+0.4f" % (k_, v))

    print("\n=== verdict ===")
    for fam in ("reservoir", "kernel", "encoder"):
        pair = [(k_, v) for k_, v in res.items() if k_.startswith(fam)]
        if len(pair) != 2:
            continue
        q = [v for k_, v in pair if "quantum" in k_][0]
        c = [v for k_, v in pair if "quantum" not in k_][0]
        flag = "BEATS" if q - c > 0.01 else "does NOT beat"
        print("  %-10s quantum %+0.4f  vs  control %+0.4f   -> %s its control"
              % (fam, q, c, flag))


if __name__ == "__main__":
    main()
