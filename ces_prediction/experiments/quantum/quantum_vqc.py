"""Variational quantum circuit (VQC) vs matched-parameter classical MLP on the CES task.

This is a **controlled quantum-vs-classical benchmark**, not a proposed replacement for
``MultimodalCESPredictor``. It exists to answer one question honestly:

    At an identical parameter budget and on identical inputs, does a variational quantum
    circuit beat a classical neural network on KSTAR CES nowcasting?

Design (every choice here is about making the comparison *fair*, not about winning):

- **Same inputs.** Both models see the SAME PCA-reduced feature vector. The quantum circuit
  cannot encode 92 raw dimensions on 8 qubits, so we reduce to ``n_qubits`` dimensions with a
  PCA fit on TRAIN samples only. The classical control gets the byte-identical reduced input,
  so any performance gap is attributable to the model, not to the dimensionality reduction.
- **Same parameter budget.** The MLP hidden width is chosen so its parameter count matches the
  VQC's as closely as possible. Comparing a ~100-parameter VQC against the 815k-parameter
  production model would be meaningless.
- **Same evaluation.** ``skill_vs_persistence = 1 - MSE_model / MSE_persistence`` in physical
  CES units, on the SAME clean non-augmented validation samples ``evaluate.py`` uses
  (same seed, same cap, same split manifest), with the same persistence baseline function.

Training runs on a **local state-vector simulator** (``default.qubit``, backprop). Training on
real QPU hardware is not merely expensive but arithmetically impossible for this project:
parameter-shift needs 2 circuit evaluations per parameter per sample, so a single gradient step
at batch 32 with ~100 parameters costs ~614,400 shots ~= 7 hours of QPU time -- more than an
entire research credit allocation, for one step. Hardware is therefore used for **inference
only**, via ``ionq_infer.py``.

Run (from repo root)::

    py ces_prediction/quantum_vqc.py

Environment variables::

    QVQC_N_QUBITS     number of qubits / PCA components   (default 8)
    QVQC_N_LAYERS     variational layers w/ data re-upload (default 4)
    QVQC_MAX_TRAIN    training samples                     (default 20000)
    QVQC_EPOCHS       training epochs                      (default 30)
    QVQC_TARGET       CES_TI or CES_VT                     (default CES_TI)
    QVQC_LR           learning rate                        (default 0.02)
    QVQC_BATCH        batch size                           (default 256)

Writes ``<output_dir>/quantum_vqc_result.json`` and ``<output_dir>/quantum_vqc_weights.pt``
(the trained circuit parameters + PCA basis, consumed by ``ionq_infer.py``).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import pennylane as qml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))  # ces_prediction/

from dataset import KSTAR_CES_Dataset, select_seeded_random_indices  # noqa: E402
from evaluate import _load_stats, _persistence_from_history  # noqa: E402

TARGET_NAMES = ("CES_TI", "CES_VT")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def load_split_tensors(data_dir, window_size, stats, manifest, max_train, max_val, seed,
                       batch_size=512):
    """Flatten every sample to a single feature vector, keeping train/val file-level split.

    Returns dicts with keys: x (N, D), target (N, 2), mask (N, 2), persist (N, 2),
    persist_obs (N, 2) -- all normalized except where noted.
    """
    dataset = KSTAR_CES_Dataset(
        data_dir=data_dir,
        window_size=window_size,
        temporal_subset_augmentation=False,
        drop_stuck_targets=os.getenv("CES_DROP_STUCK_TARGETS", "1") == "1",
    )
    dataset.set_normalization_stats(stats)

    file_names = [Path(p).name for p in dataset.valid_files]
    train_files = set(manifest["train_files"])
    val_files = set(manifest["val_files"])
    train_ids = {i for i, n in enumerate(file_names) if n in train_files}
    val_ids = {i for i, n in enumerate(file_names) if n in val_files}

    all_train = [i for i in range(len(dataset)) if int(dataset.sample_file_indices[i]) in train_ids]
    all_val = [i for i in range(len(dataset)) if int(dataset.sample_file_indices[i]) in val_ids]
    if not all_train or not all_val:
        raise ValueError("Empty train or val index set -- check split_manifest.json vs data dir.")

    # seed+202 and the val cap MATCH evaluate.py exactly, so the val samples are identical.
    train_idx = select_seeded_random_indices(all_train, max_train, seed + 101)
    val_idx = select_seeded_random_indices(all_val, max_val, seed + 202)

    def collect(indices):
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
        xs, ys, ms, ps, pos = [], [], [], [], []
        for batch in loader:
            feats = torch.cat(
                (batch["bes"], batch["ecei"], batch["mc"],
                 batch["time_features"], batch["ces_history"]),
                dim=-1,
            )  # (B, window, C_total)
            xs.append(feats.flatten(1))
            ys.append(batch["target"])
            ms.append(batch["target_mask"])
            persistence, has_obs = _persistence_from_history(batch["ces_history"])
            ps.append(persistence)
            pos.append(has_obs)
        return {
            "x": torch.cat(xs), "target": torch.cat(ys), "mask": torch.cat(ms),
            "persist": torch.cat(ps), "persist_obs": torch.cat(pos),
        }

    return dataset, collect(train_idx), collect(val_idx), len(val_ids), len(val_idx)


def fit_pca(x, n_components):
    """PCA via SVD, fit on TRAIN only. Returns (mean, components, scale)."""
    mean = x.mean(dim=0)
    xc = x - mean
    # economy SVD; components are rows of Vh
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:n_components]  # (k, D)
    z = xc @ comps.T
    scale = z.std(dim=0).clamp_min(1e-6)
    return mean, comps, scale


def apply_pca(x, mean, comps, scale):
    """Project and squash into a bounded angle range suitable for AngleEmbedding.

    tanh keeps every encoded angle inside (-pi/2, pi/2) so outliers cannot wrap around the
    Bloch sphere and alias onto a different rotation -- a real failure mode of naive angle
    encoding on z-scored data with heavy tails (which CES diagnostics have).
    """
    z = ((x - mean) @ comps.T) / scale
    return (torch.pi / 2) * torch.tanh(z)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class VQCRegressor(nn.Module):
    """Data re-uploading VQC: [AngleEmbedding -> StronglyEntanglingLayer] x n_layers -> <Z_0>.

    Data re-uploading (re-encoding the input before every variational layer) is what gives the
    circuit nonlinearity in the input: a single encoding followed by unitaries would be a
    fixed linear map into Hilbert space, and the expectation value could only realize a
    truncated Fourier series of very low degree. Re-uploading raises that degree per layer.
    """

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # backprop on default.qubit stores every intermediate statevector, so memory grows as
        # batch x 2^n_qubits x circuit depth -- it segfaults well before 16 qubits. lightning.qubit
        # with adjoint differentiation is O(1) in depth and broadcasts over the batch, which is
        # what makes the larger circuits trainable at all.
        dev = qml.device(os.getenv("QVQC_DEVICE", "default.qubit"), wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=os.getenv("QVQC_DIFF", "backprop"))
        def circuit(inputs, weights):
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer: layer + 1], wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(0.1 * torch.randn(*shape))
        # <Z> is bounded in [-1, 1]; the target is z-scored, so an affine readout is required
        # for the model to reach targets outside that range at all.
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        exp = self.circuit(x, self.weights)
        if not torch.is_tensor(exp):
            exp = torch.as_tensor(exp)
        exp = exp.to(x.dtype)
        return self.out_scale * exp + self.out_bias

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class MatchedMLP(nn.Module):
    """Classical control with (near-)identical parameter count and identical input."""

    def __init__(self, n_inputs, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def hidden_width_for_budget(n_inputs, budget):
    """Pick MLP hidden width whose param count is closest to `budget`.

    params(h) = (n_inputs*h + h) + (h + 1) = h*(n_inputs + 2) + 1
    """
    best, best_err = 1, None
    for h in range(1, 512):
        p = h * (n_inputs + 2) + 1
        err = abs(p - budget)
        if best_err is None or err < best_err:
            best, best_err = h, err
    return best


# --------------------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------------------

def train_model(model, xtr, ytr, mtr, epochs, lr, batch_size, label, log_every=5, ckpt=None):
    """Train with per-epoch checkpointing so an interrupted run resumes instead of restarting.

    A 16-qubit circuit takes ~20 minutes per epoch on CPU, so a kill at epoch 24 of 25 used to
    throw away eight hours. `ckpt` is a path; if it exists its epoch/model/optimizer state is
    restored and training continues from there.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xtr.shape[0]
    t0 = time.time()
    start_epoch = 0
    if ckpt is not None and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        print(f"  [{label}] resuming from epoch {start_epoch}/{epochs} ({ckpt})")
    for epoch in range(start_epoch, epochs):
        perm = torch.randperm(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            xb, yb, mb = xtr[idx], ytr[idx], mtr[idx]
            if mb.sum() == 0:
                continue
            pred = model(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(mb.sum())
            seen += int(mb.sum())
        if ckpt is not None:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict(), "loss": total / max(seen, 1)}, ckpt)
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  [{label}] epoch {epoch + 1:3d}/{epochs}  masked MSE {total / max(seen, 1):.5f}"
                  f"  ({time.time() - t0:.1f}s)")
    return model


@torch.no_grad()
def predict_batched(model, x, batch_size=512):
    return torch.cat([model(x[i: i + batch_size]) for i in range(0, x.shape[0], batch_size)])


def score(pred_norm, val, t_idx, target_mean, target_std):
    """Physical-unit RMSE and skill vs persistence, matching evaluate.py's definitions."""
    mask = val["mask"][:, t_idx] > 0.5
    keep = mask & val["persist_obs"][:, t_idx]
    n = int(keep.sum())
    if n == 0:
        return {"n": 0}
    mean_t, std_t = float(target_mean[t_idx]), float(target_std[t_idx])
    y = val["target"][keep, t_idx] * std_t + mean_t
    p = pred_norm[keep] * std_t + mean_t
    per = val["persist"][keep, t_idx] * std_t + mean_t
    mse_model = float(((p - y) ** 2).mean())
    mse_persist = float(((per - y) ** 2).mean())
    var = float(((y - y.mean()) ** 2).mean())
    return {
        "n": n,
        "rmse_model": mse_model ** 0.5,
        "rmse_persistence": mse_persist ** 0.5,
        "rmse_mean_baseline": var ** 0.5,
        "skill_vs_persistence": 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan"),
        "r2_vs_mean": 1.0 - mse_model / var if var > 0 else float("nan"),
    }


def main():
    # parents[2] is the repo root. This file lived at ces_prediction/quantum_vqc.py until the
    # 2026-08-09 consolidation moved it two levels down; the index was never updated, so the
    # defaults pointed at ces_prediction/data and every run had to pass paths explicitly.
    root_dir = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("CES_DATA_DIR", root_dir / "data"))
    output_dir = Path(os.getenv("CES_OUTPUT_DIR", Path(__file__).resolve().parent))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", root_dir / "data" / "splits"))
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))

    n_qubits = int(os.getenv("QVQC_N_QUBITS", "8"))
    n_layers = int(os.getenv("QVQC_N_LAYERS", "4"))
    max_train = int(os.getenv("QVQC_MAX_TRAIN", "20000"))
    epochs = int(os.getenv("QVQC_EPOCHS", "30"))
    # Each model gets its OWN best learning rate from an identical sweep, so neither side is
    # handicapped by a shared setting that happens to suit the other.
    lr_vqc = float(os.getenv("QVQC_LR_VQC", os.getenv("QVQC_LR", "0.2")))
    lr_mlp = float(os.getenv("QVQC_LR_MLP", os.getenv("QVQC_LR", "0.01")))
    batch_size = int(os.getenv("QVQC_BATCH", "256"))
    target_name = os.getenv("QVQC_TARGET", "CES_TI")
    if target_name not in TARGET_NAMES:
        raise ValueError(f"QVQC_TARGET must be one of {TARGET_NAMES}, got {target_name!r}")
    t_idx = TARGET_NAMES.index(target_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Read the normalization stats from the pipeline's own metrics.json; write results wherever
    # CES_OUTPUT_DIR points. Conflating the two meant a throwaway output dir also had to contain
    # a metrics.json, so variants could not be written anywhere but next to the source.
    metrics_path = Path(os.getenv("CES_METRICS", root_dir / "ces_prediction" / "metrics.json"))
    manifest_path = split_dir / "split_manifest.json"
    for p in (metrics_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}. Run training first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Target={target_name}  qubits={n_qubits}  layers={n_layers}  window={window_size}")
    print("Building clean (non-augmented) train/val tensors...")
    _, train, val, n_val_shots, n_val_samples = load_split_tensors(
        data_dir, window_size, stats, manifest, max_train, max_val, seed
    )
    print(f"  train {tuple(train['x'].shape)}   val {tuple(val['x'].shape)}"
          f"   ({n_val_shots} val shots)")

    # --- dimensionality reduction (TRAIN-ONLY fit, mirroring the project's normalization rule)
    mean, comps, scale = fit_pca(train["x"], n_qubits)
    explained = None
    with torch.no_grad():
        xc = train["x"] - mean
        total_var = float((xc ** 2).sum())
        kept_var = float(((xc @ comps.T) ** 2).sum())
        explained = kept_var / total_var if total_var > 0 else float("nan")
    print(f"  PCA {train['x'].shape[1]} -> {n_qubits} dims, explained variance {explained:.4f}")

    ztr = apply_pca(train["x"], mean, comps, scale)
    zval = apply_pca(val["x"], mean, comps, scale)

    ytr, mtr = train["target"][:, t_idx], (train["mask"][:, t_idx] > 0.5).float()
    target_mean = torch.as_tensor(stats["target"]["mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target"]["std"], dtype=torch.float32)

    # --- quantum
    vqc = VQCRegressor(n_qubits, n_layers)
    n_q_params = vqc.n_params()
    hidden = hidden_width_for_budget(n_qubits, n_q_params)
    mlp = MatchedMLP(n_qubits, hidden)
    print(f"  VQC params {n_q_params}   matched MLP params {mlp.n_params()} (hidden={hidden})")

    print(f"Training VQC (local state-vector simulator, backprop, lr={lr_vqc})...")
    t_vqc = time.time()
    log_every = int(os.getenv("QVQC_LOG_EVERY", "5"))
    train_model(vqc, ztr, ytr, mtr, epochs, lr_vqc, batch_size, "VQC",
                log_every=log_every, ckpt=output_dir / "vqc_train.ckpt")
    t_vqc = time.time() - t_vqc
    print(f"Training matched classical MLP (lr={lr_mlp})...")
    t_mlp = time.time()
    train_model(mlp, ztr, ytr, mtr, epochs, lr_mlp, batch_size, "MLP",
                log_every=log_every, ckpt=output_dir / "mlp_train.ckpt")
    t_mlp = time.time() - t_mlp

    res_vqc = score(predict_batched(vqc, zval), val, t_idx, target_mean, target_std)
    res_mlp = score(predict_batched(mlp, zval), val, t_idx, target_mean, target_std)

    report = {
        "target": target_name,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "window_size": window_size,
        "pca_explained_variance": explained,
        "train_samples": int(ztr.shape[0]),
        "val_samples": int(zval.shape[0]),
        "val_shots": n_val_shots,
        "epochs": epochs,
        "lr_vqc": lr_vqc,
        "lr_mlp": lr_mlp,
        "train_seconds_vqc": t_vqc,
        "train_seconds_mlp": t_mlp,
        "simulator_slowdown_factor": t_vqc / t_mlp if t_mlp > 0 else float("nan"),
        "vqc_params": n_q_params,
        "mlp_params": mlp.n_params(),
        "mlp_hidden": hidden,
        "units": "raw CES units as stored in CSV",
        "vqc": res_vqc,
        "matched_mlp": res_mlp,
    }

    hdr = f"{'model':<14} {'n':>7} {'RMSE':>11} {'RMSE_persist':>13} {'skill_vs_persist':>17}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, r in (("VQC (quantum)", res_vqc), (f"MLP (classical)", res_mlp)):
        print(f"{label:<14} {r['n']:>7} {r['rmse_model']:>11.4f} "
              f"{r['rmse_persistence']:>13.4f} {r['skill_vs_persistence']:>17.4f}")

    out_json = output_dir / "quantum_vqc_result.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "weights": vqc.weights.detach(),
            "out_scale": vqc.out_scale.detach(),
            "out_bias": vqc.out_bias.detach(),
            "pca_mean": mean, "pca_comps": comps, "pca_scale": scale,
            "n_qubits": n_qubits, "n_layers": n_layers,
            "target": target_name, "window_size": window_size,
        },
        output_dir / "quantum_vqc_weights.pt",
    )
    print(f"\nSaved {out_json}")
    print(f"Saved {output_dir / 'quantum_vqc_weights.pt'}  (for ionq_infer.py)")
    return report


if __name__ == "__main__":
    main()
