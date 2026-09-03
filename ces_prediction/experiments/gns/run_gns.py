"""Measure the gradient noise scale of the backbone, so "why batch 16" stops being
an inherited setting and becomes a measured one.

Why this exists
---------------
`main_ko.tex` S4.7 now states plainly that the optimizer, batch size and schedule were
inherited from the windowed pipeline rather than derived, and -- following this
project's own rule that a gap must name the measurement that closes it -- it names two.
This runner is the first of them.

McCandlish et al. (2018) define the *simple noise scale*

    B_simple = tr(Sigma) / |G|^2

where `G` is the true gradient and `Sigma` the per-example gradient covariance.  It is
the batch size at which the gradient's signal and its noise are comparable: below it,
doubling the batch roughly halves the number of steps needed; far above it, extra
examples buy almost nothing.

**The two-batch-size estimator does not work here, and the first attempt is kept on the
record as a negative.**  Estimating `|G|^2` as the difference of two sampled gradient
norms, `(B_big |g_big|^2 - B_small |g_small|^2) / (B_big - B_small)`, subtracts two noisy
numbers that become nearly equal once the model has converged and `|G|^2` is small.  With
40 small and 10 large draws per point it returned a **negative** `|G|^2` at two of thirteen
points and a `B_simple` ranging over 0.8-110 blocks -- unusable.  See THESIS_RESULTS 8as.

This version computes the quantity **exactly on the training set instead of sampling it**.
The sampling unit is a block (one contiguous segment), which is what the training loop
draws, so with `g_i` the gradient of block `i`'s own masked loss:

    G          = mean_i g_i                       (accumulated as a running sum)
    tr(Sigma)  = mean_i |g_i|^2 - |G|^2           (Cauchy-Schwarz keeps this >= 0)
    B_simple   = tr(Sigma) / |G|^2

Only a running gradient sum (one parameter-sized vector) and a running scalar are held, so
the cost is one batch-1 backward pass per block per measurement point and the result is
the empirical noise scale of that checkpoint rather than an estimate of it.

Scope and protocol
------------------
* CPU only.  A CUDA device was present when this was written but already held by
  another job, and nothing here is a latency claim, so the device changes only how
  long the measurement takes and not what it measures.
* Dropout is disabled for the measurement (`model.eval()`).  `B_simple` is meant to
  describe the noise from *sampling examples*; leaving dropout on would fold a second,
  unrelated noise source into `tr(Sigma)`.  The training loop's own gradient carries
  both, so the critical batch of the actual loop is at least the number reported here.
* Data treatment is **pinned explicitly** here, never inherited (experiments/README.md
  non-negotiable 1): held-free, spike cut at 3 keV, per-shot input standardization.
* Uses the **frozen B.1 split manifest** for seed 42 and reads only its `train_files`
  and `val_files`.  TEST is never loaded, never scored, and no model is promoted, so
  this needs no pre-registration entry -- it measures the optimizer, not the claim.
* The backbone, the loss and the optimizer are exactly the confirmed recipe
  (`seq_v2`, AdamW 1e-3 / wd 1e-4, masked MSE + 0.1 ReLU penalty, clip 1.0).

Batch size is reported in two units because the loss normalizes by observed labels, not
by blocks: a "batch of 16" is 16 segments, which carry a few thousand supervised rows.
Both are printed; quote whichever unit the sentence is about.

Run from the repo root:
    py ces_prediction/experiments/gns/run_gns.py            # full, ~10 epochs
    py ces_prediction/experiments/gns/run_gns.py --smoke    # 2 epochs, 40 files
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
SEQ_DIR = REPO_ROOT / "ces_prediction" / "experiments" / "seq"
sys.path.insert(0, str(SEQ_DIR))
sys.path.insert(1, str(REPO_ROOT / "ces_prediction"))

# The confirmed data treatment, pinned before seq_data is imported so nothing can
# fall back to train.py's defaults (experiments/README.md non-negotiable 1).
PINNED = {
    "CES_DROP_STUCK_TARGETS": "1",
    "CES_TI_SPIKE_CUT_EV": "3000",
    "CES_PER_SHOT_NORM": "1",
    "CES_SEQ_MODEL": "v2",
    "CES_SEQ_DEVICE": "cpu",
    "CES_LR": "1e-3",
}
for _k, _v in PINNED.items():
    os.environ[_k] = _v

from seq_data import load_grid_files, fit_stats, build_blocks  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402

BATCH = 16          # the confirmed recipe's batch, in blocks
MEASURE_AT = {1, 2, 3, 6, 9}   # plus init and the final epoch


def batch_tensors(blocks, idx, device):
    lens = [blocks[i]["x"].shape[0] for i in idx]
    n, t = len(idx), max(lens)
    c = blocks[idx[0]]["x"].shape[1]
    x = torch.zeros(n, t, c)
    y = torch.zeros(n, t, 2)
    m = torch.zeros(n, t, 2)
    lengths = torch.zeros(n, dtype=torch.long)
    for r, i in enumerate(idx):
        b = blocks[i]
        L = b["x"].shape[0]
        x[r, :L] = torch.as_tensor(b["x"], dtype=torch.float32)
        y[r, :L] = torch.as_tensor(b["y"], dtype=torch.float32)
        m[r, :L] = torch.as_tensor(b["mask"], dtype=torch.float32)
        lengths[r] = L
    return x.to(device), y.to(device), m.to(device), lengths


def loss_on(model, blocks, idx, device, zero_ti):
    x, y, m, lengths = batch_tensors(blocks, idx, device)
    pred = model(x, lengths)
    # Exactly train_seq.masked_pass: the penalty is masked too, so the gradient we
    # measure is the gradient the confirmed recipe actually takes.
    mse = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)
    pen = ((torch.relu(zero_ti - pred[..., 0]) * m[..., 0]).sum()
           / m[..., 0].sum().clamp(min=1.0))
    return mse + 0.1 * pen, float(m.sum())


def grad_sq_norm(model, blocks, idx, device, zero_ti):
    model.zero_grad(set_to_none=True)
    loss, n_lab = loss_on(model, blocks, idx, device, zero_ti)
    loss.backward()
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return s, n_lab


def measure_noise_scale(model, blocks, device, zero_ti, _rng=None):
    """Exact empirical noise scale over the training blocks at this checkpoint.

    One batch-1 backward per block; only a running gradient sum and a running scalar
    are kept, so memory is one parameter vector regardless of how many blocks there are.
    """
    n = len(blocks)
    acc = None
    sq_sum = 0.0
    lab = 0.0
    for i in range(n):
        model.zero_grad(set_to_none=True)
        loss, n_lab = loss_on(model, blocks, [i], device, zero_ti)
        loss.backward()
        flat = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                          if p.grad is not None])
        sq_sum += float(flat.pow(2).sum())
        acc = flat.clone() if acc is None else acc.add_(flat)
        lab += n_lab
    model.zero_grad(set_to_none=True)

    g_mean = acc / n
    g2 = float(g_mean.pow(2).sum())
    mean_sq = sq_sum / n
    trs = max(mean_sq - g2, 0.0)
    return {
        "n_blocks": n,
        "mean_grad_sq": mean_sq,
        "G2": g2,
        "trSigma": trs,
        "B_simple_blocks": (trs / g2) if g2 > 0 else float("nan"),
        "labels_per_block": lab / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--split", default="data/.b1_manifest_s42")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    manifest = json.loads(
        (REPO_ROOT / args.split / "split_manifest.json").read_text(encoding="utf-8"))
    train_names, val_names = list(manifest["train_files"]), list(manifest["val_files"])
    if args.smoke:
        train_names, val_names = train_names[:40], val_names[:10]

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    t0 = time.time()
    grid, dims = load_grid_files(REPO_ROOT / "data", True)
    stats = fit_stats(grid, dims, [n for n in train_names if n in grid])
    make = lambda names: [b for n in names if n in grid
                          for b in build_blocks(grid[n], dims, stats, per_shot_norm=True)]
    train_blocks, val_blocks = make(train_names), make(val_names)
    zero_ti = float((0.0 - stats["target"]["mean"][0]) / stats["target"]["std"][0])
    print("[gns] blocks train=%d val=%d (%.0fs load) device=%s"
          % (len(train_blocks), len(val_blocks), time.time() - t0, device.type),
          flush=True)

    model = SEQ_MODELS["v2"]().to(device)
    print("[gns] params=%d  batch=%d blocks  estimator=exact-per-block"
          % (model.n_params, BATCH), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(42)
    meas_rng = random.Random(1234)

    epochs = 2 if args.smoke else args.epochs
    out = {
        "protocol": dict(PINNED, batch_blocks=BATCH,
                         estimator="exact-per-block", split=args.split,
                         epochs=epochs, smoke=bool(args.smoke)),
        "n_params": int(model.n_params),
        "n_train_blocks": len(train_blocks),
        "points": [],
    }

    def record(tag, epoch):
        model.eval()
        m = measure_noise_scale(model, train_blocks, device, zero_ti, meas_rng)
        model.train()
        m["tag"], m["epoch"] = tag, epoch
        out["points"].append(m)
        print("[gns] %-9s |G|^2=%.4g  mean|g|^2=%.4g  trSigma=%.4g  B_simple=%.2f blocks"
              % (tag, m["G2"], m["mean_grad_sq"], m["trSigma"],
                 m["B_simple_blocks"]), flush=True)

    record("init", 0)
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_blocks)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        te = time.time()
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_on(model, train_blocks, idx, device, zero_ti)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        with torch.no_grad():
            v, vn = 0.0, 0
            for s in range(0, len(val_blocks), BATCH):
                idx = list(range(s, min(s + BATCH, len(val_blocks))))
                if len(idx) < 2:
                    continue
                l, _ = loss_on(model, val_blocks, idx, device, zero_ti)
                v += float(l)
                vn += 1
        print("[gns] epoch %02d/%d train=%.4f val=%.4f (%.0fs)"
              % (ep + 1, epochs, tot / max(nb, 1), v / max(vn, 1), time.time() - te),
              flush=True)
        if (ep + 1) in MEASURE_AT or (ep + 1) == epochs:
            record("epoch%02d" % (ep + 1), ep + 1)

    dest = REPO_ROOT / "data" / (".gns_smoke.json" if args.smoke else ".gns.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[gns] wrote %s" % dest, flush=True)

    bs = [p["B_simple_blocks"] for p in out["points"] if np.isfinite(p["B_simple_blocks"])]
    if bs:
        print("[gns] B_simple over training: min %.1f  max %.1f  final %.1f blocks"
              % (min(bs), max(bs), bs[-1]), flush=True)


if __name__ == "__main__":
    main()
