"""Train the seq-LSTM on full-grid sequences with loss-side per-target masking.

Reads the SAME split manifest as the control family (CES_SPLIT_DIR, read-only) so the
train/val/test shot files are identical to data/.vt_repro_*. Training keeps held values
by default (CES_DROP_STUCK_TARGETS=0), matching the control's training regime -- the
architecture/framing is the single controlled variable.

Env: CES_DATA_DIR, CES_SPLIT_DIR, CES_OUTPUT_DIR, CES_SEED, CES_INIT_SEED,
     CES_DROP_STUCK_TARGETS (default 0), CES_SEQ_EPOCHS (30), CES_SEQ_BATCH (16),
     CES_LR (1e-3), CES_SEQ_DEVICE (cuda if available), CES_SEQ_MAX_FILES (0 = all,
     smoke only). Output: weights/seq_lstm.pth + metrics.json (stats in the same
     "normalization.stats" schema evaluate._load_stats expects).
"""

import json
import os
import random
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))  # ces_prediction/

from seq_data import load_grid_files, fit_stats, build_blocks  # noqa: E402

# v1 = the §8d shared-encoder model; v2 = v1 + the iter009 V_rot routing (§8t).

# b3kN = the B.3 minimal interpretable model with an N-dim T_i latent (the ONE
# explored variable; V_rot latent fixed at 4). Named variants, not env config,
# so a checkpoint always reloads under the same architecture.
from seq_models import SEQ_MODELS  # noqa: E402  (single registry)


def chunk_blocks(blocks, ctx):
    """Split every block into non-overlapping `ctx`-step segments (B.9 axis A).

    Training a model "at reach r" means it never carries recurrent state across more
    than r steps, so the honest implementation is truncated BPTT with the state reset at
    every segment boundary — which is exactly what feeding non-overlapping length-r
    sequences does. Every row stays supervised; its position inside a segment decides how
    much context it gets (1..r), so the model learns to operate at every context length
    up to r, and eval at exactly r (`CES_SEQ_EVAL_CTX`, sliding) is in-distribution.

    No input channel is touched. `seq_data.build_blocks` already computed carry-forward /
    staleness over the WHOLE block, and the `W = 2` window control receives exactly that
    carried history, so this cuts the *recurrent* reach only — the same isolation the
    §8ac probe makes at eval time, now applied during training as §8ae requires.
    """
    if ctx < 1:
        raise SystemExit(f"CES_SEQ_TRAIN_CTX must be >= 1, got {ctx}")
    out = []
    for b in blocks:
        n = b["x"].shape[0]
        for s in range(0, n, ctx):
            out.append({k: v[s:s + ctx] for k, v in b.items()})
    return out


def batched(blocks, batch_size, device, shuffle, rng):
    order = list(range(len(blocks)))
    if shuffle:
        rng.shuffle(order)
        order.sort(key=lambda i: blocks[i]["x"].shape[0] // 256)  # coarse length bucketing
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(blocks[i]["x"].shape[0] for i in idx)
        B = len(idx)
        x = torch.zeros(B, L, blocks[0]["x"].shape[1])
        y = torch.zeros(B, L, 2)
        m = torch.zeros(B, L, 2)
        lengths = torch.zeros(B, dtype=torch.long)
        for j, i in enumerate(idx):
            n = blocks[i]["x"].shape[0]
            x[j, :n] = torch.from_numpy(blocks[i]["x"])
            y[j, :n] = torch.from_numpy(blocks[i]["y"])
            m[j, :n] = torch.from_numpy(blocks[i]["mask"])
            lengths[j] = n
        yield x.to(device), y.to(device), m.to(device), lengths


def masked_pass(model, blocks, batch_size, device, zero_ti, optimizer=None, clip=1.0, rng=None):
    training = optimizer is not None
    model.train(training)
    se_sum, n_obs, pen_sum, n_batches = 0.0, 0.0, 0.0, 0
    for x, y, m, lengths in batched(blocks, batch_size, device, training, rng or random.Random(0)):
        with torch.set_grad_enabled(training):
            # v3-family models attend over past fresh-observation steps; the loss
            # mask m IS the per-step observation indicator, used strictly causally
            # (shifted + lower-triangular inside the model).
            out = model(x, lengths, m) if getattr(model, "needs_obs", False) else model(x, lengths)
            se = ((out - y) ** 2) * m
            mse = se.sum() / m.sum().clamp(min=1.0)
            pen = (torch.relu(zero_ti - out[..., 0]) * m[..., 0]).sum() / m[..., 0].sum().clamp(min=1.0)
            loss = mse + 0.1 * pen
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        se_sum += float(se.detach().sum())
        n_obs += float(m.sum())
        pen_sum += float(pen.detach())
        n_batches += 1
    return se_sum / max(n_obs, 1.0), pen_sum / max(n_batches, 1)


def main():
    root = Path(__file__).resolve().parents[3]
    data_dir = Path(os.getenv("CES_DATA_DIR", root / "data"))
    split_dir = Path(os.environ["CES_SPLIT_DIR"])
    out_dir = Path(os.environ["CES_OUTPUT_DIR"])
    seed = int(os.getenv("CES_SEED", "42"))
    init_seed = int(os.getenv("CES_INIT_SEED", str(seed)))
    epochs = int(os.getenv("CES_SEQ_EPOCHS", "30"))
    # Budget-equalization arm (PREREGISTRATION_W2.md §4): train exactly `epochs`
    # epochs with no early stop and keep the FINAL weights (no val-based
    # checkpoint selection) -- the window pipeline's fixed-epoch regime.
    fixed_epochs = os.getenv("CES_SEQ_FIXED_EPOCHS", "0") == "1"
    batch_size = int(os.getenv("CES_SEQ_BATCH", "16"))
    lr = float(os.getenv("CES_LR", "1e-3"))
    drop_stuck = os.getenv("CES_DROP_STUCK_TARGETS", "0") == "1"
    max_files = int(os.getenv("CES_SEQ_MAX_FILES", "0"))
    variant = os.getenv("CES_SEQ_MODEL", "v1")
    if variant not in SEQ_MODELS:
        raise SystemExit(f"CES_SEQ_MODEL must be one of {sorted(SEQ_MODELS)}, got {variant!r}")
    per_shot = os.getenv("CES_PER_SHOT_NORM", "0") == "1"
    # B.9 axis A: reach the model is TRAINED at (0 = whole block, the published
    # regime). PREREGISTRATION_B9.md sec.2.1 -- eval must set CES_SEQ_EVAL_CTX to match.
    train_ctx = int(os.getenv("CES_SEQ_TRAIN_CTX", "0"))
    device = torch.device(os.getenv("CES_SEQ_DEVICE",
                                    "cuda" if torch.cuda.is_available() else "cpu"))

    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_names, val_names = list(manifest["train_files"]), list(manifest["val_files"])
    if max_files:  # smoke only
        train_names, val_names = train_names[:max_files], val_names[:max(2, max_files // 4)]

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)

    t0 = _time.time()
    grid, dims = load_grid_files(data_dir, drop_stuck)
    stats = fit_stats(grid, dims, [n for n in train_names if n in grid])
    make = lambda names: [b for n in names if n in grid
                          for b in build_blocks(grid[n], dims, stats, per_shot_norm=per_shot)]
    train_blocks, val_blocks = make(train_names), make(val_names)
    if train_ctx:
        # Both sides, so the early-stopping signal describes the regime being trained.
        train_blocks = chunk_blocks(train_blocks, train_ctx)
        val_blocks = chunk_blocks(val_blocks, train_ctx)
    n_obs = sum(int(b["mask"].sum()) for b in train_blocks)
    print(f"[seq] blocks train={len(train_blocks)} val={len(val_blocks)} "
          f"train-labels={n_obs:,} drop_stuck={int(drop_stuck)} per_shot={int(per_shot)} "
          f"model={variant} train_ctx={train_ctx or 'full'} device={device.type} "
          f"({_time.time() - t0:.0f}s load)")

    model = SEQ_MODELS[variant]().to(device)
    print(f"[seq] params={model.n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=2, factor=0.5)
    zero_ti = float((0.0 - stats["target"]["mean"][0]) / stats["target"]["std"][0])

    rng = random.Random(seed)
    best_val, best_state, best_epoch, patience = float("inf"), None, -1, 6
    for epoch in range(epochs):
        tr_mse, tr_pen = masked_pass(model, train_blocks, batch_size, device, zero_ti,
                                     optimizer=optimizer, rng=rng)
        val_mse, _ = masked_pass(model, val_blocks, batch_size, device, zero_ti)
        scheduler.step(val_mse)
        star = ""
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *"
        print(f"[seq] epoch {epoch + 1:02d}/{epochs} train_mse={tr_mse:.4f} "
              f"pen={tr_pen:.4f} val_mse={val_mse:.4f}{star}", flush=True)
        if not fixed_epochs and epoch - best_epoch >= patience:
            print(f"[seq] early stop (no val improvement for {patience} epochs)")
            break

    if not fixed_epochs:
        model.load_state_dict(best_state)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights" / "seq_lstm.pth")
    metrics = {
        "experiment": "seq_lstm_full_grid_masked_loss",
        "seq_model": variant,
        "train_ctx": train_ctx or None,
        "per_shot_input_norm": per_shot,
        # B.6 provenance: the arm's input treatment must reload identically at eval.
        "extra_vt_channels": int(dims.get("extra_vt", 0) or 0),
        "zero_mc": os.getenv("CES_SEQ_ZERO_MC", "0") == "1",
        "normalization": {"stats": {g: {k: v.tolist() for k, v in gs.items()}
                                    for g, gs in stats.items()}},
        "feature_dims": dims,
        "n_params": model.n_params,
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_val_masked_mse": best_val,
        "train_blocks": len(train_blocks),
        "train_labels": n_obs,
        "drop_stuck_targets": drop_stuck,
        "fixed_epochs": fixed_epochs,
        "seed": seed, "init_seed": init_seed,
        "loss": "per-target masked MSE + 0.1*relu(zero_ti - pred_TI) (train.py parity)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[seq] saved weights + metrics to {out_dir} (best val {best_val:.4f} @ {best_epoch + 1})",
          flush=True)


if __name__ == "__main__":
    main()
    # PyTorch CUDA teardown on Windows can fastfail (0xC0000409) at interpreter
    # exit AFTER all work is done and saved; skip teardown so rc reflects the run.
    os._exit(0)
