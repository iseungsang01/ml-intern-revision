"""B.11 stage 1 -- the val-only screen for the discrete-code readout.

PREREGISTRATION_B11.md section 3 fixes what this runs and what counts as passing, and it
was committed before any number here was seen. TEST is never loaded.

Arms: `b3k8` (the adopted interpretable rung, control) and `b3vq{4,8,16}` (same structure,
same latent width, readout weights selected by a discrete code). Three init seeds each.

The four declared screens:

  S1  codebook survival -- T_i perplexity >= 2 and at least two codes above 1% usage
  S2  val non-inferiority -- mean val masked MSE within PRACTICAL_EPS (0.02) of b3k8
  S3  the codes track target MOVEMENT -- mutual information between the code and the
      binned |delta CES_TI| beats the 99th percentile of a 1000-shuffle null, with
      I(code; shot) reported alongside as the confound diagnostic
  S4  routing -- zeroing the 15 fast channels leaves V_rot bit-identical

CPU by default: a CUDA device was present but held by another job, and b3 is a 21k-parameter
model, so the screen costs less than contending would.

Run from the repo root:
    py ces_prediction/experiments/b11_vq/run_b11.py --smoke
    py ces_prediction/experiments/b11_vq/run_b11.py
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

# Confirmed data treatment, pinned before seq_data is imported (experiments/README.md
# non-negotiable 1). Never inherited, never popped.
PINNED = {
    "CES_DROP_STUCK_TARGETS": "1",
    "CES_TI_SPIKE_CUT_EV": "3000",
    "CES_PER_SHOT_NORM": "1",
    "CES_SEQ_DEVICE": "cpu",
    "CES_LR": "1e-3",
}
for _k, _v in PINNED.items():
    os.environ[_k] = _v

from seq_data import load_grid_files, fit_stats, build_blocks  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402

BATCH = 16
LR = 1e-3
WD = 1e-4
CLIP = 1.0
PRACTICAL_EPS = 0.02          # inherited from B.9 section 3.1
ARMS = ["b3k8", "b3vq4", "b3vq8", "b3vq16"]
INIT_SEEDS = [42, 1, 7]
N_SHUFFLE = 1000
MOVE_BINS = 3                 # terciles of |delta CES_TI|, matching 8an's stratification


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


def masked_loss(model, x, y, m, lengths, zero_ti):
    """train_seq.masked_pass's loss exactly, plus the arm's own auxiliary term."""
    pred = model(x, lengths)
    mse = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)
    pen = ((torch.relu(zero_ti - pred[..., 0]) * m[..., 0]).sum()
           / m[..., 0].sum().clamp(min=1.0))
    loss = mse + 0.1 * pen
    aux = getattr(model, "aux_loss", None)
    if isinstance(aux, torch.Tensor) and aux.requires_grad:
        loss = loss + aux
    return loss, pred


def run_epoch(model, blocks, device, zero_ti, rng, opt=None):
    model.train(opt is not None)
    order = list(range(len(blocks)))
    if opt is not None:
        rng.shuffle(order)
    se, nobs = 0.0, 0.0
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        if len(idx) < 2:
            continue
        x, y, m, lengths = batch_tensors(blocks, idx, device)
        with torch.set_grad_enabled(opt is not None):
            loss, pred = masked_loss(model, x, y, m, lengths, zero_ti)
            se += float((((pred - y) ** 2) * m).sum().detach())
            nobs += float(m.sum())
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
    return se / max(nobs, 1.0)


def mutual_information(a, b):
    """I(a; b) in nats from two integer label arrays of equal length."""
    ca, cb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.zeros((ca, cb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def screen_codes(model, blocks, device, rng):
    """S3: do the codes track how much the target moves, or just which shot it is?"""
    codes, moves, shots = [], [], []
    model.eval()
    with torch.no_grad():
        for bi in range(0, len(blocks), BATCH):
            idx = list(range(bi, min(bi + BATCH, len(blocks))))
            if len(idx) < 2:
                continue
            x, y, m, lengths = batch_tensors(blocks, idx, device)
            i_ti, _ = model.codes(x, lengths)
            for r, b in enumerate(idx):
                L = int(lengths[r])
                obs = m[r, :L, 0].cpu().numpy() > 0
                yv = y[r, :L, 0].cpu().numpy()
                cv = i_ti[r, :L].cpu().numpy()
                pair = obs[1:] & obs[:-1]
                if not pair.any():
                    continue
                d = np.abs(yv[1:][pair] - yv[:-1][pair])
                c = cv[1:][pair]
                codes.append(c)
                moves.append(d)
                shots.append(np.full(d.shape, b))
    if not codes:
        return None
    c = np.concatenate(codes).astype(int)
    d = np.concatenate(moves)
    sh = np.concatenate(shots).astype(int)

    edges = np.quantile(d, np.linspace(0, 1, MOVE_BINS + 1)[1:-1])
    mbin = np.digitize(d, edges)
    obs_mi = mutual_information(c, mbin)

    null = np.empty(N_SHUFFLE)
    perm = c.copy()
    for i in range(N_SHUFFLE):
        rng.shuffle(perm)
        null[i] = mutual_information(perm, mbin)
    return {
        "n_pairs": int(c.size),
        "mi_code_move": obs_mi,
        "null_p99": float(np.quantile(null, 0.99)),
        "null_mean": float(null.mean()),
        "passes_S3": bool(obs_mi > np.quantile(null, 0.99)),
        "mi_code_shot": mutual_information(c, sh),
        "code_share": np.bincount(c, minlength=int(c.max()) + 1).tolist(),
    }


def routing_ok(model, blocks, device):
    """S4: zeroing the fast channels must leave V_rot bit-identical."""
    model.eval()
    idx = list(range(min(BATCH, len(blocks))))
    x, _, _, lengths = batch_tensors(blocks, idx, device)
    with torch.no_grad():
        a = model(x, lengths)
        x2 = x.clone()
        x2[..., :model.n_fast] = 0.0
        b = model(x2, lengths)
    return bool(torch.equal(a[..., 1], b[..., 1])), bool(not torch.equal(a[..., 0], b[..., 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--split", default="data/.b1_manifest_s42")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    manifest = json.loads(
        (REPO_ROOT / args.split / "split_manifest.json").read_text(encoding="utf-8"))
    train_names, val_names = list(manifest["train_files"]), list(manifest["val_files"])
    arms, seeds, epochs = ARMS, INIT_SEEDS, args.epochs
    if args.smoke:
        train_names, val_names = train_names[:40], val_names[:12]
        arms, seeds, epochs = ["b3k8", "b3vq8"], [42], 3

    t0 = time.time()
    grid, dims = load_grid_files(REPO_ROOT / "data", True)
    stats = fit_stats(grid, dims, [n for n in train_names if n in grid])
    make = lambda names: [b for n in names if n in grid
                          for b in build_blocks(grid[n], dims, stats, per_shot_norm=True)]
    train_blocks, val_blocks = make(train_names), make(val_names)
    zero_ti = float((0.0 - stats["target"]["mean"][0]) / stats["target"]["std"][0])
    print("[b11] blocks train=%d val=%d (%.0fs load) device=cpu"
          % (len(train_blocks), len(val_blocks), time.time() - t0), flush=True)

    out = {"protocol": dict(PINNED, batch_blocks=BATCH, lr=LR, wd=WD, clip=CLIP,
                            epochs=epochs, patience=args.patience, split=args.split,
                            arms=arms, init_seeds=seeds, practical_eps=PRACTICAL_EPS,
                            n_shuffle=N_SHUFFLE, move_bins=MOVE_BINS,
                            smoke=bool(args.smoke)),
           "runs": []}

    for arm in arms:
        for sd in seeds:
            torch.manual_seed(sd)
            np.random.seed(sd)
            random.seed(sd)
            model = SEQ_MODELS[arm]().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                               patience=2, factor=0.5)
            rng = random.Random(sd)
            best, best_ep, best_state = float("inf"), -1, None
            te = time.time()
            for ep in range(epochs):
                run_epoch(model, train_blocks, device, zero_ti, rng, opt)
                v = run_epoch(model, val_blocks, device, zero_ti, rng)
                sched.step(v)
                if v < best:
                    best, best_ep = v, ep
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                if ep - best_ep >= args.patience:
                    break
            model.load_state_dict(best_state)

            rec = {"arm": arm, "init_seed": sd, "n_params": int(model.n_params),
                   "val_mse": best, "best_epoch": best_ep + 1, "epochs_run": ep + 1,
                   "seconds": round(time.time() - te)}
            vt_same, ti_moves = routing_ok(model, val_blocks, device)
            rec["S4_routing_vt_bit_identical"] = vt_same
            rec["ti_responds_to_fast"] = ti_moves
            if arm.startswith("b3vq"):
                u = model.vq_ti.usage()
                rec["codebook_ti"] = u
                rec["S1_survives"] = bool(u["perplexity"] >= 2.0 and u["live_codes"] >= 2)
                rec["S3"] = screen_codes(model, val_blocks, device,
                                         np.random.default_rng(sd))
            out["runs"].append(rec)
            print("[b11] %-7s seed %3d  val_mse=%.4f  ep=%d/%d  %ds%s"
                  % (arm, sd, best, best_ep + 1, ep + 1, rec["seconds"],
                     ("  perp=%.2f live=%d S3=%s"
                      % (rec["codebook_ti"]["perplexity"], rec["codebook_ti"]["live_codes"],
                         rec["S3"]["passes_S3"] if rec["S3"] else "n/a"))
                     if arm.startswith("b3vq") else ""), flush=True)

    # ---- pre-registered verdict ------------------------------------------------
    def mean_val(a):
        v = [r["val_mse"] for r in out["runs"] if r["arm"] == a]
        return float(np.mean(v)) if v else float("nan")

    ctrl = mean_val("b3k8")
    verdict = {"control_val_mse": ctrl, "arms": {}}
    for a in arms:
        if a == "b3k8":
            continue
        rs = [r for r in out["runs"] if r["arm"] == a]
        s1 = all(r.get("S1_survives") for r in rs)
        s2 = (mean_val(a) - ctrl) <= PRACTICAL_EPS
        s3 = all((r.get("S3") or {}).get("passes_S3") for r in rs)
        s4 = all(r["S4_routing_vt_bit_identical"] for r in rs)
        verdict["arms"][a] = {
            "val_mse": mean_val(a), "delta_vs_control": mean_val(a) - ctrl,
            "S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "S4": bool(s4),
            "promoted_to_stage2": bool(s1 and s2 and s3 and s4),
            "mean_mi_code_move": float(np.mean([(r.get("S3") or {}).get("mi_code_move", np.nan)
                                                for r in rs])),
            "mean_mi_code_shot": float(np.mean([(r.get("S3") or {}).get("mi_code_shot", np.nan)
                                                for r in rs])),
        }
    out["verdict"] = verdict
    any_pass = any(v["promoted_to_stage2"] for v in verdict["arms"].values())
    out["stage1_result"] = "PROMOTE" if any_pass else "NEGATIVE -- do not open TEST"

    dest = REPO_ROOT / "data" / (".b11_smoke.json" if args.smoke else ".b11_screen.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("")
    print("[b11] control b3k8 val_mse=%.4f" % ctrl)
    for a, v in verdict["arms"].items():
        print("[b11] %-7s val=%.4f (%+.4f)  S1=%s S2=%s S3=%s S4=%s  -> %s"
              % (a, v["val_mse"], v["delta_vs_control"], v["S1"], v["S2"], v["S3"], v["S4"],
                 "PROMOTE" if v["promoted_to_stage2"] else "no"), flush=True)
        print("        I(code;move)=%.4f  I(code;shot)=%.4f nats"
              % (v["mean_mi_code_move"], v["mean_mi_code_shot"]), flush=True)
    print("[b11] stage 1: %s" % out["stage1_result"])
    print("[b11] wrote %s" % dest)


if __name__ == "__main__":
    main()
